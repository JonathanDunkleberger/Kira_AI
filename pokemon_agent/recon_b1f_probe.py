"""recon_b1f_probe.py — READ-ONLY diagnosis (night shift 6): why did BFS read no_route
from the hideout B1F elevator lobby (map (1,42), landed @ (24,26)) toward the Game Corner
exit door (12,2) for ~20 straight ticks (185 travel wedges), then suddenly find a 44-step
path from the SAME tile? Candidates: spin tiles walling the lobby (Grid.walkable/edge law),
stale grid content post-warp, or something else entirely. Method: reproduce the exact
elevator descent the re-grade ran, then dump the floor as ASCII + BFS verdicts + a
staleness re-read. Zero writes; the banked bundle is load-only.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
import firered_ram as ram                                  # noqa: E402
import pokemon_state as st                                 # noqa: E402
import travel as tv                                        # noqa: E402
from battle_agent import BattleAgent                       # noqa: E402
from campaign import Campaign                              # noqa: E402
import elevator_nav                                        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_SCOPE")

b = Bridge(ROM)
with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

def runner():
    return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                       log=lambda m: None, choose=lambda *a, **k: None).run(max_seconds=90)

camp = Campaign(b, battle_runner=runner, on_event=lambda s, **k: None,
                beat=lambda *a, **k: None, render=lambda: None)
camp._save_campaign = lambda *a, **k: True
camp._continuity_save = lambda *a, **k: None

print(f"spawn: map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

# ── descend: B4F door (20,23) -> car (1,46) -> panel rows until we land on (1,42) ──
camp.enter_warp(pick=(20, 23))
print(f"after door: map={tv.map_id(b)} coords={tv.coords(b)} is_car={elevator_nav.is_car(b)}",
      flush=True)
seen = {(1, 45)}
for attempt in range(12):
    if tuple(tv.map_id(b)) == (1, 42):
        break
    if not elevator_nav.is_car(b):
        # walked out somewhere that isn't B1F — go back in through the nearest car door
        print(f"  not in car (map={tv.map_id(b)}) — re-boarding", flush=True)
        ws = [tuple(w[0]) for w in tv.read_warps(b) if tuple(w[1]) == (1, 46)] \
             or [tuple(w[0]) for w in tv.read_warps(b)]
        camp.enter_warp(pick=ws[0])
        continue
    row = attempt % 5
    ok = elevator_nav.ride(b, camp, row, avoid=seen)
    print(f"  ride row={row} -> {ok}; map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
    if ok:
        seen.add(tuple(tv.map_id(b)))

m = tuple(tv.map_id(b))
print(f"landed: map={m} coords={tv.coords(b)}", flush=True)
if m != (1, 42):
    print("!! never reached B1F (1,42) — dumping where we are anyway", flush=True)

# ── the dump ─────────────────────────────────────────────────────────────────────
OFF = tv.MAP_OFFSET

def behaviors(g):
    """{(x,y): behavior_byte} for the whole playable rect."""
    ml = b.rd32(tv.GMAPHEADER)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
    mp = b.rd32(tv.BACKUP_LAYOUT + 8)
    out = {}
    for by in range(g.h):
        for bx in range(g.w):
            e = b.rd16(mp + (by * g.w + bx) * 2)
            mid = e & 0x3FF
            base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1], mid - tv.NUM_PRIMARY)
            out[(bx - OFF, by - OFF)] = b.rd32(base + idx * 4) & 0xFF
    return out

def dump(tag):
    g = tv.Grid(b)
    cur = tuple(tv.coords(b))
    beh = behaviors(g)
    warps = {tuple(w[0]): tuple(w[1]) for w in tv.read_warps(b)}
    npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
    print(f"\n== {tag}: map={tv.map_id(b)} cur={cur} grid {g.w}x{g.h} "
          f"playable x[{g.sx_lo}..{g.sx_hi}] y[{g.sy_lo}..{g.sy_hi}]", flush=True)
    spin = {t for t, bh in beh.items() if 0x54 <= bh <= 0x58}
    print(f"spin/stop tiles: {len(spin)}", flush=True)
    for y in range(g.sy_lo, g.sy_hi + 1):
        row = []
        for x in range(g.sx_lo, g.sx_hi + 1):
            t = (x, y)
            if t == cur:
                c = "@"
            elif t in warps:
                c = "W"
            elif t in npcs:
                c = "N"
            elif t in spin:
                c = "S"
            elif g.walkable(x, y):
                c = "."
            else:
                c = "#"
            row.append(c)
        print(f"y={y:3d} " + "".join(row), flush=True)
    print("warps: " + ", ".join(f"{t}->{d}" for t, d in sorted(warps.items())), flush=True)
    print("npc templates: " + str(sorted(npcs)), flush=True)
    # BFS verdicts, exactly travel's planner
    for goal in ((13, 2), (12, 3), (24, 26)):
        p = tv.bfs(g, cur, lambda t, gg=goal: t == gg, walkable=g.walkable)
        print(f"BFS {cur} -> {goal}: {'len ' + str(len(p)) if p else 'NO ROUTE'}", flush=True)
    # elevation sample along the lobby mouth
    elevs = {t: g.elev.get((t[0] + OFF, t[1] + OFF)) for t in
             [cur, (23, 25), (22, 24), (22, 12), (23, 12), (13, 2)]}
    print(f"elevations: {elevs}", flush=True)

dump("IMMEDIATELY AFTER LANDING")
for _ in range(600):
    b.run_frame()
dump("AFTER 600 IDLE FRAMES (staleness re-read)")
