"""recon_b1f_probe2.py — night shift 6, part 2: the y=19/20 'wall band' on hideout B1F
(1,42) reads as collision in our Grid but the re-grade's successful exit WALKED through it.
Hypothesis: it's the SPIN-TILE field (B1F is the documented spin floor; probe 1's spin
classifier — bh 0x54..0x58, calibrated on B4F which has zero — found none here, so the
behavior byte range is likely wrong). Method: dump raw map-grid entries (collision bits +
behavior byte) across the band, then GAME-TRUTH: walk her onto it and see if she moves.
READ-ONLY on the bundle."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
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

camp.enter_warp(pick=(20, 23))
seen = {(1, 45)}
for attempt in range(12):
    if tuple(tv.map_id(b)) == (1, 42):
        break
    if not elevator_nav.is_car(b):
        ws = [tuple(w[0]) for w in tv.read_warps(b) if tuple(w[1]) == (1, 46)] \
             or [tuple(w[0]) for w in tv.read_warps(b)]
        camp.enter_warp(pick=ws[0])
        continue
    if elevator_nav.ride(b, camp, attempt % 5, avoid=seen):
        seen.add(tuple(tv.map_id(b)))
print(f"landed: map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

g = tv.Grid(b)
OFF = tv.MAP_OFFSET
ml = b.rd32(tv.GMAPHEADER)
attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
mp = b.rd32(tv.BACKUP_LAYOUT + 8)

def raw(x, y):
    bx, by = x + OFF, y + OFF
    e = b.rd16(mp + (by * g.w + bx) * 2)
    mid = e & 0x3FF
    base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1], mid - tv.NUM_PRIMARY)
    bh = b.rd32(base + idx * 4) & 0xFF
    return {"entry": hex(e), "mid": hex(mid), "coll": (e >> 10) & 3,
            "elev": (e >> 12) & 0xF, "behavior": hex(bh)}

print("\n-- raw band dump (x=18..23, y=18..22) --", flush=True)
for y in range(18, 23):
    for x in range(18, 24):
        print(f"({x},{y}) {raw(x, y)}", flush=True)

print("\n-- game-truth walk test --", flush=True)
p = tv.bfs(g, tuple(tv.coords(b)), lambda t: t == (21, 21), walkable=g.walkable)
print(f"path to (21,21): {'len ' + str(len(p)) if p else 'NO ROUTE'}", flush=True)
if p:
    for t in p[1:]:
        if not camp._step_to(t):
            print(f"  step to {t} FAILED", flush=True)
            break
print(f"at: {tv.coords(b)}", flush=True)
for i in range(6):
    before = tuple(tv.coords(b))
    b.press("UP", 12, 24, owner="agent")
    for _ in range(30):
        b.run_frame()
    after = tuple(tv.coords(b))
    print(f"press UP #{i}: {before} -> {after}"
          + ("   <-- MOVED THROUGH THE 'WALL'" if after != before and after[1] < 21 else ""),
          flush=True)
    if after == before:
        break
print(f"final: map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
