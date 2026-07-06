"""recon_route6_wedge.py — arsenal #4 on the Route 6 (3,24)@(17,25) hard wedge (2026-07-06 resume).

Boot canonical (Route 5), cross via the PROVEN UGP pass-through, then attempt the south edge toward
Vermilion exactly as head_to_gym does. At the stall point: FRAME GRAB + object scan + behavior dump +
BFS probes — pixels and RAM together. Read-only on canonical (no persistence).
RUN: python pokemon_agent/recon_route6_wedge.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import field_moves as fm               # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\661370f2-1025-435c-8cf5-d2593621c432\scratchpad"


def dump_area(b, x0, x1, y0, y1):
    ml = b.rd32(tv.GMAPHEADER)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
    w = b.rd32(tv.BACKUP_LAYOUT); mp = b.rd32(tv.BACKUP_LAYOUT + 8)
    for sy in range(y0, y1):
        line = []
        for sx in range(x0, x1):
            e = b.rd16(mp + ((sx + 7) + w * (sy + 7)) * 2)
            mid = e & 0x3FF
            base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
            bh = b.rd32(base + idx * 4) & 0xFF
            col = (e & 0x0C00) >> 10
            line.append(f"{bh:02x}{'*' if col == 0 else ' '}")
        print(f"y={sy:02d}: " + " ".join(line), flush=True)
    print("cols:  " + "  ".join(f"{x:02d} " for x in range(x0, x1)), flush=True)


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    b.set_input_owner("agent")

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None,
                    on_event=lambda s, **k: print(f"   [voice] {s[:90]}", flush=True))
    print(f"boot: {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    # cross to Route 6 via the proven connector
    pt = camp._door_passthrough(budget_s=300)
    print(f"passthrough -> {pt}; now {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    if tuple(tv.map_id(b)) != (3, 24):
        print("!! did not reach Route 6 — aborting recon", flush=True)
        return
    b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "r6_arrival.png"))
    # attempt the south edge exactly as the spine march would
    r = camp.trav.travel(target_map=(3, 5), edge="south", max_seconds=120)
    m, c = tv.map_id(b), tv.coords(b)
    print(f"south edge attempt -> {r} (fail={getattr(camp.trav, 'last_fail_reason', '')}); "
          f"now {m}@{c}", flush=True)
    b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "r6_wedge.png"))
    print("objects:", fm.scan_field_objects(b), flush=True)
    # NPC tiles the traveler sees
    try:
        print("npc tiles:", sorted(camp.trav._npc_tiles())[:20], flush=True)
    except Exception as e:
        print("npc read err", e, flush=True)
    cx, cy = c
    dump_area(b, max(0, cx - 8), cx + 9, max(0, cy - 4), cy + 8)
    grid = tv.Grid(b)
    print(f"playable rows 0..{grid.sy_hi}, cols 0..{grid.sx_hi}", flush=True)
    p = tv.bfs(grid, tuple(c), lambda t: t[1] >= grid.sy_hi)
    print("BFS to south border:", f"PATH len {len(p)} via {p[len(p)//2]}" if p else "NONE", flush=True)


if __name__ == "__main__":
    main()
