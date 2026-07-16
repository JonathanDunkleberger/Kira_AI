"""recon_cerulean_states.py — is the Cerulean south-exit block universal or savestate-specific?

For each candidate state: load, report map/player, the live NPC tiles, and whether BFS reaches the
south edge (a) avoiding all live NPCs and (b) allowing everything. If some states have a clear south
route and others don't, the block is savestate-specific (NPC config), not a fixed game wall. Read-only.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402
from travel import Grid, bfs       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ST = os.path.join(_HERE, "states")
OFF = tv.MAP_OFFSET
OB, SZ = 0x02036E38, 0x24

CANDIDATES = [
    "workshop/misty_done.state",
    "kira/seg_cerulean.state",
    "kira/seg_cascade_badge.state",
    "archive/cerulean_entered_auto.state",
    "archive/cerulean_caught.state",
    "archive/cerulean_entered.state",
    "misty_overworld.state",
]


def npc_tiles(b):
    out = set()
    for i in range(1, 16):
        o = OB + i * SZ
        if b.rd8(o) & 1:
            out.add((b.rds16(o + 0x10) - OFF, b.rds16(o + 0x12) - OFF))
    return out


def check(b):
    grid = Grid(b)
    cur = tv.coords(b)
    if cur is None:
        return "player coords None (cutscene?)"
    npc = npc_tiles(b)

    def goal(t):
        return t[1] >= grid.sy_hi

    def mk(avoid):
        return lambda x, y: grid.walkable(x, y) and (x, y) not in avoid
    p_avoid = bfs(grid, cur, goal, walkable=mk(npc))
    p_all = bfs(grid, cur, goal, walkable=mk(set()))
    south_npcs = sorted(t for t in npc if 28 <= t[1] <= 33 and 24 <= t[0] <= 29)
    return (f"player={cur} south-area-NPCs={south_npcs} | "
            f"BFS->south avoiding NPCs: {'PATH' if p_avoid else 'BLOCKED'} | "
            f"allowing all: {'PATH(exit ' + str(p_all[-1]) + ')' if p_all else 'NONE'}")


def main():
    b = Bridge(ROM)
    for rel in CANDIDATES:
        path = os.path.join(ST, rel)
        if not os.path.exists(path):
            print(f"  {rel:42s}  (missing)", flush=True)
            continue
        with open(path, "rb") as f:
            b.load_state(f.read())
        for _ in range(60):
            b.run_frame()
        m = tv.map_id(b)
        tag = "" if m == (3, 3) else f"  [NOT Cerulean: map={m}]"
        print(f"\n>>> {rel}{tag}", flush=True)
        try:
            print("   " + check(b), flush=True)
        except Exception as e:
            print(f"   ERROR: {e!r}", flush=True)


if __name__ == "__main__":
    main()
