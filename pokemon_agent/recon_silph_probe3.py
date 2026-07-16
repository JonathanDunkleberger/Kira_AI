"""recon_silph_probe3.py — 5F full-map ASCII truth: walls, warps, objects, live NPCs, and
the BFS path to the Card Key ball. Decides WHY travel paces instead of arriving.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_silph_probe3.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SILPH = [(1, 47 + i) for i in range(11)]


def main():
    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    camp.enter_warp(pick=(33, 30))
    for _ in range(80):
        b.run_frame()
    for target in SILPH[1:5]:
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == target]
        camp.enter_warp(pick=cands[0])
        if st.in_battle(b):
            fight()
        for _ in range(60):
            b.run_frame()
    if tuple(tv.map_id(b)) != SILPH[4]:
        print(f"!! not on 5F (at {tv.map_id(b)})", flush=True)
        return 1

    cur = tuple(tv.coords(b) or (0, 0))
    g = tv.Grid(b)
    wts = {tuple(w[0]) for w in tv.read_warps(b)}
    objs = tv.read_object_templates(b)
    npc_live = set()
    try:
        npc_live = {tuple(t) for t in tv._npc_tiles(b)}
    except Exception as e:
        print(f"npc_tiles failed: {e}", flush=True)
    print(f"cur={cur} warps={sorted(wts)}", flush=True)
    print(f"templates={objs}", flush=True)
    print(f"live npc tiles={sorted(npc_live)}", flush=True)

    p = tv.bfs(g, cur, lambda t: t == (21, 21),
               walkable=lambda sx, sy: g.walkable(sx, sy)
               and (sx - tv.MAP_OFFSET, sy - tv.MAP_OFFSET) not in wts)
    path = {tuple(t) for t in (p or [])}
    print(f"path len={len(p) if p else 0}", flush=True)

    obj_t = {tuple(o[0]): o[1] for o in objs if o[2]}
    rows = []
    for y in range(-1, 30):
        line = []
        for x in range(-1, 40):
            t = (x, y)
            if t == cur:
                c = "@"
            elif t == (22, 21):
                c = "B"
            elif t in wts:
                c = "W"
            elif t in npc_live:
                c = "N"
            elif t in obj_t:
                c = "o"
            elif t in path:
                c = "*"
            elif g.walkable(x, y):    # Grid.walkable takes SAVE coords (adds OFFSET itself)
                c = "."
            else:
                c = "#"
            line.append(c)
        rows.append(f"{y:3d} {''.join(line)}")
    print("     " + "".join(str(x % 10) for x in range(-1, 40)), flush=True)
    print("\n".join(rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
