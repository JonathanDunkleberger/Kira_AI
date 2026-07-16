"""recon_9f_probe.py — ground truth for the strike13 walls: (a) the 5F pocket exit — which
neighbors of the pocket pad (10,20) are actually reachable from the ball area, and where the
east column leaks; (b) 9F connectivity from the pad landing (22,18) — to the 3F pad (9,4),
the heal woman (2,16), the stairs (16,2), and which door tiles cut the floor.
Loads the strike13 stage save (5F pocket, Card Key in hand).
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_9f_probe.py
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
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_silph")


def main():
    b = Bridge(ROM)
    with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
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

    b.set_input_owner("agent")

    # the stage state may hold an OPEN box (saved mid-script) — drain before anything
    print(f"boot box_open={dd_box(b)}", flush=True)
    for _ in range(20):
        if not dd_box(b):
            break
        b.press("A", 8, 12, camp.render, owner="agent")
        for _ in range(20):
            b.run_frame()
    print(f"post-drain box_open={dd_box(b)}", flush=True)

    def dump(tag, marks):
        cur = tuple(tv.coords(b) or (0, 0))
        g = tv.Grid(b)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        objs = tv.read_object_templates(b)
        try:
            npc_live = {tuple(t) for t in tv._npc_tiles(b)}
        except Exception:
            npc_live = set()
        obj_t = {tuple(o[0]) for o in objs if o[2]}
        print(f"\n== {tag}: map={tv.map_id(b)} cur={cur}", flush=True)
        print(f"warps={sorted(wts)}", flush=True)
        print(f"templates={[(tuple(o[0]), o[1], o[2]) for o in objs]}", flush=True)
        print(f"live npc tiles={sorted(npc_live)}", flush=True)
        # reachability with the strike's walk_path_to masking (warps + template bodies)
        seen = {cur}
        q = [cur]
        while q:
            cx, cy = q.pop()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                t = (cx + dx, cy + dy)
                if (t not in seen and g.walkable(t[0], t[1])
                        and t not in wts and t not in obj_t):
                    seen.add(t)
                    q.append(t)
        for name, t in marks.items():
            w = g.walkable(t[0], t[1])
            print(f"  {name} {t}: walkable={w} reach={'YES' if t in seen else 'no'}",
                  flush=True)
        rows = []
        for y in range(-1, 30):
            line = []
            for x in range(-1, 40):
                t = (x, y)
                if t == cur:
                    c = "@"
                elif t in wts:
                    c = "W"
                elif t in npc_live:
                    c = "N"
                elif t in obj_t:
                    c = "o"
                elif t in seen:
                    c = "+"
                elif g.walkable(x, y):
                    c = "."
                else:
                    c = "#"
                line.append(c)
            rows.append(f"{y:3d} {''.join(line)}")
        print("     " + "".join(str(x % 10) for x in range(-1, 40)), flush=True)
        print("\n".join(rows), flush=True)
        return seen

    # ── 5F pocket truth
    dump("5F pocket (key in hand)", {
        "pad(10,20)": (10, 20), "padW(9,20)": (9, 20), "padE(11,20)": (11, 20),
        "padN(10,19)": (10, 19), "padS(10,21)": (10, 21),
        "east col (35,8)": (35, 8), "east col (36,8)": (36, 8),
        "east col (35,9)": (35, 9), "east col (36,20)": (36, 20),
        "stairs6F app (28,3)": (28, 3),
    })

    # ── ride the pocket pad: approach (10,21) along row 21, press UP onto (10,20)
    cur = tuple(tv.coords(b) or (0, 0))
    g = tv.Grid(b)
    wts = {tuple(w[0]) for w in tv.read_warps(b)}
    p = tv.bfs(g, cur, lambda t: t == (10, 21),
               walkable=lambda sx, sy: g.walkable(sx, sy) and (sx, sy) not in wts)
    print(f"\npath to (10,21): {p}", flush=True)
    if p:
        for t in p[1:]:
            if not camp._step_to(tuple(t)):
                print(f"!! step {tuple(t)} failed", flush=True)
                break
    if tuple(tv.coords(b) or ()) == (10, 21):
        m0 = tuple(tv.map_id(b))
        b.press("UP", 26, 10, camp.render, owner="agent")
        for _ in range(180):
            b.run_frame()
            if tuple(tv.map_id(b)) != m0:
                break
        for _ in range(240):
            b.run_frame()
    print(f"\nafter pad: map={tv.map_id(b)} cur={tv.coords(b)}", flush=True)

    # ── 9F truth from the landing
    if tuple(tv.map_id(b)) == (1, 55):
        dump("9F from pad landing", {
            "pad3F(9,4)": (9, 4), "p3W(8,4)": (8, 4), "p3E(10,4)": (10, 4),
            "p3N(9,3)": (9, 3), "p3S(9,5)": (9, 5),
            "heal front(2,17)": (2, 17), "stairs8F(16,2)": (16, 2),
            "door(2,10)": (2, 10), "door(3,11)": (3, 11),
            "door(12,16)": (12, 16), "door(13,17)": (13, 17),
            "door(21,6)": (21, 6), "door(22,7)": (22, 7),
            "door(21,12)": (21, 12), "door(22,13)": (22, 13),
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
