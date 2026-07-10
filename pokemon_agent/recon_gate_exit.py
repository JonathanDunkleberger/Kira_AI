"""recon_gate_exit.py — decide how to exit the Route-2 aide gate (15,2) into the NORTH (cave) pocket.

The aide gate is a pass-through: south doors (18,46)/(19,46) [south pocket, sealed from the cave by
a one-way ledge], north doors (18,41)/(19,41) [north pocket, reaches the cave mouth (17,11)]. The
errand enters from the south and must exit NORTH. This probe: from banked_HM05 (north pocket), walk
to the south aide door, enter the gate (south entry), then try enter_warp(prefer='north') and report
whether she lands in the north pocket (reaches the cave-exit tile 17,12).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge                        # noqa: E402
import travel as tv                              # noqa: E402
from battle_agent import BattleAgent             # noqa: E402
from campaign import Campaign                    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
GATE = (15, 2)
CAVE_EXIT_TILE = (17, 12)


def in_north_pocket(b):
    try:
        g = tv.Grid(b)
        return bool(tv.bfs(g, tuple(tv.coords(b)), lambda t: t == CAVE_EXIT_TILE, walkable=g.walkable))
    except Exception:
        return False


def main():
    b = Bridge(ROM)
    with open("G:/temp/longrun/banked_HM05/kira_campaign.state", "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda *a, **k: None).run(max_seconds=60)
    camp = Campaign(b, battle_runner=runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    print("START", tuple(tv.map_id(b)), tuple(tv.coords(b)), "north_pocket=", in_north_pocket(b))

    # enter the gate from the NORTH door (18,41) (reachable from the north pocket)
    r = camp.enter_warp(pick=(18, 41))
    print("enter gate via (18,41) ->", r, "now", tuple(tv.map_id(b)), tuple(tv.coords(b)))
    if tuple(tv.map_id(b)) != GATE:
        print("FAIL: not inside the gate; abort"); return
    doors = sorted(camp._door_tiles())
    print("gate interior door tiles:", doors)
    warps = [(tuple(w), tuple(d)) for (w, d, _i) in tv.read_warps(b)]
    print("gate interior warps (tile->dest):", warps)
    # exit via SOUTH first (simulate the aide-side exit that traps her), report the landing pocket
    rs = camp.enter_warp(prefer="south")
    print("exit prefer=south ->", rs, "now", tuple(tv.map_id(b)), tuple(tv.coords(b)),
          "north_pocket=", in_north_pocket(b))
    # now from wherever she is, if not north pocket, re-enter and exit NORTH
    if tuple(tv.map_id(b)) == (3, 20) and not in_north_pocket(b):
        print(">> landed in SOUTH pocket (the trap). Re-entering to exit NORTH...")
        camp.enter_warp(pick=(18, 46))
        print("   re-entered gate:", tuple(tv.map_id(b)), tuple(tv.coords(b)))
        rn = camp.enter_warp(prefer="north")
        print("   exit prefer=north ->", rn, "now", tuple(tv.map_id(b)), tuple(tv.coords(b)),
              "north_pocket=", in_north_pocket(b))
    print("RESULT: north_pocket=", in_north_pocket(b), "at", tuple(tv.coords(b)))


if __name__ == "__main__":
    main()
