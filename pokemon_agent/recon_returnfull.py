"""recon_returnfull.py — reproduce the aide-pocket TRAP and verify the fixed return-cross recovers.

From banked_HM05 (north pocket), DUMP her into the SOUTH pocket (enter aide gate (15,2) via the north
door, exit the south door -> (18,47), sealed from the cave by a one-way ledge). Then run the FIXED
return-cross logic (seek the north pocket by re-entering the gate + exiting north, travel the cave-exit
tile, enter the mouth, _cross_cave east to Route 11). PASS = reaches Route 11.
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
from campaign import Campaign                     # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ROUTE11, ROUTE2 = (3, 29), (3, 20)
DIGLETT_R2_MOUTH, CAVE_EXIT_TILE, AIDE_SOUTH_DOOR = (17, 11), (17, 12), (18, 46)


def np_(b):
    try:
        g = tv.Grid(b)
        return (tuple(tv.map_id(b)) == ROUTE2 and bool(tv.bfs(
            g, tuple(tv.coords(b)), lambda t: t == CAVE_EXIT_TILE, walkable=g.walkable)))
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

    # --- DUMP her into the SOUTH pocket to reproduce the post-teach trap ---
    camp.enter_warp(pick=(18, 41))          # into the gate from the north door
    camp.enter_warp(prefer="south")         # out the south door -> (18,47) south pocket
    print(f"TRAP set: on {tuple(tv.map_id(b))}@{tuple(tv.coords(b))} north_pocket={np_(b)}")

    # --- the FIXED return-cross logic (verbatim shape from _flash_errand) ---
    for _ in range(4):
        if np_(b):
            break
        if tv.map_id(b)[0] != 3:
            camp.enter_warp(prefer="north")
        else:
            camp.enter_warp(pick=AIDE_SOUTH_DOOR)
    print(f"after north-seek: on {tuple(tv.map_id(b))}@{tuple(tv.coords(b))} north_pocket={np_(b)}")
    if tuple(tv.map_id(b)) == ROUTE2 and np_(b):
        camp.trav.travel(target_map=None, arrive_coord=CAVE_EXIT_TILE, max_steps=400)
        r = camp.enter_warp(pick=DIGLETT_R2_MOUTH)
        print(f"cave-mouth enter -> {r} (now {tuple(tv.map_id(b))})")
        if r == "warped":
            ok = camp._cross_cave(None, ROUTE11)
            print(f"_cross_cave -> {ok} (now {tuple(tv.map_id(b))}@{tuple(tv.coords(b))})")
            if ok and tuple(tv.map_id(b)) == ROUTE11:
                camp._edge_travel((3, 5), "west")
                print(f"edge west -> {tuple(tv.map_id(b))}")
                print("RESULT: PASS — trap recovered, reached Route 11 -> Vermilion")
                return
    print("RESULT: FAIL")


if __name__ == "__main__":
    main()
