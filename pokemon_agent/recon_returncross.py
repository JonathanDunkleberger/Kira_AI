"""recon_returncross.py — FAST isolated test of the post-Flash RETURN-CROSS navigation (shift 3).

Boots a Route-2 (aide-area) fixture and drives ONLY the return-cross primitives:
  (1) exit to the Route 2 overworld, (2) travel to the Diglett's Cave-exit tile (17,12)
  cutting the regrown tree, (3) enter_warp the cave mouth (17,11)->(1,36), (4) _cross_cave
  east to Route 11 (3,29). Verifies the fix without the ~10-min dex errand.

RUN:  .venv/Scripts/python.exe -u pokemon_agent/recon_returncross.py [bank_name]
      (default bank = banked_HM05, a Champion-era Route-2 state — party differs, nav is what we test)
"""
import os
import sys
import time
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge                        # noqa: E402
import travel as tv                              # noqa: E402
import firered_ram as ram                        # noqa: E402
from battle_agent import BattleAgent             # noqa: E402
from campaign import Campaign, STATES_WORKSHOP   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ROUTE11, ROUTE2 = (3, 29), (3, 20)


def main():
    bank = sys.argv[1] if len(sys.argv) > 1 else "banked_HM05"
    src = f"G:/temp/longrun/{bank}"
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(src, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda *a, **k: None).run(max_seconds=120)
    camp = Campaign(b, battle_runner=runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    # warm the world model so nav is graph-aware (copy the bank's sidecar into workshop)
    try:
        wm = os.path.join(src, "world_model.json")
        if os.path.exists(wm):
            camp.world.load(wm) if hasattr(camp.world, "load") else None
    except Exception as e:
        L(f"world load skipped: {e}")

    L(f"START map={tuple(tv.map_id(b))} coord={tuple(tv.coords(b))} "
      f"flash_known={ram.pokedex_owned_count(b)}dex")

    # --- the return-cross primitives, verbatim from _flash_errand's fixed block ---
    DIGLETT_R2_MOUTH, CAVE_EXIT_TILE = (17, 11), (17, 12)
    for _ in range(3):
        if tv.map_id(b)[0] == 3:
            break
        camp._exit_to_overworld()
    cur_r = tuple(tv.map_id(b))
    L(f"on {cur_r}@{tuple(tv.coords(b))}; routing to the cave mouth {DIGLETT_R2_MOUTH}")
    if cur_r == ROUTE2:
        r_t = camp.trav.travel(target_map=None, arrive_coord=CAVE_EXIT_TILE, max_steps=400)
        L(f"travel to {CAVE_EXIT_TILE} -> {r_t} (now {tuple(tv.map_id(b))}@{tuple(tv.coords(b))})")
        r = camp.enter_warp(pick=DIGLETT_R2_MOUTH)
        L(f"cave-mouth enter -> {r} (now on {tuple(tv.map_id(b))})")
        if r == "warped":
            ok = camp._cross_cave(None, ROUTE11)
            L(f"_cross_cave -> {ok} (now on {tuple(tv.map_id(b))}@{tuple(tv.coords(b))})")
            if ok and tuple(tv.map_id(b)) == ROUTE11:
                L("RESULT: PASS — reached Route 11 via the cave (return-cross nav VERIFIED)")
                # push one more leg: edge-travel Route 11 -> Vermilion (west)
                camp._edge_travel((3, 5), "west")
                L(f"after edge_travel west -> now on {tuple(tv.map_id(b))}")
                return
            L("RESULT: FAIL — cross did not land on Route 11")
            return
        L("RESULT: FAIL — could not enter the cave mouth")
        return
    L(f"RESULT: FAIL — not on Route 2 after exit ({cur_r})")


if __name__ == "__main__":
    main()
