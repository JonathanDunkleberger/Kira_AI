"""recon_cerulean_mart.py — identify the Cerulean Poké Mart door (to add CERULEAN_MART_DOOR to
CITY_MART_DOORS so she can autonomously buy potions/balls). Enters each Cerulean building and tests the
buy clerk: the building where the BUY list opens (MART_CURSOR responds) is the Mart."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                               # noqa: E402
from campaign import Campaign, resolve_state, MART_CLERK_FRONT, MART_CURSOR  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
# Cerulean (3,3) overworld building-entrance warps (live-read earlier); (22,19)=Pokémon Center (skip).
DOORS = [(10, 11), (30, 11), (15, 17), (31, 21), (13, 28), (29, 28), (17, 11)]


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._exit_to_overworld()
    for _ in range(30):
        b.run_frame()
    snap = b.save_state()
    print(f"Cerulean overworld map={tv.map_id(b)} coords={tv.coords(b)}\n", flush=True)

    for door in DOORS:
        b.load_state(snap)
        for _ in range(10):
            b.run_frame()
        try:
            r = camp.enter_warp(pick=door)
        except Exception as e:
            print(f"door {door}: enter_warp err {e}"); continue
        inside = tv.map_id(b)
        if inside[0] == 3:                                        # still overworld -> warp didn't trigger
            print(f"door {door}: did not enter (still overworld {inside})"); continue
        is_mart = False
        try:
            camp._step_to(MART_CLERK_FRONT)
            cur0 = b.rd8(MART_CURSOR)
            is_mart = bool(camp._mart_enter_buylist())
        except Exception as e:
            print(f"door {door} -> interior {inside}: buy-test err {e}"); continue
        print(f"door {door} -> interior {inside}: {'*** MART ***' if is_mart else 'not a mart'}", flush=True)


if __name__ == "__main__":
    main()
