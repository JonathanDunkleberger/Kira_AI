"""recon_shop_e2e.py — end-to-end test of the REAL autonomous shop path at Cerulean: exit to
overworld, buy_at_mart(potions + balls), confirm both pockets increment and no LOUD abort. No saves."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import (Campaign, resolve_state, CERULEAN_MART_DOOR,  # noqa: E402
                      ITEM_POTION, ITEM_POKE_BALL)


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
    p0, ball0, m0 = camp.bag_count(ITEM_POTION), camp._ball_count(), camp.money()
    print(f"BEFORE: potions={p0} balls={ball0} money={m0}", flush=True)
    bought = camp.buy_at_mart(CERULEAN_MART_DOOR, [(ITEM_POTION, 6), (ITEM_POKE_BALL, 6)])
    p1, ball1, m1 = camp.bag_count(ITEM_POTION), camp._ball_count(), camp.money()
    print(f"\nAFTER:  potions={p1} balls={ball1} money={m1}", flush=True)
    print(f"bought={bought}  back_on_map={tv.map_id(b)}@{tv.coords(b)}")
    ok = (p1 == p0 + 6 and ball1 == ball0 + 6 and m1 < m0 and tv.map_id(b)[0] == 3)
    print("RESULT:", "PASS — autonomous shopping works at Cerulean" if ok else "FAIL — see deltas")


ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
if __name__ == "__main__":
    main()
