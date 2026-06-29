"""recon_buytest.py — confirm the Cerulean Mart (interior (7,7), door (29,28)) buy flow WORKS in a
clean run, and read the real row->item stock order empirically. Buys one unit at each of rows 0..2 and
reports which bag item incremented + price. No canonical saves."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import Campaign, resolve_state, ITEM_NAMES, MART_CURSOR  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
MART_DOOR = (29, 28)
# scan these item ids for a count change after a buy
SCAN_IDS = [4, 3, 13, 22, 21, 14, 15, 16, 17, 18, 19, 20, 23, 25, 26]


def bag_snapshot(camp):
    return {i: camp._item_count(i) for i in SCAN_IDS}


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
    print(f"overworld map={tv.map_id(b)} player={tv.coords(b)} money={camp.money()}", flush=True)
    r = camp.enter_warp(pick=MART_DOOR)
    print(f"enter Mart: {r} -> interior {tv.map_id(b)}", flush=True)
    for _ in range(60):
        b.run_frame()
    opened = camp._mart_enter_buylist()
    print(f"_mart_enter_buylist() -> {opened}  cursor={b.rd8(MART_CURSOR)}", flush=True)
    if not opened:
        print("BUY LIST DID NOT OPEN — menu actuation failed at the real Mart."); return
    order = {}
    for row in range(9):
        if not camp._mart_goto_row(row):
            print(f"row {row}: couldn't reach (cursor stuck) — end of list"); break
        before = bag_snapshot(camp)
        m0 = camp.money()
        if m0 < 1000:
            print(f"row {row}: stopping (money {m0} low)"); break
        price = camp._mart_buy_one()
        after = bag_snapshot(camp)
        changed = [i for i in SCAN_IDS if after[i] != before[i]]
        nm = ITEM_NAMES.get(changed[0], f"#{changed[0]}") if changed else f"?(price{price})"
        if changed:
            order[row] = changed[0]
        print(f"row {row}: price={price} item={nm} id={changed[0] if changed else '?'}", flush=True)
    print(f"\nCERULEAN stock row->id: {order}")


if __name__ == "__main__":
    main()
