"""recon_fuchsia_mart2.py — SHIFT-1 (night): identify the Fuchsia Mart DOOR + control-verify the BUY-list
row order, starting from koga_retry_kit.state (she is ALREADY on the Fuchsia overworld — the real
autonomous start), NOT the flaky drive-from-fuchsia_potions that never reached the city. Reads the live
_door_tiles(), enters each, and at each interior steps to the clerk and tries to open the buy list; the
one that opens is the Mart. Then CONTROL-VERIFIES row order by buying ONE unit at each row and reading the
bag-delta (which item id rose) — ground truth for MART_STOCK[FUCHSIA].
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")

from bridge import Bridge                                          # noqa: E402
import travel as tv                                               # noqa: E402
from battle_agent import BattleAgent                               # noqa: E402
from campaign import (Campaign, resolve_state, MART_CLERK_FRONT,   # noqa: E402
                      MART_CURSOR, MART_SCROLL, ITEM_NAMES)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def bag_snapshot(camp):
    """id -> qty for the item ids we care about (heals/balls/cures), via camp.bag_count."""
    ids = [2, 3, 4, 13, 22, 21, 20, 19, 23, 28, 14, 17, 18, 86, 87]
    return {i: camp.bag_count(i) for i in ids}


def main():
    fixture = sys.argv[1] if len(sys.argv) > 1 else "koga_retry_kit.state"
    b = Bridge(ROM)
    with open(resolve_state(fixture), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    def chooser(kind, options, ctx):
        if kind == "battle_item":
            return ("use_potion" if "use_potion" in options else next(iter(options.keys())))
        return None

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: None, choose=chooser).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    m = tv.map_id(b)
    print(f"FIXTURE {fixture}: map={m} coords={tv.coords(b)} money={camp.money()}", flush=True)
    bag = bag_snapshot(camp)
    print("BAG:", {ITEM_NAMES.get(i, i): q for i, q in bag.items() if q}, flush=True)
    pc = b.rd8(0x02024029)  # not reliable across; use party count via ram in campaign
    try:
        import firered_ram as ram
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        import pokemon_state as st
        lvls = [b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54) for s in range(cnt)]
        print(f"PARTY: count={cnt} levels={lvls}", flush=True)
    except Exception as e:
        print(f"party read err {e}")

    if m[0] == 3:
        doors = camp._door_tiles()
        print(f"OVERWORLD doors ({len(doors)}): {sorted(doors)}", flush=True)
    else:
        print(f"!! not on an overworld map (group {m[0]}) — she is inside {m}. Doors unreliable.", flush=True)
        doors = camp._door_tiles()
        print(f"interior doors: {sorted(doors)}", flush=True)

    snap = b.save_state()
    mart_door = None
    for door in sorted(doors):
        b.load_state(snap)
        for _ in range(10):
            b.run_frame()
        try:
            r = camp.enter_warp(pick=door)
        except Exception as e:
            print(f"door {door}: enter_warp err {e}"); continue
        inside = tv.map_id(b)
        if inside[0] == 3:
            print(f"door {door}: did not warp (still overworld {inside})"); continue
        # dump NPC objects so a mart clerk (behind a counter, top-left) is visible
        OB, SZ = 0x02036E38, 0x24
        obs = []
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o + 0x00) & 1):
                continue
            obs.append((b.rd8(o + 0x05), b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7, b.rd8(o + 0x18) & 0x0F))
        print(f"door {door} -> interior {inside}: coords_on_entry={tv.coords(b)} objs(gfx,x,y,face)={obs}", flush=True)
        # try to open the buy list
        opened = False
        try:
            camp._step_to(MART_CLERK_FRONT)
            for _ in range(20):
                b.run_frame()
            opened = bool(camp._mart_enter_buylist())
        except Exception as e:
            print(f"door {door} -> {inside}: buylist probe err {e}")
        print(f"door {door} -> interior {inside}: buylist_opened={opened}", flush=True)
        if opened:
            mart_door = door
            print(f"\n*** FUCHSIA MART DOOR = {door} (interior {inside}) ***", flush=True)
            # CONTROL-VERIFY row order: at each row 0..6 buy one unit, read which id rose.
            rows = {}
            for row in range(7):
                if not camp._mart_goto_row(row):
                    print(f"  row {row}: goto failed"); continue
                before = bag_snapshot(camp)
                price = camp._mart_buy_one()
                after = bag_snapshot(camp)
                rose = [i for i in after if after[i] > before.get(i, 0)]
                if rose:
                    rid = rose[0]
                    rows[row] = rid
                    print(f"  row {row}: bought id {rid} ({ITEM_NAMES.get(rid, '?')}) price~{price}", flush=True)
                else:
                    print(f"  row {row}: no bag delta (price~{price}) — likely CANCEL/unaffordable/end", flush=True)
            print(f"\nMART_STOCK[FUCHSIA] row order = {[rows.get(r) for r in range(7)]}", flush=True)
            break
    if mart_door is None:
        print("\n!! NO MART DOOR FOUND — none of the doors opened a buy list", flush=True)


if __name__ == "__main__":
    main()
