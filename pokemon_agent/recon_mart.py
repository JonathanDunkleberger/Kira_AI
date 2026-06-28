"""recon_mart.py - LIVE recon of the Mart BUY menu (PART C). No buy primitive exists; this maps the
interior + the buy-menu nav signals before building. Source-first + control-verified, never blind.

STAGE 1 (this run): boot a Mart city, enter the Mart, locate the clerk, talk, and dump what the BUY
menu looks like in RAM + a screenshot, so we can find the reliable signals (highlighted item id,
quantity, the YES/NO confirm, the money decrement) to drive a buy.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_mart.py [state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                            # noqa: E402
import firered_ram as ram                            # noqa: E402
import pokemon_state as st                           # noqa: E402
import travel as tv                                  # noqa: E402
from campaign import (Campaign, resolve_state, PEWTER, PEWTER_MART_DOOR,  # noqa: E402
                      VIRIDIAN, VIRIDIAN_MART_DOOR)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SHOT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\56687edb-aec7-4d80-be7e-8134df395075\scratchpad"


def log(m):
    print(f"   [mart-recon] {m}", flush=True)


def shot(b, name):
    try:
        b.frame_rgb().save(os.path.join(SHOT, name))
        log(f"screenshot -> {name}")
    except Exception as e:
        log(f"screenshot {name} failed: {e}")


def money(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20)
    return (b.rd32(sb1 + 0x0290) ^ key) & 0xFFFFFFFF


def dump_objects(camp):
    """Every loaded object event (clerk is an NPC behind the counter)."""
    for i in range(16):
        o = camp._OB + i * camp._SZ
        if camp.b.rd8(o) & 1:
            c = (camp.b.rds16(o + 0x10) - 7, camp.b.rds16(o + 0x12) - 7)
            log(f"  obj{i}: coord={c} type={camp.b.rd8(o + 0x07)} facing={camp.b.rd8(o + 0x18) & 0x0F}")


def main():
    state = sys.argv[1] if len(sys.argv) > 1 else "brock_done"
    b = Bridge(ROM)
    p = resolve_state(state + ".state")
    if not p:
        raise SystemExit(f"state {state}.state not found")
    with open(p, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    b.set_input_owner("agent")

    camp = Campaign(b, battle_runner=lambda: "win", on_event=lambda s, **k: log(f"[event] {s}"))
    log(f"booted {state}: map={tv.map_id(b)} coords={tv.coords(b)} money={money(b)}")

    # pick the Mart door for the current city
    m = tv.map_id(b)
    door = {PEWTER: PEWTER_MART_DOOR, VIRIDIAN: VIRIDIAN_MART_DOOR}.get(m)
    if door is None:
        log(f"!! no Mart door mapped for {m} — pass a Pewter/Viridian state"); return
    log(f"entering Mart via door {door}")
    r = camp.enter_warp(pick=door)
    log(f"enter_warp -> {r}; now map={tv.map_id(b)} coords={tv.coords(b)}")
    for _ in range(60):
        b.run_frame()
    shot(b, "mart_01_interior.png")
    dump_objects(camp)

    # ── STAGE 2: talk to the clerk and CAPTURE the press-by-press menu sequence ──────────────────
    CLERK_FRONT = (2, 4)
    log(f"stepping to clerk front {CLERK_FRONT}")
    camp._step_to(CLERK_FRONT)
    log(f"at {tv.coords(b)}; facing UP + A to start the clerk")
    for _ in range(4):
        b.press("UP", 8, 8, lambda: None, owner="agent")
        b.press("A", 8, 10, lambda: None, owner="agent")
        for _ in range(16):
            b.run_frame()
        if dd_open(b):
            break
    # advance greeting + auto-confirm BUY (default top) -> the item list (4 A presses, mapped seq),
    # then let the fade fully settle (was landing mid-fade = black) before probing the list.
    for _ in range(4):
        b.press("A", 8, 10, lambda: None, owner="agent")
        for _ in range(24):
            b.run_frame()
    for _ in range(120):
        b.run_frame()
    shot(b, "mart_list_a.png")
    log(f"in list? itemid@AD30={b.rd16(ram.GSPECIALVAR_ITEMID)} money={money(b)}")

    # CONTROL-PROVE the full buy mechanic end-to-end: navigate to POTION (row 1) confirming the cursor
    # index @0x2039940, buy 1, and VERIFY money -300 + bag Potion(13) +1 (ground truth). This proves the
    # cursor address + the per-unit buy flow before the primitive is built.
    CURSOR = 0x2039940
    log(f"cursor index now = {b.rd8(CURSOR)} (expect 0 at list top)")
    bag0 = bag_count(b, 13)
    money0 = money(b)
    # navigate to row 1 (Potion), verifying the index moves
    for _ in range(6):
        if b.rd8(CURSOR) == 1:
            break
        b.press("DOWN", 8, 10, lambda: None, owner="agent")
        for _ in range(14):
            b.run_frame()
    log(f"after nav: cursor index = {b.rd8(CURSOR)} (want 1 = Potion)")
    # BUY ONE robustly: qty defaults to 1 and "OK?" defaults to YES, so we just press A (advancing the
    # slow clerk text + confirming both defaults) and STOP THE INSTANT money drops — the ground-truth
    # "bought 1" signal. Breaking on the first drop is critical: another A would re-open Potion and buy
    # a second. (The per-unit loop is what the primitive will use; qty-box address never needed.)
    bought = False
    for k in range(25):
        b.press("A", 6, 10, lambda: None, owner="agent")
        for _ in range(12):
            b.run_frame()
        if money(b) < money0:
            log(f"   money dropped after {k+1} A-press(es) -> bought 1")
            bought = True
            break
    shot(b, "mart_buy_done.png")
    bag1 = bag_count(b, 13)
    money1 = money(b)
    log(f"== BUY CONTROL: money {money0} -> {money1} (delta {money1 - money0}, expect -300) | "
        f"Potion x{bag0} -> x{bag1} (delta {bag1 - bag0}, expect +1) ==")
    log(f"   {'PASS' if (bag1 == bag0 + 1 and money1 == money0 - 300) else 'FAIL'}")
    for _ in range(8):                                       # B out of list + menu -> overworld
        b.press("B", 6, 12, lambda: None, owner="agent")
        for _ in range(14):
            b.run_frame()
    log(f"final money={money(b)} map={tv.map_id(b)}")


def bag_count(b, item_id):
    """Count of item_id across the Items pocket (0x0310, 42 slots). qty is XOR'd with the low-16 key."""
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    for s in range(42):
        slot = sb1 + 0x0310 + s * 4
        if b.rd16(slot) == item_id:
            return b.rd16(slot + 2) ^ key
    return 0


SCAN_LO, SCAN_HI = 0x02036000, 0x0203C000


def ewram_snap(b):
    return {off: b.rd16(off) for off in range(SCAN_LO, SCAN_HI, 2)}


def diff_generic(a, c, pred):
    n = 0
    for off in a:
        va, vc = a[off], c[off]
        if va != vc and pred(va, vc):
            log(f"   {hex(off)}: {va} -> {vc}")
            n += 1
            if n > 50:
                log("   (more...)"); break
    if not n:
        log("   (no match)")


def dd_open(b):
    from dialogue_drive import box_open
    return box_open(b)


def _menu_seems_up(b):
    """Heuristic 'a menu/box is up' for the greeting->buysell transition."""
    return dd_open(b)


if __name__ == "__main__":
    main()
