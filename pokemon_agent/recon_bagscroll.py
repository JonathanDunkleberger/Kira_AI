"""recon_bagscroll.py — derive the in-battle BAG LIST SCROLL address (the mart-class bug).

e4_run2 (Agatha) proved use_item_in_battle's row nav lies on scrolled lists: BAG_CURSOR
(0x0203AD04) read cursor==row==5 yet A selected CANCEL/wrong entry ("selected but item 23
NOT consumed"). The mart already taught us TRUE selection = row + scrollOffset; the battle
bag needs its own scroll address. Struct reasoning (pret gBagMenuState: pocket u16 @AD02,
cursorPosition[3] @AD04, itemsAbove[3] @AD0A) predicts 0x0203AD0A — VERIFY it here.

Boots the banked_E4 room3 state (Agatha's room, trainer 2 tiles up), opens the battle bag,
presses DOWN/UP through the Items pocket, and dumps the RAM window 0x0203ACF0..0x0203AD28
after every press so the cursor byte and the scroll byte identify themselves.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_bagscroll.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import firered_ram as ram            # noqa: E402
import travel as tv                  # noqa: E402
from battle_agent import BattleAgent, BAG_CURSOR  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_E4")
WIN0, WIN1 = 0x0203ACF0, 0x0203AD28


def main():
    b = Bridge(ROM)
    with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()
    print(f"boot map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    # walk up + talk until the battle opens
    for attempt in range(12):
        if fight_open():
            break
        b.press("UP", 26, 10, lambda: None, owner="agent")
        for _ in range(60):
            b.run_frame()
        b.press("A", 8, 12, lambda: None, owner="agent")
        for _ in range(90):
            b.run_frame()
            if fight_open():
                break
    if not fight_open():
        print("!! battle never opened — abort", flush=True)
        return 1
    print("battle OPEN", flush=True)

    ba = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                     log=lambda m: print(m, flush=True), choose=None)
    ids = [i for i, _ in ba._items_pocket()]
    print(f"Items pocket ({len(ids)}): {ids}", flush=True)
    if not ba._settle_action_menu():
        print("!! no action menu", flush=True)
        return 1
    if not ba._open_bag():
        print("!! bag would not open", flush=True)
        return 1
    for _ in range(8):
        if b.rd8(ram.GBAG_POCKET) == 0:
            break
        ba._tap("LEFT")
        ba._wait(12)
    print(f"pocket={b.rd8(ram.GBAG_POCKET)} (want 0)", flush=True)

    def dump(tag):
        vals = [b.rd8(a) for a in range(WIN0, WIN1)]
        print(f"{tag:>8}: " + " ".join(f"{v:02x}" for v in vals), flush=True)
        return vals

    base = dump("base")
    changed = {}
    seq = ["DOWN"] * 9 + ["UP"] * 9
    prev = base
    for n, key in enumerate(seq):
        ba._tap(key)
        ba._wait(12)
        vals = dump(f"{key[0]}{n}")
        for i, (a, v) in enumerate(zip(prev, vals)):
            if a != v:
                changed.setdefault(WIN0 + i, []).append((n, key[0], a, v))
        prev = vals
    print("\n=== bytes that moved with the list ===", flush=True)
    for addr in sorted(changed):
        marks = "".join(f"[{k}{n}:{a}->{v}]" for n, k, a, v in changed[addr])
        tag = ""
        if addr == BAG_CURSOR:
            tag = "  <== BAG_CURSOR (known)"
        if addr == 0x0203AD0A:
            tag = "  <== predicted itemsAbove[0]"
        print(f"  {addr:#010x}: {marks}{tag}", flush=True)
    # leave the bag cleanly (B out) so nothing dangles if a human pokes the state after
    ba._exit_bag()
    return 0


if __name__ == "__main__":
    sys.exit(main())
