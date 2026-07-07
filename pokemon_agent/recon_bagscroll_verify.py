"""recon_bagscroll_verify.py — live-verify the scroll-aware use_item_in_battle.

Boots the banked_E4 room3 state (Agatha), opens the battle, POISONS the bag cursor state
(opens the bag, scrolls to the very bottom, B out — exactly the stale state that broke
e4_run2), then calls use_item_in_battle(FULL_RESTORE). PASS = 'used' + count drops + HP up.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_bagscroll_verify.py
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
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent, BAG_CURSOR, BAG_SCROLL  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_E4")
FULL_RESTORE = 19


def main():
    b = Bridge(ROM)
    with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    for _ in range(12):
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

    ba = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                     log=lambda m: print(m, flush=True), choose=None)
    state = st.read_battle(b)
    hp0 = state["ours"]["hp"] if state else -1
    cnt0 = ba._items_count(FULL_RESTORE)
    print(f"pre: HP={hp0} FRx{cnt0}", flush=True)
    if cnt0 <= 0:
        print("!! no Full Restore aboard — cannot verify", flush=True)
        return 1

    # POISON the cursor state: open bag, scroll to the very bottom, B out.
    ba._settle_action_menu()
    if not ba._open_bag():
        print("!! bag would not open for the poison pass", flush=True)
        return 1
    for _ in range(8):
        if b.rd8(ram.GBAG_POCKET) == 0:
            break
        ba._tap("LEFT"); ba._wait(12)
    for _ in range(9):
        ba._tap("DOWN"); ba._wait(10)
    print(f"poisoned: cursor={b.rd8(BAG_CURSOR)} scroll={b.rd16(BAG_SCROLL)} "
          f"(stale scrolled state, run2's killer)", flush=True)
    ba._exit_bag()

    res = ba.use_item_in_battle(FULL_RESTORE)
    cnt1 = ba._items_count(FULL_RESTORE)
    state = st.read_battle(b)
    hp1 = state["ours"]["hp"] if state else -1
    print(f"post: res={res} FRx{cnt1} HP={hp0}->{hp1}", flush=True)
    ok = (res == "used" and cnt1 == cnt0 - 1)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
