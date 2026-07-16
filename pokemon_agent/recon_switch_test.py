"""recon_switch_test.py — prove a READBACK-based in-battle SHIFT-confirm actually swaps the active mon.
Boot route3_caught (2-mon party), enter a wild battle, drive action->POKEMON->list->slot1, then run the
NEW confirm: verify the sub-menu is open (cursor sprite-Y in {2,18,34}), drive it to SHIFT (Y==2) by
readback UP, A to confirm, advance the swap text until the active SPECIES flips. Reports SWITCHED / not.
This is the isolated proof for the _switch_to_slot rewrite (long-core eaten-press tolerant)."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import travel as tv                                                # noqa: E402
import firered_ram as ram                                          # noqa: E402
from battle_agent import BattleAgent, PARTY_CURSOR                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "route3_caught.state")

SUBMENU_CURSOR = 0x02020017      # sprite-Y of the ▶ in "Do what with X?": SHIFT=2 SUMMARY=18 CANCEL=34
SM_SHIFT_Y = 2


class _Hit(Exception):
    pass


def enter_battle(b):
    grass = sorted((x - 7, y - 7) for (x, y) in tv.Grid(b).grass)

    def runner():
        raise _Hit()
    trav = tv.Traveler(b, battle_runner=runner, render=lambda: None, log=lambda m: None, owner="agent")
    try:
        trav.travel(target_map=None, arrive_coord=grass[len(grass) // 2], max_steps=200, max_seconds=120)
    except _Hit:
        pass
    return st.in_battle(b)


def press(b, key, n=12):
    b.press(key, 8, 8, owner="agent")
    for _ in range(n):
        b.run_frame()


def submenu_open(b):
    return b.rd8(SUBMENU_CURSOR) in (2, 18, 34)


def do_switch(b, ag, slot, before_sp, log):
    """Simple SHIFT-confirm (sub-menu opens on SHIFT by default). Instrument PARTY_CURSOR + sprite-Y at
    each step to find a clean 'sub-menu opened' signal. Returns 'switched' or a failure string."""
    if not ag._settle_action_menu() or not ag._goto_pokemon():
        return "no_pokemon_menu"
    press(b, "A", 16)                                             # open party list
    try:
        b.frame_rgb().save(os.path.join(_HERE, "states", "switch_list0.png"))
    except Exception:
        pass
    log(f"   LIST @slot0 baseline: party_cursor={b.rd8(PARTY_CURSOR)}")
    # NAV to slot 1 with RIGHT (in-battle party menu is 1 big card + a right column; slot0->1 = RIGHT)
    press(b, "RIGHT", 8)
    for _ in range(10):
        b.run_frame()
    log(f"   after RIGHT: party_cursor={b.rd8(PARTY_CURSOR)}")
    try:
        b.frame_rgb().save(os.path.join(_HERE, "states", "switch_right.png"))
    except Exception:
        pass
    SUBMENU_OPEN = 0x2020521                                     # 0 = list, 1 = "Do what with X?" sub-menu up
    log(f"   LIST @slot{slot}: party_cursor={b.rd8(PARTY_CURSOR)} f521={b.rd8(SUBMENU_OPEN)}")
    # OPEN the sub-menu (A#1), verify f521->1, eaten-press tolerant (settle between presses)
    for _ in range(5):
        press(b, "A", 8)
        for _ in range(14):
            b.run_frame()
        if b.rd8(SUBMENU_OPEN) == 1:
            break
    if b.rd8(SUBMENU_OPEN) != 1:
        return f"submenu_didnt_open (f521={b.rd8(SUBMENU_OPEN)})"
    # let it become input-ready + read the sub-menu cursor (defaults to SHIFT=2)
    for _ in range(10):
        b.run_frame()
    log(f"   sub-menu OPEN: f521=1 smY={b.rd8(SUBMENU_CURSOR)} (want 2=SHIFT)")
    for _ in range(4):                                          # ensure on SHIFT (top) by readback UP
        if b.rd8(SUBMENU_CURSOR) == SM_SHIFT_Y:
            break
        press(b, "UP", 8)
        for _ in range(8):
            b.run_frame()
    # CONFIRM SHIFT (A#2) + advance the swap text until the ACTIVE species flips
    press(b, "A", 8)
    for _ in range(12):
        b.run_frame()
    try:
        b.frame_rgb().save(os.path.join(_HERE, "states", "switch_confirm.png"))
    except Exception:
        pass
    log(f"   after CONFIRM A: f521={b.rd8(SUBMENU_OPEN)} smY={b.rd8(SUBMENU_CURSOR)} species={st.read_battle(b)['ours']['species'] if st.read_battle(b) else '?'}")
    for i in range(16):
        for _ in range(10):
            b.run_frame()
        cur = st.read_battle(b)
        if cur and cur["ours"]["hp"] > 0 and cur["ours"].get("species") != before_sp:
            return "switched"
        if b.rd8(SUBMENU_OPEN) == 1 and i > 2:                  # sub-menu still up, no flip -> re-confirm
            press(b, "A", 8)
        else:
            press(b, "A", 8)
    return f"no_flip (species still {before_sp}, f521={b.rd8(SUBMENU_OPEN)})"


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    if not enter_battle(b):
        print("FAIL: no battle"); return
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    ag._reach_first_menu(time.time(), 30)
    ag._settle()
    b.set_input_owner("agent")
    cur = st.read_battle(b)
    before_sp = cur["ours"]["species"] if cur else None
    before_name = st.SPECIES_NAME.get(before_sp, "?")
    print(f"BEFORE: active species={before_sp} ({before_name}); party_count={b.rd8(ram.GPLAYER_PARTY_CNT)}")
    # slot-1 reserve HP (is it even switchable?)
    s1_sp = st.read_party_species(b, 1)
    s1_hp = b.rd16(ram.GPLAYER_PARTY + 1 * 100 + 0x56)
    print(f"slot1 reserve: species={s1_sp} ({st.SPECIES_NAME.get(s1_sp,'?')}) HP={s1_hp}")
    r = do_switch(b, ag, 1, before_sp, print)
    after = st.read_battle(b)
    after_sp = after["ours"]["species"] if after else None
    try:
        b.frame_rgb().save(os.path.join(_HERE, "states", "switch_after.png"))
        print("saved states/switch_after.png")
    except Exception as e:
        print(f"screenshot failed: {e}")
    print(f"RESULT: {r} | active species now={after_sp} ({st.SPECIES_NAME.get(after_sp,'?')})")
    print("PASS" if r == "switched" and after_sp != before_sp else "FAIL")


if __name__ == "__main__":
    main()
