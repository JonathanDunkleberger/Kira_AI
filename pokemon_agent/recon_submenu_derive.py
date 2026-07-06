"""recon_submenu_derive.py — derive the in-battle party SUB-MENU cursor ("Do what with X?" SHIFT/SUMMARY/
CANCEL) so the switch can confirm SHIFT by readback, not blind A-spam. Boot route3_caught (2-mon party),
enter a wild battle, action->POKEMON->open list->nav to slot1->A (select) -> sub-menu up. Screenshot it,
then RAM-diff a WIDE EWRAM region while pressing DOWN (SHIFT->SUMMARY->CANCEL) to find the sub-menu index
byte. Read-only w.r.t. saves (no writes to canonical)."""
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
OUT = os.path.join(_HERE, "states", "submenu.png")
WLO, WHI = 0x02020000, 0x02021000                                 # gPartyMenu region (PARTY_CURSOR=0x2020777)


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


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"party_count={b.rd8(ram.GPLAYER_PARTY_CNT)}")
    if not enter_battle(b):
        print("FAIL: no battle"); return
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    ag._reach_first_menu(time.time(), 30)
    ag._settle()
    b.set_input_owner("agent")
    if not ag._settle_action_menu() or not ag._goto_pokemon():
        print("FAIL: couldn't reach POKEMON"); return
    b.press("A", 8, 8, owner="agent")                             # open party list
    for _ in range(20):
        b.run_frame()
    print(f"party list open? white_box={ag._white_box()} (want False) party_cursor={b.rd8(PARTY_CURSOR)}")
    # nav to slot 1 (the non-active reserve) by readback, then SELECT it
    ag._goto_party_slot(1)
    print(f"after goto slot1: party_cursor={b.rd8(PARTY_CURSOR)}")
    before_sel = {a: b.rd8(a) for a in range(WLO, WHI)}
    b.press("A", 8, 10, owner="agent")                            # select -> "Do what with X?" sub-menu
    for _ in range(24):
        b.run_frame()
    # screenshot the sub-menu so a human (me) can LOOK at it
    try:
        b.frame_rgb().save(OUT)
        print(f"saved sub-menu frame -> {OUT}")
    except Exception as e:
        print(f"screenshot failed: {e}")
    sel_diff = [(hex(a), before_sel[a], b.rd8(a)) for a in range(WLO, WHI) if before_sel[a] != b.rd8(a)]
    print(f"SELECT diff (party-list -> sub-menu open), {len(sel_diff)} bytes: {sel_diff[:30]}")
    # now DOWN through the sub-menu (SHIFT->SUMMARY->CANCEL): find the index byte that increments
    open_snap = bytes(b.save_state())
    before_down = {a: b.rd8(a) for a in range(WLO, WHI)}
    b.press("DOWN", 8, 10, owner="agent")
    for _ in range(12):
        b.run_frame()
    down_diff = [(hex(a), before_down[a], b.rd8(a)) for a in range(WLO, WHI) if before_down[a] != b.rd8(a)]
    print(f"DOWN-in-submenu diff ({len(down_diff)} bytes): {down_diff[:30]}")
    # confirm each candidate tracks: reload, DOWN twice vs UP
    cand = [int(a, 16) for a, _, _ in down_diff]
    b.load_state(open_snap)
    for _ in range(8):
        b.run_frame()
    b.set_input_owner("agent")
    for a in cand[:12]:
        b.load_state(open_snap)
        for _ in range(6):
            b.run_frame()
        b.set_input_owner("agent")
        v0 = b.rd8(a)
        b.press("DOWN", 8, 8, owner="agent")
        for _ in range(8):
            b.run_frame()
        v1 = b.rd8(a)
        b.press("DOWN", 8, 8, owner="agent")
        for _ in range(8):
            b.run_frame()
        v2 = b.rd8(a)
        print(f"   {hex(a)}: open={v0} DOWN={v1} DOWN={v2}  {'<-- SUBMENU CURSOR (0/1/2)' if (v0,v1,v2)==(0,1,2) else ''}")


if __name__ == "__main__":
    main()
