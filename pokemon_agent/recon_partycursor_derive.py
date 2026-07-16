"""recon_partycursor_derive.py — derive the in-battle PARTY-LIST cursor (and a party-open mode byte) by
RAM-diff. Enter a wild battle (route3_caught, 2-mon party), action menu -> POKEMON -> open party list,
then snapshot + DOWN (slot 0->1) to find the vertical cursor byte. So the in-battle SWITCH can nav by
readback (like the move list) instead of blind DOWN*slot taps that wedge on the long core."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import travel as tv                                                # noqa: E402
import firered_ram as ram                                          # noqa: E402
from battle_agent import BattleAgent                               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "route3_caught.state")
LO, HI = 0x02023C00, 0x02024400


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


def snap(b):
    return [b.rd8(a) for a in range(LO, HI)]


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
    print(f"party_count={b.rd8(ram.GPLAYER_PARTY_CNT)}")
    if not ag._settle_action_menu() or not ag._goto_pokemon():
        print("FAIL: couldn't reach POKEMON"); return
    action_snap = snap(b)
    b.press("A", 8, 8, owner="agent")          # open the party list
    for _ in range(20):
        b.run_frame()
    print(f"party screen open? white_box={ag._white_box()} (want False)")
    open_snap = bytes(b.save_state())
    party_open_diff = [(hex(LO + i), action_snap[i], b.rd8(LO + i))
                       for i in range(len(action_snap)) if action_snap[i] != b.rd8(LO + i)]
    print(f"ACTION-menu vs PARTY-open diff (mode-byte candidates): {party_open_diff}")
    # DOWN: party cursor slot 0 -> 1. Scan a WIDE EWRAM region (the party cursor isn't in the battle menu
    # block — likely the gPartyMenu struct). Double-press (eaten-press tolerant) + grab bytes that go ->1.
    WLO, WHI = 0x02020000, 0x02040000
    wbefore = {a: b.rd8(a) for a in range(WLO, WHI)}
    b.press("DOWN", 8, 6, owner="agent"); b.press("DOWN", 8, 10, owner="agent")
    for _ in range(12):
        b.run_frame()
    d_down = [(hex(a), wbefore[a], b.rd8(a)) for a in range(WLO, WHI)
              if wbefore[a] != b.rd8(a) and b.rd8(a) == 1]
    print(f"WIDE DOWN (slot ->1) candidates (->1): {d_down[:20]}")
    # UP back to 0 (confirm it tracks)
    b.load_state(open_snap)
    for _ in range(8):
        b.run_frame()
    b.set_input_owner("agent")
    cand = [int(a, 16) for a, _, nv in d_down]
    print(f"party-cursor candidates (changed ->1 on DOWN): {[hex(a) for a in cand]}")
    for a in cand:
        b.press("DOWN", 8, 6, owner="agent")
        for _ in range(8):
            b.run_frame()
        v1 = b.rd8(a)
        b.press("UP", 8, 6, owner="agent")
        for _ in range(8):
            b.run_frame()
        v0 = b.rd8(a)
        print(f"   {hex(a)}: after extra DOWN={v1} after UP={v0}  {'<-- TRACKS (0/1)' if v1 != v0 else ''}")


if __name__ == "__main__":
    main()
