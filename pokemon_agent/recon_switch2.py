"""recon_switch2.py — find the TRUE in-battle party-selection cursor + test a real SHIFT switch.
Phase 1: open the in-battle party list, SETTLE WELL, wide RAM snapshot, screenshot; press DOWN, settle,
diff -> candidate cursor bytes (the real one tracks 0->1). Phase 2: from there open the sub-menu + confirm
SHIFT, check gBattleMons[0] species flip. Generous settles (test the 'nav press eaten by open-transition'
hypothesis). Read-only vs canonical saves."""
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
SRC = os.path.join(_HERE, "states", "workshop", "canon_battle.state")
WLO, WHI = 0x02020000, 0x02021000
SUBMENU_OPEN = 0x02020521
SUBMENU_CURSOR = 0x02020017


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


def settle(b, n):
    for _ in range(n):
        b.run_frame()


def tap(b, key, hold=6, after=18):
    b.press(key, hold, hold, owner="agent")
    settle(b, after)


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    settle(b, 20)
    b.set_input_owner("agent")
    if not enter_battle(b):
        print("FAIL: no battle"); return
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    ag._reach_first_menu(time.time(), 30)
    ag._settle()
    b.set_input_owner("agent")
    before_sp = st.read_battle(b)["ours"]["species"]
    print(f"active species={before_sp} ({st.SPECIES_NAME.get(before_sp,'?')}); "
          f"party_count={b.rd8(ram.GPLAYER_PARTY_CNT)}")
    if not ag._settle_action_menu() or not ag._goto_pokemon():
        print("FAIL: no POKEMON"); return
    tap(b, "A", after=40)                                          # open party list + SETTLE WELL
    b.frame_rgb().save(os.path.join(_HERE, "states", "s2_list0.png"))
    snapA = {a: b.rd8(a) for a in range(WLO, WHI)}
    print(f"LIST settled: 0x777={b.rd8(PARTY_CURSOR)} f521={b.rd8(SUBMENU_OPEN)} smY={b.rd8(SUBMENU_CURSOR)}")
    # single DOWN, wide diff — find the byte that tracks the REAL cursor (0->1 on a clean move)
    tap(b, "DOWN", after=30)
    b.frame_rgb().save(os.path.join(_HERE, "states", "s2_list1.png"))
    diff = [(hex(a), snapA[a], b.rd8(a)) for a in range(WLO, WHI) if snapA[a] != b.rd8(a)]
    print(f"DOWN changed {len(diff)} bytes; slot-index-ish (both <=6): "
          f"{[(a, o, n) for a, o, n in diff if o <= 6 and n <= 6][:16]}")
    print(f"   0x777 {snapA[PARTY_CURSOR]}->{b.rd8(PARTY_CURSOR)}")
    # Phase 2: open sub-menu + confirm SHIFT, check flip
    for _ in range(5):
        tap(b, "A", after=20)
        if b.rd8(SUBMENU_OPEN) == 1:
            break
    print(f"sub-menu open? f521={b.rd8(SUBMENU_OPEN)} smY={b.rd8(SUBMENU_CURSOR)}")
    for _ in range(4):
        if b.rd8(SUBMENU_CURSOR) == 2:
            break
        tap(b, "UP", after=12)
    tap(b, "A", after=24)
    flipped = False
    for _ in range(16):
        cur = st.read_battle(b)
        if cur and cur["ours"]["hp"] > 0 and cur["ours"]["species"] != before_sp:
            flipped = True
            break
        tap(b, "A", after=14)
    after_sp = (st.read_battle(b) or {}).get("ours", {}).get("species")
    b.frame_rgb().save(os.path.join(_HERE, "states", "s2_after.png"))
    print(f"RESULT: flipped={flipped} active now={after_sp} ({st.SPECIES_NAME.get(after_sp,'?')})")
    print("PASS" if flipped else "FAIL")


if __name__ == "__main__":
    main()
