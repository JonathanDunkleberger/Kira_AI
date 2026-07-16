"""recon_movecursor_verify.py — verify the move-list cursor readback: MENU_MODE (action=1 / movelist=2)
+ _goto_move RAM nav to each slot 0..3, then fight a full wild battle through the engine to confirm the
new readback path resolves turns cleanly. Uses route3_caught (FRESH core)."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import travel as tv                                                # noqa: E402
from battle_agent import BattleAgent, MOVE_CURSOR, MENU_MODE       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "route3_caught.state")


class _Hit(Exception):
    pass


def enter_battle(b):
    grass = sorted((x - 7, y - 7) for (x, y) in tv.Grid(b).grass)
    target = grass[len(grass) // 2]

    def runner():
        raise _Hit()
    trav = tv.Traveler(b, battle_runner=runner, render=lambda: None, log=lambda m: None, owner="agent")
    try:
        trav.travel(target_map=None, arrive_coord=target, max_steps=200, max_seconds=120)
    except _Hit:
        pass
    return st.in_battle(b)


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    if not enter_battle(b):
        print("FAIL: no battle"); return
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    t0 = time.time()
    ag._reach_first_menu(t0, 30)
    ag._settle()
    b.set_input_owner("agent")
    ag._home_to_fight(); ag._wait(6)
    print(f"action menu: MENU_MODE={b.rd8(MENU_MODE)} (expect 1)  movelist_open={ag._movelist_open()}")
    # open via the new RAM-corroborated loop
    opened = False
    for _ in range(12):
        if ag._movelist_open():
            opened = True; break
        ag._tap("A"); ag._wait(8)
    print(f"after open: MENU_MODE={b.rd8(MENU_MODE)} (expect 2)  movelist_open={ag._movelist_open()} opened={opened}")
    if not opened:
        print("FAIL: move list didn't open"); return
    # readback nav to each slot
    results = {}
    for slot in (0, 1, 2, 3, 2, 0):
        ok = ag._goto_move(slot)
        results[slot] = (ok, b.rd8(MOVE_CURSOR))
        print(f"   _goto_move({slot}) -> ok={ok} MOVE_CURSOR={b.rd8(MOVE_CURSOR)}")
    nav_ok = all(b_cur == slot for slot, (ok, b_cur) in
                 [(s, results[s]) for s in results])
    print(f"\nNAV all-slots-correct={nav_ok}")
    # CLEAN fight: enter a FRESH battle (no probe contamination) and fight via the engine's readback path
    if not enter_battle(b):
        print("no second battle to fight-test"); return
    ag2 = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                      log=lambda m: print("   [eng]", m))
    out = ag2.run(max_seconds=120)
    print(f"clean battle outcome={out!r} (expect 'win'/'ended', NOT 'stuck')")
    print("RESULT:", "PASS" if (nav_ok and out in ("win", "ended", "loss")) else "CHECK")


if __name__ == "__main__":
    main()
