"""recon_movecursor_derive.py — derive/verify the FRLG MOVE-LIST cursor RAM address. Drives route3_caught
into a wild battle (FRESH core = clean actuation), opens the FIGHT move list, then RAM-diffs the battle
EWRAM region after DOWN (slot 0->2) and RIGHT (slot 0->1) to find the byte(s) tracking the 2x2 cursor.
Saves the move-list-open state to states/movelist_open.state for re-use by the readback build."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import travel as tv                                                # noqa: E402
from battle_agent import BattleAgent                               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "route3_caught.state")
if not os.path.exists(SRC):
    SRC = os.path.join(_HERE, "states", "route3_caught.state")
OPEN = os.path.join(_HERE, "states", "movelist_open.state")
LO, HI = 0x02023C00, 0x02024400


class _Hit(Exception):
    pass


def walk_into_battle(b):
    # Use the proven Traveler to walk toward grass; on the FIRST warmed-up encounter its runner raises
    # so we STOP travel while still in-battle (the menu is reachable). Reliable (the manual sweep just
    # turned in place).
    grass = sorted((x - 7, y - 7) for (x, y) in tv.Grid(b).grass)
    if not grass:
        return False
    target = grass[len(grass) // 2]

    def runner():
        raise _Hit()
    trav = tv.Traveler(b, battle_runner=runner, render=lambda: None, log=lambda m: None, owner="agent")
    try:
        trav.travel(target_map=None, arrive_coord=target, max_steps=200, max_seconds=120)
    except _Hit:
        pass
    return st.in_battle(b)


def open_move_list(b, ag):
    t0 = time.time()
    ag._reach_first_menu(t0, 30)
    ag._settle()
    b.set_input_owner("agent")
    # snapshot the ACTION menu (FIGHT/BAG/POKEMON/RUN) before opening FIGHT, to diff for the "mode" byte
    ag._home_to_fight(); ag._wait(6)
    global _ACTION_SNAP
    _ACTION_SNAP = [b.rd8(a) for a in range(LO, HI)]
    for _ in range(12):
        ag._home_to_fight()
        ag._tap("A"); ag._wait(8)
        if ag._in_move_list():
            return True
        if not ag._white_box():
            ag._tap("B"); ag._wait(8)
    return ag._in_move_list()


_ACTION_SNAP = None


def snap(b):
    return [b.rd8(a) for a in range(LO, HI)]


def diff(before, b, tag):
    out = [(hex(LO + i), before[i], b.rd8(LO + i)) for i in range(len(before)) if before[i] != b.rd8(LO + i)]
    print(f"   [{tag}] changed: {out}", flush=True)
    return out


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    if not walk_into_battle(b):
        print("could not enter a wild battle"); return
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    if not open_move_list(b, ag):
        print(f"could not open the move list (in_move_list={ag._in_move_list()})"); return
    s = st.read_battle(b)
    moves = [m["name"] for m in s["ours"]["moves"]] if s else []
    print(f"move list OPEN. moves={moves} candidate 0x02024005={b.rd8(0x02024005)}", flush=True)
    with open(OPEN, "wb") as f:
        f.write(bytes(b.save_state()))
    opensnap = bytes(b.save_state())

    # DOWN: slot 0 -> 2 (bottom-left in the 2x2 grid)
    before = snap(b)
    b.press("DOWN", 8, 6, owner="agent")
    for _ in range(10):
        b.run_frame()
    d_down = diff(before, b, "DOWN 0->2")
    # reload + RIGHT: slot 0 -> 1 (top-right)
    b.load_state(opensnap)
    for _ in range(8):
        b.run_frame()
    b.set_input_owner("agent")
    before = snap(b)
    b.press("RIGHT", 8, 6, owner="agent")
    for _ in range(10):
        b.run_frame()
    d_right = diff(before, b, "RIGHT 0->1")
    # The cursor byte = one that reads 2 after DOWN AND 1 after RIGHT (single 0..3 index), OR a row byte
    # (DOWN only) + col byte (RIGHT only).
    down_addrs = {a for a, _, nv in d_down if nv in (1, 2, 3)}
    right_addrs = {a for a, _, nv in d_right if nv in (1, 2, 3)}
    print(f"\n   single-index candidates (changed by BOTH, 0..3): {sorted(down_addrs & right_addrs)}")
    print(f"   DOWN-only (row?) candidates: {sorted(down_addrs - right_addrs)}")
    print(f"   RIGHT-only (col?) candidates: {sorted(right_addrs - down_addrs)}")
    # OPEN-SIGNAL: diff the action-menu snapshot vs the move-list-open snapshot to find a 'mode' byte
    b.load_state(opensnap)
    for _ in range(8):
        b.run_frame()
    if _ACTION_SNAP is not None:
        mode = [(hex(LO + i), _ACTION_SNAP[i], b.rd8(LO + i))
                for i in range(len(_ACTION_SNAP)) if _ACTION_SNAP[i] != b.rd8(LO + i)]
        print(f"\n   ACTION-menu vs MOVE-LIST-open diff (mode-byte candidates): {mode}")


if __name__ == "__main__":
    main()
