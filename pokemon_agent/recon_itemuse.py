"""recon_itemuse.py - LIVE recon of IN-BATTLE item use (PART B). No use-item primitive exists; this
maps the bag -> Items pocket -> select Potion -> USE -> use-on-target flow on a REAL wild battle
(loaded-mid-battle states desync the menu input path, so we recon on the live leg).

Setup: boot Route 3 (route3_caught), INJECT a few Potions into the Items pocket + set the lead LOW so a
Potion has a visible effect, then wander grass into a wild encounter and probe the item-use menu with
screenshots + the pocket/cursor signals throw_ball already relies on (GBAG_POCKET, white-panel pixels).

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_itemuse.py
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
from battle_agent import BattleAgent                 # noqa: E402
from campaign import Campaign, resolve_state, ITEM_POTION, P_HP, P_MAXHP   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SHOT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\56687edb-aec7-4d80-be7e-8134df395075\scratchpad"


def log(m):
    print(f"   [itemuse-recon] {m}", flush=True)


def shot(b, name):
    try:
        b.frame_rgb().save(os.path.join(SHOT, name)); log(f"shot -> {name}")
    except Exception as e:
        log(f"shot {name} failed: {e}")


def key16(b):
    return b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF


def bag_count(b, item_id):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR); k = key16(b)
    for s in range(42):
        slot = sb1 + 0x0310 + s * 4
        if b.rd16(slot) == item_id:
            return b.rd16(slot + 2) ^ k
    return 0


def inject_potions(b, n=3):
    """Write n Potions into the first empty Items-pocket slot (id 13, qty^key)."""
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR); k = key16(b)
    for s in range(42):
        slot = sb1 + 0x0310 + s * 4
        iid = b.rd16(slot)
        if iid == ITEM_POTION:
            b.core.memory.u16.raw_write(slot + 2, n ^ k); return
        if iid == 0:
            b.core.memory.u16.raw_write(slot, ITEM_POTION)
            b.core.memory.u16.raw_write(slot + 2, n ^ k); return


def set_lead_hp(b, hp):
    b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + P_HP, hp)


def main():
    b = Bridge(ROM)
    with open(resolve_state("route3_caught.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    b.set_input_owner("agent")
    inject_potions(b, 3)
    mx = b.rd16(ram.GPLAYER_PARTY + P_MAXHP)
    set_lead_hp(b, max(1, mx // 6))               # low but alive so a Potion (+20) visibly heals
    log(f"booted route3_caught: map={tv.map_id(b)} potions={bag_count(b, ITEM_POTION)} "
        f"leadHP={b.rd16(ram.GPLAYER_PARTY + P_HP)}/{mx}")

    # wander grass until a WILD encounter — REUSE the proven Traveler (BFS, grass-aware), breaking into
    # the LIVE battle via a sentinel battle_runner that raises the instant travel hands off an encounter.
    off = tv.MAP_OFFSET
    grass = sorted({(x - off, y - off) for (x, y) in tv.Grid(b).grass})
    cur0 = tv.coords(b)
    grass.sort(key=lambda t: abs(t[0] - cur0[0]) + abs(t[1] - cur0[1]))
    log(f"{len(grass)} grass tiles; wandering (Traveler) for a wild encounter...")

    class _Wild(Exception):
        pass

    def stop_at_battle():
        raise _Wild()
    camp = Campaign(b, battle_runner=stop_at_battle, on_event=lambda s, **k: None,
                    render=lambda: None)
    camp.trav.battle_runner = stop_at_battle
    waypoints = [grass[0], grass[-1], grass[len(grass) // 2]]
    try:
        for wp in waypoints * 6:
            if st.in_battle(b):
                break
            camp.trav.travel(target_map=None, arrive_coord=wp, max_steps=120, max_seconds=60)
    except _Wild:
        pass
    if not st.in_battle(b):
        log("!! no wild encounter — re-run"); return
    log("WILD encounter (in battle)")

    # SETTLE to the real ACTION menu using the PIXEL signals (menu_up RAM is stale here): the action
    # menu is white_box AND NOT _in_move_list (pixel 160,150 is dark only over the move names). Back out
    # of the move list with B; advance blue text otherwise.
    ba = BattleAgent(b, on_event=lambda s, **k: None, render=lambda: None, log=lambda m: None)
    import time
    ba._reach_first_menu(time.time(), 60)
    ba._settle()
    for _ in range(30):
        if ba._white_box() and not ba._in_move_list():
            break
        if ba._in_move_list():
            b.press("B", 8, 10, lambda: None, owner="agent")
            for _ in range(12):
                b.run_frame()
        elif not ba._white_box():
            ba._advance_text()
        else:
            break
    shot(b, "iu_01_actionmenu.png")
    log(f"action menu: white_box={ba._white_box()} in_move={ba._in_move_list()} "
        f"menu_up={b.rd8(ram.GBATTLE_MENU_UP)} pocket={b.rd8(ram.GBAG_POCKET)}")

    # LIVE CONTROL of the PRIMITIVE: drive BattleAgent.use_item_in_battle(Potion) end to end and assert
    # the heal landed (HP rose) + a Potion was consumed + it returned 'used'. This is the real proof the
    # whole bag->Items->select->USE->target flow actuates on the live core (loaded battles desync menus).
    hp0, pot0 = b.rd16(ram.GPLAYER_PARTY + P_HP), bag_count(b, ITEM_POTION)
    res = ba.use_item_in_battle(ITEM_POTION)
    hp1, pot1 = b.rd16(ram.GPLAYER_PARTY + P_HP), bag_count(b, ITEM_POTION)
    shot(b, "iu_after_use.png")
    log(f"use_item_in_battle -> {res!r}; HP {hp0}->{hp1}  potions {pot0}->{pot1}  in_battle={st.in_battle(b)}")
    ok = (res == "used" and hp1 > hp0 and pot1 == pot0 - 1)
    log(f"PART B LIVE PRIMITIVE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
