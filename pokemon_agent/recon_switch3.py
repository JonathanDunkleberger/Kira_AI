"""recon_switch3.py — BLIND voluntary switch (mirror the WORKING _force_switch: DOWN*slot + A + A), on the
proper 3-mon fixture (Ivysaur/Rattata/Spearow). Tests switching the active Ivysaur (slot0) to Spearow
(slot2): open list -> DOWN*2 -> A (select) -> A (SHIFT) -> advance -> check gBattleMons[0] species flips.
No cursor-readback (the live cursor is in a heap struct). This is the approach the faint-switch already uses."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import firered_ram as ram                                          # noqa: E402
from battle_agent import BattleAgent                               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "canon_battle.state")


def settle(b, n):
    for _ in range(n):
        b.run_frame()


def tapk(b, key, after=16):
    b.press(key, 6, 6, owner="agent")
    settle(b, after)


def try_switch(b, ag, downs, before_sp, tag):
    """open list -> DOWN*downs -> A select -> A SHIFT -> advance; return (flipped, after_sp)."""
    if not ag._settle_action_menu() or not ag._goto_pokemon():
        return (False, "no_pokemon_menu")
    tapk(b, "A", after=45)                                        # open party list + SETTLE WELL (no eaten 1st)
    for _ in range(downs):
        tapk(b, "DOWN", after=18)
    tapk(b, "A", after=22)                                        # select the mon -> sub-menu
    tapk(b, "A", after=22)                                        # confirm SHIFT (default top)
    for _ in range(14):
        cur = st.read_battle(b)
        if cur and cur["ours"]["hp"] > 0 and cur["ours"]["species"] != before_sp:
            return (True, cur["ours"]["species"])
        tapk(b, "A", after=12)
    b.frame_rgb().save(os.path.join(_HERE, "states", f"s3_{tag}.png"))
    return (False, (st.read_battle(b) or {}).get("ours", {}).get("species"))


def main():
    for downs in (1, 2, 3, 4):
        b = Bridge(ROM)
        b.load_state(open(SRC, "rb").read())
        settle(b, 20)
        b.set_input_owner("agent")
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
        ag._reach_first_menu(time.time(), 30)
        ag._settle()
        b.set_input_owner("agent")
        cur = st.read_battle(b)
        before = cur["ours"]["species"]
        flipped, after = try_switch(b, ag, downs, before, f"down{downs}")
        print(f"DOWN*{downs}: before={st.SPECIES_NAME.get(before,'?')} "
              f"flipped={flipped} after={st.SPECIES_NAME.get(after,after)}")


if __name__ == "__main__":
    main()
