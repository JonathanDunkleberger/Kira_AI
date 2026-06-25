"""recon_execflags.py - find gBattleControllerExecFlags + verify gChosenActionByBattler[0]=player, LIVE.

The action-array write alone won't advance the turn: the engine gates on the player CONTROLLER finishing
(gBattleControllerExecFlags — the player's bit set while choosing, cleared when done). That flag is
locatable live: a u32 whose bit0 is SET at the action menu and CLEARS the instant the player's action
commits. Also verifies 0x02023DC0 (the action-array candidate) reads 0xFF at the menu (player unchosen)
then -> a small action value after the engine commits the move. Read-only.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
START = os.path.join(_HERE, "states", "seg_route3_start.state")
LO, HI = 0x02023A00, 0x02024100
ACT_ARR = 0x02023DC0


def main():
    b = Bridge(ROM)
    with open(START, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    out = {}

    def battle_runner():
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
        for _ in range(160):
            if b.rd8(ram.GBATTLE_MENU_UP) == 1:
                break
            b.press("B", 8, 8, owner="agent")
        ag._settle()
        # at the action menu (player choosing): snapshot, record u32s with bit0 set + the action array
        out["actarr_at_menu"] = [b.rd8(ACT_ARR + i) for i in range(4)]
        menu_u32 = {a: b.rd32(a) for a in range(LO, HI, 4) if b.rd32(a) & 1}
        # commit ONE player action with a per-frame hook that catches the FIRST clear of bit0
        cleared = {}
        actarr_after = {}

        def hook():
            for a, v0 in menu_u32.items():
                if a not in cleared and not (b.rd32(a) & 1):
                    cleared[a] = (v0, b.rd32(a))
            for i in range(4):
                if i not in actarr_after and b.rd8(ACT_ARR + i) != 0xFF:
                    actarr_after[i] = b.rd8(ACT_ARR + i)
        ag.render = hook
        ag.run(max_seconds=40)
        # exec-flags candidates: u32 that had bit0 set at menu and cleared during the turn
        out["execflag_cands"] = {hex(a): (hex(v0), hex(v1)) for a, (v0, v1) in cleared.items()}
        out["actarr_after"] = {i: actarr_after.get(i) for i in range(4)}
        return "loss"

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None, render=lambda: None)
    camp._suppress_heal = True
    camp.trav.travel(target_map=(3, 22), edge="north", max_steps=200, max_seconds=120)
    print(f"   [exec] action array 0x02023DC0[0..3] at menu = {out.get('actarr_at_menu')} "
          f"(expect 0xFF=unchosen for player)", flush=True)
    print(f"   [exec] action array after commit = {out.get('actarr_after')} "
          f"(player idx -> 0=USE_MOVE)", flush=True)
    print(f"   [exec] gBattleControllerExecFlags candidates (bit0 set@menu -> cleared on commit):",
          flush=True)
    for a, (v0, v1) in out.get("execflag_cands", {}).items():
        print(f"   [exec]     {a}: {v0} -> {v1}", flush=True)


if __name__ == "__main__":
    main()
