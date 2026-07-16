"""recon_inject.py - PROVE the catch throw via 2-stage controller injection on a DISPOSABLE live battle.

Addresses (all verified live, read-only): gBattleBufferB[0]=0x02023418, gBattleControllerExecFlags=
0x02023BC8 (bit0=player), gChosenActionByBattler=0x02023DC0. The engine, when the player's exec bit
clears, reads the action from gBattleBufferB[0][1] (and an item the same way). So:
  STAGE 1 (action): buffer[0][0]=1(CONTROLLER_TWORETURNVALUES), buffer[0][1]=1(B_ACTION_USE_ITEM),
                    clear exec bit0 -> engine branches to the item handler (re-sets exec).
  STAGE 2 (item):   buffer[0][1]=4(POKE_BALL low), buffer[0][2]=0(high), clear exec bit0 -> ball thrown.
Caught = party N->N+1. Disposable battle: safe to observe corruption. Read+write here is the sandbox.
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
BUFB0 = 0x020233C4          # real gBattleBufferB[0] (.sym-validated; 0x02023418 was a lookalike)
EXEC = 0x02023BC8           # gBattleControllerExecFlags
GCHOSEN = 0x02023D7C        # real gChosenActionByBattler (.sym-validated)
ACTARR = GCHOSEN
PARTYCNT = ram.GPLAYER_PARTY_CNT


def main():
    b = Bridge(ROM)
    with open(START, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w8 = b.core.memory.u8.raw_write
    w32 = b.core.memory.u32.raw_write
    log = []

    def clear_player_execbit():
        w32(EXEC, b.rd32(EXEC) & ~1)

    def battle_runner():
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
        for _ in range(160):
            if b.rd8(ram.GBATTLE_MENU_UP) == 1:
                break
            b.press("B", 8, 8, owner="agent")
        ag._settle()
        p0 = b.rd8(PARTYCNT)
        enemy0 = st.read_battle(b)
        ehp0 = enemy0["enemy"]["hp"] if enemy0 else None
        log.append(f"menu: party={p0} exec={b.rd32(EXEC):#x} actarr0={b.rd8(ACTARR)} enemyHP={ehp0}")
        # ── let the ENGINE confirm FIGHT (reliable live), and on the exec-flag 1->0 edge SWAP the
        # emitted action USE_MOVE(0)->USE_ITEM(1) in the buffer so the engine consumes USE_ITEM. ──
        # BALL THROW — FULL input control (no ag2 contention). Per turn:
        #  (1) ACTION phase: press A on FIGHT while HOLDING gBattleBufferB[0][1]=USE_ITEM -> the action
        #      emit is swapped to USE_ITEM -> engine opens the bag (chosen0 -> 1).
        #  (2) ITEM phase: press B to close the bag while FORCING gSpecialVar_ItemId=POKE_BALL -> the
        #      bag-close completion emits the Poké Ball -> engine throws it. Retry on break-out.
        # BALL THROW — SKIP THE BAG via controller-func override (.sym-sourced addresses):
        #  ag2 confirms FIGHT; the hook HOLDS gBattleBufferB[0][1]=USE_ITEM so the engine consumes it
        #  (chosen0->1) and sets gBattlerControllerFuncs[0]=OpenBagAndChooseItem. We then OVERRIDE that
        #  func -> CompleteWhenChoseItem (Thumb|1) with gSpecialVar_ItemId=ITEM, so it emits the ball
        #  with NO bag opened -> engine throws. Control: ITEM=4 throws (party N->N+1 on a weak foe);
        #  ITEM=0 (no item) must NOT throw — distinguishable.
        BASE = int(os.getenv("INJECT_BASE", "0x020233C4"), 0)
        GSPEC = int(os.getenv("INJECT_GSPEC", "0x0203AD30"), 0)
        ITEM = int(os.getenv("INJECT_ITEM", "4"))
        GBCF = 0x03004FE0                                  # gBattlerControllerFuncs[0]
        COMPLETE = 0x0803073C | 1                          # CompleteWhenChoseItem | Thumb bit
        ENEMY_HP = ram.GBATTLE_MONS + ram.GBATTLE_MON_SIZE + 0x28   # gBattleMons[1] current HP (u16)
        WEAKEN_HP = int(os.getenv("WEAKEN_HP", "1"))       # RAM-weaken the foe (sandbox) so a throw catches; 0=off
        w16 = b.core.memory.u16.raw_write
        w32 = b.core.memory.u32.raw_write
        state = {"overridden": False, "throws": 0, "consumed": 0}

        def swap_hook():
            chosen0 = b.rd8(GCHOSEN)
            if chosen0 != 1:                               # ACTION phase: hold USE_ITEM; arm next turn
                w8(BASE + 1, 1)
                if WEAKEN_HP > 0 and b.rd16(ENEMY_HP) > WEAKEN_HP:
                    w16(ENEMY_HP, WEAKEN_HP)               # weaken-not-faint (keep it alive at low HP)
                state["overridden"] = False
            else:                                          # ITEM phase: skip bag, emit the ball
                w16(GSPEC, ITEM)                           # chosen item (CompleteWhenChoseItem emits this)
                w16(0x02023D68, ITEM)                      # gLastUsedItem = the ball (throw script reads this)
                if not state["overridden"]:               # override the func ONCE (re-writing it every
                    w32(GBCF, COMPLETE)                    #   frame would loop CompleteWhenChoseItem and
                    state["consumed"] += 1                 #   block PlayerBufferExecCompleted)
                    state["throws"] += 1
                    state["overridden"] = True
        ag2 = BattleAgent(b, on_event=lambda *a, **k: None, render=swap_hook, log=lambda m: None)
        ag2.run(max_seconds=80)
        p1 = b.rd8(PARTYCNT)
        log.append(f"THROW(func-skip) ITEM={ITEM}: useitem_consumed={state['consumed']} "
                   f"party {p0}->{p1} in_battle={st.in_battle(b)} "
                   f"{'*** CAUGHT! party+1 ***' if p1 > p0 else '(no catch)'}")
        return "loss"

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None, render=lambda: None)
    camp._suppress_heal = True
    try:
        camp.trav.travel(target_map=(3, 22), edge="north", max_steps=200, max_seconds=120)
    except Exception as e:
        log.append(f"EXC: {e}")
    for ln in log:
        print(f"   [inj] {ln}", flush=True)


if __name__ == "__main__":
    main()
