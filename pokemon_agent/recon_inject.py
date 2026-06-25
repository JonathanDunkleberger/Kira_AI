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
BUFB0 = 0x02023418
EXEC = 0x02023BC8
ACTARR = 0x02023DC0
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
        # TWO-EDGE BALL THROW (only known addresses): both the chosen ACTION and the chosen ITEM flow
        # through gBattleBufferB[0][1] on the controller's completion edge. So:
        #   edge A (action controller done): swap -> USE_ITEM(1)  -> engine opens the bag controller
        #   the bag is canceled (B) -> edge B (item controller done): swap -> POKE_BALL(4) -> ball thrown
        # Repeat per turn (retry on break-out). Caterpie has a high catch rate, so retries catch it.
        BASE = int(os.getenv("INJECT_BASE", "0x020233C4"), 0)
        ITEM = int(os.getenv("INJECT_ITEM", "4"))          # ITEM_POKE_BALL
        state = {"prev": b.rd32(EXEC) & 1, "phase": "action", "edges": 0, "throws": 0, "bag_b": 0}

        def swap_hook():
            cur = b.rd32(EXEC) & 1
            if state["prev"] == 1 and cur == 0:            # a controller just completed
                state["edges"] += 1
                if state["phase"] == "action":
                    w8(BASE + 1, 1)                        # -> USE_ITEM
                    state["phase"] = "item"
                else:
                    w8(BASE + 1, ITEM); w8(BASE + 2, 0)    # -> POKE_BALL
                    state["throws"] += 1
                    state["phase"] = "action"
            state["prev"] = cur
            # while the item controller is up (bag open after USE_ITEM), tap B to cancel it -> edge B
            if state["phase"] == "item" and cur == 1:
                state["bag_b"] += 1
                if state["bag_b"] % 6 == 0:
                    b.set_keys("B", owner="agent")
        ag2 = BattleAgent(b, on_event=lambda *a, **k: None, render=swap_hook, log=lambda m: None)
        ag2.run(max_seconds=70)
        p1 = b.rd8(PARTYCNT)
        log.append(f"TWO-EDGE throw BASE={hex(BASE)} ITEM={ITEM}: edges={state['edges']} "
                   f"throws={state['throws']} party {p0}->{p1} in_battle={st.in_battle(b)} "
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
