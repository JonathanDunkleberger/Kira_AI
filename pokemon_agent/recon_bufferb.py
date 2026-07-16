"""recon_bufferb.py - locate gBattleBufferB LIVE (the last address for the catch-throw injection).

The engine copies gChosenActionByBattler[b] FROM gBattleBufferB[b][1] when the player's bit in
gBattleControllerExecFlags(0x02023BC8) clears. So at the commit frame, gBattleBufferB[0] = [1(=
CONTROLLER_TWORETURNVALUES), action, ...] and the enemy buffer [1] sits 0x200 later with the same
signature. Watch 0x02023BC8 for its 1->0 edge, snapshot EWRAM at that frame, and report (1,action)
pairs whose sibling is +/-0x200 away (the [4][0x200] buffer stride). Read-only.
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
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
START = os.path.join(_HERE, "states", "seg_route3_start.state")
EXEC = 0x02023BC8
LO, HI = 0x02022000, 0x02024800           # wide EWRAM battle region


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
        snap = {"prev": b.rd32(EXEC) & 1, "blk": None}

        def hook():
            cur = b.rd32(EXEC) & 1
            if snap["blk"] is None and snap["prev"] == 1 and cur == 0:   # the commit edge
                snap["blk"] = bytes(b.read_bytes(LO, HI - LO))
            snap["prev"] = cur
        ag.render = hook
        ag.run(max_seconds=40)
        blk = snap["blk"]
        if blk is None:
            out["err"] = "never caught the exec-flag 1->0 edge"
            return "loss"
        # (1, action) pairs: byte==1 and next byte in 0..4
        pairs = [LO + i for i in range(len(blk) - 1) if blk[i] == 1 and blk[i + 1] <= 4]
        ps = set(pairs)
        # keep those whose sibling battler buffer is +0x200 (player[0] + enemy[1])
        sib = [a for a in pairs if (a + 0x200) in ps or (a - 0x200) in ps]
        out["all_pairs"] = len(pairs)
        out["stride_pairs"] = [(hex(a), blk[a - LO + 1]) for a in sib[:12]]
        return "loss"

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None, render=lambda: None)
    camp._suppress_heal = True
    camp.trav.travel(target_map=(3, 22), edge="north", max_steps=200, max_seconds=120)
    if "err" in out:
        print(f"   [bufB] {out['err']}", flush=True)
    else:
        print(f"   [bufB] (1,action) pairs total={out.get('all_pairs')}; "
              f"those with a +/-0x200 sibling (gBattleBufferB[0]/[1] candidates):", flush=True)
        for a, act in out.get("stride_pairs", []):
            print(f"   [bufB]     {a}  -> [0]=1 [1]={act}", flush=True)


if __name__ == "__main__":
    main()
