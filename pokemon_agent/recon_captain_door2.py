"""recon_captain_door2.py — manual-press verification of the 0x6C stair at (1,6)@(30,2).

Hypothesis: enter MOVING RIGHT from (29,2). Blind-press route (30,3)->(29,3)->(29,2)->RIGHT,
reading coords after each press (bonk-verified, no Grid trust). Fallback: (31,2)->LEFT.
Staging-only. RUN: python pokemon_agent/recon_captain_door2.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = r"G:\temp\longrun\stage\kira_campaign.state"
OUT = r"G:\temp\claude\C--WINDOWS-system32\9717c34e-ef1a-4fc3-88e5-c78650ebc7f0\scratchpad"


def objects(b):
    OB, SZ = 0x02036E38, 0x24
    out = []
    for i in range(16):
        o = OB + i * SZ
        try:
            if not (b.rd8(o) & 1):
                continue
            out.append({"idx": i, "gfx": b.rd8(o + 0x05),
                        "coord": (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7)})
        except Exception:
            continue
    return out


def step(b, key, n=1, label=""):
    for _ in range(n):
        b.press(key, 8, 10, None, owner="agent")
        for _ in range(30):
            b.run_frame()
    print(f"  {label or key}: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    camp._suppress_heal = True
    print(f"boot: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
    if tuple(tv.map_id(b)) == (1, 10):
        camp.enter_warp(pick=(7, 1), budget_s=60)
    if tuple(tv.map_id(b)) == (1, 5):
        camp.trav.travel(target_map=None, arrive_coord=(4, 8), max_steps=200, max_seconds=90)
        camp._enter_directional_warp((3, 8))
    if tuple(tv.map_id(b)) != (1, 6):
        print("!! not on (1,6) — abort", flush=True)
        return
    camp.trav.travel(target_map=None, arrive_coord=(30, 3), max_steps=300, max_seconds=120)
    print(f"staged at: pos={tv.coords(b)} (want (30,3))", flush=True)

    # manual: face+step LEFT to (29,3)
    step(b, "LEFT", 2, "LEFT->29,3")
    step(b, "UP", 2, "UP->29,2")
    if tuple(tv.coords(b) or ()) == (29, 2):
        # THE ENTRY: moving RIGHT onto the 0x6C stair
        for i in range(4):
            b.press("RIGHT", 8, 10, None, owner="agent")
            for _ in range(40):
                b.run_frame()
            print(f"  RIGHT press {i}: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
            if tuple(tv.map_id(b)) != (1, 6):
                break
        if tuple(tv.map_id(b)) == (1, 6) and tuple(tv.coords(b) or ()) == (30, 2):
            print("  standing on stair — waiting 240f for a delayed fire", flush=True)
            for _ in range(240):
                b.run_frame()
                if tuple(tv.map_id(b)) != (1, 6):
                    break
            print(f"  after wait: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
    else:
        print("  couldn't reach (29,2); trying east side (31,2) LEFT entry", flush=True)
        camp.trav.travel(target_map=None, arrive_coord=(31, 3), max_steps=120, max_seconds=60)
        step(b, "UP", 2, "UP->31,2")
        for i in range(4):
            b.press("LEFT", 8, 10, None, owner="agent")
            for _ in range(40):
                b.run_frame()
            print(f"  LEFT press {i}: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
            if tuple(tv.map_id(b)) != (1, 6):
                break

    if tuple(tv.map_id(b)) == (1, 11):
        print("== CAPTAIN'S OFFICE (1,11) OPEN ==", flush=True)
        print("warps:", [(tuple(xy), tuple(d), w) for xy, d, w in tv.read_warps(b)], flush=True)
        print("NPCs:", objects(b), flush=True)
        try:
            b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "captain_office.png"))
            print("frame saved", flush=True)
        except Exception as e:
            print(f"frame failed: {e}", flush=True)
    else:
        print(f"NOT opened: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)


if __name__ == "__main__":
    main()
