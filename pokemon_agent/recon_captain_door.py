"""recon_captain_door.py — ground-truth the (1,6)@(30,2)->(1,11) captain's-office door.

Run 19 evidence: standing ON (30,2) fires nothing; enter_warp from (30,3) 'didn't warp';
(29,2) is a wall. Loads the DISPOSABLE stage state (kitchen (1,10)@(7,2)), walks to the
2F corridor (1,6), dumps behaviors around the door, and tries every entry side. If the
office opens: dump (1,11) warps + NPC objects (the captain?). Staging-only; no canonical
writes. RUN: python pokemon_agent/recon_captain_door.py
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


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"boot: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    camp._suppress_heal = True

    # 1. out of the kitchen: stand (7,2), the door (7,1) is a step-UP warp
    if tuple(tv.map_id(b)) == (1, 10):
        camp.enter_warp(pick=(7, 1), budget_s=60)
        print(f"after kitchen exit: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
    # 2. (1,5) -> (1,6) via the (3,8) 0x6d door (entered via LEFT from (4,8))
    if tuple(tv.map_id(b)) == (1, 5):
        camp.trav.travel(target_map=None, arrive_coord=(4, 8), max_steps=200, max_seconds=90)
        camp._enter_directional_warp((3, 8))
        print(f"after 1F->2F: map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
    if tuple(tv.map_id(b)) != (1, 6):
        print("!! did not reach the 2F corridor (1,6) — aborting", flush=True)
        return

    # 3. behaviors around the captain door (30,2)
    print("behaviors x=27..31, y=0..5:", flush=True)
    for y in range(0, 6):
        row = []
        for x in range(27, 32):
            try:
                beh = camp._tile_behavior(x, y)
            except Exception:
                beh = None
            row.append(f"{beh:02x}" if isinstance(beh, int) else "??")
        print(f"  y={y}: " + " ".join(row), flush=True)
    g = tv.Grid(b)
    walk = {(x, y): g.walkable(x + tv.MAP_OFFSET, y + tv.MAP_OFFSET)
            for x in range(27, 32) for y in range(0, 6)}
    print("walkable:", {k: v for k, v in sorted(walk.items())}, flush=True)
    print("NPCs here:", objects(b), flush=True)

    # 4. try entries: approach (30,3) step UP; then E->W and W->E along y=2
    camp.trav.travel(target_map=None, arrive_coord=(30, 3), max_steps=300, max_seconds=120)
    print(f"at approach: pos={tv.coords(b)}", flush=True)
    for tries, (side, key) in enumerate([((30, 3), "UP"), ((31, 2), "LEFT"), ((29, 2), "RIGHT")]):
        if tuple(tv.map_id(b)) != (1, 6):
            break
        cur = tuple(tv.coords(b) or ())
        if cur != side:
            camp.trav.travel(target_map=None, arrive_coord=side, max_steps=120, max_seconds=60)
            if tuple(tv.coords(b) or ()) != side:
                print(f"  entry try #{tries}: can't stand at {side}", flush=True)
                continue
        for _ in range(6):
            b.press(key, 8, 8, None, owner="agent")
            for _ in range(20):
                b.run_frame()
            if tuple(tv.map_id(b)) != (1, 6):
                break
        print(f"  entry try #{tries}: {side} press {key} -> map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
    # also blunt: the directional-warp primitive + enter_warp
    if tuple(tv.map_id(b)) == (1, 6):
        if camp._tile_behavior(30, 2) in camp._WARP_ENTRY:
            camp._enter_directional_warp((30, 2))
            print(f"  directional primitive -> map={tv.map_id(b)}", flush=True)
    if tuple(tv.map_id(b)) == (1, 6):
        r = camp.enter_warp(pick=(30, 2), budget_s=90)
        print(f"  enter_warp((30,2)) -> {r} map={tv.map_id(b)}", flush=True)

    if tuple(tv.map_id(b)) == (1, 11):
        print("== CAPTAIN'S OFFICE OPEN ==", flush=True)
        print("warps:", [(tuple(xy), tuple(d), w) for xy, d, w in tv.read_warps(b)], flush=True)
        print("NPCs:", objects(b), flush=True)
        try:
            b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "captain_office.png"))
            print("frame -> captain_office.png", flush=True)
        except Exception as e:
            print(f"frame failed: {e}", flush=True)
    else:
        print(f"door NOT opened; final map={tv.map_id(b)} pos={tv.coords(b)}", flush=True)
        try:
            b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "captain_door_fail.png"))
            print("frame -> captain_door_fail.png", flush=True)
        except Exception as e:
            print(f"frame failed: {e}", flush=True)


if __name__ == "__main__":
    main()
