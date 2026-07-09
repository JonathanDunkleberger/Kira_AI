"""recon_vermilion_travel.py — REPRODUCE the harbor->Surge-gym travel wedge (shift 16).
Teleports the player to the S.S. Anne exit spawn (23,33) in Vermilion, then runs the
REAL travel planner to the gym-door approach (14,26) and captures its logs. READ-ONLY
on canonical (operates on a loaded state in RAM only).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from bridge import Bridge
import travel as tv
import firered_ram as ram

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = sys.argv[1] if len(sys.argv) > 1 else r"G:/temp/longrun/stage/kira_campaign.state"
TX, TY = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (23, 33)
GX, GY = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (14, 26)

b = Bridge(ROM)
with open(STATE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.run_frame()
b.set_input_owner("agent")

# teleport: SB1 pos + player object-event buffer coords (obj0)
def w16(addr, v):
    b.core.memory.u16.raw_write(addr, v & 0xFFFF)
sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
w16(sb1 + ram.SB1_OFF_POS_X, TX)
w16(sb1 + ram.SB1_OFF_POS_Y, TY)
OB = 0x02036E38
w16(OB + 0x10, TX + 7)   # buffer x
w16(OB + 0x12, TY + 7)   # buffer y
for _ in range(8):
    b.run_frame()
print("after teleport: coords", tv.coords(b), "map", tv.map_id(b))

# npc tiles + door reachability
trav = tv.Traveler(b, battle_runner=lambda: "win", render=lambda: None,
                   log=lambda m: print(m), owner="agent")
print("NPC tiles:", sorted(trav._npc_tiles()))
g = tv.Grid(b)
p = tv.bfs(g, (TX, TY), lambda t: t == (GX, GY), walkable=g.walkable)
print(f"optimistic BFS ({TX},{TY})->({GX},{GY}):", "PATH len " + str(len(p)) if p else "NO PATH")

print(f"\n==== REAL travel ({TX},{TY}) -> ({GX},{GY}) ====")
r = trav.travel(target_map=None, arrive_coord=(GX, GY), max_steps=400, max_seconds=45)
print("RESULT:", r, "reason:", trav.last_fail_reason, "final coords:", tv.coords(b))
