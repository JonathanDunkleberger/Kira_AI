"""recon_bill_eyes.py — EYES ON the cottage: enter, screenshot, face Bill, press A, screenshot.
Writes PNGs to the scratchpad for a human/model look. Pure reads + presses on a throwaway state."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge
from battle_agent import BattleAgent
import travel as tv

OUT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\92df62cc-b82e-4538-8fdb-4a238758a51f\scratchpad"
os.makedirs(OUT, exist_ok=True)
ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = os.path.join(_HERE, "states", "campaign", "r24_for_bill.state")

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")
agent = BattleAgent(b, log=print, render=None)
trav = tv.Traveler(b, battle_runner=lambda: agent.run(max_seconds=120), render=None, log=print)

trav.travel(target_map=(3, 44), edge="east", max_steps=400, max_seconds=300)
trav.travel(target_map=None, arrive_coord=(51, 5), max_steps=400, max_seconds=900)
for _ in range(6):
    b.press("UP", 8, 8, None, owner="agent")
    for _f in range(70):
        b.run_frame()
    if tuple(tv.map_id(b)) == (30, 0):
        break
print(f"inside map={tv.map_id(b)} coords={tv.coords(b)}")
b.frame_rgb().save(os.path.join(OUT, "bill_1_inside.png"))
trav.travel(target_map=None, arrive_coord=(10, 7), max_steps=60, max_seconds=40)
b.press("UP", 8, 8, None, owner="agent")
for _f in range(20):
    b.run_frame()
b.frame_rgb().save(os.path.join(OUT, "bill_2_facing.png"))
b.press("A", 6, 12, None, owner="agent")
for _f in range(60):
    b.run_frame()
b.frame_rgb().save(os.path.join(OUT, "bill_3_afterA.png"))
b.press("A", 6, 12, None, owner="agent")
for _f in range(60):
    b.run_frame()
b.frame_rgb().save(os.path.join(OUT, "bill_4_afterA2.png"))
print("PNGs written")
