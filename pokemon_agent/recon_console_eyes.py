"""recon_console_eyes.py — stand at (4,6), face UP at Bill's computer (4,5), screenshot before/after A.
Also dumps facing byte + coords. Fast path: reuse the sequence up to inside-the-cottage."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge
from battle_agent import BattleAgent
from dialogue_drive import DialogueDriver, box_open, player_facing
import travel as tv

OUT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\92df62cc-b82e-4538-8fdb-4a238758a51f\scratchpad"
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
print(f"inside {tv.map_id(b)}@{tv.coords(b)}")
trav.travel(target_map=None, arrive_coord=(4, 6), max_steps=60, max_seconds=40)
print(f"at {tv.coords(b)} facing={player_facing(b)}")
# FACE-VERIFIED turn (readback doctrine): press UP until the facing byte reads 2 (up)
for t in range(6):
    if player_facing(b) == 2:
        break
    b.press("UP", 8, 8, None, owner="agent")
    for _f in range(16):
        b.run_frame()
    print(f"  turn try {t}: facing={player_facing(b)} coords={tv.coords(b)}")
print(f"faced: coords={tv.coords(b)} facing={player_facing(b)}")
b.frame_rgb().save(os.path.join(OUT, "console_1_facing.png"))
b.press("A", 6, 12, None, owner="agent")
for _f in range(30):
    b.run_frame()
print(f"after A: box={box_open(b)}")
if box_open(b):
    from dialogue_drive import DialogueDriver
    r = DialogueDriver(b, render=None, log=print).drive(label="console")
    print(f"console drive -> {r}")
b.frame_rgb().save(os.path.join(OUT, "console_2_afterA.png"))
