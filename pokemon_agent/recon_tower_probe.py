"""recon_tower_probe.py — why doesn't Tower 2F's up-stairs (4,10) fire? Dump behaviors +
walkability around it from the staged 2F save, snap a frame, try entries from each side.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_tower_probe.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_tower")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "tower_probe")

b = Bridge(ROM)
with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(40):
    b.run_frame()
camp = Campaign(b, battle_runner=lambda: BattleAgent(
    b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None).run(240),
    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
os.makedirs(DBG, exist_ok=True)
print(f"map={tv.map_id(b)} coords={tv.coords(b)}")
print(f"warps={tv.read_warps(b)}")
g = tv.Grid(b)
for y in range(7, 14):
    row = []
    for x in range(1, 9):
        v = camp._tile_behavior(x, y)
        row.append(f"({x},{y})={v:02x}{'.' if g.walkable(x, y) else '#'}")
    print("  " + " ".join(row))
print(f"npcs={sorted(camp.trav._npc_tiles())}")
b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, "2f_stairs.png"))
# walk to (4,11) then probe entries
camp.trav.travel(target_map=None, arrive_coord=(4, 11), max_steps=120, max_seconds=60)
print(f"at {tv.coords(b)}")
for key in ("UP", "UP", "UP"):
    c0 = tuple(tv.coords(b))
    b.press(key, 8, 10, camp.render, owner="agent")
    for _ in range(40):
        b.run_frame()
    print(f"  press {key}: {c0} -> {tv.coords(b)} map={tv.map_id(b)}")
b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, "2f_after_up.png"))
