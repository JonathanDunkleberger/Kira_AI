"""recon_east_cross.py — EMPIRICAL test of the exact failing op: travel(target_map=Cerulean, edge='east')
from the leveled pocket state (107,12). Uses the REAL Traveler + BattleAgent (fights/flees encounters like
the live loop). Watch the band rows + where it fails. No canonical mutation (state loaded in-memory only)."""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge
from battle_agent import BattleAgent
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = sorted(glob.glob(os.path.join(_HERE, "states", "campaign", "pre_reload_*.state")))[-1]
print(f"SAVE={os.path.basename(SAVE)}")

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")

agent = BattleAgent(b, log=print, render=None)


def battle_runner():
    return agent.run(max_seconds=120)


trav = tv.Traveler(b, battle_runner=battle_runner, render=None, log=print)
print(f"START map={tv.map_id(b)} coords={tv.coords(b)}")
r = trav.travel(target_map=(3, 3), edge="east", max_steps=200, max_seconds=180)
print(f"RESULT: {r}  now map={tv.map_id(b)} coords={tv.coords(b)}")
print("PASS" if tuple(tv.map_id(b)) == (3, 3) else "FAIL")
