"""recon_r21_crossing_probe.py — night shift 10: THE POISONED CROSSING. Rounds 3-5 prove
the (3,39)->(3,0) NORTHWARD surf crossing lands in a limbo that reads map (3,0) coords
(8,6) — a fully-enclosed non-walkable tile (impossible stand) with a frozen fingerprint —
until the impossible-stand tripwire reloads. Reproduce it deliberately and LOOK at the
frame (arsenal #4): what is the game actually showing during the "(8,6)" reads?
PNGs -> %TEMP%/longrun/r21_probe/."""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                # noqa: E402
import travel as tv                      # noqa: E402
import pokemon_state as st               # noqa: E402
import firered_ram as ram                # noqa: E402
from battle_agent import BattleAgent     # noqa: E402
from campaign import Campaign            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
OUT = os.path.join(LONGRUN, "r21_probe")
os.makedirs(OUT, exist_ok=True)

b = Bridge(ROM)
with open(os.path.join(LONGRUN, "banked_POSTGAME", "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()

camp = Campaign(b, battle_runner=lambda: BattleAgent(
                    b, on_event=lambda *a, **k: None, render=lambda: None,
                    log=lambda m: None).run(max_seconds=90),
                on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                render=lambda: None)
camp._save_campaign = lambda *a, **k: True
camp._continuity_save = lambda *a, **k: None

def snap(tag):
    m, c = tv.map_id(b), tv.coords(b)
    print(f"[{tag}] map={m} coords={c} frame={b.frame}", flush=True)
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, f"{tag}.png"))
    except Exception as e:
        print(f"   png failed: {e}", flush=True)

# out of the house
for _ in range(12):
    if tuple(tv.map_id(b))[0] == 3:
        break
    b.press("DOWN", 8, 10)
    for _ in range(30):
        b.run_frame()
snap("00_outside")

# LEG 1: the southward crossing (Pallet -> Route 21 north) — band + edge-mount southbound
r1 = camp.trav.travel(target_map=(3, 39), edge="south", max_steps=400, max_seconds=150)
print(f"LEG1 south -> {r1}", flush=True)
snap("01_after_south")

if tuple(tv.map_id(b)) == (3, 39):
    # LEG 2: THE POISONED ONE — replicate the round-5 recipe EXACTLY: a coord-mode leg to
    # (11,10) (Route 21 grass) that crosses the north edge mid-walk and keeps chasing the
    # now-stale coord on Pallet. Round 5 ended this at a "(8,6)" impossible-stand limbo.
    r2 = camp.trav.travel(target_map=None, arrive_coord=(11, 10), max_steps=200, max_seconds=90)
    print(f"LEG2 coord-chase -> {r2}", flush=True)
    snap("02_after_north")
    # sample the aftermath: reads + frames while whatever-it-is settles
    for i in range(20):
        for _ in range(30):
            b.run_frame()
        m, c = tv.map_id(b), tv.coords(b)
        print(f"  sample {i}: map={m} coords={c}", flush=True)
        if i in (0, 4, 9, 19):
            snap(f"03_sample_{i}")
    # is the world alive? try 4 presses and watch coords
    for k in ("LEFT", "RIGHT", "UP", "DOWN"):
        c0 = tv.coords(b)
        b.press(k, 8, 10)
        for _ in range(20):
            b.run_frame()
        print(f"  press {k}: {c0} -> {tv.coords(b)}", flush=True)
    snap("04_after_presses")
else:
    print("LEG1 did not land on (3,39) — probe inconclusive", flush=True)
