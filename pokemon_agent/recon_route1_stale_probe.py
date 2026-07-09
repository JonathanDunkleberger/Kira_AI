"""recon_route1_stale_probe.py — NIGHT TRAIN shift 1: MEASURE the post-warp stale-grid window.

Hypothesis (from code): the fresh-run parcel wedge at Route 1 (13,0) is a STALE GRID. Right after the
Viridian->Route1 edge-warp the collision buffer is mid-fade (every tile reads blocked), the world
fingerprint excludes map layout (world_fingerprint.py:134), so she can't move, fp stays identical,
fp_stall hits TRAVEL_STALL_RETRIES=4 in ~96 frames and WEDGES before the map finishes loading.

Measure it: cross Viridian->Route1 and sample grid.walkable-count + a north->south BFS every few
frames post-transition. If walkable-count starts near 0 and climbs to full, staleness is confirmed and
we get the window length (how many settle frames the fix needs).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                # noqa: E402
import travel as tv                      # noqa: E402
import pokemon_state as st               # noqa: E402
from battle_agent import BattleAgent     # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
statef = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_POSTGAME", "kira_campaign.state")

b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
b.set_input_owner("agent")

# escape house -> Pallet
_dirs = ["DOWN", "RIGHT", "LEFT", "UP"]
_di = 0
for i in range(80):
    if tuple(tv.map_id(b))[0] == 3:
        break
    before = (tuple(tv.map_id(b)), tuple(tv.coords(b) or (-1, -1)))
    b.press(_dirs[_di % 4], 8, 10, owner="agent")
    for _ in range(24):
        b.run_frame()
    after = (tuple(tv.map_id(b)), tuple(tv.coords(b) or (-1, -1)))
    if before[0] != after[0]:
        _di = 0
        continue
    if before[1] == after[1]:
        _di += 1
print(f"outside: map={tv.map_id(b)} coords={tv.coords(b)}")


def _flee(*a, **k):
    try:
        return BattleAgent(b, log=lambda *_: None).flee()
    except Exception as e:
        print("flee err", e)
        return "fled"


trav = tv.Traveler(b, battle_runner=_flee, log=lambda *_: None)

# Pallet -> Route1 -> Viridian
if trav.travel(target_map=(3, 19), edge="north", max_steps=500, max_seconds=120) != "arrived":
    print(f"!! didn't reach Route1 (at {tv.map_id(b)})"); sys.exit(1)
print(f"on Route1: coords={tv.coords(b)}")
if trav.travel(target_map=(3, 1), edge="north", max_steps=600, max_seconds=150) != "arrived":
    print(f"!! didn't reach Viridian (at {tv.map_id(b)})"); sys.exit(1)
print(f"in Viridian: coords={tv.coords(b)}")

# Now cross SOUTH from Viridian back to Route1, sampling the grid the instant the map flips.
grid = tv.Grid(b)
full_cells = sum(1 for x in range(grid.sx_lo, grid.sx_hi + 1)
                 for y in range(grid.sy_lo, grid.sy_hi + 1) if grid.walkable(x, y))
print(f"Viridian grid walkable-count (settled reference): {full_cells}")

# walk to the south edge of Viridian then press DOWN until the map flips to Route1
prev_map = tuple(tv.map_id(b))
flipped = False
for step in range(300):
    if st.in_battle(b):
        _flee()
        continue
    m = tuple(tv.map_id(b))
    if m == (3, 19):
        flipped = True
        break
    # head south
    b.press("DOWN", 8, 6, owner="agent")
    for _ in range(10):
        b.run_frame()
if not flipped:
    print(f"!! didn't cross back into Route1 (at {tv.map_id(b)} {tv.coords(b)})"); sys.exit(1)

print(f"\n== back in Route1 at coords={tv.coords(b)}; walking back UP to Viridian for the smoking gun ==")
# get back to Viridian to run the REAL deliver_parcel south leg (target=PALLET from VIRIDIAN)
if tuple(tv.map_id(b)) != (3, 1):
    if trav.travel(target_map=(3, 1), edge="north", max_steps=600, max_seconds=150) != "arrived":
        print(f"!! couldn't get back to Viridian (at {tv.map_id(b)})"); sys.exit(1)
print(f"in Viridian at {tv.coords(b)} — now the SMOKING GUN: travel(target=PALLET, edge='south')")
print("   (this is EXACTLY deliver_parcel's south leg — from Viridian, spanning Route1 to Pallet)")

# turn logging ON for this one call
trav.log = print
r = trav.travel(target_map=tv.MAP_PALLET, edge="south", max_steps=600, max_seconds=120)
print(f"\n== SMOKING-GUN southbound travel from Viridian -> {r}  "
      f"now map={tv.map_id(b)} coords={tv.coords(b)} (fail_reason={trav.last_fail_reason})")
print("   EXPECT: wedge at Route1 (~13,0) — band computed for Viridian's south edge, never "
      "recomputed for Route1's -> no_route" if r != "arrived" else "   (arrived — bug not reproduced)")
