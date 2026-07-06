"""recon_route4_reach.py — is ANY Route-4 grass Center-reachable? Decides the strand fix approach.
Loads a pre_reload (strand-pocket) state; BFS reachability from the pocket AND from the Route-4
Center door approach (12,6). No mutation."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = sys.argv[1] if len(sys.argv) > 1 else None
if SAVE is None:
    import glob
    cands = sorted(glob.glob(os.path.join(_HERE, "states", "campaign", "pre_reload_*.state")))
    SAVE = cands[-1]
print(f"SAVE={os.path.basename(SAVE)}")

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")

off = tv.MAP_OFFSET
m = tv.map_id(b)
here = tv.coords(b)
grid = tv.Grid(b)
grass = [(x - off, y - off) for (x, y) in grid.grass]
print(f"map={m} here={here} grass_tiles={len(grass)}")

CENTER_DOOR = (12, 5)
appr = (CENTER_DOOR[0], CENTER_DOOR[1] + 1)   # stand south of the door

# reachable set from HERE
def reach_from(start):
    seen = set()
    if tv.bfs(grid, start, lambda t: (seen.add(t) or False), walkable=grid.walkable) is None:
        pass
    return seen

reach_here = reach_from(here)
reach_center = reach_from(appr)
print(f"reachable-from-here tiles={len(reach_here)}; reachable-from-Center-approach tiles={len(reach_center)}")
print(f"is Center approach {appr} reachable FROM here? {appr in reach_here}")
print(f"is here {here} reachable FROM Center approach? {here in reach_center}")

grass_from_here = [g for g in grass if g in reach_here]
grass_from_center = [g for g in grass if g in reach_center]
print(f"grass reachable FROM here: {len(grass_from_here)}  e.g. {sorted(grass_from_here)[:8]}")
print(f"grass reachable FROM Center-approach (Center-safe grass): {len(grass_from_center)}  e.g. {sorted(grass_from_center)[:8]}")
# grass she can grind on AND heal from = intersection reachable from BOTH sides that also connects
safe_grind = [g for g in grass if g in reach_center]
print(f"=> Center-SAFE grass on this map: {len(safe_grind)}")
print("DONE")
