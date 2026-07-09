"""recon_viridian_north_probe.py — NIGHT TRAIN shift 1: the Viridian->Route2 north-cross wedge.

Fresh-spine ADVANCE_NORTH(Pewter) infinite-loops: from Viridian (3,1) at the north edge (19,0),
travel(target=PEWTER, edge='north') returns no_path -> advance_north falls back to enter_warp north,
which warps into buildings (15,0)->(1,0) and back forever. Viridian never reaches Route 2 (3,20).

Reproduce the Viridian north-edge cross and instrument: the north connection band (which columns of
Viridian's top row overlap Route 2), BFS from her stand + from the whole north row, and the live
travel() verdict. Find why the cross fails.
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
ROUTE2 = (3, 20)

b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
b.set_input_owner("agent")

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
    except Exception:
        return "fled"


trav = tv.Traveler(b, battle_runner=_flee, log=lambda *_: None)

# Pallet -> Route1 -> Viridian
if trav.travel(target_map=(3, 19), edge="north", max_steps=500, max_seconds=120) != "arrived":
    print(f"!! didn't reach Route1 (at {tv.map_id(b)})"); sys.exit(1)
if trav.travel(target_map=(3, 1), edge="north", max_steps=600, max_seconds=150) != "arrived":
    print(f"!! didn't reach Viridian (at {tv.map_id(b)})"); sys.exit(1)
print(f"in Viridian at coords={tv.coords(b)}")

# instrument Viridian's NORTH edge / band
grid = tv.Grid(b)
cur = tuple(tv.coords(b))
print(f"\ngrid dims: sx {grid.sx_lo}..{grid.sx_hi}  sy {grid.sy_lo}..{grid.sy_hi}")
_ny = grid.sy_lo - 1     # one row PAST the north border = the band source
row = "".join(('W' if grid.is_water(x, _ny) else ('w' if grid.walkable(x, _ny) else '.'))
              for x in range(grid.sx_lo, grid.sx_hi + 1))
print(f"north overlap row y={_ny}: {row}   [x={grid.sx_lo}..{grid.sx_hi}]")
band = {p for p in range(grid.sx_lo, grid.sx_hi + 1) if grid.walkable(p, _ny)}
print(f"north band (land): {sorted(band) or 'EMPTY -> falls back to any north-edge tile'}")

# top row (y=0) walkability + BFS from each top-row tile to a band col at y<=0
print("\n== BFS from Viridian north-edge tiles (y=0) to the north band ==")
_b = band or None
n_pass = 0
for tx in range(grid.sx_lo, grid.sx_hi + 1):
    if not grid.walkable(tx, 0):
        continue
    def _goal(t, bnd=_b):
        return t[1] <= 0 and (bnd is None or t[0] in bnd)
    p = tv.bfs(grid, (tx, 0), _goal, walkable=grid.walkable)
    if p:
        n_pass += 1
    else:
        print(f"  from ({tx},0): NO PATH to a band col")
print(f"  ({n_pass} of the walkable top-row tiles CAN reach a band col)")

# BFS from her actual stand to the north band
def _goal2(t, bnd=_b):
    return t[1] <= 0 and (bnd is None or t[0] in bnd)
p = tv.bfs(grid, cur, _goal2, walkable=grid.walkable)
print(f"\nBFS from stand {cur} to north band: {'PATH ' + str(len(p)) + ' -> ' + str(p[-1]) if p else 'NO PATH'}")

# cross into Route 2 (postgame confirms this arrives at ~(10,79))
r = trav.travel(target_map=ROUTE2, edge="north", max_steps=600, max_seconds=90)
print(f"\n== crossed to Route2 -> {r}  now map={tv.map_id(b)} coords={tv.coords(b)}")
if tuple(tv.map_id(b)) != ROUTE2:
    print("!! not on Route 2 — abort"); sys.exit(1)

# THE REAL WEDGE: from the Route 2 SOUTH-edge entry, go NORTH. Sample grid staleness + north BFS
# over ~200 frames to separate 'stale grid' from 'genuine sealed pocket'.
print("\n== Route2 entry: sampling grid + NORTH BFS over ~200 frames (stale vs sealed) ==")
for k in range(20):
    g = tv.Grid(b)
    wc = sum(1 for x in range(g.sx_lo, g.sx_hi + 1)
             for y in range(g.sy_lo, g.sy_hi + 1) if g.walkable(x, y))
    cur2 = tuple(tv.coords(b) or (-1, -1))
    pn = tv.bfs(g, cur2, lambda t: t[1] <= 0, walkable=g.walkable) if cur2[0] >= 0 else None
    print(f"  +{k*10:3d}f: coords={cur2} dims(sy_hi={g.sy_hi}) walkable={wc:4d} "
          f"north_BFS={'PATH ' + str(len(pn)) + ' -> ' + str(pn[-1]) if pn else 'NO PATH'}")
    for _ in range(10):
        b.run_frame()

# how far north can the flood reach from here, and what stops it?
g = tv.Grid(b)
cur2 = tuple(tv.coords(b))
seen = {cur2}
frontier = [cur2]
min_y = cur2[1]
while frontier:
    nf = []
    for (x, y) in frontier:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = (x + dx, y + dy)
            if t not in seen and g.walkable(*t):
                seen.add(t); nf.append(t); min_y = min(min_y, t[1])
    frontier = nf
print(f"\nland flood from {cur2}: {len(seen)} tiles, shallowest row y={min_y} (sy_lo={g.sy_lo})")
shallow = sorted(t for t in seen if t[1] == min_y)[:10]
print(f"  shallowest tiles: {shallow}")
for (x, y) in shallow[:5]:
    print(f"  above {(x, y)}: walkable={g.walkable(x, y - 1)} water={g.is_water(x, y - 1)} "
          f"or_surf={g.walkable_or_surf(x, y - 1)}")

# dump ALL Route 2 warp/door tiles + reachability from the south entry
print("\n== Route 2 warp/door tiles + reachability from the entry ==")
g = tv.Grid(b)
c0 = tuple(tv.coords(b))
warps = tv.read_warps(b)
for (wxy, wdest, wid) in sorted(warps, key=lambda w: w[0][1]):
    wx, wy = wxy
    beh = None
    try:
        beh = hex(g.col.get((wx + 7, wy + 7)))
    except Exception:
        pass
    # is the tile just SOUTH of the warp reachable (the UP-approach stand)?
    appr = (wx, wy + 1)
    reach = bool(tv.bfs(g, c0, lambda t, a=appr: t == a, walkable=g.walkable))
    reach_on = bool(tv.bfs(g, c0, lambda t, w=(wx, wy): t == w, walkable=g.walkable))
    print(f"  warp {wxy} -> {wdest}  approach{appr} reachable={reach}  on-tile reachable={reach_on}")
