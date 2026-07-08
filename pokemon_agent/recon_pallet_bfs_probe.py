"""recon_pallet_bfs_probe.py — night shift 10: in the killed e4surf window, EVERY travel
from Pallet (8,6) read no_route instantly (even north to Route 1, pure land). All three
planner BFS passes failed. Reproduce the exact BFS calls from a Pallet stand and find which
layer lies: Grid content? edge_open elevation law? the goal band? the water layer?"""
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
import field_moves as fm                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
statef = os.path.join(LONGRUN, "banked_POSTGAME", "kira_campaign.state")
b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
print(f"spawn: map={tv.map_id(b)} coords={tv.coords(b)}")

# walk out of the house (spawn is indoors at home) — press DOWN till the map flips
for i in range(12):
    if tuple(tv.map_id(b))[0] == 3:
        break
    b.press("DOWN", 8, 10)
    for _ in range(30):
        b.run_frame()
print(f"outside: map={tv.map_id(b)} coords={tv.coords(b)}")
if tuple(tv.map_id(b))[0] != 3:
    print("!! could not exit the house — abort")
    sys.exit(1)

# walk to (8,6) — the exact tile the e4surf window wedged on (all-BFS-fail from there)
_tgt = (8, 6)
for _ in range(24):
    c = tuple(tv.coords(b))
    if c == _tgt:
        break
    dx, dy = _tgt[0] - c[0], _tgt[1] - c[1]
    key = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) and dx else \
          ("DOWN" if dy > 0 else "UP")
    b.press(key, 8, 10)
    for _ in range(20):
        b.run_frame()
print(f"standing at: {tv.coords(b)} (want {_tgt})")

grid = tv.Grid(b)
cur = tuple(tv.coords(b))
print(f"grid dims: sx {grid.sx_lo}..{grid.sx_hi}  sy {grid.sy_lo}..{grid.sy_hi}  "
      f"water_tiles={len(grid.water)}  can_surf={fm.can_use(b, 'surf')}")
print(f"start tile {cur}: walkable={grid.walkable(*cur)} safe={grid.walkable_safe(*cur)} "
      f"or_surf={grid.walkable_or_surf(*cur)}")

# 0b) classify (8,6) and its neighbourhood + the other live positions from the run
for t in [(8, 6), (7, 6), (9, 6), (8, 5), (8, 7), (6, 9), (4, 19), (12, 19), (6, 7)]:
    bx, by = t[0] + 7, t[1] + 7
    print(f"  tile {t}: walkable={grid.walkable(*t)} col={grid.col.get((bx, by))} "
          f"water={grid.is_water(*t)}")
for d, dxdy in (("E", (1, 0)), ("W", (-1, 0)), ("S", (0, 1)), ("N", (0, -1))):
    print(f"  edge (8,6)->{d}: open={grid.edge_open(8, 6, *dxdy)}")

# 0) hypothetical start = (8,6), the run's wedge tile (BFS is pure — no need to stand there)
print(f"tile (8,6): walkable={grid.walkable(8, 6)} safe={grid.walkable_safe(8, 6)}")
p_h = tv.bfs(grid, (8, 6), lambda t: t[1] <= 0, walkable=grid.walkable)
print(f"north land BFS from (8,6): {'PATH len ' + str(len(p_h)) if p_h else 'NO PATH'}")
p_h2 = tv.bfs(grid, (8, 6), lambda t: t[1] >= grid.sy_hi, walkable=grid.walkable_or_surf)
print(f"south WATER BFS from (8,6): {'PATH len ' + str(len(p_h2)) if p_h2 else 'NO PATH'}")

# 1) LAND BFS north: goal = row 0 (Route 1). This worked all game — control.
p_n = tv.bfs(grid, cur, lambda t: t[1] <= 0, walkable=grid.walkable)
print(f"north land BFS from {cur}: {'PATH len ' + str(len(p_n)) if p_n else 'NO PATH'}")

# 2) LAND BFS south: goal = bottom row (should fail — sea).
p_s = tv.bfs(grid, cur, lambda t: t[1] >= grid.sy_hi, walkable=grid.walkable)
print(f"south land BFS: {'PATH len ' + str(len(p_s)) if p_s else 'NO PATH (expected — sea)'}")

# 3) WATER BFS south: walkable_or_surf — the layer that must carry the Cinnabar road.
p_w = tv.bfs(grid, cur, lambda t: t[1] >= grid.sy_hi, walkable=grid.walkable_or_surf)
print(f"south WATER BFS: {'PATH len ' + str(len(p_w)) if p_w else 'NO PATH  <-- THE DEFECT LAYER'}")

# 4) if no water path: how far does the flood get? where does it stop at the shore?
if not p_w:
    seen = {cur}
    frontier = [cur]
    max_y = cur[1]
    while frontier:
        nf = []
        for (x, y) in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                t = (x + dx, y + dy)
                if t in seen:
                    continue
                if grid.walkable_or_surf(*t):
                    seen.add(t)
                    nf.append(t)
                    max_y = max(max_y, t[1])
        frontier = nf
    shore = sorted(t for t in seen if t[1] == max_y)
    print(f"walkable_or_surf flood: {len(seen)} tiles, deepest row y={max_y} (sy_hi={grid.sy_hi})")
    print(f"  deepest tiles: {shore[:12]}")
    # what's just below the deepest row? classify the wall
    for (x, y) in shore[:6]:
        t2 = (x, y + 1)
        print(f"  below {(x, y)}: walkable={grid.walkable(*t2)} water={grid.is_water(*t2)} "
              f"or_surf={grid.walkable_or_surf(*t2)}")
