"""recon_route1_south_probe.py — NIGHT TRAIN shift 1: the fresh-run Oak's-Parcel wedge.

Fresh run wedges delivering the parcel: no_route at Route 1 (3,19) coord ~(13,0), TRAVEL WEDGE x4,
travelling SOUTH to Pallet. Reproduce the exact southbound BFS from Route 1's north edge and find
which layer lies: Grid content? the N/S connection band? the goal predicate?

Boots banked_POSTGAME (Pallet), crosses NORTH onto Route 1, walks to the north edge, then runs the
exact travel(target_map=PALLET, edge='south') call + raw BFS. Prints the south connection band under
BOTH can_surf assumptions (fresh run = can_surf False; postgame = True) so we see the fresh-run truth.
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
import field_moves as fm                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
statef = os.path.join(LONGRUN, "banked_POSTGAME", "kira_campaign.state")

b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
b.set_input_owner("agent")
print(f"spawn: map={tv.map_id(b)} coords={tv.coords(b)}")

# escape the house (spawn is indoors, maybe on 2F needing stairs) — wander toward ANY map
# transition until we're outdoors (group 3). Try each direction; a direction that changes the
# map or the coords is "progress"; cycle directions to find stairs/door.
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
        _di = 0            # map flipped (down stairs / out a door) — keep pushing same axis
        continue
    if before[1] == after[1]:
        _di += 1           # blocked — rotate direction
print(f"outside: map={tv.map_id(b)} coords={tv.coords(b)}")
if tuple(tv.map_id(b))[0] != 3:
    print("!! could not exit the house — abort"); sys.exit(1)


def _noop_battle(*a, **k):
    return "done"


trav = tv.Traveler(b, battle_runner=_noop_battle, log=print)

# 1) cross NORTH from Pallet onto Route 1 (a plain land connection — should just work)
r = trav.travel(target_map=tv.MAP_PALLET if False else (3, 19), edge="north",
                max_steps=400, max_seconds=90)
print(f"\n== north-to-Route1 travel -> {r}  now map={tv.map_id(b)} coords={tv.coords(b)}")
if tuple(tv.map_id(b)) != (3, 19):
    print("!! not on Route 1 — cannot reproduce the southbound wedge from here"); sys.exit(1)

# 2) walk to the NORTH edge of Route 1 (where the fresh run wedges, ~y=0) to match the wedge stand
for _ in range(30):
    c = tuple(tv.coords(b))
    if c[1] <= 1:
        break
    b.press("UP", 8, 10, owner="agent")
    for _ in range(20):
        b.run_frame()
print(f"at Route1 north edge: coords={tv.coords(b)}")

# 3) instrument the SOUTH connection band + BFS from here
grid = tv.Grid(b)
cur = tuple(tv.coords(b))
can_surf = fm.can_use(b, "surf")
print(f"\ngrid dims: sx {grid.sx_lo}..{grid.sx_hi}  sy {grid.sy_lo}..{grid.sy_hi}  "
      f"water_tiles={len(grid.water)}  can_surf(live)={can_surf}")
print(f"stand {cur}: walkable={grid.walkable(*cur)} safe={grid.walkable_safe(*cur)}")

# the south overlap row (one tile PAST the south border) — the band source
_py = grid.sy_hi + 1
row = "".join(('W' if grid.is_water(x, _py) else ('w' if grid.walkable(x, _py) else '.'))
              for x in range(grid.sx_lo, grid.sx_hi + 1))
print(f"south overlap row y={_py}: {row}   (W=water w=walkable .=closed) [x={grid.sx_lo}..{grid.sx_hi}]")

# band computed BOTH ways
band_land = {p for p in range(grid.sx_lo, grid.sx_hi + 1) if grid.walkable(p, _py)}
band_surf = {p for p in range(grid.sx_lo, grid.sx_hi + 1)
             if grid.walkable(p, _py) or grid.is_water(p, _py)}
print(f"south band (LAND-ONLY, the fresh-run case): {sorted(band_land) or 'EMPTY -> falls back to any edge tile'}")
print(f"south band (surf-aware): {sorted(band_surf)}")

# 4) BFS south: any-edge-tile vs band-gated (mirror travel()'s _edge_goal)
p_any = tv.bfs(grid, cur, lambda t: t[1] >= grid.sy_hi, walkable=grid.walkable)
print(f"\nland BFS to ANY south-edge tile: {'PATH len ' + str(len(p_any)) if p_any else 'NO PATH'}")
if p_any:
    print(f"  reaches south-edge tile: {p_any[-1]}")
for label, bnd in (("LAND-ONLY", band_land or None), ("surf-aware", band_surf or None)):
    def _goal(t, bnd=bnd):
        return t[1] >= grid.sy_hi and (bnd is None or t[0] in bnd)
    p = tv.bfs(grid, cur, _goal, walkable=grid.walkable)
    print(f"land BFS to south-edge WITHIN {label} band: "
          f"{'PATH len ' + str(len(p)) + ' -> ' + str(p[-1]) if p else 'NO PATH'}")

# 4b) THE WEDGE STAND: pure BFS from the NORTH-edge tiles (y=0) down to the south band — this is
# where the fresh run actually wedges (arrives from Viridian at ~(13,0)). BFS is pure.
print("\n== BFS from NORTH-edge tiles (the fresh-run wedge stand) to the south band ==")
_band = band_land or None
for tx in range(grid.sx_lo, grid.sx_hi + 1):
    if not grid.walkable(tx, 0):
        continue
    def _goal(t, bnd=_band):
        return t[1] >= grid.sy_hi and (bnd is None or t[0] in bnd)
    p = tv.bfs(grid, (tx, 0), _goal, walkable=grid.walkable)
    print(f"  from ({tx},0): {'PATH len ' + str(len(p)) + ' -> ' + str(p[-1]) if p else 'NO PATH <-- WEDGE'}")

# 4c) dump the collision of the central columns down Route 1 to expose the LEDGE structure
print("\n== central-column collision (x=10..14) down Route 1 (w=walkable .=closed) ==")
for y in range(0, grid.sy_hi + 1):
    cells = "".join('w' if grid.walkable(x, y) else '.' for x in range(10, 15))
    print(f"  y={y:2d}: {cells}")

# 5) the actual travel call (fresh-run path uses edge='south')
print("\n== reproducing trav.travel(target_map=PALLET, edge='south') ==")
r2 = trav.travel(target_map=tv.MAP_PALLET, edge="south", max_steps=600, max_seconds=90)
print(f"== southbound travel -> {r2}  now map={tv.map_id(b)} coords={tv.coords(b)} "
      f"(fail_reason={trav.last_fail_reason})")
