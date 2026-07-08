"""recon_route23_bfs_probe.py — night shift 13: the VICTORY re-grade FAILed with 153 travel
wedges, ALL on Route 23 (3,42): every northbound leg from the south half (18,74)/(18,66)/
(10,62)… reads no_route AT ITS OWN START ("no clean path AND no NPC-allowing path"), yet she
WALKED south through the same ground minutes earlier. Route 23 is ledge+water terrain — the
southbound one-way ledges are correct, so the question is what the grid thinks the NORTHBOUND
return is (the real game allows it: surf up the water / walk the gaps). Dump the terrain the
planner sees + reproduce the exact three BFS layers from the wedge tiles."""
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
statef = os.path.join(LONGRUN, "banked_VICTORY", "kira_campaign.state")
b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()
print(f"spawn: map={tv.map_id(b)} coords={tv.coords(b)}")

# get onto Route 23 (3,42). banked_VICTORY spawns near Indigo/VR; Indigo's south edge drops
# onto Route 23's north band (cols 10-12). Walk south until the map flips.
for i in range(60):
    if tuple(tv.map_id(b)) == (3, 42):
        break
    m = tuple(tv.map_id(b))
    if m == (3, 9):                      # Indigo Plateau: head for col 11 then south
        x, y = tv.coords(b)
        key = "DOWN" if x in (10, 11, 12) else ("RIGHT" if x < 11 else "LEFT")
    else:
        key = "DOWN"
    b.press(key, 8, 10)
    for _ in range(30):
        b.run_frame()
if tuple(tv.map_id(b)) != (3, 42):
    print(f"!! could not reach Route 23 — stuck on {tv.map_id(b)} at {tv.coords(b)} — abort")
    sys.exit(1)
print(f"on Route 23: coords={tv.coords(b)}")

grid = tv.Grid(b)
print(f"grid dims: sx {grid.sx_lo}..{grid.sx_hi}  sy {grid.sy_lo}..{grid.sy_hi}  "
      f"water={len(grid.water)} ledges={len(grid.ledge)} impass={len(grid.impass)} "
      f"spin={len(grid.spin)}  can_surf={fm.can_use(b, 'surf')}")

# ── the terrain the planner sees, full map (Route 23 is 20-ish wide — cheap to dump) ──────
# chars: '~'=surfable water  'G'=grass  '^v<>'=ledge (jump dir)  '#'=closed  '.'=walkable
# 'I'=directional impassable  digits=elevation of walkable tile when != mode (spotting rifts)
LEDGE_CH = {(0, -1): "^", (0, 1): "v", (-1, 0): "<", (1, 0): ">"}
OFF = tv.MAP_OFFSET
for y in range(grid.sy_lo, grid.sy_hi + 1):
    row = []
    for x in range(grid.sx_lo, grid.sx_hi + 1):
        bx, by = x + OFF, y + OFF
        if (bx, by) in grid.water:
            row.append("~")
        elif (bx, by) in grid.ledge:
            row.append(LEDGE_CH.get(grid.ledge[(bx, by)], "L"))
        elif (bx, by) in grid.impass:
            row.append("I")
        elif grid.col.get((bx, by), 1) == 0:
            row.append("G" if (bx, by) in grid.grass else ".")
        else:
            row.append("#")
    print(f"y={y:3d} " + "".join(row))

# ── classify the run's wedge tiles ────────────────────────────────────────────────────────
for t in [(18, 74), (18, 66), (18, 76), (10, 62), (10, 63), (14, 67), (13, 43), (15, 44),
          (12, 0), (21, 76)]:
    bx, by = t[0] + OFF, t[1] + OFF
    print(f"tile {t}: col={grid.col.get((bx, by))} elev={grid.elev.get((bx, by))} "
          f"water={grid.is_water(*t)} grass={(bx, by) in grid.grass} "
          f"ledge={grid.ledge.get((bx, by))} impass={grid.impass.get((bx, by))}")
    for d, dxdy in (("E", (1, 0)), ("W", (-1, 0)), ("S", (0, 1)), ("N", (0, -1))):
        if not grid.edge_open(t[0], t[1], *dxdy):
            print(f"   edge {t}->{d}: CLOSED")

# ── reproduce the exact planner ladder from the hot wedge tile, northbound ────────────────
goal = lambda t: t[1] <= 0
for name, layer in (("walkable_safe", grid.walkable_safe), ("walkable", grid.walkable),
                    ("walkable_or_surf", grid.walkable_or_surf)):
    p = tv.bfs(grid, (18, 74), goal, walkable=layer)
    print(f"north BFS from (18,74) [{name}]: {'PATH len ' + str(len(p)) if p else 'NO PATH'}")

# ── flood from (18,74) with the widest layer (or_surf) + REAL bfs edges (ledges, impass,
#    elevation law): how far north can the planner get at all, and what is the wall? ───────
start = (18, 74)
seen = {start}
frontier = [start]
min_y = start[1]
while frontier:
    nf = []
    for (cx, cy) in frontier:
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if grid.ledge_dir(cx + dx, cy + dy) == (dx, dy):
                nx, ny = cx + 2 * dx, cy + 2 * dy
            else:
                nx, ny = cx + dx, cy + dy
                if not grid.edge_open(cx, cy, dx, dy):
                    continue
            if not (grid.sx_lo <= nx <= grid.sx_hi and grid.sy_lo <= ny <= grid.sy_hi):
                continue
            t = (nx, ny)
            if t in seen or not grid.walkable_or_surf(nx, ny):
                continue
            seen.add(t)
            nf.append(t)
            min_y = min(min_y, ny)
    frontier = nf
print(f"or_surf flood from (18,74): {len(seen)} tiles, northmost row y={min_y}")
wallrow = sorted(t for t in seen if t[1] == min_y)
print(f"  northmost tiles reached: {wallrow[:14]}")
for (x, y) in wallrow[:8]:
    t2 = (x, y - 1)
    bx, by = t2[0] + OFF, t2[1] + OFF
    print(f"  above {(x, y)} -> {t2}: col={grid.col.get((bx, by))} elev={grid.elev.get((bx, by))} "
          f"water={grid.is_water(*t2)} ledge={grid.ledge.get((bx, by))} "
          f"impass={grid.impass.get((bx, by))} edgeN_open={grid.edge_open(x, y, 0, -1)}")
