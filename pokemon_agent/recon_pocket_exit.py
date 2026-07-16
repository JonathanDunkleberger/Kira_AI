"""recon_pocket_exit.py — GROUND TRUTH on the Route-4 east pocket exit. From the leveled pre_reload
state at (107,12): grid bounds, the E-connection offset to Cerulean (which border rows actually cross),
the BFS-reachable set (how far east she can walk), and an ASCII map of the east region. Pure reads."""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = sorted(glob.glob(os.path.join(_HERE, "states", "campaign", "pre_reload_*.state")))[-1]
print(f"SAVE={os.path.basename(SAVE)}")

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")

here = tv.coords(b)
m = tv.map_id(b)
grid = tv.Grid(b)
print(f"map={m} here={here} playable x:[{grid.sx_lo}..{grid.sx_hi}] y:[{grid.sy_lo}..{grid.sy_hi}]")

# connections WITH offsets (conn: +0 dir, +4 s32 offset, +8 grp, +9 num)
ch = b.rd32(0x02036DFC + 0x0C)
cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
dirn = {1: "S", 2: "N", 3: "W", 4: "E"}
for i in range(cnt):
    c = arr + i * 0xC
    d = b.rd8(c)
    off = b.rd32(c + 4)
    if off >= 0x80000000:
        off -= 0x100000000
    print(f"conn[{i}] dir={dirn.get(d, d)} offset={off} -> ({b.rd8(c+8)},{b.rd8(c+9)})")

# reachable set from here
seen = set()
tv.bfs(grid, here, lambda t: (seen.add(t) or False), walkable=grid.walkable)
xs = [t[0] for t in seen]
ys = [t[1] for t in seen]
print(f"reachable tiles={len(seen)} x:[{min(xs)}..{max(xs)}] y:[{min(ys)}..{max(ys)}]")
# which EAST-border tiles are reachable (candidate crossing rows)?
border = sorted(t for t in seen if t[0] >= grid.sx_hi)
print(f"reachable EAST-border (x={grid.sx_hi}) tiles: {border}")
near = sorted(t for t in seen if t[0] >= grid.sx_hi - 1)
print(f"reachable x>={grid.sx_hi - 1} tiles: {near}")

# ASCII map of the east region: x 96..sx_hi, y 6..22  (# wall, . walk, L ledge, G grass, P player, ~ water)
off7 = tv.MAP_OFFSET
x0, x1 = 96, grid.sx_hi
y0, y1 = 6, 22
print(f"map x:[{x0}..{x1}] y:[{y0}..{y1}]  (#=blocked .=walk L=ledge G=grass P=here *=reachable)")
for y in range(y0, y1 + 1):
    row = ""
    for x in range(x0, x1 + 1):
        bx, by = x + off7, y + off7
        if (x, y) == here:
            row += "P"; continue
        ch2 = "#"
        if grid.walkable(x, y):
            ch2 = "." if (x, y) in seen else "o"       # o = walkable but NOT reachable from here
        if (bx, by) in grid.ledge:
            dx, dy = grid.ledge[(bx, by)]
            ch2 = {(0, 1): "v", (0, -1): "^", (1, 0): ">", (-1, 0): "<"}.get((dx, dy), "L")
        elif (bx, by) in grid.grass and grid.walkable(x, y):
            ch2 = "G" if (x, y) in seen else "g"
        row += ch2
    print(f"y={y:3d} {row}")
print("DONE")
