"""recon_vermilion_geom.py — WHY can't she route harbor->Surge gym? (shift 16)
Boots the shift-15 stage state (Vermilion City, has Cut), dumps collision + field
objects (cut trees/boulders) + BFS reachability from the harbor spawn to the gym
door approach, both OPTIMISTIC (grid.walkable, = enter_warp's pre-check) and
STATIC-BLOCKED-AWARE (= travel's real planner). READ-ONLY.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ["SDL_VIDEODRIVER"] = "dummy"
from bridge import Bridge
import travel as tv
import field_moves as fm

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = sys.argv[1] if len(sys.argv) > 1 else r"G:/temp/longrun/stage/kira_campaign.state"

b = Bridge(ROM)
with open(STATE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.run_frame()

print("map", tv.map_id(b), "coords", tv.coords(b))
grid = tv.Grid(b)
GYM_DOOR = (14, 25)
GYM_APPROACH = (14, 26)
HARBOR = (23, 33)   # where she stands after the S.S. Anne warp-in

# field objects (cut trees / boulders) on the map
trees = fm.scan_field_objects(b, {fm.GFX_CUT_TREE, fm.GFX_BOULDER})
print("\nFIELD OBJECTS (cut trees / boulders):")
for ob in trees:
    print("  ", ob.get("gfx"), ob.get("coord"))
tree_tiles = {tuple(ob["coord"]) for ob in trees}

# doors
doors = []
ml = b.rd32(0x02036DFC)
attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
w = b.rd32(tv.BACKUP_LAYOUT); h = b.rd32(tv.BACKUP_LAYOUT + 4); mp = b.rd32(tv.BACKUP_LAYOUT + 8)
for by in range(h):
    for bx in range(w):
        e = b.rd16(mp + (bx + w * by) * 2); mid = e & 0x3FF
        base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
        if 0x60 <= (b.rd32(base + idx * 4) & 0xFF) <= 0x6F:
            doors.append((bx - 7, by - 7))
print("\nDOORS:", sorted(doors))

# BFS optimistic (enter_warp pre-check)
p1 = tv.bfs(grid, HARBOR, lambda t: t == GYM_APPROACH, walkable=grid.walkable)
print(f"\nOPTIMISTIC BFS harbor{HARBOR} -> gym approach{GYM_APPROACH}: "
      f"{'PATH len ' + str(len(p1)) if p1 else 'NO PATH'}")

# BFS static-blocked-aware (trees are walls)
def walk_blocked(sx, sy):
    return grid.walkable(sx, sy) and (sx, sy) not in tree_tiles
p2 = tv.bfs(grid, HARBOR, lambda t: t == GYM_APPROACH, walkable=walk_blocked)
print(f"STATIC-BLOCKED BFS (trees=walls) harbor -> gym approach: "
      f"{'PATH len ' + str(len(p2)) if p2 else 'NO PATH (a tree/obstacle blocks)'}")

# which tree(s) does the optimistic path cross?
if p1:
    crossed = [t for t in p1 if t in tree_tiles]
    print("optimistic path crosses tree tiles:", crossed)

# ASCII of the region x in [8..28], y in [22..36]
print("\nASCII (P=path opt, T=tree, D=door, #=wall, .=floor, g=grass, @=harbor, G=gymapproach):")
pathset = set(p1 or [])
for y in range(22, 37):
    row = f"{y:2d} "
    for x in range(6, 30):
        t = (x, y)
        if t == HARBOR:
            c = "@"
        elif t == GYM_APPROACH:
            c = "G"
        elif t in tree_tiles:
            c = "T"
        elif t in doors:
            c = "D"
        elif not grid.walkable(x, y):
            c = "#"
        elif t in pathset:
            c = "P"
        elif (x + 7, y + 7) in grid.grass:
            c = "g"
        else:
            c = "."
        row += c
    print(row)
print("   " + "".join(str(x % 10) for x in range(6, 30)))
