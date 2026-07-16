"""recon_bandguard_verify.py — verify the band-aware heal-safe predicate at the leveled pocket state.
(107,12) must now read heal-safe TRUE (east band to Cerulean reachable). Mirrors campaign's logic."""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CITY_PC_DOORS = {(1, 4): (21, 17), (2, 0): (16, 9), (3, 3): (22, 19), (3, 22): (12, 5)}
ROUTE3 = (3, 21)


def map_connections(b):
    ch = b.rd32(0x02036DFC + 0x0C)
    if ch < 0x02000000:
        return []
    cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
    if not (0 < cnt < 16) or arr < 0x02000000:
        return []
    dirn = {1: "S", 2: "N", 3: "W", 4: "E"}
    return [(dirn[d], (b.rd8(c + 0x08), b.rd8(c + 0x09)))
            for i in range(cnt) for c in (arr + i * 0xC,) for d in (b.rd8(c),) if d in dirn]


def edge_band_reachable(grid, here, edge):
    if edge == "east":
        line, past_d, axis = grid.sx_hi, 1, 0
    elif edge == "west":
        line, past_d, axis = grid.sx_lo, -1, 0
    elif edge == "north":
        line, past_d, axis = grid.sy_lo, -1, 1
    else:
        line, past_d, axis = grid.sy_hi, 1, 1
    if axis == 0:
        band = {p for p in range(grid.sy_lo, grid.sy_hi + 1) if grid.walkable(line + past_d, p)}
        goal = lambda t: t[0] == line and t[1] in band
    else:
        band = {p for p in range(grid.sx_lo, grid.sx_hi + 1) if grid.walkable(p, line + past_d)}
        goal = lambda t: t[1] == line and t[0] in band
    if not band:
        return False
    return bool(tv.bfs(grid, here, goal, walkable=grid.walkable))


def heal_safe(b):
    m = tv.map_id(b)
    here = tv.coords(b)
    if here is None:
        return False, "no-coords"
    if m == ROUTE3:
        return True, "route3"
    grid = tv.Grid(b)
    door = CITY_PC_DOORS.get(m)
    if door is not None:
        appr = (door[0], door[1] + 1)
        if tv.bfs(grid, here, lambda t, a=appr: t == a, walkable=grid.walkable):
            return True, f"own-center {door}"
    _EDGE = {"N": "north", "S": "south", "E": "east", "W": "west"}
    for d, nbr in map_connections(b):
        if nbr in CITY_PC_DOORS and nbr != m and d in _EDGE and edge_band_reachable(grid, here, _EDGE[d]):
            return True, f"band-crossable {_EDGE[d]} -> {nbr}"
    return False, "no heal path"


SAVE = sorted(glob.glob(os.path.join(_HERE, "states", "campaign", "pre_reload_*.state")))[-1]
b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")
ok, why = heal_safe(b)
print(f"POCKET {tv.map_id(b)}@{tv.coords(b)} -> heal_safe={ok} ({why})   [expect TRUE via east band]")
print("PASS" if ok else "FAIL")
