"""recon_strandguard_verify.py — validate the _center_reachable_here geometry BEFORE wiring it in.
Must be FALSE at the Route-4 pocket (strand) and TRUE at Cerulean Center (safe). Pure reads."""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ROUTE3 = (3, 21)
ROUTE4_PC_DOOR = (12, 5)
CITY_PC_DOORS = {(1, 4): (21, 17), (2, 0): (16, 9), (3, 0): (25, 20), (3, 22): ROUTE4_PC_DOOR}
# (approx doors; only Route4 (3,22) + Cerulean (3,3) matter here — Cerulean door added below)
CERULEAN = (3, 3)
CITY_PC_DOORS[CERULEAN] = (29, 28)   # from §0 (7,7)/door(29,28)


def map_connections(b):
    try:
        ch = b.rd32(0x02036DFC + 0x0C)
        if ch < 0x02000000:
            return []
        cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
        if not (0 < cnt < 16) or arr < 0x02000000:
            return []
        dirn = {1: "S", 2: "N", 3: "W", 4: "E"}
        return [(dirn[d], (b.rd8(c + 0x08), b.rd8(c + 0x09)))
                for i in range(cnt) for c in (arr + i * 0xC,) for d in (b.rd8(c),) if d in dirn]
    except Exception:
        return []


def edge_reachable(grid, here, direction):
    lo_x, lo_y, hi_x, hi_y = grid.sx_lo, grid.sy_lo, grid.sx_hi, grid.sy_hi
    if direction == "east":
        goal = lambda t: t[0] >= hi_x
    elif direction == "west":
        goal = lambda t: t[0] <= lo_x
    elif direction == "north":
        goal = lambda t: t[1] <= lo_y
    else:
        goal = lambda t: t[1] >= hi_y
    return bool(tv.bfs(grid, here, goal, walkable=grid.walkable))


def center_reachable(b):
    m = tv.map_id(b)
    here = tv.coords(b)
    if here is None:
        return False, "no-coords"
    if m == ROUTE3:
        return True, "route3-west-pewter"
    grid = tv.Grid(b)
    door = CITY_PC_DOORS.get(m)
    if door is not None:
        appr = (door[0], door[1] + 1)
        if tv.bfs(grid, here, lambda t, a=appr: t == a, walkable=grid.walkable):
            return True, f"own-center {door}"
    return False, f"no own-center reachable (conns={map_connections(b)})"


def probe(save, label):
    b = Bridge(ROM)
    with open(save, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.press("B", 2, 2, None, owner="agent")
    ok, why = center_reachable(b)
    print(f"{label}: map={tv.map_id(b)} here={tv.coords(b)} -> center_reachable={ok}  ({why})")
    return ok


pocket = sorted(glob.glob(os.path.join(_HERE, "states", "campaign", "pre_reload_*.state")))[-1]
canon = os.path.join(_HERE, "states", "campaign", "kira_campaign.state")
r_pocket = probe(pocket, "POCKET (expect FALSE)")
r_canon = probe(canon, "CERULEAN canonical (expect TRUE)")
print(f"VERDICT: {'PASS' if (not r_pocket and r_canon) else 'FAIL'}")
