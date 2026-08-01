"""recon_ledge_edge_cross.py — verifier for the LEDGE-OFF-EDGE seam crossing in bfs() (2026-08-01).

Live 2026-08-01 04:xx (the Cerulean<->Route 4 ping-pong): the billed-road router decided
correctly every tick ("billed leg south toward Route 5"), the south connection band computed
correctly (the Route 5 overlap row IS walkable) — and travel still returned no_path with NO
MOVEMENT, dozens of times. Root cause: Cerulean's ENTIRE south boundary row toward Route 5 is
one-way south-jump ledges. A hop over a boundary-row ledge lands at sy_hi+1 (one tile past the
playable rect, in the border buffer), and bfs()'s unconditional bound check discarded that
landing; since a ledge tile itself is never a standing tile, NO tile could satisfy the south
edge goal — the only exit read as a solid wall. head_to_gym got dead-route-pruned and the
grind/graph steers ping-ponged her west to Route 4 and back, forever.

BEFORE the fix, check [1] below fails (bfs -> None on the ledge-row south edge); the fix makes
a ledge hop whose landing is out-of-bound legal IFF the goal fires there and the tile reads
walkable. Checks, all headless on a synthetic Cerulean-shaped grid:

  1  full ledge boundary row + south edge goal -> path found, last move a 2-tile S hop to sy_hi+1
  2  wrong-way (north-jump) ledges on the same row -> still no_path (one-way stays one-way)
  3  plain walkable boundary row (no ledges)      -> paths to sy_hi by 1-tile steps, as before
  4  coord (arrive_coord-style) legs              -> never route out of the playable rect

RUN:  python3 -u recon_ledge_edge_cross.py
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import mgba.core          # noqa: F401
except ImportError:
    _root = types.ModuleType("mgba")
    for _m in ("core", "image", "vfs", "log", "png"):
        _sub = types.ModuleType(f"mgba.{_m}")
        setattr(_root, _m, _sub)
        sys.modules[f"mgba.{_m}"] = _sub
    _root.log.silence = lambda: None
    sys.modules["mgba"] = _root

from travel import bfs, direction     # noqa: E402

FAILS = []


def check(n, label, cond):
    print(f"  [{n}] {label}: {'OK' if cond else '!! FAIL'}")
    if not cond:
        FAILS.append(n)


class FakeGrid:
    """Cerulean's shape, miniature: playable rect sx 0..7 / sy 0..5, and the border-buffer
    row sy_hi+1 walkable ONLY at the connection-band columns (the neighbour map's overlap
    tiles — exactly what Grid.walkable reads live, since it bounds-checks buffer coords
    itself). Ledges are laid on by the tests; a ledge tile is never standable."""

    def __init__(self, band=(2, 3, 4)):
        self.sx_lo, self.sx_hi = 0, 7
        self.sy_lo, self.sy_hi = 0, 5
        self.band = set(band)
        self.ledges = {}                       # (sx, sy) -> one-way jump (dx, dy)

    def ledge_row_south(self, jump=(0, 1)):
        for x in range(self.sx_lo, self.sx_hi + 1):
            self.ledges[(x, self.sy_hi)] = jump
        return self

    def walkable(self, sx, sy):
        if (sx, sy) in self.ledges:
            return False                       # the game never lets you STAND on a ledge
        if self.sx_lo <= sx <= self.sx_hi and self.sy_lo <= sy <= self.sy_hi:
            return True
        return sy == self.sy_hi + 1 and sx in self.band   # the neighbour's overlap row

    def ledge_dir(self, sx, sy):
        return self.ledges.get((sx, sy))

    def edge_open(self, sx, sy, dx, dy):
        return True


START = (0, 0)


def south_edge_goal(g):
    """travel()'s _edge_goal for edge='south': at/past the boundary row, inside the band."""
    return lambda t: t[1] >= g.sy_hi and t[0] in g.band


print("== ledge-off-edge seam crossing checks (synthetic Cerulean south row) ==")

# [1] the live failure shape: the whole boundary row is south-jump ledges. The ONLY way to
#     satisfy the south edge goal is a 2-tile hop landing at sy_hi+1 (out of the playable
#     bound). Pre-fix bfs returned None here — the exact head_to_gym -> no_path wedge.
g = FakeGrid().ledge_row_south(jump=(0, 1))
path = bfs(g, START, south_edge_goal(g), walkable=g.walkable)
ok = bool(path and len(path) >= 2)
if ok:
    last, prev = path[-1], path[-2]
    ok = (last[1] == g.sy_hi + 1 and last[0] in g.band
          and (last[0] - prev[0], last[1] - prev[1]) == (0, 2)
          and direction(prev, last) == "S")            # one D-pad press hops the ledge
check(1, "south edge over a full ledge row -> 2-tile S hop lands at sy_hi+1 in-band",
      ok)

# [2] one-way stays one-way: the same row as NORTH-jump ledges must NOT be crossable south.
g = FakeGrid().ledge_row_south(jump=(0, -1))
path = bfs(g, START, south_edge_goal(g), walkable=g.walkable)
check(2, "wrong-way (north-jump) ledge row -> still no_path", path is None)

# [3] regression — a plain full-land south edge (no ledges): the goal fires ON the boundary
#     row exactly as before; 1-tile steps only, never a leak into the buffer.
g = FakeGrid()
path = bfs(g, START, south_edge_goal(g), walkable=g.walkable)
ok = bool(path and len(path) >= 2)
if ok:
    last, prev = path[-1], path[-2]
    ok = (last[1] == g.sy_hi and last[0] in g.band
          and abs(last[0] - prev[0]) + abs(last[1] - prev[1]) == 1)
check(3, "plain walkable boundary row -> paths to sy_hi by 1-tile steps (unchanged)", ok)

# [4] regression — coord legs (arrive_coord targets a tile INSIDE the playable rect) must
#     never route out of bounds, ledges present or not: the hop-off-edge exception is gated
#     on the goal firing AT the landing, which an in-rect coord goal never does.
g = FakeGrid().ledge_row_south(jump=(0, 1))
tgt = (6, 4)
path = bfs(g, START, lambda t: t == tgt, walkable=g.walkable)
ok = bool(path) and path[-1] == tgt and all(
    g.sx_lo <= x <= g.sx_hi and g.sy_lo <= y <= g.sy_hi for x, y in path)
check(4, "coord leg with ledges present -> found and every tile stays in-bounds", ok)

print("== result:", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")
sys.exit(1 if FAILS else 0)
