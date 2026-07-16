"""recon_b2f_elev.py - PROVE the B2F dead-end is an ELEVATION CLIFF the collision grid can't see.

The collision grid (travel.Grid) reads only collision bits (0x0C00); it ignores the metatile
ELEVATION (bits 0xF000). FireRed blocks movement between two tiles when BOTH have non-zero
elevation and they differ (IsZCoordMismatch); elevation 0 is a pass-through that inherits the
walker's current level. So a cave cliff (e.g. elev 3 floor beside elev 1 floor) reads as plain
walkable on the grid but is impassable in-game -> grid-BFS thinks (25,21)->(5,10) connects but the
traveler wedges on the cliff. This probe loads a B2F state and tests every warp-approach pair with
(a) plain collision BFS and (b) elevation-aware BFS over (tile,current-elev) state. A pair that is
plain-reachable but elev-unreachable = a cliff = the bug.
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_b2f_elev.py
"""
import os
import sys
import ctypes
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge      # noqa: E402
import travel as tv            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
OFF = tv.MAP_OFFSET
BL = tv.BACKUP_LAYOUT
GH = 0x02036DFC


def warps(b):
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    out = []
    for i in range(n):
        e = base + i * 8
        out.append((ctypes.c_int16(b.rd16(e)).value, ctypes.c_int16(b.rd16(e + 2)).value))
    return out


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    m = tv.map_id(b); start = tv.coords(b)
    grid = tv.Grid(b)
    w = b.rd32(BL); h = b.rd32(BL + 4); mp = b.rd32(BL + 8)

    elev = {}    # save-coord -> elevation (0xF000>>12)
    for by in range(h):
        for bx in range(w):
            e = b.rd16(mp + (by * w + bx) * 2)
            elev[(bx - OFF, by - OFF)] = (e & 0xF000) >> 12

    def walk(sx, sy):
        return grid.walkable(sx, sy)

    # plain collision BFS (node-walkable only)
    def reach_plain(src, dst):
        if not (walk(*src) and walk(*dst)):
            return False
        seen = {src}; q = deque([src])
        while q:
            c = q.popleft()
            if c == dst:
                return True
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (c[0] + dx, c[1] + dy)
                if n not in seen and grid.sx_lo <= n[0] <= grid.sx_hi and grid.sy_lo <= n[1] <= grid.sy_hi and walk(*n):
                    seen.add(n); q.append(n)
        return False

    # elevation-aware BFS over (tile, current-elev). Block move a->b if cur!=0 and zb!=0 and cur!=zb.
    def reach_elev(src, dst):
        if not (walk(*src) and walk(*dst)):
            return False
        z0 = elev.get(src, 0)
        seen = {(src, z0)}; q = deque([(src, z0)])
        while q:
            (c, cur) = q.popleft()
            if c == dst:
                return True
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (c[0] + dx, c[1] + dy)
                if not (grid.sx_lo <= n[0] <= grid.sx_hi and grid.sy_lo <= n[1] <= grid.sy_hi and walk(*n)):
                    continue
                zb = elev.get(n, 0)
                if cur != 0 and zb != 0 and cur != zb:
                    continue                       # elevation cliff: blocked
                ncur = zb if zb != 0 else cur
                if (n, ncur) not in seen:
                    seen.add((n, ncur)); q.append((n, ncur))
        return False

    wl = warps(b)
    print(f"   [b2felev] state={state} map={m} start={start} warps={wl}")
    print(f"   [b2felev] elevations at warp tiles: " +
          ", ".join(f"{wt}=z{elev.get(wt)}" for wt in wl))

    # pick a walkable approach for each warp
    def appr(wt):
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (wt[0] + dx, wt[1] + dy)
            if walk(*a):
                return a
        return None

    nodes = [(wt, appr(wt)) for wt in wl]
    print("   [b2felev] --- pairwise approach connectivity (plain collision  vs  elevation-aware) ---")
    for i, (wi, ai) in enumerate(nodes):
        for j, (wj, aj) in enumerate(nodes):
            if i >= j or ai is None or aj is None:
                continue
            p = reach_plain(ai, aj); ez = reach_elev(ai, aj)
            flag = "  <-- CLIFF (grid lies)" if p and not ez else ("" if p == ez else "  <-- elev MORE permissive?")
            print(f"   [b2felev]   {wi}~{wj}: plain={p} elev={ez}{flag}")


if __name__ == "__main__":
    main()
