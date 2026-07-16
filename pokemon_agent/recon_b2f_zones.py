"""recon_b2f_zones.py - the TRUE executable zone partition of a cave floor, under the executor's
real movement model: collision + DIRECTIONAL impassables (cave cliffs) + warp-avoidance (you may
not walk THROUGH another warp tile, it would teleport you). Flood-fills connected components over
walkable & edge-open & non-warp tiles, then labels each warp's approach tile with its component id.
Two warps are cleanly walkable between each other IFF their approaches share a component.
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_b2f_zones.py
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
GH = 0x02036DFC


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    grid = tv.Grid(b)
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    wl = [(ctypes.c_int16(b.rd16(base + i * 8)).value, ctypes.c_int16(b.rd16(base + i * 8 + 2)).value)
          for i in range(n)]
    warpset = set(wl)

    def passable(sx, sy):
        return grid.walkable(sx, sy) and (sx, sy) not in warpset

    # connected components over passable tiles with directional edges
    comp = {}
    cid = 0
    for sy in range(grid.sy_hi + 1):
        for sx in range(grid.sx_hi + 1):
            if (sx, sy) in comp or not passable(sx, sy):
                continue
            cid += 1
            comp[(sx, sy)] = cid
            q = deque([(sx, sy)])
            while q:
                c = q.popleft()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nb = (c[0] + dx, c[1] + dy)
                    if (not (grid.sx_lo <= nb[0] <= grid.sx_hi and grid.sy_lo <= nb[1] <= grid.sy_hi)
                            or nb in comp or not passable(*nb) or not grid.edge_open(c[0], c[1], dx, dy)):
                        continue
                    comp[nb] = cid
                    q.append(nb)

    print(f"   [zones] state={state} map={tv.map_id(b)} here={tv.coords(b)} components={cid} "
          f"impass_tiles={len(grid.impass)}")
    w = b.rd32(tv.BACKUP_LAYOUT); mp = b.rd32(tv.BACKUP_LAYOUT + 8)
    ml = b.rd32(0x02036DFC)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))

    def raw(sx, sy):
        bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
        e = b.rd16(mp + (by * w + bx) * 2)
        col = (e & 0x0C00) >> 10
        mid = e & 0x3FF
        bs, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1], mid - tv.NUM_PRIMARY)
        bh = b.rd32(bs + idx * 4) & 0xFF
        return col, bh
    for q in [(13, 17), (13, 16), (14, 17), (12, 17), (13, 18), (5, 11), (4, 10), (12, 22)]:
        col, bh = raw(*q)
        print(f"   [zones] cell {q}: comp={comp.get(q)} walkable={grid.walkable(*q)} "
              f"collision={col} behavior=0x{bh:02x} warp={q in warpset}")
    for wt in wl:
        comps = set()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (wt[0] + dx, wt[1] + dy)
            if a in comp:
                comps.add(comp[a])
        print(f"   [zones] warp {wt}: approach-components={sorted(comps) or 'NONE-walled-in'}")


if __name__ == "__main__":
    main()
