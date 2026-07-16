"""recon_b2f_behav.py - find the HIDDEN walls: tiles that are collision-walkable but carry a cave
cliff/drop BEHAVIOR (handoff: 0x08 / 0x32) the Grid ignores. Overlays them on the ASCII map and
re-tests warp connectivity treating them as blocked. If (25,21)~(5,10) flips True->False AND the
walls visually separate the regions, the fix is: add these behaviors to Grid blocking.
Also DUMPS the histogram of behaviors on collision-0 tiles so we don't guess which to block.
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_b2f_behav.py
"""
import os
import sys
import ctypes
from collections import deque, Counter

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
GMAPHEADER = 0x02036DFC
NUM_PRIMARY = tv.NUM_PRIMARY
BLOCK_BEHAVIORS = {0x08, 0x32}    # handoff: B2F cliff/drop tiles, collision-0 but impassable


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    grid = tv.Grid(b)
    here = tv.coords(b)
    w = b.rd32(BL); h = b.rd32(BL + 4); mp = b.rd32(BL + 8)
    ml = b.rd32(GMAPHEADER)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))

    behav = {}
    for by in range(h):
        for bx in range(w):
            e = b.rd16(mp + (by * w + bx) * 2)
            mid = e & 0x3FF
            bs, idx = (attr[0], mid) if mid < NUM_PRIMARY else (attr[1], mid - NUM_PRIMARY)
            behav[(bx - OFF, by - OFF)] = b.rd32(bs + idx * 4) & 0xFF

    # histogram of behaviors over collision-0 (walkable) tiles only
    hist = Counter(behav[(sx, sy)] for sx in range(grid.sx_hi + 1) for sy in range(grid.sy_hi + 1)
                   if grid.walkable(sx, sy))
    print(f"   [behav] state={state} here={here}")
    print("   [behav] behavior histogram over WALKABLE tiles: " +
          ", ".join(f"0x{k:02x}:{v}" for k, v in sorted(hist.items())))

    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    wl = [(ctypes.c_int16(b.rd16(base + i * 8)).value, ctypes.c_int16(b.rd16(base + i * 8 + 2)).value)
          for i in range(n)]

    def blk(sx, sy):
        return behav.get((sx, sy)) in BLOCK_BEHAVIORS

    def reach(src, dst, use_behav):
        if not (grid.walkable(*src) and grid.walkable(*dst)):
            return False
        seen = {src}; q = deque([src])
        while q:
            c = q.popleft()
            if c == dst:
                return True
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = c[0] + dx, c[1] + dy
                if not (grid.sx_lo <= nx <= grid.sx_hi and grid.sy_lo <= ny <= grid.sy_hi):
                    continue
                if (nx, ny) in seen or not grid.walkable(nx, ny):
                    continue
                if use_behav and blk(nx, ny):
                    continue
                seen.add((nx, ny)); q.append((nx, ny))
        return False

    def appr(wt):
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (wt[0] + dx, wt[1] + dy)
            if grid.walkable(*a) and not blk(*a):
                return a
        return None

    print(f"   [behav] warps={wl}  (blocking behaviors {[hex(x) for x in BLOCK_BEHAVIORS]})")
    nodes = [(wt, appr(wt)) for wt in wl]
    for i, (wi, ai) in enumerate(nodes):
        for j, (wj, aj) in enumerate(nodes):
            if i >= j or ai is None or aj is None:
                continue
            p = reach(ai, aj, False); bb = reach(ai, aj, True)
            flag = "  <-- WALL found (grid lies, behav fixes)" if p and not bb else ""
            print(f"   [behav]   {wi}~{wj}: collision-only={p} behav-aware={bb}{flag}")

    # ASCII with cliff overlay: 'v'=0x08 'c'=0x32 (collision-walkable but cliff)
    print("   [behav] map: .=walk #=wall W=warp @=you v=behav0x08 c=behav0x32")
    print("    " + "".join(str((x // 10) % 10) for x in range(grid.sx_hi + 1)))
    print("    " + "".join(str(x % 10) for x in range(grid.sx_hi + 1)))
    for sy in range(grid.sy_hi + 1):
        row = []
        for sx in range(grid.sx_hi + 1):
            t = (sx, sy)
            if t == here:
                ch = "@"
            elif t in wl:
                ch = "W"
            elif not grid.walkable(sx, sy):
                ch = "#"
            elif behav.get(t) == 0x08:
                ch = "v"
            elif behav.get(t) == 0x32:
                ch = "c"
            else:
                ch = "."
            row.append(ch)
        print(f"{sy:>3} " + "".join(row))


if __name__ == "__main__":
    main()
