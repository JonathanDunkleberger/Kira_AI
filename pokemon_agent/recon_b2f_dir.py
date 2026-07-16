"""recon_b2f_dir.py - VERIFY the directional-impassable (MB_IMPASSABLE_*) edge rule reproduces the
real B2F wall before wiring it into the engine. Edge A->B by (dx,dy) blocked if source carries the
leave-block behavior for that dir or dest carries the enter-block behavior. Expectation: with the
rule, (25,21)[lower]~(5,10)[upper] flips True->False (the real dead-end) while (31,11)[upper]~(5,10)
stays True (the real route). Run for B1F too to confirm (39,4)~(45,4) holds.
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_b2f_dir.py
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
GMAPHEADER = 0x02036DFC
NUM_PRIMARY = tv.NUM_PRIMARY

# behavior on the SOURCE tile blocks LEAVING in these (dx,dy)
LEAVE = {0x30: {(1, 0)}, 0x31: {(-1, 0)}, 0x32: {(0, -1)}, 0x33: {(0, 1)},
         0x34: {(0, -1), (1, 0)}, 0x35: {(0, -1), (-1, 0)},
         0x36: {(0, 1), (1, 0)}, 0x37: {(0, 1), (-1, 0)}}
# behavior on the DEST tile blocks ENTERING via these (dx,dy) (you enter from the opposite side)
ENTER = {0x30: {(-1, 0)}, 0x31: {(1, 0)}, 0x32: {(0, 1)}, 0x33: {(0, -1)},
         0x34: {(0, 1), (-1, 0)}, 0x35: {(0, 1), (1, 0)},
         0x36: {(0, -1), (-1, 0)}, 0x37: {(0, -1), (1, 0)}}


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    grid = tv.Grid(b)
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

    def edge_open(a, dxy):
        bA = behav.get(a)
        bn = (a[0] + dxy[0], a[1] + dxy[1])
        bB = behav.get(bn)
        if dxy in LEAVE.get(bA, ()) or dxy in ENTER.get(bB, ()):
            return False
        return True

    def reach(src, dst, dir_aware):
        if not (grid.walkable(*src) and grid.walkable(*dst)):
            return False
        seen = {src}; q = deque([src])
        while q:
            c = q.popleft()
            if c == dst:
                return True
            for dxy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (c[0] + dxy[0], c[1] + dxy[1])
                if not (grid.sx_lo <= n[0] <= grid.sx_hi and grid.sy_lo <= n[1] <= grid.sy_hi):
                    continue
                if n in seen or not grid.walkable(*n):
                    continue
                if dir_aware and not edge_open(c, dxy):
                    continue
                seen.add(n); q.append(n)
        return False

    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    wl = [(ctypes.c_int16(b.rd16(base + i * 8)).value, ctypes.c_int16(b.rd16(base + i * 8 + 2)).value)
          for i in range(n)]

    def appr(wt):
        for dxy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (wt[0] + dxy[0], wt[1] + dxy[1])
            if grid.walkable(*a):
                return a
        return None

    print(f"   [dir] state={state} map={tv.map_id(b)} here={tv.coords(b)} warps={wl}")
    nodes = [(wt, appr(wt)) for wt in wl]
    for i, (wi, ai) in enumerate(nodes):
        for j, (wj, aj) in enumerate(nodes):
            if i >= j or ai is None or aj is None:
                continue
            p = reach(ai, aj, False); d = reach(ai, aj, True)
            flag = "  <-- WALL (dir-rule blocks; grid lied)" if p and not d else ""
            print(f"   [dir]   {wi}~{wj}: grid-only={p} dir-aware={d}{flag}")


if __name__ == "__main__":
    main()
