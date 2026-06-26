"""recon_b2f_ascii.py - dump a cave floor's collision grid as ASCII to SEE where the grid lies vs
the real path. '.'=walkable, '#'=blocked(collision), digit overlay optional. Overlays warps (W),
the player (@), the known wedge point (X=37,19), and Jonny's hand-trail waypoints (o).
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_b2f_ascii.py
"""
import os
import sys
import ctypes

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


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    grid = tv.Grid(b)
    here = tv.coords(b)
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    wl = []
    for i in range(n):
        e = base + i * 8
        wl.append((ctypes.c_int16(b.rd16(e)).value, ctypes.c_int16(b.rd16(e + 2)).value))

    # Jonny's approximate B2F trail waypoints (human-read, +-5)
    trail = [(8, 36), (14, 19), (14, 11), (17, 11), (13, 8)]
    wedge = (37, 19)

    sx_hi, sy_hi = grid.sx_hi, grid.sy_hi
    print(f"   [ascii] state={state} map={tv.map_id(b)} here={here} size sx0..{sx_hi} sy0..{sy_hi}")
    print(f"   [ascii] warps={wl}  wedge(X)={wedge}  trail(o)={trail}")
    # column header (tens / ones)
    hdr1 = "    " + "".join(str((x // 10) % 10) for x in range(sx_hi + 1))
    hdr2 = "    " + "".join(str(x % 10) for x in range(sx_hi + 1))
    print(hdr1); print(hdr2)
    for sy in range(sy_hi + 1):
        row = []
        for sx in range(sx_hi + 1):
            t = (sx, sy)
            if t == here:
                ch = "@"
            elif t in wl:
                ch = "W"
            elif t == wedge:
                ch = "X"
            elif t in trail:
                ch = "o"
            elif grid.walkable(sx, sy):
                ch = "." if grid.walkable_safe(sx, sy) else ","
            else:
                ch = "#"
            row.append(ch)
        print(f"{sy:>3} " + "".join(row))


if __name__ == "__main__":
    main()
