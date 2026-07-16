"""recon_wallprobe.py - GROUND TRUTH for MB_IMPASSABLE_NORTH (0x32) semantics: physically step the
player across a 0x32 boundary in both directions and report which moves the game actually blocks.
Removes all guessing about LEAVE/ENTER direction sense. Picks a (x,13)<->(x,14) edge where row14 is
0x32. Reports: from north tile press SOUTH (enter the 0x32 tile)? from the 0x32 tile press NORTH
(leave it)? plus a control E/W step.
RUN: STATE=mtmoon_b2f_deep .venv\\Scripts\\python.exe -u pokemon_agent\\recon_wallprobe.py
"""
import os
import sys

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
GMAPHEADER = 0x02036DFC
NUM_PRIMARY = tv.NUM_PRIMARY


def main():
    state = os.getenv("STATE", "mtmoon_b2f_deep")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    render = lambda: None

    w = b.rd32(BL); h = b.rd32(BL + 4); mp = b.rd32(BL + 8)
    ml = b.rd32(GMAPHEADER)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))

    def beh(sx, sy):
        e = b.rd16(mp + ((sy + OFF) * w + (sx + OFF)) * 2)
        mid = e & 0x3FF
        bs, idx = (attr[0], mid) if mid < NUM_PRIMARY else (attr[1], mid - NUM_PRIMARY)
        return b.rd32(bs + idx * 4) & 0xFF

    def step(key):
        before = tv.coords(b)
        b.press(key, 8, 8, render, owner="agent")
        b.press(key, 8, 8, render, owner="agent")
        after = tv.coords(b)
        return before, after, before != after

    here = tv.coords(b)
    print(f"   [wall] start {here} beh(here)=0x{beh(*here):02x}")
    # neighbours of start with their behavior
    for d, (dx, dy) in (("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("W", (-1, 0))):
        t = (here[0] + dx, here[1] + dy)
        gw = tv.Grid(b).walkable(*t)
        print(f"   [wall]   {d} -> {t} beh=0x{beh(*t):02x} grid_walkable={gw}")

    # Try to find/reach a north-of-wall tile (x,13) whose south (x,14) is 0x32, near start.
    grid = tv.Grid(b)
    target_north = None
    for sx in range(here[0] - 6, here[0] + 7):
        if grid.walkable(sx, 13) and beh(sx, 14) == 0x32 and grid.walkable(sx, 14) is False:
            pass
    # simpler: scan whole row 13 for a walkable tile whose south is 0x32 AND collision-walkable
    cands = [sx for sx in range(grid.sx_hi + 1)
             if grid.walkable(sx, 13) and grid.walkable(sx, 14) and beh(sx, 14) == 0x32]
    print(f"   [wall] row13 tiles with 0x32 directly south (collision-walkable both): {cands}")

    # Walk east/west along row 13 to the nearest candidate, then probe S then N.
    if cands:
        tgt = min(cands, key=lambda sx: abs(sx - here[0]))
        print(f"   [wall] probing edge at x={tgt}: ({tgt},13)<->({tgt},14)[0x32]")
        # crude walk to (tgt,13): step E/W until aligned, staying on row 13 if possible
        for _ in range(40):
            c = tv.coords(b)
            if c == (tgt, 13):
                break
            if c[1] != 13:
                _, _, _ = step("UP" if c[1] > 13 else "DOWN")
                continue
            _, _, moved = step("RIGHT" if c[0] < tgt else "LEFT")
            if not moved:
                print(f"   [wall]   couldn't align, stuck at {c}")
                break
        c = tv.coords(b)
        print(f"   [wall] aligned at {c}")
        if c == (tgt, 13):
            bef, aft, moved = step("DOWN")
            print(f"   [wall] press SOUTH (enter 0x32 from north): {bef}->{aft} moved={moved}")
            if moved:
                bef, aft, moved = step("UP")
                print(f"   [wall] press NORTH (leave 0x32 to north): {bef}->{aft} moved={moved}")


if __name__ == "__main__":
    main()
