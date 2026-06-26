"""recon_zonemap.py - READ-ONLY: from a loaded cave state, dump the live warp table (cross-checked
vs the pokefirered disasm) and compute, for each warp, whether its approach tile is BFS-reachable
from the player's current position (a) normally and (b) AVOIDING every other warp tile.

Why: the Mt Moon route failed not on a dead-end but because walking toward a far warp CROSSES a
nearer warp tile and triggers it early (start->(5,6) actually fired (19,14)->B1F(25,4) trap). This
probe measures, purely on the collision grid, which warps the start zone can reach CLEANLY (without
crossing another warp). No emulation stepping, no map changes - pure BFS. State via env STATE.
RUN: STATE=mtmoon_interior .venv\\Scripts\\python.exe -u pokemon_agent\\recon_zonemap.py
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
GH = 0x02036DFC

# disasm ground truth (pret/pokefirered) keyed by game map id (group,num): 1F=(1,1) B1F=(1,2) B2F=(1,3)
DISASM = {
    (1, 1): [((5, 6), (1, 2), 0), ((19, 14), (1, 2), 1), ((31, 16), (1, 2), 2), ((18, 37), (3, 22), 0)],
    (1, 2): [((3, 3), (1, 1), 0), ((25, 4), (1, 1), 1), ((43, 21), (1, 1), 2), ((22, 18), (1, 3), 0),
             ((17, 5), (1, 3), 1), ((26, 36), (1, 3), 2), ((39, 4), (1, 3), 3), ((45, 4), (3, 22), 1)],
    (1, 3): [((25, 21), (1, 2), 3), ((31, 11), (1, 2), 4), ((17, 31), (1, 2), 5), ((5, 10), (1, 2), 6)],
}


def warps(b):
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    out = []
    for i in range(n):
        e = base + i * 8
        x = ctypes.c_int16(b.rd16(e)).value
        y = ctypes.c_int16(b.rd16(e + 2)).value
        warp_id = b.rd8(e + 5)
        dest = (b.rd8(e + 7), b.rd8(e + 6))   # (group, num)
        out.append(((x, y), dest, warp_id))
    return out


def main():
    state = os.getenv("STATE", "mtmoon_interior")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    m = tv.map_id(b)
    start = tv.coords(b)
    grid = tv.Grid(b)
    wl = warps(b)
    warp_tiles = {w[0] for w in wl}
    print(f"   [zonemap] state={state} map={m} start={start} ({len(wl)} warps)")
    print(f"   [zonemap] disasm match: {'YES' if [(w[0], w[1], w[2]) for w in wl] == DISASM.get(m) else 'NO / MISMATCH'}")

    def reach(appr, avoid):
        if not grid.walkable(*appr):
            return False
        if appr == start:
            return True
        walk = (lambda sx, sy: grid.walkable(sx, sy) and (sx, sy) not in avoid)
        return bool(tv.bfs(grid, start, (lambda a: lambda t: t == a)(appr), walkable=walk))

    print(f"   [zonemap] --- per-warp reachability from {start} ---")
    for (x, y), dest, wid in wl:
        appr = [(x + dx, y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)) if grid.walkable(x + dx, y + dy)]
        others = warp_tiles - {(x, y)}
        norm = any(reach(a, set()) for a in appr)
        clean = any(reach(a, others) for a in appr)
        tag = "TRAP-OK" if norm and not clean else ("CLEAN" if clean else "UNREACH")
        dis = next((d for d in DISASM.get(m, []) if d[0] == (x, y)), None)
        disn = f" disasm->{dis[1]}#{dis[2]}" if dis else ""
        print(f"   [zonemap]   warp ({x:>2},{y:>2}) -> dest{dest} id={wid}{disn}  "
              f"approaches={appr}  normal={norm} clean(avoid-warps)={clean}  [{tag}]")


if __name__ == "__main__":
    main()
