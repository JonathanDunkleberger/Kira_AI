"""recon_cerulean_routes.py — (b) FRLG route ground-truth: is she pocketed, or is the cut tree the real gate?

For clean (seg_cerulean) + live (kira_campaign) Cerulean states: read the live map-edge CONNECTIONS
(which route each edge leads to), then flood-reach from the player treating the cut-tree tile (26,32) as
PERMANENTLY blocked (it is, until Cut sets its flag). Report which of the 4 city edges are reachable
WITHOUT cutting the tree, and whether the south (Route 5) edge specifically needs the tree. This decides:
mis-positioned pocket vs genuine Cut-gate. Read-only.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402
from travel import Grid            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ST = os.path.join(_HERE, "states")
TREE = (26, 32)
DIRN = {1: "S", 2: "N", 3: "W", 4: "E"}

CANDIDATES = ["kira/seg_cerulean.state", "campaign/kira_campaign.state",
              "archive/cerulean_entered_auto.state"]


def connections(b):
    try:
        ch = b.rd32(0x02036DFC + 0x0C)
        if ch < 0x02000000:
            return []
        cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
        if not (0 < cnt < 16) or arr < 0x02000000:
            return []
        return [(DIRN[d], (b.rd8(c + 0x08), b.rd8(c + 0x09)))
                for i in range(cnt) for c in (arr + i * 0xC,) for d in (b.rd8(c),) if d in DIRN]
    except Exception as e:
        return [("ERR", str(e))]


def flood(grid, start, block):
    seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n not in seen and n not in block and 0 <= n[0] <= grid.sx_hi \
                    and 0 <= n[1] <= grid.sy_hi and grid.walkable(*n):
                seen.add(n); stack.append(n)
    return seen


def edges_reached(grid, seen):
    out = {}
    out["N(row0)"] = sorted(x for x in range(grid.sx_hi + 1) if (x, 0) in seen)
    out["S(row%d)" % grid.sy_hi] = sorted(x for x in range(grid.sx_hi + 1) if (x, grid.sy_hi) in seen)
    out["W(col0)"] = sorted(y for y in range(grid.sy_hi + 1) if (0, y) in seen)
    out["E(col%d)" % grid.sx_hi] = sorted(y for y in range(grid.sy_hi + 1) if (grid.sx_hi, y) in seen)
    return out


def main():
    b = Bridge(ROM)
    for rel in CANDIDATES:
        p = os.path.join(ST, rel)
        if not os.path.exists(p):
            print(f"\n>>> {rel}  (missing)", flush=True)
            continue
        b.load_state(open(p, "rb").read())
        for _ in range(90):
            b.run_frame()
        cur = tv.coords(b)
        m = tv.map_id(b)
        print(f"\n>>> {rel}  map={m} player={cur}", flush=True)
        print(f"   edge connections (live): {connections(b)}", flush=True)
        if cur is None or m != (3, 3):
            print("   (not standing in Cerulean / no coords — skipping flood)", flush=True)
            continue
        grid = Grid(b)
        seen_notree = flood(grid, cur, {TREE})
        seen_cut = flood(grid, cur, set())
        e_nt = edges_reached(grid, seen_notree)
        e_ct = edges_reached(grid, seen_cut)
        print(f"   reachable WITHOUT cutting tree ({len(seen_notree)} tiles):", flush=True)
        for k, v in e_nt.items():
            print(f"        {k}: {v if v else 'NONE'}", flush=True)
        print(f"   reachable IF tree cut ({len(seen_cut)} tiles):", flush=True)
        for k, v in e_ct.items():
            print(f"        {k}: {v if v else 'NONE'}", flush=True)
        s_key = [k for k in e_nt if k.startswith("S")][0]
        gate = (not e_nt[s_key]) and bool(e_ct[s_key])
        print(f"   >> SOUTH (Route 5) edge needs the tree CUT? {gate}", flush=True)


if __name__ == "__main__":
    main()
