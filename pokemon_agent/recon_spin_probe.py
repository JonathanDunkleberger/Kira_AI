"""recon_spin_probe.py — READ-ONLY diagnosis: why does BFS read no_route across the
Rocket Hideout spin floor (map (1,45), the banked_SCOPE 412-wedge burst, shift 5)?
Hypothesis: spin tiles carry a different metatile ELEVATION than the floor, and the
shift-4 per-edge elevation law in Grid.edge_open now walls them off for BFS.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_SCOPE")

b = Bridge(ROM)
with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

print(f"map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
g = tv.Grid(b)
OFF = tv.MAP_OFFSET

# classify spin tiles + their elevations vs surrounding floor
ml = b.rd32(tv.GMAPHEADER)
attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
mp = b.rd32(tv.BACKUP_LAYOUT + 8)
spins, stops = {}, {}
for by in range(g.h):
    for bx in range(g.w):
        e = b.rd16(mp + (by * g.w + bx) * 2)
        mid = e & 0x3FF
        base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1], mid - tv.NUM_PRIMARY)
        bh = b.rd32(base + idx * 4) & 0xFF
        if 0x54 <= bh <= 0x57:
            spins[(bx - OFF, by - OFF)] = ((e >> 12) & 0xF, hex(bh))
        elif bh == 0x58:
            stops[(bx - OFF, by - OFF)] = (e >> 12) & 0xF

print(f"spin tiles: {len(spins)}, stop tiles: {len(stops)}", flush=True)
if spins:
    sample = list(spins.items())[:12]
    print("spin sample (savecoord -> (elev, behavior)):", sample, flush=True)
    elevs = {v[0] for v in spins.values()}
    print(f"spin elevations seen: {elevs}", flush=True)

cur = tuple(tv.coords(b))
floor_e = g.elev.get((cur[0] + OFF, cur[1] + OFF))
print(f"floor elevation at {cur}: {floor_e}", flush=True)

for goal in ((11, 15), (11, 16), (10, 15)):
    p1 = tv.bfs(g, cur, lambda t, gg=goal: t == gg)
    # same BFS but with edge_open neutralized (pre-shift-4 behavior)
    class G2:
        def __getattr__(self, k):
            return getattr(g, k)
        def edge_open(self, *a):
            return True
        def ledge_dir(self, sx, sy):
            return g.ledge_dir(sx, sy)
        def walkable(self, sx, sy):
            return g.walkable(sx, sy)
    p2 = tv.bfs(G2(), cur, lambda t, gg=goal: t == gg)
    print(f"BFS {cur} -> {goal}: with-edge-law={'len ' + str(len(p1)) if p1 else 'NO ROUTE'} | "
          f"edge-law-off={'len ' + str(len(p2)) if p2 else 'NO ROUTE'}", flush=True)
