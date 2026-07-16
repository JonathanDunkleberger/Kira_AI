"""recon_sabrina_probe.py — ground-truth dump of the Saffron Gym grid (collision + ELEVATION).

strike2 truth: steps into (29,18)/(29,21) fail though collision reads 0 — the Silph strike9
"elevation-sealed" class. The map-grid u16 is: bits 0-9 metatile, 10-11 collision, 12-15
ELEVATION; the game blocks steps between differing non-zero elevations. This probe prints the
gym as ASCII with elevation digits so the strike's pad router can honor the real edge rule.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_sabrina_probe.py [state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("TEMP", _HERE), "longrun", "stage_sabrina", "kira_campaign.state")

b = Bridge(ROM)
with open(STATE, "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

print(f"map={tv.map_id(b)} coords={tv.coords(b)}")
w = b.rd32(tv.BACKUP_LAYOUT)
h = b.rd32(tv.BACKUP_LAYOUT + 4)
mp = b.rd32(tv.BACKUP_LAYOUT + 8)
OFF = tv.MAP_OFFSET
warps = tv.read_warps(b)
wt = {tuple(w0[0]): i for i, w0 in enumerate(warps)}
objs = {tuple(o[0]): o[1] for o in tv.read_object_templates(b) if o[2]}
cur = tuple(tv.coords(b) or (0, 0))

print(f"layout {w}x{h} (playable {w - 2 * OFF}x{h - 2 * OFF})")
print("\nWARPS (id: tile -> dest, dest_warp):")
for i, (xy, d, wid) in enumerate(warps):
    land = tuple(warps[wid][0]) if tuple(d) == tuple(tv.map_id(b)) and wid < len(warps) else None
    print(f"  {i:2d}: {tuple(xy)} -> {tuple(d)} w{wid}" + (f"  LANDS {land}" if land else ""))
print("\nOBJECTS (present):", sorted(objs.items()))

print("\nGRID — per tile 'CE' C=collision(.=0) E=elevation hex; @=her, Wnn=warp id, O=obj:")
hdr = "     " + "".join(f"{sx % 10}" for sx in range(0, w - 2 * OFF))
print(hdr)
for sy in range(0, h - 2 * OFF):
    row = []
    for sx in range(0, w - 2 * OFF):
        e = b.rd16(mp + ((sy + OFF) * w + (sx + OFF)) * 2)
        col = (e & 0x0C00) >> 10
        elev = (e >> 12) & 0xF
        if (sx, sy) == cur:
            row.append("@")
        elif (sx, sy) in wt:
            row.append("P")
        elif (sx, sy) in objs:
            row.append("O")
        elif col:
            row.append("#")
        else:
            row.append(f"{elev:x}")
    print(f"{sy:3d}  " + "".join(row))
