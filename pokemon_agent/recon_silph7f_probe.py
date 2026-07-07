"""recon_silph7f_probe.py — strike15 wedge forensics (7F pocket, (0,8) sealed).

Loads the stage save banked at the wedge, dumps LIVE gObjectEvents (not templates —
the walk_path_to mask uses template coords; a body that MOVED is invisible to it),
the pocket grid walkability, warps, and a frame PNG to LOOK at.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_silph",
                     "kira_campaign.state")
OB, SZ, N = 0x02036E38, 0x24, 16

b = Bridge(ROM)
with open(STAGE, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()

print(f"map={tv.map_id(b)} coords={tv.coords(b)} in_battle={st.in_battle(b)} "
      f"box_open={dd_box(b)}")

print("\nLIVE gObjectEvents (idx, active, live-coord, template-coord?, gfx):")
for i in range(N):
    o = OB + i * SZ
    flags = b.rd8(o)
    if not (flags & 1):
        continue
    lx, ly = b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7
    px, py = b.rds16(o + 0x0C) - 7, b.rds16(o + 0x0E) - 7   # previous/initial coords
    gfx = b.rd8(o + 0x05)
    lid = b.rd8(o + 0x08)
    print(f"  [{i:2d}] flags=0x{flags:02x} localId={lid} gfx={gfx} "
          f"live=({lx},{ly}) prev=({px},{py})")

print("\ntemplates (travel.read_object_templates):")
for o in tv.read_object_templates(b):
    print(f"  {o}")

print("\nwarps:")
for w in tv.read_warps(b):
    print(f"  {w}")

g = tv.Grid(b)
print("\npocket grid walkability (x 0..8, y 2..12; #=blocked, .=walk, P=player):")
cur = tuple(tv.coords(b) or (0, 0))
for y in range(2, 13):
    row = ""
    for x in range(0, 9):
        if (x, y) == cur:
            row += "P"
        else:
            row += "." if g.walkable(x, y) else "#"
    print(f"  y={y:2d}  {row}")

try:
    b.frame_rgb().resize((480, 320)).save(os.path.join(_HERE, "..", "logs", "longrun",
                                                       "wedge7f_frame.png"))
    print("\nframe saved -> logs/longrun/wedge7f_frame.png")
except Exception as e:
    print(f"frame save failed: {e}")
