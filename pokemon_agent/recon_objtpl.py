"""recon_objtpl.py — verify read_object_templates against disasm ground truth (B1F: 5 grunts
at (4,9)/(24,12)/(6,32)/(10,22)/(21,27), item balls Escape Rope (5,16) + Hyper Potion (1,22)),
and fingerprint OBJ_EVENT_GFX_ITEM_BALL's numeric value.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_objtpl.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge   # noqa: E402
import travel as tv         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_hideout")

b = Bridge(ROM)
with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(40):
    b.run_frame()
print(f"map={tv.map_id(b)} coords={tv.coords(b)}")
for (x, y), gfx, present in tv.read_object_templates(b):
    print(f"  obj ({x:3d},{y:3d}) gfx=0x{gfx:02x} present={present}")
