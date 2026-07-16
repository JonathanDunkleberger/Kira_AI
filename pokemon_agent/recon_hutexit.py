"""recon_hutexit.py — probe the UGP hut (1,33) exit mats from the run-11 STALL bank.
She's parked ON mat (5,8) (warps say (5,8)->(3,25)) and nothing fires. Ground-truth the
behavior bytes + find the press that exits."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_STALL")

b = Bridge(ROM)
with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()
print(f"boot: map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)


def behav(sx, sy):
    ml = b.rd32(0x02036DFC)
    attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
    w = b.rd32(tv.BACKUP_LAYOUT)
    mp = b.rd32(tv.BACKUP_LAYOUT + 8)
    e = b.rd16(mp + ((sx + 7) + w * (sy + 7)) * 2)
    mid = e & 0x3FF
    base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
    return b.rd32(base + idx * 4) & 0xFF


for yy in range(6, 10):
    print(f"y={yy}: " + " ".join(f"({xx},{yy})={behav(xx, yy):02x}" for xx in range(3, 9)), flush=True)

b.set_input_owner("agent")
from dialogue_drive import box_open      # noqa: E402
print(f"box_open={box_open(b)}", flush=True)
for _ in range(8):
    b.press("B", 8, 12, lambda: None, owner="agent")
    for _ in range(20):
        b.run_frame()
print(f"after B-drain: box_open={box_open(b)} coords={tv.coords(b)}", flush=True)
for label, seq in (("stand (6,7) then DOWN onto the 0x65 mat", ["UP", "RIGHT", "DOWN", "DOWN"]),):
    m0 = tuple(tv.map_id(b))
    for k in seq:
        b.press(k, 8, 10, lambda: None, owner="agent")
        for _ in range(40):
            b.run_frame()
        if tuple(tv.map_id(b)) != m0:
            break
    print(f"{label}: map {m0} -> {tv.map_id(b)} coords={tv.coords(b)}", flush=True)
    if tuple(tv.map_id(b)) != m0:
        print("EXIT FIRED", flush=True)
        break
