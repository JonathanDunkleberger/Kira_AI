"""recon_findmart.py — robustly identify the Cerulean Poké Mart by INTERIOR LAYOUT (clerk object at
(2,3), per the verified Viridian/Pewter signature), not flaky menu actuation. Enters each candidate
Cerulean building, dumps interior map id + all object events, flags the clerk-at-(2,3) building."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import Campaign, resolve_state                       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OB, SZ = 0x02036E38, 0x24
OFF_ACTIVE, OFF_GFX, OFF_X, OFF_Y, OFF_FACING = 0x00, 0x05, 0x10, 0x12, 0x18
# candidate building doors on Cerulean overworld (3,3); skip (30,11)=policeman robbed-house, (22,19)=Center
DOORS = [(10, 11), (15, 17), (31, 21), (13, 28), (29, 28), (17, 11)]
CLERK_TILE = (2, 3)


def objs(b):
    out = []
    for i in range(1, 16):
        o = OB + i * SZ
        if not (b.rd8(o + OFF_ACTIVE) & 1):
            continue
        out.append((b.rd8(o + OFF_GFX), b.rds16(o + OFF_X) - 7, b.rds16(o + OFF_Y) - 7,
                    b.rd8(o + OFF_FACING) & 0x0F))
    return out


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._exit_to_overworld()
    for _ in range(30):
        b.run_frame()
    print(f"overworld map={tv.map_id(b)} player={tv.coords(b)}", flush=True)
    snap = b.save_state()
    for door in DOORS:
        b.load_state(snap)
        for _ in range(10):
            b.run_frame()
        try:
            r = camp.enter_warp(pick=door)
        except Exception as e:
            print(f"door {door}: enter_warp ERR {e}"); continue
        inside = tv.map_id(b)
        if r != "warped" or inside[0] == 3:
            print(f"door {door}: did not enter ({r}, map={inside})"); continue
        ob = objs(b)
        clerk = [o for o in ob if (o[1], o[2]) == CLERK_TILE]
        tag = " *** CLERK AT (2,3) -> MART ***" if clerk else ""
        print(f"door {door} -> interior {inside}: objs={ob}{tag}", flush=True)


if __name__ == "__main__":
    main()
