"""recon_ship_2f.py — ground-truth dump of the S.S. Anne 2F corridor (1,10) from the run-16 stall bank.

Loads banked_STALL (she stands at (1,10)@(14,10)), prints: full warp table, NPC objects, the tile
behaviors along the corridor rows, and saves a frame. READ-ONLY. RUN: python pokemon_agent/recon_ship_2f.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = r"G:\temp\longrun\banked_STALL\kira_campaign.state"
OUT = r"G:\temp\claude\C--WINDOWS-system32\9717c34e-ef1a-4fc3-88e5-c78650ebc7f0\scratchpad"


def objects(b):
    OB, SZ = 0x02036E38, 0x24
    out = []
    for i in range(16):
        o = OB + i * SZ
        try:
            if not (b.rd8(o) & 1):
                continue
            out.append({"idx": i, "gfx": b.rd8(o + 0x05),
                        "coord": (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7)})
        except Exception:
            continue
    return out


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    mp, co = tv.map_id(b), tv.coords(b)
    print(f"map={mp} coords={co}")
    print("WARPS:")
    for (xy, dest, wid) in tv.read_warps(b):
        print(f"  {tuple(xy)} -> {tuple(dest)} (wid={wid})")
    print("NPC OBJECTS:")
    for o in objects(b):
        print(f"  {o}")
    g = tv.Grid(b)
    print("TILE BEHAVIORS (corridor sweep x=0..17, y=0..12; '.'=walkable-plain):")
    for y in range(0, 13):
        row = []
        for x in range(0, 18):
            try:
                beh = tv.tile_behavior(b, x, y) if hasattr(tv, "tile_behavior") else None
            except Exception:
                beh = None
            if beh is None:
                try:
                    walk = g.walkable(x + tv.MAP_OFFSET, y + tv.MAP_OFFSET)
                except Exception:
                    walk = False
                row.append("." if walk else "#")
            else:
                row.append(f"{beh:02x}" if beh else ".")
        print(f"  y={y:2d} " + " ".join(f"{c:>2}" for c in row))
    try:
        png = os.path.join(OUT, "ship2f_stall.png")
        b.screenshot(png)
        print(f"frame -> {png}")
    except Exception as e:
        print(f"screenshot failed: {e}")


if __name__ == "__main__":
    main()
