"""Land a walkable Pallet Champion overworld from the THE END grenade.

Champion-room reload sends Oak north into HoF → credits → title (live 2026-08-17).
This writes a Pallet (walkable) savestate so resume can hunt Mewtwo.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bridge import Bridge
import travel as tv
import firered_ram as ram
import field_moves as fm
from e4_strike import _land_overworld_from_credits, _pallet_grenade_path

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.path.join(_HERE, "states", "campaign", "kira_campaign.PALLET_CHAMPION_LAND.state")


class _Camp:
    def __init__(self, b):
        self.b = b
        self.render = None

    def has_badge(self, flag):
        return fm.read_flag(self.b, flag)


def main():
    src = _pallet_grenade_path()
    if not src:
        print("NO GRENADE"); return 2
    print("src", src)
    b = Bridge(ROM)
    with open(src, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    print("loaded", tv.map_id(b), tv.coords(b),
          "cb2", hex(b.rd32(ram.GMAIN_CB2)),
          "party", b.rd8(ram.GPLAYER_PARTY_CNT),
          "ow", ram.battle_cb2_dead(b))
    camp = _Camp(b)
    r = _land_overworld_from_credits(camp, print)
    print("land ->", r, tv.map_id(b), tv.coords(b),
          "cb2", hex(b.rd32(ram.GMAIN_CB2)),
          "ow", ram.battle_cb2_dead(b),
          "clear", fm.read_flag(b, 0x82C))
    c0 = tv.coords(b)
    g = tv.Grid(b)
    print("walkable_here", c0, g.walkable(*c0) if c0 else None)
    b.set_input_owner("agent")
    for d in ("DOWN", "RIGHT", "LEFT", "UP"):
        b.press(d, 8, 8, owner="agent")
        for _ in range(24):
            b.run_frame()
        c1 = tv.coords(b)
        print("step", d, c0, "->", c1)
        if c1 and c0 and c1 != c0:
            break
    moved = tv.coords(b) != c0
    data = b.save_state()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(data)
    print("wrote", OUT, "bytes", len(data), "MOVED" if moved else "NO_MOVE")
    return 0 if r == "ok" and moved else 1


if __name__ == "__main__":
    raise SystemExit(main())
