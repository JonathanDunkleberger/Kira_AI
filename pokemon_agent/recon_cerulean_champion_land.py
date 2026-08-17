"""Land a WALKABLE Cerulean Champion overworld from the THE END grenade.

The Pallet coord+CB2_Overworld poke (11:42) left Hall of Fame VRAM/scripts
loaded. Resume then "walked" Pallet/Route 1 on a frozen HoF screen.

This does a real WarpIntoMap: sWarpDestination + CB2_LoadMap. Proof bar:
Cerulean header (cave mouth warp), one tile actually moves, no Oak HoF text.
"""
import os
import shutil
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bridge import Bridge
import travel as tv
import firered_ram as ram
import field_moves as fm
from e4_strike import (
    _land_overworld_from_credits, _pallet_grenade_path, _cerulean_header_ok,
    _hof_text_live, _CERULEAN,
)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CAMP = os.path.join(_HERE, "states", "campaign")
OUT = os.path.join(CAMP, "kira_campaign.CERULEAN_MEWTWO_LAND.state")
CANON = os.path.join(CAMP, "kira_campaign.state")


class _Camp:
    def __init__(self, b):
        self.b = b
        self.render = None

    def has_badge(self, flag):
        return fm.read_flag(self.b, flag)


def _step_moved(b, c0):
    b.set_input_owner("agent")
    for d in ("DOWN", "LEFT", "RIGHT", "UP"):
        b.press(d, 8, 8, owner="agent")
        for _ in range(24):
            b.run_frame()
        c1 = tv.coords(b)
        print("step", d, c0, "->", c1)
        if c1 and c0 and c1 != c0:
            return True
    return False


def main():
    src = _pallet_grenade_path()
    if not src:
        print("NO GRENADE")
        return 2
    print("src", src)
    b = Bridge(ROM)
    with open(src, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    print("loaded", tv.map_id(b), tv.coords(b),
          "cb2", hex(b.rd32(ram.GMAIN_CB2)),
          "party", b.rd8(ram.GPLAYER_PARTY_CNT),
          "ow", ram.battle_cb2_dead(b),
          "hof_text", _hof_text_live(b))
    camp = _Camp(b)
    r = _land_overworld_from_credits(camp, print)
    mp = tuple(tv.map_id(b) or ())
    c0 = tv.coords(b)
    hdr = _cerulean_header_ok(b)
    hof = _hof_text_live(b)
    badges = sum(1 for i in range(8) if fm.read_flag(b, 0x820 + i))
    party = b.rd8(ram.GPLAYER_PARTY_CNT)
    print("land ->", r, mp, c0,
          "cb2", hex(b.rd32(ram.GMAIN_CB2)),
          "ow", ram.battle_cb2_dead(b),
          "header_ok", hdr, "hof_text", hof,
          "badges", badges, "party", party,
          "clear", fm.read_flag(b, 0x82C))
    try:
        g = tv.Grid(b)
        print("grid", g.w, g.h, "walkable_here",
              g.walkable(*c0) if c0 else None,
              "warps", tv.read_warps(b)[:8])
    except Exception as e:
        print("grid FAIL", e)
    spawn_bytes = b.save_state()
    moved = _step_moved(b, c0) if c0 else False
    print("after_step", tv.map_id(b), tv.coords(b), "MOVED" if moved else "NO_MOVE")
    if party < 4 or badges < 8:
        print("REFUSING BANK — not Champion (party/badges)")
        return 3
    if r != "ok" or mp != _CERULEAN or not hdr or not moved:
        print("LAND FAILED — not replacing canonical")
        return 1
    os.makedirs(CAMP, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(spawn_bytes)
    print("wrote", OUT, "bytes", len(spawn_bytes), "stand", c0, "MOVED-PROOF")
    shutil.copy2(OUT, CANON)
    print("copied ->", CANON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
