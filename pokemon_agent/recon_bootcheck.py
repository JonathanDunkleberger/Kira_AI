"""recon_bootcheck.py — minimal boot sanity: load canonical save, print party+coords. No mutation."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = os.path.join(_HERE, "states", "campaign", "kira_campaign.state")

P_HP, P_MAXHP = 0x56, 0x58
b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.press("B", 2, 2, None, owner="agent")
cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
print(f"party_cnt={cnt}")
for s in range(min(cnt, 6)):
    base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
    lvl = b.rd8(base + 0x54)
    hp = b.rd16(base + P_HP)
    mx = b.rd16(base + P_MAXHP)
    print(f"  slot{s} L{lvl} hp={hp}/{mx}")
print(f"map_id={tv.map_id(b)} coords={tv.coords(b)}")
print("BOOTCHECK_OK")
