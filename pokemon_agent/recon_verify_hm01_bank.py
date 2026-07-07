"""recon_verify_hm01_bank.py — the on-disk-now standard for the captain milestone.

Loads G:/temp/longrun/banked_GOAL/kira_campaign.state in a FRESH core and reads back:
FLAG_GOT_HM01 (0x237), HM01 in the bag/TM case, party HP (never promote hurt), map/coords.
READ-ONLY. RUN: python pokemon_agent/recon_verify_hm01_bank.py [bank_dir]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import field_moves as fm               # noqa: E402
import pokemon_state as st             # noqa: E402
import firered_ram as ram              # noqa: E402

PMON = st.PARTY_MON_SIZE

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = sys.argv[1] if len(sys.argv) > 1 else r"G:\temp\longrun\banked_GOAL"


def main():
    st = os.path.join(BANK, "kira_campaign.state")
    print(f"bank: {st}")
    b = Bridge(ROM)
    with open(st, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    print(f"map={tv.map_id(b)} coords={tv.coords(b)}")
    print(f"FLAG_GOT_HM01 (0x237) = {bool(fm.read_flag(b, 0x237))}")
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    hurt = False
    for s in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + s * PMON
        name = st.SPECIES_NAME.get(st.read_party_species(b, s), f"#{s}").title()
        lvl, hp, mx = b.rd8(base + 0x54), b.rd16(base + 0x56), b.rd16(base + 0x58)
        flag = "  <-- HURT" if hp < mx else ""
        hurt = hurt or hp < mx
        print(f"  {name} L{lvl} HP {hp}/{mx}{flag}")
    print(f"party verdict: {'HURT -- heal before promote' if hurt else 'FULL -- promotable'}")


if __name__ == "__main__":
    main()
