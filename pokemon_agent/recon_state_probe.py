"""recon_state_probe.py — throwaway: where is each candidate savestate, and what's in the party/bag?

Used to pick a staging state for the Revive-cursor instrument. Prints map/coords, the 6 party
slots (species/level/hp), and the Items pocket.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_state_probe.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge              # noqa: E402
import firered_ram as ram              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
TMP = os.environ.get("TEMP", _HERE)
CANDS = [
    os.path.join(TMP, "longrun", "sell_check_stage", "kira_campaign.state"),
    os.path.join(TMP, "longrun", "banked_LIVE", "kira_campaign.state"),
    os.path.join(TMP, "longrun", "banked_GOAL", "kira_campaign.state"),
    os.path.join(TMP, "longrun", "banked_TIMEOUT", "kira_campaign.state"),
    os.path.join(TMP, "longrun", "stage", "kira_campaign.state"),
]

_ITEMS_POCKET_OFF = 0x310


def pocket(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    out = []
    for s in range(42):
        slot = sb1 + _ITEMS_POCKET_OFF + s * 4
        iid = b.rd16(slot)
        if not iid:
            continue
        qty = b.rd16(slot + 2) ^ key
        if qty > 0:
            out.append((iid, qty))
    return out


def main():
    for path in CANDS:
        if not os.path.exists(path):
            print(f"--- {path}: MISSING")
            continue
        b = Bridge(ROM)
        with open(path, "rb") as f:
            b.load_state(f.read())
        for _ in range(180):
            b.run_frame()
        print(f"=== {path}")
        try:
            print(f"    map={tv.map_id(b)} coords={tv.coords(b)} in_battle={st.in_battle(b)}")
        except Exception as e:
            print(f"    map read failed: {e}")
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        print(f"    party_count={cnt}")
        for i in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + i * 100
            sp = st.read_party_species(b, i)
            print(f"      slot{i}: sp={sp} ({st.SPECIES_NAME.get(sp, '?')}) "
                  f"lvl={b.rd8(base + 0x54)} hp={b.rd16(base + 0x56)}/{b.rd16(base + 0x58)}")
        print(f"    items={pocket(b)}")
        del b


if __name__ == "__main__":
    sys.exit(main() or 0)
