"""recon_bagdump.py - READ-ONLY probe: dump the full bag + party from a COPY of a banked state.

Answers the shift-18 PP-famine question offline while run22 laps: does she already HOLD
Ethers/Elixirs (the ether-instinct would refill Razor Leaf mid-gauntlet) or TM19 Giga Drain?
Boots its own emulator on a copied .state - never touches the live run or the bank.

Run:  .venv\\Scripts\\python.exe pokemon_agent\\recon_bagdump.py [path-to-state]
"""
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge          # noqa: E402
from recon_fingerprint import bag_items, money, party  # noqa: E402

# FRLG item names we care about for the PP/heal economy (ids per pokefirered)
NAMES = {13: "Potion", 19: "Full Restore", 20: "Max Potion", 21: "Hyper Potion",
         22: "Super Potion", 23: "Full Heal", 24: "Revive", 25: "Max Revive",
         34: "Ether", 35: "Max Ether", 36: "Elixir", 37: "Max Elixir",
         38: "Lava Cookie", 44: "Rare Candy", 63: "PP Up", 107: "PP Max",
         289: "TM01", 307: "TM19 Giga Drain", 310: "TM22 SolarBeam",
         314: "TM26 Earthquake", 359: "Poke Flute"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else r"G:/temp/longrun/banked_E4/kira_campaign.state"
    tmp = os.path.join(tempfile.gettempdir(), "bagdump.state")
    shutil.copy2(src, tmp)                     # copy: the live run rewrites the bank every ~30s
    rom = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
    b = Bridge(rom)
    with open(tmp, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    print(f"money ${money(b)}")
    cnt, mons = party(b)
    print(f"party ({cnt}): {mons}")
    for pocket, iid, qty in (bag_items(b) or []):
        tag = NAMES.get(iid, "")
        print(f"  {pocket:9s} item {iid:3d} x{qty:<3d} {tag}")


if __name__ == "__main__":
    main()
