"""recon_partydump.py — quick party inspector for a savestate (level + decrypted moves).
Used night shift 15 to pick a pre-L20-ivysaur verification state for the move-learn fix.
RUN: python pokemon_agent/recon_partydump.py <state_path_or_name>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import firered_ram as ram
import pokemon_state as st

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def resolve(name):
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            return cand
    return name


def main():
    path = resolve(sys.argv[1])
    b = Bridge(ROM)
    with open(path, "rb") as f:
        b.load_state(f.read())
    for _ in range(4):
        b.run_frame()
    n = ram.read_party_count(b)
    print(f"STATE {path} — party count {n}")
    for s in range(n):
        sp = st.read_party_species(b, s)
        lvl = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
        moves = st.read_party_moves(b, s)
        pps = st.read_party_pp(b, s)
        mv = []
        for mid, pp in zip(moves, pps):
            if mid:
                tname, power = st.move_info(b, mid)
                mv.append(f"{st.MOVE_NAMES.get(mid, f'#{mid}')}({tname},{power}pw,{pp}pp)")
        print(f"  slot{s}: species={sp} L{lvl}  moves=[{', '.join(mv)}]")


if __name__ == "__main__":
    main()
