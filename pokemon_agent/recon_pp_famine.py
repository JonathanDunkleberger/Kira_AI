"""recon_pp_famine.py - READ-ONLY: verify read_party_pp / slot_has_damaging_pp against the
canonical campaign state (erika_run2 postmortem, 2026-07-07). Prints per-slot species/level/HP,
each move with its PP and ROM power, and the damaging-PP verdict the famine switch will act on.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge        # noqa: E402
import pokemon_state as st       # noqa: E402
import firered_ram as ram        # noqa: E402


def main():
    b = Bridge(os.path.join(os.path.dirname(_HERE), "roms", "firered.gba"))
    path = os.path.join(_HERE, "states", sys.argv[1] if len(sys.argv) > 1 else "kira_campaign.state")
    with open(path, "rb") as f:
        b.load_state(f.read())
    for _ in range(6):
        b.run_frame()
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"   [ppf] state={os.path.basename(path)} party_count={cnt}", flush=True)
    for s in range(min(cnt, 6)):
        sp = st.read_party_species(b, s)
        lvl = b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
        hp = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
        mx = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
        moves = st.read_party_moves(b, s)
        pps = st.read_party_pp(b, s)
        det = []
        for mid, pp in zip(moves, pps):
            if not mid:
                continue
            t, pw = st.move_info(b, mid)
            det.append(f"{st.MOVE_NAMES.get(mid, '#' + str(mid))}(pp{pp}/pow{pw})")
        ok = st.slot_has_damaging_pp(b, s)
        print(f"   [ppf] slot{s} {st.SPECIES_NAME.get(sp, sp)} L{lvl} {hp}/{mx} "
              f"damaging_pp={'YES' if ok else 'NO'} :: {' '.join(det)}", flush=True)


if __name__ == "__main__":
    main()
