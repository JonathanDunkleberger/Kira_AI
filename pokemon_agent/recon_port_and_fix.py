"""recon_port_and_fix.py — port the GRINDED specialists (Lapras+Kadabra) from the Route-18 grind
result INTO the Indigo E4 base, then restore their Ice Beam / Psychic movesets.

WHY: the E4 base (e4_base, at Indigo Plateau, past Victory Road) is the only geographically-correct
launch point for recon_e4, but its Lapras/Kadabra are too underlevelled (L39/L40) to survive the
L53-62 gauntlet. Victory Road (the adjacent strong-wild spot) is a CAVE — the grass-only grind can't
train there, and the Indigo->VR nav has no cached route. So we grind the same-lineage specialists on
Route-18 grass (harness-proven) and PORT the leveled mons back to the Indigo save. A raw 100-byte
party-struct copy carries the whole mon (PID/OTID + encrypted substructs + plaintext stats) intact.
Mode-side, STAGING ONLY (canonical untouched).

IN:  SRC (grind result, default G:/temp/longrun/banked_GRIND/kira_campaign.state)
     DEST = states/workshop/e4_base.state
OUT: states/workshop/e4_tactical_v2.state (+ e4_base sidecars)
RUN: SRC=<path> .venv/Scripts/python.exe -u pokemon_agent/recon_port_and_fix.py
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
from campaign import Campaign

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
SRC = os.environ.get("SRC", os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_GRIND", "kira_campaign.state"))
DEST = os.path.join(WS, "e4_base.state")
OUT = os.path.join(WS, "e4_tactical_v2.state")

PORT_SPECIES = (131, 64)   # Lapras, Kadabra
WORDS = st.PARTY_MON_SIZE // 4   # 25 u32 words

SURF, ICE_BEAM, BODY_SLAM, CONFUSE_RAY = 57, 58, 34, 109
PSYCHIC, PSYBEAM, RECOVER, CONFUSION = 94, 60, 105, 93
LAPRAS_SET = ([SURF, ICE_BEAM, BODY_SLAM, CONFUSE_RAY], [15, 10, 15, 10])
KADABRA_SET = ([PSYCHIC, PSYBEAM, RECOVER, CONFUSION], [10, 20, 20, 25])


def _find_slot(b, species):
    n = b.rd8(ram.GPLAYER_PARTY_CNT)
    for s in range(min(n, 6)):
        if st.read_party_species(b, s) == species:
            return s
    return -1


def _read_struct(b, slot):
    base = ram.GPLAYER_PARTY + slot * st.PARTY_MON_SIZE
    return [b.rd32(base + i * 4) for i in range(WORDS)]


def _write_struct(b, slot, words):
    base = ram.GPLAYER_PARTY + slot * st.PARTY_MON_SIZE
    for i, w in enumerate(words):
        b.core.memory.u32.raw_write(base + i * 4, w & 0xFFFFFFFF)


def _dump(b, tag):
    n = b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"--- {tag} ---")
    for s in range(min(n, 6)):
        sp = st.read_party_species(b, s)
        lvl = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
        moves = [st.MOVE_NAMES.get(m, f"#{m}") for m in st.read_party_moves(b, s) if m]
        print(f"  slot{s}: sp={sp} L{lvl}  {moves}")


def main():
    if not os.path.exists(SRC):
        print(f"!! grind result not found: {SRC} — run the grind first")
        return 1
    b = Bridge(ROM)

    # 1) read leveled structs from the grind result
    b.load_state(open(SRC, "rb").read())
    for _ in range(10):
        b.run_frame()
    grinded = {}
    for sp in PORT_SPECIES:
        s = _find_slot(b, sp)
        if s < 0:
            print(f"!! species {sp} not in grind result — abort")
            return 1
        lvl = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
        grinded[sp] = _read_struct(b, s)
        print(f"  grind src: species {sp} = L{lvl} (slot {s})")

    # 2) load the Indigo base, overwrite the species-matched slots
    b.load_state(open(DEST, "rb").read())
    for _ in range(10):
        b.run_frame()
    _dump(b, "DEST BEFORE")
    for sp in PORT_SPECIES:
        d = _find_slot(b, sp)
        if d < 0:
            print(f"!! species {sp} not in dest — abort")
            return 1
        _write_struct(b, d, grinded[sp])
    for _ in range(6):
        b.run_frame()

    # 3) restore the tactical movesets on the ported mons
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    for sp, (moves, pps) in ((131, LAPRAS_SET), (64, KADABRA_SET)):
        slot = _find_slot(b, sp)
        if slot != 0:
            camp._swap_party_slots(0, slot)
            for _ in range(4):
                b.run_frame()
        camp._set_lead_moves(moves, pps)
        for _ in range(4):
            b.run_frame()
    camp._restore_ace()
    for _ in range(6):
        b.run_frame()

    _dump(b, "DEST AFTER (ported + moves)")
    with open(OUT, "wb") as f:
        f.write(b.save_state())
    for ext in ("journey_core.json", "soul.json", "strat_memory.json", "world_model.json"):
        s = os.path.join(WS, f"e4_base.{ext}")
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(WS, f"e4_tactical_v2.{ext}"))
    print(f"WROTE {OUT} (+ sidecars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
