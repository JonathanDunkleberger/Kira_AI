"""recon_fix_e4_moves.py — TACTICAL E4 PREP: restore the level-up moves the auto-learn heuristic
wrongly DECLINED on the two E4 specialists, so Kira fields real type answers (not a Venusaur tank).

ROOT CAUSE (disasm + pokemondb Gen-3 learnsets, verified 2026-07-10):
  • Lapras learns ICE BEAM by level-up at L31 (ours is L39 → it PASSED the learn point without it;
    the "never overwrite a good move" auto-learn declined Ice Beam for Confuse Ray/Perish Song).
    Ice Beam = the Lance-dragons (4x on Dragonite) + Gary-Charizard (2x) hard counter.
  • Kadabra learns PSYBEAM@L21, RECOVER@L25, PSYCHIC@L36 — ours is L40 with only 50-pw Confusion
    (Teleport/Flash/Disable filling the other slots). All three STAB/utility moves were declined.
Both mons already PASSED these learn levels, so grinding cannot retrieve them — the in-place fix is
to write the moves they legitimately earned. Mode-side, STAGING ONLY (canonical never touched).

IN:  states/workshop/e4_base.state (badge 8, at Indigo Plateau, Venusaur L71 / Lapras L39 / Kadabra L40)
OUT: states/workshop/e4_tactical.state (+ continuity sidecars copied from e4_base)
RUN: .venv/Scripts/python.exe -u pokemon_agent/recon_fix_e4_moves.py
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
SRC = os.path.join(WS, "e4_base.state")
OUT = os.path.join(WS, "e4_tactical.state")

# move id : (name, max PP)   — Gen-3 ids
SURF, ICE_BEAM, BODY_SLAM, CONFUSE_RAY = 57, 58, 34, 109
PSYCHIC, PSYBEAM, RECOVER, CONFUSION = 94, 60, 105, 93

LAPRAS_SET = ([SURF, ICE_BEAM, BODY_SLAM, CONFUSE_RAY], [15, 10, 15, 10])
KADABRA_SET = ([PSYCHIC, PSYBEAM, RECOVER, CONFUSION], [10, 20, 20, 25])

LAPRAS_SPECIES = 131
KADABRA_SPECIES = 64


def _find_slot(camp, species):
    n = camp.b.rd8(ram.GPLAYER_PARTY_CNT)
    for s in range(min(n, 6)):
        if st.read_party_species(camp.b, s) == species:
            return s
    return -1


def _dump(camp, tag):
    n = camp.b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"--- {tag} ---")
    for s in range(min(n, 6)):
        sp = st.read_party_species(camp.b, s)
        lvl = camp.b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
        moves = st.read_party_moves(camp.b, s)
        names = [st.MOVE_NAMES.get(m, f"#{m}") for m in moves if m]
        print(f"  slot{s}: species={sp} L{lvl}  {names}")


def main():
    b = Bridge(ROM)
    b.load_state(open(SRC, "rb").read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)

    _dump(camp, "BEFORE")

    for species, (moves, pps), label in (
        (LAPRAS_SPECIES, LAPRAS_SET, "Lapras"),
        (KADABRA_SPECIES, KADABRA_SET, "Kadabra"),
    ):
        slot = _find_slot(camp, species)
        if slot < 0:
            print(f"!! {label} (species {species}) NOT in party — abort")
            return 1
        if slot != 0:
            camp._swap_party_slots(0, slot)
            for _ in range(4):
                b.run_frame()
        camp._set_lead_moves(moves, pps)
        for _ in range(4):
            b.run_frame()
        got = camp._lead_moves()
        print(f"  set {label}: {[st.MOVE_NAMES.get(m, m) for m in got]}")

    camp._restore_ace()          # Venusaur back to slot 0
    for _ in range(6):
        b.run_frame()

    _dump(camp, "AFTER")

    with open(OUT, "wb") as f:
        f.write(b.save_state())
    # ride the continuity sidecars forward
    for ext in ("journey_core.json", "soul.json", "strat_memory.json", "world_model.json"):
        s = os.path.join(WS, f"e4_base.{ext}")
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(WS, f"e4_tactical.{ext}"))
    print(f"WROTE {OUT} (+ sidecars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
