"""recon_evolve_disc.py — verify the swap-vs-evolve PID discriminator (2026-07-06 ride-along 0a).

The false "[evolve] ivysaur evolved into spearow" beat fired because _soul_after_objective watched
species-at-slot-0 blind: the strategic grind's _swap_party_slots (a legitimate party reorder) changes
slot 0's OCCUPANT, which read as an "evolution". Discriminator: evolution keeps the mon's PID; a
reorder puts a different PID in slot 0.

Three cases, all in-memory on a loaded state (ZERO writes to any save file):
  C1 party reorder (swap 0<->1)      -> NO evolve beat (the old false-positive)
  C2 same-PID species change (sim)   -> evolve beat fires (real evolution still detected)
  C3 reorder back (_restore path)    -> NO evolve beat
RUN: python pokemon_agent/recon_evolve_disc.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                      # noqa: E402
import firered_ram as ram                      # noqa: E402
import pokemon_state as st                     # noqa: E402
from campaign import Campaign, resolve_state   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


class SoulRecorder:
    def __init__(self):
        self.evolves = []

    def note_evolve(self, before, after, who=None):
        self.evolves.append((before, after))

    def note_faint(self, who):
        pass

    def note_outcome(self, won, what=None):
        pass


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    rec = SoulRecorder()
    camp.soul = rec

    sp0 = st.read_party_species(b, 0)
    pid0 = b.rd32(ram.GPLAYER_PARTY + 0)
    camp._last_lead_species, camp._last_lead_pid = sp0, pid0
    print(f"lead: {st.SPECIES_NAME.get(sp0, sp0)} pid=0x{pid0:08x}")

    # C1 — a party reorder (exactly what the grind does) must NOT read as evolution
    camp._swap_party_slots(0, 1)
    camp._soul_after_objective("FREE_ROAM", "ok")
    c1 = len(rec.evolves) == 0
    print(f"C1 reorder 0<->1 -> evolve beats: {rec.evolves}  {'PASS' if c1 else 'FAIL'}")

    # C2 — a same-PID species change (simulated via the tracker) MUST fire the beat
    camp._last_lead_species = sp0 if camp._last_lead_species != sp0 else (sp0 + 1)
    # keep the tracked PID equal to the LIVE slot-0 PID -> "same mon, species changed" = evolution
    camp._last_lead_pid = b.rd32(ram.GPLAYER_PARTY + 0)
    camp._soul_after_objective("FREE_ROAM", "ok")
    c2 = len(rec.evolves) == 1
    print(f"C2 same-PID species change -> evolve beats: {rec.evolves}  {'PASS' if c2 else 'FAIL'}")

    # C3 — swap back (the _restore_ace direction) must NOT fire
    n = len(rec.evolves)
    camp._swap_party_slots(0, 1)
    camp._soul_after_objective("FREE_ROAM", "ok")
    c3 = len(rec.evolves) == n
    print(f"C3 reorder back -> evolve beats: {rec.evolves}  {'PASS' if c3 else 'FAIL'}")

    print("\nALL PASS" if (c1 and c2 and c3) else "\nFAILURES — do not ship")


if __name__ == "__main__":
    main()
