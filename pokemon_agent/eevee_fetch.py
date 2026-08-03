"""eevee_fetch.py — the Celadon Condominiums gift Eevee, in-loop (2026-08-03 OP-team pass).

Jonny's order: "I want her seeking out and developing an OP as fuck team with cute Pokémon like
Eevee and rare or legendary ones." Eevee is the single cheapest OP acquisition in FireRed: a
GUARANTEED L25 gift Poké Ball sitting in the Condominiums roof room in a city she has already
cleared (badge 4 = Erika), no RNG, no battle — and a Thunder Stone (Dept. 4F, same city) turns it
into Jolteon, the electric slot the tidal team plan wants for Lorelei/Lance. This strike is the
executor for the 'eevee' questline capability (frlg_gates.json), dispatched from the campaign's
strike registry exactly like the Mansion/Silph/Safari strikes.

GROUND TRUTH (pret/pokefirered map JSONs, fetched 2026-08-03 — layout bins are the collision
truth at runtime via the live BFS):
  maps      : Celadon (3,6) | Condos 1F (10,7) 2F (10,8) 3F (10,9) Roof (10,10) RoofRoom (10,11)
  back door : Celadon overworld (30,4) [side tiles (29,5)/(31,5)] -> 1F lands (2,1)
  stairwells: 1F (12,2)/(4,2) -> 2F; 2F (11,2)/(2,2) -> 3F; 3F (12,2)/(4,2) -> Roof
              (two independent stair columns — the climb tries the right/back column first and
              falls back to the left/front one; the live BFS decides which is actually reachable)
  roof room : Roof (2,12) -> RoofRoom (lands (4,7)); exits (3,8)/(5,8) back to the Roof
  the ball  : RoofRoom OBJECT at (7,3) — script givemon SPECIES_EEVEE, 25
  done flag : FLAG_GOT_EEVEE 0x263 (set by the script in EVERY branch)

PARTY-FULL IS FINE: FRLG's script auto-transfers the gift to the PC when the party is full
(VAR_RESULT==1 -> GetEeveePC) — no room-making prerequisite. The PCBOX swap_keeper machinery
fields a boxed Eevee at the next Center visit (it's on the team plan's electric line).

run_strike returns 'got_eevee' | 'in_condo' (flag set but the walk-out didn't finish) |
'not_here' | 'failed'.
"""
import time

import field_moves as fm
import firered_ram as ram
import travel as tv
from mansion_strike import MansionStrike

# ── FireRed Condominiums fact table (game-knowledge layer; rule 14 portability debt) ──────────
CELADON = (3, 6)
C1F, C2F, C3F, ROOF, ROOFROOM = (10, 7), (10, 8), (10, 9), (10, 10), (10, 11)
CONDO_MAPS = {C1F, C2F, C3F, ROOF, ROOFROOM}
EEVEE_ANCHORS = {CELADON} | CONDO_MAPS
FLAG_GOT_EEVEE = 0x263                    # set by the roof-room script in every givemon branch
BALL = (7, 3)                             # the Eevee Poké Ball object (RoofRoom)

# (current_map, [candidate warp tiles in preference order], dest_map). The climb/descend loops
# look up the leg for WHEREVER she stands, so a mid-tour re-tick resumes from any floor.
CLIMB = [
    (CELADON, [(30, 4)], C1F),            # the back door (lands right by the stair columns)
    (C1F, [(12, 2), (4, 2)], C2F),
    (C2F, [(11, 2), (2, 2)], C3F),
    (C3F, [(12, 2), (4, 2)], ROOF),
    (ROOF, [(2, 12)], ROOFROOM),
]
DESCEND = [
    (ROOFROOM, [(3, 8), (5, 8)], ROOF),
    (ROOF, [(4, 2), (10, 2)], C3F),
    (C3F, [(2, 2), (11, 2)], C2F),
    (C2F, [(4, 2), (12, 2)], C1F),
    (C1F, [(2, 1), (11, 19), (13, 19)], CELADON),   # back door first; front door as fallback
]


class EeveeFetch(MansionStrike):
    """Reuses the Mansion strike's PROVEN interior primitives (live-BFS walk, go_warp, interact,
    battle/dialogue interrupt handling) — only the mission differs. No statues, no puzzles: this
    is the gentlest interior in the game, which is why it's the first strike built from ground
    truth alone instead of a recon champion run. Every leg is verified by map-id flips and the
    pickup by FLAG_GOT_EEVEE — nothing here trusts a menu byte."""

    def __init__(self, camp, log, dbg_dir=None):
        super().__init__(camp, log, dbg_dir)
        self.deadline = time.time() + 900        # a 5-floor walk; 15 min is generous

    def _leg(self, table):
        here = tuple(tv.map_id(self.b))
        return next((l for l in table if l[0] == here), None)

    def _ride(self, table, goal_map, label):
        """Advance leg-by-leg until `goal_map` (or a dead end / the deadline). Candidate warp
        tiles are tried in order — the live BFS decides which stair column is actually walkable."""
        for _hop in range(len(table) + 2):
            while self.handle_interrupts():
                pass
            if tuple(tv.map_id(self.b)) == goal_map:
                return True
            if time.time() > self.deadline:
                self.log(f"!! [{label}] deadline mid-ride")
                return False
            leg = self._leg(table)
            if leg is None:
                self.log(f"!! [{label}] no leg from {tv.map_id(self.b)} — off the mission rails")
                return False
            _, cands, dest = leg
            fired = False
            for t in cands:
                if self.go_warp(t, dest, f"{label}{t}"):
                    fired = True
                    break
            if not fired:
                self.log(f"!! [{label}] every candidate warp failed on {tv.map_id(self.b)}")
                return False
        return tuple(tv.map_id(self.b)) == goal_map

    def run(self):
        b, camp = self.b, self.camp
        got = fm.read_flag(b, FLAG_GOT_EEVEE)
        try:
            _pc = b.rd8(ram.GPLAYER_PARTY_CNT)
        except Exception:
            _pc = -1
        self.log(f"   eevee fetch: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"got_eevee={int(got)} party={_pc} "
                 f"({'gift auto-PCs' if _pc >= 6 else 'gift joins the party'})")
        if got and tuple(tv.map_id(b)) == CELADON:
            self.log("   Eevee already claimed and back on Celadon — nothing to strike")
            return "got_eevee"
        if not got:
            if not self._ride(CLIMB, ROOFROOM, "climb"):
                return self._bail()
            # THE BALL: A opens givemon -> fanfare -> 'Obtained an Eevee!' -> nickname YES/NO.
            # Drain with B (B advances text AND answers NO to the nickname box); party-full runs
            # the auto-PC branch by itself. Verify by the flag — set in every script branch.
            if not self.interact(BALL, "eevee-ball", key="B",
                                 verify=lambda: fm.read_flag(b, FLAG_GOT_EEVEE)):
                return self._bail()
            self.log("   [pickup] EEVEE obtained (flag 0x263 set)")
            try:
                camp.on_event("an EEVEE! look at it. LOOK at it. okay — this little thing is "
                              "getting a Thunder Stone the second we pass the Dept. Store.",
                              kind="catch", tier=3)
            except Exception:
                pass
        else:
            self.log("   RESUME: Eevee already claimed mid-tour — running the walk-out only")
        if not self._ride(DESCEND, CELADON, "descend"):
            return self._bail()
        self.snap("90_eevee_out")
        self.log(f"   EEVEE FETCH DONE: back on Celadon @ {tv.coords(b)} | battles {self.n_battles}")
        return "got_eevee"

    def _bail(self):
        """A leg failed. Flag already set = the objective is DONE — report in_condo so the caller
        keeps the errand alive and the walk-out retries next tick; else it's a real failure."""
        if fm.read_flag(self.b, FLAG_GOT_EEVEE):
            self.log("   fetch wedged AFTER the ball (0x263 set) — the exit owns the rest")
            return "in_condo"
        return "failed"


def run_strike(camp, log, dbg_dir=None):
    """Fetch the Celadon Condominiums gift Eevee from wherever she stands, in ONE call.
    Idempotent by state: flag-set + on Celadon short-circuits; flag-set inside runs only the
    walk-out; any interior boot resumes the climb from its own floor."""
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in EEVEE_ANCHORS:
        return "not_here"
    return EeveeFetch(camp, log, dbg_dir).run()
