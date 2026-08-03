"""expshare_fetch.py — the Route 15 aide's Exp. Share, in-loop (2026-08-03 OP-team pass).

Jonny's order: "she needs exp share with 50 species caught on route 15." Ground truth agrees:
Prof. Oak's aide on the Route 15 West Entrance gate's 2F hands over ITEM_EXP_SHARE (182) once
the dex shows >= 50 CAUGHT species — the single biggest bench-development item in the game
(the whole-team levelling the E4 demands, for free, forever). This strike is the executor for
the 'exp_share' questline capability (frlg_gates.json), dispatched from the campaign's strike
registry exactly like the Eevee fetch. The 50-caught prerequisite is enforced UPSTREAM
(campaign._expshare_gate only opens the errand at dex>=50; below that the strategist brief
runs the DEX PUSH doctrine instead) — the strike itself just claims and walks out.

GROUND TRUTH (pret/pokefirered, fetched 2026-08-03):
  maps    : Fuchsia (3,7) | Route 15 (3,33) | Gate 1F (24,0) | Gate 2F (24,1)
  gate in : Route 15 door tiles (9,11) [west/near] and (16,11) [east] -> 1F
  stairs  : 1F (9,10) -> 2F (lands (10,9)); back down the same tile
  the aide: 2F OBJECT at (5,5) — script checks GetPokedexCount >= 50, gives ITEM_EXP_SHARE
  done    : FLAG_GOT_EXP_SHARE_FROM_OAKS_AIDE 0x256 (set only on a successful hand-over)
  fuchsia : connects WEST of Route 15 on the open overworld (no forced gate passage) — the
            errand's travel leg delivers her to Route 15; the strike anchors there.

A-DRAIN IS CORRECT here (unlike the Eevee ball's B-drain): the aide opens with a YES/NO
"want me to check?" box and A confirms the default YES; every later box is plain text.
Party/bag full is a non-issue upstream of 42 item slots; the script's no-room branch simply
doesn't set the flag, so the verify stays truthful and the errand retries after bag space.

run_strike returns 'got_expshare' | 'in_gate' (flag set but the walk-out didn't finish) |
'not_here' | 'failed'.
"""
import time

import field_moves as fm
import travel as tv
from mansion_strike import MansionStrike

# ── FireRed Route-15 gate fact table (game-knowledge layer; rule 14 portability debt) ─────────
FUCHSIA = (3, 7)
ROUTE15 = (3, 33)
GATE1F, GATE2F = (24, 0), (24, 1)
EXPSHARE_ANCHORS = {ROUTE15, GATE1F, GATE2F}
FLAG_GOT_EXP_SHARE = 0x256                # FLAG_GOT_EXP_SHARE_FROM_OAKS_AIDE
AIDE = (5, 5)                             # the scientist on 2F (faces the binocular wall)

CLIMB = [
    (ROUTE15, [(9, 11), (16, 11)], GATE1F),   # either street door; west one is nearest Fuchsia
    (GATE1F, [(9, 10)], GATE2F),
]
DESCEND = [
    (GATE2F, [(10, 9)], GATE1F),
    (GATE1F, [(1, 6), (1, 7), (11, 6), (11, 7)], ROUTE15),   # any street door
]


class ExpShareFetch(MansionStrike):
    """Reuses the Mansion strike's PROVEN interior primitives (live-BFS walk, go_warp, interact,
    battle/dialogue interrupt handling) — the mission is a two-floor gatehouse, the smallest
    interior any strike has ever toured. Every leg is verified by map-id flips and the hand-over
    by FLAG 0x256 — nothing here trusts a menu byte."""

    def __init__(self, camp, log, dbg_dir=None):
        super().__init__(camp, log, dbg_dir)
        self.deadline = time.time() + 600        # two floors; 10 min is generous

    def _leg(self, table):
        here = tuple(tv.map_id(self.b))
        return next((l for l in table if l[0] == here), None)

    def _ride(self, table, goal_map, label):
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
            if not any(self.go_warp(t, dest, f"{label}{t}") for t in cands):
                self.log(f"!! [{label}] every candidate warp failed on {tv.map_id(self.b)}")
                return False
        return tuple(tv.map_id(self.b)) == goal_map

    def run(self):
        b, camp = self.b, self.camp
        got = fm.read_flag(b, FLAG_GOT_EXP_SHARE)
        self.log(f"   expshare fetch: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"got_expshare={int(got)}")
        if got and tuple(tv.map_id(b)) == ROUTE15:
            self.log("   Exp. Share already claimed and back on Route 15 — nothing to strike")
            return "got_expshare"
        if not got:
            if not self._ride(CLIMB, GATE2F, "climb"):
                return self._bail()
            # THE AIDE: A opens his YES/NO check offer (A = the default YES), then the dex-count
            # check, the giveitem fanfare, and the Exp. Share explainer — all A-drainable.
            # Verify by the flag — set ONLY when the item actually landed in the bag.
            if not self.interact(AIDE, "oak-aide", key="A",
                                 verify=lambda: fm.read_flag(b, FLAG_GOT_EXP_SHARE)):
                return self._bail()
            self.log("   [pickup] EXP. SHARE obtained (flag 0x256 set)")
            try:
                camp.on_event("the EXP. SHARE! fifty Pokémon in the dex and the aide just pays "
                              "out — my whole bench levels off every fight now. this is how you "
                              "build a team for the Elite Four.", kind="item", tier=3)
            except Exception:
                pass
        else:
            self.log("   RESUME: Exp. Share already claimed mid-tour — running the walk-out only")
        if not self._ride(DESCEND, ROUTE15, "descend"):
            return self._bail()
        self.snap("90_expshare_out")
        self.log(f"   EXP-SHARE FETCH DONE: back on Route 15 @ {tv.coords(b)} | battles {self.n_battles}")
        return "got_expshare"

    def _bail(self):
        """A leg failed. Flag already set = the objective is DONE — report in_gate so the caller
        keeps the errand alive and the walk-out retries next tick; else it's a real failure."""
        if fm.read_flag(self.b, FLAG_GOT_EXP_SHARE):
            self.log("   fetch wedged AFTER the hand-over (0x256 set) — the exit owns the rest")
            return "in_gate"
        return "failed"


def run_strike(camp, log, dbg_dir=None):
    """Claim the Route 15 aide's Exp. Share from wherever she stands, in ONE call. Idempotent by
    state: flag-set + on Route 15 short-circuits; flag-set inside runs only the walk-out; a gate
    interior boot resumes from its own floor. The dex>=50 prerequisite lives in the GATE
    (_expshare_gate), not here — a mis-dispatched early strike just gets the aide's 'not enough'
    line, fails the flag verify, and reports failed (bounded, no loop)."""
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in EXPSHARE_ANCHORS:
        return "not_here"
    return ExpShareFetch(camp, log, dbg_dir).run()
