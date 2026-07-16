"""env_puzzle.py — the general ENVIRONMENT-PUZZLE solver, instance #1: Vermilion's trash cans.

THE GENERAL SHAPE (the template gyms 4-8 inherit — spinners, statues, teleport pads, boulders):
a puzzle = SITES (interactable tiles enumerated from the LIVE map header) + HIDDEN STATE (the game
randomizes something) + FEEDBACK per interaction (readable flags/vars/dialogue) + a SUCCESS FLAG.
The solver runs an HONEST SEARCH: she visits sites in a human order, interacts, reads the outcome,
updates belief, narrates the hunt. RAM is used to VERIFY outcomes (did the switch flag set?), NEVER
to pick the answer — the constitution's honest-search law: she doesn't psychically know, she hunts,
she celebrates the find. (VAR_TEMP_0/1 DO hold the answer ids here; they are deliberately unread.)

INSTANCE #1 GROUND TRUTH (pret src/field_specials.c SetVermilionTrashCans + scripts.inc):
  - 15 cans in a 5-wide, 3-row grid; ids 1..15. The FIRST switch can is uniform-random; the SECOND
    is ALWAYS ADJACENT (idx ±1 same row / ±5 next row) — so her folk strategy "the next one's got
    to be right beside it!" is CORRECT (and endearing).
  - Feedback: finding switch 1 sets FLAG_TEMP_1 (0x001) + opens beam set 1. A WRONG second pick
    re-randomizes BOTH cans + beams back on (FLAG_TEMP_1 clears). Finding both sets
    FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES (0x264) — the door to Surge is open, PERMANENT.
  - Expected honest cost: ~8 checks to find #1, adjacent-first for #2 (2-4 neighbors) — ~3 rounds.
"""
import time

import firered_ram as ram
import travel as tv
from field_moves import read_flag

FLAG_TEMP_1 = 0x001
FLAG_BOTH_SWITCHES = 0x264
VERMILION_GYM = (9, 7)      # interior map id — VERIFIED LIVE on first entry (disasm order differs);
#                             callers should trust the gym they are STANDING IN over this constant.


class TrashCanPuzzle:
    """Honest search over the Vermilion gym's trash cans. Duck-types campaign (.b, .render,
    .trav, .on_event). run() -> 'solved' | 'already' | 'stuck'."""

    def __init__(self, campaign, log=print):
        self.c = campaign
        self.b = campaign.b
        self.log = log

    # ── sites: the can tiles from the LIVE map header (bg script events in the floor grid) ────
    def can_sites(self):
        """The 15 trash-can tiles, enumerated live. Cans are BG script events on the gym floor;
        we take all interior bg events and keep the dominant evenly-spaced cluster (a stray sign
        costs one honest interaction — harmless)."""
        evs = [xy for (xy, kind) in tv.read_bg_events(self.b) if kind <= 4]
        evs.sort(key=lambda t: (t[1], t[0]))
        return evs

    def _interact(self, site):
        """Walk beside the can and press A facing it; settle the dialogue; return True if reached."""
        x, y = site
        for stand, face in (((x, y + 1), "UP"), ((x, y - 1), "DOWN"),
                            ((x - 1, y), "RIGHT"), ((x + 1, y), "LEFT")):
            r = self.c.trav.travel(target_map=None, arrive_coord=stand, max_steps=80, max_seconds=45)
            if r == "arrived":
                for _ in range(2):                      # face (first press may be the turn) + talk
                    self.b.press(face, 8, 10, self.c.render, owner="agent")
                    for _f in range(14):
                        self.b.run_frame()
                self.b.press("A", 8, 12, self.c.render, owner="agent")
                for _f in range(40):
                    self.b.run_frame()
                self.c._drain_overworld(label="trashcan")   # walk the "just trash!/found a switch!" box
                return True
        return False

    def run(self, max_seconds=600):
        t0 = time.time()
        if read_flag(self.b, FLAG_BOTH_SWITCHES):
            return "already"
        ev = self.c.on_event
        ev("okay — the door's locked behind some switch hidden in these trash cans. gross, but "
           "fine: we're going bin-diving.", kind="gym", tier=2)
        sites = self.can_sites()
        if len(sites) < 6:
            self.log(f"   [puzzle] !! only {len(sites)} bg sites on this map — wrong room? LOUD")
            return "stuck"
        self.log(f"   [puzzle] trash-can hunt: {len(sites)} sites {sites}")
        rounds = 0
        while time.time() - t0 < max_seconds and rounds < 12:
            rounds += 1
            first = None
            # phase 1 — sweep for the first switch (row order; skip nothing: honest hunt)
            for s in sites:
                if time.time() - t0 > max_seconds:
                    return "stuck"
                if not self._interact(s):
                    continue
                if read_flag(self.b, FLAG_BOTH_SWITCHES):
                    ev("WAIT — that's BOTH switches?! the door's open! ha, trash pays off!",
                       kind="gym", tier=3)
                    return "solved"
                if read_flag(self.b, FLAG_TEMP_1):
                    first = s
                    ev("hang on — there's a SWITCH under the lid! one down. the second one's got "
                       "to be right beside it… right? that's how these things work.",
                       kind="gym", tier=2)
                    break
            if first is None:
                self.log("   [puzzle] !! swept every can, no first switch — LOUD (map/flag mismatch?)")
                return "stuck"
            # phase 2 — the adjacent hunt (her folk strategy; ALSO the game's real rule)
            fx, fy = first
            neighbors = [s for s in sites
                         if s != first and (abs(s[0] - fx) + abs(s[1] - fy) <= 3)]
            neighbors.sort(key=lambda s: abs(s[0] - fx) + abs(s[1] - fy))
            hit_reset = False
            for s in neighbors:
                if time.time() - t0 > max_seconds:
                    return "stuck"
                if not self._interact(s):
                    continue
                if read_flag(self.b, FLAG_BOTH_SWITCHES):
                    ev("YES — second switch! the beams are DOWN. okay Surge, no more hiding "
                       "behind your garbage.", kind="gym", tier=3)
                    return "solved"
                if not read_flag(self.b, FLAG_TEMP_1):
                    hit_reset = True                     # wrong pick -> the game re-randomized
                    ev("no no no — it reset?! the switches MOVED. okay. deep breath. "
                       "we go again.", kind="gym", tier=2)
                    break
            if not hit_reset:
                # neighbors exhausted without reset or win — odd; re-sweep (bounded by rounds)
                self.log("   [puzzle] neighbors exhausted without a reset — re-sweeping (odd state)")
        self.log(f"   [puzzle] !! unsolved after {rounds} rounds — LOUD")
        return "stuck"
