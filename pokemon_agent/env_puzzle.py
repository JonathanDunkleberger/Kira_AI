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
  - Live can tiles (VermilionCity_Gym/map.json): x∈{1,3,5,7,9}, y∈{10,12,14}. Pitch 2×2.
    TWO gym statues at (3,17)/(7,17) are ALSO bg script events — must NOT be treated as cans.
  - Feedback: finding switch 1 sets FLAG_TEMP_1 (0x001) + opens beam set 1. A WRONG second pick
    re-randomizes BOTH cans + beams back on (FLAG_TEMP_1 clears). Finding both sets
    FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES (0x264) — the door to Surge is open, PERMANENT.
  - Expected honest cost: ~8 checks to find #1, adjacent-first for #2 (2-4 neighbors) — ~3 rounds.
  - Leave/re-enter (map transition) re-rolls TEMP switches via OnTransition InitTrashCans; FLAG_BOTH
    stays. Used only as bounded stream-ok recovery after honest search stalls.
"""
import time

import travel as tv
from field_moves import read_flag

try:
    from dialogue_drive import box_open as dd_box_open
except Exception:  # pragma: no cover — headless unit tests stub this
    def dd_box_open(_b):
        return False

FLAG_TEMP_1 = 0x001
FLAG_BOTH_SWITCHES = 0x264
VERMILION_GYM = (9, 7)      # interior map id — VERIFIED LIVE on first entry (disasm order differs);
#                             callers should trust the gym they are STANDING IN over this constant.
VERMILION_CITY = (3, 5)
VERMILION_GYM_DOOR = (14, 25)

# pret map.json ground truth — the 5×3 can lattice (id 1..15 row-major).
CAN_XS = (1, 3, 5, 7, 9)
CAN_YS = (10, 12, 14)
CAN_PITCH_X = 2
CAN_PITCH_Y = 2


def filter_can_sites(evs):
    """Keep the 5×3 trash-can cluster; drop gym statues / stray signs.

    Prefers the pret lattice when those tiles appear in the live bg list; otherwise keeps the
    dominant evenly-spaced interior cluster (x odd, y in {10,12,14}-like pitch).
    """
    pts = sorted({(int(x), int(y)) for (x, y) in evs}, key=lambda t: (t[1], t[0]))
    if not pts:
        return []
    lattice = {(x, y) for y in CAN_YS for x in CAN_XS}
    hit = [p for p in pts if p in lattice]
    # Any pret-lattice hit beats statues — even a partial read (load flake) is safer than
    # treating (3,17)/(7,17) GymStatue tiles as cans.
    if len(hit) >= 3:
        return sorted(hit, key=lambda t: (t[1], t[0]))
    # Fallback: drop the statue row (y>=16) and keep remaining interior bg scripts.
    interior = [p for p in pts if p[1] < 16]
    if interior:
        return sorted(interior, key=lambda t: (t[1], t[0]))
    return pts


def adjacent_cans(first, sites):
    """True 5×3 can-grid neighbors (pret: idx ±1 same row / ±5 next row).

    Tile rule on the live lattice: same row ±pitch_x, or same col ±pitch_y — never Manhattan
    ball (that pulled in statues at Manhattan=3 from the bottom row).
    """
    fx, fy = first
    sites = list(sites)
    # Infer pitch from the live cluster when possible (usually 2/2).
    xs = sorted({x for x, y in sites if y == fy and x != fx})
    ys = sorted({y for x, y in sites if x == fx and y != fy})
    px = CAN_PITCH_X
    py = CAN_PITCH_Y
    if xs:
        gaps = [abs(x - fx) for x in xs]
        px = min(gaps) if gaps else px
    if ys:
        gaps = [abs(y - fy) for y in ys]
        py = min(gaps) if gaps else py
    out = []
    for s in sites:
        if s == first:
            continue
        sx, sy = s
        if sy == fy and abs(sx - fx) == px:
            out.append(s)
        elif sx == fx and abs(sy - fy) == py:
            out.append(s)
    out.sort(key=lambda s: abs(s[0] - fx) + abs(s[1] - fy))
    return out


class TrashCanPuzzle:
    """Honest search over the Vermilion gym's trash cans. Duck-types campaign (.b, .render,
    .trav, .on_event). run() -> 'solved' | 'already' | 'stuck'."""

    def __init__(self, campaign, log=print):
        self.c = campaign
        self.b = campaign.b
        self.log = log

    # ── sites: the can tiles from the LIVE map header (bg script events in the floor grid) ────
    def can_sites(self):
        """The 15 trash-can tiles, enumerated live + filtered (statues are NOT cans)."""
        evs = [xy for (xy, kind) in tv.read_bg_events(self.b) if kind <= 4]
        return filter_can_sites(evs)

    def _interact(self, site):
        """Walk beside the can and press A facing it; settle the dialogue.

        Returns True only when an interaction actually registered (dialogue opened and/or a
        switch flag flipped). Prefer stand SOUTH / face UP — FRLG cans face that way.
        """
        x, y = site
        # Prefer south (UP), then east/west, then north — NPCs often park on the aisle tiles.
        stands = (((x, y + 1), "UP"), ((x + 1, y), "LEFT"),
                  ((x - 1, y), "RIGHT"), ((x, y - 1), "DOWN"))
        temp0 = read_flag(self.b, FLAG_TEMP_1)
        both0 = read_flag(self.b, FLAG_BOTH_SWITCHES)
        if dd_box_open(self.b):
            self.c._drain_overworld(label="trashcan-pre")
        for stand, face in stands:
            r = self.c.trav.travel(target_map=None, arrive_coord=stand,
                                   max_steps=140, max_seconds=70)
            if r != "arrived":
                self.log(f"   [puzzle] travel->{stand} for can{site}: {r}")
                continue
            # Face the can (first press may only turn) then A; confirm the box/flag moved.
            for _ in range(2):
                self.b.press(face, 8, 10, self.c.render, owner="agent")
                for _f in range(14):
                    self.b.run_frame()
            self.b.press("A", 8, 12, self.c.render, owner="agent")
            saw_box = False
            for _f in range(50):
                self.b.run_frame()
                if dd_box_open(self.b):
                    saw_box = True
                    break
            self.c._drain_overworld(label="trashcan")
            temp1 = read_flag(self.b, FLAG_TEMP_1)
            both1 = read_flag(self.b, FLAG_BOTH_SWITCHES)
            if saw_box or temp1 != temp0 or both1 != both0:
                self.log(f"   [puzzle] checked can{site} from {stand} face {face} "
                         f"(box={int(saw_box)} temp1={int(temp1)} both={int(both1)})")
                return True
            self.log(f"   [puzzle] A at can{site} from {stand} did not register — trying next side")
        self.log(f"   [puzzle] !! could not interact with can{site} from any side")
        return False

    def _recover_leave_reenter(self):
        """Bounded stream-ok reset: exit gym → heal → re-enter (re-rolls TEMP switches)."""
        self.log("   [puzzle] recovery: leave gym → heal → re-enter (re-roll switches)")
        self.c.on_event("okay these cans are messing with me — I'm stepping out, healing, "
                        "and coming back fresh.", kind="gym", tier=2)
        try:
            self.c._exit_to_overworld()
        except Exception as e:
            self.log(f"   [puzzle] !! exit failed: {e}")
            return False
        for _ in range(30):
            self.b.run_frame()
        try:
            hr = self.c.heal_nearest()
            self.log(f"   [puzzle] recovery heal -> {hr}")
        except Exception as e:
            self.log(f"   [puzzle] recovery heal skipped: {e}")
        # Re-enter Vermilion Gym from the city door (Cut tree already cleared if she was inside).
        if tv.map_id(self.b) != VERMILION_CITY:
            # Best-effort: if heal left her elsewhere, try travel back via campaign warp enter only
            # when already in Vermilion; otherwise abort recovery (caller keeps hunting or stucks).
            self.log(f"   [puzzle] recovery: not in Vermilion after exit "
                     f"(map={tv.map_id(self.b)}) — cannot re-enter cleanly")
            return False
        if self.c.enter_warp(pick=VERMILION_GYM_DOOR) != "warped":
            self.log("   [puzzle] !! recovery re-enter failed")
            return False
        for _ in range(45):
            self.b.run_frame()
        ok = tv.map_id(self.b) == VERMILION_GYM or tv.map_id(self.b)[0] != 3
        self.log(f"   [puzzle] recovery re-enter -> map={tv.map_id(self.b)} ok={ok}")
        return ok

    def run(self, max_seconds=600):
        t0 = time.time()
        if read_flag(self.b, FLAG_BOTH_SWITCHES):
            self.log("   [puzzle] FLAG_BOTH already set — beams open")
            return "already"
        ev = self.c.on_event
        ev("okay — the door's locked behind some switch hidden in these trash cans. gross, but "
           "fine: we're going bin-diving.", kind="gym", tier=2)
        sites = self.can_sites()
        if len(sites) < 6:
            self.log(f"   [puzzle] !! only {len(sites)} can sites on this map — wrong room? LOUD")
            return "stuck"
        self.log(f"   [puzzle] trash-can hunt: {len(sites)} sites {sites}")
        rounds = 0
        recoveries = 0
        while time.time() - t0 < max_seconds and rounds < 12:
            rounds += 1
            first = None
            checked = 0
            failed = 0
            # phase 1 — sweep for the first switch (row order; skip nothing: honest hunt)
            for s in sites:
                if time.time() - t0 > max_seconds:
                    return "stuck"
                if not self._interact(s):
                    failed += 1
                    continue
                checked += 1
                if read_flag(self.b, FLAG_BOTH_SWITCHES):
                    ev("WAIT — that's BOTH switches?! the door's open! ha, trash pays off!",
                       kind="gym", tier=3)
                    return "solved"
                if read_flag(self.b, FLAG_TEMP_1):
                    first = s
                    ev("hang on — there's a SWITCH under the lid! one down. the second one's got "
                       "to be right beside it… right? that's how these things work.",
                       kind="gym", tier=2)
                    self.log(f"   [puzzle] FIRST switch at {s} (round {rounds})")
                    break
            if first is None:
                self.log(f"   [puzzle] !! swept cans, no first switch "
                         f"(checked={checked} failed_interact={failed} sites={len(sites)})")
                # Travel/NPC thrash: leave-heal-reenter re-rolls + clears aisle blockers once.
                if recoveries < 2 and (failed > 0 or checked > 0):
                    recoveries += 1
                    if self._recover_leave_reenter():
                        sites = self.can_sites()
                        continue
                return "stuck"
            # phase 2 — the adjacent hunt (her folk strategy; ALSO the game's real rule)
            neighbors = adjacent_cans(first, sites)
            self.log(f"   [puzzle] adjacent to {first}: {neighbors}")
            if not neighbors:
                self.log("   [puzzle] !! no grid neighbors — site filter broken? re-sweep")
                continue
            hit_reset = False
            for s in neighbors:
                if time.time() - t0 > max_seconds:
                    return "stuck"
                if not self._interact(s):
                    continue
                if read_flag(self.b, FLAG_BOTH_SWITCHES):
                    ev("YES — second switch! the beams are DOWN. okay Surge, no more hiding "
                       "behind your garbage.", kind="gym", tier=3)
                    self.log(f"   [puzzle] BOTH switches — second was {s}")
                    return "solved"
                if not read_flag(self.b, FLAG_TEMP_1):
                    hit_reset = True                     # wrong pick -> the game re-randomized
                    ev("no no no — it reset?! the switches MOVED. okay. deep breath. "
                       "we go again.", kind="gym", tier=2)
                    self.log(f"   [puzzle] wrong second {s} — reset (round {rounds})")
                    break
            if not hit_reset and not read_flag(self.b, FLAG_BOTH_SWITCHES):
                # neighbors exhausted without reset or win — odd; re-sweep (bounded by rounds)
                self.log("   [puzzle] neighbors exhausted without a reset — re-sweeping (odd state)")
                if recoveries < 2 and rounds >= 3:
                    recoveries += 1
                    if self._recover_leave_reenter():
                        sites = self.can_sites()
        self.log(f"   [puzzle] !! unsolved after {rounds} rounds — LOUD")
        return "stuck"
