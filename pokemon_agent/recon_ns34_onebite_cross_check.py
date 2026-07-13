"""recon_ns34_onebite_cross_check.py — decision verifier for NS#34 OVER-LEVEL ONE-BITE-THEN-CROSS
(campaign.py ~L11929, the fix that root-kills the shifts 30-33 Route-18 soft-treadmill).

The fix: in the grind executor's PRODUCTIVE-BITE branch, when the over-level-before-sealeg is the driver
(_ovl_sealeg_pending True — NOT the global BENCH_TO_MILESTONE flag), after ONE productive bite it retires
the over-level for THIS badge by setting _overlevel_sealeg_dry_badge = badge (+ clears _ovl_sealeg_pending,
+ marks _lopsided_grind_done). That release flag makes the REAL _overlevel_before_sealeg return None next
tick, which (a) stops LOPSIDED re-firing AND (b) lets _ensure_forward_questline open the Seafoam crossing →
she CROSSES instead of treadmilling toward the UNREACHABLE Blaine L48 milestone at Route-18's L23-29 grass.

This verifies, WITHOUT the emulator, that the exact side effects the branch performs cause the release. It
replays the branch's assignment block verbatim, then calls the REAL _overlevel_before_sealeg + confirms:
  A  BEFORE the bite: over-level ACTIVE (pending, returns the milestone) — the treadmill state.
  B  the branch is scoped to _ovl_sealeg_pending (global BENCH_TO_MILESTONE multi-bite mode untouched).
  C  AFTER the bite: over-level RELEASED (returns None) → crossing opens → she crosses (one-bite-then-cross).
  D  the release is per-BADGE (a NEW badge re-arms the over-level — not a permanent kill).

RUN:  ../.venv/Scripts/python.exe -u recon_ns34_onebite_cross_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign as C          # noqa: E402


class FakePlanner:
    def __init__(self, ms):
        self._ms = ms

    def ensure_plan(self, party, badges):
        return None

    def _next_milestone(self, badge_count, post_game):
        return ("Blaine", self._ms)


class FakeCamp:
    _overlevel_before_sealeg = C.Campaign._overlevel_before_sealeg
    _bench_to_ms_active = C.Campaign._bench_to_ms_active

    def __init__(self, milestone=48):
        self.team_planner = FakePlanner(milestone)
        self.b = object()
        self._prep_dry = 0
        self._bench_poor_maps = set()
        self._seafoam_open = True
        self._overlevel_sealeg_dry_badge = -1
        self._ovl_sealeg_pending = True          # over-level is the active driver (set at campaign.py ~9838)
        self._lopsided_grind_done = set()

    def _seafoam_gate(self):
        return object() if self._seafoam_open else None

    def apply_onebite_release(self, state, _lop_ms):
        """The EXACT side-effect block from the productive-bite branch (campaign.py ~L11929)."""
        if getattr(self, "_ovl_sealeg_pending", False):
            self._overlevel_sealeg_dry_badge = int(state.get("badge_count", -1))
            self._ovl_sealeg_pending = False
            self._lopsided_grind_done = getattr(self, "_lopsided_grind_done", set())
            self._lopsided_grind_done.add(_lop_ms)


def _mkstate(levels, badge=6):
    return {"party": [{"level": l} for l in levels], "badge_count": badge,
            "post_game": False, "next_gym": {"leader": "Blaine"}}


def main():
    C.tv.map_id = lambda b: (3, 36)
    C.OVERLEVEL_SEALEG_ENABLED = True
    C.BENCH_TO_MILESTONE = False          # the real running default — the over-level is the ONLY driver
    cases = []

    def check(name, got, want):
        ok = got == want
        cases.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")

    # the Route-18 treadmill state: badge 6, ace L58, bench floor L30, milestone L48 (unreachable here)
    st = _mkstate([58, 30, 30, 30, 31, 30], badge=6)

    # A — BEFORE the bite: over-level active (would keep re-firing LOPSIDED = the shifts 30-33 treadmill)
    fc = FakeCamp(milestone=48)
    check("A over-level ACTIVE before the bite (treadmill state)",
          fc._overlevel_before_sealeg(st), 48)

    # B — the release block is scoped to _ovl_sealeg_pending: if NOT the over-level driver, no release fires
    fc_flagmode = FakeCamp(milestone=48)
    fc_flagmode._ovl_sealeg_pending = False          # e.g. BENCH_TO_MILESTONE multi-bite mode
    fc_flagmode.apply_onebite_release(st, 48)
    check("B scoped: BENCH_TO_MILESTONE mode leaves dry_badge untouched",
          fc_flagmode._overlevel_sealeg_dry_badge, -1)

    # C — AFTER one productive bite: apply the release → over-level RELEASES (None) → crossing opens
    fc.apply_onebite_release(st, 48)
    check("C over-level RELEASED after one bite (crosses to Cinnabar)",
          fc._overlevel_before_sealeg(st), None)
    check("C2 _ovl_sealeg_pending cleared", fc._ovl_sealeg_pending, False)
    check("C3 LOPSIDED milestone marked done (no re-fire)", 48 in fc._lopsided_grind_done, True)

    # D — per-BADGE: at the NEXT badge (7) the over-level re-arms (release was badge-6-scoped, not permanent)
    st7 = _mkstate([58, 30, 30, 30, 31, 30], badge=7)
    check("D re-arms at the next badge (per-badge, not a permanent kill)",
          fc._overlevel_before_sealeg(st7), 48)

    npass = sum(1 for _, ok in cases if ok)
    print(f"\n{npass}/{len(cases)} PASS")
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
