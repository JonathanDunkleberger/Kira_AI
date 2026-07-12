"""recon_overlevel_sealeg_check.py — decision-logic verifier for PASS-3 NS#5 team-depth lever:
campaign._overlevel_before_sealeg + _bench_to_ms_active. Proves the OVER-LEVEL-BEFORE-A-GRASS-LESS-SEA-LEG
trigger fires ONLY at the Blaine/Cinnabar crossing near-side while the bench is under milestone, is byte-inert
when the flag is OFF, and is park-safe (releases on leveled / poor-map / prep-dry / crossed), WITHOUT the
emulator, by binding the real methods to a lightweight fake campaign and driving the inputs.

  1  flag ON, Blaine, seafoam open, bench L25 floor, milestone 48        -> returns 48 (grind before crossing)
  2  flag OFF (default)                                                  -> None (byte-inert revert)
  3  bench floor within BENCH_MS_CLOSE of milestone (L43)                -> None (leveled — cross)
  4  next_gym is NOT a grass-less-crossing gym (Giovanni)               -> None (scoped to Blaine)
  5  seafoam gate CLOSED (already crossed / no Surf)                    -> None (auto-release post-crossing)
  6  this grass marked a poor spot ((map,ms) in _bench_poor_maps)       -> None (out-levelled — cross)
  7  PREP STAND-DOWN active (_prep_dry >= 2, no reachable grass)         -> None (can't grind here — cross)
  8  thin party (< 3, no real bench)                                    -> None
  9  post_game                                                          -> None
 10  no static milestone (planner returns 0)                            -> None
 11  _bench_to_ms_active: flag ON + pending, BENCH_TO_MILESTONE False    -> True (locally enables the climb)
 12  _bench_to_ms_active: flag OFF, BENCH_TO_MILESTONE False             -> False (no global behavior change)

RUN:  ../.venv/Scripts/python.exe -u recon_overlevel_sealeg_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign as C          # noqa: E402


class FakePlanner:
    def __init__(self, milestone):
        self._ms = milestone

    def ensure_plan(self, party, badges):
        return None

    def _next_milestone(self, badge_count, post_game):
        return ("Blaine", self._ms)


class FakeCamp:
    """Binds the REAL _overlevel_before_sealeg + _bench_to_ms_active; stubs everything they call so the
    DECISION is under test (not the RAM/travel/gate machinery)."""
    _overlevel_before_sealeg = C.Campaign._overlevel_before_sealeg
    _bench_to_ms_active = C.Campaign._bench_to_ms_active

    def __init__(self, milestone=48, seafoam_open=True, prep_dry=0, poor=None):
        self.team_planner = FakePlanner(milestone)
        self.b = object()
        self._prep_dry = prep_dry
        self._bench_poor_maps = set(poor or ())
        self._seafoam_open = seafoam_open

    def _seafoam_gate(self):
        # non-None object ⟺ has-Surf-not-yet-crossed window (the last grass before the sea)
        return object() if self._seafoam_open else None


def _mkstate(levels, gym="Blaine", badge=6, post_game=False):
    ng = {"leader": gym} if gym else None
    return {"party": [{"level": l} for l in levels], "badge_count": badge,
            "post_game": post_game, "next_gym": ng}


def main():
    # deterministic map read (Fuchsia/Route-18 side) for the poor-map key
    C.tv.map_id = lambda b: (3, 36)
    cases = []

    def check(name, got, want):
        ok = got == want
        cases.append((name, ok, got, want))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")

    # 1 — fires: flag ON, Blaine, seafoam open, bench under milestone
    C.OVERLEVEL_SEALEG_ENABLED = True
    fc = FakeCamp(milestone=48)
    check("1 fires at Blaine near-side under-milestone",
          fc._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), 48)

    # 2 — flag OFF → byte-inert
    C.OVERLEVEL_SEALEG_ENABLED = False
    check("2 flag OFF byte-inert",
          FakeCamp(milestone=48)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), None)

    # remaining cases run with the flag ON
    C.OVERLEVEL_SEALEG_ENABLED = True

    # 3 — bench floor within BENCH_MS_CLOSE of milestone → release (leveled)
    check("3 leveled (floor within CLOSE) releases",
          FakeCamp(milestone=48)._overlevel_before_sealeg(_mkstate([57, 43, 44, 43, 45, 43])), None)

    # 4 — not a grass-less-crossing gym
    check("4 non-Blaine gym scoped out",
          FakeCamp(milestone=52)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25], gym="Giovanni")), None)

    # 5 — seafoam gate closed (already crossed / no Surf)
    check("5 seafoam gate closed releases",
          FakeCamp(milestone=48, seafoam_open=False)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), None)

    # 6 — this grass marked a poor spot for the milestone
    check("6 poor-map releases (out-levelled grass)",
          FakeCamp(milestone=48, poor={((3, 36), 48)})._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), None)

    # 7 — prep-dry stand-down (no reachable grass)
    check("7 prep-dry stand-down releases",
          FakeCamp(milestone=48, prep_dry=2)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), None)

    # 8 — thin party
    check("8 thin party (<3) skipped",
          FakeCamp(milestone=48)._overlevel_before_sealeg(_mkstate([57, 25])), None)

    # 9 — post-game
    check("9 post-game skipped",
          FakeCamp(milestone=48)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25], post_game=True)), None)

    # 10 — no static milestone
    check("10 no milestone skipped",
          FakeCamp(milestone=0)._overlevel_before_sealeg(_mkstate([57, 29, 28, 29, 29, 25])), None)

    # 11 — _bench_to_ms_active: local enable while pending even with BENCH_TO_MILESTONE off
    C.BENCH_TO_MILESTONE = False
    check("11 bench_to_ms_active True when pending (BENCH_TO_MILESTONE off)",
          FakeCamp(milestone=48)._bench_to_ms_active(_mkstate([57, 29, 28, 29, 29, 25])), True)

    # 12 — _bench_to_ms_active: flag OFF + BENCH_TO_MILESTONE off → no behavior change
    C.OVERLEVEL_SEALEG_ENABLED = False
    check("12 bench_to_ms_active False when both off",
          FakeCamp(milestone=48)._bench_to_ms_active(_mkstate([57, 29, 28, 29, 29, 25])), False)

    npass = sum(1 for _, ok, _, _ in cases if ok)
    print(f"\n{npass}/{len(cases)} PASS")
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
