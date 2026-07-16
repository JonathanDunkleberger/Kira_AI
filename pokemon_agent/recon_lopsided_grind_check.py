"""recon_lopsided_grind_check.py — decision-logic verifier for PASS-3 NS#6 team-depth lever (a):
campaign._bench_severely_lopsided. Proves the SEVERELY-lopsided-bench trigger fires ONLY on the
solo-carry + dead-weight shape and is park-proof, WITHOUT the emulator, by binding the real method to a
lightweight fake campaign and driving the inputs (party levels / prep_t / milestone / hurt / questline /
flags / prep-dry / already-done) across the scenarios that matter:

  1  severely lopsided (ace L48, bench L10-13, milestone 45, pin armed) -> returns 45 (force ONE stint)
  2  prep_t None (pin not armed)                                        -> None (executor would grind the ACE)
  3  flag POKEMON_LOPSIDED_GRIND=0                                      -> None (instant revert)
  4  participation switch unavailable                                  -> None (can't level a bench w/o it)
  5  active nav-critical questline errand                              -> None (keep the true ace leading)
  6  party hurt / critically hurt                                      -> None (heal first)
  7  thin party (< 3, no real bench)                                   -> None
  8  no static milestone (post-game / planner off)                     -> None
  9  milestone already forced this badge (_lopsided_grind_done)        -> None (bounded: <= 1 stint/badge)
 10  MODEST milestone gap (bench trails ~5 under milestone)            -> None (road-bench-XP handles it)
 11  MODEST ace gap (uniformly underleveled, ace not towering)         -> None (not the solo-carry shape)
 12  PREP STAND-DOWN active (_prep_dry >= 2, no reachable grass)        -> None

RUN:  ../.venv/Scripts/python.exe -u recon_lopsided_grind_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign as C          # noqa: E402
import battle_agent as BA     # noqa: E402


class FakePlanner:
    def __init__(self, milestone):
        self._ms = milestone

    def ensure_plan(self, party, badges):
        # no-op stub (real TeamPlanner.ensure_plan builds/recomputes plan state); the campaign method
        # calls it before _next_milestone so the planner state is live — under test the milestone is
        # injected directly, so this just needs to exist and not raise.
        return None

    def _next_milestone(self, badge_count, post_game):
        return ("NextGym", self._ms)


class FakeCamp:
    """Binds the REAL _bench_severely_lopsided; stubs everything it calls so the DECISION is under test."""
    _bench_severely_lopsided = C.Campaign._bench_severely_lopsided

    def __init__(self, milestone, hurt=False, severity="fine", questline=False,
                 prep_dry=0, done=None):
        self.team_planner = FakePlanner(milestone)
        self._hurt = hurt
        self._severity = severity
        self._active_questline = object() if questline else None
        self._prep_dry = prep_dry
        self._lopsided_grind_done = set(done or ())

    def needs_heal(self):
        return ("hurt", "…") if self._hurt else None

    def _hurt_severity(self):
        return (self._severity, "…")


def _mkstate(levels, badge=4):
    return {"party": [{"level": l} for l in levels], "badge_count": badge, "post_game": False}


def _run(name, camp, levels, prep_t, expect):
    got = camp._bench_severely_lopsided(_mkstate(levels), prep_t)
    ok = (got == expect)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: levels={levels} prep_t={prep_t} -> {got} (want {expect})")
    return [] if ok else [f"{name}: got {got} want {expect}"]


def main():
    C.STRATEGIC_GRIND_ENABLED = True
    BA.GRIND_SWITCH_ENABLED = True
    C.LOPSIDED_GRIND_ENABLED = True
    C.LOPSIDED_MS_GAP = 12
    C.LOPSIDED_ACE_GAP = 15
    fails = []

    # 1 — the target shape: solo L48 carry behind an L10-13 bench, Koga milestone 45, pin armed at floor+6
    fails += _run("severely lopsided -> fire", FakeCamp(45), [48, 13, 10, 22], prep_t=16, expect=45)

    # 2 — pin not armed (prep_t None): must NOT fire (else the 'battle' executor grinds the ACE, not bench)
    fails += _run("prep_t None", FakeCamp(45), [48, 13, 10, 22], prep_t=None, expect=None)

    # 3 — flag off
    C.LOPSIDED_GRIND_ENABLED = False
    fails += _run("flag off", FakeCamp(45), [48, 13, 10, 22], prep_t=16, expect=None)
    C.LOPSIDED_GRIND_ENABLED = True

    # 4 — participation switch unavailable
    BA.GRIND_SWITCH_ENABLED = False
    fails += _run("switch unavailable", FakeCamp(45), [48, 13, 10, 22], prep_t=16, expect=None)
    BA.GRIND_SWITCH_ENABLED = True

    # 5 — mid nav-critical errand
    fails += _run("active questline", FakeCamp(45, questline=True), [48, 13, 10, 22], prep_t=16, expect=None)

    # 6 — hurt / critical
    fails += _run("party hurt", FakeCamp(45, hurt=True), [48, 13, 10, 22], prep_t=16, expect=None)
    fails += _run("critically hurt", FakeCamp(45, severity="critical"), [48, 13, 10, 22], prep_t=16, expect=None)

    # 7 — thin party (no real bench)
    fails += _run("thin party", FakeCamp(45), [48, 13], prep_t=16, expect=None)

    # 8 — no static milestone
    fails += _run("no milestone", FakeCamp(0), [48, 13, 10, 22], prep_t=16, expect=None)

    # 9 — milestone already forced this badge (bounded to <= 1 stint/badge)
    fails += _run("milestone already done", FakeCamp(45, done={45}), [48, 13, 10, 22], prep_t=16, expect=None)

    # 10 — MODEST milestone gap: bench L40 vs milestone 45 (gap 5 < 12) -> road-bench-XP handles it
    fails += _run("modest milestone gap", FakeCamp(45), [48, 40, 41, 42], prep_t=45, expect=None)

    # 11 — MODEST ace gap: uniformly underleveled (ace L20, bench L12) -> not the solo-carry shape
    fails += _run("modest ace gap", FakeCamp(45), [20, 12, 13, 14], prep_t=18, expect=None)

    # 12 — PREP STAND-DOWN (no reachable grass): _prep_dry >= 2 releases
    fails += _run("prep stand-down", FakeCamp(45, prep_dry=2), [48, 13, 10, 22], prep_t=16, expect=None)

    print("\n" + ("ALL PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
