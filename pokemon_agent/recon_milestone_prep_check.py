"""recon_milestone_prep_check.py — decision-logic verifier for the PASS-3 mid-game MILESTONE-CAP
bench-prep (the mid-game sibling of prep_for_e4).

Rule-8 micro-test (isolating ONE mechanism the diagnosis fingered): does Campaign._prep_team_target's
WALL-LESS proactive bench-raise now cap the pin at the team-plan's NEXT gym milestone (not the
ace-relative lead-8), climb toward it in +6 bites, do ONE bite per milestone (no over-grind), re-arm
when a NEW badge raises the milestone, and STILL guard the no-milestone fallback against the ship-run-5
treadmill? Boots a real bank for a live Bridge, then drives crafted (party, badge) states with the
active-wall path stubbed out so ONLY the proactive bench logic runs. Asserts:
  (1) underleveled bench, badge 2 (Surge=25) -> milestone-capped +6 bite (min(25, floor+6)), NOT lead-8
  (2) after the floor crosses the bite -> retires (None) — one bite per milestone, no over-grind
  (3) same badge, bench still under milestone -> NO re-arm (won't re-pin every tick = no treadmill)
  (4) new badge earned (milestone RISES) -> re-arms a fresh +6 bite toward the higher bar
  (5) floor already at/above the milestone -> None (never grinds past the gym's level)
  (6) no-milestone fallback (planner returns none) -> lead-8 target, and re-arm ONLY on roster change

RUN:  ../.venv/Scripts/python.exe -u recon_milestone_prep_check.py [state=giovanni_kit_g.state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                                   # noqa: E402
from campaign import Campaign, resolve_state                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "giovanni_kit_g.state"
    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    camp = Campaign(b, battle_runner=lambda *a, **k: "win",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=lambda: None)
    # Isolate the proactive bench path: no active wall, no prep stand-down.
    camp.strat.underlevel_target = lambda: None
    camp.strat.active_wall_rec = lambda: None
    camp._prep_dry = 0

    real = camp.read_live_state()
    camp.team_planner.ensure_plan(real["party"], real["badge_count"])
    ms = (getattr(camp.team_planner, "state", None) or {}).get("level_milestones") or {}
    print(f"boot={boot}; level_milestones Surge={ms.get('Lt. Surge')} Erika={ms.get('Erika')}")

    def mk(levels, badges, post=False):
        """A crafted state: party with the given levels (species = stable distinct ids) at `badges`."""
        party = [{"species": f"m{i}", "level": lv} for i, lv in enumerate(levels)]
        return {"party": party, "badge_count": badges, "post_game": post}

    fails = []

    def want(label, got, expect, cond):
        ok = cond(got)
        print(f"{label}: -> {got}  ({'expect ' + expect})  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{label}: got {got}, {expect}")

    # reset pin state
    camp._bench_pin = None
    camp._bench_done_sig = None
    camp._bench_done_milestone = 0

    # (1) badge 2 (Surge=25), bench floor 10, ace 40. milestone-capped bite = min(25, 16) = 16 (NOT lead-8=32).
    s = mk([40, 10, 12], 2)
    want("(1) underlevel bench @badge2", camp._prep_team_target(s), "16 (=min(25,10+6)), NOT 32",
         lambda t: t == 16)
    want("    pin armed", getattr(camp, "_bench_pin", None), "16", lambda p: p == 16)

    # (2) floor crosses the bite (10 -> 16) -> retires to None (one bite per milestone; no over-grind to 25).
    s = mk([40, 16, 16], 2)
    want("(2) floor reaches the bite", camp._prep_team_target(s), "None (retired)", lambda t: t is None)
    want("    milestone recorded", getattr(camp, "_bench_done_milestone", 0), "25", lambda m: m == 25)

    # (3) same badge, bench still under the milestone (16<25) -> NO re-arm (won't treadmill toward 25).
    s = mk([42, 16, 16], 2)
    want("(3) same badge, no re-arm", camp._prep_team_target(s), "None (no treadmill)", lambda t: t is None)

    # (4) new badge (Erika=31 > 25) -> re-arm a fresh +6 bite toward the higher bar: min(31, 16+6)=22.
    s = mk([44, 16, 16], 3)
    want("(4) new badge raises milestone", camp._prep_team_target(s), "22 (=min(31,16+6))", lambda t: t == 22)

    # (5) floor already at/above the milestone -> None (never grinds past the gym's level).
    camp._bench_pin = None
    s = mk([44, 32, 33], 3)                     # floor 32 > Erika 31
    want("(5) floor >= milestone", camp._prep_team_target(s), "None (no over-grind)", lambda t: t is None)

    # (6) FALLBACK: force no milestone -> lead-8 target, re-arm ONLY on roster change (ship-run-5 guard holds).
    camp.team_planner._next_milestone = lambda *a, **k: (None, 0)
    camp._bench_pin = None
    camp._bench_done_sig = None
    camp._bench_done_milestone = 0
    s = mk([40, 10, 12], 3)                     # lead-8 = 32; bite = min(32, 16) = 16
    want("(6a) fallback lead-8 bite", camp._prep_team_target(s), "16 (=min(32,16))", lambda t: t == 16)
    s = mk([40, 16, 16], 3)                     # retire the bite (same roster)
    camp._prep_team_target(s)
    s = mk([44, 16, 16], 3)                     # ace pulled ahead, SAME roster -> must NOT re-arm
    want("(6b) fallback no-treadmill", camp._prep_team_target(s), "None (roster unchanged)",
         lambda t: t is None)
    s = mk([44, 16, 16, 5], 3)                  # a NEW mon joined -> roster changed -> re-arm allowed
    want("(6c) fallback re-arm on new mon", camp._prep_team_target(s), "11 (=min(36,5+6), new floor 5)",
         lambda t: t == 11)

    print("\n" + ("PASS — milestone-cap bench-prep: capped, one-bite-per-milestone, badge-re-arm, "
                  "fallback treadmill-safe" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
