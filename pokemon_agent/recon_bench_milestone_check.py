"""recon_bench_milestone_check.py — decision-logic verifier for PASS-3 NS#10 BENCH-TO-MILESTONE climb
(campaign._prep_team_target's productivity-gated to-milestone re-pin). NS#9 proved the one-bite-per-badge
lopsided grind (36b4998) lands the bench ~25 levels UNDER the gym milestone (L20 vs Koga L45) → the
type-answers faint at L23 → she LOSES Koga. This lets the pin RE-PIN toward the milestone in +6 bites
(instead of retiring after one) UNLESS this map's grass proved a poor spot (marked by the executor's
productivity gate), in which case the pin retires so head_to_gym is restored and she marches to better
grass. Park-proof by the poor-map release. Flag POKEMON_BENCH_TO_MILESTONE (default OFF preserves 36b4998).

Rule-8 micro-test (isolating the ONE mechanism the NS#9 evidence fingered): boots a real bank for a live
Bridge, then drives crafted (party, badge) states with the active-wall path stubbed so ONLY the proactive
bench logic runs. Asserts (flag ON unless noted):

  1  bench floor 20, milestone 45, pin just crossed -> RE-PINS toward 45 (min(45, 26+6)=32), NOT retire
  2  keeps climbing: floor 32 -> RE-PINS to 38, then 44, then within CLOSE(6) of 45 -> retires (gym-ready)
  3  current map marked POOR for this milestone -> pin RETIRES (None) so she marches to better grass
  4  floor already within CLOSE of milestone (floor 40, ms 45) -> retires (None), never over-grinds
  5  FLAG OFF (default): after one +6 bite the pin RETIRES (36b4998 preserved byte-identically)
  6  no static milestone (post-game) -> falls back to lead-8, NO to-milestone re-pin (fallback untouched)

RUN:  POKEMON_BENCH_TO_MILESTONE=1 ../.venv/Scripts/python.exe -u recon_bench_milestone_check.py [state=surge_done_kit.state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# This script drives BOTH flag states explicitly per-case (module reads the env at import, so we can't
# rely on the process flag) — instead we toggle campaign.BENCH_TO_MILESTONE directly around each case.

from bridge import Bridge                                   # noqa: E402
import travel as tv                                         # noqa: E402
import campaign as C                                        # noqa: E402
from campaign import Campaign, resolve_state                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "surge_done_kit.state"
    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    camp = Campaign(b, battle_runner=lambda *a, **k: "win",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=lambda: None)
    camp.strat.underlevel_target = lambda: None
    camp.strat.active_wall_rec = lambda: None
    camp._prep_dry = 0
    cur_map = tuple(tv.map_id(b))
    print(f"boot={boot}; live map={cur_map}; CLOSE={C.BENCH_MS_CLOSE} BITE_MIN={C.BENCH_BITE_MIN}")

    # Force a known milestone so the test is deterministic regardless of the boot bank's plan.
    class _P:
        def ensure_plan(self, *a, **k):
            return None

        def _next_milestone(self, badge, post):
            return ("Koga", 0) if post else ("Koga", 45)
    camp.team_planner = _P()

    def mk(levels, badges, post=False):
        party = [{"species": f"m{i}", "level": lv} for i, lv in enumerate(levels)]
        return {"party": party, "badge_count": badges, "post_game": post,
                "map": list(cur_map), "place": ""}

    fails = []

    def want(label, got, expect, cond):
        ok = cond(got)
        print(f"{label}: -> {got}  (expect {expect})  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{label}: got {got}, expect {expect}")

    def reset(flag):
        C.BENCH_TO_MILESTONE = flag
        camp._bench_pin = None
        camp._bench_done_sig = None
        camp._bench_done_milestone = 0
        camp._bench_poor_maps = set()

    # ── (1) flag ON: pin just crossed at floor 20 -> re-pin toward milestone 45 (not retire) ──────────
    reset(True)
    # arm a pin at 20 (as if a prior bite set it), floor now 20 -> crosses -> should re-pin to min(45,26)=26?
    # We emulate "just crossed": set pin=20, floor=20. Re-pin = min(45, 20+6)=26.
    camp._bench_pin = 20
    want("(1) floor 20 == pin 20, ms 45, map OK", camp._prep_team_target(mk([52, 20, 20, 20, 20, 20], 4)),
         "26 (re-pin, NOT None)", lambda t: t == 26)
    want("    pin re-armed", getattr(camp, "_bench_pin", None), "26", lambda p: p == 26)

    # ── (2) keeps climbing in +6 bites, then retires within CLOSE of the milestone ───────────────────
    camp._bench_pin = 26
    want("(2a) floor 26 == pin 26 -> re-pin", camp._prep_team_target(mk([52, 26, 26, 26, 26, 26], 4)),
         "32", lambda t: t == 32)
    camp._bench_pin = 32
    want("(2b) floor 32 -> re-pin", camp._prep_team_target(mk([52, 32, 32, 32, 32, 32], 4)),
         "38", lambda t: t == 38)
    camp._bench_pin = 38
    want("(2c) floor 38 -> re-pin (min(45,44)=44)", camp._prep_team_target(mk([52, 38, 38, 38, 38, 38], 4)),
         "44", lambda t: t == 44)
    # floor 40 is within CLOSE(6) of ms 45 (40 >= 39) -> _keep_climbing False -> retire.
    camp._bench_pin = 40
    want("(2d) floor 40 within CLOSE of 45 -> retire", camp._prep_team_target(mk([52, 40, 40, 40, 40, 40], 4)),
         "None (gym-ready)", lambda t: t is None)

    # ── (3) current map marked POOR for this milestone -> pin retires (march to better grass) ─────────
    reset(True)
    camp._bench_pin = 20
    camp._bench_poor_maps = {(cur_map, 45)}
    want("(3) floor 20 but map is POOR -> retire", camp._prep_team_target(mk([52, 20, 20, 20, 20, 20], 4)),
         "None (march off poor grass)", lambda t: t is None)

    # ── (4) floor already within CLOSE of the milestone -> never over-grinds ─────────────────────────
    reset(True)
    camp._bench_pin = 40
    want("(4) floor 40 within CLOSE of 45 -> retire", camp._prep_team_target(mk([52, 40, 40, 40, 40, 40], 4)),
         "None", lambda t: t is None)

    # ── (5) FLAG OFF: one +6 bite then retire (36b4998 preserved) ────────────────────────────────────
    reset(False)
    camp._bench_pin = 20
    want("(5) flag OFF: floor 20 == pin 20 -> retire (one bite)",
         camp._prep_team_target(mk([52, 20, 20, 20, 20, 20], 4)), "None", lambda t: t is None)
    want("    pin retired", getattr(camp, "_bench_pin", None), "None", lambda p: p is None)

    # ── (6) flag ON but no static milestone (post-game) -> lead-8 fallback, no to-milestone re-pin ────
    reset(True)
    camp._bench_pin = 20
    # post_game -> _next_milestone returns 45 with post=... our stub returns 0 for post -> _ms falsy ->
    # milestone = lead-8 = 52-8 = 44. floor 20 < pin 20? no, ==, crosses. _keep_climbing needs _ms truthy
    # -> False -> retires (the fallback keeps its roster-only re-arm; no to-milestone climb).
    want("(6) post-game (no _ms): floor 20 == pin -> retire (fallback untouched)",
         camp._prep_team_target(mk([52, 20, 20, 20, 20, 20], 4, post=True)), "None", lambda t: t is None)

    print()
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("ALL BENCH-TO-MILESTONE DECISION CASES PASS")


if __name__ == "__main__":
    main()
