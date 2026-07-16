"""recon_ungate_check.py — focused mechanism check for the PASS-3 catch un-gate.

Does _plan_wants_prebuild now stay load-bearing PAST 3 mons when a planned keeper is DUE and catchable
on the CURRENT map (the old `pc>2 → False` gate is why slots 4-6 were never pursued)? Boots a state that
sits on a keeper route with party>2 (default ss_ticket: post-Bill on Route 25, 4 mons, no Abra yet — Abra
is the DUE psychic-sweeper keeper for Route 24/25) and prints the decision, plus a control (map with no
on-map DUE keeper should NOT keep suppressing the forward march).

RUN:  ../.venv/Scripts/python.exe -u recon_ungate_check.py [state=ss_ticket.state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import travel as tv                                         # noqa: E402
from bridge import Bridge                                   # noqa: E402
from campaign import Campaign, resolve_state                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "ss_ticket.state"
    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    camp = Campaign(b, battle_runner=lambda *a, **k: "win",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=lambda: None)
    state = camp.read_live_state()
    camp.team_planner.ensure_plan(state["party"], state["badge_count"])

    act = camp.team_planner.assess(state["party"], state["badge_count"],
                                   dex=state.get("dex_caught"), post_game=state.get("post_game"))
    kt = camp._plan_keeper_target(state)
    pb = camp._plan_wants_prebuild(state)
    pc = state["party_count"]
    print(f"boot={boot} map={tv.map_id(b)} ({state['place']}) badges={state['badge_count']}")
    print(f"party ({pc}): {[(m['species'], m['level']) for m in state['party']]}")
    print(f"assess() -> kind={act.get('kind')!r} species={act.get('species')!r} where={act.get('where')!r}")
    print(f"_plan_keeper_target (on THIS map) -> {kt!r}")
    print(f"_plan_wants_prebuild -> {pb}   (party_count={pc})")

    fails = []
    if pc > 2:
        if kt and not pb:
            fails.append("on-map DUE keeper present but prebuild=False -> un-gate NOT working (still pc>2 gated)")
        if not kt and pb:
            fails.append("no on-map keeper yet prebuild=True -> would suppress the march with nothing to catch")
        if not kt and not pb:
            print("\nCONTROL OK: party>2, no on-map DUE keeper here -> prebuild False -> forward-march resumes")

    # FORCED-POSITIVE: simulate standing on the keeper's route (Abra IS on this map) and confirm the
    # un-gate keeps building PAST 3 mons (the exact branch the old `pc>2 -> False` gate blocked).
    if act.get("kind") == "catch_keeper" and pc > 2 and pc < 6:
        orig = camp._species_on_map
        camp._species_on_map = lambda species, mid: True     # pretend the DUE keeper lives underfoot
        try:
            kt2 = camp._plan_keeper_target(state)
            pb2 = camp._plan_wants_prebuild(state)
        finally:
            camp._species_on_map = orig
        print(f"\nFORCED (keeper on-map): _plan_keeper_target -> {kt2!r} ; _plan_wants_prebuild -> {pb2}")
        if not (kt2 and pb2):
            fails.append(f"un-gate branch broken: with the keeper on-map, expected build=True, got kt={kt2} pb={pb2}")
        else:
            print("UN-GATE ACTIVE: party>2 AND a planned keeper catchable here -> keeps building (correct)")

    print("\nPASS" if not fails else "\nFAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
