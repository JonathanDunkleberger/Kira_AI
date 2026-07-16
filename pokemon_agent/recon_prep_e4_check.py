"""recon_prep_e4_check.py — focused mechanism check for the PASS-3 prep_for_e4 wiring.

Rule-8 micro-test (isolating ONE mechanism the diagnosis fingered): does Campaign._prep_e4_target /
_prep_team_target now floor the WHOLE team to the E4 milestone at badge 8, WITHOUT livelocking on the
unboxable chaff? Boots a real badge-8 bank (giovanni_kit_g: Venusaur L68 ace + Lapras L37 / Kadabra L39
levelable + L8-14 chaff) and asserts the four cases:
  (1) underleveled bench  -> returns the E4 entry bar (~55)
  (2) whole team crossed  -> None (retires, she proceeds to VR)
  (3) only chaff under bar -> None (no livelock; chaff is box fodder, never a grind target)
  (4) levelable mons stalled -> None (retires when the reachable grass can't level them)

RUN:  ../.venv/Scripts/python.exe -u recon_prep_e4_check.py [state=giovanni_kit_g.state]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                                   # noqa: E402
import firered_ram as ram                                   # noqa: E402
import pokemon_state as st                                  # noqa: E402
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
    state = camp.read_live_state()
    camp.team_planner.ensure_plan(state["party"], state["badge_count"])
    ms = (getattr(camp.team_planner, "state", None) or {}).get("level_milestones") or {}

    lvls = [(m["species"], m["level"]) for m in state["party"]]
    print(f"boot={boot} badges={state['badge_count']} post_game={state['post_game']}")
    print(f"party: {lvls}")
    print(f"level_milestones: {ms}")

    fails = []

    # (1) underleveled bench -> E4 entry bar
    t1 = camp._prep_e4_target(state, state["party"])
    tt = camp._prep_team_target(state)
    print(f"\n(1) underleveled bench: _prep_e4_target -> {t1} ; _prep_team_target -> {tt}")
    if not (t1 and 50 <= t1 <= 58):
        fails.append(f"(1) expected E4 entry ~55, got {t1}")
    if tt != t1:
        fails.append(f"(1) _prep_team_target should DEFER to E4 target ({t1}), got {tt}")

    # (2) whole team crossed -> None
    crossed = {**state, "party": [dict(m, level=max(m["level"], 60)) for m in state["party"]]}
    t2 = camp._prep_e4_target(crossed, crossed["party"])
    print(f"(2) whole team >=60: -> {t2}  (expect None)")
    if t2 is not None:
        fails.append(f"(2) expected None once crossed, got {t2}")

    # (3) only chaff under the bar (real mons all high, plus L8-14 fodder) -> None (no livelock)
    chaffy = {**state, "party": [
        dict(state["party"][0], level=68),           # ace, crossed
        dict(state["party"][1], level=13),           # chaff (>25 under)
        dict(state["party"][2], level=8),            # chaff
        dict(state["party"][3], level=60),           # real mon, crossed
        dict(state["party"][4], level=14),           # chaff
        dict(state["party"][5], level=60),           # real mon, crossed
    ]}
    t3 = camp._prep_e4_target(chaffy, chaffy["party"])
    print(f"(3) only L8-14 chaff under bar: -> {t3}  (expect None — chaff is box fodder, not a grind target)")
    if t3 is not None:
        fails.append(f"(3) chaff must NOT pin the grind (livelock), got {t3}")

    # (4) the levelable mons are stall-marked -> None (retires so she proceeds)
    pids = set()
    for i, m in enumerate(state["party"]):
        if m["level"] < 55 and m["level"] >= 55 - 25:
            pids.add(b.rd32(ram.GPLAYER_PARTY + i * st.PARTY_MON_SIZE))
    camp._grind_stalled = set(pids)
    t4 = camp._prep_e4_target(state, state["party"])
    print(f"(4) levelable mons stalled ({len(pids)} pid): -> {t4}  (expect None — reachable grass can't level them)")
    if t4 is not None:
        fails.append(f"(4) stalled levelable mons must retire the target, got {t4}")
    camp._grind_stalled = set()

    print("\n" + ("PASS — prep_for_e4 floors the team + is livelock-proof" if not fails
                  else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
