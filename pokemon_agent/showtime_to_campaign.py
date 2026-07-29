"""showtime_to_campaign.py — stage the SHOWTIME run (states/kira) as a free-roam campaign bundle.

WHY: the showtime spine (run_segments) is a scripted opener that ENDS when its segments end
(currently Misty) and retries a lost gym in place — it is not the full-game engine. Free-roam
(--resume --free-roam, the engine that finished fresh_go_6 to credits) resumes ONLY a campaign
bundle (kira_campaign.state + 4 sidecars). This tool carries the live showtime playthrough over:

  1. Finds the NEWEST savestate anywhere under states/kira/ (segment checkpoints +
     .progress.state banks — the freshest position the spine ever wrote).
  2. Stages a sanctity-shaped bundle in states/staging_showtime_<ts>/:
     state -> kira_campaign.state; sidecars from states/kira where the show wrote them
     (journey_core.json, pokemon_soul.json -> soul.json), minimal seeds where it didn't
     (world_model/strat_memory start empty and re-learn — same as any fresh campaign).
     team_plan_state.json is deliberately NOT carried: the planner re-derives from her real
     party so the current archetype file wins (the zero-catch bug was a stale/missing plan).
  3. ROUND-TRIP: boots the staged state in a fresh core, reads map/party/badges, and patches
     journey badge_count to the RAM truth.
  4. sanctity.validate_bundle on the stage (schema/encoding/truth; no prev -> no monotonic,
     the caller archives the old campaign first).

Prints `STAGED <abs path>` on success (resume_marathon.ps1 parses this), exits nonzero on any
failure. READ-ONLY toward states/kira and states/campaign — it only writes the staging dir.

RUN:  python pokemon_agent/showtime_to_campaign.py
"""
import glob
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sanctity                                            # noqa: E402

STATES = os.path.join(_HERE, "states")
KIRA = os.path.join(STATES, "kira")
MIN_STATE_BYTES = 100_000
_BADGE_FLAGS = range(0x820, 0x828)


def newest_state():
    cands = [p for p in glob.glob(os.path.join(KIRA, "**", "*.state"), recursive=True)
             if os.path.getsize(p) >= MIN_STATE_BYTES]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def copy_or_seed(stage, dst_name, src_names, seed):
    """Copy the first existing sidecar from states/kira (trying src_names in order), else write
    the minimal-valid seed. Returns 'carried <name>' or 'seeded'."""
    for nm in src_names:
        src = os.path.join(KIRA, nm)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage, dst_name))
            return f"carried {nm}"
    with open(os.path.join(stage, dst_name), "w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)
    return "seeded"


def main():
    src_state = newest_state()
    if not src_state:
        print(f"!! no usable .state (>= {MIN_STATE_BYTES} B) found under {KIRA}")
        return 1
    age_min = (time.time() - os.path.getmtime(src_state)) / 60
    print(f"newest showtime state: {src_state}  ({age_min:.0f} min old)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    stage = os.path.join(STATES, f"staging_showtime_{ts}")
    os.makedirs(stage, exist_ok=True)
    shutil.copy2(src_state, os.path.join(stage, "kira_campaign.state"))

    print("  soul.json:         " + copy_or_seed(stage, "soul.json",
          ("pokemon_soul.json", "soul.json"), {"bonds": {}, "wants": []}))
    print("  journey_core.json: " + copy_or_seed(stage, "journey_core.json",
          ("journey_core.json",),
          {"summary": "Fresh campaign carried over from my stream run — my Squirtle line and I "
                      "took Brock's badge, crossed Mt. Moon, and set up camp in Cerulean.",
           "badge_count": 1}))
    print("  strat_memory.json: " + copy_or_seed(stage, "strat_memory.json",
          ("strat_memory.json",), {"losses": {}}))
    print("  world_model.json:  " + copy_or_seed(stage, "world_model.json",
          ("world_model.json",), {"nodes": {}}))
    for opt in ("dialogue_hints.json",):
        if os.path.exists(os.path.join(KIRA, opt)):
            shutil.copy2(os.path.join(KIRA, opt), os.path.join(stage, opt))
            print(f"  {opt}: carried")

    # round-trip: boot the staged state, read RAM truth
    from bridge import Bridge                              # noqa: E402  (heavy import last)
    import travel as tv                                    # noqa: E402
    import firered_ram as ram                              # noqa: E402
    import pokemon_state as st                             # noqa: E402
    b = Bridge(os.path.join(os.path.dirname(_HERE), "roms", "firered.gba"))
    with open(os.path.join(stage, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    sb1 = b.rd32(0x03005008)
    badges = sum(1 for fl in _BADGE_FLAGS
                 if b.rd8(sb1 + 0x0EE0 + (fl >> 3)) & (1 << (fl & 7)))
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    party = [(st.SPECIES_NAME.get(st.read_party_species(b, s), "?"),
              b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54))
             for s in range(min(cnt, 6))]
    print(f"ROUND-TRIP: map={tv.map_id(b)} coords={tv.coords(b)} badges={badges} party={party}")

    # journey badge_count must match the savestate (sanctity TRUTH check)
    jp = os.path.join(stage, "journey_core.json")
    with open(jp, encoding="utf-8") as f:
        journey = json.load(f)
    journey["badge_count"] = badges
    if not (journey.get("summary") or "").strip():
        journey["summary"] = ("My stream campaign, carried over mid-journey: "
                              f"{badges} badge(s) in, party of {len(party)}.")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(journey, f, ensure_ascii=False, indent=2)

    ok, issues = sanctity.validate_bundle(stage, prev_dir=None, live_badges=badges, log=print)
    if not ok:
        print(f"!! staging FAILED sanctity: {issues}")
        return 1
    print(f"STAGED {os.path.abspath(stage)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
