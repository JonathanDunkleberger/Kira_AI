"""archive_sweep.py — ONE-TIME save-hygiene sweep (Part A). Moves the flat states/ pile into the
three lineages: the small sherpa/live set -> states/workshop/, everything else (handicapped-era
teaching captures + orphaned scratch/recon states) -> states/archive/. Nothing is deleted, so a
misclassification is one move back. resolve_state() finds a state in any bucket, so live boots/gates
keep working after the sweep. Re-runnable (idempotent): only touches *.state files in the FLAT dir.

  python pokemon_agent/archive_sweep.py          # do the sweep
  python pokemon_agent/archive_sweep.py --dry     # print the plan only
"""
import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = os.path.join(HERE, "states")
WORKSHOP = os.path.join(STATES, "workshop")
ARCHIVE = os.path.join(STATES, "archive")

# The SMALL set that stays in the live path (states/workshop/): the sherpa checkpoints we jump
# between to lay hooks + every state a LIVE handler loads (boots + remaining GATE_NEEDS_STATE).
KEEP_WORKSHOP = {
    # sherpa milestones (jump up/down the game)
    "og_postopening.state",     # "beginning" — clean post-opening Pallet
    "brock_done.state",         # post-Boulder badge, Pewter
    "mtmoon_cleared.state",     # mtmoon_done — emerged from the cave
    "misty_done.state",         # post-Cascade badge
    # live-loaded by campaign/play_live (boots + gates) — keep until each skip is AUTO
    "viridian_parcel_done.state",   # parcel GATE + play_live default boot
    "route3_caught.state",          # catch GATE (until Skip 2 lands balls)
    "mtmoon_interior.state",        # Mt Moon approach GATE (until Skip 3)
    "brock_ready.state",            # play_live --fast boot
    "after_pick_bulbasaur.state",   # campaign.py main default boot (legacy)
}


def main():
    dry = "--dry" in sys.argv
    os.makedirs(WORKSHOP, exist_ok=True)
    os.makedirs(ARCHIVE, exist_ok=True)
    flat = sorted(f for f in os.listdir(STATES)
                  if f.endswith(".state") and os.path.isfile(os.path.join(STATES, f)))
    to_workshop = [f for f in flat if f in KEEP_WORKSHOP]
    to_archive = [f for f in flat if f not in KEEP_WORKSHOP]
    print(f"FLAT states/: {len(flat)}  ->  workshop:{len(to_workshop)}  archive:{len(to_archive)}")
    print(f"\nKEEP -> workshop/ ({len(to_workshop)}):")
    for f in to_workshop:
        print(f"   {f}")
        if not dry:
            shutil.move(os.path.join(STATES, f), os.path.join(WORKSHOP, f))
    missing = sorted(KEEP_WORKSHOP - set(flat) - set(os.listdir(WORKSHOP)))
    if missing:
        print(f"   !! WARNING: KEEP states not found anywhere: {missing}")
    print(f"\nARCHIVE -> archive/ ({len(to_archive)}):")
    for f in to_archive:
        print(f"   {f}")
        if not dry:
            shutil.move(os.path.join(STATES, f), os.path.join(ARCHIVE, f))
    print(f"\n{'DRY RUN — nothing moved' if dry else 'SWEEP COMPLETE'}.")


if __name__ == "__main__":
    main()
