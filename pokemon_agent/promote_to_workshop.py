"""promote_to_workshop.py — promote a banked_<X> checkpoint dir into a states/workshop kit fixture.

Banked dirs (banked_GRIND / banked_SEAFOAM / ...) use BARE sidecar names (world_model.json,
strat_memory.json, soul.json, journey_core.json). The recon strikes' _resolve_state loader wants
PREFIXED sidecars: <basename>.world_model.json etc. This helper copies + renames so a banked leg
output can feed the next leg's <X>_STATE=<basename>.

Usage:  python promote_to_workshop.py <banked_dir> <workshop_basename>
  e.g.  python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit
Does NOT touch canonical. Purely fills states/workshop/<basename>.{state,<sidecar>.json}.
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WORKSHOP = os.path.join(_HERE, "states", "workshop")
SIDECARS = ("world_model.json", "strat_memory.json", "soul.json", "journey_core.json")


def main():
    if len(sys.argv) < 3:
        print("usage: python promote_to_workshop.py <banked_dir> <workshop_basename>")
        sys.exit(2)
    bank_dir = sys.argv[1]
    base = sys.argv[2]
    if base.endswith(".state"):
        base = base[:-6]
    src_state = os.path.join(bank_dir, "kira_campaign.state")
    if not os.path.exists(src_state):
        print(f"!! no kira_campaign.state in {bank_dir}")
        sys.exit(1)
    os.makedirs(WORKSHOP, exist_ok=True)
    shutil.copy2(src_state, os.path.join(WORKSHOP, base + ".state"))
    print(f"  state -> workshop/{base}.state")
    for sc in SIDECARS:
        src = os.path.join(bank_dir, sc)
        if os.path.exists(src):
            dst = os.path.join(WORKSHOP, f"{base}.{sc}")
            shutil.copy2(src, dst)
            print(f"  {sc} -> workshop/{base}.{sc}")
        else:
            print(f"  (skip missing sidecar {sc})")
    print(f"PROMOTED {bank_dir} -> workshop/{base}")


if __name__ == "__main__":
    main()
