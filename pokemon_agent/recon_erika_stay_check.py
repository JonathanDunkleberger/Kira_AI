"""recon_erika_stay_check.py — Celadon Gym must not eject mid-gauntlet (no ROM)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def run():
    src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    print("recon_erika_stay_check")
    check("1a CELADON_GYM_INTERIOR", "CELADON_GYM_INTERIOR = (10, 16)" in src)
    check("1b GYM_INTERIORS map", "GYM_INTERIORS" in src and '"Erika": CELADON_GYM_INTERIOR' in src)
    check("1c ALREADY INSIDE general", "ALREADY INSIDE" in src and "_in_target_gym" in src)
    check("1d gym sanctuary", "GYM SANCTUARY" in src)
    check("1e skip Ice Beam when inside", "already inside Celadon Gym — skipping Ice Beam" in src)
    check("1f enter anyway if broke", "ENTERING ANYWAY" in src)
    check("1g Celadon tourism eject", "CELADON TOURISM" in src)
    check("1h Celadon→Erika lock", "CELADON→ERIKA LOCK" in src)
    check("1i ql_inside on beat_gym", "Deliberate gym interiority" in src)
    if fails:
        print(f"FAIL — {fails}")
        sys.exit(1)
    print("ALL PASS — Erika stay / Celadon lock wiring present.")


if __name__ == "__main__":
    run()
