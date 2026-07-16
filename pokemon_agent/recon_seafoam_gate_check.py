"""NS#12 decision-check (emulator-free) for the PROACTIVE Seafoam-crossing recognition + the seafoam
strike registry wiring — the SECOND Blaine gate, the one AFTER Surf.

Once she can Surf (FLAG_GOT_HM03) the Safari HM-prereq clears, but the billed Cinnabar sea road STILL
bounces at Route 20 (its surface is severed at the Seafoam Islands; the real crossing is the interior
boulder cascade). So _ensure_forward_questline recognizes a SECOND, post-Surf gate via _seafoam_gate:
has-Surf AND NOT-yet-crossed (FLAG_STOPPED_SEAFOAM_B3F_CURRENT unset). This replicates the decision logic
against the REAL questline KB and asserts:
  1. next_gym=Blaine + HM03-SET + seafoam-UNSET -> a Gate(missing='seafoam') is opened.
  2. next_gym=Blaine + HM03-UNSET             -> None (the Safari gate owns this window — mutual exclusion).
  3. next_gym=Blaine + seafoam-SET            -> None (already crossed; the sea road is open).
  4. next_gym=Sabrina                          -> the seafoam path does NOT fire (Blaine-scoped).
  5. The 'seafoam' capability derives success == ('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT') via the REAL
     KB -> matches the strike registry key; from_map routes to Fuchsia ('3,7').
  6. The registry lookup for ('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT') is flag-gated: None when
     SEAFOAM_STRIKE_ENABLED is off.
  7. The chain's actionable step is seafoam (surf is satisfied), NOT the surf step.
"""
import json
import os

import questline as ql

_HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror campaign's flag ids (game-knowledge layer). Kept in sync with campaign.FLAG_*_ID.
FLAG_GOT_HM03_ID = 0x239
FLAG_SEAFOAM_CROSSED_ID = 0x2D2


def proactive_seafoam_key(next_gym_leader, flags):
    """Replica of the _ensure_forward_questline SEAFOAM block + _seafoam_gate's LIVE flag checks.
    Returns the Gate.missing key to open ('seafoam'), or None."""
    if next_gym_leader != "Blaine":
        return None
    if not flags.get(FLAG_GOT_HM03_ID):         # no Surf yet -> the Safari gate owns this window
        return None
    if flags.get(FLAG_SEAFOAM_CROSSED_ID):      # already crossed -> sea road open
        return None
    return "seafoam"


def registry_hit(succ, seafoam_strike_enabled):
    """Replica of _questline_strike's registry membership + the SEAFOAM flag gate."""
    if succ != ("flag", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT"):
        return False
    return bool(seafoam_strike_enabled)


def main():
    fails = []
    HM03, SEA = FLAG_GOT_HM03_ID, FLAG_SEAFOAM_CROSSED_ID

    # 1. Blaine + Surf, not crossed -> open the seafoam gate.
    if proactive_seafoam_key("Blaine", {HM03: True}) != "seafoam":
        fails.append(f"Blaine+Surf+uncrossed should open the seafoam gate, "
                     f"got {proactive_seafoam_key('Blaine', {HM03: True})!r}")

    # 2. Blaine + no Surf -> None (Safari gate owns it; mutual exclusion on HM03).
    if proactive_seafoam_key("Blaine", {}) is not None:
        fails.append("Blaine without Surf must NOT open the seafoam gate (the Safari gate owns it)")

    # 3. Blaine + already crossed -> None.
    if proactive_seafoam_key("Blaine", {HM03: True, SEA: True}) is not None:
        fails.append("Blaine after crossing must return None (the sea road is open)")

    # 4. Sabrina -> the seafoam path does NOT fire (Blaine-scoped).
    if proactive_seafoam_key("Sabrina", {HM03: True}) is not None:
        fails.append("Sabrina must NOT trigger the seafoam path (it is Blaine-scoped)")

    # 5/7. The REAL KB derives ('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT') + the chain's actionable = seafoam.
    with open(os.path.join(_HERE, "gamedata", "frlg_gates.json"), encoding="utf-8") as f:
        kb = json.load(f)
    caps = kb.get("capabilities") or {}
    sea_cap = caps.get("seafoam")
    if not sea_cap:
        fails.append("KB has no 'seafoam' capability")
    else:
        step = ql._step_from_cap("seafoam", sea_cap)
        if step.success != ("flag", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT"):
            fails.append(f"seafoam step success should be ('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT'), "
                         f"got {step.success!r} (registry key mismatch -> strike would never fire)")
        if step.from_map != "3,7":
            fails.append(f"seafoam step from_map should route to Fuchsia '3,7', got {step.from_map!r}")
        if getattr(step, "door", None):
            fails.append(f"seafoam step must be DOOR-LESS (fires via BLOCKER-STRIKE-FIRST), got door "
                         f"{step.door!r}")
    # The prereq chain must be seafoam <- surf (so surf's satisfaction leaves seafoam actionable).
    if (sea_cap.get("obtain") or {}).get("prereq") != "surf":
        fails.append("seafoam.obtain.prereq should be 'surf' (chain = surf -> seafoam)")
    # And the flag must be resolvable in the KB (else _step_satisfied can't cross-check it).
    if "FLAG_STOPPED_SEAFOAM_B3F_CURRENT" not in (kb.get("flags") or {}):
        fails.append("FLAG_STOPPED_SEAFOAM_B3F_CURRENT missing from the flags KB (satisfied-check would fail)")

    # 6. The registry is flag-gated: fires only when SEAFOAM_STRIKE_ENABLED.
    if registry_hit(("flag", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT"), seafoam_strike_enabled=False):
        fails.append("seafoam key must NOT fire the strike when SEAFOAM_STRIKE is OFF")
    if not registry_hit(("flag", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT"), seafoam_strike_enabled=True):
        fails.append("seafoam key should fire the strike when SEAFOAM_STRIKE is ON")
    if registry_hit(("cap", "surf"), seafoam_strike_enabled=True):
        fails.append("the seafoam registry gate must not match a ('cap',*) key")

    n = 10
    if fails:
        print(f"FAIL {n - len(fails)}/{n}")
        for fl in fails:
            print("  -", fl)
        raise SystemExit(1)
    print(f"PASS {n}/{n} — Seafoam crossing: proactive recognition opens the seafoam gate (Surf-set, "
          "uncrossed), no-ops before Surf (Safari owns it) and after crossing, stays Blaine-scoped; the KB "
          "derives a door-less ('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT') step (prereq surf, routed to "
          "Fuchsia) matching the flag-gated seafoam strike registry.")


if __name__ == "__main__":
    main()
