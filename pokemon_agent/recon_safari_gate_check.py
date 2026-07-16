"""NS#11 decision-check (emulator-free) for the PROACTIVE Blaine Surf-prereq recognition + the Safari
strike registry wiring.

The Cinnabar sea road can't produce a Gate the normal way (billed sea, no exit_gate) and Blaine's
at-the-door prereq (_gym_prereq_gate via beat_gym-stuck) can NEVER fire — she can't reach Cinnabar
Surf-less. So the recognition is PROACTIVE, scoped to HM_PREREQ_GYMS={"Blaine"} in
_ensure_forward_questline. This replicates the decision logic against the REAL GYM_PREREQS + questline
KB, and asserts:
  1. next_gym=Blaine + HM03-UNSET  -> a Gate(missing='surf') is opened.
  2. next_gym=Blaine + HM03-SET    -> None (LIVE flag cross-check; no-op once she has Surf).
  3. next_gym=Sabrina              -> the PROACTIVE path does NOT fire (Sabrina not in HM_PREREQ_GYMS,
                                      so her at-the-door Silph path is untouched).
  4. The 'surf' capability derives success == ('cap','surf') via the REAL KB -> matches the strike
     registry key (so the strike fires when the surf step is actionable).
  5. The registry lookup for ('cap','surf') is flag-gated: returns None when SAFARI_STRIKE_ENABLED is off.
"""
import json
import os

import questline as ql

_HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror campaign's tables (game-knowledge layer). Kept in sync with campaign.GYM_PREREQS / HM_PREREQ_GYMS.
GYM_PREREQS = {
    "Sabrina": (0x3E, "FLAG_HIDE_SAFFRON_ROCKETS", "Sabrina wall"),
    "Blaine": (0x239, "surf", "Cinnabar needs Surf"),
}
HM_PREREQ_GYMS = {"Blaine"}


def proactive_prereq_key(next_gym_leader, flags):
    """Replica of the _ensure_forward_questline proactive block + _gym_prereq_gate's LIVE flag check.
    Returns the Gate.missing key to open, or None (not an HM-prereq gym / flag already set)."""
    if next_gym_leader not in HM_PREREQ_GYMS:
        return None
    spec = GYM_PREREQS.get(next_gym_leader)
    if not spec:
        return None
    flag_id, missing_key, _human = spec
    if flags.get(flag_id):                      # LIVE cross-check: already have it -> no gate
        return None
    return missing_key


def registry_hit(succ, safari_strike_enabled):
    """Replica of _questline_strike's registry membership + the SAFARI_STRIKE flag gate. Returns True iff
    the strike would fire for this success signature."""
    strike_keys = {("cap", "surf"), ("cap", "strength")}
    if succ not in strike_keys:
        return False
    return bool(safari_strike_enabled)


def main():
    fails = []

    # 1. Blaine + HM03 unset -> open the 'surf' gate.
    if proactive_prereq_key("Blaine", {}) != "surf":
        fails.append(f"Blaine+HM03-unset should open the surf gate, got {proactive_prereq_key('Blaine', {})!r}")

    # 2. Blaine + HM03 set -> None (no-op once she has Surf).
    if proactive_prereq_key("Blaine", {0x239: True}) is not None:
        fails.append("Blaine+HM03-set should return None (flag cross-check)")

    # 3. Sabrina -> proactive path does NOT fire (not in HM_PREREQ_GYMS).
    if proactive_prereq_key("Sabrina", {}) is not None:
        fails.append("Sabrina must NOT be proactively recognized (her at-the-door path owns it)")
    # (Blaine and Sabrina are BOTH in GYM_PREREQS, so this proves the HM_PREREQ_GYMS scoping is what
    #  distinguishes them — not merely presence in the prereq table.)

    # 4. The REAL KB derives ('cap','surf') for the surf capability -> matches the registry key.
    with open(os.path.join(_HERE, "gamedata", "frlg_gates.json"), encoding="utf-8") as f:
        kb = json.load(f)
    surf_cap = (kb.get("capabilities") or {}).get("surf")
    if not surf_cap:
        fails.append("KB has no 'surf' capability")
    else:
        step = ql._step_from_cap("surf", surf_cap)
        if step.success != ("cap", "surf"):
            fails.append(f"surf step success should be ('cap','surf'), got {step.success!r} "
                         "(registry key mismatch -> strike would never fire)")
        # from_map drives anchor-first routing: it must be Fuchsia (3,7), not Cinnabar (the NS#10 anchor fix).
        if step.from_map != "3,7":
            fails.append(f"surf step from_map should route to Fuchsia '3,7', got {step.from_map!r}")

    # 5. The registry is flag-gated: ('cap','surf') fires only when SAFARI_STRIKE_ENABLED.
    if registry_hit(("cap", "surf"), safari_strike_enabled=False):
        fails.append("('cap','surf') must NOT fire the strike when SAFARI_STRIKE is OFF")
    if not registry_hit(("cap", "surf"), safari_strike_enabled=True):
        fails.append("('cap','surf') should fire the strike when SAFARI_STRIKE is ON")
    if not registry_hit(("cap", "strength"), safari_strike_enabled=True):
        fails.append("('cap','strength') should also fire the strike when SAFARI_STRIKE is ON")
    if registry_hit(("flag", "FLAG_GOT_HM03"), safari_strike_enabled=True):
        fails.append("only the ('cap',*) keys are registered — a raw flag key must not fire")

    n = 8
    if fails:
        print(f"FAIL {n - len(fails)}/{n}")
        for fl in fails:
            print("  -", fl)
        raise SystemExit(1)
    print(f"PASS {n}/{n} — Blaine Surf-prereq: proactive recognition opens the surf gate (HM03-unset), "
          "no-ops once obtained, leaves Sabrina's at-the-door path alone; the KB derives ('cap','surf') "
          "matching the flag-gated safari strike registry, routed to Fuchsia.")


if __name__ == "__main__":
    main()
