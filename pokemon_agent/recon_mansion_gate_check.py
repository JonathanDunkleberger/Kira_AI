"""NS#13 decision-check (emulator-free) for the PROACTIVE Secret-Key recognition + the mansion strike
registry wiring — the THIRD Blaine gate, the one AFTER the Seafoam crossing.

Once she's crossed the sea (FLAG_SEAFOAM_CROSSED) she's ON Cinnabar, but Blaine's gym DOOR is LOCKED behind
the Pokémon Mansion's Secret Key. Without recognition she reads the locked door as a puzzle and STALLS
(the post-Cinnabar bounce). So _ensure_forward_questline recognizes a THIRD gate via _mansion_gate:
has-crossed AND NOT-yet-got-key (FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY unset). This replicates the
decision logic against the REAL questline KB and asserts:
  1. next_gym=Blaine + crossed + no-key   -> a Gate(missing='secret_key') is opened.
  2. next_gym=Blaine + NOT crossed         -> None (the seafoam gate owns this window — sequential).
  3. next_gym=Blaine + key already taken   -> None (Blaine's door is open).
  4. next_gym=Sabrina                       -> the mansion path does NOT fire (Blaine-scoped).
  5. The 'secret_key' capability derives success == ('flag','FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY') via
     the REAL KB -> matches the strike registry key; from_map routes to Cinnabar ('3,8'); DOOR-LESS.
  6. The registry lookup is flag-gated: None when MANSION_STRIKE_ENABLED is off.
  7. The prereq chain resolves secret_key -> seafoam -> surf (event->event->cap) without choking, so the
     crossing's satisfaction leaves secret_key actionable.
"""
import json
import os

import questline as ql

_HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror campaign's flag ids (game-knowledge layer). Kept in sync with campaign.FLAG_*_ID.
FLAG_SEAFOAM_CROSSED_ID = 0x2D2
FLAG_SECRET_KEY_ID = 0x1A8


def proactive_secret_key(next_gym_leader, flags):
    """Replica of the _ensure_forward_questline SECRET-KEY block + _mansion_gate's LIVE flag checks.
    Returns the Gate.missing key to open ('secret_key'), or None."""
    if next_gym_leader != "Blaine":
        return None
    if not flags.get(FLAG_SEAFOAM_CROSSED_ID):      # not across yet -> the seafoam gate owns this window
        return None
    if flags.get(FLAG_SECRET_KEY_ID):               # already have the key -> Blaine's door is open
        return None
    return "secret_key"


def registry_hit(succ, mansion_strike_enabled):
    """Replica of _questline_strike's registry membership + the MANSION flag gate."""
    if succ != ("flag", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY"):
        return False
    return bool(mansion_strike_enabled)


def main():
    fails = []
    SEA, KEY = FLAG_SEAFOAM_CROSSED_ID, FLAG_SECRET_KEY_ID

    # 1. Blaine + crossed, no key -> open the secret_key gate.
    if proactive_secret_key("Blaine", {SEA: True}) != "secret_key":
        fails.append(f"Blaine+crossed+no-key should open the secret_key gate, "
                     f"got {proactive_secret_key('Blaine', {SEA: True})!r}")

    # 2. Blaine + not crossed -> None (the seafoam gate owns it; sequential).
    if proactive_secret_key("Blaine", {}) is not None:
        fails.append("Blaine before crossing must NOT open the secret_key gate (the seafoam gate owns it)")

    # 3. Blaine + key already taken -> None.
    if proactive_secret_key("Blaine", {SEA: True, KEY: True}) is not None:
        fails.append("Blaine after taking the key must return None (Blaine's door is open)")

    # 4. Sabrina -> the mansion path does NOT fire (Blaine-scoped).
    if proactive_secret_key("Sabrina", {SEA: True}) is not None:
        fails.append("Sabrina must NOT trigger the secret_key path (it is Blaine-scoped)")

    # 5. The REAL KB derives ('flag','FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY'), routes to Cinnabar, door-less.
    with open(os.path.join(_HERE, "gamedata", "frlg_gates.json"), encoding="utf-8") as f:
        kb = json.load(f)
    caps = kb.get("capabilities") or {}
    sk_cap = caps.get("secret_key")
    if not sk_cap:
        fails.append("KB has no 'secret_key' capability")
    else:
        step = ql._step_from_cap("secret_key", sk_cap)
        if step.success != ("flag", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY"):
            fails.append(f"secret_key step success should be ('flag','FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY'), "
                         f"got {step.success!r} (registry key mismatch -> strike would never fire)")
        if step.from_map != "3,8":
            fails.append(f"secret_key step from_map should route to Cinnabar '3,8', got {step.from_map!r}")
        if getattr(step, "door", None):
            fails.append(f"secret_key step must be DOOR-LESS (fires via BLOCKER-STRIKE-FIRST), got door "
                         f"{step.door!r}")
        if (sk_cap.get("obtain") or {}).get("prereq") != "seafoam":
            fails.append("secret_key.obtain.prereq should be 'seafoam' (chain = seafoam -> secret_key)")
    # The flag must be resolvable in the KB (else _step_satisfied can't cross-check it).
    if "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY" not in (kb.get("flags") or {}):
        fails.append("FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY missing from the flags KB (satisfied-check fails)")

    # 7. Walk the prereq chain the way derive_questline does (obtain.prereq, seen-guarded) — it must resolve
    #    secret_key -> seafoam -> surf (event -> event -> cap) and terminate, so the crossing's satisfaction
    #    leaves secret_key as the actionable (last) step.
    chain_keys, key, seen = [], "secret_key", set()
    while key and key not in seen:
        seen.add(key)
        cap = caps.get(key)
        chain_keys.append(key)
        if not cap:
            break
        key = (cap.get("obtain") or {}).get("prereq")
    if chain_keys[:2] != ["secret_key", "seafoam"] or "surf" not in chain_keys:
        fails.append(f"prereq chain should walk secret_key -> seafoam -> surf, got {chain_keys}")

    # 6. The registry is flag-gated: fires only when MANSION_STRIKE_ENABLED.
    if registry_hit(("flag", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY"), mansion_strike_enabled=False):
        fails.append("secret_key must NOT fire the strike when MANSION_STRIKE is OFF")
    if not registry_hit(("flag", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY"), mansion_strike_enabled=True):
        fails.append("secret_key should fire the strike when MANSION_STRIKE is ON")
    if registry_hit(("flag", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT"), mansion_strike_enabled=True):
        fails.append("the mansion registry gate must not match the seafoam flag key")

    n = 10
    if fails:
        print(f"FAIL {n - len(fails)}/{n}")
        for fl in fails:
            print("  -", fl)
        raise SystemExit(1)
    print(f"PASS {n}/{n} — Secret Key: proactive recognition opens the secret_key gate (crossed, no key), "
          "no-ops before the crossing (seafoam owns it) and after the key, stays Blaine-scoped; the KB "
          "derives a door-less ('flag','FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY') step (prereq seafoam, "
          "routed to Cinnabar) matching the flag-gated mansion strike registry; chain resolves "
          "secret_key -> seafoam -> surf.")


if __name__ == "__main__":
    main()
