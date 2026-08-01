"""recon_surge_prep_check.py — headless verifier for the 2026-08-01 Surge rematch prep fix.

Proves WITHOUT an emulator that:
  (1) Wartortle L33 + paper bench vs Lt. Surge is NOT dominant / NOT ready
      (the live chalk: "GYM-PREP [Lt. Surge]: DOMINANT — top L33 >= L25+8")
  (2) Blastoise L36+ with a non-paper bench CAN be ready/dominant per remaining rules
  (3) gym_prep_bump JSON shape persists (mock file, same contract as campaign.py)

RUN:  python -u recon_surge_prep_check.py
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pokemon_planner as P          # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def _party_paper():
    """Live soak shape: Wartortle L33 + Raichu-farmable bench."""
    return [
        {"species": "wartortle", "level": 33, "species_id": 8},
        {"species": "butterfree", "level": 11, "species_id": 12},
        {"species": "rattata", "level": 14, "species_id": 19},
        {"species": "spearow", "level": 15, "species_id": 21},
        {"species": "abra", "level": 14, "species_id": 63},
        {"species": "ekans", "level": 13, "species_id": 23},
    ]


def _party_blastoise_ready():
    """Final form + bench that clears the paper-bench gate (max rest >= Surge ace 24)."""
    return [
        {"species": "blastoise", "level": 38, "species_id": 9},
        {"species": "diglett", "level": 26, "species_id": 50},   # ground answer
        {"species": "spearow", "level": 25, "species_id": 21},
        {"species": "ekans", "level": 24, "species_id": 23},
        {"species": "abra", "level": 24, "species_id": 63},
        {"species": "rattata", "level": 24, "species_id": 19},
    ]


def _save_bump(path, bumps):
    """Mirror Campaign._save_gym_prep_bump contract (bumps dict on disk)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"bumps": {k: int(v) for k, v in bumps.items() if int(v) > 0}}, f)
    os.replace(tmp, path)


def _load_bump(path):
    """Mirror Campaign._load_gym_prep_bump contract."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f) or {}
    bumps = data.get("bumps") if isinstance(data, dict) else None
    if not isinstance(bumps, dict):
        bumps = data if isinstance(data, dict) else {}
    return {str(k): int(v) for k, v in bumps.items()
            if isinstance(v, (int, float)) and int(v) > 0}


def run():
    planner = P.StrategicPlanner(log=lambda *_a, **_k: None)

    # --- (1) live chalk: Wartortle L33 + paper bench vs Surge ---
    r = planner.gym_readiness("Lt. Surge", _party_paper(), party_target=3, loss_bump=0)
    check("1a wartortle L33 paper: not dominant", r.get("dominant"), False)
    check("1b wartortle L33 paper: not ready", r.get("ready"), False)
    check("1c mid_evo_block", r.get("mid_evo_block"), True)
    check("1d paper_bench", r.get("paper_bench"), True)
    check("1e level_target >= Blastoise 36", r.get("level_target", 0) >= 36, True)

    # With a persisted-style loss bump (Blastoise bar for Surge ace 24 + margin 1 → need 11)
    r2 = planner.gym_readiness("Lt. Surge", _party_paper(), party_target=3, loss_bump=11)
    check("1f with bump=11 still not dominant", r2.get("dominant"), False)
    check("1g with bump=11 still not ready", r2.get("ready"), False)
    check("1h level_target with bump >= 36", r2.get("level_target", 0) >= 36, True)

    # --- (2) Blastoise + real bench can pass ---
    r3 = planner.gym_readiness("Lt. Surge", _party_blastoise_ready(), party_target=3, loss_bump=0)
    check("2a blastoise+bench: not mid_evo", r3.get("mid_evo_block"), False)
    check("2b blastoise+bench: not paper", r3.get("paper_bench"), False)
    check("2c blastoise+bench: ready OR dominant",
          bool(r3.get("ready") or r3.get("dominant")), True)
    check("2d blastoise+bench: has_type_answer", r3.get("has_type_answer"), True)
    check("2e blastoise+bench: ready", r3.get("ready"), True)

    # --- (3) gym_prep_bump disk persistence (mock file) ---
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gym_prep_bump.json")
        _save_bump(path, {"Lt. Surge": 11})
        check("3a bump file exists", os.path.exists(path), True)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        check("3b file has Lt. Surge=11", data.get("bumps", {}).get("Lt. Surge"), 11)
        loaded = _load_bump(path)
        check("3c reload restores bump", loaded.get("Lt. Surge"), 11)
        _save_bump(path, {})  # badge-win clear
        loaded2 = _load_bump(path)
        check("3d clear persists (no Surge bump)", "Lt. Surge" in loaded2, False)

    # --- (4) source wiring in campaign.py (no mgba import) ---
    src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("4a _gym_go_hard_blocked present", "def _gym_go_hard_blocked" in src, True)
    check("4b GYM_PREP_BUMP_JSON present", "GYM_PREP_BUMP_JSON" in src, True)
    check("4c _save_gym_prep_bump on bump path", "self._save_gym_prep_bump()" in src, True)
    check("4d Blastoise goal framing", "Evolve to Blastoise" in src, True)
    check("4e load at Campaign init", "self._gym_prep_bump = self._load_gym_prep_bump()" in src, True)
    check("4f GO-HARD block zeros dominant/momentum",
          "_dom3 = False" in src and "_mom3 = False" in src and "_gym_go_hard_blocked" in src, True)
    check("4g planner mid-evo constant", "_MID_STAGE_FINAL_EVO" in open(
        os.path.join(_HERE, "pokemon_planner.py"), encoding="utf-8").read(), True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — Surge mid-evo/paper-bench not dominant; Blastoise+bench can be ready; "
          "gym_prep_bump persists across mock relaunch.")


if __name__ == "__main__":
    run()
