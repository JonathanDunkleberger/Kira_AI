"""recon_surge_prep_check.py — headless verifier for Surge rematch prep (2026-08-01/02).

Proves WITHOUT an emulator that:
  (1) Wartortle L33 + paper bench vs Lt. Surge is NOT dominant / NOT ready
  (2) Blastoise L36 + paper bench (live soak) is NOT ready — MOMENTUM must not march
  (3) Blastoise L38 + one ~L20 bench (no Diglett) IS ready via overlevel_carry
  (4) Blastoise L38 + Diglett bench can be ready via type answer
  (5) gym_prep_bump JSON shape persists
  (6) campaign.py always consults readiness in _gym_go_hard_blocked

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


def _party_blastoise_paper():
    """2026-08-01 soak: Blastoise L36 + paper bench (Spearow L17 max)."""
    return [
        {"species": "blastoise", "level": 36, "species_id": 9},
        {"species": "ekans", "level": 13, "species_id": 23},
        {"species": "spearow", "level": 17, "species_id": 21},
        {"species": "abra", "level": 14, "species_id": 63},
        {"species": "rattata", "level": 14, "species_id": 19},
        {"species": "butterfree", "level": 13, "species_id": 12},
    ]


def _party_blastoise_rematch():
    """Jonny rematch bar: Blastoise L38 + one fieldable finisher (~L20), no Diglett."""
    return [
        {"species": "blastoise", "level": 38, "species_id": 9},
        {"species": "spearow", "level": 20, "species_id": 21},
        {"species": "ekans", "level": 16, "species_id": 23},
        {"species": "abra", "level": 15, "species_id": 63},
        {"species": "rattata", "level": 15, "species_id": 19},
        {"species": "butterfree", "level": 14, "species_id": 12},
    ]


def _party_blastoise_ready():
    """Final form + Diglett answer + bench that clears the paper-bench gate."""
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

    r2 = planner.gym_readiness("Lt. Surge", _party_paper(), party_target=3, loss_bump=11)
    check("1f with bump=11 still not dominant", r2.get("dominant"), False)
    check("1g with bump=11 still not ready", r2.get("ready"), False)
    check("1h level_target with bump >= 36", r2.get("level_target", 0) >= 36, True)

    # --- (2) Blastoise L36 + paper (the Misty-momentum GO HARD chalk) ---
    r_bp = planner.gym_readiness("Lt. Surge", _party_blastoise_paper(),
                                 party_target=3, loss_bump=0)
    check("2a blastoise L36 paper: paper_bench", r_bp.get("paper_bench"), True)
    check("2b blastoise L36 paper: not ready", r_bp.get("ready"), False)
    check("2c blastoise L36 paper: not overlevel_carry", r_bp.get("overlevel_carry"), False)
    check("2d fieldable_floor ~20", r_bp.get("fieldable_floor"), 20)

    # --- (3) Blastoise L38 + L20 bench, no Diglett → overlevel carry ---
    r_ol = planner.gym_readiness("Lt. Surge", _party_blastoise_rematch(),
                                 party_target=3, loss_bump=0)
    check("3a rematch: not paper", r_ol.get("paper_bench"), False)
    check("3b rematch: overlevel_carry", r_ol.get("overlevel_carry"), True)
    check("3c rematch: ready", r_ol.get("ready"), True)
    check("3d rematch: no type answer (still ok)", r_ol.get("has_type_answer"), False)

    # With loss bump forcing L38 grind bar (ace 24 + margin 1 + bump 13 = 38)
    r_ol2 = planner.gym_readiness("Lt. Surge", _party_blastoise_rematch(),
                                  party_target=3, loss_bump=13)
    check("3e rematch+bump13: level_target >= 38", r_ol2.get("level_target", 0) >= 38, True)
    check("3f rematch+bump13: still ready at L38", r_ol2.get("ready"), True)

    # --- (4) Blastoise + Diglett team can pass via type answer ---
    r3 = planner.gym_readiness("Lt. Surge", _party_blastoise_ready(), party_target=3, loss_bump=0)
    check("4a diglett team: not mid_evo", r3.get("mid_evo_block"), False)
    check("4b diglett team: not paper", r3.get("paper_bench"), False)
    check("4c diglett team: has_type_answer", r3.get("has_type_answer"), True)
    check("4d diglett team: ready", r3.get("ready"), True)

    # --- (5) gym_prep_bump disk persistence (mock file) ---
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gym_prep_bump.json")
        _save_bump(path, {"Lt. Surge": 13})
        check("5a bump file exists", os.path.exists(path), True)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        check("5b file has Lt. Surge=13", data.get("bumps", {}).get("Lt. Surge"), 13)
        loaded = _load_bump(path)
        check("5c reload restores bump", loaded.get("Lt. Surge"), 13)
        _save_bump(path, {})
        loaded2 = _load_bump(path)
        check("5d clear persists (no Surge bump)", "Lt. Surge" in loaded2, False)

    # --- (6) source wiring in campaign.py ---
    src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("6a _gym_go_hard_blocked present", "def _gym_go_hard_blocked" in src, True)
    check("6b always consults readiness (not only bump)",
          "if r and not r.get(\"ready\")" in src
          and "if bump and getattr" not in src.split("def _gym_go_hard_blocked")[1].split("def ")[0],
          True)
    check("6c BLASTOISE_REMATCH_LEVEL present", "BLASTOISE_REMATCH_LEVEL" in src, True)
    check("6d Blastoise rematch goal framing", "Grind Blastoise toward L" in src, True)
    check("6e load at Campaign init", "self._gym_prep_bump = self._load_gym_prep_bump()" in src, True)
    check("6f GO-HARD block zeros dominant/momentum",
          "_dom3 = False" in src and "_mom3 = False" in src and "_gym_go_hard_blocked" in src, True)
    check("6g planner overlevel carry", "OVERLEVEL_CARRY_MARGIN" in open(
        os.path.join(_HERE, "pokemon_planner.py"), encoding="utf-8").read(), True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — Surge rematch: paper Blastoise blocked; L38+bench L20 ready; "
          "GO-HARD always consults readiness.")


if __name__ == "__main__":
    run()
