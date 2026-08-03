"""recon_ace_carries_check.py — headless verifier for post-Surge Celadon march (2026-08-02).

Proves WITHOUT an emulator that:
  (1) Blastoise L36 + paper bench vs Erika → ace carries (MARCH; no Route-6 park)
  (2) Blastoise L36 + paper bench vs Lt. Surge → does NOT ace-carry (rematch chalk intact)
  (3) Wartortle L33 + paper vs Erika → mid-evo, no carry
  (4) campaign wiring: ACE-CARRIES stand-down + GO HARD + goals

RUN:  python -u recon_ace_carries_check.py
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import mgba.core          # noqa: F401
except ImportError:
    _root = types.ModuleType("mgba")
    for _m in ("core", "image", "vfs", "log", "png"):
        _sub = types.ModuleType(f"mgba.{_m}")
        setattr(_root, _m, _sub)
        sys.modules[f"mgba.{_m}"] = _sub
    _root.log.silence = lambda: None
    sys.modules["mgba"] = _root

import pokemon_planner as P   # noqa: E402
import campaign as C          # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def _party_blastoise_paper():
    return [
        {"species": "blastoise", "level": 36, "species_id": 9},
        {"species": "ekans", "level": 13, "species_id": 23},
        {"species": "spearow", "level": 17, "species_id": 21},
        {"species": "abra", "level": 14, "species_id": 63},
        {"species": "rattata", "level": 14, "species_id": 19},
        {"species": "butterfree", "level": 13, "species_id": 12},
    ]


def _party_wartortle_paper():
    return [
        {"species": "wartortle", "level": 33, "species_id": 8},
        {"species": "ekans", "level": 13, "species_id": 23},
        {"species": "spearow", "level": 17, "species_id": 21},
        {"species": "abra", "level": 14, "species_id": 63},
        {"species": "rattata", "level": 14, "species_id": 19},
        {"species": "butterfree", "level": 13, "species_id": 12},
    ]


class _FakeCampaign:
    """Minimal stand-in: only the methods under test + planner."""

    def __init__(self):
        self.planner = P.StrategicPlanner(log=lambda *_a, **_k: None)
        self._gym_prep_bump = {}

    _ace_carries_next_gym = C.Campaign._ace_carries_next_gym
    _gym_go_hard_blocked = C.Campaign._gym_go_hard_blocked


def run():
    planner = P.StrategicPlanner(log=lambda *_a, **_k: None)
    party = _party_blastoise_paper()

    r_e = planner.gym_readiness("Erika", party, party_target=3, loss_bump=0)
    check("1a Erika: paper_bench", r_e.get("paper_bench"), True)
    check("1b Erika: not ready (paper)", r_e.get("ready"), False)
    check("1c Erika: top >= level_target", r_e.get("top_level", 0) >= r_e.get("level_target", 99), True)

    r_s = planner.gym_readiness("Lt. Surge", party, party_target=3, loss_bump=0)
    check("2a Surge: paper_bench", r_s.get("paper_bench"), True)
    check("2b Surge: not ready", r_s.get("ready"), False)
    check("2c Surge: top >= level_target (raw bar)",
          r_s.get("top_level", 0) >= r_s.get("level_target", 99), True)

    camp = _FakeCampaign()
    st_e = {"next_gym": {"leader": "Erika", "city": "Celadon City"}, "party": party,
            "badge_count": 3}
    st_s = {"next_gym": {"leader": "Lt. Surge", "city": "Vermilion City"}, "party": party,
            "badge_count": 2}
    st_w = {"next_gym": {"leader": "Erika", "city": "Celadon City"},
            "party": _party_wartortle_paper(), "badge_count": 3}

    check("3a ace carries Erika (paper OK — poison/sleep cures don't block march)",
          camp._ace_carries_next_gym(st_e), True)
    check("3b does NOT ace-carry Surge (bring_cures paralysis + paper)",
          camp._ace_carries_next_gym(st_s), False)
    check("3c Wartortle mid-evo: no carry vs Erika",
          camp._ace_carries_next_gym(st_w), False)

    check("4a GO-HARD NOT blocked for Erika (ace carries)",
          camp._gym_go_hard_blocked(st_e), False)
    check("4b GO-HARD blocked for Surge paper (rematch chalk)",
          camp._gym_go_hard_blocked(st_s), True)

    src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("5a ACE-CARRIES stand-down framing", "ACE-CARRIES: prep-to-L" in src, True)
    check("5b ACE-CARRIES GO HARD tag", "ACE-CARRIES" in src and "_ace3" in src, True)
    check("5c goals include _ace_g", "_ace_g" in src, True)
    check("5d lopsided skips when ace carries",
          "Ace already clears the next gym's level bar" in src, True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — Celadon: ace carries → march; Surge paper rematch chalk intact.")


if __name__ == "__main__":
    run()
