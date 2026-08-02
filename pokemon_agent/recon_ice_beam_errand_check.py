"""recon_ice_beam_errand_check.py — headless wiring for Celadon Ice Beam errand (2026-08-02).

Proves WITHOUT an emulator that:
  (1) game_corner.py fact table matches pret (doors, map ids, costs, item ids)
  (2) campaign.beat_gym hooks IceBeamErrand before Erika prep
  (3) KB where-text says Game Corner (not Dept Store)
  (4) tidal teach_plan has early Blastoise TM13 row
  (5) coverage-teach already lists TM13 in _COVERAGE_MOVES

Live actuation (Coin Case → coins → prize → teach) needs the PC ROM + save —
run game_corner.IceBeamErrand on a Celadon save when validating end-to-end.

RUN:  python -u recon_ice_beam_errand_check.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

fails = []


def check(name, got, want=True):
    ok = got == want if not isinstance(want, bool) else bool(got) == want
    if isinstance(want, bool) and want is True:
        ok = bool(got)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}")
    if not ok:
        fails.append(name)


def run():
    import game_corner as gc

    check("1a CELADON (3,6)", gc.CELADON == (3, 6))
    check("1b GC door (34,21)", gc.GC_DOOR == (34, 21))
    check("1c Prize door (39,20)", gc.PRIZE_DOOR == (39, 20))
    check("1d Restaurant door (37,29)", gc.RESTAURANT_DOOR == (37, 29))
    check("1e GC map (10,14)", gc.GC == (10, 14))
    check("1f Prize map (10,15)", gc.PRIZE == (10, 15))
    check("1g Restaurant map (10,17)", gc.RESTAURANT == (10, 17))
    check("1h TM13 item 301", gc.ITEM_TM13 == 301)
    check("1i Coin Case item 260", gc.ITEM_COIN_CASE == 260)
    check("1j Ice Beam move 58", gc.MOVE_ICE_BEAM == 58)
    check("1k 4000 coin cost", gc.TM13_COIN_COST == 4000)
    check("1l 500-coin pack ¥10000", gc.COINS_PACK_COST == 10000)
    check("1m errand enabled by default", gc.ICE_BEAM_ERRAND_ENABLED)

    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("2a beat_gym imports IceBeamErrand", "from game_corner import IceBeamErrand" in camp)
    check("2b gated on Erika", 'if name == "Erika"' in camp and "ice-beam errand" in camp)
    # Errand runs BEFORE prep_for_gym
    i_err = camp.find("ice-beam errand")
    i_prep = camp.find("self.prep_for_gym(gym)")
    check("2c errand before prep_for_gym", i_err > 0 and i_prep > i_err)

    kb = json.load(open(os.path.join(_HERE, "gamedata", "frlg_strategy.json"), encoding="utf-8"))
    ib = (kb.get("key_moves") or {}).get("ice beam") or {}
    check("3a KB where mentions Game Corner", "Game Corner" in (ib.get("where") or ""))
    check("3b KB where NOT Dept Store lie", "Dept" not in (ib.get("where") or ""))
    check("3c KB for includes Erika", "Erika" in (ib.get("for") or []))

    plan = json.load(open(os.path.join(_HERE, "gamedata", "frlg_team_plan.json"), encoding="utf-8"))
    found = False
    for arch in plan.get("archetypes") or []:
        if arch.get("starter_branch") != "squirtle":
            continue
        for row in arch.get("teach_plan") or []:
            if (row.get("tm") == "TM13" and "blastoise" in (row.get("to") or "").lower()
                    and int(row.get("when_badge") or 99) <= 3):
                found = True
                break
    check("4a teach_plan early Blastoise TM13", found)

    check("5a _COVERAGE_MOVES has TM13", "301: (None, 13, 58, \"ice\", 95)" in camp
          or "301: (None, 13, 58, 'ice', 95)" in camp)
    # looser: item 301 ice beam line
    check("5b coverage ice beam comment", "TM13 Ice Beam" in camp)

    src = open(os.path.join(_HERE, "game_corner.py"), encoding="utf-8").read()
    check("6a IceBeamErrand.run defined", "def run(self)" in src)
    check("6b never hideout stairs", "15, 2" not in src or "hideout" in src.lower())
    check("6c verify coin+case deltas", "after_q > before_q" in src and "COINS_PACK_COST" in src)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — Ice Beam errand wired for Erika; live buy/teach needs PC ROM.")


if __name__ == "__main__":
    run()
