"""recon_stab_pick_check.py — STAB + accuracy move pick (2026-08-02 Surge chalk).

Proves WITHOUT an emulator that Blastoise vs Raichu prefers water STAB over Bite,
and that Hydro Pump's accuracy discount still loses to a stronger accurate STAB
when scores warrant it. Also: all four FIGHT slots are scored (no column bias).

RUN:  python -u recon_stab_pick_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pokemon_policy as pol  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def run():
    # Live chalk: Bite left-column vs Water Gun top-right — without STAB Bite wins;
    # WITH STAB Water Gun ties/wins (40*1.5=60 vs Bite 60) — prefer STAB on tie via
    # stab_mult in the tie-break. Water Pulse (60 STAB) must crush Bite.
    bite = {"name": "Bite", "type": "dark", "power": 60, "pp": 25, "accuracy": 100}
    wgun = {"name": "Water Gun", "type": "water", "power": 40, "pp": 25, "accuracy": 100}
    wpulse = {"name": "Water Pulse", "type": "water", "power": 60, "pp": 20, "accuracy": 100}
    tackle = {"name": "Tackle", "type": "normal", "power": 35, "pp": 35, "accuracy": 95}
    our = ["water"]
    foe = ["electric"]

    # Old yardstick (power * eff only) — document the bug
    old_bite = max(bite["power"], 1) * pol.effectiveness("dark", foe)
    old_gun = max(wgun["power"], 1) * pol.effectiveness("water", foe)
    check("0a OLD scoring: Bite > Water Gun (the live bug)", old_bite > old_gun, True)

    check("1a Water Gun STAB score == 60",
          pol.move_score(wgun, foe, our), 60.0)
    check("1b Bite score == 60 (no STAB)",
          pol.move_score(bite, foe, our), 60.0)
    check("1c Water Pulse STAB score == 90",
          pol.move_score(wpulse, foe, our), 90.0)

    moves_gun = [bite, tackle, wgun, {"name": "Growl", "type": "normal", "power": 0, "pp": 40}]
    idx, desc, _ = pol.choose_move(moves_gun, foe, 1.0, our_types=our)
    # Tie Bite/WaterGun at 60 — STAB tie-break should prefer Water Gun (slot 2)
    check("2a pick Water Gun over Bite on STAB tie", moves_gun[idx]["name"], "Water Gun")
    check("2b descriptor mentions STAB", "STAB" in desc, True)

    moves_pulse = [bite, tackle, wpulse, wgun]
    idx2, desc2, _ = pol.choose_move(moves_pulse, foe, 1.0, our_types=our)
    check("3a pick Water Pulse (best STAB)", moves_pulse[idx2]["name"], "Water Pulse")
    check("3b Pulse is slot 2 (not stuck on col-0)", idx2, 2)

    # Hydro Pump 120 @ 80% → 120*1.5*0.8 = 144 > Water Pulse 90
    hpump = {"name": "Hydro Pump", "type": "water", "power": 120, "pp": 5, "accuracy": 80}
    check("4a Hydro Pump expected > Water Pulse",
          pol.move_score(hpump, foe, our) > pol.move_score(wpulse, foe, our), True)
    moves_hp = [bite, wgun, wpulse, hpump]
    idx3, _, _ = pol.choose_move(moves_hp, foe, 1.0, our_types=our)
    check("4b pick Hydro Pump (slot 3 BR)", moves_hp[idx3]["name"], "Hydro Pump")
    check("4c slot 3 reachable in scoring (no column bias)", idx3, 3)

    # Ground Dig vs Electric should still beat water STAB (SE 2x)
    dig = {"name": "Dig", "type": "ground", "power": 60, "pp": 10, "accuracy": 100}
    check("5a Dig SE score 120 > Water Pulse 90",
          pol.move_score(dig, foe, our) > pol.move_score(wpulse, foe, our), True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — STAB + accuracy pick Water Gun/Pulse/Hydro over Bite vs Raichu; "
          "all 4 FIGHT slots scored.")


if __name__ == "__main__":
    run()
