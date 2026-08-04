"""recon_catch_discipline_test.py — weaken-then-throw discipline (2026-08-04, the badge-8
full-HP ball-burn). Pure synthetic tests of pol.chip_hit_frac / pol.chip_move_pick — the
inverted 'chip, don't kill' pick the catch flow (battle_agent.catch_pokemon/_weaken_hp)
decides with. No ROM/emulator needed (same pattern as the doubles-reader synthetic tests).

RUN: python3 pokemon_agent/recon_catch_discipline_test.py
Decision table under test:
  full-HP wild, close level      -> a SAFE chip exists       -> WEAKEN first
  low-HP wild (<=50%)            -> chip unsafe / band ready -> THROW
  overkill (L59 ace vs L20 wild) -> NO safe chip             -> EARLY THROW (never KO)
  near-level legendary (L50)     -> neutral move chips safe  -> WEAKEN first
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pokemon_policy as pol

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# Movesets as pokemon_state.read_mon builds them ({id,name,type,power,accuracy,pp}).
BLASTOISE_L59 = [
    {"id": 57, "name": "Surf", "type": "water", "power": 95, "accuracy": 100, "pp": 15},
    {"id": 58, "name": "Ice Beam", "type": "ice", "power": 95, "accuracy": 100, "pp": 10},
    {"id": 44, "name": "Bite", "type": "dark", "power": 60, "accuracy": 100, "pp": 25},
    {"id": 182, "name": "Protect", "type": "normal", "power": 0, "accuracy": 0, "pp": 10},
]
CHIPPER_L26 = [
    {"id": 33, "name": "Tackle", "type": "normal", "power": 35, "accuracy": 95, "pp": 30},
    {"id": 55, "name": "Water Gun", "type": "water", "power": 40, "accuracy": 100, "pp": 25},
]


def main():
    print("== 1. full-HP wild, close level -> WEAKEN (safe chip exists) ==")
    # L26 chipper vs L22 wild Pidgey at full HP: Tackle should be a safe chip.
    i, est, safe = pol.chip_move_pick(CHIPPER_L26, ["normal", "flying"], 26, 22,
                                      foe_hp_frac=1.0, our_types=["water", "water"])
    check("safe chip found", safe is True, f"move={CHIPPER_L26[i]['name']} est={est:.0%}")
    check("chip won't one-shot", est < 0.70, f"est={est:.0%}")

    print("== 2. low-HP wild -> THROW (no more chipping) ==")
    # Same matchup but the foe is already at 25% HP: the gentlest chip now risks the KO
    # (battle_agent also throws earlier via CATCH_READY_FRAC=0.50 — this is the policy belt).
    i2, est2, safe2 = pol.chip_move_pick(CHIPPER_L26, ["normal", "flying"], 26, 22,
                                         foe_hp_frac=0.25, our_types=["water", "water"])
    check("chip refused at low HP", safe2 is False, f"est={est2:.0%} vs 25% HP left")

    print("== 3. overkill risk (L59 Blastoise vs L20 wild) -> EARLY THROW ==")
    # Every usable move likely one-shots — the sanctioned early throw, never the KO.
    i3, est3, safe3 = pol.chip_move_pick(BLASTOISE_L59, ["normal", "flying"], 59, 20,
                                         foe_hp_frac=1.0, our_types=["water", "water"])
    check("no safe chip at a 39-level gap", safe3 is False,
          f"gentlest={BLASTOISE_L59[i3]['name']} est={est3:.0%}")
    check("gentlest is still the lowest estimate", BLASTOISE_L59[i3]["name"] == "Bite",
          f"picked {BLASTOISE_L59[i3]['name']}")

    print("== 4. near-level legendary (L59 vs L50 Moltres) -> WEAKEN first ==")
    # The legendary careful-capture flow shares this pick: neutral Bite chips safely,
    # Surf/Ice Beam (super-effective vs fire/flying) would not.
    i4, est4, safe4 = pol.chip_move_pick(BLASTOISE_L59, ["fire", "flying"], 59, 50,
                                         foe_hp_frac=1.0, our_types=["water", "water"])
    check("legendary gets a safe chip", safe4 is True,
          f"move={BLASTOISE_L59[i4]['name']} est={est4:.0%}")
    check("chip pick is estimate-ordered, not raw power",
          BLASTOISE_L59[i4]["name"] == "Bite",
          "Bite(60 neutral) under Surf/Ice Beam(95 super-effective)")

    print("== 5. edges: status-only / immune movesets never chip ==")
    status_only = [{"id": 182, "name": "Protect", "type": "normal", "power": 0, "pp": 10}]
    check("status-only -> no chip", pol.chip_move_pick(status_only, ["normal"], 30, 20)[0] is None)
    ghost_immune = [{"id": 33, "name": "Tackle", "type": "normal", "power": 35, "pp": 30}]
    check("immune (normal vs ghost) -> no chip",
          pol.chip_move_pick(ghost_immune, ["ghost", "poison"], 30, 28)[0] is None)
    depleted = [{"id": 55, "name": "Water Gun", "type": "water", "power": 40, "pp": 0}]
    check("0-PP -> no chip", pol.chip_move_pick(depleted, ["normal"], 30, 28)[0] is None)

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
