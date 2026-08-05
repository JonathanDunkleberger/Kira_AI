"""recon_catch_discipline_test.py — weaken-then-throw discipline (2026-08-04, the badge-8
full-HP ball-burn) + THE LEGENDARY BALL RESERVE / LAP NO-CATCH (2026-08-05, the Kindle Road
Meowth Ultra burn). Pure synthetic tests — pol.chip_move_pick for the weaken pick, and the
battle_agent ball-selection/divert gates driven through a stub bridge (no ROM/emulator).

RUN: python3 pokemon_agent/recon_catch_discipline_test.py
Decision table under test:
  full-HP wild, close level      -> a SAFE chip exists       -> WEAKEN first
  low-HP wild (<=50%)            -> chip unsafe / band ready -> THROW
  overkill (L59 ace vs L20 wild) -> NO safe chip             -> EARLY THROW (never KO)
  near-level legendary (L50)     -> neutral move chips safe  -> WEAKEN first
  hunt pending + only Ultras + trash target ('cheap')        -> RESERVE refusal (no throw)
  hunt target ('best')                                       -> spends Ultras freely
  mixed bag + hunt pending ('cheap')                         -> throws the cheap tier only
  badge 8 pre-credits (victory lap)                          -> dex_push divert SUPPRESSED
  post-champion (FLAG_SYS_GAME_CLEAR)                        -> dex_push restored
"""
import json
import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Mac dev box has no mgba/emulator stack — stub it so battle_agent's deps import clean
# (same pattern as recon_lap_check).
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import pokemon_policy as pol
import battle_agent as BA


class FakeBridge:
    def set_input_owner(self, owner):
        pass


def make_agent(pocket, hunt_pending, logs):
    """A BattleAgent whose bag + hunt signal are synthetic: `pocket` is the balls-pocket
    rows [(item_id, qty), ...] in display order; `hunt_pending` stubs the legendary signal."""
    ag = BA.BattleAgent(FakeBridge(), on_event=lambda *a, **k: None,
                        render=lambda: None, log=logs.append)
    ag._balls_pocket = lambda: list(pocket)
    ag._hunt_pending = lambda: bool(hunt_pending)
    ag._is_trainer_battle = lambda: False
    return ag

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

    print("== 6. THE RESERVE: 8 Ultras only + hunt pending + trash target -> REFUSE ==")
    # The 2026-08-05 live bug verbatim: pref=cheap degraded to Ultra because Ultras were
    # the only tier in the bag — the hunt's stock, spent on a Meowth L31.
    logs = []
    ag = make_agent([(2, 8)], hunt_pending=True, logs=logs)
    check("cheap pick refuses the reserved tier", ag._pick_ball("cheap") == (None, None))
    check("spendable-for-cheap reads 0 (8 Ultras all reserved)",
          ag._spendable_for_pref("cheap") == 0 and ag._ball_count() == 8)
    res = ag.throw_ball(pref="cheap")
    check("throw_ball -> no_balls, never a press", res == "no_balls")
    check("the RESERVE line is LOUD in the log",
          any("RESERVE — ultras held for the hunt" in l for l in logs), f"logs={logs}")

    print("== 7. hunt target ('best') spends Ultras freely; Master stays Mewtwo-only ==")
    ag2 = make_agent([(2, 8), (1, 1)], hunt_pending=True, logs=[])
    check("'best' picks the Ultra (row 0)", ag2._pick_ball("best") == (2, 0))
    check("'best' sees the whole spendable stack", ag2._spendable_for_pref("best") == 8)
    check("Master never picked without allow_master", ag2._pick_ball("cheap")[0] != 1)
    check("allow_master (the Mewtwo seat) picks the Master",
          ag2._pick_ball("best", allow_master=True) == (1, 1))

    print("== 8. mixed bag + hunt pending: 'cheap' spends the cheap tier, holds Ultras ==")
    ag3 = make_agent([(2, 8), (4, 5)], hunt_pending=True, logs=[])
    check("Poké Ball thrown, Ultras untouched", ag3._pick_ball("cheap") == (4, 1))
    check("spendable counts only the cheap tier", ag3._spendable_for_pref("cheap") == 5)
    ag3b = make_agent([(2, 8), (3, 2)], hunt_pending=True, logs=[])
    check("Great Ball is a genuinely-cheap tier too", ag3b._pick_ball("cheap") == (3, 1))

    print("== 9. reserve lifts: hunt done / kill switch -> old behavior ==")
    ag4 = make_agent([(2, 8)], hunt_pending=False, logs=[])
    check("no hunt pending -> cheap may spend Ultras (pre-reserve behavior)",
          ag4._pick_ball("cheap") == (2, 0) and ag4._spendable_for_pref("cheap") == 8)
    _saved = BA.HUNT_BALL_RESERVE
    BA.HUNT_BALL_RESERVE = False
    ag5 = make_agent([(2, 8)], hunt_pending=True, logs=[])
    check("POKEMON_HUNT_BALL_RESERVE=0 reverts", ag5._pick_ball("cheap") == (2, 0))
    BA.HUNT_BALL_RESERVE = _saved

    print("== 10. LAP NO-CATCH: the dex_push gate (badge 8 pre-credits suppresses) ==")
    # (badges, game_clear, aide_paid, owned_count, owns_species, spendable)
    fire, hold = BA.dex_push_gate(8, False, False, 15, False, 5)
    check("victory lap (badge 8, credits not rolled) -> SUPPRESSED", not fire and hold)
    fire, hold = BA.dex_push_gate(8, True, False, 15, False, 5)
    check("post-champion (GAME_CLEAR) -> dex_push restored", fire and not hold)
    fire, hold = BA.dex_push_gate(6, False, False, 15, False, 5)
    check("mid-climb (badge 6) -> untouched, push fires", fire and not hold)
    fire, hold = BA.dex_push_gate(8, False, False, 15, False, 0)
    check("reserved-only bag (spendable 0) never arms the divert", not fire and not hold)
    _saved_lap = BA.LAP_NO_CATCH
    BA.LAP_NO_CATCH = False
    fire, hold = BA.dex_push_gate(8, False, False, 15, False, 5)
    check("POKEMON_LAP_NO_CATCH=0 reverts the suppression", fire and not hold)
    BA.LAP_NO_CATCH = _saved_lap

    print("== 11. creator 'use an ultra ball' waives the reserve (voice override) ==")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "creator_order.json")
        ag6 = make_agent([(2, 8)], hunt_pending=True, logs=[])
        ag6._creator_catch_order_path = lambda: p
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"order": "catch_now", "raw": "catch that meowth"}, f)
        check("plain catch order -> reserve HOLDS", ag6._creator_order_wants_ultra() is False)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"order": "catch_now", "raw": "use an Ultra Ball on it"}, f)
        check("'use an ultra ball' -> waived", ag6._creator_order_wants_ultra() is True)

    print("== 12. LAP catch_keeper suppression (the 'abra -> alakazam' GM fix) ==")
    from pokemon_planner import TeamPlanner
    import pokemon_planner as pp
    tp = TeamPlanner(log=lambda *a, **k: None)
    party = [{"species": "blastoise", "level": 61, "hp": 1, "maxhp": 1, "species_id": 9}]
    act8 = tp.assess(party, badges=8)
    check("badge 8 pre-credits -> NO catch_keeper (allowlist empty)",
          (act8 or {}).get("kind") != "catch_keeper", f"got {(act8 or {}).get('kind')}")
    tp2 = TeamPlanner(log=lambda *a, **k: None)
    act6 = tp2.assess(party, badges=6)
    check("badge 6 -> keeper diverts untouched",
          (act6 or {}).get("kind") == "catch_keeper", f"got {(act6 or {}).get('kind')}")
    _saved_pp = pp.LAP_NO_CATCH
    pp.LAP_NO_CATCH = False
    tp3 = TeamPlanner(log=lambda *a, **k: None)
    act8b = tp3.assess(party, badges=8)
    check("POKEMON_LAP_NO_CATCH=0 -> planner reverts",
          (act8b or {}).get("kind") == "catch_keeper", f"got {(act8b or {}).get('kind')}")
    pp.LAP_NO_CATCH = _saved_pp

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
