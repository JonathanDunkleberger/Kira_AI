"""recon_catch_discipline_test.py — weaken-then-throw discipline (2026-08-04, the badge-8
full-HP ball-burn) + THE LEGENDARY BALL RESERVE / LAP NO-CATCH (2026-08-05, the Kindle Road
Meowth Ultra burn) + THE SEND-IN GATE / CATCH-ABORT LATCH / PARTY-WIPE GUARD (2026-08-05,
the Magmar forced-send-in wedge). Pure synthetic tests — pol.chip_move_pick for the weaken
pick, and the battle_agent ball-selection/divert/faint gates driven through a stub bridge
(no ROM/emulator).

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
  fainted active mid-catch                                   -> throw ABANDONED, send-in first
  3 consecutive throw aborts                                 -> CATCH-ABORT latch (no throws)
  abort latch / party-wipe risk                              -> catch_now RELEASED (voiced)
  last healthy mon + non-legendary target                    -> NO catch divert (fight/flee)
+ THE LEGENDARY CATCH DOCTRINE (2026-08-05, standing at Moltres with 6 Ultras):
  legendary + status at full HP                              -> NOT ready (red zone only)
  legendary chip                                             -> aims 15%, guards outrank
  Surf/Water Pulse vs fire/flying                            -> excluded (overkill estimate)
  no status on the legendary before a throw                  -> SLEEP RUNG (active / switch)
  sleep casts / sleeper switch                               -> bounded, wipe-guarded
  under-leveled bench vs a legendary                         -> chipper band admits it
  catch failed (fainted / no balls / whiteout)               -> FREE RETRY: 'pre-<quarry>'
                                                                checkpoint reload, bounded
  hunt owed + thin pocket at a mapped Ultra shelf            -> war-chest restock (20 Ultras)
+ THE FLUID LAP / PROXIMITY TRUMP (2026-08-05 EMERGENCY, the Eevee divert two tiles from
  Moltres):
  fought-flag set, UNCAUGHT, 'pre-<key>' bank exists         -> lap item STAYS PENDING
  standing in a hunt's anchor set, quarry uncaught           -> that hunt trumps EVERYTHING
  loop-burned skip/fail marks while standing there           -> refunded once per run
  lap ordering                                               -> cheapest by live cost, declared
                                                                order only as the tiebreak
  spent-but-retryable inside a hunt leg (spent_final)        -> reload, encounter live again
  boot with a battled-away uncaught legendary + bank         -> LEGENDARY REWIND to the bird
+ THE VERIFIED RATCHET (2026-08-05 URGENT, the poisoned 'pre-moltres' bank):
  loaded bank has the fought-flag set, quarry uncaught       -> POISONED: ratchet to the
                                                                next older same-region bank
  cross-region banks in the ratchet walk                     -> skipped (never a sea teleport)
  every candidate poisoned                                   -> original live state restored
  banking 'pre-<key>' while the fought-flag is set           -> REFUSED (poisoned-bank law)
  post-load flag read                                        -> fresh RAM, never cached
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

    print("== 13. SEND-IN GATE: fainted active -> catch abandoned, send-in allowed ==")
    # The 2026-08-05 14:10 wedge verbatim: Lapras fainted mid-catch_now on a Machoke; the
    # forced send-in state ate every bag-open A while the catch loop re-threw forever.
    check("our active down -> send_in (throw abandoned this turn)",
          BA.catch_turn_gate(True, 3, 0, False, False) == "send_in")
    check("send-in outranks throws on a legendary hunt too",
          BA.catch_turn_gate(True, 3, 0, False, True) == "send_in")
    check("healthy field, healthy party -> normal throw turn",
          BA.catch_turn_gate(False, 3, 0, False, False) == "throw")
    # _catch_our_down: the verified-RAM read with the pitfall-26 party cross-check.
    logs13 = []
    ag13 = make_agent([(4, 5)], hunt_pending=False, logs=logs13)
    ag13._is_double = lambda: False
    _orig_rb = BA.st.read_battle
    BA.st.read_battle = lambda b: {"ours": {"hp": 0, "maxhp": 100},
                                   "enemy": {"hp": 50, "maxhp": 80}}
    try:
        ag13._true_active_party_hp = lambda: (0, 100)
        check("gBattleMons 0 + party 0 -> DOWN (both structs agree)",
              ag13._catch_our_down() is True)
        ag13._true_active_party_hp = lambda: (100, 100)
        check("party says STANDING -> not down (stale display corpse, pitfall 26)",
              ag13._catch_our_down() is False)
    finally:
        BA.st.read_battle = _orig_rb
    ag13._catch_our_down = lambda state=None: True
    check("throw_ball REFUSES while our active is down (no press, 'our_down')",
          ag13.throw_ball(pref="cheap") == "our_down")
    check("the refusal is LOUD", any("THROW REFUSED" in l for l in logs13), f"logs={logs13}")

    print("== 14. CATCH-ABORT LATCH: 3 consecutive bag-open aborts latch the battle ==")
    logs14 = []
    ag14 = make_agent([(4, 5)], hunt_pending=False, logs=logs14)
    for _ in range(BA.CATCH_ABORT_MAX - 1):
        ag14._note_throw_abort("bag would not open")
    check(f"{BA.CATCH_ABORT_MAX - 1} aborts do NOT latch", not ag14._catch_abort)
    ag14._note_throw_abort("bag would not open")
    check(f"abort #{BA.CATCH_ABORT_MAX} LATCHES", ag14._catch_abort is True)
    check("the latch is LOUD in the log",
          any("CATCH-ABORT LATCHED" in l for l in logs14), f"logs={logs14}")
    ag14._throw_bag_aborts = 0            # a real throw resets the streak...
    check("...but the latch itself holds for the battle", ag14._catch_abort is True)
    check("gate refuses throws under the latch",
          BA.catch_turn_gate(False, 6, 0, True, False) == "abort")
    check("abort count >= max reads latched even pre-flag",
          BA.catch_turn_gate(False, 6, BA.CATCH_ABORT_MAX, False, False) == "abort")
    check("latch outranks the send-in verdict (run()'s drain owns the recovery)",
          BA.catch_turn_gate(True, 6, 0, True, False) == "abort")

    print("== 15. ORDER RELEASE: latch / party-wipe releases catch_now with a voice line ==")
    check("target fled (clean failure) -> LAW order KEPT",
          BA.catch_order_release("fled", False, 3) is False)
    check("no_balls with a healthy party -> KEPT (Mart first, then the catch)",
          BA.catch_order_release("no_balls", False, 3) is False)
    check("abort latch -> RELEASED", BA.catch_order_release("stuck", True, 3) is True)
    check("catch_abort verdict -> RELEASED", BA.catch_order_release("catch_abort", False, 3) is True)
    check("party-wipe risk (1 healthy) -> RELEASED",
          BA.catch_order_release("no_balls", False, 1) is True)
    check("party_risk verdict -> RELEASED", BA.catch_order_release("party_risk", False, 3) is True)
    logs15, voiced = [], []
    ag15 = BA.BattleAgent(FakeBridge(), on_event=lambda s, **k: voiced.append(s),
                          render=lambda: None, log=logs15.append)
    cleared = []
    ag15._clear_creator_catch_order = lambda: cleared.append(True)
    ag15._release_catch_order_loud("party-wipe risk: down to the last healthy mon",
                                   "I can't catch right now — I'm down to my last Pokémon.")
    check("release clears the latched order", bool(cleared))
    check("release is VOICED so Jonny hears why",
          any("last Pokémon" in v for v in voiced), f"voiced={voiced}")
    check("release is LOUD in the log",
          any("catch_now RELEASED" in l for l in logs15), f"logs={logs15}")

    print("== 16. PARTY-WIPE GUARD: last healthy mon -> no catch divert (non-legendary) ==")
    check("1 healthy + non-hunt target -> party_risk (fight/flee, never a catch)",
          BA.catch_turn_gate(False, 1, 0, False, False) == "party_risk")
    check("0 healthy reserves standing (mid-wipe) -> still refused",
          BA.catch_turn_gate(False, 0, 0, False, False) == "party_risk")
    check("1 healthy + LEGENDARY hunt -> attempt allowed (send-in still outranks throws)",
          BA.catch_turn_gate(False, 1, 0, False, True) == "throw")
    check("2 healthy + non-hunt -> normal catch",
          BA.catch_turn_gate(False, 2, 0, False, False) == "throw")

    print("== 17. LEGENDARY ready band: red zone only — a status never green-lights ==")
    _SLP = 0x07                                   # any nonzero status1 (sleep turns)
    check("legendary at 18% -> ready", BA.catch_ready(0.18, 0, True) is True)
    check("legendary ASLEEP at full HP -> NOT ready (the full-HP Sing throw ban)",
          BA.catch_ready(1.0, _SLP, True) is False)
    check("legendary asleep at 50% -> still NOT ready", BA.catch_ready(0.50, _SLP, True) is False)
    check("legendary asleep in the red zone -> ready", BA.catch_ready(0.19, _SLP, True) is True)
    check("generic keeps the old law: status alone -> ready", BA.catch_ready(0.95, _SLP, False))
    check("generic 45% no status -> ready", BA.catch_ready(0.45, 0, False))
    check("generic 60% no status -> NOT ready", BA.catch_ready(0.60, 0, False) is False)
    check("red-zone dials: ready 20% / chip 15%",
          BA.CATCH_READY_FRAC_LEGEND == 0.20 and BA.CATCH_CHIP_TARGET_LEGEND == 0.15)

    print("== 18. Moltres move table: Surf/Water Pulse excluded, Bite chips, guards stop ==")
    # Blastoise L63 vs L50 Moltres (fire/flying) — the live matchup at the summit.
    moltres = ["fire", "flying"]
    surf = {"id": 57, "name": "Surf", "type": "water", "power": 95, "pp": 15}
    wpulse = {"id": 352, "name": "Water Pulse", "type": "water", "power": 60, "pp": 20}
    icebeam = {"id": 58, "name": "Ice Beam", "type": "ice", "power": 95, "pp": 10}
    bite = {"id": 44, "name": "Bite", "type": "dark", "power": 60, "pp": 25}
    check("Surf estimate is pure overkill (2x SE + STAB)",
          pol.chip_hit_frac(surf, moltres, 63, 50, ["water", "water"]) > 1.0)
    check("Water Pulse likewise excluded",
          pol.chip_hit_frac(wpulse, moltres, 63, 50, ["water", "water"]) > 1.0)
    check("Ice Beam is NOT the gentle option (2x flying x 0.5 fire = neutral, still hot)",
          pol.chip_hit_frac(icebeam, moltres, 63, 50, ["water", "water"]) > pol.CHIP_KO_SAFETY)
    i18, est18, safe18 = pol.chip_move_pick([surf, wpulse, icebeam, bite], moltres, 63, 50,
                                            foe_hp_frac=1.0, our_types=["water", "water"])
    check("the pick is Bite (lowest estimate), and it is safe at full HP",
          i18 == 3 and safe18 is True, f"est={est18:.0%}")
    _, _, safe18b = pol.chip_move_pick([surf, wpulse, icebeam, bite], moltres, 63, 50,
                                       foe_hp_frac=0.33, our_types=["water", "water"])
    check("near the band even Bite is refused — throw, never the KO", safe18b is False)

    print("== 19. deep chip: legendary target 15% honored, generic stops high ==")
    logs19 = []
    ag19 = make_agent([(2, 6)], hunt_pending=True, logs=logs19)
    foe19 = {"hp": 100}
    weak = [{"id": 33, "name": "Tackle", "type": "normal", "power": 10, "pp": 30}]
    _orig_rb19, _orig_ib19 = BA.st.read_battle, BA.st.in_battle
    BA.st.read_battle = lambda b: {"ours": {"moves": list(weak), "level": 50,
                                            "types": ["normal"]},
                                   "enemy": {"hp": foe19["hp"], "maxhp": 100, "level": 50,
                                             "types": ["normal"]}}
    BA.st.in_battle = lambda b: True
    ag19._fire_move = lambda i: foe19.__setitem__("hp", foe19["hp"] - 15)
    try:
        ag19._weaken_hp(target_frac=BA.CATCH_CHIP_TARGET_LEGEND, max_hits=BA.LEGEND_CHIP_HITS)
        check("legendary chip drives into the red zone (<=15%)", foe19["hp"] <= 15,
              f"ended at {foe19['hp']}%")
        foe19["hp"] = 100
        ag19._weaken_hp()                                  # generic defaults: 30% / 4 hits
        check("generic chip stays bounded high (old behavior untouched)", foe19["hp"] >= 30,
              f"ended at {foe19['hp']}%")
    finally:
        BA.st.read_battle, BA.st.in_battle = _orig_rb19, _orig_ib19

    print("== 20. THE SLEEP RUNG: party sleeper found, casts bounded, wipe-guarded ==")
    class PartyBridge:
        """rd8/rd16 party struct stub: mons = {slot: (level, hp, maxhp)}."""
        def __init__(self, cnt, mons):
            self.cnt, self.mons = cnt, mons
        def set_input_owner(self, o):
            pass
        def rd8(self, a):
            if a == BA.ram.GPLAYER_PARTY_CNT:
                return self.cnt
            s, r = divmod(a - BA.ram.GPLAYER_PARTY, 100)
            return self.mons.get(s, (0, 0, 0))[0] if r == 0x54 else 0
        def rd16(self, a):
            s, r = divmod(a - BA.ram.GPLAYER_PARTY, 100)
            if r == 0x56:
                return self.mons.get(s, (0, 0, 0))[1]
            if r == 0x58:
                return self.mons.get(s, (0, 0, 0))[2]
            return 0
    # party: Blastoise L63 (active), Lapras L25 at 46% with Sing, Fearow L38 fainted
    logs20 = []
    ag20 = BA.BattleAgent(PartyBridge(3, {0: (63, 200, 200), 1: (25, 60, 130),
                                          2: (38, 0, 90)}),
                          on_event=lambda *a, **k: None, render=lambda: None,
                          log=logs20.append)
    _orig_rpm, _orig_rps = BA.st.read_party_moves, BA.st.read_party_species
    BA.st.read_party_moves = lambda b, s: [57, 47, 0, 0] if s == 1 else [64, 0, 0, 0]
    BA.st.read_party_species = lambda b, s=0: 131
    try:
        check("party sleeper found: Lapras slot 1 carries Sing (47)",
              ag20._party_sleeper_slot() == (1, 47))
        # rung 1: active (Blastoise) has no sleep move -> the ONE sleeper switch
        switched = []
        ag20._switch_to_slot = lambda s, sp: (switched.append(s), "switched")[1]
        ag20._legend_sleeps, ag20._legend_sleeper_tried = 0, False
        st20 = {"enemy": {"status1": 0}, "ours": {"moves": list(BLASTOISE_L59[:1]),
                                                  "species": 9}}
        check("no sleep on the active -> sleeper SWITCH consumed the turn",
              ag20._legend_sleep_rung(st20) is True and switched == [1])
        check("the switch is one-shot (second ask throws instead)",
              ag20._legend_sleep_rung(st20) is False)
        # rung 2: active now HAS Sing -> fire it, budget counts
        fired = []
        ag20._fire_move = lambda i: fired.append(i)
        sing_state = {"enemy": {"status1": 0},
                      "ours": {"moves": [{"id": 47, "name": "Sing", "power": 0, "pp": 15}],
                               "species": 131}}
        check("active Sing fires (cast 1)", ag20._legend_sleep_rung(sing_state) is True
              and fired == [0] and ag20._legend_sleeps == 1)
        ag20._legend_sleeps = BA.LEGEND_RESLEEP_MAX
        check("cast budget spent -> throw instead", ag20._legend_sleep_rung(sing_state) is False)
        ag20._legend_sleeps = 0
        check("foe already statused -> rung stands down (go throw)",
              ag20._legend_sleep_rung({"enemy": {"status1": 0x07},
                                       "ours": sing_state["ours"]}) is False)
        # wipe guard: 1 healthy mon -> no sleeper switch, ever
        ag21 = BA.BattleAgent(PartyBridge(3, {0: (63, 200, 200), 1: (25, 0, 130),
                                              2: (38, 0, 90)}),
                              on_event=lambda *a, **k: None, render=lambda: None,
                              log=logs20.append)
        ag21._legend_sleeps, ag21._legend_sleeper_tried = 0, False
        ag21._switch_to_slot = lambda s, sp: "switched"
        check("party too thin (1 healthy) -> switch refused",
              ag21._legend_sleep_rung(st20) is False)
    finally:
        BA.st.read_party_moves, BA.st.read_party_species = _orig_rpm, _orig_rps

    print("== 21. chipper band vs a legendary: the under-leveled bench qualifies ==")
    ag22 = BA.BattleAgent(PartyBridge(4, {0: (63, 200, 200), 1: (38, 90, 90),
                                          2: (25, 60, 130), 3: (63, 180, 200)}),
                          on_event=lambda *a, **k: None, render=lambda: None,
                          log=[].append)
    check("generic band vs L50: nobody fits (bench under-levels the foe)",
          ag22._catch_chipper_slot(50) is None)
    check("legendary band admits the bench, prefers the strongest under the ceiling",
          ag22._catch_chipper_slot(50, legend=True) == 1)
    check("a 13-over teammate stays excluded even for a legendary (ceiling holds)",
          ag22._catch_chipper_slot(50, legend=True) != 3)

    print("== 22. THE FREE RETRY: failed catch reloads 'pre-<quarry>', bounded ==")
    import legendary_strikes as LS
    hunt = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt.b = object()
    hunt._catch_retries = 0
    logs22 = []
    hunt.log = logs22.append
    reloads = []
    hunt.camp = types.SimpleNamespace(
        _reload_hunt_checkpoint=lambda key: (reloads.append(f"pre-{key}"), True)[1],
        on_event=lambda *a, **k: None)
    _orig_owns, _orig_flag = LS.ram.pokedex_owns, LS.fm.read_flag
    world = {"owned": False, "fought": False}
    LS.ram.pokedex_owns = lambda b, sp: world["owned"]
    LS.fm.read_flag = lambda b, f: world["fought"]
    try:
        check("encounter still live -> nothing to retry", hunt._retry_failed_catch() is False)
        world["fought"] = True
        check("fainted/failed -> RELOAD 'pre-moltres', budget counts",
              hunt._retry_failed_catch() is True and reloads == ["pre-moltres"]
              and hunt._catch_retries == 1)
        world["owned"] = True
        check("caught -> never a retry (owned outranks the fought flag)",
              hunt._retry_failed_catch() is False)
        world["owned"] = False
        hunt._catch_retries = hunt.LEGEND_CATCH_RETRIES
        check("retry budget spent -> accept 'battled', LOUD",
              hunt._retry_failed_catch() is False
              and any("retry budget is spent" in l for l in logs22))
        hunt._catch_retries = 0
        hunt.camp._reload_hunt_checkpoint = lambda key: False
        check("no checkpoint on disk -> honest decline",
              hunt._retry_failed_catch() is False and hunt._catch_retries == 0)
    finally:
        LS.ram.pokedex_owns, LS.fm.read_flag = _orig_owns, _orig_flag

    print("== 23. LABELED RELOAD picks the NEWEST 'pre-moltres' bank; war-chest dial ==")
    import campaign as C
    check("HUNT_ULTRA_TARGET is the healthy stack (20)", C.HUNT_ULTRA_TARGET == 20)
    with tempfile.TemporaryDirectory() as td:
        _orig_sc = C.STATES_CAMPAIGN
        C.STATES_CAMPAIGN = td
        try:
            root = os.path.join(td, "checkpoints")
            for name, body in (("20260805_170000_mt-ember_8b_9h00m_pre-moltres", b"OLD"),
                               ("20260805_171500_mt-ember_8b_9h15m_pre-moltres", b"NEW"),
                               ("20260805_171600_mt-ember_8b_9h16m_moltres-leg", b"LEG")):
                os.makedirs(os.path.join(root, name), exist_ok=True)
                with open(os.path.join(root, name, C.CAMPAIGN_SAVE), "wb") as f:
                    f.write(body)
            camp = C.Campaign.__new__(C.Campaign)
            loaded = []
            camp.b = types.SimpleNamespace(load_state=lambda by: loaded.append(by),
                                           save_state=lambda: b"resnap")
            camp._gain_sig = lambda: 0
            camp._wait_overworld = lambda *a, **k: True
            camp._save_campaign = lambda *a, **k: True
            check("newest 'pre-moltres' bank wins (not the older one, not the leg)",
                  camp._reload_labeled_checkpoint("pre-moltres") is True
                  and loaded == [b"NEW"])
            check("re-anchored the recent-good to the reloaded moment",
                  camp._last_good_state == b"resnap")
            check("a tag with no banks declines honestly",
                  camp._reload_labeled_checkpoint("pre-zapdos") is False)
        finally:
            C.STATES_CAMPAIGN = _orig_sc

    print("== 24. LAP HONESTY: a fought-flag never books an UNCAUGHT hunt done (bank live) ==")
    # The Eevee-divert root cause verbatim: flee after 5 broken balls set FLAG_FOUGHT_MOLTRES,
    # _lap_pending read 'done', the lap marched to eevee — two tiles from the bird.
    _orig_owns24, _orig_flag24 = C.ram.pokedex_owns, C.fm.read_flag
    world24 = {"owned": False, "fought": True, "bank": True}
    C.ram.pokedex_owns = lambda b, sp: world24["owned"]
    C.fm.read_flag = lambda b, f: world24["fought"]
    try:
        camp24 = C.Campaign.__new__(C.Campaign)
        camp24.b = object()
        camp24._lap_skipped = set()
        camp24._has_labeled_checkpoint = lambda tag: world24["bank"]
        check("fought + uncaught + 'pre-moltres' bank -> STILL PENDING (the free retry rewinds)",
              camp24._lap_pending("moltres") is True)
        world24["bank"] = False
        check("fought + no bank on disk -> honestly done ('battled' stands)",
              camp24._lap_pending("moltres") is False)
        world24["owned"], world24["bank"] = True, True
        check("caught -> done regardless of banks", camp24._lap_pending("moltres") is False)
        # _hunt_ready mirrors the same law for the GATES (the armed view).
        world24["owned"] = False
        camp24._balls_pocket_count = lambda i: 0
        check("_hunt_ready: fought + bank -> READY (gate arms; reload IS the restock)",
              camp24._hunt_ready(146, 0x052, 0x2BD, key="moltres") is None)
        world24["bank"] = False
        check("_hunt_ready: fought + no bank -> 'spent' (old law holds)",
              camp24._hunt_ready(146, 0x052, 0x2BD, key="moltres") == "spent")
    finally:
        C.ram.pokedex_owns, C.fm.read_flag = _orig_owns24, _orig_flag24

    print("== 25. THE FLUID LAP: proximity trump + cost ordering from her live position ==")
    _orig_map25, _orig_owns25 = C.tv.map_id, C.ram.pokedex_owns
    pos25 = {"here": (1, 101)}                    # Mt. Ember summit — standing at the bird
    C.tv.map_id = lambda b: pos25["here"]
    C.ram.pokedex_owns = lambda b, sp: False
    try:
        camp25 = C.Campaign.__new__(C.Campaign)
        camp25.b = object()
        camp25._lap_skipped, camp25._lap_fails = set(), {}
        pend25 = {"earthquake", "box_bench", "moltres", "articuno", "eevee", "zapdos"}
        camp25._lap_pending = lambda k: k in pend25
        check("summit -> MOLTRES (proximity trump beats earthquake/eevee/everything)",
              camp25._victory_lap_next() == "moltres")
        # loop-burned marks refund ONCE while she stands there
        camp25._lap_skipped, camp25._lap_fails = {"moltres"}, {"moltres": 6}
        camp25._lap_verdict_logged = None
        check("skipped+failed moltres AT ITS MAP -> refunded and picked anyway",
              camp25._victory_lap_next() == "moltres"
              and "moltres" not in camp25._lap_skipped
              and "moltres" not in camp25._lap_fails)
        # cost table sanity: the exact matchup Jonny named
        check("summit prices: moltres=0, eevee(Celadon)=2 cross-region",
              camp25._lap_item_cost("moltres", (1, 101)) == 0
              and camp25._lap_item_cost("eevee", (1, 101)) == 2)
        pos25["here"] = tuple(C.CELADON)          # standing in Celadon instead
        camp25b = C.Campaign.__new__(C.Campaign)
        camp25b.b = object()
        camp25b._lap_skipped, camp25b._lap_fails = set(), {}
        pend25b = {"moltres", "articuno", "eevee", "zapdos"}
        camp25b._lap_pending = lambda k: k in pend25b
        check("Celadon -> EEVEE (cost 0 beats the Kanto birds at 1, moltres ferry at 2)",
              camp25b._victory_lap_next() == "eevee")
        camp25c = C.Campaign.__new__(C.Campaign)
        camp25c.b = object()
        camp25c._lap_skipped, camp25c._lap_fails = set(), {}
        pend25c = {"articuno", "zapdos"}
        camp25c._lap_pending = lambda k: k in pend25c
        check("comparable costs -> declared order is the tiebreak (articuno before zapdos)",
              camp25c._victory_lap_next() == "articuno")
        # trump requires UNCAUGHT: an owned bird never parks her at its map
        pos25["here"] = (1, 101)
        C.ram.pokedex_owns = lambda b, sp: True
        camp25d = C.Campaign.__new__(C.Campaign)
        camp25d.b = object()
        camp25d._lap_skipped, camp25d._lap_fails = set(), {}
        pend25d = {"eevee", "zapdos"}
        camp25d._lap_pending = lambda k: k in pend25d
        check("moltres CAUGHT -> no trump, lap moves on by cost/order",
              camp25d._victory_lap_next() in ("eevee", "zapdos"))
    finally:
        C.tv.map_id, C.ram.pokedex_owns = _orig_map25, _orig_owns25

    print("== 26. spent_final: a spent-but-UNCAUGHT quarry reloads before any leg home ==")
    hunt26 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt26.b = object()
    hunt26.log = [].append
    hunt26._catch_retries = 0
    reloads26 = []
    hunt26.camp = types.SimpleNamespace(
        _reload_hunt_checkpoint=lambda key: (reloads26.append(f"pre-{key}"), True)[1],
        on_event=lambda *a, **k: None)
    _orig_owns26, _orig_flag26 = LS.ram.pokedex_owns, LS.fm.read_flag
    world26 = {"owned": False, "fought": False}
    LS.ram.pokedex_owns = lambda b, sp: world26["owned"]
    LS.fm.read_flag = lambda b, f: world26["fought"]
    try:
        hunt26.spent = lambda: False
        check("not spent -> False (hunt proceeds normally)", hunt26.spent_final() is False)
        hunt26.spent = lambda: True
        world26["fought"] = True
        check("spent + retryable -> RELOADED, reads NOT-spent (encounter live again)",
              hunt26.spent_final() is False and reloads26 == ["pre-moltres"])
        hunt26._catch_retries = hunt26.LEGEND_CATCH_RETRIES
        check("spent + budget exhausted -> True (homebound flow may run)",
              hunt26.spent_final() is True)
    finally:
        LS.ram.pokedex_owns, LS.fm.read_flag = _orig_owns26, _orig_flag26

    print("== 27. LEGENDARY REWIND AT BOOT + the bank-exists predicate ==")
    with tempfile.TemporaryDirectory() as td27:
        _orig_sc27 = C.STATES_CAMPAIGN
        C.STATES_CAMPAIGN = td27
        try:
            root27 = os.path.join(td27, "checkpoints")
            good = os.path.join(root27, "20260805_171500_mt-ember_8b_9h15m_pre-moltres")
            os.makedirs(good, exist_ok=True)
            with open(os.path.join(good, C.CAMPAIGN_SAVE), "wb") as f:
                f.write(b"BANK")
            os.makedirs(os.path.join(root27, "20260805_171600_x_pre-zapdos.partial"),
                        exist_ok=True)
            camp27 = C.Campaign.__new__(C.Campaign)
            camp27.b = object()
            check("bank predicate: real dir True, partial/missing False",
                  camp27._has_labeled_checkpoint("pre-moltres") is True
                  and camp27._has_labeled_checkpoint("pre-zapdos") is False
                  and camp27._has_labeled_checkpoint("pre-articuno") is False)
        finally:
            C.STATES_CAMPAIGN = _orig_sc27
    _orig_owns27, _orig_flag27 = C.ram.pokedex_owns, C.fm.read_flag
    world27 = {"owned": {146: False}, "fought": {0x2BD: True}}
    C.ram.pokedex_owns = lambda b, sp: world27["owned"].get(sp, True)
    C.fm.read_flag = lambda b, f: world27["fought"].get(f, False)
    try:
        camp27b = C.Campaign.__new__(C.Campaign)
        camp27b.b = object()
        reloads27, voiced27 = [], []
        camp27b._reload_hunt_checkpoint = lambda key: (reloads27.append(f"pre-{key}"), True)[1]
        camp27b.on_event = lambda *a, **k: voiced27.append(a)
        check("battled-away uncaught moltres at boot -> REWOUND to 'pre-moltres', voiced",
              camp27b._legend_rewind_at_boot() is True and reloads27 == ["pre-moltres"]
              and bool(voiced27))
        world27["owned"][146] = True
        reloads27.clear()
        check("caught -> boot rewind stands down", camp27b._legend_rewind_at_boot() is False
              and reloads27 == [])
        world27["owned"][146] = None              # unreadable dex — NEVER rewind blind
        check("unreadable dex -> honest no-op (a possibly-caught mon is never rewound)",
              camp27b._legend_rewind_at_boot() is False and reloads27 == [])
    finally:
        C.ram.pokedex_owns, C.fm.read_flag = _orig_owns27, _orig_flag27

    print("== 28. THE VERIFIED RATCHET: poisoned banks rejected, older clean bank wins ==")
    # The live failure verbatim: the newest 'pre-moltres' was banked AFTER the fled fight —
    # fought-flag set inside the savestate, summit empty. The ratchet must reject it by a
    # FRESH post-load flag read and land on the next older same-region bank that still
    # contains the bird (cross-region banks skipped; all-poisoned restores the live state).
    class RatchetBridge:
        def __init__(self):
            self.body = b"LIVE"
        def load_state(self, by):
            self.body = by
        def save_state(self):
            return self.body
    with tempfile.TemporaryDirectory() as td28:
        _orig_sc28 = C.STATES_CAMPAIGN
        C.STATES_CAMPAIGN = td28
        root28 = os.path.join(td28, "checkpoints")

        def _bank28(name, body, mp=None):
            os.makedirs(os.path.join(root28, name), exist_ok=True)
            with open(os.path.join(root28, name, C.CAMPAIGN_SAVE), "wb") as f:
                f.write(body)
            if mp is not None:
                with open(os.path.join(root28, name, "checkpoint.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"map": list(mp)}, f)
        _bank28("20260805_175000_mt-ember-summit_8b_pre-moltres", b"POISONED")
        _bank28("20260805_174500_celadon-city_8b_roam", b"KANTO", mp=(3, 6))
        _bank28("20260805_174000_mt-ember-2f_8b_moltres-leg", b"CLEAN", mp=(1, 99))
        camp28 = C.Campaign.__new__(C.Campaign)
        camp28.b = RatchetBridge()
        camp28._gain_sig = lambda: 0
        camp28._wait_overworld = lambda *a, **k: True
        camp28._save_campaign = lambda *a, **k: True
        # the flag reads key off WHICH savestate body is loaded — passing these checks IS
        # the proof the verify reads fresh post-load RAM, not a cached pre-reload value.
        _orig_owns28, _orig_flag28 = C.ram.pokedex_owns, C.fm.read_flag
        C.ram.pokedex_owns = lambda b, sp: False
        C.fm.read_flag = lambda b, f: camp28.b.body in (b"POISONED", b"KANTO")
        try:
            check("poisoned 'pre-moltres' rejected -> ratchet lands the older CLEAN sevii bank",
                  camp28._reload_hunt_checkpoint("moltres") is True
                  and camp28.b.body == b"CLEAN")
            check("the kanto bank between them was never accepted (region law held)",
                  camp28.b.body != b"KANTO")
            # all-poisoned: overwrite the clean bank with a fought-flag state too
            _bank28("20260805_174000_mt-ember-2f_8b_moltres-leg", b"POISONED", mp=(1, 99))
            camp28.b.body = b"LIVE"
            check("every candidate poisoned -> declined AND the live state is restored",
                  camp28._reload_hunt_checkpoint("moltres") is False
                  and camp28.b.body == b"LIVE")
            # verify=None keeps the old single-shot behavior (test 23's contract)
            check("no-verify reload still takes the newest tag match blind",
                  camp28._reload_labeled_checkpoint("pre-moltres") is True
                  and camp28.b.body == b"POISONED")
        finally:
            C.ram.pokedex_owns, C.fm.read_flag = _orig_owns28, _orig_flag28
            C.STATES_CAMPAIGN = _orig_sc28

    print("== 29. POISONED-BANK LAW: 'pre-<key>' never banked with the fought-flag set ==")
    logs29, banked29 = [], []
    hunt29 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt29.b = object()
    hunt29.log = logs29.append
    hunt29.camp = types.SimpleNamespace(_bank_milestone=lambda label: banked29.append(label),
                                        _lap_fails={})
    _orig_owns29, _orig_flag29 = LS.ram.pokedex_owns, LS.fm.read_flag
    world29 = {"fought": True}
    LS.ram.pokedex_owns = lambda b, sp: False
    LS.fm.read_flag = lambda b, f: world29["fought"]
    try:
        check("fought-flag set + uncaught -> 'pre-moltres' bank REFUSED, loud",
              hunt29.strike_checkpoint("pre-moltres") is False and banked29 == []
              and any("REFUSED to bank" in l for l in logs29))
        check("non-pre milestones still bank under the flag (climb durability untouched)",
              hunt29.strike_checkpoint("moltres-leg") is True
              and banked29 == ["moltres-leg"])
        world29["fought"] = False
        check("flags clear -> 'pre-moltres' banks normally",
              hunt29.strike_checkpoint("pre-moltres") is True
              and banked29 == ["moltres-leg", "pre-moltres"])
    finally:
        LS.ram.pokedex_owns, LS.fm.read_flag = _orig_owns29, _orig_flag29

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
