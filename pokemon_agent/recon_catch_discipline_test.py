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
  hunt owed + thin pocket at a mapped Ultra shelf            -> war-chest restock (50 Ultras)
+ THE FLUID LAP / PROXIMITY TRUMP (2026-08-05 EMERGENCY, the Eevee divert two tiles from
  Moltres):
  fought-flag set, UNCAUGHT, 'pre-<key>' bank exists         -> lap item STAYS PENDING
  standing in a hunt's anchor set, quarry uncaught           -> that hunt trumps EVERYTHING
  loop-burned SKIP while standing on an uncaught hunt anchor -> refunded EVERY visit (never
                                                                re-skip the bird underfoot)
  loop-burned FAIL ledger while standing there               -> cleared ONCE per key/run
  lap ordering                                               -> cheapest by live cost, declared
                                                                order only as the tiebreak
  spent-but-retryable inside a hunt leg (spent_final)        -> reload, encounter live again
  boot with a battled-away uncaught legendary + bank         -> LEGENDARY REWIND to the bird
  Sevii-stranded after HONEST SKIP of uncaught Moltres       -> RE-ARM moltres (ride-home
                                                                must not no-op forever)
  One Island Harbor sailor approach from north of (8,6)      -> stand (8,5) face DOWN
                                                                (south stand has no BFS path)
  Three Island Mart behind unarmed biker pack                -> Game Corner prime first,
                                                                then clear gauntlet (A=YES),
                                                                THEN buy (never talk-loop)
  war-chest buy got 0 + wallet < Ultra price                 -> DONE/broke latch, sail home
                                                                (never Mart re-enter loop)
+ THE VERIFIED RATCHET (2026-08-05 URGENT, the poisoned 'pre-moltres' bank):
  loaded bank has the fought-flag set, quarry uncaught       -> POISONED: ratchet to the
                                                                next older same-region bank
  loaded bank has the hide-flag set, quarry uncaught         -> POISONED (same — either bit)
  mid-script bank (flags clear, settle flips hide/fought)    -> POISONED after settle re-verify
  cross-region banks in the ratchet walk                     -> skipped (never a sea teleport)
  every candidate poisoned                                   -> original live state restored
                                                                + SCREAM if live is also spent
  banking 'pre-<key>' while fought OR hide is set            -> REFUSED (poisoned-bank law)
  post-load flag read                                        -> fresh RAM, never cached
  verified clean landing                                     -> re-banks fresh 'pre-<key>'
+ THE PP LADDER (2026-08-05 LIVE, the one-Bite Moltres ball-burn):
  chip pick re-reads PP every swing                          -> a dry move never re-picked
  gentlest over the safety margin, can't faint from here     -> rung 2 swings anyway (legend)
  active out of damaging PP / faint-guard above hard floor   -> ACE-ONLY refuse + free-retry
  between red band and hard floor                            -> LOUD sanctioned throw (no bench)
  legendary never tags Fearow/Spearow in (bird spends)       -> Blastoise chips alone
  pre-encounter chip-PP audit                                -> party PP confessed pre-bank
+ THE PP RESTORE LEG (2026-08-05, Jonny: 'restore Blastoise's PP so she can Bite repeatedly'):
  ace safe swings < 5 OR party-wide < 8 vs the quarry        -> descend, Center heal, re-climb,
                                                                re-bank a full-tank pre-bank
  once per hunt run + 2-fail camp budget / unwired hunt      -> LADDER MODE, logged loud
  descent routing                                            -> leg_home stages, Center door
                                                                (never the harbor/detour)
+ THE DOORSTEP LAW (2026-08-05 LIVE, the turn-around at the bird):
  within DOORSTEP_TILES of the quarry                        -> audit skipped, ENGAGE NOW
  persisted one-shot latch (burned the moment the leg arms)  -> restarts/rewinds never re-arm;
                                                                thin PP engages in LADDER MODE
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
    check("HUNT_ULTRA_TARGET is the healthy stack (50)", C.HUNT_ULTRA_TARGET == 50)
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
        # Skip: always discard while on the hunt's maps. Fail ledger: once-per-key.
        camp25._lap_skipped, camp25._lap_fails = {"moltres"}, {"moltres": 6}
        camp25._lap_prox_fail_clears = set()
        camp25._lap_verdict_logged = None
        check("skipped+failed moltres AT ITS MAP -> unskipped + fail-cleared once",
              camp25._victory_lap_next() == "moltres"
              and "moltres" not in camp25._lap_skipped
              and "moltres" not in camp25._lap_fails
              and "moltres" in camp25._lap_prox_fail_clears)
        # Fail ledger stays burned after the once-clear; skip still drops every visit.
        camp25._lap_skipped, camp25._lap_fails = {"moltres"}, {"moltres": 6}
        check("second proximity visit: skip drops again; fail ledger NOT re-cleared",
              camp25._victory_lap_next() == "moltres"
              and "moltres" not in camp25._lap_skipped
              and camp25._lap_fails.get("moltres") == 6)
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
    # fought/hide set (or mid-script about to flip) inside the savestate, summit empty. The
    # ratchet must reject it by a FRESH post-load (+ settle) read and land on the next older
    # same-region bank that still contains the bird (cross-region banks skipped; all-poisoned
    # restores the live state and SCREAMS when that live state is also spent).
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
        rebanked28, pinned28 = [], []
        camp28._bank_milestone = lambda label: rebanked28.append(label)
        camp28._pin_pre_hunt_promote = lambda key, name=None: pinned28.append(key)
        # the flag reads key off WHICH savestate body is loaded — passing these checks IS
        # the proof the verify reads fresh post-load RAM, not a cached pre-reload value.
        _orig_owns28, _orig_flag28 = C.ram.pokedex_owns, C.fm.read_flag
        C.ram.pokedex_owns = lambda b, sp: False
        C.fm.read_flag = lambda b, f: camp28.b.body in (b"POISONED", b"KANTO", b"LIVE_POISON")
        try:
            check("poisoned 'pre-moltres' rejected -> ratchet lands the older CLEAN sevii bank",
                  camp28._reload_hunt_checkpoint("moltres") is True
                  and camp28.b.body == b"CLEAN")
            check("the kanto bank between them was never accepted (region law held)",
                  camp28.b.body != b"KANTO")
            check("successful clean load re-banks a fresh verified 'pre-moltres'",
                  "pre-moltres" in rebanked28 and pinned28 == ["moltres"])
            # hide-only / fought-only: either bit alone poisons (_hunt_bank_live)
            C.fm.read_flag = lambda b, f: f == 0x052
            check("hide-only poison (0x052) rejected by _hunt_bank_live",
                  camp28._hunt_bank_live("moltres") is False)
            C.fm.read_flag = lambda b, f: f == 0x2BD
            check("fought-only poison (0x2BD) rejected by _hunt_bank_live",
                  camp28._hunt_bank_live("moltres") is False)
            C.fm.read_flag = lambda b, f: False
            check("both flags clear -> _hunt_bank_live accepts (off-quarry / no map decode)",
                  camp28._hunt_bank_live("moltres") is True)
            # MID-SCRIPT class (182452): flags clear at load, settle flips to spent — must
            # NOT fail-closed onto the poisoned LIVE summit when an older clean bank exists.
            _bank28("20260805_182452_mt-ember-summit_8b_pre-moltres", b"MID_SCRIPT")
            _bank28("20260805_182052_mt-ember-summit_8b_pre-moltres", b"CLEAN2", mp=(1, 99))
            camp28.b.body = b"LIVE_POISON"
            rebanked28.clear(); pinned28.clear()

            def _settle_flip(*a, **k):
                if camp28.b.body == b"MID_SCRIPT":
                    camp28.b.body = b"POISONED"

            camp28._wait_overworld = _settle_flip
            C.fm.read_flag = lambda b, f: camp28.b.body in (b"POISONED", b"KANTO",
                                                            b"LIVE_POISON")
            check("mid-script 'pre-moltres' fails settle verify -> older CLEAN2 wins "
                  "(never fail-closed into poisoned live)",
                  camp28._reload_hunt_checkpoint("moltres") is True
                  and camp28.b.body == b"CLEAN2"
                  and "pre-moltres" in rebanked28)
            # all-poisoned: overwrite the clean banks with fought-flag states too
            _bank28("20260805_174000_mt-ember-2f_8b_moltres-leg", b"POISONED", mp=(1, 99))
            _bank28("20260805_182052_mt-ember-summit_8b_pre-moltres", b"POISONED", mp=(1, 99))
            _bank28("20260805_182452_mt-ember-summit_8b_pre-moltres", b"POISONED")
            _bank28("20260805_175000_mt-ember-summit_8b_pre-moltres", b"POISONED")
            camp28.b.body = b"LIVE"
            camp28._wait_overworld = lambda *a, **k: True
            C.fm.read_flag = lambda b, f: camp28.b.body in (b"POISONED", b"KANTO",
                                                            b"LIVE_POISON")
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

    print("== 29. POISONED-BANK LAW: 'pre-<key>' never banked with fought OR hide set ==")
    logs29, banked29 = [], []
    hunt29 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt29.b = object()
    hunt29.log = logs29.append
    hunt29.camp = types.SimpleNamespace(_bank_milestone=lambda label: banked29.append(label),
                                        _lap_fails={})
    _orig_owns29, _orig_flag29 = LS.ram.pokedex_owns, LS.fm.read_flag
    world29 = {"fought": False, "hide": False}
    LS.ram.pokedex_owns = lambda b, sp: False
    LS.fm.read_flag = lambda b, f: ((f == 0x2BD and world29["fought"])
                                    or (f == 0x052 and world29["hide"]))
    try:
        world29["fought"] = True
        check("fought-only set + uncaught -> 'pre-moltres' bank REFUSED, loud",
              hunt29.strike_checkpoint("pre-moltres") is False and banked29 == []
              and any("REFUSED to bank" in l for l in logs29))
        logs29.clear()
        world29["fought"] = False
        world29["hide"] = True
        check("hide-only set + uncaught -> 'pre-moltres' bank REFUSED, loud",
              hunt29.strike_checkpoint("pre-moltres") is False and banked29 == []
              and any("REFUSED to bank" in l for l in logs29))
        check("non-pre milestones still bank under the flag (climb durability untouched)",
              hunt29.strike_checkpoint("moltres-leg") is True
              and banked29 == ["moltres-leg"])
        world29["hide"] = False
        check("flags clear -> 'pre-moltres' banks normally",
              hunt29.strike_checkpoint("pre-moltres") is True
              and banked29 == ["moltres-leg", "pre-moltres"])
    finally:
        LS.ram.pokedex_owns, LS.fm.read_flag = _orig_owns29, _orig_flag29

    print("== 30. THE PP LADDER in _weaken_hp: live PP recheck, rung-2 depth, verdicts ==")
    # (a) the live failure verbatim: Bite at 1 PP fires once; the re-pick next swing skips
    # the dry Bite, Ice Beam (est ~106% vs Moltres) can faint -> honest 'guard' stop.
    logs30 = []
    ag30 = make_agent([(2, 6)], hunt_pending=True, logs=logs30)
    moves30 = [{"id": 44, "name": "Bite", "type": "dark", "power": 60, "pp": 1},
               {"id": 58, "name": "Ice Beam", "type": "ice", "power": 95, "pp": 10}]
    foe30 = {"hp": 160, "maxhp": 160}
    fired30 = []
    _orig_rb30, _orig_ib30 = BA.st.read_battle, BA.st.in_battle
    BA.st.read_battle = lambda b: {"ours": {"moves": [dict(m) for m in moves30], "level": 63,
                                            "types": ["water", "water"]},
                                   "enemy": {"hp": foe30["hp"], "maxhp": foe30["maxhp"],
                                             "level": 50, "types": ["fire", "flying"]}}
    BA.st.in_battle = lambda b: True

    def _fire30(i):
        fired30.append(moves30[i]["name"])
        moves30[i]["pp"] = max(0, moves30[i]["pp"] - 1)
        foe30["hp"] = max(1, foe30["hp"] - 100)
    ag30._fire_move = _fire30
    try:
        v30 = ag30._weaken_hp(target_frac=BA.CATCH_CHIP_TARGET_LEGEND,
                              max_hits=BA.LEGEND_CHIP_HITS, legend=True)
        check("Bite fires once, the DRY re-pick never fires it again",
              fired30 == ["Bite"], f"fired={fired30}")
        check("overkill Ice Beam refused even on the ladder -> honest 'guard' verdict",
              v30 == "guard", f"verdict={v30}")
        # (b) rung 2: over the polite margin but CANNOT faint from current HP -> swings
        moves30b = [{"id": 33, "name": "Slam", "type": "normal", "power": 114, "pp": 20}]
        foe30b, fired30b = {"hp": 100, "maxhp": 100}, []
        # water attacker, normal move (NO STAB): est = (114/50)*0.35 ~ 80% — over the 70%
        # safety margin yet clearly unable to faint from 100% HP.
        BA.st.read_battle = lambda b: {"ours": {"moves": [dict(m) for m in moves30b],
                                                "level": 50, "types": ["water"]},
                                       "enemy": {"hp": foe30b["hp"], "maxhp": 100, "level": 50,
                                                 "types": ["normal"]}}
        ag30._fire_move = lambda i: (fired30b.append(i), foe30b.__setitem__("hp", 20))
        v30b = ag30._weaken_hp(target_frac=0.15, max_hits=4, legend=True)
        check("rung 2: est ~80% at 100% HP (unsafe by margin, can't faint) -> SWINGS, "
              "then honest guard at 20%", fired30b == [0] and v30b == "guard",
              f"fired={fired30b} verdict={v30b}")
        check("the ladder swing is LOUD", any("chip LADDER" in l for l in logs30))
        # (c) generic path unchanged: same setup, legend=False -> refuses the unsafe swing
        foe30b["hp"], fired30b[:] = 100, []
        v30c = ag30._weaken_hp(target_frac=0.30, max_hits=4, legend=False)
        check("generic (legend=False) still refuses the over-margin swing",
              fired30b == [] and v30c == "guard", f"fired={fired30b} verdict={v30c}")
        # (d) verdicts: no damaging PP anywhere on the active / already in the band
        BA.st.read_battle = lambda b: {"ours": {"moves": [{"id": 33, "name": "Tackle",
                                                           "type": "normal", "power": 35,
                                                           "pp": 0}], "level": 50,
                                                "types": ["normal"]},
                                       "enemy": {"hp": 80, "maxhp": 100, "level": 50,
                                                 "types": ["normal"]}}
        check("active PP dry -> 'no_pp' (the switch signal)",
              ag30._weaken_hp(target_frac=0.15, max_hits=4, legend=True) == "no_pp")
        BA.st.read_battle = lambda b: {"ours": {"moves": [], "level": 50, "types": ["normal"]},
                                       "enemy": {"hp": 10, "maxhp": 100, "level": 50,
                                                 "types": ["normal"]}}
        check("foe already in the band -> 'band'",
              ag30._weaken_hp(target_frac=0.15, max_hits=4, legend=True) == "band")
    finally:
        BA.st.read_battle, BA.st.in_battle = _orig_rb30, _orig_ib30

    print("== 31. ACE-ONLY legendary chip + HARD THROW FLOOR (_legend_chip_ladder) ==")
    logs31 = []
    ag31 = make_agent([(2, 6)], hunt_pending=True, logs=logs31)
    world31 = {"foe_frac": 0.60}
    _orig_rb31, _orig_ib31 = BA.st.read_battle, BA.st.in_battle
    BA.st.read_battle = lambda b: {"ours": {"moves": [], "species": 9, "level": 63,
                                            "hp": 100, "maxhp": 100},
                                   "enemy": {"hp": int(world31["foe_frac"] * 100),
                                             "maxhp": 100, "level": 50,
                                             "types": ["fire", "flying"]}}
    BA.st.in_battle = lambda b: True
    switched31 = []
    ag31._catch_chipper_slot = lambda *a, **k: 2
    ag31._switch_to_slot = lambda s, sp: (switched31.append(s), "switched")[1]
    try:
        # above hard floor -> ACE stays in, NO bench switch, refuse throw
        tried31, refuse31 = ag31._legend_chip_ladder(
            BA.CATCH_CHIP_TARGET_LEGEND, BA.LEGEND_CHIP_HITS, set())
        check("above half -> ACE-ONLY (no Fearow/Spearow switch), refuse throw",
              refuse31 is True and switched31 == []
              and any("ACE-ONLY" in l for l in logs31)
              and any("HARD FLOOR" in l or "REFUSING" in l for l in logs31))
        # refuse path must NEVER flee — deepen + KEEP FIGHTING while balls remain
        # (LIVE 2026-08-06: soft-reload mid-fight while she still had a catch chance).
        fled31, reloads31 = [], []
        ag31.flee = lambda **k: fled31.append("fled") or "fled"
        ag31._weaken_hp = lambda **k: "guard"
        ag31._legend_ether_for_chip = lambda: False
        ag31._spendable_for_pref = lambda pref: 6
        ag31._try_legend_soft_reload = lambda sp=None: (reloads31.append(1), False)[1]
        world31["foe_frac"] = 0.75
        out31 = ag31._legend_refuse_throw(0.75, "HARD FLOOR test")
        check("hard-floor refuse deepens + keep_chipping — NEVER flee / mid-fight reload",
              out31 == "keep_chipping" and fled31 == []
              and any("NEVER fleeing" in l for l in logs31)
              and any("KEEP FIGHTING" in l or "keep_chipping" in l for l in logs31))
        # Soft-reload gate: balls + live foe → REFUSED
        logs31.clear(); reloads31.clear()
        ag31r = make_agent([(2, 6)], hunt_pending=True, logs=logs31)
        ag31r._spendable_for_pref = lambda pref: 5
        BA.st.in_battle = lambda b: True
        BA.st.read_battle = lambda b: {"enemy": {"hp": 80, "maxhp": 100, "species": 146},
                                      "ours": {"hp": 100, "maxhp": 100}}
        check("soft-reload REFUSED while Ultras remain + bird alive",
              ag31r._try_legend_soft_reload(146) is False
              and any("SOFT-RELOAD REFUSED" in l for l in logs31))
        logs31.clear()
        ag31r._spendable_for_pref = lambda pref: 0
        hooked = []
        BA.LEGEND_SOFT_RELOAD = lambda key: (hooked.append(key), True)[1]
        try:
            check("soft-reload ALLOWED at 0 Ultras (catch lost)",
                  ag31r._try_legend_soft_reload(146) is True
                  and hooked == ["moltres"]
                  and any("LEGEND SOFT-RELOAD" in l and "REFUSED" not in l for l in logs31))
        finally:
            BA.LEGEND_SOFT_RELOAD = None
        # Restore the section-31 battle stub (gate tests overwrote read_battle).
        BA.st.read_battle = lambda b: {"ours": {"moves": [], "species": 9, "level": 63,
                                                "hp": 100, "maxhp": 100},
                                       "enemy": {"hp": int(world31["foe_frac"] * 100),
                                                 "maxhp": 100, "level": 50,
                                                 "types": ["fire", "flying"]}}
        BA.st.in_battle = lambda b: True
        # Ether rail: refuse tries Ether BEFORE deepen when Bite is dry
        logs31.clear()
        ether31 = []
        ag31._legend_ether_tried = False
        ag31._legend_ether_for_chip = lambda: (ether31.append(1), False)[1]
        ag31._legend_refuse_throw(0.80, "OVERKILL / empty Bite")
        check("hard-floor refuse hits Ether rail before deepen/keep-fighting",
              ether31 == [1])
        # between red band and hard floor -> sanctioned throw, still no switch
        logs31.clear(); switched31.clear()
        world31["foe_frac"] = 0.40
        tried31c, refuse31c = ag31._legend_chip_ladder(
            BA.CATCH_CHIP_TARGET_LEGEND, BA.LEGEND_CHIP_HITS, set())
        check("foe at 40% -> sanctioned throw, still no bench switch",
              refuse31c is False and switched31 == []
              and any("SANCTIONED throw" in l for l in logs31))
        # already in the band -> quiet
        logs31.clear(); switched31.clear()
        world31["foe_frac"] = 0.12
        _, refuse31d = ag31._legend_chip_ladder(
            BA.CATCH_CHIP_TARGET_LEGEND, BA.LEGEND_CHIP_HITS, set())
        check("in the red band -> no switch, no refuse",
              refuse31d is False and switched31 == []
              and not any("REFUSING" in l for l in logs31))
        check("legend_throw_allowed: 50% OK, 51% blocked",
              BA.legend_throw_allowed(0.50) is True
              and BA.legend_throw_allowed(0.51) is False)
        # _legend_ether_for_chip itself: bag Ether + post-use safe chip -> True once
        logs31.clear()
        ag31e = make_agent([(2, 6)], hunt_pending=True, logs=logs31)
        ag31e._legend_ether_tried = False
        ag31e._items_count = lambda i: 1 if i == 34 else 0
        used31 = []
        ag31e.use_item_in_battle = lambda item, **k: (used31.append(item), "used")[1]
        BA.st.read_battle = lambda b: {
            "ours": {"moves": [{"id": 44, "name": "Bite", "type": "dark",
                                "power": 60, "pp": 10}],
                     "level": 64, "types": ["water"], "species": 9,
                     "hp": 188, "maxhp": 196},
            "enemy": {"hp": 150, "maxhp": 150, "level": 50,
                      "types": ["fire", "flying"], "species": 146}}
        check("legend Ether restores Bite -> safe chip True (once)",
              ag31e._legend_ether_for_chip() is True
              and used31 == [34]
              and any("LEGEND ETHER" in l for l in logs31))
        check("legend Ether latches — second call is a no-op",
              ag31e._legend_ether_for_chip() is False)
    finally:
        BA.st.read_battle, BA.st.in_battle = _orig_rb31, _orig_ib31

    print("== 32. PRE-ENCOUNTER CHIP-PP AUDIT: safe-swing scoring vs the quarry ==")
    class AuditBridge:
        """rd8 answers only the plaintext level offset reads: slot0 L63, slot1 L25."""
        def rd8(self, a):
            s = (a - LS.ram.GPLAYER_PARTY - 0x54) // LS.pst.PARTY_MON_SIZE
            return {0: 63, 1: 25}.get(s, 0)
    hunt32 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt32.b = AuditBridge()
    logs32 = []
    hunt32.log = logs32.append
    hunt32.camp = types.SimpleNamespace(party_health=lambda: [(0, 193, 193, 1.0),
                                                              (1, 60, 130, 0.46),
                                                              (2, 0, 90, 0.0)])
    _orig_rpm32 = LS.pst.read_party_moves
    _orig_rpp32 = LS.pst.read_party_pp
    _orig_mif32 = LS.pst.move_info_full
    _MOVES32 = {57: ("water", 95, 100), 58: ("ice", 95, 100), 44: ("dark", 60, 100),
                55: ("water", 40, 100), 47: ("normal", 0, 55)}   # Sing = status (power 0)
    # the live picture: Blastoise's Bite at 1 PP (Surf/Ice Beam overkill vs Moltres);
    # Lapras' Water Gun soft and full. Fainted slot 2 must never be audited.
    LS.pst.read_party_moves = lambda b, s: ([57, 58, 44, 0] if s == 0 else [55, 47, 0, 0])
    LS.pst.read_party_pp = lambda b, s: ([15, 10, 1, 0] if s == 0 else [25, 15, 0, 0])
    LS.pst.move_info_full = lambda b, m: _MOVES32.get(m, ("normal", 0, 100))
    try:
        a32 = hunt32._chip_pp_audit()
        check("audit scores SAFE swings vs Moltres: ace 1 (the lone Bite), party 26",
              a32 == (True, 1, 26), f"got {a32}")
        check("the audit line is LOUD and names the quarry",
              any("CHIP-PP AUDIT" in l and "Moltres" in l for l in logs32), f"logs={logs32[:1]}")
        check("fainted slot 2 never audited (only the healthy bench counts)",
              not any("slot2" in l for l in logs32))
        logs32.clear()
        LS.pst.read_party_pp = lambda b, s: [0, 0, 0, 0]
        check("ZERO damaging PP party-wide -> (False, 0, 0) and SCREAMS (sleep+throw only)",
              hunt32._chip_pp_audit() == (False, 0, 0)
              and any("ZERO damaging PP" in l for l in logs32))
    finally:
        LS.pst.read_party_moves = _orig_rpm32
        LS.pst.read_party_pp = _orig_rpp32
        LS.pst.move_info_full = _orig_mif32

    print("== 33. THE PP RESTORE LEG: threshold, arming, budget, descent routing ==")
    burned33, unlatched33 = [], []
    def _mk_camp33(latched=False, **kw):
        state = {"latched": bool(latched)}
        return types.SimpleNamespace(
            on_event=lambda *a, **k: None,
            pp_restore_latched=lambda k: state["latched"],
            pp_restore_latch=lambda k: (burned33.append(k), state.__setitem__("latched", True)),
            pp_restore_unlatch=lambda k: (unlatched33.append(k),
                                         state.__setitem__("latched", False)),
            **kw)
    def _mk_hunt33(cls, audit, camp=None):
        h = cls.__new__(cls)
        h.b = object()
        h.log = logs33.append
        h.camp = camp or _mk_camp33()
        h._chip_pp_audit = lambda: audit
        return h
    logs33 = []
    # (a) thin ace (1 safe Bite) -> ARMED once; the second ask falls to LADDER MODE
    h33 = _mk_hunt33(LS.MoltresHunt, (True, 1, 26))
    check("ace below 5 safe swings -> RESTORE LEG ARMED (mode latched)",
          h33._maybe_arm_pp_restore() is True and h33._pp_restore_mode is True
          and any("PP RESTORE LEG ARMED" in l for l in logs33))
    check("arming BURNS the persisted campaign latch immediately (armed = consumed)",
          burned33 == ["moltres"], f"burned={burned33}")
    logs33.clear()
    check("second ask this run -> LADDER MODE (once per hunt instance)",
          h33._maybe_arm_pp_restore() is False
          and any("LADDER MODE" in l for l in logs33))
    # (b) healthy tank -> never armed
    logs33.clear()
    h33b = _mk_hunt33(LS.MoltresHunt, (True, 25, 60))
    check("full tank (Bite 25) -> no restore, no ladder chatter",
          h33b._maybe_arm_pp_restore() is False and not logs33)
    # (c) party-wide famine triggers even with a mid ace
    h33c = _mk_hunt33(LS.MoltresHunt, (True, 6, 7))
    check("party-wide < 8 safe swings -> armed (the OR trigger)",
          h33c._maybe_arm_pp_restore() is True)
    # (d) unwired hunt (Zapdos has no Center leg) -> LOUD ladder, never a trip
    logs33.clear()
    h33d = _mk_hunt33(LS.ZapdosHunt, (True, 1, 3))
    check("no Center leg wired -> LADDER MODE loudly, mode never set",
          h33d._maybe_arm_pp_restore() is False
          and not getattr(h33d, "_pp_restore_mode", False)
          and any("no Center leg is wired" in l for l in logs33))
    # (e) camp budget: 2 failed trips -> refuse further restore attempts
    logs33.clear()
    camp33 = types.SimpleNamespace(on_event=lambda *a, **k: None,
                                   _pp_restore_fails={"moltres": 2})
    h33e = _mk_hunt33(LS.MoltresHunt, (True, 1, 3), camp=camp33)
    check("restore budget spent (2 fails) -> LADDER MODE, no third trip",
          h33e._maybe_arm_pp_restore() is False
          and any("LADDER MODE" in l for l in logs33))
    # (f) _pp_restore_fail counts the bounded fail and drops the mode
    logs33.clear()
    h33f = _mk_hunt33(LS.MoltresHunt, (True, 1, 3))
    h33f._pp_restore_mode = True
    h33f._pp_restore_fail("descent wedged on (1, 99)")
    check("a wedged descent counts 1/2 and clears the mode (fallback logged)",
          h33f._pp_restore_mode is False
          and h33f.camp._pp_restore_fails == {"moltres": 1}
          and any("falling back to LADDER MODE" in l for l in logs33))
    # (g) descent routing: volcano maps delegate to leg_home; One Island re-keys to the PC
    h33g = _mk_hunt33(LS.MoltresHunt, (True, 1, 3))
    routed33 = []
    h33g.leg_home = lambda here: (routed33.append(("home", here)), True)[1]
    h33g.enter_step = lambda tile, dest, label: (routed33.append((label, dest)), True)[1]
    check("EMBER_2F descends via the leg_home stage",
          h33g.leg_to_center((1, 99)) is True and routed33 == [("home", (1, 99))])
    routed33.clear()
    check("ONE_ISLAND routes to the CENTER door, never the harbor",
          h33g.leg_to_center((3, 12)) is True and routed33 == [("one-pc", (32, 0))])
    check("an off-leg map declines (run()'s router counts the bounded fail)",
          h33g.leg_to_center((3, 13)) is False)

    print("== 34. THE DOORSTEP LAW + persisted latch (the turn-around at the bird) ==")
    # (a) burned latch -> thin PP still ENGAGES (no second trip, ever, across restarts)
    logs33.clear()
    h34 = _mk_hunt33(LS.MoltresHunt, (True, 1, 3), camp=_mk_camp33(latched=True))
    check("latch already burned this campaign -> ENGAGING NOW, no retreat",
          h34._maybe_arm_pp_restore() is False
          and not getattr(h34, "_pp_restore_mode", False)
          and any("ALREADY BURNED" in l and "ENGAGING NOW" in l for l in logs33))
    # empty ace tank overrides burned latch (soak 082259 Skull-Bash-only freeze)
    logs33.clear(); unlatched33.clear(); burned33.clear()
    h34empty = _mk_hunt33(LS.MoltresHunt, (True, 0, 29), camp=_mk_camp33(latched=True))
    _armed_empty = h34empty._maybe_arm_pp_restore()
    check("burned latch + EMPTY ace tank -> unlatch + Center armed (LOUD)",
          _armed_empty is True
          and unlatched33 == ["moltres"]
          and any("EMPTY ACE TANK" in l for l in logs33)
          and any("PP RESTORE LEG ARMED" in l for l in logs33))
    # (b) standing at the bird with a usable chip tank -> engage (no Center detour).
    # Ace-empty tanks are a SEPARATE law (b3) — they must Center first.
    _orig_coords34 = LS.tv.coords
    try:
        LS.tv.coords = lambda b: (9, 7)          # 1 tile from Moltres' (9, 6)
        logs33.clear()
        h34b = _mk_hunt33(LS.MoltresHunt, (True, 5, 10))
        h34b._chip_pp_audit = lambda: (True, 5, 10)   # usable Bite tank
        h34b._maybe_arm_pp_restore = lambda: (_ for _ in ()).throw(AssertionError(
            "restore must NOT arm at the doorstep when the ace still has safe chips"))
        check("adjacent to the quarry with chip PP -> ENGAGING NOW (LOUD)",
              h34b._doorstep_or_restore() is False
              and any("AT THE DOORSTEP" in l and "ENGAGING NOW" in l for l in logs33))
        # (b2) doorstep law is ABSOLUTE — free-retry never arms a Center retreat from the bird
        logs33.clear()
        h34b2 = _mk_hunt33(LS.MoltresHunt, (True, 5, 10))
        h34b2._catch_retries = 1
        h34b2._chip_pp_audit = lambda: (True, 5, 10)
        h34b2._maybe_arm_pp_restore = lambda: True
        check("doorstep AFTER free-retry -> still ENGAGE (no mountain retreat)",
              h34b2._doorstep_or_restore() is False
              and any("ENGAGING NOW" in l and "absolute" in l for l in logs33))
        # (b3) empty ace at doorstep, no Ether -> soft-reload in place, NEVER Center-retreat
        logs33.clear()
        reloads34 = []
        h34b3 = _mk_hunt33(LS.MoltresHunt, (True, 0, 0))
        h34b3._catch_retries = 0
        h34b3._chip_pp_audit = lambda: (True, 0, 29)
        h34b3._field_ether_ace = lambda: False
        h34b3.camp = types.SimpleNamespace(
            _reload_hunt_checkpoint=lambda key: (reloads34.append(key), True)[1])
        check("doorstep + empty ace (no Ether) -> soft-reload in place, ENGAGE (no retreat)",
              h34b3._doorstep_or_restore() is False
              and reloads34 == ["moltres"]
              and any("NO mountain retreat" in l for l in logs33)
              and any("ENGAGING NOW" in l for l in logs33))
        # (b4) empty ace + Ether restores Bite -> engage WITHOUT soft-reload
        logs33.clear()
        reloads34b4 = []
        _ether_calls = []
        h34b4 = _mk_hunt33(LS.MoltresHunt, (True, 0, 0))
        h34b4._catch_retries = 0
        _audit_n = {"n": 0}
        def _audit_b4():
            _audit_n["n"] += 1
            # first audit empty; after Ether, safe=5
            return (True, 0, 29) if _audit_n["n"] == 1 else (True, 5, 34)
        h34b4._chip_pp_audit = _audit_b4
        h34b4._field_ether_ace = lambda: (_ether_calls.append(1), True)[1]
        h34b4.camp = types.SimpleNamespace(
            _reload_hunt_checkpoint=lambda key: (reloads34b4.append(key), True)[1])
        check("doorstep + empty ace + Ether restores Bite -> ENGAGE, no soft-reload",
              h34b4._doorstep_or_restore() is False
              and _ether_calls == [1]
              and reloads34b4 == []
              and any("Ether rail first" in l for l in logs33)
              and any("ENGAGING NOW" in l for l in logs33))
        # (c) far from the bird (pre-approach) -> the gate delegates to the one-shot audit
        LS.tv.coords = lambda b: (29, 40)        # 54 tiles out — mid-climb
        h34c = _mk_hunt33(LS.MoltresHunt, (True, 1, 3))
        h34c._maybe_arm_pp_restore = lambda: True
        check("far from the quarry -> pre-approach audit still owns the decision",
              h34c._doorstep_or_restore() is True)
    finally:
        LS.tv.coords = _orig_coords34
    # (d) the latch sidecar round-trips on disk (survives restarts; rewinds never touch it)
    import tempfile as _tf
    _orig_ppjson34 = C.PP_RESTORE_JSON
    try:
        C.PP_RESTORE_JSON = os.path.join(_tf.mkdtemp(), "pp_restore_latch.json")
        camp34 = C.Campaign.__new__(C.Campaign)
        check("fresh campaign -> latch clear (an honest first trip is allowed)",
              C.Campaign.pp_restore_latched(camp34, "moltres") is False)
        C.Campaign.pp_restore_latch(camp34, "Moltres")
        check("burned latch persists on disk (case-folded key) and reads back True",
              C.Campaign.pp_restore_latched(camp34, "moltres") is True
              and C.Campaign.pp_restore_latched(camp34, "zapdos") is False)
    finally:
        C.PP_RESTORE_JSON = _orig_ppjson34

    print("== 35. ULTRA WAR-CHEST: thin Ultras arm Three Island Mart restock ==")
    logs35 = []
    h35 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35.b = object()
    h35.log = logs35.append
    h35.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        _balls_pocket_count=lambda i: 6 if i == 2 else 0,
        _ball_restock_fails={})
    h35.BALL_RESTOCK_WIRED = True
    h35._ultra_target = lambda: 50
    h35._ultra_min_engage = lambda: 20
    check("6 Ultras < target 50 -> WAR-CHEST ARMED",
          h35._maybe_arm_ball_restock() is True
          and h35._ball_restock_mode is True
          and any("ULTRA WAR-CHEST ARMED" in l for l in logs35))
    logs35.clear()
    h35b = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35b.b = object()
    h35b.log = logs35.append
    h35b.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        _balls_pocket_count=lambda i: 50 if i == 2 else 0,
        _ball_restock_fails={})
    h35b.BALL_RESTOCK_WIRED = True
    h35b._ultra_target = lambda: 50
    h35b._ultra_min_engage = lambda: 20
    check("50 Ultras already stacked -> no restock arm",
          h35b._maybe_arm_ball_restock() is False
          and any("at target" in l for l in logs35))
    logs35.clear()
    h35c = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35c.b = object()
    h35c.log = logs35.append
    h35c.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        _balls_pocket_count=lambda i: 25 if i == 2 else 0,
        _ball_restock_fails={})
    h35c.BALL_RESTOCK_WIRED = True
    h35c._ball_restock_done = True
    h35c._ultra_target = lambda: 50
    h35c._ultra_min_engage = lambda: 20
    check("war-chest bought (>=floor) this run -> engage even if under TARGET",
          h35c._maybe_arm_ball_restock() is False
          and any("already filled" in l for l in logs35))
    logs35.clear()
    h35d = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35d.b = object()
    h35d.log = logs35.append
    h35d.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        _balls_pocket_count=lambda i: 6 if i == 2 else 0,
        _ball_restock_fails={"moltres": 2})
    h35d.BALL_RESTOCK_WIRED = True
    h35d._ball_restock_done = True
    h35d._ultra_target = lambda: 50
    h35d._ultra_min_engage = lambda: 20
    check("soft-reload collapsed to 6 Ultras -> RE-ARM even if done/fails spent",
          h35d._maybe_arm_ball_restock() is True
          and any("COLLAPSED" in l or "RE-ARMING" in l for l in logs35))
    check("Three Island Mart is in MART_STOCK with Ultra Ball row 0",
          C.MART_STOCK.get((3, 14), [None])[0] == 2
          and C.CITY_MART_DOORS.get((3, 14)) == (18, 12))
    check("Moltres hunt wires the ball-restock leg",
          LS.MoltresHunt.BALL_RESTOCK_WIRED is True)
    check("HUNT_ULTRA_MIN_ENGAGE is the hard floor (20)", C.HUNT_ULTRA_MIN_ENGAGE == 20)
    check("old 6-ball 182052 preferred pin is CLEARED (war-chest banks win)",
          C.Campaign._HUNT_PREFERRED_PRE.get("moltres") == ())
    check("Ember EXT (14,25) is the 1F mouth (not upper 3F ledge)",
          LS.MoltresHunt._ember_ext_1f_mouth((14, 25)) is True
          and LS.MoltresHunt._ember_ext_1f_mouth((29, 7)) is False
          and LS.MoltresHunt._ember_ext_1f_mouth((39, 19)) is False)
    logs35.clear()
    h35e = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35e.b = object()
    h35e.log = logs35.append
    h35e.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        _balls_pocket_count=lambda i: 5 if i == 2 else 0,
        _ball_restock_fails={})
    h35e.BALL_RESTOCK_WIRED = True
    h35e._ultra_target = lambda: 50
    h35e._ultra_min_engage = lambda: 20
    h35e._ball_restock_mode = True
    h35e._ball_restock_fail("descent wedged on (1, 97) @ (14, 25)")
    check("war-chest fail below floor STAYS on ferry (no 'engaging with 5')",
          h35e._ball_restock_mode is True
          and any("STAYING on the ferry" in l for l in logs35)
          and not any("engaging with 5" in l for l in logs35))

    # Broke-wallet terminal: buy got 0 + can't afford Ultra → DONE, not fail-stay loop.
    logs35.clear()
    h35broke = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h35broke.b = object()
    h35broke.log = logs35.append
    h35broke.QUARRY = {"name": "Moltres"}
    h35broke.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        money=lambda: 200,
        buy_at_mart=lambda door, want: {},
        _balls_pocket_count=lambda i: 8 if i == 2 else 0,
        _ball_restock_fails={})
    h35broke.BALL_RESTOCK_WIRED = True
    h35broke._ultra_target = lambda: 50
    h35broke._ultra_min_engage = lambda: 20
    h35broke._three_bikers_cleared = lambda: True
    ok_broke = h35broke.buy_ultra_war_chest()
    check("broke wallet + bought 0 -> True (leave Mart), not False fail-stay",
          ok_broke is True
          and h35broke._ball_restock_done is True
          and h35broke._ball_restock_broke is True
          and any("WALLET EMPTY" in l for l in logs35))
    logs35.clear()
    # Broke latch must block re-arm even when pocket is under the engage floor.
    check("broke latch blocks thin-pocket ferry re-arm",
          h35broke._maybe_arm_ball_restock() is False
          and any("wallet emptied" in l for l in logs35))

    print("== 36. WAR-CHEST FERRY: Kindle Traveler west + catch_now truce + early arm ==")
    import time as _time
    _ba_restock0 = BA.BALL_RESTOCK_MODE
    try:
        logs36 = []
        released = []
        h36 = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h36.b = object()
        h36.log = logs36.append
        h36.QUARRY = {"name": "Moltres"}
        h36.camp = types.SimpleNamespace(
            on_event=lambda *a, **k: None,
            _balls_pocket_count=lambda i: 5 if i == 2 else 0,
            _ball_restock_fails={},
            _release_catch_order_for_restock=lambda: released.append(True))
        h36.BALL_RESTOCK_WIRED = True
        h36._ultra_target = lambda: 50
        h36._ultra_min_engage = lambda: 20
        check("arming war-chest mirrors BALL_RESTOCK_MODE + releases catch_now",
              h36._maybe_arm_ball_restock() is True
              and h36._ball_restock_mode is True
              and getattr(h36.camp, "_ball_restock_mode", False) is True
              and BA.BALL_RESTOCK_MODE is True
              and released == [True])
        h36._set_ball_restock_mode(False)
        check("clearing restock mode clears BALL_RESTOCK_MODE",
              BA.BALL_RESTOCK_MODE is False
              and h36._ball_restock_mode is False)

        # Kindle west prefers Traveler (live wedge: strike cross_edge @ (0,127)).
        maps = {"here": LS.KINDLE}
        crossed = []

        class _FakeTrav:
            def travel(self, **kw):
                crossed.append(kw)
                maps["here"] = LS.ONE_ISLAND
                return "arrived"

        h36k = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h36k.b = types.SimpleNamespace(run_frame=lambda: None)
        h36k.log = logs36.append
        h36k.camp = types.SimpleNamespace(trav=_FakeTrav())
        h36k.cross_edge = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cross_edge must not be sole path when Traveler arrives"))
        _orig_map = LS.tv.map_id
        _orig_coords = LS.tv.coords
        try:
            LS.tv.map_id = lambda _b: maps["here"]
            LS.tv.coords = lambda _b: (23, 12)
            ok = h36k._kindle_west_to_one()
            check("Kindle west uses Traveler edge=west -> One Island",
                  ok is True
                  and maps["here"] == LS.ONE_ISLAND
                  and crossed
                  and crossed[0].get("edge") == "west"
                  and crossed[0].get("target_map") == LS.ONE_ISLAND)
            # leg_home KINDLE: Traveler success short-circuits (no cross_edge).
            maps["here"] = LS.KINDLE
            crossed.clear()
            h36k.meteorite_in_bag = lambda: False
            _orig_rf = LS.fm.read_flag
            LS.fm.read_flag = lambda *a, **k: False
            try:
                check("leg_home(KINDLE) prefers Traveler over cross_edge",
                      h36k.leg_home(LS.KINDLE) is True
                      and maps["here"] == LS.ONE_ISLAND)
            finally:
                LS.fm.read_flag = _orig_rf
        finally:
            LS.tv.map_id = _orig_map
            LS.tv.coords = _orig_coords

        # Quiet catch_now release for restock (no 'caught what you told me' voice).
        _td = tempfile.mkdtemp(prefix="kira_restock_order_")
        _opath = os.path.join(_td, "creator_order.json")
        with open(_opath, "w", encoding="utf-8") as f:
            json.dump({"order": "catch_now", "ts": _time.time()}, f)
        _orig_co = C.CREATOR_ORDER_JSON
        voiced = []
        try:
            C.CREATOR_ORDER_JSON = _opath
            camp36 = C.Campaign.__new__(C.Campaign)
            camp36.on_event = lambda *a, **k: voiced.append(a[0] if a else "")
            camp36._release_catch_order_for_restock()
            check("restock release drops catch_now without catch-fulfill voice",
                  not os.path.exists(_opath)
                  and not any("caught what you told me" in (v or "") for v in voiced))
        finally:
            C.CREATOR_ORDER_JSON = _orig_co
            try:
                import shutil
                shutil.rmtree(_td, ignore_errors=True)
            except Exception:
                pass

        # Early-arm gate: thin pocket at hunt entry arms (same predicate as press_quarry).
        logs36.clear()
        h36e = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h36e.b = object()
        h36e.log = logs36.append
        h36e.QUARRY = {"name": "Moltres"}
        h36e.camp = types.SimpleNamespace(
            on_event=lambda *a, **k: None,
            _balls_pocket_count=lambda i: 5 if i == 2 else 0,
            _ball_restock_fails={},
            _release_catch_order_for_restock=lambda: None)
        h36e.BALL_RESTOCK_WIRED = True
        h36e._ultra_target = lambda: 50
        h36e._ultra_min_engage = lambda: 20
        h36e._ball_restock_mode = False
        # Mirror run()'s entry seam: arm before climb when thin.
        if not getattr(h36e, "_ball_restock_mode", False):
            _early = h36e._maybe_arm_ball_restock()
        else:
            _early = False
        check("hunt-entry early arm when Ultras < floor",
              _early is True and h36e._ball_restock_mode is True)

        check("sail row candidates: Three from One = pre=1 and post-Vermilion=2",
              LS.MoltresHunt._sail_row_candidates(LS.ONE_HARBOR, LS.THREE_HARBOR)
              == [1, 2])
        check("sail row candidates: Three from Two = pre=1 and post-Vermilion=2",
              LS.MoltresHunt._sail_row_candidates(LS.TWO_HARBOR, LS.THREE_HARBOR)
              == [1, 2])
        # TWO Harbor is on the war-chest rails (flaked menu lands here).
        # Armed/cleared → sail Three; unarmed → enter Two Island to prime Game Corner.
        sailed = []
        entered = []
        h36t = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h36t.b = object()
        h36t.log = logs36.append
        h36t.sail = lambda want: (sailed.append(want), True)[1]
        h36t.enter_step = lambda tile, dest, label: (entered.append((tile, dest, label)), True)[1]
        h36t._three_gauntlet_armed = lambda: True
        h36t._three_bikers_cleared = lambda: True
        check("leg_to_ball_mart(TWO_HARBOR) armed -> sails to Three",
              h36t.leg_to_ball_mart(LS.TWO_HARBOR) is True
              and sailed == [LS.THREE_HARBOR])
        sailed.clear(); entered.clear()
        h36t._three_gauntlet_armed = lambda: False
        h36t._three_bikers_cleared = lambda: False
        check("leg_to_ball_mart(TWO_HARBOR) unarmed -> enter Two Island to prime",
              h36t.leg_to_ball_mart(LS.TWO_HARBOR) is True
              and entered and entered[0][1] == LS.TWO_ISLAND
              and sailed == [])
        # Arming restock clears the moltres questline try budget.
        h36f = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h36f.b = object()
        h36f.log = logs36.append
        h36f.QUARRY = {"name": "Moltres"}
        h36f.camp = types.SimpleNamespace(
            on_event=lambda *a, **k: None,
            _balls_pocket_count=lambda i: 5 if i == 2 else 0,
            _ball_restock_fails={},
            _release_catch_order_for_restock=lambda: None,
            _ql_strike_tries_map={("flag", "FLAG_FOUGHT_MOLTRES"): 3})
        h36f.BALL_RESTOCK_WIRED = True
        h36f._ultra_target = lambda: 50
        h36f._ultra_min_engage = lambda: 20
        h36f._maybe_arm_ball_restock()
        check("arming war-chest resets moltres questline strike tries",
              h36f.camp._ql_strike_tries_map[("flag", "FLAG_FOUGHT_MOLTRES")] == 0)
    finally:
        BA.BALL_RESTOCK_MODE = _ba_restock0

    # 37. SAILOR NORTH STAND + HARBOR UNSKIP + SEVII RE-ARM (2026-08-06 live):
    #     At One Harbor (8,3) north of the sailor, BFS to (8,7) has no path — stand
    #     at (8,5)+DOWN instead. Honest skip + articuno next while Sevii-stranded
    #     with bird unspent must RE-ARM moltres (not ride-home no-op forever).
    print("== 37. SAILOR NORTH STAND + HARBOR UNSKIP + SEVII RE-ARM ==")
    h37 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h37.b = object()
    _orig_tv37 = LS.tv.map_id
    _orig_co37 = LS.tv.coords
    try:
        LS.tv.map_id = lambda _b: LS.ONE_HARBOR
        LS.tv.coords = lambda _b: (8, 3)
        stand, face = h37._sailor_stand_and_face()
        check("north of sailor -> stand (8,5) face DOWN",
              stand == LS.SAILOR_STAND_NORTH and face == "DOWN")
        LS.tv.coords = lambda _b: (8, 8)
        stand2, face2 = h37._sailor_stand_and_face()
        check("south of sailor -> stand (8,7) face UP",
              stand2 == LS.SAILOR_STAND_SOUTH and face2 == "UP")
        LS.tv.coords = lambda _b: (8, 6)
        stand3, face3 = h37._sailor_stand_and_face()
        check("on sailor tile y=6 -> south stand + UP (default)",
              stand3 == LS.SAILOR_STAND_SOUTH and face3 == "UP")
    finally:
        LS.tv.map_id = _orig_tv37
        LS.tv.coords = _orig_co37

    camp37 = C.Campaign.__new__(C.Campaign)
    camp37.b = object()
    camp37._lap_fails = {"moltres": 6}
    camp37._lap_skipped = {"moltres"}
    camp37._lap_prox_fail_clears = set()
    camp37._moltres_fought = False
    camp37._moltres_hide = False
    _orig_tv_h = C.tv.map_id
    _orig_owns_h = C.ram.pokedex_owns
    try:
        C.tv.map_id = lambda _b: LS.ONE_HARBOR   # (32, 4) — harbor, not just town
        C.ram.pokedex_owns = lambda _b, n: False if n == 146 else True
        camp37._lap_pending = lambda k: k == "moltres"
        camp37._dex_owned = lambda n: False
        nxt_h = camp37._victory_lap_next()
        check("proximity unskip on ONE_HARBOR (32,4)",
              nxt_h == "moltres" and "moltres" not in camp37._lap_skipped)
        check("ONE_HARBOR is a MOLTRES_ANCHOR",
              LS.ONE_HARBOR in LS.MOLTRES_ANCHORS)

        # Sevii RE-ARM seam (same predicate as campaign._run_victory_lap).
        camp37._lap_skipped = {"moltres"}
        camp37._lap_fails = {"moltres": 6}
        camp37._lap_sevii_stranded = lambda: True
        key = "articuno"  # what _victory_lap_next would return after honest skip
        if camp37._lap_sevii_stranded():
            bird_spent = (
                bool(camp37._dex_owned(146))
                or getattr(camp37, "_moltres_fought", False)
                or getattr(camp37, "_moltres_hide", False)
            )
            if not bird_spent:
                skipped = getattr(camp37, "_lap_skipped", None)
                if skipped and "moltres" in skipped:
                    skipped.discard("moltres")
                    fails = getattr(camp37, "_lap_fails", None) or {}
                    fails.pop("moltres", None)
                if key != "moltres":
                    key = "moltres"
        check("Sevii RE-ARM forces moltres when articuno next + bird unspent",
              key == "moltres"
              and "moltres" not in camp37._lap_skipped
              and camp37._lap_fails.get("moltres") is None)

        # Ride-home only after bird spent — and it must DRIVE, not no-op.
        camp37._moltres_fought = True
        camp37._active_questline = types.SimpleNamespace(
            gate=types.SimpleNamespace(missing="eevee"))
        driven = []
        camp37._lap_drive_moltres_ride_home = lambda: (
            driven.append("drive"), camp37._clear_questline("test"), "ok")[2]
        camp37._clear_questline = lambda why: driven.append(("clear", why))
        key2 = "articuno"
        if camp37._lap_sevii_stranded():
            bird_spent2 = camp37._lap_bird_spent()
            if not bird_spent2:
                key2 = "moltres"
            elif key2 != "moltres":
                key2 = camp37._lap_drive_moltres_ride_home()
        check("Sevii ride-home DRIVES after bird spent (not no-op)",
              key2 == "ok" and "drive" in driven)
        # Even when checklist key is still 'moltres', spent bird MUST ride-home
        # (not fall through to gate-self-suppressed caught no-op).
        driven.clear()
        key3 = "moltres"
        if camp37._lap_sevii_stranded() and camp37._lap_bird_spent():
            key3 = camp37._lap_drive_moltres_ride_home()
        check("Sevii + spent drives ride-home even when key==moltres",
              key3 == "ok" and "drive" in driven)
        # eevee gate must not arm while Sevii-stranded.
        camp37._lap_sevii_stranded = lambda: True
        eg = C.Campaign._eevee_gate(camp37, {"badge_count": 8})
        check("eevee gate suppressed while Sevii-stranded", eg is None)
        # _lap_bird_spent must see party Moltres (the missing `_dex_owned` bug).
        camp37b = C.Campaign.__new__(C.Campaign)
        camp37b.b = types.SimpleNamespace(rd8=lambda _a: 1)
        camp37b._moltres_fought = False
        camp37b._moltres_hide = False
        _owns37 = C.ram.pokedex_owns
        _rps37 = C.st.read_party_species
        _rf37 = C.fm.read_flag
        try:
            C.ram.pokedex_owns = lambda _b, n: False
            C.st.read_party_species = lambda _b, s: 146
            C.fm.read_flag = lambda _b, f: False
            check("bird_spent True when party has species 146 (dex clear)",
                  camp37b._lap_bird_spent() is True)
        finally:
            C.ram.pokedex_owns = _owns37
            C.st.read_party_species = _rps37
            C.fm.read_flag = _rf37
    finally:
        C.tv.map_id = _orig_tv_h
        C.ram.pokedex_owns = _orig_owns_h
        print()

    # 38. THREE ISLAND BIKER ROADBLOCK (2026-08-06 LIVE): Mart is north of the
    #     talk-only biker pack until Game Corner arms the gauntlet and she fights
    #     through. Unarmed war-chest must bounce to Two Island, never buy_at_mart.
    print("== 38. THREE ISLAND BIKER ROADBLOCK (war-chest Game Corner prime) ==")
    logs38 = []
    sailed38, entered38, cleared38, primed38 = [], [], [], []

    def _mk38(armed, cleared):
        h = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h.b = object()
        h.log = logs38.append
        h.deadline = 1e18
        h._three_gauntlet_armed = lambda: armed
        h._three_bikers_cleared = lambda: cleared
        h.sail = lambda want: (sailed38.append(want), True)[1]
        h.enter_step = lambda tile, dest, label: (
            entered38.append((tile, dest, label)), True)[1]
        h.cross_edge = lambda d, label: (entered38.append(("edge", d, label)), True)[1]
        h.clear_three_island_bikers = lambda: (cleared38.append(True), True)[1]
        h.prime_lostelle_quest = lambda: (primed38.append(True), True)[1]
        return h

    sailed38.clear(); entered38.clear()
    h38a = _mk38(armed=False, cleared=False)
    check("ONE_HARBOR unarmed -> sail Two (Game Corner prime, not Three)",
          h38a.leg_to_ball_mart(LS.ONE_HARBOR) is True
          and sailed38 == [LS.TWO_HARBOR])
    sailed38.clear(); entered38.clear()
    h38b = _mk38(armed=True, cleared=False)
    check("ONE_HARBOR armed -> sail Three for the gauntlet/Mart",
          h38b.leg_to_ball_mart(LS.ONE_HARBOR) is True
          and sailed38 == [LS.THREE_HARBOR])
    sailed38.clear(); entered38.clear(); primed38.clear()
    h38c = _mk38(armed=False, cleared=False)
    check("TWO_ISLAND unarmed -> prime_lostelle_quest",
          h38c.leg_to_ball_mart(LS.TWO_ISLAND) is True and primed38 == [True])
    entered38.clear(); cleared38.clear()
    h38d = _mk38(armed=True, cleared=False)
    check("THREE_ISLAND armed+uncleared -> clear_three_island_bikers",
          h38d.leg_to_ball_mart(LS.THREE_ISLAND) is True and cleared38 == [True])
    entered38.clear()
    h38e = _mk38(armed=False, cleared=False)
    check("THREE_ISLAND unarmed (stuck on pack) -> south to Port for prime bounce",
          h38e.leg_to_ball_mart(LS.THREE_ISLAND) is True
          and entered38 and entered38[0][1] == "south")
    sailed38.clear()
    h38f = _mk38(armed=False, cleared=False)
    check("THREE_HARBOR unarmed -> sail Two (not enter Port into the pack)",
          h38f.leg_to_ball_mart(LS.THREE_HARBOR) is True
          and sailed38 == [LS.TWO_HARBOR])
    # Predicate helpers: hide-flag / scene var.
    class _B38:
        def __init__(self, flags=None, vars_=None):
            self.flags = flags or {}
            self.vars = vars_ or {}
    _orig_rf, _orig_rv = LS.fm.read_flag, LS.fm.read_var
    try:
        LS.fm.read_flag = lambda b, f: bool((getattr(b, "flags", {}) or {}).get(f))
        LS.fm.read_var = lambda b, v: int((getattr(b, "vars", {}) or {}).get(v, 0))
        h38p = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h38p.b = _B38(flags={LS.FLAG_HIDE_THREE_ISLAND_BIKERS: True})
        check("hide-bikers flag SET -> cleared", h38p._three_bikers_cleared() is True)
        h38p.b = _B38(vars_={LS.VAR_MAP_SCENE_THREE_ISLAND: 4})
        check("scene var >= 4 -> cleared", h38p._three_bikers_cleared() is True)
        h38p.b = _B38(vars_={LS.VAR_MAP_SCENE_THREE_ISLAND: 2})
        check("scene var 2 -> armed, not cleared",
              h38p._three_bikers_cleared() is False
              and h38p._three_gauntlet_armed() is True)
        h38p.b = _B38(flags={LS.FLAG_HIDE_THREE_ISLAND_LONE_BIKER: True},
                      vars_={LS.VAR_MAP_SCENE_THREE_ISLAND: 0,
                             LS.VAR_MAP_SCENE_TWO_ISLAND_JOYFUL_GAME_CORNER: 0})
        check("fresh Sevii (Paxton hidden, scene 0) -> unarmed",
              h38p._three_gauntlet_armed() is False)
        h38p.b = _B38(flags={LS.FLAG_HIDE_THREE_ISLAND_LONE_BIKER: False})
        check("Paxton visible (lone-biker hide CLEAR) -> armed",
              h38p._three_gauntlet_armed() is True)
        check("GAME_CORNER_DOOR is Two Island (39,9)",
              LS.GAME_CORNER_DOOR == (39, 9))
        # YESNO loop fix: clear must swap handle_interrupts → confirm for sea_walk
        # (default B-drain = LeaveBikersAlone / walk south).
        swapped = []
        h38s = LS.MoltresHunt.__new__(LS.MoltresHunt)
        h38s.b = object()
        h38s.log = logs38.append
        h38s.deadline = 1e18
        h38s.camp = types.SimpleNamespace(render=lambda: None)
        h38s._three_bikers_cleared = lambda: False
        h38s._three_gauntlet_armed = lambda: True
        h38s.field_heal_seam = lambda **k: None
        _orig_tv38 = LS.tv.map_id
        LS.tv.map_id = lambda _b: LS.THREE_ISLAND
        h38s.handle_interrupts = lambda: False  # baseline (B-drain stand-in)
        h38s.handle_interrupts_confirm = lambda: False

        def _body():
            # During the body, handle_interrupts must be the confirm path.
            swapped.append(h38s.handle_interrupts is h38s.handle_interrupts_confirm)
            return False

        h38s._clear_three_island_bikers_body = _body
        try:
            h38s.clear_three_island_bikers()
            check("clear swaps handle_interrupts to confirm during sea_walk body",
                  swapped == [True]
                  and h38s.handle_interrupts is not h38s.handle_interrupts_confirm)
        finally:
            LS.tv.map_id = _orig_tv38
    finally:
        LS.fm.read_flag = _orig_rf
        LS.fm.read_var = _orig_rv
        print()

    # 39. METEORITE IS A KEY ITEM (2026-08-06 LIVE): bag_count (Items pocket) always
    #     missed it → false 'delivered' → Bill leave-trigger thrash at Celio's machine.
    print("== 39. METEORITE KEY-ITEM + BILL LEAVE ARMED ==")
    h39 = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h39.b = object()
    h39.log = lambda *a, **k: None
    h39.camp = types.SimpleNamespace(
        bag_count=lambda i: 0,                    # Items pocket ALWAYS empty for Meteorite
        _key_item_owned=lambda i: i == LS.ITEM_METEORITE)
    check("meteorite_in_bag reads Key Items (not bag_count Items pocket)",
          h39.meteorite_in_bag() is True)
    h39.camp._key_item_owned = lambda i: False
    check("meteorite_in_bag False when Key Items empty",
          h39.meteorite_in_bag() is False)
    _rv39 = LS.fm.read_var
    try:
        LS.fm.read_var = lambda _b, v: (
            2 if v == LS.VAR_MAP_SCENE_ONE_ISLAND_POKEMON_CENTER_1F else 0)
        check("bill_leave_armed when One-Island PC scene == 2",
              h39.bill_leave_armed() is True)
        LS.fm.read_var = lambda _b, v: (
            1 if v == LS.VAR_MAP_SCENE_ONE_ISLAND_POKEMON_CENTER_1F else 0)
        check("bill_leave_armed False when scene == 1 (Meteorite still owed)",
              h39.bill_leave_armed() is False)
    finally:
        LS.fm.read_var = _rv39
    # ONE_PC with Meteorite in Key Items must EXIT for Lostelle, not leave-trigger.
    entered39 = []
    h39b = LS.MoltresHunt.__new__(LS.MoltresHunt)
    h39b.b = object()
    h39b.log = lambda m: None
    h39b.camp = types.SimpleNamespace(
        bag_count=lambda i: 0,
        _key_item_owned=lambda i: i == LS.ITEM_METEORITE,
        on_event=lambda *a, **k: None)
    h39b.heal_here = lambda *a, **k: None
    h39b.enter_step = lambda tile, dest, label: (
        entered39.append((tile, dest, label)), True)[1]
    h39b.sea_walk = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("leave-trigger must NOT run while Meteorite still owned"))
    h39b.bill_leave_armed = lambda: False
    h39b.meteorite_in_bag = lambda: True
    h39b.one_pc_scene = lambda: 1
    _rf39 = LS.fm.read_flag
    try:
        LS.fm.read_flag = lambda _b, f: False
        check("ONE_PC + Meteorite in Key Items -> pc-out (Lostelle), not leave-trigger",
              h39b.leg_home(LS.ONE_PC) is True
              and entered39
              and entered39[0][1] == LS.ONE_ISLAND)
    finally:
        LS.fm.read_flag = _rf39
    print()

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
