# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ BADGE 4 (Erika / RAINBOW 0x823) — DONE + VERIFIED (shift 17)
coverage-teach-for-depth (5c07f01) CONFIRMED: teaches Venusaur Cut, wins Erika first-try, no blackout.
Banked -> states/workshop/erika_done_kit.state (+ sidecars).

## ✅ BADGE 4.5 QUESTLINE (Silph Scope → Pokémon Tower → Poké Flute → Snorlax) — REPLAYS AUTONOMOUSLY
From erika_done_kit the whole chain replays e2e with zero intervention: got_scope (Rocket Hideout under
Celadon Game Corner) → climbed Pokémon Tower (beat Gary 2F, rescued Fuji) → got_flute → woke+beat the
Route-12 Snorlax → traversed Routes 12-13-14-15 → Fuchsia. Venusaur solo-carries L44→L53. No fix needed.

## ✅ BADGE 5 (Koga / SOUL 0x824) — WON via potion-stall (premise PROVEN); AUTONOMY GAP = Fuchsia-Mart buy
Shift 17 PROVED it: koga_potions.state (Venusaur + injected 20 Hyper/20 Super Potions) → she heals through
the gauntlet (ITEM-INSTINCT use_potion fires, 4 potions used) + the whiff-fix keeps Venusaur in → **beats
Koga, Soul badge set, badges=5** (/g/temp/s17_koga_potions.log OUTCOME GOAL). Banked ->
states/workshop/koga_done_kit.state. ⚠️ THE ONE AUTONOMY GAP: those potions were INJECTED, not bought. For
a truly unaided run from erika_done_kit she must STOCK UP at the Fuchsia Mart before Koga — that shopping
leg is the LAST-MILE BUILD (see below). Next stretch after that = BADGE 6 Sabrina/Saffron.

## FRONTIER (build task): AUTONOMOUS FUCHSIA-MART POTION STOCK-UP before Koga — then badge 5 is unaided

★ ROOT (shift 17, fully diagnosed): she REACHES + fights Koga but LOSES without healing (5+ straight).
It is NOT type/coverage — it's raw ATTRITION: **Venusaur SOLO (L53, 156 HP) cannot tank Koga's 4-mon
gauntlet (~390 total HP) with no healing items.** Bench is L8-14 fodder (team debt #3). WITH potions she
wins (proven). So the fix is autonomous potion acquisition, exactly as shifts 7-10 scouted.

★ ROOT (shift 17, fully diagnosed): she REACHES Koga and fights him, but LOSES every time (5+ straight).
It is NOT a type/coverage problem — it's raw ATTRITION: **Venusaur SOLO (L53, 156 HP) cannot tank Koga's
4-mon gauntlet (Koffing/Muk/Koffing/Weezing, ~390 total HP) with no healing items.** Her bench is L8-14
fodder (team-building debt #3) that gets one-shot the moment she's forced to switch. Over ~15-30 turns she
takes cumulative chip and FAINTS (HP 0), then the fodder can't clean up → blackout → retry → same loss.
(Muk is only NEUTRAL vs grass/poison Venusaur — x2 grass × x0.5 poison = x1 — so it's not even a type
threat; it's just 4-on-1 attrition. The coverage-teach correctly returns no_candidate: Venusaur can learn
NONE of the useful coverage TMs in bag, only TM19 Giga Drain = resisted. Cut/normal x1 IS her best.)

★ FIX SHIPPED THIS SHIFT (43cd243, committed + verified-improving): the WHIFF-SPIRAL breaker mis-read
Muk's Minimize (FOE evasion) as HER accuracy debuff and repeatedly benched L53 Venusaur for L13 fodder to
"reset accuracy" (which can't reset the foe's evasion) — feeding the whole bench to death. Now
_any_healthy_reserve level-gates the reserve (WHIFF_RESERVE_LEVEL_BAND=15): a solo-carry never sacrifices
its ace for far-weaker fodder → "fight on" (keep swinging; misses land). WHIFF_MAX_RECOVERIES 6→2. This
turned "Muk stuck at 130" into "Venusaur grinds Muk 130→19" — real progress, but she still loses to
attrition. General fix (helps any Minimize/Double-Team foe + any solo-carry). Preserves the Gary S.S.-Anne
fix (a real team with comparable reserves still resets).

★ THE ANSWER (scouted by shifts 7-10, now the build task): **POTIONS.** A real player stocks Super/Hyper
Potions at the Fuchsia Mart and heals through the gauntlet. Shift 17 PROVED the premise (koga_potions.state
→ she heals + WINS, OUTCOME GOAL, badge 5). The remaining BUILD is
**autonomous Fuchsia-Mart stock-up before Koga**: recon_fuchsia_mart.py already mapped the Fuchsia Mart
door + buy-list rows (shift 10); the `_shopping_list`/`stock_up` machinery exists (used at Pewter/Vermilion).
Wire a pre-Koga stock-up (like the pre-gym heal gate) so head_to_gym buys potions in Fuchsia before entering
the gym. Alternative/complement: build a real 2nd attacker (team-building debt #3) — bigger project.

★ NEXT-SHIFT PLAY: (1) confirm the potion-stall win in /g/temp/s17_koga_potions.log (proves in-battle heal
actuates + potions beat Koga). (2) Build the autonomous Fuchsia-Mart stock-up (extend prep_for_gym /
head_to_gym with a mart-buy leg for gyms flagged potion-needy; Fuchsia Mart coords in recon_fuchsia_mart.py).
(3) Re-run from koga_retry_kit.state toward 0x824 → she buys potions → wins Koga → bank koga_done_kit.
FIXTURES (states/workshop/): koga_retry_kit.state (Fuchsia, Venusaur L53, badge 4, NO potions — the real
autonomous start); koga_potions.state (same + injected potions — premise test); erika_done_kit.state
(badge 4, pre-questline). recon_bagdump/partydump take a FULL path; recon_longrun takes a BASENAME.

★ RESIDUAL (watchability, fix when convenient): Koga/maze-gym junior-spin — the invisible-wall maze leaves
~4 juniors un-engageable; she tries each 4× then burns the 14-clear-round cap (~1500 log lines / minutes)
before the leader, EVERY attempt. The cap correctly bails her to Koga so it's not fatal, but it's slow.
Fix in _clear_junior_trainers (campaign.py:3182): a reachability pre-check — if no path exists to a tile
adjacent to a junior, defer it in 1 try not 4 (the reachability-pre-check law), and/or remember the
deferred set across blackout-retries.

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — recon_longrun auto-reaps via longrun.pid; kill
`taskkill //F //IM python.exe //T`). recon_longrun arg1=BASENAME in workshop/, arg2=max_minutes. GOAL
0x824=FLAG_BADGE_SOUL. Decision-counter FREEZES during a gym (watch growing log lines). Battle move-picker
pol.choose_move scores max(power,1)*eff, NO STAB. Coverage table = campaign.py:248 _COVERAGE_MOVES.
Whiff-breaker = battle_agent.py:~2509 + _any_healthy_reserve:2041. inject_potions.py <src.state> <dst.state>.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 5 Koga (she reaches +
fights him autonomously; loses on attrition; the fix is autonomous Fuchsia-Mart potion stock-up, premise
proven). Pop-in = `python pokemon_agent/watch.py`.
