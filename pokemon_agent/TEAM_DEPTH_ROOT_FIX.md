# TEAM DEPTH — the root bug: why Kira reaches the E4 with only ~2 usable mons (2026-07-10)

## THE DIAGNOSIS (evidence-backed, read-only recon)
**The team-planner brain produces the RIGHT plan but almost NONE of it is EXECUTED.** It's a voice-only
suggestion layer bolted next to an OLDER strategic-grind system that structurally only levels the ACE and
only targets a party of 3. Result: the ace hogs all XP, 1-2 type-answers get caught, everything else stays
frozen at catch-level. **The KB data (Part A) and the planning logic (Part B) are VERIFIED-good — the break
is Part C, the executor.**

### Root causes (ranked, with file:line)
1. **TeamPlanner's leveling verbs are dead ends.** `assess()` returns `grind_to` (pokemon_planner.py:474)
   and `develop_bench` (pokemon_planner.py:487) — but **ZERO consumers in campaign.py**. The only "level the
   team" actions the brain emits reach nothing but `plan_note` (campaign.py:10058 = oracle VOICE ctx =
   display-only). Per CLAUDE.md rule 1 this is "display-only, NOT wired to decision" masquerading as a team-builder.
2. **The one wired catch hook self-disables at 3 mons + never routes to keeper locations.**
   `_plan_wants_prebuild` (campaign.py:4339) hard-returns False when `pc > 2` (campaign.py:4355).
   `_plan_keeper_target` (campaign.py:4319) only returns a target if the species is ON THE CURRENT MAP
   (campaign.py:4333) — never travels to Route 24 (Abra), Diglett's Cave, etc. So slots 4-6 are never pursued.
3. **The actual leveling system targets a party of 3 and levels only the ACE.**
   `GYM_PARTY_TARGET=3` (campaign.py:235); `grind()` reads the LEAD's level only (campaign.py:4954);
   `grind_weak_members` degrades to `_restore_ace(); grind(target)` = ace-only when the participation switch
   wedges (campaign.py:5269-5273); `_prep_team_target` returns None (defers) for ALL 4+-mon rival walls
   (campaign.py:5201). The proactive bench-raise runs ONCE per roster signature then never re-fires
   (campaign.py:5234) — so the bench gets one small bump while the ace runs away to L71.
4. **No E4 team prep exists.** Only the 8 gyms have `prep_for_gym`. Nothing reads `frlg_team_plan.json`'s
   `level_milestones["E4"]` (55) to level the WHOLE party before Indigo. Whatever the bench is at badge 8 walks
   into Lorelei.

### Verification status
- Part A (KB `frlg_team_plan.json` + rosters/evos/learnsets/tms/encounters): **VERIFIED** — real 6-slot
  `balanced-classic` archetype, coverage + acquire locations + by-badge deadlines + E4 milestone. **Data is not the problem.**
- Part B (`assess()`/`next_action()`): **VERIFIED unit-level only** (recon_teamplanner_test.py, pure logic, no emulator).
- Part C (executor): **COMPILES but largely NOT WIRED** — only a sliver of `catch_keeper`; `grind_to`,
  `develop_bench`, `acquire_special`, `evolve`, `teach_tm` → voice-only.
- FINAL PROOF (fresh run builds a full 6): **NEVER VERIFIED** — abandoned mid-executor; project pivoted to nav
  blockers then to hand-grinding the E4 team via port scripts.
- **PROOF the autonomous builder is broken:** this shift's E4 workaround = manual struct-porting
  (`recon_grind_bench.py` + `recon_port_and_fix.py`) = hand-leveling specialists OUTSIDE the autonomous loop.

## THE FIX (smallest set that makes a fresh run arrive E4-ready with a real leveled 6)
Data + brain already exist — this is all Part-C WIRING in campaign.py.
1. **Wire `catch_keeper` to route + catch the WHOLE team, not just while thin.** In free_roam, when
   `assess()` returns `catch_keeper`, travel to `act["where"]` (resolve via frlg_encounters.json) then
   `catch_one(target_species=act["species"])` regardless of party size. Relax the `pc>2` gate
   (campaign.py:4355) to pursue slots 3-6 (bounded: one DUE keeper at a time). Make `_plan_keeper_target`
   fall back to ROUTING when the keeper isn't on the current map.
2. **Wire `grind_to`/`develop_bench` to a REAL bench-leveling objective.** Add an executor branch consuming
   those kinds → `grind_weak_members(target)`; FIX `grind_weak_members` so its default (switch-wedged) path
   does NOT collapse to ace-only `grind(target)` (campaign.py:5269-5273) — either finish the participation
   switch (battle_agent.py ~2540) or field-and-solo-grind each under-target bench mon in turn
   (`SOLO_WEAK_GRIND`, campaign.py:71, currently default-off).
3. **Raise the party target toward 6 for late gyms + add `prep_for_e4()`** that reads
   `level_milestones["E4"]` (55) and levels the whole party FLOOR (not the ace) before entering Indigo.
4. **Add box/replace-fodder logic** (Tier-1 #15 PC/Box, ❌ in CLAUDE.md): once a planned keeper is in hand,
   box off-plan catch-level fodder so it stops occupying slots 4-6.

**Functions to change:** `_plan_wants_prebuild`, `_plan_keeper_target` (un-gate + route); `free_roam` (add
plan-driven catch/grind objective consuming `assess()`); `grind_weak_members` (fix ace-only fallback);
`_prep_team_target` (don't blanket-defer 4+ rivals); add `prep_for_e4`; add box-fodder logic.

## VERIFICATION (the FINAL PROOF gate)
Fresh early-game state (e.g. `states/workshop/og_postopening.state`) → headless 15x look-ahead run → confirm
from the log + party growth that she PROACTIVELY catches keepers at their locations, LEVELS the bench to
milestones, and arrives at the E4 with a real leveled 6 + coverage. Bounded detours, watchable pace, narrated.

## NS#2 UPDATE (2026-07-11) — milestone-cap TARGET landed + the REAL binding constraint pinpointed by a live run
**LANDED (commit fb26e6e):** the mid-game milestone-cap bench-prep (fix #3 above, mid-game half). `_prep_team_target`
now caps the proactive bench pin at the team-plan's NEXT gym milestone (not the ace-relative lead-8) and re-arms on a
milestone RISE. Decision-verified 10/10 (`recon_milestone_prep_check.py`). The prep TARGET is now correct + wired (it
reframes the "battle" action → "train the WEAK ones to L{prep_t}" @campaign.py:8386-8401 and feeds `grind_weak_members`
@campaign.py:9492-9499 — NOT display-only).

**BEHAVIOURAL RUN (ss_ticket.state, badge 2, Ivysaur L28 + Rattata/Spearow L14 → `G:/temp/longrun/ns2_milestone.log`):**
she went Route 25 → Cerulean (bought Poké Balls) → Route 5/6 → Vermilion; team-brain fired (`catch_keeper: abra →
alakazam`), plan-catch fired (`PLAN-CATCH: seeking planned keeper 'abra'... fleeing non-targets`), and prep=20 stayed
STABLE across the lead going L28→L32+evolve (no treadmill, no road-parking). BUT the net was **lead L28→Venusaur L32,
bench FROZEN L14/L14, no keeper caught**, then WEDGED at the Vermilion Cut gate.

**THE REAL BINDING CONSTRAINT (found this run — the mid-game bench never levels, and WHY, two layers):**
1. **The bench gets ZERO XP from road trainer wins.** She won ~8 trainer fights Route24→Vermilion; ALL XP went to the
   lead. The participation-XP switch (`battle_agent.py:2554`) is gated to `PROTECT_LEAD_GRIND` — set True ONLY inside a
   dedicated `grind_weak_members()` session (`battle_agent.py:141`). During ordinary road/`head_to_gym` battles it's
   False → the ace solos → bench banks nothing. AND participation XP requires the weak mon to LEAD at battle start
   (only `grind_weak_members` reorders it to slot 0); on the road the ace leads. So organic bench-XP-on-the-road does
   NOT exist yet — it is the watchable path ("the whole team fights its way to the gym", no grind-wall) and it's unbuilt.
2. **The oracle never picks a dedicated bench-grind mid-game.** `push-when-carrying` (night-train s8, 959eb75) correctly
   marches the billed road when the ace can carry, so `grind_weak_members` never runs mid-game → PROTECT_LEAD_GRIND
   never True → the proven participation switch never even fires. (This is CORRECT for watchability — grind-walls are a
   FAILURE — which is exactly why the fix must be organic participation XP, not forced grinding.)

**⇒ THE PINPOINTED NEXT FIX (the highest-leverage team-depth piece, precisely located, NOT a safe blind edit — needs a
careful design + full re-run to verify no faint/wedge/slow-down):** make the bench bank participation XP during ORDINARY
road trainer battles when it's under its milestone target — i.e. field/register the under-milestone weak mon in road
fights (extend the `battle_agent.py:2550-2580` participation path beyond PROTECT_LEAD_GRIND, guarded so the weak mon
takes no hit and never strands, per the 2584 STRAND-ROOT note), driven by a "bench under `_prep_team_target`" signal
from campaign. This levels the bench organically as she travels — no grind-wall — so she arrives at each gym (and badge
8's `_prep_e4_target`) already near-milestone instead of L14, dissolving the frontier-#1 grind-spot-adequacy wall too.
⚠️ This touches the in-battle switch (Tier-1 #5, historically wedge-prone) — arm behind a flag, verify on a PAST-Cut-gate
bench bank (ss_ticket wedges at the Vermilion Cut tree, so use surge_done/erika_done or a fresh run that clears the gate).

## NS#3 UPDATE (2026-07-11) — NEW#1 BUILT: organic bench XP on the road (commit a998378)
**LANDED (mode-side, flag-gated, canonical untouched):** the exact NEW#1 fix. Two small helpers in
campaign.py — `_road_bench_xp_arm(pick, state)` / `_road_bench_xp_disarm()` — wired into `free_roam`
around the `_route_action` dispatch (arm right before, disarm in a `finally`). On a FORWARD-MARCH pick
(`head_to_gym` / `travel:`) when a bench member is under its `_prep_team_target` milestone, the weakest
LEVELABLE under-target mon is swapped to slot 0 (so it's "sent out" = XP-eligible) and
`battle_agent.PROTECT_LEAD_GRIND` is armed — the SAME proven participation switch grind_weak_members uses
(battle_agent.py:2554) fields the ace turn 1 so the weak mon banks a share of XP without taking a hit. The
ace is restored to slot 0 the instant the leg ends, so the weak lead NEVER outlives the march into any
readiness/heal/decision read. Flag `POKEMON_ROAD_BENCH_XP` (default ON; instant one-line revert).

**Guards (all verified):** only forward-march picks (never wander_catch/beat_gym — a weak mon must never
lead a leader or a catch); skips when hurt (`needs_heal`), thin (<2 mons), or the bench is at milestone
(prep None); excludes far-below box chaff (`< prep - E4_PREP_BAND`) and stall-marked mons; no-op when the
weakest levelable IS the ace. Mirrors grind_weak_members' selection so the two agree on "the weak one".

**VERIFICATION (three-state honest):**
- **Decision logic — VERIFIED 13/13** (`recon_road_bench_xp_check.py`, pure-logic, no ROM): non-march
  picks / prep-None / hurt / happy-path (weak→slot0 + PROTECT True, disarm restores ace + clears flag) /
  chaff-excluded / stall-marked / flag-off / weakest-is-ace.
- **Live arm+disarm — VERIFIED clean** on a `surge_done` look-ahead: armed twice (L9 Ekans led, target
  L15), disarmed/restored the ace, ZERO strand/faint/crash. It correctly did NOT arm during town/interior
  nav (erika_done Celadon run: 0 arms while shopping/questing indoors — right, no road battles in town).
- **END-TO-END — VERIFIED (bucket a)** on a `snorlax_done` look-ahead (`LONGRUN_BATTLE_LOG=1`, the
  Routes 13/16 gauntlet to Fuchsia): **5 arms → 3 LIVE participation switches in road trainer battles**
  ("GRIND SWITCH: weak lead out -> switching to ace slot 3", all SWITCHED-confirmed) → **the bench leveled
  organically** (Spearow L15→16, Ekans L9→10) while marching. No strand from the switch (0 "switch did not
  confirm"). GOTCHA banked: recon_longrun SUPPRESSES battle_agent's log unless `LONGRUN_BATTLE_LOG=1` — that
  masked the switch on the first passes. (City-boot probes surge_done/erika_done were absorbed by town/gate
  nav before road trainers; bill_done marched but its Route 24/6 trainers are spent — don't respawn.)
- **THE ONE CAVEAT (a hand-bank artifact, mitigated):** the snorlax run then WHITED OUT on the Route 13
  gauntlet. The CONTROL (flag OFF, same boot) hit the SAME critical-HP + PP-famine wall → it's pre-existing
  TOP-HEAVY-BANK attrition (L48 ace soloing while an L9-15 bench can't share the load), NOT caused by
  road-bench-xp. The participation switch does cost the ace one free enemy hit per battle, marginally
  worsening a no-heal gauntlet. **Guard added (commit 838e9fb):** don't START a bench-XP leg with the ace
  below `POKEMON_ROAD_XP_ACE_HP_FLOOR` (0.6) HP — defer to heal. On a FRESH organically-built run the bench
  shares the gauntlet from the start, so this solo-attrition shape doesn't arise (it's exactly what the fix
  prevents over time). Verified 15/15 including the two ace-HP-floor cases.

## ALSO (option-1 Lapras tactical AI — good regardless)
Battle-brain: when facing a foe the ace can't hurt (Venusaur 0.25x/1x into Charizard) and a specialist is
the type-answer, LEAD/keep that specialist in and funnel healing to IT rather than splitting FR with the
ace's 1x Cut. (run12 Gary census: Lapras landed only 10 SE hits before fainting while Venusaur threw 51 weak
hits.) This is just good battle-AI; wire it in independent of the team-depth fix.

## NS#4 UPDATE (2026-07-11) — road-bench-XP re-validated in party-6 mid-game + NEW#2 CROSS-MAP KEEPER ROUTER built
**LOOK-AHEAD FINDINGS (rule 8 — the oracle fingered the real blocker):**
- **og_postopening is an INVALID final-proof fixture:** it has NO `world_model.json` sidecar, so free_roam
  boots with an EMPTY world graph. head_to_gym to Pewter/Cerulean (the two EARLY gyms whose roads are NOT
  billed in frlg_gates — gyms 3+ are) returns `no_path` → she livelocks grinding Route 1 (Bulbasaur→Ivysaur
  L16). A TEST-FIXTURE artifact (a real playthrough accretes the graph as it walks), not a Kira-timeline bug.
  Team-depth must be tested from a MID-GAME fixture (billed roads path with an empty graph).
- **road-bench-XP (NEW#1) VALIDATED in the party-6 mid-game** on an `erika_done` look-ahead (badge 4, the
  poster-child disease: Venusaur L43 ace + a FROZEN L9-15 chaff bench: Ekans L9 / Meowth L10 / Pidgey L13 /
  Rattata+Spearow L15). Marching Celadon→(Flute questline)→toward Fuchsia it ARMED every forward leg + the
  GRIND SWITCH fired every road battle → **the bench leveled organically: Ekans L9→L14, Rattata/Spearow
  15→16** while she correctly chained the Snorlax→Flute→Silph-Scope→Tower questline. prep bite = min(milestone,
  floor+6)=15, climbing (NS#2 pacing; ~5 levels in ~9 min of run — the +6 bite is slow but monotonic).
- **THE CONFIRMED GAP = team COMPOSITION.** The planner explicitly emits `catch_keeper: abra → alakazam
  (answers Koga/Sabrina/Bruno/Agatha/Champion)` the WHOLE erika run but she never fetches it — she picks
  head_to_gym and marches past because Abra is OFF the current map and the on-map un-gate only grabs keepers
  she's STANDING on. So she arrives with LEVELED CHAFF, no coverage. This is NEW#2.

**NEW#2 BUILT — CROSS-MAP KEEPER ROUTER (campaign.py, flag `POKEMON_KEEPER_ROUTER`, DEFAULT OFF).** Offers a
BOUNDED detour (`fetch_keeper` action) to a nearby reachable map that hosts the DUE plan keeper, then the
existing on-map machinery catches it. Pieces: `_place_to_map_index` (reverse of `_PLACE_NAMES`, name→entrance
map), `_keeper_route_target` (species→hosting-map via the encounters KB, party<6, off-current-map, within
KEEPER_ROUTER_MAX_HOPS=6, AND a rideable next hop exists), `_fetch_keeper_errand` (routes one hop via the
PROVEN `_travel_to_known` traveler, targeted catch on arrival), offered in `_available_actions` before the
forward-drive prune, dispatched in `_route_action`. Recon chooser (recon_longrun.py) taught to model the
faithful "take the bounded coverage catch" pick.
- **DECISION-LOGIC VERIFIED 10/10** (`recon_keeper_router_check.py`): reverse index + fires-for-in-range /
  defers on on-map / party-full / out-of-range / unreachable / non-catch / no-rideable-hop / retired-target / flag-off.
- **BEHAVIORAL — the look-ahead caught + I FIXED a LIVELOCK (the whole reason it's default-OFF + verified).**
  First `route3_caught` run (party 2, Abra due): the router fired but the errand used the naive
  `trav.travel(target_map=)` → `no_path` while the OFFER check used `world.route` (static connections the
  learned-graph traveler can't RIDE) → MACRO-RED spin (17+ stuck ticks). FIXED: (a) offer⟺executable — require
  `_next_step_rideable` non-None so we only offer what `_travel_to_known` can ride NOW (keeps it to near/known
  keepers); (b) `_travel_to_known(hunt_on_arrival=False)` for pure routing + the errand's own targeted catch;
  (c) a STALL GUARD (`KEEPER_ROUTER_STALL_CAP`=3) retiring an un-rideable (cur,tmap) to `_keeper_unreach` so
  she falls through to head_to_gym — never livelocks. Re-verifying behaviorally now (route3_caught post-fix).

**⇒ NS#4 FRONTIER (in priority order):**
1. **FINISH the router behavioral verify + flip default ON.** Confirm on route3_caught (post-fix) she either
   routes+catches Abra cleanly OR retires the target and resumes head_to_gym (no MACRO RED). Then set
   `POKEMON_KEEPER_ROUTER` default "1", run the final-proof gate (a mid-game fixture forward) to confirm she
   assembles real coverage. ⚠️ Watchability of the detour (does she over-backtrack?) is a bucket-(b) LIVE-EYES
   item — headless can't judge it; tune KEEPER_ROUTER_MAX_HOPS down (4?) if the detours read too far.
2. **PC/BOX (Tier-1 #15)** — the pairing gap for the FULL-party case (erika_done: 6 chaff, router won't fire
   without room). recon_pcbox.py deposit flow proven; generalize `deposit_mon`/`withdraw_mon` + hook `catch_one`
   to box the lowest-value chaff on a full-party keeper catch (so the router can then add the keeper).
3. **prep bite cadence** — the +6 bite levels the bench slowly (~5 levels/9min). Consider a faster cadence or a
   bigger bite when the bench is FAR under milestone, so she arrives near-milestone without a grind-wall.
