<!-- ═══ NIGHT-SHIFT 8 (2026-07-10) — IN FLIGHT: S.S. ANNE Gary — WILD-GRIND-TO-L32 IS A DEAD END ═══
SHIFT-7 AUTOPSY (s7_verify2.log): the L32 prep-grind fix does NOT work — it's TOO SLOW. Boot ace is
ivysaur L26 (bill_done); grinding L26->L32 = 6 levels on Route 6 wilds (~L8-13 Pidgey/Rattata, ~100 XP
each to an L26+ ace = ~15k XP ≈ 150 encounters). The killed run reached only L28 after the whole clock.
Wild-grinding 6 levels on trash is neither fast NOR watchable (constitution FAIL). ABANDON this direction.

THE HUMAN ANSWER (shift-14's flagged "HIGHEST VALUE", never built): a real player does NOT wild-grind to
beat ship-Gary — they FIGHT THE S.S. ANNE CABIN TRAINERS (Gentlemen/Sailors/Fishermen, L16-18, big
TRAINER XP) on the way to the captain, which levels the WHOLE team fast + watchably, THEN fights Gary at
appropriate level. The shift-12 DIRECTED ship-nav B-LINES to the Gary warp, skipping ALL ~8-10 cabin
trainers -> she arrives underlevelled every time. THE FIX = make ship-nav ENGAGE cabin trainers before
the Gary warp (fail-open; her L26 ace beats L16-18 cabin trainers easily = the intended level-up).

GROUND TRUTH (s8_gary_observe.log, prep-grind OFF): L29 ivysaur LOSES Gary (grudge 4W-5L). Turn log shows
the exact mechanism — Pidgeotto's SAND-ATTACK triggers a WHIFF-SPIRAL (Tackle stuck missing, foe HP frozen
23/54 for many turns even while asleep) that BURNS her PP; by Charmeleon she's PP-FAMINED and forced to the
frail bench (spearow Peck barely dents it) -> loss. NOT a level problem (L29 >> Gary's L16-20); it's PP
attrition + a dead-weight bench. Only L32/Venusaur (raw power ending each mon in 1-2 hits, no famine)
reliably wins the reactive way, and wild-grind to L32 is too slow. THE FIX = the human path.

THREE FIXES BUILT THIS SHIFT (2f2e918 cabins committed; +2 more this commit):
 1. FIGHT-THE-CABINS-FIRST (campaign.py ~6435, kill-switch POKEMON_SHIP_CABIN_SWEEP=1) — VERIFIED FIRING:
    the directed hop B-lines up the stairs skipping cabins; now she sweeps this deck's un-entered cabins
    (fought ~17 cabin trainers, ace L26->L30, +$4200 prizes) before climbing. Bounded, fail-open.
 2. IMMOBILIZATION-LIVELOCK FIX (battle_agent.py ~1522) — VERIFIED BUG: a set paralysis bit (0x40) was
    blamed for EVERY non-firing turn, so a 0-PP move (PP famine) masked by paralysis spun 183 turns vs a
    full-HP Kadabra (0 damage, "fully paralyzed" every turn — impossible for real 25% paralysis). FIX:
    require pp0>0 for any immobilization credit + cap consecutive paralysis attributions at 6 -> otherwise
    rotate/flee (skip_streak). Sleep/freeze keep trust-indefinitely. Generalizes to every paralysis fight.
 3. PREP+CABIN COMPOSITION (campaign.py ~5972, POKEMON_RIVAL_PREP_MARGIN=4): wild-grinding all 6 levels
    L26->L32 up front is slow/unwatchable. Now the wild prep-grind fires ONLY within margin-4 of the L32
    target (>=L28). So: pass 1 at L26 -> prep OFF -> board -> cabins carry to ~L30 -> (if she loses Gary
    cleanly) whiteout to Vermilion at ~L30 -> prep NOW fires L30->L32/Venusaur (2 fast levels) -> re-board
    (cabins beaten, walk through) -> full-PP Venusaur solos Gary (reactive-proven win) -> Cut -> Surge.

VERIFYING (the composition, prep-grind ON = default 32):
  POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 35  (log /g/temp/s8_compose.log)
EXPECT (2 passes): board L26 -> cabins -> L30 -> lose Gary (CLEAN, no spin) -> Vermilion prep-grind
L30->L32 Venusaur -> re-board -> Gary WIN -> captain HM01 Cut -> teach -> cut Vermilion gym tree -> Surge.
IF she STILL loses at Venusaur L32: the whiff-spiral (Sand-Attack accuracy debuff PERSISTS across foe
faints, burns PP) is the last root -> fix = switch ace OUT+IN to reset accuracy when whiffing detected
(fragile Tier-1 #5 switching) OR proactively Sleep-Powder Pidgeotto turn 1 before it debuffs.
Fixtures: bill_done_kit.state (rebuild: `.venv/Scripts/python.exe pokemon_agent/recon_repair_kit.py` ->
writes pokemon_agent/states/workshop/), bill_done.state (rebuild: Copy-Item G:\temp\longrun\banked_GOAL\kira_campaign.state states\workshop\bill_done.state -Force).
═══════════════════════════════════════════════════════════════════════════════════════════════ -->

<!-- ═══ NIGHT-SHIFT 6 (2026-07-10) — verify NEUTRAL-COVERAGE fix -> Route 6 -> Gary -> Surge ═══
KEY FINDING: the S.S.Anne wall was NEVER Gary — she never reached him. She PP-FAMINED and LOST to a
3-Pidgey Bird Keeper on ROUTE 6 (grass resisted x0.5 by Pidgey/Flying), then looped loss->heal->back.
ROOT CAUSE (not the heal gate): _ensure_move_room's _value gave EVERY status move +100, so Ivysaur kept
2 powders (Sleep+Poison) and DROPPED Tackle — its only NEUTRAL move — leaving a grass-ONLY kit walled by
the Flying/Bug foes that carpet Routes 6/24/25. Abra knows only Teleport (dead weight). FIX COMMITTED
(dd5546c): demote a redundant 2nd status (+25 not +100) + protect a low-power off-type attacker; verified
[VineWhip,RazorLeaf,Poison,Sleep]->drop Poison (keeps 2 STAB + Sleep). Built bill_done_kit.state
(recon_repair_kit.py) = the kit the FIXED manager keeps [RazorLeaf,VineWhip,Tackle,Sleep].
PROGRESS (4 fixes committed): dd5546c (Tackle CLEARS Route 6 — VERIFIED) + 29756ee (heal-before-rival PP
floor 20->70 — VERIFIED heal FIRES at Vermilion Center) + 4e1200a (PREP-BEFORE-RIVAL grind gate — the wall
is TEAM STRENGTH: even full-PP L28 Ivysaur lost Gary 4W-4L via Pidgeotto Sand-Attack misses + Charmeleon
grass-resist + frail L8-12 bench) + 30a166e (prep-grind must route to Route 6 grass FIRST — it wedged in
the gym-approach pocket where grind()'s grass-finder targets a bad off-map coord; now walk_to_map((3,24),
north) then grind, target lowered L32->L30 for budget feasibility).
IN FLIGHT: final verify of the prep-grind chain:
  POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 25  (log /g/temp/s6_prepgrind2.log)
Expect: 🏋️ prep-grind fires at Vermilion -> routes to Route 6 -> grinds ace to L30 (+ bench via
participation-switch) -> back to Vermilion -> heal -> Tackle+full-PP BEATS Gary -> HM01 Cut -> teach ->
cut gym tree -> Lt. Surge (badge 3). RISK if still losing: L30 not enough (bump toward L32/Venusaur) OR
the bench needs its own leveling. RISK if grind slow: Route 6 wilds are low-XP for an L28+ ace — the grind
may eat budget before L30 (watch GRIND: trained-to lines); if so, grind may need trainer-XP or a lower
target. Fixtures: bill_done.state (degraded BUG kit), bill_done_kit.state (repaired [RazorLeaf,VineWhip,
Tackle,Sleep]). Reconstruct bill_done: Copy-Item G:\temp\longrun\banked_GOAL\kira_campaign.state states\workshop\bill_done.state -Force The kit re-run showed Tackle cleared Route 6 but she STILL lost Gary
because the heal gate's floor=20 read her ~50 post-Route-6 PP as "fine" -> boarded the Center-less ship
gauntlet under-fuelled -> Tackle hit 0-PP -> famined on Charmeleon (grass resisted x0.25). Floor now 70
(top off before the dock). IN FLIGHT: re-run from repaired-kit with the heal-floor fix:
  POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 25  (log /g/temp/s6_healfloor.log)
Expect: heal-before-rival FIRES at Vermilion (full PP) -> Tackle+Razor Leaf BEAT Gary -> HM01 Cut ->
teach -> cut gym tree -> Lt. Surge (badge 3). If she STILL loses at full PP, the frail bench (Spearow L12,
Rattata L8, Abra=Teleport-only) can't finish what Ivysaur can't solo -> next lever = grind the bench a few
levels OR the in-battle status-spam calibration. Fixtures: bill_done.state (degraded, the BUG),
bill_done_kit.state (repaired kit [RazorLeaf,VineWhip,Tackle,Sleep]).
Reconstruct bill_done if missing: Copy-Item G:\temp\longrun\banked_GOAL\kira_campaign.state states\workshop\bill_done.state -Force
═══════════════════════════════════════════════════════════════════════════════════════════════ -->

<!-- ═══ NIGHT-SHIFT 4 (2026-07-10) — IN FLIGHT: S.S. ANNE -> CUT -> SURGE (badge 3) via team-brain ═══
FIXTURE-LOSS LESSON (hard-won, capture EXPLICITLY): states/workshop/*.state is GITIGNORED SCRATCH and
got wiped between shifts — bill_done.state AND misty_done.state were GONE at shift-4 boot. BUT recon
banks every run's end-state to G:/temp/longrun/banked_<OUTCOME>/kira_campaign.state, which SURVIVES.
The shift-3 S.S.-Ticket bank = `banked_GOAL` (2026-07-09 23:51); its follow-on S.S.-Anne characterization
= `banked_TIMEOUT` (2026-07-10 00:12). RECONSTRUCT the fixture any time it's missing:
  Copy-Item G:\temp\longrun\banked_GOAL\kira_campaign.state states\workshop\bill_done.state -Force
VERIFIED bill_done party = Ivysaur L26 + Spearow L12 + Rattata L8 + Abra L10, badges 2, SS_TICKET(0x234)=1
(probe: .venv/Scripts/python.exe /tmp/probe_state.py). Surviving reactive-track fixtures on disk:
surge_done.state (badge 3, solo Venusaur L32 — the OLD reactive track, NOT the team-brain team),
flash_done.state, rt_mouth.state.

FRONTIER (shift 4): S.S. ANNE -> CUT -> VERMILION (Lt. Surge, badge 3), from bill_done.state.
DIAGNOSED (s4_ssanne.log): navigation is FINE end-to-end — Cerulean->Route5(burgled-house pass)->Route6->
Vermilion->boards the S.S. Anne, climbs the KB chain to the captain. THE WALL = rival GARY on the ship
(raticate lead, 4 mons) via PP FAMINE: bill_done Ivysaur L28 has ONE damaging move (Razor Leaf) + 2 powders
+ an EMPTY 4th slot (it LOST Vine Whip — likely the shift-15 in-battle decline-handler mis-actuation,
battle_agent.py:2219, still a HANDED-OFF bug), and she never heals PP at the Vermilion Center before
boarding, so Razor Leaf runs dry mid-4-mon fight -> loss -> whiteout -> unwatchable grind-loop (levels a
L8 Rattata toward L17). The old solo-Ivysaur track (surge_done: Razor Leaf+Vine Whip+Cut) beat Gary at
full PP — so the fight is WINNABLE; the famine is the killer.

FIX COMMITTED (a6bbbd9): HEAL-BEFORE-A-RIVAL-FIGHT gate. (1) new _lead_attack_pp_low sums only DAMAGING PP
(powders no longer mask a dry attacker). (2) _run_questline_step: a rival/trainer-gauntlet step + on the
anchor town + ace not attack-fresh -> heal_nearest() FIRST (bounded once per approach). VERIFY IN FLIGHT:
  POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done.state 25
Expect: 🩹 HEAL-BEFORE-RIVAL fires at Vermilion -> full Razor Leaf PP -> BEATS Gary -> captain -> HM01 Cut
-> teach -> cut the Vermilion gym tree -> Lt. Surge (badge 3). IF she still loses Gary at full PP, the
1-attacker kit is genuinely too thin -> either restore a 2nd attacker (fix the battle_agent:2219 in-battle
learn to REPLACE junk not decline, so a fresh run keeps Vine Whip) OR execute her narrated Diglett plan
(Route 11 east -> Diglett's Cave; needs a cave-floor catch primitive). Boot:
  Copy-Item G:\temp\longrun\banked_GOAL\kira_campaign.state states\workshop\bill_done.state -Force  # if fixture missing
═══════════════════════════════════════════════════════════════════════════════════════════════ -->

<!-- ═══ NIGHT-SHIFT 3 (2026-07-09) — GATE C BILL MILESTONE ✅ DONE (S.S. Ticket obtained e2e) ═══
GATE C's Bill leg is COMPLETE + COMMITTED (0233a86 + 24b863f). recon_longrun misty_done.state 20 ->
OUTCOME: GOAL FLAG_GOT_SS_TICKET, VERIFIED e2e: escape the (7,3) house -> build a team -> TARGETED-catch
Abra ("this is the one I came for") -> cross Route 25 -> enter Bill's cottage -> talk Bill x2 -> S.S. Ticket.
FOUR general fixes (all campaign.py, POKEMON_TEAM_PLANNER-gated, canonical Champion UNTOUCHED):
 1. INTERIOR-WEDGE ESCAPE dominance (~7820): wedged in a dead-end interior (nomove>=2, map[0]!=3),
    suppress talk_npc so leave_building dominates (get out of the wrong building).
 2. DISTANT-DOOR APPROACH (_questline_interact ~6500): target building ACROSS the arrival map (cottage at
    Route 25's far EAST, she lands WEST) -> _door_tiles scans the whole layout -> TRAVEL to the nearest
    un-entered door, enter fires next tick. + RE-ENTRY: all-doors-entered-but-flag-unset -> clear the
    map's door-burn for a clean re-approach (else one interrupted visit locks her out forever).
 3. HEAL-GATE false-eject (~8856): a gauntlet fight + the deliberate door-warp was misread as a whiteout
    relocation and ejected her from the cottage. FIX: only treat battle+relocation as a whiteout when the
    party is actually HEALED; else keep + refresh interiority.
NEW FIXTURE promoted: states/workshop/bill_done.state (badge 2, S.S. Ticket, party Ivysaur L26 +
Spearow L12 + Rattata L8 + Abra L10, in Bill's cottage). World model loads from CANONICAL (recon:235) so
the fixture only needs the .state.

FRONTIER (bank-and-continue): the S.S. ANNE -> CUT -> VERMILION (Lt. Surge, badge 3) stretch. Chain:
S.S. Ticket -> south through Cerulean (the guard gate opens with the ticket) -> Route 5/6 -> Vermilion
harbour -> board S.S. Anne -> beat rival Gary -> captain gives HM01 Cut -> teach Cut -> cut the tree to
the Vermilion gym -> Surge. CHARACTERIZED this shift (`recon_longrun bill_done.state 16` ->
OUTCOME TIMEOUT, NOT a hard stall; log G:/temp/gatec_s3_ssanne.log): she DOES progress (reached Route 6
by tick 8, Ivysaur L27->28) but SLOWLY — only ~27 ticks in 16 min. Behaviour = a lot of building-touring
(head_to_gym -> questline_talked x5 / questline_deeper x3 at a Cerulean-area building (1,6)), stock_up,
a battle_loss + blacked out once, need_heal x3, and slow grinding. The new post-ticket questline (board
S.S. Anne / Cut) needs the same destination-verification pass the Bill leg just got — check whether it's
correctly targeting the Vermilion harbour/ship vs touring the wrong Cerulean building, and whether the
grind-cycle is eating the clock. NOT a wedge (TIMEOUT), so re-run longer first to see if it self-resolves.
KNOWN watchability debt (LOW-pri, from the misty_done run): the underleveled bench (Rattata L8) grind-dies
on trainer gauntlets across a few cycles before prep-floor L14 clears — door-burn-clear + heal-gate make
it eventually SUCCEED, but a smart human fields Ivysaur solo across a gauntlet, not grinds L8 mons into
forced fights. Consider tuning strategic underlevel-prep to not grind-wall when the ACE can clear solo.
Boot: `.venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done.state 20` (arg2=MINUTES).
═══════════════════════════════════════════════════════════════════════════════════════════════ -->

# 🧠 MISSION PIVOT (2026-07-09) — BUILD THE FORWARD-PLANNING TEAM BRAIN — READ FIRST, SUPERSEDES ALL BELOW

**THE fix (CEO-approved):** stop reactive per-gym patching. Build a STANDING forward-planning team-building
brain so Kira plays like a guide-literate human who plans her whole team toward the Elite Four from the
start — catches/evolves/teaches/levels DELIBERATELY IN ADVANCE, voiced in character, at a watchable pace,
to a fresh bedroom→credits win. **END STATE:** press START → watch her plan, catch, build, evolve, and win
her way to credits like a smart lovable human who read the guide.

**FULL SPEC:** `pokemon_agent/TEAM_PLANNER_DESIGN.md` (schemas, class API, wiring, verification GATES).
Read it first every shift.

**BUILD IN DEPENDENCY ORDER — VERIFY EACH LAYER ON DISK BEFORE STACKING THE NEXT:**
- **PART A — DEEP KB** (`gamedata/`): `frlg_rosters.json` (full per-mon gym/E4/rival teams — verify ≥3 vs
  live RAM via read_enemy_species), `frlg_evolutions.json` (species→method/level/into), `frlg_learnsets.json`
  (level-up + TM compat; team-relevant species first), `frlg_tms.json` (all 50 TMs + 8 HMs), a route→species
  encounters table, and **`frlg_team_plan.json`** (the curated whole-game balanced-6 archetype: coverage map
  + acquisition ORDER/locations + level milestones). Author from FRLG knowledge + Bulbapedia (pret NOT local;
  use WebSearch/WebFetch), verify against live RAM. Gates A1-A6 in the spec.
- **PART B — THE BRAIN** (`pokemon_planner.py`): a standing `TeamPlanner` with persistent plan-state
  (target team, acquired-vs-needed, next acquisition + WHERE, evolve/level targets), **whole-game lookahead**
  (union of every future threat incl. E4 → earliest-due, highest-multiplicity gap → "grab Abra now, it
  carries Koga+Sabrina+Bruno"), emitting PROACTIVE PlanActions. Persist to the campaign bundle
  (dev-line only). Gate B = deterministic tests + resume.
- **PART C — THE EXECUTOR** (free_roam + spine): run the next PlanAction BEFORE the wall — targeted catch
  (route to the keeper), deliberate evolve (B-to-cancel for move timing, stone timing), teach TM, grind to
  milestone. prep_for_gym becomes the last-resort safety net. Gate C = a real run: proactively prepared,
  wins, keeps moving, watchable.
- **FINAL PROOF:** fresh bedroom→credits, built balanced 6, no permanent stuck, watchable → write CREDITS.

**SOUL (core requirement, DO NOT STRIP):** every plan decision flows through the `plan_note` voice seam as
HER forward-looking idea (names mons, has favorites, excited-guide-literate-kid). The archetype is a menu she
CHOOSES from, not a solver dictating. **WATCHABILITY (core):** level targets = ace+margin, bounded/narrated
detours, HARD grind cap, "win most fights / keep moving" — a grind-wall is a FAILURE.

**FIREWALL:** mode-side Pokémon brain ONLY. Core Kira general personality + persona + two-bucket firewall
OFF-LIMITS (`plan_note` is the ONLY voice interface). NEVER write states/kira/. Commit-per-fix. VERIFIED
from disk/real runs. Run-until-credits + escalate is already set (`night_shift.ps1 $RunUntilCredits`).

**Prior reactive frontier (Sabrina, below) is now the FALLBACK context, not the mission — the brain is the mission.**

---

# TEAM-BRAIN BUILD — FRONTIER (2026-07-09, mission-pivot shift IN FLIGHT) — READ FIRST

## PART A (deep KB) — ✅ DONE + COMMITTED (2549dac). gamedata/ now holds frlg_rosters.json,
## frlg_evolutions.json (68 lines), frlg_learnsets.json (22 species), frlg_tms.json (50TM+7HM),
## frlg_encounters.json, frlg_team_plan.json (balanced-classic archetype). ALL GATES A1-A6 PASS on
## disk — re-run `.venv/Scripts/python.exe -u pokemon_agent/recon_kb_verify.py` to confirm. Rosters
## Bulbapedia-verified (E4/Champion/gyms fetched live 2026-07-09; Champion = Bulbasaur branch).
##
## PART B (the brain, pokemon_planner.py TeamPlanner) — ✅ DONE + COMMITTED (55602d6). Persistent
## plan-state + whole-game lookahead + assess()/next_action() PlanActions (catch_keeper / acquire_special
## / evolve / grind_to / teach_tm / develop_bench) + first-person plan_note voice. GATE B PASS: 9
## deterministic tests + save/resume identity — re-run `.venv/Scripts/python.exe -u
## pokemon_agent/recon_teamplanner_test.py`. On the REAL Sherpa state (Venusaur L52 solo, badge 4) the
## brain diagnoses the solo-carry failure and prescribes 'catch an Abra — serves Koga AND Sabrina'.
##
## PART C (executor) — VOICE HALF ✅ (19f7940). EXECUTOR #1 TARGETED CATCH ✅ + #2 PRE-BUILD (load-bearing) ✅
## + #3 PERSISTENCE ✅ — all DONE + COMMITTED (mission-pivot shift 2, THIS shift; 2 commits after 19f7940):
##   #1 TARGETED CATCH: catch_one(target_species) — FLEE non-targets (no ball/PP waste), FORCE-catch the
##      target (judgment bypassed). free_roam's wander_catch consults the brain via _plan_keeper_target:
##      a DUE keeper that lives on the CURRENT map (frlg_encounters _species_on_map) -> SEEK it, not filler.
##   #2 PRE-BUILD DOMINANCE (rule 2 — makes the brain LOAD-BEARING): _plan_wants_prebuild — when the brain
##      says a keeper/bench-build is DUE and the team is dangerously THIN (<=2 mons), (a) don't prune
##      catch/grind to forward-drive, (b) SUPPRESS head_to_gym while a build option exists here. Bounded:
##      <=2 -> self-clears at party 3; never freezes. This fixed the core failure: forward-drive was
##      solo-charging the Nugget Bridge -> blackout -> road self-gates.
##   #3 PERSISTENCE (rule 17): team_planner save/load in _continuity_save/_load + team_plan_state.json in
##      the auto-ckpt bundle.
## GATE C VERIFIED-PARTIAL (recon_longrun misty_done.state, gatec_misty3.log): solo Ivysaur L22 @ Cerulean,
## brain voices "catch Abra" -> PRE-BUILD suppresses the solo bridge-charge -> she routes WEST to ungated
## Route 4 grass -> CATCHES Spearow L8 then Rattata L8 (party 1->3) -> PRE-BUILD self-clears -> forward-drive
## resumes toward Bill. THE BUILD-FIRST-THEN-CROSS SEQUENCE THE BRAIN PRESCRIBES, WATCHABLE. (Note: she
## caught FILLER not Abra because Abra's grass (Route 24) is NORTH past the Nugget Bridge — the reachable
## grass at Cerulean is Route 4 WEST. Targeted-Abra fires once she's ON Route 24, which is the NEXT blocker.)
##
## NEXT BLOCKER (Gate C completion): after building, the Bill questline strike walked her from Cerulean's
## WEST edge (0,22) INTO a dead-end house map (7,3) and STALLED (14 no-progress decisions; leave_building
## was offered but the oracle picked talk_npc). PRE-EXISTING door-passthrough/approach wedge (not a
## regression — PRE-BUILD only fires <=2 mons; mid-game fixtures untouched), surfaced because the Route-4
## build detour changed her Bill-approach position. To FINISH Gate C: (1) fix the (7,3) house-exit /
## Bill-approach so she crosses the Nugget Bridge onto Route 24; (2) on Route 24 the targeted catch fires
## -> she catches ABRA (Phase 1's payoff); (3) beats the bridge with a bench -> Bill -> S.S. Ticket ->
## forward. Boot: recon_longrun.py misty_done.state 13 (read gatec_misty3.log seams). Consider: have
## PRE-BUILD prefer a build route that doesn't strand her far off the forward path, OR fix the door-
## passthrough to exit a dead-end interior via a learned warp when head_to_gym reads questline_no_route.
##
## (superseded log of remaining work below):
##   1. CATCH_KEEPER(species, where): route to the encounter location (frlg_encounters keepers/areas) ->
##      TARGETED catch (extend the existing catch_one to SEEK a specific species, keep it). Reuse travel +
##      catch machinery already in campaign.py. This is the #1 piece (unblocks Abra/Diglett/Growlithe).
##   2. Wire a PROACTIVE PLAN step into free_roam: at town arrivals / each tick, if next_action is a DUE
##      catch/evolve/teach and cheaply reachable, DO it, then push to the gym. prep_for_gym becomes the
##      last-resort net.
##   3. GATE C = a real recon_longrun where she PROACTIVELY catches Abra before Koga, arrives prepared,
##      wins, keeps moving (watchable). Then the FINAL PROOF = fresh bedroom->credits.
##   4. PLAN-STATE PERSISTENCE (rule 17, do it WITH the executor): add ("team_plan_state.json", ...) to
##      the sanctity bundle list (campaign.py:9745) + call self.team_planner.save(STATES_CAMPAIGN) in
##      _continuity_save and .load() on resume — so the taught/acquired HISTORY survives a hard kill.
##      DEFERRED this shift on purpose: history isn't load-bearing until the executor consumes it, and
##      the brain re-derives slot-status from the live party each tick, so resume already works today.
## RISK NOTE: Part C drive touches free_roam nav arbitration (the shared plumbing that burned prior
## shifts) — build it isolated/guarded behind POKEMON_TEAM_PLANNER, verify each piece with a look-ahead.
## The catch primitive to extend is catch_one (campaign.py:3957) — add a target_species param that
## FLEES non-targets + force-catches the target (roster_judgment stays for untargeted forward-catch).
## BOOT: `.venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py fuchsia_gate.state 15` (Venusaur L52
## @ badge 4, the brain will voice 'catch Abra'); read the [teamplan] fold in the log.
##
## FALLBACK reactive context (Koga/Sabrina, potion-stall) is below — NOT the mission. The brain is.

---

# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 10 IN FLIGHT) — reactive fallback context

## SHIFT-10 KEY FINDING (verified e2e): **POTIONS BEAT KOGA.** Venusaur L51 + Hyper/Super Potions
## stall-and-wins Koga cleanly (in-battle actuation FIRES on the long core). Proven by injecting
## potions into the bag (`inject_potions.py`) -> `recon_longrun fuchsia_potions.state`:
##   -> *** KOGA BADGE obtained *** -> advanced to Saffron -> reached **SABRINA's gym (badge 6)**
##   -> STALLED on Sabrina's gym-layout puzzle + coverage-mon indecision (alakazam psychic wall).
## So the potion-stall strategy carries her a WHOLE gym past the shift-9 frontier. NEW frontier = SABRINA.

## THE REMAINING WORK TO MAKE THE KOGA WIN REAL (unaided): she won with INJECTED potions. To bank a
## genuine badge-5 checkpoint she must BUY potions at Fuchsia Mart (currently UNMAPPED — the
## Cerulean-Mart-unmapped class). IN PROGRESS this shift:
##   1. recon_fuchsia_mart.py — identify the Fuchsia Mart door among the learned building warps
##      (candidates (24,5),(11,15),(28,16),(14,31),(38,31),(19,31); gym=(9,32)->int(11,3),
##      PC=(25,31)->int(11,5) already known). Add FUCHSIA_MART_DOOR to CITY_MART_DOORS + MART_STOCK
##      rows (control-verify by a live bag-delta buy, like Cerulean/Vermilion).
##   2. Wire GYM-PREP (or the roam stock_up) to actually BUY potions at Fuchsia before entering Koga.
##      NOTE: in the potion run she went STRAIGHT from Fuchsia arrival to head_to_gym (never picked
##      stock_up) — so GYM-PREP must FORCE a foresight potion stock-up when walled at a gym city and
##      the bag is potion-poor. `_shopping_list(foresight=True)` + `buy_at_mart` already exist.
##   3. VERIFY: a FRESH (uninjected) `recon_longrun fuchsia_gate.state` buys potions -> beats Koga.
##      Then bank a real post-Koga checkpoint.

## SECONDARY (heal ping-pong, pre-existing, LOW priority): at boot fuchsia_gate she's hurt inside the
## Route-15 gate; heal_nearest can't reach Fuchsia's Center (split map) so it falls to the Viridian
## march (return_to_center) which PING-PONGS the gate floors ~5 legs before the range(20) cap returns
## 'stuck' and the shift-9 heal-dead breaker fires -> she pushes to the gym. WORKS but ugly/slow.
## Real fix (if time): teach heal_nearest that Fuchsia's Center is WEST across the internal gate
## (route via the gate warp like head_to_gym does) instead of a cross-region Viridian march.

## THE SABRINA FRONTIER (badge 6, after Koga is real): Venusaur (Grass/Poison) takes PSYCHIC x2 —
## dangerous vs Alakazam (fast, Psychic). Potions may not out-stall a fast special sweeper. Human
## answer per her own ctx: a BUG or GHOST or DARK coverage mon (she NARRATES wanting one). Also the
## Saffron gym is a warp-pad/teleporter puzzle (pad_nav exists). Ranked plan for the successor:
##   A. Try potions-stall vs Sabrina first (cheap; maybe Razor Leaf + Sleep Powder + Hyper Potion
##      out-damages before Alakazam sweeps — verify with LONGRUN_BATTLE_LOG=1).
##   B. If potions insufficient: catch a coverage mon (bug/ghost) — she already wants one.

## BOOT THE FRONTIER
- Koga acquisition work: `.venv\Scripts\python.exe -u pokemon_agent\recon_fuchsia_mart.py` (door probe),
  then edit CITY_MART_DOORS/MART_STOCK, then
  `LONGRUN_BATTLE_LOG=1 .venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 15`.
- Sabrina probe (once Koga is real): boot the potion run's banked Saffron state or re-run from
  fuchsia_potions and read the Sabrina battle log.

## KEY FACTS (carried from shift 9, still valid)
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages to G:/temp/longrun/stage (WIPED each run); banks to banked_<OUTCOME>. Canonical
  Champion save NEVER touched. `LONGRUN_BATTLE_LOG=1` surfaces per-move engine logs.
- Fixtures (states/workshop/): erika_done, snorlax_face, snorlax_done, fuchsia_gate (boots inside the
  Route-15 gate; heal-breaker crosses it -> Fuchsia -> Koga), **fuchsia_potions (fuchsia_gate + injected
  potions; PROVES the potion win — beats Koga -> reaches Sabrina)**.
- Koga team (FRLG): Koffing L37, Muk L39, Koffing L37, Weezing L43 — all POISON. Fuchsia gym juniors
  (arbok/sandslash) DO engage now and give XP (Venusaur L51->L52). NUKE-SLEEP handles the Self-Destruct
  koffing/weezing; Cut (neutral) + Razor Leaf chips; Hyper Potion out-stalls the poison chip.
- Item ids: Super Potion 22, Hyper Potion 21, Max Potion 20, Full Restore 19, Potion 13, Revive 24,
  Antidote 11. In-battle heal instinct fires at <=30% HP (<=50% vs super-effective foe); chooser picks
  use_potion. BATTLE_CRIT_FRAC=0.30 (battle_agent.py).
- Bag write: SaveBlock1 + 0x0310, 42 slots x4 (id u16, qty u16 ^ key); key = rd32(rd32(SB2)+0xF20)&0xFFFF.
  Write via `b.core.memory.u16.raw_write(addr, val)` (the `[addr]=` path is broken in this mgba build).

## DURABLE PATTERNS
- POTION-STALL beats a MOVEPOOL-WALL gym (Koga): when a solo carry's STAB is resisted but it has a
  neutral chip move (Cut) + a status lock (Sleep Powder vs Self-Destruct) + Hyper Potions, it out-lasts
  a poison tank team. The in-battle item instinct + recon chooser make it autonomous — the ONLY missing
  piece is ACQUIRING the potions (map the town Mart).
- INTERNAL MAP-SPLIT class (Route 12/15 gates): halves share one map id, connect only via a gate
  building; the map-granular graph is blind to it, so from the wrong side edge/center/GRASS bands read
  UNREACHABLE. head_to_gym owns the crossing (warp-routing); heal_nearest does NOT (falls to Viridian).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead PROVEN to clear Koga
(badge 5) with potions and reach Sabrina (badge 6). Making it unaided (buy potions at Fuchsia) is the
in-flight work. Pop-in = `python pokemon_agent/watch.py`.
