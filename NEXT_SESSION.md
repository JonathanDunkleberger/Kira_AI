# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 8 entry)

## SHIFT 8 HEAD (read FIRST — supersedes everything below)

**RESULT: THE ENTIRE BADGE-3 APPROACH IS UNWEDGED (FIX A, commit a3900d6). She now crosses BOTH hard
crossings for the FIRST time ever: Cerulean->Route 5 AND the Underground Path Route 5->Route 6.** She is
on Route 6 heading to Vermilion in `logs/debug/crossing_fixA.log` (grep `PASSTHROUGH: CROSSED via door
(30, 11)` = Cerulean fence past the Cut tree; `CROSSED via door (31, 31): (3,23)->(3,24)` = the UGP to
Route 6). 3 commits this shift: 14d327f (LONGRUN_FORCE), 959eb75 (push-when-carrying chooser), a3900d6
(same-map fence-crossing memory — the big unwedge). The shift-7 billed road (e932996) is VERIFIED sound.

**SHE CLEARED THE WHOLE APPROACH AUTONOMOUSLY — reached the S.S. Anne.** FIX A cascaded far past the
crossing: a brief Route6<->UGP oscillation was transient navigation (NOT a wedge — she resolved it),
then she pushed SOUTH to Vermilion, the **`cut` QUESTLINE FIRED** at the gym's Cut-tree gate
("🧭 QUESTLINE STEP: HM Cut — the seasick captain of the S.S. Anne... heading SOUTH toward the S.S. Anne,
docked at Vermilion's harbour, board with the S.S. Ticket, captain's cabin past the rival fight"), and
she **BOARDED the S.S. Anne** (`STATE IN: the S.S. Anne`) and is navigating it toward the captain. This
is autonomous badge-3-questline execution — the machinery works.

**NEXT FRONTIER = INSIDE the S.S. Anne (HM01 Cut acquisition):** read `logs/debug/crossing_fixA.log`
(grep `S.S. Anne`, `HM01`, `FLAG_GOT_HM01`, `rival`, `Gary`, `questline`, `STALL`, `Surge`) for how far
she got. Watch these machinery points: (a) navigating the S.S. Anne decks to the captain's cabin at the
bow; (b) the RIVAL (Gary) fight before the cabin (a real battle — her thin bench may struggle); (c) the
captain -> HM01 Cut -> teach it -> back out -> Cut the Vermilion gym's tree gate; (d) Lt. Surge (badge 3,
L21-24 electric — her bench rattata/spearow L14-15 vs Surge is thin; may need a real grind or a ground-
type, a SEPARATE under-level finding, NOT a machinery bug — the S.S. Anne's trainers + rival grind her on
the way). Continue from the furthest banked point; diagnose the next machinery wedge -> fix general -> re-run.

---

**HISTORICAL (shift-8 diagnosis, kept for context): billed road (e932996) reached Cerulean but she never
crossed to Route 5.** Two blockers were found:

Read shift-7's 4 verify logs (vermilion*_verify.log, 3:49–4:07 AM). Findings:
- The `ss_ticket.state` boots INSIDE Bill's house `map=(30,0)@(7,7)` (a bad bank point — an interior).
- The road fix DOES fire: vermilion3/4 show `ROAD to Vermilion City ... warped (30,0)->(3,44)` — she
  exits Bill's house onto Route 25. So the KB road is billed + `_road_step` executes the first hop. GOOD.
- **BLOCKER A (routing, UNVERIFIED end-to-end):** past the building exit she lands on Route 25 / Cerulean
  and `head_to_gym` from off-road spots returns `no_path`/`no_gym_route` in the logs (e.g. from Route 4
  (3,22)(107,12)). Whether the billed road actually COMPLETES Cerulean->Route5->UGP->Route6->Vermilion
  when routing is isolated was NEVER observed — because of Blocker B.
- **BLOCKER B (harness grind-preference, EXPECTED):** recon_longrun's FAITHFUL chooser (recon_longrun.py
  ~L345, NOT the live oracle) deliberately picks stock_up -> wander_catch (team<4) -> battle (underlevel
  grind) BEFORE head_to_gym. Her bench is L14 vs Surge L21-24 + party of 3, so the chooser grinds forever
  (even grinding a fresh L6 catch) and the run times out on Route 4 (3,22). This is a HARNESS artifact of
  the deterministic chooser — the LIVE oracle gets "GET BACK ON THE ROAD to Vermilion; never circle the
  same grass" ctx and may push forward. So it does NOT prove a live wedge; it just BLINDS the routing test.

**BILLED VERMILION ROAD = VERIFIED SOUND (commit 14d327f).** The `LONGRUN_FORCE=head_to_gym` routing-
isolation probe (logs/debug/road_isolate.log) carried her: Bill's house (30,0) -> Route25 (3,44) [warp
out] -> Route24 (3,43) -> Cerulean (3,3) -> **billed south leg -> burgled-house door-passthrough (door
(30,11) -> gatehouse (7,1) -> Cerulean south (31,8))** -> heading south to Route 5 (3,23), fighting the
Route-5 trainer. Then it need_heal-looped (forcing head_to_gym BLOCKS the heal action — a probe artifact,
NOT a road bug). CONCLUSION: routing is sound through the gatehouse toward Route 5; Blocker A (the
`no_path`/`no_gym_route` in shift-7's logs) was just the off-road-anchor flake + the grind-preference
never letting her commit — NOT a nav bug.

**FIDELITY FIX BUILT (uncommitted until verified):** recon_longrun.py chooser now has a PUSH-WHEN-CARRYING
branch — when a billed road exists + head_to_gym is offered + the LEAD out-levels the stretch prep target
(Ivysaur L28 >> target 20), a sensible player MARCHES the road (its trainers + S.S. Anne + gym ARE the
grind) instead of parking in wild grass. This lets the DEFAULT look-ahead reach S.S. Anne / Cut / Surge.

**TRUE FRONTIER FOUND (definitive, logs/debug/stretch_surge.log — the improved-chooser run):** she now
PUSHES the road (no more wild-grind blindspot) all the way to Cerulean and REPEATEDLY reaches the
Cerulean->Route5 crossing — but NEVER breaks through to Route 5. She oscillates Cerulean<->Route4 grinding
forever. **This is the wedge for the whole Vermilion->Surge->credits chain.**

**THE CROSSING MECHANICS (ground-truthed this shift, L162-206 of stretch_surge.log):**
- Cerulean(3,3)'s ONLY south gap to Route 5(3,23) is at col 26, **blocked by a CUT tree at (26,32)**
  (`no_route_hm_blocked`). She has no Cut yet (Cut comes from the S.S. Anne — chicken/egg, but the
  pre-Cut route EXISTS).
- The pre-Cut bypass: `_door_passthrough` enters the burgled-house/gatehouse **door (30,11)** -> warp to
  gate (7,1) -> pops back onto Cerulean at **(31,8)**, a within-map "fence crossing" PAST the tree region.
  From (31,8) travel south finds a path (~len 46) blocked by a **TRAINER at (33,7)**.
- She BEATS that trainer (L203 `blocker battle outcome=win`) — trainers don't respawn — but the fight
  leaves her hurt -> `head_to_gym -> need_heal` (L206). She heals at the Cerulean Center, returns north
  of the tree, and must RE-CROSS. On the re-cross the passthrough does NOT re-use the proven (30,11) door
  (see FIX A) -> flails other doors ("popped out beside the entry - not a crossing") -> `no_gym_route`
  -> after 2 strikes head_to_gym is PRUNED as a structural dead-route on Cerulean -> she can only
  grind/catch -> wild-grind loop (L528-1336 = all wild battles, zero forward progress).

**TWO CONCRETE FIX HYPOTHESES (pick/combine — VERIFY each with a re-run):**
- **FIX A (passthrough memory — APPLIED this shift, VERIFY IN FLIGHT):** `campaign.py` ~1649 only stored a
  crossing in `_pt_known` when `m_out == want_map`. The Cerulean fence crossing lands on the SAME map
  (m_out=(3,3), want_map=Route5 (3,23)), so the PROVEN (30,11) door was NEVER remembered -> every retry
  re-searched + picked a wrong door. FIX: also remember a same-map crossing (`m_out == m0`) that passed the
  d_door>3 & moved>6 guard, so retries re-use it (self-corrects via the "forgot poisoned connector" branch).
  RE-RUN `recon_longrun.py ss_ticket.state 20` -> new log; grep for `STATE IN: Route 5`/`Route 6`/`Vermilion`
  (NONE appeared pre-fix). If she crosses to Route 5, FIX A worked — carry on to the S.S. Anne stretch. If
  she STILL loops Cerulean<->Route4, FIX A wasn't enough -> try FIX B (field the ace through the crossing +
  don't prune head_to_gym on a map holding an active billed-road leg).
- **FIX B (field the ACE through a forward crossing):** she fields WEAK bench mons (strategic-grind) into
  the crossing trainers -> they faint -> need_heal interrupt mid-crossing. On a FORWARD push through a
  billed-road trainer, field the ace (Ivysaur L29 one-shots these) so she punches through in one go, no
  heal-yank. Also consider: don't prune head_to_gym on a map that holds an ACTIVE billed-road leg (the
  crossing is transiently failing, not structurally dead).
- **Cheapest unblock to test the DOWNSTREAM stretch first:** `LONGRUN_FORCE=head_to_gym` won't work (blocks
  heal). Instead bank a state already ON Route 5/6 (or inject Cut) to leapfrog the crossing and verify
  S.S. Anne boarding / HM01 Cut / Surge machinery in parallel while fixing the crossing.

**RE-RUN after a fix:** `python recon_longrun.py ss_ticket.state 20` -> a NEW log; confirm a
`STATE IN: Route 5` / `Route 6` / `Vermilion` line appears (grep it — none did this shift). Then the next
stretch is S.S. Anne (board w/ ticket, rival fight, captain -> HM01 Cut) -> Cut the gym tree -> Lt. Surge
(L21-24 electric; her thin bench may need a real grind or a ground-type = a SEPARATE finding, not a bug).
KB prior art: frlg_gates.json `cerulean_to_vermilion`, `cut` obtain, roads["Vermilion City"].

**SHIFT 8 COMMITS (proof of life):** 14d327f (LONGRUN_FORCE routing-isolation hook), 959eb75
(push-when-carrying chooser — the DEFAULT look-ahead now marches a billed road w/ a strong lead instead of
parking to wild-grind; this is what SURFACED the crossing wedge — prior shifts' grind-preference hid it).
The shift-7 billed road (e932996) is VERIFIED sound up to the crossing. NO crossing fix landed yet.

---

## SHIFT 7 HEAD (historical — the road-billing fix)

**SOUTH-ROUTING ROOT CAUSE FOUND + FIXED (e932996) — VERIFY IN FLIGHT. Bill's-house -> Vermilion.**

**ROOT CAUSE (airtight, not the shift-6 guess):** the shift-6 diagnosis ("wrong world-graph edge
Route4->(3,21)") was a symptom. The REAL cause: `head_to_gym` could not route Cerulean->Vermilion
because the sparse CANONICAL world_model.json never visited the ONLY early crossing — the **Underground
Path** (nodes 17,0/17,1/18,0 are empty unvisited stubs; Route 5's south edge dead-ends at Saffron 3,11
which is gate-blocked pre-Tea; Saffron gates refuse passage). `world.route(Cerulean,Vermilion)` = NO
ROUTE -> `head_to_gym -> no_gym_route`. The BILLED-ROAD fallback (`_gym_road`, reads
`gamedata/frlg_gates.json` "roads") is the intended driver for exactly this "graph can't route yet"
case — but the roads KB had every LATER-gym road (Celadon/Fuchsia/Saffron/Cinnabar/Viridian) and was
**MISSING "Vermilion City"**. So the fallback no-op'd and she oscillated Cerulean<->Route 4 grinding
(ekans->Arbok etc.), never going south. `travel(target_map=(3,21))` = Route 3 (a backward grind
excursion), a red herring.

**FIX (e932996, committed):** billed the "Vermilion City" road in `frlg_gates.json` — the southbound
reverse of the PROVEN Celadon road: Cerulean(3,3) -S-> Route 5(3,23) -S(via=pass)-> Underground Path ->
Route 6(3,24) -S-> Vermilion(3,5). Pure game-knowledge-layer add (gamedata), engine untouched. The
`via:pass` leg fires `_door_passthrough` on Route 5 toward Route 6 (the reverse crossing Route6->Route5
is logged "proven both directions 2026-07-06").

**SHIFT 7 ACTION (IN FLIGHT):** re-run `recon_longrun.py ss_ticket.state 40` ->
`logs/debug/vermilion3_verify.log`. EXPECTED: head_to_gym now hits the billed road ->
Cerulean -S-> Route 5 -> (Underground Path door-passthrough) -> Route 6 -> Vermilion -> board S.S.
Anne (has ticket) -> HM01 Cut -> Cut the tree -> Lt. Surge (badge 3). READ that log FIRST. If she
reaches Vermilion, the NEXT wedge is S.S. Anne boarding / HM01 Cut teach+use / Surge under-level.
If she STILL wedges before Route 5, grep the log for `ROAD to Vermilion` and `_door_passthrough` /
`road_gated` — the pass-leg may not cross Route5->Route6 southbound (only the northbound was proven);
if so, inspect the Route 5 Underground Path hut warp reachability (recon_route24.py is the geography
probe template). Watch too for Surge needing a GROUND move (her bench has no clean electric answer).

---

## SHIFT 6 HEAD (historical — the Route24->Bill win)

**ROUTE24->25->BILL FIX (7152af0) VERIFIED END-TO-END — S.S. TICKET OBTAINED.** The re-run
(`logs/debug/route25_verify.log`, OUTCOME: GOAL — FLAG_GOT_SS_TICKET set) proved the whole chain:
`QUESTLINE EXPLORE ... crossing E into known-but-forward (3, 44)` (the new visited-fallback) ->
arrived on Route 25 (NOT the Route-24 misfire) -> entered Bill's cottage door (51,4)->(30,0) ->
talked Bill -> **questline cleared (flag satisfied)**. Survived a mid-climb blackout+recovery and
still completed. The fix generalizes to the WHOLE look-ahead climb: recon_longrun loads the CANONICAL
world (every forward map pre-visited), so every stretch past here is "pre-mapped-forward" and now routes.
Post-Bill state banked -> **`states/workshop/ss_ticket.state`** (party ivysaur L27/rattata L14/spearow
L14, badge 2, ticket in bag). Canonical UNTOUCHED (sanctity refused badge 8->2, expected).

**SHIFT 6 NEXT STRIKE IN FLIGHT:** `recon_longrun.py ss_ticket.state <min>` -> `logs/debug/vermilion_verify.log`.
EXPECTED: ticket opens Cerulean's south gate -> Route 5 -> Route 6 -> Vermilion -> board S.S. Anne
(has ticket) -> HM01 Cut from the captain -> Cut the tree -> Lt. Surge gym (badge 3). READ that log,
find the next wedge, diagnose -> fix general -> re-run. (Surge needs a ground move / her bench is thin
L14 vs ace — watch for an under-level wall at Surge; the canonical climb + recon_bill_*.py are prior art.)

---

**BILLFIX (1b851bb) VERIFIED WORKING.** The shift-5 `recon_longrun.py rattata_grind.state 30`
(`logs/debug/billfix_verify.log`) proved it: she re-anchored Route 4 -> Cerulean -> crossed NORTH
onto **Route 24 (map 3,43) — the Nugget Bridge** (Ivysaur grinded the bridge trainers L24 -> Venusaur
L33). The shift-4/5 Cerulean north-crossing wedge is DEAD.

**NEXT WEDGE FOUND + FIXED THIS SHIFT — the Route 24 -> Route 25 "pre-mapped forward" arrival misfire.**
ROOT CAUSE (airtight): on Route 24 the S.S.-Ticket questline ("Bill is north of Cerulean") finds no
NORTH header edge (Route 24's only exits are S=back to Cerulean + **E -> Route 25 -> Bill**), so it
falls to the BEND-DISCOVERY frontier-explore. That branch keyed "forward" off `not world.visited(nb)`
— which holds on a FRESH world but NOT here: **recon_longrun.py:223 loads the CANONICAL world model**
(`C.WORLD_JSON`), built by the Champion run that visited EVERY map incl. Route 25/Bill. So the E ->
Route 25 edge reads VISITED, the unvisited-frontier is empty, `_ql_bend_maps` never populates,
BEND-CONTINUE can't fire, the FALSE-ARRIVAL GUARD is skipped (she IS past the anchor), and she
mis-declares **"arrived at the destination area ((3,43), no further forward edge)"** -> parks
head_to_gym "for the session" -> falls back to aimless wander+grind in a 3x3 grass pocket at
Route 24 cols 3-5 (Venusaur ground to L33 doing nothing). Bill was one map EAST the whole time.
(Same class bites late-game backtrack fetch-quests over already-mapped ground.)

FIX (applied, syntax-checked, VERIFY IN FLIGHT — `campaign.py` ~5153): when past the anchor with no
step-dir header edge AND no UNVISITED frontier, fall back to crossing a non-back connection we did NOT
just come from (anti-ping-pong via new `_ql_prev_map`), preferring the coarse-dir edge, recording it
as a bend. On Route 24 that crosses E into Route 25 (records the bend); on Route 25 the W edge is
excluded (=came-from) so the frontier empties -> she correctly "arrives" -> `_questline_interact`
enters Bill's house door -> talks Bill -> FLAG_GOT_SS_TICKET. General; fresh-world path unchanged
(unvisited frontier fires first). `_ql_prev_map` resets per-step + per-questline alongside `_ql_bend_maps`.

**SHIFT 6 ACTION IN PROGRESS:** re-run `recon_longrun.py rattata_grind.state <min>` ->
`logs/debug/route25_verify.log`. EXPECTED: re-anchor -> Route 24 (Nugget Bridge) -> **cross E to
Route 25** -> enter Bill's cottage -> S.S. Ticket -> back to Cerulean/Route 5 -> Vermilion. READ that
log FIRST. If she gets the Ticket, next wedge is Route 5/6 / Vermilion / S.S. Anne (Cut) / Surge —
recon_bill_*.py and the canonical climb are prior art. If she STILL wedges on Route 24, inspect the
tick trace for whether the E frontier-cross fired (grep "QUESTLINE EXPLORE ... known-but-forward").

---

## FRONTIER (do in order)

1. **CONFIRM the Route 24 -> Route 25 -> Bill crossing** (read route25_verify.log per above), then find
   the NEXT wedge past Bill (Route 5-6 / Vermilion / S.S. Anne-Cut / Surge). Diagnose -> fix general ->
   re-run. Iterate toward credits.
2. **WIRE THE HANDOFF INTO `play_live --show`** (unchanged from shift 5 — see below). Note the world-model
   subtlety this shift surfaced: the REAL show timeline uses a FRESH world (accretes as she walks), so
   the visited-frontier fallback won't fire there (unvisited frontier fires first) — the fix is
   look-ahead-faithfulness + late-game-backtrack robustness, NOT a live-show behavior change. Still,
   if the show ever loads a pre-mapped world on resume, the fix now covers it.
3. THEN: Phase B (bank the clean audio-OFF floor once fresh->credits is proven) + Phase C (native audio
   SIGSEGV capture via faulthandler `--audio`) remain untouched.

## KEY FACTS / TOOLS

- **Look-ahead oracle** (default verification): `python pokemon_agent/recon_longrun.py <state> <min>`
  (or `FRESH <min>` for the show spine). Non-FRESH boot = free_roam from a named state, loads the
  CANONICAL world model (`C.WORLD_JSON`) — which is WHY visited-forward maps looked "arrived" (now fixed).
- **recon_route24.py** = the prior-art geography probe (dumps each map's RAW header connection table +
  warps) — built to answer "does Route 24 have an N edge?" (answer: no — S+E only; forward is E->Route25).
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = a 2-PID shim — never taskkill your own
  run. Nothing launched within ~40 min of a handover.
- **states/campaign = SHERPA CANONICAL (untouchable). states/kira = the show timeline. states/workshop
  = scratch** (recon_longrun banks segment checkpoints + rattata_grind.state here). Canonical is
  STAGE-redirected in the look-ahead (`_save_campaign` -> a staging dir).
- Commit per fix (lowercase `git add pokemon_agent/`); VERIFIED not asserted; read the latest log not
  memory. PS 5.1: `*>` logs are UTF-16; edit THIS file with Write/Edit only.

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS / AUDIO-PLUMBING FIXES ONLY.** Core Kira identity / voice / oracle / memory / vision
  are sacred + OFF-LIMITS. Mode-state stays behind the Pokémon toggle. This shift's edit is pure
  mode-side questline-nav (campaign.py `_run_questline_step`) — additive, fresh-world path unchanged.
- **AUDIO END STATE = ON** (`POKEMON_GAME_AUDIO=1`); audio-off is only the committed floor + fallback.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS

(a) clean bedroom->credits with audio ON demonstrated (full deliverable); OR (b) ~80-85% context ->
clean handoff (rule 11) / two-consecutive-no-progress brake; OR (c) balance exhausted.

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched by this shift's nav work);
the look-ahead now crosses bedroom -> Brock -> Misty -> Cerulean -> Nugget Bridge and (fix in flight)
should reach Bill's cottage for the S.S. Ticket. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
