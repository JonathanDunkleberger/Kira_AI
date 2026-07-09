# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 6 in flight)

## SHIFT 6 HEAD (read FIRST — supersedes shift 5 below)

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
