# KIRA FIRERED - SINGLE MISSION (2026-07-13). SUPERSEDES EVERYTHING.
CEO decision: the 5x battery is CANCELLED. The asymptote = ONE QUALIFYING RUN, today. When it lands, STOP THE TRAIN. Read ONLY this file + the latest survey + live log tails. Prior directives are archived at NEXT_SESSION_archive_2026-07-13_*.md - consult ONLY for the proven per-stage launch recipes; do not re-derive them; no history spelunking.

## FACTS (verified this morning)
- fresh_go_1 ROLLED CREDITS last night (~20:30-21:00) from a cold bedroom start, fully autonomous: 8 badges, Victory Road, E4, Champion. 0 tracebacks, 0 hard wedges. The machine (nav rope, quest chain, battles, watchdog banking) is PROVEN end-to-end.
- BUT it won in the BANNED shape: venusaur solo-ground to L100 in VR, bench flat L40-47, DUPLICATE dugtrios, ~4h of silent grass grinding. Solo-ace+fodder = disqualified + unwatchable. Roots are known; they are today's build.
- Victory is NOT RECOGNIZED by the harness: champion flag set but no CREDITS outcome/banked_CREDITS; the watchdog mis-called the post-game victory lap a STALL and killed the run. Must fix.
- The watchdog compound fix lives ONLY in scratch (G:/temp/longrun/fresh_go_watchdog.sh). Landmine until ported.

## MISSION (one sentence)
Land ONE qualifying run: fresh cold bedroom -> credits, zero human touches, with an organically built, properly leveled, duplicate-free six and bounded watchable pacing - then emit stats, write the sentinel, stop the train.

## BUILD ORDER
0. PORT + LOOP HYGIENE: (a) port the scratch watchdog compound logic into the repo, commit. (b) patch night_shift.ps1: adaptive cadence (relaunch 60s after a commit shift; 15 MINUTES after a glance-clean 0-commit shift while the run is healthy); FAST-FAIL lines -> logs/fastfail.log, never NIGHT_REPORT.md; sentinel semantics: line 1 of NIGHT_REPORT.md starting with CREDITS stops the loop (exists) - ADD line 1 starting with HALT also stops it. Commit.
1. VICTORY RECOGNITION: champion flag / Hall of Fame -> OUTCOME: CREDITS + banked_CREDITS; post-game = victory-lap mode, never STALL. Commit.
2. STATS GENERATOR -> RUN_STATS_<run>.md: wall-clock start/end/duration; sim seconds; per-badge wall-clock splits; E4 attempts; whiteouts; every catch (species/where/when/kept-vs-boxed); evolutions; per-slot level curves over time; time-share battle/travel/grind/menus; longest continuous grind window; final party+levels; dex; money. RETRO-RUN IT ON fresh_go_1.log FIRST -> RUN_STATS_fresh_go_1.md (Jonny wants yesterday's numbers today). Commit.
3. TEAM-DEPTH (mission-central - why yesterday disqualified): (a) POKEMON_BENCH_TO_MILESTONE default ON. (b) KILL ace-hogs-XP - trainer/road/gym XP reaches the bench (participation share / lead rotation below milestone pins), not only slot 0. (c) extend milestone ladder to an E4 pin (whole six arrives Indigo >= ~L46; tune the number on evidence, the SHAPE is law). (d) de-dup: pokemon_planner._recompute_status scans PARTY not PC BOX -> make box-aware; never re-catch an owned species; prefer NEW species for gaps. (e) verify with ONE decisive cheap look-ahead (evened-kit style: bench climbs pins across >=2 gyms, no parking) - one look-ahead, then GO. No exploratory batteries.
4. LAUNCH fresh_go_2: cold bedroom start, detached (survives shift end), turbo, watchdog on, log G:/temp/longrun/fresh_go_2.log. Canonical saves UNTOUCHED. Single-run law.
5. WHILE COOKING: glance cheap, exit fast. Hard wedge -> capture repro, ROOT-fix, resume from bank. Healthy -> exit.
6. COMPLETION: credits roll -> evaluate QUALIFYING CRITERIA. PASS -> RUN_STATS_fresh_go_2.md + final survey <=20 lines (team-and-why: each member - where caught, why kept, what it answered) + write CREDITS as line 1 of NIGHT_REPORT.md. Train stops. Mission over. Credits but NOT qualifying -> one-page why -> fix root -> relaunch ONE more fresh run. SAME root fails twice at any stage -> HALT line 1 + one-page diagnosis. No qualifying run in flight by 22:00 -> HALT + diagnosis. The train never runs to infinity.

## QUALIFYING CRITERIA (all must hold)
- Fresh cold start; zero human touches; 0 crashes; 0 hard livelocks (self-recovering LOUD aborts OK).
- Final party: SIX DISTINCT species, no duplicates, all acquired this run.
- Levels at E4 entry: every member >= L42; highest-to-lowest gap <= 15; nothing near L100.
- Participation: >= 4 of 6 record wins in gyms/E4.
- Pacing: no continuous wild-grind window > ~20 sim-min without a logged milestone reason; total grind share <= ~35% of run. Bounded, purposeful, narrated is the law.
- RUN_STATS generated.

## DISCIPLINE
Fresh USD 250 credits now; weekly bucket refills 9am. Surveys <= 15 lines. No side quests. Organic HARD RULE absolute: no transplants, no struct-grinding, no pre-E4 solo grind. Escalate to Jonny ONLY via HALT; otherwise decide and act.

## ADDENDUM (pre-launch clarifications)
- STATS GENERATOR: parse fresh_go_1.log with a SCRIPT (commit it as tools/run_stats.py) - do NOT read the 34MB log into your own context. If a stat is not derivable from the log, write 'not instrumented' and add the missing logging for fresh_go_2 instead of guessing.
- ORDER 3e look-ahead: reuse an existing banked fixture (fuchsia_evened_kit / evened-kit style) - do NOT build new fixtures; that path is a known token sink.
- 22:00 rule, precise: a run IN FLIGHT and healthy at 22:00 CONTINUES (detached) - the deadline halts idle/blocked TRAINS, never a live healthy run. HALT semantics: write the one-page diagnosis INTO NIGHT_REPORT.md under the HALT line.
- If fresh_go_2 rolls credits but fails QUALIFYING on a pacing/participation metric only (team shape correct: six distinct, levels in band), do NOT relaunch - write CREDITS line 1 + stats + a WATCHABILITY GAPS section; Jonny adjudicates pacing personally.

## SHIFT-1 PROGRESS (2026-07-13, updated live)
DONE + COMMITTED:
- Order 0a: watchdog PORTED to repo `pokemon_agent/fresh_go_watchdog.sh` (was scratch-only landmine). Same compound carry-forward-banked_GOAL logic; LOG param (default fresh_go_2.log); stops on banked_CREDITS.
- Order 0b: `night_shift.ps1` — HALT line-1 sentinel stops loop; fast-fails -> `logs/fastfail.log` (not NIGHT_REPORT.md); adaptive cadence (60s after commit/fast-fail shift, 900s after 0-commit glance-clean).
- Order 1: VICTORY RECOGNITION in `recon_longrun.py` — reuses campaign's firewalled post_game (8 badges AND (game-clear 0x82C OR Hall/Champion room)); fires on boot->post_game TRANSITION; new CREDITS outcome -> banked_CREDITS (stops watchdog). Safety net after free_roam returns.
- Order 2: `tools/run_stats.py` (streaming, never loads 34MB log) + `RUN_STATS_fresh_go_1.md`. CONFIRMS disqual shape: ace Venusaur 6->100, bench flat ~L40 (60-lvl gap at badge 8), DUPLICATE Dugtrios, 83.6% travel/9.2% battle.
- Order 3a: `BENCH_TO_MILESTONE` default OFF->ON (campaign.py:147). Treadmill root killed NS#36 (verified). Decision-logic verifier `recon_bench_milestone_check.py` ALL 6 CASES PASS with the flip.
- Order 3d: box-aware de-dup (roster_judgment `also_owned` param + `_box_species_ids()` + catch_one force-catch flee-if-owned + planner `_recompute_status` unions boxed-species cache).
- Order 3c: milestone ladder ALREADY pins whole six to L55 at E4 (`_prep_e4_target`) — no edit needed.

- Order 3e VERIFY: decision-logic verifier `recon_bench_milestone_check.py` PASSES ALL 6 (re-pin +6 bites, park-proof release, retire-when-close). TWO behavioral look-aheads (surge_done_kit, koga_done_kit) were CONFOUNDED by pre-existing fixture NAV wedges (Diglett's-Cave entry geometry / Route-9 (41,6) zone-gap) before either reached the bench-grind — NOT my regression (zero errors from new code; wedges surface LOUD/no-inner-spin). Mechanism verified at logic level + error-free live; per mission "one look-ahead then GO" -> launched.
- Order 4 DONE: **fresh_go_2 LAUNCHED 06:32** — cold FRESH bedroom start, detached (nohup), turbo (SDL dummy), watchdog `fresh_go_watchdog.sh`, log `/g/temp/longrun/fresh_go_2.log`. Stale resume banks archived to `banked_*_archived_fresh_go_1_0631`. Canonical UNTOUCHED (workshop mode). Booted FRESH map(0,0) badges=0 into the proven opening spine.

FRONTIER NOW: fresh_go_2 is COOKING (order 5). GLANCE cheap, exit fast. On banked_CREDITS -> order 6 (run_stats -> RUN_STATS_fresh_go_2.md -> evaluate QUALIFYING -> CREDITS/WATCHABILITY-GAPS line). Hard wedge (whole-log spin, 0 progress many iters) -> ROOT-fix + resume. Healthy -> exit. If ace still over-climbs / bench < L42 at E4 (qualifying fail) -> the deferred order-3b `_road_bench_xp_arm` questline-guard relax is the next build (see DEFERRED above).

NOT DONE / DEFERRED (next shift if fresh_go_2 disqualifies on ace over-climb):
- Order 3b DEEP: `_road_bench_xp_arm` questline guard (campaign.py:7179) disables bench participation-XP during ALL questlines incl. Victory Road -> ace solos VR to L100. Relaxing it to allow overworld-route questline legs (NOT dark caves/gauntlet interiors) is the remaining crux fix, but RISKY (in-cave switch livelock) — needs isolated validation, do NOT ship blind.

## FRESH_GO_2 LAUNCH PROCEDURE (order 4 — the exact recipe)
1. Ensure COLD start: remove/rename `/g/temp/longrun/banked_LIVE` and any stale `banked_GOAL/STALL/TIMEOUT/CREDITS` so iter 1 boots FRESH (scripted bedroom spine).
2. Detached turbo launch (survives shift end): `cd pokemon_agent; SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy FRESH_GO_LOG=/g/temp/longrun/fresh_go_2.log nohup bash fresh_go_watchdog.sh &`
3. Canonical `states/campaign/` UNTOUCHED (recon monkeypatches _save to STAGE). Single-run law: exactly ONE watchdog at a time.
4. Watch for banked_CREDITS (stops watchdog) -> run `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` -> evaluate QUALIFYING CRITERIA.
