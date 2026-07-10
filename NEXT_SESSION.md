<!-- ═══ NIGHT-SHIFT #14 (2026-07-10 night) — BADGE 4 (Erika): PP-FAMINE root fixed, IN VERIFICATION ═══
FRONTIER: BADGE 4 (RAINBOW, Erika/Celadon). The full chain from surge_done_kit REPLAYS END-TO-END this
shift: Flash errand (catch to dex 10 → HM05 from Route-2 aide → teach Flash) → Rock Tunnel crossing
(lit, fights the tunnel trainers, Venusaur L35→L43) → Underground Path → Celadon → enters Erika's gym →
clears juniors → engages Erika. She REACHES and FIGHTS Erika. The wall is the LEADER FIGHT.

★ TRUE ROOT (shift #14, deeper than #1/#2 — the #2 premise was WRONG): Venusaur STILL HAS Tackle (normal
x1) — auto-learn did NOT strip it. So the coverage-teach (which only fires when the ace has NO neutral
move, best_have<1.0) correctly did nothing. The real wall is PP FAMINE: Erika's junior gauntlet is ALL
grass/poison, so Tackle (Venusaur's ONLY neutral move, 35 PP) is the best pick vs every junior → she spams
it through the gauntlet → it drains to 0 PP before the leader → choose_move falls to RESISTED Razor Leaf
(x0.25) → can't out-damage L29 Erika → BLACKS OUT. The retry only wins because whiteout REFILLS PP AND the
beaten juniors stay beaten (Tackle survives to the leader → win). VERIFIED in the s14 look-ahead log:
attempt-1 = Razor Leaf x0.25 vs victreebel (85→57) then wipe; attempt-2 = Tackle x1 vs victreebel (85→61→
35→9), winning when the run was killed. So she DOES bank badge 4 via the retry ratchet, but wastes attempt
1 + a whiteout round-trip (unwatchable) every time.

★ FIX APPLIED + COMMITTED (5c07f01, IN VERIFICATION): `_teach_gym_coverage` guard changed from "skip the
instant ONE neutral move exists (best_have>=1.0)" to "count neutral-or-better DAMAGING moves; skip only at
>=2 (real PP depth)". With 0 or 1 neutral move, teach a bag coverage move — for Erika → teaches Venusaur
CUT (HM01, still in bag, Venusaur-compatible) so she carries Tackle+Cut = 65 PP of neutral coverage and
reaches the leader with PP to spare (choose_move picks Cut 50pw first, then Tackle 35pw; both x1). Forgets
the weakest dmg move vs grass/poison = Vine Whip (Tackle SAFE). Self-limiting (2 neutral after the teach).
Added a dispatch-result log (`coverage-teach dispatch -> taught|have_coverage|...`) + a depth-only voice
line. Flag POKEMON_GYM_COVERAGE_TEACH=1 (default ON, confirmed).

VERIFY / CLIMB COMMAND (badge-4 from the fresh badge-3 bank; ~30-40 min wall clock — it does the WHOLE
Flash errand + Rock Tunnel + Celadon before Erika, so give it the full 45):
   `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x823 .venv/Scripts/python.exe -u
   pokemon_agent/recon_longrun.py surge_done_kit.state 45` → /g/temp/s14b_badge4.log. GOAL 0x823 = RAINBOW.
   WATCH: `GYM-PREP [Erika]: coverage-teach dispatch -> taught` + `COVERAGE-TEACH Cut` BEFORE `engaging
   Erika` → ace lands x1 Cut/Tackle → `GYM: won` / badge flag 0x823 set → OUTCOME: GOAL.
   grep: `grep -nE "coverage-teach dispatch|COVERAGE-TEACH|engaging Erika|GYM: won|GYM: lost|blacked|OUTCOME|badges=4" log | grep -v ctx=`.
   ON GOAL: bank at /g/temp/longrun/banked_GOAL/ (round-trip + sanctity checked). Promote to
   states/workshop/erika_done_kit.state (+ sidecars) for the badge-5 frontier.

IF SHE STILL LOSES ERIKA after the Cut teach: (a) confirm `coverage-teach dispatch -> taught` (if
`have_coverage`, the neutral_dmg count guard didn't trip — check Venusaur's actual moveset at Erika-time;
if `not_in_case`/`failed`, TM-Case nav bug, see hm_teach.py TeachFlow). (b) If taught but still loses,
the backstop is a PP-restore heal between the junior clear and the leader (a real player heals if low) —
NOT built yet; the coverage-teach was chosen over it (self-contained, no mid-gym excursion fragility).

RESIDUAL — JUNIOR-ENGAGEMENT SPIN (real watchability cost, NOT yet fixed): Erika's flower-tile layout
leaves ~4 juniors (obj1/2/5/6) "un-engageable (wandering/water-locked)". She tries each 4× (~115 log lines
each) before deferring, burns the 14-clear-round cap (~1500 lines / lots of wall-time), THEN proceeds to
the leader. The cap correctly bails her through so it's not fatal, but it's slow/unwatchable and repeats on
every gym attempt. Fix candidate: remember the deferred (un-engageable) set across clear-rounds and skip
straight to the leader once only un-engageable juniors remain, instead of re-scanning them. Do this AFTER
badge 4 banks (distance first).

RESIDUAL — Tea detour (minor): on reaching Celadon she briefly toured a building chasing the Tea (a
Saffron/badge-6 errand the questline armed early); head_to_gym correctly kept Erika as the current
objective and she entered the gym. Not blocking. The Tea gate is for LATER (Saffron).

FIXTURES: surge_done_kit.state @ pokemon_agent/states/workshop/ (badge 3 @ Vermilion; bag HM01 Cut +
TM28 Dig + TM03/39/34 + 3 balls; NO HM05 Flash and dex 6 — the Flash errand IS part of this chain and
takes ~15 min of the run). Prior erika_done.state (Jul 9) + banked_CREDITS exist — this is a RELIABILITY
RE-CLIMB (game already beaten once). Do NOT resume from states/campaign/checkpoints/ auto-banks (STALE /
champion L95). recon_bagdump.py / recon_partydump.py take a FULL path.

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill first: `taskkill //F
   //IM python.exe //T`). recon_longrun arg2 = max_minutes (wall-clock cap). SCRATCH = $TEMP/longrun =
   /g/temp/longrun; canonical Champion save NEVER touched. GOAL 0x823 = FLAG_BADGE_RAINBOW. The look-ahead
   decision-counter [NNN.Ns] FREEZES during a gym (the whole junior-clear + leader fight is ONE head_to_gym
   decision) — that's NORMAL, watch the growing log lines, not the counter. Bag decode: TM_N=288+N, HM_N=338+N.
   Battle move-picker = pol.choose_move (pokemon_policy.py:49) scores max(power,1)*eff, NO STAB; when enemy
   types are unresolved it goes raw-power (favours resisted STAB) — not the Erika root but a latent risk.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 4 Erika (chain proven to
   her gym e2e; PP-famine root fixed via coverage-teach-for-depth, in verification). Pop-in = `python
   pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
