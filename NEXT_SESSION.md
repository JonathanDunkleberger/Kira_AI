<!-- ═══ NIGHT-SHIFT 11 (2026-07-10) — S.S. Anne Gary ROOT CAUSE = a FALSE-POSITIVE whiff detector ═══
THE FRONTIER: beat the S.S. Anne rival GARY -> captain HM01 Cut -> Vermilion gym tree -> Lt. Surge (badge 3).

★ SHIFT-11 ROOT CAUSE (this ends the shift-8/9/10 "whiff-spiral / PP-famine" misdirection — it was a
  MEASUREMENT ARTIFACT). Ground truth from the shift-10 verify log (/g/temp/s10_gary.log), the losing Gary
  battle, verbatim:
      WHIFF: ... foe HP frozen at 50 (streak 1/3)
      WHIFF: ... foe HP frozen at 47 (streak 2/3)   <- 50->47, the hit LANDED
      WHIFF: ... foe HP frozen at 14 (streak 3/3)   <- 47->14, the hit LANDED
      Tackle -> charmeleon 55/55                     <- kadabra fainted (14->dead)
  The "frozen" HP is visibly DROPPING (50->47->14->dead). EVERY landing move was flagged a miss. Cause:
  `_note_whiff` (battle_agent.py) read the foe HP with a short passive `_settle` RIGHT AFTER the move
  COMMITTED (PP dropped) — but battle text needs ACTIVE advancement (the turn loop's `_settle_action_menu`
  presses A/B) for the damage to APPLY, so the passive read always saw the PRE-damage HP -> false "miss".
  The false whiff-spiral fired the shift-8/9 whiff-BREAKER's ace<->frail-bench switches every 3 turns,
  wasting tempo + exposing the L13/L8 bench -> a fight a L30 Ivysaur (10-14 levels above Gary's L16-20 team)
  should WIN was churned into a loss. Shift 8's "full-PP Venusaur missed 25 Tackles" was this artifact, not
  a real accuracy debuff (25 real misses is statistically impossible).

★ THE FIX (committed, VERIFIED-BY-<run>): race-free whiff classification. `_note_whiff` -> `_classify_prev_whiff`
  (battle_agent.py ~2071): compare the foe HP at the START of the PREVIOUS turn to the START of THIS turn
  (both CLEAN menu-up reads the loop already takes after actively advancing all text + applying damage). A
  real miss = SAME foe, HP unchanged across two consecutive turn-starts while a damaging move fired. Deferred
  by one turn; read-only (no button presses — safe, rule-12 minimal). The whiff-BREAKER is UNCHANGED and now
  fires ONLY on genuine Sand-Attack/Kinesis/Smokescreen spirals (its real purpose). Init state + call site:
  battle_agent.py run() (~2127) stores `_whiff_prev_hp/_sp/_fired`; classify at the menu-up read (~2381);
  store at the fire point (~2535). PP is decremented at commit so "damaging move fired" stays reliable.

IN FLIGHT / NEXT — THE VERIFY RUN (goaled on BADGE 3 so it stops decisively):
  POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x822 \
    .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 45   (log /g/temp/s11_gary.log)
  0x822 = FLAG_BADGE_THUNDER (Surge, badge 3). WATCH FOR: no more spurious "WHIFF" spam on landing hits;
  "GARY encounter #N recorded (WIN)" / grudge flip; then captain HM01 Cut -> Vermilion tree -> Surge -> badge.

IF SHE STILL LOSES with the detector fixed (the false whiffs gone but Gary still walls her): the log now shows
  the REAL reason unpolluted. Likely candidates + fixes, in order:
  1. She wins clean at L30 -> DONE, bank surge and climb to badge 4 (Flash/Rock Tunnel — the shift-1/2/3
     night-train arc already solved that stretch from surge_done.state; re-verify from the honest team).
  2. Genuinely worn down by 4 mons -> the prep-grind to L32 (Venusaur) + heal-before-rival (campaign.py
     ~5975-6049) already exist and fire; the 2-pass flow tops her off. Should now suffice with landing moves.
  3. A DIFFERENT battle bug surfaces -> read /g/temp/s11_gary.log battle turns and fix that specific one.

FIXTURES (states/workshop/ is GITIGNORED SCRATCH — rebuild if wiped):
  - bill_done_kit.state (Ivysaur L26 [RazorLeaf,VineWhip,Tackle,SleepPowder] + frail bench) — REBUILT this
    shift via `.venv/Scripts/python.exe pokemon_agent/recon_repair_kit.py` (writes pokemon_agent/states/workshop/).
  - bill_done.state (degraded BUG kit [RazorLeaf,PoisonPowder,SleepPowder,-]) EXISTS.

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first,
`taskkill //F //PID <both>`). recon_longrun stages to G:/temp/longrun/stage (WIPED each run), banks to
banked_<OUTCOME>; canonical Champion save NEVER touched. LONGRUN_BATTLE_LOG=1 surfaces per-move engine logs.
GOAL flags: LONGRUN_GOAL_FLAG / LONGRUN_GOAL_MAP / LONGRUN_GOAL_DEX (recon_longrun.py:67-76).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = S.S. Anne Gary, root-caused to a
false-positive whiff detector (fixed this shift); verifying to badge 3. Pop-in = `python pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
