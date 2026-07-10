<!-- ═══ NIGHT-SHIFT 14 (2026-07-10) — BADGE 3 wall = S.S. ANNE GARY (cabin-skip DEAD-ZONE regression) ═══
THE FRONTIER: BADGE 3 (Lt. Surge). Shift 13's cabin-sweep SKIP works mechanically but was set to the WRONG
level, creating a Gary-loss livelock. Shift 14 diagnosed it end-to-end (from the live s13_surge.log) and
fixed the threshold; a verify run is in flight.

★★ SHIFT-14 HEADLINE — shift 13 traded a cabin-sweep livelock for a GARY-LOSS livelock. Ground truth from
   the still-running s13_surge.log: the cabin-sweep skip fired correctly ('🥊 SHIP CABIN SWEEP SKIPPED —
   ace L28 already overpowers ... >= L24'), she B-lined to Gary at Ivysaur **L28** — and LOST. The trace is
   unambiguous (Gary battle ~line 1160-1195): Razor Leaf resisted **x0.25** by Charmeleon, **Tackle
   PP-famined** ('move slot 2 didn't fire (0-PP)'), Charmeleon's Fire hits **super-effective** vs Grass
   Ivysaur ('SE-CHUNK observed'), and the frail L8-12 bench (Spearow/Rattata/Abra) can't clean up ->
   whiteout -> she dumps at the **Cerulean** Center (not Vermilion), then oscillates Cerulean<->Vermilion
   with the recovery grind leveling the **frail bench to ~L17** (NOT the ace) -> re-board -> lose again =
   livelock. Shift 11's ONLY proven Gary-killer was **Venusaur L32**; shift 13's skip at L24 removed the
   very leveling that reaches it.

   ROOT: a THRESHOLD DEAD-ZONE (L24-L30). The cabin-sweep skip (L24) sat BELOW the rival-prep floor (L31)
   and target (L32, POKEMON_RIVAL_PREP_LEVEL). An L28 ace SKIPPED cabins (28>=24) AND the first-approach
   prep-grind didn't fire yet (28<31, no loss recorded) -> nothing leveled her -> guaranteed loss.

   FIX (SHIFT 14, campaign.py ~6507, UNCOMMITTED-until-verified): couple the sweep-skip default to the SAME
   knob the prep-grind targets — `_prep_lv = POKEMON_RIVAL_PREP_LEVEL (32)`, `_sweep_skip_lv` defaults to
   `_prep_lv`, and raise `POKEMON_SHIP_SWEEP_CAP` 4->10. Now BELOW L32 the cabins RUN (bounded by the
   persistent `_ship_cabins_swept` cap -> no livelock) and level the ace toward Venusaur; the sweep
   SELF-TERMINATES the instant the ace hits L32 (`0 < _sw_ace < 32` goes False -> skip fires -> B-line to a
   WINNABLE Gary). The lose->prep-grind->reboard path tops off any residual. Thresholds can never drift into
   a dead-zone again (both read POKEMON_RIVAL_PREP_LEVEL). SYNTAX OK; scope-checked (_questline_interact@6355,
   so it reads the env directly, NOT the out-of-scope _RIVAL_PREP_LEVEL@5974).

   VERIFY RUN IN FLIGHT (launched shift 14): `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1
   LONGRUN_GOAL_FLAG=0x822 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 55`
   -> /g/temp/s14_surge.log. WATCH FOR (grep, skip the ctx= dumps): board S.S. Anne -> '🥊 SHIP CABIN SWEEP:
   clearing this deck's trainers' (NOT skipped at L28) -> ace climbs L28->~L31 via cabins -> either wins pass
   1 at L32 OR a clean loss -> '🏋️ PREP-BEFORE-RIVAL ... grinding up' tops off to Venusaur L32 -> reboard ->
   Gary grudge encounter **won=True** -> captain -> HM01 Cut -> disembark to Vermilion -> head_to_gym -> Cut
   tree at gym door -> 'stuck' -> shift-11 probe 'TIMBER'/use_cut -> gym -> TrashCanPuzzle -> beat Surge ->
   OUTCOME: GOAL (0x822 = FLAG_BADGE_THUNDER).
   IF GOAL: commit already landed (see below) — bank surge_done, climb to BADGE 4 (Erika/Celadon — Rock
   Tunnel/Flash chain; those stretches have night-train fixes, MEMORY.md).
   IF Gary WON but no Surge: the fix is proven — the residual is downstream (captain/Cut/gym-gate); re-read
   the log for where it stalls and fix THAT (shift-11/12 fixes should cover it).
   IF Gary still LOST: read the Gary battle in /g/temp/s14_surge.log. Check the ace actually reached L32
   (grep "party=\[\('Ivys'|'Venu'"); if it never hit L32, the prep-grind after the loss isn't firing at the
   ship anchor (whiteout-to-Cerulean displaces `step_anchor` — the prep gate needs `cur_map==step_anchor`).
   That anchor-displacement is the next suspect if this run doesn't converge.

FIXTURES: bill_done_kit.state at pokemon_agent/states/workshop/ (Ivysaur L26 [RazorLeaf,VineWhip,Tackle,
   SleepPowder] + frail bench) — rebuild `.venv/Scripts/python.exe pokemon_agent/recon_repair_kit.py`. Do NOT
   resume from states/campaign/checkpoints/ auto-banks (STALE Jul-8). The full organic bill_done_kit run is
   the honest verify (reproduces the Gary wall end-to-end).

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first:
`tasklist //FI "IMAGENAME eq python.exe"` then `taskkill //F //PID <both> //T`). The `.venv` shim DETACHES and
fires a false 'completed' on background launch — the REAL worker keeps running; check via tasklist + log mtime,
NOT the task-notification. recon stages to G:/temp/longrun/stage (WIPED each run), banks to banked_<OUTCOME>;
canonical Champion save NEVER touched. GOAL flags: recon_longrun.py:67-76. Kill-switches:
POKEMON_SHIP_CABIN_SWEEP=0 disables the sweep; POKEMON_SHIP_SWEEP_SKIP_LEVEL/POKEMON_SHIP_SWEEP_CAP/
POKEMON_RIVAL_PREP_LEVEL tune the gate. The Gary battle grep trick: `grep -nE "engine|LOSS|won=True|SWEEP|
PREP-BEFORE" log | grep -v "ctx="` (the ORACLE ctx dumps otherwise flood every grep).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 3 (Surge) — cabin-skip
dead-zone fix (committed shift 14) in-flight-verify to the badge. Pop-in = `python pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
