# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## FRONTIER: BADGE 4 (Erika / Celadon — RAINBOW 0x823) — coverage-teach-for-depth fix IN VERIFY

★ SHIFT-17 KEY FINDING: the shift-14 fix (5c07f01, committed 09:35) was NEVER tested — all s14/s15/s16
runs finished 09:23–09:49 on PRE-fix code (b3f729d) or were killed early by shift-cycling. The full run
from surge_done_kit takes ~40 min and never completed inside a shift. So shift 17 built a FAST fixture.

★ FAST FIXTURE: `states/workshop/erika_retry_kit.state` (+ sidecars) = copy of banked_STALL (the s14 run's
post-blackout STALL bank). Venusaur L43 [Razor Leaf, Vine Whip, Tackle(35pp,1 neutral move), Sleep Powder];
badges 3 (boulder/cascade/thunder); rainbow OFF; HM01 Cut + HM05 Flash + TM03/28/34/39 in bag; map (3,6)
coords (59,15) (near Lavender/Rock-Tunnel, post-blackout). Loading this reaches Erika in MINUTES, not 40.

★ VERIFY COMMAND (fast — ~15-20 min to Erika from this fixture):
   `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x823 .venv/Scripts/python.exe -u
   pokemon_agent/recon_longrun.py pokemon_agent/states/workshop/erika_retry_kit.state 25` → /g/temp/s17_erika.log
   WATCH FOR: `GYM-PREP [Erika]: coverage-teach dispatch -> taught` + `COVERAGE-TEACH Cut (normal, x1 neutral)
   -> venusaur` BEFORE `engaging Erika` → ace fields Tackle+Cut (65 PP neutral) → `GYM: won` → flag 0x823.
   grep: `grep -nE "coverage-teach dispatch|COVERAGE-TEACH|engaging Erika|GYM: won|GYM: lost|blacked|OUTCOME|badges=4" LOG | grep -v ctx=`

★ EXPECTED current-code path (5c07f01): dispatch fires (has_type_answer=False) → _teach_gym_coverage counts
neutral_dmg=1 (only Tackle) <2 → finds Cut (HM01, Venusaur-compatible, normal x1) → teaches it, forgets
Vine Whip (weakest dmg vs grass/poison) → Venusaur carries Tackle+Cut = 65 PP neutral → reaches leader with
PP → wins. If it DISPATCHES but still loses: backstop = PP-restore heal between junior-clear and leader
(NOT built). If dispatch shows `have_coverage`: the neutral_dmg guard mis-counted — check Venusaur moveset
at Erika-time. If `no_candidate`: Cut not found/compatible — check _COVERAGE_MOVES + hm_compatible.

★ ON GOAL (rainbow 0x823 set): bank /g/temp/longrun/banked_GOAL → promote to
states/workshop/erika_done_kit.state (+ sidecars). Frontier advances to BADGE 5 (Koga/Fuchsia — scouted
shifts 7-9: Route-15 gate crossed, Koga is a team/movepool wall, Fuchsia Mart potions likely answer).

★ RESIDUALS (fix AFTER badge 4 banks): (1) junior-engagement spin — Erika's flower-tile layout leaves ~4
juniors un-engageable; she tries each 4× then burns the 14-clear-round cap (slow/unwatchable) before the
leader. Fix: remember deferred un-engageable set across clear-rounds, skip to leader once only they remain.
(2) Full-chain-from-surge_done_kit never completes in a shift — either speed it up or run detached.

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — recon_longrun auto-reaps predecessor via
longrun.pid). arg2 = max_minutes. GOAL 0x823 = FLAG_BADGE_RAINBOW. Decision-counter FREEZES during a gym
(whole junior-clear+leader = ONE head_to_gym decision) — watch growing log lines, not the counter. Bag
decode: TM_N=288+N, HM_N=338+N. Coverage-teach = campaign.py:3310 dispatch / :3326 `_teach_gym_coverage`.
recon_partydump.py / recon_bagdump.py take a full path.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 4 Erika (coverage-teach-
for-depth fix committed, in verification via erika_retry_kit fast fixture). Pop-in = `python
pokemon_agent/watch.py`.
