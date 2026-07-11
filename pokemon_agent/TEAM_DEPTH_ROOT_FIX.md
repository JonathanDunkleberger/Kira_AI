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

## ALSO (option-1 Lapras tactical AI — good regardless)
Battle-brain: when facing a foe the ace can't hurt (Venusaur 0.25x/1x into Charizard) and a specialist is
the type-answer, LEAD/keep that specialist in and funnel healing to IT rather than splitting FR with the
ace's 1x Cut. (run12 Gary census: Lapras landed only 10 SE hits before fainting while Venusaur threw 51 weak
hits.) This is just good battle-AI; wire it in independent of the team-depth fix.
