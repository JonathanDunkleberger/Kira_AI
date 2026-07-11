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

## ALSO (option-1 Lapras tactical AI — good regardless)
Battle-brain: when facing a foe the ace can't hurt (Venusaur 0.25x/1x into Charizard) and a specialist is
the type-answer, LEAD/keep that specialist in and funnel healing to IT rather than splitting FR with the
ace's 1x Cut. (run12 Gary census: Lapras landed only 10 SE hits before fainting while Venusaur threw 51 weak
hits.) This is just good battle-AI; wire it in independent of the team-depth fix.
