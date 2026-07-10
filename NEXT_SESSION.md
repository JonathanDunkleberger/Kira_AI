<!-- ═══ NIGHT-SHIFT #2 (2026-07-10 night) — BADGE 4 (Erika) TYPE-WALL: coverage-teach fix IN VERIFICATION ═══
FRONTIER: BADGE 4 (RAINBOW, Erika/Celadon). The badge-4 CHAIN replays cleanly from surge_done_kit
(Vermilion → Flash errand → Rock Tunnel → Celadon → enters Erika's gym). She REACHES Erika every run.
The wall is the FIGHT, and shift #1's diagnosis was INCOMPLETE.

★ TRUE ROOT (shift #2, deeper than shift #1): Erika is a GRASS/POISON gym and Kira's L43 Venusaur ace has
NO neutral coverage — her whole moveset is Grass (Razor Leaf / Vine Whip / Absorb), ALL x0.25 vs grass/poison.
Auto-learn STRIPPED her only neutral move (Tackle, present at L35 in the fixture) as she levelled L35→L43.
So even a +14-level lead can't out-damage: she spams x0.25 grass, PP-famines to status, and BLACKS OUT —
3 gym attempts, 3 whiteouts, then marks the gym a spatial wall ("gated until stronger") → post-loss stall.
The battle brain DOES try to adapt (switches to Pidgey → Gust x2), but the flyers are L12-17, too frail vs
L29 Erika mons. Shift #1's `_restore_ace` fix is NECESSARY (ace must lead) but INSUFFICIENT (the ace itself
is type-dead here). Her bag: 3 Poké Balls, HM01 Cut, TM03 Water Pulse, TM28 Dig, TM39 Rock Tomb, TM34 Shock
Wave — of which only Cut (normal) and Dig (ground) are ≥ x1 vs grass/poison (the rest resisted).

★ FIX APPLIED (uncommitted → committed this shift; IN VERIFICATION): new `_teach_gym_coverage(gym, rec)`
(campaign.py ~3326) + `_COVERAGE_MOVES` KB (~237). When prep_for_gym sees `not has_type_answer` and the
catch can't answer it, if the ACE's whole offense is resisted (best dmg-move eff < x1 vs rec['types']),
it scores the bag TMs/HMs against the gym's DEFENDING types (type chart), picks the best learnable
neutral-or-better move (prefer SE, then power, then HM), and TEACHES it to the ace (forgetting its weakest
damaging move; keeps status + best STAB). For Erika → teaches Venusaur **Cut** (normal x1, 50pw) — 4× her
current x0.25 grass; L43 neutral Cut should crush L24-29 Erika. TM move_ids ROM-VALIDATED (st.move_info)
before teaching so a wrong id is skipped, not mis-taught. Self-limits (after teach, best_have≥1 → no
re-teach). Also kept shift-1's `_restore_ace()` in beat_gym (~3322). Flag: POKEMON_GYM_COVERAGE_TEACH=1.

VERIFY / CLIMB COMMAND (badge-4 from the fresh badge-3 bank; ~10-15 min wall clock to Erika's gym):
   `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x823 .venv/Scripts/python.exe -u
   pokemon_agent/recon_longrun.py surge_done_kit.state 45` → /g/temp/s2_badge4.log. GOAL 0x823 = RAINBOW.
   WATCH: `COVERAGE-TEACH Cut` before `GYM: inside` → ace lands x1 Cut → Erika WON → BADGE 4.
   grep: `grep -nE "COVERAGE-TEACH|coverage-teach|GYM: inside|GYM: won|GYM: lost|Rainbow|BADGE|blacked" log | grep -v ctx=`.

IF SHE STILL LOSES ERIKA after the Cut teach: (a) confirm the teach actually landed (`coverage-teach Cut ->
taught`) — if `not_in_case`/`failed`, the TM-Case navigation is the bug (see hm_teach.py TeachFlow). (b) If
taught but she still loses, the SECONDARY blocker is the JUNIOR-ENGAGEMENT spin: Erika's flower-tile layout
leaves ~6 juniors "un-engageable (wandering/water-locked)" — she burns 14 clear-rounds then proceeds LOUD;
that spin + attrition may still whiteout. Fix path: make the gym-clear SKIP un-engageable juniors faster /
path straight to the leader (Erika sits at the back; you don't need every junior). (c) If Cut isn't enough
damage, escalate to Dig (TM28, ground, but 2-turn — excluded from _COVERAGE_MOVES for actuation risk; would
need the engine to drive the charge/hit) OR grind a flyer to ~L25 for Gust x2.

RESIDUALS (noted, NOT blocking): (1) MAP (18,0) Saffron gatehouse decoy dead-end on the Route-6→5 crossing —
DO NOT blind-add to gates.json no_connector: unlike the true dead-ends 17,0/17,1, 18,0 IS a real connector
(→ Saffron 3,10, guard-gated pre-Tea); permanently blocking it risks breaking Saffron access later. It
self-heals via dead-end-backout. Leave it or make it CONDITIONAL on the Saffron gate being locked. (2)
junior-engagement spin (above). (3) she tours Celadon buildings before the gym — slow/unwatchable.

FIXTURES: surge_done_kit.state @ pokemon_agent/states/workshop/ (badge 3 @ Vermilion, party [venusaur L35,
spearow L12, rattata L8, abra L10], dex 6; bag has HM01 Cut + TM28 Dig + 3 balls). Do NOT resume from
states/campaign/checkpoints/ auto-banks (STALE / champion-save L95 party — NOT the surge_done_kit run).

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill first: `taskkill //F
   //IM python.exe //T`). recon_longrun arg2 = max_minutes (wall-clock cap). setdefaults POKEMON_FIELD_MOVES=1
   + POKEMON_ITEM_PICKUP=1; stages to G:/temp/longrun/stage; canonical Champion save NEVER touched. GOAL
   0x823 = FLAG_BADGE_RAINBOW. Bag decode: TM_N item = 288+N, HM_N item = 338+N. Coverage move-ids are
   canonical Gen-3 (Cut=15 certain; TMs ROM-validated at runtime). recon_partydump.py / recon_bagdump.py
   take a FULL path (not a bare name) — pass pokemon_agent/states/workshop/<file>.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 4 Erika (chain proven to
   her gym; coverage-teach Cut fix in verification). Pop-in = `python pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
