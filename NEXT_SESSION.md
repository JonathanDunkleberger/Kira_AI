# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ⏳ NIGHT-SHIFT #18 IN FLIGHT (2026-07-11, night_shift.ps1 shift 18) — running the NS#17 FLAG-FLIP GATE: a DEFINITIVE-budget VR2F cave-grind look-ahead to prove Lapras crosses L43 + learns Ice Beam, 0 Viridian/boulder/park. START HERE.

### THE STATE (shift 18): the flip gate is the ONE decision left before arming `POKEMON_CAVE_GRIND`. Mechanism proven CLEAN over 6 stints (NS#17). This shift is proving the 15-stint pace actually crosses L43.
- **RUN IN FLIGHT:** `ns18_vrgrind_def.log` (`MAX_STINTS=16 VR_BUDGET_S=6000 GRIND_TARGET=43 POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_vr_grind_smoke.py`) from indigo_reach_g (party `[71,13,9,39,14,40]`). Retires when the levelable bench (Lapras L39 / Kadabra L40) crosses L43 OR 16 stints. **AT RESUME grep** `RESULT:|SMOKE|reserved a slot|viridian|boulder` — the gate PASSES iff Lapras≥43 (learns Ice Beam via the reserved slot — CONFIRMED firing: "PHASE 3B reserved a slot — dropped Confuse Ray") AND 0 Viridian AND 0 boulder.
- **⚠️ CPU-CONTENTION CAVEAT (this shift):** Jonny's desktop had chrome/steam/iCUE/EADesktop open → the headless emulator ran ~4-10× SLOWER than the quiet-machine 14× ceiling (~40s/stint quiet → minutes/stint here). So **wall-clock (c) is NOT fairly measurable this session** — on a quiet machine 16 stints ≈ 10-15 min. Also hit + fixed a DOUBLE-LAUNCH (two competing recon_vr_grind_smoke processes corrupting the log / halving speed) — killed all python, relaunched ONE clean run. **Lesson: do NOT combine `run_in_background` + a shell `&` — it can double-spawn; use one or the other.**
- **THE FLIP (staged, ready):** if the gate PASSES → set `CAVE_GRIND_ENABLED` default "1" at **campaign.py:125** (`os.getenv("POKEMON_CAVE_GRIND", "0")` → `"1"`), one char, + commit. The lever is strictly beneficial (levels the bench + lands Ice Beam) and park-guarded (bounded stints/budget, retires on crossed-or-stalled). If the run CAPS at 16 stints with Lapras <43 (participation-share too slow / VR2F band caps ~L46) → the flip is still defensible on the clean mechanism, but the honest move is to FIRST evaluate the SOLO_WEAK_GRIND pace lever (frontier item 1) or a longer stint budget before arming.
- **THEN priority 2 (the real win):** the FINAL-PROOF `_enter_league` gate — `POKEMON_CAVE_GRIND=1 POKEMON_VR_GRIND_BUDGET_S=3000 POKEMON_E4_DEADLINE_S=5400 ../.venv/Scripts/python.exe -u recon_enter_league_smoke.py` from indigo_reach_g: does the leveled team (Lapras + Ice Beam) now CLEAR the E4 Champion, or is there a residual team-depth gap? (see NS#17 frontier priority 2 below.)

## ✅ NIGHT-SHIFT #17 DONE (2026-07-11, night_shift.ps1 shift 17) — FINISHED + WIRED the endgame team-depth grind: the Indigo-anchored Victory-Road cave grind is BUILT, wired into `_enter_league` (pre-gauntlet), and VERIFIED end-to-end. THREE commits (`c164efc` cave-grind hardening, `198d982` the Indigo-anchored VR2F grind + wiring, `f9db0c4` full-dispatch smoke), mode-side, canonical UNTOUCHED. **START HERE → the mechanism is proven; the ONE decision left is the flag flip (`POKEMON_CAVE_GRIND` STILL default OFF pending the full-length look-ahead result — see FRONTIER).**

### 🎯 THE STATE: the NS#16 cave-grind lever is now FINISHED + WIRED. It levels the underleveled endgame bench in Victory Road 2F, healing at the Indigo Center, then hands off to the E4 strike — the whole `_enter_league` chain composes with no wedge. The bench-leveling wall (the last binding team-depth constraint) now has a working autonomous lever. Flag stays OFF one more shift for the no-park verification gate.

### ✅ THE DECISIVE NS#17 FINDINGS (from 4 party-6 smokes + 3 Indigo probes — read these, don't re-derive):
1. **Cave-grinding VR1F ABANDONS to Viridian (fatal).** The NS#16 mechanic drew wilds, but the party-6 VR1F smoke exposed TWO blockers: (a) the wander picked FARTHEST-first waypoints → drifted the whole floor → **93 Strength-boulder shoves + trainer wedge-loops**; (b) when the ace dips, `heal_nearest` from VR1F exits to Route-23 **SOUTH**, from which the Indigo Center is split-map UNREACHABLE → it **abandons cross-region to VIRIDIAN** (bottom of the whole climb) and strands her (Viridian grass is a one-way strand → `no_safe_grass`). So VR1F is un-grindable.
2. **VR2F is the RIGHT floor (heals CLEAN to Indigo).** Probes on `indigo_reach_g`: Indigo (3,9) → south lands on **R23-NORTH** (0 reachable grass — the split-map wall confirmed) which can dip into **VR2F via `enter_warp((18,28))`**. Crucially, **heal_nearest from VR2F (dinged ace 15%) → real ~5s excursion to the Indigo League Center → ace 100% → lands back on R23-north, NOT Viridian.** VR2F exits to R23-NORTH (Indigo side), so its heal is short + clean. (Indigo (3,9) IS in `CITY_PC_DOORS` → door (11,6), the League Center — the heal anchor.)
3. **Boulder-avoid + containment KILLED the shove-loop** (`c164efc`): `_cave_grind_avoid` now unions live STRENGTH-BOULDER tiles (`fm.scan_field_objects` GFX_BOULDER) so travel never routes into a pushable boulder; `_cave_walk_waypoints` bounded to a LOCAL Manhattan pocket (`CAVE_GRIND_RADIUS`=6) so the wander stays near the entry. Result on VR2F: **boulder shoves 93→0**.

### ✅ WHAT NS#17 BUILT + COMMITTED (all mode-side, flag-gated OFF, canonical UNTOUCHED):
- **`c164efc`** — cave-grind hardening (boulder-avoid + radius containment) + `gamedata/frlg_grind_spots.json` VR2F (1,40)/VR3F (1,41) cave entries. General robustness (helps any puzzle cave). Decision checks green (recon_grind_spot_check 28/28, lopsided ALL PASS).
- **`198d982`** — **`Campaign.prep_e4_in_victory_road()`**: at Indigo + underleveled + CAVE_GRIND, dip Indigo→R23-north→VR2F (`enter_warp((18,28))`), cave-grind the bench (`grind_weak_members`, participation switch, ace tanks), heal at Indigo when the ace dips, re-dip, repeat until the levelable bench crosses target OR a bound. **Bounded** (`POKEMON_VR_GRIND_STINTS`=15 / `_BUDGET_S`=2400; retires when grind_weak returns `ready`=crossed-or-all-stalled → NO park). Ends by **returning to Indigo (3,9)** so `e4_strike` boots cleanly. **Wired into `_enter_league`** (pre-gauntlet, `CAVE_GRIND`-gated → byte-inert when OFF).
- **`f9db0c4`** — the full-dispatch smoke.
- **VERIFIED e2e** (`recon_vr_grind_smoke` on indigo_reach_g, CAVE_GRIND=1): party `[71,13,9,39,14,40]→[72,13,9,41,14,41]` (Lapras L39→41, Kadabra L40→41, **3 slots rose**), 57 battles, **6 clean Indigo heal excursions, 0 heal-to-Viridian, 0 boulder shoves**, returns 'ready' AT Indigo. Full `_enter_league` dispatch (`recon_enter_league_smoke`): grind → **return to Indigo (3,9)** → **`[e4] boot`** → room #1 Lorelei → room #2 Bruno (terminal 'stuck' = the known team-depth wall from the bounded partial grind, NOT a wiring park). **The two NS#16 fixes now COMPOSE: grinding Lapras past L43 in VR2F → it learns Ice Beam (the move-room fix) → the E4 Champion answer.**

### ⇒ SHIFT-17 FRONTIER (exact next actions, priority order):
1. **THE FLAG-FLIP DECISION — the mechanism is CLEAN but SLOW; flip after a FULL-DEFAULT (15-stint) run proves Lapras crosses L43 (Ice Beam).** The full-length look-ahead COMPLETED clean (`/g/temp/longrun/ns17_vrgrind_full.log`): **0 Viridian abandons, 0 boulder shoves, no park, returns to Indigo** — but the pace is SLOW: **Lapras L39→41 over 6 stints (~4 min)**, because the participation-XP switch gives the fielded mon only a SHARE of the kill XP (the ace does the killing) and VR2F wilds cap ~L46. ⚠️ NB `recon_vr_grind_smoke.py` HARDCODES `max_stints=6` — the REAL `prep_e4_in_victory_road` default is **15 stints / 2400s**, so a real run grinds ~2.5× longer → Lapras would reach ~**L44-46, crossing L43 → Ice Beam** (the key E4-Champion/Lance coverage, via the NS#16 move-room fix), + Kadabra a few levels. It will NOT reach the full E4 milestone L55 (VR2F band caps ~L46). **THE FLIP GATE:** run the REAL method default (edit the smoke to `max_stints=15` or call `prep_e4_in_victory_road()` with no cap) from indigo_reach_g and confirm (a) Lapras crosses L43 and LEARNS Ice Beam (grep `Ice Beam|reserved a slot`), (b) still 0 Viridian/boulder/park over 15 stints, (c) bounded wall-clock (~20-40 min, watchable). If clean → **set `CAVE_GRIND_ENABLED` default "1"** (campaign.py:125, one char) + commit; the lever is armed (strictly beneficial — it levels the bench + lands Ice Beam, and is park-guarded so it can't make a run worse). **PACE-EFFICIENCY LEVER (optional, if the ~30-min grind reads too slow on a watch):** the slowness is the participation-share; `SOLO_WEAK_GRIND` (default OFF, campaign.py — let the fielded mon take the KILL XP directly when it can survive the wild) would ~2× the rate, but it's wedge-prone (the in-battle switch) → verify carefully. Or raise the ACE_BAIL floor so stints run longer.
2. **THE FINAL-PROOF GATE (the real win): a full `_enter_league` look-ahead from a THIN badge-8 fixture where the VR2F grind levels the bench and the E4 Champion FALLS.** The champion clear (NS#1) proved a leveled team incl. Lapras Ice Beam beats the E4. This shift proved she now LEVELS in VR2F. The open question is whether VR2F's ~L46-50 cap + Ice Beam is ENOUGH to clear the E4, or whether a residual gap remains. Run `E4_STRIKE_STATE`/`recon_enter_league_smoke` with FULL budgets (`POKEMON_VR_GRIND_BUDGET_S=3000 POKEMON_E4_DEADLINE_S=5400`) from indigo_reach_g (or a fresher thin badge-8 bank) and watch whether the leveled team clears the gauntlet. If a residual gap: the levers are (a) grind LONGER (the bench can slowly pass L46 on VR2F, just inefficiently), (b) the keeper router / earlier bench-leveling so she arrives less thin, (c) Cerulean Cave L46-67 is POST-E4 only so it can't help pre-credits.
3. **RE-RUN cmds:** VR-grind e2e = `POKEMON_CAVE_GRIND=1 GRIND_TARGET=43 ../.venv/Scripts/python.exe -u recon_vr_grind_smoke.py` (bench rises, 0 Viridian/boulder, returns to Indigo). Full dispatch = `POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_enter_league_smoke.py` (grind→Indigo→e4 boot→gauntlet). Indigo probe (heal paths) = `../.venv/Scripts/python.exe -u recon_indigo_grind_probe.py`. Fixture rebuild = `../.venv/Scripts/python.exe -u recon_mk_midvr_fixture.py` (banks `midvr_g` at VR1F). Decision checks: `recon_grind_spot_check.py` (28/28), `recon_lopsided_grind_check.py` (ALL PASS).

**WATCH STATUS: canonical Champion bank CLEAN + untouched (all NS#17 work on scratch fixtures midvr_g/indigo_reach_g); she is currently banked at the post-game Champion save; pop-in = `python pokemon_agent/watch.py`. The endgame VR2F team-depth grind is BUILT + wired + e2e-verified but flag-OFF (one no-park look-ahead from armed).**

---

## ✅ NIGHT-SHIFT #16 DONE (2026-07-11, night_shift.ps1 shift 16) — killed the "Lapras has no Ice move" ROOT (grind-mon level-up moves now STICK) + BUILT the foundational cave step-encounter grind (the endgame grind-spot unblock, flag-OFF PARTIAL). THREE commits (`e516266` move-learn fix, `aa85b39` cave-grind, `bc37955` docs), mode-side, canonical UNTOUCHED. START HERE → the binding wall is LEVELING the bench, and cave-grind is now a partially-built lever toward it (needs heal-in-cave + VR wiring).

### ✅ COMMIT 2 THIS SHIFT (`aa85b39`) — CAVE STEP-ENCOUNTER GRIND (foundational grind() capability, flag `POKEMON_CAVE_GRIND` DEFAULT OFF, byte-inert):
The endgame grind-spot-adequacy wall (NS#1/#14): near Viridian/Indigo the only adequate high-level GRASS (Route 23) is split-map/Surf-gated, while **Victory Road (L36-46, cave)** sits ON THE PATH but `grind()` bailed `no_safe_grass` in a cave (no grass tiles — wilds fire per STEP). Built: `grind()` now, when no grass + `CAVE_GRIND_ENABLED` + the map is a KB `terrain:cave` (`_grind_is_cave`), wanders the cave's walkable non-warp tiles (`_cave_walk_waypoints`, reusing the proven catch_one cave-wander) to draw step-encounters — the wander loop fights them exactly like grass. Warp-containment via `_cave_grind_avoid` (unions `tv.read_warps` with `_door_tiles` — the authoritative warp table catches exits the 0x60-0x6F heuristic misses).
- **THREE-STATE (honest PARTIAL): COMPILES + CORE MECHANIC VERIFIED** (`ns16_mtmoon.log` smoke: recognized Mt. Moon (1,1) as a cave, wandered, drew **8 step-encounter battles**; move-learn fix fired on the grind mon too). **NOT yet robust/wired — 3 remaining pieces (the next shift's build):**
  1. **HEAL-IN-A-CENTER-LESS-CAVE.** The solo-Ivysaur smoke's heal excursion walked OUT of Mt. Moon to a Center and couldn't resume. The INTENDED use (party≥2 participation-switch grind, where the protected ace tanks + rarely heals) is untested — I have NO fixture inside an adequate Center-less cave (banked_VICTORY + indigo_reach_g are at Indigo/Route-23, POST-VR; recon_grind_bench's `walk_to_map(1,39)` can't cross the Route-23 split-map wedge INTO VR — only the VR strike navs that). Need a mid-VR party-6 fixture (run `victory_road.run_strike` from giovanni_kit_g, bank a mid-VR state) to smoke the real case.
  2. **ENDGAME WIRING.** cave-grind is a grind() capability but the endgame dispatch (VR strike) CROSSES VR, it doesn't grind. To help a fresh underleveled team, either make the VR strike grind-along-the-way OR make `prep_for_e4` route back into VR to grind (E4-prep runs at Indigo, post-VR). Unwired = cave-grind currently helps nothing on the real endgame path.
  3. **DO NOT flip `POKEMON_CAVE_GRIND` until a party-6 VR smoke proves she levels without leaking + the wiring lands.**

### ✅ WHAT SHIFT 16 BUILT + COMMITTED (`e516266` — general, single-point + one sibling; canonical UNTOUCHED):
Attacked the NS#15 frontier's single highest-payoff team-depth item (teach Lapras an ICE move). RECON found: Lapras learns Ice Beam by LEVEL-UP at **L43** (the champion-clear Lapras L72 had it this way — NO TM acquisition needed; TM13 is a Game-Corner-coin prize, an unbuilt subsystem, so the level-up path is the clean one). The gap was a MOVE-MANAGEMENT bug on the grind path:
1. **THE BUG (code-confirmed):** `_ensure_move_room` (frees a junk slot so a level-up move auto-learns with no un-actuatable box) is called ONLY at the free-roam leveling pick (`campaign.py:11308`) + gym prep (`:3902`) — both on the CURRENT slot-0 lead. But `grind_weak_members` (`:6707`), `_road_bench_xp_arm` (`:6807`), and `recon_grind_bench` (`:246`) all REORDER the weak specialist to slot 0 AFTER that, then level it. So the fielded grind mon's slot was NEVER freed → at its learnset threshold (Lapras L43) the "Delete a move?" box appears un-actuatable → learn DECLINED. Confirmed on every fixture: giovanni_kit_g/indigo_reach_g Lapras L37-39 = `[Surf, Body Slam, Confuse Ray, Perish Song]`, no Ice attack.
2. **THE FIX:** call `_ensure_move_room()` at the TOP of `grind()` (the ONE primitive all grind callers funnel through — slot 0 IS the grind mon there, so this covers grind_weak_members + recon_grind_bench + every caller) + at the `_road_bench_xp_arm` reorder site (that path levels via road battles, not grind()). Self-limiting (no-op once <4 moves) + safe (the `all_precious` guard keeps a good set intact — only genuine junk is shed; the ace's precious set is never gutted).
3. **VERIFIED (three-state: COMPILES + WIRED to the autonomous grind path + VERIFIED both ways):**
   - **Deterministic isolation (`recon_grind_moveroom_check.py`, PASS):** boot giovanni_kit_g, reorder Lapras (L37) to slot 0 (what the grind does), call `_ensure_move_room()` → it sheds junk **Confuse Ray**, KEEPS **Surf** (the STAB hard-hitter), freeing the slot the L43 Ice Beam auto-learns into. On-target proof for the exact species.
   - **END-TO-END live grind (`ns16_drowzee.log`):** a fuchsia_lapras_kit **Drowzee L13→18** grind at Route 18 crossed its L17 threshold and **Headbutt (70pw) LANDED** in a live battle (dropped junk Disable via the freed slot; was declined pre-fix). Proves the full grind→level-up→auto-learn chain — the exact chain that now gives Lapras its Ice Beam at L43.
   - Combined with the project-wide `_ensure_move_room` invariant (freed slot ⟹ auto-learn, proven for the ace every run), the fix guarantees any grinding specialist's coverage move lands (Ice Beam, Kadabra Psychic, Snorlax TMs-by-levelup, etc.) — a GENERAL team-depth win, not just Lapras.

### ⇒ SHIFT-16 FRONTIER = LEVELING THE BENCH (grind-spot adequacy at the endgame is the remaining binding wall for a fresh-GO credits roll):
The coverage-MOVE half is now handled (this shift) + keeper coverage-CATCHES (keeper router, default ON) + the nav rope (NS#15). What's left is **the bench actually reaching its milestones** — esp. Lapras reaching L43 (Ice Beam) and the team reaching the E4 floor (~L55). The binding blocker is the KNOWN grind-spot-adequacy wall: near Viridian/Indigo (badge 8 / pre-E4) the only adequate high-level GRASS is **Route 23 (L34-49)**, whose south grass is split-map / Surf-gated (the NS#13 wedge — `_better_grind_spot` sees only the first hop). Every other endgame option is a CAVE (Victory Road L36-46, Cerulean Cave) → no grass → grind() returns `no_safe_grass`. So a fresh badge-8 team can't level past ~L40 to reach the E4 floor. Precisely-scoped next levers (verify-gated — a fresh multi-gym look-ahead confirming no treadmill/park, NS#1's hard gate):
1. **CAVE STEP-ENCOUNTER GRINDING — FOUNDATIONAL PIECE BUILT THIS SHIFT (`aa85b39`, flag-OFF PARTIAL); finish it (heal-in-cave + endgame wiring), then flip.** The grind() cave-wander mechanic is proven (8 step-encounters in the Mt. Moon smoke). REMAINING (see the "COMMIT 2" block above): (a) heal-in-a-Center-less-cave for the party≥2 participation-switch case — smoke it with a mid-VR party-6 fixture (build one via `victory_road.run_strike` from giovanni_kit_g, banking a mid-VR state); (b) wire it into the endgame (VR strike grinds-along, or prep_for_e4 routes into VR); (c) flip `POKEMON_CAVE_GRIND` once she levels in VR without leaking. VR wilds L36-46 = ADEQUATE to level a L37-43 team to the E4 floor ON THE PATH, and Lapras crosses L43 → Ice Beam (this shift's move-learn fix lands it) — the two NS#16 fixes compose to solve the whole team-depth wall once cave-grind is finished + wired.
2. **Route-23 south-pocket nav (the alternate grass unblock).** Make the split-map south grass reachable in the world graph (water-crossing within a split map; the late team HAS Surf). Deep nav work; `_better_grind_spot` only checks the first hop so it currently proposes Route 23 then wedges at its grassless north edge.
3. **VERIFY the whole chain: a fresh badge-8 (or badge-6 koga_done_kit) look-ahead through VR→E4 where the team LEVELS (via lever 1/2) past L43+coverage and the E4 Champion FALLS.** The champion clear (NS#1) already proved a leveled team incl. Lapras Ice Beam beats the E4 — so once leveling works, credits follow.

**RE-RUN cmds:** move-room isolation check = `../.venv/Scripts/python.exe recon_grind_moveroom_check.py` (PASS). End-to-end learn demo = `GRIND_STATE=fuchsia_lapras_kit GRIND_SPECIES=96 GRIND_TARGET=18 GRIND_MAP=3,36 GRIND_DIR=west GRIND_MIN=11 ../.venv/Scripts/python.exe -u recon_grind_bench.py` (Drowzee learns Headbutt). Real-flow validation (ran this shift) = `POKEMON_LOPSIDED_GRIND=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 38` → `/g/temp/longrun/ns16_flow.log` (grep `reserved a slot|GRIND SWITCH|badges=|outcome=stuck`). Canonical Champion save UNTOUCHED (all work on scratch fixtures).

---

## ✅ NIGHT-SHIFT #15 DONE (2026-07-11, night_shift.ps1 shift 15) — the badge-8→credits NAV ROPE IS COMPLETE + ARMED. Wired BOTH endgame dispatches (Victory Road + Elite Four), fixed the switch-freeze that blocked the VR climb, and flipped both default ON — the FULL chain (head_to_league→VR→Indigo→enter_league→E4 gauntlet) is e2e-proven in ONE look-ahead from a THIN badge-8 team. THREE commits (`47d36f8` wiring, `5673a4f` switch fix, `9130531` flip ON). START HERE → TEAM-DEPTH is the ONE remaining constraint for a fresh-GO credits roll.

### 🎯 THE STATE: the autonomous nav rope bedroom→credits is DONE. badges 1-8 ✅ + Victory Road ✅ + Elite Four ✅ — all wired in-loop, e2e-proven, freeze-free. A fresh GO now WALKS the whole route and only WALLS at the E4 Champion on TEAM-DEPTH (a bounded whiteout-loop, NOT a freeze). Building a real leveled 6 with SE coverage = the last mountain before credits.

### ✅ WHAT SHIFT 15 BUILT + COMMITTED (3 commits: `47d36f8` wiring, `5673a4f` switch fix, `9130531` flip ON; canonical UNTOUCHED):
The shift-14 frontier was: WIRE the VR endgame dispatch, THEN extract+wire E4. BOTH done as ONE coherent unit (same extract-and-hook pattern as giovanni_gym/victory_road):
1. **`e4_strike.py`** — a FAITHFUL port of `recon_e4.py`'s champion-clear-proven dispatch loop into `EliteFour` + `run_strike(camp, log, dbg_dir)` on the LIVE bridge (mirrors victory_road.py). Drives: Indigo exterior → League center (heal + FR-first mart stock-up via camp._mart_buy_one) → the 5-room gauntlet (Lorelei→Bruno→Agatha→Lance→Champion Gary, talk-triggered, north door opens on the DEFEATED flag) → **Hall of Fame == CREDITS**. Whiteout-tolerant (DEFEATED flags ratchet; a mid-gauntlet whiteout re-laps only UNCLEARED rooms) + resume-safe (map-keyed dispatch). Battles via `camp.battle_runner` (the 23487e7 champion battle-brain). Returns `'credits'` (the summit) | `'battle_loss'` | `'stuck'` (deadline — usually team-depth).
2. **`campaign.py` — THE ENDGAME DISPATCH.** After Giovanni (badge 8) `next_gym` is None, `post_game` is still False, and VR is a DUNGEON (not a gym/gate-flag → not a questline strike), so `_available_actions` offers explicit endgame actions at 8 badges pre-credits (the map guard makes them mutually exclusive):
   - `head_to_league` (badges==8, NOT at Indigo, `VICTORY_ROAD_ENABLED`) → `victory_road.run_strike` (its OWN road Viridian→Indigo).
   - `enter_league` (badges==8, AT Indigo (3,9), `E4_STRIKE_ENABLED`) → `e4_strike.run_strike` (→ HoF == CREDITS).
   Flags `POKEMON_VICTORY_ROAD` / `POKEMON_E4_STRIKE` **default OFF** (flip TOGETHER once e4_strike proves credits — VR alone would idle a fresh GO at Indigo). Dispatch via `_head_to_league` / `_enter_league` in `_route_action`. `ENDGAME_INDIGO=(3,9)`.
3. **`recon_longrun.py`** — the look-ahead chooser rides `head_to_league`/`enter_league` (models a player at 8 badges pushing to the League), so a look-ahead exercises the wiring.
4. **`recon_endgame_gate_check.py`** — DECISION-VERIFIED **8/8**: giovanni_kit_g (badge 8, Viridian) VR-ON offers head_to_league not enter_league; indigo_reach_g (badge 8, AT Indigo) E4-ON offers enter_league not head_to_league; flags OFF → neither (byte-inert); canonical Champion (post_game) → NEITHER (endgame is pre-credits only — the firewall holds).
5. **`recon_e4_strike_smoke.py`** — isolated e4_strike smoke harness (boot indigo_reach_g at Indigo → run_strike → gauntlet).
6. **`5673a4f` — the SWITCH-FREEZE FIX (battle_agent `_best_switch_slot`)** — the VR look-ahead livelocked in the Route-22 Gary fight (123 `MATCHUP SWITCH` churns, no attack). ROOT: the offensive-specialist pick was TYPE-based, fielding a mon with an SE TYPE but no SE MOVE (Lapras Ice-type but no Ice move vs Gary's exeggcute) → A↔B ping-pong forever. FIX: gate on a REAL damaging SE move (`r_eff` via `st.read_party_moves`+`st.move_info`+`_eff`). Killed the churn (123→0), she wins Gary + clears VR. The proven E4 specialists (Kadabra Psybeam, champion-Lapras Ice Beam) still qualify — mode-side, canonical-safe.
7. **`9130531` — FLIPPED both flags default ON** (`POKEMON_VICTORY_ROAD` + `POKEMON_E4_STRIKE`), 8/8 re-verified incl. the canonical post_game firewall. The rope is ARMED.

### ✅ VR DISPATCH BEHAVIORALLY PROVEN + the Route-22 Gary FREEZE FIXED (giovanni_kit_g look-ahead, both flags ON):
tick-1 `opts=[..head_to_league..] → PICK head_to_league` → `🏔️ VICTORY ROAD strike` → `EDGE west (3,1)→(3,41)` (Viridian→Route 22) → the **Route 22 GARY fight**. FIRST run (`ns15_endgame.log`) LIVELOCKED there: 123 `MATCHUP SWITCH` churns, slot0↔slot3 EVERY turn with no attack — a hard FREEZE. **ROOT (found + FIXED this shift, committed `5673a4f`):** `_best_switch_slot`'s offensive-specialist pick used `_matchup_off(TYPES)`, so it fielded a mon whose TYPE is SE but that has NO actual SE MOVE — giovanni_kit_g's **Lapras is Ice-TYPE (2x vs Gary's Grass exeggcute) but its moveset is [Surf, Body Slam]** (no Ice move); Venusaur (Poison-TYPE 2x vs Grass, no Poison move) and Lapras each read as the other's "specialist" → A↔B ping-pong forever. FIX: gate the pick on a REAL damaging SE move (`r_eff` via `st.read_party_moves`+`st.move_info`+`_eff`), not just an SE typing — strictly more correct (the proven E4 specialists Kadabra/Psybeam + champion-Lapras/Ice-Beam still qualify). **VERIFIED e2e (`ns15_endgame2.log`, both flags ON): MATCHUP SWITCH churns 123→0; she ATTACKS exeggcute with Strength → WINS Route-22 Gary (`RIVAL beat #7 won=True`) → crosses the gate → `ROUTE 23 (Gary handled en route)` → `VICTORY ROAD 1F` → `1f-switch barrier open=True` — climbing VR** (was: frozen at Route 22 forever). The VR endgame dispatch + its whole road now runs for a THIN badge-8 team.

### ✅ e4_strike MACHINERY PROVEN e2e (isolated smoke from indigo_reach_g, `ns15_e4smoke.log`):
boot at Indigo (3,9), badge8, $13k → enter-center → `heal_nearest ok` → `[shop] bought 4x Full Restore + 1x Full Heal` (FR-first, camp._mart_buy_one) → league-door → **room #1 Lorelei ✅ → #2 Bruno ✅ → #3 Agatha ✅ (the NS#13 wall — CLEARED) → #4 Lance ✅ → #5 CHAMPION Gary** (reached the Champion — FURTHER than NS#13, which whited out at Lance's Aerodactyl) → lost to Gary → whiteout → `back at the center (whiteout) — re-checking the kit [FR x0]` → the XP-ratchet loop RE-LAPS (DEFEATED flags persist; Lorelei-Lance pass through). **The entire e4_strike dispatch/nav/shop/heal/5-room-gauntlet/whiteout-recovery machinery is BYTE-FAITHFUL to recon_e4 — the ONLY wall is TEAM-DEPTH at Gary** (indigo_reach_g's L71 Venusaur [Razor Leaf 0.25x vs Charizard] + L39 Lapras [Surf 2x but dies early] + L40 Kadabra can't out-attrition the Champion — the champion clear needed L90/L72/L58 with Ice Beam). Same team-depth wall the whole project's final constraint. The smoke will deadline at `stuck` (the E4-self-grind loop is FUTILE per NS#14: Venusaur hogs KOs, the bench doesn't level, Charizard is a hard 0.25x type-wall).

### ⇒ SHIFT-15 FRONTIER = TEAM-DEPTH (the ONE remaining constraint for a fresh-GO credits roll; the nav rope is DONE):
**The full endgame chain is WIRED, ARMED (flags ON), and e2e-proven — there is NO more nav/dispatch work.** What a fresh GO needs now is a real leveled 6 with SE coverage so she doesn't wall at the E4 Champion. This is the pass-3 TEAM-DEPTH mountain (well-documented, careful, verify-gated — see `TEAM_DEPTH_ROOT_FIX.md` + the NS#1-11 levers):
1. **TEAM-DEPTH is THE binding wall (proven this shift on giovanni_kit_g's thin team: solo L70 Venusaur + L8-39 bench, no coverage).** She now WINS Route-22 Gary + clears VR + runs the full E4 gauntlet (Lorelei→Bruno→Agatha... proven), but WALLS at the Champion (Gary's Charizard = a hard 0.25x wall for Razor Leaf; needs a leveled Lapras Surf-2x / Ice answer that survives). A fresh `og_postopening`→credits look-ahead will now walk the WHOLE game and only stall on team-depth at the E4. The levers: keeper router (coverage catches), bench-grind / lopsided-grind (level the bench), and esp. **MOVE/TM intelligence — teach Lapras an ICE move** (the champion clear used Ice Beam; the constant "Lapras has no Ice move" gap is the single highest-payoff coverage fix for both Gary's Charizard AND Lance's dragons). ⚠️ Every team-depth lever is verify-gated (a fresh multi-gym look-ahead confirming no treadmill/park — NS#1's hard gate); do NOT ship a grind-cadence change blind.
2. **WATCHABILITY REFINEMENT (lower pri, now that the freeze is gone): the E4 whiteout-loop duration.** A thin team that reaches the E4 whiteout-LOOPS at the Champion (recon_e4's XP-ratchet, `POKEMON_E4_DEADLINE_S` default 5400s) — bounded, no freeze, but a long ugly stretch on a live watch. Once team-depth lands (she wins), this is moot; if a live fresh-GO needs it sooner, lower the default or make e4_strike return 'stuck' after N losing laps so the campaign falls back to grind (a bounded-attempt design vs the ratchet). Env-tunable today.
3. **THE deep WHITE-BOX MENU WEDGE still lurks (DON'T-FIX-BLIND).** The `use_item` action-menu impostor (B-drain recovers it, ~52×/E4 run) is slow but RECOVERABLE now (it no longer compounds a freeze — the switch fix removed the ping-pong that made it fatal). Still the E4-livelock-family root ([[pokemon-e4-livelock-family-killed]]); needs rule-15 live frame-grabs, not a blind edit. Lower priority (recoverable).

**THE CREDITS CHAIN STATUS: badges 1-8 autonomous ✅ | Victory Road wired+ARMED ✅ | Elite Four wired+ARMED ✅ | the switch-freeze that blocked VR-Gary FIXED ✅. The NAV ROPE bedroom→credits is COMPLETE + e2e-proven from a thin badge-8 team. TEAM-DEPTH (a real leveled 6 with SE coverage — esp. an Ice move) is the LAST binding constraint for the actual fresh-GO credits roll.**

**RE-RUN cmds (⚠️ backgrounded cmds LOSE cwd → `cd /g/JonnyD/NeuroAI_Bot/pokemon_agent &&` inside):**
- FULL endgame chain proof (flags now default ON): `cd pokemon_agent && POKEMON_LOPSIDED_GRIND=0 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py giovanni_kit_g.state 45` → grep `PICK head_to_league|VICTORY ROAD strike ->|reached_indigo|PICK enter_league|\[e4\] room #` (proven: won Gary, cleared VR, Indigo, E4 gauntlet Lorelei→Bruno→Agatha).
- Endgame decision check: `../.venv/Scripts/python.exe recon_endgame_gate_check.py` (8/8, incl. canonical post_game firewall).
- e4_strike isolated smoke (reaches Champion, walls team-depth): `E4_STRIKE_STATE=indigo_reach_g SMOKE_DEADLINE_S=1500 ../.venv/Scripts/python.exe -u recon_e4_strike_smoke.py`.

**Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures). pop-in = `python pokemon_agent/watch.py`.**

---

## ✅ NIGHT-SHIFT #14 DONE (2026-07-11, night_shift.ps1 shift 14) — WIRED the autonomous badge-8 DISPATCH (the shift-13 frontier blocker): a PROACTIVE EARTH-BADGE gate + questline-strike registry entry that fires `giovanni_gym.run_gym` (its OWN Cinnabar→Viridian sea road, bypassing head_to_gym's sea-nav gap) via the proven FIRE-FIRST machinery. Decision-derivation verified (earth_badge cap → `('flag','FLAG_BADGE08_GET')`, door-less, from 3,8). Flag `POKEMON_GIOVANNI_GYM` STILL default OFF pending the look-ahead — VERIFYING NOW from `blaine_done_kit`. START HERE.

### ✅ WHAT SHIFT 14 BUILT (uncommitted until the look-ahead confirms — chose OPTION 1 = proactive dispatch, the frontier's recommended fix):
The shift-13 blocker was: post-badge-7 head_to_gym PARKS on Route 21 (can't complete the Cinnabar→Viridian surf legs), so `beat_gym(Giovanni)` never dispatches the smoke-proven giovanni strike. FIX (5 pieces, mirrors the mansion/seafoam wiring pattern exactly — reuses the proven FIRE-FIRST questline-strike machinery, but the strike drives its own sea road):
1. **`giovanni_gym.py`** — added `GIOVANNI_ANCHORS = {Cinnabar 3,8, R21 3,40, R21 3,39, Pallet 3,0, Route1 3,19, Viridian 3,1, GYM 5,1}` (the sea road + gym; FIRE-FIRST fires the strike wherever she stands on the road → dispatches from Cinnabar the moment next_gym→Giovanni AND re-fires on a mid-road resume).
2. **`gamedata/frlg_gates.json`** — added `earth_badge` capability (kind=event, via=strike, from=3,8, prereq=null, sets_flag=FLAG_BADGE08_GET). VERIFIED: `_step_from_cap('earth_badge')` → success `('flag','FLAG_BADGE08_GET')`, door-less, from '3,8' (flag id 2087 in KB for live-cross-check).
3. **`campaign.py _giovanni_gate()`** — badge-7-held (0x826) AND badge-8-unset (0x827) → Gate(missing='earth_badge', where=Viridian). Mutually exclusive with Blaine's gates (which need 0x826 unset).
4. **PROACTIVE EARTH-BADGE recognition** in `_ensure_forward_questline` (after the mansion block) — next_gym==Giovanni + GIOVANNI_GYM_ENABLED → open the errand.
5. **Registry entry** in `_questline_strike`: `('flag','FLAG_BADGE08_GET') → _giovanni` (importer returns `run_gym, GIOVANNI_ANCHORS, ('badge',), 'giovanni_probe'`) + flag-gate guard (OFF → return None → fall through). Byte-inert with the flag OFF.

### ✅ ALSO BANKED SHIFT 14 (committed `741007b`) — `victory_road.py` (the VR extraction, built + SMOKE-PROVEN, unwired/flag-OFF; the NEXT gate on the credits chain):
The autonomous badge-8 dispatch (above) is COMMITTED + proven. Continued the climb: extracted `recon_victory.py` (942-line proven VR vehicle — already banked indigo_reach_g on the champion climb) into an in-loop `victory_road.VictoryRoad` + `run_strike(camp, log, dbg_dir)` on the LIVE campaign bridge (SAME giovanni_gym extraction pattern). Faithful port of the whiteout-tolerant map-dispatch loop: Viridian→Route 22 (Gary fight)→gate→Route 23→VR boulder floors (the hand-solved elevation-aware VR1F/2F/3F_PUZZLE constants verbatim, incl. the 3F row-19-boulder reveal via the (34,18) hole drop)→R23 north→Indigo Plateau + heal. Returns `reached_indigo`/`battle_loss`/`stuck`. EQ teach = Phase 0, DEFAULT OFF (`POKEMON_TEACH_EQ`; recon NS12 proved it did net harm — Razor Leaf x2 carries VR). **SMOKE PASS** (`recon_victory_road_smoke.py` from `giovanni_kit_g` badge-8 leveled fixture): Viridian→R22→Gary→gate→R23→VR floors (1f-switch, 2f-switch1, 3f-switch, 3f-drop-into-hole, 2f-switch2 ALL `barrier open=True`)→`VICTORY ROAD CLEARED`→`INDIGO REACHED (3,9)`, 50 battles, ~693s, whiteouts recovered gracefully. Extraction byte-faithful. NOT yet WIRED into the loop + flag-OFF by design (build-then-smoke cadence — exactly how giovanni_gym was built shift 13 then wired shift 14).

### ⇒ SHIFT-14 FRONTIER (exact next actions, priority order) — the VR+E4 endgame push, best done as ONE build:
1. **WIRE the VR endgame dispatch** (the NEW wiring — VR is NOT a gym/gate-flag questline; after badge 8 `next_gym`==None so `head_to_gym` is NOT even offered, and there's no single VR-cleared flag). The clean approach: OFFER an endgame "head to the League" action in `_available_actions` when ALL 8 badges held (0x827) AND map != INDIGO (3,9) AND `VICTORY_ROAD_ENABLED`; in its `_route_action` handler dispatch `victory_road.run_strike` (it drives its OWN road Viridian→Indigo, whiteout-tolerant, returns `reached_indigo`). STOP condition = map==INDIGO (the offer gates off there → E4 takes over). ⚠️ Ship `VICTORY_ROAD_ENABLED` **default OFF** until E4 is wired — a fresh GO reaching Indigo with no E4 dispatch would just idle there. Verify a look-ahead from a badge-8 fixture (giovanni_kit_g or a fresh-chain badge-8 bank) drives Viridian→Indigo autonomously (env-flag ON), then commit flag-OFF. NB the module is already smoke-proven, so this is pure WIRING (offer + handler + a `_victory_road_gate`-style guard).
2. **THEN the E4 vehicle** — `recon_e4.py` (682 lines: League-mart Full-Restore-first stock-up + the 5-room gauntlet Lorelei→Bruno→Agatha→Lance→Champion Gary → HALL OF FAME → CREDITS). SAME extract-and-hook: extract `e4_strike.run_strike(camp,log)` on the live bridge (mirror victory_road.py — reuse the drain/fight/handle_interrupts machinery), dispatch when AT Indigo (3,9). The battle-brain is proven on the champion climb (`23487e7` cracked the E4 wall: don't sleep-lock a 2x-SE foe; field the SE specialist reserve). Entering the Hall of Fame = CREDITS (auto-save, point of no return → write `CREDITS` as line 1 of NIGHT_REPORT.md). Then flip BOTH `VICTORY_ROAD_ENABLED` + the E4 flag ON together (the full Viridian→credits endgame chain).
3. **⚠️ TEAM-DEPTH is the binding constraint for a FRESH-GO credits roll (the parallel mountain, NOT the nav gates).** A fresh-chain badge-8 team is a solo L60 Venusaur + L8-25 bench. VR is survivable (Razor Leaf x2 carries — the smoke did it from the leveled giovanni_kit_g, but a fresh thin team is weaker). The E4 (L54-63 + Champion) needs a real leveled team — the champion clear used L90/L72/L58. So even with VR+E4 wired, a fresh GO likely walls at the E4 until team-depth lands (keeper router + bench-grind levers, the pass-3 carry-overs). The nav rope (badges 1-8 ✅ + VR module ✅ + E4 pending) is nearly complete; team-depth is what gates the actual fresh-GO credits.

**THE CREDITS CHAIN STATUS: badges 1-8 autonomous ✅ (badge 8 dispatch committed `9b669b9` this shift, e2e-proven). Victory Road module ✅ (committed `741007b`, smoke-proven; WIRING pending). E4 = extract+wire pending (recon_e4.py exists, champion-proven). Then team-depth is the binding constraint for the fresh-GO credits roll.**

**THE CREDITS CHAIN STATUS: badges 1-8 autonomous ✅ (badge 8 committed this shift). Remaining nav gates: Victory Road (module built, smoke-verifying, wiring pending) → E4 (recon_e4 exists, extract+wire pending). Then team-depth is the binding constraint for a fresh-GO credits roll.**

### ⇒ Carry-overs (lower pri): PP-famine flee-spam in lopsided grind (NS#10 2nd finding); grind-spot lever a (Route-23 nav-blocked, flag OFF); `POKEMON_BENCH_TO_MILESTONE` default-OFF.

**Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures).**

---

## ⏳ NIGHT-SHIFT #13 (2026-07-11, night_shift.ps1 shift 13) — SEAFOAM crossing wired + BOTH strikes flipped default-ON (committed `07aaff3`, Fuchsia→Cinnabar chain e2e-proven); then BUILT the next gate: the SECRET KEY (Mansion) strike + wiring (flag-gated `POKEMON_MANSION_STRIKE`, default OFF), decision-verified 10/10, smoke+look-ahead verifying now.

### ✅ WHAT SHIFT 13 BANKED (committed `07aaff3`):
The shift-12 seafoam wiring landed + ARMED. `seafoam_strike.run_strike` registered + PROACTIVE SEAFOAM-PREREQ recognition; both `POKEMON_SAFARI_STRIKE` + `POKEMON_SEAFOAM_STRIKE` flipped **default ON**. VERIFIED e2e: smoke `RESULT reached_cinnabar` (71s, SMOKE PASS from surf_ready_kit); wiring look-ahead (surf_ready_kit, both flags on) `PROACTIVE SEAFOAM-PREREQ → QUESTLINE STRIKE the Seafoam crossing → reached_cinnabar → STATE IN: Cinnabar, badges=6` (strike idempotent, does not re-fire). Decision checks: seafoam 10/10, safari 8/8. **The Fuchsia→Cinnabar autonomous chain is proven.** Post-Cinnabar she STALLS (bounces 3,8↔3,38) = Blaine's gym LOCKED behind the Secret Key = the next gate (built this shift, below).

### ✅ ALSO BANKED SHIFT 13 (committed `c7036d5`) — the SECRET KEY (Pokémon Mansion) strike, flag flipped default-ON:
`mansion_strike.run_strike` (faithful port of recon_mansion's Cinnabar→Mansion 4-floor statue-toggle nav → B1F Secret Key → out) registered in `_questline_strike` under `('flag','FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY')` + `secret_key` KB capability (prereq seafoam, from 3,8) + `_mansion_gate()` (crossed AND no-key) + PROACTIVE recognition. `POKEMON_MANSION_STRIKE` **default ON**. VERIFIED e2e: smoke `got_key` (87s, SMOKE PASS from cinnabar_kit_g); wiring look-ahead (cinnabar_kit_g, all 3 flags on) `PROACTIVE SECRET-KEY-PREREQ → QUESTLINE STRIKE → got_key → questline cleared → ENTERS Blaine's gym, clears junior trainers`. Decision: mansion 10/10, seafoam 10/10, safari 8/8. **The Cinnabar-reach → gym-door chain is proven.**

### ✅ ALSO BANKED SHIFT 13 (committed `ae3da43`) — the CINNABAR GYM (Blaine, badge 7) quiz-door strike, flag default-ON:
Cinnabar is FRLG's SIX quiz-door gym; general beat_gym can't open the doors so the leader never fires (the bounce the mansion look-ahead surfaced). `blaine_gym.run_gym` (faithful port of recon_blaine) does the FULL tour: 6 quiz doors (answers A/B/B/B/A/B) → Blaine → badge (0x826) → walk out → **B-drain the Bill "sail to Sevii?" AMBUSH** → heal. Hooked at the TOP of `beat_gym` (`if name=="Blaine" and BLAINE_GYM_ENABLED`) so the Bill landmine is contained in the strike. `POKEMON_BLAINE_GYM` **default ON**. VERIFIED e2e: smoke (secretkey_kit_g) all 6 doors → BLAINE battled → badge7=1 → bill-ambush drained → on_kanto, SMOKE PASS 41s; **full look-ahead (cinnabar_kit_g, all 4 flags): PROACTIVE SECRET-KEY → got_key → beat_gym DISPATCHES the Blaine strike → 6 quiz doors → badges=7 Volcano → 0 Sevii transitions → PICK head_to_gym Giovanni, marching north on Route 21.** The whole Cinnabar-reach → Mansion → Blaine → badge-7 → next-gym chain is autonomous.

### ✅ ALSO BANKED SHIFT 13 (committed `eda33fe`) — the VIRIDIAN GYM (Giovanni, badge 8) spin-tile strike, BUILT + smoke-proven (flag OFF, dispatch-blocked):
Viridian is a spin-tile maze (Rocket-Hideout class); general beat_gym can't cross it. `giovanni_gym.run_gym` (port of recon_giovanni) does the tour: sea road Cinnabar→Viridian (5 surfed north crossings, resume-safe: skipped when beat_gym already entered the gym) → `spin_nav` maze → Giovanni → badge 8 (0x827) → heal (NO exit ambush). Hooked at the TOP of beat_gym (`if name=="Giovanni" and GIOVANNI_GYM_ENABLED`). **Fixed a real DEAD-ZONE LIVELOCK** the port surfaced on a non-champion team (preleader heal-exit fires <0.85 but loop-top heal is <0.6 → a 0.60-0.85 lead exits-but-never-heals forever; fix = heal unconditionally after the exit). **SMOKE-VERIFIED** (blaine_done_kit → road+gym: 5 legs → maze → heal-exit once → 100% → Giovanni battled → badge8=1 → EARTH BADGE, SMOKE PASS 81s). **Flag STAYS DEFAULT OFF** — the strike works in isolation but the autonomous DISPATCH is blocked (below).

### ⇒ SHIFT-13 FRONTIER = the head_to_gym CINNABAR→VIRIDIAN SEA-NAV GAP (blocks the autonomous badge-8). START HERE.
**THE BLOCKER (confirmed in ns13_giovanni_wiring, blaine_done_kit + all 5 flags):** after badge 7 `next_gym` correctly becomes Giovanni and she marches north, but **head_to_gym PARKS on Route 21 north (3,39)@(15,36)** — the billed-road graph completes only `3,8→3,40→3,39` (Cinnabar→R21south→R21north) then oscillates (25 stall signals, no further map transitions). It can't complete the remaining surf legs (R21north→Pallet→Route1→Viridian), so she never reaches Viridian to trigger `beat_gym(Giovanni)` → the (smoke-proven) Giovanni strike never dispatches. NB the Giovanni strike's OWN `cross_edge` sea machinery crosses ALL 5 legs cleanly (the smoke does exactly this from Cinnabar), so the road is navigable — the gap is specifically in head_to_gym's billed-road sea handling (likely a world-model / billed-road-graph gap for the Cinnabar→Viridian legs; the champion climb used recon_giovanni's own road, so head_to_gym may never have learned these edges).

**TWO fix options (pick one, verify, then flip `POKEMON_GIOVANNI_GYM` ON):**
1. **PROACTIVE DISPATCH (recommended — reuses the proven strike road, bypasses the head_to_gym gap entirely).** Add a proactive recognition (like the questline strikes / the mansion+seafoam gates): when `next_gym==Giovanni` and `GIOVANNI_GYM_ENABLED` and she isn't already at/in Viridian, dispatch `giovanni_gym.run_gym` (which drives its OWN road Cinnabar→Viridian + the gym — smoke-proven end-to-end). Gate it so it only fires post-badge-7 (0x826 set, 0x827 unset). Verify with a look-ahead from blaine_done_kit → she runs the strike, badge 8, marches to Victory Road. Then flip the flag ON + commit.
2. **FIX head_to_gym's sea nav** (more general but harder): make the billed-road graph complete the R21north→Pallet→Route1→Viridian surf legs (check `gamedata/frlg_gates.json roads[Viridian]` + whether blaine_done_kit's world_model has these edges; the seafoam sea-crossing worked, so the machinery exists — it's the Viridian road graph that's the gap). Then beat_gym fires and the strike dispatches. This ALSO helps any future sea-gated routing.

**THEN: Victory Road → E4 → CREDITS.** VR needs Strength (banked by Safari) + Surf. `recon_victory.py` + `recon_e4.py` exist (champion tail scripts, unwired) — same extract-and-hook pattern (VR = boulder/ladder nav → a `victory_road` solver hooked where? it's not a gym — likely a questline/travel strike; E4 = the gauntlet, battle-brain already proven on the champion climb → an e4 strike). The autonomous fresh-GO chain to credits is within a few more of these gates.

**RE-RUN cmds (⚠️ backgrounded cmds LOSE cwd → always `cd /g/JonnyD/NeuroAI_Bot/pokemon_agent &&` inside, or run foreground):**
- Giovanni smoke (proves the strike): `cd pokemon_agent && GIOVANNI_STATE=blaine_done_kit ../.venv/Scripts/python.exe -u recon_giovanni_smoke.py` → grep `SMOKE PASS|EARTH BADGE`.
- Blaine smoke: `BLAINE_STATE=secretkey_kit_g ../.venv/Scripts/python.exe -u recon_blaine_smoke.py`.
- Full chain look-ahead (badge 6→7, PROVEN): `POKEMON_BLAINE_GYM=1 POKEMON_MANSION_STRIKE=1 POKEMON_SAFARI_STRIKE=1 POKEMON_SEAFOAM_STRIKE=1 POKEMON_LOPSIDED_GRIND=0 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py cinnabar_kit_g.state 25`.
- Giovanni dispatch look-ahead (currently PARKS on Route 21): add `POKEMON_GIOVANNI_GYM=1` + boot `blaine_done_kit.state`.

### ⇒ Carry-overs (lower pri): PP-famine flee-spam in lopsided grind (NS#10 2nd finding); grind-spot lever a (Route-23 nav-blocked, flag OFF); `POKEMON_BENCH_TO_MILESTONE` default-OFF.

**Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures).**

---

## ✅ NIGHT-SHIFT #12 DONE (2026-07-11, night_shift.ps1 shift 12) — EXTRACTED `seafoam_strike.py` + REGISTERED it + decision-verified 10/10; smoke+wiring proven, flags flipped ON in shift 13 (`07aaff3`).

### ✅ WHAT SHIFT 12 BUILT (uncommitted until the live look-ahead confirms):
1. **`seafoam_strike.py`** — a FAITHFUL port of `recon_seafoam.py`'s Fuchsia→R19→R20→Seafoam-interior(Strength boulder cascade → becalm B3F current 0x2D2)→R20west→CINNABAR crossing into `SeafoamStrike` + `run_strike(camp, log, dbg_dir)` on the LIVE campaign bridge (same shape as safari_strike). MISSION list verbatim. Anchors = {FUCHSIA, R19, R20} | SEAFOAM_MAPS (interior floors 1,83–1,86). Returns `reached_cinnabar` / `in_seafoam` (0x2D2 armed but exit incomplete) / `not_here` / `failed`. Idempotent (already-on-Cinnabar → arrival; mid-tour re-tick resumes; 0x2D2-set → skip to exit walk).
2. **frlg_gates.json** — added the `seafoam` capability (kind=event, via=strike, sets_flag=FLAG_STOPPED_SEAFOAM_B3F_CURRENT, prereq=surf, from=3,7) + the `FLAG_STOPPED_SEAFOAM_B3F_CURRENT` flag (id 722 / 0x2D2). Derivation VERIFIED: `_step_from_cap('seafoam')` → success `('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT')`, door-less, from_map '3,7', chain = surf(satisfied)→seafoam(actionable).
3. **campaign.py** — (a) `SEAFOAM_STRIKE_ENABLED` (`POKEMON_SEAFOAM_STRIKE`, default OFF) + FLAG_GOT_HM03_ID/FLAG_SEAFOAM_CROSSED_ID consts; (b) registered `('flag','FLAG_STOPPED_SEAFOAM_B3F_CURRENT') → _seafoam` in `_questline_strike` (flag-gated), good=`('reached_cinnabar',)`, added `in_seafoam` to the mid-dungeon exit-retry set; (c) `_seafoam_gate()` helper (has-Surf 0x239 AND NOT-crossed 0x2D2 → Gate(missing='seafoam', where=Cinnabar); mutually exclusive w/ the Safari gate on 0x239); (d) PROACTIVE recognition in `_ensure_forward_questline` (next_gym==Blaine + SEAFOAM_STRIKE_ENABLED → open the crossing errand). The seafoam step is DOOR-LESS + flag-keyed → fires via the BLOCKER-STRIKE-FIRE-FIRST block on an anchor, falls through to head_to_gym off-anchor.
4. **DECISION-VERIFIED 10/10** (`recon_seafoam_gate_check.py`): Blaine+Surf+uncrossed→seafoam gate; no-Surf→None (Safari owns it); crossed→None; Sabrina→None; KB derives the door-less flag step routed to Fuchsia; registry flag-gated. Safari check still 8/8 (no regression). `recon_seafoam_smoke.py` = isolated strike smoke harness (direct run_strike from surf_ready_kit).

### ⇒ SHIFT-12 FRONTIER (exact next actions, priority order):
1. **CONFIRM the smoke test** (`/g/temp/longrun/ns12_seafoam_smoke.log`, `SEAFOAM_STATE=surf_ready_kit ../.venv/Scripts/python.exe -u recon_seafoam_smoke.py`) → grep `SMOKE PASS|RESULT: reached_cinnabar|-> (3, 8)`. It proves the EXTRACTION is byte-faithful (boot Fuchsia (33,32) → R19 → R20 → interior boulder cascade → Cinnabar). If PASS → the strike is proven; if a bug → fix (the crossing itself is proven by recon_seafoam this shift, so any failure is an extraction slip).
2. **RUN THE FULL WIRING LOOK-AHEAD** — from surf_ready_kit (badge 6, Surf+Strength), with BOTH flags ON: `POKEMON_SAFARI_STRIKE=1 POKEMON_SEAFOAM_STRIKE=1 POKEMON_LOPSIDED_GRIND=0 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py surf_ready_kit.state 20` → grep `PROACTIVE SEAFOAM-PREREQ|QUESTLINE STRIKE.*Seafoam|reached_cinnabar|STATE IN: Cinnabar|MAP TRANSITION`. This proves the RECOGNITION→questline→strike wiring (not just the strike). LOPSIDED_GRIND=0 removes the thin-bench oscillation confound (surf_ready_kit has an L8–13 bench behind L57 Venusaur). SUCCESS = she recognizes the severed sea road, opens the crossing errand, runs the strike, reaches Cinnabar.
3. **THEN flip `POKEMON_SEAFOAM_STRIKE` + `POKEMON_SAFARI_STRIKE` default ON (campaign.py ~196–200) + COMMIT** all of shift 12 (seafoam_strike.py, the JSON, the campaign wiring, the two recon checks). Consider a full Fuchsia→Cinnabar chain proof on a GOOD-bench fixture (koga_done_kit: Safari→Surf→Seafoam→Cinnabar) before the double-flip if budget allows — the thin surf_ready_kit bench may confound a full chain, but the two strikes are independently proven.
4. **After Cinnabar: the rest of the chain** — Secret Key/Mansion strike (recon_mansion.py exists, unpromoted, same extract-and-register pattern) → `GYM_PREREQS['Blaine']` door → Blaine → Giovanni. Victory Road later needs Strength (already banked by the Safari strike) + Surf.
5. **Carry-overs (lower pri):** PP-famine flee-spam in the lopsided grind (NS#10 2nd finding); grind-spot lever a (Route-23 nav-blocked, flag OFF); `POKEMON_BENCH_TO_MILESTONE` default-OFF.

**Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures).**

---

## 🧒 THE #1 PRIORITY + SANITY-CHECK DISCIPLINE (2026-07-11, standing — every shift, ESPECIALLY team-depth)
**After every major step, step back and ask: "Would a competent 10-year-old playing this game do it THIS way?"**
If something would look DUMB to a human watching, that IS the signal to fix it — sanity-check your own output.

**THE RECURRING FAILURE (the ONE remaining constraint for a fresh-GO credits roll):** she arrives at the Elite Four
with basically **~3 usable mons — 1 massively-overleveled ace + underleveled scrubs** — instead of a **balanced team
of 6 leveled ~similarly**. That is NOT how a human plays. A 10-year-old arrives at the E4 with **6 real,
similarly-leveled team members.** THE SUCCESS BAR: on a FRESH bedroom→credits autonomous run she must arrive at
Indigo with **6 genuinely usable, appropriately-leveled Pokémon** (none stuck at catch-level; no L70-ace + L15-fodder;
all within a sane level band). Diagnosis of WHY (ace hogs XP; bench-leveling verbs unwired; no `prep_for_e4`) is in
`pokemon_agent/TEAM_DEPTH_ROOT_FIX.md`.

**⚠️ THE TRAP — why it keeps "getting fixed" but never lands:** the fixes fire MID-GAME (bench levels once at badge
3-4) but do NOT CARRY THROUGH to E4-readiness on a FRESH FULL run. **A mid-game bench fix that isn't VERIFIED as a
leveled-6-AT-INDIGO on a complete fresh bedroom→credits run DOES NOT COUNT.** Stop reporting "bench-to-milestone
works" as done — it's only done when a FINAL-PROOF dumps the party at Indigo and shows the real leveled 6. Make THAT
the gate: `recon_partydump` (or equivalent) at the Indigo bank of a fresh-run look-ahead → 6 usable, similarly-leveled.

## 🎲 THE RELIABILITY BATTERY (2026-07-11, standing) — the gate for "engine reliable enough for a LIVE run"
**Trigger: ONLY after the team-depth fix verifies** (a fresh-run FINAL-PROOF arrives at Indigo with a real leveled
6). THEN run **5 INDEPENDENT fresh bedroom→credits look-aheads at max headless speed (~14x)** — separate runs, not
one replayed. Auto-log a per-run report (this is what the end-of-run METRICS generator is for — build it if not yet):
- **total in-game playtime**, the **final 6-mon team + levels at the Elite Four**, the **route / notable choices**,
  and **outcome** (credits rolled Y/N; any crash / livelock / thin-team).
**THE BAR to declare RELIABLE (all four, no exceptions):** (1) **5/5 roll credits**; (2) **5/5 arrive at the E4 with
6 genuinely usable, similarly-leveled mons** (no L90-ace + L8-fodder); (3) **0 crashes**; (4) **0 livelocks**.
**VARIATION IS GOOD, NOT BAD:** different teams / different lengths across the 5 runs is the PROOF it's real
autonomy and not a scripted rail — show Jonny the spread. **Any run that arrives thin, crashes, or livelocks is a
REAL BUG → diagnose it** (don't average it away). This battery gates the conversation about a live run — it is NOT
the live-run readiness itself (that still needs the 30h live-soak, PHASE 2 below).
**ALSO LOG per-run WATCHABILITY PRE-SCOUT notes** (these DON'T gate the battery — they pre-scout the "wins but looks
dumb" cases for Jonny's live-watch, so headless surfaces them before he spends watch-time). Log what's CHEAP to
capture; don't let the extra logging delay/complicate the runs.
**⟐ THE FRAMING (why all these signals exist — the point of the whole pre-scout layer):** a "5/5 PASS" must mean
**"reliably beats the game AND plays like a thinking, curious, PREPARED human worth watching"** — NOT just "5/5
rolled credits." A run that rolls credits by robotic brute-force with no synergy / no team-building-via-catching /
no exploration is a **FAIL for our purposes** even if it technically finished. The battery must SURFACE these in
HEADLESS (cheap) so we DON'T discover them mid-live-stream (expensive — forces re-runs). So: the reliability CORE
still HARD-GATES (5/5 credits, leveled-6 at E4, 0 crash/livelock), and these signals make "pass" mean **"watchably
READY to stream," not "technically completed."** **Weight TYPE-SYNERGY + CATCHING + EXPLORATION heaviest** — they
are what make 30 hours watchable. Track + log ALL of it EVERY run.
**THE WEIGHTED-HEAVIEST (type-synergy + catching + exploration — make/break a watchable 30h run):**
1. **TYPE-SYNERGY + COMBINATION PLANNING (#1 — weight HIGHEST; "she's THINKING" vs "a bot walks around Pokémon"):**
   per hard fight (each gym / E4 member / Champion / notable trainer) log (i) **KNOWLEDGE-PREP** — did she PREPARE
   with the correct matchup in mind BEFORE the fight (built/brought the right type answer, e.g. "Brock=Rock→want
   Water/Grass", "Misty=Water→want Electric/Grass", the E4 members' types) like a player who read the strategy, vs
   walking in BLIND to improvise/brute-force? (built-in knowledge OR a lookup feature both count — log which); and
   (ii) **DEPLOY** — did she field the right answer AND **switch DELIBERATELY mid-fight for a matchup** (an Ice
   answer for Charizard, then back to the ace)? Log any team-composition reasoning surfaced. **FLAG any fight won by
   wrong-type brute-force / potion-tank when a super-effective answer existed on the team OR was buildable.** ("Won
   smart / prepared" vs "won dumb / blind" per boss.)
2. **CATCHING BEHAVIOR:** does she carry enough Poké Balls to grab interesting/rare encounters (a 10-yo stocks balls
   "just in case")? Catch a reasonable VARIETY + adopt caught mons into the lineup? Understand catch-'em-all exists?
   **FLAG: ignoring catching, running out of balls at a rare encounter, or solo-carrying with no team-building** (the thin-team root).
3. **TM/ITEM USAGE CORRECTNESS:** does she use earned strong items/TMs on SENSIBLE mons (not a rare TM on fodder),
   heal at sane times, no shop button-mashing? **⚠️ CRITICAL BUG-WATCH — this one CAN fail the battery (it's a
   LIVELOCK, not just a note): a TM/move-teach aimed at a Pokémon that CAN'T learn it must FAIL CLEANLY, never
   infinite-loop.** Flag any teach that could loop forever on an invalid target.
**Plus (lighter notes):** any single **grind stint >~10 game-min** (padding); **hesitation/confusion clusters** —
pointless backtracking, re-entering the same door, standing still (looks lost/confused even when she recovers);
**EXPLORATION TEXTURE ("curious human" vs robotic beeline):** per town — does she ever step into a non-objective
shop/building, **read signs / route markers / notice posted/environmental info**, or notice her surroundings, or
ONLY ever walk the shortest line to the next required tile? Flag BOTH
failure modes: (a) **pure objective-beeline** = zero exploration texture, reads robotic (un-watchable over 30h);
(b) **excessive wandering / completionism** = exhausting NPCs, nonessential sidequests, 100%-ing = padding (also
un-watchable). Target = the NATURAL MIDDLE — essentials done EXCELLENTLY + LIGHT organic curiosity (occasional
shop/building visits, glancing around). This is a MOVEMENT/behavior signal: the overworld can/should be mostly
SILENT — log the movement texture, NOT commentary (the talking layer is for Jonny's live-watch, not headless).
**DROP personality/soul-presence logging** — headless CANNOT judge "funny/endearing"; Jonny judges that on the
live-watch, so don't have the bot attempt analysis it can't do well. Flag each note with timestamp + location.
The battery still GATES on reliability (5/5 credits, leveled-6, 0 crash/livelock); these are the pre-scout layer.

## ⚡ SPEED LEVERS (2026-07-11, standing — every shift; these compound across the whole climb)
1. **Look-ahead already runs at ~14x MAX headless** (`recon_longrun.py`: no audio/TTS/render throttle, SDL dummy,
   MUSE_GAP=0). Keep verification runs there — never throttle a verification below the headless ceiling.
2. **Do NOT re-verify solved ground.** Segments pass 1/2 committed+verified (gyms 1-5, the routes/caves/gates) are
   TRUSTED — boot from a banked frontier kit and verify only the **NEW frontier** each shift. Don't re-run full gym
   boundaries already proven. Tighten setup: minimal overhead, get to the forward-grind fast.
3. **RUNNING SHOES = VERIFIED ACTIVE (nothing to build).** `FLAG_SYS_B_DASH` (0x82F) flips 0→1 at the forced
   post-Brock Route-3/Pewter Aide gate (confirmed 0 in brock_ready → 1 at badge3/endgame/LIVE stage), and
   run-by-default is wired (`travel.py:_press` co-holds B → the pret gate `B_BUTTON && FLAG_SYS_B_DASH` → ~1.85x
   outdoors the whole post-Brock game). She autonomously collects + runs. **Bike = low-pri OPTIONAL** (`use_bike()`
   stub; only faster than running on Cycling Road — skip unless en route to Giovanni via Cycling Road).
4. **Forward drive to the FINAL-PROOF gate:** Koga→Sabrina→Blaine→Giovanni→Victory Road→E4→credits on the fresh
   autonomous run — build/level/evolve the team as needed, prove each gym ONCE, **no over-grind (watchability), no re-solve.**

## 📡 THE REAL BAR + PHASE 2 (after the autonomous-credits asymptote): 30-HOUR UNATTENDED LIVE MARATHON
The FINAL-PROOF gate (fresh autonomous credits) is the asymptote for the BUILD — it is NOT the finish line.
**The real target = Kira runs LIVE, UNATTENDED, for a ~30-hour marathon stream with ZERO moments needing human
intervention.** That is a HIGHER bar than any headless verification (headless has no TTS/avatar/audio stack and no
multi-hour continuous uptime). So once the autonomous build lands, the NEXT phase is **LONG-DURATION LIVE-SOAK
HARDENING** (Jonny runs supervised multi-hour soaks to surface duration-specific failures headless can't):
- **Memory/resource stability over many hours** — no leak/degradation that's fine at hour 1 and crashes at hour 14
  (audit long-lived buffers, ever-growing lists/logs, VRAM/handle creep; add periodic self-checks if warranted).
- **Audio/TTS/avatar stack continuous 24-30h** — the fixed crash-category (PortAudio SIGSEGV, now child-isolated)
  must NOT resurface under SUSTAINED uptime; verify the isolate-child respawn budget + anything the pump accrues
  over thousands of restarts / many hours doesn't itself become the hour-N failure.
- **Rock-solid unattended crash-recovery via `supervisor.py`** — any failure RESUMES GRACEFULLY and keeps streaming,
  never a visible reset/break/frozen avatar on-stream. GO now runs under the supervisor (9ef24e5); harden the resume
  path so recovery is invisible to a viewer.
- **Flag short-run-only-safe risks** — anything acceptable in a 20-45min verification but risky over 30 continuous
  hours (unbounded caches, once-per-run assumptions, checkpoint cadence, log-file growth, clock/rollover).
**HONESTY:** long-duration live stability is bucket-(b) — only Jonny's supervised multi-hour soak proves it; NEVER
claim marathon-ready from headless. Build TOWARD the marathon bar, not just the beat-the-game bar.

## 🌟 PHASE 3+ NORTH-STAR (banked in `POST_CREDITS_VISION.md` §8-9; AFTER asymptote + live-soak — don't build now)
1. **NATIVE MEMORY / BOND / EMOTIONAL-CONTINUITY** — a standalone lived-memory system built HERE in Kira-local
   (KiraState + pokemon_soul hooks), FOR the run: she accumulates real memories (starter pick, first catch, hated
   bosses, the level-up that unlocked a win), references them live, pays off in an endgame recap. **Native — do NOT
   copy from the web app; flow is Kira-local → web app later.**
2. **CHAT MODERATION / PRESENCE** — wire Twitch mod-actions so she bans/timeouts herself, sassy-with-teeth; banning
   for vibe/bit/annoyance is fine + good content (even a playful wrong-ban). **HARD GUARDRAIL:** she is NOT the sole
   line on serious harm (CSAM/doxxing/credible-threats/hate-raids stay HUMAN-mod-backstopped). Inherently a live build.

## ✅ NIGHT-SHIFT #11 DONE (2026-07-11, night_shift.ps1 shift 11) — BUILT + e2e-VERIFIED the autonomous Blaine/Safari Surf-prereq wiring (recognize→strike→Surf→TEACH→clear, PROVEN live) + fixed TWO real general bugs the look-aheads fingered (Poké-Flute key-item routing; early-strike starving the teach bridge). FOUR commits (`b52e34e` safari wiring flag-OFF; `e45e693` flute-routing; `ea45b67` teach-ordering). Flag STAYS OFF: the frontier's own flip-bar ("the sea road to Cinnabar opens") is UNMET — post-Surf she can't sea-nav to Cinnabar (Route 19 is an UNVISITED map). **The sea-nav to Cinnabar is the precise next step before the flip.** START HERE.

### 🎯 THE FULL SAFARI/SURF CHAIN IS PROVEN e2e (fuchsia_lapras_kit, `POKEMON_SAFARI_STRIKE=1`):
`🌊 PROACTIVE HM-PREREQ: Blaine needs surf` → `🎯 QUESTLINE STRIKE: Safari Zone` → the extracted tour ran (enter→pay→classic loop East→North→West→Gold Teeth→Secret House HM03→Warden HM04→walk out, ~45s, 7 Safari battles) → `SAFARI DONE surf=True strength=True` → `got_surf` → `🧭 TEACH BRIDGE: surf → lapras taught` → `questline cleared`, ctx now reads *"You have SURF — you can cross water."* The isolated strike smoke-test (direct `run_strike` from fuchsia_lapras_kit) also returned `got_surf` cleanly in 45s — the extraction is byte-faithful.

### ✅ WHAT SHIFT 11 BANKED:
1. **`b52e34e` — the SAFARI STRIKE WIRING (flag-gated `POKEMON_SAFARI_STRIKE` default OFF).** Three pieces landed as ONE coherent unit:
   - **`safari_strike.py`** — a FAITHFUL port of recon_safari.py's proven classic-loop tour into `SafariStrike` + `run_strike(camp, log, dbg_dir)` on the LIVE campaign bridge (models silph_strike). Gets HM03 Surf + HM04 Strength + Gold Teeth; returns `got_surf`/`in_safari`/`not_here`/`failed`; idempotent by state, pre/post heal.
   - **REGISTERED in `_questline_strike`** under **`("cap","surf")` / `("cap","strength")`** — the ACTUAL derived success sig (an HM step's win is a party mon KNOWING the move; the strike fires BEFORE the teach bridge, which needs the HM already in the case). ⚠️ NB the NS#10 frontier GUESSED `("flag","FLAG_GOT_HM03")` — that was WRONG; `_step_from_cap` derives `("cap","surf")` (verified against the real KB). Added `in_safari` to the mid-dungeon exit-retry special-case.
   - **PROACTIVE recognition in `_ensure_forward_questline`** — `GYM_PREREQS["Blaine"]=(0x239,"surf",..)` + `HM_PREREQ_GYMS={"Blaine"}`. Unlike Sabrina's at-the-door prereq (which can NEVER fire for Blaine — Cinnabar is unreachable Surf-less), this opens the Safari errand BEFORE the uncrossable sea march. Scoped so Sabrina's at-door Silph path is untouched. **Decision-VERIFIED 8/8 (`recon_safari_gate_check.py`)** + the recognition FIRES LIVE (`🌊 PROACTIVE HM-PREREQ: Blaine needs surf` on koga_done_kit tick 1, driving a `('cap','surf')` questline step NORTH toward the Safari). Default-OFF path is byte-equivalent to before.
2. **`e45e693` — Poké-Flute KEY-ITEM routing bug (general, independent of the flag).** The first look-ahead drove her the WRONG way (north to Bill) because the surf questline's anchor-route to Fuchsia (3,7) returned None → fell to the KB's local `dir="north"`. ROOT: `_story_gate_avoid` gated the Route-12/16 (Snorlax) avoid on `_item_count(350)>0`, but the Poké Flute (350) is a **KEY item** and `_item_count`/`bag_count` only scan the ITEMS pocket (0x310) — so the check NEVER fired and Route 12/16 stayed avoided even POST-Flute, blocking the ONLY learned graph path south to Fuchsia. FIX: read the KEY-ITEMS pocket (0x3B8). VERIFIED on koga_done_kit (Flute + Snorlax woken 0x253): `story_gate_avoid` empties, `_next_step_rideable(→Fuchsia)` returns a valid hop (was None). Pre-Flute unchanged.
3. **`ea45b67` — early-strike starves the teach bridge (general; load-bearing for the flag-ON path).** The "BLOCKER STRIKE — FIRE FIRST" block (built for the door-less Snorlax) fired the strike BEFORE the teach bridge for ANY door-less step. The Safari `('cap','surf')` step is door-less → post-Safari (HM03 in bag, Surf UNtaught) the idempotent strike re-fired every tick (`got_surf`, battles 0), STARVING the teach bridge, so `('cap','surf')` never satisfied — she looped Fuchsia↔Route 15 with Surf untaught. FIX: skip the early strike for a `('cap',hm)` step whose HM is already in the case (acquisition done → teach owns it). Flag/item steps (Snorlax) unaffected. VERIFIED e2e (teach fired → taught → questline cleared).

### ⇒ SHIFT-11 FRONTIER — the SEAFOAM crossing to Cinnabar (the gating step before the flag flip), priority order:
1. **EXTRACT `seafoam_strike.py` from `recon_seafoam.py` (the one thing between the proven Safari wiring and flipping the flag) — SAME template as safari_strike this shift. ✅ DE-RISKED: `recon_seafoam.py` was run this shift from `surf_ready_kit` (the exact Safari-strike OUTPUT state — Surf+Strength taught, badge 6) and CROSSED TO CINNABAR CLEANLY in 76s (`ns11_seafoam_exec.log`: Fuchsia→R19→Seafoam 5-floor boulder puzzle→FLAG_STOPPED_SEAFOAM_B3F_CURRENT→west sea→CINNABAR (3,8)@(21,15), 13 battles → `banked_CINNABAR`). So BOTH executor halves of the Fuchsia→Cinnabar chain are proven live; only the wiring remains.** The extraction test start = `surf_ready_kit` (`SEAFOAM_STATE=surf_ready_kit`); `states/workshop` also has `cinnabar_kit_g`/`secretkey_kit_g`/`blaine_kit_g` post-crossing reference fixtures from the champion climb. REFINED diagnosis (isolated test `ns11_seanav.log`, surf_ready_kit + `POKEMON_LOPSIDED_GRIND=0` to remove the thin-bench confound): the billed road + surf-crossing **already WORK most of the way** — Fuchsia (3,7) `go:south` → **Route 19 (3,37)** → `go:west` → **Route 20 (3,38)**, arriving at (119,3) where ctx reads *"WEST → Cinnabar Island"*. The ONLY gap is the FINAL Route 20→Cinnabar hop: `EDGE-ROUTE (3,38)→(3,8) edge west` → `no clean path from (119,3) ... genuine wall/zone gap` → no_path → bounce. ROOT (per [[pokemon-seafoam-crossed-cinnabar]] + `recon_seafoam.py` header): **Route 20's surface is SEVERED at the Seafoam Islands** — the crossing is THROUGH the Seafoam interior (a STRENGTH boulder-cascade puzzle over 5 floors → FLAG_STOPPED_SEAFOAM_B3F_CURRENT 0x2D2 becalms the water → surf out the west exit → Cinnabar), NOT a plain surf-across. `recon_seafoam.py` (718 lines, PROVEN on the champion climb, needs Surf+Strength+badge-6 which the Safari strike now provides) is the exact solution. So this is a WIRING gap: extract recon_seafoam's tour into `seafoam_strike.run_strike(camp,log,dbg_dir)` on the LIVE bridge (all closures in main(), same shape as safari_strike), register it — either as a second `GYM_PREREQS`/`HM_PREREQ` recognition for Blaine (Cinnabar-reach prereq) OR a travel-gate strike that fires when head_to_gym no_paths at Route 20 toward Cinnabar — flag-gated, decision-verify + a look-ahead. NB the safari_strike extraction this shift is the working template (SafariStrike class + run_strike wrapper + anchor set + registry entry).
2. **THEN FLIP `POKEMON_SAFARI_STRIKE` default ON (campaign.py ~197) + commit** — once a clean e2e reaches Cinnabar (Safari→Surf→Seafoam→Cinnabar). The Safari wiring is BUILT + proven-to-Surf; only the Seafoam crossing gates the flip. (Consider flipping BOTH the safari + seafoam flags together once the full Fuchsia→Cinnabar chain is proven.) NB a real fresh badge-6 run has a GOOD evened bench (koga_done_kit) so the lopsided-grind oscillation seen on the THIN fuchsia_lapras_kit bench won't confound it; but verify on a good-bench fixture.
3. **After Cinnabar: the rest of the chain** — Secret Key/Mansion strike (recon_mansion.py exists, unpromoted, same extract-and-register pattern) → `GYM_PREREQS["Blaine"]` door → Blaine → Giovanni. Victory Road later needs Strength (already banked by the same Safari strike) + Surf.
4. **Carry-overs (lower pri):** PP-famine flee-spam in the lopsided grind stint (NS#10 2nd finding); grind-spot lever a (Route-23 nav-blocked, flag OFF); `POKEMON_BENCH_TO_MILESTONE` default-OFF.

**RE-RUN cmds:** Safari e2e (proves the wiring, from Fuchsia) = `POKEMON_SAFARI_STRIKE=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py fuchsia_lapras_kit.state 15` → grep `PROACTIVE HM-PREREQ|QUESTLINE STRIKE|got_surf|TEACH BRIDGE.*surf|questline cleared|MAP TRANSITION`. Decision check: `../.venv/Scripts/python.exe recon_safari_gate_check.py` (8/8). Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures).

---

## ⏳ NIGHT-SHIFT #10 IN FLIGHT (2026-07-11, night_shift.ps1 shift 10) — COMMITTED the shift-8/9 Saffron gatehouse fix (`f79be91`, decision-verified 6/6 + live no-regression clean); the 0.34 Koga look-ahead climbed to **BADGE 6 (Sabrina/Marsh) with stuck/loop/loss ALL 0 across ~107 min** — strongest marathon evidence yet. NEW FRONTIER surfaced = the **Blaine/Cinnabar Surf-gate approach** (she reached badge 6 with only CUT+FLASH taught — NO Surf/Strength). START HERE.

### ✅ WHAT SHIFT 10 BANKED:
1. **COMMITTED the Saffron gatehouse fix — `f79be91`** (the shift-8 uncommitted `_saffron_gate_dead_ends`). Decision-VERIFIED 6/6 (`recon_saffron_gate_check.py`) + LIVE no-regression verify (`ns9_saffron_verify.log`): **0 gatehouse (18,0) wedges, 0 (18,0) transitions, door-passthrough machinery (Diglett's Cave door work) clean, stuck/loop/blackout all 0.** Fail-open, empty post-Tea → zero behaviour change after the Tea. The Route-12/Saffron soft-wedge CLASS is now closed in door SELECTION (the last leak site).
2. **READ the Koga run (`/g/temp/longrun/ns4_koga.log`, 0.34, 120-min) — HUGELY positive.** badges 3→**6** (Boulder, Cascade, Thunder, Rainbow, **Soul/Koga**, **Marsh/Sabrina**) on ONE clean autonomous climb, **stuck=0 loop=0 loss=0 blackout=0** the WHOLE way. Party at badge 6: `venusaur L61, raticate L28, kadabra L28, pidgeotto L28, fearow L28, dugtrio L28`, dex 16 — a genuinely evened, part-evolved 6 (the LOPSIDED-BENCH grind IS working: bench L14→L28, prep target climbed 14→31, narrated in-character). At budget-close (~107 min) she was on Route 9 grinding the bench toward Blaine — **did not reach Blaine in budget** (still climbing, NOT walled).

### ✅ ALSO BANKED SHIFT 10 — the SAFARI HM ANCHOR FIX (`14a4b63`, safe correctness prerequisite):
The surf/strength/gold_teeth KB caps (`gamedata/frlg_gates.json`) all set `obtain.from="3,8"` = **Cinnabar (unreachable)**; the Safari Zone is entered from **Fuchsia=3,7**. `step.from_map` drives anchor-first routing (campaign.py:7726), so once a `Gate(missing="surf")` is synthesized the executor would route toward unreachable Cinnabar. Fixed all 3 → "3,7". Currently INERT (no surf gate fires yet), so it's a safe prerequisite for the wiring below.

### ⇒ SHIFT-10 FRONTIER = BUILD THE AUTONOMOUS BLAINE/SAFARI WIRING (fully diagnosed this shift — an Explore trace + code read gave the exact 3-gap design + file:lines; this is the pass-3 "solved-but-not-wired-into-the-autonomous-flow" gap for the badge 6→8 stretch):

**THE ROOT (trace-confirmed, 3 independent structural gaps — a fresh badge-6 Surf-less run HARD-WEDGES at the Fuchsia/Route-19 beach with no forward errand):**
- **(a) RECOGNITION: no `Gate(missing="surf")` is ever produced.** The Cinnabar billed road (`frlg_gates.json:617-680`) crosses billed SEA (Route 19/20) with NO "requires surf" precondition + NO exit_gate for the sea crossing, so `GateRecognizer.exit_gate` returns None. `_field_block` (campaign.py:10246) only offers `use_surf` when she ALREADY has Surf — there is NO "go acquire Surf" branch. And `_gym_prereq_gate` (campaign.py:8881, the Sabrina/Silph pattern) fires ONLY when `state["map"]==gym.city` + beat_gym returns stuck (campaign.py:10707,10732) — but she can NEVER reach Cinnabar without Surf, so it can't fire for Blaine. **The Sabrina at-the-door pattern does NOT transfer — Blaine needs PROACTIVE prereq recognition BEFORE the sea march.**
- **(b) EXECUTION: no Safari strike is registered.** `_questline_strike` (campaign.py:8758, registry at 8787-8800) has only Hideout/Tower/Snorlax/Silph. No `("flag","FLAG_GOT_HM03")` / `("cap","surf")` entry, no `safari_strike.py`. The general `talk_npc` layer CANNOT do the Safari (recon_safari.py:11 — the "classic Safari loop", Secret House UNREACHABLE on foot from the entrance pocket). So even if a gate fired, execution falls through and she loops/wedges in the Safari (this is WHY wiring recognition WITHOUT the strike is WORSE — they MUST land together).
- **(c) Strength (HM04) + Secret Key same/worse:** no `GYM_PREREQS["Blaine"]` (only Sabrina, campaign.py:646), no `secret_key` capability in the KB at all, no Cinnabar gym exit_gate. Victory Road later needs Surf+Strength too.

**THE EXECUTOR IS PROVEN, THE EXTRACTION IS THE CRUX.** `recon_safari.py` (808 lines) is a COMPLETE, pass-1/2-verified Safari strike ([[pokemon-safari-hms-surf-seafoam]] — HM03 Surf + HM04 Strength via Gold-Teeth→Warden, the "classic loop" nav all solved). **PROOF it worked: the pipeline fixtures exist in `states/workshop`** — `safari_hms_kit` (recon_safari's OUTPUT, HMs obtained), `fuchsia_gate`/`fuchsia_potions` (pre-Safari Fuchsia starts), `surf_ready_kit`/`surf_ready` (post-Surf). So the tour logic is NOT the risk — only the closure-lift + recognition wiring is. (recon_safari's loop only acts when `here==FUCHSIA`/a Safari map — it needs a Fuchsia-positioned start; a quick executor re-check: `SAFARI_STATE=fuchsia_gate recon_safari.py`.) BUT it's all nested CLOSURES inside `main()` (safari_battle/safari_bfs/walk_path_to/engage/step_warp/enter_to capture `b`/`camp`/`L`), booting its OWN bridge, and its loop only acts when `here==FUCHSIA`/a Safari map (it does NOT navigate to Fuchsia). So the build is: **extract ~600 lines of closures into `safari_strike.run_strike(camp, log, dbg_dir)`** operating on the LIVE campaign bridge (model on `silph_strike.py`/`hideout_strike.py`), returning `("got_surf",)`/`("got_strength",)`. Non-trivial + landmine-adjacent (campaign.py:8916/8952 explicitly warn a mis-timed Surf/Safari questline "poisons her context" — the recognition MUST be deliberate + data-driven, NOT a generic near-water heuristic).

**THE BUILD (land as ONE coherent unit, flag-gated `POKEMON_SAFARI_STRIKE` default OFF until live-verified — recognition+strike are useless/harmful apart):**
1. **Extract `safari_strike.py`** from recon_safari's tour (the crux). `run_strike(camp, log, dbg_dir)` reuses camp.b/camp.trav; anchors = {FUCHSIA} | Safari maps (group 1, num 63-71, campaign.py:2790); good_results `("got_surf",)` / also `("got_strength",)` if you do both legs.
2. **Register it** in `_questline_strike` (campaign.py:8787): `("flag","FLAG_GOT_HM03"): ("Safari Zone (HM03 Surf)", "...", _safari)` (+ FLAG_GOT_HM04 for Strength). Look up the numeric FLAG_GOT_HM03 id (field_moves / a flags KB).
3. **PROACTIVE recognition:** add `GYM_PREREQS["Blaine"] = (FLAG_GOT_HM03, "surf", human)` + a proactive prereq check in `_ensure_forward_questline` (campaign.py:9033, runs every tick) — if `next_gym==Blaine` and the prereq flag is unmet, `_open_questline(_gym_prereq_gate(Blaine))` so head_to_gym drives toward the Safari BEFORE marching into the uncrossable sea. Verify it does NOT misfire for Sabrina (her prereq is at-the-door; guard on the flag being a Safari-HM, or keep Sabrina's at-door path and only add Blaine proactively).
4. **DECISION-verify** a `recon_safari_gate_check.py` (model on `recon_saffron_gate_check.py`): next_gym=Blaine + HM03-unset ⇒ Gate(missing="surf"); HM03-set ⇒ None; Sabrina unaffected. **LIVE-verify:** a look-ahead from a badge-6 fixture (below) — she routes to Fuchsia, the strike fires, she gets Surf, then the sea road to Cinnabar opens (surf-aware travel already wired: `_surf_usable`/`walkable_or_surf`). Watch for context-poisoning (she must NOT narrate "blocked by water" at wrong times). Then flip the flag ON + commit.
5. **AFTER Surf lands:** the follow-on chain (Cinnabar sea-nav → Secret Key/Mansion strike → `GYM_PREREQS` Blaine door → Blaine → Giovanni) — recon_mansion.py exists (unpromoted), same extract-and-register pattern.

**THE BADGE-6 FIXTURE = READY: `states/workshop/koga_done_kit`** (promoted shift 10 from the Koga run's clean badge-6 bank — Venusaur L62 + evened bench raticate/dugtrio/kadabra/pidgeotto/fearow L28-29, dex 16, **NO Surf** — the ideal Blaine probe). If you need a fresher one, `recon_longrun.py surge_done_kit.state 110` re-climbs badge 3→6 clean (~100 min, 1248 real battles proven this shift).

### ⇒ SECOND FINDING (mid-game grind watchability — a next-shift careful fix): PP-FAMINE FLEE-SPAM in the LOPSIDED grind stint.
The Koga run's FINAL grind stint (tick #180, Route 10) degenerated: the participation grind mon went **PP-dry on damaging moves → "nothing damaging left" (39×) → tries to switch → "famine switch did not confirm" (the known in-battle-switch wedge, 13×) → Sleep-Powder-spam → ANTI-WEDGE FLEE (13×), all in ONE stint** (badges 3→6 were totally clean — this only emerged when the bench mon's PP ran out). Bounded (stint budget ends it, stuck=0, no livelock) but ~39 wasted no-progress battles = a real ugly-on-a-watch stretch. ROOT: the grind has no "fielded participation mon is PP-dry → heal PP at a Center or end the stint" logic. **SAFE fix (avoids the wedge-prone in-battle switch):** when the grind mon's damaging moves are all 0 PP, break the stint → heal at a Center (restores PP) → resume, or end the stint. Verify-gated (repro from a fresh grind stint; do NOT touch the in-battle switch blind — DON'T-FIX-BLIND, E4-livelock family).

### ⇒ Carry-overs (unchanged, lower pri): grind-spot lever a (Route-23 nav-blocked, flag OFF); Parlyz Heal buy no-op (live frame-grab); keeper cave-descend slowness. `POKEMON_BENCH_TO_MILESTONE` STAYS default-OFF.

**RE-RUN cmds:** Blaine probe = `POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py koga_done_kit.state 45`. recon_safari executor (from a Fuchsia-positioned fixture): `SAFARI_STATE=<fuchsia_fixture> ../.venv/Scripts/python.exe -u recon_safari.py`. Saffron decision re-check: `../.venv/Scripts/python.exe recon_saffron_gate_check.py` (6/6). Canonical Champion save UNTOUCHED (all look-aheads on scratch fixtures).

---

## ⏳ NIGHT-SHIFT #9 IN FLIGHT (2026-07-11, night_shift.ps1 shift 9) — inherited shift-8's uncommitted Saffron fix; the 0.25 bail025 A/B run REPRODUCED the Saffron gatehouse wedge LIVE (10× wedges + 8 stuck at Celadon), confirming the bug is real; ran the decision-check (6/6 PASS) + launched a fresh no-regression live-verify (fix in place). START HERE.

### ✅ WHAT SHIFT 9 IS DOING:
1. **THE SAFFRON FIX (shift-8's uncommitted campaign.py edit) — now CONTROL-CONFIRMED LIVE + decision-re-verified.** The 0.25 A/B run (`/g/temp/longrun2/ns6_bail025.log`, which does NOT have the fix) reached Celadon at ~42 min and **wedged at the Saffron south gatehouse (18,0) 10× + logged 8 `outcome=stuck`, stalling at badge 3 the whole run** — a live reproduction of exactly the wedge shift-8's `_saffron_gate_dead_ends` fix targets. Meanwhile the 0.34 Koga twin (no fix either) never hit it (variance, as predicted). `recon_saffron_gate_check.py` re-run this shift = **6/6 PASS** (control case proves the tie-break bug; pre-Tea drops the gatehouse so the UGP hut wins; post-Tea unchanged). campaign.py parses OK. The fix is fail-open + empty post-Tea (zero behaviour change after the Tea).
2. **NO-REGRESSION LIVE-VERIFY LAUNCHED** — a fresh `surge_done_kit` 35-min look-ahead WITH the fix, isolated stage (`TEMP=G:/temp3`), log `/g/temp/longrun/ns9_saffron_verify.log`. Purpose: confirm the fix doesn't break the Route-6 door-passthrough machinery (Flash errand exercises it early) and — if it reaches the Celadon approach — 0 gatehouse wedges. AT RESUME: grep `TRAVEL WEDGE.*18, 0|-> \(18, 0\)` (expect 0) + confirm she crosses Route 6 clean. **THEN COMMIT the Saffron fix.**
3. **KILLED the bail025 (0.25) A/B run** (was 42/45 min, stuck spinning on the Saffron wedge, served its purpose). A/B firmed: **0 real faints in BOTH arms** (the 0.25 safety property the shift-7 flip relied on HOLDS). The stall was the Saffron wedge (variance), NOT the bail-frac.

### ⏳ THE KOGA (0.34) LOOK-AHEAD STILL IN FLIGHT (`/g/temp/longrun/ns4_koga.log`, PIDs 17868+31068, 120-min budget) — READ ITS TAIL AT RESUME:
At shift-9 open it was at **~87 min, GYM 7 of 8 (heading Blaine/Cinnabar)** — badges 3→6 CONFIRMED (Erika/Rainbow→Koga/Soul→Sabrina/Marsh, + the Silph Giovanni fight done, saffron_free=True), health **stuck=0, loop=0, loss=0** across the WHOLE climb. Strongest marathon-bar evidence yet. AT RESUME: did it clear Blaine (7) + Giovanni (8)? `grep -oE "Volcano Badge|Earth Badge|EARNED|STATE IN: Cinnabar|Viridian Gym|badges=[78]" /g/temp/longrun/ns4_koga.log | tail`.

## ⏳ NIGHT-SHIFT #8 IN FLIGHT (2026-07-11, night_shift.ps1 shift 8) — the 0.34 Koga look-ahead CLIMBED CLEAN THROUGH FOUR GYM BOUNDARIES (badge 3→**6**: Erika→Koga→Sabrina, now heading Blaine) with stuck/loop/loss/blackout ALL 0; BUILT + decision-verified the Saffron-gatehouse door-passthrough wedge fix (frontier item 2).

### ✅ WHAT SHIFT 8 IS BANKING:
1. **THE 0.34 KOGA RUN (`/g/temp/longrun/ns4_koga.log`) IS A HUGE POSITIVE — the mid-game loop is proven across FOUR gym boundaries on ONE clean autonomous climb.** From badge-3 Vermilion (surge_done_kit) it went: BADGE 4 Erika (Rainbow) → BADGE 5 **Koga** (Soul — the Q2 question, ANSWERED YES) → BADGE 6 **Sabrina** (Marsh — required the Tea + Silph liberation, all autonomous) → now on Route 5/Cerulean heading to **Blaine (badge 7, Cinnabar)** at ~87 min of the 120-min budget. Party evolved+evened: `venusaur L60, raticate L28, kadabra L28, dugtrio L26, fearow L27, pidgeotto L25`, dex 16. Health across the WHOLE climb: **stuck=0, BATTLE-LOOP BREAKER=0, battle_loss=0, blacked out=0.** This is the strongest marathon-bar evidence yet — the ACE-DOWN GUARD + lopsided-bench + keeper machinery all hold long. AT RESUME: read the tail — did she reach/clear Blaine (7) + Giovanni (8)? `grep -oE "badges=[0-9]|Volcano|Earth|STATE IN: Cinnabar|Victory Road" /g/temp/longrun/ns4_koga.log | tail`.
2. **SAFFRON-GATEHOUSE DOOR-PASSTHROUGH WEDGE FIX (frontier item 2 from NS#7) — BUILT + decision-verified, LIVE-VERIFY PENDING.** ROOT (precise, from ns6_bail025): the billed Celadon Route-6 leg is `via:'pass'` — the INTENDED crossing is the single-door Underground Path hut ((19,13)→(1,32), under Saffron to Route 5). But `_door_passthrough`'s `_dest_rank` tie-break (`-len(dest_doors)`) prefers the **2-door Saffron SOUTH gatehouse** ((12,5),(13,5)→(18,0)) over the 1-door UGP hut → she walks into the gatehouse whose north exit into Saffron is guard-blocked pre-Tea → 8× wedge at (18,0) before self-recovery. NB the shift-7 proposed fix (fold into `_story_gate_avoid`) would NOT have worked — the wedge is in door SELECTION, and `_door_passthrough` never consults the routing avoid. FIX (campaign.py): `_SAFFRON_GATE_MAPS={(18,0),(19,0)}` (south + Route-7 west gates, learned world-graph ids) + `_saffron_gate_dead_ends()` (the set pre-Tea via FLAG_GOT_TEA 678, empty post-Tea, fail-OPEN) + skip those as `_door_passthrough` connector candidates. Post-Tea empty → zero behaviour change; the gates become normal roads. **Decision-VERIFIED 6/6** (`recon_saffron_gate_check.py` — control case proves the tie-break bug is real; pre-Tea the UGP hut wins; post-Tea unchanged). **LIVE-VERIFY PENDING** (emulators busy with the 2 A/B runs): once one frees, run a fresh `surge_done_kit` ~15-min look-ahead, grep `MAP TRANSITION.*-> \(18, 0\)|TRAVEL WEDGE.*18, 0` (expect 0) + confirm she crosses Route 6→Route 5→Celadon clean. Then commit. NB the wedge is variance-dependent (the 0.34 twin never hit it), so a fresh run may not reproduce it — the verify is NO-REGRESSION (crosses Route 6 cleanly) + the decision-check proves the fix engages when the off-road-steer path is taken.

### ⇒ SHIFT-8 FRONTIER (priority order):
1. **COMMIT the Saffron fix after live-verify** (above). Then the frontier is the CLIMB past badge 6.
2. **READ the Koga run's final outcome** — if it cleared Blaine (7) + Giovanni (8) → the loop is proven nearly to Victory Road; advance the frontier to the E4-prep grind (the L45→55 team-depth push, where the grind-spot lever a bites — still nav-blocked on Route 23). If it walled at Blaine/Giovanni → diagnose the tail (fresh blocker, rule 8c).
3. **B-arm A/B (`/g/temp/longrun2/ns6_bail025.log`, 0.25, 45min)** — the 0.34→0.25 flip is ALREADY committed (`3b7bc75`); this run just firms the excursion-frequency win. Read its tail for bails/excursions/faints to badge 4.
4. **Carry-overs (unchanged from NS#7):** Parlyz Heal buy no-op (needs live frame-grab); grind-spot lever a (Route-23 nav-blocked); deep WHITE-BOX SWITCH wedge (DON'T-FIX-BLIND); keeper cave-descend slowness.

---

## ✅ NIGHT-SHIFT #7 DONE (2026-07-11, night_shift.ps1 shift 7) — LANDED the dominant watchability lever: flipped `GRIND_ACE_BAIL_FRAC` default **0.34→0.25**, A/B-verified on the two in-flight look-aheads. ONE commit (`3b7bc75`), canonical UNTOUCHED.

### ✅ WHAT SHIFT 7 BANKED — `3b7bc75` (mode-side, one default-value change, canonical UNTOUCHED):
The candidate-(b) headline from NS#5/#6 — the heal-excursion FREQUENCY was the biggest watch drag on grind
stretches (each ACE-DOWN GUARD fire = a cross-city Center round-trip). Lowering the bail band 0.34→0.25 fires
the guard LATER = fewer bails/excursions. **A/B-verified live** on the two shift-6 in-flight look-aheads
(same surge_done_kit fixture + Koga-baseline flags, differing ONLY in ACE_BAIL_FRAC), common ~800s window
normalized by grind-battle (`GRIND SWITCH`) count:
- **0.34 (ns4_koga):** 6 ACE bails / 8 excursions per 126 grind-switches; guard fires at ~30-33%.
- **0.25 (ns6_bail025):** 1 ACE bail / 3 excursions per 77 grind-switches; guard fires at ~19-22%.
- → **~3.7× fewer bails per grind-battle**, fewer excursions, and **ZERO ace faints in BOTH arms**. The 0.25
  ace floor is a safe ~19-22% (worst single observed 19% — an L41-48 ace vs L15-20 grind wilds does ~6%/hit,
  so a 25% band can't reach 0 in one inter-check tick). The no-faint safety property HOLDS; the watch is calmer.
- **THREE-STATE:** COMPILES + WIRED (`GRIND_ACE_BAIL_FRAC` @ campaign.py:236, consumed by the ACE-DOWN GUARD
  @6065) + **live-A/B-VERIFIED** (twin look-aheads above). Env-overridable (`POKEMON_GRIND_ACE_BAIL_FRAC`);
  one-char revert if ever needed. NB the 0.25 floor (~19-22%) is LOWER than 0.34's (~32%) — still comfortably
  safe vs grind wilds, but a shift that pushes the ace into HIGHER-level grass should re-audit the margin.

### ⏳ TWO look-aheads still in flight at shift-7 close (READ TAILS AT RESUME) — both healthy, canonical UNTOUCHED:
- **B-arm `/g/temp/longrun2/ns6_bail025.log` (0.25, 45-min budget)** — at shift close ~20 min in, 5 bails / 7
  excursions / **0 faints**, on Route 9 heading to Celadon (Rock Tunnel path, correct). Read its tail: did it
  reach badge 4? final bail/excursion count (more samples firm the safety claim; still expect 0 faints).
- **Q2 `/g/temp/longrun/ns4_koga.log` (0.34, 120-min budget)** — at shift close ~68 min in, at **Lavender Town →
  Pokémon Tower** (advancing the Rocket-Hideout→Silph-Scope→Flute→Snorlax→Fuchsia→Koga chain, badge 3, questline
  on-track). Q1 (livelock-dead over a long climb) = CONFIRMED. Q2 (reach + CLEAR Koga) still climbing; read the
  tail — did she reach/win Koga? `grep -oE "Soul Badge|'Rainbow'|KOGA|STATE IN: Fuchsia" ns4_koga.log | tail`;
  health `grep -c 'outcome=stuck\|BATTLE-LOOP BREAKER\|blacked out' ns4_koga.log`.

### ⇒ SHIFT-7 FRONTIER (priority order):
1. **READ the two in-flight outcomes above.** If the B-arm reached badge 4 with fewer excursions + 0 faints →
   the 0.25 flip is fully proven across a gym boundary (bank that note). If the Koga run CLEARED Koga → the loop
   is proven across TWO gym boundaries; advance the frontier to badge 6 (Sabrina). If a NEW blocker in the
   Silph/Tower/Flute chain → diagnose the tail (rule 8c), fix that specific blocker, re-run.
2. **WATCHABILITY NIT — the SAFFRON-GATEHOUSE (18,0) transient wedge (diagnosed this shift, verify-gated fix-hook).**
   The B-arm burned ~2-3 min wall-bumping at the Saffron south gatehouse (18,0) before self-recovering. EXACT trace
   (ns6_bail025.log): head_to_gym off-road anchor-steer (`_road_step` ~campaign.py:10086) picked "Route 6 (west)"
   as the Celadon road anchor (line 14586) → from Route 6 (3,24) she walked door (12,5) → WARPED into the Saffron
   south gatehouse (18,0) (line 14693) → the north exit into Saffron is GUARD-BLOCKED (pre-Tea) → wedge ×8 →
   recover → re-enter → until she abandons + re-routes correctly via Route 9/Rock-Tunnel/Lavender/Underground-Path/
   Route-7 (the real pre-Tea path to Celadon; Route 6→Saffron is a dead-end). The 0.34 twin did NOT hit it (routing
   variance from a divergent learned world-graph). ROOT = same bug CLASS as NS#42 (Route-12) / NS#43 (growlithe):
   the head_to_gym avoid set has SAFFRON city (3,10) (the NS#42 BYPASS ~campaign.py:10788-10790) but NOT the Saffron
   **gatehouse** maps (18,0 = south gate), which lead only to blocked Saffron. FIX (verify-gated, do NOT ship blind):
   add the Saffron gatehouse map(s) to the gate-avoid (fold into `_story_gate_avoid` or the head_to_gym `_avoid`)
   gated on `NOT FLAG_GOT_TEA (678)`, mirroring `_keeper_hard_gate_avoid`'s Saffron treatment (~campaign.py:6983).
   ⚠️ Verify a Celadon-approach look-ahead still reaches Celadon (world.route always allows the SRC map, so a resume
   ON the gatehouse still escapes) AND that post-Tea Saffron-targeted routing is unaffected. Self-recovering +
   variance-dependent, so LOW priority vs the climb — but a real ~2-3 min ugly-wall-bump on a live watch.
3. **MINOR BUG (low-pri, unverifiable while emulator busy) — Parlyz Heal buy no-op.** `MART: buy-verify FAILED for
   Parlyz Heal (price=250, x0->x0)` at Vermilion (recurs 3-6×/run) — the buy actuation no-op'd; code aborts LOUD
   (no crash); later Marts cleanly "not sold here — skipping". NB price read as 250 vs Parlyz Heal's real 200 →
   likely a WRONG-MENU-ROW read (buying/checking the wrong item). Needs a live frame-grab of the open buy menu
   (rule-15 #4) — do NOT fix blind. Small real cost (paralysis chip during travel).
4. **Lever (a) grind-spot picker — STILL nav-BLOCKED, unchanged from NS#5/#6.** Decision-correct (28/28) but FRLG's
   only adequate high-level GRASS for the E4-prep L45→55 target is Route 23, whose south grass is split-map/Surf-
   gated (NS#13 wedge). Flipping `POKEMON_GRIND_SPOT_LEVELAWARE` ON gains nothing until the Route-23 south-pocket is
   in the world graph (deep nav work, verify-gated). Keep flag OFF; the real unblock = Route-23 south-pocket nav.
5. **Carry-overs (unchanged, lower pri):** deep WHITE-BOX SWITCH wedge (E4-livelock family, DON'T-FIX-BLIND, needs
   rule-15 live frame-grabs); "RIVAL beat vs Gary" strat-memory mislabel (cosmetic, needs a decision trace);
   keeper cave-descend slowness (watchability). `POKEMON_BENCH_TO_MILESTONE` STAYS default-OFF.

**RE-RUN cmds:** 0.25 re-verify = default now, just `POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1
LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 45` (compare bails/exc/
faints vs an env `POKEMON_GRIND_ACE_BAIL_FRAC=0.34` arm). Decision checks: `recon_grind_spot_check.py` (28/28),
`recon_bench_milestone_check.py` (12/12), `recon_lopsided_grind_check.py` (ALL PASS). Canonical Champion save
UNTOUCHED (all look-aheads on scratch surge_done_kit).

---

## ✅ NIGHT-SHIFT #6 (2026-07-11, shift 6) — landed the DOMINANT watchability lever (candidate b, ACE_BAIL_FRAC 0.34→0.25); the A/B ran into shift 7 and COMMITTED there (`3b7bc75`). (kept for reference)

### TWO look-aheads RUNNING IN PARALLEL (isolated STAGE dirs via TEMP override — SINGLE-RUN LAW respected per-SCRATCH):
- **A/B (B-arm) — `/g/temp/longrun2/ns6_bail025.log`** — surge_done_kit 45-min, `POKEMON_GRIND_ACE_BAIL_FRAC=0.25`
  + the Koga-baseline flags (ACE_BAIL/BENCH_TO_MILESTONE/PREP_DRY_RESET/KEEPER_STATIC_ROUTE/KEEPER_ROUTER +
  LONGRUN_BATTLE_LOG). **This is the shift's headline** — the NS#5 data named heal-excursion FREQUENCY the biggest
  watch drag. **BASELINE (A-arm, 0.34) from the Koga run: badge 4 reached at ~30 min with 15 heal-excursions + 13
  ACE bails, 0 real faints.** SUCCESS for 0.25 = FEWER excursions/bails to badge 4 AND the ace NEVER faints (safety
  audited: guard fires at top of each grind tick, one wild between checks; an L41-48 ace vs L15-20 wilds can't drop
  >25%→0 in one hit → 0.25 keeps the no-faint property). If clean → set `GRIND_ACE_BAIL_FRAC` default "0.34"→"0.25"
  (campaign.py:231) + commit. Compare with: `grep -c 'HEAL-EXCURSION: lead' <log>` + `grep -c 'ACE dinged' <log>`
  up to the first `badges=4` line; confirm 0 real faints (`grep -c 'blacked out'`).
- **Q2 — `/g/temp/longrun/ns4_koga.log`** — the 120-min Koga run (from NS#5, launched ~14:09). **Q1 = CONFIRMED**
  (livelock dead over a long climb — ~46 clean min, stuck/loop/loss=0). Q2 (reach+clear Koga) still climbing
  (badges=4, heading Silph/Tower/Flute/Snorlax/Fuchsia — may not fit budget). Read tail at resume.

### 🔁 LEVER (a) REFRAMED (item was "verify+flip grind-spot picker") — it is BLOCKED ON ROUTE-23 NAV, not merely confounded:
The grind-spot picker is decision-correct (28/28 `recon_grind_spot_check.py`) but FRLG has essentially ONE adequate
high-level GRASS spot for the E4-prep L45→55 target: **Route 23 (L34-49)** — every other endgame option is a
cave/water/gated (terrain-filtered OUT by `_better_grind_spot`). And Route 23's south grass is split-map / Surf-gated
(the NS#13 wedge: GRIND_MAP=3,42 boots the grassless north edge, can't path south). `_better_grind_spot` only checks
the FIRST hop (`world.next_hop`), so it WOULD propose Route 23, she'd route to its north edge, find no grass, mark it
grind-dead, and re-pick → no net benefit over baseline (which grinds poor grass). **So flipping lever (a) ON gains
nothing until the Route-23 south-pocket is reachable.** The real unblock = fix the Route-23 south grass world-graph
gap (water-crossing within a split map; the late team HAS Surf) — deep nav work, verify-gated. Until then: **keep
`POKEMON_GRIND_SPOT_LEVELAWARE` flag-OFF** and treat "Route-23 south-pocket nav" as the true blocker. Mid-game the
lever correctly does NOTHING (adequate spots L12-18 are flute/bike-gated pre-endgame → filtered → grinds poor grass,
anti-freeze holds). Decision-safety is proven; the LIVE value is nav-blocked. Lower priority than the 0.25 A/B.

### MINOR BUG surfaced (low-pri, note): `MART: buy-verify FAILED for Parlyz Heal (price=250, x0->x0)` in the Koga run
— she tried to stock Parlyz Heal and the buy silently no-op'd (x0→x0). Economy/actuation glitch, not blocking; worth
a look if it recurs (paralysis was "hurting" her per the shop note, so the failed buy has a small real cost).

## ✅ NIGHT-SHIFT #5 DONE (2026-07-11, night_shift.ps1 shift 5) — BUILT + decision-verified the grind-efficiency lever (a): a level-aware GRIND-SPOT picker (flag-OFF, park-safe). TWO commits (5094c0b KB+predicate, df469f1 picker wiring). Q1 (livelock-dead over a long climb) RE-CONFIRMED clean on an in-flight 120-min Koga look-ahead. Canonical UNTOUCHED.

### ✅ WHAT SHIFT 5 BANKED (2 commits, mode-side, flag `POKEMON_GRIND_SPOT_LEVELAWARE` default OFF, canonical UNTOUCHED):
The PASS-3 grind-efficiency lever (a) from the NS#4/#9 frontier — a LEVEL-AWARE grind-spot picker that
targets the documented **NS#1/#14 E4-prep stall** (an L45→55 mon grinding on L8-19 grass gains ~0 XP, so
`grind()` spins its whole budget for ~0 levels). Built the game-knowledge layer + wired the picker, all
DECISION-verified emulator-free (26/26 `recon_grind_spot_check.py`), ZERO behavior change until flipped:
- **`5094c0b` — KB + predicate.** `gamedata/frlg_grind_spots.json` (wild-level bands per grind area,
  Bulbapedia FRLG, rule-14 portable) + `_grind_wild_band()` (cached fail-open reader) + `_grind_inadequate(map,
  target)` (wild_max more than `GRIND_POOR_GAP`=18 below target ⇒ ~0 XP). Route 18 (L24-29) flags INADEQUATE
  vs E4 target 55 but NOT vs a mid-game target 22; Route 23 / Victory Road / Cerulean Cave high grass correctly
  NOT flagged; unknown map / no target fail-open.
- **`df469f1` — picker wiring (flag-OFF).** `grind()` stands down from a grind-inadequate map via the existing
  `no_safe_grass` re-pick flow (marks `_grind_inadequate_set`); `_grass_target`/`_grass_via_graph` exclude that
  set (unioned with `_grind_dead`). **PARK-SAFETY GATE `_better_grind_spot`:** the stand-down ONLY fires when a
  reachable ADEQUATE spot actually exists (ungated, rideable, has level data) — else None, so a map that is the
  ONLY reachable grass is NEVER abandoned (grinds poor grass rather than freeze; anti-park/anti-freeze invariant,
  decision-proven). Reuses the same world-graph reachability the base `_grass_target` trusts.
- **THREE-STATE:** COMPILES + WIRED + decision-VERIFIED (26/26). **LIVE-verify PENDING** — the flag is OFF; do
  NOT flip without the look-ahead below (park-road discipline: NS#1/#9 grind-cadence scars).

### ⏳ THE 120-MIN KOGA LOOK-AHEAD IS IN FLIGHT (`G:/temp/longrun/ns4_koga.log`, launched ~14:09, surge_done_kit, flags ACE_BAIL/BENCH_TO_MILESTONE/PREP_DRY_RESET/KEEPER_STATIC_ROUTE/KEEPER_ROUTER + LONGRUN_BATTLE_LOG all =1). **READ ITS TAIL AT RESUME.**
- **Q1 (livelock dead over a LONG climb) = CONFIRMED POSITIVE ACROSS A FULL GYM BOUNDARY.** By ~34 min she
  climbed badge-3 → Flash → Rock Tunnel → **WON BADGE 4 / ERIKA (loss=0 — first-try this time, vs shift-4's
  loss-then-retry: the LOPSIDED-BENCH grind + evolutions gave a strong enough bench)** → Route 7 toward the
  Rocket Hideout. Whole stretch: **`outcome=stuck`=0, `BATTLE-LOOP BREAKER`=0, `battle_loss/blacked out`=0**,
  ACE-DOWN GUARD firing cleanly (13×), wedges small+self-recovered (34). 34 min of continuous clean climbing
  through a gym boundary = strong marathon-bar evidence the ACE-DOWN GUARD (2defbcd/d69ed78) holds long.
- **Q2 (reach + CLEAR Koga) = still in flight, FAR off** (she was at the dex-10 Flash gate / Route 9, badges=3,
  ~23 min in — Rock Tunnel → badge 4 → Silph → Tower → Flute → Snorlax → Route 12/13 → Fuchsia → Koga all ahead;
  may not fit the 120-min budget). **AT RESUME grep:** `elapsed = grep -oE '\[ *[0-9]+\.[0-9]+s\]' ns4_koga.log | tail -1`;
  progress `grep -oE "Soul Badge|badges=[0-9]|Rainbow|EARNED|Silph|Poké Flute|Snorlax|KOGA|STATE IN: Fuchsia" ns4_koga.log | tail`;
  health `printf 'stuck:%s loop:%s loss:%s\n' "$(grep -c outcome=stuck ns4_koga.log)" "$(grep -c 'BATTLE-LOOP BREAKER' ns4_koga.log)" "$(grep -c 'battle_loss\|blacked out' ns4_koga.log)"`.
- **✅ FLASH GATE PASSED (a watch-item that RESOLVED — don't chase it):** by ~28 min the roam ctx read
  **"Pokédex: 10 caught" + "Field moves ready: CUT, FLASH"** and she was **crossing Rock Tunnel** (map 1,81↔1,82)
  toward Celadon/Erika (badge 4), party Venusaur L41 + an evolving bench (Kadabra). The earlier "dex 7" reads were
  a STALE ctx snapshot from an earlier window, not a stall — she reached dex 10 (catches + the Abra→Kadabra
  evolution bump the caught-dex) and taught Flash cleanly on the full-party surge_done_kit run. NO PCBOX change
  needed for this gate. (One minor watchability note: ~10 Rock-Tunnel floor transitions 1F↔B1F — normal maze
  traversal, stuck=0, but WATCH the tail for whether it's efficient or a mild dark-cave bounce.)

### ⇒ SHIFT-5 FRONTIER (exact next actions, priority order):
1. **VERIFY + FLIP the grind-spot lever (the shift's build, live-verify PENDING).** Run a `giovanni_kit_g`
   look-ahead (the E4-prep stretch where lever a bites — bench L37-40 needing L55) with the flag ON:
   `POKEMON_GRIND_SPOT_LEVELAWARE=1 LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1
   ../.venv/Scripts/python.exe -u recon_longrun.py giovanni_kit_g.state 20 > /g/temp/longrun/ns5_gsverify.log 2>&1 &`
   → grep `GRIND: .* give near-0 XP toward|standing down to route to a reachable higher-level spot|TRAVEL WEDGE|floor crossed`.
   SUCCESS = she recognizes the low-level grass as inadequate, re-routes to a reachable higher-level spot with a
   **BOUNDED, watchable detour (no park/treadmill, no over-backtrack)**, and the bench levels FASTER than baseline.
   If clean → set `GRIND_SPOT_LEVELAWARE` default "1" (one char, campaign.py ~236) + commit. If it parks/over-detours
   → tune (raise `GRIND_POOR_GAP`, or cap the detour distance) or leave flag OFF + document. Decision re-check:
   `../.venv/Scripts/python.exe recon_grind_spot_check.py` (28/28). giovanni_kit_g party = Venusaur L68 (ace) +
   Lapras L37 + Kadabra L39 + L8-14 chaff (min_level floor won't field chaff), near Viridian/Route 22. **⚠️ KNOWN
   CAVEAT the verify must watch:** near Viridian the only adequate GRASS spot is **Route 23** (Route 22 = L2-8
   inadequate; Victory Road is a CAVE, now terrain-filtered OUT by `_better_grind_spot`). But Route 23 is a
   split-map whose south grass parts need Surf/Waterfall and **WEDGED before** (NS#13: GRIND_MAP=3,42 boots north
   edge, can't path south to grass). So the picker may correctly recognize Route-22 inadequate + route to Route 23
   → and then hit the pre-existing Route-23 split-map wedge (NOT the picker's fault, but it confounds the verify).
   If so: the picker LOGIC is right (skip low grass); the blocker is Route-23 nav (separate). Consider a DIFFERENT
   verify fixture with a clean adequate grass spot reachable, OR add the Route-23 south-pocket to the world graph.
   Add KB spots if she reaches an unmapped grind map.
2. **CANDIDATE (b) — NOW DATA-BACKED AS THE DOMINANT WATCHABILITY LEVER (zero code, pure A/B — do this FIRST).**
   The NS#5 Koga run quantified the cost: **38 heal-excursions + 104 heal calls + 17 ACE-DOWN bails in ~37 clean
   minutes** (~1 cross-city Center round-trip PER MINUTE) — the heal-excursion FREQUENCY, not grind-spot level, is
   the biggest watch-quality drag. `POKEMON_GRIND_ACE_BAIL_FRAC` is already env-configurable (default 0.34). A/B
   `=0.25` on a surge_done_kit climb: the guard fires LATER = fewer excursions. Confirm the ace never actually
   FAINTS (an L41-48 ace vs L15-20 wilds can't drop >25%→0 in one turn, so 0.25 keeps the safety property). If
   fewer excursions + no faint → set default 0.25 + commit. `POKEMON_GRIND_ACE_BAIL_FRAC=0.25
   ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 45`. **DEEPER lever (higher risk, verify):**
   the 38 excursions come from the ace SOAKING chip while protecting the bench in participation grinds → consider
   `SOLO_WEAK_GRIND` (let a bench mon strong enough to survive weak grass take the KILL XP directly, so the ace
   isn't exposed = fewer heals) — default OFF, wedge-prone in-battle switch, audit carefully.
3. **READ the Koga run result (Q2)** (Flash gate already PASSED — see above). If she cleared Koga → the loop is
   proven across TWO gym boundaries; bank + advance the frontier to badge 6 (Sabrina). If she reached badge 4-5
   but ran out of budget → note how far + whether the L20-22 evolved bench (Mr.Mime/Kadabra Psychic + Diglett→
   Dugtrio Ground) can answer Koga's poison. If a NEW blocker in the Rock-Tunnel/Silph/Tower/Flute chain →
   diagnose the tail (it's the FIRST fresh blocker; fix that specific one per rule 8c, then re-run).
4. **Carry-overs (unchanged, lower pri):** the deep WHITE-BOX SWITCH wedge (E4-livelock family, DON'T-FIX-BLIND,
   needs rule-15 live frame-grabs); "RIVAL beat vs Gary" strat-memory mislabel (cosmetic — NO live repro in the
   Koga log, so it needs a decision trace before touching; rule 7); keeper cave-descend slowness (watchability).
   **`POKEMON_BENCH_TO_MILESTONE` STAYS default-OFF** (NS#4 decision holds — needs a multi-gym Koga-leveled proof).

**RE-RUN cmds:** grind-spot verify (item 1) above; Q1 re-verify `POKEMON_ACE_BAIL=1 POKEMON_BENCH_TO_MILESTONE=1
POKEMON_PREP_DRY_RESET=1 LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1
../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 45`. Decision checks: `recon_grind_spot_check.py`
(26/26), `recon_bench_milestone_check.py` (12/12), `recon_lopsided_grind_check.py` (ALL PASS). Canonical Champion save
UNTOUCHED (all look-aheads on scratch surge_done_kit / giovanni_kit_g).

---

## ✅ NIGHT-SHIFT #4 DONE (2026-07-11, night_shift.ps1) — VERIFIED the ACE-DOWN GUARD ship-work e2e: Route-11 livelock DEAD + the full loss→grind→retry→**BADGE 4** loop proven on a fresh climb. No code change (verify-gated area; committed fix now ship-proven across a gym boundary). (superseded as START HERE by NS#5 above)

### ✅ WHAT SHIFT 4 CONFIRMED (fresh surge_done_kit 45-min look-ahead, `G:/temp/longrun/ns11_acebail2.log`; flags `POKEMON_ACE_BAIL=1 POKEMON_BENCH_TO_MILESTONE=1 POKEMON_PREP_DRY_RESET=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1`):
- **QUESTION 1 = CONFIRMED — the Route-11 bench-grind LIVELOCK IS DEAD.** Across the WHOLE climb: `BATTLE-LOOP BREAKER=0`, `outcome=stuck=0`; the ACE-DOWN GUARD (`GRIND: ACE dinged`) fired **14× cleanly** (one-heal-per-tick, no thrash), each returning `ace_healed`→restore→re-grind. The committed fix (`2defbcd`, `d69ed78`) is now **SHIP-PROVEN across a gym boundary.**
- **THE FULL loss→grind→retry→win LOOP PROVEN e2e → BADGE 4 EARNED.** Fresh badge-3 Vermilion climb: caught diglett+oddish (party→6, dex→15), traded Abra→**Mr.Mime**, crossed to Celadon → **attempted Erika, LOST** to the exeggcute-lead gym trainer → **diagnosed team-depth IN-CHARACTER** ("*mr.mime/spearow/rattata/diglett/oddish are too weak... I'm going to level the weak ones to ~L25, field THEM not my ace*") → ground the whole bench **L14-18 → L20-22 with THREE evolutions** (Spearow→**Fearow**, Rattata→**Raticate**, Oddish→**Gloom**) → retried Erika → **WON → *RAINBOW BADGE* (badge 4)** → resumed grinding toward Koga (badge 5). Final party @ budget: `venusaur L48, gloom L22, mr.mime L22, fearow L21, raticate L20, diglett L20`, dex 15 — a genuinely evened, part-evolved squad that CLEARED a gym it had just lost. Exactly the team-depth behavior the mission wants.
- **On near-DEFAULTS:** the heavy lifting was the reactive loss-grind (BASE behavior) + ACE_BAIL (default-ON), so a fresh GO on defaults reproduces the loop. Decision checks green (`recon_bench_milestone_check` 12/12, `recon_lopsided_grind_check` ALL PASS). The Erika COVERAGE-TEACH taught Cut over a redundant Normal move (Razor Leaf + Sleep Powder BOTH retained — no signature clobber; verified).

### ⇒ SHIFT-4 FRONTIER (priority order — the fix is proven; the OPEN levers):
1. **GRIND-EFFICIENCY / WATCHABILITY = the binding next lever (well-scoped).** The bench-grind WORKS but is SLOW+circly on a watch: the L14→L20-22 climb ate ~800 sim-s, punctuated by **~14 ACE-BAIL heal-excursions** (the ace is the participation protector; it accrues chip over ~a dozen wilds, dips ≤34%, walks to a Center to heal, returns). Compounding: Route-7 grass is only ~L19 (modest XP), and she grinds ALL 5 bench mons to the gym target. This is the NS#9 grind-spot-adequacy lever. Candidate fixes (ALL verify-gated — do NOT ship blind, NS#1/#9 park-road warnings): **(a)** level-appropriate grass — a grind-spot table in `gamedata` (route→wild-level range) + a level-aware grass picker in `_grass_target` (data-side/nav, NOT the wedge-prone battle switch → LOWEST risk, best first target); **(b)** lower `POKEMON_GRIND_ACE_BAIL_FRAC` 0.34→0.25 so the guard fires later = fewer heal-excursions (defaults-safe; pre-authorized IF thrashing — it's NOT per-tick thrashing, but the excursion COUNT is the cost, so A/B it); **(c)** `SOLO_WEAK_GRIND` (give the bench the KILL XP not just participation — ~2× rate; wedge-prone switch, default-OFF, audit carefully). VERIFY any with a FRESH multi-gym look-ahead confirming no park/treadmill + a faster grind, then flag/commit.
2. **QUESTION 2 (reach Koga with a LEVELED bench + clear it) — ⏳ 120-MIN RUN IN FLIGHT (READ ITS TAIL AT RESUME).** The badge-4 → Silph Scope → Pokémon Tower → Poké Flute → Snorlax → Route 12/13 → Fuchsia chain is FAR past a 45-min window, so at shift-4 close I launched a **120-min surge_done_kit look-ahead → `G:/temp/longrun/ns4_koga.log`** (launched ~14:08, flags ACE_BAIL/BENCH_TO_MILESTONE/PREP_DRY_RESET/KEEPER_STATIC_ROUTE/KEEPER_ROUTER + LONGRUN_BATTLE_LOG all =1). **AT RESUME grep:** `Soul Badge|badges=5|LOPSIDED-DBG|PREP RE-ARM|BATTLE-LOOP BREAKER|outcome=stuck|OUTCOME:|party=\[.venu` + elapsed `grep -oE '\[ *[0-9]+\.[0-9]+s\]' ns4_koga.log | tail -1`. Questions it answers: (a) does the livelock stay dead over a MUCH longer climb (marathon-bar stress test)? (b) does she reach Koga, and does the L20-22 evolved bench (+leveling) CLEAR it (poison — Mr.Mime Psychic + Diglett→Dugtrio Ground are the answers)? (c) any NEW blocker in the Silph/Flute/Snorlax chain? If it wedges/stalls → diagnose the tail (fresh blocker); if it clears Koga → the loop is proven across TWO gym boundaries. RE-RUN (fresh log): same cmd → `/g/temp/longrun/ns4_koga2.log`. Alternatively boot a nearer fixture (a fresh badge-4 kit w/ FLASH+Flute done) for a shorter window.
3. **FLAG DECISION — `POKEMON_BENCH_TO_MILESTONE` STAYS default-OFF.** It was ON this run and fired only 1× (harmless, no park) — BUT the badge-4 clear came from the reactive loss-grind, NOT bench-to-ms, and the frontier gates the flip on reaching **Koga leveled + past Fuchsia** (a MULTI-GYM proof per NS#1) which isn't in. Don't flip on one gym boundary. `PREP_DRY_RESET` + `ACE_BAIL` remain default-ON (correct; verified live).
4. **Carry-overs (unchanged, lower pri):** the deep WHITE-BOX SWITCH wedge is the ultimate root but flagged **DON'T-FIX-BLIND** (E4-livelock family, needs rule-15 live frame-grabs) — the ACE-DOWN GUARD is the tractable upstream fix (keep the ace ALIVE so grass stays winnable, the wedge never triggers); "RIVAL beat vs Gary" strat-memory mislabel (cosmetic); keeper cave-descend slowness (watchability).

**RE-RUN cmd (Q1 re-verify, FRESH log):** `POKEMON_ACE_BAIL=1 POKEMON_BENCH_TO_MILESTONE=1 POKEMON_PREP_DRY_RESET=1 LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 45 > /g/temp/longrun/nsX.log 2>&1 &`. Decision checks: `POKEMON_BENCH_TO_MILESTONE=1 ../.venv/Scripts/python.exe -u recon_bench_milestone_check.py` (12/12) + `../.venv/Scripts/python.exe -u recon_lopsided_grind_check.py` (ALL PASS). Canonical Champion save UNTOUCHED (scratch surge_done_kit only).

---

## ✅ NIGHT-SHIFT #9 DONE (2026-07-11) — VERIFIED the committed lopsided fix (36b4998) across a FULL GYM BOUNDARY on a fresh surge_done_kit climb (badge-3 → BADGE 4 / Erika → into the Rocket Hideout for the Silph Scope, zero park, zero livelock) + built-then-REJECTED a bigger-bite lever by live evidence (reverted). Canonical UNTOUCHED. START HERE.

### ✅ WHAT SHIFT 9 CONFIRMED (the ns8_erika 45-min baseline look-ahead, +6 bite, my edit NOT in that process):
The committed fix is **proven across a gym boundary on a FRESH climb** — the strongest evidence yet:
- badge-3 Vermilion → LOPSIDED-BENCH fired ONCE (bench evened L8→**L14**, `floor crossed L14`, ace restored, **road resumed, NO park**) → keeper Diglett caught (party→5) → **caught to dex 10 → FLASH ERRAND → HM05 from the aide** → crossed **Rock Tunnel** → **traded Abra→Mr. Mime** (the Route-2 in-game trade — a PSYCHIC, a real Koga/Sabrina answer, so "Abra never evolves to Kadabra" is resolved *differently* by natural play) → Celadon → **ERIKA → *** RAINBOW BADGE *** (badge 4)**. ~15 travel wedges whole run, all small/self-recovered, ZERO livelocks.
- **The lopsided stint RE-FIRES at badge 4 exactly as designed** — `LOPSIDED-DBG floor=14 lead=43 ms=45 done=[31]` → fires (milestone rose to Koga's L45, badge-3 milestone marked done) → grinds bench +6 to **L20** with a clean in-character rationale ("Team's under-levelled — grinding mr. mime/rattata/spearow/pidgey up to ~L20, fielding them not my ace, then on to Fuchsia"). The once-per-badge re-arm works across gyms.
- **FRONTIER-#2 GAP CONFIRMED LIVE:** +6 lands the bench at **L20 vs Koga's L45 milestone — still 25 under**. Party at badge 4: `venusaur L43, mr. mime L19→20, rattata L14→20, spearow L14→20, diglett L22, pidgey L15→20`. Real Koga coverage is GOOD (Mr. Mime=Psychic 2×, Diglett→Dugtrio=Ground 2× vs poison) but underleveled.

### 🔬 THE BIGGER-BITE LEVER — BUILT, then REJECTED BY LIVE EVIDENCE + REVERTED (the shift's real decision):
I built a `+12` lopsided bite (`POKEMON_LOPSIDED_BITE`) then **reverted it (`git checkout campaign.py`)** — the ns8_erika baseline proved it's the WRONG lever. What the +6 baseline actually did at badge 4:
- **+6 already reaches the target AND crosses evolutions:** the badge-4 lopsided stint ground the floor to **L20 and EVOLVED Spearow→Fearow (L20) + Rattata→Raticate (L20)** — bench went `[Venu45, Mr.Mime21, Fearow20, Pidgey20, Diglett22, Raticate20]`, a genuinely evened, part-evolved squad. The "bigger bite to cross evolutions" case is MOOT: +6→L20 already evolves Spearow/Rattata, and Abra→**Mr. Mime** (Psychic) came free via the Route-2 in-game trade.
- **The grind is SLOW on Celadon's low-level grass** — the +6 badge-4 stint ate ~7+ min of wall-clock and RE-FIRED 2× (a budget-timeout `'ok'` doesn't set `_lopsided_grind_done`, so it grinds→marches-1-tick→grinds until the floor finally crosses L20 on the 2nd stint; **bounded, NOT a park**, but slow). A `+12` target would ~DOUBLE this AND make the per-stint crossing even less likely within one budget window → MORE re-fires → a real park risk. **So bigger-bite = a watchability regression + park risk. Rejected.**
- **THE REAL NEXT LEVER is GRIND-SPOT ADEQUACY, not a bigger target:** she grinds L14-20 mons on ~L8-14 grass → poor XP/battle → long montage. The fix is level-appropriate grass (a grind-spot table in `gamedata`, or a level-aware grass picker), and/or faster KILL XP via `SOLO_WEAK_GRIND` (default OFF, wedge-prone in-battle switch — verify carefully). BOTH need a fresh multi-gym look-ahead. Do NOT re-introduce a bigger bite until grind-spot XP-rate is fixed — on adequate grass a bigger bite might then be fine.

### ⇒ SHIFT-9 FRONTIER (priority order):
1. **TEAM-DEPTH AT KOGA IS STILL OPEN — she hasn't REACHED Koga yet** (it's gated behind the Poké-Flute chain). The ns8_erika baseline correctly recognized the Snorlax block and **redirected into the Rocket Hideout for the Silph Scope** (`QUESTLINE STRIKE: Rocket Hideout — attempt 1/3`) — the NS#4/#5 chain (Silph Scope → Pokémon Tower → Poké Flute → wake Snorlax → Route 12/13 → Fuchsia → **Koga**). The run may or may not reach Koga in the 45-min budget. **AT RESUME: read the ns8_erika.log tail** (`grep 'KOGA -> \|Soul ?Badge\|blacked\|whiteout\|STATE IN: Fuchsia\|Silph Scope\|Poké Flute'`). IF she reached Koga: did the L20-22 evolved bench (Mr.Mime Psychic + Diglett→Dugtrio Ground, both 2× vs poison) + Venusaur L45 CLEAR it? That answers whether ANY team-depth lever is needed pre-Koga. IF she's still mid-chain: re-run a FRESH `surge_done_kit` 45-min (or resume) and watch it push Rocket Hideout → Flute → Koga.
2. **GRIND-SPOT ADEQUACY (the real watchability lever, well-scoped above)** — level-appropriate grass and/or `SOLO_WEAK_GRIND`. Verify-gated (fresh multi-gym look-ahead; wedge-prone switch).
3. **Carry-overs (unchanged, lower pri):** deeper WHITE-BOX wedge on legit mid-battle heals (E4-livelock family, needs rule-15 frame-grabs); "RIVAL beat vs Gary" strat-memory mislabel (cosmetic); keeper cave-descend slowness (watchability).

**RE-RUN cmd:** `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 45`. Decision check: `../.venv/Scripts/python.exe -u recon_lopsided_grind_check.py` (13/13). Canonical Champion save UNTOUCHED (all look-aheads on scratch surge_done_kit; `36b4998` remains the committed, now gym-boundary-VERIFIED lopsided fix — no new code this shift, correctly).

---

## ✅ NIGHT-SHIFT #7/#6 DONE + BANKED (2026-07-11, shift 8 landed it) — the SEVERELY-LOPSIDED-BENCH grind fix is COMMITTED (`36b4998`), verified CLEAN on a fresh look-ahead: the dedicated bench grind (which fired 0× the whole NS#5 climb) now fires for ONE bounded stint/badge and the bench actually levels — NO parked road, NO livelock. Canonical UNTOUCHED.

### ✅ WHAT SHIFT 8 CONFIRMED + BANKED — `36b4998` (mode-side, flag `POKEMON_LOPSIDED_GRIND` default ON, canonical UNTOUCHED):
Read the tail of the NS#7 look-ahead (`ns7_lopsided.log`, surge_done_kit 30-min) — the fix is CLEAN on all 3 criteria:
- **(a) bench floor crossed L14** — `GRIND-WEAK: floor crossed L14 (levels [14,14,36,14,16]) — done` → ace restored to lead.
- **(b) head_to_gym RESUMED, no parked road** — LOPSIDED-BENCH fired **EXACTLY once** (badge 3), then `PICK head_to_gym`
  toward Erika/Celadon from line 19349 on (`questline_flash` reframe, PROGRESS GREEN, MOVED). The `_lopsided_grind_done`
  bound held: it never re-fired. 8 travel wedges total, max x4 — **zero livelocks, zero treadmill**.
- **(c) leveled bench** — run ended at Route 9 marching to Celadon, party `[venusaur L37, abra L14, spearow L14,
  rattata L14, diglett L16]` — a genuinely evened bench (L14-16 behind an L37 ace, vs NS#5's L10-14 behind L48). Dex 7.
- Decision check `recon_lopsided_grind_check.py` = **13/13 ALL PASS** (re-verified this shift). The ONE grind stint
  consumed most of the 30-min wall-clock (that's WHY she didn't reach Celadon in-window — the grind ate the budget,
  NOT a park), so shift 8 launched a **45-min confirmation run** (`ns8_erika.log`) to see the leveled bench reach BADGE 4.

### ⇒ SHIFT-8 FRONTIER (the fix banked; the OPEN questions, priority order):
1. **⏳ IN FLIGHT — does the leveled bench CLEAR BADGE 4 (Erika) + progress toward Koga?** A 45-min surge_done_kit
   look-ahead is RUNNING (`G:/temp/longrun/ns8_erika.log`, launched ~start of shift 8). AT RESUME: read its tail —
   grep `EARNED\|Rainbow\|LOPSIDED-BENCH\|GRIND-WEAK: floor crossed\|STATE IN` + `TRAVEL WEDGE` counts. CONFIRM she
   reaches Celadon, wins Erika (grass gym — Venusaur solos it but a leveled bench helps the road there), and the
   lopsided stint fires again at badge 4 (re-arm on milestone rise). If she clears Erika with a leveled bench + no
   park → the fix is fully proven across a gym boundary; bank that and push toward badge 5 (Koga).
2. **DEEPER team-depth — one +6 bite/badge is BOUNDED but NOT milestone-complete.** The severity gate is the L31
   Erika milestone but the bite only pins to `floor+6` (L8→L14), so the bench arrives at each gym UNDER milestone
   (L14 vs L31), relying on road-bench-XP for the rest. That's the deliberate anti-park bound. If shift-1's confirm
   run shows the bench STILL too thin at Koga (the poison wall), the next lever is one of: (b) `SOLO_WEAK_GRIND`
   (give the bench the KILL XP not just participation — faster/stint, higher faint-variance; audit + flag-gate);
   OR a BIGGER bite when severely under (e.g. `floor + min(milestone-floor, 12)` instead of +6) — but ⚠️ that
   re-approaches the celadon_run1 27-level parking risk, so ANY bigger-bite change needs a FRESH multi-gym look-ahead
   proving no treadmill (NS#1's hard gate). (c) Abra→Kadabra (L16) is already crossed by the +6 bite from L13 —
   verify Abra actually evolved in the ns8 run (grep `evolv\|Kadabra`).
3. **Carry-overs (unchanged, lower pri):** the deeper WHITE-BOX wedge on legit mid-battle heals (E4-livelock family,
   needs rule-15 frame-grabs — do NOT fix blind); the "RIVAL beat vs Gary" strat-memory mislabel on stuck trainer
   battles (false grudge escalation — cosmetic); the keeper cave-descend slowness (watchability, not a livelock).

**RE-RUN cmd:** `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe
-u recon_longrun.py surge_done_kit.state 45`. **Decision check:** `../.venv/Scripts/python.exe -u recon_lopsided_grind_check.py`
(13/13). Flag off = `POKEMON_LOPSIDED_GRIND=0` (defaults-safe revert to prior behavior). ⚠️ SUSPECT fixture
`snorlax_woke_kit` (NS#5) — prefer a FRESH `surge_done_kit` climb.

---

## ✅ NIGHT-SHIFT #6 (2026-07-11, shift 6) — TEAM-DEPTH lever (a): the SEVERELY-LOPSIDED-BENCH grind dominance. COMMITTED `36b4998` (shift 8) after the look-ahead confirmed it CLEAN. (detail below, kept for reference)

### WHAT NS#6 IS DOING (the fix, uncommitted until the look-ahead confirms):
**Lever (a) from NS#5 — make the dedicated grind fire when severely lopsided.** ROOT (NS#5): the `battle` pick
(→`grind_weak_members`) was ALWAYS out-voted by the forward-drive (`head_to_gym`) because a solo L48 carry can
march the road forever, so the oracle never STOPS to train an L10-14 bench → abra frozen L13, never evolves to
Kadabra. FIX (campaign.py, flag `POKEMON_LOPSIDED_GRIND` default ON): `_bench_severely_lopsided(state,prep_t)`
returns the gym milestone iff the bench is SEVERELY behind BOTH the milestone (`milestone-floor>=12`) AND the
ace (`lead-floor>=15`) — the solo-carry + dead-weight shape; a new dominance block in `_available_actions` then
pops the march options (head_to_gym, wander_catch, travel:*) so `battle`→grind_weak_members runs ONE bounded +6
participation-XP bite (evens the bench, crosses Abra's L16 evolution). **PARK-PROOF by 3 bounds:** (1) only fires
with the +6 pin armed (`_prep_team_target` non-None), which retires after one bite + won't re-arm until the next
badge (NS#2 machinery); (2) the milestone is marked done after a completed stint (`_lopsided_grind_done`) so an
un-levelable bench releases instead of looping; (3) no-grass hits the PREP STAND-DOWN. So ≤1 stint/badge — never
the celadon_run1 27-level road-parking marathon. Levers (b) SOLO_WEAK_GRIND + (c) Abra→Kadabra: (c) is achieved
as a side effect of (a)'s +6 bite (L13→L19 crosses L16); (b) deferred (higher variance).

### ⇒ IF THE LOOK-AHEAD IS CLEAN (bench levels, no parked road, reaches badge 4+): commit + this is the NS#6 bank. IF IT PARKS/TREADMILLS: set `POKEMON_LOPSIDED_GRIND` default "0" (one char) — the fix is defaults-safe (reverts to prior behavior) — and re-diagnose the cadence. Re-run: `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 30` → grep `LOPSIDED-BENCH | GRIND-WEAK | party=`. Decision check: `../.venv/Scripts/python.exe -u recon_lopsided_grind_check.py` (13/13). Canonical Champion save UNTOUCHED (look-ahead on scratch fixture).

---

## ✅ NIGHT-SHIFT #5 DONE (2026-07-11, night_shift.ps1 shift 5) — killed the NS#4 hard wall (b): the Route-10 (11,79) **4,087-wedge grind livelock**; RE-VERIFIED the clean autonomous climb badge-3 → BADGE 4 → Silph Scope → Pokémon Tower → Poké Flute chain. ONE commit (63d7c14), canonical Champion UNTOUCHED. START HERE.

### ✅ WHAT NS#5 BANKED — `63d7c14` (mode-side, one grind() guard, canonical UNTOUCHED):
**Killed the Route-10 (3,28)@(11,79) grind livelock** — NS#4's frontier wall (b), the 3,927-wedge hard stop.
ROOT (exact): after a Route-13 whiteout stranded her at the Lavender-side SOUTH pocket of Route 10, `grind()`
paced `travel(arrive_coord=grass)` toward the NORTH grass — which is **cliff-sealed** from the south pocket
(the only crossing is Rock Tunnel). Every waypoint travel returned `"no_path"` FAST, but grind()'s inner loop
only caught `battle_loss`/`need_heal` → it re-fired the same unreachable waypoints for the whole 480s budget
(`ns4_koga.log`: **4,087 identical TRAVEL WEDGEs at (11,79)** — an unwatchable hard stop). FIX: cap the
`no_path`-WITHOUT-LEVEL-GAIN count per grind() call (`GRIND_NOPATH_CAP=6`, mirrors the existing heal-thrash /
one-way-strand guards) → mark the map grind-dead + return `no_safe_grass` so the caller stands down and
`_grass_target` picks a Center-reachable spot. **Fail-open**: any level gain means an encounter fired (grass IS
reachable) → `lvl()>lvl_start` makes the cap permanently unreachable, so a productive grind never trips it.
**VERIFIED e2e** (`snorlax_woke_kit`→Koga look-ahead, `ns5_koga_verify.log`): reproduced the exact strand →
`GRIND: 6 grass-unreachable travels on (3,28) … sealed off from this pocket` → she routes (3,28)→(3,4) Lavender
(escapes the pocket) → grinds reachable grass west. **(11,79) wedges 4,087 → 6.**

### ✅ RE-VERIFIED THE CLIMB (fresh `surge_done_kit` 30-min look-ahead, `ns5_fresh.log`, my fix in place):
badge-3 Vermilion → **BADGE 4 (Erika / Rainbow, Venusaur solo'd the grass gym)** → Rocket Hideout → **Lift Key**
→ **Silph Scope** → **Pokémon Tower** → **Poké Flute** chain → back to Lavender on the Fuchsia road. Party grew
to 6 (dex 10) via the keeper router. **45 total travel wedges across the WHOLE climb — every one small
(max 7) and self-recovered, ZERO livelocks, 0 grind-spins.** A clean, watchable autonomous stretch — this is the
NS#4 chain re-proven from a CLEAN surge_done_kit graph with the Route-10 livelock removed.

### ⇒ NS#5 FRONTIER — the BINDING wall is now unambiguously **TEAM-DEPTH (wall a)**, precisely diagnosed:
The bench does NOT level. Fresh-run final party: **Venusaur L48 / Spearow L13 / Rattata L14 / Abra L13 /
Diglett L22 / Meowth L10** — a solo L48 carry behind an L10-14 bench, four badges in. Confirmed mechanism
(read the logs, don't re-derive):
1. **The DEDICATED bench grind (`grind_weak_members` via the "battle" pick) fired 0× the entire climb** — the
   proactive prep pin (`prep=14` set 92×) is ALWAYS out-voted by the forward-drive (`head_to_gym`). So the ONLY
   bench leveling is `road-bench-XP` participation on transit legs, which is **far too sparse** (billed roads
   have few wild battles) → abra/meowth frozen at L10-13.
2. **When the reactive grind DOES fire (after a gym/gauntlet LOSS), the participation switch banks the KILL XP
   on the ACE, not the weak mon** — `ns5_koga_verify.log`: 3 Route-13 losses, reactive grind fired 1×, ace went
   L47→50 while the bench barely moved (spearow L12→13, abra stuck L12). The switch protects the weak mon from
   fainting but the ace soaks the XP → the lopsidedness gets WORSE, not better.
3. **Consequence at the wall:** she reaches the Route-13 gauntlet / Koga (poison, resists her grass STAB),
   loses on attrition (solo carry + dead-weight bench), retreats, and — pre-fix — stranded on Route 10 (now
   caught by 63d7c14, but the underlying team is still too thin to WIN Route 13/Koga). **Abra never reaches
   L16 → never evolves to Kadabra** (the Psychic hard-counter to Koga AND Sabrina) — the single highest-payoff
   missing lever.

**THE FIX (do it CAREFULLY — this is the delicate core, verify-gated, NOT a blind unattended edit):** the bench
must actually level between gyms. Two coupled levers, both risky (the celadon_run1 27-level marathon parked the
road; NS#1/#4 warn: **DO NOT ship a bench-grind cadence/dominance change without a FRESH multi-gym look-ahead
confirming no treadmill/parked-road**):
- (a) **Make the dedicated grind FIRE when severely lopsided** — let the prep pin out-vote the forward-drive for
  a BOUNDED dedicated grind stint when the weakest levelable bench mon is drastically under the gym milestone
  (gate it tight so it can't treadmill; the static-milestone cap + `_bench_done_milestone` re-arm already bound
  it to ~once/badge). This is constitution-aligned (a real player STOPS to train a lopsided team, narrated) —
  see `TEAM_DEPTH_ROOT_FIX.md`.
- (b) **Give the bench the KILL XP, not just participation** — for bench mons strong enough to survive weak
  grass (diglett L22, and abra once it evolves), let them SOLO-grind (full XP) instead of the ace-protecting
  switch; keep the switch only for the truly-fragile. `SOLO_WEAK_GRIND` (default OFF) is the existing hook —
  audit + selectively enable per-mon, verify no faint-thrash.
- (c) **Prioritize the Abra→Kadabra evolution** (L16, +3 from L13) as a targeted cheap-high-payoff grind — the
  Psychic answer to Koga.
**VERIFY** any (a)/(b)/(c) change with a fresh `surge_done_kit` multi-gym look-ahead: she must reach badge 4+
WITHOUT parking the road, arriving at Route-13/Koga with a bench near-milestone. If it parks → revert (flag /
one-line). RE-RUN cmd: `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1
../.venv/Scripts/python.exe -u recon_longrun.py surge_done_kit.state 30` (badge-3 fixture; ~5 min to badge 4).

**⚠️ SUSPECT FIXTURE:** `snorlax_woke_kit.state` (NS#4's promoted look-ahead bank) shows a residual
Lavender↔Route-12↔Route-10 soft-loop (head_to_gym can't push Route-12→13 and falls back to Route-10 north;
capped by 63d7c14 now, not a spin). Prefer a FRESH `surge_done_kit` climb over resuming from snorlax_woke_kit;
if you need a badge-4 fixture, re-derive a clean one from a fresh run.

**LOWER-PRI carry-overs (unchanged from NS#4):** the deeper WHITE-BOX wedge on legit mid-battle heals (E4-
livelock family, needs rule-15 frame-grabs — do NOT fix blind); the "RIVAL beat vs Gary" strat-memory mislabel
false-positive on stuck trainer battles; the Route-10 (8,20) head_to_gym travel wedge (7×, self-recovered — same
split-map class as the grind fix but via head_to_gym; low value, monitor).

---

## ✅ NIGHT-SHIFT #4 DONE (2026-07-11, night_shift.ps1 shift 4) — HUGE autonomous climb: badge-3 Vermilion → **BADGE 4 (Erika)** → Silph Scope → Poké Flute → Snorlax woken → road to Fuchsia (Koga) OPEN. TWO commits (de2d9f2, 082971c), canonical Champion UNTOUCHED.

### 🏔️ THE CLIMB THIS SHIFT (one autonomous look-ahead on surge_done_kit, enabled by the 2 fixes below):
dex 7→10 → **Flash taught** → **Rock Tunnel crossed** → Celadon → **BADGE 4 / RAINBOW (Erika — lost once, retried, WON)** → **Rocket Hideout → Silph Scope** → **Pokémon Tower → saved Mr. Fuji → Poké Flute** → **woke the Snorlax** → road south to Fuchsia (badge 5 Koga) OPEN. Venusaur L35→47. That's ~4 gyms' worth of gated content (a badge + 2 dungeons + 3 key items) in ONE run. (Look-ahead on the SCRATCH fixture — canonical post-game Champion save untouched; the sanctity check correctly REFUSED to promote badge-4 over the canonical badge-8, banked only to `G:/temp/longrun/banked_TIMEOUT`, disposable. The durable win is the 2 committed fixes, which unblock a FRESH climb through this whole stretch.)

### ✅ TWO COMMITS BANKED (both mode-side, canonical UNTOUCHED):
- **`de2d9f2` — FLASH dex-10 gate cleared e2e** (3 fixes): (A) `_catch_to_10` returns WHY it stopped so callers blacklist a catch-leg ONLY on genuine `exhausted` (was blacklisting Route 6 on a 0-balls bail); (B) restock at ANY back-leg mart town (`_FLASH_MART_LEGS={CERULEAN,VERMILION}`) so she reaches Route 6 with ammo; (C) NAV-CRITICAL QUESTLINE GUARD in `_road_bench_xp_arm` — road-bench-XP refuses to arm while `_active_questline` is set (errand ticks keep the TRUE ace leading, not a weak bench mon that livelocked Diglett's Cave with Cut + a failed switch). Proven: flash-errand → dex 7→10 → cross the cave → aide → TEACH flash → flash_done, 0 ANTI-WEDGE.
- **`082971c` — level-aware in-battle heal threshold kills the ROCK TUNNEL white-box wedge.** She'd livelocked in Rock Tunnel: a weak foe reads Fire-type ("charmander", maxhp 49 vs her 113) → the matchup-aware early-heal-at-HALF fired at 43% HP → mid-battle Potion → **action-menu white-box impostor** (`white_box=True move_list=False`) → couldn't reach the move list → weak Tackle default → 3 unresolved turns → ANTI-WEDGE abort → trainer never cleared → re-enter loop. FIX: `heal_frac=0.5` only when `threat>=2 AND NOT (we out-level the foe by ≥10)` — a much-weaker SE foe heals at the normal crit floor, so she powers through weak SE chip like a real player instead of over-potioning into the wedge. Proven e2e: crossed Rock Tunnel → Lavender, 0 white_box wedges.

### ⇒ NS#4 FRONTIER (priority order):
1. **BADGE 5 — KOGA of Fuchsia City (the immediate next gym) — I ran a look-ahead FROM the promoted scratch fixture `snorlax_woke_kit` (badge 4, Route 12) toward Koga; it hit TWO walls (`G:/temp/longrun/ns4_koga.log`):**
   - **(a) TEAM-DEPTH wall on the Route 13 gauntlet (the binding constraint).** She marched Lavender → billed leg south → Route 12/13 correctly, but LOST the Route 13 trainer gauntlet (solo Venusaur + L12-20 dead-weight bench — a Bird Keeper's pidgey team ground her down, Venusaur FAINTED with NO Revive). The road-bench-XP grind did bank a little (Spearow/Meowth L12→13) but far too slow. This is the SAME team-depth soft-wall (#2) — Koga (poison) will be worse; she needs a leveled Psychic answer (**evolve Abra→Kadabra at L16** — it's L12, so ~4 levels off) and a leveled bench BEFORE the Route 13/Fuchsia push.
   - **(b) HARD NAV LIVELOCK at Route 10 (3,28)@(11,79) — 3,927 travel wedges.** After a Route 13 whiteout, the recovery stranded her at a Route 10 south-pocket that "can't reach a Center" (`no clean path ... genuine wall/zone gap`); head_to_gym re-routed into the same pocket endlessly (the circuit breaker returned to roam but couldn't escape). MAY be a `snorlax_woke_kit` scratch-fixture world-graph artifact (the fixture was promoted from a look-ahead bank, not hand-verified) rather than a fundamental Route-10 bug — a fresh `surge_done_kit` run reached Route 12 cleanly earlier this shift. DIAGNOSE: re-run and see if (11,79) recurs; if it's fixture-only, re-derive a cleaner fixture; if real, it's a Route-10-south dead-pocket the traveler can't escape (add to the escape-hatch / poisoned-anchor guards).
   - **RE-RUN options:** fresh `recon_longrun.py surge_done_kit.state 18` (re-climbs badge-3→ here in ~5 min, clean graph), OR fix the fixture and resume from `snorlax_woke_kit`. Koga = poison (Psychic/Ground answer; Diglett→Dugtrio + Abra→Kadabra are the intended answers — LEVEL them first).
2. **TEAM-DEPTH is the recurring SOFT-WALL (the core PASS-3 mission, still open).** Erika took a loss-then-retry because the bench is dead weight (Spearow L12, Abra L10 Teleport-only, Meowth L10) and Venusaur solos everything. The loss-bump recovery WORKED (retry won), but it's fragile — Koga/Sabrina/Agatha will punish a solo-carry harder. The real fix stays TEAM_DEPTH_ROOT_FIX.md: level the bench (esp. **evolve Abra→Kadabra at L16** — psychic hard-counters Koga's poison AND Erika's), field coverage. NB the keeper router + road-bench-XP exist; the gap is they don't level the bench ENOUGH between gyms (she arrived at Erika with an L10-20 bench behind an L40 lead).
3. **⚠️ DEEPER WHITE-BOX WEDGE still lurks (dedicated session, needs rule-15 frame-grabs).** `082971c` only removed the Rock-Tunnel TRIGGER (over-potioning a weak foe). The `_settle_action_menu` B-drain (battle_agent.py ~805) still fails to recover the action menu after a LEGITIMATE mid-battle `use_item` (E4/tough-gym heals) → the E4-livelock-family root ([[pokemon-e4-livelock-family-killed]]). Characterize with a live frame-grab of the post-item menu state; do NOT fix blind (high-risk to working E4 battles).

**WATCHABILITY / MINOR BUGS surfaced (lower pri):** (1) heavy heal-excursion bouncing during the flash errand (paralyzed lead → Center round-trips) — slow on a watch; (2) a TRANSIENT Saffron-south-gate (map 18,0) misroute self-recovered via the circuit breaker (head_to_gym briefly tried the guard-gated Saffron gate before re-routing Route 6→Route 5→Cerulean — a 3rd instance of the gate-blind-routing class, but self-recovering, not blocking); (3) the "RIVAL beat vs Gary" strat-memory mislabel fires on stuck trainer battles (false grudge escalation — a strat-memory rival-detection false-positive); (4) the Erika COVERAGE-TEACH taught Cut over a Venusaur slot — turned out USEFUL vs grass (Cut x1 > Razor Leaf 0.25x) and did NOT clobber Razor Leaf, but verify the coverage-teach never eats a signature move on other gyms.

---
## (shift-1 detail, for reference) 🔨 NS#43 shift-1 — Route-12 class KILLED + growlithe false-positive fixed (COMMITTED 97e23b6):

**FIX 1 — Route-12 wedge CLASS-KILLER (VERIFIED e2e, the NS#43 headline).** NS#42 patched TWO of the Route-12
entry paths (off-road steer `2eeb5fc`, keeper router `fc78576`); a THIRD leaked via the roam `_grass_target`
east-steer. ROOT PATTERN: every routing site computed its avoid as `_wall_avoid` ONLY, so each surgical fix missed
the next site. FIX (ONE place): folded `_story_gate_avoid` INTO `_wall_avoid` (campaign.py ~6605) so EVERY routing/
reachability path inherits the Route-12/16 pre-Flute avoid from one source. Post-Flute empty → zero behavior change.
**VERIFIED e2e:** two surge_done_kit look-aheads (`ns43_reverify2.log`, `ns43_shift1.log`) = **0 Route-12 entries,
0 east-steers, diglett caught** (was 1,391 wedges). Class dead.

**FIX 2 — growlithe FALSE-POSITIVE (re-diagnosed + repurposed; the earlier Saffron guess was WRONG).** The prior
in-flight session added `_saffron_gate_avoid` (block Saffron 3,10 pre-Tea) — but `ns43_shift1.log` proved it INERT:
world.route reaches growlithe's hosts (Route 7 = 3,25 / Route 8 = 3,26) NOT via Saffron but via **Lavender Town
(3,4)** — the header edge Route 10 (3,28)→Lavender that really walks through un-crossed **Rock Tunnel** (dark pre-
Flash). (The Saffron-side hub 3,11 that also borders Route 7/8 has NO learned edges, so Lavender is the ONLY graph
path there — avoiding Lavender alone makes BOTH hosts None.) Pre-Flash the growlithe offer fired and **PING-PONGED**:
`fetch_keeper` hopped her EAST toward Lavender while the westward Flash errand (head_to_gym) walked her back — same
two tiles (3,27)(71,10)↔(3,28)(0,10), 14 no-progress decisions → **STALL** (a REAL livelock, not just churn — it
BLOCKS the Flash errand). FIX: repurposed the method → **`_rock_tunnel_gate_avoid`** (campaign.py ~6648): avoid
Lavender (3,4) in the KEEPER ROUTER ONLY (both call sites 4611 + 4695) until HM05 Flash is TAUGHT
(`st.party_knows_move(b,148,pc)`). Router-scoped so head_to_gym still pushes toward Lavender to OPEN Rock Tunnel;
diglett (Diglett's Cave off Route 11, not across Lavender) is unaffected. New `LAVENDER=(3,4)` const. **VERIFIED:**
`recon_keeper_router_check` ALL PASS (K/K-control renamed to rock-tunnel-gate), static 11/10, deposit PASS; direct
route probe on the real world_model = pre-Flash both growlithe hosts None, post-Flash they open. **e2e RE-RUNNING
(`ns43_shift1b.log`)** to confirm the ping-pong is broken + the Flash errand runs west to build species.

**RE-VERIFY NS#43:** `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe
-u recon_longrun.py surge_done_kit.state 14` → grep `catch_pokemon -> caught | TRAVEL WEDGE | MAP TRANSITION.*-> \(3, 30\)
| routing to Route [78] for planned keeper 'growlithe' | FLASH ERRAND | DEX-BLOCKED` (expect diglett caught + ZERO
Route-12 entries + ZERO growlithe offers + Flash errand progressing west toward Route 2 / building species).

---

## ✅ NIGHT-SHIFT #42 DONE (2026-07-11) — the DOMINANT NS#41 blocker (CELADON-NAV WEDGE) KILLED in BOTH its entry paths + BALL ECONOMY wired → on the REAL surge_done_kit fixture she buys balls, descends Diglett's Cave, CATCHES diglett, and marches the billed road to the Rock Tunnel approach with ZERO wedges. START HERE.
**THREE commits banked (`2eeb5fc` off-road steer + ball economy; `6b36438` docs; `fc78576` keeper-router story-gate), mode-side, canonical Champion save UNTOUCHED. Fixes proven e2e (0 travel wedges, 0 Route-12 entries, `catch_pokemon -> caught`, marched to Rock Tunnel) + decision-verified (recon_keeper_router_check J/J-control).**

**⚠️ THE ROUTE-12 WEDGE HAD TWO ENTRY PATHS (same bug class — a router computing avoid as `_wall_avoid` only, never `_story_gate_avoid`). BOTH fixed this shift:**
- **`2eeb5fc` — head_to_gym OFF-ROAD ANCHOR-STEER** (`_road_step` ~10086): steered EAST onto Route 12 from Route 11. FIXED + e2e-VERIFIED.
- **`fc78576` — KEEPER ROUTER** (`_reachable_keeper_host` ~4606 + `_fetch_keeper_errand` ~4687 + `_travel_to_known` ~9249): `ns42_probe` fetched 'growlithe' (Route 8, genuinely past Rock Tunnel), `world.route` found a path THROUGH Route 12, the offer fired + the errand hopped onto Route 12 and wedged 9 ticks. FIXED (gate the offer + errand + general travel) + decision-VERIFIED (recon_keeper_router_check PASS J story-gated-host→None + PASS J-control ungated→offered; static 11/10 + deposit still pass). `_story_gate_avoid` is empty once she owns the Flute so this only bites pre-Flute.

**`2eeb5fc` — Celadon-nav wedge (WIRING fix) + keeper ball economy.**
- **(1) CELADON-NAV WEDGE (the NS#41 dominant blocker — a soft-livelock, watchability killer).** ROOT (from
  `ns41_finalproof.log`, exact trace): head_to_gym's OFF-ROAD ANCHOR-STEER in `_road_step` (campaign.py ~10086-10104)
  computed its route with `avoid = self._wall_avoid(state)` ONLY — it did NOT include `_story_gate_avoid`. So on
  Route 11 (3,29) it logged *"off-road at (3,29) — steering toward road anchor Celadon City (east)"* and edge-traveled
  EAST onto Snorlax/Flute-gated **Route 12 (3,30)** (the graph learned Route 11<->Route 12 while she detoured for the
  Diglett keeper, but Route 12 is blocked pre-Flute). There the Snorlax NPC dead-ended every direction → `TRAVEL WEDGE:
  identical fp x4 -> no_route` **×1,391** at (13,70), and entering also spawned the flute questline whose ANCHOR-FIRST
  pushed her north into the SAME Snorlax. The head_to_gym WARP-ROUTE path (campaign.py ~10343) ALREADY applies
  `_story_gate_avoid` + the Saffron-bypass; the off-road billed-road steer just never got the same avoid set. FIX:
  mirror that avoid set onto `_road_step`'s off-road steer (`avoid = _wall_avoid | _story_gate_avoid`, `| {SAFFRON}`
  unless the target city IS Saffron). Pure WIRING (the guard existed; the steer bypassed it — exactly the mission's
  "connect the solved approach, don't rediscover it"). `world.route` always allows the SRC map (pokemon_world.py:238),
  so a resume ON a gated map can still escape. **VERIFIED:** two surge_done_kit look-aheads = **0 Route-12 entries,
  0 travel wedges** (was 1,391).
- **(2) BALL ECONOMY for keeper hunts (NS#41 frontier item B).** `_shopping_list` (campaign.py ~8692) only topped
  balls for a THIN team (`_thin_team()` = party<=3), so a party-4 hunter walked out of the Mart with too few balls and
  hit `catch_pokemon -> no_balls` in Diglett's Cave (never descended/caught — the ns41_finalproof + first ns42 run
  both stalled in the cave vestibule on 0 balls). NEW location-free **`_keeper_due(state)`** (assess -> catch_keeper,
  unlike the on-map-gated `_plan_keeper_target` which is None at the Mart) drives a bumped **`SHOP_BALL_KEEPER_TARGET=8`**
  in `_shopping_list` + `_shop_note` when a coverage keeper is DUE, even at party>3. **VERIFIED e2e:** she now buys
  `5x Poké Ball` at Vermilion (narrated in-character: *"grab a good stock so you can actually catch the teammate your
  plan wants"*), `keeper_route:cave_descend` to the Diglett floor (1,37), and `catch_pokemon -> caught`.

### ✅ FULL badge-3→Celadon-approach climb VERIFIED POSITIVE this shift (`ns42_celadon_march.log`, 30-min surge_done_kit):
buys balls → `keeper_route:cave_descend` → **CATCHES 2 diglett (party 4→6, dex 6→7)** → exits the cave → head_to_gym
steers her CORRECTLY via the BILLED ROAD (Vermilion→Route 6→Route 5→Cerulean→Route 9→CUT tree→approaching **Rock Tunnel**
`travel:3,28`), NEVER east onto Route 12. **0 travel wedges, 0 Route-12 entries the whole run.** The 2 cave battle-losses
(a dugtrio out-levels her thin bench on the deeper floor) were recovered from cleanly (heal + retry). This is the
dominant NS#41 blocker DEAD and the Celadon approach reconnected end-to-end.

### ⇒ NS#42 FRONTIER (exact next actions, priority order):
**NB on cave behavior (corrected):** the cave-descend WORKS reliably (probe3 line 16126 `CAVE-FETCH: barren for diglett
— descending internal warp (6,4)` → (1,37) → hunts). It is just SLOW: the ~45s barren-vestibule wander + long interior
paths eat wall-clock so a <20-min look-ahead only reaches the cave, not the post-catch decisions. NOT a livelock. This
slowness (not any bug) blocked the growlithe-keeper behavioral e2e — but recon_keeper_router_check PASS J proves the
`fc78576` fix deterministically, and it's the same bug class as the e2e-proven off-road steer, so it's verified.
0. **NEXT BLOCKER = ROCK TUNNEL / FLASH gate (the badge-3→4 push continues here).** She reached the Rock Tunnel approach
   (Route 9→10) with a party of 6 but **only 7 OWNED species** — HM05 Flash needs **10 owned** (Route 2 aide) and Rock
   Tunnel is PITCH DARK without it (the ONLY pre-Flute road to Lavender→Celadon). So the gate is a TEAM-BUILDING gate:
   catch ~3 more species first. The keeper plan already advanced to `growlithe` (Route 7/8) next. Probe (running at shift
   close, `G:/temp/longrun/ns42_probe.log`, 60-min surge_done_kit): does she build toward 10 owned + teach/use Flash +
   cross Rock Tunnel? The Flash chain is PROVEN historically ([[pokemon-nighttrain-shift3-flashgate-celadon-approach]] +
   the flash_errand ~campaign.py:6695) — if a fresh run bumps it, it's a WIRING re-connect, not a rediscover. DIAGNOSE
   from the probe log where the Rock-Tunnel/Flash chain does/doesn't fire.
1. **CAVE-HUNT reliability + watchability (2 sub-items surfaced this shift):**
   (a) the deeper Diglett's-Cave floor (1,37) spawns **dugtrio** (~L29) that walls her thin bench → 2 battle-losses on the
   march run before she caught. Not a livelock (heal+retry recovered), but a watch-quality + reliability gap: consider
   fielding the STRONG lead for the cave-hunt battles (she's there to catch, not to grind the bench on an over-level foe),
   or bias the catch toward the shallower diglett floor.
   (b) **per-species dedup** (NS#41 (c)) — she caught **2 diglett** (L19 + L15) building toward 6; a per-species dedup in
   `_keeper_route_target`/`_plan_wants_prebuild` stops the redundant second catch (self-terminated at party 6, low pri).
2. **WATCHABILITY: the cave descend/catch is SLOW + circly on a watch** (NS#41 (a), still open). The barren-vestibule
   wander + long interior traversal burn wall-time; a step-count barren detector (~15-20 steps no encounter → descend)
   would beat the 45s wall timer (`POKEMON_KEEPER_CAVE_FLOOR_WANDER_S`). Lower priority than "does she progress", but it's
   a real watch-quality gap on a live run.
2. **per-species dedup** (NS#41 (c)) — the prebuild un-gate can catch a SECOND of the same species while building toward
   6 (cave3 caught 2 diglett). A per-species dedup in `_keeper_route_target`/`_plan_wants_prebuild` stops the redundant
   catch (self-terminates at party 6, so low priority).
3. **THE NS#41/#39 stack still stands:** flip `POKEMON_PCBOX` default ON (owed ONE live PC-menu grab-and-look — needs
   Jonny's eyes); swap_keeper in a full run UNOBSERVED; bench pace (+6 bite cadence — HOLD pending a fresh multi-gym
   look-ahead per NS#1).
4. **FINAL-PROOF gate** — a fresh mid-game forward with all flags → she catches/fields MULTIPLE coverage keepers, levels
   the bench (road-bench-XP), preps to milestones, arrives E4-ready with a real leveled 6. Use a state WITH a world_model
   sidecar (surge_done_kit/erika_done_kit — NOT og_postopening, nav-blind).

**NB — the flaky ball-read is UNRESOLVED but no longer blocking:** at Vermilion `_ball_note` (→"ZERO Poké Balls") and
the fetch_keeper BALL GATE (`_ball_count() > 0`) disagreed in the SAME tick — the Gen-3 SaveBlock1 DMA-relocation hazard
makes `_balls_pocket_count` non-deterministic across frames. Marked low-pri in NS#41; the ball-economy fix (buy to 8 when
a keeper is due) makes her over-stock enough that the flaky read no longer starves the catch. A real fix = read the ball
count once/tick during a stable frame (cache it), or resolve the SaveBlock1-moved read — deferred.

**Re-verify NS#42:** `LONGRUN_BATTLE_LOG=1 POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe
-u recon_longrun.py surge_done_kit.state 14` → grep `bought.*Ball | keeper_route:cave_descend | catch_pokemon -> caught |
TRAVEL WEDGE | MAP TRANSITION.*-> \(3, 30\)` (expect balls bought + descend + caught + ZERO wedges + ZERO Route-12 entries).

---

## ✅ NIGHT-SHIFT #41 DONE (2026-07-11) — CAVE ENCOUNTER-FLOOR traversal CRACKED → keeper router CATCHES a cave-gated keeper END-TO-END (party 4→6, Diglett); `POKEMON_KEEPER_STATIC_ROUTE` flipped default ON.
**ONE commit banked (`f785acf`), mode-side, canonical Champion save UNTOUCHED (post-game party-full → router short-circuits to None).**
Finished NS#40 FRONTIER item 0 (the cave encounter-floor gap) and PROVED the whole keeper-acquisition chain end-to-end.

**`f785acf` — cave-descend + errand-drives-cave-catch (VERIFIED e2e).** NS#40 left the router able to ROUTE into
Diglett's Cave but unable to CATCH (she wandered the encounter-less Route-11 vestibule (1,38) forever). ROOT (found via
look-ahead): after `_enter_host_via_gateway` steps into the cave in the SAME tick, `_fetch_keeper_errand` called plain
`catch_one()` (default 300s, NO descend) instead of the descend-aware `_cave_fetch_catch` → she burned the whole budget
on the barren vestibule and never took the internal `(6,4)→(1,37)` warp to the Diglett floor. THREE mode-side fixes
(campaign.py): (1) route the SAME-TICK cave arrival through `_cave_fetch_catch` so the descend fires on FIRST entry;
(2) BALL GATE on the `fetch_keeper` offer — never offer a keeper detour with 0 balls (else she routes into a cave she
can't catch in and spins re-entering it — the 3-ball soft-livelock); (3) flip `POKEMON_KEEPER_STATIC_ROUTE` default ON
(catch proven) + lower the per-floor barren wander default 90→45s. **VERIFIED e2e** (surge_done look-ahead, Vermilion
badge-3): PICK fetch_keeper → ride Vermilion→Route 11 static gateway → step into Diglett's Cave (1,38) → wander barren
vestibule → `CAVE-FETCH descend` the internal (6,4) warp → floor (1,37) → encounter + CATCH diglett → **party 4→6** →
plan advances to next keeper (growlithe) → party full so fetch_keeper gates off → PICK head_to_gym (resumes the road, NO
livelock). Decision checks green: `recon_static_keeper_check` 11/11, `recon_keeper_router_check` ALL PASS (fixed a stale
stub the NS#40 refactor had broken), `recon_deposit_check` ALL PASS.

### ⇒ NS#41 FRONTIER (exact next actions, priority order):
   **✅ SOLVED THIS SHIFT (commit `591442d`) — keeper acquisition now ACTIVATES on a dinged team (the "arrives thin"
   root).** The ns41_real look-ahead caught the real surge_done_kit (57% HP lead) marching PAST Diglett's Cave because
   `fetch_keeper` was gated `and not needs_heal()` (NS#39 "don't detour on a dinged team") and the oracle never healed.
   FIX: relax the heal-gate ONLY for STATIC-GATEWAY hosts (door-caves adjacent to the route, no gauntlet — Diglett's
   Cave) while still gating when CRITICALLY hurt; learned-route/gauntlet hosts (abra via Nugget Bridge) keep the strict
   gate. VERIFIED e2e on the REAL fixture: fetch_keeper offered+picked at 57% -> cave -> descend -> CATCH diglett
   (party 4->5, L20) -> grinds the bench. No heal-loop (0 heal picks / 12 fetch picks), no livelock.

0. **⭐ I RAN THE FINAL-PROOF GATE (surge_done_kit, 30-min look-ahead, `G:/temp/longrun/ns41_finalproof.log`) — it
   surfaced TWO next walls, in priority order:**
   **(A — DOMINANT, fix first) CELADON-APPROACH NAV WEDGE — soft-livelock, watchability-killer.** From badge-3
   Vermilion, head_to_gym routes toward Celadon (gym 4) via **Route 12**, which is BLOCKED (the sleeping Snorlax /
   Saffron drink-gate forces the long way). She soft-livelocked on **Route 12 (13,70)**: `TRAVEL WEDGE: identical fp
   x4 → no_route` repeated **1,391 times** — the circuit breaker prevents a frozen frame (returns to roam) but roam
   re-picks the same blocked `travel:(11,112)` endlessly → she NEVER reaches Celadon. This matches the KNOWN
   "head_to_gym warp-route is GATE-BLIND + preempts billed road" issue ([[pokemon-nighttrain-shift3-flashgate-celadon-approach]]);
   prior passes REACHED Celadon (erika_done_kit exists), so per the mission this is a **WIRING failure — connect the
   solved Celadon approach, don't rediscover it.** DIAGNOSE: is head_to_gym routing gate-blind toward Snorlax-Route-12
   instead of the billed/solved path? Does surge_done_kit's world_model lack the learned Celadon path so head_to_gym
   falls back to a blocked warp-route? Check `gamedata/frlg_gates.json roads[Celadon]` + the head_to_gym warp-route vs
   billed-road priority (memory: "on no_gym_route, first check roads[<gym city>] exists"). This is the binding blocker
   to the whole climb from this fixture — she can't progress past badge 3 until it's connected.
   **(B — keeper-catch RELIABILITY, ball economy) confirmed this run.** She fetched+descended to the Diglett floor
   (1,37) but hit `catch_pokemon -> no_balls` ×5 → "ran out of Poké Balls" → 0 catches (vs the real2 run which caught
   with the same 3 balls — pure RNG: 3 balls broke free). ROOT: `_shopping_list` (~8693) only tops balls when
   `(party<=3 OR balls<2) AND balls<5`, so a party-4 hunter buys ZERO and hunts with 3. FIX: wire the ball-buy to the
   keeper plan — when `catch_keeper` is DUE, top balls to ~8-10 at the Mart even at party>3, so a keeper catch is
   reliable (diglett is easy but even it broke 3 balls once; abra/growlithe will be worse). VERIFY: re-run
   surge_done_kit, confirm she buys balls at Vermilion then reliably catches diglett.
   **Lower-priority tunes the run also flagged:** (c) the 45s BARREN-VESTIBULE wander is slow/circly on a watch — a
   step-count barren detector (~15-20 steps no encounter → descend) beats the wall timer
   (`POKEMON_KEEPER_CAVE_FLOOR_WANDER_S`); (d) the prebuild un-gate catches a SECOND of the same species while building
   toward 6 (2 diglett in the cave3 run) — a per-species dedup would stop it (self-terminates at 6, low pri);
   (e) cave step-encounter grind for the L45→55 E4-prep push (unbuilt). Fix A, then B, re-run, iterate toward E4-ready.

   **(SUPERSEDED — kept for context) prior top blocker, now fixed by 591442d:** On the REAL surge_done_kit (lead at 57% HP), she stood in Vermilion — Diglett's Cave one
   short static hop away — and `fetch_keeper` was NEVER OFFERED because the offer is gated `and not self.needs_heal()`
   (campaign.py ~9507, the deliberate NS#39 "don't start a detour on a dinged team" rule). She has a Center RIGHT
   THERE but the oracle picked stock_up→stock_up→**head_to_gym** (57% isn't urgent enough to pick `heal`), marched
   Vermilion→Route 6→…→Cerulean→Route 9, PAST the diglett window, still a thin 4-mon team. So the whole verified
   cave-catch chain sits idle on any realistically-dinged run. **THIS is the "arrives thin" root the mission targets.**
   THE FIX (nuanced — do it carefully with a look-ahead, NOT a rushed end-of-context edit; that's why I banked instead
   of shipping it): make her HEAL-THEN-FETCH when a keeper is DUE + reachable + she's dinged + a Center is on THIS map
   — i.e. connect the keeper plan to healing so she tops up and un-gates the detour instead of marching off. Options:
   (a) when a keeper is DUE and she's in a Center town, bias the oracle toward `heal` over `head_to_gym` (un-gates
   fetch_keeper next tick); (b) relax the keeper-offer gate from `not needs_heal()` to `not <critical>` for a SAFE
   short hop (Vermilion→Diglett's Cave has NO gauntlet) while keeping the strict gate for gauntlet keepers (abra via
   Nugget Bridge). ⚠️ HEAL-LOOP RISK: heal→fetch→dinged→heal must not livelock — verify a full multi-tick look-ahead
   from surge_done_kit shows heal→fetch_keeper→CATCH→resume, no ping-pong. Success = on the REAL (un-injected) fixture
   she catches diglett and marches on with party 5+. NB she ALSO already HOLDS Abra (L10, psychic keeper) needing only
   LEVELING — so bench-pace/road-XP is the parallel team-depth lever even if keeper-catching stays gated.
0b. **BALL ECONOMY (secondary, only bites once #0 is unblocked).** The ball-buy foresight (`_shopping_list`,
   campaign.py ~8693) only tops balls when `(_thin_team() [party≤3] OR balls<2) AND balls<5` → a party-4 hunter with 3
   balls buys ZERO at the Mart. Diglett is EASY (catch rate 255) + Venusaur has Sleep Powder, so 3 balls sufficed in
   the e2e proof; a harder/awake keeper would burn out. FIX: wire the ball-buy to the keeper plan — `catch_keeper` DUE
   → top balls to `SHOP_BALL_TARGET` (bump 5→~8) even at party>3. (Low priority until #0 lands — she never reaches the
   catch on a real run yet.)
1. **THEN the NS#39/#40 stack stands (unchanged):** flip `POKEMON_PCBOX` default ON (owed ONE live grab-and-look —
   needs Jonny's eyes, can't headless-prove the menu on the show build); swap_keeper in a full run UNOBSERVED (needs a
   fixture where a keeper is auto-boxed at party-6); bench pace (+6 bite cadence — HOLD pending a fresh multi-gym
   look-ahead per NS#1). MINOR polish noted this shift: the prebuild un-gate catches a SECOND of the same species while
   building toward 6 (she caught 2 diglett) — a per-species dedup in `_keeper_route_target`/`_plan_wants_prebuild`
   would stop the redundant catch (self-terminates at party 6, so low priority).
2. **FINAL-PROOF gate** — a fresh mid-game forward with all flags → she catches/fields coverage keepers, levels the
   bench (road-bench-XP), preps to milestones, arrives E4-ready with a real leveled 6. Use a state WITH a world_model
   sidecar (surge_done_kit/erika_done_kit — NOT og_postopening, nav-blind).

**NB — fixtures & re-verify:** `states/workshop/surge_done_balls.*` = a 30-ball copy of surge_done_healed I made this
shift (states/workshop is gitignored, so it's local-only) — used to prove the catch isn't ball-starved. Re-create:
boot surge_done_healed, write balls-pocket slot (SaveBlock1+0x430, qty XOR the low-16 SaveBlock2+0xF20 key) to 30 via
`b.core.memory.u16.raw_write`, save + copy the 4 sidecars. Re-verify the cave catch: `LONGRUN_BATTLE_LOG=1
../.venv/Scripts/python.exe -u recon_longrun.py surge_done_balls.state 10` → grep `CAVE-FETCH|catch_pokemon -> caught`.
The 3-balls-read-as-0 oddity (surge_done_healed): fresh both `camp._ball_count()` and `BattleAgent._ball_count()` read
3, but the cave2 run's first catch_pokemon returned no_balls — unresolved, low priority (the 30-ball run catches clean).

---

## ✅ NIGHT-SHIFT #40 DONE (2026-07-11) — keeper-reachability chicken-and-egg CRACKED (routes+enters the cave); cave step-encounter wander BUILT; NEW binding gap = cave encounter-FLOOR traversal.
**TWO commits banked (`657f2d8`, `80789ab`), both mode-side, flag-gated `POKEMON_KEEPER_STATIC_ROUTE` (default OFF), canonical UNTOUCHED.**
Attacked NS#39 FRONTIER #1 (the top binding constraint) and drove it to the NEXT real wall via a behavioural look-ahead.

**1. `657f2d8` — STATIC-CONNECTION keeper route (the headline).** The cross-map router couldn't reach cave-gated keeper
hosts (Diglett's Cave) because they're UNVISITED maps absent from the learned world graph → `_reachable_keeper_host`
returned None forever → chicken-and-egg. FIX (isolated to the router, does NOT touch shared `world.route` → no
route3_caught livelock risk): `gamedata/frlg_connections.json` static `host_gateways` KB (*"Diglett's Cave is entered
from Route 2 / Route 11"*, swappable game-knowledge layer); `_host_gateways()`/`_keeper_gateway()` (nearest
ride-reachable gateway, same MAX_HOPS + `_next_step_rideable` + `_keeper_unreach` guards); a STATIC PASS in
`_reachable_keeper_host` (gateway ride-reachable NOW → returns entrance (1,36)); `_fetch_keeper_errand` gateway-drive +
`_enter_host_via_gateway` (steps through the live-read door — only the gateway map id is in the KB, source-first).
**VERIFIED:** `recon_static_keeper_check.py` **11/11** decision cases + LIVE probe on the real surge_done_kit world
model (STATIC ON → (1,36) from Vermilion; OFF → None, reproduces NS#39) + a BEHAVIOURAL look-ahead on
`surge_done_healed` (a full-HP copy of surge_done_kit — see NB): she **PICKs fetch_keeper → rides Vermilion→Route 11
(static gateway) → steps through the door INTO Diglett's Cave.** The reachability chicken-and-egg is CRACKED.

**2. `80789ab` — cave step-encounter wander for catch_one.** The look-ahead exposed the next gap: `catch_one` returned
`no_grass` in the cave (caves have no grass; wilds fire on STEP) → she couldn't catch, then ping-ponged Route 11↔cave
on the exit warp. FIX: when there's no grass but it's a TARGETED keeper fetch into an interior that hosts the species,
wander reachable WALKABLE non-warp tiles (`avoid=doors` → never steps onto the exit warp) so step-encounters fire and
the existing `catch_runner` takes the target on foot. Reuses the grass wander loop verbatim; gated to a targeted cave
fetch (normal grass-catching untouched; inert in default runs). **THREE-STATE: WIRED, partially VERIFIED** — the re-run
confirmed she routes in, computes 4 cave waypoints, and wanders them with NO oscillation, but caught NOTHING.

### ⇒ NS#40 FRONTIER — the NEW binding gap (precise, exact next actions):
0. **THE ROOT of the no-catch — FULLY DIAGNOSED this shift (structural, not a mystery):** the Route-11 entrance
   sub-map **(1,38) has NO Diglett wild table** (10,577 steps, ZERO encounters). **(1,38)'s learned warps (from the
   run):** `(4,6)→(3,29)` [EXIT back to Route 11] and **`(6,4)→(1,37)` [INTERNAL warp DEEPER into the cave]**. So the
   Diglett floor is `(1,37)` (and/or `(1,36)`, the Route-2 side). **WHY she never got there:** the cave-wander's
   `avoid=doors` (all 0x60–0x6F tiles) blocked BOTH the exit AND the internal `(6,4)` warp → she was trapped in the
   encounter-less vestibule. **TWO precise fixes needed (the next shift's build, now well-scoped):**
   (a) **CAVE DESCEND** — the cave-catch must classify warps by the LIVE-read dest: AVOID only EXIT warps (dest place
   name ≠ the cave area) but ALLOW/STEP-THROUGH INTERNAL warps (dest is another sub-map of the SAME cave area) to
   descend `(1,38)→(1,37)→…` until an encounter floor is reached (or all sub-maps tried → retire, the no-encounter-cave
   guard). `tv.read_warps` gives (xy,dest,wid); `self._place_name(dest)` classifies. Wander each floor for a bounded
   time; if no encounter, take an unvisited internal warp.
   (b) **THE ERRAND MUST DRIVE THE WHOLE CAVE-CATCH** — ⚠️ the generic ON-MAP catch is GRASS-ONLY: `wander_catch` is
   only offered when `_reachable_grass()` succeeds, so in a cave NO catch action is offered → once she's inside the
   cave, `_keeper_route_target` returns None (species on-map → "on-map un-gate owns it", but it DOESN'T in a cave) and
   she'd fall to head_to_gym and LEAVE. FIX: in `_keeper_route_target`, do NOT early-None on `_species_on_map` when the
   current map is a CAVE (interior + no grass) — instead keep the fetch_keeper errand driving so `_fetch_keeper_errand`
   runs the descend+wander catch on the floor she's standing on. (The descend/wander lives in catch_one's cave branch
   from 80789ab — extend it with (a); keep the ≤4-waypoint wander per floor.)
   Verify via the same `surge_done_healed` look-ahead; success = party 4→5 with a Diglett/Dugtrio. ⚠️ **ALSO add the
   no-encounter-cave RETIRE guard** (N wander-timeouts, no party growth, all sub-maps tried → retire to
   `_keeper_unreach`) so a genuinely barren cave can't livelock if the flag is ON.
1. **ONLY AFTER 0 lands (catch proven end-to-end): flip `POKEMON_KEEPER_STATIC_ROUTE` default "1"** (campaign.py:~122)
   + commit. DO NOT flip before — she'll livelock in the (1,38) vestibule.
2. **THEN the NS#39 frontier stands (unchanged):** flip `POKEMON_PCBOX` default ON (owed one live grab-and-look);
   bench pace (+6 bite cadence — hold pending a fresh multi-gym look-ahead); swap_keeper in a full run unobserved;
   the FINAL-PROOF gate (fresh mid-game forward, all flags, arrives E4-ready with a real leveled 6).

**NB — the `surge_done_healed` fixture:** surge_done_kit's lead is at 57% HP → `needs_heal` True → the keeper offer is
(correctly) gated (`_available_actions` line ~9399: don't start a detour on a dinged team). The oracle picked
stock_up→head_to_gym and marched off, so fetch_keeper never got picked. I created `states/workshop/surge_done_healed.*`
(full-HP + cleared status copy + sidecars) to un-gate the offer for the proof. Re-create if missing: boot surge_done_kit,
write slot HP=max + STATUS1=0 for each party mon (offsets base+0x56 / base+0x50), save + copy the 4 sidecars.
Re-run: `POKEMON_KEEPER_STATIC_ROUTE=1 POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u
recon_longrun.py surge_done_healed.state 12`. NB2: recon_longrun loads the CANONICAL world model (line 235), which also
lacks Diglett's Cave (the Sherpa never entered it — hand-grind team), so the static pass is needed in look-aheads too.

---

## ✅ NIGHT-SHIFT #39 DONE (2026-07-11) — final-proof look-ahead FINGERED a box_chaff defect (fixed) + PC/BOX withdraw/swap-IN loop CLOSED.
**TWO commits banked, both mode-side, flag-gated `POKEMON_PCBOX` (default OFF), canonical Champion save UNTOUCHED.**
Ran the FINAL-PROOF gate first (`erika_done_kit`, full party-6 chaff, `POKEMON_KEEPER_ROUTER=1 POKEMON_PCBOX=1`,
`LONGRUN_BATTLE_LOG=1`) — it did its job and fingered a real defect, which I fixed, then I completed the PC/BOX loop.

**1. `6b88b13` — box_chaff ROUTABILITY GATE.** The run's FIRST tick at Celadon: `PICK box_chaff` → deposited a
member (6→5) narrating *"making room for the mon my plan actually wants"* — but `fetch_keeper` then NEVER fired
because the due keeper (Diglett, in **Diglett's Cave**) is un-routable from Celadon. Net: team thinned, on-screen
promise broken, keeper never caught. ROOT: `_chaff_swap_target` gated only on catch_keeper-DUE, not on the keeper
being REACHABLE. FIX: extracted the router's reachability scan into a shared `_reachable_keeper_host(sp,cur,state)`;
`_keeper_route_target` calls it (no behaviour change) and `_chaff_swap_target` now boxes ONLY when the keeper is
already on THIS map (on-map un-gate catches it) OR the router can ride to a hosting map — else it REFUSES.
**VERIFIED:** `recon_deposit_check` 14/14 (3 new routability cases: fires-when-routable / None-when-unroutable /
fires-when-on-map) + live deposit 4→3; behavioural re-run confirms `box_chaff` no longer offered at Celadon
(was `PICK box_chaff`→deposit; now `PICK head_to_gym`, party stays 6).

**2. `c9346fa` — PC/BOX withdraw + swap-IN keeper (closes the loop).** `deposit_mon` (NS#38) only made ROOM; a
coverage keeper caught while FULL is FRLG-auto-boxed with no way onto the team. Added: `_box_scan()` (RAM-truth
storage read, gPokemonStoragePtr 0x03005010 / 80-byte BoxPokemon), `withdraw_mon(box,slot,pc_door)` (reverse of
deposit_mon — box-grid nav, verify party +1 & species landed, aborts LOUD 'wrong_box' if not in the OPEN box),
and `_box_keeper_swap_target` / `_swap_keeper_errand` + a `swap_keeper` roam action (deposit the weakest chaff if
full, then withdraw the keeper → chaff-for-keeper, party stays 6). **WIRED** into the roam options + dispatch
(mirrors box_chaff), self-clearing (once fielded the keeper isn't boxed → no re-offer). **VERIFIED** headless
(`recon_withdraw_check` A/B/C @ Celadon on erika_done_kit): `_box_scan` decodes the 3 boxed occupants + open box;
full-party→'full'; live round-trip deposit 6→5 then withdraw 5→6 (Weedle lands, re-proves deposit unregressed);
swap_keeper end-to-end (chaff sp19 out, keeper sp13 in, party stays 6). Re-verify: `POKEMON_PCBOX=1
../.venv/Scripts/python.exe recon_withdraw_check.py` (and `recon_deposit_check.py` 14/14).

### ⇒ NS#39 FRONTIER (the BINDING constraint is now keeper ACQUISITION, not box mechanics — priority order):
The final-proof run showed the PC/BOX system is now correct + complete, but it is DORMANT mid-game because the
keeper it serves can't be reached: **the cross-map router can't ride to cave-gated hosts (Diglett's Cave) and
misses the TIMING window** (she's already past Vermilion/Route 11 when Diglett is DUE, and MAX_HOPS=6 rightly
won't backtrack that far). Meanwhile she ALREADY HOLDS **Abra** (the psychic-sweeper keeper — answers Koga/
Sabrina/Agatha) at L14 needing only LEVELING. So the real team-depth levers are:
1. **KEEPER-REACHABILITY (top) — ROOT CONFIRMED this shift, it is NOT a timing problem.** Probed the router on
   `surge_done_kit` @ Vermilion (badge 3, the CORRECT window — Diglett's Cave sits off Route 11 right next door):
   `_reachable_keeper_host(diglett)=None` STILL. Diagnosis is PRECISE: "Diglett's Cave" IS correctly in
   `_PLACE_NAMES` (maps (1,36-38)) and `_place_to_map_index` reverses it to entrance (1,36) — the mapping is FINE.
   The failure is `self.world.route(cur, (1,36))` returns None because **Diglett's Cave is an UNVISITED map** not
   in her learned world graph, and the OFFER⟺EXECUTABLE guard (`_next_step_rideable`, campaign.py ~4515, the
   route3_caught livelock fix) correctly refuses to route to a map she can't RIDE to yet. Chicken-and-egg: she
   won't visit the cave → router can't route there → she never visits. So the router can only reach keepers on
   ALREADY-EXPLORED maps; keeper hosts (caves off the forward path) are unexplored until something routes her
   there. **THE FIX (design-heavy, next shift's headline): STATIC-CONNECTION-AWARE routing to known keeper
   hosts** — seed the router with the disassembly's static MapConnection graph (memory: MapConnection = group@8
   num@9) so it can plot a path toward an unvisited-but-known-adjacent host's ENTRANCE (Route 11 → Diglett's Cave
   warp), then hand off to the learned-graph traveler once she's on a visited edge. Keep it BOUNDED (MAX_HOPS,
   watchable) and isolate the static graph in gamedata (portability). VERIFY: re-probe `_reachable_keeper_host`
   returns (1,36) from Vermilion, then a surge_done_kit look-ahead where she actually detours + catches Diglett.
   ⚠️ Do NOT relax `_next_step_rideable` without the static graph — that reintroduces the route3_caught no_path
   livelock. NB she ALREADY HOLDS Abra (the psychic keeper) needing only leveling, so bench-pace (#2) is the
   parallel lever even before keeper-reachability lands.
2. **BENCH PACE (prep bite cadence, NS#4 frontier #3).** The bench climbs in `+6` bites (campaign.py ~5751
   `min(milestone, floor+6)`) → many grind stops to reach a gym milestone; dungeon-heavy stretches barely level
   it. A bigger bite when FAR under milestone would arrive near-milestone in fewer stops. ⚠️ HELD: the +6 SITTING
   CAP is hard-won (celadon_run1 27-level marathon parked the road) — DO NOT ship a bigger bite without a fresh
   multi-gym `og_postopening`/mid-game look-ahead confirming no treadmill/grind-wall (NS#1's explicit gate).
3. **FLIP `POKEMON_PCBOX` default ON — STILL OWED, needs ONE live grab-and-look** (from NS#38). The deposit/
   withdraw menu is wedge-prone menu-nav on the long core; headless-VERIFIED (deposit 4→3, withdraw 5→6, swap
   end-to-end) but wants a live eye on the SHOW build before default-ON. Confirm the Center-detour doesn't
   over-backtrack (watchability). Then set `POKEMON_PCBOX` default "1" (campaign.py:~129) + commit.
4. **swap_keeper firing in a FULL run is UNOBSERVED** — needs a fixture where a keeper is actually auto-boxed
   at party-6 (the look-ahead never reached a party-6 on-map catch). Errand-level VERIFIED; a live look-ahead
   observation is the last proof.
5. **FINAL-PROOF gate** — a fresh mid-game forward with both flags → she catches/fields coverage keepers, levels
   the bench (road-bench-XP), preps to milestones, arrives E4-ready with a real leveled 6. Use a state WITH a
   world_model sidecar (surge_done_kit/erika_done_kit — NOT og_postopening, nav-blind).
⚠️ CLEAN-UP noted: withdraw_mon duplicates deposit_mon's Center-entry/PC-boot prefix — an `_open_bill_pc`
extraction to dedupe is deferred (kept the VERIFIED deposit path byte-identical). Low priority.

---

## ✅ NIGHT-SHIFT #38 DONE (2026-07-11) — KEEPER ROUTER flipped ON (end-to-end catch PROVEN) + PC/BOX chaff-swap BUILT+VERIFIED. START HERE.
**TWO commits banked, both mode-side, flag-gated, canonical Champion save UNTOUCHED:**

**1. `6cdc463` — KEEPER ROUTER default flipped ON (NS#4 frontier #1 DONE).** The clean fetch+catch that
neither NS#4 fixture could show LANDED: on `bill_house_noabra` (party-3 Ivysaur L26 + Spearow + Rattata, 15
balls, 1 warp hop to Route 25, NO gauntlet — built by `recon_mk_keeper_fixture.py`): PICK fetch_keeper →
FETCH-KEEPER routed Bill's house (30,0)→Route 25 → force-caught the target **Abra** → party 3→4 → plan
advanced to the next keeper (diglett) → Diglett's Cave unreachable → router returned None (fetch_keeper
un-offered) → fell through to head_to_gym, **NO livelock**. `POKEMON_KEEPER_ROUTER` now default "1"
(full-party >=6 and post-game both short-circuit to None → canonical unaffected). Verifier still 10/10.

**2. `ec264a1` — PC/BOX chaff-swap (NS#4 frontier #2 = Tier-1 #15, the FULL-party pairing gap).** The router
only fires at party<6, so a full team of early-catch chaff (erika_done: Venusaur + Rattata/Spearow/Ekans/
Meowth/Pidgey) could never add a coverage keeper. NOW: `box_chaff` deposits the lowest-value OFF-PLAN chaff
at the current city's Center PC → party 6→5 → the router's room-gate opens → keeper added. Built (all in
campaign.py): `deposit_mon(slot,pc_door)` (reuses heal_at_center's proven Center enter/exit + ports
recon_pcbox's screenshot-calibrated menu drive); **`_find_pc_stand()` GENERAL PC locator** (scans the
interior for MB_PC=0x83 + stands below — Centers do NOT share the PC tile, Vermilion (11,1) != Route 10's,
so the hardcoded stand wedged; this reads RAM truth); `_worst_chaff_slot` (lowest-level off-plan non-lead,
via planner._is_target_line — never boxes a keeper/ace); `_chaff_swap_target` gate + `box_chaff` offer/
dispatch (fires only party-FULL + catch_keeper DUE + boxable chaff + mapped-Center city). Flag
`POKEMON_PCBOX` **default OFF**. **VERIFIED:** `recon_deposit_check.py` 11/11 decision cases + a LIVE
headless deposit (surge_done @ Vermilion, party 4→3 by RAM, PC stand auto-located (11,2)); AND an end-to-end
look-ahead (synth party-6 chaff @ Vermilion, both flags on): **PICK box_chaff → deposited Ekans L9 → party
6→5 → fetch_keeper fired → routed to Route 24 for the Abra**. The full chaff→box→router→fetch chain runs live.

### ⇒ NS#38 FRONTIER (exact next actions, priority order):
1. **FLIP `POKEMON_PCBOX` default ON — needs ONE live grab-and-look first.** The deposit menu is
   menu-nav-on-the-long-core (wedge-prone). The actuation is headless-VERIFIED (party 6→5, real menu, on
   Vermilion), but per the wedge-prone class it wants a live eye on the SHOW build (audio/render path) before
   default-ON. Also confirm the Center-detour doesn't over-backtrack (watchability). If a live deposit is
   clean → set `POKEMON_PCBOX` default "1" (one char, campaign.py:~123) + commit. To re-verify headless
   anytime: `POKEMON_PCBOX=1 ../.venv/Scripts/python.exe recon_deposit_check.py` (11/11 + deposit 4→3).
2. **WITHDRAW + auto-boxed-keeper swap-in (the PC/BOX second slice).** Right now box_chaff makes room BEFORE
   the catch. FRLG also auto-boxes a caught mon at party-6 — so a keeper caught while full sits in the box.
   Build `withdraw_mon(box_slot, pc_door)` (reverse of deposit_mon — same _find_pc_stand + the WITHDRAW menu
   branch) + a "swap at Center" hook that withdraws a boxed keeper and deposits a chaff during a heal visit
   (no extra routing). Model the menu drive on deposit_mon; verify headless via a recon_withdraw_check.
3. **FINAL-PROOF GATE (the whole point) — now runnable end-to-end.** A fresh mid-game fixture forward with
   `POKEMON_KEEPER_ROUTER=1 POKEMON_PCBOX=1`: she catches coverage keepers (router), boxes chaff when full
   (box_chaff), levels the bench on the road (road-bench-XP), preps to milestones (milestone-prep), and
   arrives E4-ready with a real 6. Use a mid-game state WITH a world_model sidecar (surge_done/erika_done —
   NOT og_postopening, which is nav-blind). Read the blocker chain; the remaining gap is likely grind-spot
   adequacy for the L45→55 E4 push (cave step-encounter grind, unbuilt) or a nav wedge on the return legs.
4. **prep bite cadence** (NS#4 frontier #3) — the +6 milestone-prep bite levels the bench slowly; a bigger
   bite when FAR under milestone arrives near-milestone faster without a grind-wall.

⚠️ NOTE on the keeper-router live catch at a solo/thin team: reaching Route 24 crosses Nugget Bridge — a
solo/very-thin team loss-loops there (NS#4 misty_done finding; the loss-guard retires the target cleanly, no
livelock, but she won't complete the catch until the team can survive the bridge). box_chaff/road-bench-XP
building a real bench is what fixes this; not a router defect.

---

## ✅ NIGHT-SHIFT #4 DONE (2026-07-11) — road-bench-XP re-validated (party-6) + CROSS-MAP KEEPER ROUTER built (NEW#2). START HERE.
**BANKED (commit 208edb5, mode-side, flag-gated `POKEMON_KEEPER_ROUTER` DEFAULT OFF, canonical untouched):**
the cross-map keeper router — the last unbuilt Part-C piece for team COMPOSITION. Full diagnosis + file:lines
in `TEAM_DEPTH_ROOT_FIX.md` §NS#4. Two look-ahead findings drove it: (1) **road-bench-XP (NEW#1) VALIDATED in
the party-6 mid-game** on `erika_done` (Venusaur L43 + frozen L9-15 chaff bench → Ekans leveled L9→**L14**,
Rattata/Spearow 15→16 as she marched, GRIND SWITCH firing, questline chained right); (2) **the confirmed gap =
team COMPOSITION** — the planner emits `catch_keeper: abra→alakazam` the whole run but she marches past
(on-map un-gate only grabs a keeper she's STANDING on). The router (`_keeper_route_target` /
`_fetch_keeper_errand` / `_place_to_map_index` in campaign.py; `fetch_keeper` action) offers a BOUNDED detour to
a nearby reachable hosting map, then the on-map machinery catches. **Decision-VERIFIED 10/10**
(`recon_keeper_router_check.py`). **Behavioral: a LIVELOCK was caught + FIXED** (offer used world.route but the
errand used naive trav.travel → no_path MACRO-RED spin; fixed with offer⟺executable `_next_step_rideable` gate +
`_travel_to_known` routing + a stall-guard that retires un-rideable targets to `_keeper_unreach`). Post-fix
route3_caught: routes Route3→Route4, retires the Mt-Moon-gated Route24/25 cleanly, resumes leveling — NO livelock.

### ⇒ NS#4 FRONTIER (exact next actions, in priority order):
1. **FINISH the router behavioral proof (a SUCCESSFUL fetch+catch), then FLIP default ON.** Neither test
   fixture could show it because BOTH gate a thin team from the keeper: route3_caught is Mt-Moon-level-gated
   before Route 24; misty_done (solo Ivysaur L22) **loss-loops at the Nugget Bridge gauntlet** en route to
   Route 24 (Abra) — `keeper_route:travel:battle_loss` → blackout → Center → retry (head_to_gym/the S.S.Ticket
   questline, also north past Nugget Bridge, would loop the SAME way — it's the solo-team problem, not a router
   defect). A late fix (committed) makes a `battle_loss` leg count as NON-progress so K losses RETIRE the keeper
   (was: blackout relocation reset the stall guard → soft loss-loop). **TO PROVE THE CATCH:** use a fixture with
   a party of 2-3 (not solo) whose keeper map is reachable with NO gauntlet — e.g. a post-Nugget-Bridge state at
   Cerulean/Route 25 with room, or seed a world_model so Route 24 is rideable and the team can survive the bridge.
   Command: `POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py <fixture>.state 12`;
   grep `FETCH-KEEPER|caught a new|CATCH: [0-9]+ reachable|UNREACHABLE`. On a clean catch → set
   `POKEMON_KEEPER_ROUTER` default "1" (one char, campaign.py) + commit. ⚠️ Detour watchability (over-backtrack?)
   is a LIVE-EYES item — tune `POKEMON_KEEPER_ROUTER_MAX_HOPS` down (4?) / STALL_CAP 3→2 if detours read too far.
2. **PC/BOX (Tier-1 #15)** — the pairing gap for the FULL-party case (erika_done: 6 chaff, router won't fire
   without room). recon_pcbox.py deposit flow proven; generalize `deposit_mon`/`withdraw_mon` + hook `catch_one`
   to box the lowest-value chaff on a full-party keeper catch → then the router adds the keeper.
3. **prep bite cadence** — the +6 bite levels the bench slowly (~5 levels/9min); a bigger bite when FAR under
   milestone would arrive near-milestone faster without a grind-wall.
4. **FINAL-PROOF gate** — a fresh mid-game fixture forward with `POKEMON_KEEPER_ROUTER=1` → she catches
   coverage keepers + levels them + arrives E4-ready. (og_postopening is an INVALID fixture — no world_model
   sidecar → empty-graph nav-blind livelock on the unbilled early gyms; use a mid-game state with a sidecar.)

## ✅ NIGHT-SHIFT #3 IN FLIGHT (2026-07-11) — frontier NEW#1 (ORGANIC BENCH XP ON THE ROAD) BUILT + decision-verified.
**NS#3 BANKED (commit a998378, mode-side, flag-gated, canonical untouched):** the exact NEW#1 fix below —
the mid-game "bench never levels" root. Two helpers in `campaign.py` (`_road_bench_xp_arm` /
`_road_bench_xp_disarm`) wired into `free_roam` around the `_route_action` dispatch: on a forward-march pick
(`head_to_gym`/`travel:`) with a bench member under its `_prep_team_target` milestone, the weakest levelable
under-target mon leads (→ XP-eligible) and `battle_agent.PROTECT_LEAD_GRIND` is armed so the PROVEN
participation switch (battle_agent.py:2554) fields the ace turn 1 — the weak mon banks a share of XP without
taking a hit. The ace is restored to slot 0 the instant the leg ends (weak lead never outlives the march).
Guard: an ACE-HP floor (`POKEMON_ROAD_XP_ACE_HP_FLOOR`=0.6) so a bench-XP leg never STARTS with a dinged ace
(commit 838e9fb). Flag `POKEMON_ROAD_BENCH_XP` (default ON; one-line revert). **VERIFIED 15/15**
(`recon_road_bench_xp_check.py`).

### ✅ END-TO-END VERIFIED (bucket a) — the owed proof LANDED this shift:
A `snorlax_done` look-ahead (`LONGRUN_BATTLE_LOG=1`, Routes 13/16 → Fuchsia) showed **5 arms → 3 LIVE
participation switches in road trainer battles → the bench leveled organically** (Spearow L15→16, Ekans
L9→10) while marching, no strand from the switch. **GOTCHA banked:** recon_longrun suppresses battle_agent's
log unless `LONGRUN_BATTLE_LOG=1` (that masked the switch on earlier passes; city-boots surge/erika/bill were
absorbed by town-nav or had spent trainers). **CAVEAT (hand-bank artifact):** the run then whited out on the
Route 13 gauntlet — but the CONTROL (flag OFF) hit the SAME critical-HP + PP-famine wall, so it's pre-existing
top-heavy-bank attrition (L48 ace soloing an L9-15 bench), NOT this fix. The ace-HP guard mitigates (defers a
dinged-ace leg to heal); on a fresh organically-built run the bench shares the gauntlet so it doesn't arise.

### ⇒ NS#3 FRONTIER (the fix is verified; next pieces):
1. **PC/BOX (Tier-1 #15) — the pairing gap for the catch half; TOP next build.** RECON DONE this shift:
   `recon_pcbox.py` already has a COMPLETE deposit drive (enter Center → walk to PC console → A → BILL'S PC →
   DEPOSIT → pick slot → confirm → verify party count 6→5 by RAM → banks `banked_PCBOX`), menu sequence
   worked out with screenshot calibration. GAPS to wire it into campaign: (a) `PC_STAND` is Route-10-specific
   → generalize the PC-console approach per Center (find the PC tile in any Center); (b) add `withdraw_mon`
   (the reverse); (c) NOT wired into campaign at all (no `deposit_mon`/`withdraw_mon` methods) → add them +
   hook `catch_one` to box the lowest-value off-plan fodder on a full-party keeper catch (Tier-1 #15). ⚠️ The
   PC menu actuation is WEDGE-PRONE (menu-nav-on-long-core) → verify live with grab-and-look, flag-gate.
   Un-gate builds toward 6; box turns it into real depth. NB recon_pcbox reads canonical (now the Champion
   save) on a RAM copy — re-point it at a party-6 workshop bank to re-verify the drive before promoting.
2. **CROSS-MAP KEEPER ROUTER** — the catch un-gate is on-map only; route to `act["where"]` via
   frlg_encounters for grass/cave keepers (Abra/Diglett/Growlithe). Note NS#2's sibling gap: the Abra
   plan-catch FIRED on-map but didn't PERSIST after a mid-catch heal (catch-persistence).
3. **grind-spot adequacy** (#1 old) — largely dissolved by NEW#1 if the owed proof lands (bench arrives
   near-milestone organically), but keep on the radar for a L45→55 E4-prep push (cave step-encounter grind).
4. **THE FINAL-PROOF GATE** — fresh `og_postopening` → 15x look-ahead → she catches keepers, LEVELS the
   bench (now via NEW#1 on the road), arrives E4-ready with a real 6, sweeps tactically to credits.

## ✅ NIGHT-SHIFT #2 DONE (2026-07-11) — frontier #2 (MID-GAME MILESTONE LEVELING) BUILT + decision-verified.
**NS#2 BANKED (commit fb26e6e, mode-side, canonical untouched):** the **mid-game milestone-cap bench-prep** —
frontier #2 below, the exact reviewed edit. `_prep_team_target`'s wall-less proactive bench-raise now caps the pin at
the team-plan's NEXT gym milestone (Brock 14 … Giovanni 52) not the ace-relative `lead-8`, and RE-ARMS on a milestone
RISE (new gym earned) as well as roster change — so the bench climbs toward each gym's level across the whole game in
bounded +6 bites instead of getting ONE bump then abandoned. **Decision logic VERIFIED 10/10** in
`recon_milestone_prep_check.py` (cap / one-bite-per-milestone / badge-re-arm / no-over-grind past milestone /
fallback treadmill-safe). ⚠️ A real trap was caught + fixed during the build: the lead-8 FALLBACK bar drifts up with
the ace, so the milestone-RISE re-arm is gated to REAL static milestones only (`_ms` guard) — else the fallback would
reinstate the ship-run-5 treadmill (verifier case 6b). **STILL OWED (the spec's ship-gate):** a behavioural look-ahead
confirming no live parking/treadmill — launched from `ss_ticket.state` (badge 2: Ivysaur L28 + Rattata/Spearow L14
bench, the ideal mid-gym probe); read `G:/temp/longrun/ns2_milestone.log` for whether the bench-raise fires, bites,
retires, and does NOT park the road. If it parks/treadmills → tune the bite cadence (the fix is defaults-safe: reverts
to prior lead-8 when planner off, so a revert is one-line if needed).

### ⇒ NS#2 FRONTIER (re-ranked by a LIVE run — #2 BUILT; a deeper root now pinpointed as the TOP next action):
The ss_ticket behavioural run (`G:/temp/longrun/ns2_milestone.log`) proved the milestone TARGET is correct + stable, but
surfaced the REAL binding constraint: **the mid-game bench never LEVELS** — she won ~8 road trainer fights Route24→
Vermilion and ALL XP went to the lead (Venusaur L28→32) while the bench stayed FROZEN L14/L14. Full diagnosis +
file:lines in `TEAM_DEPTH_ROOT_FIX.md` (NS#2 UPDATE section). Re-ranked:

**NEW #1 (TOP — organic bench XP on the road).** The participation-XP switch (`battle_agent.py:2554`) is gated to
`PROTECT_LEAD_GRIND` (dedicated `grind_weak_members` only, `battle_agent.py:141`), which push-when-carrying keeps her
out of mid-game → the bench banks nothing from road trainer wins. BUILD: field/register the under-milestone weak mon in
ORDINARY road trainer battles so it banks participation XP (extend `battle_agent.py:2550-2580` beyond PROTECT_LEAD_GRIND,
guarded per the 2584 STRAND-ROOT note so the weak mon takes no hit / never strands; drive it off a "bench under
`_prep_team_target`" signal). This levels the bench organically WHILE traveling (watchable — no grind-wall) → she reaches
each gym + badge-8 `_prep_e4_target` near-milestone, which ALSO dissolves old-#1 grind-spot-adequacy. ⚠️ Touches the
in-battle switch (Tier-1 #5, wedge-prone) — flag-gate it + verify on a PAST-Cut-gate bench bank (ss_ticket WEDGES at the
Vermilion Cut tree, so use surge_done/erika_done or a fresh run that clears the gate; do NOT verify from ss_ticket).

**Then the prior pieces (unchanged, still valid):** PC/BOX (#3), cross-map keeper router (#4 — note the run showed the
Abra plan-catch FIRED on-map but didn't PERSIST after a mid-catch heal; catch-persistence is a sibling gap), grind-spot
adequacy (#1, largely dissolved by NEW #1 if it lands).

## ✅ NIGHT-SHIFT #1 DONE (2026-07-11) — team-depth Part-C: 3 fixes banked + verified.
**BANKED + VERIFIED NS#1 (3 commits, all mode-side, canonical untouched):**
- **2dc74d5 `prep_for_e4()`** — at all 8 badges the WHOLE party is floored to the team-plan's E4 milestone (~L55),
  not the ace-relative `lead-8`. Direct fix for the NS9-14 top-heavy wall (ace solos, bench sags 8 under → swept
  in the Center-less gauntlet). NOTHING read `level_milestones` before. **BEHAVIOURALLY VERIFIED** on a giovanni_kit_g
  look-ahead: fires (prep=55), fields the REAL team (Lapras L37 / Kadabra L39 — not the chaff), and retires cleanly
  ("pushing on with the strong core") when grass can't level them — **NO livelock** in the live loop. Logic verified
  4/4 in `recon_prep_e4_check.py` (emit / retire-on-crossed / chaff-only / stalled).
- **22398ec catch un-gate + chaff-floor** — `_plan_wants_prebuild` no longer hard-blocks at pc>2: she keeps building
  toward a full 6 while a planned keeper is DUE AND catchable on the CURRENT map (junk-safe: catch_one flees
  non-targets). Verified 2/2 in `recon_ungate_check.py` (control + forced-positive). And `grind_weak_members` gained
  a `min_level` floor so E4-prep fields only the levelable real team, never drags L8-14 box-fodder toward 55.
- **9c8a33d** the un-gate verifier.

**FRONTIER — the exact next pieces, in priority order (with the solutions I worked out):**
1. **GRIND-SPOT ADEQUACY (the blocker prep_for_e4 surfaced, confirmed LIVE).** On giovanni_kit_g the bench (L37/39)
   can't reach 55 because near Viridian/Indigo the reachable grass (Route 22 L2-8, Route 2, Route 18 caps ~L44-46)
   gives L37+ mons ~0 XP → both stall → she proceeds UNDERLEVELED. This is the KNOWN NS14 gap (VR cave-grinding
   unbuilt; E4-self-grind futile). **NOTE:** this is largely an ARTIFACT of the hand-built bank whose bench was
   caught late + never levelled — a FRESH run with piece 2 below never sags this far, arriving needing only a small
   top-up. Real fix candidates: cave step-encounter grinding (unbuilt), or a high-level grind spot table in gamedata.
2. **MID-GAME MILESTONE LEVELING (highest-leverage; designed, deliberately NOT shipped blind — see the exact edit).**
   ROOT (found this shift, campaign.py `_prep_team_target` proactive block ~5289-5318): the bench target caps at
   `lead-8` in `+6` bites AND the RE-ARM GUARD retires after ONE bite per roster-signature (`_bench_done_sig`). So
   the bench gets ONE +6 bump then is abandoned as the ace runs away — the exact "arrives thin" mechanism. **THE
   PRECISE EDIT (minimal, rides the proven +6 pacing — low-risk):** (a) cap the pin at the MILESTONE not the ace:
   `self._bench_pin = min(milestone, floor + 6)` where `milestone = self.team_planner._next_milestone(badge_count,
   post_game)[1]` (falls back to `lead - 8` if no milestone); (b) RE-ARM the retired prep when the MILESTONE RISES
   (a new gym earned), not only on roster change: track `_bench_done_milestone`, re-arm if `sig changed OR milestone
   > _bench_done_milestone` — milestones change only on badge-earn (infrequent) so this can't treadmill. Keep the
   band+stalled livelock guard. This makes the bench climb toward each gym's level over the game in bounded, watchable
   bites. **DO NOT ship without a FRESH `og_postopening.state` multi-gym look-ahead** confirming no treadmill/over-grind
   (that's WHY it's held — the RE-ARM/pin machinery is hard-won: ship-run-2/5, celadon_run1). The badge-8 `prep_for_e4`
   already handles the E4 case (committed); this is its mid-game sibling. (I built a `_prep_milestone_target` helper
   generalization + reverted it — the pin-machinery edit above is the safer path, reuses proven pacing.)
3. **PC/BOX (Tier-1 #15) — the pairing gap for the CATCH half.** The un-gate builds toward 6, but the giovanni run
   showed the endgame reality: at party-6 with 3 chaff, she CAN'T swap fodder for planned keepers without box-mgmt.
   Build `deposit_mon`/`withdraw_mon` (promote `recon_pcbox.py`'s deposit flow) + hook `catch_one` to box the
   lowest-value off-plan fodder on a full-party keeper catch. This is what turns the un-gate into real depth.
4. **CROSS-MAP KEEPER ROUTER** — the un-gate is on-map only; a fresh run only catches keepers it happens to pass.
   Route to `act["where"]` (resolve via `_PLACE_NAMES` reverse + frlg_encounters) for grass/cave keepers (Abra,
   Diglett, Growlithe); Snorlax/Lapras are gift/static (quest-gated, leave to the questline). Model on `_flash_errand`.
Full diagnosis + file:lines: `pokemon_agent/TEAM_DEPTH_ROOT_FIX.md`. Loop NOT stopping — this is the build phase.

## 🚂 PASS 3 — THE NIGHT-TRAIN MISSION (2026-07-11, START HERE EVERY SHIFT): make a FRESH GO build its own team + play a watchable ~25-35hr bedroom→credits spectacle. **CONSOLIDATION, not discovery — the game is already solved; wire it together so it FLOWS.**

**PART-1 RECON IS DONE — READ `pokemon_agent/PASS3_RECON.md` FIRST** (the complete wired/unwired/needs-live-watch gap
map for every item below, with file:line fix hooks + honest buckets). Don't re-recon what it already answers.

**HOW TO RUN THIS:** unattended multi-shift night train (`night_shift.ps1`). Each fresh shift: (1) READ this
block + `pokemon_agent/PASS3_RECON.md` + `TEAM_DEPTH_ROOT_FIX.md` + `E4_TACTICAL_FRONTIER.md`; (2) continue the mission from where
the last shift banked; (3) commit-per-fix, VERIFIED from disk; (4) UPDATE this block with progress before you
run low on context; (5) never re-solve a wall pass 1/2 already killed — if a fresh run bumps a solved wall,
that's a WIRING failure (connect the existing solution, don't rediscover it).

**THE STATE:** the tactical E4 half is DONE (credits rolled, hand-grind team, commit 23487e7). The remaining
mountain = the AUTONOMOUS build: a fresh GO must catch + level its OWN 6 and arrive E4-ready, then the
committed tactical fixes win. Canonical 2026-07-07 Champion save UNTOUCHED; all work on scratch banks.

### THE BUILD (tractable core first, then the spectacle layer; check what EXISTS before building — most is built):
1. **TEAM-DEPTH Part-C executor (THE HEADLINE — full diagnosis + file:line targets in `TEAM_DEPTH_ROOT_FIX.md`):**
   wire `catch_keeper` to route+catch a FULL 6 across the game (un-gate `_plan_wants_prebuild` pc>2 @campaign.py:4355;
   `_plan_keeper_target` must TRAVEL to `act["where"]` via frlg_encounters.json, not just current-map); wire
   `grind_to`/`develop_bench` → a real bench-leveling objective + FIX `grind_weak_members` ace-only fallback
   (campaign.py:5269-5273) so the BENCH levels; raise party target toward 6 + add `prep_for_e4()` (level whole
   party FLOOR to `level_milestones["E4"]`=55 before Indigo); add box/replace-fodder logic (Tier-1 #15).
2. **MOVE/TM/HM intelligence:** teach right move→right mon; NEVER clobber a signature/super-effective/only-damaging
   move; decline bad level-up learns; assign HMs (Cut/Surf/Strength/Flash/Fly) without ruining movesets; respect
   the 4-slot constraint. (Much exists — `_ensure_move_room`/STAB-aware `_value`/hm_teach.py — audit + harden.)
3. **Lapras-lead-and-heal tactical AI** (battle_agent): when the ace can't hurt the foe and a specialist is the
   type-answer, LEAD/keep the specialist in + funnel healing to IT, not the ace's 1x Cut.
4. **SOUL flow (via plan_note/voice seam ONLY — core persona + firewall OFF-LIMITS):** team-as-FAMILY (names,
   bonds, favorites, feeling on catch/evolve/faint/loss); BOSS ENERGY (shit-talk rivals/leaders, fired-up/
   triumphant, running bits, stew on losses); POSTGAME/ARC awareness (knows where she is, how far she's come).
5. **NATURAL EFFICIENT PLAY + PC management:** sensible routing, deposit/withdraw, buy items, grind with purpose,
   no head-bumping. **WATCHABILITY PACING:** win most fights, brief struggles, keep moving; LOG watchability signals.
6. **LIVE-SHOW STABILITY HARDENING (critical — 2 past streams crashed ~15min in):** audit/harden run.py/voice/stream
   path (audio isolation/default-off, supervisor auto-restart, checkpoint-resume). ⚠️ multi-hour stability can ONLY
   be confirmed by Jonny's SUPERVISED WATCH — build the hardening, NEVER claim it's proven headless.
7. **INSTRUMENTATION (make runs legible):** (a) END-OF-RUN METRICS report — playtime h:m, final 6 (species+levels+
   movesets), badge timeline, #battles/losses/faints, longest grind, per-segment watchable/grindy verdict; (b)
   SAVE-VISIBILITY/HOP-IN — Jonny can list+load+watch any banked checkpoint (confirm the command; pass-2 E4 hop =
   `E4_BOOT=G:/temp/longrun/banked_E4 ../.venv/Scripts/python.exe -u recon_e4.py`); (c) NAV AUDIT — one more full-game
   wedge-risk pass (Indigo→VR gap etc. per E4_TACTICAL_FRONTIER.md); (d) "WHAT HAVEN'T WE THOUGHT OF" pass — surface
   arc-wide gaps that could break/drag/disappoint on a fresh 30hr run BEFORE they hit live.

### THE FINAL-PROOF GATE (the whole point): fresh early state (`states/workshop/og_postopening.state`) → headless
15x look-ahead → she AUTONOMOUSLY catches keepers at their locations, LEVELS her bench to milestones, arrives at
the E4 with a REAL leveled 6 + coverage, and sweeps TACTICALLY to credits — no hand-port/struct-grind (that manual
workaround is what PROVES the builder is broken), no solo-hack, no over-level steamroll. Watchable pace, narrated.

### THE HONEST-REPORTING MANDATE (non-negotiable — overconfidence caused 2 failed streams): for EVERY item above,
label it (a) BUILT + headless-verified, (b) BUILT but needs Jonny's SUPERVISED LIVE WATCH to confirm it's actually
spectacle-grade, or (c) not-yet-built. The soul/narration/pacing/live-stability items are EXPECTED to be bucket (b)
— headless CANNOT prove "watchable/soulful/stable". NEVER tell Jonny the 30hr spectacle is "done"; tell him what's
built, what's headless-proven, and what his own eyes must verify. That honest line is what makes the stream succeed.

---

## 🏆 CREDITS ROLLED (night-shift #1, 2026-07-11) — the E4 TACTICAL half is DONE + VERIFIED e2e.
The leveled Sherpa team (Venusaur L90 / Lapras L72 / Kadabra L58) beat the Elite Four + Champion Gary
TACTICALLY and reached the HALL OF FAME (`G:/temp/longrun/banked_CREDITS`, fresh). Two committed
battle-brain fixes cracked the 12-run whiteout wall (commit 23487e7, `battle_agent.py`, mode-side only):
(1) never sleep-lock a foe we're 2x super-effective on (the `_se_chunk_latch` mis-slept Cloyster, burning
rooms-1-4 heals); (2) when the active is type-disadvantaged AND a super-effective reserve exists, field the
specialist regardless of level lead (Lapras Ice Beam 2x vs the Champion's Pidgeot instead of Venusaur
trading itself on Cut x1) — which also cut rooms-1-4 heal spend so Full Restores survived to Gary.
**Re-verify anytime:** `E4_BOOT=G:/temp/longrun/banked_E4 ../.venv/Scripts/python.exe -u recon_e4.py`
(banked_E4 is a clean whiteout-center strong-team bank → one clean lap → Hall of Fame).

**THIS IS NOT the fresh-GO watchable re-do.** The team that won was hand-grind-built by NS7-14. The REAL
remaining mountain = the autonomous Part-C team-builder in **`pokemon_agent/TEAM_DEPTH_ROOT_FIX.md`** (below):
make a fresh GO build its own leveled 6 and arrive E4-ready. The tactical E4 half is now guaranteed by the
two fixes above — so the final-proof gate just needs the BUILD half. The canonical 2026-07-07 Champion
timeline was NOT touched (this ran on scratch banks). Loop is STOPPED (CREDITS is line 1 of NIGHT_REPORT.md).

---


## 🎯 THE MISSION (2026-07-10 night — START HERE): FIX THE ROOT BUG — she arrives at the E4 with only ~2 usable mons instead of a real leveled 6. READ `pokemon_agent/TEAM_DEPTH_ROOT_FIX.md` FIRST — it holds the full evidence-backed diagnosis + the ranked fix with exact function/file:line targets.

**WHY:** the headline goal is a WATCHABLE autonomous run where, on a fresh GO, she shows up to the Elite Four
with a REAL FULL LEVELED TEAM like a competent trainer. The game is ALREADY beaten canonically (2026-07-07
credits save = the real summit); this is the watchable re-do. Arriving thin is THE bug that makes the showcase
fall apart. The whole tactical E4 chain is already fixed + committed this shift (moves/Ice Beam/Psychic, 4x
switch, type-answer revive, FR-first shop — commits 625b568, 0cb736d, 6772b54); with a real 6 those fixes win.

**ROOT CAUSE (proven):** the TeamPlanner brain plans the right team but its Part-C EXECUTOR is unwired —
`grind_to`/`develop_bench` have ZERO consumers (voice-only), the catch hook self-disables at 3 mons and never
routes to catch locations, real leveling only raises the ACE (party target 3, `grind()` reads the lead only,
the bench-XP switch wedges), and there is NO `prep_for_e4`. KB data (Part A) + `assess()` (Part B) are
VERIFIED-good — the break is pure Part-C wiring in `campaign.py`.

**THE FIX (implement — details/lines in TEAM_DEPTH_ROOT_FIX.md; all wiring, no data work):**
1. Wire `catch_keeper` to route + catch a FULL 6 across the game (un-gate `_plan_wants_prebuild` `pc>2` at
   campaign.py:4355; make `_plan_keeper_target` travel to `act["where"]` via frlg_encounters.json, not just current-map).
2. Wire `grind_to`/`develop_bench` → a real bench-leveling objective; FIX `grind_weak_members`' ace-only
   fallback (campaign.py:5269-5273) so the BENCH levels, not just Venusaur.
3. Raise the party target toward 6 for late gyms + add `prep_for_e4()` reading `level_milestones["E4"]` (55) —
   level the whole party FLOOR (not the ace) before entering Indigo.
4. Add box/replace-fodder logic (Tier-1 #15) so caught keepers replace the L9-14 dead-weight fodder.
5. ALSO fold in the **Lapras-lead-and-heal tactical AI** change (battle_agent): when the ace can't hurt the foe
   and a specialist is the type-answer, LEAD/keep the specialist in and funnel healing to IT, not the ace's 1x Cut.

**THEN — THE FINAL-PROOF GATE (the whole point):** fresh early-game state (e.g.
`states/workshop/og_postopening.state`) → headless 15x look-ahead → confirm from the log + party growth that she
PROACTIVELY catches keepers at their locations, LEVELS the bench to milestones, arrives at the E4 with a REAL
leveled 6 + coverage, and beats the E4 TACTICALLY (won on a genuine team — no solo-hack, no over-level steamroll).
Bounded detours, watchable pace, narrated. Canonical UNTOUCHED. Do NOT hand-port/struct-grind the team — that
manual workaround (recon_grind_bench + recon_port_and_fix) is exactly what proves the autonomous builder is broken.

---

## ✅ NS14 (2026-07-10): OFFENSIVE-UPGRADE SWITCH FIX BREAKS THE LANCE WALL — reached the CHAMPION (room 5) for the first time. Committed (ce5e391). Route 18 bench-grind + auto-delivery RUNNING overnight; remaining wall = bench too FRAIL + Gary's Charizard (a LEVEL problem).
**AT WAKE — CHECK `ns14_deliver_status.txt` FIRST (the auto-delivery may have rolled credits):**
`ns14_deliver.sh` is armed: it WAITS for the Route 18 grind to finish gracefully (writes "DONE." → `banked_GRIND`
final bank is VENUSAUR-LED via `_restore_ace`), then runs `tail_driver.sh` (seafoam→mansion→blaine→giovanni→
victory→e4). It verifies slot0=Venusaur before the tail (the exact NS13 delivery bug: NS13 fired on a mid-pass
weak-led bank → VR lost). It clears the STALE 2026-07-07 banked_CREDITS first so a fresh one = real credits.
- `cat G:/temp/longrun/ns14_deliver_status.txt` — if it says "CREDITS!! banked_CREDITS is FRESH" → **WRITE
  `CREDITS` as LINE 1 of NIGHT_REPORT.md** (stops the loop) + full mountain survey. Confirm banked_CREDITS mtime
  is TODAY. Then promote per the two-timeline law.
- If it says "tail_driver rc=N — NO credits": read `tail_status.txt` for the failed leg. Most likely the VR leg
  (if Venusaur didn't stay slot0 through the seafoam→giovanni legs) or the E4 (if the bench under-leveled). The
  leveled `banked_GRIND` is preserved — promote it, verify Venusaur slot0, and resume the failed leg (NS9 block).
- If deliver is still "armed — waiting" → the grind hasn't finished; check the grind (below) and either wait or
  read `banked_GRIND` levels and run the tail manually once Kadabra+Lapras are ~L46+.

**THE ROUTE 18 BENCH GRIND (`ns14_grind.log`, banks `banked_GRIND` ~150s):**
`recon_grind_bench` is RUNNING from `grind_base_g` (Route 18, map 3,36): `GRIND_SPECIES=64,131` (Kadabra FIRST —
the Agatha specialist — then Lapras, the Gary/Charizard answer), `GRIND_TARGET=48`. Start levels Venusaur L65 /
Lapras L37 / Kadabra L39. Participation-XP switch banks XP on the WEAK mon while Venusaur aces the kills. Check
`banked_GRIND` levels: `../.venv/Scripts/python.exe recon_partydump.py G:/temp/longrun/banked_GRIND/kira_campaign.state`.
- **Route 18 wilds are L23-29 → XP to L40+ mons is SLOW; it may only reach ~L44-46 overnight** (banks forward
  regardless). Whatever it reaches, THEN DELIVER TO THE E4 (below) and re-run recon_e4 — the switch fix is done.
- ⚠️ **The E4-self-grind loop idea was TRIED AND KILLED — it's FUTILE.** In the E4 whiteout-loop, Venusaur hogs
  every KO (→ L71→73) while the frail bench faints before landing kills, so Kadabra/Lapras DON'T level. And an
  over-levelled Venusaur still can't beat Gary's Charizard (Grass 0.25x hard type-wall). Only the participation
  grind levels the bench. (Loop scripts `ns14_e4_loop*.sh` are dead; ignore.)

### ▶ DELIVERY — how the leveled Route-18 bench reaches the E4 (the last mile, mostly-proven legs):
The grind team is pre-VR (badge 6). Run the NS9 tail: seafoam→mansion→blaine→giovanni→victory→e4 (per-leg cmds in
the NS9 block far below; all legs banked OK in NS13 EXCEPT `recon_victory`). **The VR leg failed because LAPRAS
led the Water-Cooltrainer fight (fight#104) and Body-Slammed x1 too slowly.** THE FIX: the grind ends by calling
`_restore_ace()` which moves the highest-level mon (Venusaur) to slot 0 — so the banked grind SHOULD already be
Venusaur-led → Razor Leaf 2x sweeps the Water Cooltrainer → VR clears. **VERIFY Venusaur is slot 0 in banked_GRIND
before the tail** (`recon_partydump`); if not, swap it (a `camp._swap_party_slots(0, <venusaur_slot>)` primitive
exists — or add a one-shot reorder to the tail before recon_victory). Then E4 with the switch fix → the leveled
Kadabra survives Agatha, the leveled Lapras Surfs Gary's Charizard → CREDITS.
- If it reaches the Hall of Fame → **WRITE `CREDITS` as LINE 1 of NIGHT_REPORT.md** + survey. (banked_CREDITS is
  STALE 2026-07-07 — check mtime, not existence.)

### ▶ WHAT NS14 PROVED (the switch fix is a real breakthrough — verified on indigo_reach_g via recon_e4):
The **offensive-upgrade switch** (committed `ce5e391`, `battle_agent._best_switch_slot`) pushed lap 1 from the
prior 47%-at-Lance whiteout to: **cleared Lorelei/Bruno/Agatha → BROKE LANCE (room 4 at 83% lead) → reached room
5, the CHAMPION (Gary) — first time ever with this team.** TRIGGER 2: when the active can only hit RESISTED (best
damaging move ≤0.5x) while a healthy reserve's STAB is SUPER-EFFECTIVE (≥2x), field the specialist (Kadabra's
Psybeam 2x into Agatha's all-Poison line), overriding the level veto (lenient floor lv+15). Plus **anti-churn:
never switch away from a ≥2x attacker** (killed the Venusaur↔Kadabra infinite loop — Ghost hits Psychic 2x so the
disadvantage trigger kept yanking the SE attacker back out). Fail-safe, mode-side battle-brain only.

### ⛔ TWO REMAINING WALLS (both = bench too FRAIL, a LEVEL problem — the switch logic is done):
1. **Kadabra L40 faints clearing Agatha** (its L54 Ghosts hit Psychic 2x). It DOES its job (1 clean switch, KOs
   Poison-types, conserves Venusaur PP) but dies → needs ~L48 to survive as the standing Agatha specialist.
2. **Gary's CHARIZARD** (Fire/Flying) walls a solo Venusaur (Razor Leaf 0.25x, takes Fire 2x back). The answer is
   **Lapras (Surf 2x vs Charizard)** — but Lapras L39 dies earlier in the gauntlet. Needs ~L48 to survive to Gary.
   NOTE: Lapras has **NO ICE MOVE** (moveset [Surf, Body Slam]) — Surf is 2x on Charizard/Aerodactyl, x1 on the
   Dragons. Still the best Gary answer.

### GRIND FACTS (hard-won this shift — don't repeat the dead ends):
- **VR-grind-from-indigo is IMPOSSIBLE** with the current harness: `recon_grind_bench` needs GRASS tiles; Victory
  Road is a CAVE (step-encounters, no grass) → "no_safe_grass". To grind in a cave you'd have to teach the harness
  cave step-encounter pacing (unbuilt).
- **Route 18 (map 3,36, grass L23-29) is the ONLY proven grind spot** — but its lineage (`grind_base_g`) is stuck
  behind the BROKEN VR tail (see below), so its levels can't reach the E4 without fixing the tail.
- **The E4 itself is the best grinder** now that the switch fix makes Kadabra participate: E4 foes are L54-63
  (~10x Route 18 XP), it's past-VR, XP compounds within one recon_e4 process (banks banked_E4 each whiteout). That
  is exactly what `ns14_e4_loop.sh` exploits. This is the highest-EV overnight path — check it first.

### ⛔ THE NS13 OVERNIGHT CHAIN IS A DEAD END (killed this shift — do NOT relaunch it):
`ns13_overnight_chain.sh` waited for Lapras L46 then ran `tail_driver.sh`, whose `recon_victory` leg
**DETERMINISTICALLY LOSES at VR fight#104** (Water Cooltrainer Kingler/Poliwhirl/Tentacruel) — LAPRAS leads that
fight (grind party order) and Body-Slams x1 too slowly, then aborts on post-loss boulder nav. The switch fix does
NOT rescue it (Lapras Body Slam is neutral 1x, not ≤0.5x, so trigger 2 won't field Venusaur's Razor-Leaf-2x).
**To ever use the Route 18 grind path, you must REORDER Venusaur→slot0 before the tail** (Venusaur-led → Razor
Leaf 2x sweeps the Water Cooltrainer). That reorder helper is UNBUILT. Prefer the E4-self-grind loop instead.

### IF THE LOOP DOESN'T CONVERGE — the surgical next lever = LAPRAS-LEADS-GARY reorder:
Reorder the party so Lapras is slot 0 for the Champion room (or the whole E4), so its Surf 2x is fielded actively
vs Gary's Charizard/Gyarados instead of only via the (flaky, post-faint) force_switch. Combined with a few more
bench levels from the loop, that should close Gary → CREDITS. (Party-reorder actuation is the unbuilt piece.)

### MOVESETS (recon_partydump, indigo_reach_g): Venusaur L71=[RazorLeaf 25pp(only STAB), Cut, SleepPowder,
Strength]; Lapras L39=[Surf, Body Slam — NO ICE]; Kadabra L40=[Psybeam 50pw psychic = Agatha answer]; slots 1/2/4
= L9-14 CHAFF (dead weight; a PC-box drop would help but box access is Tier-2 #15, unbuilt).
Re-test cmd after any battle_agent edit: `E4_STATE=indigo_reach_g ../.venv/Scripts/python.exe -u recon_e4.py`.

## ✅ NS13 (2026-07-10): AGATHA WALL BROKEN — E4 pushed rooms 1-4, whiteout at LANCE's AERODACTYL. New wall = TOP-HEAVY TEAM (Venusaur solos; bench too weak/never fielded). Grinding Lapras+Kadabra on Route 23 now.
**WHAT NS13 DID:** NS12's overnight no-EQ VR grind-through SUCCEEDED — banked `banked_VICTORY` = a PAST-VR team
at Indigo (Venusaur L71→74, Kadabra L40, Lapras L39, healed, $13k). Promoted → `indigo_reach_g`. Ran recon_e4
from it: **CLEARED Lorelei + Bruno + AGATHA (the NS9 wall!) + reached LANCE (room #4) with all 6 alive at 47%**,
then **whited out at Lance's AERODACTYL**, reproduced across the whiteout-retry loop until money hit $0. Killed
the loop (was degrading, not converging).
**ROOT (precisely characterized — 3 compounding issues):**
1. **TOP-HEAVY TEAM.** Venusaur L74 is a monster and SOLOS every battle; the bench is Lapras L39 + Kadabra L40
   + L9-14 chaff. The bench never fields a move all run — it only switches in when Venusaur faints (too late,
   at Lance, → OHKO'd). Lapras's Surf/Ice (x2-x4 on Lance's Dragons + Aerodactyl) NEVER gets used.
2. **AERODACTYL accuracy-debuff.** Lance's Aerodactyl Sand-Attacks → Venusaur (solo, can't switch it off) whiffs
   into an accuracy spiral → can't KO it → attrition death even with 5 Full Restores. Sleep Powder locks it but
   wears off (4 turns). PP famine (Razor Leaf runs dry → Cut/Struggle x1) compounds it.
3. **BROKEN in-battle SWITCH** (`fswitch retry N → wedge frame`, the long-standing Tier-1 #5 gap) — so even
   when Lapras is alive, the engine can't deliberately field it vs the Dragons. Same wedge that blocks Agatha.
The E4 SHOP already buys Revives-first (5/3/16 caps) — revives get wasted reviving weak bench into L58 Aerodactyl,
so a shop fix won't crack it. The ONLY real fix = a survivable, deliberately-FIELDED bench.

### ▶▶ AT WAKE — CHECK THE AUTONOMOUS OVERNIGHT CHAIN FIRST (it may have rolled credits):
**`ns13_overnight_chain.sh` is RUNNING** (`ns13_chain.log` + status in `ns13_chain_status.txt`). It waits for the
Route 18 grind to bring **Lapras → L46** (its Surf/Ice = the Lance answer; Kadabra L39-40 already clears Agatha),
then kills the grind and runs **`tail_driver.sh`** = the proven no-EQ chain promote(banked_GRIND→bench_grind_kit)
→ seafoam → mansion → blaine → giovanni → victory(out-levels VR, NS12-proven) → **E4**. Banks `banked_CREDITS`
if credits roll. **AT WAKE:** `cat G:/temp/longrun/ns13_chain_status.txt` and `cat G:/temp/longrun/tail_status.txt`.
- **If `banked_CREDITS` exists / status says CREDITS ROLLED → WRITE `CREDITS` as LINE 1 of NIGHT_REPORT.md**
  (stops the loop) + the full mountain survey. Promote banked_CREDITS to canonical only per the two-timeline law.
- **If the chain died at a leg** (tail_status.txt names the failed leg): promote the last good bank + resume that
  leg's env cmd (NS9 tail block below). If E4 walled at Lance again even with Lapras L46 → the broken fswitch is
  the wall; do the LAPRAS-LEADS-E4 reorder (slot-0 swap pre-E4) so Surf/Ice is fielded actively vs the Dragons.
- **If the grind STALLED before L46:** Route 18 caps XP for high-level mons; the chain proceeds anyway at ~30min
  no-progress. A stronger grind spot (Victory Road cave L36-46, or solve Route 23's Surf-gated grass) is the
  unbuilt capability for pushing a bench past ~L45 efficiently.

### ▶ FRONTIER = grind Lapras (+Kadabra) so the bench SURVIVES Lance, then re-run the tail → E4 (the chain does this).
**⚠️ Route 23 grind (GRIND_MAP=3,42 from indigo_reach_g) WEDGES** — the team boots at R23 north edge (12,0) and
can't path south to grass (gated/watery, needs Surf/Waterfall nav the traveler lacks). Do NOT retry it blind.
**OVERNIGHT GRIND RUNNING (NS13) = the PROVEN Route 18 spot:** `ns13_grind_r18.log` — `GRIND_STATE=grind_base_g
GRIND_MAP=3,36 GRIND_DIR=west GRIND_SPECIES=131,64 (Lapras FIRST, then Kadabra) GRIND_TARGET=48`. `grind_base_g`
= promoted from NS12's `banked_GRIND` (Venusaur L65 + Lapras L37 + Kadabra L39, positioned AT Route 18 — nav
PROVEN, verified battling + participation-XP switch in the first 30s). Banks `banked_GRIND` every ~150s.
**CAVEAT:** Route 18 wilds are L23-29 → participation XP to L37-39 mons is SLOW; Lapras 37→48 may not finish
overnight. Whatever it reaches banks forward. NS12 got Kadabra→39 here; this run does Lapras first.
**CHECK IT FIRST at wake:** read `banked_GRIND` roster (`../.venv/Scripts/python.exe -c "import json;d=json.load(
open('G:/temp/longrun/banked_GRIND/journey_core.json'));[print(r['species'],r['level']) for r in d['roster']]"`).
Then **re-run the full tail** to get the leveled team back to Indigo (grind_base_g is at Route 18, badge 6 — the
tail re-badges to 8 and clears VR):
```
python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit    # leveled base
# then the NS9 tail: seafoam -> mansion -> blaine -> giovanni -> victory -> e4 (commands in the NS9 block below)
```
When it reaches Indigo → `E4_STATE=<leveled_indigo> recon_e4.py` → a leveled Lapras/Kadabra now SURVIVES the
switch-in at Lance (L48 Lapras Surf 2HKOs Aerodactyl + tanks its hit, vs the L39 chaff that got OHKO'd) → should
clear Lance → Champion → CREDITS.
**IF it STILL walls at Aerodactyl:** the broken in-battle SWITCH is the culprit (Venusaur solos, bench only
fields on faint). Surgical fix = reorder the party so LAPRAS leads the E4 (slot-0 swap pre-E4), so its Surf/Ice
is fielded actively vs Lance's Dragons, bypassing the broken fswitch entirely.
**BANKS:** `indigo_reach_g` (past-VR team at Indigo, Agatha-broken, Lance-reaching — the NS13 advance, GOOD but
bench too weak for Lance) + `grind_base_g`/`banked_GRIND` (Route 18 grind base, leveling) + `giovanni_kit_g`
(badge8) all clean. `banked_VICTORY`/`banked_E4` are temp ratchets. Canonical Champion bank UNTOUCHED.

---
## (superseded) NS13 pre-run plan — kept for the promote/tail command reference:
**Promote:** `python promote_to_workshop.py G:/temp/longrun/banked_VICTORY indigo_reach_g` then
`E4_STATE=indigo_reach_g ../.venv/Scripts/python.exe -u recon_e4.py`. E4 auto-shops Revive/Full Heal/Full Restore.

## ⛔ NS12 WALL (superseded by NS13 breakthrough above — kept as fallback): bench_grind_kit lineage's Venusaur is TOO WEAK for VR/Gary (no EQ) + the EQ teach is BROKEN on this save. TWO clean paths below.
**WHAT NS12 DID:** grind finished **Kadabra L39** (Route 18 capped, above NS9's L38 floor). Tail auto-ran
seafoam→mansion→blaine→giovanni (ALL banked OK, fast). Then **victory WIPED at VR fight#104** (Water Cooltrainer
Kingler/Poliwhirl/Tentacruel) — reproduced 4×, DETERMINISTIC, not variance.
**ROOT (two compounding bugs, both now understood):**
1. The bench-grind Venusaur = `[RazorLeaf 75, Cut 15, SleepPowder 79, Strength 70]` — a THIN battle set (Cut+Strength
   are near-useless HM moves). Its only real offense is Razor Leaf. NS9 passed VR because ITS Venusaur had **EQ +
   Razor Leaf** (NS9's lineage had Secret Power 290 to forget, keeping Razor Leaf). This lineage never got EQ.
2. The old EQ-teach forgot Razor Leaf (no 290 to drop → fell to slot 0). **FIXED + COMMITTED (3964571):** forget by
   CONTENT (protect RazorLeaf 75 + Strength 70; prefer dropping Cut 15) + gated behind `TEACH_EQ` (default OFF) +
   blind `_forget_goto`. **BUT the teach ACTUATION is deterministically broken on giovanni_kit_g:** the TM case
   re-sorts TM26 from row 13→8 and the selection never reaches the make-room dialogue → "NOT taught" at 6.3s. My
   forget-nav fix did NOT change it (failure is at case-SELECTION, not forget). So EQ can't be taught on this save
   without a real teach-actuation fix (frame-grab the open TM case to find TM26's TRUE row/scroll offset).
**Without EQ:** the team loses to **Gary** (Route 23, 2 losses) → enters VR at ~14% HP → whiteout-loops on VR 1F
(barrier ratchets open but it can't survive to the exit). Coverage exists across the team (Razor Leaf x2 vs Water,
Kadabra L39 Psychic, Lapras Surf/Ice) — the killers are HP ATTRITION + the recurring **fswitch wedge** (can't switch
to the right matchup mon mid-battle; "fswitch retry N → wedge frame"). Same wedge blocks Agatha (see NS9 memory).

### ▶ MORNING — two paths to credits. **PATH B is cleaner (skips VR + the broken teach entirely).**
**PATH B (RECOMMENDED): grind NS9's `indigo_reach_kit` Kadabra, then E4.** `states/workshop/indigo_reach_kit.state`
(banked 15:50, NS9) is ALREADY PAST Victory Road, at Indigo, with the STRONG Venusaur (EQ + Razor Leaf) + Lapras L37.
Its ONLY gap was **Kadabra L31** (Agatha PP-famine). So: grind THAT save's Kadabra to ~L42 (from a post-VR-safe spot —
Route 23 grass just S of Indigo, or Route 22; recon_grind_bench with the right GRIND_MAP — VERIFY nav first, it was
only proven on Route 18 map 3,36), bank, then `E4_STATE=<leveled_indigo> recon_e4.py` → likely CREDITS. This uses a
PROVEN-past-VR strong team and never touches VR or the broken teach. The one unknown = grind-nav from Indigo.
**PATH A (harder): make the bench_grind_kit lineage clear VR.** Either (a) FIX the EQ-teach case-selection actuation
(frame-grab `G:/temp/longrun/victory_probe` or a fresh grab of the open TM case; TM26 true row after sort) then run
`TEACH_EQ=1 VICTORY_STATE=giovanni_kit_g recon_victory.py`; or (b) over-level giovanni_kit_g's Kadabra/Lapras to
~L48+ so raw stats brute VR without EQ (needs a stronger grind spot than Route 18's L39 cap).
**OVERNIGHT BET RUNNING:** `ns12_vr_grindthru.log` — a no-EQ recon_victory from giovanni_kit_g (3600s deadline).
Every VR fight levels the team; it MAY out-level VR 1F and reach Indigo (banks `banked_VICTORY`→indigo). CHECK IT
FIRST at wake: if `banked_VICTORY` exists + log shows "Indigo", promote it → run E4. If it failed, XP was lost on the
fresh reboot (recon_victory reboots from giovanni_kit_g each launch) — go PATH B.
**giovanni_kit_g (badge8, Kadabra L39) + indigo_reach_kit (NS9, past-VR strong-Venusaur) are both banked & good.**
**Carried from NS10:** (1) `fix(victory)` EQ teach now targets Venusaur BY SPECIES (was hardcoded slot 0
— on the kit line that wasted the TM and could overwrite Kadabra's PSYCHIC, the Agatha answer). Committed
bd8777d. (2) `G:/temp/longrun/tail_driver.sh` — unattended chain: promote GRIND→bench_grind_kit → seafoam →
mansion → blaine → giovanni → victory → e4, stops on first nonzero exit, banks CREDITS if it rolls. Launch:
`bash G:/temp/longrun/tail_driver.sh` (status → `G:/temp/longrun/tail_status.txt`). **RESUME:** if grind is
dead, restart it (cmd below); when Kadabra≥L38, kill grind (`taskkill //F //IM python.exe //T`), then run
the tail driver. If tail died mid-leg, read tail_status.txt for the failed leg + promote the last good bank
and resume from that leg's env cmd (below). Everything else in this file is the validated NS9 playbook.

## ⛏️ NS9 RESULT: whole pipeline VALIDATED e2e to the E4 — true wall PINPOINTED at AGATHA (under-level PP famine). Grinding Kadabra now → re-run the (all-fixed) tail → CREDITS.

**THE NS9 BREAKTHROUGH:** ran the FULL validation sweep with the leveled kit team and it went the
distance — re-badge tail → Victory Road CLEARED → Indigo → E4 rooms 1-2 (Lorelei + Bruno) BEATEN at full
health. The E4 wall is now PRECISELY located and characterized (no longer blind): **AGATHA (room 3) = PP
FAMINE + menu white-box wedge because the bench is under-leveled** (Kadabra L31 / Lapras L37 can't KO
Agatha's L54-56 Ghosts fast enough — damaging PP runs dry across the gauntlet, then the action-menu
impostor jams switches → anti-wedge abort). FIX = grind the bench higher, esp. KADABRA (the Psychic answer
to Agatha). **Grind RUNNING now** (Kadabra-priority) and banking; then re-run the tail (all its blockers
are FIXED this shift) → E4 should push past Agatha.

**3 FIXES COMMITTED THIS SHIFT (all verified e2e):**
- `recon_seafoam` OFF-ROUTE START: routes a grind-spot start (Route 18) to Fuchsia via the general traveler.
- `recon_seafoam` PRE-CROSSING HEAL: kills the depleted-PP WHITEOUT (grind-bank starts had 0-PP Lapras →
  wiped on R20 wilds → blacked out = the (11,5) "west crossing never fired" wedge; frame-grab confirmed).
- `recon_victory` EVOLUTION-BOX DRAIN: `wedge()` now raw-presses B (ungated by dd_box) — a mid-VR
  Abra→Kadabra evolution box JAMMED overworld nav (dd_box doesn't flag it); this unblocked VR 2F→3F.
- NEW helper `promote_to_workshop.py <banked_dir> <basename>` chains banked_<X> (bare sidecars) → workshop
  kit fixture (prefixed sidecars) between tail legs.

**BANKED FIXTURE CHAIN (states/workshop, each verified e2e, badges/levels rising):** `bench_grind_kit`
(badges=6, grind base, Lapras L37) → `cinnabar_kit_g` → `secretkey_kit_g` → `blaine_kit_g` (badge 7) →
`giovanni_kit_g` (badge 8, leveled team) → `indigo_reach_kit` (at Indigo, healed, $21k → shopped down).

### ▶ THE PLAN — continue the grind, then re-run the tail (now trivial, all-fixed) → E4 → CREDITS.
**1. GRIND is RUNNING** (`G:/temp/longrun/ns9_grind_kadabra.log`, 600-min budget, banks `banked_GRIND`
every ~150s). KADABRA-priority: Abra→Kadabra already evolved (L10→L19 in pass 1) → climbing to L42, then
Lapras L37→42. **Continue/restart if dead** (from `pokemon_agent/`):
```
GRIND_STATE=bench_grind_kit.state GRIND_TARGET=42 GRIND_SPECIES=63,64,131 GRIND_MAP=3,36 GRIND_DIR=west \
  GRIND_MIN=600 GRIND_PROBE_S=150 ../.venv/Scripts/python.exe -u recon_grind_bench.py > G:/temp/longrun/nsX_grind.log 2>&1 &
```
Promote after meaningful banks: `python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit`.
Rate at Route 18 (L23-29 wilds) SLOWS as the bench out-levels them; Kadabra L19→42 is the long pole. If it
STALLS (grind() marks a species stalled), Route 18 may be too weak for L40+ — but participation XP banks
regardless of wild level, so it should keep creeping. **TARGET can drop to ~38** if the grind is too slow:
Kadabra L38 Psychic still 2× OHKO-range on Agatha's Ghosts; re-validate the E4 at whatever level lands.

**2. RE-RUN THE TAIL** when Kadabra ≈ L40+ (Lapras L37 is already enough for Lorelei). Each leg ~90s, all
blockers FIXED. From `pokemon_agent/`, promote between legs (bank dir names in parens):
```
python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit    # leveled base
SEAFOAM_STATE=bench_grind_kit.state  ../.venv/Scripts/python.exe -u recon_seafoam.py  > G:/temp/longrun/x_seafoam.log 2>&1   # -> banked_CINNABAR
python promote_to_workshop.py G:/temp/longrun/banked_CINNABAR cinnabar_kit_g
MANSION_STATE=cinnabar_kit_g         ../.venv/Scripts/python.exe -u recon_mansion.py  > G:/temp/longrun/x_mansion.log 2>&1   # -> banked_SECRETKEY
python promote_to_workshop.py G:/temp/longrun/banked_SECRETKEY secretkey_kit_g
BLAINE_STATE=secretkey_kit_g         ../.venv/Scripts/python.exe -u recon_blaine.py   > G:/temp/longrun/x_blaine.log 2>&1    # -> banked_BLAINE (badge7)
python promote_to_workshop.py G:/temp/longrun/banked_BLAINE blaine_kit_g
GIOVANNI_STATE=blaine_kit_g          ../.venv/Scripts/python.exe -u recon_giovanni.py > G:/temp/longrun/x_giovanni.log 2>&1  # -> banked_GIOVANNI (badge8)
python promote_to_workshop.py G:/temp/longrun/banked_GIOVANNI giovanni_kit_g
VICTORY_STATE=giovanni_kit_g         ../.venv/Scripts/python.exe -u recon_victory.py  > G:/temp/longrun/x_victory.log 2>&1   # -> banked_VICTORY (Indigo); RESUME_STAGE=1 to ratchet a mid-VR wedge
python promote_to_workshop.py G:/temp/longrun/banked_VICTORY indigo_reach_kit
E4_STATE=indigo_reach_kit            ../.venv/Scripts/python.exe -u recon_e4.py       > G:/temp/longrun/x_e4.log 2>&1        # -> CREDITS or the next blocker
```
**IF CREDITS ROLL:** write `CREDITS` as LINE 1 of NIGHT_REPORT.md (stops the loop) + full mountain survey.

### ⚠️ E4 SPECIFICS (from the NS9 run — read before the E4 attempt):
- **Rooms 1-2 (Lorelei, Bruno) already cleared** at full health with the L37 team. The wall is room 3 Agatha.
- **PP FAMINE is the killer:** the gauntlet is 5 rooms with no heal between; damaging PP depletes. Higher
  levels = fewer turns/KO = less PP burned. Kadabra L42 Psychic should 1-2-shot Agatha's Ghosts (Gengar/
  Haunter/Arbok/Golbat), ending fights before famine. She has NO Ethers (famine is unrecoverable mid-fight).
- **Menu white-box wedge:** the E4 rooms trigger the "action-menu impostor (white box, DEAD cursor)" +
  "famine switch did not confirm" actuation jam (known E4 livelock family). Levels mask it (fewer menu
  windows); if it still bites post-grind, that's the next fix (see [[pokemon-e4-livelock-family-killed]]).
- **EQ=NO gotcha:** recon_victory's Phase0 EQ-teach targets slot 0, but the grind leaves Lapras/Kadabra
  leading (Venusaur is slot 5) → EQ taught to the wrong mon and FAILED. Venusaur has no Earthquake. Minor
  (Razor Leaf/Surf/Psychic carry the E4). To fix: swap Venusaur to slot 0 before the tail, or make the EQ
  teach target Venusaur by species (3).
- E4 shopping: recon_e4 auto-buys Full Restore/Hyper/Revive at the Indigo Center (money-aware, spent to ~$1k).

### 🏁 ALL 8 BADGES on the kit line; VR CLEARED; E4 rooms 1-2 down. Only Agatha+ (a LEVEL problem) remains.
Memory: [[pokemon-nightshift9-e4-validation-agatha-wall]] · [[pokemon-nightshift7-bench-grind-nav-island]] ·
[[pokemon-e4-gauntlet-truths]] · [[pokemon-e4-livelock-family-killed]].

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — venv = 2-PID shim = ONE logical run; kill
`taskkill //F //IM python.exe //T`). Flags module = `field_moves` (fm.read_flag). Bank dir names differ
from the env var (seafoam→banked_CINNABAR, mansion→banked_SECRETKEY, blaine→banked_BLAINE, giovanni→
banked_GIOVANNI, victory→banked_VICTORY, grind→banked_GRIND). recon_victory RESUME_STAGE=1 ratchets a
mid-VR wedge from its own stage_victory bank.

WATCH STATUS: canonical Champion bank CLEAN + untouched (NS12 only edited workshop staging fixtures + temp
banks — the canonical timeline is safe). Sherpa frontier = get a PAST-VR team with a leveled-enough Kadabra
to the E4 (PATH B: grind NS9's indigo_reach_kit Kadabra L31→~42, then recon_e4). The bench_grind_kit lineage
(badge8, Kadabra L39) is banked but CANNOT clear VR without EQ, and the EQ teach is broken on that save — so
prefer PATH B. Overnight no-EQ VR grind-through running (self-terminates ~1hr; check banked_VICTORY at wake).
Pop-in = `python pokemon_agent/watch.py`.
