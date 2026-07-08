# NEXT_SESSION — THE STANDING NIGHT-TRAIN MANDATE (rewritten 2026-07-08 night shift 9, mid-re-grade)

## ⚡ SHIFT 9 STATE (current)
0. **FULL 15-ARC SWEEP LANDED 04:36 (pre-e2d5f9a code): 10 PASS / 4 WARN / 1 FAIL.**
   Snapshot preserved: `logs\longrun\DESCENT_PREGRADE_full_shift7.md` (the live
   DESCENT_PREGRADE.md gets overwritten by every re-grade — read the snapshot).
1. **e2d5f9a VERIFIED: ROCKTUNNEL re-grade FAIL→PASS** (twedge 128→2, nav 1→0; log
   `descent_regrade_rocktunnel_shift9.log`). Keyboard-wedge class gone. HONESTY NOTE: the
   nickname guard's positive line ("declining with B") was NOT hit this window — she
   didn't take the roof-NPC gift path; guard positive-path = logic-only.
2. **WARN CLASS DIAGNOSED (all 4 = end_nomove_streak≥2) + FIXES BUILT (campaign.py,
   COMPILES+WIRED, re-grade in flight):**
   (a) BENIGN-STILL outcomes (questline_talked/worked_room/step_done/done/passthrough/
   deeper/entered, need_heal, healed_retry) no longer count as SILENT NO-MOVE nor poison
   _dead_moves — they were pruning head_to_gym MID-QUESTLINE (SABRINA/SCOPE class);
   (b) ALL-DEAD PRUNE: when every offered option is a proven dead route, prune them anyway
   — the EMPTY-OPTIONS FLOOR refills with regroup-at-Center (kills the banked_E4 Saffron
   dead-air: 4×no_route travels re-offered ~25 straight decisions);
   (c) plain "no_route" added to the STRUCTURAL dead-route set (else the heal excursion's
   movement resurrected the same doomed travels — the oscillation).
3. **IN FLIGHT: re-grade of banked_SCOPE,SURF_TAUGHT,SABRINA,E4 on the fixed code** — log
   `logs\longrun\descent_regrade_warns_shift9.log` (launched ~04:45, ~10 min). SINGLE-RUN
   LAW: no other emulator recon while it runs. If dead when read fresh: relaunch with
   `$env:DESCENT_ARCS='banked_SCOPE,banked_SURF_TAUGHT,banked_SABRINA,banked_E4'`.
   EXPECT: SABRINA/SCOPE → PASS; E4 → PASS or the regroup floor visible; SURF_TAUGHT may
   still WARN — its head_to_gym returns 'stuck' on Route 19 (0,42) water edge (billed road
   wants Surf legs?) — if it WARNs again, diagnose THAT (real nav gap, not grader noise).
4. **NEXT (in order):** (a) read the WARN re-grade, fix/commit what's left; (b) level-up
   early beat verify — `.venv\Scripts\python.exe -u pokemon_agent\recon_lvlbeat_verify.py`
   (default banked_HM05; arms a real level-up by -1 on the lead's level byte);
   (c) Viridian Gym spin maze (spin_assist un-verified live); (d) if all green: FULL
   15-arc sweep on tonight's code for the clean table → ranked spot-watch list for Jonny.

## SHIFT 8 STATE (background — landed + verified this shift)
0. **ROCKTUNNEL (7,4) MYSTERY = SOLVED, FIX COMMITTED e2d5f9a (COMPILES+WIRED, verify
   owed):** the "who blocks (7,4)?" probe was never needed — descent_full_shift5.log tick
   14 shows the TEA sweep talked the Celadon Mansion ROOF NPC → gift EEVEE → "give a
   nickname?" Yes/No defaults YES → the drain's A opened the naming KEYBOARD (full-screen,
   invisible to box_open's bottom band) → 300 blind A-presses → timeout with the field
   locked FOREVER. The (7,4) "blocker" was a ghost — presses eaten by the keyboard. THE
   NICKNAME-KEYBOARD CLASS (Lapras lesson) existed in battle_agent + starter flow but NOT
   in dialogue_drive, the primitive every unattended npc-talk uses. Fixes (all in the one
   drain): (1) "nickname" line → advance with B never A; (2) blind-lock (≥8 no-box/
   no-control iters) → B + periodic START+A keyboard-class escape (START→OK, A confirm,
   empty name = clean decline); (3) _close_box last-resort START+A + timeout path runs
   _close_box (never hands back a locked field); (4) BATTLE GATE at the primitive — the
   live shift-7 sweep showed "[dlg engage] max_steps=300": an engage intro box leads INTO
   a battle, drain now returns "in_battle" the moment one opens (erika_run2 class, was
   entry-only in _drain_overworld).
1. **IN FLIGHT: FULL 15-arc sweep (launched by shift 7, 04:05)** — log
   `logs\longrun\descent_full_shift7.log`, lands ~04:40; regenerates
   `pokemon_agent/DESCENT_PREGRADE.md` (full table incl. twedge). NOTE: it runs
   PRE-e2d5f9a code, so ROCKTUNNEL/engage wedges in ITS table are EXPECTED — the fix
   verify is the re-grade after. SINGLE-RUN LAW: no other emulator recon while it runs.
   If this file is read fresh and the sweep is dead: relaunch
   `.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120` first.
2. **NEXT (in order):** (a) read the sweep table; (b) VERIFY e2d5f9a: re-grade ROCKTUNNEL
   (`$env:DESCENT_ARCS='banked_ROCKTUNNEL'; .venv\Scripts\python.exe -u
   pokemon_agent\recon_descent_grade.py 120`) — expect the (7,4) storm gone + the Eevee
   decline in the log ("nickname prompt — declining with B"); (c) fix whatever else the
   sweep flags (diagnose → fix → re-grade that arc → commit); (d) level-up early beat
   verify — `recon_lvlbeat_verify.py` is BUILT+COMMITTED (e2d5f9a), run:
   `.venv\Scripts\python.exe -u pokemon_agent\recon_lvlbeat_verify.py` (default
   banked_HM05; arms a real level-up by -1 on the lead's level byte, exp untouched);
   (e) Viridian Gym spin maze (spin_assist un-verified live).

## SHIFT 6 STATE (background — all landed + committed)
1. **ROAD-ANCHOR PARKING — COMMITTED 7c2025b (VERIFIED in re-grade #1):** `campaign.
   _next_step_rideable` wraps `world.next_step` with a travel-grade BFS reachability
   pre-check from her FEET before any warp ride (NPC-blind, fail-OPEN on read flakes);
   unreachable hop → parked for THIS query, route re-asked (`world.warp_tiles` rides any
   reachable door of a multi-door hop); nothing rideable → honest `no_gym_route` (the
   structural outcome that stops forward-drive's re-frame). Wired at all three warp-ride
   sites (_road_step, head_to_gym warp-route, questline anchor-first). Re-grade #1 proof:
   tick-1 head_to_gym parked cleanly, B4F wedges 271→17.
2. **SEALED-BY-A-GUARD EXIT CHAIN — COMMITTED aba03c5 (VERIFIED e2e):** re-grade #1 still
   FAILED on twedge=223 — all in `_exit_to_overworld` on hideout B1F. Probes
   (recon_b1f_probe/probe2, standing) + pret: **RocketHideout_B1F holds a TRAINER-GATED
   BARRIER at (20-21,19-21)** — coll-3 wall until TRAINER_TEAM_ROCKET_GRUNT_12 (the guard
   at (21,27)) falls, then setmetatile opens the floor. THE GENERAL LESSON: scripts change
   the FLOOR, not just who's here — a sealed walk-region's opener can be a PERSON. Fixes:
   (a) SEALED-DOOR SKIP — flood pre-check before any exit-door leg, skipped doors NOT
   consumed (retried the moment the floor unseals); (b) PERSISTENT ELEVATOR ROW CURSOR
   (bag TRUE-row law) + ride all remaining rows while aboard; (c) `_engage_exit_guard` —
   no reachable door → talk/fight nearest reachable object event (live coords, once per
   map) → re-plan on the fresh grid. VERIFY (recon_exit_verify.py, standing): B4F → car →
   B1F → guard fight WON → barrier open → Game Corner → street, ONE tick, 38s, 4 wedges.
3. **NEXT (in order):** (a) read SCOPE re-grade #2 (log → descent_regrade_scope2_shift6.log,
   expect PASS/WARN); (b) FULL 15-arc sweep on tonight's code
   (`.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120`, ~35 min) →
   DESCENT_PREGRADE.md regenerates with the twedge column → fix what it flags; (c)
   ROCKTUNNEL WARN (nav=1): the TEA questline climbs Celadon Mansion to map (10,11), ends
   wedged at (7,4) — travel reads "path blocked at (7,4), blocker NPC tile=None, npcs
   nearby=[(3,5)]" and can't take the FIRST STEP; the exit-loop fans doomed doors
   (tonight's door-skip will cheapen it, but the blocked-first-step class is unfixed —
   needs a probe: who blocks (7,4)? wanderer body the template read misses?); (d) level-up
   early beat verify (printed-events run); (e) Viridian Gym spin maze (spin_assist
   un-verified).

## SHIFT 5 STATE (background — all landed + committed)
Shift 4 CLOSED with: water-aware travel COMMITTED (e9f1715 — surf-capable Grid/BFS layer +
shoreline mount; the Pallet→R21 shore-bonk class), questline FENCED-BEND passthrough + the gym
ping-pong breaker COMMITTED (1cc70e6), and the 4-arc re-grade LANDED: **ROCKTUNNEL / SABRINA /
BLAINE = PASS; SILPH = WARN (nav=4)**. A shift-end SILPH re-grade (descent_regrade_shift4b.log,
killed at handover) shows the WARN's true face: inside Saffron Gym (14,3) EVERY walk-route
reads no_route — it's the TELEPORT-PAD MAZE (warp-partitioned interior); the gym handler
travel-wedges ×4 (the nav tripwires), false-clears juniors off the one reachable object, and
A-mashes Sabrina from 11 tiles away. The fix is known + strike-proven: recon_sabrina.py's
`pad_plan` (runtime pad-graph: same-map warp events = pads, dest_warp_id = landing; flood-fill
walk-regions elevation-strict, meta-BFS with pad rides as edges) + `ride_pad` (long-hold mount).
**SHIFT 5 LANDED (all committed; verifies chained):**
1. **PAD-ROUTER PORT — COMMITTED 2a13802 (COMPILES+WIRED):** `pad_nav.py` (plan/ride/walk/
   goto_region, strike-verbatim; region flood STRICT to seed elevation — edge_open's
   0-as-transition would weld the x29 void strip to the rooms) + campaign: beat_gym arms the
   router when the interior has same-map warps; `_gym_move` is THE single interior mover
   (travel is warp-blind — its paths RIDE pads mid-route, so armed gyms never use it);
   junior engagement, leader-advance, leader approach, post-badge walk-out all pad-aware.
2. **F-7(c) slice 2 — COMMITTED 94157e6 (COMPILES+WIRED):** LEVEL-UP EARLY BEAT fires inside
   the post-faint drain (any party slot; level byte flips while the grew-to box is up);
   play_live's post-drain emit is now the deduped FALLBACK (battle_agent.LEVELUP_EMITTED).
3. **ELEVATOR PRIMITIVE + SPIN-ASSIST — COMMITTED 2eb2b05 (COMPILES+WIRED):** the sweep caught
   banked_SCOPE burning 412 identical travel wedges. DIAGNOSIS (recon_spin_probe.py, standing):
   the spawn is hideout B4F's boss corridor, walk-sealed BY DESIGN — exits = the ELEVATOR
   (all warps dest (127,127); the exit loop entered the car and walked back out forever) then
   SPIN floors above. NOT the elevation law (probe: zero spin tiles on B4F, no-route with law
   on/off). Wire: `elevator_nav.py` (is_car + panel ride, ported from recon_hideout.ride;
   rows unbilled — self-correct on landing) hooked into `_exit_to_overworld` top; travel gains
   `spin_assist` (field_clear injection pattern; fires ONCE per wedged leg on maps with the
   new `Grid.spin` set) → campaign._spin_assist runs spin_nav's glide crosser.
**VERIFY RESULTS (shift 5 close):**
1. **FULL SWEEP (pre-tonight code): 13 PASS / 2 WARN** — SILPH (nav=4) + ROCKTUNNEL (nav=1).
   Snapshot: `logs\longrun\DESCENT_PREGRADE_full_shift5.md` (+ .json sibling).
2. **SILPH re-grade = PASS — AND SHE WON BADGE 6 (Marsh) AUTONOMOUSLY in the window** (pad
   plans 3 rides deep, juniors + Sabrina beaten, nav 4→0). The pad router is VERIFIED
   end-to-end. Log: `descent_regrade_silph_shift5.log`.
3. **SCOPE re-grades (rounds 1-3, the hideout-B4F exit rope):** round 1 = elevator RODE
   (panel+multichoice verified live) but blind rows looped B4F↔B1F → built the LANDING
   ORACLE (SB1+0x14 dynamicWarp reads the selected floor BEFORE stepping out; commit
   5f0a45c). Round 2 = oracle PROVED the mechanism: D-pad presses are EATEN before the
   multichoice is input-ready → select-A hits the default row every time → MENU SETTLE fix
   (90f settle + spaced downs; commit 262e609). **Round 3 RESULT: SHE EXITED THE HIDEOUT** —
   elevator rode to B1F (row 2 retargeted post-settle; rows 0-1 still eaten sometimes — the
   oracle absorbs it), then late-window the exit chain ran `(1,42) → (10,14) Game Corner →
   (3,6) STREET`. Grade still **FAIL on twedge=271**: while the exit machinery worked,
   head_to_gym's road steering hammered the walk-unreachable (11,15) stairs EVERY TICK.
   **THE NEXT ARC-FIX (first thing next shift): structural parking for unreachable same-map
   road anchors** — when travel returns no_route for the same road-anchor tile N times on one
   map, park that anchor (the ping-pong-breaker class) so the oracle picks something else;
   log: `logs\longrun\descent_regrade_scope3_shift5.log`. Then re-grade SCOPE → expect
   PASS/WARN, then the FULL 15-arc sweep on tonight's code (twedge column everywhere).
4. **Grader upgraded (d23dd01):** travel wedges now graded (>20 WARN / >100 FAIL) — SCOPE's
   412-bonk PASS blind spot closed. DESCENT_PREGRADE.md currently holds only the last
   re-graded arc row — REGENERATE the full 15-arc table (now with the twedge column) once
   SCOPE is green: `.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120`.
5. **ROCKTUNNEL WARN characterized (not yet fixed):** the badge-3→Erika questline works the
   Celadon pass-through chain, ends hammering a blocked doorway (10,11)@(7,4)
   (`questline_wrong_building` ×2 + npc_block wedge storm) — the FENCED-BEND rough edge.
   Next arc-fix after SCOPE.
NOTE: venv python is a shim — TWO PIDs per launch; never taskkill your own run.
Known general gaps still open: VIRIDIAN GYM spin maze (spin_assist may now cover it —
un-verified); level-up early beat WIRED not verified (grade-harness on_event doesn't print;
needs a printed-events run, e.g. recon_winbeat_verify pattern); evolution early beat unbuilt
(post-battle cutscene, own seam).

Paste this to the fresh session / this IS the night-train's standing order. Employment terms
in force: loop until done, bank per phase, honest three-state surveys, stop ONLY for a
needs-eyes item on the ledger below or ~85% context (then rewrite THIS file frontier-first
and hand off). Rule 16 covers everything else — decide and execute.

## BURN DISCIPLINE (CEO directive — density is mandatory)
1. **THE VALUE LINE.** Every shift survey ends: `this shift bought: [deliverable] —
   [shipped/verified/landed]`. Can't write a true one → STOP, surface needs-eyes.
2. **BOUNDED RECON.** Build-list in ≤1 shift, then BUILD.
3. **DEPTH OVER SPREAD.** 3 things fully > 10 things partially.
4. **NO IDLE-GRIND.** Blocked on Jonny → close early and cheap.

## ⛰️ THE DESCENT — HUMANITY-PASS DOCTRINE (CEO, after 3 watched soul-on runs; the showcase gate)
The Sherpa run proved the mountain climbs; the showcase requires it climbed like a HUMAN.
**HUMAN-PACE PLAY:** the rope (Sherpa knowledge) is her INSTINCT, not her GPS. In showcase/
soul mode she plays like a person — reads signs, talks to NPCs (mom!), notices rooms, fishes
if she wants, lingers, voices detours. **Speedrun behavior (beelines, skipped dialogue,
robotic efficiency) is a DEFECT in this mode even when mechanically optimal.** Endearing
beats efficient — the supreme law governs. The descent (F-11) is the quality gate in front
of the Kira-timeline GO; phases E/G/H/I/K remain behind it.

---

## FRONTIER (night shift 2, 2026-07-08 — in flight)
**Shift 2 landed so far (all committed):**
- **F-8 LOCATION CONTEXT BLOCK — WIRED+VERIFIED** (the decision-side half, done): every soul
  tick and every want ask now LEADS with a grounded where-am-I line (`campaign._location_block`)
  — indoor/outdoor truth from map group + travel's new disasm-grounded `G1_CAVES`/`G1_OUTDOOR`
  tables (group 1 is MIXED: the Hall of Fame read "underground in a real cave" until tonight —
  same class as Oak's-lab-as-cave), only live-confirmed traits asserted, and an UNKNOWN place
  explicitly instructs curiosity-not-assertion. `_PLACE_NAMES` extended to FULL-ROPE coverage
  from pret map_groups.json cross-checked live (routes 7-25 incl. the Route 21 north/south
  split at (3,39)/(3,40), Mt Moon, S.S. Anne, Underground Path, Diglett's, Hideout, Silph,
  Mansion, Safari, Cerulean Cave, Rock Tunnel, Seafoam, Tower, Power Plant — Route 12 read
  "an unfamiliar area" before). Standing verify: `recon_locblock_verify.py` (8 banked spawns,
  ALL PASS). travel `_muse_seed` uses the same tables (Lorelei's room no longer seeds cave
  gloom; Safari/forest/deck muse as open air).
- **F-9 VERBAL-TIC GOVERNOR — WIRED (live feel needs a watched take):** `repetition_guard`
  gains `overused_phrases`/`tic_ban_block` (a tic = distinctive 3-5-word phrase across ≥3
  distinct lines of a 40-line window; the old line-level guard couldn't see it). bot.py
  Pokémon-gated seams (CORE touch, flagged, additive): `_execute_interjection` records
  Pokémon lines (memory_query marker), `_pokemon_react` folds the hard-ban into the ask —
  deliberately NO self-aware escape ("I almost said—" is the ban-evasion Jonny flagged).
  Prompt byte-identical when she varies naturally. Unit-verified.
- **F-11 MACHINE PRE-GRADE HARNESS — BUILT (`recon_descent_grade.py`):** runs the REAL loop
  headless off every banked arc spawn (rope order, banked_CREDITS excluded — the mid-ceremony
  grenade), grades LOCOMOTION (watchdog/nav-tripwire/no-move), LIVENESS (voids/abandons/
  crashes), GROUNDING (unnamed-map payload), TEXTURE (decision variety) → PASS/WARN/FAIL +
  ranked riskiest arcs → `pokemon_agent/DESCENT_PREGRADE.md` + `%TEMP%/longrun/
  descent_grade.json`. READ-ONLY on all bundles (persistence no-op'd). **IF THE SWEEP DIDN'T
  FINISH THIS SHIFT: run it —** `.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120`
  (~35 min, 15 arcs) — then fix what it flags, re-grade the flagged arcs, commit, and put the
  ranked list under Jonny's needs-eyes for spot-watch picks.

**NEXT after the sweep (in order):**
1. **Fix what the pre-grade flags** (each FAIL/WARN arc → diagnose → fix → re-grade that arc).
2. **F-7(c) SPECULATIVE PREFETCH — slice 1 BUILT shift 3 (verify in flight):** the CERTAIN-WIN
   EARLY BEAT in battle_agent: when a faint leaves ZERO live mons in gEnemyParty (party HP is
   synced every damage write — pret Cmd_datahpupdate), ONE merged win line fires AT THE FAINT
   (in place of "took it down"; _finish dedups; catch flow + double-faint + switch-in guarded)
   so the ~4s LLM chain runs during the 5-15s victory drain and her "we won!" lands ON the
   victory screen instead of ~10s into the overworld. Verify = `recon_winbeat_verify.py`
   (trainer=Agatha premature-win guard off banked_E4, wild=Route 2 off banked_HM05, log →
   logs/longrun/winbeat_verify.log). If it PASSED and uncommitted: commit. Remaining F-7(c)
   beats (build later, same pattern): level-up/evolution fire post-drain in play_live (late by
   the whole drain); dialogue reader timing unexamined.
3. **F-10 TEXTBOX WARM-UP STALL** — self-diagnoses via the shipped F-7(a) telemetry on the
   next watched take; if a take happens, read the [queue age + POST ms] numbers first.
(F-6 SOCIAL FABRIC shipped shift 2 — commit 3ef4da9.)

## PHASE F — remaining ledger (watched-take items ride the quiet window)
- **F-1 LOCOMOTION:** architecture landed + loop-verified (wander tripwire). Sign-off needs a
  soul-on watched take (bot).
- **F-7:** (a/b) shipped; (c) speculative prefetch open; (d) prefetch A/B needs bot restart +
  2 min of Jonny.
- **F-5 EXIT BAR:** Jonny grades throwaway round 3 once F-1 feel-take is green.
- **F-4 PROTECT list** (self-naming, canon Gary loss, post-loss continuity, grudge arc,
  judged catches) → Phase-H regression list — untouched, keep it that way.

## OTHER PHASES (unchanged, behind the descent gate)
- **E GO BUTTON:** built + rehearsed; cold-open recap on a RESUME untested (all takes fresh).
- **G MAGIC AUDIT:** G-2/G-3 shipped flag-OFF (feel-tests pending); G-4 cohost smoke = needs-eyes.
- **H REGRESSION GUARD:** standing law; F-4's protect list joins it.
- **I ONE-KIRA MODE MATRIX:** I-1 audit banked (MODE_TRANSITION_AUDIT.md); I-2 attention
  director (extend SessionIntensity); I-3 media pacing (dead _PACING table gets a consumer).
- **K THE CLIPPER:** remaining: kira_moment logging (core touch), D3 paths, 007 exit run →
  needs-eyes item 6.

## NEEDS-EYES LEDGER (the ONLY loop-stoppers)
1. **Descent spot-watches** — pick from DESCENT_PREGRADE.md's ranked list when it exists.
2. **Round-3 throwaway grading** (F-5 bar) — after the F-1 feel-take is green.
3. **Prefetch A/B B-side** — bot restart w/ flag + one conversation (2 min of Jonny).
4. **F-9 tic governor feel** — rides any watched take (listen for wallpaper phrases gone).
5. **Cohost smoke session** (G-4 exit, 20 min).
6. **Tri-mode session** (Phase I exit, 15 min).
7. **Clipper output review** (K exit).
8. **Final showtime sign-off** — the Kira-timeline launch is HIS press, always.

## WORKING-TREE LAW
kira/* = Jonny's + approved core work under loud-log law (F-9 touched bot.py +
repetition_guard.py — Pokémon-gated + additive, flagged in the commit). Prune targets (C-4
leftovers): untracked recon_* fleet remainder, repo-root states/, verify_dialogue.py.
Throwaway sandboxes: `go.py --clean-throwaways` + `watch.py --clean`.

---

WATCH STATUS: canonical bank is CLEAN — the TRUE post-game: the Champion at home in Pallet
Town ((4,0)@(4,8), full healthy party — Venusaur L95, Persian, Fearow, Raticate, Ekans,
Lapras — badges 8, player in control; sanctity VALID). She is at home, victory lap ahead
(Cerulean Cave open; her stated want: catch Mewtwo). Pop-in = `python pokemon_agent/watch.py`
→ spawn 'postgame' (or --canonical, safe). Her tick context now opens grounded: "You're at
home — indoors, inside a building."
