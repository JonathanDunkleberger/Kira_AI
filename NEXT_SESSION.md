<!-- ═══ NIGHT-SHIFT 16 (2026-07-10) — ★ S.S. ANNE GARY WALL BEATEN; frontier = SURGE GYM-DOOR CUT TREE ═══
THE WIN: the 12-shift S.S. Anne Gary wall is BEATEN e2e (verified, s16e_surge.log). She now grinds the ace
to Venusaur L32 BEFORE boarding, cabin-skips, WINS Gary first try, and obtains HM01 Cut from the captain.
NEW FRONTIER: BADGE 3 (Lt. Surge) gym ENTRY — she has Cut but can't field-Cut the gym-door tree.

★★ THE ROOT (finally): the strategic FODDER-grind pre-empted the ace prep, AND a stale flag hid it.
   Chain of discovery this shift (each committed fix exposed the next layer):
   1. aff5dbc — After a Gary loss the STRATEGIC path (PICK OUT: battle) dominates the roam tick and, with
      the in-battle switch armed, `_prep_team_target` took the FODDER-floor branch (underlevel_target≈L19,
      field the weak mons). So she ground the bench to L19 while the ACE — the only thing that beats a 4-mon
      rival — stalled at L30, "readiness" crossed on the team FLOOR, she boarded at L30 and lost = livelock.
      FIX: route 4+-mon rival prep to ACE-OVERPOWER.
   2. f1bc0db — Ace-overpower grinding at Cerulean dove down the one-way Route-4 ledge she can't path back up
      (stranded, treadmilled L35→39). FIX: DEFER 4+-mon rival prep to the questline anchor-path; RIVAL-RETURN
      guard stops off-anchor trap-grinds.
   3. Post-loss deadlock: a Gary loss records the S.S. Anne as a SPATIAL WALL that gates the route back to the
      anchor "until stronger" — but she can only get stronger at the anchor she can't route to (s16b STALL).
   4. 6643dd7 + 78d8c4d — Bypass the whole post-loss mess: PREP TO L32 ON THE FIRST APPROACH, at the actual
      BOARDING warp (never lose, never white out).
   5. ★ 57d18c4 — THE BUG that hid the prep for 12 shifts: on the FIRST approach `_rival_won_here` was TRUE
      because the detection loose-matches ANY historical won rival encounter whose place contains "s.s. anne"
      — and the **bill_done_kit fixture carries a won "s.s. anne" encounter in strat memory**. That
      `not _rival_won_here` guard silently skipped the prep in BOTH the 6044 block and the new boarding gate,
      so she boarded at L28 EVERY shift and lost Gary. Dropped it from the boarding gate (standing at the
      HM-Cut boarding warp already proves she hasn't won — no Cut, gym still shut).
   VERIFIED e2e (s16e_surge.log): `BOARDING GATE: ace=L28 -> GRIND FIRST` → Route 6 grind → ivysaur→venusaur
   → `ace=L32 -> board` → `CABIN SWEEP SKIPPED — ace L32 overpowers` → `KIRA obtained HM01 from the CAPTAIN!`
   (reaching the captain REQUIRES beating Gary — so Gary is WON). The 12-shift wall is DEAD.

★★ THE NEW FRONTIER — BADGE 3 SURGE GYM ENTRY (Cut-tree-at-door; she has Cut, can't actuate the field-cut):
   After HM01 she disembarks, returns to Vermilion (Venusaur L39, over-grinded), and beat_gym tries
   `enter_warp(gym.door)` — but the Vermilion gym door is behind a CUT TREE at (19,24) on map (3,5). beat_gym
   returns "stuck" → shift-11 fix runs `_gym_gate_probe(gym)` (which scans for the tree, walks adjacent, and
   `clear_obstacle('cut')` if `can_use('cut')`). can_use SHOULD be True (Cascade badge ✓ + Venusaur knows Cut
   — she used it in battle). BUT the field-cut never actuates (no "auto use_cut"/"TIMBER" log after HM01).
   DIAGNOSIS: after 2 gym-stucks the GYM-INTERIOR PING-PONG BREAKER (campaign.py ~8760) prunes head_to_gym
   STRUCTURALLY on the Vermilion map — she's then stranded at the NORTH of Vermilion (24,0), ~24 tiles from
   the tree at (19,24), oscillating Vermilion↔Route 6. `_gym_gate_probe`/`_obstacle_probe` (campaign.py
   7088/7106) needs her walked ADJACENT to the tree to scan+cut, but the structural-park keeps her away, so
   the cut never fires. Likely ALSO an `_obstacle_probe` reachability issue (the yard fence: the code comments
   at 7098-7103 warn the Grid is optimistic about fences + a distance-first walk drifts to the BEACH).
   NEXT-SHIFT PLAY (self-help arsenal, rule 15): (a) GRAB A FRAME of the Vermilion gym yard (`_gym_gate_probe`
   run with a frame dump) to SEE the tree/fence/door geometry — is she able to stand adjacent to (19,24)?
   (b) make `_gym_gate_probe` fire BEFORE the structural-park prunes head_to_gym (run it on the FIRST stuck,
   or exempt the gym-approach from the ping-pong park when a Cut tree gate is recognized + Cut is owned);
   (c) verify `can_use('cut')` actually returns True post-HM01 (field_moves.py:178) — maybe it checks a
   flag/badge that reads False; (d) if `_obstacle_probe`'s walk-to-adjacent fails, fix the standing-tile BFS
   for the gym yard. Once the tree is cut, she enters → beat_gym clears juniors → Surge → BADGE 3.

   ALSO NOTED (not blocking, cosmetic-ish): she OVER-GRINDS after getting strong — Venusaur L39→41 via the
   proactive-bench + `grind(lead+2)` treadmill while stuck at the gym. Once the gym-entry is unblocked this
   mostly resolves (she stops being stuck); if it persists, suppress proactive-bench/lead+2 while an active
   gym-gate (Cut tree) is unresolved.

VERIFY COMMAND (reproduces the WIN + lands on the gym-door frontier):
   `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x822 .venv/Scripts/python.exe -u
   pokemon_agent/recon_longrun.py bill_done_kit.state 55` -> /g/temp/s17_surge.log. WATCH: `BOARDING GATE:
   ace=L28 -> GRIND FIRST` -> `-> venusaur` -> `ace=L32 -> board` -> `CABIN SWEEP SKIPPED` -> `HM01 from the
   CAPTAIN` (Gary WON) -> disembark -> `couldn't enter the Lt. Surge gym` / `beat_gym stuck` (THE FRONTIER).
   grep: `grep -nE "BOARDING GATE|-> venusaur|SWEEP SKIP|HM01|couldn't enter|beat_gym stuck|auto use_cut|TIMBER|GYM-GATE PROBE|GOAL|OUTCOME" log | grep -v ctx=`.

FIXTURES: bill_done_kit.state at pokemon_agent/states/workshop/ (Ivysaur L26 + frail bench) — rebuild
   `.venv/Scripts/python.exe pokemon_agent/recon_repair_kit.py`. ⚠️ NOTE: this fixture's strat memory carries
   a STALE won "s.s. anne" rival encounter (that's what tripped `_rival_won_here` for 12 shifts) — harmless
   now that the boarding gate ignores it, but if you rebuild the fixture, be aware. Do NOT resume from
   states/campaign/checkpoints/ auto-banks (STALE Jul-8).

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill first: `taskkill //F
   //IM python.exe //T`). recon_longrun setdefaults POKEMON_FIELD_MOVES=1 + POKEMON_ITEM_PICKUP=1 (field-cut
   IS armed in look-ahead). recon stages to G:/temp/longrun/stage; canonical Champion save NEVER touched.
   GOAL 0x822 = FLAG_BADGE_THUNDER. Kill-switches: POKEMON_RIVAL_PREP_LEVEL (32) = the ace prep target knob.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 3 Surge gym ENTRY (Gary
   WALL BEATEN this shift; residual = field-Cut the gym-door tree). Pop-in = `python pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
