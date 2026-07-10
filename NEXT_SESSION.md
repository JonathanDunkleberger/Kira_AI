<!-- ═══ NIGHT-SHIFT 15 (2026-07-10) — BADGE 3 wall = S.S. ANNE GARY (off-anchor prep-grind dead spot) ═══
THE FRONTIER: BADGE 3 (Lt. Surge). Shift 14 closed the cabin-skip DEAD-ZONE (cabins now level the ace
L26->L30). Shift 15 caught the NEXT link of the same livelock live and fixed it; a verify run is in flight.

★★ SHIFT-15 HEADLINE — the ace-to-L32 prep-grind can't fire after a Gary loss because the loss whites her
   out OFF-ANCHOR. Ground truth from s14_surge.log (shift-14's own verify run, watched live):
   - Cabin sweep worked (shift-14 fix good): ace climbed **L26 -> L30** fighting cabin trainers.
   - She B-lined to Gary at **Ivysaur L30** and LOST (grudge 4W-3L; Gary = charmeleon+kadabra+pidgeotto+
     raticate). Shift-11's ONLY proven Gary-killer is **Venusaur L32** (Ivysaur evolves at L32 — a bulk/
     power spike). L30 loses.
   - The whiteout dumped her at the **CERULEAN** Center (last-healed Center), NOT Vermilion.
   - Back in Cerulean the clean, narrated **PREP-BEFORE-RIVAL** grind (self.grind(L32), the thing that
     tops the ace to the winning level) NEVER FIRED — it was gated on `cur_map == step_anchor` (she must be
     standing on the ship's Vermilion approach). Off-anchor, only the GENERIC underlevel-prep runs, and that
     targets the BENCH FLOOR (~L19) and STOPS — the ace crept only to L31 via a wedgy generic grind, never
     L32 -> she'd trek back, reboard at ~L30-31, lose again = LIVELOCK.

   ROOT: `cur_map == step_anchor` gate on the prep-grind (campaign.py ~6014). A rival LOSS displaces the
   respawn town (Cerulean != the step's from_map anchor Vermilion), so the ace-to-L32 grind can never
   re-fire from where the whiteout actually put her. (Exactly the shift-14-flagged next-suspect: "whiteout-
   to-Cerulean displaces step_anchor — the prep gate needs cur_map==step_anchor".)

   FIX (SHIFT 15, campaign.py ~6006, COMMITTED — see git log): once she has LOST to this rival
   (`_rival_lost_here`), fire the prep-grind from WHEREVER she healed, not only at the anchor. New:
   `_prep_at_anchor = step_anchor and cur_map==step_anchor`; `_prep_off_anchor_afterloss = _rival_lost_here
   and not _prep_at_anchor`; gate fires on `(_prep_at_anchor or _prep_off_anchor_afterloss)`. The anchor
   path keeps the hardcoded Route-6 walk_to_map (grind()'s grass-finder wedges in the gym-approach POCKET at
   the Cut tree); the OFF-anchor path skips it and lets grind() find LOCAL grass (safe from a normal town
   like Cerulean). The re-arm at `cur_map!=step_anchor` clears `_ql_prefight_grind` each off-anchor tick, so
   the grind re-fires until the ace hits L32 (the `0 < _ace_lv < _RIVAL_PREP_LEVEL` guard self-terminates ->
   evolve to Venusaur -> head back -> board -> cabins skip (>=32) -> WINNABLE Gary). General per rule 14:
   helps every later rival (Tower/Silph) whose loss also displaces the respawn town. SYNTAX OK.

   VERIFY RUN IN FLIGHT (launched shift 15): `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1
   LONGRUN_GOAL_FLAG=0x822 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py bill_done_kit.state 55`
   -> /g/temp/s15_surge.log. WATCH FOR (grep -vE "ctx="): cabin sweep L26->L30 -> Gary loss at ~L30 ->
   whiteout to Cerulean -> **'PREP-BEFORE-RIVAL ... grinding up BEFORE boarding off-anchor (post-loss
   respawn town)'** (THE NEW LINE — this is the fix proving out) -> ace L30->L31->L32 -> Venusaur -> trek
   back to Vermilion -> reboard -> 'SHIP CABIN SWEEP SKIPPED — ace L32 ... >= L32' -> Gary grudge
   encounter **won=True** -> captain -> HM01 Cut -> disembark -> head_to_gym -> Cut tree at gym door ->
   'stuck' -> shift-11 gym-gate probe -> TrashCanPuzzle -> beat Surge -> OUTCOME: GOAL (0x822 =
   FLAG_BADGE_THUNDER).
   IF GOAL: bank surge_done, climb to BADGE 4 (Erika/Celadon — Rock Tunnel/Flash chain has night-train
   fixes, MEMORY.md).
   IF Gary WON but no Surge: fix proven — residual is downstream (captain/Cut/gym-gate); re-read the log
   for the stall (shift-11/12 fixes should cover it).
   IF the 'PREP-BEFORE-RIVAL ... off-anchor' line never appears: check `_rival_lost_here` is truthy in
   Cerulean (it matches strat.rival encounters whose place contains 's.s. anne' — grep "LOSS recorded vs
   trainer:the S.S. Anne:charmeleon"). If it fires but the ace still won't reach L32, the residual suspect
   is the wedgy in-battle GRIND SWITCH ("grind switch did not confirm -> fighting (fail-safe)" + weak-mon
   Peck-and-flee, s14 tail) — the ace isn't reliably taking the KO turns, so XP creeps. That's Tier-1 #5
   (in-battle switch actuation), a deeper separate fix; the ace still levels, just slowly.

KNOWN INEFFICIENCY (NOT fixed this shift, noted): the grind's GRIND SWITCH (put weak lead out, ace fights
   for participation XP) frequently "did not confirm -> fighting (fail-safe)", so a weak mon Pecks
   ineffectively and the anti-wedge floor FLEES after 3 turns — the grind still nets ace XP but slowly and
   unwatchably. Tier-1 #5 in-battle switch actuation. Candidate next fix if the badge-3 run converges.

FIXTURES: bill_done_kit.state at pokemon_agent/states/workshop/ (Ivysaur L26 [RazorLeaf,VineWhip,Tackle,
   SleepPowder] + frail bench Spearow/Rattata/Abra L8-12) — rebuild `.venv/Scripts/python.exe
   pokemon_agent/recon_repair_kit.py`. Do NOT resume from states/campaign/checkpoints/ auto-banks (STALE
   Jul-8). The full organic bill_done_kit run reproduces the Gary wall end-to-end (the honest verify).

KEY FACTS: venv python `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first:
`tasklist //FI "IMAGENAME eq python.exe"` then `taskkill //F //IM python.exe //T`). The `.venv` shim
DETACHES + fires a false 'completed' on background launch — the REAL worker keeps running; check via
tasklist + log mtime, NOT the task-notification. recon stages to G:/temp/longrun/stage (WIPED each run),
banks to banked_<OUTCOME>; canonical Champion save NEVER touched. GOAL flags: recon_longrun.py:67-76.
Kill-switches: POKEMON_SHIP_CABIN_SWEEP=0 disables the sweep; POKEMON_RIVAL_PREP_LEVEL (32) is the ONE knob
both the sweep-skip and the prep-grind target (shift-14 coupled them — no dead-zone). Gary battle grep:
`grep -nE "PREP-BEFORE|SWEEP|won=True|LOSS recorded vs.*S.S|BADGE|GOAL|OUTCOME" log | grep -v "ctx="`.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 3 (Surge) — off-anchor
prep-grind fix (committed shift 15) in-flight-verify to the badge. Pop-in = `python pokemon_agent/watch.py`.
═══════════════════════════════════════════════════════════════════════════════════════════════ -->
