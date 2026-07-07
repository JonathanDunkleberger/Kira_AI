# NEXT_SESSION — resume prompt (write date 2026-07-07 ~15:40, night shift #11 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅 **CANONICAL = giovanni_badge8: Viridian City (3,1)@(36,11), ALL EIGHT BADGES,
party HEALED, sanctity VALID** (backup pre_giovanni_badge8_backup_20260707_131428).
Party: Venusaur L61+ (Razor Leaf/Sleep Powder/EARTHQUAKE — EQ taught in phase 0 each
run) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 / Lapras L25 (SURF). Money
~$69k on the staging line (whiteouts) — E4 shop plan is money-aware down to ~$20k.

⚔️ **SHIFT-11 LIVE OBJECTIVE: VICTORY ROAD — `RESUME_STAGE=1` victory_run9+
(pokemon_agent/recon_victory.py, logs logs/longrun/victory_runN.log).**
If a run is in flight, READ ITS LOG END first; promote any bank
(`python pokemon_agent/promote_bank.py %TEMP%/longrun/banked_VICTORY indigo_reach`).

- **RUN8 POSTMORTEM (SOLVED, wired):** "no boulder on row 19" was GROUND TRUTH,
  not a scan miss — the 2F row-19 boulder (33,19) is **FLAG-0x058-HIDDEN** until
  the 3F (33,18) boulder drops through hole (34,18)
  (HandleBoulderFallThroughHole clears the reveal flag stored in the object's
  trainer_type — field_control_avatar.c:1066). Shift-8's "present from game
  start, 0x058 clear" was a misread. **THE DETOUR IS NOW WIRED into
  recon_victory.py** (dispatches on flag 0x058, both 2F and 3F branches):
  2F (34,9) ladder (beh 0x61, only up-ladder reachable pre-barrier2) → 3F push
  (32,5): **U2, L21 along row 3, D1, L5, D3, R1** → switch (7,7) opens 3F
  barrier (12,12-13) → push (33,18) RIGHT into hole → jump in after it → lands
  2F (34,19) beside the revealed boulder → existing row-19 LEFT×19 → switch
  (14,19) → barrier2 open → (36,17) → 3F pocket → (37,10) → 2F east → (48,12)
  exit → R23 north → Indigo → BANK indigo_reach.
  Derivations: vr3f_probe2.py/probe3.py in G:/temp/longrun/pret (two-body push
  BFS). **Row 5 is Alexa-blocked** (trainer at (21,5)) — row 3 is the road.
  **NEVER push boulder (35,13)** — the (37,10) pocket is a sealed island and
  that boulder can wedge INTO its corridor (no switch2 bypass exists; verified).
- **RESUME_STAGE=1 boots the stage bank** (%TEMP%/longrun/stage_victory —
  2F, switch1 open, EQ taught, R22 Gary WON, gauntlet drained, 1F done).
  Unset RESUME_STAGE (or delete stage_victory) for a from-canonical full run.
- Drain armor + gMoveToLearn decline + 2-stuck-fight abort are IN (80db047),
  verified by run8's clean fights.
- ⚠️ Before launching: confirm no orphan python/bash watchers
  (`Get-Process python*`; taskkill //F). Never launch probes that outlive the shift.

**After indigo_reach: E4 = the LAST vehicle — `recon_e4.py` (COMPILES, unrun):**
League mart stock-up FIRST (Full Restore×10/Revive×6/Full Heal×4, money-aware),
then Lorelei→Bruno→Agatha→Lance→CHAMPION GARY → **CREDITS** (write CREDITS as
NIGHT_REPORT line 1 + full mountain survey).
⚠️ Agatha wall: Venusaur has NO move that touches Gengar/Haunter (ghost immune to
Normal, Razor Leaf x0.25, EQ blocked by Levitate) — plan: sleep-lock + Bite users
(Persian/Raticate) + Full Restore attrition, or lean EQ on Arbok + fodder-revive.
⚠️ Lorelei opener: sleep-lock the ice line, EQ Jynx.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead
weight (Ekans/Mankey); _step_to grass verify window (filed); Mansion item balls +
VR loot (Rare Candy (12,3) 1F etc.) skipped = backlog.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8 homecoming
+ THE VICTORY-ROAD CLIMB + Gary-before-the-gate = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F. Never kill a strike between "badge/goal=True"
and "BANKED" — read the log end first.

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (giovanni_badge8 — ALL EIGHT BADGES, healed,
standing in Viridian City); she is currently mid-Victory-Road on the staging line
(2F, the hidden-boulder detour ahead); press GO on canonical and you'll see her set
out west for Route 22 — Gary's last stand on the road to the Indigo Plateau;
pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
