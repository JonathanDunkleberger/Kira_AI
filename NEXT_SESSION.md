# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #8 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅 **CANONICAL = giovanni_badge8: Viridian City (3,1)@(36,11), ALL EIGHT BADGES,
party HEALED, sanctity VALID** (backup pre_giovanni_badge8_backup_20260707_131428).
Party: Venusaur L61 (Razor Leaf/STRENGTH/Sleep Powder/EARTHQUAKE after the strike's
phase 0) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 / Lapras L25 (SURF).
Money ~$86k.

⚔️ **SHIFT-8 LIVE OBJECTIVE: VICTORY ROAD — pokemon_agent/recon_victory.py,
logs logs/longrun/victory_runN.log (run3+ = the fixed vehicle).** If a run is in
flight, READ ITS LOG END first; promote any bank (`python pokemon_agent/
promote_bank.py %TEMP%/longrun/banked_VICTORY indigo_reach`).
- **victory_run2 postmortem SOLVED (shift 8):** "1F no puzzle" was an
  elevation-blind vr_solve.py artifact + distance-culled live boulders let the
  first BFS plan through the (4,12) plug. TRUTH (vr1f_probe*.py in
  %TEMP%/longrun/pret, all elevation-aware, every stand verified): EVERY floor
  barrier opens only by pushing a boulder onto its 0x20 STRENGTH_BUTTON switch
  (boulder-lands-on-switch fires the coord event, field_control_avatar.c:1076).
  - **1F:** barrier (12,14-15) GATES THE LADDER. Chain: arm Strength at (7,18),
    push D1, R4, U1 [stand (11,20) = entrance arrow tile 0x65, warps on DOWN
    only — we press UP; sea_walk allow= handles it], R1, U1, R7, U2, R1, D1 →
    lands (20,16) → ladder (3,2).
  - **2F:** puzzle1 (6,17): D1, L2, D1, L2 → (2,19) opens barrier1 (13,10-11)
    [old L,L,D,D,L,L recipe was elevation-ILLEGAL]. Then boulder (33,19)
    (present from start, FLAG 0x058 clear) LEFT×19 → (14,19) opens barrier2
    (33,16-17). Reach the (34,19) stand via the row-20 bypass (BFS finds it).
    If (33,19) missing/wedged: 3F reset detour = (34,9) ladder → push (32,5)
    onto (7,7) → push (33,18) into hole (34,18) → fall → fresh boulder.
  - **3F pocket + 2F east + exit (48,12, arrow 0x65):** no puzzle, verified.
- Vehicle also covers: EQ teach (done, verified in run2: moves [75,70,79,89]) →
  R22 GARY (won in run2, repeats each run — trigger re-fires off stage, fine) →
  R23 badge gauntlet (drains) → VR → Indigo heal → BANK indigo_reach.

**After indigo_reach: E4 = the LAST vehicle** — shop Full Restores/Revives at the
League mart FIRST (~$86k), then Lorelei→Bruno→Agatha→Lance→CHAMPION GARY →
**CREDITS** (write CREDITS as NIGHT_REPORT line 1 + full survey).
⚠️ Agatha wall: Venusaur has NO move that touches Gengar/Haunter (ghost immune
to Normal, Razor Leaf x0.25, EQ blocked by Levitate) — plan: sleep-lock + Bite
users (Persian/Raticate) + Full Restore attrition, or lean EQ on Arbok +
fodder-revive cycling.
⚠️ Lorelei opener: sleep-lock the ice line, EQ Jynx.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead
weight (Ekans/Mankey); _step_to grass verify window (filed); Mansion item balls
(TM22/TM14/Full Restore) skipped = backlog; VR loot (Rare Candy (12,3), TM02
(14,1), Full Restore hidden (16,1) on 1F; TM07/TM37/Guard Spec/Full Heal 2F;
Max Revive/TM50 3F) skipped = backlog.
**SOUL-DEBT:** Seafoam crossing + key hunt + Lapras first-Surf + quiz gym +
badge-8 homecoming + THE VICTORY-ROAD CLIMB + Gary-before-the-gate = prime
narration set-pieces owed at next play-live.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F. Never kill a strike between "badge/goal=True"
and "BANKED" — read the log end first.

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (giovanni_badge8 — ALL EIGHT BADGES, healed,
standing in Viridian City); press GO and you'll see her set out west for Route 22 —
the rematch with Gary on the road to Victory Road and the Indigo Plateau;
pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
