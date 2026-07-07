# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #4, IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

**CANONICAL = surf_taught: Fuchsia City (3,7)@(33,32), badges 6, sanctity VALID** (backup
pre_surf_taught_backup_20260707_102403). Party: Venusaur L57 (Razor Leaf/STRENGTH/Sleep
Powder/Secret Power) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 / **LAPRAS L25
(SURF/Body Slam/Confuse Ray/Perish Song)**. Mankey L10 in the box. ~$71k.

⚔️ **LIVE OBJECTIVE: THE SEAFOAM CROSSING → CINNABAR → BLAINE (badge 7).**
**recon_seafoam.py is the strike vehicle** (shift 4 build; check
`logs/longrun/seafoam_run*.log` tail FIRST — if a run banked, promote via
`python pokemon_agent/promote_bank.py %TEMP%\longrun\banked_CINNABAR cinnabar_reach`).

**THE SOLVED TRUTH (derivation = recon_seafoam_plan.py, all pret ground truth, cached
G:\temp\longrun\pret\ incl. CURRENT_STOPPED layout bins + scripts.inc):**
- R20 surface SEVERED (shift-2 dual-flood). Interior meta-BFS (floors as nodes, ladders
  as edges, water-as-road) finds NO route with currents active — the west-exit cluster
  {F1 exit region ← B1F (32,14)-region ← B2F (31,17)-region ← B3F east block} is sealed
  behind the B3F current field.
- **THE MECHANISM (scripts.inc):** B3F current stops when both B3F boulders PRESENT
  (HIDE flags 0x046/0x047 cleared by falls) → FLAG 0x2D2 → layout swaps to
  CURRENT_STOPPED (currents→calm water). Boulders cascade down the MB_FALL_WARP (0x66)
  hole chain: 1F→B1F→B2F→B3F. Falling into B3F with <2 boulders = FORCED RIDE to B4F.
- **THE MISSION** (in recon_seafoam.MISSION, all coords pret-verified): F1 b1 (22,12)
  UP×4 LEFT×1 → hole (21,8); b2 (32,9) UP×1 LEFT×2 → hole (30,8); fall (21,8) → B1F
  push b1 RIGHT×1 → fall (23,8) → B2F push b1 RIGHT×2 [b1 done] → ladders (7,4),(10,6)
  up → fall (30,8) → B1F push b2 LEFT×2 → fall (28,8) → B2F push b2 LEFT×3 [b2 done]
  → fall (27,8) → B3F becalmed water → surf to ladder (31,16) → B2F (32,14) → B1F
  (28,19) → F1 exit (32,21) → R20 (72,14) WEST sea → cross_edge west → CINNABAR (3,8).
- Hole-landing warp-event tiles ((21,8)/(29,8) B1F class) are one-way anchors, PLAIN
  behavior — safe to stand on. Strength re-arms PER MAP (flag 0x805 verifies; face
  boulder + A + drain-A = the prompt path). Push verified by live gObjectEvents coord
  (fm.scan_field_objects), never the template table.
- Seafoam floor ids: F1 (1,83) B1F (1,84) B2F (1,85) B3F (1,86) B4F (1,87). Articuno +
  the B4F puzzle = post-game episode, DO NOT detour.

**IF THE STRIKE WALLED:** diagnose per the playbook (instrument coords → deterministic
replay from canonical → probe the STAGE save `%TEMP%\longrun\stage_seafoam` offline →
screenshot → pret). Suspect list in build order: (1) Strength prompt actuation (first
live use of the class — if 0x805 never sets, try hm_teach.TeachFlow.use_field_move as
the menu-path fallback); (2) push actuation timing (hold 40f + settle 70f — boulder
should slide 1 tile); (3) falling onto becalmed water surf-state; (4) the R20-east
approach to door (60,8) (sea_walk over east sea — proven machinery).

**AFTER CINNABAR (chain, don't stop):** heal → **POKEMON MANSION** (Secret Key for the
gym — item ball in the basement; pad_plan-class interior, fetch pret maps same way) →
**BLAINE = badge 7** → Giovanni (Viridian badge 8; spin_nav.py exists UNWIRED) → Route
22/23 → Victory Road (Strength boulders again — recon_seafoam push machinery reuses) →
E4 → **CREDITS**.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight
(Ekans/Mankey); spin_nav unwired; _step_to grass move-verify window (filed,
campaign-shared); safari catches were silent RAM catches. **SOUL-DEBT:** Lapras's first
Surf beat owed at next play-live; the Seafoam crossing (first real dungeon-on-water) is
a narration set-piece candidate.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit or sweep.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire,
arsenal, single-run law, ground-truth-only, frontier-first NEXT_SESSION rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (surf_taught — badge 6, Surf on Lapras, Strength
on Venusaur; Fuchsia, outside the Warden's house); press GO and you'll see her set out
across the southern sea for the Seafoam Islands — the badge-7 leg; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
