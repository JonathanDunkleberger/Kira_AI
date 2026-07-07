# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #4, post-Seafoam)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅 **CANONICAL = cinnabar_reach: Cinnabar Island (3,8)@(21,15), badges 6, sanctity
VALID, round-trip verified** (backup pre_cinnabar_reach_backup_20260707_121633).
**THE SEAFOAM CROSSING IS DONE** (seafoam_run10: 153s Fuchsia→Cinnabar, fully
autonomous — boulder cascade, becalmed B3F surf, arrow-warp exit, west sea). Party:
Venusaur L59 (hurt at bank — heal owed, Cinnabar PC door now in CITY_PC_DOORS
(14,11)) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 / Lapras L25 (SURF).

⚔️ **LIVE OBJECTIVE: SECRET KEY → BLAINE (badge 7).** recon_mansion.py IS IN FLIGHT
(mansion_run1.log — read its tail FIRST; if BANKED appears:
`python pokemon_agent/promote_bank.py %TEMP%\longrun\banked_SECRETKEY secret_key`).
- **Mansion route (derived offline, mansion_route.json in G:\temp\longrun\pret\):**
  door (8,3) → 1F toggle statue (5,5) ON → stairs (10,13)→2F → (27,17)→3F →
  (18,18)→1F balcony → (25,27)→B1F → toggle (24,29) OFF → toggle (27,5) ON → KEY
  ball (5,7) → out (34,29) → south mats (8,33). ONE global switch state FLAG 0x26C;
  statues = bg events, face+A+YES; setmetatile arg-4 IS the collision bit; key
  pickup verified by FLAG 0x1A8. Floors 1F-B1F = (1,59)-(1,62).
- **Then BLAINE:** gym (12,0), door (20,4) on Cinnabar (Secret Key gates it);
  Blaine at (5,4) leader_front (5,5)-ish; 7 juniors (quiz doors — wrong answer =
  battle, either way clears); roster tops at Arcanine L47 vs Venusaur L59+
  (Sleep Powder + STRENGTH move carry; Razor Leaf is resisted — fine on level).
  Build a GymSpec row + reuse the standing beat_gym pattern (juniors → leader),
  or clone the koga vehicle. Bank blaine_badge7.
- **Then:** Giovanni (Viridian gym, badge 8 — spin_nav.py exists UNWIRED for the
  maze) → Route 22/23 → Victory Road (recon_seafoam's push/arrow machinery reuses
  directly: Strength boulders + fall holes) → E4 → **CREDITS.**

**SHIFT-4 ENGINE KILLS (commits 7bb5a31→d2e5570, all live-verified in run10):**
box_open DUAL-CLAUSE (bright tilesets: ≥40% pure-white AND ≥78% bright — ice
false-positive + trainer-box false-negative both measured); per-edge ELEVATION LAW
in Grid.edge_open engine-wide (water shoreline exempt = mount seam); arrow-warp
doors (MB 0x62-0x65: approach from arrow-opposite side, hold arrow key);
distance-culled live objects (template-guided approach, live-verified pushes);
live-npc-body BFS masks (wanderers park off-template). Strength actuation class
VERIFIED LIVE (face+A+YES, flag 0x805, per-map re-arm).

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead
weight (Ekans/Mankey); spin_nav unwired; _step_to grass verify window (filed);
Articuno/B4F = post-game episode (do NOT detour). **SOUL-DEBT:** the Seafoam
crossing (Lapras carrying her through a dark ice cave, taming the currents) is a
PRIME narration set-piece — owed at next play-live; Lapras's first-Surf beat owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit or
sweep. **py-spy is pip-installed in .venv** — first tool for a silent wedge.

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (cinnabar_reach — badge 6, the Seafoam
severance crossed, she's standing on Cinnabar Island for the first time with the
Mansion looming west); press GO and you'll see her set off for the burned-out
Pokemon Mansion hunting the gym's Secret Key — the badge-7 leg; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
