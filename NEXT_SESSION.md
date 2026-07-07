# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #4 CLOSE)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = secret_key: Cinnabar Island (3,8)@(8,4), badges 6, SECRET KEY IN
BAG, party HEALED, sanctity VALID, round-trip verified** (backup
pre_secret_key_backup_20260707_123848). Party: Venusaur L59 (169HP; Razor Leaf/
STRENGTH/Sleep Powder/Secret Power) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 /
Lapras L25 (SURF). Two promotions this shift: **cinnabar_reach** (the Seafoam crossing,
run10 = 153s Fuchsia→Cinnabar fully autonomous) then **secret_key** (mansion_run10 =
72s door→key→out incl. heal).

⚔️ **NEXT OBJECTIVE: BLAINE = badge 7. Everything is staged:**
- Gym door (20,4) on Cinnabar — the Secret Key UNLOCKS it on approach (the locked-door
  coord event (20,5) fires an unlock script when FLAG key is held; if it still bounces
  her, enter via go_warp on (20,4) after the script clears VAR).
- Gym (12,0): Blaine at (5,4) (stand (5,5)?, face UP — verify from CinnabarGym.json
  bg/leader row); 7 juniors (quiz machines: wrong answer = trainer battle, either way
  the door opens — fighting through is the autonomous path). Roster tops Arcanine L47
  vs Venusaur L59: Sleep Powder + STRENGTH carry (Razor Leaf resisted — fine on level).
- Clone the koga/sabrina gym vehicle pattern (juniors → leader → badge flag). Cinnabar
  Gym maps cached: G:\temp\longrun\pret\CinnabarGym.json. Bank blaine_badge7.
- **Then:** Giovanni (Viridian gym badge 8; spin_nav.py exists UNWIRED) → Route 22/23
  → Victory Road (recon_seafoam's push/arrow/stair machinery reuses DIRECTLY) → E4 →
  **CREDITS.**

**SHIFT-4 ENGINE KILLS (commits 7bb5a31→7a9b52e, all live-verified):**
1. box_open DUAL-CLAUSE (≥40% >242 AND ≥78% >200) — Seafoam-ice false-positive
   (silent drain livelock) + trainer-intro false-negative both measured and killed.
2. Per-edge ELEVATION LAW into travel.Grid.edge_open engine-wide (water exempt).
3. WARP DOOR CLASSES solved generally: arrow warps 0x62-0x65 (press arrow ON the mat),
   directional stairs 0x6C-0x6F (walk in along stair direction; step off first if on
   one), fall holes 0x66 (walk on), plain-behavior warp events = LANDING ANCHORS that
   NEVER fire (route edges must filter to trigger behaviors).
4. COORD-EVENT script tiles masked in BFS (Cinnabar (20,5) GymDoorLocked bounce class);
   reader = MapHeader.events coordEvents stride 0x10.
5. Distance-culled live objects; live-npc-body masks; box-interrupts-burn-budget (only
   battles refund); 4-same-coord-replan wedge detector with frame snap.
6. Strength actuation VERIFIED LIVE (face+A+YES, flag 0x805, re-arms per map).
7. Mansion toggle-state pad_plan class: ONE global switch flag (0x26C), statues =
   FACING_NORTH bg events (stand below, face up), setmetatile arg-4 = collision bit.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight
(Ekans/Mankey — Blaine may want a water answer: Lapras L25 too soft, level her or lean
Venusaur); spin_nav unwired; _step_to grass verify window (filed); Articuno/B4F +
Mansion item balls (TM22/TM14/Full Restore skipped) = post-game/backlog.
**SOUL-DEBT:** the Seafoam crossing (Lapras through the dark ice cave, taming the
currents with boulders) + the burned-mansion key hunt = PRIME narration set-pieces owed
at next play-live; Lapras first-Surf beat owed; safari catches still un-narrated.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit/sweep.
**py-spy is in .venv** — first tool for any silent wedge. Monitors/watchers: filter
must include op lines AND failures; kill orphan runs with taskkill //F (bash kill can
silently fail on Windows — the run-3 ghost).

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law, ground-truth-only,
frontier-first rewrites. GO — Blaine is one clean strike away.

---

WATCH STATUS: canonical bank is CLEAN (secret_key — badge 6 + Secret Key, healed, she's
standing in front of the Cinnabar Mansion with the gym key in her bag); press GO and
you'll see her walk up to the long-locked Cinnabar Gym and open it — the badge-7
showdown with Blaine; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
