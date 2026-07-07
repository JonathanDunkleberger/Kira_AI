# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #5 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅 **CANONICAL = blaine_badge7: Cinnabar Island (3,8)@(20,5), BADGES 7, party
HEALED, sanctity VALID, round-trip verified** (backup
pre_blaine_badge7_backup_20260707_130328). Party: Venusaur L60 (Razor Leaf/
STRENGTH/Sleep Powder/Secret Power) / Persian 37 / Fearow 35 / Raticate 31 /
Ekans 15 / Lapras L25 (SURF). Blaine fell FIRST TRY in blaine_run4 (59s total)
once the SE-CHUNK sleep-lock landed (battle_agent commit 9e3a447); Bill's
One-Island ambush B-declined; recon_blaine.py = the reusable quiz-gym vehicle.

⚔️ **SHIFT-5 LIVE OBJECTIVE: GIOVANNI badge 8 — strike vehicle = pokemon_agent/
recon_giovanni.py, runs logging to logs/longrun/giovanni_runN.log.** If a run is
in flight, READ ITS LOG END first; promote any bank before anything else
(`python pokemon_agent/promote_bank.py %TEMP%/longrun/banked_GIOVANNI
giovanni_badge8`).

**THE DERIVED TRUTH (pret cached G:\temp\longrun\pret\ViridianGym*.json/.inc):**
- THE ROAD: Cinnabar→Viridian = FIVE consecutive NORTH edge crossings
  (Cinnabar → R21S → R21N → Pallet → Route1 → Viridian); sea legs surf on
  Lapras (recon_seafoam mount/sea_walk/cross_edge machinery cloned verbatim).
- THE DOOR: ViridianCity OnTransition unlocks the gym with badges 2-7 held (she
  has 1-7) — coord event (36,11) dies on first city transition; warp (36,10) →
  gym map (5,1).
- THE GYM: spin-tile floor maze, NO doors/quizzes — spin_nav.SpinNav.cross()
  (the proven hideout glide crosser) gets its second customer. 8 juniors WITH
  sight 2-3 (spotting battles fine: Razor Leaf x2 into ground/rock; SE-chunk
  sleep-lock covers Nido poison). Giovanni (2,2) face DOWN → front (2,3), face
  UP. Post-win: **BADGE 8 = flag 0x827** + TM26 (A-drain) + his removeobject
  fade. NO exit ambush.
- ⚠️ After badge 8, VAR_MAP_SCENE_ROUTE22=3 arms GARY on Route 22 (westbound,
  his strongest pre-E4 team) — that's the NEXT objective's opening fight.

**Then (the standing chain):** Route 22 (Gary) → Route 23 badge-gate → Victory
Road (Strength/boulder machinery from recon_seafoam reuses directly) → Indigo
Plateau → E4 (stock Full Restores/Revives first — she has ~$72k) → **CREDITS.**

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead
weight (Ekans/Mankey); spin_nav unwired; _step_to grass verify window (filed);
Mansion item balls (TM22/TM14/Full Restore) skipped = backlog.
**SOUL-DEBT:** Seafoam crossing + burned-mansion key hunt + Lapras first-Surf =
prime narration set-pieces owed at next play-live; safari catches un-narrated;
the quiz gym is a NATURAL soul beat (her sweating trivia questions) — note for
the play-live pass.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F.

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (secret_key — badge 6 + Secret Key, healed,
standing in front of the Cinnabar Mansion); press GO and you'll see her open the
long-locked Cinnabar Gym for the badge-7 showdown with Blaine; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
