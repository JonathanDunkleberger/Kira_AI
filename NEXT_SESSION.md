# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #5 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = secret_key: Cinnabar Island (3,8)@(8,4), badges 6, SECRET KEY IN
BAG, party HEALED, sanctity VALID** (backup pre_secret_key_backup_20260707_123848).
Party: Venusaur L59 (Razor Leaf/STRENGTH/Sleep Powder/Secret Power) / Persian 37 /
Fearow 35 / Raticate 31 / Ekans 15 / Lapras L25 (SURF).

⚔️ **SHIFT-5 LIVE OBJECTIVE: BLAINE badge 7 — strike vehicle = pokemon_agent/
recon_blaine.py (clone of the sabrina gym vehicle), runs logging to
logs/longrun/blaine_runN.log.** If a run is in flight, READ ITS LOG END first;
promote any bank before anything else (`python pokemon_agent/promote_bank.py
%TEMP%/longrun/banked_BLAINE blaine_badge7`).

**THE FULL GYM TRUTH (derived from pret scripts, cached
G:\temp\longrun\pret\CinnabarGym_scripts.inc + CinnabarGym.json — trust it):**
- Island gym warp = (20,4) on Cinnabar (3,8); the locked-door coord event (20,5)
  only fires when VAR_TEMP_1==0; OnTransition sets VAR_TEMP_1=1 whenever
  FLAG 0x1A8 (Secret Key collected) is set — she has it, door is OPEN now.
  If entry ever bounces: enter/exit the Poke Center to re-fire OnTransition.
- Gym = map (12,0). SIX QUIZ DOORS, each opens on EITHER a correct answer OR
  beating that room's trainer (wrong answer = the trainer WALKS TO YOU, battle,
  door opens anyway — the gym is fail-safe both ways; a botched YES/NO press can
  never wedge the run, it just costs a battle we win).
- Quiz machines = FACING_NORTH bg-event PAIRS (stand at (x,y+1), face UP, press A):
  Q1 (22-23,10) answer **YES** (A-drain) → flag 0x265, door1 (26-28,8-10)
  Q2 (15-16,2)  answer **NO** (B-drain) → flag 0x267, door2 (17-19,8-10)
  Q3 (13-14,10) answer **NO**           → flag 0x268, door3 (17-19,15-17)
  Q4 (13-14,17) answer **NO**           → flag 0x269, door4 (11,21-23)
  Q5 (1-2,18)   answer **YES**          → flag 0x26A, door5 (5-7,16-18)
  Q6 (1-2,10)   answer **NO**           → flag 0x26B, door6 (5-7,8-10)
  (B advances plain msgboxes AND selects NO on the YES/NO — one key per station.)
- Trainers: Quinn(25,11) Erik(25,4) Avery(17,5) Ramon(16,11) Derek(16,18)
  Dusty(4,19) Zac(4,11) — sight 0, none spot; Erik guards no door. Blaine (5,4)
  face DOWN → front (5,5) face UP. Arcanine L47 tops the roster; Venusaur L59
  Sleep-Powder+Strength carries. Post-win: FLAG_DEFEATED_BLAINE 0x4B6 +
  **BADGE7 = flag 0x826** + TM38 gift (A-drain, no Y/N).
- ⚠️ **THE BILL AMBUSH (the trap):** beating Blaine sets VAR_MAP_SCENE_CINNABAR=1;
  the FIRST transition back onto the island fires a FORCED lockall scene — Bill
  (spawns (20,7), right at the gym door) runs up with a YES/NO "sail to One
  Island?". **A-drain = YES = shipped to the Sevii Islands mid-bank.** The island
  drain post-badge must be **B-ONLY** until stable (decline → Bill leaves to the
  Center, scene never re-fires). Only then heal_nearest + bank.

**Then (the standing chain):** Giovanni badge 8 (Viridian gym — spin_nav.py exists
UNWIRED, the gym is a spin-tile maze + juniors) → Route 22/23 → Victory
Road (Strength machinery from recon_seafoam reuses directly) → E4 → **CREDITS.**

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
