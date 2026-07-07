# NEXT_SESSION — resume prompt (write date 2026-07-07 ~16:40, night shift #13 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur **L66** (Razor Leaf/Sleep
Powder/EQ/Secret Power) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26.

⚔️ **SHIFT-13 LIVE OBJECTIVE: THE ELITE FOUR — `recon_e4.py` e4_run7+ IN FLIGHT
(logs logs/longrun/e4_runN.log). THE CREDITS ARE THE NEXT BANK.**
FIRST MOVE: check for a live python process + read the newest e4_runN.log END.
If banked_CREDITS exists: promote it —
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
— then write CREDITS as NIGHT_REPORT.md line 1 + the mountain survey. Never kill a
run between "HALL OF FAME" and "BANKED".

**RUN-6 TRUTH (the current best attempt — killed EXTERNALLY, not defeated):**
fresh canonical boot → shopped 10 Full Restores + 6 Revives + 4 Full Heals ($22,278
left) → Lorelei fell (~50s, sleep-lock + RL x2) → Bruno fell → entered Agatha
(lead 69%, alive 6) → was chipping Agatha's ghosts (RL x0.5, FRs firing correctly,
5 left) when the shift-12→13 handover killed the python tree at 16:35. THE VEHICLE
IS SOUND — relaunch is the correct move, fresh from canonical (E4_BOOT unset):
`.venv\Scripts\python.exe -u pokemon_agent\recon_e4.py` with stdout to
logs/longrun/e4_run7.log. A fresh boot re-loads canonical ($63k restored), re-shops,
re-clears rooms 1-2 in ~3 min at 14x.

**THE GAUNTLET LAW (shift-12, commit 2d7234d): E4 RESETS ON WHITEOUT** — DEFEATED
flags do NOT survive a whiteout (run5 proof: post-whiteout she re-fought Lorelei).
E4 = ONE UNBROKEN RUN Lorelei→Bruno→Agatha→Lance→Gary on one tank of PP
(~RL25+EQ10+SP15+SecretPower20). Per-room banked_E4 = diagnosis only. Whiteout-
attrition does NOT accumulate; each attempt is fresh-from-canonical. The doors lock
behind her — no mid-chain center heal.

**KNOWN WALLS (in kill order):**
- Agatha = the PP sink (ghosts resist RL x0.5, EQ dead vs Levitate/Gengar,
  Secret Power dead vs Ghost; she Full-Restores + Hypnosis/Confuse Ray stalls).
  Run3 attempt-1 DID clear her at L66 (Struggle-recoil finish) — she is beatable
  on one tank.
- Lance killed run3 attempt-1 (arrived with 3 fainted). With FRs now actually
  consuming (bag TRUE-row fix 928dd53) arrival state should be far better.
- **REVIVES bought but NEVER offered in-battle** — the #1 unbuilt lever: if
  Venusaur faints vs Agatha/Lance, a Revive→FR resurrection from a sacrificial
  bench mon turns a loss into a continue. The party-screen AIM machinery already
  exists (post-switch item AIM, shift 12); wiring 'use_revive' into the
  ITEM-INSTINCT offer when ace fainted = the highest-value fix if runs keep dying
  at Lance. Build it on a STAGING copy while a run flies.
- run3's PHANTOM battle re-attach (fight_open pointer stale after whiteout) —
  anti-wedge now dumps forensics to G:/temp/longrun/e4_probe; read that on any
  wedge. Class: needs LIVENESS check not pointer check.

**KNOWN GAPS (owed, non-blocking):** double-battle target actuation (E4 all
singles — fine); Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight;
VR loot backlog.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8
homecoming + VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET +
whiteout-and-comeback arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F //T (single-run law). Never kill a strike between
"goal=True" and "BANKED".

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on
the staging line; press GO on canonical and you'll see her walk into the Pokémon
League for the Elite Four and Gary's final stand; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
