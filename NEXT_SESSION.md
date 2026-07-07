# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #7 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅 **CANONICAL = giovanni_badge8: Viridian City (3,1)@(36,11), ALL EIGHT BADGES,
party HEALED, sanctity VALID** (backup pre_giovanni_badge8_backup_20260707_131428).
Party: Venusaur L61 (Razor Leaf/STRENGTH/Sleep Powder/**EARTHQUAKE** once the
victory strike's phase 0 banks — canonical still has Secret Power) / Persian 37 /
Fearow 35 / Raticate 31 / Ekans 15 / Lapras L25 (SURF). Money ~$86k.

⚔️ **SHIFT-7 LIVE OBJECTIVE: THE ROAD TO THE PLATEAU — pokemon_agent/
recon_victory.py, logs logs/longrun/victory_runN.log.** If a run is in flight,
READ ITS LOG END first; promote any bank (`python pokemon_agent/promote_bank.py
%TEMP%/longrun/banked_VICTORY indigo_reach`).
- **victory_run1 postmortem SOLVED:** the EQ teach B-out was the FORGET-screen
  cursor — `_FORGET_TOPS` rows 2-4 were unmeasured probes (67/90/112); measured
  truth (recon_forget_probe.py) = tops **18/46/74/102/130** (28px spacing, border
  = box top+bottom runs, cursor WRAPS 4→0). Fixed in hm_teach.py, teach verified
  end-to-end on a throwaway core (Secret Power → EQ, 4.6s).
- Vehicle covers: EQ teach (TM26→Venusaur over Secret Power, forget_idx 3) →
  R22 GARY (trigger col 33, var==3, his strongest pre-E4 team; loss =
  whiteout-retry loop) → gate (28,0) → R23 badge gauntlet (7 drain-scenes) →
  VICTORY ROAD: 1F (11,20)→(3,2) no puzzle; 2F THE ONE PUZZLE = boulder (6,17)
  L,L,D,D,L,L onto switch (2,19) (offline-derived, all tiles verified) →
  (36,17)→3F (39,17)→(37,10)→2F east (38,9)→(48,12)→R23 north → INDIGO PLATEAU
  → heal → bank indigo_reach.

**After indigo_reach: E4 = the LAST vehicle** — shop Full Restores/Revives at
the League mart FIRST (~$86k available), then Lorelei→Bruno→Agatha→Lance→
CHAMPION GARY → **CREDITS** (write CREDITS as NIGHT_REPORT line 1 + full survey).
⚠️ Agatha wall: Venusaur has NO move that touches Gengar/Haunter (ghost immune
to Normal, Razor Leaf x0.25, EQ blocked by Levitate) — plan: sleep-lock + Bite
users (Persian/Raticate) + Full Restore attrition, or lean EQ on Arbok +
fodder-revive cycling.
⚠️ Lorelei opener: Dewgong/Cloyster/Slowbro all take Razor Leaf x2 EXCEPT
Dewgong (x1) and Jynx/Lapras (x1/x0.5-ish) — sleep-lock the ice line, EQ Jynx.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead
weight (Ekans/Mankey); _step_to grass verify window (filed); Mansion item balls
(TM22/TM14/Full Restore) skipped = backlog.
**SOUL-DEBT:** Seafoam crossing + burned-mansion key hunt + Lapras first-Surf +
the quiz gym + the badge-8 homecoming (Viridian gym finally open) = prime
narration set-pieces owed at next play-live; safari catches un-narrated.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F. Never kill a strike between "badge/goal=True"
and "BANKED" — read the log end first.

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (giovanni_badge8 — ALL EIGHT BADGES, healed,
standing in Viridian City); press GO and you'll see her set out west for Route 22
— the rematch with Gary on the road to Victory Road and the Indigo Plateau;
pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
