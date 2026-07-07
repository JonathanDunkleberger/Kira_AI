# NEXT_SESSION — resume prompt (write date 2026-07-07 ~16:05, night shift #12 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur **L66** (Razor Leaf/Sleep
Powder/EQ/Secret Power) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26
(SURF). Money $63,678 — covers the full E4 kit.

⚔️ **SHIFT-12 LIVE OBJECTIVE: THE ELITE FOUR — `recon_e4.py` e4_run3+ IN FLIGHT
(logs logs/longrun/e4_runN.log). THE CREDITS ARE THE NEXT BANK.**
If a run is in flight, READ ITS LOG END first (never kill between "HALL OF FAME"
and "BANKED"). Promote any bank:
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
(banked_E4 = per-room ratchet bank if it died mid-chain — room3 = Agatha's room).

**SHIFT-12 KILLS (commit 928dd53) — the run-1/2 postmortem, both VERIFIED:**
- **Bag TRUE-row law:** in-battle bag selection = cursor(0x0203AD04) +
  scroll(0x0203AD0A itemsAbove), BOTH persist between opens. run2's "selected but
  NOT consumed" = A on Revive/CANCEL off a stale scroll — the Full Restore kit was
  a coin-flip. use_item_in_battle now navigates the true row (live-verified from
  the poisoned state: FR consumed, HP 23→193 PASS; harness recon_bagscroll*.py).
- **WAR-MUST-ADVANCE:** a can't-flee trainer battle with all moves streaked/0-PP
  used to submit NO action → the turn-based game waits forever (run2's Agatha
  livelock: famine → abort → re-enter, the foe never got a turn, so she couldn't
  even LOSE her way to the whiteout ratchet). Now: re-fire the best PP-having move
  (immune-damaging last resort), zero-PP → FIGHT+A = Struggle. Battle ALWAYS
  resolves; the whiteout→center→re-enter ratchet (DEFEATED flags persist) refills
  PP for the next attempt.
- E4 room chain truth (runs 1-2): Lorelei + Bruno fall cleanly first try (~50s,
  sleep-lock + Razor Leaf/EQ). Agatha = the PP sink (ghosts resist RL x0.5, EQ
  dead vs Levitate, Secret Power dead vs Ghost) — expect whiteout-attrition:
  each fresh full-PP arrival + working Full Restores should take her. Then Lance
  (RL x0.5 chip + Secret Power x1 + sleep-lock), then GARY (room 5), room 6 =
  HALL OF FAME → banked_CREDITS → credits drain → **write CREDITS as
  NIGHT_REPORT.md line 1 + full mountain survey**.

**KNOWN GAPS (owed):** Revives bought but NEVER offered in-battle (the item
instinct has no use_revive offer — mid-fight ace resurrection unbuilt; would
shorten Agatha/Lance attrition); double-battle target actuation missing (E4 is
all singles — fine for credits); Venusaur "AAAAAAAAAA" (Name Rater, Lavender);
bench dead weight (Ekans/Mankey); VR loot backlog.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8
homecoming + VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET +
whiteout-and-comeback arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F (run1 ghosted alive under run2 this shift — the
single-run law is real). Never kill a strike between "goal=True" and "BANKED".

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on
the staging line with the Agatha kill-chain fixed; press GO on canonical and
you'll see her walk into the Pokémon League for the Elite Four and Gary's final
stand; pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
