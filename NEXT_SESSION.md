# NEXT_SESSION — resume prompt (write date 2026-07-07 ~15:46, night shift #11 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED at the League center, sanctity VALID**
(backup pre_indigo_reach_backup_20260707_154528). Party: Venusaur **L66**
(Razor Leaf/Sleep Powder/EQ) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 /
Lapras 26 (SURF). Money $63,678 — covers the full E4 kit (money-aware to ~$20k).

⚔️ **SHIFT-11 LIVE OBJECTIVE: THE ELITE FOUR — `recon_e4.py` e4_run1+ IN FLIGHT
(logs logs/longrun/e4_runN.log). THE CREDITS ARE THE NEXT BANK.**
If a run is in flight, READ ITS LOG END first. Promote any bank:
`python pokemon_agent/promote_bank.py %TEMP%/longrun/banked_CREDITS hall_of_fame`
(banked_E4 = per-room ratchet banks if it died mid-chain).

- recon_e4 boots CANONICAL (no RESUME_STAGE): heal → League mart stock-up
  (FR×10/Revive×6/Full Heal×4, per-unit money+bag verify) → League door (4,1) →
  room chain (one template: south (6,12), trainer (6,5), north door (6,2) opens
  on DEFEATED flag) → Lorelei→Bruno→Agatha→Lance→GARY (room 5, arrive (6,19),
  he's at (6,8)) → room 6 = HALL OF FAME → bank banked_CREDITS → credits drain →
  **write CREDITS as NIGHT_REPORT.md line 1 + the full mountain survey**.
- Whiteout = respawn at the League center, DEFEATED flags persist (cleared rooms
  ratchet); the vehicle re-heals, re-checks the kit (shopped un-latches), re-enters.
- ⚠️ Agatha wall (if she stalls): Venusaur can't touch Gengar/Haunter (Normal
  immune, Razor Leaf x0.25, EQ vs Levitate) — sleep-lock + Full Restore attrition
  + EQ on Arbok/Golbat-less turns; if 2-stuck aborts fire here, the fix is
  battle-agent move choice vs Levitate-ghosts, NOT the vehicle.
- ⚠️ Lorelei opener: sleep-lock the ice line, EQ Jynx.

**VICTORY ROAD is DONE (shift 11):** the 0x058 hidden-boulder detour (3F switch
row-3 push U2,L21,D1,L5,D3,R1 → hole drop → hole-jump → row-19 LEFT×19) is wired
+ VERIFIED LIVE in recon_victory.py (runs 9-11; f3485dc + 17a5b49). Ray+Tyra's
DOUBLE battle on the 3F pocket path is dodged via column 36 (avoid (38,14)/(39,14)).
**FILED GAP: battle agent has NO double-battle target actuation** (move select →
target submenu → stuck) — E4 is all singles, fine for credits; fix before any
double-battle route. League center heal registered (CITY_PC_DOORS (3,9)→(11,6),
NURSE_FRONT_OVERRIDES (13,11) — the League center is NOT the shared PC layout).

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

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, standing at the gates of the Indigo Plateau); the E4 strike
is running on the staging line; press GO on canonical and you'll see her walk into
the Pokémon League — the Elite Four and Gary's final stand are all that's left;
pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
