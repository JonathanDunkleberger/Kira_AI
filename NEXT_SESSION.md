# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #2, pre-strike rewrite)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

**CANONICAL = fuchsia_south: Fuchsia City (3,7)@(47,21), badges 6, $71,886, sanctity VALID**
(banked 08:41; backup pre_fuchsia_south_backup_20260707_084122). Party: Venusaur L57 /
Persian 37 / Fearow 35 / Raticate 31 / Ekans 15 / **LAPRAS L25 (slot 5 — the Surf carrier,
withdrawn from Bill's PC by recon_lapras)**. Mankey L10 is in the box.

⚔️ **IN FLIGHT AT WRITE: the SAFARI STRIKE, attempt 12 (recon_safari.py → log
logs/longrun/safari_strike12.log).** FIRST MOVE: read that log's END.
- Targets in one entry: **GOLD TEETH** ball (Area 3 West (28,14)) + **HM03 SURF** (Secret
  House attendant (6,5), West door (12,7)) → exit → Warden's house (Fuchsia door (33,31))
  → **HM04 STRENGTH**. Success = items 341+342 in the TM pocket → banks to
  %TEMP%/longrun/banked_SAFARI → promote as **safari_hms**
  (`python pokemon_agent/promote_bank.py <bank> safari_hms`).
- **THE POND TRUTH (strikes 7-11, night shift 2 — do NOT try the west doors again):** the
  Center's pond splits it into two components; the WEST (8,17-19) + NORTH (25-27,5) doors
  are on the far SHELF, unreachable on foot from the entrance pocket. The route is the
  classic tour chain **Center → EAST (43,15-17) → Area 1 (NW doors (8,9-11)) → Area 2
  (S doors (20-22,34)/(10-12,34)) → Area 3 West** — billed from pret map.json + live
  probe, wired into recon_safari (strike 12). Return leg REVERSES the chain (West's
  (40,26-28) doors land on the shelf).
- The script is FLAG-IDEMPOTENT and boots from canonical each run — a dead/killed strike
  just relaunches (`.venv\Scripts\python.exe -u pokemon_agent\recon_safari.py`).
- **Strike history (do NOT re-diagnose):** 1 = entry-trigger froze enter_to (fixed:
  deliberate step onto (4,3) + A-drain pays); 2 = grass-free planner reads no_route where
  GRASS IS THE ROAD (fixed: step_warp on grass-inclusive walk_path_to); 3 = st.in_battle
  BLIND to safari battles — gate on GBATTLE_RES_PTR alone (fight_open); 4 = battles must
  not consume walk try-budget; 5 = ball economy — ONE attempt per species per run
  (thrown_species), and the pay script can auto-warp her into the Center (committed
  0c53e3c); 6 = healthy, killed by shift-1 close mid-grass at 94s; 7/8 = the SHORE
  TREADMILL, probe-diagnosed (night shift 2): safari-pond water reads RAW COLLISION 0
  (gated by behavior, not collision) → BFS planned across the pond, the blocked step's
  nudge landed in grass, the battle branch skipped dead-marking → (35,17)↔(34,17)
  forever. KILLED GENERALLY: Grid.walkable now excludes Grid.water (travel.py — surf
  planners must OR water back in), + walk_path_to dead-marks battle-interrupted failed
  steps. recon_cinnabar.py (DRAFT, unrun) already ORs water back in via sea_ok.

**THE CHAIN AFTER SAFARI (bank each, keep climbing):**
1. **Teach SURF → LAPRAS slot 5** (hm_teach.HMTeach.teach('surf',5) — the standing vehicle;
   forget a Mist-class move). Then HM04 Strength → whoever fits (Venusaur? check compat via
   the ROM table @0x08252BC8 — never hand-tables).
2. **Surf actuation** — fm.surf_edge_adjacent exists in field_moves; the WATER STEP is the
   next capability seam (Route 19/20 south of Fuchsia). Build general: face water edge →
   A → YES → riding state.
3. Route 19/20 surf south-west → **CINNABAR** → Mansion (SECRET KEY, gym locked without it)
   → **BLAINE = BADGE 7**.
4. **GIOVANNI, Viridian = BADGE 8** (gym needs spin_nav.py — built, NOT wired into
   travel/campaign; the maze is spin tiles).
5. Route 22/23 → **VICTORY ROAD** (Strength boulders — HM04 from this safari strike) →
   **E4 → CREDITS.**

**KNOWN GAPS (owed, carried):** Venusaur still named "AAAAAAAAAA" (Name Rater, Lavender —
cheap soul beat when passing); bench dead weight (Ekans L15 + boxed Mankey L10 — E4 needs a
real squad; Lapras helps); spin_nav.py unwired (Viridian Gym blocker); questline can't CLIMB
interiors generically; safari catches are silent RAM catches — no judged-catch narration.

**SOUL-DEBT (flag while passing):** LAPRAS joined unmet, unnamed, via a PC menu — her first
fielded moment (the first Surf!) deserves the roster-bond beat. Gary 4W-2L is live narrative.

**WORKING-TREE LAW:** kira/bot.py, kira/brain/cost_tracker.py, kira/modes/vn_autopilot.py,
kira/senses/vision_agent.py + gemini_vision.py = **Jonny's Gemini-vision WIP — NEVER commit,
never sweep into pokemon commits.** Stage pokemon files by explicit path only.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only (grid-dump/battle-trace before believing any wall),
frontier-first NEXT_SESSION.md rewrites (BEFORE long strikes + at every bank). GO.

---

WATCH STATUS: canonical bank is CLEAN (fuchsia_south — badge 6 on her jacket, Lapras
finally at her side, standing in Fuchsia with the Safari Zone gate to the north); she is
about to run the Safari for Surf + Strength, the badge-7 unlock; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
