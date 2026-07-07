# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #1 of the new loop, pre-strike rewrite)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

⚔️ **IN FLIGHT AT WRITE: the SAFFRON GYM STRIKE (recon_sabrina.py, log
logs/longrun/sabrina_strike1.log).** Run8 (longrun) confirmed the billed wall in 79s: the
gym interior (14,3) is warp-partitioned — travel BFS = no_route from the entrance pocket,
campaign's gym handler false-latches "juniors cleared" and A-mashes Sabrina from 11 tiles
away. The strike carries the fix: **pad_plan() — a runtime PAD-GRAPH ROUTER** (warps whose
dest is the current map are pads; dest_warp_id indexes the landing tile; flood-fill
walk-regions, meta-BFS with pad rides as edges — zero hardcoded room sequence, ports to any
teleport maze). FIRST MOVE: read that log's END. If badge 0x825 → bank is
%TEMP%/longrun/banked_SABRINA → `python pokemon_agent/promote_bank.py <bank> sabrina_badge6`
→ next chain link (Surf/HM03, below). If failed → snap frames are in
%TEMP%/longrun/sabrina_probe/, fix, relaunch sabrina_strike2.

🏢✅ **SILPH CO CLEARED + PROMOTED (attended strike16, 07:58 — 359s end-to-end, exit 0):**
Card Key → 9F heal → pad chain → **GARY #6 BEATEN** → **LAPRAS banked (flag=True, → Bill's
PC, party full)** → 11F south door (the pad-landing column is sealed; the door fallback is
the real approach) → **GIOVANNI BEATEN (0x3E saffron_free=True)** → **MASTER BALL from the
president (flag 0x250=True)** → walked out + healed. Log: logs/longrun/silph_strike16.log.

**CANONICAL = silph_cleared: Saffron City (3,10)@(33,31) — outside Silph, badges 5, SAFFRON
FREE, Master Ball in bag, Venusaur L57 full HP, sanctity VALID, round-trip verified**
(backup pre_silph_cleared_backup_20260707_075833). Party: Venusaur L57 / Persian 37 /
Fearow 35 / Raticate 31 / Ekans 15 / Mankey 10. LAPRAS is in Bill's PC.

🔮 **LIVE OBJECTIVE: SABRINA = BADGE 6 (longrun goal 0x825).** Saffron is free and the gym
door is unblocked. GymSpec billed (shift 8). Interior = the teleport-pad maze — pads are
WARPS, the Silph ride_pad/enter_to primitives should carry. Launch the longrun at the gym,
iterate per wall, bank sabrina_badge6.

**Already killed by shifts 9-11 + the attended pass (do NOT re-diagnose — STATE §0 has the
full postmortems; commits 1a0d16d, accd57e, 7634d33, 1fd4e74):**
- 1F lobby↔street livelock: the entrance mat (8,20) is a 0x65 DOWN-arrow warp — LAW: never
  step off a directional warp tile in its fire direction (fixed recon_silph + campaign).
- Card Key banks in-run ~30s: 9F pad (22,18) → 5F pocket → ball (22,21) from the EAST front.
- The tap-turn ghost (campaign `_step_to`): an 8-frame tap in an unfaced direction only
  TURNS her; re-tap the SAME key. Every "elevation-sealed tile" in the siege was this.
- walk_path_to = NPC-masked static BFS (grid BFS is NPC-blind; beaten trainers still stand).
- 9F door algebra: WMID doors (12-13,16-17) unseal the hostage HEAL woman (2,16) — a FREE
  repeatable full heal; WMID+WEST (2-3,10-11) unseal the 3F-pad corridor.
- The pad chain: 9F (9,4)↔3F (2,14) · 3F (13,14)↔7F (5,4) · 7F (5,8)↔11F (2,5). The 7F
  Gary/Lapras pocket is PAD-ONLY; Giovanni (6,11) is open from the 11F pad landing (2,5).
- **GARY #6 BEATEN in strike15** (grudge 4W-2L, in strat memory via the stage saves).
- The "7F wedge" = the LAPRAS GIFT NICKNAME KEYBOARD (frame-proven, wedge7f_frame.png):
  A-drains on "give a nickname? [YES]" opened the keyboard, which ate all overworld input —
  the same bug class that once named Venusaur "AAAAAAAAAA". Fixed: engage(key="B") B-drains
  the gift end-to-end + a name_entry(b,"") START→OK escape hatch.

**THE CHAIN AFTER (bank each, keep climbing):** badge 7 = BLAINE, Cinnabar — needs SURF
(HM03, Safari Zone Secret House, Fuchsia) + a Surf-capable teammate (Venusaur can't —
**LAPRAS in Bill's PC is the natural surfer**: box-withdraw = competency #15, unbuilt, likely
the next real capability build) → badge 8 = GIOVANNI, Viridian → Route 22/23 → Victory Road
→ E4 → **CREDITS**.

**KNOWN GAPS (owed, carried):** Venusaur still named "AAAAAAAAAA" (Name Rater, Lavender —
cheap soul beat when passing); bench dead weight (Ekans L15/Mankey L10 — E4 needs a real
squad; Lapras helps); spin_nav.py not wired into travel/campaign (Viridian Gym needs it);
questline can't CLIMB interiors generically (two strikes prove the spec — promote the
state-machine pattern when a third dungeon bites); passthrough nearest-door vs multi-warp
ordering.

**SOUL-DEBT (flag while passing):** LAPRAS joined the family unnamed and unmet, straight to
the PC — her first withdrawal deserves the roster-bond beat. Gary 4W-2L is live narrative.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only (grid-dump/battle-trace before believing any wall),
frontier-first NEXT_SESSION.md rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (silph_cleared — Team Rocket just driven out of
Saffron, Master Ball in her bag, Lapras waiting in Bill's PC); she is outside Silph Co,
Sabrina's gym next; pop-in = `.venv\Scripts\python.exe -u pokemon_agent\play_live.py
--resume --free-roam`.
