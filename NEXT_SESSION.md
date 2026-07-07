# NEXT_SESSION — resume prompt (write date 2026-07-07, attended intervention after shift 11)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 NIGHT SHIFT #9/#10 blocks first. Context: the night
loop ran 11 shifts; shift 8 died at the context wall MID-STRIKE without closing, so this file
sat at shift-7 vintage while shifts 9-11 flew off it — repaired attended 07:45. The night
contract now demands frontier-first rewrites BEFORE launching long strikes (night_shift.ps1
preface point 3). Never trust this file over STATE §0 + NIGHT_REPORT.md if they disagree.

**CANONICAL = saffron_reach: Saffron City (3,10)@(47,13) — AT the gym door, badges 5
(Boulder/Cascade/Thunder/Rainbow/Soul), TEA in bag, Venusaur L55 full HP, sanctity VALID**
(backup pre_saffron_reach_backup_20260707_060438). Party is FULL (6) — the Silph Lapras gift
transfers to Bill's PC.

🏢 **LIVE OBJECTIVE: FINISH + BANK THE SILPH CO STRIKE — ~95% cracked.**
`pokemon_agent/recon_silph.py` boots from canonical and runs the whole tower in ~2-4 min at
max speed. FIRST MOVE: check whether the attended strike16 already finished —
`logs/longrun/silph_strike16.log` + `%TEMP%\longrun\banked_SILPH` + `%TEMP%\longrun\
stage_silph`. If `saffron_free` (flag 0x3E, Giovanni beaten) fired: **bank + promote
silph_cleared via promote_bank.py** (full sanctity bundle — the stage saves carry Gary #6 and
Lapras), then go straight to Sabrina. If not, the log names the wall; every wall in this
tower so far fell to a probe (recon_silph_probe*.py / recon_silph7f_probe.py patterns).

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

🔮 **THEN: SABRINA = BADGE 6 (longrun goal 0x825).** 0x3E frees Saffron and unblocks the gym
door canonical is standing AT. GymSpec billed (shift 8). Interior = the teleport-pad maze —
pads are WARPS, the Silph ride_pad/enter_to primitives should carry. Longrun at the gym,
iterate per wall, bank sabrina_badge6.

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

WATCH STATUS: canonical bank is CLEAN; she is at the Saffron gym door with badges 5, the
Silph strike running from it headless; pop-in = `.venv\Scripts\python.exe -u
pokemon_agent\play_live.py --resume --free-roam`. Attended strike watch: set `WATCH=1` then
`.venv\Scripts\python.exe -u pokemon_agent\recon_silph.py`.
