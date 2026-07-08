# NEXT_SESSION — THE STANDING NIGHT-TRAIN MANDATE (rewritten 2026-07-08 night shift 10, pre-re-grade)

## ⚡ SHIFT 10 STATE (current)
0. **SHIFT-9 E4/SURF RE-GRADE WAS KILLED AT HANDOVER** (log `descent_regrade_e4surf_shift9.log`,
   495KB, no grade rows written — DESCENT_PREGRADE.md still shows the 04:48 4-arc table).
   Shift 10 read the corpse in full. THE FINDS (all diagnosed from the log + live probes):
   (a) **TRANSPOSED MapConnection read** in `_reenter_at_column` (campaign.py) — (mapNum,
   mapGroup) vs our (group,num): Route 1 (3,19) became (19,3) → every re-entry leg no_routed
   forever (the alternating (19,3)/(39,3) storm at Pallet). FIXED.
   (b) **Center-less regroup floor**: Pallet has no CITY_PC_DOORS entry + no connector fired →
   `_regroup_walk` looped 'stuck' ~25 straight decisions (the terminal dead-air). FIXED: rides
   the world graph one hop per call toward the nearest known Center city (Pallet → Viridian).
   (c) **PARTIAL-VOID / IMPOSSIBLE-STAND class (new QW-4 variant)**: the window read her
   "standing" at Pallet (8,6) — probe `recon_pallet_bfs_probe.py` (standing tool) proves that
   tile is col=1 with ALL FOUR neighbours col=1: physically impossible. A desynced/rebooted
   world wearing a plausible map id (map (3,0), party 6, badges 8) — `_world_lost` only sees
   the full title signature (map (0,0)+party 0). Also seen: tick-1 garbage party read ("Persian
   0/106, 5 down" then "already full" 60s later) and a spontaneous (0,0)→(1,80) "transition"
   mid-leg (reboot → in-game save reload). FIXED (detector): IMPOSSIBLE-STAND tripwire in
   regroup's stuck-path (overworld-gated, enclosed-stand signature) → `_void_recover()`.
   WHAT'S NOT FOUND: the reboot TRIGGER itself (same open question as shift 1's QW-4).
   (d) travel start line now logs `surf-capable` — the silent can_surf gate is loud (rule 3).
   Probes also confirmed: can_use(surf)=True on banked_E4; Pallet water BFS paths fine
   (south crossing len 15) — the water layer is HEALTHY; the storm was the desync, not surf.
   ALL COMMITTED: **f59484a** (+ shift-9's regroup map-change truth folded in).
1. **RE-GRADE ROUND 1 (shift10.log, killed deliberately) FOUND THE REAL E4 DEFECT — THE
   SURF EDGE-MOUNT GAP, FIXED 1bf0329:** the dead-air reproduced with REAL geometry (not the
   desync): she stands at Pallet (12,19), the south EDGE row, surf-capable — and travel's
   at-edge branch blind-pressed S ×6 into the sea, all EATEN (the mount ceremony only lived
   on the MID-PATH land→water step, never at the edge-cross press). Every south-sea travel
   died there. Round 1 also proved: `surf-capable` logging live ✓; the Center-less regroup
   fallback FIRES ("riding toward Cinnabar Island") ✓ — it failed only on the same mount gap.
   Fix: at-edge + past-tile-is-water + unmounted → run the shoreline mount ceremony first.
2. **IN FLIGHT: re-grade round 2 on 1bf0329** — log
   `logs\longrun\descent_regrade_e4surf_shift10b.log` (~5 min). If dead fresh:
   `$env:DESCENT_ARCS='banked_E4,banked_SURF_TAUGHT'; .venv\Scripts\python.exe -u
   pokemon_agent\recon_descent_grade.py 120 *> logs\longrun\descent_regrade_e4surf_shift10b.log`.
   EXPECT: "SURF EDGE-MOUNT" in the log; E4 roams south to Cinnabar/R21 freely; dead-air gone.
3. **NEXT (in order):** (a) read the re-grade, fix/commit what's left; (b) level-up early beat
   verify — `.venv\Scripts\python.exe -u pokemon_agent\recon_lvlbeat_verify.py` (default
   banked_HM05; arms a real level-up by -1 on the lead's level byte); (c) Viridian Gym spin
   maze (banked_GIOVANNI is POST-badge-8 so descent arcs never cross it; honest verify needs a
   pre-badge state — check `%TEMP%\longrun\giovanni_probe` first); (d) if green: FULL 15-arc
   sweep on tonight's code (`.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py
   120`, ~35 min) → clean table → ranked spot-watch list for Jonny.
4. **OPEN QUESTION (characterized, not solved): the REBOOT TRIGGER.** Something mid-window
   reboots the game to title (QW-4 family). Tonight's detector catches the aftermath; the
   cause is unfound. If a re-grade reproduces it, look for a MAP TRANSITION to (0,0) or (1,x)
   that no warp explains.

## SHIFT 9 STATE (background — landed + verified)
- 4-arc re-grade landed 04:48: SCOPE/SURF_TAUGHT/SABRINA → PASS (benign-still fix verified
  live); E4 WARN → diagnosed → the two travel fixes committed 3ddfaf9 (gate-locked city
  travel via _next_step_rideable + regroup floor); EDGE-ROW RETRY 9917b8d; NO-MOVE guard
  honesty 3c3522e. ROCKTUNNEL re-grade FAIL→PASS on e2d5f9a (nickname-keyboard class killed;
  guard positive-path = logic-only, not hit live).

## STANDING TRUTHS (carry forward)
- Full 15-arc sweep snapshot (pre-e2d5f9a code): `logs\longrun\DESCENT_PREGRADE_full_shift7.md`
  — 10 PASS / 4 WARN / 1 FAIL at that code level; every flagged arc since re-graded PASS
  except tonight's E4 work.
- venv python is a shim — TWO PIDs per launch; never taskkill your own run. SINGLE-RUN LAW:
  one emulator recon at a time.
- Grade harness is READ-ONLY on bundles (persistence no-op'd); banked_CREDITS excluded (the
  mid-ceremony grenade).
- Known general gaps still open: Viridian Gym spin maze (spin_assist un-verified live);
  level-up early beat WIRED not verified; evolution early beat unbuilt.

## BURN DISCIPLINE (CEO directive — density is mandatory)
1. THE VALUE LINE in every survey. 2. Bounded recon. 3. Depth over spread. 4. No idle-grind.

## NEEDS-EYES LEDGER (the ONLY loop-stoppers)
1. Descent spot-watches — pick from DESCENT_PREGRADE.md's ranked list.
2. Round-3 throwaway grading (F-5 bar) — after the F-1 feel-take is green.
3. Prefetch A/B B-side — bot restart w/ flag + one conversation (2 min of Jonny).
4. F-9 tic governor feel — rides any watched take.
5. Cohost smoke session (G-4 exit, 20 min).
6. Tri-mode session (Phase I exit, 15 min).
7. Clipper output review (K exit).
8. Final showtime sign-off — the Kira-timeline launch is HIS press, always.

## WORKING-TREE LAW
kira/* = Jonny's + approved core work under loud-log law. Prune targets (C-4 leftovers):
untracked recon_* fleet remainder, repo-root states/, verify_dialogue.py. Throwaway
sandboxes: `go.py --clean-throwaways` + `watch.py --clean`.

---

WATCH STATUS: canonical bank is CLEAN — the TRUE post-game: the Champion at home in Pallet
Town ((4,0)@(4,8), full healthy party — Venusaur L95, Persian, Fearow, Raticate, Ekans,
Lapras — badges 8, player in control; sanctity VALID). She is at home, victory lap ahead
(Cerulean Cave open; her stated want: catch Mewtwo). Pop-in = `python pokemon_agent/watch.py`
→ spawn 'postgame' (or --canonical, safe). Her tick context now opens grounded: "You're at
home — indoors, inside a building."
