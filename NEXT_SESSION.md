# NEXT_SESSION — THE STANDING NIGHT-TRAIN MANDATE (rewritten 2026-07-08 night shift 10)

## ⚡ SHIFT 10 STATE (current)
0. **SHIFT-9 E4/SURF RE-GRADE WAS KILLED AT HANDOVER** (log `descent_regrade_e4surf_shift9.log`).
   Shift 10 read the corpse + ran SEVEN instrumented re-grade rounds. THE FINDS, all committed:
   (a) **TRANSPOSED MapConnection read** in `_reenter_at_column` — (mapNum,mapGroup) vs our
   (group,num): Route 1 (3,19) became (19,3), every re-entry leg no_routed forever. f59484a.
   (b) **Center-less regroup floor**: Pallet has no CITY_PC_DOORS entry → `_regroup_walk`
   looped 'stuck' ~25 decisions. Now rides the world graph one hop per call toward the
   nearest known Center city. f59484a.
   (c) travel start line logs `surf-capable` (the silent can_surf gate, rule 3). f59484a.
1. **SURF EDGE-MOUNT (1bf0329):** at a map edge whose past-border tile is WATER, an unmounted
   D-pad press is EATEN — the mount ceremony only lived on the mid-path land→water step.
   Now mounts FIRST at the edge, then crosses.
2. **N/S CONNECTION BAND (9699ab6):** "N/S flip at the edge, no overlap" is only true for
   full-width land connections. The neighbour row IS readable one tile past the border
   (probe: Pallet y=20 reads Route 21 water at cols 7-10 — the real bay; col 12, where she
   blind-pressed ×6, is closed; y=-1 reads the Route 1 gap at cols 12-13). N/S goals now
   gate to crossable columns exactly like E/W; empty band = old behavior.
3. **ROUND 3 PROOF: THE SEA IS CROSSED** — Pallet → Route 21 live ("south connection band
   cols", "SURF MOUNT"); terminal dead-airs DEAD. Remaining wedges = two bounces, one class:
4. **WATER-START REACHABILITY LAW (bb6f125):** a land-only BFS dies AT ITS OWN START on a
   surf-mounted (water) stand — grass read unreachable on every sea route (the R21
   north↔south wander ping-pong ×16 ticks), and _door_passthrough's door-reach silently
   skipped the Seafoam entrances across open water. Fixed at _reachable_grass,
   catch_one._reach, _door_passthrough + new `_surf_usable()` (fail-closed). Round 4: E4 ran
   76 battles in 7 decisions — she finds and grinds sea-route grass now. NOTE: Route 20 is
   HONESTLY split at Seafoam (current tiles aren't SURFABLE_WATER; the billed road runs
   THROUGH the islands via:pass — interior puzzle is strike-only, not general).
5. **MAP-BOUND WANDER (3d03ed2) + COORD-FRAME LAW (1f442a1):** the "world tour" class
   (Indigo→Viridian→Pewter→Cerulean→Pallet inside ONE wander_catch) = grass waypoints
   scanned on one map chased across map flips (drift even counted as progress). catch walks
   are now bound to the scanned map, AND travel coord-mode legs END at any map transition
   ('transitioned' — an arrive_coord is the starting map's frame; warp-ride callers already
   check map_id after and are unaffected).
6. **PARTIAL-VOID / IMPOSSIBLE-STAND detector (f59484a + tick-top commits):** the QW-4 family
   can wear a PLAUSIBLE map id while she "stands" on a fully-enclosed col=1 tile
   (`_world_lost` only sees the full title signature). `_impossible_stand()` fires in
   regroup's stuck-path AND at the tick top (gated on wedge growth; streak 1) →
   `_void_recover()`. VERIFIED LIVE twice: "world restored at (3,40)@(13,45)" (round 5) and
   at PEWTER (round 7). **ROOT-CAUSE THREAD (open): the desynced coord READ keeps x and
   shifts y DOWN** — Pallet real ~(8,19) read (8,6) Δ13; Pewter real (20,25) read (20,4)
   Δ21. A rebase/offset artifact after certain map flips. Grep future logs for the tripwire
   line and diff read-vs-restored y. Shift-9's "(3,0)→(0,0)→(1,80)" transitions (pre-detector
   code) remain unexplained; if (0,0)/(1,x) transitions no warp explains reappear, reopen.
7. **ROUND 8 (FINAL, shift10h.log): BOTH ARCS PASS — "riskiest arcs: none".** E4 twedge 11
   (trend across the fix chain: 26→22→32→27→21→11), SURF_TAUGHT PASS steady (14). The Pewter
   y-shift desync recurred 3× in-window; the tick-top tripwire recovered all three (bounded
   budget 3 — a 4th in one window abandons loud, which is correct). The last two flagged
   descent arcs are green; DESCENT_PREGRADE.md holds the 2-arc table (the full 15-arc sweep
   regenerates it).
8. **NEXT (in order):** (a) read round 8, commit anything left; (b) level-up early beat
   verify — `.venv\Scripts\python.exe -u pokemon_agent\recon_lvlbeat_verify.py` (default
   banked_HM05; arms a real level-up by -1 on the lead's level byte); (c) Viridian Gym spin
   maze — banked_GIOVANNI/stage_giovanni are both POST-badge (banked 13:14 after the badge-8
   PNG); the spin TILES persist post-badge, so an honest locomotion verify = spawn banked_E4/
   postgame → walk into Viridian Gym → cross the maze with the REAL loop (spin_assist);
   (d) if green: FULL 15-arc sweep on tonight's code
   (`.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120`, ~35 min) →
   clean table → ranked spot-watch list for Jonny.

## STANDING TRUTHS (carry forward)
- Re-grade command: `$env:DESCENT_ARCS='banked_E4,banked_SURF_TAUGHT';
  .venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120 *> logs\longrun\<log>`.
- Full 15-arc sweep snapshot (older code): `logs\longrun\DESCENT_PREGRADE_full_shift7.md` —
  every flagged arc since re-graded PASS except the E4 work above.
- Standing probes from tonight: `recon_pallet_bfs_probe.py` (impossible-stand/desync
  diagnoser + overlap-row reader), `recon_surfgate_probe.py` (can_use gate off any bundle),
  `recon_r21_crossing_probe.py` (sea-crossing repro + frame dumps → %TEMP%\longrun\r21_probe).
- venv python is a shim — TWO PIDs per launch; never taskkill your own run. SINGLE-RUN LAW:
  one emulator recon at a time.
- Grade harness is READ-ONLY on bundles; banked_CREDITS excluded (mid-ceremony grenade).
- Known general gaps: Viridian spin maze (spin_assist un-verified live); level-up early beat
  WIRED not verified; evolution early beat unbuilt; Seafoam interior = strike-only.
- ⚠️ PS 5.1 mangles this file's UTF-8 via Get-Content/Set-Content round-trips — edit it with
  the Write/Edit tools only.

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
(Cerulean Cave open; her stated want: catch Mewtwo — and the sea roads south of Pallet now
actually carry her). Pop-in = `python pokemon_agent/watch.py` → spawn 'postgame'
(or --canonical, safe). Her tick context opens grounded: "You're at home — indoors, inside
a building."
