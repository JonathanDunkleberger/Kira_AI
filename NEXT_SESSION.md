# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 9 IN-FLIGHT) — READ FIRST

## FRONTIER = CROSS THE ROUTE 15 GATE INTO FUCHSIA -> Koga (badge 5).
## She stalls one crossing from Fuchsia. Shift-9 re-diagnosed the REAL cause (shift-8's
## NEXT_SESSION pointed at the WRONG code) and applied the load-bearing fix. VERIFY IN FLIGHT.

### CORRECTED ROOT CAUSE (shift 9 — proven via recon_longrun fuchsia_gate.state, 203s STALL)
The stall is NOT `_door_passthrough` / `_dest_rank` (shift-8's theory). It is the **critical-heal
freeze**, in `_route_action`'s heal branch (campaign.py ~8100):
1. She boots INSIDE gate (24,0), critically hurt (Venusaur 43/160 = 27% < 30% crit) -> options
   pruned to `['heal']` only.
2. `heal_nearest()` exits to Route 15 (3,33) EAST side (via door (11,6)), correctly proves
   "no graph route to any known Center from (3,33)" (Fuchsia is WEST across the internal split),
   then falls back to a nonsensical **Viridian** return that PING-PONGS through the gate's two
   floors (24,0)<->(24,1) for 20 legs and returns `"stuck"`.
3. The shift-8 heal-dead mark required `_hp1 == _hp0` (no movement). The ping-pong MOVED her ->
   heal-dead NEVER marked -> the shift-8 freeze breaker never armed -> critical re-picks `'heal'`
   forever. Hard freeze at (24,0)@(9,10).

### FIX APPLIED (shift 9, campaign.py ~8110, UNCOMMITTED until verified)
Mark heal-dead on ANY `heal_nearest()=="stuck"` (an honest "no Center reachable" proof — movement
or not), and mark BOTH the start map AND the map the failing router left her on. Then the freeze
breaker fires: critical + heal-dead -> offer head_to_gym -> she pushes WEST toward Fuchsia's Center
(a blackout en route just respawns her healed).

### VERIFY IN FLIGHT (the run that decides this shift)
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 25`
Watch for: after the heal 'stuck', next tick opts should include `head_to_gym` (freeze breaker
fired). THE OPEN QUESTION the run answers: does head_to_gym then CROSS THE GATE WEST to Fuchsia,
or take the (9,10) dead-end stairs? If it takes the stairs, the SECONDARY fix is the gate
passthrough dead-end demote (see below) — but the freeze breaker is the load-bearing half.

### SECONDARY (only if the run shows head_to_gym looping the (9,10) stairs)
GROUND TRUTH (recon_fuchsia_gate.py): gate (24,0) doors -> WEST (1,6)/(1,7) -> (3,33)@(9,11)
[Route 15 WEST, toward Fuchsia]; EAST (11,6)/(11,7) -> (3,33)@(16,11) [east, where she came];
(9,10) = STAIRS up to DEAD-END floor (24,1) whose only door loops back. `_dest_rank`
(campaign.py:1534) ranks unvisited interiors (rank 1) ABOVE known ground (rank 2), so the dead-end
upper floor out-ranks the exit doors. Fix = demote a DEAD-END interior (dest has <=1 outbound door)
below known overworld doors, or a `_pt_deadend` memo so a proven dead-end isn't re-preferred.

### BOOT the frontier directly
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 25`
(states/workshop/fuchsia_gate.state = STALL inside gate (24,0)@(9,10), Venusaur L51, badges=4.)

### KOGA (badge 5, Fuchsia gym): poison/confusion L37-43. Bench L9-16 under Venusaur L51 — expect a
grind/team-build need; Venusaur STAB grass + status may brute-force juniors. Fuchsia has a Center
(heal reachable forward once she arrives) + Safari Zone (HM sources).

### AFTER KOGA: Tea already obtained (Saffron open) -> Sabrina (badge 6) -> Blaine (7) -> Giovanni
(8) -> Victory Road -> Elite Four -> CREDITS.

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages to G:/temp/longrun/stage (WIPED each run start); banks to
  G:/temp/longrun/banked_<OUTCOME>; snapshot OUT before relaunching. Canonical Champion NEVER touched.
- Fixtures (states/workshop/): surge_done, rt_mouth, flash_done, erika_done, snorlax_face,
  snorlax_done, **fuchsia_gate (THE FRONTIER — stalled inside the Route-15 gate)**.
- Topology: Lavender(3,4) -> Route12(3,30,Snorlax) -> R13(3,31) -> R14(3,32) -> R15(3,33,internal
  gate (24,0)/(24,1)) -> Fuchsia(3,7). Route N = map (3,18+N).

## DURABLE PATTERNS
- HEAL-DEAD-ON-STUCK (shift 9): a `heal_nearest()=="stuck"` is the honest "no Center reachable"
  signal; mark heal-dead regardless of movement (the Viridian-fallback ping-pong defeats no-move
  guards) so the critical-heal freeze breaker (offers head_to_gym) can arm.
- INTERNAL MAP-SPLIT class: a route whose halves share one map id but connect only through a gate
  building — the map-granular world graph is blind to the split; from the wrong side, edge/center
  bands read UNREACHABLE. The freeze breaker's forward-push covers heal; head_to_gym must own the
  actual gate crossing.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead at badge 4, post-Snorlax,
frozen INSIDE the Route-15 gate one crossing from Fuchsia — shift-9 fix (heal-dead-on-stuck) applied,
verifying in flight. Pop-in = `python pokemon_agent/watch.py`.
