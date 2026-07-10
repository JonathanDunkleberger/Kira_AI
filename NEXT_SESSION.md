# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 8 IN-FLIGHT) — READ FIRST

## SHIFT-8 BANKED (2 commits): SNORLAX wake VERIFIED e2e + critical-heal FREEZE broken -> she now
## drives Route 12 -> 13 -> 14 -> 15 to Fuchsia's doorstep. FRONTIER = enter Fuchsia -> Koga (badge 5).

**Commit fd22167 — Route 12 SNORLAX wake strike (badge-5 gate).** Post-Flute she walked to the
Snorlax and STALLED at (3,30)@(12,0) (run7). Fixed: (1) bill the snorlax gate at Route 12 "3,30"-south
too (frlg_gates.json) so the wake questline opens from Route 12 (fixture + rule-17 resume-safety);
(2) fire the strike FIRST in `_run_questline_step` for DOOR-LESS steps (the snorlax has no door hint)
before the ANCHOR-FIRST routing routes her back to Lavender. `snorlax_strike.py` = in-loop port of
recon_snorlax.py. VERIFIED e2e (snorlax_face.state): questline opens -> STRIKE -> pass gate -> play
Flute -> wake+beat -> FLAG_WOKE_UP_ROUTE_12_SNORLAX set -> crossed to Route 13.

**Commit a39e9c2 — critical-heal FREEZE breaker (Route-12 gatehouse split).** Post-snorlax she's hurt on
Route 13; the survival floor collapses to ONLY 'heal', but the heal-return can't reach a Center — Route
12 is INTERNALLY split by a gatehouse (both halves = map (3,30)), so the map-granular next_hop returns a
plain (3,30)->(3,4) north edge whose band is UNREACHABLE from her south-section feet; heal abandons ->
hard FREEZE at (3,31)@(63,0). Fix (mirrors the non-critical STRAND GUARD): mark a map heal-dead when a
'heal' pick returns 'stuck' with zero movement, and at CRITICAL severity on a heal-dead map fall through
to offer head_to_gym (push to the next town's Center) instead of freezing; a blackout en route respawns
her healed. VERIFIED e2e (snorlax_done.state): freeze broke -> she drove Route 13 -> 14 -> 15 (healing
via a Lavender blackout-respawn en route) to Route 15's WEST edge = Fuchsia City ("a Mart, a Pokémon
Center", one step away). recon_healcross.py = the gatehouse-split diagnosis probe.

## FRONTIER (PRECISELY DIAGNOSED — fix half-designed below): CROSS THE ROUTE 15 GATE INTO FUCHSIA.
She now drives all the way to Route 15 but STALLS at its WEST end: the coastal road Route 12->13->14->15
is proven crossable (s8_run3), but from Route 15's far-east she can't walk the west map-edge to Fuchsia
(3,7) — "no clean path ... genuine wall/zone gap" — so head_to_gym's PASSTHROUGH diverts her into the
Route-15 internal GATE building (24,0) and she LOOPS there forever (s8_run3 STALL at (24,0)@(9,10)).

**GROUND TRUTH (recon_fuchsia_gate.py — all doors of (24,0) tested by pick):**
  - (24,0) is Route 15's internal gate (crosses a Route-15 barrier). ALL FOUR cardinal doors exit to
    Route 15 (3,33): WEST doors (1,6)/(1,7) -> (3,33)@(9,11) [Route 15 WEST side, toward Fuchsia];
    EAST doors (11,6)/(11,7) -> (3,33)@(16,11) [Route 15 EAST side, where she came from].
  - (9,10) is a STAIRS warp (behavior 0x6c) up to a DEAD-END upper floor (24,1) whose only door
    (10,9) leads straight back to (24,0). That stairs<->back loop is the ping-pong.
  - So the crossing SHE NEEDS: take a WEST door (1,6)/(1,7) -> Route 15 west side (9,11) -> continue
    west to the Fuchsia (3,7) edge. She instead keeps taking the (9,10) stairs.

**ROOT CAUSE + FIX LOCATION:** `_door_passthrough` candidate ranking, `_dest_rank` at campaign.py:1534.
It ranks UNVISITED interiors (rank 1) ABOVE known overworld ground (rank 2) — "unvisited interiors beat
known ground" — so the unexplored dead-end upper floor (24,1) via the (9,10) stairs OUT-RANKS the
Route-15 exit doors, and she climbs the dead-end first every tick. NOT a safe one-liner: you can't just
demote stairs-warps, because legit connectors (the Underground Path huts, line ~1618) ALSO cross via
stairs. The right fix distinguishes a DEAD-END interior (dest has <=1 outbound door = only the entry)
from a THROUGH interior (dest has a further exit) and demotes the dead-end BELOW known overworld doors,
OR marks a proven dead-end so it isn't re-preferred each tick (per-tick _door_passthrough resets `tried`;
only PROVEN crossings persist via `_pt_known`, dead-ends don't — so add a `_pt_deadend` memo per (map,door)).
Then verify: recon_longrun snorlax_done.state -> she should exit (24,0) WEST -> Route 15 west -> Fuchsia ->
enter the gym -> Koga.

**BOOT the frontier directly (she's stalled INSIDE the gate):**
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 20`
(states/workshop/fuchsia_gate.state = s8_run3 STALL, inside gate (24,0)@(9,10), Venusaur L51, badges=4.)
OR from further back: `recon_longrun.py snorlax_done.state 25` (post-snorlax on Route 13, drives the
whole 13->15 leg then hits the gate). recon_fuchsia_gate.py re-probes the door destinations on demand.

**KOGA (badge 5, Fuchsia gym):** poison/confusion, L37-43. Her bench is L9-15 (huge gap under Venusaur
L48) and she has NO clean psychic/ground answer — expect a grind/team-build need. Venusaur L48 with STAB
grass + status may brute-force the juniors; watch for the gym-entry (any Fuchsia gym approach gate) and
the leader fight. Also: Fuchsia has the Safari Zone (HM sources) + a Pokémon Center (heal now reachable
forward, which retro-fixes the heal-dead mark once she arrives).

## AFTER KOGA: badge-6 chain. Tea is ALREADY obtained (Saffron guards open) -> Saffron/Silph Co ->
Sabrina (badge 6, Marsh). Then Cinnabar (Blaine, badge 7) -> Viridian (Giovanni, badge 8) -> Victory
Road -> Elite Four -> CREDITS.

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages persistence to G:/temp/longrun/stage (WIPED each run start); world_model+strat
  load from CANONICAL, soul from the boot bundle. Canonical Champion save NEVER touched. Banks to
  G:/temp/longrun/banked_<OUTCOME>; snapshot OUT before relaunching (STAGE is wiped).
- Fixtures (states/workshop/): surge_done, rt_mouth, flash_done, erika_done (badge 4 @ Celadon),
  snorlax_face (Route 12 @ Snorlax — pre-wake), snorlax_done (post-wake, Route 13),
  **fuchsia_gate (stalled INSIDE the Route-15 gate (24,0)@(9,10) — THE FRONTIER)**.
- Topology south of Lavender: Lavender(3,4) -> Route 12(3,30, SNORLAX, internal gatehouse-split) ->
  Route 13(3,31) -> Route 14(3,32) -> Route 15(3,33) -> Fuchsia City(3,?). Route N = map (3,18+N).
- Snorlax facts (game-knowledge layer, snorlax_strike.py): Route 12=(3,30); FLAG_WOKE=0x253/595; body
  (14,70); north gate doors ((14,15),(15,15)). Route 12 internal gatehouse = building (23,0).

## DURABLE PATTERNS
- STRIKE-REGISTRY (shift 7-8): dungeon/blocker rituals ported as in-loop strike modules (hideout/tower/
  snorlax_strike), dispatched from `_questline_strike` keyed by step.success. Dungeon errands fire at
  the door-hint hook; face-a-blocker errands (snorlax) fire early in `_run_questline_step` (door-less).
- INTERNAL MAP-SPLIT class (Route 12 gatehouse, shift 8): a map whose halves share one id but connect
  ONLY through a gatehouse — the map-granular world graph is blind to the split, so next_hop returns a
  cardinal edge the feet can't reach. Heal now DETECTS this (no-op stuck -> heal-dead -> push forward).
  A fuller fix (multi-hop gatehouse traversal in the heal router) is deferred; forward-push covers it.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead is at badge 4, post-Snorlax,
standing at the Route-15 gate ONE crossing from Fuchsia — next fix = the gate-passthrough dead-end-stairs
demote, then Koga (badge 5). Pop-in = `python pokemon_agent/watch.py`.
