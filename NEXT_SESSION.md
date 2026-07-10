# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 9 BANKED) — READ FIRST

## SHIFT-9 BANKED (1 commit d3b1d2b): ROUTE-15 GATE FREEZE BROKEN -> she crosses to Fuchsia,
## heals, and now REACHES KOGA's gym. FRONTIER = BEAT KOGA (badge 5) — a real team/movepool wall.

### WHAT SHIFT 9 FIXED (committed, VERIFIED e2e)
Shift-8's NEXT_SESSION pointed at the WRONG code (`_door_passthrough`/`_dest_rank`). The look-ahead
proved the real culprit was the **critical-heal freeze** in `_route_action` (campaign.py ~8110):
inside the Route-15 gate, hurt, `heal_nearest()` proves no Center reachable then its Viridian
fallback PING-PONGS the gate's two floors and returns "stuck"; the shift-8 heal-dead mark required
`_hp1==_hp0` (no move), the ping-pong MOVED her, so the freeze breaker never armed -> she re-picked
'heal' forever. FIX: mark heal-dead on ANY heal "stuck" (movement or not), both start+end maps.
VERIFIED: freeze broke -> head_to_gym took the gate's WEST door (1,6) -> Route 15 west -> Fuchsia
Center heal -> gym. The `_dest_rank` secondary fix was NOT needed.

### THE FRONTIER: BEAT KOGA (Fuchsia gym, badge 5). She reaches him and LOSES — precisely diagnosed
### via a battle-logged run (LONGRUN_BATTLE_LOG=1 recon_longrun fuchsia_gate.state). Root causes:
1. **MOVEPOOL WALL.** Venusaur L52's moves = Razor Leaf/Vine Whip (GRASS, resisted 0.5x by Koga's
   all-poison team) + Cut (normal, 50pwr, her ONLY neutral move) + Sleep Powder. The engine correctly
   sleep-locks the Self-Destruct koffing (NUKE-SLEEP) then attacks with Cut (best effective dmg) —
   but Cut is far too weak to burn Koga's tanks (Muk 130 HP) before status/chip drop Venusaur.
2. **DEAD BENCH.** Rattata L15 / Spearow L16 / Ekans L9 / Meowth L10 / Pidgey L13 — 25-40 levels under
   Koga (L37-43). Once Venusaur faints they come in one-at-a-time (Peck/Gust/Bite chip) and fold ->
   blackout. No backup, no coverage.
3. **NO ITEMS.** Items pocket = 3 misc items (ids 110/35/94), NO healing items, NO revive, NO coverage
   TM (checked the raw bag). So no stall-heal and no Ground/Psychic TM to teach Venusaur.
4. **GRIND-REACH WEDGE.** GYM-PREP wants to grind (target ~L46) but wedges: from Fuchsia she routes
   to Route 15 and gets stuck at (0,12) map (3,33) "genuine wall/zone gap" — the SAME internal-split
   that froze the gate: her reachable-grass pick / travel target sits across the gate barrier. NOTE:
   `_grass_target`/`_grass_via_graph` already have `_grass_unreach`/`_grind_dead` memos ("koga_run3/5")
   but they guard 'route' hops, NOT a 'here' grass tile that BFS calls reachable yet travel can't
   reach (the (0,12) case). That gap is the likely wedge.

### NEXT-SHIFT PLAN (ranked; the grind path is load-bearing but LONG)
A. **UNBLOCK THE GRIND (load-bearing, general).** Fix the Route-15 grass-reach wedge so GYM-PREP can
   actually level her. Likely fix: `_reachable_grass`/`_grass_target` 'here' tile needs a travel-
   reachability guard + a per-tile unreach memo (mirror the 'route' `_grass_unreach`), so a BFS-says-
   reachable-but-travel-wedges grass tile is abandoned and she routes to grass she can truly reach
   (Route 15 EAST via re-crossing the gate, or Route 13/14). CAUTION: `_grass_target` is used game-
   wide — minimal/guarded edit only (rule 12). Then a LONG grind run (Venusaur to ~L58-60 brute-forces
   Cut through Koga; or level the bench — slower). Verify with LONGRUN_BATTLE_LOG=1.
B. **POTIONS (faster if the Mart is reachable).** Fuchsia HAS a Mart. Wire a stock_up of Super/Hyper
   Potions into GYM-PREP (it currently only catch+grinds, never buys) so Venusaur L52 can stall-heal
   through the slow sleep-lock+Cut grind and win WITHOUT a big level grind. Check CITY_MART_DOORS has
   Fuchsia (may be UNMAPPED — the Cerulean-Mart-unmapped class). In-battle item actuation is
   historically flaky on the long core — verify it fires.
C. **COVERAGE TEAMMATE (best long-term, a detour).** A Ground type ERASES Koga's poison (Razor Leaf
   already hit a junior's sandslash x4 SE in the log). Dugtrio via Diglett's Cave (she's crossed it)
   or an Abra line. Big detour; last resort.

### BOOT THE FRONTIER
`LONGRUN_BATTLE_LOG=1 .venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 20`
(fuchsia_gate.state + the shift-9 fix now auto-crosses the gate -> Fuchsia -> gym -> Koga at ~108s
wall; battle log shows every move. No separate post-gate fixture needed — the crossing is fixed.)
recon_fuchsia_gate.py re-probes gate doors on demand.

### AFTER KOGA: Tea already obtained (Saffron open) -> Sabrina (badge 6) -> Blaine (7) -> Giovanni
(8) -> Victory Road -> Elite Four -> CREDITS.

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages to G:/temp/longrun/stage (WIPED each run); banks to banked_<OUTCOME>. Canonical
  Champion save NEVER touched. `LONGRUN_BATTLE_LOG=1` surfaces per-move engine logs (crucial for Koga).
- Fixtures (states/workshop/): erika_done, snorlax_face, snorlax_done, **fuchsia_gate (boots inside the
  Route-15 gate; the fix crosses it -> Fuchsia -> Koga)**.
- Koga team (FRLG): Koffing L37, Muk L39, Koffing L37, Weezing L43 — all POISON (grass resisted).
  Fuchsia gym is the INVISIBLE-WALL/spinner maze: junior trainers obj4/obj6 read "un-engageable
  (wandering/water-locked)" -> she gets NO junior XP, goes straight to Koga.
- Topology: Lavender(3,4) -> R12(3,30,Snorlax) -> R13(3,31) -> R14(3,32) -> R15(3,33, internal gate
  (24,0)/(24,1)) -> Fuchsia(3,7). Fuchsia has a Center + Mart + Safari Zone.

## DURABLE PATTERNS
- HEAL-DEAD-ON-STUCK (shift 9): a `heal_nearest()=="stuck"` is the honest "no Center reachable" proof;
  mark heal-dead regardless of movement (the Viridian-fallback ping-pong defeats no-move guards) so the
  critical-heal freeze breaker (offers head_to_gym forward-push) can arm. Both start+end maps marked.
- INTERNAL MAP-SPLIT class (Route 12/15 gates): halves share one map id, connect only via a gate
  building; the map-granular graph is blind to it, so from the wrong side edge/center/GRASS bands read
  UNREACHABLE. Heal's freeze breaker covers the survival case; head_to_gym owns the crossing; the GRIND
  path still wedges here (frontier item A).
- MOVEPOOL-WALL GYM (Koga): a solo carry whose STAB is RESISTED by the gym's type + a dead bench + no
  coverage TM/items = a real wall. Human answer: potions to stall + level, or a coverage teammate.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead advanced from a hard freeze
at the Route-15 gate to standing IN Koga's gym at badge 4 — she crosses/heals/reaches Koga cleanly but
loses the fight (movepool + dead bench + no items). Pop-in = `python pokemon_agent/watch.py`.
