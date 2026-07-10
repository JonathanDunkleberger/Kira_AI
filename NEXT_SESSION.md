# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 7 IN-FLIGHT) — READ FIRST

## SHIFT-7: SILPH SCOPE STRIKE DONE + VERIFIED e2e (commit 93ee1ab). TOWER STRIKE wired, verifying.
`pokemon_agent/hideout_strike.py` (Rocket Hideout: poster -> spin mazes -> Lift Key -> elevator ->
Giovanni -> Scope -> exit) + `tower_strike.py` (Pokémon Tower climb: Gary 2F -> Marowak ghost 6F ->
7F grunts -> Mr. Fuji -> Poké Flute) are faithful in-loop ports of the champion's recon_hideout.py/
recon_hideout_exit.py/recon_tower.py, driven by `camp`, dispatched from `_questline_strike` (a
registry keyed by step.success: ('item',359)=Scope, ('item',350)=Flute), hooked at the top of
`_questline_interact`. KEY LESSON: the frontier lead enters dungeons WORN (erika_done: Venusaur
76/135) and there's no Center below the boss -> each strike HEALS to full (HP+PP) before descending;
the Tower strike also has an attrition heal-valve (descend+heal when lead <50%). The "underlevel"
losses were really un-healed L43 Venusaur. SILPH SCOPE verified e2e (run6: questline_strike_done,
scope in bag, advanced to the Poké Flute step). TOWER strike verification = run7 (in flight when this
was written). If the Flute isn't confirmed, check logs/nightshift7/s7_run7.log for the tower strike
trace + G:/temp/longrun/tower_probe/*.png frames.

## SHIFT-6 BANKED: ROCK TUNNEL CROSSING FIXED + VERIFIED e2e -> BADGE 4 (RAINBOW / Erika) WON.
The badge-3->4 stretch is DONE. She now crosses Rock Tunnel on the billed road and goes on to beat
Erika autonomously. Commit 230d6eb. NEW FRONTIER = the badge-5 SILPH SCOPE / ROCKET HIDEOUT arc.

## FRONTIER: badge 5 (Koga/Fuchsia) — but GATED behind the SILPH SCOPE -> POKE FLUTE -> Snorlax chain.
Post-badge-4 she PROACTIVELY opens the Silph Scope questline (correct — it's the gate toward Fuchsia
via the Flute/Snorlax road) and STALLS in Celadon. Precise diagnosis below.

**BOOT (post-badge-4, stalled in a Celadon building — the exact frontier):**
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py erika_done.state 30`
(states/workshop/erika_done.state = shift-6 banked_STALL, badges=4, Venusaur L40 + bench L9-15, IN
Celadon, Silph Scope questline armed. She has the TEA already (Saffron unlocked for badge 6).)

## THE SILPH SCOPE STALL — PRECISELY CHARACTERIZED (do this next)
The Silph Scope lives with Giovanni on **B4F of the Rocket Hideout under the Celadon Game Corner**.
What WORKS (verified in shift6_run1.log):
  - questline door-hint enters the Game Corner: "QUESTLINE ENTER (door-hint): inside (10, 14)" — the
    KB door (34,21)->interior (10,14) is correct and fires.
What's UNWIRED (the stall — she loops Celadon buildings "wrong_building/worked_room" then STALLs):
  1. **POSTER ACTUATION** — inside the Game Corner (10,14) she "works the room" (presses slot
     machines) but never presses the specific POSTER wall-tile that opens the hidden staircase down
     to the hideout. So she never descends. (The (1,42) hits in the log are world-graph warp reads,
     NOT her position — she does NOT reach the hideout.)
  2. **DEEP HIDEOUT NAV** — even once down: B1F (1,42) -> arrow-tile spin mazes -> grab the LIFT KEY
     (dropped by a grunt) -> ride the elevator B1->B4 -> trainer maze -> Giovanni -> Scope. The general
     free_roam questline has none of this.
THE CHAMPION CLEARED THIS via bespoke STRIKE code that EXISTS in the repo — WIRE IT into the general
questline for the hideout maps:
  - `recon_hideout` / `elevator_nav.py` (is_car + panel ride + SB1+0x14 dynamicWarp LANDING ORACLE;
    memory: DESCENT shift 5) — the elevator crosser.
  - `spin_nav.py` (general slide/arrow-tile crosser; memory: Silph Scope->Flute->Snorlax) — the
    arrow-tile mazes.
  - poster-gate + Lift Key pickup (memory: "Ghost-Vileplume + Hideout — poster gate; spin-tile
    crosser; Lift Key"). Hideout maps = (1, 42..46). Scope confirm = BAG item 359 (0x037 is HIDE-class
    and reads set SPURIOUSLY — confirm the ITEM, per the KB confirm_note, else she skips the hideout).
Recommended approach: reproduce from erika_done.state, watch her in the Game Corner (10,14), and first
solve the POSTER press (get her into (1,42)); then invoke the proven hideout strike code for the
floors. This is the arc that has historically eaten shifts (contract warns: "shift 8 died mid-Silph-
strike") — take it with FRESH budget, wire the EXISTING strike code, don't rebuild.

## AFTER SILPH SCOPE: the full badge-5 chain
Silph Scope -> Pokemon Tower (Lavender, rescue Mr. Fuji past the ghost Marowak on 6F) -> POKE FLUTE ->
wake the Route 12 Snorlax -> Route 12/13/14/15 south -> Fuchsia -> Koga (badge 5, Soul). NOTE the Tea
is ALREADY obtained this shift, so Saffron's guards are open for the badge-6 (Sabrina) approach later.

## SHIFT-6 FIX SHIPPED + VERIFIED (commit 230d6eb) — cave-pass leg beats the warp-route on Route 10
ROOT CAUSE (why shift-4's `_cross_tunnel_leg` never even ran): on Route 10 (3,28) head_to_gym's
world-graph WARP-ROUTE (`_next_step_rideable`) runs BEFORE the billed road, and the CANONICAL world
model has Route 10's SOUTH map-connection to Lavender (3,4) as a plain walkable EDGE. That band is
CLIFF-SEALED (only crossing = Rock Tunnel), so `_next_step_rideable` returned EDGE-ROUTE south,
`_edge_travel` dead-ended on no_path, head_to_gym got pruned as SPLIT-MAP DEAD ROAD, `_road_step`/
`_cross_tunnel_leg` NEVER RAN, she grind-locked on Route 10. FIX (campaign.py head_to_gym ~8123): when
standing ON a billed 'pass' leg with a `cave` tag, run `_road_step` FIRST so `_cross_tunnel_leg` owns
the leg before the warp-route. VERIFIED e2e (rt_mouth.state): Route 10 -> Flash-lit Rock Tunnel maze
(1,81/1,82) -> Lavender -> Route 8 (UGP2, Saffron bypass) -> Route 7 -> Celadon -> Tea -> Erika ->
BADGE 4.

## GATED-SHORTCUT PATTERN (durable — 3 instances now)
head_to_gym's warp-route (`_next_step_rideable`) runs BEFORE the billed road and is GATE/PHYSICS-BLIND:
routes through story-gated shortcuts (Snorlax Route 12, guard-blocked Saffron — shift 3) AND across
phantom walkable edges that are cliff-sealed cave crossings (Route 10 south = Rock Tunnel — shift 6).
GENERAL future fix: make `_next_step_rideable` skip an EDGE hop whose band is unwalkable from her feet
(reachability pre-check on the edge, mirroring the warp `_warp_hop_reachable` check) — kills the
phantom-edge class generally. For now: `_story_gate_avoid` + Saffron-avoid + cave-pass priority.

## TOPOLOGY (durable)
- Route N = map (3, 18+N). Rock Tunnel interior = (1,81)/(1,82); both Route-10 mouths dest (3,28).
- Celadon road (VERIFIED): Vermilion(3,5)->Route6(3,24,UGP)->Route5(3,23)->Cerulean(3,3)->Route9(3,27)
  ->Route10(3,28,TUNNEL)->Lavender(3,4)->Route8(3,26,UGP2)->Route7(3,25)->Celadon(3,6).
- Celadon Game Corner door (34,21) -> interior (10,14). Rocket Hideout maps = (1,42..46). Saffron=(3,10).

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages ALL persistence to G:/temp/longrun/stage (STAGE redirect); STAGE is WIPED at
  each run start — snapshot it OUT before relaunching. Loads world_model+strat from CANONICAL, soul
  from the boot bundle. Canonical Champion save NEVER touched.
- states/workshop/*.state = the fixture library: surge_done (badge 3 @ Vermilion), rt_mouth (Route 10
  mouth, Flash lit), flash_done (post-Flash Viridian), erika_done (badge 4 @ Celadon — THE FRONTIER).
- states/campaign = SHERPA CANONICAL (Champion, untouchable). Commit per fix.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead is at badge 4 in Celadon,
arming the Silph Scope / Rocket Hideout chain toward badge 5 (Koga). Pop-in = `python pokemon_agent/watch.py`.
