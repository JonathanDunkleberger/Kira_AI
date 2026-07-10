# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 6) — READ FIRST

## SHIFT-6 STATUS (in flight): FIXED the real Rock-Tunnel blocker (warp-route preempted the billed
cave-pass leg) and am VERIFYING the crossing from a fast Route-10 fixture. If verified e2e (crosses
Rock Tunnel -> Lavender -> Route 8/7 -> Celadon), BANK a post-tunnel checkpoint + advance to Erika.

## FRONTIER: the ROCK TUNNEL crossing (Route 10 -> tunnel -> Lavender -> Route 8 (2nd UGP) -> Route 7
-> Celadon -> Erika badge 4). Fix shipped this shift; verifying it actually crosses.

**BOOT (fast — Route 10 mouth, Flash lit, lineage party Venusaur L36 + bench L9-15, dex 10):**
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py rt_mouth.state 30`
(states/workshop/rt_mouth.state = snapshotted from shift-6's in-flight run AT Route 10, Flash taught.)
Fallback full path: `... recon_longrun.py surge_done.state 40` (badge 3 @ Vermilion; re-runs the whole
verified Flash gate + approach — ~35min to reach Route 10).

## SHIFT-6 FIX SHIPPED (uncommitted until verified) — cave-pass leg priority in head_to_gym
ROOT CAUSE of the Rock-Tunnel stall (why shift-4's `_cross_tunnel_leg` never even ran): on Route 10
(3,28), `head_to_gym`'s world-graph WARP-ROUTE (`_next_step_rideable`) runs BEFORE the billed road,
and the CANONICAL world model has Route 10's SOUTH map-connection to Lavender (3,4) as a plain walkable
EDGE. But that band is CLIFF-SEALED — the only crossing is Rock Tunnel — so `_next_step_rideable`
returned `EDGE-ROUTE (3,28)->(3,4) south`, `_edge_travel` dead-ended on `no_path`, head_to_gym got
pruned as a SPLIT-MAP DEAD ROAD, and `_road_step`/`_cross_tunnel_leg` NEVER RAN. She then grind-locked
on Route 10 forever. Same class as shift-3's gated-shortcut preemption (Snorlax/Saffron) but via a
PHANTOM WALKABLE EDGE, not a gated warp.
FIX (campaign.py head_to_gym, ~line 8123): when she's standing ON a billed 'pass' leg with a `cave`
tag (Route 10 only), run `_road_step` FIRST (which dispatches `_cross_tunnel_leg`) before the
warp-route. Minimal blast radius — the guard only fires on a cave-pass leg's own map.

## IF THE VERIFY STALLS — likely spots
- Tunnel INTERIOR maze crossing (`_cross_warp_maze`, maps (1,81)/(1,82)) — ported from proven
  recon_rocktunnel; if it wedges, grep `[tunnel]` for WHICH room/warp and read `_cross_warp_maze`
  (campaign.py:5497). Both Route-10 mouths dest (3,28); sentinel m0=(3,255) makes BOTH count as exits.
- Emergence: after crossing she's on the SOUTH part of Route 10 (3,28); `_edge_travel(Lavender, south)`
  carries her into Lavender (3,4). If the south band is ALSO sealed post-tunnel, edge-travel to (3,4).
- Route 8 (3,26) 2nd Underground Path (Saffron bypass) — shift-3 Saffron-avoid fix covers it.

## GATED-SHORTCUT PATTERN (durable — now with a 3rd instance)
`head_to_gym`'s world-graph warp-route (`_next_step_rideable`) runs BEFORE the billed road and is
GATE/PHYSICS-BLIND: it routes through story-gated shortcuts (Snorlax Route 12, guard-blocked Saffron —
shift 3) AND across phantom walkable edges that are actually cliff-sealed cave crossings (Route 10
south = Rock Tunnel — shift 6). Fix pattern = either add the bad map to the warp-route `avoid` set
(`_story_gate_avoid`, Saffron-avoid) OR, for a billed crossing the road must own, run the billed
`_road_step` first (cave-pass priority). GENERAL future fix: make `_next_step_rideable` skip an EDGE
hop whose band is unwalkable from her feet (reachability pre-check on the edge, like the warp
`_warp_hop_reachable` check) — that would kill the phantom-edge class generally.

## THEN: Erika (badge 4, GOAL flag 0x823 Rainbow). grass/poison, ace vileplume L29. Her spearow
(normal/flying L15) + oddish are OWNED; `prep_for_gym` grinds, NO catch needed. Pidgey L13 (normal/
flying) is the textbook Erika answer but dead on the bench — prep should level it.

## TOPOLOGY (durable)
- Route N = map (3, 18+N). Route 9=(3,27), Route 10=(3,28), Route 11=(3,29), Route 12=(3,30).
- Rock Tunnel interior maps: (1,81)/(1,82). Route 10 (3,28) mouth -> tunnel warp; both mouths dest (3,28).
- Lavender=(3,4). Route 8=(3,26), Route 7=(3,25), Celadon=(3,6), Saffron=(3,10).
- Diglett's Cave: R11 (3,29)->...->R2 (3,20). Celadon road: Vermilion(3,5)->Route6(3,24,UGP)->Route5
  (3,23)->Cerulean(3,3)->Route9(3,27)->Route10(3,28,TUNNEL)->Lavender(3,4)->Route8(3,26,UGP2)->Route7
  (3,25)->Celadon(3,6).

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages ALL persistence to G:/temp/longrun/stage (STAGE redirect) — canonical Champion
  save NEVER touched. STAGE is WIPED at each run start — snapshot it OUT before relaunching.
- recon_longrun loads world_model + strat from CANONICAL (Champion), only soul from the boot bundle.
  So the Route10->Lavender phantom edge comes from the Champion's world graph regardless of fixture.
- states/campaign = SHERPA CANONICAL (Champion, untouchable). states/workshop = scratch. Commit per fix.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Frontier = cross Rock Tunnel to Celadon/Erika.
Pop-in (Sherpa) = `python pokemon_agent/watch.py`.
