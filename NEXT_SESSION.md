# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 3) — READ FIRST

## FRONTIER: badge-4 (Erika/Celadon). The ENTIRE Flash gate + the Celadon APPROACH routing solved
## this shift (5 commits). Verifying the full approach e2e (Rock Tunnel -> Celadon) in run8.
**BOOT (fresh):** `.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py surge_done.state 35`
(surge_done.state = badge 3 @ Vermilion, dex 5, in states/workshop/.)

## ✅ SHIPPED + VERIFIED THIS SHIFT (5 commits: c99cbc6, c4fe551, 06d96be, 47f6add, 7b86de8)
The badge-4 Flash gate AND the Celadon approach are now clean. Blockers found+fixed+verified:

1. **Return-cross enters the cave + target-directed cross (c99cbc6).** shift-2's return called
   `_cross_cave(None,..)` while inside the aide gatehouse -> never entered the cave. FIX: travel the
   cave-exit tile (17,12) (travel CUTS the regrown (18,26) tree enter_warp's pre-check sees as a wall),
   enter mouth (17,11), and made `_cross_cave` TARGET-DIRECTED (exit only via a warp whose dest IS
   out_map, else progress via cave warps). Verified recon_returncross.py.

2. **Route-11 catch LIVELOCK killed (c4fe551).** Catch-judgment SKIP FOUGHT every dupe -> PP famine ->
   battle 'stuck' -> within-tick freeze-spin (run4: 34 stuck). FIX: FLEE dupes during the dex errand
   (_dex_catch_all) + a STUCK_CAP(6) circuit breaker in catch_one (rule 18). Verified run5: 0 stuck,
   80 clean flees, dex 5->10, teach.

3. **Return-cross escapes the aide LEDGE TRAP (06d96be).** The aide gate (15,2) is a pass-through:
   SOUTH door (7,10)->(18,47) drops into a pocket walled from the cave by a ONE-WAY LEDGE; NORTH door
   (7,1)->(18,41) reaches the mouth. FIX: seek the north pocket (exit north; re-enter (18,46)+exit north
   if dumped south). Verified recon_returnfull.py + run6 (real run: north_pocket=True -> cave -> Route 11).

4. **Celadon route: Rock Tunnel not the Snorlax (47f6add).** From Vermilion the graph warp-route sent
   her EAST to Route 12 (Snorlax-blocked pre-Flute) -> wedge. FIX: `_story_gate_avoid` avoids the
   Flute-gated maps (Route 12 (3,30), Route 16 (3,34)) until ITEM_POKE_FLUTE(350) is owned -> falls to
   the billed road (north via Route 6). Verified run7: from Vermilion she routes NORTH to Route 6.

5. **Bypass guard-blocked Saffron (7b86de8).** The graph then drove her into the Route-6<->Saffron
   gatehouse (18,0) and oscillated on the guard-blocked (4,1) warp toward Saffron (3,10). FIX: in the
   warp-route, avoid Saffron unless it's the destination gym (badge 6) -> falls to the billed road's
   Underground-Path 'pass' leg (Route 6 -> Route 5). Verifying run8.

## ✅ VERIFIED e2e (shift3_run8.log) — the full Celadon APPROACH up to the Rock Tunnel mouth
run8 confirmed the WHOLE approach clean: dex sweep -> teach -> return-cross -> Vermilion -> NORTH
Route 6 -> **Underground Path (warp (19,13)->(1,32), BYPASSING Saffron)** -> Route 5 -> Cerulean ->
Route 9 -> Route 10 (Rock Tunnel mouth), where she GRINDS with purpose (raised the whole bench floor
to L15, then Venusaur L35->L39 pre-gym). 0 stuck battles the whole run. The Snorlax AND Saffron fixes
both fire on the real post-Flash leg. So the badge-4 Flash gate + the Celadon approach routing are
DONE.

## 🔴 THE NEW FRONTIER: the ROCK TUNNEL crossing (map (1,81)/(1,82)) -> Lavender -> Celadon
She reaches Route 10 and preps, but the look-ahead had NOT yet entered Rock Tunnel at handoff (still
grinding to L39). Rock Tunnel is a DARK cave (Flash IS taught now) with a trainer gauntlet; the
Champion crossed it but THIS lineage hasn't re-exercised it. Re-run `recon_longrun surge_done.state
35` (or boot G:/temp/longrun/banked_ROCKTUNNEL — Champion @ Route 10 (3,4), but Champion party/Flute)
and watch the Rock Tunnel entry + crossing. Grep: `Rock Tunnel|Lavender|dark|hm_blocked|no_gym_route|
Celadon City|==== OUTCOME|STALL`. If it wedges entering/crossing the tunnel, that's the blocker to
fix. AFTER Rock Tunnel: Lavender -> Route 8 (2nd Underground Path, another Saffron bypass — the fix
covers it) -> Route 7 -> Celadon -> Erika.

## THEN: Erika (badge 4, GOAL flag 0x823 Rainbow). grass/poison, ace vileplume L29. Her spearow
(normal/flying, L15) is the OWNED type answer — `prep_for_gym` grinds it, NO catch needed. She also
picked up oddish/pidgey/meowth in the dex sweep (bench options).

## GATED-SHORTCUT PATTERN (durable lesson — the shift's meta-finding)
head_to_gym tries the world-graph WARP-ROUTE (`_next_step_rideable`) BEFORE the billed-road fallback,
and the graph is CAPABILITY/GATE-BLIND — it routes through story-gated shortcuts (Snorlax Route 12,
guard-blocked Saffron) the billed road deliberately bypasses. Fix pattern = add the gated map to the
warp-route `avoid` set (conditioned on the key item / not-the-destination) so it falls to the billed
road. If MORE gated shortcuts surface, extend `_story_gate_avoid` / the Saffron-style guard. A future
GENERAL fix: make `_next_step_rideable` skip any hop into a KB-gated map whose gate is unmet.

## TOPOLOGY (durable, learned this shift)
- Diglett's Cave: R11 (3,29) --(6,7)--> (1,38) --(6,4)--> (1,37) --(3,3)/(82,71)--> (1,36) --(4,6)--> R2
  (3,20)@(17,12). R2 cave mouth = (17,11)->(1,36).
- R2 aide gate (15,2): NORTH door (7,1)->(18,41) [reaches cave]; SOUTH doors (7,10)/(6,10)/(8,10)->
  (18,46/47) [ledge-sealed]. Aide (HM05) inside (15,2). One cut tree on R2 at (18,26).
- Saffron = (3,10); Route 6<->Saffron south gatehouse = (18,0), its Saffron warp (4,1) is guard-blocked.
- Route N = map (3, 18+N). Route 12=(3,30), Route 16=(3,34).

## TOOLS ADDED (fast, canonical-safe, read-only bank probes)
`recon_returncross.py [bank]`, `recon_gate_exit.py`, `recon_returnfull.py` — return-cross/gate probes.

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages ALL persistence (STAGE redirect) — canonical Champion save NEVER touched.
- states/campaign = SHERPA CANONICAL (Champion, untouchable). states/workshop = scratch. Commit per fix.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Fresh-run look-ahead now crosses bedroom ->
badge 3 -> the WHOLE Flash gate (dex sweep+cave+teach+return, fixed+verified) -> the Celadon approach
routing (Rock Tunnel not Snorlax, bypass Saffron — fixed, verifying run8). Frontier = confirm she
reaches Celadon/Erika e2e (run8), then badge 4. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.
