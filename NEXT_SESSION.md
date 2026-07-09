# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 2) — READ FIRST

## FRONTIER: badge-4 (Erika/Celadon). Flash dex-10 gate MECHANICS CLEARED+VERIFIED.
## Open: badge-4 APPROACH routing (Rock-Tunnel path) has run-to-run variance.

**BOOT (fresh):** `.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py surge_done.state 25`
**BOOT (post-Flash fixture, at Viridian):** `... recon_longrun.py flash_done.state 20` (banked this shift).
(rebuild states/workshop/*.state + sidecars from `G:/temp/longrun/banked_GOAL/` (surge) or the STALL
bank if pruned.)

## ✅ SHIPPED + VERIFIED THIS SHIFT — the Flash dex-10 chain (commits d43c5d7, 55dc423, 2387af7)
Flash (HM05) needs OWNED>=10; badge-3 fixture boots dex 5. Verified end-to-end in `logs/debug/shift2_cross.log`
(dex 5->10 -> Diglett's Cave (3,29)->(1,38)->(3,20) -> aide gatehouse -> TEACH flash -> flash_done -> TAUGHT):
1. **Route-6 leg-catch** (`_flash_errand` PHASE 1, `_FLASH_CATCH_LEGS`={Route 6}). Route 9/10 EXCLUDED — Route
   10's west exit is NPC-pinched at (7,6) → catching there = travel oscillation. Per-leg exhausted-memo.
2. **Dex-gate full-party override** (`_dex_catch_all` + catch-judgment ~campaign.py:4057): NEW species is a
   must-catch even at party 6 (it BOXES, dex ticks). Verified: "team's full ... box it for the dex, it counts."
3. **Vermilion ball restock** (PHASE 1, cur==VERMILION): buys up to 12 balls when dex<10. Verified.
4. **Route 11 is NEVER a dex-stall** — from Route 11 always cross the cave (guard excludes it + PHASE-2 memo).

## 🔨 COMMITTED-BUT-UNVERIFIED THIS SHIFT — post-Flash return-cross (commit 391fdd3)
Post-teach she was dumped at the Route 2 aide (a SW dead-pocket) and forward-drive PING-PONGED
Viridian<->Route 2 to a STALL (the billed Celadon road-head is VERMILION, across the Diglett's Cave
warp-maze the world-graph won't route). FIX: the instant Flash is learnt, the errand re-crosses the cave
EAST to Route 11 -> Vermilion, where the billed Celadon road picks up (VERIFIED that road works from
Vermilion at boot: "ROAD to Celadon City: on Vermilion City — billed leg north toward Route 6"). **NOT yet
exercised end-to-end** — the one verification re-run (shift2_return.log) STALLED EARLIER at Route 9 (see
below) before reaching the teach, so the return-cross never fired. RE-RUN to exercise it.

## THE OPEN BLOCKER — badge-4 APPROACH routing variance (the successor's #1 job)
The billed Celadon road (roads["Celadon City"]) is: Vermilion -> Route 6 (Underground Path pass, bypasses
Saffron) -> Route 5 -> Cerulean -> Route 9 (Cut tree at its mouth) -> **Route 10 -> THROUGH ROCK TUNNEL
(Flash-gated; the `_road_move` gate-check arms the flash errand HERE)** -> Lavender -> Route 8 (2nd
Underground Path) -> Route 7 -> Celadon. So Flash IS required and the errand engages at the Route-10 Rock
Tunnel gate. BUT across runs the APPROACH is unstable:
- shift2_cross.log: reached Route 10 -> flash errand engaged -> full chain -> TAUGHT. ✅
- shift2_return.log: STALLED at Route 9 (3,27)@(20,13), dex 5, flash errand NEVER engaged, head_to_gym
  PRUNED from options. Early travel WEDGES at map (18,0) (an Underground Path interior — the Route 6 hut
  warp (19,13)->(1,32)) x8, then couldn't complete Route 9 -> Route 10 east edge (Cut tree at Route 9's
  mouth? field_clear may not have fired). → no_gym_route → stall.
DIAGNOSE: why head_to_gym intermittently fails the Cerulean->Route 9->Route 10 legs (the Cut tree at Route
9's mouth via `field_clear`, and the Underground Path (18,0) wedge). Make the approach to the Rock Tunnel
gate RELIABLE, then the (verified) flash errand + (committed) return-cross carry her to Celadon.

## THEN: Erika (badge 4, GOAL flag 0x823 Rainbow). grass/poison, ace vileplume L29. Her spearow (flying)
is an OWNED type answer, just underleveled (L15) — `prep_for_gym` grinds it, NO catch needed.

## KEY FACTS / TOOLS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors before re-run).
- Fixtures: surge_done.state (badge 3 @ Vermilion, dex 5), flash_done.state (post-Flash @ Viridian, dex 10).
- `recon_flash_errand.py [state] [min]` drives _flash_errand tight; `FLASH_INJECT_BALLS=N` isolates balls.
- states/campaign = SHERPA CANONICAL (Champion, untouchable). states/workshop = scratch. Commit per fix.
- Canonical Champion save already rolled credits — NEVER clobber. recon persistence STAGE-redirected.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Fresh-run look-ahead crosses bedroom -> badge 3
(Surge) -> clears the badge-4 Flash dex-10 gate (dex 5->10, cave, TEACH flash — verified). Frontier =
stabilize the Rock-Tunnel APPROACH routing (Route 9->10) so the errand engages every run, then Celadon ->
Erika. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.
