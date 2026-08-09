# HANDOFF — Zapdos / Power Plant push

**TL;DR (today):**

- **Gatehouses fixed.** Route-10-NORTH staging now handles the indoor legs (Underground Path,
  both Saffron gatehouses, Route 24/25 dead-routes) that made it return `None` and bounce her out.
- **Fly flag + north-pad guard.** `POKEMON_FLY_FETCH` defaults OFF (unwired Route-16 house), and
  the Zapdos questline now only fires on Route 10 NORTH (y≤50) or in the plant — no more
  Lavender↔Route-12 ping-pong tripping the seam-thrash breaker.
- **Cut tree on Route 9 (2,8) FIXED.** No party mon knew Cut (only Diglett can, benched in the
  PC). `_zapdos_north_staging` now gates on `_zapdos_cut_ready()` and, if Cut is absent, runs the
  `_zapdos_cut_prep` state machine (route to nearest PC → withdraw Diglett → teach HM01 Cut), then
  `_clear_route9_seam()` resets the Cerulean↔Route 9 breaker so she walks clean to Route 10 NORTH.

---

**Status update (today):**

- **Gatehouses fixed.** The Route-10-NORTH staging now owns the indoor legs she kept bouncing
  on — the Underground Path (Route 7↔8), both Saffron gatehouses, and the Route 24/25 dead-end
  dead-routes — so the questline no longer returns `None` and kicks her back out.
- **Fly flag + north-pad guard.** `POKEMON_FLY_FETCH` now defaults OFF (the Route-16 house entry
  is unwired — she walks to Zapdos instead of parking on a broken Fly errand), and the Zapdos
  questline only fires when she's actually on Route 10 NORTH (y≤50) or inside the plant, killing
  the Lavender↔Route-12 ping-pong that tripped the seam-thrash breaker.
- **IMMEDIATE NEXT TASK — Cut tree on Route 9 (2,8).** It blocks Cerulean→Route 9→Route 10-NORTH
  and NO party member knows Cut (only Diglett can, benched in the PC). Next: check
  `fm.can_use(b,"cut")` in `_zapdos_north_staging`; if absent, route to the nearest PC
  (Cerulean/Celadon), withdraw Diglett, teach HM01 Cut, and reset the Route 9 seam-thrash breaker.
