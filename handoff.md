# HANDOFF — Zapdos / Power Plant push

**TL;DR (today):**

- **Route 9 Cut tree is DOWN — verified end-to-end.** The first cut-prep pass (withdraw Diglett →
  teach Cut) had three live bugs, all fixed + look-ahead-proven: (1) the PC march used the shared
  Cinnabar-first order whose graph route ran THROUGH the tree → now `_cut_prep_pc_march` picks the
  NEAREST Center (Cerulean, 1 hop); (2) box_bench re-deposited Diglett the same lap (it ALREADY
  knows Cut) → bench plan now reserves Diglett/Dugtrio while zapdos is owed; (3) wedge memory kept
  the tree tile hard-blocked from a pre-Cut session → travel now releases a cut-tree/boulder block
  once its HM is usable. Oracle: she withdrew Diglett, auto-cut (2,8), crossed to Route 10 NORTH,
  Surf'd the strip, and engaged Zapdos in the Power Plant.
- **IMMEDIATE NEXT TASK — finish the Zapdos catch (Milestone 1).** The strike reaches the bird but
  the catch didn't close in the look-ahead: ace Blastoise had 0 safe chip-PP ("chip PP THIN",
  "SKIPPING pre-zapdos bank"), so the fight can't weaken Zapdos cleanly. Next: let her heal/restore
  PP at the Route 10 Center (door (13,20)) and re-engage with balls, or teach a safe chip move.
- **Then: Milestones 2-4.** Fill the 6 for E4, Victory Road → Lorelei/Bruno/Agatha/Lance/Champion
  → credits, then post-game Mewtwo (Cerulean Cave, Master Ball reserved).

---

**Status update (today):**

- **Gatehouses fixed.** The Route-10-NORTH staging now owns the indoor legs she kept bouncing
  on — the Underground Path (Route 7↔8), both Saffron gatehouses, and the Route 24/25 dead-end
  dead-routes — so the questline no longer returns `None` and kicks her back out.
- **Fly flag + north-pad guard.** `POKEMON_FLY_FETCH` now defaults OFF (the Route-16 house entry
  is unwired — she walks to Zapdos instead of parking on a broken Fly errand), and the Zapdos
  questline only fires when she's actually on Route 10 NORTH (y≤50) or inside the plant, killing
  the Lavender↔Route-12 ping-pong that tripped the seam-thrash breaker.
- **Route 9 Cut tree FIXED + VERIFIED (look-ahead reached Zapdos).** `_zapdos_north_staging` gates
  on `_zapdos_cut_ready()` and runs `_zapdos_cut_prep` (nearest-PC march → withdraw Diglett → it
  already knows Cut → `_clear_route9_seam()`). Three shuttle/wedge bugs fixed this session:
  `_cut_prep_pc_march` (nearest Center, not Cinnabar-through-the-tree), the `_lap_bench_plan`
  Diglett-reserve while zapdos is owed, and travel's field-obstacle block release once the HM is
  usable. Oracle run: march→withdraw→cut (2,8)→Route 10 NORTH→Surf→Power Plant→Zapdos engaged.
- **NEXT — close the Zapdos catch.** Strike reaches the doorstep but the ace had 0 safe chip-PP in
  the look-ahead, so the weaken/catch didn't finish. Heal/restore PP (Route 10 Center (13,20)) and
  re-engage, then continue the lap (repack → head_to_league → Victory Road → E4 → credits → Mewtwo).
