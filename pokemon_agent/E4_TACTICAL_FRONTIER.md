# E4 TACTICAL FINISH — frontier (2026-07-10 night, tactical-not-tank push)

**SHIFT HEADER:** `badge 8 → E4 | tactical clear (Ice Beam/Psychic + switching, not tank) | 3 commits pending | grinding specialists`

## The real root cause (disasm + pokemondb, VERIFIED from disk)
The E4 specialists were crippled by **level-up moves the auto-learn heuristic wrongly DECLINED**, not just low levels:
- **Lapras** learns **Ice Beam @ L31** — ours was L39 WITHOUT it (declined for Confuse Ray/Perish Song). Ice Beam = 4× on Lance's Dragon/Flying trio, 2× on Gary's Charizard.
- **Kadabra** learns **Psybeam@L21, Recover@L25, Psychic@L36** — ours was L40 with only 50-pw Confusion (Teleport/Flash/Disable filling slots).
- Both PASSED those learn levels, so grinding can't retrieve them → fixed by restoring the earned moves via `_set_lead_moves` (NOT TM injection — TM13/TM29 are NOT in the bag; nearest legit source = Seafoam/Saffron backtrack, too costly from Indigo). **SOUL-DEBT for the Kira timeline: acquire TM13 Ice Beam (Seafoam) + TM29 Psychic (Saffron) or Two-Island Move Reminder en route.**

## Fixes shipped this session (mode-side, canonical untouched)
1. **`pokemon_state.MOVE_NAMES`** += Ice Beam/Psychic/Psybeam/Recover/Confusion/Body Slam/Confuse Ray/… — she narrates "Ice Beam!" not "#58" (watchability).
2. **`battle_agent._best_switch_slot`** — the offensive-specialist veto now lets a **≥4× answer override the level floor** (bulky L39 Lapras Ice Beam vs L55 dragons was wrongly benched by the lv+15 fodder floor → Venusaur tanked to a whiteout).
3. **`recon_fix_e4_moves.py`** — builds `e4_tactical.state` = e4_base (Indigo, badge 8) + restored Lapras/Kadabra movesets.
4. **`recon_port_and_fix.py`** — ports GRINDED Lapras+Kadabra structs from the Route-18 grind into the Indigo base → `e4_tactical_v2.state` + re-restores moves.

## Verified behavior (recon_e4 runs 1-2 from e4_tactical)
- Switch **fields Kadabra Psychic 2× at Agatha** ✓ and **Lapras Ice Beam at Lance** ✓ (after the veto fix) — the tactical display WORKS.
- **But both specialists FAINT** — L39/L40 too frail vs L53-56 E4 → whiteout at Lance. → need a **minimal survivability floor (~L48, still UNDER the E4 — win on type not tank)**.

## Grind blocker + workaround (the nav lesson)
- Victory Road (adjacent strong wilds L36-46) is a **CAVE** — the grass-only `grind()` can't train there, AND Indigo→VR has **no cached travel route** (VR entered at top via cave geometry; Route23→VR warp at **(5,28)→VR1F**, (18,28)→VR2F not in world model; Route 23 also needs Surf).
- Also: **`recon_grind_bench` does NOT set `POKEMON_FIELD_MOVES=1`** (only recon_longrun does) — always pass it or water/cut nav wedges.
- **Workaround = grind on Route-18 grass (harness-proven) from `bench_grind_kit`, then PORT the leveled Lapras+Kadabra into the Indigo `e4_base`** (raw 100-byte party-struct copy carries the whole mon).

## RESUME (exact next steps)
1. Grind running: `POKEMON_FIELD_MOVES=1 GRIND_STATE=bench_grind_kit GRIND_SPECIES=131,64 GRIND_TARGET=48 GRIND_MAP=here recon_grind_bench.py` → banks `G:/temp/longrun/banked_GRIND`.
2. When done: `.venv/Scripts/python.exe -u pokemon_agent/recon_port_and_fix.py` → `e4_tactical_v2.state`.
3. Run: `E4_STATE=e4_tactical_v2 .venv/Scripts/python.exe -u pokemon_agent/recon_e4.py` → watch for **Hall of Fame → banked_CREDITS**.
4. If still walls: bump GRIND_TARGET (50-52) or check whether Lapras is being fielded vs the dragons (grep `SWITCHED to species 131`).

**WATCH STATUS:** canonical untouched; sherpa E4 push on staging (e4_tactical_v2); she is at Indigo Plateau, one leveled-specialist port away from a tactical Elite Four run.
