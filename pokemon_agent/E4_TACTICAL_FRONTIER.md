# E4 TACTICAL FINISH — frontier (2026-07-10 night, tactical-not-tank push)

**SHIFT HEADER:** `badge 8 → E4 | tactical clear (Ice Beam/Psychic + switching, not tank) | 3 commits pending | grinding specialists`

## The real root cause (disasm + pokemondb, VERIFIED from disk)
The E4 specialists were crippled by **level-up moves the auto-learn heuristic wrongly DECLINED**, not just low levels:
- **Lapras** learns **Ice Beam @ L43** (curated KB `frlg_learnsets.json`; the earlier pokemondb "L31" was a Gen-1 bleed). Ours was L39 = simply not yet at L43 → **the grind past L43 gives Ice Beam ORGANICALLY** (my port re-applies it as insurance vs the decline bug). Ice Beam = 4× on Lance's Dragon/Flying trio, 2× on Gary's Charizard.
- **Kadabra** learns Confusion@16, Disable@18, Psybeam@21, Recover@31, Future-Sight@38 — NO level-up Psychic (Psychic is **TM29** only). Ours was L40 with only Confusion (Psybeam/Recover/FutureSight all DECLINED by auto-learn) → genuinely a declined-moves victim. Restored Psybeam+Recover (level-up) + Psychic (TM29-legal) via `_set_lead_moves`.
- **SOUL-DEBT for the Kira timeline (legit sources):** TM13 Ice Beam = **Celadon Dept Store 4F (~4000)**; TM29 Psychic = **Sabrina/Saffron gym reward** or Celadon Game Corner. (Neither is in the bag now; the sherpa line restores the moves directly since Lapras earns Ice Beam @L43 anyway and Kadabra can TM29 Psychic.)

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

## Battle-AI tuning (runs 3-4 postmortem — IMPORTANT)
- **Switch stays at `best_move_eff <= 0.5`** (TRIGGER 2). Tried widening to `<=1x` (field SE specialist when ace is merely neutral) to get Lapras Ice Beam vs Lance's PURE-Dragon Dragonairs (Ice=2x, Venusaur=1x) — but run4 proved it **over-fields the FRAIL Kadabra** (base HP 40) into OHKOs at Bruno/Agatha, burning the bench before Lance (worse than the tank line). The right distinction is BULK (field bulky Lapras aggressively, frail Kadabra only when the ace is resisted) — no base-stat table exists to gate on it cleanly; a **maxHP proxy (rd16 base+0x58)** is the future enhancement for a truly tactical Dragonair sweep.
- With `<=0.5`: Kadabra fields vs Agatha (Venusaur 0.5x/immune), Lapras fields vs **Dragonite (>=4x override)** AND **Gary's Charizard (Venusaur 0.25x)** — the key moments. Venusaur tanks the pure-Dragon Dragonairs at 1x.
- **THE WALL = Lapras dying before Gary.** Venusaur has NO answer to Gary's Charizard (Grass 0.25x, TM26 EQ is 0x into Flying) — **Lapras Ice Beam (2x) is the ONLY answer, so Lapras MUST reach Gary alive.** At L44 it died at Agatha/Lance. → grinding both specialists to **L50** (bulky Lapras survives to Gary; still under the E4's L53-62 = win on type). recon_e4 runs must be let RUN TO COMPLETION (I killed run3 mid-Gary — don't).

## RESUME (exact next steps)
1. Grind running: `POKEMON_FIELD_MOVES=1 GRIND_STATE=bench_grind_kit GRIND_SPECIES=131,64 GRIND_TARGET=48 GRIND_MAP=here recon_grind_bench.py` → banks `G:/temp/longrun/banked_GRIND`.
2. When done: `.venv/Scripts/python.exe -u pokemon_agent/recon_port_and_fix.py` → `e4_tactical_v2.state`.
3. Run: `E4_STATE=e4_tactical_v2 .venv/Scripts/python.exe -u pokemon_agent/recon_e4.py` → watch for **Hall of Fame → banked_CREDITS**.
4. If still walls: bump GRIND_TARGET (50-52) or check whether Lapras is being fielded vs the dragons (grep `SWITCHED to species 131`).

**WATCH STATUS:** canonical untouched; sherpa E4 push on staging (e4_tactical_v2); she is at Indigo Plateau, one leveled-specialist port away from a tactical Elite Four run.
