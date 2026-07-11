# E4 TACTICAL FINISH — frontier (2026-07-10 night, tactical-not-tank push)

## ⛰️ HONEST BOTTOM LINE (2026-07-10 ~22:00, after 12 recon_e4 runs + grinds L48→60)
**The tactical chain is COMPLETE and works end-to-end; the blocker is now STRUCTURAL (team depth), not tactics.**
Every run reaches Gary; run12 (Lapras L60) fought the Champion's full team for 230s and lost the attrition war.
Gary-fight move census (run12): Lapras landed only **10 super-effective hits** (7 Ice Beam + 3 Surf) before
fainting 3× (3 Revives spent), while Venusaur threw **51 weak hits** (30 Cut @1× + 21 Razor Leaf) and **8 Full
Restores** burned. Gary's 6 mons (L58-62) need ~20 SE hits to fall.
**ROOT: the team has only ~2 usable mons** (Venusaur L71 — but Grass/Normal can't hurt Charizard; Lapras — the
Ice/Water sweeper, but dies repeatedly) **+ Kadabra (frail, Agatha-only) + 3 dead-weight fodder (L8-15).** 2-vs-6
is a losing attrition war regardless of Lapras's level — MORE LEVELS DON'T FIX A DEPTH/SUSTAIN PROBLEM (L60 Lapras
still faints; Venusaur's Cut still does 1×). The prior credits roll (2026-07-07) used a DEEPER/higher team.
**THE DECISION (Jonny's call — competing priorities):** to finish, EITHER (a) accept heavier over-leveling
(Lapras/Venusaur ~L68-70 to genuinely solo-sweep 6 — edges toward the "steamroller" the mandate discouraged, but
is the fastest path to credits), OR (b) build real team DEPTH (a 3rd/4th usable mon — box the fodder, catch/level
a coverage mon; slower, more watchable, truer to the tactical goal), OR (c) a battle-AI change to make LAPRAS the
sole Gary sweeper (lead it + funnel all healing to it instead of splitting with Venusaur's 1× Cut). Recommend (c)
then (a) if (c) falls short. Canonical UNTOUCHED throughout; the 2026-07-07 credits save is the real summit already.


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

## STATUS (2026-07-10 ~21:26): commits 625b568, 0cb736d, 6772b54. **RUN 11 REACHES CHAMPION CHARIZARD.**
The full tactical chain works e2e: Ice Beam sweeps Lance's dragons, FR-first shop (4 FR) survives Lance,
the type-answer revive fires cleanly (USED item 24) to bring Lapras back for Gary, Lapras Surf/Ice-Beam
2x on Charizard (190->150). Remaining margin: thin team (Venusaur + Lapras + Kadabra + 3 fodder) is
out-DPS'd at Charizard (L55 Lapras ~40/hit vs 190 HP; FR runs out). FIX IN FLIGHT: **grind Lapras L55->60**
(in the E4 band — Dragonite is L60 — so it SWEEPS Gary with a damage/survival margin, not trades).
Base for the grind = `grind60_base` (Lapras L55 / Kadabra L50 / Venusaur L71). Kadabra stays L50 (Agatha-only).

## STATUS (2026-07-10 ~20:40): committed 625b568 (moves+4x switch) + 0cb736d (type-answer revive).
Runs 1-6 all reach Gary (room 5) but whiteout there: thin team (Venusaur + Lapras + Kadabra + 3
dead fodder) depletes by Gary. At L50 the specialists TRADE (die after 1-2 KOs) rather than SWEEP;
Venusaur arrives at Gary at ~29 HP (Full Restores exhausted at Lance's 5-dragon wave). The type-answer
revive fires correctly for Charizard but LIVELOCKED at L50 (revive half-HP Lapras into faster Pidgeot ->
re-KO). FIX IN FLIGHT: grind specialists to the **E4 band (L55)** so they SWEEP (OHKO dragons, survive
hits) — kills the livelock AND lets Venusaur reach Gary healthier (fewer dragon hits tanked).

## RESUME (exact next steps)
1. **Grind (running):** `POKEMON_FIELD_MOVES=1 POKEMON_GRIND_SWITCH=0 GRIND_STATE=grind55_base GRIND_SPECIES=131,64 GRIND_TARGET=55 GRIND_MAP=here recon_grind_bench.py` → banks `G:/temp/longrun/banked_GRIND`. (GRIND_SWITCH=0 = direct-fight = ~2x faster since the specialists outlevel Route-18 wilds.)
2. **Port:** `.venv/Scripts/python.exe -u pokemon_agent/recon_port_and_fix.py` → `e4_tactical_v2.state` (ports leveled Lapras+Kadabra into the Indigo base + re-applies Ice Beam/Psychic).
3. **Run to credits:** `E4_STATE=e4_tactical_v2 .venv/Scripts/python.exe -u pokemon_agent/recon_e4.py` → watch `*** HALL OF FAME — CREDITS INBOUND ***` / `banked_CREDITS`.
4. **If still walls at Gary:** (a) reorder recon_e4 `SHOPPING` (line 78) to FR-FIRST so Venusaur arrives healthier; (b) bump GRIND_TARGET to 58; (c) cap the type-answer revive (per-battle) if it still livelocks.
Do NOT kill recon_e4 mid-Gary — let it run to a terminal event (HoF or whiteout).

**WATCH STATUS:** canonical untouched; sherpa E4 push on staging (e4_tactical_v2); she is at Indigo Plateau, one leveled-specialist port away from a tactical Elite Four run.
