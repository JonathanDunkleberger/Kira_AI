# AUTONOMOUS_GAME_HARNESS.md — the portable autonomous-game-playing harness (game #1 = FireRed)

The real product is **NOT a FireRed bot** — it's a *generalizable* harness that drives Kira's personality
through autonomous play of any RAM-accessible game. FireRed is instance #1 / the proving ground. The
post-credits goal: port to game #2 (e.g. Pokémon Emerald) by **swapping the game-knowledge layer +
re-verifying with the same engine**, not rebuilding. This doc is the accumulated wisdom so game #2 starts
from the bedrock + pitfalls, not from scratch. (See CLAUDE.md rule 14.)

## THE CLEAN LINE — engine (reusable) vs game-knowledge (swappable)

**ENGINE (game-agnostic — keep free of hardcoded game facts where feasible):**
- The **look-ahead oracle** (`recon_longrun.py`): boot the real save, run the REAL decision loop headless at
  max emulator speed (~14× real-time), to GOAL or genuine STALL, then read the sped-up log for the real next
  blocker. Protects the canonical save (persists to a staging dir) + banks round-trip-verified checkpoints.
- **Checkpoints / resumable progress / durable-docs continuity** (`STATE_OF_PROJECT.md`, world/strat/soul
  sidecars). The harness + checkpoints + STATE doc are the durable state that survives context resets.
- **The 15-block bedrock player-competency map** (CLAUDE.md): the competencies ANY player needs for ANY game
  — forward-drive, team-building, strategic-grind, battle competence, resource/economy, healing/survival,
  stuck-resolution, spatial literacy, NPC/dialogue literacy, gate-unlock questline, danger awareness, etc.
- **The run-the-rope operating loop** (rules 8-13): max-speed look-ahead → blocker → diagnose → build the
  GENERAL fix → re-run → bank → advance. Solve once, reuse everywhere.
- **The firewall** (rule 12): core Kira identity sacred; mode-state behind the toggle; plumbing improvable.

**GAME-KNOWLEDGE LAYER (swappable per game — lives in `gamedata/` + clearly-tagged constants):**
- `gamedata/frlg_gates.json` — typed gates (HM_OBSTACLE/STORY_NPC/ITEM_GATE/BADGE_GATE) + capability chains.
- Warp tables, flags (badge/story), gym order, item/HM sources, map IDs, **Mart doors + stock orders**,
  type chart specifics, RAM offsets (`firered_ram.py`, `pokemon_state.py`).
- **PORTABILITY DEBT (FireRed specifics still coupled into engine code — isolate when porting):** the
  city/door/stock constants (`CITY_MART_DOORS`, `MART_STOCK`, `CITY_PC_DOORS`, gym-door coords, `GYM_SPINE`)
  live in `campaign.py`, not `gamedata/`. For game #2, lift these into a per-game KB the engine reads.

## THE PITFALLS (hard-won — each cost real time; expect analogues in every game)

1. **Don't assume the model knows the game.** Kira is a BLANK SLATE — without an explicit game-model (goal /
   win-condition / what a team is for / catching / the arc) her choices are random. Wire the player's mental
   model into her decision ctx as HER grounded understanding (soul-preserving: she understands like a player,
   still discovers unencountered content). → CLAUDE.md "FOUNDATIONAL GAME-MODEL".
2. **Reactive-only objectives strand her.** Recognising a gate only INSIDE the execution branch means at
   decision-time she picks a backward grind on equal footing. Open the forward objective PROACTIVELY each
   tick BEFORE building the action set; prune backward options.
3. **Underleveled-with-no-team wall.** Over-levelling a single ace is slow (low wild XP) and type-coverage
   gaps (one damaging move) still wall you. The real fix is team-building (catch + choose + field/level) +
   strategic grind (field the WEAK members, not the ace). A loss → build strength while staying pointed at
   the objective; never knock backward to a cleared dead-end.
4. **Disasm map-numbers are unreliable** — ALWAYS cross-check route/map IDs against live RAM (the disasm
   route-number export disagreed with live headers; e.g. Route 24 = (3,43), not a contiguous pattern).
5. **The Mt-Moon wrong-warp class** — a hand-spotted physical stuck (a ladder/warp that teleports you off
   the intended path). Capture non-obvious nav nuances EXPLICITLY in durable docs (rule 11). Treat OTHER
   warp tiles on a floor as avoid-tiles when pathing to a target warp.
6. **In-battle / in-menu actuation degrades on the LONG-RUNNING core.** Menu nav that works on a fresh core
   can wedge after the core has run a while (the libmgba continuous-core artifact). Navigate menus by RAM
   **cursor-readback** (verify each press moved the cursor), not blind taps. This recurs at every
   gym/Mart/E4 (bag use, party switch, buy list). The `_goto_bag` / `_mart_goto_row` cursor-readback is the
   template; the party-switch menu still needs it (gated `POKEMON_BATTLE_SWITCH=0`).
7. **Pocket-aware item reads.** Items live in separate bag pockets. Reading the wrong pocket silently
   returns 0 — e.g. `bag_count` (Items pocket) can't see Poké Balls (balls pocket +0x430), which made the
   Mart buy-verify ALWAYS fail for balls (she "couldn't buy balls"). Dispatch the count by item-id range.
8. **Misidentification by flaky actuation.** Identifying a thing by whether a menu actuates (e.g. "is this
   the Mart? does the buy list open?") is fragile — a long-core actuation failure reads as a false negative.
   Prefer a ROBUST structural signal (interior layout: clerk object at a fixed tile) over actuation. This is
   how the prior session mis-tagged the policeman-house as the Mart.
9. **Verify before build; observe the REAL failure.** Don't reconstruct a prior session's theory — run the
   thing and read what actually happens (the policeman NPC was stationary, not a wanderer; the Mart buy flow
   actually works; etc.). Default verification = the look-ahead, not bite-sized micro-tests (rule 8).
10. **Story-gates force a route order** — a parked NPC/obstacle (Slowbro/policeman until S.S. Ticket) means
    "go fetch X first." Recognise typed gates → derive an ordered questline from the KB → execute it via the
    travel layer (the general gate-unlock capability). Generalises to Cut/Surf/Strength/Flash/Fly/item-gates.

## "HOW TO TEACH KIRA A NEW RAM-ACCESSIBLE GAME" (the port playbook)
1. **RAM map first:** find party/inventory/money/flags/map-id/coords offsets (use the game's disasm —
   pret/* for Pokémon — + a RAM differ; cross-check live). Populate the per-game KB.
2. **Wire the game-model** into her decision ctx (goal, win-condition, what progress means, the arc).
3. **Stand up the look-ahead harness** pointing at the new ROM + save; reuse the engine wholesale.
4. **Run the rope:** look-ahead → blocker → map the blocker to a bedrock competency → reuse the general
   capability (or build it general the first time it appears) → re-verify → bank → climb.
5. **Resources when stuck:** the disassembly (authoritative), wikis (Bulbapedia), prior solved projects
   ("GPT/Claude beats Pokémon"), and your own reasoning. Almost never bounce a blocker to Jonny (rule 13).

## STATUS (FireRed, instance #1): climbing — see `STATE_OF_PROJECT.md` §0 for the live frontier.
