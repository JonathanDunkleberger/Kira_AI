# MODE_TRANSITION_AUDIT — Phase I-1 (2026-07-07 night shift 2)

Read-only recon per the Phase I mandate: *"she never resets personality across a mode
switch; callbacks, opinions, and running jokes CARRY."* Ground truth from a full core
sweep (bot.py 11,260 lines + brain/ + modes/ + dashboard/ + memory/); file:line pointers
verified at write time. Spot-checked: the two dead-code claims below (zero callers)
re-verified by grep before commit.

## HEADLINE

**The persona firewall HOLDS. No mode switch resets her.** The base system prompt
(`KIRA_PERSONALITY + IMPROV_DISPOSITION + comedic disposition + TOOL_AND_FORMAT_RULES`)
is built once (`ai_core.py:354-357`), documented "stable across mode flips"
(`ai_core.py:655`), and used by every inference path including VN. Additionally:
- `_reset_session_takes` (bot.py:3868) is deprecated with ZERO callers — takes/opinions
  persist across switches by design.
- `activate_game_mode` (bot.py:3938) resets only session highlights/scene-log + re-latches
  `current_want`; running bits, takes, activity brief, and `conversation_history` (never
  cleared, bot.py:571) all carry.

**What drops across modes is per-turn CONTEXT ASSEMBLY, not persona** — the reply path
and the proactive path assemble different context. Every seam below is that asymmetry.

## THE MODE MAP (for orientation)

- **Axis A** `self.mode`: companion ⟷ streamer (dashboard mode_toggle,
  control_server.py:1061). Gates only chat-question injection/spotlight + silence
  thresholds. No persona/memory effect.
- **Axis B** `game_mode_controller.activity_type`: general / vn / game / media / music
  (activate_game_mode bot.py:3938, dashboard activity_go).
- **Overlays**: pokemon_mode, chess, codenames, VN autopilot, audio mode, presence dial,
  Lock-In, Director. All funnel through `_reconcile_modes` (bot.py:2820) which touches
  ONLY perception plumbing — no memory/persona reset.
- **MediaWatch (the old watch-party module) is RETIRED** — `self.media_watch = None`
  permanently (bot.py:611-614); superseded by always-on Turbo Vision.

## THE TWO ASSEMBLY PATHS

- **Path R (replies — mic voice `process_and_respond` bot.py:10435; chat batch
  bot.py:7351):** the healthy path. Voice carries: continuity block, activity brief
  (opinions), current_want, Jonny-bond, favorites, chatters brief, kira_state
  theories, playthrough memory, running bits, semantic memory, audio/visual. Chess and
  Codenames blocks are ADDED to this, not substituted.
- **Path D (proactive interjections — `_build_scene_block` 9521 + `_build_self_block`
  7121 + Director prompt 7194):** narrows context by mode via early-returns. This is
  where the "reset feel" lives.

## SEAM LEDGER (prioritized; the Phase I-1 fix list)

| # | Seam | Where | Effect |
|---|------|-------|--------|
| 1 | VN narration asides drop bits/takes/opinions/favorites (own third assembly) | vn_autopilot.py:2639-2662 | Her self-initiated VN lines lose running jokes + standing opinions (chat replies during VN unaffected — Path R) |
| 2 | Chess early-return strips proactive context to board+rules only | bot.py:9529-9532 | Chess interjections lose playthrough/story/takes; her chess REPLIES keep full context — asymmetric |
| 3 | Media drive-path skips playthrough memory; reply-path injects it unconditionally | bot.py:9595 vs 10623 | A movie night gets gamer-framed memory in replies that the drive path deliberately suppresses — pick ONE doctrine |
| 4 | Chat-batch path omits favorites/bond/want/kira_state vs voice path | bot.py:7351 vs 10537 | "what's YOUR favorite" via chat has no favorites block; via voice it does |
| 5 | Pokémon live run-state block is Path-R-voice only | bot.py:10635 | During play, chat answers get saga only, no live party — flag as possibly-intended |
| 6 | `_PACING` table + `_effective_react_gap` are DEAD (consumer was retired MediaWatch) | bot.py:2752-2766 | A fully-designed per-mode×intensity riff-window spec (film: CALM 0.7 / TENSE 1.5 / CUTSCENE 1.6, clamps 15-75s) with zero callers |

## POKÉMON FIREWALL VERDICT

**No mode-state leak found.** Live party/badges/questline (`_pokemon_state_block_for_voice`
bot.py:3476) is `pokemon_mode`-gated, Path-R only, and self-suppresses on stale (>45s)
snapshots. The journey SAGA block (bot.py:3447) surfaces in all modes **by design**
(lived experience carries; that's her, not mode-state). Identity flip in
`_build_self_block` (7166-7177) is gated with a byte-preserved cohost else-branch.
**One follow-up owed:** the saga file's WRITE side lives in pokemon_agent — verify the
writer never persists content that reads as live mode-state (party/where-she-is-now)
which the ungated saga block would then surface in normal chat.

## PHASE I-2 / I-3 GROUNDWORK (found, don't rebuild)

- **The focus-state machine already exists** as `SessionIntensity`
  (kira_state.py:48-56: CALM/BUILDING/TENSE/INTENSE/EMOTIONAL/CLIMACTIC/CUTSCENE),
  set live every observer tick by `_classify_moment` (bot.py:4493), consumed by
  interjection cadence (9511-9515), suppress-gates (kira_state.py:804), and audio-mood
  register directives (bot.py:4478-4491 — the film register already encodes
  "don't talk over the swell"). I-2's attention director should EXTEND this, renaming
  nothing, adding narrated attention shifts + chat-sampling modulation on top.
- **I-3's media pacing spec is already written** — the dead `_PACING` table (seam 6).
  The build is a live consumer for it (wire into the interjection cadence in place of
  the retired MediaWatch react_gap_fn), not a new design.
- "Perception barely surfaces vs chat" (memory) confirmed architecturally: chat gets
  per-message reply candidacy; perception only surfaces via silence-gated Path D. Any
  I-2 rebalance must give perception events a candidacy trigger, not just wider gates.

## UNKNOWNs (honest)

- Chat-batch suppression depth under `immersive=True` beyond focus_block tightening
  (bot.py:7726) — affects how hard seam 1 bites.
- Pokémon saga write-side (see firewall verdict).
- `playthrough_memory` injection is whole-manifest flat JSON in every included mode
  (ChromaDB retrieval is aspirational, playthrough_memory.py:139/181/198/870).
- The `music` activity reply path was not exhaustively traced (assumed Path R).

## SUGGESTED FIX ORDER (when I-1 moves from audit to build)

1. Seam 3 (one-line doctrine decision: gate Path R's playthrough injection on the same
   `is_media` test) — smallest, kills a real inconsistency.
2. Seam 1 (fold `_active_bits_for_prompt` + takes into VN's narration assembly) — the
   most audience-visible "same person" win.
3. Seam 2 (chess: append the self/takes block after the board rules instead of
   early-returning).
4. Seam 4 (chat-batch: add favorites/want blocks — cheap, bounded).
5. Seam 6 rides Phase I-3, not I-1.
