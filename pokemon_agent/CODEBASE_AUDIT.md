# Pokémon agent — state-of-the-codebase audit (2026-06-28)

Honest audit per the THREE-STATE standard. **COMPILES** = runs, no crash. **WIRED** = the data/effect
actually reaches the system meant to use it (her DECISION/voice, not just the display) — with where it's
consumed. **VERIFIED** = proven to work (test/trace/live). "reaches: brain / display / both" is called out
because *wired-to-display ≠ wired-to-brain* (the goals-not-wired bug was exactly this).

## Big recent features — three-state status

| Feature | Compiles | Wired (where consumed) | Verified | Reaches |
|---|---|---|---|---|
| Battle flee floor (anti-wedge) | ✓ | ✓ — `battle_agent` run loop (`_unresolved_turns`→flee/abort) | ✓ **LIVE** (08:20 watch: "Tactical retreat") | brain (hands) |
| Identity flip (play-mode frame) | ✓ | ✓ — `bot._build_self_block` header, all interjections | ✓ **LIVE** (first-person throughout watch) | brain (voice) |
| FIX 1 repetition floor (battle/dialogue/overworld) | ✓ | ✓ — `_skip_streak` move pick; `pokemon_voice` dedup; `dialogue_drive` cycle-detect; `free_roam` repeat nudge | regression 3/3 + dedup unit; **live stuck-break PENDING** | brain |
| FIX 2 run-state → voice/decision | ✓ | ✓ — `_pokemon_react`, `process_and_respond` (pokemon_mode), `_pokemon_choose` | content ✓ vs health.json; **live PENDING** | both |
| 3-tier goal-layers | ✓ | ✓ NOW — decision (`_soul_choose` place-seam) + voice (FIX 2) + dashboard | content ✓; live PENDING | both ⚠ |
| Recalibration (`_active_objective`) | ✓ | ✓ — `free_roam` ctx nudge + health.json + dashboard | **PENDING** (detour→resume live test) | both |
| Strategic-stuck floor + readiness→GO | ✓ | ✓ — `_available_actions` prune + oracle ctx | two-tier logic unit ✓; live lose→return PENDING | brain |
| World-model (`pokemon_world`) | ✓ | ✓ — spatial brief + travel targets into oracle ctx | persists-across-resume claimed, not re-checked | brain |
| Catch procedure (weaken + PP fix) | ✓ | ✓ — `catch_pokemon` | **PENDING** (no catch in watch; `_can_weaken`/`need_pp` unverified live) | brain (hands) |

## HALF-WIRED (highest priority — the "lies" to find before a watch)

1. **Gary nemesis arc — PARTIAL.** `note_rival_encounter` is called ONLY for the opening Oak's-Lab rival
   (`campaign.py:1561`). The Nugget-Bridge / Cerulean Gary and every later rival fight are NOT detected as
   rival encounters → recorded as a generic "trainer" (the watch log said "beat a trainer", not "beat
   Gary"). The grudge machinery (`rival_grudge_note`, persisted) never escalates past encounter #1. NOTE:
   the death-loop WALL tracking works (Gary-as-trainer key), so the *gameplay* fix is fine — it's the
   *narrative grudge arc* that's half-wired. FIX: detect the rival at the known later rival maps/coords (or
   by trainer-class id) and call `note_rival_encounter` there too.
2. **Journey saga (`_pokemon_journey_block`) — chat-only.** WIRED into idle-chat context (`bot.py:7594`)
   but NOT into `_pokemon_react`. So her saga/grudge callbacks fire in conversation but her in-game
   reactions don't reference them. FIX 2 now gives reactions the live health.json state, but not the saga
   beats. (Acceptable for now; flag for the next continuity pass.)
3. **Goals → voice — WAS half-wired, FIXED today.** Goals were display + decision-ctx but never reached her
   voice (she asked Jonny her own goal while the panel showed it). Fixed by FIX 2. Listed as the canonical
   example of this failure class.
4. **HUD goal refresh — display-only quirk.** The stream HUD's goal didn't refresh live; the operator
   dashboard is now the source of truth. Low priority (HUD being redone).

## GATED-OFF (flag default off + why + verified?)

- `POKEMON_FIELD_MOVES=0` — Cut/Surf/Strength actuation. Detection WIRED; actuation UNVERIFIED on a
  long-running core. The two real gatekeepers for gyms 7–8 — need a live verification session.
- `POKEMON_ITEM_PICKUP=0` — overworld item pickups. Reuses the proven talk pattern; UNVERIFIED on long core.
- `POKEMON_GUIDE_SEARCH=0` — silent strategy-guide lookup. Google Custom Search API is DISABLED on the
  project (enable + arm to use). Enrichment only, never a dependency.
- `CATCH_SUBCORE=0` — legacy fresh-sub-core catch fallback (causes a stream jump-cut). Retired to opt-in.

## WIP / deferred

- `travel.py` TODOs: ledge-awareness (one-way `MB_JUMP_*` edges), east-crossing connection-offset row
  targeting.
- Forward gyms 3–8: data-bill staged (`FORWARD_CLIMB_STAGING.md`); coords need live recon; Surf/Strength
  verification session pending.
- "↩ resuming" dashboard pill — parked until recalibration is verified live (Jonny's call).

## BROKEN / DEPRECATED / DEAD (prune candidates)

- `play_live.py --cable` arg — DEPRECATED/UNUSED (emulator audio no longer routes to the cable).
- Repo-root `control_server.py` (~900 lines) — DEAD per CLAUDE.md (superseded by
  `kira/dashboard/control_server.py`); safe to delete in a separate cleanup.
- `pokemon_state.py` offsets marked "CANDIDATE/UNVERIFIED" — de-facto verified by the passing battle
  regression; the comment is stale.
- ~80 `recon_*.py` scripts — methodology archive, not imported by the runtime; clutter, not dead.

## STUCK-VECTOR LIST (FIX 3) — with Fix-1 coverage

| Vector | Where | Covered by Fix-1? |
|---|---|---|
| Depleted / 0-PP move re-pick (Mankey) | `battle_agent._select_and_verify` | ✓ COVERED — `_skip_streak` rotates the whole moveset, flees when exhausted |
| Can't-flee trainer wedge | `battle_agent` run loop | ✓ COVERED — flee floor aborts LOUD |
| **No-effect (type-immune) move** | battle policy | ⚠ PARTIAL — a 0-effect move that still consumes PP "resolves", so `_skip_streak` won't catch it; policy can re-pick it if all score low. NEEDS: exclude effectiveness==0 in the picker |
| Looping NPC (multi-line cycle, Slowbro) | `dialogue_drive` / `pokemon_voice` | ✓ COVERED — cycle-detect (disengage) + recent-window dedup (stop re-commentary) |
| **Re-talking a beaten gym leader / trainer** (restarts lines) | overworld | ⚠ PARTIAL — cycle-detect catches it once she re-talks; better fix = don't re-initiate talk with a beaten NPC |
| Warp / spinner / boulder loops | overworld | ⚠ PARTIAL — repeat-pick nudge (non-GREEN) + ProgressLedger RED→hard-recovery backstop; not a dedicated handler |
| Catch throw-verify / waypoint-wander sub-tick stalls | `battle_agent.throw_ball`, `campaign.catch_one` | ✗ NOT — bounded by budget but spins 5–40s BELOW the watchdog (the "wrong altitude" class) |
| Heal-at-center sub-tick stalls | `campaign.heal_at_center` | ✗ NOT — bounded; repeat-pick nudge won't fire (fp changes mid-walk) |
| ProgressLedger tick-granularity | `world_fingerprint` | ✗ ARCHITECTURAL — sub-tick hangs invisible until the next tick; accept or add a per-action micro-watchdog |

**Next anti-stuck targets** (not yet built): (a) exclude type-immune moves in the picker; (b) "don't
re-talk a beaten NPC" memory; (c) a per-long-action micro-watchdog (sample the fingerprint every ~5s
INSIDE catch/heal so a sub-tick stall surfaces, per the standing "no inner loop spins silently" rule).
