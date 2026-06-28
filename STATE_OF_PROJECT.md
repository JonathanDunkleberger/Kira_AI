# STATE OF PROJECT — reality audit (2026-06-28)

The honest map of what's REAL vs a disconnected GHOST vs DEAD. Three-state per the operating rules
(`CLAUDE.md`): **COMPILES** (runs) · **WIRED** (data reaches the system meant to use it — esp. Kira's
DECISION/voice, with where) · **VERIFIED** (proven, or "needs live eyes"). The #1 column is **REACHES**:
BRAIN (decision/voice) vs DISPLAY-ONLY vs DEAD. "Exists but unwired" is the most important category —
that's the failure that's burned us (goals were built+displayed but never reached her brain).

Companion docs: `pokemon_agent/CODEBASE_AUDIT.md` (pokemon detail + stuck-vector list),
`pokemon_agent/FORWARD_CLIMB_STAGING.md` (gym 3→8 plan).

---

## 1. HOW IT FLOWS TOGETHER (architecture at decision time)

The Pokémon harness is a **separate subprocess** (`pokemon_agent/play_live.py`) that drives the emulator.
It talks to core Kira over HTTP (`KiraVoice` → control_server). Core Kira owns ALL personality/voice; the
harness owns game mechanics. Four channels:

- **DECISIONS:** `campaign._soul_choose` → `voice.choose` (HTTP) → `/cmd/pokemon_choose` → `bot._pokemon_choose`
  → LLM. The LLM prompt = `_POKEMON_CHARACTER_RULES` + `_POKEMON_DECIDE_FRAMING` + **live run-state block
  (FIX 2)** + the ctx (`place` seam carrying goal-layers/recalibration/wall-awareness from campaign) +
  `_build_self_block` (her mood/want/bond). Returns her pick. → **run-state + goals REACH the brain here.**
- **REACTIONS:** `campaign.on_event` → `voice.emit` (HTTP, **deduped** — FIX 1) → `/cmd/pokemon_event` →
  `bot._pokemon_react` → `_execute_interjection` → LLM. Prompt = `_POKEMON_CHARACTER_RULES` + **run-state
  (FIX 2)** + **saga on tier≥2 (B-4)** + `_build_self_block`.
- **STATE/DISPLAY:** `campaign._publish_health` → `health.json` → `/cmd/pokemon_health` → operator dashboard
  + `/pokemon_hud.json` → stream HUD. The bot ALSO reads `health.json` for the brain (FIX 2) — **one source
  of truth shared by display AND decision.**
- **IDENTITY:** `bot.pokemon_mode` (auto-set True on launch) flips the `_build_self_block` header to
  player-mode; off = cohost (byte-identical). **CONTINUITY:** `voice.journey` → `journey_core.json` →
  `_pokemon_journey_block` (idle chat + now react tier≥2).

The battle ENGINE (`battle_agent`) is deterministic policy (type-chart), NOT the LLM — the oracle is only
consulted for items + (gated) switching. Move selection is the hands; her voice reacts.

---

## 2. POKÉMON HARNESS — feature reality

| Feature | COMPILES | WIRED (where) | VERIFIED | REACHES |
|---|---|---|---|---|
| Battle flee floor (anti-wedge) | ✓ | run loop `_unresolved_turns` | ✓ LIVE (08:20 watch) | BRAIN |
| Repetition floor (FIX 1: 0-PP/dialogue/overworld) | ✓ | move pick + emit dedup + dialogue cycle + roam nudge | ✓ regression 3/3 + unit | BRAIN |
| Ineffective-move aversion (B-1) | ✓ | `_select_and_verify` pick | ✓ offline (Normal→Ghost=0) | BRAIN |
| In-battle party switch (B-1) | ✓ | run loop, gated `POKEMON_BATTLE_SWITCH=0` | matchup math ✓ offline; **actuation needs live eyes** | BRAIN (gated) |
| Run-state → voice/decision (FIX 2) | ✓ | `_pokemon_react`/`process_and_respond`/`_pokemon_choose` | content ✓ vs health.json; live pending | BRAIN+DISPLAY |
| 3-tier goal-layers | ✓ | decision place-seam + voice (FIX 2) + dashboard | content ✓; live pending | BRAIN+DISPLAY |
| Recalibration (`_active_objective`) | ✓ | roam ctx + health + dashboard | pending live (detour→resume) | BRAIN+DISPLAY |
| Strategic-stuck floor + readiness→GO | ✓ | `_available_actions` prune + ctx | two-tier unit ✓; live pending | BRAIN |
| World-model (`pokemon_world`) | ✓ | spatial brief + travel targets → oracle | persists-resume (claimed) | BRAIN |
| Catch procedure (weaken+PP) | ✓ | `catch_pokemon` | **pending live** (no catch in watch) | BRAIN |
| Resolved/looping-NPC guard (B-2) | ✓ | `_drain_overworld`→`_looped_spots`→talk gates | regression ✓; live trigger pending | BRAIN |
| Travel routes around plain blocking NPCs (LAYER A) | ✓ | travel gauntlet→unified `_blocked_npcs`→plan/talk both read it; `no_route_npc_blocked`→oracle | wiring ✓ (shared-by-ref); **live Slowbro state pending** | BRAIN |
| Universal wall-clock watchdog (LAYER B) | ✓ | `wf.StuckWatch`←play_live render feed→`_stuck_request`→roam disengage + travel cancel | unit ✓ 8/8 (frozen-box/Slowbro toggle/legit-read); **live timing pending** | BRAIN |
| Warp/spinner position-loop escape (B-3) | ✓ | `travel` sliding-window → `stuck` | bounded logic ✓; live trigger pending | BRAIN |
| Gary arc at ALL encounters (B-4) | ✓ | `_observed_battle_runner` → `note_rival_encounter` | regression no-false-fire ✓; live rival pending | BRAIN |
| Saga → in-game reactions (B-4) | ✓ | `_pokemon_react` tier≥2 | code path ✓; live pending | BRAIN |
| Identity flip (play-mode) | ✓ | `_build_self_block` header | ✓ LIVE (first-person watch) | BRAIN |

**Gated-OFF (with reason):** `POKEMON_BATTLE_SWITCH=0` (actuation unverified), `POKEMON_FIELD_MOVES=0`
(Cut/Surf/Strength actuation unverified — gym 7/8 gatekeepers), `POKEMON_ITEM_PICKUP=0` (unverified),
`POKEMON_GUIDE_SEARCH=0` (Google Custom Search API disabled), `CATCH_SUBCORE=0` (legacy jump-cut path).

**Pokémon GHOSTS / half-wired:** *(the two big ones are now FIXED today)* — Gary arc (was opening-only →
**FIXED B-4**), saga-in-reactions (was chat-only → **FIXED B-4**). Remaining: HUD goal refresh is
display-only (HUD being redone — low priority).

---

## 3. CORE KIRA — feature reality (from the core audit)

**Good news: the major core features REACH THE BRAIN** (verified by the audit tracing prompt injection):
repetition-awareness (`avoidance_block`), emotional state, current-want, Jonny-bond, sentiment/memory
ledger, entity theories + called-shots, chat director, salience gating, visual perception + staleness,
ambient audio + dialogue summary, running bits, voice guardrails. None of these are ghosts.

**GHOSTS / unwired / aspirational (core):**
- **Dread→struggle→catharsis arcs, vendetta, naivety:** the audit found them only as comments in
  `streamer_overlay.py`, NOT tracked/injected. **CAVEAT:** memory says batch-7 shipped these via
  `persona/private/personality.txt`, which is **GITIGNORED** — so they're live-local in Jonny's persona
  file (reaching the prompt as persona text) but invisible to a code audit and uncommitted. **Status:
  WIRED-via-persona (live-local), NOT in code.** Decision needed: promote to tracked state, or accept as
  persona-only.
- **Activity-Director taxonomy** (`DIRECTOR_TAXONOMY_ENABLED`): shapes the REPLY path only; proactive
  interjections bypass it (always base shape). Partial — reply-only, not a full ghost.
- **`web_search` import** (bot.py ~38): imported per a TODO, never wired to a command. DEAD import — prune.
- **`LOOPBACK_POST_TTS_COOLDOWN_S` / `LOOPBACK_SUMMARY_AGEOUT_S`:** declared in config, grep finds no
  consumption — likely orphaned env vars. Verify/prune.
- **VRAM telemetry:** diagnostic logging only (intentional, not a ghost).

**DUPLICATES / parallel (core):**
- **Self-block split:** interjections use `_build_self_block` (compact); replies assemble self piecemeal
  across `dynamic_context` (~50 lines). BOTH reach the LLM but via different scaffolding — a mood tweak can
  affect replies vs drives differently. Not broken; fragile. **Canonical to keep:** `_build_self_block`;
  recommend replies call the same factory. (Not done — flagged, no behavior bug today.)
- Jonny-bond renders twice in replies (in `get_state_block` ctx + `_build_self_block`) — minor redundancy.

**DEAD/DEPRECATED (confirmed):** repo-root `control_server.py` — the audit confirms it does NOT exist
(good; only `kira/dashboard/control_server.py`). `play_live --cable` arg deprecated/unused.

---

## 4. WIP / deferred (with reason)

- **In-battle switch actuation** — built+wired+gated; needs a live control (savestate + Jonny) before
  arming `POKEMON_BATTLE_SWITCH=1`. Deferred-armed because unverified menu-nav could wedge a battle.
- **Surf/Strength HM actuation** — gym 7/8 gatekeepers; dedicated live-verify session pending.
- **Forward gyms 3–8** — data-bill staged; coords need recon (`FORWARD_CLIMB_STAGING.md`).
- **Full warp/spinner puzzle-solver** — only the loop-ESCAPE is built (can't-get-stuck); the solver
  (route a spinner/warp deliberately) is scoped-next, not needed before gym 3.
- **Catch procedure live-verify** — no catch happened in the last watch; `_can_weaken`/`need_pp` unproven live.
- **Self-block unification** (core) — flagged; no behavior bug, low priority.
- **Type-immune defensive matchup** returns 1.0 not 0 in `_matchup_def` (cosmetic; logic driven by offense).

## 5. PRUNABLE (cleanup candidates, non-blocking)
- `web_search` dead import (bot.py). Orphaned loopback env vars. `--cable` arg. ~80 `recon_*.py` archive
  scripts (not dead — methodology; clutter). Stale "CANDIDATE/UNVERIFIED" comments on de-facto-verified
  `pokemon_state.py` offsets.

---

## 7. QUEUED — post-watch (do not lose)

1. **In-battle switch — dedicated actuation verify → then arm.** The verb is built/wired/gated
   (`POKEMON_BATTLE_SWITCH=0`). Run a controlled live check (savestate: active mon out-typed + a stronger
   reserve) with Jonny watching that the party-menu nav lands the switch and the battle continues. Only
   then set `POKEMON_BATTLE_SWITCH=1`. (E4-blocking until done.)
2. **Gym-3 GymSpec build (Lt. Surge / Vermilion).** Needs live coord-recon (gym door tile + Surge front
   tile + junior count, same method as Brock/Misty) AND Jonny's decision on the trash-can switch puzzle
   (he vetoed hardcoded presses → capability-not-script preferred). Until built, `head_to_gym` grace-wanders
   at Vermilion (no freeze). See `pokemon_agent/FORWARD_CLIMB_STAGING.md`.
3. **Persona-only emotional arcs (dread→struggle→catharsis, vendetta, naivety).** Currently live-local in
   the GITIGNORED `persona/private/personality.txt` (reach the prompt as persona text, but not tracked in
   code or committed). Jonny to decide: promote to tracked `kira_state` arc-tracking (like called-shots),
   or accept as persona-deep. Either way, record the decision here.
4. **Off-thread decision/event HTTP (the DEEPER lag fix).** `_soul_choose`→`voice.choose` and
   `voice.emit`/`on_dialogue` are SYNCHRONOUS blocking `urllib` calls on the MAIN render thread
   (`pokemon_voice.py:271-287`) — every LLM decision freezes game render + music for its duration. The
   post-watch throttle (silent-no-move guard) stops the *rapid* stutter by ending the stuck re-pick loop,
   but a single blocking decision still micro-stutters even during normal play. DEEPER FIX (queued, NOT
   mid-firefight): run these HTTP calls off the render thread (worker thread / async) so LLM latency never
   touches the frame loop. Risky surgery on the live path — schedule a dedicated pass with Jonny.
5. **Lapras/foreknowledge confabulation (HELD for Jonny).** She invents game knowledge she hasn't seen
   this run ("get Lapras"). Source = the play-mode oracle prompt `_POKEMON_DECIDE_FRAMING` (`bot.py:3233`)
   has no "only reference what you've actually encountered this run" grounding. Fix = one line there, but
   it's CORE-KIRA voice + overlaps the gitignored naivety arc → needs Jonny's sign-off before touching.
7. **Warp-routing: Cerulean→Vermilion forward chain — IN PROGRESS, NOT yet traversing (HANDOFF DETAIL).**
   STATUS by part (three-state):
   - **Warp ENGINE — DONE + VERIFIED (offline).** `travel.read_warps(b)` reads the live map-header warp
     table (verified vs disasm: Route 4 = (19,5)→MtMoon/(12,5)→PC/(32,5)→MtMoon, save-coords, null=0).
     World-model has warp edges; `route()`/`next_step()` traverse EDGES∪WARPS; `head_to_gym` executes an
     edge hop OR a warp hop (travel-to-tile + `enter_warp(pick=tile)`); warps learned live + persisted.
     Verified offline: route(Route4→MtMoon(1,1)), next_step→('warp',(19,5)) vs ('edge','north'), save/load.
   - **Live geography cross-checked (DONE):** Cerulean (3,3) connections live = N→Route24 (3,43, Nugget
     Bridge), **S→Route5 (3,23)**, W→Route4 (3,22), E→Route9 (3,27). So **Cerulean→Route5 is a plain south
     EDGE** (head_to_gym already walks it). The Underground Path warp is ON Route 5 (past hop 1). NOTE: the
     disasm route-number export is unreliable (Route24 is (3,43), not the contiguous pattern) → ALWAYS
     cross-check route IDs live. City block (0-10) is reliable (Vermilion=(3,5) etc.).
   - **HARNESS — BUILT:** `pokemon_agent/recon_warptrace.py` — stub oracle picks head_to_gym each tick,
     runs the REAL recovery machinery (Layer-A route-around + watchdog + no-move guard + off-spine), reads
     each map's warps LIVE, no-ops canonical saves, heal-patches HP each tick. `--fight` forces real
     battles; default flees wilds for speed. Confirmed it learns warps live (read Cerulean's 14 warps).
   - **CONFIRMED VERDICT (headless trace, utf-8-fixed harness, misty_done, 2026-06-28):** she does NOT
     traverse. **HOP 1 FAILS at the Cerulean SOUTH EXIT to Route 5:** two PLAIN NPCs at **(26,31)+(27,31)**
     sit on the ONLY gap to the south edge. Layer A correctly IDs them as plain NPCs + marks them, finds
     **no route around** → `no_route_npc_blocked`. The recovery then works CLEANLY (no freeze/crash):
     no-move guard prunes head_to_gym → she diverts → but tick 4 she goes WEST to Route 4 to grind. So she
     never reaches Route 5; **no forward progress to Vermilion.** Recovery is healthy; the EXIT is walled.
   - **THE BUG TO FIX (hop 1, gates everything):** Layer A sticky-blocks plain NPCs assuming they're
     STATIONARY (the Slowbro case) and routes AROUND — but at a SOLE exit gap there IS no around, and these
     Cerulean NPCs may be WANDERERS (sticky-blocking them PERMANENTLY walls the exit even after they step
     off). NEEDED: for an EDGE-CROSSING exit blocked by plain NPCs, do NOT permanently sticky-block —
     instead WAIT/TTL for wanderers to clear and re-attempt the crossing (the FRLG Cerulean south exit IS
     passable once the NPCs shift), and only sticky-block a confirmed-stationary NPC that actually has a
     route around. CONFIRM first whether (26,31)/(27,31) are wanderers (watch their tiles over ~10s in a
     trace) vs a scripted wall. Fix lives in `travel.py` (the no_path/gauntlet branch + `_blocked_npcs` TTL
     for edge-exit NPCs) and/or campaign's no_route handling. This is the SAME family as Slowbro but the
     sole-exit variant the current fix doesn't cover.
   - ⚠️ POSSIBLE LATENT LIVE BUG (separate): the harness crash was `UnicodeEncodeError` ('→' in a logged
     decision-ctx on a cp1252 console). If play_live's console is cp1252, a real run could hit the same —
     verify play_live forces utf-8 stdout (or strip '→' from logged ctx). Harness itself is now utf-8-safe.
   - **EXACT NEXT STEPS for a fresh context:** (a) run `recon_warptrace.py` as a BACKGROUND/long job (it
     progresses, just slowly) OR speed it for tracing — TEMPORARILY shorten the per-tick stall windows
     (`POKEMON_WATCHDOG_STUCK_S` low, travel `TRAVEL_STALL_RETRIES` / no_path waits) so ticks finish fast;
     (b) read tick-1 RESULT + `pos MOVED/NO MOVEMENT` + whether `chokepoint blocker … PLAIN NPC` / OFF-SPINE
     / WATCHDOG fired → does she reach Route 5 (3,23)? (c) IF NOT → real bug: extend the Slowbro/Layer-A
     NPC-gap solution to the south-EDGE-crossing case (the gauntlet's plain-NPC mark+reroute currently
     lives in the no_path branch but the edge-cross path may bypass it). (d) Once on Route 5, let _learn_map
     read its warps live → the UGP warp appears → continue the chain Route5→UGP→Route6→Vermilion, each hop
     cross-checked live (NOT disasm-blind). (e) THEN Vermilion GymSpec coords + the Bill→S.S.Ticket→HM Cut
     questline (separate). NO watch until the headless trace reaches Vermilion (3,5).
8. **Vision confirming-vote + Gemini swap (recon delivered, NOT built).** Layer B is wired for a pixel-vote
   to plug in later; core-Kira vision OpenAI→Gemini swap recon done (valid key = `GEMINI_IMAGE_API_KEY`;
   `google-genai` installed; recommend `gemini-3.1-flash-lite` heartbeat → `gemini-3-flash-preview` escalate).
   Separate step; firewall (all modes). Needs a real-frame test before commit.

## 6. BOTTOM LINE
Core Kira's decision-wiring is healthy (the major features reach the brain). The Pokémon harness had two
half-wires (Gary arc, saga-in-reactions) — both fixed today. The remaining honest gaps are all either
GATED-with-reason (switch/HM actuation pending live verify) or flagged WIP — none are silent "looks done
but isn't." The one core decision outstanding: whether the dread/vendetta/naivety persona arcs should be
promoted from gitignored-persona-only into tracked code.
