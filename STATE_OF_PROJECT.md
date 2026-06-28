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
   - **BUILT + VERIFIED (offline) 2026-06-28 — GENERAL gate-unlock questline capability (Phases 1-4,
     commits 5b1100c→ac7f2e0).** New `pokemon_agent/questline.py` + `gamedata/frlg_gates.json` (curated
     disasm KB) + `campaign` wiring: **recognise** a typed Gate (HM_OBSTACLE / STORY_NPC / ITEM_GATE /
     BADGE_GATE) → **derive** an ordered questline from the KB capability chain (live-cross-checked, prereqs
     first) → **execute** it via `head_to_gym` (routes the unlock ERRAND instead of the gated wall, reusing
     travel). VERIFIED headless on the live Cerulean save: she recognises the Slowbro story-gate, OPENS the
     S.S.-Ticket questline, narrates it in character ("I need the S.S. Ticket — a guy named Bill, north…"),
     and drives NORTH to Cerulean's Nugget-Bridge edge (reverses off the south wall), persisting across
     ticks + a blackout, reaching her DECISION ctx via the place seam + health.json (dashboard). Self-clears
     when `FLAG_GOT_SS_TICKET` reads set. Generalises to Surf/Strength/Fly/Flash/item-gates by the SAME
     pipeline (proven via synthetic-KB test). GuideSearch is wired as the secondary deriver fallback
     (no-op until the Custom Search 403 clears). `POKEMON_QUESTLINE=0` disables. **NOT yet live-verified:**
     the Nugget-Bridge gauntlet → Bill's cottage → ticket COMPLETION needs a healthy live run (heal between
     the un-fleeable trainers); the shipped KB is Cerulean/Bill/Cut only (other gates added disasm-checked
     as she nears them).
   - **FIXED 2026-06-28 — PROACTIVE FORWARD DRIVE (the backward-grind root fix). REACHES: BRAIN.** The live
     bug: post-Misty she WANTED 'grind on the way' → chose `travel:3,22` (Route 4, a cleared dead-end BEHIND
     her); she walked backward to grind, never bonked the Slowbro south gate, so the questline never opened.
     ROOT CAUSE (recon, not symptom): the gate/questline was recognised **only REACTIVELY** inside the
     `head_to_gym` execution branch, so at DECISION time `_available_actions` offered the backward grind
     (`travel:*`/`battle`/`wander_catch`) on EQUAL footing with `head_to_gym`, and a grind-want picked
     backward. Worse, the canonical save sits ON Route 4 (a side-branch WEST of base camp) and `head_to_gym`'s
     own routing would walk the local 'south' edge to Route 3 — **further backward**. THREE-PART FIX (all
     mode-side `campaign.py`, firewall intact, flag `POKEMON_FORWARD_DRIVE=1`):
       (1) `_ensure_forward_questline(state)` — recognises the forward (south) gate and OPENS the questline
           PROACTIVELY each tick BEFORE the action set is built (no longer waits for a wall-bonk).
       (2) `_available_actions` forward-drive — when a forward-unlock questline is open OR she's drifted
           off-branch (graph can't route to the gym yet AND she's off the base-camp city), `head_to_gym` is
           reframed as the DOMINANT forward pull and the backward-grind options are PRUNED (travel targets
           no closer to base camp + standalone grind; grind now happens ON THE WAY via the forward march).
           Stands down for survival (critical-heal) + the strategic-stuck floor (which owns the lost-
           repeatedly case). Strictly conditional/reversible (feature OFF restores the full set).
       (3) `head_to_gym` FORWARD-SPINE recovery + `_base_camp(state)` — when the graph can't route to the gym
           city yet, route toward the base-camp city (GYM_SPINE predecessor, e.g. Cerulean for Vermilion)
           instead of blindly walking 'south' into a further-backward branch; the proactive questline takes
           over once she's there. VERIFIED headless from the ACTUAL live Route-4 save (`recon_forward_drive.py`
           end-to-end: Route 4 → EAST to Cerulean → questline OPENS → heads NORTH toward Bill, never backward
           to Route 3; `recon_forward_drive2.py` action-set: backward pruned, forward kept, reframed,
           reversible). Reaches her DECISION ctx (the reframed `head_to_gym` description + questline narration
           via the place seam) AND the dashboard (health.json `questline`/`rationale`). Fixed a latent `→`
           UnicodeEncodeError in the new log line (now ASCII `->`). NOTE: she ends short of Bill in 8 ticks
           because her L8/L10 teammates lose the un-fleeable Nugget-Bridge gauntlet (real game difficulty —
           team is underlevelled, not a fix bug); the heal floor correctly interrupts + RECALIBRATE resumes
           the questline objective after.
   - **FIX (BUG 2) 2026-06-28 — dashboard RATIONALE freshness. REACHES: DISPLAY (already WIRED; lag fixed).**
     RECON FINDING (contra the handoff's "not wired" premise): the live "why I'm doing this" rationale was
     ALREADY fully wired end-to-end and committed (081dfd7): `campaign._rationale_line` → `self._rationale` →
     `health.json` (`_publish_health`, line ~4385) → `pokemon_proc.health()` `game` → control-server
     `pokemon_health` → `web_dashboard/index.html` renders `g.rationale` (line 768); the `/` dashboard is
     served `no-store` (not browser-cached). The one real defect was a 1-tick LAG: `_publish_health` runs at
     the TOP of the tick (for the watchdog light) BEFORE the pick/rationale exist, so the dashboard showed the
     PREVIOUS tick's 'why' during the (visible) action execution. FIX: re-publish health right after the
     rationale is computed (before the action runs) so the dashboard reflects the CURRENT decision live.
     VERIFIED: `health.json` carries a fresh non-empty `rationale` after a run (+ the `questline` field so
     Jonny reads WHY she's off the direct path). Dashboard pixel-render is code-traced (no-cache + `g.rationale`
     bound) — only literal live-eyes pending.
   - **GROUND TRUTH RESOLVED 2026-06-28 (pret/pokefirered disasm + live RAM): the immediate gate is a
     STORY-GATE, not Cut. `kira_campaign.state` is NOT mis-positioned — it's a valid post-Misty/pre-Bill
     state.** `CeruleanCity_MapScripts → CeruleanCity_OnTransition` calls `CeruleanCity_EventScript_BlockExits`
     **on every map entry while `FLAG_GOT_SS_TICKET` (0x234) is UNSET**, which does
     `setobjectxyperm SLOWBRO 26,31` + `LASS 27,31` + `POLICEMAN 30,12` — deliberately parking the Slowbro
     (gfx 0x81) on the sole south gap to WALL the exit until you fetch the S.S. Ticket. So the correct
     canonical next step from here is **NORTH** (Route 24 Nugget Bridge → Route 25 → Bill's house →
     S.S. Ticket); the game FORCES north-first. Cut (HM01) comes LATER, from the S.S. Anne captain in
     Vermilion. The cut tree at (26,32) (gfx 95, flag `FLAG_TEMP_13`) is a SECONDARY/adjacent obstacle
     (the LittleBoy hint "if Slowbro wasn't there, could cut tree"); whether the post-ticket south road
     also needs Cut for it is a residual nuance (geometry leans yes; the canonical walkthrough walks south
     freely post-ticket) — does NOT change the immediate action (go north). So this hop is a STORY/QUESTLINE
     gate she must learn to follow, not a nav bug and not (yet) a Cut gate. See the gate-unlock design.
   - **RE-DIAGNOSED 2026-06-28 (the handoff's "two wanderer NPCs" call was WRONG — STOP-and-report):**
     the Cerulean south exit is **gated by a CUTTABLE TREE**, not blockable NPCs. Source-confirmed via
     live RAM + the rendered frame: the sole gap in the south fence is tile **(26,32)**, occupied by an
     object event with **graphicsId 95 = `GFX_CUT_TREE`** (`field_moves.GFX_CUT_TREE`, source-cited from
     pokefirered). It is flanked by two real NPCs at (26,31) gfx 129 (a pink Pokémon) + (27,31) gfx 22 (a
     person). Reachability proof (live `kira_campaign.state`, player wedged at (27,30)): her 528-tile
     reachable area reaches **ZERO** south-edge tiles without the tree; with the tree cut, all Route-5
     exits (cols 15-18, 29-32) open. So the tree is **THE SOLE south gate.**
   - **WHY THE HANDOFF MISDIAGNOSED IT:** the prior trace booted `workshop/misty_done` (player far at
     (31,22)). At that distance the cut-tree object isn't spawned (FRLG object-spawn radius) — only the two
     flanking NPC objects load — so the trace saw "two NPCs on the gap" and guessed "wanderers." The live
     campaign save (player adjacent at (27,30)) spawns the tree and reveals the truth. Confirmed the NPCs
     are STATIONARY (0 movement over 90s), same elevation (no z-mismatch), and talking/pushing does nothing.
   - **THE PRESCRIBED "wait/TTL for wanderers" FIX MUST NOT BE BUILT** — a tree never moves; waiting would
     spin forever. The Layer-A sticky-block isn't the bug either; it's just mislabelling a Cut obstacle as
     a "plain NPC."
   - **WHAT'S ACTUALLY NEEDED (two parts, both bigger than a travel.py tweak):**
     1. *Recognition (small, correct, do-able now):* in `travel._npc_tiles`/Layer-A, treat object-event
        gfx **95/97/92** (cut-tree / boulder / item-ball) as HM/field obstacles — NOT plain NPCs to
        permanent-block. `field_moves.scan_field_objects` already detects them (source-cited). Surface
        "there's a Cut tree here; I need Cut" to the oracle instead of sticky-blocking. This makes the
        diagnosis HONEST in-game but does **not** unblock the chain by itself.
     2. *Progression (the real gate — WIP, currently OFF):* she does **NOT** know HM01 Cut (party moves
        carry no move-id 15; she DOES have the Cascade badge, so the badge gate is satisfied — she just
        lacks the move). Passing the tree needs the **Cut questline** (Bill on Route 25 → S.S. Ticket →
        S.S. Anne in Vermilion → HM01 Cut) AND Cut **actuation** (`field_moves`, `POKEMON_FIELD_MOVES=0`,
        actuation unverified on a long-running core). NOTE the apparent chicken-and-egg (Cut comes from
        Vermilion, which is south past the tree) — needs a ground-truth pass on the intended FRLG route
        (is there an alternate early path, or is this save mis-positioned?). **This is a real progression
        wall, not a nav bug — Cerulean→Vermilion CANNOT traverse headless until Cut + actuation exist.**
   - **LATENT LIVE CRASH — FIXED 2026-06-28:** `play_live.py` did NOT force utf-8 stdout, and `campaign.py`
     logs `→` in goal/plan ctx lines (2994/3747/4262). On a cp1252 console that raises `UnicodeEncodeError`
     and kills the run on that tick (the same crash that hit the trace harness). Added the utf-8
     `sys.stdout/stderr.reconfigure` guard at the top of `play_live.py` (isolated, additive; syntax-checked).
   - **EXACT NEXT STEPS for a fresh context:** (a) DECISION for Jonny/PM: build the Cut questline +
     field-move actuation (the real unblock), or re-confirm the intended FRLG Cerulean→Vermilion route
     first (recon whether an alternate early path exists / whether `kira_campaign.state` is mis-positioned
     in a Mart pocket). (b) Land the small "recognise gfx 95/97/92 as field obstacles, don't plain-block"
     change in travel/Layer-A so the in-game state is honest. (c) Only after Cut works: continue the chain
     Route5→UGP→Route6→Vermilion, each hop cross-checked live. NO watch until the headless trace reaches
     Vermilion (3,5). Recon scripts for this live in `pokemon_agent/recon_cerulean_*.py` +
     `recon_choke_verdict.py` (read-only).
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
