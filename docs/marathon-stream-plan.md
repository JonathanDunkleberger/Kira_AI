# The Marathon Stream — FireRed, Start to Credits, While Jonny Is in Frankfurt

**Status: ACTIVE MISSION** (kicked off 2026-07-28 from the web-app session; this doc is the
baton pass to PC-side agent sessions)

## The Goal — the Frankfurt Test

Jonny clicks *Start Streaming*, says hi to chat, and leaves. He can fly to Frankfurt first
class, drink champagne, sleep, and come back — and Kira is still live: playing FireRed at
a human let's-play pace, narrating in character, talking to chat, closing in on the Elite
Four. No dead air, no visible crash, no baby-sitting. Start of stream → Hall of Fame
credits, fully autonomous.

**The bar (from autopilot-phase2):** a stranger can watch any 20 minutes with dead chat
and not click away — sustained for the whole marathon.

**Why:** it's art. And it's the funnel — the marathon is the attention event that drives
people to the web app (xoxokira.com). Clips from the VOD are the ad campaign.

## Where We Are (receipts, not vibes)

- **The engine is PROVEN.** `fresh_go_6` (2026-07-15, docs/RUN_STATS_fresh_go_6.md):
  bedroom → 8 badges → E4 → **Hall of Fame credits**, autonomously. Venusaur 87,
  $29,120, 3,850 battles, ~6h08m wall at ~14× headless speed.
- **The show layer is ~70% BUILT** (code audit 2026-07-28 — this corrects the earlier
  claim that it didn't exist):
  - **Windowed 1× play exists.** `pokemon_agent/watch.py` and `go.py` launch a pygame
    window ("Kira plays Pokemon - SOUL ON", 3× scale) with a real-time ~60fps pacer
    (`play_live.py` ~488-510, `POKEMON_FPS_CAP`). Headless 14× is only the oracle mode
    (`recon_longrun.py`).
  - **Narration rail exists.** `pokemon_agent/pokemon_voice.py` (salience tiers T0-T3,
    fire-rate limits, stale-drop) → POST `/cmd/pokemon_event|pokemon_choose` →
    `kira/bot.py::_pokemon_react` (~3271) → LLM reaction → TTS. Travel dead-air filler
    exists ("TRAVEL MUSE" in `travel.py` ~900+). Recovery paths speak in character
    (escape-hatch / deep-wedge lines in `campaign.py` ~14628-14698).
  - **Chat runs concurrently.** Two processes: `run.py` (bot: Twitch chat, TTS, VTS,
    control server :8766) + supervisor→`play_live.py` (mGBA in-process). `watch.py`/
    `go.py` refuse to start if the bot is down. pokemon_mode keeps mic + chat alive
    (`control_server.py::_apply_pokemon_mode` ~976-983).
  - **Crash recovery exists.** `supervisor.py` relaunches with `--resume` from the
    banked save on hang (health.json stale >300s) or crash; in-process StuckWatch +
    roam-disengage + deep-wedge revert handle soft stalls in the same window.
- **The dev loop changed.** night_shift.ps1 drove autonomous Claude Code sessions;
  Claude Code is no longer available (banned). Cursor (IDE agent + `cursor-agent` CLI)
  is the replacement driver.

## LAUNCH DOCTRINE (learned the hard way, 2026-07-28 soak)

The first live soak (2h25m: Brock beaten, died mid-Cerulean, stayed dead) taught two
laws. Get these right before anything else:

- **`watch.py` is a COUCH TEST, never a show.** It runs `play_live` bare — no
  supervisor, so the first crash ends the show permanently — and it plays inside a
  DISPOSABLE sandbox under `%TEMP%\kira_watch\` that NEVER writes canonical
  `states/campaign`. Progress from a watch session must be rescued with
  `python pokemon_agent/promote_bank.py <sandbox_dir> <label>` before it evaporates.
- **The stream/marathon launch is the supervised free-roam:**
  `python run.py` (bot first), then
  `python pokemon_agent/supervisor.py --timeline sherpa --audio`
  — windowed, true speed, banks canonical progress continuously (~every 5 ticks +
  ~12-min checkpoints), auto-relaunches on crash/hang/window-close with `--resume`.
- **`go.py` (showtime spine) is NOT marathon-ready:** `build_segments()` ends at
  `beat_misty` — the show would declare itself complete after badge 2. Extending the
  spine is optional later work; sherpa free-roam is the marathon path today.
- Crash forensics live in `logs/debug/playlive_crash_*.log` and
  `logs/debug/playlive_faulthandler.log` (native SIGSEGV class leaves no traceback).

## The Gap — what is ACTUALLY missing (audit-verified, effort-ordered)

### 1. The 1× soak (validation, not code — DO THIS FIRST, costs $0 agent time)
The credits proof ran headless at 14×. STATUS 2026-07-29: **first soak done** — 2h25m
live (Brock, Mt. Moon, Cerulean, two Misty attempts) with good narration. Punch list it
produced: (a) zero catches — root-caused to the missing Squirtle archetype in
`frlg_team_plan.json`, FIXED 2026-07-29; (b) process death stayed dead — launch
doctrine above; (c) third-person addressing — couch rule added to
`_POKEMON_CHARACTER_RULES`; (d) assistant-voice flips — Claude stream failures falling
to Llama, rescue retry added + `ENABLE_CLAUDE_STREAMING=false` mitigation.

### 2. Config + small wiring (one short agent session)
- `POKEMON_AGENT_ENABLED` defaults **false** (`kira/config.py:122`) — must be true or
  she plays mute. Add to the stream-day checklist.
- **Chat-vs-narration priority:** arbitration today is generic turn-taking
  (`is_speaking`, `_ok_to_self_speak`, chat catch-up bank). Missing: on T3/climactic
  events (badge fight, E4, rival), defer chat replies; on calm travel, chat owns the
  floor. Small patch in the bot's speaking arbitration.
- **Intensity reuse (optional):** VN autopilot's System-1 energy model
  (`kira/modes/vn_autopilot.py`, `kira_state.SessionIntensity`) is NOT driven by the
  pokemon agent — it has its own parallel tier system. Mapping T0-T3 → SessionIntensity
  buys the calm/building/climactic/aftermath pacing for free.

### 3. Seamless supervisor restart (the one real remaining build)
In-process recoveries are already watchable (same window, spoken lines). But a
supervisor-level kill+relaunch drops the pygame window and cold-opens a new one —
seconds of dead screen. Needed: freeze-frame or OBS scene fallback + a scripted
in-character re-entry line on every `--resume`. This is the only "fatal-looking on
stream" case left.

### 4. Team-building reliability (recon 2026-07-29 — verify live, then patch)
The catch engine exists (BattleAgent.catch_pokemon, catch_one, TeamPlanner, keeper
router, shiny catch-at-all-costs — all default-ON). The archetype data bug is fixed
(squirtle/charmander branches added; simulation verified `catch_keeper: abra @ Route
24/25` for her exact resume state). Remaining hardening, in order — each verified with
a live segment before the next:
- **Free-roam nursery force:** the LLM chooser only *softly* prefers catching; the
  winning headless run HARD-preferred `fetch_keeper`/nursery `wander_catch` when
  party < 4 + balls (`recon_longrun.py:454-474`). Port that priority into
  `free_roam`'s action choice.
- **Forward-drive exemption:** `_available_actions` prunes `wander_catch` during open
  questlines (`campaign.py:11536-11542`) — exactly when Abra is due post-Misty. The
  data fix arms `_plan_wants_prebuild`, which should stand it down; VERIFY live, and
  if pruning still wins, add the thin-team nursery exemption.
- **Grind cap:** past runs over-leveled the ace (Venusaur 87 vs the plan's Champion-60
  milestone). Enforce `level_milestones` as a ceiling for `grind_to`/overlevel prep
  (ace stops at milestone + small margin). This is the single biggest 80h→50h lever.

### 5. Pace & length tuning (after it's watchable, not before)
Data: 22,112s wall @ ~14× ≈ **~86 hours at true 1×**. HLTB human average is ~41h.
Jonny's target band: **50-60h marathon**. Known fat to cut: 34 `enter_league` false
starts, 73% of decisions were travel (routing waste), grind inefficiency.
Also on the table: **director's pace** — run travel/grind at 1.25-1.5× emulator speed
(reads as brisk walking on stream) and hard 1× during battles/dialogue/story. A long
marathon is not a bug (subathon energy) — but aim for the 50s.

### 6. Marathon ops (the week-of checklist)
OBS scene (game + Kira model + caption overlay), stream title/announcement, VOD →
clips pipeline (scripts/cut_clips.py + transcribe_vod.py exist), and the funnel: pinned
chat message + periodic plug → xoxokira.com.

## How We Work Now (the operating loop, post-Claude-Code)

- **Driver:** Cursor on the PC. Interactive sessions in the IDE; overnight autonomy by
  porting night_shift.ps1 from the `claude` CLI to `cursor-agent` (Cursor CLI). The
  port is a Session-1 task — same loop: read frontier doc → work → write report →
  relaunch. Keep the brake logic.
- **Rules of engagement:** AGENTS.md at repo root (tracked) + the local CLAUDE.md /
  STATE_OF_PROJECT.md on the PC (gitignored — they never left that machine; READ THEM
  FIRST, they carry the run-the-rope loop and the 15-block competency map).
- **Verification doctrine unchanged:** the look-ahead oracle (recon_longrun headless at
  max speed) is how claims get proven. No "should work" — run the rope, read the log,
  bank the checkpoint.
- **The firewall:** Kira's core identity is sacred. Mode behavior stays behind toggles.
  The canonical save is protected by the sanctity gate — never write to it directly.

## Session 1 Checklist (first Cursor session on the PC)

Budget rule: agent tokens are expensive; Kira runtime is pennies. Prefer letting HER
run (soaks, oracle segments) over speculative agent exploration. The audit pointers in
"The Gap" above are line-accurate — don't re-derive them.

1. `git pull` — this doc + AGENTS.md arrive.
2. Read local CLAUDE.md + STATE_OF_PROJECT.md + NEXT_SESSION.md (gitignored, PC-only).
3. Boot sanity: `.\.venv\Scripts\Activate.ps1; python run.py` → dashboard at
   http://127.0.0.1:8766/ — confirm keys/TTS/STT alive after a month idle. Set
   `POKEMON_AGENT_ENABLED=true`.
4. Start the Gap #1 soak: `python pokemon_agent/watch.py` (windowed 1×, soul on) and
   let it run 2-3 hours while doing Gap #2 wiring in parallel. Collect the punch list
   from the soak before doing ANY speculative fixes.
5. Then Gap #3 (seamless restart). Do not touch pace tuning (Gap #4) until #1-#3 are
   demo-able end-to-end.

## The path to air (budget-shaped)

1. Session 1 (one agent day): soak + small wiring + restart UX punch-fix.
2. Free validation: Jonny watches the soak VOD like a viewer; punch list.
3. Half session: fix the punch list.
4. **Pilot stream** — "Kira plays FireRed by herself, Ep. 1: bedroom → Brock" (4-6h,
   announced). Clippable proof, low stakes, builds the marathon audience.
5. Marathon the following weekend.

## Definition of Done

A scheduled stream where Jonny presses Start, says hi, closes the laptop — and the next
human input of any kind is him joining his own chat as a viewer. She plays to credits.
That's the art. Then we clip it, funnel it, and sell the web app.
