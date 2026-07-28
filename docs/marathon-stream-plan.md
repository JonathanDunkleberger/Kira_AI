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
- **The show does not exist yet.** That run was the look-ahead oracle: invisible,
  silent, max-speed, no narration, no chat, no rendering. Watchdog restarted stalled
  segments 2 times out of 5 — fine headless, fatal-looking on stream.
- **The dev loop changed.** night_shift.ps1 drove autonomous Claude Code sessions;
  Claude Code is no longer available (banned). Cursor (IDE agent + `cursor-agent` CLI)
  is the replacement driver.

## The Gap — build list, in order

### 1. Show-pace run mode (the foundation — build first)
A run mode with the emulator visible/capturable at let's-play pace. Everything was
tuned at 14× headless; frame/timing assumptions MUST be revalidated at 1× (actuation
waits, cursor-readback timing, battle text pacing). Deliverable: a 1-hour 1× segment,
zero interventions, watchable window OBS can capture.

### 2. Narration rail (the voice of the run)
Pipe the decision loop's events — PICKs, battles, catches, evolutions, badge moments,
close calls, whiteouts — into Kira's soul/oracle as narration beats, spoken via the
existing TTS path. Gate by the phase-2 System 1 energy model:
- calm/travel → ramble, theorize, talk to chat, running bits
- building/gym approach → anticipation
- climactic (badge fight, E4, rival) → near-silence or ONE heavy line, never both
- aftermath (badge earned, evolution) → process it, celebrate, callback
Plus System 2 dead-chat behavior: she must carry an empty room without it feeling sad.

### 3. Stall-proofing on camera
The watchdog's segment restarts must become invisible: an in-character beat ("hold on,
let me get my bearings") + fast resume from the banked checkpoint, not a frozen screen.
The pitfall bible is pokemon_agent/AUTONOMOUS_GAME_HARNESS.md — the strand/wedge classes
documented there are the enemies; the cursor-readback doctrine is the law.

### 4. Chat on the couch
Reuse the core bot's Twitch chat handling during play. Priority: narration owns
climactic moments; chat replies own calm stretches. She acknowledges regulars, riffs on
chat theories, and plugs the web app organically a few times per stream, never spammy.

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

1. `git pull` — this doc + AGENTS.md arrive.
2. Read local CLAUDE.md + STATE_OF_PROJECT.md + NEXT_SESSION.md (gitignored, PC-only).
3. Boot sanity: `.\.venv\Scripts\Activate.ps1; python run.py` → dashboard at
   http://127.0.0.1:8766/ — confirm keys/TTS/STT alive after a month idle.
4. Engine sanity: one short recon_longrun headless segment — confirm the oracle still
   reaches GREEN progress ticks from the canonical save.
5. Start Gap #1 (show-pace mode). Do not touch Gap #5 until #1-#3 are demo-able.

## Definition of Done

A scheduled stream where Jonny presses Start, says hi, closes the laptop — and the next
human input of any kind is him joining his own chat as a viewer. She plays to credits.
That's the art. Then we clip it, funnel it, and sell the web app.
