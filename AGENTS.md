# AGENTS.md — rules of engagement for AI agents in this repo

This is **local Kira** ("OG Kira"): a Python AI VTuber / companion — voice loop, memory,
Twitch/YouTube, VN autopilot, and the autonomous Pokémon FireRed harness that has
**beaten the game start-to-credits by itself** (docs/RUN_STATS_fresh_go_6.md).

## Current mission

**docs/marathon-stream-plan.md** — make the autonomous playthrough a *watchable,
narrated, chat-aware marathon stream* (the "Frankfurt Test"). Read it before anything.

## Read-first list (in order)

1. `docs/marathon-stream-plan.md` — the mission and build order.
2. `CLAUDE.md` (repo root, gitignored — exists only on Jonny's PC) — the accumulated
   operating rules: the run-the-rope loop, the 15-block competency map, the sanctity
   gate. If present, it outranks this file on process questions.
3. `STATE_OF_PROJECT.md` + `NEXT_SESSION.md` (also gitignored, PC-only) — durable
   frontier state from prior autonomous sessions.
4. `pokemon_agent/AUTONOMOUS_GAME_HARNESS.md` — the engine/game-knowledge split and
   the hard-won pitfall list. Every pitfall in there cost real hours; do not re-pay.

## Non-negotiables

- **The canonical save is sacred.** Never write to it directly; all progress flows
  through the checkpoint/banking path (the sanctity gate).
- **Kira's identity is sacred.** Persona and soul stay intact; game/mode behavior lives
  behind toggles.
- **Verify by running, not by reading.** The look-ahead oracle (`recon_longrun.py`,
  headless at max speed) is the proof standard. Claims need logs.
- **Menus by cursor-readback, never state bytes alone** (see pitfall 13 — the
  immortal-battle wedge).
- **No ROMs, saves, or Nintendo assets** may ever enter this repo or its history.

## Runtime facts

- Runs on Jonny's Windows PC (CUDA GPU required for Whisper; VB-Audio Cable; mpv).
- Boot: `.\.venv\Scripts\Activate.ps1` then `python run.py` → dashboard at
  http://127.0.0.1:8766/.
- Overnight autonomous dev used night_shift.ps1 → Claude Code (now unavailable);
  porting it to the Cursor CLI (`cursor-agent`) is an open Session-1 task.
