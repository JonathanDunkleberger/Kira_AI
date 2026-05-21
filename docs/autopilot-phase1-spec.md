# Autopilot Phase 1 — Dialogue-Only Autoplay with Choice Failsafe
**Status: IMPLEMENTED** (see `vn_autopilot.py`)

## Goal
Prove the mechanical loop: Kira can read a VN autonomously, react at a flat rate, advance dialogue, pause safely at choices, and stay responsive to voice input throughout.

---

## What Phase 1 Does

**The loop:**
Screenshot → classify screen → handle → advance → repeat

**Screen classification** (GPT-4o-mini vision):
- `DIALOGUE` — a text box is present; read and possibly react
- `CHOICE` — a decision menu; trigger failsafe, hand off to Jonny
- `SAVE_PROMPT` — a save/load dialog; advance past it automatically
- `TRANSITION` — animated transition; wait and continue
- `UNKNOWN` — anything else; pause briefly and retry

**Dialogue handling:**
- Transcribe the text box
- Deduplicate (skip if same as previous box)
- Decide whether to react (~1 in 5 boxes by default; prompt-guided decision)
- If reacting: generate a short reaction via Claude, speak it via TTS, log it
- Pacing delay (base + random max, configurable per slider)
- Advance (space / enter / left click — configurable)

**Rolling narrative summary:**
- Every 15 dialogue boxes, summarise the last N boxes into a ~150-word plot summary
- Stored as `vn_narrative_summary` on the autopilot instance
- Fed into reaction decisions so Kira has context for what she's reacting to

**Choice failsafe:**
- Stops the loop immediately
- Kira speaks a handoff line ("there's a choice — Jonny, over to you")
- Calls `on_failsafe` callback; bot sets `autopilot_paused_for_input = True`
- Dashboard shows PAUSED state with a Resume button
- `resume_after_failsafe()` restarts the loop from the next screen

**Dashboard controls:**
- Enable/disable toggle
- Advance key selector (Space / Enter / Left Click)
- Base pacing slider (0.5–5.0 s)
- Max additional pacing slider (3.0–12.0 s)
- Status label (RUNNING / PAUSED / OFF)
- Resume button

---

## What Phase 1 Does NOT Do
- Dynamic energy variation (flat reaction rate) — Phase 2
- Solo/dead-chat behavior — Phase 2
- Theory-building — Phase 2
- Per-character emotional attachment — Phase 2
- Narrative weight detection — Phase 2
- Audio understanding — Phase 2
- Soft-pause for Jonny conversations — Phase 2

**Phase 1 is a success when the loop runs cleanly and the failsafe works reliably. Flat and robotic is fine here — that's the point. Phase 2 is what makes it watchable.**

---

## Key Files
- `vn_autopilot.py` — `VNAutopilot`, `VNInputController`
- `bot.py` — `_autopilot_speak`, `_autopilot_on_failsafe`, `_autopilot_watchdog` callbacks
- `dashboard.py` — `_build_left()` AUTONOMOUS VN MODE section

## Recommended First Test Game
**Planetarian** — gentle, fully kinetic, no choices, clear emotional beats, short. Ideal for tuning the reaction rate and verifying the loop before attempting anything with branch points.
