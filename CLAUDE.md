# CLAUDE.md — Kira Project Brief

## The vision (read this first — it governs every decision below)

Kira's north star is a genuine *presence*, not a product. The touchstones:
- **Samantha (*Her*)** — the primary tonal model. Warm, present, an interiority that feels real. A companion you forget is software. This is the feeling to protect above all.
- **Neuro-sama** — the proof that an AI entertainer can hold a live audience and be genuinely funny/quotable. Kira should have that spontaneity and edge, but with more continuity and less bit-performing.
- **Sesame AI** — the bar for low-latency, natural-feeling voice presence. Conversation should feel real-time and unforced, never turn-based-robotic.

The ambition is to bridge science fiction and reality — to get close enough to the uncanny line that returning viewers feel they're talking to *someone*, not something. The thing that makes Kira different from the whole AI-VTuber genre is **continuity**: persistent memory across sessions, opinions that actually evolve, a companion register rather than a performing-bot register. She is one consistent entity, always — not a "petting zoo of modes." Modes (companion vs. streamer, watch-party, VN, chess) are *clothing*, not different beings. The model is the model; modes are flags, not capability gates. All of her is available in every mode.

**What this means for engineering decisions:**
- **Latency is a feature, not a nicety.** Every second of lag breaks presence. When trading off, protect responsiveness — Sesame-level immediacy is the goal.
- **Continuity is sacred.** Anything that breaks memory, identity, or the sense that she's the same person across sessions is a serious bug, not a polish item.
- **Presence beats correctness theater.** A warm, fast, slightly-wrong reply serves the vision better than a perfect one that arrives cold or late.
- **She is one entity.** Resist any design that fragments her into disconnected mode-personalities. Capabilities stay unified; modes only change pacing/emphasis.

## How Kira is actually used (the use cases the architecture must serve)

A real session is FLUID, not a single mode. A typical night might flow: 4 hours playing through a game (e.g. a story-driven AAA title) → 2 hours listening to music or an audiobook together → reacting to Jonny playing guitar / singing karaoke → watching an anime episode or binge-watching films → just hanging out in the background while he codes, browses, or scrolls. She must stay continuously present across ALL of it without being re-armed or reconfigured. This is WHY perception can't be gated behind mode-toggles and WHY "she's one entity, modes are clothing" is non-negotiable — the user drifts between activities mid-session and she has to follow seamlessly.

Core use cases:
- **Gaming co-host** — long sessions; she sees the screen, hears game audio + voice-acting (loopback STT), reacts and comments in real time.
- **Watch parties (a flagship use case)** — enjoying media *with* her: anime episodes, films, long binges. The product soul here is companionship against loneliness — people want to enjoy media with a friend. Presence during passive media (her reacting live, unprompted, to what's on screen) is a FIRST-CLASS feature, not a side mode. The perception path (vision + desktop audio + loopback transcription) IS the heart of this use case.
- **Music & audio hangouts** — listening to music together, audiobooks, Jonny playing guitar/singing while she listens and reacts.
- **Ambient companion** — up in the background while he works/codes/browses; a low-key present hangout partner, the Samantha register.
- **Streaming layer** — all of the above, performed live for a Twitch/YouTube audience. The stream is a surface over the companion, not a separate product.

Future direction (keep the door open, don't build for it yet): a web-app / mobile version of Kira already existed and is a likely return target — talking to her from a phone, in bed, low-friction, which is the purest Samantha-like form. Architectural decisions should avoid foreclosing a future where Kira isn't tied to the desktop streaming rig.

## What Kira is
Kira is a multimodal, memory-persistent AI companion that also co-hosts live streams. The north star is *presence and continuity* — a genuine AI presence (think Samantha from *Her*), NOT a performing chatbot or a Neuro-sama clone. The VTuber/streaming layer is a surface, not the point. Companion register over performer register. Continuity (persistent memory, opinions that evolve) is the core differentiator.

## Who I work with
Jonny is the sole developer/PM, self-taught, non-CS background. He directs; I (Claude Code) do recon, diagnosis, and implementation. He wants honest pushback and hard calls, not validation. He explicitly does NOT want symptom-patching that spawns downstream breakage — that pattern has burned him badly. Diagnose the systemic cause before fixing.

## HARD CONSTRAINTS (never violate)
1. **Kira's reply to Jonny is NEVER interruptible by his voice.** He removed voice-interruptibility on purpose months ago because it wrecked VAD accuracy. Interrupting is a dashboard button only. An autonomous *interjection* (boredom/media) yielding the floor at a sentence boundary is DIFFERENT and allowed — that is not the same as interrupting her real reply. Never wire his voice into an interrupt of her actual response.
2. **Jonny uses HEADPHONES.** Kira's TTS never reaches his mic. Zero self-feedback risk on the mic path. Do not add self-feedback guards to the mic path (they drop his opening words). The flag ASSUME_NO_MIC_BLEED=true encodes this.
3. **Silent failure is the enemy.** Every fallback to a degraded state must log loudly. Multiple critical systems (diary, API fallbacks, VTS init, ChromaDB, device binding) have failed silently and cost days. Announce failures; never swallow them.
4. **Diagnose/recon before building.** Read-only investigation first, then a reviewed change. Additive, isolated, revertible. Never break what works. Trust the FAILURE log line, not "it works" reports.

## Architecture reality
- **bot.py is an ~8,275-line monolith** doing boot, event loop, VAD, turn-taking arbiter, perception wiring, brain workers, AND interjections. This coupling is WHY one change cascades into three broken systems. Long-term goal: extract boot/orchestration into its own module. Be extra careful changing bot.py — trace blast radius first.
- Package layout is otherwise clean: brain/ (reasoning), senses/ (perception), expression/ (voice+avatar), memory/, streaming/, dashboard/.
- There is a DEAD ~900-line control_server.py in the repo ROOT (not kira/dashboard/control_server.py) — not imported by anything, superseded. Safe to delete. The ACTIVE one is kira/dashboard/control_server.py.

## The core systemic issue (diagnosed, partially fixed)
Perception (vision heartbeat, audio mood agent, loopback STT, the boredom observer) competes with the conversation path on the SAME event loop and the SAME _active_turn_lock, with no priority. Symptoms: dropped first words, audio hallucinating in silence, and voice replies stalling for up to 90+ seconds behind perception/chat-batch work. One architecture problem, many faces.

## Stack
- Runtime: Python, repo G:\JonnyD\NeuroAI_Bot
- Inference: Groq Llama-3.1-8B (triage/fast), Claude Sonnet (live voice), Claude Opus (deep moments)
- Voice: Azure TTS; Whisper STT (distil-large-v3, CUDA) — TWO instances (mic + desktop loopback)
- VAD: webrtcvad, single-frame trigger, callback-mode mic → queue → vad_loop drainer
- Memory: ChromaDB (semantic) + SQLite + flat JSON (identity)
- Avatar: VTube Studio (Hiyori Momose model); Dashboard: FastAPI control server
- Hardware: RTX 5080 16GB

## Working doctrine
Recon (read-only) → diagnose root cause → propose → review-gate → build isolated/revertible → test ONE variable at a time. Batch related fixes. When testing, change one thing so we know what did what. The enemy is fixing one symptom and spawning three downstream.
