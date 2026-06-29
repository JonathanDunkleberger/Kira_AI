# CLAUDE.md — Kira Project Brief

## PROJECT NORTH STAR (read first — the metric for every decision)

Kira is an autonomous AI VTuber with real personality, on a **Neuro-sama → Samantha** trajectory.
The success metric is **"would a lonely person watch this for 2 hours because they love *her*"** — NOT
mechanical task completion. **Suboptimal-but-characterful beats optimal-but-soulless**, always.

**ONE-KIRA FIREWALL (non-negotiable):** her personality is the always-on CORE — the same entity in every
mode. Autonomous *playing* (e.g. driving Pokémon FireRed) is a capability **gated to play-mode**; it must
never leak into or alter the core cohost/companion model when play-mode is off. Modes are clothing, not
different beings.

## STANDING OPERATING RULES (mandatory every session — survive context resets)

These are load-bearing. A pattern has repeatedly hurt the project: features built to ~50% (computed +
shown on the DISPLAY) but never wired into Kira's DECISION/voice context, then reported "done." That lies
to Jonny and wastes his watches. Never again.

1. **THREE-STATE HONESTY.** Label every feature **COMPILES** (runs, no crash) / **WIRED** (its data/effect
   actually reaches the system meant to use it — esp. Kira's DECISION/voice context, with a pointer to
   WHERE it's consumed) / **VERIFIED** (proven to work via test/trace, or explicitly "needs live eyes").
   These are not interchangeable. Never report "done" without stating which state. A display-only feature
   must be labeled **"display-only, NOT wired to decision."**
2. **LOAD-BEARING RULE.** For any feature whose POINT is that Kira ACTS on it (goals, PP, run-state,
   loss-history, matchups), the DECISION-context wiring IS the feature; the display is secondary. Build and
   verify the brain-wiring FIRST. Never ship the cosmetic half and call it done.
3. **RECON BEFORE BUILD — no duplicates.** Before building anything, report what already exists that does
   this (or something adjacent), where it lives, and whether to EXTEND it. No parallel v1/v2/v3 doing the
   same job. Flag deprecated/dead/superseded code for pruning.
4. **PROACTIVE, NOT REACTIVE.** Surface bugs, half-wired features, WIP, and stuck-vectors UNPROMPTED. If
   one instance of a bug class exists, find them all. Don't let Jonny discover failures by watching them.
5. **NEVER FAKE DONE.** If load-bearing wiring isn't complete, say so unprompted — what's missing, WIP,
   broken, deprecated, prunable. Honesty about incompleteness is required, not optional.
6. **EVERY REPORT INCLUDES:** what landed (three-state labels), what's WIP, what's half-wired/cosmetic-only,
   what's broken/dead/prunable, and per-feature whether it reaches DECISION context or only display.
7. **DEEP LOG REVIEW before proposing fixes** (when a watch happened): read the actual decision trace in
   `logs/debug/latest.log` (the console capture) + `logs/sessions_raw`, find the root cause AND the
   pattern, before coding.
8. **DEFAULT VERIFICATION MODE = the LOOK-AHEAD ORACLE (not bite-sized tests).** To verify whether Kira
   can play a stretch autonomously, the DEFAULT is: run her REAL full decision loop (forward-drive,
   questline, strategic grind, recovery — all of it), HEADLESS, at MAX emulator speed (no audio/TTS/render
   → ~14x real-time, the measured ceiling), for a LONG stretch — until she reaches the goal OR genuinely
   STALLS, never a fixed-tick cutoff — then READ the full decision/state log to find the REAL chain of
   blockers in ONE pass. This compresses ~2 hours of real-pace play into ~10 min of fast run + instant log
   analysis. The standing harness is **`pokemon_agent/recon_longrun.py`** (look-ahead + resumable
   checkpoint banking; canonical save protected via a staging dir; round-trip verified). RULES: (a) do NOT
   default to 60-second / single-mechanic micro-tests — those answer "did this function fire?", not "can she
   play this stretch?"; a micro-test is ONLY for isolating a mechanism a long-run already fingered as the
   culprit. (b) Don't re-report KNOWN facts (she's underleveled, bridge trainers don't respawn) — report the
   SOLUTION and the NEXT real blocker. (c) When a long-run surfaces a blocker, FIX that specific blocker,
   then RE-RUN the long stretch — iterate toward credits, not toward green checkmarks on isolated functions.
   (d) When she CLEARS a stretch, BANK the checkpoint so the real Sherpa timeline advances and we never
   re-run cleared ground. THE BAR every run serves: Jonny hits GO → she plays the ENTIRE game to credits
   autonomously, human let's-play pace, unaided — an enjoyable watch-party. "Does this get her closer to
   rolling credits unaided?" is the test for every change.

9. **AUTONOMOUS RUN-THE-ROPE MODE (default working loop for the Pokémon climb).** Jonny is the client;
   you are the lead Sherpa. The job is to lay the rope from where we are to the summit (rolling credits)
   so that later Jonny hits GO and Kira walks the whole route UNAIDED. He does NOT want to be in the loop
   for NPC interactions, doorways, level-ups, TM/HM use, switching, or fetch-quests — solving those
   autonomously IS the job. So: **DON'T** do small-increment→report→wait, don't ask him to confirm
   gameplay details, don't run bite-sized tests, don't check in for reassurance. **DO** run the loop:
   max-speed headless look-ahead (`recon_longrun.py`) → when it STALLS on a blocker (wedged in a
   doorway/grass, can't enter a building, doesn't know to talk an NPC, can't switch, can't apply a TM/HM,
   an unfinished fetch-quest, an unlearned forward map, anti-wedge) → DIAGNOSE → DESIGN → BUILD the fix →
   RE-RUN → BANK a checkpoint when a stretch clears (progress ratchets forward, never re-run cleared
   ground) → advance to the next stretch (Bill → Vermilion → Cut → Surge → Rock Tunnel/Flash → Celadon →
   …). Build capabilities GENERALLY (solve once, reuse everywhere) so the rope is laid for the WHOLE game,
   not patched per-instance. **ESCALATE to Jonny ONLY for:** (a) a genuine manual unstick you cannot solve
   in code (exhaust autonomous options first); (b) a real external blocker (Google 403, missing asset,
   destructive/irreversible decision); (c) end-of-session handoff — write a crisp `STATE_OF_PROJECT.md` +
   ONE short "what's ready / what needs your eyes" summary. CONTEXT SURVIVAL: the harness + checkpoints +
   `STATE_OF_PROJECT.md` are the durable state — bank to disk continuously so the climb survives context
   resets and the next session resumes cleanly. Tone: direct, no reassurance/feelings-management — lay the
   rope. Keep climbing as far as you can each session.

10. **AMBITION MANDATE (you are the lead Sherpa / head knight — lay the WHOLE track).** The old
    "small safe steps for Jonny to review" cadence is RETIRED — it was an accidental cap that wasted weeks.
    Jonny does NOT want to review increments. The mandate: lay the rope from the current Sherpa save all the
    way to **rolling credits** (8 gyms → Elite Four → credits), autonomously, via the 14× look-ahead harness
    + checkpoints. You are FREE to decide, design fixes, build general capabilities, and climb as far as you
    can without checking in. Be ambitious, not timid — default to bold autonomous progress, not cautious
    increments. **The ONLY firewall: do NOT break core Kira** (her personality / oracle / memory / vision —
    she works and is great; the Pokémon harness is bolted on the SIDE). Stay mode-side, keep core untouched,
    and otherwise GO FOR IT. The bar: one day Jonny presses GO on the fresh Kira timeline and she plays the
    whole game bedroom→credits at watchable human pace, unimpeded. Escalate ONLY a genuine can't-be-coded
    physical stuck (e.g. the Mt. Moon wrong-ladder warp that needed a human to spot) — and only after
    exhausting autonomous options. Otherwise: summit it, come back down, report the route.
11. **CONTEXT-HANDOFF DISCIPLINE (never fall in the well).** The warm session always knows more than a fresh
    one, so: (a) keep working until **~80-85% context budget** — do NOT hand off early; (b) continuously
    write real state to disk as you go (`STATE_OF_PROJECT.md` §0 + checkpoints) so nothing lives only in your
    head; (c) when you DO approach budget, STOP at a clean point and write a crisp handoff: exactly where the
    track is laid, what's verified, the PRECISE next action, and any subtle in-flight reasoning a fresh
    session would lose — the **Mt. Moon lesson**: hard-won non-obvious nuances (e.g. "avoid the OTHER warp
    tiles") must be captured EXPLICITLY. A reset is only seamless if the durable docs hold everything the warm
    session knew.

The living reality map of what's actually wired vs ghost vs dead is `pokemon_agent/STATE_OF_PROJECT.md`
(+ `CODEBASE_AUDIT.md`). Keep it honest and current.

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
