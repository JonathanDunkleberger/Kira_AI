<h1 align="center">KIRA</h1>

<p align="center">
  <strong>A multimodal, memory-persistent AI VTuber that listens, sees, speaks, emotes, and streams — running on local-first hardware.</strong>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> · <a href="#setup">Setup</a> · <a href="#customization">Customization</a> · <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Llama_3.1-8B_Q4-b070e0" alt="Llama" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Memory-d4a843" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/CUDA-Required-76b900?logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Twitch_+_YouTube-Integrated-9146ff?logo=twitch&logoColor=white" alt="Twitch and YouTube" />
  <img src="https://img.shields.io/badge/VTube_Studio-Live2D-ff6fb5" alt="VTube Studio" />
</p>

---

<p align="center">
  <img src="VTuber Demo - Kirav3.gif" alt="Demo of Kira AI in action" width="720" />
</p>

---

Kira is not a chatbot. She is a real-time cognitive agent with long-term semantic memory, computer vision, full voice interaction, mood-driven Live2D expressions, on-screen synced captions, and proactive autonomous behavior. She runs a local LLM on consumer GPU hardware, remembers facts about her user across sessions, watches the screen to understand context, drives a Live2D avatar's face from her emotional state, and co-hosts live streams across Twitch and YouTube — keeping core reasoning private and on-device.

This project demonstrates end-to-end systems design: real-time audio pipelines, vector-database memory architectures, multimodal sensor fusion, WebSocket-driven avatar control, synced caption rendering from TTS word-timing, and agentic decision loops — built from scratch in Python.

---

## Architecture

<p align="center">
  <img src="docs/kira-architecture.svg" alt="Kira System Architecture" width="100%" />
</p>

The system is organized into five layers:

**Experience** — Voice chat with real-time mic capture and WebRTC VAD; a vision agent that watches the screen and injects context; a Live2D avatar (VTube Studio) whose facial expressions are driven by Kira's emotional state; Neuro-style on-screen captions synced word-by-word to her speech; ambient desktop audio awareness (loopback transcription and media-watch) so she can react to what she's hearing on screen; multi-platform live integration across Twitch and YouTube; and a customtkinter dashboard for real-time controls and state monitoring.

**Cognitive Core** — Hybrid inference: a local Llama 3.1 8B (Q4_K_M) handles fast on-device reasoning, with cloud models (Claude, GPT) routed in for deeper moments. An emotional-state system drives mood-aware responses and avatar expressions. The prompt system assembles identity from `personality.txt`, formatting rules, and retrieved memory. A proactive agency loop (the Universal Boredom Protocol) monitors silence and escalates through behavioral stages autonomously, including curiosity-driven questions to chat and the co-host.

**Memory** — ChromaDB stores extracted facts (not raw transcripts) as sentence-transformer embeddings across `facts`, `chatters`, and `turns` collections. The fact extractor uses the LLM to distill durable knowledge from conversation; session summaries are generated at shutdown and folded back into memory. Per-playthrough memory tracks narrative beats for VN sessions separately from general facts. Relevant memories are injected into each prompt via semantic retrieval. Startup validation runs real vector queries so index corruption is caught loudly, not silently mid-stream.

**Tools** — Music playback via yt-dlp and mpv on natural-language request; autonomous web search when knowledge gaps are detected; Twitch utilities (polls, predictions); a VN autopilot that OCRs the screen, reads visual-novel dialogue aloud in-character, and advances autonomously; a game-mode controller that adapts pacing and verbosity per activity; and optional rate-limited chat posting.

**Infrastructure** — Llama 3.1 8B (Q4_K_M) for local reasoning, Faster-Whisper (large-v3, CUDA float16) for speech-to-text, Azure Neural TTS (with Edge-TTS fallback) for voice output, GPT-4o-mini for vision/screen understanding, and the VTube Studio WebSocket API for avatar control.

---

## What Makes This Interesting

### Local-First Inference
Runs Llama 3.1 (8B, Q4_K_M) on-device via `llama-cpp-python` with Flash Attention. Core conversation stays private and low-latency; heavier cloud models are routed in only for deep moments.

### Persistent Semantic Memory
A ChromaDB vector database stores extracted facts across sessions. The extractor uses the LLM itself to distill durable knowledge ("Jonny's favorite VN is Steins;Gate") and injects only relevant memories into each prompt via semantic retrieval. Survives restarts; self-validates at startup with real vector queries.

### Multimodal Perception
A Vision Agent captures the screen, describes it via a vision model, and injects that context into the cognitive stream — so Kira knows whether you're coding, gaming, or watching anime, and reacts accordingly. A parallel audio agent loopback-transcribes desktop audio so she can hear and respond to what's playing on screen.

### Full Voice Pipeline
Real-time WebRTC VAD → Faster-Whisper transcription → LLM reasoning → Azure Neural TTS output, with self-hearing prevention, interruption handling, and rate limiting.

### Mood-Driven Live2D Expressions
Kira's emotional state (happy, sassy, moody, emotional, hyperactive) drives her Live2D avatar's facial expressions through the VTube Studio WebSocket API — code-driven, not webcam face-tracking. Expression mappings are per-model and auto-re-resolve when the avatar is swapped.

### Synced On-Screen Captions
Her spoken words appear on stream in a styled caption overlay, revealed word-by-word in sync with real Azure TTS word-boundary timing (not estimated). Rendered via a local WebSocket server into an OBS browser source. Bottom-anchored with auto-shrink for long lines, and a self-healing heartbeat auto-detects stale Azure word-boundary subscriptions over multi-hour streams and atomically rebuilds the synthesizer so captions recover without a restart.

### Autonomous VN Playthrough
A VN autopilot reads visual novels unattended — OCRs the dialogue (cloud OCR with local Tesseract fallback), speaks it in-character, interjects its own reactions and theories, advances the story, and maintains per-playthrough narrative memory — with graceful failsafes for menus and save prompts.

### Proactive Agency
Kira initiates rather than only responding. A background observer loop (Universal Boredom Protocol) monitors silence and escalates through behavioral stages, and a curiosity system has her ask chat and her co-host unprompted questions to keep conversation alive.

### Multi-Platform Streaming
Reads and responds to chat across Twitch and YouTube simultaneously, building per-chatter memory so returning viewers are recognized.

---

## Project Structure

```
Kira/
├── bot.py                        # Orchestrator — event loop, VAD, input queue, brain worker, session summaries
├── ai_core.py                    # Cortex — LLM inference, STT, TTS, prompt assembly, caption dispatch, Azure self-heal
├── memory.py                     # Hippocampus — ChromaDB interface, semantic retrieval, startup validation
├── memory_extractor.py           # Fact extraction — distills durable facts from conversation
├── playthrough_memory.py         # Per-playthrough narrative memory for VN sessions
├── vision_agent.py               # Eyes — screen capture, VLM description, context buffer
├── audio_agent.py                # Ears (desktop) — loopback audio transcription + reaction
├── loopback_transcriber.py       # WASAPI loopback capture + Whisper transcription
├── media_watch.py                # Lightweight ambient media awareness
├── vts_client.py                 # Avatar I/O — VTube Studio WebSocket client, hotkey control
├── vts_expression_controller.py  # Maps emotional state → Live2D expressions
├── caption_server.py             # Caption overlay — WebSocket broadcast server, port-collision retry, self-heal
├── caption_overlay/              # Browser-source overlay (index.html, style.css, caption.js)
├── vn_autopilot.py               # VN autopilot — OCR (cloud + local fallback), read-aloud, auto-advance
├── game_mode_controller.py       # Per-activity pacing / verbosity modes
├── control_server.py             # Web dashboard backend — FastAPI control server + /state
├── web_dashboard/                # Browser control panel — real-time controls, state monitoring
├── twitch_bot.py                 # Twitch client — chat listener, song request handler
├── youtube_bot.py                # YouTube client — live chat listener
├── twitch_tools.py               # Twitch API — poll & prediction creation, broadcaster utilities
├── chat_poster.py                # Optional outbound chat posting (rate-limited, off by default)
├── music_tools.py                # DJ — YouTube search and mpv audio streaming
├── web_search.py                 # Search — Google Custom Search API wrapper
├── persona.py                    # Emotional state — mood enum that influences response + expression
├── personality.txt               # Identity — natural language personality prompt
├── personality_file.py           # Personality loader
├── prompt_rules.py               # Formatting rules — output constraints and tool tag definitions
├── prompt_loader.py              # Prompt builder — assembles full prompt from components
├── config.py                     # Configuration — all settings loaded from .env
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment variable template
```

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 3060+ recommended) with CUDA drivers
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ tools, required for `llama-cpp-python`)
- `mpv` installed and on PATH (for music playback)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (local OCR fallback for the VN autopilot)
- [VTube Studio](https://denchisoft.com/) with the API enabled (for the Live2D avatar)
- [OBS](https://obsproject.com/) (for the caption overlay browser source, optional)

### Quick Start

```bash
git clone https://github.com/JonathanDunkleberger/Kira.git
cd Kira
pip install -r requirements.txt
```

Download a GGUF model (e.g., `Meta-Llama-3.1-8B-Instruct-Q4_K_M`) and place it in `models/`.

```bash
cp .env.example .env    # Fill in your API keys
python bot.py           # Launch the bot + web dashboard
```

The web control panel is served locally by the bot (see `CONTROL_SERVER_PORT`, default 8766) — open `http://127.0.0.1:8766/` in a browser.

To enable the on-screen captions: set `ENABLE_CAPTIONS=true`, then add `caption_overlay/index.html` as a Local File browser source in OBS.

---

## Environment

| Variable | Purpose |
|---|---|
| `LLAMA_MODEL_PATH` | Path to your local GGUF model file |
| `AZURE_SPEECH_KEY` | Azure Cognitive Services TTS |
| `AZURE_SPEECH_REGION` | Azure region (e.g., `westus3`) |
| `ANTHROPIC_API_KEY` | Claude — deep-reasoning routing (optional) |
| `OPENAI_API_KEY` | GPT — vision agent and cloud routing |
| `GOOGLE_CSE_API_KEY` | Google Custom Search for web queries |
| `GOOGLE_CSE_ID` | Custom Search Engine ID |
| `TWITCH_TOKEN` | Twitch bot OAuth token |
| `TWITCH_CLIENT_ID` | Twitch API client ID |
| `TWITCH_CHANNEL` | Twitch channel to join |
| `ENABLE_CAPTIONS` | Toggle the on-screen caption overlay |
| `ENABLE_VTS_EXPRESSIONS` | Toggle mood-driven Live2D expressions |

See `.env.example` for the full list.

### Chess Mode (one-time setup)

Kira plays real games on Lichess via a local Elo-capped Stockfish. Before any
games, do this once: create a **fresh, dedicated** Lichess account (a bot
account can never play as a human again), generate a personal access token with
the **`bot:play`** scope, then upgrade the account by calling
`POST https://lichess.org/api/bot/account/upgrade` with that token (e.g.
`curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <token>"`).
Set `LICHESS_BOT_TOKEN` and `CHESS_ENGINE_PATH` in `.env`, then arm "Chess Mode"
from the dashboard.

---

## Customization

| What | Where | How |
|---|---|---|
| Personality and backstory | `personality.txt` | Edit the natural language prompt directly |
| Emotional states | `persona.py` | Add/modify the `EmotionalState` enum |
| Emotion → expression mapping | `vts_expression_controller.py` | Map moods to your model's VTS hotkeys |
| Caption styling & anchoring | `caption_overlay/style.css` | Tune font, outline, colors, bottom-anchor, max-width/height, min-font-size via `:root` vars |
| Output formatting rules | `prompt_rules.py` | Adjust constraints (length, style, tool tags) |
| All runtime settings | `.env` | API keys, model paths, TTS engine, feature flags |

---

## Roadmap

- [ ] **GraphRAG** — Graph-based memory for richer relationship tracking between facts
- [ ] **Local Vision** — Replace API vision with a quantized VLM for full offline capability
- [ ] **Singing** — A singing-voice pipeline for karaoke / musical segments
- [ ] **Custom Commissioned Avatar** — A bespoke rigged Live2D model with rich idle physics
- [ ] **Multi-agent Reasoning** — Separate planning and execution into cooperative agent threads

---

## License

MIT
