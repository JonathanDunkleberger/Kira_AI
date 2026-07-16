# Kira — Quick Start (≈15 minutes)

A minimal path to a talking Kira. This skips every optional feature (Twitch,
YouTube, chess, captions, VN autopilot, clipping) — just the core voice loop
plus the local web dashboard. Add the rest later from the full
[README](README.md) and [.env.example](.env.example).

---

## 1. System prerequisites

| Tool | Why | Required? |
|---|---|---|
| **Python 3.10+** | Runtime | Yes |
| **mpv** (on PATH) | Music playback | Yes (import-time) |
| **Tesseract OCR** | Local OCR fallback for the VN autopilot | Optional — only for VN mode |
| **NVIDIA GPU + CUDA** | Faster-Whisper mic transcription runs on CUDA float16 | Strongly recommended — STT init expects a CUDA device |
| **VB-Audio Virtual Cable** | Desktop-audio capture for the audio agent | Only if `ENABLE_AUDIO_AGENT=true` (the default) |

> **GPU note:** perception (Whisper STT) is built for CUDA float16. There is no
> documented CPU STT path, so a CUDA-capable NVIDIA GPU is effectively required
> for the voice loop. Everything else (Groq, Claude, Azure TTS) is cloud and
> needs no GPU.

Install **mpv**: https://mpv.io/installation/ — make sure `mpv --version` works
in a fresh terminal.

---

## 2. Clone + install

```powershell
git clone https://github.com/JonathanDunkleberger/Kira.git
cd Kira
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default **groq** inference backend needs **no compiler and no CUDA toolkit**
to install — `requirements.txt` installs cleanly on its own.

---

## 3. Three first-run gotchas (read these)

1. **`llama-cpp-python` is NOT installed — and you don't need it.**
   It only powers the *optional* on-device Llama fallback
   (`INFERENCE_BACKEND=local`), and it compiles a native C++/CUDA extension that
   trips up most fresh clones. The default groq backend skips it entirely. If you
   ever want the local fallback: `pip install -r requirements-local.txt`.

2. **Faster-Whisper downloads ~1.5 GB on first boot — and there are TWO models.**
   The first time you run Kira, the mic Whisper weights (`large-v3`) are fetched
   to `models/whisper/`. Because loopback transcription is now **on by default**
   (`ENABLE_LOOPBACK_TRANSCRIBER=true`), a *second* Whisper (`distil-large-v3`,
   another ~1.5 GB / ~1.5 GB VRAM) also downloads + loads for desktop-audio ASR.
   This looks like a long hang — it isn't. Let it finish once; subsequent boots
   are instant. To run a single Whisper, set `ENABLE_LOOPBACK_TRANSCRIBER=false`.

3. **Stockfish is only for chess.**
   `CHESS_ENGINE_PATH=stockfish.exe` is irrelevant unless you arm Chess Mode.
   Ignore it for the quick start.

---

## 4. Configure the five required keys

```powershell
Copy-Item .env.example .env
```

Open `.env` and fill in **only** the keys in the `⭐ MINIMUM REQUIRED TO BOOT ⭐`
block at the very top:

| Key | What it's for | Free / Paid | Get it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Kira's main brain (Claude) | Paid | https://console.anthropic.com/ |
| `GROQ_API_KEY` | Fast triage inference | Free tier | https://console.groq.com/keys |
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | Default TTS voice | Free tier | https://portal.azure.com/ → "Speech service" |
| `GOOGLE_API_KEY` | Vision + audio agent (both on by default) | Paid | https://aistudio.google.com/apikey/api-keys |

> **Heads-up — perception is on out of the box.** A fresh clone boots with
> vision (`ENABLE_VISION=true`) and audio-mood (`AUDIO_MOOD_ALWAYS_ON=true`)
> awake, so OpenAI bills on idle, and a second local Whisper
> (`ENABLE_LOOPBACK_TRANSCRIBER=true`) loads on the GPU. To boot cheap/quiet,
> set `ENABLE_VISION=false`, `ENABLE_AUDIO_AGENT=false`, and
> `ENABLE_LOOPBACK_TRANSCRIBER=false` in `.env` and leave `GOOGLE_API_KEY` blank.

Everything else in `.env` already has a working default — leave it untouched for
the quick start.

---

## 5. (Optional) Persona

Kira falls back to the shipped template persona automatically. To make her your
own, copy the example and edit it:

```powershell
Copy-Item persona\example\personality.example.txt persona\private\personality.txt
```

A `[persona] Loaded from:` line on startup confirms which file was used.

---

## 6. Run

```powershell
python run.py
```

### Did it work?

- The console prints a `[persona] Loaded from:` line and backend init logs.
- Faster-Whisper loads (after the one-time download).
- Open the dashboard: **http://127.0.0.1:8766/**
- Speak into your mic — Kira transcribes, thinks, and replies out loud.

If the dashboard loads and you hear a reply, you're up. From here, add features
(Twitch, YouTube, captions, chess) by filling in the matching section of `.env`
— see the full [README](README.md).

