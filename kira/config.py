# Kira AI - Config loads from .env for secrets and sensitive info
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Repo root = parent of the kira/ package dir. Used for default data/model paths
# so they resolve at the repo root regardless of where config.py lives.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model and runtime config (safe to share)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
# ↑ Primary VRAM lever. -1 = all layers on GPU (~6-7 GB for 8B Q4_K_M).
#   If streaming a VRAM-hungry game (e.g. Bond at 4K) causes OOM, drop to ~28
#   to offload the remainder to CPU. CPU layers add ~30-80ms latency per token
#   but keep the bot stable. Profile with torch.cuda.memory_reserved() first.
N_CTX = int(os.getenv("N_CTX", 16384))
N_BATCH = int(os.getenv("N_BATCH", 512))

LLM_MAX_RESPONSE_TOKENS = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", 512))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", os.path.join(_REPO_ROOT, "models", "whisper"))
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() == "true"
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
# TTS backend selector — overrides TTS_ENGINE for choosing Azure vs Fish Audio.
# "azure"  -> Azure Cognitive Speech SDK (default; full word-boundary timing for captions)
# "fish"   -> Fish Audio SDK (pay-as-you-go; falls back to Azure on any error)
# Switchable at runtime via the dashboard TTS toggle without editing code.
TTS_BACKEND = os.getenv("TTS_BACKEND", "azure")
# Fish Audio credentials (from .env — never commit real values)
FISH_API_KEY    = os.getenv("FISH_API_KEY", "")
FISH_VOICE_ID   = os.getenv("FISH_VOICE_ID", "")  # reference_id of the chosen voice model
FISH_LATENCY    = os.getenv("FISH_LATENCY", "balanced")  # "normal" | "balanced"
FISH_FORMAT     = os.getenv("FISH_FORMAT", "mp3")         # "mp3" | "wav" | "pcm" | "opus"
AI_NAME = os.getenv("AI_NAME", "Kira")
# Tuning VAD for faster response (0.4s silence triggers end-of-speech)
PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", 0.4))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
MEMORY_PATH = os.getenv("MEMORY_PATH", os.path.join(_REPO_ROOT, "memory_db"))

# Secrets and API keys (must be in .env, never commit real values)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "")
AZURE_PROSODY_PITCH = os.getenv("AZURE_PROSODY_PITCH", "")
AZURE_PROSODY_RATE = os.getenv("AZURE_PROSODY_RATE", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TWITCH_OAUTH_TOKEN = os.getenv("TWITCH_OAUTH_TOKEN", "")
TWITCH_BOT_USERNAME = os.getenv("TWITCH_BOT_USERNAME", "")
TWITCH_CHANNEL_TO_JOIN = os.getenv("TWITCH_CHANNEL_TO_JOIN", "")
VIRTUAL_AUDIO_DEVICE = os.getenv("VIRTUAL_AUDIO_DEVICE", "")

# Audio understanding config
ENABLE_AUDIO_AGENT = os.getenv("ENABLE_AUDIO_AGENT", "true").lower() == "true"
# Loopback Whisper transcriber — SEPARATE opt-in flag from the audio-MOOD agent.
# Distil-large-v3 is English-only and produces token-loop garbage on JP/VN content.
# Default OFF: enable only for English stream-watching. When OFF, the WhisperModel is never loaded.
ENABLE_LOOPBACK_TRANSCRIBER = os.getenv("ENABLE_LOOPBACK_TRANSCRIBER", "false").lower() == "true"
# Whether the loopback STT auto-starts when MEDIA audio mode is turned on.
# Default false (AAA-safe). Flip to true if you always want loopback running.
# Even with false, you can start it manually via the dashboard LOOPBACK STT toggle.
LOOPBACK_STT_DEFAULT = os.getenv("LOOPBACK_STT_DEFAULT", "false").lower() == "true"

# Smart Game Mode configuration (dashboard ACTIVATE button).
# When true (default), clicking ACTIVATE in the Game Mode panel automatically configures
# all subsystems to stream-ready state: Vision ON, Audio=MEDIA, Immersive OFF,
# Highlight Extraction ON, Loopback per LOOPBACK_STT_DEFAULT.
# Set false to restore the old manual "dumb" activation behavior.
GAME_MODE_AUTO_CONFIGURE = os.getenv("GAME_MODE_AUTO_CONFIGURE", "true").lower() == "true"

# Highlight Extraction global kill-switch.
# When true (default), the background Opus loop fires on the interval below whenever
# an activity is active (ACTIVITY_GAME, VN, or MEDIA). Set false to fully disable.
HIGHLIGHT_EXTRACTION_ENABLED = os.getenv("HIGHLIGHT_EXTRACTION_ENABLED", "true").lower() == "true"

# How often (seconds) the highlight extraction loop fires.
# Default 300s (5 min) = 12 Opus calls/hr. Old default was 90s (40 calls/hr).
# Tune via env: HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS=180
HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS = int(os.getenv("HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS", "300"))

# How often (seconds) the playthrough crash-recovery checkpoint is flushed to disk.
# Default 300s (5 min): a crash loses at most 5 minutes of reactions/chat moments.
# Lower for finer granularity; raise if the file I/O is a concern on slow storage.
CHECKPOINT_INTERVAL_SECONDS = int(os.getenv("CHECKPOINT_INTERVAL", "300"))

# Persistent stream logging (transcript.md + events.jsonl + summary.md per session).
# Written to logs/streams/YYYY-MM-DD_HH-MM_<activity>/. All writes are async and
# non-blocking. Set false only for development / offline testing.
# Note: "Test / Companion Mode" preset also forces logging off via the dashboard.
STREAM_LOGGING_ENABLED = os.getenv("STREAM_LOGGING_ENABLED", "true").lower() == "true"

AUDIO_HEARTBEAT_SECONDS = float(os.getenv("AUDIO_HEARTBEAT_SECONDS", "12.0"))
AUDIO_CLIP_SECONDS = float(os.getenv("AUDIO_CLIP_SECONDS", "8.0"))
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "gpt-4o-mini-audio-preview-2024-12-17")

# Feature Flags
ENABLE_TWITCH_CHAT = os.getenv("ENABLE_TWITCH_CHAT", "true").lower() == "true"
ENABLE_YOUTUBE_CHAT = os.getenv("ENABLE_YOUTUBE_CHAT", "true").lower() == "true"
# Allow messages from the broadcaster/channel-owner account to reach Kira's brain.
# In TwitchIO 2.x, message.echo=True whenever author==bot.nick. If the OAuth token
# belongs to the broadcaster account, ALL broadcaster messages are silently dropped.
# Set this true when bot-account == broadcaster-account (common in small setups).
# Safe to leave true in production — it only passes the broadcaster's OWN messages
# through; all other accounts are unaffected by this flag.
ALLOW_BROADCASTER_CHAT = os.getenv("ALLOW_BROADCASTER_CHAT", "false").lower() == "true"

# Chat POSTING (Kira sending messages, separate from reading). Default OFF —
# turn it on only when you've decided you want her to be able to send. Hard
# rate-limited via CHAT_POST_COOLDOWN_SEC to keep this flavor, not spam.
# Twitch posting works out of the box via the existing TWITCH_OAUTH_TOKEN.
# YouTube posting is NOT yet implemented (pytchat is read-only; sending
# requires a separate OAuth2 flow on the YouTube Data API).
ENABLE_CHAT_POSTING = os.getenv("ENABLE_CHAT_POSTING", "false").lower() == "true"
CHAT_POST_COOLDOWN_SEC = float(os.getenv("CHAT_POST_COOLDOWN_SEC", "60.0"))
CHAT_POST_MAX_LEN = int(os.getenv("CHAT_POST_MAX_LEN", "450"))  # Twitch hard cap is 500
ENABLE_YOUTUBE_POSTING = os.getenv("ENABLE_YOUTUBE_POSTING", "false").lower() == "true"

# Kira's [CHAT: ...] tool — her own typing-in-chat channel. These caps layer ON
# TOP of CHAT_POST_COOLDOWN_SEC above (the transport floor shared by all posters).
# The bar: a viewer should occasionally see "kira just typed in chat??" — rare
# enough to be an event, never spam. Tighter length than the transport cap too.
CHAT_POST_KIRA_INTERVAL_SEC = float(os.getenv("CHAT_POST_KIRA_INTERVAL_SEC", "300.0"))  # min 5 min between her posts
CHAT_POST_KIRA_MAX_PER_SESSION = int(os.getenv("CHAT_POST_KIRA_MAX_PER_SESSION", "8"))   # hard session ceiling
CHAT_POST_KIRA_MAX_LEN = int(os.getenv("CHAT_POST_KIRA_MAX_LEN", "200"))                  # chat messages stay SHORT

# Discord daily-diary webhook (Phase 1). Kira writes an in-character end-of-
# session diary entry that is SAVED for review, NOT auto-posted. Posting to the
# webhook is a deliberate manual action from the dashboard ("Post to Discord").
# DISCORD_AUTOPOST stays false until the tone is trusted over several sessions;
# flipping it true would let the diary fire to the webhook automatically.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_AUTOPOST = os.getenv("DISCORD_AUTOPOST", "false").lower() == "true"

# On-screen captions (Neuro-sama style word-by-word overlay).
# When ENABLE_CAPTIONS=true, a local WebSocket server starts on
# CAPTION_SERVER_PORT and broadcasts Kira's spoken lines (with Azure
# word-boundary timing) to caption_overlay/index.html — add that file as
# an OBS browser source. Fail-graceful: TTS never blocks on caption I/O.
# Only the Azure TTS engine supplies per-word timing; on Edge TTS the
# captions degrade to a single-frame reveal at the line start.
ENABLE_CAPTIONS = os.getenv("ENABLE_CAPTIONS", "false").lower() == "true"
CAPTION_SERVER_PORT = int(os.getenv("CAPTION_SERVER_PORT", "8765"))
CAPTION_CLEAR_DELAY_MS = int(os.getenv("CAPTION_CLEAR_DELAY_MS", "1500"))

# Web dashboard control server (FastAPI + uvicorn, 127.0.0.1 only)
# Separate from the caption WS on 8765. Provides /state, /ws (500ms push),
# /vision/thumbnail, and POST /cmd/{action} for all dashboard commands.
CONTROL_SERVER_PORT = int(os.getenv("CONTROL_SERVER_PORT", "8766"))

# Chat batching config
CHAT_BATCH_WINDOW = float(os.getenv("CHAT_BATCH_WINDOW", "5.0"))
CHAT_RESPONSE_COOLDOWN = float(os.getenv("CHAT_RESPONSE_COOLDOWN", "8.0"))

# Presence dial — probability that a bored-loop line becomes a question to chat.
# One value per presence level (Sleepy / Normal / Chatty). The dial maps onto
# EXISTING mode + carry behavior; these just make the chat-question rate tunable
# per level instead of hardcoded in the observer loop.
ASK_CHAT_P_SLEEPY = float(os.getenv("ASK_CHAT_P_SLEEPY", "0.05"))
ASK_CHAT_P_NORMAL = float(os.getenv("ASK_CHAT_P_NORMAL", "0.15"))
ASK_CHAT_P_CHATTY = float(os.getenv("ASK_CHAT_P_CHATTY", "0.25"))
# Threshold multiplier per presence level — scales the observer-loop silence
# thresholds. >1 = waits longer before speaking unprompted (sleepier);
# <1 = fills dead air sooner (chattier).
PRESENCE_THRESHOLD_MULT_SLEEPY = float(os.getenv("PRESENCE_THRESHOLD_MULT_SLEEPY", "1.8"))
PRESENCE_THRESHOLD_MULT_NORMAL = float(os.getenv("PRESENCE_THRESHOLD_MULT_NORMAL", "1.0"))
PRESENCE_THRESHOLD_MULT_CHATTY = float(os.getenv("PRESENCE_THRESHOLD_MULT_CHATTY", "0.8"))
ENABLE_CHATTER_MEMORY = os.getenv("ENABLE_CHATTER_MEMORY", "true").lower() == "true"
# Twitch native polls require affiliate status. Set true only if you're affiliate;
# otherwise [POLL:] tags get stripped silently (no failed API call).
ENABLE_TWITCH_POLLS = os.getenv("ENABLE_TWITCH_POLLS", "false").lower() == "true"

# Inference backend for local-Llama-style calls (triage, classification,
# emotion analysis, response generation fallback). Claude (Sonnet/Opus) is
# routed separately and is NOT affected by this flag.
#   "groq"  -> Groq cloud (llama-3.1-8b-instant). Saves ~6-7 GB VRAM.
#   "local" -> local llama_cpp GGUF (legacy path).
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "5.0"))
# Fallback behavior when INFERENCE_BACKEND=groq and Groq fails:
#   "false"      -> raise; caller decides what to do (no VRAM cost, no resilience).
#   "true"       -> local Llama is kept loaded at startup as a warm fallback.
#   "lazy_load"  -> local Llama is loaded into VRAM only on first Groq failure,
#                   then reused for the rest of the session. Best of both worlds.
GROQ_FALLBACK_TO_LOCAL = os.getenv("GROQ_FALLBACK_TO_LOCAL", "lazy_load").lower()

# Hybrid Brain
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Canonical Claude model-name constants — single source of truth. Reference these
# (CLAUDE_SONNET_MODEL / CLAUDE_OPUS_MODEL) instead of hard-coding model strings.
# Env var names are unchanged (CLAUDE_DEEP_MODEL / CLAUDE_CHAT_MODEL) so .env stays valid.
CLAUDE_OPUS_MODEL   = os.getenv("CLAUDE_DEEP_MODEL", "claude-opus-4-7")
CLAUDE_SONNET_MODEL = os.getenv("CLAUDE_CHAT_MODEL", "claude-sonnet-4-6")
# Haiku — cheapest tier, for high-frequency structured calls (per-turn memory
# extraction) where Sonnet-grade reasoning isn't visible in the output.
CLAUDE_HAIKU_MODEL  = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5")
# Legacy aliases — keep existing import sites working unchanged.
CLAUDE_DEEP_MODEL = CLAUDE_OPUS_MODEL
CLAUDE_CHAT_MODEL = CLAUDE_SONNET_MODEL
ENABLE_CLAUDE_BRAIN = os.getenv("ENABLE_CLAUDE_BRAIN", "true").lower() == "true"
ENABLE_CLAUDE_CHAT = os.getenv("ENABLE_CLAUDE_CHAT", "true").lower() == "true"
ENABLE_PROMPT_CACHING = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
ENABLE_CLAUDE_STREAMING = os.getenv("ENABLE_CLAUDE_STREAMING", "true").lower() == "true"

# AudD audio fingerprinting (paid API; only fires on explicit user song-ID intent).
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "")

# Storytime / Puppet Show (pre-generated shadow-puppet shows).
# Image generation uses Google Gemini 2.5 Flash Image ("Nano-Banana") for its
# style/character consistency. The key below is a FRESH Google AI Studio key,
# SEPARATE from GOOGLE_API_KEY (which is Custom Search). Do not conflate them.
GEMINI_IMAGE_API_KEY = os.getenv("GEMINI_IMAGE_API_KEY", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
# Which image provider the Storytime pipeline uses. Swappable: the client is a
# provider-agnostic module, so this can change without touching the orchestrator.
STORYTIME_IMAGE_PROVIDER = os.getenv("STORYTIME_IMAGE_PROVIDER", "gemini").lower()

# Cutscene-aware observer suppression — only active during ACTIVITY_GAME mode.
# When true, the observer loop and per-turn triage check for cutscene cues
# (vision scene summary + audio mood keywords) and suppress interjections while
# a cutscene is likely playing. Has zero effect outside ACTIVITY_GAME mode.
CUTSCENE_AWARE = os.getenv("CUTSCENE_AWARE", "true").lower() == "true"

# VTube Studio integration — drives Live2D facial expressions from Kira's emotional state.
# Default ON. Token is auto-created on first approved handshake and persisted to disk.
ENABLE_VTS_EXPRESSIONS = os.getenv("ENABLE_VTS_EXPRESSIONS", "true").lower() == "true"
VTS_WS_URL = os.getenv("VTS_WS_URL", "ws://localhost:8001")
VTS_PLUGIN_NAME = os.getenv("VTS_PLUGIN_NAME", "Kira AI")
VTS_PLUGIN_DEVELOPER = os.getenv("VTS_PLUGIN_DEVELOPER", "JonnyD")
VTS_TOKEN_PATH = os.getenv("VTS_TOKEN_PATH", os.path.join(_REPO_ROOT, ".vts_token"))

# Chess Mode (Phase 1) — Kira plays Lichess against Stockfish, on stream.
# Disabled unless armed from the dashboard. Requires a SEPARATE Lichess BOT
# account + a bot:play token (see README for the one-time upgrade step). The
# engine is a local Stockfish binary, Elo-capped so Kira plays (and blunders)
# like a human club player. CPU only — no GPU contention with vision/STT.
LICHESS_BOT_TOKEN = os.getenv("LICHESS_BOT_TOKEN", "")
CHESS_ENGINE_PATH = os.getenv("CHESS_ENGINE_PATH", "stockfish.exe")
CHESS_KIRA_ELO    = int(os.getenv("CHESS_KIRA_ELO", "1400"))
CHESS_MOVETIME_MS = int(os.getenv("CHESS_MOVETIME_MS", "150"))

# Auto-clip pipeline (offline post-session) — scripts/cut_clips.py turns Kira's
# clip-candidate artifacts into pre-cut, pre-titled video files. Runs AFTER a
# stream, never during. OBS_RECORDINGS_DIR is where OBS writes local recordings;
# cut clips land in OBS_RECORDINGS_DIR/clips/YYYY-MM-DD/.
OBS_RECORDINGS_DIR = os.getenv("OBS_RECORDINGS_DIR", "")
# Window around the detected moment: lead-in before (setup) + payoff after.
# Widened 2026-06-13 after reel review: pre 5→6.5s, post 3→3.5s (cuts ran a
# touch early on both ends). Override via env or --pre/--post CLI flags.
# Min clip length is 12s (enforced in clip_cutter.py regardless).
CLIP_PRE_SECONDS   = float(os.getenv("CLIP_PRE_SECONDS", "6.5"))
CLIP_POST_SECONDS  = float(os.getenv("CLIP_POST_SECONDS", "3.5"))
# Minimum session length (minutes) below which the reel is skipped.
# Also requires at least 3 aligned candidates. Override via env.
REEL_MIN_MINUTES   = int(os.getenv("REEL_MIN_MINUTES", "20"))
# Recording container extensions to scan, comma-separated.
CLIP_VIDEO_EXTS    = os.getenv("CLIP_VIDEO_EXTS", ".mkv,.mp4,.flv,.mov,.ts")

# ── YouTube auto-connect ──────────────────────────────────────────────────────
# Set YOUTUBE_CHANNEL_ID in .env to enable polling on boot.  Kira will poll
# the YouTube Data API v3 search.list endpoint every YT_AUTO_CONNECT_POLL_S
# seconds for up to YT_AUTO_CONNECT_TIMEOUT_S total (default: every 60s /
# 15 minutes), then give up and wait for a manual connect via the dashboard.
YOUTUBE_CHANNEL_ID        = os.getenv("YOUTUBE_CHANNEL_ID", "")
YT_AUTO_CONNECT_TIMEOUT_S = int(os.getenv("YT_AUTO_CONNECT_TIMEOUT_S", "900"))   # 15 min
YT_AUTO_CONNECT_POLL_S    = int(os.getenv("YT_AUTO_CONNECT_POLL_S", "60"))

# ── Chat queue / instrumentation ─────────────────────────────────────────────
# ACK_THRESHOLD_S: if the oldest pending chat message has been waiting longer
# than this (seconds), a brief acknowledgment directive is injected into the
# next response prompt so the chatter's name gets a quick mention while Kira
# finishes the current turn.
ACK_THRESHOLD_S = float(os.getenv("ACK_THRESHOLD_S", "20.0"))

# Talk-budget governor — off by default until instrumentation data is collected.
# When enabled, per-chatter response counts are used to bias batch ordering so
# chatters with fewer responses get answered first.
CHAT_BUDGET_ENABLED       = os.getenv("CHAT_BUDGET_ENABLED", "false").lower() == "true"
CHAT_BUDGET_RESPOND_ALL_N = int(os.getenv("CHAT_BUDGET_RESPOND_ALL_N", "5"))

# ── Phrase throttle ───────────────────────────────────────────────────────────
# Prevents distinctive constructions from becoming templates by tracking per-
# session n-grams and injecting a soft do-not-reuse constraint into every LLM
# prompt when a phrase hits PHRASE_THROTTLE_THRESHOLD uses.
# PHRASE_THROTTLE_WATCHLIST: comma-separated list of specific phrases to monitor
# even if they wouldn't surface from the n-gram statistics (e.g. short idioms).
PHRASE_THROTTLE_ENABLED   = os.getenv("PHRASE_THROTTLE_ENABLED", "true").lower() == "true"
PHRASE_THROTTLE_THRESHOLD = int(os.getenv("PHRASE_THROTTLE_THRESHOLD", "2"))
PHRASE_THROTTLE_WATCHLIST = [
    p.strip() for p in os.getenv(
        "PHRASE_THROTTLE_WATCHLIST",
        "three words and a vibe,and a vibe,I respect it,I respect the commitment,I'll wait",
    ).split(",") if p.strip()
]

# Deterministic cooldown (seconds) for the word-count narration tic — the
# "three words and a vibe" / "one word and a vibe" family. Once Kira fires the
# construction, _kira_voice_guardrails hard-bans it (and any narration of the
# user's brevity) for this many seconds. Separate from the n-gram throttle so
# it catches every variant, not just exact repeats.
FRAGMENT_QUIP_COOLDOWN_S = int(os.getenv("FRAGMENT_QUIP_COOLDOWN_S", "240"))

# ── Chess owner / data dir ────────────────────────────────────────────────────
# Challenges from CHESS_OWNER_LICHESS_ID (case-insensitive) are Jonny's own
# practice games and do NOT trigger the spectate embed on stream.
CHESS_OWNER_LICHESS_ID = os.getenv("CHESS_OWNER_LICHESS_ID", "Militele3")

# Local state files (gitignored) — lifetime chess stats, etc.
DATA_DIR = os.path.join(_REPO_ROOT, "data")
