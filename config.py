# Kira AI - Config loads from .env for secrets and sensitive info
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Model and runtime config (safe to share)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
N_CTX = int(os.getenv("N_CTX", 8192))
N_BATCH = int(os.getenv("N_BATCH", 512))

LLM_MAX_RESPONSE_TOKENS = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", 512))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "G:\\JonnyD\\NeuroAI_Bot\\models\\whisper")
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() == "true"
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
AI_NAME = os.getenv("AI_NAME", "Kira")
# Tuning VAD for faster response (0.4s silence triggers end-of-speech)
PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", 0.4))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
MEMORY_PATH = os.getenv("MEMORY_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_db"))

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
AUDIO_HEARTBEAT_SECONDS = float(os.getenv("AUDIO_HEARTBEAT_SECONDS", "12.0"))
AUDIO_CLIP_SECONDS = float(os.getenv("AUDIO_CLIP_SECONDS", "8.0"))
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "gpt-4o-mini-audio-preview-2024-12-17")

# Feature Flags
ENABLE_TWITCH_CHAT = os.getenv("ENABLE_TWITCH_CHAT", "true").lower() == "true"
ENABLE_YOUTUBE_CHAT = os.getenv("ENABLE_YOUTUBE_CHAT", "true").lower() == "true"

# Chat batching config
CHAT_BATCH_WINDOW = float(os.getenv("CHAT_BATCH_WINDOW", "5.0"))
CHAT_RESPONSE_COOLDOWN = float(os.getenv("CHAT_RESPONSE_COOLDOWN", "8.0"))
ENABLE_CHATTER_MEMORY = os.getenv("ENABLE_CHATTER_MEMORY", "true").lower() == "true"

# Hybrid Brain
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_DEEP_MODEL = os.getenv("CLAUDE_DEEP_MODEL", "claude-opus-4-7")
ENABLE_CLAUDE_BRAIN = os.getenv("ENABLE_CLAUDE_BRAIN", "true").lower() == "true"
CLAUDE_CHAT_MODEL = os.getenv("CLAUDE_CHAT_MODEL", "claude-sonnet-4-6")
ENABLE_CLAUDE_CHAT = os.getenv("ENABLE_CLAUDE_CHAT", "true").lower() == "true"
ENABLE_PROMPT_CACHING = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
ENABLE_CLAUDE_STREAMING = os.getenv("ENABLE_CLAUDE_STREAMING", "true").lower() == "true"

# AudD audio fingerprinting (paid API; only fires on explicit user song-ID intent).
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "")
