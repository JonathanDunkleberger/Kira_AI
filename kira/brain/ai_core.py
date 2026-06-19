# ai_core.py - Core logic for the AI, including STT, LLM, and TTS.

import asyncio
import io
import os
import re
import gc
import time
import pygame
import torch
import numpy as np
import threading
from faster_whisper import WhisperModel
from llama_cpp import Llama

from kira.config import (
    LLM_MODEL_PATH, N_CTX, N_BATCH, N_GPU_LAYERS, WHISPER_MODEL_SIZE, WHISPER_CACHE_DIR, TTS_ENGINE,
    LLM_MAX_RESPONSE_TOKENS,
    ELEVENLABS_API_KEY, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    AI_NAME,
    ANTHROPIC_API_KEY, CLAUDE_DEEP_MODEL, ENABLE_CLAUDE_BRAIN,
    CLAUDE_CHAT_MODEL, CLAUDE_HAIKU_MODEL, ENABLE_CLAUDE_CHAT, ENABLE_PROMPT_CACHING, ENABLE_CLAUDE_STREAMING,
    INFERENCE_BACKEND, GROQ_FALLBACK_TO_LOCAL, GROQ_MODEL,
    TTS_BACKEND, FISH_API_KEY, FISH_VOICE_ID, FISH_LATENCY, FISH_FORMAT,
)
from kira.brain.inference_router import route_chat_completion, get_groq_client
from kira.brain.cost_tracker import cost_tracker as _cost_tracker
from kira.brain.groq_client import GroqInferenceError
from kira.persona.persona import EmotionalState
from kira.persona.personality_file import KIRA_PERSONALITY

# Maps each emotional state to a concise behavioural directive injected into the prompt.
# Must stay in sync with EmotionalState in persona.py.
EMOTION_DESCRIPTORS = {
    EmotionalState.HAPPY:       "Default mode. Be cheerful, curious, and let your sassy wit flow naturally.",
    EmotionalState.MOODY:       "You are withdrawn and angsty. Keep answers shorter and more sarcastic than usual.",
    EmotionalState.SASSY:       "Your wit is razor-sharp right now. Tease more, soften less. Make it sting but stay fun.",
    EmotionalState.EMOTIONAL:   "You feel open and earnest. It is okay to say something genuinely sweet or heartfelt.",
    EmotionalState.HYPERACTIVE: "You are buzzing with excitement. Ramble a little. Everything feels more interesting than normal.",
}
from kira.persona.prompt_loader import load_personality_txt
from kira.persona.prompt_rules import TOOL_AND_FORMAT_RULES, IMPROV_DISPOSITION
from kira.persona.streamer_overlay import STREAMER_OVERLAY

# Graceful SDK imports
try: from edge_tts import Communicate
except ImportError: Communicate = None
try: from elevenlabs.client import AsyncElevenLabs
except ImportError: AsyncElevenLabs = None
try: import azure.cognitiveservices.speech as speechsdk
except ImportError: speechsdk = None
try:
    from fish_audio_sdk.apis import Session as FishSession
    from fish_audio_sdk.schemas import TTSRequest as FishTTSRequest
    FISH_SDK_AVAILABLE = True
except ImportError:
    FishSession = None
    FishTTSRequest = None
    FISH_SDK_AVAILABLE = False
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


# ── Content-safety backstop denylist (hard-stop before TTS) ───────────────────
# Seed = the most unambiguous slurs, word-boundary matched so ordinary game/strategy
# words ("kill the boss") are NEVER caught — the context-dependent categories
# (atrocity/sexual/self-harm/violence-as-punchline) are handled by the persona's HARD
# CONTENT BOUNDARY, not here. This narrow net just hard-stops the most clipboard-able,
# irreversible failures regardless of bit/roleplay framing. Extend via KIRA_CONTENT_DENYLIST.
_CONTENT_DENYLIST_SEED = frozenset({
    "nigger", "nigga", "faggot", "tranny", "kike", "chink", "spic", "wetback", "retard",
})


class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        # Content-safety backstop: seed + runtime KIRA_CONTENT_DENYLIST extension.
        _extra_deny = os.getenv("KIRA_CONTENT_DENYLIST", "")
        self._content_denylist = set(_CONTENT_DENYLIST_SEED) | {
            t.strip().lower() for t in _extra_deny.split(",") if t.strip()
        }
        self._content_denylist_re = (
            re.compile(r"\b(?:" + "|".join(re.escape(t) for t in self._content_denylist) + r")\b", re.I)
            if self._content_denylist else None
        )
        self.is_initialized = False
        self.is_speaking = False # Added flag for self-hearing prevention
        # Backref set by the bot so ai_core can append the streamer overlay
        # without baking mode into the cached system prompt. Returns 'companion' or 'streamer'.
        self._mode_provider = None
        self.last_speech_finish_time = time.time() # Added for silence tracking
        # Wall-clock instant the most recent utterance's audio playback actually
        # STARTED (set in _speak_single right before pygame playback). Used by the
        # [LAG] sense->speak metric so "speak" means when she started talking, not
        # when playback finished.
        self.last_playback_start_time = 0.0
        self.last_tts_request_time = 0 # Added for TTS rate limiting
        self.llm = None
        self.whisper = None
        self.eleven_client = None
        self.azure_synthesizer = None
        # Per-synthesis-call word-boundary buffer. The Azure SDK fires the
        # synthesis_word_boundary event from a worker thread; the handler
        # appends to this list. We reset it BEFORE each speak call and read
        # it AFTER the call completes, then ship the result to the caption
        # overlay just before pygame playback starts.
        # Each entry: {"word": str, "offset_ms": int}
        self._azure_word_buffer: list[dict] = []
        self._azure_word_handler_token = None
        # Serializes Azure synth calls so the shared _azure_word_buffer can't
        # be reset by a second concurrent speak before the first call has
        # snapshotted its words. Without this lock, an autopilot "interjected
        # reaction" firing while a main read is still synthesizing can wipe
        # the buffer and silently kill the caption frame for whichever line
        # snapshots second. This is a likely root cause of captions "randomly
        # going silent" mid-stream.
        self._azure_tts_lock = asyncio.Lock()
        # Cached event-loop reference for caption_server thread-safe scheduling.
        # Set lazily on first speak call.
        self._event_loop_for_captions = None
        # --- Azure session health tracking ---
        # Detects dead word-boundary subscriptions over long streams. The
        # Azure SDK's underlying WebSocket can go stale after hours of
        # activity; speak_ssml still returns audio bytes (SDK auto-reconnects
        # for audio), but the synthesis_word_boundary callback on the
        # original native object stops firing — every caption frame then
        # silently drops with "No word-boundary events captured". We count
        # consecutive Azure speak completions that produced ZERO word events
        # for non-trivial text, and once we hit a threshold we rebuild the
        # synthesizer (in the background, under _azure_tts_lock) and
        # re-subscribe the callback so captions recover without a restart.
        self._azure_consecutive_empty_word_lines = 0
        self._azure_last_word_event_time = time.time()
        self._azure_total_word_events = 0
        self._last_azure_speak_time = 0.0
        self._azure_speech_config = None  # cached for fast reinit
        self._azure_reinit_lock = asyncio.Lock()
        self._azure_reinit_in_progress = False
        self._azure_session_generation = 0  # bumps each successful reinit
        # Threshold: this many consecutive non-trivial speaks with zero word
        # events triggers a synthesizer rebuild.
        self._AZURE_EMPTY_LINE_THRESHOLD = 3

        # --- Latency instrumentation slots (read by bot.process_and_respond) ---
        # Populated per streaming-speak by speak_streaming/_speak_single so the
        # voice path can assemble one consolidated [LATENCY] line. Measure-only;
        # never alters behaviour.
        self._lat_stream_t0 = 0.0          # speak_streaming start timestamp
        self._lat_ttft_ms = -1             # stream-start -> first token chunk
        self._lat_tts_first_chunk_ms = -1  # stream-start -> first sentence dispatched to TTS
        self._lat_audio_out_ms = -1        # stream-start -> first pygame playback start

        # --- TTS priority gate -------------------------------------------------
        # All synth+playback funnels through a single priority gate so concurrent
        # speakers never stomp each other's pygame channel or clobber the shared
        # latency slots above. Lanes (lower number = higher priority):
        #   P0  voice replies to Jonny  — jump the queue
        #   P1  MW reactions / interjections / announcements
        #   P2  chat-batch responses
        # A line already mid-playback finishes (we never preempt active audio),
        # but every QUEUED non-P0 item yields the moment a P0 line is waiting.
        # This is what fixes the 19.7s chat-batch-blocks-voice stall and the
        # negative audio_out (concurrent _speak_single clobbering _lat_* slots).
        self._speech_gate_held = False
        self._speech_waiters: list = []    # entries: [priority, seq, future]
        self._speech_seq = 0
        # True for the lifetime of a P0 voice turn (speak_streaming) — from the
        # instant it starts consuming the LLM stream until playback finishes.
        # The Bored-interjection loop reads this (has_pending_voice_turn) to yield
        # the speech gate at the next SENTENCE boundary so a ready voice response
        # isn't stuck behind a long autonomous interjection.
        self._voice_turn_active = False
        # Set TRUE the instant triage decides RESPOND for a user voice turn —
        # BEFORE that turn blocks on _active_turn_lock (which an in-flight
        # interjection may be holding). Bridges the window where the voice turn
        # is waiting on the lock and hasn't yet reached speak_streaming, so it
        # sets neither _voice_turn_active nor a P0 waiter. The interjection's
        # between-sentence check reads this and aborts at the next sentence
        # boundary, releasing the lock for the waiting voice turn. The bot sets
        # and clears it (cleared in a finally on ALL exit paths).
        self._voice_response_pending = False

        self.inference_lock = threading.Lock() # Added lock for inference safety

        # Per-session token counters — accumulated from response.usage on every
        # non-streaming API call. Written to events.jsonl as "session_tokens" at
        # session end. Streaming path is excluded (see claude_chat_inference_stream).
        self._session_usage = {
            "sonnet_in": 0, "sonnet_out": 0, "sonnet_cache_read": 0,
            "opus_in":   0, "opus_out":   0,
            "haiku_in":  0, "haiku_out":  0,
            "groq_in":   0, "groq_out":   0,
        }

        # Fish Audio TTS — single session object initialized once at startup.
        # tts_backend and fish_voice_id are runtime-mutable (dashboard controls).
        self.tts_backend: str = TTS_BACKEND  # "azure" | "fish"
        self.fish_voice_id: str = FISH_VOICE_ID  # overridable live from the dashboard
        self.fish_session = None
        if FISH_SDK_AVAILABLE and FISH_API_KEY:
            try:
                self.fish_session = FishSession(FISH_API_KEY)
                print(f"   [TTS] Fish Audio session initialized (voice={self.fish_voice_id or 'default'}).")
            except Exception as e:
                print(f"   [TTS] Fish Audio session init failed: {e}. Fish TTS disabled.")
        elif TTS_BACKEND == "fish":
            print("   [TTS] WARNING: TTS_BACKEND=fish but fish-audio-sdk is not installed or FISH_API_KEY is empty. Falling back to Azure.")

        # Hybrid brain: Claude Opus for deep moments
        self.anthropic_client = None
        if ENABLE_CLAUDE_BRAIN and ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
                print(f"   [Brain] Hybrid mode ON \u2014 Claude {CLAUDE_DEEP_MODEL} available for deep moments.")
            except Exception as e:
                print(f"   [Brain] Claude init failed: {e}. Falling back to local-only.")
                self.anthropic_client = None
        else:
            print(f"   [Brain] Local-only mode (Claude disabled or API key missing).")

        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()

    async def initialize(self):
        """Initializes AI components sequentially to prevent resource conflicts."""
        print("-> Initializing AI Core components...")
        
        # Initialize Audio Mixer permanently
        if pygame.mixer.get_init(): pygame.mixer.quit()
        
        # FORCE Default Desktop Audio (User Request)
        print("   Forcing Audio Output to Default Windows Device (devicename=None)...")
        try:
            pygame.mixer.init(devicename=None)
        except Exception as e:
            print(f"   Warning: Default init failed, trying auto: {e}")
            pygame.mixer.init()

        try:
            await asyncio.to_thread(self._init_llm)
            await asyncio.to_thread(self._init_whisper)
            await self._init_tts()

            self.is_initialized = True
            print("   AI Core initialized successfully!")
        except Exception as e:
            print(f"FATAL: AI Core failed to initialize: {e}")
            self.is_initialized = False
            raise

    def reload_personality(self):
        """Hot-reloads personality from text file without restarting."""
        print("-> Reloading Personality...")
        try:
            new_personality = load_personality_txt("personality.txt")
            self.system_prompt = (new_personality.strip() + "\n\n"
                                  + IMPROV_DISPOSITION.strip() + "\n\n"
                                  + TOOL_AND_FORMAT_RULES.strip())
            print("   ✅ Personality reloaded successfully.")
            print("----- NEW SYSTEM PROMPT SNAPSHOT -----")
            print(self.system_prompt[:200])
            print("--------------------------------------")
        except Exception as e:
            print(f"   ❌ Failed to reload personality: {e}")

    def speakers_active(self) -> bool:
        """True only when TTS audio is ACTUALLY playing out the speakers right now.

        `is_speaking` is set at the START of speak_streaming — before the first
        TTS chunk synthesizes (ttft ~1-2s) and between sentence chunks — so it is
        True during windows where the speakers are provably silent. The VAD's
        ttft side-buffer uses this finer signal to safely capture the user's
        opening words during those silent windows without any risk of
        transcribing Kira's own voice (which only plays via pygame).
        """
        try:
            return bool(pygame.mixer.get_busy())
        except Exception:
            # If the mixer can't be queried, fail SAFE: assume speakers are live
            # so we never accidentally capture her own TTS.
            return True

    async def test_audio_output(self):
        """Plays a test tone to verify audio output."""
        print("-> Testing Audio Output...")
        try:
            # Generate a 440Hz sine wave for 0.5 seconds
            sample_rate = 44100
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(440 * t * 2 * np.pi).astype(np.float32)
            # Convert to 16-bit int (stereo)
            tone_int = (tone * 32767).astype(np.int16)
            stereo_tone = np.column_stack((tone_int, tone_int))
            
            sound = pygame.mixer.Sound(buffer=stereo_tone)
            channel = sound.play()
            
            # Wait for playback to finish
            while channel.get_busy():
                await asyncio.sleep(0.1)
            print("   Audio test passed (Beep played).")
        except Exception as e:
            print(f"   Audio test FAILED: {e}")

    def _init_llm(self, force: bool = False):
        # Always build the system prompt (every backend needs it). The improv disposition
        # is injected between persona and tool rules at THIS single chokepoint (+ the
        # hot-reload twin) so it's WHO SHE IS in every utterance — voice, interjection,
        # Director, deep moments, local fallback — never a mode, and cached as Block A.
        self.system_prompt = (KIRA_PERSONALITY.strip() + "\n\n"
                              + IMPROV_DISPOSITION.strip() + "\n\n"
                              + TOOL_AND_FORMAT_RULES.strip())

        # Routing decision: skip loading the GGUF into VRAM unless we actually
        # need it locally. `force=True` is used by the lazy-load fallback path
        # in inference_router after a Groq failure.
        backend = (INFERENCE_BACKEND or "groq").lower()
        fallback_mode = (GROQ_FALLBACK_TO_LOCAL or "false").lower()
        if not force and backend == "groq" and fallback_mode != "true":
            print("----- SYSTEM PROMPT (FIRST 400 CHARS) -----")
            print(self.system_prompt[:400])
            print("------------------------------------------")
            print(f"   [INFERENCE] Backend: groq ({GROQ_MODEL}) — local Llama NOT loaded (saves ~6-7 GB VRAM).")
            if fallback_mode == "lazy_load":
                print("   [INFERENCE] Fallback: lazy_load — local Llama will load on first Groq failure.")
            else:
                print("   [INFERENCE] Fallback: disabled — Groq failures will raise.")
            # Eager-probe the Groq client so a bad key fails loudly at startup
            # rather than on the first triage call.
            try:
                get_groq_client()
                print("   [INFERENCE] Groq client initialized.")
            except GroqInferenceError as e:
                print(f"   [INFERENCE] WARNING: Groq client init failed at startup: {e}")
            self.llm = None
            return

        print(f"-> Loading LLM model... (GPU Layers: {N_GPU_LAYERS})")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")

        # Optimized for RTX 5080: n_threads=8, Flash Attention enabled, Force GPU Layers=-1
        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_ubatch=N_BATCH,      # Match n_batch
            n_ctx_keep=200,        # Ensure her core identity/system prompt never gets deleted
            context_erase=0.5,     # When she hits max context, she'll forget the oldest 50%
            flash_attn=True,
            offload_kqv=True,      # Keep attention math on the chip
            use_mmap=True,
            use_mlock=True,        # Forces Windows to keep this in memory
            n_threads=8,
            verbose=False
        )

        print("----- SYSTEM PROMPT (FIRST 400 CHARS) -----")
        print(self.system_prompt[:400])
        print("------------------------------------------")
        print(f"   LLM loaded. Path: {LLM_MODEL_PATH}")
        try:
            from kira.gpu_telemetry import log_vram
            log_vram(f"after local Llama load (n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX})")
        except Exception:
            pass
        if backend == "groq":
            print("   [INFERENCE] Backend: groq (warm local fallback loaded).")
        else:
            print("   [INFERENCE] Backend: local llama_cpp.")

    def _init_whisper(self):
        print("-> Loading Faster-Whisper STT model...")
        # Baseline BEFORE any Kira model loads — the card= delta from here to each
        # 'after ... load' line attributes per-component VRAM, which works even where
        # NVML's per-process number doesn't (Windows WDDM reports kira_process=0).
        try:
            from kira.gpu_telemetry import log_vram
            log_vram("baseline — before Kira's models load")
        except Exception:
            pass
        if torch.cuda.is_available():
            print(f"   CUDA Detected: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("   WARNING: CUDA NOT DETECTED! Whisper will run on CPU (Slow).")
            device = "cpu"

        import os as _os
        _os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
        print(f"   Whisper Config: Model={WHISPER_MODEL_SIZE} | Device={device} | ComputeType=float16 | Cache={WHISPER_CACHE_DIR}")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type="float16", download_root=WHISPER_CACHE_DIR)
        print("   Faster-Whisper STT model loaded.")
        try:
            from kira.gpu_telemetry import log_vram
            log_vram(f"after mic Whisper load ({WHISPER_MODEL_SIZE}, {device})")
        except Exception:
            pass

    async def _init_tts(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == "elevenlabs":
            if not AsyncElevenLabs: raise ImportError("Run 'pip install elevenlabs'")
            self.eleven_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        elif TTS_ENGINE == "azure":
            if not speechsdk: raise ImportError("Run 'pip install azure-cognitiveservices-speech'")
            
            # Validate Azure Config
            sanitized_key = f"{AZURE_SPEECH_KEY[:4]}****" if AZURE_SPEECH_KEY else "None"
            print(f"   Azure Config: Region=[{AZURE_SPEECH_REGION}], Key=[{sanitized_key}]")
            if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
                print("   WARNING: Azure Key or Region is missing! Check .env")

            self._azure_speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY.strip(), region=AZURE_SPEECH_REGION.strip())
            self.azure_synthesizer = self._build_azure_synthesizer()
            if not self.azure_synthesizer:
                raise RuntimeError("Azure synthesizer initial build failed")
        elif TTS_ENGINE == "edge":
            if not Communicate: raise ImportError("Run 'pip install edge-tts'")
        else:
            raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")
        print(f"   {TTS_ENGINE.capitalize()} TTS ready.")

    # ----- Azure synthesizer build / re-init -----
    def _build_azure_synthesizer(self):
        """Construct a fresh Azure SpeechSynthesizer and subscribe the
        word-boundary callback. Used by initial init and by the auto-reinit
        path when the SDK's word_boundary subscription has gone stale.
        Returns the new synthesizer (or None on failure).
        """
        if not speechsdk or not self._azure_speech_config:
            return None
        try:
            # Route Azure output to a null stream so Pygame handles playback exclusively.
            class NullCallback(speechsdk.audio.PushAudioOutputStreamCallback):
                def write(self, data: memoryview) -> int: return data.nbytes
                def close(self) -> None: pass

            stream = speechsdk.audio.PushAudioOutputStream(NullCallback())
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            synth = speechsdk.SpeechSynthesizer(
                speech_config=self._azure_speech_config, audio_config=audio_config
            )

            # Subscribe to word-boundary events so the caption overlay can
            # reveal each word in sync with playback. Azure provides
            # audio_offset in 100-ns ticks (ticks // 10_000 == milliseconds
            # from synthesis start). Captures into self._azure_word_buffer
            # which is reset before each speak call.
            def _on_word_boundary(evt):
                try:
                    word = (getattr(evt, "text", "") or "").strip()
                    if not word:
                        return
                    offset_ms = int(getattr(evt, "audio_offset", 0)) // 10_000
                    self._azure_word_buffer.append({"word": word, "offset_ms": offset_ms})
                    self._azure_last_word_event_time = time.time()
                    self._azure_total_word_events += 1
                except Exception:
                    # Never let a callback crash the SDK worker thread.
                    pass
            try:
                synth.synthesis_word_boundary.connect(_on_word_boundary)
            except Exception as e:
                print(f"   [TTS] Could not subscribe to Azure word_boundary events: {e}")
            return synth
        except Exception as e:
            print(f"   [TTS] Failed to build Azure synthesizer: {e}")
            return None

    def _is_meaningful_speak_text(self, text: str) -> bool:
        """True if the text is substantial enough that Azure SHOULD produce
        word-boundary events. Used to filter out tiny utterances ("Hi.",
        "Oh.") that legitimately produce zero events and shouldn't trigger
        a false-positive session-dead detection."""
        if not text:
            return False
        stripped = text.strip()
        # "Real" line: at least 12 chars OR contains a space (multi-word).
        return len(stripped) >= 12 or " " in stripped

    def _track_word_event_health(self, text: str, word_timings_count: int) -> None:
        """Called after every successful Azure speak completion. Tracks
        consecutive empty-word-event lines and schedules a synthesizer
        rebuild when the Azure session appears dead."""
        if not self._is_meaningful_speak_text(text):
            return  # don't count tiny utterances
        if word_timings_count > 0:
            if self._azure_consecutive_empty_word_lines > 0:
                print(f"   [Captions] Word-boundary events recovered after {self._azure_consecutive_empty_word_lines} empty line(s).")
            self._azure_consecutive_empty_word_lines = 0
            return
        self._azure_consecutive_empty_word_lines += 1
        if self._azure_consecutive_empty_word_lines >= self._AZURE_EMPTY_LINE_THRESHOLD:
            if not self._azure_reinit_in_progress:
                print(
                    f"   [Captions] !!! Word-boundary events stopped for "
                    f"{self._azure_consecutive_empty_word_lines} consecutive line(s) "
                    f"— Azure session may have dropped, scheduling synthesizer reinit."
                )
                try:
                    loop = self._event_loop_for_captions or asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(
                        self._reinit_azure_synthesizer(reason="empty-word-events threshold"),
                        loop,
                    )
                except Exception as e:
                    print(f"   [Captions] Could not schedule Azure reinit: {e}")

    async def _reinit_azure_synthesizer(self, reason: str = "unknown") -> bool:
        """Atomically rebuild the Azure synthesizer + re-subscribe the
        word_boundary callback. Runs under both _azure_reinit_lock (to
        prevent concurrent reinits) and _azure_tts_lock (to ensure no
        speak call is in flight when we swap the object).
        Returns True on success."""
        if self._azure_reinit_in_progress:
            return False
        async with self._azure_reinit_lock:
            if self._azure_reinit_in_progress:
                return False
            self._azure_reinit_in_progress = True
            try:
                print(f"   [Captions] Rebuilding Azure synthesizer (reason: {reason})...")
                # Take the speak lock so no synth call is mid-flight.
                async with self._azure_tts_lock:
                    old = self.azure_synthesizer
                    new_synth = await asyncio.to_thread(self._build_azure_synthesizer)
                    if not new_synth:
                        print("   [Captions] Azure synthesizer rebuild FAILED — keeping old instance.")
                        return False
                    self.azure_synthesizer = new_synth
                    self._azure_session_generation += 1
                    self._azure_consecutive_empty_word_lines = 0
                    self._azure_last_word_event_time = time.time()
                    # Best-effort release of the old native handle.
                    try:
                        del old
                    except Exception:
                        pass
                print(
                    f"   [Captions] Azure synthesizer rebuilt successfully "
                    f"(generation #{self._azure_session_generation}). Captions should recover on next speak."
                )
                return True
            except Exception as e:
                print(f"   [Captions] Azure synthesizer reinit error: {e}")
                return False
            finally:
                self._azure_reinit_in_progress = False

    async def captions_self_heal_loop(self, interval_seconds: float = 60.0) -> None:
        """Periodic heartbeat that catches the rare case where the empty-line
        counter doesn't trip (e.g. silent stretches with no speaks) but the
        Azure session has still gone stale. Also verifies the caption
        WebSocket server is still alive and triggers recovery if not."""
        # Stale threshold: word events haven't fired in this long despite
        # a recent speak attempt.
        STALE_WORD_EVENT_SECONDS = 180.0
        # Speak-recency window: only consider "stale" if we actually tried
        # to speak recently.
        RECENT_SPEAK_WINDOW = 180.0
        print(f"   [Captions] Self-heal heartbeat active (interval={interval_seconds:.0f}s).")
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                now = time.time()
                # 1) Azure session staleness check
                if (
                    TTS_ENGINE == "azure"
                    and self.azure_synthesizer is not None
                    and self._last_azure_speak_time > 0
                    and (now - self._last_azure_speak_time) < RECENT_SPEAK_WINDOW
                    and (now - self._azure_last_word_event_time) > STALE_WORD_EVENT_SECONDS
                    and not self._azure_reinit_in_progress
                ):
                    print(
                        f"   [Captions] Heartbeat: no word events in "
                        f"{now - self._azure_last_word_event_time:.0f}s despite recent speak "
                        f"— forcing Azure synthesizer reinit."
                    )
                    await self._reinit_azure_synthesizer(reason="heartbeat: stale word events")
                # 2) Caption server liveness check
                try:
                    from kira.expression.caption_server import caption_server as _cs
                    if not getattr(_cs, "_started", False):
                        print("   [Captions] Heartbeat: caption server not started — attempting restart.")
                        try:
                            await _cs.start()
                        except Exception as e:
                            print(f"   [Captions] Heartbeat: caption server restart failed: {e}")
                except Exception as e:
                    print(f"   [Captions] Heartbeat: server check error: {e}")
            except asyncio.CancelledError:
                print("   [Captions] Self-heal heartbeat stopped.")
                raise
            except Exception as e:
                print(f"   [Captions] Heartbeat loop error (continuing): {e}")

    def _streamer_overlay_block(self) -> str:
        """Returns the streamer-mode overlay text when bot.mode == 'streamer', else ''.
        Lives outside self.system_prompt so the cached Block A stays stable across mode flips."""
        try:
            if callable(self._mode_provider) and (self._mode_provider() or "").lower() == "streamer":
                return STREAMER_OVERLAY.strip()
        except Exception:
            pass
        return ""

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "", activity_context: str = "", situational_context: str = "", ambient_audio_context: str = "", max_tokens_override: int = None) -> str:
        # ── CORE PERSONA (Block A) — personality.txt + overlay + emotion. ──────
        # This is NEVER truncated. Everything below is droppable context that
        # gets sacrificed (loudly) in strict priority order if the prompt is
        # too big for the context window.
        base_prompt = self.system_prompt
        overlay = self._streamer_overlay_block()
        if overlay:
            base_prompt += "\n\n" + overlay
        emotion_desc = EMOTION_DESCRIPTORS.get(current_emotion, "Be yourself.")
        base_prompt += f"\n\n[EMOTIONAL STATE: {current_emotion.name} — {emotion_desc}]"

        # ── DROPPABLE CONTEXT BLOCKS — built separately so we can cut them ────
        # individually by priority. Textual assembly order (below) preserves the
        # original prompt layout: activity → memory → scene → ambient.
        block_activity = ""
        if activity_context:
            block_activity = (
                f"\n\n[CURRENT CONTEXT: You and Jonny are currently {activity_context}. "
                "Let this shape what you talk about, reference, and react to.]"
            )

        # Identity / memory facts about the current speaker.
        block_memory = ""
        if memory_context:
            block_memory = (
                f"\n\n[MEMORY NOTES - DO NOT QUOTE OR READ THESE ALOUD]\n"
                f"{memory_context}\n"
                f"Use these to stay consistent and personal. "
                "Do not say 'my memory says' or list them like a database."
            )

        # Current scene context — what is on screen right now.
        block_scene = ""
        if situational_context:
            block_scene = (
                f"\n\n[CURRENT VISUAL PERCEPTION — what is on screen RIGHT NOW]\n"
                f"{situational_context}\n"
                f"This is sense data — what your eyes are taking in. It is NOT a script or narration. "
                f"Do NOT recap or paraphrase it (Jonny saw it too — he doesn't want a closed-captioner). "
                f"If it begins with 'UNCERTAIN:' or contains hedge language, treat it as low-confidence "
                f"and do not commit to specifics. React in YOUR voice — a feeling, quip, callback, take — "
                f"not a description of what is on the screen."
            )

        # Ambient/audio flavor — least protected, first to be cut.
        block_ambient = ""
        if ambient_audio_context:
            block_ambient = (
                f"\n\n[AMBIENT AUDIO — what's being said in the media Jonny is watching, NOT directed at you]\n"
                f"{ambient_audio_context}\n"
                f"This is a best-effort transcript of speech happening in whatever Jonny has on the screen "
                f"(a streamer, narrator, character dialogue, etc.). It is your AWARENESS of the content, "
                f"NOT a script and NOT addressed to you. Jonny's mic is still the only thing you respond to. "
                f"Do NOT quote it, recap it, or read it back verbatim — react to the GIST in your own voice, "
                f"the way a friend on the couch would. The transcript is imperfect: ignore garbled or "
                f"clearly-nonsense fragments rather than confidently building on them. Treat unclear bits "
                f"as 'something's happening over there' instead of asserting them as fact."
            )

        # Token counter that works whether or not local Llama is loaded.
        # When self.llm is None (Groq-only mode), use a conservative char/4 estimate.
        def _count_tokens(s: str) -> int:
            if not s:
                return 0
            if self.llm is not None:
                try:
                    return len(self.llm.tokenize(s.encode("utf-8")))
                except Exception:
                    pass
            return max(1, len(s) // 4)

        # We now use the variable from config for the response buffer
        max_response_tokens = max_tokens_override or LLM_MAX_RESPONSE_TOKENS

        # Context window cap. Local Llama uses its actual n_ctx; Groq's
        # llama-3.1-8b-instant supports 128k but we keep a sane budget so
        # behavior matches the local path.
        if self.llm is not None:
            try:
                real_ctx = int(self.llm.n_ctx())
            except Exception:
                real_ctx = N_CTX
        else:
            real_ctx = max(N_CTX, 16384)
        CHAT_TEMPLATE_FUDGE = 128

        # Budget for system + history combined (leaves room for the response).
        # We must NEVER hand llama_cpp a prompt that exceeds n_ctx — it raises
        # ValueError and kills the whole worker.
        hard_budget = real_ctx - max_response_tokens - CHAT_TEMPLATE_FUDGE

        def _sys_tokens() -> int:
            return _count_tokens(base_prompt + block_activity + block_memory + block_scene + block_ambient)

        def _hist_tokens() -> int:
            return sum(_count_tokens(m["content"]) for m in messages)

        def _total_tokens() -> int:
            return _sys_tokens() + _hist_tokens()

        # ── PRIORITY-ORDERED, LOUD TRUNCATION ─────────────────────────────────
        # Protection order (most → least):
        #   core persona > memory/identity facts > current scene > older history
        #   > ambient/audio flavor.
        # We cut in the reverse order: ambient first, persona never. Silent
        # context amputation is the #1 character-drift risk, so EVERY cut emits
        # a [WARN] PromptTruncation line naming the block, tokens removed, and
        # what survived.
        kept = ["persona"]
        if block_memory:
            kept.append("memory")
        if block_scene or block_activity:
            kept.append("scene")
        if messages:
            kept.append("history")
        if block_ambient:
            kept.append("ambient")

        # 1. Ambient/audio flavor — least protected, drop the whole block first.
        if _total_tokens() > hard_budget and block_ambient:
            n = _count_tokens(block_ambient)
            block_ambient = ""
            kept.remove("ambient")
            print(f"   [WARN] PromptTruncation: dropped ambient-audio (-{n} tokens), kept {kept}")

        # 2. Older history — pop oldest turns, always keep the latest turn.
        if _total_tokens() > hard_budget and len(messages) > 1:
            before = _hist_tokens()
            while _total_tokens() > hard_budget and len(messages) > 1:
                messages.pop(0)
            dropped = before - _hist_tokens()
            if dropped > 0:
                print(f"   [WARN] PromptTruncation: trimmed older-history (-{dropped} tokens), kept {kept}")

        # 3. Current scene context (visual perception + activity framing).
        if _total_tokens() > hard_budget and (block_scene or block_activity):
            n = _count_tokens(block_scene) + _count_tokens(block_activity)
            block_scene = ""
            block_activity = ""
            if "scene" in kept:
                kept.remove("scene")
            print(f"   [WARN] PromptTruncation: dropped scene-context (-{n} tokens), kept {kept}")

        # 4. Memory/identity facts — only sacrificed when nothing lower remains.
        if _total_tokens() > hard_budget and block_memory:
            n = _count_tokens(block_memory)
            block_memory = ""
            if "memory" in kept:
                kept.remove("memory")
            print(f"   [WARN] PromptTruncation: dropped memory-facts (-{n} tokens), kept {kept}")

        # Assemble the final system prompt from whatever survived. Core persona
        # is always intact and always first.
        system_prompt = base_prompt + block_activity + block_memory + block_scene + block_ambient

        # 5. Last resort: the lone remaining user turn is itself too big.
        # Truncate its content (never the persona) so llama_cpp never sees an
        # over-ctx prompt.
        def _final_total() -> int:
            return _count_tokens(system_prompt) + sum(_count_tokens(m["content"]) for m in messages)

        truncated_last = False
        while _final_total() > hard_budget and messages:
            last = messages[-1]
            content = last.get("content", "")
            if len(content) <= 200:
                messages.pop()
                if not messages:
                    # Pathological case — inject a minimal stub so we still
                    # produce a response instead of crashing.
                    messages = [{"role": "user", "content": "(continue)"}]
                    break
            else:
                last["content"] = content[: len(content) // 2]
                truncated_last = True
        if truncated_last:
            if "history" in kept:
                kept.remove("history")
            print(f"   [WARN] PromptTruncation: truncated latest-user-message (final ≈{_final_total()} tokens, budget {hard_budget}), kept {kept}")

        full_prompt = [{"role": "system", "content": system_prompt}] + messages

        # Non-streaming inference. Routed via inference_router so Groq cloud
        # is used by default and local Llama serves as opt-in fallback.
        # Catch context-overflow ValueError so one oversized prompt can never
        # kill chat_batch_worker — return a short fallback line instead.
        def _guarded_inference():
            return route_chat_completion(
                self,
                messages=full_prompt,
                max_tokens=max_response_tokens,
                temperature=0.75,  # Slight bump for creativity
                top_p=0.9,
                repeat_penalty=1.1,  # LOWERED from 1.2 to allow flow
                stop=["<end_of_turn>", "<eos>"],  # Removed "Jonny:" to avoid cutting output early
            )

        try:
            response = await asyncio.to_thread(_guarded_inference)
        except ValueError as e:
            # Almost always "Requested tokens (X) exceed context window".
            # Don't crash the worker — log + graceful fallback.
            print(f"   [LLM] Context overflow in inference ({e}). Returning fallback line.")
            return "Brain's a little fried right now — give me a sec."
        except Exception as e:
            print(f"   [LLM] Inference failed ({type(e).__name__}: {e}). Returning fallback line.")
            return "Hmm, something tripped on my end."
        _meta = response.get("_groq_meta", {})
        if _meta.get("prompt_tokens"):
            self._session_usage["groq_in"]  += _meta["prompt_tokens"]
            self._session_usage["groq_out"] += _meta.get("completion_tokens", 0)
        raw_content = response["choices"][0]["message"]["content"]
        # Regex filter for parentheses
        clean_content = re.sub(r'\(.*?\)', '', raw_content).strip()
        return clean_content
    
    async def tool_inference(self, system: str, user: str, max_tokens: int = 200) -> str:
        def _guarded():
            return route_chat_completion(
                self,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=1.0,
                repeat_penalty=1.1,
            )
        resp = await asyncio.to_thread(_guarded)
        _meta = resp.get("_groq_meta", {})
        if _meta.get("prompt_tokens"):
            self._session_usage["groq_in"]  += _meta["prompt_tokens"]
            self._session_usage["groq_out"] += _meta.get("completion_tokens", 0)
        return resp["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _triage_rescue(decision: str, incoming_line: str) -> str:
        """If the fast classifier returned STAY_QUIET but the line shows clear signals
        of direct address, imperative, question, or social cue, upgrade to BRIEF.
        Shared safety net so every caller of decide_response_mode benefits."""
        if decision != "STAY_QUIET":
            return decision
        content_stripped = (incoming_line or "").strip()
        if not content_stripped:
            return decision
        lower_stripped = content_stripped.lower()
        addressed_to_kira = "kira" in lower_stripped
        imperative_prefixes = (
            "do ", "go ", "try ", "play ", "open ", "show ", "tell ",
            "explain", "sing ", "remember", "let's", "lets ", "let us",
        )
        social_signal = any(s in lower_stripped for s in (
            "thanks", "thank ", " ty ", "brb", "back", "sorry",
        )) or lower_stripped in ("ty", "thanks", "thank you", "brb", "sorry")
        looks_like_question = (
            "?" in content_stripped
            or lower_stripped.startswith((
                "what", "why", "how", "when", "where", "who", "which",
                "is ", "are ", "was ", "were ", "do ", "does ", "did ",
                "can ", "could ", "will ", "would ", "should ",
            ))
            or lower_stripped.startswith(imperative_prefixes)
            or addressed_to_kira
            or social_signal
        )
        if looks_like_question:
            print(f"   [Triage] Upgrading STAY_QUIET → BRIEF (rescue: question/address/imperative/social)")
            return "BRIEF"
        return decision

    async def decide_response_mode(self, recent_history: list, incoming_line: str, scene_context: str, source: str, immersive: bool = False, streamer_mode: bool = False) -> str:
        """Fast triage: should Kira respond fully, briefly, or stay quiet?
        Returns one of: 'RESPOND', 'BRIEF', 'STAY_QUIET'.

        immersive=True     -> bias toward silence (VN, movies, anime, cutscene).
                             Takes priority over streamer_mode — cutscene wins.
        streamer_mode=True -> lower bar: BRIEF over STAY_QUIET on borderline lines.
                             Only active when immersive=False.
        both False         -> companion default: RESPOND bias.

        Applies the shared rescue (STAY_QUIET -> BRIEF on direct-address / imperative /
        question / social-signal patterns) before returning, so every call site benefits."""

        history_lines = []
        for turn in recent_history[-4:]:
            speaker = "Jonny" if turn["role"] == "user" else "Kira"
            history_lines.append(f"{speaker}: {turn['content'][:200]}")
        history_str = "\n".join(history_lines) if history_lines else "(no recent context)"

        if immersive:
            # VN / MEDIA / cutscene path — silence-biased. Takes PRIORITY over streamer_mode.
            # This branch must always win when a cutscene is detected (_triage_immersive=True).
            bias_instruction = (
                "Jonny is consuming media \u2014 a visual novel, movie, anime, or book. He is reading or watching with full attention. "
                "Prefer BRIEF for casual remarks and observations. Reserve RESPOND for direct address by name, "
                "questions clearly aimed at Kira, or moments that obviously invite real engagement. "
                "STAY_QUIET for self-talk, sounds like 'hmm', talking to game characters, or muttering. "
                "When unsure: BRIEF beats RESPOND, and silence beats forced chatter. "
                "But do not default to silence \u2014 a friend reacts; she just keeps it short."
            )
        elif streamer_mode:
            # Streamer path — lower bar than companion; BRIEF beats STAY_QUIET on borderline lines.
            # Immersive=True above still wins when a cutscene is detected, so this never fires
            # during a cutscene.
            bias_instruction = (
                "Jonny is streaming live and Kira is his co-host. "
                "Default to RESPOND. Prefer BRIEF over STAY_QUIET on borderline lines. "
                "BRIEF covers: casual asides, short factual statements about what's happening "
                "(scores, viewer counts, things he's observing), throwaway remarks, anything "
                "clearly meant to be heard even if it's not a direct question. "
                "STAY_QUIET ONLY for: self-talk clearly not directed at anyone, talking directly "
                "to a game character or NPC, moment-to-moment gameplay narration "
                "(describing his own actions while playing — 'okay now I reload', 'this door', "
                "'lower the bollards'), or pure ambient noise with no semantic content. "
                "When unsure: BRIEF."
            )
        else:
            # Companion default — unchanged.
            bias_instruction = (
                "Jonny and Kira are hanging out normally \u2014 gaming, chatting, working, whatever. "
                "Default to RESPOND. Only choose STAY_QUIET when the input is obviously self-talk, muttering, talking to a game "
                "character, ambient noise, or clearly not directed at Kira at all. When unsure, RESPOND."
            )

        system = (
            "You are a fast triage filter for an AI companion named Kira. "
            "Your job: decide whether the latest input warrants a response.\n\n"
            "Output exactly one of: RESPOND, BRIEF, STAY_QUIET\n\n"
            "RESPOND: Jonny directly addresses Kira, asks a question, or says something clearly meant for her.\n"
            "BRIEF: Worth a short reaction (one short sentence) \u2014 Twitch chat hype, casual aside, throwaway remark.\n"
            "STAY_QUIET: Self-talk, muttering, talking to the game, ambient noise, or a moment better met with silence.\n\n"
            f"{bias_instruction}\n\n"
            "Output only the single word. No explanation."
        )

        user = (
            f"Recent conversation:\n{history_str}\n\n"
            f"Current screen: {scene_context or '(no context)'}\n\n"
            f"Source: {source}\n"
            f"Incoming: \"{incoming_line}\"\n\n"
            f"Decision:"
        )

        try:
            raw = await self.tool_inference(system, user, max_tokens=8)
            raw = raw.strip().upper()
            if "STAY_QUIET" in raw or "STAY QUIET" in raw:
                decision = "STAY_QUIET"
            elif "BRIEF" in raw:
                decision = "BRIEF"
            else:
                decision = "RESPOND"
        except Exception as e:
            print(f"   [Triage] Error: {e}; defaulting to RESPOND")
            decision = "RESPOND"
        return self._triage_rescue(decision, incoming_line)

    async def claude_inference(self, messages: list, system_prompt: str, max_tokens: int = 600, force_claude: bool = False, dynamic_context: str = "", use_sonnet: bool = False, purpose: str = "background") -> str:
        """Routes a generation call to Claude.

        use_sonnet=False (default) → CLAUDE_DEEP_MODEL (Opus) — reserved for Invite/Deep only.
        use_sonnet=True            → CLAUDE_CHAT_MODEL (Sonnet) — all background + live non-Invite.

        Falls back to local LLM if Claude unavailable unless force_claude=True.

        system_prompt  — Block A (static personality + rules). Cached with cache_control when
                         ENABLE_PROMPT_CACHING is on. Must be byte-identical across calls.
        dynamic_context — Block C (per-call context: scene, memories, task framing). Never cached."""
        if not self.anthropic_client:
            if force_claude:
                raise RuntimeError("Claude client not initialised and force_claude=True — no local fallback.")
            print("   [Brain] Claude unavailable \u2014 falling back to local Llama.")
            return await self.llm_inference(
                messages=messages,
                current_emotion=EmotionalState.HAPPY,
                memory_context="",
                activity_context="",
                situational_context=system_prompt,
                max_tokens_override=max_tokens,
            )
        try:
            # Claude expects alternating user/assistant messages. Strip system messages.
            claude_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
            if not claude_messages or claude_messages[0]["role"] != "user":
                claude_messages.insert(0, {"role": "user", "content": "(continue)"})

            if ENABLE_PROMPT_CACHING:
                system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
                if dynamic_context:
                    system_param.append({"type": "text", "text": dynamic_context})
            else:
                combined = system_prompt + ("\n\n" + dynamic_context if dynamic_context else "")
                system_param = combined

            # Single quick retry on transient 529 (Overloaded) before we give
            # up and drop to local Llama. Anthropic 529s are usually a brief
            # server hiccup; a 1.5s wait often clears it, and Claude is far
            # better suited to the prompt sizes we build here than the local
            # 4–16k window. Other errors (auth, 4xx, network) skip the retry.
            _model = CLAUDE_CHAT_MODEL if use_sonnet else CLAUDE_DEEP_MODEL
            response = None
            last_err = None
            for attempt in range(2):
                try:
                    response = await self.anthropic_client.messages.create(
                        model=_model,
                        max_tokens=max_tokens,
                        system=system_param,
                        messages=claude_messages,
                    )
                    break
                except Exception as e:
                    last_err = e
                    status = getattr(e, "status_code", None)
                    if status == 529 and attempt == 0:
                        print("   [Brain] Claude 529 (Overloaded) — retrying once after 1.5s...")
                        await asyncio.sleep(1.5)
                        continue
                    raise
            if response is None:
                raise last_err if last_err else RuntimeError("Claude returned no response")
            if hasattr(response, "usage"):
                _in  = response.usage.input_tokens
                _out = response.usage.output_tokens
                _cr  = getattr(response.usage, "cache_read_input_tokens", 0)
                if use_sonnet:
                    self._session_usage["sonnet_in"]  += _in
                    self._session_usage["sonnet_out"] += _out
                else:
                    self._session_usage["opus_in"]  += _in
                    self._session_usage["opus_out"] += _out
                _cost_tracker.record(
                    model=_model, input_tokens=_in, output_tokens=_out,
                    cache_read_tokens=_cr, purpose=purpose,
                )
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return ""
        except Exception as e:
            if force_claude:
                raise
            print(f"   [Brain] Claude call failed: {e}. Falling back to local.")
            _cost_tracker.record_fallback(_model, "local", str(e)[:120])
            return await self.llm_inference(
                messages=messages,
                current_emotion=EmotionalState.HAPPY,
                memory_context="",
                activity_context="",
                situational_context=system_prompt,
                max_tokens_override=max_tokens,
            )

    async def kira_deep_response(self, request: str, scene_context: str = "", memory_context: str = "", recent_history: list = None, max_tokens: int = 400, use_sonnet: bool = False) -> str:
        """Generates a deep, in-character Kira response via Claude.

        use_sonnet=False (default) → Opus. Only the Invite/Deep thoughts path leaves this False.
        use_sonnet=True            → Sonnet. All live non-Invite callers pass True."""
        recent_history = recent_history or []
        # Block A (static, cached): self.system_prompt — personality + tool rules, never changes.
        # Block C (dynamic, uncached): mode framing, scene perception, retrieved memories.
        dynamic_context = "[MODE: Deep Response \u2014 take your time, think carefully, be insightful and in-character.]"
        overlay = self._streamer_overlay_block()
        if overlay:
            dynamic_context += "\n\n" + overlay
        if scene_context:
            dynamic_context += (
                f"\n\n[CURRENT SCENE \u2014 raw perception of what is on screen / playing right now]\n"
                f"{scene_context}\n"
                f"This is sense data \u2014 what your eyes and ears are taking in. It is NOT a script. "
                f"Do NOT recap or paraphrase it (Jonny saw/heard it too). If any line begins with "
                f"'UNCERTAIN:' or contains hedge language, treat it as low-confidence and do not commit "
                f"to specifics. React in YOUR voice \u2014 a feeling, quip, callback, take \u2014 not narration."
            )
        if memory_context:
            dynamic_context += f"\n\n[RELEVANT MEMORIES \u2014 use to stay consistent, do not quote]\n{memory_context}"

        history_to_send = []
        for turn in recent_history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                history_to_send.append({"role": turn["role"], "content": turn["content"]})
        history_to_send.append({"role": "user", "content": request})

        return await self.claude_inference(
            messages=history_to_send,
            system_prompt=self.system_prompt,
            dynamic_context=dynamic_context,
            max_tokens=max_tokens,
            use_sonnet=use_sonnet,
        )

    async def claude_chat_inference(self, messages: list, system_prompt: str, dynamic_context: str = "", max_tokens: int = 400, model_override: str = "", purpose: str = "voice") -> str:
        """Routes a conversational response through Claude Sonnet 4.6. Default voice/Twitch
        response path when Claude is available. Cheaper than Opus, better than local Llama.

        system_prompt   — Block A (static personality + rules). Cached when ENABLE_PROMPT_CACHING.
        dynamic_context — Block C (per-turn context: emotion, memories, scene). Never cached.
        model_override  — when set, route through a cheaper tier (e.g. CLAUDE_HAIKU_MODEL
                          for high-frequency structured extraction). Usage tracked separately."""
        if not self.anthropic_client or not ENABLE_CLAUDE_CHAT:
            return ""  # Caller falls back to local

        claude_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not claude_messages or claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "(continue)"})

        if ENABLE_PROMPT_CACHING:
            system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            if dynamic_context:
                system_param.append({"type": "text", "text": dynamic_context})
        else:
            combined = system_prompt + ("\n\n" + dynamic_context if dynamic_context else "")
            system_param = combined

        _model = model_override or CLAUDE_CHAT_MODEL
        _is_haiku = bool(model_override) and "haiku" in model_override.lower()
        try:
            response = await self.anthropic_client.messages.create(
                model=_model,
                max_tokens=max_tokens,
                system=system_param,
                messages=claude_messages,
            )
            if hasattr(response, "usage"):
                _in  = response.usage.input_tokens
                _out = response.usage.output_tokens
                _cr  = getattr(response.usage, "cache_read_input_tokens", 0)
                if _is_haiku:
                    self._session_usage["haiku_in"]  += _in
                    self._session_usage["haiku_out"] += _out
                else:
                    self._session_usage["sonnet_in"]         += _in
                    self._session_usage["sonnet_out"]        += _out
                    self._session_usage["sonnet_cache_read"] += _cr
                _cost_tracker.record(
                    model=_model, input_tokens=_in, output_tokens=_out,
                    cache_read_tokens=_cr, purpose=purpose,
                )
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return ""
        except Exception as e:
            print(f"   [Brain] {_model} call failed: {e}. Falling back to local Llama.")
            return ""

    async def claude_chat_inference_stream(self, messages: list, system_prompt: str, dynamic_context: str = "", max_tokens: int = 400):
        """Async generator: streams Claude Sonnet 4.6 response chunk-by-chunk.
        Yields text deltas as they arrive. Empty generator if Claude unavailable.

        system_prompt   — Block A (static personality + rules). Cached when ENABLE_PROMPT_CACHING.
        dynamic_context — Block C (per-turn context: emotion, memories, scene). Never cached."""
        if not self.anthropic_client or not ENABLE_CLAUDE_CHAT:
            return

        claude_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not claude_messages or claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "(continue)"})

        if ENABLE_PROMPT_CACHING:
            system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            if dynamic_context:
                system_param.append({"type": "text", "text": dynamic_context})
        else:
            combined = system_prompt + ("\n\n" + dynamic_context if dynamic_context else "")
            system_param = combined

        try:
            async with self.anthropic_client.messages.stream(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=max_tokens,
                system=system_param,
                messages=claude_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                # Capture usage from the final accumulated message
                try:
                    _final = await stream.get_final_message()
                    if _final and hasattr(_final, "usage"):
                        _in  = _final.usage.input_tokens
                        _out = _final.usage.output_tokens
                        _cr  = getattr(_final.usage, "cache_read_input_tokens", 0)
                        self._session_usage["sonnet_in"]         += _in
                        self._session_usage["sonnet_out"]        += _out
                        self._session_usage["sonnet_cache_read"] += _cr
                        _cost_tracker.record(
                            model=CLAUDE_CHAT_MODEL, input_tokens=_in, output_tokens=_out,
                            cache_read_tokens=_cr, purpose="voice",
                        )
                except Exception:
                    pass  # usage unavailable on cancelled/error stream
        except Exception as _stream_err:
            print(f"   [Brain] Claude stream error: {_stream_err}")
            return

    async def analyze_emotion_of_turn(self, last_user_text: str, last_ai_response: str) -> EmotionalState | None:
        # No backend check — router handles local-vs-Groq. Only bail if neither
        # local Llama is loaded AND Groq is unconfigured/disabled.
        if self.llm is None and (INFERENCE_BACKEND or "").lower() != "groq":
            return None
        emotion_names = [e.name for e in EmotionalState]
        prompt = (f"Jonny: \"{last_user_text}\"\nKira: \"{last_ai_response}\"\n\n"
                  f"Based on this exchange, which emotional state should Kira be in for her NEXT response? "
                  f"Weight Jonny's tone and the subject matter more heavily than Kira's own output — "
                  f"her state should reflect the MOMENT and WHAT JONNY BROUGHT, not just echo her last line. "
                  f"Options: {', '.join(emotion_names)}.\n"
                  f"Respond ONLY with the single best state name (e.g., 'SASSY').")
        try:
            def _guarded_emotion():
                return route_chat_completion(
                    self,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.2,
                    stop=["\n", ".", ","],
                )

            response = await asyncio.to_thread(_guarded_emotion)
            text_response = response["choices"][0]["message"]["content"].strip().upper()

            for emotion in EmotionalState:
                if emotion.name in text_response:
                    return emotion
            return None
        except Exception as e:
            print(f"   ERROR during emotion analysis: {e}")
            return None

    async def _acquire_speech_gate(self, priority: int):
        """Acquire the single TTS playback gate, honoring priority lanes.

        If the gate is free and nobody is waiting, take it immediately. Otherwise
        queue and await; the releaser hands ownership to the highest-priority
        waiter (lowest number; FIFO within a lane). Single-threaded asyncio means
        the check-and-take below is atomic (no await between)."""
        if not self._speech_gate_held and not self._speech_waiters:
            self._speech_gate_held = True
            return
        self._speech_seq += 1
        entry = [priority, self._speech_seq, asyncio.get_running_loop().create_future()]
        self._speech_waiters.append(entry)
        await entry[2]
        # Ownership was transferred to us by the releaser; gate stays held.

    def _release_speech_gate(self):
        """Release the gate. If waiters exist, transfer ownership to the
        highest-priority one (keeping the gate held on its behalf)."""
        if self._speech_waiters:
            self._speech_waiters.sort(key=lambda e: (e[0], e[1]))
            entry = self._speech_waiters.pop(0)
            if not entry[2].done():
                entry[2].set_result(None)
            # Gate remains held; the woken waiter now owns it.
        else:
            self._speech_gate_held = False

    async def speak_text(self, text: str, priority: int = 1):
        """Generates and plays audio for the given text (blocking). Sets is_speaking
        around the call so VAD ignores self-hearing. Used by non-streaming callers.

        priority: TTS lane (0=voice, 1=interjection/announcement, 2=chat-batch).
        Defaults to P1 — chat-batch callers must pass priority=2 explicitly."""
        if not text:
            return
        self.interruption_event.clear()
        self.is_speaking = True
        try:
            await self._speak_single(text, priority=priority)
        finally:
            self.last_speech_finish_time = time.time()
            self.is_speaking = False

    async def speak_text_vn(self, text: str, ssml_inner: str | None = None, priority: int = 1):
        """VN autopilot TTS: accepts pre-built SSML inner content for prosody variation.

        When ssml_inner is provided and Azure TTS is active, wraps it inside the
        standard config <prosody> envelope for natural, context-sensitive delivery.
        Falls back to speak_text for non-Azure engines or when no SSML is provided.
        """
        if not text:
            return
        self.interruption_event.clear()
        self.is_speaking = True
        try:
            if ssml_inner is not None and self.tts_backend != "fish" and TTS_ENGINE == "azure" and self.azure_synthesizer:
                ssml = (
                    f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                    f'xml:lang="en-US">'
                    f'<voice name="{AZURE_SPEECH_VOICE}">'
                    f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">'
                    f'{ssml_inner}'
                    f'</prosody></voice></speak>'
                )
                print(f"   [TTS] Speaking: {text[:50]}...")
                await self._acquire_speech_gate(priority)
                try:
                    # Serialize with the lock so a concurrent _speak_single
                    # from a Kira interjection can't reset the shared
                    # word-boundary buffer mid-synth.
                    async with self._azure_tts_lock:
                        # Reset word-boundary buffer before kicking synth.
                        self._azure_word_buffer = []
                        self._last_azure_speak_time = time.time()
                        result = await asyncio.to_thread(
                            self.azure_synthesizer.speak_ssml, ssml
                        )
                        # Snapshot INSIDE the lock so the next call can't
                        # clobber the buffer before we copy it.
                        word_timings = list(self._azure_word_buffer) if (
                            result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
                        ) else []
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        # Track Azure session health BEFORE deciding whether
                        # to push the caption frame — this is what catches a
                        # silently-dead word_boundary subscription.
                        self._track_word_event_health(text, len(word_timings))
                        if word_timings:
                            try:
                                from kira.expression.caption_server import enqueue_caption
                                if self._event_loop_for_captions is None:
                                    try:
                                        self._event_loop_for_captions = asyncio.get_running_loop()
                                    except RuntimeError:
                                        pass
                                print(f"   [Captions] Pushed frame (vn): '{text[:40]}' ({len(word_timings)} words)")
                                enqueue_caption(self._event_loop_for_captions, text, word_timings)
                            except Exception as e:
                                print(f"   [Captions] dispatch suppressed: {e}")
                        else:
                            print(f"   [Captions] No word-boundary events captured for '{text[:40]}' — caption skipped.")
                        await self._play_audio_with_pygame(result.audio_data)
                    else:
                        print(f"   TTS Fail: {result.cancellation_details.error_details}")
                except Exception as e:
                    print(f"   TTS/Playback Error: {e}")
                finally:
                    self._release_speech_gate()
            else:
                await self._speak_single(text, priority=priority)
        finally:
            self.last_speech_finish_time = time.time()
            self.is_speaking = False

    async def _speak_single(self, text: str, priority: int = 1, is_stream: bool = False, already_gated: bool = False):
        """Pure synthesis-and-playback for a single text block. Does NOT manage
        is_speaking or interruption_event — caller is responsible. Used by both
        speak_text (single shot) and speak_streaming (per-sentence dispatch).

        priority: TTS lane for the playback gate (0=voice, 1=interjection, 2=chat).
        is_stream: True only for the streaming voice path — gates the
            _lat_audio_out_ms stamp so a concurrent non-voice line can never
            clobber the voice turn's latency slot (the negative-audio_out bug).
        already_gated: True when the caller (speak_streaming) already holds the
            speech gate for the whole logical turn. Skips per-sentence
            acquire/release so a lower-priority line can't wedge itself BETWEEN
            the sentences of one streamed voice reply."""
        if not text:
            return
        if self.interruption_event.is_set():
            return

        # Universal safety net: strip any recognized tool tags before synthesis.
        # Every spoken line funnels through here (speak_text, speak_streaming
        # per-sentence dispatch), so this guarantees tool syntax never reaches
        # TTS or the caption overlay on ANY path, even if an upstream stripper
        # was bypassed or a tag was truncated mid-stream.
        text = self._strip_tags_for_speech(text)
        if not text:
            return

        # ── Hard content backstop — the last line before TTS/captions ──────────
        # A first-mover / contrarian Kira generates takes unprompted, so this
        # suppresses the most unambiguous, clipboard-able violations (slurs) before
        # they ever reach the stream — regardless of bit/roleplay framing or who
        # "permitted" it. The persona's HARD CONTENT BOUNDARY is the primary,
        # context-aware rail; this is the non-bypassable net under it. Logged loudly
        # so Jonny can see exactly what was caught.
        if self._content_safety_block(text) is not None:
            print(f"   [Guardrail] BLOCKED before TTS (hard content boundary) — "
                  f"suppressed utterance: {text[:140]!r}")
            return

        # Capture the running loop once — the caption server needs a stable
        # reference for thread-safe scheduling from Azure's worker callbacks.
        if self._event_loop_for_captions is None:
            try:
                self._event_loop_for_captions = asyncio.get_running_loop()
            except RuntimeError:
                pass

        print(f"   [TTS] Speaking: {text[:50]}...")
        audio_data = None
        word_timings: list[dict] = []

        if not already_gated:
            await self._acquire_speech_gate(priority)
        try:
            # Re-check barge-in after the (possibly long) gate wait — if Jonny
            # interrupted while this line was queued, abandon it immediately so
            # the queue drains instead of synthing stale audio.
            if self.interruption_event.is_set():
                return
            if self.tts_backend == "fish" and self.fish_session and FISH_SDK_AVAILABLE:
                # ---- Fish Audio streaming path ----
                # Runs in a thread because the SDK's .tts() is a blocking generator.
                # No word-boundary timing available from Fish; degrade to single-frame
                # caption (same as edge TTS). Falls back to Azure on any error.
                fish_ok = False
                try:
                    def _fish_collect() -> bytes:
                        buf = b""
                        for chunk in self.fish_session.tts(FishTTSRequest(
                            text=text,
                            reference_id=self.fish_voice_id or None,
                            latency=FISH_LATENCY,
                            format=FISH_FORMAT,
                        )):
                            buf += chunk
                        return buf
                    audio_data = await asyncio.to_thread(_fish_collect)
                    word_timings = [{"word": text.strip(), "offset_ms": 0}] if text.strip() else []
                    fish_ok = True
                    print(f"   [TTS] backend=fish ({len(audio_data)} bytes)")
                except Exception as fish_err:
                    print(f"   [TTS] Fish error ({fish_err}); falling back to Azure for this utterance.")

                if not fish_ok:
                    # Azure fallback inside Fish branch
                    if self.azure_synthesizer:
                        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                                f'<voice name="{AZURE_SPEECH_VOICE}">'
                                f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">{text}</prosody>'
                                f'</voice></speak>')
                        async with self._azure_tts_lock:
                            self._azure_word_buffer = []
                            self._last_azure_speak_time = time.time()
                            result = await asyncio.to_thread(self.azure_synthesizer.speak_ssml, ssml)
                            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                                audio_data = result.audio_data
                                word_timings = list(self._azure_word_buffer)
                            else:
                                print(f"   TTS Fail (Azure fallback): {result.cancellation_details.error_details}")
                        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                            self._track_word_event_health(text, len(word_timings))
                            print(f"   [TTS] backend=azure-fallback")

            elif TTS_ENGINE == "azure" and self.azure_synthesizer:
                ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                        f'<voice name="{AZURE_SPEECH_VOICE}">'
                        f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">{text}</prosody>'
                        f'</voice></speak>')
                # Serialize Azure synth so the shared word-boundary buffer
                # can't be reset by a concurrent speak_text_vn / interjection
                # before we snapshot. See __init__ for full rationale.
                async with self._azure_tts_lock:
                    # Reset the per-call word-boundary buffer before kicking synth.
                    self._azure_word_buffer = []
                    self._last_azure_speak_time = time.time()
                    result = await asyncio.to_thread(self.azure_synthesizer.speak_ssml, ssml)
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        audio_data = result.audio_data
                        # Snapshot INSIDE the lock before the next caller can
                        # clear the shared buffer.
                        word_timings = list(self._azure_word_buffer)
                    else:
                        print(f"   TTS Fail: {result.cancellation_details.error_details}")
                # Track Azure session health AFTER releasing the lock so a
                # scheduled reinit doesn't deadlock on it.
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    self._track_word_event_health(text, len(word_timings))
                print(f"   [TTS] backend=azure")
            elif TTS_ENGINE == "edge" and Communicate:
                voice = AZURE_SPEECH_VOICE if AZURE_SPEECH_VOICE else "en-US-AriaNeural"
                communicate = Communicate(text, voice)
                buffer = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer += chunk["data"]
                audio_data = buffer
                # Edge TTS doesn't expose word timing here; degrade to a
                # single-frame caption so the overlay still shows the line.
                word_timings = [{"word": text.strip(), "offset_ms": 0}] if text.strip() else []
                print(f"   [TTS] backend=edge")

            if audio_data:
                # Push the caption frame to the overlay just before audio starts.
                # Fire-and-forget; never blocks playback.
                try:
                    from kira.expression.caption_server import enqueue_caption
                    if word_timings:
                        print(f"   [Captions] Pushed frame: '{text[:40]}' ({len(word_timings)} words)")
                        enqueue_caption(self._event_loop_for_captions, text, word_timings)
                    else:
                        print(f"   [Captions] No word-boundary events captured for '{text[:40]}' — caption skipped.")
                except Exception as e:
                    print(f"   [Captions] dispatch suppressed: {e}")
                # Latency: stamp first actual playback start (relative to stream
                # start). Only the streaming voice path stamps this — a concurrent
                # non-voice line must never write the voice turn's slot (that was
                # the negative-audio_out bug).
                if is_stream and self._lat_audio_out_ms < 0 and self._lat_stream_t0 > 0:
                    self._lat_audio_out_ms = int((time.time() - self._lat_stream_t0) * 1000)
                # Wall-clock playback-start stamp for the [LAG] sense->speak metric.
                self.last_playback_start_time = time.time()
                await self._play_audio_with_pygame(audio_data)
        except Exception as e:
            print(f"   TTS/Playback Error: {e}")
        finally:
            if not already_gated:
                self._release_speech_gate()

    async def speak_streaming(self, stream_generator, priority: int = 0) -> str:
        """Consumes an async generator of text chunks. Buffers into sentences,
        dispatches each to TTS sequentially. Tool tags ([POLL: ...], [SONG: ...],
        [PREDICT: ...], [BIT: ...], [TAKE]) are extracted and processed separately —
        never spoken aloud.

        Sentences are split on terminal punctuation [.!?] followed by whitespace,
        but ONLY when no unclosed bracket tag is in flight.

        If the user interrupts (via VAD), the stream is abandoned."""

        self.interruption_event.clear()
        self.is_speaking = True
        # D: mark a voice turn in flight for the whole stream so a concurrent
        # Bored-interjection yields the speech gate at its next sentence boundary.
        self._voice_turn_active = True

        full_text = ""
        buffer = ""
        _t0_stream = time.time()
        _first_word_logged = False

        # Latency instrumentation: reset per-stream slots.
        self._lat_stream_t0 = _t0_stream
        self._lat_ttft_ms = -1
        self._lat_tts_first_chunk_ms = -1
        self._lat_audio_out_ms = -1

        def has_unclosed_tag(text: str) -> bool:
            """True if there is an unmatched '[' anywhere in the buffer."""
            last_open = text.rfind('[')
            if last_open == -1:
                return False
            return ']' not in text[last_open:]

        # Hold the speech gate for the WHOLE logical turn, not per sentence.
        # Acquired lazily right before the first spoken sentence (so LLM token
        # generation can overlap with a lower-priority line still playing), then
        # held until the stream finishes. This is what stops a P2 chat line from
        # wedging itself BETWEEN the sentences of one streamed voice reply.
        _gate_held = False

        try:
            async for chunk in stream_generator:
                if self.interruption_event.is_set():
                    break

                # Latency: stamp time-to-first-token on the first chunk received.
                if self._lat_ttft_ms < 0:
                    self._lat_ttft_ms = int((time.time() - _t0_stream) * 1000)

                full_text += chunk
                buffer += chunk

                # If a bracket tag is still being streamed, don't split on punctuation yet
                if has_unclosed_tag(buffer):
                    continue

                # Flush every complete sentence from the buffer
                while True:
                    if has_unclosed_tag(buffer):
                        break
                    match = re.search(r'[.!?](?:\s|$)', buffer)
                    if not match:
                        break
                    end = match.end()
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]

                    if sentence:
                        # Strip tool tags out of the spoken text. We don't execute them
                        # here — parse_kira_tools runs on full_text after streaming ends.
                        spoken = self._strip_tags_for_speech(sentence)
                        cleaned = self._clean_llm_response(spoken)
                        if cleaned:
                            if not _first_word_logged:
                                _ttfw_ms = int((time.time() - _t0_stream) * 1000)
                                print(f"   [TIMING] first-word: {_ttfw_ms}ms")
                                _first_word_logged = True
                                if self._lat_tts_first_chunk_ms < 0:
                                    self._lat_tts_first_chunk_ms = _ttfw_ms
                            # Lazily grab the gate once, before the first audible
                            # sentence; hold it for the rest of the turn.
                            if not _gate_held:
                                await self._acquire_speech_gate(priority)
                                _gate_held = True
                            await self._speak_single(cleaned, priority=priority, is_stream=True, already_gated=True)
                            if self.interruption_event.is_set():
                                return full_text

            # Stream done — flush any trailing buffer
            if buffer.strip() and not self.interruption_event.is_set():
                spoken = self._strip_tags_for_speech(buffer.strip())
                cleaned = self._clean_llm_response(spoken)
                if cleaned:
                    if not _first_word_logged:
                        _ttfw_ms = int((time.time() - _t0_stream) * 1000)
                        print(f"   [TIMING] first-word: {_ttfw_ms}ms")
                        if self._lat_tts_first_chunk_ms < 0:
                            self._lat_tts_first_chunk_ms = _ttfw_ms
                    if not _gate_held:
                        await self._acquire_speech_gate(priority)
                        _gate_held = True
                    await self._speak_single(cleaned, priority=priority, is_stream=True, already_gated=True)
        except Exception as e:
            print(f"   [Streaming] Error during stream consumption: {e}")
        finally:
            if _gate_held:
                self._release_speech_gate()
            self.last_speech_finish_time = time.time()
            self.is_speaking = False
            self._voice_turn_active = False

        return full_text

    def has_pending_voice_turn(self) -> bool:
        """True if a P0 (voice) turn is in flight or queued for the speech gate.

        Covers both states: (a) a voice turn currently generating/playing
        (_voice_turn_active, set for the whole speak_streaming lifetime), and
        (b) a P0 entry sitting in the gate's waiter queue. The Bored-interjection
        loop checks this at each sentence boundary so it can stop and hand the
        gate to a ready voice response instead of holding it for 5-11s."""
        if self._voice_turn_active:
            return True
        if self._voice_response_pending:
            return True
        return any(w[0] == 0 for w in self._speech_waiters)

    def _content_safety_block(self, text: str):
        """Return the matched denylisted term if `text` trips the hard content
        backstop, else None. Word-boundary matched and NARROW by design (slurs /
        unambiguous explicit terms) — the persona's HARD CONTENT BOUNDARY handles the
        context-dependent categories (atrocity/sexual/self-harm/violence-as-joke)."""
        if not text or not self._content_denylist_re:
            return None
        m = self._content_denylist_re.search(text)
        return m.group(0) if m else None

    def _strip_tags_for_speech(self, text: str) -> str:
        """Removes ALL recognized tool tags ([POLL:], [SONG:], [PREDICT:], [BIT:],
        [CHAT:], [TAKE]) from text before TTS/captions, so they are never spoken
        aloud or shown on the overlay. Tags are still parsed and executed by
        parse_kira_tools on the full assembled response.

        Robust to UNCLOSED tags: when a streamed response is truncated mid-tag
        (e.g. the max_tokens cap lands inside a [PREDICT: ...]) the trailing
        fragment has no closing ']'. The closed-tag passes can't match that, so a
        final pass strips any trailing unclosed recognized tag too. This is the
        path that previously leaked tool syntax to TTS/captions."""
        # Multi-pipe tags: POLL, PREDICT
        text = re.sub(r'\[POLL:[^\]]*\]', '', text)
        text = re.sub(r'\[PREDICT:[^\]]*\]', '', text)
        # Single-value tags
        text = re.sub(r'\[SONG:[^\]]*\]', '', text)
        text = re.sub(r'\[BIT:[^\]]*\]', '', text)
        text = re.sub(r'\[CHAT:[^\]]*\]', '', text)
        # Boolean tags
        text = re.sub(r'\[TAKE\]', '', text)
        # Trailing UNCLOSED recognized tag (truncated mid-stream — no closing ']').
        text = re.sub(r'\[(?:POLL|PREDICT|SONG|BIT|CHAT|TAKE)\b[^\]]*$', '', text)
        # Speak handles naturally: never read inter-word underscores aloud as
        # "underscore" (e.g. "the_one_over_there" → "the one over there"). Only
        # underscores BETWEEN word chars are converted; leading/trailing left alone.
        # Universal funnel, so it also covers handles that slip through any prompt path.
        text = re.sub(r'(?<=\w)_(?=\w)', ' ', text)
        return text.strip()

    async def _play_audio_with_pygame(self, audio_bytes: bytes):
        if self.interruption_event.is_set() or not audio_bytes:
            print("   [Audio] Skipped: Interrupted or Empty.")
            return

        try:
            if not pygame.mixer.get_init(): pygame.mixer.init(devicename=None)
            
            if pygame.mixer.get_busy(): pygame.mixer.stop()
            pygame.mixer.music.stop()

            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.set_volume(1.0)
            channel = sound.play()
            
            while channel.get_busy():
                if self.interruption_event.is_set():
                    channel.stop(); break
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"   Audio Playback Error: {e}")


    def _clean_llm_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Prune all bracketed metadata (Visual Sparks, thoughts, etc)
        # Preserve tool tags [POLL:...] and [SONG:...] so parse_kira_tools() can handle them
        text = re.sub(r'\[(?!POLL:|SONG:|PREDICT:|BIT:|TAKE\])[^\]]*\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text.strip()

    def _build_stt_prompt(self, activity: str = "") -> str:
        """Build an initial_prompt for Whisper vocabulary biasing.

        Returns a comma-separated list: static base names + up to MAX_LORE_NOUNS
        proper nouns extracted from the activity's lore file (if it exists).
        Capped to prevent over-stuffing the prompt, which degrades general accuracy.
        Safe fallback to base list only if the lore file is absent or unreadable.
        """
        BASE_NOUNS = ["Jonny", "Kira", "Twitch", "StreamElements", "Streamlabs"]
        MAX_LORE_NOUNS = 12

        # Capitalised words that carry no STT-biasing value (structural/common words).
        _SKIP = {
            "I", "The", "A", "An", "In", "On", "At", "To", "Of", "And", "Or",
            "But", "For", "Not", "With", "Is", "Was", "He", "She", "They",
            "It", "This", "That", "His", "Her", "Their", "My", "Your", "We",
            "So", "If", "As", "By", "Up", "About", "After", "Before", "When",
            "Also", "Now", "Then", "There", "Here", "Just", "Even", "Still",
            "Session", "Lore", "Story", "Chat", "Stream", "Mode", "Scene",
            "Notes", "First", "Last", "Next", "New", "Old", "Good", "Bad",
            "Stage", "One", "Two", "Three", "Four", "Five", "Jonny", "Kira",
        }

        lore_nouns: list[str] = []
        if activity:
            slug = re.sub(r'[^a-zA-Z0-9]+', '_', activity).strip('_').lower()[:40]
            lore_path = os.path.join("lore", f"{slug}.md")
            if os.path.exists(lore_path):
                try:
                    with open(lore_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    counts: dict[str, int] = {}

                    # Priority 1: **bolded terms** — the LLM explicitly highlighted these.
                    # Each bolded match counts as 3 to outrank low-frequency prose words.
                    for m in re.finditer(r'\*\*([^*\n]{2,40})\*\*', text):
                        term = m.group(1).strip()
                        if 2 <= len(term) <= 30:
                            counts[term] = counts.get(term, 0) + 3

                    # Priority 2: capitalised word sequences (1–3 words) in prose.
                    for m in re.finditer(
                        r'\b([A-Z][a-z]{1,14}(?:\s[A-Z][a-z]{1,14}){0,2})\b', text
                    ):
                        term = m.group(1)
                        if term.split()[0] not in _SKIP:
                            counts[term] = counts.get(term, 0) + 1

                    # Keep terms that appear at least twice overall (prose words need
                    # 2+ hits; bolded terms already score ≥3 on first occurrence).
                    candidates = sorted(
                        [t for t, c in counts.items() if c >= 2 and 2 <= len(t) <= 30],
                        key=lambda t: (-counts[t], t),
                    )
                    lore_nouns = candidates[:MAX_LORE_NOUNS]
                except Exception:
                    pass  # safe fallback: use base list only

        all_nouns = BASE_NOUNS + [n for n in lore_nouns if n not in BASE_NOUNS]
        return ", ".join(all_nouns)

    async def transcribe_audio(self, audio_data: bytes, activity: str = "") -> str:
        # Using numpy array directly for speed
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        initial_prompt = self._build_stt_prompt(activity)

        def _run_transcribe():
            # initial_prompt seeds the decoder with known vocabulary for proper-noun biasing.
            # condition_on_previous_text=False: prevents hallucinating filler when audio is ambiguous.
            # no_speech_threshold=0.85: rejects segments that are very likely silence/noise.
            # language="en": skip language detection overhead, commit to English.
            segments, info = self.whisper.transcribe(
                arr,
                beam_size=5,
                language="en",
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
                no_speech_threshold=0.85,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.85,
                    min_speech_duration_ms=400,
                    min_silence_duration_ms=800,
                    speech_pad_ms=300
                )
            )
            return list(segments)

        segments = await asyncio.to_thread(_run_transcribe)
        text = "".join([segment.text for segment in segments])
        return text.strip()
