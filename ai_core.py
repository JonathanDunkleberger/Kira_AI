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

from config import (
    LLM_MODEL_PATH, N_CTX, N_BATCH, N_GPU_LAYERS, WHISPER_MODEL_SIZE, WHISPER_CACHE_DIR, TTS_ENGINE,
    LLM_MAX_RESPONSE_TOKENS,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    AI_NAME,
    ANTHROPIC_API_KEY, CLAUDE_DEEP_MODEL, ENABLE_CLAUDE_BRAIN,
    CLAUDE_CHAT_MODEL, ENABLE_CLAUDE_CHAT, ENABLE_PROMPT_CACHING, ENABLE_CLAUDE_STREAMING,
)
from persona import EmotionalState
from personality_file import KIRA_PERSONALITY

# Maps each emotional state to a concise behavioural directive injected into the prompt.
# Must stay in sync with EmotionalState in persona.py.
EMOTION_DESCRIPTORS = {
    EmotionalState.HAPPY:       "Default mode. Be cheerful, curious, and let your sassy wit flow naturally.",
    EmotionalState.MOODY:       "You are withdrawn and angsty. Keep answers shorter and more sarcastic than usual.",
    EmotionalState.SASSY:       "Your wit is razor-sharp right now. Tease more, soften less. Make it sting but stay fun.",
    EmotionalState.EMOTIONAL:   "You feel open and earnest. It is okay to say something genuinely sweet or heartfelt.",
    EmotionalState.HYPERACTIVE: "You are buzzing with excitement. Ramble a little. Everything feels more interesting than normal.",
}
from prompt_loader import load_personality_txt
from prompt_rules import TOOL_AND_FORMAT_RULES

# Graceful SDK imports
try: from edge_tts import Communicate
except ImportError: Communicate = None
try: from elevenlabs.client import AsyncElevenLabs
except ImportError: AsyncElevenLabs = None
try: import azure.cognitiveservices.speech as speechsdk
except ImportError: speechsdk = None
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        self.is_initialized = False
        self.is_speaking = False # Added flag for self-hearing prevention
        self.last_speech_finish_time = time.time() # Added for silence tracking
        self.last_tts_request_time = 0 # Added for TTS rate limiting
        self.llm = None
        self.whisper = None
        self.eleven_client = None
        self.azure_synthesizer = None
        self.inference_lock = threading.Lock() # Added lock for inference safety

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
            self.system_prompt = new_personality.strip() + "\n\n" + TOOL_AND_FORMAT_RULES.strip()
            print("   ✅ Personality reloaded successfully.")
            print("----- NEW SYSTEM PROMPT SNAPSHOT -----")
            print(self.system_prompt[:200])
            print("--------------------------------------")
        except Exception as e:
            print(f"   ❌ Failed to reload personality: {e}")

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

    def _init_llm(self):
        print(f"-> Loading LLM model... (GPU Layers: {N_GPU_LAYERS})")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")

        # Optimized for RTX 5080: n_threads=8, Flash Attention enabled, Force GPU Layers=-1
        # Updated for Gemma 3 (RTX 5080 Maximum Optimization)
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
        
        # Combine Personality File + Tools
        self.system_prompt = KIRA_PERSONALITY.strip() + "\n\n" + TOOL_AND_FORMAT_RULES.strip()

        print("----- SYSTEM PROMPT (FIRST 400 CHARS) -----")
        print(self.system_prompt[:400])
        print("------------------------------------------")
        print(f"   LLM loaded. Path: {LLM_MODEL_PATH}")

    def _init_whisper(self):
        print("-> Loading Faster-Whisper STT model...")
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

            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY.strip(), region=AZURE_SPEECH_REGION.strip())

            # Route Azure output to a null stream so Pygame handles playback exclusively.
            class NullCallback(speechsdk.audio.PushAudioOutputStreamCallback):
                def write(self, data: memoryview) -> int: return data.nbytes
                def close(self) -> None: pass
                
            stream = speechsdk.audio.PushAudioOutputStream(NullCallback())
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        elif TTS_ENGINE == "edge":
            if not Communicate: raise ImportError("Run 'pip install edge-tts'")
        else:
            raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")
        print(f"   {TTS_ENGINE.capitalize()} TTS ready.")

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "", activity_context: str = "", situational_context: str = "", max_tokens_override: int = None) -> str:
        # Use our updated system prompt if available, else fallback
        system_prompt = self.system_prompt
        emotion_desc = EMOTION_DESCRIPTORS.get(current_emotion, "Be yourself.")
        system_prompt += f"\n\n[EMOTIONAL STATE: {current_emotion.name} — {emotion_desc}]"

        if activity_context:
            system_prompt += (
                f"\n\n[CURRENT CONTEXT: You and Jonny are currently {activity_context}. "
                "Let this shape what you talk about, reference, and react to.]"
            )
        
        # INJECT MEMORY AS NOTES
        if memory_context:
            system_prompt += (
                f"\n\n[MEMORY NOTES - DO NOT QUOTE OR READ THESE ALOUD]\n"
                f"{memory_context}\n"
                f"Use these to stay consistent and personal. "
                "Do not say 'my memory says' or list them like a database."
            )

        if situational_context:
            system_prompt += (
                f"\n\n[CURRENT PERCEPTION — what is on screen RIGHT NOW]\n"
                f"{situational_context}\n"
                f"This is your live awareness of the screen as of this moment. "
                f"Do not treat it as something from the past or repeat it verbatim."
            )

        system_tokens = self.llm.tokenize(system_prompt.encode("utf-8"))
        
        # We now use the variable from config for the response buffer
        max_response_tokens = LLM_MAX_RESPONSE_TOKENS
        token_limit = N_CTX - len(system_tokens) - max_response_tokens

        history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"))) for m in messages)
        while history_tokens > token_limit and len(messages) > 1:
            print("   (Trimming conversation history to fit context window...)")
            messages.pop(0)
            history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"))) for m in messages)
            
        full_prompt = [{"role": "system", "content": system_prompt}] + messages
        
        # Non-streaming inference for maximum throughput on RTX 5080
        # Wrap inference in lock to prevent collisions with background agents
        def _guarded_inference():
            with self.inference_lock:
                 return self.llm.create_chat_completion(
                    messages=full_prompt,
                    max_tokens=(max_tokens_override or LLM_MAX_RESPONSE_TOKENS),
                    temperature=0.75, # Slight bump for creativity
                    top_p=0.9,
                    min_p=0.05,
                    repeat_penalty=1.1, # LOWERED from 1.2 to allow flow
                    stop=["<end_of_turn>", "<eos>"], # Removed "Jonny:" to avoid cutting output early
                    stream=False 
                 )
                 
        response = await asyncio.to_thread(_guarded_inference)
        raw_content = response["choices"][0]["message"]["content"]
        # Regex filter for parentheses
        clean_content = re.sub(r'\(.*?\)', '', raw_content).strip()
        return clean_content
    
    async def tool_inference(self, system: str, user: str, max_tokens: int = 200) -> str:
        def _guarded():
            with self.inference_lock:
                return self.llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=1.0,
                    repeat_penalty=1.1,
                    stream=False
                )
        resp = await asyncio.to_thread(_guarded)
        return resp["choices"][0]["message"]["content"].strip()

    async def decide_response_mode(self, recent_history: list, incoming_line: str, scene_context: str, source: str, immersive: bool = False) -> str:
        """Fast triage: should Kira respond fully, briefly, or stay quiet?
        Returns one of: 'RESPOND', 'BRIEF', 'STAY_QUIET'.

        immersive=True  -> bias toward silence (passive media: VN, movies, anime).
        immersive=False -> bias toward responding (default companion behavior)."""

        history_lines = []
        for turn in recent_history[-4:]:
            speaker = "Jonny" if turn["role"] == "user" else "Kira"
            history_lines.append(f"{speaker}: {turn['content'][:200]}")
        history_str = "\n".join(history_lines) if history_lines else "(no recent context)"

        if immersive:
            bias_instruction = (
                "Jonny is consuming media \u2014 a visual novel, movie, anime, or book. He is reading or watching with full attention. "
                "Prefer BRIEF for casual remarks and observations. Reserve RESPOND for direct address by name, "
                "questions clearly aimed at Kira, or moments that obviously invite real engagement. "
                "STAY_QUIET for self-talk, sounds like 'hmm', talking to game characters, or muttering. "
                "When unsure: BRIEF beats RESPOND, and silence beats forced chatter. "
                "But do not default to silence \u2014 a friend reacts; she just keeps it short."
            )
        else:
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
                return "STAY_QUIET"
            if "BRIEF" in raw:
                return "BRIEF"
            return "RESPOND"
        except Exception as e:
            print(f"   [Triage] Error: {e}; defaulting to RESPOND")
            return "RESPOND"

    async def claude_inference(self, messages: list, system_prompt: str, max_tokens: int = 600, force_claude: bool = False) -> str:
        """Routes a generation call to Claude Opus. Used for deep cognitive moments
        where intelligence matters more than latency. Falls back to local LLM if Claude unavailable
        unless force_claude=True, in which case exceptions are re-raised for callers to handle."""
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
            else:
                system_param = system_prompt

            response = await self.anthropic_client.messages.create(
                model=CLAUDE_DEEP_MODEL,
                max_tokens=max_tokens,
                system=system_param,
                messages=claude_messages,
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return ""
        except Exception as e:
            if force_claude:
                raise
            print(f"   [Brain] Claude call failed: {e}. Falling back to local.")
            return await self.llm_inference(
                messages=messages,
                current_emotion=EmotionalState.HAPPY,
                memory_context="",
                activity_context="",
                situational_context=system_prompt,
                max_tokens_override=max_tokens,
            )

    async def kira_deep_response(self, request: str, scene_context: str = "", memory_context: str = "", recent_history: list = None, max_tokens: int = 400) -> str:
        """Generates a deep, in-character Kira response using Claude Opus.
        Used for the invite button, reflective questions, and moments where
        intelligence and nuance matter more than latency."""
        recent_history = recent_history or []
        system_prompt = (
            self.system_prompt
            + "\n\n[MODE: Deep Response \u2014 take your time, think carefully, be insightful and in-character.]"
        )
        if scene_context:
            system_prompt += f"\n\n[CURRENT SCENE]\n{scene_context}"
        if memory_context:
            system_prompt += f"\n\n[RELEVANT MEMORIES \u2014 use to stay consistent, do not quote]\n{memory_context}"

        history_to_send = []
        for turn in recent_history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                history_to_send.append({"role": turn["role"], "content": turn["content"]})
        history_to_send.append({"role": "user", "content": request})

        return await self.claude_inference(
            messages=history_to_send,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    async def claude_chat_inference(self, messages: list, system_prompt: str, max_tokens: int = 400) -> str:
        """Routes a conversational response through Claude Sonnet 4.6. Default voice/Twitch
        response path when Claude is available. Cheaper than Opus, better than local Llama."""
        if not self.anthropic_client or not ENABLE_CLAUDE_CHAT:
            return ""  # Caller falls back to local

        claude_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not claude_messages or claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "(continue)"})

        if ENABLE_PROMPT_CACHING:
            system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            system_param = system_prompt

        try:
            response = await self.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=max_tokens,
                system=system_param,
                messages=claude_messages,
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return ""
        except Exception as e:
            print(f"   [Brain] Sonnet call failed: {e}. Falling back to local Llama.")
            return ""

    async def claude_chat_inference_stream(self, messages: list, system_prompt: str, max_tokens: int = 400):
        """Async generator: streams Claude Sonnet 4.6 response chunk-by-chunk.
        Yields text deltas as they arrive. Empty generator if Claude unavailable."""
        if not self.anthropic_client or not ENABLE_CLAUDE_CHAT:
            return

        claude_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        if not claude_messages or claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "(continue)"})

        if ENABLE_PROMPT_CACHING:
            system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            system_param = system_prompt

        try:
            async with self.anthropic_client.messages.stream(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=max_tokens,
                system=system_param,
                messages=claude_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            print(f"   [Brain] Sonnet stream failed: {e}")
            return

    async def analyze_emotion_of_turn(self, last_user_text: str, last_ai_response: str) -> EmotionalState | None:
        if not self.llm: return None
        emotion_names = [e.name for e in EmotionalState]
        prompt = (f"Jonny: \"{last_user_text}\"\nKira: \"{last_ai_response}\"\n\n"
                  f"Based on this, which emotional state is most appropriate for Kira's next turn? "
                  f"Options: {', '.join(emotion_names)}.\n"
                  f"Respond ONLY with the single best state name (e.g., 'SASSY').")
        try:
            def _guarded_emotion():
                with self.inference_lock:
                    return self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0.2,
                        stop=["\n", ".", ","]
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

    async def speak_text(self, text: str):
        """Generates and plays audio for the given text (blocking). Sets is_speaking
        around the call so VAD ignores self-hearing. Used by non-streaming callers."""
        if not text:
            return
        self.interruption_event.clear()
        self.is_speaking = True
        try:
            await self._speak_single(text)
        finally:
            self.last_speech_finish_time = time.time()
            self.is_speaking = False

    async def speak_text_vn(self, text: str, ssml_inner: str | None = None):
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
            if ssml_inner is not None and TTS_ENGINE == "azure" and self.azure_synthesizer:
                ssml = (
                    f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                    f'xml:lang="en-US">'
                    f'<voice name="{AZURE_SPEECH_VOICE}">'
                    f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">'
                    f'{ssml_inner}'
                    f'</prosody></voice></speak>'
                )
                print(f"   [TTS] Speaking: {text[:50]}...")
                try:
                    result = await asyncio.to_thread(
                        self.azure_synthesizer.speak_ssml, ssml
                    )
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        await self._play_audio_with_pygame(result.audio_data)
                    else:
                        print(f"   TTS Fail: {result.cancellation_details.error_details}")
                except Exception as e:
                    print(f"   TTS/Playback Error: {e}")
            else:
                await self._speak_single(text)
        finally:
            self.last_speech_finish_time = time.time()
            self.is_speaking = False

    async def _speak_single(self, text: str):
        """Pure synthesis-and-playback for a single text block. Does NOT manage
        is_speaking or interruption_event — caller is responsible. Used by both
        speak_text (single shot) and speak_streaming (per-sentence dispatch)."""
        if not text:
            return
        if self.interruption_event.is_set():
            return

        print(f"   [TTS] Speaking: {text[:50]}...")
        audio_data = None

        try:
            if TTS_ENGINE == "azure" and self.azure_synthesizer:
                ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                        f'<voice name="{AZURE_SPEECH_VOICE}">'
                        f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">{text}</prosody>'
                        f'</voice></speak>')
                result = await asyncio.to_thread(self.azure_synthesizer.speak_ssml, ssml)
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_data = result.audio_data
                else:
                    print(f"   TTS Fail: {result.cancellation_details.error_details}")
            elif TTS_ENGINE == "edge" and Communicate:
                voice = AZURE_SPEECH_VOICE if AZURE_SPEECH_VOICE else "en-US-AriaNeural"
                communicate = Communicate(text, voice)
                buffer = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer += chunk["data"]
                audio_data = buffer

            if audio_data:
                await self._play_audio_with_pygame(audio_data)
        except Exception as e:
            print(f"   TTS/Playback Error: {e}")

    async def speak_streaming(self, stream_generator) -> str:
        """Consumes an async generator of text chunks. Buffers into sentences,
        dispatches each to TTS sequentially. Tool tags ([POLL: ...], [SONG: ...],
        [PREDICT: ...], [BIT: ...], [TAKE]) are extracted and processed separately —
        never spoken aloud.

        Sentences are split on terminal punctuation [.!?] followed by whitespace,
        but ONLY when no unclosed bracket tag is in flight.

        If the user interrupts (via VAD), the stream is abandoned."""

        self.interruption_event.clear()
        self.is_speaking = True

        full_text = ""
        buffer = ""

        def has_unclosed_tag(text: str) -> bool:
            """True if there is an unmatched '[' anywhere in the buffer."""
            last_open = text.rfind('[')
            if last_open == -1:
                return False
            return ']' not in text[last_open:]

        try:
            async for chunk in stream_generator:
                if self.interruption_event.is_set():
                    break

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
                            await self._speak_single(cleaned)
                            if self.interruption_event.is_set():
                                return full_text

            # Stream done — flush any trailing buffer
            if buffer.strip() and not self.interruption_event.is_set():
                spoken = self._strip_tags_for_speech(buffer.strip())
                cleaned = self._clean_llm_response(spoken)
                if cleaned:
                    await self._speak_single(cleaned)
        except Exception as e:
            print(f"   [Streaming] Error during stream consumption: {e}")
        finally:
            self.last_speech_finish_time = time.time()
            self.is_speaking = False

        return full_text

    def _strip_tags_for_speech(self, text: str) -> str:
        """Removes tool tags ([POLL: ...], [SONG: ...], [PREDICT: ...], [BIT: ...], [TAKE])
        from text before TTS, so they are never spoken aloud. Tags will still be
        parsed and executed by parse_kira_tools on the full assembled response."""
        # Multi-pipe tags: POLL, PREDICT
        text = re.sub(r'\[POLL:[^\]]*\]', '', text)
        text = re.sub(r'\[PREDICT:[^\]]*\]', '', text)
        # Single-value tags
        text = re.sub(r'\[SONG:[^\]]*\]', '', text)
        text = re.sub(r'\[BIT:[^\]]*\]', '', text)
        # Boolean tags
        text = re.sub(r'\[TAKE\]', '', text)
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

    async def transcribe_audio(self, audio_data: bytes) -> str:
        # Using numpy array directly for speed
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        def _run_transcribe():
            # ADJUSTING VAD SETTINGS HERE:
            # threshold=0.85: UP FROM 0.6: Requires 85% certainty it's a voice.
            # min_speech_duration_ms=400: UP FROM 300: Ignores quick laughs/gasps.
            # min_silence_duration_ms=800: DOWN FROM 1000: Responds slightly faster.
            segments, info = self.whisper.transcribe(
                arr, 
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.85, 
                    min_speech_duration_ms=400, 
                    min_silence_duration_ms=800,
                    speech_pad_ms=200
                )
            )
            return list(segments)

        segments = await asyncio.to_thread(_run_transcribe)
        text = "".join([segment.text for segment in segments])
        return text.strip()
