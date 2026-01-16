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
import llama_cpp # Needed for Q4_0 constants
from faster_whisper import WhisperModel
from llama_cpp import Llama

from config import (
    LLM_MODEL_PATH, N_CTX, N_BATCH, N_GPU_LAYERS, WHISPER_MODEL_SIZE, TTS_ENGINE,
    LLM_MAX_RESPONSE_TOKENS,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    VIRTUAL_AUDIO_DEVICE, AI_NAME
)
from persona import EmotionalState
from personality_file import KIRA_PERSONALITY
from prompt_loader import load_personality_txt
from prompt_rules import TOOL_AND_FORMAT_RULES

# Graceful SDK imports
try: from edge_tts import Communicate
except ImportError: Communicate = None
try: from elevenlabs.client import AsyncElevenLabs
except ImportError: AsyncElevenLabs = None
try: import azure.cognitiveservices.speech as speechsdk
except ImportError: speechsdk = None


class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        self.is_initialized = False
        self.is_speaking = False # Added flag for self-hearing prevention
        self.llm = None
        self.whisper = None
        self.eleven_client = None
        self.azure_synthesizer = None
        self.inference_lock = threading.Lock() # Added lock for inference safety
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

        # --- DYNAMIC INSTRUCTION APPEND ---
        self.system_prompt += (
            "\n\n[SITUATION: You are in Gaming Mode]\n"
            "You are hanging out with Jonny. Be yourself. "
            "If you see something interesting, say it. "
            "If you want to talk about something else, do that. "
            "Just be natural and avoid forced 'assistant' vibes."
        )

        self.system_prompt += (
            "\n\n[CONTEXTUAL LOCK]\n"
            "Your primary reality is the current visual input and the current conversation thread. "
            "Prioritize facts found in the current session over older facts in your long-term memory. "
            "If the screen shows a different world or interface than what you discussed 10 minutes ago, "
            "assume the situation has changed and stay locked into the new scene."
        )
        
        print("----- SYSTEM PROMPT (FIRST 400 CHARS) -----")
        print(self.system_prompt[:400])
        print("------------------------------------------")

        print(f" LLM loaded. Path: {LLM_MODEL_PATH}")

    def _init_whisper(self):
        print("-> Loading Faster-Whisper STT model...")
        if torch.cuda.is_available():
            print(f"   CUDA Detected: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("   WARNING: CUDA NOT DETECTED! Whisper will run on CPU (Slow).")
            device = "cpu"

        # Upgrade to medium.en for better accuracy
        # We use float16 because your 5080 handles it like a champ.
        print(f"   Whisper Config: Model=medium.en | Device={device} | ComputeType=float16")
        self.whisper = WhisperModel("medium.en", device=device, compute_type="float16")
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
            # Prevent Azure from auto-playing sound so Pygame can handle it exclusively
            # We do this by routing Azure output to a Null stream
            # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False) # This API varies by version
            # The most reliable way for raw data extraction is to NOT set audio_config to None (which means default speaker)
            # but to set it to a pull stream or similar.
            # However, for now, let's keep it simple: If we set audio_config=None, it PLAYS. 
            # If we want it SILENT, we probably need `audio_config=speechsdk.audio.AudioConfig(device_name="non_existent")`? No.
            # Correct solution: Use `speechsdk.audio.AudioConfig(stream=None)`? No.
            
            # Use `audio_config=None` (Default Speaker) BUT we are using pygame too.
            # Conflict: Azure holds the device handle.
            # Solution: Tell Azure to output to an internal stream we don't listen to, or just Mute it?
            # Better Solution: Use PullAudioOutputStream to "catch" the audio without playing it.
            
            # Prevent Azure from auto-playing sound by using a PullStream (Memory Stream)
            # This satisfies the "No WAV File" requirement and prevents speaker conflict.
            # We don't read from this stream, we just let Azure write to it so it doesn't block.
            self.null_stream = speechsdk.audio.PullAudioOutputStream(
                speechsdk.audio.PullAudioOutputStreamCallback() 
            ) if False else None # Callback is complex to implement in one line.
            
            # SIMPLER TRICK: Use a PushAudioOutputStream with a dummy write callback.
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

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "") -> str:
        # Use our updated system prompt if available, else fallback
        system_prompt = self.system_prompt
        system_prompt += f"\n\n[Your current emotional state is: {current_emotion.name}. Let this state subtly influence your response style and word choice.]"
        
        # INJECT MEMORY AS NOTES
        if memory_context:
            system_prompt += (
                f"\n\n[MEMORY NOTES - DO NOT QUOTE OR READ THESE ALOUD]\n"
                f"{memory_context}\n"
                f"Use these to stay consistent and personal. "
                "Do not say 'my memory says' or list them like a database."
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
                    max_tokens=LLM_MAX_RESPONSE_TOKENS,
                    temperature=0.7,
                    top_p=0.9,
                    min_p=0.1,
                    repeat_penalty=1.2,
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

    # Legacy method wrapper if needed, but we are using streaming now.
    # The brain_worker calls llm_inference directly and expects a generator.
    async def _legacy_inference(self):
        # ... kept for reference ...
        pass

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
        """Generates and plays audio for the given text (Blocking)."""
        if not text: return
        
        # --- FIX: CLEAR INTERRUPTION FLAG BEFORE STARTING ---
        self.interruption_event.clear() 
        
        self.is_speaking = True # Mute ears
        print(f"   [TTS] Speaking: {text[:50]}...")
        audio_data = None
        
        try:
            # --- AZURE TTS ---
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

            # --- EDGE TTS (Fallback) ---
            elif TTS_ENGINE == "edge" and Communicate:
                 communicate = Communicate(text, float(AZURE_PROSODY_PITCH) if False else "en-US-AriaNeural")
                 buffer = b""
                 async for chunk in communicate.stream():
                     if chunk["type"] == "audio":
                         buffer += chunk["data"]
                 audio_data = buffer

            # Play Audio
            if audio_data:
                await self._play_audio_with_pygame(audio_data)

        except Exception as e:
            print(f"   TTS/Playback Error: {e}")
        finally:
            self.is_speaking = False

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
        text = re.sub(r'\[.*?\]', '', text) 
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
