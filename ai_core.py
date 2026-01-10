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
import llama_cpp # Needed for Q4_0 constants
from llama_cpp.llama_chat_format import Llava15ChatHandler
from faster_whisper import WhisperModel
from llama_cpp import Llama
from transformers import pipeline

from config import (
    LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, WHISPER_MODEL_SIZE, TTS_ENGINE,
    LLM_MAX_RESPONSE_TOKENS, PROMPT_BUDGET_TOKENS, VISION_ENABLED, # Added Budget
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    VIRTUAL_AUDIO_DEVICE, AI_NAME
)
from persona import AI_PERSONALITY_PROMPT, EmotionalState
from utils_logger import logger, sanitize_messages

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
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()

    async def initialize(self):
        """Initializes AI components sequentially to prevent resource conflicts."""
        logger.info("-> Initializing AI Core components...")
        
        # Initialize Audio Mixer permanently
        if pygame.mixer.get_init(): pygame.mixer.quit()
        
        # FORCE Default Desktop Audio (User Request)
        logger.info("   Forcing Audio Output to Default Windows Device (devicename=None)...")
        try:
            pygame.mixer.init(devicename=None)
        except Exception as e:
            logger.warning(f"   Warning: Default init failed, trying auto: {e}")
            pygame.mixer.init()

        try:
            await asyncio.to_thread(self._init_llm)
            await asyncio.to_thread(self._init_whisper)
            await self._init_tts()

            self.is_initialized = True
            logger.info("   AI Core initialized successfully!")
        except Exception as e:
            logger.critical(f"FATAL: AI Core failed to initialize: {e}")
            self.is_initialized = False
            raise

    async def test_audio_output(self):
        """Plays a test tone to verify audio output."""
        logger.info("-> Testing Audio Output...")
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
            logger.info("   Audio test passed (Beep played).")
        except Exception as e:
            logger.error(f"   Audio test FAILED: {e}")

    def _init_llm(self):
        logger.info(f"-> Loading LLM model... (GPU Layers: {N_GPU_LAYERS})")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        # Optimized for RTX 5080: n_threads=8, Flash Attention enabled, Force GPU Layers=-1
        # Check for Vision Projector
        chat_handler = None
        mmproj_path = "models/gemma-3-12b-it-mmproj-f16.gguf"
        
        if VISION_ENABLED and os.path.exists(mmproj_path):
            logger.info(f"   Vision Projector found: {mmproj_path}")
            try:
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            except Exception as e:
                logger.error(f"   Failed to load Vision Projector: {e}")
        elif not VISION_ENABLED:
             logger.info("   Vision disabled in config. Skipping MMA Projector load.")
        else:
            logger.warning("   Vision Projector NOT found. Vision capabilities disabled.")

        # Updated for Gemma 3 12B (RTX 5080 Maximum Optimization)
        # model_path="models/google_gemma-3-12b-it-Q5_K_M.gguf"
        self.llm = Llama(
            model_path="models/google_gemma-3-12b-it-Q5_K_M.gguf", 
            chat_handler=chat_handler,
            n_gpu_layers=-1,       # Force all 48 layers to GPU
            n_ctx=2048,            # The "Sweet Spot" - bigger than 2048, but safer than 4096
            n_batch=128,           # LOWER THIS. This reduces the "Work Area" VRAM significantly
            n_ubatch=128,          # Match n_batch
            n_ctx_keep=200,        # Ensure her core identity/system prompt never gets deleted
            context_erase=0.5,     # When she hits 3072, she'll forget the oldest 1500
            flash_attn=True,
            offload_kqv=True,      # Keep attention math on the chip
            use_mmap=True,
            use_mlock=True,        # Forces Windows to keep this in memory
            n_threads=8,
            verbose=False,
            logos_processor=None     # Ensure no weird logging injection
        )
        logger.info(f"   LLM loaded (Gemma 3). (Ctx: 2048 | Batch: 128 | Context Shift: ON | Keep: 200)")

    def _init_whisper(self):
        logger.info("-> Loading Faster-Whisper STT model...")
        if torch.cuda.is_available():
            logger.info(f"   CUDA Detected: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            logger.warning("   WARNING: CUDA NOT DETECTED! Whisper will run on CPU (Slow).")
            device = "cpu"

        # Upgrade to medium.en for better accuracy
        # We use float16 because your 5080 handles it like a champ.
        logger.info(f"   Whisper Config: Model=medium.en | Device={device} | ComputeType=float16")
        self.whisper = WhisperModel("medium.en", device=device, compute_type="float16")
        logger.info("   Faster-Whisper STT model loaded.")

    async def _init_tts(self):
        logger.info(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == "elevenlabs":
            if not AsyncElevenLabs: raise ImportError("Run 'pip install elevenlabs'")
            self.eleven_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        elif TTS_ENGINE == "azure":
            if not speechsdk: raise ImportError("Run 'pip install azure-cognitiveservices-speech'")
            
            # Validate Azure Config
            sanitized_key = f"{AZURE_SPEECH_KEY[:4]}****" if AZURE_SPEECH_KEY else "None"
            logger.info(f"   Azure Config: Region=[{AZURE_SPEECH_REGION}], Key=[{sanitized_key}]")
            if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
                logger.warning("   WARNING: Azure Key or Region is missing! Check .env")

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
        logger.info(f"   {TTS_ENGINE.capitalize()} TTS ready.")

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "", image_buffer: str = None) -> str:
        t_start = time.perf_counter()
        
        # Use a clean list for Gemma's specific format
        formatted_messages = []
        
        # 1. Add System Instruction ONLY ONCE at the start
        system_text = AI_PERSONALITY_PROMPT
        system_text += f"\n\n[Your current emotional state is: {current_emotion.name}. Let this state subtly influence your response style and word choice.]"
        
        # Inject Memory Context explicitly as a SYSTEM note before the recent history
        if memory_context and "No memories" not in memory_context:
             system_text += f"\n\n[Context Memory about Jonny]: {memory_context}"
        
        formatted_messages.append({"role": "system", "content": system_text})
        
        # --- TOKEN BUDGETING ---
        # Heuristic: 1 token ~= 3 chars. 
        # Reserve for System + Response
        sys_tokens = (len(system_text) // 3) + 20
        reserve_tokens = LLM_MAX_RESPONSE_TOKENS + 50
        image_tokens = 256 if image_buffer else 0 # Image tokens check
        
        available_history = PROMPT_BUDGET_TOKENS - sys_tokens - reserve_tokens - image_tokens
        if available_history < 100: available_history = 100 # Minimum floor
        
        history_to_add = []
        current_cost = 0
        
        # Always take the last message (User Trigger)
        if messages:
             last_msg = messages[-1]
             history_to_add.append(last_msg)
             # Add cost
             content_str = last_msg["content"]
             if isinstance(content_str, list):
                  content_str = next((item["text"] for item in content_str if item["type"] == "text"), "")
             current_cost += (len(content_str) // 3) + 5
        
        # Fill rest from newest to oldest
        if len(messages) > 1:
            for msg in reversed(messages[:-1]):
                content_str = msg["content"]
                if isinstance(content_str, list):
                     content_str = next((item["text"] for item in content_str if item["type"] == "text"), "")
                
                cost = (len(content_str) // 3) + 5
                if current_cost + cost <= available_history:
                    history_to_add.insert(0, msg) # Prepend
                    current_cost += cost
                else:
                    break # Budget full
        
        # Normalize and Add to formatted_messages
        for msg in history_to_add:
             if isinstance(msg["content"], list):
                  text_only = next((item["text"] for item in msg["content"] if item["type"] == "text"), "")
                  formatted_messages.append({"role": msg["role"], "content": text_only})
             else:
                  formatted_messages.append(msg)

        # 3. Process image for the CURRENT turn only
        has_image = False
        if image_buffer:
            if formatted_messages and formatted_messages[-1]["role"] == "user":
                 last_content = formatted_messages[-1]["content"]
                 if isinstance(last_content, list):
                      last_text = next((item["text"] for item in last_content if item["type"] == "text"), "")
                 else:
                      last_text = last_content
                 
                 # Correct Vision Message Format
                 formatted_messages[-1]["content"] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_buffer}"}},
                    {"type": "text", "text": last_text}
                ]
                 has_image = True
        
        t_build_end = time.perf_counter()
        
        # --- ASSERTIONS & DIAGNOSTICS ---
        assert formatted_messages[0]["role"] == "system", "First message must be system"
        # Validate role alternation (ignoring initial systems)
        dialogue_msgs = [m for m in formatted_messages if m["role"] != "system"]
        if dialogue_msgs:
            assert dialogue_msgs[-1]["role"] == "user", "Last message must be user"
            
        # Safe Log: Summary only
        mem_count = 1 if (memory_context and "No memories" not in memory_context) else 0
        logger.info(f"   [Prompt] Msgs={len(formatted_messages)} | EstTokens={current_cost+sys_tokens} | MemoryItems={mem_count} | HasImage={has_image}")
        
        # 4. Generate with strict constraints
        max_response_tokens = LLM_MAX_RESPONSE_TOKENS # Use config value
        
        t_infer_start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=max_response_tokens, 
            stop=["USER:", "Jonny says:", "<start_of_turn>", "You are Kira"], 
            logits_processor=None 
        )
        t_infer_end = time.perf_counter()
        
        t_total = t_infer_end - t_start
        t_build = t_build_end - t_start
        t_infer = t_infer_end - t_infer_start
        generated_text = response["choices"][0]["message"]["content"]
        gen_toks = len(generated_text) // 4 # Rough estimate
        tps = gen_toks / t_infer if t_infer > 0 else 0
        
        logger.info(f"   [Perf] Total={t_total:.3f}s | Build={t_build:.3f}s | Infer={t_infer:.3f}s | TPS={tps:.1f}")
        
        return generated_text

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
            response = await asyncio.to_thread(
                self.llm, prompt=prompt, max_tokens=10, temperature=0.2, stop=["\n", ".", ","]
            )
            text_response = response['choices'][0]['text'].strip().upper()
            for emotion in EmotionalState:
                if emotion.name in text_response:
                    return emotion
            return None
        except Exception as e:
            logger.error(f"   ERROR during emotion analysis: {e}")
            return None

    async def speak_text(self, text: str):
        """Generates and plays audio for the given text (Blocking)."""
        if not text: return
        
        self.is_speaking = True # Mute ears
        logger.info(f"   [TTS] Speaking: {text[:50]}...")
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
                     logger.error(f"   TTS Fail: {result.cancellation_details.error_details}")

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
            logger.error(f"   TTS/Playback Error: {e}")
        finally:
            self.is_speaking = False

    async def _play_audio_with_pygame(self, audio_bytes: bytes):
        if self.interruption_event.is_set() or not audio_bytes:
            logger.info("   [Audio] Skipped: Interrupted or Empty.")
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
            logger.error(f"   Audio Playback Error: {e}")


    def _clean_llm_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.replace('</s>', '').strip()
        text = text.replace('*', '')
        return text

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