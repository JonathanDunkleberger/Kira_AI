# ai_core.py - Core logic for the AI, including STT, LLM, and TTS.

import asyncio
import io
import os
import re
import pygame
import torch
import numpy as np
import whisper  # Native OpenAI Whisper import
from llama_cpp import Llama
from transformers import pipeline

from config import (
    LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, WHISPER_MODEL_SIZE, TTS_ENGINE,
    LLM_MAX_RESPONSE_TOKENS,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    VIRTUAL_AUDIO_DEVICE, AI_NAME
)
from persona import AI_PERSONALITY_PROMPT, EmotionalState

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
        self.llm = None
        self.whisper = None
        self.eleven_client = None
        self.azure_synthesizer = None
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
        self.llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
        print("   LLM model loaded.")

    def _init_whisper(self):
        print("-> Loading Whisper STT model...")
        # FORCE CUDA for RTX 5080 (Using Native OpenAI Whisper)
        if torch.cuda.is_available():
            print(f"   CUDA Detected: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("   WARNING: CUDA NOT DETECTED! Whisper will run on CPU (Slow).")
            device = "cpu"
            
        print(f"   Whisper Config: Model={WHISPER_MODEL_SIZE} | Device={device} | FP16=True")
        
        # Native OpenAI Whisper Load
        self.whisper = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        print("   Whisper STT model loaded.")

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
            
            # Prevent Azure from auto-playing sound so Pygame can handle it exclusively
            # We do this by routing Azure output to a Null stream so it doesn't seize the audio device.
            stream = speechsdk.audio.PushAudioOutputStream(speechsdk.audio.AudioStreamFormat(samples_per_second=24000, bits_per_sample=16, channels=1))
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        elif TTS_ENGINE == "edge":
            if not Communicate: raise ImportError("Run 'pip install edge-tts'")
        else:
            raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")
        print(f"   {TTS_ENGINE.capitalize()} TTS ready.")

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "") -> str:
        system_prompt = AI_PERSONALITY_PROMPT
        system_prompt += f"\n\n[Your current emotional state is: {current_emotion.name}. Let this state subtly influence your response style and word choice.]"
        if memory_context and "No memories" not in memory_context:
            system_prompt += f"\n[Memory Context]:\n{memory_context}"

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
        
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=full_prompt,
                # --- UPDATED: Use the new variable for max_tokens ---
                max_tokens=LLM_MAX_RESPONSE_TOKENS,
                temperature=0.8,
                top_p=0.9,
                stop=["\nJonny:", "\nKira:", "</s>"]
            )
            raw_text = response['choices'][0]['message']['content']
            return self._clean_llm_response(raw_text)
        except Exception as e:
            print(f"   ERROR during LLM inference: {e}")
            return "Oops, my brain just short-circuited. What were we talking about?"

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
            print(f"   ERROR during emotion analysis: {e}")
            return None

    async def speak_text(self, text: str):
        if not text: return
        print(f"<<< {AI_NAME} says: {text}")
        self.interruption_event.clear()
        audio_bytes = b''
        try:
            if TTS_ENGINE == "elevenlabs":
                # ... elevenlabs logic ...
                pass
            
            elif TTS_ENGINE == "azure":
                try:
                    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                            f'<voice name="{AZURE_SPEECH_VOICE}">'
                            f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">{text}</prosody>'
                            f'</voice></speak>')
                    
                    # Ensure we pull the stream instead of letting Azure play it
                    # This prevents the SDK from seizing the default audio device
                    result = await asyncio.to_thread(self.azure_synthesizer.speak_ssml, ssml)
                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        print(f"   [Azure] Synthesis Success. Audio Duration: {result.audio_duration}")
                        audio_bytes = result.audio_data
                    else:
                        cancellation_details = result.cancellation_details
                        error_type = cancellation_details.reason
                        error_details = cancellation_details.error_details
                        
                        # 401 Auth Check
                        if "401" in str(error_details) or "Authentication" in str(error_details):
                            print(f"\n   !!! AZURE AUTH FAILED: Check Key/Region in .env !!!\n   Region: {AZURE_SPEECH_REGION}\n")
                        
                        raise Exception(f"Azure TTS Failed: {error_type} - {error_details}")
                        
                except Exception as e:
                    if "401" in str(e):
                        print("\n   !!! AZURE AUTH 401: Authentication Failed !!!\n")
                    raise e

            elif TTS_ENGINE == "edge":
                # ... edge logic ...
                pass
            
            if not self.interruption_event.is_set():
                # Zero Latency Playback (Removed Wait)
                await self._play_audio_with_pygame(audio_bytes)
        except Exception as e:
            print(f"   ERROR during TTS generation: {e}")

    async def _play_audio_with_pygame(self, audio_bytes: bytes):
        if self.interruption_event.is_set() or not audio_bytes:
            print("   [Audio] Skipped: Interrupted or Empty.")
            return
        try:
            if not pygame.mixer.get_init(): pygame.mixer.init(devicename=None)
            
            # Stop any currently playing audio
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            pygame.mixer.music.stop()
            try: pygame.mixer.music.unload() 
            except: pass

            print(f"   [Audio] Playing {len(audio_bytes)} bytes...")
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.set_volume(1.0) # Ensure max volume
            channel = sound.play()
            
            while channel.get_busy():
                if self.interruption_event.is_set():
                    channel.stop(); break
                await asyncio.sleep(0.1)
            print("   [Audio] Playback finished.")
        except Exception as e:
            print(f"   Audio Playback Error: {e}")
        finally:
             # Clean up
            try: pygame.mixer.music.unload() 
            except: pass

    def _clean_llm_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.replace('</s>', '').strip()
        text = text.replace('*', '')
        return text

    async def transcribe_audio(self, audio_data: bytes) -> str:
        # Save temporary file for native Whisper (it prefers file paths or direct arrays)
        # Using numpy array directly for speed
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Native Whisper Transcribe with fp16=True (RTX 5080 Optimization)
        result = await asyncio.to_thread(self.whisper.transcribe, arr, fp16=True)
        return result.get("text", "").strip()