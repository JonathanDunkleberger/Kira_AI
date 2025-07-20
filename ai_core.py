# ai_core.py
import asyncio
import torch
import re
import io
import os
import pygame
import numpy as np
from llama_cpp import Llama
from transformers import pipeline

from config import (
    LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS,
    WHISPER_MODEL_SIZE, TTS_ENGINE,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    VIRTUAL_AUDIO_DEVICE
)
from persona import AI_PERSONALITY_PROMPT

# Optional imports for TTS engines
try: from edge_tts import Communicate
except ImportError: Communicate = None
try: from elevenlabs.client import AsyncElevenLabs
except ImportError: AsyncElevenLabs = None

class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        self.is_initialized = False
        self.llm = None
        self.whisper = None
        self.eleven = None

    # --- THIS IS THE REVISED FUNCTION ---
    async def initialize(self):
        """Initializes AI components sequentially to prevent resource conflicts."""
        print("-> Initializing AI Core components...")
        
        # Pre-initialize pygame mixer once to avoid issues in async loops
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()

        # Load components one by one
        await asyncio.to_thread(self._init_llm)
        await asyncio.to_thread(self._init_whisper)
        await self._init_tts()
        
        self.is_initialized = True
        print("   AI Core initialized successfully!")

    def _init_llm(self):
        print("-> Loading LLM model...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        self.llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)

    def _init_whisper(self):
        print("-> Loading Whisper STT model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.whisper = pipeline("automatic-speech-recognition", model=f"openai/whisper-{WHISPER_MODEL_SIZE}", device=device)
        print("   Whisper STT model loaded.")

    async def _init_tts(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == "elevenlabs" and AsyncElevenLabs:
            self.eleven = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        elif not (TTS_ENGINE == "edge" and Communicate):
             raise ValueError(f"Unsupported/uninstalled TTS_ENGINE: {TTS_ENGINE}")
        print(f"   {TTS_ENGINE.capitalize()} TTS ready.")


    async def _play_audio_with_pygame(self, audio_bytes: bytes):
        """Plays audio bytes to the specified virtual audio device using pygame."""
        if self.interruption_event.is_set() or not audio_bytes:
            return

        try:
            pygame.mixer.init(devicename=VIRTUAL_AUDIO_DEVICE)
            audio_sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            channel = audio_sound.play()
            while channel.get_busy():
                if self.interruption_event.is_set():
                    channel.stop()
                    break
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error during audio playback: {e}")
        finally:
            if pygame.mixer.get_init():
                pygame.mixer.quit()

    async def speak_text(self, text: str):
        if not text: return
        self.interruption_event.clear()
        audio_bytes = b''
        try:
            if TTS_ENGINE == "elevenlabs":
                audio_stream = await self.eleven.text_to_speech.stream(text=text, voice_id=ELEVENLABS_VOICE_ID)
                async for chunk in audio_stream:
                    if self.interruption_event.is_set(): return
                    audio_bytes += chunk
            else: # Default to Edge TTS
                comm = Communicate(text, "en-US-AriaNeural", rate="+15%")
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                    if self.interruption_event.is_set(): return
            
            await self._play_audio_with_pygame(audio_bytes)
        except Exception as e:
            print(f"ERROR during TTS generation: {e}")

    # --- The rest of your ai_core.py file remains the same ---
    def clean_llm_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = text.replace('---', '').replace('*laughs*', 'Haha,').replace('*giggles*', 'Hehe,')
        return text.strip()

    async def transcribe_audio(self, audio_data: bytes) -> str:
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(self.whisper, arr)
        return result.get("text", "").strip()

    async def llm_inference(self, messages: list, memory_context: str = "") -> str:
        system_messages = [{"role": "system", "content": AI_PERSONALITY_PROMPT}]
        if memory_context and "No memories" not in memory_context:
            system_messages.append({"role": "system", "content": memory_context})
        full_prompt = system_messages + messages
        response = await asyncio.to_thread(
            self.llm.create_chat_completion,
            messages=full_prompt, max_tokens=128, temperature=0.9, top_p=0.9,
        )
        raw_text = response['choices'][0]['message']['content']
        return self.clean_llm_response(raw_text)