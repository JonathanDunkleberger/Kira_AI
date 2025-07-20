# ai_core.py - Core logic for the AI, including STT, LLM, and TTS.

import asyncio
import io
import os
import re
import pygame
import numpy as np
from llama_cpp import Llama
from transformers import pipeline

from config import (
    LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, WHISPER_MODEL_SIZE, TTS_ENGINE,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    AZURE_SPEECH_VOICE, AZURE_PROSODY_PITCH, AZURE_PROSODY_RATE,
    VIRTUAL_AUDIO_DEVICE, AI_NAME, EmotionalState
)
from persona import AI_PERSONALITY_PROMPT

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
        try:
            # Initialize components one by one to prevent console conflicts on Windows
            await asyncio.to_thread(self._init_llm)
            await asyncio.to_thread(self._init_whisper)
            await self._init_tts()

            self.is_initialized = True
            print("   AI Core initialized successfully!")
        except Exception as e:
            print(f"FATAL: AI Core failed to initialize: {e}")
            self.is_initialized = False
            raise

    def _init_llm(self):
        print("-> Loading LLM model...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        self.llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
        print("   LLM model loaded.")

    def _init_whisper(self):
        print("-> Loading Whisper STT model...")
        device = "cpu"
        print(f"   Whisper STT will use device: {device}")
        self.whisper = pipeline("automatic-speech-recognition", model=f"openai/whisper-{WHISPER_MODEL_SIZE}", device=device)
        print("   Whisper STT model loaded.")

    async def _init_tts(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == "elevenlabs":
            if not AsyncElevenLabs: raise ImportError("Run 'pip install elevenlabs'")
            self.eleven_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        elif TTS_ENGINE == "azure":
            if not speechsdk: raise ImportError("Run 'pip install azure-cognitiveservices-speech'")
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            # FIX: By passing `audio_config=None`, the SDK defaults to an in-memory stream,
            # which is what we need to get the audio bytes for Pygame.
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
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

        full_prompt = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=full_prompt, max_tokens=150, temperature=0.8, top_p=0.9,
                stop=["\nJonny:", "\nKira:", "</s>"]
            )
            raw_text = response['choices'][0]['message']['content']
            return self._clean_llm_response(raw_text)
        except Exception as e:
            print(f"   ERROR during LLM inference: {e}")
            return "Oops, my brain just short-circuited. What were we talking about?"

    async def speak_text(self, text: str):
        if not text: return
        print(f"<<< {AI_NAME} says: {text}")
        self.interruption_event.clear()
        audio_bytes = b''
        try:
            if TTS_ENGINE == "elevenlabs":
                audio_stream = await self.eleven_client.text_to_speech.stream(text=text, voice_id=ELEVENLABS_VOICE_ID)
                async for chunk in audio_stream:
                    if self.interruption_event.is_set(): return
                    audio_bytes += chunk
            elif TTS_ENGINE == "azure":
                ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                        f'<voice name="{AZURE_SPEECH_VOICE}">'
                        f'<prosody rate="{AZURE_PROSODY_RATE}" pitch="{AZURE_PROSODY_PITCH}">{text}</prosody>'
                        f'</voice></speak>')
                
                # FIX: Correctly await the async synthesis call
                result = await asyncio.to_thread(self.azure_synthesizer.speak_ssml, ssml)

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_bytes = result.audio_data
                else:
                    cancellation_details = result.cancellation_details
                    raise Exception(f"Azure TTS failed: {cancellation_details.reason} - {cancellation_details.error_details}")
            elif TTS_ENGINE == "edge":
                # Note: Edge TTS doesn't support pitch control, only rate.
                comm = Communicate(text, AZURE_SPEECH_VOICE, rate=AZURE_PROSODY_RATE)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        if self.interruption_event.is_set(): return
                        audio_bytes += chunk["data"]
            
            if not self.interruption_event.is_set():
                await self._play_audio_with_pygame(audio_bytes)
        except Exception as e:
            print(f"   ERROR during TTS generation: {e}")

    async def _play_audio_with_pygame(self, audio_bytes: bytes):
        if self.interruption_event.is_set() or not audio_bytes: return
        try:
            pygame.mixer.init(devicename=VIRTUAL_AUDIO_DEVICE)
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            channel = sound.play()
            while channel.get_busy():
                if self.interruption_event.is_set():
                    channel.stop()
                    break
                await asyncio.sleep(0.1)
        finally:
            if pygame.mixer.get_init(): pygame.mixer.quit()

    def _clean_llm_response(self, text: str) -> str:
        # Removes role tags, empty lines, and other artifacts
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*<\|.*?\|>\s*$', '', text, flags=re.MULTILINE) # Remove empty system tags
        text = text.replace('</s>', '').strip()
        return text

    async def transcribe_audio(self, audio_data: bytes) -> str:
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(self.whisper, arr)
        return result.get("text", "").strip()