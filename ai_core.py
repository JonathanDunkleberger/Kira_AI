# ai_core.py - Core logic for the AI, including STT, LLM, and TTS.

import asyncio
import torch
import numpy as np
from llama_cpp import Llama
from transformers import pipeline
import warnings
import sys
import os
import functools
import io
import traceback
import tempfile
import sounddevice as sd
import soundfile as sf

# TTS Imports
try:
    from edge_tts import Communicate
except ImportError:
    Communicate = None
try:
    from elevenlabs.client import AsyncElevenLabs
except ImportError:
    AsyncElevenLabs = None

# Config and Persona Imports
from config import (
    LLM_MODEL_PATH, N_GPU_LAYERS, N_CTX, WHISPER_MODEL_SIZE,
    TTS_ENGINE, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
)
from persona import AI_PERSONALITY_PROMPT

warnings.simplefilter("ignore", FutureWarning)

class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        self.is_initialized = False
        self.llm = None
        self.whisper_pipeline = None
        self.elevenlabs_client = None
        self.last_bot_response = None

    async def initialize(self):
        print("Initializing AI Core components...")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._initialize_llm)
            await loop.run_in_executor(None, self._initialize_whisper)
            await self._initialize_tts()
            self.is_initialized = True
            print("\nAI Core initialized successfully!")
        except Exception as e:
            print(f"ERROR during AI Core initialization: {e}")
            self.is_initialized = False
            raise

    def _initialize_llm(self):
        print("-> Loading LLM model...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        self.llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=True)
        print("   LLM loaded.")

    def _initialize_whisper(self):
        print("-> Loading Whisper STT model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"   Whisper STT will use device: {device}")
        model_name = f"openai/whisper-{WHISPER_MODEL_SIZE}"
        self.whisper_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=device)
        print("   Whisper STT model loaded.")

    async def _initialize_tts(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == 'elevenlabs':
            if not all([AsyncElevenLabs, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID]) or "YOUR_API_KEY" in ELEVENLABS_API_KEY:
                raise ValueError("ElevenLabs configuration is missing, placeholder values are present, or library not installed.")
            self.elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("   ElevenLabs client initialized.")
        elif TTS_ENGINE == 'edge':
            if Communicate is None: raise ImportError("edge-tts package not installed.")
            print("   Edge TTS ready.")
        else:
            raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")

    async def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        print("   Transcribing audio...")
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            pipeline_input = {"array": audio_np, "sampling_rate": sample_rate}
            whisper_call = functools.partial(self.whisper_pipeline, pipeline_input, generate_kwargs={"language": "en"})
            result = await loop.run_in_executor(None, whisper_call)
            text = result.get("text", "").strip()
            print(f"   You said: '{text}'")
            return text
        except Exception as e:
            print(f"   Error transcribing: {e}")
            traceback.print_exc()
            return ""

    async def llm_inference(self, conversation_history: list, user_input: str) -> str:
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{AI_PERSONALITY_PROMPT}<|eot_id|>"
        for turn in conversation_history:
            role, content = turn['role'], turn['content']
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        try:
            loop = asyncio.get_running_loop()
            llm_call = functools.partial(self.llm, prompt=prompt, max_tokens=200, stop=["<|eot_id|>"], temperature=0.7, echo=False)
            response = await loop.run_in_executor(None, llm_call)
            cleaned_text = response['choices'][0]['text'].strip()
            if not cleaned_text:
                print("   Warning: LLM returned an empty string.")
                return "I... zoned out for a sec. What were we talking about?"
            print(f"   LLM Response: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            print(f"   LLM error: {e}")
            traceback.print_exc()
            return "Oops, my brain just short-circuited! Can you repeat that?"

    async def _interruptible_playback(self, data, samplerate):
        """Plays audio data on the default output device and allows for interruption."""
        loop = asyncio.get_running_loop()
        
        # Use the default output device by not specifying a device index.
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=data.ndim if data.ndim > 1 else 1,
            dtype=data.dtype
        )
        stream.start()
        print("   Playing audio on default device...")
        try:
            chunk_size = 1024
            for i in range(0, len(data), chunk_size):
                if self.interruption_event.is_set():
                    print("   Playback interrupted by user.")
                    stream.stop()
                    stream.abort()
                    return
                chunk = data[i:i + chunk_size]
                await loop.run_in_executor(None, stream.write, chunk)
        finally:
            if not stream.stopped:
                stream.stop()
            if not stream.closed:
                stream.close()

    async def speak_text(self, text: str):
        print(f"   Speaking: '{text}'")
        self.interruption_event.clear()
        self.last_bot_response = text
        try:
            if TTS_ENGINE == 'elevenlabs':
                await self._speak_elevenlabs(text)
            elif TTS_ENGINE == 'edge':
                await self._speak_edge_tts(text)
        except Exception as e:
            print(f"   ERROR speaking with {TTS_ENGINE}: {e}")

    async def _speak_elevenlabs(self, text: str):
        audio_stream = await self.elevenlabs_client.text_to_speech.stream(text=text, voice_id=ELEVENLABS_VOICE_ID)
        buffer = io.BytesIO()
        async for chunk in audio_stream:
            if self.interruption_event.is_set(): return
            buffer.write(chunk)
        if buffer.getbuffer().nbytes > 0:
            buffer.seek(0)
            data, samplerate = sf.read(buffer, dtype='float32')
            await self._interruptible_playback(data, samplerate)

    async def _speak_edge_tts(self, text: str):
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = audio_file.name
        audio_file.close()
        try:
            communicate = Communicate(text, voice="en-US-AriaNeural", rate="+15%")
            await communicate.save(temp_path)
            if self.interruption_event.is_set(): return
            data, samplerate = sf.read(temp_path, dtype='float32')
            await self._interruptible_playback(data, samplerate)
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)