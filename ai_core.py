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
        
        print("   Warming up the LLM... (this may take a moment)")
        warmup_prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        self.llm(warmup_prompt, max_tokens=2, stop=["<|eot_id|>"])
        print("   LLM is warm and ready!")

    def _initialize_whisper(self):
        print("-> Loading Whisper STT model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"   Whisper STT will use device: {device}")
        model_name = f"openai/whisper-{WHISPER_MODEL_SIZE}"
        self.whisper_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=device)
        print("   Whisper STT model loaded.")

        print("   Warming up Whisper model... (this may cause a brief freeze)")
        silent_audio = np.zeros(16000, dtype=np.float32) # 1 second of silence
        self.whisper_pipeline(silent_audio)
        print("   Whisper model is warm and ready!")

    async def _initialize_tts(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == 'elevenlabs':
            if not all([AsyncElevenLabs, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID]) or "YOUR_API_KEY" in ELEVENLABS_API_KEY:
                raise ValueError("ElevenLabs configuration is missing or uses placeholder values.")
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
            whisper_call = functools.partial(self.whisper_pipeline, pipeline_input)

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
            llm_call = functools.partial(self.llm, prompt=prompt, max_tokens=512, stop=["<|eot_id|>"], temperature=0.7, echo=False)
            response = await loop.run_in_executor(None, llm_call)
            cleaned_text = response['choices'][0]['text'].strip()
            if not cleaned_text:
                cleaned_text = "I'm not sure what to say to that!"
            print(f"   LLM Response: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            print(f"   LLM error: {e}")
            traceback.print_exc()
            return "Oops, my brain just short-circuited! Can you repeat that?"

    async def _interruptible_playback(self, data, samplerate):
        VIRTUAL_CABLE_NAME = "CABLE Input"
        cable_device_index = None
        default_device_index = None
        loop = asyncio.get_running_loop()

        try:
            default_device_index = sd.default.device[1]
            default_device_info = sd.query_devices(default_device_index)
            print(f"   User hearing on: {default_device_info['name']}")

            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if VIRTUAL_CABLE_NAME in device['name'] and device['max_output_channels'] > 0:
                    cable_device_index = i
                    print(f"   VTube Studio listening on: {device['name']}")
                    break
            if cable_device_index is None:
                print(f"   WARNING: '{VIRTUAL_CABLE_NAME}' not found for VTube Studio.")

        except Exception as e:
            print(f"   Error querying audio devices: {e}")
            return

        stream_default, stream_cable = None, None
        try:
            channels = data.shape[1] if data.ndim > 1 else 1
            stream_default = sd.OutputStream(device=default_device_index, samplerate=samplerate, channels=channels, dtype=data.dtype)
            stream_default.start()
            if cable_device_index is not None:
                stream_cable = sd.OutputStream(device=cable_device_index, samplerate=samplerate, channels=channels, dtype=data.dtype)
                stream_cable.start()

            chunk_size = 2048
            for i in range(0, len(data), chunk_size):
                if self.interruption_event.is_set():
                    print("   Playback interrupted by user.")
                    break
                chunk = data[i:i + chunk_size]
                await loop.run_in_executor(None, stream_default.write, chunk)
                if stream_cable:
                    await loop.run_in_executor(None, stream_cable.write, chunk)

        except Exception as e:
            print(f"   Error during dual playback: {e}")
        finally:
            if stream_default: stream_default.stop(); stream_default.close()
            if stream_cable: stream_cable.stop(); stream_cable.close()

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
        audio_stream = self.elevenlabs_client.text_to_speech.stream(text=text, voice_id=ELEVENLABS_VOICE_ID)
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