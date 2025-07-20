# bot.py - Main application file with advanced memory and web search.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback

from ai_core import AI_Core
from memory import MemoryManager
from summarizer import SummarizationManager
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS,
    ENABLE_PROACTIVE_THOUGHTS, PROACTIVE_THOUGHT_INTERVAL, PROACTIVE_THOUGHT_CHANCE,
    ENABLE_WEB_SEARCH, ENABLE_TWITCH_CHAT, EmotionalState
)


class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.ai_core = AI_Core(self.interruption_event)
        self.memory = MemoryManager()
        self.summarizer = SummarizationManager(self.ai_core, self.memory)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        self.last_interaction_time = time.time()
        self.pyaudio_instance = None
        self.stream = None
        self.frames_per_buffer = int(16000 * 30 / 1000)
        
        self.bg_tasks = []
        self.conversation_history = []
        self.conversation_segment = [] # For summarization
        self.twitch_chat_queue = asyncio.Queue()
        self.current_emotion = EmotionalState.HAPPY

    async def run(self):
        try:
            await self.ai_core.initialize()
            if not self.ai_core.is_initialized: return

            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=self.frames_per_buffer
            )
            print(f"\n--- {AI_NAME} is now running. Press Ctrl+C to exit. ---\n")

            if ENABLE_TWITCH_CHAT:
                twitch_bot = TwitchBot(self.twitch_chat_queue)
                self.bg_tasks.append(asyncio.create_task(twitch_bot.start()))
            
            self.bg_tasks.append(asyncio.create_task(self.background_loop()))
            await self.vad_loop()
        except KeyboardInterrupt:
            print("\nShutdown requested by user.")
        finally:
            print("--- Cleaning up resources... ---")
            for task in self.bg_tasks: task.cancel()
            await asyncio.gather(*self.bg_tasks, return_exceptions=True)
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.pyaudio_instance: self.pyaudio_instance.terminate()
            print("--- Cleanup complete. ---")

    async def vad_loop(self):
        frames = collections.deque()
        triggered = False
        silent_chunks = 0
        max_silent_chunks = int(PAUSE_THRESHOLD * 1000 / 30)

        while True:
            data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer, exception_on_overflow=False)
            is_speech = self.vad.is_speech(data, 16000)

            if self.processing_lock.locked() and is_speech:
                self.interruption_event.set()
                continue
            
            if not self.processing_lock.locked():
                if is_speech:
                    if not triggered:
                        print("🎤 Recording...")
                        triggered = True
                    frames.append(data)
                    silent_chunks = 0
                elif triggered:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        audio_data = b"".join(frames)
                        frames.clear()
                        triggered = False
                        self.last_interaction_time = time.time()
                        asyncio.create_task(self.handle_audio(audio_data))

    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            user_text = await self.ai_core.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3: return
            
            print(f">>> You said: {user_text}")
            await self.process_and_respond(user_text, "user")

    async def process_and_respond(self, text: str, role: str):
        # Add to long-term conversation history for LLM context
        self.conversation_history.append({"role": role, "content": text})
        # Add to short-term segment for summarization
        self.conversation_segment.append({"role": role, "content": text})

        mem_ctx = self.memory.search_memories(text, n_results=3)
        response = await self.ai_core.llm_inference(self.conversation_history, self.current_emotion, mem_ctx)
        
        if response:
            await self.ai_core.speak_text(response)
            self.conversation_history.append({"role": "assistant", "content": response})
            self.conversation_segment.append({"role": "assistant", "content": response})
            self.memory.add_memory(user_text=text, ai_text=response)

    async def background_loop(self):
        while True:
            await asyncio.sleep(10) # Check every 10 seconds
            async with self.processing_lock:
                # Summarize conversation if segment is long enough
                if len(self.conversation_segment) >= 6: # e.g., 3 user turns, 3 AI turns
                    print("\n--- Summarizing conversation segment... ---")
                    await self.summarizer.consolidate_and_store(self.conversation_segment)
                    self.conversation_segment.clear() # Clear segment after summarization

                # Proactive thought logic
                is_idle = (time.time() - self.last_interaction_time) > PROACTIVE_THOUGHT_INTERVAL
                if ENABLE_PROACTIVE_THOUGHTS and is_idle and random.random() < PROACTIVE_THOUGHT_CHANCE:
                    print("\n--- Proactive thought triggered... ---")
                    # Simplified proactive thought logic
                    prompt = "Generate a brief, interesting observation or a random thought."
                    thought = await self.ai_core.llm_inference([{"role": "user", "content": prompt}], self.current_emotion)
                    if thought:
                        await self.process_and_respond(thought, "assistant")
                    self.last_interaction_time = time.time() # Reset timer


if __name__ == "__main__":
    bot = VTubeBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nApplication shutting down.")