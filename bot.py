# bot.py
import asyncio
import collections
import time
import traceback
import random
import webrtcvad
import pyaudio

# Core Components
from ai_core import AI_Core
from memory import MemoryManager
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS,
    ENABLE_PROACTIVE_THOUGHTS, PROACTIVE_THOUGHT_INTERVAL,
    PROACTIVE_THOUGHT_CHANCE, ENABLE_WEB_SEARCH,
    ENABLE_TWITCH_CHAT
)

class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.ai = AI_Core(self.interruption_event)
        self.memory = MemoryManager()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.processing_lock = asyncio.Lock()
        self.last_interaction_time = time.time()
        self.stream = None
        self.pyaudio_instance = None
        self.frames_per_buffer = int(16000 * 30 / 1000)
        self.bg_tasks = []
        self.conversation_history = []
        self.twitch_chat_queue = asyncio.Queue()

    async def run(self):
        try:
            await self.ai.initialize()

            if ENABLE_TWITCH_CHAT:
                twitch_bot = TwitchBot(self.twitch_chat_queue)
                self.bg_tasks.append(asyncio.create_task(twitch_bot.start()))

            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1,
                rate=16000, input=True,
                frames_per_buffer=self.frames_per_buffer
            )

            self.bg_tasks.append(asyncio.create_task(self.chat_and_thought_loop()))
            self.bg_tasks.append(asyncio.create_task(self.vad_loop()))

            print(f"\n{AI_NAME} is now running. Press Ctrl+C to shut down.")
            # Keep the main coroutine alive until it's cancelled
            await asyncio.Event().wait()

        except asyncio.CancelledError:
            print("Main run task was cancelled.")
        finally:
            print("\n--- Shutting down all tasks... ---")
            for task in self.bg_tasks:
                task.cancel()
            await asyncio.gather(*self.bg_tasks, return_exceptions=True)

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()

    async def chat_and_thought_loop(self):
        while True:
            try:
                await asyncio.sleep(5)
                async with self.processing_lock:
                    # Limit history to last 6 turns for LLM context
                    short_history = self.conversation_history[-6:]
                    # Priority 1: Process Twitch Chat
                    if not self.twitch_chat_queue.empty():
                        messages = [await self.twitch_chat_queue.get() for _ in range(self.twitch_chat_queue.qsize())]
                        prompt = "Briefly react to these chat messages:\n- " + "\n- ".join(messages)
                        response = await self.ai.llm_inference(short_history + [{"role": "user", "content": prompt}])
                        if response:
                            await self.ai.speak_text(response)
                            self.conversation_history.append({"role": "assistant", "content": response})
                            self.last_interaction_time = time.time()
                        continue

                    # Priority 2: Proactive Thought
                    is_idle = (time.time() - self.last_interaction_time) > PROACTIVE_THOUGHT_INTERVAL
                    if ENABLE_PROACTIVE_THOUGHTS and is_idle and random.random() < PROACTIVE_THOUGHT_CHANCE:
                        prompt = "Generate a new, interesting, or funny thought."
                        response = await self.ai.llm_inference(short_history + [{"role": "user", "content": prompt}])
                        if response:
                            await self.ai.speak_text(response)
                            self.conversation_history.append({"role": "assistant", "content": response})
                            self.last_interaction_time = time.time()
            except asyncio.CancelledError:
                break

    async def vad_loop(self):
        print(f"{AI_NAME} is now listening...")
        while True:
            frames = collections.deque()
            try:
                await asyncio.to_thread(self._detect_speech, frames)
                if frames:
                    audio_data = b"".join(frames)
                    asyncio.create_task(self.handle_audio(audio_data))
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in VAD loop: {e}")
                await asyncio.sleep(1)

    def _detect_speech(self, frames):
        """This synchronous function runs in a thread to not block asyncio"""
        triggered = False
        silent_chunks = 0
        max_silent_chunks = int((PAUSE_THRESHOLD * 1000) / 30)

        while True:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            is_speech = self.vad.is_speech(data, 16000)

            if self.processing_lock.locked():
                if is_speech: self.interruption_event.set()
                continue

            if is_speech:
                if not triggered:
                    print("  🎤 recording...")
                    triggered = True
                frames.append(data)
                silent_chunks = 0
            elif triggered:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    # End of speech detected
                    return

    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            user_text = await self.ai.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3:
                return

            print(f">>> You said: {user_text}")
            self.conversation_history.append({"role": "user", "content": user_text})
            mem_ctx = self.memory.search_memories(user_text, n_results=2)
            # Limit history to last 6 turns for LLM context
            short_history = self.conversation_history[-6:]
            response = await self.ai.llm_inference(short_history, mem_ctx)

            if response:
                await self.ai.speak_text(response)
                self.conversation_history.append({"role": "assistant", "content": response})
                self.last_interaction_time = time.time()

if __name__ == "__main__":
    bot = VTubeBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")