# bot.py - Main application file with advanced memory and web search.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback
import random

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
        self.conversation_segment = []
        # --- UPDATED: Changed from a queue to a simple list for more control ---
        self.unseen_chat_messages = []
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
                # --- UPDATED: Pass the list instead of the queue ---
                twitch_bot = TwitchBot(self.unseen_chat_messages)
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
                        asyncio.create_task(self.handle_audio(audio_data))

    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            user_text = await self.ai_core.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3: return
            
            print(f">>> You said: {user_text}")

            # --- UPDATED: "Conversational Weaving" Logic ---
            contextual_prompt = f"Jonny says: \"{user_text}\""
            
            # Check for unseen chat messages and weave them into the prompt
            if self.unseen_chat_messages:
                chat_summary = "\n- ".join(self.unseen_chat_messages)
                contextual_prompt += (
                    f"\n\nWhile you were listening, your Twitch chat said:\n- {chat_summary}\n\n"
                    f"Give a single, natural response that addresses Jonny and also acknowledges the chat if it makes sense."
                )
                # Clear the messages now that they've been seen
                self.unseen_chat_messages.clear()
            
            await self.process_and_respond(user_text, contextual_prompt, "user")

    async def process_and_respond(self, original_text: str, contextual_prompt: str, role: str):
        print(f"   (Kira's current emotion is: {self.current_emotion.name})")

        self.conversation_history.append({"role": role, "content": original_text})
        self.conversation_segment.append({"role": role, "content": original_text})

        mem_ctx = self.memory.search_memories(original_text, n_results=3)
        response = await self.ai_core.llm_inference(self.conversation_history[:-1] + [{"role": role, "content": contextual_prompt}], self.current_emotion, mem_ctx)
        
        if response:
            await self.ai_core.speak_text(response)
            self.conversation_history.append({"role": "assistant", "content": response})
            self.conversation_segment.append({"role": "assistant", "content": response})
            if role == "user":
                 self.memory.add_memory(user_text=original_text, ai_text=response)
            await self.update_emotional_state(original_text, response)
        
        self.last_interaction_time = time.time()

    async def update_emotional_state(self, user_text, ai_response):
        new_emotion = await self.ai_core.analyze_emotion_of_turn(user_text, ai_response)
        if new_emotion and new_emotion != self.current_emotion:
            print(f"   ✨ Emotion state changing from {self.current_emotion.name} to {new_emotion.name}")
            self.current_emotion = new_emotion
        elif random.random() < 0.1:
            if self.current_emotion != EmotionalState.HAPPY:
                 print(f"   ✨ Emotion state resetting to HAPPY")
            self.current_emotion = EmotionalState.HAPPY

    async def background_loop(self):
        while True:
            await asyncio.sleep(5)
            
            if self.processing_lock.locked():
                continue

            # --- UPDATED: This loop now handles idle activity ---
            is_idle = (time.time() - self.last_interaction_time) > PROACTIVE_THOUGHT_INTERVAL
            if is_idle:
                async with self.processing_lock:
                    # Priority 1: Respond to chat if idle
                    if self.unseen_chat_messages:
                        print("\n--- Responding to idle chat... ---")
                        chat_summary = "\n- ".join(self.unseen_chat_messages)
                        chat_prompt = (
                            "You've been quiet for a moment. Briefly react to these recent messages from your Twitch chat:\n- " 
                            + chat_summary
                        )
                        self.unseen_chat_messages.clear()
                        await self.process_and_respond(f"[Idle Twitch Chat]: {chat_summary}", chat_prompt, "user")

                    # Priority 2: Proactive thought if no chat to read
                    elif ENABLE_PROACTIVE_THOUGHTS and random.random() < PROACTIVE_THOUGHT_CHANCE:
                        print("\n--- Proactive thought triggered... ---")
                        prompt = "Generate a brief, interesting observation or a random thought."
                        thought = await self.ai_core.llm_inference([{"role": "user", "content": prompt}], self.current_emotion)
                        if thought:
                            await self.process_and_respond(thought, thought, "assistant")

            # Low priority: Summarize conversation if needed
            if len(self.conversation_segment) >= 6:
                async with self.processing_lock:
                    print("\n--- Summarizing conversation segment... ---")
                    await self.summarizer.consolidate_and_store(self.conversation_segment)
                    self.conversation_segment.clear()


if __name__ == "__main__":
    bot = VTubeBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nApplication shutting down.")