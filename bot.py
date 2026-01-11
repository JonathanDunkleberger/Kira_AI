# bot.py - Main application file with advanced memory and web search.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback
import random
import re
from typing import List, Callable # Added for type hinting

from ai_core import AI_Core
from memory import MemoryManager
from summarizer import SummarizationManager
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from twitch_tools import start_twitch_poll
from music_tools import play_kira_song
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS
)
from persona import EmotionalState

# Define these here since they are not in config.py
ENABLE_PROACTIVE_THOUGHTS = True
PROACTIVE_THOUGHT_INTERVAL = 60
PROACTIVE_THOUGHT_CHANCE = 0.2
ENABLE_WEB_SEARCH = True
ENABLE_TWITCH_CHAT = True


def parse_kira_tools(text):
    """
    Scans for [POLL: Question | Opt1 | Opt2] 
    or [SONG: Name]
    """
    # Look for Poll Tag
    poll_match = re.search(r'\[POLL:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]', text)
    if poll_match:
        question, opt1, opt2 = poll_match.groups()
        start_twitch_poll(question, [opt1, opt2])
        # Strip the tag so Kira doesn't SAY the code out loud
        text = re.sub(r'\[POLL:.*?\]', '', text)

    # Look for Song Tag
    song_match = re.search(r'\[SONG:\s*(.*?)\]', text)
    if song_match:
        song_name = song_match.group(1)
        play_kira_song(song_name)
        text = re.sub(r'\[SONG:.*?\]', '', text)

    return text.strip()


class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.ai_core = AI_Core(self.interruption_event)
        self.memory = MemoryManager()
        self.summarizer = SummarizationManager(self.ai_core, self.memory)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # --- NEW: Shared Input Queue ---
        self.input_queue = asyncio.Queue()
        
        self.last_interaction_time = time.time()
        self.pyaudio_instance = None
        self.stream = None
        self.frames_per_buffer = int(16000 * 30 / 1000)
        
        self.bg_tasks = set() # Use a set for easier task management
        self.conversation_history = []
        self.conversation_segment = []
        self.unseen_chat_messages = []
        self.current_emotion = EmotionalState.HAPPY
        self.last_idle_chat = "" # Track the last idle chat summary

    def reset_idle_timer(self):
        self.last_interaction_time = time.time()

    async def run(self):
        # --- UPDATED: Moved main logic into a separate task for graceful shutdown ---
        main_task = asyncio.create_task(self._main_loop())
        self.bg_tasks.add(main_task)
        await main_task

    async def _main_loop(self):
        """Contains the primary startup and listening logic."""
        try:
            await self.ai_core.initialize()
            if not self.ai_core.is_initialized: return

            # Test Audio Output (Beep)
            await self.ai_core.test_audio_output()

            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=self.frames_per_buffer
            )

            print(f"\n--- {AI_NAME} is now running. Press Ctrl+C to exit. ---\n")
            
            tasks = []
            
            # 1. Start Twitch Bot (if enabled)
            if ENABLE_TWITCH_CHAT:
                print("   [System] Connecting to Twitch Chat...")
                # Pass the queue to TwitchBot
                twitch_bot = TwitchBot(self.unseen_chat_messages, self.reset_idle_timer, self.input_queue)
                tasks.append(twitch_bot.start())
            
            # 2. Start Brain Worker (The new logic brain)
            print("   [System] Starting Brain Worker...")
            tasks.append(self.brain_worker())
            
            print("   [System] Starting Background Tasks...")
            tasks.append(self.background_loop())
            
            # 3. Start Voice Recorder (This is the main loop effectively)
            print("   [System] Starting Voice Recorder (VAD)...")
            tasks.append(self.vad_loop())

            # Run everything concurrently
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            print("Main loop cancelled.")
        finally:
            print("--- Cleaning up resources... ---")
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.pyaudio_instance: self.pyaudio_instance.terminate()
            print("--- Cleanup complete. ---")


    async def vad_loop(self):
        # This function's logic remains the same
        frames = collections.deque()
        triggered = False
        silent_chunks = 0
        max_silent_chunks = int(PAUSE_THRESHOLD * 1000 / 30)

        while True:
            try:
                data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer, exception_on_overflow=False)

                # Prevent Self-Hearing: Default to silence if AI is speaking
                if self.ai_core.is_speaking:
                    continue

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
                            # Trim the last 0.2s of silence to avoid padding whisper
                            keep_chunks = len(frames) - int(silent_chunks * 0.5) 
                            audio_data = b"".join(list(frames)[:keep_chunks])
                            
                            frames.clear()
                            triggered = False
                            self.reset_idle_timer()
                            
                            # Process audio in background
                            task = asyncio.create_task(self.handle_audio(audio_data))
                            self.bg_tasks.add(task)
                            task.add_done_callback(self.bg_tasks.discard)
            except Exception as e:
                print(f"Error in VAD loop: {e}")
                await asyncio.sleep(0.1)


    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            user_text = await self.ai_core.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3: return
            
            print(f">>> You said: {user_text}")

            # --- NEW: Ignore duplicate inputs ---
            if any(h["content"] == user_text for h in self.conversation_history):
                print(f"(Duplicate input ignored: {user_text})")
                return
            
            # --- PUSH VOICE TO QUEUE ---
            await self.input_queue.put(("voice", user_text))


    async def brain_worker(self):
        """Worker that processes items from the input queue individually."""
        print("   [System] Brain Worker started.")
        while True:
            # Get the next item (this blocks until an item is available)
            source, content = await self.input_queue.get()
            
            try:
                print(f"   [Brain] Processing {source} input: {content[:30]}...")
                contextual_prompt = f"Jonny says: \"{content}\""
                
                # If it's Twitch, mention that
                if source == "twitch":
                     contextual_prompt = f"Twitch Chat says: \"{content}\"\nRespond to this chat message."

                # If previously unseen chat exists (and this is voice), include it
                if source == "voice" and self.unseen_chat_messages:
                     chat_summary = "\n- ".join(self.unseen_chat_messages)
                     contextual_prompt += f"\n\n(While you were listening, Twitch said: {chat_summary})"
                     self.unseen_chat_messages.clear()

                await self.process_and_respond(content, contextual_prompt, "user")
                
            except Exception as e:
                print(f"   [Brain] Error processing item: {e}")
                traceback.print_exc()
            finally:
                self.input_queue.task_done()


    async def process_and_respond(self, original_text: str, contextual_prompt: str, role: str):
        print(f"   (Kira's current emotion is: {self.current_emotion.name})")

        # --- ROLE ALTERNATION ENFORCEMENT ---
        # 1. Merge consecutive messages from same role
        if self.conversation_history and self.conversation_history[-1]["role"] == role:
             print("   [Logic] Merging consecutive message.")
             self.conversation_history[-1]["content"] += f"\n\n{original_text}"
             if self.conversation_segment: 
                 self.conversation_segment[-1]["content"] += f"\n\n{original_text}"
        else:
             # 2. Add new message if role is different
             self.conversation_history.append({"role": role, "content": original_text})
             self.conversation_segment.append({"role": role, "content": original_text})
        
        # --- SLIDING WINDOW: Limit context to last 15 turns ---
        if len(self.conversation_history) > 15:
            # Keep system prompt if we had one? 
            # Current impl injects system prompt in llm_inference every time.
            # So we just truncate the list.
            self.conversation_history = self.conversation_history[-15:]

        # --- SANITY CHECK: Ensure last message is NOT assistant ---
        # If the last message in history is assistant, we cannot ask for completion unless we want them to continue.
        # But here we are responding to a user trigger.
        if self.conversation_history[-1]["role"] == "assistant":
             print("   [Logic] Warning: Attempting to respond to myself. Aborting this turn.")
             return

        mem_ctx = self.memory.search_memories(original_text, n_results=3)
        
        # --- NON-STREAMING LLM & TTS ---
        full_response_text = await self.ai_core.llm_inference(self.conversation_history, self.current_emotion, mem_ctx)
        
        # Clean the response
        full_response_text = self.ai_core._clean_llm_response(full_response_text)
        
        # --- TOOL INTERCEPTOR ---
        # Scan for polls/songs and strip tags before TTS
        full_response_text = parse_kira_tools(full_response_text)
        
        if full_response_text:
            print(f">>> Kira: {full_response_text}")
            
            # Speak the full response
            await self.ai_core.speak_text(full_response_text)
            
            # Update history
            self.conversation_history.append({"role": "assistant", "content": full_response_text})
            self.conversation_segment.append({"role": "assistant", "content": full_response_text})
            if role == "user":
                 self.memory.add_memory(user_text=original_text, ai_text=full_response_text)
            await self.update_emotional_state(original_text, full_response_text)
        
        self.reset_idle_timer()

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

            # Task 1: Read chat during shorter lulls
            is_chat_lull = (time.time() - self.last_interaction_time) > 5.0
            if is_chat_lull and self.unseen_chat_messages:
                async with self.processing_lock:
                    print("\n--- Responding to idle chat... ---")
                    chat_summary = "\n- ".join(self.unseen_chat_messages)
                    if chat_summary != self.last_idle_chat:  # Only respond to new summaries
                        chat_prompt = (
                            "You've been quiet for a moment. Briefly react to these recent messages from your Twitch chat:\n- " 
                            + chat_summary
                        )
                        self.unseen_chat_messages.clear()
                        await self.process_and_respond(f"[Idle Twitch Chat]: {chat_summary}", chat_prompt, "user")
                        self.last_idle_chat = chat_summary  # Update the last idle chat summary
                    continue

            # Task 2: Proactive thoughts ONLY during long periods of total silence.
            is_truly_idle = (time.time() - self.last_interaction_time) > PROACTIVE_THOUGHT_INTERVAL
            if ENABLE_PROACTIVE_THOUGHTS and is_truly_idle and not self.unseen_chat_messages and random.random() < PROACTIVE_THOUGHT_CHANCE:
                async with self.processing_lock:
                    print("\n--- Proactive thought triggered... ---")
                    prompt = "Generate a brief, interesting observation or a random thought."
                    thought = await self.ai_core.llm_inference([], self.current_emotion, prompt)
                    if thought:
                        await self.process_and_respond(thought, thought, "assistant")
                        continue

            # Task 3: Summarize conversation
            if len(self.conversation_segment) >= 8:
                async with self.processing_lock:
                    print("\n--- Summarizing conversation segment... ---")
                    await self.summarizer.consolidate_and_store(self.conversation_segment)
                    self.conversation_segment.clear()


# --- UPDATED: Graceful Shutdown Logic ---
async def main():
    bot = VTubeBot()
    try:
        await bot.run()
    except asyncio.CancelledError:
        print("Main task cancelled.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nApplication shutting down...")
    finally:
        # Gracefully cancel all running tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        # Gather all cancelled tasks to let them finish
        group = asyncio.gather(*tasks, return_exceptions=True)
        loop.run_until_complete(group)
        loop.close()