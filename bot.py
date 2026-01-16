# bot.py - Main application file with advanced memory and web search.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback
import random
import re
import gc # Added garbage collection
from typing import List, Callable # Added for type hinting

from ai_core import AI_Core
from memory import MemoryManager
from summarizer import SummarizationManager
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from twitch_tools import start_twitch_poll
from music_tools import play_kira_song
from memory_extractor import extract_memories
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS
)
from persona import EmotionalState
from universal_media_bridge import UniversalMediaBridge
from vision_agent import UniversalVisionAgent
from game_mode_controller import GameModeController

# Define these here since they are not in config.py
ENABLE_PROACTIVE_THOUGHTS = True
PROACTIVE_THOUGHT_INTERVAL = 60
PROACTIVE_THOUGHT_CHANCE = 0.2
ENABLE_WEB_SEARCH = True
ENABLE_TWITCH_CHAT = True


def parse_kira_tools(text, allow_music=False):
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
        if allow_music:
            song_name = song_match.group(1)
            play_kira_song(song_name)
            text = re.sub(r'\[SONG:.*?\]', '', text)
        else:
            print("   [System] Music request denied (Voice/System source).")
            text = re.sub(r'\[SONG:.*?\]', '(Music request denied: Twitch Chat only)', text)

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

        # Gaming Mode
        self.media_bridge = UniversalMediaBridge(self.input_queue)
        self.vision_agent = UniversalVisionAgent()
        self.game_mode_controller = GameModeController(self.vision_agent, self.media_bridge)
        
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
        self.turn_count = 0 
        self.is_paused = False # Dashboard control flag
        self.interjected_in_current_silence = False

    def reset_idle_timer(self, human_speech=False):
        self.last_interaction_time = time.time()
        # Only reset the interjection flag if HUMAN speech was detected
        if human_speech:
            self.interjected_in_current_silence = False

    async def run(self):
        # --- UPDATED: Moved main logic into a separate task for graceful shutdown ---
        # Self-healing loop
        while True:
            try:
                # Re-initialize everything cleanly if restarting
                main_task = asyncio.create_task(self._main_loop())
                self.bg_tasks.add(main_task)
                await main_task
                break # If main_loop returns normally, exit
            except asyncio.CancelledError:
                print("Main loop cancelled.")
                break
            except Exception as e:
                print(f"CRITICAL ERROR in Main Loop: {e}")
                traceback.print_exc()
                print(">>> Attempting Self-Healing Restart in 5 seconds...")
                await asyncio.sleep(5)
                # Cleanup before restart
                if self.stream: 
                    try: self.stream.close()
                    except: pass
                if self.pyaudio_instance: 
                    try: self.pyaudio_instance.terminate()
                    except: pass


    async def _main_loop(self):
        """Contains the primary startup and listening logic."""
        try:
            if not self.ai_core.is_initialized:
                 await self.ai_core.initialize()
            
            # Start Senses
            # Vision is now On-Demand, no start() needed

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
            
            # --- NEW: Start Log Bridge (Always start, but it idles if not active) ---
            print("   [System] Starting Universal Media Bridge...")
            tasks.append(self.media_bridge.start())

            # --- NEW: Start Vision Heartbeat ---
            print("   [System] Starting Vision Heartbeat...")
            tasks.append(self.vision_agent.heartbeat_loop())
            
            # --- NEW: Start Dynamic Observer (Visual Spark) ---
            tasks.append(self.dynamic_observer_loop())

            print("   [System] Starting Background Tasks...")
            tasks.append(self.background_loop())
            
            # 3. Start Voice Recorder (This is the main loop effectively)
            print("   [System] Starting Voice Recorder (VAD)...")
            tasks.append(self.vad_loop())

            # Run everything concurrently
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            print("Main loop cancelled.")
            raise
        except Exception as e:
            print(f"Error in internal main loop: {e}")
            raise # Propagate to the self-healing wrapper
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
                # If paused, sleep to save resources instead of spinning
                if self.is_paused:
                    await asyncio.sleep(0.5)
                    continue
                
                # --- FIX: AGGRESSIVE SELF-HEARING PROTECTION ---
                if self.ai_core.is_speaking:
                    # Clear buffer so we don't process old audio when she stops
                    frames.clear() 
                    triggered = False
                    await asyncio.sleep(0.1) 
                    continue
                # -----------------------------------------------

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
                            self.reset_idle_timer(human_speech=True)
                            
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
        print("   [System] Brain Worker started.")
        while True:
            source, content = await self.input_queue.get()
            
            try:
                # 1. Vision Gating Logic (Optimized for Cost vs Detail)
                visual_desc = ""
                if self.game_mode_controller.is_active:
                    # Gate: Only trigger expensive High-Res Vision if explicitly asked
                    VISION_KEYWORDS = ['see', 'look', 'read', 'watching', 'view', 'screen']
                    vision_trigger = any(word in content.lower() for word in VISION_KEYWORDS)
                    
                    if vision_trigger:
                        print("   [Vision] High-Detail Snapshot Requested (Triggered by keyword)...")
                        # Force a high-res capture (is_heartbeat=False) because user asked
                        visual_desc = await self.vision_agent.capture_and_describe(is_heartbeat=False)
                    else:
                        # Default: Use the cached, low-cost heartbeat context (Instant)
                        visual_desc = self.vision_agent.get_vision_context()

                # 2. Get Game Logs (Non-blocking)
                log_context = self.media_bridge.get_latest_events() if self.game_mode_controller.is_active else ""
                
                # 3. Construct Contextual Prompt Logic
                contextual_prompt = ""
                
                if source == "twitch":
                    contextual_prompt = f"Twitch Chat says: \"{content}\""
                elif source == "game_chat":
                     contextual_prompt = f"[In-Game Chat] Jonny types: \"{content}\""
                elif source == "game_event":
                    contextual_prompt = f"[Game Event] {content}"
                else: 
                     contextual_prompt = f"Jonny says: \"{content}\""
                
                # Valid Stream-of-Consciousness Injection
                if visual_desc or log_context:
                    # Note: We put this AFTER the user text so it feels like "Current Context"
                    context_block = ""
                    if visual_desc:
                        context_block += f"\n[Internal Perception: {visual_desc}]"
                    if log_context:
                        context_block += f"\n[Game Events: {log_context}]"
                    
                    contextual_prompt += f"\n\n{context_block}"

                # Pass to LLM
                await self.process_and_respond(content, contextual_prompt, "user", source=source)
                
            except Exception as e:
                print(f"   [Brain] Error: {e}")
                traceback.print_exc()
            finally:
                self.input_queue.task_done()


    async def dynamic_observer_loop(self):
        print("   [System] Guaranteed 15s Observer Active.")
        while True:
            # Check every 2 seconds for high responsiveness
            await asyncio.sleep(2) 

            # 1. Condition: 15s Silence + Not already interjected + Gamer Mode Active
            silence_duration = time.time() - self.last_interaction_time
            if silence_duration < 15:
                continue
            
            # GATE: Only proceed if we haven't already sparked in this specific silence
            if self.interjected_in_current_silence:
                continue
                
            if (self.processing_lock.locked() or 
                self.ai_core.is_speaking or 
                not self.game_mode_controller.is_active or
                self.unseen_chat_messages):
                continue
            
            async with self.processing_lock:
                print("\n--- Visual Spark (Guaranteed)... ---")
                self.interjected_in_current_silence = True # Lock until next human speech
                
                # 3. Get Instant Fresh Context
                visual_desc = self.vision_agent.get_vision_context()
                recent_history = self.conversation_history[-3:] if self.conversation_history else []
                
                if not visual_desc or "Initializing" in visual_desc:
                    continue

                # 4. The 'Cognitive' Prompt - Locked In
                prompt = (
                    f"Jonny has been silent for 15s. Current Vision: {visual_desc}. "
                    f"Recent Chat History: {recent_history}. "
                    "Social Rule: If a previous thread was personal, follow up on it. "
                    "Otherwise, comment on the current game state or ask Jonny a question. "
                    "Do not narrate; be a companion. Stay locked into the present. Under 15 words."
                )

                # 5. Call LLM (Inference Only)
                response = await self.ai_core.llm_inference(
                    messages=self.conversation_history + [{"role": "system", "content": prompt}],
                    current_emotion=self.current_emotion,
                    memory_context="(Social Awareness Mode)"
                )
                
                cleaned_response = self.ai_core._clean_llm_response(response)
                
                # 6. Act or suppress
                if "[SILENCE]" in cleaned_response or len(cleaned_response) < 2:
                    print("   (Kira chose silence)")
                else:
                    print(f"   >>> Visual Spark Triggered: {cleaned_response}")
                    await self.process_and_respond(
                        original_text=cleaned_response, 
                        contextual_prompt=f"[Thought]: {visual_desc}", 
                        role="system", 
                        skip_generation=True
                    )


    async def process_and_respond(self, original_text: str, contextual_prompt: str, role: str, source: str = "voice", skip_generation: bool = False):
        print(f"   (Kira's current emotion is: {self.current_emotion.name})")

        # Define what the LLM sees vs what Memory stores
        llm_user_text = contextual_prompt
        raw_user_text = original_text

        # --- ROLE ALTERNATION ENFORCEMENT ---
        # 1. Merge consecutive messages from same role
        if self.conversation_history and self.conversation_history[-1]["role"] == role:
             print("   [Logic] Merging consecutive message.")
             self.conversation_history[-1]["content"] += f"\n\n{llm_user_text}"
             if self.conversation_segment: 
                 self.conversation_segment[-1]["content"] += f"\n\n{llm_user_text}"
        else:
             # 2. Add new message if role is different
             self.conversation_history.append({"role": role, "content": llm_user_text})
             self.conversation_segment.append({"role": role, "content": llm_user_text})
        
        # --- SLIDING WINDOW: Limit context to last 15 turns ---
        if len(self.conversation_history) > 15:
            self.conversation_history = self.conversation_history[-15:]

        # --- SANITY CHECK: Ensure last message is NOT assistant ---
        # If we are strictly responding (skip_generation=False), we can't respond to ourselves.
        if not skip_generation and self.conversation_history[-1]["role"] == "assistant":
             print("   [Logic] Warning: Attempting to respond to myself. Aborting this turn.")
             return

        # --- MEMORY RETRIEVAL (Structured) ---
        memory_context = self.memory.get_semantic_context(raw_user_text)

        # --- GENERATION OR PASS-THROUGH ---
        if skip_generation:
            full_response_text = original_text # The thought itself is the response
            print(f">>> Kira (Thought): {full_response_text}")
        else:
            # Non-streaming LLM Generation
            full_response_text = await self.ai_core.llm_inference(
                messages=self.conversation_history, 
                current_emotion=self.current_emotion, 
                memory_context=memory_context
            )
        
        # Clean the response
        full_response_text = self.ai_core._clean_llm_response(full_response_text)
        
        # --- TOOL INTERCEPTOR ---
        # Scan for polls/songs and strip tags before TTS
        allow_music = (source == "twitch")
        full_response_text = parse_kira_tools(full_response_text, allow_music=allow_music)
        
        if full_response_text:
            if not skip_generation:
                print(f">>> Kira: {full_response_text}")
            
            # Speak the full response
            await self.ai_core.speak_text(full_response_text)
            
            # Update history (The Assistant's Turn)
            self.conversation_history.append({"role": "assistant", "content": full_response_text})
            self.conversation_segment.append({"role": "assistant", "content": full_response_text})
            
            # Store raw turn in "Turns" collection (for analytics)
            if role == "user":
                 self.memory.add_turn(user_text=raw_user_text, ai_text=full_response_text, source=source)

                 # --- FACT EXTRACTION (Voice Only) ---
                 if source == "voice":
                     memories = await extract_memories(self.ai_core, raw_user_text)
                     if memories:
                         self.memory.store_extracted_memories(memories, source="voice")
            
            await self.update_emotional_state(raw_user_text, full_response_text)
        
        # --- GARBAGE COLLECTION & CLEANUP ---
        self.turn_count += 1
        if self.turn_count % 10 == 0:
            print("   [System] Running Garbage Collection...")
            gc.collect()

        self.reset_idle_timer(human_speech=False)

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

            # (Old Proactive Thoughts Task removed in favor of Dynamic Observer)

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