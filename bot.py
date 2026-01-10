# bot.py - Main application file with advanced memory and web search.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback
import random
import base64
import io
import hashlib
import concurrent.futures
from typing import List, Callable, Optional, Union, Tuple
from PIL import ImageGrab, Image, ImageOps

from ai_core import AI_Core
from memory import MemoryManager
from summarizer import SummarizationManager
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS,
    VISION_ALWAYS_ON, VISION_COOLDOWN_S, VISION_WAIT_MS
)
from persona import EmotionalState
from utils_logger import logger

# Define these here since they are not in config.py
ENABLE_PROACTIVE_THOUGHTS = True
PROACTIVE_THOUGHT_INTERVAL = 60
PROACTIVE_THOUGHT_CHANCE = 0.2
ENABLE_WEB_SEARCH = True
ENABLE_TWITCH_CHAT = True


class VisionManager:
    def __init__(self):
        self.last_capture_time = 0.0
        self.last_hash = None
        self.last_image_b64 = None
        # ThreadPool for blocking Screenshot operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def should_capture(self, text: str) -> bool:
        if VISION_ALWAYS_ON: return True
        
        # Stricter triggers to prevent false positives (e.g. "I like this")
        triggers = [
            "see", "look at", "screen", "what is on", "read this", 
            "what do you see", "describe", "!vision"
        ]
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in triggers):
            # Check cooldown
            if (time.time() - self.last_capture_time) < VISION_COOLDOWN_S:
                logger.info("[Vision] Triggered but skipped due to cooldown.")
                return False
            return True
        return False

    def capture_sync(self) -> Tuple[Optional[str], bool]:
        """
        Blocking capture logic to be run in executor.
        Returns: (base64_image_str, is_cached_frame)
        """
        try:
            start_t = time.perf_counter()
            img = ImageGrab.grab()
            
            # Resize to 896x896 (Gemma 3 Native)
            img = img.resize((896, 896), Image.Resampling.LANCZOS)
            
            # Grayscale for Hashing (Faster than full RGB hash)
            gray = ImageOps.grayscale(img)
            # Use buffer for hash to avoid saving to disk
            hash_val = hashlib.md5(gray.tobytes()).hexdigest()
            
            # Check cache
            if hash_val == self.last_hash and self.last_image_b64:
                # logger.info(f"[Vision] Frame unchanged. Reusing cache. ({time.perf_counter()-start_t:.4f}s)")
                return self.last_image_b64, True 
            
            # Encode if new
            buffered = io.BytesIO()
            img = img.convert("RGB")
            img.save(buffered, format="JPEG", quality=80) 
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            self.last_hash = hash_val
            self.last_image_b64 = img_str
            self.last_capture_time = time.time()
            
            total_t = time.perf_counter() - start_t
            logger.info(f"[Vision] Captured new frame. Capture+Process={total_t:.3f}s")
            return img_str, False
            
        except Exception as e:
            logger.error(f"[Vision] Capture failed: {e}")
            return None, False

    async def capture_async(self):
        """Returns a future that resolves to the image data."""
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.capture_sync)


class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.ai_core = AI_Core(self.interruption_event)
        self.memory = MemoryManager()
        self.summarizer = SummarizationManager(self.ai_core, self.memory)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        self.vision_manager = VisionManager() # NEW: Vision Manager Instance
        
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

            logger.info(f"\n--- {AI_NAME} is now running. Press Ctrl+C to exit. ---\n")
            
            tasks = []
            
            # 1. Start Twitch Bot (if enabled)
            if ENABLE_TWITCH_CHAT:
                logger.info("   [System] Connecting to Twitch Chat...")
                # Pass the queue to TwitchBot
                twitch_bot = TwitchBot(self.unseen_chat_messages, self.reset_idle_timer, self.input_queue)
                tasks.append(twitch_bot.start())
            
            # 2. Start Brain Worker (The new logic brain)
            logger.info("   [System] Starting Brain Worker...")
            tasks.append(self.brain_worker())
            
            logger.info("   [System] Starting Background Tasks...")
            tasks.append(self.background_loop())
            
            # 3. Start Voice Recorder (This is the main loop effectively)
            logger.info("   [System] Starting Voice Recorder (VAD)...")
            tasks.append(self.vad_loop())

            # Run everything concurrently
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("Main loop cancelled.")
        finally:
            logger.info("--- Cleaning up resources... ---")
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.pyaudio_instance: self.pyaudio_instance.terminate()
            logger.info("--- Cleanup complete. ---")


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
                            logger.info("🎤 Recording...")
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
                logger.error(f"Error in VAD loop: {e}")
                await asyncio.sleep(0.1)


    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            # OPTIMIZATION: Start Vision Capture EARLY if Always On
            vision_task = None
            if VISION_ALWAYS_ON:
                 vision_task = asyncio.create_task(self.vision_manager.capture_async())

            user_text = await self.ai_core.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3: return
            
            logger.info(f">>> You said: {user_text}")

            # --- NEW: Ignore duplicate inputs ---
            # Simple check: if last message content matches
            if self.conversation_history:
                last_msg = self.conversation_history[-1]
                last_content = last_msg["content"]
                if isinstance(last_content, list):
                     # Extract text from complex message
                     last_text = next((item["text"] for item in last_content if item["type"] == "text"), "")
                     if last_text == user_text:
                         logger.info(f"(Duplicate input ignored: {user_text})")
                         return
                elif last_content == user_text:
                    logger.info(f"(Duplicate input ignored: {user_text})")
                    return
            
            # --- VISION TRIGGER ---
            # Check Trigger if not already capturing
            if not vision_task and self.vision_manager.should_capture(user_text):
                 logger.info("   [Vision] Trigger detected. Starting capture...")
                 vision_task = asyncio.create_task(self.vision_manager.capture_async())

            # --- PUSH VOICE TO QUEUE ---
            # If vision task exists, pass dictionary with task
            if vision_task:
                 await self.input_queue.put(("voice", {"text": user_text, "image_task": vision_task}))
            else:
                await self.input_queue.put(("voice", user_text))


    async def brain_worker(self):
        """Worker that processes items from the input queue individually."""
        logger.info("   [System] Brain Worker started.")
        while True:
            # Get the next item (this blocks until an item is available)
            source, content = await self.input_queue.get()
            
            try:
                user_text = content
                image_data = None
                
                if isinstance(content, dict):
                    user_text = content.get("text", "")
                    vision_task = content.get("image_task")
                    
                    # Wait for vision if task exists
                    if vision_task:
                        try:
                            wait_s = VISION_WAIT_MS / 1000.0
                            logger.info(f"   [Brain] Waiting up to {VISION_WAIT_MS}ms for vision...")
                            img_str, is_cached = await asyncio.wait_for(vision_task, timeout=wait_s)
                            
                            if img_str:
                                image_data = img_str
                                if is_cached:
                                    logger.info("   [Brain] Using cached vision frame.")
                        except asyncio.TimeoutError:
                             logger.warning("   [Brain] Vision capture timed out. Proceeding without image.")
                        except Exception as e:
                             logger.error(f"   [Brain] Vision error: {e}")

                    logger.info(f"   [Brain] Processing {source} input (Vision): {user_text[:30]}...")
                else:
                    logger.info(f"   [Brain] Processing {source} input: {content[:30]}...")
                    user_text = content
                
                # Default to raw user text (User is Jonny)
                contextual_prompt = user_text
                
                # If it's Twitch, mention that
                if source == "twitch":
                     contextual_prompt = f"[Twitch Chat Message]: {user_text}"

                # If previously unseen chat exists (and this is voice), include it
                if source == "voice" and self.unseen_chat_messages:
                     chat_summary = "\n- ".join(self.unseen_chat_messages)
                     contextual_prompt += f"\n\n(While you were listening, Twitch said: {chat_summary})"
                     self.unseen_chat_messages.clear()
                     
                if image_data:
                    contextual_prompt += "\n(A screenshot of the user's screen is attached.)"

                await self.process_and_respond(user_text, contextual_prompt, "user", image_data=image_data)
                
            except Exception as e:
                logger.error(f"   [Brain] Error processing item: {e}")
                traceback.print_exc()
            finally:
                self.input_queue.task_done()


    async def process_and_respond(self, original_text: str, contextual_prompt: str, role: str, image_data: str = None):
        logger.info(f"   (Kira's current emotion is: {self.current_emotion.name})")

        # Prepare message content (Text or Multimodal)
        # We use contextual_prompt for the history so the AI knows who is speaking (Jonny vs Twitch)
        if image_data:
            # Important: Gemma 3 / LLaVA often expects image first or specific format.
            # We will use the format expected by ai_core which handles the specific message structure.
            # We store it temporarily as text, but ai_core will attach the image dynamically.
            # actually, let's keep it simple here and pass image_data down.
            
            # NOTE: We are NOT modifying conversation_history with the image here anymore.
            # We rely on ai_core to handle the "current" image attachment to prevent history bloat.
            # We just store the text prompt in history.
            message_content = contextual_prompt 
            log_content = original_text + " [Image Attached]"
        else:
            message_content = contextual_prompt
            log_content = original_text

        # --- ROLE ALTERNATION ENFORCEMENT ---
        # 1. Merge consecutive messages from same role
        # Note: We avoid merging if the structure is complex (list vs string) to prevent errors.
        if self.conversation_history and self.conversation_history[-1]["role"] == role:
             logger.info("   [Logic] Merging consecutive message.")
             last_msg_content = self.conversation_history[-1]["content"]
             
             # Only merge if both are strings
             if isinstance(last_msg_content, str) and isinstance(message_content, str):
                 self.conversation_history[-1]["content"] += f"\n\n{message_content}"
                 # Also update segment
                 if self.conversation_segment: 
                     self.conversation_segment[-1]["content"] += f"\n\n{log_content}"
             else:
                 # If type mismatch or multimodal, just append as new turn
                 self.conversation_history.append({"role": role, "content": message_content})
                 self.conversation_segment.append({"role": role, "content": log_content})
        else:
             # 2. Add new message if role is different
             self.conversation_history.append({"role": role, "content": message_content})
             self.conversation_segment.append({"role": role, "content": log_content})
        
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
             logger.warning("   [Logic] Warning: Attempting to respond to myself. Aborting this turn.")
             return

        mem_ctx = self.memory.search_memories(original_text, n_results=3)
        
        # --- NON-STREAMING LLM & TTS ---
        # Pass image_data to ai_core
        full_response_text = await self.ai_core.llm_inference(self.conversation_history, self.current_emotion, mem_ctx, image_buffer=image_data)
        
        # Clean the response
        full_response_text = self.ai_core._clean_llm_response(full_response_text)
        
        if full_response_text:
            logger.info(f">>> Kira: {full_response_text}")
            
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
            logger.info(f"   ✨ Emotion state changing from {self.current_emotion.name} to {new_emotion.name}")
            self.current_emotion = new_emotion
        elif random.random() < 0.1:
            if self.current_emotion != EmotionalState.HAPPY:
                 logger.info(f"   ✨ Emotion state resetting to HAPPY")
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
                    logger.info("\n--- Responding to idle chat... ---")
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
                    logger.info("\n--- Proactive thought triggered... ---")
                    prompt = "Generate a brief, interesting observation or a random thought."
                    thought = await self.ai_core.llm_inference([], self.current_emotion, prompt)
                    if thought:
                        await self.process_and_respond(thought, thought, "assistant")
                        continue

            # Task 3: Summarize conversation
            if len(self.conversation_segment) >= 8:
                async with self.processing_lock:
                    # print("\n--- Summarizing conversation segment... ---")
                    await self.summarizer.consolidate_and_store(self.conversation_segment)
                    self.conversation_segment.clear()


# --- UPDATED: Graceful Shutdown Logic ---
async def main():
    bot = VTubeBot()
    try:
        await bot.run()
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("\nApplication shutting down...")
    finally:
        # Gracefully cancel all running tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        # Gather all cancelled tasks to let them finish
        group = asyncio.gather(*tasks, return_exceptions=True)
        loop.run_until_complete(group)
        loop.close()