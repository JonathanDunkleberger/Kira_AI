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
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS, ENABLE_TWITCH_CHAT
)
from persona import EmotionalState
from vision_agent import UniversalVisionAgent
from game_mode_controller import GameModeController, ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL

# Graceful pyautogui import (required for VN auto-play only)
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("   [Info] pyautogui not installed. VN auto-play requires: pip install pyautogui")


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

        # Observer Mode
        self.vision_agent = UniversalVisionAgent()
        self.game_mode_controller = GameModeController(self.vision_agent)
        
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
        self.is_paused = False
        self.silence_stage = 0
        self.is_running = True

        # Activity context — describes what Kira and Jonny are currently doing
        self.current_activity = ""

        # Dashboard feed — rolling log of Twitch messages for display
        self.twitch_log: list[str] = []
        
        # TIMING CONFIGURATION
        self.silence_thresholds = {
            1: 30.0,  # Stage 1: Casual Check-in (User requested 30s)
            2: 60.0,  # Stage 2: Provocation
            3: 120.0  # Stage 3: Chaos
        }

    def reset_idle_timer(self, human_speech=False):
        self.last_interaction_time = time.time()
        if human_speech:
            self.silence_stage = 0

    # ── Activity Detection ──────────────────────────────────────────────────────

    def _detect_activity_change(self, text: str) -> str | None:
        """Parses a voice input for activity-setting phrases. Returns activity string or None."""
        patterns = [
            r"(?:we(?:'re| are)|let(?:'s| us)|i(?:'m| am)|gonna|going to|start)\s+"
            r"(?:playing|watching|reading|listening to|streaming|doing)\s+(.+?)[\.\.\!\?]?$",
            r"(?:play|watch|stream|read|put on|boot up|launch|open)\s+(.+?)[\.\.\!\?]?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.strip(), re.IGNORECASE)
            if match:
                activity = match.group(1).strip().rstrip(' .,!?')
                if 3 < len(activity) < 80:
                    return activity
        return None

    def _classify_activity_type(self, activity: str) -> str:
        """Maps a free-form activity string to a known ACTIVITY_* constant."""
        lower = activity.lower()
        VN_KEYWORDS = ["visual novel", " vn ", "clannad", "katawa", "fate/",
                       "doki doki", "steins", "little busters", "kanon",
                       "planetarian", "rewrite", "angel beats", "renpy", "ren'py"]
        MEDIA_KEYWORDS = ["movie", "anime", "episode", "youtube", "netflix",
                          "crunchyroll", "watching"]
        for kw in VN_KEYWORDS:
            if kw in lower:
                return ACTIVITY_VN
        for kw in MEDIA_KEYWORDS:
            if kw in lower:
                return ACTIVITY_MEDIA
        return ACTIVITY_GAME  # generic fallback

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
            
            # --- Start Vision Heartbeat ---
            print("   [System] Starting Vision Heartbeat...")
            tasks.append(self.vision_agent.heartbeat_loop())
            
            # --- NEW: Start Dynamic Observer (Visual Spark) ---
            tasks.append(self.dynamic_observer_loop())

            # --- VN Auto-Play Agent (standby until activity_type == ACTIVITY_VN) ---
            tasks.append(self.vn_gameplay_loop())

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

        while self.is_running:
            try:
                if not self.is_running: break

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

                # --- SAFE READ ---
                try:
                    data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer, exception_on_overflow=False)
                except (OSError, IOError) as e:
                    if e.errno == -9988 or not self.is_running: 
                        break # Stream closed, exit quietly
                    print(f"VAD Stream Error: {e}")
                    await asyncio.sleep(1)
                    continue
                # -----------------

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
                # Log Twitch messages for the dashboard feed
                if source == "twitch":
                    self.twitch_log.append(content)
                    if len(self.twitch_log) > 100:
                        self.twitch_log = self.twitch_log[-100:]

                # Activity auto-detection from voice (natural language sets context)
                if source == "voice":
                    detected = self._detect_activity_change(content)
                    if detected and detected != self.current_activity:
                        self.current_activity = detected
                        new_type = self._classify_activity_type(detected)
                        self.game_mode_controller.activity_type = new_type
                        self.vision_agent.activity_type = new_type
                        print(f"   [Activity] Set to: '{detected}' (type: {new_type})")

                # 1. Vision Gating Logic (Optimized for Cost vs Detail)
                visual_desc = ""
                if self.game_mode_controller.is_active:
                    # Gate: Only trigger expensive High-Res Vision if explicitly asked
                    VISION_KEYWORDS = ['see', 'look', 'read', 'watching', 'view', 'screen']
                    vision_trigger = any(word in content.lower() for word in VISION_KEYWORDS)
                    
                    if vision_trigger:
                        # Check Cache Freshness
                        time_since_last = time.time() - self.vision_agent.last_capture_time
                        if time_since_last < 7:
                            print(f"   [Vision] Using Cached Context (Fresh: {int(time_since_last)}s)...")
                            visual_desc = self.vision_agent.last_description
                        else:
                            print("   [Vision] High-Detail Snapshot Requested (Cache Stale > 7s)...")
                            # Force a high-res capture
                            visual_desc = await self.vision_agent.capture_and_describe(is_heartbeat=False)
                    else:
                        # Default: Use the cached, low-cost heartbeat context (Instant)
                        visual_desc = self.vision_agent.get_vision_context()

                # 2. Construct Contextual Prompt Logic
                contextual_prompt = ""
                
                if source == "twitch":
                    contextual_prompt = f"Twitch Chat says: \"{content}\""
                else: 
                    contextual_prompt = f"Jonny says: \"{content}\""
                
                # Inject vision context if available
                if visual_desc:
                    if self.game_mode_controller.activity_type == ACTIVITY_VN:
                        contextual_prompt += f"\n\n[What you can see on screen right now: {visual_desc}]"
                    else:
                        contextual_prompt += f"\n\n[Internal Perception: {visual_desc}]"

                # Pass to LLM
                await self.process_and_respond(content, contextual_prompt, "user", source=source)
                
            except Exception as e:
                print(f"   [Brain] Error: {e}")
                traceback.print_exc()
            finally:
                self.input_queue.task_done()


    async def dynamic_observer_loop(self):
        print("   [System] Observer Loop Active (Universal Boredom Protocol).")
        while self.is_running:
            await asyncio.sleep(1.0) # Check every second
            
            # Don't interrupt if speaking or processing
            if self.processing_lock.locked() or self.ai_core.is_speaking:
                continue

            last_activity = max(self.last_interaction_time, self.ai_core.last_speech_finish_time)
            silence_duration = time.time() - last_activity

            in_vn_mode = (
                self.game_mode_controller.is_active and
                self.game_mode_controller.activity_type == ACTIVITY_VN
            )

            if in_vn_mode:
                # ── VN OBSERVER MODE: calm couch-buddy behaviour ──────────────
                # Stage 2 (240s): thoughtful story question
                if silence_duration > 240.0 and self.silence_stage < 2:
                    async with self.processing_lock:
                        self.silence_stage = 2
                        vn_ctx = self.vision_agent.get_vision_context()
                        await self._execute_interjection(
                            f"You and Jonny are watching a visual novel together. "
                            f"Current screen: {vn_ctx}\n\n"
                            f"Ask Jonny one thoughtful question about the story, characters, or "
                            f"something you're both curious about. Keep it brief and natural, "
                            f"like a friend on the couch.",
                            memory_query=f"visual novel story {self.current_activity}",
                        )

                # Stage 1 (90s): react naturally to what's on screen
                elif silence_duration > 90.0 and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        vn_ctx = self.vision_agent.get_vision_context()
                        await self._execute_interjection(
                            f"You and Jonny are watching a visual novel together. "
                            f"Current screen: {vn_ctx}\n\n"
                            f"Make a short natural remark about what just happened or what you noticed "
                            f"on screen — like a friend watching with him. One or two sentences only. "
                            f"Do NOT ask a generic 'what are you thinking' question.",
                            memory_query=f"visual novel {self.current_activity}",
                        )
                # No chaos stage during VN — silence is normal while reading

            else:
                # ── DEFAULT MODE: boredom escalation ─────────────────────────
                # STAGE 3: CHAOS (120s)
                if silence_duration > self.silence_thresholds[3] and self.silence_stage < 3:
                    async with self.processing_lock:
                        self.silence_stage = 3
                        await self._execute_interjection(
                            "The stream is dead. Say something completely unhinged to wake Jonny up.",
                            memory_query="recent things Jonny mentioned",
                        )

                # STAGE 2: ROAST (60s)
                elif silence_duration > self.silence_thresholds[2] and self.silence_stage < 2:
                    async with self.processing_lock:
                        self.silence_stage = 2
                        await self._execute_interjection(
                            "Jonny has been quiet for a minute. Roast him for being boring.",
                            memory_query="what makes Jonny laugh or react",
                        )

                # STAGE 1: CHECK-IN (30s)
                elif silence_duration > self.silence_thresholds[1] and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        await self._execute_interjection(
                            "It's been quiet for 30 seconds. Ask Jonny a casual question about what he's thinking.",
                            memory_query="what is Jonny interested in",
                        )

    async def vn_gameplay_loop(self):
        """
        Autonomous Visual Novel gameplay agent.
        Activates when game_mode_controller.activity_type == ACTIVITY_VN and observer is on.
        Reads screen text, advances dialogue with spacebar, and picks choices via keyboard.

        Requirements:
          - Observer Mode ON (vision enabled)
          - OPENAI_API_KEY set (vision uses GPT-4o-mini)
          - pip install pyautogui
          - VN window must be in focus when choices need to be made
        """
        print("   [System] VN Gameplay Agent on standby.")
        VN_TICK = 8.0  # seconds between screen checks

        while self.is_running:
            await asyncio.sleep(VN_TICK)

            # Only run when in VN mode with observer active
            if (not self.game_mode_controller.is_active or
                    self.game_mode_controller.activity_type != ACTIVITY_VN):
                continue

            # Don't interrupt active speech or input processing
            if self.processing_lock.locked() or self.ai_core.is_speaking:
                continue

            if not PYAUTOGUI_AVAILABLE:
                print("   [VN] pyautogui not installed. Run: pip install pyautogui")
                await asyncio.sleep(60)
                continue

            # Ensure vision is active in VN mode
            if not self.vision_agent.is_active:
                self.vision_agent.is_active = True

            # Capture structured VN state from screen
            vn_state = await self.vision_agent.capture_vn_state()
            if not vn_state:
                continue

            dialogue = vn_state.get("dialogue", "").strip()
            choices  = vn_state.get("choices", [])
            speaker  = vn_state.get("speaker", "Narration")
            scene    = vn_state.get("scene", "")

            if not dialogue:
                continue

            async with self.processing_lock:
                if choices:
                    # ── CHOICE MENU ─────────────────────────────────────────────
                    choice_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    choice_prompt = (
                        f"You are playing a Visual Novel. A choice menu appeared.\n"
                        f"Speaker: {speaker}\n"
                        f"Last line: \"{dialogue}\"\n"
                        f"Scene: {scene}\n\n"
                        f"Your choices:\n{choice_list}\n\n"
                        f"Pick the option that fits your personality and what you think is interesting for the story. "
                        f"Start with ONLY the choice NUMBER, then react naturally in 1-2 sentences."
                    )
                    response = await self.ai_core.llm_inference(
                        messages=self.conversation_history[-6:] + [{"role": "user", "content": choice_prompt}],
                        current_emotion=self.current_emotion,
                        memory_context="",
                        activity_context=self.current_activity,
                    )
                    cleaned = self.ai_core._clean_llm_response(response)

                    # Extract the chosen number
                    match = re.search(r'\b([1-9])\b', cleaned)
                    choice_num = int(match.group(1)) if match else 1
                    choice_num = min(choice_num, len(choices))

                    print(f"   [VN] Kira picks choice {choice_num}: {choices[choice_num - 1]}")

                    # Navigate using keyboard (works for Ren'Py and most VN engines)
                    try:
                        for _ in range(choice_num - 1):
                            pyautogui.press('down')
                            await asyncio.sleep(0.15)
                        await asyncio.sleep(0.3)
                        pyautogui.press('enter')
                    except Exception as e:
                        print(f"   [VN] Input error: {e}")

                    if cleaned and "[SILENCE]" not in cleaned and len(cleaned) > 5:
                        print(f">>> Kira (VN Choice): {cleaned}")
                        await self.ai_core.speak_text(cleaned)
                        self.conversation_history.append({"role": "assistant", "content": cleaned})
                        self.ai_core.last_speech_finish_time = time.time()

                else:
                    # ── DIALOGUE LINE: advance text, occasionally comment ─────
                    try:
                        pyautogui.press('space')
                    except Exception as e:
                        print(f"   [VN] Input error: {e}")
                        continue

                    # ~30% of the time, react to the line naturally
                    if random.random() < 0.30:
                        comment_prompt = (
                            f"You are playing a Visual Novel.\n"
                            f"Character speaking: {speaker}\n"
                            f"Line just shown: \"{dialogue}\"\n"
                            f"Scene: {scene}\n\n"
                            f"React in 1-2 sentences as yourself. Be genuine — this is your real reaction to the story. "
                            f"If it is boring, say so. If something is interesting, engage with it."
                        )
                        response = await self.ai_core.llm_inference(
                            messages=self.conversation_history[-4:] + [{"role": "user", "content": comment_prompt}],
                            current_emotion=self.current_emotion,
                            memory_context="",
                            activity_context=self.current_activity,
                        )
                        cleaned = self.ai_core._clean_llm_response(response)
                        if cleaned and "[SILENCE]" not in cleaned and len(cleaned) > 5:
                            print(f">>> Kira (VN): {cleaned}")
                            await self.ai_core.speak_text(cleaned)
                            self.conversation_history.append({"role": "assistant", "content": cleaned})
                            self.ai_core.last_speech_finish_time = time.time()

    async def _execute_interjection(self, prompt, memory_query: str = ""):
        """Runs a proactive interjection with real memory context."""
        memory_context = self.memory.get_semantic_context(memory_query or prompt)
        response = await self.ai_core.llm_inference(
            messages=self.conversation_history + [{"role": "system", "content": prompt}],
            current_emotion=self.current_emotion,
            memory_context=memory_context,
            activity_context=self.current_activity,
        )
        cleaned = self.ai_core._clean_llm_response(response)
        if len(cleaned) > 2 and "[SILENCE]" not in cleaned:
            print(f"   >>> Kira (Bored): {cleaned}")
            await self.ai_core.speak_text(cleaned)
            self.conversation_history.append({"role": "assistant", "content": cleaned})


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
        
        # --- SLIDING WINDOW: Keep 20 turns for better conversational memory ---
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

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
                memory_context=memory_context,
                activity_context=self.current_activity,
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

                 # --- FACT EXTRACTION (UPDATED) ---
                 # Only run if user spoke, and pass the HISTORY for context
                 if source == "voice":
                     # Fire and forget - don't await this, let it run in background
                     asyncio.create_task(self._run_memory_extraction(raw_user_text))
            
            await self.update_emotional_state(raw_user_text, full_response_text)
        
        # --- GARBAGE COLLECTION & CLEANUP ---
        self.turn_count += 1
        if self.turn_count % 10 == 0:
            print("   [System] Running Garbage Collection...")
            gc.collect()

        # REMOVED: self.reset_idle_timer(human_speech=False) to prevents AI from resetting silence timer

    async def _run_memory_extraction(self, text):
        """Wrapper to run memory extraction without blocking the main conversation"""
        try:
            # Pass a snapshot of history so it knows what "it" refers to
            memories = await extract_memories(self.ai_core, text, self.conversation_history)
            if memories:
                self.memory.store_extracted_memories(memories, source="voice")
        except Exception as e:
            print(f"   [Async Memory Error]: {e}")

    async def update_emotional_state(self, user_text, ai_response):
        new_emotion = await self.ai_core.analyze_emotion_of_turn(user_text, ai_response)
        if new_emotion and new_emotion != self.current_emotion:
            print(f"   \u2728 Emotion: {self.current_emotion.name} \u2192 {new_emotion.name}")
            self.current_emotion = new_emotion

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