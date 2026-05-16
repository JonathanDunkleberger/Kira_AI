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
        self.mode = "companion"  # 'companion' or 'streamer'
        self.event_loop = None   # set when _main_loop starts, used by dashboard for cross-thread calls
        self.vn_autoplay_enabled = False  # When True, Kira actively reads and advances VNs
        self.immersive = False   # When True, Kira stays quiet unless invited. Auto-enables for VN/MEDIA.
        self.session_scene_log: list = []  # Recent scene summaries during this session
        self.session_highlights: list = []  # Highlights captured this session
        self.last_highlight_check_time = 0

        # Activity context — describes what Kira and Jonny are currently doing
        self.current_activity = ""

        # Dashboard feed — rolling log of Twitch messages for display
        self.twitch_log: list[str] = []
        
        # TIMING CONFIGURATION
        self.silence_thresholds = {
            1: 45.0,   # Light casual remark (streamer mode only)
            2: 90.0,   # Slightly bigger nudge (streamer mode only)
        }

    def reset_idle_timer(self, human_speech=False):
        self.last_interaction_time = time.time()
        if human_speech:
            self.silence_stage = 0

    # ── Activity Detection ──────────────────────────────────────────────────────

    def _detect_activity_change(self, text: str) -> str | None:
        """Parses a voice input for activity-setting phrases. Returns activity string or None.
        Strict: only matches clear activity declarations, NOT imperative requests like
        'read the text' or 'look at the screen'. Those are commands to Kira, not activity changes."""
        stripped = text.strip()
        lower = stripped.lower()

        # Hard filter: ignore imperative "do this for me" requests
        imperative_signals = [
            "read the", "read this", "read that", "read it", "read all",
            "look at", "see the", "see this", "see that", "see if",
            "tell me", "show me", "check the", "what does",
            "can you", "could you", "will you", "would you",
            "please ", "kira,", "kira ", "hey kira",
        ]
        for sig in imperative_signals:
            if sig in lower:
                return None

        # Only match explicit activity declarations
        patterns = [
            r"^(?:let(?:'s| us))\s+(?:play|watch|read|stream|start)\s+(.+?)[\.|\!|\?]?$",
            r"^(?:we(?:'re| are)|i(?:'m| am))\s+(?:playing|watching|reading|streaming)\s+(.+?)[\.|\!|\?]?$",
            r"^(?:gonna|going to)\s+(?:play|watch|read|stream|start)\s+(.+?)[\.|\!|\?]?$",
            r"^(?:start(?:ing)?|boot(?:ing)? up|launch(?:ing)?|open(?:ing)?)\s+(.+?)[\.|\!|\?]?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, stripped, re.IGNORECASE)
            if match:
                activity = match.group(1).strip().rstrip(' .,!?')
                # Reject very generic activities — must look like a title or media name
                if 3 < len(activity) < 60 and not activity.lower().startswith(("the ", "a ", "this ", "that ", "it ", "all ")):
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
                    except Exception: pass
                if self.pyaudio_instance: 
                    try: self.pyaudio_instance.terminate()
                    except Exception: pass


    async def _main_loop(self):
        """Contains the primary startup and listening logic."""
        self.event_loop = asyncio.get_running_loop()
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

            # --- Highlight Extraction Loop (long-term memory layer) ---
            tasks.append(self.highlight_extraction_loop())

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
            # Save session memory before tearing down (if we were in an immersive session)
            if self.immersive and (self.session_scene_log or self.session_highlights):
                try:
                    await self._generate_session_summary()
                except Exception as e:
                    print(f"   [Session] Final summary failed: {e}")
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
                        # Passive media auto-enables immersive; everything else turns it off.
                        old_immersive = self.immersive
                        self.immersive = new_type in (ACTIVITY_VN, ACTIVITY_MEDIA)
                        print(f"   [Immersive] {self.immersive}")

                        # Adjust vision heartbeat for the mode
                        self.vision_agent.heartbeat_interval = 10.0 if self.immersive else 30.0

                        # If switching OUT of immersive (activity changed), save a session summary
                        if old_immersive and not self.immersive and self.session_scene_log:
                            asyncio.create_task(self._generate_session_summary())

                # 1. Vision Gating Logic (Optimized for Cost vs Detail)
                visual_desc = ""
                if self.game_mode_controller.is_active:
                    lower = content.lower()
                    # Explicit READ intent — speak the verbatim transcription directly, bypass LLM
                    read_phrases = [
                        'read the text', 'read this', 'read the screen', 'read all',
                        'read it', 'read me', 'read what', 'read out',
                        'reading this', 'reading the', 'reading what',
                        'what does it say', 'what does that say', 'what does the screen say',
                        'what does it read', 'transcribe',
                    ]
                    is_read_request = any(p in lower for p in read_phrases)

                    VISION_KEYWORDS = ['see', 'look', 'read', 'watching', 'view', 'screen']
                    vision_trigger = any(word in lower for word in VISION_KEYWORDS)

                    if is_read_request:
                        print("   [Vision] READ intent — transcribing screen verbatim...")
                        transcribed = await self.vision_agent.capture_and_transcribe()

                        # Speak the transcription DIRECTLY, no LLM paraphrasing
                        if (transcribed
                                and "NO TEXT VISIBLE" not in transcribed.upper()
                                and len(transcribed.strip()) > 10):
                            preamble = random.choice([
                                "Okay — ",
                                "Sure, it says: ",
                                "Here's what's on screen: ",
                                "It reads: ",
                                "Alright — ",
                            ])
                            speak_text = preamble + transcribed.strip()
                            print(f"   [Vision] Bypassing LLM. Speaking verbatim ({len(transcribed)} chars).")

                            user_line = f"Jonny says: \"{content}\""
                            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                                self.conversation_history[-1]["content"] += f"\n\n{user_line}"
                            else:
                                self.conversation_history.append({"role": "user", "content": user_line})

                            await self.ai_core.speak_text(speak_text)
                            self.conversation_history.append({"role": "assistant", "content": speak_text})
                            self.ai_core.last_speech_finish_time = time.time()
                            continue  # Skip the rest of this brain_worker iteration
                        else:
                            # No readable text — fall through to normal LLM path with a hint
                            visual_desc = (
                                "I tried to read the screen but there's nothing legible right now — "
                                "probably a transition, image-only frame, or animation. "
                                "Acknowledge briefly that there's nothing to read at the moment."
                            )
                    elif vision_trigger:
                        time_since_last = time.time() - self.vision_agent.last_capture_time
                        if time_since_last < 7:
                            print(f"   [Vision] Using Cached Context (Fresh: {int(time_since_last)}s)...")
                            visual_desc = self.vision_agent.last_description
                        else:
                            print("   [Vision] High-Detail Snapshot Requested (Cache Stale > 7s)...")
                            visual_desc = await self.vision_agent.capture_and_describe(is_heartbeat=False)
                    else:
                        visual_desc = self.vision_agent.get_vision_context()

                # 2. Construct dialogue line (history-clean — no screen state)
                if source == "twitch":
                    dialogue_line = f"Twitch Chat says: \"{content}\""
                else:
                    dialogue_line = f"Jonny says: \"{content}\""

                # Speech triage — decide whether to respond, react briefly, or stay quiet
                scene_ctx = self.vision_agent.get_vision_context() if self.game_mode_controller.is_active else ""
                decision = await self.ai_core.decide_response_mode(
                    recent_history=self.conversation_history,
                    incoming_line=content,
                    scene_context=scene_ctx,
                    source=source,
                    immersive=self.immersive,
                )

                if decision == "STAY_QUIET":
                    # QUESTION OVERRIDE: direct questions always get a response, never STAY_QUIET.
                    # A friend who ignores your questions is broken, immersive or not.
                    content_stripped = content.strip()
                    looks_like_question = (
                        "?" in content_stripped
                        or content_stripped.lower().startswith((
                            "what", "why", "how", "when", "where", "who", "which",
                            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
                            "can ", "could ", "will ", "would ", "should ", "kira",
                        ))
                    )
                    if looks_like_question:
                        print(f"   [Triage] Upgrading STAY_QUIET \u2192 BRIEF (question detected)")
                        decision = "BRIEF"
                    else:
                        print(f"   [Triage] STAY_QUIET \u2014 letting it pass.")
                        continue

                brief_mode = (decision == "BRIEF")
                # In immersive mode, even RESPOND defaults to brief — Kira gets to the point
                # during media. Direct addresses still respond, just shorter.
                if self.immersive and decision == "RESPOND":
                    brief_mode = True
                print(f"   [Triage] {decision}")

                # Pass to LLM (vision context injected fresh at inference, not persisted to history)
                await self.process_and_respond(
                    content,
                    dialogue_line,
                    "user",
                    source=source,
                    situational_context=visual_desc,
                    brief_mode=brief_mode,
                )
                
            except Exception as e:
                print(f"   [Brain] Error: {e}")
                traceback.print_exc()
            finally:
                self.input_queue.task_done()


    async def request_thoughts(self):
        """Triggered by the dashboard 'Invite' button. Asks Kira to share her honest
        take on whatever is happening on screen right now. Uses the deep brain (Claude Opus)
        when available \u2014 this is the moment where intelligence matters most."""
        if self.processing_lock.locked() or self.ai_core.is_speaking:
            return
        async with self.processing_lock:
            scene = self.vision_agent.get_vision_context()
            memory = self.memory.get_semantic_context(f"thoughts on {self.current_activity}")

            request = (
                f"Jonny just invited you to share your thoughts on what's happening right now. "
                f"This is a moment between you two \u2014 a couch friend sharing a take, not a chatbot. "
                f"React to a specific character, plot beat, or detail you noticed. Use their names. "
                f"Be funny, sassy, sweet, weird, blunt, whatever fits the moment. Keep it conversational, "
                f"2-4 sentences. Don't ask what Jonny thinks \u2014 share your own take."
            )

            response = await self.ai_core.kira_deep_response(
                request=request,
                scene_context=scene,
                memory_context=memory,
                recent_history=self.conversation_history,
            )
            cleaned = self.ai_core._clean_llm_response(response)
            if cleaned and len(cleaned) > 2:
                print(f"   >>> Kira (Invite/Deep): {cleaned}")
                await self.ai_core.speak_text(cleaned)
                self.conversation_history.append({"role": "assistant", "content": cleaned})
                self.ai_core.last_speech_finish_time = time.time()

    async def dynamic_observer_loop(self):
        print("   [System] Observer Loop Active (Universal Boredom Protocol).")
        while self.is_running:
            await asyncio.sleep(1.0) # Check every second
            
            # Don't interrupt if speaking or processing
            if self.processing_lock.locked() or self.ai_core.is_speaking:
                continue

            # Immersive mode: more conservative thresholds, scene-change gating,
            # and skip if dialogue text is actively advancing on screen (Jonny is reading).
            if self.immersive:
                # Suppress speech while user is actively reading new dialogue
                time_since_dialogue_change = time.time() - getattr(self.vision_agent, "last_dialogue_change_time", 0)
                if time_since_dialogue_change < 8.0:
                    # Text just advanced — Jonny is reading, hold off
                    continue

                # Use conservative immersive thresholds (override default streamer thresholds)
                immersive_stage_1 = 120.0  # ~2 min before first soft remark
                immersive_stage_2 = 300.0  # ~5 min for second-level nudge

                if silence_duration > immersive_stage_2 and self.silence_stage < 2:
                    async with self.processing_lock:
                        self.silence_stage = 2
                        scene = self.vision_agent.get_vision_context()
                        await self._execute_interjection(
                            f"You and Jonny are watching {self.current_activity or 'something'} together. "
                            f"Current scene: {scene}\n\n"
                            f"Drop a brief, natural observation about a character, the mood, or something on screen — "
                            f"like a friend on the couch. One short sentence. No questions. Observational, not interrogative.",
                            memory_query=f"reactions to {self.current_activity}",
                        )
                elif silence_duration > immersive_stage_1 and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        scene = self.vision_agent.get_vision_context()
                        await self._execute_interjection(
                            f"You and Jonny are watching {self.current_activity or 'something'} together. "
                            f"Current scene: {scene}\n\n"
                            f"Make a short, natural remark about what just happened or what stands out — "
                            f"like a friend reacting under their breath. One short sentence. "
                            f"No questions. Observational, not interrogative.",
                            memory_query=f"reactions to {self.current_activity}",
                        )
                continue  # Don't fall through to the streamer-mode logic below

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
                # ── STREAMER MODE: boredom escalation ────────────────────────
                # STAGE 2: nudge (90s)
                if silence_duration > self.silence_thresholds[2] and self.silence_stage < 2:
                    async with self.processing_lock:
                        self.silence_stage = 2
                        await self._execute_interjection(
                            "Jonny has been quiet for a while. Make a brief, natural observation \u2014 something on your mind, something about the scene, anything. Don't ask him what he's thinking. One short sentence.",
                            memory_query="what makes Jonny laugh or react",
                        )

                # STAGE 1: light remark (45s)
                elif silence_duration > self.silence_thresholds[1] and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        await self._execute_interjection(
                            "It's been quiet for a bit. Drop a short, natural remark \u2014 light, friendly, like something you'd say to a friend on the couch. Don't ask him what he's thinking. One short sentence.",
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

            # Auto-play is a separate opt-in from VN activity context
            if not self.vn_autoplay_enabled:
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
        """Runs a proactive interjection. Routes through Claude Opus when available —
        Claude follows the anti-fabrication instruction reliably; local Llama 8B does not."""
        memory_context = self.memory.get_semantic_context(memory_query or prompt)

        anti_fabrication = (
            "\n\nCRITICAL RULES:\n"
            "- Do NOT reference past events, games, conversations, or shared experiences "
            "that you cannot verify from the memory notes provided.\n"
            "- Do NOT invent shared history ('that game we played', 'remember when', etc.) "
            "unless it is explicitly in the memory notes.\n"
            "- React only to the current moment and what is actually present right now.\n"
            "- If you have nothing real to react to, make a small genuine observation about the present scene "
            "or stay closer to a simple aside. No fabricated nostalgia."
        )

        full_prompt = prompt + anti_fabrication

        # Route through Claude when available — local Llama 8B can't reliably follow the anti-fabrication rule
        if self.ai_core.anthropic_client:
            scene = self.vision_agent.get_vision_context() if self.game_mode_controller.is_active else ""
            try:
                response = await self.ai_core.kira_deep_response(
                    request=full_prompt,
                    scene_context=scene,
                    memory_context=memory_context,
                    recent_history=self.conversation_history,
                )
            except Exception as e:
                print(f"   [Interjection] Claude failed, falling back to local: {e}")
                response = await self.ai_core.llm_inference(
                    messages=self.conversation_history + [{"role": "system", "content": full_prompt}],
                    current_emotion=self.current_emotion,
                    memory_context=memory_context,
                    activity_context=self.current_activity,
                )
        else:
            response = await self.ai_core.llm_inference(
                messages=self.conversation_history + [{"role": "system", "content": full_prompt}],
                current_emotion=self.current_emotion,
                memory_context=memory_context,
                activity_context=self.current_activity,
            )

        cleaned = self.ai_core._clean_llm_response(response)
        if len(cleaned) > 2 and "[SILENCE]" not in cleaned:
            print(f"   >>> Kira (Bored): {cleaned}")
            await self.ai_core.speak_text(cleaned)
            self.conversation_history.append({"role": "assistant", "content": cleaned})


    async def process_and_respond(self, original_text: str, dialogue_line: str, role: str, source: str = "voice", skip_generation: bool = False, situational_context: str = "", brief_mode: bool = False):
        print(f"   (Kira's current emotion is: {self.current_emotion.name})")

        # Define what the LLM sees vs what Memory stores
        llm_user_text = dialogue_line
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
            effective_situational = situational_context
            if brief_mode:
                brief_instruction = "[BRIEF MODE: Respond in one short, natural sentence. No elaboration, no follow-up question.]"
                effective_situational = (situational_context + "\n\n" + brief_instruction) if situational_context else brief_instruction

            full_response_text = await self.ai_core.llm_inference(
                messages=self.conversation_history,
                current_emotion=self.current_emotion,
                memory_context=memory_context,
                activity_context=self.current_activity,
                situational_context=effective_situational,
                max_tokens_override=(50 if brief_mode else None),
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

    async def highlight_extraction_loop(self):
        """Background loop. Every 90s during immersive media, asks Claude Opus
        if any moment in the recent scene history is worth remembering."""
        print("   [System] Highlight Extraction Loop active.")
        while self.is_running:
            await asyncio.sleep(90.0)
            if not self.is_running:
                break
            if not self.immersive:
                continue
            if self.processing_lock.locked() or self.ai_core.is_speaking:
                continue

            scene_summary = getattr(self.vision_agent, "scene_summary", "")
            if not scene_summary or len(scene_summary) < 40:
                continue

            # Append current scene to session log for end-of-session summary
            self.session_scene_log.append({
                "time": time.time(),
                "summary": scene_summary,
            })
            # Cap log size
            if len(self.session_scene_log) > 100:
                self.session_scene_log = self.session_scene_log[-100:]

            try:
                await self._extract_highlight(scene_summary)
            except Exception as e:
                print(f"   [Highlight] Extraction failed: {e}")

    async def _extract_highlight(self, scene_summary: str):
        """One Claude Opus call: is anything in the recent scenes memorable?"""
        recent = self.session_scene_log[-4:]
        context_lines = []
        for entry in recent:
            rel_time = int((time.time() - entry["time"]) / 60)
            context_lines.append(f"[~{rel_time}min ago] {entry['summary']}")
        context = "\n\n".join(context_lines)

        system_prompt = (
            "You are an emotional and narrative archivist for an AI companion named Kira "
            "who watches media with her friend Jonny. Your job: identify any moment in the "
            "recent scenes that is genuinely memorable \u2014 funny, emotional, shocking, beautiful, "
            "character-defining, or otherwise worth preserving as a long-term memory.\n\n"
            "Reference characters by name. Be specific about WHAT happened, not vague vibes. "
            "If nothing in the recent scenes meets the bar, output exactly: NONE\n\n"
            "Otherwise output exactly two lines:\n"
            "HIGHLIGHT: <one specific sentence with character names and what happened>\n"
            "KIRA_TAKE: <one short sentence \u2014 how Kira would react to this moment, in her voice>"
        )

        user = (
            f"Activity: {self.current_activity}\n\n"
            f"Recent scenes:\n{context}\n\n"
            f"Identify any standout moment, or NONE."
        )

        response = await self.ai_core.claude_inference(
            messages=[{"role": "user", "content": user}],
            system_prompt=system_prompt,
            max_tokens=200,
        )

        if not response or "NONE" in response.upper()[:20]:
            return

        highlight = ""
        take = ""
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("HIGHLIGHT:"):
                highlight = stripped[len("HIGHLIGHT:"):].strip()
            elif stripped.upper().startswith("KIRA_TAKE:"):
                take = stripped[len("KIRA_TAKE:"):].strip()

        if highlight:
            self.session_highlights.append({"highlight": highlight, "take": take})
            self.memory.add_highlight(
                activity=self.current_activity or "unspecified",
                highlight=highlight,
                kira_take=take,
            )

    async def _generate_session_summary(self):
        """When a media session ends (activity changes or bot shuts down), generate
        a single paragraph recap and store it as long-term memory."""
        if not self.session_scene_log and not self.session_highlights:
            return

        activity = self.current_activity or "the session"
        scene_count = len(self.session_scene_log)
        duration_min = 0
        if scene_count > 1:
            duration_min = int(
                (self.session_scene_log[-1]["time"] - self.session_scene_log[0]["time"]) / 60
            )

        highlights_text = "\n".join(
            f"- {h['highlight']} (Kira: {h['take']})" if h.get("take") else f"- {h['highlight']}"
            for h in self.session_highlights[-12:]
        ) or "(no highlights captured)"

        last_scene = self.session_scene_log[-1]["summary"] if self.session_scene_log else ""

        system_prompt = (
            "You are Kira, summarizing a session you just shared with Jonny. Write a single "
            "paragraph (4-6 sentences) recapping what you two watched/played together. "
            "Reference characters by name, mention specific plot beats, and end with which moment "
            "stuck with you most. This is going into long-term memory \u2014 be specific and personal, "
            "not generic. Write in first person as Kira."
        )

        user = (
            f"Activity: {activity}\n"
            f"Approximate session duration: {duration_min} minutes\n"
            f"Final scene state: {last_scene}\n\n"
            f"Highlights captured during the session:\n{highlights_text}\n\n"
            f"Write Kira's session recap paragraph."
        )

        try:
            summary = await self.ai_core.claude_inference(
                messages=[{"role": "user", "content": user}],
                system_prompt=system_prompt,
                max_tokens=400,
            )
            if summary:
                self.memory.add_session_summary(activity=activity, summary=summary)
                print(f"   [Session] Recap stored for: {activity}")
        except Exception as e:
            print(f"   [Session] Summary generation failed: {e}")
        finally:
            # Reset session state for the next activity
            self.session_scene_log = []
            self.session_highlights = []

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