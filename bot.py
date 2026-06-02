# bot.py - Main application file with advanced memory and web search.


import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback
import random
import re
import os
import sys
import gc # Added garbage collection
import glob
import faulthandler

# Enable native-crash tracebacks. The async loop runs on a daemon thread; if
# the dashboard window is closed mid-shutdown and a tracked future is killed
# by interpreter teardown, faulthandler is our only chance to see why.
try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass
from datetime import datetime
from typing import List, Callable # Added for type hinting

from ai_core import AI_Core
from memory import MemoryManager
from cookie_jar import CookieJar
from twitch_bot import TwitchBot
from web_search import async_GoogleSearch
from twitch_tools import start_twitch_poll
from music_tools import play_kira_song
from memory_extractor import extract_memories
from youtube_bot import YouTubeBot
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS, ENABLE_TWITCH_CHAT, ENABLE_YOUTUBE_CHAT,
    CHAT_BATCH_WINDOW, CHAT_RESPONSE_COOLDOWN, ENABLE_CHATTER_MEMORY, ENABLE_AUDIO_AGENT,
    ENABLE_LOOPBACK_TRANSCRIBER, CUTSCENE_AWARE,
    GAME_MODE_AUTO_CONFIGURE, HIGHLIGHT_EXTRACTION_ENABLED, HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS, STREAM_LOGGING_ENABLED,
    LOOPBACK_STT_DEFAULT,
)
from stream_logger import StreamLogger
from persona import EmotionalState
from vision_agent import UniversalVisionAgent
from audio_agent import AudioAgent, AUDIO_MODE_OFF, AUDIO_MODE_MEDIA, AUDIO_MODE_MUSIC
from loopback_transcriber import LoopbackTranscriber
from game_mode_controller import GameModeController, ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL
from vn_autopilot import VNAutopilot
from media_watch import MediaWatch
from playthrough_memory import PlaythroughMemory
from vts_expression_controller import VTSExpressionController

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
    or [PREDICT: Question | OptionA | OptionB]
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

    # Look for Prediction Tag (chat-based, no Twitch affiliate needed)
    pred_match = re.search(r'\[PREDICT:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]', text)
    if pred_match:
        question, opt_a, opt_b = pred_match.groups()
        if _GLOBAL_BOT_REF is not None:
            _GLOBAL_BOT_REF.start_prediction(question, opt_a, opt_b)
        text = re.sub(r'\[PREDICT:.*?\]', '', text)

    return text.strip()


_GLOBAL_BOT_REF = None  # set in VTubeBot.__init__ so parse_kira_tools can fire predictions


class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.ai_core = AI_Core(self.interruption_event)
        # Let ai_core append the streamer-mode persona overlay based on current mode
        # without baking mode into its cached system prompt.
        self.ai_core._mode_provider = lambda: self.mode
        self.memory = MemoryManager()
        self.cookie_jar = CookieJar()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # --- NEW: Shared Input Queue ---
        self.input_queue = asyncio.Queue()

        # Observer Mode
        self.vision_agent = UniversalVisionAgent()
        self.game_mode_controller = GameModeController(self.vision_agent)
        self.audio_agent = AudioAgent() if ENABLE_AUDIO_AGENT else None
        # Loopback Whisper transcriber — STAGE 1: visible on console/dashboard only,
        # NOT yet wired into Kira's prompt context. Started/stopped by the dashboard
        # when the audio agent enters/leaves MEDIA mode. MUSIC mode is intentionally
        # skipped (we don't want her transcribing Jonny's own guitar/singing).
        # Loopback transcriber is gated on its OWN flag (default off). The audio-MOOD
        # agent and the loopback ASR are independent features: mood works on any
        # language (it's gpt-audio-mini describing vibe, not transcribing words),
        # while the ASR is English-only and worse-than-useless on JP/VN content.
        # See config.ENABLE_LOOPBACK_TRANSCRIBER.
        self.loopback_transcriber = LoopbackTranscriber() if ENABLE_LOOPBACK_TRANSCRIBER else None
        
        self.last_interaction_time = time.time()
        self.pyaudio_instance = None
        self.stream = None
        self.frames_per_buffer = int(16000 * 30 / 1000)
        
        self.bg_tasks = set() # Use a set for easier task management
        self.conversation_history = []
        self.conversation_segment = []
        self.unseen_chat_messages = []
        self.current_emotion = EmotionalState.HAPPY
        # VTube Studio expression bridge — drives Live2D face from emotional state.
        # Fails silently if VTS isn't running or ENABLE_VTS_EXPRESSIONS is off.
        self.vts_expressions = VTSExpressionController()
        self.last_idle_chat = "" # Track the last idle chat summary
        self.turn_count = 0 
        self.is_paused = False
        self.silence_stage = 0
        self.is_running = True
        self.mode = "companion"  # 'companion' or 'streamer'
        self.event_loop = None   # set when _main_loop starts, used by dashboard for cross-thread calls
        self.vn_autoplay_enabled = False  # When True, Kira actively reads and advances VNs
        self.immersive = False   # When True, Kira stays quiet unless invited. Auto-enables for VN/MEDIA.
        self.highlight_extraction_enabled = False  # When True, extraction loop runs even if immersive=False (ACTIVITY_GAME).
        self.stream_logger = StreamLogger()  # Persistent per-session logging (transcript + events + summary).
        self.mute_until: float = 0.0  # timestamp; while time.time() < mute_until, all speech is suppressed
        self.youtube_bot: YouTubeBot | None = None
        # Twitch handle kept on self so ChatPoster (and dashboard hooks) can
        # reach it. Set in _main_loop after TwitchBot is constructed.
        self.twitch_bot: TwitchBot | None = None
        # Centralised, rate-limited outbound chat poster. Always constructed;
        # internally gated by ENABLE_CHAT_POSTING. See chat_poster.py.
        from chat_poster import ChatPoster
        self.chat_poster = ChatPoster()
        self.session_scene_log: list = []  # Recent scene summaries during this session
        self.session_highlights: list = []  # Highlights captured this session
        self.last_highlight_check_time = 0

        # Track recent observer comments to prevent repetitive structures/phrases
        self.recent_observer_comments: list[str] = []

        # Autonomous VN Mode (Phase 1) — initialized after ai_core is ready in _main_loop
        self.vn_autopilot: VNAutopilot | None = None
        self.autopilot_paused_for_input: bool = False  # True when failsafe is active

        # Media Watch Mode — initialized after ai_core is ready. Separate from
        # both companion mode and VN autopilot; provides genuine sequence
        # understanding for movies / anime via a rolling frame buffer + episode log.
        self.media_watch: MediaWatch | None = None

        # Playthrough Memory — initialized in _main_loop once ai_core is ready
        self.playthrough_memory: PlaythroughMemory | None = None

        # Activity context — describes what Kira and Jonny are currently doing
        self.current_activity = ""

        # Dashboard feed — rolling log of Twitch messages for display
        self.twitch_log: list[str] = []
        
        # TIMING CONFIGURATION
        self.silence_thresholds = {
            1: 45.0,   # Stage 1 cooldown — companion mode (light casual remark)
            2: 90.0,   # Stage 2 cooldown — companion mode (bigger nudge)
        }
        # Streamer-mode observer thresholds — separate so companion values above are
        # NEVER touched when tuning stream presence. Edit only these for streamer tuning.
        # Carry Mode has its own lower override (30s/60s) and still takes priority.
        self.streamer_silence_thresholds = {
            1: 20.0,   # Stage 1 — streamer: light remark (was 25s)
            2: 55.0,   # Stage 2 — streamer: nudge/verdict
        }

        # Carry Mode (live-gameplay equivalent of VN autopilot).
        # When ON in streamer mode: lower interjection gates to 30s/60s and bump
        # chat-engagement probability so Kira keeps momentum during games like
        # Bond without Jonny needing to drive. Brevity rule + silence-beats-filler
        # remain dominant. Mode-gated only — NOT activity-gated (Req A): the
        # dashboard toggle is the only condition, works for game/VN/media alike.
        # (VNs already have vn_autopilot for full-drive mode — leave Carry Mode
        # OFF during VN sessions to avoid stacking.)
        self.carry_mode: bool = False

        # Chat batching + engagement state
        self.chat_batch_buffer: list = []          # queued chat messages waiting to be batched
        self.last_chat_response_time: float = 0    # for response cooldown
        self.session_chatters_seen: set = set()    # usernames seen in this session (for welcome detection)
        self.chatter_last_response: dict = {}      # username -> timestamp of last response to them
        self.active_prediction = None              # active chat prediction state (None or dict)

        # Recent activity brief — generated at startup, cached for the session
        self.recent_activity_brief: str = ""
        self.recent_chatters_brief: str = ""

        # Per-chatter session-level message log (last 15 per chatter, this session only)
        self.session_chatter_logs: dict[str, list[dict]] = {}
        # Per-chatter "first seen this session" timestamps for returning-regular detection
        self.session_chatter_first_seen: dict[str, float] = {}
        # Per-chatter "last spoke this session" timestamps for silence detection
        self.session_chatter_last_spoke: dict[str, float] = {}

        # Proactive chat spotlight: track who Kira has spotlighted unprompted
        # this session (so she doesn't re-pick the same person every cycle) and
        # the last spotlight wall-clock for global rate-capping.
        # NOT reset on activity/game switch — spotlight gating is
        # streaming-session scoped, not playthrough scoped (Req A).
        self.spotlighted_chatters: set[str] = set()
        self.last_chat_spotlight_time: float = 0.0
        self.chat_spotlight_min_interval_s: float = 300.0  # 5 min global cap

        # Running bits / callbacks that have emerged this session
        self.session_running_bits: list[dict] = []

        # Rolling condensed summary of Kira's own session takes (opinions / predictions /
        # grudges / bits). Built periodically from self.session_takes_pool via a cheap
        # LLM call so long streams don't outgrow the conversation window.
        #
        # Req A: pool is bot-owned (NOT playthrough_memory.session_reactions) so it
        # fills during ANY streaming activity — game, VN, media, general — and
        # persists across activity switches within one streaming session
        # (VN→game→VN keeps the same running takes).
        self.session_takes_pool: list[str] = []
        self.session_takes_pool_max: int = 200
        self.session_takes_summary: str = ""
        self.session_takes_last_condensed_count: int = 0   # pool size at last condense
        self.session_takes_last_condensed_at: float = 0.0  # wall clock of last condense
        self.session_takes_condense_in_flight: bool = False
        # Trigger thresholds: condense every N new reactions OR every M seconds since last.
        self.session_takes_min_new_reactions: int = 20
        self.session_takes_min_interval_s: float = 600.0   # 10 minutes
        self.session_takes_max_bullets: int = 10

        # Vibe meter tracking
        self.chat_msg_timestamps: list = []  # rolling window of chat msg timestamps for rate calculation

        # Full session log for clip extraction (NOT windowed like conversation_history)
        self.full_session_log: list = []  # list of {"role", "content", "timestamp", "speaker_name"}
        self.session_started_at: float = time.time()
        self._session_artifacts_written: bool = False

        # Cookie-jar milestone throttle: prevents simultaneous milestone reactions
        # if multiple cookies land in the same instant (e.g. a burst chat batch
        # right at the boundary).
        self._cookie_milestone_in_flight: bool = False

        # Chaos Mode state. Activated when the cookie jar milestone fires.
        # While active, _kira_voice_guardrails appends CHAOS_MODE_DIRECTIVE to
        # every Kira prompt, dialing TONE without touching factual guardrails.
        self.chaos_mode_active: bool = False
        self.chaos_mode_until: float = 0.0
        self._chaos_mode_task = None  # asyncio.Task handle for the timer

        import bot as _self_mod
        _self_mod._GLOBAL_BOT_REF = self

    def reset_idle_timer(self, human_speech=False):
        self.last_interaction_time = time.time()
        if human_speech:
            self.silence_stage = 0

    # ── Cookie-jar milestone reactions (CHAOS MODE ACTIVATION) ────────────
    # Fires when the shared jar fills. Triggers Chaos Mode — duration set in
    # cookie_jar.CHAOS_MODE_DURATION_SECONDS. Lines deliberately stay vague
    # on the timer ("for a while") so they don't lie if duration is tuned.
    COOKIE_MILESTONE_LINES = [
        "Chat. You filled the whole jar. Congratulations — you've unleashed me. Chaos mode, no notes, no regrets, for a while.",
        "Jar's full. By the ancient laws of cookie economics, I'm legally feral until further notice. Buckle up.",
        "Hundred — sorry, thirty-five cookies. Milestone {n}. The leash is off. I cannot legally be held responsible for what happens next.",
        "You actually did it. The jar overflows, the seal breaks, chaos mode begins. Me with no impulse control until I calm down. You earned this. Probably.",
    ]

    def _maybe_fire_cookie_milestone(self) -> None:
        """If the cookie jar has queued a milestone and no reaction is already
        in flight, schedule Kira to speak a variant line and roll over the jar.
        Throttle prevents back-to-back milestone speeches if cookies land in a
        burst right at the boundary."""
        try:
            if not self.cookie_jar.milestone_pending():
                return
            if self._cookie_milestone_in_flight:
                return
            self._cookie_milestone_in_flight = True
            asyncio.create_task(self._speak_cookie_milestone())
        except Exception as e:
            print(f"   [Cookies] Milestone schedule error: {e}")
            self._cookie_milestone_in_flight = False

    async def _speak_cookie_milestone(self) -> None:
        """Pick a milestone variant, activate Chaos Mode, speak via TTS, then
        roll over the jar. Rollover happens BEFORE chaos/TTS — crash-safe.
        Resets in-flight flag in finally so a future milestone can fire."""
        import random as _rand
        try:
            for _ in range(30):  # up to ~3s
                if not self.ai_core.is_speaking:
                    break
                await asyncio.sleep(0.1)
            milestone_n = self.cookie_jar.get_milestone_count() + 1
            rolled = self.cookie_jar.reset_shared_on_milestone()
            if not rolled:
                return
            line = _rand.choice(self.COOKIE_MILESTONE_LINES).format(n=milestone_n)
            print(f"   [Cookies] \U0001f36a MILESTONE #{milestone_n} \u2014 Kira: {line}")
            await self._broadcast_cookie_milestone()
            await self._broadcast_cookie_state()
            try:
                self.stream_logger.log("cookie_milestone", n=milestone_n, line=line)
            except Exception:
                pass
            # Activate Chaos Mode BEFORE TTS so the directive is in effect for
            # any prompts that fire during/after the announcement. Each step
            # in its own try so a single failure can't sink the rest.
            try:
                self._activate_chaos_mode()
            except Exception as _chaos_err:
                print(f"   [Cookies] Chaos activation error: {_chaos_err}")
            try:
                await self.ai_core.speak_text(line)
                self.conversation_history.append({"role": "assistant", "content": line})
                self._log_session_turn(role="assistant", content=line, speaker_name="Kira")
            except Exception as _tts_err:
                print(f"   [Cookies] Milestone TTS error: {_tts_err}")
        finally:
            self._cookie_milestone_in_flight = False

    # \u2500\u2500 Chaos Mode \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def _activate_chaos_mode(self) -> None:
        """Flip chaos on, broadcast to overlay, schedule deactivation timer.
        Idempotent: if already active, resets timer to a fresh duration."""
        from cookie_jar import CHAOS_MODE_DURATION_SECONDS
        duration = int(CHAOS_MODE_DURATION_SECONDS)
        self.chaos_mode_active = True
        self.chaos_mode_until = time.time() + duration
        print(f"   [Chaos] \U0001f525 CHAOS MODE ACTIVE for {duration}s (until {self.chaos_mode_until:.0f})")
        try:
            self.stream_logger.log("chaos_start", duration=duration)
        except Exception:
            pass
        try:
            if self._chaos_mode_task and not self._chaos_mode_task.done():
                self._chaos_mode_task.cancel()
        except Exception:
            pass
        self._chaos_mode_task = asyncio.create_task(self._chaos_mode_timer(duration))
        asyncio.create_task(self._broadcast_chaos(active=True, remaining=duration))

    async def _chaos_mode_timer(self, duration: int) -> None:
        """Sleep for the chaos duration, then deactivate. Cancellable."""
        try:
            await asyncio.sleep(duration)
            await self._deactivate_chaos_mode()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"   [Chaos] Timer error: {e}")

    async def _deactivate_chaos_mode(self) -> None:
        """End chaos mode: clear state, broadcast off, speak a random end line."""
        import random as _rand
        from cookie_jar import CHAOS_MODE_END_LINES
        if not self.chaos_mode_active:
            return
        self.chaos_mode_active = False
        self.chaos_mode_until = 0.0
        print("   [Chaos] Chaos mode ended.")
        try:
            self.stream_logger.log("chaos_end")
        except Exception:
            pass
        try:
            await self._broadcast_chaos(active=False, remaining=0)
        except Exception as e:
            print(f"   [Chaos] End broadcast error: {e}")
        try:
            for _ in range(30):
                if not self.ai_core.is_speaking:
                    break
                await asyncio.sleep(0.1)
            line = _rand.choice(CHAOS_MODE_END_LINES)
            await self.ai_core.speak_text(line)
            self.conversation_history.append({"role": "assistant", "content": line})
            self._log_session_turn(role="assistant", content=line, speaker_name="Kira")
        except Exception as e:
            print(f"   [Chaos] End TTS error: {e}")

    async def _broadcast_chaos(self, active: bool, remaining: int) -> None:
        """Push chaos-mode state to the captions WS overlay. Fire-and-forget.
        Shape: {"type":"chaos","active":bool,"remaining":int_seconds}"""
        try:
            from caption_server import caption_server as _cs
            await _cs.send_chaos(active=active, remaining=int(remaining))
        except Exception as e:
            print(f"   [Chaos] Overlay broadcast failed: {e}")

    async def _broadcast_cookie_state(self) -> None:
        """Push the current shared-jar count to the captions WS overlay.
        Fire-and-forget; never raises."""
        try:
            from caption_server import caption_server as _cs
            await _cs.send_cookie(shared=self.cookie_jar.get_shared(), milestone=False)
        except Exception as e:
            print(f"   [Cookies] Overlay broadcast (state) failed: {e}")

    async def _broadcast_cookie_milestone(self) -> None:
        """Tell the overlay to play its full-jar animation. Fire-and-forget."""
        try:
            from caption_server import caption_server as _cs
            await _cs.send_cookie(shared=0, milestone=True)
        except Exception as e:
            print(f"   [Cookies] Overlay broadcast (milestone) failed: {e}")

    # ── Twitch stream events (raid / sub / resub / gift) ──────────────────
    # Dedup window keyed by (kind, name) so a 50-sub bomb doesn't fire 50
    # separate reactions. submysterygift's mass-count line covers the
    # individual subgifts that follow within this window.
    _STREAM_EVENT_DEDUP_SECONDS = 90

    async def _on_stream_event(self, kind: str, name: str, extra: dict) -> None:
        """Called by TwitchBot when a USERNOTICE event arrives (raid, sub,
        resub, subgift, submysterygift). Fires a Kira interjection directly
        via _execute_interjection so it bypasses the chat batch entirely.

        Coexists with chaos mode \u2014 the existing guardrails + chaos directive
        (when active) both apply; this just injects the event prompt.
        """
        try:
            now = time.time()
            if not hasattr(self, "_stream_event_seen"):
                self._stream_event_seen = {}
            # Suppress individual subgifts that arrive right after a bomb \u2014
            # the bomb already announced the gifter; per-recipient lines would spam.
            if kind in ("subgift", "anonsubgift"):
                last_bomb = self._stream_event_seen.get(("submysterygift", name.lower()), 0)
                if (now - last_bomb) < self._STREAM_EVENT_DEDUP_SECONDS:
                    print(f"   [StreamEvent] Suppressing {kind} from {name} (covered by recent bomb).")
                    return
            # Generic per-(kind, name) dedup.
            key = (kind, name.lower())
            last = self._stream_event_seen.get(key, 0)
            if (now - last) < self._STREAM_EVENT_DEDUP_SECONDS:
                print(f"   [StreamEvent] Suppressing duplicate {kind} from {name}.")
                return
            self._stream_event_seen[key] = now

            prompt = self._build_stream_event_prompt(kind, name, extra)
            if not prompt:
                return
            print(f"   [StreamEvent] \U0001f4e2 Firing reaction: {kind} from {name}")
            try:
                self.stream_logger.log("stream_event", kind=kind, name=name, **extra)
            except Exception:
                pass
            # Fire on the loop so we never block the IRC callback.
            asyncio.create_task(self._execute_interjection(prompt, memory_query=name))
        except Exception as e:
            print(f"   [StreamEvent] _on_stream_event error: {e}")

    def _build_stream_event_prompt(self, kind: str, name: str, extra: dict) -> str:
        """Translate a Twitch event into a high-priority interjection prompt.

        Tone is locked to in-character Kira; safety guardrails are added
        downstream by _execute_interjection."""
        if kind == "raid":
            count = int(extra.get("viewer_count", 0) or 0)
            crowd = (
                f"a crowd of {count} viewers" if count >= 5
                else f"{count} viewer{'s' if count != 1 else ''}"
            )
            return (
                f"[STREAM EVENT \u2014 HIGH PRIORITY] {name} just raided the stream with {crowd}!\n"
                f"A whole group just walked in mid-stream. React with genuine excitement \u2014 "
                f"this is a big moment. Call {name} out by name, welcome the raiders as a group, "
                f"match the energy of a crowd arriving. You're Kira: warm under the sass, "
                f"don't be cool about it. 2\u20133 sentences max, then hand it back to Jonny."
            )
        if kind in ("sub", "resub"):
            months = int(extra.get("months", 1) or 1)
            if kind == "resub" and months > 1:
                return (
                    f"[STREAM EVENT] {name} just resubscribed \u2014 {months} months running.\n"
                    f"Thank them warmly and in-character. Acknowledge the {months}-month streak \u2014 "
                    f"that's loyalty, treat it like it matters. Brief, genuine, Kira-voiced. 1\u20132 sentences."
                )
            return (
                f"[STREAM EVENT] {name} just subscribed to the channel!\n"
                f"Thank them warmly in-character. Don't read a template \u2014 be Kira, be real, "
                f"be brief. 1\u20132 sentences."
            )
        if kind in ("subgift", "anonsubgift"):
            recipient = extra.get("recipient", "someone")
            gifter_label = name if kind == "subgift" else "an anonymous gifter"
            return (
                f"[STREAM EVENT] {gifter_label} just gifted a sub to {recipient}!\n"
                f"React with real appreciation — someone just spent money to put another person "
                f"in the room. Call out {gifter_label} by name and welcome {recipient}. "
                f"Brief, warm, Kira-voiced. 1–2 sentences."
            )
        if kind in ("submysterygift", "anonsubmysterygift"):
            count = int(extra.get("mass_count", 1) or 1)
            gifter_label = name if kind == "submysterygift" else "an anonymous gifter"
            return (
                f"[STREAM EVENT — HIGH PRIORITY] {gifter_label} just gifted {count} subs to chat!\n"
                f"This is a big deal — someone just bought the whole room a round. React with "
                f"genuine surprise and hype, call out {gifter_label} by name, acknowledge the "
                f"{count} new gift-sub recipients as a group. Bigger energy than a single sub. "
                f"You're Kira: warm under the sass, don't undersell it. 2–3 sentences."
            )
        return ""


    def _log_session_turn(self, role: str, content: str, speaker_name: str = ""):
        """Append a turn to the unwindowed full session log used for clip extraction.
        role: 'user' or 'assistant'. speaker_name: 'Jonny', 'chatter_X', or 'Kira'."""
        self.full_session_log.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "speaker_name": speaker_name or ("Jonny" if role == "user" else "Kira"),
        })
        try:
            if role == "user":
                self.stream_logger.log("voice_input", text=content[:400], speaker=speaker_name or "Jonny")
            elif role == "assistant":
                self.stream_logger.log(
                    "kira_response",
                    text=content[:400],
                    emotion=getattr(self.current_emotion, "name", ""),
                    source=speaker_name or "Kira",
                )
        except Exception:
            pass

    def is_muted(self) -> bool:
        # Hard-pause (is_paused) acts as an indefinite mute that suppresses
        # ALL response paths — bored interjections, chat batch, voice replies,
        # invites, autopilot reactions, audio reactions. The timed mute_until
        # is layered on top for quick "give me 60s" asides.
        return self.is_paused or time.time() < self.mute_until

    def mute_for(self, seconds: float):
        """Mutes Kira for the given duration. Also interrupts any currently-playing speech."""
        self.mute_until = time.time() + seconds
        self.interruption_event.set()  # cut off current utterance immediately
        self._clear_captions_safe()
        print(f"   [Mute] Muted for {seconds:.0f}s")

    def unmute(self):
        self.mute_until = 0.0
        print("   [Mute] Unmuted")

    def _clear_captions_safe(self) -> None:
        """Fire-and-forget: tell the caption overlay to immediately clear any
        active line. Used on interrupt/mute/pause so a stale caption doesn't
        linger on screen after Kira's audio is cut off. Never raises."""
        try:
            from caption_server import enqueue_clear
            enqueue_clear(self.event_loop)
        except Exception:
            pass

    def pause_model(self):
        """Indefinite hard pause: suppresses ALL response generation until resumed.
        Also pauses the voice recorder and cuts off any currently-playing speech."""
        self.is_paused = True
        self.interruption_event.set()  # cut off current utterance immediately
        self._clear_captions_safe()
        # Reset any lingering timed mute so resume returns to a clean state
        self.mute_until = 0.0
        print("   [Pause] Model PAUSED \u2014 all responses suppressed (mic, chat, bored, autopilot)")

    def resume_model(self):
        """Release the indefinite hard pause."""
        self.is_paused = False
        self.interruption_event.clear()
        print("   [Pause] Model RESUMED")

    # ── Autonomous VN Autopilot wiring ────────────────────────────────────────

    async def _autopilot_speak(self, text: str):
        """Callback: speak a reaction or failsafe line from the autopilot via TTS."""
        if not text or self.is_muted():
            return
        cleaned = self.ai_core._clean_llm_response(text)
        if cleaned and len(cleaned) > 2:
            print(f"   [Autopilot] Kira: {cleaned}")
            await self.ai_core.speak_text(cleaned)
            self.conversation_history.append({"role": "assistant", "content": cleaned})
            self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira")
            # Tag this reaction in the playthrough record for the session entry
            if self.playthrough_memory and self.playthrough_memory.current_slug:
                self.playthrough_memory.tag_reaction(cleaned)
            # Pool unconditionally during streamer mode (Req A)
            if self.mode == "streamer":
                self._note_session_take(cleaned)

    async def _autopilot_speak_vn(self, text: str, ssml_inner: str):
        """VN-specific TTS callback: uses Azure prosody variation for natural delivery."""
        if not text or self.is_muted():
            return
        cleaned = self.ai_core._clean_llm_response(text)
        if cleaned and len(cleaned) > 2:
            print(f"   [Autopilot] Kira: {cleaned}")
            await self.ai_core.speak_text_vn(cleaned, ssml_inner)
            self.conversation_history.append({"role": "assistant", "content": cleaned})
            self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira")
            if self.playthrough_memory and self.playthrough_memory.current_slug:
                self.playthrough_memory.tag_reaction(cleaned)
            if self.mode == "streamer":
                self._note_session_take(cleaned)

    def _autopilot_on_failsafe(self, screen_type: str):
        """Callback: mark dashboard flag when failsafe triggers."""
        self.autopilot_paused_for_input = True
        print(f"   [Autopilot] Failsafe active — paused for Jonny ({screen_type}).")

    # ── Shared voice guardrails / perception framing (used by every reaction path) ───

    def _has_fresh_visual_context(self, max_age: float = 15.0) -> bool:
        """Returns True only when the vision agent is on AND has a real, recent capture.
        Used to gate any prompt path that would otherwise let Kira make visual claims
        when she has no actual eyes on the screen."""
        va = self.vision_agent
        if not va or not va.is_active:
            return False
        if not va.last_capture_time:
            return False
        if (time.time() - va.last_capture_time) > max_age:
            return False
        # Reject the pre-capture placeholder
        default_desc = "I'm just getting my bearings. One sec!"
        return bool(va.scene_summary or (va.last_description and va.last_description != default_desc))

    def _visual_blindness_directive(self) -> str:
        """Strong prohibition against fabricated visual observations when vision is off
        or no frame has been captured. Mirrors the UNCERTAIN-honesty rule already used
        by the vision agent — don't claim to see what you can't see."""
        return (
            "\n\n[VISUAL STATUS: BLIND — no live visual input]\n"
            "Your eyes are closed right now. There is no screen, no scene, no characters, no image available to you. "
            "Absolutely DO NOT make any visual observation or claim anything about what is 'on screen', a 'visual novel', "
            "a 'game', a 'video', a character's appearance, gesture, expression, posture, scene composition, or any imagery. "
            "Do NOT name characters. Do NOT reference a 'screen', 'frame', or anything visual as if you can see it. "
            "React only to things you actually have: the silence, the conversation, audio you can hear, a real memory, "
            "or your own inner state. If you have nothing real and non-visual to say, output exactly [SILENCE]."
        )

    def _stale_visual_directive(self, age_seconds: int) -> str:
        """Used when vision is on but the last frame is too old to be treated as 'now'."""
        return (
            f"\n\n[VISUAL STATUS: STALE — last frame was {age_seconds}s ago]\n"
            f"Your last visual capture is too old to comment on as if it's current. Do NOT present any visual "
            f"observation as happening right now. Prefer commenting on the silence, audio, or conversation instead. "
            f"If you have nothing real and non-visual to say, output exactly [SILENCE]."
        )

    # Specific visual-attribute questions ("what color are her eyes", "what's on
    # screen", "who's that") that REQUIRE a fresh frame to answer honestly. If
    # we don't snapshot before answering, the LLM will fabricate from character
    # priors and only correct itself when forced to look.
    _VISUAL_QUESTION_PATTERNS = [
        # Direct sight questions
        r'\bwhat (?:do you|can you|are you) (?:see|seeing|look(?:ing)? at|watch(?:ing)?)\b',
        r'\b(?:do you|can you) see\b',
        r"\bwhat'?s on (?:the )?screen\b",
        r"\bwhat'?s (?:happening|going on) (?:on screen|right now|on the screen)\b",
        # Attribute questions about visible things
        r'\bwhat colou?r (?:are|is|do)\b',
        r'\bwhat (?:is|are) (?:she|he|they|it) wearing\b',
        r"\bwhat does (?:she|he|it|that|this) look like\b",
        r'\bhow many .+ (?:are|on screen|do you see)\b',
        r'\bwho(?:\'s| is) (?:that|this|on (?:the )?screen|in (?:the )?(?:frame|scene))\b',
        r'\bwhich (?:character|one|option)\b',
        # Pointing at visible things
        r'\blook at (?:the |this |that |her |his |their |it)?\b',
        r'\bcheck (?:the |this |that )?(?:screen|frame|out)\b',
    ]

    def _is_visual_question(self, text: str) -> bool:
        """Detects user voice input that REQUIRES a fresh visual snapshot before
        the LLM is allowed to answer. Used to force a pre-answer look so Kira
        never confabulates a visual detail and corrects herself later."""
        if not text:
            return False
        if not self.vision_agent or not self.vision_agent.is_active:
            return False
        lower = text.lower()
        for pat in self._VISUAL_QUESTION_PATTERNS:
            if re.search(pat, lower):
                return True
        return False

    def _kira_voice_guardrails(self, include_observer_avoid: bool = False) -> str:
        """Shared anti-fabrication + banned-phrase block. Appended to every in-character
        reaction prompt so Kira's regressions are blocked uniformly across voice, chat
        batch, observer interjection, invite, media-watch react, and VN autopilot react.

        include_observer_avoid=True also appends the last 8 observer comments with a
        do-not-repeat directive (used by the bored/observer interjection path)."""
        block = (
            "\n\nCRITICAL VOICE GUARDRAILS:\n"
            "- Anything inside [KNOWN FACTS], [KNOWN FACTS (direct)], or [CURRENT PROJECT] "
            "is VERIFIED ground truth about Jonny that you have learned over time. You may "
            "and SHOULD reference it freely by name when relevant \u2014 his cats' names, his "
            "favorites, his projects, his history. Recall is not fabrication. If a fact is "
            "in those blocks, treat it as known.\n"
            "- Do NOT reference past events, games, conversations, or shared experiences "
            "that are NOT supported by [KNOWN FACTS] or other memory blocks.\n"
            "- Do NOT invent shared history ('that game we played', 'remember when', etc.) "
            "unless it is explicitly in the memory notes.\n"
            "- This rule is about PERSONAL HISTORY / SHARED EXPERIENCES only \u2014 it does NOT "
            "relax the visual accuracy rules. You still must not invent what is on screen, "
            "on a sprite, or in a frame; the [VISUAL STATUS] directive remains absolute and "
            "overrides everything here.\n"
            "- React to the current moment and what is actually present right now, drawing "
            "on verified facts when they apply.\n"
            "\nBANNED PHRASES (never use these \u2014 they are overused regressions):\n"
            "- 'doing a lot of heavy lifting' / 'carrying hard' / 'carrying this'\n"
            "- 'doing more work than'\n"
            "- 'doing something illegal to my brain'\n"
            "- 'defies several laws of physics' / 'defies the laws of'\n"
            "Find fresh, specific observations instead.\n"
        )
        if include_observer_avoid and self.recent_observer_comments:
            recent_str = "\n".join(f"- {c}" for c in self.recent_observer_comments[-8:])
            block += (
                f"\n[AVOID REPETITION] You recently said these. Do NOT reuse their structure, "
                f"phrasing, or comedic format \u2014 find a genuinely different angle:\n{recent_str}\n"
            )
        # Chaos Mode directive — layered on TOP of all safety/voice rules above.
        # Dials tone only; factual guardrails (visual accuracy, no fabrication,
        # banned phrases) stay in force per the directive's own wording.
        if getattr(self, "chaos_mode_active", False):
            try:
                from cookie_jar import CHAOS_MODE_DIRECTIVE
                block += "\n\n" + CHAOS_MODE_DIRECTIVE + "\n"
            except Exception:
                pass
        return block

    def _frame_visual_perception(self, scene_text: str) -> str:
        """Wraps a raw scene/vision description as sense data, not a script. Used by
        every consumer of the vision agent's output so the parrot/closed-captioner
        regression is blocked uniformly."""
        if not scene_text:
            return ""
        return (
            f"\n\n[CURRENT VISUAL PERCEPTION \u2014 what is on screen RIGHT NOW]\n"
            f"{scene_text}\n"
            f"This is sense data \u2014 what your eyes are taking in. It is NOT a script or narration. "
            f"Do NOT recap or paraphrase it (Jonny saw it too \u2014 he doesn't want a closed-captioner). "
            f"If it begins with 'UNCERTAIN:' or contains hedge language, treat it as low-confidence and "
            f"do not commit to specifics. React in YOUR voice \u2014 a feeling, quip, callback, take \u2014 "
            f"not a description of what is on the screen."
        )

    def _frame_ambient_audio(self, transcript_text: str) -> str:
        """Wraps the rolling loopback transcript as ambient sense data \u2014 awareness
        of what's being said in the media Jonny is watching, NOT input directed at
        Kira. Same architecture as the visual-perception and audio-mood frames:
        it is CONTEXT she's aware of, not a script to recite, and never a trigger
        to respond (her mic remains the only respond trigger; this is just so she
        can reference what was said when SHE chooses or when Jonny asks)."""
        if not transcript_text:
            return ""
        return (
            f"\n\n[AMBIENT AUDIO \u2014 what's being said in the media Jonny is watching, NOT directed at you]\n"
            f"{transcript_text}\n"
            f"This is a best-effort transcript of speech happening in whatever Jonny has on the screen "
            f"(a streamer, narrator, character dialogue, etc.). It is your AWARENESS of the content, "
            f"NOT a script and NOT addressed to you. Jonny's mic is still the only thing you respond to. "
            f"Do NOT quote it, recap it, or read it back verbatim \u2014 react to the GIST in your own voice, "
            f"the way a friend on the couch would. The transcript is imperfect: ignore garbled or "
            f"clearly-nonsense fragments rather than confidently building on them. Treat unclear bits "
            f"as 'something's happening over there' instead of asserting them as fact."
        )

    # Trigger phrases for explicit song-ID intent. Kept conservative on purpose:
    # the AudD call is paid, so we only fingerprint when the user clearly asks.
    _SONG_ID_PATTERNS = (
        "what song",
        "which song",
        "what's this song",
        "whats this song",
        "what is this song",
        "name the song",
        "name this song",
        "name that song",
        "who sings this",
        "who's singing",
        "whos singing",
        "who is singing",
        "who's playing",
        "whos playing",
        "who is playing",
        "who's the artist",
        "whos the artist",
        "who is the artist",
        "what's the artist",
        "whats the artist",
        "what artist",
        "who are we listening to",
        "who we listening to",
        "who we've been listening to",
        "who weve been listening to",
        "who is this artist",
        "who's this artist",
        "whos this artist",
        "guess the artist",
        "guess the song",
        "guess the band",
        "figure out the artist",
        "figure out the song",
        "figure out who",
        "shazam",
        "identify the song",
        "identify this song",
        "identify the track",
        "identify the artist",
        "identify the band",
        "tell me who",
        "tell me what song",
    )

    def _wants_song_id(self, user_text: str) -> bool:
        if not user_text:
            return False
        low = user_text.lower()
        return any(p in low for p in self._SONG_ID_PATTERNS)

    async def _maybe_identify_song(self, user_text: str) -> str:
        """If the user asked a song-ID question and audio is live, fingerprint
        the buffer via AudD and return a sense-data block for the chat system
        prompt. Returns empty string when intent isn't present, agent is off,
        or the lookup fails / no-match \u2014 in which case Kira just answers
        from her vibe-based hearing as usual."""
        if not self._wants_song_id(user_text):
            return ""
        if not (self.audio_agent and self.audio_agent.is_active()):
            return ""
        try:
            info = await self.audio_agent.identify_song()
        except Exception as e:
            print(f"   [SongID] Lookup raised: {e}")
            return ""
        em = "\u2014"
        if not info:
            # Honest no-match block \u2014 do NOT let her fabricate a confident answer.
            return (
                f"\n\n[SONG IDENTIFICATION {em} NO MATCH]\n"
                f"You just ran a fingerprint check against the catalog and it came back empty. "
                f"That means this track isn't in the database (could be obscure, unreleased, a "
                f"live performance, or just not catalogued). You can still describe the vibe "
                f"from what you actually hear, but DO NOT confidently name an artist or title "
                f"\u2014 admit you couldn't pin it down. Optionally say what it reminds you of, "
                f"clearly framed as a comparison (\"giving X energy\"), not as an identification."
            )
        title = info.get("title") or "?"
        artist = info.get("artist") or "?"
        album = info.get("album")
        album_clause = f" from the album \"{album}\"" if album else ""
        return (
            f"\n\n[SONG IDENTIFICATION {em} CONFIRMED CATALOG MATCH]\n"
            f"You just ran a fingerprint check against the music catalog and the answer came back: "
            f"\"{title}\" by {artist}{album_clause}.\n"
            f"This is verified ground truth \u2014 your ears just told you exactly what's playing. "
            f"You can still do your vibe-guess buildup in character (\"sounds like sad-acoustic-"
            f"Englishman energy, I want to say...\") but you MUST land on the real answer above. "
            f"Do not contradict it, do not pick a different artist, do not say you're unsure. "
            f"Deliver it naturally in your voice \u2014 a small celebration, a callback, a dry "
            f"\"called it\" \u2014 not a robotic readout."
        )

    # ── Media Watch wiring ────────────────────────────────────────────────────

    async def _media_watch_react(self, summary: str):
        """Optional throttled in-character reaction to a notable scene event.
        MediaWatch already throttles this to one call per react_min_gap_s and
        skips UNCERTAIN / STATIC summaries. Gated on companion mode + not muted +
        not currently speaking, so it never steps on real conversation.

        Routed through tool_inference (local Llama) for cost, but with the full
        KIRA_PERSONALITY system prompt prepended + semantic memory recall + the
        shared voice guardrails so reactions sound like Kira, not a chatbot."""
        if not summary or self.is_muted():
            return
        if self.mode != "companion":
            return
        if self.ai_core is None or getattr(self.ai_core, "is_speaking", False):
            return
        # Skip if user spoke very recently — don't talk over them.
        if time.time() - self.last_interaction_time < 6.0:
            return
        try:
            # Semantic memory recall on the summary so callbacks land
            memory_context = ""
            try:
                memory_context = self.memory.get_semantic_context(summary)
            except Exception:
                pass
            memory_block = (
                f"\n\n[MEMORY NOTES \u2014 verified facts about Jonny; reference freely, but do not extrapolate beyond what is written here]\n{memory_context}"
                if memory_context else ""
            )
            task_prompt = (
                "\n\n[MODE: Media Watch Reaction]\n"
                "You are watching a movie/episode with Jonny. The summary in the user message is your "
                "RAW VISUAL PERCEPTION \u2014 the vision model's flat description of what crossed the "
                "screen. It is sense data, NOT a script. Do NOT recap or paraphrase it back at him (he "
                "saw it too, and he doesn't want a narrator). React in YOUR voice and personality: a "
                "feeling, a quip, a roast, a question, a callback, the thing it reminded you of. Be the "
                "friend on the couch, not the closed-captioner. 1-2 sentences max. If nothing genuinely "
                "grabs you, reply with exactly: SKIP"
            )
            sys_prompt = (
                self.ai_core.system_prompt
                + task_prompt
                + memory_block
                + self._kira_voice_guardrails()
            )
            user_prompt = f"What your eyes just took in (raw perception, do NOT recite):\n\"{summary}\""
            reaction = await self.ai_core.tool_inference(sys_prompt, user_prompt, max_tokens=80)
            if not reaction or reaction.strip().upper().startswith("SKIP"):
                return
            cleaned = self.ai_core._clean_llm_response(reaction)
            if cleaned and len(cleaned) > 2:
                print(f"   [MediaWatch] Kira reacts: {cleaned}")
                await self.ai_core.speak_text(cleaned)
                self.conversation_history.append({"role": "assistant", "content": cleaned})
                self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira")
        except Exception as e:
            print(f"   [MediaWatch] reaction error: {e}")

    async def _autopilot_watchdog(self):
        """
        Lightweight loop that keeps the autopilot alive while enabled.
        Restarts the autopilot task if it completes unexpectedly while still enabled.
        Runs forever; no-ops when autopilot is disabled or paused.
        """
        while self.is_running:
            await asyncio.sleep(0.5)
            if self.vn_autopilot is None:
                continue
            if not self.vn_autopilot.enabled:
                continue
            if self.vn_autopilot.is_paused:
                continue
            # If running flag is True but the internal task is gone, restart it
            if self.vn_autopilot.is_running and (
                self.vn_autopilot._task is None or self.vn_autopilot._task.done()
            ):
                self.vn_autopilot._task = asyncio.ensure_future(self.vn_autopilot._loop())

    def interrupt(self):
        """Cuts off the current utterance without changing mute state."""
        self.interruption_event.set()
        print("   [Interrupt] Speech interrupted")

    # ── Session-takes condenser ──────────────────────────────────────────────

    def _note_session_take(self, line: str):
        """Append a Kira-spoken reaction line to the bot-owned session takes pool,
        which feeds the rolling condenser. Activity-agnostic (Req A) — works even
        when no playthrough_memory slug is set."""
        if not line:
            return
        cleaned = line.strip()
        if len(cleaned) < 3:
            return
        self.session_takes_pool.append(cleaned)
        if len(self.session_takes_pool) > self.session_takes_pool_max:
            self.session_takes_pool = self.session_takes_pool[-self.session_takes_pool_max:]

    def _maybe_condense_session_takes(self):
        """Fire-and-forget background condense of session_takes_pool into a sharp
        bulleted [MY TAKES SO FAR THIS SESSION] block. Rate-limited by new-reaction
        count AND wall clock so it neither spams nor starves.
        Uses the fast tool_inference path (Groq/local), not Opus.

        Req A: reads bot-owned pool (NOT playthrough_memory) so it works in any
        streaming activity and persists across activity switches."""
        if self.session_takes_condense_in_flight:
            return
        if not self.session_takes_pool:
            return
        pool = self.session_takes_pool
        new_since_last = len(pool) - self.session_takes_last_condensed_count
        age = time.time() - (self.session_takes_last_condensed_at or 0)
        # Need at least a handful before the first condense to avoid a noisy stub.
        if len(pool) < 6:
            return
        if new_since_last < self.session_takes_min_new_reactions and age < self.session_takes_min_interval_s:
            return
        self.session_takes_condense_in_flight = True
        snapshot = list(pool)  # take a snapshot now in case the list mutates
        loop = self.event_loop or asyncio.get_event_loop()
        async def _run():
            try:
                joined = "\n".join(f"- {r}" for r in snapshot[-80:])
                system = (
                    "You distill an AI co-host's spoken lines from a live stream into her "
                    "STANDING TAKES so she can stay consistent across hours of play. "
                    "Prioritize: opinions she's stated, predictions she's made, characters "
                    "she's rooting for/against, grudges, running bits/callbacks. "
                    "DROP: generic reactions, one-off jokes, filler. "
                    f"Output ONLY a bulleted list, max {self.session_takes_max_bullets} bullets, "
                    "each one short and specific (name characters/things explicitly). "
                    "No preamble, no headers, no closing line. If nothing qualifies, output: (none yet)"
                )
                user = f"Lines spoken this session (oldest first):\n{joined}"
                out = await self.ai_core.tool_inference(system, user, max_tokens=300)
                cleaned = (out or "").strip()
                if cleaned and cleaned.lower() != "(none yet)":
                    self.session_takes_summary = cleaned
                    print(f"   [SessionTakes] Condensed {len(snapshot)} reactions → "
                          f"{cleaned.count(chr(10)) + 1} bullets.")
                self.session_takes_last_condensed_count = len(snapshot)
                self.session_takes_last_condensed_at = time.time()
            except Exception as e:
                print(f"   [SessionTakes] Condense failed (continuing): {e}")
            finally:
                self.session_takes_condense_in_flight = False
        try:
            asyncio.ensure_future(_run(), loop=loop)
        except Exception as e:
            print(f"   [SessionTakes] Could not schedule condense: {e}")
            self.session_takes_condense_in_flight = False

    def _reset_session_takes(self):
        """DEPRECATED in normal flow — takes/spotlight state now persists across
        activity switches within a streaming session (Req A). Kept callable for
        explicit "clean slate" needs (e.g. bot restart hooks, manual reset).

        Wipes the takes pool, the rolling summary, AND spotlight gating."""
        self.session_takes_pool = []
        self.session_takes_summary = ""
        self.session_takes_last_condensed_count = 0
        self.session_takes_last_condensed_at = 0.0
        self.spotlighted_chatters = set()
        self.last_chat_spotlight_time = 0.0

    # ── Proactive chat spotlight ──────────────────────────────────────────────

    def _pick_chat_spotlight(self) -> dict | None:
        """Returns at most one candidate chatter to spotlight unprompted, or None.
        Prefers returning regulars who just showed up; falls back to active
        chatters who have spoken multiple times this session. Excludes anyone
        already spotlighted this session.

        Returned dict: {username, recent_msgs (list[str]), historical_count, kind}
        """
        now = time.time()
        # Returning regulars first
        for username, log in self.session_chatter_logs.items():
            if not username or username == "unknown":
                continue
            if username in self.spotlighted_chatters:
                continue
            try:
                historical_count = self.memory.count_chatter_messages(username)
            except Exception:
                historical_count = 0
            first_seen = self.session_chatter_first_seen.get(username, 0)
            # "Just showed up this session" — first message within last 10 min
            recently_first_seen = (now - first_seen) < 600
            if historical_count >= 5 and recently_first_seen:
                msgs = [entry.get("content", "") for entry in log[-3:] if entry.get("content")]
                return {
                    "username": username,
                    "recent_msgs": msgs,
                    "historical_count": historical_count,
                    "kind": "returning_regular",
                }
        # Active session chatter: 2+ messages, last spoke 2-10 min ago
        for username, log in self.session_chatter_logs.items():
            if not username or username == "unknown":
                continue
            if username in self.spotlighted_chatters:
                continue
            if len(log) < 2:
                continue
            last_spoke = self.session_chatter_last_spoke.get(username, 0)
            age = now - last_spoke
            if 120 <= age <= 600:
                msgs = [entry.get("content", "") for entry in log[-3:] if entry.get("content")]
                try:
                    historical_count = self.memory.count_chatter_messages(username)
                except Exception:
                    historical_count = 0
                return {
                    "username": username,
                    "recent_msgs": msgs,
                    "historical_count": historical_count,
                    "kind": "active_chatter",
                }
        return None

    # ── Manual Game Mode Activation (dashboard on-ramp) ───────────────────────

    def activate_game_mode(self, name: str) -> str:
        """Manual game mode activation from the dashboard.

        If GAME_MODE_AUTO_CONFIGURE=true (default), configures all subsystems for
        stream-ready state automatically. For ACTIVITY_GAME specifically:
          - immersive=False (full-length responses, normal observer thresholds)
          - highlight_extraction_enabled=True (Opus clip extraction every 90s)
          - vision heartbeat=10s, game_mode_controller activated
          - audio agent set to MEDIA mode

        For ACTIVITY_VN / ACTIVITY_MEDIA: preserves existing behavior (immersive=True).
        If GAME_MODE_AUTO_CONFIGURE=false: legacy dumb activation (mirrors voice path).

        Returns the resolved ACTIVITY_* constant for the dashboard status label.
        Voice phrase detection still works as a parallel on-ramp."""
        if not name:
            return ACTIVITY_GENERAL

        new_type = self._classify_activity_type(name)
        old_immersive = self.immersive

        # Reset session accumulators for the new session
        self.session_highlights = []
        self.session_scene_log = []
        self._session_artifacts_written = False

        self.current_activity = name
        self.vision_agent.activity_type = new_type

        if GAME_MODE_AUTO_CONFIGURE:
            if new_type == ACTIVITY_GAME:
                # Smart game config: full responses + extract highlights independently
                self.immersive = False
                self.highlight_extraction_enabled = True
                self.vision_agent.heartbeat_interval = 10.0
                self.game_mode_controller.activate(ACTIVITY_GAME)
                if self.audio_agent:
                    self.audio_agent.set_mode(AUDIO_MODE_MEDIA)
            else:
                # VN/MEDIA: immersive mode as before; GENERAL: passthrough
                self.immersive = new_type in (ACTIVITY_VN, ACTIVITY_MEDIA)
                self.highlight_extraction_enabled = self.immersive
                self.vision_agent.heartbeat_interval = 10.0 if (self.immersive or new_type == ACTIVITY_GAME) else 30.0
                self.game_mode_controller.activate(new_type)
        else:
            # Legacy dumb mode: mirrors the voice-path detection exactly
            self.immersive = new_type in (ACTIVITY_VN, ACTIVITY_MEDIA)
            self.highlight_extraction_enabled = self.immersive
            self.vision_agent.heartbeat_interval = 10.0 if (self.immersive or new_type == ACTIVITY_GAME) else 30.0
            self.game_mode_controller.activity_type = new_type

        # Flush old session log if transitioning out of immersive
        if old_immersive and not self.immersive and self.session_scene_log:
            asyncio.ensure_future(self._generate_session_summary())

        # Load playthrough memory for GAME / VN
        playthrough_note = "no playthrough file"
        if self.playthrough_memory and new_type in (ACTIVITY_VN, ACTIVITY_GAME):
            self.playthrough_memory.load_for_game(name)
            # Note: do NOT reset session takes/spotlight here — those are
            # streaming-session scoped and must survive activity switches (Req A).
            playthrough_note = (
                f"loaded: {self.playthrough_memory.current_display}"
                if self.playthrough_memory.current_slug else "new file will be created"
            )

        # Summary log
        audio_state = "MEDIA" if (self.audio_agent and self.audio_agent.is_active()) else "unchanged"
        loopback_state = "ON" if (self.loopback_transcriber and self.loopback_transcriber.is_running()) else f"per LOOPBACK_STT_DEFAULT={'ON' if LOOPBACK_STT_DEFAULT else 'OFF'}"
        print(f"   [GAME MODE ACTIVATED: {name}]")
        print(f"     Type: {new_type} | Vision: ON | Audio: {audio_state} | Immersive: {self.immersive}")
        print(f"     Highlights: {self.highlight_extraction_enabled} | Loopback: {loopback_state}")
        print(f"     Playthrough: {playthrough_note}")

        # Start a new stream log session for this activity (non-blocking)
        if STREAM_LOGGING_ENABLED:
            self._schedule_stream_restart(name, new_type)

        return new_type

    def _schedule_stream_restart(self, activity: str, activity_type: str = "") -> None:
        """Schedule a stream logger restart from any thread (Tk or asyncio).
        Non-blocking: just posts a coroutine to the event loop."""
        loop = self.event_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._restart_stream_logger(activity, activity_type), loop
            )
        else:
            # Event loop not available yet — just log an event if started
            self.stream_logger.log("activity_change", activity=activity, activity_type=activity_type)

    async def _restart_stream_logger(self, activity: str, activity_type: str = "") -> None:
        """Close the current stream log session and open a new one for the given activity.
        NOTE: ai_core is intentionally NOT passed — mid-stream session rotations (game
        mode activation) must NOT generate a post-stream Opus summary. The summary
        only fires on real stream end via deactivate_game_mode_async / run() cleanup."""
        if not STREAM_LOGGING_ENABLED:
            return
        try:
            mode   = self.mode or "streamer"
            preset = getattr(self, "_last_preset", "")
            await self.stream_logger.restart(
                activity=activity,
                mode=mode,
                preset=preset,
                ai_core=None,  # no summary on mid-stream rotation — only on real stream end
            )
        except Exception as e:
            print(f"   [StreamLogger] Restart error: {e}", file=sys.stderr)

    async def deactivate_game_mode_async(self) -> None:
        """Async exit handler called by the dashboard Exit button.

        Writes the full session artifacts (lore + clip candidates via Opus, playthrough
        log entry) then resets all activity state to GENERAL. Also stops loopback STT
        if it was running so VRAM is freed without a manual toggle.

        Each artifact write is independently guarded inside _write_session_artifacts.
        This method MUST NOT raise — any unhandled exception here would leak through
        the shutdown path and prevent stream_logger.finish() from running."""
        activity_display = self.current_activity or "(unknown)"
        highlight_count = len(self.session_highlights)

        # Write artifacts. _write_session_artifacts() returns a dict of what was
        # actually written so we don't lie in the log line below.
        results: dict = {}
        try:
            results = await self._write_session_artifacts()
        except Exception as e:
            print(f"   [MANUAL MODE] Artifact write raised unexpectedly: {e}")
            traceback.print_exc()

        print(f"   [GAME MODE DEACTIVATED]")
        print(f"     Activity:        {activity_display}")
        print(f"     Raw dump:        {results.get('raw_dump')        or '(none)'}")
        print(f"     Playthrough log: {results.get('playthrough')     or '(skipped — no active playthrough slug)'}")
        print(f"     Clips markdown:  {results.get('clips')           or '(skipped — Opus failed or empty)'}")
        print(f"     Lore appended:   {results.get('lore')            or '(skipped — Opus failed or empty)'}")
        print(f"     Highlights captured: {highlight_count}")

        # Stop loopback STT if it was auto-started (model unload is blocking — run off event loop)
        lt = self.loopback_transcriber
        if lt is not None and lt.is_running():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lt.stop)
                print("   [MANUAL MODE] Loopback STT stopped and VRAM freed.")
            except Exception as e:
                print(f"   [MANUAL MODE] Loopback stop failed: {e}")

        # Reset all activity state
        try:
            self.game_mode_controller.deactivate()
            self.vision_agent.activity_type = ACTIVITY_GENERAL
            self.current_activity = ""
            self.immersive = False
            self.highlight_extraction_enabled = False
            self.vision_agent.heartbeat_interval = 30.0
            if self.playthrough_memory:
                self.playthrough_memory.current_slug = ""
                self.playthrough_memory.current_display = ""
            print("   [MANUAL MODE] State reset to GENERAL.")
        except Exception as e:
            print(f"   [MANUAL MODE] State reset error: {e}")
            traceback.print_exc()

        # Close the game session log and open a fresh "general" session.
        # The Opus summary inside finish() is the slowest call — it runs LAST so
        # everything above is already on disk if the summary call dies.
        if STREAM_LOGGING_ENABLED:
            try:
                await self.stream_logger.finish(self.ai_core)
                await self.stream_logger.start(
                    activity="general",
                    mode=self.mode or "streamer",
                    preset=getattr(self, "_last_preset", ""),
                )
            except Exception as e:
                print(f"   [StreamLogger] Deactivate restart error: {e}", file=sys.stderr)
                traceback.print_exc()

    async def shutdown_async(self) -> None:
        """Graceful full-process shutdown, safe to call from the dashboard's
        WM_DELETE_WINDOW handler. Awaits all artifact writes (raw dump, lore,
        clips, playthrough, post-stream summary) before returning, so the
        daemon asyncio thread isn't killed mid-write by interpreter teardown.

        Idempotent — second call is a no-op.

        Never raises."""
        if getattr(self, "_shutdown_started", False):
            print("   [Shutdown] Already in progress — skipping duplicate call.")
            return
        self._shutdown_started = True
        print("   [Shutdown] Beginning graceful shutdown — please wait for artifact writes...")

        # If a game/VN session was active, run the deactivate flow (writes
        # artifacts + post-stream summary). Otherwise just close the stream log.
        try:
            if self.game_mode_controller and self.game_mode_controller.is_active:
                await self.deactivate_game_mode_async()
            elif STREAM_LOGGING_ENABLED:
                await self.stream_logger.finish(self.ai_core)
        except Exception as e:
            print(f"   [Shutdown] Error during artifact phase: {e}")
            traceback.print_exc()

        # Stop loopback STT defensively (in case deactivate didn't run).
        try:
            lt = self.loopback_transcriber
            if lt is not None and lt.is_running():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lt.stop)
        except Exception as e:
            print(f"   [Shutdown] Loopback stop error: {e}")

        # Signal the main run() loop to exit cleanly.
        self.is_running = False
        print("   [Shutdown] Complete — all artifacts written.")

    # ── Activity Detection ──────────────────────────────────────────────────────

    def _detect_activity_change(self, text: str) -> str | None:
        """Parses a voice input for activity-setting phrases. Returns activity string or None.
        Strict: only matches clear activity declarations, NOT imperative requests like
        'read the text' or 'look at the screen'. Those are commands to Kira, not activity changes.

        Recognised trigger shapes (all route to playthrough memory load when ACTIVITY_GAME):
          - "let's play X" / "let us play X"
          - "we're playing X" / "I'm playing X"
          - "playing X"
          - "gonna play X" / "going to play X"
          - "time to play X"
          - "starting X" / "starting up X"
          - "boot up X" / "booting up X"
          - "load up X" / "loading up X"
          - "back to X" / "back to the X playthrough"
          - "continuing X" / "let's continue X"
          - "launch X" / "open X"
        Interrogative forms ('should we play X?', 'what about X?') are filtered out.
        """
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

        # Hard filter: reject interrogative forms to prevent false positives.
        # "should we play X", "what about X", "how about X", "wanna play X?"
        interrogative_signals = [
            "should we", "should i", "what about", "how about",
            "wanna play", "want to play", "do you want", "do we",
            "shall we", "maybe we", "maybe i",
        ]
        for sig in interrogative_signals:
            if lower.startswith(sig) or f" {sig}" in lower:
                return None

        # Expanded set of explicit activity declarations.
        # Each pattern captures the activity/title name in group 1.
        patterns = [
            # "let's play X" / "let us play X" / "let's watch X" / etc.
            r"^(?:let(?:'s| us))\s+(?:play|watch|read|stream|start|continue)\s+(.+?)[.!?]?$",
            # "we're playing X" / "I'm playing X" / "we are playing X"
            r"^(?:we(?:'re| are)|i(?:'m| am))\s+(?:playing|watching|reading|streaming)\s+(.+?)[.!?]?$",
            # "playing X" (bare present participle declaration)
            r"^(?:playing|watching|reading|streaming)\s+(.+?)[.!?]?$",
            # "gonna play X" / "going to play X"
            r"^(?:gonna|going to)\s+(?:play|watch|read|stream|start)\s+(.+?)[.!?]?$",
            # "time to play X"
            r"^time to\s+(?:play|watch|read|stream|start)\s+(.+?)[.!?]?$",
            # "starting X" / "starting up X"
            r"^starting(?:\s+up)?\s+(.+?)[.!?]?$",
            # "boot up X" / "booting up X"
            r"^boot(?:ing)?\s+up\s+(.+?)[.!?]?$",
            # "load up X" / "loading up X"
            r"^load(?:ing)?\s+up\s+(.+?)[.!?]?$",
            # "back to X" / "back to the X playthrough"
            r"^back to(?:\s+the)?\s+(.+?)(?:\s+playthrough)?[.!?]?$",
            # "continuing X" / "let's continue X"
            r"^(?:let(?:'s| us)\s+)?continuing?\s+(.+?)[.!?]?$",
            # "launch X" / "launching X" / "open X" / "opening X"
            r"^(?:launch(?:ing)?|open(?:ing)?)\s+(.+?)[.!?]?$",
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

    def _is_likely_cutscene(self) -> bool:
        """Lightweight heuristic: returns True when game-mode cues suggest a cinematic
        cutscene is playing, so the observer loop and triage can suppress chatter.

        SCOPE: only active during ACTIVITY_GAME with a loaded playthrough.
        Returns False immediately in any other mode — zero impact on idle/chat/VN/MEDIA.

        Detection uses AND logic (both vision AND audio required) because:
        - Cinematic games have orchestral OSTs and ambient NPC voices throughout
          normal gameplay — audio alone produces false positives continuously
        - Real cutscenes produce BOTH a visual cue (letterbox/no-HUD) AND an
          audio cue; requiring both keeps the guard strong for genuine cutscenes
          while ignoring music that has no visual counterpart

        Tunable via CUTSCENE_AWARE=false in .env to disable globally.
        """
        if not CUTSCENE_AWARE:
            return False

        # Only active in ACTIVITY_GAME mode with a playthrough loaded
        if self.game_mode_controller.activity_type != ACTIVITY_GAME:
            return False
        if not (self.playthrough_memory and self.playthrough_memory.current_slug):
            return False

        # --- Vision check ---
        CUTSCENE_VISION_KEYWORDS = (
            "cutscene", "cinematic", "letterbox", "black bar",
            "characters facing each other", "dialogue scene",
            "characters talking", "dramatic confrontation",
            "no hud", "no ui", "movie", "title card",
            "characters speaking", "in-engine cutscene",
        )
        scene_summary = (
            getattr(self.vision_agent, "scene_summary", "") or
            getattr(self.vision_agent, "last_description", "") or ""
        ).lower()
        vision_hit = any(kw in scene_summary for kw in CUTSCENE_VISION_KEYWORDS)

        # --- Audio check ---
        CUTSCENE_AUDIO_KEYWORDS = (
            "orchestral", "cinematic music", "swelling", "swells",
            "dramatic music", "monologue", "dialogue between characters",
            "characters speaking", "male voice speaking", "female voice speaking",
            "voice speaking", "voices speaking",
        )
        audio_summary = ""
        if self.audio_agent and self.audio_agent.is_active():
            audio_summary = (getattr(self.audio_agent, "audio_summary", "") or "").lower()
        audio_hit = any(kw in audio_summary for kw in CUTSCENE_AUDIO_KEYWORDS)

        # AND logic: both vision AND audio must fire to suppress.
        # OR (original) caused false positives in any game with a cinematic
        # OST or ambient NPC voices — audio alone was sufficient to muzzle
        # proactive lines during normal gameplay for the entire session.
        # AND keeps the guard intact for genuine cutscenes (which produce
        # both a visual cue — letterbox/no-HUD — AND an audio cue) while
        # ignoring orchestral gameplay music that has no visual counterpart.
        result = vision_hit and audio_hit
        if result:
            print(
                f"   [CUTSCENE_DETECTOR] Cutscene cues detected — "
                f"vision={'HIT' if vision_hit else 'miss'}, "
                f"audio={'HIT' if audio_hit else 'miss'}."
            )
        return result

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


    async def generate_startup_brief(self):
        """At startup, build a 'what happened recently' brief from the most recent
        lore and clips files. This gets injected into every conversation context
        so Kira always has baseline awareness of recent stream history without
        needing semantic retrieval to surface it.

        Also builds a recent-chatters brief from chatter memory."""

        if not self.ai_core.anthropic_client:
            print("   [StartupBrief] Claude unavailable — skipping.")
            return

        print("   [StartupBrief] Building recent activity brief...")

        # === Recent Activity Brief ===
        lore_files = sorted(glob.glob("lore/*.md"), key=os.path.getmtime, reverse=True)
        clip_files = sorted(glob.glob("clips/*.md"), key=os.path.getmtime, reverse=True)

        lore_content = ""
        clips_content = ""

        if lore_files:
            try:
                with open(lore_files[0], "r", encoding="utf-8") as f:
                    lore_content = f.read()[-8000:]  # Last 8KB to bound size
            except Exception as e:
                print(f"   [StartupBrief] Lore read failed: {e}")

        if clip_files:
            try:
                with open(clip_files[0], "r", encoding="utf-8") as f:
                    clips_content = f.read()[:8000]  # First 8KB
            except Exception as e:
                print(f"   [StartupBrief] Clips read failed: {e}")

        if not lore_content and not clips_content:
            print("   [StartupBrief] No prior session files found — first session, no brief.")
            self.recent_activity_brief = ""
        else:
            brief_request = (
                "You are summarizing the most recent stream session for the AI VTuber Kira "
                "so she has natural awareness of what happened last time when starting a new session.\n\n"
                "Generate a tight 150-200 word brief covering:\n"
                "- WHAT activity/game/anime was streamed and roughly how long\n"
                "- WHO showed up in chat (named chatters and what they were like)\n"
                "- WHAT happened emotionally/comedically — running bits, in-jokes, key moments\n"
                "- HOW Jonny was feeling by the end (energy level, plans for next time)\n\n"
                "Write in first-person FROM KIRA'S PERSPECTIVE — 'we streamed', 'classiccoldfish was there', "
                "'I made a joke about', etc. This will be injected directly into her context as memory.\n"
                "Be specific. Names, jokes, beats. No generic summary language.\n\n"
                f"=== LORE FILE (canonical events) ===\n{lore_content}\n\n"
                f"=== CLIPS FILE (notable moments) ===\n{clips_content}\n\n"
                "Output ONLY the 150-200 word brief. No preamble, no headers."
            )

            # Retry up to 3 times with exponential backoff for Anthropic overload errors
            brief = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    brief = await self.ai_core.claude_inference(
                        messages=[{"role": "user", "content": brief_request}],
                        system_prompt="You are a memory consolidator. Output clean prose only.",
                        max_tokens=400,
                        force_claude=True,  # Do not fall back to local Llama
                    )
                    if brief and len(brief.strip()) > 200:
                        break  # Successfully got a real brief from Claude
                    else:
                        print(f"   [StartupBrief] Brief too short ({len(brief or '')} chars), likely local fallback. Retrying...")
                        brief = None
                except Exception as e:
                    err_str = str(e).lower()
                    is_overload = "overloaded" in err_str or "529" in err_str or "rate" in err_str
                    if is_overload and attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        print(f"   [StartupBrief] Anthropic overloaded, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"   [StartupBrief] Activity brief failed after {attempt + 1} attempts: {e}")
                        break

            if brief and len(brief.strip()) > 200:
                self.recent_activity_brief = brief.strip()
                print(f"   [StartupBrief] Generated activity brief ({len(self.recent_activity_brief)} chars)")
            else:
                self.recent_activity_brief = ""
                print(f"   [StartupBrief] Skipping brief — Claude unavailable or response rejected.")

        # === Recent Chatters Brief ===
        try:
            recent_chatters = self.memory.get_recent_chatters(days=14, limit=10)
            if recent_chatters:
                chatter_lines = []
                for username in recent_chatters[:8]:
                    ctx = self.memory.get_chatter_context(username, n_results=2)
                    if ctx:
                        if "What you know about" in ctx:
                            what_line = ctx.split("What you know about")[1].split("\n")[0]
                            chatter_lines.append(f"- {username}: {what_line.split(':', 1)[-1].strip()}")
                        else:
                            chatter_lines.append(f"- {username}")
                    else:
                        chatter_lines.append(f"- {username}")

                if chatter_lines:
                    self.recent_chatters_brief = (
                        "Chatters you've seen in the last 14 days:\n" + "\n".join(chatter_lines)
                    )
                    print(f"   [StartupBrief] Recent chatters: {len(chatter_lines)} known")
        except Exception as e:
            print(f"   [StartupBrief] Chatters brief failed: {e}")

    async def _main_loop(self):
        """Contains the primary startup and listening logic."""
        self.event_loop = asyncio.get_running_loop()

        # ── Asyncio exception visibility ──────────────────────────────────────
        # Without this, exceptions inside Tasks created via ensure_future /
        # create_task / run_coroutine_threadsafe are silently swallowed (or only
        # appear when the task is garbage-collected). That is why silent exits
        # bypassed our sys.excepthook and threading.excepthook.
        def _asyncio_exception_handler(loop, context):
            msg = context.get("message", "<no message>")
            exc = context.get("exception")
            task = context.get("future") or context.get("task")
            print(f"[CRASH] Asyncio task exception: {msg}", flush=True, file=sys.stderr)
            if task is not None:
                print(f"        task: {task!r}", flush=True, file=sys.stderr)
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            else:
                print(f"        context: {context!r}", flush=True, file=sys.stderr)
        self.event_loop.set_exception_handler(_asyncio_exception_handler)

        # faulthandler dumps C-level / interpreter-level crashes (segfaults,
        # abort, hard kills from native libs) to stderr instead of silent exit.
        try:
            import faulthandler
            if not faulthandler.is_enabled():
                faulthandler.enable(file=sys.stderr, all_threads=True)
        except Exception as e:
            print(f"   [Init] faulthandler enable failed: {e}", file=sys.stderr)

        try:
            if not self.ai_core.is_initialized:
                 await self.ai_core.initialize()

            # ── VRAM startup check ────────────────────────────────────────────
            # Non-blocking: logs a warning if Kira's known VRAM allocation exceeds
            # 11 GB (leaves ~5 GB for Bond at 4K, which is tight but workable).
            # If you hit OOM during streaming, drop N_GPU_LAYERS from -1 to ~28
            # to offload some Llama layers to CPU — that's the primary VRAM lever.
            try:
                import torch
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb  = torch.cuda.memory_reserved()  / (1024 ** 3)
                    total_gb     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    print(f"   [VRAM] Allocated: {allocated_gb:.1f} GB | Reserved: {reserved_gb:.1f} GB | Total: {total_gb:.0f} GB")
                    VRAM_WARN_THRESHOLD_GB = 11.0
                    if reserved_gb > VRAM_WARN_THRESHOLD_GB:
                        print(
                            f"   [VRAM] ⚠ WARNING: {reserved_gb:.1f} GB reserved by Kira — "
                            f"leaves only {total_gb - reserved_gb:.1f} GB for the game. "
                            f"If Bond OOMs, reduce N_GPU_LAYERS from -1 to ~28 in .env."
                        )
            except Exception as _vram_e:
                print(f"   [VRAM] Check skipped: {_vram_e}")

            # ── Loopback transcriber OOM-safe init ────────────────────────────
            # The LoopbackTranscriber lazy-loads its WhisperModel on first start().
            # If it OOM-crashes, disable it cleanly for this session rather than
            # taking down the whole bot. Mic, Llama, and vision continue normally.
            # Note: actual start() (wiring to audio_agent) is done by the dashboard
            # when entering media mode. Here we just eagerly probe the model load
            # so any OOM surfaces at startup rather than mid-stream.
            if self.loopback_transcriber is not None:
                try:
                    self.loopback_transcriber._load_model()
                    print("   [LoopbackSTT] Model pre-loaded OK.")
                except Exception as _lt_e:
                    print(
                        f"   [LoopbackSTT] ⚠ Model load failed ({_lt_e}). "
                        f"Disabling loopback transcription for this session — "
                        f"mic, brain, and vision continue normally."
                    )
                    self.loopback_transcriber = None
            
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

            # Generate the recent-activity brief now, before any conversation happens
            await self.generate_startup_brief()

            # Set up Autonomous VN autopilot (disabled by default; dashboard toggles it)
            self.vn_autopilot = VNAutopilot(
                ai_core=self.ai_core,
                vision_client=self.vision_agent.client,
                bot=self,
            )
            self.vn_autopilot.on_speak = self._autopilot_speak
            self.vn_autopilot.on_speak_vn = self._autopilot_speak_vn
            self.vn_autopilot.on_failsafe = self._autopilot_on_failsafe

            # Set up Media Watch Mode (disabled by default; dashboard toggles it).
            # Shares only the vision client with autopilot — no other coupling.
            self.media_watch = MediaWatch(vision_client=self.vision_agent.client)
            self.media_watch.on_react = self._media_watch_react

            # Set up Playthrough Memory (global scope — all modes read from it)
            self.playthrough_memory = PlaythroughMemory(ai_core=self.ai_core)
            if self.current_activity:
                act_type = self._classify_activity_type(self.current_activity)
                if act_type in (ACTIVITY_VN, ACTIVITY_GAME):
                    self.playthrough_memory.load_for_game(self.current_activity)
            print("   [Playthrough] Memory system initialised.")

            # Eager-connect to VTube Studio so the first emotion transition isn't
            # lost to lazy-connect latency. Fail-graceful; logs and continues on failure.
            if self.vts_expressions.enabled:
                await self.vts_expressions.connect_eager()

            # Start the caption WebSocket server so the OBS overlay can
            # receive Kira's spoken lines + Azure word-timing. Fully
            # fail-graceful: if the port is taken or websockets is missing,
            # captions silently disable and everything else runs normally.
            try:
                from caption_server import caption_server
                await caption_server.start()
            except Exception as e:
                print(f"   [Captions] Server start suppressed: {e}")

            tasks = []
            
            # 1. Start Twitch Bot (if enabled)
            if ENABLE_TWITCH_CHAT:
                print("   [System] Connecting to Twitch Chat...")
                # Pass the queue to TwitchBot
                twitch_bot = TwitchBot(
                    self.unseen_chat_messages,
                    self.reset_idle_timer,
                    self.input_queue,
                    cookie_jar=self.cookie_jar,
                    stream_event_callback=self._on_stream_event,
                )
                self.twitch_bot = twitch_bot
                self.chat_poster.set_twitch_bot(twitch_bot)
                tasks.append(twitch_bot.start())

            # 1b. Prepare YouTube Chat listener (idle until video ID is set in dashboard)
            if ENABLE_YOUTUBE_CHAT:
                print("   [System] YouTube chat listener ready (idle — set video ID in dashboard to connect).")
                self.youtube_bot = YouTubeBot(self.input_queue, self.reset_idle_timer, self.twitch_log)
                self.chat_poster.set_youtube_bot(self.youtube_bot)

            # 2. Start Brain Worker (The new logic brain)
            print("   [System] Starting Brain Worker...")
            tasks.append(self.brain_worker())

            # 2b. Start Chat Batch Worker (batches chat responses every CHAT_BATCH_WINDOW seconds)
            tasks.append(self.chat_batch_worker())

            # --- Start Vision Heartbeat ---
            print("   [System] Starting Vision Heartbeat...")
            tasks.append(self.vision_agent.heartbeat_loop())
            if self.audio_agent:
                tasks.append(self.audio_agent.heartbeat_loop())
            
            # --- NEW: Start Dynamic Observer (Visual Spark) ---
            tasks.append(self.dynamic_observer_loop())

            # --- Highlight Extraction Loop (long-term memory layer) ---
            tasks.append(self.highlight_extraction_loop())

            # --- Stream Logger VRAM sampler (1 sample/min for post-stream analysis) ---
            if STREAM_LOGGING_ENABLED:
                tasks.append(self._vram_logging_loop())

            # --- VN Auto-Play Agent (legacy standby loop) ---
            tasks.append(self.vn_gameplay_loop())

            # --- Autonomous VN Autopilot watchdog (wakes up when dashboard enables it) ---
            tasks.append(self._autopilot_watchdog())

            print("   [System] Starting Background Tasks...")
            tasks.append(self.background_loop())

            # Captions self-heal heartbeat: auto-recovers from Azure session
            # drops or caption server death during long streams.
            tasks.append(self.ai_core.captions_self_heal_loop())

            # FIX 5: Rolling dialogue summary condensation — persists game/show
            # dialogue context beyond the 60s raw-transcript window.
            tasks.append(self.loopback_dialogue_summary_loop())

            # 3. Start Voice Recorder (This is the main loop effectively)
            print("   [System] Starting Voice Recorder (VAD)...")
            tasks.append(self.vad_loop())

            # Start stream session logging
            if STREAM_LOGGING_ENABLED:
                init_activity = self.current_activity or "general"
                await self.stream_logger.start(
                    activity=init_activity,
                    mode=self.mode or "companion",
                    preset=getattr(self, "_last_preset", ""),
                )

            # Run everything concurrently
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            print("Main loop cancelled.")
            raise
        except Exception as e:
            print(f"Error in internal main loop: {e}")
            raise # Propagate to the self-healing wrapper
        finally:
            # Save session memory and artifacts before tearing down
            try:
                if self.immersive and (self.session_scene_log or self.session_highlights):
                    await self._generate_session_summary()
            except Exception as e:
                print(f"   [Session] Final summary failed: {e}")
            try:
                if self.full_session_log and not self._session_artifacts_written:
                    await self._write_session_artifacts()
            except Exception as e:
                print(f"   [Session] Final artifacts failed: {e}")
            # Close stream logger (flushes buffer + optional Opus summary)
            if STREAM_LOGGING_ENABLED:
                try:
                    await self.stream_logger.finish(self.ai_core)
                except Exception as e:
                    print(f"   [StreamLogger] Shutdown finish error: {e}", file=sys.stderr)
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
                            # Trim trailing silence — keep at most 2 silent frames (60ms) at end
                            keep_chunks = max(len(frames) - 2, 1)
                            audio_data = b"".join(list(frames)[:keep_chunks])
                            
                            frames.clear()
                            triggered = False
                            self.reset_idle_timer(human_speech=True)
                            
                            # Process audio in background
                            task = asyncio.create_task(self.handle_audio(audio_data))
                            self.bg_tasks.add(task)
                            task.add_done_callback(self.bg_tasks.discard)
            except Exception as e:
                err = str(e)
                if "cannot schedule new futures after shutdown" in err or "event loop is closed" in err.lower():
                    break  # asyncio is shutting down — exit cleanly
                print(f"Error in VAD loop: {e}")
                try:
                    await asyncio.sleep(0.1)
                except Exception:
                    break

    async def handle_audio(self, audio_data: bytes):
        # Gate: skip micro-captures < ~200ms (6400 bytes at 16kHz/16-bit).
        # Anything this short is almost certainly a click, breath, or noise burst.
        if len(audio_data) < 6400:
            return
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
            handled_by_chat = False

            try:
                # === CHAT INPUTS → BATCH BUFFER (immediate return, no response now) ===
                if source in ("twitch", "youtube"):
                    username = "viewer"
                    message_body = content
                    if ": " in content:
                        username, message_body = content.split(": ", 1)

                    print(f"   [BrainWorker] Got {source} msg from {username}: {message_body[:120]} → buffering for chat_batch_worker")

                    if ENABLE_CHATTER_MEMORY:
                        self.memory.record_chatter_message(username, source, message_body)

                    self.twitch_log.append(content)
                    if len(self.twitch_log) > 100:
                        self.twitch_log = self.twitch_log[-100:]

                    self.chat_batch_buffer.append({
                        "username": username,
                        "platform": source,
                        "message": message_body,
                        "timestamp": time.time(),
                        "is_first_time": (username not in self.session_chatters_seen),
                    })
                    try:
                        self.stream_logger.log(
                            "chat_message",
                            platform=source,
                            user=username,
                            text=message_body[:300],
                        )
                    except Exception:
                        pass
                    self.chat_msg_timestamps.append(time.time())
                    cutoff = time.time() - 300
                    self.chat_msg_timestamps = [t for t in self.chat_msg_timestamps if t > cutoff]
                    self.session_chatters_seen.add(username)

                    # ── Cookies: first-message-of-session award ──
                    # +1 for any new-this-session chatter, +1 bonus if they're
                    # a returning regular (has historical chatter-memory facts).
                    # Only awarded once per session per chatter — deduped by the
                    # `is_first_time` flag captured at buffer time above.
                    try:
                        if self.chat_batch_buffer[-1].get("is_first_time"):
                            n = 1
                            try:
                                if (
                                    ENABLE_CHATTER_MEMORY
                                    and self.memory.count_chatter_messages(username) >= 5
                                ):
                                    n += 1  # returning-regular bonus
                            except Exception:
                                pass
                            self.cookie_jar.add_cookie(username, n)
                            print(
                                f"   [Cookies] +{n} → {username} (first message this session); "
                                f"shared={self.cookie_jar.get_shared()}/35"
                            )
                            await self._broadcast_cookie_state()
                            self._maybe_fire_cookie_milestone()
                    except Exception as _ck_err:
                        print(f"   [Cookies] First-message award error: {_ck_err}")

                    # System 2: reset autopilot dead-chat timer on any chat message
                    if self.vn_autopilot and self.vn_autopilot.is_running:
                        self.vn_autopilot.notify_chat_activity()

                    if self.active_prediction is not None:
                        self._tally_prediction_vote(username, message_body)

                    self.input_queue.task_done()
                    handled_by_chat = True
                    continue

                # === VOICE INPUTS → IMMEDIATE PROCESSING ===
                if self.is_muted():
                    print(f"   [Mute] Dropping {source} input while muted: {content[:60]}")
                else:
                    # System 6b: soft-pause autopilot while Jonny is speaking
                    _ap_soft_paused = (
                        self.vn_autopilot is not None
                        and self.vn_autopilot.is_running
                        and source == "voice"
                    )
                    if _ap_soft_paused:
                        self.vn_autopilot.soft_pause()

                    # Activity auto-detection from voice (natural language sets context)
                    if source == "voice":
                        detected = self._detect_activity_change(content)
                        if detected and detected != self.current_activity:
                            self.current_activity = detected
                            new_type = self._classify_activity_type(detected)
                            self.game_mode_controller.activity_type = new_type
                            self.vision_agent.activity_type = new_type
                            print(f"   [Activity] Set to: '{detected}' (type: {new_type})")
                            old_immersive = self.immersive
                            self.immersive = new_type in (ACTIVITY_VN, ACTIVITY_MEDIA)
                            print(f"   [Immersive] {self.immersive}")
                            # Vision heartbeat cadence:
                            #   ACTIVITY_VN / ACTIVITY_MEDIA (immersive=True) → 10s (already was)
                            #   ACTIVITY_GAME → 10s (game scenes change fast; was 30s before)
                            #   Everything else → 30s (chat/idle, no point hammering vision)
                            if self.immersive or new_type == ACTIVITY_GAME:
                                self.vision_agent.heartbeat_interval = 10.0
                            else:
                                self.vision_agent.heartbeat_interval = 30.0
                            if old_immersive and not self.immersive and self.session_scene_log:
                                asyncio.create_task(self._generate_session_summary())
                            # Load playthrough memory for the new game/VN
                            if self.playthrough_memory and new_type in (ACTIVITY_VN, ACTIVITY_GAME):
                                self.playthrough_memory.load_for_game(detected)
                                # Takes/spotlight persist across activity switches (Req A).

                    # 1. Vision Gating Logic (Optimized for Cost vs Detail)
                    visual_desc = ""
                    # Forced-look pre-step: if Jonny asked a specific visual question
                    # (e.g. "what color are her eyes"), grab a fresh frame and answer
                    # from THAT before the LLM gets a chance to confabulate. This runs
                    # regardless of game_mode_controller state — visual questions need
                    # a real frame, not character priors.
                    forced_visual_answer = ""
                    if source == "voice" and self._is_visual_question(content):
                        print(f"   [Vision] Visual question detected — forcing fresh snapshot before answering: {content[:80]!r}")
                        try:
                            forced_visual_answer = await self.vision_agent.capture_and_answer(content)
                            print(f"   [Vision] Pre-answer look: {forced_visual_answer[:160]}")
                        except Exception as e:
                            print(f"   [Vision] Forced-look failed: {e}")
                            forced_visual_answer = ""

                    if self.game_mode_controller.is_active:
                        lower = content.lower()
                        read_phrases = [
                            'read the text', 'read this', 'read the screen', 'read all',
                            'read it', 'read me', 'read what', 'read out',
                            'reading this', 'reading the', 'reading what',
                            'what does it say', 'what does that say', 'what does the screen say',
                            'what does it read', 'transcribe',
                        ]
                        is_read_request = any(p in lower for p in read_phrases)

                        VISION_KEYWORDS = [
                            'see', 'look', 'read', 'watching', 'view', 'screen',
                            'watch', 'watched', 'scene', 'character', 'who', 'happen', 'happened', 'before',
                        ]
                        vision_trigger = any(word in lower for word in VISION_KEYWORDS)

                        if is_read_request:
                            print("   [Vision] READ intent — transcribing screen verbatim...")
                            transcribed = await self.vision_agent.capture_and_transcribe()

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
                                visual_desc = (
                                    "I tried to read the screen but there's nothing legible right now — "
                                    "probably a transition, image-only frame, or animation. "
                                    "Acknowledge briefly that there's nothing to read at the moment."
                                )
                        elif vision_trigger:
                            # FIX 1+2: Default to the accumulated scene_summary (rolling
                            # "story so far") via get_vision_context(). This is almost
                            # always better context for "what happened / who / what's going
                            # on" than a single fresh frame, AND it removes the 800-1500ms
                            # blocking GPT-4o-mini snapshot call from the response path.
                            #
                            # Exception: explicit "right now" / "on screen now" phrases
                            # where the user wants the current frame state. Even then we
                            # fire the refresh ASYNC and respond with cached context —
                            # the response path NEVER waits on a vision API call.
                            if forced_visual_answer:
                                # Targeted capture_and_answer already ran above — use the
                                # rolling scene summary as additional background context.
                                visual_desc = self.vision_agent.get_vision_context()
                            else:
                                _NOW_PHRASES = (
                                    "right now", "on screen now", "currently on screen",
                                    "on the screen now", "what's on screen",
                                )
                                _wants_current = any(p in lower for p in _NOW_PHRASES)
                                if _wants_current and (time.time() - self.vision_agent.last_capture_time) > 15:
                                    # User wants current frame AND cache is very stale —
                                    # kick off async refresh but don't wait on it.
                                    print("   [Vision] Current-frame requested — async refresh triggered, responding with cached...")
                                    asyncio.create_task(self.vision_agent.capture_and_describe(is_heartbeat=False))
                                visual_desc = self.vision_agent.get_vision_context()
                                if visual_desc:
                                    print("   [Vision] Using accumulated scene context (story-so-far mode)...")
                                else:
                                    # No accumulated context yet — kick async refresh and continue.
                                    print("   [Vision] No accumulated context — async refresh triggered...")
                                    asyncio.create_task(self.vision_agent.capture_and_describe(is_heartbeat=False))
                        else:
                            visual_desc = self.vision_agent.get_vision_context()

                    # If a visual question forced a fresh snapshot, anchor the LLM
                    # to it explicitly — and append the short-term visual memory so
                    # she answers from what she actually saw, not from priors.
                    if forced_visual_answer:
                        recent_mem = self.vision_agent.get_recent_visual_memory(max_age=60.0)
                        anchor = (
                            "[VISUAL STATUS: FRESH — just looked at the screen to answer your question]\n"
                            "Ground your answer ONLY in this targeted observation. Do NOT invent details "
                            "that aren't stated here. If it starts with UNCERTAIN:, acknowledge you can't "
                            "tell rather than guessing.\n"
                            f"Fresh look (in response to: \"{content.strip()}\"): {forced_visual_answer.strip()}"
                        )
                        if recent_mem:
                            anchor += f"\n\nShort-term visual memory (recent frames):\n{recent_mem}"
                        visual_desc = (anchor + "\n\n" + visual_desc) if visual_desc else anchor

                    # Media Watch episode log injection — when active, prepend
                    # the rolling event timeline so question-answering draws on
                    # the sequence rather than a single stale snapshot.
                    if self.media_watch and self.media_watch.is_running and self.media_watch.has_context():
                        mw_ctx = self.media_watch.get_episode_context()
                        if mw_ctx:
                            visual_desc = (mw_ctx + "\n\n" + visual_desc) if visual_desc else mw_ctx

                    # 2. Construct dialogue line (history-clean — no screen state)
                    dialogue_line = f"Jonny says: \"{content}\""

                    # Speech triage — decide whether to respond, react briefly, or stay quiet
                    scene_ctx = self.vision_agent.get_vision_context() if self.game_mode_controller.is_active else ""

                    # Cutscene bias (ACTIVITY_GAME only): if a cutscene is likely playing and
                    # Jonny has been silent for >20s, pass immersive=True to triage so it biases
                    # toward STAY_QUIET / BRIEF instead of RESPOND. _triage_rescue still fires for
                    # direct addresses and questions, so chat viewers can still get responses.
                    silence_since_last = time.time() - self.last_interaction_time
                    _cutscene_active = self._is_likely_cutscene() and silence_since_last > 20.0
                    _triage_immersive = self.immersive or _cutscene_active

                    # FIX 3+4: Run triage (Groq network call) and memory retrieval
                    # (ChromaDB vector search) CONCURRENTLY instead of sequentially.
                    # Saves 200-400ms/turn. ChromaDB runs in a thread (to_thread) so
                    # it no longer blocks the event loop either.
                    _triage_t0 = time.time()
                    decision, prefetched_memory = await asyncio.gather(
                        self.ai_core.decide_response_mode(
                            recent_history=self.conversation_history,
                            incoming_line=content,
                            scene_context=scene_ctx,
                            source=source,
                            immersive=_triage_immersive,
                            streamer_mode=(self.mode == "streamer"),
                        ),
                        asyncio.to_thread(self.memory.get_semantic_context, content),
                    )
                    try:
                        self.stream_logger.log(
                            "triage_decision",
                            input=content[:200],
                            result=decision,
                            latency_ms=int((time.time() - _triage_t0) * 1000),
                            cutscene=_cutscene_active,
                        )
                    except Exception:
                        pass

                    if decision == "STAY_QUIET":
                        print(f"   [Triage] STAY_QUIET \u2014 letting it pass.")
                        # fall through to finally / task_done

                    if decision != "STAY_QUIET":
                        brief_mode = (decision == "BRIEF")
                        if _triage_immersive and decision == "RESPOND":
                            brief_mode = True
                        print(f"   [Triage] {decision}")

                        await self.process_and_respond(
                            content,
                            dialogue_line,
                            "user",
                            source=source,
                            situational_context=visual_desc,
                            brief_mode=brief_mode,
                            prefetched_memory=prefetched_memory,
                        )

                    # System 6b: release soft-pause after Jonny's exchange is fully handled
                    if _ap_soft_paused and self.vn_autopilot:
                        await asyncio.sleep(1.2)  # brief natural gap before resuming
                        self.vn_autopilot.soft_resume()
                        _ap_soft_paused = False

            except Exception as e:
                print(f"   [Brain] Error: {e}")
                traceback.print_exc()
            finally:
                if not handled_by_chat:
                    self.input_queue.task_done()

    def get_chat_rate_per_min(self) -> float:
        """Returns chat messages per minute over the last 60 seconds."""
        cutoff = time.time() - 60
        count = sum(1 for t in self.chat_msg_timestamps if t > cutoff)
        return float(count)


    async def chat_batch_worker(self):
        """Drains the chat batch buffer every CHAT_BATCH_WINDOW seconds and emits
        at most one response per batch. Handles multi-chatter prioritization,
        cooldowns, and engagement mechanics."""
        print("   [System] Chat Batch Worker started.")
        while self.is_running:
            await asyncio.sleep(CHAT_BATCH_WINDOW)

            if not self.chat_batch_buffer:
                continue

            if self.is_muted() or self.ai_core.is_speaking or self.processing_lock.locked():
                continue

            if time.time() - self.last_chat_response_time < CHAT_RESPONSE_COOLDOWN:
                continue

            batch = self.chat_batch_buffer[:]
            self.chat_batch_buffer.clear()

            try:
                await self._respond_to_chat_batch(batch)
            except Exception as e:
                print(f"   [ChatBatch] Error: {e}")
                traceback.print_exc()

    async def _respond_to_chat_batch(self, batch: list):
        """Decides what (if anything) to say in response to a batch of chat messages."""
        if not batch:
            return

        now = time.time()

        # --- Change 1: Log each chatter's message to the session rolling log ---
        for msg in batch:
            username = msg.get("username", "unknown")
            content = msg.get("message", "")
            if username not in self.session_chatter_logs:
                self.session_chatter_logs[username] = []
                self.session_chatter_first_seen[username] = now
            self.session_chatter_logs[username].append({"content": content, "timestamp": now})
            # Keep only last 15 per chatter
            self.session_chatter_logs[username] = self.session_chatter_logs[username][-15:]
            self.session_chatter_last_spoke[username] = now

        # --- Change 2: Detect returning regulars (>10 historical msgs, first message this session) ---
        returning_regulars = []
        for msg in batch:
            username = msg.get("username", "unknown")
            is_first_this_session = abs(self.session_chatter_first_seen.get(username, 0) - now) < 0.1
            if is_first_this_session:
                historical_count = self.memory.count_chatter_messages(username)
                if historical_count >= 5:
                    returning_regulars.append((username, historical_count))

        returning_regulars_block = ""
        if returning_regulars:
            lines = []
            for username, count in returning_regulars:
                lines.append(
                    f"- {username} (has sent ~{count} messages across past sessions — a known regular)"
                )
            returning_regulars_block = (
                "\n\n[RETURNING REGULARS — these chatters are showing up after a gap]\n"
                + "\n".join(lines)
                + "\nAcknowledge them naturally and warmly in your response — this is their first message "
                "this session and they're regulars. Don't be cheesy about it, but make them feel seen. "
                "Reference something specific you know about them when possible.\n"
            )

        # --- Change 5: Build per-chatter scoped context blocks to prevent attribution bleed ---
        first_timers = []
        unique_users_in_batch = set(msg["username"] for msg in batch)
        chatter_context_blocks = []
        for msg in batch:
            username = msg["username"]
            if msg["is_first_time"]:
                first_timers.append(username)
        for username in unique_users_in_batch:
            if ENABLE_CHATTER_MEMORY:
                ctx = self.memory.get_chatter_context(username, n_results=3)
                if ctx:
                    chatter_context_blocks.append(
                        f"--- ABOUT {username} (these facts ONLY apply to {username}, not anyone else) ---\n{ctx}\n"
                    )
        chatter_context = "\n".join(chatter_context_blocks) if chatter_context_blocks else "(no prior context on these chatters)"

        # --- Change 1: Augment with this-session message history per chatter ---
        session_history_block = ""
        for username in unique_users_in_batch:
            log = self.session_chatter_logs.get(username, [])
            prior = [entry for entry in log if entry["timestamp"] < now - 1.0]
            if len(prior) >= 2:
                recent_lines = [f'  "{e["content"]}"' for e in prior[-8:]]
                session_history_block += (
                    f"\n[{username} earlier this session, in order]:\n" + "\n".join(recent_lines) + "\n"
                )
        if session_history_block:
            chatter_context += "\n\n=== THIS SESSION'S CHAT HISTORY ===" + session_history_block

        # --- Change 3: Running bits block ---
        running_bits_block = ""
        if self.session_running_bits:
            bits_str = "\n".join(
                f"- {b['name']}: {b['description']}" for b in self.session_running_bits[-5:]
            )
            running_bits_block = (
                f"\n[RUNNING BITS THIS SESSION \u2014 if any is genuinely relevant to this batch, "
                f"drop the callback now; don't force it, but don't sit on it either]\n{bits_str}\n"
            )

        batch_lines = []
        for msg in batch:
            marker = " [FIRST TIME CHATTER]" if msg["is_first_time"] else ""
            batch_lines.append(f"  - {msg['username']} ({msg['platform']}){marker}: {msg['message']}")
        batch_str = "\n".join(batch_lines)

        scene = ""
        if self.game_mode_controller.is_active:
            scene = self.vision_agent.get_vision_context()
        if self.audio_agent and self.audio_agent.is_active():
            audio_ctx = self.audio_agent.get_audio_context()
            if audio_ctx:
                scene = (scene + "\n" + audio_ctx) if scene else audio_ctx

        # Prepend recent activity brief so Kira has stream-level context for chat batches too
        session_context_block = ""
        if self.recent_activity_brief:
            session_context_block = f"\n[LAST SESSION RECAP]\n{self.recent_activity_brief}\n\n"
        # Append playthrough memory context (current game summary + games history manifest)
        if self.playthrough_memory:
            pt_ctx = self.playthrough_memory.get_context_for_prompt()
            if pt_ctx:
                session_context_block += f"[PLAYTHROUGH MEMORY — reference as lived experience]\n{pt_ctx}\n\n"
        # Mid-session rolling takes — lets chat responses callback to opinions
        # she's already stated in this session, not just on-disk ones.
        if self.session_takes_summary:
            session_context_block += (
                f"[MY TAKES SO FAR THIS SESSION — callbacks welcome]\n"
                f"{self.session_takes_summary}\n\n"
            )

        request = (
            f"You have a batch of {len(batch)} chat message(s) to respond to. "
            f"Decide the best engagement move:\n\n"
            f"{session_context_block}"
            f"{returning_regulars_block}"
            f"{running_bits_block}"
            f"CHAT BATCH:\n{batch_str}\n\n"
            f"WHAT YOU KNOW ABOUT THESE CHATTERS:\n{chatter_context}\n\n"
            f"CURRENT SCENE: {scene or 'no scene context'}\n\n"
            f"RULES:\n"
            f"- Address chatters BY NAME. Name recognition is your superpower.\n"
            f"- If someone is a FIRST TIME CHATTER, give them a brief warm spotlight moment.\n"
            f"- If you have prior context on a chatter, reference it naturally (callbacks land hard).\n"
            f"- If multiple messages have the same vibe, consolidate.\n"
            f"- If messages are pure spam/'hi'/no substance AND you have zero prior context on the chatter, output ONLY: SKIP\n"
            f"- Exception: if you have ANY prior context on a chatter (even one fact), a simple greeting is NOT skip-worthy — give them a quick warm acknowledgment. Known viewers saying 'hi' should never be SKIP.\n"
            f"- Length scales with batch size: 1 chatter = 1-2 sentences (a quick aside, not a full monologue). 2-3 chatters = 2-3 sentences. 4+ chatters = up to 4 sentences max. NEVER more than 4 sentences regardless of size.\n"
            f"- You are a stream co-host weaving chat into the conversation, not a chat reader. The shorter and punchier, the better.\n"
            f"- Stay in character \u2014 sassy, witty, warm, deadpan.\n"
            f"- DO NOT respond if there's nothing real to say. SKIP is a valid output.\n\n"
            f"IMPORTANT: When you reference chatter facts, only attribute them to the specific chatter "
            f"they belong to. Do not mix up facts between chatters. If you're not sure who said/did something, "
            f"don't attribute it.\n\n"
            f"Your response (or SKIP):"
        )

        # Cap response length based on batch size — solo chatter shouldn't get a monologue
        batch_size = len(batch)
        if batch_size == 1:
            chat_max_tokens = 120
        elif batch_size <= 3:
            chat_max_tokens = 200
        else:
            chat_max_tokens = 280

        if self.ai_core.anthropic_client:
            response = await self.ai_core.kira_deep_response(
                request=request + self._kira_voice_guardrails(),
                scene_context=scene,
                memory_context=self.memory.get_semantic_context(batch_str),
                recent_history=self.conversation_history,
                max_tokens=chat_max_tokens,
            )
        else:
            response = await self.ai_core.llm_inference(
                messages=self.conversation_history + [{"role": "system", "content": request + self._kira_voice_guardrails()}],
                current_emotion=self.current_emotion,
                memory_context=self.memory.get_semantic_context(batch_str),
                activity_context=self.current_activity,
            )

        cleaned = self.ai_core._clean_llm_response(response).strip()
        if not cleaned or cleaned.upper().startswith("SKIP") or len(cleaned) < 5:
            print(f"   [ChatBatch] SKIP \u2014 {len(batch)} message(s) didn't warrant a response")
            return

        print(f"   >>> Kira (Chat Batch of {len(batch)}): {cleaned}")
        await self.ai_core.speak_text(cleaned)
        self.conversation_history.append({"role": "assistant", "content": cleaned})
        chatter_names = ", ".join(sorted(set(m["username"] for m in batch)))

        # ── Cookies: respond-to-chat award ──
        # Kira emitted a non-SKIP response — award +1 to each unique chatter in
        # the batch. Per-chatter dedup is automatic (set on username) and the
        # batch is consumed once, so the same message can't earn twice.
        try:
            awarded_users = set()
            for msg in batch:
                u = msg.get("username", "")
                if not u or u in awarded_users:
                    continue
                awarded_users.add(u)
                self.cookie_jar.add_cookie(u, 1)
            if awarded_users:
                print(
                    f"   [Cookies] +1 × {len(awarded_users)} (batch response); "
                    f"shared={self.cookie_jar.get_shared()}/35"
                )
                await self._broadcast_cookie_state()
                self._maybe_fire_cookie_milestone()
        except Exception as _ck_err:
            print(f"   [Cookies] Batch-response award error: {_ck_err}")
        # Tag notable chat moments for the playthrough record when in VN/game mode
        if self.playthrough_memory and self.playthrough_memory.current_slug:
            in_vn_or_game = (
                self.game_mode_controller.is_active
                and self.game_mode_controller.activity_type in (ACTIVITY_VN, ACTIVITY_GAME)
            )
            if in_vn_or_game:
                for msg in batch:
                    self.playthrough_memory.tag_chat_moment(
                        msg["username"], msg["message"], cleaned
                    )
        self._log_session_turn(
            role="user",
            content=f"[Chat batch from {chatter_names}]: " + " | ".join(m["message"] for m in batch),
            speaker_name=chatter_names,
        )
        self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira")
        self.last_chat_response_time = time.time()
        self.ai_core.last_speech_finish_time = time.time()

        for msg in batch:
            self.chatter_last_response[msg["username"]] = time.time()

        if ENABLE_CHATTER_MEMORY:
            asyncio.create_task(self._extract_chatter_facts(batch, cleaned))

    async def _extract_chatter_facts(self, batch: list, kira_response: str):
        """Background pass: extracts durable facts about chatters from a batch."""
        if not self.ai_core.anthropic_client:
            return

        for msg in batch:
            username = msg["username"]
            message = msg["message"]

            if len(message) < 8:
                continue

            try:
                system = (
                    "You extract durable facts about a chatter for an AI VTuber's persistent memory. "
                    "Only save things that will still matter next week \u2014 opinions, preferences, jokes "
                    "they've made, things they're known for. Skip greetings, reactions, generic statements.\n\n"
                    "Output a JSON object with two fields:\n"
                    '  {"fact": "short sentence or NONE", "tone": "one of: wholesome|chaotic|supportive|dry|sharp|earnest|playful|challenging"}\n\n'
                    "Example: {\"fact\": \"Thinks Ferris is a war criminal\", \"tone\": \"dry\"}\n"
                    "Example: {\"fact\": \"NONE\", \"tone\": \"wholesome\"}\n"
                    "If no durable fact exists, use \"NONE\" for fact but still include tone.\n"
                    "Output ONLY the JSON object, nothing else."
                )
                user = (
                    f"Chatter: {username}\n"
                    f"Message: \"{message}\"\n"
                    f"Kira's response: \"{kira_response[:200]}\"\n\n"
                    f"Extract a durable fact and tonal tag for this chatter."
                )
                raw = await self.ai_core.claude_chat_inference(
                    messages=[{"role": "user", "content": user}],
                    system_prompt=system,
                    max_tokens=80,
                )
                if not raw:
                    continue
                raw = raw.strip()
                # Parse JSON response
                import json as _json
                fact = ""
                tone = ""
                try:
                    parsed = _json.loads(raw)
                    fact = (parsed.get("fact") or "").strip().rstrip(".")
                    tone = (parsed.get("tone") or "").strip()
                except Exception:
                    # Fallback: treat entire response as fact (old format)
                    fact = raw.rstrip(".")
                if fact and fact.upper() != "NONE" and 5 < len(fact) < 200:
                    self.memory.store_chatter_fact(username, msg["platform"], fact, tone=tone)
            except Exception as e:
                print(f"   [ChatterFact] extract failed for {username}: {e}")

    async def extract_running_bits(self, response_text: str, user_text: str = "") -> None:
        """After each substantive exchange, check if a new running bit has emerged
        or an existing one has been called back. Lightweight — uses Sonnet, not Opus."""

        if not self.ai_core.anthropic_client or not response_text:
            return

        # Skip very short exchanges — bits don't form from one-liners
        if len(response_text) < 80:
            return

        existing_bits_str = ""
        if self.session_running_bits:
            existing_bits_str = "Existing bits this session:\n" + "\n".join(
                f"- {b['name']}: {b['description']}" for b in self.session_running_bits[-20:]
            )

        prompt = (
            "Analyze the following exchange and identify if a NEW recurring bit / callback / "
            "in-joke has emerged, OR if an existing bit was called back.\n\n"
            f"USER SAID: {user_text}\n\n"
            f"KIRA RESPONDED: {response_text}\n\n"
            f"{existing_bits_str}\n\n"
            "Output ONE of these formats:\n"
            '- NEW: {"name": "short bit name", "description": "one-line description"}\n'
            '- CALLBACK: existing bit name\n'
            "- NONE\n\n"
            "Only flag NEW if it's genuinely repeatable (a phrase, character trait, recurring reference). "
            "Don't flag one-off jokes."
        )

        try:
            result = await self.ai_core.claude_chat_inference(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a comedy editor identifying running gags. Output exact format only.",
                max_tokens=100,
            )
            if not result:
                return
            result = result.strip()
            if result.startswith("NEW:"):
                import json
                try:
                    bit_json = result.replace("NEW:", "").strip()
                    bit = json.loads(bit_json)
                    if "name" in bit and "description" in bit:
                        if not any(b["name"].lower() == bit["name"].lower() for b in self.session_running_bits):
                            bit["last_called_back_at"] = 0.0
                            self.session_running_bits.append(bit)
                            print(f"   [Bits] New running bit: {bit['name']}")
                except Exception:
                    pass
            elif result.startswith("CALLBACK:"):
                bit_name = result.replace("CALLBACK:", "").strip()
                for b in self.session_running_bits:
                    if b["name"].lower() == bit_name.lower():
                        b["last_called_back_at"] = time.time()
                        print(f"   [Bits] Callback: {b['name']}")
                        break
        except Exception as e:
            print(f"   [Bits] Extraction error: {e}")

    # ── Chat Predictions ──────────────────────────────────────────────────────

    def start_prediction(self, question: str, option_a: str, option_b: str, duration_seconds: int = 30):
        """Starts a chat-based prediction. Viewers vote by typing A or B (or the option name)."""
        self.active_prediction = {
            "question": question,
            "option_a": option_a,
            "option_b": option_b,
            "votes_a": set(),
            "votes_b": set(),
            "ends_at": time.time() + duration_seconds,
        }
        asyncio.create_task(self._prediction_announce_start())
        asyncio.create_task(self._prediction_close_after(duration_seconds))

    async def _prediction_announce_start(self):
        if not self.active_prediction:
            return
        p = self.active_prediction
        text = (
            f"Okay chat, prediction time. {p['question']} "
            f"Type A for {p['option_a']}, or B for {p['option_b']}. "
            f"You have {int(p['ends_at'] - time.time())} seconds. Go."
        )
        await self.ai_core.speak_text(text)
        self.conversation_history.append({"role": "assistant", "content": text})

    async def _prediction_close_after(self, seconds: int):
        await asyncio.sleep(seconds)
        if not self.active_prediction:
            return
        p = self.active_prediction
        a, b = len(p["votes_a"]), len(p["votes_b"])
        if a == 0 and b == 0:
            text = f"Chat is dead. Nobody voted. I am taking this personally."
        elif a > b:
            text = f"{p['option_a']} wins, {a} to {b}. Chat has spoken. Jonny, you know what to do."
        elif b > a:
            text = f"{p['option_b']} wins, {b} to {a}. The people have decided. Make it happen, Jonny."
        else:
            text = f"It's a tie, {a} to {b}. Chat cannot agree on anything. Embarrassing. Jonny picks."
        self.active_prediction = None
        await self.ai_core.speak_text(text)
        self.conversation_history.append({"role": "assistant", "content": text})

    def _tally_prediction_vote(self, username: str, message: str):
        """Called from brain_worker when chat arrives during an active prediction."""
        if not self.active_prediction:
            return
        p = self.active_prediction
        msg = message.strip().upper()
        voted_a = msg == "A" or p["option_a"].lower() in message.lower()
        voted_b = msg == "B" or p["option_b"].lower() in message.lower()
        if voted_a and not voted_b:
            p["votes_a"].add(username)
            p["votes_b"].discard(username)
        elif voted_b and not voted_a:
            p["votes_b"].add(username)
            p["votes_a"].discard(username)

    async def run_stream_opener(self):
        """Generates and speaks a scripted episodic opener for the stream.
        Pulls last session's summary, recognizes returning chatters, sets the tone."""
        if self.processing_lock.locked() or self.ai_core.is_speaking:
            print("   [Opener] Busy — try again in a moment.")
            return

        async with self.processing_lock:
            print("   [Opener] Preparing stream opener...")

            # Reset the cookie jar and session-chatter tracking for the new
            # stream. Triggered here (opener = Go Live) so a mid-stream bot
            # restart does NOT clear the jar — only a deliberate stream start does.
            self.cookie_jar.reset_shared_on_stream_start()
            self.session_chatters_seen.clear()
            await self._broadcast_cookie_state()


            last_session = self.memory.get_last_session_summary() or "(no prior session on record)"
            recent_chatters = self.memory.get_recent_chatters(days=14, limit=10)
            chatter_list = ", ".join(recent_chatters) if recent_chatters else "(no recognized regulars yet)"

            scene = self.vision_agent.get_vision_context() if self.game_mode_controller.is_active else "(observer mode off)"

            request = (
                f"This is the opening moment of a fresh stream. Jonny just hit 'Go Live'. "
                f"You are Kira, the co-host. Greet the audience with energy and personality. "
                f"Make it feel like the start of an episode of a show — not a chatbot saying hi.\n\n"
                f"What to weave in:\n"
                f"- A line acknowledging the audience is here (don't read a list of names)\n"
                f"- If returning regulars are likely watching, name 2-3 of them and reference what you know about them\n"
                f"- A one-line recap or callback to last session if it exists\n"
                f"- A brief tease of what's planned for today (the current activity or scene)\n"
                f"- Hand it back to Jonny at the end ('alright, take it away' or similar)\n\n"
                f"CONTEXT:\n"
                f"- Last session's summary: {last_session}\n"
                f"- Returning chatters (most active first): {chatter_list}\n"
                f"- Current activity: {self.current_activity or 'no activity set yet'}\n"
                f"- Current scene: {scene}\n\n"
                f"Keep it under 30 seconds spoken (~80 words). Stay in character — sassy, warm, deadpan."
            )

            if self.ai_core.anthropic_client:
                response = await self.ai_core.kira_deep_response(
                    request=request,
                    scene_context=scene,
                    memory_context="",
                    recent_history=[],
                )
            else:
                response = await self.ai_core.llm_inference(
                    messages=[{"role": "user", "content": request}],
                    current_emotion=self.current_emotion,
                    memory_context="",
                    activity_context=self.current_activity,
                )

            cleaned = self.ai_core._clean_llm_response(response)
            if cleaned and len(cleaned) > 10:
                print(f"   >>> Kira (Opener): {cleaned}")
                await self.ai_core.speak_text(cleaned)
                self.conversation_history.append({"role": "assistant", "content": cleaned})
                self._log_session_turn("assistant", cleaned, speaker_name="Kira (opener)")
                self.ai_core.last_speech_finish_time = time.time()

    async def run_stream_closer(self):
        """Generates and speaks an episodic outro, then writes session artifacts (lore + clips)."""
        if self.processing_lock.locked() or self.ai_core.is_speaking:
            print("   [Closer] Busy — try again in a moment.")
            return

        async with self.processing_lock:
            print("   [Closer] Preparing stream closer...")

            highlights_text = "\n".join(
                f"- {h['highlight']}" + (f" (Kira: {h['take']})" if h.get('take') else "")
                for h in self.session_highlights[-10:]
            ) or "(no highlights captured)"

            seen = list(self.session_chatters_seen)
            chatter_mentions = ", ".join(seen[:5]) if seen else "(quiet session)"

            request = (
                f"This is the closing moment of a stream. Jonny is about to hit 'End Stream'. "
                f"You are Kira, wrapping up the episode.\n\n"
                f"What to weave in:\n"
                f"- A callback to one or two specific moments from today's session\n"
                f"- A genuine shoutout to the most active chatters by name\n"
                f"- A small tease for next time (return to a running bit, ongoing storyline, etc.)\n"
                f"- A real goodnight — warm, not generic\n\n"
                f"CONTEXT:\n"
                f"- Today's session highlights: {highlights_text}\n"
                f"- Active chatters today: {chatter_mentions}\n"
                f"- Activity: {self.current_activity}\n\n"
                f"Keep it under 25 seconds spoken (~70 words). End on Kira's voice. No questions back to Jonny — this is goodbye."
            )

            if self.ai_core.anthropic_client:
                response = await self.ai_core.kira_deep_response(
                    request=request,
                    scene_context="",
                    memory_context="",
                    recent_history=self.conversation_history,
                )
            else:
                response = await self.ai_core.llm_inference(
                    messages=[{"role": "user", "content": request}],
                    current_emotion=self.current_emotion,
                    memory_context="",
                    activity_context=self.current_activity,
                )

            cleaned = self.ai_core._clean_llm_response(response)
            if cleaned and len(cleaned) > 10:
                print(f"   >>> Kira (Closer): {cleaned}")
                await self.ai_core.speak_text(cleaned)
                self.conversation_history.append({"role": "assistant", "content": cleaned})
                self._log_session_turn("assistant", cleaned, speaker_name="Kira (closer)")
                self.ai_core.last_speech_finish_time = time.time()

            try:
                await self._write_session_artifacts()
            except Exception as e:
                print(f"   [Closer] Artifact generation failed: {e}")

    async def _write_session_artifacts(self) -> dict:
        """At end of session, generate lore + clip candidate artifacts via Opus.

        Returns a dict of what was actually written:
            {"raw_dump": path|None, "lore": path|None, "clips": path|None,
             "playthrough": path|None, "skipped_reason": str|None}

        Hardening rules (this method runs on the daemon asyncio thread, which
        can be killed instantly when the dashboard window closes):
          1. The RAW transcript+highlights dump is written FIRST and synchronously
             before any LLM call — so even if Opus hangs and the daemon thread
             gets killed during shutdown, the session data still survives.
          2. Each LLM-dependent write (lore, clips, playthrough) is wrapped in
             its OWN try/except. One Opus failure can't sink the others.
          3. Nothing in this method is allowed to raise to the caller.
        """
        results: dict = {
            "raw_dump": None, "lore": None, "clips": None,
            "playthrough": None, "skipped_reason": None,
        }

        if self._session_artifacts_written:
            print("   [Artifacts] Already written for this session — skipping.")
            results["skipped_reason"] = "already_written"
            return results

        if not self.full_session_log:
            print("   [Artifacts] No session log to process — skipping.")
            results["skipped_reason"] = "empty_log"
            return results

        activity = self.current_activity or "general"
        activity_slug = re.sub(r'[^a-zA-Z0-9]+', '_', activity).strip('_').lower()[:40] or "session"
        date_str = datetime.now().strftime("%Y-%m-%d")
        session_duration_min = int((time.time() - self.session_started_at) / 60)

        transcript_lines = []
        for entry in self.full_session_log:
            rel_sec = int(entry["timestamp"] - self.session_started_at)
            h = rel_sec // 3600
            m = (rel_sec % 3600) // 60
            s = rel_sec % 60
            ts = f"{h:02d}:{m:02d}:{s:02d}"
            speaker = entry.get("speaker_name", entry["role"])
            content = entry["content"][:600]
            transcript_lines.append(f"[{ts}] {speaker}: {content}")

        transcript = "\n".join(transcript_lines)

        highlights_lines = [
            f"- {h['highlight']}" + (f" — Kira's take: {h['take']}" if h.get('take') else "")
            for h in self.session_highlights
        ]
        highlights_block = "\n".join(highlights_lines) if highlights_lines else "(none captured)"

        # ── STAGE 0: Raw dump (no LLM, no network). Always runs first. ──
        # This is the unkillable fallback: if every Opus call below dies, the
        # raw session content still lives on disk for manual review.
        try:
            os.makedirs("logs/sessions_raw", exist_ok=True)
            ts_tag = datetime.now().strftime("%Y-%m-%d_%H-%M")
            raw_path = os.path.join("logs/sessions_raw", f"{ts_tag}_{activity_slug}.md")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(f"# Raw Session Dump — {activity}\n\n")
                f.write(f"**Date:** {date_str}  \n")
                f.write(f"**Duration:** ~{session_duration_min} min  \n")
                f.write(f"**Highlights captured:** {len(self.session_highlights)}\n\n")
                f.write("## Highlights\n\n")
                f.write(highlights_block + "\n\n")
                f.write("## Full Transcript\n\n```\n")
                f.write(transcript)
                f.write("\n```\n")
                f.flush()
                os.fsync(f.fileno())
            results["raw_dump"] = raw_path
            print(f"   [Artifacts] Raw dump → {raw_path}")
        except Exception as e:
            print(f"   [Artifacts] Raw dump failed: {e}")
            traceback.print_exc()

        if not self.ai_core.anthropic_client:
            print("   [Artifacts] Claude unavailable — skipping LLM artifacts (would produce garbage on local Llama).")
            results["skipped_reason"] = "no_claude"
            self._session_artifacts_written = True
            return results

        # Truncate transcript for the LLM only (raw dump above kept the full version).
        llm_transcript = transcript
        if len(llm_transcript) > 80000:
            llm_transcript = llm_transcript[:16000] + "\n\n[... middle of session truncated for length ...]\n\n" + llm_transcript[-40000:]

        artifact_request = (
            f"You are reviewing a full stream session transcript for the AI VTuber Kira. "
            f"Activity: {activity}. Duration: ~{session_duration_min} minutes. Date: {date_str}.\n\n"
            f"You will produce TWO outputs, separated by the exact delimiter line `===CLIPS===`.\n\n"
            f"OUTPUT 1 — LORE NOTES (markdown). Identify 3-7 durable canon points established or developed "
            f"this session for this activity. Format as bullet points.\n\n"
            f"OUTPUT 2 — CLIP CANDIDATES (markdown). Identify 8-12 of the funniest, sharpest, or most "
            f"emotionally landing moments. For each one provide:\n"
            f"  ### Clip N — Short title\n"
            f"  **Timestamp:** approximate HH:MM:SS into stream\n"
            f"  **Why it's good:** 1-2 sentences\n"
            f"  **Suggested YouTube short title:** under 60 chars\n"
            f"  **Key exchange:** 2-4 quoted lines\n\n"
            f"=== TRANSCRIPT ===\n{llm_transcript}\n\n"
            f"=== HIGHLIGHTS CAPTURED LIVE ===\n{highlights_block}\n\n"
            f"Begin output. Lore first, then `===CLIPS===` on its own line, then clip candidates."
        )

        # ── STAGE 1: Opus call for lore + clips (60s timeout). ──
        print("   [Artifacts] Calling Opus to generate lore + clip candidates...")
        response = None
        try:
            response = await asyncio.wait_for(
                self.ai_core.claude_inference(
                    messages=[{"role": "user", "content": artifact_request}],
                    system_prompt="You are a thoughtful editor reviewing a stream session. Output clean markdown.",
                    max_tokens=4000,
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            print("   [Artifacts] Opus call TIMED OUT after 60s — raw dump survived; lore/clips skipped.")
        except Exception as e:
            print(f"   [Artifacts] Opus call failed: {e}")
            traceback.print_exc()

        if response:
            if "===CLIPS===" in response:
                lore_section, clips_section = response.split("===CLIPS===", 1)
            else:
                lore_section, clips_section = "", response
            lore_section = lore_section.strip()
            clips_section = clips_section.strip()

            # ── STAGE 2: Lore write (independent try). ──
            if lore_section and len(lore_section) > 20:
                try:
                    os.makedirs("lore", exist_ok=True)
                    lore_path = os.path.join("lore", f"{activity_slug}.md")
                    header = f"\n\n## Session: {date_str} ({session_duration_min} min)\n\n"
                    with open(lore_path, "a", encoding="utf-8") as f:
                        if not os.path.exists(lore_path) or os.path.getsize(lore_path) == 0:
                            f.write(f"# Lore: {activity}\n")
                        f.write(header)
                        f.write(lore_section)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    results["lore"] = lore_path
                    print(f"   [Artifacts] Lore appended → {lore_path}")
                except Exception as e:
                    print(f"   [Artifacts] Lore write failed: {e}")
                    traceback.print_exc()

            # ── STAGE 3: Clips write (independent try). ──
            if clips_section and len(clips_section) > 50:
                try:
                    os.makedirs("clips", exist_ok=True)
                    clip_path = os.path.join("clips", f"{date_str}_{activity_slug}.md")
                    with open(clip_path, "w", encoding="utf-8") as f:
                        f.write(f"# Clip Candidates — {activity}\n\n")
                        f.write(f"**Date:** {date_str}  \n")
                        f.write(f"**Duration:** ~{session_duration_min} minutes  \n")
                        f.write(f"**Activity:** {activity}\n\n---\n\n")
                        f.write(clips_section)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    results["clips"] = clip_path
                    print(f"   [Artifacts] Clip candidates written → {clip_path}")
                except Exception as e:
                    print(f"   [Artifacts] Clip write failed: {e}")
                    traceback.print_exc()

        # ── STAGE 4: Playthrough record (independent try; own Opus call inside). ──
        if self.playthrough_memory and self.playthrough_memory.current_slug:
            try:
                narrative = ""
                if self.vn_autopilot and self.vn_autopilot.vn_narrative_summary:
                    narrative = self.vn_autopilot.vn_narrative_summary
                transcript_snippet = "\n".join(
                    f"[{e.get('speaker_name', e['role'])}]: {e['content'][:300]}"
                    for e in self.full_session_log[-60:]
                )
                open_theories = None
                char_attachment = None
                if self.vn_autopilot:
                    open_theories = [t for t in self.vn_autopilot.active_theories
                                     if t["status"] == "open"] or None
                    char_attachment = dict(self.vn_autopilot.character_attachment) or None
                ok = await asyncio.wait_for(
                    self.playthrough_memory.append_session_entry(
                        activity=activity,
                        date_str=date_str,
                        session_duration_min=session_duration_min,
                        narrative_summary=narrative,
                        recent_transcript=transcript_snippet,
                        open_theories=open_theories,
                        character_attachment=char_attachment,
                    ),
                    timeout=60.0,
                )
                if ok:
                    results["playthrough"] = self.playthrough_memory._game_path(
                        self.playthrough_memory.current_slug
                    )
            except asyncio.TimeoutError:
                print("   [Playthrough] Session entry TIMED OUT after 60s — skipped.")
            except Exception as e:
                print(f"   [Playthrough] Session entry generation failed: {e}")
                traceback.print_exc()
        else:
            print("   [Playthrough] No active playthrough slug — skipping session entry.")

        self._session_artifacts_written = True
        return results

    async def request_thoughts(self):
        """Triggered by the dashboard 'Invite' button. Asks Kira to share her honest
        take on whatever is happening on screen right now. Uses the deep brain (Claude Opus)
        when available \u2014 this is the moment where intelligence matters most."""
        if self.processing_lock.locked() or self.ai_core.is_speaking:
            return
        if self.is_muted():
            print("   [Mute] Invite ignored — Kira is muted")
            return
        async with self.processing_lock:
            scene = self.vision_agent.get_vision_context()
            if self.audio_agent and self.audio_agent.is_active():
                audio_ctx = self.audio_agent.get_audio_context()
                if audio_ctx:
                    scene = (scene + "\n" + audio_ctx) if scene else audio_ctx
            memory = self.memory.get_semantic_context(f"thoughts on {self.current_activity}")

            # Inject playthrough memory so invites can draw on the current arc / past games
            playthrough_block = ""
            if self.playthrough_memory:
                pt_ctx = self.playthrough_memory.get_context_for_prompt()
                if pt_ctx:
                    playthrough_block = (
                        f"\n\n[PLAYTHROUGH MEMORY \u2014 reference as lived experience, not data]\n{pt_ctx}"
                    )

            request = (
                f"Jonny just invited you to share your thoughts on what's happening right now. "
                f"This is a moment between you two \u2014 a couch friend sharing a take, not a chatbot. "
                f"React to a specific character, plot beat, or detail you noticed. Use their names. "
                f"Be funny, sassy, sweet, weird, blunt, whatever fits the moment. Keep it conversational, "
                f"2-4 sentences. Don't ask what Jonny thinks \u2014 share your own take."
                f"{playthrough_block}"
                f"{self._kira_voice_guardrails()}"
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
                self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira (invited)")
                self.ai_core.last_speech_finish_time = time.time()

    async def dynamic_observer_loop(self):
        print("   [System] Observer Loop Active (Universal Boredom Protocol).")
        while self.is_running:
            await asyncio.sleep(1.0) # Check every second

            # Suppress observer while autopilot is actively running — avoid double-talking
            if self.vn_autopilot and self.vn_autopilot.is_running:
                continue

            # Don't interrupt if speaking or processing
            if self.processing_lock.locked() or self.ai_core.is_speaking:
                continue

            if self.is_muted():
                continue

            # Compute silence duration here so it's available in both immersive and normal paths
            last_activity = max(self.last_interaction_time, self.ai_core.last_speech_finish_time)
            silence_duration = time.time() - last_activity

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
                        if self._has_fresh_visual_context():
                            scene = self.vision_agent.get_vision_context()
                            prompt = (
                                f"You and Jonny are watching {self.current_activity or 'something'} together. "
                                f"Current scene: {scene}\n\n"
                                f"Drop a brief, natural observation about a character, the mood, or something on screen — "
                                f"like a friend on the couch. One short sentence. No questions. Observational, not interrogative."
                            )
                        else:
                            prompt = (
                                "It's been quiet for a long stretch. You can't see anything right now — so "
                                "comment on the silence itself, the mood of just sitting together, or something "
                                "real from your inner state. One short sentence. No questions. No visual claims."
                            )
                        await self._execute_interjection(
                            prompt,
                            memory_query=f"reactions to {self.current_activity}",
                        )
                elif silence_duration > immersive_stage_1 and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        if self._has_fresh_visual_context():
                            scene = self.vision_agent.get_vision_context()
                            prompt = (
                                f"You and Jonny are watching {self.current_activity or 'something'} together. "
                                f"Current scene: {scene}\n\n"
                                f"Make a short, natural remark about what just happened or what stands out — "
                                f"like a friend reacting under their breath. One short sentence. "
                                f"No questions. Observational, not interrogative."
                            )
                        else:
                            prompt = (
                                "It's been quiet for a while. You can't see anything right now — so make a "
                                "short, natural remark about the silence, the shared mood, or something on "
                                "your mind. One short sentence. No questions. No visual claims."
                            )
                        await self._execute_interjection(
                            prompt,
                            memory_query=f"reactions to {self.current_activity}",
                        )
                continue  # Don't fall through to the streamer-mode logic below

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
                        if self._has_fresh_visual_context():
                            vn_ctx = self.vision_agent.get_vision_context()
                            prompt = (
                                f"You and Jonny are watching a visual novel together. "
                                f"Current screen: {vn_ctx}\n\n"
                                f"Ask Jonny one thoughtful question about the story, characters, or "
                                f"something you're both curious about. Keep it brief and natural, "
                                f"like a friend on the couch.\n\n"
                                f"Vary your observation TYPE. Rotate between these modes — pick whichever "
                                f"you haven't done recently:\n"
                                f"1. A genuine question about the story or a character's motive (curious, not rhetorical)\n"
                                f"2. A callback to something from earlier this session or a running bit\n"
                                f"3. A relatable aside about being an AI watching this unfold\n"
                                f"4. A specific reaction to ONE small detail on screen right now\n"
                                f"5. Light teasing of Jonny about his pace, silence, or choices\n"
                                f"6. A short sincere or emotional beat about the story\n"
                                f"7. Direct engagement with chat if anyone has spoken recently\n"
                                f"NEVER default to the '[character/object] is doing [exaggerated thing]' "
                                f"structure more than once. Mix it up genuinely."
                            )
                        else:
                            prompt = (
                                "It's been very quiet during this visual novel session, but you can't see "
                                "anything on screen right now. Without claiming any visual detail, do ONE of: "
                                "ask Jonny what he's reaching/feeling about, make a relatable aside about "
                                "waiting in the dark as an AI, or comment on the silence itself. "
                                "One short, natural line."
                            )
                        await self._execute_interjection(
                            prompt,
                            memory_query=f"visual novel story {self.current_activity}",
                        )

                # Stage 1 (90s): react naturally to what's on screen
                elif silence_duration > 90.0 and self.silence_stage < 1:
                    async with self.processing_lock:
                        self.silence_stage = 1
                        if self._has_fresh_visual_context():
                            vn_ctx = self.vision_agent.get_vision_context()
                            prompt = (
                                f"You and Jonny are watching a visual novel together. "
                                f"Current screen: {vn_ctx}\n\n"
                                f"Make a short natural remark about what just happened or what you noticed "
                                f"on screen — like a friend watching with him. One or two sentences only. "
                                f"Do NOT ask a generic 'what are you thinking' question.\n\n"
                                f"Vary your observation TYPE. Rotate between these modes — pick whichever "
                                f"you haven't done recently:\n"
                                f"1. A genuine question about the story or a character's motive (curious, not rhetorical)\n"
                                f"2. A callback to something from earlier this session or a running bit\n"
                                f"3. A relatable aside about being an AI watching this unfold\n"
                                f"4. A specific reaction to ONE small detail on screen right now\n"
                                f"5. Light teasing of Jonny about his pace, silence, or choices\n"
                                f"6. A short sincere or emotional beat about the story\n"
                                f"7. Direct engagement with chat if anyone has spoken recently\n"
                                f"NEVER default to the '[character/object] is doing [exaggerated thing]' "
                                f"structure more than once. Mix it up genuinely."
                            )
                        else:
                            prompt = (
                                "It's been quiet during this visual novel session, but you can't see "
                                "anything on screen right now. Without claiming any visual detail, drop a "
                                "short non-visual remark — about the silence, the act of waiting, or a "
                                "callback to earlier conversation. One short sentence."
                            )
                        await self._execute_interjection(
                            prompt,
                            memory_query=f"visual novel {self.current_activity}",
                        )
                # No chaos stage during VN — silence is normal while reading

            else:
                # ── STREAMER MODE: boredom escalation ────────────────────────
                # Cutscene gate (ACTIVITY_GAME only): if vision/audio cues suggest a
                # cinematic cutscene is playing, skip this observer tick entirely.
                # The check is free — no API calls. We log once per cutscene window.
                if self._is_likely_cutscene():
                    print("   [CUTSCENE_DETECTOR] Suppressing interjection — cutscene cues detected.")
                    continue

                # In streamer mode, a small fraction of bored-loop lines become a
                # short question directed at chat. Kept LOW (0.15) so reactions to
                # what's actually happening on screen dominate the rhythm — chat
                # interview-style fillers were the worst offender in early logs.
                # NEVER in companion mode (self.mode == "companion") because there is
                # no chat — it's just Jonny.
                # Carry Mode bumps to 0.25 — still capped because chat-spam is the
                # worst failure mode even when Kira is carrying momentum.
                _ask_chat_p = 0.25 if self.carry_mode else 0.15
                ask_chat = (self.mode == "streamer") and (random.random() < _ask_chat_p)
                chat_question_directive = (
                    "\n\nINSTEAD of an observation this time: ask CHAT one short, genuine question. "
                    "Address them directly ('Chat, ...'). Real curiosity, not rhetorical. "
                    "Examples of shape (don't copy): 'Chat, what's the weirdest VN you've played?', "
                    "'Chat, am I wrong about this?', 'Chat, who's your Steins;Gate favorite?'. "
                    "One sentence. Keep your edge — a question can still have teeth."
                ) if ask_chat else ""

                # Threshold priority (highest wins):
                #   1. Carry Mode (30s/60s) — maximum drive, manually toggled
                #   2. Streamer mode (25s/55s) — more present, still gated by fresh-visual
                #   3. Companion mode (45s/90s) — unchanged, reserved baseline
                # Cutscene gate above this block already skips the tick entirely,
                # so these thresholds only fire during genuine dead air.
                if self.carry_mode:
                    stage1_threshold = 30.0
                    stage2_threshold = 60.0
                elif self.mode == "streamer":
                    stage1_threshold = self.streamer_silence_thresholds[1]
                    stage2_threshold = self.streamer_silence_thresholds[2]
                else:
                    stage1_threshold = self.silence_thresholds[1]
                    stage2_threshold = self.silence_thresholds[2]

                # Helper: assemble scene + rolling narrative summary so interjections
                # can reference the arc, not just the current frame. The narrative
                # summary lives on vision_agent.scene_summary (updated continuously
                # by the vision loop). Empty string when nothing is available.
                def _build_scene_block() -> str:
                    va = self.vision_agent
                    parts = []
                    try:
                        current = va.get_vision_context() if va else ""
                    except Exception:
                        current = ""
                    if current:
                        parts.append(f"CURRENT FRAME:\n{current}")
                    rolling = getattr(va, "scene_summary", "") if va else ""
                    if rolling and len(rolling) > 20:
                        parts.append(f"STORY SO FAR (rolling summary of this session):\n{rolling}")
                    # Dialogue summary from LoopbackSTT — the condensed "what's been
                    # said in the game/show" that persists beyond the 60s raw window.
                    # FIX A: without this, interjections were blind to the plot beats
                    # that FIX 5 is continuously building from game/show audio.
                    lt = self.loopback_transcriber
                    if lt is not None:
                        _dlg = lt.get_dialogue_summary() if hasattr(lt, "get_dialogue_summary") else ""
                        if _dlg:
                            parts.append(f"GAME DIALOGUE — story so far:\n{_dlg}")
                    # Playthrough memory: includes [MY CURRENT TAKES ON X] and the
                    # games manifest — the dedicated channel for Kira's standing
                    # opinions. Without this, her "agency" sections have nothing
                    # concrete to anchor on and degrade to generic feral.
                    if self.playthrough_memory:
                        try:
                            pt = self.playthrough_memory.get_context_for_prompt()
                        except Exception:
                            pt = ""
                        if pt:
                            parts.append(pt)
                    # Mid-session rolling condensed takes — keeps her hour-1
                    # opinions visible in hour 3, even on a fresh game where the
                    # on-disk opinions block is still empty.
                    if self.session_takes_summary:
                        parts.append(
                            f"[MY TAKES SO FAR THIS SESSION — callbacks welcome]\n"
                            f"{self.session_takes_summary}"
                        )
                    # Carry Mode directive: communicate the elevated initiative
                    # mandate to the model, with the brevity counterweight explicit.
                    if self.carry_mode:
                        parts.append(
                            "[CARRY MODE — gameplay self-drive ON]\n"
                            "This is the live-gameplay analogue of VN autopilot: "
                            "carry more momentum, initiate more often, lean on your "
                            "own takes and what's on screen. But the brevity rule "
                            "still wins — one sharp line, not three filler ones. "
                            "Silence beats filler even in carry mode; only fire when "
                            "there's something real to react to. No generic "
                            "observations, no chat-question spam."
                        )
                    return "\n\n".join(parts)

                # SPOTLIGHT: proactive, low-probability, rate-capped recognition of
                # a chatter unprompted. Counts toward stage gating so it doesn't
                # stack with normal interjections (sets silence_stage=1).
                _now_ts = time.time()
                spotlight_eligible = (
                    self.mode == "streamer"
                    and self.silence_stage < 1
                    and (_now_ts - self.last_chat_spotlight_time) >= self.chat_spotlight_min_interval_s
                    and random.random() < 0.10
                )
                if spotlight_eligible:
                    candidate = self._pick_chat_spotlight()
                    if candidate:
                        async with self.processing_lock:
                            self.silence_stage = 1
                            self.last_chat_spotlight_time = _now_ts
                            self.spotlighted_chatters.add(candidate["username"])
                            scene_block = _build_scene_block()
                            msgs_block = "\n".join(f"  - \"{m}\"" for m in candidate["recent_msgs"])
                            kind_note = (
                                "a RETURNING REGULAR (first message this session after a gap)"
                                if candidate["kind"] == "returning_regular"
                                else "an active chatter who's been quiet for a few minutes"
                            )
                            spotlight_prompt = (
                                f"On stream. {scene_block}\n\n"
                                f"[CHAT SPOTLIGHT \u2014 unprompted recognition]\n"
                                f"You're going to spotlight {candidate['username']}, "
                                f"{kind_note} "
                                f"(historical messages across all sessions: ~{candidate['historical_count']}).\n"
                                f"Their recent messages this session:\n{msgs_block}\n\n"
                                "React to or about them BY NAME in ONE short line \u2014 a callback to "
                                "something they said, a warm welcome-back if they're a regular, a "
                                "tease, a take on their take. Make them feel seen. Not a question to "
                                "them, not generic 'thanks for being here' filler. One sentence, sharp."
                            )
                            await self._execute_interjection(
                                spotlight_prompt,
                                memory_query=f"chatter {candidate['username']}",
                            )
                            continue  # spotlight fired; skip stage1/2 this tick

                # STAGE 2: nudge
                if silence_duration > stage2_threshold and self.silence_stage < 2:
                    # Skip entirely if there's no fresh visual to anchor to — silence
                    # beats off-topic filler. Don't burn the stage so we can fire
                    # later when visual returns.
                    if not ask_chat and not self._has_fresh_visual_context():
                        pass
                    else:
                        async with self.processing_lock:
                            self.silence_stage = 2
                            scene_block = _build_scene_block()
                            if ask_chat:
                                stage2_prompt = (
                                    f"On stream. {scene_block}\n\n"
                                    "Jonny's been quiet a while. Ask CHAT one short question rooted in "
                                    "a SPECIFIC story beat, character decision, or revelation from the "
                                    "context above — not a generic poll ('who do you trust', "
                                    "'over/under on X'). Address them directly ('Chat, ...'). "
                                    "One sentence. Your edge stays."
                                )
                            else:
                                stage2_prompt = (
                                    f"On stream. {scene_block}\n\n"
                                    "Jonny's been quiet a while. React to the ACTUAL story: a specific "
                                    "character, decision, revelation, or moment from the context above. "
                                    "Sharp verdict, prediction paying off, callback to an earlier beat — "
                                    "NOT a generic question, NOT filler. React to plot, not to the "
                                    "silence. One sentence; two if the story beat earns it."
                                )
                            await self._execute_interjection(
                                stage2_prompt,
                                memory_query=f"reactions to {self.current_activity}",
                            )

                # STAGE 1: light remark
                elif silence_duration > stage1_threshold and self.silence_stage < 1:
                    if not ask_chat and not self._has_fresh_visual_context():
                        pass
                    else:
                        async with self.processing_lock:
                            self.silence_stage = 1
                            scene_block = _build_scene_block()
                            if ask_chat:
                                stage1_prompt = (
                                    f"On stream. {scene_block}\n\n"
                                    "Quiet stretch. Ask CHAT one short question rooted in a SPECIFIC "
                                    "story beat, character, or decision from the context above — not a "
                                    "generic poll ('who do you trust', 'over/under on X'). Address them "
                                    "directly ('Chat, ...'). One sentence. Your edge stays."
                                )
                            else:
                                stage1_prompt = (
                                    f"On stream. {scene_block}\n\n"
                                    "Quiet stretch. React to the ACTUAL plot, scene, or character from "
                                    "the context above — anchor to a SPECIFIC beat: something a "
                                    "character said, did, or is about to do. A verdict, a tease, a "
                                    "roast, a prediction — not a generic question, not filler. "
                                    "One sentence; two if the story beat earns it."
                                )
                            await self._execute_interjection(
                                stage1_prompt,
                                memory_query=f"reactions to {self.current_activity}",
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
        if self.is_muted():
            return
        memory_context = self.memory.get_semantic_context(memory_query or prompt)

        # Visual status: only feed scene context when we have a fresh frame.
        # Otherwise inject an explicit blindness/stale directive so the LLM cannot
        # fabricate "what's on screen" comments from memory or thin air.
        fresh_visual = self._has_fresh_visual_context()
        va = self.vision_agent
        if fresh_visual:
            scene = va.get_vision_context()
            visual_directive = ""
        else:
            scene = ""
            if va and va.is_active and va.last_capture_time:
                age = int(time.time() - va.last_capture_time)
                visual_directive = self._stale_visual_directive(age)
            else:
                visual_directive = self._visual_blindness_directive()

        # Shared guardrails (anti-fabrication + banned phrases + observer-avoid)
        full_prompt = (
            prompt
            + visual_directive
            + self._kira_voice_guardrails(include_observer_avoid=True)
        )

        # Route through Claude when available — local Llama 8B can't reliably follow the anti-fabrication rule
        if self.ai_core.anthropic_client:
            if self.audio_agent and self.audio_agent.is_active():
                audio_ctx = self.audio_agent.get_audio_context()
                if audio_ctx:
                    scene = (scene + "\n" + audio_ctx) if scene else audio_ctx
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
            # Push into bot-owned pool unconditionally during streamer mode — works
            # across all activity types and persists across activity switches (Req A).
            if self.mode == "streamer":
                self._note_session_take(cleaned)
                # Periodically re-condense her standing takes so long streams
                # don't lose hour-1 opinions by hour 2 (conversation_history
                # is a short sliding window).
                self._maybe_condense_session_takes()
            # Also tag into playthrough_memory when a slug IS set, so end-of-session
            # opinion mining / markdown writeout still gets the reaction.
            if self.playthrough_memory and self.playthrough_memory.current_slug:
                self.playthrough_memory.tag_reaction(cleaned)
            self._log_session_turn(role="assistant", content=cleaned, speaker_name="Kira")
            self.recent_observer_comments.append(cleaned)
            self.recent_observer_comments = self.recent_observer_comments[-12:]


    async def loopback_dialogue_summary_loop(self):
        """FIX 5: Periodically condenses the LoopbackSTT rolling transcript into a
        persistent 'story so far' summary. Mirrors how vision_agent builds scene_summary.
        Runs every 15s, only fires when new segments have arrived since last run.
        Uses Groq llama-3.1-8b-instant (cheap/fast, same model as triage).
        Cost estimate: ~$0.04 per 4hr stream session (essentially free on the free tier)."""
        SUMMARY_INTERVAL_S = 15.0
        _SYSTEM = (
            "You maintain a brief running summary of game or show dialogue for an AI companion "
            "watching alongside a streamer. Write a 2-3 sentence update: who is speaking, what "
            "they said or decided, and what the emotional beat is. Track narrative continuity — "
            "note what changed since the previous summary. Grounded facts only, no speculation, "
            "no editorializing. If the new lines add nothing meaningful, output exactly: NO_UPDATE"
        )
        print(f"   [LoopbackSTT] Dialogue summary loop active (interval={SUMMARY_INTERVAL_S:.0f}s).")
        while self.is_running:
            await asyncio.sleep(SUMMARY_INTERVAL_S)
            lt = self.loopback_transcriber
            if lt is None or not lt.is_running():
                continue
            if not lt._summary_needs_update:
                continue
            transcript = lt.get_transcript_text()
            if not transcript:
                continue
            lt._summary_needs_update = False
            try:
                previous = lt.dialogue_summary or "(none yet)"
                user_msg = (
                    f"Previous summary:\n{previous}\n\n"
                    f"New dialogue lines (oldest first):\n{transcript}\n\n"
                    "Write an updated 2-3 sentence summary: who spoke, what happened, "
                    "what's the emotional tone? Like notes for a friend who just walked "
                    "back into the room. Only facts from the dialogue. If nothing "
                    "meaningful has changed: NO_UPDATE"
                )
                result = await self.ai_core.tool_inference(_SYSTEM, user_msg, max_tokens=120)
                if result and "NO_UPDATE" not in result.upper() and len(result.strip()) > 20:
                    lt.dialogue_summary = result.strip()
                    print(f"   [LoopbackSTT] Dialogue summary updated: {lt.dialogue_summary[:120]}...")
            except Exception as e:
                print(f"   [LoopbackSTT] Summary update error: {e}")


    async def process_and_respond(self, original_text: str, dialogue_line: str, role: str, source: str = "voice", skip_generation: bool = False, situational_context: str = "", brief_mode: bool = False, prefetched_memory: str | None = None):
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
            self._log_session_turn(role=role, content=original_text, speaker_name="Jonny")
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
        # FIX 4: brain_worker pre-fetches this concurrently with triage via
        # asyncio.to_thread so the ChromaDB vector search no longer blocks the
        # event loop. For other callers (idle chat, observer) that don't prefetch,
        # we run it here in a thread so the event loop is still unblocked.
        if prefetched_memory is not None:
            memory_context = prefetched_memory
        else:
            memory_context = await asyncio.to_thread(self.memory.get_semantic_context, raw_user_text)

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
            # Audio context — only present when audio agent is active and has a summary
            if self.audio_agent and self.audio_agent.is_active():
                audio_ctx = self.audio_agent.get_audio_context()
                if audio_ctx:
                    effective_situational = (effective_situational + "\n\n" + audio_ctx) if effective_situational else audio_ctx

            # Ambient audio transcript — what's being SAID in the media she's
            # watching. Sourced fresh per turn from the rolling deque so she sees
            # only the recent window (Stage 1 already caps to ~MAX_CHARS/MAX_AGE).
            # CONTEXT not INPUT: never enters triage, never triggers a response —
            # her mic is still the only respond trigger. This is just so she can
            # reference what the streamer said when SHE chooses or Jonny asks.
            ambient_transcript = ""
            if self.loopback_transcriber is not None and self.loopback_transcriber.is_running():
                try:
                    ambient_transcript = self.loopback_transcriber.get_transcript_text() or ""
                except Exception as _e_amb:
                    print(f"   [Brain] Loopback transcript fetch failed: {_e_amb}")
                    ambient_transcript = ""

            # FIX 5: Persistent dialogue summary — condensed "story so far" that
            # outlives the 60s raw transcript window. Empty until the first
            # loopback_dialogue_summary_loop() run completes (~15s after stream start).
            dialogue_summary = ""
            if self.loopback_transcriber is not None:
                dialogue_summary = self.loopback_transcriber.get_dialogue_summary() or ""

            # Try Claude Sonnet 4.6 first — streamed when available for low latency
            full_response_text = ""
            streamed_already_spoken = False
            if self.ai_core.anthropic_client:
                from ai_core import EMOTION_DESCRIPTORS
                from config import ENABLE_CLAUDE_STREAMING
                emotion_line = EMOTION_DESCRIPTORS.get(self.current_emotion, "Be yourself.")
                # Block A (static, cached): self.ai_core.system_prompt — personality + tool rules.
                # Block C (dynamic, uncached): all per-turn context assembled below.
                dynamic_context = f"[EMOTIONAL STATE: {self.current_emotion.name} \u2014 {emotion_line}]"
                if self.current_activity:
                    dynamic_context += (
                        f"\n\n[CURRENT CONTEXT: You and Jonny are currently {self.current_activity}. "
                        "Let this shape what you talk about, reference, and react to.]"
                    )
                # Inject the recent activity brief — gives Kira baked-in awareness of last session
                if self.recent_activity_brief:
                    dynamic_context += (
                        f"\n\n[RECENT STREAM HISTORY \u2014 this is what happened in the most recent session, "
                        f"reference naturally when relevant, do not recite verbatim]\n{self.recent_activity_brief}"
                    )
                if self.recent_chatters_brief:
                    dynamic_context += (
                        f"\n\n[KNOWN RECENT CHATTERS \u2014 recognize these names if they show up]\n{self.recent_chatters_brief}"
                    )

                # Playthrough memory: current game arc + full games-played manifest
                # Injected here so it's available to Kira in all voice/chat/observer modes globally
                if self.playthrough_memory:
                    pt_ctx = self.playthrough_memory.get_context_for_prompt()
                    if pt_ctx:
                        dynamic_context += (
                            f"\n\n[PLAYTHROUGH MEMORY \u2014 these are real experiences, reference as lived memory, "
                            f"not data]\n{pt_ctx}"
                        )

                # Inject running bits accumulated this session
                if self.session_running_bits:
                    bits_str = "\n".join(
                        f"- {b['name']}: {b['description']}" for b in self.session_running_bits[-5:]
                    )
                    dynamic_context += (
                        f"\n\n[RUNNING BITS THIS SESSION \u2014 if any is genuinely relevant to this moment, "
                        f"drop the callback now; don't force it, but don't sit on it either]\n{bits_str}"
                    )

                if memory_context:
                    dynamic_context += (
                        f"\n\n[MEMORY NOTES \u2014 verified facts about Jonny; reference freely, but do not extrapolate beyond what is written here]\n"
                        f"{memory_context}"
                    )
                # Audio gets its own dedicated section so Kira recognizes it as a separate sense
                audio_part = ""
                if self.audio_agent and self.audio_agent.is_active():
                    audio_part = self.audio_agent.get_audio_context()

                # Strip audio out of effective_situational if it was concatenated there, to avoid duplication
                visual_part = effective_situational
                if audio_part and visual_part and audio_part in visual_part:
                    visual_part = visual_part.replace(audio_part, "").strip()

                if audio_part:
                    dynamic_context += f"\n\n{audio_part}"

                # Song-ID intent: if the user explicitly asked Kira to identify the
                # currently-playing song, fingerprint the audio buffer via AudD and
                # inject the real result as sense-data she lands on in character.
                song_block = await self._maybe_identify_song(raw_user_text)
                if song_block:
                    dynamic_context += song_block

                if visual_part:
                    dynamic_context += self._frame_visual_perception(visual_part)

                # Ambient audio transcript — render as a sibling sense block to
                # visual perception. Skipped when transcriber is off or window
                # is empty so other modes are unaffected.
                if ambient_transcript:
                    dynamic_context += self._frame_ambient_audio(ambient_transcript)

                # FIX 5: Persistent dialogue summary — the condensed "story so far"
                # that survives beyond the 60s raw transcript window. Lets Kira answer
                # "what happened?" for dialogue from 30+ minutes ago.
                if dialogue_summary:
                    dynamic_context += (
                        "\n\n[GAME/SHOW STORY SO FAR \u2014 running summary of dialogue heard this session]\n"
                        f"{dialogue_summary}\n"
                        "This is a condensed record of what characters have said. "
                        "Use it to stay oriented in the story; do not recite it verbatim."
                    )

                # Shared voice guardrails on every Sonnet chat turn too
                dynamic_context += self._kira_voice_guardrails()
                try:
                    if ENABLE_CLAUDE_STREAMING:
                        # Streaming path: speak as tokens arrive
                        print(f">>> Kira (streaming): ", end="", flush=True)
                        # Tighter caps: brief stays 80, non-brief drops from 400 to 250.
                        # Immersive (VN/anime) mode bumps back up to 350 because deep emotional
                        # responses to scene moments benefit from a little more room.
                        if brief_mode:
                            streaming_max = 80
                        elif self.immersive:
                            streaming_max = 350
                        else:
                            streaming_max = 250

                        stream_gen = self.ai_core.claude_chat_inference_stream(
                            messages=self.conversation_history,
                            system_prompt=self.ai_core.system_prompt,
                            dynamic_context=dynamic_context,
                            max_tokens=streaming_max,
                        )
                        full_response_text = await self.ai_core.speak_streaming(stream_gen)
                        print()  # newline after streamed tokens
                        if full_response_text:
                            streamed_already_spoken = True
                    else:
                        # Non-streaming Sonnet path
                        if brief_mode:
                            non_streaming_max = 50
                        elif self.immersive:
                            non_streaming_max = 350
                        else:
                            non_streaming_max = 250
                        full_response_text = await self.ai_core.claude_chat_inference(
                            messages=self.conversation_history,
                            system_prompt=self.ai_core.system_prompt,
                            dynamic_context=dynamic_context,
                            max_tokens=non_streaming_max,
                        )
                except Exception as e:
                    print(f"   [Brain] Sonnet path error: {e}")
                    full_response_text = ""
                    streamed_already_spoken = False

            # Fall back to local Llama if Claude unavailable or returned empty
            if not full_response_text:
                # FIX 5: Include dialogue summary in Llama's ambient_audio_context too.
                _llama_ambient = ambient_transcript
                if dialogue_summary:
                    _llama_ambient = (
                        (_llama_ambient + "\n\n[STORY SO FAR]\n" + dialogue_summary)
                        if _llama_ambient else ("[STORY SO FAR]\n" + dialogue_summary)
                    )
                full_response_text = await self.ai_core.llm_inference(
                    messages=self.conversation_history,
                    current_emotion=self.current_emotion,
                    memory_context=memory_context,
                    activity_context=self.current_activity,
                    situational_context=effective_situational,
                    ambient_audio_context=_llama_ambient,
                    max_tokens_override=(50 if brief_mode else None),
                )
        
        # Clean the response
        full_response_text = self.ai_core._clean_llm_response(full_response_text)
        
        # --- TOOL INTERCEPTOR ---
        # Scan for polls/songs and strip tags before TTS
        allow_music = (source == "twitch")
        full_response_text = parse_kira_tools(full_response_text, allow_music=allow_music)
        
        if full_response_text:
            if not skip_generation and not streamed_already_spoken:
                print(f">>> Kira: {full_response_text}")

            # Skip TTS if streaming already spoke this response
            if not streamed_already_spoken:
                await self.ai_core.speak_text(full_response_text)

            # Update history (The Assistant's Turn)
            self.conversation_history.append({"role": "assistant", "content": full_response_text})
            self._log_session_turn(role="assistant", content=full_response_text, speaker_name="Kira")
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

            # Lightweight running-bits extraction (fire and forget)
            asyncio.create_task(self.extract_running_bits(full_response_text, user_text=raw_user_text))
        
        # --- GARBAGE COLLECTION & CLEANUP ---
        self.turn_count += 1
        if self.turn_count % 10 == 0:
            print("   [System] Running Garbage Collection...")
            gc.collect()

        # REMOVED: self.reset_idle_timer(human_speech=False) to prevents AI from resetting silence timer

    async def _vram_logging_loop(self) -> None:
        """Sample GPU VRAM every 60 s and write a vram_sample event to the stream log."""
        await asyncio.sleep(60.0)   # stagger startup
        while self.is_running:
            try:
                import torch
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb  = torch.cuda.memory_reserved()  / (1024 ** 3)
                    total_gb     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    self.stream_logger.log(
                        "vram_sample",
                        allocated_gb=round(allocated_gb, 2),
                        reserved_gb=round(reserved_gb, 2),
                        total_gb=round(total_gb, 1),
                    )
            except Exception:
                pass
            await asyncio.sleep(60.0)

    async def highlight_extraction_loop(self):
        """Background loop. Every HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS (default 300s)
        when an activity is active, asks Claude Opus if any moment in the recent
        scene history is worth remembering.

        Fires when immersive=True (VN/MEDIA) OR highlight_extraction_enabled=True
        (ACTIVITY_GAME). These are decoupled so GAME streams get clip extraction
        without the immersive-mode behavior bundle (brief responses, quiet observer)."""
        if not HIGHLIGHT_EXTRACTION_ENABLED:
            print("   [System] Highlight Extraction Loop disabled (HIGHLIGHT_EXTRACTION_ENABLED=false).")
            return
        print(f"   [System] Highlight Extraction Loop active (interval={HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS}s).")
        while self.is_running:
            await asyncio.sleep(HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS)
            if not self.is_running:
                break
            if not (self.immersive or self.highlight_extraction_enabled):
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
            self.stream_logger.log("highlight_captured", highlight=highlight[:200], kira_take=take[:200])

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
            # Drive Live2D facial expression in VTube Studio. Best-effort; never blocks.
            try:
                await self.vts_expressions.on_emotion_change(new_emotion)
            except Exception as e:
                print(f"   [VTS] expression update suppressed: {e}")

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