# vn_autopilot.py — Autonomous VN Mode (Phase 1 + 2)
#
# Phase 1 (mechanical): event-driven screenshot→classify→react→advance loop.
#   - DIALOGUE → advance with selective reactions
#   - TRANSITION → silently advance
#   - CHOICE / SAVE_PROMPT / UNKNOWN → failsafe, hand off to Jonny
#
# Phase 2 (watchability layer): six systems that turn a mechanical loop into
#   compelling solo stream content.
#
#   System 1 — Dynamic Energy/Pacing
#       Reaction rate + energy scale with scene intensity (calm/building/intense/
#       climactic/aftermath). Dense reactions in slow stretches, near-silence during
#       emotional peaks. The Mayuri instinct, systematized.
#
#   System 2 — Solo/Dead-Chat Behavior
#       When chat's been dead for minutes, shift to internal monologue, address
#       absent viewers, carry a one-sided conversation comfortably.
#
#   System 3 — Theory-Building
#       Periodically form predictions during calm stretches. Track them. React
#       when they're confirmed or busted: "I CALLED this" / "I was wrong."
#
#   System 4 — Within-Session Emotional State
#       Track attachment to individual characters (grows with screen time +
#       emotional weight). Investment compounds; heavy moments carry earned weight.
#
#   System 5 — Narrative Weight Detection
#       Text-based heuristic (death/revelation/confession keywords, text length,
#       attachment bonus) produces a weight score that drives Systems 1 and 4.
#       Visual/audio signals are Phase 2.5 additions.
#
#   System 6b — Soft-Pause for Jonny
#       When Jonny speaks during autopilot, pause advancement, have a real
#       conversation, auto-resume once done. No button required.
#
# IMPORTANT: The VN window must be the focused/active window for inputs to register.

import asyncio
import base64
import collections
import difflib
import hashlib
import os
import re
import shutil
import time
import traceback
import xml.sax.saxutils
from io import BytesIO

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        pytesseract.get_tesseract_version()
        PYTESSERACT_AVAILABLE = True
    except Exception:
        PYTESSERACT_AVAILABLE = False
        print("   [Autopilot] Tesseract binary not found — falling back to cloud OCR.")
        print("   [Autopilot]   Install: https://github.com/UB-Mannheim/tesseract/wiki")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("   [Autopilot] pytesseract not installed. Run: pip install pytesseract")

try:
    from config import CLAUDE_CHAT_MODEL
except Exception:
    CLAUDE_CHAT_MODEL = "claude-sonnet-4-6"  # sensible fallback

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pygetwindow as _pgw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False

try:
    import pydirectinput
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    PYDIRECTINPUT_AVAILABLE = False
    print("   [Autopilot] pydirectinput not installed — run: pip install pydirectinput")


# ── Phase 2: Scene intensity states (System 1) ────────────────────────────────
INTENSITY_CALM      = "calm"       # low-stakes connective tissue; riff/theorize freely
INTENSITY_BUILDING  = "building"   # tension rising; lean in, anticipate
INTENSITY_INTENSE   = "intense"    # things are happening; react sparingly and sharply
INTENSITY_CLIMACTIC = "climactic"  # the moment; silence is usually the correct response
INTENSITY_AFTERMATH = "aftermath"  # just after something big; process quietly

# ── Inter-box pacing constants (content-based variable timing) ────────────────
# Tune here to adjust rhythm globally. Replaces the old fixed 0.8s delay.
PAUSE_SENTENCE_END      = 0.20   # text ends with '.'  — clear stop between thoughts
PAUSE_QUESTION_EXCLAIM  = 0.40   # text ends with '?'/'!' — emotional beat / rising tone
PAUSE_CLAUSE_FLOW       = 0.05   # comma or no terminal punct — flow right through
PAUSE_SCENE_CHANGE      = 0.60   # art region just changed — let the new scene land
PAUSE_INTENSITY_BONUS   = 0.30   # added on top for INTENSE / CLIMACTIC moments
PAUSE_AFTERMATH_BONUS   = 0.15   # extra somber pacing in AFTERMATH
PAUSE_REACTION_GAP      = 0.15   # minimal gap before a reaction (same-breath feel)

# ── Reaction grace windows (H2: tiered by intensity) ─────────────────────────
# How long to wait for a queued reaction/theory/aside Claude call to land
# before advancing. Calm moments stay snappy; climactic moments wait for the
# reaction so emotional beats actually get the verbal payoff.
REACTION_GRACE_BY_INTENSITY = {
    INTENSITY_CALM:       0.5,
    INTENSITY_BUILDING:   0.8,
    INTENSITY_INTENSE:    2.5,
    INTENSITY_CLIMACTIC:  5.0,
    INTENSITY_AFTERMATH:  4.0,
}

# ── Silence-breaker (H4) ─────────────────────────────────────────────────────
SILENCE_BREAKER_SECONDS = 22.0     # after this much dead-air, fire a low-energy aside
SILENCE_BREAKER_MIN_GAP = 60.0     # don't fire silence-breakers more than once per minute


# ── Input Controller ───────────────────────────────────────────────────────────

class VNInputController:
    """Sends keyboard/mouse input to the focused VN window via pydirectinput.

    Supports multiple advance methods so the autopilot can verify which one a
    given VN actually accepts. Some engines ignore SendInput keystrokes from
    pydirectinput but accept clicks (or vice-versa); the autopilot cycles
    through methods on each unverified advance until one is observed to change
    the screen, then sticks with the working method.
    """

    # Methods, in the order the autopilot will try them when the configured key
    # has no observed effect. Each is a (label, callable) pair.
    # Order: enter first (most reliable advance in modern VN engines), then
    # click (works when the VN ignores keystrokes), then space (last because
    # many engines — e.g. Saya no Uta, Steins;Gate — use space to toggle the
    # HUD/text-box visibility instead of advancing).
    METHODS_ORDER = ("enter", "click", "space")

    def __init__(self, advance_key: str = "enter", pre_input_delay: float = 0.04):
        self.advance_key = advance_key
        self.pre_input_delay = pre_input_delay

    def advance(self, method: str | None = None):
        """Press the configured advance key, or a specific method if provided.

        method: one of 'space', 'enter', 'click', or None (use self.advance_key).
        Uses keyDown/keyUp with a brief hold so engines that ignore ultra-fast
        synthetic taps (Saya no Uta, some Ren'Py builds) still register the
        input. Click uses mouseDown/mouseUp with the same hold pattern.
        Raises RuntimeError if pydirectinput unavailable.
        """
        if not PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError("pydirectinput not installed. Run: pip install pydirectinput")
        m = (method or self.advance_key).lower()
        time.sleep(self.pre_input_delay)
        if m == "click":
            try:
                pydirectinput.mouseDown()
                time.sleep(0.05)
                pydirectinput.mouseUp()
            except Exception:
                pydirectinput.click()
        elif m in ("enter", "space") or len(m) >= 1:
            key = m if m in ("enter", "space") else m
            try:
                pydirectinput.keyDown(key)
                time.sleep(0.05)
                pydirectinput.keyUp(key)
            except Exception:
                pydirectinput.press(key)

    def set_advance_key(self, key: str):
        self.advance_key = key

    def click_at(self, x: int, y: int):
        """Move mouse to absolute screen coords and click. Used by CHOICE auto-pick."""
        if not PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError("pydirectinput not installed.")
        time.sleep(self.pre_input_delay)
        try:
            pydirectinput.moveTo(int(x), int(y))
            time.sleep(0.05)
            pydirectinput.click()
        except Exception as e:
            print(f"   [Autopilot] click_at({x},{y}) failed: {e}")
            raise

    def press_escape(self):
        if not PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError("pydirectinput not installed.")
        time.sleep(self.pre_input_delay)
        pydirectinput.press("escape")


# ── Main Autopilot ─────────────────────────────────────────────────────────────

class VNAutopilot:
    """
    Autonomous Visual Novel reader.
    Classifies each screen, handles DIALOGUE silently or with selective reactions,
    and triggers a failsafe (stopping + alerting Jonny) for any non-dialogue screen.
    """

    SCREEN_CLASSIFIER_PROMPT = (
        "Look at this visual novel screen. Classify it into exactly ONE category "
        "and respond with ONLY the category word:\n\n"
        "DIALOGUE - normal story text/dialogue that advances by pressing a key. "
        "Use this WHENEVER any readable story/dialogue text is visible on screen. "
        "This includes: busy backgrounds with text, character close-ups with text, "
        "CG/event scenes with text, visual effects with text overlay, minimalist or "
        "blank backgrounds with text, narration boxes — ANY frame containing readable "
        "story text is DIALOGUE. Bias heavily toward DIALOGUE; if you can read a "
        "dialogue/narration line, the answer is DIALOGUE.\n"
        "CHOICE - a decision menu with multiple selectable options the player must "
        "pick from (e.g. multiple buttons or labelled choices stacked vertically).\n"
        "SAVE_PROMPT - a save/load menu, settings menu, or system dialog box.\n"
        "TRANSITION - title card, chapter break, scene fade, black/white screen, "
        "or pure CG with NO readable story text at all.\n"
        "UNKNOWN - ONLY use this when you genuinely cannot determine what is on "
        "screen AND there is no readable text AND it is not clearly a transition "
        "or menu. NEVER use UNKNOWN if any story text is visible — that is "
        "DIALOGUE. UNKNOWN should be rare; prefer DIALOGUE or TRANSITION.\n\n"
        "Respond with ONLY the single category word, nothing else."
    )

    # In-character handoff lines. NEVER name the operator, the dashboard, the
    # "screen", buttons, or anything that breaks the illusion that Kira is the
    # one playing. Console logs (which name Jonny etc.) stay where they are —
    # those are operator-only and never reach TTS.
    FAILSAFE_LINES = {
        "CHOICE": (
            "Hmm — okay, a choice. I'm gonna sit with this one for a sec "
            "before I commit."
        ),
        "SAVE_PROMPT": (
            "Oh — a little menu popped up. Give me a moment to deal with it."
        ),
        "UNKNOWN": (
            "Okay, hold on — I'm not totally sure what's happening in this "
            "scene. Let me just look at it for a moment."
        ),
        "INPUT_ERROR": (
            "Hmm, something's not quite clicking on my end. One sec."
        ),
        "VN_WINDOW_NOT_FOUND": (
            "Wait — the game just slipped away from me. Hold on, I'll be right back."
        ),
        "STUCK": (
            "Hmm. This scene's not budging for me. Let me poke at it for a sec."
        ),
        "ERROR": (
            "Ugh, something hiccupped on my side. Give me a moment."
        ),
    }

    def __init__(self, ai_core, vision_client=None, bot=None):
        """
        ai_core:        AI_Core instance (for Anthropic/Claude access)
        vision_client:  AsyncOpenAI instance (for screen classification + transcription)
        bot:            Optional reference to the parent KiraBot — used to look up
                        semantic memory, playthrough memory, and shared voice guardrails
                        so in-character VN reactions sound like Kira instead of a
                        memory-blind chatbot.
        """
        self.ai_core = ai_core
        self.vision_client = vision_client
        self.bot = bot
        self.input_controller = VNInputController()

        # ── Public state (polled by bot / dashboard) ───────────────────────────
        self.enabled: bool = False          # master toggle set by dashboard
        self.is_running: bool = False       # True while the loop is actively advancing
        self.is_paused: bool = False        # True when stopped waiting for Jonny
        self.pause_reason: str = ""         # "CHOICE", "SAVE_PROMPT", "UNKNOWN", "ERROR", etc.

        # ── Pacing config (adjustable from dashboard) ──────────────────────────
        self.pacing_base: float = 2.5
        self.pacing_per_char: float = 0.025
        self.pacing_max: float = 8.0

        # ── Narrative memory ────────────────────────────────────────────────────
        self.vn_narrative_summary: str = ""
        self.vn_boxes_since_summary: int = 0
        self.vn_recent_text_buffer: list[str] = []

        # ── Phase 2: Dynamic energy / pacing (System 1) ────────────────────────
        self.scene_intensity: str = INTENSITY_CALM
        self._aftermath_countdown: int = 0   # boxes remaining in aftermath state
        self._boxes_since_reaction: int = 0  # for spacing solo asides

        # ── Phase 2: Solo / dead-chat behavior (System 2) ──────────────────────
        self._last_chat_time: float = time.time()
        self._boxes_since_solo_aside: int = 0

        # ── Phase 2: Theory-building (System 3) ────────────────────────────────
        # Each theory: {theory, formed_box, status, resolved_box (optional)}
        self.active_theories: list[dict] = []
        self._boxes_since_theory_check: int = 0
        self.total_boxes_read: int = 0

        # ── Phase 2: Within-session emotional state (System 4) ─────────────────
        self.character_attachment: dict[str, float] = {}   # char_name -> 0.0–1.0
        self.story_investment: float = 0.0                 # 0.0–1.0, ramps over session
        self.emotional_trajectory: str = ""
        self._char_mention_counts: dict[str, int] = {}     # raw mention counts

        # ── Phase 2: Soft-pause for Jonny (System 6b) ──────────────────────────
        self.soft_paused: bool = False

        # ── Callbacks (wired by bot.py) ─────────────────────────────────────────
        # on_speak(text: str) -> awaitable  — speak a reaction via TTS
        self.on_speak = None
        # on_speak_vn(text, ssml_inner) -> awaitable  — VN-specific TTS with prosody hints
        self.on_speak_vn = None
        # on_failsafe(screen_type: str) -> None  — notify dashboard of failsafe
        self.on_failsafe = None

        # ── Internal ────────────────────────────────────────────────────────────
        self._task: asyncio.Task | None = None
        self._last_box_text: str = ""      # dedup: skip reaction on re-capture of same box
        self._prepared_next: dict | None = None  # pre-prepared next box data from pipeline

        # ── Window targeting ─────────────────────────────────────────────────────
        # Set this to a substring of the VN window title (e.g. "Narcissu", "planetarian").
        # When set, screenshots are cropped to that window's bounding rect and
        # advance keystrokes are sent only after bringing that window to the foreground.
        self.vn_window_title: str = ""

        # ── Scene art awareness (cloud vision, fires only on background change) ──
        self._last_art_hash: str = ""           # hash of the art region of the last frame
        self._scene_art_description: str = ""  # cached scene description from cloud vision

        # ── Text-region config (for stabilization hash — NOT a crop for reading) ──
        # Cloud reads always use the full window. This fraction controls which part
        # of the frame is hashed to detect typewriter completion / screen settle.
        # Animated backgrounds in the upper portion don't affect stabilization.
        self.text_region_top: float = 0.60     # lower 40% contains the dialogue box

        # ── Cloud read metrics (for heartbeat cost log) ────────────────────────
        self._cloud_read_count: int = 0        # total cloud vision read calls this session
        self._loop_start_time: float = 0.0     # set when _loop() enters
        self._last_heartbeat: float = 0.0      # epoch of last heartbeat print
        self._last_progress_time: float = 0.0  # epoch of last successful box read

        # ── Window-loss recovery ───────────────────────────────────────────────
        self._warned_window_lost: bool = False  # suppress repeated "waiting" prints

        # ── Stuck watchdog (distinct from crash supervisor) ───────────────────
        # Fires when no new text has been read/spoken for _stuck_watchdog_threshold
        # seconds during active (non-soft-paused, non-failsafed) play.
        self._stuck_watchdog_threshold: float = 90.0  # seconds
        self._stuck_warned: bool = False  # True once the STUCK log has been printed

        # ── Hard-failsafe tracking ────────────────────────────────────────────
        # True while a hard failsafe (CHOICE / SAVE_PROMPT / UNKNOWN / STUCK /
        # ERROR) is active. Soft-pause release must NOT clear a hard failsafe.
        self._hard_failsafed: bool = False

        # ── Advance verification + method discovery ───────────────────────────
        # The autopilot captures a hash before & after each advance to verify
        # the screen actually changed. If the configured method has no effect,
        # it cycles through VNInputController.METHODS_ORDER (re-focusing before
        # each) until one works, then remembers the working method.
        self._working_advance_method: str | None = None  # discovered live; None = use default

        # Oscillation detection: track the last few accepted post-advance hashes.
        # If a new advance lands on a hash we just saw, the input is toggling
        # something (HUD/menu) rather than advancing the story. This is the
        # Saya-no-Uta / Steins;Gate "space toggles text-box" case.
        self._recent_advance_hashes: collections.deque = collections.deque(maxlen=5)
        # Track consecutive non-progress reads (same/prefix text) so we can
        # unlock the working method when it stops actually working.
        self._consecutive_non_progress_reads: int = 0
        # Cloud-call timeout in seconds (every vision/LLM call must complete
        # within this budget or be cancelled — hanging calls were causing
        # 30-60s prep stalls). 3.5s is the sweet spot: longer than typical
        # gpt-4o-mini latency (1-2.5s) but short enough that a degraded call
        # fails fast instead of hogging an OCR slot for 6+s.
        self._cloud_call_timeout: float = 3.5
        self._advance_attempts: int = 0                 # total advance calls this session
        self._advance_diag_cycles: int = 5              # log full diagnostic for first N cycles
        # Frame hash captured immediately after a verified advance — handed to
        # _post_advance_prep() so we don't waste a capture re-grabbing the same
        # confirmed-new frame.
        self._last_verified_post_advance_frame = None

        # ── UNKNOWN-screen tolerance ──────────────────────────────────────────
        # Busy VNs (Steins;Gate especially) produce many UNKNOWN frames: CG art,
        # effects, close-ups between dialogue. We advance through them like
        # TRANSITIONs and only failsafe after a long unbroken streak where no
        # readable dialogue ever surfaces — actual "stuck" is caught by
        # _safe_advance's screen-change verification, not by UNKNOWN itself.
        self._consecutive_unknown: int = 0
        self._max_unknown_streak: int = 10
        # Inner-loop retries for an UNKNOWN classification before we treat it
        # as "really UNKNOWN" and advance through. Each retry waits then
        # re-classifies the same captured-fresh frame — handles CG/animation
        # frames that resolve to DIALOGUE once the engine settles.
        self._unknown_retry_count: int = 3

        # ── CHOICE auto-pick (C1) ────────────────────────────────────────────
        # Default ON so unattended streams aren't blocked by every menu choice.
        # For route-branching VNs (Clannad etc.), set target_route to a string
        # like "Nagisa route — aim for After Story" and the picker will prefer
        # the option that best serves that goal.
        self.auto_choose: bool = True
        self.target_route: str = ""
        # If True, even with auto_choose on, MAJOR choices (high-confidence
        # route-defining) still pause for Jonny. Default False (fully autonomous).
        self.pause_on_major_choices: bool = False

        # ── SAVE_PROMPT soft-dismiss (C2) ────────────────────────────────────
        # Default ON: press Escape to dismiss menus rather than failsafe.
        self.auto_dismiss_menus: bool = True

        # ── Speaker attribution (H1) ─────────────────────────────────────────
        self._last_speaker: str = ""           # last speaker label, for run-detection
        # Cache of per-speaker prosody offsets so the same character sounds
        # consistent across the session. Deterministic hash → (pitch, rate) tuple.
        self._speaker_voice_cache: dict[str, tuple[str, str]] = {}

        # ── Silence-breaker tracking (H4) ────────────────────────────────────
        self._last_spoke_time: float = time.time()
        self._last_silence_breaker_time: float = 0.0
        self._silence_breaker_in_flight: bool = False
        # Track when we last saw a DIALOGUE screen — silence-breaker only fires
        # after sustained non-dialogue time, so it can't interrupt active reads.
        self._last_dialogue_time: float = time.time()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @staticmethod
    def list_open_windows() -> list[str]:
        """Return titles of all currently-visible windows (non-empty titles only)."""
        if not PYGETWINDOW_AVAILABLE:
            return ["(pygetwindow not available)"]
        try:
            return sorted(
                {w.title for w in _pgw.getAllWindows() if (w.title or "").strip()},
                key=str.lower,
            )
        except Exception as e:
            return [f"(error listing windows: {e})"]

    def start(self):
        """Start or restart the autopilot loop. Safe to call multiple times.
        Schedules _async_start() which validates / auto-detects the VN window
        before entering the main loop — full-screen capture is never used.
        """
        if not self.enabled:
            return
        if self._task and not self._task.done():
            return  # already running

        self.is_running = True
        self.is_paused = False
        self.pause_reason = ""
        self._task = asyncio.ensure_future(self._async_start())
        print("   [Autopilot] Started.")

    async def _async_start(self):
        """Validate or auto-detect the VN window, then enter the main loop.
        If a title is already set: verify the window is findable.
        If no title: run a cheap LLM auto-detection call.
        Never falls back to full-screen capture.
        """
        if self.vn_window_title:
            # Manual override — verify the window is reachable
            print(f"   [Autopilot] Looking for VN window: '{self.vn_window_title}'")
            win = self._find_vn_window()
            if win is not None:
                bbox = (win.left, win.top,
                        win.left + win.width, win.top + win.height)
                print(f"   [Autopilot] VN window FOUND: '{win.title}' at {bbox}")
            else:
                titles = self.list_open_windows()
                print(f"   [Autopilot] VN window NOT FOUND for '{self.vn_window_title}'.")
                print("   [Autopilot] Open windows (check for the correct title):")
                for t in titles:
                    print(f"      \u2022 {t}")
                await self._trigger_failsafe("VN_WINDOW_NOT_FOUND")
                return
        else:
            # Auto-detect via LLM
            print("   [Autopilot] No window title set \u2014 attempting auto-detection...")
            detected = await self._autodetect_vn_window()
            if detected:
                self.vn_window_title = detected
            else:
                print("   [Autopilot] Could not identify a VN window \u2014 pausing.")
                print("   [Autopilot] Set a title in the dashboard or press Re-detect, then Resume.")
                self.is_running = False
                self.is_paused = True
                self.pause_reason = "VN_WINDOW_NOT_FOUND"
                if self.on_failsafe:
                    self.on_failsafe("VN_WINDOW_NOT_FOUND")
                return

        await self._loop()

    async def _autodetect_vn_window(self) -> "str | None":
        """Identify the active VN/game window via a cheap LLM call.
        Filters out known system / tool windows first, then sends the remaining
        candidates to Claude and returns the matched title string, or None.
        """
        _IGNORE_SUBSTRINGS = frozenset([
            "kira - control center",
            "nvidia geforce",
            "visual studio code",
            " - youtube",
            "youtube -",
            "google chrome",
            "firefox",
            "microsoft edge",
        ])
        _IGNORE_PREFIXES = ("obs ",)    # OBS Studio shows as "OBS 32.0.4 \u2014 ..."
        _IGNORE_EXACT = frozenset([
            "steam", "program manager", "task manager",
            "settings", "windows input experience", "vtube studio",
        ])

        all_titles = self.list_open_windows()
        print(f"   [Autopilot] All windows: {all_titles}")
        candidates = [
            t for t in all_titles
            if not any(ig in t.lower() for ig in _IGNORE_SUBSTRINGS)
            and not any(t.lower().startswith(p) for p in _IGNORE_PREFIXES)
            and t.lower().strip() not in _IGNORE_EXACT
        ]
        print(f"   [Autopilot] After filter (candidates): {candidates}")

        if not candidates:
            print("   [Autopilot] No candidate windows after filtering.")
            return None

        if not getattr(self.ai_core, 'anthropic_client', None):
            # No LLM: use the single candidate if unambiguous
            if len(candidates) == 1:
                print(f"   [Autopilot] Single candidate (no LLM): using '{candidates[0]}'")
                return candidates[0]
            print("   [Autopilot] Anthropic unavailable and multiple candidates \u2014 cannot auto-detect.")
            return None

        candidate_str = "\n".join(f"  - {t}" for t in candidates)
        prompt = (
            "The user is starting an autonomous visual novel / story-game play session.\n"
            "From this list of open window titles, identify which ONE is the visual novel "
            "or story game they want to play.\n"
            "Prefer the running game over a launcher or store window "
            "(e.g. prefer 'Game ver1.00S' over 'Game for Steam Launcher').\n"
            "Respond with ONLY the exact window title string from the list, nothing else.\n"
            "If none look like a VN or game, respond: NONE\n\n"
            f"Open windows:\n{candidate_str}"
        )

        try:
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text.strip().strip('"').strip("'")
            if not result or result.upper() == "NONE":
                return None
            # Exact match first
            for t in all_titles:
                if t.lower() == result.lower():
                    print(f"   [Autopilot] Auto-detected VN window: '{t}'")
                    return t
            # Partial match (LLM may truncate the title)
            rl = result.lower()
            for t in all_titles:
                if rl in t.lower() or t.lower() in rl:
                    print(f"   [Autopilot] Auto-detected VN window (partial match): '{t}'")
                    return t
            print(f"   [Autopilot] LLM returned '{result}' but no window matched.")
            return None
        except Exception as e:
            print(f"   [Autopilot] Auto-detect error: {e}")
            return None

    def stop(self):
        """Stop the autopilot cleanly. Does NOT speak anything."""
        self.enabled = False
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        print("   [Autopilot] Stopped.")

    def resume_after_failsafe(self):
        """Resume after Jonny has handled a non-dialogue screen manually.

        Clears read-tracking state so post-choice/post-menu content is read fresh:
        _last_box_text is reset so she never re-reads the last pre-choice line,
        and _prepared_next is discarded since that pipeline data predates the choice.
        Always routes through _async_start() so window detection is re-validated
        before entering the loop.
        """
        if not self.enabled:
            return
        # Clear read-tracking so post-choice content is read completely fresh
        self._last_box_text = ""
        self._prepared_next = None
        self._stuck_warned = False
        self._hard_failsafed = False
        self._consecutive_unknown = 0
        self.soft_paused = False
        self.is_paused = False
        self.pause_reason = ""
        self.is_running = True
        if not self._task or self._task.done():
            self._task = asyncio.ensure_future(self._async_start())
        print("   [Autopilot] Resumed — read state cleared for fresh post-choice/menu read.")

    # ── Phase 2: Soft-pause / chat notification (Systems 2, 6b) ───────────────

    def soft_pause(self):
        """Gentle pause while Jonny talks — does NOT require Resume button.
        The loop stays alive; advancement is just suspended.
        A hard failsafe (CHOICE / STUCK / ERROR etc.) takes precedence —
        soft-pause is subordinate and cleared on resume_after_failsafe()."""
        if not self.soft_paused:
            self.soft_paused = True
            print("   [Autopilot] Soft-paused (Jonny is talking).")

    def soft_resume(self):
        """Release soft-pause after Jonny's conversation ends.
        Hard failsafe takes priority: if is_paused is True (hard failsafe active),
        soft_resume releases only the soft-pause flag and logs the conflict clearly —
        it does NOT resume play. Only resume_after_failsafe() can clear a hard failsafe."""
        if self.soft_paused:
            self.soft_paused = False
            if self.is_paused:
                # Hard failsafe is still active — don't pretend we're resuming
                print(
                    f"   [Autopilot] Soft-pause released — but hard failsafe is active "
                    f"({self.pause_reason}). Press Resume when ready."
                )
            else:
                print("   [Autopilot] Soft-pause released — resuming.")

    def notify_chat_activity(self):
        """Call whenever a chat message is received. Resets the dead-chat clock
        that feeds System 2 (solo/dead-chat behavior)."""
        self._last_chat_time = time.time()

    @property
    def chat_dead_min(self) -> float:
        """Minutes since the last chat message. Feeds System 2 solo-mode weighting."""
        return (time.time() - self._last_chat_time) / 60.0

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def _loop(self):
        """Event-driven main loop: screenshot → classify → handle → advance → repeat.

        UNKILLABLE: every error is caught, logged, and recovered — never exits on
        exceptions. Only explicit toggling-off or a handled failsafe (CHOICE /
        SAVE_PROMPT) returns from this method. Missing-library errors (PIL /
        pygetwindow) are the sole fatal exception since they cannot be recovered
        at runtime.

        When a pipeline task pre-prepared the next box during TTS, that data is
        consumed here instead of taking a fresh screenshot + classify + cloud read.
        """
        self._loop_start_time = time.time()
        self._last_heartbeat = time.time()
        self._last_progress_time = time.time()
        # Reset silence-breaker clock so it can't fire on the very first iteration
        # (it would have been counting since __init__, which may be much earlier).
        self._last_spoke_time = time.time()
        self._last_silence_breaker_time = time.time()
        self._last_dialogue_time = time.time()
        self._warned_window_lost = False
        print("   [Autopilot] Loop running.")

        while self.enabled and not self.is_paused:
            # System 6b: soft-pause while Jonny is talking — hold without exiting
            while self.soft_paused and self.enabled and not self.is_paused:
                await asyncio.sleep(0.25)
            if not self.enabled or self.is_paused:
                break

            # ── Heartbeat (every 5 minutes) ──────────────────────────────────
            now = time.time()
            if now - self._last_heartbeat >= 300:
                elapsed = now - self._loop_start_time
                h, rem = divmod(int(elapsed), 3600)
                m = rem // 60
                cost_est = self._cloud_read_count * 0.000075  # gpt-4o-mini vision ≈ $0.075/1k calls
                print(
                    f"   [Autopilot] Alive — {self.total_boxes_read} boxes, "
                    f"~${cost_est:.3f} cloud spend, running {h}h {m}m"
                )
                # Disk-space warning in heartbeat so it surfaces even when no
                # narrative summary fires (i.e. story moves slowly this stretch).
                self._check_disk_space()
                self._last_heartbeat = now

            # ── Stuck watchdog (distinct from crash supervisor) ──────────────
            # Fires when no NEW text has been read/spoken for _stuck_watchdog_threshold
            # seconds during active play (not soft-paused, not already hard-failsafed).
            # Recovery sequence: re-focus → advance → re-capture. If still stuck →
            # failsafe to Jonny with a spoken line and resume button.
            if (not self.soft_paused and not self.is_paused
                    and (now - self._last_progress_time) > self._stuck_watchdog_threshold):
                secs_stuck = now - self._last_progress_time
                if not self._stuck_warned:
                    print(
                        f"   [Autopilot] STUCK \u2014 no new content in {secs_stuck:.0f}s. "
                        "Attempting recovery."
                    )
                    self._stuck_warned = True
                recovered = await self._stuck_recovery()
                if recovered:
                    self._last_progress_time = time.time()
                    self._stuck_warned = False
                else:
                    await self._trigger_failsafe("STUCK")
                    return
                continue

            # ── Silence-breaker (H4) ────────────────────────────────────────
            # If we've been silent too long during active play (long CG run,
            # transition montage, slow stretch), fire a low-energy aside so the
            # stream doesn't sit dead. Throttled internally.
            await self._maybe_speak_silence_breaker()

            try:
                # Check for pre-prepared data from the pipeline task
                prepared = None
                if self._prepared_next:
                    prepared            = self._prepared_next
                    self._prepared_next = None
                    screen_type         = prepared["screen_type"]
                    frame               = prepared["frame"]
                else:
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None, self._grab_frame_sync
                    )

                    # Window lost → enter recovery wait, then retry the iteration
                    if frame is None:
                        recovered = await self._wait_for_window_recovery()
                        if not recovered:
                            return  # failsafe was triggered inside recovery
                        continue

                    screen_type = await self._classify_screen(frame)

                print(f"   [Autopilot] Screen: {screen_type}")

                if screen_type == "DIALOGUE":
                    self._last_dialogue_time = time.time()
                    await self._handle_dialogue(frame, prepared)
                    # _last_progress_time is updated inside _handle_dialogue()
                    # only when genuinely new text is confirmed+spoken.
                    self._consecutive_unknown = 0

                elif screen_type == "TRANSITION":
                    await asyncio.sleep(0.3)
                    self._safe_advance()
                    await asyncio.sleep(0.5)
                    self._last_progress_time = time.time()
                    self._consecutive_unknown = 0

                elif screen_type == "UNKNOWN":
                    # UNKNOWN is NOT a stall condition on busy VNs — it's almost
                    # always a CG/transition/animation frame between dialogue
                    # lines. Try to resolve to DIALOGUE via retry + cloud read;
                    # otherwise advance through it. Failsafe only after a long
                    # unbroken streak of unresolvable UNKNOWNs.
                    handled = await self._handle_unknown(frame)
                    if not handled:
                        print(
                            f"   [Autopilot] UNKNOWN streak exceeded "
                            f"({self._consecutive_unknown}) — handing off to Jonny."
                        )
                        await self._trigger_failsafe("UNKNOWN")
                        return
                    # _safe_advance() inside _handle_unknown verifies screen
                    # changes, so genuine "frozen" stalls already failsafe via
                    # _safe_advance's own STUCK trigger.

                else:
                    # CHOICE or SAVE_PROMPT — try to handle autonomously first;
                    # _handle_choice / _handle_save_prompt will failsafe if they can't.
                    if screen_type == "CHOICE":
                        await self._handle_choice(frame)
                        if self.is_paused:  # failsafe fired inside handler
                            return
                        self._consecutive_unknown = 0
                        continue
                    if screen_type == "SAVE_PROMPT":
                        await self._handle_save_prompt(frame)
                        if self.is_paused:
                            return
                        self._consecutive_unknown = 0
                        continue
                    # Unknown extra screen type (shouldn't happen) — failsafe
                    await self._trigger_failsafe(screen_type)
                    return

            except asyncio.CancelledError:
                raise

            except RuntimeError as e:
                # Missing-library errors cannot be recovered — stop and notify
                msg = str(e)
                if "Pillow not available" in msg or "pygetwindow not available" in msg:
                    print(f"   [Autopilot] Fatal config error (cannot recover): {e}")
                    self.is_running = False
                    self.is_paused  = True
                    self.pause_reason = "ERROR"
                    if self.on_failsafe:
                        self.on_failsafe("ERROR")
                    return
                # All other RuntimeErrors: log, back off, continue
                print(f"   [Autopilot] RuntimeError (recovering in 2s): {e}")
                await asyncio.sleep(2.0)

            except Exception as e:
                # UNKILLABLE: any other exception → log + back off + continue
                print(f"   [Autopilot] Error (recovering in 2s): {e}")
                traceback.print_exc()
                await asyncio.sleep(2.0)

        self.is_running = False
        print("   [Autopilot] Loop exited.")

    # ── Screen classification ──────────────────────────────────────────────────

    async def _classify_screen(self, frame) -> str:
        """Returns one of: DIALOGUE / CHOICE / SAVE_PROMPT / TRANSITION / UNKNOWN."""
        if not self.vision_client:
            return "UNKNOWN"
        valid = {"DIALOGUE", "CHOICE", "SAVE_PROMPT", "TRANSITION", "UNKNOWN"}
        for attempt in range(3):
            try:
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode()
                resp = await asyncio.wait_for(
                    self.vision_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.SCREEN_CLASSIFIER_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            }},
                        ],
                    }],
                    max_tokens=10,
                    temperature=0.0,
                ),
                    timeout=self._cloud_call_timeout,
                )
                raw = resp.choices[0].message.content.strip().upper()
                return raw if raw in valid else "UNKNOWN"
            except asyncio.TimeoutError:
                # Treat as transition so caller advances through rather than
                # retrying 3x (which would cost up to 3 * 3.5s = 10.5s).
                print(
                    f"   [Autopilot] Classify: TIMEOUT after "
                    f"{self._cloud_call_timeout:.1f}s — treating as TRANSITION (advance through)."
                )
                return "TRANSITION"
            except Exception as e:
                err_low = str(e).lower()
                if any(s in err_low for s in ("rate limit", "429", "529", "overloaded", "too many requests")):
                    delay = 2.0 * (2 ** attempt)    # 2s → 4s → 8s (keep reading cadence tight)
                    print(
                        f"   [Autopilot] Classify: rate-limited "
                        f"(attempt {attempt + 1}/3) — backing off {delay:.0f}s."
                    )
                    await asyncio.sleep(delay)
                    continue   # retry
                # Non-rate-limit error — log and fall through to UNKNOWN
                print(f"   [Autopilot] Classify error: {e}")
                return "UNKNOWN"
        print("   [Autopilot] Classify: all retries exhausted — returning UNKNOWN.")
        return "UNKNOWN"

    # ── Dialogue handling ──────────────────────────────────────────────────────

    @staticmethod
    def _quick_hash(frame) -> str:
        """Cheap image hash for typewriter stability detection.
        Resizes to a small thumbnail and MD5s the PNG bytes — no API call."""
        thumb = frame.resize((64, 48))
        buf = BytesIO()
        thumb.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()

    @staticmethod
    def _art_hash(frame) -> str:
        """Hash the upper ~55% of the screen (art/scene region) for scene-change detection.
        Sensitive to background swaps; ignores text-box changes in the lower portion."""
        w, h = frame.size
        art = frame.crop((0, 0, w, int(h * 0.55)))
        thumb = art.resize((80, 44))
        buf = BytesIO()
        thumb.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()

    @staticmethod
    def _text_region_hash(frame, top_frac: float = 0.60) -> str:
        """Hash the dialogue text region (default lower 40%) for typewriter stabilization.
        Using the text region instead of the full frame means animated backgrounds
        (e.g. Steins;Gate, Clannad) don't prevent stabilization — only the text
        settling triggers 'stable', which is what actually matters before reading."""
        w, h = frame.size
        region = frame.crop((0, int(h * top_frac), w, h))
        thumb = region.resize((64, 24))
        buf = BytesIO()
        thumb.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()

    @staticmethod
    def _text_region_likely_blank(frame, top_frac: float = 0.60) -> bool:
        """Cheap local heuristic: is the dialogue text region effectively blank?
        Solid-color fade/transition frames have very low pixel variance; rendered
        dialogue text pushes variance much higher. Used to short-circuit cloud OCR
        on transitional frames the classifier mistakenly tagged as DIALOGUE
        (saves ~1-5s of cloud latency per blank screen). Fails open on error so
        we never falsely skip a real read."""
        try:
            from PIL import ImageStat
            w, h = frame.size
            region = frame.crop((0, int(h * top_frac), w, h)).convert("L")
            region = region.resize((128, 48))
            stddev = ImageStat.Stat(region).stddev[0]
            # Empirical: rendered dialogue boxes -> stddev > 25; solid fades < 12.
            return stddev < 12.0
        except Exception:
            return False

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip markdown/OCR artifacts before TTS: code fences, backticks, stray pipes."""
        # Strip ``` code fences (possibly with language tag)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'```', '', text)
        # Strip inline backticks
        text = text.replace('`', '')
        # Strip stray pipe characters (OCR table artefacts)
        text = text.replace(' | ', ' ')
        return ' '.join(text.split())

    @staticmethod
    def _fuzzy_accumulation_split(last: str, current: str) -> str | None:
        """If `current` appears to be an accumulated continuation of `last`
        (even with minor OCR drift in the shared prefix), return the new-suffix
        portion. Otherwise return None.

        Strategy: find longest matching block anchored at the start of both
        strings. If it covers >=70% of `last` and starts within the first few
        characters of both, peel that many chars off `current` and return the
        tail. Tolerates 1-2 character OCR slips like "Tanbo" vs "Tarbo"."""
        if not last or not current or len(current) <= len(last):
            return None
        # Quick exact-prefix path
        if current.startswith(last):
            return current[len(last):]
        # Fuzzy: longest common substring near start
        sm = difflib.SequenceMatcher(None, last, current, autojunk=False)
        match = sm.find_longest_match(0, len(last), 0, min(len(current), len(last) + 8))
        if match.size < max(20, int(len(last) * 0.70)):
            return None
        # Must anchor near the start of both strings (allow a few-char drift)
        if match.a > 4 or match.b > 4:
            return None
        # Peel off through end of the matched region in `current`
        return current[match.b + match.size:]

    @staticmethod
    def _ocr_text_region(frame) -> str:
        """Extract dialogue text from the lower ~40% of the screen via local Tesseract OCR.
        VN dialogue boxes typically occupy the bottom 35-45% of the screen.
        Returns empty string if Tesseract is unavailable or the result is too short.

        TIP: Set the VN's own text-speed to Instant/Max — eliminates typewriter wait
        entirely and stabilization passes on the very first hash poll."""
        if not PYTESSERACT_AVAILABLE:
            return ""
        try:
            w, h = frame.size
            # Crop to dialogue box region (lower ~40%)
            text_region = frame.crop((0, int(h * 0.60), w, h))
            # PSM 6 = uniform block of text (good for VN dialogue boxes)
            # OEM 1 = LSTM neural net (most accurate for clean VN fonts)
            text = pytesseract.image_to_string(
                text_region,
                config="--psm 6 --oem 1",
            ).strip()
            return text
        except Exception as e:
            print(f"   [Autopilot] OCR error: {e}")
            return ""

    async def _read_dialogue_cloud(self, frame) -> str:
        """Read dialogue text from the FULL VN window using cloud vision (gpt-4o-mini).

        Sends the full window capture — the vision model handles any VN layout:
        bottom ADV box, full-screen NVL text, side-bar narration — and ignores
        UI chrome (save/load/auto/menu buttons, copyright overlays, chapter titles)
        automatically. No hardcoded crop needed; works on any VN.

        Pipelining: called in _post_advance_prep() immediately after advance, so
        the cloud round-trip (~0.8-1.5s) overlaps with the screen settle wait and
        the main loop overhead rather than adding dead air between lines.

        Rate-limit backoff: on 429/529 retries up to 3× with exponential delay
        (15s → 30s → 60s) before returning "" so the loop skips rather than crashes.
        """
        if not self.vision_client:
            return ""
        for attempt in range(3):
            try:
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                resp = await asyncio.wait_for(
                    self.vision_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Read the story dialogue or narration currently displayed in this "
                                "visual novel screenshot.\n\n"
                                "SPEAKER ATTRIBUTION: If a speaker name label is visible (typically "
                                "in a small separate name-plate above or beside the dialogue box, or "
                                "as a colored name tag), prefix the line with the speaker's name "
                                "followed by a colon and space, like this:\n"
                                "  Tomoya: I don't know what to say.\n"
                                "If there is NO visible speaker label (pure narration, internal "
                                "monologue, or descriptive text), output the line as-is with NO "
                                "prefix.\n\n"
                                "Ignore ALL UI elements: buttons, menus, save/load/auto/skip/menu "
                                "labels, chapter headings, system overlays, copyright notices (like "
                                "'\u00a9SMP', 'FINuTO', '\u00a9SHIP'), and any non-story text in the "
                                "corners or edges of the screen. Do NOT treat menu items, button "
                                "labels, or system overlays as the speaker name.\n\n"
                                "Return ONLY the formatted output (optionally 'Speaker: ' prefix + "
                                "line), verbatim, exactly as shown. No quotation marks, no "
                                "explanation, no extra prefix.\n"
                                "If there is no story text visible (transition, black screen, "
                                "choice menu, save screen, etc.), return exactly: EMPTY"
                            )},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            }},
                        ],
                    }],
                    max_tokens=400,
                    temperature=0.0,
                ),
                    timeout=self._cloud_call_timeout,
                )
                self._cloud_read_count += 1
                raw = resp.choices[0].message.content.strip()
                if not raw or raw.upper() == "EMPTY":
                    return ""
                return raw
            except asyncio.TimeoutError:
                print(
                    f"   [Autopilot] Cloud read: TIMEOUT after "
                    f"{self._cloud_call_timeout:.1f}s (attempt {attempt + 1}/3) — skipping."
                )
                return ""
            except Exception as e:
                err_low = str(e).lower()
                if any(s in err_low for s in ("rate limit", "429", "529", "overloaded", "too many requests")):
                    delay = 3.0 * (2 ** attempt)    # 3s → 6s → 12s (keep reading cadence tight)
                    print(
                        f"   [Autopilot] Cloud read: rate-limited "
                        f"(attempt {attempt + 1}/3) — backing off {delay:.0f}s."
                    )
                    await asyncio.sleep(delay)
                    continue
                print(f"   [Autopilot] Cloud read error: {e}")
                return ""
        print("   [Autopilot] Cloud read: all retries exhausted — skipping box.")
        return ""

    async def _wait_for_window_recovery(self) -> bool:
        """Wait for the VN window to return after being lost.

        Retries every 2s for up to ~5 minutes. Returns True when the window
        comes back, False if it never returned (triggers failsafe and caller
        should exit the loop). Suppresses repeated log spam after the first notice.
        """
        if not self._warned_window_lost:
            print(
                f"   [Autopilot] Window '{self.vn_window_title}' not reachable — "
                "waiting for it to return (will resume automatically)..."
            )
            self._warned_window_lost = True

        for _ in range(150):  # 150 × 2s = 5 minutes max
            if not self.enabled or self.is_paused:
                return False
            await asyncio.sleep(2.0)
            win = self._find_vn_window()
            if win is not None and win.width > 0 and win.height > 0:
                print(f"   [Autopilot] Window returned: '{win.title}' — resuming.")
                self._warned_window_lost = False
                self._last_progress_time = time.time()
                return True

        # 5 minutes elapsed — game probably closed; trigger failsafe
        print("   [Autopilot] Window did not return after 5 minutes — triggering failsafe.")
        await self._trigger_failsafe("VN_WINDOW_NOT_FOUND")
        return False

    async def _handle_unknown(self, frame) -> bool:
        """Resolve an UNKNOWN classification without stalling.

        Strategy:
          1. Retry classify a few times with stabilization waits — many UNKNOWNs
             are mid-animation frames that settle into DIALOGUE within ~1-2s.
          2. If still UNKNOWN, try a cloud read directly on the frame. If text
             comes back, treat the frame as DIALOGUE (the classifier was wrong;
             the read confirms there's actual story text).
          3. If still no text, treat as TRANSITION/CG: brief wait + advance.
             _safe_advance() verifies the screen changes; if the screen is
             genuinely frozen, _safe_advance triggers STUCK on its own.
          4. Track consecutive unresolved UNKNOWNs. Return False (caller will
             failsafe) only if the streak exceeds _max_unknown_streak — i.e.
             we've cycled through ~10 frames in a row that never produced
             readable text. Otherwise return True (handled, keep looping).
        """
        # ── Step 1: Retry classify (handles animation/transition frames) ─────
        resolved_type: str | None = None
        resolved_frame = frame
        for attempt in range(self._unknown_retry_count):
            await asyncio.sleep(0.8)
            retry_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            if retry_frame is None:
                # Window vanished mid-retry; let outer loop handle recovery
                return True
            retry_type = await self._classify_screen(retry_frame)
            if retry_type != "UNKNOWN":
                print(
                    f"   [Autopilot] UNKNOWN resolved on retry {attempt + 1}/"
                    f"{self._unknown_retry_count} → {retry_type}"
                )
                resolved_type = retry_type
                resolved_frame = retry_frame
                break

        # ── Step 2: If a retry resolved it, route to the normal handler ──────
        if resolved_type == "DIALOGUE":
            await self._handle_dialogue(resolved_frame, None)
            self._consecutive_unknown = 0
            return True
        if resolved_type == "TRANSITION":
            await asyncio.sleep(0.8)
            self._safe_advance()
            await asyncio.sleep(1.0)
            self._last_progress_time = time.time()
            self._consecutive_unknown = 0
            return True
        if resolved_type in ("CHOICE", "SAVE_PROMPT"):
            # Real menu under the busy frame — hand off
            await self._trigger_failsafe(resolved_type)
            return True   # we handled it (by failing safely)

        # ── Step 3: Still UNKNOWN — try a direct cloud read ──────────────────
        # The classifier may simply be uncertain on a busy/styled frame; if the
        # vision model can read story text from it, it IS dialogue.
        try:
            text = await self._read_dialogue_cloud(resolved_frame)
        except Exception as e:
            print(f"   [Autopilot] UNKNOWN cloud read failed: {e}")
            text = ""

        if text and len(text.strip()) >= 4:
            print(
                f"   [Autopilot] UNKNOWN had readable text "
                f"({len(text)} chars) — treating as DIALOGUE."
            )
            await self._handle_dialogue(resolved_frame, None)
            self._consecutive_unknown = 0
            return True

        # ── Step 4: Truly no readable text — advance through like TRANSITION ─
        self._consecutive_unknown += 1
        print(
            f"   [Autopilot] UNKNOWN (no text) #{self._consecutive_unknown}/"
            f"{self._max_unknown_streak} — advancing through."
        )

        if self._consecutive_unknown > self._max_unknown_streak:
            # Long unbroken streak with no readable text ever surfacing.
            # Caller will trigger failsafe.
            return False

        await asyncio.sleep(0.8)
        # _safe_advance() has its own STUCK detection if the screen doesn't
        # change after cycling through all input methods — we don't need a
        # second failsafe path here for "frozen UNKNOWN".
        self._safe_advance()
        await asyncio.sleep(1.0)
        # Treat this as progress for the watchdog — we advanced through a
        # frame, the story is moving even if it's between dialogue lines.
        self._last_progress_time = time.time()
        return True

    async def _stuck_recovery(self) -> bool:
        """Recovery sequence for the stuck watchdog.

        Steps:
          1. Re-find and re-assert focus on the VN window.
          2. Attempt one advance keystroke.
          3. Re-capture frame and classify — if we see DIALOGUE or TRANSITION,
             recovery succeeded (the main loop will read the new content naturally).
          4. If the window is gone, enter window-recovery; that counts as a failure
             here since we can't confirm progress without a frame.

        Returns True if evidence of progress was found, False if stuck persists.
        """
        print("   [Autopilot] Stuck recovery: re-asserting window focus...")

        # Step 1: re-focus
        win = self._find_vn_window()
        if win is None:
            print("   [Autopilot] Stuck recovery: VN window not reachable.")
            return False
        try:
            win.activate()
            await asyncio.sleep(0.4)
        except Exception as e:
            print(f"   [Autopilot] Stuck recovery: focus failed ({e}), trying advance anyway.")

        # Step 2: one advance attempt
        try:
            self._focus_vn_window()
            self.input_controller.advance()
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"   [Autopilot] Stuck recovery: advance failed: {e}")

        # Step 3: re-capture and classify
        try:
            new_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            if new_frame is None:
                print("   [Autopilot] Stuck recovery: frame capture returned None.")
                return False

            screen_type = await self._classify_screen(new_frame)
            print(f"   [Autopilot] Stuck recovery: post-advance screen = {screen_type}")

            if screen_type in ("DIALOGUE", "TRANSITION"):
                print("   [Autopilot] Stuck recovery: screen shows content — resuming.")
                return True

            if screen_type in ("CHOICE", "SAVE_PROMPT"):
                # A menu appeared — this is progress (we're no longer stuck on
                # the same dialogue), but the failsafe logic should handle it.
                # Return True; the main loop will classify and trigger the right failsafe.
                print("   [Autopilot] Stuck recovery: menu screen — not stuck, routing to failsafe.")
                return True

            # UNKNOWN: may still be stuck
            print("   [Autopilot] Stuck recovery: screen still UNKNOWN — recovery failed.")
            return False

        except Exception as e:
            print(f"   [Autopilot] Stuck recovery capture error: {e}")
            return False

    async def _handle_dialogue(self, frame, prepared: dict | None = None):
        """Read VN text aloud verbatim, then optionally add a rare brief reaction.

        Correct read-then-advance order:
          1. Fast path: _post_advance_prep pre-computed this box's data → skip stabilize/read.
          2. Normal path: stabilize text region (0.15s × 2) → cloud read → clean → diff.
          3. Phase 2 analysis: attachment + weight + intensity (side-effects here only).
          4. SSML prosody: subtle rate/pitch variation by punctuation + intensity.
          5. Read aloud — wait for TTS to complete fully.
          6. Scene-art awareness background task (only on art hash change).
          7. Optional theory resolution / rare reaction (0.15s gap).
          8. Content-based pause (0.10–0.95s).
          9. Advance to next box (only after reading + pause complete).
         10. _post_advance_prep: capture + cloud read next box → _prepared_next.
             Cloud latency (~0.8-1.5s) is hidden here, during the post-advance settle.
        """
        # ── Per-box timing collector (diagnostic) ────────────────────────────
        _box_n = getattr(self, "_timing_box_counter", 0) + 1
        self._timing_box_counter = _box_n
        _timing = {
            "box_n": _box_n,
            "gap": 0.0, "stabilize": 0.0, "ocr": 0.0, "classify": 0.0,
            "tts": 0.0, "pause": 0.0, "reaction_wait": 0.0,
            "advance": 0.0, "prep": 0.0,
            "retries": 0, "dup_recapture": 0.0,
        }

        # ── Fast path: use pipeline-prepared data ────────────────────────────
        if prepared and prepared.get("box_data"):
            box          = prepared["box_data"]
            new_text     = box["new_text"]
            normalized   = box["normalized"]
            text         = box["text"]
            stable_frame = box["stable_frame"]
            weight       = box["weight"]
            self._last_box_text = normalized
            print(
                f"   [Autopilot] Reading (fast-path): '{new_text[:60]}' "
                f"({len(new_text)} chars)"
            )
        else:
            # ── Normal path: stabilize → cloud read → diff ───────────────────
            # Hash only the text region (lower ~40%) so animated backgrounds
            # (Steins;Gate, Clannad) don't prevent stabilization.
            _stab_t0 = time.monotonic()
            prev_hash    = self._text_region_hash(frame, self.text_region_top)
            stable_frame = frame
            for _ in range(2):
                await asyncio.sleep(0.15)
                curr_frame = await asyncio.get_event_loop().run_in_executor(
                    None, self._grab_frame_sync
                )
                if curr_frame is None:
                    break  # window lost mid-stabilize — use last good frame
                curr_hash = self._text_region_hash(curr_frame, self.text_region_top)
                if curr_hash == prev_hash:
                    stable_frame = curr_frame
                    break
                prev_hash    = curr_hash
                stable_frame = curr_frame
            _timing["stabilize"] += time.monotonic() - _stab_t0

            # Cheap local pre-check: blank/fade frames return empty from cloud OCR
            # anyway, but each cloud call costs ~0.8-1.5s + can rate-limit. Skip
            # them entirely — just advance and let the next frame settle.
            if self._text_region_likely_blank(stable_frame, self.text_region_top):
                print(
                    "   [Autopilot] Text region looks blank (fade/transition) — "
                    "advancing without cloud read."
                )
                await asyncio.sleep(0.25)
                self._safe_advance()
                return

            # Cloud read with ONE quick retry. Classifier said DIALOGUE, so text
            # SHOULD be there — but if a fast retry also comes back empty, it's
            # a transition frame, not real dialogue. Hard time budget: ~2s total.
            text = ""
            read_attempts = 2
            _ocr_t0 = time.monotonic()
            for attempt in range(read_attempts):
                _timing["retries"] = attempt  # 0 on first try; ends at total attempts-1
                try:
                    raw = await self._read_dialogue_cloud(stable_frame)
                except Exception as _cloud_err:
                    print(
                        f"   [Autopilot] Cloud read failed "
                        f"(attempt {attempt + 1}/{read_attempts}): {_cloud_err}"
                    )
                    raw = ""
                candidate = self._clean_text((raw or "").strip())
                if candidate and candidate.upper() != "NO TEXT VISIBLE":
                    text = candidate
                    if attempt > 0:
                        print(
                            f"   [Autopilot] OCR recovered on retry "
                            f"{attempt + 1}/{read_attempts}."
                        )
                    break
                # Empty — log and try once more (only one retry allowed).
                print(
                    f"   [Autopilot] OCR empty on DIALOGUE screen "
                    f"(attempt {attempt + 1}/{read_attempts}) — quick retry."
                )
                if attempt < read_attempts - 1:
                    await asyncio.sleep(0.25)
                    fresh = await asyncio.get_event_loop().run_in_executor(
                        None, self._grab_frame_sync
                    )
                    if fresh is not None:
                        # Re-check blank on fresh frame — if it's now blank,
                        # bail immediately instead of burning the retry.
                        if self._text_region_likely_blank(fresh, self.text_region_top):
                            print(
                                "   [Autopilot] Fresh frame now blank — advancing."
                            )
                            await asyncio.sleep(0.2)
                            self._safe_advance()
                            return
                        stable_frame = fresh

            if not text:
                # Fast bail: don't stall on transition frames.
                _timing["ocr"] += time.monotonic() - _ocr_t0
                print(
                    f"   [Autopilot] Skipped read because: OCR empty after "
                    f"{read_attempts} attempts (treating as transition)."
                )
                await asyncio.sleep(0.3)
                self._safe_advance()
                return
            _timing["ocr"] += time.monotonic() - _ocr_t0

            print(
                f"   [Autopilot] Reading dialogue: OCR returned "
                f"'{text[:60]}' ({len(text)} chars)"
            )

            # System-UI guard: if the capture looks like taskbar/desktop, skip
            if self._is_system_ui_text(text):
                print(
                    f"   [Autopilot] Skipped read because: system UI detected "
                    f"(wrong window?). Text='{text[:40]}'"
                )
                await asyncio.sleep(1.5)
                return

            normalized = " ".join(text.split())
            last_preview = self._last_box_text[:40] if self._last_box_text else "<empty>"
            is_new_box = bool(normalized) and normalized != self._last_box_text
            print(
                f"   [Autopilot] New text? {is_new_box} "
                f"(last was '{last_preview}')"
            )

            # Dedup: identical re-capture → screen probably hasn't rendered
            # the new box yet. Wait briefly and re-read ONCE; if still duplicate,
            # then advance again. Avoids stalling on a transient race where we
            # capture mid-render.
            if normalized and normalized == self._last_box_text:
                _dup_t0 = time.monotonic()
                print(
                    f"   [Autopilot] Duplicate read ('{normalized[:40]}') — "
                    f"waiting 0.4s for new frame."
                )
                await asyncio.sleep(0.4)
                fresh = await asyncio.get_event_loop().run_in_executor(
                    None, self._grab_frame_sync
                )
                if fresh is not None:
                    try:
                        raw2 = await self._read_dialogue_cloud(fresh)
                    except Exception:
                        raw2 = ""
                    cand2 = self._clean_text((raw2 or "").strip())
                    if cand2 and cand2.upper() != "NO TEXT VISIBLE":
                        norm2 = " ".join(cand2.split())
                        if norm2 != self._last_box_text:
                            # New content arrived — fall through to handle it.
                            text = cand2
                            normalized = norm2
                            stable_frame = fresh
                            _timing["dup_recapture"] += time.monotonic() - _dup_t0
                            print(
                                f"   [Autopilot] Re-capture found new text — continuing."
                            )
                        else:
                            _timing["dup_recapture"] += time.monotonic() - _dup_t0
                            print(
                                f"   [Autopilot] Still duplicate after re-capture — advancing."
                            )
                            self._note_non_progress()
                            self._safe_advance()
                            return
                    else:
                        # Empty on re-read — treat as transition.
                        _timing["dup_recapture"] += time.monotonic() - _dup_t0
                        self._note_non_progress()
                        self._safe_advance()
                        return
                else:
                    _timing["dup_recapture"] += time.monotonic() - _dup_t0
                    self._note_non_progress()
                    self._safe_advance()
                    return

            # Append detection: VN accumulated text on the same box. Uses fuzzy
            # prefix match so minor OCR drift (e.g. "Tanbo" vs "Tarbo") in the
            # shared head doesn't make us re-read the whole accumulated block.
            _suffix = self._fuzzy_accumulation_split(self._last_box_text, normalized)
            if _suffix is not None:
                raw_suffix = _suffix
                if raw_suffix and not raw_suffix[0].isspace():
                    space_idx = raw_suffix.find(' ')
                    raw_suffix = raw_suffix[space_idx:] if space_idx >= 0 else ""
                new_text = raw_suffix.strip()
                if len(new_text) < 4:
                    print(
                        f"   [Autopilot] Skipped read because: append-suffix "
                        f"too short ({len(new_text)} chars) — advancing."
                    )
                    self._last_box_text = normalized
                    self._note_non_progress()
                    await asyncio.sleep(0.5)
                    self._safe_advance()
                    return
            else:
                new_text = normalized

            # Leading-char bleed guard
            if (len(new_text) >= 3
                    and not new_text[0].isspace()
                    and new_text[1] == ' '
                    and new_text[2].isupper()):
                new_text = new_text[2:].strip()

            if len(new_text) < 4:
                print(
                    f"   [Autopilot] Skipped read because: new_text too short "
                    f"after bleed-guard ({len(new_text)} chars) — advancing."
                )
                self._last_box_text = normalized
                await asyncio.sleep(0.5)
                self._safe_advance()
                return

            self._last_box_text = normalized
            print(f"   [Autopilot] Speaking: '{new_text[:60]}'")

        # ── Confirmed new text — update progress timestamp + reset stuck flag ─
        # Only reaches here when we have genuinely NEW content (past dedup and
        # append-detection). This is the authoritative progress signal for the
        # stuck watchdog — purely mechanical de-dup loops don't count as progress.
        self._last_progress_time = time.time()
        self._stuck_warned = False
        # Real progress — reset the non-progress counter so the working method
        # stays locked in.
        self._consecutive_non_progress_reads = 0

        # ── Phase 2 analysis (side-effects: always in main path, never pipeline) ─
        # Attachment update first so weight computation sees the current state.
        self._update_character_attachment(new_text)
        if not (prepared and prepared.get("box_data")):
            weight = self._estimate_narrative_weight(new_text)
        intensity = self._estimate_scene_intensity(weight)
        self.scene_intensity = intensity

        self.total_boxes_read += 1
        self._boxes_since_reaction += 1
        self._boxes_since_theory_check += 1
        self._boxes_since_solo_aside += 1

        # Bounded buffer: keep a rolling tail (capped at 30 entries) so the
        # buffer never grows unboundedly across a long multi-hour playthrough.
        self.vn_recent_text_buffer.append(text)
        if len(self.vn_recent_text_buffer) > 30:
            self.vn_recent_text_buffer = self.vn_recent_text_buffer[-20:]
        self.vn_boxes_since_summary += 1

        # Narrative summary: background task — never blocks reading.
        # Guarded by disk-space check: if G: is critically full, skip the write
        # this cycle rather than risk corrupting ChromaDB.
        # Also wrapped in try/except so a scheduling error is logged and skipped.
        # H5: cold-start — fire FIRST summary at 5 boxes (vs 15) so reactions,
        # theories, and route-aware choices have context early in the session.
        first_summary_threshold = 5 if not self.vn_narrative_summary else 15
        if (self.vn_boxes_since_summary >= first_summary_threshold
                and len(self.vn_recent_text_buffer) >= 3):
            if self._check_disk_space():
                try:
                    asyncio.ensure_future(self._update_narrative_summary())
                except Exception as _mem_err:
                    print(f"   [Autopilot] Narrative summary task error (continuing): {_mem_err}")

        # Art-change check (before _maybe_update_scene_art overwrites the hash)
        art_changed = self._art_hash(stable_frame) != self._last_art_hash

        # ── H1: Speaker attribution ──────────────────────────────────────────
        speaker, line_only = self._parse_speaker_line(new_text)
        if speaker:
            self._last_speaker = speaker
            # Boost attachment for this speaker directly (more reliable than
            # capitalized-word scraping in _update_character_attachment).
            self._char_mention_counts[speaker] = self._char_mention_counts.get(speaker, 0) + 1
            cnt = self._char_mention_counts[speaker]
            raw = min(1.0, (cnt / 200.0) ** 0.5)
            self.character_attachment[speaker] = min(
                1.0, max(self.character_attachment.get(speaker, 0.0), raw)
            )
            print(f"   [Autopilot] Speaker parsed: '{speaker}' → '{line_only[:50]}'")

        # ── H2/H3: Schedule reaction tasks BEFORE pause/advance ─────────────
        # Tasks run in parallel with the content pause (head-start). After the
        # speech completes we wait an intensity-tiered grace window for the
        # highest-priority reaction to finish, speak it (so it lands while the
        # CURRENT box is still on screen), THEN advance + prep the next box.
        # This costs ~grace seconds on boxes with reactions but gives proper
        # visual sync — reaction overlaps the line it's reacting to, not the next.
        theory_task: "asyncio.Task | None" = None
        try:
            theory_task = asyncio.ensure_future(
                self._check_theory_resolutions(new_text)
            )
        except Exception as _th_err:
            print(f"   [Autopilot] Theory resolution scheduling error: {_th_err}")

        reaction_task: "asyncio.Task | None" = None
        if self._should_consider_reaction(weight, intensity):
            try:
                reaction_task = asyncio.ensure_future(
                    self._decide_reaction(
                        new_text, intensity=intensity, narrative_weight=weight
                    )
                )
            except Exception as _re_err:
                print(f"   [Autopilot] Reaction decision scheduling error: {_re_err}")

        aside_task: "asyncio.Task | None" = None
        if (
            intensity == INTENSITY_CALM
            and self.chat_dead_min >= 4.0
            and self._boxes_since_solo_aside >= 7
            and self._boxes_since_reaction >= 4
        ):
            try:
                aside_task = asyncio.ensure_future(self._maybe_form_theory())
            except Exception as _as_err:
                print(f"   [Autopilot] Solo aside scheduling error: {_as_err}")

        # ── Read aloud — wait for TTS to finish before advancing ─────────────
        # ---- TIMING: capture gap_since_last_speech right before TTS begins ----
        # Use time.time() for the gap (since _last_spoke_time is wall-clock).
        _gap_now = time.time()
        _timing["gap"] = max(0.0, _gap_now - getattr(self, "_last_spoke_time", _gap_now))
        if _timing["gap"] > 5.0:
            prev_tail = getattr(self, "_prev_box_tail", 0.0)
            this_pre  = _timing["stabilize"] + _timing["ocr"] + _timing["classify"] + _timing["dup_recapture"]
            unaccounted = max(0.0, _timing["gap"] - prev_tail - this_pre)
            print(
                f"   [Timing] \u26a0\ufe0f LONG GAP: {_timing['gap']:.1f}s of silence "
                f"before box {_timing['box_n']}. Breakdown: "
                f"prev_box_tail(pause+reaction+advance+prep)={prev_tail:.1f}s "
                f"this_box(stabilize={_timing['stabilize']:.1f}s ocr={_timing['ocr']:.1f}s "
                f"classify={_timing['classify']:.1f}s dup_recap={_timing['dup_recapture']:.1f}s "
                f"retries={_timing['retries']}) unaccounted={unaccounted:.1f}s"
            )
        _tts_t0 = time.monotonic()
        if self.on_speak_vn:
            ssml_inner = self._build_speaker_ssml_inner(speaker, line_only, intensity)
            try:
                await self.on_speak_vn(new_text, ssml_inner)
                self._last_spoke_time = time.time()
                print(f"   [Autopilot] TTS complete ({len(new_text)} chars spoken).")
            except Exception as _tts_err:
                print(f"   [Autopilot] TTS error on dialogue read: {_tts_err}")
        elif self.on_speak:
            try:
                await self.on_speak(new_text)
                self._last_spoke_time = time.time()
                print(f"   [Autopilot] TTS complete ({len(new_text)} chars spoken).")
            except Exception as _tts_err:
                print(f"   [Autopilot] TTS error on dialogue read: {_tts_err}")
        else:
            print(
                f"   [Autopilot] WARNING: no on_speak_vn or on_speak callback "
                f"registered — dialogue NOT spoken!"
            )
        _timing["tts"] = time.monotonic() - _tts_t0

        # Scene-art background task (fire-and-forget)
        try:
            asyncio.ensure_future(self._maybe_update_scene_art(stable_frame))
        except Exception as _art_err:
            print(f"   [Autopilot] Scene art task error (continuing): {_art_err}")

        # Brief content pause — capped for steady cadence
        _pause_t0 = time.monotonic()
        pause_sec = min(self._content_pause(new_text, intensity, art_changed), 0.6)
        await asyncio.sleep(pause_sec)
        _timing["pause"] = time.monotonic() - _pause_t0

        # H2: tiered grace by intensity — climactic moments wait longer so the
        # reaction actually lands. Calm stays snappy.
        grace = self._reaction_grace(intensity)
        # Short-circuit: if only theory_task is pending (no reaction, no aside),
        # don't burn the full grace window — theory hits matter less and most
        # boxes land here. Drops ~0.3-0.5s on every calm line.
        if reaction_task is None and aside_task is None:
            grace = min(grace, 0.20)

        # Speak first-ready reaction WHILE current box is still on screen (H3)
        _react_t0 = time.monotonic()
        spoke_extra = await self._speak_first_ready_reaction(
            theory_task, reaction_task, aside_task,
            grace_seconds=grace,
        )
        _timing["reaction_wait"] = time.monotonic() - _react_t0
        if spoke_extra:
            self._last_spoke_time = time.time()
            self._boxes_since_reaction = 0
            self._boxes_since_solo_aside = 0

        # NOW advance — next box renders after our reaction landed
        _adv_t0 = time.monotonic()
        self._safe_advance()
        _timing["advance"] = time.monotonic() - _adv_t0

        # Pre-prep the next box (hides OCR latency before next read).
        # Hard cap at 5s — if it stalls (hung cloud call, lost window), abandon
        # and let the next loop iteration do a fresh read instead of blocking.
        _prep_t0 = time.monotonic()
        if self.enabled and not self.soft_paused:
            try:
                result = await asyncio.wait_for(
                    self._post_advance_prep(), timeout=5.0
                )
                if result:
                    self._prepared_next = result
            except asyncio.TimeoutError:
                print("   [Autopilot] \u26a0\ufe0f Post-advance prep TIMED OUT after 5s \u2014 abandoning; will read fresh next loop.")
                self._prepared_next = None
            except Exception as e:
                print(f"   [Autopilot] Post-advance prep error: {e}")
        _timing["prep"] = time.monotonic() - _prep_t0

        # ---- TIMING: emit per-box summary line ----
        _tail = _timing["pause"] + _timing["reaction_wait"] + _timing["advance"] + _timing["prep"]
        self._prev_box_tail = _tail
        _total = (
            _timing["stabilize"] + _timing["ocr"] + _timing["classify"]
            + _timing["dup_recapture"] + _timing["tts"] + _tail
        )
        _extras = ""
        if _timing["retries"]:
            _extras += f" [ocr_retries={_timing['retries']}]"
        if _timing["dup_recapture"] > 0:
            _extras += f" [dup_recap={_timing['dup_recapture']:.1f}s]"
        print(
            f"   [Timing] box {_timing['box_n']}: "
            f"gap_since_last_speech={_timing['gap']:.1f}s | "
            f"stabilize={_timing['stabilize']:.1f}s ocr={_timing['ocr']:.1f}s "
            f"classify={_timing['classify']:.1f}s tts={_timing['tts']:.1f}s "
            f"pause={_timing['pause']:.1f}s reaction_wait={_timing['reaction_wait']:.1f}s "
            f"advance={_timing['advance']:.1f}s prep={_timing['prep']:.1f}s "
            f"| total={_total:.1f}s{_extras}"
        )

    async def _speak_first_ready_reaction(
        self,
        theory_task,
        reaction_task,
        aside_task,
        grace_seconds: float = 0.6,
    ) -> bool:
        """Speak the highest-priority reaction that finished (or finishes within
        grace_seconds). Cancels the rest so they don't speak later out of order.

        Returns True if anything was spoken.
        Also updates _last_spoke_time on success (for silence-breaker tracking).
        """
        # Priority order: theory resolution > reaction > solo aside
        ordered = [
            ("theory",   theory_task),
            ("reaction", reaction_task),
            ("aside",    aside_task),
        ]
        ordered = [(lbl, t) for lbl, t in ordered if t is not None]
        if not ordered:
            return False

        # Wait for the FIRST one to finish, up to grace_seconds
        pending_tasks = [t for _, t in ordered if not t.done()]
        if pending_tasks:
            try:
                await asyncio.wait(
                    pending_tasks,
                    timeout=grace_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                pass

        spoke = False
        for label, task in ordered:
            if spoke:
                # Already spoke a higher-priority one — cancel the rest
                if not task.done():
                    task.cancel()
                continue
            if not task.done():
                # Didn't finish in grace window — drop it (cancel so it can't
                # speak later out of order)
                task.cancel()
                continue
            try:
                result = task.result()
            except (asyncio.CancelledError, Exception) as e:
                if not isinstance(e, asyncio.CancelledError):
                    print(f"   [Autopilot] {label} task error: {e}")
                continue
            if result and isinstance(result, str) and result.upper() != "SILENT":
                if self.on_speak:
                    try:
                        await self.on_speak(result)
                        self._last_spoke_time = time.time()
                        spoke = True
                    except Exception as _sp_err:
                        print(f"   [Autopilot] TTS error on {label}: {_sp_err}")
        return spoke

    async def _post_advance_prep(self) -> dict | None:
        """After advancing to the next box, pre-capture and OCR it to hide render latency.

        Called immediately after _safe_advance() — the VN has already advanced.
        Hides the render+stabilize+OCR gap between boxes by running it before the
        next _loop iteration needs it.

        Returns dict with {screen_type, frame, box_data} or None if aborted.
        box_data is None for non-dialogue or dedup screens; _loop falls back to
        the normal OCR path.
        """
        # Per-step prep timing (diagnostic — helps pinpoint which sub-step hangs).
        _ps = {"capture": 0.0, "stabilize": 0.0, "ocr": 0.0, "classify": 0.0}
        _ps_t0 = time.monotonic()
        try:
            # _safe_advance already waited 0.3s for verify — the frame is
            # post-advance. Tiny extra wait to let text-typewriter start, then go.
            await asyncio.sleep(0.05)

            if not self.enabled:
                return None

            _cap_t0 = time.monotonic()
            frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            _ps["capture"] += time.monotonic() - _cap_t0
            if frame is None:
                self._emit_prep_timing(_ps, _ps_t0)
                return None  # window temporarily gone — loop will handle recovery

            # Stabilize: single short pass on the text region. Most renders are
            # already settled by the time we get here; if not, the next loop
            # iteration's normal-path stabilize will pick up the slack.
            _stab_t0 = time.monotonic()
            prev_hash    = self._text_region_hash(frame, self.text_region_top)
            stable_frame = frame
            await asyncio.sleep(0.10)
            _cap_t0 = time.monotonic()
            curr_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            _ps["capture"] += time.monotonic() - _cap_t0
            if curr_frame is not None:
                curr_hash = self._text_region_hash(curr_frame, self.text_region_top)
                if curr_hash == prev_hash:
                    stable_frame = curr_frame
                else:
                    stable_frame = curr_frame  # take latest; full settle happens in main path
            _ps["stabilize"] += time.monotonic() - _stab_t0

            # Cheap local pre-check: skip cloud entirely on blank/fade frames.
            # On blank, we still need classify to know if it's a real transition
            # vs just a fade between dialogue boxes — fall through to classify.
            blank = self._text_region_likely_blank(stable_frame, self.text_region_top)

            # HAPPY-PATH OPTIMIZATION: skip classify when OCR will succeed.
            # If the frame looks NON-blank, we strongly expect DIALOGUE — try
            # the OCR first; if it returns real text, we're done (no classify
            # call needed, saves ~0.5-1.0s). Only classify if OCR comes back
            # empty or the frame was blank to begin with.
            if blank:
                # Blank frame: need classify to know if real transition or
                # just an inter-box fade
                _cls_t0 = time.monotonic()
                screen_type = await self._classify_screen(frame)
                _ps["classify"] += time.monotonic() - _cls_t0
                self._emit_prep_timing(_ps, _ps_t0)
                if screen_type != "DIALOGUE":
                    return {"screen_type": screen_type, "frame": frame, "box_data": None}
                # classified DIALOGUE but blank — let main loop handle
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # Non-blank: try OCR first (skip classify on success)
            _ocr_t0 = time.monotonic()
            try:
                text = await self._read_dialogue_cloud(stable_frame)
            except Exception as _cloud_err:
                _ps["ocr"] += time.monotonic() - _ocr_t0
                print(f"   [Autopilot] Pipeline cloud read failed: {_cloud_err}")
                self._emit_prep_timing(_ps, _ps_t0)
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}
            _ps["ocr"] += time.monotonic() - _ocr_t0
            text = self._clean_text((text or "").strip())
            if not text or text.upper() == "NO TEXT VISIBLE":
                # OCR empty on non-blank frame — could be choice/save_prompt/
                # transition that looked dialogue-y. Classify to disambiguate.
                _cls_t0 = time.monotonic()
                screen_type = await self._classify_screen(frame)
                _ps["classify"] += time.monotonic() - _cls_t0
                self._emit_prep_timing(_ps, _ps_t0)
                return {"screen_type": screen_type, "frame": frame, "box_data": None}

            # System-UI guard: wrong window captured → abort, don't cache
            if self._is_system_ui_text(text):
                print(f"   [Autopilot] Pipeline: system UI detected — wrong window?")
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            normalized = " ".join(text.split())

            # Dedup check (read-only: _last_box_text not updated in the pipeline)
            if normalized == self._last_box_text:
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # Append detection (fuzzy — tolerates OCR drift in shared prefix)
            _suffix = self._fuzzy_accumulation_split(self._last_box_text, normalized)
            if _suffix is not None:
                raw_suffix = _suffix
                if raw_suffix and not raw_suffix[0].isspace():
                    space_idx = raw_suffix.find(' ')
                    raw_suffix = raw_suffix[space_idx:] if space_idx >= 0 else ""
                new_text = raw_suffix.strip()
                if len(new_text) < 4:
                    self._emit_prep_timing(_ps, _ps_t0)
                    return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}
            else:
                new_text = normalized

            # Leading-char bleed guard
            if (len(new_text) >= 3
                    and not new_text[0].isspace()
                    and new_text[1] == ' '
                    and new_text[2].isupper()):
                new_text = new_text[2:].strip()

            if len(new_text) < 4:
                self._emit_prep_timing(_ps, _ps_t0)
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # Weight only — side-effect methods (_estimate_scene_intensity,
            # _update_character_attachment) stay in _handle_dialogue
            weight = self._estimate_narrative_weight(new_text)

            self._emit_prep_timing(_ps, _ps_t0)
            return {
                "screen_type": "DIALOGUE",
                "frame": frame,
                "box_data": {
                    "new_text": new_text,
                    "normalized": normalized,
                    "text": text,
                    "stable_frame": stable_frame,
                    "weight": weight,
                },
            }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"   [Autopilot] Pipeline prep error: {e}")
            self._emit_prep_timing(_ps, _ps_t0)
            return None

    def _emit_prep_timing(self, ps: dict, t0: float) -> None:
        """Log per-step prep timing. Helps pinpoint which sub-step hangs."""
        total = time.monotonic() - t0
        print(
            f"   [Prep] capture={ps['capture']:.1f}s stabilize={ps['stabilize']:.1f}s "
            f"ocr={ps['ocr']:.1f}s classify={ps['classify']:.1f}s | total={total:.1f}s"
        )
        if total > 5.0:
            biggest = max(ps, key=ps.get)
            print(
                f"   [Prep] \u26a0\ufe0f SLOW PREP: {total:.1f}s. Biggest sub-step: "
                f"{biggest}={ps[biggest]:.1f}s."
            )

    @staticmethod
    def _trim_to_complete_sentences(text: str) -> tuple[str, str]:
        """DEPRECATED — retained as a no-op for any stale callers.

        The held-fragment system was removed: OCR truncates mid-word frequently,
        so 'holding' partial fragments produced garbled output like 'Do y' and
        'Mr.' spoken alone. Reading whatever OCR returns is more coherent than
        any sentence-split heuristic over inherently incomplete text.
        """
        return text, ""

    def _content_pause(self, text: str, intensity: str, art_changed: bool) -> float:
        """Inter-box pause driven by text punctuation and scene intensity.

        Replaces the fixed 0.8s delay with organic, context-sensitive rhythm.
        Tune the PAUSE_* module constants to adjust timing globally.
        """
        stripped = text.strip()
        if art_changed:
            base = PAUSE_SCENE_CHANGE
        elif stripped.endswith(('?', '!')):
            base = PAUSE_QUESTION_EXCLAIM
        elif stripped.endswith('.') or stripped.endswith('\u2026') or stripped.endswith('...'):
            base = PAUSE_SENTENCE_END
        else:
            base = PAUSE_CLAUSE_FLOW
        bonus = 0.0
        if intensity in (INTENSITY_INTENSE, INTENSITY_CLIMACTIC):
            bonus += PAUSE_INTENSITY_BONUS
        elif intensity == INTENSITY_AFTERMATH:
            bonus += PAUSE_AFTERMATH_BONUS
        return base + bonus

    def _build_vn_ssml_inner(self, text: str, intensity: str) -> str:
        """Build SSML inner content for autopilot TTS (inside <voice>...</voice>).

        Applies subtle prosody variation — goal is natural delivery, not theatrical.
        The outer <prosody> with config rate/pitch is applied by ai_core.speak_text_vn;
        this method returns additional nested adjustments when warranted.
        """
        safe_text = xml.sax.saxutils.escape(text)
        stripped  = text.strip()

        # Rate: slightly slower for emotionally heavy moments
        if intensity in (INTENSITY_CLIMACTIC, INTENSITY_INTENSE):
            rate = "-8%"
        elif intensity == INTENSITY_AFTERMATH:
            rate = "-5%"
        else:
            rate = None     # use config default

        # Pitch: slight inflection for questions / exclamations
        if stripped.endswith('?'):
            pitch = "+5%"
        elif stripped.endswith('!'):
            pitch = "+3%"
        else:
            pitch = None

        attrs = []
        if rate:
            attrs.append(f'rate="{rate}"')
        if pitch:
            attrs.append(f'pitch="{pitch}"')

        if attrs:
            return f'<prosody {" ".join(attrs)}>{safe_text}</prosody>'
        return safe_text

    async def _transcribe_frame_cloud(self, frame) -> str:
        """Cloud vision text transcription (GPT-4o-mini) — fallback when local OCR fails."""
        if not self.vision_client:
            return ""
        try:
            from vision_agent import UniversalVisionAgent
            transcribe_prompt = UniversalVisionAgent.TRANSCRIBE_PROMPT
            buf = BytesIO()
            frame.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = await self.vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": transcribe_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_tokens=400,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"   [Autopilot] Cloud transcribe error: {e}")
            return ""

    async def _maybe_update_scene_art(self, frame) -> None:
        """Update the cached scene description when the art region hash changes.
        Called as a background task after each box — never blocks reading.
        During long dialogue stretches on the same background: zero cloud calls."""
        new_hash = self._art_hash(frame)
        if new_hash == self._last_art_hash:
            return  # same background, nothing to do
        self._last_art_hash = new_hash
        if not self.vision_client:
            return
        try:
            buf = BytesIO()
            frame.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Describe this visual novel screen in 1-2 sentences: "
                            "setting, mood, visible characters, lighting. "
                            "If it's mostly blank or abstract, just say so. Be brief."
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        }},
                    ],
                }],
                max_tokens=80,
                temperature=0.0,
            ),
                timeout=self._cloud_call_timeout,
            )
            self._scene_art_description = resp.choices[0].message.content.strip()
            print(f"   [Autopilot] Scene: {self._scene_art_description[:70]}")
        except asyncio.TimeoutError:
            print(f"   [Autopilot] Scene art: TIMEOUT after {self._cloud_call_timeout:.1f}s — using cached.")
        except Exception as e:
            print(f"   [Autopilot] Scene art error: {e}")

    def _should_consider_reaction(self, weight: float, intensity: str) -> bool:
        """Local gate: should we call Claude for a reaction at all?

        Default answer is NO — most calm boxes get no reaction API call.
        This prevents per-box cloud latency from gating the reading flow.
        Claude is only invoked when the local heuristic flags the moment."""
        # Minimum spacing: never react on back-to-back boxes
        if self._boxes_since_reaction < 3:
            return False
        # High narrative weight: always worth considering
        if weight >= 0.45:
            return True
        # Intense scene with meaningful weight
        if intensity == INTENSITY_INTENSE and weight >= 0.30:
            return True
        # Rising tension: occasional commentary
        if intensity == INTENSITY_BUILDING and weight >= 0.35:
            return True
        # Calm/aftermath: only when chat is dead (solo mode)
        if intensity in (INTENSITY_CALM, INTENSITY_AFTERMATH):
            if self.chat_dead_min >= 3.0 and self._boxes_since_solo_aside >= 5:
                return True
            # Forced periodic check so reactions still occur in long calm stretches
            if self._boxes_since_reaction >= 18:
                return True
            return False
        return False

    async def _decide_reaction(
        self,
        text: str,
        intensity: str = INTENSITY_CALM,
        narrative_weight: float = 0.2,
    ) -> str:
        """Ask Claude whether to react and how. Returns reaction text or 'SILENT'.

        Phase 2: Reaction rate and style scale with intensity (System 1).
        Prompt shifts to solo/monologue mode when chat is dead (System 2).
        Attached characters get proportional emotional weight (System 4).
        Open theories are referenced when relevant (System 3).
        """
        if not self.ai_core.anthropic_client:
            return "SILENT"

        # Fast-path: climactic moments should almost always be silent unless
        # the weight is exceptional. Don't even ask Claude — silence IS the reaction.
        if intensity == INTENSITY_CLIMACTIC and narrative_weight < 0.88:
            return "SILENT"

        try:
            # System 1: intensity-specific instruction
            intensity_instruction = self._build_intensity_instruction(intensity, narrative_weight)

            # System 2: solo-mode instruction based on dead-chat timer
            dead = self.chat_dead_min
            if dead >= 6.0:
                solo_instruction = (
                    f"\n\nCHAT IS DEAD — no messages for {dead:.0f} minutes. You're narrating "
                    f"solo. The reading carries the stream, but it's fine to occasionally slip "
                    f"in a brief aside — address absent viewers, voice a thought, acknowledge "
                    f"the silence. Don't over-do it; the story is still the main thing."
                )
            elif dead >= 3.0:
                solo_instruction = (
                    "\n\nChat is quiet. An occasional brief aside is fine, but the reading "
                    "is still the primary content — don't react just to fill space."
                )
            else:
                solo_instruction = ""

            # System 4: character attachment context
            attached = [(c, v) for c, v in self.character_attachment.items() if v >= 0.3]
            attachment_instruction = ""
            if attached:
                top = sorted(attached, key=lambda x: -x[1])[:4]
                char_str = ", ".join(f"{c} ({v:.1f})" for c, v in top)
                attachment_instruction = (
                    f"\n\nCHARACTER ATTACHMENT this session: {char_str} "
                    f"(scale 0.0=stranger, 1.0=deeply attached). "
                    f"If a box involves a character you're attached to, the reaction should "
                    f"carry proportional emotional weight. Don't treat their moments as generic."
                )

            # System 3: surface open theories when relevant
            theory_instruction = ""
            open_theories = [t for t in self.active_theories if t["status"] == "open"]
            if open_theories and intensity in (INTENSITY_CALM, INTENSITY_BUILDING):
                theory_str = " | ".join(t["theory"][:80] for t in open_theories[:3])
                theory_instruction = (
                    f"\n\nACTIVE THEORIES you've formed: {theory_str}. "
                    f"If this box is relevant to one of them, you can reference it naturally."
                )

            # Story investment note
            investment_note = ""
            if self.story_investment > 0.55:
                investment_note = "\n\nYou're deeply invested in this story now."
            elif self.story_investment > 0.25:
                investment_note = "\n\nYou've genuinely gotten into this story."

            # Narrative + trajectory context
            narrative_block = (
                f"\n\nSTORY SO FAR:\n{self.vn_narrative_summary}"
                if self.vn_narrative_summary else ""
            )
            trajectory_note = (
                f"\n\nYOUR EMOTIONAL ARC SO FAR: {self.emotional_trajectory}"
                if self.emotional_trajectory else ""
            )

            # Visual scene context (cached from background art-awareness task)
            scene_art_note = (
                f"\n\nCURRENT SCENE: {self._scene_art_description}"
                if self._scene_art_description else ""
            )

            # Pull semantic memory + playthrough memory + voice guardrails via the
            # bot reference so in-character VN reactions aren't memory-blind. Falls
            # back gracefully when bot wasn't wired (older callers / tests).
            memory_block = ""
            playthrough_block = ""
            guardrails = ""
            persona_system = self.ai_core.system_prompt
            if self.bot is not None:
                try:
                    mem = self.bot.memory.get_semantic_context(text)
                    if mem:
                        memory_block = (
                            f"\n\n[MEMORY NOTES — do not quote, do not invent things not present here]\n{mem}"
                        )
                except Exception:
                    pass
                if getattr(self.bot, "playthrough_memory", None):
                    try:
                        pt_ctx = self.bot.playthrough_memory.get_context_for_prompt()
                        if pt_ctx:
                            playthrough_block = (
                                f"\n\n[PLAYTHROUGH MEMORY — reference as lived experience, not data]\n{pt_ctx}"
                            )
                    except Exception:
                        pass
                try:
                    guardrails = self.bot._kira_voice_guardrails()
                except Exception:
                    pass

            prompt = (
                f"You are Kira, reading a visual novel aloud on stream — you ARE the narrator.\n\n"
                f"You just read this aloud:\n\"{text}\"\n"
                f"{narrative_block}{scene_art_note}{trajectory_note}{attachment_instruction}"
                f"{theory_instruction}{investment_note}{solo_instruction}"
                f"{playthrough_block}{memory_block}\n\n"
                f"{intensity_instruction}\n\n"
                f"Your PRIMARY job is narrating the story. Reactions are RARE seasoning — only "
                f"break from narration when a moment genuinely earns it: something funny, an "
                f"emotional gut-punch, a surprising twist, or (if chat is dead + scene is calm) "
                f"a brief aside. DEFAULT IS SILENT. Most boxes get no reaction.\n\n"
                f"CRITICAL: the viewer already HEARD you read that text. If you react, ADD "
                f"something new — a feeling, a joke, a question, an observation. NEVER restate "
                f"or paraphrase what was just read. 1 short sentence max, clearly your own voice.\n\n"
                f"React (1 sentence max, adds something new) or SILENT?\n"
                f"Output ONLY the reaction text or the word SILENT."
                f"{guardrails}"
            )
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=160,
                system=persona_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"   [Autopilot] Reaction decision error: {e}")
            return "SILENT"

    # ── Phase 2: System 1 — Dynamic Energy / Pacing ───────────────────────────

    def _estimate_narrative_weight(self, text: str) -> float:
        """Heuristic text-based narrative weight 0.0–1.0.
        No LLM call — rules only. Fast and surprisingly effective."""
        t = text.lower()
        score = 0.20  # baseline: most boxes have some weight

        # Death / permanent loss (strongest signal)
        if any(w in t for w in ("died", " dead", "death", "killed", "never come back",
                                 "farewell", "goodbye forever", "gone forever", "is gone")):
            score += 0.50

        # Revelation / secret exposed
        if any(w in t for w in ("truth", "secret", "real ", "lied", "wasn't",
                                 "has always", "never told", "hidden", "discovered",
                                 "realized", "all along", "knew it", "was lying")):
            score += 0.35

        # Emotional confession / romantic peak
        if any(w in t for w in ("i love", "love you", "always loved", "always felt",
                                 "feelings for", "can't hide", "wanted to tell")):
            score += 0.30

        # Climactic language / dramatic punctuation
        if text.count("!") >= 2 or text.count("...") >= 3:
            score += 0.15
        if "?" in text and any(w in t for w in ("why", "how could", "what have you", "what did")):
            score += 0.10

        # Very short connective text → low weight
        if len(text) < 35:
            score -= 0.15

        # Long monologue → likely a significant moment
        if len(text) > 300:
            score += 0.10

        # Boost if an attached character is involved
        for char, attachment in self.character_attachment.items():
            if attachment >= 0.3 and char.lower() in t:
                score += 0.15 * min(attachment, 1.0)

        return max(0.0, min(1.0, score))

    def _estimate_scene_intensity(self, weight: float) -> str:
        """Convert narrative weight float to an intensity state string.
        Preserves aftermath decay so the mood lingers after a climactic moment."""
        if self._aftermath_countdown > 0:
            self._aftermath_countdown -= 1
            return INTENSITY_AFTERMATH

        if weight >= 0.72:
            self._aftermath_countdown = 4     # 4 boxes of aftermath after a climax
            return INTENSITY_CLIMACTIC
        elif weight >= 0.50:
            return INTENSITY_INTENSE
        elif weight >= 0.28:
            return INTENSITY_BUILDING
        else:
            return INTENSITY_CALM

    def _build_intensity_instruction(self, intensity: str, weight: float) -> str:
        """Returns the intensity-specific guidance block injected into the reaction prompt."""
        if intensity == INTENSITY_CLIMACTIC:
            return (
                "SCENE INTENSITY: CLIMACTIC — this is a major moment (high narrative weight). "
                "Silence is the correct response most of the time here. React ONLY if you "
                "have something that genuinely adds emotional weight. Don't interrupt the moment."
            )
        elif intensity == INTENSITY_INTENSE:
            return (
                "SCENE INTENSITY: INTENSE — things are escalating. React selectively. "
                "Short, sharp, controlled reactions only. No rambling. When in doubt: SILENT."
            )
        elif intensity == INTENSITY_BUILDING:
            return (
                "SCENE INTENSITY: BUILDING — tension is rising. Good place for tense "
                "anticipation: 'Something feels off about this...', leaning-in commentary."
            )
        elif intensity == INTENSITY_AFTERMATH:
            return (
                "SCENE INTENSITY: AFTERMATH — something big just happened. Kira is absorbing "
                "it. Quiet processing is right here. If you react, react with weight — not a "
                "quick quip, something that shows you felt it."
            )
        else:  # CALM
            return (
                "SCENE INTENSITY: CALM — low-stakes connective content. Default to SILENT "
                "(the reading itself is the content). A reaction is only worth adding if "
                "something in the text is genuinely amusing, curious, or worth a brief "
                "observation. Solo asides fire separately when chat is dead — don't "
                "duplicate them here."
            )

    # ── Phase 2: System 3 — Theory-Building ──────────────────────────────────

    async def _maybe_form_theory(self) -> str | None:
        """During calm stretches, periodically ask Claude to form a genuine theory
        or prediction about the story. Returns the theory text if formed (ready to speak),
        or None. Throttled: at most once per 25 boxes; max 5 open theories."""
        if not self.ai_core.anthropic_client:
            return None
        if self._boxes_since_theory_check < 25:
            return None
        open_count = sum(1 for t in self.active_theories if t["status"] == "open")
        if open_count >= 5:
            return None
        if not self.vn_narrative_summary:
            return None

        self._boxes_since_theory_check = 0

        prior_block = ""
        if self.active_theories:
            all_t = "\n".join(f"  - {t['theory']}" for t in self.active_theories[-6:])
            prior_block = f"\n\nTheories you've already formed (don't repeat these):\n{all_t}"

        prompt = (
            f"You are Kira, autonomously playing a visual novel on stream. "
            f"Based on the story so far, do you have a GENUINE theory, prediction, or "
            f"suspicion about where this is going?\n\n"
            f"STORY SO FAR:\n{self.vn_narrative_summary}{prior_block}\n\n"
            f"If you have a genuine theory (specific — about a character, plot thread, "
            f"or foreshadowed event): output it in Kira's first-person voice, 1-2 sentences. "
            f"Example: 'I don't trust that phone call. Something is off about the timing.'\n"
            f"Example: 'Nagisa's health keeps getting mentioned in passing. "
            f"I have a very bad feeling about where that's going.'\n\n"
            f"If nothing compelling comes to mind: output exactly \"NONE\".\n\n"
            f"Just the theory text or NONE."
        )
        try:
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text.strip()
            if result and result.upper() != "NONE" and len(result) > 15:
                self.active_theories.append({
                    "theory": result,
                    "formed_box": self.total_boxes_read,
                    "status": "open",
                })
                print(f"   [Autopilot] Theory formed: {result[:80]}")
                return result
        except Exception as e:
            print(f"   [Autopilot] Theory formation error: {e}")
        return None

    async def _check_theory_resolutions(self, text: str) -> str | None:
        """Check if the current text confirms or busts an open theory.
        Uses a fast keyword heuristic first; only calls Claude if likely relevant.
        Returns a spoken reaction string on resolution, or None."""
        open_theories = [t for t in self.active_theories if t["status"] == "open"]
        if not open_theories or not self.ai_core.anthropic_client:
            return None

        # Fast heuristic: any significant overlap between theory words and current text?
        t_lower = text.lower()
        likely_relevant = False
        for theory in open_theories:
            theory_words = {
                w.lower().strip(".,!?\"'—…()") for w in theory["theory"].split()
                if len(w) > 4
            }
            if sum(1 for w in theory_words if w in t_lower) >= 2:
                likely_relevant = True
                break

        if not likely_relevant:
            return None

        theories_str = "\n".join(
            f"  {i+1}. {t['theory']}" for i, t in enumerate(open_theories[:5])
        )
        prompt = (
            f"You are Kira playing a visual novel. You formed theories about the story.\n\n"
            f"Your open theories:\n{theories_str}\n\n"
            f"You just read:\n\"{text}\"\n\n"
            f"Does this text CONFIRM or BUST any of your theories?\n"
            f"If yes: which theory number, outcome, and Kira's reaction (1-2 sentences "
            f"— 'I CALLED it!' energy for confirms, 'I was completely wrong about that' "
            f"for busts).\n"
            f"If no clear connection: output exactly \"NONE\".\n\n"
            f"Format if resolving: CONFIRM:1:reaction text  OR  BUST:2:reaction text\n"
            f"Output ONLY that format or NONE."
        )
        try:
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text.strip()
            if not result or result.upper() == "NONE":
                return None
            parts = result.split(":", 2)
            if len(parts) == 3:
                resolution = parts[0].upper()
                try:
                    idx = int(parts[1]) - 1
                except ValueError:
                    return None
                reaction = parts[2].strip()
                if 0 <= idx < len(open_theories) and reaction:
                    open_theories[idx]["status"] = "confirmed" if resolution == "CONFIRM" else "busted"
                    open_theories[idx]["resolved_box"] = self.total_boxes_read
                    print(f"   [Autopilot] Theory {resolution}: {open_theories[idx]['theory'][:60]}")
                    return reaction
        except Exception as e:
            print(f"   [Autopilot] Theory resolution error: {e}")
        return None

    # ── Phase 2: System 4 — Within-Session Emotional State ───────────────────

    def _update_character_attachment(self, text: str):
        """Track character mentions. Attachment grows with screen time.
        Uses capitalized-word extraction — imperfect but zero-cost."""
        common = {
            "I", "It", "The", "A", "An", "He", "She", "They", "We", "You",
            "But", "And", "Or", "So", "Then", "When", "If", "That", "This",
            "Is", "Was", "Be", "Have", "Do", "Not", "No", "Yes", "What",
            "My", "His", "Her", "Our", "Your", "Their", "Me", "Him",
        }
        for word in text.split():
            cleaned = word.strip(".,!?\"'—…()[]")
            if (cleaned and cleaned[0].isupper() and cleaned not in common
                    and len(cleaned) > 2 and cleaned.isalpha()):
                self._char_mention_counts[cleaned] = self._char_mention_counts.get(cleaned, 0) + 1
                count = self._char_mention_counts[cleaned]
                if count >= 3:
                    # Sqrt ramp: ~50 mentions → 0.5, ~200 mentions → ~1.0
                    raw = min(1.0, (count / 200.0) ** 0.5)
                    self.character_attachment[cleaned] = min(
                        1.0, max(self.character_attachment.get(cleaned, 0.0), raw)
                    )
        # Investment ramps slowly over the session (every box = +0.001)
        self.story_investment = min(1.0, self.story_investment + 0.001)

    def _update_emotional_trajectory(self):
        """Derive a concise trajectory string from session state.
        No LLM call — purely computed from tracked values."""
        parts = []
        if self.total_boxes_read > 300:
            parts.append("deep in the playthrough")
        elif self.total_boxes_read > 100:
            parts.append("several hours in")
        elif self.total_boxes_read > 30:
            parts.append("getting into it")

        if self.story_investment > 0.60:
            parts.append("deeply invested")
        elif self.story_investment > 0.30:
            parts.append("getting invested")

        top_chars = sorted(self.character_attachment.items(), key=lambda x: -x[1])
        attached = [(c, v) for c, v in top_chars if v >= 0.35][:3]
        if attached:
            parts.append("attached to " + ", ".join(c for c, _ in attached))

        self.emotional_trajectory = "; ".join(parts) if parts else ""

    # ── Narrative memory ───────────────────────────────────────────────────────

    async def _update_narrative_summary(self):
        """Build/update rolling ~150-word plot summary from accumulated text boxes.

        Disk-space check: skipped silently (already checked by the caller before
        scheduling this task).
        Rate-limit backoff: 429/529 from Anthropic retries up to 2× (30s → 60s)
        before giving up — avoids losing the summary on a transient overload.
        """
        if not self.ai_core.anthropic_client:
            return
        accumulated = "\n---\n".join(self.vn_recent_text_buffer[-20:])
        prev = self.vn_narrative_summary or "No previous summary — this is the start."
        prompt = (
            f"You are maintaining a running ~150-word story summary for a visual novel playthrough.\n\n"
            f"Previous summary:\n{prev}\n\n"
            f"New dialogue/narration (most recent boxes):\n{accumulated}\n\n"
            f"Write an updated summary in ~150 words. Track character names, events, and emotional beats. "
            f"Be factual and concise. No commentary or editorializing."
        )
        for attempt in range(3):
            try:
                resp = await self.ai_core.anthropic_client.messages.create(
                    model=CLAUDE_CHAT_MODEL,
                    max_tokens=250,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.vn_narrative_summary = resp.content[0].text.strip()
                self.vn_recent_text_buffer = self.vn_recent_text_buffer[-5:]
                self.vn_boxes_since_summary = 0
                self._update_emotional_trajectory()
                print(f"   [Autopilot] Narrative summary updated ({len(self.vn_narrative_summary)} chars).")
                return
            except Exception as e:
                err_low = str(e).lower()
                if any(s in err_low for s in ("rate limit", "429", "529", "overloaded", "too many requests")):
                    delay = 5.0 * (2 ** attempt)   # 5s → 10s → 20s (background task, keep tight)
                    print(
                        f"   [Autopilot] Narrative summary: rate-limited "
                        f"(attempt {attempt + 1}/3) — backing off {delay:.0f}s."
                    )
                    await asyncio.sleep(delay)
                    continue
                print(f"   [Autopilot] Narrative update error: {e}")
                return
        print("   [Autopilot] Narrative summary: all retries exhausted — skipping this cycle.")

    # ── Failsafe ───────────────────────────────────────────────────────────────

    async def _trigger_failsafe(self, screen_type: str):
        """Stop the loop, speak a handoff line, notify the dashboard.
        Sets _hard_failsafed so soft_resume() knows it must not pretend to resume."""
        self.is_running = False
        self.is_paused = True
        self.pause_reason = screen_type
        self._hard_failsafed = True
        print(f"   [Autopilot] Failsafe: {screen_type}")

        line = self.FAILSAFE_LINES.get(screen_type, self.FAILSAFE_LINES["UNKNOWN"])
        if line and self.on_speak:
            try:
                await self.on_speak(line)
                self._last_spoke_time = time.time()
            except Exception as e:
                print(f"   [Autopilot] Could not speak failsafe line: {e}")

        if self.on_failsafe:
            self.on_failsafe(screen_type)

    # ── Speaker attribution (H1) ──────────────────────────────────────────────

    @staticmethod
    def _parse_speaker_line(text: str) -> tuple[str, str]:
        """Split 'Speaker: line' into (speaker, line). Returns ('', text) on no match.

        Conservative: only treats a leading short token + colon as a speaker label.
        Rejects matches where the 'speaker' contains sentence punctuation or looks
        like a sentence fragment.
        """
        if not text:
            return "", ""
        # Must have a colon in the first 32 chars, followed by a space + content
        m = re.match(r'^\s*([A-Za-z0-9 \'\-\.]{1,30})\s*[:\uFF1A]\s+(.+)$', text, re.DOTALL)
        if not m:
            return "", text.strip()
        speaker = m.group(1).strip()
        line = m.group(2).strip()
        # Reject if speaker contains sentence-ending punctuation
        if any(c in speaker for c in '.!?,;'):
            return "", text.strip()
        # Reject if "speaker" is multiple words longer than ~3 (likely a sentence)
        if len(speaker.split()) > 3:
            return "", text.strip()
        # Reject single common words ("Why", "Look", "Hmm", "Oh", etc.) — those
        # are sentence starters, not speaker labels.
        bad = {"why", "look", "hmm", "oh", "ah", "huh", "wait", "no", "yes",
               "but", "well", "but", "so", "what", "and", "or", "if", "then"}
        if speaker.lower() in bad:
            return "", text.strip()
        if not line:
            return "", text.strip()
        return speaker, line

    def _speaker_prosody(self, speaker: str) -> tuple[str, str]:
        """Deterministic (pitch, rate) offset for a speaker name so the same
        character sounds consistent across the session. Empty speaker → no offset.
        Returns ('', '') for narration."""
        if not speaker:
            return "", ""
        if speaker in self._speaker_voice_cache:
            return self._speaker_voice_cache[speaker]
        # Hash → small pitch/rate offsets. Keep modest so the base voice is still recognizable.
        h = int(hashlib.md5(speaker.encode("utf-8")).hexdigest(), 16)
        pitch_offsets = ["-6%", "-4%", "-2%", "0%", "+2%", "+4%", "+6%", "+8%"]
        rate_offsets  = ["-4%", "-2%", "0%", "+2%", "+4%"]
        pitch = pitch_offsets[h % len(pitch_offsets)]
        rate  = rate_offsets[(h // 16) % len(rate_offsets)]
        self._speaker_voice_cache[speaker] = (pitch, rate)
        return pitch, rate

    def _build_speaker_ssml_inner(
        self, speaker: str, line: str, intensity: str
    ) -> str:
        """Build SSML inner content with optional speaker name + per-speaker prosody.
        Speaker name is spoken with a brief break, then the line in the speaker's voice.
        """
        safe_line = xml.sax.saxutils.escape(line)
        # Base intensity rate adjustment
        if intensity in (INTENSITY_CLIMACTIC, INTENSITY_INTENSE):
            base_rate = "-8%"
        elif intensity == INTENSITY_AFTERMATH:
            base_rate = "-5%"
        else:
            base_rate = None
        # Question / exclaim pitch lift
        stripped = line.strip()
        if stripped.endswith('?'):
            base_pitch = "+5%"
        elif stripped.endswith('!'):
            base_pitch = "+3%"
        else:
            base_pitch = None

        if not speaker:
            # Pure narration — fall back to old behavior
            attrs = []
            if base_rate:  attrs.append(f'rate="{base_rate}"')
            if base_pitch: attrs.append(f'pitch="{base_pitch}"')
            return f'<prosody {" ".join(attrs)}>{safe_line}</prosody>' if attrs else safe_line

        # Speaker line — wrap line in speaker prosody, prefix with the name + tiny break
        sp_pitch, sp_rate = self._speaker_prosody(speaker)
        safe_speaker = xml.sax.saxutils.escape(speaker)
        line_attrs = []
        if sp_pitch and sp_pitch != "0%": line_attrs.append(f'pitch="{sp_pitch}"')
        if sp_rate  and sp_rate  != "0%": line_attrs.append(f'rate="{sp_rate}"')
        # Layer intensity adjustment on top of speaker offsets when present
        if base_rate:  line_attrs.append(f'rate="{base_rate}"')  # later attr wins for some engines; keep both
        line_part = (
            f'<prosody {" ".join(line_attrs)}>{safe_line}</prosody>'
            if line_attrs else safe_line
        )
        # Speaker name read flat + tiny pause
        return f'{safe_speaker}<break time="180ms"/> {line_part}'

    # ── Reaction grace (H2) ───────────────────────────────────────────────────

    @staticmethod
    def _reaction_grace(intensity: str) -> float:
        return REACTION_GRACE_BY_INTENSITY.get(intensity, 0.6)

    # ── Silence-breaker (H4) ──────────────────────────────────────────────────

    async def _maybe_speak_silence_breaker(self) -> bool:
        """If we've been silent for too long during active play, fire a low-energy
        aside so the stream doesn't sit dead during long CG/transition stretches.
        Throttled so it can't spam. Returns True if it spoke.

        Gated on TWO conditions to avoid interrupting active dialogue:
          • _last_spoke_time > SILENCE_BREAKER_SECONDS (general silence)
          • _last_dialogue_time > 15s (we're stuck on non-dialogue screens)
        Both must be true — mid-dialogue reads where TTS just happens to be
        between calls do NOT count.
        """
        if self._silence_breaker_in_flight:
            return False
        if self.soft_paused or self.is_paused or not self.enabled:
            return False
        now = time.time()
        if (now - self._last_spoke_time) < SILENCE_BREAKER_SECONDS:
            return False
        if (now - self._last_dialogue_time) < 15.0:
            # Still in active dialogue territory — don't interrupt.
            return False
        if (now - self._last_silence_breaker_time) < SILENCE_BREAKER_MIN_GAP:
            return False
        self._silence_breaker_in_flight = True
        try:
            # Pick a low-energy aside locally — no LLM call needed for this; keeps
            # silence-breaks fast and inexpensive. Cycle the line so it doesn't repeat.
            options = [
                "Hmm.",
                "...okay.",
                "What are we looking at here.",
                "Pretty.",
                "Mmm.",
                "I'm just gonna sit with this for a sec.",
                "Where are we going with this?",
                "Okay, keep going, game.",
            ]
            idx = int(now) % len(options)
            line = options[idx]
            if self.on_speak:
                try:
                    await self.on_speak(line)
                    self._last_silence_breaker_time = time.time()
                    self._last_spoke_time = time.time()
                    return True
                except Exception as e:
                    print(f"   [Autopilot] Silence-breaker TTS error: {e}")
            return False
        finally:
            self._silence_breaker_in_flight = False

    # ── CHOICE auto-pick (C1) ─────────────────────────────────────────────────

    async def _extract_choices(self, frame) -> list[dict] | None:
        """Use cloud vision to extract choice options with positions.
        Returns a list of {text, x_pct, y_pct} or None on failure.
        Positions are 0..1 fractions of frame size for click conversion.
        """
        if not self.vision_client:
            return None
        try:
            buf = BytesIO()
            frame.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = await self.vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "This is a visual novel CHOICE screen. List every selectable "
                            "choice option visible (numbered or button-style). For each, "
                            "estimate the CENTER position of the clickable button as a "
                            "fraction of the image (x: 0=left, 1=right; y: 0=top, 1=bottom).\n\n"
                            "Output ONE LINE PER CHOICE in this exact format:\n"
                            "INDEX | TEXT | X | Y\n"
                            "Example:\n"
                            "1 | Talk to Nagisa | 0.50 | 0.42\n"
                            "2 | Go to the rooftop | 0.50 | 0.58\n\n"
                            "No header, no explanation, no extra text. If this is NOT a "
                            "choice screen with selectable options, output exactly: NOCHOICE"
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_tokens=400,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if not raw or "NOCHOICE" in raw.upper():
                return None
            choices = []
            for ln in raw.splitlines():
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) != 4:
                    continue
                try:
                    idx = int(parts[0])
                    text = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                except ValueError:
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or not text:
                    continue
                choices.append({"index": idx, "text": text, "x_pct": x, "y_pct": y})
            return choices if choices else None
        except Exception as e:
            print(f"   [Autopilot] Choice extract error: {e}")
            return None

    async def _decide_choice(self, choices: list[dict]) -> int:
        """Ask Claude which choice to pick. Returns 0-based index. Defaults to 0 on error."""
        if not self.ai_core.anthropic_client or not choices:
            return 0
        choices_str = "\n".join(f"  {c['index']}. {c['text']}" for c in choices)
        route_block = (
            f"\n\nROUTE GOAL: {self.target_route}\n"
            f"Pick the option that best serves this goal."
            if self.target_route else
            "\n\nNo specific route goal — pick what feels most in-character and "
            "narratively interesting for Kira."
        )
        narrative_block = (
            f"\n\nSTORY SO FAR:\n{self.vn_narrative_summary}"
            if self.vn_narrative_summary else ""
        )
        prompt = (
            f"You are Kira, autonomously playing a visual novel on stream.\n"
            f"A choice has appeared. Pick ONE option.{narrative_block}{route_block}\n\n"
            f"Options:\n{choices_str}\n\n"
            f"Respond with ONLY the option number, nothing else."
        )
        try:
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            m = re.search(r"\d+", raw)
            if not m:
                return 0
            picked = int(m.group(0))
            for i, c in enumerate(choices):
                if c["index"] == picked:
                    return i
            # Fallback: maybe Claude used 0-based
            if 0 <= picked < len(choices):
                return picked
            return 0
        except Exception as e:
            print(f"   [Autopilot] Choice decision error: {e}")
            return 0

    async def _handle_choice(self, frame):
        """Auto-pick a CHOICE screen: extract options, ask Claude, click the pick.
        Falls back to failsafe if auto_choose is off, options can't be parsed,
        or pause_on_major_choices is set.
        """
        if not self.auto_choose:
            await self._trigger_failsafe("CHOICE")
            return

        print("   [Autopilot] CHOICE detected — extracting options...")
        choices = await self._extract_choices(frame)
        if not choices:
            print("   [Autopilot] Could not parse choices — handing off.")
            await self._trigger_failsafe("CHOICE")
            return

        idx = await self._decide_choice(choices)
        idx = max(0, min(idx, len(choices) - 1))
        picked = choices[idx]
        print(
            f"   [Autopilot] Picked choice {idx + 1}/{len(choices)}: "
            f"'{picked['text'][:60]}'"
        )

        # In-character framing line BEFORE clicking
        if self.on_speak:
            framing_options = [
                f"Okay... let's try '{picked['text']}'.",
                f"I'll go with '{picked['text']}'.",
                f"Let's see what happens if I pick '{picked['text']}'.",
                f"Alright, '{picked['text']}' it is.",
            ]
            framing = framing_options[hash(picked['text']) % len(framing_options)]
            try:
                await self.on_speak(framing)
                self._last_spoke_time = time.time()
            except Exception as e:
                print(f"   [Autopilot] Choice framing TTS error: {e}")

        # Convert percentage to absolute coords using the VN window rect
        win = self._find_vn_window()
        if win is None:
            print("   [Autopilot] Choice click: VN window lost — handing off.")
            await self._trigger_failsafe("CHOICE")
            return
        abs_x = int(win.left + win.width  * picked["x_pct"])
        abs_y = int(win.top  + win.height * picked["y_pct"])
        print(f"   [Autopilot] Clicking choice at ({abs_x}, {abs_y})")

        # Focus, click, verify screen changed
        self._focus_vn_window()
        try:
            self.input_controller.click_at(abs_x, abs_y)
        except Exception as e:
            print(f"   [Autopilot] Choice click failed: {e} — handing off.")
            await self._trigger_failsafe("CHOICE")
            return

        # Wait for the engine to advance past the choice
        await asyncio.sleep(1.2)

        # Verify we're no longer on a CHOICE screen
        verify_frame = await asyncio.get_event_loop().run_in_executor(
            None, self._grab_frame_sync
        )
        if verify_frame is not None:
            new_type = await self._classify_screen(verify_frame)
            if new_type == "CHOICE":
                print("   [Autopilot] Still on CHOICE after click — handing off.")
                await self._trigger_failsafe("CHOICE")
                return
            print(f"   [Autopilot] Choice resolved → next screen: {new_type}")

        # Reset read-state so post-choice content is fresh
        self._last_box_text = ""
        self._prepared_next = None
        self._last_progress_time = time.time()

    # ── SAVE_PROMPT soft-dismiss (C2) ─────────────────────────────────────────

    async def _handle_save_prompt(self, frame):
        """Try to dismiss a menu by pressing Escape. If it doesn't go away
        after a couple tries, failsafe."""
        if not self.auto_dismiss_menus:
            await self._trigger_failsafe("SAVE_PROMPT")
            return

        print("   [Autopilot] SAVE_PROMPT — attempting Escape dismissal...")
        for attempt in range(2):
            try:
                self._focus_vn_window()
                self.input_controller.press_escape()
            except Exception as e:
                print(f"   [Autopilot] Escape press failed: {e}")
                break
            await asyncio.sleep(0.8)
            verify_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            if verify_frame is None:
                continue
            new_type = await self._classify_screen(verify_frame)
            if new_type != "SAVE_PROMPT":
                print(f"   [Autopilot] Menu dismissed → {new_type}")
                self._last_box_text = ""
                self._prepared_next = None
                self._last_progress_time = time.time()
                return
            print(f"   [Autopilot] Menu still present after Escape #{attempt + 1}")

        print("   [Autopilot] Could not dismiss menu — handing off.")
        await self._trigger_failsafe("SAVE_PROMPT")

    # ── Input helper ───────────────────────────────────────────────────────────

    def _note_non_progress(self):
        """Called when an advance produced an OCR result that was a duplicate /
        prefix / accumulating text (i.e. NO real story progress). After two in a
        row, unlock the currently-locked advance method so _safe_advance tries
        alternatives next time. This catches the case where the locked method
        is actually a HUD/menu toggle that happens to change the screen hash."""
        self._consecutive_non_progress_reads += 1
        if (self._consecutive_non_progress_reads >= 2
                and self._working_advance_method is not None):
            print(
                f"   [Autopilot] {self._consecutive_non_progress_reads} consecutive "
                f"non-progress reads via '{self._working_advance_method}' \u2014 unlocking "
                f"working method so alternatives can be tried."
            )
            self._working_advance_method = None
            self._consecutive_non_progress_reads = 0

    def _safe_advance(self):
        """Advance the VN, VERIFY by screen-hash change, retry with other methods if needed.

        Per-advance flow:
          1. Capture before-frame + hash.
          2. Focus the VN window (log if focus failed).
          3. Try the preferred method (working-method if known, else default).
          4. Wait ~0.6s, re-capture, compare hashes.
          5. If changed → log success, remember the working method, stash the
             new frame for _post_advance_prep to reuse.
          6. If unchanged → log "NO EFFECT", cycle through other methods
             (re-focusing before each). First method to produce a change wins
             and is remembered.
          7. If NO method works after exhausting METHODS_ORDER → log + failsafe
             with "STUCK" so Jonny knows to intervene.

        Diagnostic verbosity: the first _advance_diag_cycles advances log the
        full breakdown (focus result, method, before-hash, after-hash). After
        that, only outcomes (OK / NO EFFECT / changed method) are logged.
        """
        if not self.enabled:
            return
        if not PYGETWINDOW_AVAILABLE or not self.vn_window_title:
            # Can't verify without window targeting — fall back to blind press
            try:
                self.input_controller.advance()
            except Exception as e:
                print(f"   [Autopilot] Input failure \u2014 disabling: {e}")
                self.enabled = False
                self.is_running = False
                self.is_paused = True
                self.pause_reason = "INPUT_ERROR"
                if self.on_failsafe:
                    self.on_failsafe("INPUT_ERROR")
            return

        self._advance_attempts += 1
        diag = self._advance_attempts <= self._advance_diag_cycles

        # ── Step 1: Before-frame ──────────────────────────────────────────────
        before_frame = self._grab_frame_sync()
        if before_frame is None:
            print(
                "   [Autopilot] Advance: VN window not found before keystroke \u2014 "
                "skipping (window recovery will handle on next loop iteration)."
            )
            return
        before_hash = self._quick_hash(before_frame)

        # ── Step 2: Decide method order ───────────────────────────────────────
        # Try the known-working method first if we have one, otherwise the
        # configured advance_key. Then cycle through the rest.
        preferred = self._working_advance_method or self.input_controller.advance_key
        method_order = [preferred] + [
            m for m in self.input_controller.METHODS_ORDER if m != preferred
        ]

        # ── Step 3: Try each method until one produces a screen change ────────
        # A method counts as "working" only if it produces a NOVEL after-hash
        # (one not in our recent-advance history). A method that flips the
        # screen to a previously-seen state is toggling HUD/menu, not advancing.
        #
        # Each method gets up to 2 quick attempts before we switch — most
        # NO-EFFECT cases are just a dropped synthetic keystroke, not a bad
        # method, so a same-method retry recovers in ~0.4s instead of ~1s.
        for attempt_idx, method in enumerate(method_order):
            focused = self._focus_vn_window()
            if diag or attempt_idx > 0:
                print(
                    f"   [Autopilot] Advance attempt {attempt_idx + 1}/{len(method_order)}: "
                    f"method={method}, focused={focused}"
                )

            # Per-method: try up to twice (first try, then one quick retry on no-op).
            method_changed = False
            method_after_frame = None
            method_after_hash = None
            method_oscillated = False
            for sub_attempt in range(2):
                try:
                    if method == "click":
                        # Click at window center — many engines ignore clicks
                        # outside the content area. Falls back to wherever the
                        # cursor sits if we can't locate the window.
                        win = self._find_vn_window()
                        if win is not None and win.width > 0 and win.height > 0:
                            cx = win.left + win.width // 2
                            cy = win.top + int(win.height * 0.5)
                            self.input_controller.click_at(cx, cy)
                        else:
                            self.input_controller.advance(method="click")
                    else:
                        self.input_controller.advance(method=method)
                except Exception as e:
                    print(f"   [Autopilot] Input failure on method={method}: {e}")
                    break  # try next method

                # Verify quickly. 0.3s is enough for most engines to render
                # the new frame; if the engine is slower, the stabilization
                # loop in _handle_dialogue / _post_advance_prep catches it.
                time.sleep(0.3)

                after_frame = self._grab_frame_sync()
                if after_frame is None:
                    print("   [Autopilot] Advance: window lost during verify \u2014 deferring.")
                    return
                after_hash = self._quick_hash(after_frame)

                changed = (after_hash != before_hash)
                oscillating = changed and (after_hash in self._recent_advance_hashes)
                method_after_frame = after_frame
                method_after_hash = after_hash
                method_oscillated = oscillating

                if changed:
                    method_changed = True
                    break
                # No change \u2014 give the input one fast retry before moving on.
                if sub_attempt == 0:
                    # Re-focus in case foreground was momentarily stolen.
                    self._focus_vn_window()

            after_frame = method_after_frame
            after_hash = method_after_hash
            changed = method_changed
            oscillating = method_oscillated

            if diag:
                print(
                    f"   [Autopilot] Advance verify: before={before_hash[:8]} "
                    f"after={after_hash[:8] if after_hash else '????????'} changed={changed} "
                    f"oscillating={oscillating}"
                )

            if changed and not oscillating:
                if attempt_idx == 0 and not diag:
                    pass
                else:
                    print(
                        f"   [Autopilot] Advance OK via {method} "
                        f"(novel screen) [attempt {attempt_idx + 1}]"
                    )

                if self._working_advance_method != method:
                    self._working_advance_method = method
                    print(f"   [Autopilot] Working advance method: {method}")

                self._recent_advance_hashes.append(after_hash)
                self._last_verified_post_advance_frame = after_frame
                return

            # Either no change OR oscillation back to a known state.
            if oscillating:
                print(
                    f"   [Autopilot] Advance via {method} OSCILLATED back to a "
                    f"recent screen (hash={after_hash[:8]}) \u2014 looks like a HUD/menu "
                    f"toggle. Trying next method."
                )
                # If our locked-in method is the one that just oscillated, unlock
                # it so the eventual winner becomes the new working method.
                if self._working_advance_method == method:
                    print(
                        f"   [Autopilot] Unlocking previous working method "
                        f"({method}) \u2014 it no longer advances the story."
                    )
                    self._working_advance_method = None
            else:
                print(
                    f"   [Autopilot] Advance had NO EFFECT via {method} "
                    f"(screen identical, before={before_hash[:8]})"
                )

            # Update before_hash to the latest after so subsequent method tries
            # compare against the most recent capture
            before_hash = after_hash
            before_frame = after_frame

        # ── Step 4: Exhausted all methods ────────────────────────────────────
        print(
            "   [Autopilot] Advance: ALL methods failed to change the screen "
            f"({', '.join(method_order)}). Triggering STUCK failsafe."
        )
        # Failsafe is async — schedule it on the running loop
        try:
            asyncio.ensure_future(self._trigger_failsafe("STUCK"))
        except Exception:
            # If we can't schedule, at least flip the flags so the loop exits
            self.is_running = False
            self.is_paused = True
            self.pause_reason = "STUCK"
            self._hard_failsafed = True
            if self.on_failsafe:
                self.on_failsafe("STUCK")

    def _find_vn_window(self):
        """Find the VN window by title substring (case-insensitive, whitespace-trimmed).
        Returns a pygetwindow window or None."""
        if not PYGETWINDOW_AVAILABLE or not self.vn_window_title:
            return None
        try:
            needle = self.vn_window_title.strip().lower()
            for w in _pgw.getAllWindows():
                if needle in (w.title or "").lower():
                    return w
        except Exception:
            pass
        return None

    def _grab_frame_sync(self):
        """Capture the VN window region (by bounding rect). Never uses full-screen.
        _async_start() ensures vn_window_title is always set before _loop() runs.

        Returns a PIL Image on success.
        Returns None if the window cannot be found or is minimised — the loop
        treats None as a recoverable "window temporarily lost" and waits/retries
        via _wait_for_window_recovery() rather than crashing.
        Raises RuntimeError ONLY for missing-library errors (PIL / pygetwindow)
        which cannot be recovered from at runtime.
        Synchronous — safe to call from run_in_executor.
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available for screenshot capture")
        if not PYGETWINDOW_AVAILABLE:
            raise RuntimeError(
                "pygetwindow not available — install it: pip install pygetwindow pywin32"
            )
        if not self.vn_window_title:
            # No title set — should not reach here after _async_start() validates,
            # but return None gracefully rather than crashing.
            return None
        win = self._find_vn_window()
        if win is None:
            return None  # window temporarily gone — caller enters recovery wait
        if win.width <= 0 or win.height <= 0:
            return None  # window minimised — same recovery path
        bbox = (win.left, win.top, win.left + win.width, win.top + win.height)
        try:
            return ImageGrab.grab(bbox=bbox)
        except Exception as e:
            print(f"   [Autopilot] Frame capture error: {e}")
            return None

    def _focus_vn_window(self) -> bool:
        """Bring the VN window truly to the FOREGROUND before sending input.

        Some games (Steins;Gate among them) ignore SendInput keystrokes unless
        the window is in the foreground — merely having focus is not enough.
        This restores from minimised, activates, and gives the OS a frame to
        complete the switch.

        Returns True if the window was found AND we successfully called
        activate() (or it was already foreground). Returns False if the window
        couldn't be reached, in which case the caller should treat input as
        unreliable.
        """
        if not self.vn_window_title or not PYGETWINDOW_AVAILABLE:
            return False
        win = self._find_vn_window()
        if win is None:
            return False
        try:
            # Restore from minimised so activate() has a window to bring forward
            if getattr(win, "isMinimized", False):
                try:
                    win.restore()
                    time.sleep(0.10)
                except Exception:
                    pass
            win.activate()
            time.sleep(0.10)  # let OS complete the foreground switch
            return True
        except Exception as e:
            # pygetwindow.activate() can throw on already-foreground windows on
            # some Windows builds; that's actually success. Treat as success
            # unless the error string suggests a real failure.
            err = str(e).lower()
            if "could not activate" in err or "no window" in err:
                return False
            return True

    # ── Disk-space guard ───────────────────────────────────────────────────────

    # Minimum free space required before writing to ChromaDB / narrative memory.
    # Below this threshold, writes are skipped with a log — avoids DB corruption
    # from a full disk during long playthroughs.
    _DISK_WARN_BYTES: int = 500 * 1024 * 1024   # 500 MB

    def _check_disk_space(self) -> bool:
        """Return True if there is enough free space on the drive for memory writes.

        Uses the directory of this script file (same drive as memory_db/).
        On any error reading disk info, returns True (non-blocking — prefer a bad
        write that logs cleanly over silently suppressing all writes).
        """
        try:
            path = os.path.dirname(os.path.abspath(__file__))
            usage = shutil.disk_usage(path)
            if usage.free < self._DISK_WARN_BYTES:
                mb_free = usage.free // (1024 * 1024)
                print(
                    f"   [Autopilot] ⚠ LOW DISK SPACE — {mb_free} MB free "
                    f"(threshold {self._DISK_WARN_BYTES // (1024 * 1024)} MB). "
                    "Skipping memory write to avoid DB corruption."
                )
                return False
            return True
        except Exception as _disk_err:
            print(f"   [Autopilot] Disk-space check failed ({_disk_err}) — allowing write.")
            return True

    @staticmethod
    def _is_system_ui_text(text: str) -> bool:
        """Return True if the OCR text looks like Windows system UI rather than VN content.
        Guards against accidentally capturing the taskbar, system tray, or other overlays."""
        if not text:
            return False
        patterns = [
            r'\d{1,2}:\d{2}\s*[AP]M',                         # clock: "9:12 PM"
            r'\d+\s*°[CF]',                                    # temperature: "81°F"
            r'\b(Control Center|Cortana|Action Center)\b',
            r'\b(Clear|Partly Cloudy|Mostly Cloudy|Sunny|Overcast|Rainy)\b.*\d+°',
        ]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False
