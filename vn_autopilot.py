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
import hashlib
import re
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
PAUSE_SENTENCE_END      = 0.35   # text ends with '.'  — clear stop between thoughts
PAUSE_QUESTION_EXCLAIM  = 0.55   # text ends with '?'/'!' — emotional beat / rising tone
PAUSE_CLAUSE_FLOW       = 0.10   # comma or no terminal punct — flow right through
PAUSE_SCENE_CHANGE      = 0.90   # art region just changed — let the new scene land
PAUSE_INTENSITY_BONUS   = 0.40   # added on top for INTENSE / CLIMACTIC moments
PAUSE_AFTERMATH_BONUS   = 0.20   # extra somber pacing in AFTERMATH
PAUSE_REACTION_GAP      = 0.15   # minimal gap before a reaction (same-breath feel)

# ── Pacing helper ──────────────────────────────────────────────────────────────

def compute_read_delay(
    text_length: int,
    base: float = 2.5,
    per_char: float = 0.025,
    max_delay: float = 8.0,
) -> float:
    """Time (seconds) to leave a text box visible so viewers can read along."""
    return min(base + (text_length * per_char), max_delay)


# ── Input Controller ───────────────────────────────────────────────────────────

class VNInputController:
    """Sends keyboard/mouse input to the focused VN window via pydirectinput."""

    def __init__(self, advance_key: str = "space", pre_input_delay: float = 0.1):
        self.advance_key = advance_key
        self.pre_input_delay = pre_input_delay

    def advance(self):
        """Press the configured advance key. Raises RuntimeError if pydirectinput unavailable."""
        if not PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError("pydirectinput not installed. Run: pip install pydirectinput")
        time.sleep(self.pre_input_delay)
        if self.advance_key == "click":
            pydirectinput.click()
        else:
            pydirectinput.press(self.advance_key)

    def set_advance_key(self, key: str):
        self.advance_key = key


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
        "Use this whenever ANY readable story text is visible on screen, even if the "
        "art is sparse, minimalist, abstract, or mostly blank/white. DIALOGUE is the "
        "default — always bias toward DIALOGUE when text is present.\n"
        "CHOICE - a decision menu with multiple selectable options the player must pick from.\n"
        "SAVE_PROMPT - a save/load menu, settings menu, or system dialog.\n"
        "TRANSITION - a title card, chapter break, or scene fade/black screen with "
        "NO readable text at all.\n"
        "UNKNOWN - ONLY use this when there is genuinely no readable text AND the screen "
        "is not clearly a transition. Never use UNKNOWN if any text is visible.\n\n"
        "Respond with ONLY the single category word, nothing else."
    )

    FAILSAFE_LINES = {
        "CHOICE": (
            "Okay, this is a choice. Jonny, I'm handing this one to you — "
            "I don't want to accidentally doom us to a bad ending."
        ),
        "SAVE_PROMPT": (
            "Looks like a menu popped up. Jonny, you handle the buttons, "
            "I'll just sit here looking pretty."
        ),
        "UNKNOWN": "I'm not sure what I'm looking at here. Jonny, take a look?",
        "INPUT_ERROR": "Something went wrong with my input system. Jonny, you'll need to take over.",
        "ERROR": "I ran into an error. Jonny, I'm pausing — come check on me?",
    }

    def __init__(self, ai_core, vision_client=None):
        """
        ai_core:        AI_Core instance (for Anthropic/Claude access)
        vision_client:  AsyncOpenAI instance (for screen classification + transcription)
        """
        self.ai_core = ai_core
        self.vision_client = vision_client
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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Start or restart the autopilot loop. Safe to call multiple times."""
        if not self.enabled:
            return
        if self._task and not self._task.done():
            return  # already running
        self.is_running = True
        self.is_paused = False
        self.pause_reason = ""
        self._task = asyncio.ensure_future(self._loop())
        print("   [Autopilot] Started.")

    def stop(self):
        """Stop the autopilot cleanly. Does NOT speak anything."""
        self.enabled = False
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        print("   [Autopilot] Stopped.")

    def resume_after_failsafe(self):
        """Resume after Jonny has handled a non-dialogue screen manually."""
        if not self.enabled:
            return
        self.is_paused = False
        self.pause_reason = ""
        self.is_running = True
        if not self._task or self._task.done():
            self._task = asyncio.ensure_future(self._loop())
        print("   [Autopilot] Resumed after failsafe.")

    # ── Phase 2: Soft-pause / chat notification (Systems 2, 6b) ───────────────

    def soft_pause(self):
        """Gentle pause while Jonny talks — does NOT require Resume button.
        The loop stays alive; advancement is just suspended."""
        if not self.soft_paused:
            self.soft_paused = True
            print("   [Autopilot] Soft-paused (Jonny is talking).")

    def soft_resume(self):
        """Release soft-pause after Jonny's conversation ends."""
        if self.soft_paused:
            self.soft_paused = False
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

        When a pipeline task pre-prepared the next box during TTS, that data is
        consumed here instead of taking a fresh screenshot + classify + OCR.
        """
        print("   [Autopilot] Loop running.")
        while self.enabled and not self.is_paused:
            # System 6b: soft-pause while Jonny is talking — hold without exiting
            while self.soft_paused and self.enabled and not self.is_paused:
                await asyncio.sleep(0.25)
            if not self.enabled or self.is_paused:
                break

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
                    screen_type = await self._classify_screen(frame)

                print(f"   [Autopilot] Screen: {screen_type}")

                if screen_type == "DIALOGUE":
                    await self._handle_dialogue(frame, prepared)

                elif screen_type == "TRANSITION":
                    # Pipeline may have already advanced to get here; we now advance
                    # *through* the transition (N+1 → N+2) and wait for it to clear.
                    await asyncio.sleep(0.8)
                    self._safe_advance()
                    await asyncio.sleep(1.0)

                else:
                    # CHOICE / SAVE_PROMPT / UNKNOWN.
                    if screen_type == "UNKNOWN":
                        await asyncio.sleep(1.0)
                        retry_frame = await asyncio.get_event_loop().run_in_executor(
                            None, self._grab_frame_sync
                        )
                        screen_type = await self._classify_screen(retry_frame)
                        print(f"   [Autopilot] UNKNOWN retry → {screen_type}")
                        if screen_type == "DIALOGUE":
                            await self._handle_dialogue(retry_frame, None)
                            continue
                        elif screen_type == "TRANSITION":
                            await asyncio.sleep(0.8)
                            self._safe_advance()
                            await asyncio.sleep(1.0)
                            continue
                    await self._trigger_failsafe(screen_type)
                    return

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"   [Autopilot] Unhandled error: {e}")
                traceback.print_exc()
                await self._trigger_failsafe("ERROR")
                return

        self.is_running = False
        print("   [Autopilot] Loop exited.")

    # ── Screen classification ──────────────────────────────────────────────────

    async def _classify_screen(self, frame) -> str:
        """Returns one of: DIALOGUE / CHOICE / SAVE_PROMPT / TRANSITION / UNKNOWN."""
        if not self.vision_client:
            return "UNKNOWN"
        try:
            buf = BytesIO()
            frame.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = await self.vision_client.chat.completions.create(
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
            )
            raw = resp.choices[0].message.content.strip().upper()
            valid = {"DIALOGUE", "CHOICE", "SAVE_PROMPT", "TRANSITION", "UNKNOWN"}
            return raw if raw in valid else "UNKNOWN"
        except Exception as e:
            print(f"   [Autopilot] Classify error: {e}")
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

    async def _handle_dialogue(self, frame, prepared: dict | None = None):
        """Read VN text aloud verbatim, then optionally add a rare brief reaction.

        Correct read-then-advance order:
          1. Fast path: _post_advance_prep pre-computed this box's data → skip stabilize/OCR.
          2. Normal path: stabilize (0.15s × 2) → local OCR → clean → diff.
          3. Phase 2 analysis: attachment + weight + intensity (side-effects here only).
          4. SSML prosody: subtle rate/pitch variation by punctuation + intensity.
          5. Read aloud — wait for TTS to complete fully.
          6. Scene-art awareness background task (only on art hash change).
          7. Optional theory resolution / rare reaction (0.15s gap).
          8. Content-based pause (0.10–0.95s).
          9. Advance to next box (only after reading + pause complete).
         10. _post_advance_prep: capture+OCR next box to hide render latency → _prepared_next.
        """
        # ── Fast path: use pipeline-prepared data ────────────────────────────
        if prepared and prepared.get("box_data"):
            box          = prepared["box_data"]
            new_text     = box["new_text"]
            normalized   = box["normalized"]
            text         = box["text"]
            stable_frame = box["stable_frame"]
            weight       = box["weight"]
            self._last_box_text = normalized
        else:
            # ── Normal path: stabilize → OCR → diff ──────────────────────────
            # CHANGE 5: 0.15s × 2 = 0.30s max (down from 0.6s)
            prev_hash    = self._quick_hash(frame)
            stable_frame = frame
            for _ in range(2):
                await asyncio.sleep(0.15)
                curr_frame = await asyncio.get_event_loop().run_in_executor(
                    None, self._grab_frame_sync
                )
                curr_hash = self._quick_hash(curr_frame)
                if curr_hash == prev_hash:
                    stable_frame = curr_frame
                    break
                prev_hash    = curr_hash
                stable_frame = curr_frame

            # OCR: local first, cloud fallback if empty
            text = self._ocr_text_region(stable_frame)
            if not text or len(text.strip()) < 5:
                text = await self._transcribe_frame_cloud(stable_frame)

            text = self._clean_text((text or "").strip())
            if not text or text.upper() == "NO TEXT VISIBLE":
                await asyncio.sleep(1.5)
                self._safe_advance()
                return

            # System-UI guard: if the capture looks like taskbar/desktop, skip
            if self._is_system_ui_text(text):
                print(f"   [Autopilot] System UI detected — wrong window? Skipping.")
                await asyncio.sleep(1.5)
                return

            normalized = " ".join(text.split())

            # Dedup: identical re-capture → skip, advance quietly
            if normalized == self._last_box_text:
                await asyncio.sleep(0.5)
                self._safe_advance()
                return

            # Append detection: VN accumulated text on the same box
            if self._last_box_text and normalized.startswith(self._last_box_text):
                raw_suffix = normalized[len(self._last_box_text):]
                if raw_suffix and not raw_suffix[0].isspace():
                    space_idx = raw_suffix.find(' ')
                    raw_suffix = raw_suffix[space_idx:] if space_idx >= 0 else ""
                new_text = raw_suffix.strip()
                if len(new_text) < 4:
                    self._last_box_text = normalized
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
                self._last_box_text = normalized
                await asyncio.sleep(0.5)
                self._safe_advance()
                return

            self._last_box_text = normalized

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

        self.vn_recent_text_buffer.append(text)
        self.vn_boxes_since_summary += 1

        # Narrative summary: background task — never blocks reading
        if self.vn_boxes_since_summary >= 15 and len(self.vn_recent_text_buffer) >= 5:
            asyncio.ensure_future(self._update_narrative_summary())

        # Art-change check (before _maybe_update_scene_art overwrites the hash)
        art_changed = self._art_hash(stable_frame) != self._last_art_hash

        # ── Read aloud — wait for TTS to finish before advancing ─────────────
        if self.on_speak_vn:
            ssml_inner = self._build_vn_ssml_inner(new_text, intensity)
            await self.on_speak_vn(new_text, ssml_inner)
        elif self.on_speak:
            await self.on_speak(new_text)

        # Scene-art awareness: background task, only acts on art hash change
        asyncio.ensure_future(self._maybe_update_scene_art(stable_frame))

        # ── Theory resolution ─────────────────────────────────────────────────
        theory_reaction = await self._check_theory_resolutions(new_text)
        spoke_extra = False
        if theory_reaction and self.on_speak:
            await asyncio.sleep(PAUSE_REACTION_GAP)
            await self.on_speak(theory_reaction)
            self._boxes_since_reaction = 0
            spoke_extra = True

        # ── Optional rare reaction (local gate — no per-box latency) ──────────
        if not spoke_extra and self._should_consider_reaction(weight, intensity):
            reaction = await self._decide_reaction(
                new_text, intensity=intensity, narrative_weight=weight
            )
            if reaction and reaction.upper() != "SILENT":
                if self.on_speak:
                    await asyncio.sleep(PAUSE_REACTION_GAP)
                    await self.on_speak(reaction)
                self._boxes_since_reaction = 0
                self._boxes_since_solo_aside = 0
                spoke_extra = True

        # ── Solo aside when dead chat + calm stretch ──────────────────────────
        if (
            not spoke_extra
            and intensity == INTENSITY_CALM
            and self.chat_dead_min >= 4.0
            and self._boxes_since_solo_aside >= 7
            and self._boxes_since_reaction >= 4
        ):
            aside = await self._maybe_form_theory()
            if aside and self.on_speak:
                await asyncio.sleep(PAUSE_REACTION_GAP)
                await self.on_speak(aside)
                self._boxes_since_solo_aside = 0
                self._boxes_since_reaction = 0

        # ── Content-based pause (replaces fixed 0.8s) ────────────────────────
        await asyncio.sleep(self._content_pause(new_text, intensity, art_changed))

        # ── Advance only after reading is complete ────────────────────────────
        self._safe_advance()

        # ── Pre-prep next box to hide render+OCR latency ─────────────────────
        if self.enabled and not self.soft_paused:
            try:
                result = await self._post_advance_prep()
                if result:
                    self._prepared_next = result
            except Exception as e:
                print(f"   [Autopilot] Post-advance prep error: {e}")

    async def _post_advance_prep(self) -> dict | None:
        """After advancing to the next box, pre-capture and OCR it to hide render latency.

        Called immediately after _safe_advance() — the VN has already advanced.
        Hides the render+stabilize+OCR gap between boxes by running it before the
        next _loop iteration needs it.

        Returns dict with {screen_type, frame, box_data} or None if aborted.
        box_data is None for non-dialogue or dedup screens; _loop falls back to
        the normal OCR path.
        """
        try:
            await asyncio.sleep(0.2)    # let VN render the next frame

            if not self.enabled:
                return None

            frame = await asyncio.get_event_loop().run_in_executor(
                None, self._grab_frame_sync
            )
            screen_type = await self._classify_screen(frame)

            if screen_type != "DIALOGUE":
                return {"screen_type": screen_type, "frame": frame, "box_data": None}

            # Stabilize: 0.15s × 2
            prev_hash    = self._quick_hash(frame)
            stable_frame = frame
            for _ in range(2):
                await asyncio.sleep(0.15)
                curr_frame = await asyncio.get_event_loop().run_in_executor(
                    None, self._grab_frame_sync
                )
                curr_hash = self._quick_hash(curr_frame)
                if curr_hash == prev_hash:
                    stable_frame = curr_frame
                    break
                prev_hash    = curr_hash
                stable_frame = curr_frame

            # OCR: local first, cloud fallback
            text = self._ocr_text_region(stable_frame)
            if not text or len(text.strip()) < 5:
                text = await self._transcribe_frame_cloud(stable_frame)
            text = self._clean_text((text or "").strip())
            if not text or text.upper() == "NO TEXT VISIBLE":
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # System-UI guard: wrong window captured → abort, don't cache
            if self._is_system_ui_text(text):
                print(f"   [Autopilot] Pipeline: system UI detected — wrong window?")
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            normalized = " ".join(text.split())

            # Dedup check (read-only: _last_box_text not updated in the pipeline)
            if normalized == self._last_box_text:
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # Append detection
            if self._last_box_text and normalized.startswith(self._last_box_text):
                raw_suffix = normalized[len(self._last_box_text):]
                if raw_suffix and not raw_suffix[0].isspace():
                    space_idx = raw_suffix.find(' ')
                    raw_suffix = raw_suffix[space_idx:] if space_idx >= 0 else ""
                new_text = raw_suffix.strip()
                if len(new_text) < 4:
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
                return {"screen_type": "DIALOGUE", "frame": frame, "box_data": None}

            # Weight only — side-effect methods (_estimate_scene_intensity,
            # _update_character_attachment) stay in _handle_dialogue
            weight = self._estimate_narrative_weight(new_text)

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
            return None

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
            resp = await self.vision_client.chat.completions.create(
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
            )
            self._scene_art_description = resp.choices[0].message.content.strip()
            print(f"   [Autopilot] Scene: {self._scene_art_description[:70]}")
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

            prompt = (
                f"You are Kira, reading a visual novel aloud on stream — you ARE the narrator.\n\n"
                f"You just read this aloud:\n\"{text}\"\n"
                f"{narrative_block}{scene_art_note}{trajectory_note}{attachment_instruction}"
                f"{theory_instruction}{investment_note}{solo_instruction}\n\n"
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
            )
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=160,
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
        """Build/update rolling ~150-word plot summary from accumulated text boxes."""
        if not self.ai_core.anthropic_client:
            return
        try:
            accumulated = "\n---\n".join(self.vn_recent_text_buffer[-20:])
            prev = self.vn_narrative_summary or "No previous summary — this is the start."
            prompt = (
                f"You are maintaining a running ~150-word story summary for a visual novel playthrough.\n\n"
                f"Previous summary:\n{prev}\n\n"
                f"New dialogue/narration (most recent boxes):\n{accumulated}\n\n"
                f"Write an updated summary in ~150 words. Track character names, events, and emotional beats. "
                f"Be factual and concise. No commentary or editorializing."
            )
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            self.vn_narrative_summary = resp.content[0].text.strip()
            # Keep a small tail for continuity overlap on next summary cycle
            self.vn_recent_text_buffer = self.vn_recent_text_buffer[-5:]
            self.vn_boxes_since_summary = 0
            # System 4: update trajectory now that we have fresh narrative context
            self._update_emotional_trajectory()
            print(f"   [Autopilot] Narrative summary updated ({len(self.vn_narrative_summary)} chars).")
        except Exception as e:
            print(f"   [Autopilot] Narrative update error: {e}")

    # ── Failsafe ───────────────────────────────────────────────────────────────

    async def _trigger_failsafe(self, screen_type: str):
        """Stop the loop, speak a handoff line, notify the dashboard."""
        self.is_running = False
        self.is_paused = True
        self.pause_reason = screen_type
        print(f"   [Autopilot] Failsafe: {screen_type}")

        line = self.FAILSAFE_LINES.get(screen_type, self.FAILSAFE_LINES["UNKNOWN"])
        if line and self.on_speak:
            try:
                await self.on_speak(line)
            except Exception as e:
                print(f"   [Autopilot] Could not speak failsafe line: {e}")

        if self.on_failsafe:
            self.on_failsafe(screen_type)

    # ── Input helper ───────────────────────────────────────────────────────────

    def _safe_advance(self):
        """Press the advance key with full error handling. Disables autopilot on failure.

        If a window title is configured but the window cannot be found, the advance
        is skipped rather than sending a blind keystroke to whatever is focused.
        """
        if not self.enabled:
            return
        # Safety: if we're targeting a specific window, confirm it's findable
        if self.vn_window_title and PYGETWINDOW_AVAILABLE:
            if self._find_vn_window() is None:
                print(f"   [Autopilot] VN window not found — skipping advance keystroke.")
                return
        try:
            self._focus_vn_window()
            self.input_controller.advance()
        except Exception as e:
            print(f"   [Autopilot] Input failure — disabling: {e}")
            self.enabled = False
            self.is_running = False
            self.is_paused = True
            self.pause_reason = "INPUT_ERROR"
            if self.on_failsafe:
                self.on_failsafe("INPUT_ERROR")

    def _find_vn_window(self):
        """Find the VN window by title substring. Returns a pygetwindow window or None."""
        if not PYGETWINDOW_AVAILABLE or not self.vn_window_title:
            return None
        try:
            title_lower = self.vn_window_title.lower()
            all_wins = _pgw.getAllWindows()
            for w in all_wins:
                if title_lower in (w.title or "").lower():
                    return w
        except Exception:
            pass
        return None

    def _grab_frame_sync(self):
        """Capture the VN window region (if title configured), else full screen.
        Synchronous — safe to call from run_in_executor."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available for screenshot capture")
        if self.vn_window_title and PYGETWINDOW_AVAILABLE:
            win = self._find_vn_window()
            if win is not None:
                try:
                    bbox = (win.left, win.top,
                            win.left + win.width, win.top + win.height)
                    if win.width > 0 and win.height > 0:
                        return ImageGrab.grab(bbox=bbox)
                    # Window minimised or zero-size — fall through to full screen
                except Exception:
                    pass
            else:
                # Window title set but window not found — warn once per call
                print(f"   [Autopilot] VN window '{self.vn_window_title}' not found — "
                      f"is the game open?")
        return ImageGrab.grab()

    def _focus_vn_window(self):
        """Bring the VN window to the foreground before sending keystrokes.
        No-op when no window title is configured or pygetwindow unavailable."""
        if not self.vn_window_title or not PYGETWINDOW_AVAILABLE:
            return
        win = self._find_vn_window()
        if win is None:
            return
        try:
            win.activate()
            time.sleep(0.05)   # brief settle — focus needs a frame
        except Exception:
            pass

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
