# vn_autopilot.py — Autonomous VN Mode (Phase 1: Dialogue-Only Autoplay)
#
# Architecture: event-driven loop. Kira controls the clock — she advances,
# screenshots, reads, optionally reacts, waits a pacing beat, then advances again.
# One screenshot per text box. NO fixed-interval screenshotting.
#
# Phase 1 safety contract:
#   - DIALOGUE screens → advance autonomously
#   - TRANSITION screens → silently advance (short wait)
#   - Everything else (CHOICE, SAVE_PROMPT, UNKNOWN) → STOP, alert Jonny, wait for Resume
#
# IMPORTANT: The VN window must be the focused/active window for inputs to register.

import asyncio
import base64
import time
import traceback
from io import BytesIO

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
    import pydirectinput
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    PYDIRECTINPUT_AVAILABLE = False
    print("   [Autopilot] pydirectinput not installed — run: pip install pydirectinput")


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
        "DIALOGUE - normal story text/dialogue that advances by pressing a key. The default.\n"
        "CHOICE - a decision menu with multiple selectable options the player must pick from.\n"
        "SAVE_PROMPT - a save/load menu, settings menu, or system dialog.\n"
        "TRANSITION - a title card, chapter break, black screen, or scene transition with no text to read.\n"
        "UNKNOWN - anything you cannot confidently classify.\n\n"
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

        # ── Callbacks (wired by bot.py) ─────────────────────────────────────────
        # on_speak(text: str) -> awaitable  — speak a reaction via TTS
        self.on_speak = None
        # on_failsafe(screen_type: str) -> None  — notify dashboard of failsafe
        self.on_failsafe = None

        # ── Internal ────────────────────────────────────────────────────────────
        self._task: asyncio.Task | None = None

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

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def _loop(self):
        """Event-driven main loop: screenshot → classify → handle → advance → repeat."""
        print("   [Autopilot] Loop running.")
        while self.enabled and not self.is_paused:
            try:
                # 1. Capture fresh screenshot
                if not PIL_AVAILABLE:
                    raise RuntimeError("Pillow not available for screenshot capture")
                frame = await asyncio.get_event_loop().run_in_executor(None, ImageGrab.grab)

                # 2. Classify screen type
                screen_type = await self._classify_screen(frame)
                print(f"   [Autopilot] Screen: {screen_type}")

                if screen_type == "DIALOGUE":
                    await self._handle_dialogue(frame)

                elif screen_type == "TRANSITION":
                    # Silently wait through transitions, then advance
                    await asyncio.sleep(1.5)
                    self._safe_advance()

                else:
                    # CHOICE / SAVE_PROMPT / UNKNOWN — trigger failsafe, exit loop
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

    async def _handle_dialogue(self, frame):
        """Transcribe text, optionally react, wait pacing beat, advance."""
        text = await self._transcribe_frame(frame)
        text = (text or "").strip()

        if text and text.upper() != "NO TEXT VISIBLE":
            # Store in narrative buffer
            self.vn_recent_text_buffer.append(text)
            self.vn_boxes_since_summary += 1

            # Update rolling story summary every ~15 boxes
            if self.vn_boxes_since_summary >= 15 and len(self.vn_recent_text_buffer) >= 5:
                await self._update_narrative_summary()

            # Decide whether to react
            reaction = await self._decide_reaction(text)

            if reaction and reaction.upper() != "SILENT":
                # Speak the reaction — TTS duration provides natural pacing
                if self.on_speak:
                    await self.on_speak(reaction)
                # Short beat after speech before advancing
                await asyncio.sleep(0.8)
                self._safe_advance()
            else:
                # Silent: leave box on screen long enough for viewers to read
                delay = compute_read_delay(
                    len(text), self.pacing_base, self.pacing_per_char, self.pacing_max
                )
                await asyncio.sleep(delay)
                self._safe_advance()
        else:
            # Unreadable / no text — advance after short wait
            await asyncio.sleep(1.5)
            self._safe_advance()

    async def _transcribe_frame(self, frame) -> str:
        """Extract raw text from screen using the existing TRANSCRIBE_PROMPT."""
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
            print(f"   [Autopilot] Transcribe error: {e}")
            return ""

    async def _decide_reaction(self, text: str) -> str:
        """
        Ask Claude whether to react. Returns reaction text (1-2 sentences) or "SILENT".
        Targets ~1-in-5 reaction rate — only genuinely funny/emotional/surprising boxes.
        """
        if not self.ai_core.anthropic_client:
            return "SILENT"
        try:
            narrative_block = (
                f"\n\nRECENT STORY CONTEXT:\n{self.vn_narrative_summary}"
                if self.vn_narrative_summary else ""
            )
            prompt = (
                f"You are Kira, autonomously playing through a visual novel on stream. "
                f"You just read this text box:\n\n\"{text}\"\n"
                f"{narrative_block}\n\n"
                f"You react to roughly 1 in 5 boxes — only the genuinely funny, emotional, "
                f"surprising, or plot-significant ones. Most mundane dialogue you advance silently.\n\n"
                f"Decide: react or stay silent?\n\n"
                f"If REACT: output a SHORT spoken reaction (1-2 sentences, in your voice — "
                f"witty, warm, deadpan). Just the reaction text, nothing else.\n"
                f"If SILENT: output exactly \"SILENT\" and nothing else.\n\n"
                f"Do not react just to react. Silence between reactions makes reactions land harder. "
                f"Be selective. When in doubt, output SILENT."
            )
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"   [Autopilot] Reaction decision error: {e}")
            return "SILENT"

    # ── Narrative memory ───────────────────────────────────────────────────────

    async def _update_narrative_summary(self):
        """Build/update rolling 150-word plot summary from accumulated text boxes."""
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
        """Press the advance key with full error handling. Disables autopilot on failure."""
        if not self.enabled:
            return
        try:
            self.input_controller.advance()
        except Exception as e:
            print(f"   [Autopilot] Input failure — disabling: {e}")
            self.enabled = False
            self.is_running = False
            self.is_paused = True
            self.pause_reason = "INPUT_ERROR"
            if self.on_failsafe:
                self.on_failsafe("INPUT_ERROR")
