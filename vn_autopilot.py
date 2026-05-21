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


# ── Phase 2: Scene intensity states (System 1) ────────────────────────────────
INTENSITY_CALM      = "calm"       # low-stakes connective tissue; riff/theorize freely
INTENSITY_BUILDING  = "building"   # tension rising; lean in, anticipate
INTENSITY_INTENSE   = "intense"    # things are happening; react sparingly and sharply
INTENSITY_CLIMACTIC = "climactic"  # the moment; silence is usually the correct response
INTENSITY_AFTERMATH = "aftermath"  # just after something big; process quietly

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
        """Event-driven main loop: screenshot → classify → handle → advance → repeat."""
        print("   [Autopilot] Loop running.")
        while self.enabled and not self.is_paused:
            # System 6b: soft-pause while Jonny is talking — hold without exiting
            while self.soft_paused and self.enabled and not self.is_paused:
                await asyncio.sleep(0.25)
            if not self.enabled or self.is_paused:
                break

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
        """Transcribe text, run all Phase 2 analysis, optionally react, pace, advance."""
        text = await self._transcribe_frame(frame)
        text = (text or "").strip()

        if text and text.upper() != "NO TEXT VISIBLE":
            self.total_boxes_read += 1
            self._boxes_since_reaction += 1
            self._boxes_since_theory_check += 1
            self._boxes_since_solo_aside += 1

            # Store in narrative buffer
            self.vn_recent_text_buffer.append(text)
            self.vn_boxes_since_summary += 1

            # Update rolling story summary every ~15 boxes
            if self.vn_boxes_since_summary >= 15 and len(self.vn_recent_text_buffer) >= 5:
                await self._update_narrative_summary()

            # System 4: update character attachment from this box
            self._update_character_attachment(text)

            # System 5: estimate narrative weight (text-heuristic, no LLM)
            weight = self._estimate_narrative_weight(text)

            # System 1: derive scene intensity from weight; update running state
            intensity = self._estimate_scene_intensity(weight)
            self.scene_intensity = intensity

            # System 3: check if this box resolves an open theory (before primary reaction)
            theory_reaction = await self._check_theory_resolutions(text)
            if theory_reaction and self.on_speak:
                await self.on_speak(theory_reaction)
                self._boxes_since_reaction = 0
                await asyncio.sleep(0.8)

            # Primary reaction decision with full Phase 2 context
            reaction = await self._decide_reaction(text, intensity=intensity, narrative_weight=weight)

            spoke = bool(theory_reaction)  # already spoken above
            if reaction and reaction.upper() != "SILENT":
                if self.on_speak:
                    await self.on_speak(reaction)
                self._boxes_since_reaction = 0
                self._boxes_since_solo_aside = 0
                spoke = True
                await asyncio.sleep(0.8)

            # System 2 + 3: if silent and in a calm dead-chat stretch, generate
            # an unprompted solo aside (theory, address absent chat, internal monologue)
            if (
                not spoke
                and intensity == INTENSITY_CALM
                and self.chat_dead_min >= 4.0
                and self._boxes_since_solo_aside >= 7
                and self._boxes_since_reaction >= 4
            ):
                aside = await self._maybe_form_theory()
                if aside and self.on_speak:
                    await self.on_speak(aside)
                    self._boxes_since_solo_aside = 0
                    self._boxes_since_reaction = 0

            # Pacing delay — longer for emotionally heavy moments
            delay = compute_read_delay(len(text), self.pacing_base, self.pacing_per_char, self.pacing_max)
            if intensity in (INTENSITY_CLIMACTIC, INTENSITY_INTENSE, INTENSITY_AFTERMATH):
                delay = max(delay, 4.0)
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
                    f"\n\nCHAT IS DEAD — no messages for {dead:.0f} minutes. You're playing "
                    f"completely solo right now. That's fine — comfortable on an empty stage. "
                    f"Internal monologue, talking to absent viewers ('okay chat, if anyone's "
                    f"lurking out there—'), self-directed tangents. Don't go silent just because "
                    f"nobody's chatting. Carry the stream yourself."
                )
            elif dead >= 3.0:
                solo_instruction = (
                    "\n\nChat is quiet (a few minutes of silence). Lean a bit more into "
                    "internal monologue — talking to yourself is fine."
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

            prompt = (
                f"You are Kira, autonomously playing through a visual novel on stream.\n\n"
                f"You just read:\n\"{text}\"\n"
                f"{narrative_block}{trajectory_note}{attachment_instruction}"
                f"{theory_instruction}{investment_note}{solo_instruction}\n\n"
                f"{intensity_instruction}\n\n"
                f"If REACT: output a SHORT spoken reaction in Kira's voice. Calibrate the "
                f"energy to the intensity guidance above — low-stakes rambles for calm, "
                f"sharp/controlled for intense, quiet/heavy for aftermath. 1-3 sentences max.\n"
                f"If SILENT: output exactly \"SILENT\" and nothing else.\n\n"
                f"Just the reaction text or SILENT. No labels, no preamble."
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
                "SCENE INTENSITY: CALM — low-stakes connective content. This is good "
                "space-filling time. React more freely: ramble, theorize, make an aside, "
                "talk to absent chat. Energy should be relaxed and natural. "
                "If nothing in this text is interesting, say something interesting yourself."
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
