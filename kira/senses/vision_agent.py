import base64
import time
import asyncio
import re
from collections import deque
from io import BytesIO
from PIL import ImageGrab
try:
    import pygetwindow as _pgw          # window-target capture (Pokémon vision-lock); already a VN-mode dep
    PYGETWINDOW_AVAILABLE = True
except Exception:
    _pgw = None
    PYGETWINDOW_AVAILABLE = False
from openai import AsyncOpenAI
from kira.config import (OPENAI_API_KEY, ENABLE_VISION, VISION_CALM_HEARTBEAT_SECONDS,
                         VISION_CAPTURE_DEDUP_ENABLED, VISION_CAPTURE_DEDUP_WINDOW_S,
                         TURBO_VISION_SLIDESHOW_ENABLED, TURBO_VISION_CAPTURE_INTERVAL_S,
                         TURBO_VISION_BUFFER_SIZE, TURBO_VISION_ANALYSIS_INTERVAL_S)

# Request-size bounds for the rolling scene summary. The summary feeds its own
# output back in as `previous` each cycle, so without a clamp an over-long model
# response compounds until the request exceeds the gateway's size limit (the
# 431 "Request headers are too large" failure). These keep the request small.
_SCENE_SUMMARY_MAX_CHARS = 800
_SCENE_DESC_MAX_CHARS = 1500


def _record_vision_usage(response) -> None:
    """Record gpt-4o-mini usage to cost_tracker. Best-effort; never raises."""
    try:
        from kira.brain.cost_tracker import cost_tracker as _ct
        u = getattr(response, "usage", None)
        if u:
            _ct.record(
                model="gpt-4o-mini",
                input_tokens=getattr(u, "prompt_tokens", 0),
                output_tokens=getattr(u, "completion_tokens", 0),
                purpose="vision",
            )
    except Exception:
        pass

class ContextBuffer:
    def __init__(self, maxlen=3):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, description):
        timestamp = time.strftime("%H:%M:%S")
        self.buffer.append(f"[{timestamp}] {description}")
        
    def get_context_string(self):
        if not self.buffer:
            return "No visual history yet."
        return "\n".join(self.buffer)

class UniversalVisionAgent:
    # General-purpose screen description (used when no specific activity is set)
    DESCRIBE_PROMPT = (
        "You are looking at Jonny's computer screen. "
        "Describe what is happening in 2 clear, specific sentences.\n\n"
        "If it is a Game: Describe the action, the environment, "
        "and the current vibe (intense, chill, chaotic).\n"
        "If it is Video/Media: Describe the scene, the people, "
        "or the topic being discussed.\n"
        "If it is Desktop/Code: Summarize what he is working on.\n\n"
        "Be factual and specific. Do not editorialize, add jokes, "
        "or write commentary about what Kira might say. Just describe what you see.\n\n"
        "HONESTY: If the frame is mid-transition, mostly blank, blurred, or you cannot "
        "reliably tell what is happening, START your reply with 'UNCERTAIN:' and briefly "
        "say what you CAN see. Do not invent characters, dialogue, or actions you cannot "
        "actually see. Better to say 'I can't quite tell' than to guess wrong."
    )

    # Visual Novel extraction prompt — returns structured parseable output
    VN_DESCRIBE_PROMPT = (
        "You are looking at a Visual Novel game screen. Extract the following information exactly:\n"
        "SPEAKER: [the name of the character speaking, or 'Narration' if no speaker label]\n"
        "DIALOGUE: [the exact dialogue or narration text visible on screen]\n"
        "CHOICES: [if a choice menu is visible, list each option numbered like '1. Option text'; otherwise write 'none']\n"
        "SCENE: [one sentence describing the scene mood or background]\n\n"
        "Be precise. Copy dialogue text exactly as shown. If the screen is a loading/transition screen, "
        "write DIALOGUE: [loading screen] and CHOICES: none."
    )

    # Media/watching prompt
    MEDIA_DESCRIBE_PROMPT = (
        "You are looking at something Jonny is watching (anime, movie, YouTube, etc.). "
        "Describe in 2 sentences: what is on screen, who is in it, and the emotional tone of the scene. "
        "Be specific enough that Kira could make a relevant comment or question.\n\n"
        "HONESTY: If the frame is a transition, blur, blank, or you cannot reliably tell what is "
        "happening, START your reply with 'UNCERTAIN:' and say only what you CAN see. Do not invent "
        "character names, dialogue, or plot details you cannot actually see on screen."
    )

    SCENE_SUMMARY_PROMPT = (
        "You maintain a brief running summary of what is happening on Jonny's screen "
        "for an AI companion named Kira who is watching with him.\n\n"
        "Previous summary:\n{previous}\n\n"
        "Newest observation:\n{newest}\n\n"
        "Write an updated summary in 2-3 sentences. Track narrative continuity \u2014 who is in the scene, "
        "what just happened, what the emotional tone is. Treat this like notes a friend would keep if "
        "they were watching alongside someone and wanted to remember the thread of the story. "
        "Do not editorialize. Do not add jokes. Just the running story state."
    )

    # Turbo Vision slideshow — multi-frame "what HAPPENED across these frames" prompt.
    # Ported from MediaWatch's SEQUENCE_ANALYSIS_PROMPT, reworded for FULL-SCREEN
    # always-on capture (not a target window). Same HONESTY RULES so the timeline
    # stays trustworthy (UNCERTAIN:/STATIC: prefixes, no invented names/dialogue).
    SLIDESHOW_ANALYSIS_PROMPT = (
        "You are watching Jonny's screen alongside him. The following frames are "
        "consecutive moments captured a few seconds apart from the same screen. "
        "Your job is to describe what HAPPENED across this sequence — focus on "
        "ACTIONS, SEQUENCE, CHANGE, and CAUSE-AND-EFFECT. Not static description.\n\n"
        "Tell the story of this stretch in 2-4 sentences:\n"
        "  - Who appears and what they do (use any visible character names from "
        "    subtitles or name labels — otherwise describe by appearance).\n"
        "  - What changes between frames (movement, scene cuts, who arrives/leaves, "
        "    objects appearing, expressions shifting).\n"
        "  - Any dialogue / on-screen text you can read, attributed if possible.\n"
        "  - The emotional beat of the moment if it's clear.\n\n"
        "HONESTY RULES (important):\n"
        "  - If the frames are mostly transitions, blur, or you cannot tell what is "
        "    happening with reasonable confidence, start your reply with "
        "    'UNCERTAIN:' and briefly say what you CAN see.\n"
        "  - If frames look essentially identical (still scene, talking head, "
        "    paused video, idle desktop), say 'STATIC: <brief>' instead of inventing motion.\n"
        "  - Do not invent character names you cannot read on screen.\n"
        "  - Do not invent dialogue. Only quote text you can actually see.\n\n"
        "Be concrete. This summary will be Kira's memory of what happened."
    )

    TRANSCRIBE_PROMPT = (
        "Your ONLY job is to transcribe text visible on the screen, character-for-character. "
        "DO NOT describe the scene. DO NOT summarize. DO NOT add commentary. DO NOT paraphrase. "
        "DO NOT mention what characters look like or what they are doing visually. "
        "ONLY transcribe written text that is actually visible on screen.\n\n"
        "Format rules:\n"
        "- If a character is speaking (dialogue with a name label), write: SPEAKER_NAME: \"exact dialogue verbatim\"\n"
        "- If it is narration or text-box prose with no speaker, write it on its own line\n"
        "- If there are menu options or choices, list each on its own line\n"
        "- If the screen has NO readable text (transition, animation, pure image), "
        "output ONLY the literal string: NO TEXT VISIBLE\n\n"
        "Begin transcription:"
    )

    def __init__(self):
        self.client = None
        self.api_key = OPENAI_API_KEY
        if self.api_key:
             self.client = AsyncOpenAI(api_key=self.api_key)
        else:
             print("   [Vision] Warning: OPENAI_API_KEY not found in config.")

        self.context_buffer = ContextBuffer(maxlen=3)
        self.last_description = "I'm just getting my bearings. One sec!"
        self.last_capture_time = 0
        # Vision capture dedup: stamp when an ON-DEMAND capture starts so a heartbeat
        # capture can skip if one is in flight (avoids a 2x concurrent gpt-4o-mini call).
        self._last_ondemand_capture_ts = 0.0
        # Most recent captured frame (downscaled PIL image), served by the
        # dashboard's /vision/thumbnail endpoint. Written on every capture in
        # capture_and_describe(); None until the first heartbeat fires.
        self.last_frame = None
        self.is_active = ENABLE_VISION   # heartbeat running (parked by reconciler during Media Watch)
        # master_enabled = the user's Vision intent (the ONE master switch).
        # Unlike is_active, this is NEVER parked by the reconciler — it only flips
        # when the user toggles Vision on/off. EVERY on-demand capture path checks
        # this so "Vision off = blind" holds everywhere, even while a heartbeat is
        # parked for Media Watch (where general sight should still answer).
        self.master_enabled = ENABLE_VISION
        self.quality_mode = "fast"       # 'fast' or 'high'
        self.activity_type = "general"   # Set by GameModeController / bot
        # POKÉMON VISION-LOCK (Phase 1A): when non-empty, EVERY grab is restricted to the window whose
        # title contains this substring (the game window) — she sees the game and NOTHING else (no
        # desktop/code leak). Empty "" = normal full-screen capture (default; zero change off-mode).
        # Set by control_server on pokemon_start, cleared on pokemon_stop.
        self.capture_window_title = ""

        # Shared Buffer (populated by dashboard to avoid double-capturing)
        self.shared_frame = None
        self.shared_frame_time = 0

        # Scene memory
        self.scene_summary: str = ""               # Rolling narrative summary
        self.previous_dialogue: str = ""           # For dialogue-change detection
        self.last_dialogue_change_time: float = 0  # Timestamp of last screen text change
        self.heartbeat_interval = VISION_CALM_HEARTBEAT_SECONDS  # calm general cadence; bot overrides to 10.0 in fast (game/media) mode

        # ── Turbo Vision slideshow (multi-frame "what happened" timeline) ──────
        # Only active while Turbo Vision is engaged AND the flag is on. A dedicated
        # full-screen capture loop fills _slideshow_frames; an analysis loop folds
        # the buffer into self.episode_log (the timeline Kira answers from). All
        # default-OFF / dormant until start_slideshow() is called by the bot.
        self._slideshow_running: bool = False
        self._slideshow_frames: deque = deque(maxlen=max(2, TURBO_VISION_BUFFER_SIZE))
        self.episode_log: deque = deque(maxlen=200)
        self._slideshow_capture_task = None
        self._slideshow_analysis_task = None
        self._slideshow_start_ts: float = 0.0
        self._slideshow_calls: int = 0
        self._slideshow_last_content_mid_ts: float = 0.0
        self._slideshow_cloud_timeout: float = 12.0

    def update_shared_frame(self, frame):
        """Receives a frame from the dashboard to prevent double-capturing."""
        self.shared_frame = frame
        self.shared_frame_time = time.time()

    def get_vision_context(self):
        """Returns the rolling scene summary if available, else the last raw description.

        Freshness honesty: if the last real capture has gone stale (heartbeat parked
        for Media Watch / Chess, frozen loop, or never captured), do NOT hand back an
        old scene summary dressed up with a soft 'Ns ago' tag — the LLM will happily
        confabulate off it. Past the stale cutoff, return an explicit 'I can't see the
        current screen' directive. A blunt admission beats a confident wrong guess."""
        # Master switch: Vision off = blind. Never leak stale scene cache as ghost data.
        if not self.master_enabled:
            return "[Vision is OFF — I can't see the screen right now.]"

        age = (time.time() - self.last_capture_time) if self.last_capture_time else None
        # Stale cutoff: comfortably beyond a normal heartbeat tick (10-30s) so a
        # single slow/missed frame doesn't trip it, but tight enough to catch a real
        # freeze. Tracks heartbeat_interval so immersive (10s) and normal (30s) modes
        # both get a sane window.
        stale_cutoff = max(75.0, float(getattr(self, "heartbeat_interval", 30.0)) * 3.0)

        if age is None or age > stale_cutoff:
            if self.scene_summary and age is not None:
                mins = int(age // 60)
                ago = f"~{mins} min" if mins >= 1 else f"~{int(age)}s"
                return (
                    f"[VISION FROZEN — my eyes are stale. The last frame I actually saw was {ago} ago; "
                    f"I CANNOT see the current screen. Say plainly that you can't see right now / your "
                    f"vision's frozen rather than guessing. The following is the LAST scene from before "
                    f"the freeze, for reference ONLY — do not describe it as if it's live: {self.scene_summary}]"
                )
            return "[VISION FROZEN — I can't see the screen right now.]"

        if self.scene_summary:
            return f"[Scene summary, last updated {int(age)}s ago] {self.scene_summary}"
        if not self.last_description:
            return "Vision Initializing..."
        return f"[Seen {int(age)}s ago] {self.last_description}"

    def get_recent_visual_memory(self, max_age: float = 60.0) -> str:
        """Returns the short-term rolling buffer of recent frame descriptions
        (last few heartbeat captures), filtered to entries newer than max_age.
        Used to give the LLM a 'short-term visual memory' so it can answer
        from what was actually recently seen, rather than fabricating."""
        if not self.context_buffer.buffer:
            return ""
        # The buffer entries are timestamped strings like "[HH:MM:SS] desc".
        # We can't easily age-filter without a parallel timestamp store, so we
        # rely on last_capture_time as a coarse freshness gate.
        if not self.last_capture_time or (time.time() - self.last_capture_time) > max_age:
            return ""
        return self.context_buffer.get_context_string()

    async def capture_and_answer(self, question: str) -> str:
        """Forces a fresh, targeted snapshot specifically to answer a visual
        question (e.g. 'what color are her eyes', 'who's on screen'). Returns
        a direct answer from the image, or an UNCERTAIN: prefix if the model
        can't see well enough. Updates last_capture_time + last_description so
        downstream cached-context paths benefit from the new frame.

        This is the anti-confabulation pre-step: never let the LLM answer a
        visual question from priors when a fresh look is possible.
        """
        # Master switch: Vision off = blind. No screen grab, in character.
        if not self.master_enabled:
            return "I can't see your screen right now — my vision is switched off."
        if not self.client:
            return "UNCERTAIN: vision unavailable (missing API key)."
        try:
            def process_image():
                img = self._grab_region()
                if img is None:
                    return None
                img.thumbnail((1920, 1080))
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            base64_image = await asyncio.to_thread(process_image)
            if base64_image is None:
                return "I can't see the game window right now — give me a sec."

            prompt = (
                "You are looking at Jonny's computer screen. Jonny just asked Kira "
                "(an AI watching with him) the following question and she needs a "
                "factual answer grounded ONLY in what is actually visible right now.\n\n"
                f"Question: \"{question.strip()}\"\n\n"
                "Answer the question directly and specifically based on what you can see. "
                "1-2 sentences. Be precise about colors, names visible on screen, counts, "
                "positions, and on-screen text. Quote any visible names exactly as shown.\n\n"
                "HONESTY RULES (critical):\n"
                "- If the answer is not actually visible (off-screen, occluded, blurred, "
                "transition frame, ambiguous), START your reply with 'UNCERTAIN:' and say "
                "what you CAN see plus why you can't answer precisely.\n"
                "- Do NOT guess. Do NOT infer from genre tropes or character archetypes. "
                "If a character's eye color, hair color, outfit detail, etc. is not clearly "
                "visible in this frame, say UNCERTAIN.\n"
                "- Do NOT invent character names. Only use names that appear as on-screen "
                "labels or UI text."
            )

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        }},
                    ]}],
                    max_tokens=180,
                    temperature=0,
                ),
                timeout=15,
            )
            content = (response.choices[0].message.content or "").strip()
            _record_vision_usage(response)
            self.last_capture_time = time.time()
            # Mirror into last_description so other cached paths see a real frame,
            # but keep the description short and grounded.
            if content:
                self.last_description = content
                # Also push into the rolling buffer so the short-term memory benefits.
                self.context_buffer.add(content)
            return content or "UNCERTAIN: empty vision response."
        except asyncio.TimeoutError:
            print("   [WARN] vision_agent (capture_and_answer) LLM call timed out after 15s — skipping")
            return "UNCERTAIN: vision call timed out."
        except Exception as e:
            print(f"   [WARN] vision_agent: capture_and_answer failed: {e}")
            return f"UNCERTAIN: vision call failed ({e})."

    async def heartbeat_loop(self):
        print("   [System] Eco-Mode Vision Active (dynamic heartbeat).")
        while True:
            if self.is_active:
                # skip_next_frame is set by the VRAM auto-degrade monitor when
                # reserved VRAM exceeds 14.5 GB. Clear the flag, then skip the
                # API call so no additional VRAM pressure is incurred this tick.
                if getattr(self, "skip_next_frame", False):
                    self.skip_next_frame = False
                    print("[VisionAgent] Skipping heartbeat frame (VRAM pressure).")
                else:
                    desc = await self.capture_and_describe(is_heartbeat=True)
                    if desc:
                        self.last_description = desc
                        await self._update_scene_summary(desc)
                        self._check_dialogue_change(desc)
            await asyncio.sleep(self.heartbeat_interval)

    async def capture_and_transcribe(self) -> str:
        """High-fidelity screen text extraction. Used when the user explicitly asks
        Kira to read what's on screen."""
        # Master switch: Vision off = blind. No screen grab, in character.
        if not self.master_enabled:
            return "I can't read the screen right now — my vision is switched off."
        if not self.client:
            return "Vision unavailable (Missing API Key)."
        try:
            def process_image():
                img = self._grab_region()
                if img is None:
                    return None
                img.thumbnail((1920, 1080))
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=90)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            base64_image = await asyncio.to_thread(process_image)
            if base64_image is None:
                return "I can't read the game window right now."

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": self.TRANSCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        }},
                    ]}],
                    max_tokens=600,
                    temperature=0,
                ),
                timeout=15,
            )
            content = response.choices[0].message.content
            _record_vision_usage(response)
            self.last_capture_time = time.time()
            return content.strip() if content else "NO TEXT VISIBLE"
        except asyncio.TimeoutError:
            print("   [WARN] vision_agent (capture_and_transcribe) LLM call timed out after 15s — skipping")
            return "Could not transcribe screen: vision call timed out."
        except Exception as e:
            print(f"   [WARN] vision_agent: capture_and_transcribe failed: {e}")
            return f"Could not transcribe screen: {e}"

    async def _update_scene_summary(self, new_description: str):
        """Roll the scene summary forward with the latest frame observation."""
        if not self.client or not new_description:
            return
        try:
            # Bound the request size. The rolling summary feeds its own output
            # back in as `previous` every cycle, so an over-long model response
            # would compound until the request blows the gateway's size limit
            # (the 431 "Request headers are too large" we were seeing). Clamp
            # both inputs before formatting so the request stays small.
            _prev = (self.scene_summary or "(none yet)")[:_SCENE_SUMMARY_MAX_CHARS]
            _newest = new_description[:_SCENE_DESC_MAX_CHARS]
            prompt = self.SCENE_SUMMARY_PROMPT.format(
                previous=_prev,
                newest=_newest,
            )
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180,
                    temperature=0.3,
                ),
                timeout=15,
            )
            _summary = response.choices[0].message.content.strip()
            _record_vision_usage(response)
            # Clamp the stored summary too — defends the next roll's `previous`.
            self.scene_summary = _summary[:_SCENE_SUMMARY_MAX_CHARS]
        except asyncio.TimeoutError:
            print("   [WARN] vision_agent (_update_scene_summary) LLM call timed out after 15s — skipping")
        except Exception as e:
            print(f"   [WARN] vision_agent: scene summary update failed: {e}")

    def _check_dialogue_change(self, description: str):
        """Tracks when on-screen text changes. Used downstream for VN silence-awareness."""
        dialogue = ""
        for line in description.splitlines():
            stripped = line.strip()
            if stripped.startswith("DIALOGUE:"):
                dialogue = stripped[len("DIALOGUE:"):].strip()
                break
        if not dialogue:
            dialogue = description.strip()

        if dialogue and dialogue != self.previous_dialogue:
            self.previous_dialogue = dialogue
            self.last_dialogue_change_time = time.time()

    async def capture_and_describe(self, is_heartbeat=False):
        """Captures screen and calls Vision API. Uses activity-aware prompts."""
        if not self.is_active:
            return None
        if not self.client:
            return "Vision unavailable (Missing API Key.)."

        # Dedup: a HEARTBEAT capture skips if an on-demand capture started within the
        # window (it's likely still in flight) — prevents a 2nd concurrent gpt-4o-mini
        # call. On-demand always proceeds and stamps the window. OFF -> never skips.
        if (VISION_CAPTURE_DEDUP_ENABLED and is_heartbeat
                and (time.time() - self._last_ondemand_capture_ts) < VISION_CAPTURE_DEDUP_WINDOW_S):
            print("   [Vision] heartbeat capture skipped — on-demand capture in flight (dedup).")
            return None
        if not is_heartbeat:
            self._last_ondemand_capture_ts = time.time()

        try:
            # Capture and Scale based on quality mode
            # Run blocking image capture/processing in a thread
            def process_image():
                img = self._grab_region()
                if img is None:
                    return None

                if self.quality_mode == "high" and not is_heartbeat:
                    img.thumbnail((1920, 1080)) # 1080p for high detail
                    quality = 85
                else:
                    img.thumbnail((1280, 720)) # 720p for fast/heartbeat
                    quality = 60

                # Stash the downscaled frame for the dashboard EYES preview
                # (/vision/thumbnail). Done before the base64 conversion that
                # otherwise discards the PIL image.
                self.last_frame = img

                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            base64_image = await asyncio.to_thread(process_image)
            if base64_image is None:
                # Vision-lock (Pokémon) and the game window is gone — skip this tick rather than
                # describing a stale/desktop frame. The loud log already fired in _grab_region.
                return self.last_description

            # One-time confirmation that the EYES thumbnail frame is being
            # captured (served by the dashboard's /vision/thumbnail endpoint).
            if self.last_frame is not None and not getattr(self, "_thumb_logged", False):
                self._thumb_logged = True
                _w, _h = self.last_frame.size
                print(f"   [Vision] last_frame set ({_w}x{_h}) — EYES thumbnail live.")

            # Select prompt based on activity type
            if self.activity_type == "vn":
                prompt = self.VN_DESCRIBE_PROMPT
            elif self.activity_type == "media":
                prompt = self.MEDIA_DESCRIBE_PROMPT
            else:
                prompt = self.DESCRIBE_PROMPT

            # Auto-detect context
            if self.context_buffer.buffer and self.activity_type != "vn":
                prompt += f"\nPrevious context: {self.context_buffer.get_context_string()}"
            
            # Dynamic Detail: heartbeat ticks use low detail (cheap, ~constant
            # background polling); on-demand describes keep high detail for cognition.
            visual_detail = "low" if is_heartbeat else "high"

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini", # Fastest vision model
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": visual_detail}}
                    ]}],
                    max_tokens=200
                ),
                timeout=15,
            )
            
            content = response.choices[0].message.content
            _record_vision_usage(response)
            if self.activity_type != "vn":
                self.context_buffer.add(content)
            self.last_capture_time = time.time()
            return content
        except asyncio.TimeoutError:
            print("   [WARN] vision_agent (capture_and_describe) LLM call timed out after 15s — skipping")
            return None
        except Exception as e:
            print(f"   [WARN] vision_agent: capture_and_describe failed: {e}")
            return f"My vision is a bit glitchy: {e}"

    async def capture_vn_state(self) -> dict:
        """
        VN-specific capture: returns a structured dict with keys:
          speaker, dialogue, choices (list of strings), scene
        Returns None if capture fails or screen is transitioning.
        """
        raw = await self.capture_and_describe(is_heartbeat=False)
        if not raw or "loading screen" in raw.lower():
            return None
        try:
            result = {"speaker": "", "dialogue": "", "choices": [], "scene": ""}
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("SPEAKER:"):
                    result["speaker"] = line[len("SPEAKER:"):].strip()
                elif line.startswith("DIALOGUE:"):
                    result["dialogue"] = line[len("DIALOGUE:"):].strip()
                elif line.startswith("SCENE:"):
                    result["scene"] = line[len("SCENE:"):].strip()
                elif line.startswith("CHOICES:"):
                    choices_raw = line[len("CHOICES:"):].strip()
                    if choices_raw.lower() != "none":
                        # Parse "1. Option A, 2. Option B" or newline-separated
                        found = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', choices_raw)
                        result["choices"] = [c.strip().rstrip(',') for c in found if c.strip()]
            # Only return if we got actual dialogue
            if result["dialogue"] and result["dialogue"] != "[loading screen]":
                return result
            return None
        except Exception:
            return None

    # ── Turbo Vision slideshow (multi-frame "what happened" timeline) ──────────
    # Ported from MediaWatch into the always-on path. Full-screen capture (no window
    # targeting). Gated entirely by start_slideshow()/stop_slideshow(), which the bot
    # calls from apply_deep_senses() only when TURBO_VISION_SLIDESHOW_ENABLED.

    def start_slideshow(self):
        """Start the dedicated full-screen capture + multi-frame analysis loops.
        Idempotent. No-op if the flag is off or already running."""
        if not TURBO_VISION_SLIDESHOW_ENABLED:
            return
        if self._slideshow_running:
            return
        if not self.client:
            print("   [TurboVision] slideshow not started — no vision client (missing API key).")
            return
        self._slideshow_running = True
        self._slideshow_frames.clear()
        self.episode_log.clear()
        self._slideshow_start_ts = time.time()
        self._slideshow_calls = 0
        self._slideshow_last_content_mid_ts = 0.0
        self._slideshow_capture_task = asyncio.ensure_future(self._slideshow_capture_loop())
        self._slideshow_analysis_task = asyncio.ensure_future(self._slideshow_analysis_loop())
        print(f"   [TurboVision] Slideshow ON — full-screen capture every "
              f"{TURBO_VISION_CAPTURE_INTERVAL_S:.2f}s, analysis every "
              f"{TURBO_VISION_ANALYSIS_INTERVAL_S:.1f}s, buffer={self._slideshow_frames.maxlen} "
              f"(~{TURBO_VISION_CAPTURE_INTERVAL_S * self._slideshow_frames.maxlen:.0f}s window).")

    def stop_slideshow(self):
        """Stop the slideshow loops. Keeps episode_log intact for recall until next start."""
        if not self._slideshow_running:
            return
        self._slideshow_running = False
        for t in (self._slideshow_capture_task, self._slideshow_analysis_task):
            if t and not t.done():
                t.cancel()
        self._slideshow_capture_task = None
        self._slideshow_analysis_task = None
        print(f"   [TurboVision] Slideshow OFF — {self._slideshow_calls} analysis calls, "
              f"{len(self.episode_log)} events.")

    def _grab_region(self, allow_shared=True):
        """The single capture chokepoint. THREE modes, in priority:
          1. POKÉMON VISION-LOCK — when self.capture_window_title is set, grab ONLY that window's
             rect (the game window). She sees the game and nothing else. If the window is gone /
             minimised / pygetwindow is missing, return None and log LOUD — we NEVER silently fall
             back to the full desktop (that silent fallback IS the leak we're fixing; Constraint #3).
          2. SHARED — a fresh dashboard-pushed frame (<1.0s old) when allowed.
          3. FULL-SCREEN — the default ImageGrab.grab().
        Returns a PIL image, or None ONLY in vision-lock mode when the window is unavailable
        (callers skip the frame / answer honestly rather than describing the desktop)."""
        title = self.capture_window_title
        if title:
            if not PYGETWINDOW_AVAILABLE:
                print(f"   [Vision] !! vision-lock '{title}' requested but pygetwindow missing — "
                      f"skipping frame (pip install pygetwindow pywin32)", flush=True)
                return None
            try:
                needle = title.strip().lower()
                win = next((w for w in _pgw.getAllWindows()
                            if needle in (w.title or "").lower()), None)
            except Exception as e:
                print(f"   [Vision] !! vision-lock '{title}' lookup failed: {e} — skipping frame",
                      flush=True)
                return None
            if win is None or win.width <= 0 or win.height <= 0:
                print(f"   [Vision] !! vision-lock '{title}' window not found/minimised — skipping "
                      f"frame (NOT leaking the desktop)", flush=True)
                return None
            try:
                return ImageGrab.grab(
                    bbox=(win.left, win.top, win.left + win.width, win.top + win.height))
            except Exception as e:
                print(f"   [Vision] !! vision-lock '{title}' grab failed: {e} — skipping frame",
                      flush=True)
                return None
        # normal (un-locked) capture
        if allow_shared and self.shared_frame and (time.time() - self.shared_frame_time) < 1.0:
            return self.shared_frame.copy()
        return ImageGrab.grab()

    def _grab_fullscreen_b64(self):
        """Sync full-screen grab → downscaled JPEG base64. Reuses a fresh shared
        frame if the dashboard just captured one (avoids a double grab)."""
        try:
            img = self._grab_region()
            if img is None:
                return None
            img.thumbnail((1280, 720))
            # EYES backfill (precedence: heartbeat wins when live). When the heartbeat
            # is PARKED (is_active False — e.g. chess), it isn't writing last_frame /
            # last_capture_time, so the dashboard EYES freeze. The slideshow is grabbing
            # fresh full-screen frames anyway, so surface them to the EYES WHILE PARKED.
            # Only when is_active is False — we never clobber a live heartbeat's frame.
            if not self.is_active:
                self.last_frame = img
                self.last_capture_time = time.time()
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=65)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"   [TurboVision] capture error: {e}")
            return None

    async def _slideshow_capture_loop(self):
        """Periodically grab a full-screen frame into the rolling buffer."""
        try:
            while self._slideshow_running:
                try:
                    # Honor the master Vision switch — don't grab when vision is OFF.
                    if self.master_enabled:
                        b64 = await asyncio.to_thread(self._grab_fullscreen_b64)
                        if b64:
                            self._slideshow_frames.append({"ts": time.time(), "b64": b64})
                except Exception as e:
                    print(f"   [TurboVision] capture tick error: {e}")
                await asyncio.sleep(max(0.5, TURBO_VISION_CAPTURE_INTERVAL_S))
        except asyncio.CancelledError:
            pass

    async def _slideshow_analysis_loop(self):
        """Periodically send the frame buffer to vision and log what happened."""
        try:
            await asyncio.sleep(min(TURBO_VISION_ANALYSIS_INTERVAL_S, TURBO_VISION_CAPTURE_INTERVAL_S * 2))
            while self._slideshow_running:
                try:
                    if len(self._slideshow_frames) >= 2:
                        await self._run_slideshow_analysis_once()
                except Exception as e:
                    print(f"   [TurboVision] analysis tick error: {e}")
                # Floor lowered 5.0 -> 2.0 (2026-06-22) so TURBO_VISION_ANALYSIS_INTERVAL_S
                # is honored below 5s if Jonny dials it (cost scales linearly; 2s = sane min).
                await asyncio.sleep(max(2.0, TURBO_VISION_ANALYSIS_INTERVAL_S))
        except asyncio.CancelledError:
            pass

    async def _run_slideshow_analysis_once(self):
        if not self.client:
            return
        frames_snapshot = list(self._slideshow_frames)
        content = [{"type": "text", "text": self.SLIDESHOW_ANALYSIS_PROMPT}]
        for f in frames_snapshot:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}", "detail": "low"},
            })
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": content}],
                    max_tokens=260,
                    temperature=0.2,
                ),
                timeout=self._slideshow_cloud_timeout,
            )
        except asyncio.TimeoutError:
            print(f"   [TurboVision] analysis TIMEOUT after {self._slideshow_cloud_timeout:.1f}s — skipping interval.")
            return
        except Exception as e:
            print(f"   [TurboVision] analysis call failed: {e}")
            return
        _record_vision_usage(resp)
        try:
            summary = (resp.choices[0].message.content or "").strip()
        except Exception:
            summary = ""
        if not summary:
            return
        self._slideshow_calls += 1
        upper = summary.upper()
        uncertain = upper.startswith("UNCERTAIN")
        static = upper.startswith("STATIC")
        now = time.time()
        t_rel = now - self._slideshow_start_ts
        try:
            _f_ts = [f["ts"] for f in frames_snapshot if "ts" in f]
            content_mid_ts = (min(_f_ts) + max(_f_ts)) / 2.0 if _f_ts else now
        except Exception:
            content_mid_ts = now
        self.episode_log.append({
            "ts": now, "t_rel_s": t_rel, "summary": summary,
            "uncertain": uncertain, "static": static, "content_mid_ts": content_mid_ts,
        })
        self._slideshow_last_content_mid_ts = content_mid_ts
        # EYES backfill (precedence: heartbeat wins when live). When the heartbeat is
        # PARKED, mirror a SUBSTANTIVE slideshow summary into the scene fields the
        # dashboard + get_vision_context read, so the EYES summary + her prompt-side
        # sight stay live during the park. Skipped entirely when the heartbeat is
        # running (is_active) — it owns scene_summary/last_description then.
        if not self.is_active and not uncertain and not static:
            self.scene_summary = summary[:_SCENE_SUMMARY_MAX_CHARS]
            self.last_description = summary
            self.last_capture_time = now
        h, rem = divmod(int(t_rel), 3600)
        m, s = divmod(rem, 60)
        tag = "UNCERTAIN " if uncertain else ("STATIC " if static else "")
        print(f"   [TurboVision] +{h:02d}:{m:02d}:{s:02d} {tag}"
              f"{summary[:140]}{'…' if len(summary) > 140 else ''}")

    # ── Episode-timeline accessors (parity with MediaWatch's API) ──────────────

    def slideshow_has_context(self) -> bool:
        return bool(self.episode_log)

    def get_last_episode_content_mid_ts(self) -> float:
        return self._slideshow_last_content_mid_ts

    def get_latest_episode_summary(self) -> str:
        """Most recent SUBSTANTIVE summary (skips UNCERTAIN/STATIC)."""
        for e in reversed(self.episode_log):
            if not e.get("uncertain") and not e.get("static"):
                return e.get("summary", "") or ""
        return ""

    def get_episode_context(self, max_entries: int = 10, char_budget: int = 2600) -> str:
        """Formatted timeline for prompt injection. Recent events verbatim + a rolled
        'earlier' digest so the block stays bounded no matter how long the session runs.
        Mirrors MediaWatch.get_episode_context (relabeled TURBO VISION)."""
        if not self.episode_log:
            return ""
        all_entries = list(self.episode_log)
        recent = all_entries[-max_entries:]
        older = all_entries[:-max_entries] if len(all_entries) > max_entries else []

        def _stamp(e) -> str:
            t = int(e["t_rel_s"]); h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _first_sentence(text: str) -> str:
            text = (text or "").strip()
            for tag in ("UNCERTAIN:", "STATIC:"):
                if text.upper().startswith(tag):
                    text = text[len(tag):].strip()
            for sep in (". ", "! ", "? "):
                idx = text.find(sep)
                if 0 < idx < 160:
                    return text[:idx + 1].strip()
            return text[:160].strip()

        lines = ["[TURBO VISION — recent on-screen event timeline, oldest first]"]
        if older:
            substantive = [e for e in older if not e.get("uncertain") and not e.get("static")]
            picks = substantive or older
            if len(picks) > 3:
                picks = [picks[0], picks[len(picks) // 2], picks[-1]]
            digest_bits = [f"{_stamp(e)} {_first_sentence(e['summary'])}" for e in picks]
            span = f"{_stamp(older[0])}–{_stamp(older[-1])}"
            lines.append(f"  [EARLIER, condensed | {span}]: " + " ".join(digest_bits))

        verbatim = [f"  [{_stamp(e)}] {e['summary']}" for e in recent]
        block = "\n".join(verbatim)
        i = 0
        while len(block) > char_budget and i < len(recent) - 1:
            verbatim[i] = f"  [{_stamp(recent[i])}] {_first_sentence(recent[i]['summary'])}"
            block = "\n".join(verbatim)
            i += 1
        lines.append(block)
        lines.append(
            "[NOTE] This is Kira's actual visual record of what just happened on screen. "
            "When asked about earlier moments, refer to this timeline. If a specific moment "
            "isn't here, say so honestly — do not invent. Vision is visual-only (no audio)."
        )
        return "\n".join(lines)
