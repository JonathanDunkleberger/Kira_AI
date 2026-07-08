# gemini_vision.py — Google Gemini vision client (core-Kira "eyes", ALL modes).
#
# Replaces the prior OpenAI gpt-4o-mini vision path. This is the single chokepoint
# for every screen-understanding call in the codebase: the always-on heartbeat
# (vision_agent), on-demand answer/transcribe, the Turbo Vision slideshow, AND the
# VN autopilot's classify/read/scene/choice calls. Vision must never be half-OpenAI
# / half-Gemini, so both call sites share THIS module.
#
# HYBRID TIERING (mirrors Kira's Groq→Sonnet→Opus pattern):
#   - HEARTBEAT  (gemini-3.1-flash-lite): cheap/fast, continuous all-day background
#                vision. Used for heartbeat ticks, classify, slideshow.
#   - ESCALATED  (gemini-3-flash-preview): richer reasoning for salient moments
#                (on-demand answers, transcription, dialogue OCR, choice extraction).
#
# COST/LATENCY levers (multi-hour sessions):
#   - thinking minimized (ThinkingLevel.LOW) on every call.
#   - frames downscaled to <=384px before send (biggest cost lever, ~3.5x savings).
#     Callers that already downscale (VN OCR needs legible text) pass max_px to opt
#     into a larger cap; default 384 for the always-on path.
#
# ANTI-CONFABULATION: a shared "describe only what you see" guard is appended to
# every describe-style prompt (both Gemini models over-infer game mechanics — they
# invent "disobeying trainer" / "Rest" etc. when left unconstrained). Verbatim-OCR
# and structured-extraction prompts are exempt (they have their own strict format).

import asyncio
import base64
from io import BytesIO

from google.genai import Client, types
from google.genai import errors as genai_errors

from kira.config import GEMINI_IMAGE_API_KEY

# ── Model tiers ───────────────────────────────────────────────────────────────
GEMINI_VISION_HEARTBEAT_MODEL = "gemini-3.1-flash-lite"   # always-on, cheap/fast
GEMINI_VISION_ESCALATED_MODEL = "gemini-3-flash-preview"  # salient moments, richer

# Always-on downscale cap (px, longest side). 384 is the cost sweet-spot for the
# heartbeat; callers needing legible OCR text (VN dialogue) override via max_px.
DEFAULT_MAX_PX = 384

# Anti-confabulation guard appended to describe-style prompts (NOT to verbatim-OCR
# or structured-extraction prompts, which have their own strict output contracts).
SEE_ONLY_GUARD = (
    "\n\nGROUNDING (critical): Describe ONLY what is literally, visually present in "
    "this frame — colors, shapes, on-screen text, characters, UI you can actually "
    "see. Do NOT infer game mechanics, rules, internal state, or what is 'about to "
    "happen' (e.g. do not say a character is 'disobeying', 'resting', 'poisoned', "
    "or 'about to attack' unless that exact word is visible on screen). Report "
    "observed visual facts, never inferred game logic."
)


class GeminiVisionClient:
    """Async Gemini vision wrapper. One shared instance feeds vision_agent AND
    vn_autopilot (constructed once in vision_agent, passed into the autopilot —
    same single-swap seam the old AsyncOpenAI client used).

    Exposes:
      - .available           — True when an API key is present (mirrors old `client` truthiness).
      - describe(...)        — image + prompt -> text, with tiering + see-only guard.
      - generate_text(...)   — text-only prompt -> text (scene-summary roll; no image).
    Encoding is base64-data-URL-free: bytes go straight to types.Part.from_bytes.
    """

    def __init__(self):
        self._client = None
        if GEMINI_IMAGE_API_KEY:
            try:
                self._client = Client(api_key=GEMINI_IMAGE_API_KEY)
            except Exception as e:
                print(f"   [Vision] !! Gemini client init failed: {e}")
                self._client = None
        else:
            print("   [Vision] Warning: GEMINI_IMAGE_API_KEY not found in config — vision blind.")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── image encoding ─────────────────────────────────────────────────────────
    @staticmethod
    def encode_jpeg(img, max_px: int = DEFAULT_MAX_PX, quality: int = 70) -> bytes:
        """Downscale a PIL image to <=max_px (longest side) and JPEG-encode to bytes.
        The downscale is the dominant cost lever for multi-hour vision."""
        w, h = img.size
        longest = max(w, h)
        if longest > max_px:
            scale = max_px / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def _record_usage(resp, model: str) -> None:
        """Record Gemini usage to cost_tracker. Best-effort; never raises."""
        try:
            from kira.brain.cost_tracker import cost_tracker as _ct
            u = getattr(resp, "usage_metadata", None)
            if u:
                _ct.record(
                    model=model,
                    input_tokens=getattr(u, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(u, "candidates_token_count", 0) or 0,
                    purpose="vision",
                )
        except Exception:
            pass

    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        """True for retriable rate-limit / overload errors (429 / 503 / RESOURCE_EXHAUSTED)."""
        if isinstance(e, genai_errors.APIError):
            code = getattr(e, "code", None)
            if code in (429, 503, 529):
                return True
        msg = str(e).lower()
        return any(s in msg for s in (
            "rate limit", "429", "503", "529", "resource_exhausted",
            "overloaded", "too many requests", "unavailable",
        ))

    def _gen_config(self, max_tokens: int, temperature: float):
        return types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            # MINIMAL (not LOW) thinking. Two reasons, both verified live:
            #  1. Cost/latency: zero thinking tokens on the always-on path.
            #  2. Correctness: LOW spends ~7-57 reasoning tokens BEFORE any output,
            #     which STARVES tight output caps — a max_output_tokens=10 classify
            #     call returns EMPTY (finish_reason=MAX_TOKENS) under LOW. MINIMAL
            #     emits no thinking tokens, so even a 10-token cap answers cleanly
            #     while full describes keep their quality.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        )

    # ── core calls ──────────────────────────────────────────────────────────────
    async def describe(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        escalate: bool = False,
        max_tokens: int = 200,
        temperature: float = 0.0,
        see_only: bool = True,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Image + prompt -> text. `escalate` picks the richer model. `see_only`
        appends the anti-confabulation guard (off for verbatim-OCR / structured calls).
        Returns the response text, or raises the underlying genai error (callers own
        timeout/backoff/UNCERTAIN semantics, exactly as the OpenAI path did)."""
        if not self._client:
            return ""
        model = GEMINI_VISION_ESCALATED_MODEL if escalate else GEMINI_VISION_HEARTBEAT_MODEL
        full_prompt = prompt + SEE_ONLY_GUARD if see_only else prompt
        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                full_prompt,
            ],
            config=self._gen_config(max_tokens, temperature),
        )
        self._record_usage(resp, model)
        return (resp.text or "").strip()

    async def describe_multi(
        self,
        image_bytes_list: list[bytes],
        prompt: str,
        *,
        escalate: bool = False,
        max_tokens: int = 260,
        temperature: float = 0.2,
        see_only: bool = True,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Multi-frame variant (Turbo Vision slideshow: N frames + one prompt)."""
        if not self._client:
            return ""
        model = GEMINI_VISION_ESCALATED_MODEL if escalate else GEMINI_VISION_HEARTBEAT_MODEL
        full_prompt = prompt + SEE_ONLY_GUARD if see_only else prompt
        parts = [types.Part.from_bytes(data=b, mime_type=mime_type) for b in image_bytes_list]
        parts.append(full_prompt)
        resp = await self._client.aio.models.generate_content(
            model=model, contents=parts, config=self._gen_config(max_tokens, temperature),
        )
        self._record_usage(resp, model)
        return (resp.text or "").strip()

    async def generate_text(
        self, prompt: str, *, max_tokens: int = 180, temperature: float = 0.3,
    ) -> str:
        """Text-only prompt -> text (scene-summary roll — no image)."""
        if not self._client:
            return ""
        model = GEMINI_VISION_HEARTBEAT_MODEL
        resp = await self._client.aio.models.generate_content(
            model=model, contents=[prompt], config=self._gen_config(max_tokens, temperature),
        )
        self._record_usage(resp, model)
        return (resp.text or "").strip()
