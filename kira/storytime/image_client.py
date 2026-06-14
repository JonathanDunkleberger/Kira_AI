# image_client.py — Swappable image-generation client for Storytime.
# ─────────────────────────────────────────────────────────────────────────────
# Provider-agnostic by design. The Storytime orchestrator talks ONLY to the
# ImageProvider interface, so the backend (Gemini today) can be swapped for
# another model later without touching the show pipeline.
#
# Default provider: Google Gemini 2.5 Flash Image ("Nano-Banana"), chosen for
# its style/character CONSISTENCY — you can pass a reference key frame with every
# later beat so all scenes look like one coherent shadow-puppet show.
#
# The Gemini key is GEMINI_IMAGE_API_KEY (a fresh Google AI Studio key), kept
# strictly separate from GOOGLE_API_KEY (Custom Search). See kira/config.py.

from __future__ import annotations

import abc
import asyncio
import io

from kira.config import (
    GEMINI_IMAGE_API_KEY,
    GEMINI_IMAGE_MODEL,
    STORYTIME_IMAGE_PROVIDER,
)


class ImageGenError(RuntimeError):
    """Raised when image generation fails (missing SDK/key, API error, or the
    model returned no image). The orchestrator surfaces the message as a
    per-beat error so a single ugly/failed scene never crashes a show."""


class ImageProvider(abc.ABC):
    """The only surface the Storytime pipeline depends on. Implement `generate`."""

    name: str = "abstract"

    @abc.abstractmethod
    async def generate(self, prompt: str, *,
                       reference_image: bytes | None = None) -> bytes:
        """Return PNG bytes for `prompt`. If `reference_image` (PNG bytes) is
        given, the provider should condition on it for style/character
        consistency. Must raise ImageGenError on any failure."""
        raise NotImplementedError


class GeminiImageProvider(ImageProvider):
    """Google Gemini 2.5 Flash Image ("Nano-Banana") provider.

    The SDK client is built lazily on first use so an absent SDK or empty key
    never crashes bot startup — it only fails when a show is actually prepared.
    """

    name = "gemini"

    def __init__(self, api_key: str = "", model: str = ""):
        self._api_key = (api_key or GEMINI_IMAGE_API_KEY or "").strip()
        self._model = (model or GEMINI_IMAGE_MODEL or "gemini-2.5-flash-image").strip()
        self._client = None  # built lazily

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ImageGenError(
                "GEMINI_IMAGE_API_KEY is empty — set a Google AI Studio key in "
                ".env (separate from GOOGLE_API_KEY / Custom Search)."
            )
        try:
            from google import genai  # google-genai SDK
        except Exception as e:  # pragma: no cover - import guard
            raise ImageGenError(
                "google-genai SDK not installed. Run: pip install google-genai"
            ) from e
        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as e:
            raise ImageGenError(f"Could not build Gemini client: {e}") from e
        return self._client

    def _generate_sync(self, prompt: str, reference_image: bytes | None) -> bytes:
        client = self._ensure_client()
        # Build the multimodal contents: text prompt, plus the reference image
        # (as a PIL image — the SDK accepts PIL.Image objects directly) when we
        # want cross-scene consistency.
        contents: list = [prompt]
        if reference_image:
            try:
                from PIL import Image
                contents.append(Image.open(io.BytesIO(reference_image)))
            except Exception:
                # If the reference can't be decoded, fall back to text-only —
                # a slightly less consistent image beats a hard failure.
                contents = [prompt]
        try:
            resp = client.models.generate_content(
                model=self._model,
                contents=contents,
            )
        except Exception as e:
            raise ImageGenError(f"Gemini image request failed: {e}") from e

        # Pull the first inline image part out of the response.
        png = self._extract_image_bytes(resp)
        if not png:
            raise ImageGenError(
                "Gemini returned no image (prompt may have been refused — keep "
                "scene prompts tame)."
            )
        return png

    @staticmethod
    def _extract_image_bytes(resp) -> bytes | None:
        try:
            candidates = getattr(resp, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        return inline.data
        except Exception:
            return None
        return None

    async def generate(self, prompt: str, *,
                       reference_image: bytes | None = None) -> bytes:
        # The SDK call is synchronous — run it off the event loop so a ~15s image
        # gen never blocks the bot. Provider-agnostic callers just await this.
        return await asyncio.to_thread(self._generate_sync, prompt, reference_image)


def get_image_provider(name: str = "") -> ImageProvider:
    """Factory. Returns the configured provider. Swap by changing
    STORYTIME_IMAGE_PROVIDER (env) or passing `name`."""
    choice = (name or STORYTIME_IMAGE_PROVIDER or "gemini").lower()
    if choice == "gemini":
        return GeminiImageProvider()
    raise ImageGenError(f"Unknown image provider '{choice}'. Known: gemini.")
