# chat_poster.py
# Centralised, rate-limited outbound chat posting for Kira.
#
# Design intent:
#   * Reading chat is one thing; SENDING is a deliberate, occasional action.
#     This module is the single chokepoint so we can enforce one global
#     cooldown across every platform Kira can post to (no flooding by
#     posting to Twitch + YouTube + future Discord in the same second).
#   * Hard-gated behind ENABLE_CHAT_POSTING. If the flag is off, every
#     post() call no-ops cleanly and logs once. Safe to wire into prompts /
#     tool layers regardless.
#   * The cooldown is GLOBAL (not per-platform) on purpose: from chat's
#     perspective, "Kira posting" is one persona, regardless of where it
#     lands. One firehose limiter > one-per-platform.
#
# Not in scope here:
#   * Calling this from the LLM tool layer / function-calling \u2014 that wires
#     in separately. This module just exposes the capability.
#   * YouTube posting via Data API OAuth \u2014 stubbed because pytchat is
#     read-only. If/when added, plug it into YouTubeBot.post_message.

from __future__ import annotations

import asyncio
import random
import time
from typing import Optional, TYPE_CHECKING

from kira.config import (
    ENABLE_CHAT_POSTING,
    CHAT_POST_COOLDOWN_SEC,
    CHAT_POST_MAX_LEN,
)

if TYPE_CHECKING:
    from kira.streaming.twitch_bot import TwitchBot
    from kira.streaming.youtube_bot import YouTubeBot


# A small rotation of ASCII cats. Kept short \u2014 Twitch line height matters and
# multi-line ASCII art in chat is messy at best. These are one-liners that
# survive line-wrapping.
_ASCII_CATS = [
    "/ᐠ｡ꞈ｡ᐟ\\",
    "(=^\u30fb\u03c9\u30fb^=)",
    "(\uff89\u25d5\u30ee\u25d5\uff89\uff0a:\uff65\uff9f\u2727",
    "\u2230\u035c\u0296\u035c\u2230",
    "(\u02c3\u1d25\u02c2)\u2661",
    "(\u0e07 \u2022\u035c\u0296 \u2022)\u0e57",
    "/^\u2022\u30ce\u30fb<)~\u301c\u301c\u301c",
]


class ChatPoster:
    """Centralised outbound poster with global rate limiting.

    Construct once per bot lifetime; attach platform handles via
    ``set_twitch_bot()`` / ``set_youtube_bot()`` once they're ready. Calls to
    ``post()`` before handles are attached are safe no-ops.
    """

    def __init__(self) -> None:
        self.enabled = ENABLE_CHAT_POSTING
        self.cooldown_sec = CHAT_POST_COOLDOWN_SEC
        self.max_len = CHAT_POST_MAX_LEN
        self._last_post_ts: float = 0.0
        self._lock = asyncio.Lock()
        self._twitch_bot: Optional["TwitchBot"] = None
        self._youtube_bot: Optional["YouTubeBot"] = None
        self._disabled_log_emitted = False

        if not self.enabled:
            # Only log once at construction so we don't repeat every call.
            print("   [ChatPoster] Disabled (ENABLE_CHAT_POSTING=false). post() calls will no-op.")
        else:
            print(f"   [ChatPoster] Enabled. Global cooldown: {self.cooldown_sec:.0f}s, max len: {self.max_len}.")

    # ── wiring ────────────────────────────────────────────────────────────
    def set_twitch_bot(self, bot: "TwitchBot") -> None:
        self._twitch_bot = bot

    def set_youtube_bot(self, bot: "YouTubeBot") -> None:
        self._youtube_bot = bot

    # ── rate gate ─────────────────────────────────────────────────────────
    def can_post(self) -> bool:
        """True if the global cooldown has elapsed AND posting is enabled."""
        if not self.enabled:
            return False
        return (time.time() - self._last_post_ts) >= self.cooldown_sec

    def seconds_until_ready(self) -> float:
        """Time (s) until the next post() will be permitted. 0.0 if ready now."""
        if not self.enabled:
            return float("inf")
        remaining = self.cooldown_sec - (time.time() - self._last_post_ts)
        return max(0.0, remaining)

    # ── core post API ─────────────────────────────────────────────────────
    async def post(
        self,
        text: str,
        *,
        platforms: Optional[tuple[str, ...]] = None,
        force: bool = False,
    ) -> bool:
        """Send ``text`` to one or more chat platforms, subject to the global
        cooldown. Returns True if at least one platform accepted the post.

        Args:
            text: Message to send. Trimmed and truncated to ``max_len``.
            platforms: Tuple of platform names to target. None \u2192 every connected
                platform. Valid names: ``"twitch"``, ``"youtube"``.
            force: Bypass the cooldown gate. Use sparingly \u2014 e.g. for the
                user invoking the poster manually from the dashboard.

        Never raises \u2014 all underlying send errors are caught and logged.
        """
        if not self.enabled:
            return False

        clean = self._sanitize(text)
        if not clean:
            print("   [ChatPoster] Refusing to post empty/whitespace message.")
            return False

        async with self._lock:
            if not force and not self.can_post():
                wait = self.seconds_until_ready()
                print(f"   [ChatPoster] Suppressed (cooldown: {wait:.0f}s remaining). Message dropped: {clean[:60]!r}")
                return False

            targets = platforms or ("twitch", "youtube")
            sent_any = False

            if "twitch" in targets and self._twitch_bot is not None:
                ok = await self._safe_send_twitch(clean)
                if ok:
                    print(f"   [ChatPoster] Posted to Twitch: {clean[:80]!r}")
                    sent_any = True

            if "youtube" in targets and self._youtube_bot is not None:
                ok = await self._safe_send_youtube(clean)
                if ok:
                    print(f"   [ChatPoster] Posted to YouTube: {clean[:80]!r}")
                    sent_any = True

            if sent_any:
                self._last_post_ts = time.time()
            else:
                print("   [ChatPoster] No platform accepted the post (none connected or all errored).")
            return sent_any

    async def post_ascii_cat(self, *, force: bool = False) -> bool:
        """Post a random ASCII cat. Subject to the same global cooldown as
        plain text. Convenience wrapper around ``post()`` for the canonical
        flavor command."""
        cat = random.choice(_ASCII_CATS)
        return await self.post(cat, force=force)

    # ── internals ─────────────────────────────────────────────────────────
    def _sanitize(self, text: str) -> str:
        """Strip, collapse newlines (chat doesn't render them well), and
        truncate to platform-safe length. Twitch silently drops messages > 500
        chars; we cap at CHAT_POST_MAX_LEN (default 450) for headroom."""
        if not text:
            return ""
        # Collapse newlines/tabs to single spaces so multi-line LLM output
        # arrives as one chat line.
        flat = " ".join(text.split())
        if len(flat) > self.max_len:
            flat = flat[: self.max_len - 1].rstrip() + "\u2026"
        return flat

    async def _safe_send_twitch(self, text: str) -> bool:
        try:
            return await self._twitch_bot.post_message(text)  # type: ignore[union-attr]
        except Exception as e:
            # Auth failures must be LOUD and disable the feature — never silent.
            emsg = str(e).lower()
            if any(k in emsg for k in (
                "auth", "login", "token", "unauthor", "401", "forbidden", "403", "scope",
            )):
                print(f"   [ERROR] ChatPoster auth failed — feature disabled: {e}")
                self.enabled = False
            else:
                print(f"   [ChatPoster] Twitch send error: {e}")
            return False

    async def _safe_send_youtube(self, text: str) -> bool:
        try:
            return await self._youtube_bot.post_message(text)  # type: ignore[union-attr]
        except Exception as e:
            print(f"   [ChatPoster] YouTube send error: {e}")
            return False
