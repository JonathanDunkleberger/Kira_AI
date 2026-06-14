# discord_poster.py
# Minimal Discord webhook poster for Kira's daily diary (Phase 1).
#
# Design intent:
#   * REVIEW MODE by default. Nothing here is called automatically at session
#     end — the diary is generated and saved to disk first, and posting is a
#     deliberate manual action triggered from the dashboard. This module is just
#     the transport: given approved text, fire it at the webhook.
#   * No discord.py, no bot token, no gateway — a webhook is a plain HTTPS POST
#     that returns 204 on success. Keeps the dependency surface at aiohttp,
#     which is already in the environment.
#   * Fail-graceful: a missing URL or network error returns False and logs once;
#     it never raises into the caller (session shutdown / dashboard handler).

from __future__ import annotations

import aiohttp

from kira.config import DISCORD_WEBHOOK_URL

# Discord hard-caps a single message at 2000 chars. Leave headroom for the
# header line we prepend.
_DISCORD_MAX_LEN = 1900


async def post_discord_message(content: str, *, webhook_url: str = "") -> tuple[bool, str]:
    """POST *content* to a Discord webhook. Returns (ok, detail).

    webhook_url falls back to config's DISCORD_WEBHOOK_URL. The text is trimmed
    to Discord's length limit. Never raises — failures come back as (False, why).
    """
    url = (webhook_url or DISCORD_WEBHOOK_URL or "").strip()
    if not url:
        return False, "no webhook URL configured (set DISCORD_WEBHOOK_URL in .env)"

    text = (content or "").strip()
    if not text:
        return False, "empty content — nothing to post"
    if len(text) > _DISCORD_MAX_LEN:
        text = text[:_DISCORD_MAX_LEN - 1].rstrip() + "\u2026"

    payload = {"content": text}
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status in (200, 204):
                    return True, f"posted (HTTP {resp.status})"
                body = ""
                try:
                    body = (await resp.text())[:200]
                except Exception:
                    pass
                return False, f"webhook returned HTTP {resp.status}: {body}"
    except Exception as e:
        return False, f"post failed: {e}"
