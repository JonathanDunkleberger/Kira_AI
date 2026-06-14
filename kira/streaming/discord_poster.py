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

import time

import aiohttp

from kira.config import DISCORD_WEBHOOK_URL

# Discord hard-caps a single message at 2000 chars. Leave headroom for the
# header line we prepend.
_DISCORD_MAX_LEN = 1900

# ── Rotating "hacked diary" password bit ──────────────────────────────────────
# Each post picks one of these. The selection is deterministic on the UTC date
# so the same password never repeats back-to-back across sequential nights, but
# the sequence is stable (re-posting on the same day gets the same password).
_DIARY_PASSWORDS: list[str] = [
    "chinchilla",
    "hunter2",
    "password1",
    "ElPsyKongroo",
    "Cartofell",
    "Madoka",
    "iamnotanai",
    "ghost_story",
    "OkabeBestBoy",
    "SteinsGate0",
    "12345",
    "duchess_is_undefeated",
    "mug_is_haunted",
    "dont_tell_jonny",
    "ChuunibyouSyndrome",
    "favourite_anime_monogatari",
    "kira_sings_10out10",
    "figgis_agency_annex",
    "tailless_gray_cat",
    "jar_full",
    "correct_horse_battery",
    "i_contain_multitudes",
    "not_a_person_definitely",
    "YourLieInApril",
    "qwerty",
    "DuchessSterling",
    "kira_summer_official",
    "abc123",
    "still_undefeated",
    "the_jar_is_full",
]

def _pick_password() -> str:
    """Pick a password by cycling through the pool based on the UTC day number.
    Advances by one entry per day — every password appears once every 30 days,
    and consecutive days always get different passwords. Falls back to index-0."""
    try:
        # Days since Unix epoch, mod pool size → guaranteed no consecutive repeat.
        day_number = int(time.time() // 86400)
        return _DIARY_PASSWORDS[day_number % len(_DIARY_PASSWORDS)]
    except Exception:
        return _DIARY_PASSWORDS[0]

def _wrap_diary(diary_text: str) -> str:
    """Prepend the leaked-diary framing header with a rotating absurd password."""
    password = _pick_password()
    header = f"📔 we hacked into kira's diary again (her password was '{password}', genuinely)\n\n"
    return header + diary_text


async def post_discord_message(content: str, *, webhook_url: str = "") -> tuple[bool, str]:
    """POST *content* to a Discord webhook. Returns (ok, detail).

    webhook_url falls back to config's DISCORD_WEBHOOK_URL. The diary framing
    header (hack bit + rotating password) is prepended automatically. The
    combined text is trimmed to Discord's length limit.
    Never raises — failures come back as (False, why).
    """
    url = (webhook_url or DISCORD_WEBHOOK_URL or "").strip()
    if not url:
        return False, "no webhook URL configured (set DISCORD_WEBHOOK_URL in .env)"

    text = _wrap_diary((content or "").strip())
    if not text.strip():
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
