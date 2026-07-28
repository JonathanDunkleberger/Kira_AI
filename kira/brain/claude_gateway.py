"""claude_gateway.py — the single construction point for Claude-speaking clients.

WHY THIS EXISTS (2026-07-28): Anthropic banned the account and denied the unban
request, so direct api.anthropic.com auth is dead. OpenRouter exposes an
Anthropic-compatible endpoint (the "Anthropic Skin": POST /api/v1/messages) that
accepts the native Anthropic SDK wire format — same request/response shapes,
thinking blocks, tool use, prompt caching pass through. That means every existing
`client.messages.create(...)` call site in this codebase works unchanged; only the
client construction moves here.

Auth mechanics (per OpenRouter's Claude Code integration docs):
  - base_url = https://openrouter.ai/api  (SDK appends /v1/messages)
  - the OpenRouter key (sk-or-...) goes in as `auth_token` -> Authorization: Bearer
  - ANTHROPIC_API_KEY must be blank so the SDK never auths against Anthropic
    directly. We enforce that here by scrubbing the env var in OpenRouter mode,
    since the dead key may still be sitting in .env.

Model names: the Skin maps bare Claude model names (claude-opus-4-7 etc.) to the
matching OpenRouter Anthropic listings. If a name ever fails to map, set
CLAUDE_DEEP_MODEL / CLAUDE_CHAT_MODEL / CLAUDE_HAIKU_MODEL in .env to full
OpenRouter slugs (e.g. "anthropic/claude-opus-4.7").

RULE: never construct anthropic.Anthropic / AsyncAnthropic anywhere else.
"""

import os

from kira.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_ROUTE,
    OPENROUTER_ANTHROPIC_BASE_URL,
    OPENROUTER_API_KEY,
)

try:
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    Anthropic = None
    AsyncAnthropic = None
    ANTHROPIC_SDK_AVAILABLE = False


def gateway_mode() -> str:
    """Which route Claude traffic takes: 'openrouter', 'anthropic', or 'none'.
    Resolved once in kira.config (CLAUDE_ROUTE) so model-name normalization and
    client construction can never disagree."""
    return CLAUDE_ROUTE


def claude_key_present() -> bool:
    """True when SOME usable route to Claude exists. Use this everywhere the old
    code gated on `if not ANTHROPIC_API_KEY`."""
    return gateway_mode() != "none"


def gateway_label() -> str:
    """Human-readable route name for boot logs."""
    return {
        "openrouter": "OpenRouter (Anthropic Skin)",
        "anthropic": "Anthropic direct",
        "none": "NO ROUTE — set OPENROUTER_API_KEY",
    }[gateway_mode()]


def _client_kwargs() -> dict:
    mode = gateway_mode()
    if mode == "openrouter":
        # Scrub the dead direct key so the SDK can't pick it up from env and
        # send a conflicting x-api-key header (mirrors OpenRouter's own
        # "ANTHROPIC_API_KEY must be explicitly empty" guidance).
        os.environ["ANTHROPIC_API_KEY"] = ""
        return {
            "auth_token": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_ANTHROPIC_BASE_URL,
        }
    if mode == "anthropic":
        return {"api_key": ANTHROPIC_API_KEY}
    raise RuntimeError(
        "No route to Claude: set OPENROUTER_API_KEY (preferred — the Anthropic "
        "account is banned) or ANTHROPIC_API_KEY in .env."
    )


def make_async_claude_client() -> "AsyncAnthropic":
    """The client the whole bot shares (ai_core.anthropic_client and friends)."""
    if not ANTHROPIC_SDK_AVAILABLE:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    return AsyncAnthropic(**_client_kwargs())


def make_sync_claude_client() -> "Anthropic":
    """For synchronous offline scripts (backfills)."""
    if not ANTHROPIC_SDK_AVAILABLE:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    return Anthropic(**_client_kwargs())
