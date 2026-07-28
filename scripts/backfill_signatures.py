#!/usr/bin/env python3
"""
backfill_signatures.py — One-time script to populate the Signature Moments section
in existing playthrough files from their historical session entries.

Usage:
    python backfill_signatures.py

Safe to re-run — fully overwrites the Signature Moments section each time.
Requires a Claude route in .env (OPENROUTER_API_KEY — same key the bot uses).
"""
import asyncio
import os
import sys

# Repo root on path so first-party imports resolve when run as scripts/backfill_signatures.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config.py is the single dotenv source — importing it loads .env and exposes secrets.
from kira.config import CLAUDE_SONNET_MODEL as _CHAT_MODEL
from kira.brain.claude_gateway import claude_key_present, make_async_claude_client

# Games to backfill: (display name passed to load_for_game, slug used in filename)
GAMES = [
    ("007 First Light", "007_first_light"),
    ("planetarian",     "planetarian"),
]


class _MinimalAI:
    """Minimal stub — exposes only what PlaythroughMemory.backfill_signature_moments uses.
    Avoids importing ai_core (which pulls in pygame, faster-whisper, etc.)."""

    def __init__(self):
        if not claude_key_present():
            raise RuntimeError(
                "No Claude route: set OPENROUTER_API_KEY (preferred) or "
                "ANTHROPIC_API_KEY in .env and try again."
            )
        self.anthropic_client = make_async_claude_client()

    async def claude_chat_inference(
        self,
        messages: list,
        system_prompt: str,
        dynamic_context: str = "",
        max_tokens: int = 400,
    ) -> str:
        response = await self.anthropic_client.messages.create(
            model=_CHAT_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return ""


async def main():
    from kira.memory.playthrough_memory import PlaythroughMemory

    ai = _MinimalAI()
    pm = PlaythroughMemory(ai_core=ai)

    for display, slug in GAMES:
        path = os.path.join("playthroughs", f"{slug}.md")
        if not os.path.exists(path):
            print(f"\n[backfill] Skipping '{display}' — file not found: {path}")
            continue

        print(f"\n[backfill] ── {display} ──────────────────────")
        pm.load_for_game(display)
        ok = await pm.backfill_signature_moments(slug=slug)
        if ok:
            print(f"[backfill] '{display}' — done.")
        else:
            print(f"[backfill] '{display}' — nothing written (no extractable reactions?).")

    print("\n[backfill] All done.")


if __name__ == "__main__":
    asyncio.run(main())
