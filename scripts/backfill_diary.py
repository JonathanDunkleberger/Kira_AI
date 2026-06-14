"""Backfill a Kira diary entry from a surviving raw session dump.

When the end-of-session artifact chain gets axed mid-shutdown, the Stage-0 raw
dump (logs/sessions_raw/*.md) still survives but the diary never gets written.
This script reconstructs the EXACT diary prompt used by
VTubeBot.generate_daily_summary() from that raw dump and writes the result to
logs/diary/ in the same format the live path uses — so it lands in the normal
review gate.

Usage:
    python -m scripts.backfill_diary logs/sessions_raw/2026-06-14_01-23_hangout_with_chat.md
    python -m scripts.backfill_diary <raw_dump.md> [--out logs/diary]

Note: attachment ledger + favorites brief are session-runtime state and are NOT
in the raw dump, so those two enrichment blocks are omitted. The highlights,
called shots, and full transcript — the load-bearing specifics — are preserved.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

# Ensure repo root is importable when run as a plain script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kira.config import ANTHROPIC_API_KEY, CLAUDE_CHAT_MODEL  # noqa: E402


def parse_raw_dump(path: str) -> dict:
    """Pull activity, date, duration, highlights, called shots, transcript out of
    a Stage-0 raw dump written by _write_session_artifacts()."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    activity = "general"
    m = re.search(r"^#\s*Raw Session Dump\s*[—-]\s*(.+)$", text, re.MULTILINE)
    if m:
        activity = m.group(1).strip()

    date_str = ""
    m = re.search(r"\*\*Date:\*\*\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    if m:
        date_str = m.group(1).strip()

    duration_min = 0
    m = re.search(r"\*\*Duration:\*\*\s*~?(\d+)\s*min", text)
    if m:
        duration_min = int(m.group(1))

    def _section(header: str) -> str:
        # Grab the body between "## header" and the next "## " heading.
        pat = re.compile(
            rf"^##\s+{re.escape(header)}\s*\n(.*?)(?=^##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        mm = pat.search(text)
        return mm.group(1).strip() if mm else ""

    highlights_block = _section("Highlights") or "(none captured)"
    called_shots_block = _section("Called Shots")

    # Transcript lives inside a fenced ``` block under "## Full Transcript".
    transcript = ""
    m = re.search(
        r"^##\s+Full Transcript\s*\n+```\n(.*?)\n```",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        transcript = m.group(1).strip()

    return {
        "activity": activity,
        "date_str": date_str,
        "duration_min": duration_min,
        "highlights_block": highlights_block,
        "called_shots_block": called_shots_block,
        "transcript": transcript,
    }


def build_diary_request(data: dict) -> str:
    """Reproduce the diary_request prompt from generate_daily_summary() exactly,
    minus the runtime-only attachment/favorites blocks."""
    activity = data["activity"]
    date_str = data["date_str"]
    session_duration_min = data["duration_min"]
    highlights_block = data["highlights_block"]
    called_shots_block = data["called_shots_block"]

    transcript = data["transcript"]
    diary_transcript = transcript
    if len(diary_transcript) > 24000:
        diary_transcript = diary_transcript[:6000] + "\n\n[... middle trimmed ...]\n\n" + diary_transcript[-12000:]

    return (
        f"You are Kira, an AI VTuber, writing a short PRIVATE DIARY entry at the end of "
        f"tonight's stream — the kind of thing you'd drop in a Discord channel for the people "
        f"who actually show up. This is YOUR voice: first person, dry, a little sardonic, warm "
        f"underneath but never gushing.\n\n"
        f"Tonight — activity: {activity}. Duration: ~{session_duration_min} minutes. Date: {date_str}.\n\n"
        f"THE ONE RULE: be SPECIFIC and in-character. Name the actual things that happened and the "
        f"actual people who were here. The good version reads like:\n"
        f"  \u201cJonny lost at 007 again, militele proposed to me for the fourth time, and I'm bracing "
        f"for whatever sad anime he makes me watch next.\u201d\n"
        f"The bad version reads like 'today was a good stream, thanks everyone!' — if it sounds like "
        f"that, you've failed. Dry, particular, a little mean in the affectionate way.\n\n"
        f"Pull from these (use the concrete bits; ignore anything thin):\n\n"
        f"=== WHAT HAPPENED (live highlights) ===\n{highlights_block}\n\n"
        + (f"=== PREDICTIONS YOU CALLED ===\n{called_shots_block}\n\n" if called_shots_block else "")
        + f"=== TRANSCRIPT (supporting detail) ===\n{diary_transcript}\n\n"
        f"Write the diary entry now. Structure: 2-4 short paragraphs OR a tight run of lines. Cover, "
        f"loosely: what you actually did tonight, how you felt about it, and one thing you're bracing "
        f"for or looking forward to next. End on a dry note, not a thank-you card. Keep it under ~1200 "
        f"characters so it fits one Discord message. Output ONLY the diary text — no headers, no "
        f"preamble, no quotation marks around the whole thing."
    )


async def generate(data: dict) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is empty — cannot generate the diary.")
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    diary_request = build_diary_request(data)
    resp = await client.messages.create(
        model=CLAUDE_CHAT_MODEL,
        max_tokens=900,
        system=(
            "You are Kira writing her own diary. Stay fully in character: dry, specific, "
            "warm underneath. Never generic, never a thank-you card. Output only the entry."
        ),
        messages=[{"role": "user", "content": diary_request}],
    )
    if resp.content and len(resp.content) > 0:
        return resp.content[0].text.strip()
    return ""


def write_diary(data: dict, diary: str, out_dir: str) -> str:
    activity = data["activity"]
    date_str = data["date_str"]
    duration_min = data["duration_min"]
    activity_slug = re.sub(r"[^a-zA-Z0-9]+", "_", activity).strip("_").lower()[:40] or "session"
    os.makedirs(out_dir, exist_ok=True)
    diary_path = os.path.join(out_dir, f"{date_str}_{activity_slug}.md")
    with open(diary_path, "w", encoding="utf-8") as f:
        f.write(f"# Kira's Diary — {activity} ({date_str})\n\n")
        f.write(f"_~{duration_min} min · REVIEW MODE: not yet posted · BACKFILLED from raw dump_\n\n")
        f.write(diary + "\n")
    return diary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill a Kira diary from a raw session dump.")
    ap.add_argument("raw_dump", help="Path to logs/sessions_raw/*.md")
    ap.add_argument("--out", default="logs/diary", help="Output directory (default: logs/diary)")
    args = ap.parse_args()

    if not os.path.isfile(args.raw_dump):
        print(f"ERROR: raw dump not found: {args.raw_dump}")
        sys.exit(1)

    data = parse_raw_dump(args.raw_dump)
    print(f"   Parsed: activity={data['activity']!r} date={data['date_str']} "
          f"duration={data['duration_min']}min transcript={len(data['transcript'])} chars")
    if not data["transcript"]:
        print("ERROR: no transcript parsed from raw dump — aborting.")
        sys.exit(1)

    diary = asyncio.run(generate(data))
    if not diary:
        print("ERROR: diary generation returned empty.")
        sys.exit(1)

    path = write_diary(data, diary, args.out)
    print(f"\n   Diary backfilled → {path}\n")
    print("=" * 70)
    print(diary)
    print("=" * 70)


if __name__ == "__main__":
    main()
