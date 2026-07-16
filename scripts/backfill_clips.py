#!/usr/bin/env python3
"""backfill_clips.py — Regenerate missing clip-candidate files from stream transcripts.

Reads every logs/streams/{slug}/transcript.md for a given date range, combines
them into a single ordered transcript, and calls Claude (Sonnet) with the same
prompt _write_session_artifacts uses to produce the clips/*.md artifact.

Usage:
    # Regenerate for June 12 (all CDT sessions in VOD: 19:48 through 21:47)
    python scripts/backfill_clips.py --date 2026-06-12 --min-hour 19 --activity general

    # Explicit session globs
    python scripts/backfill_clips.py --date 2026-06-12 --sessions 19-48 20-08 20-41 21-02 21-15 21-47

    # Dry-run (shows which sessions would be used, estimated token count)
    python scripts/backfill_clips.py --date 2026-06-12 --min-hour 19 --dry-run

Writes output to:  clips/YYYY-MM-DD_<activity>.md
Backs up any existing file to: clips/YYYY-MM-DD_<activity>.md.bak
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Repo root on path so first-party imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kira.config import ANTHROPIC_API_KEY, CLAUDE_SONNET_MODEL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sessions(date: str, activity: str, min_hour: int, explicit_hours: list[str]) -> list[Path]:
    """Return sorted list of session dirs matching the criteria."""
    base = Path("logs/streams")
    if not base.exists():
        sys.exit(f"[ERROR] {base} does not exist — run from the repo root.")

    if explicit_hours:
        # explicit_hours are like ["19-48", "20-08", ...]
        dirs = []
        for hh in explicit_hours:
            pattern = f"{date}_{hh}_{activity}"
            candidates = list(base.glob(pattern))
            if not candidates:
                print(f"  [WARN] Session not found: {pattern}")
            dirs.extend(candidates)
        return sorted(dirs)

    # Auto-discover by date + min_hour + activity
    dirs = []
    for d in sorted(base.iterdir()):
        name = d.name  # e.g. 2026-06-12_19-48_general
        if not d.is_dir():
            continue
        if not name.startswith(date):
            continue
        if activity and not name.endswith(f"_{activity}"):
            continue
        # Extract hour from name like 2026-06-12_19-48_...
        m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{2})-\d{2}", name)
        if m:
            hour = int(m.group(1))
            if hour >= min_hour:
                dirs.append(d)
    return sorted(dirs)


def _read_transcript(session_dir: Path) -> str:
    t = session_dir / "transcript.md"
    if not t.exists():
        return ""
    return t.read_text(encoding="utf-8", errors="replace")


def _truncate_session(text: str, budget: int) -> str:
    """Keep first 30% + last 70% of budget to capture arc."""
    if len(text) <= budget:
        return text
    head = int(budget * 0.30)
    tail = budget - head
    return (
        text[:head]
        + f"\n\n[... {len(text) - budget} chars trimmed from middle of session ...]\n\n"
        + text[-tail:]
    )


def _build_combined_transcript(sessions: list[Path], total_budget: int = 80_000) -> str:
    """Combine session transcripts with proportional truncation."""
    transcripts = [(s, _read_transcript(s)) for s in sessions]
    transcripts = [(s, t) for s, t in transcripts if t.strip()]

    total_raw = sum(len(t) for _, t in transcripts)
    parts = []
    for session_dir, text in transcripts:
        session_budget = max(4000, int(total_budget * len(text) / total_raw))
        trimmed = _truncate_session(text, session_budget)
        name = session_dir.name
        parts.append(f"## SESSION: {name}\n\n{trimmed}")

    return "\n\n---\n\n".join(parts)


def _estimate_duration_min(sessions: list[Path]) -> int:
    """Sum durations across sessions from first/last timestamps."""
    total = 0
    for s in sessions:
        text = _read_transcript(s)
        times = re.findall(r"\[(\d{2}):(\d{2}):\d{2}\]", text)
        if len(times) >= 2:
            h0, m0 = int(times[0][0]), int(times[0][1])
            h1, m1 = int(times[-1][0]), int(times[-1][1])
            delta = (h1 * 60 + m1) - (h0 * 60 + m0)
            total += max(delta, 1)
    return max(total, 1)


def _build_prompt(activity: str, date_str: str, duration_min: int, combined_transcript: str, num_clips: int = 12) -> str:
    # D4 (Phase K): head + format spec shared with bot.py's session-end
    # artifacts via prompt_spec — one copy, no drift.
    from kira.clips.prompt_spec import candidate_prompt_head, CANDIDATE_PROMPT_TAIL
    return (
        candidate_prompt_head(activity, date_str, duration_min,
                              clip_count=str(num_clips), wide_net=True)
        + f"=== TRANSCRIPT (multiple sessions from the same stream, in chronological order) ===\n"
        + f"{combined_transcript}\n\n"
        + CANDIDATE_PROMPT_TAIL
    )


async def _call_claude(prompt: str, api_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        sys.exit("[ERROR] anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    print("  [Claude] Sending request (Sonnet)...")
    message = client.messages.create(
        model=CLAUDE_SONNET_MODEL,
        max_tokens=8000,
        system="You are a thoughtful editor reviewing a stream session. Output clean markdown.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _write_output(response: str, date_str: str, activity_slug: str, duration_min: int) -> None:
    if "===CLIPS===" in response:
        _lore_section, clips_section = response.split("===CLIPS===", 1)
    else:
        clips_section = response

    clips_section = clips_section.strip()
    if not clips_section or len(clips_section) < 50:
        print("  [WARN] Clips section too short — not writing.")
        return

    os.makedirs("clips", exist_ok=True)
    clip_path = f"clips/{date_str}_{activity_slug}.md"

    # Backup existing file if present
    if os.path.exists(clip_path):
        bak = clip_path + ".bak"
        os.replace(clip_path, bak)
        print(f"  [Backup] Existing file backed up → {bak}")

    with open(clip_path, "w", encoding="utf-8") as f:
        f.write(f"# Clip Candidates — {activity_slug}\n\n")
        f.write(f"**Date:** {date_str}  \n")
        f.write(f"**Duration:** ~{duration_min} minutes  \n")
        f.write(f"**Activity:** {activity_slug}\n\n---\n\n")
        f.write(clips_section)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"  [Done] Clip candidates written → {clip_path}")
    print(f"         ({len(clips_section)} chars, ~{clips_section.count('### Clip')} clips)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate missing clip-candidate file from transcripts.")
    parser.add_argument("--date", required=True, help="Date prefix e.g. 2026-06-12")
    parser.add_argument("--activity", default="general", help="Activity slug (default: general)")
    parser.add_argument("--min-hour", type=int, default=0, help="Skip sessions starting before this hour (local 24h)")
    parser.add_argument("--sessions", nargs="+", help="Explicit session hour-min suffixes e.g. 19-48 20-08")
    parser.add_argument("--budget", type=int, default=80_000, help="Max combined transcript chars for LLM (default 80000)")
    parser.add_argument("--num-clips", type=int, default=12, help="How many clip candidates to request from Claude (default 12)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only — don't call Claude or write files")
    args = parser.parse_args()

    activity_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.activity).strip("_").lower() or "session"
    date_str = args.date

    sessions = _find_sessions(date_str, activity_slug, args.min_hour, args.sessions or [])
    if not sessions:
        sys.exit(f"[ERROR] No sessions found for date={date_str} activity={activity_slug} min_hour={args.min_hour}")

    print(f"\nSessions to process ({len(sessions)}):")
    total_raw = 0
    for s in sessions:
        t = _read_transcript(s)
        print(f"  {s.name}  ({len(t):,} chars)")
        total_raw += len(t)
    print(f"\nTotal raw transcript: {total_raw:,} chars  →  budget: {args.budget:,} chars")

    if args.dry_run:
        print("\n[DRY RUN] Would combine and send to Claude. Exiting.")
        return

    if not ANTHROPIC_API_KEY:
        sys.exit("[ERROR] ANTHROPIC_API_KEY not set in .env")

    combined = _build_combined_transcript(sessions, total_budget=args.budget)
    duration_min = _estimate_duration_min(sessions)
    prompt = _build_prompt(activity_slug, date_str, duration_min, combined, num_clips=args.num_clips)

    print(f"\nCombined transcript: {len(combined):,} chars  |  Estimated duration: {duration_min} min")
    print(f"Calling Claude {CLAUDE_SONNET_MODEL}...\n")

    response = asyncio.run(_call_claude(prompt, ANTHROPIC_API_KEY))

    print(f"  [Claude] Response received ({len(response):,} chars)")
    _write_output(response, date_str, activity_slug, duration_min)


if __name__ == "__main__":
    main()
