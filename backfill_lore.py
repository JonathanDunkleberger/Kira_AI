#!/usr/bin/env python3
"""backfill_lore.py — Regenerate missing lore entries from stream transcripts.

For every logs/streams/{date}_{time}_{slug}/transcript.md that represents a
session newer than the last entry already present in lore/{slug}.md, this script
calls Claude (claude-opus-4-7 by default) with the same lore-extraction prompt
used by _write_session_artifacts and appends the result.

It also picks up any PENDING_{slug}.json checkpoint files left by the
_checkpoint_loop (crash survivors) and processes those first.

Run from the repo root:

    python backfill_lore.py                          # all slugs, after last lore date
    python backfill_lore.py --dry-run                # preview what would be written
    python backfill_lore.py --slug 007_first_light   # one slug only
    python backfill_lore.py --after 2026-06-02       # explicit date cutoff
    python backfill_lore.py --pending-only           # only process PENDING_*.json files

Writes are appended to lore/{slug}.md with a "(backfilled)" marker on the
session header so they're distinguishable from live writes.
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

from config import ANTHROPIC_API_KEY, CLAUDE_SONNET_MODEL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def last_lore_date(slug: str) -> str:
    """Return the date string of the most recent '## Session:' entry in lore/{slug}.md,
    or '0000-00-00' if the file is absent or has no entries."""
    path = os.path.join("lore", f"{slug}.md")
    if not os.path.exists(path):
        return "0000-00-00"
    last = "0000-00-00"
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^## Session: (\d{4}-\d{2}-\d{2})", line)
            if m and m.group(1) > last:
                last = m.group(1)
    return last


def estimate_duration(transcript: str) -> int:
    """Estimate session duration in minutes from the first/last HH:MM:SS timestamps."""
    times = re.findall(r"\[(\d{2}):(\d{2}):\d{2}\]", transcript)
    if len(times) < 2:
        return 0
    h0, m0 = int(times[0][0]), int(times[0][1])
    h1, m1 = int(times[-1][0]), int(times[-1][1])
    return max((h1 * 60 + m1) - (h0 * 60 + m0), 1)


def build_lore_prompt(activity: str, date_str: str, duration_min: int,
                      transcript: str, highlights: list[str]) -> str:
    """Construct the same prompt _write_session_artifacts uses for lore extraction."""
    # Truncate transcript the same way the live code does
    llm_transcript = transcript
    if len(llm_transcript) > 80000:
        llm_transcript = (
            llm_transcript[:16000]
            + "\n\n[... middle of session truncated for length ...]\n\n"
            + llm_transcript[-40000:]
        )

    highlights_block = (
        "\n".join(f"- {h}" for h in highlights) if highlights else "(none captured)"
    )

    return (
        f"You are reviewing a full stream session transcript for the AI VTuber Kira. "
        f"Activity: {activity}. Duration: ~{duration_min} minutes. Date: {date_str}.\n\n"
        f"You will produce TWO outputs, separated by the exact delimiter line `===CLIPS===`.\n\n"
        f"OUTPUT 1 — LORE NOTES (markdown). Identify 3-7 durable canon points established or developed "
        f"this session for this activity. Format as bullet points.\n\n"
        f"OUTPUT 2 — CLIP CANDIDATES (markdown). Identify 8-12 of the funniest, sharpest, or most "
        f"emotionally landing moments. For each one provide:\n"
        f"  ### Clip N — Short title\n"
        f"  **Timestamp:** approximate HH:MM:SS into stream\n"
        f"  **Why it's good:** 1-2 sentences\n"
        f"  **Suggested YouTube short title:** under 60 chars\n"
        f"  **Key exchange:** 2-4 quoted lines\n\n"
        f"=== TRANSCRIPT ===\n{llm_transcript}\n\n"
        f"=== HIGHLIGHTS CAPTURED LIVE ===\n{highlights_block}\n\n"
        f"Begin output. Lore first, then `===CLIPS===` on its own line, then clip candidates."
    )


def append_lore(slug: str, activity: str, date_str: str, duration_min: int,
                lore_text: str, dry_run: bool) -> str:
    """Append a lore section to lore/{slug}.md.  Returns the path written."""
    lore_path = os.path.join("lore", f"{slug}.md")
    header = f"\n\n## Session: {date_str} (~{duration_min} min, backfilled)\n\n"
    if dry_run:
        print(f"  [dry-run] Would append to {lore_path}:")
        print(f"  {header.strip()}")
        for line in lore_text.splitlines()[:6]:
            print(f"    {line}")
        if lore_text.count("\n") > 5:
            print(f"    ... ({lore_text.count(chr(10))} lines total)")
        return lore_path

    os.makedirs("lore", exist_ok=True)
    with open(lore_path, "a", encoding="utf-8") as f:
        if os.path.getsize(lore_path) == 0:
            f.write(f"# Lore: {activity}\n")
        f.write(header)
        f.write(lore_text)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return lore_path


def write_clips(slug: str, activity: str, date_str: str, duration_min: int,
                clips_text: str, dry_run: bool) -> str | None:
    """Write a clips file to clips/{date}_{slug}.md if it doesn't already exist."""
    clip_path = os.path.join("clips", f"{date_str}_{slug}.md")
    if os.path.exists(clip_path):
        return None  # already written by live code or a previous backfill run
    if dry_run:
        print(f"  [dry-run] Would write {clip_path}")
        return clip_path

    os.makedirs("clips", exist_ok=True)
    with open(clip_path, "w", encoding="utf-8") as f:
        f.write(f"# Clip Candidates — {activity}\n\n")
        f.write(f"**Date:** {date_str}  \n")
        f.write(f"**Duration:** ~{duration_min} minutes  \n")
        f.write(f"**Activity:** {activity} (backfilled)\n\n---\n\n")
        f.write(clips_text)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return clip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool, only_slug: str | None,
               after_date: str | None, pending_only: bool) -> None:
    api_key = ANTHROPIC_API_KEY  # loaded via config (single dotenv source)
    model = CLAUDE_SONNET_MODEL  # Sonnet — matches live lore path
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set — cannot call Claude.")
        sys.exit(1)

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = AsyncAnthropic(api_key=api_key)

    # ── Build work list ───────────────────────────────────────────────────────
    # Each entry: (date_str, slug, transcript_text, highlights, duration_min, source_label)
    work: list[tuple[str, str, str, list[str], int, str]] = []
    processed_slugs_dates: set[tuple[str, str]] = set()  # (slug, date) dedup

    # 1. PENDING_*.json checkpoint files (highest priority — most recent data)
    for pending_path in sorted(glob.glob("logs/sessions_raw/PENDING_*.json")):
        try:
            with open(pending_path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            slug = data.get("activity_slug", "")
            date_str = data.get("date", "")
            activity = data.get("activity", slug.replace("_", " "))
            transcript = data.get("transcript", "")
            highlights = data.get("highlights", [])
            duration_min = data.get("duration_min", estimate_duration(transcript))
            if not slug or not date_str or not transcript:
                continue
            if only_slug and slug != only_slug:
                continue
            if after_date and date_str <= after_date:
                continue
            key = (slug, date_str)
            if key not in processed_slugs_dates:
                work.append((date_str, slug, transcript, highlights, duration_min,
                              f"PENDING checkpoint ({pending_path})"))
                processed_slugs_dates.add(key)
        except Exception as e:
            print(f"[WARN] Could not read {pending_path}: {e}")

    # 2. Stream transcript.md files (covers sessions without a checkpoint file)
    if not pending_only:
        for d in sorted(os.listdir("logs/streams")):
            m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})_(.+)$", d)
            if not m:
                continue
            sess_date, _sess_time, slug = m.group(1), m.group(2), m.group(3)
            if only_slug and slug != only_slug:
                continue
            if after_date and sess_date <= after_date:
                continue
            key = (slug, sess_date)
            if key in processed_slugs_dates:
                continue  # already covered by a PENDING checkpoint
            t_path = os.path.join("logs/streams", d, "transcript.md")
            if not os.path.exists(t_path):
                continue
            transcript = open(t_path, encoding="utf-8", errors="replace").read()
            if len(transcript) < 200:
                continue  # too short to be a real session
            duration_min = estimate_duration(transcript)
            work.append((sess_date, slug, transcript, [], duration_min,
                          f"stream transcript ({t_path})"))
            processed_slugs_dates.add(key)

    if not work:
        print("Nothing to backfill.")
        return

    # ── Filter out sessions already in lore ──────────────────────────────────
    lore_dates: dict[str, str] = {}
    filtered: list = []
    for item in work:
        date_str, slug = item[0], item[1]
        if slug not in lore_dates:
            lore_dates[slug] = last_lore_date(slug)
        if date_str > lore_dates[slug]:
            filtered.append(item)

    if not filtered:
        print("All sessions already present in lore files — nothing to backfill.")
        return

    # Sort by date so entries are appended in chronological order
    filtered.sort(key=lambda x: x[0])

    print(f"{'[DRY RUN] ' if dry_run else ''}Backfilling {len(filtered)} session(s):\n")
    for date_str, slug, _, _, dur, source in filtered:
        print(f"  {date_str}  {slug:35s}  ~{dur}min  [{source}]")
    print()

    # ── Process each session ─────────────────────────────────────────────────
    for date_str, slug, transcript, highlights, duration_min, source in filtered:
        activity = slug.replace("_", " ")
        print(f"── {date_str} {slug} (~{duration_min}min) ──")
        print(f"   source: {source}")

        prompt = build_lore_prompt(activity, date_str, duration_min, transcript, highlights)

        response = None
        for attempt in range(1, 3):  # up to 2 attempts
            try:
                resp = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=4000,
                        system=(
                            "You are a thoughtful editor reviewing a stream session. "
                            "Output clean markdown."
                        ),
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=90.0,
                )
                response = resp.content[0].text
                break
            except asyncio.TimeoutError:
                print(f"   [attempt {attempt}] Claude timed out after 90s.")
            except Exception as e:
                print(f"   [attempt {attempt}] Claude error: {e}")
                traceback.print_exc()
            if attempt < 2:
                await asyncio.sleep(5)

        if not response:
            print(f"   [SKIP] Could not get a response from Claude — skipping.\n")
            continue

        # Split lore / clips the same way the live code does
        if "===CLIPS===" in response:
            lore_section, clips_section = response.split("===CLIPS===", 1)
        else:
            lore_section, clips_section = response, ""
        lore_section = lore_section.strip()
        clips_section = clips_section.strip()

        if not lore_section or len(lore_section) < 20:
            print(f"   [SKIP] Claude returned empty lore section.\n")
            continue

        lore_path = append_lore(slug, activity, date_str, duration_min,
                                lore_section, dry_run)
        print(f"   [OK] Lore → {lore_path}")

        if clips_section and len(clips_section) > 50:
            clips_path = write_clips(slug, activity, date_str, duration_min,
                                     clips_section, dry_run)
            if clips_path:
                print(f"   [OK] Clips → {clips_path}")
            else:
                print(f"   [skip] Clips file already exists for {date_str} — not overwriting.")

        # Update the cached last-lore-date so later sessions for the same slug
        # don't get filtered out now that we've just written an entry
        if not dry_run:
            lore_dates[slug] = date_str

        print()

    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill missing lore entries from stream transcripts."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be written without touching any files."
    )
    parser.add_argument(
        "--slug", metavar="SLUG",
        help="Only backfill this activity slug (e.g. 007_first_light)."
    )
    parser.add_argument(
        "--after", metavar="YYYY-MM-DD",
        help="Only process sessions strictly after this date. "
             "Defaults to the date of the last lore entry for each slug."
    )
    parser.add_argument(
        "--pending-only", action="store_true",
        help="Only process PENDING_*.json checkpoint files, not stream transcripts."
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            dry_run=args.dry_run,
            only_slug=args.slug,
            after_date=args.after,
            pending_only=args.pending_only,
        )
    )
