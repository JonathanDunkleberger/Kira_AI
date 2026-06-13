#!/usr/bin/env python3
"""match_clips_to_vod.py — Match clip candidates from a markdown file to a
VOD transcript, then write a cuts manifest JSON ready for cut_from_manifest.py.

Given:
  - A clips/YYYY-MM-DD_activity.md file with key exchanges and session timestamps
  - A logs/vod_transcripts/<vod_name>.md file (output of transcribe_vod.py)

Outputs: scripts/cuts_YYYY-MM-DD_activity.json

Matching strategy:
  For each clip candidate, the punchline (last Kira line in the key exchange) is
  normalised (lowercase, punctuation stripped) and compared to every Whisper
  transcript line using word-overlap scoring. The top match gives the VOD anchor
  timestamp. A fallback uses the session-log timestamp (HH:MM:SS) to estimate a
  search window when overlap is weak.

Usage:
    python scripts/match_clips_to_vod.py \\
        --clips clips/2026-06-11_general.md \\
        --transcript logs/vod_transcripts/007_with_kira.md \\
        --out scripts/cuts_007.json \\
        --pre 22 --post 8

    # Dry run — show matches without writing JSON:
    python scripts/match_clips_to_vod.py \\
        --clips clips/2026-06-11_general.md \\
        --transcript logs/vod_transcripts/007_with_kira.md \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# --- helpers -----------------------------------------------------------

def ts_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_overlap(a: str, b: str) -> float:
    """Fraction of words in `a` that appear in `b`."""
    wa = set(normalise(a).split())
    wb = set(normalise(b).split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


# --- parsers -----------------------------------------------------------

def parse_clip_candidates(path: str) -> list[dict]:
    """Parse clips/YYYY-MM-DD_activity.md into a list of clip dicts."""
    clips = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Split on ### Clip N headings
    blocks = re.split(r"\n### Clip (\d+)", content)
    # blocks[0] = header, then alternating: id, block_text
    i = 1
    while i < len(blocks) - 1:
        clip_id = int(blocks[i])
        block = blocks[i + 1]

        title_m = re.search(r"^.*?—\s*(.+)", block.split("\n")[0])
        title = title_m.group(1).strip() if title_m else f"Clip {clip_id}"

        ts_m = re.search(r"\*\*Timestamp:\*\*\s*(\d+:\d+:\d+)", block)
        session_ts = ts_m.group(1) if ts_m else None

        score_m = re.search(r"\*\*Score:\*\*\s*([\d.]+)", block)
        score = float(score_m.group(1)) if score_m else 7.0

        # Key exchange lines (lines starting with ">")
        exchange_lines = re.findall(r"^>\s*(.+)", block, re.MULTILINE)
        # Punchline: last Kira line
        kira_lines = [l for l in exchange_lines if re.match(r"Kira:", l)]
        punchline = kira_lines[-1].replace("Kira:", "").strip().strip('"') if kira_lines else ""

        clips.append({
            "id": clip_id,
            "title": title,
            "score": score,
            "session_ts": session_ts,
            "punchline": punchline,
            "exchange": exchange_lines,
        })
        i += 2

    return clips


def parse_vod_transcript(path: str) -> list[dict]:
    """Parse logs/vod_transcripts/<name>.md into list of {ts_str, seconds, text}."""
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\[(\d+:\d+:\d+)\]\s*(.+)", line.strip())
            if m:
                ts_str = m.group(1)
                text = m.group(2).strip()
                segments.append({
                    "ts_str": ts_str,
                    "seconds": ts_to_seconds(ts_str),
                    "text": text,
                })
    return segments


# --- matching ----------------------------------------------------------

def find_best_match(
    punchline: str,
    session_ts: str | None,
    segments: list[dict],
    window_s: float = 600.0,
    min_score: float = 0.25,
) -> dict | None:
    """Return the transcript segment best matching the punchline.

    If session_ts is given, restrict search to ±window_s around the estimated
    VOD position (using the session offset). Without session_ts, search all.
    """
    if not punchline:
        return None

    # Determine search window
    if session_ts:
        session_s = ts_to_seconds(session_ts)
        candidates = [s for s in segments if abs(s["seconds"] - session_s) <= window_s]
        if not candidates:
            candidates = segments  # fallback: search all
    else:
        candidates = segments

    best_seg = None
    best_score = 0.0
    for seg in candidates:
        sc = word_overlap(punchline, seg["text"])
        if sc > best_score:
            best_score = sc
            best_seg = seg

    if best_score >= min_score:
        best_seg = dict(best_seg)
        best_seg["match_score"] = best_score
        return best_seg
    return None


def seconds_to_ts(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# --- main --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True, help="Path to clips/*.md file")
    p.add_argument("--transcript", required=True, help="Path to VOD transcript .md")
    p.add_argument("--out", default=None, help="Output JSON path (default: scripts/cuts_DATE_activity.json)")
    p.add_argument("--pre", type=float, default=22.0, help="Default pre-anchor seconds (default 22)")
    p.add_argument("--post", type=float, default=8.0, help="Default post-anchor seconds (default 8)")
    p.add_argument("--window", type=float, default=600.0, help="Search window ±s around session_ts")
    p.add_argument("--min-match", type=float, default=0.20, help="Min word-overlap score to accept match")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.clips):
        sys.exit(f"[ERROR] Clips file not found: {args.clips}")
    if not os.path.exists(args.transcript):
        sys.exit(f"[ERROR] Transcript not found: {args.transcript}")

    clips = parse_clip_candidates(args.clips)
    segments = parse_vod_transcript(args.transcript)
    print(f"Clips: {len(clips)}  |  Transcript segments: {len(segments)}")
    print()

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        clips_stem = Path(args.clips).stem  # e.g. 2026-06-11_general
        out_path = Path("scripts") / f"cuts_{clips_stem}.json"

    manifest = []
    unmatched = []

    for clip in clips:
        match = find_best_match(
            clip["punchline"],
            clip["session_ts"],
            segments,
            window_s=args.window,
            min_score=args.min_match,
        )

        if match:
            confidence = "HIGH" if match["match_score"] >= 0.5 else "LOW"
            print(f"[{clip['id']:2d}] {clip['title']}")
            print(f"      session_ts={clip['session_ts']}  →  VOD {match['ts_str']}  "
                  f"(overlap={match['match_score']:.2f} {confidence})")
            print(f"      Punchline: {clip['punchline'][:80]}")
            print(f"      Matched:   {match['text'][:80]}")
            print()

            manifest.append({
                "id": clip["id"],
                "title": clip["title"],
                "score": clip["score"],
                "vod_anchor_ts": match["ts_str"],
                "pre": args.pre,
                "post": args.post,
                "note": f"match_score={match['match_score']:.2f} ({confidence})",
            })
        else:
            print(f"[{clip['id']:2d}] {clip['title']}  — NO MATCH (session_ts={clip['session_ts']})")
            print(f"      Punchline: {clip['punchline'][:80]}")
            print()
            unmatched.append(clip)

    print(f"Matched: {len(manifest)}/{len(clips)}")
    if unmatched:
        print(f"Unmatched: {[c['id'] for c in unmatched]}")
        print("  → These clips may be from sessions not in the VOD. Add manually if needed.")

    if args.dry_run:
        print("\n[DRY RUN] Not writing JSON.")
        return

    manifest.sort(key=lambda c: ts_to_seconds(c["vod_anchor_ts"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written → {out_path}  ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
