#!/usr/bin/env python3
"""cut_from_manifest.py — Cut clips from a VOD using a JSON timestamp manifest.

Each entry in the JSON has:
  id, title, vod_anchor_ts (HH:MM:SS), pre (seconds before anchor), post (after)

Usage:
    python scripts/cut_from_manifest.py \\
        --vod "G:/path/to/stream.mp4" \\
        --manifest scripts/cuts_2026-06-12.json \\
        --out "G:/path/to/clips/2026-06-12"

Flags:
    --reencode   Re-encode with H.264 instead of stream copy (fixes seeking/concat)
    --ids 1,4,7  Only cut these clip IDs (comma-separated)
    --dry-run    Print ffmpeg commands without executing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or H:MM:SS to float seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def safe_filename(title: str) -> str:
    return re.sub(r'[^\w\- ]', '', title).strip().replace(' ', '_')[:60]


def cut_clip(
    vod_path: str,
    out_path: str,
    start_s: float,
    duration_s: float,
    reencode: bool,
    dry_run: bool,
) -> bool:
    start_s = max(0.0, start_s)
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", vod_path,
            "-t", f"{duration_s:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", vod_path,
            "-t", f"{duration_s:.3f}",
            "-c", "copy",
            out_path,
        ]
    if dry_run:
        print("  DRY RUN:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return True
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  [ERROR] ffmpeg exit {r.returncode}")
        print(r.stderr.decode()[-400:])
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vod", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="Output directory for clips")
    p.add_argument("--reencode", action="store_true")
    p.add_argument("--ids", default=None, help="Comma-separated clip IDs to cut")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.vod):
        sys.exit(f"[ERROR] VOD not found: {args.vod}")

    with open(args.manifest, encoding="utf-8") as f:
        clips = json.load(f)

    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        clips = [c for c in clips if c["id"] in wanted]
        if not clips:
            sys.exit(f"[ERROR] No clips match IDs {wanted}")

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cutting {len(clips)} clip(s) from: {Path(args.vod).name}")
    print(f"Output → {out_dir}\n")

    ok = 0
    for clip in clips:
        anchor_s = ts_to_seconds(clip["vod_anchor_ts"])
        pre = float(clip.get("pre", 10))
        post = float(clip.get("post", 5))
        start_s = anchor_s - pre
        duration_s = pre + post

        fname = f"clip{clip['id']:02d}_{safe_filename(clip['title'])}.mp4"
        out_path = str(out_dir / fname)

        from_ts = f"{int(start_s//3600):02d}:{int((start_s%3600)//60):02d}:{start_s%60:05.2f}"
        to_ts_s = start_s + duration_s
        to_ts = f"{int(to_ts_s//3600):02d}:{int((to_ts_s%3600)//60):02d}:{to_ts_s%60:05.2f}"
        print(f"[{clip['id']}] {clip['title']}")
        print(f"     {from_ts} → {to_ts}  ({duration_s:.0f}s)")
        if clip.get("note"):
            print(f"     ({clip['note']})")

        success = cut_clip(args.vod, out_path, start_s, duration_s, args.reencode, args.dry_run)
        if success:
            if not args.dry_run:
                size_mb = Path(out_path).stat().st_size / 1_048_576
                print(f"     ✓ {fname}  ({size_mb:.1f} MB)")
            ok += 1
        print()

    print(f"Done: {ok}/{len(clips)} clips cut.")


if __name__ == "__main__":
    main()
