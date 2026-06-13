#!/usr/bin/env python3
"""stitch_clips.py — Stitch individual clip files into a reel or recap video.

Two modes:
  --reel    Tight highlight (3-4 min). Keeps top-scoring clips up to the target
            runtime, then re-orders them chronologically for output.
  --recap   Long-form digest (10-12 min, default RECAP_TARGET_MINUTES).
            Takes ALL clips sorted chronologically and includes everything up
            to the runtime cap, dropping lowest-score clips if over budget.

Input clips are read from either:
  --clips-dir   directory of .mp4 files (sorted alphabetically = chronological
                when named clip01_, clip02_, etc.)
  --manifest    the JSON cuts manifest (cuts_*.json) — uses id, title, score,
                vod_anchor_ts for ordering/filtering. Clips must already exist
                in --clips-dir.

Output:
  reels/YYYY-MM-DD/REEL_<activity>.mp4    for --reel
  recaps/YYYY-MM-DD/RECAP_<activity>.mp4  for --recap

Usage examples:
  # Stitch tonight's 9 general clips into a reel:
  python scripts/stitch_clips.py \\
      --clips-dir "G:/OBS/clips/2026-06-12" \\
      --out-base "G:/OBS" \\
      --date 2026-06-12 --activity general \\
      --reel

  # Build a 12-minute 007 recap from manifest:
  python scripts/stitch_clips.py \\
      --clips-dir "G:/OBS/clips/2026-06-11" \\
      --manifest scripts/cuts_007.json \\
      --out-base "G:/OBS" \\
      --date 2026-06-11 --activity 007_first_light \\
      --recap --recap-minutes 12

  # Build just the tight reel from the same manifest:
  python scripts/stitch_clips.py \\
      --clips-dir "G:/OBS/clips/2026-06-11" \\
      --manifest scripts/cuts_007.json \\
      --out-base "G:/OBS" \\
      --date 2026-06-11 --activity 007_first_light \\
      --reel --reel-minutes 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REEL_TARGET_MINUTES = 4.0
RECAP_TARGET_MINUTES = 12.0
REEL_MIN_SCORE = 7.5  # clips below this score skipped in reel mode


def get_clip_duration(path: str) -> float:
    """Return clip duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return 0.0
    return float(json.loads(r.stdout)["format"]["duration"])


def stitch(clip_paths: list[str], out_path: str, dry_run: bool = False) -> bool:
    """Concatenate clips using ffmpeg concat demuxer (re-encode for compatibility)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'\n")
        concat_file = f.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
        if dry_run:
            print("  DRY RUN:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
            print(f"  Would stitch {len(clip_paths)} clips → {out_path}")
            return True
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print(f"[ERROR] ffmpeg concat failed:")
            print(r.stderr.decode()[-600:])
            return False
        return True
    finally:
        try:
            os.unlink(concat_file)
        except Exception:
            pass


def ts_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips-dir", required=True, help="Directory containing individual .mp4 clips")
    p.add_argument("--manifest", default=None, help="JSON cuts manifest (for score/ordering info)")
    p.add_argument("--out-base", required=True, help="Base output directory (reels/ and recaps/ created here)")
    p.add_argument("--date", required=True, help="Stream date YYYY-MM-DD (used in output path)")
    p.add_argument("--activity", required=True, help="Activity label (e.g. general, 007_first_light)")
    p.add_argument("--reel", action="store_true", help="Produce tight reel (top moments, ~3-4 min)")
    p.add_argument("--recap", action="store_true", help="Produce long recap (all moments, ~10-12 min)")
    p.add_argument("--reel-minutes", type=float, default=REEL_TARGET_MINUTES)
    p.add_argument("--recap-minutes", type=float, default=RECAP_TARGET_MINUTES)
    p.add_argument("--reel-min-score", type=float, default=REEL_MIN_SCORE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.reel and not args.recap:
        sys.exit("[ERROR] Specify at least one of --reel or --recap")

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        sys.exit(f"[ERROR] Clips dir not found: {clips_dir}")

    # Load all .mp4 clips from the directory
    all_clips = sorted(clips_dir.glob("*.mp4"), key=lambda x: x.name)
    if not all_clips:
        sys.exit(f"[ERROR] No .mp4 clips found in {clips_dir}")

    print(f"Found {len(all_clips)} clip(s) in {clips_dir}")

    # Build clip info list: (filename, score, vod_seconds)
    # Start with defaults: score=8.0 for all, order by filename
    # Deduplicate by id — keep the newest file when multiple clips share the same id prefix
    seen_ids: dict[int, Path] = {}
    for cp in all_clips:
        m = re.match(r"clip(\d+)_", cp.name)
        clip_id = int(m.group(1)) if m else 999
        if clip_id not in seen_ids or cp.stat().st_mtime > seen_ids[clip_id].stat().st_mtime:
            seen_ids[clip_id] = cp

    clip_info: list[dict] = []
    for clip_id, cp in seen_ids.items():
        dur = get_clip_duration(str(cp))
        clip_info.append({
            "path": str(cp),
            "name": cp.name,
            "id": clip_id,
            "score": 8.0,
            "duration": dur,
            "vod_seconds": clip_id * 100,  # fallback ordering
        })

    # Override with manifest data if available; restrict to manifest IDs only
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        score_map = {int(c["id"]): float(c.get("score", 8.0)) for c in manifest}
        ts_map = {int(c["id"]): ts_to_seconds(c.get("vod_anchor_ts", "00:00:00")) for c in manifest}
        # Drop any clips whose ID isn't in the manifest (avoids picking up old clips)
        clip_info = [ci for ci in clip_info if ci["id"] in score_map]
        for ci in clip_info:
            ci["score"] = score_map[ci["id"]]
            ci["vod_seconds"] = ts_map[ci["id"]]
        print(f"  Loaded scores/timestamps from manifest: {args.manifest}")
        print(f"  Restricted to {len(clip_info)} manifest clip(s)")

    # Sort chronologically by VOD position
    clip_info.sort(key=lambda c: c["vod_seconds"])

    total_dur = sum(c["duration"] for c in clip_info)
    print(f"Total clip duration: {total_dur/60:.1f} min  ({len(clip_info)} clips)")
    print()

    def make_output(mode: str, target_min: float, selected: list[dict]) -> None:
        out_dir = Path(args.out_base) / f"{mode}s" / args.date
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{mode.upper()}_{args.activity}.mp4")

        total = sum(c["duration"] for c in selected)
        titles = [f"  [{c['id']}] {Path(c['name']).stem}  ({c['duration']:.0f}s, score={c['score']})"
                  for c in selected]
        print(f"=== {mode.upper()} ({target_min:.0f}-min target) ===")
        for t in titles:
            print(t)
        print(f"Total runtime: {total/60:.1f} min  ({len(selected)} clips)")
        print(f"Output: {out_path}")

        paths = [c["path"] for c in selected]
        ok = stitch(paths, out_path, dry_run=args.dry_run)
        if ok and not args.dry_run:
            size_mb = Path(out_path).stat().st_size / 1_048_576
            print(f"✓ {mode.upper()} written  ({size_mb:.0f} MB)")
        print()

    if args.reel:
        target_s = args.reel_minutes * 60
        # Filter by min score, already in chronological order
        eligible = [c for c in clip_info if c["score"] >= args.reel_min_score]
        # Sort by score desc to fill budget with best clips, then re-sort chrono
        eligible_by_score = sorted(eligible, key=lambda c: -c["score"])
        selected = []
        runtime = 0.0
        for c in eligible_by_score:
            if runtime + c["duration"] <= target_s + 30:  # 30s grace
                selected.append(c)
                runtime += c["duration"]
        # Re-sort chronologically
        selected.sort(key=lambda c: c["vod_seconds"])
        make_output("reel", args.reel_minutes, selected)

    if args.recap:
        target_s = args.recap_minutes * 60
        # All clips chronologically (exclude explicit score=0 skip markers),
        # then drop lowest-score ones if over budget
        selected = [c for c in clip_info if c["score"] > 0]
        total = sum(c["duration"] for c in selected)
        if total > target_s + 60:  # 60s grace
            # Drop lowest-score clips until under budget
            selected_sorted_score = sorted(selected, key=lambda c: c["score"])
            while sum(c["duration"] for c in selected) > target_s + 60 and selected_sorted_score:
                to_drop = selected_sorted_score.pop(0)
                selected = [c for c in selected if c["id"] != to_drop["id"]]
            selected.sort(key=lambda c: c["vod_seconds"])
        make_output("recap", args.recap_minutes, selected)


if __name__ == "__main__":
    main()
