#!/usr/bin/env python
"""Cut & title clips from a stream's clip-candidate artifacts.

Run the morning after a stream — OBS recorded locally, Kira wrote her
clip candidates at session end, this turns them into ready-to-post,
pre-titled .mp4 files under OBS_RECORDINGS_DIR/clips/YYYY-MM-DD/.

Default run produces BOTH ranked individual clips AND a highlight reel.
Use --no-reel or --no-clips to suppress either output.

Usage:
    python scripts/cut_clips.py                         # today: clips + reel
    python scripts/cut_clips.py --date 2026-06-11
    python scripts/cut_clips.py --activity general
    python scripts/cut_clips.py --reencode              # force NVENC for individual clips too
    python scripts/cut_clips.py --dry-run               # plan only, cut nothing
    python scripts/cut_clips.py --no-reel               # individual clips only (no reel)
    python scripts/cut_clips.py --no-clips              # reel only (no individual clip sidecars)
    python scripts/cut_clips.py --vod "path/to/download.mp4"
        # use a specific VOD file; Whisper anchor-matches to find stream-start epoch.

Requires OBS_RECORDINGS_DIR in .env and ffmpeg/ffprobe on PATH.
--vod additionally requires faster-whisper (pip install faster-whisper).
"""
import argparse
import asyncio
import os
import sys
from datetime import datetime

# Repo root on path + as CWD so first-party imports and clips/ , logs/ resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kira.clips.clip_cutter import cut_session


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cut & title clips from session artifacts.")
    p.add_argument("--date", default="today",
                   help="Date to process: 'today' (default) or YYYY-MM-DD (local date).")
    p.add_argument("--activity", default=None,
                   help="Only this activity slug (e.g. 'general'). Default: all that day.")
    p.add_argument("--pre", type=float, default=None,
                   help="Seconds of lead-in before the moment (default from config).")
    p.add_argument("--post", type=float, default=None,
                   help="Seconds of payoff after the moment (default from config).")
    p.add_argument("--reencode", action="store_true",
                   help="Force frame-accurate re-encode (NVENC) instead of trying stream-copy.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan and align only; do not cut, title, or write files.")
    p.add_argument("--top", type=int, default=15, metavar="N",
                   help="Maximum individual clips to cut per activity (default 15). "
                        "Has no effect when reel is also being built (all candidates cut).")
    p.add_argument("--no-reel", action="store_true",
                   help="Skip highlight-reel assembly; produce ranked individual clips only.")
    p.add_argument("--no-clips", action="store_true",
                   help="Skip individual clip title/sidecar generation; produce reel only.")
    p.add_argument("--legacy-layout", action="store_true",
                   help="Use the old flat layout (ranked clips + one chronological reel) "
                        "instead of the Phase-4 four labeled outputs (clips-by-type / best-of "
                        "reel / highlight VOD with cold-open / short candidate).")
    p.add_argument("--vod", default=None, metavar="PATH",
                   help="Path to a VOD file (e.g. YouTube download) that lacks a reliable "
                        "container creation_time. Whisper anchor-matching will derive the "
                        "actual stream-start epoch from the events.jsonl quotes. "
                        "When omitted, OBS_RECORDINGS_DIR is scanned as normal.")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    date_str = datetime.now().strftime("%Y-%m-%d") if args.date == "today" else args.date

    do_reel  = not args.no_reel
    do_clips = not args.no_clips
    # Reel requires reencode internally; --reencode additionally forces it for
    # individual clips in --no-reel mode.
    force_reencode = args.reencode

    vod_path = None
    if args.vod:
        vod_path = os.path.abspath(args.vod)
        if not os.path.isfile(vod_path):
            print(f"\nERROR: --vod file not found: {vod_path}")
            return 1

    try:
        result = await cut_session(
            date_str=date_str,
            activity=args.activity,
            pre=args.pre,
            post=args.post,
            force_reencode=force_reencode,
            dry_run=args.dry_run,
            top=args.top,
            reel=do_reel,
            clips=do_clips,
            vod_path=vod_path,
            phase4=not args.legacy_layout,
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return 1

    print("\n" + "=" * 60)
    print(f"CLIP CUT SUMMARY — {result['date']}{' [DRY RUN]' if args.dry_run else ''}")
    print("=" * 60)
    total_cut = 0
    for s in result["sessions"]:
        capped_str = f"  capped={s['capped']}" if s.get('capped') else ""
        print(f"  {s['activity']:20s}  candidates={s['total']}  "
              f"cut={s['cut']}{capped_str}  missed={s['missed']}  errors={s['error']}")
        if s.get('capped'):
            print(f"      (re-run with --top {s['total']} to cut all, or higher --top for more)")
        print(f"      out: {s['out_dir']}")
        print(f"      report: {s['report']}")
        if s.get("reel_path"):
            label = "BEST-OF REEL" if not args.legacy_layout else "REEL"
            print(f"      {label}: {s['reel_path']}")
            if args.legacy_layout:
                print(f"      reel report: {s.get('reel_report', '')}")
        if s.get("highlight_path"):
            print(f"      HIGHLIGHT VOD: {s['highlight_path']}")
        if s.get("short_paths"):
            for sp in s["short_paths"]:
                print(f"      SHORT (9:16, captioned): {sp}")
        elif s.get("short_path"):
            print(f"      SHORT CANDIDATE: {s['short_path']}")
        total_cut += s["cut"]
    if not result["sessions"]:
        print("  (no artifacts processed)")
    if do_reel and result.get("vod_align"):
        print("\n--- VOD alignment ---")
        print(result["vod_align"])
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
