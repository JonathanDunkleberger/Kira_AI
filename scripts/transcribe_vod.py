#!/usr/bin/env python3
"""transcribe_vod.py — Transcribe a VOD video file directly with Whisper.

Produces a self-contained timestamped transcript of the actual video audio,
independent of any session logs. Use this as the ground truth for clipping.

Usage:
    # Transcribe the full VOD (uses GPU if available)
    python scripts/transcribe_vod.py --vod "G:/path/to/stream.mp4"

    # Output goes to:  logs/vod_transcripts/<vod_basename>.md
    # (overrideable with --out)

    # Dry-run: just probe duration, no transcription
    python scripts/transcribe_vod.py --vod "G:/path/to/stream.mp4" --dry-run

    # Restrict to a time window (saves time if you know the session window)
    python scripts/transcribe_vod.py --vod "G:/path/to/stream.mp4" --start 7500 --duration 3600

Output format (each line):
    [HH:MM:SS] text of spoken segment

After transcribing, you can grep/search the output for known quotes
from the session transcript to find exact VOD timestamps for clipping.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kira.config import WHISPER_MODEL_SIZE, WHISPER_CACHE_DIR


def _probe_duration(vod_path: str) -> float:
    """Return VOD duration in seconds via ffprobe."""
    import subprocess, json
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", vod_path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def _extract_audio(vod_path: str, start_s: float, duration_s: float, out_wav: str) -> None:
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", vod_path,
        "-t", str(duration_s),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[-500:]}")


def _transcribe(wav_path: str, device: str, compute_type: str) -> list[dict]:
    from faster_whisper import WhisperModel
    print(f"   [Whisper] Loading {WHISPER_MODEL_SIZE} on {device}/{compute_type}…")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        download_root=WHISPER_CACHE_DIR,
        device=device,
        compute_type=compute_type,
    )
    print(f"   [Whisper] Transcribing…")
    segs, _ = model.transcribe(wav_path, language="en", beam_size=5,
                                vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vod", required=True, help="Path to VOD video file")
    p.add_argument("--out", default=None, help="Output .md path (default: logs/vod_transcripts/<name>.md)")
    p.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default: 0)")
    p.add_argument("--duration", type=float, default=None, help="Max seconds to transcribe (default: full)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Whisper device. 'auto' tries CUDA then CPU.")
    p.add_argument("--dry-run", action="store_true", help="Probe only, no transcription")
    args = p.parse_args()

    vod_path = args.vod
    if not os.path.exists(vod_path):
        sys.exit(f"[ERROR] VOD not found: {vod_path}")

    total_s = _probe_duration(vod_path)
    h, rem = divmod(int(total_s), 3600)
    m, s = divmod(rem, 60)
    print(f"VOD: {Path(vod_path).name}  duration={h:02d}:{m:02d}:{s:02d}  ({total_s:.0f}s)")

    start_s = args.start
    duration_s = args.duration or (total_s - start_s)
    duration_s = min(duration_s, total_s - start_s)

    print(f"Window: {_fmt_ts(start_s)} → {_fmt_ts(start_s + duration_s)}  ({duration_s/60:.1f} min)")

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        return

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("logs/vod_transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(vod_path).stem
        suffix = f"_{_fmt_ts(start_s).replace(':','-')}" if args.start > 0 else ""
        out_path = out_dir / f"{stem}{suffix}.md"

    # Extract audio to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        print(f"   [ffmpeg] Extracting audio window…")
        _extract_audio(vod_path, start_s, duration_s, wav_path)

        # Pick device
        if args.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device, compute = "cuda", "float16"
                    print(f"   [Whisper] GPU available — using CUDA/float16")
                else:
                    device, compute = "cpu", "int8"
                    print(f"   [Whisper] No GPU — using CPU/int8 (will be slow for long videos)")
            except ImportError:
                device, compute = "cpu", "int8"
        elif args.device == "cuda":
            device, compute = "cuda", "float16"
        else:
            device, compute = "cpu", "int8"

        segments = _transcribe(wav_path, device, compute)
        print(f"   [Whisper] Done — {len(segments)} segment(s)")

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    # Write output — timestamps are relative to start_s (absolute VOD time)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# VOD Transcript: {Path(vod_path).name}\n\n")
        f.write(f"Window: {_fmt_ts(start_s)} → {_fmt_ts(start_s + duration_s)}  ({duration_s/60:.1f} min)\n\n")
        f.write("---\n\n")
        for seg in segments:
            # Timestamp is absolute VOD time
            abs_start = start_s + seg["start"]
            f.write(f"[{_fmt_ts(abs_start)}] {seg['text']}\n")

    print(f"\nTranscript written → {out_path}")
    print(f"({len(segments)} lines, {out_path.stat().st_size // 1024} KB)")
    print(f"\nTo find a clip moment, grep the output for a known quote:")
    print(f'  Select-String -Path "{out_path}" -Pattern "your quote here"')


if __name__ == "__main__":
    main()
