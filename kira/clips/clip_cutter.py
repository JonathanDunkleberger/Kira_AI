"""Auto-clip pipeline — offline post-session clip cutting engine.

Turns Kira's existing clip-candidate artifacts (``clips/YYYY-MM-DD_<activity>.md``)
into a folder of ready-to-post, pre-cut, pre-titled video files. Runs OFFLINE
after a stream — never during.

Pipeline
--------
1. Parse clip candidates from the day's ``clips/*.md`` artifact(s).
2. Anchor each candidate to a precise wall-clock instant by matching its
   verbatim *key exchange* quotes against the per-line UTC timestamps in
   ``logs/streams/<date>_*/events.jsonl``. (The ``**Timestamp:**`` field in the
   .md is only an LLM *approximation* — accurate to the minute at best — so it
   is used only as a fallback when no quote matches.)
3. Map each anchor to the OBS recording whose wall-clock span contains it, and
   compute the offset into that file. Candidates that fall in no recording are
   reported as "missed — not recorded".
4. Cut with ffmpeg: stream-copy first (instant); auto-fall-back to an NVENC
   re-encode if the copy is keyframe-misaligned (wrong duration) or fails.
5. Title every clip in Kira's voice with a single batched LLM call; write a
   sidecar ``NN_title.txt`` next to each ``.mp4``.
6. Write ``clips_report.md`` in the day folder and print a console summary.

The CLI wrapper is ``scripts/cut_clips.py``. This module is import-safe so a
future phase-2 shutdown auto-trigger can call :func:`cut_session` directly.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

from kira.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_HAIKU_MODEL,
    OBS_RECORDINGS_DIR,
    CLIP_PRE_SECONDS,
    CLIP_POST_SECONDS,
    CLIP_VIDEO_EXTS,
    REEL_MIN_MINUTES,
    WHISPER_CACHE_DIR,
    WHISPER_MODEL_SIZE,
)

# Minimum output clip length (seconds). Applied after pre/post to protect against
# very short anchor windows (single-line clips at tight post=3s).
_MIN_CLIP_SECONDS = 12.0

# Quote-match confidence threshold (difflib ratio). Below this, a quote line is
# treated as "not found" in the event log.
_MATCH_THRESHOLD = 0.72


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One clip candidate parsed from a clips/*.md artifact."""
    index: int
    title: str
    approx_hms: str | None           # "HH:MM:SS" approximate offset into stream (LLM guess)
    why: str
    suggested_title: str
    quotes: list[str] = field(default_factory=list)   # verbatim exchange lines (speaker stripped)
    raw_quotes: list[str] = field(default_factory=list)  # original "> Speaker: ..." lines

    # Score from LLM (clip-worthiness 1-10); 0 means not scored (legacy artifacts):
    score: int = 0

    # Resolved during alignment:
    anchor_start_epoch: float | None = None  # earliest matched quote ts (UTC epoch)
    anchor_end_epoch: float | None = None    # latest matched quote ts
    match_confidence: float = 0.0            # best difflib ratio achieved
    matched_via: str = "unmatched"           # "quote" | "approx_timestamp" | "unmatched"

    # Resolved during recording mapping:
    recording_path: str | None = None
    clip_start_offset: float | None = None   # seconds into the recording
    clip_duration: float | None = None

    # Output:
    out_path: str | None = None
    status: str = "pending"                  # "cut" | "missed" | "error" | "pending"
    error: str | None = None
    cut_method: str | None = None            # "copy" | "nvenc" | "libx264"

    # Titling:
    final_title: str | None = None
    description: str | None = None
    hashtags: list[str] = field(default_factory=list)


@dataclass
class Recording:
    """An OBS recording file with its resolved wall-clock span (UTC epoch)."""
    path: str
    start_epoch: float
    end_epoch: float
    duration: float
    start_source: str   # "metadata_creation_time" | "mtime_minus_duration"


@dataclass
class EventLine:
    """A single timestamped utterance from events.jsonl."""
    epoch: float
    speaker: str
    text: str
    norm: str           # normalized text for matching


# ─────────────────────────────────────────────────────────────────────────────
# Text + time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, strip quotes/punctuation, collapse whitespace — for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[\"'“”‘’]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_iso_utc(ts: str) -> float:
    """Parse an events.jsonl ISO-8601 UTC timestamp (e.g. '2026-06-12T00:36:18.504Z')
    into an absolute epoch (seconds). Returns NaN-safe float."""
    s = ts.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _hms_to_seconds(hms: str) -> int | None:
    """'HH:MM:SS' or 'MM:SS' → seconds. None if unparseable."""
    parts = hms.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return None
    return h * 3600 + m * 60 + s


def _slugify(text: str, maxlen: int = 50) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:maxlen] or "clip"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parse clip candidates
# ─────────────────────────────────────────────────────────────────────────────

def parse_candidates(md_path: str) -> list[Candidate]:
    """Parse a clips/*.md artifact into Candidate objects.

    The format (produced by bot.py's _write_session_artifacts):
        ### Clip N — Title
        **Timestamp:** 00:22:57
        **Why it's good:** ...
        **Suggested YouTube short title:** ...
        **Key exchange:**
        > Jonny: "..."
        > Kira: "..."
    """
    with open(md_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    candidates: list[Candidate] = []
    # Split into per-clip blocks on the "### Clip N" heading.
    blocks = re.split(r"^###\s+", text, flags=re.MULTILINE)
    for block in blocks:
        head = block.splitlines()[0] if block else ""
        m = re.match(r"Clip\s+(\d+)\s*[—\-:]\s*(.+)$", head.strip())
        if not m:
            continue
        idx = int(m.group(1))
        title = m.group(2).strip().strip('"“”')

        ts_m = re.search(r"\*\*Timestamp:\*\*\s*([0-9:]+)", block)
        score_m = re.search(r"\*\*Score:\*\*\s*(\d+)", block)
        why_m = re.search(r"\*\*Why it'?s good:\*\*\s*(.+?)(?:\n\*\*|\n>|\Z)", block, re.DOTALL)
        sug_m = re.search(r"\*\*Suggested[^:]*:\*\*\s*(.+?)(?:\n\*\*|\n>|\Z)", block, re.DOTALL)

        raw_quotes = re.findall(r"^>\s?(.*)$", block, flags=re.MULTILINE)
        quotes: list[str] = []
        for q in raw_quotes:
            q = q.strip()
            if not q:
                continue
            # Strip a leading "Speaker:" label, keep the spoken content.
            stripped = re.sub(r'^[A-Z][\w .\-]{0,24}:\s*', "", q)
            stripped = stripped.strip().strip('"“”')
            if stripped:
                quotes.append(stripped)

        candidates.append(Candidate(
            index=idx,
            title=title,
            approx_hms=ts_m.group(1) if ts_m else None,            score=int(score_m.group(1)) if score_m else 0,            why=(why_m.group(1).strip() if why_m else ""),
            suggested_title=(sug_m.group(1).strip().strip('"“”') if sug_m else title),
            quotes=quotes,
            raw_quotes=[q.strip() for q in raw_quotes if q.strip()],
        ))

    candidates.sort(key=lambda c: c.index)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build wall-clock index from events.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def load_event_index(date_str: str, base_dir: str = "logs/streams",
                     activity: str | None = None,
                     last_only: bool = False) -> tuple[list[EventLine], float | None]:
    """Load every voice_input / kira_response line (with UTC epoch) from all
    session dirs for ``date_str``. Returns (events, earliest_session_start_epoch).

    activity: when provided along with last_only=True, only load from the LAST
    session dir on ``date_str`` that matches this activity slug. This avoids
    cross-session quote contamination when cutting from a specific recording.

    Session dirs are named ``YYYY-MM-DD_HH-MM_<activity>`` using the LOCAL date at
    session start, which matches the clips/*.md filename date.
    """
    events: list[EventLine] = []
    earliest_start: float | None = None

    pattern = os.path.join(base_dir, f"{date_str}_*")
    all_dirs = sorted(glob.glob(pattern))

    # When last_only is requested, narrow to just the last session for that activity.
    if last_only and activity:
        matching = [d for d in all_dirs
                    if os.path.basename(d).endswith(f"_{activity}")]
        if matching:
            all_dirs = [matching[-1]]  # last = latest timestamp prefix

    for sess_dir in all_dirs:
        ev_path = os.path.join(sess_dir, "events.jsonl")
        if not os.path.isfile(ev_path):
            continue
        with open(ev_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = obj.get("type")
                ts = obj.get("ts")
                if not ts:
                    continue
                try:
                    epoch = _parse_iso_utc(ts)
                except Exception:
                    continue
                if etype == "session_start":
                    if earliest_start is None or epoch < earliest_start:
                        earliest_start = epoch
                    continue
                if etype in ("voice_input", "kira_response"):
                    txt = obj.get("text", "")
                    if not txt:
                        continue
                    speaker = obj.get("speaker") or obj.get("source") or (
                        "Kira" if etype == "kira_response" else "Speaker")
                    events.append(EventLine(
                        epoch=epoch,
                        speaker=speaker,
                        text=txt,
                        norm=_norm(txt),
                    ))
    events.sort(key=lambda e: e.epoch)
    return events, earliest_start


# ─────────────────────────────────────────────────────────────────────────────
# 3. Align candidates → wall-clock anchors
# ─────────────────────────────────────────────────────────────────────────────

def _best_match(quote_norm: str, events: list[EventLine]) -> tuple[float, float] | None:
    """Return (epoch, ratio) of the best-matching event line for a normalized
    quote, or None if nothing clears the threshold."""
    from difflib import SequenceMatcher

    if not quote_norm:
        return None
    best_epoch = None
    best_ratio = 0.0
    for ev in events:
        if not ev.norm:
            continue
        # Fast path: containment in either direction is a strong signal.
        if quote_norm in ev.norm or ev.norm in quote_norm:
            ratio = max(0.9, SequenceMatcher(None, quote_norm, ev.norm).ratio())
        else:
            ratio = SequenceMatcher(None, quote_norm, ev.norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_epoch = ev.epoch
    if best_epoch is not None and best_ratio >= _MATCH_THRESHOLD:
        return best_epoch, best_ratio
    return None


def align_candidate(cand: Candidate, events: list[EventLine],
                    session_start_epoch: float | None) -> None:
    """Resolve cand.anchor_start_epoch / anchor_end_epoch via quote matching,
    falling back to the approximate HH:MM:SS offset + session start.

    Matching every key-exchange quote and taking the earliest/latest hits anchors
    the clip to the *whole* exchange (setup line → payoff line), which is far more
    robust than the single LLM-estimated timestamp.
    """
    matched_epochs: list[float] = []
    best_conf = 0.0
    for q in cand.quotes:
        hit = _best_match(_norm(q), events)
        if hit:
            matched_epochs.append(hit[0])
            best_conf = max(best_conf, hit[1])

    if matched_epochs:
        cand.anchor_start_epoch = min(matched_epochs)
        cand.anchor_end_epoch = max(matched_epochs)
        # Sanity check: if the window is unreasonably wide (>90s), one quote
        # almost certainly matched a spurious late event. Collapse to the
        # start anchor so the fixed `post` seconds determine the tail.
        if cand.anchor_end_epoch - cand.anchor_start_epoch > 90.0:
            cand.anchor_end_epoch = cand.anchor_start_epoch
        cand.match_confidence = best_conf
        cand.matched_via = "quote"
        return

    # Fallback: approximate timestamp into the stream + session start.
    if cand.approx_hms and session_start_epoch is not None:
        secs = _hms_to_seconds(cand.approx_hms)
        if secs is not None:
            anchor = session_start_epoch + secs
            cand.anchor_start_epoch = anchor
            cand.anchor_end_epoch = anchor
            cand.matched_via = "approx_timestamp"
            return

    cand.matched_via = "unmatched"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gather OBS recordings and their wall-clock spans
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe_format(path: str) -> dict:
    """Return ffprobe 'format' dict (duration + tags) for a media file."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:format_tags=creation_time",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout or "{}")
    return data.get("format", {})


def gather_recordings(recordings_dir: str, exts: list[str]) -> list[Recording]:
    """Scan recordings_dir for video files and resolve each one's UTC span.

    Recording start is taken from the container ``creation_time`` metadata tag
    (UTC, written by OBS) when present — the most reliable source. Otherwise it
    falls back to ``mtime - duration`` (file finalized when recording stops).
    """
    recordings: list[Recording] = []
    if not recordings_dir or not os.path.isdir(recordings_dir):
        return recordings

    # Collect candidate video files, then de-duplicate OBS remux twins: when a
    # .mkv and its remuxed .mp4 share the same basename, prefer the .mp4 and
    # skip the .mkv so the same session isn't processed twice.
    by_stem: dict[str, list[str]] = {}
    for name in sorted(os.listdir(recordings_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in exts:
            continue
        path = os.path.join(recordings_dir, name)
        if not os.path.isfile(path):
            continue
        by_stem.setdefault(os.path.splitext(name)[0], []).append(path)

    chosen: list[str] = []
    for stem, paths in by_stem.items():
        if len(paths) == 1:
            chosen.append(paths[0])
            continue
        mp4 = next((p for p in paths if p.lower().endswith(".mp4")), None)
        if mp4:
            twins = [os.path.basename(p) for p in paths if p != mp4]
            print(f"   [Recordings] remux twins for '{stem}': preferring "
                  f"{os.path.basename(mp4)}, skipping {', '.join(twins)}.")
            chosen.append(mp4)
        else:
            chosen.extend(paths)

    for path in sorted(chosen):
        name = os.path.basename(path)
        try:
            fmt = _ffprobe_format(path)
            duration = float(fmt.get("duration", 0.0))
            if duration <= 0:
                print(f"   [Recordings] WARN: {name} has zero/unknown duration — skipping.")
                continue
            tags = fmt.get("tags", {}) or {}
            ctime = tags.get("creation_time")
            if ctime:
                start_epoch = _parse_iso_utc(ctime)
                start_source = "metadata_creation_time"
            else:
                start_epoch = os.path.getmtime(path) - duration
                start_source = "mtime_minus_duration"
            recordings.append(Recording(
                path=path,
                start_epoch=start_epoch,
                end_epoch=start_epoch + duration,
                duration=duration,
                start_source=start_source,
            ))
        except Exception as e:
            print(f"   [Recordings] ERROR probing {name}: {e}")
            traceback.print_exc()

    recordings.sort(key=lambda r: r.start_epoch)
    return recordings


def map_to_recording(cand: Candidate, recordings: list[Recording],
                     pre: float, post: float) -> None:
    """Map a candidate's wall-clock anchor onto a recording + offset/duration.

    Sets cand.recording_path / clip_start_offset / clip_duration, or marks the
    candidate 'missed' if its moment falls in no recording.
    """
    if cand.anchor_start_epoch is None:
        cand.status = "missed"
        cand.error = "no wall-clock anchor (no quote match, no usable timestamp)"
        return

    moment = cand.anchor_start_epoch
    rec = next((r for r in recordings if r.start_epoch <= moment <= r.end_epoch), None)
    if rec is None:
        cand.status = "missed"
        cand.error = "moment falls outside every recording span — not recorded"
        return

    anchor_end = cand.anchor_end_epoch or cand.anchor_start_epoch
    clip_start = (cand.anchor_start_epoch - rec.start_epoch) - pre
    clip_end = (anchor_end - rec.start_epoch) + post
    clip_start = max(0.0, clip_start)
    clip_end = min(rec.duration, clip_end)
    if clip_end <= clip_start:
        cand.status = "missed"
        cand.error = "computed clip window is empty after clamping to recording bounds"
        return

    cand.recording_path = rec.path
    cand.clip_start_offset = clip_start
    # Enforce minimum clip length so tight-post cuts don't produce unusable stubs.
    actual_dur = clip_end - clip_start
    if actual_dur < _MIN_CLIP_SECONDS:
        extra = _MIN_CLIP_SECONDS - actual_dur
        clip_end = min(rec.duration, clip_end + extra)
    cand.clip_duration = clip_end - clip_start


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cutting (ffmpeg: stream-copy first, NVENC re-encode fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _nvenc_available() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30)
        return "h264_nvenc" in out.stdout
    except Exception:
        return False


def _probe_duration(path: str) -> float:
    try:
        fmt = _ffprobe_format(path)
        return float(fmt.get("duration", 0.0))
    except Exception:
        return 0.0


def cut_clip(cand: Candidate, out_path: str, force_reencode: bool,
             nvenc_ok: bool) -> None:
    """Cut cand's window from its recording into out_path.

    Strategy: try instant stream-copy; if the result is missing, zero, or its
    duration deviates from the request by >1.5s (keyframe snap), re-encode for
    frame-accurate cuts (NVENC if available, else libx264). A clip that cannot be
    produced is marked 'error' and reported — never silently skipped.
    """
    src = cand.recording_path
    start = f"{cand.clip_start_offset:.3f}"
    dur = f"{cand.clip_duration:.3f}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    requested = cand.clip_duration or 0.0

    if not force_reencode:
        copy_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", start, "-i", src, "-t", dur,
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", out_path,
        ]
        try:
            res = _run(copy_cmd)
            if res.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                actual = _probe_duration(out_path)
                if abs(actual - requested) <= 1.5:
                    cand.status = "cut"
                    cand.out_path = out_path
                    cand.cut_method = "copy"
                    return
                # else: keyframe misalignment → fall through to re-encode.
        except Exception as e:
            print(f"   [Cut] copy attempt errored for clip {cand.index}: {e}")

    # Re-encode for frame accuracy.
    if nvenc_ok:
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]
        method = "nvenc"
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        method = "libx264"
    enc_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", start, "-i", src, "-t", dur,
        *venc, "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", out_path,
    ]
    try:
        res = _run(enc_cmd)
        if res.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            cand.status = "cut"
            cand.out_path = out_path
            cand.cut_method = method
            return
        cand.status = "error"
        cand.error = f"ffmpeg re-encode failed (rc={res.returncode}): {res.stderr.strip()[:300]}"
    except Exception as e:
        cand.status = "error"
        cand.error = f"ffmpeg re-encode exception: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Titling — one batched LLM call in Kira's voice
# ─────────────────────────────────────────────────────────────────────────────

async def generate_titles(candidates: list[Candidate], activity: str) -> None:
    """Title every cut clip in one batched Haiku call. Mutates candidates in place.

    Failure is loud but non-fatal: clips keep their candidate title as a fallback
    so the pipeline still produces usable, named output.
    """
    cut = [c for c in candidates if c.status == "cut"]
    if not cut:
        return
    if not ANTHROPIC_API_KEY:
        print("   [Titles] ANTHROPIC_API_KEY not set — using candidate titles as-is.")
        for c in cut:
            c.final_title = c.suggested_title or c.title
        return

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("   [Titles] anthropic package not installed — using candidate titles as-is.")
        for c in cut:
            c.final_title = c.suggested_title or c.title
        return

    items = []
    for c in cut:
        exchange = "\n".join(c.raw_quotes) if c.raw_quotes else "(no transcript captured)"
        items.append(
            f"CLIP {c.index}\n"
            f"Working title: {c.title}\n"
            f"Why it lands: {c.why}\n"
            f"Exchange:\n{exchange}"
        )
    joined = "\n\n---\n\n".join(items)

    prompt = (
        f"You are Kira, an AI VTuber co-host with a sharp, witty, slightly chaotic voice. "
        f"Activity: {activity}. Below are {len(cut)} clip moments from a stream. For EACH clip, "
        f"write punchy, clip-worthy YouTube Shorts metadata IN YOUR OWN VOICE.\n\n"
        f"Return ONLY a JSON array (no prose, no markdown fences). One object per clip, in order:\n"
        f'  {{"n": <clip number>, "title": "<under 60 chars, clippy, hooky>", '
        f'"description": "<one punchy line>", "hashtags": ["tag1","tag2","tag3"]}}\n\n'
        f"Rules: titles under 60 characters; no surrounding quotes inside the title; "
        f"hashtags without the # symbol; keep it in Kira's voice.\n\n"
        f"{joined}"
    )

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    response = None
    for attempt in range(1, 3):
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=CLAUDE_HAIKU_MODEL,
                    max_tokens=2000,
                    system="You output only valid JSON arrays. No commentary.",
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=60.0,
            )
            response = resp.content[0].text
            break
        except Exception as e:
            print(f"   [Titles] attempt {attempt} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(3)

    by_index = {c.index: c for c in cut}
    if response:
        # Tolerate stray fences / prose around the JSON array.
        m = re.search(r"\[.*\]", response, re.DOTALL)
        raw = m.group(0) if m else response
        try:
            data = json.loads(raw)
            for obj in data:
                n = int(obj.get("n", -1))
                c = by_index.get(n)
                if not c:
                    continue
                t = (obj.get("title") or "").strip().strip('"“”')
                c.final_title = t[:60] if t else (c.suggested_title or c.title)
                c.description = (obj.get("description") or "").strip()
                hh = obj.get("hashtags") or []
                c.hashtags = [str(h).lstrip("#").strip() for h in hh if str(h).strip()][:3]
        except Exception as e:
            print(f"   [Titles] JSON parse failed: {e} — using candidate titles as-is.")

    # Backfill any clip the model skipped.
    for c in cut:
        if not c.final_title:
            c.final_title = c.suggested_title or c.title


def write_title_sidecar(cand: Candidate) -> None:
    """Write NN_title.txt next to the clip's .mp4."""
    if not cand.out_path:
        return
    sidecar = os.path.splitext(cand.out_path)[0] + "_title.txt"
    tags = " ".join(f"#{h}" for h in cand.hashtags) if cand.hashtags else ""
    lines = [
        cand.final_title or cand.title,
        "",
        cand.description or cand.why,
    ]
    if tags:
        lines += ["", tags]
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"   [Titles] Failed to write sidecar for clip {cand.index}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(candidates: list[Candidate], day_dir: str, date_str: str,
                 activity: str, recordings: list[Recording]) -> str:
    """Write clips_report.md into the day folder and return its path.

    Cut candidates appear first (in rank/output order). Candidates that were
    capped by --top appear in a separate 'not cut (below cap)' section so
    Jonny can cherry-pick extras with --top 30 or a future --only flag.
    """
    cut = [c for c in candidates if c.status == "cut"]
    capped = [c for c in candidates if c.status == "capped"]
    missed = [c for c in candidates if c.status == "missed"]
    errored = [c for c in candidates if c.status == "error"]

    os.makedirs(day_dir, exist_ok=True)
    report_path = os.path.join(day_dir, "clips_report.md")
    lines = [
        f"# Clip Cut Report — {date_str} ({activity})",
        "",
        f"- Candidates: **{len(candidates)}**",
        f"- Cut: **{len(cut)}**",
        f"- Capped (below --top N): **{len(capped)}**",
        f"- Missed (not recorded): **{len(missed)}**",
        f"- Errors: **{len(errored)}**",
        f"- Recordings scanned: **{len(recordings)}**",
        "",
    ]
    if recordings:
        lines.append("## Recordings")
        for r in recordings:
            s = datetime.fromtimestamp(r.start_epoch).strftime("%H:%M:%S")
            e = datetime.fromtimestamp(r.end_epoch).strftime("%H:%M:%S")
            lines.append(f"- `{os.path.basename(r.path)}` — {s}→{e} "
                         f"({r.duration:.0f}s, start via {r.start_source})")
        lines.append("")

    def _candidate_block(c: Candidate) -> list[str]:
        score_str = f" (score {c.score}/10)" if c.score else ""
        out = [
            "",
            f"### {c.final_title or c.title}{score_str}",
            f"- Status: **{c.status}**" + (f" ({c.cut_method})" if c.cut_method else ""),
        ]
        if c.matched_via == "quote":
            anchor = datetime.fromtimestamp(c.anchor_start_epoch).strftime("%H:%M:%S")
            out.append(f"- Anchor: {anchor} (quote match, confidence {c.match_confidence:.2f})")
        elif c.matched_via == "approx_timestamp":
            out.append(f"- Anchor: ~{c.approx_hms} (approx timestamp fallback — lower accuracy)")
        else:
            out.append("- Anchor: none (could not align)")
        if c.out_path:
            out.append(f"- File: `{os.path.relpath(c.out_path, day_dir)}`")
        if c.description:
            out.append(f"- Description: {c.description}")
        if c.hashtags:
            out.append(f"- Hashtags: {' '.join('#' + h for h in c.hashtags)}")
        out.append(f"- Why: {c.why}")
        if c.error:
            out.append(f"- ⚠️ {c.error}")
        return out

    if cut:
        lines.append("## Cut Clips (ranked best-first)")
        for c in cut:  # already in rank order from cut_session
            lines.extend(_candidate_block(c))
        lines.append("")

    if capped:
        lines.append("## Not Cut — Below Cap (cherry-pick with --top N)")
        for c in capped:
            lines.extend(_candidate_block(c))
        lines.append("")

    other = missed + errored
    if other:
        lines.append("## Not Cut — Missed / Errors")
        for c in other:
            lines.extend(_candidate_block(c))
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# 5b. VOD start-epoch estimation via Whisper anchor matching
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, start_s: float, duration_s: float, out_wav: str) -> None:
    """Extract a mono 16 kHz WAV segment from a video file (fast, no re-encode)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", video_path, "-t", f"{duration_s:.3f}",
        "-ac", "1", "-ar", "16000", "-vn", out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {r.stderr.strip()[:300]}")


def _whisper_segments(wav_path: str) -> list[dict]:
    """Run faster-whisper on a WAV and return a list of {start, end, text} dicts.

    Always runs on CPU (int8) for VOD alignment — CUDA is skipped entirely here
    because the bot may hold the GPU and a CUDA fatal error during inference kills
    the whole process before Python can catch it and fall back.  CPU int8 accuracy
    is identical for fuzzy text anchor matching.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "pip install faster-whisper  (required for --vod Whisper anchor matching)."
        )
    print(f"   [VodAlign] Loading Whisper on CPU/int8 (GPU skipped for alignment)…")
    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            download_root=WHISPER_CACHE_DIR,
            device="cpu",
            compute_type="int8",
        )
        segs, _ = model.transcribe(wav_path, language="en", beam_size=5)
        result = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
        return result
    except Exception as e:
        raise RuntimeError(f"Whisper inference failed on CPU: {e}") from e


def estimate_vod_start_epoch(
    vod_path: str,
    events: list[EventLine],
    session_start_epoch: float,
    search_window_s: float = 1200.0,
    start_offset_s: float = 0.0,
) -> tuple[float, float, str]:
    """Derive the UTC epoch when a VOD recording started via Whisper anchor matching.

    Strategy:
    1. Pick the 3 most specific (longest-text, early-in-session) event lines as
       anchors.  These will be spoken somewhere in the VOD — we just don't know
       where yet.
    2. Extract the first ``search_window_s`` seconds of VOD audio (covers up to
       20 min; that's almost certainly enough if the session started near the top).
    3. Transcribe with Whisper, fuzzy-match each anchor against every segment.
    4. For each hit: vod_start = event_epoch − segment.start
    5. Cross-check: if multiple anchors agree within 5 s → high confidence.
       If they disagree by >5 s → warn and use the median.

    Returns (vod_start_epoch, confidence_0_to_1, human_readable_details).
    Raises RuntimeError if no anchor can be matched.
    """
    from difflib import SequenceMatcher
    import tempfile

    # Pick anchors: earliest voice_input events from Jonny (not Kira TTS, not chat bots).
    # ONLY voice_input lines work as Whisper anchors — they're actually spoken in the VOD.
    # Chat messages (streamlabs, viewers, etc.) are never in the audio stream.
    _kira_speakers = {"kira", "assistant", "ai"}
    _bot_speakers = {"streamlabs", "nightbot", "moobot", "fossabot", "bot"}
    early = [
        e for e in events
        if e.epoch <= session_start_epoch + 900 and len(e.text.split()) >= 6
    ]
    jonny_pool = sorted(
        [e for e in early
         if e.speaker.lower() not in _kira_speakers
         and e.speaker.lower() not in _bot_speakers
         and not e.text.startswith("[")],   # filter out [Chat batch from ...] lines
        key=lambda e: (e.epoch, -len(e.text)),
    )
    kira_pool = sorted(
        [e for e in early
         if e.speaker.lower() in _kira_speakers
         and not e.text.startswith("[")],
        key=lambda e: (e.epoch, -len(e.text)),
    )
    # Fill up to 3: human first, Kira only as fallback.
    anchors = (jonny_pool[:3] + kira_pool)[:3]
    if not anchors:
        raise RuntimeError(
            "No usable anchor quotes found in the first 15 min of the session events "
            "to use for Whisper VOD alignment."
        )

    print(f"   [VodAlign] Anchors selected ({len(anchors)}):")
    for a in anchors:
        print(f"      [{datetime.fromtimestamp(a.epoch).strftime('%H:%M:%S')}] "
              f"'{a.text[:60]}...' " if len(a.text) > 60 else
              f"      [{datetime.fromtimestamp(a.epoch).strftime('%H:%M:%S')}] '{a.text}'")

    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        _start = max(0.0, start_offset_s)
        print(f"   [VodAlign] Extracting {search_window_s:.0f}s of audio from VOD @ offset {_start:.0f}s…")
        _extract_audio_wav(vod_path, _start, search_window_s, wav_path)

        print(f"   [VodAlign] Running Whisper ({WHISPER_MODEL_SIZE}) on audio window…")
        segments = _whisper_segments(wav_path)
        print(f"   [VodAlign] Whisper returned {len(segments)} segment(s).")

        derived_starts: list[float] = []
        anchor_lines: list[str] = []

        for anchor in anchors:
            anorm = _norm(anchor.text)
            best_ratio = 0.0
            best_start = None
            for seg in segments:
                snorm = _norm(seg["text"])
                if not snorm:
                    continue
                # Fast-path: substring containment
                ratio = (
                    max(0.88, SequenceMatcher(None, anorm, snorm).ratio())
                    if (anorm[:30] in snorm or snorm[:30] in anorm)
                    else SequenceMatcher(None, anorm[:60], snorm[:80]).ratio()
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = seg["start"]

            if best_start is not None and best_ratio >= 0.52:
                derived = anchor.epoch - (_start + best_start)
                derived_starts.append(derived)
                line = (
                    f"  '{anchor.text[:55]}…' → VOD {_start + best_start:.1f}s "
                    f"(ratio {best_ratio:.2f}) → vod_start {datetime.fromtimestamp(derived).strftime('%H:%M:%S UTC')}"
                )
                anchor_lines.append(line)
                print(f"   [VodAlign]{line}")
            else:
                anchor_lines.append(
                    f"  '{anchor.text[:55]}' → NO MATCH (best ratio {best_ratio:.2f})"
                )
                print(f"   [VodAlign]{anchor_lines[-1]}")

        if not derived_starts:
            raise RuntimeError(
                f"No anchor quotes matched in the first {search_window_s:.0f}s of the VOD "
                f"(tried {len(anchors)} anchors). "
                f"The session may start later — re-run with a larger search window."
            )

        spread = max(derived_starts) - min(derived_starts) if len(derived_starts) > 1 else 0.0
        if spread > 5.0:
            print(f"   [VodAlign] WARNING: anchors disagree by {spread:.1f}s (>5s) — using median.")
            derived_starts.sort()
            vod_start = derived_starts[len(derived_starts) // 2]
            confidence = 0.45
        else:
            vod_start = sum(derived_starts) / len(derived_starts)
            confidence = min(0.95, 0.65 + 0.15 * len(derived_starts))

        low_conf_tag = ""
        if len(derived_starts) < 2:
            low_conf_tag = "[LOW CONFIDENCE — single anchor matched] "
            print(f"   [VodAlign] WARNING: only 1 of {len(anchors)} anchor(s) matched. "
                  f"Review reel_report.md carefully before posting.")

        details = (
            f"{low_conf_tag}VOD start derived from {len(derived_starts)}/{len(anchors)} anchor(s), "
            f"spread={spread:.1f}s, confidence={confidence:.0%}\n"
            + "\n".join(anchor_lines)
        )
        print(f"   [VodAlign] Final → vod_start_epoch={vod_start:.1f} "
              f"({datetime.fromtimestamp(vod_start).strftime('%Y-%m-%d %H:%M:%S UTC')}), "
              f"confidence={confidence:.0%}")
        return vod_start, confidence, details

    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 8. Reel assembly (--reel mode)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_reel_title(candidates: list[Candidate], activity: str) -> tuple[str, str]:
    """One Haiku call: reel title + description in Kira's voice. Returns (title, description)."""
    cut = [c for c in candidates if c.status == "cut" and c.out_path]
    if not cut:
        return f"Kira Highlights — {activity}", ""
    if not ANTHROPIC_API_KEY:
        return f"Kira Highlights — {activity}", ""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return f"Kira Highlights — {activity}", ""

    moments = "\n".join(
        f"- {c.final_title or c.title}: {c.why[:80]}"
        for c in cut[:20]
    )
    prompt = (
        f"You are Kira, an AI VTuber with a sharp, dry, chaotic wit. "
        f"Tonight's highlight reel ({activity}) contains these moments:\n{moments}\n\n"
        f"Write ONE punchy YouTube reel TITLE (≤70 chars) and ONE punchy description "
        f"sentence — both in your voice. Return JSON only: "
        f'{{\"title\": \"...\", \"description\": \"...\"}}'
    )
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        resp = await asyncio.wait_for(
            client.messages.create(
                model=CLAUDE_HAIKU_MODEL,
                max_tokens=200,
                system="Output only valid JSON. No commentary.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30.0,
        )
        raw = resp.content[0].text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            title = (obj.get("title") or "").strip().strip('"""')[:70] or f"Kira Highlights — {activity}"
            desc = (obj.get("description") or "").strip()
            return title, desc
    except Exception as e:
        print(f"   [Reel] Title generation failed: {e}")
    return f"Kira Highlights — {activity}", ""


def write_reel_report(
    candidates: list[Candidate],
    reel_path: str,
    reel_title: str,
    reel_description: str,
    out_dir: str,
    date_str: str,
    activity: str,
    vod_align_details: str = "",
) -> str:
    """Write reel_report.md. Returns its path."""
    cut = sorted(
        [c for c in candidates if c.status == "cut" and c.out_path],
        key=lambda c: c.anchor_start_epoch or 0.0,
    )
    report_path = os.path.join(out_dir, "reel_report.md")

    total_s = sum(c.clip_duration or 0.0 for c in cut)
    total_m, total_s_rem = divmod(int(total_s), 60)

    lines = [
        f"# Highlight Reel — {date_str} ({activity})",
        "",
        f"**Title:** {reel_title}",
        f"**Description:** {reel_description}",
        f"**File:** `{os.path.basename(reel_path)}`",
        f"**Segments:** {len(cut)}",
        f"**Total duration:** {total_m}m {total_s_rem}s",
        "",
    ]
    if vod_align_details:
        lines += ["## VOD Alignment", "```", vod_align_details.strip(), "```", ""]

    lines += ["## Segments (chronological)", ""]
    cumulative = 0.0
    for i, c in enumerate(cut, 1):
        dur = c.clip_duration or 0.0
        reel_m, reel_s = divmod(int(cumulative), 60)
        anchor_str = (
            datetime.fromtimestamp(c.anchor_start_epoch).strftime("%H:%M:%S")
            if c.anchor_start_epoch else "?"
        )
        lines += [
            f"### {i}. {c.final_title or c.title}",
            f"- Reel position: {reel_m}:{reel_s:02d}",
            f"- Stream timestamp: {anchor_str} (wall-clock, local)",
            f"- Duration: {dur:.1f}s",
            f"- {c.why}",
            "",
        ]
        cumulative += dur

    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def cut_reel(
    candidates: list[Candidate],
    out_dir: str,
    activity_slug: str,
    nvenc_ok: bool,
) -> str:
    """Concatenate all cut segments in chronological order into REEL_<slug>.mp4.

    Segments must already be cut (status='cut', out_path set) and encoded with
    uniform settings (NVENC or libx264) so the concat demuxer can join cleanly.
    Returns the reel path.
    """
    cut = sorted(
        [c for c in candidates if c.status == "cut" and c.out_path],
        key=lambda c: c.anchor_start_epoch or 0.0,
    )
    if not cut:
        raise RuntimeError("No successfully cut segments to assemble into a reel.")

    os.makedirs(out_dir, exist_ok=True)
    reel_slug = f"REEL_{activity_slug}"
    reel_path = os.path.join(out_dir, f"{reel_slug}.mp4")
    concat_list = os.path.join(out_dir, f"{reel_slug}_concat.txt")

    with open(concat_list, "w", encoding="utf-8") as f:
        for c in cut:
            # ffmpeg concat demuxer: forward-slash paths, single-quoted
            safe = c.out_path.replace("\\", "/")
            f.write(f"file '{safe}'\n")

    if nvenc_ok:
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        *venc, "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", reel_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    try:
        os.unlink(concat_list)
    except Exception:
        pass

    if r.returncode != 0 or not os.path.isfile(reel_path) or os.path.getsize(reel_path) == 0:
        raise RuntimeError(f"Reel concat failed (rc={r.returncode}): {r.stderr.strip()[:400]}")

    dur = _probe_duration(reel_path)
    print(f"   [Reel] {reel_path}  ({dur:.0f}s total, {len(cut)} segments)")
    return reel_path


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def find_clip_artifacts(date_str: str, activity: str | None,
                        clips_dir: str = "clips") -> list[str]:
    """Return clips/*.md artifact paths for the date (optionally one activity)."""
    if activity:
        p = os.path.join(clips_dir, f"{date_str}_{activity}.md")
        return [p] if os.path.isfile(p) else []
    return sorted(glob.glob(os.path.join(clips_dir, f"{date_str}_*.md")))


async def cut_session(date_str: str, activity: str | None = None, *,
                      pre: float | None = None, post: float | None = None,
                      force_reencode: bool = False, dry_run: bool = False,
                      top: int = 15, reel: bool = True, clips: bool = True,
                      vod_path: str | None = None) -> dict:
    """Run the full pipeline for one day (optionally one activity).

    Default behaviour (reel=True, clips=True): produces BOTH ranked individual
    clips (top N by score) AND a chronological highlight reel from all aligned
    candidates.  Reel mode implies frame-accurate re-encode (clean concat seams).
    Use --no-reel / --no-clips CLI flags to skip either output.

    reel=True  : cut ALL aligned candidates, concatenate into REEL_<slug>.mp4.
                 Skipped automatically when session < REEL_MIN_MINUTES or < 3
                 aligned candidates ("session too short for reel").
    clips=True : title + sidecar .txt for individual clip files.
    vod_path   : path to a VOD file lacking reliable creation_time metadata;
                 Whisper derives the actual stream-start epoch from events quotes.

    Returns a summary dict. Raises only on hard configuration errors.
    """
    pre = CLIP_PRE_SECONDS if pre is None else pre
    post = CLIP_POST_SECONDS if post is None else post
    exts = [e if e.startswith(".") else "." + e
            for e in (x.strip().lower() for x in CLIP_VIDEO_EXTS.split(",")) if e]

    if not vod_path:
        if not OBS_RECORDINGS_DIR:
            raise RuntimeError("OBS_RECORDINGS_DIR is not set. Add it to your .env "
                               "(the folder where OBS writes local recordings).")
        if not os.path.isdir(OBS_RECORDINGS_DIR):
            raise RuntimeError(f"OBS_RECORDINGS_DIR does not exist: {OBS_RECORDINGS_DIR}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found on PATH.")

    artifacts = find_clip_artifacts(date_str, activity)
    if not artifacts:
        raise RuntimeError(f"No clip artifacts found for {date_str}"
                           + (f" / {activity}" if activity else "")
                           + " in clips/.")

    nvenc_ok = _nvenc_available()

    # ── Recording sources ────────────────────────────────────────────────────
    # If a specific VOD file is given, do Whisper anchor-matching first to
    # derive its start_epoch, then synthesise a Recording object for it.
    # Otherwise scan OBS_RECORDINGS_DIR as normal.
    vod_align_details = ""

    if vod_path:
        print(f"   [Clips] VOD override: {os.path.basename(vod_path)}")
        fmt = _ffprobe_format(vod_path)
        vod_duration = float(fmt.get("duration", 0.0))
        if vod_duration <= 0:
            raise RuntimeError(f"Could not determine duration of VOD: {vod_path}")

        # We'll do the Whisper step per-artifact below (inside the loop) since
        # we need activity-filtered events. Pre-probe duration now.
    else:
        recordings = gather_recordings(OBS_RECORDINGS_DIR, exts)
        print(f"   [Clips] {len(recordings)} recording(s) in {OBS_RECORDINGS_DIR}; "
              f"NVENC {'available' if nvenc_ok else 'NOT available (libx264 fallback)'}.")
        if not recordings:
            print("   [Clips] WARNING: no recordings found — every candidate will be 'missed'.")

    summaries = []
    for md_path in artifacts:
        act = re.match(rf"^{re.escape(date_str)}_(.+)\.md$", os.path.basename(md_path))
        act_name = act.group(1) if act else "general"
        candidates = parse_candidates(md_path)
        print(f"\n   [Clips] {os.path.basename(md_path)} → {len(candidates)} candidate(s).")

        # Load events scoped to this activity. When vod_path is given, only use
        # the LAST matching session dir to avoid cross-session quote contamination.
        events, session_start = load_event_index(
            date_str,
            activity=act_name,
            last_only=bool(vod_path),
        )
        print(f"   [Clips] {len(events)} utterance(s) indexed "
              f"({'last ' + act_name + ' session only' if vod_path else 'all sessions'}).")

        # ── VOD Whisper alignment (done per-artifact so anchors are scoped) ─
        if vod_path:
            if dry_run:
                vod_start_epoch = session_start or 0.0
                vod_align_details = "[dry-run: Whisper skipped — vod_start assumed = session_start]"
                print("   [VodAlign] Dry-run: skipping Whisper, vod_start assumed = session_start.")
            else:
                # Smart offset: when the session spans are deep in the VOD (e.g. bot
                # restarted mid-stream), search from ~10 min before the session start
                # rather than from t=0. Avoids Whisper scanning an hour of unrelated audio.
                session_span = (
                    (events[-1].epoch - events[0].epoch) if len(events) >= 2 else 0.0
                )
                search_offset = max(0.0, vod_duration - session_span - 600)
                print(f"   [VodAlign] Session span={session_span/60:.0f}min, "
                      f"VOD={vod_duration/60:.0f}min → search offset={search_offset/60:.0f}min.")
                vod_start_epoch, align_conf, vod_align_details = estimate_vod_start_epoch(
                    vod_path, events, session_start or 0.0,
                    start_offset_s=search_offset,
                )
                if align_conf < 0.5:
                    print(f"   [VodAlign] WARNING: low confidence ({align_conf:.0%}). "
                          f"Review reel_report.md before posting.")

            recordings = [Recording(
                path=vod_path,
                start_epoch=vod_start_epoch,
                end_epoch=vod_start_epoch + vod_duration,
                duration=vod_duration,
                start_source="whisper_anchor_match" if not dry_run else "assumed_session_start",
            )]
            print(f"   [Clips] VOD span: "
                  f"{datetime.fromtimestamp(vod_start_epoch).strftime('%H:%M:%S')} → "
                  f"{datetime.fromtimestamp(vod_start_epoch + vod_duration).strftime('%H:%M:%S')} "
                  f"(local); NVENC {'available' if nvenc_ok else 'NOT available'}.")

        for c in candidates:
            align_candidate(c, events, session_start)
            map_to_recording(c, recordings, pre, post)

        if vod_path:
            rec_dir = os.path.dirname(vod_path)
        else:
            rec_dir = OBS_RECORDINGS_DIR
        day_dir = os.path.join(rec_dir, "clips", date_str)

        # ── Sort & cap ───────────────────────────────────────────────────────
        # reel=True (default): cut ALL aligned candidates in chronological order
        #   — reel needs them all; individual clips are a ranked subset view.
        # reel=False, clips=True: cut top N by score; cap the rest.
        cuttable = [c for c in candidates if c.status == "pending"]

        if reel:
            # All aligned candidates, chronological — no cap.
            cuttable.sort(key=lambda c: c.anchor_start_epoch or float("inf"))
            cut_list = cuttable
            force_cut = True   # reel concat requires uniform frame-accurate encoding
        elif clips:
            # Clips-only (--no-reel): top N by score, stream-copy OK.
            cuttable.sort(key=lambda c: c.score, reverse=True)
            cap_set = set(id(c) for c in cuttable[top:])
            for c in candidates:
                if id(c) in cap_set:
                    c.status = "capped"
            cut_list = cuttable[:top]
            force_cut = force_reencode
        else:
            cut_list = []
            force_cut = force_reencode

        # ── Cut ─────────────────────────────────────────────────────────────
        rank = 1
        for c in cut_list:
            if c.status in ("missed", "error"):
                continue
            slug = _slugify(c.title)
            out_path = os.path.join(day_dir, f"{rank:02d}_{slug}.mp4")
            if dry_run:
                score_str = f" score={c.score}/10" if c.score else ""
                print(f"      Clip {c.index} [rank {rank:02d}{score_str}]: "
                      f"would cut → {os.path.basename(out_path)} "
                      f"(@{c.clip_start_offset:.1f}s +{c.clip_duration:.1f}s, "
                      f"via {c.matched_via})")
                c.out_path = out_path
                c.status = "cut"
                c.final_title = c.suggested_title or c.title
            else:
                cut_clip(c, out_path, force_cut, nvenc_ok)
                if c.status == "cut":
                    score_str = f" score={c.score}/10" if c.score else ""
                    print(f"      Clip {c.index} [rank {rank:02d}{score_str}]: "
                          f"CUT ({c.cut_method}) → {os.path.basename(out_path)}")
                else:
                    print(f"      Clip {c.index}: ERROR — {c.error}")
            rank += 1

        # ── Report capped/missed ─────────────────────────────────────────────
        for c in candidates:
            if c.status == "capped":
                score_str = f" score={c.score}/10" if c.score else ""
                print(f"      Clip {c.index}{score_str}: CAPPED (below top {top})")
            elif c.status == "missed":
                print(f"      Clip {c.index}: MISSED — {c.error}")

        if clips and not dry_run:
            await generate_titles(candidates, act_name)
            for c in candidates:
                if c.status == "cut":
                    write_title_sidecar(c)

        report_path = write_report(candidates, day_dir, date_str, act_name, recordings)
        print(f"   [Clips] Report → {report_path}")

        # ── Reel assembly ────────────────────────────────────────────────────
        reel_path_out = None
        reel_report_out = None
        if reel and not dry_run:
            # Short-session guard: skip reel if session is too brief or has
            # too few candidates to be worth a reel.
            aligned_count = sum(1 for c in candidates if c.anchor_start_epoch is not None)
            session_span_min = (
                (events[-1].epoch - events[0].epoch) / 60.0 if len(events) >= 2 else 0.0
            )
            if aligned_count < 3:
                print(f"   [Reel] SKIP — only {aligned_count} aligned candidate(s) "
                      f"(minimum 3 required). Session too short for a reel.")
            elif session_span_min < REEL_MIN_MINUTES:
                print(f"   [Reel] SKIP — session span {session_span_min:.0f} min "
                      f"< REEL_MIN_MINUTES ({REEL_MIN_MINUTES} min). Session too short for a reel.")
            else:
                try:
                    reel_path_out = cut_reel(candidates, day_dir, act_name, nvenc_ok)
                    reel_title, reel_desc = await generate_reel_title(candidates, act_name)
                    reel_report_out = write_reel_report(
                        candidates, reel_path_out, reel_title, reel_desc,
                        day_dir, date_str, act_name, vod_align_details,
                    )
                    print(f"   [Reel] Report → {reel_report_out}")
                except RuntimeError as e:
                    print(f"   [Reel] Assembly failed: {e}")

        summaries.append({
            "artifact": md_path,
            "activity": act_name,
            "total": len(candidates),
            "cut": sum(1 for c in candidates if c.status == "cut"),
            "capped": sum(1 for c in candidates if c.status == "capped"),
            "missed": sum(1 for c in candidates if c.status == "missed"),
            "error": sum(1 for c in candidates if c.status == "error"),
            "report": report_path,
            "out_dir": day_dir,
            "reel_path": reel_path_out,
            "reel_report": reel_report_out,
            "candidates": candidates,
        })

    return {"date": date_str, "sessions": summaries, "vod_align": vod_align_details}
