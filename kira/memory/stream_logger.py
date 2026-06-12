# stream_logger.py — Persistent stream session logging.
#
# Writes three artifacts per stream session to:
#   logs/streams/YYYY-MM-DD_HH-MM_<activity>/
#       transcript.md  — human-readable markdown timeline
#       events.jsonl   — machine-readable JSONL (one JSON object per line)
#       summary.md     — Opus-generated post-stream analysis (written at session end)
#
# Design rules:
#   • log() is always sync + thread-safe — zero blocking on hot paths.
#   • A background asyncio task drains the buffer to disk every 30 s.
#   • All disk I/O is wrapped in try/except; a failure sets _disk_error=True
#     and suspends writes without crashing Kira.
#   • Disk-full (OSError) is handled the same way.
#   • Session directory is created lazily at start(); logs/ parent is created
#     if it doesn't exist.

import asyncio
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any


class StreamLogger:
    """Non-blocking, thread-safe stream session logger.

    Typical lifecycle
    -----------------
    logger = StreamLogger()
    await logger.start(activity="007 First Light", mode="streamer", preset="Action Game Stream")

    logger.log("voice_input", text="Alright, let's go")         # sync, anywhere
    logger.log("triage_decision", result="RESPOND", latency_ms=230)

    await logger.finish(ai_core=bot.ai_core)  # flushes + optional Opus summary
    """

    FLUSH_INTERVAL = 30.0        # seconds between automatic disk flushes

    def __init__(self, base_dir: str = "logs/streams"):
        self._base_dir = base_dir
        self._session_dir: str = ""
        self._transcript_path: str = ""
        self._events_path: str = ""
        self._summary_path: str = ""

        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._disk_error: bool = False
        self._started: bool = False
        self._is_running: bool = False
        self._writer_task: asyncio.Task | None = None
        self._session_start_ts: float = 0.0

        self._activity: str = ""
        self._mode: str = ""
        self._preset: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def log(self, type: str, **fields: Any) -> None:
        """Non-blocking, thread-safe event log. Safe from any sync or async context."""
        if self._disk_error or not self._started:
            return
        try:
            event: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "type": type,
            }
            event.update(fields)
            with self._lock:
                self._buffer.append(event)
        except Exception as e:
            print(f"   [StreamLogger] log() error: {e}", file=sys.stderr)

    async def start(self, activity: str, mode: str, preset: str) -> None:
        """Create session directory and files, start background writer task.
        Safe to call multiple times — subsequent calls are no-ops if already started."""
        if self._started:
            return
        try:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", activity or "general").strip("_").lower()[:30] or "general"
            ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self._session_dir = os.path.join(self._base_dir, f"{ts_str}_{slug}")
            os.makedirs(self._session_dir, exist_ok=True)

            self._transcript_path = os.path.join(self._session_dir, "transcript.md")
            self._events_path     = os.path.join(self._session_dir, "events.jsonl")
            self._summary_path    = os.path.join(self._session_dir, "summary.md")

            date_display = datetime.now().strftime("%Y-%m-%d %H:%M")
            header = (
                f"# Stream Session: {date_display}\n"
                f"## Activity: {activity or 'General'}\n"
                f"## Mode: {mode} / {preset}\n\n"
                f"---\n\n"
            )
            with open(self._transcript_path, "w", encoding="utf-8") as f:
                f.write(header)
            with open(self._events_path, "w", encoding="utf-8") as f:
                pass  # touch

            self._activity         = activity
            self._mode             = mode
            self._preset           = preset
            self._session_start_ts = time.time()
            self._is_running       = True
            self._started          = True
            self._disk_error       = False

            self._writer_task = asyncio.create_task(
                self._writer_loop(), name="StreamLogger-writer"
            )

            # First event
            self.log(
                "session_start",
                activity=activity,
                mode=mode,
                preset=preset,
            )
            print(f"   [StreamLogger] Session started → {self._session_dir}")

        except Exception as e:
            print(f"   [StreamLogger] Failed to start session: {e}", file=sys.stderr)
            self._disk_error = True

    async def finish(self, ai_core=None) -> None:
        """Flush remaining events, optionally generate summary.md, stop writer task.
        Safe to call even if start() was never called or previously failed.
        Summary generation is fully bulletproofed — any failure (Claude, fallback,
        whatever) is caught, logged with traceback, and SWALLOWED. finish() must
        never crash the process."""
        if not self._started:
            return
        self._is_running = False

        duration_s = int(time.time() - self._session_start_ts)
        self.log("session_end", duration_s=duration_s)

        try:
            await self._flush()
        except Exception as e:
            print(f"   [StreamLogger] Final flush error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"   [StreamLogger] Writer task cancel error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        if ai_core is not None and not self._disk_error:
            # Bulletproof summary: ANY failure here is caught and logged, never propagates.
            try:
                await self._generate_summary(ai_core)
            except BaseException as e:
                # BaseException catches SystemExit / KeyboardInterrupt too — a runaway
                # summary path must not be allowed to kill the process.
                print(f"   [StreamLogger] Summary generation failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        self._started = False
        print(f"   [StreamLogger] Session closed → {self._session_dir}")

    async def restart(self, activity: str, mode: str, preset: str, ai_core=None) -> None:
        """Close the current session and open a new one. Awaitable — call from async context."""
        await self.finish(ai_core=ai_core)
        await self.start(activity=activity, mode=mode, preset=preset)

    # ── Background writer ──────────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        while self._is_running:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            try:
                await self._flush()
            except Exception as e:
                print(f"   [StreamLogger] Writer loop error: {e}", file=sys.stderr)

    async def _flush(self) -> None:
        """Drain the in-memory buffer to disk atomically."""
        if self._disk_error:
            return
        with self._lock:
            if not self._buffer:
                return
            events, self._buffer = self._buffer[:], []

        try:
            jsonl_chunk = "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(jsonl_chunk)
                f.flush()

            transcript_lines = []
            for e in events:
                line = self._format_transcript_line(e)
                if line:
                    transcript_lines.append(line)
            if transcript_lines:
                with open(self._transcript_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(transcript_lines) + "\n")
                    f.flush()

        except OSError as e:
            print(
                f"   [StreamLogger] Disk write failed ({e}) — logging suspended for this session.",
                file=sys.stderr,
            )
            self._disk_error = True
        except Exception as e:
            print(f"   [StreamLogger] Unexpected flush error: {e}", file=sys.stderr)

    # ── Transcript line formatter ──────────────────────────────────────────────

    def _format_transcript_line(self, event: dict) -> str:
        """Convert a structured event dict to a human-readable markdown line.
        Returns empty string for unknown types (still written to JSONL)."""
        try:
            ts_raw  = event.get("ts", "")
            hms     = ts_raw[11:19] if len(ts_raw) >= 19 else ts_raw
            t       = f"**[{hms}]**"
            etype   = event.get("type", "")

            if etype == "session_start":
                return (
                    f"{t} [SESSION START] Activity: {event.get('activity', '')} | "
                    f"Mode: {event.get('mode', '')} | Preset: {event.get('preset', '')}\n"
                )

            if etype == "session_end":
                dur = event.get("duration_s", 0)
                h, rem = divmod(dur, 3600)
                m, s   = divmod(rem, 60)
                return f"\n---\n\n{t} [SESSION END] Duration: {h:02d}:{m:02d}:{s:02d}\n"

            if etype == "voice_input":
                text = event.get("text", "")[:200]
                return f"{t} [VOICE] Jonny: \"{text}\""

            if etype == "loopback_transcript":
                text = event.get("text", "")[:120]
                return f"{t} [LOOPBACK] \"{text}\""

            if etype == "chat_message":
                plat = event.get("platform", "").upper()
                user = event.get("user", "")
                text = event.get("text", "")[:150]
                return f"{t} [CHAT/{plat}] {user}: \"{text}\""

            if etype == "triage_decision":
                result  = event.get("result", "")
                reason  = event.get("reason", "")
                lat     = event.get("latency_ms", "")
                rsuf    = f" ({reason})" if reason else ""
                lsuf    = f" — {lat}ms"  if lat != "" else ""
                return f"{t} [TRIAGE] {result}{rsuf}{lsuf}"

            if etype == "kira_response":
                emotion = event.get("emotion", "")
                text    = event.get("text", "")[:200]
                source  = event.get("source", "")
                src_tag = f" via {source}" if source else ""
                return f"{t} [KIRA{src_tag}] ({emotion}): \"{text}\""

            if etype == "chat_batch_response":
                n    = event.get("batch_size", "")
                text = event.get("text", "")[:150]
                return f"{t} [KIRA (Chat Batch of {n})]: \"{text}\""

            if etype == "activity_switch":
                act  = event.get("activity", "")
                atyp = event.get("activity_type", "")
                atyp_str = f" ({atyp})" if atyp else ""
                return (
                    f"\n---\n\n"
                    f"{t} [ACTIVITY SWITCHED: {act}{atyp_str}]\n"
                )

            if etype == "activity_change":
                act  = event.get("activity", "")
                atyp = event.get("activity_type", "")
                return f"{t} [ACTIVITY] → {act} ({atyp})"

            if etype == "preset_load":
                return f"{t} [PRESET] {event.get('preset', '')}"

            if etype == "mode_change":
                return f"{t} [MODE] → {event.get('mode', '')}"

            if etype == "vision_call":
                lat     = event.get("latency_ms", "")
                summary = event.get("summary", "")[:80]
                return f"{t} [VISION] {summary} ({lat}ms)"

            if etype == "audio_summary":
                return f"{t} [AUDIO] {event.get('summary', '')[:100]}"

            if etype == "cutscene_detected":
                v      = event.get("vision_hit", False)
                a      = event.get("audio_hit", False)
                action = event.get("action", "suppressed")
                return (
                    f"{t} [CUTSCENE] vision={'HIT' if v else 'miss'}, "
                    f"audio={'HIT' if a else 'miss'} — {action}"
                )

            if etype == "vram_sample":
                # New schema: whole-card NVML read (used_gb). Fall back to the
                # old allocated_gb key for historical logs.
                used  = event.get("used_gb", event.get("allocated_gb", 0))
                total = event.get("total_gb", 0)
                return f"{t} [VRAM] {used:.1f} / {total:.0f} GB used (whole card)"

            if etype == "auto_degrade":
                action  = event.get("action", "")
                from_v  = event.get("from", "")
                to_v    = event.get("to", "")
                trigger = event.get("trigger", "")
                return f"{t} [AUTO-DEGRADE] {action} {from_v} → {to_v} (trigger: {trigger})"

            if etype == "auto_degrade_reset":
                return f"{t} [AUTO-DEGRADE RESET] {event.get('detail', '')}"

            if etype == "highlight_captured":
                return f"{t} [HIGHLIGHT] {event.get('highlight', '')[:120]}"

            if etype == "memory_extraction":
                count = event.get("count", 1)
                return f"{t} [MEMORY] {count} fact(s) extracted"

            if etype == "error":
                return f"{t} [ERROR] {event.get('message', '')[:200]}"

            if etype == "warning":
                return f"{t} [WARNING] {event.get('message', '')[:200]}"

            if etype == "llm_fallback":
                reason = event.get("reason", "unknown")
                model  = event.get("model", "local")
                return f"{t} [⚠ LLM FALLBACK → {model}] reason: {reason}"

            if etype == "kira_response_model":
                return f"{t} [LLM] model={event.get('model', '?')}"

            if etype == "moment_type":
                return f"{t} [MOMENT] → {event.get('moment', '?').upper()}"

            if etype == "drive_mode_on":
                src = event.get("source", "auto")
                agenda = event.get("agenda", [])
                items = "; ".join(agenda[:3]) if agenda else "(seeding...)"
                return f"{t} [DRIVE MODE ON] source={src} | agenda: {items}"

            if etype == "drive_mode_off":
                return f"{t} [DRIVE MODE OFF]"

            if etype == "drive_agenda_seeded":
                agenda = event.get("agenda", [])
                items = " | ".join(f"{i+1}. {a}" for i, a in enumerate(agenda))
                return f"{t} [DRIVE AGENDA] {items}"

            if etype == "session_tokens":
                s_in  = event.get("sonnet_in",  0)
                s_out = event.get("sonnet_out", 0)
                s_cr  = event.get("sonnet_cache_read", 0)
                o_in  = event.get("opus_in",  0)
                o_out = event.get("opus_out", 0)
                h_in  = event.get("haiku_in",  0)
                h_out = event.get("haiku_out", 0)
                g_in  = event.get("groq_in",  0)
                g_out = event.get("groq_out", 0)
                return (
                    f"{t} [SESSION TOKENS] "
                    f"sonnet={s_in}in/{s_out}out(cache_read={s_cr}) | "
                    f"opus={o_in}in/{o_out}out | "
                    f"haiku={h_in}in/{h_out}out | "
                    f"groq={g_in}in/{g_out}out"
                )

            # Unknown type — still written to JSONL but skipped in transcript
            return ""

        except Exception:
            return ""

    # ── Opus summary generation ────────────────────────────────────────────────

    async def _generate_summary(self, ai_core) -> None:
        """Read events.jsonl + transcript.md and call Claude Opus to produce summary.md.
        Requires ai_core.claude_inference() to be available."""
        if not self._events_path or not os.path.exists(self._events_path):
            return
        if not getattr(ai_core, "anthropic_client", None):
            print("   [StreamLogger] Skipping summary — Claude client unavailable.", file=sys.stderr)
            return

        try:
            with open(self._events_path,     "r", encoding="utf-8") as f:
                raw_events = f.read()
            with open(self._transcript_path, "r", encoding="utf-8") as f:
                raw_transcript = f.read()
        except Exception as e:
            print(f"   [StreamLogger] Could not read session files for summary: {e}", file=sys.stderr)
            return

        # Preserve the full transcript for the PENDING fallback (backfill_lore can
        # regenerate the summary later from this raw material).
        full_transcript = raw_transcript

        # Truncate aggressively — Opus context window is large but these files can be huge
        def _trunc(text: str, head: int = 20000, tail: int = 10000) -> str:
            if len(text) <= head + tail:
                return text
            return text[:head] + "\n\n... [truncated] ...\n\n" + text[-tail:]

        raw_events     = _trunc(raw_events,     head=20000, tail=10000)
        raw_transcript = _trunc(raw_transcript, head=20000, tail=10000)

        duration_s = int(time.time() - self._session_start_ts)
        h, rem = divmod(duration_s, 3600)
        m, _   = divmod(rem, 60)
        duration_str = f"{h}h {m:02d}m"

        prompt = (
            "You are reviewing a Kira AI VTuber stream session log for post-stream analysis.\n\n"
            f"Activity : {self._activity}\n"
            f"Mode     : {self._mode} / Preset: {self._preset}\n"
            f"Duration : {duration_str}\n\n"
            "=== TRANSCRIPT (excerpt) ===\n"
            f"{raw_transcript}\n\n"
            "=== EVENT LOG (excerpt) ===\n"
            f"{raw_events}\n\n"
            "Produce a detailed summary with EXACTLY these markdown sections:\n"
            "## Stats\n"
            "Count: total Kira responses, voice inputs, chat messages (Twitch/YouTube breakdown), "
            "triage breakdown (RESPOND/BRIEF/STAY_QUIET percentages), "
            "cutscene suppressions, memory extractions, highlights captured.\n"
            "Include average triage latency if visible.\n\n"
            "## Notable Issues\n"
            "VRAM events, API timeouts/errors, audio meta-replies that escaped suppression, "
            "unexpected STAY_QUIET decisions, anything that looked wrong.\n\n"
            "## Personality Highlights\n"
            "Memorable moments, running bits, lore built for chatters, standout Kira lines.\n\n"
            "## Suggestions\n"
            "2–5 specific, actionable optimisations based on what you observed. "
            "Reference timestamps when relevant.\n\n"
            "Be data-driven and specific. Do not pad."
        )

        print("   [StreamLogger] Generating post-stream summary via Sonnet...")
        try:
            response = await asyncio.wait_for(
                ai_core.claude_inference(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=(
                        "You are a technical analyst reviewing an AI companion's stream session logs. "
                        "Be concise, specific, and reference timestamps. Do not speculate beyond the data."
                    ),
                    max_tokens=2000,
                    use_sonnet=True,  # K: post-stream tech summary — Sonnet
                ),
                timeout=45,
            )
        except asyncio.TimeoutError:
            print("   [WARN] stream_logger summary LLM call timed out after 45s — "
                  "writing PENDING checkpoint for later backfill", file=sys.stderr)
            self._write_pending_summary(full_transcript, duration_s)
            return
        except Exception as e:
            print(f"   [WARN] stream_logger: Sonnet summary call failed: {e} — "
                  f"writing PENDING checkpoint for later backfill", file=sys.stderr)
            self._write_pending_summary(full_transcript, duration_s)
            return

        if not response:
            self._write_pending_summary(full_transcript, duration_s)
            return

        date_display = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self._summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Session Summary — {date_display}\n\n")
            f.write(f"**Activity:** {self._activity}  \n")
            f.write(f"**Duration:** {duration_str}  \n")
            f.write(f"**Mode:** {self._mode} / {self._preset}\n\n")
            f.write("---\n\n")
            f.write(response.strip())
            f.write("\n")

        print(f"   [StreamLogger] Summary written → {self._summary_path}")

    def _write_pending_summary(self, transcript: str, duration_s: int) -> None:
        """Persist raw session material as a PENDING_<slug>.json checkpoint when the
        post-stream summary couldn't be generated (timeout/failure). backfill_lore.py
        picks these up (logs/sessions_raw/PENDING_*.json) and regenerates the summary
        later, so a hung Opus call never costs us the session's raw material."""
        try:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", self._activity or "general").strip("_").lower()[:30] or "general"
            pending_dir = os.path.join("logs", "sessions_raw")
            os.makedirs(pending_dir, exist_ok=True)
            payload = {
                "activity_slug": slug,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "activity": self._activity or "general",
                "transcript": transcript or "",
                "highlights": [],
                "duration_min": max(0, int(duration_s) // 60),
            }
            pending_path = os.path.join(pending_dir, f"PENDING_{slug}.json")
            tmp_path = pending_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, pending_path)
            print(f"   [StreamLogger] PENDING checkpoint written → {pending_path} "
                  f"(run backfill_lore.py --pending-only to regenerate the summary)", file=sys.stderr)
        except Exception as e:
            print(f"   [WARN] stream_logger: could not write PENDING checkpoint: {e}", file=sys.stderr)
