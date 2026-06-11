# media_watch.py — Media Watch Mode
# ─────────────────────────────────────────────────────────────────────────────
# A SEPARATE, isolated mode for genuine episode/movie understanding.
#
# What it does (and what it does NOT do):
#   - Captures a frame every ~2-3s from a target window into a rolling buffer.
#   - Every ~15-20s, sends the buffered sequence to cloud vision asking
#     "what HAPPENED across these frames" — actions, sequence, scene change.
#   - Appends the result to a bounded EPISODE LOG (timestamped event timeline).
#   - Exposes the episode log for question-answering (Kira draws on the timeline
#     instead of a single stale snapshot).
#   - Optional throttled in-character reactions on notable events.
#
#   - Does NOT modify default companion mode, VN autopilot, or their loops.
#   - Does NOT replace vision_agent / observer loop / heartbeat.
#   - Visual-only: no audio understanding (documented limitation).
#
# Cost: ~75-100 vision calls per 24-min episode at 18s interval, gpt-4o-mini
# pricing for ~8 low-detail frames per call ≈ ~$0.10-0.25 per episode.

import asyncio
import base64
import time
from collections import deque
from io import BytesIO

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pygetwindow as _pgw
    PYGETWINDOW_AVAILABLE = True
except Exception:
    PYGETWINDOW_AVAILABLE = False


# ─── Prompts ─────────────────────────────────────────────────────────────────

SEQUENCE_ANALYSIS_PROMPT = (
    "You are watching a video alongside a viewer. The following frames are "
    "consecutive moments captured a few seconds apart from the same video. "
    "Your job is to describe what HAPPENED across this sequence — focus on "
    "ACTIONS, SEQUENCE, CHANGE, and CAUSE-AND-EFFECT. Not static description.\n\n"
    "Tell the story of this stretch in 2-4 sentences:\n"
    "  - Who appears and what they do (use any visible character names from "
    "    subtitles or name labels — otherwise describe by appearance).\n"
    "  - What changes between frames (movement, scene cuts, who arrives/leaves, "
    "    objects appearing, expressions shifting).\n"
    "  - Any dialogue / on-screen text you can read, attributed if possible.\n"
    "  - The emotional beat of the moment if it's clear.\n\n"
    "HONESTY RULES (important):\n"
    "  - If the frames are mostly transitions, blur, or you cannot tell what is "
    "    happening with reasonable confidence, start your reply with "
    "    'UNCERTAIN:' and briefly say what you CAN see.\n"
    "  - If frames look essentially identical (still scene, talking head, "
    "    paused video), say 'STATIC: <brief>' instead of inventing motion.\n"
    "  - Do not invent character names you cannot read on screen.\n"
    "  - Do not invent dialogue. Only quote text you can actually see.\n\n"
    "Be concrete. This summary will be Kira's memory of what happened."
)


# ─── Module ──────────────────────────────────────────────────────────────────

class MediaWatch:
    """Rolling-buffer sequence analyzer for movies / anime / video episodes.

    Architecture:
        capture_loop  — every frame_interval_s grabs window into self._frames
        analysis_loop — every analysis_interval_s sends self._frames to vision,
                        appends result to self.episode_log

    Both are bounded so 3-hour movies stay safe.
    """

    # gpt-4o-mini vision rough rate per call with ~8 low-detail images.
    # 8 frames * ~85 image tokens (low-detail) ≈ 680 input image tokens,
    # plus prompt+output overhead. Empirically ~$0.0015-0.003 per call.
    _COST_PER_CALL_USD = 0.002

    def __init__(self, vision_client, *,
                 frame_interval_s: float = 2.5,
                 analysis_interval_s: float = 18.0,
                 buffer_size: int = 8,
                 episode_log_max: int = 200,
                 cloud_call_timeout: float = 12.0):
        self.vision_client = vision_client

        # Config (tunable from dashboard at runtime).
        self.frame_interval_s: float = float(frame_interval_s)
        self.analysis_interval_s: float = float(analysis_interval_s)
        self.buffer_size: int = int(buffer_size)
        self._cloud_call_timeout: float = float(cloud_call_timeout)

        # State.
        self.enabled: bool = False
        self.is_running: bool = False
        self.window_title: str = ""

        # Rolling frame buffer — most recent N frames with timestamps.
        # Each entry: {"ts": float, "b64": str}
        self._frames: deque = deque(maxlen=self.buffer_size)

        # Episode log — bounded timeline of "what happened so far".
        # Each entry: {"ts": float, "t_rel_s": float, "summary": str,
        #              "uncertain": bool, "static": bool}
        self.episode_log: deque = deque(maxlen=episode_log_max)

        # Cost tracking.
        self._session_start_ts: float = 0.0
        self._calls_count: int = 0
        self._calls_cost_usd: float = 0.0
        self._last_analysis_ts: float = 0.0
        # Wall-clock midpoint of the frames the most-recent analysis covered.
        self._last_content_mid_ts: float = 0.0

        # Tasks.
        self._capture_task: asyncio.Task | None = None
        self._analysis_task: asyncio.Task | None = None

        # Optional reaction callback — invoked with (summary_text) after each
        # successful, non-static, non-uncertain analysis (throttled).
        # Set by bot wiring; left None means no spoken reactions.
        self.on_react = None
        # Single source of truth for the "React to scenes" toggle. The handler
        # (on_react) stays wired for the whole session; this bool decides whether
        # it actually fires. Reported verbatim in /status so the dashboard
        # checkbox can never desync (kills the old inverting blind-flip).
        self.reactions_enabled: bool = True
        self._last_react_ts: float = 0.0
        self.react_min_gap_s: float = 45.0  # at most one spoken react per 45s

        # Called with a short reason string when start() aborts after .enabled
        # was already set True (window not found, missing dep, etc). The bot
        # wires this to re-run _reconcile_modes so vision un-parks immediately —
        # the reconciler must never enforce an intent that failed to materialize.
        self.on_start_failed = None
        self.last_start_error: str = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _fail_start(self, reason: str):
        """Abort a start() that already had .enabled=True. Resets the intent flag
        so the reconciler doesn't keep enforcing a mode that never materialized,
        records the reason for the dashboard, and pokes the reconciler so vision
        un-parks right away."""
        self.enabled = False
        self.is_running = False
        self.last_start_error = reason
        print(f"   [MediaWatch] start FAILED — enabled reset ({reason}).")
        if self.on_start_failed:
            try:
                self.on_start_failed(reason)
            except Exception as e:
                print(f"   [MediaWatch] on_start_failed callback error: {e}")

    def start(self):
        """Start capture + analysis loops. Safe to call multiple times."""
        if not self.enabled:
            return
        if self.is_running:
            return
        if not PIL_AVAILABLE:
            self._fail_start("Pillow not available — cannot capture frames")
            return
        if not PYGETWINDOW_AVAILABLE:
            self._fail_start("pygetwindow not available — cannot target window")
            return
        if not self.window_title.strip():
            self._fail_start("no window title set")
            return
        if self.vision_client is None:
            self._fail_start("no vision client")
            return

        win = self._find_window()
        if win is None:
            titles = self.list_open_windows()
            print("   [MediaWatch] Open windows:")
            for t in titles[:20]:
                print(f"      • {t}")
            self._fail_start(f"window not found for '{self.window_title}'")
            return

        self.is_running = True
        self._session_start_ts = time.time()
        self._calls_count = 0
        self._calls_cost_usd = 0.0
        self._frames.clear()
        self.episode_log.clear()
        self._last_analysis_ts = 0.0
        self._last_react_ts = 0.0
        self._last_content_mid_ts = 0.0
        self.last_start_error = ""
        self._capture_task = asyncio.ensure_future(self._capture_loop())
        self._analysis_task = asyncio.ensure_future(self._analysis_loop())
        print(
            f"   [MediaWatch] Started — window='{self.window_title}', "
            f"capture every {self.frame_interval_s:.1f}s, "
            f"analysis every {self.analysis_interval_s:.1f}s, "
            f"buffer={self.buffer_size}."
        )

    def stop(self):
        """Stop the loops. Keeps episode_log intact for end-of-session recall."""
        self.is_running = False
        self.enabled = False
        for t in (self._capture_task, self._analysis_task):
            if t and not t.done():
                t.cancel()
        self._capture_task = None
        self._analysis_task = None
        h, rem = divmod(int(time.time() - self._session_start_ts), 3600)
        m = rem // 60
        print(
            f"   [MediaWatch] Stopped. {self._calls_count} analysis calls, "
            f"~${self._calls_cost_usd:.3f} cloud spend, ran {h}h {m}m. "
            f"Episode log: {len(self.episode_log)} events."
        )

    # ── Status / cost (dashboard reads these) ────────────────────────────────

    def get_status_str(self) -> str:
        if not self.is_running:
            if self._calls_count:
                return (
                    f"MediaWatch: OFF — last session "
                    f"{self._calls_count} calls / ${self._calls_cost_usd:.3f}"
                )
            return "MediaWatch: OFF"
        runtime = time.time() - self._session_start_ts
        h, rem = divmod(int(runtime), 3600)
        m = rem // 60
        return (
            f"MediaWatch: ON — {len(self.episode_log)} events, "
            f"{self._calls_count} calls, ~${self._calls_cost_usd:.3f}, "
            f"{h}h {m}m"
        )

    # ── Window helpers ───────────────────────────────────────────────────────

    @staticmethod
    def list_open_windows() -> list[str]:
        if not PYGETWINDOW_AVAILABLE:
            return ["(pygetwindow not available)"]
        try:
            return sorted(
                {w.title for w in _pgw.getAllWindows() if (w.title or "").strip()},
                key=str.lower,
            )
        except Exception as e:
            return [f"(error listing windows: {e})"]

    @staticmethod
    def get_foreground_window_title() -> str:
        """Return the title of the current foreground (focused) window, or "".

        Used for C1 auto-target: when Media Watch is toggled on with an empty
        window field, we pick whatever is in focus so the field becomes an
        OVERRIDE, not a requirement."""
        if not PYGETWINDOW_AVAILABLE:
            return ""
        try:
            win = _pgw.getActiveWindow()
            return (getattr(win, "title", "") or "").strip()
        except Exception:
            return ""

    def _find_window(self):
        if not PYGETWINDOW_AVAILABLE or not self.window_title:
            return None
        try:
            needle = self.window_title.strip().lower()
            for w in _pgw.getAllWindows():
                if needle in (w.title or "").lower():
                    return w
        except Exception:
            pass
        return None

    def _grab_frame_sync(self):
        """Sync capture of the target window. Returns PIL Image or None."""
        if not PIL_AVAILABLE or not PYGETWINDOW_AVAILABLE:
            return None
        win = self._find_window()
        if win is None or win.width <= 0 or win.height <= 0:
            return None
        bbox = (win.left, win.top, win.left + win.width, win.top + win.height)
        try:
            return ImageGrab.grab(bbox=bbox)
        except Exception as e:
            print(f"   [MediaWatch] Capture error: {e}")
            return None

    # ── Capture loop ─────────────────────────────────────────────────────────

    async def _capture_loop(self):
        """Periodically grab a frame and push it into the rolling buffer."""
        try:
            while self.is_running:
                try:
                    img = await asyncio.to_thread(self._grab_frame_sync)
                    if img is not None:
                        # Downscale to keep tokens cheap. 720p long-edge is plenty
                        # for sequence sense; gpt-4o-mini "low" detail is ~85 tokens
                        # per image regardless, but smaller jpeg keeps upload tiny.
                        def _encode(im):
                            im.thumbnail((1280, 720))
                            buf = BytesIO()
                            im.save(buf, format="JPEG", quality=65)
                            return base64.b64encode(buf.getvalue()).decode()
                        b64 = await asyncio.to_thread(_encode, img)
                        self._frames.append({"ts": time.time(), "b64": b64})
                except Exception as e:
                    print(f"   [MediaWatch] capture tick error: {e}")
                await asyncio.sleep(max(0.5, self.frame_interval_s))
        except asyncio.CancelledError:
            pass

    # ── Analysis loop ────────────────────────────────────────────────────────

    async def _analysis_loop(self):
        """Periodically send the frame buffer to vision and log the result."""
        try:
            # Small warmup so the first call has at least a couple of frames.
            await asyncio.sleep(min(self.analysis_interval_s, self.frame_interval_s * 2))
            while self.is_running:
                try:
                    if len(self._frames) >= 2:
                        await self._run_analysis_once()
                except Exception as e:
                    print(f"   [MediaWatch] analysis tick error: {e}")
                await asyncio.sleep(max(5.0, self.analysis_interval_s))
        except asyncio.CancelledError:
            pass

    async def _run_analysis_once(self):
        if not self.vision_client:
            return

        frames_snapshot = list(self._frames)
        # Build OpenAI message content with all frames + the prompt.
        content = [{"type": "text", "text": SEQUENCE_ANALYSIS_PROMPT}]
        for f in frames_snapshot:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{f['b64']}",
                    "detail": "low",
                },
            })

        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": content}],
                    max_tokens=260,
                    temperature=0.2,
                ),
                timeout=self._cloud_call_timeout,
            )
        except asyncio.TimeoutError:
            print(
                f"   [MediaWatch] Analysis TIMEOUT after "
                f"{self._cloud_call_timeout:.1f}s — skipping this interval."
            )
            return
        except Exception as e:
            print(f"   [MediaWatch] Analysis call failed: {e}")
            return

        latency = time.monotonic() - t0
        try:
            summary = (resp.choices[0].message.content or "").strip()
        except Exception:
            summary = ""
        if not summary:
            return

        self._calls_count += 1
        self._calls_cost_usd += self._COST_PER_CALL_USD

        upper = summary.upper()
        uncertain = upper.startswith("UNCERTAIN")
        static = upper.startswith("STATIC")

        now = time.time()
        t_rel = now - self._session_start_ts
        # Wall-clock midpoint of the frames this analysis actually covers — the
        # "when did this content happen" anchor for sense->speak lag metrics.
        try:
            _f_ts = [f["ts"] for f in frames_snapshot if "ts" in f]
            content_mid_ts = (min(_f_ts) + max(_f_ts)) / 2.0 if _f_ts else now
        except Exception:
            content_mid_ts = now
        entry = {
            "ts": now,
            "t_rel_s": t_rel,
            "summary": summary,
            "uncertain": uncertain,
            "static": static,
            "content_mid_ts": content_mid_ts,
        }
        self.episode_log.append(entry)
        self._last_analysis_ts = now
        self._last_content_mid_ts = content_mid_ts

        h, rem = divmod(int(t_rel), 3600)
        m, s = divmod(rem, 60)
        tag = "UNCERTAIN " if uncertain else ("STATIC " if static else "")
        # Print one-line digest so dashboard/terminal shows the event timeline.
        print(
            f"   [MediaWatch] +{h:02d}:{m:02d}:{s:02d} {tag}"
            f"({latency:.1f}s, ~${self._calls_cost_usd:.3f}) "
            f"{summary[:140]}{'…' if len(summary) > 140 else ''}"
        )

        # Fire optional reaction callback for substantive events only.
        if (
            self.on_react
            and self.reactions_enabled
            and not uncertain
            and not static
            and (now - self._last_react_ts) >= self.react_min_gap_s
        ):
            self._last_react_ts = now
            try:
                # Fire-and-forget: don't block the analysis loop.
                maybe = self.on_react(summary)
                if asyncio.iscoroutine(maybe):
                    asyncio.ensure_future(maybe)
            except Exception as e:
                print(f"   [MediaWatch] on_react error: {e}")

    # ── Context for question answering ───────────────────────────────────────

    def get_episode_context(self, max_entries: int = 10,
                            char_budget: int = 2600) -> str:
        """Return a formatted timeline string for prompt injection.

        Keeps the most recent `max_entries` events VERBATIM (the beats Kira is most
        likely to react to), and rolls everything older into a compact "earlier in
        the episode" digest so the block stays bounded no matter how long the movie
        runs. This is what keeps the FILM/EPISODE LOG from growing unbounded and
        evicting the dialogue summary from the scene-block budget.

        `char_budget` is a soft cap on the verbatim section; older recent entries are
        themselves condensed to first-sentence if the tail is still too large.
        """
        if not self.episode_log:
            return ""
        all_entries = list(self.episode_log)
        recent = all_entries[-max_entries:]
        older = all_entries[:-max_entries] if len(all_entries) > max_entries else []

        def _stamp(e) -> str:
            t = int(e["t_rel_s"])
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _first_sentence(text: str) -> str:
            text = (text or "").strip()
            # Strip the UNCERTAIN:/STATIC: tag for the digest.
            for tag in ("UNCERTAIN:", "STATIC:"):
                if text.upper().startswith(tag):
                    text = text[len(tag):].strip()
            for sep in (". ", "! ", "? "):
                idx = text.find(sep)
                if 0 < idx < 160:
                    return text[:idx + 1].strip()
            return text[:160].strip()

        lines = ["[MEDIA WATCH — episode event timeline, oldest first]"]

        # Rolled "earlier in the episode" digest from older substantive entries.
        if older:
            substantive = [e for e in older
                           if not e.get("uncertain") and not e.get("static")]
            picks = substantive or older
            # Spread the picks across the older span (start / middle / end) so the
            # digest reflects the arc, not just the oldest few beats.
            if len(picks) > 3:
                picks = [picks[0], picks[len(picks) // 2], picks[-1]]
            digest_bits = [f"{_stamp(e)} {_first_sentence(e['summary'])}" for e in picks]
            span = f"{_stamp(older[0])}–{_stamp(older[-1])}"
            lines.append(
                f"  [EARLIER IN THE EPISODE, condensed | {span}]: "
                + " ".join(digest_bits)
            )

        # Recent entries verbatim, trimmed to budget if needed.
        verbatim = []
        for e in recent:
            verbatim.append(f"  [{_stamp(e)}] {e['summary']}")
        block = "\n".join(verbatim)
        # If the verbatim tail blows the budget, condense the oldest of the recent
        # window to first-sentence until it fits.
        i = 0
        while len(block) > char_budget and i < len(recent) - 1:
            verbatim[i] = f"  [{_stamp(recent[i])}] {_first_sentence(recent[i]['summary'])}"
            block = "\n".join(verbatim)
            i += 1
        lines.append(block)

        lines.append(
            "[NOTE] This is Kira's actual visual record of what happened. "
            "When asked about earlier scenes, refer to this timeline. If a "
            "specific moment isn't in the timeline, say so honestly — do not "
            "invent details. Vision is visual-only (no audio)."
        )
        return "\n".join(lines)

    def has_context(self) -> bool:
        return len(self.episode_log) > 0

    def get_last_content_mid_ts(self) -> float:
        """Wall-clock midpoint of the frames the most-recent analysis covered.
        0.0 if no analysis has landed yet. Used for sense->speak lag metrics."""
        return self._last_content_mid_ts

    def get_latest_summary(self) -> str:
        """Most recent SUBSTANTIVE analysis summary (skips UNCERTAIN/STATIC).

        Used by the bot's stronger-signal media intensity gate: suppression in
        media mode keys off MediaWatch's own scene analysis, not a keyword in an
        audio caption."""
        for e in reversed(self.episode_log):
            if not e.get("uncertain") and not e.get("static"):
                return e.get("summary", "") or ""
        return ""