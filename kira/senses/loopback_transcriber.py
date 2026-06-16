# loopback_transcriber.py — Local Whisper transcription of system loopback audio.
#
# STAGE 1 SCOPE:
# Pumps a dedicated faster-whisper instance (distil-large-v3) on the same 16kHz
# mono float buffer that audio_agent already maintains in MEDIA mode. Produces a
# rolling, dedup'd, filtered transcript of foreground speech in whatever Jonny is
# watching. Visible on console + dashboard pane. NOT yet wired into Kira's
# llm_inference context — that comes in Stage 2 once the transcript is verified
# sane and the hallucination filters are confirmed to kill the "Thanks for
# watching!" class of Whisper artifacts.
#
# DESIGN NOTES:
# - Separate model instance (not the mic large-v3) so mic STT latency is never
#   blocked by loopback transcription. distil-large-v3 fp16 is ~1.5 GB and ~6×
#   faster than large-v3, with near-identical English WER. Fits inside the 5080's
#   headroom after llama-8B (~6-7 GB) + mic Whisper large-v3 (~3 GB) are loaded.
# - Reads audio_agent.buffer (deque of float32 at 16kHz). Audio agent already
#   downmixes/downsamples — we just snapshot the tail.
# - 5s tick / 8s window / 2s overlap. Each tick covers ~5s of new audio; the
#   2s overlap protects against words being clipped at boundaries. Duplicate
#   segments produced by the overlap are filtered by the dedupe step.
# - MEDIA mode only. MUSIC mode is OFF (mic capture of Jonny's own
#   guitar/singing — we don't want Kira reading her own input back as ambient
#   "speech").

import asyncio
import gc
import os
import re
import threading
import time
from collections import Counter, deque
from typing import Callable, Optional, TYPE_CHECKING

# HuggingFace on Windows tries to create symlinks in its cache, which requires
# either admin or Developer Mode and otherwise crashes with WinError 1314
# ("A required privilege is not held by the client"). Force file copies instead
# — costs a little extra disk, needs no privilege change, and must be set BEFORE
# huggingface_hub / faster_whisper is imported below.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    DEPS_AVAILABLE = True
except Exception as _e:
    np = None
    torch = None
    WhisperModel = None
    DEPS_AVAILABLE = False
    _DEPS_ERR = _e

if TYPE_CHECKING:
    from kira.senses.audio_agent import AudioAgent

LOOPBACK_LOG_DIR = "logs"

# Tunables (user greenlit "looser" defaults — tighten later only if reactions feel laggy).
TICK_SECONDS: float = 5.0          # how often we transcribe
WINDOW_SECONDS: float = 8.0        # how much audio each transcription consumes
# Overlap (2s) is implicit in WINDOW - TICK = 3s; the dedupe layer removes the
# duplicate segments that the overlap re-emits.

# Rolling transcript bounds.
TRANSCRIPT_MAX_AGE_SECONDS: float = 60.0   # drop segments older than this
TRANSCRIPT_MAX_CHARS: int = 1200           # hard cap on rendered string size

# Per-segment filter thresholds.
NO_SPEECH_PROB_THRESHOLD: float = 0.6      # drop segments where Whisper itself flags "no speech"
MIN_SEGMENT_CHARS: int = 3                 # drop ultra-short fragments
MIN_AVG_LOGPROB: float = -1.0              # drop very low-confidence segments
SILENCE_RMS_THRESHOLD: float = 0.003       # don't even call Whisper on near-silent windows

# Common Whisper hallucinations on music / silence / non-speech audio.
# All lowercased, matched after normalization (strip punctuation/whitespace).
HALLUCINATION_PHRASES = frozenset({
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "thank you for watching!",
    "thank you",
    "thank you.",
    "thank you!",
    "thanks",
    "thanks!",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "see you in the next video",
    "see you next video",
    "bye",
    "bye.",
    "bye!",
    "goodbye",
    "music",
    "[music]",
    "(music)",
    "♪",
    "♪♪",
    "♪ music ♪",
    "applause",
    "[applause]",
    "(applause)",
    "silence",
    "(silence)",
    "you",
    "you.",
    "yeah",
    "yeah.",
    "okay",
    "ok",
    "uh",
    "um",
    "hmm",
    "hm",
    "...",
    ".",
    "you know",
    "i don't know",
    "i'm sorry",
    "sorry",
})

# Pattern: a single token repeated 3+ times adjacent (Whisper's classic loop
# hallucination, e.g. 'no no no no').
_REPEATED_TOKEN_RE = re.compile(r"^(\S+)(?:[\s,.]+\1){2,}[\s.!?]*$", re.IGNORECASE)

# Within-segment burst thresholds — catches loops the adjacency regex misses,
# e.g. 'oh my god guys guys guys guys guys guys I don't' (top token isn't
# adjacent across the whole segment but dominates the token count).
_BURST_MIN_TOKENS: int = 4
_BURST_MIN_TOP_COUNT: int = 4
_BURST_MIN_TOP_RATIO: float = 0.4

# Across-segment burst: 3 consecutive *short* segments whose first word shares
# the same first 3 chars — catches the 'B-Words / B-Ward / B-Wartes' family of
# stutter loops Whisper emits when it gets confused by ambient music.
_SHORT_BURST_MAX_LEN: int = 18
_SHORT_BURST_PREFIX_CHARS: int = 3
_SHORT_BURST_WINDOW: int = 3
_SHORT_BURST_MIN_MATCHES: int = 2

# Self-TTS watcher polling cadence. Must be << the time between TTS sentences
# so we capture the actual moment speech ends, not the next 5s tick.
SPEECH_POLL_INTERVAL: float = 0.1


def _normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace/punctuation, collapse internal whitespace."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _has_repeated_token_burst(text: str) -> bool:
    """True if a single token dominates the segment (4+ occurrences and >=40%
    of all word tokens). Catches loops like 'guys guys guys guys guys' even
    when separated by filler words that defeat the adjacency regex."""
    tokens = re.findall(r"[a-z']+", text.lower())
    if len(tokens) < _BURST_MIN_TOKENS:
        return False
    _top_token, top_count = Counter(tokens).most_common(1)[0]
    if top_count >= _BURST_MIN_TOP_COUNT and (top_count / len(tokens)) >= _BURST_MIN_TOP_RATIO:
        return True
    return False


def _is_hallucination(text: str) -> bool:
    norm = _normalize(text).rstrip(".!?,")
    if not norm:
        return True
    if norm in HALLUCINATION_PHRASES:
        return True
    if _REPEATED_TOKEN_RE.match(text.strip()):
        return True
    if _has_repeated_token_burst(text):
        return True
    # Strip trailing punctuation again and check
    stripped = re.sub(r"[^\w\s']", "", norm).strip()
    if stripped in HALLUCINATION_PHRASES:
        return True
    return False


class LoopbackTranscriber:
    """Owns a dedicated faster-whisper model and a background pump thread that
    transcribes the audio_agent's loopback buffer on a fixed cadence. Maintains
    a rolling deque of (timestamp, text) segments accessible via get_transcript_text().

    Stage 1: not wired into Kira's context — just visible on console + dashboard
    pane for transcript-quality verification.
    """

    MODEL_NAME = "distil-large-v3"
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join("models", "whisper")
        self.model: Optional["WhisperModel"] = None
        self._audio_agent: Optional["AudioAgent"] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Rolling transcript: deque of dicts {"ts": float, "text": str}
        self._segments: deque = deque()
        self._lock = threading.Lock()

        # State for dedupe — last few normalized texts we already accepted.
        self._recent_normalized: deque = deque(maxlen=8)
        # Across-segment short-prefix-burst tracker (see _SHORT_BURST_* tunables).
        self._recent_short_prefixes: deque = deque(maxlen=_SHORT_BURST_WINDOW)

        # Self-TTS gating — set by start() from the dashboard. While Kira is
        # speaking we skip ticks entirely (her TTS goes through the same speakers
        # the loopback records). We also wait WINDOW_SECONDS after speech ends
        # before transcribing again, because the buffer's tail still contains
        # her TTS audio. Without this she sees her own lines as ambient speech.
        #
        # A dedicated watcher thread polls is_speaking_fn at SPEECH_POLL_INTERVAL
        # cadence so that _speech_last_active_ts reflects the *actual* end of
        # speech (within ~0.1s), not just the moment of the last 5s transcription
        # tick — otherwise the cooldown starts up to 5s too early and the buffer
        # tail still contains her voice when transcription resumes.
        self._is_speaking_fn: Optional[Callable[[], bool]] = None
        self._speech_last_active_ts: float = 0.0
        self._speech_watcher_thread: Optional[threading.Thread] = None
        self.total_ticks_skipped_self_tts: int = 0

        self._available = DEPS_AVAILABLE
        if not DEPS_AVAILABLE:
            print(f"   [LoopbackSTT] Disabled — dependencies missing: {_DEPS_ERR}")

        # Cosmetic boot state — set True the moment a start is queued (before the
        # ~20s model load completes) so the dashboard shows "starting…" instead of
        # a false "idle/off" flash during the autostart window. Cleared once the
        # pump thread is alive, or on a failed/aborted start.
        self._starting = False

        # Concurrency guard — makes the check-load-spawn sequence in start() and
        # the tear-down sequence in stop() mutually exclusive. Without this lock,
        # concurrent callers (dashboard toggle, API endpoint, audio-mode auto-start)
        # all pass the is_running() check during the ~20s model-load window, each
        # load a separate WhisperModel(cuda), and each spawn their own pump thread.
        # Three concurrent loads = ~4.5-6 GB of stolen CUDA compute that starves NVENC.
        self._start_lock = threading.Lock()

        # Diagnostics
        self.last_tick_time: float = 0.0
        self.last_tick_latency_ms: float = 0.0
        self.last_segment_count: int = 0
        self.total_segments_accepted: int = 0
        self.total_segments_filtered: int = 0

        # FIX 5: Rolling condensed dialogue summary (persists for the whole session,
        # unlike the 60s raw transcript window). Updated by bot.loopback_dialogue_summary_loop().
        self.dialogue_summary: str = ""
        self._summary_needs_update: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def mark_starting(self) -> None:
        """Flag that a start has been queued but not yet completed. Purely
        cosmetic — lets get_status_summary() report "starting…" during the
        model-load window so the dashboard never shows a false "idle/off" on
        first paint. Cleared by start() once the pump is alive (or on failure)."""
        if self._available:
            self._starting = True

    def start(
        self,
        audio_agent: "AudioAgent",
        is_speaking_fn: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Begin background transcription on the given audio agent's buffer.
        Returns True if started (or already running), False if unavailable.

        Thread-safe: _start_lock ensures exactly one WhisperModel instance and
        one pump thread exist no matter how many concurrent callers arrive. A
        second start() during the ~20s model-load window is a no-op.

        ``is_speaking_fn`` is an optional zero-arg callable that returns True
        while Kira's TTS is actively playing. When provided, the transcriber
        skips ticks during speech and for WINDOW_SECONDS afterward, preventing
        her own voice from leaking into the transcript via the loopback."""
        if not self._available:
            return False

        with self._start_lock:
            # Re-check inside the lock — another thread may have completed the
            # load + spawn between our availability check and lock acquisition.
            # This is the ONLY guard needed: if the pump thread is alive, we're
            # already running. The model-not-None guard that was here previously
            # was redundant (the lock makes it impossible for two callers to reach
            # this point simultaneously) and broke the startup pre-load path where
            # bot.py calls _load_model() directly before start() is ever called.
            if self.is_running():
                print("   [LoopbackSTT] start() ignored — instance already active")
                self._starting = False
                return True

            self._audio_agent = audio_agent
            self._is_speaking_fn = is_speaking_fn
            self._speech_last_active_ts = 0.0
            self.total_ticks_skipped_self_tts = 0

            # Lazy model load — only pay the VRAM cost when MEDIA mode is actually
            # toggled on.
            try:
                self._load_model()
            except Exception as e:
                print(f"   [ERROR] Loopback STT DISABLED — model load failed: {e}")
                self._available = False
                self._starting = False
                return False

            # Reset rolling state on each start.
            with self._lock:
                self._segments.clear()
                self._recent_normalized.clear()
                self.dialogue_summary = ""
                self._summary_needs_update = False

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._pump_loop, daemon=True, name="LoopbackSTT")
            self._thread.start()
            # Start a high-frequency watcher so the cooldown timestamp reflects the
            # actual moment her TTS ends, not the next 5s transcription tick.
            if self._is_speaking_fn is not None:
                self._speech_watcher_thread = threading.Thread(
                    target=self._speech_watcher_loop, daemon=True, name="LoopbackSTT-SpeechWatch"
                )
                self._speech_watcher_thread.start()
            print(f"   [LoopbackSTT] Started — {self.MODEL_NAME} on {self.DEVICE} "
                  f"(tick={TICK_SECONDS}s, window={WINDOW_SECONDS}s)")
            self._starting = False
            return True

    def stop(self):
        with self._start_lock:
            # Hold the lock for the entire tear-down sequence so a concurrent
            # start() can't slip through while we're mid-unload. The lock is
            # released only after _thread is None and model is None, giving
            # start() a clean slate if it runs immediately after.
            if not self.is_running():
                # Even if the pump isn't running, the WhisperModel may still be
                # resident in VRAM from a previous session — release it.
                self._unload_model()
                return
            self._stop_event.set()
            self._thread.join(timeout=3.0)
            if self._speech_watcher_thread is not None:
                self._speech_watcher_thread.join(timeout=1.0)
                self._speech_watcher_thread = None
            self._thread = None
            self._audio_agent = None
            self._is_speaking_fn = None
            # Release VRAM. faster-whisper / CTranslate2 holds CUDA buffers until
            # the model object is collected, so a simple `self.model = None` is
            # insufficient on its own — we must drop the reference AND force a gc
            # pass AND empty the torch CUDA allocator cache, or the ~1.5GB stays
            # resident forever and squeezes streaming / VTube / vision / audio.
            self._unload_model()
            print("   [LoopbackSTT] Stopped.")

    def _unload_model(self):
        """Free the WhisperModel and its CUDA buffers. Safe to call when the
        model was never loaded (no-op). Called from stop() so that toggling
        MEDIA mode off truly releases VRAM, restoring Jonny's pre-transcriber
        VRAM budget for streaming + VTube Studio + vision + audio."""
        if self.model is None:
            return
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        try:
            gc.collect()
        except Exception:
            pass
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                print(f"   [LoopbackSTT] VRAM release warning: {e}")
        print("   [LoopbackSTT] Model unloaded — VRAM released.")

    def _load_model(self):
        if self.model is not None:
            # Model already in VRAM (e.g. from the bot's startup pre-load probe).
            # Reuse it — no reload, no extra VRAM, no 20s wait.
            return
        if not DEPS_AVAILABLE:
            raise RuntimeError("faster-whisper / torch / numpy not importable")
        device = self.DEVICE
        compute_type = self.COMPUTE_TYPE
        if not torch.cuda.is_available():
            print("   [LoopbackSTT] CUDA not available — falling back to CPU int8 (will be slow).")
            device = "cpu"
            compute_type = "int8"
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"   [LoopbackSTT] Loading {self.MODEL_NAME} ({device} / {compute_type}) "
              f"from cache_dir={self.cache_dir} ...")
        t0 = time.time()
        self.model = WhisperModel(
            self.MODEL_NAME,
            device=device,
            compute_type=compute_type,
            download_root=self.cache_dir,
        )
        print(f"   [LoopbackSTT] Model ready ({time.time() - t0:.1f}s).")

    # ── Pump loop ────────────────────────────────────────────────────────

    def _speech_watcher_loop(self):
        """Polls is_speaking_fn every SPEECH_POLL_INTERVAL seconds and stamps
        _speech_last_active_ts whenever speech is active. This makes the cooldown
        start from the *real* end of TTS playback rather than from the last
        coarse 5s transcription tick. Without this, speech that ends between
        ticks leaves up to 5s of her voice in the buffer tail when transcription
        eventually resumes — which is exactly what produced the leaks of
        'wins what another chat loses their minds about it.' and 'person inside it.'
        in prior testing."""
        fn = self._is_speaking_fn
        if fn is None:
            return
        while not self._stop_event.is_set():
            try:
                if bool(fn()):
                    self._speech_last_active_ts = time.time()
            except Exception:
                pass
            if self._stop_event.wait(timeout=SPEECH_POLL_INTERVAL):
                break

    def _pump_loop(self):
        # Sleep TICK_SECONDS at top of loop so the first transcription has actual audio
        # to chew on rather than firing on an empty buffer.
        next_tick = time.time() + TICK_SECONDS
        while not self._stop_event.is_set():
            now = time.time()
            sleep_for = max(0.05, next_tick - now)
            if self._stop_event.wait(timeout=sleep_for):
                break
            next_tick = time.time() + TICK_SECONDS
            try:
                self._do_one_tick()
            except Exception as e:
                print(f"   [LoopbackSTT] Tick error: {e}")
            self._prune_old_segments()

    def _do_one_tick(self):
        if self._audio_agent is None or self.model is None or np is None:
            return
        agent = self._audio_agent
        if not agent.is_active():
            return
        # T2-C: skip transcription during MUSIC mode (Jonny's own singing/guitar).
        # The loopback buffer in MUSIC mode contains his voice via speakers — transcribing
        # it would surface his own lyrics as if they were ambient speech from a game/show.
        if agent.mode == "music":
            return

        # Self-TTS gate: skip while Kira is talking, AND for WINDOW_SECONDS
        # afterward (the trailing portion of the buffer still contains her
        # voice). Without this the transcriber happily picks up her own lines
        # and — in Stage 2 — she'd react to herself.
        #
        # The watcher thread (_speech_watcher_loop) stamps _speech_last_active_ts
        # at ~100ms cadence so the cooldown reflects the actual end of TTS,
        # not the last 5s tick boundary. We still poll here as a belt-and-braces
        # check in case TTS started between the watcher's last poll and this tick.
        if self._is_speaking_fn is not None:
            try:
                speaking_now = bool(self._is_speaking_fn())
            except Exception:
                speaking_now = False
            if speaking_now:
                self._speech_last_active_ts = time.time()
                self.total_ticks_skipped_self_tts += 1
                return
            if self._speech_last_active_ts > 0.0:
                elapsed = time.time() - self._speech_last_active_ts
                if elapsed < WINDOW_SECONDS:
                    self.total_ticks_skipped_self_tts += 1
                    return

        sample_rate = agent.sample_rate
        window_samples = int(sample_rate * WINDOW_SECONDS)

        # Snapshot the tail of the loopback buffer. The audio_agent's buffer is a
        # deque of float32 in [-1, 1] at 16kHz mono — exactly what faster-whisper
        # expects, so no resampling / format conversion needed.
        buf = agent.buffer
        if len(buf) < int(sample_rate * 2):
            return  # not enough audio to bother

        # Build numpy array from the most recent window_samples samples.
        # list() on a deque is O(n); fine for ~128k samples (8s @ 16kHz).
        snapshot = list(buf)
        if len(snapshot) > window_samples:
            snapshot = snapshot[-window_samples:]
        samples = np.asarray(snapshot, dtype=np.float32)

        # Silence gate — skip Whisper call entirely on near-silent windows.
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        if rms < SILENCE_RMS_THRESHOLD:
            return

        t0 = time.time()
        accepted_this_tick = 0
        filtered_this_tick = 0
        try:
            segments, info = self.model.transcribe(
                samples,
                language="en",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                beam_size=1,
                no_speech_threshold=NO_SPEECH_PROB_THRESHOLD,
                temperature=0.0,
            )
            # faster-whisper returns a generator; iterate to materialize.
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    filtered_this_tick += 1
                    continue
                # Per-segment confidence filters.
                no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
                if no_speech > NO_SPEECH_PROB_THRESHOLD:
                    filtered_this_tick += 1
                    continue
                if avg_logprob < MIN_AVG_LOGPROB:
                    filtered_this_tick += 1
                    continue
                if len(text) < MIN_SEGMENT_CHARS:
                    filtered_this_tick += 1
                    continue
                if _is_hallucination(text):
                    filtered_this_tick += 1
                    continue

                norm = _normalize(text)
                # Dedupe vs recent accepted segments (overlap re-emission, or
                # Whisper repeating a line across windows).
                if norm in self._recent_normalized:
                    filtered_this_tick += 1
                    continue
                # Fuzzy dedupe: skip if normalized text is a substring of any
                # recent accepted segment (handles "anything but this" vs
                # "Lazarus, anything but this").
                if any(norm in prev or prev in norm for prev in self._recent_normalized if prev):
                    filtered_this_tick += 1
                    continue
                # Short-prefix burst across segments — catches the
                # 'B-Words / B-Ward / B-Wartes' family of stutter loops. Only
                # short content-free segments are candidates; long segments
                # carrying real meaning reset the tracker.
                bare = re.sub(r"[^\w\s']", "", norm).strip()
                if bare and len(bare) <= _SHORT_BURST_MAX_LEN:
                    first_word = bare.split()[0] if bare else ""
                    if len(first_word) >= _SHORT_BURST_PREFIX_CHARS:
                        sig = first_word[:_SHORT_BURST_PREFIX_CHARS]
                        matches = sum(1 for p in self._recent_short_prefixes if p == sig)
                        if matches >= _SHORT_BURST_MIN_MATCHES:
                            self._recent_short_prefixes.append(sig)
                            filtered_this_tick += 1
                            continue
                        self._recent_short_prefixes.append(sig)
                    else:
                        self._recent_short_prefixes.append("")
                else:
                    self._recent_short_prefixes.clear()

                self._append_segment(text, norm)
                accepted_this_tick += 1
        except Exception as e:
            print(f"   [WARN] loopback_transcriber: transcription failed: {e}")
            return
        finally:
            self.last_tick_time = time.time()
            self.last_tick_latency_ms = (self.last_tick_time - t0) * 1000.0
            self.last_segment_count = accepted_this_tick
            self.total_segments_accepted += accepted_this_tick
            self.total_segments_filtered += filtered_this_tick

        if accepted_this_tick > 0:
            tail = list(self._segments)[-accepted_this_tick:]
            preview = " | ".join(s["text"] for s in tail)
            print(f"   [LoopbackSTT] +{accepted_this_tick} seg "
                  f"({filtered_this_tick} filtered, {self.last_tick_latency_ms:.0f}ms, "
                  f"rms={rms:.3f}): {preview[:200]}")
            self._write_log(tail)
        elif filtered_this_tick > 0:
            # Only print filtering activity when nothing was accepted — keeps the
            # console signal-to-noise reasonable.
            print(f"   [LoopbackSTT] tick {self.last_tick_latency_ms:.0f}ms — "
                  f"0 accepted, {filtered_this_tick} filtered (rms={rms:.3f})")

    def _append_segment(self, text: str, norm: str):
        entry = {"ts": time.time(), "text": text}
        with self._lock:
            self._segments.append(entry)
            self._recent_normalized.append(norm)
            self._summary_needs_update = True  # FIX 5: signal background summarizer

    def _prune_old_segments(self):
        cutoff = time.time() - TRANSCRIPT_MAX_AGE_SECONDS
        with self._lock:
            while self._segments and self._segments[0]["ts"] < cutoff:
                self._segments.popleft()

    # ── Output / introspection ──────────────────────────────────────────

    def get_segments(self) -> list:
        """Returns a list of {"ts","text"} segments within the rolling window
        (caller-owned copy, safe to iterate)."""
        with self._lock:
            return list(self._segments)

    def get_transcript_text(self) -> str:
        """Rendered, time-tagged transcript string, oldest first, capped at
        TRANSCRIPT_MAX_CHARS. Stage 1: visible on dashboard pane only.
        Stage 2: will become the ambient_audio_context block in llm_inference."""
        segments = self.get_segments()
        if not segments:
            return ""
        now = time.time()
        # Render oldest-first, with relative age tags.
        lines = []
        total = 0
        # Walk newest-first so we keep the most recent content when we hit the cap.
        rendered = []
        for seg in reversed(segments):
            age = int(now - seg["ts"])
            line = f"[-{age}s] {seg['text']}"
            if total + len(line) + 1 > TRANSCRIPT_MAX_CHARS:
                break
            rendered.append(line)
            total += len(line) + 1
        # Flip back to chronological order for natural reading.
        lines = list(reversed(rendered))
        return "\n".join(lines)

    def get_dialogue_summary(self) -> str:
        """Returns the condensed rolling 'story so far' summary of game dialogue.
        Persists for the full session lifetime, unlike the raw transcript which
        expires at TRANSCRIPT_MAX_AGE_SECONDS (60s).
        Updated in the background by bot.loopback_dialogue_summary_loop().
        Returns empty string until the first summarization has run."""
        return self.dialogue_summary

    def get_status_summary(self) -> str:
        """One-line status string for the dashboard."""
        if not self._available:
            return "Loopback STT: unavailable (deps missing)"
        if not self.is_running():
            return "Loopback STT: starting…" if self._starting else "Loopback STT: idle"
        seg_count = len(self._segments)
        rel = int(time.time() - self.last_tick_time) if self.last_tick_time else -1
        rel_str = f"{rel}s ago" if rel >= 0 else "never"
        return (f"Loopback STT: running · {seg_count} seg in window · "
                f"last tick {rel_str} ({self.last_tick_latency_ms:.0f}ms) · "
                f"{self.total_segments_accepted} accepted / "
                f"{self.total_segments_filtered} filtered")

    # ── Logging ──────────────────────────────────────────────────────────

    def _write_log(self, new_segments: list):
        # Raw Whisper dump goes to a throwaway location — kept for debugging but
        # never in a stream session folder and not worth archiving.
        try:
            log_dir = os.path.join(LOOPBACK_LOG_DIR, "_loopback_raw")
            os.makedirs(log_dir, exist_ok=True)
            from datetime import datetime
            now = datetime.now()
            path = os.path.join(log_dir, f"loopback_{now.strftime('%Y%m%d')}.log")
            with open(path, "a", encoding="utf-8") as fh:
                for seg in new_segments:
                    ts = datetime.fromtimestamp(seg["ts"]).strftime("%H:%M:%S")
                    fh.write(f"[{ts}] {seg['text']}\n")
        except Exception as e:
            print(f"   [LoopbackSTT] Log write failed: {e}")
