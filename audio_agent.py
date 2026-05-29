# audio_agent.py — System / mic audio understanding for Kira
import asyncio
import base64
import os
import time
import wave
import io
import threading
from collections import deque
from datetime import datetime
from typing import Optional

AUDIO_LOG_DIR = "logs"

try:
    import pyaudiowpatch as pyaudio
    import numpy as np
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    np = None
    PYAUDIO_AVAILABLE = False

from openai import AsyncOpenAI, NotFoundError as OpenAINotFoundError
from config import OPENAI_API_KEY, AUDIO_HEARTBEAT_SECONDS, AUDIO_CLIP_SECONDS, AUDIO_MODEL, AUDD_API_TOKEN


AUDIO_MODE_OFF = "off"
AUDIO_MODE_MEDIA = "media"
AUDIO_MODE_MUSIC = "music"

# Names of common virtual audio devices to AVOID when auto-picking
VIRTUAL_DEVICE_KEYWORDS = (
    "cable", "vb-audio", "virtual", "voicemeeter", "ndi",
    "obs", "stream", "broadcast", "vac"
)


class AudioAgent:
    # Prompts are written as TRANSCRIPTION-STYLE directives, not as a chat partner asking
    # the model a question. gpt-audio-mini is heavily RLHF'd toward conversational
    # assistance, so any phrasing like "please describe" or "can you tell me" triggers it
    # to reply CONVERSATIONALLY ("Sure! Please provide more context...", "Understood, I'll
    # adhere to the instructions...", "I cannot hear audio directly..."). The fix is to
    # frame the task as a fixed-format report with explicit FORBIDDEN OUTPUTS and an
    # exact silent token. The meta-reply filter in _describe_current_buffer is the
    # belt-and-suspenders second line of defense.
    #
    # IMPORTANT: Do NOT use phrases like "the audio IS attached" or "the attached audio" —
    # those phrases tell the model to expect a file, which triggers its conversational
    # upload-request mode. Instead frame everything as a live-feed sensor reading task.
    MEDIA_PROMPT = (
        "You are a real-time audio sensor. A live audio frame is embedded in this message.\n"
        "OUTPUT: 2-3 declarative sentences describing ONLY what is audible RIGHT NOW. Nothing else.\n"
        "\n"
        "DESCRIBE:\n"
        "- BGM (mood, instrumentation, tempo)\n"
        "- Voice tone (emotional state — calm, panicked, sad, angry — never invent dialogue)\n"
        "- Notable sound effects (phone ringing, door slam, gunshot, footsteps, etc.)\n"
        "\n"
        "ABSOLUTE RULES — violation means immediate rejection:\n"
        "- Write only declarative observations. Example: 'Soft piano, slow tempo, melancholic mood.'\n"
        "- NEVER use first person. Not 'I hear', not 'I notice', not 'I can', not 'I cannot'.\n"
        "- NEVER address the user. No 'sure', 'understood', 'I'll', 'here is', 'as requested'.\n"
        "- NEVER ask for anything. No questions, no 'please', no 'could you', no requests.\n"
        "- NEVER say you cannot process, analyze, listen to, or access audio.\n"
        "- NEVER apologize, explain yourself, or describe your own limitations.\n"
        "- NEVER request files, uploads, attachments, clips, or links.\n"
        "- NEVER invent lyrics, dialogue, or specific instruments you cannot clearly identify.\n"
        "- Do NOT write meta-commentary, retrospective notes, or self-reflection.\n"
        "\n"
        "IF THE FRAME IS SILENT, AMBIENT-ONLY, OR INAUDIBLE: output exactly: AUDIO_SILENT\n"
        "IF UNCERTAIN: prefix with 'UNCERTAIN:' and describe only general mood, presence/absence of voice, and tempo."
    )

    MUSIC_PROMPT = (
        "You are a real-time audio sensor. A live audio frame from a live music performance is embedded in this message.\n"
        "OUTPUT: 2-3 declarative sentences describing ONLY what is audible RIGHT NOW. Nothing else.\n"
        "\n"
        "DESCRIBE:\n"
        "- Instrumentation, genre, style\n"
        "- Chord progression vibe (major/minor mood, notable changes)\n"
        "- If singing is clearly audible: pitch accuracy, emotional delivery, intelligible lyrics only\n"
        "\n"
        "ABSOLUTE RULES — violation means immediate rejection:\n"
        "- Write only declarative observations. Example: 'Acoustic guitar, slow fingerpicking, minor key.'\n"
        "- NEVER use first person. Not 'I hear', not 'I notice', not 'I can', not 'I cannot'.\n"
        "- NEVER address the user. No 'sure', 'understood', 'I'll', 'here is', 'as requested'.\n"
        "- NEVER ask for anything. No questions, no 'please', no 'could you', no requests.\n"
        "- NEVER say you cannot process, analyze, listen to, or access audio.\n"
        "- NEVER apologize, explain yourself, or describe your own limitations.\n"
        "- NEVER request files, uploads, attachments, clips, or links.\n"
        "- NEVER invent lyrics or pitch judgements you cannot clearly identify.\n"
        "- Do NOT write meta-commentary, retrospective notes, or self-reflection.\n"
        "\n"
        "IF AUDIO IS SILENT OR NOTHING MUSICAL IS HAPPENING: output exactly: AUDIO_SILENT\n"
        "IF UNCERTAIN: prefix with 'UNCERTAIN:' and describe only what is clearly audible."
    )

    # Substring fingerprints of model meta-replies that should be treated as silence.
    # All lowercased. Match is substring-anywhere because the model often opens with one
    # of these phrases and then rambles for another sentence of self-explanation.
    # Expanded 2026-05 after Steins;Gate session leaked variants like "please upload the
    # audio file", "please attach", "please provide the audio clip", "I currently can't
    # process", "Apologies, I currently cannot listen", "Got it. Please upload",
    # "Sure! Please attach", "Retrospective note".
    _META_REPLY_FINGERPRINTS = (
        "i cannot hear", "i can't hear", "i cannot directly", "i can't directly",
        "i cannot listen", "i can't listen", "cannot listen to", "can't listen to",
        "i cannot process", "i can't process", "i currently can't process", "i currently cannot process",
        "i cannot analyze", "i can't analyze", "i cannot analyse", "i can't analyse",
        "i'm unable to", "i am unable to", "unable to hear", "unable to process",
        "unable to listen", "unable to analyze", "unable to analyse",
        "please provide", "could you provide", "can you provide",
        "please upload", "could you please upload", "please share the audio",
        "please attach", "could you attach", "could you please attach",
        "upload the file", "upload the audio", "share the audio", "attach the audio",
        "attach the file", "attach the clip", "attach a clip", "attach an audio",
        "provide the audio", "provide the clip", "provide an audio", "provide a clip",
        "provide a link", "proceed with the description", "audio description report",
        "i'll adhere", "i will adhere", "i'll follow", "i will follow",
        "understood!", "understood,", "understood.", "got it!", "got it,", "got it.",
        "sure!", "sure,", "sure.", "of course!", "of course,",
        "apologies", "i apologize", "i'm sorry", "i am sorry", "unfortunately, i",
        "i'm ready to", "i am ready to", "ready to focus",
        "as an ai", "i don't have the ability", "i do not have the ability",
        "i lack the ability", "i'm not able to", "i am not able to",
        "more context", "specific scene", "describe the audio",
        "i'm here to help", "happy to help",
        "retrospective note", "meta note", "meta-note", "side note:", "note to self",
        "as a language model", "as a text-based",
        # Expanded 2026-05-24 after long Steins;Gate session: variants still leaking.
        "upcoming audio analysis", "please wait for the audio", "waiting to receive",
        "once received", "once the audio", "once you upload", "once you provide",
        "once it is uploaded", "once available", "once provided", "once visible",
        "i don't have the capability", "i do not have the capability",
        "don't have the capability", "do not have the capability",
        "currently unable", "currently can't", "currently cannot",
        "can't directly listen", "cannot directly listen",
        "the attached audio", "the attachment", "check the attachment",
        "let me know", "feel free to", "go ahead and",
        "i can help", "i can analyze", "i'll analyze", "i will analyze",
        "i'll provide", "i will provide", "i'll generate", "i will generate",
        "i'll deliver", "i will deliver", "i'll proceed", "i will proceed",
        "based on the criteria", "based on your instructions", "as instructed",
        "as requested", "as described above", "according to the criteria",
        "format described", "format above", "structure above",
    )

    @classmethod
    def _is_meta_reply(cls, text: str) -> bool:
        """Return True if the model's output is conversational meta-chatter rather than
        an actual description of the audio. Matched outputs are treated identically to
        AUDIO_SILENT so they don't poison downstream context with assistant boilerplate.

        Two layers:
          1. Substring fingerprint match against known meta-phrases.
          2. Positive-shape sanity check — a real description is short, declarative,
             and never asks questions or addresses the user. Any of these disqualify:
               - contains a '?' (real descriptions don't ask questions)
               - contains the word 'please' (always a request, never a description)
               - starts with first-person 'I ' / "I'm" / "I'll" (assistant mode)
               - contains 'upload', 'attach', 'audio file', 'audio clip' as requests
        """
        if not text:
            return False
        low = text.lower().strip()
        if any(fp in low for fp in cls._META_REPLY_FINGERPRINTS):
            return True
        # Positive-shape checks: a valid mood description never contains these.
        if "?" in low or "please" in low:
            return True
        if low.startswith(("i ", "i'm ", "i'll ", "i am ", "i will ", "i can ", "i cannot ", "i can't ")):
            return True
        # Standalone request keywords that only appear in assistant boilerplate.
        for kw in ("upload", "attach", "audio file", "audio clip", "the clip you", "the file you"):
            if kw in low:
                return True
        return False

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        if OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        else:
            print("   [Audio] Warning: OPENAI_API_KEY not set — audio agent inactive.")

        self.mode: str = AUDIO_MODE_OFF
        self.audio_summary: str = ""
        self.last_capture_time: float = 0
        self.consecutive_silent: int = 0
        # Track consecutive model_not_found responses. We require several in a row before
        # permanently disabling the agent — a single 404 can be a stale model string, a
        # momentary org/project routing glitch, or a transient gateway error. See the
        # 2026-05 incident where gpt-4o-audio-preview was renamed to gpt-audio and a
        # single 404 had the agent shut hearing off for the whole session.
        self.consecutive_not_found: int = 0
        self.NOT_FOUND_DISABLE_THRESHOLD: int = 3

        self.sample_rate = 16000
        self.channels = 1
        self.buffer: deque = deque(maxlen=int(self.sample_rate * AUDIO_CLIP_SECONDS))
        # Parallel high-fidelity buffer at the native loopback rate (typically 48kHz mono).
        # The 16kHz `self.buffer` above is downsampled for gpt-audio-mini (speech-grade),
        # which is far too low for AudD fingerprinting — hence the duplicate capture.
        # Sized for up to 15s at 48kHz worst case (~5.7MB of float32 — negligible).
        self.hifi_buffer: deque = deque(maxlen=48000 * 15)
        self._hifi_sample_rate: int = self.sample_rate  # updated when capture starts

        self._pa = None
        self._stream = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._capture_sample_rate = self.sample_rate
        self._capture_channels = 1
        self.preferred_loopback_name: Optional[str] = None  # set by dashboard to override auto-pick

    def is_active(self) -> bool:
        return self.mode != AUDIO_MODE_OFF and self._stream is not None

    def set_mode(self, mode: str):
        if mode not in (AUDIO_MODE_OFF, AUDIO_MODE_MEDIA, AUDIO_MODE_MUSIC):
            print(f"   [Audio] Unknown mode: {mode}")
            return

        previous = self.mode

        # Always stop cleanly first if a capture is active
        if previous != AUDIO_MODE_OFF and self._stream is not None:
            self._stop_capture()
            # Give PyAudio time to release the audio device at the C layer.
            # Without this, opening a new stream on the same/different device
            # too quickly crashes the process.
            time.sleep(0.5)

        self.mode = mode

        if mode == AUDIO_MODE_OFF:
            self.audio_summary = ""
            self.consecutive_silent = 0
            print("   [Audio] Mode OFF")
            return

        # Reset the silence counter and audio summary on mode start/change
        self.consecutive_silent = 0
        self.audio_summary = ""

        self._start_capture()

    def _list_loopback_devices(self) -> list:
        """Returns all WASAPI loopback devices with their info."""
        devices = []
        if not self._pa:
            return devices
        try:
            for device in self._pa.get_loopback_device_info_generator():
                devices.append(device)
        except Exception as e:
            print(f"   [Audio] Loopback enumeration failed: {e}")
        return devices

    def _is_virtual_device(self, device: dict) -> bool:
        name = device.get("name", "").lower()
        return any(kw in name for kw in VIRTUAL_DEVICE_KEYWORDS)

    def _find_loopback_device(self, preferred_name: Optional[str] = None) -> Optional[dict]:
        """Finds the best WASAPI loopback device. Priority:
        1. preferred_name match (user explicitly chose)
        2. Default WASAPI loopback IF it's not a virtual device
        3. First non-virtual loopback in enumeration
        4. Any loopback (last resort)
        """
        devices = self._list_loopback_devices()
        if not devices:
            print("   [Audio] No WASAPI loopback devices found at all.")
            return None

        # Priority 1: explicit preferred name match
        if preferred_name:
            for d in devices:
                if preferred_name.lower() in d.get("name", "").lower():
                    return d
            print(f"   [Audio] Preferred device '{preferred_name}' not found, falling through to auto-pick.")

        # Priority 2: default WASAPI loopback if not virtual
        try:
            default = self._pa.get_default_wasapi_loopback()
            if default and not self._is_virtual_device(default):
                return default
            elif default:
                print(f"   [Audio] Default loopback '{default.get('name')}' is virtual — looking for a real device.")
        except Exception:
            pass

        # Priority 3: first non-virtual loopback
        for d in devices:
            if not self._is_virtual_device(d):
                print(f"   [Audio] Auto-selected non-virtual loopback: {d.get('name')}")
                return d

        # Priority 4: anything (last resort, warn user)
        print(f"   [Audio] Only virtual loopback devices found. Using: {devices[0].get('name')}")
        print("           Configure your real speakers in Windows Sound Settings or set device manually in dashboard.")
        return devices[0]

    def list_available_loopback_devices(self) -> list:
        """Returns list of (name, is_virtual) tuples for dashboard display.
        Initializes PyAudio temporarily if not running."""
        devices_list = []
        temp_pa = False
        pa = self._pa
        if not pa:
            if not PYAUDIO_AVAILABLE:
                return []
            try:
                pa = pyaudio.PyAudio()
                temp_pa = True
            except Exception:
                return []
        try:
            for device in pa.get_loopback_device_info_generator():
                name = device.get("name", "Unknown")
                is_virtual = any(kw in name.lower() for kw in VIRTUAL_DEVICE_KEYWORDS)
                devices_list.append((name, is_virtual))
        except Exception:
            pass
        finally:
            if temp_pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
        return devices_list

    def _start_capture(self):
        if not PYAUDIO_AVAILABLE:
            print("   [Audio] pyaudiowpatch not installed — run: pip install PyAudioWPatch")
            self.mode = AUDIO_MODE_OFF
            return
        if not self.client:
            print("   [Audio] No OpenAI client — capture disabled")
            self.mode = AUDIO_MODE_OFF
            return

        try:
            self._pa = pyaudio.PyAudio()

            if self.mode == AUDIO_MODE_MEDIA:
                # MEDIA mode requires WASAPI loopback — abort rather than fall back to mic
                device = self._find_loopback_device(preferred_name=self.preferred_loopback_name)
                if not device:
                    print("   [Audio] WASAPI loopback unavailable — MEDIA mode requires this. Switching to OFF.")
                    print("           Tip: try Music mode (mic capture) for now, or check Windows audio settings.")
                    self.mode = AUDIO_MODE_OFF
                    self._pa.terminate()
                    self._pa = None
                    return

                device_index = device["index"]
                device_rate = int(device["defaultSampleRate"])
                device_channels = int(device["maxInputChannels"])
                print(f"   [Audio] Using WASAPI loopback device: {device['name']} "
                      f"({device_rate}Hz, {device_channels}ch)")

                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=device_channels,
                    rate=device_rate,
                    frames_per_buffer=1024,
                    input=True,
                    input_device_index=device_index,
                    stream_callback=None,
                )
                self._capture_sample_rate = device_rate
                self._capture_channels = device_channels
                self._hifi_sample_rate = device_rate
                self.hifi_buffer.clear()

            else:
                # MUSIC mode — default mic input
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    frames_per_buffer=1024,
                    input=True,
                )
                self._capture_sample_rate = self.sample_rate
                self._capture_channels = 1
                self._hifi_sample_rate = self.sample_rate
                self.hifi_buffer.clear()
                print(f"   [Audio] Mic capture started ({self.sample_rate}Hz, mono)")

            # Start background capture thread
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            print(f"   [Audio] Mode {self.mode.upper()} — capture started")

        except Exception as e:
            print(f"   [Audio] Failed to start capture: {e}")
            self.mode = AUDIO_MODE_OFF
            if self._pa:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
            self._stream = None

    def _capture_loop(self):
        """Reads from the PyAudio stream in a background thread, downmixes to mono 16kHz,
        and appends samples to the rolling buffer."""
        try:
            while not self._stop_event.is_set() and self._stream:
                try:
                    raw = self._stream.read(1024, exception_on_overflow=False)
                except Exception:
                    break
                samples = np.frombuffer(raw, dtype=np.int16)

                # Downmix stereo/multichannel to mono
                if self._capture_channels > 1:
                    samples = samples.reshape(-1, self._capture_channels).mean(axis=1).astype(np.int16)

                # Convert to float32 normalized
                float_samples = samples.astype(np.float32) / 32768.0

                # Append the native-rate, mono float samples to the hi-fi buffer FIRST,
                # before the 16kHz downsample. AudD fingerprints music far better at 48kHz
                # than at 16kHz (Shazam-style algos need spectral content up to ~5kHz+).
                self.hifi_buffer.extend(float_samples.tolist())

                # Downsample to 16kHz if needed (linear interpolation) — for gpt-audio-mini
                if self._capture_sample_rate != self.sample_rate:
                    ratio = self.sample_rate / self._capture_sample_rate
                    new_len = int(len(float_samples) * ratio)
                    if new_len > 0:
                        idx = np.linspace(0, len(float_samples) - 1, new_len).astype(np.int32)
                        float_samples = float_samples[idx]

                self.buffer.extend(float_samples.tolist())
        except Exception as e:
            print(f"   [Audio] Capture loop error: {e}")

    def _stop_capture(self):
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self.buffer.clear()
        self.hifi_buffer.clear()

    async def heartbeat_loop(self):
        print("   [System] Audio Agent heartbeat ready (idle until mode != OFF).")
        while True:
            await asyncio.sleep(AUDIO_HEARTBEAT_SECONDS)
            if not self.is_active() or not self.client:
                continue
            if len(self.buffer) < self.sample_rate * 3:
                continue
            try:
                await self._describe_current_buffer()
            except Exception as e:
                print(f"   [Audio] Heartbeat error: {e}")

    def _log_summary(self, summary: str) -> None:
        """Print the full audio summary to console and append to a rolling daily log file.

        The dashboard panel truncates summaries for display; this gives the developer
        a full, reviewable record of what the audio agent actually heard each heartbeat.
        """
        now = datetime.now()
        mode_tag = self.mode.upper() if self.mode else "OFF"
        ts = now.strftime("%H:%M:%S")
        print(f"   [Audio] ({mode_tag}) {ts} summary: {summary}")
        try:
            os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
            log_path = os.path.join(AUDIO_LOG_DIR, f"audio_{now.strftime('%Y%m%d')}.log")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] ({mode_tag}) {summary}\n")
        except Exception as e:
            print(f"   [Audio] Failed to write audio log file: {e}")

    # RMS threshold (on float32 samples in [-1, 1]) below which the buffer is
    # considered silent and the audio model is NOT called. Eliminates cold-start
    # meta-garbage (model invents "please upload the audio" replies when handed an
    # essentially empty WAV) AND saves API spend on genuine silence. Real music or
    # voice at any reasonable monitoring volume sits well above this (>0.01); pure
    # loopback silence from an idle output device is typically <0.0005.
    # Raised 2026-05-24 from 0.005 → 0.010 — marginal-ambient buffers (0.005-0.010)
    # were the main source of meta-replies; the model gets confused by near-silent
    # input and falls into assistant-mode rather than describing it.
    SILENCE_RMS_THRESHOLD: float = 0.010

    async def _describe_current_buffer(self):
        # Silence-gate: don't waste an API call (or risk a meta-reply hallucination)
        # on a buffer that contains no real signal. Computed on the same samples we
        # would have shipped, before building the WAV. See SILENCE_RMS_THRESHOLD.
        samples_snapshot = np.array(list(self.buffer), dtype=np.float32)
        if samples_snapshot.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(samples_snapshot * samples_snapshot)))
        if rms < self.SILENCE_RMS_THRESHOLD:
            print(f"   [Audio] Buffer silent (RMS={rms:.5f} < {self.SILENCE_RMS_THRESHOLD}) — skipping model call")
            self.consecutive_silent += 1
            if self.consecutive_silent >= 3:
                self.audio_summary = "(quiet)"
            return

        def build_wav():
            int16 = (samples_snapshot * 32767).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(int16.tobytes())
            return buf.getvalue()

        wav_bytes = await asyncio.to_thread(build_wav)
        b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        prompt = self.MEDIA_PROMPT if self.mode == AUDIO_MODE_MEDIA else self.MUSIC_PROMPT

        try:
            response = await self.client.chat.completions.create(
                model=AUDIO_MODEL,
                modalities=["text"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}},
                        ],
                    },
                ],
                max_tokens=180,
            )
            content = (response.choices[0].message.content or "").strip()

            # Treat AUDIO_SILENT, empty, AND model meta-chatter the same way — none of
            # them are real descriptions and all of them would pollute Kira's context
            # if forwarded. The meta-reply filter is essential because gpt-audio-mini
            # frequently outputs assistant boilerplate ("Sure! Please provide...",
            # "Understood, I'll adhere to the instructions...") instead of describing
            # the clip. See _META_REPLY_FINGERPRINTS for the pattern list.
            is_meta = self._is_meta_reply(content)
            if "AUDIO_SILENT" in content.upper() or not content or is_meta:
                if is_meta:
                    print(f"   [Audio] Suppressed meta-reply: {content[:120]!r}")
                self.consecutive_silent += 1
                # After 3 consecutive silences, decay to a clean quiet state (no recursive nesting)
                if self.consecutive_silent >= 3:
                    self.audio_summary = "(quiet)"
                # Otherwise leave the previous real summary untouched for a few more cycles
                return

            # Real audio detected — reset silence counter and store clean summary
            self.consecutive_silent = 0
            self.consecutive_not_found = 0  # any success clears the 404 streak
            self.audio_summary = content
            self.last_capture_time = time.time()
            self._log_summary(content)
        except OpenAINotFoundError as e:
            # Log the exact response body verbatim — had we done this the first time,
            # the 2026-05 "model_not_found" would have been obvious instead of being
            # mistaken for a permanent deprecation of the entire audio capability.
            body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
            self.consecutive_not_found += 1
            print(f"   [Audio] model_not_found for '{AUDIO_MODEL}' "
                  f"({self.consecutive_not_found}/{self.NOT_FOUND_DISABLE_THRESHOLD}): {body}")
            if self.consecutive_not_found >= self.NOT_FOUND_DISABLE_THRESHOLD:
                print(f"   [Audio] Disabling audio agent for this session after "
                      f"{self.consecutive_not_found} consecutive 404s. Check AUDIO_MODEL in .env "
                      f"(the rolling aliases are 'gpt-audio' and 'gpt-audio-mini').")
                self.mode = AUDIO_MODE_OFF
            return
        except Exception as e:
            # Transient errors (429 rate limit, 529 overload, timeouts, network) land here.
            # Heartbeat will retry on the next tick — do NOT permanently disable.
            print(f"   [Audio] API call failed (transient, will retry next heartbeat): {e}")

    async def identify_song(self) -> Optional[dict]:
        """Fingerprint the current audio buffer against AudD's catalog.

        Returns a dict like {"title": str, "artist": str, "album": str|None,
        "release_date": str|None} on a confirmed match, or None on no-match /
        config error / API failure. Only invoke on explicit user intent —
        AudD is a paid API and the heartbeat must never call this.
        """
        if not AUDD_API_TOKEN:
            print("   [SongID] AUDD_API_TOKEN not set in .env — cannot fingerprint.")
            return None
        if not self.is_active() or not self.hifi_buffer or np is None:
            print("   [SongID] Audio agent inactive or hi-fi buffer empty — nothing to identify.")
            return None
        hifi_rate = self._hifi_sample_rate or self.sample_rate
        if len(self.hifi_buffer) < hifi_rate * 3:
            print(f"   [SongID] Hi-fi buffer too short (< 3s @ {hifi_rate}Hz) — need more audio to fingerprint.")
            return None

        # Build an ~8s WAV clip from the LOUDEST 8s window inside the hi-fi buffer
        # (AudD recommends 2-12s; 8s is a sweet spot for fingerprint reliability).
        # Why "loudest window" instead of "tail"? Users naturally pause/lower the music
        # to ASK Kira to identify it, and that pause lands in the buffer tail — so a
        # tail-grab returns mostly silence with a sliver of pre-pause music, which AudD
        # cannot fingerprint. Scanning the whole 15s buffer for the highest-RMS 8s
        # window finds the real music regardless of whether it's at the start, middle,
        # or end. 1-second stride keeps the scan cheap (~7 RMS evals over a 15s buffer).
        clip_seconds = 8
        clip_samples = hifi_rate * clip_seconds
        full_buffer = np.array(self.hifi_buffer, dtype=np.float32)
        total_samples = full_buffer.size

        if total_samples <= clip_samples:
            # Buffer not yet larger than the clip window — use everything we have.
            start_idx = 0
            samples_list = full_buffer.tolist()
            window_rms = float(np.sqrt(np.mean(full_buffer * full_buffer))) if total_samples else 0.0
        else:
            stride = hifi_rate  # 1-second stride
            best_rms = -1.0
            best_start = 0
            scan_start = 0
            while scan_start + clip_samples <= total_samples:
                window = full_buffer[scan_start:scan_start + clip_samples]
                w_rms = float(np.sqrt(np.mean(window * window)))
                if w_rms > best_rms:
                    best_rms = w_rms
                    best_start = scan_start
                scan_start += stride
            start_idx = best_start
            samples_list = full_buffer[best_start:best_start + clip_samples].tolist()
            window_rms = best_rms
            print(f"   [SongID] selected loudest window {start_idx/hifi_rate:.1f}-"
                  f"{(start_idx+clip_samples)/hifi_rate:.1f}s "
                  f"(window RMS={window_rms:.5f}) from {total_samples/hifi_rate:.1f}s buffer")

        def build_wav_bytes() -> bytes:
            samples = np.array(samples_list, dtype=np.float32)
            int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(hifi_rate)
                wf.writeframes(int16.tobytes())
            return buf.getvalue()

        def post_to_audd(wav_bytes: bytes):
            import requests
            return requests.post(
                "https://api.audd.io/",
                data={"api_token": AUDD_API_TOKEN, "return": "apple_music,spotify"},
                files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                timeout=20,
            )

        try:
            wav_bytes = await asyncio.to_thread(build_wav_bytes)
            duration_s = len(samples_list) / hifi_rate
            # Diagnostic: RMS of the exact samples sent. If near zero, the hi-fi capture
            # path is broken even though the 16kHz path was happily producing summaries.
            samples_arr = np.array(samples_list, dtype=np.float32)
            peak = float(np.max(np.abs(samples_arr))) if samples_arr.size else 0.0
            rms = float(np.sqrt(np.mean(samples_arr * samples_arr))) if samples_arr.size else 0.0
            print(f"   [SongID] hifi buffer RMS={rms:.5f} peak={peak:.5f}, "
                  f"{len(samples_list)} samples @ {hifi_rate}Hz")
            print(f"   [SongID] encoded {duration_s:.1f}s WAV @ {hifi_rate}Hz mono → AudD ({len(wav_bytes)} bytes)")
            # Diagnostic dump: write the exact WAV we send to AudD so it can be played
            # back and verified by ear. Lets us tell silence/wrong-channel/noise apart
            # from "AudD just didn't match real music".
            try:
                os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
                debug_path = os.path.join(
                    AUDIO_LOG_DIR,
                    f"songid_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                )
                with open(debug_path, "wb") as fh:
                    fh.write(wav_bytes)
                print(f"   [SongID] dumped debug WAV → {debug_path}")
            except Exception as e:
                print(f"   [SongID] Failed to write debug WAV: {e}")
            resp = await asyncio.to_thread(post_to_audd, wav_bytes)
            if resp.status_code != 200:
                print(f"   [SongID] AudD HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            payload = resp.json()
        except Exception as e:
            print(f"   [SongID] AudD request failed: {e}")
            return None

        status = payload.get("status")
        result = payload.get("result")
        if status != "success":
            err = payload.get("error", {})
            print(f"   [SongID] AudD error response: {err}")
            return None
        if not result:
            print("   [SongID] AudD query \u2192 no match (not in catalog, live cut, or too obscure).")
            return None

        info = {
            "title": result.get("title"),
            "artist": result.get("artist"),
            "album": result.get("album"),
            "release_date": result.get("release_date"),
        }
        print(f"   [SongID] AudD query \u2192 \"{info['title']}\" by {info['artist']}"
              + (f" ({info['album']})" if info["album"] else ""))
        # Append to the same rolling audio log so the developer can audit fingerprint calls
        try:
            os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
            now = datetime.now()
            log_path = os.path.join(AUDIO_LOG_DIR, f"audio_{now.strftime('%Y%m%d')}.log")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] (SONGID) {info['title']} \u2014 {info['artist']}"
                        + (f" / {info['album']}" if info["album"] else "") + "\n")
        except Exception as e:
            print(f"   [SongID] Failed to write log entry: {e}")
        return info

    def get_audio_context(self) -> str:
        """Returns the current audio summary as a directive sense-injection.
        Framed as RAW PERCEPTION she should react to in character, NOT as a script
        to recite. The audio model's clinical wording is reference data for her ears,
        not the words that should come out of her mouth."""
        if not self.is_active() or not self.audio_summary or self.audio_summary == "(quiet)":
            return ""
        rel = int(time.time() - self.last_capture_time) if self.last_capture_time else 0
        label = "playing/singing" if self.mode == AUDIO_MODE_MUSIC else "audio/music/video"
        em = "\u2014"
        return (
            f"[KIRA'S HEARING {em} ACTIVE {em} heard {rel}s ago]\n"
            f"You CAN HEAR right now. This is your raw perception of what is playing in "
            f"Jonny's environment ({label}):\n"
            f"\"{self.audio_summary}\"\n"
            f"This is sense data {em} information your ears are giving you. It is NOT a script. "
            f"Do NOT recite or paraphrase this description back. React to it as YOU would, "
            f"in your own voice: how it makes you feel, what it reminds you of, your take, a "
            f"quip, a roast, a vibe check. You are a person hearing music or a scene, not an "
            f"audio engineer labeling instrumentation. If asked what you hear, answer naturally "
            f"from this perception {em} don't read it off like a chart. When asked if you can "
            f"hear, the answer is YES {em} your hearing is currently active."
        )
