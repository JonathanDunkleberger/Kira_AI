# audio_agent.py — System / mic audio understanding for Kira
import asyncio
import base64
import time
import wave
import io
import threading
from collections import deque
from typing import Optional

try:
    import pyaudiowpatch as pyaudio
    import numpy as np
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    np = None
    PYAUDIO_AVAILABLE = False

from openai import AsyncOpenAI
from config import OPENAI_API_KEY, AUDIO_HEARTBEAT_SECONDS, AUDIO_CLIP_SECONDS, AUDIO_MODEL


AUDIO_MODE_OFF = "off"
AUDIO_MODE_MEDIA = "media"
AUDIO_MODE_MUSIC = "music"

# Names of common virtual audio devices to AVOID when auto-picking
VIRTUAL_DEVICE_KEYWORDS = (
    "cable", "vb-audio", "virtual", "voicemeeter", "ndi",
    "obs", "stream", "broadcast", "vac"
)


class AudioAgent:
    MEDIA_PROMPT = (
        "You are listening to audio from a game, anime, movie, or YouTube video that Jonny is watching/playing. "
        "Describe what you hear in 2-3 sentences. Focus on:\n"
        "- BGM (music mood, instrumentation, tempo shifts)\n"
        "- Voice acting tone (whose performance, emotional state — calm, panicked, sad, angry)\n"
        "- Notable sound effects (phone ringing, door slam, gunshot, etc.)\n"
        "Focus on SOUND DESIGN and PERFORMANCE that text alone can't capture. "
        "If the audio is mostly silent or just ambient room noise, output exactly: AUDIO_SILENT"
    )

    MUSIC_PROMPT = (
        "You are listening to Jonny playing guitar and/or singing. Describe what you hear "
        "in 2-3 sentences as a friend reacting in real time:\n"
        "- Instrumentation, genre, style\n"
        "- Chord progression vibe (major/minor, mood, notable changes)\n"
        "- If singing: pitch (in tune? flat? sharp?), emotional delivery, intelligible lyrics\n"
        "- Specific, constructive reactions Kira could ride on\n"
        "If nothing musical is happening or audio is just silence, output exactly: AUDIO_SILENT"
    )

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

        self.sample_rate = 16000
        self.channels = 1
        self.buffer: deque = deque(maxlen=int(self.sample_rate * AUDIO_CLIP_SECONDS))

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

                # Downsample to 16kHz if needed (linear interpolation)
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

    async def _describe_current_buffer(self):
        def build_wav():
            samples = np.array(list(self.buffer), dtype=np.float32)
            int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
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

            if "AUDIO_SILENT" in content.upper() or not content:
                self.consecutive_silent += 1
                # After 3 consecutive silences, decay to a clean quiet state (no recursive nesting)
                if self.consecutive_silent >= 3:
                    self.audio_summary = "(quiet)"
                # Otherwise leave the previous real summary untouched for a few more cycles
                return

            # Real audio detected — reset silence counter and store clean summary
            self.consecutive_silent = 0
            self.audio_summary = content
            self.last_capture_time = time.time()
        except Exception as e:
            print(f"   [Audio] API call failed: {e}")

    def get_audio_context(self) -> str:
        """Returns the current audio summary as a directive sense-injection.
        Framed strongly so the LLM treats this as ground truth about its own hearing."""
        if not self.is_active() or not self.audio_summary or self.audio_summary == "(quiet)":
            return ""
        rel = int(time.time() - self.last_capture_time) if self.last_capture_time else 0
        label = "playing/singing" if self.mode == AUDIO_MODE_MUSIC else "audio/music/video"
        return (
            f"[KIRA'S HEARING \u2014 ACTIVE \u2014 heard {rel}s ago]\n"
            f"You CAN HEAR right now. The audio agent is processing live audio and has just reported:\n"
            f"\"{self.audio_summary}\"\n"
            f"This is not hypothetical. This is what is playing in Jonny's environment right now. "
            f"When asked if you can hear, the answer is YES. Describe the audio naturally as part "
            f"of conversation. Do not deny having hearing \u2014 your hearing is currently active."
        )
