"""pokemon_audio.py - ISOLATED game-audio output (Stage 2). bridge.py is NOT touched.

The vendored mgba core already GENERATES audio (blip stereo buffers); bridge.py just never
drained it. This module reaches THROUGH the existing Bridge (`b.core`) and:
  1. configures the core's stereo audio buffer + resamples to a real output rate,
  2. wraps `b.core.run_frame` (instance-level, from here — the bridge.py FILE is untouched) so
     every emulated frame's samples are drained and written to one or more output devices,
  3. the blocking write provides BACKPRESSURE -> the emulator self-throttles to ~real-time 60fps
     (the run no longer sprints headless), so the music plays at the right pitch/speed.

Route targets: Jonny's HEADPHONES (so he hears it) + the VB-Audio Virtual Cable (so OBS captures
it). BONUS: once game audio is on the desktop output, Kira's existing audio classifier
(senses/audio_agent.py, desktop loopback) hears it too — zero new wiring.

Touches NO Kira/personality code and NO bridge.py. Pure additive output tap.
"""
import os
import sys

import numpy as np

# Single output-volume multiplier (covers BOTH sinks — headphones + OBS cable — since one _drain
# feeds them). Jonny's live ask: ~60% of raw. Env-tunable live (restart) without a code edit.
AUDIO_VOL = float(os.getenv("POKEMON_AUDIO_VOL", "0.25"))

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover - surfaced loudly at use
    sd = None
    _SD_ERR = e

# the SAME cffi instance the vendored mgba uses (vendor/ is on sys.path via bridge import)
try:
    from mgba._pylib import ffi as _ffi
except Exception:  # resolved again lazily if the vendor path isn't up yet at import
    _ffi = None


def list_devices():
    """Print output devices so Jonny can pick his headphones / the cable by name."""
    if sd is None:
        print(f"   [pkmn-audio] sounddevice unavailable: {_SD_ERR}"); return
    print("   [pkmn-audio] OUTPUT devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            mark = "  <- default" if i == sd.default.device[1] else ""
            print(f"      [{i:>2}] {d['name']}  ({d['max_output_channels']}ch){mark}")


def _resolve(name_or_idx):
    """Resolve a device name-substring or index to an output device index, or None."""
    if name_or_idx is None or name_or_idx == "":
        return None
    try:
        return int(name_or_idx)
    except (TypeError, ValueError):
        pass
    low = str(name_or_idx).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and low in d["name"].lower():
            return i
    return None


class AudioPump:
    """Drains core audio per frame and plays it in real time to one or more devices."""

    def __init__(self, bridge, phones=None, cable=None, rate=48000,
                 buffer_size=2048, volume=None, log=print):
        if sd is None:
            raise RuntimeError(f"sounddevice unavailable: {_SD_ERR}")
        global _ffi
        if _ffi is None:                    # vendor path is up by now (bridge imported first)
            from mgba._pylib import ffi as _ffi
        self.b = bridge
        self.rate = rate
        self.volume = AUDIO_VOL if volume is None else float(volume)   # ~0.6 = ~60% (Jonny's ask)
        self.log = log
        self._streams = []
        self._orig_run_frame = None
        self._closed = False

        # ── configure the core's audio buffer + resample to the output rate ──
        self.b.core.set_audio_buffer_size(buffer_size)
        self.stereo = self.b.core.get_audio_channels()
        self.stereo.set_rate(rate)          # blip resamples the GBA clock -> `rate` Hz stereo
        self.stereo.clear()                 # drop the boot/settle backlog so we don't blast it

        # ── resolve the route target: the DESKTOP output ONLY (Jonny's monitor, e.g. Leviathan) ──
        # DELIBERATELY NOT the virtual cable: VTube Studio opens Kira's mouth off the cable, so the
        # cable must carry ONLY her voice — game music on the cable makes her lip-sync the soundtrack.
        # Game music goes to the desktop where Jonny hears it (and OBS can capture desktop audio
        # directly); her LOOPBACK hearing of that desktop music is separately handled by Pokémon-mode
        # audio suppression. The `cable` arg is retained for signature stability but is NOT a sink.
        targets, seen = [], set()
        idx = _resolve(phones)
        if idx is not None:
            targets.append(("desktop", idx, phones)); seen.add(idx)
        if not targets:                     # fall back to the system default output
            di = sd.default.device[1] if sd.default.device else None
            dn = (sd.query_devices(di)["name"] if di is not None else "default")
            if any(v in dn.lower() for v in ("cable", "vb-audio", "vb audio")):
                self.log(f"   [pkmn-audio] !! WARNING: no --phones given and the system default "
                         f"output is a VIRTUAL CABLE ({dn!r}) — VTS would flap Kira's mouth to the "
                         f"music. Pass --phones with your DESKTOP/headphone device.")
            targets = [("default-out", None, "default")]
        for label, idx, spec in targets:
            try:
                st = sd.OutputStream(samplerate=rate, channels=2, dtype="int16",
                                     device=idx, blocksize=0, latency="low")
                st.start()
                self._streams.append((label, st))
                dev = sd.query_devices(idx)["name"] if idx is not None else "default"
                self.log(f"   [pkmn-audio] routing -> {label}: {dev!r}")
            except Exception as e:
                self.log(f"   [pkmn-audio] !! could not open {label} ({spec!r}): {e}")
        if not self._streams:
            raise RuntimeError("no audio output stream could be opened")

        # ── wrap b.core.run_frame so EVERY frame (b.run_frame AND b.press internals) drains ──
        self._orig_run_frame = self.b.core.run_frame
        self.b.core.run_frame = self._run_frame_with_audio
        self.log(f"   [pkmn-audio] LIVE: {rate}Hz, {len(self._streams)} sink(s), vol={self.volume:.2f}, "
                 f"real-time throttle ON (blocking write paces the emulator).")

    def _run_frame_with_audio(self):
        self._orig_run_frame()
        try:
            self._drain()
        except Exception as e:                       # never let audio crash the run (Constraint #3)
            self.log(f"   [pkmn-audio] !! drain error (audio dropped, run continues): {e}")

    def _drain(self):
        n = self.stereo.available
        if n <= 0:
            return
        raw = self.stereo.read(n)                    # cffi short[2*count], interleaved L,R,L,R...
        pcm = np.frombuffer(_ffi.buffer(raw), dtype="<i2").reshape(-1, 2)
        if pcm.size == 0:
            return
        if self.volume != 1.0:                       # scale DOWN only -> overflow-safe int16
            pcm = (pcm.astype(np.float32) * self.volume).astype("<i2")
        # first stream blocks (the real-time throttle); the rest are best-effort mirrors
        for i, (_label, st) in enumerate(self._streams):
            try:
                st.write(pcm)
            except Exception:
                if i == 0:
                    raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._orig_run_frame is not None:
            self.b.core.run_frame = self._orig_run_frame   # restore the bridge's frame stepping
        for _label, st in self._streams:
            try:
                st.stop(); st.close()
            except Exception:
                pass
        self.log("   [pkmn-audio] stopped; bridge frame-stepping restored.")


if __name__ == "__main__":
    # quick device listing: .venv\Scripts\python.exe pokemon_agent\pokemon_audio.py --list
    if "--list" in sys.argv:
        list_devices()
