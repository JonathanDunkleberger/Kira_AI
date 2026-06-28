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


# ── VTS FIREWALL: substrings that mark a device as a virtual cable. Game audio routed to ANY of
# these makes VTube Studio lip-sync Kira's mouth to the soundtrack (Phase-1 regression, fixed before).
# The cable must carry ONLY her TTS. This list is the single source of truth for "is this a cable?".
_CABLE_MARKERS = ("cable", "vb-audio", "vb audio", "voicemeeter", "vac ", "virtual audio",
                  "virtual cable")


def _is_cable(name):
    """True if a device name looks like a virtual cable VTS would lip-sync from."""
    low = (name or "").lower()
    return any(m in low for m in _CABLE_MARKERS)


def _dev_name(idx):
    try:
        return sd.query_devices(idx)["name"] if idx is not None else "default"
    except Exception:
        return "default"


def _first_real_output():
    """First non-cable output device index, or None if every output is a virtual cable."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and not _is_cable(d["name"]):
            return i
    return None


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


# SINGLE SOURCE OF TRUTH for the emulator's output device (every launch path: run.py, the dashboard
# launcher, any standalone). POKEMON_PHONES overrides; the default is Jonny's desktop headphones.
DESKTOP_DEVICE = os.getenv("POKEMON_PHONES", "Leviathan")


def resolve_desktop_sink(phones=None, log=print):
    """The ONE place the emulator's output device is decided. Forces a CONCRETE desktop/headphone device
    and NEVER the virtual cable, and NEVER the bare system 'default' (which on Jonny's rig IS the cable —
    that's the bug: emulator music on the cable both flaps Kira's mouth AND doubles into his headphones).
    Order: explicit `phones` arg -> POKEMON_PHONES (default 'Leviathan') -> the system default ONLY if
    it's a real non-cable device -> the first real (non-cable) output. Returns a concrete output index,
    or None only if EVERY output is a cable (caller hard-fails). Loud at each step (Constraint #3).
    Launch-path-independent: works even when no --phones is passed (e.g. python run.py)."""
    for label, spec in (("--phones", phones), ("POKEMON_PHONES", DESKTOP_DEVICE)):
        if not spec:
            continue
        idx = _resolve(spec)
        if idx is None:
            log(f"   [pkmn-audio] {label}={spec!r} not found among outputs — trying next")
            continue
        if _is_cable(_dev_name(idx)):
            log(f"   [pkmn-audio] !! REFUSED {label}={spec!r}: that's a VIRTUAL CABLE "
                f"({_dev_name(idx)!r}) — emulator audio must NEVER touch the cable. Trying next.")
            continue
        return idx
    # named desktop device unavailable -> system default, but ONLY if it's a real (non-cable) device
    di = sd.default.device[1] if sd.default.device else None
    if di is not None and not _is_cable(_dev_name(di)):
        log(f"   [pkmn-audio] using system default output {_dev_name(di)!r} (real, non-cable)")
        return di
    if di is not None:
        log(f"   [pkmn-audio] !! system default output is a VIRTUAL CABLE ({_dev_name(di)!r}) — "
            f"REFUSING it (it would flap her mouth + double the audio); picking a real output instead.")
    real = _first_real_output()
    if real is not None:
        log(f"   [pkmn-audio] routing emulator audio to first REAL output {_dev_name(real)!r}")
    return real


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

        # ── route target: the DESKTOP output ONLY (Jonny's monitor/headphones, e.g. Leviathan) ──
        # The virtual cable carries Kira's TTS ONLY (VTS lip-syncs her mouth off the cable). Emulator
        # music/SFX must go to the DESKTOP device and NEVER the cable — emulator audio on the cable both
        # flaps her mouth to the soundtrack AND doubles into Jonny's headphones (the bug). resolve_
        # desktop_sink is the SINGLE SOURCE OF TRUTH used on EVERY launch path (run.py / dashboard /
        # standalone): it forces a CONCRETE non-cable device, never the bare 'default'. Exactly ONE sink
        # is opened — no duplicate output. (The `cable` arg is retained for signature stability, NOT a sink.)
        idx = resolve_desktop_sink(phones, self.log)
        if idx is None:
            raise RuntimeError(
                "pkmn-audio FIREWALL: every output device is a virtual cable — refusing to route "
                "emulator audio anywhere (it would flap Kira's mouth + double her audio). Enable a real "
                "output device or set POKEMON_PHONES to your headphones.")
        try:
            st = sd.OutputStream(samplerate=rate, channels=2, dtype="int16",
                                 device=idx, blocksize=0, latency="low")
            st.start()
            self._streams.append(("desktop", st))
            self.log(f"   [pkmn-audio] routing -> desktop ONLY: {_dev_name(idx)!r} "
                     f"(exactly one sink; the cable carries ONLY Kira's TTS)")
        except Exception as e:
            self.log(f"   [pkmn-audio] !! could not open desktop sink {_dev_name(idx)!r}: {e}")
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
