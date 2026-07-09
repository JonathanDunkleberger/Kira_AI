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
import struct
import subprocess
import sys
import threading
import time
import queue as _queue

import numpy as np

# Single output-volume multiplier (covers BOTH sinks — headphones + OBS cable — since one _drain
# feeds them). Jonny's live ask: ~60% of raw. Env-tunable live (restart) without a code edit.
AUDIO_VOL = float(os.getenv("POKEMON_AUDIO_VOL", "0.25"))

# ── PROCESS ISOLATION (2026-07-09): the game-audio OUTPUT path (PortAudio/WASAPI write) is the ONLY
# reproducible hard-crash in the run — a native abort() (0xC0000409) at the Viridian-parcel fanfare
# that kills the emulator. The mgba-side drain (stereo.read) is proven safe (repro_parcel_crash).
# So by DEFAULT the OutputStream now lives in a CHILD process (audio_child.py): the parent drains PCM
# and streams it over a pipe; a native abort kills only the child; the parent respawns it with backoff
# and KEEPS THE GAME RUNNING. Set POKEMON_AUDIO_ISOLATE=0 to fall back to the legacy in-process pump.
AUDIO_ISOLATE = os.getenv("POKEMON_AUDIO_ISOLATE", "1") == "1"
# Real-time wall-clock pacing (GBA ≈ 60fps). With the write in a child, the child's blocking write no
# longer paces the parent, so the parent paces itself here (independent of child health → a dead audio
# child never makes the game sprint). Matches play_live's fallback pacer. Env override for both.
AUDIO_FPS_CAP = float(os.getenv("POKEMON_FPS_CAP", "60"))
_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_child.py")

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
    """Drains core audio per frame and plays it in real time.

    DEFAULT (isolated): the PortAudio OutputStream lives in a CHILD process (audio_child.py); this
    parent only drains mgba PCM (proven crash-safe) and streams it over a pipe. A native WASAPI abort()
    kills only the child — the emulator survives and the child is respawned with backoff. The parent
    paces itself to real time (wall-clock) so a dead audio child never makes the game sprint.
    LEGACY (POKEMON_AUDIO_ISOLATE=0): the old in-process pump — OutputStream in THIS process, its
    blocking write both plays and paces. Kept for revert/debug; it is the crash-prone path.
    """

    def __init__(self, bridge, phones=None, cable=None, rate=48000,
                 buffer_size=2048, volume=None, log=print, isolate=None, paced=True):
        if sd is None:
            raise RuntimeError(f"sounddevice unavailable: {_SD_ERR}")
        global _ffi
        if _ffi is None:                    # vendor path is up by now (bridge imported first)
            from mgba._pylib import ffi as _ffi
        self.b = bridge
        self.rate = rate
        self.volume = AUDIO_VOL if volume is None else float(volume)   # ~0.6 = ~60% (Jonny's ask)
        self.log = log
        self.isolate = AUDIO_ISOLATE if isolate is None else bool(isolate)
        self.paced = bool(paced)
        self._streams = []                  # legacy in-process sinks
        self._orig_run_frame = None
        self._closed = False
        # isolated-mode state
        self._child = None                  # subprocess.Popen owning the OutputStream
        self._q = None                      # bounded parent->manager queue of PCM bytes (drop-on-full)
        self._mgr = None                    # manager thread: owns child lifecycle + pipe writes
        self._child_lock = threading.Lock()
        self._device_idx = None
        self._restarts = 0
        self._audio_dead = False            # True once restarts are exhausted (silent, game continues)
        self._pace_clock = None
        self._target_dt = 1.0 / max(1.0, AUDIO_FPS_CAP)

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
        # standalone): it forces a CONCRETE non-cable device, never the bare 'default'. Device
        # ENUMERATION is done here in the parent (cheap, not the crash site — the abort is the stream
        # WRITE); only the resolved index crosses to the child.
        idx = resolve_desktop_sink(phones, self.log)
        if idx is None:
            raise RuntimeError(
                "pkmn-audio FIREWALL: every output device is a virtual cable — refusing to route "
                "emulator audio anywhere (it would flap Kira's mouth + double her audio). Enable a real "
                "output device or set POKEMON_PHONES to your headphones.")
        self._device_idx = idx

        if self.isolate:
            self._q = _queue.Queue(maxsize=48)   # ~0.8s of frames; drop-on-full so we never block emu
            self._mgr = threading.Thread(target=self._manager_loop, name="pkmn-audio-child",
                                         daemon=True)
            self._mgr.start()
            self._orig_run_frame = self.b.core.run_frame
            self.b.core.run_frame = self._run_frame_isolated
            self.log(f"   [pkmn-audio] LIVE (ISOLATED): {rate}Hz -> child process on "
                     f"{_dev_name(idx)!r}, vol={self.volume:.2f}, "
                     f"{'wall-clock paced ~%gfps' % AUDIO_FPS_CAP if self.paced else 'unpaced'}. "
                     f"A PortAudio abort kills only the child; the game keeps running.")
            return

        # ── LEGACY in-process pump (POKEMON_AUDIO_ISOLATE=0) ──
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
        self._orig_run_frame = self.b.core.run_frame
        self.b.core.run_frame = self._run_frame_with_audio
        self.log(f"   [pkmn-audio] LIVE (LEGACY in-process): {rate}Hz, {len(self._streams)} sink(s), "
                 f"vol={self.volume:.2f}, blocking write paces the emulator (crash-prone path).")

    # ────────────────────────────── shared PCM drain ──────────────────────────────
    def _drain_pcm(self):
        """Read the mgba-side audio buffer (SAFE — no PortAudio) and return volume-scaled int16
        stereo PCM as a numpy array, or None if there's nothing this frame."""
        n = self.stereo.available
        if n <= 0:
            return None
        raw = self.stereo.read(n)                    # cffi short[2*count], interleaved L,R,L,R...
        pcm = np.frombuffer(_ffi.buffer(raw), dtype="<i2").reshape(-1, 2)
        if pcm.size == 0:
            return None
        if self.volume != 1.0:                       # scale DOWN only -> overflow-safe int16
            pcm = (pcm.astype(np.float32) * self.volume).astype("<i2")
        return pcm

    # ────────────────────────────── isolated mode ──────────────────────────────
    def _run_frame_isolated(self):
        self._orig_run_frame()
        try:
            pcm = self._drain_pcm()
            if pcm is not None and not self._audio_dead:
                try:
                    self._q.put_nowait(pcm.tobytes())    # never blocks the emulator
                except _queue.Full:
                    pass                                 # child slow/dead — drop this frame's audio
        except Exception as e:                           # never let audio crash the run (Constraint #3)
            self.log(f"   [pkmn-audio] !! drain error (audio dropped, run continues): {e}")
        if self.paced:
            self._pace()

    def _pace(self):
        """Wall-clock pace to ~AUDIO_FPS_CAP so the parent runs at real time regardless of the child."""
        now = time.perf_counter()
        if self._pace_clock is None:
            self._pace_clock = now
            return
        self._pace_clock += self._target_dt
        slack = self._pace_clock - now
        if slack > 0:
            time.sleep(slack)
        elif slack < -0.25:                              # fell >250ms behind (an LLM stall) — resync,
            self._pace_clock = time.perf_counter()       # don't sprint to "catch up"

    def _spawn_child(self):
        """Start the audio child process. Returns the Popen or None on failure."""
        try:
            child = subprocess.Popen(
                [sys.executable, _CHILD, "--device", str(self._device_idx),
                 "--rate", str(self.rate), "--channels", "2"],
                stdin=subprocess.PIPE, stdout=None, stderr=None)
            return child
        except Exception as e:
            self.log(f"   [pkmn-audio] !! could not spawn audio child: {e}")
            return None

    def _manager_loop(self):
        """Own the child's whole lifecycle: spawn -> stream PCM over its stdin -> on death (native
        abort or exit) log LOUD and respawn with backoff. The emulator never touches this thread; if
        the pipe blocks (child stalled), ONLY this thread blocks — the emulator keeps running silent."""
        MAX_RESTARTS = int(os.getenv("POKEMON_AUDIO_MAX_RESTARTS", "8"))
        backoff = 1.0
        while not self._closed:
            child = self._spawn_child()
            if child is None:
                self._restarts += 1
                if self._restarts > MAX_RESTARTS:
                    break
                time.sleep(min(backoff, 15.0)); backoff *= 2
                continue
            with self._child_lock:
                self._child = child
            self.log(f"   [pkmn-audio] audio child started (pid={child.pid}).")
            stdin = child.stdin
            spawned_at = time.perf_counter()
            try:
                while not self._closed:
                    try:
                        data = self._q.get(timeout=0.5)
                    except _queue.Empty:
                        if child.poll() is not None:     # child died during silence
                            break
                        continue
                    if data is None:                     # shutdown sentinel from close()
                        break
                    if child.poll() is not None:         # child already gone
                        break
                    stdin.write(struct.pack("<I", len(data)))
                    stdin.write(data)
                    stdin.flush()
            except (BrokenPipeError, OSError):
                pass                                     # child died mid-write — handled below
            except Exception as e:
                self.log(f"   [pkmn-audio] !! manager write error: {e}")
            # ── child is gone (or we're closing) ──
            try:
                stdin.close()
            except Exception:
                pass
            if self._closed:
                break
            code = child.poll()
            self._restarts += 1
            alive_s = time.perf_counter() - spawned_at
            self.log(f"   [pkmn-audio] !! AUDIO CHILD DIED (exit={code}, ran {alive_s:.0f}s, "
                     f"restart {self._restarts}/{MAX_RESTARTS}) — GAME CONTINUES; respawning audio.")
            if self._restarts > MAX_RESTARTS:
                self._audio_dead = True
                self.log("   [pkmn-audio] !! audio child died too many times — DISABLING audio for "
                         "this run (game keeps running, silent). Investigate the output device.")
                break
            # reset backoff if the child had survived a while (transient, not a hard-loop crash)
            backoff = 1.0 if alive_s > 30 else min(backoff * 2, 15.0)
            time.sleep(backoff)
        # drain any queued audio so a lingering put doesn't wedge on shutdown
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass

    # ────────────────────────────── legacy in-process mode ──────────────────────────────
    def _run_frame_with_audio(self):
        self._orig_run_frame()
        try:
            pcm = self._drain_pcm()
            if pcm is None:
                return
            for i, (_label, st) in enumerate(self._streams):
                try:
                    st.write(pcm)                        # first stream blocks = the real-time throttle
                except Exception:
                    if i == 0:
                        raise
        except Exception as e:                           # never let audio crash the run (Constraint #3)
            self.log(f"   [pkmn-audio] !! drain error (audio dropped, run continues): {e}")

    # ────────────────────────────── shutdown ──────────────────────────────
    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._orig_run_frame is not None:
            self.b.core.run_frame = self._orig_run_frame   # restore the bridge's frame stepping
        if self.isolate:
            try:
                if self._q is not None:
                    self._q.put_nowait(None)               # nudge the manager out of get()
            except Exception:
                pass
            with self._child_lock:
                child = self._child
            if child is not None:
                try:
                    if child.stdin:
                        child.stdin.close()
                except Exception:
                    pass
                try:
                    child.wait(timeout=2.0)
                except Exception:
                    try:
                        child.terminate()
                    except Exception:
                        pass
            if self._mgr is not None:
                self._mgr.join(timeout=2.0)
            self.log("   [pkmn-audio] stopped (isolated); audio child closed; frame-stepping restored.")
            return
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
