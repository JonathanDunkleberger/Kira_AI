"""audio_child.py — the ISOLATED audio-OUTPUT process (the PortAudio blast shield).

WHY THIS EXISTS: the game-audio output path (sounddevice → PortAudio → WASAPI) is the ONLY
place the Pokémon run ever hard-crashes — a native abort() (0xC0000409) inside the WASAPI
write layer (the reproducible Viridian-parcel fanfare SIGSEGV). The mgba-side audio *read*
(`stereo.read`) is proven SAFE (repro_parcel_crash.py: REPRO_AUDIO drains every frame, never
crashes). So the fix is a process boundary: the parent (the emulator) drains PCM and ships it
here over stdin; THIS process owns the OutputStream and does the dangerous `st.write`. If
PortAudio abort()s, it kills ONLY this child — the emulator + main run keep going, and the
parent respawns us. Audio becomes NON-FATAL by construction.

PROTOCOL (parent → child over stdin, binary): a stream of frames, each = a little-endian
uint32 byte-length N followed by N bytes of int16 stereo-interleaved PCM (already volume-scaled
by the parent). N==0 or EOF = clean shutdown. All status goes to STDERR (stdout is unused) so
it interleaves into the run log without corrupting the PCM pipe.

RUN (normally spawned by pokemon_audio.AudioPump; standalone only for debugging):
  python pokemon_agent/audio_child.py --device <idx> --rate 48000
"""
import argparse
import struct
import sys


def _log(msg):
    print(f"   [pkmn-audio-child] {msg}", file=sys.stderr, flush=True)


def _read_exactly(stream, n):
    """Read exactly n bytes from a binary stream, or return None on EOF/short read."""
    chunks = []
    got = 0
    while got < n:
        b = stream.read(n - got)
        if not b:                      # EOF — parent closed the pipe (clean shutdown or parent gone)
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=-1, help="output device index (-1 = default)")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=2)
    args = ap.parse_args()

    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:                                 # pragma: no cover
        _log(f"!! sounddevice/numpy unavailable ({e}) — audio child cannot start; exiting.")
        return 2

    device = None if args.device is None or args.device < 0 else args.device
    try:
        st = sd.OutputStream(samplerate=args.rate, channels=args.channels, dtype="int16",
                             device=device, blocksize=0, latency="low")
        st.start()
    except Exception as e:
        # A failure to OPEN the device is a real config problem, not a transient abort. Report loud
        # and exit non-zero so the parent's restart-accounting eventually gives up (rather than thrash).
        _log(f"!! could not open OutputStream (device={device}, rate={args.rate}): {e} — exiting.")
        return 3

    _log(f"LIVE: device={device}, {args.rate}Hz, {args.channels}ch — writing until parent EOF.")
    stdin = sys.stdin.buffer
    frames = 0
    try:
        while True:
            hdr = _read_exactly(stdin, 4)
            if hdr is None:                                # EOF → parent done with us
                _log("parent closed the pipe (EOF) — clean shutdown.")
                break
            (n,) = struct.unpack("<I", hdr)
            if n == 0:                                      # explicit sentinel
                _log("received shutdown sentinel — clean shutdown.")
                break
            payload = _read_exactly(stdin, n)
            if payload is None:
                _log("pipe ended mid-frame — shutting down.")
                break
            pcm = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
            # THE DANGEROUS CALL. A native WASAPI abort() here kills ONLY this process; the parent
            # (emulator) is a separate process and survives + respawns us. That is the whole design.
            st.write(pcm)
            frames += 1
    except (BrokenPipeError, OSError) as e:
        _log(f"pipe/OS error ({e}) — shutting down.")
    finally:
        try:
            st.stop(); st.close()
        except Exception:
            pass
    _log(f"stopped after {frames} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
