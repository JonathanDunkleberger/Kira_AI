"""repro_parcel_crash.py — reproduce the reproducible hard-mgba-kill at the Viridian parcel handoff.

Drives deliver_parcel() headless from a pre-parcel state with:
  * faulthandler — on a C-level SIGSEGV (libmgba), dump the Python stack at the fault (pinpoints the
    EXACT line calling into the emulator when it died — the thing a hard kill normally hides).
  * read instrumentation — flag any bridge read with a WILD address (a bad pointer deref) LOUD before
    it reaches libmgba, so the last flushed line names the crashing op even if faulthandler can't.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\repro_parcel_crash.py [state]   (default after_pick_bulbasaur)
"""
import os
import sys
import time
import faulthandler

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

faulthandler.enable()                      # dump C-fault Python stack to stderr on SIGSEGV

from bridge import Bridge                  # noqa: E402
import travel as tv                        # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def log(m):
    print(f"   [parcel-repro] {m}", flush=True)


# Valid GBA read regions (base, size). A read outside these is a WILD address — almost certainly a
# garbage/stale pointer deref, the classic hard-kill source.
_REGIONS = [
    (0x00000000, 0x00004000),  # BIOS
    (0x02000000, 0x00040000),  # EWRAM
    (0x03000000, 0x00008000),  # IWRAM
    (0x04000000, 0x00000400),  # IO
    (0x05000000, 0x00000400),  # palette
    (0x06000000, 0x00018000),  # VRAM
    (0x07000000, 0x00000400),  # OAM
    (0x08000000, 0x02000000),  # ROM
    (0x0E000000, 0x00010000),  # SRAM/Flash
]


def _addr_ok(addr):
    if not isinstance(addr, int) or addr < 0 or addr > 0xFFFFFFFF:
        return False
    for base, size in _REGIONS:
        if base <= addr < base + size:
            return True
    return False


def instrument(b):
    """Wrap the bridge reads so a WILD address is logged LOUD (with caller) BEFORE it hits libmgba."""
    import traceback
    for name in ("rd8", "rd16", "rd32", "rds16"):
        orig = getattr(b, name)
        def make(nm, fn):
            def w(addr):
                if not _addr_ok(addr):
                    c = traceback.extract_stack(limit=3)[0]
                    print(f"   [parcel-repro] !! WILD {nm}(0x{addr & 0xFFFFFFFF:08X}) "
                          f"<- {os.path.basename(c.filename)}:{c.lineno} — this read would hit libmgba",
                          flush=True)
                    return 0                      # don't pass the wild addr to the emulator
                return fn(addr)
            return w
        setattr(b, name, make(name, orig))
    orig_rb = b.read_bytes
    def rb(addr, n):
        if not _addr_ok(addr) or not _addr_ok(addr + max(0, n - 1)) or n < 0 or n > 0x10000:
            print(f"   [parcel-repro] !! WILD read_bytes(0x{addr & 0xFFFFFFFF:08X}, n={n})", flush=True)
            return b"\x00" * max(0, min(n, 64))
        return orig_rb(addr, n)
    b.read_bytes = rb


def main():
    state = sys.argv[1] if len(sys.argv) > 1 else "after_pick_bulbasaur"
    b = Bridge(ROM)
    p = resolve_state(state + ".state")
    if not p:
        raise SystemExit(f"state {state}.state not found")
    with open(p, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    b.set_input_owner("agent")
    instrument(b)

    # AUDIO-DRAIN test (no output device — safe on a live rig): reproduce the mgba-side audio read
    # the live AudioPump does every frame (self.stereo.read), WITHOUT PortAudio. If the hard kill is
    # the fanfare hitting the audio resampler, it'll crash HERE (the item-get jingle at the handoff).
    if os.getenv("REPRO_AUDIO", "1") == "1":
        try:
            b.core.set_audio_buffer_size(2048)
            _stereo = b.core.get_audio_channels()
            _stereo.set_rate(48000)
            _stereo.clear()
            _orig_rf = b.core.run_frame
            _stats = {"max_avail": 0, "frames": 0}
            def _rf_drain():
                _orig_rf()
                n = _stereo.available
                _stats["frames"] += 1
                if n > _stats["max_avail"]:
                    _stats["max_avail"] = n
                    if n > 8192:
                        print(f"   [parcel-repro] !! LARGE audio available={n} (frame {_stats['frames']})", flush=True)
                if n > 0:
                    _stereo.read(n)          # the exact mgba audio read the live pump does
            b.core.run_frame = _rf_drain
            log("AUDIO DRAIN active (mgba-side read every frame, no device) — REPRO_AUDIO=0 to disable")
        except Exception as e:
            log(f"audio-drain setup failed ({e}) — continuing without it")

    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: log(f"[event] {s}"))
    log(f"booted {state}: map={tv.map_id(b)} coords={tv.coords(b)}")
    log("calling deliver_parcel() — watching for the hard kill at the Viridian clerk handoff…")
    t0 = time.time()
    out = camp.deliver_parcel()
    log(f"deliver_parcel returned {out!r} in {time.time()-t0:.0f}s — NO crash this run "
        f"(map={tv.map_id(b)}, balls={camp._ball_count()})")


if __name__ == "__main__":
    main()
