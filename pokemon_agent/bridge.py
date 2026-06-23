"""bridge.py - thin, isolated wrapper over the vendored libmgba-py core.

ZERO dependency on any Kira module. Exposes only the 4 primitives proven in M0
recon (load / step / press / read) plus framebuffer access, behind a tiny API so
M0 and M1 don't repeat the cffi plumbing.

Vendored mgba lives in pokemon_agent/vendor/ (libmgba 0.10.2, py3.10 win64).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import mgba.core      # noqa: E402
import mgba.image     # noqa: E402
import mgba.log       # noqa: E402

mgba.log.silence()    # mute native BIOS/DMA spam (recon: drowns stdout otherwise)

# GBA key names -> set via core.KEY_* at runtime (constants live on the core).
KEY_NAMES = ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "L", "R")


class Bridge:
    """One running FireRed core + framebuffer."""

    def __init__(self, rom_path: str):
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        self.core = mgba.core.load_path(rom_path)
        if self.core is None:
            raise RuntimeError(f"mgba.core.load_path returned None for {rom_path}")
        self.width, self.height = self.core.desired_video_dimensions()
        self.image = mgba.image.Image(self.width, self.height)
        # CRITICAL: set the video buffer BEFORE reset() — the renderer latches the
        # buffer pointer at reset; setting it after leaves the framebuffer blank.
        self.core.set_video_buffer(self.image)
        self.core.reset()
        self._keymask = {name: getattr(self.core, f"KEY_{name}") for name in KEY_NAMES}
        self._owner = None        # claimed input owner (None = unattributed/setup)
        self._trace = False       # when True, log every attributed input + its caller
        # advance one frame so the framebuffer is populated
        self.core.run_frame()

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def game_code(self) -> str:
        return self.core.game_code

    @property
    def game_title(self) -> str:
        return self.core.game_title

    @property
    def frame(self) -> int:
        return self.core.frame_counter

    # ── (b) step ────────────────────────────────────────────────────────────
    def run_frame(self):
        self.core.run_frame()

    # ── input ownership (the anti-phantom invariant) ──────────────────────────
    # The emulator pad has EXACTLY ONE deliberate owner at a time. Once an owner is
    # claimed, any set_keys/release/press from a different (or unattributed) source
    # is logged LOUDLY *with its caller* and DROPPED — phantom presses can never be
    # silent again (Constraint #3), and we can SEE where a stray press came from.
    # Before any claim (setup/boot), input is unattributed and allowed.
    def set_input_owner(self, owner, trace=False):
        prev = getattr(self, "_owner", None)
        self._owner = owner
        self._trace = trace
        if prev and prev != owner:
            print(f"   [Bridge] INPUT OWNER: {prev!r} -> {owner!r} (handoff)", flush=True)
        else:
            print(f"   [Bridge] INPUT OWNER = {owner!r} (sole writer; others dropped+logged)",
                  flush=True)

    def _owner_ok(self, owner, what):
        active = getattr(self, "_owner", None)
        if active is None:                 # nobody claimed yet → setup/boot, allow
            return True
        if owner == active:
            if getattr(self, "_trace", False):
                import traceback
                c = traceback.extract_stack(limit=3)[0]
                print(f"   [Bridge] input {what} by {owner!r} <- {c.filename.split(chr(92))[-1]}:{c.lineno}",
                      flush=True)
            return True
        import traceback
        c = traceback.extract_stack(limit=3)[0]
        print(f"   [Bridge] !! PHANTOM INPUT DROPPED: {what} owner={owner!r} but active "
              f"owner is {active!r}  <- {c.filename.split(chr(92))[-1]}:{c.lineno}", flush=True)
        return False

    # ── (c) press ───────────────────────────────────────────────────────────
    def set_keys(self, *names, owner=None):
        """Hold exactly these keys (replaces full key state). No names = release all.
        `owner` must match the claimed input owner once one is set (else dropped+logged)."""
        if not self._owner_ok(owner, "set_keys"):
            return
        mask = 0
        for n in names:
            mask |= self._keymask[n]
        self.core.set_keys(mask)

    def release(self, owner=None):
        if not self._owner_ok(owner, "release"):
            return
        self.core.set_keys(0)

    def press(self, name, hold_frames=8, gap_frames=8, on_frame=None, owner=None):
        """Hold `name` for hold_frames, release for gap_frames. `on_frame(i)` is
        called after every advanced frame (for rendering)."""
        self.set_keys(name, owner=owner)
        for _ in range(hold_frames):
            self.core.run_frame()
            if on_frame:
                on_frame()
        self.release(owner=owner)
        for _ in range(gap_frames):
            self.core.run_frame()
            if on_frame:
                on_frame()

    # ── (d) read memory (typed, full address space) ──────────────────────────
    def rd8(self, addr):
        return self.core.memory.u8[addr]

    def rd16(self, addr):
        return self.core.memory.u16[addr]

    def rd32(self, addr):
        return self.core.memory.u32[addr]

    def rds16(self, addr):
        return self.core.memory.s16[addr]

    def read_bytes(self, addr, n):
        return bytes(self.core.memory.u8[addr + i] for i in range(n))

    # ── framebuffer ───────────────────────────────────────────────────────────
    def frame_rgb(self):
        """Current framebuffer as a PIL RGB image (240x160)."""
        return self.image.to_pil().convert("RGB")

    # ── savestate (for handing a battle position to M1) ───────────────────────
    def save_state(self):
        return self.core.save_state()

    def load_state(self, data):
        # core.load_state wants a VFile, not raw bytes — wrap the saved bytes in an
        # in-memory VFile (save_state() returns bytes; this is the inverse).
        from mgba.vfs import VFile
        vf = VFile.fromEmpty()
        vf.write(data, len(data))
        vf.seek(0, 0)
        return self.core.load_state(vf)
