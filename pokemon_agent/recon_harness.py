"""recon_harness.py - CORRECTED recon methodology (locked in 2026-06-24 after the "non-determinism"
ghost). The bug: reusing one Bridge across trials accumulates hidden cycle/timing state, so each
press lands on a core disturbed by the previous trial -> false "non-deterministic input". An
emulator is deterministic; the fix is a FRESH core per trial + a clean settle before each press.

USE THIS for any input/menu recon. Never reuse a Bridge across trials you want to compare.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")


def fresh(state=None, boot=10, owner="agent"):
    """A FRESH core (no carried-over state) loaded from `state`, advanced `boot` frames, input
    owner claimed. The unit of every comparable trial."""
    b = Bridge(ROM)
    if state:
        with open(os.path.join(STATES, state), "rb") as f:
            b.load_state(f.read())
    for _ in range(boot):
        b.run_frame()
    if owner:
        b.set_input_owner(owner)
    return b


def clean_tap(b, key, settle=18, hold=2, after=14):
    """A clean, settled key tap: settle (no input) so the menu is stable, a SHORT hold (a long
    hold reads as held on these menus), then release + advance. Returns nothing; read state after."""
    for _ in range(settle):
        b.run_frame()
    b.core.set_keys(b._keymask[key])
    for _ in range(hold):
        b.run_frame()
    b.core.set_keys(0)
    for _ in range(after):
        b.run_frame()


def trials(state, action, n=6, **fresh_kw):
    """Run `action(b)` on `n` FRESH cores from `state`; return the list of results. Use this to
    prove determinism (all-equal) instead of eyeballing a reused-bridge run."""
    out = []
    for _ in range(n):
        b = fresh(state, **fresh_kw)
        out.append(action(b))
    return out
