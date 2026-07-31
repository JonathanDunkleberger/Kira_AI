"""recon_watchdog_latch_check.py — verifier for the watchdog-latch lifecycle (the Center bail storm).

Live 2026-07-31 14:15 log: the nurse-heal jingle tripped the watchdog (frozen_box, one static box
8s+), then the Center-EXIT sub-loop launched 45+ travel legs that ALL instabailed on the same latch
("bailing this leg LOUD" storm) — and the 45s TTL self-heal never fired. She stood frozen in the
Cerulean Center while her goal said "go to Vermilion" (the Route-4↔Cerulean 'stuck' Jonny watched).

Proves campaign._stuck_latched WITHOUT the emulator, by binding the real method to a fake campaign:

  1  no latch                         -> False (no bail), bail counter stays 0
  2  fresh latch, first leg           -> True (ONE bail — the intended unwind-to-roam-top signal)
  3  same latch, legs 2..3            -> leg 2 True, leg 3 SELF-HEALS (storm breaker): latch cleared,
                                         watch reset, False returned — play resumes in ~3 legs not 45s
  4  latch older than the 45s TTL     -> stale self-heal still fires (belt kept)
  5  roam top consumed the latch      -> counter resets; a NEW latch gets a fresh 3-bail budget

RUN:  ../.venv/Scripts/python.exe -u recon_watchdog_latch_check.py
"""
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# _stuck_latched is pure latch bookkeeping — no emulator needed. On a machine without mgba
# (the Mac dev box), stub the import chain so `import campaign` succeeds; the PC venv has the
# real thing and skips this.
try:
    import mgba.core          # noqa: F401
except ImportError:
    _root = types.ModuleType("mgba")
    for _m in ("core", "image", "vfs", "log", "png"):
        _sub = types.ModuleType(f"mgba.{_m}")
        setattr(_root, _m, _sub)
        sys.modules[f"mgba.{_m}"] = _sub
    _root.log.silence = lambda: None
    sys.modules["mgba"] = _root

import campaign as C          # noqa: E402


class FakeWatch:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeCamp:
    _stuck_latched = C.Campaign._stuck_latched

    def __init__(self):
        self._stuck_request = None
        self._stuckwatch = FakeWatch()
        self._latch_bails = 0


FAIL = 0


def check(name, got, want):
    global FAIL
    ok = got == want
    print(f"  [{'ok' if ok else 'XX'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAIL += 1


print("== 1: no latch -> no bail ==")
c = FakeCamp()
check("no latch", c._stuck_latched(), False)
check("counter stays 0", c._latch_bails, 0)

print("== 2+3: fresh latch — one honored bail, storm broken at leg 3 ==")
c = FakeCamp()
c._stuck_request = {"reason": "frozen_box", "ts": time.time()}
check("leg 1 bails (unwind signal)", c._stuck_latched(), True)
check("leg 2 bails", c._stuck_latched(), True)
check("leg 3 SELF-HEALS (storm breaker)", c._stuck_latched(), False)
check("latch cleared", c._stuck_request, None)
check("watch reset", c._stuckwatch.resets, 1)
check("leg 4 clean (no residue)", c._stuck_latched(), False)

print("== 4: stale TTL self-heal still fires ==")
c = FakeCamp()
c._stuck_request = {"reason": "frozen_world", "ts": time.time() - C.WATCHDOG_LATCH_TTL_S - 5}
check("stale latch self-heals", c._stuck_latched(), False)
check("latch cleared", c._stuck_request, None)
check("watch reset", c._stuckwatch.resets, 1)

print("== 5: roam-top consumption resets the storm budget ==")
c = FakeCamp()
c._stuck_request = {"reason": "frozen_box", "ts": time.time()}
check("leg 1 bails", c._stuck_latched(), True)
c._stuck_request = None                      # roam top consumed it (the normal recovery path)
check("post-consume clean", c._stuck_latched(), False)
c._stuck_request = {"reason": "frozen_box", "ts": time.time()}   # a NEW latch later
check("new latch: leg 1 bails again", c._stuck_latched(), True)
check("new latch: leg 2 bails", c._stuck_latched(), True)
check("new latch: leg 3 self-heals", c._stuck_latched(), False)

print()
if FAIL:
    print(f"RESULT: {FAIL} check(s) FAILED")
    sys.exit(1)
print("RESULT: all checks passed")
