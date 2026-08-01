"""recon_trashcan_puzzle_check.py — headless verifier for TrashCanPuzzle logic (2026-08-01).

No ROM. Proves the Surge trash-can solver's pure helpers + short-circuit / re-entry contracts
that the live thrash exposed:

  1  filter_can_sites drops gym statues, keeps the pret 5×3 lattice
  2  adjacent_cans uses grid pitch (not Manhattan≤3 that pulled in statues)
  3  already-solved short-circuit when FLAG_BOTH is set
  4  campaign head_to_gym re-enters beat_gym when standing IN Vermilion Gym
  5  structural park of head_to_gym is waived while FLAG_BOTH is unset (source assert)

RUN:  python3 -u recon_trashcan_puzzle_check.py
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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

import env_puzzle as EP       # noqa: E402
import campaign as C          # noqa: E402

FAILS = []


def check(n, label, cond):
    print(f"  [{n}] {label}: {'OK' if cond else '!! FAIL'}")
    if not cond:
        FAILS.append(n)


print("== TrashCanPuzzle logic checks ==")

# pret VermilionCity_Gym bg events: 15 cans + 2 statues
RAW = [
    (3, 17), (7, 17),                          # statues (NOT cans)
    (1, 10), (3, 10), (5, 10), (7, 10), (9, 10),
    (1, 12), (3, 12), (5, 12), (7, 12), (9, 12),
    (1, 14), (3, 14), (5, 14), (7, 14), (9, 14),
]
cans = EP.filter_can_sites(RAW)
check(1, "filter keeps 15 cans, drops 2 statues",
      len(cans) == 15 and (3, 17) not in cans and (7, 17) not in cans
      and (1, 10) in cans and (9, 14) in cans)

# Adjacent to can id 8 @ (5,12): neighbors (3,12),(7,12),(5,10),(5,14) — NOT (3,14) diagonal,
# NOT statue (3,17)/(7,17) which Manhattan≤3 from (5,14) / (3,14) wrongly included.
nbr_mid = EP.adjacent_cans((5, 12), cans)
check(2, "mid-can adjacency is 4 grid neighbors (no diagonal)",
      set(nbr_mid) == {(3, 12), (7, 12), (5, 10), (5, 14)})

nbr_corner = EP.adjacent_cans((1, 10), cans)
check(3, "corner can adjacency is (3,10)+(1,12) only",
      set(nbr_corner) == {(3, 10), (1, 12)})

# Old Manhattan≤3 bug: from (3,14) statue (3,17) is Manhattan 3
old_manhattan = [s for s in RAW
                 if s != (3, 14) and abs(s[0] - 3) + abs(s[1] - 14) <= 3]
check(4, "Manhattan≤3 wrongly includes statue (3,17) — our filter must not",
      (3, 17) in old_manhattan and (3, 17) not in EP.adjacent_cans((3, 14), cans))

nbr_bottom = EP.adjacent_cans((3, 14), cans)
check(5, "bottom-row neighbors exclude statues",
      set(nbr_bottom) == {(1, 14), (5, 14), (3, 12)} and (3, 17) not in nbr_bottom)


# already-solved short-circuit
class _FlagBridge:
    def __init__(self, both=False, temp=False):
        self.both, self.temp = both, temp


class _Camp:
    def __init__(self, both=False):
        self.b = _FlagBridge(both=both)
        self.events = []
        self.render = lambda: None
        self.trav = None

    def on_event(self, *a, **k):
        self.events.append(a[0] if a else "")


_real_rf = EP.read_flag


def _fake_rf(b, flag):
    if flag == EP.FLAG_BOTH_SWITCHES:
        return bool(getattr(b, "both", False))
    if flag == EP.FLAG_TEMP_1:
        return bool(getattr(b, "temp", False))
    return False


EP.read_flag = _fake_rf
try:
    pz = EP.TrashCanPuzzle(_Camp(both=True), log=lambda m: None)
    check(6, "FLAG_BOTH set -> run() returns 'already'", pz.run() == "already")
finally:
    EP.read_flag = _real_rf

# campaign constants / re-entry contract
check(7, "VERMILION_GYM_INTERIOR is (9, 7)",
      getattr(C, "VERMILION_GYM_INTERIOR", None) == (9, 7))
check(8, "SURGE_FRONT is (5, 3)", C.SURGE_FRONT == (5, 3))

src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
check(9, "head_to_gym re-enters beat_gym when inside Vermilion Gym",
      "VERMILION_GYM_INTERIOR" in src
      and "_in_surge_gym" in src
      and "ALREADY INSIDE" in src)
check(10, "structural park waived while Surge FLAG_BOTH unset",
      "SURGE PUZZLE STILL OPEN" in src and "FLAG_BOTH" in src)
check(11, "park discard of head_to_gym present for surge puzzle-open",
      'discard("head_to_gym")' in src or "discard('head_to_gym')" in src)

check(12, "GO-HARD cap waived while Surge puzzle open",
      "GO-HARD cap WAIVED" in src)

# filter fallback: only statues + few cans still returns something sane
sparse = EP.filter_can_sites([(3, 17), (7, 17), (1, 10), (3, 10), (5, 10)])
check(13, "sparse lattice (≥3 cans matching pret) kept without statues",
      len(sparse) == 3 and (3, 17) not in sparse)

print("== result:", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")
sys.exit(1 if FAILS else 0)
