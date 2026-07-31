"""recon_rescue_tp_check.py — verifier for the RESCUE TELEPORT escape hatch (the border-war TP).

Live 2026-07-31 ~15:00, Jonny: "she's still stuck at the gap between route 4 and cerulean.
please tp her somewhere so she can actually finish this game." The escape hatch is Abra's field
TELEPORT (move 100) — the game's own no-pathfinding warp to the last Pokémon Center — fired by
(a) a one-shot repo seed (RESCUE_TP.json) at the roam-tick top, and (b) the seam-thrash breaker.

Proves the gating logic WITHOUT the emulator, by binding the real methods to a fake campaign:

  seed checks (_rescue_tp_seed_check):
  1  no seed file                      -> no-op (no fire, nothing consumed)
  2  badge mismatch                    -> no fire, seed left UNconsumed (a later badge may match)
  3  match but NOT on a trap map      -> consumed as unnecessary, NO teleport fired
  4  match + on a trap map            -> fires; a VERIFIED teleport consumes the id
  5  fired but teleport FAILED        -> NOT consumed (next relaunch retries); bounded at 3/process
  6  consumed id                       -> never fires again

  teleport checks (_teleport_rescue):
  7  mid-battle                        -> refused (False), TeachFlow never constructed
  8  no party mon knows Teleport      -> refused (False)
  9  happy path (map changed)         -> True; seam history + nomove streak cleared

RUN:  python3 -u recon_rescue_tp_check.py
"""
import json
import os
import sys
import tempfile
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

import campaign as C          # noqa: E402

FAILS = []


def check(n, label, cond):
    print(f"  [{n}] {label}: {'OK' if cond else '!! FAIL'}")
    if not cond:
        FAILS.append(n)


class FakeBridge:
    def rd8(self, addr):
        return 6


class FakeWorld:
    def __init__(self, name="Route 4"):
        self._n = name

    def name(self, m):
        return self._n


class FakeCamp:
    _rescue_tp_seed_check = C.Campaign._rescue_tp_seed_check
    _teleport_rescue = C.Campaign._teleport_rescue

    def __init__(self, place="Route 4"):
        self.b = FakeBridge()
        self.world = FakeWorld(place)
        self.tp_calls = 0
        self.tp_result = True
        self._seam_hist = [(1, 2, 3)]
        self._nomove_streak = 5
        self._dead_moves = {"x"}

    def has_badge(self, flag):
        return flag in (0x820, 0x821)          # 2 badges

    def on_event(self, *a, **k):
        pass

    def _wait_overworld(self, *a, **k):
        pass

    def _teleport_rescue_stub(self, reason):
        self.tp_calls += 1
        return self.tp_result


def seed_env(tmp, badge_count=2, only_near=("route 4", "cerulean"), consumed=()):
    C.RESCUE_TP_JSON = os.path.join(tmp, "RESCUE_TP.json")
    C.RESCUE_TP_DONE = os.path.join(tmp, "rescue_tp_consumed.json")
    with open(C.RESCUE_TP_JSON, "w", encoding="utf-8") as f:
        json.dump({"id": "tp_test", "badge_count": badge_count,
                   "only_near": list(only_near)}, f)
    if consumed:
        with open(C.RESCUE_TP_DONE, "w", encoding="utf-8") as f:
            json.dump(list(consumed), f)


def consumed():
    if not os.path.exists(C.RESCUE_TP_DONE):
        return []
    with open(C.RESCUE_TP_DONE, encoding="utf-8") as f:
        return json.load(f)


print("== rescue-tp checks ==")
_json0, _done0 = C.RESCUE_TP_JSON, C.RESCUE_TP_DONE
_map_g0 = C.tv.map_id
C.tv.map_id = lambda b: (3, 22)          # she "stands on Route 4" for every seed check

# [1] no seed file
with tempfile.TemporaryDirectory() as tmp:
    C.RESCUE_TP_JSON = os.path.join(tmp, "nope.json")
    C.RESCUE_TP_DONE = os.path.join(tmp, "done.json")
    cp = FakeCamp()
    cp._teleport_rescue = cp._teleport_rescue_stub
    cp._rescue_tp_seed_check()
    check(1, "no seed file -> no-op", cp.tp_calls == 0 and consumed() == [])

# [2] badge mismatch -> no fire, unconsumed
with tempfile.TemporaryDirectory() as tmp:
    seed_env(tmp, badge_count=5)
    cp = FakeCamp()
    cp._teleport_rescue = cp._teleport_rescue_stub
    cp._rescue_tp_seed_check()
    check(2, "badge mismatch -> no fire, unconsumed", cp.tp_calls == 0 and consumed() == [])

# [3] match but not on a trap map -> consumed unnecessary, no fire
with tempfile.TemporaryDirectory() as tmp:
    seed_env(tmp)
    cp = FakeCamp(place="Vermilion City")
    cp._teleport_rescue = cp._teleport_rescue_stub
    cp._rescue_tp_seed_check()
    check(3, "off trap map -> consumed, NO teleport", cp.tp_calls == 0 and consumed() == ["tp_test"])

# [4] match + trap map + verified teleport -> fired once, consumed
with tempfile.TemporaryDirectory() as tmp:
    seed_env(tmp)
    cp = FakeCamp(place="Route 4")
    cp._teleport_rescue = cp._teleport_rescue_stub
    cp._rescue_tp_seed_check()
    check(4, "trap map + verified -> fired, consumed", cp.tp_calls == 1 and consumed() == ["tp_test"])
    cp._rescue_tp_seed_check()
    check(6, "consumed id -> never fires again", cp.tp_calls == 1)

# [5] teleport fails -> unconsumed, bounded retries
with tempfile.TemporaryDirectory() as tmp:
    seed_env(tmp)
    cp = FakeCamp(place="Route 4")
    cp.tp_result = False
    cp._teleport_rescue = cp._teleport_rescue_stub
    for _ in range(6):
        cp._rescue_tp_seed_check()
    check(5, "failed teleport -> unconsumed, capped at 3 attempts",
          cp.tp_calls == 3 and consumed() == [])

# teleport-side guards: monkeypatch the state/travel reads the real method makes
_in_battle0, _knows0 = C.st.in_battle, C.st.party_knows_move
_map0, _coords0 = C.tv.map_id, C.tv.coords
_species0 = C.st.read_party_species
try:
    C.tv.map_id = lambda b: (3, 22)
    C.tv.coords = lambda b: (10, 10)
    C.st.read_party_species = lambda b, s: 63          # Abra
    # [7] mid-battle -> refused
    C.st.in_battle = lambda b: True
    C.st.party_knows_move = lambda b, m, n: 4
    cp = FakeCamp()
    check(7, "mid-battle -> refused", cp._teleport_rescue("t") is False)
    # [8] nobody knows Teleport -> refused
    C.st.in_battle = lambda b: False
    C.st.party_knows_move = lambda b, m, n: None
    cp = FakeCamp()
    check(8, "no teleporter in party -> refused", cp._teleport_rescue("t") is False)
    # [9] happy path: TeachFlow "warps" (map change), verify -> True + slates cleared
    C.st.party_knows_move = lambda b, m, n: 4

    class FakeFlow:
        def __init__(self, camp, log=None, on_event=None):
            pass

        def use_field_move(self, slot, verify, label="", max_seconds=60):
            C.tv.map_id = lambda b: (3, 2)             # the warp: new map
            return "used" if verify() else "failed"
    import hm_teach
    _flow0 = hm_teach.TeachFlow
    hm_teach.TeachFlow = FakeFlow
    try:
        cp = FakeCamp()
        ok = cp._teleport_rescue("seam-thrash breaker")
        check(9, "happy path -> True, border-war slates cleared",
              ok is True and cp._seam_hist == [] and cp._nomove_streak == 0
              and cp._dead_moves == set())
    finally:
        hm_teach.TeachFlow = _flow0
finally:
    C.st.in_battle, C.st.party_knows_move = _in_battle0, _knows0
    C.tv.map_id, C.tv.coords = _map_g0, _coords0
    C.st.read_party_species = _species0
    C.RESCUE_TP_JSON, C.RESCUE_TP_DONE = _json0, _done0

print("== result:", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")
sys.exit(1 if FAILS else 0)
