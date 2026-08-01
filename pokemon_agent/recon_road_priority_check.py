"""recon_road_priority_check.py — verifier for the BILLED-ROAD PRIORITY gym routing (2026-08-01).

Live 2026-08-01 03:5x (the "repeating repeating" morning): head_to_gym ran the world-graph
warp-route FIRST and the billed road only as fallback. The seeded Diglett's-Cave corridor made
the graph "able" to route Cerulean -> Vermilion the long way WEST (Route 4 -> Mt. Moon -> ...),
so momentum marched her onto Route 4, the next leg wedged, head_to_gym got structurally pruned,
and her menu collapsed to talk_npc forever. The fix flips the order: a billed road OWNS the leg;
the graph is the fallback.

Proves _road_step (the road executor) drives the RIGHT moves using the REAL KB roads, headless:

  1  standing IN Cerulean (3,3)  -> billed leg: walk SOUTH into Route 5 (3,23) — never west
  2  standing ON Route 4 (3,22)  -> off-road steer: EAST back to the Cerulean road anchor
  3  standing ON Route 5 (3,23)  -> 'pass' leg: the Underground Path door-passthrough is tried

RUN:  python3 -u recon_road_priority_check.py
"""
import json
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

import campaign as C          # noqa: E402

FAILS = []


def check(n, label, cond):
    print(f"  [{n}] {label}: {'OK' if cond else '!! FAIL'}")
    if not cond:
        FAILS.append(n)


class FakeGate:
    def recognize(self, cur, blocked_dir=None):
        return None


class FakeWorld:
    def edge_neighbor(self, m, go):
        return None            # no live binding — the KB's expected ids stand

    def name(self, m):
        return str(m)


class FakeCamp:
    _gym_road = C.Campaign._gym_road
    _road_step = C.Campaign._road_step

    def __init__(self, cur):
        with open(os.path.join(_HERE, "gamedata", "frlg_gates.json"), encoding="utf-8") as f:
            self._questline_kb = json.load(f)   # the REAL roads drive the checks
        self.world = FakeWorld()
        self._gate_recognizer = FakeGate()
        self.state = {"map": cur, "next_gym": {"city": "Vermilion City", "leader": "Lt. Surge"}}
        self.calls = []

    def on_event(self, *a, **k):
        pass

    def _wall_avoid(self, state):
        return set()

    def _story_gate_avoid(self, state):
        return set()

    def _edge_travel(self, nxt, go):
        self.calls.append(("edge", tuple(nxt) if nxt else None, go))
        return "arrived"

    def _door_passthrough(self, want_map=None):
        self.calls.append(("passthrough", want_map))
        return "crossed"

    def _next_step_rideable(self, cur, dst, avoid):
        # the learned graph reaches only walked ground: Cerulean yes (east of Route 4),
        # Route 5/6/Vermilion no (never visited)
        if tuple(dst) == (3, 3):
            return ((3, 3), "edge", "east")
        return None


print("== billed-road priority checks (real KB roads) ==")

# [1] in Cerulean: the billed leg walks SOUTH into Route 5
cp = FakeCamp((3, 3))
road = cp._gym_road(cp.state["next_gym"])
out = cp._road_step(cp.state, road)
check(1, "Cerulean -> billed SOUTH to Route 5 (3,23)",
      out == "arrived" and cp.calls == [("edge", (3, 23), "south")])

# [2] on Route 4 (off-road): steer EAST back to the Cerulean anchor — never deeper west
cp = FakeCamp((3, 22))
road = cp._gym_road(cp.state["next_gym"])
out = cp._road_step(cp.state, road)
check(2, "Route 4 (off-road) -> steer EAST to Cerulean anchor",
      out == "arrived" and cp.calls and cp.calls[0] == ("edge", (3, 3), "east"))

# [3] on Route 5: the 'pass' leg goes through the Underground Path door-passthrough
_map0 = C.tv.map_id
C.tv.map_id = lambda b: (3, 24)          # after the crossing she reads as standing on Route 6
try:
    cp = FakeCamp((3, 23))
    cp.b = None
    road = cp._gym_road(cp.state["next_gym"])
    out = cp._road_step(cp.state, road)
    check(3, "Route 5 -> Underground Path door-passthrough tried (and lands on Route 6)",
          out == "arrived" and any(c[0] == "passthrough" for c in cp.calls))
finally:
    C.tv.map_id = _map0

# [4] the retired teleport is invoked from NO automatic path (source-level assert)
src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
live_calls = [ln for ln in src.splitlines()
              if "self._teleport_rescue(" in ln and not ln.strip().startswith("#")
              and "def _teleport_rescue" not in ln]
check(4, "no automatic _teleport_rescue call sites remain", live_calls == [])

print("== result:", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")
sys.exit(1 if FAILS else 0)
