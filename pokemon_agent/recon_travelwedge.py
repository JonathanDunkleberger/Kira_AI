"""recon_travelwedge.py - CONTROL for increment-3.5: travel-layer progress guard + catch reachability.

The live bug (commit 709fa05, watched): from (9,60) on Route 2, the catch-router handed travel
UNREACHABLE grass targets ((2,2),(6,7)); travel reported "no clean path / genuine wall" and the
router re-tried the same dead targets ~40x for MINUTES, below roam-tick granularity, so the macro
watchdog never got a tick. Two fixes, both "surface up, never spin":

A) TRAVEL GUARD: travel() to an UNREACHABLE coord must STOP after TRAVEL_STALL_RETRIES identical-
   fingerprint retries and return a loud structured 'no_path' (+ last_fail_reason) FAST — not spin
   to its time budget. A REACHABLE coord must still 'arrive' (the guard never false-fires).
B) CATCH BOUND: catch_one() must (pre-check) drop unreachable grass and, if travel keeps pathing
   nowhere, return 'no_reachable_target' UP to roam after a BOUNDED number of tries — not cycle the
   dead target set until its 300s budget.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_travelwedge.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import world_fingerprint as wf                 # noqa: E402
import travel as tv                             # noqa: E402
from bridge import Bridge                        # noqa: E402
from campaign import Campaign                    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WORKSHOP = os.path.join(_HERE, "states", "workshop")


def _load(name):
    b = Bridge(ROM)
    with open(os.path.join(WORKSHOP, name + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    return b


def part_a():
    print("\n==== PART A: travel-layer progress guard ====")
    ok = True
    b = _load("brock_done")
    logs = []
    trav = tv.Traveler(b, battle_runner=lambda: "win", render=lambda: None,
                       log=lambda m: logs.append(m), owner="agent")

    # (1) UNREACHABLE coord (200,200 is outside any map's playable rect) -> must wedge fast + loud.
    t0 = time.time()
    r = trav.travel(target_map=None, arrive_coord=(200, 200), max_steps=400, max_seconds=120)
    dt = time.time() - t0
    wedge_logged = any("TRAVEL WEDGE" in m for m in logs)
    print(f"  unreachable (200,200) -> {r!r} reason={trav.last_fail_reason!r} in {dt:.1f}s; "
          f"TRAVEL WEDGE logged={wedge_logged}")
    ok &= (r == "no_path") and trav.last_fail_reason in ("no_route", "npc_block") and wedge_logged
    ok &= (dt < 60)                                   # FAST surface, not a 120s spin

    # (2) REACHABLE coord nearby -> still ARRIVES (the guard must not false-fire on a real route).
    g = tv.Grid(b)
    cur = tv.coords(b)
    target = None
    for d in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (2, 0), (0, -2), (-2, 0)):
        cand = (cur[0] + d[0], cur[1] + d[1])
        if tv.bfs(g, cur, lambda t: t == cand, walkable=g.walkable):
            target = cand
            break
    if target is None:
        print("  (no reachable neighbour found to test arrival - skipping (2))")
    else:
        r2 = trav.travel(target_map=None, arrive_coord=target, max_steps=200, max_seconds=60)
        print(f"  reachable {target} -> {r2!r} (want 'arrived')")
        ok &= (r2 == "arrived")

    print("  PART A:", "PASS" if ok else "FAIL")
    return ok


def part_b():
    print("\n==== PART B: catch_one reachability + bounded-spin ====")
    ok = True
    b = _load("route3_caught")                       # Route 3: has grass
    camp = Campaign(b, battle_runner=lambda: "win", on_event=lambda *a, **k: None)

    calls = [0]
    real_travel = camp.trav.travel

    def fake_travel(*a, **k):
        """Simulate every grass waypoint being unreachable RIGHT NOW (NPC won't clear / walled)."""
        calls[0] += 1
        camp.trav.last_fail_reason = "no_route"
        return "no_path"

    camp.trav.travel = fake_travel
    t0 = time.time()
    out = camp.catch_one(max_seconds=300)            # would have spun ~minutes pre-fix
    dt = time.time() - t0
    camp.trav.travel = real_travel
    print(f"  catch_one (all targets unreachable) -> {out!r} in {dt:.1f}s; travel calls={calls[0]}")
    # Must surface a structured failure UP, bounded — NOT spin to the 300s budget.
    ok &= out in ("no_reachable_target", "no_grass")
    ok &= (dt < 60) and (calls[0] <= 8)              # bounded retries, fast return
    print("  PART B:", "PASS" if ok else "FAIL")
    return ok


def main():
    a = part_a()
    b = part_b()
    print("\n==== RESULT:", "ALL CONTROLS PASS" if (a and b) else "SOME CONTROLS FAILED", "====")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
