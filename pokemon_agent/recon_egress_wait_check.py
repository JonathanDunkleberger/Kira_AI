"""OFFLINE egress NPC-patience check (2026-08-05, the Cinnabar-Center door loop) — no emulator/ROM.

THE BUG CLASS: interior door egress with a roaming blocker. Pret ground truth for the two
live-run buildings (geometry verified against pokefirered map.json):
  - Cinnabar PC 1F: exit mats (6,8)/(7,8)/(8,8) — only (7,8) is the firing arrow — with a
    WANDER_AROUND Gentleman boxed (8-10, 6-8), overlapping the mat row + exit corridor;
  - One Island Network Center: a SINGLE exit warp (9,9) with a WANDER_AROUND kid boxed
    (5-7, 7-9) beside the doorway.
A wanderer blocks a tile for SECONDS; the old executor treated it as a wall on every tick
(re-plan -> detour -> blocker moved -> re-plan back = the visible undershoot/overshoot door
loop, ending in the escape hatch). The fix: _wait_for_npc_clear — HOLD POSITION, poll the
live object list (readback law), resume the SAME plan; bounded, with a motion gate so
stationary trainers/squatters keep the old fast interact path.

This drives the REAL Traveler.travel() loop against a scripted sim world:
  1. only-gap roaming blocker (site C, the no-path probe): she HOLDS ([egress]) and walks
     through when he steps off — never marks him / never no_route_npc_blocked;
  2. committed-step cross (site A, plan hysteresis): a wanderer steps INTO the plan
     mid-walk -> wait + resume the SAME plan (no detour, no blocked-mark);
  3. failed-press race (site B): the body lands on the tile between plan and press ->
     wait + retry the same step (no static-obstacle mark);
  4. stationary squatter: never moves -> bounded fallback to the OLD machinery
     (no_route_npc_blocked) — the escape hatch stays the LAST resort, not the routine.
Run:  python3 recon_egress_wait_check.py   (from pokemon_agent/) — PASS/FAIL per check.
"""
import os
import sys
import types

os.environ["POKEMON_RUN"] = "0"          # tile-atomic walk presses in the sim

# Mac dev box has no mgba/emulator stack — stub it (logic-only test, recon_lap_check pattern).
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import travel as T                        # noqa: E402
import pokemon_state as st                # noqa: E402
import world_fingerprint as wf            # noqa: E402

# speed the bounded waits up for the suite (live values: 12s timeout / 3.5s motion probe)
T.NPC_WAIT_TIMEOUT_S = 1.0
T.NPC_WAIT_PROBE_S = 0.25
T.NPC_WAIT_POLL_FRAMES = 4

PASS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class Sim:
    """A tiny interior: walls, one player, one NPC driven by a frame schedule and/or
    position triggers. NPCs block steps exactly like FRLG bodies do."""
    KEYS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

    def __init__(self, w, h, walls, player, npc, npc_cycle=None, npc_period=0):
        self.w, self.h = w, h
        self.walls = set(walls)
        self.player = tuple(player)
        self.npc = tuple(npc) if npc else None
        self.cycle = list(npc_cycle or ())
        self.ci = 0
        self.period = npc_period
        self.frames = 0
        self.dialogue = False
        self.visited = [tuple(player)]
        self.trigger = None              # (fire_pred(sim) -> bool, on_fire(sim))
        self.after_fire = None           # (delay_frames, npc_tile) — scheduled post-trigger move
        self._fired_at = None

    def tick(self):
        self.frames += 1
        if self.trigger is not None and self._fired_at is None and self.trigger[0](self):
            self.trigger[1](self)
            self._fired_at = self.frames
        if (self._fired_at is not None and self.after_fire is not None
                and self.frames - self._fired_at >= self.after_fire[0]):
            self.npc = tuple(self.after_fire[1])
            self.after_fire = None
        if self.cycle and self.period and self.frames % self.period == 0:
            j = (self.ci + 1) % len(self.cycle)
            if tuple(self.cycle[j]) != self.player:   # bodies never step onto the player
                self.ci = j
                self.npc = tuple(self.cycle[j])

    def step(self, key):
        d = self.KEYS.get(key)
        if d is None:
            if key == "A":
                self.dialogue = True     # talking to whatever she faces
            elif key == "B":
                self.dialogue = False
            return
        t = (self.player[0] + d[0], self.player[1] + d[1])
        if (t in self.walls or not (0 <= t[0] < self.w and 0 <= t[1] < self.h)
                or t == self.npc):
            return                        # blocked press = turn only (coords unchanged)
        self.player = t
        self.visited.append(t)


class SimBridge:
    def __init__(self, sim):
        self.sim = sim

    def run_frame(self):
        self.sim.tick()

    def set_input_owner(self, o):
        pass

    def press(self, key, *a, **k):
        self.sim.step(key)
        for _ in range(4):
            self.sim.tick()

    def rd8(self, a):
        return 0

    def rd16(self, a):
        return 0

    def rd32(self, a):
        return 0

    def rds16(self, a):
        return 0


class FakeGrid:
    def __init__(self, sim):
        self.sim = sim
        self.spin = set()
        self.sx_lo, self.sx_hi = 0, sim.w - 1
        self.sy_lo, self.sy_hi = 0, sim.h - 1

    def walkable(self, x, y):
        return (0 <= x < self.sim.w and 0 <= y < self.sim.h
                and (x, y) not in self.sim.walls)

    def walkable_safe(self, x, y):
        return self.walkable(x, y)

    def is_water(self, x, y):
        return False

    def walkable_or_surf(self, x, y):
        return self.walkable(x, y)

    def ledge_dir(self, x, y):
        return None

    def edge_open(self, x, y, dx, dy):
        return True


class FP:
    """World-fingerprint stand-in: equality over (player, npc, dialogue)."""

    def __init__(self, sim):
        self.key = (sim.player, sim.npc, sim.dialogue)
        self.menu_or_dialogue = sim.dialogue

    def __eq__(self, o):
        return isinstance(o, FP) and o.key == self.key


def make_traveler(sim):
    b = SimBridge(sim)
    T.coords = lambda bb: sim.player
    T.map_id = lambda bb: (12, 5)              # any interior map id (constant: no transitions)
    T.Grid = lambda bb: FakeGrid(sim)
    wf.fingerprint = lambda bb: FP(sim)
    wf.brief = lambda fp: "sim-fp"
    st.in_battle = lambda bb: False
    logs = []
    trav = T.Traveler(b, battle_runner=lambda: "won", log=lambda s: logs.append(s))
    trav._npc_tiles = lambda: ({sim.npc} if sim.npc else set())
    return trav, logs


def corridor_room():
    """15x9 room, wall row y=5 with the single gap at x=7 — the single-file door approach
    (the Cinnabar mat / One Island (9,9) door class)."""
    return [(x, 5) for x in range(15) if x != 7]


def main():
    print("== 1. only-gap ROAMING blocker: hold, he wanders off, she walks through ==")
    sim = Sim(15, 9, corridor_room(), player=(7, 2), npc=(7, 4),
              npc_cycle=[(7, 4), (8, 4)], npc_period=160)
    trav, logs = make_traveler(sim)
    r = trav.travel(target_map=None, arrive_coord=(7, 8), max_steps=120, max_seconds=20)
    waited = any("[egress] blocker on (7, 4)" in ln for ln in logs)
    resumed = any("[egress] blocker cleared" in ln for ln in logs)
    check("arrives at the mat", r == "arrived" and sim.player == (7, 8))
    check("[egress] wait fired on the gap blocker", waited)
    check("[egress] wait RESUMED (blocker wandered off)", resumed)
    check("blocker never marked into shared block memory (no oscillation seed)",
          not trav.blocked_npcs)
    check("never surfaced no_route_npc_blocked", r != "no_route_npc_blocked")

    print("== 2. committed-step cross: wanderer steps INTO the plan mid-walk ==")
    walls = [(x, 6) for x in range(3, 12)] + [(x, 8) for x in range(3, 12)]
    sim2 = Sim(15, 9, walls, player=(2, 7), npc=(8, 5))
    sim2.trigger = (lambda s: s.player == (7, 7), lambda s: setattr(s, "npc", (8, 7)))
    sim2.after_fire = (120, (8, 6))            # he wanders back off the corridor
    trav2, logs2 = make_traveler(sim2)
    r2 = trav2.travel(target_map=None, arrive_coord=(12, 7), max_steps=120, max_seconds=20)
    check("arrives through the corridor", r2 == "arrived" and sim2.player == (12, 7))
    check("[egress] committed-step wait fired",
          any("(committed step)" in ln for ln in logs2))
    check("no detour: she held the row-7 plan (no undershoot/overshoot)",
          all(y == 7 for _x, y in sim2.visited))
    check("no blocked-mark / no plain-NPC mark", not trav2.blocked_npcs
          and not any("static obstacle" in ln for ln in logs2))

    print("== 3. failed-press race: body lands on the tile between plan and press ==")
    sim3 = Sim(15, 9, walls, player=(2, 7), npc=(8, 5))
    # the trigger fires INSIDE the press: sim blocks the step the same frame he arrives
    _orig_step = sim3.step

    def _racing_step(key):
        if key == "RIGHT" and sim3.player == (7, 7) and sim3._fired_at is None:
            sim3.npc = (8, 7)
            sim3._fired_at = sim3.frames
        _orig_step(key)

    sim3.step = _racing_step
    sim3.after_fire = (120, (8, 6))
    trav3, logs3 = make_traveler(sim3)
    r3 = trav3.travel(target_map=None, arrive_coord=(12, 7), max_steps=120, max_seconds=20)
    check("arrives despite the race", r3 == "arrived" and sim3.player == (12, 7))
    check("[egress] failed-step wait fired",
          any("(failed E-step)" in ln for ln in logs3))
    check("tile never marked a static obstacle",
          not any("static obstacle" in ln for ln in logs3))
    check("no detour off row 7", all(y == 7 for _x, y in sim3.visited))

    print("== 4. STATIONARY squatter: bounded fallback to the old machinery ==")
    sim4 = Sim(15, 9, corridor_room(), player=(7, 2), npc=(7, 5))   # parked ON the gap, never moves
    trav4, logs4 = make_traveler(sim4)
    import time as _time
    t0 = _time.time()
    r4 = trav4.travel(target_map=None, arrive_coord=(7, 8), max_steps=120, max_seconds=20)
    took = _time.time() - t0
    check("old machinery surfaces the honest failure (escape hatch stays last resort)",
          r4 == "no_route_npc_blocked")
    check("motion gate bailed the wait fast (stationary detected)",
          any("stationary (trainer/squatter)" in ln for ln in logs4))
    check("bounded: the leg didn't burn the budget waiting", took < 10)

    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
