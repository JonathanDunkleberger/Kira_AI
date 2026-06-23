"""travel.py - the FEET: deterministic collision-grid pathfinder. NO LLM in movement.

Milestone 1: walk Route 1 (3,19) -> Viridian City (3,1), collision-only.

Firewall (same doctrine as the battle engine):
  HANDS = this deterministic BFS over the VERIFIED walkable grid. The LLM never
  freehand-picks a direction (that's what loops agents forever). VOICE (her self)
  decides WHERE to go at decision points + narrates; here the WHERE is fixed
  (Viridian) so milestone 1 is pure hands.

Reads the runtime collision grid from gBackupMapLayout (collision bits 0x0C00 of
each map-grid u16; 0 = walkable). Plans with BFS, executes ONE verified tile-step at
a time with coord feedback, LOUD-ABORT stuck detection (never silent-spin), and
hands off to the 5/5 battle engine on a wild encounter, resuming after.

Deferred (milestone 2+, flagged TODO): ledge-awareness (MB_JUMP_* one-way edges),
door/warp tiles beyond map-edge connections, free exploration / her-chosen targets.

Offline proof:  .venv\\Scripts\\python.exe pokemon_agent\\travel.py --prove
"""
import os
import sys
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import firered_ram as ram          # noqa: E402

GMAPHEADER = 0x02036DFC
BACKUP_LAYOUT = 0x03005040         # {s32 width, s32 height, u16 *map}
MAP_OFFSET = 7                     # save-coord + 7 = buffer index
SB1_MAP_GROUP, SB1_MAP_NUM = 0x04, 0x05
NUM_PRIMARY = 640                  # metatile ids < 640 use the primary tileset
# tall-grass / encounter behavior. 0x02 = MB_TALL_GRASS (the walkable grass that
# spawns wild battles), read from the u32 metatileAttributes low byte. Planning
# PREFERS grass-free tiles but falls back to crossing grass when there's no dry route
# (north Route 1 is a grass sea) - then the encounter handoff fights the battle.
GRASS_BEHAVIORS = {0x02}

# milestone-1 route endpoints (verified via map connections)
MAP_ROUTE1 = (3, 19)
MAP_VIRIDIAN = (3, 1)
MAP_PALLET = (3, 0)

DIR_KEY = {"N": "UP", "S": "DOWN", "W": "LEFT", "E": "RIGHT"}
DIR_DELTA = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


# ── map / coord helpers ──────────────────────────────────────────────────────
def map_id(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    return b.rd8(sb1 + SB1_MAP_GROUP), b.rd8(sb1 + SB1_MAP_NUM)


def coords(b):
    co = ram.read_player_coords(b)
    if co is None or co == (0, 0) or not (0 <= co[0] < 1000 and 0 <= co[1] < 1000):
        return None
    return co


# ── collision grid (cached per map load) ─────────────────────────────────────
class Grid:
    """Snapshot of the current map's collision, indexed in SAVE coordinates.
    walkable(sx,sy) = the map-grid collision bits are 0. Border tiles outside the
    buffer are treated as blocked (the map-edge crossing is handled separately)."""

    def __init__(self, bridge):
        b = bridge
        self.w = b.rd32(BACKUP_LAYOUT)
        self.h = b.rd32(BACKUP_LAYOUT + 4)
        mp = b.rd32(BACKUP_LAYOUT + 8)
        # tileset metatileAttributes for behavior (grass) reads; tolerate failure
        try:
            ml = b.rd32(GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
        except Exception:
            attr = None
        self.col, self.grass = {}, set()
        for by in range(self.h):
            row = mp + by * self.w * 2
            for bx in range(self.w):
                e = b.rd16(row + bx * 2)
                self.col[(bx, by)] = (e & 0x0C00) >> 10
                if attr:
                    mid = e & 0x3FF
                    base, idx = (attr[0], mid) if mid < NUM_PRIMARY else (attr[1], mid - NUM_PRIMARY)
                    # FireRed metatileAttributes is a u32 array (stride 4); behavior =
                    # low byte. (u16 stride read 0x00 for the real grass and silently
                    # missed it - the grass-sea bug.)
                    if (b.rd32(base + idx * 4) & 0xFF) in GRASS_BEHAVIORS:
                        self.grass.add((bx, by))
        # playable save-coord bounds (exclude the 7-tile border on every side)
        self.sx_lo, self.sx_hi = 0, self.w - 2 * MAP_OFFSET - 1
        self.sy_lo, self.sy_hi = 0, self.h - 2 * MAP_OFFSET - 1

    def walkable(self, sx, sy):
        bx, by = sx + MAP_OFFSET, sy + MAP_OFFSET
        if not (0 <= bx < self.w and 0 <= by < self.h):
            return False
        return self.col.get((bx, by), 1) == 0

    def walkable_safe(self, sx, sy):
        """walkable AND not tall grass - the encounter-free planning layer."""
        bx, by = sx + MAP_OFFSET, sy + MAP_OFFSET
        return self.walkable(sx, sy) and (bx, by) not in self.grass


def bfs(grid, start, goal_test, bound=None, walkable=None):
    """Breadth-first over 4-neighbours of walkable tiles. Returns the tile path
    (list incl. start+goal) or None. `bound` limits explored save-coords to
    (sx_lo,sy_lo,sx_hi,sy_hi). `walkable` overrides grid.walkable (e.g. the
    grass-free layer)."""
    walk = walkable or grid.walkable
    if bound is None:
        # default: the PLAYABLE rectangle only. Border tiles read collision-0 too,
        # so planning unbounded would leak into the 7-tile border on every side;
        # the map-edge crossing is handled as an explicit step, not via BFS.
        bound = (grid.sx_lo, grid.sy_lo, grid.sx_hi, grid.sy_hi)
    bx_lo, by_lo, bx_hi, by_hi = bound
    came = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if goal_test(cur):
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        cx, cy = cur
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (bx_lo <= nx <= bx_hi and by_lo <= ny <= by_hi):
                continue
            nxt = (nx, ny)
            if nxt in came or not walk(nx, ny):
                continue
            came[nxt] = cur
            q.append(nxt)
    return None


def direction(frm, to):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if dx == 1:
        return "E"
    if dx == -1:
        return "W"
    if dy == 1:
        return "S"
    if dy == -1:
        return "N"
    return None


# ── the executor: walk a planned path, one VERIFIED tile at a time ───────────
HOLD = 8
STUCK_LIMIT = 6          # consecutive no-progress steps -> LOUD ABORT
EXIT_TRIES = 5           # presses off the map edge before giving up loudly


class Traveler:
    """Deterministic feet. Plans BFS to the north exit each step, presses ONE tile,
    verifies the coord moved, hands off to the battle engine on encounters, and
    LOUD-ABORTS on no-progress (never silent-spin). Hooks are injected so the same
    engine serves headless tests and the live voiced session.

    battle_runner() : called on a wild encounter; runs the 5/5 battle engine to
                      completion and returns its outcome string. Owner returns to us.
    on_event(text)  : NEUTRAL game-event -> her voice (or print headless).
    """

    def __init__(self, bridge, battle_runner, render=None, on_event=None,
                 log=print, owner="agent"):
        self.b = bridge
        self.battle_runner = battle_runner
        self.render = render or (lambda: None)
        self.on_event = on_event or (lambda s: None)
        self.log = log
        self.owner = owner

    def _press(self, d):
        self.b.press(DIR_KEY[d], HOLD, HOLD, self.render, owner=self.owner)

    def _warmup_battle(self):
        """Advance the battle intro to a SETTLED action menu (phase==action-menu AND
        a valid 0-3 cursor) before the engine acts. The live handoff failed because
        the engine read the action cursor before the menu initialized (garbage ->
        'cursor desync' -> premature stuck-abort); from a settled menu the engine wins
        (verified on the captured wild battle). Battle engine stays untouched."""
        import pokemon_state as st
        for _ in range(240):
            if not st.in_battle(self.b):
                return
            ph = self.b.rd32(ram.GBATTLE_PHASE)
            cur = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if ph == ram.PHASE_ACTION_MENU and cur in (0, 1, 2, 3):
                return
            self.b.press("A", 6, 6, self.render, owner=self.owner)

    def travel(self, target_map=MAP_VIRIDIAN, max_steps=800):
        import pokemon_state as st     # local import: engine stays bot/battle-agnostic
        self.b.set_input_owner(self.owner)
        grid = Grid(self.b)
        cur_map = map_id(self.b)
        self.log(f"   [travel] start map={cur_map} coords={coords(self.b)} -> "
                 f"target map={target_map}")
        stuck = exit_tries = 0
        for step in range(max_steps):
            # 1) encounter -> hand the pad to the battle engine, then resume
            if st.in_battle(self.b):
                self.log("   [travel] ENCOUNTER -> warming up the menu, then handing off")
                self.on_event("a wild Pokemon jumped out - time to fight")
                self._warmup_battle()          # settle the menu so the engine doesn't desync
                outcome = self.battle_runner()
                self.log(f"   [travel] battle outcome={outcome}; resuming pathfind")
                if outcome == "loss":
                    self.on_event("we got knocked out... I need to regroup")
                    return "battle_loss"
                self.b.set_input_owner(self.owner)   # reclaim from the battle agent
                grid = Grid(self.b)
                stuck = 0
                continue

            # 2) map transition / arrival
            m = map_id(self.b)
            if m != cur_map:
                self.log(f"   [travel] MAP TRANSITION {cur_map} -> {m}")
                cur_map = m
                grid = Grid(self.b)
                stuck = exit_tries = 0
            if m == target_map:
                self.log(f"   [travel] ARRIVED at target map {m} coords={coords(self.b)}")
                self.on_event("made it to Viridian City")
                return "arrived"

            cur = coords(self.b)
            if cur is None:                         # mid-transition / cutscene
                for _ in range(8):
                    self.b.run_frame(); self.render()
                continue

            # 3) plan to the north exit row (y==0). Prefer a GRASS-FREE route (no wild
            # encounters); only if none exists fall back to crossing grass (the handoff
            # then catches any battle). Milestone 1's Route 1 has a grass-free path.
            path = bfs(grid, cur, lambda t: t[1] == 0, walkable=grid.walkable_safe)
            if not path:
                path = bfs(grid, cur, lambda t: t[1] == 0)
                if path and len(path) >= 2:
                    self.log(f"   [travel] no grass-free route from {cur}; crossing grass "
                             f"(encounters possible -> battle handoff)")
            if not path or len(path) < 2:
                if cur[1] == 0:                     # on the exit gap -> cross north
                    exit_tries += 1
                    if exit_tries > EXIT_TRIES:
                        self.log(f"   [travel] !! at exit {cur} but {exit_tries} north "
                                 f"presses didn't transition - ABORT LOUD")
                        self.on_event("I'm at the edge but I can't get through - stuck")
                        return "stuck"
                    self._press("N")
                    continue
                self.log(f"   [travel] !! NO PATH from {cur} to the north exit - ABORT LOUD")
                self.on_event("I can't find a way through here - I'm stuck")
                return "no_path"

            d = direction(cur, path[1])
            self._press(d)
            after = coords(self.b)
            if after == cur:                        # turned-not-stepped? try once more
                self._press(d)
                after = coords(self.b)
            if after == cur:
                stuck += 1
                if stuck >= STUCK_LIMIT:
                    self.log(f"   [travel] !! STUCK at {cur} ({stuck} no-progress steps, "
                             f"dir {d}) - ABORT LOUD (never silent-spin)")
                    self.on_event("ugh, I'm stuck - I can't get past this")
                    return "stuck"
            else:
                stuck = 0
                if step % 10 == 0:
                    self.log(f"   [travel] step {step}: {cur}->{after} map={m} "
                             f"(path {len(path)} to exit)")
        self.log(f"   [travel] hit step cap at {coords(self.b)} map={map_id(self.b)}")
        return "timeout"


# ── offline proof: does BFS reach the Route-1 north exit from her boot tile? ──
def prove():
    from bridge import Bridge
    rom = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
    boot = os.path.join(_HERE, "states", "after_pick_bulbasaur.state")
    b = Bridge(rom)
    with open(boot, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    start = coords(b)
    m = map_id(b)
    grid = Grid(b)
    print(f"   [prove] boot map={m} coords={start} buffer {grid.w}x{grid.h} "
          f"playable sx[0..{grid.sx_hi}] sy[0..{grid.sy_hi}]")
    if m != MAP_ROUTE1:
        print(f"   [prove] WARNING - expected Route 1 {MAP_ROUTE1}, got {m}")

    # flood from start (PLAYABLE area only) to find the minimum reachable y
    # (closest to the north edge / Viridian).
    def in_play(t):
        return grid.sx_lo <= t[0] <= grid.sx_hi and grid.sy_lo <= t[1] <= grid.sy_hi
    seen = {start}
    q = deque([start])
    min_y = start[1]
    while q:
        cx, cy = q.popleft()
        min_y = min(min_y, cy)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (cx + dx, cy + dy)
            if n not in seen and in_play(n) and grid.walkable(*n):
                seen.add(n)
                q.append(n)
    top_tiles = sorted(t for t in seen if t[1] == min_y)
    print(f"   [prove] reachable tiles={len(seen)}  northmost reachable y={min_y}  "
          f"exit-row tiles={top_tiles}")

    # BFS path to the northmost row
    path = bfs(grid, start, lambda t: t[1] == min_y)
    if path:
        print(f"   [prove] BFS path start->north-exit: LEN={len(path)} "
              f"end={path[-1]}")
        print(f"   [prove] first 8 steps: {path[:8]}")
        print(f"   [prove] last 6 steps:  {path[-6:]}")
    else:
        print("   [prove] BFS FOUND NO PATH to the north exit (!!)")

    # show the top rows of the collision map so the exit gap is visible
    print(f"   [prove] collision map top rows (y=0..{min_y+3}); .=walk #=block P=start:")
    for sy in range(0, min(min_y + 4, grid.sy_hi + 1)):
        line = []
        for sx in range(grid.sx_lo, grid.sx_hi + 1):
            if (sx, sy) == start:
                line.append("P")
            elif (sx, sy) in seen:
                line.append(".")
            else:
                line.append("#" if not grid.walkable(sx, sy) else " ")
        print(f"   y{sy:>2} {''.join(line)}")
    return path is not None


if __name__ == "__main__":
    if "--prove" in sys.argv:
        ok = prove()
        print(f"\n   [prove] RESULT: BFS-reaches-north-exit = {'YES' if ok else 'NO'}")
