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
import time
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
# LEDGE / one-way jump behaviors (Gen-3 metatileAttributes low byte). You can only cross a
# ledge in its jump direction (hopping OVER it, landing the tile beyond); it blocks the reverse.
# So in pathfinding a ledge is a DIRECTED edge: from the tile on the approach side, moving in the
# jump direction, you land 2 tiles along. Confirmed on Route 3 (0x3b) + Route 4 (0x38/0x39/0x3b).
LEDGE_DIRS = {0x38: (1, 0), 0x39: (-1, 0), 0x3A: (0, -1), 0x3B: (0, 1)}   # E / W / N / S

# milestone-1 route endpoints (verified via map connections)
MAP_ROUTE1 = (3, 19)
MAP_VIRIDIAN = (3, 1)
MAP_PALLET = (3, 0)
# Pewter (3,2) connections — discovered live from the map connection table (recon_map_connections.py):
#   SOUTH offset=12 -> Route 2 (3,20);   EAST offset=10 -> Route 3 (3,21).
# NOTE (recon): the naive "reach the east column" cross WEDGES at Pewter — the east connection only
# transitions within its offset band; walking east outside it (e.g. row 21) is game-blocked despite
# collision=0/elev=3/behavior=0x00. East crossing needs connection-offset-aware row targeting (TODO).
MAP_PEWTER = (3, 2)
MAP_ROUTE2 = (3, 20)
MAP_ROUTE3 = (3, 21)

DIR_KEY = {"N": "UP", "S": "DOWN", "W": "LEFT", "E": "RIGHT"}
DIR_DELTA = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
# edge-crossing geometry: which axis (0=x col, 1=y row) and which extreme line each map-edge uses,
# plus the press that steps OFF that edge. East/West are the Pewter->Route 3 unlock (additive — the
# proven north/south chain is unchanged). exit_line is resolved per-map from the live Grid.
EDGE_DIR = {"north": "N", "south": "S", "east": "E", "west": "W"}
EDGE_AXIS = {"north": 1, "south": 1, "east": 0, "west": 0}      # N/S cross a row (y); E/W a column (x)

# TRAVEL MUSE (Pokémon-mode environmental chatter): on a long DEAD-AIR walk (no encounter/arrival to
# react to) she'd go silent for 30-50s. This fires a NEUTRAL "taking in the surroundings" beat after
# this many idle seconds so her self colors the journey — gap-filler only, never over a real beat.
# Pokémon-harness layer; feeds the existing voice seam, edits no core. 0 disables.
MUSE_GAP_S = float(os.getenv("POKEMON_TRAVEL_MUSE_GAP_S", "14.0"))
MUSE_SEEDS = (
    "you keep walking, taking in the area around you",
    "you press on along the route, eyes on what's ahead",
    "you make your way through, the place quiet around you",
)


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
        self.col, self.grass, self.ledge = {}, set(), {}
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
                    bh = b.rd32(base + idx * 4) & 0xFF
                    if bh in GRASS_BEHAVIORS:
                        self.grass.add((bx, by))
                    elif bh in LEDGE_DIRS:
                        self.ledge[(bx, by)] = LEDGE_DIRS[bh]    # buffer-coord -> jump (dx,dy)
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

    def ledge_dir(self, sx, sy):
        """If (sx,sy) is a ledge tile, the (dx,dy) you can only jump it in; else None. Save coords."""
        return self.ledge.get((sx + MAP_OFFSET, sy + MAP_OFFSET))


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
            # LEDGE HOP: if the adjacent tile is a ledge whose one-way jump direction matches the
            # move, we hop OVER it and land 2 tiles along (the ledge itself is never a standing
            # tile). Crossing the ledge the wrong way isn't offered -> the directed one-way edge.
            if grid.ledge_dir(cx + dx, cy + dy) == (dx, dy):
                nx, ny = cx + 2 * dx, cy + 2 * dy
            else:
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
    # handles ±1 (normal step) AND ±2 (a ledge hop, where the BFS edge skips the ledge tile and
    # lands 2 along): one D-pad press in this direction makes the game hop the ledge.
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if dx > 0:
        return "E"
    if dx < 0:
        return "W"
    if dy > 0:
        return "S"
    if dy < 0:
        return "N"
    return None


# ── the executor: walk a planned path, one VERIFIED tile at a time ───────────
HOLD = 8
STUCK_LIMIT = 16         # consecutive no-progress steps -> LOUD ABORT (patient: towns
                         # have NPC clusters that need waiting/rerouting, not a fast give-up)
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
                 log=print, owner="agent", beat=None, pause_check=None):
        self.b = bridge
        self.battle_runner = battle_runner
        # optional callback checked AFTER each battle; if it returns truthy, travel() yields
        # control to the caller with "need_heal" (the heal-when-low interrupt) so the campaign
        # can route back to the Center before resuming the leg.
        self.pause_check = pause_check or (lambda: False)
        self.render = render or (lambda: None)
        self.on_event = on_event or (lambda s: None)
        # PERFORMANCE BEAT hook: at notable moments the hands HOLD until her voice
        # for the moment lands (same clock as the battle engine's `pace`). No-op
        # headless. Boring connective tissue (walking) never calls this - it stays
        # brisk; only the savor-worthy beats gate. That contrast is the streamer feel.
        self.beat = beat or (lambda s: None)
        self.log = log
        self.owner = owner

    def _press(self, d):
        self.b.press(DIR_KEY[d], HOLD, HOLD, self.render, owner=self.owner)

    def _npc_tiles(self):
        """Live tiles occupied by NPCs (object events), re-read every plan so newly
        position-activated and wandering NPCs are accounted for. We block the NPC tile
        AND the tile it's stepping toward (movementDirection) - mid-step an NPC occupies
        two tiles, and the coord read lags by one, so blocking the facing tile too
        prevents walking into a wanderer the read hasn't caught up to yet."""
        OB, SZ = 0x02036E38, 0x24
        DELTA = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}   # down/up/left/right
        out = set()
        for i in range(1, 16):                  # skip obj0 (the player)
            o = OB + i * SZ
            if not (self.b.rd8(o) & 1):
                continue
            x, y = self.b.rds16(o + 0x10) - MAP_OFFSET, self.b.rds16(o + 0x12) - MAP_OFFSET
            out.add((x, y))
            mv = self.b.rd8(o + 0x18) & 0x0F     # current facing/movement nibble
            dx, dy = DELTA.get(mv, (0, 0))
            out.add((x + dx, y + dy))
        return out

    def _warmup_battle(self):
        """Let the encounter solidly BEGIN, then hand to the engine - which does its own
        intro advance (run()->_reach_first_menu). DO NOT mash A here on GBATTLE_PHASE: that
        register is a free-running FRAME COUNTER, not a phase (it hits 0x580 once every ~16
        frames during animations too), so the old loop mashed up to 240 A-presses on a bogus
        condition and DESYNCED the menu before the engine started - wedging Forest battles.
        Just settle a beat in-battle; the engine owns the intro from here."""
        import pokemon_state as st
        for _ in range(20):
            if not st.in_battle(self.b):
                return
            self.b.run_frame()
            self.render()

    def travel(self, target_map=MAP_VIRIDIAN, max_steps=800, arrive_coord=None,
               max_seconds=300, edge="north"):
        """Walk to a connected map (cross its north edge, or its south edge if edge='south'
        - for the heal-return back to Viridian) OR, if arrive_coord is set,
        BFS to that specific tile on the current map and stop there (for warp doors /
        gym-interior nav). Same robust NPC-aware, grass-aware, battle-handoff stepping.
        WALL-CLOCK budget (max_seconds): a leg that grinds far past it (e.g. a battle-
        heavy grass maze) LOUD-ABORTS instead of silently running for hours."""
        import pokemon_state as st     # local import: engine stays bot/battle-agnostic
        self.b.set_input_owner(self.owner)
        grid = Grid(self.b)
        cur_map = map_id(self.b)
        t0 = time.time()
        self.log(f"   [travel] start map={cur_map} coords={coords(self.b)} -> "
                 f"{'coord ' + str(arrive_coord) if arrive_coord else 'map ' + str(target_map)} "
                 f"(budget {max_seconds:.0f}s)")
        stuck = exit_tries = no_path = 0
        blocked = {}              # tile -> step it was blocked (TTL-aged dynamic obstacles:
        BLOCK_TTL = 12            # NPCs, movement-blocked-but-collision-walkable tiles)
        # goal: a specific tile (coord mode) or the chosen exit edge (edge-crossing mode). N/S cross
        # a ROW (north=row 0, south=bottom row sy_hi); E/W cross a COLUMN (east=right col sx_hi,
        # west=left col sx_lo). exit_axis picks which coordinate the goal/cross test reads.
        exit_dir = EDGE_DIR.get(edge, "N")
        exit_axis = EDGE_AXIS.get(edge, 1)
        exit_line = {"north": 0, "south": grid.sy_hi,
                     "east": grid.sx_hi, "west": grid.sx_lo}.get(edge, 0)
        goal = ((lambda t: t == arrive_coord) if arrive_coord is not None
                else (lambda t: t[exit_axis] == exit_line))
        _muse_t = [time.time()]   # last-voiced clock for the travel-muse dead-air filler
        _muse_i = [0]
        for step in range(max_steps):
            # TRAVEL MUSE: fill a long silent WALK (no encounter/arrival to react to) with a neutral
            # "taking it in" beat so she isn't dead-air for 30-50s. Never during a battle; gated by
            # the voice floor like any beat. Her self colors the neutral seed.
            if (MUSE_GAP_S and not st.in_battle(self.b)
                    and (time.time() - _muse_t[0]) > MUSE_GAP_S and coords(self.b) is not None):
                self.beat(MUSE_SEEDS[_muse_i[0] % len(MUSE_SEEDS)])
                _muse_i[0] += 1
                _muse_t[0] = time.time()
            # WALL-CLOCK WATCHDOG (loud): a leg should not run for hours. If we blow the
            # budget we're grinding (battle-heavy grass maze) or wedged - ABORT LOUD with
            # exactly where we are, so a slow run TELLS us instead of spinning silently.
            elapsed = time.time() - t0
            if elapsed > max_seconds:
                self.log(f"   [travel] !! WALL-CLOCK BUDGET BLOWN ({elapsed:.0f}s > "
                         f"{max_seconds:.0f}s) at {coords(self.b)} map={map_id(self.b)} "
                         f"step={step} - ABORT LOUD (grinding / wedged, not progressing fast)")
                self.on_event("this is taking forever - I'm stuck grinding, let me regroup")
                return "stuck"
            if step % 40 == 0 and step:        # periodic LOUD heartbeat (always visible)
                self.log(f"   [travel] HEARTBEAT step={step} map={map_id(self.b)} "
                         f"coords={coords(self.b)} elapsed={elapsed:.0f}s")
            # 1) encounter. Beat ORDER for the streamer feel: react ON-TIME to the
            # surprise (intro "Wild X appeared!" still on screen) BEFORE we mash the
            # intro, THEN warm up the menu (stable start) and hand to the battle engine
            # (which names the foe + fights).
            if st.in_battle(self.b):
                self.log("   [travel] ENCOUNTER -> on-time beat, warm up, hand off")
                self.beat("a wild Pokemon leaps out at you")   # gated: her surprise lands now
                _muse_t[0] = time.time()                        # a real beat resets the muse clock
                self._warmup_battle()                           # then settle (proven stable)
                outcome = self.battle_runner()
                self.log(f"   [travel] battle outcome={outcome}; resuming pathfind")
                if outcome == "loss":
                    self.on_event("we got knocked out... I need to regroup")
                    return "battle_loss"
                self.b.set_input_owner(self.owner)   # reclaim from the battle agent
                grid = Grid(self.b)
                stuck = 0
                if self.pause_check():                # heal-when-low: yield to the caller
                    self.log("   [travel] post-battle PAUSE (heal-when-low) - yielding to caller")
                    return "need_heal"
                continue

            # 2) map transition / arrival
            m = map_id(self.b)
            if m != cur_map:
                self.log(f"   [travel] MAP TRANSITION {cur_map} -> {m}")
                cur_map = m
                grid = Grid(self.b)
                stuck = exit_tries = 0
            if arrive_coord is None and target_map is not None and m == target_map:
                self.log(f"   [travel] ARRIVED at target map {m} coords={coords(self.b)}")
                # NEW-AREA beat: gate so her arrival line lands as the new map comes up.
                self.beat("made it to a new area")
                return "arrived"

            cur = coords(self.b)
            if cur is None:                         # mid-transition / cutscene
                for _ in range(8):
                    self.b.run_frame(); self.render()
                continue

            # 3) plan to the north exit row (y==0), routing AROUND dynamic blocks. A step
            # that fails despite the tile being collision-walkable (a wandering NPC, a
            # movement-blocked tile) marks that tile blocked (TTL-aged so NPCs get retried)
            # and we RE-PLAN around it - the path-exists-but-executor-stalls fix. Prefer a
            # grass-free route; fall back to crossing grass (handoff catches the battle).
            cutoff = step - BLOCK_TTL
            npc = self._npc_tiles()            # live NPC tiles, re-read every plan
            def free(sx, sy):
                return (sx, sy) not in npc and blocked.get((sx, sy), -10 ** 9) <= cutoff
            path = bfs(grid, cur, goal,
                       walkable=lambda sx, sy: grid.walkable_safe(sx, sy) and free(sx, sy))
            if not path:
                path = bfs(grid, cur, goal,
                           walkable=lambda sx, sy: grid.walkable(sx, sy) and free(sx, sy))
            if not path or len(path) < 2:
                if arrive_coord is not None and cur == arrive_coord:
                    self.log(f"   [travel] reached target coord {arrive_coord}")
                    return "arrived"
                if arrive_coord is None and cur[exit_axis] == exit_line:   # on the exit gap -> cross
                    exit_tries += 1
                    if exit_tries > EXIT_TRIES:
                        self.log(f"   [travel] !! at exit {cur} but {exit_tries} {exit_dir} "
                                 f"presses didn't transition - ABORT LOUD")
                        self.on_event("I'm at the edge but I can't get through - stuck")
                        return "stuck"
                    self._press(exit_dir)
                    continue
                # no path right now - most likely an NPC is standing on the only gap.
                # WAIT (bounded) for a wanderer to step off, re-reading NPC positions each
                # retry, before giving up loud. ~25 x 24f ~= 10s of patience.
                no_path += 1
                if no_path > 25:
                    self.log(f"   [travel] !! NO PATH from {cur} to the exit after waiting "
                             f"~10s for NPCs to clear - ABORT LOUD")
                    self.on_event("there's someone blocking the way and they won't move - stuck")
                    return "no_path"
                if no_path == 1:
                    self.log(f"   [travel] path blocked at {cur} (NPC on the gap?) - waiting")
                for _ in range(24):
                    self.b.run_frame(); self.render()
                continue
            no_path = 0

            d = direction(cur, path[1])
            nxt = path[1]
            self._press(d)
            after = coords(self.b)
            if after == cur:                        # turned-not-stepped? try once more
                self._press(d)
                after = coords(self.b)
            if after == cur:
                blocked[nxt] = step                 # dynamic block -> re-plan around it
                stuck += 1
                if stuck % 4 == 0:                  # likely a wandering NPC: wait for it
                    # The failed step LEFT US FACING the blocker. A chokepoint trainer
                    # standing on the only gap isn't in walk-past line-of-sight, so it must
                    # be TALKED to (A) to start the battle; a path-blocking NPC's dialogue
                    # advances harmlessly. (Until the bridge binding fix on 2026-06-25 a
                    # phantom A was emitted on every release and did this BY ACCIDENT; with
                    # input now clean, interacting with a blocking trainer/NPC is explicit.)
                    self.b.press("A", HOLD, HOLD, self.render, owner=self.owner)
                    if st.in_battle(self.b):        # trainer triggered -> let the loop hand off
                        continue
                    for _ in range(24):             # else wait for a wanderer to step aside
                        self.b.run_frame(); self.render()
                if stuck >= STUCK_LIMIT:
                    self.log(f"   [travel] !! STUCK at {cur} ({stuck} blocked dirs, last "
                             f"{d}->{nxt}) - ABORT LOUD (never silent-spin)")
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
