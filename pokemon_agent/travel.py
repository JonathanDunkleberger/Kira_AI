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
import world_fingerprint as wf      # noqa: E402  (MICRO watchdog: tell a trainer from a plain NPC)

GMAPHEADER = 0x02036DFC
BACKUP_LAYOUT = 0x03005040         # {s32 width, s32 height, u16 *map}
MAP_OFFSET = 7                     # save-coord + 7 = buffer index
SB1_MAP_GROUP, SB1_MAP_NUM = 0x04, 0x05
NUM_PRIMARY = 640                  # metatile ids < 640 use the primary tileset
# B-3 POSITION-LOOP ESCAPE: if she's confined to <=POS_LOOP_DISTINCT tiles over the last
# POS_LOOP_WINDOW steps without arriving, she's spinning (warp/spinner) -> bail. Env-tunable.
POS_LOOP_WINDOW = int(os.getenv("POKEMON_POS_LOOP_WINDOW", "18"))
POS_LOOP_DISTINCT = int(os.getenv("POKEMON_POS_LOOP_DISTINCT", "3"))
# tall-grass / encounter behavior. 0x02 = MB_TALL_GRASS (the walkable grass that
# spawns wild battles), read from the u32 metatileAttributes low byte. Planning
# PREFERS grass-free tiles but falls back to crossing grass when there's no dry route
# (north Route 1 is a grass sea) - then the encounter handoff fights the battle.
GRASS_BEHAVIORS = {0x02}
# Surfable-water metatile behaviors (MB_POND_WATER 0x10 / MB_DEEP_WATER 0x12 /
# MB_OCEAN_WATER 0x15 — source: include/constants/metatile_behaviors.h). Classified
# into Grid.water for the Phase-2 field-move layer; does NOT affect walkability.
SURFABLE_WATER = {0x10, 0x12, 0x15}
# LEDGE / one-way jump behaviors (Gen-3 metatileAttributes low byte). You can only cross a
# ledge in its jump direction (hopping OVER it, landing the tile beyond); it blocks the reverse.
# So in pathfinding a ledge is a DIRECTED edge: from the tile on the approach side, moving in the
# jump direction, you land 2 tiles along. Confirmed on Route 3 (0x3b) + Route 4 (0x38/0x39/0x3b).
LEDGE_DIRS = {0x38: (1, 0), 0x39: (-1, 0), 0x3A: (0, -1), 0x3B: (0, 1)}   # E / W / N / S
# DIRECTIONAL IMPASSABLE behaviors (MB_IMPASSABLE_* 0x30-0x37). These are collision-0 (the grid
# reads them as plain floor) but block movement in a direction in-game - the CAVE CLIFF the
# collision grid is blind to (Mt Moon B2F: 0x32 MB_IMPASSABLE_NORTH bands seal levels apart, so
# grid-BFS plans straight through a wall and the traveler wedges). Confirmed in-game (2026-06-26):
# pressing into a 0x32 tile from the north is blocked. Two predicates per FRLG's paired collision
# funcs: the SOURCE tile blocks LEAVING in a dir; the DEST tile blocks ENTERING from the opposite.
IMPASS_LEAVE = {0x30: {(1, 0)}, 0x31: {(-1, 0)}, 0x32: {(0, -1)}, 0x33: {(0, 1)},
                0x34: {(0, -1), (1, 0)}, 0x35: {(0, -1), (-1, 0)},
                0x36: {(0, 1), (1, 0)}, 0x37: {(0, 1), (-1, 0)}}   # E/W/N/S + NE/NW/SE/SW
IMPASS_ENTER = {0x30: {(-1, 0)}, 0x31: {(1, 0)}, 0x32: {(0, 1)}, 0x33: {(0, -1)},
                0x34: {(0, 1), (-1, 0)}, 0x35: {(0, 1), (1, 0)},
                0x36: {(0, -1), (-1, 0)}, 0x37: {(0, -1), (1, 0)}}
IMPASS_BEHAVIORS = set(IMPASS_LEAVE)

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
RUN_STEP_CAP = 16   # max frames to hold a running step before giving up (a run tile is ~8f; cap covers
#                     a turn-only / blocked press whose coord never changes — caller retries those)
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
# PHASE 4 — DEAD-AIR FILLER that reacts to the PLACE (Lavender-Town-dread style): on a long silent
# walk she muses ABOUT WHERE SHE IS, not just generic "still walking". Keyed by map; a cave/interior
# (map group != 3) gets enclosed/gloom seeds, the open road gets road seeds. Her self colors the
# neutral seed (the voice floor still gates it). Extensible — add map ids as she reaches new places.
PLACE_MUSE = {
    (3, 21): ("the trees thin out along Route 3, trainers everywhere looking for a fight",
              "you take in Route 3 — your first real road east, the world opening up"),
    (3, 22): ("Route 4 stretches toward the mountains, the air drier here",),
}
CAVE_MUSE = (
    "the cave presses in close around you, every sound echoing off the rock",
    "it's dim and quiet down here, the path winding deeper into the dark",
    "you pick your way through the cavern, watching the ground in the gloom",
)
ROAD_MUSE = (
    "the route opens up ahead, grass swaying at the edges of the path",
    "you walk on under open sky, the next town somewhere past the horizon",
)


def _muse_seed(b, i):
    """A place-flavored dead-air seed for the current map (falls back to the generic road muses)."""
    try:
        mp = map_id(b)
        if mp in PLACE_MUSE:
            seeds = PLACE_MUSE[mp]
        elif mp[0] != 3:                 # not the overworld surface -> a cave/interior
            seeds = CAVE_MUSE
        else:
            seeds = ROAD_MUSE
        return seeds[i % len(seeds)]
    except Exception:
        return MUSE_SEEDS[i % len(MUSE_SEEDS)]


# ── map / coord helpers ──────────────────────────────────────────────────────
def map_id(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    return b.rd8(sb1 + SB1_MAP_GROUP), b.rd8(sb1 + SB1_MAP_NUM)


def coords(b):
    co = ram.read_player_coords(b)
    if co is None or co == (0, 0) or not (0 <= co[0] < 1000 and 0 <= co[1] < 1000):
        return None
    return co


# ── WARP TABLE (live map-header events) — the keystone of warp-routing ────────────────────────
# Reads the CURRENT map's warp events from gMapHeader.events.warps (ROM-pointed). VERIFIED 2026-06-28
# against the disasm: Route 4 live = (19,5)->MtMoon / (12,5)->PokéCenter / (32,5)->MtMoon, matching the
# disasm-sourced campaign constants exactly; coords are SAVE coords (no border offset); Route 3 has a
# null events ptr (0 warps). dest = (mapGroup, mapNum). This is what lets the world-model route THROUGH
# warps/dungeons, not just map edges (the Cerulean->Underground Path->Vermilion class).
_WARP_STRIDE = 8       # WarpEvent: x s16@0, y s16@2, elevation u8@4, warpId u8@5, mapNum u8@6, mapGroup u8@7


def _valid_ptr(p):
    return (0x02000000 <= p < 0x02400000) or (0x08000000 <= p < 0x0A000000)


def read_warps(b):
    """[(x, y), dest_map=(grp,num), warp_id] for every warp on the current map. [] if none / unreadable.
    Pure read (safe in a loop). (x,y) are SAVE coords — directly comparable to coords()/door tiles."""
    try:
        ev = b.rd32(GMAPHEADER + 0x04)            # MapHeader.events (ROM)
        if not _valid_ptr(ev):
            return []
        n = b.rd8(ev + 0x01)                      # warpCount
        arr = b.rd32(ev + 0x08)                   # WarpEvent* (ROM)
        if not _valid_ptr(arr) or not (0 < n <= 64):
            return []
        out = []
        for i in range(n):
            w = arr + i * _WARP_STRIDE
            out.append(((b.rds16(w), b.rds16(w + 2)),
                        (b.rd8(w + 7), b.rd8(w + 6)),       # dest (group, num)
                        b.rd8(w + 5)))                      # warp id
        return out
    except Exception:
        return []


def read_bg_events(b):
    """[(x, y), kind] for every BG event (sign / console / script tile — Bill's Cell-Separation machine
    class) on the current map. SAVE coords; kind 0-4 = script/sign (facing variants), 7 = hidden item.
    Same MapHeader.events read as read_warps (+0x03 bgEventCount, +0x10 BgEvent*, stride 12). Pure read."""
    try:
        ev = b.rd32(GMAPHEADER + 0x04)
        if not _valid_ptr(ev):
            return []
        n = b.rd8(ev + 0x03)                      # bgEventCount
        arr = b.rd32(ev + 0x10)                   # BgEvent* (ROM)
        if not _valid_ptr(arr) or not (0 < n <= 64):
            return []
        return [((b.rds16(e), b.rds16(e + 2)), b.rd8(e + 5))
                for i in range(n) for e in (arr + i * 12,)]
    except Exception:
        return []


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
        # `water` is ADDITIVE (Phase 2 field-moves): buffer-coords whose behavior is a
        # surfable-water tile. It does NOT change walkability/BFS — water stays collision-
        # blocked; field_moves.surf_edge_adjacent reads this set to know Surf is offerable.
        self.col, self.grass, self.ledge, self.impass, self.water = {}, set(), {}, {}, set()
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
                    elif bh in IMPASS_BEHAVIORS:
                        self.impass[(bx, by)] = bh               # buffer-coord -> directional wall
                    elif bh in SURFABLE_WATER:
                        self.water.add((bx, by))                 # buffer-coord -> surfable water
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

    def edge_open(self, sx, sy, dx, dy):
        """A normal 1-tile step from (sx,sy) by (dx,dy): False if a DIRECTIONAL impassable
        (MB_IMPASSABLE_*) on the SOURCE blocks leaving that way, or on the DEST blocks entering
        from the opposite side - the cave-cliff edges the 2-bit collision grid can't see."""
        a = self.impass.get((sx + MAP_OFFSET, sy + MAP_OFFSET))
        if a is not None and (dx, dy) in IMPASS_LEAVE.get(a, ()):
            return False
        d = self.impass.get((sx + dx + MAP_OFFSET, sy + dy + MAP_OFFSET))
        if d is not None and (dx, dy) in IMPASS_ENTER.get(d, ()):
            return False
        return True


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
                # DIRECTIONAL impassable (cave cliff): a normal step blocked by a one-way wall the
                # collision grid reads as floor. Only on 1-tile steps (ledge hops are their own edge).
                if hasattr(grid, "edge_open") and not grid.edge_open(cx, cy, dx, dy):
                    continue
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
                 log=print, owner="agent", beat=None, pause_check=None, stuck_check=None,
                 blocked_npcs=None, field_clear=None):
        self.b = bridge
        self.battle_runner = battle_runner
        # CAPABILITY-IN-HAND obstacle clearing (east run 1): campaign-provided callback
        # `field_clear(hm_key, face_key) -> 'used'|...` — when the chokepoint blocker is a cut
        # tree / boulder and the party KNOWS the HM, travel clears it in-leg instead of
        # remembering a hard block it could open.
        self.field_clear = field_clear
        # optional callback checked AFTER each battle; if it returns truthy, travel() yields
        # control to the caller with "need_heal" (the heal-when-low interrupt) so the campaign
        # can route back to the Center before resuming the leg.
        self.pause_check = pause_check or (lambda: False)
        # LAYER B cooperative cancel: polled at the top of every step. When the UNIVERSAL wall-clock
        # watchdog (fed from the live render hook) has latched a disengage, this returns truthy and
        # travel bails the leg LOUD ("stuck", reason=watchdog) so the wedge unwinds to the roam loop's
        # top-level recovery — no sub-tick loop keeps spinning below the per-tick ledger's sight.
        self.stuck_check = stuck_check or (lambda: False)
        # LAYER A — UNIFIED "NPCs that block me" memory, a {(map_id, body_tile)} set OWNED by the
        # campaign and shared by reference (so a plain NPC blocked once is routed AROUND on every later
        # leg — the per-call static_blocked was thrown away each tick, so the Slowbro chokepoint was
        # re-discovered + re-bumped forever). travel READS it to exclude those tiles from planning and
        # WRITES to it when its chokepoint gauntlet confirms a plain (non-trainer) NPC on the only gap.
        self.blocked_npcs = blocked_npcs if blocked_npcs is not None else set()
        self.render = render or (lambda: None)
        self.on_event = on_event or (lambda s: None)
        # PERFORMANCE BEAT hook: at notable moments the hands HOLD until her voice
        # for the moment lands (same clock as the battle engine's `pace`). No-op
        # headless. Boring connective tissue (walking) never calls this - it stays
        # brisk; only the savor-worthy beats gate. That contrast is the streamer feel.
        self.beat = beat or (lambda s: None)
        self.log = log
        self.owner = owner
        # PHASE 4 RUNNING SHOES: run by default (act like a player who has the shoes). Co-holds B with
        # the direction in _press. The GAME self-gates running (outdoors+shoes -> ~1.85x faster; cave /
        # building / no shoes -> B is inert, she just walks), and B-cohold lands EXACTLY one tile per
        # press (verified) so the BFS path-follow stays in sync. POKEMON_RUN=0 forces walking.
        self.run = os.getenv("POKEMON_RUN", "1") != "0"

    def _press(self, d):
        key = DIR_KEY[d]
        if not self.run:
            self.b.press(key, HOLD, HOLD, self.render, owner=self.owner)
            return
        # RUN, SMOOTH (Batch 4 Phase 1 fix). The old fixed 8-hold + 8-release STUTTERED: a run tile
        # finishes in ~8 frames (vs ~16 walking), so the 8-frame release became dead standing time ->
        # move-freeze-move-freeze. Instead, co-hold dir+B and hold ONLY until this tile actually
        # completes (coord changes), then release with a single settle frame so consecutive presses
        # GLIDE continuously like a held key (verified ~8 f/tile, no dead gap). Tile-atomic -> exactly
        # one tile per call so the BFS path-follow stays in sync; the cap covers a blocked/turn-only
        # press (no coord change), which the caller's `after == cur` retry already handles.
        c0 = coords(self.b)
        self.b.set_keys(key, "B", owner=self.owner)
        for _ in range(RUN_STEP_CAP):
            self.b.run_frame(); self.render()
            if coords(self.b) != c0:
                break
        self.b.release(owner=self.owner)
        self.b.run_frame(); self.render()       # 1-frame settle: clean coord read, no dead gap

    def _blocker_npc_check(self):
        """MICRO watchdog (increment 2): we A-interacted with a path blocker and it did NOT start a
        battle. Is it a PLAIN NPC rather than a trainer? Signature: a dialogue box is up, and a
        SECOND A leaves the WORLD identical (the shared world-fingerprint - coords/battle/party/...
        unchanged), i.e. the box just re-shows, nothing is progressing. If so, tap B to close it and
        report True so the caller marks the tile impassable and re-paths AROUND - instead of mashing
        A into the same NPC forever (the free-roam wedge this watchdog exists to kill). Returns False
        for 'no box / a trainer after all / the world moved' (let the normal logic continue)."""
        import pokemon_state as st
        fp0 = wf.fingerprint(self.b)
        if fp0 is None or not fp0.menu_or_dialogue:
            return False                               # no dialogue box up -> not the plain-NPC case
        self.b.press("A", HOLD, HOLD, self.render, owner=self.owner)
        for _ in range(16):
            self.b.run_frame(); self.render()
        if st.in_battle(self.b):
            return False                               # a delayed trainer trigger -> let the caller fight
        fp1 = wf.fingerprint(self.b)
        same = fp1 is not None and fp1 == fp0 and fp1.menu_or_dialogue
        self.log(f"   [travel-uwatch] {wf.brief(fp1)} blocker box re-check same={same} "
                 f"(limit {wf.STALL_N_BLOCKER})")
        if same:
            self.b.press("B", HOLD, HOLD, self.render, owner=self.owner)   # close the NPC's box
            for _ in range(12):
                self.b.run_frame(); self.render()
            return True
        return False

    def _fight_blocker(self):
        """A blocked path tile turned out to be a TRAINER (an A-interact started a battle). Warm up,
        fight to completion, reclaim input. Returns the battle outcome string. General gauntlet
        handling - the same primitive clears cave/gym chokepoint trainers (Rock Tunnel, Victory Road)."""
        import pokemon_state as st
        self.log("   [travel] blocker is a TRAINER - fighting through")
        self.beat("a trainer's blocking the path - bring it on")
        self._warmup_battle()
        outcome = self.battle_runner()
        self.log(f"   [travel] blocker battle outcome={outcome}; resuming pathfind")
        self.b.set_input_owner(self.owner)
        return outcome

    def _sweep_interact(self):
        """Face each of the 4 directions and press A, looking for an adjacent stationary blocker
        (a trainer on the only gap that BFS routed us NEXT TO but won't step onto, since it avoids
        NPC tiles). Returns True if a battle started (caller fights it)."""
        import pokemon_state as st
        if st.in_battle(self.b):
            return True
        for sd in ("N", "S", "E", "W"):
            self._press(sd)                                   # turn to face that side
            self.b.press("A", HOLD, HOLD, self.render, owner=self.owner)
            for _ in range(16):
                self.b.run_frame(); self.render()
            if st.in_battle(self.b):
                return True
        return False

    def _npc_tiles(self):
        """Live tiles occupied by NPC BODIES (object events), re-read every plan.

        We block ONLY the body tile - NOT the tile the NPC faces. A stationary LINE-OF-SIGHT
        trainer beside a corridor faces ACROSS it; blocking its facing tile walls off the whole
        walkable corridor the player must climb (the Mt Moon column-13 gauntlet: col-14 trainers
        facing west made bfs see no route up col 13 and wedge). The path SHOULD route through the
        sight line - stepping into it auto-triggers the battle, which we fight. The old facing-tile
        block (for mid-step wanderer coord-lag) cost us the entire trainer-corridor class; an
        occasional bump into a wanderer is cheap by comparison (the blocked-TTL reroute handles it)."""
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):                  # skip obj0 (the player)
            o = OB + i * SZ
            if not (self.b.rd8(o) & 1):
                continue
            x, y = self.b.rds16(o + 0x10) - MAP_OFFSET, self.b.rds16(o + 0x12) - MAP_OFFSET
            out.add((x, y))
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
               max_seconds=300, edge="north", avoid=None):
        """Walk to a connected map (cross its north edge, or its south edge if edge='south'
        - for the heal-return back to Viridian) OR, if arrive_coord is set,
        BFS to that specific tile on the current map and stop there (for warp doors /
        gym-interior nav). Same robust NPC-aware, grass-aware, battle-handoff stepping.
        WALL-CLOCK budget (max_seconds): a leg that grinds far past it (e.g. a battle-
        heavy grass maze) LOUD-ABORTS instead of silently running for hours."""
        import pokemon_state as st     # local import: engine stays bot/battle-agnostic
        # AVOID: tiles the PLAN must not step on while walking to the goal. For cave warp-to-warp
        # nav these are the OTHER warp tiles on this floor - walking THROUGH a warp triggers it and
        # teleports us off the intended path (the Mt Moon "(5,6) trap": pathing to a far warp crossed
        # a nearer warp and fired it). Treated exactly like a permanent static block.
        avoid = frozenset(avoid or ())
        self.b.set_input_owner(self.owner)
        grid = Grid(self.b)
        cur_map = map_id(self.b)
        t0 = time.time()
        self.log(f"   [travel] start map={cur_map} coords={coords(self.b)} -> "
                 f"{'coord ' + str(arrive_coord) if arrive_coord else 'map ' + str(target_map)} "
                 f"(budget {max_seconds:.0f}s)")
        stuck = exit_tries = no_path = 0
        # TRAVEL-LAYER PROGRESS GUARD (increment 3.5): fingerprint the world across no-path RETRIES.
        # If it stays identical (player can't move, world unchanged) for TRAVEL_STALL_RETRIES, STOP
        # spinning and surface a loud STRUCTURED failure UP to the roam loop (where the watchdog is) —
        # instead of re-trying an impossible target for minutes. last_fail_reason tags WHY (npc_block
        # / no_route) for the oracle + logs; it's additive, the return strings stay backward-compatible.
        fp_stall = 0
        last_fp = None
        npc_block_seen = False     # set at the no_path==4 probe: did an NPC-allowing path exist?
        self.last_fail_reason = ""
        blocked = {}              # tile -> step it was blocked (TTL-aged dynamic obstacles:
        BLOCK_TTL = 12            # NPCs, movement-blocked-but-collision-walkable tiles)
        # STATIC obstacles: a step that fails on a NON-NPC tile is an un-encoded boulder/rock/wall
        # (collision-walkable in the grid but impassable in-game - e.g. Mt Moon's rock formations).
        # Those NEVER move, so mark them PERMANENTLY impassable (not TTL-aged) - else BLOCK_TTL keeps
        # expiring and BFS re-routes back INTO the boulder, wedging forever (the Mt Moon B2F bounce).
        static_blocked = set()
        fail_count = {}          # tile -> times a step into it failed (3+ on a non-NPC = static)
        # goal: a specific tile (coord mode) or the chosen exit edge (edge-crossing mode). N/S cross
        # a ROW (north=row 0, south=bottom row sy_hi); E/W cross a COLUMN (east=right col sx_hi,
        # west=left col sx_lo). exit_axis picks which coordinate the goal/cross test reads.
        exit_dir = EDGE_DIR.get(edge, "N")
        exit_axis = EDGE_AXIS.get(edge, 1)
        exit_line = {"north": 0, "south": grid.sy_hi,
                     "east": grid.sx_hi, "west": grid.sx_lo}.get(edge, 0)
        # EDGE-CROSS PREDICATE: cross when AT OR PAST the edge, not exactly ON it. E/W map connections
        # load a DEEP overlap of the neighbour's tiles into the border (Pewter east reaches x=55, well
        # past the sx_hi=48 edge — disasm offset=10 -> Route 3). An exact "x==sx_hi" goal TRAPS the
        # agent: the instant it steps into the overlap the goal unsatisfies and BFS drags it back west
        # (the "1-tile prison" / collision paradox). ">= edge" lets it keep walking east THROUGH the
        # overlap until the map flips. N/S keep firing the exit press at the edge exactly as before.
        _PASS = {"north": lambda v: v <= 0, "south": lambda v: v >= grid.sy_hi,
                 "east": lambda v: v >= grid.sx_hi, "west": lambda v: v <= grid.sx_lo}
        exit_cmp = _PASS.get(edge, lambda v: v == exit_line)
        # CONNECTION BAND (E/W only): an edge crosses ONLY where the neighbour's tiles overlap into
        # the border — the rows where the tile JUST PAST the edge is walkable. BFS would otherwise
        # route to the NEAREST edge tile, which on a partial connection (Pewter<->Route 3, offset=10)
        # is OUTSIDE the band: the press hits a hard wall and never transitions (the west-cross at
        # row 9 failed for exactly this). Gate the goal to band rows so we approach a crossable tile.
        # N/S are untouched (they flip AT the edge with no overlap, so band stays None -> any edge tile).
        perp_axis = 1 - exit_axis
        band = None
        if edge in ("east", "west"):
            past = (grid.sx_hi + 1) if edge == "east" else (grid.sx_lo - 1)
            cand = {p for p in range(grid.sy_lo, grid.sy_hi + 1) if grid.walkable(past, p)}
            band = cand or None        # no overlap detected -> fall back to any edge tile
            if band:
                self.log(f"   [travel] {edge} connection band rows: {sorted(band)}")
        def _edge_goal(t):
            return exit_cmp(t[exit_axis]) and (band is None or t[perp_axis] in band)
        goal = ((lambda t: t == arrive_coord) if arrive_coord is not None else _edge_goal)
        _muse_t = [time.time()]   # last-voiced clock for the travel-muse dead-air filler
        _muse_i = [0]
        # B-3 — POSITION-LOOP ESCAPE: a warp/spinner reads as normal floor to the collision grid and can
        # cycle her between a few tiles forever (coords keep CHANGING, so the no-path/wall-clock guards
        # are slow to catch it). Track a sliding window of recent tiles; if she's been confined to <=3
        # distinct tiles over the whole window without arriving, she's looping -> bail LOUD fast (~5s)
        # so free_roam re-routes. NOT a puzzle solver (deterministic warps are a separate scoped item) —
        # this is the can't-loop-forever floor, the overworld sibling of the battle flee / dialogue cycle floors.
        _pos_window = deque(maxlen=POS_LOOP_WINDOW)
        _bstuck = [0]   # 2026-07-06 BATTLE-LOOP BREAKER: consecutive unresolved 'stuck' battles
        for step in range(max_steps):
            # LAYER B cooperative cancel: the universal watchdog latched a disengage -> bail this leg
            # LOUD so the roam loop runs its top-level recovery (don't keep spinning a sub-tick loop).
            if self.stuck_check():
                self.log("   [travel] !! WATCHDOG disengage requested — bailing this leg LOUD")
                self.last_fail_reason = "watchdog"
                return "stuck"
            _pc = coords(self.b)
            if _pc is not None and not st.in_battle(self.b):
                _pos_window.append(tuple(_pc))
                if len(_pos_window) == _pos_window.maxlen and len(set(_pos_window)) <= POS_LOOP_DISTINCT:
                    self.log(f"   [travel] !! POSITION-LOOP — confined to {len(set(_pos_window))} tile(s) "
                             f"over {_pos_window.maxlen} steps at {_pc} map={map_id(self.b)} "
                             f"(warp/spinner?) -> ABORT LOUD so free_roam re-routes")
                    self.on_event("I keep getting moved in circles here — let me find another way around.")
                    self.last_fail_reason = "position_loop"
                    return "stuck"
            # TRAVEL MUSE: fill a long silent WALK (no encounter/arrival to react to) with a neutral
            # "taking it in" beat so she isn't dead-air for 30-50s. Never during a battle; gated by
            # the voice floor like any beat. Her self colors the neutral seed.
            if (MUSE_GAP_S and not st.in_battle(self.b)
                    and (time.time() - _muse_t[0]) > MUSE_GAP_S and coords(self.b) is not None):
                self.beat(_muse_seed(self.b, _muse_i[0]))   # PHASE 4: place-flavored dead-air filler
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
                self.last_fail_reason = "budget"
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
                # 2026-07-06 BATTLE-LOOP BREAKER: a 'stuck' outcome with the battle STILL open means
                # the runner cannot resolve THIS battle — re-detecting it as a fresh encounter forever
                # was the south_run1 ×27 spin (an abandoned no-balls catch battle). Three consecutive
                # -> abort the leg LOUD; roam's recovery (no-move pruning / hard recovery) owns it.
                if outcome == "stuck" and st.in_battle(self.b):
                    _bstuck[0] += 1
                    if _bstuck[0] >= 3:
                        self.log("   [travel] !! BATTLE-LOOP BREAKER: 3 consecutive unresolved 'stuck' "
                                 "battles — aborting the leg LOUD (no infinite re-entry)")
                        self.on_event("something's off with this fight — I need to step back and reset")
                        self.last_fail_reason = "battle_loop"
                        return "stuck"
                else:
                    _bstuck[0] = 0
                self.b.set_input_owner(self.owner)   # reclaim from the battle agent
                grid = Grid(self.b)
                stuck = 0
                # CLEAR accumulated obstacle memory: a battle disrupts position/NPC layout, and the
                # step-failures logged mid-combat (a wild triggered as we pressed) wrongly mark good
                # floor as permanent static walls - over a battle-heavy cave that accumulates until
                # the path is severed (the (12,22) wedge). Real walls are modelled by collision +
                # directional impassables now, so dropping these is safe; a true boulder re-marks.
                blocked.clear(); fail_count.clear(); static_blocked.clear()
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
            # LAYER A: tiles on THIS map held by a plain NPC we've already confirmed blocks the only gap
            # (persists across legs/ticks via the shared set) -> plan AROUND them, never back into them.
            blocked_here = {t for (m, t) in self.blocked_npcs if m == cur_map}
            def free(sx, sy):
                return ((sx, sy) not in npc and (sx, sy) not in static_blocked
                        and (sx, sy) not in avoid and (sx, sy) not in blocked_here
                        and blocked.get((sx, sy), -10 ** 9) <= cutoff)
            path = bfs(grid, cur, goal,
                       walkable=lambda sx, sy: grid.walkable_safe(sx, sy) and free(sx, sy))
            if not path:
                path = bfs(grid, cur, goal,
                           walkable=lambda sx, sy: grid.walkable(sx, sy) and free(sx, sy))
            if not path or len(path) < 2:
                if arrive_coord is not None and cur == arrive_coord:
                    self.log(f"   [travel] reached target coord {arrive_coord}")
                    return "arrived"
                if arrive_coord is None and exit_cmp(cur[exit_axis]):   # at/past the edge -> cross
                    before = cur
                    self._press(exit_dir)
                    if coords(self.b) == before:      # press didn't move us -> a genuinely wasted try
                        exit_tries += 1
                        if exit_tries > EXIT_TRIES:
                            self.log(f"   [travel] !! at exit {cur} but {exit_tries} {exit_dir} "
                                     f"presses didn't transition - ABORT LOUD")
                            self.on_event("I'm at the edge but I can't get through - stuck")
                            self.last_fail_reason = "exit_blocked"
                            return "stuck"
                    else:
                        exit_tries = 0                # stepped deeper into the overlap -> still progressing
                    continue
                # no path right now - most likely an NPC is standing on the only gap.
                # WAIT (bounded) for a wanderer to step off, re-reading NPC positions each
                # retry, before giving up loud. ~25 x 24f ~= 10s of patience.
                no_path += 1
                # PROGRESS GUARD: has the WORLD changed at all across these no-path retries? (The
                # player is stuck, so coords/party/etc sit still; this counts CONSECUTIVE frozen
                # retries.) Checked below, AFTER the no_path==4 gauntlet probe has had its shot.
                _fp = wf.fingerprint(self.b)
                fp_stall = (fp_stall + 1) if (last_fp is not None and _fp == last_fp) else 0
                last_fp = _fp
                # CHOKEPOINT TRAINER: BFS avoids NPC tiles, so a trainer standing on the only gap
                # yields no clean path - and it can be 2+ tiles ahead (the adjacent sweep can't reach
                # it). Re-plan ALLOWING npc tiles: if THAT path exists, an NPC/trainer is the sole
                # blocker. Walk along that path toward it and, when we reach the tile before the NPC,
                # interact (trainer -> battle, the gauntlet primitive; LoS trainers auto-trigger as we
                # approach). This makes the run CONTINUOUS - fight, then immediately flow on.
                if no_path == 4:
                    npc_path = bfs(grid, cur, goal, walkable=lambda sx, sy: grid.walkable(sx, sy)
                                   and (sx, sy) not in static_blocked and (sx, sy) not in avoid
                                   and (sx, sy) not in blocked_here)   # LAYER A: skip known plain blockers
                    nplist = sorted(self._npc_tiles())
                    npc_block_seen = bool(npc_path and len(npc_path) >= 2)   # NPC on the gap vs real wall
                    if npc_path and len(npc_path) >= 2:
                        blk = next((t for t in npc_path if t in npc), None)
                        self.log(f"   [travel] no clean path from {cur}; NPC-allowing path EXISTS "
                                 f"(len {len(npc_path)}), blocker NPC tile={blk}, npcs nearby={nplist} "
                                 f"-> approaching to interact/trigger")
                        for _ in range(14):                       # walk up to the blocker
                            cur2 = coords(self.b)
                            ap = bfs(grid, cur2, goal, walkable=lambda sx, sy: grid.walkable(sx, sy)
                                     and (sx, sy) not in static_blocked and (sx, sy) not in avoid
                                     and (sx, sy) not in blocked_here)
                            if st.in_battle(self.b) or not ap or len(ap) < 2:
                                break
                            nx = ap[1]; d2 = direction(cur2, nx)
                            self._press(d2)                       # face/step toward the blocker
                            if st.in_battle(self.b):              # LoS trainer triggered mid-approach
                                break
                            if coords(self.b) == cur2:            # couldn't step -> blocker adjacent: talk
                                self.b.press("A", HOLD, HOLD, self.render, owner=self.owner)
                                for _ in range(16):
                                    self.b.run_frame(); self.render()
                                if st.in_battle(self.b):
                                    break
                        if st.in_battle(self.b):
                            if self._fight_blocker() == "loss":
                                self.on_event("knocked out fighting through - I need to regroup")
                                return "battle_loss"
                            grid = Grid(self.b)
                            blocked.clear(); fail_count.clear(); static_blocked.clear()
                            no_path = stuck = fp_stall = 0; last_fp = None
                            # heal-when-low after a BLOCKER (gauntlet) trainer too — mirrors the
                            # post-encounter yield at the top. Without this a lone starter walks a
                            # trainer gauntlet (Route 3) with no heal between fights and faints.
                            if self.pause_check():
                                self.log("   [travel] post-blocker PAUSE (heal-when-low) - yielding to caller")
                                return "need_heal"
                            continue
                        # 2026-07-06 HONESTY FIX (STATE §7 prescription, built): a cut-tree/boulder on the
                        # gap is a FIELD OBSTACLE, not a "stuck NPC" — it answers A with a box ("This
                        # tree looks like it can be CUT down!") so the plain-NPC check misclassified it
                        # (south_run1: the Cerulean south tree read as no_route_npc_blocked). Classify
                        # by the object's graphicsId (the game's own mechanism) and surface DISTINCTLY.
                        if blk is not None:
                            try:
                                import field_moves as _fmv
                                _fobjs = {ob["coord"]: ob for ob in _fmv.scan_field_objects(
                                    self.b, {_fmv.GFX_CUT_TREE, _fmv.GFX_BOULDER})}
                            except Exception:
                                _fobjs = {}
                            if blk in _fobjs:
                                _is_tree = _fobjs[blk]["gfx"] == _fmv.GFX_CUT_TREE
                                # CAPABILITY IN HAND (east run 1, the GENERAL fix the badge-3
                                # cascade owed): she KNOWS the HM — clear the obstacle right
                                # here and continue the leg. Only the gym-door probe auto-cut
                                # before; Route 9's mouth tree stalled her at Cerulean (35,33).
                                try:
                                    _hm = "cut" if _is_tree else "strength"
                                    _cur0 = coords(self.b)
                                    _face = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT",
                                             (1, 0): "RIGHT"}.get((blk[0] - _cur0[0],
                                                                   blk[1] - _cur0[1]))
                                    if (self.field_clear is not None and _face
                                            and _fmv.can_use(self.b, _hm)):
                                        self.on_event("that little tree's in my way — good thing "
                                                      "I carry Cut now. TIMBER!" if _is_tree else
                                                      "a boulder — STRENGTH time. heave!")
                                        if self.field_clear(_hm, _face) == "used":
                                            self.log(f"   [travel] FIELD OBSTACLE {blk} on "
                                                     f"{cur_map} CLEARED with {_hm} — "
                                                     f"continuing the leg")
                                            no_path = stuck = fp_stall = 0
                                            last_fp = None
                                            continue
                                except Exception as _ce:
                                    self.log(f"   [travel] obstacle-clear attempt failed: {_ce!r}")
                                # REMEMBER the obstacle as a hard block (until the HM is owned) and
                                # REPLAN in-leg first: the NPC-allowing BFS prefers the SHORTEST path,
                                # so a near tree beats a far real detour (Cerulean: the garden corridor
                                # lost to the tree every replan — the crossed→tree→crossed loop). With
                                # the tile in shared block memory the next plan finds the long way.
                                self.blocked_npcs.add((cur_map, blk))
                                blocked_here = {t for (m, t) in self.blocked_npcs if m == cur_map}
                                reroute = bfs(grid, coords(self.b), goal,
                                              walkable=lambda sx, sy: grid.walkable(sx, sy)
                                              and (sx, sy) not in static_blocked and (sx, sy) not in avoid
                                              and (sx, sy) not in blocked_here)
                                if reroute and len(reroute) >= 2:
                                    self.log(f"   [travel] FIELD OBSTACLE {blk} on {cur_map} "
                                             f"({'CUT tree' if _is_tree else 'STRENGTH boulder'}) — "
                                             f"remembered as a hard block; a LONGER route exists, rerouting")
                                    no_path = stuck = fp_stall = 0; last_fp = None
                                    continue
                                self.last_fail_reason = "hm_blocked:cut" if _is_tree else "hm_blocked:strength"
                                self.on_event("that's one of those small trees — the kind you can CUT "
                                              "down with the right HM. I can't clear it yet, so this "
                                              "way's closed for now — I need another route."
                                              if _is_tree else
                                              "a boulder's plugging the gap — that needs STRENGTH. "
                                              "Another route for now.")
                                self.log(f"   [travel] !! FIELD OBSTACLE on the only gap: {blk} on "
                                         f"{cur_map} is a {'CUT tree' if _is_tree else 'STRENGTH boulder'}"
                                         f" -> no_route_hm_blocked (honest; not a stuck NPC)")
                                return "no_route_hm_blocked"
                        # LAYER A — PLAIN NPC on the only gap (we approached, NO battle started). This is
                        # the live Slowbro wedge: re-bumping its dialogue forever. Confirm it's a re-showing
                        # box (not a delayed trainer) and ADD it to the SHARED block memory so this leg AND
                        # every later one route AROUND it — instead of the per-call static_blocked that was
                        # thrown away each tick. If the goal is then unreachable, surface a DISTINCT failure
                        # so the oracle picks a different objective rather than re-issuing into the same door.
                        if blk is not None and self._blocker_npc_check():
                            self.blocked_npcs.add((cur_map, blk))
                            blocked_here = {t for (m, t) in self.blocked_npcs if m == cur_map}
                            self.log(f"   [travel] chokepoint blocker {blk} on {cur_map} is a PLAIN NPC "
                                     f"(dialogue, no battle) -> added to shared block memory "
                                     f"({len(self.blocked_npcs)} total) + routing around (LAYER A)")
                            reroute = bfs(grid, coords(self.b), goal,
                                          walkable=lambda sx, sy: grid.walkable(sx, sy)
                                          and (sx, sy) not in static_blocked and (sx, sy) not in avoid
                                          and (sx, sy) not in blocked_here)
                            if not (reroute and len(reroute) >= 2):
                                self.last_fail_reason = "npc_blocked"
                                self.on_event("the only way through is blocked by someone who won't budge "
                                              "— I'll have to find another way or do something else.")
                                self.log(f"   [travel] !! NO ROUTE around plain-NPC blocker {blk} on "
                                         f"{cur_map} -> no_route_npc_blocked (oracle should pick differently)")
                                return "no_route_npc_blocked"
                            no_path = stuck = fp_stall = 0; last_fp = None
                            continue
                    else:
                        self.log(f"   [travel] no clean path from {cur} AND no NPC-allowing path "
                                 f"(npcs nearby={nplist}) - genuine wall/zone gap, will time out")
                # TRAVEL WEDGE GUARD: world frozen across TRAVEL_STALL_RETRIES no-path retries -> this
                # target is impossible from here (unreachable coord / a plain NPC that won't move /
                # a real wall). STOP spinning and surface UP to roam with a REASON, so the macro ledger
                # + oracle pick a different action — instead of re-trying the same dead target for
                # minutes. Fires AFTER the no_path==4 gauntlet probe, so a real trainer-on-the-gap has
                # already started a battle (which moves the fingerprint and resets this).
                if fp_stall >= wf.TRAVEL_STALL_RETRIES:
                    self.last_fail_reason = "npc_block" if npc_block_seen else "no_route"
                    self.log(f"   [travel] !! TRAVEL WEDGE: identical fp x{fp_stall} retries at {cur} "
                             f"map={map_id(self.b)} (reason={self.last_fail_reason}) -> returning to roam "
                             f"LOUD (no inner spin)")
                    self.on_event("I can't find a way through here — let me rethink this")
                    return "no_path"
                if no_path > 25:
                    self.log(f"   [travel] !! NO PATH from {cur} to the exit after waiting "
                             f"~10s for NPCs to clear - ABORT LOUD")
                    self.on_event("there's someone blocking the way and they won't move - stuck")
                    self.last_fail_reason = self.last_fail_reason or "no_route"
                    return "no_path"
                if no_path == 1:
                    self.log(f"   [travel] path blocked at {cur} (NPC on the gap?) - waiting")
                # WATCHDOG TRUCE (2026-07-05): this wait is DELIBERATE stillness (letting a walker NPC
                # step off the gap) — the POSITION-LOOP guard reading it as a spinner-wedge aborted the
                # Route-25 door approach mid-gauntlet (two watchdogs fighting, the dueling-recovery
                # class). Clear the confinement window while waiting; the guard still catches REAL
                # spinner/warp loops (those cycle position involuntarily with no waiting branch active).
                _pos_window.clear()
                for _ in range(24):
                    self.b.run_frame(); self.render()
                continue
            no_path = fp_stall = 0; last_fp = None     # a path exists -> progressing; clear the guard

            d = direction(cur, path[1])
            nxt = path[1]
            self._press(d)
            after = coords(self.b)
            if st.in_battle(self.b):                # the step walked into a wild encounter:
                continue                            # NOT a movement failure - loop top fights it
            if after == cur:                        # turned-not-stepped? try once more
                self._press(d)
                after = coords(self.b)
            if st.in_battle(self.b):
                continue
            if after == cur:
                blocked[nxt] = step                 # dynamic block -> re-plan around it
                fail_count[nxt] = fail_count.get(nxt, 0) + 1
                # BLOCKER vs TERRAIN: a tile the grid says is walkable but we can't step onto is
                # either un-encoded terrain (boulder) OR a stationary NPC/TRAINER the object-event
                # read missed (Mt Moon B2F: a corridor trainer that never wanders -> we'd wait
                # forever and false-mark it a wall). The failed press left us FACING it, so INTERACT
                # (A): a trainer starts a battle we fight through (the cave/gym gauntlet - general
                # for Rock Tunnel / Victory Road); a plain NPC's dialogue advances harmlessly. Only
                # if interaction does NOT yield a battle do we fall through to terrain-marking.
                # A wanderer would have stepped off by the 2nd failure, so a tile STILL failing is a
                # STATIONARY blocker - a trainer (detected NPC or not) or terrain. Interact to tell
                # them apart, regardless of npc-detection (the last Mt Moon trainers ARE detected, so
                # gating on 'not in npc' wrongly skipped them and we waited forever).
                if (grid.walkable(*nxt) and fail_count[nxt] == 2 and not st.in_battle(self.b)):
                    self._press(d)                              # ensure we face the blocker
                    self.b.press("A", HOLD, HOLD, self.render, owner=self.owner)
                    for _ in range(20):
                        self.b.run_frame(); self.render()
                    if st.in_battle(self.b):
                        if self._fight_blocker() == "loss":
                            self.on_event("knocked out fighting through - I need to regroup")
                            return "battle_loss"
                        grid = Grid(self.b)
                        blocked.clear(); fail_count.clear(); static_blocked.clear()
                        stuck = 0
                        if self.pause_check():
                            self.log("   [travel] post-blocker-battle PAUSE (heal-when-low) - yielding")
                            return "need_heal"
                        continue
                    # not a trainer: is it a PLAIN NPC blocking the gap? (MICRO watchdog) close its
                    # box + mark the tile impassable NOW so we re-path AROUND, instead of A-mashing
                    # the same NPC across the remaining stuck loop (the free-roam wedge).
                    if self._blocker_npc_check():
                        static_blocked.add(nxt)
                        self.log(f"   [travel] blocker at {nxt} is a PLAIN NPC (dialogue, no battle) - "
                                 f"closed it + marking impassable, routing around (micro-watchdog)")
                        stuck = 0
                        continue
                if nxt not in npc and fail_count[nxt] >= 3:   # a NON-NPC tile that fails REPEATEDLY
                    static_blocked.add(nxt)                   # = static obstacle (boulder/un-encoded
                    self.log(f"   [travel] static obstacle at {nxt} (failed 3x, not an NPC) - "
                             f"marking impassable, routing around")   # wall) -> permanent block
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
