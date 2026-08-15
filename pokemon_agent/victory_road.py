"""victory_road.py — THE ROAD TO THE PLATEAU (post-badge-8 -> Indigo), in-loop (night shift).

A FAITHFUL port of the proven recon_victory.py vehicle into an in-loop module driven by the live `camp`
bridge, so the endgame push can call it as ONE decision (the same shape as giovanni_gym / blaine_gym /
mansion_strike / seafoam_strike). FireRed coords/puzzles are isolated here (rule 14 portability debt: the
Route-22/23 + Victory-Road boulder-puzzle fact tables, swap per game).

Ground truth (pret; the puzzle sequences are recon_victory's offline-solved, elevation-aware constants —
vr1f/2f/3f_probe*.py; proven on the champion climb, banked indigo_reach_g):
- Phase 1: Viridian west edge -> Route 22. GARY trigger (col 33, armed by badge 8) fires a forced scene ->
  his strongest pre-E4 team; handle_interrupts owns it. A loss whiteouts to Viridian -> the loop re-crosses.
- Phase 2: R22 -> the north-entrance gate (28,0) -> Route 23.
- Phase 3: Route 23 south leg northward through the six badge-guard lockall scenes (all msgbox drains — she
  holds all 8) -> the Victory Road door (5,28) -> VR 1F.
- Phase 4: every VR-floor barrier opens ONLY by pushing a boulder onto its STRENGTH switch. The three-floor
  boulder chain (1F/2F/3F, incl. the 3F row-19-boulder reveal via the (34,18) hole drop) is the hand-solved
  VRnF_PUZZLE constants below. NEVER push boulder (35,13) — it seals the (37,10) pocket.
- Phase 5: VR 2F east pocket -> R23 north -> Indigo Plateau exterior (3,9) -> heal at the League center.

WHITEOUT-TOLERANT: progress RATCHETS (Gary's var, the gauntlet scenes and every VR switch var persist in
the save), so the dispatch loop keys on the CURRENT map every iteration and skips already-open barriers —
a mid-VR whiteout costs a re-cross, never solved ground.

Resume-safe: already at Indigo -> 'reached_indigo'; anywhere on the road/floors -> the map dispatch picks
up from there. OFF-CORRIDOR overworld boots (2026-08-10 LIVE: post-Zapdos Route 10) hop the learned world
graph to Viridian first — doors on outdoor maps are entrances, never the exit vehicle. run_strike returns:
  'reached_indigo' — at the Indigo Plateau exterior, healed. (Success — the E4 vehicle takes it from here.)
  'battle_loss'    — a fight loss loop the caller's recovery should own (rare; the loop self-recovers most).
  'stuck'          — a wedge cap hit (puzzle/warp/edge) or the deadline. Surfaces LOUD.

EQ teach (Phase 0) is DEFAULT OFF (POKEMON_TEACH_EQ) — recon_victory NS12 proved it did net harm on a
non-EQ kit (flaky TM-case actuation + it could forget Razor Leaf, Venusaur's only Grass STAB). Razor Leaf
x2 carries VR's Water/Rock/Ground; enable only with a clean droppable slot + a fixed teacher.
"""
import os
import time

import boulder_puzzle as bpz
import travel as tv
import pokemon_state as st
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box

# ── FireRed Route-22/23 + Victory-Road fact table (game-knowledge layer; rule 14 portability debt) ──────
VIRIDIAN = (3, 1)
R22 = (3, 41)
R23 = (3, 42)
GATE = (28, 0)                       # Route22 North-Entrance gatehouse (group 28)
VR1F, VR2F, VR3F = (1, 39), (1, 40), (1, 41)
INDIGO = (3, 9)
VR1F_DOOR = (5, 28)                  # Route 23 west cave mouth (south entrance)
# y<=30 was used as "north of VR / Indigo side". That is WRONG on the stoop:
# the door is (5, 28), so (5, 29)/(5, 30) are the SOUTH approach, not past VR.
# Live 2026-08-13: cy<=30 sent (5, 30) into to-indigo-band; BFS has no overworld
# path around the mountain → watchdog freeze. The east exit is (18, 28).
FLAG_BADGE_EARTH = 0x827             # badge 8 — the prereq; also the strike's own preflight guard
FLAG_STR_ACTIVE = 0x805             # Strength armed for the session
TM26_ITEM, MOVE_EQ = 314, 89
TEACH_EQ = os.getenv("POKEMON_TEACH_EQ") == "1"    # default OFF — see module docstring

KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN"}
DIRN_OF = {"south": 1, "north": 2, "west": 3, "east": 4}
MOVE_CUT, DIGLETT_SP, KADABRA_SP = 15, 50, 64      # Cut escort constants (2026-08-10)
# Jonny 2026-08-13: Diglett was Cut escort only. Kadabra L19 / Lapras L26 are
# not the E4 plan. Steamroll = Blastoise + the three birds. Empty seats stay empty.
DECLARED_SIX = (9, 146, 144, 145)                 # Blastoise, Moltres, Articuno, Zapdos
# CREDITS MARCH (2026-08-13): ONE corridor to Viridian. Do not invent a second.
# Proven live walls:
#   * Route 7 ↔ Celadon ping-pong (graph BFS has no Bike edge, so Celadon only
#     neighbors Route 7 — soak 20260811_121822)
#   * Cycling Road gate: "No pedestrians are allowed on CYCLING ROAD!" (no Bike)
#   * Route 4 → Route 3 is a bogus learned edge (real door is Mt. Moon)
# She HAS walked Saffron → Vermilion → Diglett's Cave → Route 2 this timeline.
# Celadon/Route 7 JOIN east onto that corridor. Never hop west onto Route 16.
CREDITS_MARCH = (
    (3, 3),    # Cerulean
    (3, 23),   # Route 5
    (17, 1),   # Underground Path (Route 5 ↔ Saffron)
    (3, 10),   # Saffron
    (3, 24),   # Route 6
    (3, 5),    # Vermilion
    (3, 29),   # Route 11 (Diglett's Cave south mouth)
    (1, 38),   # Diglett's Cave vestibule (from Route 11)
    (1, 37),   # Diglett's Cave long floor
    (1, 36),   # Diglett's Cave (Route 2 mouth)
    (3, 20),   # Route 2 — north half must go THROUGH Viridian Forest (pret)
    VIRIDIAN,  # (3, 1)
)
# Side doors onto the march. Celadon leaves EAST. Route 16 (bike-gate strand) backs out.
# Forest chain: pret/pokefirered Route2 + gate + ViridianForest map.json (2026-08-13).
FOREST = (1, 0)          # gMapGroup_Dungeons[0] ViridianForest
GATE_N = (15, 3)         # IndoorRoute2[3] Route2_ViridianForest_NorthEntrance
GATE_S = (15, 0)         # IndoorRoute2[0] Route2_ViridianForest_SouthEntrance
R2_HOUSE = (15, 1)       # IndoorRoute2[1] Route2_House (Diglett-side cottage)
R2_LEDGE = (15, 2)       # IndoorRoute2[2] Route2_EastBuilding (ledge skip)
CREDITS_JOIN = {
    (3, 6): (3, 25),    # Celadon → Route 7 (east — NEVER Route 16)
    (3, 25): (19, 0),   # Route 7 → Underground → Saffron
    (19, 0): (3, 10),   # Underground → Saffron (joins CREDITS_MARCH)
    (3, 34): (3, 6),    # Route 16 bike-gate strand → back to Celadon
    GATE_N: FOREST,     # north gate south mats → forest
    FOREST: GATE_S,     # forest y=62 mats → south gate
    GATE_S: (3, 20),    # south gate south mats → Route 2 (5,51)
}

# ── the hand-solved, elevation-aware boulder-push puzzles (recon_victory constants, verbatim) ───────────
VR1F_PUZZLE = [("strength", (7, 18)),
               ("push", (7, 18), "DOWN", 1),
               ("push", (7, 19), "RIGHT", 4),
               # stand (11,20) = the entrance arrow tile (0x65: warps on DOWN only; this push presses UP)
               ("push", (11, 19), "UP", 1, ((11, 20),)),
               ("push", (11, 18), "RIGHT", 1),
               ("push", (12, 18), "UP", 1),
               ("push", (12, 17), "RIGHT", 7),
               ("push", (19, 17), "UP", 2),
               ("push", (19, 15), "RIGHT", 1),
               ("push", (20, 15), "DOWN", 1)]      # lands (20,16) = the switch
VR2F_PUZZLE1 = [("strength", (6, 17)),
                ("push", (6, 17), "DOWN", 1),
                ("push", (6, 18), "LEFT", 2),
                ("push", (4, 18), "DOWN", 1),
                ("push", (4, 19), "LEFT", 2)]      # lands (2,19) = switch 1
FLAG_2F_BOULDER_HIDDEN = 0x058     # FLAG_HIDE_VICTORY_ROAD_2F_BOULDER: set = not yet dropped from 3F
VR3F_SWITCH_PUZZLE = [("strength", (32, 5)),
                      ("push", (32, 5), "UP", 2),
                      ("push", (32, 3), "LEFT", 21),
                      ("push", (11, 3), "DOWN", 1),
                      ("push", (11, 4), "LEFT", 5),
                      ("push", (6, 4), "DOWN", 3),
                      ("push", (6, 7), "RIGHT", 1)]  # lands (7,7) = the 3F switch


class VictoryRoad:
    def __init__(self, camp, log, dbg_dir=None):
        self.camp = camp
        self.b = camp.b
        self.log = log
        self.dbg = dbg_dir
        if dbg_dir:
            try:
                os.makedirs(dbg_dir, exist_ok=True)
            except Exception:
                self.dbg = None
        self.n_battles = 0
        self.wedges = {}
        self.deadline = time.time() + 3600
        self._last_hop_src = None            # map-id we just hopped FROM (hysteresis guard)

    # ── snap / battle / dialogue drains ────────────────────────────────────────────────────────────────
    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def fight(self):
        self.n_battles += 1
        return self.camp.battle_runner()

    def fight_open(self):
        return ram.valid_ewram_ptr(self.b.rd32(ram.GBATTLE_RES_PTR))

    def drain(self, max_n=40, key="B"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_n):
            if self.fight_open():
                return
            if dd_box(b):
                stable = 0
                b.press(key, 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    def handle_interrupts(self):
        if self.fight_open():
            self.fight()
            self.drain()
            return True
        if dd_box(self.b):
            self.drain()
            return True
        return False

    def settle(self, n=90):
        for _ in range(n):
            self.b.run_frame()

    # ── water/edge machinery (seafoam/giovanni verbatim; VR adds boulders to the obstacle set) ───────────
    def water_save(self, g):
        return {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for bx, by in g.water}

    def sea_ok(self, g, wset):
        def ok(sx, sy):
            bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
            if not (0 <= bx < g.w and 0 <= by < g.h):
                return False
            if g.col.get((bx, by), 1) != 0:
                return False
            return g.walkable(sx, sy) or (sx, sy) in wset
        return ok

    def on_water(self):
        g = tv.Grid(self.b)
        return tuple(tv.coords(self.b) or (99, 99)) in self.water_save(g)

    def mount(self, face_key):
        b, camp = self.b, self.camp
        if self.on_water():
            return True
        for attempt in range(4):
            b.press(face_key, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
            self.drain(key="A")
            for _ in range(240):
                b.run_frame()
                if self.on_water():
                    break
            if self.fight_open():
                self.fight()
                self.drain()
            if self.on_water():
                self.log(f"   [surf] MOUNTED at {tv.coords(b)} (attempt {attempt + 1})")
                return True
        self.log(f"!! [surf] mount failed at {tv.coords(b)} facing {face_key}")
        return False

    def live_npc_tiles(self):
        b = self.b
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                     b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def live_boulders(self):
        return [ob["coord"] for ob in fm.scan_field_objects(self.b, {fm.GFX_BOULDER})]

    def step_to(self, tile, wset=None):
        b, camp = self.b, self.camp
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
        if wset is None:
            wset = self.water_save(tv.Grid(b))
        if tile in wset and cur not in wset and not self.on_water():
            return self.mount(key)
        for _attempt in range(3):
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(50):
                b.run_frame()
                if tuple(tv.coords(b) or ()) == tile:
                    break
            if self.fight_open() or dd_box(b):
                return True
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return False

    def sea_walk(self, goal_test, label, tries=14, avoid=(), allow=()):
        b = self.b
        budget = tries
        while budget > 0:
            budget -= 1
            if self.handle_interrupts():
                budget += 1
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            g = tv.Grid(b)
            wset = self.water_save(g)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = self.live_npc_tiles() | {tuple(o[0]) for o in
                                            tv.read_object_templates(b)
                                            if o[2] and o[1] != fm.GFX_BOULDER}
            npcs |= {tuple(t) for t in self.live_boulders()}
            ok0 = self.sea_ok(g, wset)
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: ok0(sx, sy) and (sx, sy) not in wts
                       and (sx, sy) not in npcs and (sx, sy) not in avoid)
            self.log(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                self.log(f"   [{label}] no path from {cur}")
                self.snap(f"nopath_{label[:12]}_{cur[0]}_{cur[1]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if self.handle_interrupts():
                    budget += 1
                    break
                if not self.step_to(tuple(t), wset):
                    self.log(f"   [{label}] step blocked {tuple(tv.coords(b) or ())} -> {tuple(t)} "
                             f"(npcs {sorted(self.live_npc_tiles())[:6]})")
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

    @staticmethod
    def _s32(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    def connections(self):
        b = self.b
        out = {}
        hdr = b.rd32(tv.GMAPHEADER + 0x0C)
        if not hdr or hdr < 0x02000000:
            return out
        n = self._s32(b.rd32(hdr))
        arr = b.rd32(hdr + 4)
        if not (0 < n < 16) or arr < 0x02000000:
            return out
        for i in range(n):
            c = arr + i * 0xC
            out.setdefault(b.rd8(c), []).append(self._s32(b.rd32(c + 4)))
        return out

    def cross_edge(self, direction, label):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        conns = self.connections().get(DIRN_OF[direction])
        if not conns:
            self.log(f"   [{label}] no {direction} connection on {m0} — skip")
            return False
        off = conns[0]
        key = {"south": "DOWN", "north": "UP", "west": "LEFT", "east": "RIGHT"}[direction]
        for round_ in range(6):
            g = tv.Grid(b)
            wset = self.water_save(g)
            ok0 = self.sea_ok(g, wset)
            if direction in ("south", "north"):
                extreme = g.sy_hi if direction == "south" else 0
                band = [(x, extreme) for x in range(max(g.sx_lo, off), g.sx_hi + 1)
                        if ok0(x, extreme)]
            else:
                extreme = g.sx_hi if direction == "east" else 0
                band = [(extreme, y) for y in range(max(g.sy_lo, off), g.sy_hi + 1)
                        if ok0(extreme, y)]
            if not band:
                self.log(f"!! [{label}] no {direction}-edge band on {m0}")
                self.settle(120)
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not self.sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                self.settle(120)
                continue
            for _hold in range(4):
                cur2 = tuple(tv.coords(b) or (0, 0))
                nxt = {"south": (cur2[0], cur2[1] + 1), "north": (cur2[0], cur2[1] - 1),
                       "west": (cur2[0] - 1, cur2[1]), "east": (cur2[0] + 1, cur2[1])}[direction]
                g2 = tv.Grid(b)
                w2 = self.water_save(g2)
                if nxt in w2 and cur2 not in w2:
                    if not self.mount(key):
                        break
                    continue
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(90):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if self.handle_interrupts():
                    continue
                if tuple(tv.map_id(b)) != m0:
                    self.settle(120)
                    self.log(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} "
                             f"@ {tv.coords(b)}")
                    return True
        self.log(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def tile_behavior(self, t):
        b = self.b
        try:
            ml = b.rd32(tv.GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
            bw = b.rd32(tv.BACKUP_LAYOUT)
            mp0 = b.rd32(tv.BACKUP_LAYOUT + 8)
            mid = b.rd16(mp0 + ((t[1] + tv.MAP_OFFSET) * bw
                                + (t[0] + tv.MAP_OFFSET)) * 2) & 0x3FF
            base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1],
                                                                     mid - tv.NUM_PRIMARY)
            return b.rd32(base + idx * 4) & 0xFF
        except Exception:
            return 0

    def go_warp(self, tile, dest, label, avoid=()):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = self.tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]
        for attempt in range(4):
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach",
                                     avoid=avoid):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile and not arrow:
                # standing ON a plain (beh-0) warp tile that never fires on step — the real exit is walking
                # OUT through the door frame beyond it (the gate-mat class). Try outward-first.
                g0 = tv.Grid(b)
                order = []
                if tile[1] >= g0.sy_hi - 2:
                    order.append("DOWN")
                if tile[1] <= 2:
                    order.append("UP")
                if tile[0] >= g0.sx_hi - 2:
                    order.append("RIGHT")
                if tile[0] <= 2:
                    order.append("LEFT")
                order += [k for k in ("DOWN", "UP", "LEFT", "RIGHT") if k not in order]
                for k2 in order:
                    b.press(k2, 26, 10, camp.render, owner="agent")
                    for _ in range(120):
                        b.run_frame()
                        if tuple(tv.map_id(b)) != m0:
                            break
                    if tuple(tv.map_id(b)) != m0:
                        break
                    if tuple(tv.coords(b) or ()) != tile:      # stepped off — go back
                        self.sea_walk(lambda c, t=tile: c == t, f"{label}-remount",
                                      allow=(tile,))
                if self.handle_interrupts():
                    continue
                if tuple(tv.map_id(b)) == dest:
                    self.settle(180)
                    self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (door walk-out)")
                    return True
                if tuple(tv.map_id(b)) != m0:
                    self.log(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                    self.settle(180)
                    return False
                continue
            key = (arrow if arrow and cur == tile
                   else KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1])) or arrow)
            if key is None:
                continue
            for _press in range(4 if arrow else 1):
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) == dest:
                self.settle(180)
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (beh {hex(beh)})")
                return True
            if tuple(tv.map_id(b)) != m0:
                self.log(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                self.settle(180)
                return False
        self.log(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)}, beh {hex(beh)})")
        self.snap(f"warpfail_{label[:16]}")
        return False

    def nearest_boulder(self, approx, radius=8):
        for _attempt in range(3):
            bs = [t for t in self.live_boulders()
                  if abs(t[0] - approx[0]) + abs(t[1] - approx[1]) <= radius]
            if bs:
                return min(bs, key=lambda t: abs(t[0] - approx[0]) + abs(t[1] - approx[1]))
            cur = tuple(tv.coords(self.b) or (0, 0))
            if abs(cur[0] - approx[0]) + abs(cur[1] - approx[1]) <= 3:
                return None
            if not self.sea_walk(lambda c, a=approx: abs(c[0] - a[0]) + abs(c[1] - a[1]) <= 3,
                                 "boulder-approach"):
                return None
        return None

    def ensure_strength(self, approx):
        b, camp = self.b, self.camp
        if fm.read_flag(b, FLAG_STR_ACTIVE):
            return True
        bl = self.nearest_boulder(approx)
        if bl is None:
            self.log(f"!! [strength] no live boulder near {approx} on {tv.map_id(b)}")
            return False
        for attempt in range(3):
            nbs = [(bl[0] + dx, bl[1] + dy) for dx, dy in
                   ((0, 1), (0, -1), (1, 0), (-1, 0))]
            if not self.sea_walk(lambda c, s=set(nbs): c in s, "str-approach"):
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((bl[0] - cur[0], bl[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            self.settle(40)
            self.drain(key="A")
            self.settle(60)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                self.log(f"   [strength] ARMED (flag 0x805) at {tv.coords(b)}")
                return True
        self.log(f"!! [strength] flag 0x805 never set (boulder {bl})")
        self.snap("strength_fail")
        return False

    def push(self, approx, key, n, allow=()):
        b, camp = self.b, self.camp
        d = DELTA[key]
        for i in range(n):
            bl = self.nearest_boulder(approx)
            if bl is None:
                self.log(f"!! [push] boulder near {approx} vanished (i={i})")
                return False
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not self.sea_walk(lambda c, s=stand: c == s, f"push-approach{i}",
                                 avoid={tuple(bl)}, allow=allow):
                self.log(f"!! [push] can't reach {stand} to push {bl} {key}")
                return False
            moved = False
            for _try in range(4):
                if self.handle_interrupts():
                    continue
                b.press(key, 40, 10, camp.render, owner="agent")
                self.settle(70)
                b2l = self.nearest_boulder((bl[0] + d[0], bl[1] + d[1]))
                if b2l != bl:
                    moved = True
                    break
            if not moved:
                self.log(f"!! [push] {bl} would not move {key} (player {tv.coords(b)})")
                self.snap(f"push_fail_{bl[0]}_{bl[1]}")
                return False
            approx = (bl[0] + d[0], bl[1] + d[1])
            self.log(f"   [push] {bl} -> {approx} ({key}, {i + 1}/{n})")
            self.settle(30)
        return True

    def _vr_checkpoint(self, reason):
        """Milestone bank between pushes (2026-08-05 addendum): a mid-puzzle recovery resumes
        seconds back on this floor with the chain intact (savestates keep pushed boulders)."""
        try:
            self.camp._bank_milestone(f"vr-{reason}")
        except Exception as e:
            self.log(f"   [ckpt] vr milestone '{reason}' skipped: {e}")
        return True

    def run_puzzle(self, ops, barrier_tile, label):
        # 2026-08-05 #3 (the Mt. Ember loop, applied here BEFORE the E4 run stalls the same
        # way): the op table is the same hand-solved recon_victory ground truth, but executed
        # by the shared idempotent chain engine — a same-map retry resumes mid-chain from live
        # boulder positions instead of re-running op 0 into an over-push; a whiteout re-entry
        # (template board) re-derives from the start. No exit mid-puzzle: VR rooms define no
        # reset (the whiteout ratchet is the only board reset that ever helps here).
        room = bpz.room_from_ops(tuple(tv.map_id(self.b)), label, ops, ckpt_every=4)
        if not bpz.solve_room(self, room, checkpoint=self._vr_checkpoint, log=self.log):
            return False
        self.settle(150)                                   # switch scene (SE + map redraw)
        self.drain()
        g_now = tv.Grid(self.b)
        opened = g_now.col.get((barrier_tile[0] + tv.MAP_OFFSET,
                                barrier_tile[1] + tv.MAP_OFFSET), 1) == 0
        self.log(f"   [{label}] barrier {barrier_tile} open={opened}")
        self.snap(f"{label}_done")
        return opened

    def barrier_open(self, tile):
        g = tv.Grid(self.b)
        return g.col.get((tile[0] + tv.MAP_OFFSET, tile[1] + tv.MAP_OFFSET), 1) == 0

    def puzzle2_2f(self):
        # the row-19 boulder may sit ANYWHERE x14..33 after a partial chain — find it, push LEFT the
        # remaining distance onto the (14,19) switch
        if not self.ensure_strength((33, 19)):
            return False
        bl = None
        for ax in (33, 27, 21, 16):
            c = self.nearest_boulder((ax, 19), radius=6)
            if c and c[1] == 19 and 14 <= c[0] <= 33:
                bl = c
                break
        if bl is None:
            self.log("!! [2f-sw2] no boulder on row 19 — 3F reset detour needed")
            return False
        if bl[0] == 14:
            return True
        return self.push(bl, "LEFT", bl[0] - 14)

    def wedge(self, label, cap=4):
        b = self.b
        self.wedges[label] = self.wedges.get(label, 0) + 1
        # NS9: a post-battle EVOLUTION box (won mid-VR) JAMS overworld nav — dd_box does NOT flag it, so
        # drain()/handle_interrupts skip it and the go_warp step loop presses into a dead box forever. RAW
        # press-through (B, ungated by dd_box) on the early wedges to punch past it, then abort if stuck.
        if self.wedges[label] < cap:
            for _ in range(20):
                b.press("B", 8, 12, self.camp.render, owner="agent")
            return False
        self.log(f"!! [{label}] failed x{cap} — abort LOUD")
        self.snap(f"wedge_{label[:14]}")
        return True

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def badge8(self):
        return fm.read_flag(self.b, FLAG_BADGE_EARTH)

    # ── PHASE 0: teach EARTHQUAKE (TM26 -> Venusaur, over a droppable slot) — DEFAULT OFF ────────────────
    def _teach_eq(self):
        b, camp = self.b, self.camp
        eq_slot = next((s for s in range(6) if st.read_party_species(b, s) == 3), None)
        have = (st.read_party_moves(b, eq_slot) or []) if eq_slot is not None else []
        if eq_slot is None:
            self.log("   EQ teach SKIPPED — Venusaur (species 3) not in party")
            return
        if MOVE_EQ in have:
            self.log(f"   EQ already known (Venusaur slot {eq_slot}) — skipping teach")
            return
        if not TEACH_EQ:
            self.log(f"   EQ teach SKIPPED (POKEMON_TEACH_EQ off) — Venusaur keeps {have}; Razor Leaf "
                     f"carries VR")
            return
        try:
            import hm_teach as ht
            _PROTECT = {75, 70, MOVE_EQ}       # Razor Leaf, Strength (boulders), EQ
            forget_idx = next((have.index(m) for m in (15, 290) if m in have), None)
            if forget_idx is None:
                forget_idx = next((i for i, m in enumerate(have) if m not in _PROTECT), None)
            self.log(f"   [teach-eq] forget_idx={forget_idx} (moves before={have})")
            teacher = ht.TeachFlow(camp, log=self.log)
            r = teacher.teach("surf", eq_slot, forget_idx=forget_idx,
                              item_override=TM26_ITEM, move_override=MOVE_EQ)
            after = st.read_party_moves(b, eq_slot) or []
            self.log(f"   [teach-eq] -> {r}; moves now {after} (EQ={'YES' if MOVE_EQ in after else 'NO'})")
            self.drain(key="B")
            self.settle(60)
        except Exception as e:
            self.log(f"   [teach-eq] errored: {e} — continuing without EQ (LOUD)")

    # ── OFF-ROUTE APPROACH (2026-08-10 LIVE, the Route-10 door-spin): overworld graph hops ─────────────
    def _warp_to_dest(self, dest):
        """Enter the first warp on this map whose destination is `dest` (pret dest_map)."""
        dest = tuple(dest)
        try:
            for wxy, d, _i in tv.read_warps(self.b):
                if tuple(d) == dest:
                    return self.camp.enter_warp(pick=tuple(wxy)) == "warped"
        except Exception as e:
            self.log(f"   route2-viridian: warp-to {dest} failed ({e})")
        return False

    def _route2_unlatch(self):
        try:
            self.camp._stuck_request = None
            if self.camp._stuckwatch is not None:
                self.camp._stuckwatch.reset()
        except Exception:
            pass

    def _route2_south_to_viridian(self, _depth=0):
        """Route 2 north cannot walk overworld-south to Viridian.

        pret/pokefirered Route2/map.json: Viridian is a `down` connection at the
        BOTTOM of the map. Forest north doors are (5,13)/(6,13) → GATE_N (15,3).
        Forest south doors are (5,51)/(6,51) → GATE_S (15,0). The Cut tree at
        (11,13) is the 'NPC' travel saw at (11,14). Live 2026-08-13: south-edge
        BFS from y=14 is a wall. North half (y<40) goes through the forest.
        South half (y>=40, past forest/ledges) walks the Viridian connection."""
        if _depth > 6:
            self.log("   route2-viridian: chain depth cap — failed")
            return "failed"
        camp, b = self.camp, self.b
        ROUTE2 = (3, 20)
        self._route2_unlatch()
        here = tuple(tv.map_id(b))
        if here == VIRIDIAN:
            return "moved"

        if here == R2_LEDGE or here == R2_HOUSE:
            self.log(f"   route2-viridian: interior {here} — exit south onto Route 2")
            if not self._warp_to_dest(ROUTE2):
                try:
                    camp._exit_to_overworld(max_tries=4)
                except Exception:
                    camp.enter_warp(prefer="south")
            here = tuple(tv.map_id(b))

        if here == GATE_N:
            # pret: south mats (6/7/8,10) → FOREST warp 2; north (7,1) → Route 2.
            self.log("   route2-viridian: north forest gate -> forest")
            if not self._warp_to_dest(FOREST):
                camp.enter_warp(prefer="south")
            here = tuple(tv.map_id(b))

        if here == FOREST:
            # pret: south mats (28/29/30,62) → GATE_S; north (4/5/6,9) → GATE_N.
            self.log("   route2-viridian: Viridian Forest -> south gate")
            old = camp.trav.battle_runner
            try:
                if getattr(camp, "_flee_runner", None):
                    camp.trav.battle_runner = camp._flee_runner
                if not self._warp_to_dest(GATE_S):
                    camp.enter_warp(prefer="south")
            finally:
                camp.trav.battle_runner = old
            here = tuple(tv.map_id(b))

        if here == GATE_S:
            # pret: south mats (6/7/8,10) → Route 2 warp 2 = (5,51); north (7,1) → forest.
            self.log("   route2-viridian: south forest gate -> Route 2 south")
            if not self._warp_to_dest(ROUTE2):
                camp.enter_warp(prefer="south")
            here = tuple(tv.map_id(b))

        if here == VIRIDIAN:
            self._last_hop_src = ROUTE2
            return "moved"
        if here != ROUTE2:
            if here in (GATE_N, FOREST, GATE_S, R2_LEDGE, R2_HOUSE):
                return self._route2_south_to_viridian(_depth=_depth + 1)
            self.log(f"   route2-viridian: off Route 2 at {here} — failed")
            return "failed"

        xy = tuple(tv.coords(b) or (0, 0))
        if xy[1] < 40:
            # NORTH half. pret forest doors (5,13)/(6,13). Do NOT overworld-south.
            self.log(f"   route2-viridian: north Route 2 {xy} — through Viridian Forest "
                     f"(pret doors (5,13)/(6,13) → {GATE_N})")
            # Cut tree is (11,13) — one tile north of the live wedge. Open sand is y<=12.
            if xy[1] >= 13:
                camp.trav.travel(target_map=None, arrive_coord=(11, 12),
                                 avoid={(17, 11)}, max_steps=80, max_seconds=40)
            self._route2_unlatch()
            entered = False
            for door, appr in (((6, 13), (6, 14)), ((5, 13), (5, 14))):
                camp.trav.travel(target_map=None, arrive_coord=appr,
                                 avoid={(17, 11)}, max_steps=200, max_seconds=90)
                self._route2_unlatch()
                if camp.enter_warp(pick=door) == "warped":
                    entered = True
                    break
            if not entered:
                self.log("   route2-viridian: forest door missed — cutting and retrying")
                try:
                    if camp.field and camp.field.clear_obstacle("cut", "LEFT") == "used":
                        self._route2_unlatch()
                    camp.enter_warp(pick=(6, 13))
                except Exception as e:
                    self.log(f"   route2-viridian: forest enter failed ({e})")
            if tuple(tv.map_id(b)) != ROUTE2:
                return self._route2_south_to_viridian(_depth=_depth + 1)
            return "failed"

        # SOUTH half (y>=40): past the forest / ledge-house landing. Map connection down.
        self.log(f"   route2-viridian: south Route 2 {xy} — edge south to Viridian")
        avoid = {tuple(w[0]) for w in tv.read_warps(b)}
        camp.trav.travel(target_map=VIRIDIAN, edge="south", avoid=avoid, max_seconds=180)
        if tuple(tv.map_id(b)) == VIRIDIAN:
            self._last_hop_src = ROUTE2
            return "moved"
        return "failed"

    def _graph_hop_to(self, dst):
        """One learned-world-graph hop toward `dst`. Mirrors _travel_to_known's hop actuation
        (warp walk-in + enter_warp / _edge_travel edge cross), heal-aware. Clears a stale
        watchdog latch FIRST so a prior leg's disengage can't insta-bail this hop (the 12:24
        chalk: a latched disengage killed every retry leg before it started). Returns
        'moved' | 'healed' | 'hm_blocked' | 'failed'."""
        camp, b = self.camp, self.b
        here = tuple(tv.map_id(b))
        if here == tuple(dst):
            return "moved"
        if here in ((3, 20), FOREST, GATE_N, GATE_S, R2_LEDGE, R2_HOUSE) and tuple(dst) in (
                VIRIDIAN, FOREST, GATE_S, (3, 20)):
            return self._route2_south_to_viridian()
        # BOUNDARY HYSTERESIS (2026-08-10 ping-pong fix): never hop back to the map we just
        # crossed FROM this tick. The graph BFS has no notion of "I just left there" — so on a
        # two-map border it picks the first-neighbor hop toward dst, which is the map we came
        # from (proven: Route 7 <-> Celadon loop at line 660-672). Track the previous map per
        # strike instance; the next hop from here must NOT target it. Only blocks the immediate
        # reverse — legitimate rerouting through that map on a LATER tick is unaffected.
        prev = self._last_hop_src
        if prev is not None and tuple(dst) == prev:
            self.log(f"   graph hop: hysteresis — dst {dst} == last src {prev}, seeking alternate")
        try:
            camp._stuck_request = None
            if camp._stuckwatch is not None:
                camp._stuckwatch.reset()
        except Exception:
            pass
        try:
            step = camp._next_step_rideable(here, tuple(dst), avoid=set())
        except Exception as e:
            self.log(f"!! graph hop toward {dst} errored ({e}) — LOUD")
            return "failed"
        if step is None:
            self.log(f"!! no world-graph route {here} -> {dst} from her feet (LOUD)")
            return "failed"
        nxt, kind, detail = step
        # CYCLING-ROAD POISON (2026-08-13): never hop onto Route 16 / Celadon-west when the
        # destination is Viridian. No Bike = the gate NPC is a hard wall (soak 20260811).
        _bike_wall = {(3, 34), (3, 35), (3, 36)}
        if tuple(dst) == VIRIDIAN and tuple(nxt) in _bike_wall:
            self.log(f"   graph hop: refusing Cycling Road {nxt} (no Bike) — excluding it")
            try:
                alt = camp._next_step_rideable(here, tuple(dst), avoid=list(_bike_wall))
                if alt is not None:
                    nxt, kind, detail = alt
                else:
                    self.log(f"!! no Viridian route excluding Cycling Road from {here} (LOUD)")
                    return "failed"
            except Exception as e:
                self.log(f"!! Cycling Road exclude errored ({e}) — LOUD")
                return "failed"
        # HYSTERESIS CHECK: if the first-hop neighbor is the map we just came from, ask the
        # world graph for the route EXCLUDING that previous map — the BFS then finds the real
        # forward path (or None if the previous map was the only bridge, in which case we fail
        # honestly rather than oscillate forever).
        if prev is not None and tuple(nxt) == prev:
            self.log(f"   graph hop: hysteresis — next hop {nxt} == last src {prev}, excluding it")
            try:
                alt = camp._next_step_rideable(here, tuple(dst), avoid=[prev])
                if alt is not None:
                    nxt, kind, detail = alt
                else:
                    self.log(f"!! hysteresis: excluding {prev} leaves no route {here} -> {dst} (LOUD)")
                    return "failed"
            except Exception as e:
                self.log(f"!! hysteresis re-route errored ({e}) — LOUD")
                return "failed"
        self.log(f"   off-route overworld {here}: graph hop toward {dst} "
                 f"({('warp ' + str(detail)) if kind == 'warp' else detail} -> {nxt})")
        if kind == "warp":
            before = tuple(tv.map_id(b))
            camp.trav.travel(target_map=None, arrive_coord=detail, max_steps=300)
            if tuple(tv.map_id(b)) == before:
                camp.enter_warp(pick=detail)
            self._last_hop_src = tuple(here)
            return "moved" if tuple(tv.map_id(b)) != before else "failed"
        r = camp._edge_travel(nxt, detail)
        if r == "need_heal":
            camp.heal_nearest()
            return "healed"
        moved = tuple(tv.map_id(b)) != here
        if not moved:
            self._blocked_nxt = tuple(nxt)
        if r == "no_route_hm_blocked":
            return "hm_blocked"
        if moved:
            self._last_hop_src = tuple(here)
            return "moved"
        for alt in ("west", "east", "north", "south"):
            if alt == detail or tuple(tv.map_id(b)) != here:
                continue
            self.log(f"   hop edge {detail} wedged — probing {alt} for {nxt}")
            camp._edge_travel(nxt, alt, budget_s=90)
            if tuple(tv.map_id(b)) == tuple(nxt):
                self.log(f"   hop edge CORRECTED: {alt} reached {nxt} (model edge self-heals)")
                self._last_hop_src = tuple(here)
                return "moved"
        return "failed"

    def _nearest_pc_map(self, here, avoid_maps=frozenset()):
        """Nearest mapped-PC-door map by learned-graph path length (the Cut escort's fetch point).
        The graph is capability-blind, so `avoid_maps` drops every candidate whose route runs
        through a map we just proved hm-blocked (from Route 9 the naive nearest is Cerulean —
        which is on the FAR side of the very Cut tree that blocked us)."""
        from campaign import CITY_PC_DOORS
        best, best_len = None, None
        for pm in CITY_PC_DOORS:
            pm = tuple(pm)
            if pm == tuple(here):
                return pm
            if pm in avoid_maps:
                continue
            try:
                r = self.camp.world.route(here, pm)
            except Exception:
                r = None
            if not r or any(tuple(m) in avoid_maps for m in r[1:]):
                continue
            if best_len is None or len(r) < best_len:
                best, best_len = pm, len(r)
        return best

    def _box_mon_moves(self, bx, sl):
        """Decrypt the 4 move IDs of a BOXED mon — mirrors _box_scan's BoxPokemon decryption
        (gPokemonStoragePtr 0x03005010, 80-byte BoxPokemon) but reads the Attacks substructure
        (pret PokemonSubstruct1 { u16 moves[4]; }), same scheme as read_party_moves. Read-only."""
        GSTORAGE_PTR, BOX_MON_SIZE, PER_BOX = 0x03005010, 80, 30
        b = self.camp.b
        base0 = b.rd32(GSTORAGE_PTR)
        if not base0:
            return []
        mbase = base0 + 4 + (bx * PER_BOX + sl) * BOX_MON_SIZE
        pid = b.rd32(mbase)
        if pid == 0 and b.rd32(mbase + 4) == 0:
            return []
        key = pid ^ b.rd32(mbase + 4)
        order = st._SUBSTRUCT_ORDER[pid % 24]
        a = mbase + 32 + order.index("A") * 12
        w0 = b.rd32(a + 0) ^ key
        w1 = b.rd32(a + 4) ^ key
        return [w0 & 0xFFFF, (w0 >> 16) & 0xFFFF, w1 & 0xFFFF, (w1 >> 16) & 0xFFFF]

    def _ensure_cut_escort(self):
        """CUT ESCORT (2026-08-10 LIVE, the Route-9 tree — the ONLY land gate from eastern Kanto
        to Viridian): the party's only Cut user (Diglett) was boxed for the Zapdos seat, so the
        endgame march dead-ended on 'no_route_hm_blocked'. Field it: at a mapped-Center map,
        deposit the Kadabra passenger (Jonny's call — never the lead; the planner-default chaff
        would box Moltres, the only off-plan member), then withdraw a boxed mon that KNOWS Cut
        from the OPEN box (knows-first: the box also holds cut-CAPABLE rattata/tentacool that
        would withdraw useless). The retried hop's travel then clears the tree in-leg.
        Returns
          'ready'   — a party mon already knows Cut (travel clears the tree itself)
          'need_pc' — no mapped PC door on this map (caller hops toward the nearest one)
          'swapped' — Cut user fielded (retry the hop)
          'none'    — nothing boxed knows Cut / swap failed (caller wedge-caps)."""
        camp, b = self.camp, self.b
        pc = b.rd8(ram.GPLAYER_PARTY_CNT)
        if st.party_knows_move(b, MOVE_CUT, pc) is not None:
            return "ready"
        from campaign import CITY_PC_DOORS
        here = tuple(tv.map_id(b))
        pc_door = CITY_PC_DOORS.get(here)
        if not pc_door:
            return "need_pc"
        try:
            cb, occ = camp._box_scan()
            cand = next(((bx, sl, sp) for (bx, sl), sp in sorted(occ.items())
                         if bx == cb and MOVE_CUT in self._box_mon_moves(bx, sl)), None)
            if cand is None:
                self.log("!! CUT ESCORT: no boxed mon KNOWS Cut in the open box (LOUD)")
                return "none"
            bx, sl, sp = cand
            dep_slot = next((s for s in range(1, pc)
                             if st.read_party_species(b, s) == KADABRA_SP), None)
            if dep_slot is not None:
                rd = camp.deposit_mon(dep_slot, pc_door)
                if rd != "deposited":
                    self.log(f"!! CUT ESCORT: Kadabra deposit failed ({rd}) — party intact (LOUD)")
                    return "none"
                self._escort_deposited = KADABRA_SP
                cb, occ = camp._box_scan()          # deposit adds an occupant — re-locate
                if (bx, sl) not in occ or bx != cb:
                    self.log("!! CUT ESCORT: box shifted after deposit — aborting (Kadabra "
                             "boxed, benign; the escort retries next leg)")
                    return "none"
            else:
                from collections import Counter
                _c0 = Counter(camp._box_scan()[1].values())
                if not camp._box_swap_for_hm("cut", camp.read_live_state()):
                    self.log("!! CUT ESCORT: no Kadabra in party and the NS#28 chaff swap "
                             "found no depositable mon (LOUD)")
                    return "none"
                _added = list((Counter(camp._box_scan()[1].values()) - _c0).elements())
                self._escort_deposited = _added[0] if _added else None
                cb, occ = camp._box_scan()
                cand = next(((bx2, sl2, sp2) for (bx2, sl2), sp2 in sorted(occ.items())
                             if bx2 == cb and MOVE_CUT in self._box_mon_moves(bx2, sl2)), None)
                if cand is None:
                    return "none"
                bx, sl, sp = cand
            rw = camp.withdraw_mon(bx, sl, pc_door)
            if rw != "withdrawn":
                self.log(f"!! CUT ESCORT: withdraw failed ({rw}) (LOUD)")
                return "none"
            self.log(f"   CUT ESCORT: fielded {st.SPECIES_NAME.get(sp, sp)} from box{bx} "
                     f"slot{sl} — the Route 9 tree is next")
            camp.on_event(f"none of my team knows Cut — but there's a "
                          f"{st.SPECIES_NAME.get(sp, 'Diglett')} sitting in the box. swapping "
                          f"them in; one tree stands between us and the League.",
                          kind="roster", tier=2)
            return "swapped"
        except Exception as e:
            self.log(f"!! CUT ESCORT errored ({e}) — LOUD")
            return "none"

    def _cut_blocker_likely(self):
        """The 'hm_blocked' signal DIES once the tree tile sits in the persistent blocked-NPC
        memory: the planning BFS excludes it, so travel aborts with a generic no_route before
        ever identifying the tree (the 17:20 chalk — wedged at (71,10), escort never fired).
        Tree evidence is a NEARBY scanned Cut tree OR a persistent blocked tile on THIS map —
        at 70 tiles the object scan is empty (objects only load close-in), so the poisoned block
        memory IS the far-detection fingerprint. True iff that evidence holds, no party member
        can use Cut, and a boxed mon KNOWS Cut (the Diglett) — i.e. the escort can fix this wall.
        Pure RAM reads; one attempt per map per strike; never raises."""
        try:
            import field_moves as _fmv
            if _fmv.can_use(self.b, "cut"):
                return False                      # Cut usable -> travel's release+auto-cut owns it
            here = tuple(tv.map_id(self.b))
            if getattr(self, "_escort_likely_tried", None) == here:
                return False                      # one escort attempt per map per strike
            tree_near = bool(_fmv.scan_field_objects(self.b, {_fmv.GFX_CUT_TREE}))
            blocked_here = {tuple(t) for (m, t) in getattr(self.camp, "_blocked_npcs", ())
                            if tuple(m) == here}
            if not (tree_near or blocked_here):
                return False                      # no tree / no block -> not a Cut wall
            self._escort_likely_tried = here
            cb, occ = self.camp._box_scan()
            # a boxed mon that KNOWS Cut — capable-only would withdraw a Rattata/Tentacool that
            # can learn Cut but doesn't, which can't clear the tree (the live box has both).
            return any(bx == cb and MOVE_CUT in self._box_mon_moves(bx, sl)
                       for (bx, sl), _sp in occ.items())
        except Exception:
            return False

    def _release_cut_blocks(self, tree_map):
        """Cut is now fielded but the tree tile is still in the poisoned blocked-NPC memory, and
        travel's FIELD-OBSTACLE RELEASE can't see the tree at range (object scan empty 69 tiles
        away), so it never un-blocks. Clear the blocked marks on the tree's map so the BFS can
        path TO the tree; the in-leg chokepoint logic re-identifies it up close and auto-cuts
        (can_use(cut) is now True). Releasing a legit plain-NPC mark is harmless — travel
        re-encounters and re-marks it on arrival."""
        try:
            bn = getattr(self.camp, "_blocked_npcs", None)
            if not bn:
                return
            tm = tuple(tree_map)
            removed = sorted(t for (m, t) in list(bn) if tuple(m) == tm)
            for t in removed:
                bn.discard((tm, t))
            if removed:
                self.log(f"   CUT ESCORT: Cut fielded — released poisoned block(s) {removed} on "
                         f"{tm} so travel can path to the tree and auto-cut")
        except Exception:
            pass

    def _march_hop(self):
        """One hop toward VIRIDIAN along CREDITS_MARCH. Never Celadon-west / Cycling Road."""
        here = tuple(tv.map_id(self.b))
        if here == VIRIDIAN:
            return "moved"
        join = CREDITS_JOIN.get(here)
        if join is not None:
            self.log(f"   credits-march JOIN {here} -> {join}")
            return self._graph_hop_to(join)
        if here in CREDITS_MARCH[:-1]:
            nxt = CREDITS_MARCH[CREDITS_MARCH.index(here) + 1]
            self.log(f"   credits-march {here} -> {nxt}")
            if here == (3, 29) and nxt == (1, 38):
                try:
                    self.camp.on_event(
                        "Diglett's Cave is the tunnel to Route 2 — Viridian is the other side, "
                        "not a detour.",
                        kind="travel", tier=2)
                except Exception:
                    pass
            return self._graph_hop_to(nxt)
        return self._graph_hop_to(VIRIDIAN)

    def _cut_usable_and_walled(self, here):
        """A party mon KNOWS Cut but a poisoned blocked tile still walls the BFS on this map.
        travel's FIELD-OBSTACLE RELEASE can't un-block it (the object scan is empty 69 tiles
        away), so the strike must release it itself. True iff Cut usable AND a blocked mark
        exists on `here`."""
        try:
            import field_moves as _fmv
            if not _fmv.can_use(self.b, "cut"):
                return False
            return any(tuple(m) == tuple(here)
                       for (m, _t) in getattr(self.camp, "_blocked_npcs", ()))
        except Exception:
            return False

    def _hop_toward_viridian(self):
        """One dispatch-loop step toward VIRIDIAN from an off-corridor overworld map. Normally a
        plain graph hop; when the leg is Cut-blocked (the Route-9 tree), runs the Cut escort:
        hop to the nearest mapped Center, box the Kadabra passenger, field the boxed Cut
        learner — then the retried hop's travel clears the tree in-leg. The escort fires on the
        honest 'hm_blocked' signal OR on a generic hop failure when a Cut tree is the likely
        blocker (the poisoned blocked-NPC-memory case — see _cut_blocker_likely). Returns True
        on any progress (moved / healed / escort step), False when genuinely walled (caller
        wedge-caps)."""
        here = tuple(tv.map_id(self.b))
        if here == VIRIDIAN:
            return True
        r = self._graph_hop_to(VIRIDIAN)
        if r in ("moved", "healed"):
            return True
        if r not in ("hm_blocked", "failed"):
            return False
        # CASE A: a party mon already KNOWS Cut but a poisoned blocked tile still walls the BFS
        # (travel's release can't see the tree at range). Release it and retry — the in-leg
        # chokepoint logic will auto-cut up close.
        if self._cut_usable_and_walled(here):
            self._release_cut_blocks(here)
            return True
        # CASE B: no party Cut user — run the escort to field one.
        if not self._cut_blocker_likely():
            return False
        # CUT-BLOCKED: run the escort to completion (bounded) — hop to the nearest mapped PC
        # THIS side of the tree, swap the passenger for the boxed Cut learner, then the next
        # dispatch iteration retries the hop and travel clears the tree in-leg.
        tree_map = here
        for _ in range(4):
            esc = self._ensure_cut_escort()
            if esc == "swapped":
                self._release_cut_blocks(tree_map)
                return True
            if esc != "need_pc":                    # 'ready' / 'none' — wedge counts
                return False
            avoid = set()
            bn = getattr(self, "_blocked_nxt", None)
            if bn:
                avoid.add(tuple(bn))
            pc_map = self._nearest_pc_map(here, avoid_maps=avoid)
            if pc_map is None or pc_map == here:
                self.log("!! CUT ESCORT: no reachable mapped PC from here (LOUD)")
                return False
            self.log(f"   CUT ESCORT: no PC on {here} — fetching the Cut user via {pc_map}")
            if self._graph_hop_to(pc_map) not in ("moved", "healed"):
                return False
            here = tuple(tv.map_id(self.b))
        return False

    def _restore_escort_party(self):
        """Viridian PC: dump Cut-escort + dead-weight, pull the three birds.

        Diglett was only here to Cut trees. Kadabra L19 / Lapras L26 are not the
        E4 plan (Jonny 2026-08-13). Box anyone not in DECLARED_SIX, withdraw
        Moltres, walk into Route 22 as Blastoise + Articuno + Zapdos + Moltres.
        Empty seats stay empty — no L18 fodder in the faint-through. Called at
        Viridian (pre-Route 22) with an Indigo-heal backstop. Does not latch
        until the party is only stompers."""
        if getattr(self, "_escort_restored", False):
            return
        camp, b = self.camp, self.b
        try:
            from campaign import CITY_PC_DOORS
            here = tuple(tv.map_id(b))
            pc_door = CITY_PC_DOORS.get(here)
            if not pc_door:
                return
            # Ace stays slot 0 — deposit_mon refuses slot 0.
            try:
                ace = next((s for s in range(b.rd8(ram.GPLAYER_PARTY_CNT))
                            if st.read_party_species(b, s) == 9), 0)
                if ace:
                    camp._swap_party_slots(0, ace)
            except Exception:
                pass
            dumped = []
            for _ in range(4):
                pc = b.rd8(ram.GPLAYER_PARTY_CNT)
                dump = next((s for s in range(1, pc)
                             if st.read_party_species(b, s) not in DECLARED_SIX), None)
                if dump is None:
                    break
                sp = st.read_party_species(b, dump)
                name = st.SPECIES_NAME.get(sp, sp)
                rd = camp.deposit_mon(dump, pc_door)
                if rd != "deposited":
                    self.log(f"!! STEAMROLL: boxing {name} failed ({rd}) — retry next PC (LOUD)")
                    return
                dumped.append(name)
                self.log(f"   STEAMROLL: boxed {name} at {camp.world.name(here)} "
                         f"(Cut escort / dead weight — not the E4)")
            for _ in range(3):
                pc = b.rd8(ram.GPLAYER_PARTY_CNT)
                if pc >= 6:
                    break
                have = {st.read_party_species(b, s) for s in range(pc)}
                missing = [sp for sp in DECLARED_SIX if sp not in have]
                if not missing:
                    break
                cb, occ = camp._box_scan()
                cand = next(((bx, sl, sp) for (bx, sl), sp in sorted(occ.items())
                             if bx == cb and sp in missing), None)
                if cand is None:
                    self.log(f"!! STEAMROLL: {st.SPECIES_NAME.get(missing[0], missing[0])} "
                             f"not in the open box (LOUD)")
                    break
                rw = camp.withdraw_mon(cand[0], cand[1], pc_door)
                if rw != "withdrawn":
                    self.log(f"!! STEAMROLL: withdraw "
                             f"{st.SPECIES_NAME.get(cand[2], cand[2])} failed ({rw}) — "
                             f"retrying at the next PC (LOUD)")
                    return
                self.log(f"   STEAMROLL: {st.SPECIES_NAME.get(cand[2], cand[2])} back on "
                         f"the team")
            pc = b.rd8(ram.GPLAYER_PARTY_CNT)
            have = {st.read_party_species(b, s) for s in range(pc)}
            extras = [st.SPECIES_NAME.get(sp, sp) for sp in have if sp not in DECLARED_SIX]
            names = [f"{st.SPECIES_NAME.get(st.read_party_species(b, s), '?')}"
                     for s in range(pc)]
            self.log(f"   STEAMROLL PARTY: {names} dumped={dumped or 'none'}")
            if not extras and 146 in have:
                self._escort_restored = True
        except Exception as e:
            self.log(f"!! STEAMROLL restore errored ({e}) — LOUD")

    # ── the strike ───────────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        self.log(f"   victory road strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"badge8={self.badge8()} lead={self.lead_frac():.0%}")
        if not self.badge8():
            self.log("!! badge 8 not held — wrong state, abort (this is a post-badge-8 vehicle)")
            return "stuck"
        if tuple(tv.map_id(b)) == INDIGO:
            self.log("   already at the Indigo Plateau — nothing to strike")
            return "reached_indigo"

        self._teach_eq()

        # ── PHASES 1-5: ONE WHITEOUT-TOLERANT DISPATCH LOOP (map-keyed; progress ratchets in the save) ──
        r23_logged = vr_logged = False
        while time.time() < self.deadline:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            if here == INDIGO:
                break
            if here == VIRIDIAN:
                if self.lead_frac() < 0.9:
                    camp.heal_nearest()
                    continue
                self._restore_escort_party()   # box the Cut user, back the passenger — E4 six whole
                if not self.cross_edge("west", "to-r22") and self.wedge("viridian-west"):
                    return "stuck"
            elif here == R22:
                # westward crosses Gary's trigger col 33 (scene + battle fire mid-path; handle_interrupts
                # owns them; a loss whiteouts and this loop recovers via the center)
                if not self.go_warp((8, 5), GATE, "gate-south"):
                    if tuple(tv.map_id(b)) == R22 and self.wedge("gate-south"):
                        self.snap("gate_fail")
                        return "stuck"
            elif here == GATE:
                cands = [tuple(xy) for xy, d, _w in tv.read_warps(b) if tuple(d) == R23]
                if not cands:
                    self.log("!! no R23 warp inside the gate — abort")
                    return "stuck"
                cands.sort(key=lambda t: t[1])            # north side = lowest y
                if not self.go_warp(cands[0], R23, "gate-thru"):
                    self.drain()
                    if self.wedge("gate-thru", 6):
                        return "stuck"
            elif here == R23:
                cx, cy = tuple(tv.coords(b) or (0, 0))
                # Stoop of VR 1F door (5, 28) is the SOUTH entrance. y<=30 includes
                # it and BFS-stalls trying to walk to Indigo around the mountain.
                at_vr1_stoop = (abs(cx - VR1F_DOOR[0]) <= 4 and 26 <= cy <= 34)
                past_vr = (cy <= 30) and not at_vr1_stoop
                if past_vr:
                    if not self.cross_edge("north", "to-indigo") and self.wedge("r23-north"):
                        return "stuck"
                else:
                    if not r23_logged:
                        self.log(f"   ROUTE 23 @ {tv.coords(b)} after {self.n_battles} battles "
                                 f"(Gary handled en route) [lead {self.lead_frac():.0%}]")
                        r23_logged = True
                    if at_vr1_stoop:
                        self.log(f"   R23 south stoop @ {(cx, cy)} — VR 1F door "
                                 f"{VR1F_DOOR}, not indigo-band")
                        try:
                            self.camp._release_wedge_marks_on(
                                [R23], "vr1-stoop — phantom watchdog marks from indigo-band stall")
                        except Exception:
                            pass
                    if not self.go_warp(VR1F_DOOR, VR1F, "vr-door") and self.wedge("vr-door"):
                        return "stuck"
            elif here == VR1F:
                if not vr_logged:
                    self.log(f"   VICTORY ROAD 1F @ {tv.coords(b)} [lead {self.lead_frac():.0%}]")
                    vr_logged = True
                if not self.barrier_open((12, 14)):
                    if not self.run_puzzle(VR1F_PUZZLE, (12, 14), "1f-switch") \
                            and self.wedge("1f-switch", 3):
                        return "stuck"
                elif not self.go_warp((3, 2), VR2F, "1f-ladder") and self.wedge("1f-ladder"):
                    return "stuck"
            elif here == VR2F:
                cx, cy = tuple(tv.coords(b) or (0, 0))
                if cx >= 36 and cy <= 13:                 # east pocket (from 3F drop)
                    if (self.go_warp((48, 12), R23, "vr-exit")
                            or self.go_warp((47, 13), R23, "vr-exit-b")
                            or self.go_warp((49, 13), R23, "vr-exit-c")):
                        self.log(f"   VICTORY ROAD CLEARED -> R23 north @ {tv.coords(b)} "
                                 f"[lead {self.lead_frac():.0%}, battles {self.n_battles}]")
                    elif self.wedge("vr-exit"):
                        return "stuck"
                elif not self.barrier_open((13, 10)):
                    if not self.run_puzzle(VR2F_PUZZLE1, (13, 10), "2f-switch1") \
                            and self.wedge("2f-switch1", 3):
                        return "stuck"
                elif not self.barrier_open((33, 16)):
                    if fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN):
                        # the row-19 boulder hasn't dropped from 3F yet — up the (34,9) ladder
                        if not self.go_warp((34, 9), VR3F, "2f-to-3f-detour") \
                                and self.wedge("2f-to-3f-detour"):
                            return "stuck"
                    elif self.puzzle2_2f():
                        self.settle(150)
                        self.drain()
                        self.log(f"   [2f-switch2] barrier (33,16) open={self.barrier_open((33, 16))}")
                    elif self.wedge("2f-switch2", 3):
                        return "stuck"
                elif not self.go_warp((36, 17), VR3F, "2f-to-3f") and self.wedge("2f-to-3f"):
                    return "stuck"
            elif here == VR3F:
                if fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN):
                    # THE RESET/REVEAL DETOUR — leg 1: boulder (32,5) -> switch (7,7) opens the 3F barrier
                    # (12,12-13); leg 2: push (33,18) RIGHT into hole (34,18) (drops it to 2F (33,19), clears
                    # 0x058), then jump in after it -> lands 2F (34,19). NEVER push boulder (35,13).
                    if not self.barrier_open((12, 12)):
                        if not self.run_puzzle(VR3F_SWITCH_PUZZLE, (12, 12), "3f-switch") \
                                and self.wedge("3f-switch", 3):
                            return "stuck"
                    elif not self.ensure_strength((33, 18)):
                        if self.wedge("3f-drop-strength"):
                            return "stuck"
                    elif not self.push((33, 18), "RIGHT", 1):
                        if self.wedge("3f-drop"):
                            return "stuck"
                    else:
                        self.settle(120)
                        self.log(f"   [3f-drop] boulder down the hole (34,18) — 2F boulder "
                                 f"hidden={fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN)}")
                        # the hole is a warp tile (0x66) — jump in after it
                        if not self.sea_walk(lambda c: c == (34, 18), "hole-jump",
                                             allow=((34, 18),)) and self.wedge("hole-jump"):
                            return "stuck"
                        self.settle(180)
                elif not self.go_warp((37, 10), VR2F, "3f-to-2f-east",
                                      # Ray+Tyra (38,13)/(39,13) are a trainerbattle_DOUBLE the battle agent
                                      # can't target — dodge their sight tiles via column 36
                                      avoid=((38, 14), (39, 14))) and self.wedge("3f-to-2f"):
                    return "stuck"
            else:
                # off-route — NOT on the Viridian->Indigo corridor.
                # OVERWORLD (map group 3 — the 2026-08-10 LIVE wedge): doors on outdoor maps are
                # ENTRANCES, not exits — the strike booted on Route 10 (post-Zapdos catch) and the
                # blind enter_warp spun 'no reachable door warped (entry geometry?)' forever. Hop
                # the learned world graph toward Viridian (the corridor's entry) instead; the loop
                # re-keys on each new map (whiteout-tolerant like every other branch).
                # INTERIOR (whiteout center, etc.) — exit to the overworld as before.
                # OVERWORLD + the Diglett's Cave tunnel (group 1) are on CREDITS_MARCH —
                # hop the chain. Other interiors (Centers, connectors) still exit toward Viridian.
                _on_march = (here in CREDITS_MARCH or here in CREDITS_JOIN
                             or (here and here[0] == 3))
                if _on_march:
                    _before = here
                    _hr = self._march_hop()
                    try:
                        _now = tuple(tv.map_id(b))
                        if _now != _before and not st.in_battle(b):
                            self.camp._bank_milestone(f"credits-march-{_now}")
                    except Exception:
                        pass
                    if _hr not in ("moved", "healed") and self.wedge("offroute-hop", 6):
                        return "stuck"
                else:
                    self.log(f"   off-route at {here} - exiting to the overworld")
                    # SETTLE FIRST (2026-08-10 dry-run chalk): right after a warp the coord read
                    # is STALE (still the old map's tile), so enter_warp's BFS starts off-grid and
                    # every door reads unreachable. Let the arrival land before picking a door.
                    self.settle(40)
                    # TARGET-AWARE EXIT (2026-08-10 dry-run chalk): connector interiors
                    # (Route 7<->Saffron passage) have doors BOTH ways; prefer="south" picked
                    # the door BACK to Route 7 and she ping-ponged. Pick the door whose warp
                    # destination is the next map on the world route to VIRIDIAN.
                    _pick_door = None
                    try:
                        _rt = camp.world.route(here, VIRIDIAN)
                        if _rt and len(_rt) > 1:
                            _want = tuple(_rt[1])
                            for _wxy, _d, _i in tv.read_warps(b):
                                if tuple(_d) == _want:
                                    _pick_door = tuple(_wxy)
                                    break
                            if _pick_door:
                                self.log(f"   target-aware exit: door {_pick_door} -> {_want}")
                    except Exception:
                        _pick_door = None
                    # approach side varies per door (top-edge doors need stand-below/step-UP,
                    # bottom-edge stand-above/step-DOWN) — chain both before falling back blind.
                    _xr = None
                    if _pick_door:
                        _xr = camp.enter_warp(pick=_pick_door, prefer="nearest")
                        if _xr == "no_warp":
                            _xr = camp.enter_warp(pick=_pick_door, prefer="south")
                    if _xr != "warped" and camp.enter_warp(prefer="south") == "no_warp" \
                            and self.wedge("offroute-exit", 6):
                        return "stuck"
                    self.settle(80)

        if tuple(tv.map_id(b)) != INDIGO:
            self.log(f"!! never reached Indigo Plateau (at {tv.map_id(b)}@{tv.coords(b)}) — deadline/exhausted")
            self.snap("70_fail")
            return "stuck"

        self.log(f"   INDIGO PLATEAU @ {tv.coords(b)} — healing at the League center")
        try:
            camp.heal_nearest()
        except Exception as e:
            self.log(f"   Indigo heal errored: {e} — continuing (LOUD)")
        self.drain(key="B")
        self._restore_escort_party()           # backstop: if the Cut user rode this far, box it here
        self.log(f"   INDIGO REACHED: pos {tv.map_id(b)}@{tv.coords(b)} | lead {self.lead_frac():.0%} | "
                 f"battles {self.n_battles} | money ${camp.money()}")
        self.snap("80_final")
        return "reached_indigo"


def run_strike(camp, log, dbg_dir=None):
    """Run the Victory-Road push (post-badge-8 -> Indigo Plateau) from wherever she stands, in ONE call.
    Resume-safe (map-keyed dispatch; progress ratchets in the save). Drives Viridian -> Route 22 (Gary) ->
    the gate -> Route 23 -> the Victory-Road boulder floors -> Route 23 north -> Indigo, heals at the League
    center. Returns 'reached_indigo' (success) | 'battle_loss' | 'stuck'."""
    return VictoryRoad(camp, log, dbg_dir).run()
