"""safari_strike.py — THE SAFARI ZONE STRIKE, in-loop (night shift #11, badge-6→7 build).

The badge-7 unlock (Blaine is on Cinnabar; the sea road there needs Surf; Victory Road later
needs Strength — BOTH live behind the Safari Zone). The general questline lands her ON Fuchsia
but there is no NPC-talk primitive that can run the Safari's classic loop (the Secret House is
UNREACHABLE on foot from the entrance pocket — the pond splits it). The champion cleared it with
the bespoke recon_safari.py strike; this is a FAITHFUL port of that proven script, driven by the
live `camp` so the general questline can call it as ONE decision (the same shape as beat_gym /
silph_strike / hideout_strike / tower_strike). FireRed coords are isolated here (rule 14 —
portability debt: a Kanto-Safari-Zone fact table, swap per game).

Ground truth (disasm pret, fetched 2026-07-07 — see recon_safari.py for the full pond/elevation notes):
  Fuchsia (3,7): Safari entrance door (24,5); Warden's House door (33,31).
  Entrance interior: ENTRY TRIGGERS at (3-5,3) — walking north fires the $500 join prompt (YES pays,
  30 balls); warp (4,1) -> SAFARI CENTER. The tour = the classic Safari loop: Center -EAST(43,15-17)->
  Area 1 East -> Area 2 North -> Area 3 West (Gold Teeth ball (28,14) + Secret House (12,7): HM03 Surf).
  Return = REVERSE the chain. Warden's house: give Gold Teeth -> HM04 STRENGTH.
  HM item ids: HM03 Surf = 341, HM04 Strength = 342 (TM-case pocket); Gold Teeth = key item.
SAFARI RULES: 600-step limit (running out = scripted warp back to the entrance — the strike treats an
unexpected entrance/city map as RE-ENTER-AND-RESUME, objectives are flag-idempotent); wild encounters
use the BALL/BAIT/ROCK/RUN menu (battle_agent's FIGHT menu doesn't exist) — the handler throws balls at
NEW species (dex doctrine) then flees; ALL battle-end drains use B (the catch flow ends in the nickname
keyboard — the AAAAAAAAAA class).
Success = items 341 + 342 in the TM pocket. run_strike returns "got_surf" | "got_strength" | "in_safari"
(objective in bag but still inside) | "not_here" (no strike from here) | "failed".
"""
import os
import time

import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Safari-Zone fact table (game-knowledge layer; rule 14 portability debt) ──────────────
FUCHSIA = (3, 7)
ENTRANCE_DOOR = (24, 5)              # on Fuchsia -> Safari entrance building
WARDEN_DOOR = (33, 31)              # on Fuchsia -> Warden's house
GOLD_TEETH_BALL = (28, 14)          # West (Area 3)
SECRET_DOOR_WEST = (12, 7)          # West -> Secret House
HM03, HM04 = 341, 342
FLAG_GOT_HM03 = 0x239               # Surf obtained (Secret House) — the Blaine sea-gate prereq
FLAG_GOT_HM04 = 0x23A              # Strength obtained (Warden) — Victory Road prereq
# The Safari maps are group 1 nums 63-71 (disasm; frlg_grind_spots '1,63' = Safari Zone). Anchors
# for the questline strike dispatcher: FUCHSIA (where the strike ENTERS) + every Safari interior
# (so a mid-tour re-tick re-fires the strike rather than ejecting her).
SAFARI_MAPS = {(1, n) for n in range(63, 72)}
SAFARI_ANCHORS = {FUCHSIA} | SAFARI_MAPS

KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}


class SafariStrike:
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
        self.thrown_species = set()      # ball economy: ONE cheap attempt per species (dex doctrine)
        self.safari_maps = set()         # resolved live as the tour discovers them
        self.wedges = {}
        self.start_map = tuple(tv.map_id(self.b))
        # resolved live from the city warp table on entry
        self.ENTRANCE = self.WARDEN_HOUSE = None
        self.CENTER = self.EAST = self.NORTH = self.WEST = self.SECRET = None

    # ── low-level readers ────────────────────────────────────────────────────────────────────────
    def fight(self):
        self.n_battles += 1
        return self.camp.battle_runner()

    def tm_pocket(self):
        return ht.pocket_items(self.b, ht.TM_CASE_OFF, 64)

    def key_items(self):
        return ht.pocket_items(self.b, ht.KEY_ITEMS_OFF, 30)

    def have_surf(self):
        return HM03 in self.tm_pocket()

    def have_strength(self):
        return HM04 in self.tm_pocket()

    def dest_of(self, xy):
        for wxy, d, _w in tv.read_warps(self.b):
            if tuple(wxy) == xy:
                return tuple(d)
        return None

    def fight_open(self):
        """Battle-open gate that SEES safari encounters. st.in_battle sanity-checks gBattleMons[0]
        (the PLAYER'S battle mon) which a safari battle never fields; the battle-resources pointer
        alone is the truth (valid EWRAM only in battle)."""
        return ram.valid_ewram_ptr(self.b.rd32(ram.GBATTLE_RES_PTR))

    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    # ── SAFARI BATTLE HANDLER (BALL/BAIT/ROCK/RUN menu — not the FIGHT menu) ──────────────────────
    def safari_battle(self):
        b, camp = self.b, self.camp
        self.n_battles += 1
        sp = st.read_enemy_species(b, 0)
        new = (sp and ram.pokedex_owns(b, sp) is False
               and sp not in self.thrown_species)
        if new:
            self.thrown_species.add(sp)
        nm = st.SPECIES_NAME.get(sp, f"#{sp}")
        self.log(f"   [safari] wild {nm} (new={new}) @ {tv.coords(b)}")

        def settle(n):
            for _ in range(n):
                b.run_frame()
                if not self.fight_open():
                    return False
            return True

        def bdrain():
            for _ in range(30):
                if self.fight_open() or not dd_box(b):
                    return
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()

        deadline = time.time() + 90
        threw = 0
        boxed = 0
        while self.fight_open() and time.time() < deadline:
            if dd_box(b) and boxed < 5:
                boxed += 1
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                continue
            boxed = 0
            if new and threw < 2:
                b.press("A", 8, 12, camp.render, owner="agent")   # cursor home = BALL
                threw += 1
                if not settle(500):
                    break
                continue
            for combo in (("DOWN", "RIGHT"), ("RIGHT", "DOWN"), ("DOWN",), ("RIGHT",), ()):
                for k in combo:
                    b.press(k, 8, 10, camp.render, owner="agent")
                    for _ in range(12):
                        b.run_frame()
                b.press("A", 8, 12, camp.render, owner="agent")
                if not settle(320):
                    break
                if dd_box(b):
                    bdrain()
                if not self.fight_open():
                    break
        for _ in range(40):
            if not dd_box(b):
                break
            b.press("B", 8, 12, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
        self.log(f"   [safari] battle over (in_battle={self.fight_open()}, threw {threw})")
        return "safari"

    def drain(self, max_a=40, key="B"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
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

    def on_battle(self):
        if tuple(tv.map_id(self.b)) in self.safari_maps:
            return self.safari_battle()
        return self.fight()

    # ── movement / geometry ──────────────────────────────────────────────────────────────────────
    def elev_of(self, sx, sy):
        b = self.b
        w = b.rd32(tv.BACKUP_LAYOUT)
        h = b.rd32(tv.BACKUP_LAYOUT + 4)
        mp = b.rd32(tv.BACKUP_LAYOUT + 8)
        bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
        if not (0 <= bx < w and 0 <= by < h):
            return -1
        return (b.rd16(mp + (by * w + bx) * 2) >> 12) & 0xF

    def safari_step(self, t):
        """One-tile step, tap-turn aware, NO SIDEWAYS NUDGE (nudges burn Safari steps + bounce onto
        adjacent door warps). Returns 'ok' | 'battle' | 'blocked' | 'warped'."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        cur = tuple(tv.coords(b) or (0, 0))
        d = (t[0] - cur[0], t[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)         # LEDGE HOP: one press, the game jumps 2
        key = KEY_OF.get(d)
        if key is None:
            return "blocked"
        for _attempt in range(3):              # tap 1 may only TURN her (tap-turn law)
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(60):
                b.run_frame()
                if tuple(tv.coords(b) or ()) == t:
                    break
            if self.fight_open():
                return "battle"
            if tuple(tv.map_id(b)) != m0:
                return "warped"
            if tuple(tv.coords(b) or ()) == t:
                return "ok"
        if dd_box(b):
            return "battle"
        return "blocked"

    def safari_bfs(self, g, start, goal_test, tile_ok):
        """tv.bfs + THE PER-EDGE ELEVATION LAW (a step is legal iff elevations match or either side
        is 0/0xF; ledge hops are exempt)."""
        from collections import deque
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
            ec = self.elev_of(cx, cy)
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if g.ledge_dir(cx + dx, cy + dy) == (dx, dy):
                    nx, ny = cx + 2 * dx, cy + 2 * dy
                else:
                    nx, ny = cx + dx, cy + dy
                    if not g.edge_open(cx, cy, dx, dy):
                        continue
                    en = self.elev_of(nx, ny)
                    if ec not in (0, 0xF) and en not in (0, 0xF) and en != ec:
                        continue
                if not (g.sx_lo <= nx <= g.sx_hi and g.sy_lo <= ny <= g.sy_hi):
                    continue
                if (nx, ny) in came or not tile_ok(nx, ny):
                    continue
                came[(nx, ny)] = cur
                q.append((nx, ny))
        return None

    def walk_path_to(self, tile, label, tries=8):
        b = self.b
        dead = set()
        budget = tries
        hops = 0
        while budget > 0:
            hops += 1
            if hops > 400:
                self.log(f"   [{label}] hop cap hit at {tv.coords(b)}")
                return False
            budget -= 1
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            p = self.safari_bfs(g, cur, lambda t: t == tile,
                                lambda sx, sy: g.walkable(sx, sy)
                                and g.ledge_dir(sx, sy) is None
                                and (sx, sy) not in wts and (sx, sy) not in npcs)
            self.log(f"   [{label}] replan at {cur} -> {tile} (len {len(p) if p else 0}, "
                     f"budget {budget}, head {[tuple(x) for x in (p or [])[1:4]]})")
            if not p:
                self.log(f"   [{label}] no NPC-free static path {cur} -> {tile} (dead={sorted(dead)})")
                return False
            for t in p[1:]:
                r = self.safari_step(tuple(t))
                if r == "battle":
                    self.on_battle()
                    self.drain()
                    budget += 1
                    break
                if r == "warped":
                    return False
                if r == "blocked":
                    dead.add(tuple(t))
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
            if tuple(tv.map_id(b)) != self.start_map:
                return False
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return tuple(tv.coords(b) or ()) == tile

    def engage(self, front, face, label, drains=3, key="B"):
        b, camp = self.b, self.camp
        if not self.walk_path_to(front, label):
            self.log(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if self.fight_open():
                self.on_battle()
                self.drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    self.drain(key=key)
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def step_warp(self, wt, label, tries=3):
        """Deliberately step onto a warp tile: approach a free neighbor (grass INCLUDED), then one
        LONG-HOLD press onto the warp. Success = map changed."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        g = tv.Grid(b)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        cur0 = tuple(tv.coords(b) or (0, 0))
        cands = []
        for nb, kin in (((wt[0] - 1, wt[1]), "RIGHT"), ((wt[0] + 1, wt[1]), "LEFT"),
                        ((wt[0], wt[1] - 1), "DOWN"), ((wt[0], wt[1] + 1), "UP")):
            if nb in wts or not g.walkable(nb[0], nb[1]):
                continue
            p = self.safari_bfs(g, cur0, lambda t, a=nb: t == a,
                                lambda sx, sy: g.walkable(sx, sy)
                                and g.ledge_dir(sx, sy) is None
                                and (sx, sy) not in wts and (sx, sy) not in npcs) \
                if cur0 != nb else [cur0]
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not self.walk_path_to(nb, f"{label}-approach",
                                                                         tries=14):
                if tuple(tv.map_id(b)) != m0:
                    return True
                continue
            for _try in range(tries):
                b.press(kin, 26, 10, camp.render, owner="agent")
                for _ in range(150):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if self.fight_open():
                    self.on_battle()
                    self.drain()
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(60):
                        b.run_frame()
                    self.log(f"   [{label}] stepped warp {wt}: {m0} -> {tuple(tv.map_id(b))} "
                             f"@ {tv.coords(b)}")
                    return True
            break
        if not cands:
            self.log(f"   [{label}] ZERO approach candidates for {wt}: grid {g.w}x{g.h}, "
                     f"cur {cur0} walkable={g.walkable(*cur0)}")
        self.log(f"!! [{label}] warp step {wt} failed (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def enter_to(self, dest, label):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            self.log(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        if cur in cands:
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            g6 = tv.Grid(b)
            for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                       (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
                if nb not in wts and g6.walkable(nb[0], nb[1]):
                    camp._step_to(nb)
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            camp.enter_warp(pick=wt)
            if self.fight_open():
                self.on_battle()
                self.drain()
                camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def wedge(self, k, cap, msg):
        b = self.b
        self.wedges[k] = self.wedges.get(k, 0) + 1
        if self.wedges[k] >= cap:
            self.snap(f"wedge_{k}")
            self.log(f"!! {msg} x{cap} — abort LOUD")
            return True
        self.drain()
        for _ in range(150):
            b.run_frame()
        return False

    # ── the tour ─────────────────────────────────────────────────────────────────────────────────
    def run(self):
        """Run the Safari tour from wherever she stands (Fuchsia or mid-Safari), obtaining HM03 Surf
        + HM04 Strength, then walk out to Fuchsia + heal. Returns a run_strike result string."""
        b, camp = self.b, self.camp
        here = tuple(tv.map_id(b))
        self.log(f"   safari strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"surf={self.have_surf()} strength={self.have_strength()} money=${camp.money()}")

        # already have both HMs — just get out (idempotent re-entry)
        if self.have_surf() and self.have_strength():
            return self._walk_out()

        # resolve the city warps (must be on/reach Fuchsia to enter)
        if here == FUCHSIA:
            self.ENTRANCE = self.dest_of(ENTRANCE_DOOR)
            self.WARDEN_HOUSE = self.dest_of(WARDEN_DOOR)
            if not self.ENTRANCE or not self.WARDEN_HOUSE:
                self.log(f"!! safari strike: entrance/warden door not on the Fuchsia warp table — abort "
                         f"(entrance={self.ENTRANCE}, warden={self.WARDEN_HOUSE})")
                return "failed"
            self.log(f"   resolved: entrance={self.ENTRANCE} warden_house={self.WARDEN_HOUSE}")
        elif here not in SAFARI_MAPS:
            return "not_here"

        deadline = time.time() + 1500
        last_map = None
        while time.time() < deadline:
            if self.have_surf() and self.have_strength():
                break
            here = tuple(tv.map_id(b))
            self.start_map = here
            if here != last_map:
                last_map = here
            if self.fight_open():
                self.on_battle()
                self.drain()
                continue
            if dd_box(b):
                self.drain()
                continue

            if here == FUCHSIA:
                if self.ENTRANCE is None:
                    self.ENTRANCE = self.dest_of(ENTRANCE_DOOR)
                    self.WARDEN_HOUSE = self.dest_of(WARDEN_DOOR)
                if self.have_surf() and not self.have_strength():
                    if not self.enter_to(self.WARDEN_HOUSE, "warden-door"):
                        if self.wedge("warden_door", 3, "can't enter the Warden's house"):
                            return "failed"
                    continue
                if not self.enter_to(self.ENTRANCE, "safari-entrance"):
                    if self.wedge("entrance", 3, "can't enter the Safari building"):
                        return "failed"
                continue

            if here == self.ENTRANCE:
                if self.have_surf():
                    if not self.enter_to(FUCHSIA, "exit-to-city"):
                        if self.wedge("exit_city", 3, "can't leave the entrance building"):
                            return "failed"
                    continue
                # pay + go in: the ENTRY TRIGGER sits ON (3-5,3)
                if self.CENTER is None:
                    self.CENTER = self.dest_of((4, 1))
                    if self.CENTER:
                        self.safari_maps.add(self.CENTER)
                self.walk_path_to((4, 4), "to-trigger")
                camp._step_to((4, 3))
                camp._step_to((4, 3))
                for _ in range(90):
                    b.run_frame()
                self.drain(key="A")                     # the join prompt: A = YES (default)
                for _ in range(60):
                    b.run_frame()
                self.drain(key="A")
                if tuple(tv.map_id(b)) != self.CENTER:
                    self.step_warp((4, 1), "into-center")
                if tuple(tv.map_id(b)) == self.CENTER:
                    for _ in range(180):
                        b.run_frame()
                    self.wedges.pop("into_center", None)
                elif self.wedge("into_center", 6, "can't reach the Safari Center"):
                    return "failed"
                continue

            if self.CENTER and here == self.CENTER:
                if self.EAST is None:
                    self.EAST = self.dest_of((43, 15))
                    if self.EAST:
                        self.safari_maps.add(self.EAST)
                if self.have_surf():
                    if not any(self.step_warp(w, "center-exit") for w in ((26, 30), (25, 30), (27, 30))):
                        if self.wedge("center_exit", 4, "can't exit the Safari Center"):
                            return "failed"
                    continue
                if not any(self.step_warp(w, "to-east") for w in ((43, 16), (43, 15), (43, 17))):
                    if self.wedge("to_east", 4, "can't reach Area 1 (East)"):
                        return "failed"
                continue

            if self.EAST and here == self.EAST:
                if self.NORTH is None:
                    self.NORTH = self.dest_of((8, 9))
                    if self.NORTH:
                        self.safari_maps.add(self.NORTH)
                if self.have_surf():
                    if not any(self.step_warp(w, "east-back") for w in ((8, 27), (8, 26), (8, 28))):
                        if self.wedge("east_back", 4, "can't get back to the Center"):
                            return "failed"
                    continue
                if not any(self.step_warp(w, "to-north") for w in ((8, 10), (8, 9), (8, 11))):
                    if self.wedge("to_north", 4, "can't reach Area 2 (North)"):
                        return "failed"
                continue

            if self.NORTH and here == self.NORTH:
                if self.WEST is None:
                    self.WEST = self.dest_of((10, 34))
                    if self.WEST:
                        self.safari_maps.add(self.WEST)
                if self.have_surf():
                    if not any(self.step_warp(w, "north-back") for w in ((48, 32), (48, 31), (48, 33))):
                        if self.wedge("north_back", 4, "can't get back to Area 1"):
                            return "failed"
                    continue
                if not any(self.step_warp(w, "to-west") for w in ((11, 34), (10, 34), (12, 34),
                                                                  (21, 34), (20, 34), (22, 34))):
                    if self.wedge("to_west", 4, "can't reach Area 3 (West)"):
                        return "failed"
                continue

            if self.WEST and here == self.WEST:
                if self.SECRET is None:
                    self.SECRET = self.dest_of(SECRET_DOOR_WEST)
                    if self.SECRET:
                        self.safari_maps.add(self.SECRET)
                teeth_ball = next((o for o in tv.read_object_templates(b)
                                   if tuple(o[0]) == GOLD_TEETH_BALL), None)
                if teeth_ball is not None and teeth_ball[2]:
                    got0 = set(self.key_items())
                    for face, front in (("UP", (GOLD_TEETH_BALL[0], GOLD_TEETH_BALL[1] + 1)),
                                        ("DOWN", (GOLD_TEETH_BALL[0], GOLD_TEETH_BALL[1] - 1)),
                                        ("RIGHT", (GOLD_TEETH_BALL[0] - 1, GOLD_TEETH_BALL[1])),
                                        ("LEFT", (GOLD_TEETH_BALL[0] + 1, GOLD_TEETH_BALL[1]))):
                        self.engage(front, face, "gold-teeth", drains=2)
                        tb = next((o for o in tv.read_object_templates(b)
                                   if tuple(o[0]) == GOLD_TEETH_BALL), None)
                        if tb is None or not tb[2]:
                            break
                        if tuple(tv.map_id(b)) != self.WEST:
                            break
                    tb = next((o for o in tv.read_object_templates(b)
                               if tuple(o[0]) == GOLD_TEETH_BALL), None)
                    if tb is not None and tb[2]:
                        if self.wedge("teeth", 4, "Gold Teeth ball not collected"):
                            return "failed"
                    else:
                        self.log(f"   GOLD TEETH banked (key items "
                                 f"{sorted(set(self.key_items()) - got0)} added)")
                    continue
                if not self.have_surf():
                    if not self.step_warp(SECRET_DOOR_WEST, "secret-house"):
                        if self.wedge("secret", 4, "can't reach the Secret House"):
                            return "failed"
                    continue
                if not any(self.step_warp(w, "west-back") for w in ((31, 5), (30, 5), (32, 5),
                                                                    (38, 5), (37, 5), (39, 5))):
                    if self.wedge("west_back", 4, "can't get back to Area 2"):
                        return "failed"
                continue

            if self.SECRET and here == self.SECRET:
                if not self.have_surf():
                    r = self.engage((6, 6), "UP", "hm03-man", drains=8)
                    self.drain(key="B")
                    self.log(f"   HM03 attendant -> {r} (surf in bag: {self.have_surf()})")
                    if self.have_surf():
                        self.snap("30_hm03")
                    elif self.wedge("hm03", 3, "HM03 not granted"):
                        return "failed"
                    continue
                if not self.step_warp((4, 9), "house-exit"):
                    if self.wedge("house_exit", 3, "can't leave the Secret House"):
                        return "failed"
                continue

            if here == self.WARDEN_HOUSE:
                if not self.have_strength():
                    npcs = [o for o in tv.read_object_templates(b) if o[2]]
                    tgt = tuple(npcs[0][0]) if npcs else (4, 4)
                    r = self.engage((tgt[0], tgt[1] + 1), "UP", "warden", drains=8)
                    self.drain(key="B")
                    self.log(f"   WARDEN -> {r} (strength in bag: {self.have_strength()})")
                    if self.have_strength():
                        self.snap("40_hm04")
                    elif self.wedge("warden", 3, "HM04 not granted"):
                        return "failed"
                    continue
                if not self.enter_to(FUCHSIA, "warden-exit"):
                    if self.wedge("warden_exit", 3, "can't leave the Warden's house"):
                        return "failed"
                continue

            # off-route (rest house detour, step-limit bounce): step out toward a known warp
            self.log(f"   off-route at {here}@{tv.coords(b)} — stepping out south/known warps")
            if not (self.CENTER and self.enter_to(self.CENTER, "reroute-center")):
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
                if tuple(tv.map_id(b)) == here:
                    if self.wedge(("offroute", here), 3, f"stuck off-route at {here}"):
                        return "failed"

        if not (self.have_surf() and self.have_strength()):
            self.log(f"!! safari strike incomplete (surf={self.have_surf()} "
                     f"strength={self.have_strength()}) — the flag holds; recovery owns the exit")
            self.snap("70_fail")
            return "in_safari" if self.have_surf() else "failed"
        return self._walk_out()

    def _walk_out(self):
        """Walk back to Fuchsia + heal, then report the win. Best-effort: the flag/item is the durable
        win (teach bridge + surf-aware travel handle the rest), so a stuck exit still returns success."""
        b, camp = self.b, self.camp
        out_deadline = time.time() + 300
        while tuple(tv.map_id(b)) != FUCHSIA and time.time() < out_deadline:
            here = tuple(tv.map_id(b))
            if here == self.ENTRANCE:
                self.enter_to(FUCHSIA, "final-exit")
            elif self.CENTER and here == self.CENTER:
                self.step_warp((26, 30), "final-center-exit")
            elif self.WEST and here == self.WEST:
                any(self.step_warp(w, "final-west-exit") for w in ((31, 5), (30, 5), (38, 5)))
            elif self.NORTH and here == self.NORTH:
                any(self.step_warp(w, "final-north-exit") for w in ((48, 32), (48, 31), (48, 33)))
            elif self.EAST and here == self.EAST:
                any(self.step_warp(w, "final-east-exit") for w in ((8, 27), (8, 26), (8, 28)))
            elif self.SECRET and here == self.SECRET:
                self.step_warp((4, 9), "final-house-exit")
            elif here == self.WARDEN_HOUSE:
                self.enter_to(FUCHSIA, "final-warden-exit")
            else:
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
            if self.fight_open():
                self.on_battle()
                self.drain()
        if tuple(tv.map_id(b)) == FUCHSIA:
            try:
                camp.heal_nearest()
            except Exception as e:
                self.log(f"   safari strike: post-tour heal errored ({e}) — continuing")
        self.log(f"   SAFARI DONE: surf={self.have_surf()} strength={self.have_strength()} "
                 f"key_items={self.key_items()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
                 f"battles {self.n_battles}")
        self.snap("80_final")
        if not self.have_surf():
            return "failed"
        if tuple(tv.map_id(b)) != FUCHSIA:
            return "in_safari"                 # HMs in bag but still inside — caller retries the exit
        return "got_surf"


def run_strike(camp, log, dbg_dir=None):
    """Run the Safari Zone strike (HM03 Surf + HM04 Strength) from wherever she stands, in ONE call.
    Returns:
      'got_surf'   — Surf (and Strength) obtained AND back out on Fuchsia (healed).
      'got_strength' — synonym returned when only the strength leg completed a partial resume.
      'in_safari'  — the HM(s) are in the bag but the exit didn't complete (still inside; caller retries).
      'not_here'   — no strike applies from here (caller falls through to the general layer).
      'failed'     — strike aborted mid-way (caller surfaces / recovery reacts).
    Idempotent by state: already-having-both short-circuits to the walk-out; a partial run resumes by map."""
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in SAFARI_ANCHORS:
        return "not_here"
    ss = SafariStrike(camp, log, dbg_dir=dbg_dir)
    # STOCK + HEAL before a long Safari run (the tour has no Center between the pay-in and the HMs):
    # a worn lead gets swept by wilds mid-crossing (the recon_safari pre-crossing-heal lesson). Best-effort.
    if here == FUCHSIA:
        try:
            hr = camp.heal_nearest()
            log(f"   safari strike: pre-tour heal -> {hr} (at {tv.map_id(camp.b)}@{tv.coords(camp.b)})")
        except Exception as e:
            log(f"   safari strike: pre-tour heal errored ({e}) — entering as-is (LOUD)")
    return ss.run()
