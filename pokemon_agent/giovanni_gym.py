"""giovanni_gym.py — THE VIRIDIAN GYM STRIKE (Giovanni / badge 8), in-loop (night shift).

A FAITHFUL port of the proven recon_giovanni.py strike into an in-loop module driven by the live `camp`
bridge, so the general questline can call it as ONE decision (the same shape as beat_gym / blaine_gym /
mansion_strike / safari_strike / seafoam_strike). FireRed coords are isolated here (rule 14 — portability
debt: a Viridian-gym fact table + the Cinnabar->Viridian sea road, swap per game).

Ground truth (pret maps/scripts, cached ViridianGym*.json/.inc):
- THE ROAD: Cinnabar -> Viridian is FIVE consecutive NORTH edge crossings
  (Cinnabar -> R21South -> R21North -> Pallet -> Route1 -> Viridian), sea legs surfed
  (Lapras has SURF; recon_seafoam's mount/sea_walk/cross_edge machinery verbatim).
- THE DOOR: ViridianCity OnTransition unlocks the gym when badges 2-7 are held (she has 1-7) — the
  locked-door coord event (36,11) dies on her first city transition. Gym warp (36,10) -> gym map (5,1).
- THE GYM: no doors/quizzes — a SPIN-TILE floor maze (the Rocket-Hideout class; spin_nav.SpinNav = the
  proven glide crosser, wired here for its second customer). 8 juniors WITH sight 2-3 (spotting battles
  are expected and fine — Razor Leaf is x2 into Giovanni's ground/rock rosters). Giovanni (2,2) face
  DOWN -> front (2,3), face UP.
- Post-win: FLAG_DEFEATED_LEADER_GIOVANNI + BADGE 8 = flag 0x827 + TM26 gift (A-drain, NO Y/N) +
  Giovanni removeobject fade. NO EXIT AMBUSH (the badge-8 Gary fight arms on ROUTE 22, westbound — the
  NEXT objective's problem, NOT this strike's). Success = flag 0x827.

Resume-safe: badge already held -> 'badge'; already INSIDE the gym (beat_gym enters via enter_warp before
calling the strike) -> SKIP the sea road, straight to the spin-maze + Giovanni; started off-route (e.g.
Cinnabar smoke) -> run the road (five surfed north crossings) to Viridian, enter, then the gym.

run_gym returns one of the beat_gym contract strings: 'badge' | 'needs_heal' | 'battle_loss' | 'stuck'.
"""
import os
import time

import travel as tv
import pokemon_state as st
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box
from spin_nav import SpinNav

# ── FireRed Viridian-Gym + sea-road fact table (game-knowledge layer; rule 14 portability debt) ──────
CINNABAR = (3, 8)
VIRIDIAN = (3, 1)
GYM = (5, 1)
GIOVANNI_FRONT = (2, 3)              # Giovanni (2,2), face UP
FLAG_BADGE_EARTH = 0x827
FLAG_DEFEATED_GIOVANNI = 0x4B7       # FLAG_DEFEATED_LEADER_GIOVANNI (unused if wrong; badge is truth)
GYM_EXIT_BAND = {(16, 21), (17, 21), (18, 21)}   # above the entrance mats (16-18,22)

KEY_OF = {(0, 1): "DOWN", (0, -1): "UP", (1, 0): "RIGHT", (-1, 0): "LEFT"}
DIRN_OF = {"south": 1, "north": 2, "west": 3, "east": 4}


class GiovanniGym:
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
        self.deadline = time.time() + 1800
        # the proven spin-tile glide crosser (Rocket-Hideout class), wired for its Viridian customer
        self.nav = SpinNav(self.b, self.camp, self.fight, self.drain, log=self.log)

    # ── snap / battle / dialogue drains ──────────────────────────────────────────────────────────────
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

    # ── water/edge machinery (recon_seafoam verbatim, via recon_giovanni's road closures) ────────────
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

    def sea_walk(self, goal_test, label, tries=10, avoid=()):
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
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            npcs = self.live_npc_tiles() | {tuple(o[0]) for o in
                                            tv.read_object_templates(b) if o[2]}
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
                for _ in range(120):
                    b.run_frame()
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not self.sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                for _ in range(120):
                    b.run_frame()
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
                    for _ in range(120):
                        b.run_frame()
                    self.log(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} "
                             f"@ {tv.coords(b)}")
                    return True
        self.log(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
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
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                self.log(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):
                self.log(f"   [{label}] battle on approach -> {self.fight()}")
                self.drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    # ── recon_giovanni tour closures ─────────────────────────────────────────────────────────────────
    def badge(self):
        return fm.read_flag(self.b, FLAG_BADGE_EARTH)

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    # ── the strike ───────────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        self.log(f"   giovanni strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"badge8={self.badge()} lead={self.lead_frac():.0%} "
                 f"badge7={fm.read_flag(b, 0x826)}")
        if self.badge():
            self.log("   Earth Badge already held — nothing to strike")
            return "badge"

        # ── PHASE A: THE SEA ROAD HOME (five north crossings) ────────────────────
        # Resume-safe: no-ops when already INSIDE the gym (beat_gym enters via enter_warp before calling
        # us) OR already at Viridian — the loop condition + the `here == GYM` break skip the road entirely.
        legs = 0
        while time.time() < self.deadline and tuple(tv.map_id(b)) != VIRIDIAN:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            if here == GYM:
                break                                    # already there (in-gym resume) — skip the road
            if not self.cross_edge("north", f"leg{legs}"):
                self.wedges["road"] = self.wedges.get("road", 0) + 1
                if self.wedges["road"] >= 3:
                    self.snap("10_road_wedge")
                    self.log(f"!! northbound crossing wedged x3 at {here}@{tv.coords(b)} — abort")
                    return "stuck"
                self.settle(120)
                continue
            self.wedges.pop("road", None)
            legs += 1
            self.settle(180)
        if tuple(tv.map_id(b)) == VIRIDIAN:
            self.log(f"   VIRIDIAN reached after {legs} legs, {self.n_battles} battles — healing first")
            camp.heal_nearest()

        # ── PHASE B: THE GYM (spin maze -> Giovanni) ─────────────────────────────
        while time.time() < self.deadline and not self.badge():
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            if here == VIRIDIAN:
                if self.lead_frac() < 0.6:
                    self.log(f"   lead at {self.lead_frac():.0%} — healing at the Viridian Center first")
                    camp.heal_nearest()
                    continue
                if not self.enter_to(GYM, "gym-door"):
                    self.wedges["door"] = self.wedges.get("door", 0) + 1
                    if self.wedges["door"] >= 3:
                        self.snap("20_no_gym_door")
                        self.log("!! can't enter the gym x3 — abort LOUD")
                        return "stuck"
                    self.drain()
                continue
            if here != GYM:
                self.log(f"   off-route at {here} (whiteout?) — exiting to the overworld")
                camp.enter_warp(prefer="south")
                self.settle(80)
                if tuple(tv.map_id(b)) == here:
                    self.log(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                    self.snap("21_offroute")
                    return "stuck"
                continue

            # hurt mid-gym: leave, heal, come back (beaten juniors stay beaten on reload)
            if self.lead_frac() < 0.5:
                self.wedges["heal_exit"] = self.wedges.get("heal_exit", 0) + 1
                if self.wedges["heal_exit"] >= 3:
                    self.snap("30_heal_exhaust")
                    self.log("   left the gym to heal x3 without a win — needs_heal (beaten juniors stay beaten)")
                    return "needs_heal"
                self.log(f"   lead at {self.lead_frac():.0%} — leaving to heal (beaten trainers stay beaten)")
                if self.nav.cross(lambda c: c in GYM_EXIT_BAND, "heal-exit"):
                    self.enter_to(VIRIDIAN, "exit-to-heal")
                continue

            self.log(f"   gym floor — spin-crossing to Giovanni's front {GIOVANNI_FRONT} "
                     f"[lead {self.lead_frac():.0%}]")
            if not self.nav.cross(lambda c: c == GIOVANNI_FRONT, "to-giovanni"):
                self.wedges["cross"] = self.wedges.get("cross", 0) + 1
                if self.wedges["cross"] >= 4:
                    self.snap("30_cross_wedge")
                    self.log(f"!! spin crosser never reached Giovanni x4 (at {tv.coords(b)}) — abort")
                    return "stuck"
                # THE BEATEN-BODY SEAL (recon run1, (10,5)): a sight-walking trainer STOPS adjacent to
                # engage and his body STAYS there after losing — in a 1-wide corridor that seals the walk
                # route. The game's own reset: objects respawn at TEMPLATE positions on map reload (beaten
                # trainers stay beaten) — exit + re-enter, then replan on a clean floor.
                self.log("   cross blocked — beaten-body seal suspected: exit + re-enter to reset "
                         "object positions to templates")
                self.snap(f"31_seal_{self.wedges['cross']}")
                if self.nav.cross(lambda c: c in GYM_EXIT_BAND, "seal-reset-out"):
                    self.enter_to(VIRIDIAN, "seal-reset")
                else:
                    self.drain()
                continue

            # PRE-LEADER HEAL (NS5 solo-carry attrition fix): the ~6 spin-floor juniors chip the lone
            # Venusaur to near-0% DURING the cross, so she arrived at Giovanni too weak and was KO'd
            # first-turn before her in-battle heal could fire (run1 loss at 2% -> whiteout). Giovanni's
            # Ground/Rock team is x2 into Razor Leaf — a full-HP Venusaur sweeps — so top up BEFORE
            # engaging. Beaten juniors stay beaten (object templates on reload), so the re-cross after
            # healing does NOT re-chip her; she reaches Giovanni's front fresh next pass.
            if self.lead_frac() < 0.85:
                self.log(f"   at Giovanni's front but lead={self.lead_frac():.0%} — the juniors chipped "
                         f"the lone carry; backing out to heal before the leader (beaten juniors stay beaten)")
                if self.nav.cross(lambda c: c in GYM_EXIT_BAND, "preleader-heal-exit"):
                    self.enter_to(VIRIDIAN, "preleader-heal")
                    # DEAD-ZONE FIX (NS#13, the L69-carry livelock): the preleader gate fires at <0.85 but
                    # the road/loop-top heal gate is <0.6, so a lead in the 0.60-0.85 band exits to Viridian
                    # and NEVER heals -> re-enter -> re-exit forever. (recon_giovanni's L90 champion carry
                    # never dropped below 0.85 so this never surfaced.) Heal UNCONDITIONALLY here so she
                    # re-enters full and reaches Giovanni's front fresh (>0.85), then engages.
                    try:
                        self.camp.heal_nearest()
                    except Exception as e:
                        self.log(f"   preleader heal errored: {e} — continuing (LOUD)")
                else:
                    self.drain()
                continue

            self.log(f"   Giovanni's front reached — engaging [lead {self.lead_frac():.0%}]")
            nb0 = self.n_battles
            r = "nothing"
            for _ in range(8):
                b.press("UP", 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(30):
                    b.run_frame()
                if self.fight_open():
                    self.log(f"   [giovanni] battle -> {self.fight()}")
                    self.drain(key="A")
                    r = "battled"
                    break
                if dd_box(b):
                    self.drain(key="A")
                    if self.fight_open():
                        self.log(f"   [giovanni] battle -> {self.fight()}")
                        self.drain(key="A")
                        r = "battled"
                        break
                    r = "talked"
            for _ in range(300):                        # badge fanfare + TM26 + his exit fade
                b.run_frame()
                if dd_box(b):
                    b.press("A", 8, 12, camp.render, owner="agent")
            self.drain(key="A")
            self.log(f"   GIOVANNI -> {r} (battles {nb0}->{self.n_battles}) badge8={self.badge()} "
                     f"lead={self.lead_frac():.0%}")
            if self.badge():
                self.snap("40_badge8")
                break
            if tuple(tv.map_id(b)) != GYM:
                # map left the gym interior with no badge -> she whited out to the city PC (a leader loss).
                # Propagate so the caller's blackout-recovery respawns + re-runs the gym (beaten juniors
                # stay beaten). Matches beat_gym/blaine_gym's leader-loss -> 'battle_loss'.
                self.log("   not in the gym after Giovanni (whiteout) — leader loss; caller retries the gym")
                self.snap("55_gio_blackout")
                return "battle_loss"
            self.wedges["gio"] = self.wedges.get("gio", 0) + 1
            if self.wedges["gio"] >= 3:
                self.snap("50_gio_wedge")
                self.log("!! Giovanni engaged x3 without a badge — abort LOUD")
                return "stuck"

        if not self.badge():
            self.log(f"!! Earth Badge NOT won (at {tv.map_id(b)}@{tv.coords(b)}) — deadline/exhausted")
            self.snap("70_fail")
            return "stuck"

        return self._walk_out()

    def _walk_out(self):
        """Post-badge: WALK OUT to Viridian + heal, then report the badge. NO exit ambush drain — unlike
        Blaine's Bill Sevii-sail, Giovanni has none (the badge-8 Gary fight arms on Route 22, westbound —
        the next objective's problem). If the walk-out is incomplete, the badge still holds; recovery owns
        the exit."""
        b, camp = self.b, self.camp
        self.log("   badge in hand — walking out (no ambush; Gary arms on Route 22, not here)")
        out_deadline = time.time() + 300
        while tuple(tv.map_id(b)) != VIRIDIAN and time.time() < out_deadline:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            if here == GYM:
                if self.nav.cross(lambda c: c in GYM_EXIT_BAND, "walk-out"):
                    self.enter_to(VIRIDIAN, "out-door")
                else:
                    self.drain()
            else:
                camp.enter_warp(prefer="south")
                self.settle(80)
        if tuple(tv.map_id(b)) == VIRIDIAN:
            try:
                camp.heal_nearest()
            except Exception as e:
                self.log(f"   post-badge heal errored: {e} — continuing (LOUD)")
        else:
            self.log(f"!! walk-out incomplete (at {tv.map_id(b)}) — the badge holds; recovery owns the exit")

        self.log(f"   EARTH BADGE: flag={self.badge()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
                 f"lead {self.lead_frac():.0%} | battles {self.n_battles}")
        self.snap("80_final")
        return "badge"


def run_gym(camp, log, dbg_dir=None):
    """Run the Viridian-Gym (Giovanni / badge 8) strike from wherever she stands, in ONE call. Resume-safe:
    already inside the gym -> straight to the spin maze; off-route -> the five surfed north crossings
    (Cinnabar -> Viridian) first. Spin-nav crosses the spin-tile floor maze to Giovanni (beating the
    sight-juniors that spot her en route), beats Giovanni, drives the badge/TM26 award, WALKS OUT and
    heals (NO exit ambush — the Gary fight arms on Route 22). Returns one of the beat_gym contract strings:
      'badge'       — Giovanni beaten, FLAG_BADGE_EARTH (0x827) set, healed, back at Viridian. (Success.)
      'needs_heal'  — left the gym to heal x3 without a win (juniors/spin-floor progress stays done).
      'battle_loss' — blacked out mid-gym (map left the gym interior after engaging, no badge); caller
                      recovers + re-runs the gym.
      'stuck'       — road wedged x3 / can't enter the gym x3 / spin-crosser wedged x4 / Giovanni engaged
                      x3 without a badge / off-route stuck / deadline."""
    return GiovanniGym(camp, log, dbg_dir).run()
