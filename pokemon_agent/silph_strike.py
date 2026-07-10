"""silph_strike.py — THE SILPH CO. LIBERATION STRIKE, in-loop (night shift #2, badge-6 build).

The general questline lands her ON Saffron's street walled at Sabrina's gym — the gym door is
Rocket-BLOCKED until Silph Co. is cleared (FLAG_HIDE_SAFFRON_ROCKETS 0x3E). The blind room-tour +
GO-DEEPER can NOT crack Silph: an 11-floor TELEPORT-PAD MAZE sits behind a Card-Key door gate.
The champion cleared it with the bespoke recon_silph.py strike; this is a FAITHFUL port of that
proven script, driven by the live `camp` so the general questline can call it as ONE decision (the
same shape as beat_gym / hideout_strike / tower_strike). FireRed coords are isolated here (rule 14 —
portability debt: a Kanto-Silph-Co fact table, swap per game).

Ground truth (disasm pret + the champion's proven recon_silph runs):
  Street door on Saffron (33,30) -> Silph 1F (1,47); the guard (34,31) is asleep once
  FLAG_RESCUED_MR_FUJI (0x23C) is set (she has it post-Tower). Maps: SilphCo_1F..11F = (1,47)..(1,57).
  Stairs pair adjacent floors (enter_to(next) needs no stair coord). 5F: CARD KEY ball (22,21) in a
  SEALED south pocket reachable ONLY via the 9F pad (22,18)->landing (10,20). With the key-pickup flag
  set, an A-press opens the card-locked BG 'sign' door tiles for good. PAD CHAIN: 9F (9,4)<->3F (2,14),
  3F (13,14)<->7F (5,4), 7F (5,8)<->11F (2,5). 7F west pocket (PAD-ONLY): Gary auto-rival at (2,4)/(2,5),
  free LAPRAS at (0,7). 9F hostage (2,16) = a FREE full heal (no flag). 11F: GIOVANNI (6,11) — beating
  him sets 0x3E (frees the city, opens Sabrina's gym); PRESIDENT (9,9) then hands the Master Ball.
Success = flag 0x3E. Exit via reverse pads + stairs -> Saffron street -> heal.
"""
import os

import field_moves as fm
import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Silph-Co fact table (game-knowledge layer; rule 14 portability debt) ────────────────
SAFFRON = (3, 10)
SILPH_DOOR = (33, 30)                             # Saffron street -> Silph 1F
SILPH = [(1, 47 + i) for i in range(11)]          # [0]=1F .. [10]=11F
F1, F3, F5, F7, F9, F11 = SILPH[0], SILPH[2], SILPH[4], SILPH[6], SILPH[8], SILPH[10]
SILPH_MAPS = set(SILPH) | {(1, 57), (1, 58)}      # incl. the elevator maps (never our route)
CARD_KEY_ITEM = 355
FLAG_SAFFRON_FREE = 0x3E                          # FLAG_HIDE_SAFFRON_ROCKETS — the strike's win
FLAG_CARD_KEY = 0x192                             # 5F ball picked up (doors check THIS)
FLAG_FUJI = 0x23C
CARD_BALL_5F = (22, 21)
DOORS_3F_WEST = [(9, 12), (10, 12), (9, 13), (10, 13)]
DOORS_3F_EAST = [(20, 12), (21, 12), (20, 13), (21, 13)]
DOORS_9F_WMID = [(13, 17), (12, 17), (13, 16), (12, 16)]   # unseals the heal woman
DOORS_9F_WEST = [(2, 10), (3, 10), (2, 11), (3, 11)]       # unseals the 3F-pad corridor
DOORS_11F = [(5, 16), (6, 16), (5, 17), (6, 17)]
GARY_TRIGGER_7F = (2, 5)
GARY_TRIGGERS_7F = {(2, 4), (2, 5)}              # NS3: coord-trigger tiles (disasm) — mask to skip Gary
LAPRAS_FRONT_7F = (0, 8)
LAPRAS_FLAG = 0x246
PAD_3F_TO_7F = (13, 14)
PAD_7F_TO_11F = (5, 8)
PAD_9F_TO_3F = (9, 4)
PAD_3F_TO_9F = (2, 14)
PAD_7F_TO_3F = (5, 4)
PAD_11F_TO_7F = (2, 5)
HEAL_9F_FRONT = (2, 17)                           # hostage woman (2,16), face UP
GIOVANNI_FRONT = (6, 12)                          # Giovanni (6,11), face UP
PRESIDENT_FRONT = (9, 10)                         # president (9,9), face UP


class SilphStrike:
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
        self.gary_done = False
        self.lapras_done = False
        # NS3: route-around DISABLED — proven geometrically IMPOSSIBLE. With Gary's trigger tiles
        # (2,4)/(2,5) masked, the approach to the 11F pad (5,8) fails from (5,5): the pad is ONLY
        # reachable THROUGH the trigger row (the champion's recon_silph fought Gary for the same reason).
        # So Gary is unavoidable — she must WIN. The real fix is the battle FINISH-THE-FOE guard
        # (battle_agent._maybe_use_item) that stops the potion-loop/PP-famine that lost the out-chipped
        # fight. Kept the route-around code below (disabled) as the documented negative result.
        self._gary_route_around = False
        self.heal_mode = False           # None once latched-off for the run
        self.wedges = {}

    # ── low-level helpers (ported from recon_silph.py) ──────────────────────────────────────────
    def fight(self):
        self.n_battles += 1
        return self.camp.battle_runner()

    def key_items(self):
        return ht.pocket_items(self.b, ht.KEY_ITEMS_OFF, 30)

    def saffron_free(self):
        return fm.read_flag(self.b, FLAG_SAFFRON_FREE)

    def have_key(self):
        return CARD_KEY_ITEM in self.key_items() or fm.read_flag(self.b, FLAG_CARD_KEY)

    def lead_frac(self):
        cur = self.b.rd16(ram.GPLAYER_PARTY + 0x56)
        mx = self.b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def drain(self, max_a=40, key="A"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
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

    def walk_path_to(self, tile, label, tries=6, avoid=None):
        """Deterministic same-map mover: static BFS with WARPS + template-NPC BODIES masked, stepped
        tile-by-tile. A step that fails OUTSIDE battle = a collision-walkable-but-game-blocked tile
        (elevation quirk) -> DEAD for this call (recon_silph strike truth). `avoid` = extra tiles to
        treat as blocked (NS3: Gary's COORD-trigger tiles (2,4)/(2,5), which are walkable floor but fire
        the rival battle — masking them lets the BFS route to the 11F pad WITHOUT engaging the rival)."""
        b, camp = self.b, self.camp
        dead = set()
        avoid = ({tuple(a) for a in avoid} - {tile}) if avoid else set()
        for _ in range(tries):
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]} | dead | avoid) - {tile}
            p = tv.bfs(g, cur, lambda t: t == tile,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            if not p:
                self.log(f"   [{label}] no NPC-free static path {cur} -> {tile} (dead={sorted(dead)})")
                return False
            for t in p[1:]:
                ok = camp._step_to(tuple(t))
                if st.in_battle(b):
                    self.log(f"   [{label}] battle mid-path -> {self.fight()}")
                    self.drain()
                    break
                if not ok:
                    for _ in range(120):
                        b.run_frame()
                    if dd_box(b):
                        self.drain()
                    if st.in_battle(b):
                        self.log(f"   [{label}] step was a trainer spotting -> {self.fight()}")
                        self.drain()
                        break
                    dead.add(tuple(t))
                    self.log(f"   [{label}] step into {tuple(t)} failed — dead-marked, recompute")
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return tuple(tv.coords(b) or ()) == tile

    def goto(self, tile, label):
        b, camp = self.b, self.camp
        if self.walk_path_to(tile, label):
            return True
        av = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120,
                             avoid=av)
        if st.in_battle(b):
            self.log(f"   [{label}] battle en route -> {self.fight()}")
            self.drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250,
                                 max_seconds=120, avoid=av)
        if r != "arrived" and not camp._step_to(tile):
            return False
        return tuple(tv.coords(b) or ()) == tile

    def engage(self, front, face, label, drains=1, key="A"):
        b, camp = self.b, self.camp
        if not self.goto(front, label):
            self.log(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if st.in_battle(b):
                self.log(f"   [{label}] battle -> {self.fight()}")
                self.drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    self.drain(key=key)
                    if st.in_battle(b):
                        self.log(f"   [{label}] battle -> {self.fight()}")
                        self.drain()
                        return "battled"
                    for _ in range(40):
                        b.run_frame()
                break
        return out

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

    def ride_pad(self, pad, label, avoid=None):
        """Ride a teleport pad: fires on STEP-ON contact; approach a free neighbor then one LONG-HOLD
        press onto the pad (turn+walk in one continuous input) (recon_silph strike11 truth). `avoid` =
        extra tiles to keep the approach BFS off (NS3: Gary's rival-trigger tiles) — masked in BOTH the
        neighbor-reachability probe and the walk so she can't be routed onto a trigger en route."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        g = tv.Grid(b)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        avoidset = ({tuple(a) for a in avoid} - {pad}) if avoid else set()
        blocked = npcs | avoidset
        cur0 = tuple(tv.coords(b) or (0, 0))
        cands = []
        for nb, kin in (((pad[0] - 1, pad[1]), "RIGHT"), ((pad[0] + 1, pad[1]), "LEFT"),
                        ((pad[0], pad[1] - 1), "DOWN"), ((pad[0], pad[1] + 1), "UP")):
            if nb in wts or not g.walkable(nb[0], nb[1]) or nb in avoidset:
                continue
            p = tv.bfs(g, cur0, lambda t, a=nb: t == a,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in blocked) \
                if cur0 != nb else [cur0]
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not self.walk_path_to(nb, f"{label}-approach", avoid=avoidset):
                continue
            for _try in range(3):
                b.press(kin, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(60):
                        b.run_frame()
                    self.log(f"   [{label}] rode pad {pad}: {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                    return True
            break
        self.log(f"!! [{label}] pad ride {pad} failed (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def open_doors(self, tiles, label):
        """A-press the card-locked door barrier tiles. With the key-pickup flag set the script opens
        the door permanently. Stand at a reachable orthogonal neighbor, face the tile, press."""
        b = self.b
        if not self.have_key():
            self.log(f"!! [{label}] no Card Key yet — doors won't open")
            return False
        opened = False
        grid = tv.Grid(b)
        cur = tuple(tv.coords(b) or (0, 0))
        for dt in tiles:
            for (nb, face) in (((dt[0] + 1, dt[1]), "LEFT"), ((dt[0] - 1, dt[1]), "RIGHT"),
                               ((dt[0], dt[1] + 1), "UP"), ((dt[0], dt[1] - 1), "DOWN")):
                if not tv.bfs(grid, cur, lambda t, a=nb: t == a, walkable=grid.walkable):
                    continue
                if not self.goto(nb, f"{label}-stand"):
                    continue
                b.press(face, 8, 10, self.camp.render, owner="agent")
                b.press("A", 8, 12, self.camp.render, owner="agent")
                for _ in range(40):
                    b.run_frame()
                if dd_box(b):
                    self.drain()
                    self.log(f"   [{label}] door script fired at {dt}")
                    opened = True
                    grid = tv.Grid(b)
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
            if opened:
                break
        return opened

    def heal_9f(self):
        """The 9F hostage (2,16) fully heals the party — free, no flag. Her room is card-sealed from
        the pad landing: WMID (12-13,16-17) is the key door."""
        b = self.b
        self.open_doors(DOORS_9F_WMID, "9f-heal-door")
        r = self.engage(HEAL_9F_FRONT, "UP", "9f-heal", drains=4)
        for _ in range(240):
            b.run_frame()
        self.drain()
        self.log(f"   9F hostage heal -> {r}; lead {self.lead_frac():.0%}")
        return self.lead_frac() > 0.9

    # ── the climb — a state machine on the current floor (ported from recon_silph.main) ──────────
    def climb(self, deadline):
        import time
        b, camp, L = self.b, self.camp, self.log
        while time.time() < deadline and not self.saffron_free():
            here = tuple(tv.map_id(b))
            if self.lead_frac() < 0.5 and self.heal_mode is not None:
                self.heal_mode = True
            if self.heal_mode and self.lead_frac() > 0.9:
                self.heal_mode = False
                self.wedges.pop("heal", None)
            if self.heal_mode:
                self.wedges["heal"] = self.wedges.get("heal", 0) + 1
                if self.wedges["heal"] > 12:
                    L("!! heal detour abandoned after 12 attempts — continuing the mission hurt")
                    self.heal_mode = None
                    continue
                cur = tuple(tv.coords(b) or (0, 0))
                if self.have_key() and here in SILPH:
                    if here == F9:
                        self.heal_9f()
                    elif here == F3:
                        self.open_doors(DOORS_3F_WEST if cur[0] < 15 else DOORS_3F_EAST, "heal-3f-door")
                        self.ride_pad(PAD_3F_TO_9F, "heal-to-9f") or self.drain()
                    elif here == F7 and cur[0] <= 6:
                        self.ride_pad(PAD_7F_TO_3F, "heal-to-3f") or self.drain()
                    elif here == F11:
                        self.ride_pad(PAD_11F_TO_7F, "heal-to-7f") or self.drain()
                    elif here == F5 and cur[1] >= 19:
                        self.ride_pad((10, 20), "heal-pocket-exit") or self.drain()
                    else:
                        i9 = SILPH.index(here)
                        self.enter_to(SILPH[i9 + (1 if i9 < 8 else -1)], "heal-stairs")
                    continue
                L(f"   lead at {self.lead_frac():.0%} — descending to heal (from {here})")
                if here in SILPH and here != F1:
                    self.enter_to(SILPH[SILPH.index(here) - 1], "heal-descent")
                elif here == F1:
                    self.enter_to(SAFFRON, "silph-exit")
                elif here == SAFFRON:
                    camp.heal_nearest()
                else:
                    camp.enter_warp(prefer="south")
                    for _ in range(80):
                        b.run_frame()
                continue
            if here == SAFFRON:
                if not self.enter_to(F1, "silph-reenter"):
                    self.snap("25_reenter_fail")
                    return "failed"
                continue
            if here not in SILPH:
                L(f"   off-route at {here} (heal interior?) — exiting to the overworld")
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
                if tuple(tv.map_id(b)) == here:
                    L(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                    self.snap("26_offroute")
                    return "failed"
                continue
            idx = SILPH.index(here)

            cur = tuple(tv.coords(b) or (0, 0))
            warp_tiles = {tuple(w[0]) for w in tv.read_warps(b)}
            if cur in warp_tiles:
                ent = camp._WARP_ENTRY.get(camp._tile_behavior(*cur))
                fire = ent[1] if ent else None
                g5 = tv.Grid(b)
                for nb, k in (((cur[0], cur[1] + 1), "DOWN"), ((cur[0] + 1, cur[1]), "RIGHT"),
                              ((cur[0] - 1, cur[1]), "LEFT"), ((cur[0], cur[1] - 1), "UP")):
                    if fire is not None and (nb[0] - cur[0], nb[1] - cur[1]) == fire:
                        continue
                    if nb not in warp_tiles and g5.walkable(nb[0], nb[1]):
                        b.press(k, 26, 10, camp.render, owner="agent")
                        for _ in range(20):
                            b.run_frame()
                        break
                if tuple(tv.map_id(b)) != here:
                    continue

            # PHASE A — no Card Key yet: climb to 9F, ride the pad into the 5F pocket, grab the ball.
            if not self.have_key():
                if here == F5 and cur[1] >= 19:
                    for face, front in (("RIGHT", (CARD_BALL_5F[0] - 1, CARD_BALL_5F[1])),
                                        ("LEFT", (CARD_BALL_5F[0] + 1, CARD_BALL_5F[1])),
                                        ("DOWN", (CARD_BALL_5F[0], CARD_BALL_5F[1] - 1)),
                                        ("UP", (CARD_BALL_5F[0], CARD_BALL_5F[1] + 1))):
                        if tuple(tv.map_id(b)) != F5:
                            break
                        self.engage(front, face, "card-key-ball")
                        if self.have_key():
                            break
                    L(f"   CARD KEY: item={CARD_KEY_ITEM in self.key_items()} "
                      f"flag={fm.read_flag(b, FLAG_CARD_KEY)}")
                    if not self.have_key():
                        if tuple(tv.map_id(b)) != F5 or tuple(tv.coords(b) or (0, 0))[1] < 19:
                            continue
                        self.snap("30_no_key")
                        L("!! Card Key not obtained in the 5F pocket — abort LOUD")
                        return "failed"
                    continue
                if here == SILPH[8]:
                    if not self.ride_pad((22, 18), "pad-to-pocket"):
                        self.wedges[here] = self.wedges.get(here, 0) + 1
                        if self.wedges[here] >= 3:
                            self.snap("29_no_pad_ride")
                            L("!! can't ride the 9F pad x3 — abort")
                            return "failed"
                        self.drain()
                    continue
                step_a = 1 if idx < 8 else -1
                nxt = SILPH[idx + step_a] if 0 <= idx + step_a < len(SILPH) else None
                if nxt and not self.enter_to(nxt, f"floor{idx + 1 + step_a}"):
                    if tuple(tv.map_id(b)) == here:
                        self.wedges[here] = self.wedges.get(here, 0) + 1
                        if self.wedges[here] >= 3:
                            self.snap(f"31_climb_wedge_{idx + 1}F")
                            L(f"!! climb wedged x3 on {idx + 1}F — abort")
                            return "failed"
                        self.drain()
                else:
                    self.wedges.pop(here, None)
                continue

            # PHASE B — key in hand: ride the pad chain to Giovanni (11F).
            if here == F7 and cur[0] <= 6:
                # ROUTE-AROUND GARY (night shift 3 — the whiteout-LOOP fix). Gary (object (2,6),
                # COORD-triggers (2,4)/(2,5), disasm-confirmed) is a BONUS rival whose ace (Charizard)
                # HARD-COUNTERS a solo Venusaur: it resists her all-Grass offense 0.25x while Fire /
                # Flying / Psychic across his 6-mon team all hit her 2x. A lone carry LOSES the gauntlet
                # and whiteout-LOOPS the entire tower (verified NS3: 6 losses / 6 full re-climbs, the
                # strike's internal deadline burned to a FAIL). The MISSION is GIOVANNI on 11F, not Gary;
                # Giovanni's Ground/Rock team is 2x-weak to Grass (trivial for Venusaur). The 3F->7F
                # landing (5,4) and the 11F pad (5,8) sit in the SAME column x=5 with Gary's triggers
                # off at x=2, so ride the 11F pad STRAIGHT from the landing — the shortest-path approach
                # runs down column 5 and never touches x=2, skipping the unwinnable rival AND the optional
                # Lapras. Fall back to clearing Gary/Lapras only if the pad is genuinely unreachable
                # (disasm says it isn't). A STRONGER team revisits Gary later — logged as a soul debt.
                if self._gary_route_around:
                    if self.ride_pad(PAD_7F_TO_11F, "pad-11f-direct", avoid=GARY_TRIGGERS_7F):
                        continue
                    if st.in_battle(b):
                        L(f"   GARY auto-engaged en route to the 11F pad -> {self.fight()}")
                        self.drain()
                        self.gary_done = True
                        continue
                    # direct pad unreachable without a fight — disable the shortcut, use the fallback
                    self.wedges["direct11f"] = self.wedges.get("direct11f", 0) + 1
                    if self.wedges["direct11f"] >= 2:
                        L("!! 11F pad not directly reachable x2 — falling back to the Gary-cleared route")
                        self._gary_route_around = False
                    self.drain()
                    continue
                if not self.gary_done:
                    L("   7F pocket: walking the rival trigger row (Gary auto-engages)")
                    nb0 = self.n_battles
                    ok = self.goto(GARY_TRIGGER_7F, "gary-trigger")
                    if st.in_battle(b):
                        L(f"   GARY -> {self.fight()}")
                    self.drain()
                    if self.n_battles > nb0 or ok:
                        self.gary_done = True
                        self.snap("40_post_gary")
                    else:
                        self.wedges["gary"] = self.wedges.get("gary", 0) + 1
                        if self.wedges["gary"] >= 3:
                            L("!! Gary trigger row unreachable x3 — proceeding to 11F LOUD")
                            self.gary_done = True
                    continue
                if not self.lapras_done:
                    r = self.engage(LAPRAS_FRONT_7F, "UP", "lapras", drains=8, key="B")
                    self.drain(key="B")
                    got = fm.read_flag(b, LAPRAS_FLAG)
                    if not got and not dd_box(b) and not st.in_battle(b):
                        try:
                            from naming import name_entry
                            name_entry(b, "", render=camp.render)
                            self.drain(key="B")
                            got = fm.read_flag(b, LAPRAS_FLAG)
                        except Exception as e:
                            L(f"   lapras naming escape skipped: {e}")
                    if r != "nothing" and got:
                        self.lapras_done = True
                        L(f"   LAPRAS banked -> {r} (flag={got})")
                    elif r != "nothing" or got:
                        self.lapras_done = True
                        L(f"   LAPRAS guy -> {r} (flag={got})")
                    else:
                        self.wedges["lapras"] = self.wedges.get("lapras", 0) + 1
                        if self.wedges["lapras"] >= 3:
                            L("!! Lapras guy unreachable x3 — proceeding LOUD (bonus, not mission)")
                            self.lapras_done = True
                    continue
                if not self.ride_pad(PAD_7F_TO_11F, "pad-11f"):
                    self.wedges[here] = self.wedges.get(here, 0) + 1
                    if self.wedges[here] >= 3:
                        self.snap("50_no_11f")
                        L("!! can't ride the 11F pad from the pocket x3 — abort")
                        return "failed"
                    self.drain()
                continue
            if here == F11:
                r = self.engage(GIOVANNI_FRONT, "UP", "giovanni", drains=3)
                if r == "nothing":
                    r = self.engage((7, 11), "LEFT", "giovanni-side", drains=3)
                if r == "nothing":
                    self.open_doors(DOORS_11F, "11f-door")
                    r = self.engage(GIOVANNI_FRONT, "UP", "giovanni-2", drains=3)
                L(f"   GIOVANNI -> {r}; saffron_free={self.saffron_free()}")
                self.drain()
                for _ in range(240):
                    b.run_frame()
                    if dd_box(b):
                        b.press("A", 8, 12, camp.render, owner="agent")
                if self.saffron_free():
                    self.snap("60_giovanni_down")
                    r2 = self.engage(PRESIDENT_FRONT, "UP", "president", drains=8)
                    self.drain()
                    L(f"   PRESIDENT -> {r2} (master ball flag={fm.read_flag(b, 0x250)})")
                    break
                self.wedges[here] = self.wedges.get(here, 0) + 1
                if self.wedges[here] >= 3:
                    self.snap("55_giovanni_wedge")
                    L("!! Giovanni not falling / not reachable x3 — abort")
                    return "failed"
                continue
            if here == F9:
                self.open_doors(DOORS_9F_WMID, "9f-wmid")
                if self.lead_frac() < 0.85:
                    self.heal_9f()
                self.open_doors(DOORS_9F_WEST, "9f-west")
                if not self.ride_pad(PAD_9F_TO_3F, "pad-3f"):
                    self.wedges[here] = self.wedges.get(here, 0) + 1
                    if self.wedges[here] >= 3:
                        self.snap("33_no_3f_pad")
                        L("!! can't ride the 9F->3F pad x3 — abort")
                        return "failed"
                    self.drain()
                continue
            if here == F3:
                doors = DOORS_3F_WEST if cur[0] < 15 else DOORS_3F_EAST
                if not self.ride_pad(PAD_3F_TO_7F, "pad-7f"):
                    self.open_doors(doors, "3f-doors")
                    if not self.ride_pad(PAD_3F_TO_7F, "pad-7f-2"):
                        self.wedges[here] = self.wedges.get(here, 0) + 1
                        if self.wedges[here] >= 3:
                            self.snap("35_no_7f")
                            L("!! can't reach the 7F pad from 3F — abort")
                            return "failed"
                        self.drain()
                continue
            if here == F5 and cur[1] >= 19:
                if not self.ride_pad((10, 20), "pocket-exit") and tuple(tv.map_id(b)) == F5:
                    self.wedges[here] = self.wedges.get(here, 0) + 1
                    if self.wedges[here] >= 3:
                        self.snap("32_pocket_stuck")
                        L("!! can't exit the 5F pocket x3 — abort")
                        return "failed"
                    self.drain()
                continue
            step = 1 if idx < 8 else -1
            if not self.enter_to(SILPH[idx + step], f"to9f-via{idx + 1 + step}F"):
                self.wedges[here] = self.wedges.get(here, 0) + 1
                if self.wedges[here] >= 3:
                    self.snap(f"36_route9f_wedge_{idx + 1}F")
                    L(f"!! routing to 9F wedged on {idx + 1}F — abort")
                    return "failed"
                self.drain()
            else:
                self.wedges.pop(here, None)
        return "freed" if self.saffron_free() else "timeout"

    def walk_out(self, deadline):
        """Reverse pads + stairs -> Saffron street, then heal (recon_silph section 3)."""
        import time
        b, L = self.b, self.log
        while tuple(tv.map_id(b)) != SAFFRON and time.time() < deadline:
            here = tuple(tv.map_id(b))
            if here == F11:
                self.enter_to(F7, "out-7f")
            elif here == F7:
                self.enter_to(F3, "out-3f")
            elif here in SILPH:
                i = SILPH.index(here)
                if here == F3:
                    self.open_doors(DOORS_3F_EAST, "out-3f-door")
                if i == 0:
                    self.enter_to(SAFFRON, "out-street")
                else:
                    self.enter_to(SILPH[i - 1], "out-down")
            else:
                self.camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
        if tuple(tv.map_id(b)) == SAFFRON:
            self.camp.heal_nearest()
            return True
        L(f"!! walk-out incomplete (at {tv.map_id(b)}) — flag holds; the loop's recovery owns the exit")
        return False


def run_strike(camp, log, dbg_dir=None):
    """Free Saffron by clearing Silph Co., from wherever she stands, in ONE call. Returns:
      'freed_saffron' — FLAG_HIDE_SAFFRON_ROCKETS set AND back out on the Saffron street (healed).
      'in_silph'      — the flag is set but the exit didn't complete (still inside; caller retries).
      'not_here'      — no strike applies from here (caller falls through to the general layer).
      'failed'        — strike aborted mid-way (caller surfaces / recovery reacts).
    Idempotent by state: already-freed short-circuits; a partial run resumes by map."""
    import time
    b = camp.b
    ss = SilphStrike(camp, log, dbg_dir=dbg_dir)
    here = tuple(tv.map_id(b))

    # already freed the city: just get out (or already out)
    if ss.saffron_free():
        if here == SAFFRON:
            return "not_here"
        if here in SILPH_MAPS:
            return "freed_saffron" if ss.walk_out(time.time() + 420) else "in_silph"
        return "not_here"

    if here == SAFFRON:
        # HEAL TO FULL first (HP+PP): there is NO Center above Giovanni's boss fight; a worn lead
        # gets swept in the tower (the hideout-strike lesson). Then enter Silph via the street door.
        try:
            frac = ss.lead_frac()
        except Exception:
            frac = 0.0
        if frac < 0.95:
            hr = camp.heal_nearest()
            log(f"   silph strike: pre-dungeon heal (lead {frac:.0%}) -> {hr} "
                f"(now {tv.map_id(b)}@{tv.coords(b)})")
        if fm.read_flag(b, FLAG_FUJI) == 0:
            log("   silph strike: FLAG_RESCUED_MR_FUJI unset — the Silph door guard may block; "
                "attempting entry anyway (LOUD)")
        if not ss.enter_to(F1, "silph-door"):
            ss.snap("10_no_silph")
            log("!! silph strike: couldn't enter Silph Co. from Saffron — abort")
            return "failed"
        here = tuple(tv.map_id(b))

    if here not in SILPH_MAPS:
        return "not_here"

    # inside the tower — run the climb, then walk out
    r = ss.climb(time.time() + 1500)
    if not ss.saffron_free():
        log(f"   silph strike: climb -> {r}; Saffron NOT freed (at {tv.map_id(b)}@{tv.coords(b)})")
        return "failed"
    log(f"   silph strike: SAFFRON FREED (flag 0x3E) — walking out to heal + bank")
    ss.snap("80_freed")
    return "freed_saffron" if ss.walk_out(time.time() + 420) else "in_silph"
