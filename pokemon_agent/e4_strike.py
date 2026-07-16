"""e4_strike.py — THE LAST VEHICLE, in-loop: League mart stock-up + Elite Four + Champion -> CREDITS.

A FAITHFUL port of the proven recon_e4.py vehicle into an in-loop module driven by the live `camp` bridge,
so the endgame push can call it as ONE decision (the same shape as victory_road / giovanni_gym / blaine_gym).
Entering the Hall of Fame is the point of no return: the game auto-saves and the credits roll.

Ground truth (pret; recon_e4's champion-clear-proven constants, verbatim):
- IndigoPlateau exterior (3,9): the League center door is the only warp -> center (13,0).
- League center: clerk stand (2,7) FACE LEFT to shop; nurse heals; League door (4,1) -> Lorelei's room.
- Mart rows (0-based): 2 FULL RESTORE(19), 4 REVIVE(24), 5 FULL HEAL(23) — the true-index engine
  (row+scroll @0x02039940/42), each unit verified by money drop + bag qty (camp._mart_buy_one).
- E4 rooms are ONE TEMPLATE: arrive south warp, trainer talk-triggered, north door OPENS on the
  FLAG_DEFEATED_* the win sets. Chain: Lorelei -> Bruno -> Agatha -> Lance -> Champion (Gary) ->
  HALL OF FAME -> CREDITS. Whiteout = respawn at THIS center (last heal); DEFEATED flags persist so
  cleared rooms pass straight through -> the dispatch loop re-heals/re-shops and re-enters.

WHITEOUT-TOLERANT + resume-safe: the loop keys on the CURRENT map each iteration (exterior/center/room/
HoF) so a mid-gauntlet whiteout costs a re-lap of the UNCLEARED rooms, never solved ground; a resume from
disk anywhere on the League maps picks up in place. Battles run through camp.battle_runner() (the same
battle-brain that cracked the E4 wall on the champion climb — commit 23487e7: never sleep-lock a 2x-SE foe;
field the super-effective specialist), and the item instinct takes the offered heal/cure/revive.

run_strike returns:
  'credits'     — the Hall of Fame reached; auto-save fired, the credits are rolling. (THE SUMMIT.)
  'battle_loss' — a persistent whiteout the caller's recovery should own (rare; the loop self-recovers most).
  'stuck'       — the deadline hit without the Hall of Fame (a real wall — usually team-depth: a thin team
                  can't out-attrition Lance/Gary). Surfaces LOUD so the caller can grind + retry.
"""
import os
import time

import travel as tv
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box

# ── FireRed Indigo-Plateau / League fact table (game-knowledge layer; rule 14 portability debt) ─────────
INDIGO_EXT = (3, 9)                       # the Indigo Plateau exterior
LEAGUE_CENTER = (13, 0)                   # the League Pokémon Center (KB ground truth, learned live e4_run3)
CLERK_STAND = (2, 7)                      # stand here, FACE LEFT, A to shop
LEAGUE_DOOR = (4, 1)                      # the center's door into Lorelei's room
FLAG_BADGE_EARTH = 0x827                  # badge 8 — the strike's preflight guard
FULL_RESTORE, MAX_POTION, REVIVE, FULL_HEAL = 19, 20, 24, 23
# (item id, mart true-index row, want, unit price) — FR-first tuning (recon_e4 runs 5-9 postmortem: every
# lap died at Lance with FR x0; 4 FR tanks the 5-dragon wave, 2 Revive for the type-answer comeback, 1 Full
# Heal for the Jynx/Hypnosis sleep). The stock_up scaler clamps each line to money (poverty-safe order).
SHOPPING = [(FULL_RESTORE, 2, 4, 3000), (REVIVE, 4, 2, 1500), (FULL_HEAL, 5, 1, 600)]
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}


class EliteFour:
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
        # Whiteout laps compound the ace's XP; give the gauntlet room to converge (recon_e4 ran a 4h window,
        # but in-loop this is ONE decision — bound it so a genuinely thin team returns 'stuck' to grind).
        self.deadline = time.time() + int(os.getenv("POKEMON_E4_DEADLINE_S", "5400"))

    # ── snap / battle / dialogue drains (victory_road/giovanni shape) ────────────────────────────────────
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
        # pointer validity + LIVENESS (gMain.callback2) — a stale res_ptr after a whiteout is a corpse, not a
        # fight (the run3 phantom re-attach livelock).
        return (ram.valid_ewram_ptr(self.b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(self.b))

    def drain(self, max_n=60, key="B"):
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

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def party_alive(self):
        b = self.b
        n = 0
        for i in range(6):
            base = ram.GPLAYER_PARTY + i * 100
            if b.rd16(base + 0x56) > 0 and b.rd16(base + 0x58) > 0:
                n += 1
        return n

    def live_npc_tiles(self):
        b = self.b
        OB, SZ = 0x02036E38, 0x24
        out = []
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.append((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                        b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    # ── interior walk / warp (plain — the League maps have no water/boulders) ────────────────────────────
    def walk(self, goal_test, label, tries=12, allow=()):
        b, camp = self.b, self.camp
        budget = tries
        frozen, last_pos = 0, None
        map_start = tuple(tv.map_id(b))
        while budget > 0:
            budget -= 1
            if self.handle_interrupts():
                budget += 1
                continue
            if tuple(tv.map_id(b)) != map_start:
                self.log(f"   [{label}] map changed {map_start} -> {tuple(tv.map_id(b))} — bail")
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            # FREEZE ARMOR (e4_run1: post-whiteout she planned len-20 paths for 22 replans without moving —
            # a hidden modal box eating inputs dd_box can't see). 3 no-move replans -> blind B/A drain.
            if cur == last_pos:
                frozen += 1
                if frozen >= 3:
                    self.log(f"!! [{label}] FROZEN at {cur} x{frozen} replans — hidden box? blind B/A drain")
                    self.snap(f"frozen_{label[:12]}")
                    for k in ("B", "B", "A", "B"):
                        b.press(k, 8, 12, camp.render, owner="agent")
                        for _ in range(30):
                            b.run_frame()
                    self.drain()
                    frozen = 0
                    budget += 1
            else:
                frozen = 0
            last_pos = cur
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = set(self.live_npc_tiles())
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            self.log(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                self.log(f"   [{label}] no path from {cur}")
                self.snap(f"nopath_{label[:12]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if self.handle_interrupts():
                    budget += 1
                    break
                if not camp._step_to(tuple(t)):
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
        return goal_test(tuple(tv.coords(b) or ()))

    def go_warp(self, tile, label):
        """Walk to a neighbor of `tile`, step in; True on any map change."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        for _attempt in range(4):
            cur = tuple(tv.coords(b) or ())
            if cur != tile and cur not in nbs:
                if not self.walk(lambda c, s=set(nbs): c in s, f"{label}-approach", allow=(tile,)):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            for _press in range(4):
                if key:
                    b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) != m0:
                self.settle(180)
                self.log(f"   [{label}] {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        self.snap(f"warpfail_{label[:16]}")
        return False

    # ── the mart engine (tm_errand's true-index machinery, Items-pocket verify) ──────────────────────────
    def shop_index(self):
        return self.b.rd16(0x02039940) + self.b.rd16(0x02039942)

    def list_live(self):
        b, camp = self.b, self.camp
        c0 = self.shop_index()
        b.press("DOWN", 8, 10, camp.render, owner="agent")
        for _ in range(20):
            b.run_frame()
        if self.shop_index() != c0:
            return True
        b.press("UP", 8, 10, camp.render, owner="agent")
        for _ in range(12):
            b.run_frame()
        return False

    def shop_goto_index(self, target, tries=20):
        b, camp = self.b, self.camp
        for _ in range(tries):
            idx = self.shop_index()
            if idx == target:
                for _ in range(20):
                    b.run_frame()
                if self.shop_index() == target:
                    return True
                continue
            b.press("DOWN" if idx < target else "UP", 8, 10, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
        return self.shop_index() == target

    def stock_up(self):
        b, camp, L = self.b, self.camp, self.log
        need = [(iid, row, want - camp.bag_count(iid), price)
                for iid, row, want, price in SHOPPING
                if camp.bag_count(iid) < want]
        money = camp.money()
        # COMEBACK FLOOR (ns22): reserve 1 Revive (+1 Full Heal) BEFORE Full Restores eat a
        # poor budget. At a realistic ~$13k arrival, FR-first spent the whole wad and bought
        # ZERO revives -> the TYPE-ANSWER revive (the Charizard/Gary counter) had no item to
        # use and the leveled Lapras stayed fainted (ns22 e4_v2: died at Gary, revive_item=None).
        # 1 Revive + 1 Full Heal ($2.1k) still leaves ~3 Full Restores at $13k (the Lance tank
        # wave survives on 3), while guaranteeing the champion answer can actually fire.
        FLOOR = {REVIVE: 1, FULL_HEAL: 1}
        alloc = {iid: 0 for iid, _r, _n, _p in need}
        for iid, row, n, price in need:                       # pass 1: comeback floor first
            f = min(n, FLOOR.get(iid, 0), max(0, money // price))
            alloc[iid] += f
            money -= f * price
        for iid, row, n, price in need:                       # pass 2: FR (SHOPPING order) takes the surplus
            extra = min(n - alloc[iid], max(0, money // price))
            alloc[iid] += extra
            money -= extra * price
        plan = [(iid, row, alloc[iid], price) for iid, row, n, price in need if alloc[iid] > 0]
        if not plan:
            L(f"   [shop] {'kit SHORT but broke' if need else 'stocked already'} (money ${camp.money()})")
            return True
        L(f"   [shop] plan {plan} (money ${camp.money()})")
        if not self.walk(lambda c: c == CLERK_STAND, "clerk-approach"):
            return False
        opened = False
        for _ in range(8):
            b.press("LEFT", 8, 8, camp.render, owner="agent")
            b.press("A", 8, 10, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
                if dd_box(b):
                    opened = True
                    break
            if opened:
                break
        if not opened:
            L(f"!! [shop] clerk never opened a dialog (coords {tv.coords(b)})")
            self.snap("shop_no_greeting")
            return False
        stable = 0
        for _ in range(30):
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    break
                for _ in range(30):
                    b.run_frame()
        if not self.list_live():
            b.press("A", 8, 10, camp.render, owner="agent")   # BUY (top of BUY/SELL)
            for _ in range(120):
                b.run_frame()
            if not self.list_live():
                L("!! [shop] BUY list didn't confirm — abort shop")
                self.snap("shop_entry_fail")
                return False
        for iid, row, n, price in plan:
            for _u in range(n):
                q0 = camp.bag_count(iid)
                if not self.shop_goto_index(row) or self.shop_index() != row:
                    L(f"!! [shop] couldn't hold index {row} — abort shop")
                    return False
                got = camp._mart_buy_one()
                q1 = camp.bag_count(iid)
                if got <= 0 or got > price + 500 or q1 != q0 + 1:
                    L(f"!! [shop] buy-verify FAILED item {iid} (price={got}, bag x{q0}->x{q1}) — abort shop")
                    self.snap("shop_buy_fail")
                    return False
            L(f"   [shop] bought {n} x item {iid} (bag x{camp.bag_count(iid)}, money ${camp.money()})")
        for _ in range(8):
            b.press("B", 6, 12, camp.render, owner="agent")
            for _ in range(14):
                b.run_frame()
        self.drain()
        try:
            camp._save_campaign("e4_shopped")
        except Exception:
            pass
        return True

    # ── the dispatch loop: exterior -> center (heal+shop) -> door chain -> CREDITS ───────────────────────
    def run(self):
        b, camp, L = self.b, self.camp, self.log
        try:
            if not fm.read_flag(b, FLAG_BADGE_EARTH):
                L("!! [e4] badge 8 not held — not the endgame; abort")
                return "stuck"
        except Exception:
            return "stuck"
        L(f"[e4] boot map={tv.map_id(b)} coords={tv.coords(b)} lead={self.lead_frac():.0%} "
          f"alive={self.party_alive()} money=${camp.money()} FR x{camp.bag_count(FULL_RESTORE)}")
        center = LEAGUE_CENTER
        shopped = False
        healed = False
        seen_rooms = []               # map ids in door-chain order (Lorelei..Champion)
        prev_here = None
        while time.time() < self.deadline:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            came_from, prev_here = prev_here, here
            if here == center and seen_rooms and came_from is not None \
                    and came_from not in (center, INDIGO_EXT):
                # back at the center FROM the chain = a whiteout — respawn healed the party but the Full
                # Restore kit is drained: re-check the shop (money-aware; no-op if the bag is still stocked).
                L(f"   [e4] back at the center from the chain (whiteout) — re-checking the kit "
                  f"[money ${camp.money()}, FR x{camp.bag_count(FULL_RESTORE)}]")
                shopped = False
                healed = False
            warps = [(tuple(xy), tuple(d)) for xy, d, _w in tv.read_warps(b)]
            if here == INDIGO_EXT:
                if not warps:
                    self.settle(60)
                    continue
                tgt = min(warps, key=lambda w: abs(w[0][0] - 11))[0]
                if not self.go_warp(tgt, "enter-center"):
                    L("!! [e4] can't enter the League center — stuck")
                    return "stuck"
            elif here == center:
                if not healed:
                    r = camp.heal_nearest()
                    L(f"   [e4] heal_nearest -> {r} (lead {self.lead_frac():.0%})")
                    self.drain()
                    healed = self.lead_frac() > 0.99
                    continue
                if not shopped:
                    if not self.stock_up():
                        L("!! [e4] shopping failed — proceeding with what's aboard (LOUD)")
                    shopped = True
                    continue
                if not self.go_warp(LEAGUE_DOOR, "league-door"):
                    L("!! [e4] League door failed — stuck")
                    return "stuck"
            elif here != center:
                # inside the door chain: an E4 room, the Champion's room, or the HoF
                if here not in seen_rooms:
                    seen_rooms.append(here)
                    L(f"   [e4] room #{len(seen_rooms)}: map {here} @ {tv.coords(b)} "
                      f"[lead {self.lead_frac():.0%}, alive {self.party_alive()}]")
                    self.snap(f"room{len(seen_rooms)}_enter")
                if len(seen_rooms) >= 6:
                    # room #6 past the champion = HALL OF FAME — the credits are rolling
                    L("   *** [e4] HALL OF FAME — CREDITS INBOUND ***")
                    self.snap("hall_of_fame")
                    self.on_event_safe("...that's it. that's the whole thing. eight badges, the Elite Four, "
                                       "the Champion — I actually did it. we did it.", tier=2)
                    break
                npcs = self.live_npc_tiles()
                room_warps = sorted([t for t, _d in warps], key=lambda t: t[1])
                north = room_warps[0] if room_warps else (6, 2)
                if npcs:
                    trainer = min(npcs, key=lambda t: t[1])
                    stand = (trainer[0], trainer[1] + 1)
                    if tuple(tv.coords(b) or ()) != stand:
                        if not self.walk(lambda c, s=stand: c == s, "trainer-approach"):
                            self.settle(180)
                            self.drain()
                            continue
                    fought = False
                    for _try in range(3):
                        b.press("UP", 8, 8, camp.render, owner="agent")
                        b.press("A", 8, 12, camp.render, owner="agent")
                        for _ in range(120):
                            b.run_frame()
                            if self.fight_open():
                                break
                        if self.fight_open():
                            L(f"   [e4] battle #{len(seen_rooms)} OPENS [lead {self.lead_frac():.0%}, "
                              f"alive {self.party_alive()}, FR x{camp.bag_count(FULL_RESTORE)}]")
                            self.fight()
                            self.drain()
                            fought = True
                            break
                        self.drain(key="A")
                        if not dd_box(b) and not self.fight_open():
                            break
                    for _ in range(30):
                        if self.handle_interrupts():
                            continue
                        if tuple(tv.map_id(b)) != here:
                            break
                        self.settle(30)
                        if not dd_box(b) and not self.fight_open():
                            break
                    if tuple(tv.map_id(b)) != here:
                        continue
                    if fought:
                        try:
                            camp._save_campaign(f"e4_room{len(seen_rooms)}")
                        except Exception:
                            pass
                if not self.go_warp(north, "north-door"):
                    if self.party_alive() == 0 or self.lead_frac() == 0:
                        L("   [e4] whiteout state — loop recovers via the center")
                    self.settle(120)
            else:
                L(f"   [e4] off-route at {here} — exiting")
                camp.enter_warp(prefer="south")
                self.settle(80)
        if len(seen_rooms) < 6:
            L(f"!! [e4] deadline without the Hall of Fame (rooms {len(seen_rooms)}/5, battles {self.n_battles}) "
              f"— usually team-depth: a thin team can't out-attrition Lance/Gary. Grind + retry.")
            return "stuck"
        # ── CREDITS: drain the Hall of Fame scene + let the credits roll ─────────────────────────────────
        L("   [e4] draining the Hall of Fame scene + credits (A every ~2s)")
        t_cred = time.time()
        last_map = tuple(tv.map_id(b))
        while time.time() - t_cred < 600:
            for _ in range(120):
                b.run_frame()
            if dd_box(b):
                b.press("A", 8, 12, camp.render, owner="agent")
            m = tuple(tv.map_id(b))
            if m != last_map:
                L(f"   [e4] [credits] map {last_map} -> {m} at +{time.time() - t_cred:.0f}s")
                last_map = m
        self.snap("99_post_credits")
        L(f"   [e4] CREDITS SEQUENCE DRAINED | battles {self.n_battles} | money ${camp.money()}")
        return "credits"

    def on_event_safe(self, msg, tier=2):
        try:
            self.camp.on_event(msg, kind="milestone", tier=tier)
        except Exception:
            pass


def run_strike(camp, log, dbg_dir=None):
    """Run the Elite-Four gauntlet (Indigo Plateau -> Lorelei..Champion -> Hall of Fame -> CREDITS) from
    wherever she stands on the League maps, in ONE call. Whiteout-tolerant + resume-safe (map-keyed dispatch;
    DEFEATED flags ratchet in the save). Returns 'credits' (the summit) | 'battle_loss' | 'stuck'."""
    return EliteFour(camp, log, dbg_dir).run()
