"""spin_nav.py — THE SPIN-TILE SLIDE CROSSER (general engine asset, extracted from
recon_hideout.py after the Silph Scope strike proved it end-to-end, 2026-07-07).

Ground truth (pret metatile_behaviors.h + hideout3/10/11 live derivation):
  0x54-0x57 = MB_SPIN_RIGHT/LEFT/UP/DOWN — stepping on one starts a forced SLIDE; each
  spinner crossed REDIRECTS it; 0x58 (MB_STOP_SPINNING) and walls stop it; PLAIN FLOOR
  DOES NOT stop it (momentum carries); a press while standing ON a wall-stopped spinner
  resumes that spinner's own direction. NPCs and item balls block glides — and the live
  object array is DISTANCE-CULLED, so planning uses travel.read_object_templates (the
  B2F Moon-Stone class). A maze can be SEALED by a collectible item ball: the crosser
  grabs reachable balls and replans (also: free loot — what a human does).

Travel's BFS treats spinners as plain floor, so its plans diverge the instant she touches
one (the position-loop class). Here: simulate every glide deterministically, BFS over REST
tiles with glide edges (plain steps are 1-tile edges), execute press-by-press, replan on
divergence/battles. Used by: recon_hideout.py (Rocket Hideout B2F/B3F). Ahead: Viridian
Gym's spin maze rides this too — wire into travel/campaign when a longrun bites it.
"""
import travel as tv
import pokemon_state as st
from dialogue_drive import box_open as _box_open

SPIN = {0x54: (1, 0), 0x55: (-1, 0), 0x56: (0, -1), 0x57: (0, 1)}
GFX_ITEM_BALL = 0x5C            # verified live on B2F vs disasm (recon_objtpl, 2026-07-07)


class SpinNav:
    """One instance per (bridge, campaign) pair. `fight` runs a battle to completion,
    `drain` advances/closes dialogue boxes, `log` prints."""

    def __init__(self, b, camp, fight, drain, log=lambda m: print(m, flush=True)):
        self.b, self.camp, self.fight, self.drain, self.L = b, camp, fight, drain, log

    def _ball_tiles(self):
        return {t for t, gfx, present in tv.read_object_templates(self.b)
                if present and gfx == GFX_ITEM_BALL}

    def _collect_ball(self, label):
        """Glide to a rest tile adjacent to a reachable PRESENT item ball, face it, A,
        verify the template flag flipped. Returns True iff a ball left the map."""
        b, camp = self.b, self.camp
        balls = self._ball_tiles()
        if not balls:
            return False

        def adj(t):
            return any(abs(t[0] - bx) + abs(t[1] - by) == 1 for bx, by in balls)

        for k in range(2):
            r = self._cross_once(adj, f"{label}-ball#{k + 1}")
            cur = tuple(tv.coords(b) or ())
            if r is True or (cur and adj(cur)):
                break
        else:
            return False
        cur = tuple(tv.coords(b))
        ball = next((bb for bb in balls
                     if abs(cur[0] - bb[0]) + abs(cur[1] - bb[1]) == 1), None)
        if ball is None:
            return False
        key = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, -1): "UP", (0, 1): "DOWN"}[
            (ball[0] - cur[0], ball[1] - cur[1])]
        for _ in range(3):
            b.press(key, 8, 10, camp.render, owner="agent")     # turn toward it (tile is blocked)
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            self.drain()
            if ball not in self._ball_tiles():
                self.L(f"   [{label}] ITEM BALL at {ball} collected — topology changed, replanning")
                return True
        return False

    def cross(self, target_pred, label, rounds=3, node_ok=None):
        grabbed = 0
        fail_counts = {}
        avoid = set()          # step tiles that failed twice — wanderer squatting (exit1 grunt)
        for _round in range(rounds + 6):
            r = self._cross_once(target_pred, f"{label}#{_round + 1}", node_ok=node_ok,
                                 avoid=avoid)
            if r is True:
                return True
            if tuple(tv.coords(self.b) or ()) and target_pred(tuple(tv.coords(self.b))):
                return True
            if isinstance(r, tuple) and r[0] == "blocked_step":
                t = r[1]
                fail_counts[t] = fail_counts.get(t, 0) + 1
                if fail_counts[t] >= 2:
                    avoid.add(t)
                    self.L(f"   [{label}] step tile {t} failed twice — avoiding it in replans")
                continue
            if r == "no_route":
                if grabbed < 6 and self._collect_ball(label):
                    grabbed += 1
                    continue
                if avoid:      # maybe the avoid set sealed the only route — relax it once
                    self.L(f"   [{label}] no route with avoid={sorted(avoid)} — relaxing")
                    avoid.clear()
                    continue
                return False
        return False

    def _cross_once(self, target_pred, label, max_hops=60, node_ok=None, avoid=None):
        b, camp, L = self.b, self.camp, self.L
        g = tv.Grid(b)
        npc = set(camp.trav._npc_tiles())      # live NPCs block glides (a grunt IS a wall here)
        # + templates of DISTANCE-CULLED objects ONLY (the far-item-ball class). The old
        # blanket union also kept every LIVE object's SPAWN tile as a wall — a beaten LoS
        # trainer standing where he stopped then blocks TWO tiles, and that phantom sealed
        # the Viridian 12-spinner maze (banked_BLAINE WARN, shift 12: "no route from (10,5)").
        npc |= tv.culled_template_tiles(b)
        npc |= set(avoid or ())                # twice-failed step tiles (wanderer squatting)
        # playable dims (BACKUP_LAYOUT includes the +14 border) — Grid reads WRAP at the edges,
        # which planned a LEFT glide off x=0 (hideout7)
        w_play = b.rd32(tv.BACKUP_LAYOUT) - 14
        h_play = b.rd32(tv.BACKUP_LAYOUT + 4) - 14

        def in_bounds(x, y):
            return 0 <= x < w_play and 0 <= y < h_play

        def bh(t):
            return camp._tile_behavior(*t)

        def glide(frm, d):
            """One press from `frm` moving `d` — the exact slide mechanics (module docstring)."""
            x, y = frm
            dx, dy = d
            sliding = False
            v0 = bh(frm)
            if v0 in SPIN:
                dx, dy = SPIN[v0]
                sliding = True
            for _ in range(300):
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny) or not g.walkable(nx, ny) or (nx, ny) in npc:
                    return (x, y) if (x, y) != frm else None
                x, y = nx, ny
                v = bh((x, y))
                if v in SPIN:
                    dx, dy = SPIN[v]
                    sliding = True
                    continue
                if v == 0x58 or not sliding:
                    return (x, y)
                # plain floor mid-slide: momentum carries — keep going
            return (x, y)

        last = tuple(tv.coords(b))         # settle any in-flight slide before planning
        still = 0
        for _ in range(300):
            b.run_frame()
            cur = tuple(tv.coords(b))
            if cur == last:
                still += 1
                if still >= 40:
                    break
            else:
                last, still = cur, 0
        # Spotting-freeze race (shift 12, banked_BLAINE): an LoS trainer's approach eats the
        # step presses ("press UP never moved"), and the NEXT round can start planning while
        # the battle is opening — BFS over frozen coords reads "no route". Fight it first.
        if st.in_battle(b):
            L(f"   [{label}] battle open before planning -> {self.fight()}")
            self.drain()
            for _ in range(60):
                b.run_frame()
        start = tuple(tv.coords(b))
        from collections import deque
        prev = {start: None}
        q = deque([start])
        goal = None
        while q:
            cur = q.popleft()
            if target_pred(cur):
                goal = cur
                break
            for key, d in (("RIGHT", (1, 0)), ("LEFT", (-1, 0)), ("UP", (0, -1)), ("DOWN", (0, 1))):
                dst = glide(cur, d)
                if dst and dst not in prev and (node_ok is None or node_ok(dst)):
                    prev[dst] = (cur, key)
                    q.append(dst)
        if goal is None:
            L(f"!! [{label}] spin-BFS found no route from {start}")
            return "no_route"           # truthy sentinel — callers must test `is True`
        plan = []
        n = goal
        while prev[n] is not None:
            p, k = prev[n]
            plan.append((k, n))
            n = p
        plan.reverse()
        L(f"   [{label}] spin plan: {plan}")
        for key, expect in plan[:max_hops]:
            ok = False
            for _ in range(4):                     # first press can be an eaten TURN
                if _box_open(b):                   # an OPEN BOX eats every direction press
                    self.L(f"   [{label}] dialogue box open mid-cross — draining "
                           f"(beaten-grunt re-talk class, exit1)")
                    self.drain()
                c0 = tuple(tv.coords(b))
                b.press(key, 8, 10, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                # wait for the glide to SETTLE (coords stable across a beat)
                last = tuple(tv.coords(b))
                still = 0
                for _ in range(240):
                    b.run_frame()
                    cur = tuple(tv.coords(b))
                    if cur == last:
                        still += 1
                        if still >= 30:
                            break
                    else:
                        last, still = cur, 0
                if st.in_battle(b):                # LoS trainer mid-route: fight, then REPLAN
                    L(f"   [{label}] battle mid-cross -> {self.fight()}")
                    self.drain()
                    return False
                if last == expect:
                    ok = True
                    break
                if last != c0:                     # moved but NOT where simulated — replan
                    L(f"!! [{label}] glide diverged: pressed {key} at {c0}, "
                      f"expected {expect}, got {last}")
                    return False
            if not ok:
                L(f"!! [{label}] press {key} never moved off {tuple(tv.coords(b))} "
                  f"(toward {expect}) — settling, then replanning")
                for _ in range(90):                # give a wandering NPC time to clear the tile
                    b.run_frame()
                return ("blocked_step", expect)
        return True
