"""snorlax_strike.py — WAKE THE ROUTE 12 SNORLAX, in-loop (night shift #7): the flute's payoff.

The questline walks her to the Snorlax's face on Route 12 but nothing wires "play the Poké Flute at
the sleeping blocker" — head_to_gym just no_paths into the body (the (12,0) SPLIT-MAP DEAD ROAD). The
ritual (KB frlg_gates + Bulbapedia, proven by the champion's recon_snorlax.py): pass the Route-12
NORTH GATE, face the SNORLAX object, A -> "...play the POKE FLUTE?" -> YES -> it wakes and ATTACKS
(a catchable wild L30) -> beat it -> FLAG_WOKE_UP_ROUTE_12_SNORLAX (0x253) sets and the body leaves the
road south to Fuchsia. Faithful in-loop port driven by `camp`, dispatched from _questline_strike for
the snorlax errand (step.success==('flag','FLAG_WOKE_UP_ROUTE_12_SNORLAX')). FireRed coords isolated
here (rule 14).
"""
import os

import field_moves as fm
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Route-12 Snorlax fact table (game-knowledge layer; rule 14) ──────────────────────────
ROUTE12 = (3, 30)
FLAG_WOKE = 0x253
SNORLAX = (14, 70)                  # disasm body tile; verified vs the live template list
GATE_DOORS = ((14, 15), (15, 15))   # Route-12 north gate: -> (23,0); exit its south door
ANCHORS = {ROUTE12}


class SnorlaxStrike:
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

    def woke(self):
        return fm.read_flag(self.b, FLAG_WOKE)

    def drain(self, max_a=40):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception:
            pass

    def wake_snorlax(self):
        """Pass the north gate, face the Snorlax, play the Flute, win the battle. Returns
        'woke_snorlax' | 'failed'."""
        b, camp, L = self.b, self.camp, self.log
        if self.woke():
            return "woke_snorlax"
        if tuple(tv.map_id(b)) != ROUTE12:
            return "failed"

        # 1. NORTH GATE: from the top of Route 12 (y<15) enter the gate (prefer='south' = approach
        #    y-1, step DOWN), cross, exit its south door onto the south road.
        if tuple(tv.coords(b))[1] < 15:
            m0 = tuple(tv.map_id(b))
            for door in GATE_DOORS:
                if camp.enter_warp(prefer="south", pick=door) == "warped" \
                        and tuple(tv.map_id(b)) != m0:
                    break
            if tuple(tv.map_id(b)) != m0:
                for _ in range(60):
                    b.run_frame()
                camp.enter_warp(prefer="south")
                for _ in range(60):
                    b.run_frame()
            L(f"   snorlax: past the gate -> {tv.map_id(b)}@{tv.coords(b)}")
            if tuple(tv.map_id(b)) != ROUTE12 or tuple(tv.coords(b))[1] < 16:
                self.snap("snorlax_gate_fail")
                L("!! snorlax: gate pass-through didn't land on the south road — abort")
                return "failed"

        # 2. find the Snorlax body (disasm (14,70), else nearest PRESENT template within 4)
        cur = tuple(tv.coords(b))
        present = {t for t, _g, p in tv.read_object_templates(b) if p}
        body = SNORLAX if SNORLAX in present else next(
            (t for t in sorted(present, key=lambda t: abs(t[0] - SNORLAX[0]) + abs(t[1] - SNORLAX[1]))
             if abs(t[0] - SNORLAX[0]) + abs(t[1] - SNORLAX[1]) <= 4), None)
        if body is None:
            L(f"!! snorlax: no present template near {SNORLAX} — already woken? flag={self.woke()}")
            return "woke_snorlax" if self.woke() else "failed"
        L(f"   SNORLAX body at {body} (she is at {cur})")

        # 3. face + A from each adjacent tile (nearest first): "...play the POKE FLUTE?" -> YES -> wake
        def adj_key(f):
            return {(1, 0): "LEFT", (-1, 0): "RIGHT", (0, -1): "DOWN", (0, 1): "UP"}[
                (f[0] - body[0], f[1] - body[1])]

        fronts = sorted(((body[0] + dx, body[1] + dy) for dx, dy in
                         ((0, -1), (0, 1), (1, 0), (-1, 0))),
                        key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        woke_battle = False
        for front in fronts:
            if tuple(tv.coords(b)) != front:
                r = camp.trav.travel(target_map=None, arrive_coord=front, max_steps=300,
                                     max_seconds=240)
                if st.in_battle(b):                    # fisherman gauntlet en route
                    L(f"   snorlax: battle en route -> {camp.battle_runner()}")
                    self.drain()
                    r = camp.trav.travel(target_map=None, arrive_coord=front, max_steps=300,
                                         max_seconds=240)
                if r != "arrived" and not camp._step_to(front):
                    continue
            key = adj_key(front)
            for _ in range(6):
                b.press(key, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(30):
                    b.run_frame()
                if dd_box(b):
                    self.drain()                    # "...play the POKE FLUTE?" -> A = YES -> music
                    for _ in range(400):            # the wake cutscene
                        b.run_frame()
                        if st.in_battle(b):
                            break
                    self.drain()
                if st.in_battle(b):
                    L("   SNORLAX WOKE — battle!")
                    out = camp.battle_runner()
                    L(f"   snorlax battle -> {out}")
                    self.drain()
                    woke_battle = True
                    break
            if woke_battle:
                break

        for _ in range(120):
            b.run_frame()
        gone = body not in {t for t, _g, p in tv.read_object_templates(b) if p}
        flag = self.woke()
        L(f"   snorlax: woke flag={flag} body_gone={gone} (battled={woke_battle})")
        self.snap("snorlax_final")
        return "woke_snorlax" if (flag or gone) else "failed"


def run_strike(camp, log, dbg_dir=None):
    """Wake the Route-12 Snorlax from Route 12, in ONE call. Returns:
      'woke_snorlax' — FLAG_WOKE set / body gone (the road south to Fuchsia is open).
      'not_here'     — no strike applies from here (caller falls through).
      'failed'       — aborted mid-way (caller surfaces / recovery reacts)."""
    b = camp.b
    ss = SnorlaxStrike(camp, log, dbg_dir=dbg_dir)
    if ss.woke():
        return "not_here"
    if tuple(tv.map_id(b)) not in ANCHORS:
        return "not_here"
    return ss.wake_snorlax()
