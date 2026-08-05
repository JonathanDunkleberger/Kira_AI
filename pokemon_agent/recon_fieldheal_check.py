"""OFFLINE smoke check for the FIELD-HEAL DOCTRINE (2026-08-05, the Mt. Ember climb:
'she is not healing outside of battle when she probably should') — no emulator, no ROM.

Stubs the bridge + the TeachFlow bag rails and drives the pure doctrine logic:
  1. _cheapest_adequate_heal walks the ladder cheapest-first (a Potion on an 18-point chip,
     a Hyper on a 144-point crater), falls back to the biggest bottle present when nothing
     covers fully, NEVER wastes a Full Restore while a Potion covers, and reads None on an
     empty pocket;
  2. _field_heal_pick: ace under 50% heals, bench under 35% heals, the ace OUTRANKS the
     bench when both are low, a healthy party picks nothing, a FAINTED mon is skipped
     (potions are refused on corpses), and the TOP-UP mode raises the ace's bar to ~95%
     for the pre-legendary seam;
  3. empty-pocket honest skip: a hurt ace + zero bottles logs [fieldheal] LOUD and returns
     None — never a wedge, never a TeachFlow launch;
  4. field_heal_check end-to-end (stubbed TeachFlow): heals ace-then-bench in one seam,
     bounded by FIELDHEAL_MAX_PER_SEAM; a failed bag-drive latches the 10-min backoff
     (no re-launch); the in-battle / open-box / kill-switch guards all stand down;
  5. STATUS RIDE-ALONG: a poisoned climber routes through the existing field_cure flow
     from the same seam (the strike loop has no roam tick to catch it);
  6. the PRE-LEGENDARY TOP-UP: press_quarry fires field_heal_seam(top_up=True) before the
     static A-press (driven through the real LegendaryHunt method);
  7. the _lap_restock_balls potion RIDE-ALONG: a thin heal pocket adds the shelf's best
     potion row to the SAME buy list (balls still row 0); a stocked pocket adds nothing;
     a bag-read flake degrades to balls-only (best-effort, never blocks the restock);
  8. WORLD-BACK POSTCONDITION (2026-08-05, the Mt. Ember bag wedge): _confirm_world_back
     Bs until gMain.callback2 reads the overworld again (bounded), hammers extra Bs LOUD
     when it won't, and fails OPEN on an unreadable cb2;
  9. the REAL TeachFlow.field_heal end-to-end on scripted rails: a verified heal whose
     menu never closes returns 'menu_stuck' (never a silent leak) and the campaign books
     it as a failure (backoff latched); the same rails with the callback restored return
     'healed';
 10. the WATCHDOG side: _sweep_stray_menus close-confirm requires pixels clear AND cb2
     back ('closed'/'stuck'/None), and _disengage_step1 closes a classified stray menu +
     RELEASES the phantom wedge marks of the frozen window INSTEAD of wedge-marking;
 11. the STRIKE-LEG guard: GiovanniGym.handle_interrupts treats a closed stray menu as a
     handled interrupt (one cb2 read gates it; overworld cb2 never sweeps);
 12. THE START-MENU BLIND SPOT (2026-08-05 #2, the EXIT-cursor wedge): START runs UNDER
     CB2_Overworld — ram.start_menu_open (gTasks scan for Task_StartMenuHandleInput) sees
     it where cb2 + bag/party pixels are all blind; wired through _stray_menu_kind, the
     sweep, the disengage rung, the strike-leg guard, and _confirm_world_back;
 13. the UNIVERSAL B-FIRST rung: a drift-calibrated screen-change probe closes menus no
     classifier knows yet (2 bounded Bs), skipping the phantom wedge-mark; ambient tile
     animation alone never reads as a menu.
Run:  python3 recon_fieldheal_check.py   (from pokemon_agent/) — prints PASS/FAIL per check.
"""
import sys
import time
import types

# Mac dev box has no mgba/emulator stack — stub it so campaign imports (logic-only test).
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import campaign as C
import dialogue_drive as dd
import firered_ram as ram
import giovanni_gym as GG
import hm_teach as ht
import legendary_strikes as LS
import pokemon_state as st

RealTeachFlow = ht.TeachFlow      # captured BEFORE main() swaps in the fake (sections 8-9)

PASS = []
WORLD = {}          # the ONE mutable world every patched reader closes over
LOGS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def set_world(party=(), bag=None, **kw):
    """party = list of dicts {hp, mx, lv, status}; bag = {item_id: qty}."""
    WORLD.clear()
    WORLD.update({"party": [dict(m) for m in party], "bag": dict(bag or {}),
                  "in_battle": False, "box": False, "heal_result": "healed"})
    WORLD.update(kw)
    FakeTeachFlow.calls.clear()
    LOGS.clear()


def make_b():
    def rd8(a):
        if a == ram.GPLAYER_PARTY_CNT:
            return len(WORLD["party"])
        off = a - ram.GPLAYER_PARTY
        if off >= 0 and off % st.PARTY_MON_SIZE == 0x54:
            s = off // st.PARTY_MON_SIZE
            if s < len(WORLD["party"]):
                return WORLD["party"][s].get("lv", 10)
        return 0

    def rd16(a):
        off = a - ram.GPLAYER_PARTY
        if off >= 0:
            s, r = divmod(off, st.PARTY_MON_SIZE)
            if s < len(WORLD["party"]):
                if r == C.P_HP:
                    return WORLD["party"][s]["hp"]
                if r == C.P_MAXHP:
                    return WORLD["party"][s]["mx"]
        return 0

    def rd32(a):
        off = a - ram.GPLAYER_PARTY
        if off >= 0:
            s, r = divmod(off, st.PARTY_MON_SIZE)
            if s < len(WORLD["party"]) and r == C.P_STATUS:
                return WORLD["party"][s].get("status", 0)
        return 0

    return types.SimpleNamespace(rd8=rd8, rd16=rd16, rd32=rd32)


def make_camp():
    camp = C.Campaign.__new__(C.Campaign)
    camp.b = make_b()
    camp.on_event = lambda *a, **k: None
    camp.bag_count = lambda iid: WORLD["bag"].get(iid, 0)
    return camp


class FakeTeachFlow:
    """Stands in for the proven bag rails: applies the heal/cure to WORLD and records."""
    calls = []

    def __init__(self, camp, log=None, on_event=None):
        self.camp = camp

    def field_heal(self, item_id, mon_slot):
        FakeTeachFlow.calls.append(("heal", item_id, mon_slot))
        res = WORLD.get("heal_result", "healed")
        if res == "healed":
            WORLD["party"][mon_slot]["hp"] = WORLD["party"][mon_slot]["mx"]
            WORLD["bag"][item_id] = WORLD["bag"].get(item_id, 0) - 1
        return res

    def field_cure(self, item_id, mon_slot):
        FakeTeachFlow.calls.append(("cure", item_id, mon_slot))
        WORLD["party"][mon_slot]["status"] = 0
        WORLD["bag"][item_id] = WORLD["bag"].get(item_id, 0) - 1
        return "cured"


# canonical live party: Blastoise L61 ace (244 max) + Lapras L25 (130 max)
BLASTOISE = dict(hp=244, mx=244, lv=61)
LAPRAS = dict(hp=130, mx=130, lv=25)

_CB2_MENU = 0x08107EE0 | 1        # a bag-menu callback — anything not overworld/whiteout


class _BlackPx:
    """frame_rgb().load() stand-in: every pixel black -> _classify never reads party/bag."""

    def __getitem__(self, xy):
        return (0, 0, 0)


class RailsBridge:
    """Scripted bridge for the REAL TeachFlow (sections 8-9). Simulates just enough rails:
    the START cursor (DOWN/UP navigable), the bag pocket byte (LEFT/RIGHT), the party HP
    reads, and gMain.callback2 — which flips back to the overworld only after
    `world_after_b` total B presses (None = the menu NEVER closes, the live wedge)."""

    def __init__(self, heal_after_a=6, world_after_b=None, cb2_raises=False):
        self.a = self.bs = 0
        self.start_cursor, self.pocket = 5, 1
        self.heal_after_a, self.world_after_b = heal_after_a, world_after_b
        self.cb2_raises = cb2_raises
        self.healed = False

    def set_input_owner(self, owner):
        pass

    def run_frame(self):
        pass

    def frame_rgb(self):
        return types.SimpleNamespace(load=lambda: _BlackPx())

    def press(self, key, hold, rel, render, owner=None):
        if key == "A":
            self.a += 1
            if self.a >= self.heal_after_a and not self.healed and WORLD["party"]:
                self.healed = True                      # the drink lands: HP up, bottle gone
                WORLD["party"][0]["hp"] = WORLD["party"][0]["mx"]
                WORLD["bag"][21] = max(0, WORLD["bag"].get(21, 0) - 1)
        elif key == "B":
            self.bs += 1
        elif key == "DOWN":
            self.start_cursor = min(6, self.start_cursor + 1)
        elif key == "UP":
            self.start_cursor = max(0, self.start_cursor - 1)
        elif key == "RIGHT":
            self.pocket = min(2, self.pocket + 1)
        elif key == "LEFT":
            self.pocket = max(0, self.pocket - 1)

    def rd8(self, a):
        if a == ht.START_CURSOR:
            return self.start_cursor
        if a == ht.BAG_POCKET:
            return self.pocket
        return 0

    def rd16(self, a):
        off = a - ram.GPLAYER_PARTY
        if off >= 0:
            s, r = divmod(off, st.PARTY_MON_SIZE)
            if s < len(WORLD["party"]):
                if r == C.P_HP:
                    return WORLD["party"][s]["hp"]
                if r == C.P_MAXHP:
                    return WORLD["party"][s]["mx"]
        return 0

    def rd32(self, a):
        if a == ram.GMAIN_CB2:
            if self.cb2_raises:
                raise RuntimeError("cb2 flake")
            if self.world_after_b is not None and self.bs >= self.world_after_b:
                return ram._CB2_OVERWORLD
            return _CB2_MENU
        return 0


def make_real_tf(bridge):
    tf = RealTeachFlow.__new__(RealTeachFlow)
    tf.b = bridge
    tf.c = types.SimpleNamespace(render=lambda: None, b=bridge)
    tf.log = lambda s: LOGS.append(str(s))
    tf.emit = lambda *a, **k: None
    return tf


class StartBridge:
    """THE START-WEDGE SHAPE (section 12): gMain.callback2 reads OVERWORLD the whole time
    (the blind spot), while Task_StartMenuHandleInput sits alive in gTasks slot 3 until
    `closes_after` B presses (None = never closes; 0 = menu not open)."""

    _SLOT = ram.GTASKS + 3 * ram.TASK_SIZE

    def __init__(self, closes_after=1, active=True):
        self.closes_after, self.active, self.bs = closes_after, active, 0

    def _menu_up(self):
        return self.closes_after is None or self.bs < self.closes_after

    def press(self, key, hold, rel, render, owner=None):
        if key == "B":
            self.bs += 1

    def run_frame(self):
        pass

    def set_input_owner(self, owner):
        pass

    def frame_rgb(self):
        return types.SimpleNamespace(load=lambda: _BlackPx())

    def rd8(self, a):
        if a == self._SLOT + 4 and self._menu_up():
            return 1 if self.active else 0
        return 0

    def rd16(self, a):
        return 0

    def rd32(self, a):
        if a == ram.GMAIN_CB2:
            return ram._CB2_OVERWORLD
        if a == self._SLOT and self._menu_up():
            return 0x0806F1F0 | 1              # Task_StartMenuHandleInput, thumb bit
        return 0


class DriftScreen:
    """B-first probe rig (section 13): a uniform white 'menu' that one B flips to a uniform
    green 'world' — or a menu-less frozen world where B does nothing. `drift_cols` sampled
    columns animate on a 20-frame cadence like water tiles (the ambient-drift calibration)."""

    def __init__(self, menu=True, drift_cols=0):
        self.menu, self.tick = menu, 0
        self.cols = (20, 60, 100, 140, 180, 220)[:drift_cols]

    def press(self, key, hold, rel, render, owner=None):
        if key == "B" and self.menu:
            self.menu = False

    def run_frame(self):
        self.tick += 1

    def frame_rgb(self):
        menu, tick, cols = self.menu, self.tick, self.cols

        class _P:
            def __getitem__(_s, xy):
                if xy[0] in cols:
                    return (0, 0, 0) if (tick // 20) % 2 else (200, 200, 200)
                return (255, 255, 255) if menu else (40, 120, 60)

        return types.SimpleNamespace(load=lambda: _P())


def make_start_camp(bridge):
    """Campaign rig with the REAL _stray_menu_kind/_sweep_stray_menus over a StartBridge."""
    camp = C.Campaign.__new__(C.Campaign)
    camp.b = bridge
    camp.render = lambda: None
    camp.on_event = lambda *a, **k: None
    return camp


def make_menu_camp(bs_to_close):
    """Campaign rig for the sweep/disengage checks: a stray bag screen that closes after
    `bs_to_close` B presses (None = never), with cb2 tracking the menu state."""
    camp = C.Campaign.__new__(C.Campaign)
    state = {"left": bs_to_close}

    def press(key, hold, rel, render, owner=None):
        if key == "B" and state["left"] is not None and state["left"] > 0:
            state["left"] -= 1

    def rd32(a):
        if a == ram.GMAIN_CB2:
            return ram._CB2_OVERWORLD if state["left"] == 0 else _CB2_MENU
        return 0

    camp.b = types.SimpleNamespace(press=press, run_frame=lambda: None, rd32=rd32,
                                   rd8=lambda a: 0, rd16=lambda a: 0)
    camp.render = lambda: None
    camp.on_event = lambda *a, **k: None
    camp._stray_menu_kind = lambda: "bag/case" if state["left"] != 0 else None
    camp._menu_state = state
    return camp


def main():
    # one-time reader patches, all closing over WORLD
    ht.TeachFlow = FakeTeachFlow
    st.read_party_species = lambda b, slot=0: 9 if slot == 0 else 131
    st.in_battle = lambda b: WORLD.get("in_battle", False)
    C.dd_box_open = lambda b: WORLD.get("box", False)
    _oldlog = C.log
    C.log = lambda s: LOGS.append(str(s)) or _oldlog(s)

    print("== 1. cheapest-adequate ladder ==")
    set_world(bag={13: 5, 22: 4, 21: 3})
    camp = make_camp()
    check("144 missing -> Hyper (Super's 50 can't cover)",
          camp._cheapest_adequate_heal(144) == 21)
    check("18 missing -> plain Potion (never burn a big bottle on a chip)",
          camp._cheapest_adequate_heal(18) == 13)
    check("60 missing -> Hyper (smallest tier that covers)",
          camp._cheapest_adequate_heal(60) == 21)
    WORLD["bag"] = {19: 2}
    check("nothing covers? -> biggest bottle present (Full Restore as last resort)",
          camp._cheapest_adequate_heal(20) == 19)
    WORLD["bag"] = {13: 1, 19: 2}
    check("a Potion beside a Full Restore -> the Potion takes the 20-point chip",
          camp._cheapest_adequate_heal(20) == 13)
    WORLD["bag"] = {}
    check("empty pocket -> None", camp._cheapest_adequate_heal(50) is None)

    print("== 2. picker thresholds (ace 50% / bench 35% / top-up 95%) ==")
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3})
    camp2 = make_camp()
    pick = camp2._field_heal_pick()
    check("ace at 41% -> picked, Hyper sized to the 144-point hole",
          pick == (0, 100, 244, 21))
    set_world(party=[dict(BLASTOISE, hp=150), dict(LAPRAS, hp=30)], bag={13: 3, 22: 3})
    check("ace at 61% holds; Lapras at 23% -> bench heal (Super = biggest present)",
          make_camp()._field_heal_pick() == (1, 30, 130, 22))
    set_world(party=[dict(BLASTOISE, hp=80), dict(LAPRAS, hp=20)], bag={21: 5})
    check("both low -> the ACE outranks the bench",
          make_camp()._field_heal_pick()[0] == 0)
    set_world(party=[dict(BLASTOISE, hp=200), dict(LAPRAS, hp=80)], bag={21: 5})
    check("healthy party (82%/61%) -> nothing to do",
          make_camp()._field_heal_pick() is None)
    set_world(party=[dict(BLASTOISE, hp=200), dict(LAPRAS, hp=0)], bag={21: 5})
    check("a FAINTED mon is skipped (potions refuse corpses; Revive/Center owns it)",
          make_camp()._field_heal_pick() is None)
    set_world(party=[dict(BLASTOISE, hp=200), LAPRAS], bag={21: 5})
    campT = make_camp()
    check("TOP-UP: 82% ace holds normally but heals under the pre-legendary bar",
          campT._field_heal_pick() is None
          and campT._field_heal_pick(top_up=True) == (0, 200, 244, 21))
    set_world(party=[BLASTOISE, LAPRAS], bag={21: 5})
    check("TOP-UP at full HP -> nothing (never a wasted press)",
          make_camp()._field_heal_pick(top_up=True) is None)

    print("== 3. empty-pocket honest skip ==")
    set_world(party=[dict(BLASTOISE, hp=60), LAPRAS], bag={})
    camp3 = make_camp()
    n3 = camp3.field_heal_check(reason="strike")
    check("hurt ace + zero bottles -> honest skip, LOUD, no bag launch",
          n3 == 0 and not FakeTeachFlow.calls
          and any("heal pocket is EMPTY" in ln for ln in LOGS))

    print("== 4. field_heal_check end-to-end (stubbed bag rails) ==")
    set_world(party=[dict(BLASTOISE, hp=100), dict(LAPRAS, hp=30)], bag={21: 3, 22: 2})
    camp4 = make_camp()
    n4 = camp4.field_heal_check(reason="strike")
    check("one seam heals ace FIRST then the bench (2 drinks, right sizes: Lapras "
          "missing 100 -> Hyper, Super's 50 can't cover)",
          n4 == 2 and [c for c in FakeTeachFlow.calls if c[0] == "heal"]
          == [("heal", 21, 0), ("heal", 21, 1)])
    check("...and both stand at full afterward",
          WORLD["party"][0]["hp"] == 244 and WORLD["party"][1]["hp"] == 130)
    check("no swallowed exception in the seam",
          not any("check skipped" in ln for ln in LOGS))
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3},
              heal_result="failed")
    camp5 = make_camp()
    n5 = camp5.field_heal_check(reason="strike")
    calls_after_fail = list(FakeTeachFlow.calls)
    n5b = camp5.field_heal_check(reason="strike")
    check("a FAILED bag-drive latches the 10-min backoff (one launch, then stand-down)",
          n5 == 0 and len(calls_after_fail) == 1
          and n5b == 0 and FakeTeachFlow.calls == calls_after_fail
          and camp5._field_heal_backoff > time.time())
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3}, in_battle=True)
    check("mid-battle -> stands down (battle_agent owns in-fight items)",
          make_camp().field_heal_check() == 0 and not FakeTeachFlow.calls)
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3}, box=True)
    check("open dialogue box (scripted scene) -> stands down",
          make_camp().field_heal_check() == 0 and not FakeTeachFlow.calls)
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3})
    _sv = C.FIELD_HEAL_ENABLED
    C.FIELD_HEAL_ENABLED = False
    off = make_camp().field_heal_check()
    C.FIELD_HEAL_ENABLED = _sv
    check("POKEMON_FIELD_HEAL=0 kill switch", off == 0 and not FakeTeachFlow.calls)

    print("== 5. status ride-along (the strike loop has no roam tick) ==")
    set_world(party=[dict(BLASTOISE, status=0x08), LAPRAS], bag={14: 2})
    camp6 = make_camp()
    n6 = camp6.field_heal_check(reason="strike")
    check("poisoned-but-full ace -> Antidote via the existing field_cure flow",
          n6 == 0 and FakeTeachFlow.calls == [("cure", 14, 0)]
          and WORLD["party"][0]["status"] == 0)

    print("== 6. the PRE-LEGENDARY TOP-UP seam (press_quarry) ==")
    ram.pokedex_owns = lambda b, sp: True                 # quarry reads spent -> instant return
    seam_calls = []
    hunt = LS.ZapdosHunt.__new__(LS.ZapdosHunt)
    hunt.b = make_b()
    hunt.log = lambda s: LOGS.append(str(s))
    hunt.deadline = time.time() + 60
    hunt.camp = types.SimpleNamespace(
        on_event=lambda *a, **k: None,
        field_heal_check=lambda reason="", top_up=False:
            seam_calls.append((reason, top_up)) or 1)
    r6 = hunt.press_quarry()
    check("press_quarry fires the strike TOP-UP before the static A-press",
          r6 is True and seam_calls == [("strike", True)])
    seam_calls.clear()
    hunt.camp.field_heal_check = lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
    hunt.field_heal_seam()
    check("a heal wedge can never void the hunt (seam swallows + logs LOUD)",
          any("strike seam skipped" in ln for ln in LOGS))

    print("== 7. restock potion ride-along (compose with _lap_restock_balls) ==")
    C.tv.map_id = lambda b: (3, 8)                        # standing on Cinnabar
    set_world(bag={22: 2})                                # 2 Supers — thin pocket
    camp7 = make_camp()
    camp7._lap_skipped, camp7._lap_fails = set(), {}
    camp7._balls_pocket_count = lambda i: 0
    camp7.money = lambda: 35000
    buys = []
    camp7.buy_at_mart = lambda door, want: buys.append((door, want)) or {want[0][0]: 8}
    r7 = camp7._lap_restock_balls({"map": (3, 8)}, "moltres")
    check("thin heal pocket -> Hyper Potion row rides the SAME ball-restock buy "
          "(balls still row 0)",
          r7 == "ok" and buys and buys[0][1][0][0] == 2
          and (21, 6) in buys[0][1])
    set_world(bag={21: 5, 22: 4})                         # 9 bottles — stocked
    camp8 = make_camp()
    camp8._lap_skipped, camp8._lap_fails = set(), {}
    camp8._balls_pocket_count = lambda i: 0
    camp8.money = lambda: 35000
    buys8 = []
    camp8.buy_at_mart = lambda door, want: buys8.append(want) or {want[0][0]: 8}
    camp8._lap_restock_balls({"map": (3, 8)}, "moltres")
    check("stocked pocket (>=8) -> no potion row added",
          buys8 and all(iid not in (13, 22, 21, 20, 19) for iid, _q in buys8[0]))
    set_world(bag={})
    camp9 = make_camp()
    camp9._lap_skipped, camp9._lap_fails = set(), {}
    camp9.bag_count = lambda iid: (_ for _ in ()).throw(AttributeError("stub"))
    camp9._balls_pocket_count = lambda i: 0
    camp9.money = lambda: 35000
    buys9 = []
    camp9.buy_at_mart = lambda door, want: buys9.append(want) or {want[0][0]: 8}
    r9 = camp9._lap_restock_balls({"map": (3, 8)}, "moltres")
    check("a bag-read flake degrades to balls-only (ride-along is best-effort)",
          r9 == "ok" and buys9 and buys9[0][0][0] == 2
          and any("ride-along skipped" in ln for ln in LOGS))

    print("== 8. WORLD-BACK postcondition (_confirm_world_back, real method) ==")
    set_world(party=[dict(BLASTOISE, hp=100)], bag={21: 3})
    br = RailsBridge(world_after_b=3)
    tf = make_real_tf(br)
    check("cb2 back after 3 Bs -> True (3 unwind + 2 safety Bs, never more)",
          tf._confirm_world_back("fieldheal") is True and br.bs == 5)
    br = RailsBridge(world_after_b=None)
    tf = make_real_tf(br)
    r8 = tf._confirm_world_back("fieldheal")
    check("menu NEVER closes -> False after 10 bounded + 6 LOUD hammer Bs",
          r8 is False and br.bs == 16
          and any("STILL open" in ln for ln in LOGS))
    br = RailsBridge(cb2_raises=True)
    tf = make_real_tf(br)
    check("unreadable cb2 fails OPEN (a read flake never voids a verified heal)",
          tf._confirm_world_back("fieldheal") is True and br.bs == 0)

    print("== 9. REAL field_heal on scripted rails: the Mt. Ember leak, replayed ==")
    _rows, _qty = ht.items_pocket_rows, ht.items_pocket_qty
    ht.items_pocket_rows = lambda b: [(iid, q) for iid, q in WORLD["bag"].items() if q > 0]
    ht.items_pocket_qty = lambda b, iid: WORLD["bag"].get(iid, 0)
    set_world(party=[dict(BLASTOISE, hp=4)], bag={21: 3})
    br9 = RailsBridge(world_after_b=None)                 # the live wedge: Bs all eaten
    r9a = make_real_tf(br9).field_heal(21, 0)
    check("heal LANDS (4 -> 244, bottle gone) but the menu never closes -> 'menu_stuck', LOUD",
          r9a == "menu_stuck" and WORLD["party"][0]["hp"] == 244 and WORLD["bag"][21] == 2
          and any("MENU STACK never closed" in ln for ln in LOGS))
    set_world(party=[dict(BLASTOISE, hp=4)], bag={21: 3})
    br9b = RailsBridge(world_after_b=13)                  # _b_cascade's 12 + 1 confirm B
    check("same rails, callback restored after the unwind -> 'healed'",
          make_real_tf(br9b).field_heal(21, 0) == "healed"
          and any("world callback restored" in ln for ln in LOGS))
    ht.items_pocket_rows, ht.items_pocket_qty = _rows, _qty
    set_world(party=[dict(BLASTOISE, hp=100), LAPRAS], bag={21: 3},
              heal_result="menu_stuck")
    camp9b = make_camp()
    n9 = camp9b.field_heal_check(reason="strike")
    check("campaign books 'menu_stuck' as a FAILED drive: no heal counted, backoff latched",
          n9 == 0 and camp9b._field_heal_backoff > time.time())

    print("== 10. watchdog side: sweep close-confirm + the disengage MENU rung ==")
    campA = make_menu_camp(bs_to_close=2)
    check("stray bag closes after 2 Bs -> 'closed' (pixels clear AND cb2 back)",
          campA._sweep_stray_menus(reason="watchdog frozen_world") == "closed")
    campB = make_menu_camp(bs_to_close=None)
    check("menu that will NOT close -> 'stuck' after the bounded cascade",
          campB._sweep_stray_menus(reason="watchdog frozen_world") == "stuck")
    campN = make_menu_camp(bs_to_close=0)
    campN.b.press = lambda *a, **k: check("no stray menu -> no presses", False)
    check("no stray menu -> None (nothing pressed)",
          campN._sweep_stray_menus() is None)

    dd.box_open = lambda b: WORLD.get("box", False)       # _disengage_step1 imports fresh
    req = {"reason": "frozen_world", "map": (1, 97), "coords": (15, 33), "facing": 2}
    now = time.time()
    campD = make_menu_camp(bs_to_close=2)
    campD._blocked_npcs = {((1, 97), (15, 34)), ((1, 97), (2, 2)), ((3, 8), (5, 5))}
    campD._looped_spots = {((1, 97), (15, 33))}
    campD._wedge_mem_ts = {("blocked_npcs", (1, 97), (15, 34)): now - 60,     # frozen window
                           ("blocked_npcs", (1, 97), (2, 2)): now - 3600,     # a real old trap
                           ("blocked_npcs", (3, 8), (5, 5)): now - 60,        # other map
                           ("looped_spots", (1, 97), (15, 33)): now - 60}
    campD._save_wedge_memory = lambda: None
    marked = []
    campD._mark_wedge_spot = lambda r: marked.append(r)
    set_world(party=[BLASTOISE], bag={})
    rung = campD._disengage_step1(req)
    check("stray menu at disengage -> 'menu' rung: closed, NO wedge-mark",
          rung == "menu" and not marked)
    check("...phantom marks of the frozen window RELEASED (route tile (15,34) freed, "
          "stand tile unlatched), old + other-map marks kept",
          campD._blocked_npcs == {((1, 97), (2, 2)), ((3, 8), (5, 5))}
          and campD._looped_spots == set())
    campE = make_menu_camp(bs_to_close=None)
    campE._mark_wedge_spot = lambda r: marked.append(r)
    check("menu that won't close -> 'menu_stuck' rung, STILL no phantom wedge-mark",
          campE._disengage_step1(req) == "menu_stuck" and not marked)
    campF = make_menu_camp(bs_to_close=0)                 # a genuinely frozen WORLD
    campF._mark_wedge_spot = lambda r: marked.append(r)
    check("no box, no menu -> the classic 'mark' rung is untouched",
          campF._disengage_step1(req) == "mark" and marked == [req])

    print("== 11. strike-leg guard (GiovanniGym.handle_interrupts) ==")
    GG.dd_box = lambda b: False
    sweeps = []
    g = GG.GiovanniGym.__new__(GG.GiovanniGym)
    gstate = {"cb2": _CB2_MENU}
    g.b = types.SimpleNamespace(
        rd32=lambda a: gstate["cb2"] if a == ram.GMAIN_CB2 else 0)
    g.camp = types.SimpleNamespace(
        _sweep_stray_menus=lambda **kw: sweeps.append(kw) or "closed")
    check("leaked menu mid-leg (cb2 non-overworld) -> sweep fires, interrupt HANDLED",
          g.handle_interrupts() is True and sweeps == [{"reason": "strike leg"}])
    sweeps.clear()
    gstate["cb2"] = ram._CB2_OVERWORLD
    check("healthy overworld cb2 -> one free read, sweep never called",
          g.handle_interrupts() is False and not sweeps)
    gstate["cb2"] = _CB2_MENU
    g.camp._sweep_stray_menus = lambda **kw: "stuck"
    check("sweep 'stuck' -> NOT handled (deadline machinery owns it, no busy loop)",
          g.handle_interrupts() is False)

    print("== 12. THE START-MENU BLIND SPOT (cb2 overworld, EXIT-cursor wedge) ==")
    check("gTasks scan: Task_StartMenuHandleInput alive -> start_menu_open True",
          ram.start_menu_open(StartBridge(closes_after=None)) is True)
    check("menu not open / task inactive -> False (stale func ptr never lies)",
          ram.start_menu_open(StartBridge(closes_after=0)) is False
          and ram.start_menu_open(StartBridge(active=False)) is False)
    _boom = types.SimpleNamespace(
        rd32=lambda a: (_ for _ in ()).throw(RuntimeError("flake")), rd8=lambda a: 0)
    check("unreadable gTasks fails CLOSED (old behavior, outer layers unaffected)",
          ram.start_menu_open(_boom) is False)
    set_world(party=[BLASTOISE], bag={})
    campS = make_start_camp(StartBridge(closes_after=None))
    check("_stray_menu_kind sees 'start' where cb2 truth AND bag/party pixels are blind",
          campS._stray_menu_kind() == "start")
    campS2 = make_start_camp(StartBridge(closes_after=1))
    check("sweep closes the START menu: 1 B kills the task, cb2 was overworld all along "
          "-> 'closed'", campS2._sweep_stray_menus(reason="tick top") == "closed")
    campS3 = make_start_camp(StartBridge(closes_after=1))
    campS3._blocked_npcs, campS3._looped_spots, campS3._wedge_mem_ts = set(), set(), {}
    campS3._save_wedge_memory = lambda: None
    marked12 = []
    campS3._mark_wedge_spot = lambda r: marked12.append(r)
    check("watchdog disengage on the START wedge -> 'menu' rung, closed, NO wedge-mark",
          campS3._disengage_step1({"reason": "frozen_world", "map": (1, 96),
                                   "coords": (24, 6)}) == "menu" and not marked12)
    sweeps12 = []
    g2 = GG.GiovanniGym.__new__(GG.GiovanniGym)
    g2.b = StartBridge(closes_after=None)
    g2.camp = types.SimpleNamespace(
        _sweep_stray_menus=lambda **kw: sweeps12.append(kw) or "closed")
    check("strike-leg guard fires on START despite a healthy overworld cb2",
          g2.handle_interrupts() is True and sweeps12 == [{"reason": "strike leg"}])
    br12 = StartBridge(closes_after=2)
    tf12 = make_real_tf(br12)
    check("_confirm_world_back: cb2-overworld-but-START-open is NOT 'world back' — "
          "Bs until the task dies (2 + 2 safety Bs)",
          tf12._confirm_world_back("fieldheal") is True and br12.bs == 4)

    print("== 13. universal B-FIRST rung (menus we haven't met yet) ==")
    campP = make_start_camp(DriftScreen(menu=True, drift_cols=1))
    check("an UNCLASSIFIED menu: first B flips the screen far beyond drift -> True",
          campP._b_first_probe() is True
          and any("B-FIRST rung" in ln for ln in LOGS))
    campQ = make_start_camp(DriftScreen(menu=False, drift_cols=1))
    check("a menu-less frozen world: tile animation alone never reads as a menu "
          "(2 Bs, no false positive)", campQ._b_first_probe() is False)
    campR = make_menu_camp(bs_to_close=0)                 # no classified menu
    campR._b_first_probe = lambda: True
    released13, marked13 = [], []
    campR._release_recent_wedge_marks = lambda r: released13.append(r)
    campR._mark_wedge_spot = lambda r: marked13.append(r)
    req13 = {"reason": "frozen_world", "map": (1, 96), "coords": (24, 6)}
    check("disengage: no classifier match but B changed the screen -> 'bfirst' rung, "
          "marks released, NO wedge-mark",
          campR._disengage_step1(req13) == "bfirst"
          and released13 == [req13] and not marked13)
    campU = make_menu_camp(bs_to_close=0)
    campU._b_first_probe = lambda: False
    campU._mark_wedge_spot = lambda r: marked13.append(r)
    check("...and a B that changes nothing still falls through to the classic wedge-mark",
          campU._disengage_step1(req13) == "mark" and marked13 == [req13])

    C.log = _oldlog
    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
