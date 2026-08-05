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
     a bag-read flake degrades to balls-only (best-effort, never blocks the restock).
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
import firered_ram as ram
import hm_teach as ht
import legendary_strikes as LS
import pokemon_state as st

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

    C.log = _oldlog
    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
