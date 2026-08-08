"""recon_fly_nav_test.py — Fly destination table + victory-lap Fly wiring (no ROM).

RUN: python3 pokemon_agent/recon_fly_nav_test.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Mac / cloud agents have no mgba — stub so campaign imports clean.
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import campaign as C
import fly_nav


def check(name, cond):
    ok = bool(cond)
    print(("PASS" if ok else "FAIL") + f": {name}")
    if not ok:
        check.fails += 1


check.fails = 0


def main():
    print("== fly_nav.resolve_dest / can_fly_here ==")
    z = fly_nav.resolve_dest("zapdos")
    check("zapdos alias -> Route 10", z and z["map"] == (3, 28))
    check("powerplant alias -> Route 10",
          fly_nav.resolve_dest("power_plant")["map"] == (3, 28))
    check("cerulean -> (3,3)", fly_nav.resolve_dest("cerulean")["map"] == (3, 3))
    check("route10 -> (3,28)", fly_nav.resolve_dest("route10")["map"] == (3, 28))
    check("route4 -> (3,22) live id (NOT Route 1)",
          fly_nav.resolve_dest("route4")["map"] == (3, 22))
    check("indigo -> (3,9) (NOT Kindle (3,45))",
          fly_nav.resolve_dest("indigo")["map"] == (3, 9))
    check("map tuple (3,28) resolves",
          fly_nav.resolve_dest((3, 28))["map"] == (3, 28))
    check("unknown -> None", fly_nav.resolve_dest("mt_moon") is None)

    class B:
        pass

    b = B()
    # can_fly_here needs tv.map_id — stub via monkeypatch
    import travel as tv
    _orig = tv.map_id
    try:
        tv.map_id = lambda _b: (1, 85)  # Seafoam B2F
        check("Seafoam refuses Fly", fly_nav.can_fly_here(b) is False)
        tv.map_id = lambda _b: (3, 38)  # Route 20
        check("Route 20 allows Fly", fly_nav.can_fly_here(b) is True)
        tv.map_id = lambda _b: (1, 95)  # Power Plant indoors
        check("Power Plant indoors refuses Fly", fly_nav.can_fly_here(b) is False)
    finally:
        tv.map_id = _orig

    print("== victory lap order: fly before zapdos ==")
    order = C.VICTORY_LAP_ORDER
    check("fly in VICTORY_LAP_ORDER", "fly" in order)
    check("fly before zapdos",
          order.index("fly") < order.index("zapdos"))
    check("articuno before fly",
          order.index("articuno") < order.index("fly"))

    print("== lap fly anchors = Route 16 (3,34), not Route 4 ==")
    camp = C.Campaign.__new__(C.Campaign)
    anch = camp._lap_anchor_sets().get("fly") or set()
    check("fly anchors include Celadon", (3, 6) in anch)
    check("fly anchors include Route 16 (3,34)", (3, 34) in anch)
    check("fly anchors do NOT use Route 4 as Route 16", (3, 22) not in anch)

    print("== Seafoam B2F (29,12) east sealed (LIVE 18:29 strand) ==")
    import legendary_strikes as LS
    import travel as tv
    pos = {"m": (1, 85), "c": (29, 12)}
    _om, _oc = tv.map_id, tv.coords
    tv.map_id = lambda _b: pos["m"]
    tv.coords = lambda _b: pos["c"]
    try:
        h = LS.ArticunoHunt.__new__(LS.ArticunoHunt)
        h.b = object()
        check("(29,12) is east sealed pocket", h._b2f_east_sealed_pocket() is True)
        pos["c"] = (32, 14)
        check("(32,14) UP ladder is NOT sealed", h._b2f_east_sealed_pocket() is False)
        check("east re-drop table leads with (27,8)",
              LS.ArticunoHunt._B2F_EAST_REDROP[0] == (27, 8))
    finally:
        tv.map_id, tv.coords = _om, _oc

    print("== fly_slot returns INT slot (never bool True→Lapras) ==")
    import field_moves as fm
    import pokemon_state as st
    class _B: pass
    b = _B()
    b.rd8 = lambda _a: 5
    _ou = fm.usable_hms
    _orm, _orp, _ors = st.read_party_moves, st.read_party_public, st.read_party_species
    try:
        fm.usable_hms = lambda _b, _c=6: {
            "fly": {"slot": 2, "badge_ok": True, "name": "Fly"}}
        st.read_party_moves = lambda _b, s: ([19] if s == 2 else [])
        st.read_party_public = lambda _b, s: {
            "hp": 0 if s == 1 else 80, "species": 22 if s == 2 else 131, "maxhp": 100}
        st.read_party_species = lambda _b, s: 22 if s == 2 else 131
        slot = fly_nav.fly_slot(b)
        check("fly_slot is int 2 (Fearow), not bool True", slot == 2 and type(slot) is int)
        check("fly_slot is never True (Lapras coerce bug)", slot is not True)
    finally:
        fm.usable_hms = _ou
        st.read_party_moves = _orm
        st.read_party_public = _orp
        st.read_party_species = _ors

    print("== Route 20 east here_xy (Seafoam mouth) ==")
    pos20 = {"c": (60, 9)}
    _om2, _oc2 = tv.map_id, tv.coords
    tv.map_id = lambda _b: (3, 38)
    tv.coords = lambda _b: pos20["c"]
    try:
        check("R20@(60,9) cursor ~ (11,14) east", fly_nav._here_xy(b) == (11, 14))
        pos20["c"] = (10, 9)
        check("R20@(10,9) cursor ~ (5,14) west", fly_nav._here_xy(b) == (5, 14))
    finally:
        tv.map_id, tv.coords = _om2, _oc2

    print("== R20 mouth lockout: rope landing never walks back in ==")
    camp3 = C.Campaign.__new__(C.Campaign)
    camp3.b = b
    camp3._blocked_npcs = set()
    camp3._save_wedge_memory = lambda: None
    camp3.render = None
    pressed = []
    b.press = lambda k, h, r, cb=None, owner=None: pressed.append(k)
    b.run_frame = lambda: None
    _om3, _oc3, _orw, _og = tv.map_id, tv.coords, tv.read_warps, tv.Grid
    tv.map_id = lambda _b: (3, 38)
    tv.coords = lambda _b: (60, 9)
    tv.read_warps = lambda _b: [((60, 8), (1, 83), 0), ((72, 14), (1, 83), 1)]
    tv.Grid = lambda _b: type("G", (), {"walkable": staticmethod(
        lambda x, y: (x, y) != (60, 8))})()
    try:
        camp3._seafoam_mouth_lockout("test")
        check("both R20 mouths blocked in wedge memory",
              ((3, 38), (60, 8)) in camp3._blocked_npcs
              and ((3, 38), (72, 14)) in camp3._blocked_npcs)
        check("stepped away from the door (presses fired)", len(pressed) >= 1)
        check("never pressed UP into the mouth", "UP" not in pressed)
    finally:
        tv.map_id, tv.coords, tv.read_warps, tv.Grid = _om3, _oc3, _orw, _og

    print("== fly pending: Cut re-teachable from case -> NOT skipped ==")
    import hm_teach as _ht
    camp2 = C.Campaign.__new__(C.Campaign)
    camp2.b = b
    camp2._lap_skipped = {"fly"}          # previously latched 'no Cut'
    camp2.world = type("W", (), {"has_cap": staticmethod(lambda n: False)})()
    _otc, _opk, _ocu = _ht.tm_case_row, st.party_knows_move, C.fm.can_use
    try:
        # HM02 NOT in case, flag clear, nobody knows Cut — but HM01 IS in the case.
        _ht.tm_case_row = lambda _b, item: (0 if item == 339 else None)
        st.party_knows_move = lambda _b, m, c=6: None
        C.fm.can_use = lambda _b, k, c=6: False
        _orf = C.fm.read_flag
        C.fm.read_flag = lambda _b, f: False
        camp2._lap_skip = lambda k, why: camp2._lap_skipped.add(k)
        pend = camp2._lap_pending("fly")
        check("HM01 in case -> fly PENDING (Cut teach-in-place unblocks fetch)",
              pend is True)
        check("stale 'no Cut' skip refunded", "fly" not in camp2._lap_skipped)
        # No HM01 anywhere -> genuine dead end, honest skip.
        camp2._lap_skipped = set()
        _ht.tm_case_row = lambda _b, item: None
        pend2 = camp2._lap_pending("fly")
        check("no Cut + no HM01 -> honest skip", pend2 is False
              and "fly" in camp2._lap_skipped)
        # Fail budget SPENT -> refund stops, skip STANDS (LIVE 19:07 50x loop).
        camp2._lap_skipped = {"fly"}
        camp2._lap_fails = {"fly": C.VICTORY_LAP_MAX_FAILS}
        _ht.tm_case_row = lambda _b, item: (0 if item == 339 else None)
        pend3 = camp2._lap_pending("fly")
        check("fails >= MAX -> skip stands (no refund loop)", pend3 is False
              and "fly" in camp2._lap_skipped)
    finally:
        _ht.tm_case_row, st.party_knows_move, C.fm.can_use = _otc, _opk, _ocu
        C.fm.read_flag = _orf

    if check.fails:
        print(f"\n{check.fails} FAILED")
        sys.exit(1)
    print("\nALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
