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

    if check.fails:
        print(f"\n{check.fails} FAILED")
        sys.exit(1)
    print("\nALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
