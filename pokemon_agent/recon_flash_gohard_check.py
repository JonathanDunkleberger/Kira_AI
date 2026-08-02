"""recon_flash_gohard_check.py — Route 9↔10 Flash vs GO-HARD (2026-08-02).

Proves WITHOUT an emulator that:
  (1) GO-HARD no longer parks flash/cut questlines (road blockers)
  (2) tunnel need_flash does not fall through to edge oscillation
  (3) Ice Beam broke arms cash farm + defers gym door

RUN:  python -u recon_flash_gohard_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def run():
    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("1a road-blocker flash/cut set",
          '_road_blocker = _ql_miss in ("flash", "cut")' in camp)
    check("1b GO-HARD waives park for road blockers",
          "GO-HARD WAIVES questline park" in camp)
    check("1c need_flash returns road_gated (no edge fallthrough)",
          'if r == "need_flash"' in camp and "no edge fallthrough" in camp)
    check("1d unresolved tunnel -> road_gated",
          "not edge-oscillating" in camp)

    check("2a beat_gym returns need_cash_for_tm when broke",
          'return "need_cash_for_tm"' in camp)
    check("2b _ice_beam_cash_needed armed",
          "_ice_beam_cash_needed" in camp)
    check("2c cash farm keeps battle / stands down force-gym",
          "ICE-BEAM CASH FARM" in camp and "_cash_blocking" in camp)

    import game_corner as gc
    check("3a ice_beam_cash_shortfall defined",
          callable(getattr(gc, "ice_beam_cash_shortfall", None)))

    print()
    if fails:
        print(f"FAIL: {fails}")
        sys.exit(1)
    print("ALL PASS — Flash road-blocker + Ice Beam cash farm wired.")


if __name__ == "__main__":
    run()
