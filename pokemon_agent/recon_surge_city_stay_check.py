"""recon_surge_city_stay_check.py — headless verifier for the 2026-08-01 Vermilion Cut→stay fix.

Live chalk: after TIMBER on Surge's gym tree she armed a beach `surf` questline (anchor Fuchsia),
GO-HARD parked it, discovery marched her onto Route 6 / UGP, then back to Vermilion → cut → loop.

Proves WITHOUT an emulator that campaign.py's at-city stuck handler + _gym_gate_probe contain:
  1. cut_cleared signal + immediate door re-enter
  2. no-surf-questline / water-gate discard at Vermilion for Lt. Surge
  3. surf-poison clear when next gym is Surge and she's in the city
  4. SURGE-STAY grass ban for Route 6 / UGP
  5. post-Cut stay (no leave-city questline exile)

Also re-runs recon_surge_prep_check + recon_trashcan_puzzle_check.

RUN:  python3 -u recon_surge_city_stay_check.py
"""
import os
import subprocess
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import mgba.core          # noqa: F401
except ImportError:
    _root = types.ModuleType("mgba")
    for _m in ("core", "image", "vfs", "log", "png"):
        _sub = types.ModuleType(f"mgba.{_m}")
        setattr(_root, _m, _sub)
        sys.modules[f"mgba.{_m}"] = _sub
    _root.log.silence = lambda: None
    sys.modules["mgba"] = _root

import campaign as C          # noqa: E402

FAILS = []


def check(n, label, cond):
    print(f"  [{n}] {label}: {'OK' if cond else '!! FAIL'}")
    if not cond:
        FAILS.append(n)


print("== Surge city-stay (Cut → no Route 6 exile) ==")

src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()

# --- probe contract ---
check(1, "_gym_gate_probe returns cut_cleared (not bare None) after TIMBER",
      'return "cut_cleared"' in src
      and "signaling cut_cleared" in src)

check(2, "probe refuses water/recognize fallthrough when door cut-tree owns the stuck",
      "REFUSING water/recognize fallthrough" in src
      and "DISCARDING water/surf gate at gym door" in src)

check(3, "probe ignores non-cut/strength gates at gym door",
      "IGNORING non-door-tree gate at gym city" in src
      or "IGNORING non-cut/strength gym-gate" in src)

# --- at-city stuck handler ---
check(4, "at-city handler does immediate re-enter on cut_cleared",
      'gate == "cut_cleared"' in src
      and "_retry_gym_door_enter" in src
      and "immediate door" in src
      and "post-Cut beat_gym re-call" in src)

check(5, "no-surf-questline while door is in gym city",
      "no-surf-questline while door is in" in src
      and "DISCARDING water/surf gym-gate" in src)

check(6, "surf poison cleared at Vermilion when next gym is Lt. Surge",
      "surf poison at Vermilion" in src
      and '== "surf"' in src
      and "Lt. Surge" in src)

check(7, "post-Cut stuck STAYS in city — no questline exile to Route 6/UGP",
      "no questline exile to Route 6/UGP" in src
      and "STAYING in" in src
      and "REFUSING questline step that would leave" in src)

check(8, "_retry_gym_door_enter walks stand adjacent to door (bounded 2 tries)",
      "def _retry_gym_door_enter" in src
      and "post-Cut door approach" in src
      and "tries=2" in src)

check(9, "VERMILION_GYM_DOOR is (14, 25)",
      getattr(C, "VERMILION_GYM_DOOR", None) == (14, 25))

# --- grass stay ---
check(10, "SURGE-STAY bans Route 6 / UGP as grass targets at Vermilion",
      "SURGE-STAY" in src
      and "_SURGE_BACK_GRASS" in src
      and "(3, 24)" in src
      and "range(30, 36)" in src)

# --- constants wiring ---
check(11, "Lt. Surge GymSpec door is VERMILION_GYM_DOOR",
      C.GYMS["Lt. Surge"].door == C.VERMILION_GYM_DOOR
      and C.GYMS["Lt. Surge"].city == C.VERMILION)

print("== result (source/contract):", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")

# --- sibling recons must still pass ---
print("\n== sibling recon_surge_prep_check ==")
r1 = subprocess.run([sys.executable, "-u", os.path.join(_HERE, "recon_surge_prep_check.py")],
                    cwd=_HERE)
print("== sibling recon_trashcan_puzzle_check ==")
r2 = subprocess.run([sys.executable, "-u", os.path.join(_HERE, "recon_trashcan_puzzle_check.py")],
                    cwd=_HERE)

if r1.returncode != 0:
    FAILS.append("recon_surge_prep_check")
if r2.returncode != 0:
    FAILS.append("recon_trashcan_puzzle_check")

print("\n== FINAL:", "ALL OK" if not FAILS else f"FAILED {FAILS}", "==")
sys.exit(1 if FAILS else 0)
