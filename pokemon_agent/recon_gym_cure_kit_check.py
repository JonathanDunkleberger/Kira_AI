"""recon_gym_cure_kit_check.py — headless verifier for KB bring_cures / pre-gym cure kit.

Proves WITHOUT an emulator:
  (1) frlg_strategy.json lists bring_cures for Lt. Surge (paralysis) and Koga (poison)
  (2) campaign source wires _stock_status_cures_for_gym into beat_gym
  (3) shopping_list foresight pulls KB cures (source contract)
  (4) Discord diary is PARKED by default

RUN:  python -u recon_gym_cure_kit_check.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def run():
    kb_path = os.path.join(_HERE, "gamedata", "frlg_strategy.json")
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    threats = kb.get("threats") or {}

    surge = threats.get("Lt. Surge") or {}
    koga = threats.get("Koga") or {}
    check("1a Surge bring_cures has paralysis",
          "paralysis" in (surge.get("bring_cures") or []), True)
    check("1b Koga bring_cures has poison",
          "poison" in (koga.get("bring_cures") or []), True)

    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("2a _stock_status_cures_for_gym defined",
          "def _stock_status_cures_for_gym" in camp, True)
    check("2b beat_gym calls cure stock",
          "self._stock_status_cures_for_gym(gym)" in camp, True)
    check("2c shopping foresight uses _kb_bring_cures",
          "_kb_bring_cures" in camp and "GYM_CURE_TARGET" in camp, True)
    check("2d Vermilion sells Parlyz Heal (18)",
          "VERMILION: [4, 22, 16, 17, 18, 86]" in camp, True)

    cfg = open(os.path.join(_ROOT, "kira", "config.py"), encoding="utf-8").read()
    check("3a DISCORD_DIARY_PARKED in config", "DISCORD_DIARY_PARKED" in cfg, True)
    bot = open(os.path.join(_ROOT, "kira", "bot.py"), encoding="utf-8").read()
    check("3b bot skips diary when parked",
          "DISCORD_DIARY_PARKED" in bot and "PARKED — skipping diary" in bot, True)

    # Live import: planner sees bring_cures
    import pokemon_planner as P
    planner = P.StrategicPlanner(log=lambda *_a, **_k: None)
    check("4a planner Surge bring_cures",
          (planner.threats.get("Lt. Surge") or {}).get("bring_cures"), ["paralysis"])

    from kira.config import DISCORD_DIARY_PARKED, DISCORD_AUTOPOST
    check("5a diary parked by default", DISCORD_DIARY_PARKED, True)
    check("5b autopost forced off while parked", DISCORD_AUTOPOST, False)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — Surge/Koga cure kit in KB; beat_gym stocks; Discord diary parked.")


if __name__ == "__main__":
    run()
