"""recon_gym_cure_kit_check.py — headless verifier for KB bring_cures / pre-gym cure kit.

Proves WITHOUT an emulator:
  (1) frlg_strategy.json lists bring_cures for Surge / Erika / Koga
  (2) campaign source wires _stock_pre_gym_kit into beat_gym (potions+cures, one trip)
  (3) Celadon Dept Store stock + buy path exist
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
    erika = threats.get("Erika") or {}
    koga = threats.get("Koga") or {}
    check("1a Surge bring_cures has paralysis",
          "paralysis" in (surge.get("bring_cures") or []), True)
    check("1b Erika bring_cures has poison+sleep",
          set(erika.get("bring_cures") or []) >= {"poison", "sleep"}, True)
    check("1c Koga bring_cures has poison",
          "poison" in (koga.get("bring_cures") or []), True)

    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("2a _stock_pre_gym_kit defined",
          "def _stock_pre_gym_kit" in camp, True)
    check("2b beat_gym calls pre-gym kit",
          "self._stock_pre_gym_kit(gym)" in camp, True)
    check("2c Celadon Dept buy path",
          "def buy_at_celadon_dept" in camp and "CELADON_DEPT_DOOR" in camp, True)
    check("2d Celadon shelf has Super Potion+Antidote",
          "CELADON: [3, 22, 24, 14, 18, 17, 15, 16, 87]" in camp, True)
    check("2e shopping foresight uses _kb_bring_cures",
          "_kb_bring_cures" in camp and "GYM_CURE_TARGET" in camp, True)

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
    print("ALL PASS — Surge/Erika/Koga cure kit in KB; pre-gym kit + Celadon Dept wired.")


if __name__ == "__main__":
    run()
