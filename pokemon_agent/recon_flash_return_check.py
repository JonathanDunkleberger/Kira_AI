"""recon_flash_return_check.py — static wiring for post-Flash Celadon road-head (no ROM).

PASS = campaign refuses early flash_done west of Diglett, latches `_flash_returned` only on
Celadon spine, free-roam rescue + Pewter/Route3/Route4 avoid exist.

RUN:  python -u recon_flash_return_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def run():
    src = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    print("recon_flash_return_check")

    check("1a diglett_east helper", "def _diglett_east_to_vermilion" in src)
    check("1b free-roam rescue hook", "def _rescue_post_flash_detour" in src)
    check("1c celadon spine set", "_CELADON_SPINE_MAPS" in src)
    check("1d west pocket set", "_FLASH_WEST_POCKET" in src)

    check("2a early flash_done gated on spine",
          "Flash already taught but still WEST" in src
          and "forcing Vermilion road-head" in src)
    check("2b teach-path holds questline until spine",
          "questline stays open until she's on the Celadon spine" in src)
    check("2c diglett east called after Flash learnt",
          "Flash learnt — Diglett EAST to Vermilion road-head" in src)

    check("3a Pewter/Route3/Route4 in post-flash avoid",
          "return {PEWTER, ROUTE3, ROUTE4}" in src)
    check("3b Route4 east force", "POST-CASCADE Route 4 — forcing EAST" in src)
    check("3c free-roam calls rescue before mt.moon escape",
          src.find("rescue_flash_detour") < src.find('ledger.note_action("escape_mt_moon"'))

    check("4a west pocket literals",
          "ROUTE2, PEWTER, ROUTE3" in src and "_DIGLETT_CAVE_MAPS" in src)
    check("4b spine includes Rock Tunnel + Vermilion",
          "(1, 81), (1, 82)" in src and "VERMILION," in src
          and "CERULEAN," in src)

    if fails:
        print(f"FAIL — {len(fails)}: {fails}")
        sys.exit(1)
    print("ALL PASS — post-Flash return wiring looks sound (live Diglett cross needs PC ROM).")


if __name__ == "__main__":
    run()
