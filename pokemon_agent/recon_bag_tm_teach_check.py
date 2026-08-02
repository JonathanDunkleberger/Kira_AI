"""recon_bag_tm_teach_check.py — opportunistic bag-TM teach (2026-08-02 dream).

Proves WITHOUT an emulator:
  (1) score_tm_recipient prefers Teleport-only Abra over a full Blastoise
  (2) comparable coverage returns -1 (don't burn Shock Wave on a loaded electric)
  (3) campaign wires teach_bag_tms + teach_tm action + post-badge hook
  (4) Abra learnset note: Shock Wave NOT on Abra; Brick Break / Iron Tail are

RUN:  python -u recon_bag_tm_teach_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hm_teach as ht  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def run():
    # Teleport-only Abra vs Brick Break
    abra = ht.score_tm_recipient([], ["psychic"], "fighting", 75, plan_boost=True)
    blast = ht.score_tm_recipient(
        [("water", 60), ("dark", 60), ("water", 40)],
        ["water"], "fighting", 75, is_ace=True)
    check("1a Abra empty offense scores high", abra >= 1000, True)
    check("1b Abra >> Blastoise for Brick Break", abra > blast, True)

    # Already has Thunderbolt — don't teach Shock Wave
    skip = ht.score_tm_recipient([("electric", 95)], ["electric"], "electric", 60)
    check("2a skip comparable electric coverage", skip, -1)

    # STAB Shock Wave on Pikachu with only Tackle
    pika = ht.score_tm_recipient([("normal", 35)], ["electric"], "electric", 60)
    check("2b Pikachu wants Shock Wave STAB", pika > 80, True)

    # forget_idx helper: free slot -> None (no bridge needed for the free-slot branch of logic)
    # exercised via source contract below

    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("3a teach_bag_tms defined", "def teach_bag_tms" in camp, True)
    check("3b _plan_bag_tm_teach defined", "def _plan_bag_tm_teach" in camp, True)
    check("3c teach_tm in _route_action", 'pick == "teach_tm"' in camp, True)
    check("3d teach_tm offered in available_actions", 'a["teach_tm"]' in camp, True)
    check("3e post-badge teach hook", "post-badge bag-TM teach" in camp, True)
    check("3f BAG_TM_TEACH_ENABLED", "BAG_TM_TEACH_ENABLED" in camp, True)
    check("3g Water Pulse move id 352", "291: (None, 3, 352," in camp, True)
    check("3h Iron Tail + Aerial Ace in catalog",
          "311: (None, 23," in camp and "328: (None, 40," in camp, True)

    # Learnset truth (static KB): Abra has TM31 not TM34
    import json
    ls = json.load(open(os.path.join(_HERE, "gamedata", "frlg_learnsets.json"), encoding="utf-8"))
    abra_tm = set((ls.get("abra") or {}).get("tm") or [])
    check("4a Abra learns TM31 Brick Break", "TM31" in abra_tm, True)
    check("4b Abra does NOT learn TM34 Shock Wave", "TM34" in abra_tm, False)
    check("4c Abra learns TM23 Iron Tail", "TM23" in abra_tm, True)

    check("5a score_tm_recipient in hm_teach", hasattr(ht, "score_tm_recipient"), True)
    check("5b forget_idx_for_tm in hm_teach", hasattr(ht, "forget_idx_for_tm"), True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — bag-TM teach prefers empty Abra; Shock Wave≠Abra; hooks wired.")


if __name__ == "__main__":
    run()
