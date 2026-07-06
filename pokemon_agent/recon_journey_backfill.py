"""recon_journey_backfill.py — restore the Gary win to her story (2026-07-06 ride-along 0b).

THE CORRUPTION: run-5 (gary_run5.log:89) recorded `RIVAL beat #3 vs Gary (won=True, place='Cerulean
City') — grudge now 1W-2L`, but runs 6-8 restarted from a bank that predated it, so the promoted
canonical strat_memory.json carries only the two losses → journey_core.json tells her "he's still got
your number" — a story LIE about her proudest moment.

THE FIX (surgical, backed up):
  1. backup strat_memory.json + journey_core.json (.bak_<ts>)
  2. append the run-5 win to strat.rival.encounters (fields mirrored from the run-5 call site:
     his lead pidgeotto, her party 3, her level 29)
  3. regenerate journey_core.json through the REAL campaign._journey_narrative() (also clears the
     mojibake — the old file had double-encoded text from a copy-tool round-trip)
RUN: python pokemon_agent/recon_journey_backfill.py
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

STRAT = os.path.join(_HERE, "states", "campaign", "strat_memory.json")
JOURNEY = os.path.join(_HERE, "states", "campaign", "journey_core.json")

RUN5_WIN = {"won": True, "place": "Cerulean City", "lead": "pidgeotto",
            "my_party": 3, "my_level": 29}


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    for p in (STRAT, JOURNEY):
        shutil.copy2(p, p + f".bak_{ts}")
        print(f"backup: {p}.bak_{ts}")

    with open(STRAT, encoding="utf-8") as f:
        s = json.load(f)
    enc = s.setdefault("rival", {"name": "Gary", "encounters": []}).setdefault("encounters", [])
    wins = [e for e in enc if e.get("won")]
    if wins:
        print(f"ABORT: a win is already recorded ({len(wins)}) — nothing to backfill")
        return
    enc.append(RUN5_WIN)
    tmp = STRAT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f)
    os.replace(tmp, STRAT)
    w = sum(1 for e in enc if e.get("won"))
    print(f"strat backfilled: rival encounters now {len(enc)} (grudge {w}W-{len(enc) - w}L)")

    # regenerate journey_core through the REAL machinery (booted read-only on the canonical save)
    from bridge import Bridge                      # noqa: E402
    from campaign import Campaign, resolve_state   # noqa: E402
    b = Bridge(os.path.join(os.path.dirname(_HERE), "roms", "firered.gba"))
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp.strat.load(STRAT)
    if camp.soul is not None:
        try:
            camp.soul.load(os.path.join(_HERE, "states", "campaign", "soul.json"))
        except Exception:
            pass
    narrative = camp._journey_narrative()
    tmp = JOURNEY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(narrative, f, ensure_ascii=False, indent=2)
    os.replace(tmp, JOURNEY)
    print("\nregenerated journey_core.json:")
    print("  grudge:", narrative.get("grudge"))
    print("  summary:", narrative.get("summary")[:220])
    bad = [tok for tok in ("Ã", "â€") if tok in json.dumps(narrative)]
    print("  mojibake check:", "CLEAN" if not bad else f"STILL DIRTY {bad}")


if __name__ == "__main__":
    main()
