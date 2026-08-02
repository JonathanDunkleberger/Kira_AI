"""recon_fight_reset_check.py — refuse mid-victory fight re-entry (no ROM)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def run():
    ba = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
    tr = open(os.path.join(_HERE, "travel.py"), encoding="utf-8").read()
    pl = open(os.path.join(_HERE, "play_live.py"), encoding="utf-8").read()
    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    print("recon_fight_reset_check")
    check("1a decided_win helper", "def _decided_win(self)" in ba)
    check("1b decided-win drain", "def _drain_decided_win" in ba)
    check("1c drain120 refuses re-entry", "drain120_decided_win" in ba)
    check("1d timeout extends decided win", "budget exhausted" in ba and "win DECIDED" in ba)
    check("1e travel wiped-party finish", "enemy party is WIPED" in tr)
    check("1f trainer budget 420", "420 if trainer else 180" in pl)
    check("1g no mid-battle campaign save", "mid-battle — not banking a fight" in camp)
    check("1h no mid-battle known-good", "mid-battle (refuse fight-rewind escape)" in camp)
    check("1i defer mid-battle CKPT", "AUTO-CHECKPOINT deferred — mid-battle" in camp)
    if fails:
        print(f"FAIL — {fails}")
        sys.exit(1)
    print("ALL PASS — fight-reset rewind guards present.")


if __name__ == "__main__":
    run()
