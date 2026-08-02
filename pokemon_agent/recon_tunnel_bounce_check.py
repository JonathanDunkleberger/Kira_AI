"""recon_tunnel_bounce_check.py — Rock Tunnel must not Flash→walk-out the entry (no ROM)."""
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
    print("recon_tunnel_bounce_check")
    check("1a ENTRY mouth ban", "ENTRY mouth BANNED for re-exit" in src)
    check("1b defer exits until interior ride", "deferring" in src and "anti bounce-out" in src)
    check("1c farthest exit not nearest", "farthest of" in src)
    check("1d bounce-out detect in tunnel leg", "BOUNCE-OUT" in src and "near entry mouth" in src)
    check("1e refuse entry backtrack", "refusing ENTRY-mouth backtrack" in src)
    if fails:
        print(f"FAIL — {fails}")
        sys.exit(1)
    print("ALL PASS — Rock Tunnel anti-bounce wiring present.")


if __name__ == "__main__":
    run()
