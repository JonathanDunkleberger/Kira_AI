"""recon_solo_overlevel_check.py — decision-logic verifier for PASS-3 NS#26 team-depth lever:
the SELECTIVE SOLO gate in battle_agent (BattleAgent._solo_overlevel_ok + the GRIND SWITCH block).

The participation GRIND SWITCH hands the KO to the ace, so the fielded weak lead banks only a SHARE of
participation XP — the real throttle behind the slow endgame bench climb (measured baseline: floor L28->L31
over ~21 min sim on the ns25 look-ahead). The lever: when the weak lead SAFELY out-levels THIS foe it
one-shots the wild, so SKIP the switch and let it SOLO for the FULL kill XP (~2x the share). It SUPPRESSES a
switch (never adds one) and keeps PROTECT_LEAD_GRIND True so the MATCHUP switch stays off (no strand/churn).

Proves WITHOUT a live emulator (the flip still needs a look-ahead) that the gate fires on exactly the intended
shape and is byte-inert OFF, by calling the REAL _solo_overlevel_ok against synthetic battle states, plus a
SOURCE-STRUCTURE assertion that the solo branch never mutates PROTECT_LEAD_GRIND (the wedge-safety invariant).

RUN:  ../.venv/Scripts/python.exe -u recon_solo_overlevel_check.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import battle_agent as B          # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


def _ok(state):
    # _solo_overlevel_ok touches no instance state -> call unbound with self=None.
    return B.BattleAgent._solo_overlevel_ok(None, state)


def _st(our, foe):
    return {"ours": {"level": our}, "enemy": {"level": foe}}


def run():
    # --- defaults (flipped default-ON 2026-07-12 NS#26 after the look-ahead) ---
    check("default flag ON", B.SOLO_OVERLEVEL_GRIND, True)
    check("default margin 8", B.SOLO_OVERLEVEL_MARGIN, 8)

    # --- flag OFF: byte-inert regardless of the level gap ---
    B.SOLO_OVERLEVEL_GRIND = False
    B.SOLO_OVERLEVEL_MARGIN = 8
    check("1 flag OFF -> never solo (huge gap)", _ok(_st(60, 20)), False)

    # --- flag ON ---
    B.SOLO_OVERLEVEL_GRIND = True
    check("2 gap 10 >= margin 8 -> solo", _ok(_st(30, 20)), True)
    check("3 gap 8 == margin 8 -> solo (>=)", _ok(_st(30, 22)), True)
    check("4 gap 5 < margin 8 -> PROTECT (switch)", _ok(_st(30, 25)), False)
    check("5 gap 0 (even) -> PROTECT", _ok(_st(30, 30)), False)
    check("6 lead UNDER foe -> PROTECT (the fragile early-game case)", _ok(_st(20, 30)), False)

    # --- defensive / missing data -> PROTECT (never solo on a bad read) ---
    check("7 state None -> PROTECT", _ok(None), False)
    check("8 missing our level -> PROTECT", _ok({"enemy": {"level": 10}}), False)
    check("9 missing foe level -> PROTECT", _ok({"ours": {"level": 40}}), False)
    check("10 zero levels -> PROTECT", _ok(_st(0, 0)), False)

    # --- margin env override raises the bar (per-foe self-correcting) ---
    B.SOLO_OVERLEVEL_MARGIN = 12
    check("11 gap 10 < margin 12 -> PROTECT", _ok(_st(30, 20)), False)
    check("12 gap 12 >= margin 12 -> solo", _ok(_st(32, 20)), True)

    # --- WEDGE-SAFETY INVARIANT: the solo branch must SUPPRESS a switch (ace=None), never mutate
    #     PROTECT_LEAD_GRIND (which would re-arm the matchup switch gated on `not PROTECT_LEAD_GRIND`). ---
    src = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
    m = re.search(r"if ace is not None and self\._solo_overlevel_ok\(state\):(.*?)\n(?= {20}# ALREADY-ACE)",
                  src, re.S)
    check("13 solo gate present in GRIND SWITCH block", m is not None, True)
    if m:
        body = m.group(1)
        check("13a solo branch suppresses the switch (ace = None)", "ace = None" in body, True)
        check("13b solo branch never mutates PROTECT_LEAD_GRIND", "PROTECT_LEAD_GRIND" not in body, True)

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — selective solo fires iff the weak lead out-levels the foe by >= margin, byte-inert OFF, "
          "defensive on bad reads, margin-tunable, and the solo branch suppresses a switch without touching "
          "PROTECT_LEAD_GRIND (matchup switch stays off).")


if __name__ == "__main__":
    run()
