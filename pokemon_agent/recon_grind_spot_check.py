"""NS#5 decision-check (emulator-free) for the GRIND-SPOT LEVEL KB + inadequacy predicate.

PASS-3 grind-efficiency lever (a): a level-aware picker must recognise a GRIND-INADEQUATE map — one
whose wild_max is far below the team's grind target (the documented NS#1/#14 E4-prep stall: an L45->55
mon on L8-19 grass gains ~0 XP, so grind() spins its whole budget for ~0 levels). This verifies the KB
reader (_grind_wild_band) + the predicate (_grind_inadequate) deterministically, with NO emulator:
a bare Campaign instance (via __new__) exercises only the JSON-backed methods.

The PICKER WIRING (mark such a map grind-inadequate + prefer a reachable higher-level spot) is the
verify-gated next step — it must be proven on a giovanni_kit_g look-ahead before POKEMON_GRIND_SPOT_LEVELAWARE
is flipped ON. This check guards the KB + predicate so that wiring builds on a proven base.

RUN: ../.venv/Scripts/python.exe recon_grind_spot_check.py
"""
import campaign

c = campaign.Campaign.__new__(campaign.Campaign)   # no __init__ -> no emulator; methods only read the KB

R6, R7, R18, R23 = (3, 24), (3, 25), (3, 36), (3, 42)
VICTORY_ROAD, CERULEAN_CAVE, UNKNOWN = (1, 39), (1, 72), (9, 9)

# (label, callable -> actual, expected)
CASES = [
    # --- _grind_wild_band: reads the KB band, unknown -> None (fail-open) ---
    ("band Route 6",            lambda: c._grind_wild_band(R6),          (13, 16)),
    ("band Route 7",            lambda: c._grind_wild_band(R7),          (18, 20)),
    ("band Route 18",           lambda: c._grind_wild_band(R18),         (24, 29)),
    ("band Route 23",           lambda: c._grind_wild_band(R23),         (34, 49)),
    ("band Victory Road",       lambda: c._grind_wild_band(VICTORY_ROAD),(36, 46)),
    ("band Cerulean Cave",      lambda: c._grind_wild_band(CERULEAN_CAVE),(46, 67)),
    ("band UNKNOWN -> None",    lambda: c._grind_wild_band(UNKNOWN),     None),

    # --- _grind_inadequate: the documented E4-prep stall (target ~55) ---
    ("E4-prep(55): Route 18 INADEQUATE",   lambda: c._grind_inadequate(R18, 55),           True),
    ("E4-prep(55): Route 6 INADEQUATE",    lambda: c._grind_inadequate(R6, 55),            True),
    ("E4-prep(55): Route 23 adequate",     lambda: c._grind_inadequate(R23, 55),           False),
    ("E4-prep(55): Victory Road adequate", lambda: c._grind_inadequate(VICTORY_ROAD, 55),  False),
    ("E4-prep(55): Cerulean Cave adequate",lambda: c._grind_inadequate(CERULEAN_CAVE, 55), False),

    # --- mid-game targets: Route 18/7 are FINE (never falsely block a legit mid-game grind) ---
    ("mid(22): Route 18 adequate",         lambda: c._grind_inadequate(R18, 22),           False),
    ("mid(22): Route 7 adequate",          lambda: c._grind_inadequate(R7, 22),            False),
    ("badge5(30): Route 7 INADEQUATE",     lambda: c._grind_inadequate(R7, 30),            False),  # 20 >= 30-18=12
    ("badge5(45): Route 7 INADEQUATE",     lambda: c._grind_inadequate(R7, 45),            True),   # 20 < 45-18=27

    # --- fail-open: unknown map / no target never blocks a grind ---
    ("UNKNOWN map never inadequate",       lambda: c._grind_inadequate(UNKNOWN, 55),       False),
    ("no target never inadequate",         lambda: c._grind_inadequate(R18, None),         False),
    ("target 0 never inadequate",          lambda: c._grind_inadequate(R18, 0),            False),
]

fails = 0
for label, fn, expected in CASES:
    try:
        actual = fn()
    except Exception as e:
        actual = f"EXC:{e}"
    ok = actual == expected
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {actual!r} want {expected!r}")

n = len(CASES)
print(f"\n{n - fails}/{n} {'ALL PASS' if not fails else 'FAILURES'}  (POKEMON_GRIND_POOR_GAP={campaign.GRIND_POOR_GAP})")
raise SystemExit(1 if fails else 0)
