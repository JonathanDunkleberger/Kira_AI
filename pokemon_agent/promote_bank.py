"""promote_bank.py — the standing sanctity-gated PROMOTION of a banked checkpoint to canonical.

The persistent-world law (CLAUDE.md rule 18): exactly ONE canonical Sherpa state; staging banks
only move forward into it through THIS gate. What it does:
  1. sanctity.validate_bundle(bank, prev_dir=canonical) — schema/encoding/truth/MONOTONIC (the
     lost-Gary-win class is caught here; a FAILED validation aborts, nothing is touched).
  2. Backs up the whole canonical bundle to states/campaign/pre_<label>_backup_<ts>/.
  3. Copies the bank's bundle (state + 4 sidecars) over canonical.
  4. ROUND-TRIP verify: loads the promoted .state in a fresh core and prints map/coords/party —
     eyeball these against the bank's known truth before trusting the promotion.

RUN:  python pokemon_agent/promote_bank.py [bank_dir] [label]
  bank_dir default = %TEMP%/longrun/banked_GOAL ; label default = "promo".
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sanctity                                            # noqa: E402

CANON = os.path.join(_HERE, "states", "campaign")
BUNDLE = ("kira_campaign.state",) + sanctity.SIDECARS


def main():
    bank = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("TEMP", _HERE), "longrun", "banked_GOAL")
    label = sys.argv[2] if len(sys.argv) > 2 else "promo"
    if not os.path.isdir(bank):
        print(f"!! bank dir missing: {bank}"); return 1
    missing = [f for f in BUNDLE if not os.path.exists(os.path.join(bank, f))]
    if missing:
        print(f"!! bank bundle incomplete — missing {missing}; NOT promoting"); return 1

    ok, issues = sanctity.validate_bundle(bank, prev_dir=CANON, log=print)
    if not ok:
        print(f"!! SANCTITY FAILED — DO NOT PROMOTE: {issues}"); return 1
    print("sanctity: VALID (vs current canonical, monotonic included)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(CANON, f"pre_{label}_backup_{ts}")
    os.makedirs(backup, exist_ok=True)
    for f in BUNDLE + sanctity.OPTIONAL_SIDECARS:
        src = os.path.join(CANON, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup, f))
    print(f"canonical backed up -> {backup}")

    for f in BUNDLE:
        shutil.copy2(os.path.join(bank, f), os.path.join(CANON, f))
    for f in sanctity.OPTIONAL_SIDECARS:                 # copy-if-present, never required
        src = os.path.join(bank, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(CANON, f))
    print(f"PROMOTED {bank} -> {CANON}")

    # round-trip verify
    from bridge import Bridge                              # noqa: E402  (heavy import last)
    import travel as tv                                    # noqa: E402
    import firered_ram as ram                              # noqa: E402
    import pokemon_state as st                             # noqa: E402
    b = Bridge(os.path.join(os.path.dirname(_HERE), "roms", "firered.gba"))
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    party = [(st.SPECIES_NAME.get(st.read_party_species(b, s), "?"),
              b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54),
              b.rd16(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x56))
             for s in range(min(cnt, 6))]
    print(f"ROUND-TRIP: map={tv.map_id(b)} coords={tv.coords(b)} party={party}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
