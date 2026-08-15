"""recon_revive_land.py — E2E PROOF that a Revive gets aimed, confirmed, consumed, and resurrects.

The wall this closes (live 17:3x Lorelei, docs/soak-reports/20260814_173116): 6 Revives in the bag,
a dead Zapdos at party row 2, three attempts, every one bailing with
    "REFUSE A - still on living lead after RIGHT/DOWN"
Cause: the item-use party walk trusted PARTY_CURSOR 0x02020777, which on THAT screen is the
highlight's blink counter, not a cursor (recon_revive_cursor2). It read the target row number by
coincidence on lap 1, broke out of the walk without pressing a single D-pad, and the 0-HP guard
then correctly refused to A the living lead.

This runs the REAL BattleAgent.use_item_in_battle on the baked Lorelei fixture and demands the
whole chain: returns 'used', the bag count drops, and the corpse reads hp > 0.

Also runs the ANTI-REGRESSION half: with NOBODY fainted the same call must refuse to fire
(no_effect/failed) and must NOT burn a Revive on a living mon.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_land.py
     DEAD_SLOT=3 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_land.py   (any row)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                 # noqa: E402
import firered_ram as ram                 # noqa: E402
import pokemon_state as st                # noqa: E402
from battle_agent import BattleAgent      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "e4_revive_stage.state")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_land")
REVIVE = 24
RAW = None


def log(m):
    print(m, flush=True)


def hp(b, slot):
    return b.rd16(ram.GPLAYER_PARTY + slot * 100 + 0x56)


def boot():
    global RAW
    if RAW is None:
        with open(SRC, "rb") as f:
            RAW = f.read()
    b = Bridge(ROM)
    b.load_state(RAW)
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=log)
    ag.owner = "agent"
    return b, ag


def party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    return [(i, st.SPECIES_NAME.get(st.read_party_species(b, i), "?"), hp(b, i))
            for i in range(min(cnt, 6))]


def case_revive(dead_slot):
    """The real thing: a corpse at `dead_slot`, aim='fainted', demand a resurrection."""
    log("")
    log("=" * 78)
    log(f"CASE: revive the corpse at party row {dead_slot}")
    log("=" * 78)
    b, ag = boot()
    # the fixture already kills row 2; move the corpse if the caller asked for another row
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        want = 0 if s == dead_slot else b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56, want)
    n0 = ag._items_count(REVIVE)
    log(f"before: party={party(b)} revives={n0}")
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    hp_after = hp(b, dead_slot)
    log(f"after:  party={party(b)} revives={n1}  result={res!r}  "
        f"row{dead_slot}.hp={hp_after}")
    os.makedirs(OUT, exist_ok=True)
    b.frame_rgb().save(os.path.join(OUT, f"revive_row{dead_slot}.png"))
    ok = (res == "used" and n1 == n0 - 1 and hp_after > 0)
    log(f"  -> {'PASS' if ok else 'FAIL'} "
        f"(want result='used', revives {n0}->{n0 - 1}, row{dead_slot}.hp>0)")
    del b
    return ok


def case_nobody_down():
    """Anti-regression: nobody fainted -> must NOT spend a Revive on a living mon."""
    log("")
    log("=" * 78)
    log("CASE: nobody is down — the Revive must NOT be spent")
    log("=" * 78)
    b, ag = boot()
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56,
                                    b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58))
    n0 = ag._items_count(REVIVE)
    log(f"before: party={party(b)} revives={n0}")
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    log(f"after:  party={party(b)} revives={n1}  result={res!r}")
    ok = (n1 == n0 and res != "used")
    log(f"  -> {'PASS' if ok else 'FAIL'} (want the count UNCHANGED and result != 'used')")
    del b
    return ok


def case_battle_still_fightable():
    """After a successful revive the battle must be left in a fightable state (no menu wedge)."""
    log("")
    log("=" * 78)
    log("CASE: after the revive, the battle is still fightable (action menu reachable)")
    log("=" * 78)
    b, ag = boot()
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    ok_menu = ag._settle_action_menu()
    log(f"result={res!r} in_battle={st.in_battle(b)} action_menu_reachable={ok_menu} "
        f"bag_open={ag._bag_screen()} party_open={ag._party_screen()}")
    ok = bool(res == "used" and st.in_battle(b) and ok_menu)
    log(f"  -> {'PASS' if ok else 'FAIL'}")
    del b
    return ok


def case_instinct_path():
    """THE LIVE PATH: don't call use_item_in_battle directly — run the real battle loop and
    make the ITEM-INSTINCT fire the revive itself, exactly as it does on stream. This is the
    path the 17:3x Lorelei log walked (offer -> FORCED -> pick -> aim -> REFUSE A)."""
    log("")
    log("=" * 78)
    log("CASE: the LIVE instinct path — real battle loop offers and fires the revive")
    log("=" * 78)
    b, ag = boot()
    n0 = ag._items_count(REVIVE)
    dead = [s for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)) if hp(b, s) == 0]
    log(f"before: party={party(b)} revives={n0} dead_rows={dead}")

    lines = []

    def cap(m):
        lines.append(m)
        print(m, flush=True)

    seen = {"offered": 0, "picked": 0}

    def choose(ptype, offers, ctx):
        if "use_revive" in offers:
            seen["offered"] += 1
            seen["picked"] += 1
            return "use_revive"
        return "keep_fighting"

    res = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                      log=cap, choose=choose).run(max_seconds=240)
    n1 = ag._items_count(REVIVE)
    used_n = sum(1 for ln in lines if "use_item: USED item 24" in ln)
    landed = [ln for ln in lines if "revive WALK landed" in ln and "hp=0)" in ln]
    refused = [ln for ln in lines if "REFUSE A" in ln]
    latched = [ln for ln in lines if "REVIVE-FAIL LATCH" in ln]
    nobag = [ln for ln in lines if "bag never opened" in ln or "bag wouldn't open" in ln]
    log(f"after:  battle={res!r} party={party(b)} revives={n0}->{n1}")
    log(f"        revives consumed={used_n}  0-HP landings={len(landed)}  "
        f"REFUSE-A={len(refused)}  battle-wide LATCH={len(latched)}")
    log(f"        (separate pre-existing defect) bag-never-opened turns={len(nobag)}")
    for ln in (refused + latched)[:6]:
        log(f"        {ln.strip()}")
    # What this case proves is the AIMING contract, so assert exactly that:
    #   * at least one Revive actually got consumed through the live instinct path
    #   * every Revive we spent was confirmed onto a 0-HP row before the A
    #   * we never pressed A on a living mon (REFUSE-A is the guard firing, i.e. a miss)
    #   * the battle-wide revive latch never armed (that was the "3 Revives sat unused" wall)
    # Do NOT assert 'the corpse is alive at the END' — a revived mon can (and here does) faint
    # again later in the same fight. And do NOT gate on the bag-open flake: it is a distinct
    # defect (it costs a turn and self-retries) tracked separately.
    ok = bool(used_n > 0 and n1 < n0 and len(landed) >= used_n
              and not refused and not latched)
    log(f"  -> {'PASS' if ok else 'FAIL'} (want >=1 consumed, every one landed on a 0-HP row, "
        f"no REFUSE-A, no battle-wide latch)")
    log(f"  ..  battle outcome was {res!r}")
    del b
    return ok


def main():
    if not os.path.exists(SRC):
        log(f"!! fixture missing: {SRC}\n   run: python -u pokemon_agent/recon_revive_stage.py")
        return 1
    os.makedirs(OUT, exist_ok=True)
    results = {}
    only = os.environ.get("DEAD_SLOT")
    slots = [int(only)] if only else [1, 2, 3]
    for s in slots:
        results[f"revive_row{s}"] = case_revive(s)
    results["nobody_down"] = case_nobody_down()
    results["still_fightable"] = case_battle_still_fightable()
    results["instinct_path"] = case_instinct_path()
    log("")
    log("=" * 78)
    for k, v in results.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    log(f"VERDICT: {'ALL PASS' if ok else 'FAIL'}")
    log("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
