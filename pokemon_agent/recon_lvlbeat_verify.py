"""recon_lvlbeat_verify.py — F-7(c) slice 2 LEVEL-UP EARLY BEAT verification (printed events).

The change under test (battle_agent 94157e6): the party level byte (+0x54) flips the moment
the level-up APPLIES — while "grew to LV. N!" is still on screen — and the agent now emits ONE
"my X just leveled up to level N" beat inside the post-faint drain (any slot), so the LLM chain
overlaps the drain and her line lands on the jingle. The grade harness's on_event doesn't
print, so this is the printed-events run the wiring has owed since shift 5.

MECHANIC — GUARANTEED LEVEL-UP IN ONE BATTLE: pre-battle (overworld, never mid-fight) we
decrement the lead's raw level byte by 1 (u32 read-modify-raw_write, the _swap_party_slots
primitive family). Exp is untouched, so it already sits past the threshold for the true level:
the game's exp-gain path re-applies the level on the FIRST exp it awards → a real "grew to
LV!" box in battle #1, no 12-battle bench grind. Stats recalc to the values they already had.
In-memory only; READ-ONLY on the bundle (never saves state).

PASS = exactly one "just leveled up to level" emit, fired with in_battle=True (in the drain,
not after exit), naming the true (restored) level. FAIL = no emit (detect missed the flip),
emit after exit (no lead time bought), or >1 (dedup broke).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_lvlbeat_verify.py [bundle]
     (default bundle: banked_HM05 — Route 2 grass adjacent, lead one-shots L2-5 wilds)
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge              # noqa: E402
import firered_ram as ram              # noqa: E402
import pokemon_state as st             # noqa: E402
import travel as tv                    # noqa: E402
from battle_agent import BattleAgent   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
LVL_MARKER = "just leveled up to level"
FAILS = []


def _load(bundle):
    b = Bridge(ROM)
    with open(os.path.join(LONGRUN, bundle, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()
    b.set_input_owner("agent")
    return b


def _lead_level_rmw(b, delta):
    """Read-modify-write the lead's level byte (+0x54) via the u32 word that holds it —
    the same raw_write primitive _swap_party_slots uses. Overworld only."""
    addr = ram.GPLAYER_PARTY + 0x54
    word_addr = addr & ~3
    shift = (addr & 3) * 8
    w = b.rd32(word_addr)
    lvl = (w >> shift) & 0xFF
    new = lvl + delta
    assert 2 <= new <= 100, f"refusing level write {lvl} -> {new}"
    b.core.memory.u32.raw_write(word_addr, (w & ~(0xFF << shift)) | (new << shift))
    return lvl, new


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "banked_HM05"
    print(f"==== LEVEL-UP EARLY BEAT verify (bundle={bundle}) ====", flush=True)
    from campaign import Campaign
    b = _load(bundle)
    print(f"   boot map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

    lead_sp = st.read_party_species(b, 0)
    lead_nm = st.SPECIES_NAME.get(lead_sp, f"species#{lead_sp}")
    true_lvl, armed_lvl = _lead_level_rmw(b, -1)
    print(f"   lead = {lead_nm} true L{true_lvl} -> armed at L{armed_lvl} "
          f"(exp untouched; first exp gain re-levels to {true_lvl})", flush=True)

    events = []
    result = {}

    def sink(summary, **k):
        events.append((time.time(), st.in_battle(b), summary))
        print(f"   [event] in_battle={st.in_battle(b)} :: {summary}", flush=True)

    def battle_runner():
        agent = BattleAgent(b, on_event=sink, render=lambda: None,
                            log=lambda m: print(f"   {m}", flush=True))
        out = agent.run(max_seconds=180)
        result["outcome"] = out
        t0 = time.time()
        for _ in range(1200):
            if not st.in_battle(b):
                break
            b.run_frame()
        print(f"   [check] outcome={out}; battle exited {time.time() - t0:.1f}s after run() "
              f"returned; lead level now "
              f"L{b.rd8(ram.GPLAYER_PARTY + 0x54)}", flush=True)
        return "loss"     # stop travel after ONE battle (recon_catch diagnosis pattern)

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None,
                    render=lambda: None)
    camp._save_campaign = lambda *a, **k: True     # READ-ONLY: no banking from a verify
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    camp._suppress_heal = True

    def fight_open():
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    # travel deliberately routes AROUND grass — walk TO the nearest grass tile on the boot map,
    # then PACE it until a wild fires (the recon_winbeat_verify phase_wild pattern, verbatim).
    off = tv.MAP_OFFSET
    gs = [(x - off, y - off) for (x, y) in tv.Grid(b).grass]
    if not gs:
        print("RESULT: INCONCLUSIVE — no grass on this map", flush=True)
        sys.exit(2)
    cur = tv.coords(b) or (0, 0)
    target = min(gs, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
    camp.trav.travel(arrive_coord=target, max_steps=250, max_seconds=150)
    gset = set(gs)
    for _pace in range(120):
        if "outcome" in result:
            break
        if fight_open():
            battle_runner()
            break
        c = tv.coords(b) or (0, 0)
        moved = False
        for (dx, dy, key) in ((0, 1, "DOWN"), (0, -1, "UP"), (1, 0, "RIGHT"), (-1, 0, "LEFT")):
            if (c[0] + dx, c[1] + dy) in gset:
                b.press(key, 26, 10, lambda: None, owner="agent")
                for _ in range(45):
                    b.run_frame()
                    if fight_open():
                        break
                moved = True
                break
        if not moved:      # walked off the patch — return to the target tile
            camp.trav.travel(arrive_coord=target, max_steps=60, max_seconds=45)
        if fight_open():
            battle_runner()
            break

    if "outcome" not in result:
        print("RESULT: INCONCLUSIVE — no wild battle triggered (grass unreached?)", flush=True)
        sys.exit(2)

    beats = [(t, ib, s) for (t, ib, s) in events if LVL_MARKER in s]
    if result.get("outcome") != "win":
        print(f"RESULT: INCONCLUSIVE — battle outcome={result['outcome']} (need a win for exp)",
              flush=True)
        sys.exit(2)
    if not beats:
        FAILS.append("won with an armed level flip but NO level-up beat fired (detect missed)")
    else:
        if len(beats) != 1:
            FAILS.append(f"level-up beat count = {len(beats)} (want exactly 1 — dedup broke)")
        if not beats[0][1]:
            FAILS.append("level-up beat fired AFTER battle exit (in_battle=False) — bought no "
                         "lead time; the play_live fallback already did that")
        # the named level must be >= the true level (a battle can push past it if exp sat
        # just under the NEXT threshold too — the box chain shows both, the beat fires once)
        try:
            named = int(beats[0][2].rsplit("level", 1)[1].strip().rstrip(".!"))
        except Exception:
            named = -1
        if named < true_lvl:
            FAILS.append(f"beat names the wrong level: {beats[0][2]!r} (true level {true_lvl})")

    if FAILS:
        print("==== RESULT: FAIL ====", flush=True)
        for f in FAILS:
            print(f"   !! {f}", flush=True)
        sys.exit(1)
    print(f"==== RESULT: PASS ==== ({beats[0][2]!r} fired in the drain)", flush=True)


if __name__ == "__main__":
    main()
