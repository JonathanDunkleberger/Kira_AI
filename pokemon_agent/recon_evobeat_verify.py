"""recon_evobeat_verify.py — EVOLUTION EARLY BEAT + any-slot drive-gate verification.

The change under test (night shift 11): (a) _drive_evolution gates on ANY party slot
leveling (was lead-only — a fielded bench mon's evolution left the cutscene undriven);
(b) EARLY BEAT: the 'What? X is evolving!' box is read from the gStringVar block DURING
the cutscene and emits ONE beat immediately, so the LLM chain overlaps the ~20-30s
animation (the F-7c level-up-beat design, applied to evolution).

MECHANIC — GUARANTEED EVOLUTION IN ONE BATTLE: the evolution table is read from the ROM
FILE (scan for Bulbasaur's LEVEL-16->Ivysaur entry; base = hit - 40). A party mon is
"past-due" when its CURRENT level >= its LEVEL-method threshold but it hasn't evolved
(the level was gained without crossing... or via prior level-writes). We field the
past-due mon (save-safe _swap_party_slots), decrement its plaintext level byte by 1
(exp untouched — the lvlbeat primitive), and win ONE wild battle: the exp path re-applies
the true level, the game runs its evolution check (level >= threshold), the cutscene
plays, the wrapper's _drive_evolution drives it. In-memory only; READ-ONLY on bundles.

PASS = exactly one "is evolving!" beat emitted (the early beat) AND the fielded slot's
species == the ROM table's target after the drive. FAIL = species changed but no beat
(the gStringVar read missed the box) / beat but no change (drive broke) / >1 (dedup).
INCONCLUSIVE = no bundle offers a past-due evolver, or no battle/win.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_evobeat_verify.py [bundle...]
     (default candidates: banked_POSTGAME, banked_E4, banked_VICTORY, banked_SILPH)
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
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
CANDIDATES = ["banked_POSTGAME", "banked_E4", "banked_VICTORY", "banked_SILPH"]
EVO_LEVEL = 4                          # pret EVO_LEVEL method id
EVO_MARKER = "is evolving"


def rom_evo_table():
    """{species: (threshold_level, target_species)} for LEVEL-method evolutions, read from
    the ROM file. Base found by scanning for Bulbasaur's entry (04 00 10 00 02 00 00 00);
    stride 40 (5 evolutions x 8 bytes). No emulator, no hand-table."""
    with open(ROM, "rb") as f:
        rom = f.read()
    hit = rom.find(bytes((0x04, 0x00, 0x10, 0x00, 0x02, 0x00, 0x00, 0x00)))
    assert hit > 0, "evolution table pattern not found in ROM"
    base = hit - 40                                    # species 0 is an all-zero row
    out = {}
    for sp in range(1, 412):
        for e in range(5):
            o = base + sp * 40 + e * 8
            method = int.from_bytes(rom[o:o + 2], "little")
            param = int.from_bytes(rom[o + 2:o + 4], "little")
            target = int.from_bytes(rom[o + 4:o + 6], "little")
            if method == EVO_LEVEL and 0 < target <= 411:
                out[sp] = (param, target)
                break
    return out


def _party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    return [(st.read_party_species(b, s),
             b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54))
            for s in range(min(cnt, 6))]


def _lead_level_rmw(b, delta):
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
    bundles = sys.argv[1:] or CANDIDATES
    evo = rom_evo_table()
    print(f"==== EVOLUTION EARLY BEAT verify ==== (ROM LEVEL-evos: {len(evo)} species)",
          flush=True)

    b = camp = None
    slot = threshold = target = None
    for name in bundles:
        p = os.path.join(LONGRUN, name, "kira_campaign.state")
        if not os.path.exists(p):
            continue
        bb = Bridge(ROM)
        with open(p, "rb") as f:
            bb.load_state(f.read())
        for _ in range(240):
            bb.run_frame()
        bb.set_input_owner("agent")
        party = _party(bb)
        pick = next(((s, lv, evo[sp][0], evo[sp][1]) for s, (sp, lv) in enumerate(party)
                     if sp in evo and lv > evo[sp][0]), None)   # strictly past-due (relevel
        #                                                         must land ABOVE threshold)
        names = [(st.SPECIES_NAME.get(sp, sp), lv) for sp, lv in party]
        print(f"   {name}: party={names} -> "
              f"{'ELIGIBLE slot ' + str(pick[0]) if pick else 'no past-due evolver'}",
          flush=True)
        if pick:
            slot, _, threshold, target = pick
            b = bb
            bundle = name
            break
        del bb
    if b is None:
        print("RESULT: INCONCLUSIVE — no candidate bundle has a past-due LEVEL evolver",
              flush=True)
        sys.exit(2)

    events = []

    def runner():
        return BattleAgent(b, on_event=lambda s, **k: events.append(s), render=lambda: None,
                           log=lambda m: print(f"   {m}", flush=True)).run(max_seconds=150)

    camp = Campaign(b, battle_runner=runner,
                    on_event=lambda s, **k: (events.append(s),
                                             print(f"   [event] {s}", flush=True)),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True      # READ-ONLY: never bank from a verify
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    camp._suppress_heal = True

    if slot != 0:
        camp._swap_party_slots(0, slot)
        print(f"   fielded past-due evolver from slot {slot} -> 0", flush=True)
    sp0 = st.read_party_species(b, 0)
    true_lvl, armed = _lead_level_rmw(b, -1)
    print(f"   lead = {st.SPECIES_NAME.get(sp0, sp0)} true L{true_lvl} -> armed L{armed} "
          f"(evo threshold {threshold}, target {st.SPECIES_NAME.get(target, target)})",
          flush=True)

    # reach grass: exit any building, then hop north (Pallet -> Route 1) until the map has grass
    if tuple(tv.map_id(b))[0] != 3:
        camp._exit_to_overworld()
    hops = {tv.MAP_PALLET: tv.MAP_ROUTE1}
    for _ in range(3):
        if tv.Grid(b).grass:
            break
        nxt = hops.get(tuple(tv.map_id(b)))
        if nxt is None:
            break
        camp.trav.travel(target_map=nxt, max_steps=600, max_seconds=240, edge="north")
    off = tv.MAP_OFFSET
    gs = [(x - off, y - off) for (x, y) in tv.Grid(b).grass]
    if not gs:
        print(f"RESULT: INCONCLUSIVE — no grass reachable from {tv.map_id(b)}", flush=True)
        sys.exit(2)

    def fight_open():
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    cur = tv.coords(b) or (0, 0)
    tgt = min(gs, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
    camp.trav.travel(arrive_coord=tgt, max_steps=250, max_seconds=150)
    gset = set(gs)
    fought = False
    for _pace in range(150):
        if fight_open():
            camp.battle_runner()       # the OBSERVED wrapper: fights, then _drive_evolution
            fought = True
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
        if not moved:
            camp.trav.travel(arrive_coord=tgt, max_steps=60, max_seconds=45)
        if fight_open():
            camp.battle_runner()
            fought = True
            break
    if not fought:
        print("RESULT: INCONCLUSIVE — no wild battle triggered", flush=True)
        sys.exit(2)

    post0 = st.read_party_species(b, 0)
    beats = [e for e in events if EVO_MARKER in e.lower()]
    fails = []
    if post0 != target:
        if post0 == sp0 and b.rd8(ram.GPLAYER_PARTY + 0x54) == armed:
            print("RESULT: INCONCLUSIVE — no exp landed (battle lost/fled?); level still "
                  "armed", flush=True)
            sys.exit(2)
        fails.append(f"species after drive = {st.SPECIES_NAME.get(post0, post0)} "
                     f"(want {st.SPECIES_NAME.get(target, target)}) — cutscene not driven "
                     f"to completion")
    if not beats:
        fails.append("evolution completed but NO early beat fired (gStringVar box read "
                     "missed 'is evolving!')")
    elif len(beats) != 1:
        fails.append(f"early-beat count = {len(beats)} (want exactly 1)")

    if fails:
        print("==== RESULT: FAIL ====", flush=True)
        for f in fails:
            print(f"   !! {f}", flush=True)
        sys.exit(1)
    print(f"==== RESULT: PASS ==== ({beats[0]!r} fired mid-cutscene; "
          f"{st.SPECIES_NAME.get(sp0, sp0)} -> {st.SPECIES_NAME.get(post0, post0)})",
          flush=True)


if __name__ == "__main__":
    main()
