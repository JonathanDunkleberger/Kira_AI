"""recon_grind_bench.py — BENCH POWER-LEVEL (soul-debt #3, load-bearing at the finish).

THE WALL this pays: the E4 gauntlet (L53-63) + Champion Gary's Charizard type-wall are not
beatable by a solo Venusaur + L8-16 fodder. NS6's convergent-VR push proved the bad path:
it leveled the SOLO carry (Venusaur L60->L65) but NEVER the bench (each trainer fight solos
Venusaur; the participation-switch only fires while GRINDING), and it bled money $36k->$4.5k
across wipes. The right path (CEO analysis, data-confirmed): from the CLEAN rich giovanni_done_kit
($36k, full PP), GRIND the two real teammates — Lapras (Water: Charizard + Lance answer) and
Kadabra (Psychic: Agatha + Bruno answer) — up to a competitive level BEFORE Victory Road, so VR
trainers + the E4 fall first-try with no wipes and no money loss.

MECHANIC (participation-XP grind, no PC-box needed): field ONE target teammate as lead, arm
`battle_agent.PROTECT_LEAD_GRIND` so every wild battle SWITCHES the weak lead out to the ace
(Venusaur) turn 1 — the weak mon participated (banks XP) without eating the hit, Venusaur tanks
and KOs. camp.grind(target, fragile=True) loops this until the LEAD (slot 0) crosses `target`.
We drive it per-teammate (Lapras, then Kadabra) so the fodder (Rattata/Spearow/Drowzee) is never
fielded — the grind never wastes XP on dead weight, and no boxing is required for this phase.
Grinding ALSO re-levels the ace (Venusaur gets participation XP too), so it keeps pace.

Bench-grind ALSO fixes movesets for free: Kadabra learns stronger Psychic moves by level-up;
Lapras firms up as a bulky Water wall.

Boot = kit fixture (GRIND_STATE env, default giovanni_done_kit.state) resolved from states/workshop
with its <name>.<sidecar>.json continuity. Grind spot = GRIND_MAP "group,num" + GRIND_DIR edge
(walk there first); default = Route 22 (3,41) west of Viridian (guaranteed grass + Viridian Center
adjacent — a MECHANIC PROBE; relocate to a strong-wild spot once the switch is confirmed leveling
the bench). Banks every re-entry -> %TEMP%/longrun/banked_GRIND (promote -> bench_grind_kit).

RUN (from pokemon_agent/):
  GRIND_TARGET=42 ../.venv/Scripts/python.exe -u recon_grind_bench.py
  GRIND_STATE=giovanni_done_kit.state GRIND_MAP=3,41 GRIND_DIR=west GRIND_MIN=25 ... recon_grind_bench.py
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.environ.get("WATCH") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_GRIND_SWITCH", "1")   # participation-XP switch (the whole point) ON
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cp1252 pipe vs travel emoji
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
import battle_agent                  # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_grind")
BANK = os.path.join(SCRATCH, "banked_GRIND")

LAPRAS, KADABRA, ABRA = 131, 64, 63  # teammates worth leveling (species ids; Abra->Kadabra @L16)


def _resolve_state(name):
    """BASENAME or path -> (state_path, sidecar_dir, sidecar_prefix). Kit fixtures live in
    states/workshop as <name>.state + <name>.<sidecar>.json."""
    if not name:
        return (os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign")
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            base = os.path.basename(cand)
            pref = base[:-6] if base.endswith(".state") else base
            return (cand, d, pref)
    return (name, os.path.dirname(name) or CANON, "kira_campaign")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    TARGET = int(os.environ.get("GRIND_TARGET", "42"))
    # PRIORITY-ORDERED species to grind (grind each to TARGET before the next). Default = Lapras first
    # (the essential Charizard/Lance answer), then the psychic mon (Abra->Kadabra), then Kadabra.
    GRIND_SPECIES = [int(x) for x in os.environ.get("GRIND_SPECIES", f"{LAPRAS},{ABRA},{KADABRA}").split(",")]
    BUDGET_MIN = float(os.environ.get("GRIND_MIN", "40"))
    PROBE_S = int(os.environ.get("GRIND_PROBE_S", "300"))
    # fragile=False by default: the participation-switch swaps the weak lead OUT turn 1, so it never
    # takes a hit / never faints -> the fragile one-way-strand filter (which refuses Route-15's grass
    # from its west-edge entry) is needless here. The ace (Venusaur) tanks + heals at the adjacent
    # Fuchsia Center. Set GRIND_FRAGILE=1 only for a Center-less/ledge-locked spot.
    FRAGILE = os.environ.get("GRIND_FRAGILE", "0") == "1"
    b = Bridge(ROM)
    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("GRIND_STATE", "giovanni_done_kit.state"))
    _boot = state_path
    if os.environ.get("RESUME_STAGE") == "1":
        _sp = os.path.join(STAGE, "kira_campaign.state")
        if os.path.exists(_sp):
            _boot = _sp
    with open(_boot, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    L(f"boot state = {_boot}  TARGET=L{TARGET}  budget={BUDGET_MIN}min")

    n_battles = [0]

    def fight():
        n_battles[0] += 1
        r = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                        log=lambda m: print(m, flush=True)).run(max_seconds=300)
        return r

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(STAGE, exist_ok=True)

    def _stage_save(reason="tick"):
        try:
            with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
                f.write(b.save_state())
            return True
        except Exception as e:
            L(f"!! STAGE SAVE FAILED [{reason}]: {e}")
            return False

    def _stage_continuity():
        try:
            camp.world.save(os.path.join(STAGE, "world_model.json"))
            camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
            if camp.soul is not None:
                camp.soul.save(os.path.join(STAGE, "soul.json"))
            with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
                json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception as e:
            L(f"!! stage continuity failed: {e}")

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
    camp._continuity_load = lambda *a, **k: None
    for loader, side, fallback in (
            (camp.world.load, os.path.join(sc_dir, sc_pref + ".world_model.json"), C.WORLD_JSON),
            (camp.strat.load, os.path.join(sc_dir, sc_pref + ".strat_memory.json"), C.STRAT_JSON)):
        try:
            loader(side if os.path.exists(side) else fallback)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            _soul = os.path.join(sc_dir, sc_pref + ".soul.json")
            camp.soul.load(_soul if os.path.exists(_soul) else os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    def species_levels():
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        out = []
        for s in range(min(cnt, 6)):
            sp = st.read_party_species(b, s)
            lv = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
            out.append((s, sp, lv))
        return out

    def bank(tag=""):
        _stage_save("bank")
        _stage_continuity()
        try:
            if os.path.isdir(BANK):
                shutil.rmtree(BANK)
            shutil.copytree(STAGE, BANK)
        except Exception as e:
            L(f"!! BANK copy failed: {e}")
        L(f"   BANKED{(' ' + tag) if tag else ''} -> {BANK}  levels={[(sp, lv) for _, sp, lv in species_levels()]}")

    # ── optional: walk to a grind spot first ──────────────────────────────────
    gmap = os.environ.get("GRIND_MAP", "3,41")            # Route 22 default (west of Viridian)
    gdir = os.environ.get("GRIND_DIR", "west")
    if gmap and gmap.lower() != "here":
        try:
            tgt = tuple(int(x) for x in gmap.split(","))
            here = tuple(tv.map_id(b))
            if here != tgt:
                L(f"   walking {here} -> {tgt} ({gdir}) to reach the grind grass")
                r = camp.walk_to_map(tgt, gdir)
                L(f"   walk_to_map -> {r}; now at {tuple(tv.map_id(b))} {tv.coords(b)}")
        except Exception as e:
            L(f"!! walk_to_map errored: {e} — grinding at current map")
    # Some grass routes are guardhouse-DIVIDED: the edge-crossing lands on the wrong side (no grass
    # reachable) and the grass sits past an interior WARP. GRIND_WARP="x,y" steps onto that warp first
    # (e.g. Route 15 warp (9,11)->(24,0) puts her on the grass side). Retries a few times (the warp is a
    # travel-to-coord that fires on arrival).
    gwarp = os.environ.get("GRIND_WARP", "")
    if gwarp:
        try:
            wx, wy = (int(x) for x in gwarp.split(","))
            for attempt in range(4):
                before = tuple(tv.coords(b) or (0, 0))
                camp.trav.travel(target_map=None, arrive_coord=(wx, wy), max_steps=60, max_seconds=60)
                # a warp fires on stepping onto (wx,wy); coords jump elsewhere when it triggers
                after = tuple(tv.coords(b) or (0, 0))
                L(f"   GRIND_WARP attempt {attempt}: {before} -> {after}")
                if after != before and after != (wx, wy):
                    break
        except Exception as e:
            L(f"!! GRIND_WARP errored: {e}")
        L(f"   after warp: now at {tuple(tv.map_id(b))} {tv.coords(b)}")

    # ── the fielded participation-XP grind ────────────────────────────────────
    battle_agent.PROTECT_LEAD_GRIND = True                # every grind battle: weak lead -> ace turn 1
    L(f"   PROTECT_LEAD_GRIND armed; GRIND_SWITCH_ENABLED={battle_agent.GRIND_SWITCH_ENABLED}")
    L(f"   start levels = {species_levels()}")
    stalled = set()                                       # species we've proven can't gain XP here

    def next_target():
        """The next mon to field, in PRIORITY order (GRIND_SPECIES): grind each species to TARGET
        before moving to the next, so Lapras (the essential answer) finishes first. Returns the
        (slot, species, level) of the highest-priority species still under TARGET and not stalled."""
        levels = species_levels()
        for want_sp in GRIND_SPECIES:
            for s, sp, lv in levels:
                if sp == want_sp and lv < TARGET and sp not in stalled:
                    return (s, sp, lv)
        return None

    passes = 0
    while time.time() - t0 < BUDGET_MIN * 60:
        tgt = next_target()
        if tgt is None:
            L(f"   ALL teammates at/over L{TARGET} (or stalled) — grind complete. {species_levels()}")
            break
        slot, sp, lv = tgt
        if slot != 0:
            L(f"   fielding slot {slot} (species {sp} L{lv}) as lead")
            camp._swap_party_slots(0, slot)
        lv_before = b.rd8(ram.GPLAYER_PARTY + 0x54)
        passes += 1
        L(f"   [pass {passes}] grinding species {sp} L{lv_before} -> {TARGET} (probe {PROBE_S}s)")
        try:
            r = camp.grind(TARGET, fragile=FRAGILE, budget_s=PROBE_S)
        except Exception as e:
            L(f"!! grind errored: {e}")
            r = "error"
        lv_after = b.rd8(ram.GPLAYER_PARTY + 0x54)
        L(f"   [pass {passes}] grind -> {r!r}; species {sp} L{lv_before} -> L{lv_after}")
        bank(f"pass{passes}")
        if r in ("no_safe_grass", "no_effective_move"):
            L(f"!! grind spot unusable ({r}) — stopping LOUD (relocate GRIND_MAP to a strong-wild spot)")
            break
        if r == "battle_loss":
            L("   battle_loss — healing and continuing")
            try:
                camp.heal_nearest()
            except Exception as e:
                L(f"   heal_nearest errored: {e}")
        if lv_after <= lv_before and r not in ("battle_loss",):
            # zero progress this pass on this species — mark it un-grindable HERE so we don't spin
            stalled.add(sp)
            L(f"   species {sp} made NO progress (L{lv_before}) — marking stalled on this map")

    camp._restore_ace()
    bank("final")
    L(f"DONE. battles={n_battles[0]} passes={passes} final levels={species_levels()}")


if __name__ == "__main__":
    main()
