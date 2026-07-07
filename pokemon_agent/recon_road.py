"""recon_road.py — THE ROAD STRIKE: deterministic world-graph walk to a target map.

fuchsia_road1 truth (night shift #1): with badge 6 banked, head_to_gym targets BLAINE on
CINNABAR — an island the graph can't route to without Surf — so the longrun stalls in
Saffron. The road to the SURF unlock (Fuchsia: Safari Zone) is fully learned (shift 7
walked it northbound); this strike just walks it: loop world.next_step(cur, TARGET) and
execute each hop with campaign's own executors (warp hop = travel-to-tile + enter_warp;
edge hop = _edge_travel with the pass-through fallback). Battles run the real agent; heal
bounces to the nearest Center. Banks on arrival for promote_bank.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_road.py [grp,num] [label]
     default target = 3,7 (Fuchsia City), label = fuchsia_south
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
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")


def main():
    tgt = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "3,7").split(","))
    label = sys.argv[2] if len(sys.argv) > 2 else "fuchsia_south"
    stage = os.path.join(SCRATCH, f"stage_road_{label}")
    bank = os.path.join(SCRATCH, f"banked_ROAD_{label}")
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    n_battles = [0]

    def fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(stage, exist_ok=True)

    def _stage_save(reason="tick"):
        try:
            with open(os.path.join(stage, "kira_campaign.state"), "wb") as f:
                f.write(b.save_state())
            return True
        except Exception as e:
            L(f"!! STAGE SAVE FAILED [{reason}]: {e}")
            return False

    def _stage_continuity():
        try:
            camp.world.save(os.path.join(stage, "world_model.json"))
            camp.strat.save(os.path.join(stage, "strat_memory.json"))
            if camp.soul is not None:
                camp.soul.save(os.path.join(stage, "soul.json"))
            with open(os.path.join(stage, "journey_core.json"), "w", encoding="utf-8") as jf:
                json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception as e:
            L(f"!! stage continuity failed: {e}")

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
    camp._continuity_load = lambda *a, **k: None
    for loader, path in ((camp.world.load, C.WORLD_JSON), (camp.strat.load, C.STRAT_JSON)):
        try:
            loader(path)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def drain(max_a=30):
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} -> target {tgt} lead={lead_frac():.0%}")
    deadline = time.time() + 1500
    wedges = {}
    while time.time() < deadline:
        if st.in_battle(b):
            L(f"   battle -> {camp.battle_runner()}")
            drain()
            continue
        if dd_box(b):
            drain()
            continue
        here = tuple(tv.map_id(b))
        if here == tgt:
            break
        if lead_frac() < 0.4:
            L(f"   lead {lead_frac():.0%} — healing")
            camp.heal_nearest()
            continue
        step = None
        try:
            step = camp.world.next_step(here, tgt)
        except Exception as e:
            L(f"   next_step error: {e}")
        if step is None:
            L(f"!! no graph route {here} -> {tgt} — abort LOUD")
            return 1
        nxt_map, kind, detail = step
        key = (here, kind, str(detail))
        before = tuple(tv.map_id(b))
        if kind == "warp":
            camp.trav.travel(target_map=None, arrive_coord=tuple(detail), max_steps=300,
                             max_seconds=120,
                             avoid={tuple(w[0]) for w in tv.read_warps(b)} - {tuple(detail)})
            if st.in_battle(b):
                L(f"   battle en route -> {camp.battle_runner()}")
                drain()
                continue
            if tuple(tv.map_id(b)) == before:
                camp.enter_warp(pick=tuple(detail))
            r = "warped" if tuple(tv.map_id(b)) != before else "warp_failed"
        else:
            r = camp._edge_travel(nxt_map, detail, budget_s=180)
            if st.in_battle(b):
                L(f"   battle en route -> {camp.battle_runner()}")
                drain()
                continue
            if r == "need_heal":
                camp.heal_nearest()
                continue
        moved = tuple(tv.map_id(b)) != before
        L(f"   hop {before} -> {nxt_map} [{kind} {detail}] = {r} "
          f"(now {tv.map_id(b)}@{tv.coords(b)})")
        if moved:
            wedges.pop(key, None)
            _stage_save("hop")
        else:
            wedges[key] = wedges.get(key, 0) + 1
            if wedges[key] >= 4:
                L(f"!! hop {key} wedged x4 — abort LOUD")
                return 1
            drain()

    if tuple(tv.map_id(b)) != tgt:
        L(f"!! target {tgt} not reached (at {tv.map_id(b)}) — NOT banking")
        return 1
    camp.heal_nearest()
    L(f"   ARRIVED {tgt} @ {tv.coords(b)} | battles {n_battles[0]} | lead {lead_frac():.0%}")
    _stage_save("arrived")
    _stage_continuity()
    if os.path.isdir(bank):
        shutil.rmtree(bank, ignore_errors=True)
    shutil.copytree(stage, bank)
    L(f"BANKED -> {bank}")
    L(f"promote: python pokemon_agent/promote_bank.py {bank} {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
