"""recon_victory_road_smoke.py — ISOLATED smoke test of victory_road.run_strike on the LIVE campaign bridge.

Proves the extraction (recon_victory.py -> victory_road.VictoryRoad) is byte-faithful: boots a Campaign like
recon_victory.py, then hands control to run_strike(camp, L). From a badge-8 fixture (giovanni_kit_g = at
Viridian, badges 8, leveled team) the resume-safe vehicle drives Viridian -> Route 22 (Gary) -> the gate ->
Route 23 -> the Victory-Road boulder floors -> Route 23 north -> the Indigo Plateau. Success = returns
'reached_indigo' and she is on the Indigo Plateau exterior (3,9).

RUN: VICTORY_ROAD_STATE=giovanni_kit_g ../.venv/Scripts/python.exe -u recon_victory_road_smoke.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.environ.get("WATCH") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
import victory_road as vr            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "victory_road_smoke")
INDIGO = (3, 9)


def _resolve_state(name):
    if not name:
        return os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign"
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            base = os.path.basename(cand)
            pref = base[:-6] if base.endswith(".state") else base
            return cand, d, pref
    return name, os.path.dirname(name) or CANON, "kira_campaign"


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("VICTORY_ROAD_STATE", "giovanni_kit_g"))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    nb = [0]

    def fight():
        nb[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=420)

    camp = Campaign(b, battle_runner=fight, on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    for loader, side, fb in ((camp.world.load, sc_pref + ".world_model.json", C.WORLD_JSON),
                             (camp.strat.load, sc_pref + ".strat_memory.json", C.STRAT_JSON)):
        try:
            p = os.path.join(sc_dir, side)
            loader(p if os.path.exists(p) else fb)
        except Exception:
            pass
    L(f"boot state = {state_path} map={tv.map_id(b)} coords={tv.coords(b)} money=${camp.money()} "
      f"badge8={int(fm.read_flag(b, vr.FLAG_BADGE_EARTH))}")
    res = vr.run_strike(camp, L, dbg_dir=DBG)
    at_indigo = tuple(tv.map_id(b)) == INDIGO
    L(f"RESULT: {res} | final pos {tv.map_id(b)}@{tv.coords(b)} | at_indigo={at_indigo} | battles {nb[0]}")
    ok = res == "reached_indigo" and at_indigo
    L(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
