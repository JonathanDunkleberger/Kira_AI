"""recon_e4_strike_smoke.py — ISOLATED smoke test of e4_strike.run_strike on the LIVE campaign bridge.

Proves the extraction (recon_e4.py -> e4_strike.EliteFour) is byte-faithful + reaches (as far as team-depth
allows) the Hall of Fame: boots a Campaign like recon_e4.py at a badge-8 at-Indigo fixture, then hands
control to run_strike(camp, L). The vehicle drives Indigo exterior -> League center (heal + mart stock-up)
-> the 5-room gauntlet (Lorelei..Champion) -> the Hall of Fame == CREDITS. Success = returns 'credits'.
A 'stuck' with rooms>0 still proves the dispatch/nav/shop/room machinery (the wall is then team-depth).

RUN: E4_STRIKE_STATE=indigo_reach_g ../.venv/Scripts/python.exe -u recon_e4_strike_smoke.py
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
import e4_strike as e4               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "e4_strike_smoke")


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

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("E4_STRIKE_STATE", "indigo_reach_g"))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    nb = [0]

    # take the offered heal/cure/revive in every E4 fight (a competent player does — recon_e4's _choose)
    def _choose(ptype, offers, ctx):
        for k in ("use_potion", "use_cure", "use_ether", "use_revive"):
            if k in offers:
                return k
        return "keep_fighting"

    def fight():
        nb[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True), choose=_choose).run(max_seconds=600)

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
      f"badge8={int(fm.read_flag(b, e4.FLAG_BADGE_EARTH))} FR x{camp.bag_count(e4.FULL_RESTORE)}")
    # bound a smoke so a team-depth wall doesn't run the full 90-min deadline
    os.environ.setdefault("POKEMON_E4_DEADLINE_S", os.environ.get("SMOKE_DEADLINE_S", "1800"))
    res = e4.run_strike(camp, L, dbg_dir=DBG)
    L(f"RESULT: {res} | final pos {tv.map_id(b)}@{tv.coords(b)} | battles {nb[0]}")
    ok = res == "credits"
    L(f"SMOKE {'PASS (CREDITS)' if ok else 'RESULT ' + str(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
