"""recon_descent_grade.py — F-11 THE DESCENT: headless PRE-GRADE sweep over the banked arc spawns.

The showcase gate (NEXT_SESSION F-11) re-runs every major arc SOUL-ON and grades it against the
humanity doctrine. This is the machine-gradable HALF that needs no bot: off every banked arc
bundle, run the REAL free_roam loop (faithful chooser, same pattern as recon_longrun) for a
bounded window and grade the wedge classes a viewer would feel:

  LOCOMOTION  — F-1 nav-tripwire fires, silent no-moves, watchdog trips (dithering/wedges)
  LIVENESS    — void-core trips, abandons, crashes (the show can never die on screen)
  GROUNDING   — F-8: any tick standing on a map with NO name (her ctx would read "an
                unfamiliar area" on ground the rope already crossed) → a concrete fix payload
  TEXTURE     — decision variety + battles fought (a stretch where she only idles is a red flag)

Grades: PASS / WARN / FAIL per arc + a ranked riskiest-arcs list → DESCENT_PREGRADE.md.
This ranking picks Jonny's spot-watch arcs (his notes stay the final gate — this pass only
catches what a machine can). Canonical + banked bundles are READ-ONLY here: all campaign
persistence is no-op'd (grading, not banking).

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_descent_grade.py [seconds_per_arc]
ENV:  DESCENT_ARCS="banked_E4,banked_SILPH"  (subset override)
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
# GRADING BUDGETS: one grind/battle action must not eat the whole arc window (first sweep
# lesson: the HM05 arc spent 20+ min inside ONE 'battle' tick — 78 wild encounters — because
# the weak-grind budget default is sized for real climbing, not grading).
os.environ["POKEMON_GRIND_WEAK_BUDGET_S"] = "60"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
import firered_ram as ram                                  # noqa: E402
import pokemon_state as st                                 # noqa: E402
import travel as tv                                        # noqa: E402
from battle_agent import BattleAgent                       # noqa: E402
import campaign as C                                       # noqa: E402
from campaign import Campaign                              # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
REPORT = os.path.join(_HERE, "DESCENT_PREGRADE.md")
MAX_DECISIONS = int(os.getenv("DESCENT_MAX_DECISIONS", "30"))

# Rope order (early -> late). banked_CREDITS deliberately EXCLUDED — the known mid-ceremony
# grenade (resuming it rolls credits -> title); 'postgame' is its safe replacement.
ARCS = ["banked_HM05", "banked_ROCKTUNNEL", "banked_SCOPE", "banked_FLUTE", "banked_SNORLAX",
        "banked_SAFARI", "banked_SURF_TAUGHT", "banked_SILPH", "banked_SABRINA",
        "banked_CINNABAR", "banked_BLAINE", "banked_GIOVANNI", "banked_VICTORY",
        "banked_E4", "banked_POSTGAME"]


class _Done(Exception):
    def __init__(self, why):
        self.why = why


def grade_arc(name, seconds):
    p = os.path.join(LONGRUN, name)
    statef = os.path.join(p, "kira_campaign.state")
    if not os.path.exists(statef):
        return {"arc": name, "grade": "SKIP", "why": "no bundle"}
    b = Bridge(ROM)
    with open(statef, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    events = []
    battles = [0]

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: None, choose=chooser).run(max_seconds=90)
        battles[0] += 1
        return out

    camp = Campaign(b, battle_runner=runner,
                    on_event=lambda s, **k: events.append((k.get("kind", ""), s)),
                    beat=lambda *a, **k: None, render=lambda: None)
    # READ-ONLY GRADE: no banking, no canonical writes, no sidecar writes.
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    for side, loader in (("world_model.json", camp.world.load),
                         ("strat_memory.json", camp.strat.load)):
        try:
            sp = os.path.join(p, side)
            if os.path.exists(sp):
                loader(sp)
        except Exception:
            pass
    try:
        sp = os.path.join(p, "soul.json")
        if camp.soul is not None and os.path.exists(sp):
            camp.soul.load(sp)
    except Exception:
        pass

    decisions = [0]
    unnamed = set()          # maps she stood on with NO place name (F-8 gap payload)
    picks = {}
    nomove = [0]
    t0 = [time.time()]       # arc clock (chooser hard-wall reads it via closure)

    def chooser(kind, options, ctx):
        # hard wall FIRST, on ANY consult (a wander_catch's stream of catch_judgments never
        # reaches the tick-top window check — first sweep ran one action for 7+ minutes)
        if time.time() - t0[0] > seconds * 2.5:
            raise _Done("hard wall (arc overran window x2.5)")
        if kind == "battle_item":
            if isinstance(options, dict) and options:
                return "use_potion" if "use_potion" in options else next(iter(options))
            return None
        if kind != "action":
            return None
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        decisions[0] += 1
        mp = tv.map_id(b)
        if tuple(mp) not in camp._PLACE_NAMES:
            unnamed.add(tuple(mp))
        if decisions[0] > MAX_DECISIONS:
            raise _Done("decision budget")
        party = [(st.read_party_species(b, s),
                  b.rd16(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x56),
                  b.rd16(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x58))
                 for s in range(min(b.rd8(ram.GPLAYER_PARTY_CNT), 6))]
        down = sum(1 for _, hp, _ in party if hp == 0)
        lhp, lmx = (party[0][1], party[0][2]) if party else (1, 1)
        critical = (lmx and lhp / lmx < 0.25) or down >= max(1, len(party) // 2)
        pick = None
        if len(opts) == 1:
            pick = opts[0]
        else:
            for pref in ((("heal",) if critical else ())
                         + ("stock_up", "head_to_gym", "wander_catch", "battle")):
                if pref in opts:
                    pick = pref
                    break
            if pick is None:
                pick = next((o for o in opts if o.startswith("travel:")),
                            "talk_npc" if "talk_npc" in opts else (opts[0] if opts else None))
        picks[pick] = picks.get(pick, 0) + 1
        return pick

    camp._oracle_choose = chooser
    why = "window done"
    crash = ""
    t0[0] = time.time()
    try:
        camp.free_roam(max_ticks=100000, max_seconds=seconds, want_every=999)
    except _Done as d:
        why = d.why
    except Exception as e:
        import traceback
        crash = f"{e!r}"
        why = "CRASH"
        print(traceback.format_exc())
    wall = round(time.time() - t0[0], 1)

    voids = sum(1 for k, s in events if k == "recover" and "went dark" in s)
    abandons = sum(1 for k, _ in events if k == "abandoned")
    trips = getattr(camp, "_watchdog_trips", 0) or 0
    navfires = getattr(camp, "_nav_tripwire_total", 0) or 0
    # TRAVEL-WEDGE COUNT (shift 5 grader gap): banked_SCOPE bonked a wall 412 times in one
    # window and graded PASS — only roam-level tripwires were counted. A viewer FEELS every
    # travel wedge (she paces at a wall); a handful per window is normal probing, a storm
    # is a locomotion defect.
    twedges = getattr(getattr(camp, "trav", None), "wedge_total", 0) or 0
    nomove[0] = getattr(camp, "_nomove_streak", 0) or 0
    res = {"arc": name, "why": why, "wall_s": wall, "decisions": decisions[0],
           "battles": battles[0], "picks": picks, "watchdog_trips": trips,
           "nav_tripwire": navfires, "travel_wedges": twedges, "voids": voids,
           "abandons": abandons, "unnamed_maps": sorted(unnamed),
           "end_nomove_streak": nomove[0], "crash": crash, "events": len(events)}
    # ── the grade ──
    if crash or abandons or voids:
        res["grade"] = "FAIL"
    elif trips >= 2 or (decisions[0] == 0 and battles[0] == 0) or twedges > 100:
        res["grade"] = "FAIL"
    elif unnamed or trips == 1 or navfires > 2 or nomove[0] >= 2 or twedges > 20:
        res["grade"] = "WARN"
    else:
        res["grade"] = "PASS"
    return res


def main():
    # SINGLE-RUN LAW (same class as recon_longrun's reap; the Bash background launcher has
    # been observed spawning the command TWICE — a twin sweep skews wall-clock grades).
    import subprocess
    pidf = os.path.join(LONGRUN, "descent_grade.pid")
    try:
        if os.path.exists(pidf):
            old = int(open(pidf).read().strip() or 0)
            if old and old != os.getpid():
                chk = subprocess.run(["tasklist", "/FI", f"PID eq {old}"],
                                     capture_output=True, text=True).stdout
                if str(old) in chk and "python" in chk.lower():
                    subprocess.run(["taskkill", "/F", "/PID", str(old)], capture_output=True)
                    print(f"[reap] killed predecessor sweep PID {old}", flush=True)
        os.makedirs(LONGRUN, exist_ok=True)
        with open(pidf, "w") as pf:
            pf.write(str(os.getpid()))
    except Exception as _re:
        print(f"[reap] predecessor check failed: {_re}", flush=True)
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    arcs = [a for a in os.getenv("DESCENT_ARCS", "").split(",") if a] or ARCS
    results = []
    for name in arcs:
        print(f"\n==== GRADING {name} ({seconds:.0f}s window) ====", flush=True)
        try:
            r = grade_arc(name, seconds)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"arc": name, "grade": "FAIL", "why": f"harness crash: {e!r}"}
        results.append(r)
        print(f"==== {name}: {r['grade']} ({r.get('why','')}) decisions={r.get('decisions')} "
              f"battles={r.get('battles')} wd={r.get('watchdog_trips')} nav={r.get('nav_tripwire')} "
              f"twedge={r.get('travel_wedges')} voids={r.get('voids')} "
              f"unnamed={r.get('unnamed_maps')}", flush=True)

    order = {"FAIL": 0, "WARN": 1, "PASS": 2, "SKIP": 3}
    ranked = sorted((r for r in results if r["grade"] in ("FAIL", "WARN")),
                    key=lambda r: order[r["grade"]])
    lines = ["# DESCENT PRE-GRADE (machine half of F-11) — " + time.strftime("%Y-%m-%d %H:%M"),
             "",
             "Headless real-loop grade per banked arc spawn. PASS = no machine-visible wedge "
             "class fired. This does NOT replace Jonny's spot-watches — it picks them.",
             "",
             "| arc | grade | why ended | decisions | battles | watchdog | nav-tripwire "
             "| travel-wedges | voids | unnamed maps |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['arc']} | **{r['grade']}** | {r.get('why','')} | {r.get('decisions','-')} "
                     f"| {r.get('battles','-')} | {r.get('watchdog_trips','-')} | {r.get('nav_tripwire','-')} "
                     f"| {r.get('travel_wedges','-')} | {r.get('voids','-')} | {r.get('unnamed_maps') or ''} |")
    lines += ["", "## Riskiest arcs (spot-watch these first)", ""]
    lines += [f"- **{r['arc']}** — {r['grade']}: wd={r.get('watchdog_trips')} nav={r.get('nav_tripwire')} "
              f"twedge={r.get('travel_wedges')} voids={r.get('voids')} crash={r.get('crash','')} "
              f"unnamed={r.get('unnamed_maps')}"
              for r in ranked] or ["- none — all graded arcs PASS"]
    all_unnamed = sorted({m for r in results for m in (r.get("unnamed_maps") or [])})
    lines += ["", f"## F-8 name-gap payload (maps crossed with no _PLACE_NAMES entry): {all_unnamed or 'NONE'}"]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(LONGRUN, "descent_grade.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n==== REPORT -> {REPORT} ====", flush=True)


if __name__ == "__main__":
    main()
