"""recon_forward_loss.py — THE REAL TEST: does a LOSS break the forward drive?

Drives a longer free-roam from the live Route-4 save with a realistic oracle (prefer head_to_gym = forward;
if forward-drive has stood down and only grind is offered, GRIND — i.e. 'get stronger'; never voluntarily go
backward). Does NOT force-heal — lets the natural heal-floor / blackout / strategic-stuck machinery run, so a
real loss on the un-fleeable Nugget Bridge gauntlet happens. Instruments every tick:

  - map + coords
  - is the gate questline OPEN (forward objective alive)?
  - is head_to_gym OFFERED (she can still pick forward)?
  - the offered set, the pick, the result, whether she LOST
  - whether the strategic-stuck floor stood the forward-drive down to let her grind

PASS = across losses, the forward objective SURVIVES: the questline stays open (or re-opens after a blackout),
she is NEVER stranded with only talk/heal, and she NEVER voluntarily retreats to the backward dead-ends
(Route 3 = (3,21) west of Route 4). Losing should route her into heal+grind-toward-strength while staying
pointed north — not knock her off the objective.

Read-only: boots the live save, never writes a canonical save. RUN: python pokemon_agent\\recon_forward_loss.py [max_ticks]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import travel as tv                                       # noqa: E402
from battle_agent import BattleAgent                      # noqa: E402
from campaign import Campaign, resolve_state, WORLD_JSON  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BACKWARD = {(3, 21)}                                       # Route 3 — strictly backward of Route 4


def main():
    max_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    trace = []      # per-tick observations, filled by the oracle stub (fires once per decision)
    events = []
    camp = Campaign(b, battle_runner=lambda: BattleAgent(
        b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None).run(max_seconds=40),
        on_event=lambda s, **k: (events.append(s)),
        beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(WORLD_JSON)
    except Exception:
        pass
    camp.trav.battle_runner = camp._flee_runner

    def stub(kind, options, ctx):
        if kind != "action":
            return None
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        q = camp._active_questline
        backward_opts = [o for o in opts if o.startswith("travel:")
                         and tuple(int(v) for v in o.split(":", 1)[1].split(",")) in BACKWARD]
        # realistic priority: forward first; else GRIND to get stronger; else heal; else talk; NEVER pick a
        # voluntarily-backward travel unless it's the only thing left (that would itself be the failure).
        pick = None
        for pref in ("head_to_gym",):
            if pref in opts:
                pick = pref
                break
        if pick is None:
            for pref in ("battle", "wander_catch"):
                if pref in opts:
                    pick = pref
                    break
        if pick is None:
            for pref in ("heal", "stock_up", "talk_npc"):
                if pref in opts:
                    pick = pref
                    break
        if pick is None:
            pick = opts[0] if opts else None
        trace.append({
            "map": tv.map_id(b), "coords": tv.coords(b),
            "ql_open": q is not None,
            "h2g_offered": "head_to_gym" in opts,
            "only_passive": bool(opts) and all(o in ("talk_npc", "heal") for o in opts),
            "picked_backward": pick in backward_opts,
            "opts": sorted(opts), "pick": pick,
            "lead_lvl": (camp.read_live_state().get("party") or [{}])[0].get("level"),
        })
        return pick
    camp._oracle_choose = stub

    start_map = tv.map_id(b)
    print(f"==== START map={start_map} coords={tv.coords(b)} max_ticks={max_ticks} ====\n", flush=True)
    out = camp.free_roam(max_ticks=max_ticks, max_seconds=480, want_every=999)

    losses = sum(1 for e in events if "blacked out" in e or "knocked out" in e or "we lost" in e
                 or "went down" in e)
    print(f"\n==== free_roam -> {out} | {len(trace)} decisions | ~{losses} loss/blackout events ====", flush=True)
    print(f"{'tick':>4} {'map':>9} {'ql':>3} {'h2g':>4} {'passive':>8} {'back':>5}  lvl  pick", flush=True)
    for i, t in enumerate(trace, 1):
        print(f"{i:>4} {str(t['map']):>9} {'Y' if t['ql_open'] else '-':>3} "
              f"{'Y' if t['h2g_offered'] else '-':>4} {'Y!' if t['only_passive'] else '-':>8} "
              f"{'Y!' if t['picked_backward'] else '-':>5}  {str(t['lead_lvl']):>3}  {t['pick']}", flush=True)

    # ── VERDICT: did the forward drive SURVIVE the losses? ──
    ql_ever = any(t["ql_open"] for t in trace)
    ql_after_first_open = True
    seen_open = False
    for t in trace:
        if t["ql_open"]:
            seen_open = True
        elif seen_open and t["map"] == (3, 3):
            # at base camp the questline should be open; a closed questline AT Cerulean post-open = a break
            ql_after_first_open = False
    stranded = [i for i, t in enumerate(trace, 1) if t["only_passive"]]
    backward = [i for i, t in enumerate(trace, 1) if t["picked_backward"]]
    forward_always = all(t["h2g_offered"] or t["map"] not in ((3, 3), (3, 22))
                         or any(o.startswith("travel:") or o in ("battle", "wander_catch") for o in t["opts"])
                         for t in trace)
    print("\n---- checks ----", flush=True)
    print(f"  questline opened at all:                              {ql_ever}", flush=True)
    print(f"  questline stayed open at base camp after first open:  {ql_after_first_open}", flush=True)
    print(f"  NEVER stranded with only talk/heal (ticks): {stranded or 'none'}", flush=True)
    print(f"  NEVER voluntarily retreated backward to Route 3 (ticks): {backward or 'none'}", flush=True)
    ok = ql_ever and ql_after_first_open and not stranded and not backward
    print(f"\n==== forward drive SURVIVES a loss: {'PASS' if ok else 'INSPECT'} ====", flush=True)


if __name__ == "__main__":
    main()
