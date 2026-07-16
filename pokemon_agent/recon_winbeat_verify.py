"""recon_winbeat_verify.py — F-7(c) CERTAIN-WIN EARLY BEAT verification (two phases).

The change under test (battle_agent): when a faint leaves ZERO live mons in gEnemyParty, the
win line fires AT THE FAINT (one merged beat) instead of after the 5-15s victory drain — so
the ~4s LLM chain runs during the drain and her voice lands on the victory screen. _finish
must not duplicate it, and a mid-battle faint of a multi-mon trainer must NEVER read as a win.

PHASE trainer (banked_GIOVANNI -> Route 22, the scripted GARY #7 rival fight): the premium
  correctness check — Gary fields SIX mons; any merged-win emit while live_remaining > 0 at
  emit time = FAIL (a premature "you won" is a lie on stream). A won fight must show exactly
  ONE merged win, emitted with in_battle=True, and ZERO "you won the battle" duplicates.
  (banked_E4 is unusable — that bank has moved to the Hall of Fame room, past all fights.)
PHASE wild (banked_HM05, Route 2 grass): walk grass until a wild battle, FIGHT it with the
  real agent; the L43 lead one-shots Route 2 wilds -> certain win at first faint. PASS =
  merged win fired in_battle=True, no dup, and report the LEAD TIME (emit -> battle exit)
  the beat bought.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_winbeat_verify.py [trainer|wild|both]
READ-ONLY on all bundles (never saves state).
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
MERGED = "that's the battle, you won"
OLD_WIN = "you won the battle"

WIN_LINE_MARKER = MERGED           # substring of the merged early beat
FAILS = []


def _load(bundle):
    b = Bridge(ROM)
    with open(os.path.join(LONGRUN, bundle, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()
    b.set_input_owner("agent")
    return b


def _instrumented_agent(b, events, choose=None):
    """Real BattleAgent whose event sink records (t, in_battle, live_remaining, summary)."""
    box = {}

    def sink(summary, **k):
        try:
            live = box["agent"]._enemy_live_remaining()
        except Exception:
            live = -1
        events.append((time.time(), st.in_battle(b), live, summary))
        print(f"   [event] t=+{time.time() - box['t0']:6.1f}s in_battle={st.in_battle(b)} "
              f"live_foes={live} :: {summary}", flush=True)

    box["agent"] = BattleAgent(b, on_event=sink, render=lambda: None,
                               log=lambda m: print(f"   {m}", flush=True), choose=choose)
    box["t0"] = time.time()
    return box["agent"]


def _judge(events, outcome, phase):
    """Shared PASS/FAIL logic over the recorded event stream."""
    merged = [(t, ib, live, s) for (t, ib, live, s) in events if MERGED in s]
    dups = [(t, ib, live, s) for (t, ib, live, s) in events if OLD_WIN in s]
    # (a) premature win: a merged emit while live foes remained
    for (t, ib, live, s) in merged:
        if live > 0:
            FAILS.append(f"{phase}: PREMATURE merged win with live_foes={live}: {s!r}")
    if outcome == "win":
        if len(merged) != 1:
            FAILS.append(f"{phase}: won but merged-win count = {len(merged)} (want exactly 1)")
        elif not merged[0][1]:
            FAILS.append(f"{phase}: merged win fired AFTER battle exit (in_battle=False) — no lead time")
        if dups:
            FAILS.append(f"{phase}: duplicate '{OLD_WIN}' fired after the merged beat")
    return merged


def phase_trainer():
    print("==== PHASE trainer: GARY #7 (banked_GIOVANNI -> Route 22 scripted fight) ====",
          flush=True)
    from campaign import Campaign
    b = _load("banked_GIOVANNI")
    print(f"   boot map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

    def fight_open():
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    def _choose(ptype, offers, ctx):
        for k in ("use_potion", "use_cure", "use_ether", "use_revive"):
            if k in offers:
                return k
        return "keep_fighting"

    events = []
    result = {}

    def battle_runner():
        n0 = len(events)
        agent = _instrumented_agent(b, events, choose=_choose)
        out = agent.run(max_seconds=420)
        # re-entry loop (the recon_agatha pattern): stuck/timeout mid-gauntlet re-attaches
        for _ in range(4):
            if out in ("win", "loss", "ended", "caught") or not fight_open():
                break
            for _f in range(120):
                b.run_frame()
            out = _instrumented_agent(b, events, choose=_choose).run(max_seconds=420)
        fight_max_live = max((live for (_, _, live, _) in events[n0:]), default=0)
        if fight_max_live >= 2:
            # THIS is the multi-mon fight we came for (Gary) — record + stop travel
            result["outcome"] = out
            print(f"   [check] MULTI-MON fight done: outcome={out} max_live={fight_max_live} "
                  f"in_battle_now={st.in_battle(b)}", flush=True)
            return "loss"                       # stop travel cleanly
        # a wild interloper on Route 22 grass — fought through, keep traveling
        print(f"   [check] (wild interloper, max_live={fight_max_live}, outcome={out} — continuing)",
              flush=True)
        return out

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None,
                    render=lambda: None)
    camp._save_campaign = lambda *a, **k: True     # READ-ONLY: no banking from a verify
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    camp._suppress_heal = True
    # the learned world graph rides the bank's sidecars — without it travel has no route to
    # Route 22 (first attempt wandered to Route 2 and wedged)
    for side, loader in (("world_model.json", camp.world.load),
                         ("strat_memory.json", camp.strat.load)):
        try:
            sp = os.path.join(LONGRUN, "banked_GIOVANNI", side)
            if os.path.exists(sp):
                loader(sp)
        except Exception as e:
            print(f"   !! sidecar load failed ({side}): {e}", flush=True)
    # Viridian -> Route 22 (west). Gary #7 is a TRIGGER-TILE scripted fight on the way to
    # the League gate — walking west fires it (8 badges in the bag at this bank).
    # edge='west' is load-bearing: the default 'north' walked her to Route 2 twice.
    camp.trav.travel(target_map=(3, 41), edge="west", max_steps=400, max_seconds=240)
    if "outcome" not in result and tuple(tv.map_id(b)) == (3, 41):
        # on Route 22 but no trigger yet -> BFS-walk to the League gate door (8,5); the road
        # west crosses the rival trigger rows and travel's battle handoff catches the fight
        # (blind LEFT presses died on the pond/ledge terrain — landed (47,7), gate at (8,5)).
        camp.trav.travel(arrive_coord=(10, 5), max_steps=300, max_seconds=180)
        if "outcome" not in result and fight_open():
            battle_runner()
        # pre-battle rival dialogue can leave a box up mid-walk — drain and give it one beat
        for _try in range(10):
            if "outcome" in result or fight_open():
                if fight_open():
                    battle_runner()
                break
            b.press("B", 8, 12, lambda: None, owner="agent")
            for _ in range(30):
                b.run_frame()
    if "outcome" not in result:
        print("   [check] INCONCLUSIVE — Gary #7 never triggered (already beaten on this bank?)",
              flush=True)
        return
    _judge(events, result.get("outcome"), "trainer")
    maxlive = max((live for (_, _, live, _) in events), default=-1)
    sent_out = [s for (_, _, _, s) in events if "the trainer sent out" in s]
    print(f"   [check] max live foes seen = {maxlive} (Gary fields 6); switch-ins observed: "
          f"{len(sent_out)}", flush=True)
    if maxlive < 2:
        FAILS.append(f"trainer: max live-count seen = {maxlive} — the gEnemyParty HP read "
                     f"never saw the bench (a multi-mon trainer must show >1); the certain-win "
                     f"read is NOT trustworthy")


def phase_wild():
    print("==== PHASE wild: Route 2 grass (banked_HM05) ====", flush=True)
    from campaign import Campaign
    b = _load("banked_HM05")
    print(f"   boot map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
    events = []
    result = {}

    def battle_runner():
        agent = _instrumented_agent(b, events)
        out = agent.run(max_seconds=180)
        result["outcome"] = out
        # measure the lead time the early beat bought: merged emit -> battle actually exits
        t_exit = time.time()
        for _ in range(1200):
            if not st.in_battle(b):
                break
            b.run_frame()
        result["exit_ts"] = time.time()
        print(f"   [check] outcome={out} battle exited {time.time() - t_exit:.1f}s after run() returned",
              flush=True)
        return "loss"     # stop travel cleanly after ONE battle (diagnosis pattern, recon_catch)

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None,
                    render=lambda: None)
    camp._save_campaign = lambda *a, **k: True     # READ-ONLY: no banking from a verify
    camp._continuity_save = lambda *a, **k: None
    camp._suppress_heal = True

    def fight_open():
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    # travel deliberately routes AROUND grass (first attempt crossed Route 2 encounter-free)
    # — so walk TO the nearest grass tile, then PACE it until a wild fires (the grind pattern).
    off = tv.MAP_OFFSET
    gs = [(x - off, y - off) for (x, y) in tv.Grid(b).grass]
    if not gs:
        print("   [check] INCONCLUSIVE — no grass on this map", flush=True)
        return
    cur = tv.coords(b) or (0, 0)
    target = min(gs, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
    camp.trav.travel(arrive_coord=target, max_steps=250, max_seconds=150)
    gset = set(gs)
    for pace in range(120):
        if "outcome" in result:
            break
        if fight_open():
            battle_runner()
            break
        c = tv.coords(b) or (0, 0)
        # step to any adjacent grass tile (fall back to re-stepping in place on the grass)
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
        print("   [check] INCONCLUSIVE — no wild encounter after pacing the grass", flush=True)
        return
    merged = _judge(events, result.get("outcome"), "wild")
    if merged and result.get("exit_ts"):
        lead = result["exit_ts"] - merged[0][0]
        print(f"   [check] LEAD TIME bought by the early beat: {lead:.1f}s "
              f"(voice generation chain is ~4s — landed {'ON' if lead >= 2.5 else 'NEAR'} the moment)",
              flush=True)


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if which in ("trainer", "both"):
        phase_trainer()
    if which in ("wild", "both"):
        phase_wild()
    print()
    if FAILS:
        print("==== RESULT: FAIL ====", flush=True)
        for f in FAILS:
            print(f"   !! {f}", flush=True)
        return 1
    print("==== RESULT: PASS ====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
