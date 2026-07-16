"""recon_spinmaze_verify.py — VIRIDIAN GYM SPIN-MAZE locomotion verify (NEXT_SESSION 9a).

The capability under test: the REAL travel loop's spinner-floor handling — Grid.spin
classification + the travel-wedge -> spin_assist hand-off (travel.py ~1185) ->
campaign._spin_assist -> spin_nav.SpinNav — un-verified LIVE on this floor since 2eb2b05
(badge 8 was won via the strike harness, not the general loop).

HONEST SETUP (per NEXT_SESSION): banked_GIOVANNI/stage_giovanni are POST-badge banks but the
spin TILES persist post-badge, so we spawn banked_POSTGAME (Champion at home, Pallet) and
WALK IN with the general machinery: exit house -> Route 1 -> Viridian -> gym door (36,10)
-> map (5,1) -> cross the maze to Giovanni's platform (2,2/2,3 — he's hidden post-badge,
FLAG_HIDE_VIRIDIAN_GIOVANNI, the tile is open floor per pret ViridianGym.json).

PASS  = she stands within 1 tile of (2,3) inside the gym, having crossed the spinner field
        with the real loop (assist fired, or BFS + glides got there without a wedge).
FAIL  = never arrives (wedge storm / abandon) or gym floor reads 0 spin tiles (classifier
        or warp landed wrong).
INCONCLUSIVE = the approach (house exit / Pallet->Viridian / door) fails before the maze —
        that's a different (already-graded) capability, reported as such.

READ-ONLY on the bundle: no banking, no sidecar writes. SINGLE-RUN LAW applies.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_spinmaze_verify.py [bundle]
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
import pokemon_state as st             # noqa: E402
import travel as tv                    # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")

MAP_VIRIDIAN_GYM = (5, 1)
GYM_DOOR = (36, 10)                    # Viridian City -> gym (pret ViridianCity.json warp 2)
MAZE_TARGET = (2, 3)                   # tile below Giovanni's old stand (2,2) — far side of the maze


def _load_campaign(bundle):
    p = os.path.join(LONGRUN, bundle)
    with open(os.path.join(p, "kira_campaign.state"), "rb") as f:
        state = f.read()
    b = Bridge(ROM)
    b.load_state(state)
    for _ in range(240):
        b.run_frame()
    b.set_input_owner("agent")

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(f"   {m}", flush=True)).run(max_seconds=120)

    camp = Campaign(b, battle_runner=runner,
                    on_event=lambda s, **k: print(f"   [event] {s}", flush=True),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True      # READ-ONLY: never bank from a verify
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    camp._suppress_heal = True
    for side, loader in (("world_model.json", camp.world.load),
                         ("strat_memory.json", camp.strat.load)):
        try:
            sp = os.path.join(p, side)
            if os.path.exists(sp):
                loader(sp)
        except Exception:
            pass
    return b, camp


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "banked_POSTGAME"
    print(f"==== VIRIDIAN SPIN-MAZE verify (bundle={bundle}) ====", flush=True)
    b, camp = _load_campaign(bundle)
    print(f"   boot map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

    # tee travel's log so the assist hand-off is machine-checkable
    trav_lines = []
    _orig_log = camp.trav.log

    def tee(m):
        trav_lines.append(m)
        _orig_log(m)
    camp.trav.log = tee

    # ---- APPROACH (covered ground — failures here are INCONCLUSIVE, not the maze) ----
    if tuple(tv.map_id(b))[0] != 3:
        print("   indoors — exiting to overworld", flush=True)
        camp._exit_to_overworld()
        if tuple(tv.map_id(b))[0] != 3:
            print("RESULT: INCONCLUSIVE — building exit failed (approach, not maze)", flush=True)
            sys.exit(2)

    t0 = time.time()
    hops = {tv.MAP_PALLET: tv.MAP_ROUTE1, tv.MAP_ROUTE1: tv.MAP_VIRIDIAN}
    for _ in range(8):
        cur = tuple(tv.map_id(b))
        if cur == tv.MAP_VIRIDIAN:
            break
        nxt = hops.get(cur)
        if nxt is None:
            print(f"RESULT: INCONCLUSIVE — off the Pallet->Viridian rope at {cur}", flush=True)
            sys.exit(2)
        print(f"   leg: {cur} -> {nxt} (north edge)", flush=True)
        camp.trav.travel(target_map=nxt, max_steps=600, max_seconds=240, edge="north")
        if time.time() - t0 > 900:
            print("RESULT: INCONCLUSIVE — approach overran 15 min", flush=True)
            sys.exit(2)
    if tuple(tv.map_id(b)) != tv.MAP_VIRIDIAN:
        print(f"RESULT: INCONCLUSIVE — never reached Viridian (at {tv.map_id(b)})", flush=True)
        sys.exit(2)
    print(f"   in Viridian at {tv.coords(b)} — heading to gym door {GYM_DOOR}", flush=True)

    camp.trav.travel(target_map=None, arrive_coord=GYM_DOOR, max_steps=300, max_seconds=180)
    if tuple(tv.map_id(b)) == tv.MAP_VIRIDIAN:
        camp.enter_warp(pick=GYM_DOOR)
    if tuple(tv.map_id(b)) != MAP_VIRIDIAN_GYM:
        print(f"RESULT: INCONCLUSIVE — gym door didn't take (at {tv.map_id(b)} "
              f"{tv.coords(b)})", flush=True)
        sys.exit(2)

    # ---- THE TEST: cross the spinner field with the real loop ----
    grid = tv.Grid(b)
    off = tv.MAP_OFFSET
    spin_ct = len(grid.spin)
    print(f"   INSIDE GYM {tv.map_id(b)} at {tv.coords(b)} — Grid.spin = {spin_ct} tiles",
          flush=True)
    if spin_ct == 0:
        print("==== RESULT: FAIL ==== (0 spin tiles read on the Viridian Gym floor — "
              "classifier or warp landed wrong)", flush=True)
        sys.exit(1)

    trav_lines.clear()
    w0 = camp.trav.wedge_total
    camp.trav.travel(target_map=None, arrive_coord=MAZE_TARGET, max_steps=500, max_seconds=300)
    cur = tuple(tv.coords(b) or (-9, -9))
    dist = abs(cur[0] - MAZE_TARGET[0]) + abs(cur[1] - MAZE_TARGET[1])
    assist_fired = any("glide-crosser assist" in m for m in trav_lines)
    wedges = camp.trav.wedge_total - w0

    # one bounded retry: a single leg can die on an unlucky glide; the real loop re-picks too
    if dist > 1 and tuple(tv.map_id(b)) == MAP_VIRIDIAN_GYM:
        print(f"   first leg ended at {cur} (dist {dist}, wedges {wedges}) — one retry",
              flush=True)
        camp.trav.travel(target_map=None, arrive_coord=MAZE_TARGET, max_steps=500,
                         max_seconds=300)
        cur = tuple(tv.coords(b) or (-9, -9))
        dist = abs(cur[0] - MAZE_TARGET[0]) + abs(cur[1] - MAZE_TARGET[1])
        assist_fired = assist_fired or any("glide-crosser assist" in m for m in trav_lines)
        wedges = camp.trav.wedge_total - w0

    print(f"   final: map={tv.map_id(b)} coords={cur} dist_to_target={dist} "
          f"spin_assist_fired={assist_fired} leg_wedges={wedges}", flush=True)
    if tuple(tv.map_id(b)) == MAP_VIRIDIAN_GYM and dist <= 1:
        how = ("spin_assist hand-off fired and crossed" if assist_fired
               else "plain travel crossed (no wedge -> assist not needed; assist path "
                    "still unexercised on this floor)")
        print(f"==== RESULT: PASS ==== ({how}; wedges this leg: {wedges})", flush=True)
        sys.exit(0)
    print(f"==== RESULT: FAIL ==== (never reached {MAZE_TARGET}; ended {cur} on "
          f"{tv.map_id(b)}; assist_fired={assist_fired}, wedges={wedges})", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
