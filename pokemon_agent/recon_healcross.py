"""recon_healcross.py — probe the Route-12 gatehouse NORTHBOUND crossing for the heal-return.

After waking the Snorlax she ends up hurt on Route 13 (3,31); the heal-return wants Lavender's
Center (north) but next_hop treats (3,30)->(3,4) as a plain north edge, and her feet are in Route
12's SOUTH section (split from the north section by the gatehouse), so _edge_band_reachable says
UNREACHABLE and the whole heal abandons -> hard stall. This probe boots the post-snorlax fixture,
walks her north into Route 12, and tries candidate crossers to find the one that reaches Lavender:
  (A) _next_step_rideable((3,30),(3,4))  — the warp-aware router head_to_gym uses
  (B) enter_warp(prefer='north')         — step through the northmost gatehouse door
  (C) the door-passthrough head_to_gym falls back on (PASSTHROUGH connectors)
Reports map/coords after each so the minimal heal fix reuses whatever crosses.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_healcross.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LAVENDER = (3, 4)
ROUTE12 = (3, 30)


def show(b, tag):
    print(f"  [{tag}] map={tuple(tv.map_id(b))} coords={tuple(tv.coords(b))}", flush=True)


def main():
    boot = resolve_state("snorlax_done.state")
    print(f"boot={boot}", flush=True)
    b = Bridge(ROM)
    with open(boot, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda *a, **k: "win",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=lambda: None)
    try:
        camp.world.load(C.WORLD_JSON)
    except Exception as e:
        print(f"world load: {e}", flush=True)
    show(b, "BOOT")

    # step 1: get onto Route 12 (she boots on Route 13 (3,31)); walk north to the map transition.
    if tuple(tv.map_id(b)) == (3, 31):
        camp.trav.travel(target_map=ROUTE12, edge="north")
        show(b, "after north->Route12")

    cur = tuple(tv.map_id(b))
    if cur != ROUTE12:
        print(f"!! not on Route 12 (on {cur}) — can't probe the gatehouse from here", flush=True)
        return

    # (A) warp-aware router toward Lavender
    try:
        ws = camp._next_step_rideable(cur, LAVENDER, avoid=set())
        print(f"  (A) _next_step_rideable({cur}->{LAVENDER}) = {ws}", flush=True)
    except Exception as e:
        print(f"  (A) raised: {e!r}", flush=True)

    # (B) northmost gatehouse door step-through
    before = tuple(tv.map_id(b))
    try:
        r = camp.enter_warp(prefer="north")
        print(f"  (B) enter_warp(prefer=north) -> {r}", flush=True)
        show(b, "after (B)")
        # if we warped into the gatehouse building, exit its north side toward Lavender
        for hop in range(4):
            m = tuple(tv.map_id(b))
            if m == LAVENDER:
                break
            if m == ROUTE12 and tuple(tv.coords(b))[1] < 15:
                camp.trav.travel(target_map=LAVENDER, edge="north")
                show(b, f"after edge-north hop{hop}")
                break
            r2 = camp.enter_warp(prefer="north")
            show(b, f"after (B) chain hop{hop} ({r2})")
    except Exception as e:
        print(f"  (B) raised: {e!r}", flush=True)

    print(f"RESULT: reached Lavender = {tuple(tv.map_id(b)) == LAVENDER} "
          f"(now {tuple(tv.map_id(b))}@{tuple(tv.coords(b))})", flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
