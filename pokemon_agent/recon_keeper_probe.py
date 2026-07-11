"""recon_keeper_probe.py — behavioral probe: boot a fixture like recon_longrun and report the LIVE
keeper-router decision (assess + _keeper_route_target), ball supply, party, badge, current map, and the
world.route hop distance to Route 24/25 (Abra). Read-only: NO run, NO bank. Picks the fixture to test.
Run: POKEMON_KEEPER_ROUTER=1 ../.venv/Scripts/python.exe recon_keeper_probe.py <fixture>
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")
os.environ.setdefault("POKEMON_KEEPER_ROUTER", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
import travel as tv
import campaign as C
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "surge_done_kit.state"
    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(C.WORLD_JSON)
    except Exception as e:
        print("world load fail:", e)

    cur = tuple(tv.map_id(b))
    co = tv.coords(b)
    ls = camp.read_live_state()
    party = ls.get("party") or []
    pc = ls.get("party_count") or len(party)
    bc = ls.get("badge_count", 0)
    print(f"BOOT {boot}")
    print(f"  map={cur} ({camp._place_name(cur)}) coords={co} badge_count={bc} party_count={pc}")
    print(f"  party species: {[st.SPECIES_NAME.get(s, s) if isinstance(s,int) else s for s in party]}")
    try:
        balls = camp._balls_pocket_count(C.ITEM_POKE_BALL)
    except Exception as e:
        balls = f"ERR {e}"
    print(f"  poke balls: {balls}")

    act = camp.team_planner.assess(party, bc, bag=ls.get("bag"),
                                   dex=ls.get("dex_caught"), post_game=bool(ls.get("post_game")))
    print(f"  assess() -> {act}")

    tgt = camp._keeper_route_target(ls)
    print(f"  _keeper_route_target() -> {tgt}")

    avoid = camp._wall_avoid(ls)
    for place, tmap in [("Route 24", (3, 43)), ("Route 25", (3, 44)), ("Diglett's Cave", (1, 36))]:
        r = camp.world.route(cur, tmap, avoid)
        hops = (len(r) - 1) if r else None
        step = camp._next_step_rideable(cur, tmap, avoid) if r else None
        print(f"  route {cur}->{place}{tmap}: hops={hops} rideable_next={step}")


if __name__ == "__main__":
    main()
