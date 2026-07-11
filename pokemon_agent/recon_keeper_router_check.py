"""recon_keeper_router_check.py — decision-logic verifier for the CROSS-MAP KEEPER ROUTER
(PASS 3 team-depth NEW#2, campaign.py _keeper_route_target/_place_to_map_index). Pure logic, no ROM:
stubs team_planner.assess / world.route / _species_on_map and asserts the router fires for a DUE off-map
keeper within range + room, and DEFERS (None) for on-map / party-full / out-of-range / unreachable /
non-catch / flag-off. Mirrors the NS#1-3 verifier style. Run: ../.venv/Scripts/python.exe recon_keeper_router_check.py
"""
import os, json, types
os.environ['POKEMON_KEEPER_ROUTER'] = '1'
import campaign as C          # noqa: E402
import travel as tv           # noqa: E402

enc = json.load(open(os.path.join("gamedata", "frlg_encounters.json"), encoding="utf-8"))


class Planner:
    encounters = enc
    def __init__(self, kind, sp): self._act = {"kind": kind, "species": sp}
    def assess(self, *a, **k): return dict(self._act)


class World:
    def __init__(self, routable): self.routable = routable
    def route(self, cur, tmap, avoid):
        h = self.routable.get((cur, tmap))
        return list(range(h + 1)) if h is not None else None


def make(cur, kind, sp, on_map, routable, rideable=True, unreach=None):
    s = types.SimpleNamespace()
    s._PLACE_NAMES = C.Campaign._PLACE_NAMES
    s._place2map_cache = None
    s._place_to_map_index = types.MethodType(C.Campaign._place_to_map_index, s)
    s.team_planner = Planner(kind, sp)
    s.world = World(routable)
    s.b = None
    s._cur = cur
    s._species_on_map = lambda species, mid: species.lower() in on_map
    s._wall_avoid = lambda st: set()
    # offer<=>executable: the router only offers a target the learned-graph traveler can ride NOW
    s._next_step_rideable = lambda cur, dst, avoid: (("hop", "edge", "N") if rideable else None)
    s._keeper_unreach = set(unreach or [])
    # NS#40 static-route refactor: _keeper_route_target now delegates to _reachable_keeper_host. Bind it
    # + inert static-gateway stubs (no door-entered hosts here → the static pass is a no-op, so these
    # cases exercise the learned-graph scan exactly as before the extraction).
    s._reachable_keeper_host = types.MethodType(C.Campaign._reachable_keeper_host, s)
    s._host_gateways = lambda: {}
    s._keeper_gateway = lambda *a, **k: None
    return s


_orig = tv.map_id


def run(s, st):
    tv.map_id = lambda b: s._cur
    try:
        return C.Campaign._keeper_route_target(s, st)
    finally:
        tv.map_id = _orig


def main():
    base = {"party": [1, 2, 3], "party_count": 3, "badge_count": 2, "bag": {}, "dex_caught": []}
    full = dict(base); full["party"] = [1] * 6; full["party_count"] = 6
    idx = C.Campaign._place_to_map_index(make((0, 0), "catch_keeper", "abra", set(), {}))
    assert idx["Route 24"] == (3, 43) and idx["Route 25"] == (3, 44) and idx["Diglett's Cave"] == (1, 36)
    print("PASS reverse index (Route24/25, Diglett entrance)")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 3, ((3, 3), (3, 44)): 5}), base) == ("abra", (3, 43))
    print("PASS A off-map in-range -> nearest hosting map")
    assert run(make((3, 43), "catch_keeper", "abra", {"abra"}, {((3, 43), (3, 44)): 2}), base) is None
    print("PASS B on-map -> None (on-map un-gate owns it)")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 3}), full) is None
    print("PASS C party full -> None")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 9}), base) is None
    print("PASS D out-of-range -> None")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {}), base) is None
    print("PASS E unreachable -> None")
    assert run(make((3, 3), "grind_to", "abra", set(), {((3, 3), (3, 43)): 3}), base) is None
    print("PASS F non-catch action -> None")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 3}, rideable=False), base) is None
    print("PASS H offer<=>executable: no rideable next hop -> None (livelock fix)")
    assert run(make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 3, ((3, 3), (3, 44)): 5},
                    unreach={((3, 3), (3, 43))}), base) == ("abra", (3, 44))
    print("PASS I retired target skipped -> falls to next hosting map")
    os.environ['POKEMON_KEEPER_ROUTER'] = '0'
    import importlib; importlib.reload(C)
    s = make((3, 3), "catch_keeper", "abra", set(), {((3, 3), (3, 43)): 3})
    s._PLACE_NAMES = C.Campaign._PLACE_NAMES
    s._place_to_map_index = types.MethodType(C.Campaign._place_to_map_index, s)
    assert run(s, base) is None
    print("PASS G flag OFF -> None")
    print("\nALL DECISION-LOGIC CASES PASS")


if __name__ == "__main__":
    main()
