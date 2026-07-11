"""recon_static_keeper_check.py — decision-logic verifier for the STATIC-CONNECTION keeper route
(PASS 3 NS#40, campaign.py _reachable_keeper_host static pass + _keeper_gateway + _host_gateways +
gamedata/frlg_connections.json). Pure logic, no ROM: stubs world.route / _next_step_rideable and asserts
that a DOOR-entered cave host (Diglett's Cave) invisible to the LEARNED graph becomes reachable when an
overworld GATEWAY (Route 2 / Route 11) is ride-reachable — the NS#39 keeper-reachability chicken-and-egg —
while every guard (flag off, gateway unreachable, out of range, retired, non-host species) still defers.
Also asserts the existing LEARNED overworld path (Abra on Route 24/25) is untouched.
Run: ../.venv/Scripts/python.exe recon_static_keeper_check.py
"""
import os, json, types
os.environ['POKEMON_KEEPER_ROUTER'] = '1'
os.environ['POKEMON_KEEPER_STATIC_ROUTE'] = '1'
import campaign as C          # noqa: E402

enc = json.load(open(os.path.join("gamedata", "frlg_encounters.json"), encoding="utf-8"))

VERMILION, ROUTE11, ROUTE2, DIGLETT = (3, 5), (3, 29), (3, 20), (1, 36)
ROUTE24, CERULEAN = (3, 43), (3, 3)


class Planner:
    encounters = enc


class World:
    """route() is True only for pairs in `routable` (learned-graph truth)."""
    def __init__(self, routable): self.routable = routable
    def route(self, cur, dst, avoid=None):
        h = self.routable.get((tuple(cur), tuple(dst)))
        return list(range(h + 1)) if h is not None else None


def make(routable, rideable_pairs=None, unreach=None, static=True):
    """rideable_pairs: pairs for which _next_step_rideable returns a hop (else None). Default: every
    routable pair is rideable (the common case)."""
    s = types.SimpleNamespace()
    s._PLACE_NAMES = C.Campaign._PLACE_NAMES
    s._place2map_cache = None
    s._host_gw_cache = None
    s._keeper_unreach = set(unreach or [])
    s.team_planner = Planner()
    s.world = World(routable)
    rp = set(rideable_pairs) if rideable_pairs is not None else set(routable.keys())
    s._place_to_map_index = types.MethodType(C.Campaign._place_to_map_index, s)
    s._host_gateways = types.MethodType(C.Campaign._host_gateways, s)
    s._keeper_gateway = types.MethodType(C.Campaign._keeper_gateway, s)
    s._place_name = types.MethodType(C.Campaign._place_name, s)
    s._species_on_map = lambda species, mid: False
    s._wall_avoid = lambda st: set()
    s._story_gate_avoid = lambda st: set()   # NS#42: no Flute-gated maps in these static-gateway cases
    s._next_step_rideable = lambda cur, dst, avoid: (("hop", "edge", "N") if (tuple(cur), tuple(dst)) in rp else None)
    return s


def reach(s, sp, cur):
    return C.Campaign._reachable_keeper_host(s, sp, cur, {})


def main():
    ok = 0

    # KB sanity: the static gateway table loaded from gamedata/frlg_connections.json
    gws = make({})._host_gateways()
    assert gws.get("Diglett's Cave") == [(3, 20), (3, 29)], gws
    print("PASS KB: Diglett's Cave gateways = Route 2 (3,20), Route 11 (3,29)"); ok += 1

    # 1. THE HEADLINE: Vermilion -> Diglett unreachable in the LEARNED graph, but Route 11 gateway is
    #    ride-reachable -> static pass returns the Diglett entrance (1,36). (was None pre-NS#40)
    s = make({(VERMILION, ROUTE11): 1})          # NO (Vermilion,Diglett) route; Route 11 is 1 hop
    assert reach(s, "diglett", VERMILION) == DIGLETT
    print("PASS 1: Vermilion->Diglett via Route 11 gateway (static) -> (1,36)"); ok += 1

    # 2. Flag OFF -> the static pass is skipped -> None (proves the gate + the pre-NS#40 behaviour)
    s = make({(VERMILION, ROUTE11): 1}, static=True)
    os.environ['POKEMON_KEEPER_STATIC_ROUTE'] = '0'
    import importlib; importlib.reload(C)
    s2 = types.SimpleNamespace(**s.__dict__)
    # rebind methods to the reloaded module
    for m in ("_place_to_map_index", "_host_gateways", "_keeper_gateway", "_place_name"):
        setattr(s2, m, types.MethodType(getattr(C.Campaign, m), s2))
    s2._host_gw_cache = None; s2._place2map_cache = None
    assert C.Campaign._reachable_keeper_host(s2, "diglett", VERMILION, {}) is None
    print("PASS 2: flag OFF -> static pass skipped -> None (pre-NS#40 behaviour)"); ok += 1
    os.environ['POKEMON_KEEPER_STATIC_ROUTE'] = '1'; importlib.reload(C)

    # 3. NO gateway ride-reachable (no route to Route 2 or Route 11) -> None (never a phantom offer)
    s = make({})                                  # nothing routable
    assert reach(s, "diglett", CERULEAN) is None
    print("PASS 3: no reachable gateway -> None"); ok += 1

    # 4. Gateway routable but the next hop is NOT rideable (offer<=>executable guard) -> None
    s = make({(VERMILION, ROUTE11): 1}, rideable_pairs=set())
    assert reach(s, "diglett", VERMILION) is None
    print("PASS 4: gateway routable but no rideable hop -> None (livelock guard)"); ok += 1

    # 5. Gateway beyond MAX_HOPS -> None (bounded / watchable)
    far = C.KEEPER_ROUTER_MAX_HOPS + 2
    s = make({(VERMILION, ROUTE11): far})
    assert reach(s, "diglett", VERMILION) is None
    print(f"PASS 5: gateway {far} hops (> MAX {C.KEEPER_ROUTER_MAX_HOPS}) -> None"); ok += 1

    # 6. Standing ON a gateway (cur == Route 11) -> host reachable (step through the door next)
    s = make({})                                  # no learned route needed; cur IS a gateway
    assert reach(s, "diglett", ROUTE11) == DIGLETT
    print("PASS 6: standing on the Route 11 gateway -> (1,36)"); ok += 1

    # 7. LEARNED overworld host (Abra, Route 24) still resolves via the learned scan, static untouched
    s = make({(CERULEAN, ROUTE24): 2})
    assert reach(s, "abra", CERULEAN) == ROUTE24
    print("PASS 7: learned overworld host (Abra Route 24) unchanged by static pass"); ok += 1

    # 8. LEARNED route to the cave exists -> learned scan owns it, static pass never consulted
    s = make({(ROUTE11, DIGLETT): 1})
    assert reach(s, "diglett", ROUTE11) == DIGLETT
    print("PASS 8: learned route to the cave -> learned scan (not static)"); ok += 1

    # 9. Retired gateway pair (in _keeper_unreach) is skipped -> None
    s = make({(VERMILION, ROUTE11): 1}, unreach={(VERMILION, ROUTE11)})
    assert reach(s, "diglett", VERMILION) is None
    print("PASS 9: retired gateway pair skipped -> None"); ok += 1

    # 10. Non-host species (mankey — overworld only, no static gateway) -> None from an isolated map
    s = make({})
    assert reach(s, "mankey", VERMILION) is None
    print("PASS 10: species with no reachable host -> None"); ok += 1

    print(f"\nALL {ok}/10 static-keeper-route decision cases PASS")


if __name__ == "__main__":
    main()
