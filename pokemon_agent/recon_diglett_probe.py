"""NS#16 probe: why does head_to_gym wedge in Diglett's Cave -> Route-2 pocket?
Boots banked_LIVE (Diglett's Cave), loads the canonical world graph like recon_longrun,
and tests: (1) her boot map/coords, (2) live-read warps on this map, (3) whether her feet
can travel to the deeper warp toward Route 11 (1,38), (4) _warp_hop_reachable for each warp,
(5) route to Vermilion. Read-only diagnosis; banks nothing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from bridge import Bridge
import travel as tv
import campaign as C
from campaign import Campaign, resolve_state

ROM = C.ROM if hasattr(C, "ROM") else os.path.join(os.path.dirname(__file__), "firered.gba")
BOOT = os.environ.get("PROBE_STATE", "G:/temp/longrun/banked_LIVE/kira_campaign.state")

b = Bridge(ROM)
with open(resolve_state(BOOT), "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.run_frame()

camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
try:
    camp.world.load(C.WORLD_JSON)
except Exception as e:
    print("world load err", e)

m = tuple(tv.map_id(b)); cur = tuple(tv.coords(b))
print(f"BOOT map={m} coords={cur} place={camp._place_name(m) if hasattr(camp,'_place_name') else '?'}")

# live warps on this map
try:
    warps = tv.read_warps(b)
    print("LIVE warps on this map:")
    for w in warps:
        print("   ", w)
except Exception as e:
    print("read_warps err", e)

# reachability of each warp tile from her feet
try:
    for w in warps:
        xy = w[0]
        r = camp._warp_hop_reachable(tuple(xy))
        print(f"  _warp_hop_reachable({xy}) = {r}  -> dest {w[1] if len(w)>1 else '?'}")
except Exception as e:
    print("warp_hop err", e)

# route to Vermilion (3,5) on current (canonical+live) graph
print("route(cur_map -> 3,5):", camp.world.route(m, (3, 5)))
print("next_step(cur_map -> 3,5):", camp.world.next_step(m, (3, 5)))

# Try to physically travel to the deep warp toward 1,38 (Route 11 side)
deep = None
for w in warps:
    if len(w) > 1 and tuple(w[1]) == (1, 38):
        deep = tuple(w[0]); break
if deep:
    print(f"\nAttempt travel to deep warp {deep} (toward 1,38/Route 11)...")
    before = tuple(tv.map_id(b))
    try:
        camp.trav.travel(target_map=None, arrive_coord=deep, max_steps=400)
    except Exception as e:
        print("travel err", e)
    after = tuple(tv.map_id(b)); ac = tuple(tv.coords(b))
    print(f"  after: map={after} coords={ac}  (warped={after!=before})")
else:
    print("\nNo live warp toward 1,38 found on this map")
