"""recon_silph_probe.py — ground truth on the Silph 1F entrance-mat livelock (shift #9).

strike3/4 loop: enter 1F @ (8,20) -> something bounces her back to the street before the
floor2-up dispatch. The only actuator between the two map reads is the step-off-arrival-warp
guard. Probe: enter exactly as the strike does, dump the mat's behavior byte + the guard's
gate decision + neighbors, then run the guard's exact block and watch map/coords.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_silph_probe.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
F1 = (1, 47)


def main():
    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "skip",
                    on_event=lambda s, **k: print(f"[event] {s}", flush=True),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    print(f"boot {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    r = camp.enter_warp(pick=(33, 30))
    print(f"enter_warp(33,30) -> {r} | now {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    for _ in range(80):
        b.run_frame()
    m = tuple(tv.map_id(b))
    cur = tuple(tv.coords(b) or (0, 0))
    print(f"settled {m}@{cur}", flush=True)
    if m != F1:
        print("!! not inside — abort probe", flush=True)
        return 1

    warps = tv.read_warps(b)
    print(f"warps: {[(tuple(w[0]), tuple(w[1])) for w in warps]}", flush=True)
    warp_tiles = {tuple(w[0]) for w in warps}
    bh = camp._tile_behavior(*cur)
    print(f"behavior{cur} = {bh:#04x}  in _WARP_ENTRY = {bh in camp._WARP_ENTRY}", flush=True)
    g = tv.Grid(b)
    for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
               (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
        print(f"  nb {nb}: warp={nb in warp_tiles} "
              f"walkable={g.walkable(nb[0] + tv.MAP_OFFSET, nb[1] + tv.MAP_OFFSET)} "
              f"beh={camp._tile_behavior(*nb):#04x}", flush=True)

    # the strike's exact step-off block
    if cur in warp_tiles and camp._tile_behavior(*cur) in camp._WARP_ENTRY:
        for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                   (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
            if nb not in warp_tiles and g.walkable(nb[0] + tv.MAP_OFFSET, nb[1] + tv.MAP_OFFSET):
                print(f"GUARD FIRES -> _step_to({nb})", flush=True)
                camp._step_to(nb)
                break
        for _ in range(60):
            b.run_frame()
        print(f"after step-off: {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    else:
        print("GUARD does NOT fire (gate held)", flush=True)

    # regardless: can she path to the 2F stair (31,2) from here?
    m2 = tuple(tv.map_id(b))
    if m2 == F1:
        cur2 = tuple(tv.coords(b) or (0, 0))
        g2 = tv.Grid(b)
        p = tv.bfs(g2, cur2, lambda t: t == (31, 2), walkable=g2.walkable)
        print(f"bfs {cur2} -> (31,2): {'PATH len ' + str(len(p)) if p else 'NO PATH'}", flush=True)
        st_beh = camp._tile_behavior(31, 2)
        print(f"behavior(31,2) = {st_beh:#04x} in _WARP_ENTRY = {st_beh in camp._WARP_ENTRY}",
              flush=True)
        r2 = camp.enter_warp(pick=(31, 2))
        print(f"enter_warp(31,2) -> {r2} | now {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
