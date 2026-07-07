"""recon_elev_probe.py — hideout10 postmortem probe: B1F is SPLIT by a full wall at y19-20;
its elevator doors (23-25,25) live in the SOUTH half, served by stairs (15,30)<->B2F (23,12).
From the staged B1F@(17,2) save: climb back to B2F, dump the B2F grid, ride (23,12) down,
then approach the elevator doors from (24,26). Read-only recon + boarding attempt; no banking.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_elev_probe.py
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
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_hideout")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "hideout_probe")


def main():
    b = Bridge(ROM)
    with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=fight, on_event=lambda s, **k: print(f"[event] {s}", flush=True),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    def L(m):
        print(m, flush=True)

    def dump_grid(tag):
        g = tv.Grid(b)
        npc = set(camp.trav._npc_tiles())
        w_play = b.rd32(tv.BACKUP_LAYOUT) - 14
        h_play = b.rd32(tv.BACKUP_LAYOUT + 4) - 14
        lines = [f"map={tv.map_id(b)} coords={tv.coords(b)} dims={w_play}x{h_play} npcs={sorted(npc)}"]
        for y in range(h_play):
            row = []
            for x in range(w_play):
                v = camp._tile_behavior(x, y)
                mark = "N!" if (x, y) in npc else ("##" if not g.walkable(x, y) else "..")
                # show the BEHAVIOR byte even on blocked tiles (doors read as walls in collision)
                row.append(f"{v:02x}" if v else mark)
                row.append(" ")
            lines.append(f"y{y:02d} " + "".join(row))
        p = os.path.join(DBG, f"grid_{tag}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        L(f"grid -> {p}")

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)}")
    if tuple(tv.map_id(b)) != (1, 42):
        L("!! stage save not on B1F — abort")
        return 1
    # behaviors of the B1F elevator door row (doors read as # in collision; what's the byte?)
    for y in (24, 25, 26):
        L("B1F row y%d behaviors: %s" % (y, [f"({x},{y})={camp._tile_behavior(x, y):02x}"
                                             for x in range(20, 28)]))
    # 1. back up to B2F
    if camp.enter_warp(pick=(17, 2)) != "warped":
        L("!! couldn't ride B1F (17,2) up — abort")
        return 1
    for _ in range(80):
        b.run_frame()
    L(f"on B2F: map={tv.map_id(b)} coords={tv.coords(b)}")
    dump_grid("b2f")
    # 2. down the mid-map stairs (23,12) -> B1F south half
    m0 = tuple(tv.map_id(b))
    r = camp.enter_warp(pick=(23, 12))
    L(f"enter_warp (23,12) -> {r}; now {tv.map_id(b)}@{tv.coords(b)}")
    if tuple(tv.map_id(b)) == m0:
        return 1
    for _ in range(80):
        b.run_frame()
    # 3. approach the elevator doors from below; UP-step into (24,25)
    m1 = tuple(tv.map_id(b))
    for door in ((24, 25), (23, 25), (25, 25)):
        r = camp.enter_warp(pick=door)
        L(f"enter_warp {door} -> {r}; now {tv.map_id(b)}@{tv.coords(b)}")
        if tuple(tv.map_id(b)) != m1:
            L("BOARDED THE ELEVATOR")
            dump_grid("elevator")
            return 0
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, "probe_doors_fail.png"))
    except Exception:
        pass
    dump_grid("b1f_south")
    L("!! doors did not warp from the south half — read the grids/frame")
    return 1


if __name__ == "__main__":
    sys.exit(main())
