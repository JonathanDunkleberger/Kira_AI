"""recon_vermilion_map.py — map Vermilion by the proven Cerulean recipe (2026-07-06, task 6).

Boots the newest state that stands IN Vermilion (3,5) — banked_GOAL first, then stage, then canonical.
READ-ONLY on the world (no canonical writes): for each reachable door — enter, read the interior
SIGNATURE (Mart = clerk object at the fixed counter tile + the (0,7)-class layout; Center = nurse
counter + the healing machine; Gym = trainers + the leader), record `door -> interior map + verdict`,
exit. Prints the CITY_MART_DOORS / CITY_PC_DOORS / gym-door candidates to register in campaign.py.
RUN: python pokemon_agent/recon_vermilion_map.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CAND = [r"G:\temp\longrun\banked_GOAL\kira_campaign.state",
        r"G:\temp\longrun\stage\kira_campaign.state"]
OUT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\661370f2-1025-435c-8cf5-d2593621c432\scratchpad"


def objects(b):
    OB, SZ = 0x02036E38, 0x24
    out = []
    for i in range(16):
        o = OB + i * SZ
        try:
            if not (b.rd8(o) & 1):
                continue
            out.append({"idx": i, "gfx": b.rd8(o + 0x05),
                        "coord": (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7)})
        except Exception:
            continue
    return out


def main():
    b = Bridge(ROM)
    picked = None
    for p in CAND + [resolve_state("kira_campaign.state")]:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                b.load_state(f.read())
            for _ in range(40):
                b.run_frame()
            if tuple(tv.map_id(b)) == (3, 5):
                picked = p
                break
    if picked is None:
        print("!! no state standing in Vermilion (3,5) — run the look-ahead there first")
        return
    print(f"state: {picked}  pos={tv.coords(b)}", flush=True)
    b.set_input_owner("agent")

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    camp._suppress_heal = True
    b.frame_rgb().resize((720, 480)).save(os.path.join(OUT, "vermilion.png"))
    warps = tv.read_warps(b)
    print(f"Vermilion warps ({len(warps)}):", flush=True)
    for (xy, dest, wid) in warps:
        print(f"   {xy} -> {dest} (id {wid})", flush=True)
    grid = tv.Grid(b)
    cur = tuple(tv.coords(b))
    for (xy, dest, wid) in warps:
        appr = (xy[0], xy[1] + 1)
        reach = bool(tv.bfs(grid, cur, lambda t, a=appr: t == a, walkable=grid.walkable))
        if not reach:
            print(f"-- door {xy} -> {dest}: approach unreachable, skipping", flush=True)
            continue
        if camp.enter_warp(pick=tuple(xy), budget_s=180) != "warped":
            print(f"-- door {xy} -> {dest}: entry failed", flush=True)
            continue
        m_in = tuple(tv.map_id(b))
        obs = objects(b)
        iw = len(tv.read_warps(b))
        print(f"== door {xy} -> interior {m_in}: {len(obs)} object(s) {[(o['gfx'], o['coord']) for o in obs][:6]} "
              f"warps={iw}", flush=True)
        b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, f"verm_{m_in[0]}-{m_in[1]}.png"))
        camp._exit_to_overworld()
        if tuple(tv.map_id(b)) != (3, 5):
            print("!! stranded — stopping the sweep", flush=True)
            break
        grid = tv.Grid(b)
        cur = tuple(tv.coords(b))
    print("sweep done — match interiors to Mart/Center/gym by signature (clerk tile / nurse counter)",
          flush=True)


if __name__ == "__main__":
    main()
