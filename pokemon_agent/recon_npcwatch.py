"""recon_npcwatch.py — does the (30,12) NPC blocking the Cerulean Mart door MOVE (wanderer) or
sit (stationary)? Watch its position over ~40s of frames. Read-only."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import Campaign, resolve_state                       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OB, SZ = 0x02036E38, 0x24
OFF_ACTIVE, OFF_GFX, OFF_X, OFF_Y, OFF_FACING = 0x00, 0x05, 0x10, 0x12, 0x18


def find_gfx(b, gfx):
    """Scan all slots for an active object with this gfx; return (x,y,facing) or None."""
    for i in range(1, 16):
        o = OB + i * SZ
        if not (b.rd8(o + OFF_ACTIVE) & 1):
            continue
        if b.rd8(o + OFF_GFX) == gfx:
            return (b.rds16(o + OFF_X) - 7, b.rds16(o + OFF_Y) - 7, b.rd8(o + OFF_FACING) & 0x0F)
    return None


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._exit_to_overworld()
    for _ in range(30):
        b.run_frame()
    print(f"overworld map={tv.map_id(b)} player={tv.coords(b)}")
    # walk adjacent to the chokepoint so the NPC is on-screen and animates
    camp.trav.travel(target_map=None, arrive_coord=(29, 12), max_steps=200, max_seconds=60)
    print(f"positioned at {tv.coords(b)} (target adjacent (29,12))")
    seen = {}
    for t in range(40):                       # 40 * 60 frames ~= 40s
        for _ in range(60):
            b.run_frame()
        r = find_gfx(b, 60)
        if r is None:
            if t % 4 == 0:
                print(f"  t={t:2}s gfx60 NOT loaded")
            continue
        x, y, f = r
        seen[(x, y)] = seen.get((x, y), 0) + 1
        if t % 4 == 0:
            print(f"  t={t:2}s gfx60 pos=({x},{y}) facing={f}")
    print(f"\ndistinct positions visited by gfx60: {dict(seen)}")
    print("VERDICT:", "WANDERER (moves)" if len(seen) > 1 else "STATIONARY (never moved)")


if __name__ == "__main__":
    main()
