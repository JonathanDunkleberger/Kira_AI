"""recon_misty_gate.py — GROUND TRUTH on the Misty gym stuck (night-train shift 3).

Loads the banked_CRASH savestate (frozen at the Misty stuck: in the Cerulean gym at (8,7), a junior
falsely marked 'beaten' after a nav wedge). Reads the LOADED trainer objects (the unbeaten juniors),
and tests LAND reachability of each trainer's front tiles from the player — so we know if the swimmer
at (10,12) is genuinely unreachable (water/elevation) or was a transient block.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_misty_gate.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402
import firered_ram as ram          # noqa: E402
import pokemon_state as st         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = os.getenv("STATE", r"G:\temp\longrun\banked_CRASH\kira_campaign.state")

_OB, _SZ = 0x02036E38, 0x24


def gym_objects(b):
    """All active object events: (idx, coord, facing, trainerType)."""
    out = []
    for i in range(1, 16):
        o = _OB + i * _SZ
        if b.rd8(o) & 1:
            c = (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7)
            out.append((i, c, b.rd8(o + 0x18) & 0x0F, b.rd8(o + 0x07)))
    return out


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    b.set_input_owner("agent")
    mp = tv.map_id(b)
    me = tv.coords(b)
    print(f"MAP={mp}  PLAYER={me}")
    print("OBJECTS (idx, coord, facing, trainerType):")
    for (i, c, f, tt) in gym_objects(b):
        tag = "TRAINER" if tt in (1, 2) else "npc/sign"
        print(f"   obj{i}: {c} facing={f} tt={tt} [{tag}]")

    # Build the walk grid and test reachability from the player to each trainer's 4 front tiles.
    try:
        grid = tv.Grid(b)
    except Exception as e:
        grid = None
        print(f"Grid build failed: {e}")
    print("\nREACHABILITY (land BFS from player) to each TRAINER's front tiles:")
    for (i, c, f, tt) in gym_objects(b):
        if tt not in (1, 2):
            continue
        fronts = [(c[0] + dx, c[1] + dy) for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0))]
        reach = {}
        for ft in fronts:
            r = None
            try:
                r = tv.bfs(grid, me, (lambda g: (lambda cc: cc == g))(ft)) if grid is not None else None
            except Exception as e:
                r = f"err:{e}"
            standable = grid.walkable(*ft) if grid is not None else "?"
            reach[ft] = (f"REACH len{len(r)}" if isinstance(r, list) else "no_path") + f"(stand={standable})"
        print(f"   trainer obj{i} at {c}: " + ", ".join(f"{ft}->{v}" for ft, v in reach.items()))

    # Dump the passability of the tiles around the pool so we can see the water footprint.
    print("\nTILE GRID  o=walkable  ~=water  .=blocked  P=player  T=trainer   rows y=2..20, x=2..16:")
    trs = {c for (_i, c, _f, tt) in gym_objects(b) if tt in (1, 2)}
    for y in range(2, 21):
        row = []
        for x in range(2, 17):
            ch = "?"
            if (x, y) == me:
                ch = "P"
            elif (x, y) in trs:
                ch = "T"
            elif grid is not None:
                try:
                    if grid.is_water(x, y):
                        ch = "~"
                    else:
                        ch = "o" if grid.walkable(x, y) else "."
                except Exception:
                    ch = "?"
            row.append(ch)
        print(f"   y={y:2d} " + "".join(row))


if __name__ == "__main__":
    main()
