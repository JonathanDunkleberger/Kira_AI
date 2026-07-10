"""recon_fuchsia_gate.py — SEE the Route 15 -> Fuchsia gatehouse she wedges in (rule 15.4).

s8_run3 STALLED inside gatehouse (24,0)@(9,10): head_to_gym's door-passthrough loops the
(9,10)<->(10,9) doors between (24,0) and (24,1) and never takes the door that opens onto Fuchsia
City (west). This probe boots the wedge fixture, grabs a frame, prints her pos + the object/warp
layout, then walks WEST toward Fuchsia and reports where each gatehouse door actually lands, so the
fix knows which door -> Fuchsia.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_fuchsia_gate.py
"""
import os
import sys

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
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "fuchsia_gate")


def show(b, tag):
    print(f"  [{tag}] map={tuple(tv.map_id(b))} coords={tuple(tv.coords(b))}", flush=True)


def snap(b, name):
    os.makedirs(DBG, exist_ok=True)
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
        print(f"  snap -> {name}.png", flush=True)
    except Exception as e:
        print(f"  snap failed: {e}", flush=True)


def main():
    boot = resolve_state("fuchsia_gate.state")
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
    snap(b, "boot")
    try:
        tmpl = tv.read_object_templates(b)
        print(f"  object templates ({len(tmpl)}): "
              f"{[(t, p) for t, _g, p in tmpl][:12]}", flush=True)
    except Exception as e:
        print(f"  templates: {e}", flush=True)

    # Test each specific door of (24,0) by pick to learn its TRUE destination (Fuchsia = 3,7).
    start = tuple(tv.map_id(b))
    for door in [(1, 6), (1, 7), (11, 6), (11, 7)]:
        b2 = Bridge(ROM)
        with open(boot, "rb") as f:
            b2.load_state(f.read())
        for _ in range(20):
            b2.run_frame()
        c2 = Campaign(b2, battle_runner=lambda *a, **k: "win",
                      on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                      render=lambda: None)
        try:
            c2.world.load(C.WORLD_JSON)
        except Exception:
            pass
        try:
            r = c2.enter_warp(pick=door)
        except Exception as e:
            r = f"raised {e!r}"
        m2 = tuple(tv.map_id(b2))
        tag = "FUCHSIA!" if m2 == (3, 7) else ("Route15" if m2 == (3, 33) else str(m2))
        print(f"  door {door} -> {r}; now {m2}@{tuple(tv.coords(b2))}  [{tag}]", flush=True)
        if m2 != start:
            snap(b2, f"door_{door[0]}_{door[1]}_{m2[0]}_{m2[1]}")


if __name__ == "__main__":
    main()
