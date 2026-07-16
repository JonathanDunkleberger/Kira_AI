"""recon_gymscan.py - READ-ONLY ground truth for the GENERAL gym handler.

From cerulean_caught: travel to Cerulean -> enter the gym -> SAVE states/cerulean_gym.state
(a fast iteration checkpoint) -> DUMP the full ObjectEvent table so we positively identify
the junior TRAINERS (trainerType!=0) vs the LEADER (Misty) vs the guide NPC, with each one's
coords + facing + sight range. This tells the handler exactly who to engage and where their
line of sight is - so we fight the trainers FIRST, then the gated leader. No guessing.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_gymscan.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge        # noqa: E402
import travel as tv              # noqa: E402
import pokemon_state as st       # noqa: E402
from campaign import Campaign    # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
CERULEAN = (3, 3)
GYM_DOOR = (31, 21)
OB, SZ, N = 0x02036E38, 0x24, 16
MAP_OFFSET = 7
FACE = {1: "down(0,+1)", 2: "up(0,-1)", 3: "left(-1,0)", 4: "right(+1,0)"}


def dump_objects(b, tag=""):
    print(f"\n=== OBJECT TABLE {tag}  player={tv.coords(b)} map={tv.map_id(b)} ===", flush=True)
    for i in range(N):
        o = OB + i * SZ
        if not (b.rd8(o) & 1):
            continue
        gfx = b.rd8(o + 0x05)
        mvt = b.rd8(o + 0x06)
        ttype = b.rd8(o + 0x07)
        trange = b.rd8(o + 0x1A)
        coord = (b.rds16(o + 0x10) - MAP_OFFSET, b.rds16(o + 0x12) - MAP_OFFSET)
        face = b.rd8(o + 0x18) & 0x0F
        tag2 = "  <<< TRAINER" if ttype in (1, 2) else ("  (player)" if i == 0 else "")
        print(f"   obj{i:2} gfx=0x{gfx:02x} mvt=0x{mvt:02x} trainerType={ttype} "
              f"range={trange} coord={coord} facing={face}={FACE.get(face,'?')}{tag2}", flush=True)


def main():
    b = Bridge(ROM)
    gym_ckpt = os.path.join(STATES, "cerulean_gym.state")
    boot = gym_ckpt if os.path.exists(gym_ckpt) else os.path.join(STATES, "cerulean_caught.state")
    with open(boot, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")

    def fr():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(f"      {m}", flush=True)).run(120)
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._suppress_heal = True

    if boot.endswith("cerulean_gym.state"):
        print(f"loaded gym checkpoint @{tv.coords(b)} map={tv.map_id(b)}", flush=True)
    else:
        if tv.map_id(b) != CERULEAN:
            camp.trav.travel(target_map=CERULEAN, edge="east", max_steps=400, max_seconds=240)
        camp.trav.travel(target_map=None, arrive_coord=(GYM_DOOR[0], GYM_DOOR[1] + 1),
                         max_steps=400, max_seconds=120)
        before = tv.map_id(b)
        for _ in range(6):
            b.press("UP", 8, 10, lambda: None, owner="agent")
            for _ in range(20):
                b.run_frame()
            if tv.map_id(b) != before:
                break
        for _ in range(60):
            b.run_frame()
        with open(gym_ckpt, "wb") as f:
            f.write(bytes(b.save_state()))
        print(f"inside gym map={tv.map_id(b)} @{tv.coords(b)} - banked cerulean_gym.state", flush=True)

    dump_objects(b, "AT GYM ENTRANCE")

    # Navigate up toward Misty's area (proven reachable at (8,7)); fight any trainer LoS en route,
    # then re-scan from the TOP so Misty + the 2nd trainer (proximity-loaded) resolve.
    for attempt in range(6):
        out = camp.trav.travel(target_map=None, arrive_coord=(8, 7), max_steps=200, max_seconds=90)
        print(f"   travel->{out} now @{tv.coords(b)} in_battle={st.in_battle(b)}", flush=True)
        if st.in_battle(b):
            rb = st.read_battle(b)
            if rb:
                print(f"   battle vs species={rb['enemy']['species']} "
                      f"{st.SPECIES_NAME.get(rb['enemy']['species'],'?')} -> fighting", flush=True)
            print(f"   battle -> {fr()}", flush=True)
            for _ in range(40):
                b.run_frame()
            b.set_input_owner("agent")
            dump_objects(b, f"AFTER TRAINER FIGHT {attempt}")
            continue
        break
    dump_objects(b, "AT MISTY AREA (top)")


if __name__ == "__main__":
    main()
