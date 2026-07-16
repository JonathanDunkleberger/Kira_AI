"""recon_caveexec.py - EXECUTE the planned Mt Moon route + prove the far-side emergence (WATCH=1).

Loads mtmoon_plan.json (region-graph planner output) and runs CaveNav.run_plan from
mtmoon_interior. WATCH=1 (default) shows Kira traversing the maze live; INVINCIBLE=1 isolates nav
from the party=1 survival problem. On a REAL emergence + east-cross to Cerulean, banks
cerulean_entered.state and verifies the map is genuinely Cerulean (not (3,22)).
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_caveexec.py
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WATCH = os.getenv("WATCH", "1") == "1"
INVINCIBLE = os.getenv("INVINCIBLE", "1") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402
from cave_nav import CaveNav          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")


def main():
    plan = json.load(open(os.path.join(STATES, "mtmoon_plan.json")))
    b = Bridge(ROM)
    with open(os.path.join(STATES, "mtmoon_interior.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write

    def heal():
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))
    heal()

    render = (lambda: None)
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - Mt Moon planned traversal (watch)")

        def render():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            if INVINCIBLE and st.in_battle(b):
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
    else:
        def render():
            if INVINCIBLE and st.in_battle(b):
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))

    def fr():
        BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                    log=lambda m: print(f"      {m}", flush=True)).run(90)
        heal(); return "win"
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=render)
    camp._suppress_heal = True
    nav = CaveNav(b, camp, fog_path=None, on_event=lambda *a, **k: None, render=render,
                  log=lambda m: print(m, flush=True))
    print(f"   [exec] plan={plan}", flush=True)
    res = nav.run_plan([(tuple(m), tuple(w)) for m, w in plan], fwd_edge="east")
    final = tv.map_id(b)
    print(f"   [exec] RESULT={res} final_map={final} coords={tv.coords(b)}", flush=True)
    if res == "exited" and final[0] != 1 and final != (3, 22):
        with open(os.path.join(STATES, "cerulean_entered.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print(f"   [exec] *** banked cerulean_entered.state on map {final} (genuine far side) ***", flush=True)
    else:
        print(f"   [exec] NOT banking (res={res}, map={final}) - no genuine far-side emergence", flush=True)

    if WATCH:
        import pygame
        print("   [exec] holding window OPEN for inspection (close it or Ctrl-C to exit)", flush=True)
        clock = pygame.time.Clock()
        try:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                b.run_frame()
                surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
                screen.blit(pygame.transform.scale(surf, win), (0, 0))
                pygame.display.flip()
                clock.tick(60)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
