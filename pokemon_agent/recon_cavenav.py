"""recon_cavenav.py - run CaveNav on Mt Moon. Control: reaches the far exit (no infinite loop).

From mtmoon_interior.state (inside Mt Moon 1F, entered from Route 3 (3,21)), explore the warp
graph to the Cerulean-side exit. flee wild battles / win trainers; sandbox-heal so HP economy
(no reachable Center) doesn't sabotage the traversal we're testing.
RUN: .venv\\Scripts\\python.exe pokemon_agent\\recon_cavenav.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WATCH = os.getenv("WATCH", "1") == "1"      # visible window so Jonny watches; WATCH=0 = headless
INVINCIBLE = os.getenv("INVINCIBLE", "0") == "1"   # top up battle HP each frame: ISOLATE nav from
#                                                    survival (party=1 fixture can't reserve-switch)
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402
from cave_nav import CaveNav          # noqa: E402
import pokemon_state as st            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
FOG = os.path.join(_HERE, "states", "mtmoon_fog.json")
ENTRANCE_MAP = (3, 21)     # Route 3 - we came in from here; never treat as exit / step back


def main():
    if os.path.exists(FOG):
        os.remove(FOG)                 # fresh fog each recon run
    b = Bridge(ROM)
    with open(os.path.join(STATES, "mtmoon_interior.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write
    for s in range(ram.read_party_count(b)):
        base = ram.GPLAYER_PARTY + s * 100
        w16(base + 0x56, b.rd16(base + 0x58))     # sandbox-heal (HP economy is separate work)

    # ── VISIBLE window so Jonny can watch the traversal live ────────────────────
    render = (lambda: None)
    if WATCH:
        import pygame
        pygame.init()
        scale = 3
        win = (b.width * scale, b.height * scale)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - Mt Moon cave-nav (watch)")

        def render():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            if INVINCIBLE and st.in_battle(b):       # NAV ISOLATION: un-KO-able so a mid-battle
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))   # HP := maxHP
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
    else:
        def render():
            if INVINCIBLE and st.in_battle(b):
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))

    def heal_party():
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))

    def fight_runner():
        # FIGHT (not flee): cave foes are often faster than a flee can escape -> the
        # flee-can't-escape infinite loop. Fighting guarantees the battle ENDS; a healed
        # Bulbasaur sweeps Zubat/Geodude (Vine Whip), forced faint-switch covers a KO.
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                          log=lambda m: print(f"      {m}", flush=True)).run(max_seconds=90)
        if os.getenv("HEAL_AFTER", "1") == "1":
            heal_party()      # ISOLATE NAV from survival: no attrition blackout (HP economy is
        return out            # the separate roster/nearest-Center concern, tested elsewhere)

    camp = Campaign(b, battle_runner=fight_runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=render)
    camp._suppress_heal = True
    nav = CaveNav(b, camp, fog_path=FOG, on_event=lambda *a, **k: None,
                  render=render, log=lambda m: print(m, flush=True))
    res = nav.clear_cave(ENTRANCE_MAP, max_hops=int(os.getenv("MAX_HOPS", "30")),
                         max_seconds=int(os.getenv("MAX_SECONDS", "1200")))
    print(f"   [cave] RESULT: {res}  final_map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
    if res == "exited":
        with open(os.path.join(STATES, "mtmoon_exit.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print("   [cave] banked mtmoon_exit.state", flush=True)


if __name__ == "__main__":
    main()
