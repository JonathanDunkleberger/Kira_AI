"""recon_misty.py - BADGE 2 capability test (general gym handler shape). From cerulean_caught:
travel to Cerulean -> HEAL at the Cerulean Center (generalized nearest-center) -> enter the gym ->
BFS-navigate to Misty (the trainer-gauntlet auto-handles Luis/Diana) -> fight her with the CURRENT
team/moves (NOT a scripted sequence - capability, not decision) -> drain the award -> confirm the
Cascade Badge flag. INVINCIBLE=1 isolates gym MECHANICS from team difficulty (like the Mt Moon proof).

RUN: WATCH=1 INVINCIBLE=0 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_misty.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "1") == "1"
INVINCIBLE = os.getenv("INVINCIBLE", "0") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
CERULEAN = (3, 3)
PC_DOOR = (22, 19)
GYM_DOOR = (31, 21)
MISTY_FRONT = (8, 7)        # tile directly below Misty (8,6); face UP to engage
CASCADE_FLAG = 0x821        # FLAG_BADGE02_GET (consecutive after Boulder 0x820)


def main():
    b = Bridge(ROM)
    with open(os.path.join(STATES, os.getenv("START", "cerulean_caught") + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write

    screen = win = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - Misty / Cascade Badge (watch)")

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
        if INVINCIBLE and st.in_battle(b):
            w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))

    def has_cascade():
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
        fa = sb1 + 0x0EE0 + (CASCADE_FLAG >> 3)
        return bool(b.rd8(fa) & (1 << (CASCADE_FLAG & 7)))

    def fr():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                           log=lambda m: print(f"      {m}", flush=True)).run(120)
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=render)
    camp._suppress_heal = True

    def step(s):
        print(f"\n=== {s} ===", flush=True)

    # 1) back to Cerulean
    step("travel to Cerulean City")
    if tv.map_id(b) != CERULEAN:
        camp.trav.travel(target_map=CERULEAN, edge="east", max_steps=400, max_seconds=240)
    print(f"   on {tv.map_id(b)}@{tv.coords(b)}", flush=True)

    # 2) heal at the Cerulean Center (generalized)
    step("heal at the Cerulean Pokemon Center")
    if tv.map_id(b) == CERULEAN:
        print(f"   heal result: {camp.heal_at_center(pc_door=PC_DOOR)}", flush=True)

    # 3) enter the gym
    step("enter the Cerulean Gym")
    if tv.map_id(b) == CERULEAN:
        camp.trav.travel(target_map=None, arrive_coord=(GYM_DOOR[0], GYM_DOOR[1] + 1),
                         max_steps=400, max_seconds=120)
        before = tv.map_id(b)
        for _ in range(6):
            b.press("UP", 8, 10, render, owner="agent")
            for _ in range(20):
                b.run_frame()
            if tv.map_id(b) != before:
                break
    gym_map = tv.map_id(b)
    print(f"   inside gym map={gym_map}@{tv.coords(b)}", flush=True)
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")

    # 4) navigate to Misty, RESOLVING gym-trainer battles first (Luis/Diana engage via LoS on
    # arrival -> we must FIGHT them out, not mistake the leftover battle for Misty), then re-nav.
    step("navigate to Misty (resolve gym trainers en route)")
    for attempt in range(6):
        camp.trav.travel(target_map=None, arrive_coord=MISTY_FRONT, max_steps=300, max_seconds=150)
        if st.in_battle(b):
            print(f"   gym-trainer battle -> {fr()}", flush=True)
            for _ in range(40):                          # let the post-battle dialogue settle
                b.run_frame()
            b.set_input_owner("agent")
            continue
        break
    print(f"   at {tv.coords(b)} (Misty front {MISTY_FRONT}) in_battle={st.in_battle(b)}", flush=True)
    # NOW engage MISTY herself (talk: face up + A) - only after all trainers are cleared
    for _ in range(24):
        if st.in_battle(b):
            break
        b.press("UP", 8, 8, render, owner="agent")
        b.press("A", 8, 8, render, owner="agent")
        for _ in range(18):
            b.run_frame()

    # 5) fight Misty (current team/moves)
    step("fight Misty")
    print(f"   DIAG: in_battle={st.in_battle(b)} at {tv.coords(b)} before fight", flush=True)
    if st.in_battle(b):
        rb = st.read_battle(b)
        if rb:
            print(f"   DIAG: opponent species={rb['enemy']['species']} "
                  f"({st.SPECIES_NAME.get(rb['enemy']['species'],'?')}) hp={rb['enemy']['hp']}/{rb['enemy']['maxhp']} "
                  f"(Staryu=120 Starmie=121 -> Misty; else a gym trainer)", flush=True)
        print(f"   MISTY battle -> {fr()}", flush=True)
        rb2 = st.read_battle(b)
        print(f"   DIAG: after fight in_battle={st.in_battle(b)} "
              f"enemy={(rb2['enemy']['species'], rb2['enemy']['hp']) if rb2 and rb2.get('enemy') else 'n/a'}", flush=True)

    # 6) drain award until the Cascade flag sets
    step("drain award -> Cascade Badge")
    got = False
    for k in range(160):
        b.press("A", 8, 8, render, owner="agent")
        for _ in range(16):
            b.run_frame()
        if has_cascade():
            got = True
            print(f"   badge flag set at award-press {k}", flush=True)
            break
    if got:
        for _ in range(20):
            b.press("A", 8, 8, render, owner="agent")
            for _ in range(12):
                b.run_frame()
    print(f"\n   *** RESULT: Cascade Badge = {has_cascade()}  party_lead_hp="
          f"{b.rd16(ram.GPLAYER_PARTY+0x56)}/{b.rd16(ram.GPLAYER_PARTY+0x58)}  "
          f"map={tv.map_id(b)} ***", flush=True)
    if has_cascade():
        with open(os.path.join(STATES, "misty_done.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print("   *** CASCADE BADGE obtained - banked misty_done.state ***", flush=True)

    if WATCH:
        import pygame
        clock = pygame.time.Clock()
        print("   [hold] window open - close to exit", flush=True)
        try:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                b.run_frame()
                surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
                screen.blit(pygame.transform.scale(surf, win), (0, 0)); pygame.display.flip()
                clock.tick(60)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
