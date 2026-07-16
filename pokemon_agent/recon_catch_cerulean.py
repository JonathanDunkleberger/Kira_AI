"""recon_catch_cerulean.py - VISIBLE autonomous catch near Cerulean (Batch 1 #2). From
cerulean_entered_auto she walks WEST into Route 4 grass, triggers a wild encounter, WEAKENS it
SAFELY with a status move (Sleep Powder/PoisonPowder - no KO risk; her L19 would OHKO with damage),
then THROWS Poke Balls until caught. Behavioral control: party 1 -> 2. No HP injection - real catch.
RUN (visible): WATCH=1 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_catch_cerulean.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "1") == "1"
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
OFF = tv.MAP_OFFSET


def main():
    start = os.getenv("START", "cerulean_entered_auto")
    b = Bridge(ROM)
    with open(os.path.join(STATES, start + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")

    screen = win = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - autonomous catch (watch)")

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()

    p_start = ram.read_party_count(b)
    print(f"   [catch] start map={tv.map_id(b)} coords={tv.coords(b)} party={p_start}", flush=True)

    def _use_slot(ag, slot):
        """Select a SPECIFIC move slot (for the safe status weaken), reusing the engine's nav."""
        ag._home_to_fight()
        for _ in range(8):
            if ag._in_move_list():
                break
            b.press("A", ag.hold, ag.hold, render, owner="agent"); ag._wait(10)
        ag._nav_move(slot)
        b.press("A", ag.hold, ag.hold, render, owner="agent"); ag._wait(40)

    def catch_runner():
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                         log=lambda m: print(f"      {m}", flush=True))
        t0 = time.time()
        ag._reach_first_menu(t0, 60)
        if ag._is_trainer_battle():
            print("      [catch] trainer battle (can't catch) - winning it", flush=True)
            return "win" if ag.run(90) != "loss" else "loss"
        # read foe + our moves, RETRYING (read_battle is None mid-transition -> the earlier
        # weaken-skip bug: it read None right after the intro and never weakened).
        rb = None
        for _ in range(20):
            rb = st.read_battle(b)
            if rb and rb.get("ours") and rb.get("enemy"):
                break
            ag._wait(4)
        foe = st.SPECIES_NAME.get(rb["enemy"]["species"], "?") if rb else "?"
        print(f"      [catch] WILD {foe} - COMMIT: weaken (status, no KO) then throw till caught", flush=True)
        # SAFE WEAKEN: one status move (power 0), prefer SLEEP (best catch boost, zero KO risk -
        # her L19 attacks would OHKO these Route-4 weaklings, so we never use a damaging move).
        if rb:
            moves = rb["ours"]["moves"]
            print(f"      [catch] her moves: {[(m.get('name'), m.get('power'), m.get('pp')) for m in moves]}", flush=True)
            # select the SLEEP move by NAME (don't gate on the pp read - it's reporting 0 for status
            # moves even when usable). Sleep also stops Spearow/Rattata FLEEING so we can commit.
            status = [i for i, m in enumerate(moves) if m.get("id", 0) and m.get("power", 0) == 0]
            sleep = [i for i in status if "sleep" in str(moves[i].get("name", "")).lower()]
            slot = (sleep or status or [None])[0]
            if slot is not None:
                print(f"      [catch] status-weaken with slot {slot} ({moves[slot].get('name','?')})", flush=True)
                _use_slot(ag, slot)
            else:
                print("      [catch] no status move available - throwing at full HP", flush=True)
        # COMMIT: throw at the SAME wild until caught / it leaves / out of balls (throw_ball no
        # longer flees on a break, so each broke_free returns us to the menu to throw AGAIN).
        for n in range(12):
            ag._settle()                              # let the turn fully resolve BEFORE judging the
            if ram.read_party_count(b) > p_start:     # battle state (a mid-animation in_battle read
                return "caught"                       # was a false 'wild left' right after the weaken)
            if not st.in_battle(b):
                print("      [catch] the wild left (fled/fainted) - moving on", flush=True)
                return "broke_free"
            r = ag.throw_ball(max_seconds=45)
            print(f"      [catch] throw {n + 1} -> {r}", flush=True)
            if r in ("caught", "no_balls", "trainer"):
                return r
        return "stuck"

    camp = Campaign(b, battle_runner=catch_runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=render)
    camp._suppress_heal = True

    # 1) get onto Route 4 (west of Cerulean), then walk into GRASS to trigger a wild encounter
    print("   [catch] heading WEST to Route 4 grass...", flush=True)
    camp.trav.travel(target_map=(3, 22), edge="west", max_steps=200, max_seconds=120)
    print(f"   [catch] now map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)

    # 2) walk into grass and wiggle until an encounter (catch_runner handles it)
    caught = False
    for attempt in range(400):
        if ram.read_party_count(b) > p_start:
            caught = True; break
        if st.in_battle(b):                           # encounter -> catch
            catch_runner()
            if ram.read_party_count(b) > p_start:
                caught = True; break
            continue
        grid = tv.Grid(b); here = tv.coords(b)
        # nearest grass tile (save coords), BFS, walk onto it; then wiggle within grass
        grass = [(bx - OFF, by - OFF) for (bx, by) in grid.grass]
        grass = [g for g in grass if grid.walkable(*g)]
        if not grass:
            print("   [catch] no grass on this map - stepping west to find some", flush=True)
            b.press("LEFT", 8, 8, render, owner="agent"); continue
        target = min(grass, key=lambda g: abs(g[0] - here[0]) + abs(g[1] - here[1]))
        if here not in grass:
            camp.trav.travel(target_map=None, arrive_coord=target, max_steps=80, max_seconds=40)
        # wiggle in grass to roll encounters
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            b.press(d, 8, 8, render, owner="agent")
            if st.in_battle(b):
                break

    p_end = ram.read_party_count(b)
    print(f"\n   *** CATCH RESULT: party {p_start} -> {p_end}  ({'CAUGHT party+1' if caught else 'NOT caught'}) ***", flush=True)
    if caught:
        with open(os.path.join(STATES, "cerulean_caught.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print("   *** banked cerulean_caught.state (party grew) ***", flush=True)
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
