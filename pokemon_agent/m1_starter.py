"""m1_starter.py - M1 (retargeted): the STARTER PICK. STANDALONE harness, no Kira.

The milestone's SOUL is the self-driven CHOICE. Standalone this uses a STUB choice
(--pick or her stated default); the LIVE run binds choose_starter_fn =
bot._pokemon_choose_starter so her real self picks. on_event prints the NEUTRAL
events that route to bot._pokemon_react in the live path.

TWO STEPS (honest - the lab route + ball tiles are NOT assumed):

1) CAPTURE (you do this once, ~2 min):
   .venv\\Scripts\\python.exe pokemon_agent\\m1_starter.py --capture
   Boots + auto-clears the intro, then YOU drive (arrows/Z/X) from the bedroom to
   Oak's lab and up to the table. At each ball, stand on its tile and press:
       1 = mark Bulbasaur ball   2 = Charmander ball   3 = Squirtle ball
   Then press S to save the savestate. Writes states/starter.state +
   states/starter_waypoints.json. This gives me the route + verifies ball positions.

2) RUN:
   .venv\\Scripts\\python.exe pokemon_agent\\m1_starter.py [--pick bulbasaur]
   Loads the savestate, asks the self (stub here), walks to that ball, selects,
   reacts. Prints the M1 RESULT block.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import navigate as nav           # noqa: E402
import starter as stx            # noqa: E402
import m0_sandbox as m0          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
STATE = os.path.join(STATES, "starter.state")
WPTS = os.path.join(STATES, "starter_waypoints.json")
SCALE = 3


def log(m):
    print(f"   [M1] {m}", flush=True)


def reach_overworld(bridge, render, max_seconds=95):
    sched = m0.masher_keys()
    import time
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        k = next(sched)
        bridge.set_keys(k) if k else bridge.release()
        bridge.run_frame(); render()
        if nav.coords(bridge) is not None:
            return True
    return False


def capture(bridge, render, pygame, args):
    log("CAPTURE: auto-clearing intro...")
    if not reach_overworld(bridge, render):
        log("FAIL - never reached overworld"); return
    stx.clear_dialogue(bridge, taps=8, render=render)   # clear the wake-up dialogue lock
    log("DRIVE to Oak's lab + table. 1/2/3 = mark ball under you, S = save state.")
    wpts = {}
    import time
    keymap = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_z: "A", pygame.K_x: "B",
              pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT"}
    win = pygame.display.get_surface().get_size()

    def blit():  # render WITHOUT draining events (capture's loop owns the event pump)
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        pygame.display.get_surface().blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    name = stx.BALL_ORDER[ev.key - pygame.K_1]
                    wpts[name] = list(nav.coords(bridge) or (0, 0))
                    log(f"marked {name} ball @ {wpts[name]}")
                elif ev.key == pygame.K_s:
                    os.makedirs(STATES, exist_ok=True)
                    with open(STATE, "wb") as f:
                        f.write(bytes(bridge.save_state()))
                    with open(WPTS, "w") as f:
                        json.dump(wpts, f, indent=2)
                    log(f"SAVED {STATE} + waypoints {wpts}")
        pressed = pygame.key.get_pressed()
        keys = [v for k, v in keymap.items() if pressed[k]]
        bridge.set_keys(*keys) if keys else bridge.release()
        bridge.run_frame(); blit()


def run(bridge, render, args):
    result = dict(reached_lab="n", read_screen="n", self_chose="(none)", why="",
                  selected="n", events_through_gate="n (live bot only)")
    if not os.path.exists(STATE) or not os.path.exists(WPTS):
        log(f"FAIL - need a capture first. Run: m1_starter.py --capture")
        print_result(result); return
    with open(STATE, "rb") as f:
        bridge.load_state(f.read())
    wpts = json.load(open(WPTS))
    for _ in range(4):
        bridge.run_frame(); render()
    co = nav.coords(bridge)
    result["reached_lab"] = "y" if co is not None else "n"
    result["read_screen"] = "y" if all(k in wpts for k in stx.BALL_ORDER) else "partial"
    log(f"at table coords={co}; ball waypoints={wpts}")

    # THE CHOICE (stub standalone; live bot uses _pokemon_choose_starter)
    choice = (args.pick or "bulbasaur").lower()
    result["self_chose"] = choice
    result["why"] = "(STUB - standalone; live run uses her real self)"
    log(f"SELF (stub) chose: {choice}")
    if choice not in wpts:
        log(f"FAIL - no captured waypoint for {choice} (have {list(wpts)})")
        print_result(result); return

    events = []
    ok = stx.select_starter(bridge, choice, tuple(wpts[choice]), render=render,
                            on_event=lambda s, **k: (events.append(s), log(f"EVENT->Kira: {s!r}")))
    result["selected"] = "y" if ok else "n"
    log(f"events: {events}")
    print_result(result)


def print_result(r):
    print("\n" + "=" * 60 + "\n   M1 RESULT (starter pick)\n" + "=" * 60)
    print(f"   reached lab/table ........ {r['reached_lab']}")
    print(f"   read 3-ball screen ....... {r['read_screen']}")
    print(f"   self chose ............... {r['self_chose']}  {r['why']}")
    print(f"   selection executed ....... {r['selected']}")
    print(f"   events via _ok_to_self_speak {r['events_through_gate']}")
    print("=" * 60, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--pick", default=None, help="standalone stub choice")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=300.0)
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    bridge = Bridge(ROM)
    log(f"loaded {bridge.game_code} {bridge.game_title!r}")
    win = (bridge.width * SCALE, bridge.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("M1 - Kira's starter pick")

    def render():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    try:
        if args.capture:
            capture(bridge, render, pygame, args)
        else:
            run(bridge, render, args)
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
