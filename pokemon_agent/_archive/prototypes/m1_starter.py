"""m1_starter.py - M1 (retargeted): the STARTER PICK. STANDALONE harness, no Kira.

The milestone's SOUL is the self-driven CHOICE. Standalone this uses a STUB choice
(--pick or her stated default); the LIVE run binds choose_starter_fn =
bot._pokemon_choose_starter so her real self picks. on_event prints the NEUTRAL
events that route to bot._pokemon_react in the live path.

TWO STEPS (honest - the lab route + ball tiles are NOT assumed):

1) CAPTURE (you do this once, ~2 min) - PER-BALL prompt-ready savestates:
   .venv\\Scripts\\python.exe pokemon_agent\\m1_starter.py --capture
   Boots + auto-clears the intro, then YOU drive (arrows/Z/X) from the bedroom to
   Oak's lab and up to the table. At EACH ball: face it, press A until the
   "Do you want this POKEMON?" YES/NO menu is OPEN, then press:
       1 = save ball_bulbasaur.state   2 = ball_charmander.state   3 = ball_squirtle.state
   Each save logs YESNO_menu_detected=y/n - 'n' means you grabbed the wrong box;
   redo that ball. These three states are what the autonomous grab loads + confirms.

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
    # MASHER PHASE: the intro auto-masher owns the pad here; the human keyboard is
    # NOT read yet. (Pressing keys now does nothing - that's the masher "fighting"
    # you. Wait for the HANDED OFF line.)
    log("MASHER ACTIVE - auto-clearing intro; your keyboard is IGNORED until 'HANDED OFF'.")
    if not reach_overworld(bridge, render):
        log("FAIL - never reached overworld"); return
    stx.clear_dialogue(bridge, taps=8, render=render)   # clear the wake-up dialogue lock
    # HANDOFF: masher is done. Release any held key, then claim the human as the sole
    # input owner. From here Bridge drops+logs any non-human press (no phantom).
    bridge.release()
    bridge.set_input_owner("human")
    log("HANDED OFF - only your keyboard reaches the emulator now (phantoms dropped+logged).")
    log("PER-BALL CAPTURE: walk to a ball, FACE it, press A until the "
        "'Do you want this POKEMON?' YES/NO menu is OPEN, THEN press:")
    log("   1 = save ball_bulbasaur.state   2 = save ball_charmander.state   "
        "3 = save ball_squirtle.state   (S = save states/<save_as>.state)")
    log("Each save reports YESNO_menu_detected=y/n. If it says n you saved the "
        "WRONG box (a text message, not the YES/NO) - redo that ball.")
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

    WFL = getattr(pygame, "WINDOWFOCUSLOST", None)
    # EDGE-TRIGGERED: `down` = keys physically down (up->down transition detector) so a
    # held/stuck/OS-repeated key fires the 1/2/3/S saves ONCE, never per-frame. `held` =
    # GBA pad keys held. Both cleared on focus loss; immune to the launch-key stuck phantom.
    down, held = set(), set()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
            if (WFL is not None and ev.type == WFL) or \
               (ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0):
                if down or held:
                    down.clear(); held.clear(); log("focus lost -> keys cleared (no stuck-key phantom)")
            elif ev.type == pygame.KEYUP:
                down.discard(ev.key)
                if ev.key in keymap:
                    held.discard(keymap[ev.key])
            elif ev.type == pygame.KEYDOWN:
                if ev.key in down:
                    continue                       # duplicate KEYDOWN (repeat) -> ignore
                down.add(ev.key)
                if ev.key in keymap:
                    held.add(keymap[ev.key])
                if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    # Per-ball prompt-ready save: stand AT the ball with the "Do you
                    # want this POKEMON?" YES/NO menu OPEN, then 1/2/3 saves it.
                    import numpy as np
                    name = stx.BALL_ORDER[ev.key - pygame.K_1]
                    os.makedirs(STATES, exist_ok=True)
                    path = os.path.join(STATES, f"ball_{name}.state")
                    with open(path, "wb") as f:
                        f.write(bytes(bridge.save_state()))
                    # The YES/NO menu is a bright box in the TOP-RIGHT. A plain text box
                    # ("Those are POKE BALLS") has only a bottom box and NO top-right menu.
                    a = np.asarray(bridge.frame_rgb())
                    yesno = bool(a[0:48, 150:240, :].mean() > 120)
                    log(f"SAVED ball_{name}.state @ {nav.coords(bridge)}  "
                        f"YESNO_menu_detected={'y - GOOD' if yesno else 'n - WRONG BOX! you saved a text message, not the YES/NO. Re-do this ball.'}")
                elif ev.key == pygame.K_s:
                    os.makedirs(STATES, exist_ok=True)
                    save_path = os.path.join(STATES, f"{args.save_as}.state")
                    with open(save_path, "wb") as f:
                        f.write(bytes(bridge.save_state()))
                    log(f"SAVED state -> {save_path}")
        bridge.set_keys(*held, owner="human") if held else bridge.release(owner="human")
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
    ap.add_argument("--save-as", default="starter",
                    help="basename for S-saved state (e.g. 'after_pick' -> states/after_pick.state)")
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
