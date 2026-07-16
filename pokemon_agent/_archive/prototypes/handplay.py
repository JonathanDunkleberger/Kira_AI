"""handplay.py - throwaway MANUAL hand-play utility (no agent, no Kira, no masher).

Boots FireRed in a visible window from the canonical post-pick state so you start
with Bulbasaur in the overworld, lets YOU drive with the keyboard, and saves the
emulator state to a fixed path on F5 - so you can hand-play the Oak's Parcel quest
and bank pokemon_agent/states/viridian_parcel_done.state for the campaign harness.

Same input firewall as session.py: EXACTLY ONE owner ("human"), event-tracked held
keys (immune to the launch-Enter stuck key), held keys cleared on focus loss. NO
agent, NO auto-masher, NO reaction/voice path - pure manual play.

RUN:  .venv\\Scripts\\python.exe pokemon_agent\\handplay.py [--boot X.state] [--save Y.state]
KEYS: arrows = move | Z = A | X = B | Enter = START | Backspace = SELECT
SAVE: F5  -> writes states/<--save> (default viridian_parcel_done.state; LOUD confirm w/ full path)
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
# canonical post-pick boot state (Bulbasaur picked, in the overworld)
BOOT_STATE = os.path.join(STATES, "after_pick_bulbasaur.state")
SAVE_PATH = os.path.join(STATES, "viridian_parcel_done.state")
SCALE = 3


def log(m):
    print(f"   [handplay] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", default=BOOT_STATE, help="boot savestate (name in states/ or a path)")
    ap.add_argument("--save", default=SAVE_PATH, help="F5 target savestate (name in states/ or a path)")
    args = ap.parse_args()
    boot_state = args.boot if os.path.isabs(args.boot) else os.path.join(STATES, args.boot)
    save_path = args.save if os.path.isabs(args.save) else os.path.join(STATES, args.save)

    import pygame
    pygame.init()
    b = Bridge(ROM)
    log(f"loaded {b.game_code} {b.game_title!r}")
    if not os.path.exists(boot_state):
        log(f"FAIL - boot state missing: {boot_state}"); pygame.quit(); return
    with open(boot_state, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    log(f"booted {os.path.basename(boot_state)}: coords={ram.read_player_coords(b)} "
        f"party={ram.read_party_count(b)} map=({b.rd8(sb1 + 4)},{b.rd8(sb1 + 5)})")

    win = (b.width * SCALE, b.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("HAND-PLAY  |  arrows=move Z=A X=B Enter=START  |  F5=SAVE parcel state")
    keymap = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_z: "A", pygame.K_x: "B",
              pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT"}

    def blit():
        surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    def save_state():
        try:
            os.makedirs(STATES, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(bytes(b.save_state()))
            co = ram.read_player_coords(b)
            sb = b.rd32(ram.GSAVEBLOCK1_PTR)
            print("\n" + "=" * 64, flush=True)
            print(f"   [handplay] *** SAVED *** -> {os.path.abspath(save_path)}", flush=True)
            print(f"   [handplay] map=({b.rd8(sb + 4)},{b.rd8(sb + 5)}) coords={co} "
                  f"party={ram.read_party_count(b)}", flush=True)
            print("=" * 64 + "\n", flush=True)
        except Exception as e:
            log(f"!! SAVE FAILED: {e}")

    b.set_input_owner("human")     # Bridge enforces: keyboard is the sole writer
    log("=" * 60)
    log("YOU HAVE CONTROL - drive the Oak's Parcel quest by hand.")
    log("  arrows = move | Z = A | X = B | Enter = START | Backspace = SELECT")
    log(f"  PRESS  F5  TO SAVE  ->  {os.path.abspath(save_path)}")
    log("=" * 60)

    WFL = getattr(pygame, "WINDOWFOCUSLOST", None)
    held = set()      # event-tracked (NOT get_pressed): immune to the launch-Enter stuck key
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if (WFL is not None and ev.type == WFL) or \
                   (ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0):
                    if held:
                        held.clear(); log("focus lost -> held keys cleared (no stuck-key)")
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F5:
                    save_state()
                elif ev.type == pygame.KEYDOWN and ev.key in keymap:
                    held.add(keymap[ev.key])
                elif ev.type == pygame.KEYUP and ev.key in keymap:
                    held.discard(keymap[ev.key])
            b.set_keys(*held, owner="human") if held else b.release(owner="human")
            b.run_frame(); blit()
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
