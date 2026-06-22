"""m1_battle.py - M1 harness: test the HANDS in isolation, NO Kira.

Loads a battle savestate (states/battle.state), opens the window, DUMPS the battle
state so you can verify/correct the CANDIDATE offsets, then runs the battle agent -
printing the NEUTRAL event summaries it would send to Kira's _pokemon_react seam.

How to make states/battle.state:
  1. .venv\\Scripts\\python.exe pokemon_agent\\m0_sandbox.py   (drive into a trainer/gym
     battle with the keyboard: arrows/Z/X), then close the window. M0 auto-saves
     states/overworld.state; for a BATTLE state, use --save-on-quit below instead.

Run:
  .venv\\Scripts\\python.exe pokemon_agent\\m1_battle.py            (load states/battle.state)
  .venv\\Scripts\\python.exe pokemon_agent\\m1_battle.py --capture   (drive manually, S = save battle.state)
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge          # noqa: E402
import pokemon_state as st         # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = os.path.join(_HERE, "states", "battle.state")
SCALE = 3
KEYMAP_DEF = {"UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT",
              "z": "A", "x": "B", "RETURN": "START", "BACKSPACE": "SELECT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="drive manually; S saves battle.state")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame
    pygame.init()
    bridge = Bridge(ROM)
    print(f"   [M1] loaded {bridge.game_code} {bridge.game_title!r}")

    win = (bridge.width * SCALE, bridge.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("M1 - battle harness (Z/X/arrows; S=save state)")
    keymap = {getattr(pygame, f"K_{k}"): v for k, v in KEYMAP_DEF.items()}

    def render():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                os.makedirs(os.path.dirname(STATE), exist_ok=True)
                with open(STATE, "wb") as f:
                    f.write(bytes(bridge.save_state()))
                print(f"   [M1] saved battle state -> {STATE}")
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    try:
        if args.capture:
            print("   [M1] CAPTURE mode: drive into a battle, press S to save battle.state, close window")
            import time
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                pressed = pygame.key.get_pressed() if not args.headless else None
                keys = [v for k, v in keymap.items() if pressed and pressed[k]] if pressed else []
                bridge.set_keys(*keys) if keys else bridge.release()
                bridge.run_frame(); render()
            return

        if not os.path.exists(STATE):
            print(f"   [M1] FAIL - no battle savestate at {STATE}. Run with --capture first.")
            return
        with open(STATE, "rb") as f:
            bridge.load_state(f.read())
        for _ in range(4):
            bridge.run_frame(); render()

        print("   [M1] --- VERIFY CANDIDATE BATTLE OFFSETS ---")
        st.dump_battle_state(bridge)
        if not st.in_battle(bridge):
            print("   [M1] WARNING - savestate is not in a battle (or GBATTLE_TYPE_FLAGS offset wrong)")

        events = []
        agent = BattleAgent(bridge, on_event=lambda s, **k: (events.append(s), print(f"   [EVENT->Kira] {s!r}")),
                            render=render)
        outcome = agent.run(max_seconds=args.seconds)
        print("\n" + "=" * 60)
        print("   M1 RESULT")
        print("=" * 60)
        print(f"   state-read works ......... {'y' if st.in_battle(bridge) or events else 'n (verify offsets)'}")
        print(f"   events emitted ........... {events}")
        print(f"   battle outcome ........... {outcome}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("   [M1] window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
