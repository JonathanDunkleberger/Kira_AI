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
import navigate as nav             # noqa: E402
import firered_ram as ram          # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = os.path.join(_HERE, "states", "battle.state")
BOOT_STATE = os.path.join(_HERE, "states", "after_pick_bulbasaur.state")  # capture starts here
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
            # Boot from the post-pick Bulbasaur overworld (NOT a fresh intro) so you can
            # walk straight to grass / the first trainer. No masher exists here.
            if not os.path.exists(BOOT_STATE):
                print(f"   [M1] FAIL - boot state missing: {BOOT_STATE}"); return
            with open(BOOT_STATE, "rb") as f:
                bridge.load_state(f.read())
            for _ in range(40):
                bridge.run_frame()
            print(f"   [M1] booted {os.path.basename(BOOT_STATE)} @ {nav.coords(bridge)} "
                  f"party={bridge.rd8(0x02024029)}")
            bridge.release()
            bridge.set_input_owner("human")      # single owner; phantoms dropped+logged
            print("   [M1] HANDED OFF - only your keyboard reaches the emulator. Drive "
                  "(arrows/Z/X) into TALL GRASS for a wild battle (or the first trainer).")
            print("   [M1] PHASE CAPTURE - save a state at each battle phase you can catch:")
            print("        1=battle_menu (FIGHT/BAG/POKEMON/RUN up)  2=battle_movelist (4-move list)")
            print("        3=battle_animation (mid-move)  4=battle_damagetext (text scrolling)  "
                  "5=battle_enemyturn (enemy attacking)")
            print("        S=battle.state   D=live raw dump.   Catch even 3 of 5 - then close.")

            def blit():  # capture's loop owns the event pump; this only draws
                surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                               (bridge.width, bridge.height), "RGB")
                screen.blit(pygame.transform.scale(surf, win), (0, 0))
                pygame.display.flip()

            def save_phase(name):
                path = os.path.join(os.path.dirname(STATE), name + ".state")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(bytes(bridge.save_state()))
                print(f"   [M1] SAVED {name}.state  in_battle={st.in_battle(bridge)}  "
                      f"res_ptr={hex(bridge.rd32(ram.GBATTLE_RES_PTR))}")

            PHASE_KEYS = {pygame.K_1: "battle_menu", pygame.K_2: "battle_movelist",
                          pygame.K_3: "battle_animation", pygame.K_4: "battle_damagetext",
                          pygame.K_5: "battle_enemyturn"}
            import time
            WFL = getattr(pygame, "WINDOWFOCUSLOST", None)
            # Event-tracked held keys (NOT get_pressed): immune to the launch-Enter stuck
            # key (its KEYDOWN predates pygame, so it never enters `held`). Cleared on
            # focus loss so a lost KEYUP can't strand a key either.
            held = set()
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if (WFL is not None and ev.type == WFL) or \
                       (ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0):
                        if held:
                            held.clear(); print("   [M1] focus lost -> held keys cleared")
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key in PHASE_KEYS:
                            save_phase(PHASE_KEYS[ev.key])
                        elif ev.key == pygame.K_s:
                            save_phase(os.path.splitext(os.path.basename(STATE))[0])
                        elif ev.key == pygame.K_d:
                            st.dump_battle_state(bridge)
                        elif ev.key in keymap:
                            held.add(keymap[ev.key])      # only real GBA keys go to the pad
                    elif ev.type == pygame.KEYUP and ev.key in keymap:
                        held.discard(keymap[ev.key])
                bridge.set_keys(*held, owner="human") if held else bridge.release(owner="human")
                bridge.run_frame(); blit()
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
