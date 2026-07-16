"""m0_sandbox.py - M0: prove the emulator loop. STANDALONE, zero Kira imports.

Run (windowed, watchable - YOU can nudge it with the keyboard if the blind
masher stalls at FireRed's name-entry screen):
    .venv\\Scripts\\python.exe pokemon_agent\\m0_sandbox.py

Run (headless self-check, bounded - no window, auto-mash only, for verification):
    .venv\\Scripts\\python.exe pokemon_agent\\m0_sandbox.py --headless --seconds 40

Window keyboard passthrough: arrows = D-pad, Z = A, X = B, Enter = START,
Backspace = SELECT. Use these to drive to the overworld if needed; the live RAM
readout + coord-change detection then confirm the offsets.

Proves: (1) bridge loads FireRed, (2) pygame renders the live framebuffer,
(3) auto-mash advances the intro, (4) reads FireRed RAM (party count + player
coords) live, (5) a button press changes state (coords move). Prints a M0 RESULT
block. Loud PASS/FAIL throughout. Honest: only claims overworld on REAL coords.
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge          # noqa: E402
import firered_ram as ram          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCALE = 3
EXPECT_CODE = "AGB-BPRE"


def log(tag, msg):
    print(f"   [M0] {tag}: {msg}", flush=True)


def screen_arr(bridge):
    return np.asarray(bridge.frame_rgb(), dtype=np.int16)


def real_coords(bridge):
    """In-game player coords, or None. Honest: rejects the boot state where the
    SaveBlock1 pointer is valid but points at zeroed memory (coords == (0,0))."""
    co = ram.read_player_coords(bridge)
    if co is None or co == (0, 0) or not (0 <= co[0] < 1000 and 0 <= co[1] < 1000):
        return None
    return co


# ── auto-masher: per-frame key schedule to advance the intro ─────────────────
def masher_keys():
    """Generator yielding a key name (held this frame) or None. Mostly A, a START
    every so often, and a DOWN+A burst to escape the name-entry menu by picking a
    preset name. Blind - if it stalls, the human can take over via the keyboard."""
    cycle = 0
    while True:
        cycle += 1
        if cycle % 7 == 0:                 # periodic START (title / confirmations)
            for _ in range(8):
                yield "START"
            for _ in range(8):
                yield None
        elif cycle % 5 == 0:               # name-screen escape: DOWN then A (pick preset)
            for _ in range(8):
                yield "DOWN"
            for _ in range(6):
                yield None
            for _ in range(8):
                yield "A"
            for _ in range(8):
                yield None
        else:                              # default: advance text with A
            for _ in range(8):
                yield "A"
            for _ in range(8):
                yield None


KEYMAP = {}   # pygame key -> core key name (filled after pygame import)


def human_keys(pygame):
    pressed = pygame.key.get_pressed()
    out = [name for k, name in KEYMAP.items() if pressed[k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame
    pygame.init()
    KEYMAP.update({
        pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT",
        pygame.K_z: "A", pygame.K_x: "B", pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT",
    })

    result = dict(bridge_loads=False, window=False, in_game=False, hold_frames=None,
                  party_addr=hex(ram.GPLAYER_PARTY_CNT), party_confirmed=False,
                  coord_addr=f"*{hex(ram.GSAVEBLOCK1_PTR)}+{hex(ram.SB1_OFF_POS_X)}",
                  coord_confirmed=False, button_changes_state=False)

    # (1) load + assert
    try:
        bridge = Bridge(ROM)
        result["bridge_loads"] = True
        log("LOAD", f"game_code={bridge.game_code!r} title={bridge.game_title!r}")
        log("LOAD", "PASS - FireRed USA (AGB-BPRE)" if bridge.game_code == EXPECT_CODE
            else f"FAIL - expected {EXPECT_CODE}, got {bridge.game_code!r}")
    except Exception as e:
        log("LOAD", f"FAIL - {type(e).__name__}: {e}")
        print_result(result)
        return

    win_w, win_h = bridge.width * SCALE, bridge.height * SCALE
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("M0 - Kira plays FireRed (arrows/Z/X/Enter to drive)")
    result["window"] = True
    clock = pygame.time.Clock()

    def render():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, (win_w, win_h)), (0, 0))
        pygame.display.flip()

    try:
        # warm up past boot logos so calibration sees the title
        for _ in range(360):
            bridge.run_frame()
            render()

        # (3a) calibrate hold-frames: smallest START hold that transitions the title
        log("CALIBRATE", "finding reliable hold-frame count (START at title)...")
        hold = None
        for h in (2, 4, 6, 8, 10, 12):
            before = screen_arr(bridge)
            bridge.set_keys("START")
            for _ in range(h):
                bridge.run_frame(); render()
            bridge.release()
            for _ in range(24):
                bridge.run_frame(); render()
            d = float(np.abs(screen_arr(bridge) - before).mean())
            log("CALIBRATE", f"hold={h:>2} -> screen d={d:6.2f}")
            if d > 6.0:
                hold = h
                break
        if hold is None:
            hold = 8
            log("CALIBRATE", f"no clear transition; defaulting hold={hold}")
        else:
            log("CALIBRATE", f"CONFIRMED hold_frames={hold}")
        result["hold_frames"] = hold

        # (3b/4/5) unified loop: auto-mash until human takes over; live RAM readout;
        # coord-change detection proves button-changes-state.
        log("RUN", "auto-mashing toward in-game (take over with arrows/Z/X/Enter if it stalls)...")
        sched = masher_keys()
        t0 = last = time.time()
        first_ingame_coord = None
        while time.time() - t0 < args.seconds:
            hk = [] if args.headless else human_keys(pygame)
            if hk:                                  # human override
                bridge.set_keys(*hk)
            else:
                k = next(sched)
                bridge.set_keys(k) if k else bridge.release()
            bridge.run_frame()
            render()
            if not args.headless:
                clock.tick(120)                     # keep it watchable, not max-speed

            co = real_coords(bridge)
            now = time.time()
            if now - last >= 1.0:
                last = now
                pc = ram.read_party_count(bridge)
                sb1 = bridge.rd32(ram.GSAVEBLOCK1_PTR)
                raw = ram.read_player_coords(bridge)
                log("RAM", f"frame={bridge.frame} party={pc} SB1={hex(sb1)} "
                           f"coords_raw={raw} in_game={co is not None}")
            if co is not None:
                if not result["in_game"]:
                    result["in_game"] = True
                    first_ingame_coord = co
                    log("RUN", f"OVERWORLD confirmed: coords={co}")
                elif first_ingame_coord is not None and co != first_ingame_coord:
                    if not result["button_changes_state"]:
                        result["button_changes_state"] = True
                        result["coord_confirmed"] = True
                        log("BTN", f"PASS - coords changed {first_ingame_coord} -> {co} "
                                   f"(button/movement changes state; coord offset CONFIRMED)")
                    # one-time: auto-prove with a scripted DOWN burst then stop driving
        # party-count plausibility
        pc = ram.read_party_count(bridge)
        result["party_confirmed"] = (0 <= pc <= 6)
        log("RAM", f"final party_count {hex(ram.GPLAYER_PARTY_CNT)}={pc} "
                   f"({'plausible 0..6' if result['party_confirmed'] else 'OUT OF RANGE - suspect'})")

        if result["in_game"] and not result["button_changes_state"]:
            log("BTN", "auto DOWN burst to prove coord movement...")
            base = real_coords(bridge)
            for _ in range(6):
                for _ in range(hold):
                    bridge.set_keys("DOWN"); bridge.run_frame(); render()
                for _ in range(hold):
                    bridge.release(); bridge.run_frame(); render()
            after = real_coords(bridge)
            if base and after and after != base:
                result["button_changes_state"] = result["coord_confirmed"] = True
                log("BTN", f"PASS - {base} -> {after}")
            else:
                log("BTN", f"FAIL - coords static ({base} -> {after}); offset suspect or blocked")

        if result["in_game"]:
            try:
                os.makedirs(os.path.join(_HERE, "states"), exist_ok=True)
                with open(os.path.join(_HERE, "states", "overworld.state"), "wb") as f:
                    f.write(bytes(bridge.save_state()))
                log("STATE", "saved states/overworld.state for reuse")
            except Exception as e:
                log("STATE", f"savestate skipped: {e}")

    except KeyboardInterrupt:
        log("RUN", "window closed / interrupted")
    finally:
        pygame.quit()
    print_result(result)


def print_result(r):
    print("\n" + "=" * 60)
    print("   M0 RESULT")
    print("=" * 60)
    print(f"   bridge loads ............. {'y' if r['bridge_loads'] else 'n'}")
    print(f"   window renders ........... {'y' if r['window'] else 'n'}")
    print(f"   reached in-game .......... {'y' if r['in_game'] else 'n'}")
    print(f"   hold-frames found ........ {r['hold_frames']}")
    print(f"   party-count offset ....... {'y' if r['party_confirmed'] else 'n'}  ({r['party_addr']})")
    print(f"   coord offset confirmed ... {'y' if r['coord_confirmed'] else 'n'}  ({r['coord_addr']})")
    print(f"   button changes state ..... {'y' if r['button_changes_state'] else 'n'}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
