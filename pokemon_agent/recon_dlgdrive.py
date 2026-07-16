"""recon_dlgdrive.py - prove the general dialogue primitive. Load misty_done (frozen mid badge/TM
cutscene), DRIVE the dialogue to a clean overworld close, and verify the player can then MOVE.
Also logs box_open + buffer each step so the pacing/close detection is visible.

RUN: WATCH=1 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_dlgdrive.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "0") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                       # noqa: E402
import travel as tv                             # noqa: E402
from dialogue_drive import DialogueDriver, box_open  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
START = os.getenv("START", "misty_done")


def main():
    b = Bridge(ROM)
    with open(os.path.join(STATES, START + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")

    # REAL STREAM CONDITIONS so pacing is judged at TRUE speed: the emulator free-runs ~10-15x
    # real-time headless, so AUDIO=1 starts the SAME AudioPump play_live uses - its blocking write
    # throttles the emulator to real-time 60fps AND routes game music to the headphones. Without
    # audio we cap the window to 60fps with the pygame clock so visuals are still real-time.
    AUDIO = os.getenv("AUDIO", "0") == "1"
    PHONES = os.getenv("POKEMON_PHONES", "Leviathan")
    audio = None
    if AUDIO:
        try:
            import pokemon_audio
            audio = pokemon_audio.AudioPump(b, phones=PHONES, log=print)
            print(f"   [demo] AudioPump ON (phones~{PHONES!r}) - real-time + music", flush=True)
        except Exception as e:
            print(f"   [demo] !! AudioPump failed ({e}) - falling back to 60fps cap, SILENT", flush=True)

    screen = win = clock = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - dialogue pacing (watch, REAL-TIME)")
        clock = pygame.time.Clock()

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
            if audio is None:                 # audio's blocking write already paces real-time;
                clock.tick(60)                # otherwise cap the loop to true 60fps ourselves

    dd = DialogueDriver(b, render=render, log=lambda m: print(m, flush=True))
    print(f"start coords={tv.coords(b)} box_open={box_open(b)}", flush=True)
    res = dd.drive(label="misty-award")
    print(f"\n   drive -> {res}  box_open={box_open(b)} coords={tv.coords(b)}", flush=True)

    # prove overworld control is back: try to step
    c0 = tv.coords(b)
    moved = None
    for d in ("DOWN", "LEFT", "RIGHT", "UP"):
        b.press(d, 8, 8, render, owner="agent")
        for _ in range(8):
            b.run_frame()
            render()
        if tv.coords(b) != c0:
            moved = (d, tv.coords(b))
            break
    print(f"   MOVE PROBE from {c0}: moved={moved}", flush=True)
    if res == "closed" and moved:
        with open(os.path.join(STATES, "misty_overworld.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print("   *** banked misty_overworld.state (clean post-badge overworld) ***", flush=True)
    if audio is not None:
        try:
            audio.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
