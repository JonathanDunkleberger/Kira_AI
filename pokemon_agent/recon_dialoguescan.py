"""recon_dialoguescan.py - READ-MOSTLY ground truth for the general dialogue-advance primitive.

Load misty_done (post-badge, in the Cerulean gym). Talk to Misty again and log the gStringVar3
buffer + player coords across the WHOLE open->advance->close cycle, so we learn:
  (a) does the buffer go EMPTY when the box closes, or stay STALE? (box-closed detector)
  (b) how do pages change as we press A? (advance detection)
  (c) when can the player move again? (overworld-regained detector via a coord probe)

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_dialoguescan.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from dialogue_reader import DialogueReader  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")


def main():
    b = Bridge(ROM)
    with open(os.path.join(STATES, "misty_done.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    dr = DialogueReader(b)
    print(f"start coords={tv.coords(b)} buffer={dr._read_buffer()[:60]!r}", flush=True)

    def snap(tag):
        print(f"   {tag:14} coords={tv.coords(b)} buf={dr._read_buffer()[:64]!r}", flush=True)

    # 1) face Misty (up) and talk
    for d in ("UP",):
        b.press(d, 8, 8, owner="agent")
    for _ in range(10):
        b.run_frame()
    b.press("A", 6, 12, owner="agent")
    for _ in range(20):
        b.run_frame()
    snap("after-talk-A")

    # 2) press A repeatedly, logging each page transition
    for k in range(24):
        b.press("A", 6, 12, owner="agent")
        for _ in range(14):
            b.run_frame()
        snap(f"A#{k}")

    # 3) box should be closed now - is the buffer empty or stale? can we MOVE?
    snap("post-close")
    c0 = tv.coords(b)
    b.press("DOWN", 8, 8, owner="agent")
    for _ in range(8):
        b.run_frame()
    c1 = tv.coords(b)
    print(f"   MOVE PROBE: {c0} -> {c1}  (moved={c0 != c1})  buf={dr._read_buffer()[:40]!r}", flush=True)


if __name__ == "__main__":
    main()
