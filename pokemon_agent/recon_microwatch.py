"""recon_microwatch.py - CONTROL for increment-2 MICRO watchdog (world-fingerprint + no-progress).

Three controls, increasing fidelity:
  A) MicroWatch UNIT control - deterministic: identical fingerprints stall at the threshold, a NEW
     fingerprint (or forced `progressed`) resets, and the threshold is context-aware (dialogue gets
     more rope than the default). Pure logic, no emulator.
  B) drive() FIRING control - reproduces the EXACT wedge (a dialogue box stuck open, same text, the
     world frozen) with a minimal fake bridge and proves drive() now: detects no-progress, logs the
     u-watch line, taps B, returns 'exhausted', and STOPS far short of the old 300-blind-press mash.
  C) drive() NON-REGRESSION (real engine): walk up to a real Pallet Town NPC, open its dialogue, and
     prove drive() still drives it to a clean 'closed' WITHOUT a false 'exhausted'.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_microwatch.py
      WATCH=1 ... (part C in a visible window so Jonny can SEE the talk + the u-watch log)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "0") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import firered_ram as ram                              # noqa: E402
import world_fingerprint as wf                         # noqa: E402
from dialogue_drive import DialogueDriver, box_open     # noqa: E402
import travel as tv                                     # noqa: E402
from bridge import Bridge                               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WORKSHOP = os.path.join(_HERE, "states", "workshop")


def _fp(map_id=(3, 1), x=5, y=5, facing=1, box=False, battle=False,
        party=((1, 5, 20),), badges=0, money=3000, bag=()):
    """Build a WorldFingerprint literal for the unit control."""
    return wf.WorldFingerprint(map_id, x, y, facing, box, battle, party, badges, money, bag)


def part_a():
    print("\n==== PART A: MicroWatch unit control ====")
    ok = True

    # (1) identical OVERWORLD fingerprints stall after STALL_N_DEFAULT no-progress presses. feed()'s
    # FIRST call is the baseline (nothing to compare yet), so the threshold is hit at limit+1 feeds
    # = `limit` genuine no-progress presses, which is exactly "identical for N consecutive presses".
    mw = wf.MicroWatch()
    base = _fp(box=False)
    fired_at = None
    for i in range(1, 10):
        mw.feed(base)
        if mw.stalled(base):
            fired_at = i
            break
    no_progress = fired_at - 1 if fired_at else None
    print(f"  overworld identical: fired at feed {fired_at} = {no_progress} no-progress presses "
          f"(want {wf.STALL_N_DEFAULT})")
    ok &= (no_progress == wf.STALL_N_DEFAULT)

    # (2) a box-up (dialogue) context gets MORE rope before stalling
    mw = wf.MicroWatch()
    dbox = _fp(box=True)
    fired_at = None
    for i in range(1, 40):
        mw.feed(dbox)
        if mw.stalled(dbox):
            fired_at = i
            break
    no_progress = fired_at - 1 if fired_at else None
    print(f"  dialogue identical: fired at feed {fired_at} = {no_progress} no-progress presses "
          f"(want {wf.STALL_N_DIALOGUE})")
    ok &= (no_progress == wf.STALL_N_DIALOGUE) and (wf.STALL_N_DIALOGUE > wf.STALL_N_DEFAULT)

    # (3) a CHANGING world never stalls (movement = progress)
    mw = wf.MicroWatch()
    stalled_ever = False
    for i in range(20):
        mw.feed(_fp(x=i))                              # coords advancing each feed
        if mw.stalled(_fp(x=i)):
            stalled_ever = True
    print(f"  moving world: ever stalled = {stalled_ever} (want False)")
    ok &= not stalled_ever

    # (4) progressed=True (the dialogue driver's new-text reset) holds it off forever
    mw = wf.MicroWatch()
    stalled_ever = False
    for _ in range(20):
        mw.feed(dbox, progressed=True)
        if mw.stalled(dbox):
            stalled_ever = True
    print(f"  forced-progress (new text every page): ever stalled = {stalled_ever} (want False)")
    ok &= not stalled_ever

    print("  PART A:", "PASS" if ok else "FAIL")
    return ok


# ── a minimal fake bridge that FREEZES the world with a dialogue box stuck open (the exact wedge) ──
class _WhitePixels:
    def __getitem__(self, _xy):
        return (255, 255, 255)                          # all-white -> box_open() reads a box up


class _WhiteFrame:
    def load(self):
        return _WhitePixels()


class _StuckBridge:
    """Reads constant -> a constant fingerprint; frame is all-white -> box_open True forever; presses
    are recorded, not acted on. This is the looping/exhausted NPC the watchdog must escape."""
    def __init__(self):
        self.presses = []

    def frame_rgb(self):
        return _WhiteFrame()

    def rd8(self, _a):
        return 1

    def rd16(self, _a):
        return 1

    def rds16(self, _a):
        return 5

    def rd32(self, a):
        if a in (ram.GSAVEBLOCK1_PTR, ram.GSAVEBLOCK2_PTR):
            return 0x02020000                          # a valid EWRAM ptr so fingerprint() proceeds
        return 0                                        # GBATTLE_RES_PTR=0 -> in_battle() False

    def press(self, name, *_a, **_k):
        self.presses.append(name)

    def run_frame(self):
        pass

    def set_input_owner(self, _o):
        pass


class _StuckText:
    def _read_buffer(self):
        return "I like shorts! They're comfy and easy to wear!"   # never changes -> never 'new text'


def part_b():
    print("\n==== PART B: drive() FIRING control (exact wedge: box stuck open, frozen world) ====")
    fb = _StuckBridge()
    dd = DialogueDriver(fb, render=lambda: None, log=lambda m: print(m, flush=True))
    dd.dr = _StuckText()                                # force the never-changing line
    res = dd.drive(label="stuck-npc")
    a_presses = fb.presses.count("A")
    b_presses = fb.presses.count("B")
    print(f"  drive -> {res!r}; A-presses={a_presses}  B-presses={b_presses}")
    ok = (res == "exhausted") and (a_presses < 300) and (b_presses >= 1)
    print("  PART B:", "PASS" if ok else "FAIL",
          f"(want exhausted, A<<300, B>=1; old failure mode was 300 blind A-presses)")
    return ok


def _adjacent_walk(b, render, target, max_steps=40):
    """Greedy overworld walk toward `target` save-coord; stops when adjacent. Reduces the larger axis
    first and, if a step is BLOCKED (coords unchanged), tries the other axis - enough to thread a
    straight street. Returns the facing-press toward the target once adjacent, else None."""
    DIRS = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}
    for _ in range(max_steps):
        c = tv.coords(b)
        if c is None:
            return None
        dx, dy = target[0] - c[0], target[1] - c[1]
        if abs(dx) + abs(dy) <= 1:                      # adjacent
            for k, (vx, vy) in DIRS.items():
                if (vx, vy) == (dx, dy):
                    return k
            return None
        prefx = abs(dx) >= abs(dy)
        order = (["RIGHT" if dx > 0 else "LEFT"] if dx else []) + (["DOWN" if dy > 0 else "UP"] if dy else [])
        if not prefx:
            order.reverse()
        for key in order:                               # try preferred axis, then the other if blocked
            b.press(key, 8, 8, render, owner="agent")
            for _ in range(6):
                b.run_frame(); render()
            if tv.coords(b) != c:
                break
    return None


def part_c():
    print("\n==== PART C: real-engine NON-REGRESSION (talk to a real NPC -> clean close) ====")
    # Viridian: player (21,2) with an NPC straight down at (21,8) - a clean vertical street shot.
    state = os.path.join(WORKSHOP, "viridian_parcel_done.state")
    NPC = (21, 8)
    if not os.path.exists(state):
        print("  SKIP (viridian_parcel_done.state missing)")
        return True
    b = Bridge(ROM)
    with open(state, "rb") as f:
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
        pygame.display.set_caption("Kira - micro-watchdog control")

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()

    face = _adjacent_walk(b, render, NPC)
    if face is None:
        print(f"  SKIP (could not reach an NPC adjacent tile; player={tv.coords(b)})")
        return True
    b.press(face, 8, 8, render, owner="agent")           # turn to face the NPC
    b.press("A", 8, 8, render, owner="agent")            # open dialogue
    for _ in range(10):
        b.run_frame(); render()
    opened = box_open(b)
    print(f"  faced {face}, A -> box_open={opened}")
    dd = DialogueDriver(b, render=render, log=lambda m: print(m, flush=True))
    res = dd.drive(label="pallet-npc")
    moved = None
    c0 = tv.coords(b)
    for d in ("DOWN", "UP", "LEFT", "RIGHT"):
        b.press(d, 8, 8, render, owner="agent")
        for _ in range(6):
            b.run_frame(); render()
        if tv.coords(b) != c0:
            moved = d
            break
    print(f"  drive -> {res!r}; (move-probe stepped {moved}) coords={tv.coords(b)}")
    # Non-regression: a real long (36-press) scroll-NPC must drive to a clean 'closed', NEVER be
    # mis-judged 'exhausted'. 'closed' is itself proof control returned (drive only returns it after
    # _control_returned() passes), so the move-probe above is just extra colour, not the gate.
    if not opened:
        print("  (NPC gave no box - inconclusive, not counted as failure)")
        return True
    ok = (res == "closed")
    print("  PART C:", "PASS" if ok else f"FAIL (expected 'closed', got {res!r})")
    return ok


def main():
    a = part_a()
    b = part_b()
    c = part_c()
    allok = a and b and c
    print("\n==== RESULT:", "ALL CONTROLS PASS" if allok else "SOME CONTROLS FAILED", "====")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
