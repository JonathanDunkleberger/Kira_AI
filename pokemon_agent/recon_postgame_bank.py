"""recon_postgame_bank.py — build the TRUE post-game bank (banked_POSTGAME) from banked_CREDITS.

WHY (night shift 2026-07-08, frame-proven): banked_CREDITS is captured MID-Hall-of-Fame ceremony
(Oak beside her, machine dialogue pending, player NOT in control). Any resume that drains that
dialogue completes the ceremony -> CREDITS ROLL -> post-credits SoftReset -> TITLE SCREEN — the
QW-4 "void core". The canonical watch spawn is therefore a live grenade.

THIS SCRIPT: load banked_CREDITS -> A-drain the ceremony -> let the credits roll (headless max
speed) -> title screen -> START -> CONTINUE -> verify a LIVE Champion world (party 6, badges 8,
real map, player in control: a test walk moves her) -> bank savestate + sidecars as
banked_POSTGAME -> round-trip verify. Promotion to canonical is a SEPARATE explicit step.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_postgame_bank.py
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                    # noqa: E402
import firered_ram as ram                    # noqa: E402
import travel as tv                          # noqa: E402
from dialogue_drive import box_open          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
SRC = os.path.join(SCRATCH, "banked_CREDITS")
DST = os.path.join(SCRATCH, "banked_POSTGAME")
OUT = os.path.join(SCRATCH, "voidcore_probe")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def L(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def world_alive(b):
    try:
        return (tv.map_id(b) != (0, 0) and tv.coords(b) is not None
                and b.rd8(ram.GPLAYER_PARTY_CNT) > 0)
    except Exception:
        return False


def snap(b, name):
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, name + ".png"))
        L(f"   frame -> {name}.png")
    except Exception as e:
        L(f"   snap failed: {e}")


def main():
    b = Bridge(ROM)
    with open(os.path.join(SRC, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    L(f"boot: map={tv.map_id(b)} coords={tv.coords(b)} (the mid-ceremony HoF moment)")

    # 1) drain the ceremony: A through the machine dialogue; the fade+credits follow on their own.
    #    Stop pressing once the world reads dead (credits/title own the screen) or the map leaves HoF.
    L("draining the Hall-of-Fame ceremony (A) …")
    for i in range(400):
        if not world_alive(b):
            L(f"   world went dark after {i} presses — ceremony complete, credits rolling")
            break
        b.press("A", 6, 10)
        for _ in range(20):
            b.run_frame()
    else:
        L("!! ceremony never completed (still alive in HoF after 400 A-presses) — ABORT")
        snap(b, "postgame_stuck_ceremony")
        return 1

    # 2) run out the credits to the title screen (no input — just frames, headless max speed).
    L("running out the credits …")
    t_cred = time.time()
    while time.time() - t_cred < 600:
        for _ in range(600):
            b.run_frame()
        # title signature: dead world + no box; poke START to leave "PRESS START" once it's up
        if not world_alive(b):
            b.press("START", 8, 12)
            for _ in range(30):
                b.run_frame()
            if box_open(b):                     # the main menu list (CONTINUE/NEW GAME) is a box
                L(f"   main menu up after {time.time() - t_cred:.0f}s of credits")
                break
    snap(b, "postgame_menu")

    # 3) CONTINUE is the top entry — A, then settle; then A once more through the save summary.
    L("selecting CONTINUE …")
    for i in range(30):
        b.press("A", 8, 12)
        for _ in range(60):
            b.run_frame()
        if world_alive(b):
            break
    if not world_alive(b):
        L("!! CONTINUE never restored a live world — ABORT")
        snap(b, "postgame_stuck_continue")
        return 1
    for _ in range(120):
        b.run_frame()
    badges = sum(1 for i in range(8) if (lambda sb1: (b.rd8(sb1 + 0x0EE0 + ((0x820 + i) >> 3))
                                                      >> ((0x820 + i) & 7)) & 1)(b.rd32(ram.GSAVEBLOCK1_PTR)))
    L(f"CONTINUED: map={tv.map_id(b)} coords={tv.coords(b)} "
      f"party={b.rd8(ram.GPLAYER_PARTY_CNT)} badges={badges}")
    snap(b, "postgame_world")

    # 4) control check: a test step must MOVE her (player in control, no pending script).
    c0 = tv.coords(b)
    for d in ("DOWN", "UP", "LEFT", "RIGHT"):
        b.press(d, 10, 8)
        for _ in range(20):
            b.run_frame()
        if tv.coords(b) != c0:
            break
    in_control = tv.coords(b) != c0
    L(f"control check: {'MOVED' if in_control else 'no movement (script may own input)'} "
      f"{c0} -> {tv.coords(b)}")
    if not (b.rd8(ram.GPLAYER_PARTY_CNT) == 6 and badges == 8 and in_control):
        L("!! verification failed (party/badges/control) — NOT banking")
        return 1

    # 5) bank: savestate + the CREDITS bundle's sidecars (her story is the same story).
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    for side in ("journey_core.json", "soul.json", "strat_memory.json", "world_model.json"):
        src = os.path.join(SRC, side)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DST, side))
    L(f"banked -> {DST}")

    # 6) round-trip verify.
    b2 = Bridge(ROM)
    with open(os.path.join(DST, "kira_campaign.state"), "rb") as f:
        b2.load_state(f.read())
    for _ in range(60):
        b2.run_frame()
    ok = (world_alive(b2) and b2.rd8(ram.GPLAYER_PARTY_CNT) == 6
          and tv.map_id(b2) == tv.map_id(b))
    L(f"round-trip: map={tv.map_id(b2)} party={b2.rd8(ram.GPLAYER_PARTY_CNT)} -> "
      f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
