"""recon_forget_probe.py — measure the FORGET/'KNOWN MOVES' cursor border rows.

victory_run1 truth: _forget_goto(3) failed; _FORGET_TOPS rows 2-4 (67/90/112) are
comment-admitted PROBES, never measured (row0=18/row1=45 are the only ground truth;
spacing suggests ~27px, so rows 2-4 sit near 72/99/126 — outside the ±3 window).
This probe drives the real teach flow (TM26 -> Venusaur slot 0) from the canonical
save until the forget screen shows, then per DOWN press scans column x=118-126 for
the red-orange border and logs the y-runs. Frames saved to %TEMP%/longrun/forget_probe.
Fail-safe: _forget_goto is replaced by the scanner and returns False -> B-cascade out,
nothing taught, canonical untouched (we boot a throwaway core).
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_forget_probe.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import hm_teach as ht                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign", "kira_campaign.state")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "forget_probe")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    os.makedirs(DBG, exist_ok=True)
    b = Bridge(ROM)
    with open(CANON, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    class Duck:
        pass
    duck = Duck()
    duck.b = b
    duck.render = lambda: None

    teacher = ht.TeachFlow(duck, log=lambda m: print(m, flush=True))

    def scan(tag):
        img = b.frame_rgb()
        p = img.load()
        hits = {}
        for y in range(img.size[1]):
            n = 0
            for x in (116, 119, 122, 125, 128):
                r, g, bl = p[x, y][:3]
                if r > 200 and g < 120 and bl < 80:
                    n += 1
            if n:
                hits[y] = n
        # compress into runs
        runs, cur = [], None
        for y in sorted(hits):
            if cur and y == cur[1] + 1:
                cur[1] = y
            else:
                cur = [y, y]
                runs.append(cur)
        L(f"   [{tag}] border-y runs @x116-128: {[(a, z, sum(hits[y] for y in range(a, z + 1))) for a, z in runs]}")
        img.save(os.path.join(DBG, f"{tag}.png"))
        return runs

    if os.environ.get("MEASURE") == "1":
        def measuring_goto(target, tries=12):
            L(f"FORGET SCREEN REACHED (target row {target}) — measuring")
            scan("forget_row_initial")
            for k in range(5):
                teacher._press("DOWN", settle=20)
                scan(f"forget_after_down{k + 1}")
            return False                   # fail-safe B out — nothing taught
        teacher._forget_goto = measuring_goto

    r = teacher.teach("surf", 0, forget_idx=3, item_override=314, move_override=89)
    import pokemon_state as st
    moves = st.read_party_moves(b, 0) or []
    L(f"teach returned {r}; slot-0 moves {moves} (EQ={'YES' if 89 in moves else 'NO'}) "
      f"[throwaway core — canonical untouched]")
    return 0 if (os.environ.get("MEASURE") == "1" or 89 in moves) else 1


if __name__ == "__main__":
    sys.exit(main())
