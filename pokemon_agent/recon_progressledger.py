"""recon_progressledger.py - CONTROL for increment-3 MACRO ProgressLedger + oracle feedback.

A) ProgressLedger UNIT control (deterministic): a static fingerprint escalates GREEN -> YELLOW(3)
   -> RED(6); a moving fingerprint stays GREEN; a box-up tick does NOT escalate (context-aware);
   a changed fingerprint resets to GREEN.
B) free_roam INTEGRATION (real Campaign, fake oracle): on a real overworld state, with her chosen
   action stubbed to a no-op so the world stays frozen, the loop must (1) drive the light
   GREEN->YELLOW->RED in the WATCH log, and (2) FEED the stuck-awareness into the oracle ctx on
   YELLOW+ (the fake oracle records the `place` field it receives) so she'd see it and could change
   course. Proves the seam end-to-end without needing the live bot/LLM.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_progressledger.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import world_fingerprint as wf                 # noqa: E402
from bridge import Bridge                       # noqa: E402
from campaign import Campaign                   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WORKSHOP = os.path.join(_HERE, "states", "workshop")


def _fp(x=5, y=5, box=False, hp=20):
    return wf.WorldFingerprint((3, 2), x, y, 1, box, False, ((1, 5, hp),), 1, 4824, ())


def part_a():
    print("\n==== PART A: ProgressLedger unit control ====")
    ok = True

    # (1) static fingerprint -> GREEN x3, YELLOW at stuck=3, RED at stuck=6
    L = wf.ProgressLedger()
    seq = []
    for _ in range(8):
        seq.append(L.observe(_fp()))
        L.note_action("head_to_gym", "no_gym_route")
    print(f"  static escalation: {seq}")
    ok &= seq[:3] == ["GREEN"] * 3 and seq[3] == "YELLOW" and seq[6] == "RED"
    print(f"  YELLOW at tick {seq.index('YELLOW')+1} (want {wf.PROGRESS_YELLOW_TICKS+1}), "
          f"RED at tick {seq.index('RED')+1} (want {wf.PROGRESS_RED_TICKS+1})")

    # (2) a MOVING world never escalates
    L = wf.ProgressLedger()
    moving = [L.observe(_fp(x=i)) for i in range(6)]
    print(f"  moving world: {moving}")
    ok &= all(s == "GREEN" for s in moving)

    # (3) CONTEXT-AWARE: box-up ticks don't escalate. Feed static-but-box-up frames in the middle of
    #     a stuck run; the stuck count must NOT climb on those ticks.
    L = wf.ProgressLedger()
    L.observe(_fp()); L.observe(_fp())                 # stuck=1
    s_before = L.stuck
    L.observe(_fp(box=True)); L.observe(_fp(box=True))  # box up -> held, no escalation
    s_after = L.stuck
    print(f"  box-up ticks: stuck {s_before} -> {s_after} (want unchanged, no escalation)")
    ok &= (s_before == s_after)

    # (4) RESET: after climbing, a changed fingerprint drops back to GREEN
    L = wf.ProgressLedger()
    for _ in range(7):
        L.observe(_fp())
    red = L.state
    g = L.observe(_fp(x=99))                            # the world MOVED
    print(f"  reset after {red}: moved -> {g} (stuck now {L.stuck})")
    ok &= (red == "RED" and g == "GREEN" and L.stuck == 0)

    print("  PART A:", "PASS" if ok else "FAIL")
    return ok


def part_b():
    print("\n==== PART B: free_roam INTEGRATION (real Campaign, fake oracle, frozen world) ====")
    state_path = os.path.join(WORKSHOP, "brock_done.state")
    if not os.path.exists(state_path):
        print("  SKIP (brock_done.state missing)")
        return True
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    seen_places = []                                   # what `place` the oracle received each action tick

    def fake_oracle(kind, options, ctx):
        """Stand-in for her self/LLM: records the situational `place` it's handed, always picks the
        same action (head_to_gym) so we can watch the world FAIL to move and the light escalate."""
        if kind != "action":
            return ""                                  # ignore 'want' beats in this control
        seen_places.append(ctx.get("place", ""))
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        return "head_to_gym" if "head_to_gym" in opts else (opts[0] if opts else "")

    camp = Campaign(b, battle_runner=lambda: "win", on_event=lambda *a, **k: None, choose=fake_oracle)
    # Stub her action to a NO-OP so the world stays frozen -> the macro light must escalate. (We're
    # testing the LEDGER + feedback wiring, not the real handlers; this monkeypatch is test-only.)
    camp._route_action = lambda pick, st: "noop_frozen"

    buf = io.StringIO()
    with redirect_stdout(buf):
        camp.free_roam(max_ticks=8, max_seconds=60, want_every=99)
    out = buf.getvalue()

    progress_lines = [ln.strip() for ln in out.splitlines() if "PROGRESS:" in ln]
    macro_seq = []
    for ln in progress_lines:
        for s in ("GREEN", "YELLOW", "RED"):
            if f"PROGRESS: {s}" in ln:
                macro_seq.append(s); break
    fed = [p for p in seen_places if "hasn't changed" in p]
    print(f"  per-tick macro light: {macro_seq}")
    print(f"  oracle `place` tick1 (clean?): {seen_places[0][:60]!r}" if seen_places else "  (no oracle calls)")
    yellow_tick = next((i + 1 for i, p in enumerate(seen_places) if "hasn't changed" in p), None)
    print(f"  stuck-awareness first reached the oracle at action-tick {yellow_tick} "
          f"(YELLOW@{wf.PROGRESS_YELLOW_TICKS+1})")
    if fed:
        print(f"  fed example: {fed[-1][:110]!r}")
    saw_feedback_log = "MACRO" in out and "feeding awareness back" in out
    ok = ("YELLOW" in macro_seq and "RED" in macro_seq and bool(fed)
          and (seen_places and "hasn't changed" not in seen_places[0]) and saw_feedback_log)
    print("  PART B:", "PASS" if ok else "FAIL")
    return ok


def main():
    a = part_a()
    b = part_b()
    print("\n==== RESULT:", "ALL CONTROLS PASS" if (a and b) else "SOME CONTROLS FAILED", "====")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
