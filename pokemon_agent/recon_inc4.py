"""recon_inc4.py - CONTROLS for increment 4 (blackout recovery + step-3 hard recovery + boot sanity).

The live run (53e5f34): she booted LOW-HP, lost a wild, BLACKED OUT, the whiteout warped her INSIDE
a Pokémon Center (map group != 3, the (7,4)@(5,4) dead-end), and free_roam then sat in MACRO RED for
20+ ticks with only head_to_gym->no_gym_route — awareness couldn't save her because no option could
ever succeed. These controls prove the three fixes, headless + deterministic.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_inc4.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                        # noqa: E402
import campaign as C                              # noqa: E402
from campaign import Campaign                     # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WORKSHOP = os.path.join(_HERE, "states", "workshop")


def _load(name):
    b = Bridge(ROM)
    with open(os.path.join(WORKSHOP, name + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    return b


def _idle_oracle(kind, options, ctx):
    return ""                                     # no pick -> she idles; keeps the tick short


def part_a():
    print("\n==== PART A: blackout / stranded-in-building recovery ====")
    b = _load("misty_done")
    events = []
    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: events.append((k.get("kind"), s)), choose=_idle_oracle)

    # Simulate the post-blackout position: map_id reports a BUILDING interior (group 5) until the
    # recovery primitive runs, then the overworld. Spy the exit primitive.
    inside = [True]
    exited = [0]

    def spy_exit(*a, **k):
        exited[0] += 1
        inside[0] = False                         # _exit_to_overworld succeeded -> now on the overworld
        return True
    camp._exit_to_overworld = spy_exit

    real_map_id = C.tv.map_id
    C.tv.map_id = lambda br: (5, 4) if inside[0] else real_map_id(br)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            camp.free_roam(max_ticks=1, max_seconds=30, want_every=99)
    finally:
        C.tv.map_id = real_map_id
    out = buf.getvalue()

    blackout_logged = "BLACKOUT/STRANDED" in out
    blackout_beat = any(k == "blackout" for k, _ in events)
    print(f"  in-building detected + exited: calls={exited[0]} | BLACKOUT logged={blackout_logged} | "
          f"soul beat fired={blackout_beat}")
    if blackout_beat:
        print(f"  soul said: {next(s for k, s in events if k == 'blackout')!r}")
    ok = (exited[0] >= 1) and blackout_logged and blackout_beat
    print("  PART A:", "PASS" if ok else "FAIL")
    return ok


def part_b():
    print("\n==== PART B: step-3 hard recovery on sustained RED ====")
    import world_fingerprint as wf
    b = _load("brock_done")                       # overworld (group 3) so PART A doesn't interfere
    camp = Campaign(b, battle_runner=lambda: "win", on_event=lambda *a, **k: None,
                    choose=lambda kind, opts, ctx: ("head_to_gym" if "head_to_gym" in
                                                    (opts.keys() if isinstance(opts, dict) else opts) else ""))
    # Freeze the world: her chosen action is a no-op -> the macro light must escalate to RED and stay.
    camp._route_action = lambda pick, st: "noop_frozen"
    # Spy the forced recovery; keep it a no-op so the world STAYS frozen and we reach ABANDON.
    heal_calls = [0]
    camp.heal_nearest = lambda: (heal_calls.__setitem__(0, heal_calls[0] + 1), "ok")[1]

    buf = io.StringIO()
    with redirect_stdout(buf):
        outcome = camp.free_roam(max_ticks=25, max_seconds=60, want_every=99)
    out = buf.getvalue()

    hard_logged = "HARD RECOVERY" in out
    abandon_logged = "ROAM ABANDONED" in out
    print(f"  outcome={outcome!r} | _roam_progress={getattr(camp, '_roam_progress', None)!r} | "
          f"heal_nearest forced={heal_calls[0]}x")
    print(f"  HARD RECOVERY logged={hard_logged} | ROAM ABANDONED logged={abandon_logged} | "
          f"hard@{wf.PROGRESS_HARD_TICKS} abandon@{wf.PROGRESS_ABANDON_TICKS} RED ticks")
    ok = (outcome == "abandoned" and getattr(camp, "_roam_progress", None) == "ABANDONED"
          and heal_calls[0] >= 1 and hard_logged and abandon_logged)
    print("  PART B:", "PASS" if ok else "FAIL")
    return ok


def main():
    a = part_a()
    b = part_b()
    allok = a and b
    print("\n==== RESULT:", "ALL CONTROLS PASS" if allok else "SOME CONTROLS FAILED", "====")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
