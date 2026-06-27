"""recon_batch2.py - CONTROLS for BATCH 2 (survival instincts), built A -> C -> B.

PART A (heal-when-hurt): the documented catastrophe is a 7%-HP boot (misty_done.state = Ivysaur ~4/60)
walking straight into grass and blacking out. _boot_state_sanity SCREAMS about it; PART A makes her ACT
on it — the party-wide hurt signal surfaces 'heal' HARD to the oracle so a badly-hurt Kira reliably
heals first. These controls prove (headless, deterministic):
  A1  a critically-hurt party -> severity 'critical', 'heal' OFFERED, the hurt note reaches the oracle
      ctx, the heal handler runs when she picks it, and the 'patched up' soul beat fires.
  A2  a HEALTHY party -> severity None, heal NOT force-surfaced (no over-firing on a full team).

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_batch2.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                            # noqa: E402
from campaign import Campaign, resolve_state         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def _load(name):
    b = Bridge(ROM)
    p = resolve_state(name + ".state")
    if not p:
        raise SystemExit(f"FAIL: state {name}.state not found")
    with open(p, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    return b


def _camp(b, oracle):
    events = []
    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: events.append((k.get("kind"), s)),
                    choose=oracle)
    return camp, events


def part_a():
    print("\n==== PART A: heal-when-hurt instinct ====")
    ok = True

    # ---- A1: critically-hurt party reliably surfaces + takes the heal ----
    b = _load("misty_done")
    seen = {"ctx": None}

    def heal_oracle(kind, options, ctx):
        if kind == "action":
            seen["ctx"] = ctx
            return "heal" if "heal" in options else None
        return None                                   # decline 'want' so the tick stays short

    camp, events = _camp(b, heal_oracle)
    ph = camp.party_health()
    sev, note = camp._hurt_severity()
    worst = min(ph, key=lambda t: t[3]) if ph else None
    print(f"   A1 party_health={[(s, hp, mx, round(f, 2)) for s, hp, mx, f in ph]}")
    print(f"   A1 worst={worst} severity={sev!r}")
    print(f"   A1 note={note!r}")
    avail = camp._available_actions(camp.read_live_state())
    print(f"   A1 actions offered={list(avail.keys())}")

    heal_calls = [0]

    def heal_spy():
        heal_calls[0] += 1
        return "ok"
    camp.heal_nearest = heal_spy                       # don't actually navigate headless

    buf = io.StringIO()
    with redirect_stdout(buf):
        camp.free_roam(max_ticks=1, max_seconds=60, want_every=99)
    roam_log = buf.getvalue()

    a1_sev = (sev == "critical")
    a1_dominates = (list(avail.keys()) == ["heal"])      # reckless options PRUNED -> heal dominates
    a1_crit_word = ("FIRST" in avail.get("heal", ""))
    a1_ctx = bool(seen["ctx"]) and ("badly hurt" in (seen["ctx"] or {}).get("place", ""))
    a1_handler = (heal_calls[0] >= 1)
    a1_beat = any(k == "heal" and "patched up" in s for k, s in events)
    for name, val in [("severity critical", a1_sev),
                      ("heal DOMINATES (catch/wander/gym pruned at near-death)", a1_dominates),
                      ("heal worded HARD ('FIRST')", a1_crit_word),
                      ("hurt note reached oracle ctx", a1_ctx),
                      ("heal handler ran on her pick", a1_handler),
                      ("'patched up' soul beat fired", a1_beat)]:
        print(f"   A1 [{'PASS' if val else 'FAIL'}] {name}")
        ok = ok and val
    if not a1_handler:
        print("   A1 (roam tail)\n     " + "\n     ".join(roam_log.strip().splitlines()[-12:]))

    # ---- A2: a HEALTHY party does NOT force-surface heal (no over-firing) ----
    b2 = _load("after_pick_bulbasaur")
    camp2, _ev2 = _camp(b2, lambda *a, **k: None)
    ph2 = camp2.party_health()
    sev2, _n2 = camp2._hurt_severity()
    print(f"\n   A2 party_health={[(s, hp, mx, round(f, 2)) for s, hp, mx, f in ph2]} severity={sev2!r}")
    a2 = (sev2 is None)
    print(f"   A2 [{'PASS' if a2 else 'FAIL'}] healthy party -> severity None (heal not force-surfaced)")
    ok = ok and a2

    return ok


def main():
    results = {"PART A": part_a()}
    print("\n" + "=" * 56)
    for k, v in results.items():
        print(f"   {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 56)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
