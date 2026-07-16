"""recon_social_verify.py — F-6 SOCIAL FABRIC verify (headless, sandbox-only, no bot).

Off banked_POSTGAME (she spawns at HOME, (4,0) — Mom is the resident key figure):
  1. greet:mom is OFFERED in the honest action set (with the warm framing)
  2. executing it WALKS to her, opens a real dialogue box, and returns 'greeted'
  3. the met-mark lands in world.social and round-trips through world_model save/load
  4. after greeting, greet:mom is NO LONGER offered
  5. the skip-voicing hook fires when leaving the map with an un-greeted figure (simulated)

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_social_verify.py
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
from campaign import Campaign                              # noqa: E402
import travel as tv                                        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BUNDLE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_POSTGAME")


def main():
    b = Bridge(ROM)
    with open(os.path.join(BUNDLE, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    events = []
    camp = Campaign(b, battle_runner=lambda: "skipped",
                    on_event=lambda s, **k: events.append((k.get("kind", ""), s)),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True     # sandbox: no banking
    camp._continuity_save = lambda *a, **k: None

    state = camp.read_live_state()
    print(f"spawn: {state['map']}@{state['coords']} place={state['place']!r}")
    assert tuple(state["map"]) == (4, 0), "expected the postgame HOME spawn"

    avail = camp._available_actions(state)
    print(f"options: {sorted(avail)}")
    assert "greet:mom" in avail, "greet:mom not offered at home"
    print(f"offer: {avail['greet:mom']!r}")

    out = camp._route_action("greet:mom", state)
    print(f"greet:mom -> {out}")
    assert out == "greeted", f"greet failed: {out}"
    assert camp.world.met("mom"), "met-mark missing"

    # persistence round-trip
    tmp = os.path.join(tempfile.gettempdir(), "f6_world_roundtrip.json")
    camp.world.save(tmp)
    from pokemon_world import WorldModel
    w2 = WorldModel(log=lambda *a, **k: None)
    w2.load(tmp)
    assert w2.met("mom"), "met-mark did not survive save/load"

    avail2 = camp._available_actions(camp.read_live_state())
    assert "greet:mom" not in avail2, "greet re-offered after greeting"

    # skip-voicing: simulate an offer on one map then a map change with an un-greeted figure
    camp.world.social.pop("daisy", None)
    camp._social_last = ((4, 2), (("daisy", "Daisy, Gary's sister"),))
    camp._social_skip_voiced = set()
    camp._social_tick(camp.read_live_state())      # current map (4,0) != (4,2) -> voices the skip
    skips = [s for k, s in events if k == "social"]
    print(f"skip beats: {skips}")
    assert any("Daisy" in s for s in skips), "skip not voiced"
    n = len(skips)
    camp._social_last = ((4, 2), (("daisy", "Daisy, Gary's sister"),))
    camp._social_tick(camp.read_live_state())      # once per run — must NOT re-voice
    assert len([s for k, s in events if k == "social"]) == n, "skip re-voiced"

    # the ctx pull rides the place seam for unmet figures (mom now met -> no pull at home)
    blk = camp._location_block(camp.read_live_state())
    pulls = camp._salient_unmet(camp.read_live_state())
    print(f"location block: {blk}")
    print(f"remaining pulls here: {pulls}")
    assert pulls == [], "mom still pulling after greet"

    print("\n==== ALL PASS (offer -> greet -> persist -> de-offer -> skip-voice) ====")


if __name__ == "__main__":
    main()
