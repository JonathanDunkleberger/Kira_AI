"""travel_session.py - harness around the travel engine (the FEET in action).

Boots the canonical Route-1 state and walks her to Viridian City with the
deterministic BFS pathfinder, handing wild encounters to the 5/5 battle engine and
narrating in her real voice. travel.py stays pure (no pygame/bot/battle imports);
this is the wiring, mirroring session.py / m1_battle.py.

MODES:
  --headless   end-to-end OFFLINE proof (no window, no bot voice): does she walk
               Route 1 -> Viridian, fighting any wild battles with the battle
               engine's hands? Reports outcome + final map + steps. RNG-dependent.
  (default)    LIVE windowed run: you watch her walk Route 1 -> Viridian; events go
               to her voice via the bot (needs the bot up + POKEMON_AGENT_ENABLED).

RUN:
  .venv\\Scripts\\python.exe pokemon_agent\\travel_session.py --headless
  .venv\\Scripts\\python.exe pokemon_agent\\travel_session.py
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge            # noqa: E402
import firered_ram as ram           # noqa: E402
import travel as tv                 # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BOOT_STATE = os.path.join(_HERE, "states", "after_pick_bulbasaur.state")
SCALE = 3
BOT = "http://127.0.0.1:8766"


def log(m):
    print(f"   [travel-session] {m}", flush=True)


def boot(bridge):
    if not os.path.exists(BOOT_STATE):
        log(f"FAIL - boot state missing: {BOOT_STATE}"); return False
    with open(BOOT_STATE, "rb") as f:
        bridge.load_state(f.read())
    for _ in range(40):
        bridge.run_frame()
    co = tv.coords(bridge)
    log(f"booted map={tv.map_id(bridge)} coords={co} party={ram.read_party_count(bridge)}")
    return co is not None


# ── headless: end-to-end offline proof (hands only, no voice) ────────────────
def run_headless(seconds):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    bridge = Bridge(ROM)
    log(f"loaded {bridge.game_code} {bridge.game_title!r}")
    if not boot(bridge):
        return

    def battle_runner():
        # hands-only battle (events just print; no bot). Returns outcome.
        agent = BattleAgent(bridge, on_event=lambda s, **k: log(f"[battle] {s}"),
                            render=lambda: None)
        return agent.run(max_seconds=120)

    trav = tv.Traveler(bridge, battle_runner=battle_runner,
                       on_event=lambda s: log(f"[event] {s}"), log=print)
    outcome = trav.travel(target_map=tv.MAP_VIRIDIAN)
    final = tv.map_id(bridge)
    print("\n" + "=" * 60)
    print("   TRAVEL (headless) RESULT")
    print("=" * 60)
    print(f"   outcome .................. {outcome}")
    print(f"   final map ................ {final}  "
          f"({'VIRIDIAN - PASS' if final == tv.MAP_VIRIDIAN else 'not Viridian'})")
    print(f"   final coords ............. {tv.coords(bridge)}")
    print("=" * 60, flush=True)


# ── live: windowed, voiced ───────────────────────────────────────────────────
def run_live(seconds):
    import json
    import time
    import urllib.request
    import pygame
    pygame.init()
    bridge = Bridge(ROM)
    log(f"loaded {bridge.game_code} {bridge.game_title!r}")
    if not boot(bridge):
        pygame.quit(); return
    win = (bridge.width * SCALE, bridge.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("Kira travels - Route 1 -> Viridian (autonomous BFS)")

    def render():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    def post(action, **body):
        try:
            req = urllib.request.Request(f"{BOT}/cmd/{action}",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.load(r)
        except Exception as e:
            log(f"voice post {action} failed (bot up + POKEMON_AGENT_ENABLED?): {e}")
            return None

    def is_speaking():
        try:
            with urllib.request.urlopen(f"{BOT}/state", timeout=3) as r:
                return bool(json.load(r).get("is_speaking"))
        except Exception:
            return False

    def on_event(summary):
        log(f"[event->Kira] {summary!r}")
        post("pokemon_event", name=summary)

    def pace(summary):
        t0 = time.time()
        while time.time() - t0 < 3.5 and not is_speaking():
            bridge.run_frame(); render()
        t1 = time.time()
        while time.time() - t1 < 14.0 and is_speaking():
            bridge.run_frame(); render()
        for _ in range(24):
            bridge.run_frame(); render()

    def battle_runner():
        agent = BattleAgent(bridge, on_event=on_event, render=render, pace=pace)
        return agent.run(max_seconds=seconds)

    trav = tv.Traveler(bridge, battle_runner=battle_runner, render=render,
                       on_event=on_event, log=print)
    try:
        outcome = trav.travel(target_map=tv.MAP_VIRIDIAN)
        log(f"travel outcome: {outcome}  final map={tv.map_id(bridge)}")
    except KeyboardInterrupt:
        log("window closed / interrupted")
    finally:
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=300.0)
    args = ap.parse_args()
    if args.headless:
        run_headless(args.seconds)
    else:
        run_live(args.seconds)


if __name__ == "__main__":
    main()
