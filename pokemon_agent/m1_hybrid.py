"""m1_hybrid.py - M1 HYBRID: you drive to the table, HER SELF picks the starter.

The HANDS (this emulator window) and the VOICE (the live Kira bot) are SEPARATE
processes that sync over the bot's existing HTTP control server (127.0.0.1:8766):
  - POST /cmd/pokemon_choose_starter  -> her self picks (Sonnet) + verbatim reasoning
  - POST /cmd/pokemon_event           -> fires _pokemon_react (she reacts in her voice)

HOW TO RUN (two things, in order):
  1) Start the live Kira bot as usual, with POKEMON_AGENT_ENABLED=true in .env.
     Watch the boot log: confirm there are NO  ⚠ [FALLBACK]  lines before you trust
     her pick - if she's on Llama-8B it's the 8B's choice, not HERS.
  2) .venv\\Scripts\\python.exe pokemon_agent\\m1_hybrid.py
     Drive (arrows / Z=A / X=B) into Oak's lab, clear the intro speech, and stand
     DIRECTLY IN FRONT of the LEFTMOST ball (Bulbasaur), facing UP. Then press P.
     The bot asks her self; her choice drives the final confirm; she reacts.

Keys: arrows=D-pad, Z=A, X=B, Enter=START, Backspace=SELECT, P=run her pick.
"""
import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import navigate as nav           # noqa: E402
import starter as stx            # noqa: E402
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCALE = 3
BOT = "http://127.0.0.1:8766/cmd"
STEP_FROM_BULBA = {"bulbasaur": 0, "charmander": 1, "squirtle": 2}   # +x along the front row


def post(action, **body):
    req = urllib.request.Request(f"{BOT}/{action}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def log(m):
    print(f"   [M1-hybrid] {m}", flush=True)


def run_pick(bridge, render):
    """Anchor = player's CURRENT tile (you left them in front of Bulbasaur). Ask her
    self, step right to her ball, advance the dialogue (settle cadence) to party 1."""
    anchor = nav.coords(bridge)
    if anchor is None:
        log("FAIL - no overworld coords; are you in the lab, intro cleared?"); return
    log(f"anchor (Bulbasaur tile) = {anchor}; asking her self...")
    try:
        res = post("pokemon_choose_starter")
    except Exception as e:
        log(f"FAIL - couldn't reach the bot ({e}). Is the bot running + POKEMON_AGENT_ENABLED?"); return
    choice = (res.get("choice") or "bulbasaur").lower()
    log(f"HER SELF CHOSE: {choice.upper()}")
    log(f"  verbatim reasoning: {res.get('reasoning','')!r}")
    post("pokemon_event", name="looking at the three starter Pokemon")

    step = STEP_FROM_BULBA.get(choice, 0)
    target = (anchor[0] + step, anchor[1])
    reached, final, _ = nav.walk_to(bridge, target, hold=8, render=render, log=lambda m: log(m))
    if not reached:
        log(f"FAIL - couldn't reach {choice} ball tile {target} (stuck {final})"); return
    bridge.press("UP", 8, 8, render)          # face the ball

    before = ram.read_party_count(bridge)
    bridge.press("A", 4, 8, render)           # open the selection dialogue
    got = stx.advance_dialogue(bridge, lambda: ram.read_party_count(bridge) > before,
                               render=render, max_presses=50, log=lambda m: None)
    pc = ram.read_party_count(bridge)
    sid = st.read_party_species(bridge, 0) if pc >= 1 else None
    sname = st.SPECIES_NAME.get(sid, f"id{sid}") if sid is not None else "(none)"
    match = (st.STARTER_SPECIES.get(choice) == sid)
    print("\n" + "=" * 60 + "\n   M1 RESULT (hybrid starter pick)\n" + "=" * 60)
    print(f"   her self chose ........... {choice}")
    print(f"   party 0->1 ............... {'y' if pc >= 1 else 'n'}  (count={pc})")
    print(f"   species read ............. {sname}  (match={match})")
    print("=" * 60, flush=True)
    if pc >= 1 and match:
        post("pokemon_event", name=f"chose {choice}")
        post("pokemon_event", name=f"{choice} joined the team")
        log("PASS - she picked, the hands confirmed it, she's reacting.")
    else:
        log("FAIL - pick did not complete cleanly; not firing 'joined the team'.")


def main():
    import pygame
    pygame.init()
    bridge = Bridge(ROM)
    log(f"loaded {bridge.game_code} {bridge.game_title!r}")
    win = (bridge.width * SCALE, bridge.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("M1 hybrid - drive to the table, P = her self picks")
    keymap = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_z: "A", pygame.K_x: "B",
              pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT"}

    def blit():
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    # clear the wake-up intro so you can drive
    import m0_sandbox as m0
    sched = m0.masher_keys()
    import time
    t0 = time.time()
    while time.time() - t0 < 90 and nav.coords(bridge) is None:
        k = next(sched); bridge.set_keys(k) if k else bridge.release()
        bridge.run_frame(); blit()
    log("intro cleared - drive to Oak's lab; stand in front of the LEFTMOST ball facing UP, then press P")
    try:
        while True:
            triggered = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                    triggered = True
            if triggered:
                run_pick(bridge, blit)
            pressed = pygame.key.get_pressed()
            keys = [v for k, v in keymap.items() if pressed[k]]
            bridge.set_keys(*keys) if keys else bridge.release()
            bridge.run_frame(); blit()
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
