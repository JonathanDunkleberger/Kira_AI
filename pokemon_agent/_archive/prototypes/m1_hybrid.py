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
     Drive (arrows / Z=A / X=B) into Oak's lab, up to the table. Press P: her SELF
     picks and ANNOUNCES it in her own voice. Then YOU grab that ball by hand - the
     moment it lands in your party, she reacts. (Auto-grabbing blind proved
     unreliable - camera-vs-coord offset - so the human does the final button-press;
     her self still makes the genuine choice + every reaction. The autonomous grab
     is a later pass on real overworld nav.)

Keys: arrows=D-pad, Z=A, X=B, Enter=START, Backspace=SELECT, P=she decides.
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
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCALE = 3
BOT = "http://127.0.0.1:8766/cmd"


def post(action, **body):
    req = urllib.request.Request(f"{BOT}/{action}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def log(m):
    print(f"   [M1-hybrid] {m}", flush=True)


_choice = {"value": None}   # her announced pick, awaiting the human's grab


def run_choice(bridge):
    """P = ask her SELF (the soul). She announces her pick in her own voice; YOU then
    grab that ball by hand (Z). The bot watches party 0->1 and reacts when it lands.
    (Auto-positioning the grab blind proved unreliable - camera-vs-coord offset - so
    the human does the final button-press; her self still makes the genuine choice.)"""
    if nav.coords(bridge) is None:
        log("not in the overworld yet - clear the intro / get into the lab first"); return
    try:
        res = post("pokemon_choose_starter")
    except Exception as e:
        log(f"FAIL - couldn't reach the bot ({e}). Bot running + POKEMON_AGENT_ENABLED?"); return
    choice = (res.get("choice") or "bulbasaur").lower()
    _choice["value"] = choice
    log(f"HER SELF CHOSE: {choice.upper()}")
    log(f"  reasoning: {res.get('reasoning','')!r}")
    # she SAYS it, in her voice (neutral event -> _pokemon_react -> her self)
    post("pokemon_event", name=f"you're reaching for {choice} at the table")
    log(f">>> now WALK to the {choice.upper()} ball and grab it by hand (Z). I'll react when it lands.")


def watch_pickup(bridge):
    """Fire the completion reaction the instant party goes 0->1 (you grabbed it)."""
    if _choice["value"] is None:
        return
    if ram.read_party_count(bridge) >= 1:
        sid = st.read_party_species(bridge, 0)
        sname = st.SPECIES_NAME.get(sid, f"id{sid}")
        chose = _choice["value"]; _choice["value"] = None
        match = (st.STARTER_SPECIES.get(chose) == sid)
        print("\n" + "=" * 60 + "\n   M1 RESULT (hybrid starter pick)\n" + "=" * 60)
        print(f"   her self chose ........... {chose}")
        print(f"   party 0->1 ............... y  (species read = {sname})")
        print(f"   matches her choice ....... {match}")
        print("=" * 60, flush=True)
        post("pokemon_event", name=f"{sname} joined the team")
        log(f"PASS - {sname} is on the team; she's reacting." if match
            else f"NOTE - she wanted {chose} but {sname} landed (you grabbed a different ball).")


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
    log("intro cleared - drive to Oak's lab table, then press P to have HER decide.")
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                    run_choice(bridge)
            pressed = pygame.key.get_pressed()
            keys = [v for k, v in keymap.items() if pressed[k]]
            bridge.set_keys(*keys) if keys else bridge.release()
            bridge.run_frame(); blit()
            watch_pickup(bridge)        # fires her reaction the instant you grab her ball
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
