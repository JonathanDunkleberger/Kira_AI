"""session.py - the canonical Kira Pokemon session (SKIP-STARTER).

We DO NOT automate the starter cutscene (camera-vs-coord dead end, zero gameplay
value). Instead this BOOTS from a post-pick savestate (Bulbasaur already in the
party, player in normal overworld control) and keeps the SOUL as VOICE: on start,
her real self (Sonnet, via the bot's control server) announces what she'd have
picked. It's commentary over an already-picked state, not hands in a cutscene.

INPUT OWNERSHIP (the show-stopper this design kills):
  There is EXACTLY ONE input writer at a time, enforced by Bridge's owner guard
  (set_input_owner + owner= on set_keys/release). No intro masher
  exists here (we boot past the intro), so nothing fights the human/agent for the
  pad. Any input attempt from a non-active owner is logged LOUDLY and dropped -
  phantom presses can never be silent again (Constraint #3).

RUN (normally launched by the dashboard Start button; manual fallback):
  1) Live Kira bot running with POKEMON_AGENT_ENABLED=true (watch boot log: NO
     ⚠ [FALLBACK] lines, or her pick is the weak model's, not hers).
  2) .venv\\Scripts\\python.exe pokemon_agent\\session.py
     Boots Bulbasaur overworld; she announces her pick; you drive (arrows/Z/X).
Keys: arrows=D-pad, Z=A, X=B, Enter=START, Backspace=SELECT.
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
STATES = os.path.join(_HERE, "states")
BOOT_STATE = os.path.join(STATES, "after_pick_bulbasaur.state")  # canonical start
SCALE = 3
BOT = "http://127.0.0.1:8766/cmd"


def log(m):
    print(f"   [pkmn-session] {m}", flush=True)


# ── INPUT ARCHITECTURE (single owner + paceable, NOT a blind masher) ──────────
# Input ownership is enforced at the single chokepoint, Bridge (set_input_owner +
# the owner= guard on set_keys/release/press). There is NO masher/timer/heartbeat
# here — we boot past the intro, so nothing auto-presses. Exactly one deliberate
# owner writes at a time ("human" now; "agent" later).
#
# FUTURE autonomous loop (deliberately NOT a speed-masher) — the agent loop is
# decide -> (optional commentary beat) -> press, one deliberate press at a time:
#
#     bridge.set_input_owner("agent")
#     while playing:
#         state  = read_state(bridge)               # coords / battle / dialogue
#         action = agent.decide(state)              # ONE deliberate decision
#         if is_performance_beat(state, action):    # naming, new area, battle, NPC gag
#             yield_to_commentary(state, action)    # let her VOICE land before advancing
#         bridge.set_keys(*action.keys, owner="agent")   # single owner, deliberate
#         # boring connective tissue (hallways, "got 5 Potions") just presses briskly
#
# The point: hands and voice on different clocks; the commentary gate sits BETWEEN
# decide and press, so a future pacing layer slots in without re-architecting. The
# owner guard guarantees no second source can ever inject a press mid-beat.


def post(action, **body):
    req = urllib.request.Request(f"{BOT}/{action}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def boot(bridge):
    """Load the canonical post-pick state, settle, and VERIFY party (loud)."""
    if not os.path.exists(BOOT_STATE):
        log(f"FAIL - boot state missing: {BOOT_STATE}")
        return False
    with open(BOOT_STATE, "rb") as f:
        bridge.load_state(f.read())
    for _ in range(40):
        bridge.run_frame()
    co = nav.coords(bridge)
    pc = ram.read_party_count(bridge)
    sid = st.read_party_species(bridge, 0) if pc >= 1 else None
    sname = st.SPECIES_NAME.get(sid, f"id{sid}")
    log(f"booted {os.path.basename(BOOT_STATE)}: coords={co} party={pc} starter={sname} (dex#{sid})")
    if co is None or pc < 1:
        log("FAIL - not in overworld control with a party mon (bad boot state)")
        return False
    return True


def announce_pick():
    """Her real self announces what she'd have picked (voice commentary, not hands).
    Posts to the bot; if the bot isn't up, we log loudly and continue (never silent)."""
    try:
        res = post("pokemon_choose_starter")
    except Exception as e:
        log(f"!! voice-pick SKIPPED - bot unreachable ({e}). Bot up + POKEMON_AGENT_ENABLED?")
        return
    choice = (res.get("choice") or "?").lower()
    log(f"HER SELF would pick: {choice.upper()}")
    log(f"  reasoning: {res.get('reasoning', '')!r}")
    # She SAYS it (neutral event -> _pokemon_react -> her self -> _ok_to_self_speak).
    # The party is Bulbasaur; if she names Bulbasaur it lands clean, if not she can riff.
    try:
        post("pokemon_event", name=f"you boot up your Pokemon save - Bulbasaur is your "
                                    f"partner, and you muse that you'd have picked {choice}")
    except Exception as e:
        log(f"!! announce event failed ({e})")


def main():
    import time
    import pygame
    pygame.init()
    bridge = Bridge(ROM)
    log(f"loaded {bridge.game_code} {bridge.game_title!r}")
    if not boot(bridge):
        pygame.quit(); return
    win = (bridge.width * SCALE, bridge.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("Kira Pokemon - skip-starter (Bulbasaur); Z/X/arrows")
    keymap = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_z: "A", pygame.K_x: "B",
              pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT"}

    def blit():
        surf = pygame.image.fromstring(bridge.frame_rgb().tobytes(),
                                       (bridge.width, bridge.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    bridge.set_input_owner("human")    # Bridge enforces: keyboard is the sole writer
    announce_pick()                    # her voice-pick commentary, once, on start
    log("you have control - drive with arrows / Z=A / X=B. (no masher: nothing fights you)")
    WFL = getattr(pygame, "WINDOWFOCUSLOST", None)
    # Event-tracked held keys (NOT pygame.key.get_pressed): immune to the launch-Enter
    # stuck key. When you start the session by pressing Enter in the terminal, the
    # window grabs focus as Enter releases and the KEYUP is eaten -> get_pressed() would
    # report Enter held forever (constant START). Here Enter's KEYDOWN predates pygame,
    # so it never enters `held`. Cleared on focus loss so a lost KEYUP can't strand a key.
    held = set()
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if (WFL is not None and ev.type == WFL) or \
                   (ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0):
                    if held:
                        held.clear(); log("focus lost -> held keys cleared (no stuck-key phantom)")
                elif ev.type == pygame.KEYDOWN and ev.key in keymap:
                    held.add(keymap[ev.key])
                elif ev.type == pygame.KEYUP and ev.key in keymap:
                    held.discard(keymap[ev.key])
            bridge.set_keys(*held, owner="human") if held else bridge.release(owner="human")
            bridge.run_frame(); blit()
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
