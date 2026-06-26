"""free_roam.py - SESSION CAPSTONE: let Kira loose. A continuous, VISIBLE, goal-directed autonomous
loop from cerulean_caught.state - the first "turn her on and watch her play" preview. Loose goal:
BUILD A TEAM (catch wilds intentfully) then HEAD TOWARD MISTY (gym 2). Strings together the wired
Batch-1 mechanics (catch = Sleep-Powder weaken + commit; nav; battle) into one self-running session.

HONEST SCOPE (this is a PREVIEW, not a gate): WIRED = roam/nav, intentful catch, battle/flee.
NOT-YET-WIRED (Batch 2 / noted as gaps, she does what she can): autonomous SHOP decisions at the
Mart, heal at the CERULEAN Center (heal-return is Viridian-hardcoded), NPC conversations, and her
SELF/VOICE/WANTS driving choices (the personality layer). When she's out of balls she notes the
shop gap; when low she flees to survive and notes the Center-heal gap.

RUN (visible): WATCH=1 .venv\\Scripts\\python.exe -u pokemon_agent\\free_roam.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "1") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
OFF = tv.MAP_OFFSET
TEAM_TARGET = int(os.getenv("TEAM_TARGET", "4"))      # roam-catch until the party reaches this
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "900"))    # let her run ~15 min by default


def main():
    b = Bridge(ROM)
    with open(os.path.join(STATES, os.getenv("START", "cerulean_caught") + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write

    screen = win = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - FREE ROAM (watch her play)")

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()

    def lead_hp_frac():
        base = ram.GPLAYER_PARTY
        hp, mx = b.rd16(base + 0x56), b.rd16(base + 0x58)
        return hp / mx if mx else 1.0

    def say(msg):
        print(f"\n>>> KIRA: {msg}", flush=True)

    def _use_slot(ag, slot):
        ag._home_to_fight()
        for _ in range(8):
            if ag._in_move_list():
                break
            b.press("A", ag.hold, ag.hold, render, owner="agent"); ag._wait(10)
        ag._nav_move(slot)
        b.press("A", ag.hold, ag.hold, render, owner="agent"); ag._wait(40)

    p_ref = [ram.read_party_count(b)]

    def catch_runner():
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                         log=lambda m: print(f"      {m}", flush=True))
        t0 = time.time()
        ag._reach_first_menu(t0, 60)
        if ag._is_trainer_battle():
            return "win" if ag.run(90) != "loss" else "loss"
        rb = None
        for _ in range(20):
            rb = st.read_battle(b)
            if rb and rb.get("ours") and rb.get("enemy"):
                break
            ag._wait(4)
        foe = st.SPECIES_NAME.get(rb["enemy"]["species"], "?") if rb else "?"
        say(f"ooh, a wild {foe} — I'm gonna try to catch this one.")
        if rb:
            moves = rb["ours"]["moves"]
            status = [i for i, m in enumerate(moves) if m.get("id", 0) and m.get("power", 0) == 0]
            sleep = [i for i in status if "sleep" in str(moves[i].get("name", "")).lower()]
            slot = (sleep or status or [None])[0]
            if slot is not None:
                print(f"      [roam] weaken: {moves[slot].get('name','?')}", flush=True)
                _use_slot(ag, slot)
        for n in range(12):
            ag._settle()
            if ram.read_party_count(b) > p_ref[0]:
                return "caught"
            if not st.in_battle(b):
                return "broke_free"
            r = ag.throw_ball(max_seconds=45)
            print(f"      [roam] throw {n + 1} -> {r}", flush=True)
            if r == "caught":
                return "caught"
            if r == "no_balls":
                return "no_balls"
        return "stuck"

    # battle_runner for ROAMING (not actively catching): flee wilds to preserve HP, win trainers
    def roam_battle():
        ag = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                         log=lambda m: print(f"      {m}", flush=True))
        r = ag.flee(90)
        return "loss" if r == "loss" else "win"

    camp = Campaign(b, battle_runner=roam_battle, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=render)
    camp._suppress_heal = True

    t0 = time.time()
    gaps = set()
    say("Okay — I'm in Cerulean. Goal: build out my team, then go take on Misty. Let's explore.")

    # ── PHASE 1: build a team (roam Route 4 grass, catch intentfully until target or out of balls) ──
    out_of_balls = False
    while ram.read_party_count(b) < TEAM_TARGET and time.time() - t0 < MAX_SECONDS and not out_of_balls:
        if lead_hp_frac() < 0.30:
            say("my lead's hurting — I should heal at the Center... (Cerulean Center heal isn't wired yet)")
            gaps.add("heal at Cerulean Pokemon Center (heal-return is Viridian-hardcoded)")
            break
        say(f"team's at {ram.read_party_count(b)} — heading to the Route 4 grass to find another.")
        camp.battle_runner = catch_runner
        if tv.map_id(b) != (3, 22):
            camp.trav.travel(target_map=(3, 22), edge="west", max_steps=200, max_seconds=120)
        # walk into grass and wiggle to roll an encounter; catch_runner handles it
        got = False
        for _ in range(120):
            if ram.read_party_count(b) >= TEAM_TARGET:
                got = True; break
            if st.in_battle(b):
                res = catch_runner()
                if res == "no_balls":
                    out_of_balls = True; break
                if ram.read_party_count(b) > p_ref[0]:
                    p_ref[0] = ram.read_party_count(b)
                    say(f"gotcha! roster is now {p_ref[0]}.")
                    got = True; break
                continue
            grid = tv.Grid(b); here = tv.coords(b)
            grass = [(bx - OFF, by - OFF) for (bx, by) in grid.grass if grid.walkable(bx - OFF, by - OFF)]
            if not grass:
                b.press("LEFT", 8, 8, render, owner="agent"); continue
            target = min(grass, key=lambda g: abs(g[0] - here[0]) + abs(g[1] - here[1]))
            if here not in grass:
                camp.trav.travel(target_map=None, arrive_coord=target, max_steps=80, max_seconds=40)
            for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                b.press(d, 8, 8, render, owner="agent")
                if st.in_battle(b):
                    break
        if out_of_balls:
            say("I'm out of Poké Balls — I'd swing by the Mart for more... (autonomous shopping isn't wired yet)")
            gaps.add("autonomous SHOP at the Cerulean Mart (Balls/Potions/Repels)")
            break

    # ── PHASE 2: head toward Misty / the Cerulean Gym (roam back into the city, explore) ──
    say(f"team's at {ram.read_party_count(b)}. Time to head for Misty's gym.")
    camp.battle_runner = roam_battle
    if tv.map_id(b) != (3, 3):
        camp.trav.travel(target_map=(3, 3), edge="east", max_steps=300, max_seconds=180)
    say("I'm back in Cerulean. The gym's around here somewhere — (autonomous gym-entry nav + NPC "
        "talking are next session).")
    gaps.add("autonomous gym-entry nav + Misty fight")
    gaps.add("NPC conversations / discovery")
    gaps.add("SELF/VOICE/WANTS driving choices (the Batch 2 personality layer)")

    # explore a little so Jonny sees free movement in the city
    for _ in range(30):
        if time.time() - t0 > MAX_SECONDS:
            break
        for d in ("UP", "RIGHT", "UP", "LEFT"):
            b.press(d, 8, 8, render, owner="agent")

    with open(os.path.join(STATES, "free_roam_end.state"), "wb") as f:
        f.write(bytes(b.save_state()))
    say(f"that's my run — team of {ram.read_party_count(b)}, in Cerulean, ready for Misty.")
    print("\n   === FREE-ROAM SUMMARY ===", flush=True)
    print(f"   final: map={tv.map_id(b)} coords={tv.coords(b)} party={ram.read_party_count(b)} "
          f"lead_hp={lead_hp_frac():.0%} elapsed={time.time()-t0:.0f}s", flush=True)
    print("   NOT-YET-WIRED (Batch 2 spec):", flush=True)
    for g in sorted(gaps):
        print(f"     - {g}", flush=True)
    print("   banked free_roam_end.state", flush=True)

    if WATCH:
        import pygame
        clock = pygame.time.Clock()
        print("   [hold] window open - close to exit", flush=True)
        try:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                b.run_frame()
                surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
                screen.blit(pygame.transform.scale(surf, win), (0, 0)); pygame.display.flip()
                clock.tick(60)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
