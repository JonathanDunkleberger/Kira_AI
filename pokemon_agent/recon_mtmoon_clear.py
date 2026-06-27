"""recon_mtmoon_clear.py - FULL visible Mt Moon clear, entrance -> Cerulean side, with the scripted
FOSSIL scene handled. Warp chain (disasm-verified):
  1F(15,30) --(5,6)--> B1F(3,3) --(22,18)--> B2F(25,21)
  ... B2F traversal (trainer gauntlet handled by travel) up to the fossil room ...
  GRAB a fossil (Dome@(13,7) or Helix@(14,7)) by standing adjacent + A -> Miguel@(13,11) steps aside
  --(5,10)--> B1F(39,4) --(45,4)--> ROUTE4 -> cross east -> Cerulean
Miguel is a SCRIPTED blocker (not a trainer: talking loops); only taking a fossil clears him.
RUN (visible): WATCH=1 INVINCIBLE=1 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_mtmoon_clear.py
"""
import os
import sys
import ctypes

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "1") == "1"
INVINCIBLE = os.getenv("INVINCIBLE", "1") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402
from cave_nav import CaveNav          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
GH = 0x02036DFC
FOSSILS = [((13, 7), (13, 8), "UP"), ((14, 7), (14, 8), "UP")]   # (fossil, stand-tile, face-dir)
MIGUEL = (13, 11)


def main():
    start = os.getenv("START", "mtmoon_interior")   # boot near the end (e.g. mtmoon_endgame) to
    b = Bridge(ROM)                                  # iterate the fossil+exit WITHOUT re-running the cave
    from campaign import resolve_state               # states moved into lineage buckets (archive sweep)
    with open(resolve_state(start + ".state") or os.path.join(STATES, start + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write

    def heal():
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))
    heal()

    screen = win = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - Mt Moon FULL clear (watch)")

    def render():
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
        if INVINCIBLE and st.in_battle(b):
            w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))
        if WATCH:
            import pygame
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()

    def fr():
        agent = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                            log=lambda m: print(f"      {m}", flush=True))
        if INVINCIBLE:
            agent.run(90); heal(); return "win"           # nav-isolation: win all + free heal
        # SURVIVAL (thin party, no heal-return in Mt Moon): FLEE wild Zubats to save HP, WIN forced
        # trainers (flee() delegates trainers to run()). NO between-battle heal - real roster test.
        r = agent.flee(90)
        return "loss" if r == "loss" else "win"
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=render)
    camp._suppress_heal = True
    nav = CaveNav(b, camp, fog_path=None, on_event=lambda *a, **k: None, render=render,
                  log=lambda m: print(m, flush=True))

    def npc_tiles():
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if b.rd8(o) & 1:
                out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET, b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def step(p):
        print(f"\n=== {p} ===", flush=True)

    # --- warp chain to B2F (skipped when booting from a B2F endgame checkpoint) ---
    if tv.map_id(b) != (1, 3):
        step("1F -> B1F via (5,6)")
        if nav._enter((5, 6)) is None:
            print("   ABORT: (5,6) unenterable"); return
        step("B1F -> B2F via (22,18)")
        if nav._enter((22, 18)) is None:
            print("   ABORT: (22,18) unenterable"); return
    print(f"   on {tv.map_id(b)}@{tv.coords(b)} (expect B2F (1,3))", flush=True)

    # --- approach a fossil and grab it (clears Miguel) ---
    step("B2F: approach a fossil, grab it to clear Miguel")
    grabbed = False
    for fossil, stand, facing in FOSSILS:
        print(f"   trying fossil {fossil}: travel to stand-tile {stand}", flush=True)
        r = camp.trav.travel(target_map=None, arrive_coord=stand, max_steps=600, max_seconds=300)
        here = tv.coords(b)
        print(f"   travel -> {r}; now at {here} (Miguel at {MIGUEL} present={MIGUEL in npc_tiles()})", flush=True)
        if here == stand:
            with open(os.path.join(STATES, "mtmoon_endgame.state"), "wb") as f:
                f.write(bytes(b.save_state()))
            print(f"   *** banked mtmoon_endgame.state at fossil stand-tile {stand} "
                  f"(boot here with START=mtmoon_endgame to iterate fast) ***", flush=True)
            print(f"   at stand-tile; pressing {facing} + A to grab fossil {fossil}", flush=True)
            b.press(facing, 8, 8, render, owner="agent")
            for _ in range(10):                      # mash A to grab + advance the script dialogue
                b.press("A", 8, 8, render, owner="agent")
            for _ in range(40):
                b.run_frame(); render()
            print(f"   after grab: Miguel present={MIGUEL in npc_tiles()} npcs={sorted(npc_tiles())}", flush=True)
            grabbed = True
            break
    if not grabbed:
        print("   COULD NOT REACH A FOSSIL - reporting position, no phantom", flush=True)

    # --- continue out: (5,10) -> B1F(39,4) -> (45,4) -> Route 4 -> Cerulean ---
    step("B2F -> B1F via (5,10)")
    if nav._enter((5, 10)) is None:
        print(f"   ABORT: (5,10) still unenterable at {tv.coords(b)} (Miguel cleared? {MIGUEL not in npc_tiles()})")
        _hold(b, screen, win); return
    print(f"   on {tv.map_id(b)}@{tv.coords(b)} (expect B1F (1,2) near (39,4))", flush=True)
    step("B1F -> Route4 via (45,4) [FORWARD EXIT]")
    nm = nav._enter((45, 4))
    print(f"   _enter((45,4)) -> {nm} now {tv.map_id(b)}@{tv.coords(b)}", flush=True)
    if tv.map_id(b) == (3, 22):
        step("emerged on Route 4 - crossing EAST to Cerulean")
        camp.trav.travel(target_map=(99, 99), edge="east", max_steps=400, max_seconds=200)
    final = tv.map_id(b)
    print(f"\n   *** FINAL: map={final} coords={tv.coords(b)} ***", flush=True)
    if final[0] != 1 and final != (3, 22):
        with open(os.path.join(STATES, "cerulean_entered_auto.state"), "wb") as f:
            f.write(bytes(b.save_state()))
        print(f"   *** CLEARED Mt Moon -> emerged on {final}; banked cerulean_entered_auto.state ***", flush=True)
    else:
        print(f"   not fully out (map={final}) - report, no phantom", flush=True)
    _hold(b, screen, win)


def _hold(b, screen, win):
    if not WATCH:
        return
    import pygame
    print("   [hold] window OPEN for inspection (close it to exit)", flush=True)
    clock = pygame.time.Clock()
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            b.run_frame()
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
