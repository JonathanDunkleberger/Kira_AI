"""recon_opening.py - watch the AUTONOMOUS OPENING (the personality debut) at TRUE stream speed.

Fresh ROM boot -> title -> New Game -> Kira names herself + her rival (her choice) -> picks gender ->
bedroom -> downstairs -> Pallet -> Oak intercept -> his lab -> PICKS her starter (her choice) -> the
rival battle -> walks out. All driven by campaign.drive_opening (capability-not-script).

RUN (the watch): WATCH=1 AUDIO=1 .venv\\Scripts\\python.exe -u pokemon_agent\\recon_opening.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
WATCH = os.getenv("WATCH", "1") == "1"
AUDIO = os.getenv("AUDIO", "1") == "1"
INVINCIBLE = os.getenv("INVINCIBLE", "0") == "1"
PHONES = os.getenv("POKEMON_PHONES", "Leviathan")
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge                       # noqa: E402
import firered_ram as ram                       # noqa: E402
import pokemon_state as st                       # noqa: E402
from battle_agent import BattleAgent            # noqa: E402
from dialogue_reader import DialogueReader, decode  # noqa: E402
from campaign import Campaign                    # noqa: E402

# HER CHOICES (capability-not-script: Batch 2 routes these through her soul; here, sensible debut picks)
PLAYER_NAME = os.getenv("PLAYER_NAME", "KIRA")
RIVAL_NAME = os.getenv("RIVAL_NAME", "GARY")
GIRL = os.getenv("GIRL", "1") == "1"
STARTER = int(os.getenv("STARTER", "0"))         # 0 Bulbasaur / 1 Charmander / 2 Squirtle


def main():
    b = Bridge(os.path.join(os.path.dirname(_HERE), "roms", "firered.gba"))
    for _ in range(400):
        b.run_frame()
    b.set_input_owner("agent")

    audio = None
    if AUDIO:
        try:
            import pokemon_audio
            audio = pokemon_audio.AudioPump(b, phones=PHONES, log=print)
            print(f"   [opening] AudioPump ON (phones~{PHONES!r}) - real-time + music", flush=True)
        except Exception as e:
            print(f"   [opening] !! AudioPump failed ({e}) - 60fps cap, silent", flush=True)

    screen = win = clock = None
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - the autonomous OPENING (debut)")
        clock = pygame.time.Clock()

    w16 = b.core.memory.u16.raw_write

    def render():
        if INVINCIBLE and st.in_battle(b):
            try:
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))
            except Exception:
                pass
        if WATCH:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
            if audio is None:
                clock.tick(60)

    dr = DialogueReader(b)
    for k in range(80):                          # title -> New Game -> Oak's intro
        if "welcome" in dr._read_buffer().lower() or "boy" in dr._read_buffer().lower():
            break
        b.press("A" if k % 2 else "START", 4, 8, render, owner="agent")
        for _ in range(10):
            b.run_frame()
            render()

    def battle_runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                          log=lambda m: print(f"      {m}", flush=True)).run(150)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=battle_runner,
                    on_event=lambda s, **k: print(f"   [kira] {s}", flush=True),
                    beat=lambda *a, **k: None, render=render)
    final = camp.drive_opening(PLAYER_NAME, RIVAL_NAME, girl=GIRL, starter_choice=STARTER)

    sb2 = b.rd32(ram.GSAVEBLOCK2_PTR)
    nm, _ = decode(b.read_bytes(sb2 + 0, 8))
    sp = st.read_party_species(b, 0) if b.rd8(ram.GPLAYER_PARTY_CNT) > 0 else 0
    print(f"\n   *** OPENING DONE: map={final} player={nm!r} starter={st.SPECIES_NAME.get(sp, '?')} "
          f"party={b.rd8(ram.GPLAYER_PARTY_CNT)} ***", flush=True)

    if WATCH:
        import pygame
        print("   [hold] window open - close to exit", flush=True)
        try:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                b.run_frame()
                render()
        except KeyboardInterrupt:
            pass
    if audio is not None:
        try:
            audio.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
