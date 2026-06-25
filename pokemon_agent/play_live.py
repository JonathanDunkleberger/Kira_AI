"""play_live.py - THE recordable SOUL-ON autonomous run.

The committed autonomous Brock arc (viridian_parcel_done -> heal/Forest -> Pewter ->
Brock -> Boulder Badge) with KIRA'S VOICE WIRED IN. Every battle moment, level-up,
NPC/sign line, and the badge win now flows through the EXISTING pokemon_event seam to
her self - so she's PRESENT and reacting, not grinding in silence.

This is pure WIRING - it modifies no engine/personality code. It constructs the
already-built Campaign + BattleAgent + DialogueReader and attaches reaction hooks via
their existing seams:
  - campaign on_event + beat            -> KiraVoice.emit/beat  (POST /cmd/pokemon_event)
  - every battle's on_event             -> KiraVoice.emit       (faints, win/loss, beats)
  - level-ups (read RAM around battles) -> KiraVoice.emit       (engine drains these silently)
  - DialogueReader, polled via a wrapped render -> KiraVoice.on_dialogue

VERIFY (her actual voice): start the live Kira bot (control server :8766), then run this
and watch the bot log - she reacts to battles/NPCs/level-ups and celebrates the badge IN
HER VOICE. The POST contract is identical to the already-proven `m1_battle.py --live`.
PROVE the wiring with the bot down: point --url at the capture stub (_stub_kira.py) to see
the exact event stream reach her seam.

Run:
  .venv\\Scripts\\python.exe pokemon_agent\\play_live.py                  (windowed, live bot, paced)
  .venv\\Scripts\\python.exe pokemon_agent\\play_live.py --headless --no-pace --url http://127.0.0.1:8766
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCALE = 3
LEAD_LEVEL = 0x54        # lead party-mon level byte (off GPLAYER_PARTY)


def log(m):
    print(f"   [play-live] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no window (proof runs)")
    ap.add_argument("--url", default="http://127.0.0.1:8766", help="Kira control server")
    ap.add_argument("--boot", default="viridian_parcel_done.state", help="start savestate")
    ap.add_argument("--poll-every", type=int, default=6, help="frames between dialogue polls")
    ap.add_argument("--no-pace", action="store_true",
                    help="don't hold the hands for her voice (fast proof runs; live uses pacing)")
    ap.add_argument("--fast", action="store_true",
                    help="FAST TEST (~90s): boot at Pewter and run JUST the gym (trainer + Brock + "
                         "badge) so you can feel pacing/interrupt/dialogue/music without the Forest grind")
    ap.add_argument("--audio", action="store_true",
                    help="play the emulator's game audio in real time (headphones + OBS cable)")
    ap.add_argument("--list-audio", action="store_true", help="list output devices and exit")
    ap.add_argument("--phones", default=os.getenv("POKEMON_PHONES", ""),
                    help="output device name-substring for YOUR headphones (so you hear it)")
    ap.add_argument("--cable", default=os.getenv("POKEMON_CABLE", "CABLE Input"),
                    help="output device name-substring for OBS capture (VB-Audio cable)")
    args = ap.parse_args()

    if args.list_audio:
        import pokemon_audio
        pokemon_audio.list_devices()
        return

    if args.fast and "--boot" not in sys.argv:
        args.boot = "brock_ready.state"          # Pewter, L13, just outside the gym
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from bridge import Bridge                 # noqa: E402
    import firered_ram as ram                 # noqa: E402
    import pokemon_state as st                 # noqa: E402
    import travel as tv                       # noqa: E402
    from battle_agent import BattleAgent      # noqa: E402
    from campaign import Campaign, build_objectives, STATES  # noqa: E402
    from dialogue_reader import DialogueReader  # noqa: E402
    import pokemon_voice as pv                  # noqa: E402
    from pokemon_voice import KiraVoice         # noqa: E402

    b = Bridge(ROM)
    boot_path = os.path.join(STATES, args.boot)
    if not os.path.exists(boot_path):
        log(f"FAIL - boot state missing: {boot_path}"); return
    with open(boot_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    b.set_input_owner("agent")
    log(f"booted {args.boot}: map={tv.map_id(b)} coords={tv.coords(b)}  url={args.url}")

    voice = KiraVoice(url=args.url, log=print)
    dialogue = DialogueReader(b, on_dialogue=voice.on_dialogue, state={}, log=print)

    # optional live window (so Jonny SEES the run while recording)
    screen = None
    if not args.headless:
        import pygame
        pygame.init()
        win = (b.width * SCALE, b.height * SCALE)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira plays Pokemon - SOUL ON")

    _frame = [0]

    def _draw():
        if screen is None:
            return
        import pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
        surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
        screen.blit(pygame.transform.scale(surf, (b.width * SCALE, b.height * SCALE)), (0, 0))
        pygame.display.flip()

    def _dialogue_hold(tier):
        """DIALOGUE-SYNC: when she reacts to a SALIENT line (T2/T3), reveal that text at READING
        PACE so the viewer reads what she's reacting to — instead of read-ahead-and-mash. We release
        the pad (stop the caller's fast advance), keep each page up while it's read + her voice lands,
        then advance one page, until the message buffer clears or a page budget. Pure A-advance with
        NO dialogue poll inside, so it can't re-enter; the caller's later A-mash hits a drained box."""
        if args.no_pace:
            return
        read_s = 2.6 if tier >= 3 else 1.8        # per-page dwell = viewer reading time
        max_pages = 6 if tier >= 3 else 2
        speak_cap = 8.0 if tier >= 3 else 4.5     # bound on waiting out her reaction (page 0)
        try:
            b.release(owner="agent")              # stop the caller mashing; we pace the reveal
        except Exception:
            pass
        try:
            start_buf = dialogue._read_buffer()
        except Exception:
            start_buf = None
        for page in range(max_pages):
            t0 = time.time()                      # keep the current page visible (reading time)
            while time.time() - t0 < read_s:
                b.run_frame(); _draw()
            if page == 0:                         # sync: let her one reaction beat land with the text
                t1 = time.time()
                while voice.is_speaking() and time.time() - t1 < speak_cap:
                    b.run_frame(); _draw()
            try:
                cur = dialogue._read_buffer()
            except Exception:
                cur = start_buf
            if not cur or cur != start_buf:
                break                             # message closed / advanced on its own -> done
            b.press("A", 6, 8, _draw, owner="agent")   # reveal the next page at reading pace

    def render():
        """Per-frame hook the engine already calls everywhere. We piggyback the overworld
        DialogueReader poll here (throttled by frame count); a T2+ line gets a short LANDING hold."""
        _frame[0] += 1
        if _frame[0] % args.poll_every == 0:
            try:
                fired = dialogue.poll()
            except Exception as e:
                fired = None
                print(f"   [play-live] dialogue poll error: {e}", flush=True)
            if fired and (voice.last_dialogue_tier or 0) >= 2:
                _dialogue_hold(voice.last_dialogue_tier)
        _draw()

    # POST-FIGHT HOLD lever - the "standing-still" fix, SCALED BY SALIENCE TIER (not the coarse
    # beat=True flag): a routine grind win barely pauses (she trash-talks over her shoulder and
    # keeps moving); a big win dwells. (wait-for-voice-start, hold-while-speaking, tail-frames):
    #   T1 brisk -> ~0.3s    T2 savor -> up to ~6s    T3 big -> up to ~17s (full savor)
    HOLD = {1: (0.0, 0.4, 18), 2: (2.0, 4.0, 20), 3: (3.5, 14.0, 24)}

    def pace(summary=None):
        """PERFORMANCE BEAT: hold the hands so her voice lands - but only as long as the moment
        deserves. No-op with --no-pace or a dead bot. The m1_battle --live pattern, tier-scaled."""
        if args.no_pace:
            return
        tier = max(1, voice.tier_of(summary or "")) if summary else 1
        wait_s, hold_s, tail = HOLD.get(tier, HOLD[1])
        t0 = time.time()
        while time.time() - t0 < wait_s and not voice.is_speaking():
            b.run_frame(); render()
        t1 = time.time()
        while time.time() - t1 < hold_s and voice.is_speaking():
            b.run_frame(); render()
        for _ in range(tail):
            b.run_frame(); render()

    def battle_runner():
        """Every campaign battle (wild, forest trainer, gym trainer, Brock) runs through here,
        wired to her voice. Reads SALIENCE CONTEXT from RAM the event text can't carry
        (trainer-vs-wild + a rare foe -> Tier 2), and detects level-ups + EVOLUTIONS around the
        battle (the engine drains both chains silently) -> Tier 2 / Tier 3."""
        lvl0 = b.rd8(ram.GPLAYER_PARTY + LEAD_LEVEL)
        sp0 = st.read_party_species(b, 0)
        trainer = bool(b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08)
        _rb = st.read_battle(b)
        foe = st.SPECIES_NAME.get(_rb["enemy"]["species"], "").lower() if _rb else ""
        rare = foe in pv.RARE_SPECIES
        voice.set_context(trainer=trainer, rare=rare)
        if trainer or rare:
            log(f"battle context: trainer={trainer} foe={foe or '?'} rare={rare} -> Tier 2 savor")
        out = BattleAgent(b, on_event=voice.emit, render=render,
                          pace=(None if args.no_pace else pace),
                          log=lambda m: None).run(max_seconds=180)
        voice.clear_context()
        # EVOLUTION (a species change on the lead) -> Tier 3 big beat
        sp1 = st.read_party_species(b, 0)
        if sp1 and sp1 != sp0:
            nm0 = st.SPECIES_NAME.get(sp0, "my Pokemon")
            nm1 = st.SPECIES_NAME.get(sp1, "something new")
            voice.emit(f"my {nm0} evolved into {nm1}!", kind="evolve", tier=3)
            pace("evolved into")
        lvl1 = b.rd8(ram.GPLAYER_PARTY + LEAD_LEVEL)
        if lvl1 > lvl0:
            voice.emit(f"my Pokemon just leveled up to level {lvl1}", kind="levelup", tier=2)
            pace("leveled up")
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=battle_runner, on_event=voice.emit,
                    beat=voice.beat, render=render)

    # ── GAME AUDIO (Stage 2): real-time game sound -> headphones + OBS cable -> Kira hears it too ──
    audio = None
    if args.audio:
        try:
            import pokemon_audio
            audio = pokemon_audio.AudioPump(b, phones=args.phones, cable=args.cable, log=print)
        except Exception as e:
            log(f"!! AUDIO disabled — could not start ({e}). Run will continue SILENT.")

    # ── objectives: FAST gym-only test, or the full arc ──
    if args.fast:
        objectives = [("BEAT_GYM", "Brock", "FAST TEST: Pewter Gym -> Brock -> Boulder Badge")]
        voice.emit("alright, the gym's right here - Brock's Boulder Badge is mine", kind="intro", tier=2)
    else:
        objectives = build_objectives()
        # neutral framing opener (a game moment, not a script); tier=2 so "Boulder Badge" doesn't false-T3.
        voice.emit("you're back on the road - next stop is Pewter Gym and Brock's Boulder Badge",
                   kind="intro", tier=2)

    try:
        outcome = camp.run(objectives)
    except KeyboardInterrupt:
        outcome = "window-closed"
    finally:
        if audio is not None:
            audio.close()
        if screen is not None:
            import pygame
            pygame.quit()

    print("\n" + "=" * 64, flush=True)
    print(f"   PLAY-LIVE RESULT: {outcome}", flush=True)
    print(f"   final: map={tv.map_id(b)} coords={tv.coords(b)}  "
          f"badge={camp.has_boulder_badge()}", flush=True)
    voice.report()
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
