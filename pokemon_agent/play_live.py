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

# CLIMAX / scripted-cutscene markers: the post-Brock award + TM gift is a MULTI-BOX cutscene the
# campaign's own A-drain (beat_gym) rips through. The reading-pace dialogue-hold must NOT pace these
# — stacking a multi-second hold per box froze the badge moment ~100s (looked hung). On any of these
# the hold yields immediately and lets the scripted drain land the climax. (Item 4.)
CUTSCENE_MARKERS = (
    "boulderbadge", "boulder badge", "take this", "technical machine", "tm39", "tm 39",
    "rock tomb", "teaches", "as proof", "proof of", "received", "officially", "use it on",
)


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
    ap.add_argument("--go", action="store_true",
                    help="CONTINUOUS RUN (WORKSHOP): play the resumable SEGMENT MANIFEST "
                         "(campaign.build_segments) — resume-from-furthest in states/workshop/, "
                         "bank there; never writes states/kira/")
    ap.add_argument("--show", action="store_true",
                    help="SHOW MODE: the canonical zero-skip spine from a FRESH boot (or resume from a "
                         "states/kira/ save). Banks progress into states/kira/. Any GATE_NEEDS_STATE "
                         "fallback logs a LOUD SHOW-MODE SKIP VIOLATION — a clean run has ZERO.")
    ap.add_argument("--free-roam", action="store_true",
                    help="FREE ROAM (Batch-2 soul): boot the --boot state and RELEASE control — she "
                         "decides her own next move each tick via the soul oracle (wander_catch / battle "
                         "/ heal / head_to_gym), state-aware, unscripted. Use with --boot misty_done.state.")
    ap.add_argument("--roam-ticks", type=int, default=12, help="free-roam: max decision ticks")
    ap.add_argument("--resume", action="store_true",
                    help="PERSISTENT CAMPAIGN (Batch 5 — the Sherpa-timeline GO): resume her LIVING "
                         "campaign save (states/campaign/kira_campaign.state) so she picks up the CLIMB "
                         "from where she actually is, never rappelled to a frozen fragment. First run "
                         "with no campaign save yet SEEDS from --boot. Pair with --free-roam.")
    ap.add_argument("--fresh-kira", action="store_true",
                    help="With --show: timestamp-archive the current states/kira/ playthrough to "
                         "states/kira/archive_<ts>/ and START A NEW Kira run from the bedroom (never "
                         "clobbers her save). Without it, --show RESUMES the existing Kira run.")
    ap.add_argument("--audio", action="store_true",
                    help="play the emulator's game audio in real time (headphones + OBS cable)")
    ap.add_argument("--list-audio", action="store_true", help="list output devices and exit")
    ap.add_argument("--phones", default=os.getenv("POKEMON_PHONES", ""),
                    help="output device name-substring for YOUR headphones (so you hear it)")
    ap.add_argument("--cable", default=None,
                    help="DEPRECATED/UNUSED — game music no longer routes to the virtual cable (the "
                         "cable carries ONLY Kira's voice so VTS doesn't lip-sync the soundtrack). "
                         "Music goes to --phones (desktop); OBS captures desktop audio directly.")
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
    from campaign import Campaign, build_objectives, build_segments, STATES  # noqa: E402
    from dialogue_reader import DialogueReader  # noqa: E402
    import pokemon_voice as pv                  # noqa: E402
    from pokemon_voice import KiraVoice         # noqa: E402

    b = Bridge(ROM)
    if args.show:
        from campaign import archive_kira_save, kira_checkpoints
        if args.fresh_kira:
            arch = archive_kira_save(time.strftime("%Y%m%d_%H%M%S"))
            log(f"SHOW MODE: --fresh-kira -> archived previous Kira run to {arch}" if arch
                else "SHOW MODE: --fresh-kira -> no existing Kira save to archive; starting fresh")
        cps = kira_checkpoints()
        log(f"SHOW MODE: {'RESUMING Kira run from ' + str(cps[-1]) if cps else 'starting a NEW Kira run (fresh bedroom)'}")
        # SHOW = canonical spine. Do NOT load a sherpa boot state; start from a FRESH ROM (segment 0
        # the_opening drives title->New Game->bedroom). run_segments(mode='show') will instead resume
        # from a states/kira/ checkpoint if one exists. This is the no-skip, no-hand-bank path.
        for _ in range(40):
            b.run_frame()
        b.set_input_owner("agent")
        log(f"SHOW MODE: fresh ROM boot (canonical spine): map={tv.map_id(b)} url={args.url}")
    else:
        from campaign import resolve_state, STATES_CAMPAIGN, CAMPAIGN_SAVE
        camp_path = os.path.join(STATES_CAMPAIGN, CAMPAIGN_SAVE)
        if args.resume and os.path.exists(camp_path):
            # RESUME the climb (Batch 5 P1): load her living campaign anchor — she continues from where
            # she actually is, never reset to a fragment.
            with open(camp_path, "rb") as f:
                b.load_state(f.read())
            for _ in range(40):
                b.run_frame()
            b.set_input_owner("agent")
            log(f"⛰️  RESUMING CAMPAIGN from campaign/{CAMPAIGN_SAVE}: map={tv.map_id(b)} "
                f"coords={tv.coords(b)} — she keeps climbing from here  url={args.url}")
        else:
            boot_path = resolve_state(args.boot)
            if not boot_path:
                log(f"FAIL - boot state missing: {args.boot} "
                    f"(searched campaign/workshop/kira/states/archive)"); return
            with open(boot_path, "rb") as f:
                b.load_state(f.read())
            for _ in range(40):
                b.run_frame()
            b.set_input_owner("agent")
            seed = (" — SEEDING the campaign (first climb from here; progress anchors as she goes)"
                    if args.resume else "")
            log(f"booted {args.boot}: map={tv.map_id(b)} coords={tv.coords(b)}{seed}  url={args.url}")

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
    _holding = [False]        # reentrancy guard: never stack dialogue-holds (the ~100s climax freeze)

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

    def _dialogue_hold(tier, line=""):
        """DIALOGUE-SYNC: when she reacts to a SALIENT line (T2/T3), reveal that text at READING
        PACE so the viewer reads what she's reacting to — instead of read-ahead-and-mash.

        TUNED (items 2+4): dwell ONLY while there's MORE text to read — land the page at reading
        pace, advance, and STOP the instant a box has no next page (no fixed multi-second freeze on
        a line that's already drained). A reentrancy guard + a hard total cap mean a hold can never
        STACK or run long; the scripted CLIMAX cutscene (badge/TM award) is skipped outright so the
        campaign's own A-drain lands it (this is what froze the badge moment ~100s before)."""
        if args.no_pace or _holding[0]:
            return
        # CLIMAX guard: never reading-pace the badge/TM award cutscene — yield to the scripted drain.
        try:
            buf_now = dialogue._read_buffer() or ""
        except Exception:
            buf_now = ""
        blob = f"{line} {buf_now}".lower()
        if any(m in blob for m in CUTSCENE_MARKERS):
            print("   [play-live] dialogue-hold: climax cutscene — yielding to the scripted "
                  "award/TM drain (no reading-pace hold)", flush=True)
            return
        _holding[0] = True
        try:
            read_s = 1.6 if tier >= 3 else 1.1    # per-page dwell = viewer reading time (snappier)
            max_pages = 4 if tier >= 3 else 2
            cap = 5.0 if tier >= 3 else 3.0       # HARD total backstop — a hold can't run long
            try:
                b.release(owner="agent")          # stop the caller mashing; we pace the reveal
            except Exception:
                pass
            try:
                prev = dialogue._read_buffer()
            except Exception:
                prev = None
            t_start = time.time()
            for _page in range(max_pages):
                t0 = time.time()                  # land THIS page at reading pace (or until capped)
                while time.time() - t0 < read_s and time.time() - t_start < cap:
                    b.run_frame(); _draw()
                if time.time() - t_start >= cap:
                    break
                b.press("A", 6, 8, _draw, owner="agent")   # advance one page
                try:
                    cur = dialogue._read_buffer()
                except Exception:
                    cur = prev
                if not cur or cur == prev:
                    break                         # no more text -> land it, don't freeze on it
                prev = cur
        finally:
            _holding[0] = False

    def render():
        """Per-frame hook the engine already calls everywhere. We piggyback the overworld
        DialogueReader poll here (throttled by frame count); a T2+ line gets a short LANDING hold."""
        _frame[0] += 1
        # CLIMAX: while beat_gym drains the scripted badge/TM award, keep the window LIVE but do NOT
        # poll/hold dialogue — let the proven A-drain land the badge (mirror headless; item 4 fix).
        if getattr(camp, "draining_award", False):
            _draw()
            return
        if _frame[0] % args.poll_every == 0:
            try:
                fired = dialogue.poll()
            except Exception as e:
                fired = None
                print(f"   [play-live] dialogue poll error: {e}", flush=True)
            if fired and not _holding[0] and (voice.last_dialogue_tier or 0) >= 2:
                _dialogue_hold(voice.last_dialogue_tier, voice._last_summary or "")
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
                          choose=voice.choose,          # PART B: in-battle "use your items" instinct -> her
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
                    beat=voice.beat, render=render, choose=voice.choose)

    # ── GAME AUDIO (Stage 2): real-time game sound -> headphones + OBS cable -> Kira hears it too ──
    audio = None
    if args.audio:
        try:
            import pokemon_audio
            audio = pokemon_audio.AudioPump(b, phones=args.phones, cable=args.cable, log=print)
        except Exception as e:
            log(f"!! AUDIO disabled — could not start ({e}). Run will continue SILENT.")

    # ── objectives: continuous SEGMENT MANIFEST (--go), FAST gym-only test, or the full single arc ──
    objectives = None
    if args.go or args.show:
        voice.emit("alright — continuous run. badge by badge, all the way.", kind="intro", tier=2)
    elif args.fast:
        objectives = [("BEAT_GYM", "Brock", "FAST TEST: Pewter Gym -> Brock -> Boulder Badge")]
        voice.emit("alright, the gym's right here - Brock's Boulder Badge is mine", kind="intro", tier=2)
    elif args.free_roam:
        # FREE ROAM: no scripted intro — her FIRST surface_want is her opening beat (soul-true).
        pass
    else:
        objectives = build_objectives()
        # neutral framing opener (a game moment, not a script); tier=2 so "Boulder Badge" doesn't false-T3.
        voice.emit("you're back on the road - next stop is Pewter Gym and Brock's Boulder Badge",
                   kind="intro", tier=2)

    try:
        if args.show:
            outcome = camp.run_segments(build_segments(), mode="show")
        elif args.go:
            outcome = camp.run_segments(build_segments(), mode="workshop")
        elif args.free_roam:
            outcome = camp.free_roam(max_ticks=args.roam_ticks)
        else:
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
