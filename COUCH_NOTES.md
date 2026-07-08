# COUCH_NOTES.md — the couch session ledger (2026-07-08)

The first human watch of the Pokémon showcase, with fixes applied same-day. Jonny at the
couch; Claude driving boots, watches, and fixes. Newest context last.

## What the session established
- **Machine layer graded CLEAN.** All 15 descent arcs PASS; the F-5 throwaway ran the opening
  end-to-end (naming, starter, parcel) with zero mechanical faults Jonny flagged. Every note
  was PERSONA direction, not bugs.
- **Two throwaway runs watched.** Run 1 → Fix Pass 1 (the persona dial). Run 2 → two targeted
  items (parcel stall + mic). General pace judged "fine" on run 2.

## FIX PASS 1 — THE PERSONA DIAL (shipped: 560ac89)
- **P-1 First-playthrough energy.** (a) FIRSTS-AS-EVENTS — first wild battle / trainer fight /
  catch / Center each get one T2 beat (world-model persisted, badge-gated so a veteran world
  backfills silently). (b) GYM FOLKLORE — "word on the road" about the next leader rides the
  spine ctx (anticipation habit; folklore, never walkthrough). (c) Showcase speech budget —
  show-mode voice dials FLOOR 0.8 / GRIND_GAP 1.2 / AMBIENT_GAP 3.0 (module dials; workshop
  untouched).
- **P-2 Social warmth.** Mom is a MUST-STOP on journey day one — a real goodbye beat anchored
  in drive_opening (the F-6 salience only rode free-roam ticks; the opening walks that floor
  once, so the stop is anchored there).
- **P-3 Battle streamer layer.** Battle-start STAKES beat (trainer/rare, team-shape aware);
  END-EXHALE T1 when no bigger beat (clutch/faint/evolve/level) owned the ending. In-fight
  focus unchanged. Crit/super-effective mid-battle deferred (no clean RAM signal this pass).
- **P-4 BANKED** → POST_CREDITS_VISION.md §7: the "played like Nintendo intended" full-
  completionist showcase FORMAT (guides + secrets + full dex energy) — a future episode format.
- **P-5 Self-canon.** Charmander is 3-for-3 across independent sessions → recorded as durable
  identity (SELF_CANON in pokemon_soul); rides the starter ctx as HER remembered taste (she
  still chooses; nothing overrides it).
- **P-6 Pace telemetry** shipped in dialogue_drive (per-box visible wall-time). See below.

## ROUND 2 — the two targeted items (shipped)
- **R2-1 Parcel-handover stall — FIXED (a61f922).** Telemetry proved the boxes render fast
  (2.5–2.84s each). The ~25s was her VOICING every box as a "you read:" line — four back-to-
  back, incl. two pure system boxes ("received parcel", "put in pocket"), each a TTS round-trip
  in show mode. Fix: on_dialogue no longer voices pure system/bookkeeping boxes (received /
  put-in-pocket / obtained / found); they still advance at reading pace. Regex probe: catches
  the mechanical boxes, 0 false-positives on real NPC dialogue. Serves the CEO pacing law
  ("as fast as a human reader could follow, never slower").
- **R2-2 "Kira can't hear Jonny" — DIAGNOSED; not what it looked like.** The MIC works (63
  voice-onsets, 32 real transcriptions of his speech; she replied). The dead path was the
  LOOPBACK STT: his `Headphones (Leviathan)` Bluetooth device was disconnected at boot, so
  loopback fell back to the VB-Audio CABLE — which in his rig carries HER OWN TTS. Her ear was
  pointed at her own mouth (the "hearing yourself" chaos he called out live). Root cause =
  HARDWARE (Bluetooth headset drops → its mic AND the loopback source vanish); the system
  already loud-logs it (8 warnings). **GO-LIVE RITUAL:** confirm the audio device is present/
  selected BEFORE booting the bot. Jonny reboots on his Focusrite and runs the live "cheddar"
  mic check himself. **FLAGGED, not cowboyed:** whether loopback should refuse to fall back
  onto the TTS-carrying cable is rig-specific + touches core-Kira's sacred perception path
  (HARD CONSTRAINT #2) — left for Jonny's decision; no core code touched.

## MACHINE ITEM (noted, deprioritized by Jonny)
- Segment-driver recovery rung (560ac89) did NOT fire at Route 1 (13,0): it's a genuine
  no-route boundary, not an impossible-stand, so a different class than the rung covers. Jonny
  is moving to later-game spot-watches rather than re-running the intro. Filed to the couch list.

## CLOSE-OUT BUILDS (this session)
- **P-7 OpenAI purge — COMPLETE (a1d54f2).** Last consumer (audio mood agent) neutered;
  dependency + key removed; repo grep clean; smoke-boots with no key and with openai force-
  uninstalled. Audio MOOD perception now inactive (flagged) — desktop SPEECH hearing unaffected.
  Jonny to revoke the old key + kill auto-recharge (belt-and-suspenders).
- **Save-file card — SHIPPED (db8c83b).** Game Boy / FireRed "CONTINUE" screen at
  `/pokemon_savecard`, presentation over the existing /pokemon_hud.json (badges / Pokédex /
  play time / location / party). Route live on next bot boot.
- **Clipper final spec + Phase-J receipts — CONFIRMED wired** (392b872): 10 ranked shorts /
  1200s superfan / 300s midform; cost_tracker.write_receipt → logs/receipts/ + LEDGER.jsonl.

## GO-LIVE RITUAL (permanent)
Before booting the bot for a live session: **confirm the audio input/output device is present
and selected** (esp. after a Bluetooth drop) — the loopback binds at boot and won't hot-rebind
a vanished device. Then boot, then run a quick mic check.
