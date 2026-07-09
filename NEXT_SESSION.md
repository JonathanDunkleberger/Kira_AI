# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-08)

**MISSION (the deliverable):** a **clean bedroom -> credits** FireRed run, autonomous, at
**human-realistic pace (~25-35h, NOT 67h)**, **no permanent stucks, no crash-loops**, AND
**game audio ON (`POKEMON_GAME_AUDIO=1`) + her voice/TTS ON** — watchable and recordable as
evergreen content. **A silent playthrough is NOT the deliverable.** Audio-off is a MITIGATION and a
FALLBACK FLOOR, not the finish line: the native audio crash must be genuinely diagnosed and fixed,
not sidestepped. You are the lead Sherpa. Employment terms + standing rules 1-18 in CLAUDE.md apply.

## LIVE STATUS (night-train shift 1, updated in-flight)

- **PHASE A item 1 — PARCEL WEDGE FIXED + COMMITTED (`cfb353d`).** ROOT CAUSE (not staleness, not
  ledges — both ruled out by probes): `travel()` computed the connection `band` (the edge-crossing
  columns) ONCE for the STARTING map, but `deliver_parcel` calls `travel(target=PALLET, edge='south')`
  from Viridian and that ONE leg spans Viridian->Route1->Pallet. The band (Viridian's south cols
  [0-11,22-25,36-48]) was never recomputed after the mid-travel map transition, so on Route 1 the BFS
  goal required Viridian's columns while Route 1's Pallet-exit cols are 12,13 -> false `no_route` ->
  TRAVEL WEDGE x4 at (3,19)(13,0). FIX (general, kills the whole class): `_compute_band(grid)` helper,
  recomputed at every map transition inside `travel()`. VERIFIED: `recon_route1_stale_probe.py` drives
  the real Viridian->south leg headless -> pre-fix wedges, post-fix logs band [0-11,22-25,36-48] ->
  [12,13,24] and ARRIVES at Pallet. Also confirmed LIVE in the fresh-spine run (band recomputes at the
  Pallet->Route1 transition). Standing probes: `recon_route1_south_probe.py`, `recon_route1_stale_probe.py`.
- **PHASE A item 3 — FRESH-SPINE LOOK-AHEAD running.** Added a `FRESH` boot branch to `recon_longrun.py`
  (`python pokemon_agent/recon_longrun.py FRESH <minutes>`) that boots a clean ROM and runs the real show
  SPINE (`run_segments(the_opening -> ... -> Misty)`, mode=workshop, canonical protected via STAGE) —
  the faithful early-game look-ahead. Reuses the harness's chooser/runner. Log:
  `logs/debug/freshspine.log`. Drove opening -> starter (Bulbasaur default headless) -> rival -> Pallet
  -> Route 1 cleanly. WATCHING for the next early wedge.
- **NEXT WEDGE FOUND + DEEPLY DIAGNOSED (the frontier): VIRIDIAN FOREST is unbuilt in the show spine.**
  After the parcel, the fresh spine (SEGMENT 2 `pallet_to_brock`, OBJECTIVE `ADVANCE_NORTH` Viridian->
  Route2->Forest->Pewter) LOOPS forever: Viridian -> cross to Route 2 (lands south edge (10,79)) ->
  `advance_north` finds no north EDGE (the way north is through the Viridian Forest GATE BUILDINGS, a
  warp, not an edge) -> `enter_warp(prefer="north")` -> warps into a Forest gate (15,x) -> Viridian
  Forest (1,0) -> can't find the Forest's NORTH exit -> gets hurt -> HEAL back to Viridian -> repeats
  (bounded by max_legs=60 -> "timeout" -> segment STOPs loud; not a true freeze, but zero progress).
  KEY DATA (probe `recon_viridian_north_probe.py`, postgame): from Route 2 south entry (10,79) the land
  flood reaches only y=46 (a wall) — she is confined to Route 2's south half; the way north is the Forest.
  Route 2's REACHABLE forward warps are `(18,46)/(19,46) -> (15,2)` and `(5,51)/(6,51) -> (15,0)` (the
  Forest gates); the y=13/41 warps are north of the y=46 wall = unreachable (correctly skipped). So she
  DOES get into the Forest (1,0); the gap is **navigating Viridian Forest to its north exit gate** — a
  MAZE the current `advance_north` (dumb travel-north / warp-north) can't solve.
  WHY NO GRAPH FALLBACK: the canonical world graph (`states/campaign/world_model.json`) has **edges=None**
  for ALL early nodes (Route 1 `3,19`, Route 2 `3,20`, Pewter `3,2`, Forest `1,0`) — the early climb was
  never edge-recorded, so `world.next_hop` can't route Route2->Pewter. The Forest needs REAL maze nav.
  FIX DIRECTIONS for next shift (build, don't rush overnight): (a) a CURATED Forest route (Mt-Moon
  `run_plan` style: Route2 south -> the correct gate -> Forest -> north gate -> Route2 north -> Pewter,
  tile-by-tile from a manual drive + the pret disasm `ViridianForest`/`Route2` warp_events), OR (b) a
  general Forest/maze traversal that chains warp-to-warp toward the target (needs a "which gate advances"
  signal since `prefer=north` alone loops). Verify via `python pokemon_agent/recon_longrun.py FRESH 40`
  (the fresh-spine look-ahead I added) -> it should clear the Forest and reach Pewter/Brock.
  SEPARATE FINDING (honest): `build_segments()` scripts the SHOW spine only through **Misty (badge 2)** —
  `play_live --show` -> `run_segments(build_segments(), mode="show")` has NO segments past Misty, so the
  fresh bedroom->credits SHOW path is unbuilt beyond badge 2 regardless of the Forest. The canonical
  Sherpa credits were rolled via `recon_longrun` free_roam + checkpoints, NOT this show spine.
- THEN: Phase B (bank the clean audio-OFF floor) + Phase C (native audio SIGSEGV capture via faulthandler
  with `--audio`) remain untouched this shift.

## SEQUENCING — BANK THE SAFE WINS FIRST, RISKY WORK LAST (do in this order)

The audio crash is uncertain deep work; it must NOT eat the night and leave nothing. So order the
work so the guaranteed wins are banked BEFORE the risky work, and the risky work can't destroy them:

### PHASE A — NAV: fix the wedge + general resilience (SAFE, HIGH-VALUE) -> commit
1. **Fix the Route 1 -> Pallet parcel-delivery wedge.** On a fresh run she wedges delivering Oak's
   Parcel: `no_route genuine wall/zone gap` at **map (3,19) coord (13,0)** (Route 1 sub-area /
   boundary), TRAVEL WEDGE x4 -> campaign STOP. Known "Route-1 boundary wedge, fresh-spine intro
   only" class the impossible-stand rung doesn't cover. Fix via the warp-graph nav harness
   (`recon_longrun.py`, 14.3x headless): reproduce, instrument the connection-band / warp routing at
   (3,19)->(3,0/Pallet), fix the specific no-route, RE-RUN, confirm parcel delivery completes.
2. **HARDEN NAVIGATION GENERALLY — fix the CLASS, not the instance.** Don't just patch this one wedge:
   build resilience so she doesn't hit a novel stuck on stream tomorrow. Add/strengthen: early
   stuck-state DETECTION (fingerprint repeats, no-net-progress), robust RECOVERY + RE-ROUTE (back
   off, try alternate bands/warps, widen search), and class-level resilience to the wall type. Solve
   once, reuse everywhere. Commit each fix.
3. Drive forward via 14.3x headless look-ahead; when it stalls on the NEXT wedge, DIAGNOSE -> FIX
   (general) -> RE-RUN -> BANK a checkpoint -> advance. Log every stuck/crash/wedge as a discrete
   note -> fix -> revalidate item in NIGHT_REPORT.md (commit-per-fix = proof of life).

### PHASE B — PROVE THE AUDIO-OFF FULL PATH (THE GUARANTEED FLOOR) -> commit
4. **Demonstrate a clean bedroom -> credits path with audio OFF** via the 14.3x headless look-ahead
   (zero permanent stucks / crash-loops), and BANK it. **This is Jonny's guaranteed floor: worst
   case he wakes up to a silent-but-COMPLETE, working run.** Commit. Do NOT proceed to the risky
   audio work until this floor is banked.

### PHASE C — THE NATIVE AUDIO CRASH: capture -> diagnose -> fix (DEEP WORK, LAST) -> commit
5. **CAPTURE FIRST. Do NOT theorize about the cause until you have a real crash artifact.** The
   reason prior attempts failed: the crash was never captured — it's a **native SIGSEGV in the
   SDL/PortAudio/pygame layer** (no Python traceback, no dump) and it always fired BEFORE faulthandler
   was armed. faulthandler is now armed from module-import on every play_live path (`play_live.py:52`)
   -> `logs/debug/playlive_faulthandler.log`. So: **reproduce the crash DELIBERATELY with audio ON**
   (`POKEMON_GAME_AUDIO=1` -> `--audio` -> AudioPump/PortAudio output active — the LIVE path, since
   headless `repro_parcel_crash.py` did NOT repro; it's live-output-only) **and faulthandler + any
   native debugging armed, and CAPTURE the actual dump / C-stack.** Drive through the previously-
   fingered crash points (esp. the **Viridian Mart item-get fanfare** at parcel pickup).
6. **From the captured evidence, find the TRUE root cause** (audio-device init? buffer/callback
   threading? a specific fanfare/sfx trigger? sample-rate/driver mismatch? PortAudio output-device
   binding?) and **fix it at the source** — proper audio init/teardown, a safer backend/driver,
   buffering, or **isolating audio into a subprocess so a native fault can't kill the run**.
7. **PROVE the fix, VERIFIED not asserted:** a real stretch of gameplay with **audio ON** through the
   fingered crash points (Viridian fanfare + a few more sfx/fanfare triggers) with **zero native
   crashes**, confirmed from logs/replay. Then land **audio-ON stable on top of the already-working
   run** (Phase B floor stays intact underneath).
8. **FALLBACK RULE (only if real diagnosis is exhausted):** if you cannot make audio-on stable
   tonight, fall back to audio-OFF so the run still completes, and write an EXPLICIT NIGHT_REPORT.md
   note: what you CAPTURED (the dump / stack), what you TRIED, and what's NEEDED — so Jonny wakes to
   real evidence, not another guess. Never sidestep silently.

**STOP CONDITIONS:** (a) a clean bedroom -> credits with audio ON is demonstrated (the full
deliverable); OR (b) the loop's walls fire (~80-85% context -> clean handoff per rule 11; or the
two-consecutive-no-progress brake); OR (c) **balance exhausted** (auto-recharge is OFF; the launcher
stops cleanly on a billing/credit failure and writes a final standup). **Escalate-don't-quit on
game crash-loops** (supervisor auto-resumes + crash-loop-guards; diagnose via the faulthandler dump /
crash firewall / py-spy / grab-a-frame, then continue).

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS / AUDIO-PLUMBING FIXES ONLY. Do NOT touch core Kira personality or the two-bucket
  firewall.** Her identity / voice / oracle / memory / vision are sacred and OFF-LIMITS. Mode-state
  stays behind the Pokemon toggle. Audio-crash fixes live in the game/audio plumbing
  (`pokemon_audio.py` / play_live audio init-teardown / a subprocess wrapper) — mode-side, minimal,
  additive, flagged (rule 12). If a fix would touch shared/core, STOP and flag for Jonny.
- **AUDIO END STATE = ON.** The deliverable ships with `POKEMON_GAME_AUDIO=1`. Audio-off is only the
  committed floor + the documented fallback (Phase B / step 8). Do not quietly leave it off.
- **COMMIT PER FIX**, clear messages, `git add` lowercase `kira/` (capital-K silently fails).
- **VERIFIED, never asserted.** Every "it works" proven by reading state from disk / a real replay /
  a log trace (for the audio fix: a real captured dump, then a real crash-free audio-ON stretch) —
  not claimed. Three-state honesty (COMPILES / WIRED / VERIFIED), rule 1.
- **FRESHNESS over staleness** — read the actual latest decision/state/crash log, not memory of it.

## KNOWN-GOOD BASELINE (landed this session, pre-night-train)

- **Opening-replay loop FIXED** (`909e46a` + `1b019df`, restore point `e6c573d`): intro fires only on
  a genuine `--fresh-kira` start (not on `--show` resume relaunch); the supervisor waits for the bot
  endpoint before (re)launching (no hot-relaunch into a dead bot). A fresh GO plays clean single-intro
  bedroom -> rival -> Route 1 -> Viridian -> parcel pickup, THEN hits the parcel wedge.
- **faulthandler** armed from module-import on every play_live path -> a native crash dumps its
  C-stack to `logs/debug/playlive_faulthandler.log`. **This is the capture tool for Phase C.**
- **Crash firewall** in play_live dumps any Python exception to `logs/debug/playlive_crash_*.log` +
  emergency-banks (poison-guarded) + exits non-zero so the supervisor resumes.
- All dashboard GO/Resume/Stop route through `supervisor.py` (crash-safe auto-resume, crash-loop guard).
- **Game audio gating:** `POKEMON_GAME_AUDIO` (default 0) in `pokemon_proc._audio_args` / `go.py` /
  `watch.py`; =1 passes `--audio` -> play_live AudioPump (`pokemon_audio.py`) -> PortAudio output.
  The night's job is to make that path crash-free, then default it ON.

## STANDING TRUTHS (carry forward — operational law)

- **Default verification = the LOOK-AHEAD ORACLE** (`pokemon_agent/recon_longrun.py`), 14.3x headless,
  run until goal or genuine stall, then read the full decision/state log in ONE pass. NOT bite-sized
  micro-tests (a micro-test is only for isolating a mechanism a long-run already fingered).
- **THE SELF-HELP ARSENAL** (rule 15) before any blocker slows the climb: (1) the disassembly
  pret/pokefirered, (2) the wikis Bulbapedia/Serebii, (3) prior art ("GPT/Claude beats Pokemon" +
  SDL/PortAudio/pygame crash writeups for the audio bug), (4) YOUR OWN EYES — grab a frame and LOOK,
  (5) the 14.3x look-ahead. "Stuck" is a checklist, not a state.
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python is a shim (TWO PIDs per launch) —
  never taskkill your own run. Nothing launched within ~40 min of a handover.
- **PS 5.1 gotchas:** `*>` logs are UTF-16 (grep fails silently — use Select-String or decode first);
  PS 5.1 mangles THIS file's UTF-8 via Get-Content/Set-Content round-trips — edit it with the
  Write/Edit tools only.
- **kira/\* = Jonny's + approved core work under loud-log law.** `git add Kira/` capital-K silently
  fails — lowercase `kira/`.
- **BANK cleared stretches** (`promote_bank.py` / checkpoint) so the climb ratchets forward and never
  re-runs cleared ground. Sanctity-validate at every canonical bank.

## GO-LIVE RITUAL (before any live 1x watch)

1. Confirm the audio device is present + selected BEFORE booting (loopback binds at boot, won't
   hot-rebind a vanished device). 2. Boot `python run.py`. 3. Mic check ("cheddar"). Save-file card
   is live at `http://127.0.0.1:8766/pokemon_savecard` once booted.

---

WATCH STATUS: the SHOWTIME/kira timeline is a fresh bedroom->credits validation run in progress;
canonical Sherpa save already rolled credits (Champion at home, untouched by nav-fix work). The fresh
run currently plays clean bedroom -> parcel pickup, then hits the Route 1 -> Pallet wedge — first
thing fixed (Phase A). Deliverable ships audio-ON once Phase C lands. Pop-in (Sherpa) =
`python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
