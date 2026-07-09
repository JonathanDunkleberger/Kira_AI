# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-08)

**MISSION (the deliverable):** get Kira to play a **clean bedroom -> credits** FireRed run
autonomously, at **human-realistic pace (~25-35h target, NOT 67h)**, with **no permanent stucks**
and **no crash-loops**. You are the lead Sherpa. Lay the rope wedge-by-wedge until a full run is
clean, then the credits roll. Employment terms + standing rules 1-18 in CLAUDE.md are in force.

## THE FIRST BLOCKER (fix this before anything else) — travel-wedge (c)

On a fresh bedroom->credits run she **wedges delivering Oak's Parcel: routing Route 1 -> Pallet
Town**. Exact stall: `no_route genuine wall/zone gap` at **map (3,19) coord (13,0)** (Route 1
sub-area / boundary), TRAVEL WEDGE x4 -> campaign STOP. This is the KNOWN "Route-1 boundary wedge,
fresh-spine intro only" class the impossible-stand rung does not cover (was item 9 on the old couch
list, deprioritized then; it is the priority now). Fix it via the **look-ahead / warp-graph nav
harness** (`recon_longrun.py` at 14.3x headless): reproduce the stall, instrument the connection-band
/ warp-graph routing at (3,19)->(3,0/Pallet), fix the specific no-route, RE-RUN, confirm parcel
delivery completes.

## TONIGHT'S FRONTIER (in order)

1. **Fix the Route 1 -> Pallet parcel-delivery wedge** so parcel delivery completes end-to-end.
   Nav / warp-graph fix only — general engine code (travel/campaign), timeline-agnostic.
2. **Then keep laying the rope forward.** After the parcel wedge clears, drive the run forward via
   14.3x headless look-ahead (`recon_longrun.py`) toward credits; when it STALLS on the NEXT wedge
   (doorway, grass, building entry, NPC, switch, TM/HM, fetch-quest, unlearned map, anti-wedge, crash),
   DIAGNOSE -> FIX (general, solve-once-reuse-everywhere) -> RE-RUN -> BANK a checkpoint -> advance.
3. **Log every stuck / crash / wedge as a discrete note -> fix -> revalidate item** in NIGHT_REPORT.md
   (one line per shift per the loop contract; commit-per-fix is your proof of life).
4. **Validate watch-readiness on the SHOWTIME/kira timeline at true 1x, voice ON, game audio OFF.**
   Iteration is 14.3x headless (fast wedge-finding); watch-readiness is confirmed by playing a
   FIXED stretch at 1x with her voice on and game audio off (`watch.py` / showtime path). NOTE: a
   full 1x bedroom->credits is 25-35h real-time and CANNOT complete in one shift — validate the
   stretches you fixed at 1x, not the whole game every shift. The full clean run is proven via the
   14.3x headless look-ahead reaching credits with no wedge/crash.

**STOP CONDITION:** run until a **clean bedroom->credits** is demonstrated (a 14.3x headless full
run reaches credits with zero permanent stucks / crash-loops), OR the loop's own walls fire
(~80-85% context budget -> clean handoff per rule 11; or the two-consecutive-no-progress brake).
**Escalate-don't-quit on crash-loops** — the supervisor already auto-resumes + crash-loop-guards;
diagnose the crash (faulthandler dump, crash firewall, py-spy, grab-a-frame) and continue.

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS FIXES ONLY. Do NOT touch core Kira personality or the two-bucket firewall.** Her
  identity / voice / oracle / memory / vision are sacred and OFF-LIMITS. Mode-state stays behind the
  Pokemon toggle. If a fix would touch shared plumbing, keep it minimal + additive + flagged (rule 12).
- **COMMIT PER FIX**, clear messages, `git add` lowercase `kira/` (capital-K silently fails).
- **VERIFIED, never asserted.** Every "it works" must be proven by reading state from disk / a real
  replay / a log trace — not claimed. Three-state honesty (COMPILES / WIRED / VERIFIED), rule 1.
- **FRESHNESS over staleness** — read the actual latest decision/state log, not memory of it.
- **AUDIO STAYS OFF.** Game audio is off by default on every path (`POKEMON_GAME_AUDIO=0`); do not
  re-enable it. Her voice/TTS is a separate path and stays on.

## KNOWN-GOOD BASELINE (landed this session, pre-night-train)

- **The opening-replay loop is FIXED** (commits `909e46a` + `1b019df`, restore point `e6c573d`):
  (a) the scripted intro fires only on a genuine `--fresh-kira` start, never on a `--show` resume
  relaunch; (b) the supervisor waits for the bot endpoint before (re)launching, so it no longer
  hot-relaunches into a dead bot and thrashes the intro. A fresh GO now plays a clean single-intro
  bedroom -> rival -> Route 1 -> Viridian -> parcel pickup, THEN hits wedge (c).
- **faulthandler** is armed from module-import on every play_live path (showtime included) —
  a native crash now dumps its C-stack to `logs/debug/playlive_faulthandler.log`.
- **Crash firewall** in play_live dumps any Python exception to `logs/debug/playlive_crash_*.log`
  + emergency-banks (poison-guarded) + exits non-zero so the supervisor resumes.
- All dashboard GO/Resume/Stop route through `supervisor.py` (crash-safe auto-resume, crash-loop guard).

## STANDING TRUTHS (carry forward — operational law)

- **Default verification = the LOOK-AHEAD ORACLE** (`pokemon_agent/recon_longrun.py`), 14.3x headless,
  run until goal or genuine stall, then read the full decision/state log in ONE pass. NOT bite-sized
  micro-tests (those answer "did this fire?", not "can she play this stretch?").
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python is a shim (TWO PIDs per launch) —
  never taskkill your own run. Nothing launched within ~40 min of a handover.
- **THE SELF-HELP ARSENAL** (rule 15) before any blocker slows the climb: (1) the disassembly
  pret/pokefirered, (2) the wikis Bulbapedia/Serebii, (3) prior art ("GPT/Claude beats Pokemon"),
  (4) YOUR OWN EYES — grab a frame and LOOK, (5) the 14.3x look-ahead. "Stuck" is a checklist, not a state.
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

WATCH STATUS: the SHOWTIME/kira timeline is a **fresh bedroom->credits validation run** in progress;
canonical Sherpa save already rolled credits (Champion at home, untouched by nav-fix work). The
fresh run currently plays clean bedroom -> parcel pickup, then hits the Route 1 -> Pallet wedge (c) —
that is the first thing the night train fixes. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
