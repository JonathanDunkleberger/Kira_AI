# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 3 in flight)

**MISSION (the deliverable):** a **clean bedroom -> credits** FireRed run, autonomous, at
**human-realistic pace (~25-35h)**, **no permanent stucks, no crash-loops**, AND **game audio ON
(`POKEMON_GAME_AUDIO=1`) + her voice/TTS ON** — watchable/recordable evergreen content. A silent
playthrough is the FALLBACK FLOOR, not the finish line. You are the lead Sherpa. Employment terms +
standing rules 1-18 in CLAUDE.md apply.

## LIVE STATUS (night-train shift 3 — the show spine now reaches badge 2 clean; free_roam handoff added)

**THE FRESH SHOW SPINE (`build_segments`) NOW CLEARS BADGE 1 AND BADGE 2 AUTONOMOUSLY.** Two verified
wins this shift, then the structural handoff:

- **BROCK (badge 1) — VERIFIED CLEARED (shift-2 `grind_pre_brock`, confirmed this shift).** The
  in-flight FRESH 45 run drove bedroom -> `grind_pre_brock` (Lv6->Lv13 on Route 2) -> **Boulder Badge**
  -> Route 3 -> Mt Moon -> Cerulean, Bulbasaur evolved to Ivysaur. The shift-2 under-leveling wall is
  dead. (Log: `logs/debug/freshspine_shift2.log`.)

- **MISTY (badge 2) — FIXED + VERIFIED (`be5f60b`).** ROOT CAUSE (NOT a junior gate — proven: badge
  wins with both swimmers deferred): she arrives at Misty's front tile (8,7) via a ZERO-STEP travel,
  so Misty (at (8,6), looking DOWN onto (8,7)) never gets a fresh step into her line-of-sight -> her
  battle never fires -> A opens a GREETING box -> award drain finds no badge -> STUCK ("You have to
  face other TRAINERS…"). FIX (general, every gym): `_los_retrigger(front)` steps off the leader front
  and back so the sight re-fires (wired into `beat_gym`'s leader engage, `has_badge`-guarded). Also
  hardened junior handling: `_engage_trainer` attempts only LAND-standable fronts (a pool swimmer's
  water-side fronts wedged travel x4) + a nudge; `_clear_gym_trainers` no longer FALSELY marks a
  failed-engage junior 'beaten' (bounded retries -> defer LOUD). VERIFIED: `recon_misty_fix.py` from
  the pre-gym Cerulean state -> real battles (swimmers Horsea/Shellder/Goldeen + Misty Staryu/Starmie)
  -> `*** MISTY BADGE ***` -> clean exit, twice. Ground-truth probe: `recon_misty_gate.py`.

- **SPINE -> FREE_ROAM HANDOFF — ADDED, VERIFICATION IN FLIGHT.** `build_segments` scripts the SHOW
  spine ONLY through Misty (badge 2). The rest of the game (badge 3 -> E4 -> credits) is handed to
  `free_roam` — the SAME proven autonomous climber that rolled canonical credits. Wired into
  `recon_longrun.py` FRESH branch: after `run_segments` returns `all_segments_complete`, it calls
  `camp.free_roam(...)` for the remaining budget (canonical protected — `_save_campaign` is
  STAGE-redirected in the look-ahead). **VERIFY:** `python pokemon_agent/recon_longrun.py FRESH 90`
  -> should reach Pewter/Brock, Cerulean/Misty, THEN free_roam past badge 2. READ
  `logs/debug/freshspine_shift3.log` for the next wedge past Misty, then diagnose -> fix -> re-run.

## FRONTIER (do in order)

1. **VERIFY the free_roam handoff climbs past badge 2** via the FRESH 90 look-ahead. Find the FIRST
   wedge in the post-Misty free_roam (Bill / Route 5-6 / Vermilion / SS Anne-Cut / Surge …), diagnose
   -> fix general -> re-run. Iterate toward credits (canonical already proved bedroom->credits is
   reachable via free_roam, so the machinery exists; the fresh-context climb just needs its wedges
   cleared). NOTE: the fresh run soloed Ivysaur (Route 3 CATCH failed to add a 2nd mon, party=1) — a
   real player would have a team; watch whether the solo survives deeper, and whether team-building
   (#3) needs attention for the mid-game.
2. **WIRE THE HANDOFF INTO `play_live --show` (the REAL show path) — CAREFULLY, FIREWALLED.** Today
   `--show` runs ONLY `run_segments` (ends at Misty). Add the same free_roam handoff after
   `all_segments_complete`. **FIREWALL — the subtle part (verified this shift by reading go.py:117-134):**
   `go.py` already sets `POKEMON_CAMPAIGN_DIR` for the real show, but to a **DISPOSABLE per-run scratch
   tempdir** (`campaign_scratch`) — because the current show never calls `_save_campaign` (run_segments
   banks segment checkpoints to states/kira via ckpt_dir). So canonical is already safe, BUT a free_roam
   anchor written there would be **LOST on the next run** (thrown away) -> the show can't resume past
   badge 2. So item 2 needs go.py to point the show's `POKEMON_CAMPAIGN_DIR` at a **PERSISTENT
   kira-lineage dir** (e.g. `states/kira` or `states/kira/campaign`), NOT the scratch dir, so the
   free_roam living anchor + sanctity sidecars (world/strat/journey/soul) persist and resume. That ALSO
   moves health.json into the kira lineage — CHECK the dashboard timeline-health read doesn't regress
   (see the timeline-bleed memory). Then make the handoff RESUME-SAFE: on a crash relaunch `run_segments`
   returns immediately with `b` at fresh bedroom, so LOAD the kira living anchor before free_roam or the
   show replays the whole game. Canonical (states/campaign) stays untouched by construction (redirected).
   Core-adjacent (rule 12): minimal, additive, flagged; VERIFY via a headless --show run reaching
   free_roam + a kill-resume kill-test, confirming canonical bytes unchanged. (The look-ahead already
   PROVED the handoff concept: fresh -> Brock -> Misty -> free_roam climbs badge 3.)
3. THEN: Phase B (bank the clean audio-OFF floor once fresh->credits is proven) + Phase C (native
   audio SIGSEGV capture via faulthandler `--audio`) remain untouched.

## KEY FACTS / TOOLS

- **Look-ahead oracle** (default verification): `python pokemon_agent/recon_longrun.py FRESH <min>` —
  fresh ROM -> real show spine -> (now) free_roam handoff, headless ~14x, canonical STAGE-protected.
  Non-FRESH boot = free_roam from a named state. Log = `logs/debug/freshspine*.log`.
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = a 2-PID shim — never taskkill your
  own run. Nothing launched within ~40 min of a handover.
- `beat_gym` / gyms: leaders are NOT junior-gated in FRLG; the LoS re-trigger is the general
  reliability fix for "arrived without a fresh step -> leader won't fight."
- **states/campaign = SHERPA CANONICAL (untouchable). states/kira = the show timeline. states/workshop
  = scratch (the FRESH look-ahead banks segment checkpoints here).**
- Commit per fix (lowercase `git add kira/`); VERIFIED not asserted; read the latest log not memory.
- PS 5.1 gotchas: `*>` logs are UTF-16; edit THIS file with Write/Edit only (PS mangles its UTF-8).

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS / AUDIO-PLUMBING FIXES ONLY.** Core Kira identity / voice / oracle / memory / vision
  are sacred + OFF-LIMITS. Mode-state stays behind the Pokémon toggle. If a fix would touch shared/core
  (e.g. the play_live --show handoff), keep it minimal/additive/flagged and TELL Jonny (rule 12).
- **AUDIO END STATE = ON** (`POKEMON_GAME_AUDIO=1`); audio-off is only the committed floor + fallback.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS

(a) clean bedroom->credits with audio ON demonstrated (full deliverable); OR (b) ~80-85% context ->
clean handoff (rule 11) / two-consecutive-no-progress brake; OR (c) balance exhausted. Escalate-don't-
quit on crash-loops (supervisor auto-resumes; diagnose via faulthandler dump / py-spy / grab-a-frame).

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched by this shift's nav work);
the FRESH show timeline now clears bedroom -> Brock -> Misty autonomously and hands to free_roam for
the rest — verifying how far the free_roam climb reaches past badge 2. Pop-in (Sherpa) =
`python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
