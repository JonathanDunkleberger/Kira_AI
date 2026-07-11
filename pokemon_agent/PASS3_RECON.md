# PASS 3 — Part-1 gap map (wired vs unwired vs needs-live-watch). Read before building.

Consolidation audit for the fresh-GO autonomous run. Companion to `TEAM_DEPTH_ROOT_FIX.md` (the team-depth
executor gap — the headline). Labels: **WIRED** (reaches decision) / **COMPILES-not-WIRED** / **MISSING** /
**needs-LIVE-WATCH** (headless can't prove it). Fix hooks have file:line.

## 1) TEAM-DEPTH Part-C executor — **COMPILES-not-WIRED** (see TEAM_DEPTH_ROOT_FIX.md for full diagnosis+fix)
Brain plans the right full-6 team; `grind_to`/`develop_bench` have ZERO consumers (voice-only); catch hook
self-disables at 3 mons + never routes to catch locations; only the ACE levels; no `prep_for_e4`. THE headline build.

## 2) MOVE / TM / HM management — **BUILT + robust heuristic, WIRED** (with 3 real gaps)
Design = "the in-battle 'Delete a move?' box is un-actuatable, so PREVENT it, decide proactively." (campaign.py:2177)
- **`_value()` move-scoring (campaign.py:2320) — WIRED, battle-hardened.** STAB +40, unique-type coverage +60
  (weak filler denied), off-type lifeline +55 on a STAB-locked mon, one status opener +100 / redundant 2nd +25;
  `best_slot` never droppable (2343); **no-gut floor** MOVE_JUNK_FLOOR=50 (keep all 4 + decline rather than gut a
  good set, 2364); **damage-poverty override** (shed a redundant move so an attacker can land, 2375 — the PP-famine
  fix). Called proactively via `_ensure_move_room()` (2271) at grind (9345) + before gyms (3632). All 3 named past
  failures (mashed Vine Whip / tossed Razor Leaf STAB / double-powder famine) have postmortem guards. VERIFIED by prior live runs.
- **TM teaching — BUILT but NARROW.** `hm_teach.TeachFlow` drives any TM, ROM-truth compat (`gTMHMLearnsets@0x08252BC8`,
  hm_teach.py:400). Strategic teach = `_teach_gym_coverage()` (campaign.py:3469) — picks the ace, scores bag TMs by
  type vs the gym's defenders, forgets the ace's weakest DAMAGING move (never status). Recipient-correct + protects the set.
- **HM assignment — BUILT, general.** `default_plan()` (hm_teach.py:437) = lowest-level compatible non-lead with a
  free slot, protects `_PRECIOUS`. ROM-truth compat.
- **GAPS to close:** (a) **in-battle level-up learn is DECLINE-ONLY** (battle_agent.py:2434) — a great move learned on
  a path `_ensure_move_room` didn't pre-clear is LOST; fix: consult `_value()` before declining, or call
  `_ensure_move_room` at battle-entry universally. (b) **TM strategy fires only at gym walls** — a strong bag TM is
  never proactively taught otherwise; fix: a "useful TM → best ROM-compatible recipient, protect set" pass reusing
  `_teach_gym_coverage` scoring + `default_plan` recipient logic. (c) **No HM-slave role / no Move-Deleter cleanup** —
  an HM can permanently eat a keeper's attack slot on a full team; fix: bias `default_plan` to a utility mon on a full team.

## 3) PC / BOX management — **MISSING** (CLAUDE.md Tier-1 #15 ❌ confirmed accurate)
No deposit/withdraw/party↔box anywhere in campaign.py (every "PC" ref is the Pokémon **Center**/healing). Code admits
it: campaign.py:966 *"boxed — can't happen pre-box-mgmt."* Only box code = `recon_pcbox.py`, a standalone **deposit-only**
prototype (RAM-copy presses, banks to scratch), **not imported**, no withdraw. Full-party keeper catch relies on the
game's NATIVE auto-box (campaign.py:4501) → **keeper stranded in the box, never fielded**.
- **FIX (build the block):** promote `recon_pcbox.py`'s proven deposit flow → `Campaign.deposit_mon()`; add the mirror
  `withdraw_mon()` (genuinely new); wire a hook in `catch_one` (campaign.py:4365) — on a full-party keeper catch, box the
  lowest-value off-plan fodder first (or withdraw the keeper after native auto-box), consult TeamPlanner/`roster_judgment`
  for who is fodder. **This is the single change that turns "keeper stranded in box" into real team depth** — pairs with the Part-C build.

## 4) SOUL / NARRATION — machinery **BUILT + wired** through the sanctioned seams; **delight is LIVE-WATCH-ONLY by construction**
**The seam (no firewall violation, no core edits needed):** two channels, both need the live bot on `:8766` —
`emit` beats (`on_event`→`KiraVoice.emit`→`bot._pokemon_react`, pokemon_voice.py:293 / kira/bot.py:3271) for spoken
reactions, and the `place`/plan_note DECISION seam (`_oracle_choose`→`KiraVoice.choose`→`bot._pokemon_choose`,
kira/bot.py:3728). The `place` string (campaign.py:10025-10080) folds in `_spine_and_history`, `plan_note`,
`roster_awareness`, hints, goals + party-by-name.
**⚠️ TWO STRUCTURAL TRUTHS the night train MUST respect:**
- **`_pokemon_choose` DECIDES but does NOT speak** (kira/bot.py:3740; the `reasoning` is discarded, pokemon_voice.py:409).
  Her spoken narration rides on **pre-authored `emit` beats + soul hooks**, NOT the oracle voicing its own per-tick "why."
  (Candidate future enhancement: surface the oracle's reasoning as a beat — pairs with soul-debt "does she say WHY.")
- **HEADLESS BYPASSES ALL OF IT.** `recon_longrun.py` stubs `_oracle_choose` and has NO `KiraVoice` (recon_longrun.py:430;
  `_soul_choose`→None with no bot, campaign.py:7883); headless catches get species-name placeholders. **So a headless
  long-run CANNOT exercise or prove any soul/narration — items below are BUILT-and-wired but "actually endearing on
  stream" is bucket (b), Jonny's live watch only.**
- **1. TEAM-AS-FAMILY — BUILT + wired.** Bond store + hooks (pokemon_soul.py:56-234: note_caught/faint/clutch/evolve/met/
  name_reason, felt history, PID-guarded); naming is HERS via `_soul_choose("name",…)` (campaign.py:7813); roster-as-
  relationship into chat + tier≥2 reactions (`_journey_narrative` campaign.py:11202 → bot._pokemon_journey_block); persists
  cross-session (states/kira/pokemon_soul.json). ⚠️ **Machinery built; the ROSTER it decorates is still thin** (the Part-C
  team-depth gap) — build the real 6 and this layer has a family to bond with.
- **2. BOSS ENERGY — BUILT + wired** (the Jun-28 CODEBASE_AUDIT "PARTIAL" is STALE). Gary grudge escalates EVERY fight
  (campaign.py:1042, running W-L tally + payoff, timeout artifacts excluded); struggle→catharsis micro-arc (1075);
  leader shit-talk emit beats throughout; tier-3 beats promote to durable saga → callbacks hours later (kira/bot.py:3312,3334).
- **3. ARC / POSTGAME AWARENESS — BUILT + wired to decision `place` seam.** `_spine_and_history` (campaign.py:10921) folds
  the CLAUDE.md game-model verbatim (8 badges→E4→credits, team-of-6, dex), "you're on gym {n}/8, next {leader}", post-game
  victory-lap + E4 branches, story-so-far, into EVERY free-roam decision. ⚠️ own docs say "live-verify it SHAPES behaviour."
- **4. WATCHABILITY PACING SIGNALS — MISSING (biggest instrument gap).** Only guardrails (grind caps campaign.py:89,
  watchable-milestone logic, flee-floor) + raw `health.json` exist. **No per-segment watchable/grindy verdict, no
  grind-vs-battle-time ratio, no ace-vs-fodder-usage logging, no automated "this stretch was grindy" flag.** This is the
  ONE soul-adjacent thing that COULD be partly headless-measured (grind-tick/battle-ratio/ace-share are RAM/log-derivable)
  — BUILD IT so grindiness is catchable without a human watching every hour. (Delight still needs live eyes; grindiness needn't.)

**Stale-doc cleanup flag:** `pokemon_agent/CODEBASE_AUDIT.md` (Jun 28) HALF-WIRED items 1-2 (Gary arc, saga-into-reactions)
are now fully wired — update it.

---
## HONEST BUCKETS AT A GLANCE (for the Part-4 readiness report)
- **BUILDABLE-and-headless-verifiable now (the tractable core):** #1 team-depth Part-C executor, #2 move/TM/HM gaps,
  #3 PC/box (build it), Lapras lead-and-heal AI, watchability-signal logging (§4-item-4), end-of-run metrics generator.
- **BUILT already — reuse, don't rebuild:** move `_value`/`_ensure_move_room`, gym TM-coverage, HM `default_plan`,
  the whole soul machinery (family/boss/arc), save-hop (`watch.py`), audio isolation + supervisor.
- **needs-Jonny's-LIVE-WATCH to confirm (bucket b — expected, not failure):** soul delight (family/boss/arc *feel*),
  behaviour-shaping of the game-model, watchable PACE, and multi-hour LIVE-SHOW STABILITY. **Never claim these proven headless.**
- **OPERATE-THE-SHOW note:** run the real stream under **`supervisor.py`**, not `go.py` (go.py has no auto-restart).


## 5) LIVE-SHOW STABILITY — hardening BUILT (headless); multi-hour = **needs-LIVE-WATCH**. ⚠️ one real gap.
- **Audio SIGSEGV — HARDENED by construction (BUILT+headless-verified).** The PortAudio/WASAPI native abort (Viridian
  parcel fanfare repro) is now process-isolated: parent drains PCM safely and pipes to `audio_child.py` which owns the
  dangerous `st.write`; a native abort kills only the child, parent respawns (backoff, MAX_RESTARTS=8), game keeps
  running silent. Default ON (`POKEMON_AUDIO_ISOLATE=1`); `faulthandler` dumps native stacks. (pokemon_audio.py:31-352, audio_child.py:85, play_live.py:53)
- **Supervisor — BUILT (design solid, not multi-hour-proven).** `supervisor.py` resume-on-crash (`--resume --free-roam`
  only, never fresh), rapid-crash backoff + CRASH_LOOP_THRESHOLD=5, hang detector (kills if health.json heartbeat stale
  >300s), bot-down guard, dead-man's-switch ping. (supervisor.py:159-304)
- **⚠️ THE GAP (biggest unattended-stability risk): `go.py` (the standing GO button) launches play_live via a bare
  `subprocess.call` with NO restart wrapper (go.py:163)** — a crash takes the show dark until a human relaunches.
  Auto-restart ONLY exists under `supervisor.py --timeline showtime`. **FIX/OPERATE: run the real show under
  `supervisor.py`, not `go.py`** (or wrap go.py's launch in the supervisor). Then do a long SUPERVISED SOAK.
- **HONEST LINE:** the specific past crash is hardened, but **multi-hour live stability is UNPROVEN — only Jonny's
  supervised watch-run confirms it.** Never claim proven.

## 6) INSTRUMENTATION — metrics **MISSING** (data half-exists); save-hop **BUILT**; nav mostly solved.
- **End-of-run METRICS report — MISSING.** No generator exists; the "mountain survey" is hand-written. Buildable from
  data already banked: **playtime** (`_playthrough_elapsed`, campaign.py:10775 — ⚠️ but it's cross-session wall-clock,
  reads 299h, NOT per-run human time → a NEW per-run clock is needed); **health.json** (badges, party species+levels+
  HP/types, dex, goals — but NO movesets); **dense checkpoints** (`checkpoint.json`: ts/map/badges/party-count/playtime
  per gain-seam — a **badge timeline is reconstructable** from these but nothing does it); **losses** (strat_memory.json).
  **MISSING capture:** battle/win/faint counters, final movesets, persisted badge-timestamp table, longest-grind,
  per-segment watchable verdict. FIX: a post-run generator walking `checkpoints/*/checkpoint.json` + `strat_memory.json`
  + a party moveset dump + a NEW per-run playtime clock + battle/faint counters added to the loop.
- **SAVE-HOP — BUILT+works.** `python pokemon_agent/watch.py --list` (dense checkpoints + milestone banks + canonical);
  `watch.py --at <alias>` copies the full sanctity bundle to a disposable sandbox (canonical never written) and launches
  windowed at true speed with audio+voice. **Watch the E4:** `watch.py --at hall-of-fame` (→banked_E4, the win ceremony),
  `--at pre-e4` (→banked_LORELEI, the fight from the doorstep), `--at e4-approach` (→banked_VICTORY). Headless re-prove:
  `E4_BOOT=G:/temp/longrun/banked_E4 ../.venv/Scripts/python.exe -u recon_e4.py`. **Bot must be up (`run.py`) first.**
  ⚠️ avoid `--at summit` (mid-ceremony grenade → drains credits → title). NOTE: these E4 banks are NS7-14 hand-grind scratch
  banks, not a segment of one continuous organic run.
- **NAV — forward route solved/banked; residual risks (all OFF the forward critical path):** (a) **Indigo→VR has no
  world-model travel edges** (Route23→VR warps (5,28)/(18,28) uncached; R23 needs Surf) → any BACKTRACK to VR can wedge;
  (b) **Route 23 grass grind wedges** (split/surf-gated — grind confined to Route 18); (c) **cave grinding unbuilt**
  (`recon_grind_bench` needs grass; VR caves → `no_safe_grass`); (d) **`recon_grind_bench` doesn't set
  `POKEMON_FIELD_MOVES=1`** (footgun — pass it for water/Cut nav). The recurring wedge classes (band-recompute, door
  same-map crossings, split-map bands) all have committed fixes. Do a fresh-run nav soak to confirm zero forward wedges.

