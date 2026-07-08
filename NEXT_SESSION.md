# NEXT_SESSION — POST-COUCH (SAGA WRAP, 2026-07-08)

**The showcase-critical bucket is DONE.** First human couch-watch happened; Fix Pass 1 (persona
dial) + the two round-2 items (parcel stall, mic diagnosis) + P-7 OpenAI purge + the save-file
card all shipped and verified same-day. Full record: **COUCH_NOTES.md**. All 15 descent arcs
PASS; clipper final spec + Phase-J receipts confirmed wired. **SAGA WRAP, not CLOSED — "CLOSED"
is Jonny's call after his own spot-watch.**

**SHOWTIME LATENCY PASS DONE (d56b183):** the showcase-BLOCKER — game-event reactions landing
30-44s late (content_age 34-40s; she joked about a Rattata fight ~40s after it ended) — is FIXED.
Root: fire-and-forget event reactions serialized on the turn lock behind TTS, aging in the queue.
Fix = a pokemon-scoped FRESHNESS CEILING (7s) + SUPERSEDE at the speak point: a grind beat older
than the ceiling (or superseded by a fresher beat) is DROPPED — she reacts live or stays silent,
never to a corpse. TIER-PROTECTED: tier≥2 milestone beats (badges, evolutions, Champion monologue,
mom's goodbye) ALWAYS deliver. Also: KIRA_TTS_PREFETCH now DEFAULT-ON (TTS pipelining; kill switch
=0); YouTube 403 spam silenced (log once → disable auto-search for the session). Verified headless
(freshness_verify: flood all-dropped, fresh delivers <1s, tier≥2 protected). **THE GO/NO-GO:
Jonny's next spot-watch confirms live content_age stays <6s — that's the showcase green light.**

## GO-LIVE RITUAL (do this before any live boot)
1. **Confirm the audio device is present + selected BEFORE booting** (esp. after a Bluetooth
   drop — the loopback binds at boot and won't hot-rebind a vanished device; that was the whole
   "she can't hear me" episode). 2. Boot `python run.py`. 3. Mic check ("cheddar"). Save-file
   card is live at `http://127.0.0.1:8766/pokemon_savecard` once booted.

## THE FINAL COUCH LIST (needs Jonny's eyes / not showcase-blocking — ride the free weekly window)
0. **THE GO/NO-GO — game-event latency live-watch** (showcase gate): reboot, run a fast-battle
   stretch, confirm reactions land <6s and aligned (watch for `[Freshness] DROP` lines culling the
   stale backlog, and `[LAG]` content_age staying low). This is the showcase green light.
1. **Live mic check** on the Focusrite ("cheddar") — confirm a fresh `>>> You said:` after reboot.
2. **Loopback-cable decision** (R2-2): should loopback refuse to fall back onto the TTS-carrying
   VB-cable? Rig-specific + core-senses (HARD CONSTRAINT #2) — flagged, not cowboyed. Your call.
3. **Later-game descent spot-watches** — all arcs PASS; pick any (VICTORY / BLAINE / POSTGAME).
4. **Prefetch A/B** (KIRA_TTS_PREFETCH=1) → decide the default (pump complete; default undecided).
5. **20-min cohost smoke** — optionally feel-test the new flags one at a time (MEDIA_PACING /
   ATTENTION_DIRECTOR / CHAT_ADVISORS, all default-OFF).
6. **Tri-mode session** (companion → pokemon → watch-party) — firewall check.
7. **First clipper manifest review** on a real VOD (10 shorts / superfan / midform).
8. **Deep latency war** beyond the shipped fixes; **live-chat conversation tuning**; **Attention
   Director feel-tuning**; **Hitman / cohost-mode recon** — all parked, need eyes + the weekly window.
9. **Segment-driver Route-1 boundary wedge** (fresh-spine intro only) — a genuine no-route class
   the impossible-stand rung doesn't cover; deprioritized (you're watching later-game, not intros).
10. **Audio-mood via Gemini** — the P-7 casualty; migrate like vision if you want music/mood back.

---
Below = the CLOSING-BELL mandate as it stood pre-couch (historical; the couch list above supersedes).

## DEFINITION OF DONE (historical)

## DEFINITION OF DONE

**A) POKÉMON SHOWCASE-READY**
- [~] Fix the banked_VICTORY 153-twedge regression (the known blocker). **ROOT CAUSE FOUND +
  FIX SHIPPED shift 14** (see in-flight block below) — re-grade RUNNING.
- [ ] Full descent re-sweep on current code = ALL 15 arcs PASS; DESCENT_PREGRADE.md regenerated
  complete (run AFTER the VICTORY re-grade passes; ~35 min; not within 40 min of handover).
- [ ] GO button + watch rig + soul stack verified against final code (go.py / watch.py smoke).
- [ ] Fresh 10-min throwaway passes the F-5 bar: bedroom→starter ~90s travel, zero wall-grinding,
  voiced choices, mom acknowledged, no stale reactions.
- [ ] Then ONE human watch round (Jonny) → his notes get ONE fix pass → done.

**B) GENERAL KIRA AT HER BEST** (the 60% backlog → 100% of machine-shippable)
- [ ] Latency war finished: full de-block of the 3860ms chain, prefetch default decision,
  freshness windows tuned.
- [ ] Conversation engine tuned for live chat: restraint/timing, advisors polish (G-2),
  reject-with-reason, moderation hooks + output-side liability filter.
- [ ] Attention Director wired (I-1b — EXTEND the existing Activity Director, rule 3).
- [ ] Media-pacing profiles (I-1c).
- [ ] 'Heavy lifting' tic governor (mode-side now, core recon flagged).
- [ ] Cost receipts live (Phase J).
- [ ] Clipper COMPLETE: all three output tiers + ranked shorts manifest per spec (10 ranked
  shorts + 20-min superfan cut + 3-5min midform, ONE dated folder, caption source audited for
  reuse of existing transcripts, review queue).
- [ ] Regression: every core touch re-verified in sandbox; firewall loud-logs.

**BURN DISCIPLINE UNCHANGED:** value lines, no idle-grinding while blocked, bounded recon.
If the list empties except needs-eyes: STOP, write the couch list, stop burning.

## ⚡ SHIFT 14 IN FLIGHT (rewrite as you bank)
- **✅ banked_VICTORY 153-twedge FAIL → PASS twedge=13 (commit 2158e43).** ROOT CAUSE: Route 23
  (3,42) is a SPLIT MAP by design — Victory Road IS the road between its halves (warps
  (5,28)→(1,39), (18,28)→(1,40); south exit = gate warps (8,153)/(9,154)→(28,0), NO south
  edge). The heal ladder was component-blind and **enter_warp's door pre-check was LAND-ONLY**
  (the shift-10 water-start law never reached it — the surf-only gate door read "no south
  warp"). Fixes: `_edge_band_reachable(walkable=or_surf)`, feet-level band pre-checks on the
  adjacent-city excursion + each cardinal graph hop, `return_to_center` per-leg direction
  (Viridian-adjacent aware) + band pre-check, surf-aware enter_warp. Re-grade VERIFIED PASS.
- **✅ SPLIT-MAP ROAD MEMORY (f54ed42) VERIFIED:** banked_SURF_TAUGHT re-grade **PASS
  twedge 15 → 2** (log `descent_regrade_shift14_surf.log`) — first no_path at the Seafoam
  split parked BOTH legs for the session; memory KEPT across every exit; metronome dead.
- **✅ Full 15-arc sweep (2158e43-era code): 14 PASS / 1 WARN** (SURF_TAUGHT pre-fix; table
  banked in git 90091a4). VICTORY PASS confirmed on a second window (twedge 14, 81 battles).
- **🏁 FINAL full 15-arc sweep on HEAD: ALL 15 PASS — riskiest-arcs list EMPTY** (aeb2333;
  log `descent_final_shift14.log`). SURF_TAUGHT 15→2, SAFARI 15→1, VICTORY PASS ×3 windows.
  DESCENT_PREGRADE.md regenerated complete = the DoD-A machine artifact, DONE.
- **Evobeat verify: INCONCLUSIVE (as pre-briefed)** — no bundle has a past-due LEVEL evolver
  (Ekans max L17, needs L22). Evolution early beat stays WIRED-not-VERIFIED; it verifies
  organically the first time a bench evolver crosses its level (or on a spot-watch).
- **MACHINE LIST EMPTY — only needs-eyes remain.** Per the mandate: burning stopped. The
  couch list below is THE deliverable; SAGA CLOSED goes on NIGHT_REPORT line 1 after the
  couch sitting + the one fix pass.
- **✅ B-BATCH SHIPPED (commit 392b872, core touches flagged, defaults byte-identical):**
  Phase-J receipts (write_receipt → logs/receipts/ + LEDGER.jsonl, at-shutdown, WIRED);
  output-side liability filter (KIRA_LIABILITY_FILTER ON, narrow secrets/PII on the pre-TTS
  choke; KIRA_MODERATION_REGEX hook; probe 0 FP/0 FN VERIFIED); I-1c media-pacing
  (MEDIA_PACING_ENABLED OFF); I-1b Attention Director (ATTENTION_DIRECTOR_ENABLED OFF,
  activity-aware _has_fresh_sense + [ATTENTION] prompt lead). Clipper final spec: 10 ranked
  shorts, superfan 1200s, midform 300s, caption-source audit documented; .env mask-checked.
  **Recon corrections:** TTS prefetch pump is COMPLETE (only the default-ON decision is
  A/B-gated); F-9 "no wink escape" tic ban IS core repetition_guard (mode inherits).
- **✅ GO refusal rail re-verified on final code** (bot down → rc=2, loud, no launch). watch.py/
  play_live.py unchanged since their live verifies; today's campaign.py edits are exercised by
  the sweep itself.
- **⏸ DESK-PRESENCE: Jonny at the machine (idle=0 min at ~07:50)** → NO bot boots, NO voiced
  throwaway (the 21:30 law). F-5 throwaway + soul-stack live verify PARKED on the couch list.
- Residual known-crevasse (unchanged): Victory Road + Seafoam interiors STRIKE-ONLY; travel's
  strength-push primitive can mis-clear/misread a boulder as a trainer (bounded, filed).

## 🛋️ THE FINAL COUCH LIST (the ONLY remaining work — ONE sitting, ~1.5-2h total)
1. **Fresh 10-min throwaway (F-5 bar) + descent spot-watches.** Boot her (`python run.py`,
   wait for :8766), then `python pokemon_agent/go.py --throwaway` (~10 min; bar: bedroom→
   starter ~90s travel, zero wall-grinding, voiced choices, mom acknowledged, no stale
   reactions). Spot-watches: `python pokemon_agent/watch.py` — all arcs graded PASS, so pick
   any 2-3 for feel (suggest VICTORY + BLAINE + POSTGAME). Cleanup:
   `go.py --clean-throwaways` + `watch.py --clean`.
2. **Prefetch A/B (2 min):** restart the bot with `KIRA_TTS_PREFETCH=1`, one conversation —
   decide the default (pump is complete; only the default is undecided).
3. **20-min cohost smoke** (G-4 exit) — optionally with `MEDIA_PACING_ENABLED=true` +
   `ATTENTION_DIRECTOR_ENABLED=true` + `CHAT_ADVISORS_ENABLED=true` to feel-test the new
   flags one at a time per the cadence plan.
4. **Tri-mode session** (Phase I exit, 15 min): companion → pokemon-play → watch-party,
   checking the firewall (no Pokémon leak outside play-mode).
5. **First clipper manifest review** (K exit): run the clipper on a real VOD
   (`python scripts/cut_clips.py --date <date>`), review manifest.json (now 10 shorts /
   20-min superfan / 5-min midform), flip approved flags.
Then ONE fix pass on your notes → write **SAGA CLOSED** on NIGHT_REPORT.md line 1.
(Final showtime sign-off — the Kira-timeline launch is HIS press, always.)

## STANDING TRUTHS (carry forward — operational law)
- Re-grade command: `$env:DESCENT_ARCS='<arcs>'; .venv\Scripts\python.exe -u
  pokemon_agent\recon_descent_grade.py 120 *> logs\longrun\<log>`. Full sweep = no DESCENT_ARCS.
- venv python is a shim — TWO PIDs per launch; never taskkill your own run. SINGLE-RUN LAW: one
  emulator recon at a time; nothing launched within ~40 min of a handover (night loop kills
  in-flight runs).
- Grade harness is READ-ONLY on bundles; banked_CREDITS excluded (mid-ceremony grenade).
- PS 5.1 `*>` logs are UTF-16 — grep fails silently; use Select-String or decode first.
- ⚠️ PS 5.1 mangles this file's UTF-8 via Get-Content/Set-Content round-trips — edit it with
  the Write/Edit tools only.
- NIGHT-SHIFT BOT ETIQUETTE: never launch run.py unattended without checking desk presence
  (:8766 + recent mic) first; kill the tree cleanly after.
- kira/* = Jonny's + approved core work under loud-log law. `git add Kira/` capital-K silently
  fails — lowercase `kira/`.
- Known general gaps: Seafoam + Victory Road interiors = strike-only; evolution early beat
  WIRED-not-VERIFIED; desynced coord-read root-cause thread open (detector recovers it).

---

WATCH STATUS: canonical bank is CLEAN — the TRUE post-game: the Champion at home in Pallet Town
((4,0)@(4,8), full healthy party — Venusaur L95, Persian, Fearow, Raticate, Ekans, Lapras —
badges 8, player in control; sanctity VALID). She is at home, victory lap ahead (Cerulean Cave
open; her stated want: catch Mewtwo). Pop-in = `python pokemon_agent/watch.py` → spawn
'postgame' (or --canonical, safe).

READY FOR THE TRAIN.
