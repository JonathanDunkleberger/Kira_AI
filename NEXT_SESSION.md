# NEXT_SESSION — THE CLOSING BELL (FINAL MANDATE, CEO 2026-07-08 07:00)

This REPLACES all prior phase structures. ONE list. Everything ships TODAY-scale. No sequencing
courtesy between items — parallelize where the emulator allows, batch everything batchable, close
items COMPLETELY, value line per shift. The ONLY stops: a genuine needs-eyes, or the list EMPTY.
When both A and B are done → write **SAGA CLOSED** on line 1 of NIGHT_REPORT.md.

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
- **banked_VICTORY 153-twedge FAIL — ROOT CAUSE:** Route 23 (3,42) is a SPLIT MAP by design:
  its south half is walled from its north half — **Victory Road IS the road between them**
  (warps (5,28)→(1,39) and (18,28)→(1,40); south exit = gate warps (8,153)/(9,154)→(28,0),
  NO south edge). Hurt in the south half, the heal ladder was component-blind: (1) "Indigo is
  adjacent north" excursion → structurally no_route ×20/tile; (2) graph multi-hop to Indigo →
  same; (3) Viridian fallback said "no south warp" — a LIE: enter_warp's door pre-check was
  LAND-ONLY (`grid.walkable`), and the gate door sits across the lake. The water-start
  reachability law (shift 10) never reached enter_warp.
- **FIX SHIPPED (campaign.py, 5 edits, COMPILES):** `_edge_band_reachable` takes a walkable
  layer (or_surf when `_surf_usable`); heal ladder adjacent-city step band-pre-checks from the
  FEET before any excursion (LOUD skip); graph-hop loop pre-checks each cardinal hop (fail-fast
  to next rung); `return_to_center` picks per-leg direction (Viridian-adjacent aware — from
  Route 22 Viridian is EAST) + band-pre-check skips straight to warp; `enter_warp` pre-check is
  surf-aware. Net: heal from south Route 23 = surf south → gate (28,0) → Route 22 → east →
  Viridian. All general (helps every water-adjacent door + every split map).
- **NOW RUNNING:** targeted re-grade `$env:DESCENT_ARCS='banked_VICTORY'` →
  `logs\longrun\descent_regrade_shift14_victory.log`. PASS bar: twedge ≤ 20 (expect residual =
  VR-interior ~7 + grass-remember one-offs). If PASS → launch the FULL 15-arc sweep. NOTE the
  grader OVERWRITES DESCENT_PREGRADE.md per run (full table = git history).
- **Residual known-crevasse (NOT tonight's rope):** Victory Road interior is STRIKE-ONLY
  (recon_victory.py hand-derived boulder pushes); wedges inside (1,40) are bounded. Portability
  debt filed: general boulder_assist (spin_assist pattern + push tables as gamedata).
- Queued behind the sweep (SINGLE-RUN LAW): evobeat verify
  (`.venv\Scripts\python.exe -u pokemon_agent\recon_evobeat_verify.py`) — INCONCLUSIVE if no
  bundle has a past-due LEVEL evolver; then the beat stays WIRED-not-VERIFIED.

## NEEDS-EYES LEDGER — THE FINAL COUCH LIST (batch for ONE sitting; surface TOGETHER, never one at a time)
1. Fresh throwaway watch (F-5 bar) + descent spot-watches from DESCENT_PREGRADE.md (one sitting).
2. Prefetch A/B (2 min — bot restart w/ flag + one conversation).
3. 20-min cohost smoke with new eyes (G-4 exit).
4. Tri-mode session (Phase I exit, 15 min).
5. First clipper manifest review (K exit).
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
