MISSION CONTINUES 2026-07-13 14:00 — fresh_go_3 IN FLIGHT after the ATTENDED questline-bench-guard fix (commit 5add821). Cold FRESH bedroom start, detached + watchdog, log G:/temp/longrun/fresh_go_3.log. Monitor per NEXT_SESSION.md. The shift-10 HALT + full prior shift history are archived at NIGHT_REPORT_archive_2026-07-13_1105_HALT.md.

## ATTENDED SESSION (Jonny at desk, ~13:40-14:00) — the questline-guard relax SHIPPED + VERIFIED + relaunched

WHAT SHIPPED (commit 5add821, campaign.py). The fresh_go_1/fresh_go_2 disqualify root = the ace runs away while the bench freezes. Deeper recon found it is TWO guards, not one, and the seafoam-crossing log proved the leak is the whole DUNGEON-HEAVY badge-5..8 back half (Hideout/Tower/Silph/Mansion/Seafoam are all caves/gauntlets), not a stray sea leg. BOTH guards returned early on ANY active questline:
  1. `_road_bench_xp_arm` (~7317) — no organic bench XP on the march during a questline.
  2. `_bench_severely_lopsided` (~7480) — the forced catch-up grind suppressed during a questline.
So through the questline-dense back half the bench got ZERO catch-up → arrived at Indigo 30+ levels lopsided.
FIX = relax BOTH, re-gated on MAP-TYPE (two new helpers `_on_overworld_now` / `_questline_march_bench_ok`, flag `POKEMON_QUESTLINE_BENCH_RELAX` default ON in code so it survives the watchdog env-scrub):
  • bench leads/grinds ONLY on OPEN GROUND (group-3 route/town or tv.G1_OUTDOOR); inside a real cave (tv.G1_CAVES) or a building interior the TRUE ace still leads (unchanged proven behavior).
  • the relaxed forward-march leg uses PURE PRE-LEG REORDER with the in-battle switch OFF (`PROTECT_LEAD_GRIND = not _ql_leg`) and only for allowlisted overworld/open-sea questlines → no switch actuation can flee-loop even if a leg strays (CEO: pre-leg ordering, not mid-battle switching).

VERIFY (3 layers, all PASS):
  • Gate logic: standalone probe, 12/12 correct (relax on open ground for overworld/sea questlines; keep the ace in EVERY G1_CAVES map + building interiors + for cave-crossing errands flash/seafoam/secret_key).
  • Overworld-behavioral: rt_mouth look-ahead — GRIND-WEAK + participation switch fired clean, bench L9→L15, ace held, switch actuated every time, ACE-DOWN auto-heal, 0 wedge.
  • LIVE-QUESTLINE behavioral (the decisive one the fixtures couldn't give — surge_done/rt_mouth hit the known fixture confounds): on fresh_go_3 Route 4, WITH the S.S.-Ticket questline ACTIVE + on open ground, `LOPSIDED-BENCH` FIRED and fielded the weak bench mon → rattata L8→L10, ace held L24, 0 tracebacks, 0 wedge. Under the old code that grind was suppressed (questline active) — this is the fix working on the real run.

fresh_go_3 LAUNCHED (cold FRESH bedroom, detached, watchdog, log fresh_go_3.log). Cleared the whole opening spine clean → Misty = badge 2 → free_roam, 0 tracebacks; now building the six at the Nugget-Bridge team-depth wall (same spot fresh_go_2 handled). fresh_go_2's banks archived `*_archived_fresh_go_2_final_*`; canonical states/campaign/ UNTOUCHED. RUN_STATS_fresh_go_2.md generated (confirms the disqualify curve: ace L65→L100 solo at the E4 while the bench froze L26→L34).

ACUTE WATCH (the qualifying gate) at E4 entry: every member ≥L42, ace-bench gap ≤15, ace nowhere near L100. On banked_CREDITS → run order 6 (run_stats + qualifying eval → CREDITS / WATCHABILITY-GAPS / HALT per rules).
