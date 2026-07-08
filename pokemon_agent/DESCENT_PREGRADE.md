# DESCENT PRE-GRADE (machine half of F-11) — merged 2026-07-08 shift 5 close

Headless real-loop grade per banked arc spawn. PASS = no machine-visible wedge class fired.
This does NOT replace Jonny's spot-watches — it picks them.
MERGED TABLE: the 15-arc full sweep (03:00, PRE-shift-5 code — no twedge column then) with the
shift-5 re-graded rows folded in (marked ♻ = graded on tonight's code, twedge counted).
Full-sweep snapshot preserved at `logs/longrun/DESCENT_PREGRADE_full_shift5.md`.

| arc | grade | why ended | decisions | battles | watchdog | nav-tripwire | travel-wedges | voids | unnamed maps |
|---|---|---|---|---|---|---|---|---|---|
| banked_HM05 | **PASS** | window done | 7 | 94 | 0 | 0 | n/a | 0 |  |
| banked_ROCKTUNNEL | **WARN** | window done | 15 | 2 | 0 | 1 | n/a | 0 |  |
| banked_SCOPE ♻ | **FAIL** | decision budget | 31 | 3 | 0 | 0 | 271 | 0 |  |
| banked_FLUTE | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_SNORLAX | **PASS** | window done | 9 | 12 | 0 | 0 | n/a | 0 |  |
| banked_SAFARI | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_SURF_TAUGHT | **PASS** | decision budget | 31 | 11 | 0 | 0 | n/a | 0 |  |
| banked_SILPH ♻ | **PASS** | window done | 14 | 62 | 0 | 0 | 13 | 0 |  |
| banked_SABRINA | **PASS** | window done | 16 | 64 | 0 | 0 | n/a | 0 |  |
| banked_CINNABAR | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_BLAINE | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_GIOVANNI | **PASS** | window done | 2 | 79 | 0 | 0 | n/a | 0 |  |
| banked_VICTORY | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_E4 | **PASS** | decision budget | 31 | 0 | 0 | 0 | n/a | 0 |  |
| banked_POSTGAME | **PASS** | window done | 3 | 69 | 0 | 0 | n/a | 0 |  |

## Riskiest arcs (spot-watch these first)

- **banked_SCOPE** — FAIL (twedge=271): the B4F spawn's elevator exit now WORKS (rode to B1F,
  exited via the Game Corner onto the street late-window — rounds 1-3 built the landing oracle
  + menu settle), but head_to_gym's road steering hammers the walk-unreachable (11,15) stairs
  every tick meanwhile. Next fix: structural parking for unreachable same-map road anchors.
- **banked_ROCKTUNNEL** — WARN (nav=1): Celadon questline pass-through ends hammering blocked
  doorway (10,11)@(7,4) (`questline_wrong_building`, npc_block storm) — FENCED-BEND rough edge.
- ~~banked_SILPH~~ — **CLEARED**: re-graded PASS on the pad-router port; she WON BADGE 6
  (Marsh) autonomously inside the re-grade window (juniors + Sabrina through the teleport maze).

## F-8 name-gap payload (maps crossed with no _PLACE_NAMES entry): NONE

## NOTE for the next full sweep
Regenerate this whole table on tonight's code once SCOPE is green (the twedge column then
covers every arc): `.venv\Scripts\python.exe -u pokemon_agent\recon_descent_grade.py 120`
