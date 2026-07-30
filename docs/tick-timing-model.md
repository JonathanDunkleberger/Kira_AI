# Tick timing model + the thinking-gate fix (2026-07-30)

One page. What a single free-roam tick costs, what blocks on what, which watchdogs are armed
during each stage — and the systemic bug this mapping exposed (shipped fixed the same day).

## The confirmed root cause (prime suspect CONFIRMED)

**The 8s wall-clock StuckWatch could not tell "thinking" from "wedged".** Evidence, soak report
`20260730_101644`, `tail_supervisor_playlive_2026-07-30_09-48-40.log`:

- Trip at exactly `frozen 8.0s` at `(7,7)@(2,5)` — *inside the Cerulean Poké Mart*, an innocent
  tile, immediately marked as a wedge spot and **persisted** to `wedge_memory.json` (line ~395).
- 40+ consecutive travel legs bailing at step 0 (`[travel] start` → `!! WATCHDOG disengage
  requested` with nothing in between, lines 3–110): one false trip latches `_stuck_request`,
  which travel polls every step but the roam loop only clears at the NEXT tick top — a single
  false trip shreds the entire remainder of the tick (every door-exit candidate, every leg).
- The dialogue three-layer net (`af10c96`) never got to act at the Nugget-Bridge Bill NPC:
  false trips fired ~8s into a healthy conversation and `_disengage_overworld_npc` B-closed it
  and marked the NPC as looping — long before any 120s conversation cap could matter.

## A single tick, stage by stage

| Stage | Wall time | Frames run? | Watchdog fed? | Armed? |
|---|---|---|---|---|
| Tick top: fingerprint, ledger, health publish | <0.1s | no | no | clock running |
| Stuck-latch honor + recovery (if latched) | 1–10s | yes | yes | reset first |
| Oracle ctx build (place/world/strategy notes) | 0.1–1s | no | no | clock running |
| `surface_want` (every 3rd tick) — **LLM** | 2–12s | pump (raw frames) | no | **was: clock running → FIXED: held** |
| Action decision `_soul_choose` — **LLM**, timeout 12s | 2–12s | pump (raw frames) | no | **same — held** |
| Actuator (travel / talk / grind / battle) | 5–120s | yes | yes (per ~6 frames) | armed — correct |
| Voice beats during actuator (`emit`) | ~0 (async queue) | n/a | n/a | n/a |
| `pace()` savor-holds (battle beats; T3 up to ~17.5s) | 0.3–17.5s | yes | **yes, static world** | **was: armed → FIXED: held** |
| `_dialogue_hold` read-holds (T2/T3 lines, cap 5s) | 1–5s | raw frames | no | **held now** |
| Dialogue `drive()` page advance | 0.02–0.11s/page | yes | yes | armed; page turn = new key = progress (correct) |

Key wall-clock fact: StuckWatch measures `now - t0` where `t0` is the last *new* world/text key.
It does not need to be fed during a stall to trip — the first feed *after* a silent 8s+ span
(exactly what an LLM call is) tripped it retroactively. So "the pump keeps the world live" was
true for the viewer and false for the watchdog.

## The fix (shipped with this commit)

1. **`StuckWatch.hold(reason)` / `release()`** (`world_fingerprint.py`) — deliberate stillness is
   a first-class state. Feeds while held reset the clock; `release()` resets it again, so the held
   span never counts even when nothing fed during it. Nestable.
2. **One chokepoint for the brain**: `campaign._soul_choose` wraps every oracle round-trip
   (action / want / name / catch_judgment / move_drop — all kinds) in `watchdog_hold("oracle:…")`.
3. **Deliberate holds declared**: `play_live.pace()` (voice savor) and `_dialogue_hold`
   (read-pace reveal) wrapped the same way.
4. **Poisoned memory purged**: `wedge_memory.json` gets schema `v2`; a pre-gate (v1/unversioned)
   file is discarded wholesale on load, loud. The PC's accumulated false marks self-purge on the
   next relaunch.
5. **8s stays.** With thinking excluded, everything legit that holds the world still is declared;
   what remains frozen for 8s is a real wedge, and on stream every wedge-second is dead air.

Reflex/brain split status: travel ("NO LLM in movement") and `dialogue_drive` (pure press/pixel/
RAM loops) were already LLM-free — pressing A never waited on the LLM *mechanically*; it was the
watchdog conflation that made it look and behave that way. Battles are exempt by design
(`battle_active` fingerprints auto-reset the watch), so in-battle LLM item choices are safe.

Proof: `test_stuckwatch.py` — 11/11 green, including `thinking_hold_never_trips` (20s stall,
feeds during hold), `silent_hold_never_trips` (the real live shape: zero feeds during the call),
and `nested_holds`. Genuine post-hold freezes still trip at 8s.

## Dead-weight audit (mandate item 4 — recommendations, no deletions shipped today)

- **Default-OFF flags, never exercised live**: `POKEMON_PCBOX=0`, `POKEMON_ITEM_PICKUP=0`,
  `POKEMON_FIELD_MOVES=0`. PC-box + item-pickup are safe pruning candidates. **Field moves is
  NOT dead weight — it's a future blocker**: Cut/Surf/Strength are mandatory for badges 3+.
  It's off because actuation isn't trusted yet; it needs a recon pass before Vermilion, not
  deletion.
- **Showtime/segment paths** (`--show`/`--go`, `run_segments`, `states/kira` banking): unused by
  the marathon (free-roam) timeline but explicitly Jonny's demo-day tool — keep, per the brief.
- **Recovery layers** (disengage → escape-hatch reload → deep-wedge ring → abandon): looked
  duplicated only because false trips were exercising all of them constantly. With trips now
  meaning real wedges, the ladder is the correct escalation order. Re-evaluate after a clean
  24h soak, not before.
- One-shot recon/debug scripts in `pokemon_agent/` (recon_*, m1_*): inert, zero interaction
  surface; prune for hygiene whenever convenient.
