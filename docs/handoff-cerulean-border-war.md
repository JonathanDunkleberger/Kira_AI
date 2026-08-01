# Handoff — the Cerulean Border War (2026-08-01, ~04:45 PT)

For the next agent picking this up. Read this top to bottom before touching anything.
Companion context: `AGENTS.md`, `docs/marathon-stream-plan.md`, `pokemon_agent/AUTONOMOUS_GAME_HARNESS.md`.

## Mission state in one paragraph

Kira (autonomous FireRed harness + VTuber persona) is mid-marathon-run: **2 badges,
canonical save on Route 4, party led by Wartortle L30, ~10.5h played**. She has been
stuck for ~2 days ping-ponging between Cerulean City and Route 4 instead of walking
south to Vermilion (badge 3, Lt. Surge). The root cause chain is now fully diagnosed
and the final fix is **in flight** (see "Live right now"). Jonny is extremely
frustrated — multiple explicit "fix this now, last warning" messages. Do not ship
half-measures; verify with logs before declaring victory.

## Live right now (VERIFY BEFORE TRUSTING)

A background worker was launched at ~04:40 to implement and push, in one commit on `main`:

1. **`pokemon_agent/travel.py` — ledge hops off a map edge = legal seam crossing.**
   In `bfs()` (~line 479-521), a ledge-hop landing (2-tiles-along) that falls outside
   the playable bound must be accepted when it satisfies `goal_test` and `walk()`.
   This is THE fix for the stuck run: Cerulean's south boundary to Route 5 is a
   one-way south-jump ledge row; the old bounds check made the only exit read as a
   solid wall → `head_to_gym -> no_path` every tick.
2. **`pokemon_agent/campaign.py` — road leg authoritative.** When on the billed road
   for the next gym (`_gym_road`/`_road_step`, ~line 13500), never fall back to the
   world-graph route (the graph is what dragged her backward west into Route 4,
   creating the ping-pong).
3. **`pokemon_agent/resume_marathon.ps1` — missing `PROMOTE_TARGET.txt` now defaults
   to a CANONICAL launch** instead of inventory-only (the one-shot consumption of that
   file stranded Jonny 3 times: script kills Kira's processes then launches nothing).
   Repo copy of `PROMOTE_TARGET.txt` re-armed with `CANONICAL`.
   Plus a synthetic recon test `pokemon_agent/recon_ledge_edge_cross.py`.

If the worker finished: confirm the commit is on `origin/main`, tests passed, then tell
Jonny to rerun the usual one-liner (below). If it didn't finish or push failed, finish
the job per the spec above.

## The evidence that led here (don't re-derive)

`docs/soak-reports/20260801_043515/tail_supervisor_playlive_2026-08-01_04-19-27.log`:

- `[roam] ROAD to Vermilion City: on Cerulean City — billed leg south toward Route 5`
  → the billed-road router (commit `da8a7f9`) works; the DECISION is correct every tick.
- `[travel] start map=(3,3) ... -> map (3,23)` then
  `south connection band cols: [15..32, 48]` then `RESULT: head_to_gym -> no_path`
  with NO MOVEMENT — from every position tried. The band proves the Route 5 overlap
  row IS walkable; BFS just can't reach it (ledge row, see fix #1).
- After the south failures, travel legs to map `(3,22)` (Route 4) appear — the
  backward graph fallback → the checkpoint trail alternates
  `cerulean-city` / `route-4` every ~30-60s (see the auto-checkpoint inventory in
  `docs/soak-reports/20260801_043515/resume_marathon.log`).

## Fix history this session (all pushed to main)

- `da8a7f9` — billed road prioritized over world-graph in `head_to_gym`; Teleport
  rescue retired from all automatic paths (it was the Abra→Summary menu loop:
  `fixed_row=0` selects SUMMARY, not TELEPORT); `WEDGE_MEM_SCHEMA` bumped to 3
  (wholesale wedge-memory purge at next boot).
- `34aa478` — fieldable-floor (moveless mons like Abra don't gate the bench pin);
  momentum latches on every badge win; one-shot momentum seed for the Misty win.
- `5c5b960` / `af396c2` — momentum rule persisted (`momentum.json`), seam-thrash
  breaker, forward-grass bias via KB `billed_road`.
- `514a99a` — post-Misty deafness: Claude calls get 90s timeout + 1 retry
  (`kira/brain/claude_gateway.py`); `_TrackedLock` + turn-lock watchdog actively
  recovers hung conversation turns (`kira/bot.py`).
- `9b00788` — dominance margin 5→8, never dominant on shared type (water-vs-water);
  heals never go to a mon out-leveled by 8+ (ace-first healing).
- `69b9981` — Misty-gym talk-loop: `_engage_trainer` reads defeated-state
  authoritatively, drains post-battle dialogue, session-wide `talked` set.
- Earlier: watchdog `hold/release` around all LLM calls and voice pacing (the
  original "LLM in the reflex path" fix), creator-order LAW system
  (`creator_order.json`, regex latch in `kira/bot.py`), HUD active-mon + run-time
  seed, nickname RAM writes, damage-aware weakening, catch flow for flee-risk
  species (Abra: throw immediately).

## The workflow (how code reaches the PC, how Kira runs)

- Kira runs on **Jonny's Windows PC** (`G:\JonnyD\NeuroAI_Bot`). This Mac only writes
  code and pushes to GitHub (`JonathanDunkleberger/Kira_AI`, branch `main`).
- Jonny runs: `cd G:\JonnyD\NeuroAI_Bot; powershell -ExecutionPolicy Bypass -File
  pokemon_agent\resume_marathon.ps1` — it kills Kira processes, `git pull`, collects
  forensics into `docs/soak-reports/<ts>/` and **pushes them back to GitHub** (that's
  how you get logs: `git pull` here after he runs it), then launches per
  `PROMOTE_TARGET.txt` (`CANONICAL` = launch canonical save; `CKPT <substr>` =
  promote newest matching auto-checkpoint; after fix #3, a missing file also =
  CANONICAL).
- **The canonical save is sacred** — never write it directly; everything flows
  through checkpoints/banking. No ROMs/saves/Nintendo assets in the repo, ever.
- Verify by running, not reading: recon scripts headless on the PC are the proof
  standard. On this Mac there is no emulator/ROM — synthetic/mocked recon tests only.

## Known live pitfalls (cost real hours; do not re-pay)

- Menus by cursor readback, never state bytes alone (pitfall 13).
- `hm_teach.use_field_move` `fixed_row=0` = SUMMARY in FRLG party submenus, not the
  field move. Teleport rescue is retired; don't resurrect it.
- The world graph has a poisoned/seeded Diglett's Cave corridor that routes
  Cerulean→Vermilion the long way west — that's why the billed road must win.
- `resume_marathon.ps1` pushes soak reports; a git *push* stderr line in his console
  ("NativeCommandError ... main -> main") is NOT a failure — PowerShell renders
  git's stderr progress as an error. Check for the `x..y main -> main` line.
- Jonny's Twitch is now `TheKiraAgency` (alias registered at boot; needs
  `TWITCH_CHANNEL_TO_JOIN` right in the PC `.env`).

## Immediate next steps

1. Confirm the worker's commit landed on `origin/main` and tests pass.
2. Tell Jonny to rerun the one-liner. She should launch (CANONICAL default), walk
   Cerulean → hop the south ledge → Route 5 → Underground Path → Route 6 → Vermilion.
3. When his next soak report lands, `git pull` and grep the new supervisor log for
   `ROAD to Vermilion` → the travel start → confirm **map flips to (3,23)/Route 5**
   and no more `no_path`. That log line pair is the acceptance test.
4. If south STILL fails after the ledge fix: the next suspects are the exit-press
   executor (post-BFS path walk assuming in-bounds tiles) and the band/goal
   interaction — instrument, don't guess.
