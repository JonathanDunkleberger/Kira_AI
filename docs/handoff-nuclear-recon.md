# NUCLEAR RECON HANDOFF — Kira × Pokémon FireRed, 40–45h watchable playthrough

You are taking over from a previous agent session that hit its context limit. Read this whole
document before doing anything. Your single mission: **make "press go, walk away" real** — Kira
(the AI VTuber) plays Pokémon FireRed start-to-credits in 40–45 wall-clock hours, at a watchable
human let's-play pace, interacting with chat, with ZERO operator intervention. Jonny (the owner)
has watched multiple streams die to stuck-loops and is out of patience. Symptom-patching is over;
your job is to find and fix the SYSTEMIC causes.

## Environment and workflow (do not rediscover this — it works)

- Mac working copy (you are here): `/Users/jonnydunkleberger/Desktop/OG Kira`
- GitHub: `JonathanDunkleberger/Kira_AI` (branch `main`). You have push access via the local repo.
- The game runs on Jonny's Windows PC at `G:\JonnyD\NeuroAI_Bot`. You NEVER touch the PC directly.
  Jonny runs exactly ONE command there:
  `cd G:\JonnyD\NeuroAI_Bot; git pull; powershell -ExecutionPolicy Bypass -File pokemon_agent\resume_marathon.ps1`
- That script: taskkills the running bot, pulls main, collects forensics into
  `docs/soak-reports/<stamp>/` (canonical health.json, 1500-line supervisor log tails, campaign
  snapshot inventory), executes the directive in `pokemon_agent/PROMOTE_TARGET.txt`, pushes the
  report to GitHub, and relaunches (bot + `supervisor.py --timeline sherpa` free-roam).
- Directives you can pin in `PROMOTE_TARGET.txt` (file consumed+deleted after each run):
  `CANONICAL` (relaunch current campaign save), `SNAPSHOT <file.state>` (promote a banked recovery
  snapshot from `states/campaign/`), `SNAPSHOT kira/<file.state>` (promote a showtime segment
  checkpoint, e.g. `seg_cerulean.state`), `RESTORE_WORLD`, `MIGRATE_SHOWTIME`, `NEW_CAMPAIGN`.
- Your read-loop: push code + directive → Jonny runs the command → pull the new
  `docs/soak-reports/` → analyze → repeat.

## Current game state (as of 2026-07-30 ~10:30)

Campaign save: Cerulean City area, 1 badge (Boulder), party Wartortle ~L23 + Spearow ~L8, broke
(whiteouts halve money), zero Poké Balls at last check. Next objective: build team → Nugget
Bridge → Misty. She was teleported here via `SNAPSHOT kira/seg_cerulean.state` after a morning of
Mt-Moon loops.

## Today's incident timeline (all commits on main, 2026-07-30)

1. `07fed47` — wedge-episode budgets reset on GREEN only (fixed an overnight infinite
   reload loop) + **wedge memory persisted to disk** (`states/campaign/wedge_memory.json`, TTL 12h).
2. `ea10c88` (yesterday) — **watchdog tightened 15s→8s** (`POKEMON_WATCHDOG_STUCK_S`), travel
   STUCK_LIMIT 16→10, corner jiggle.
3. `6f2c406` — wild encounters can no longer become "strategic walls" (a whiteout to a wild
   zubat at Mt Moon B1F had rewritten all her goals for hours; `pokemon_strategy.py` now filters
   non-trainer losses + purges persisted `wild:*` walls on load).
4. `edff81f` — SNAPSHOT teleport machinery (worked; she spawned in Cerulean).
5. `af10c96` — dialogue three-layer loop net: 120s wall-clock cap per conversation
   (`dialogue_drive.py::drive` `max_wall_s`), loop-marking on `timeout` (not just `exhausted`),
   cross-call re-drain counter in `campaign.py::_drain_overworld` (3 box-open drains at the same
   tile within 120s → mark NPC resolved, persist).
6. **The net was live in the 10:16 relaunch and the stream STILL died the same way**: she resumed
   near the Nugget Bridge "Bill" NPC, re-entered the conversation, appeared to "analyze"
   instead of pressing A, then self-heal reset her a couple of tiles away, and the loop repeated.
   Chat was typing "Kira, press A". Jonny ended the stream. This is your starting crime scene.

## Verified architecture facts (from a full loop-class audit, file:line refs)

Engine lives in `pokemon_agent/`: `campaign.py` (~15k lines, the free-roam brain),
`dialogue_drive.py`, `travel.py`, `battle_agent.py`, `world_fingerprint.py`, `play_live.py`
(process entry + render/audio + voice), `supervisor.py` (auto-restart wrapper),
`pokemon_strategy.py`, `pokemon_planner.py`, `pokemon_voice.py`, `cave_nav.py`.

- Decision loop: `campaign.py::free_roam` ticks; each tick builds a HUGE oracle context (easily
  2-4KB of prose — see any `[soul] ORACLE decision` line in a supervisor log tail) and calls the
  LLM ("oracle") for the next action. She acts via deterministic actuators (travel, battle,
  talk, grind).
- Watchdog stack: `world_fingerprint.py::StuckWatch` (trips when world fingerprint + on-screen
  text sit still for `WATCHDOG_STUCK_S=8`s; fed every live frame from `play_live.py` ~303-308) →
  latched `_stuck_request` honored at the top of the roam tick (`campaign.py` ~13176) →
  disengage (B + step away + `_mark_wedge_spot` = PERSISTED blocked tiles) → repeat trips
  escalate: forced heal → save reload (`_escape_hatch_reload`, max 2/episode) → deep-wedge ring
  revert → abandon + Discord dead-man alert.
- `ProgressLedger` (macro GREEN/YELLOW/RED) **skips ticks where a dialogue box is up**
  (`world_fingerprint.py` ~215-234) — dialogue can never escalate RED.
- `StuckWatch` **treats every new dialogue page as progress** (text is in the key, ~304-326) —
  a cycling conversation resets it forever.
- `DialogueDriver.drive()` **never checks the stuck latch mid-conversation**; it also fires
  `line_sink` (dialogue-hint extractor) synchronously per line, and her voice-reaction system
  (LLM + TTS) rides dialogue lines. Measure what a single A-press actually costs in wall time.
- The questline system **clears `_talked_npcs` up to 2× per room** (`campaign.py` ~9313-9572) —
  deliberately re-engages NPCs. Marked/looped NPCs (`_npc_is_resolved`) survive that clear, but
  anything not yet marked gets re-talked.
- Oracle repeat-pick nudges are soft (ctx text only) and require non-GREEN macro; talking keeps
  macro GREEN, so an oracle that keeps choosing `talk_npc` is never hard-pruned.

## PRIME SUSPECT you must confirm or kill FIRST (the previous agent's strongest hypothesis)

**The LLM is inside the reflex path, and the watchdog cannot tell "thinking" from "wedged".**
An oracle call with that giant context plausibly takes >8s. While it runs, the emulator keeps
rendering (the stream must not freeze), the world fingerprint is static, so the 8s StuckWatch
trips DURING NORMAL THINKING. Consequences, each observed live today: constant self-heal
disengages ("she resets and skips a couple feet"), false wedge-spot marks at innocent tiles —
now PERSISTED to `wedge_memory.json` and compounding across restarts (blocked tiles poisoning
travel routing) — and conversations aborted mid-drive before any dialogue net can conclude.
Check: does anything pause/feed-gate StuckWatch during oracle calls, voice synthesis, or
read-along holds? (`grep` for pause/suspend/thinking around the StuckWatch feed in
`play_live.py`.) If nothing does, THIS plus the 15→8s tightening is likely the "weirdly often
self-healing" Jonny described, and the fix priority is:
1. Gate the watchdog (and any wedge-marking) while an LLM call / TTS / deliberate read-hold is
   in flight — "she is thinking" must be a first-class engine state, not indistinguishable from
   "she is wedged".
2. Audit `wedge_memory.json` content on the PC (next soak report) for false marks accumulated
   today; add a purge/validation (e.g. drop marks created within N seconds of an oracle call, or
   just reset the file once the gating fix ships).
3. Re-evaluate whether 8s is sane once thinking is excluded (it probably is; 15s was masking the
   real bug).

## Your mandate (in order)

1. **Architecture map first.** Read `play_live.py`, the `free_roam` tick loop, `DialogueDriver`,
   the StuckWatch feed, and the voice pipeline END TO END. Produce a one-page timing model of a
   single tick: what blocks on what, worst-case wall time of each stage (LLM call, TTS, hint
   sink, read-along holds, actuator execution), and which watchdogs are armed during each stage.
   Most of today's chaos lives in the interactions, not the components.
2. **Confirm/kill the prime suspect** with log evidence from the next soak report (grep the
   supervisor tails for `WATCHDOG` trips timestamped inside `[soul] ORACLE` windows).
3. **Fix the reflex/brain split.** North star: mechanical reflexes (advancing dialogue,
   navigating menus, walking a computed path) must NEVER wait on — or be interrupted because
   of — an LLM call. The LLM decides WHAT to do at seam points and narrates asynchronously;
   pressing A three times to exit a conversation requires zero intelligence and must cost zero
   latency. Restructure as needed; this is the systemic fix Jonny is paying for.
4. **Dead-weight audit** (Jonny explicitly asked): inventory what's old/unused — showtime/segment
   paths vs free-roam, default-OFF flagged features (`FIELD_MOVES_ENABLED`, `ITEM_PICKUP_ENABLED`,
   PC-box), duplicated recovery layers — and recommend/execute pruning that reduces the
   interaction surface. Do not delete showtime without asking; it's his demo-day tool.
5. **Validate headlessly before shipping.** Unit tests exist (`test_stuckwatch.py`, etc.). The
   bridge can boot save states headless; `states/campaign/replaced_*.state` backups on the PC
   (and everything in the snapshot inventory of report `20260730_101644`) reproduce today's
   scenes. Build a smoke that boots a save next to a chatty NPC and proves: conversation exits
   in seconds, no watchdog trip during a simulated 20s oracle stall, no false wedge marks.
6. **Ship via the workflow**: push, pin `CANONICAL`, have Jonny run the one command, read the
   next soak report, iterate. Never ask Jonny to type anything on the PC other than that command.

## Communication rules (Jonny's preferences, hard-earned)

- Lead with outcomes, plain language, no jargon walls. He is non-technical about internals but
  sharp about product symptoms — trust his observations; they've been right every time today.
- He is cost-sensitive: no performative recon. Every investigation must end in a shipped fix or
  a falsified hypothesis with evidence.
- The product story: OG Kira's marathon stream funnels viewers to the Kira web app
  (kira-web, separate repo at `/Users/jonnydunkleberger/Desktop/Kira/Kira_App`). The stream must
  be copyright-safe (game audio + her voice only). A "Kira, press A" chat moment is a product
  failure, not a cute bug.
- HUD (`web_dashboard/pokemon_hud.html`, OBS browser source reading `pokemon_hud.json`) can be
  edited live mid-run; game code changes need the relaunch cycle.
