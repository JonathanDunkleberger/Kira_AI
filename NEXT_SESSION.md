# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 8 IN-FLIGHT) — READ FIRST

## SHIFT-8 STATE: SNORLAX STRIKE written+wired (UNCOMMITTED), VERIFYING from a Route-12 fixture.
Shift 7 wired the Rocket Hideout strike (Silph Scope, commit 93ee1ab) + Pokémon Tower strike (Poké
Flute, commit f4ac740), both verified reaching their objective. Shift 7 then hit the NEXT blocker:
after the Flute she walks to the **Route 12 Snorlax** and STALLS at `(3,30)@(12,0)` — nothing wired
"play the Flute at the sleeping blocker", so head_to_gym no_paths into the body (SPLIT-MAP DEAD ROAD).
That is the run7 STALL (banked to G:/temp/longrun/banked_STALL — preserved as the fixture below).

**SHIFT-8 WORK (uncommitted at this writing):**
- NEW `pokemon_agent/snorlax_strike.py` — faithful in-loop port of the champion's recon_snorlax.py:
  pass Route-12 north gate -> face the Snorlax body (disasm (14,70)) -> A -> "...play the POKE FLUTE?"
  -> YES -> it wakes and ATTACKS (catchable wild L30) -> beat it -> FLAG_WOKE_UP_ROUTE_12_SNORLAX
  (0x253/595) sets, road south opens. run_strike(camp, log, dbg_dir) returns woke_snorlax|not_here|failed.
- `campaign.py` (2 hunks): (1) `_questline_strike` registry gains key
  `('flag','FLAG_WOKE_UP_ROUTE_12_SNORLAX') -> _snorlax` importer. (2) NEW hook at the STEP handler
  (`_run_questline_step` ~line 5904) — calls `_questline_strike(step)` BEFORE dir/bend discovery,
  because the snorlax errand never reaches the 'arrived' `_questline_interact` hook (head_to_gym
  no_paths at the blocker first). Cheap no-op unless she's ON Route 12 (3,30). Verified wiring:
  questline `_step_from_cap` gives the snorlax cap success=('flag','FLAG_WOKE_UP_ROUTE_12_SNORLAX')
  (sets_flag branch), matching the registry key; head_to_gym -> _run_questline_step -> _questline_strike.

**VERIFY BOOT (she's AT the snorlax with the Flute, badges=4):**
`.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py snorlax_face.state 15`
(states/workshop/snorlax_face.state = run7 banked_STALL: Route 12 (3,30), Venusaur L44 + weak bench,
Poké Flute in bag, Silph Scope done, badges=4.) Watch for `🎯 QUESTLINE STRIKE: Route 12 Snorlax` ->
`woke_snorlax`. Debug frames land in G:/temp/longrun/snorlax_probe/*.png (snorlax_gate_fail /
snorlax_final). If the gate-pass logic mislands (y<15 branch), the frames + the "gate pass-through
didn't land on the south road" log line pinpoint it.

## IF SNORLAX VERIFIES: the rest of the badge-5 chain (keep climbing, don't stop)
Snorlax awake -> Route 12/13/14/15 SOUTH coastal road -> Fuchsia City -> **Koga (badge 5, Soul)**.
Koga = poison/confusion, L37-43 — she has no clean psychic/ground answer (bench is L9-15). Expect a
grind/team-build need before Koga. The badge-4->5 gym leg (Fuchsia routing) is next unproven ground.
NOTE the TEA is already obtained (Saffron guards open for the badge-6 Sabrina approach later).

## KEY FACTS
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages ALL persistence to G:/temp/longrun/stage (STAGE redirect, WIPED each run start);
  world_model+strat load from CANONICAL, soul from the boot bundle. Canonical Champion save NEVER touched.
- Fixtures (states/workshop/): surge_done (badge3 @ Vermilion), rt_mouth (Route 10 mouth), flash_done
  (post-Flash Viridian), erika_done (badge 4 @ Celadon), **snorlax_face (Route 12 @ Snorlax, Flute in
  bag — THE FRONTIER)**. states/campaign = SHERPA CANONICAL (Champion, untouchable). Commit per fix.
- Snorlax facts (game-knowledge layer): Route 12 = map (3,30); FLAG_WOKE = 0x253 (id 595); body disasm
  (14,70); north gate doors ((14,15),(15,15)). Isolated in snorlax_strike.py per rule 14.

## GATED-SHORTCUT PATTERN (durable — recurring) + the strike-registry pattern
- head_to_gym's warp-route (`_next_step_rideable`) is GATE/PHYSICS-BLIND + preempts the billed road
  through story-gated shortcuts / phantom cliff-sealed edges (Snorlax R12, Saffron guards, Route 10 =
  Rock Tunnel). Handled by `_story_gate_avoid` + Saffron-avoid + cave-pass priority.
- STRIKE-REGISTRY (shift 7-8): dungeon/blocker rituals the general room-tour can't crack are ported as
  bespoke in-loop strike modules (hideout_strike / tower_strike / snorlax_strike), dispatched from
  `_questline_strike` keyed by step.success. Dungeon errands fire at `_questline_interact` (door-hint
  first); face-a-blocker errands (snorlax) fire at the `_run_questline_step` hook (head_to_gym path).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead is at badge 4, post-Flute,
standing at the Route 12 Snorlax — verifying the wake strike toward badge 5 (Koga/Fuchsia).
Pop-in = `python pokemon_agent/watch.py`.
