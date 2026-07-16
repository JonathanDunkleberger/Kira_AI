# TEAM PLANNER — forward-planning team-building brain (design spec, 2026-07-09)

**THE FIX:** she improvises at each gym door because nothing plans her team across the game. Build a
STANDING planner that reads a real guide, plans a team toward the E4 from the start, and catches/evolves/
teaches/levels DELIBERATELY IN ADVANCE — voiced in character, at a watchable pace, to a fresh
bedroom→credits win.

**END STATE:** press START → watch Kira plan, catch, build, evolve, and win her way to credits like a
smart, lovable, guide-literate human who planned her whole team.

## BUILD ORDER (dependency order — VERIFY each layer before stacking the next)

`A (data) → B (brain) → C (executor) → fresh-run proof`. Each gate below must pass on disk before moving on.

## SOUL + WATCHABILITY = CORE REQUIREMENTS (not hedges)

- **SOUL:** every plan decision flows through the `plan_note` VOICE seam (the sanctioned interface) as HER
  forward-looking idea — "I'm grabbing an Abra now, I'll want Psychic for Koga *and* Sabrina." She picks
  her target team with personality (seeded by her actual starter + favorites), NAMES mons, has opinions.
  The archetype is a MENU she chooses from, never a solver dictating. The LLM oracle can still flavor/override.
- **WATCHABILITY:** level targets = leader-ace + small margin (beatable, brief struggle) — NEVER max-grind.
  Detours are BOUNDED (efficient routes, targeted catch, hard grind cap) and only fire when an acquisition
  is DUE and cheaply reachable. Narrate every detour. "Win most fights, learn fast, keep moving." A
  grind-wall is a FAILURE, not a success.
- **FIREWALL:** mode-side Pokémon brain ONLY. Core Kira general personality + persona + the two-bucket
  firewall are off-limits; `plan_note` is the ONLY voice interface. NEVER write states/kira/. Plan-state
  banks to the dev/campaign line (sanctity bundle), per-timeline.

---

## PART A — DEEP KNOWLEDGE BASE (the guide). `pokemon_agent/gamedata/`

Hand-authored from FRLG knowledge + Bulbapedia (pret not local), **VERIFIED against live RAM** where
possible (`read_enemy_species`/`read_enemy_level` on gEnemyParty at each real fight — the shift-9 method).

- **`frlg_rosters.json`** — full per-mon teams (NOT just ace names):
  `{"gyms":{"Brock":[{"species":"geodude","level":12,"moves":["tackle","defense-curl","rock-tomb"]},{"species":"onix","level":14,"moves":[...]}], ...},"e4":{"Lorelei":[...],...},"rivals":{"gary_ssanne":[...],"gary_e4":[...]},"champion":{...}}`
  GATE A1: loads; each gym/E4 team present; spot-check ≥3 vs live RAM (Brock, Koga=Koffing L37/Muk L39/Koffing L37/Weezing L43, Sabrina).
- **`frlg_evolutions.json`** — `{"charmander":{"into":"charmeleon","method":"level","level":16},"eevee":{"method":"stone","options":{"water-stone":"vaporeon","thunder-stone":"jolteon","fire-stone":"flareon"}},"pikachu":{"method":"stone","stone":"thunder-stone","into":"raichu"},"abra":{"into":"kadabra","method":"level","level":16},"kadabra":{"method":"trade","into":"alakazam"}, ...}`
  GATE A2: loads; ≥40 lines incl. all starters, stone evos, Abra line, trade cases.
- **`frlg_learnsets.json`** — `{"bulbasaur":{"level_up":[[1,"tackle"],[7,"leech-seed"],[13,"vine-whip"],[20,"razor-leaf"],...],"tm":["tm06","tm09","tm22",...]}}` (151 species; heavy — author in passes, prioritize keeper/starter/answer species first).
  GATE A3: loads; the ~20 team-relevant species complete; TM-compat present for coverage moves.
- **`frlg_tms.json`** — full TM01–TM50 + HM01–HM08: `{"TM13":{"name":"ice-beam","type":"ice","power":95,"where":"Celadon Dept 4F (~4000)"},...}`.
  GATE A4: loads; 50 TMs + 8 HMs; key coverage TMs have `where`.
- **`frlg_encounters.json`** (or extend species_quality) — route→species table for TARGETED catching:
  `{"Route 24":[{"species":"abra","rate":"..."}],"Route 25":[...],"Diglett's Cave":[{"species":"diglett"}],...}`.
  GATE A5: loads; every keeper in the team-plan has a resolvable catch location.
- **`frlg_team_plan.json`** — THE curated whole-game archetype(s) she plans against:
  `{"archetypes":[{"name":"balanced-classic","slots":[{"role":"grass-starter","species":["venusaur"],"covers":["grass","poison"]},{"role":"psychic-sweeper","species":["alakazam"],"acquire":{"species":"abra","where":"Route 24/25","by_badge":2},"covers":["psychic"]},{"role":"ground","acquire":{"species":"diglett","where":"Diglett's Cave","by_badge":3},"covers":["ground"]},{"role":"ice/water-tank","species":["lapras"],"acquire":{"where":"Silph gift","by_badge":6},"covers":["ice","water"]},{"role":"fire","acquire":{"species":"growlithe","where":"Route 7/8"}},{"role":"flyer/normal","species":["snorlax","pidgeot"]}],"coverage_target":["grass","psychic","ground","ice","water","fire","electric","flying"],"level_milestones":{"Brock":14,"Misty":21,"Surge":24,"Erika":31,"Koga":45,"Sabrina":45,"Blaine":49,"Giovanni":52,"E4":56}}]}`
  GATE A6: loads; covers every gym+E4 weakness type; every acquire has a location + a by-badge deadline.

## PART B — THE STANDING TEAM PLANNER (the brain). Extend `pokemon_planner.py`.

A `TeamPlanner` (new class, or StrategicPlanner extension) holding a PERSISTENT plan-state, banked in the
campaign sanctity bundle (`team_plan_state.json`; dev/campaign line only).

- **plan-state:** `{archetype, target_slots:[{role,species,covers,acquire,evolve_to,evolve_at,status:planned|acquired|evolved}], acquired:{slot->live mon}, milestones:[{threat,level_target,needed_types,needed_moves}], history:[...]}`.
- **API:**
  - `init_plan(party, badges, soul)` — pick an archetype seeded by her ACTUAL starter + soul favorites (SHE chooses; voiced). Set milestones from `frlg_rosters`+`frlg_team_plan`.
  - `assess(party, badges, bag, dex)` → scans ALL remaining threats (current badge → E4, NOT just next),
    aggregates needed coverage, and returns the SINGLE highest-leverage next action + a WHY (e.g. "Psychic
    needed for Koga+Sabrina+Bruno → acquire Abra now, one catch serves three walls").
  - `next_action()` → `PlanAction{kind: catch_keeper|evolve|teach_tm|grind_to|develop_bench, species, where, level, tm, mon, why, voice}` + a plan_note-ready voiced line.
  - `on_acquire/on_evolve/on_teach(...)` — update + persist plan-state.
  - persistence: `save()/load()` to the bundle; resume-safe (Fix C / dense checkpoints).
- **Whole-game lookahead** (the missing brain): union of every future threat's `answer_types`, matched
  against target-team coverage + acquisition deadlines (`by_badge`) → earliest-due, highest-multiplicity
  gap first. This is what makes prep PROACTIVE ("I planned for Koga 2 towns ago").
- **SOUL:** target-team + each action carry a first-person voiced rationale; favorites/names respected.
- Flag-gated (`POKEMON_TEAM_PLANNER`, default on); mode-gated.
- GATE B: deterministic tests — given (party, badges) fixtures, assess() returns the correct next action
  (e.g. solo-starter@badge1 → "catch a teammate + a Psychic-line for the mid-game"; post-Surge thin bench
  → "level the bench + grab Abra before Koga"); plan-state persists + resumes identical.

## PART C — THE PROACTIVE EXECUTOR. Wire into free_roam + the spine.

Turn `next_action` into a real objective run BEFORE the wall (not at the gym door):
- **CATCH_KEEPER(species, where):** route to the encounter location (encounters table) → TARGETED catch
  (seek the species; extend `catch_one` to accept a target + use roster_judgment/keeper). 
- **EVOLVE(mon, method):** level-evo → ensure level (grind) + proceed; stone-evo → ensure stone (buy/find)
  + use it; **B-to-cancel for move-timing** where the plan says "learn move X first"; deliberate timing.
- **TEACH_TM(tm, mon):** acquire (buy Celadon Dept / find) + teach.
- **GRIND_TO(level, mons):** bring bench/laggards to the milestone level ahead of the gym.
- **Wiring:** a PROACTIVE PLAN step in free_roam (checked at town arrivals / each tick): if an acquisition
  is DUE and cheaply reachable, do it, THEN push to the gym. In the spine, planner-driven prep between
  gyms replaces the hardcoded GRIND/CATCH objectives. `prep_for_gym` becomes the LAST-RESORT safety net,
  not the primary path (the plan should mean she arrives already prepared).
- **Reuse:** catch_one (→targeted), grind, _drive_evolution, hm_teach, travel, encounters/where.
- **WATCHABILITY:** bounded detours; hard grind cap; only-when-due; narrate each; forward-progress bias.
- GATE C (real run): from a mid-game frontier, she PROACTIVELY catches/evolves/teaches ahead of a wall,
  arrives prepared, wins, keeps moving — verified from the log + party growth + watchable pace.

## FINAL PROOF
Fresh bedroom→credits headless run: she plans a team, executes it across the game, wins the gyms + E4 +
Champion with a built balanced 6, no permanent stuck, watchable pace. THEN write CREDITS.
