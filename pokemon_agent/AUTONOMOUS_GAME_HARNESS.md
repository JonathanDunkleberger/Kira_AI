# AUTONOMOUS_GAME_HARNESS.md — the portable autonomous-game-playing harness (game #1 = FireRed)

The real product is **NOT a FireRed bot** — it's a *generalizable* harness that drives Kira's personality
through autonomous play of any RAM-accessible game. FireRed is instance #1 / the proving ground. The
post-credits goal: port to game #2 (e.g. Pokémon Emerald) by **swapping the game-knowledge layer +
re-verifying with the same engine**, not rebuilding. This doc is the accumulated wisdom so game #2 starts
from the bedrock + pitfalls, not from scratch. (See CLAUDE.md rule 14.)

## THE CLEAN LINE — engine (reusable) vs game-knowledge (swappable)

**ENGINE (game-agnostic — keep free of hardcoded game facts where feasible):**
- The **look-ahead oracle** (`recon_longrun.py`): boot the real save, run the REAL decision loop headless at
  max emulator speed (~14× real-time), to GOAL or genuine STALL, then read the sped-up log for the real next
  blocker. Protects the canonical save (persists to a staging dir) + banks round-trip-verified checkpoints.
- **Checkpoints / resumable progress / durable-docs continuity** (`STATE_OF_PROJECT.md`, world/strat/soul
  sidecars). The harness + checkpoints + STATE doc are the durable state that survives context resets.
- **The 15-block bedrock player-competency map** (CLAUDE.md): the competencies ANY player needs for ANY game
  — forward-drive, team-building, strategic-grind, battle competence, resource/economy, healing/survival,
  stuck-resolution, spatial literacy, NPC/dialogue literacy, gate-unlock questline, danger awareness, etc.
- **The run-the-rope operating loop** (rules 8-13): max-speed look-ahead → blocker → diagnose → build the
  GENERAL fix → re-run → bank → advance. Solve once, reuse everywhere.
- **The firewall** (rule 12): core Kira identity sacred; mode-state behind the toggle; plumbing improvable.

**GAME-KNOWLEDGE LAYER (swappable per game — lives in `gamedata/` + clearly-tagged constants):**
- `gamedata/frlg_gates.json` — typed gates (HM_OBSTACLE/STORY_NPC/ITEM_GATE/BADGE_GATE) + capability chains.
- Warp tables, flags (badge/story), gym order, item/HM sources, map IDs, **Mart doors + stock orders**,
  type chart specifics, RAM offsets (`firered_ram.py`, `pokemon_state.py`).
- **PORTABILITY DEBT (FireRed specifics still coupled into engine code — isolate when porting):** the
  city/door/stock constants (`CITY_MART_DOORS`, `MART_STOCK`, `CITY_PC_DOORS`, gym-door coords, `GYM_SPINE`)
  live in `campaign.py`, not `gamedata/`. For game #2, lift these into a per-game KB the engine reads.

## THE PITFALLS (hard-won — each cost real time; expect analogues in every game)

1. **Don't assume the model knows the game.** Kira is a BLANK SLATE — without an explicit game-model (goal /
   win-condition / what a team is for / catching / the arc) her choices are random. Wire the player's mental
   model into her decision ctx as HER grounded understanding (soul-preserving: she understands like a player,
   still discovers unencountered content). → CLAUDE.md "FOUNDATIONAL GAME-MODEL".
2. **Reactive-only objectives strand her.** Recognising a gate only INSIDE the execution branch means at
   decision-time she picks a backward grind on equal footing. Open the forward objective PROACTIVELY each
   tick BEFORE building the action set; prune backward options.
3. **Underleveled-with-no-team wall.** Over-levelling a single ace is slow (low wild XP) and type-coverage
   gaps (one damaging move) still wall you. The real fix is team-building (catch + choose + field/level) +
   strategic grind (field the WEAK members, not the ace). A loss → build strength while staying pointed at
   the objective; never knock backward to a cleared dead-end.
4. **Disasm map-numbers are unreliable** — ALWAYS cross-check route/map IDs against live RAM (the disasm
   route-number export disagreed with live headers; e.g. Route 24 = (3,43), not a contiguous pattern).
5. **The Mt-Moon wrong-warp class** — a hand-spotted physical stuck (a ladder/warp that teleports you off
   the intended path). Capture non-obvious nav nuances EXPLICITLY in durable docs (rule 11). Treat OTHER
   warp tiles on a floor as avoid-tiles when pathing to a target warp.
6. **In-battle / in-menu actuation degrades on the LONG-RUNNING core.** Menu nav that works on a fresh core
   can wedge after the core has run a while (the libmgba continuous-core artifact). Navigate menus by RAM
   **cursor-readback** (verify each press moved the cursor), not blind taps. This recurs at every
   gym/Mart/E4 (bag use, party switch, buy list). The `_goto_bag` / `_mart_goto_row` cursor-readback is the
   template; the party-switch menu still needs it (gated `POKEMON_BATTLE_SWITCH=0`).
7. **Pocket-aware item reads.** Items live in separate bag pockets. Reading the wrong pocket silently
   returns 0 — e.g. `bag_count` (Items pocket) can't see Poké Balls (balls pocket +0x430), which made the
   Mart buy-verify ALWAYS fail for balls (she "couldn't buy balls"). Dispatch the count by item-id range.
8. **Misidentification by flaky actuation.** Identifying a thing by whether a menu actuates (e.g. "is this
   the Mart? does the buy list open?") is fragile — a long-core actuation failure reads as a false negative.
   Prefer a ROBUST structural signal (interior layout: clerk object at a fixed tile) over actuation. This is
   how the prior session mis-tagged the policeman-house as the Mart.
9. **Verify before build; observe the REAL failure.** Don't reconstruct a prior session's theory — run the
   thing and read what actually happens (the policeman NPC was stationary, not a wanderer; the Mart buy flow
   actually works; etc.). Default verification = the look-ahead, not bite-sized micro-tests (rule 8).
10. **Story-gates force a route order** — a parked NPC/obstacle (Slowbro/policeman until S.S. Ticket) means
    "go fetch X first." Recognise typed gates → derive an ordered questline from the KB → execute it via the
    travel layer (the general gate-unlock capability). Generalises to Cut/Surf/Strength/Flash/Fly/item-gates.
11. **A recovery checkpoint captured in an un-healable spot is POISON** — worse than none. The escape-hatch
    banks "last known-good" on every PROGRESSING (GREEN) tick; a weak-field grind that walks into a one-way
    ledge pocket (Route-4 (84,15): the Center is ledge-isolated from *every* grass tile — proven, 0/84) keeps
    scoring GREEN as it levels, so the snapshot is taken INSIDE the strand → on the faint, the escape-hatch
    reloads straight back into the pocket → infinite reload STALL. FIX: guard the capture — only bank from a
    Center-reachable (heal-safe) position (`_center_reachable_here`: own-map Center door BFS-reachable, or a
    known-safe route). Use the ACCURATE signal only: "can reach a border tile ⇒ can cross to the neighbour
    city" is a FALSE proxy (the pocket's east border is reachable but doesn't cross to Cerulean). Over-
    conservative = harmless (skip a tick); a false "safe" = fatal (re-poisons). Belt: on a true strand, fall
    back through the deep-wedge RING (gain-seam checkpoints) when recent-good is absent/declines.
12. **Two switch systems will fight each other.** A participation-XP GRIND switch (field the weak mon, swap
    it to the tanky ace turn 1 so it banks XP without a hit) and a MATCHUP switch (swap out an out-typed
    active) both firing in the same turn = the grind switch brings the ace in, the matchup switch immediately
    pulls it back out and re-fields the fragile mon → it faints → strand. This was a Route-4 strand root
    (not geography — the geography just made the faint un-healable). FIX: suppress the matchup switch while
    the participation grind is active (`not PROTECT_LEAD_GRIND`); during a grind the ace STAYS and tanks. This
    made bench-leveling safe ON the local grass — no need to route the fragile grind to a far "safe map"
    (which was also geographically blocked: Route 4's ledges are one-way, so Route 3 is unreachable on foot
    from Cerulean). General lesson: when two autonomous behaviors can act on the same tick, gate the lower-
    priority one OFF while the higher-priority intent owns the turn.
13. **THE IMMORTAL-BATTLE WEDGE — a stale menu-state byte defeats EVERY recovery layer at once.** After an
    in-battle item use, the "move list is open" RAM byte (MENU_MODE) stayed 2 → the open-check short-circuited
    True BEFORE the FIGHT-opening press was ever sent → cursor nav pressed into the wrong screen (cursor
    readback correctly refused to lie) → the move never fired ("didn't fire" then MISDIAGNOSED as 0-PP, since
    a wedged menu also produces no PP-drop) → the anti-wedge FLEE also failed against the same phantom state →
    the travel layer re-detected the STILL-OPEN battle as a fresh encounter and re-entered it ~50× (the same
    undamaged foe every time — ONE immortal battle, zero XP, the whole grind budget burned). THE TELL in a
    log: the "different" encounters are all the same species at exactly full HP. THE DOCTRINE (now twice-paid:
    the GBATTLE_PHASE counter lesson and this): NEVER trust a state byte alone — verify menus by
    CURSOR-RESPONSE (probe a press, read the cursor back; only a responding cursor counts as open). Every
    layered recovery (rotation → flee → re-entry) assumed the same wrong state, so all of them failed
    identically; a single verified-state primitive at the bottom fixes the whole stack.

14. **A press queued during a warp fade executes ON ARRIVAL.** Nudging through a door mat with a
    blind key sequence buffers the NEXT press through the screen fade — it fires the instant control
    returns and can step you onto the twin mat straight back inside (the Cerulean (31,9) re-entry:
    pop out of the house, instantly walk back in, zero travel steps logged). RULE: one press at a
    time, SETTLE ~30 frames, re-read the map, only then judge/continue. (Same family as pitfall 6:
    never trust an action landed without reading the world back.)
15. **Shortest-path planning prefers a NEAR hard-obstacle over the real long detour.** The
    NPC-allowing fallback BFS re-picked the cut tree (18 tiles) over the burgled-house corridor
    (~50 tiles) on every replan — a crossed→tree→crossed loop. When an obstacle is UNCLEARABLE
    (an HM you don't own), add it to persistent block memory the first time you classify it, then
    REPLAN — the long way only wins once the short way is remembered as closed.
16. **Phantom mojibake: a default-encoding read on Windows fakes story corruption.** `open()` without
    `encoding=` decodes UTF-8 sidecars as cp1252 — "Pokémon" reads as "PokÃ©mon" and you'll ship a
    "fix" for a file that was never broken. All sidecar reads/writes: explicit utf-8; validators must
    read explicit too (ours does).
17. **A one-way pocket makes 'go heal' a stall loop.** Post-gate regions you can't walk BACK from
    (Cerulean's south strip) leave the known Center unreachable; re-offering heal ping-pongs two
    tiles forever — and a 2-tile oscillation defeats a position-based no-move guard. On a failed
    heal-route: remember the map as heal-dead, suppress the offer while non-critical, and push to
    the NEXT town's Center (what a real player does). Clear the memo on the next successful heal.
18. **Forced-movement floors defeat tile-BFS planning — simulate the mechanic, BFS over REST states.**
    Spin tiles (and ice/currents in other games) redirect or carry the player, so a walkability BFS
    diverges the instant she touches one (the position-loop class). The general fix (`spin_nav.py`,
    proven on Rocket Hideout B2F/B3F, 2026-07-07): a deterministic glide SIMULATOR + BFS whose nodes
    are REST tiles and whose edges are whole glides, executed press-by-press with replan-on-divergence.
    Bonus truth: such a maze can be SEALED by a collectible object (an item ball) — "no route" can
    mean "pick the thing up first", which is also what a human does.
19. **Live object reads are DISTANCE-CULLED; whole-map planners need the STATIC template list.** The
    live object array holds only nearby spawns, so a long plan routes through a tile where a far item
    ball/NPC will be standing on arrival. Read the map header's object-event TEMPLATES + spawn flags
    (collected/gone truth) and union with the live read (`travel.read_object_templates`).
20. **An open dialogue box eats EVERY direction press.** Any press-executor (glide plans, mount
    rituals) must drain boxes before pressing — a beaten trainer's re-talk box silently starves a
    whole plan into a fake wedge.
21. **Scene-triggered battles start DURING travel; an observer that attaches late mis-detects.** A
    rival/boss cutscene fires as the approach leg arrives; by the time the battle wrapper attaches,
    fingerprint detection can miss and the win never reaches the story ledger. Record from a
    start-of-battle hook, not attach-time; until then backfill from the log (ground truth > blank).
22. **Interior CLIMBS are not destinations.** A questline that navigates to a building's door and
    then "talks NPCs" strands on any multi-floor dungeon (tower/hideout). The proven shape: a STATE
    MACHINE on the current map (heal bounces re-dispatch; beaten trainers stay beaten) + warp-DEST
    routing (pick the warp whose table destination is the next floor — never hardcode stair coords;
    directional stairs may need mounting from ON the tile, and the mount press is routinely eaten as
    a turn — press ×3). Two scripted strikes (hideout, tower) are the spec to fold into the questline.

23. **NUKE-TRADE FOES need a pre-emptive answer, not a reactive one.** A Self-Destruct-class move
    one-shots even a dominant ace at ANY matchup — no damage-race logic sees it coming (Koga's L37
    Koffing detonated on a L54 Venusaur turn one; the bench then fed itself piecemeal). The human
    answer is species-triggered: know the game's nuke families (KB layer) and open with SLEEP —
    a sleeping foe can't detonate, a KO'd one can't either. Reactive threat models (is it
    super-effective vs me?) never fire on these.
24. **A PLAN THAT CANNOT EXECUTE must STAND DOWN, loudly.** "Train the team first" is correct advice
    that becomes a stall engine when no training ground is reachable (one-way ledge pockets, water
    routes, unexplored frontiers). If the advisory layer keeps folding the plan while its executor
    keeps failing, the oracle loyally picks the failing action forever. Count consecutive dry
    attempts; after N, drop the plan and let the forward objective (rematch, next town) win. Reset
    only on REAL execution (grinding happened), never on mere travel arrival — an A↔B shuttle
    'arrives' every tick.
25. **ONE-WAY GEOGRAPHY breaks flat map-graphs.** A ledge-pocket makes reachability POSITIONAL: the
    same map is enterable westbound but its east exit is gone once you're in the pocket. A flat
    (map→map) edge graph will keep proposing routes the pathfinder can't walk. Remember (from-map →
    target) travel failures and veto them as candidates; remember grind-dead maps (grassless/strand-
    only) separately — the optimistic "route number ⇒ has grass" heuristic sent her to a water route.
26. **The stale display-struct at attach cuts BOTH ways.** Joining a battle mid-scene, the save's
    battle display struct can hold the LAST fight's data: a stale foe-corpse mis-arms "enemy already
    down" (harmless — fresh-enemy detect recovers) but a stale OUR-corpse mis-arms the forced-switch
    chain, which silently B-drains a live fight until timeout. Every re-entry flag needs a live-read
    disarm: if the flag's premise isn't true NOW, drop it and fight.
27. **TAP-TURN vs STEP is the mother of phantom walls (grid games).** A short d-pad tap in a
    direction the avatar isn't facing only TURNS them; coords don't change. A stepper that reads
    "didn't move" as "blocked" — and then blind-nudges a fixed direction "to unbox" — UNDOES the turn
    and oscillates forever, or drifts the avatar in the nudge direction across the map. Weeks of
    "elevation-sealed tiles", "one-way columns" and dead-marked walkable tiles in Silph Co were this
    ONE actuation bug (strike13 trace: tap=turn-only; 26-frame hold=TWO tiles/overshoot; the cure is
    tap-again — turn first, then exactly one step). LAW: before believing any step-failure is
    geometry, retry the same key once; and calibrate tap/hold frame windows per game EARLY — turn
    threshold, one-tile walk, two-tile run — they gate every mover built on top.

28. **A REGION FLOOD without bounds or elevation welds sealed rooms together (2026-07-07,
    the Saffron Gym strikes).** Two leak classes in any homebrew flood-fill: (a) BORDER tiles
    outside the playable rectangle read collision-0 — an unbounded flood escapes around the
    border ring; (b) the map-grid u16 carries an ELEVATION nibble (bits 12-15) the 2-bit
    collision read ignores — void strips read collision-0/elevation-0 beside elevation-3
    floor, and the game blocks cross-elevation steps (the whole Silph "elevation-sealed"
    class). Planners must bound to the playable rect AND hold the seed's elevation (0xF =
    multi-level pass). Indoors this is load-bearing; outdoors prefer the engine BFS.
29. **Menu LISTS remember their cursor across close/reopen (2026-07-07, the PC storage
    menu).** After driving one submenu (DEPOSIT), the reopened list is NOT on its top item —
    a blind "A = first item" reopened the deposit GUI and nearly boxed the wrong mon. Any
    reopened list needs a cursor reset (UPs to the top) or a screen-verified pick.
30. **Interactables are findable by METATILE BEHAVIOR, not coordinates (2026-07-07, the PC
    console).** "All Centers share a layout" was false for stand tiles; the console IS
    behavior 0x83 (MB_PC) wherever it sits. Scanning the behavior byte finds consoles/maps/
    signs in ANY room — prefer behavior-scan over per-room coordinate calibration.
31. **Special battle UIs need their own handler + map-scoped dispatch (2026-07-07, Safari).**
    A zone can replace the FIGHT menu wholesale (BALL/BAIT/ROCK/RUN) — the battle agent's
    cursor model is meaningless there, and inside such a zone EVERY battle is that kind, so
    dispatch by map-id. End-of-battle drains use B everywhere a catch/gift flow can reach the
    nickname keyboard.
32. **Grass/hazard-free planning seals hazard-priced zones (2026-07-07, Safari West).** The
    engine's "avoid tall grass" layer is right for roads and DEAD WRONG where grass is the
    only road — the planner reads no_route and fallbacks wander onto exit warps. Zone-scoped
    movers must plan hazard-INCLUSIVE and pay the encounter cost through the battle handler.
33. **Scrolling lists lie twice: TRUE selection = visible cursor + scroll offset, and BOTH
    persist across close/reopen (2026-07-07, the E4 bag).** The mart taught row+scroll; the
    battle bag repeated it — nav by the raw cursor byte "reached row 5" while A landed on
    CANCEL off a scroll left over from the PREVIOUS open (items "selected but not consumed",
    the healing kit a coin-flip). Treat EVERY list UI in the game as (cursor, scroll) until
    proven single-page, and derive both addresses per-list (they differ per menu family).
34. **WAR-MUST-ADVANCE: in a turn-based game, submitting NO action is the one unrecoverable
    move (2026-07-07, the Agatha livelock).** A can't-flee fight where every move looked
    dead made the agent idle "rather than waste a turn" — but turns only pass when you act,
    so the foe never moved, the battle never resolved, and every recovery layer (abort →
    re-enter) just re-entered the same frozen state. She couldn't even LOSE her way to the
    respawn ratchet that would have refilled her PP. Any turn-based agent needs a bottom
    rung that ALWAYS submits something: worst usable move > forced fallback (Struggle) >
    never idle. Losing is progress; idling is the only true dead end.
35. **THE ABILITY LAYER: type-chart picks are wrong without per-target ability overrides
    (2026-07-07, Levitate).** "Earthquake is super-effective vs Poison" donated 4+ turns per
    Gengar — Levitate zeroes ALL Ground damage. A pure chart lookup is a half-truth in any
    game with per-unit passives; the effectiveness function must compose chart × ability
    (× item, × weather as they appear). One `_eff(move, enemy)` chokepoint, unit-tested,
    feeding EVERY pick path — never chart math inline at call sites.
36. **PP-ECONOMY / AMMO CONVERGENCE: when a gauntlet outlasts your ammo, the answer is a
    compounding loop, not a better single run (2026-07-07, the E4).** ~50 damaging PP vs 26
    bosses could not be won level-flat; no mid-gauntlet refill existed (verified, not
    assumed). The winning shape: every attempt banks XP + money (the ratchet) → each lap is
    stronger + better-kitted → one-shot rates rise → PP-per-KO falls → the distribution's
    tail crosses the finish. Generalize: attrition walls fall to persistent-gain loops;
    verify what persists across defeat (XP/items yes, cash halves) and SPEND-THE-WAD into
    the persistent form before dying.
37. **THE GAUNTLET-RESET LAW: multi-stage bosses reset to stage 1 on defeat — per-stage
    banks are DIAGNOSIS-ONLY (2026-07-07).** A mid-gauntlet savestate is a lie for progress
    purposes (the game will replay the whole chain); only the pre-gauntlet anchor + the
    persistent-gain ratchet are real. Bank rooms to STUDY them, never to resume them.
38. **THE ZERO-COST BOUNCE WINDOW: restart/relaunch runs only at the moment nothing is in
    flight and everything gained is banked (2026-07-07, whiteout-center).** Killing a run
    mid-attempt loses the lap's gains; the respawn point right after the auto-bank is free.
    Every long loop should surface its own "safe to bounce NOW" line in the log so an
    operator (or the night loop) never guesses.
39. **THE MENU-TIME ORDER LAW: a UI can physically REARRANGE the underlying data while open
    and restore it on close (2026-07-07, the FRLG party menu).** Row i IS slot i only while
    the menu is up; carrying an index across the open/close boundary mis-aims every later
    use (revives on corpses, switches to the wrong body). Re-derive selection indices AT
    menu time from the live struct, every time — never cache across a UI boundary.

**STATUS ADDENDUM (2026-07-07): GAME #1 SUMMITED.** FireRed credits rolled autonomously
(bedroom → 8 badges → E4 → Champion). The engine list above is what did it; the post-credits
phase is soul-polish (first-timer dialogue register, overheard-intel ledger, gift-mon
introductions, post-game context flip) — all engine-side patterns a game #2 inherits free.

**ENGINE CAPABILITY (added 2026-07-07): THE PAD-GRAPH ROUTER (recon_sabrina.pad_plan).**
Teleport mazes (warps whose dest is the CURRENT map) are routable with zero hardcoding: warp
events come in id order, so pad tile -> warps[dest_warp_id] tile is the whole edge list;
flood-fill walk-regions (warps+NPCs masked, bounded, elevation-held), meta-BFS over regions
with pad rides as edges, execute ride-by-ride with replan. Ports to any teleport/portal maze.

**ENGINE CAPABILITY (added 2026-07-06): THE DOOR PASS-THROUGH.** When an edge crossing has no
overworld route (fenced region), buildings are the remaining connectors: try reachable doors
(multi-warp buildings first — a connector fingerprint), walk the interior's warps farthest-first,
multi-hop up to 6 rooms (depth 1 = a pass-through house, depth 3 = an underground tunnel), pop out,
retry the edge. Remember proven connectors per map. This one primitive covers pass-through houses,
gatehouses, and under-city tunnels — every "the road is fenced; the way through is a building" gate.

**ENGINE PATTERN (added 2026-07-08, the descent): GROUNDED-LOCATION CONTEXT.** Every decision tick
LEADS with a grounded where-am-I line built ONLY from live reads (indoor/outdoor from map metadata,
terrain traits from confirmed visits), and an UNKNOWN place explicitly instructs curiosity-not-
assertion. PITFALL it kills (twice now): assuming a map GROUP encodes the setting — FireRed's
"Dungeons" group holds caves AND ships AND office towers AND the open-air Safari Zone; the model
confabulated "cave" inside a lab and inside the Hall of Fame. Classify per-map from the disasm;
keep the classification in the game-knowledge layer.

**ENGINE PATTERN (added 2026-07-08): VERBAL-TIC GOVERNOR.** Line-level repetition guards (near-
duplicate detection over a short window) cannot see a PHRASE tic recurring across many different
lines ("X is doing a lot of heavy lifting" every ~10 lines). Track distinctive n-grams across a
LONG window of her generated lines; when one recurs in >=3 distinct lines, fold a HARD ban into
the ask — with NO self-aware escape hatch (the "I almost said—" wink becomes its own tic).

**ENGINE PATTERN (added 2026-07-08): KEY-FIGURE SALIENCE (social fabric).** A human player treats
family/named characters differently from scenery NPCs. Curate a small map-keyed table of key
figures; un-greeted ones become first-class warmly-framed options, exert pull in the decision ctx,
persist greeted-state in the world model (survives resume), and a walked-past skip is VOICED once
(a deliberate choice the viewer hears). Salience feeds the choice; the pathfinder executes.

**ENGINE PATTERN (added 2026-07-08): THE MACHINE PRE-GRADE SWEEP.** Before human spot-watches,
run the real loop headless off every banked arc checkpoint and grade the machine-visible humanity
harness properties (locomotion wedges, liveness/void trips, grounding gaps, decision variety) →
a ranked riskiest-arcs report that PICKS the human's watch list. Graders need GRADING budgets:
cap grind/battle time per action or one tick eats the arc window (the 78-encounter lesson).

**ENGINE PATTERN (added 2026-07-08, night shift 4): TERRAIN-CAPABILITY ROUTING (water/Surf).**
A traversal capability the player EARNS (surf, lava-walk, double-jump) must reach the PATHFINDER,
not just the action menu — ours offered "use Surf" as a choice while the BFS held "water is not a
road" absolutely, so every sea road was unroutable and the agent stood announcing plans at the
shore. The general shape: (1) classify the special terrain in the grid (by tile behavior, not
collision — special terrain often reads collision-0); (2) plan in capability LAYERS, preferred
terrain first, special terrain as last resort; (3) the EXECUTOR owns the transition ceremony (our
surf mount = face water + A-prompt confirm; dismount is automatic) — a raw movement press at the
boundary reads as a wall and poisons the fail-accounting.

**ENGINE PITFALL (added 2026-07-08, night shift 4): THE HARDCODED FORWARD-DIRECTION.** Any
"forward = <compass>" assumption baked into a proactive helper WILL misfire when the road bends:
our every-tick gate recognizer probed "south" (badges 1-3 geometry) and at a town whose road ran
WEST it armed an unrelated southern gate's questline, which then outranked the correct billed
road forever. Forward direction must be DERIVED (from the billed road leg under her feet) with
the heuristic only as fallback. Sibling pitfall: "no forward edge + nothing unexplored" is NOT
"arrived" — a dead-end pocket satisfies both; arrival needs anchor/bend membership, else the
step interacts with nothing forever. And when a route action is proven structurally dead on a
map, the forward-drive prioritizer must stop reframing it as the dominant pull (it pruned every
working option and left one dead action as the whole set).

**ENGINE PITFALL (added 2026-07-08, night shift 4): THE ENTER-STUCK-EXIT PING-PONG.** A goal
building whose interior defeats the general navigator (teleport-pad maze) creates a movement
livelock the no-progress guards can't see: enter (moves), wedge inside, route-out-to-city (moves),
re-enter — every cycle moves, so movement-cleared dead-route memory never bites. Count consecutive
interior failures per GOAL (not per tile); two strikes → park the approach structurally on that
map + voice one honest "it has me beat for now" beat. Movement is not progress.

## "HOW TO TEACH KIRA A NEW RAM-ACCESSIBLE GAME" (the port playbook)
1. **RAM map first:** find party/inventory/money/flags/map-id/coords offsets (use the game's disasm —
   pret/* for Pokémon — + a RAM differ; cross-check live). Populate the per-game KB.
2. **Wire the game-model** into her decision ctx (goal, win-condition, what progress means, the arc).
3. **Stand up the look-ahead harness** pointing at the new ROM + save; reuse the engine wholesale.
4. **Run the rope:** look-ahead → blocker → map the blocker to a bedrock competency → reuse the general
   capability (or build it general the first time it appears) → re-verify → bank → climb.
5. **Resources when stuck:** the disassembly (authoritative), wikis (Bulbapedia), prior solved projects
   ("GPT/Claude beats Pokémon"), and your own reasoning. Almost never bounce a blocker to Jonny (rule 13).

## STATUS (FireRed, instance #1): climbing — see `STATE_OF_PROJECT.md` §0 for the live frontier.
