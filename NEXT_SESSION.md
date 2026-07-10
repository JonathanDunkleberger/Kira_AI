# 🧠 MISSION PIVOT (2026-07-09) — BUILD THE FORWARD-PLANNING TEAM BRAIN — READ FIRST, SUPERSEDES ALL BELOW

**THE fix (CEO-approved):** stop reactive per-gym patching. Build a STANDING forward-planning team-building
brain so Kira plays like a guide-literate human who plans her whole team toward the Elite Four from the
start — catches/evolves/teaches/levels DELIBERATELY IN ADVANCE, voiced in character, at a watchable pace,
to a fresh bedroom→credits win. **END STATE:** press START → watch her plan, catch, build, evolve, and win
her way to credits like a smart lovable human who read the guide.

**FULL SPEC:** `pokemon_agent/TEAM_PLANNER_DESIGN.md` (schemas, class API, wiring, verification GATES).
Read it first every shift.

**BUILD IN DEPENDENCY ORDER — VERIFY EACH LAYER ON DISK BEFORE STACKING THE NEXT:**
- **PART A — DEEP KB** (`gamedata/`): `frlg_rosters.json` (full per-mon gym/E4/rival teams — verify ≥3 vs
  live RAM via read_enemy_species), `frlg_evolutions.json` (species→method/level/into), `frlg_learnsets.json`
  (level-up + TM compat; team-relevant species first), `frlg_tms.json` (all 50 TMs + 8 HMs), a route→species
  encounters table, and **`frlg_team_plan.json`** (the curated whole-game balanced-6 archetype: coverage map
  + acquisition ORDER/locations + level milestones). Author from FRLG knowledge + Bulbapedia (pret NOT local;
  use WebSearch/WebFetch), verify against live RAM. Gates A1-A6 in the spec.
- **PART B — THE BRAIN** (`pokemon_planner.py`): a standing `TeamPlanner` with persistent plan-state
  (target team, acquired-vs-needed, next acquisition + WHERE, evolve/level targets), **whole-game lookahead**
  (union of every future threat incl. E4 → earliest-due, highest-multiplicity gap → "grab Abra now, it
  carries Koga+Sabrina+Bruno"), emitting PROACTIVE PlanActions. Persist to the campaign bundle
  (dev-line only). Gate B = deterministic tests + resume.
- **PART C — THE EXECUTOR** (free_roam + spine): run the next PlanAction BEFORE the wall — targeted catch
  (route to the keeper), deliberate evolve (B-to-cancel for move timing, stone timing), teach TM, grind to
  milestone. prep_for_gym becomes the last-resort safety net. Gate C = a real run: proactively prepared,
  wins, keeps moving, watchable.
- **FINAL PROOF:** fresh bedroom→credits, built balanced 6, no permanent stuck, watchable → write CREDITS.

**SOUL (core requirement, DO NOT STRIP):** every plan decision flows through the `plan_note` voice seam as
HER forward-looking idea (names mons, has favorites, excited-guide-literate-kid). The archetype is a menu she
CHOOSES from, not a solver dictating. **WATCHABILITY (core):** level targets = ace+margin, bounded/narrated
detours, HARD grind cap, "win most fights / keep moving" — a grind-wall is a FAILURE.

**FIREWALL:** mode-side Pokémon brain ONLY. Core Kira general personality + persona + two-bucket firewall
OFF-LIMITS (`plan_note` is the ONLY voice interface). NEVER write states/kira/. Commit-per-fix. VERIFIED
from disk/real runs. Run-until-credits + escalate is already set (`night_shift.ps1 $RunUntilCredits`).

**Prior reactive frontier (Sabrina, below) is now the FALLBACK context, not the mission — the brain is the mission.**

---

# TEAM-BRAIN BUILD — FRONTIER (2026-07-09, mission-pivot shift IN FLIGHT) — READ FIRST

## PART A (deep KB) — ✅ DONE + COMMITTED (2549dac). gamedata/ now holds frlg_rosters.json,
## frlg_evolutions.json (68 lines), frlg_learnsets.json (22 species), frlg_tms.json (50TM+7HM),
## frlg_encounters.json, frlg_team_plan.json (balanced-classic archetype). ALL GATES A1-A6 PASS on
## disk — re-run `.venv/Scripts/python.exe -u pokemon_agent/recon_kb_verify.py` to confirm. Rosters
## Bulbapedia-verified (E4/Champion/gyms fetched live 2026-07-09; Champion = Bulbasaur branch).
##
## PART B (the brain, pokemon_planner.py TeamPlanner) — ✅ DONE + COMMITTED (55602d6). Persistent
## plan-state + whole-game lookahead + assess()/next_action() PlanActions (catch_keeper / acquire_special
## / evolve / grind_to / teach_tm / develop_bench) + first-person plan_note voice. GATE B PASS: 9
## deterministic tests + save/resume identity — re-run `.venv/Scripts/python.exe -u
## pokemon_agent/recon_teamplanner_test.py`. On the REAL Sherpa state (Venusaur L52 solo, badge 4) the
## brain diagnoses the solo-carry failure and prescribes 'catch an Abra — serves Koga AND Sabrina'.
##
## PART C (executor) — the LOAD-BEARING VOICE HALF is ✅ DONE + COMMITTED (19f7940): TeamPlanner.plan_note
## now folds into the decision/oracle ctx (campaign.py ~9098) with PRECEDENCE over the reactive
## StrategicPlanner (fallback when the brain is on-track/silent). campaign imports clean; wiring verified.
## THE REMAINING WORK = the EXECUTOR DRIVE HALF (the actual proactive navigation): turn next_action into a
## real objective run BEFORE the wall —
##   1. CATCH_KEEPER(species, where): route to the encounter location (frlg_encounters keepers/areas) ->
##      TARGETED catch (extend the existing catch_one to SEEK a specific species, keep it). Reuse travel +
##      catch machinery already in campaign.py. This is the #1 piece (unblocks Abra/Diglett/Growlithe).
##   2. Wire a PROACTIVE PLAN step into free_roam: at town arrivals / each tick, if next_action is a DUE
##      catch/evolve/teach and cheaply reachable, DO it, then push to the gym. prep_for_gym becomes the
##      last-resort net.
##   3. GATE C = a real recon_longrun where she PROACTIVELY catches Abra before Koga, arrives prepared,
##      wins, keeps moving (watchable). Then the FINAL PROOF = fresh bedroom->credits.
##   4. PLAN-STATE PERSISTENCE (rule 17, do it WITH the executor): add ("team_plan_state.json", ...) to
##      the sanctity bundle list (campaign.py:9745) + call self.team_planner.save(STATES_CAMPAIGN) in
##      _continuity_save and .load() on resume — so the taught/acquired HISTORY survives a hard kill.
##      DEFERRED this shift on purpose: history isn't load-bearing until the executor consumes it, and
##      the brain re-derives slot-status from the live party each tick, so resume already works today.
## RISK NOTE: Part C drive touches free_roam nav arbitration (the shared plumbing that burned prior
## shifts) — build it isolated/guarded behind POKEMON_TEAM_PLANNER, verify each piece with a look-ahead.
## The catch primitive to extend is catch_one (campaign.py:3957) — add a target_species param that
## FLEES non-targets + force-catches the target (roster_judgment stays for untargeted forward-catch).
## BOOT: `.venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py fuchsia_gate.state 15` (Venusaur L52
## @ badge 4, the brain will voice 'catch Abra'); read the [teamplan] fold in the log.
##
## FALLBACK reactive context (Koga/Sabrina, potion-stall) is below — NOT the mission. The brain is.

---

# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 10 IN FLIGHT) — reactive fallback context

## SHIFT-10 KEY FINDING (verified e2e): **POTIONS BEAT KOGA.** Venusaur L51 + Hyper/Super Potions
## stall-and-wins Koga cleanly (in-battle actuation FIRES on the long core). Proven by injecting
## potions into the bag (`inject_potions.py`) -> `recon_longrun fuchsia_potions.state`:
##   -> *** KOGA BADGE obtained *** -> advanced to Saffron -> reached **SABRINA's gym (badge 6)**
##   -> STALLED on Sabrina's gym-layout puzzle + coverage-mon indecision (alakazam psychic wall).
## So the potion-stall strategy carries her a WHOLE gym past the shift-9 frontier. NEW frontier = SABRINA.

## THE REMAINING WORK TO MAKE THE KOGA WIN REAL (unaided): she won with INJECTED potions. To bank a
## genuine badge-5 checkpoint she must BUY potions at Fuchsia Mart (currently UNMAPPED — the
## Cerulean-Mart-unmapped class). IN PROGRESS this shift:
##   1. recon_fuchsia_mart.py — identify the Fuchsia Mart door among the learned building warps
##      (candidates (24,5),(11,15),(28,16),(14,31),(38,31),(19,31); gym=(9,32)->int(11,3),
##      PC=(25,31)->int(11,5) already known). Add FUCHSIA_MART_DOOR to CITY_MART_DOORS + MART_STOCK
##      rows (control-verify by a live bag-delta buy, like Cerulean/Vermilion).
##   2. Wire GYM-PREP (or the roam stock_up) to actually BUY potions at Fuchsia before entering Koga.
##      NOTE: in the potion run she went STRAIGHT from Fuchsia arrival to head_to_gym (never picked
##      stock_up) — so GYM-PREP must FORCE a foresight potion stock-up when walled at a gym city and
##      the bag is potion-poor. `_shopping_list(foresight=True)` + `buy_at_mart` already exist.
##   3. VERIFY: a FRESH (uninjected) `recon_longrun fuchsia_gate.state` buys potions -> beats Koga.
##      Then bank a real post-Koga checkpoint.

## SECONDARY (heal ping-pong, pre-existing, LOW priority): at boot fuchsia_gate she's hurt inside the
## Route-15 gate; heal_nearest can't reach Fuchsia's Center (split map) so it falls to the Viridian
## march (return_to_center) which PING-PONGS the gate floors ~5 legs before the range(20) cap returns
## 'stuck' and the shift-9 heal-dead breaker fires -> she pushes to the gym. WORKS but ugly/slow.
## Real fix (if time): teach heal_nearest that Fuchsia's Center is WEST across the internal gate
## (route via the gate warp like head_to_gym does) instead of a cross-region Viridian march.

## THE SABRINA FRONTIER (badge 6, after Koga is real): Venusaur (Grass/Poison) takes PSYCHIC x2 —
## dangerous vs Alakazam (fast, Psychic). Potions may not out-stall a fast special sweeper. Human
## answer per her own ctx: a BUG or GHOST or DARK coverage mon (she NARRATES wanting one). Also the
## Saffron gym is a warp-pad/teleporter puzzle (pad_nav exists). Ranked plan for the successor:
##   A. Try potions-stall vs Sabrina first (cheap; maybe Razor Leaf + Sleep Powder + Hyper Potion
##      out-damages before Alakazam sweeps — verify with LONGRUN_BATTLE_LOG=1).
##   B. If potions insufficient: catch a coverage mon (bug/ghost) — she already wants one.

## BOOT THE FRONTIER
- Koga acquisition work: `.venv\Scripts\python.exe -u pokemon_agent\recon_fuchsia_mart.py` (door probe),
  then edit CITY_MART_DOORS/MART_STOCK, then
  `LONGRUN_BATTLE_LOG=1 .venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py fuchsia_gate.state 15`.
- Sabrina probe (once Koga is real): boot the potion run's banked Saffron state or re-run from
  fuchsia_potions and read the Sabrina battle log.

## KEY FACTS (carried from shift 9, still valid)
- venv python: `.venv/Scripts/python.exe` (2-PID shim; SINGLE-RUN LAW — kill predecessors first).
- recon_longrun stages to G:/temp/longrun/stage (WIPED each run); banks to banked_<OUTCOME>. Canonical
  Champion save NEVER touched. `LONGRUN_BATTLE_LOG=1` surfaces per-move engine logs.
- Fixtures (states/workshop/): erika_done, snorlax_face, snorlax_done, fuchsia_gate (boots inside the
  Route-15 gate; heal-breaker crosses it -> Fuchsia -> Koga), **fuchsia_potions (fuchsia_gate + injected
  potions; PROVES the potion win — beats Koga -> reaches Sabrina)**.
- Koga team (FRLG): Koffing L37, Muk L39, Koffing L37, Weezing L43 — all POISON. Fuchsia gym juniors
  (arbok/sandslash) DO engage now and give XP (Venusaur L51->L52). NUKE-SLEEP handles the Self-Destruct
  koffing/weezing; Cut (neutral) + Razor Leaf chips; Hyper Potion out-stalls the poison chip.
- Item ids: Super Potion 22, Hyper Potion 21, Max Potion 20, Full Restore 19, Potion 13, Revive 24,
  Antidote 11. In-battle heal instinct fires at <=30% HP (<=50% vs super-effective foe); chooser picks
  use_potion. BATTLE_CRIT_FRAC=0.30 (battle_agent.py).
- Bag write: SaveBlock1 + 0x0310, 42 slots x4 (id u16, qty u16 ^ key); key = rd32(rd32(SB2)+0xF20)&0xFFFF.
  Write via `b.core.memory.u16.raw_write(addr, val)` (the `[addr]=` path is broken in this mgba build).

## DURABLE PATTERNS
- POTION-STALL beats a MOVEPOOL-WALL gym (Koga): when a solo carry's STAB is resisted but it has a
  neutral chip move (Cut) + a status lock (Sleep Powder vs Self-Destruct) + Hyper Potions, it out-lasts
  a poison tank team. The in-battle item instinct + recon chooser make it autonomous — the ONLY missing
  piece is ACQUIRING the potions (map the town Mart).
- INTERNAL MAP-SPLIT class (Route 12/15 gates): halves share one map id, connect only via a gate
  building; the map-granular graph is blind to it, so from the wrong side edge/center/GRASS bands read
  UNREACHABLE. head_to_gym owns the crossing (warp-routing); heal_nearest does NOT (falls to Viridian).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa look-ahead PROVEN to clear Koga
(badge 5) with potions and reach Sabrina (badge 6). Making it unaided (buy potions at Fuchsia) is the
in-flight work. Pop-in = `python pokemon_agent/watch.py`.
