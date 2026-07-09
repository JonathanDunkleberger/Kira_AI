# 🚂 TONIGHT'S NIGHT-TRAIN MISSION (2026-07-09) — READ FIRST, SUPERSEDES ALL BELOW

**GO/NO-GO BAR:** prove Kira can complete a fresh FireRed run at a watchable ~25-35 hr human pace —
catches + levels a team, ADAPTS after a loss (loss → grind/get a new teammate, NEVER retry the same
solo team), and NEVER permanently sticks. She must complete headless start-to-finish clean before Jonny
streams her again.

**FRONTIER-FIRST — DO NOT RESTART FROM THE BEDROOM.** Bedroom→Brock→Misty→Surge (badges 1-3) are already
PROVEN + banked. Resume from the frontier and push forward.

**LAUNCH POINT (the frontier, NOT `FRESH`):**
```
.venv\Scripts\python.exe -u pokemon_agent\recon_longrun.py surge_done.state <minutes>   # ~25-30 per look-ahead
```
- `surge_done.state` = map (3,5) **Vermilion City, BADGE 3 (post-Surge)**, party **[ivysaur L31, rattata L15,
  spearow L15, ekans L9]** (ace + a badly underleveled bench — the bench-leveling IS the forward challenge).
- recon_longrun is canonical-SAFE (stages all writes; never touches states/kira). It reaps predecessors
  (SINGLE-RUN LAW). Field moves + item pickup are armed by default (shift-16 fidelity fix).

**CURRENT WALL:** badge-4 (Erika/Celadon) approach — shift-16 left it at the **Flash/Rock-Tunnel gate**
(stalled ~Route 10; the Flash aide is EAST via Route 11/Diglett's Cave, she routed north). Start here.

**THE LOOP (frontier-first, class-not-instance):**
1. `recon_longrun.py surge_done.state 25` → read the log → find the REAL next blocker.
2. At each GYM (Erika→Koga→Sabrina→Blaine→Giovanni), each RIVAL, and the E4: verify `prep_for_gym`
   fired and she arrived PREPARED (team size / levels vs the KB ace / type answer). Log **prepared-Y/N,
   win-Y/N, stuck-Y/N** per threat. If not prepared, she must catch/grind FIRST.
3. Every wall/stuck/loss-loop → discrete note → FIX the CLASS → re-run from the frontier → BANK a new
   checkpoint + rewrite THIS block's frontier when a stretch clears.
4. **RUN UNTIL CREDITS — DO NOT STOP EARLY (2026-07-09, Jonny).** The budget/balance stop AND the
   2-shift no-progress brake are DISABLED (`night_shift.ps1 $RunUntilCredits=$true`). The ONLY acceptable
   stop is a CLEAN FRESH RUN TO CREDITS (write `CREDITS` as line 1 of NIGHT_REPORT.md). Keep grinding
   shift after shift until then. Run SILENT — do not ping Jonny (he's asleep).

**ESCALATE-DON'T-QUIT (mandatory on every hard wall):** a gym/fight/nav gap you can't pass is NOT a stop.
Try MULTIPLE DISTINCT strategies like a resourceful human with a guide open: grind the bench HIGHER, catch
a better-TYPED counter (on grassy routes — see the catch-location gap), RE-ORDER the team to field the
answer, use ITEMS/TMs/HMs, consult `gamedata/frlg_strategy.json` (rosters/answers/keepers) + the
disassembly/Bulbapedia. Log the wall as a discrete note, attempt several genuinely different angles, bank
each attempt, and keep pushing. Only leave a `needs eyes:` note if you've truly tried it several ways and
it needs a human decision — and even then keep working OTHER parts of the run. `_bump_gym_prep` already
escalates prep each loss; lean on it + intra-segment resume so a loss re-preps HARDER, never replays the
same solo team.

**TODAY'S NEW TOOLS (built this session — USE THEM):**
- **`prep_for_gym`/`gym_readiness`** (Fix B, `fb4fa44`) — beat_gym now ENFORCES pre-gym readiness from the
  KB (catch a team / type answer / grind to ace+margin; on-loss `_bump_gym_prep` escalates). ⚠ **KNOWN GAP
  (likely first wall):** the catch fires at the gym CITY (no grass) → `no_reachable_target` → she stays
  thin. **FIX THE CATCH LOCATION** = catch on the grassy routes she passes through BEFORE the gym (fold an
  ensure-team catch into travel/grind on a grass route), so "loss → get a new teammate" actually lands.
- **Dense checkpoints** (`4acbf5b`): `watch.py --list` / `--at <name>` (canonical-safe reload).
- **Intra-segment resume** (Fix C, `83cc39b`): a deep failure resumes at the gym, not the segment start.
- **Gym-leader nav** (Fix A, `6d4de1d`): no more routing-around-the-leader thrash.

**FIREWALL:** gameplay/strategy/harness ONLY. Core Kira personality + persona + two-bucket firewall
OFF-LIMITS. NEVER write states/kira/. Commit-per-fix. VERIFIED from disk/real runs, not asserted.

---

# NIGHT-TRAIN SHIFT 1 (2026-07-09) — FRONTIER: badge-4 FLASH, blocker fully characterized — READ FIRST

**BOOT:** `surge_done.state` (rebuilt from `G:/temp/longrun/banked_GOAL/` → `states/workshop/surge_done.state`
+ sidecars) for the full look-ahead, OR the NEW fast fixture `states/workshop/route11_flash.state` (banked
this shift — boots AT Route 11, skips the ~15-min back-legs) via `recon_flash_errand.py` to iterate on the
downstream. Commit: `15a3c49` (+ this shift's follow-ups).

## WHAT SHIPPED + VERIFIED (commit 15a3c49 and follow-ups)
The badge-4 Flash wall was NOT one bug — it's a chain. Built THE FLASH ERRAND (general dex-prereq questline
+ a ported-from-`recon_hm05` errand). Each piece VERIFIED from real recon runs:
1. ✅ **Dex/owned prerequisite (general):** `frlg_gates.json` flash gate bills `requires_owned:10`;
   `questline.derive_questline` injects a synthetic catch step (`via='catch'`, `success=('dex',10)`) before
   the flash step when live dex<10; `_step_satisfied` handles `kind=='dex'`. Derive-test + live runs confirm.
2. ✅ **Errand ROUTING (back-legs):** `campaign._flash_errand` walks the billed `_FLASH_BACK_LEGS` EAST —
   Route 10→9→Cerulean→5→6(Underground Path)→Vermilion→Route 11 — every leg MOVED (shift1_flash_errand*.log).
3. ✅ **Diglett's Cave CROSSING:** `_cross_cave` (ported verbatim from the proven recon_hm05) works end-to-end:
   `WARPED (3,29)→(1,38)` into the cave → `WARPED (1,36)→(3,20)` OUT to Route 2 (the aide's map). VERIFIED.
4. ✅ **Freeze-spin guard (rule 18):** DEX-STALL guard surfaces `flash_stuck` after 2 no-gain catch phases
   instead of spinning the catch forever on exhausted grass (was 182 no-gain ticks). [verify in flight]

## THE BINDING BLOCKER (fully diagnosed — this is where the successor starts)
**DEX-10 IS UNREACHABLE ON THE ERRAND'S CATCHABLE GRASS — she caps at OWNED=8.** Proven with 15 balls
injected (`shift1_r11_ballsinject.log`, isolating species-availability from the ball economy): Route 11 grass
gives +2 (dex 5→7), Route 2 grass gives +1 (→8), then NO new species are reachable. The missing 2 live in:
- **Diglett's Cave (Diglett/Dugtrio)** — but `catch_one` only works on GRASS tiles; cave-floor random
  encounters need a NEW primitive (walk the cave floor triggering encounters + commit a ball). NOT built.
- **Off-errand routes** — Route 24/25 (N of Cerulean: Abra/Oddish/Bellsprout/Pidgey), Route 3/4 (Mankey/
  Jigglypuff/Zubat/Sandshrew). She passes Cerulean on the back-legs; a north detour could supply 2-3.

**Ball economy is SECONDARY** (not binding): she boots with ~4 balls + ~$1036; even with 15 injected she
capped at 8, so balls aren't the wall — SPECIES AVAILABILITY is. (Vermilion Mart IS mapped/buyable —
`CITY_MART_DOORS[VERMILION]` — if you do need to top up balls, buy there on the back-legs.)

**Aide gatehouse + TEACH are UNVERIFIED** — she never reached dex 10, so `_flash_gatehouse` + the Flash teach
never fired live (they're ported verbatim from recon_hm05 which DID clear them on the Champion climb, so
low-risk, but not proven in THIS integration).

## THE SUCCESSOR'S CHOICE (two clean paths to dex 10 → then the errand completes)
**RECOMMENDED — systemic (competency #3, helps the WHOLE game):** make her CATCH NEW SPECIES during normal
forward travel so she arrives at the Flash gate already at ~9-10 owned (a real player has 12+ by badge 3;
Kira has 6 because the catch drive is gated `party<4`→nursery-off + no dex drive). Then the errand's dex-step
is already satisfied and it goes straight to the aide. This is the RIGHT fix and removes the retro-catch
scramble entirely.
**OR — tactical (finish the errand as-is):** add a CAVE-FLOOR catch primitive (walk Diglett's Cave triggering
encounters, commit balls for Diglett/Dugtrio) so PHASE 3's `_catch_to_10("Diglett's Cave")` actually lands 2
species → dex 10 → cross to Route 2 → aide → teach. Diglett is also her long-wanted Ground mon (Surge insurance).

**Then:** verify aide-gatehouse-talk + Flash teach live, cross Rock Tunnel (Champion did it — machinery exists),
Route 8 → Celadon → Erika (badge 4). Re-run `recon_longrun.py surge_done.state 25` from the frontier.

## TOOLS (this shift)
- `recon_flash_errand.py [state] [min]` — drives `_flash_errand` in a tight loop (no chooser), banks
  `route11_flash.state` on first Route-11 arrival, `FLASH_INJECT_BALLS=N` env to isolate the ball economy.
- Fixture `states/workshop/route11_flash.state` (+ `.world_model.json`) = badge-3 team AT Route 11, dex 5.
- Logs: `shift1_flash_errand.log` (full back-legs+cave), `shift1_r11_ballsinject.log` (dex-cap proof).

---

# NEXT_SESSION — INTERACTIVE SESSION (2026-07-09, post-shift-16) — READ FIRST

Interactive session with Jonny (not a night-train shift). Landed, each committed + VERIFIED from disk:

1. **AUDIO OUTPUT ISOLATION** (`ce05293`) — the Viridian-parcel PortAudio SIGSEGV can no longer kill the
   emulator: the `sounddevice` OutputStream runs in a CHILD process (`pokemon_agent/audio_child.py`); the
   parent drains mgba PCM (proven safe) + streams over a pipe; a native abort kills only the child, which
   respawns with backoff. Parent paces itself wall-clock so a dead audio child never sprints the game.
   `POKEMON_AUDIO_ISOLATE=0` = legacy in-process. VERIFIED: 288s parcel drive audio-ON, 0 child deaths.
2. **GAME AUDIO ON BY DEFAULT** (`f9417e9`) — `POKEMON_GAME_AUDIO` default 0→1 (go/watch/pokemon_proc).
3. **DENSE AUTO-CHECKPOINTS** (`4acbf5b`,`5cb3124`) — free_roam banks a full sanctity bundle every gain
   seam + ~12min to `states/campaign/checkpoints/<ts>_<place>_<badges>_<playtime>/` (keep 40). List/reload:
   `watch.py --list` (dense section) + `watch.py --at <name>` (canonical-safe sandbox). Firewalled to the
   dev/campaign line. `POKEMON_AUTO_CKPT`.
4. **INTRA-SEGMENT CHECKPOINTS / Fix C** (`83cc39b`) — the SHOW spine banks a rolling `<segckpt>.progress.state`
   after each completed objective; on resume the in-progress segment reloads it + runs only the REMAINING
   objectives. A gym loss retries AT the gym, NOT from the bedroom (the "restart-looks-like-a-crash" pain).
   `POKEMON_SEG_PROGRESS`.
5. **GYM LEADER NAV / Fix A** (`6d4de1d`) — `_los_retrigger` no longer nudges onto the leader's OWN tile
   (Brock (6,5)); travel was treating the standing leader as a plain-NPC blocker and burning its budget
   routing AROUND him. Now excludes the leader tile (computed from `leader_dir`).
6. **ENFORCED PRE-GYM PREP / Fix B** (`fb4fa44`) — beat_gym runs `prep_for_gym` FIRST: `StrategicPlanner.
   gym_readiness(gym, party)` reads the KB (ace/level_band/weak_to/answer_species); if thin / no type
   answer / underleveled → CATCH on nearby grass + GRIND to `ace_level+margin`. On a loss `_bump_gym_prep`
   escalates the target so the retry preps HIGHER (not the same solo team). Suppressed on a goal-pin.
   `POKEMON_GYM_PREP`. LOGIC verified deterministically. **E2E real run (seg_opening→Brock, headless,
   785s): PARTIAL — the decisive half PROVEN, one gap found.**
   - ✅ **LEVEL/TYPE/BEAT-THE-GYM enforcement WORKS AND IS DECISIVE:** prep detected `party 1/3, topL 13/15,
     type_answer=True` → GRINDED Bulbasaur to **L15** (KB ace L14+1) instead of the old WARN-and-walk-in-at-13
     → **BEAT BROCK** (geodude/sandshrew/geodude/onix → BOULDER BADGE). The earlier solo-Charmander run LOST
     at L14; this WON. The direct cause of the Brock loss (underleveled solo) is FIXED.
   - ❌ **CATCH-A-TEAM half fired but caught NOTHING** (`no_reachable_target` ×3) → still solo (party=1).
     ROOT CAUSE: `prep_for_gym` runs at the gym CITY (Pewter has NO grass), so `catch_one` had no wild
     target. **FIX NEEDED (req #1):** the team-catch must run on the GRASSY ROUTES she passes through BEFORE
     the gym (she already grinds on Route 2 grass with 5 balls from the parcel) — e.g. fold an ensure-team
     catch into the pre-gym GRIND phase, or have prep route to the nearest grass route first. Well-scoped
     follow-up; the readiness LOGIC + grind + beat are all proven, only the catch LOCATION is wrong.

**DIAGNOSIS that drove 4-6:** the post-Brock "crash" was NO crash — she LOST to Brock with a SOLO L14
Charmander (party=1), the BEAT_GYM stuck-watchdog cleanly stopped, and the supervisor replayed the whole
segment from Pallet (looked like a crash). Audio isolation was vindicated (247k frames clean). Root cause
= thin-bench/solo-underleveled play (same as the Gary wall) → Fix B; replay-from-bedroom → Fix C; gym nav
thrash → Fix A.

**CORE-KIRA FIREWALL: untouched all session** (audio/harness/gameplay-strategy only; two-bucket firewall intact).

---

# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 16)

## SHIFT 16 HEAD (read FIRST — supersedes everything below)

**THE VERMILION/SURGE "GYM-ENTRY WALL" WAS A LOOK-AHEAD FIDELITY BUG, NOT A GAME WALL.** Shift-15's
refinement ("likely a FALSE Cut gate / door-warp bug") was WRONG. Ground-truth diagnosis (shift 16,
`recon_vermilion_travel.py` — teleport to the S.S. Anne harbor spawn (23,33), run the REAL travel
planner to the gym door approach (14,26)): travel routes the long way around and reaches (19,23) where
it finds **a REAL Cut tree at (19,24)** gating the only path to the Surge gym → returns
`no_route_hm_blocked` / `hm_blocked:cut`. There is NO tree-free route (the gym is walled behind it,
exactly like real FRLG). The tree is DISTANCE-CULLED when she stands far away (at (24,0) `scan_field_objects`
returns empty), which is why shift-15 saw a confusing "genuine wall/zone gap" instead of a clean cut gate.

**ROOT CAUSE = the look-ahead was UNFAITHFUL to live.** Live play loads `.env` which sets
`POKEMON_FIELD_MOVES=1` (+ `POKEMON_ITEM_PICKUP=1`) — so the LIVE run CUTS this tree. But `recon_longrun.py`
never loaded `.env`, so `FIELD_MOVES_ENABLED` (campaign.py:195, `os.getenv("POKEMON_FIELD_MOVES","0")`)
defaulted OFF → `self.field=None` → Cut/Surf/Strength actuation disabled → EVERY cut-tree gate wedged the
look-ahead, a stall the live show would never hit. **FIX (shift 16): recon_longrun now `setdefault`s
POKEMON_FIELD_MOVES=1 + POKEMON_ITEM_PICKUP=1** to mirror live (rule 8 fidelity; shell override still wins).
The gym-entry orchestration ALREADY handles it: beat_gym "stuck" → head_to_gym calls `_gym_gate_probe`
(campaign.py:5768) → walks to the tree object + auto-`clear_obstacle("cut", face)` (line 5804, gated by
FIELD_MOVES_ENABLED). With field moves armed the chain should complete.

**CONFIRMED MECHANISM (shift 16 look-ahead `logs/debug/shift16_surge_fieldmoves.log`, field moves ON):**
the FULL Vermilion chain is CORRECT and now unblocked — reach Vermilion → try gym → cut-tree (19,24) blocks
→ gym-gate-probe arms `hm_obstacle/cut` questline → routes SOUTH to the S.S. Anne → board → beat Gary →
captain gives HM01 → **TEACH BRIDGE** (campaign.py:5143, `step.success==('cap','cut')`) teaches Cut to
ivysaur (PROVEN to fire: shift-15 log line 1344 "cut -> ivysaur ... taught") → return to Vermilion → the
gym-gate-probe's **auto-cut** (campaign.py:5804, gated by `self.field`/FIELD_MOVES) NOW fires (was the ONLY
missing piece — shift-15 taught Cut fine but `self.field=None` blocked actuation) → enter gym → trash-can
puzzle → Surge. NOTE: she may try the gym BEFORE boarding the ship (legit — she doesn't have HM01 yet), so
the get-Cut questline detour is expected, not a bug.

**✅ VERIFIED — BADGE 3 (THUNDER) CLEARED FRESH, end-to-end** (`logs/debug/shift16_surge_focused.log`,
GOAL flag 0x822 in 129s wall): from a Cut-known Vermilion fixture she CUTS the (19,24) gym tree → warps
into the gym → **SOLVES the trash-can switch puzzle** (env_puzzle.TrashCanPuzzle's FIRST-EVER live run:
found switch 1, adjacent switch 2, motorized door opened — the FLAG_TEMP_1=0x001 id is CORRECT) → clears
the juniors (smart NUKE-SLEEP on the Voltorbs) → BEATS Lt. Surge → badge 3, narrated ("YES — we DID it!
Lt. Surge is DOWN... badge number 3"). ivysaur L30+Razor Leaf overpowered Surge — NO Diglett needed.
Sanctity correctly REFUSED to promote the 3-badge bank over the 8-badge Champion (canonical protected).
Fixture builder: inject Cut into a post-Gary team state via `camp._set_lead_moves` (see how vermilion_cut.state
was built in the shift-16 transcript); recon_surge_focused.py drives beat_gym directly.

## NEW FRONTIER — BADGE 4 (RAINBOW, Erika/Celadon)
A badge-4 forward look-ahead is/was running: `recon_longrun.py surge_done.state 20` (boot = the banked
badge-3 Vermilion state copied to workshop/surge_done.state; GOAL flag 0x823 Rainbow). Boots ivysaur L31 +
full team, badges=3, canonical world model loaded (nav pre-solved → GATE blockers surface, not nav noise).
The badge-4 stretch = Vermilion → Cerulean → cut Route-9 tree (she CUTS it, verified this run) → Route 10 →
**ROCK TUNNEL (pitch dark — HARD HM05 FLASH capability-gate; `frlg_gates.json:139`)** → Lavender → Route 8 →
Celadon → Erika (grass gym). She evolved **ivysaur→Venusaur L32** en route. Flash = HM05 from the Route-2
east-gate aide, gated on **≥10 OWNED species** (`frlg_gates.json:43`).

**OBSERVED STALL (shift16_badge4_forward.log): Flash-gated at Route 10, dex 6.** She reaches Route 10 (the
Rock Tunnel mouth), correctly arms the Flash questline, but then STICKS: the questline logs
`QUESTLINE ANCHOR-FIRST: 'HM Flash...' anchors on an UNFAMILIAR area and we're at Route 10 — edge -> (3,4)`
repeatedly and doesn't actually route to the Route-2 gatehouse — so head_to_gym gets no-move-pruned and she
falls back to grinding in place. TWO sub-blockers for the successor: (1) **dex is stuck at 6, needs 10** —
her catching drive isn't reaching the Flash gate's species count (verify she has Poké Balls + a catch-toward-
dex-target drive; the #3 team-building gap); (2) **the Flash questline can't route to the Route-2 aide from
Route 10** — "anchors on an unfamiliar area" even though the canonical world model is loaded (the aide's
gatehouse map/anchor may be unresolved in the questline graph). DIAGNOSE which bites first: if she can't
reach 10 dex she never earns Flash regardless of routing; if routing is broken she can't reach the aide even
at 10 dex. FIX the first, re-run `recon_longrun.py surge_done.state 20` (boots badge-3 Venusaur L32 team at
Vermilion), iterate — same loop that cleared badge 3.

**ROUTE CORRECTION (key insight):** the Flash aide is reached **EAST via Route 11 → Diglett's Cave → Route 2
east gatehouse** (`frlg_gates.json:201-203`: obtain.from="3,29" Route 11, "through Diglett's Cave"), NOT the
north Rock-Tunnel road she took. So the intended badge-4 order is: from Vermilion go EAST to Route 11 →
**Diglett's Cave** (CATCH A DIGLETT here — a new species toward the dex-10 count AND her long-wanted ground
mon) → Route 2 aide → HM05 (once ≥10 dex) → teach Flash → THEN north to Rock Tunnel → Lavender → Celadon →
Erika. She has 4 Poké Balls (not ball-starved), dex 6 — she needs ~4 more species; Diglett's Cave + the
Route-11/Route-2 wilds supply them. The bug: her forward-drive routed her NORTH to the Rock Tunnel mouth
(nearest gym-ward edge) and armed the Flash questline THERE, but the Flash anchor (Route 11/Diglett's Cave,
east) reads "unfamiliar area" and won't route from Route 10 — so she spin-grinds. Likely fix: the Flash
questline should route to its Diglett's-Cave anchor from the EAST (Route 11 off Vermilion), and/or the
catch-to-dex-target drive should dominate over "strengthen-first" grinding when a dex-gated capability
(Flash) blocks the road. Champion crossed Rock Tunnel WITH Flash, so the tunnel machinery exists — the gap
is the fresh-run PATH TO Flash. Fixture `states/workshop/surge_done.state` (gitignored, rebuild from
`G:/temp/longrun/banked_GOAL/` if pruned).

### IF SURGE ITSELF WALLS (team strength, next objective): catch a **Diglett** (Diglett's Cave — west end
off Route 2 near Viridian, or east off Route 11 by Vermilion) = Ground immunity to Electric AND SE on him,
the clean answer; she already narrates wanting one. OR grind the bench (rattata/spearow/pidgey) toward L20.

## SHIFT 15 HEAD (superseded above; kept for the Gary fix reference)

**THE 4-SHIFT S.S. ANNE RIVAL-GARY WALL IS BROKEN (shift 15, committed 6af6410).** Gary was a
PP-famine loss because ivysaur fought with ONE effective attacker. Root cause = `_ensure_move_room`'s
crude move-value model with TWO bugs: (1) it PROTECTED weak non-STAB Tackle (unique-coverage bonus) so
ivysaur DECLINED Razor Leaf at L20 and kept Tackle; (2) it DROPPED a 2nd same-type STAB attacker (Vine
Whip) as low-value once Razor Leaf shared the type. FIX = **STAB-aware `_value`** (a move matching the
mon's type is precious +40; a weak non-STAB filler no longer earns the unique-coverage bonus; `best_slot`
ranks by VALUE not raw power) + a **DAMAGE-POVERTY override** (a ≤1-attacker full set sheds a redundant
status so an incoming attacker auto-learns). Safe: proactive prep only, no battle-turn actuation, no nav.

**VERIFIED both ways:** `recon_movevalue_test.py mtmoon_endgame.state` → drops Tackle + frees a slot (so
Razor Leaf auto-learns at L20), keeps Vine Whip; on the razorleaf fixture → KEEP-ALL (Vine Whip no longer
dropped). END-TO-END (`recon_longrun` from `ss_ticket_razorleaf.state`, a fixture with the fix's moveset)
→ she **BEATS Gary (grudge 5W-2L)**, gets **HM01 Cut** from the captain, reaches Vermilion + engages the
Lt. Surge gym. Log: `logs/debug/shift15_gary_decisive.log`.

### THE NEW FRONTIER — LT. SURGE (badge 3), two sub-walls:
1. **GYM ENTRY (diagnose FIRST — likely NOT a real Cut gate):** she reaches the Vermilion gym at map (3,5)
   but `beat_gym` "couldn't enter" x3 → `GYM-INTERIOR WALL`. The gym-gate-probe then arms an
   `hm_obstacle/cut` gate on a nearby tree — **but this is probably a FALSE gate**: the FRLG Vermilion Gym
   door is directly enterable (no Cut needed AT the door — the puzzle is the two hidden TRASH-CAN switches
   INSIDE that open the electric barrier to Surge). The gate-probe's own comment (campaign.py:5782) warns it
   mis-attributes a failed door-warp to a nearby decorative tree/water. So the REAL issue is likely
   **beat_gym's door-warp not firing** at the Vermilion gym, not a Cut obstacle. ALSO NOTE: recon_longrun
   does NOT set `POKEMON_FIELD_MOVES=1` (default OFF), so even a real Cut/`use_cut` gate can't actuate
   headless — **re-run the Surge stretch with `POKEMON_FIELD_MOVES=1`**. And `clear_obstacle`'s Cut-on-tree
   actuation is RECON-FLAGGED/UNVERIFIED in its docstring (only Surf's water-prompt is source-confirmed).
   DO: grab a frame at the Vermilion gym, check the pret VermilionGym warp/door, confirm whether it's a
   door-warp-fire bug vs a real gate. This is nav/gate (within mandate).
2. **BEAT SURGE:** electric L21-24, + the trash-can switch puzzle inside. She has NO ground answer and a
   frail bench (rattata L14/spearow L15/pidgey L13 vs ivysaur L30). Catch a **Diglett** (Diglett's Cave —
   west end off Route 2, or east off Vermilion Route 11) = Ground immunity to Electric AND super-effective —
   the clean answer. OR level the bench. She already narrates wanting a Diglett.

### HANDED OFF (careful battle work, next):
- **The in-battle DECLINE handler mis-actuates (battle_agent.py:2227).** Its B,A mash to "decline" a
  level-up move can ACCIDENTALLY LEARN the move over a good slot (it dropped Vine Whip this run — she still
  won with Razor Leaf, but the actuation is unreliable). The proper fix is a real **in-battle
  learn-and-replace**: read `GMOVE_TO_LEARN` (0x02024022), identify the leveling mon via `GBATTLER_PARTY_IDX`
  (0x02023BCE), decide via the soul move-value policy, and navigate the forget screen with `hm_teach`'s
  `_forget_cursor`/`_forget_goto` (rows 0-3 = move slots), fail-safe decline on any uncertainty + RAM-verify.
  This covers the BENCH too (bench mons also decline good damaging moves during weak-grind — e.g. "Bite").
- **SANCTITY LEAK (prunable):** recon runs drop `pre_reload_*.state` / `pre_deepwedge_*.state` scratch
  files into the **canonical `states/campaign/` dir** — the deep-wedge escape-hatch bypasses recon_longrun's
  STAGE redirect (it writes to `STATES_CAMPAIGN` directly). The Champion save (`kira_campaign.state`, Jul 8
  20:46) is UNTOUCHED, but the dir is littered. Fix: redirect the escape-hatch/deep-wedge bank path to STAGE
  in a recon run (don't use `POKEMON_CAMPAIGN_DIR` — recon reads canonical sidecars at boot). Low priority.

## KEY FACTS / TOOLS
- **venv python:** `G:/JonnyD/NeuroAI_Bot/.venv/Scripts/python.exe` (NO bare `python`).
- **Look-ahead:** `recon_longrun.py <state> <min>`; env `LONGRUN_BATTLE_LOG=1` for in-battle `[engine]`
  turns. Logs UTF-8. Persistence STAGE-redirected (except the escape-hatch leak above).
- **Party dump:** `recon_partydump.py <state>` — level + moves + PP per slot.
- **Move fixture:** `recon_setmoves.py` writes `ss_ticket_razorleaf.state` (ivysaur [VineWhip,RazorLeaf,
  Poison,Sleep]) for Gary tests. **Decision test:** `recon_movevalue_test.py <state>`.
- **ss_ticket.state** boots at Bill's house → re-walks to the ship → Gary (~2 min). ivysaur's REAL set there
  is [Tackle, Vine Whip, Poison, Sleep] (the weak-Tackle bug case). `ss_ticket_razorleaf.state` = the fixed set.
- **Gary (S.S. Anne rival):** charmeleon+kadabra+pidgeotto+raticate (~L16-20). Beaten with Razor Leaf.
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = 2-PID shim.
- **states/campaign = SHERPA CANONICAL (Champion, untouchable). states/workshop = scratch.** Commit per fix.

## GUARDRAILS
- Shift-15's edit is mode-side team-building policy (proactive move-room prep) — additive, fail-safe, no
  battle-turn/nav change. Core Kira identity/voice/oracle/memory/vision sacred + OFF-LIMITS.
- The in-battle handler fix TOUCHES BATTLE turns — do carefully, verified via look-ahead.
- AUDIO END STATE = ON (`POKEMON_GAME_AUDIO=1`); audio-off is the committed floor only.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS
(a) clean bedroom→credits with audio ON; OR (b) ~80-85% context → clean handoff (rule 11) /
two-consecutive-no-progress brake; OR (c) balance exhausted.

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched — it already rolled credits).
The fresh-run look-ahead now crosses bedroom → Brock → Misty → Cerulean → Bill → Vermilion → S.S. Anne →
**BEATS Gary** → HM01 Cut → **CUTS the Vermilion gym tree → SOLVES the trash-can puzzle → BEATS Lt. Surge
(BADGE 3, Thunder)** [all shift-16-verified], evolves ivysaur→Venusaur, reaches Route 10 (Rock Tunnel mouth).
Frontier = the badge-4 FLASH gate: catch to ≥10 dex (has 6, 4 Poké Balls) + route EAST to the Route-2 aide
via Route 11/Diglett's Cave for HM05 Flash → cross Rock Tunnel → Celadon → Erika. Pop-in (Sherpa) =
`python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
