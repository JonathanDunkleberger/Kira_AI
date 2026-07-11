# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ NIGHT-SHIFT #38 DONE (2026-07-11) — KEEPER ROUTER flipped ON (end-to-end catch PROVEN) + PC/BOX chaff-swap BUILT+VERIFIED. START HERE.
**TWO commits banked, both mode-side, flag-gated, canonical Champion save UNTOUCHED:**

**1. `6cdc463` — KEEPER ROUTER default flipped ON (NS#4 frontier #1 DONE).** The clean fetch+catch that
neither NS#4 fixture could show LANDED: on `bill_house_noabra` (party-3 Ivysaur L26 + Spearow + Rattata, 15
balls, 1 warp hop to Route 25, NO gauntlet — built by `recon_mk_keeper_fixture.py`): PICK fetch_keeper →
FETCH-KEEPER routed Bill's house (30,0)→Route 25 → force-caught the target **Abra** → party 3→4 → plan
advanced to the next keeper (diglett) → Diglett's Cave unreachable → router returned None (fetch_keeper
un-offered) → fell through to head_to_gym, **NO livelock**. `POKEMON_KEEPER_ROUTER` now default "1"
(full-party >=6 and post-game both short-circuit to None → canonical unaffected). Verifier still 10/10.

**2. `ec264a1` — PC/BOX chaff-swap (NS#4 frontier #2 = Tier-1 #15, the FULL-party pairing gap).** The router
only fires at party<6, so a full team of early-catch chaff (erika_done: Venusaur + Rattata/Spearow/Ekans/
Meowth/Pidgey) could never add a coverage keeper. NOW: `box_chaff` deposits the lowest-value OFF-PLAN chaff
at the current city's Center PC → party 6→5 → the router's room-gate opens → keeper added. Built (all in
campaign.py): `deposit_mon(slot,pc_door)` (reuses heal_at_center's proven Center enter/exit + ports
recon_pcbox's screenshot-calibrated menu drive); **`_find_pc_stand()` GENERAL PC locator** (scans the
interior for MB_PC=0x83 + stands below — Centers do NOT share the PC tile, Vermilion (11,1) != Route 10's,
so the hardcoded stand wedged; this reads RAM truth); `_worst_chaff_slot` (lowest-level off-plan non-lead,
via planner._is_target_line — never boxes a keeper/ace); `_chaff_swap_target` gate + `box_chaff` offer/
dispatch (fires only party-FULL + catch_keeper DUE + boxable chaff + mapped-Center city). Flag
`POKEMON_PCBOX` **default OFF**. **VERIFIED:** `recon_deposit_check.py` 11/11 decision cases + a LIVE
headless deposit (surge_done @ Vermilion, party 4→3 by RAM, PC stand auto-located (11,2)); AND an end-to-end
look-ahead (synth party-6 chaff @ Vermilion, both flags on): **PICK box_chaff → deposited Ekans L9 → party
6→5 → fetch_keeper fired → routed to Route 24 for the Abra**. The full chaff→box→router→fetch chain runs live.

### ⇒ NS#38 FRONTIER (exact next actions, priority order):
1. **FLIP `POKEMON_PCBOX` default ON — needs ONE live grab-and-look first.** The deposit menu is
   menu-nav-on-the-long-core (wedge-prone). The actuation is headless-VERIFIED (party 6→5, real menu, on
   Vermilion), but per the wedge-prone class it wants a live eye on the SHOW build (audio/render path) before
   default-ON. Also confirm the Center-detour doesn't over-backtrack (watchability). If a live deposit is
   clean → set `POKEMON_PCBOX` default "1" (one char, campaign.py:~123) + commit. To re-verify headless
   anytime: `POKEMON_PCBOX=1 ../.venv/Scripts/python.exe recon_deposit_check.py` (11/11 + deposit 4→3).
2. **WITHDRAW + auto-boxed-keeper swap-in (the PC/BOX second slice).** Right now box_chaff makes room BEFORE
   the catch. FRLG also auto-boxes a caught mon at party-6 — so a keeper caught while full sits in the box.
   Build `withdraw_mon(box_slot, pc_door)` (reverse of deposit_mon — same _find_pc_stand + the WITHDRAW menu
   branch) + a "swap at Center" hook that withdraws a boxed keeper and deposits a chaff during a heal visit
   (no extra routing). Model the menu drive on deposit_mon; verify headless via a recon_withdraw_check.
3. **FINAL-PROOF GATE (the whole point) — now runnable end-to-end.** A fresh mid-game fixture forward with
   `POKEMON_KEEPER_ROUTER=1 POKEMON_PCBOX=1`: she catches coverage keepers (router), boxes chaff when full
   (box_chaff), levels the bench on the road (road-bench-XP), preps to milestones (milestone-prep), and
   arrives E4-ready with a real 6. Use a mid-game state WITH a world_model sidecar (surge_done/erika_done —
   NOT og_postopening, which is nav-blind). Read the blocker chain; the remaining gap is likely grind-spot
   adequacy for the L45→55 E4 push (cave step-encounter grind, unbuilt) or a nav wedge on the return legs.
4. **prep bite cadence** (NS#4 frontier #3) — the +6 milestone-prep bite levels the bench slowly; a bigger
   bite when FAR under milestone arrives near-milestone faster without a grind-wall.

⚠️ NOTE on the keeper-router live catch at a solo/thin team: reaching Route 24 crosses Nugget Bridge — a
solo/very-thin team loss-loops there (NS#4 misty_done finding; the loss-guard retires the target cleanly, no
livelock, but she won't complete the catch until the team can survive the bridge). box_chaff/road-bench-XP
building a real bench is what fixes this; not a router defect.

---

## ✅ NIGHT-SHIFT #4 DONE (2026-07-11) — road-bench-XP re-validated (party-6) + CROSS-MAP KEEPER ROUTER built (NEW#2). START HERE.
**BANKED (commit 208edb5, mode-side, flag-gated `POKEMON_KEEPER_ROUTER` DEFAULT OFF, canonical untouched):**
the cross-map keeper router — the last unbuilt Part-C piece for team COMPOSITION. Full diagnosis + file:lines
in `TEAM_DEPTH_ROOT_FIX.md` §NS#4. Two look-ahead findings drove it: (1) **road-bench-XP (NEW#1) VALIDATED in
the party-6 mid-game** on `erika_done` (Venusaur L43 + frozen L9-15 chaff bench → Ekans leveled L9→**L14**,
Rattata/Spearow 15→16 as she marched, GRIND SWITCH firing, questline chained right); (2) **the confirmed gap =
team COMPOSITION** — the planner emits `catch_keeper: abra→alakazam` the whole run but she marches past
(on-map un-gate only grabs a keeper she's STANDING on). The router (`_keeper_route_target` /
`_fetch_keeper_errand` / `_place_to_map_index` in campaign.py; `fetch_keeper` action) offers a BOUNDED detour to
a nearby reachable hosting map, then the on-map machinery catches. **Decision-VERIFIED 10/10**
(`recon_keeper_router_check.py`). **Behavioral: a LIVELOCK was caught + FIXED** (offer used world.route but the
errand used naive trav.travel → no_path MACRO-RED spin; fixed with offer⟺executable `_next_step_rideable` gate +
`_travel_to_known` routing + a stall-guard that retires un-rideable targets to `_keeper_unreach`). Post-fix
route3_caught: routes Route3→Route4, retires the Mt-Moon-gated Route24/25 cleanly, resumes leveling — NO livelock.

### ⇒ NS#4 FRONTIER (exact next actions, in priority order):
1. **FINISH the router behavioral proof (a SUCCESSFUL fetch+catch), then FLIP default ON.** Neither test
   fixture could show it because BOTH gate a thin team from the keeper: route3_caught is Mt-Moon-level-gated
   before Route 24; misty_done (solo Ivysaur L22) **loss-loops at the Nugget Bridge gauntlet** en route to
   Route 24 (Abra) — `keeper_route:travel:battle_loss` → blackout → Center → retry (head_to_gym/the S.S.Ticket
   questline, also north past Nugget Bridge, would loop the SAME way — it's the solo-team problem, not a router
   defect). A late fix (committed) makes a `battle_loss` leg count as NON-progress so K losses RETIRE the keeper
   (was: blackout relocation reset the stall guard → soft loss-loop). **TO PROVE THE CATCH:** use a fixture with
   a party of 2-3 (not solo) whose keeper map is reachable with NO gauntlet — e.g. a post-Nugget-Bridge state at
   Cerulean/Route 25 with room, or seed a world_model so Route 24 is rideable and the team can survive the bridge.
   Command: `POKEMON_KEEPER_ROUTER=1 LONGRUN_BATTLE_LOG=1 ../.venv/Scripts/python.exe -u recon_longrun.py <fixture>.state 12`;
   grep `FETCH-KEEPER|caught a new|CATCH: [0-9]+ reachable|UNREACHABLE`. On a clean catch → set
   `POKEMON_KEEPER_ROUTER` default "1" (one char, campaign.py) + commit. ⚠️ Detour watchability (over-backtrack?)
   is a LIVE-EYES item — tune `POKEMON_KEEPER_ROUTER_MAX_HOPS` down (4?) / STALL_CAP 3→2 if detours read too far.
2. **PC/BOX (Tier-1 #15)** — the pairing gap for the FULL-party case (erika_done: 6 chaff, router won't fire
   without room). recon_pcbox.py deposit flow proven; generalize `deposit_mon`/`withdraw_mon` + hook `catch_one`
   to box the lowest-value chaff on a full-party keeper catch → then the router adds the keeper.
3. **prep bite cadence** — the +6 bite levels the bench slowly (~5 levels/9min); a bigger bite when FAR under
   milestone would arrive near-milestone faster without a grind-wall.
4. **FINAL-PROOF gate** — a fresh mid-game fixture forward with `POKEMON_KEEPER_ROUTER=1` → she catches
   coverage keepers + levels them + arrives E4-ready. (og_postopening is an INVALID fixture — no world_model
   sidecar → empty-graph nav-blind livelock on the unbilled early gyms; use a mid-game state with a sidecar.)

## ✅ NIGHT-SHIFT #3 IN FLIGHT (2026-07-11) — frontier NEW#1 (ORGANIC BENCH XP ON THE ROAD) BUILT + decision-verified.
**NS#3 BANKED (commit a998378, mode-side, flag-gated, canonical untouched):** the exact NEW#1 fix below —
the mid-game "bench never levels" root. Two helpers in `campaign.py` (`_road_bench_xp_arm` /
`_road_bench_xp_disarm`) wired into `free_roam` around the `_route_action` dispatch: on a forward-march pick
(`head_to_gym`/`travel:`) with a bench member under its `_prep_team_target` milestone, the weakest levelable
under-target mon leads (→ XP-eligible) and `battle_agent.PROTECT_LEAD_GRIND` is armed so the PROVEN
participation switch (battle_agent.py:2554) fields the ace turn 1 — the weak mon banks a share of XP without
taking a hit. The ace is restored to slot 0 the instant the leg ends (weak lead never outlives the march).
Guard: an ACE-HP floor (`POKEMON_ROAD_XP_ACE_HP_FLOOR`=0.6) so a bench-XP leg never STARTS with a dinged ace
(commit 838e9fb). Flag `POKEMON_ROAD_BENCH_XP` (default ON; one-line revert). **VERIFIED 15/15**
(`recon_road_bench_xp_check.py`).

### ✅ END-TO-END VERIFIED (bucket a) — the owed proof LANDED this shift:
A `snorlax_done` look-ahead (`LONGRUN_BATTLE_LOG=1`, Routes 13/16 → Fuchsia) showed **5 arms → 3 LIVE
participation switches in road trainer battles → the bench leveled organically** (Spearow L15→16, Ekans
L9→10) while marching, no strand from the switch. **GOTCHA banked:** recon_longrun suppresses battle_agent's
log unless `LONGRUN_BATTLE_LOG=1` (that masked the switch on earlier passes; city-boots surge/erika/bill were
absorbed by town-nav or had spent trainers). **CAVEAT (hand-bank artifact):** the run then whited out on the
Route 13 gauntlet — but the CONTROL (flag OFF) hit the SAME critical-HP + PP-famine wall, so it's pre-existing
top-heavy-bank attrition (L48 ace soloing an L9-15 bench), NOT this fix. The ace-HP guard mitigates (defers a
dinged-ace leg to heal); on a fresh organically-built run the bench shares the gauntlet so it doesn't arise.

### ⇒ NS#3 FRONTIER (the fix is verified; next pieces):
1. **PC/BOX (Tier-1 #15) — the pairing gap for the catch half; TOP next build.** RECON DONE this shift:
   `recon_pcbox.py` already has a COMPLETE deposit drive (enter Center → walk to PC console → A → BILL'S PC →
   DEPOSIT → pick slot → confirm → verify party count 6→5 by RAM → banks `banked_PCBOX`), menu sequence
   worked out with screenshot calibration. GAPS to wire it into campaign: (a) `PC_STAND` is Route-10-specific
   → generalize the PC-console approach per Center (find the PC tile in any Center); (b) add `withdraw_mon`
   (the reverse); (c) NOT wired into campaign at all (no `deposit_mon`/`withdraw_mon` methods) → add them +
   hook `catch_one` to box the lowest-value off-plan fodder on a full-party keeper catch (Tier-1 #15). ⚠️ The
   PC menu actuation is WEDGE-PRONE (menu-nav-on-long-core) → verify live with grab-and-look, flag-gate.
   Un-gate builds toward 6; box turns it into real depth. NB recon_pcbox reads canonical (now the Champion
   save) on a RAM copy — re-point it at a party-6 workshop bank to re-verify the drive before promoting.
2. **CROSS-MAP KEEPER ROUTER** — the catch un-gate is on-map only; route to `act["where"]` via
   frlg_encounters for grass/cave keepers (Abra/Diglett/Growlithe). Note NS#2's sibling gap: the Abra
   plan-catch FIRED on-map but didn't PERSIST after a mid-catch heal (catch-persistence).
3. **grind-spot adequacy** (#1 old) — largely dissolved by NEW#1 if the owed proof lands (bench arrives
   near-milestone organically), but keep on the radar for a L45→55 E4-prep push (cave step-encounter grind).
4. **THE FINAL-PROOF GATE** — fresh `og_postopening` → 15x look-ahead → she catches keepers, LEVELS the
   bench (now via NEW#1 on the road), arrives E4-ready with a real 6, sweeps tactically to credits.

## ✅ NIGHT-SHIFT #2 DONE (2026-07-11) — frontier #2 (MID-GAME MILESTONE LEVELING) BUILT + decision-verified.
**NS#2 BANKED (commit fb26e6e, mode-side, canonical untouched):** the **mid-game milestone-cap bench-prep** —
frontier #2 below, the exact reviewed edit. `_prep_team_target`'s wall-less proactive bench-raise now caps the pin at
the team-plan's NEXT gym milestone (Brock 14 … Giovanni 52) not the ace-relative `lead-8`, and RE-ARMS on a milestone
RISE (new gym earned) as well as roster change — so the bench climbs toward each gym's level across the whole game in
bounded +6 bites instead of getting ONE bump then abandoned. **Decision logic VERIFIED 10/10** in
`recon_milestone_prep_check.py` (cap / one-bite-per-milestone / badge-re-arm / no-over-grind past milestone /
fallback treadmill-safe). ⚠️ A real trap was caught + fixed during the build: the lead-8 FALLBACK bar drifts up with
the ace, so the milestone-RISE re-arm is gated to REAL static milestones only (`_ms` guard) — else the fallback would
reinstate the ship-run-5 treadmill (verifier case 6b). **STILL OWED (the spec's ship-gate):** a behavioural look-ahead
confirming no live parking/treadmill — launched from `ss_ticket.state` (badge 2: Ivysaur L28 + Rattata/Spearow L14
bench, the ideal mid-gym probe); read `G:/temp/longrun/ns2_milestone.log` for whether the bench-raise fires, bites,
retires, and does NOT park the road. If it parks/treadmills → tune the bite cadence (the fix is defaults-safe: reverts
to prior lead-8 when planner off, so a revert is one-line if needed).

### ⇒ NS#2 FRONTIER (re-ranked by a LIVE run — #2 BUILT; a deeper root now pinpointed as the TOP next action):
The ss_ticket behavioural run (`G:/temp/longrun/ns2_milestone.log`) proved the milestone TARGET is correct + stable, but
surfaced the REAL binding constraint: **the mid-game bench never LEVELS** — she won ~8 road trainer fights Route24→
Vermilion and ALL XP went to the lead (Venusaur L28→32) while the bench stayed FROZEN L14/L14. Full diagnosis +
file:lines in `TEAM_DEPTH_ROOT_FIX.md` (NS#2 UPDATE section). Re-ranked:

**NEW #1 (TOP — organic bench XP on the road).** The participation-XP switch (`battle_agent.py:2554`) is gated to
`PROTECT_LEAD_GRIND` (dedicated `grind_weak_members` only, `battle_agent.py:141`), which push-when-carrying keeps her
out of mid-game → the bench banks nothing from road trainer wins. BUILD: field/register the under-milestone weak mon in
ORDINARY road trainer battles so it banks participation XP (extend `battle_agent.py:2550-2580` beyond PROTECT_LEAD_GRIND,
guarded per the 2584 STRAND-ROOT note so the weak mon takes no hit / never strands; drive it off a "bench under
`_prep_team_target`" signal). This levels the bench organically WHILE traveling (watchable — no grind-wall) → she reaches
each gym + badge-8 `_prep_e4_target` near-milestone, which ALSO dissolves old-#1 grind-spot-adequacy. ⚠️ Touches the
in-battle switch (Tier-1 #5, wedge-prone) — flag-gate it + verify on a PAST-Cut-gate bench bank (ss_ticket WEDGES at the
Vermilion Cut tree, so use surge_done/erika_done or a fresh run that clears the gate; do NOT verify from ss_ticket).

**Then the prior pieces (unchanged, still valid):** PC/BOX (#3), cross-map keeper router (#4 — note the run showed the
Abra plan-catch FIRED on-map but didn't PERSIST after a mid-catch heal; catch-persistence is a sibling gap), grind-spot
adequacy (#1, largely dissolved by NEW #1 if it lands).

## ✅ NIGHT-SHIFT #1 DONE (2026-07-11) — team-depth Part-C: 3 fixes banked + verified.
**BANKED + VERIFIED NS#1 (3 commits, all mode-side, canonical untouched):**
- **2dc74d5 `prep_for_e4()`** — at all 8 badges the WHOLE party is floored to the team-plan's E4 milestone (~L55),
  not the ace-relative `lead-8`. Direct fix for the NS9-14 top-heavy wall (ace solos, bench sags 8 under → swept
  in the Center-less gauntlet). NOTHING read `level_milestones` before. **BEHAVIOURALLY VERIFIED** on a giovanni_kit_g
  look-ahead: fires (prep=55), fields the REAL team (Lapras L37 / Kadabra L39 — not the chaff), and retires cleanly
  ("pushing on with the strong core") when grass can't level them — **NO livelock** in the live loop. Logic verified
  4/4 in `recon_prep_e4_check.py` (emit / retire-on-crossed / chaff-only / stalled).
- **22398ec catch un-gate + chaff-floor** — `_plan_wants_prebuild` no longer hard-blocks at pc>2: she keeps building
  toward a full 6 while a planned keeper is DUE AND catchable on the CURRENT map (junk-safe: catch_one flees
  non-targets). Verified 2/2 in `recon_ungate_check.py` (control + forced-positive). And `grind_weak_members` gained
  a `min_level` floor so E4-prep fields only the levelable real team, never drags L8-14 box-fodder toward 55.
- **9c8a33d** the un-gate verifier.

**FRONTIER — the exact next pieces, in priority order (with the solutions I worked out):**
1. **GRIND-SPOT ADEQUACY (the blocker prep_for_e4 surfaced, confirmed LIVE).** On giovanni_kit_g the bench (L37/39)
   can't reach 55 because near Viridian/Indigo the reachable grass (Route 22 L2-8, Route 2, Route 18 caps ~L44-46)
   gives L37+ mons ~0 XP → both stall → she proceeds UNDERLEVELED. This is the KNOWN NS14 gap (VR cave-grinding
   unbuilt; E4-self-grind futile). **NOTE:** this is largely an ARTIFACT of the hand-built bank whose bench was
   caught late + never levelled — a FRESH run with piece 2 below never sags this far, arriving needing only a small
   top-up. Real fix candidates: cave step-encounter grinding (unbuilt), or a high-level grind spot table in gamedata.
2. **MID-GAME MILESTONE LEVELING (highest-leverage; designed, deliberately NOT shipped blind — see the exact edit).**
   ROOT (found this shift, campaign.py `_prep_team_target` proactive block ~5289-5318): the bench target caps at
   `lead-8` in `+6` bites AND the RE-ARM GUARD retires after ONE bite per roster-signature (`_bench_done_sig`). So
   the bench gets ONE +6 bump then is abandoned as the ace runs away — the exact "arrives thin" mechanism. **THE
   PRECISE EDIT (minimal, rides the proven +6 pacing — low-risk):** (a) cap the pin at the MILESTONE not the ace:
   `self._bench_pin = min(milestone, floor + 6)` where `milestone = self.team_planner._next_milestone(badge_count,
   post_game)[1]` (falls back to `lead - 8` if no milestone); (b) RE-ARM the retired prep when the MILESTONE RISES
   (a new gym earned), not only on roster change: track `_bench_done_milestone`, re-arm if `sig changed OR milestone
   > _bench_done_milestone` — milestones change only on badge-earn (infrequent) so this can't treadmill. Keep the
   band+stalled livelock guard. This makes the bench climb toward each gym's level over the game in bounded, watchable
   bites. **DO NOT ship without a FRESH `og_postopening.state` multi-gym look-ahead** confirming no treadmill/over-grind
   (that's WHY it's held — the RE-ARM/pin machinery is hard-won: ship-run-2/5, celadon_run1). The badge-8 `prep_for_e4`
   already handles the E4 case (committed); this is its mid-game sibling. (I built a `_prep_milestone_target` helper
   generalization + reverted it — the pin-machinery edit above is the safer path, reuses proven pacing.)
3. **PC/BOX (Tier-1 #15) — the pairing gap for the CATCH half.** The un-gate builds toward 6, but the giovanni run
   showed the endgame reality: at party-6 with 3 chaff, she CAN'T swap fodder for planned keepers without box-mgmt.
   Build `deposit_mon`/`withdraw_mon` (promote `recon_pcbox.py`'s deposit flow) + hook `catch_one` to box the
   lowest-value off-plan fodder on a full-party keeper catch. This is what turns the un-gate into real depth.
4. **CROSS-MAP KEEPER ROUTER** — the un-gate is on-map only; a fresh run only catches keepers it happens to pass.
   Route to `act["where"]` (resolve via `_PLACE_NAMES` reverse + frlg_encounters) for grass/cave keepers (Abra,
   Diglett, Growlithe); Snorlax/Lapras are gift/static (quest-gated, leave to the questline). Model on `_flash_errand`.
Full diagnosis + file:lines: `pokemon_agent/TEAM_DEPTH_ROOT_FIX.md`. Loop NOT stopping — this is the build phase.

## 🚂 PASS 3 — THE NIGHT-TRAIN MISSION (2026-07-11, START HERE EVERY SHIFT): make a FRESH GO build its own team + play a watchable ~25-35hr bedroom→credits spectacle. **CONSOLIDATION, not discovery — the game is already solved; wire it together so it FLOWS.**

**PART-1 RECON IS DONE — READ `pokemon_agent/PASS3_RECON.md` FIRST** (the complete wired/unwired/needs-live-watch gap
map for every item below, with file:line fix hooks + honest buckets). Don't re-recon what it already answers.

**HOW TO RUN THIS:** unattended multi-shift night train (`night_shift.ps1`). Each fresh shift: (1) READ this
block + `pokemon_agent/PASS3_RECON.md` + `TEAM_DEPTH_ROOT_FIX.md` + `E4_TACTICAL_FRONTIER.md`; (2) continue the mission from where
the last shift banked; (3) commit-per-fix, VERIFIED from disk; (4) UPDATE this block with progress before you
run low on context; (5) never re-solve a wall pass 1/2 already killed — if a fresh run bumps a solved wall,
that's a WIRING failure (connect the existing solution, don't rediscover it).

**THE STATE:** the tactical E4 half is DONE (credits rolled, hand-grind team, commit 23487e7). The remaining
mountain = the AUTONOMOUS build: a fresh GO must catch + level its OWN 6 and arrive E4-ready, then the
committed tactical fixes win. Canonical 2026-07-07 Champion save UNTOUCHED; all work on scratch banks.

### THE BUILD (tractable core first, then the spectacle layer; check what EXISTS before building — most is built):
1. **TEAM-DEPTH Part-C executor (THE HEADLINE — full diagnosis + file:line targets in `TEAM_DEPTH_ROOT_FIX.md`):**
   wire `catch_keeper` to route+catch a FULL 6 across the game (un-gate `_plan_wants_prebuild` pc>2 @campaign.py:4355;
   `_plan_keeper_target` must TRAVEL to `act["where"]` via frlg_encounters.json, not just current-map); wire
   `grind_to`/`develop_bench` → a real bench-leveling objective + FIX `grind_weak_members` ace-only fallback
   (campaign.py:5269-5273) so the BENCH levels; raise party target toward 6 + add `prep_for_e4()` (level whole
   party FLOOR to `level_milestones["E4"]`=55 before Indigo); add box/replace-fodder logic (Tier-1 #15).
2. **MOVE/TM/HM intelligence:** teach right move→right mon; NEVER clobber a signature/super-effective/only-damaging
   move; decline bad level-up learns; assign HMs (Cut/Surf/Strength/Flash/Fly) without ruining movesets; respect
   the 4-slot constraint. (Much exists — `_ensure_move_room`/STAB-aware `_value`/hm_teach.py — audit + harden.)
3. **Lapras-lead-and-heal tactical AI** (battle_agent): when the ace can't hurt the foe and a specialist is the
   type-answer, LEAD/keep the specialist in + funnel healing to IT, not the ace's 1x Cut.
4. **SOUL flow (via plan_note/voice seam ONLY — core persona + firewall OFF-LIMITS):** team-as-FAMILY (names,
   bonds, favorites, feeling on catch/evolve/faint/loss); BOSS ENERGY (shit-talk rivals/leaders, fired-up/
   triumphant, running bits, stew on losses); POSTGAME/ARC awareness (knows where she is, how far she's come).
5. **NATURAL EFFICIENT PLAY + PC management:** sensible routing, deposit/withdraw, buy items, grind with purpose,
   no head-bumping. **WATCHABILITY PACING:** win most fights, brief struggles, keep moving; LOG watchability signals.
6. **LIVE-SHOW STABILITY HARDENING (critical — 2 past streams crashed ~15min in):** audit/harden run.py/voice/stream
   path (audio isolation/default-off, supervisor auto-restart, checkpoint-resume). ⚠️ multi-hour stability can ONLY
   be confirmed by Jonny's SUPERVISED WATCH — build the hardening, NEVER claim it's proven headless.
7. **INSTRUMENTATION (make runs legible):** (a) END-OF-RUN METRICS report — playtime h:m, final 6 (species+levels+
   movesets), badge timeline, #battles/losses/faints, longest grind, per-segment watchable/grindy verdict; (b)
   SAVE-VISIBILITY/HOP-IN — Jonny can list+load+watch any banked checkpoint (confirm the command; pass-2 E4 hop =
   `E4_BOOT=G:/temp/longrun/banked_E4 ../.venv/Scripts/python.exe -u recon_e4.py`); (c) NAV AUDIT — one more full-game
   wedge-risk pass (Indigo→VR gap etc. per E4_TACTICAL_FRONTIER.md); (d) "WHAT HAVEN'T WE THOUGHT OF" pass — surface
   arc-wide gaps that could break/drag/disappoint on a fresh 30hr run BEFORE they hit live.

### THE FINAL-PROOF GATE (the whole point): fresh early state (`states/workshop/og_postopening.state`) → headless
15x look-ahead → she AUTONOMOUSLY catches keepers at their locations, LEVELS her bench to milestones, arrives at
the E4 with a REAL leveled 6 + coverage, and sweeps TACTICALLY to credits — no hand-port/struct-grind (that manual
workaround is what PROVES the builder is broken), no solo-hack, no over-level steamroll. Watchable pace, narrated.

### THE HONEST-REPORTING MANDATE (non-negotiable — overconfidence caused 2 failed streams): for EVERY item above,
label it (a) BUILT + headless-verified, (b) BUILT but needs Jonny's SUPERVISED LIVE WATCH to confirm it's actually
spectacle-grade, or (c) not-yet-built. The soul/narration/pacing/live-stability items are EXPECTED to be bucket (b)
— headless CANNOT prove "watchable/soulful/stable". NEVER tell Jonny the 30hr spectacle is "done"; tell him what's
built, what's headless-proven, and what his own eyes must verify. That honest line is what makes the stream succeed.

---

## 🏆 CREDITS ROLLED (night-shift #1, 2026-07-11) — the E4 TACTICAL half is DONE + VERIFIED e2e.
The leveled Sherpa team (Venusaur L90 / Lapras L72 / Kadabra L58) beat the Elite Four + Champion Gary
TACTICALLY and reached the HALL OF FAME (`G:/temp/longrun/banked_CREDITS`, fresh). Two committed
battle-brain fixes cracked the 12-run whiteout wall (commit 23487e7, `battle_agent.py`, mode-side only):
(1) never sleep-lock a foe we're 2x super-effective on (the `_se_chunk_latch` mis-slept Cloyster, burning
rooms-1-4 heals); (2) when the active is type-disadvantaged AND a super-effective reserve exists, field the
specialist regardless of level lead (Lapras Ice Beam 2x vs the Champion's Pidgeot instead of Venusaur
trading itself on Cut x1) — which also cut rooms-1-4 heal spend so Full Restores survived to Gary.
**Re-verify anytime:** `E4_BOOT=G:/temp/longrun/banked_E4 ../.venv/Scripts/python.exe -u recon_e4.py`
(banked_E4 is a clean whiteout-center strong-team bank → one clean lap → Hall of Fame).

**THIS IS NOT the fresh-GO watchable re-do.** The team that won was hand-grind-built by NS7-14. The REAL
remaining mountain = the autonomous Part-C team-builder in **`pokemon_agent/TEAM_DEPTH_ROOT_FIX.md`** (below):
make a fresh GO build its own leveled 6 and arrive E4-ready. The tactical E4 half is now guaranteed by the
two fixes above — so the final-proof gate just needs the BUILD half. The canonical 2026-07-07 Champion
timeline was NOT touched (this ran on scratch banks). Loop is STOPPED (CREDITS is line 1 of NIGHT_REPORT.md).

---


## 🎯 THE MISSION (2026-07-10 night — START HERE): FIX THE ROOT BUG — she arrives at the E4 with only ~2 usable mons instead of a real leveled 6. READ `pokemon_agent/TEAM_DEPTH_ROOT_FIX.md` FIRST — it holds the full evidence-backed diagnosis + the ranked fix with exact function/file:line targets.

**WHY:** the headline goal is a WATCHABLE autonomous run where, on a fresh GO, she shows up to the Elite Four
with a REAL FULL LEVELED TEAM like a competent trainer. The game is ALREADY beaten canonically (2026-07-07
credits save = the real summit); this is the watchable re-do. Arriving thin is THE bug that makes the showcase
fall apart. The whole tactical E4 chain is already fixed + committed this shift (moves/Ice Beam/Psychic, 4x
switch, type-answer revive, FR-first shop — commits 625b568, 0cb736d, 6772b54); with a real 6 those fixes win.

**ROOT CAUSE (proven):** the TeamPlanner brain plans the right team but its Part-C EXECUTOR is unwired —
`grind_to`/`develop_bench` have ZERO consumers (voice-only), the catch hook self-disables at 3 mons and never
routes to catch locations, real leveling only raises the ACE (party target 3, `grind()` reads the lead only,
the bench-XP switch wedges), and there is NO `prep_for_e4`. KB data (Part A) + `assess()` (Part B) are
VERIFIED-good — the break is pure Part-C wiring in `campaign.py`.

**THE FIX (implement — details/lines in TEAM_DEPTH_ROOT_FIX.md; all wiring, no data work):**
1. Wire `catch_keeper` to route + catch a FULL 6 across the game (un-gate `_plan_wants_prebuild` `pc>2` at
   campaign.py:4355; make `_plan_keeper_target` travel to `act["where"]` via frlg_encounters.json, not just current-map).
2. Wire `grind_to`/`develop_bench` → a real bench-leveling objective; FIX `grind_weak_members`' ace-only
   fallback (campaign.py:5269-5273) so the BENCH levels, not just Venusaur.
3. Raise the party target toward 6 for late gyms + add `prep_for_e4()` reading `level_milestones["E4"]` (55) —
   level the whole party FLOOR (not the ace) before entering Indigo.
4. Add box/replace-fodder logic (Tier-1 #15) so caught keepers replace the L9-14 dead-weight fodder.
5. ALSO fold in the **Lapras-lead-and-heal tactical AI** change (battle_agent): when the ace can't hurt the foe
   and a specialist is the type-answer, LEAD/keep the specialist in and funnel healing to IT, not the ace's 1x Cut.

**THEN — THE FINAL-PROOF GATE (the whole point):** fresh early-game state (e.g.
`states/workshop/og_postopening.state`) → headless 15x look-ahead → confirm from the log + party growth that she
PROACTIVELY catches keepers at their locations, LEVELS the bench to milestones, arrives at the E4 with a REAL
leveled 6 + coverage, and beats the E4 TACTICALLY (won on a genuine team — no solo-hack, no over-level steamroll).
Bounded detours, watchable pace, narrated. Canonical UNTOUCHED. Do NOT hand-port/struct-grind the team — that
manual workaround (recon_grind_bench + recon_port_and_fix) is exactly what proves the autonomous builder is broken.

---

## ✅ NS14 (2026-07-10): OFFENSIVE-UPGRADE SWITCH FIX BREAKS THE LANCE WALL — reached the CHAMPION (room 5) for the first time. Committed (ce5e391). Route 18 bench-grind + auto-delivery RUNNING overnight; remaining wall = bench too FRAIL + Gary's Charizard (a LEVEL problem).
**AT WAKE — CHECK `ns14_deliver_status.txt` FIRST (the auto-delivery may have rolled credits):**
`ns14_deliver.sh` is armed: it WAITS for the Route 18 grind to finish gracefully (writes "DONE." → `banked_GRIND`
final bank is VENUSAUR-LED via `_restore_ace`), then runs `tail_driver.sh` (seafoam→mansion→blaine→giovanni→
victory→e4). It verifies slot0=Venusaur before the tail (the exact NS13 delivery bug: NS13 fired on a mid-pass
weak-led bank → VR lost). It clears the STALE 2026-07-07 banked_CREDITS first so a fresh one = real credits.
- `cat G:/temp/longrun/ns14_deliver_status.txt` — if it says "CREDITS!! banked_CREDITS is FRESH" → **WRITE
  `CREDITS` as LINE 1 of NIGHT_REPORT.md** (stops the loop) + full mountain survey. Confirm banked_CREDITS mtime
  is TODAY. Then promote per the two-timeline law.
- If it says "tail_driver rc=N — NO credits": read `tail_status.txt` for the failed leg. Most likely the VR leg
  (if Venusaur didn't stay slot0 through the seafoam→giovanni legs) or the E4 (if the bench under-leveled). The
  leveled `banked_GRIND` is preserved — promote it, verify Venusaur slot0, and resume the failed leg (NS9 block).
- If deliver is still "armed — waiting" → the grind hasn't finished; check the grind (below) and either wait or
  read `banked_GRIND` levels and run the tail manually once Kadabra+Lapras are ~L46+.

**THE ROUTE 18 BENCH GRIND (`ns14_grind.log`, banks `banked_GRIND` ~150s):**
`recon_grind_bench` is RUNNING from `grind_base_g` (Route 18, map 3,36): `GRIND_SPECIES=64,131` (Kadabra FIRST —
the Agatha specialist — then Lapras, the Gary/Charizard answer), `GRIND_TARGET=48`. Start levels Venusaur L65 /
Lapras L37 / Kadabra L39. Participation-XP switch banks XP on the WEAK mon while Venusaur aces the kills. Check
`banked_GRIND` levels: `../.venv/Scripts/python.exe recon_partydump.py G:/temp/longrun/banked_GRIND/kira_campaign.state`.
- **Route 18 wilds are L23-29 → XP to L40+ mons is SLOW; it may only reach ~L44-46 overnight** (banks forward
  regardless). Whatever it reaches, THEN DELIVER TO THE E4 (below) and re-run recon_e4 — the switch fix is done.
- ⚠️ **The E4-self-grind loop idea was TRIED AND KILLED — it's FUTILE.** In the E4 whiteout-loop, Venusaur hogs
  every KO (→ L71→73) while the frail bench faints before landing kills, so Kadabra/Lapras DON'T level. And an
  over-levelled Venusaur still can't beat Gary's Charizard (Grass 0.25x hard type-wall). Only the participation
  grind levels the bench. (Loop scripts `ns14_e4_loop*.sh` are dead; ignore.)

### ▶ DELIVERY — how the leveled Route-18 bench reaches the E4 (the last mile, mostly-proven legs):
The grind team is pre-VR (badge 6). Run the NS9 tail: seafoam→mansion→blaine→giovanni→victory→e4 (per-leg cmds in
the NS9 block far below; all legs banked OK in NS13 EXCEPT `recon_victory`). **The VR leg failed because LAPRAS
led the Water-Cooltrainer fight (fight#104) and Body-Slammed x1 too slowly.** THE FIX: the grind ends by calling
`_restore_ace()` which moves the highest-level mon (Venusaur) to slot 0 — so the banked grind SHOULD already be
Venusaur-led → Razor Leaf 2x sweeps the Water Cooltrainer → VR clears. **VERIFY Venusaur is slot 0 in banked_GRIND
before the tail** (`recon_partydump`); if not, swap it (a `camp._swap_party_slots(0, <venusaur_slot>)` primitive
exists — or add a one-shot reorder to the tail before recon_victory). Then E4 with the switch fix → the leveled
Kadabra survives Agatha, the leveled Lapras Surfs Gary's Charizard → CREDITS.
- If it reaches the Hall of Fame → **WRITE `CREDITS` as LINE 1 of NIGHT_REPORT.md** + survey. (banked_CREDITS is
  STALE 2026-07-07 — check mtime, not existence.)

### ▶ WHAT NS14 PROVED (the switch fix is a real breakthrough — verified on indigo_reach_g via recon_e4):
The **offensive-upgrade switch** (committed `ce5e391`, `battle_agent._best_switch_slot`) pushed lap 1 from the
prior 47%-at-Lance whiteout to: **cleared Lorelei/Bruno/Agatha → BROKE LANCE (room 4 at 83% lead) → reached room
5, the CHAMPION (Gary) — first time ever with this team.** TRIGGER 2: when the active can only hit RESISTED (best
damaging move ≤0.5x) while a healthy reserve's STAB is SUPER-EFFECTIVE (≥2x), field the specialist (Kadabra's
Psybeam 2x into Agatha's all-Poison line), overriding the level veto (lenient floor lv+15). Plus **anti-churn:
never switch away from a ≥2x attacker** (killed the Venusaur↔Kadabra infinite loop — Ghost hits Psychic 2x so the
disadvantage trigger kept yanking the SE attacker back out). Fail-safe, mode-side battle-brain only.

### ⛔ TWO REMAINING WALLS (both = bench too FRAIL, a LEVEL problem — the switch logic is done):
1. **Kadabra L40 faints clearing Agatha** (its L54 Ghosts hit Psychic 2x). It DOES its job (1 clean switch, KOs
   Poison-types, conserves Venusaur PP) but dies → needs ~L48 to survive as the standing Agatha specialist.
2. **Gary's CHARIZARD** (Fire/Flying) walls a solo Venusaur (Razor Leaf 0.25x, takes Fire 2x back). The answer is
   **Lapras (Surf 2x vs Charizard)** — but Lapras L39 dies earlier in the gauntlet. Needs ~L48 to survive to Gary.
   NOTE: Lapras has **NO ICE MOVE** (moveset [Surf, Body Slam]) — Surf is 2x on Charizard/Aerodactyl, x1 on the
   Dragons. Still the best Gary answer.

### GRIND FACTS (hard-won this shift — don't repeat the dead ends):
- **VR-grind-from-indigo is IMPOSSIBLE** with the current harness: `recon_grind_bench` needs GRASS tiles; Victory
  Road is a CAVE (step-encounters, no grass) → "no_safe_grass". To grind in a cave you'd have to teach the harness
  cave step-encounter pacing (unbuilt).
- **Route 18 (map 3,36, grass L23-29) is the ONLY proven grind spot** — but its lineage (`grind_base_g`) is stuck
  behind the BROKEN VR tail (see below), so its levels can't reach the E4 without fixing the tail.
- **The E4 itself is the best grinder** now that the switch fix makes Kadabra participate: E4 foes are L54-63
  (~10x Route 18 XP), it's past-VR, XP compounds within one recon_e4 process (banks banked_E4 each whiteout). That
  is exactly what `ns14_e4_loop.sh` exploits. This is the highest-EV overnight path — check it first.

### ⛔ THE NS13 OVERNIGHT CHAIN IS A DEAD END (killed this shift — do NOT relaunch it):
`ns13_overnight_chain.sh` waited for Lapras L46 then ran `tail_driver.sh`, whose `recon_victory` leg
**DETERMINISTICALLY LOSES at VR fight#104** (Water Cooltrainer Kingler/Poliwhirl/Tentacruel) — LAPRAS leads that
fight (grind party order) and Body-Slams x1 too slowly, then aborts on post-loss boulder nav. The switch fix does
NOT rescue it (Lapras Body Slam is neutral 1x, not ≤0.5x, so trigger 2 won't field Venusaur's Razor-Leaf-2x).
**To ever use the Route 18 grind path, you must REORDER Venusaur→slot0 before the tail** (Venusaur-led → Razor
Leaf 2x sweeps the Water Cooltrainer). That reorder helper is UNBUILT. Prefer the E4-self-grind loop instead.

### IF THE LOOP DOESN'T CONVERGE — the surgical next lever = LAPRAS-LEADS-GARY reorder:
Reorder the party so Lapras is slot 0 for the Champion room (or the whole E4), so its Surf 2x is fielded actively
vs Gary's Charizard/Gyarados instead of only via the (flaky, post-faint) force_switch. Combined with a few more
bench levels from the loop, that should close Gary → CREDITS. (Party-reorder actuation is the unbuilt piece.)

### MOVESETS (recon_partydump, indigo_reach_g): Venusaur L71=[RazorLeaf 25pp(only STAB), Cut, SleepPowder,
Strength]; Lapras L39=[Surf, Body Slam — NO ICE]; Kadabra L40=[Psybeam 50pw psychic = Agatha answer]; slots 1/2/4
= L9-14 CHAFF (dead weight; a PC-box drop would help but box access is Tier-2 #15, unbuilt).
Re-test cmd after any battle_agent edit: `E4_STATE=indigo_reach_g ../.venv/Scripts/python.exe -u recon_e4.py`.

## ✅ NS13 (2026-07-10): AGATHA WALL BROKEN — E4 pushed rooms 1-4, whiteout at LANCE's AERODACTYL. New wall = TOP-HEAVY TEAM (Venusaur solos; bench too weak/never fielded). Grinding Lapras+Kadabra on Route 23 now.
**WHAT NS13 DID:** NS12's overnight no-EQ VR grind-through SUCCEEDED — banked `banked_VICTORY` = a PAST-VR team
at Indigo (Venusaur L71→74, Kadabra L40, Lapras L39, healed, $13k). Promoted → `indigo_reach_g`. Ran recon_e4
from it: **CLEARED Lorelei + Bruno + AGATHA (the NS9 wall!) + reached LANCE (room #4) with all 6 alive at 47%**,
then **whited out at Lance's AERODACTYL**, reproduced across the whiteout-retry loop until money hit $0. Killed
the loop (was degrading, not converging).
**ROOT (precisely characterized — 3 compounding issues):**
1. **TOP-HEAVY TEAM.** Venusaur L74 is a monster and SOLOS every battle; the bench is Lapras L39 + Kadabra L40
   + L9-14 chaff. The bench never fields a move all run — it only switches in when Venusaur faints (too late,
   at Lance, → OHKO'd). Lapras's Surf/Ice (x2-x4 on Lance's Dragons + Aerodactyl) NEVER gets used.
2. **AERODACTYL accuracy-debuff.** Lance's Aerodactyl Sand-Attacks → Venusaur (solo, can't switch it off) whiffs
   into an accuracy spiral → can't KO it → attrition death even with 5 Full Restores. Sleep Powder locks it but
   wears off (4 turns). PP famine (Razor Leaf runs dry → Cut/Struggle x1) compounds it.
3. **BROKEN in-battle SWITCH** (`fswitch retry N → wedge frame`, the long-standing Tier-1 #5 gap) — so even
   when Lapras is alive, the engine can't deliberately field it vs the Dragons. Same wedge that blocks Agatha.
The E4 SHOP already buys Revives-first (5/3/16 caps) — revives get wasted reviving weak bench into L58 Aerodactyl,
so a shop fix won't crack it. The ONLY real fix = a survivable, deliberately-FIELDED bench.

### ▶▶ AT WAKE — CHECK THE AUTONOMOUS OVERNIGHT CHAIN FIRST (it may have rolled credits):
**`ns13_overnight_chain.sh` is RUNNING** (`ns13_chain.log` + status in `ns13_chain_status.txt`). It waits for the
Route 18 grind to bring **Lapras → L46** (its Surf/Ice = the Lance answer; Kadabra L39-40 already clears Agatha),
then kills the grind and runs **`tail_driver.sh`** = the proven no-EQ chain promote(banked_GRIND→bench_grind_kit)
→ seafoam → mansion → blaine → giovanni → victory(out-levels VR, NS12-proven) → **E4**. Banks `banked_CREDITS`
if credits roll. **AT WAKE:** `cat G:/temp/longrun/ns13_chain_status.txt` and `cat G:/temp/longrun/tail_status.txt`.
- **If `banked_CREDITS` exists / status says CREDITS ROLLED → WRITE `CREDITS` as LINE 1 of NIGHT_REPORT.md**
  (stops the loop) + the full mountain survey. Promote banked_CREDITS to canonical only per the two-timeline law.
- **If the chain died at a leg** (tail_status.txt names the failed leg): promote the last good bank + resume that
  leg's env cmd (NS9 tail block below). If E4 walled at Lance again even with Lapras L46 → the broken fswitch is
  the wall; do the LAPRAS-LEADS-E4 reorder (slot-0 swap pre-E4) so Surf/Ice is fielded actively vs the Dragons.
- **If the grind STALLED before L46:** Route 18 caps XP for high-level mons; the chain proceeds anyway at ~30min
  no-progress. A stronger grind spot (Victory Road cave L36-46, or solve Route 23's Surf-gated grass) is the
  unbuilt capability for pushing a bench past ~L45 efficiently.

### ▶ FRONTIER = grind Lapras (+Kadabra) so the bench SURVIVES Lance, then re-run the tail → E4 (the chain does this).
**⚠️ Route 23 grind (GRIND_MAP=3,42 from indigo_reach_g) WEDGES** — the team boots at R23 north edge (12,0) and
can't path south to grass (gated/watery, needs Surf/Waterfall nav the traveler lacks). Do NOT retry it blind.
**OVERNIGHT GRIND RUNNING (NS13) = the PROVEN Route 18 spot:** `ns13_grind_r18.log` — `GRIND_STATE=grind_base_g
GRIND_MAP=3,36 GRIND_DIR=west GRIND_SPECIES=131,64 (Lapras FIRST, then Kadabra) GRIND_TARGET=48`. `grind_base_g`
= promoted from NS12's `banked_GRIND` (Venusaur L65 + Lapras L37 + Kadabra L39, positioned AT Route 18 — nav
PROVEN, verified battling + participation-XP switch in the first 30s). Banks `banked_GRIND` every ~150s.
**CAVEAT:** Route 18 wilds are L23-29 → participation XP to L37-39 mons is SLOW; Lapras 37→48 may not finish
overnight. Whatever it reaches banks forward. NS12 got Kadabra→39 here; this run does Lapras first.
**CHECK IT FIRST at wake:** read `banked_GRIND` roster (`../.venv/Scripts/python.exe -c "import json;d=json.load(
open('G:/temp/longrun/banked_GRIND/journey_core.json'));[print(r['species'],r['level']) for r in d['roster']]"`).
Then **re-run the full tail** to get the leveled team back to Indigo (grind_base_g is at Route 18, badge 6 — the
tail re-badges to 8 and clears VR):
```
python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit    # leveled base
# then the NS9 tail: seafoam -> mansion -> blaine -> giovanni -> victory -> e4 (commands in the NS9 block below)
```
When it reaches Indigo → `E4_STATE=<leveled_indigo> recon_e4.py` → a leveled Lapras/Kadabra now SURVIVES the
switch-in at Lance (L48 Lapras Surf 2HKOs Aerodactyl + tanks its hit, vs the L39 chaff that got OHKO'd) → should
clear Lance → Champion → CREDITS.
**IF it STILL walls at Aerodactyl:** the broken in-battle SWITCH is the culprit (Venusaur solos, bench only
fields on faint). Surgical fix = reorder the party so LAPRAS leads the E4 (slot-0 swap pre-E4), so its Surf/Ice
is fielded actively vs Lance's Dragons, bypassing the broken fswitch entirely.
**BANKS:** `indigo_reach_g` (past-VR team at Indigo, Agatha-broken, Lance-reaching — the NS13 advance, GOOD but
bench too weak for Lance) + `grind_base_g`/`banked_GRIND` (Route 18 grind base, leveling) + `giovanni_kit_g`
(badge8) all clean. `banked_VICTORY`/`banked_E4` are temp ratchets. Canonical Champion bank UNTOUCHED.

---
## (superseded) NS13 pre-run plan — kept for the promote/tail command reference:
**Promote:** `python promote_to_workshop.py G:/temp/longrun/banked_VICTORY indigo_reach_g` then
`E4_STATE=indigo_reach_g ../.venv/Scripts/python.exe -u recon_e4.py`. E4 auto-shops Revive/Full Heal/Full Restore.

## ⛔ NS12 WALL (superseded by NS13 breakthrough above — kept as fallback): bench_grind_kit lineage's Venusaur is TOO WEAK for VR/Gary (no EQ) + the EQ teach is BROKEN on this save. TWO clean paths below.
**WHAT NS12 DID:** grind finished **Kadabra L39** (Route 18 capped, above NS9's L38 floor). Tail auto-ran
seafoam→mansion→blaine→giovanni (ALL banked OK, fast). Then **victory WIPED at VR fight#104** (Water Cooltrainer
Kingler/Poliwhirl/Tentacruel) — reproduced 4×, DETERMINISTIC, not variance.
**ROOT (two compounding bugs, both now understood):**
1. The bench-grind Venusaur = `[RazorLeaf 75, Cut 15, SleepPowder 79, Strength 70]` — a THIN battle set (Cut+Strength
   are near-useless HM moves). Its only real offense is Razor Leaf. NS9 passed VR because ITS Venusaur had **EQ +
   Razor Leaf** (NS9's lineage had Secret Power 290 to forget, keeping Razor Leaf). This lineage never got EQ.
2. The old EQ-teach forgot Razor Leaf (no 290 to drop → fell to slot 0). **FIXED + COMMITTED (3964571):** forget by
   CONTENT (protect RazorLeaf 75 + Strength 70; prefer dropping Cut 15) + gated behind `TEACH_EQ` (default OFF) +
   blind `_forget_goto`. **BUT the teach ACTUATION is deterministically broken on giovanni_kit_g:** the TM case
   re-sorts TM26 from row 13→8 and the selection never reaches the make-room dialogue → "NOT taught" at 6.3s. My
   forget-nav fix did NOT change it (failure is at case-SELECTION, not forget). So EQ can't be taught on this save
   without a real teach-actuation fix (frame-grab the open TM case to find TM26's TRUE row/scroll offset).
**Without EQ:** the team loses to **Gary** (Route 23, 2 losses) → enters VR at ~14% HP → whiteout-loops on VR 1F
(barrier ratchets open but it can't survive to the exit). Coverage exists across the team (Razor Leaf x2 vs Water,
Kadabra L39 Psychic, Lapras Surf/Ice) — the killers are HP ATTRITION + the recurring **fswitch wedge** (can't switch
to the right matchup mon mid-battle; "fswitch retry N → wedge frame"). Same wedge blocks Agatha (see NS9 memory).

### ▶ MORNING — two paths to credits. **PATH B is cleaner (skips VR + the broken teach entirely).**
**PATH B (RECOMMENDED): grind NS9's `indigo_reach_kit` Kadabra, then E4.** `states/workshop/indigo_reach_kit.state`
(banked 15:50, NS9) is ALREADY PAST Victory Road, at Indigo, with the STRONG Venusaur (EQ + Razor Leaf) + Lapras L37.
Its ONLY gap was **Kadabra L31** (Agatha PP-famine). So: grind THAT save's Kadabra to ~L42 (from a post-VR-safe spot —
Route 23 grass just S of Indigo, or Route 22; recon_grind_bench with the right GRIND_MAP — VERIFY nav first, it was
only proven on Route 18 map 3,36), bank, then `E4_STATE=<leveled_indigo> recon_e4.py` → likely CREDITS. This uses a
PROVEN-past-VR strong team and never touches VR or the broken teach. The one unknown = grind-nav from Indigo.
**PATH A (harder): make the bench_grind_kit lineage clear VR.** Either (a) FIX the EQ-teach case-selection actuation
(frame-grab `G:/temp/longrun/victory_probe` or a fresh grab of the open TM case; TM26 true row after sort) then run
`TEACH_EQ=1 VICTORY_STATE=giovanni_kit_g recon_victory.py`; or (b) over-level giovanni_kit_g's Kadabra/Lapras to
~L48+ so raw stats brute VR without EQ (needs a stronger grind spot than Route 18's L39 cap).
**OVERNIGHT BET RUNNING:** `ns12_vr_grindthru.log` — a no-EQ recon_victory from giovanni_kit_g (3600s deadline).
Every VR fight levels the team; it MAY out-level VR 1F and reach Indigo (banks `banked_VICTORY`→indigo). CHECK IT
FIRST at wake: if `banked_VICTORY` exists + log shows "Indigo", promote it → run E4. If it failed, XP was lost on the
fresh reboot (recon_victory reboots from giovanni_kit_g each launch) — go PATH B.
**giovanni_kit_g (badge8, Kadabra L39) + indigo_reach_kit (NS9, past-VR strong-Venusaur) are both banked & good.**
**Carried from NS10:** (1) `fix(victory)` EQ teach now targets Venusaur BY SPECIES (was hardcoded slot 0
— on the kit line that wasted the TM and could overwrite Kadabra's PSYCHIC, the Agatha answer). Committed
bd8777d. (2) `G:/temp/longrun/tail_driver.sh` — unattended chain: promote GRIND→bench_grind_kit → seafoam →
mansion → blaine → giovanni → victory → e4, stops on first nonzero exit, banks CREDITS if it rolls. Launch:
`bash G:/temp/longrun/tail_driver.sh` (status → `G:/temp/longrun/tail_status.txt`). **RESUME:** if grind is
dead, restart it (cmd below); when Kadabra≥L38, kill grind (`taskkill //F //IM python.exe //T`), then run
the tail driver. If tail died mid-leg, read tail_status.txt for the failed leg + promote the last good bank
and resume from that leg's env cmd (below). Everything else in this file is the validated NS9 playbook.

## ⛏️ NS9 RESULT: whole pipeline VALIDATED e2e to the E4 — true wall PINPOINTED at AGATHA (under-level PP famine). Grinding Kadabra now → re-run the (all-fixed) tail → CREDITS.

**THE NS9 BREAKTHROUGH:** ran the FULL validation sweep with the leveled kit team and it went the
distance — re-badge tail → Victory Road CLEARED → Indigo → E4 rooms 1-2 (Lorelei + Bruno) BEATEN at full
health. The E4 wall is now PRECISELY located and characterized (no longer blind): **AGATHA (room 3) = PP
FAMINE + menu white-box wedge because the bench is under-leveled** (Kadabra L31 / Lapras L37 can't KO
Agatha's L54-56 Ghosts fast enough — damaging PP runs dry across the gauntlet, then the action-menu
impostor jams switches → anti-wedge abort). FIX = grind the bench higher, esp. KADABRA (the Psychic answer
to Agatha). **Grind RUNNING now** (Kadabra-priority) and banking; then re-run the tail (all its blockers
are FIXED this shift) → E4 should push past Agatha.

**3 FIXES COMMITTED THIS SHIFT (all verified e2e):**
- `recon_seafoam` OFF-ROUTE START: routes a grind-spot start (Route 18) to Fuchsia via the general traveler.
- `recon_seafoam` PRE-CROSSING HEAL: kills the depleted-PP WHITEOUT (grind-bank starts had 0-PP Lapras →
  wiped on R20 wilds → blacked out = the (11,5) "west crossing never fired" wedge; frame-grab confirmed).
- `recon_victory` EVOLUTION-BOX DRAIN: `wedge()` now raw-presses B (ungated by dd_box) — a mid-VR
  Abra→Kadabra evolution box JAMMED overworld nav (dd_box doesn't flag it); this unblocked VR 2F→3F.
- NEW helper `promote_to_workshop.py <banked_dir> <basename>` chains banked_<X> (bare sidecars) → workshop
  kit fixture (prefixed sidecars) between tail legs.

**BANKED FIXTURE CHAIN (states/workshop, each verified e2e, badges/levels rising):** `bench_grind_kit`
(badges=6, grind base, Lapras L37) → `cinnabar_kit_g` → `secretkey_kit_g` → `blaine_kit_g` (badge 7) →
`giovanni_kit_g` (badge 8, leveled team) → `indigo_reach_kit` (at Indigo, healed, $21k → shopped down).

### ▶ THE PLAN — continue the grind, then re-run the tail (now trivial, all-fixed) → E4 → CREDITS.
**1. GRIND is RUNNING** (`G:/temp/longrun/ns9_grind_kadabra.log`, 600-min budget, banks `banked_GRIND`
every ~150s). KADABRA-priority: Abra→Kadabra already evolved (L10→L19 in pass 1) → climbing to L42, then
Lapras L37→42. **Continue/restart if dead** (from `pokemon_agent/`):
```
GRIND_STATE=bench_grind_kit.state GRIND_TARGET=42 GRIND_SPECIES=63,64,131 GRIND_MAP=3,36 GRIND_DIR=west \
  GRIND_MIN=600 GRIND_PROBE_S=150 ../.venv/Scripts/python.exe -u recon_grind_bench.py > G:/temp/longrun/nsX_grind.log 2>&1 &
```
Promote after meaningful banks: `python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit`.
Rate at Route 18 (L23-29 wilds) SLOWS as the bench out-levels them; Kadabra L19→42 is the long pole. If it
STALLS (grind() marks a species stalled), Route 18 may be too weak for L40+ — but participation XP banks
regardless of wild level, so it should keep creeping. **TARGET can drop to ~38** if the grind is too slow:
Kadabra L38 Psychic still 2× OHKO-range on Agatha's Ghosts; re-validate the E4 at whatever level lands.

**2. RE-RUN THE TAIL** when Kadabra ≈ L40+ (Lapras L37 is already enough for Lorelei). Each leg ~90s, all
blockers FIXED. From `pokemon_agent/`, promote between legs (bank dir names in parens):
```
python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit    # leveled base
SEAFOAM_STATE=bench_grind_kit.state  ../.venv/Scripts/python.exe -u recon_seafoam.py  > G:/temp/longrun/x_seafoam.log 2>&1   # -> banked_CINNABAR
python promote_to_workshop.py G:/temp/longrun/banked_CINNABAR cinnabar_kit_g
MANSION_STATE=cinnabar_kit_g         ../.venv/Scripts/python.exe -u recon_mansion.py  > G:/temp/longrun/x_mansion.log 2>&1   # -> banked_SECRETKEY
python promote_to_workshop.py G:/temp/longrun/banked_SECRETKEY secretkey_kit_g
BLAINE_STATE=secretkey_kit_g         ../.venv/Scripts/python.exe -u recon_blaine.py   > G:/temp/longrun/x_blaine.log 2>&1    # -> banked_BLAINE (badge7)
python promote_to_workshop.py G:/temp/longrun/banked_BLAINE blaine_kit_g
GIOVANNI_STATE=blaine_kit_g          ../.venv/Scripts/python.exe -u recon_giovanni.py > G:/temp/longrun/x_giovanni.log 2>&1  # -> banked_GIOVANNI (badge8)
python promote_to_workshop.py G:/temp/longrun/banked_GIOVANNI giovanni_kit_g
VICTORY_STATE=giovanni_kit_g         ../.venv/Scripts/python.exe -u recon_victory.py  > G:/temp/longrun/x_victory.log 2>&1   # -> banked_VICTORY (Indigo); RESUME_STAGE=1 to ratchet a mid-VR wedge
python promote_to_workshop.py G:/temp/longrun/banked_VICTORY indigo_reach_kit
E4_STATE=indigo_reach_kit            ../.venv/Scripts/python.exe -u recon_e4.py       > G:/temp/longrun/x_e4.log 2>&1        # -> CREDITS or the next blocker
```
**IF CREDITS ROLL:** write `CREDITS` as LINE 1 of NIGHT_REPORT.md (stops the loop) + full mountain survey.

### ⚠️ E4 SPECIFICS (from the NS9 run — read before the E4 attempt):
- **Rooms 1-2 (Lorelei, Bruno) already cleared** at full health with the L37 team. The wall is room 3 Agatha.
- **PP FAMINE is the killer:** the gauntlet is 5 rooms with no heal between; damaging PP depletes. Higher
  levels = fewer turns/KO = less PP burned. Kadabra L42 Psychic should 1-2-shot Agatha's Ghosts (Gengar/
  Haunter/Arbok/Golbat), ending fights before famine. She has NO Ethers (famine is unrecoverable mid-fight).
- **Menu white-box wedge:** the E4 rooms trigger the "action-menu impostor (white box, DEAD cursor)" +
  "famine switch did not confirm" actuation jam (known E4 livelock family). Levels mask it (fewer menu
  windows); if it still bites post-grind, that's the next fix (see [[pokemon-e4-livelock-family-killed]]).
- **EQ=NO gotcha:** recon_victory's Phase0 EQ-teach targets slot 0, but the grind leaves Lapras/Kadabra
  leading (Venusaur is slot 5) → EQ taught to the wrong mon and FAILED. Venusaur has no Earthquake. Minor
  (Razor Leaf/Surf/Psychic carry the E4). To fix: swap Venusaur to slot 0 before the tail, or make the EQ
  teach target Venusaur by species (3).
- E4 shopping: recon_e4 auto-buys Full Restore/Hyper/Revive at the Indigo Center (money-aware, spent to ~$1k).

### 🏁 ALL 8 BADGES on the kit line; VR CLEARED; E4 rooms 1-2 down. Only Agatha+ (a LEVEL problem) remains.
Memory: [[pokemon-nightshift9-e4-validation-agatha-wall]] · [[pokemon-nightshift7-bench-grind-nav-island]] ·
[[pokemon-e4-gauntlet-truths]] · [[pokemon-e4-livelock-family-killed]].

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — venv = 2-PID shim = ONE logical run; kill
`taskkill //F //IM python.exe //T`). Flags module = `field_moves` (fm.read_flag). Bank dir names differ
from the env var (seafoam→banked_CINNABAR, mansion→banked_SECRETKEY, blaine→banked_BLAINE, giovanni→
banked_GIOVANNI, victory→banked_VICTORY, grind→banked_GRIND). recon_victory RESUME_STAGE=1 ratchets a
mid-VR wedge from its own stage_victory bank.

WATCH STATUS: canonical Champion bank CLEAN + untouched (NS12 only edited workshop staging fixtures + temp
banks — the canonical timeline is safe). Sherpa frontier = get a PAST-VR team with a leveled-enough Kadabra
to the E4 (PATH B: grind NS9's indigo_reach_kit Kadabra L31→~42, then recon_e4). The bench_grind_kit lineage
(badge8, Kadabra L39) is banked but CANNOT clear VR without EQ, and the EQ teach is broken on that save — so
prefer PATH B. Overnight no-EQ VR grind-through running (self-terminates ~1hr; check banked_VICTORY at wake).
Pop-in = `python pokemon_agent/watch.py`.
