# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ BADGE 5 (Koga / SOUL 0x824) — WON FULLY UNAIDED + BANKED (night-shift 1, b10bfdf)
The last-mile is CLOSED. recon_longrun from koga_retry_kit (badge4, 0 potions) → GOAL 0x824 in 109.9s:
she autonomously bought 30 Super Potions at the Fuchsia Mart (oracle stock_up 6 + the new pre-gym
POTION-STALL leg 24), stalled through Koga's 4-mon gauntlet via in-battle use_potion, took the Soul Badge.
Banked → states/workshop/koga_done_kit.state (badges=5, Fuchsia (9,33); round-trip verified). See
[[pokemon-nightshift1-fuchsia-mart-mapped]]. WHAT WAS BUILT (campaign.py):
- Fuchsia Mart MAPPED: door (11,15) animated 0x69 → interior (11,1), row2 Super Potion(22,700) live-verified.
  In CITY_MART_DOORS + MART_STOCK[FUCHSIA]=[2,3,22,28,23,88].
- `DOOR_APPROACH_WAYPOINTS[(FUCHSIA,(11,15))]=[(20,24),(19,18),(15,16)]` — enter_warp walks these first
  (the mart door is in a pond-walled pocket where direct BFS-travel oscillates). Extend the KB if other
  in-town doors show the same oscillation.
- `POTION_STALL_GYMS={"Koga":30}` + `_stock_potions_for_gym` (called in beat_gym, fires even goal-pinned).

## FRONTIER: BADGE 6 (Sabrina / Saffron, MARSH 0x825) — ROOT DIAGNOSED: no armed SILPH CO. questline
Night-shift 1 ran the look-ahead (`LONGRUN_GOAL_FLAG=0x825 recon_longrun koga_done_kit.state 30`, log
/g/temp/s1_badge6.log). RESULT: she autonomously reaches **Saffron City** — the **TEA errand + gatehouses
ALREADY WORK** (FLAG_GOT_TEA wired in exit_gates). Then she STALLS at Sabrina:
```
!! GYM-INTERIOR WALL: beat_gym stuck x5 on Sabrina — head_to_gym structurally parked on (3,10)
!! STRUCTURAL DEAD-ROUTE: ['head_to_gym'] proven dead ... until she leaves it (or a questline arms)
```
**ROOT: Sabrina's gym door is Rocket-BLOCKED until Silph Co. is cleared, and NO questline arms to send her
into Silph Co.** The gate-questline deriver exists (campaign.py ~5533 `_derive... unlock errand`; the Tea
errand proves it works) but the gate KB (gamedata/frlg_gates.json) has only a `saffron_unlock` NOTE (line
395) + a Sabrina-door note (line 594) — NO executable Silph Co. step. So head_to_gym parks and prunes.

**THE BUILD (badge-6 = a hideout-class dungeon strike):** wire a Silph Co. liberation questline that arms
when she's walled at Saffron's Rocket-blocked gym:
- Silph Co. overworld door = **(33,30)** on Saffron; Saffron GYM door = **(46,12)** (campaign.py:381-385).
- Interior = a TELEPORT-PAD MAZE (11 floors) — reuse `pad_nav.PadNav` (the Silph/Sabrina pad-router;
  DESCENT shift 5) + read_warps. Prior-art probes: recon_silph.py, recon_silph_probe{,2,3,4}.py,
  recon_silph7f_probe.py (from the original credits climb — Silph WAS cleared then; mine them for the pad
  graph + Card Key/Giovanni floors). Silph Co. maps named (1,47)-(1,58) at campaign.py:2569.
- Chain: enter Silph (33,30) → get **Card Key on 5F** (opens the locked pad doors) → reach **Giovanni #2 on
  11F** + free the Silph president → sets the liberation flag → Saffron frees, gym door opens → head_to_gym
  works → beat Sabrina (Marsh 0x825). Find the liberation flag id in the disasm/gates KB.
- Sabrina KB: Psychic ("sees your moves"; folk remedy ghosts/bugs). Venusaur solo-carry L53 may wall on
  Alakazam — watch for an attrition/type stall (team debt #3; the potion-stall leg generalizes if so — add
  "Sabrina" to POTION_STALL_GYMS, and Saffron HAS a Mart+Dept store for the buy).

Approach: mine recon_silph*.py for the pad graph, wire the arming (gate KB step keyed to the Saffron gym
lock), long-run from koga_done_kit, diagnose the first Silph-interior stall, iterate, bank silph/sabrina
checkpoints as she clears. This is a multi-run dungeon build — the biggest remaining pre-E4 stretch.

## RESIDUAL (watchability, non-fatal, unchanged): gym maze junior-spin
The invisible-wall maze (Koga; likely Sabrina too) leaves ~4 juniors un-engageable; she tries each 4× then
burns the 14 clear-round cap (~minutes of log spam) before bailing to the leader. Correctly bails (not
fatal). Fix in `_clear_junior_trainers` (campaign.py ~3200): reachability pre-check — defer an unreachable
junior in 1 try not 4, remember the deferred set across blackout-retries.

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW; kill `taskkill //F //IM python.exe //T`).
recon_longrun arg1 = state BASENAME **with .state extension** (resolve_state returns None without it),
arg2 = max_minutes, goal via `LONGRUN_GOAL_FLAG=0x825`. Banks to G:/temp/longrun/banked_GOAL — promote its
{kira_campaign.state,journey_core,soul,strat_memory,world_model}.json into workshop/ with the fixture
basename to advance the Sherpa line. Flags module = `field_moves` (fm.read_flag). Mart-clerk sig = gfx68 @(2,3).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = post-Koga (badges=5) at Fuchsia,
heading for Saffron/Sabrina via the Tea→Silph chain. Pop-in = `python pokemon_agent/watch.py`.
