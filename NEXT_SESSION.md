# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ NS4 BANKED: SILPH CO. SOLVED UNAIDED (Gary type-wall broken) → frontier = SABRINA BADGE (0x825)
**VERIFIED e2e, fully autonomous from a clean koga_done_kit (NO injected items):** HYPER-STALL bought
15 Hyper + 5 Revive at Saffron Mart ($30k→…) → silph_strike → GARY WON 5-2 (out-heals Charizard's Fire)
→ Lapras → Giovanni → SAFFRON FREED (0x3E) in 226s. Banked → promoted **`silph_done_kit.state`**
(workshop; Venusaur L56, Saffron freed, ~$16k, leftover ~11 Hyper + 3 Revive). Commits: 1e9e43c (diagnosis
+ battle guard), db84e10 (Saffron Mart + stock_hyper_potions kit). See [[pokemon-nightshift3-silph-gary-skip]].

### FRONTIER — BADGE 6 (Sabrina / Marsh 0x825), gym now OPEN:
Run `LONGRUN_GOAL_FLAG=0x825 LONGRUN_BATTLE_LOG=1 recon_longrun silph_done_kit.state 40`. Sabrina's team
is PSYCHIC (Kadabra/MrMime/Venomoth/Alakazam) — Psychic hits Grass/POISON Venusaur ×2, so it's the SAME
out-heal-with-Hyper dynamic as Charizard. She carries leftover Hyper+Revive and Saffron Mart is right
there. WATCH FOR: (a) beat_gym pad-router crossing Sabrina's INTERIOR teleport maze (port seam =
recon_sabrina if the general walk wedges); (b) if she runs low on Hyper mid-gym, a Sabrina POTION-STALL
— the Saffron Mart is now mapped (door (40,21), Hyper row1) + `stock_hyper_potions` is reusable, or add
"Sabrina" to POTION_STALL_GYMS (but that buys the cheapest potion — prefer wiring stock_hyper_potions into
the pre-Sabrina beat_gym leg so she buys HYPER, not Super). If Alakazam is a hard attrition wall, the same
Hyper+Revive kit that beat Charizard should carry it.

## (superseded) NS4 root-cause detail — BADGE 6 Silph Gary = a TYPE-MATCHUP wall (Charizard), NOT PP
**NS4 CORRECTS NS3's diagnosis (which was WRONG).** The whole Silph chain still FIRES e2e (koga_done_kit
→ Saffron → prereq gate → silph_strike → climbs 1F→9F, grabs Card Key, pad-chains to 7F, Gary auto-engages).
NS3's committed "FINISH-THE-FOE guard" (battle_agent) + route-around-disable (silph_strike) are still
UNCOMMITTED in the tree and are HARMLESS but did NOT fix the wall.
**TRUE ROOT CAUSE (battle-log autopsy, LONGRUN_BATTLE_LOG=1, run ns3_silph4.log):** the loss event is
**Venusaur FAINTING to Gary's CHARIZARD** (Fire/Flying), the worst possible matchup for Grass/Poison
Venusaur — Charizard's Fire hits her **2×** while it **quad-resists her Grass (Razor Leaf ×0.25)** and her
only neutral move is weak **Cut (Normal 50BP, no STAB)**. In every famine-switch context Venusaur (sp 3)
already reads **hp=0** — the "PP FAMINE" is a downstream red herring on the FODDER mon after Venusaur is
already down. She DOES heal, but only with **Super Potions (50 HP)** which ~= Charizard's chip, so she
treads water and dies with Charizard at ~29/119. The Max Ether (1 in bag) never fired — the ether-instinct
gate only checks move-SLOT-0 (Razor Leaf, ×0.25 useless vs Charizard), so `use_ether` was offered 0× across
5 famines (secondary bug, not the wall). Route-around Gary is GEOMETRICALLY IMPOSSIBLE (NS3, confirmed).
**DELIVERY BUILT (NS4, verifying):** Saffron Mart mapped from pret (door (40,21), stock [GreatBall,
HyperPotion,Revive,FullHeal,EscapeRope,MaxRepel] → Hyper=row1) → added SAFFRON_MART_DOOR +
CITY_MART_DOORS[SAFFRON] + MART_STOCK[SAFFRON]=[3,21,24,23,85,88]. New camp method
`stock_hyper_potions(target=SILPH_HYPER_TARGET=20)` counts Hyper(21) SPECIFICALLY (30 Supers must not
read as stocked) + buys the shortfall at Saffron Mart. Wired into `silph_strike.run_strike` at the
`here==SAFFRON` block (before the pre-dungeon heal). VERIFY (autonomous, NO injection):
`LONGRUN_GOAL_FLAG=0x3E LONGRUN_BATTLE_LOG=1 recon_longrun koga_done_kit.state 35` → expect HYPER-STALL
buys ~20 Hyper at Saffron Mart → GARY WON → 0x3E freed. Then run 0x825 (Sabrina badge) + bank.
**THE FIX DIRECTION (PROVEN THIS shift via injected fixture):** out-heal Charizard's Fire with **HYPER POTIONS (200 HP)**
instead of Super Potions (50). `_HEAL_ITEMS_PREF` ALREADY prefers Hyper(21)>Super(22) — so if she CARRIES
Hyper Potions the battle instinct auto-uses them; NO battle-code change needed. She has $30,330 + Saffron
has a Mart/Dept (Hyper Potions). **EXPERIMENT IN FLIGHT:** injected 20 Hyper Potions →
`silph_hyper_test.state`, running `LONGRUN_GOAL_FLAG=0x3E LONGRUN_BATTLE_LOG=1 recon_longrun
silph_hyper_test.state 35`. If Venusaur now out-heals Charizard and WINS Gary → 0x3E freed → theory proven →
BUILD the delivery: add Saffron Hyper-Potion stall (POTION_STALL_GYMS + fire before the Silph strike;
Saffron Mart mapped door (40,21) Hyper row1 per NS3). If she STILL loses with unlimited Hyper Potions →
it's a hard TEAM wall (needs a leveled Charizard-counter / higher-level Venusaur) — pivot to a team leg.
**SOUL DEBT (unchanged, now the crux):** Silph Gary punishes the solo carry — a real team makes it clean.

### (NS2 build — still valid, all COMPILES/WIRED/now VERIFIED-to-fire):
- **`pokemon_agent/silph_strike.py`** (NEW) — faithful port of the proven `recon_silph.py` state machine
  into the camp-driven strike-module shape (same as hideout_strike/tower_strike). `run_strike(camp, log,
  dbg_dir)` → `'freed_saffron'` on FLAG_HIDE_SAFFRON_ROCKETS (0x3E) set + walked out to Saffron. Enters
  Silph via Saffron street door (33,30)→(1,47), pre-dungeon heal, then the pad-maze climb: 9F pad→5F pocket
  Card Key (355) → pad chain 9F(9,4)→3F(2,14), 3F(13,14)→7F(5,4) [Gary + Lapras], 7F(5,8)→11F → Giovanni
  #2 → president. Returns `in_silph` (flag set, exit WIP) / `not_here` / `failed`.
- **`gamedata/frlg_gates.json`** — added flag `FLAG_HIDE_SAFFRON_ROCKETS` (id 62 / 0x3E) + a capability of
  the same key (`via:"strike"`, NO door → strike fires first per campaign.py:6125, success=("flag",...)).
- **`campaign.py`**: (1) `GYM_PREREQS` table (Sabrina → 0x3E) + `_gym_prereq_gate(gym)` — synthesizes a
  STORY_NPC gate when a gym's DOOR is story-locked (no tree for the HM probe). (2) Wired into the gym-stuck
  branch (~line 9037): story-prereq gate armed FIRST (before `_gym_gate_probe`) when `_active_questline is
  None`. (3) Registered `("flag","FLAG_HIDE_SAFFRON_ROCKETS")` → `_silph` in the `_questline_strike`
  registry; `in_silph` handled alongside `in_hideout` in the exit-WIP branch.

**THE FLOW:** koga_done_kit (badges=5, Fuchsia; has Tea + Fuji) → Tea gatehouses → Saffron → beat_gym stuck
on Sabrina → `_gym_prereq_gate` arms the Silph questline → `_run_questline_step` → door-less strike step →
`_questline_strike` runs `silph_strike` → 0x3E set → gym unblocks → beat_gym enters the pad maze (already
handled by beat_gym's recon_sabrina pad-router) → Sabrina (Marsh 0x825).

**VERIFY COMMAND (isolates the NEW code):**
`LONGRUN_GOAL_FLAG=0x3E .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py koga_done_kit.state 35`
→ GOAL when Saffron is freed. Then re-run with `LONGRUN_GOAL_FLAG=0x825 ... 45` for the full Sabrina badge.
Bank the GOAL checkpoint (G:/temp/longrun/banked_GOAL) → promote to workshop/silph_done_kit.state (then
sabrina_done_kit.state). If the Silph strike stalls mid-tower, read the log for the first floor/pad that
wedges and fix in silph_strike.py (the pad coords are the recon_silph ground truth — check read_warps live).

**WATCH FOR (predicted next blockers after Silph):** (a) beat_gym pad-router crossing Sabrina's interior
(strike-solved historically; port seam = recon_sabrina if the general walk wedges); (b) an ATTRITION/type
wall vs Sabrina's Alakazam (Venusaur solo-carry) — the generalizes-to-Sabrina fix is adding "Sabrina" to
`POTION_STALL_GYMS` (Saffron HAS a Mart + Dept store; the potion-stall leg is proven vs Koga).

## RESIDUAL (watchability, non-fatal, unchanged): gym maze junior-spin
Invisible-wall gyms (Koga; likely Sabrina) leave ~4 juniors un-engageable; she tries each 4× then burns
the clear-round cap before bailing to the leader (correct bail, not fatal). Fix in `_clear_junior_trainers`
(campaign.py ~3200): reachability pre-check — defer an unreachable junior in 1 try, remember across retries.

## ✅ BANKED PRIOR: BADGE 5 (Koga) WON FULLY UNAIDED (NS-1, b10bfdf)
koga_done_kit.state (badges=5, Fuchsia (9,33)). Fuchsia Mart mapped + POTION_STALL_GYMS={"Koga":30}.
See [[pokemon-nightshift1-fuchsia-mart-mapped]].

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW; kill `taskkill //F //IM python.exe //T`).
recon_longrun arg1 = state BASENAME **with .state**, arg2 = max_minutes, goal via `LONGRUN_GOAL_FLAG=0x825`.
Flags module = `field_moves` (fm.read_flag). Silph anchors = {(3,10)} | SILPH_MAPS ((1,47)..(1,58)).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = post-Koga (badges=5) at Fuchsia,
Silph liberation questline WIRED, verification long-run in flight. Pop-in = `python pokemon_agent/watch.py`.
