# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ⛏️ NS7 IN FLIGHT: BENCH POWER-LEVEL (the fix for the E4 team wall) → then VICTORY ROAD → E4 → CREDITS
**DECISION (data-backed):** NS6's convergent-VR push is CONFIRMED BAD — its `stage_victory` shows it leveled
the SOLO carry (Venusaur L60→**L65** + taught Earthquake) but the **bench NEVER grew** (Kadabra L16→18, Lapras
L25→27) because trainer fights solo Venusaur (the participation-switch only fires while GRINDING), and it bled
money **$36k→$4.5k** across wipe-halvings. So NS7 pivots to the CEO's prescribed fix: from the CLEAN rich
`giovanni_done_kit` ($36k, full PP, Venusaur L60), **GRIND the two real teammates BEFORE Victory Road** so
VR + E4 fall first-try with no wipes.
**HARNESS BUILT: `pokemon_agent/recon_grind_bench.py`** (participation-XP grind, NO PC-box needed): fields ONE
teammate as lead — **Lapras** (species 131, Water = Charizard + Lance answer) then **Kadabra** (species 64,
Psychic = Agatha + Bruno answer) — arms `battle_agent.PROTECT_LEAD_GRIND` so every wild battle switches the weak
lead → Venusaur (ace) turn 1 (weak mon banks XP, takes no hit; Venusaur tanks+KOs). Fodder (Rattata/Spearow/
Drowzee) is never fielded → no wasted XP, no boxing. Grinding re-levels Venusaur too. Also fixes movesets free
(Kadabra learns Psychic by level-up). Banks every re-entry → G:/temp/longrun/banked_GRIND (promote → `bench_grind_kit`).
### ▶▶ GRIND IS RUNNING & BANKING (Route 18) — CONTINUE IT, then re-badge 7/8, then VR/E4.
**MECHANIC PROVEN + PRODUCTIVE SPOT FOUND.** The participation-XP switch fires and banks XP to the weak lead.
The BLOCKER all shift was LOCATION (details below); the working answer is **Route 18 (Cycling Road south),
OPEN grass L23-29, one hop WEST of Fuchsia, Fuchsia Center adjacent**. Base = **`surf_ready_kit.state`** (the
Fuchsia kit fixture — its world graph is CONNECTED, unlike giovanni_done_kit whose Viridian is a graph-ISLAND).
**LAUNCH / CONTINUE (RESUME_STAGE=1 continues the banked_GRIND ratchet across restarts — DON'T re-roll):**
```
cd pokemon_agent
GRIND_STATE=surf_ready_kit.state GRIND_TARGET=38 GRIND_SPECIES=131,63,64 GRIND_MAP=3,36 GRIND_DIR=west \
  GRIND_MIN=60 GRIND_PROBE_S=150 RESUME_STAGE=1 ../.venv/Scripts/python.exe -u recon_grind_bench.py \
  > G:/temp/longrun/ns8_grind.log 2>&1 &
```
GRIND_SPECIES is PRIORITY-ORDERED: **131=Lapras** (Charizard/Lance answer — the must-have) grinds to TARGET
first, then **63=Abra** (participation XP evolves it → Kadabra @L16 → learns Confusion → keeps leveling — the
Agatha answer), then 64 if already Kadabra. **RATE IS MODEST** (~1 level / 2.5 min at Route 18; Lapras L25→38
Slow-group ≈ 60k XP ≈ 1-1.5 hrs; the psychic climb from L10 is longer) — this is a MULTI-SHIFT grind that banks
progressively (banked_GRIND updates every pass). **TARGET can drop to ~L35** if pressed: Lapras L35 landing
Surf on Charizard (2×) after a Venusaur Sleep Powder + Hyper stall is enough; the psychic mon ~L34 for Agatha.
**RE-BADGE TAIL (after the grind — badges=6 on this base):** with the leveled team, re-run the proven kit
strikes to re-earn badges 7+8 (each TRIVIAL now): `recon_seafoam` (SEAFOAM_STATE=<grind bank>) → `recon_mansion`
→ `recon_blaine` → `recon_giovanni` — same env-parametrized pattern. Bank → a `giovanni_done_kit`-equivalent
with a STRONG bench. THEN: box fodder (recon_lapras deposit flow) → `recon_victory.py` (VR falls first-try) →
port+run `recon_e4.py` (E4_STATE env) → Champion → **CREDITS**.
**⚠️ NAV LESSONS (hard-won this shift — the Mt-Moon-class nuances):** (1) `giovanni_done_kit`'s **Viridian is a
world-graph ISLAND** (9 nodes, only weak Route-2 grass) — the kit reached Viridian via the SEA so the overland
links were never learned; you CANNOT grind giovanni directly without repairing that graph. (2) **Route 15 grass
is GUARDHOUSE-DIVIDED** — walk_to_map's east edge lands at (0,12) where grass (x≥20) is UNREACHABLE; the grass
sits past warps (9,11)/(16,11)→(24,0) which do NOT fire on plain step-on (need a door/UP trigger — unsolved).
Route 18 (west, OPEN) sidesteps this. (3) `grind()` needs GRASS tiles + a reachable Center; **caves (VR) have
no grass** so grind() can't use them, and Route 22 wilds are L2-8 (useless). Use `recon_grind_bench.py`'s scan
pattern (reachable-grass BFS from the entry) to vet any new spot before committing.
**ALT (avoids the re-badge tail, UNVERIFIED):** repair giovanni_done_kit's Viridian graph-island so route() can
path to central-Kanto grass (drive Viridian→Route2→Pewter→Route3→…→Route18 overland via a multi-hop walk_to_map
chain, learning edges en route), then grind the already-badge-8 team directly. Cleaner if it works.

## 🏁🏁 NS5 BANKED: ALL 8 BADGES (7 Blaine + 8 Giovanni) WON UNAIDED → frontier = VICTORY ROAD → E4 → CREDITS
**Badge 8 (Earth/Giovanni 0x827) WON** from `blaine_done_kit` via kit-parametrized `recon_giovanni.py`
(GIOVANNI_STATE env): 5 north sea crossings Cinnabar→Viridian (Surf) → Viridian Gym spin-tile maze
(spin_nav) → Giovanni. **SOLO-CARRY ATTRITION FIX committed** (6e20c98): the ~6 spin-floor juniors chip
the lone Venusaur to 2% *during* the cross → she was KO'd first-turn at Giovanni (run1 loss). Added a
PRE-LEADER HEAL (if lead<85% at Giovanni's front, back out to Viridian, heal, re-cross — beaten juniors
stay beaten so no re-chip, engage fresh). Verified e2e: 2% → healed → re-crossed to 100% → Giovanni WON →
**badges=8**. Banked **`giovanni_done_kit`** at Viridian (3,1), full HP. Party UNCHANGED: Venusaur L59 +
Lapras L25 + Kadabra L16 + Spearow/Rattata/Drowzee bench.

### ▶ FRONTIER — VICTORY ROAD → ELITE FOUR → CHAMPION → CREDITS (write CREDITS as line 1 of NIGHT_REPORT to stop the loop).
**NS6 IN FLIGHT:** `recon_victory.py` is now **kit-parametrized** (commit: VICTORY_STATE env + `_resolve_state`
+ kit sidecars + utf-8 stdout — same mechanical port as recon_giovanni). Launch cmd:
`VICTORY_STATE=giovanni_done_kit.state ../.venv/Scripts/python.exe -u recon_victory.py` (run from
`pokemon_agent/`, log → G:/temp/longrun/ns6_victory1.log). It does Phase0 **EQ teach** (TM26→Venusaur over
Secret Power — the 100-BP neutral carry for E4) → Phase1 **R22 Gary** → Phase2-3 **R23 badge gates** →
Phase4 **VR boulder puzzle** (fully offline-solved, elevation-aware) → Phase5 **Indigo Plateau bank**
(banked_VICTORY). If it wedges mid-VR, `RESUME_STAGE=1` ratchets from stage_victory (don't re-roll Gary).
On success: promote banked_VICTORY → `states/workshop/indigo_reach_kit.*` then port **recon_e4.py** the same
way (E4_STATE env). Chain: (1) R22 Gary, (2) R23 gates (all 8 ✓), (3) VR, (4) E4 Lorelei→Bruno→Agatha→
Lance→Champion Gary (`recon_e4.py`/`recon_agatha.py` exist — parametrize identically).

### ⚠️ THE E4 TEAM WALL — **CONFIRMED (NS6): it manifests ALREADY at VICTORY ROAD 1F**, not just the E4.
NS6 ran the kit-victory strike and it CONVERGES SLOWLY but bleeds money. Observed truth: VR 1F has an **Ace
Trainer with a full starter-evolution team (Raticate/Charmeleon/Charizard/Ivysaur, ~L43)** sitting in the
ladder corridor — you MUST WIN it to pass (a LOSS never sets its defeated-flag, so it re-fights every life).
The kit's **solo Venusaur + Lapras + L8-16 fodder** loses it: Venusaur/Lapras get worn, then the bench faints
on switch-in (Ivysaur sits at 107/107 while L13 Spearow/etc. get OHKO'd). She whites out → Viridian free-heal
→ re-crosses R22/R23 (Gary already consumed) → re-enters. **Net progress DOES ratchet** (won trainers stay
beaten; NS6 reached VR **2F, 2f-switch1 barrier open** by life 3) — BUT each wipe **HALVES money** ($36k start),
and the 1F boulder puzzle re-solves each life (boulders reset on map reload; only trainer-defeated flags
persist). So it's a convergent-but-costly grind that may reach Indigo underfunded for the E4 kit.
**THE REAL FIX (soul-debt #3, now unavoidable): BENCH-GRIND before VR.** Level **Lapras L25→~L45** (Surf
carry + Perish Song) and **Kadabra L16→~L45** (Psychic) so she fields a genuine 2-3 mon squad. Then VR
trainers fall first-try (no wipes, no money loss) AND the E4 is winnable. Grind harness: `recon_longrun`
strategic-grind (POKEMON_STRATEGIC_GRIND=1, save-safe weak-member reorder — see
[[pokemon-strategic-underlevel-grind]]) from `giovanni_done_kit`, OR a bespoke grind strike on VR-adjacent
L40+ wilds. **DO THIS FIRST next shift if NS6's victory run didn't bank Indigo cheaply.**

The E4 proper is a 5-battle gauntlet (RESETS to Lorelei on any whiteout) of L53-63 teams. The kit's **SOLO Venusaur
L59 + fodder bench is thin for it**: Agatha's Gengar (Ghost/Poison) resists Grass; Lance's Dragonite/Gyarados
hit hard; **Champion Gary's Charizard 2×-burns Venusaur while quad-resisting Razor Leaf** (the exact Silph-Gary
type wall — NS4). Venusaur has Sleep Powder + Razor Leaf + Strength + Cut, Lapras has Surf (×2 vs Lance's
Ground/Rock-adjacent, ×4 vs Dragonite via Ice? no — Surf is Water; Lapras also had no Ice-damage move, Body
Slam only). LIKELY NEEDED FIRST: level the bench (esp. Lapras L25→40s as a second wall + Kadabra) and/or a
heavy Hyper-Potion stall (she has 11 Hyper + 29 Super + 3 Revive; buy more at any Mart). The credits-line E4
(memory: pokemon-e4-gauntlet-truths / e4-livelock-family-killed) was cleared with a STRONGER team — expect the
kit's solo carry to need a grind/stall pass BEFORE the gauntlet. This is the soul-debt #3 finally load-bearing.
Consider a bench-grind long-run (recon_longrun strategic-grind) before the E4 strike. See
[[pokemon-e4-gauntlet-truths]] + [[pokemon-e4-livelock-family-killed]].

### (task 5, still deferred) Kira-timeline AUTONOMY wiring: tonight's badges 7-8 used kit-parametrized recon
strikes (env-driven) to bank distance FAST. The autonomous campaign wiring (register safari_strike +
surf/mansion/seafoam questlines so recon_longrun fires the whole chain from sabrina_done_kit unaided) is the
remaining rope-laying — the look-ahead DID autonomously reach Viridian + enter every gym; the only gaps the
general loop couldn't self-crack were the bespoke interiors (Safari pond-maze, Seafoam boulder-cascade,
Mansion statue-toggle, gym spin-floors) — each has a proven strike ready to register in `_questline_strike`.

## (done NS5) BADGE 7 (Blaine/Volcano 0x826) WON FULLY UNAIDED — full Surf-gated chain banked:

**The ENTIRE Surf-gated frontier fell in one shift.** All legs = proven credits-line recon strikes,
parametrized for the kit line via env (`LAPRAS_STATE`/`SAFARI_STATE`/`SURFTEACH_STATE`/`SEAFOAM_STATE`/
`MANSION_STATE`/`BLAINE_STATE` — each resolves a workshop basename + its `.<sidecar>.json`). Banked chain
(states/workshop, each verified e2e): `lapras_fielded_kit` (PC withdraw — recon_lapras) → `fuchsia_lapras_kit`
(look-ahead Saffron→Fuchsia) → `safari_hms_kit` (recon_safari: HM03 Surf + HM04 Strength, 62s) →
`surf_ready_kit` (recon_surf_teach: Surf→Lapras, Strength→Venusaur) → `cinnabar_kit` (recon_seafoam: R19→R20
→Seafoam boulder cascade 0x2D2→Cinnabar, 88s) → `secretkey_kit` (recon_mansion: Secret Key 0x1A8, 83s) →
**`blaine_done_kit`** (recon_blaine: 6 quiz doors A/B/B/B/A/B + Blaine → **badge 0x826, badges=7**, 72s).
Party now: Venusaur L59 (Razor Leaf/Cut/Sleep Powder/Strength) + Lapras L25 (Surf/Body Slam/Confuse Ray/
Perish Song) + Kadabra L16 (Abra evolved vs Blaine) + Spearow/Rattata/Drowzee bench. She's at Cinnabar (3,8).
6 commits. Shared fix: travel.py "🌊 SURF MOUNT" emoji crashed cp1252-piped strikes → utf-8 stdout added.

### ▶ FRONTIER — BADGE 8 (Giovanni / Viridian Gym, Earth 0x827), then Victory Road → E4 → CREDITS.
Run `LONGRUN_GOAL_FLAG=0x827 recon_longrun blaine_done_kit.state 18`. Billed road to Viridian = **Route 21
SURF NORTH → Pallet Town (3,0) → Route 1 → Viridian (3,1)** (Route 21 is Seafoam-FREE open sea — the surf
travel handles it; NS5 close saw her surfing north up R21 fine, minor no_route wedges self-recover via roam).
Giovanni's gym is GROUND-type: Venusaur's Grass ×2 + Lapras's Surf ×2 = trivially winnable. **recon_giovanni.py
EXISTS** (parametrize with a GIOVANNI_STATE env like the others if the general beat_gym wedges on the
spin-tile floor). After badge 8: **VICTORY ROAD needs Surf + Strength — she has BOTH now** (recon_victory.py);
then **Elite Four** (recon_e4.py / recon_agatha.py) → Champion → **CREDITS ROLL** (write CREDITS as line 1 of
NIGHT_REPORT.md to stop the loop). The port pattern is mechanical: add `_resolve_state` + `<X>_STATE` env +
kit-sidecar loader (copy from recon_blaine.py) + utf-8 stdout; run from the prior kit fixture; bank forward.

## (done NS5) CINNABAR SEA ROAD + BLAINE (0x826) — see banked chain above.
### (historical) Surf DONE — 3 legs banked, 1 to go.
**NS5 BANKED (all e2e-verified, kit line, in states/workshop):** (1) `lapras_fielded_kit` — withdrew the
boxed Silph Lapras (PC deposit Ekans + withdraw Lapras). (2) `fuchsia_lapras_kit` — look-ahead forward-drive
Saffron→Route15 gatehouse→Fuchsia (GOAL_MAP=3,7, ~45s). (3) `safari_hms_kit` — recon_safari from Fuchsia:
Gold Teeth→Secret House→**HM03 Surf** + Warden→**HM04 Strength** (62s). (4) `surf_ready_kit` — recon_surf_teach:
**Surf→Lapras**, **Strength→Venusaur**; can_use surf/strength=True, all of Cut/Surf/Strength/Flash usable.
She is at **Fuchsia (3,7)@(33,32)**, party Venusaur L57 (Razor Leaf/Cut/Sleep Powder/Strength) + Lapras L25
(Surf/Body Slam/Confuse Ray/Perish Song) + bench (Spearow/Rattata/Abra/Drowzee). Water is now a road.

### ▶ FRONTIER (task 4, IN FLIGHT): CINNABAR SEA ROAD → BLAINE. Run recon_longrun from surf_ready_kit toward
`LONGRUN_GOAL_FLAG=0x826`. Billed road: Fuchsia(3,7)→**Route19(3,37) sea, west**→**Route20(3,38) via=pass,
THROUGH Seafoam Islands (boulder/current puzzle — general pass-through does NOT solve the interior; the
proven strike is recon_cinnabar.py, cinnabar_reach)**→Cinnabar(3,8)→Blaine gym. If the sea-road wedges at
Seafoam, port recon_cinnabar.py to run from surf_ready_kit (SAFARI_STATE-style env). BLAINE GYM = 6 quiz
doors (recon_blaine.py: Q1 YES/Q2 NO/Q3 NO/Q4 NO/Q5 YES/Q6 NO; Bill ambush after — press B not A; Blaine
(5,4) Arcanine L47 vs Venusaur L57+Sleep Powder+Strength). Blaine GymSpec + surf-aware travel already in
campaign.py; if the general walk wedges on the quiz interior, port recon_blaine.py. Bank blaine_done_kit
(badges=7, flag 0x826). See [[pokemon-nightshift4-badge6-sabrina-silph-hyperstall]].

### task 5 (autonomy, deferred): wire safari_strike + surf-gate arming into campaign so the Kira timeline
auto-fires the whole chain (withdraw→Safari→teach) when the billed road to Cinnabar hits the Route-19 water
without Surf. Tonight's legs used kit-parametrized recon strikes (LAPRAS_STATE/SAFARI_STATE/SURFTEACH_STATE
env) to bank distance; the autonomous registration is the remaining rope-laying.

## (superseded by NS5 legs above) NS4: the "no water mon" read was WRONG — she OWNS a boxed Lapras.
**CRITICAL CORRECTION (NS5 recon):** flag **0x246 (Lapras received) = TRUE** in sabrina_done_kit — the
Silph Co. gift **Lapras is already hers, sitting in the PC BOX** (party was full 6/6 during the Silph
strike so it auto-boxed). NS4's frontier ("no water mon — catch one") is WRONG: she doesn't need to catch
anything, she needs to **WITHDRAW the Lapras she already owns**. Confirmed party (sabrina_done_kit):
Venusaur L57 + Spearow L13/Rattata L8/Abra L10/Drowzee L13/Ekans L14 (all boxable weaklings). Bag confirmed
HM01 Cut(339)+HM05 Flash(343) but **NOT HM03 Surf(341)** — Surf still to be acquired. Money $21,786; 2 Poke
Balls + 1 Master Ball; 29 Super + 11 Hyper + 3 Revive.

**THE REAL BADGE-7 CHAIN (all legs PROVEN on the credits timeline — recon_safari/surf_teach/cinnabar/blaine
exist):** (1) **PC withdraw** — deposit a weakling, withdraw Lapras into party (Venusaur CAN'T learn Surf, so
the TEACH BRIDGE fails "no compatible party mon" without a fielded Lapras). This is Tier-2 #15 (PC/Box, ❌)
+ team-building #3 made load-bearing. recon_pcbox.py does DEPOSIT (proven); withdraw is the mirror menu.
(2) **Safari Zone strike** → HM03 Surf (Secret House man, face UP from (6,6)) + Gold Teeth → Warden → HM04
Strength. recon_safari.py = full proven interior state machine (pond-split tour chain: Center→EAST(43,15-17)
→Area1→Area2→Area3 West, Secret House door (12,7)). Port → safari_strike.py (run_strike(camp) like
silph_strike) + register in _questline_strike. (3) **Teach** Surf(57)→Lapras + Strength(70)→carrier
(recon_surf_teach/hm_teach TeachFlow). (4) **Cinnabar sea road** (recon_cinnabar: water-as-road, Seafoam
interior crossing) → **Blaine gym** (recon_blaine: 6 quiz doors Q1 YES/Q2 NO/Q3 NO/Q4 NO/Q5 YES/Q6 NO,
Bill ambush after — press B) → **BADGE 7 = flag 0x826**. Blaine GymSpec + surf-aware travel already in
campaign.py; Blaine quiz interior may need a blaine_strike if the general walk wedges (campaign.py:404).
See [[pokemon-nightshift4-badge6-sabrina-silph-hyperstall]].

## ✅ NS4 BANKED: SILPH CO. SOLVED UNAIDED (Gary type-wall broken) → frontier = SABRINA BADGE (0x825)
**VERIFIED e2e, fully autonomous from a clean koga_done_kit (NO injected items):** HYPER-STALL bought
15 Hyper + 5 Revive at Saffron Mart ($30k→…) → silph_strike → GARY WON 5-2 (out-heals Charizard's Fire)
→ Lapras → Giovanni → SAFFRON FREED (0x3E) in 226s. Banked → promoted **`silph_done_kit.state`**
(workshop; Venusaur L56, Saffron freed, ~$16k, leftover ~11 Hyper + 3 Revive). Commits: 1e9e43c (diagnosis
+ battle guard), db84e10 (Saffron Mart + stock_hyper_potions kit). See [[pokemon-nightshift3-silph-gary-skip]].

### ✅ BADGE 6 (Sabrina/Marsh 0x825) DONE — beaten unaided from silph_done_kit in the same NS4 run
Sabrina's Alakazam fell to L57 Venusaur's Razor Leaf with ZERO items used (the leftover Hyper/Revive
kit wasn't even needed). Banked `sabrina_done_kit.state` (badges=6, Venusaur L57, $21.8k). The gym's
teleport-pad interior + junior-maze navigated fine (residual: ~1 junior deferred un-engageable, not fatal).

### 🔨 FRONTIER — BADGE 7 (Blaine / Cinnabar, Volcano 0x826): GATED ON **SURF** (team-building gap)
Run `LONGRUN_GOAL_FLAG=0x826 LONGRUN_BATTLE_LOG=1 recon_longrun sabrina_done_kit.state 40`. CONFIRMED
BLOCKER (NS4 look-ahead probe): she wedges routing toward Cinnabar at **map (3,33)@(0,12)** ("no clean
path… genuine wall/zone gap") — Cinnabar is ACROSS THE SEA and her party (Venusaur + Spearow/Rattata/Abra/
Drowzee/Ekans) has **NO Surf-capable mon**. This is the #3 team-building soul-debt made load-bearing: to
reach Blaine she needs a **water/Surf teammate** (catch one — e.g. a Tentacool/Poliwag/Krabby off a
Surf-adjacent route or the Safari) AND **HM03 Surf**. CONFIRMED via bag dump: the fixture holds HM01 Cut
(339) + HM05 Flash (343) but **NOT HM03 Surf (341)** — so HM03 must be ACQUIRED (FRLG: it's the reward in
the **Safari Zone Secret House**, a Fuchsia-area errand — she's already been to Fuchsia). The travel
circuit-breaker handled the wedge gracefully (returned to roam LOUD, no freeze-spin — watch-safe), so the
canonical fixture is NOT wedged. NEXT SHIFT (a big, multi-step team-building stretch — start fresh with
full context): (1) Safari Zone → grab HM03 Surf (Secret House); (2) catch + level a water mon and teach it
Surf (hm_teach.py exists; the #3 team-building behaviour deferred all climb is now MANDATORY); (3) sea route to
Cinnabar → Blaine (Fire gym — Venusaur's Grass is ×0.5 into Fire, so a Water/Rock teammate doubles as the
Blaine answer). SABRINA POTION-STALL not needed (she won without items); the Saffron Mart + reusable
`stock_hyper_potions` remain available if a future Fire/attrition wall wants Hyper Potions.

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
