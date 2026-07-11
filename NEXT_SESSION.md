# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ NS14 (2026-07-10): OFFENSIVE-UPGRADE SWITCH FIX BREAKS THE LANCE WALL — reached the CHAMPION (room 5) for the first time. Committed (ce5e391). Overnight E4 self-grind loop RUNNING to converge; new wall = bench too FRAIL + Gary's Charizard.
**AT WAKE — CHECK THE OVERNIGHT E4 SELF-GRIND LOOP FIRST (it may have rolled credits):**
`ns14_e4_loop.sh` is RUNNING (`ns14_e4_loop_status.txt` + per-lap logs `ns14_e4_loop.log.N`). It re-launches
`recon_e4` from `indigo_reach_g` (lap 1) then from the leveled `banked_E4` (lap ≥2), self-grinding the bench off
L54-63 E4 foes (~10x Route 18 XP), until credits or ~8.5h. **CHECK:** `cat G:/temp/longrun/ns14_e4_loop_status.txt`.
- **If it says "HALL OF FAME"/"CREDITS DETECTED", or `banked_CREDITS` is re-dated TODAY** (check mtime — the old
  one is STALE 2026-07-07) → **WRITE `CREDITS` as LINE 1 of NIGHT_REPORT.md** (stops the loop) + full survey.
  Promote banked_CREDITS to canonical only per the two-timeline law.
- **If it's still looping without credits:** read the latest `ns14_e4_loop.log.N` — grep `room #` (furthest) and
  the bench levels (`revive-check` lines dump `party sp/hp/lv`). If Kadabra/Lapras have climbed toward ~L48 but
  it's not converging, promote the leveled `banked_E4` → `indigo_reach_g` and keep looping, OR do the LAPRAS-slot0
  reorder (below) so Lapras's Surf 2x is fielded actively vs Gary's Charizard.

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
