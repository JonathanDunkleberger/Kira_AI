# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ⛔ NS12 WALL (2026-07-10): bench_grind_kit lineage's Venusaur is TOO WEAK for VR/Gary (no EQ) + the EQ teach is BROKEN on this save. Credits did NOT roll. TWO clean paths below — PATH B recommended.
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

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = bench-grind (Kadabra L19→42) at
Route 18; the whole tail+VR+E4-rooms-1-2 is proven; only Agatha+ (a level gap) is left. Pop-in =
`python pokemon_agent/watch.py`.
