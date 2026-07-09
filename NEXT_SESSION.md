# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 13 entry)

## SHIFT 13 HEAD (read FIRST — supersedes everything below)

**THE SHIFT-12 DIRECTED-NAV FIX WORKS — VERIFIED.** `captain_fix.log` (UTF-8) shows she climbs the S.S.
Anne interior on the KB chain: `QUESTLINE DIRECTED (1,4)->(1,5) via (32,14)` → `(1,5)->(1,6) via (3,8)`
→ reaches **2F** and pathfinds toward the captain warp `(30,2)`. **The 4-shift Route-5/ship-interior
wedge (shifts 8-12) is DEAD.** She now reliably reaches the top of the 2F stairs.

**THE TRUE NEW FRONTIER = the RIVAL GARY fight (a TEAM-STRENGTH / TYPE wall, NOT a nav bug).** On the 2F
approach to the captain, a TRAINER blocks the warp tile `(30,5)` — that is **Gary**
(`trainer:the S.S. Anne:charmeleon+kadabra+pidgeotto+raticate`). She fights (`_fight_blocker` →
`battle_runner`) and **LOSES** → `RIVAL beat #7 vs Gary (won=False)`. This is a GENUINE bad matchup, not a
bug: her lone carry is **ivysaur (grass/poison) L29**, and **3 of Gary's 4 mons hit it super-effectively
AND resist its grass STAB** — Charmeleon (fire 2×), Pidgeotto (flying 2×), Kadabra (psychic 2× vs poison).
Solo ivysaur gets worn down; the frail **L14 rattata / L15 spearow bench is swept on entry** → party wipe.
This is the known **Tier-1 #3 team-building (❌) + #5 battle-competence/switching (🔨) debt**, surfaced.

**THE RECOVERY LIVELOCK (shift-13 FIX APPLIED + COMMITTED).** After the Gary loss she correctly plans
"level the weak ones", but the recovery **livelocked**: she ace-grinds on **Route 4's east grass**, whose
OWN Center `(12,5)` is UNREACHABLE (split from the east grass by the Mt-Moon ledge). Every few battles →
low → an EXPENSIVE cross-city HEAL-EXCURSION → back → grind → low → … The `captain_fix` run logged **22
heal-excursions + 582 travel legs + ~0 net XP** (ivysaur L29 vs L3-6 wilds earns nothing; target L31 never
approached) before the deep-wedge ring finally reverted — an unwatchable ~1000-game-second spin. The stall
watchdog never fired (she makes *spatial* micro-progress while looping *strategically*).
- **FIX (campaign.py, `grind()` + `_heal_excursion` + new `GRIND_HEAL_EXCURSION_CAP=4`):** bail a grind
  that racks up **≥4 cross-city heal-excursions with ZERO level gain** — mark the map grind-dead, return
  `no_safe_grass` (→ prep stand-down / GRIND-WEAK re-pick). Distinguishes the useless ace-thrash (0
  progress) from a legit deep grind (`grind_pre_brock` earns levels per excursion → cap never trips).
  Fail-open + general (every split-route heal-thrash, not just Route 4). Mode-side, additive.

**VERIFY THE FIX:** `.venv/Scripts/python.exe pokemon_agent/recon_longrun.py ss_ticket.state 20` →
`logs/debug/grindcap_fix.log` (**UTF-8** — read directly). Grep:
- `QUESTLINE DIRECTED .* via warp \(3, 8\)` + `via warp \(30, 2\)` (still climbs to 2F — nav NOT broken),
- `RIVAL beat #7 vs Gary \(won=False` (reaches + fights Gary — the wall),
- `GRIND: .* cross-city heal-excursions .* split-route heal-thrash` (**the fix fired** — grind bailed
  instead of the 22-excursion thrash),
- `PREP STAND-DOWN` / `strategically_stuck` / a 2nd `RIVAL beat #7` (she re-decides / gets loss #2 → the
  strategic-stuck floor can now fire at count=2-not-stronger, breaking the die→recharge→die loop).

**THE REAL WORK FOR THE SUCCESSOR — make her BEAT Gary (a fresh session's full budget; do NOT rush it
unattended — it touches battle/team-building core).** Three routes, pick/combine:
1. **Field + level the BENCH (the correct systemic fix).** rattata + spearow are **NORMAL-type = neutral
   vs Gary's ENTIRE team** (no 2× weakness) — leveled bodies (L18-20) let her switch/chip so ivysaur
   cleans up. The machinery exists (`grind_weak_members`, `_prep_team_target`, `POKEMON_SOLO_WEAK_GRIND`
   /`GRIND_SWITCH`) but is 🔨 — verify it actually fields+levels the weak (the `captain_fix` run NEVER
   logged a `GRIND-WEAK` line; she only ever ace-grinds because forward-drive keeps picking head_to_gym
   and the deliberate grind path is starved). **This is the endearing half of #3 (she CHOOSES to build a
   real squad) — the SOUL-DEBT, not just mechanics.**
2. **Smarter in-battle play vs a TYPE wall.** `battle_runner` likely spams resisted grass STAB (Razor
   Leaf) into Charmeleon/Pidgeotto and eats 2× back. Ivysaur L29 CAN win with neutral moves (Take
   Down/Body Slam) + **Sleep Powder** on the fire/psychic threats + **Leech Seed** sustain. Check whether
   the battle brain uses status/neutral moves vs a resisted matchup (the 🔨 #5 gap).
3. **Catch a counter** en route (a normal/rock mon; a Diglett from Diglett's Cave west of Vermilion is the
   Surge answer too) — the DEX-doctrine cheap-catch is soul-positive.

**IF she gets past Gary → captain (HM01 Cut) → teach Cut → Cut the Vermilion gym tree → Lt. Surge (badge
3, electric L21-24).** Surge ALSO needs a ground type or a real grind (thin bench L14-16 vs L21-24) — a
separate under-level finding, not a nav bug; the Diglett's Cave answer covers both Gary-bench and Surge.

---

## SHIFT 12 HEAD (superseded — DIRECTED nav VERIFIED working; frontier moved to the Gary fight)

**THE DECOY-HOUSE FIX (shift 11) + DIRECTED INTERIOR NAV (shift 12, 87579d5) both VERIFIED.** She crosses
Cerulean → Route 5 → Underground Path → Route 6 → Vermilion, boards the S.S. Anne, and now (shift-12 fix)
climbs 1F→2F on the KB chain in `gamedata/frlg_gates.json interior_routes` + `campaign._questline_interior_route`.
The blind GO-DEEPER cabin-tour never climbed to 2F (the canonical world model pre-loaded 2F as 'vis' so the
stairs sorted behind every new cabin); the DIRECTED-NAV block at the top of GO-DEEPER takes the KB warp
chain instead. **Verified: she reaches the 2F captain-approach and the rival Gary — the new wall.**

## KEY FACTS / TOOLS

- **venv python:** `G:/JonnyD/NeuroAI_Bot/.venv/Scripts/python.exe` (there is NO bare `python` on PATH).
- **captain_fix.log / grindcap_fix.log are UTF-8** (recon_longrun writes UTF-8 — read directly, do NOT
  iconv). Only PowerShell `*>` redirect logs are UTF-16LE (`iconv -f UTF-16LE -t UTF-8`).
- **Look-ahead oracle** (default verification): `recon_longrun.py <state> <min>`; non-FRESH boot =
  free_roam from a named state, loads the CANONICAL world model (`C.WORLD_JSON`). Persistence is
  STAGE-redirected (`$TEMP/longrun/stage`) — canonical is NEVER clobbered.
- **ss_ticket.state** (workshop) boots INSIDE Bill's house (30,0)@(7,7) with the ticket + badge 2; the run
  re-walks Bill → Cerulean → … → Vermilion → boards the ship (~4-5 min) each iteration (no ship state is
  banked; unavoidable from ss_ticket). Party at the ship: ivysaur L29, rattata L14, spearow L15 (+ catches).
- **hm01_bank_20260706_203730/world_model.json** = the canonical post-Cut bank; holds the full ship warp
  graph. 2F map `(1,6)` warp `(30,2) -> (1,11)` = the captain's cabin (dead-end). Gary blocks the
  `(30,5)/(30,6)` approach to `(30,2)` — you MUST beat him to pass.
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = a 2-PID shim — never taskkill your own run.
- **states/campaign = SHERPA CANONICAL (untouchable, Champion save Jul 8 20:46). states/workshop = scratch.**
  Commit per fix (`git add pokemon_agent/ gamedata/`); VERIFIED not asserted.

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS / MODE-SIDE FIXES ONLY.** Core Kira identity / voice / oracle / memory / vision are
  sacred + OFF-LIMITS. Shift-13's edit is pure grind-loop watchdog (split-route heal-thrash bail) —
  additive, fail-open, the deep-grind path (`grind_pre_brock`) unchanged (cap trips only on 0-level-gain).
- **The Gary team-building/battle fix TOUCHES BATTLE CORE — do it carefully, in a fresh session, verified
  via the look-ahead; do NOT rush a behavioral patch to the battle brain unattended** (regressing the
  verified climb / core battle competence is the failure mode Jonny fears most).
- **AUDIO END STATE = ON** (`POKEMON_GAME_AUDIO=1`); audio-off is only the committed floor/fallback.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS

(a) clean bedroom→credits with audio ON demonstrated; OR (b) ~80-85% context → clean handoff (rule 11)
/ two-consecutive-no-progress brake; OR (c) balance exhausted.

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched). The look-ahead crosses
bedroom → Brock → Misty → Cerulean → Nugget Bridge → Bill → Vermilion → boards the S.S. Anne → climbs to
the 2F captain-approach and the rival Gary — the new wall (a team-strength/type fight, not a nav bug).
Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
