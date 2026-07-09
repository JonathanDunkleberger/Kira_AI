# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 16 entry)

## SHIFT 15 HEAD (read FIRST — supersedes everything below)

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

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched). The look-ahead now crosses
bedroom → Brock → Misty → Cerulean → Bill → Vermilion → boards the S.S. Anne → **BEATS the rival Gary**
(the 4-shift wall, via the move-value fix so ivysaur keeps Razor Leaf) → gets HM01 Cut → reaches Vermilion
and knocks on the Lt. Surge gym. Frontier = enter the Surge gym (Cut-tree / trash-can puzzle) + beat Surge
(catch a Diglett or level the bench). Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
