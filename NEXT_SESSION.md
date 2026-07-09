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
