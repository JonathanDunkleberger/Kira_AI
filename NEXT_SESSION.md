# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #5 close)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 NIGHT SHIFT #5 block first. Last shift's arc:

👻 **THE VILEPLUME WALL WAS A GHOST** — no Route 8 trainer ever beat her. The tick-top
"indoors = blackout" heuristic fired on the questline's legit mid-route hut tick and
note_blackout laundered the STALE last_foe (Erika's gym Vileplume, still in the save's RAM)
into a phantom wall. Fixed (15b63f6): blackout now demands BATTLE EVIDENCE. Lesson: verify
walls with a battle trace + money-halving check before believing the loss record.

✅ **BANKED + PROMOTED: tm43_secret_power** — canonical = Celadon (3,6)@(11,15), party healed,
badges 4, $13,378. Venusaur + Fearow both know SECRET POWER (bought at the Dept 2F TM clerk,
taught via TeachFlow overrides). recon_tm_errand.py is the reusable TM buy+teach vehicle.
General kills riding every future run: directional-door + mat-row enter_warp fallbacks (huts,
dept stairs, exit mats), shop TRUE-INDEX nav (row+scroll — the row byte alone LIES on deep
lists), teach START open-verify, tm_compatible() ROM truth.

🏰 **HIDEOUT 80% CRACKED (fb51898, recon_hideout.py — deterministic, ~40s from canonical):**
poster grunt → poster → stairs → B1F→B2F→B3F → **spin maze crossed** (the slide crosser is
BUILT: 0x54-0x57 redirect, 0x58/walls stop, plain floor does NOT stop momentum, wall-stopped
spinner resumes its own dir on any press, NPCs block, replan-after-battle) → B4F → **LIFT KEY
IN BAG** → back up to B2F.

⛔ **FIRST MOVE: finish the hideout.** The boss corridor (grunts (16,14)/(19,14) → GIOVANNI
(19,4) → SCOPE ball (20,5)) is elevator-only. The ride is CODED in recon_hideout.py (panel bg
(0,2) in the elevator map, floor multichoice, self-correcting landing) but the B2F glide graph
finds NO route from (15,8)/(21,2) to the elevator doors (28-29,16). Do: re-run
`python pokemon_agent/recon_hideout.py` (reproduces to B2F in ~30s), then at the abort point
dump the B2F behavior grid (`camp._tile_behavior`, x=12-30 y=2-20) + a `b.frame_rgb()` frame —
derive the elevator entry. SUSPECT: the glide-only BFS can't represent plain WALK corridors
between rest points — add plain single-step edges to the BFS alongside glide edges (a walk
step onto plain floor is always a valid 1-tile edge; only spinner-touching steps glide).
Complete: ride → beat 2 door grunts → Giovanni → scope ball → verify ('item',359) in Key
Items → banked_SCOPE → `python pokemon_agent/promote_bank.py G:/temp/longrun/banked_SCOPE silph_scope`.

THEN: relaunch the flute longrun — `LONGRUN_GOAL_FLAG=0x23D LONGRUN_BATTLE_LOG=1
POKEMON_SLEEP_LOCK=1 POKEMON_CATCH_JUDGMENT=1 POKEMON_PROACTIVE_BENCH=0 python
pokemon_agent/recon_longrun.py kira_campaign.state 70 > logs/longrun/flute_runN.log 2>&1` —
it drives Lavender → Tower (Scope unmasks the ghost; channelers are normal fights; Fuji top
floor) → Poké Flute (0x23D) → promote → wake the Route 12 Snorlax → Routes 12-15 → Fuchsia →
Koga. **Koga has no GYMS row yet — add it (disasm FuchsiaCity_Gym map.json) before the
badge-5 strike.**

KNOWN LONGRUN GAPS (run-12 postmortem, fix when they bite): the questline "north-most warp"
heuristic tours the WRONG building repeatedly (Celadon mansion ↔ city loop = the run-12
stall) — needs entered-door memory; destination interactions need KB interior RITUALS (the
grunt/poster class — recon_hideout is the scripted spec). The Tower may need the same ritual
treatment; if the tour stalls there, strike it scripted the same way.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only (battle trace + money check before believing any wall),
NEXT_SESSION.md at close. GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **Promoted:** tm43_secret_power (Secret Power on Venusaur + Fearow — the moveset-famine cure).
- **The night's real story:** the Vileplume "wall" that shaped shifts 4-5's plans was a GHOST
  (stale RAM + an over-eager blackout heuristic). Killed at the definition; the evidence gate
  guards every future run.
- **Hideout:** everything but the final elevator hop is cracked and deterministic — poster
  gate, the spin-maze slide crosser (a permanent asset: Victory Road/ice floors ride it too),
  Lift Key in bag.
- **Owed/honest:** Silph Scope not yet in bag (elevator approach = the one open blocker);
  Koga GYMS row missing; wrong-building tour loop in the questline heuristic; moveset-gap
  auto-arbitration not wired general (the errand script is the vehicle + playbook); Venusaur
  still named AAAAAAAAAA (Name Rater in Lavender — soul call).

WATCH STATUS: canonical bank is CLEAN; she is at Celadon (11,15), badge 4, healed, Secret
Power learned, outside the Dept Store; pop-in = `python pokemon_agent/play_live.py --resume --free-roam`.
