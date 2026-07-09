# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 14 entry)

## SHIFT 14 HEAD (read FIRST — supersedes everything below)

**THE GARY WALL IS NOW FULLY DIAGNOSED WITH BATTLE-TURN EVIDENCE (shift 14).** The shift-12 DIRECTED
nav still reliably climbs the S.S. Anne 1F->2F to the rival Gary (nav is SOLID). Two distinct problems
surfaced, one FIXED this shift, one handed off:

### PROBLEM A — the post-loss ping-pong (FIXED + COMMITTED this shift)
After losing to Gary she whites out to **Cerulean's Pokémon Center** (two towns back — her last-healed
Center; she never heals at Vermilion before boarding). Root cause: the DIRECTED interior nav sets
`_ql_inside_target=True` for the ship, and a Gary-loss whiteout left that flag **stale-True**, which
SUPPRESSED the blackout/stranded-in-building recovery (both tick-top branches require
`not _ql_inside_target`). So she sat INSIDE the Cerulean Center where `head_to_gym` can't route out,
re-picking `head_to_gym` into a `no_route` wedge **~11 ticks in a row** (an unwatchable spin inside a
building) before the deep-wedge ring bailed her.
- **FIX (campaign.py, roam-loop post-action ~line 8368):** on `out in ("battle_loss","blackout")` clear
  `_ql_inside_target=False` — a whiteout warps her to a Center, so deliberate interiority is void; the
  existing exit-to-overworld recovery then fires next tick. Safe + general: a loss is never "still inside
  the target"; a WIN never hits this branch (ship exploration preserved). Reproduced in BOTH shift-14 runs.
- **VERIFY:** `.venv/Scripts/python.exe pokemon_agent/recon_longrun.py ss_ticket.state 12`
  -> `logs/debug/shift14_verify.log` (UTF-8). After the `RIVAL beat #7 vs Gary (won=False)` line, grep
  for `whiteout after a loss -> clearing _ql_inside_target` and confirm she EXITS to the overworld
  (`BLACKOUT/STRANDED ... exiting`) instead of ~11x `PICK OUT: head_to_gym` inside `a building in Cerulean`.

### PROBLEM B — Gary is a genuine TEAM-STRENGTH wall (HANDED OFF — careful battle/team work)
With `LONGRUN_BATTLE_LOG=1` I captured the full Gary battle turns. **She loses because ivysaur L30's
moveset is ONE resisted damaging move (Vine Whip, grass) + TWO powders (Sleep/Poison Powder).** It burns
Vine Whip PP on Pidgeotto+Kadabra, then hits **PP FAMINE** — only status moves left vs Charmeleon +
Raticate. It sleep-locks + poisons Charmeleon to death and burns 2 Potions, but **can't damage Raticate**;
the frail L14/L15 bench switches in, chips ~nothing, faints -> wipe. The battle brain plays *reasonably*
(sleep-lock armed, poison chip, potions, PP-famine switch) — the loss is **team strength**, not a bug.

**KEY NEW INSIGHT:** the shift-12 DIRECTED nav B-lines her straight to the Gary warp, **skipping ALL ~8-10
S.S. Anne cabin trainers (L16-18)** — she fights ZERO ship trainers before Gary (confirmed: 0 battle-menu
lines between boarding and Gary). Those cabin trainers ARE the intended level-up: a human clears them
(bench -> L18-20 + ivysaur gains PP-fresh levels), THEN beats Gary. She skips them and arrives underleveled
every attempt.

**THE SUCCESSOR'S JOB — make her BEAT Gary (fresh full budget; touches battle/team core — do carefully,
verify via look-ahead, do NOT rush/regress the verified climb).** Best routes, in order of cleanliness:
1. **Fight the ship's cabin trainers before the Gary warp when underleveled** (nav/questline, NOT
   battle-brain — within mandate). The intended, human, on-ship (no strand), near-level (great XP) level-up.
   RISK: re-introducing the shift-8-12 cabin-tour nav wedge — make it FAIL-OPEN (fall back to the DIRECTED
   Gary warp = status quo). This is the single highest-value fix (levels the WHOLE team in one pass).
2. **Give ivysaur a real damaging kit** — it's missing Razor Leaf (strong STAB, 25 PP) and a neutral move
   (Take Down/Body Slam) and Leech Seed. Its 1-attack/2-powder set is WHY PP-famine loses. Check the
   move-learn/keep-strongest logic (🔨 #8) — she's keeping TWO redundant powders over damage/sustain.
3. **Catch a Diglett** (Diglett's Cave, west end reachable from Route 2 / east from Vermilion Route 11) —
   Ground type = super-effective vs Raticate/normal chip AND the **Lt. Surge answer** (Surge is the NEXT
   wall, electric L21-24; Ground is immune). One catch solves Gary-bench AND Surge. DEX-doctrine cheap catch.
   NB: she already catches a **Mankey L12** (Fighting) on Route 4 in the recovery — a partial counter, but
   Mankey is 2x-weak to Kadabra(psychic)+Pidgeotto(flying); a Diglett is the cleaner pick.

**SECONDARY (glacial-grind, lower priority):** post-loss she grinds the bench on **Route 4 east (L3-6
wilds, Center unreachable across the Mt-Moon ledge)** via participation-XP — a L12 Mankey there gains ~0
XP and never approaches L20 (glacial). The shift-13 heal-excursion cap only catches *excursion*-thrash, not
this *potion-sustained-no-gain* spin. If she whited out to **Vermilion** instead (heal at Vermilion's
Center before boarding), recovery-grind would be Vermilion-side (Route 6/11, near-level wilds, Center
reachable) — far better. Consider: heal at the adventuring town's Center before a hard local wall, so a
loss respawns LOCAL not two towns back.

## KEY FACTS / TOOLS

- **venv python:** `G:/JonnyD/NeuroAI_Bot/.venv/Scripts/python.exe` (NO bare `python` on PATH).
- **Look-ahead oracle:** `recon_longrun.py <state> <min>`. Set env **`LONGRUN_BATTLE_LOG=1`** to capture
  in-battle turn-by-turn (`[engine]` lines) — REQUIRED to diagnose battle quality (default OFF = silent).
  recon_longrun logs are **UTF-8** (read directly). Persistence STAGE-redirected — canonical never touched.
- **ss_ticket.state** (workshop) boots in Bill's house; re-walks Bill->Cerulean->Vermilion->boards ship
  (~4-5 min) each run. Party at ship: ivysaur L29, rattata L14, spearow L15.
- **Gary (S.S. Anne rival):** charmeleon+kadabra+pidgeotto+raticate (~L16-20). Blocks the (30,2) captain
  warp on 2F (map (1,6)) at tile (30,5). MUST beat him to pass -> captain -> HM01 Cut -> Vermilion gym tree.
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = 2-PID shim — never taskkill your own run.
- **states/campaign = SHERPA CANONICAL (Champion, untouchable). states/workshop = scratch.** Commit per
  fix (`git add pokemon_agent/ gamedata/`); VERIFIED not asserted.

## GUARDRAILS (non-negotiable)
- Shift-14's edit is pure nav/loop (clear a stale interiority flag on whiteout) — additive, fail-safe,
  mode-side. Core Kira identity/voice/oracle/memory/vision sacred + OFF-LIMITS.
- The Gary team/battle fix TOUCHES BATTLE + TEAM-BUILD + MOVE-LEARN core — do it carefully in a fresh
  session, verified via the look-ahead; do NOT rush a behavioral patch to the battle brain unattended.
- AUDIO END STATE = ON (`POKEMON_GAME_AUDIO=1`); audio-off is only the committed floor.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS
(a) clean bedroom->credits with audio ON; OR (b) ~80-85% context -> clean handoff (rule 11) /
two-consecutive-no-progress brake; OR (c) balance exhausted.

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched). The look-ahead crosses
bedroom -> Brock -> Misty -> Cerulean -> Bill -> Vermilion -> boards the S.S. Anne -> climbs to the 2F
captain-approach and the rival Gary — a TEAM-STRENGTH wall (underleveled: PP-starved ivysaur + frail
bench, cabin trainers skipped). Shift 14 fixed the post-loss ping-pong (stale interiority flag);
beating Gary = the successor's careful team-building job. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
