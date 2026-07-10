# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ✅ BADGE 4 (Erika / RAINBOW 0x823) — DONE + VERIFIED this shift (17)
The coverage-teach-for-depth fix (5c07f01) is CONFIRMED working: from the erika_retry_kit fast fixture she
dispatches coverage-teach (has_type_answer=False), counts neutral_dmg=1<2, teaches Venusaur CUT (HM01, x1
neutral, forgets Vine Whip), fields Cut vs Erika's victreebel/tangela/vileplume, and WINS badge 4 on the
FIRST engagement — no PP-famine blackout, no retry ratchet. Banked -> states/workshop/erika_done_kit.state
(+ sidecars). This was the badge-4 PP-famine wall, now dead.

## FRONTIER: BADGE 5 (Koga / Fuchsia — SOUL 0x824) — questline chain REPLAYS autonomously (shift 17)

★ THE BADGE-5 APPROACH IS A LONG QUESTLINE CHAIN (not a routing bug). Route 12 south out of Lavender is
BLOCKED by the sleeping Snorlax → needs the POKÉ FLUTE → rescue Mr. Fuji atop POKÉMON TOWER → needs the
SILPH SCOPE to pass the ghost Marowak on 6F → Scope held by Giovanni in the ROCKET HIDEOUT under Celadon's
Game Corner. The questline machinery KNOWS this whole chain (anchor-first routing) and REPLAYS it e2e:
   - QUESTLINE STRIKE Rocket Hideout -> got_scope ✓ (Silph Scope obtained)
   - QUESTLINE STRIKE Pokémon Tower: climbed floors 5→6→7F, beat Gary on 2F (grudge 5W-2L), rescued Fuji
     -> got_flute ✓ (Poké Flute item+flag, key_item 350)
   - QUESTLINE STRIKE Route 12 Snorlax (Poké Flute wake) — in progress at last check
   - THEN: Routes 12→13→14→15 → Fuchsia → Koga gym.
Venusaur solo-carries and levels through it (L44→L46+). The chain is fully billed in
gamedata/frlg_gates.json roads["Fuchsia City"] (Celadon→R7→R8→Lavender→R12 south→R13/14/15→Fuchsia).

★ VERIFY/CLIMB COMMAND (badge 5 from the fresh badge-4 bank; give it the full 40 — Hideout+Tower+Snorlax+
4 routes + Koga is a LOT of content):
   `POKEMON_TEAM_PLANNER=1 LONGRUN_BATTLE_LOG=1 LONGRUN_GOAL_FLAG=0x824 .venv/Scripts/python.exe -u
   pokemon_agent/recon_longrun.py erika_done_kit.state 40` → /g/temp/s17_koga.log (0x824 = FLAG_BADGE_SOUL)
   WATCH: got_scope → got_flute → got_snorlax (flag 0x253) → Fuchsia City → GYM-PREP [Koga] → GYM: won → 0x824.
   grep: `grep -nE "got_scope|got_flute|got_snorlax|Snorlax|Fuchsia City|GYM-PREP \[Koga|engaging Koga|GYM: won|GYM: lost|OUTCOME" LOG | grep -v ctx=`

★ WATCH FOR at Koga: memory says Koga = a team/movepool wall; the original climb used a NUKE-SLEEP opener
(Sleep Powder → nuke). Venusaur has Sleep Powder + Razor Leaf + Cut. If she loses Koga: the coverage-teach
now also arms depth (poison gym; psychic/ground/psychic coverage). A LOSS during the tower (drowzee
whiteout-backstop) set an active_wall + UNDERLEVEL-PREP (team floor < L27) — watch it doesn't divert her
into an endless grind; her bench is L8-14 (solo-carry, the classic team-building debt).

★ ON GOAL (soul 0x824 set): bank /g/temp/longrun/banked_GOAL → promote to
states/workshop/koga_done_kit.state (+ sidecars). Frontier advances to BADGE 6 (Sabrina/Saffron — needs
the Tea to pass Saffron's gatehouses; the Tea gate was armed early, from the Celadon dept-store lady).

★ RESIDUAL (fix AFTER badge 5): Erika junior-engagement spin — flower-tile layout leaves ~4 juniors
un-engageable; she tries each 4× then burns the 14-clear-round cap before the leader (slow/unwatchable).
Fix: remember deferred un-engageable set across clear-rounds, skip to leader once only they remain.

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — recon_longrun auto-reaps predecessor via
longrun.pid). arg2 = max_minutes. resolve_state needs a BASENAME in workshop/ (e.g. erika_done_kit.state),
NOT a path. Decision-counter FREEZES during a gym/dungeon strike (watch growing log lines). Bag decode:
TM_N=288+N, HM_N=338+N. Coverage-teach = campaign.py:3310 dispatch / :3326 _teach_gym_coverage.
recon_partydump.py / recon_bagdump.py take a full path. FAST-FIXTURE trick: a post-blackout STALL bank
near a gym city makes a minutes-long verify instead of a 40-min full chain.

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = BADGE 5 Koga (Scope→Flute→
Snorlax questline replaying autonomously from the fresh badge-4 bank; heading Route 12→Fuchsia). Pop-in =
`python pokemon_agent/watch.py`.
