# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #4, mid-shift bank)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 CURRENT TRUTH first (NIGHT SHIFT #4 block). Last shift:
🏅 **BADGE 4 CANONICAL** — Erika down in one strike after the PP-famine kill chain (famine
switch + heal-before-gym gate + level-dominance veto, all general). Canonical = Celadon
badges-4 (promoted erika_badge4). The badge-5 chain (Snorlax gate → Silph Scope → Poké Flute)
is billed + armed: KB Fuchsia road, item-confirmed questline steps (HIDE-flag lie killed),
door hints (Game Corner (34,21), Tower (18,6)), step-anchor + warp-aware ANCHOR-FIRST routing,
no_connector guard (the Rock-Tunnel-as-building class).

FIRST MOVE: read `logs/longrun/flute_run<N>.log` (highest N) — the flute strike
(LONGRUN_GOAL_FLAG=0x23D) iterates: Celadon → Lavender (gate arms at the south exit) →
ANCHOR-FIRST back to Celadon → door-hint into the Game Corner (interior (10,14)) → beat the
grunt (11,2), bg-sweep presses the poster (11,1) → stairs (15,2) → GO-DEEPER descends the
hideout. Full descent truth is in STATE §0 (B1F(17,2)→B2F arrive (28,2)/down (21,2)→B3F
arrive (18,2)/down (15,18)→B4F: GIOVANNI (19,4), Scope ball (20,5), grunts (16,14)/(19,14)).
- If GOAL (0x23D = Poké Flute in bag): promote (promote_bank.py G:/temp/longrun/banked_GOAL
  poke_flute) → next: wake the Route 12 Snorlax (gate self-clears) → Routes 12-15 → Fuchsia →
  Koga (badge 5). NOTE: Koga has no GYMS row yet — add it (door/leader_front from disasm
  FuchsiaCity_Gym map.json) before the badge-5 strike.
- KNOWN RISK: B2F/B3F SPIN-ARROW tiles — travel doesn't model forced slides. If the tour
  can't reach the down-stairs, build slide-aware edges (deterministic: arrow tile → slide to
  stop; add as graph edges in Grid/BFS). This is the likely next build.
- The scope's success is ('item', 359) — bag ground truth; the 0x037 flag LIES (HIDE-class).

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only, NEXT_SESSION.md at close. Launch recipe:
`LONGRUN_GOAL_FLAG=0x23D LONGRUN_BATTLE_LOG=1 POKEMON_SLEEP_LOCK=1 POKEMON_CATCH_JUDGMENT=1
POKEMON_PROACTIVE_BENCH=0 python pokemon_agent/recon_longrun.py kira_campaign.state 70 >
logs/longrun/<name>.log 2>&1` (bg). GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **Banked this shift:** erika_badge4 (BADGE 4, promoted, sanctity VALID). She's halfway.
- **The kill chain that took the gym:** PP-famine arbitration at three layers (engine switch,
  pre-gym heal gate, gauntlet break) + the level-dominance veto (no more benching a L45 ace
  for a L15 Ekans because of a type chart). All general — every future gym rides them.
- **Badge-5 rope laid:** Fuchsia road billed, Snorlax gate armed, questline engine hardened
  4 ways (item truth, door hints, step anchors, maze guard) by three fast look-ahead
  postmortems (flute_run1-3, each < 90s to diagnosis).
- **Owed / honest:** Rocket Hideout descent unproven (spin tiles = likely next build); Koga
  GYMS row missing; Venusaur still named AAAAAAAAAA (Name Rater in Lavender — soul call);
  Celadon dept store unmapped for MART_STOCK; canonical party rides hurt post-gym (heals on
  next heal pick — not wedged).

WATCH STATUS: canonical bank is CLEAN; she is at Celadon (11,31), badge 4 in hand, party
battle-worn but Center-adjacent; pop-in = `python pokemon_agent/play_live.py --resume --free-roam`.
