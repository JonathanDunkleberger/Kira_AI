# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #2 close)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 CURRENT TRUTH first (NIGHT SHIFT #2 block). Last shift:
🏙️ **CELADON CANONICAL** — the full chain banked in one night: dex 11 (judged Mankey catch) →
HM05 fetched (Diglett's Cave crossed both ways) → **FLASH taught to Venusaur** → **ROCK TUNNEL
crossed LIT** (use_field_move debut) → Lavender → Route 8 → UGP #2 under Saffron → Route 7 →
**CELADON CITY** (celadon_reach promoted; Center (48,11) + gym (11,30) + Erika registered).

FIRST MOVE: read `logs/longrun/erika_run1.log` (highest erika_run N) — the badge-4 strike
(LONGRUN_GOAL_FLAG=0x823) was in flight at close, deep in the gym gauntlet.
- If GOAL: promote (promote_bank.py G:/temp/longrun/banked_GOAL erika_badge4) → then the chain:
  Rocket Hideout (Game Corner basement, Silph Scope) → Pokémon Tower → Poké Flute → badge 5.
  KB frlg_gates.json bills every gate.
- If it failed at the LEADER approach: ERIKA_FRONT=(4,4) in campaign.py GYMS is a first guess —
  the gym's top NPC row was (5,4)/(6,4)/(7,4); grab a frame at the cleared gym top, fix the
  front tile, re-run. Everything else (door, juniors, Center) is probed truth.
- Party note: Ekans L15 rides fainted (harmless — core-down doctrine); heal at Celadon Center
  (registered) happens naturally next heal pick.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only, NEXT_SESSION.md at close. Launch recipe:
`LONGRUN_GOAL_FLAG=0x823 LONGRUN_BATTLE_LOG=1 POKEMON_SLEEP_LOCK=1 POKEMON_CATCH_JUDGMENT=1
POKEMON_PROACTIVE_BENCH=0 python pokemon_agent/recon_longrun.py kira_campaign.state 70 >
logs/longrun/<name>.log 2>&1` (bg). PROACTIVE_BENCH stays 0 on road/gym runs until the
Celadon-side grass is mapped (the bench pin fights Center-less roads). GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **Banked tonight (all sanctity-gated):** dex11_mankey → hm05_flash → rocktunnel_lavender →
  celadon_reach. That's Route 4 → CELADON in one shift: two cave systems, an HM fetch+teach,
  and the whole east-west road.
- **Soul beats that landed:** the judged Mankey catch ("fighting coverage — a real gap filled"),
  "and there's light!" in the tunnel, "Lavender Town... we made it through the dark."
- **The night's kills (12+ commits):** stale-foe catch judgment (gEnemyParty truth), sleep-then-
  throw, ball-less mart teeth, ROM TM/HM compat (killed shift-1's wrong "no Flash learner" call),
  Grid mid-transition guard, plan hysteresis, section-DFS maze crosser, TM-case sort fix,
  use_field_move primitive, segment-aware gates, CRITICAL=fighting-core (the heal-spin stall
  class killed at the definition), Lavender+Celadon Centers registered.
- **Owed / honest:** Erika front tile unverified; Venusaur is still named AAAAAAAAAA (Name Rater
  = Lavender; owning-it beat vs rename — soul call); Celadon Mart (dept store) unmapped for
  MART_STOCK; quasi-dupe catch refinement still parked.
