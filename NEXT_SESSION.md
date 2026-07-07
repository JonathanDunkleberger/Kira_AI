# NEXT_SESSION — resume prompt (write date 2026-07-06 late night; badge 3 banked)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 CURRENT TRUTH first (SESSION 7 LATE block). Last session:
⚡ **BADGE 3 CANONICAL** — Surge fell to the full automatic cascade (teach → TIMBER auto-cut →
trash-can solver debut → badge), flags 0x822/0x237/0x264 verified on-disk, party FULL, sanctity
VALID (badge3_bank_20260706_212559 promoted). Ship arc CLOSED (HM01, Gary 2-2 aboard, departed).

Tonight's chain (the credits path, badge 4 next):
1. EAST RUN 2 RESULT (read `logs/longrun/east_run2.log` for detail): STALL after 31 min at
   Route 6 (3,24)@(14,30), last opts=['heal'] — a HEAL SPIN on an OVERWORLD map (the interior-
   first rung doesn't apply; Route 6 is one edge from Vermilion so the adjacent-city excursion
   should have fired — diagnose WHY it didn't; grep 'HEAL' in the log tail). Party reached
   Venusaur L46/Persian L35/Fearow L33 and she toured ~(47,20)-coord maps (Route 9 side) before
   the faint. Run 1 proved Vermilion→Cerulean northbound autonomous; the in-leg auto-clear
   (field_clear) rode in run 2 — check whether it fired ('CLEARED with' in the log).
   FIRST MOVE: fix the Route-6 heal wedge (one wedge per cycle), relaunch GOAL_MAP=3,28.
   If a run GOALs: verify + heal-cycle + promote (promote_bank.py; trees REGROW on reload —
   the heal may need a re-cut first).
2. DEX 10 → FLASH: she's dex 9. One judged catch on Route 9/10 (Voltorb is NEW there) closes the
   Flash aide gate (KB bills HM05/aide/Route-2-gate already). Teach Flash needs a slot judgment.
3. ROCK TUNNEL: bill it into frlg_gates.json (entrances Route 10 north/south, Flash-dark), then
   the tunnel strike ITSELF (Flash taught + lit) → Lavender. DO NOT enter dark.
4. Then the badge-4 road: Lavender → Route 8 → UGP #2 → Celadon → Erika (KB route_notes has
   celadon_reach). Rocket Hideout/Tower are AFTER badge 4 on the chain.

Rules in force: EMPLOYMENT TERMS (CLAUDE.md top — two-wall shift ends, bank-and-continue,
5-15+ human-hours/shift), tripwire, arsenal, ground-truth-only (tasklist + raw logs; monitors
are decoration), single-run law (recon_longrun reaps predecessors), burn honest, NEXT_SESSION.md
at close. Launch recipe: `LONGRUN_GOAL_MAP=... / LONGRUN_GOAL_FLAG=... LONGRUN_BATTLE_LOG=1
POKEMON_SLEEP_LOCK=1 POKEMON_CATCH_JUDGMENT=1 POKEMON_PROACTIVE_BENCH=1 python
pokemon_agent/recon_longrun.py kira_campaign.state 75 > logs/longrun/<name>.log 2>&1` (bg).
Badge-4 goal flag when you get there: 0x823. GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **What banked:** HM01 canonical (pre_hm01 backup) → BADGE 3 canonical (pre_badge3 backup).
  Both promotions sanctity-gated, flags read back from disk in fresh cores.
- **The night's kills (all committed on feature/pokemon-agent):** ship-tour trilogy (re-sweep
  bound, inside-marker, per-map talk budget), interior-first heal + street gradient, captain's
  0x6C stair on-tile entry, mart sold-here fallback, HM-capability questline success, teach-flow
  trilogy (case home, raw-row, border-run cursor), gym-tree auto-cut.
- **What stalled / owed:** quasi-dupe catch refinement; Name Rater for AAAAAAAAAA (soul debt,
  Lavender is on the path!); trees regrow on savestate reload (characterized; heal tool re-cuts);
  east_run2 outcome unread if the session closed before it finished.
- **Day's first move:** read east_run2.log END → promote or fix → keep the chain rolling to
  Rock Tunnel/Flash (dex 9→10 catch opens the aide).
