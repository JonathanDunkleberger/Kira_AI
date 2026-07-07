# NEXT_SESSION — resume prompt (write date 2026-07-07 night shift #2)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 CURRENT TRUTH first (NIGHT SHIFT #2 block). Last session:
🔦 **HM05/FLASH CANONICAL** — dex 11 (judged Mankey catch, the choice framework's live debut),
HM05 fetched through Diglett's Cave (aide gate cracked warp-truth), **FLASH taught to Venusaur**
(case-sort teach fix, moves [75,77,79,148] on-disk), canonical = hm05_flash @ Route 2 (19,28).
Rock Tunnel strike (recon_rocktunnel.py) was IN FLIGHT at close — **read
`logs/longrun/tunnel_run<N>.log` (highest N) FIRST**: she enters the tunnel, LIGHTS FLASH
(use_field_move verified vs flag 0x806 — the party-menu field-move primitive works live), and
crosses the 1F/B1F maze by section-DFS (grid-BFS reachability + backtracking).

The chain (badge 4 critical path):
1. If tunnel_run banked → `python pokemon_agent/promote_bank.py G:/temp/longrun/banked_ROCKTUNNEL
   rocktunnel_lavender` → canonical = LAVENDER. If it failed, the wedge class is in the maze
   crosser (recon_rocktunnel.py cross_warp_maze) — known residual risk: exhausted-section
   ping-pong needs a breadcrumb stack (backtrack via the ORIGINAL entry warp, not the last one).
2. BADGE-4 ROAD: from Lavender, `LONGRUN_GOAL_MAP=3,6` recon_longrun (the Celadon road is
   KB-billed: Lavender → Route 8 → UGP #2 pass → Route 7 → Celadon; the road follower drives it
   via head_to_gym). Then ERIKA: `LONGRUN_GOAL_FLAG=0x823`. Watch the gym-tree (Cut known) +
   Erika is grass — Venusaur's Razor Leaf is resisted; Fearow's flying moves are the coverage.
3. Then the chain: Rocket Hideout (Game Corner) → Silph Scope → Pokémon Tower → Poké Flute →
   badges 5-8 (KB frlg_gates.json bills every gate to credits).

SOUL DEBT parked: her ACE Venusaur is nicknamed **"AAAAAAAAAA"** (day-one naming accident).
The Name Rater is in Lavender. Options: build the rename flow (naming-keyboard UI, a real
build) OR wire one owning-it beat ("yes, his name is AAAAAAAAAA — he's earned it"). Oracle/
Jonny call; don't let it block the road.

Rules in force: EMPLOYMENT TERMS (CLAUDE.md top — two-wall shift ends, bank-and-continue),
tripwire, arsenal, single-run law, ground-truth-only, NEXT_SESSION.md at close. Launch recipe:
`LONGRUN_GOAL_MAP=... / LONGRUN_GOAL_FLAG=... LONGRUN_BATTLE_LOG=1 POKEMON_SLEEP_LOCK=1
POKEMON_CATCH_JUDGMENT=1 POKEMON_PROACTIVE_BENCH=1 python pokemon_agent/recon_longrun.py
kira_campaign.state 75 > logs/longrun/<name>.log 2>&1` (bg). GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **What banked tonight (shift 2):** dex11_mankey (judged catch live debut — she reasoned
  "fighting coverage gap" and it was TRUE this time) → hm05_flash (HM05 flag + Flash on the ace,
  verified on-disk). Possibly rocktunnel_lavender if the last run GOAL'd — check NIGHT_REPORT.
- **The night's kills (commits df3c208, 945a8a8, f92e7a3 +):** catch-judgment stale-foe
  (gEnemyParty truth), sleep-then-throw, ball-less mart teeth (verified end-to-end round trip),
  ROM TM/HM compat table (killed a wrong shift-1 conclusion), Grid mid-transition guard,
  travel plan hysteresis (replan tie-flip churn), destination-aware maze crossing, TM-case
  sort-on-open teach fix, use_field_move primitive (Flash lit in-tunnel, verified).
- **Honesty:** the voltorb hunt caught a dupe ekans under the OLD bug (staging only, never
  banked); Rock Tunnel maze is the largest warp maze yet — if it failed overnight it's the
  breadcrumb-stack residual, one fix from done.
