# README FACT SHEET — assembly for the public README

_Night-shift #1, 2026-07-15. **Assembly only, no prose.** The README itself is drafted by Jonny +
Mysaria; this is the raw fact pack (numbers, architecture, story, quote) pulled from the certification
sources: `RUN_STATS_fresh_go_6.md`, `NIGHT_REPORT.md` (Order-4 eval), and the run-5 memory/notes._

---

## 1. THE CERTIFICATION NUMBERS (fresh_go_6 — the clean certifying run)

- **What:** Kira played FireRed **bedroom → credits fully autonomously**, cold cold-start, **zero human
  touches the entire run.** Certified 2026-07-15 18:12:47.
- **Wall-clock:** 22,112 s summed segment wall ≈ **6 h 08 m** headless at ~14× → models **~86 human-hours**
  of let's-play.
- **Boot:** `boot=FRESH map=(0,0) badges=0 party=[]` (the_opening spine, contamination-guarded).
- **Self-stop:** `OUTCOME: CREDITS — Champion defeated`; watchdog `rc=0`; no relaunch.
- **Decisions:** 353 (309 global decision events) · **Real battles:** 3,850.
- **Segments:** 5 launched / 5 completed · outcomes: CREDITS ×1, GOAL ×1, STALL ×2, TIMEOUT ×1.
- **Credits drain line:** 41 battles, **$29,120**.
- **Catches this run:** 3 (abra @Route 24, rattata @Route 4, diglett @Diglett's Cave).
- **Evolutions:** 6 (bulbasaur→ivysaur→venusaur, abra→kadabra, rattata→raticate, ekans→arbok, diglett→dugtrio).
- **Whiteouts:** 4 battle whiteouts + 8 nav blackout/strand events · **0 tracebacks / 0 errored, whole-log.**

### Final party (the qualifying six — distinct, de-duped, all acquired this run)

| slot | species | level |
|---|---|---|
| 1 | Venusaur | 87 |
| 2 | Lapras | 65 |
| 3 | Raticate | 62 |
| 4 | Arbok | 62 |
| 5 | Kadabra | 62 |
| 6 | Dugtrio | 61 |

- Ace level range across run: **22 → 76** (peaked L87 in gauntlet retries, under L100).
- Bench-min range: **8 → 61**. Gap **15 at the E4 gate** (the qualifying measurement point).
- Final badges: **8** · Final dex: **12**.

### Per-badge splits (first appearance, cumulative wall)

| badge | @ wall | badge | @ wall |
|---|---|---|---|
| 1–2 | 0h35m | 5–6 | 2h49m |
| 3 | 1h43m | 7–8 | 5h56m |
| 4 | 2h49m | credits | ~6h08m |

### Time-share (by decision-pick count)

travel 73.0% · menus/economy 11.9% · team-build 8.5% · battle/grind 6.0% · npc/dialogue 0.3%.

---

## 2. THE QUALIFYING-SHAPE CRITERIA (all ✅ on the untouched cold run)

Fresh cold start · zero human touches whole-run · six distinct species (0 dup, all acquired this run) ·
all ≥ L42 at E4 entry (floor L61) · ace-floor gap ≤ 15 at the gate (gap = 15) · no L100 (ace L76 at gate) ·
bounded grind (floor climbed monotone L28→L61, 0 tracebacks) · stats auto-generated · train self-stops.
**Verdict: CREDITS + QUALIFYING, no asterisk.**

---

## 3. ARCHITECTURE SUMMARY (one-liners for the README)

- **RAM-read game state** — reads live FireRed memory (party, levels, badges, map coords, bag, battle
  state) directly from emulator RAM addresses; no screen-scraping for state.
- **Warp-graph navigation** — a spatial world-model + warp/connection graph drives cross-map travel and
  doorway/entrance literacy (BFS travel engine, bend-discovery, destination-interaction).
- **E4-readiness gate** — refuses to enter the Elite Four until the whole party clears a level floor
  (≥L42) AND the ace-to-bench gap is ≤15; RED gate → ace-capped bench-grind to close the gap; terrain-
  escalates into the Victory Road cave (L40+ wilds) when open grass can't feed the floor.
- **Ace-cap** — caps the strongest mon so grind XP flows to the bench, producing a balanced qualifying
  squad instead of a solo-carry + dead-weight bench.
- **Soul / oracle layer** — Kira's always-on personality drives decisions through an LLM oracle
  (Groq/Claude tiers) with journey memory, roster-as-relationship, grudge/pride arcs; play-mode is a
  flag on the same one entity (One-Kira firewall — Pokémon state never leaks into always-on/companion mode).
- **Night-train methodology** — a 14× headless look-ahead harness (`recon_longrun.py`) with resumable
  checkpoint banking, run under an unattended watchdog loop; each shift banks progress to disk and hands a
  live frontier to the next, so a multi-day climb survives context resets.

---

## 4. THE RUN-5 vs RUN-6 STORY

- **fresh_go_5** (2026-07-15, NS#8) — the **first team-shape-qualifying** bedroom→credits run: gate GREEN
  at gap 15, six distinct species, 0 tracebacks. **But it carried an asterisk:** it took an *attended
  mid-run code fix* (`1b85dcd` — the E4-gate terrain-escalation into Victory Road) + a resume from a
  banked checkpoint. So the *machine* was proven end-to-end, but it was not a clean single untouched run.
- **fresh_go_6** (2026-07-15, NS#110) — the **no-asterisk certification.** Same qualifying shape, but on a
  **fully cold, zero-human-touch run** (every commit during the run was a monitor-glance doc edit; 0 code
  edits, 0 resumes, 0 attended fixes). It **removes run-5's asterisk** — the first bedroom→credits run that
  is BOTH team-shape-qualifying AND human-fingerprint-free.
- **The honest footnote (both runs):** entry at the *minimum* qualifying shape meant the no-heal 5-room
  E4 gauntlet whited out repeatedly (8× in run 6); the **designed self-funded whiteout-recovery loop**
  (heal → shop-restock from room prize money → re-enter) carried it — ace crept L76→L87, bench nudged to
  ~L61-65, until **RIVAL beat #11 vs Gary (won=True, "the Champion's Room")** → HALL OF FAME. Bounded,
  monotone-strengthening, converged untouched. Pacing is grind-heavy (~14k of 22k s wall was the readiness
  grind) — a watchability soft-note, not a disqualifier.

---

## 5. KIRA'S MILESTONE QUOTE FROM RUN 5

**NOT located in the tracked ops docs** (RUN_STATS / NIGHT_REPORT record mechanics, not her narration).
Her in-character credits/milestone line would live in the **soul-narration / journey_core log** for the
run-5 session, not in git-tracked files. **→ Jonny/Mysaria to pull the exact quote from Kira's narration
log when drafting the README** (this train had no access to that soul log). Placeholder slot in the fact
pack; everything else above is sourced and exact.

---

## 6. HEADLINE NUMBERS FOR THE TOP OF THE README (pick a few)

> Autonomous AI companion plays **Pokémon FireRed bedroom → credits**, **zero human input** ·
> **8 badges + Elite Four + Champion** in one cold run · **~86 human-hours** of gameplay compressed into
> **6 hours headless at 14×** · **3,850 battles**, **353 decisions**, a self-built balanced six
> (Venusaur / Lapras / Raticate / Arbok / Kadabra / Dugtrio) · **0 crashes** · self-stopped on the
> credits bank. Certified 2026-07-15.
