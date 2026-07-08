# NEXT_SESSION — resume prompt (write date 2026-07-07 ~19:45, night shift #17 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Canonical party: Venusaur L66 / Persian 38 /
Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26. **The E4 ratchet bank (banked_E4) is
far ahead of canonical: Venusaur L78+ and climbing.**

⚔️ **LIVE: `recon_e4.py` run21 DETACHED (launched ~19:45 shift-17), booted
E4_BOOT=G:/temp/longrun/banked_E4 (whiteout_center ratchet bank: Venusaur ~L78, money
~$0-2k, kit empty), 4h deadline.** CHECK FOR A LIVE PYTHON FIRST (two PIDs: venv shim +
real — taskkill //F //T the tree), read logs/longrun/e4_run21.log END before touching
anything. If banked_CREDITS exists: promote it —
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
— then write CREDITS as NIGHT_REPORT.md line 1 + the mountain survey. Never kill a
run between "HALL OF FAME" and "BANKED".

🔧 **SHIFT-17 FIXES (run20 postmortem — THE POVERTY DEATH SPIRAL, committed):** run20
proved the impostor fix (dozens of clean "impostor -> B-drain" recoveries, zero
livelocks) and the XP ratchet (L68→L78 in ~40 min — E4 XP is rich), but the ECONOMY
COLLAPSED: 6+ whiteouts, $19k → $0, and the shop's $2,000 cash reserve + Full-
Restore-first priority meant poverty laps ($2-6k E4 payouts, halved each whiteout)
bought NOTHING — "[shop] stocked already (money $2120)" was a lie; every lap ran
itemless and died at Lorelei's Jynx (SE psychic + sleep) with the L74+ ace unrevivable
(revive_item=None; the worthy-gate itself is FINE — worthy=0 = the dead ace's slot).
THE FIXES: (1) recon_e4 SHOPPING is now POVERTY-FIRST — Revives ($1500, the comeback
cycle; fainting also clears sleep), then Full Heals ($600, the Jynx counter), then
Full Restores; (2) the $2,000 reserve is GONE (money halves on whiteout, items don't —
reserved cash is half-wasted); (3) the can't-afford case now logs LOUD instead of
"stocked already"; (4) battle_agent LAST-BODY INSURANCE lost its 50% HP floor — at
alive==1 with >=2 revives a bench body is ALWAYS worth the turn (run19: a healthy
last-body ace walked all of Lance with no spare body; one crit = whiteout; the gate
self-closes at alive==2). ALL COMPILE; VERIFICATION = run21 live (watch for "[shop]
plan" lines actually buying on poverty laps, revives consuming, deeper attempts).

**THE CONVERGENCE MODEL (why run21 should work):** depth → payout → items → depth.
Steady-state itemless money is ~$2-6k/lap which now buys 1-4 Revives + Full Heals per
lap; each revive extends the attempt (more rooms = more payout); the XP ratchet
(~L78 and compounding, banked at every whiteout-center) drives toward one-shots.
Lorelei's Jynx is the current chokepoint; a Full Heal in bag cures the sleep-lock and
the cure instinct already consumes it (aim='active', menu-time rows).

**IF THE CURVE STILL PLATEAUS across run21's 4h (in order):** (a) Double-Edge move
tutor (VR 2F, 15PP/120pow, Venusaur-compatible — better one-shot math than Razor Leaf
vs Lorelei's bulky waters); (b) check bag for TM19 Giga Drain (Erika) — sustain vs
waters, but 5 PP; (c) power-level Lapras + Ice Beam vs Lance's dragons (big grind,
last resort). ⛔ DEAD ENDS, do NOT build: VR loot sweep (no Elixir-class in FRLG VR);
RAM-poking money (cheating — taints the whole proof).

**THE ORDER LAW (shift-14, b7c21d0):** gPlayerParty HP is LIVE always. While the
in-battle party MENU is open the game rearranges gPlayerParty into display order and
restores on close. Menu row i IS gPlayerParty[i] ONLY while open. NEVER carry a slot
index across the menu-open boundary: decide WHAT before the menu, resolve WHICH ROW by
content at menu time (battle_agent._menu_rows). gBattlerPartyIndexes = 0x02023BCE.

**THE GAUNTLET LAW (shift-12, 2d7234d): E4 RESETS ON WHITEOUT** — one unbroken run
Lorelei→Bruno→Agatha→Lance→Gary on one tank of PP. Per-room banked_E4 = ratchet/
crash-resume only. **THE XP RATCHET:** banked_E4 is re-banked at every whiteout-center
(levels survive process death via E4_BOOT); a fresh CANONICAL boot resets the ace to
L66 — never do that now, the ratchet bank is 12+ levels ahead.

**KNOWN WALLS (in kill order):** Lorelei→Lance itemless treadmill (run21's fix), then
GARY THE CHAMPION (reached once, run18; the last wall before credits).

**KNOWN GAPS (owed, non-blocking):** double-battle target actuation (E4 all singles);
Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight; Sleep-Powder spam at
0 damaging PP (cosmetic, resolves via whiteout).
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8 homecoming +
VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET + whiteout-and-comeback
arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit/sweep.
**py-spy is in .venv** — first tool for any silent wedge. Kill orphan runs with
taskkill //F //T (single-run law). Never kill a strike between "goal=True" and
"BANKED". Launch: `.venv\Scripts\python.exe -u pokemon_agent\recon_e4.py` with
E4_BOOT=G:/temp/longrun/banked_E4 and stdout to logs/longrun/e4_runN.log (recon_e4
sets BATTLE_DEBUG_DIR itself → G:/temp/longrun/e4_probe; frames land on every wedge —
LOOK at them).

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY ROAD
CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on the
staging line; press GO on canonical and you'll see her walk into the Pokémon League
for the Elite Four and Gary's final stand; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
