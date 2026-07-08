# NEXT_SESSION — resume prompt (write date 2026-07-07 ~18:05, night shift #14 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur **L66** (Razor Leaf/Sleep
Powder/EQ/Secret Power) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26.

⚔️ **SHIFT-14 LIVE OBJECTIVE: THE ELITE FOUR — `recon_e4.py` run15 DETACHED**
(Start-Process — it SURVIVES the shift handover; CHECK FOR A LIVE PYTHON FIRST,
read logs/longrun/e4_run15.log END before touching anything). run15 carries the
**PARTY-MENU ORDER LAW fix (b7c21d0)** — the wall that killed every run14 attempt.
If banked_CREDITS exists: promote it —
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
— then write CREDITS as NIGHT_REPORT.md line 1 + the mountain survey. Never kill a
run between "HALL OF FAME" and "BANKED".

**THE ORDER LAW (shift-14, b7c21d0 — supersedes ALL prior party-walk models; probe
recon_partytruth.py, frame+RAM proof):** gPlayerParty HP is LIVE and accurate at all
times. While the in-battle party MENU is open, the game PHYSICALLY rearranges
gPlayerParty into display order (gBattlePartyCurrentOrder nibbles) and RESTORES it on
menu close. Menu row i IS gPlayerParty[i] — but ONLY while the menu is open. NEVER
carry a slot index across the menu-open boundary: decide WHAT to target before the
menu (species / 'active' / 'fainted'), resolve WHICH ROW by content at menu time
(battle_agent._menu_rows). gBattlerPartyIndexes = 0x02023BCE confirmed.

**RUN14 POSTMORTEM (why every attempt died):** ace faints → Revive aimed at pre-menu
"slot 0" → confirmed the healthy active mon's lead panel → "It won't have any
effect." boxes ate the sweep; fswitch's blind DOWN focus-probe moved the SEND OUT
sub-menu cursor to SUMMARY → the confirm A opened the summary screen → 3-minute
churns into corpses → whiteout, repeat. Fixed: _party_focus (B-first on the sub-menu,
pixel discriminator (210,130)+(230,130) white; DOWN-probe requires real movement;
never A unfocused), menu-time row picks everywhere (fswitch, _switch_to_slot, item
aim). VERIFIED on the forced ace-faint Agatha repro (recon_revive_verify.py): revive
consumed at menu rows 3 AND 5 across two opens, 0 NOT-consumed, 0 focus failures,
ace resurrected, Agatha WON. That repro (boots banked_E4, writes ace hp=1 via
b.core.memory.u16.raw_write — note: raw_write, the vendored __setitem__ is broken)
= the standing fixture for any party-walk regression.

**THE GAUNTLET LAW (shift-12, 2d7234d): E4 RESETS ON WHITEOUT** — one unbroken run
Lorelei→Bruno→Agatha→Lance→Gary on one tank of PP. Per-room banked_E4 = diagnosis
only. Fresh recon_e4 boot re-loads canonical ($63k restored), re-shops (FR x10,
Revive x6, Full Heal x4), re-clears rooms 1-2 in ~3 min at 14x. In-run whiteouts
drain money to $0 → later attempts in the SAME run can't re-shop; the kit persists
through whiteout though (items aren't lost, only unconsumed).

**KNOWN WALLS (in kill order):**
- Agatha = the PP sink (ghosts resist RL, EQ dead vs Levitate/Gengar, Secret Power
  dead vs Ghost; she Full-Restores + status-stalls). Beatable at L66 — proven twice
  now (run7 attempt-1, revive_verify with a forced faint chain).
- Lance = the frontier wall. Arrival state is everything: with revives now actually
  consuming, arrivals should carry a standing ace instead of 3 corpses.
- Watch run15 for the famine/voluntary switch path ("[engine] switch:" lines) — the
  _switch_to_slot rewrite (species-pinned confirm) is fixture-verified only
  indirectly; it is fail-safed (B-out → keeps fighting) but eyes on first live use.

**KNOWN GAPS (owed, non-blocking):** double-battle target actuation (E4 all singles);
Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight; VR loot backlog.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8 homecoming +
VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET + whiteout-and-comeback
arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit/sweep.
**py-spy is in .venv** — first tool for any silent wedge. Kill orphan runs with
taskkill //F //T (single-run law). Never kill a strike between "goal=True" and
"BANKED". Launch: `.venv\Scripts\python.exe -u pokemon_agent\recon_e4.py` with stdout
to logs/longrun/e4_runN.log (recon_e4 sets BATTLE_DEBUG_DIR itself → G:/temp/longrun/
e4_probe; frames land there on every wedge — LOOK at them, one glance beats an hour
of log archaeology).

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on
the staging line; press GO on canonical and you'll see her walk into the Pokémon
League for the Elite Four and Gary's final stand; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
