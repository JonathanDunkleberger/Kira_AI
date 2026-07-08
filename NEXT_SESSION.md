# NEXT_SESSION — resume prompt (write date 2026-07-07 ~19:15, night shift #16 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur L66 (L68+ on the E4 ratchet
banks) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26.

⚔️ **LIVE: `recon_e4.py` run20 DETACHED (launched ~19:15 shift-16), booted
E4_BOOT=G:/temp/longrun/banked_E4 (run19's room-4 bank: LANCE's doorstep, Venusaur
L68 lead 77%, alive 4, Revive x5 / FR x1, rooms 1-3 defeated flags intact, $2,478),
4h deadline.** CHECK FOR A LIVE PYTHON FIRST (two PIDs: venv shim + real — taskkill
//F //T the tree), read logs/longrun/e4_run20.log END before touching anything.
If banked_CREDITS exists: promote it —
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
— then write CREDITS as NIGHT_REPORT.md line 1 + the mountain survey. Never kill a
run between "HALL OF FAME" and "BANKED".

🔧 **SHIFT-16 HEADLINE FIX — THE IMPOSTOR-WHITE-BOX LIVELOCK (battle_agent.py,
run19 postmortem):** run19 attempt 1 reached LANCE (rooms 1-4 banked by 195s; the
shift-15 stack WORKED — revives consumed at menu-time rows, LAST-BODY INSURANCE
armed AND fired at 9/193). Then the insurance revive's result box ("PERSIAN's HP
was restored by 52 point(s)." — party screen, revive = half of 104) was left
UNDRAINED: use_item's old drain exited on `_white_box and not _bag_screen`, which
the party-screen box satisfies. Every downstream path then believed the ACTION MENU
was up (same pixels), walked a STALE GBATTLE_ACTION_CURSOR that never moved, and
NEVER pressed A/B — anti-wedge abort → re-enter → identical screen, ~1 wedge
frame/sec for minutes. THE FIX (class-killer at the chokepoint): pixel truth is not
enough — `_action_cursor_alive()` demands the cursor RESPOND to a horizontal tap
(retry for eaten taps); `_settle_action_menu` B-drains impostors (B dismisses
message boxes, backs out of party/bag screens, no-op at the real menu) and logs
"action-menu impostor ... -> B-drain"; use_item's post-use drain routes through it.
COMPILES + committed; VERIFICATION = run20 live (watch for the impostor line — the
same famine/revive sequence recurs at Lance).

**THE PP WALL + XP RATCHET (unchanged, shift-15):** ~50 attack PP vs 26 E4 mons;
center heal restores ALL PP so the famine binds only within one unbroken attempt;
the cure is the level curve (one-shots → PP-per-kill → 1). Rooms 1-3 ate ~14 Full
Restores in run19 — the kit funds ONE deep attempt per shop; whiteouts halve money
($63k → $2.4k already); in-process attempts after the first run broke, riding
center-PP-refills + banked XP. ⛔ DEAD END, do NOT build: VR loot sweep (no
Elixir-class items in FRLG VR; Ethers unbuyable). If the curve stalls across run20's
4h: (a) Double-Edge move tutor (VR 2F, 15PP/120pow, Venusaur-compatible); (b)
power-level Lapras + Ice Beam vs Lance's dragons; (c) CANDIDATE (run19 postmortem):
mid-fight FODDER revives — worthy-gate currently revives nothing while ≥2 alive
(worthy=None past 4 corpses all through Lance); reviving a bench body buys free
Full-Restore turns for the ace. All detours — only if the curve provably plateaus.

**THE ORDER LAW (shift-14, b7c21d0 — supersedes ALL prior party-walk models):**
gPlayerParty HP is LIVE and accurate at all times. While the in-battle party MENU is
open, the game PHYSICALLY rearranges gPlayerParty into display order
(gBattlePartyCurrentOrder nibbles) and RESTORES it on menu close. Menu row i IS
gPlayerParty[i] — but ONLY while the menu is open. NEVER carry a slot index across
the menu-open boundary: decide WHAT to target before the menu (species / 'active' /
'fainted'), resolve WHICH ROW by content at menu time (battle_agent._menu_rows).
gBattlerPartyIndexes = 0x02023BCE confirmed.

**THE GAUNTLET LAW (shift-12, 2d7234d): E4 RESETS ON WHITEOUT** — one unbroken run
Lorelei→Bruno→Agatha→Lance→Gary on one tank of PP. Per-room banked_E4 = ratchet/
crash-resume only (E4_BOOT keeps XP + defeated flags for the CURRENT unbroken
attempt; a whiteout resets the rooms). Fresh recon_e4 boot re-loads canonical
($63,678 restored), re-shops, re-clears rooms 1-3 in ~3 min at 14x.

**THE ETHER TRUTHS (shift-14, 8a811ec):** the aimed item walk aims EXACTLY ONCE
(focus + menu-time row on first party-screen sight) then confirms BLIND — the Ether
opens a MOVE-SELECT sub-box after the mon confirm; per-iteration re-focus
B-cancelled it every lap. The ether OFFER is gated on move slot 0 damaging +
CONNECTS + 0 PP. gBattleMons rebuilds during the battle intro — RAM writes stick
only AFTER the action menu is up (GBATTLE_MENU_UP==1).

**KNOWN WALLS (in kill order):** Lance (arrival state is everything — revives now
consume AND the box drains; watch run20), then GARY THE CHAMPION (reached once,
run18 attempt 1, alive=1 → whiteout; the last wall before credits).

**KNOWN GAPS (owed, non-blocking):** double-battle target actuation (E4 all
singles); Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8 homecoming
+ VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET + whiteout-and-
comeback arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill orphan
runs with taskkill //F //T (single-run law). Never kill a strike between "goal=True"
and "BANKED". Launch: `.venv\Scripts\python.exe -u pokemon_agent\recon_e4.py` with
stdout to logs/longrun/e4_runN.log (recon_e4 sets BATTLE_DEBUG_DIR itself →
G:/temp/longrun/e4_probe; frames land there on every wedge — LOOK at them, one
glance beats an hour of log archaeology).

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on
the staging line; press GO on canonical and you'll see her walk into the Pokémon
League for the Elite Four and Gary's final stand; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
