# NEXT_SESSION — resume prompt (write date 2026-07-07 ~19:05, night shift #15 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur L66→**L69 in-run** (Razor
Leaf/Sleep Powder/EQ/Secret Power + Strength as move#70 hits) / Persian 38 / Fearow 36 /
Raticate 31 / Ekans 15 / Lapras 26.

🗡️ **SHIFT-15 HEADLINE: LANCE WAS BEATEN (first time ever) — run18 attempt 1 cleared
ALL FOUR E4 ROOMS** (Lorelei+Bruno+Agatha in ~2 min wall, Lance by 226s) and entered
**room #5 = GARY THE CHAMPION** with alive=1 (Venusaur 45%, FR x0) → whiteout at 236s,
money $36,798. **GARY IS THE LAST WALL BEFORE CREDITS.**

⚔️ **LIVE: `recon_e4.py` run19 DETACHED (launched ~19:20), looping attempts on a 4h
deadline** (Start-Process — survives handover; CHECK FOR A LIVE PYTHON FIRST, read
logs/longrun/e4_run19.log END before touching anything). run19 = shift-14 stack PLUS
the shift-15 trio:
- **LAST-BODY INSURANCE revive (9e8fd18):** worthy gate now also fires when the active
  mon is the LAST body, hurt (<=50%), with >=2 revives — run18 postmortem: worthy=None
  past 3-5 corpses all through Lance (ace stood), so she entered Gary alive=1 and an
  ace faint = whiteout. Live-verify via "LAST-BODY INSURANCE armed" + USED count-drop.
  NOTE: standing fixtures (recon_revive_verify/ether_verify) COULDN'T pre-verify this —
  run18 overwrote banked_E4 with the 1-body Gary state (no kit); the walk itself is
  untouched + fail-safed. run19's fresh room banks restore fixture-capable states.
- **SPEND-THE-WAD kit:** items persist through whiteout, unspent cash HALVES — run18
  bled $63k->$0 in 4 attempts with $24k never converted. run19 shops FR x16 + Revive
  x8 + FH x2 (~$61k stored as items).
- **XP RATCHET + 4h deadline:** ace levels compound only within one process (L66->71
  across run18's two real attempts; a restart resets to canonical L66) — run19 banks
  banked_E4 at every whiteout-center (crash-resume via E4_BOOT=dir keeps the XP) and
  has 4h to converge on the level curve.
**run18 history: attempt 1 = LANCE BEATEN, died at Gary alive=1; attempt 2 = beat
Agatha on Struggle (ZERO PP), instant death at Lance; attempts 3-5 = broke, kitless.**
**THE PP WALL, CORRECTLY UNDERSTOOD (shift-15):** ~50 attack PP (RL25/EQ10/Str15) vs
26 E4 mons — but a CENTER HEAL RESTORES ALL PP, so the famine only binds WITHIN one
unbroken attempt. The cure is the level curve: one-shots drive PP-per-kill toward 1,
and the fat kit keeps each attempt alive deep enough to farm E4 XP (~+2-3 levels per
deep attempt). ⛔ DEAD END, do NOT build: VR loot sweep — Bulbapedia confirms FRLG
Victory Road has NO Elixir-class items (only Rare Candy 1F, hidden Full Restore 1F,
Max Revive 3F, TMs). Ethers are unbuyable, full stop. If run19's curve stalls, the
next real accelerants: (a) Double-Edge move tutor (VR 2F, 15PP/120pow, Venusaur-
compatible) widens the PP pool; (b) power-level a 2nd attacker (Lapras + Ice Beam
x4 vs Lance's dragons) — both detours, only if the curve provably plateaus.
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

**THE ETHER TRUTHS (shift-14, 8a811ec — run15 Agatha livelock postmortem):** the aimed
item walk now aims EXACTLY ONCE (focus + menu-time row on first party-screen sight)
then confirms BLIND — the Ether opens a MOVE-SELECT sub-box after the mon confirm, and
per-iteration re-focus B-cancelled it every lap (itemfail_34 forever). The ether OFFER
is gated on move slot 0 being damaging + CONNECTS + 0 PP — an IMMUNITY famine (foe-aware
famine class) cannot be cured by PP ("won't have any effect" ate 8 walks on full-PP
fodder vs Gengar). Probe note: gBattleMons rebuilds during the battle intro — RAM writes
to it only stick AFTER the action menu is up (GBATTLE_MENU_UP==1).

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
