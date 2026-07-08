# NEXT_SESSION — resume prompt (write date 2026-07-07 ~17:15, night shift #13 IN FLIGHT)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

🏅🏅 **CANONICAL = indigo_reach: INDIGO PLATEAU (3,9)@(12,19), ALL EIGHT BADGES +
VICTORY ROAD CLEARED, party HEALED, sanctity VALID, money $63,678** (backup
pre_indigo_reach_backup_20260707_154528). Party: Venusaur **L66** (Razor Leaf/Sleep
Powder/EQ/Secret Power) / Persian 38 / Fearow 36 / Raticate 31 / Ekans 15 / Lapras 26.

⚔️ **SHIFT-13 LIVE OBJECTIVE: THE ELITE FOUR — `recon_e4.py` e4_run12+ IN FLIGHT
(logs logs/longrun/e4_runN.log) with the FULL fix stack: cb2 liveness, display-order
walks (+contradiction guard +target rotation, dfe3646), fswitch FOCUS PROBE (c7d7b6d —
run11's "has no will to fight!" message box ate every tap; a lit border does NOT mean
list focus, probe with a tap and require MOVEMENT), dirty-screen famine guard,
revive/ether instincts + a chooser that PICKS them (85190d0 — run9 died at Bruno
declining 6 revive offers). Run11 proved the full instinct stack live: aimed FRs,
2 mid-battle revives, the Ether at famine, Bruno fell through a full faint chain.
gBattlePartyCurrentOrder is LAGGING/eventually-consistent in singles — trust it only
with the contradiction guard + rotation + focus probe around it.
THE CREDITS ARE THE NEXT BANK.**
FIRST MOVE: check for a live python process + read the newest e4_runN.log END.
If banked_CREDITS exists: promote it —
`python pokemon_agent/promote_bank.py G:/temp/longrun/banked_CREDITS hall_of_fame`
— then write CREDITS as NIGHT_REPORT.md line 1 + the mountain survey. Never kill a
run between "HALL OF FAME" and "BANKED".

**RUN 7/8 TRUTH CHAIN (shift-13 postmortems — read before touching the vehicle):**
- run7 attempt 1 CLEARED AGATHA (again — she IS beatable on one tank) and entered
  LANCE's room (alive 3, lead 0%); whiteout at Lance; ALL 10 FRs burned on Agatha.
- The re-chain then died in the BURNED-FAMINE LIVELOCK (root-caused + FIXED
  18ec09b/later): the famine switch used to fire the same turn an item flow ended,
  with the BAG still on screen → "_goto_pokemon failed" → the once-per-species
  famine try was CONSUMED → status-spam → all-dry → Struggle/abort forever. NOW:
  dirty-screen guard (bag closed first, try not consumed).
- PHANTOM-BATTLE CLASS KILLED (run3's suspicion, now wired): st.in_battle +
  recon_e4.fight_open now require gMain.callback2 (0x030030F4) NOT be
  CB2_Overworld/CB2_WhiteOut (0x080565B5/0x080566A5 thumb) — a stale
  GBATTLE_RES_PTR after whiteout is a corpse, never re-attach. LIVE-VERIFIED:
  battle frames show only 0x08010509/0x08011101 over 1200 frames incl. menus.
- Engine upgrades shipped 18ec09b+8233b90: FOE-AWARE famine (immune-only PP = famine;
  Levitate table for Gengar-line EQ hole), REVIVE instinct (_revive_worthy_slot:
  fainted mon out-levels all standing), PP-RESTORE instinct (Ether/Elixir at famine;
  canonical bag holds x1 Ether), item AIM wired (1a5ed9f built it, nothing called it)
  via border-readback + wait-for-party-screen.
- ⚠️ OPEN AT WRITE TIME: use_revive actuation still "selected but NOT consumed" in
  agatha_diag2/3 (logs logs/longrun/agatha_diagN.log) — instrumented walk (per-press
  party/bag/white + itemfail frame to agatha_probe/) riding agatha_diag4; read that
  log's use_item walk lines for the mechanism. recon_agatha.py now boots banked_E4
  (the live room3 bank) — the standing Agatha repro fixture.
- recon_e4.py now sets BATTLE_DEBUG_DIR itself (runs 7-8 aborted with ZERO frames
  because the env never rode my launches — never trust the launcher shell).
- Launch: `.venv\Scripts\python.exe -u pokemon_agent\recon_e4.py` with stdout to
  logs/longrun/e4_runN.log. Fresh boot re-loads canonical ($63k restored), re-shops,
  re-clears rooms 1-2 in ~3 min at 14x.

**THE GAUNTLET LAW (shift-12, commit 2d7234d): E4 RESETS ON WHITEOUT** — DEFEATED
flags do NOT survive a whiteout (run5 proof: post-whiteout she re-fought Lorelei).
E4 = ONE UNBROKEN RUN Lorelei→Bruno→Agatha→Lance→Gary on one tank of PP
(~RL25+EQ10+SP15+SecretPower20). Per-room banked_E4 = diagnosis only. Whiteout-
attrition does NOT accumulate; each attempt is fresh-from-canonical. The doors lock
behind her — no mid-chain center heal.

**KNOWN WALLS (in kill order):**
- Agatha = the PP sink (ghosts resist RL x0.5, EQ dead vs Levitate/Gengar,
  Secret Power dead vs Ghost; she Full-Restores + Hypnosis/Confuse Ray stalls).
  Run3 attempt-1 DID clear her at L66 (Struggle-recoil finish) — she is beatable
  on one tank.
- Lance killed run3 attempt-1 (arrived with 3 fainted). With FRs now actually
  consuming (bag TRUE-row fix 928dd53) arrival state should be far better.
- **REVIVES bought but NEVER offered in-battle** — the #1 unbuilt lever: if
  Venusaur faints vs Agatha/Lance, a Revive→FR resurrection from a sacrificial
  bench mon turns a loss into a continue. The party-screen AIM machinery already
  exists (post-switch item AIM, shift 12); wiring 'use_revive' into the
  ITEM-INSTINCT offer when ace fainted = the highest-value fix if runs keep dying
  at Lance. Build it on a STAGING copy while a run flies.
- run3's PHANTOM battle re-attach (fight_open pointer stale after whiteout) —
  anti-wedge now dumps forensics to G:/temp/longrun/e4_probe; read that on any
  wedge. Class: needs LIVENESS check not pointer check.

**KNOWN GAPS (owed, non-blocking):** double-battle target actuation (E4 all
singles — fine); Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight;
VR loot backlog.
**SOUL-DEBT:** Seafoam crossing + Lapras first-Surf + quiz gym + badge-8
homecoming + VICTORY-ROAD CLIMB + Gary-before-the-gate + THE E4 GAUNTLET +
whiteout-and-comeback arcs = prime narration set-pieces owed.

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER
commit/sweep. **py-spy is in .venv** — first tool for any silent wedge. Kill
orphan runs with taskkill //F //T (single-run law). Never kill a strike between
"goal=True" and "BANKED".

Rules in force: EMPLOYMENT TERMS, tripwire, arsenal, single-run law,
ground-truth-only, frontier-first rewrites. GO.

---

WATCH STATUS: canonical bank is CLEAN (indigo_reach — ALL EIGHT BADGES, VICTORY
ROAD CLEARED, healed, at the gates of the Indigo Plateau); the E4 strike runs on
the staging line; press GO on canonical and you'll see her walk into the Pokémon
League for the Elite Four and Gary's final stand; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
