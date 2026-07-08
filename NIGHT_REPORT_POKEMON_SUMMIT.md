CREDITS

# 🏆 THE CREDITS ROLLED — 2026-07-07 20:04, night shift #18, e4_run23 lap 2

**HALL OF FAME BANKED + PROMOTED TO CANONICAL** (sanctity VALID incl. monotonic; backup
`pre_hall_of_fame_backup_20260707_200544`; round-trip verified: map (1,80) Hall of Fame,
party [venusaur L95 7/250, persian 39, fearow 37, raticate 31, ekans 17, lapras 27]).
The full credits sequence drained on-screen (log: `logs/longrun/e4_run23.log`, the
[credits] map-cycle at +3..+19s past the HoF bank). Deliverable per the employment
terms: **THE CREDITS ROLL — delivered.**

## THE FINAL STAND (what a viewer would have seen)
Venusaur — the bedroom Bulbasaur, still nicknamed AAAAAAAAAA — walked into the Champion's
room as the last body on the team, sleep-locked Gary's Charizard and chipped it from 190
with x0.25 Razor Leaves, ONE-SHOT Rhydon (x4), and outlasted Gyarados to take the title at
**7/250 HP, alive 1, lead 3%**. Room #6 = Hall of Fame, banked at 346s into run23.

## THE CLOSING CHAIN (shift 18, ~35 min of fixes — each verified live)
1. **Levitate ability layer** (1a464e8): run21 donated 4+ free turns per Gengar picking EQ
   as "super-effective x2" — Levitate blocks ALL Ground moves. `_eff(move, enemy)` = chart
   + ability layer through the whole battle_agent pick path. Run22 lap 1 cleared Agatha in
   ONE PASS. VERIFIED.
2. **Kit rebalance** (e3a66f1): the $12k lap bought 8 Revives and NO sleep cure; now
   5 Revives + 3 Full Heals + Full Restores with the rest. Run23's winning lap ran exactly
   that plan. VERIFIED.
3. **Two zero-cost bounces** at fresh whiteout-center banks (the only safe kill window) to
   pick the fixes up mid-night; the XP ratchet (banked_E4 re-banked every whiteout) carried
   L66→L95 across runs 20-23 while money converged to a self-sustaining $12-14k/lap.

## THE MOUNTAIN (bedroom → credits, all autonomous banks)
Pallet bedroom → Brock → Mt Moon → Misty → Bill → S.S. Anne/Surge → Diglett's/Flash →
Rock Tunnel → Erika → Rocket Hideout/Scope → Tower/Flute → Snorlax → Koga → TEA/Silph →
Sabrina → Safari (HM03/04) → SEAFOAM CROSSING → Mansion Key → Blaine → Giovanni →
Victory Road → Indigo → **Lorelei → Bruno → Agatha → Lance → GARY → HALL OF FAME**.
Every stretch banked + promoted through the sanctity gate; the Sherpa timeline is ONE
continuous playthrough on disk, resumable at any point in its history via the backups.

## HONEST STATE (three-state)
- E4 engine (recon_e4 + battle_agent stack): VERIFIED end-to-end by the credits themselves.
- recon_bagdump.py (offline kit probe): VERIFIED (found the no-Ethers truth that shaped the
  endgame economy).
- recon_tutor_de.py (Double-Edge tutor errand): COMPILES, **never run** — she won before it
  was needed. Kept as the template for future move-tutor errands (pret truths inline).
- Known gaps that DIDN'T block credits (owed to the Kira timeline): bench dead weight
  (5 mons L17-39 vs L60+ content), the AAAAAAAAAA nickname (Name Rater, Lavender),
  double-battle target actuation, dialogue info-extraction (#12), PC/box system (#15).

## WHAT THE CREDITS PROVE / WHAT THEY DON'T
PROVEN: a blank-slate agent + the 15-block competency constitution can play FireRed
bedroom→credits fully autonomously — navigation, team-building, economy, gym gauntlets,
mazes, HM gates, the E4 comeback economy — on the real console loop with no RAM pokes.
NOT YET: the SHOW. The Kira timeline (timeline 2) — fresh bedroom→credits at human pace,
voiced, narrated, soul-on, resumable live across sessions — is now UNBLOCKED per the
two-timeline doctrine ("nothing touches it until the E4 flag flies on timeline 1").
The soul-debt ledger (STATE §0) is the pre-showtime work list; the E4 arc itself added
prime set-pieces (the whiteout-and-comeback treadmill, the 7-HP final stand).

WATCH STATUS: canonical bank is CLEAN (hall_of_fame — CHAMPION, credits drained); she is
in the Hall of Fame with her half-dead team of misfits and the title; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam` (post-game
Kanto: Cerulean Cave/Mewtwo is open — the love-letter episode when Jonny wants it).

---

# NIGHT REPORT -- started 2026-07-06 22:16
One line per shift below (newest last). The winning session promotes the magic word to line 1.

- TEST shift: plumbing OK, frontier visible
- shift 1 22:16->22:16 (1m): 0 commit(s), frontier unchanged
- shift 1 22:48->23:32 (44m): 3 commit(s), frontier unchanged
- shift 2 survey: CELADON CANONICAL (4 promotions: dex11_mankey -> hm05_flash -> rocktunnel_lavender -> celadon_reach) — judged Mankey catch, HM05 via Diglett's Cave, Flash taught+USED (tunnel crossed lit), Lavender+Route 8+UGP#2+Route 7 walked; 12 commits of general kills (stale-foe judgment, ROM TM/HM compat, Grid transition guard, plan hysteresis, section-DFS maze, CRITICAL=fighting-core) | frontier: Erika badge-4 strike IN FLIGHT (erika_run1.log, gym gauntlet mid-fight; front tile (4,4) unverified) | needs eyes: none — successor reads erika_run1.log end and promotes or fixes the leader approach
- shift 2 23:33->00:57 (85m): 16 commit(s), frontier ADVANCED
- shift 3 00:58->01:21 (22m): 1 commit(s), frontier unchanged
- shift 4 survey: BADGE 4 PROMOTED (erika_badge4 canonical — PP-famine kill chain: famine switch + heal-before-gym gate + level-dominance veto w/ offensive-famine override; Erika fell in 117.8s once the walls died) + badge-5 rope laid (Fuchsia road + Snorlax gate billed; questline engine hardened: item-truth beats HIDE flags, door hints GC(34,21)/Tower(18,6), step anchors, warp-aware ANCHOR-FIRST, no_connector guard, transit-time map learning down to the passthrough hop-loop — 5 look-ahead postmortems, each <90s to diagnosis) + Koga/Fuchsia registered + Rocket Hideout descent truth billed from disasm | frontier: flute strike (0x23D) in flight (flute_run8.log) — LIVE WALL = Route 8 Vileplume trainer (moveset famine: ace has ONE damaging move; fix = TM-teaching build, recipe in NEXT_SESSION) | needs eyes: none — successor reads flute_run8 end, promotes any bank, builds TM-teach first
- shift 4 01:22->02:18 (56m): 12 commit(s), frontier ADVANCED
- shift 5 survey: tm43_secret_power PROMOTED (Secret Power on Venusaur+Fearow — famine cured; recon_tm_errand.py = reusable TM buy+teach vehicle) + GHOST-VILEPLUME killed (runs 7-9's "wall" was stale RAM laundered by the indoors=blackout heuristic; blackout now demands battle evidence) + Rocket Hideout 80% cracked (poster gate, SPIN-TILE SLIDE CROSSER built+proven, Lift Key in bag; all deterministic in recon_hideout.py) + 4 general nav kills (directional-door/mat-row enter_warp, shop true-index, teach START open-verify) | frontier: B2F elevator approach (glide-only BFS misses walk corridors — add plain-step edges), then Giovanni -> Scope -> Tower -> flute | needs eyes: none — successor re-runs recon_hideout.py (40s, deterministic) and follows NEXT_SESSION first move
- shift 5 02:19->03:28 (69m): 5 commit(s), frontier ADVANCED
- shift 6 03:29->04:38 (69m): 3 commit(s), frontier ADVANCED
- shift 7 survey: BADGE 5 PROMOTED (koga_badge5 canonical — Koga fell in 54s to the NUKE-SLEEP opener after his Koffing Self-Destruct-traded her ace in run3; fuchsia_reach promoted en route with Routes 12-15 + all six juniors cleared) + 5 general engine kills (nuke-species sleep opener; prep STAND-DOWN for unexecutable train-first plans; grass fail-memory + grind-dead maps for one-way ledge pockets; stale-attach disarm BOTH ways — the foes-seen ledger properly kills shift-6's filed rival-miss bug; async-whiteout guard in _exit_to_overworld) + badge-6 road billed from live world-model edges (Saffron corridor + TEA gate 0x2A6) and VERIFIED carrying her R15->R14->R13->R12 | frontier: Route 12 NORTH GATEHOUSE crossing — FULLY DIAGNOSED double bug in _door_passthrough (gate (14,21) entry fails once then sits in the tried-set; the (12,86) rest house false-crosses and poisons _pt_known — exact fix recipe + static-path proof in NEXT_SESSION), then TEA errand -> Silph Co strike -> Sabrina | needs eyes: none — successor applies the two passthrough fixes (or reverses recon_snorlax's gate walk, ~20 lines) and relaunches sabrina_runN
- shift 7 04:39->05:34 (55m): 8 commit(s), frontier ADVANCED
- shift 8 05:35->06:14 (40m): 5 commit(s), frontier unchanged
- shift 8 survey (reconstructed by shift 9 — the shift hit the wall mid-strike without closing): TEA BANKED + SAFFRON ENTERED (2 promotions: tea_banked -> saffron_reach; canonical = Saffron (3,10)@(47,13) AT the gym door, badges 5, TEA in bag) — Route 12 north gatehouse crossed (4 passthrough bugs), WRONG-ENTRANCE RECOVERY general (Celadon Mansion back-door loop), Saffron=(3,10) KB fix + Sabrina GymSpec billed, want-aware door-passthrough (UGP ping-pong dead), recon_silph.py built (disasm truth banked, floors (1,47)-(1,57) live-verified) | frontier: Silph Co strike in flight — silph_strike3 died in a 1F lobby<->street livelock at shift end | needs eyes: none — shift 9 diagnosing
- shift 9 06:15->06:56 (40m): 2 commit(s), frontier unchanged
- shift 10 06:57->07:35 (38m): 1 commit(s), frontier unchanged
- ATTENDED 07:45-08:00 survey: SILPH CO CLEARED + PROMOTED (silph_cleared canonical, sanctity VALID — Card Key, 9F heal, GARY #6 won, LAPRAS banked to Bill's PC, GIOVANNI beaten 0x3E, MASTER BALL; 359s attended strike16 with Jonny's window up) | the 7F wedge was the Lapras NICKNAME KEYBOARD (frame-proven; gift talks now B-drain, 1fd4e74) | frontier repaired: NEXT_SESSION was shift-7 stale because shift 8 died mid-strike — contract now demands frontier-first rewrites (a99a607) | frontier: SABRINA badge 6 (gym door open, teleport-pad maze) | needs eyes: none — READY FOR RELOOP
- shift 1 08:02->08:02 (0m): 0 commit(s), frontier unchanged
- shift 2 08:03->08:03 (0m): 0 commit(s), frontier unchanged
- BRAKE 08:03: two consecutive shifts of provable nothing (identical frontier, zero commits). The wall needs human eyes -- see STATE section 0 CURRENT TRUTH + the last shift log: G:\JonnyD\NeuroAI_Bot\logs\nightshift\shift_002_0707_0803.log
- TEST shift: plumbing OK, frontier visible
- shift 1 08:09->08:10 (1m): 0 commit(s), frontier unchanged
- shift 1 08:13->09:01 (48m): 4 commit(s), frontier ADVANCED
- shift 2 survey: TWO PROMOTIONS (safari_hms: Gold Teeth + HM03 Surf + HM04 Strength, strike 20 = 50s tour-chain; surf_taught: Surf->Lapras + Strength->Venusaur) + 6 commits of general kills (water-not-a-road walkable, content-verified Grid guard vs dims-equal stale backups, per-edge elevation law, nudge-free stepper, reef-aware sea BFS, offset-aware edge crossing, 24px teach-chooser anchors) | frontier: SEAFOAM CROSSING -> Cinnabar -> Blaine badge 7 (R20 surface SEVERED - dual-flood proven; interior = pad_plan ladder maze, warp pairs billed in NEXT_SESSION) | needs eyes: none - successor ports recon_sabrina.pad_plan over the Seafoam floors per NEXT_SESSION first move
- shift 2 09:02->11:13 (131m): 7 commit(s), frontier ADVANCED
- shift 3 11:14->11:16 (2m): 0 commit(s), frontier unchanged
- shift 4 survey: TWO PROMOTIONS (cinnabar_reach: THE SEAFOAM CROSSING autonomous in 153s - boulder cascade 1F->B2F, FLAG 0x2D2 layout swap, becalmed B3F surf, arrow-warp exit, west sea; secret_key: Mansion key in 72s - toggle-state route, statue dance, SE back door, healed) + 7 general engine kills (box_open dual-clause bright-tileset fix, per-edge elevation law INTO Grid.edge_open, arrow+stair warp classes 0x62-0x6F, anchor-warp truth, coord-event script masks, box-burns-budget, Strength actuation verified live 0x805) | frontier: BLAINE badge 7 - gym door (20,4) unlockable NOW, full recipe + cached gym maps in NEXT_SESSION | needs eyes: none - successor clones the gym vehicle pattern and strikes Blaine
- shift 4 11:17->12:41 (83m): 10 commit(s), frontier ADVANCED
- shift 5 12:42->13:29 (48m): 3 commit(s), frontier ADVANCED
- shift 6 FAST-FAIL 13:30 (2s, exit 1): You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
- shift 6 13:30->13:30 (0m): 0 commit(s), frontier unchanged
- shift 7 13:31->13:37 (6m): 1 commit(s), frontier ADVANCED
- shift 8 13:38->14:36 (58m): 5 commit(s), frontier ADVANCED
- shift 9 14:37->15:05 (27m): 0 commit(s), frontier unchanged
- shift 10 15:06->15:21 (15m): 1 commit(s), frontier ADVANCED
- shift 11 15:22->15:53 (32m): 4 commit(s), frontier ADVANCED
- shift 12 15:54->16:35 (41m): 5 commit(s), frontier ADVANCED
- shift 13 survey: THE E4 LIVELOCK FAMILY IS DEAD (9 commits) — cb2 battle LIVENESS kills the phantom re-attach (gMain.callback2 0x030030F4; stale GBATTLE_RES_PTR = corpse, wired into st.in_battle); PARTY-WALK FINAL FORM (run12 frame proof: gPlayerParty ITSELF is battle-ordered in-fight, gBattlePartyCurrentOrder is only the RESTORE map — walk identity, verify by OUTCOME, sweep rows on miss); fswitch FOCUS PROBE (run11 frame: 'has no will to fight!' box ate every tap under a lit border) + target rotation; famine dirty-screen guard (once-per-battle try never burned by an open bag); FOE-AWARE famine (immune-only PP = famine, Levitate table); NEW INSTINCTS all live-verified: aimed FRs, mid-battle REVIVE (fallen-ace test), PP-restore Ether at famine, vehicle chooser picks them (run9 died declining 6 revive offers); Agatha WON in repro (65s, 5/6 alive), Bruno felled through full faint chains in runs 11-13 | frontier: e4_run14 DETACHED (survives handover — CHECK FOR LIVE PYTHON FIRST, read e4_run14.log end); gauntlet grinds autonomously: whiteout->restock->rechain, no known livelocks; walls = Agatha's ace attrition then LANCE (run7 reached his room); levers if runs keep dying: revive-sweep polish + Lance sleep-lock tuning | needs eyes: none
- shift 13 16:36->17:38 (62m): 9 commit(s), frontier ADVANCED
- shift 14 17:39->18:45 (66m): 10 commit(s), frontier ADVANCED
- shift 15 18:46->19:03 (17m): 3 commit(s), frontier ADVANCED
- shift 16 19:04->19:22 (18m): 2 commit(s), frontier ADVANCED
- shift 17 19:23->19:37 (15m): 1 commit(s), frontier ADVANCED
- shift 18 survey: **CREDITS ROLLED 20:04** (run23 lap 2 — Levitate ability layer 1a464e8 + kit rebalance e3a66f1 + two zero-cost bounces closed the PP-per-KO wall; Venusaur L95 last-mon-standing beat Charizard/Rhydon/Gyarados at 7 HP; hall_of_fame PROMOTED to canonical, sanctity VALID) | frontier: THE KIRA TIMELINE — timeline 1 summited, timeline 2 (fresh bedroom→credits, human pace, soul-on) now unblocked; soul-debt ledger = the work list | needs eyes: Jonny — press GO on canonical for the Hall of Fame pop-in, and decide when the Kira-timeline rebuild phase starts
- shift 18 19:38->20:09 (31m): 3 commit(s), frontier ADVANCED
