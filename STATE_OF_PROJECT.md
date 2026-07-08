# STATE OF PROJECT — reality audit (2026-06-28)

The honest map of what's REAL vs a disconnected GHOST vs DEAD. Three-state per the operating rules
(`CLAUDE.md`): **COMPILES** (runs) · **WIRED** (data reaches the system meant to use it — esp. Kira's
DECISION/voice, with where) · **VERIFIED** (proven, or "needs live eyes"). The #1 column is **REACHES**:
BRAIN (decision/voice) vs DISPLAY-ONLY vs DEAD. "Exists but unwired" is the most important category —
that's the failure that's burned us (goals were built+displayed but never reached her brain).

Companion docs: `pokemon_agent/CODEBASE_AUDIT.md` (pokemon detail + stuck-vector list),
`pokemon_agent/FORWARD_CLIMB_STAGING.md` (gym 3→8 plan).

---

## 0. CURRENT TRUTH — read THIS first (2026-07-05 night; archaeology is the appendix below)

### ▶ GO PROCEDURE — Jonny may press GO on the Sherpa save AT ANY MOMENT (watch-readiness is permanent)
The Sherpa save IS Jonny's own playthrough; he can watch it live whenever, no ritual. **Standing launch:**
```
python pokemon_agent/play_live.py --resume --free-roam        # resume the LIVING campaign save, she decides
```
(The launcher pins `--phones=Leviathan` for the VTS mouth-flap firewall; live = windowed + music, NOT --headless.)
**POP-IN (Jonny taps the glass, ~30s):** if a headless climb is running, it finishes the tick + banks
live→canonical + stops; then `python pokemon_agent/play_live.py --resume --free-roam` hands him the seat from
EXACTLY where she is. Whatever state he leaves when he steps away IS the new canonical (bank it, resume from
there — his session is part of the same one playthrough, not a detour). **RECOVERY if she wedges mid-watch:**
re-launch the same command — `--resume` reboots from the last banked checkpoint; the escape-hatch/deep-wedge
ring + the new heal-strand breaker self-recover. **PERSISTENCE LAW:** exactly ONE canonical Sherpa state
always; headless farming ADVANCES that one world; **NEVER bank a wedged/hurt state forward** — WIP on clearly-
named disposable staging copies, only bank clean forward states (full sanctity bundle: .state + journey_core
+ strat + world + soul). **WATCH-READY NOW:** canonical `kira_campaign.state` is clean (healthy party, not
wedged); on GO she shops → grinds Ivysaur → forward-drives to the Nugget-Bridge Gary.

### ── 2026-07-07 NIGHT SHIFT #16 — THE IMPOSTOR-WHITE-BOX LIVELOCK KILLED; run20 ratchet in flight ──
**CANONICAL unchanged = indigo_reach. e4_run20 IN FLIGHT (logs/longrun/e4_run20.log, detached,
launched ~19:15, 4h deadline, booted E4_BOOT=banked_E4).** run19 postmortem (found wedged live
at 19:05, killed, fixed, relaunched):
- **run19 attempt 1 reached LANCE in 195s** (rooms 1-4 banked; shift-15 stack ALL VERIFIED
  LIVE: spend-the-wad shop, revives consuming at menu-time rows, **LAST-BODY INSURANCE armed
  AND fired at 9/193** — the 9e8fd18 live-verify owed from shift-15 is PAID).
- **THE WEDGE [ROOT-CAUSED via frame + log, killed 53979b3]:** the insurance revive's result
  box ("PERSIAN's HP was restored by 52 point(s)." — party screen; revive = half of 104) lights
  the SAME white-panel pixels as the action menu. use_item's drain exited on those pixels →
  box left up → `_settle_action_menu` believed the menu was up → `_goto_fight/_goto_pokemon`
  walked a STALE GBATTLE_ACTION_CURSOR that never moved → **no path ever pressed A/B** →
  anti-wedge abort → re-enter → identical screen, ~1 frame/sec for minutes (the run15/18
  itemfail/death class, finally understood).
- **THE FIX [VERIFIED live in run20]:** `_action_cursor_alive()` — the action menu is only
  REAL if the cursor RESPONDS to a horizontal tap (readback moves; retry for eaten taps);
  `_settle_action_menu` B-drains impostors (B: dismisses boxes, backs out party/bag, no-op at
  the real menu), loudly logged; use_item's post-use drain routes through it. The impostor
  line fires on ~every aimed item use in run20 and RECOVERS every time; famine switch landed
  post-fix ("SWITCHED to species 20").
- **THE ECONOMY IS SELF-SUSTAINING [measured, run20]:** E4 payouts beat the whiteout halving
  ($2,118 → $10,518 across attempt 2). The whiteout-center re-shop (already in recon_e4)
  rebuys money-aware every lap (FR x2-3 + revive per attempt). Attempt cadence ~3-4 min:
  whiteout → center bank (XP ratchet) → heal → re-shop → re-climb. Attempt 1 died at Lance
  (last FR spent, Sleep-Powder stall at 0 damaging PP); attempt 2 died at Agatha but arrived
  with ALL 6 ALIVE (vs run18's corpse parades). Ace L68 and compounding.
- **ACCELERANT CANDIDATES if the curve plateaus (in order):** (a) mid-fight FODDER revives —
  worthy-gate revives nothing while >=2 alive (worthy=None past 4 corpses through Lance);
  reviving a bench body buys free Full-Restore turns for the ace; (b) Double-Edge tutor
  (VR 2F); (c) power-leveled Lapras+Ice Beam vs Lance. ⛔ VR loot sweep stays DEAD.
- **Watch item (cosmetic):** Sleep Powder spam when only status PP remains (war-must-advance
  picks it over Struggle) — battle still resolves via whiteout, just slow.

### ── 2026-07-07 NIGHT SHIFT #15 — LANCE BEATEN; GARY = THE LAST WALL; run19 economy stack ──
**CANONICAL unchanged = indigo_reach. e4_run19 IN FLIGHT (logs/longrun/e4_run19.log, detached,
4h deadline, launched ~19:20).** run18 (shift-14's detached strike) = the deepest run ever:
- **ATTEMPT 1 BEAT LANCE (first time)** — Lorelei+Bruno+Agatha cleared in ~2 min wall, Lance's
  room banked 131s, **room #5 = GARY THE CHAMPION entered at 226s** with alive=1 (Venusaur 45%,
  FR x0) → whiteout 236s. ONLY GARY REMAINS between her and the credits.
- **Attempt 2:** beat Agatha ON STRUGGLE (zero PP anywhere), entered Lance with 1 body + 0 PP →
  instant whiteout. Attempts 3-5: broke ($0), kitless, early deaths.
- **ROOT CAUSE, Gary arrival [VERIFIED in-log]:** `_revive_worthy_slot` only revives a fainted
  mon that OUT-LEVELS all standing — while the L70 ace stood, worthy=None past 3-5 corpses (all
  of Lance), so the bench stayed dead and an ace faint = whiteout. **FIX (9e8fd18) [COMPILES,
  live-verify owed]: LAST-BODY INSURANCE** — active mon is the LAST body + hurt (<=50%) + >=2
  revives → revive strongest fainted regardless of level (a body = the comeback cycle). Walk
  machinery untouched. Standing fixtures couldn't pre-verify (run18 overwrote banked_E4 with a
  1-body kitless Gary state); run19's room banks restore fixture-capable states — verify live
  via "LAST-BODY INSURANCE armed" + count-drop.
- **ECONOMY LAWS (069d28f) [WIRED into recon_e4, verified in run19's boot log]:** (1) items
  persist through whiteout, unspent cash HALVES → SPEND THE WAD (FR x16 + Revive x8 + FH,
  ~$61k as items; run18 bled $63k→$0 with $24k never converted). (2) XP compounds only within
  one process (L66→71 across run18's two real attempts; restart = canonical L66) → 4h deadline
  + XP RATCHET (bank banked_E4 at every whiteout-center; E4_BOOT=dir resumes with levels).
- **THE PP WALL, resolved conceptually:** center heal restores ALL PP → famine binds only
  within one attempt; the level curve (one-shots → 1 PP/kill) is the cure and run19 farms it.
  ⛔ VR loot sweep = DEAD END (Bulbapedia: FRLG VR has NO Elixir-class items). If the curve
  plateaus: Double-Edge tutor (VR 2F) or power-leveled Lapras+Ice Beam — detours, not yet.
- **Live-verified in run18/19:** voluntary switch (species-pinned) fired live and landed
  clean x2; shop scaler; whiteout re-shop loop; north-door chain. **Known cosmetic:** post-
  whiteout "clerk-approach FROZEN x3" always self-recovers via the B/A drain (watch item).

### ── 2026-07-07 NIGHT SHIFT #14 — THE ORDER LAW: menu-time content walks (run14 wall killed) ──
**CANONICAL unchanged = indigo_reach. e4_run15 IN FLIGHT (logs/longrun/e4_run15.log, detached)
carrying b7c21d0.** Run14 postmortem (every attempt died at a fainted ace): the shift-13 "final
form" was STILL half-wrong — new probe `recon_partytruth.py` (frame+RAM, banked_E4 fixture)
settles it for good:
- **THE ORDER LAW [VERIFIED, probe]:** gPlayerParty HP is LIVE always (Raticate ticked
  37→24→11→7→0 at its own slot). While the party MENU is open the game PHYSICALLY rearranges
  gPlayerParty into display order and RESTORES it on close. Menu row i IS gPlayerParty[i]
  ONLY while the menu is up. The whole family (run12 double-convert, run14 revive-on-wrong-row,
  voluntary switch mis-lands) = carrying a slot index ACROSS the menu-open boundary.
- **run14's actual churn, frame-diagnosed:** Revive aimed at pre-menu "slot 0" confirmed the
  HEALTHY active mon's panel → "no effect" boxes ate the sweep; the fswitch focus-probe's blind
  DOWN moved the SEND OUT sub-menu cursor to ▶SUMMARY → confirm A opened the summary screen →
  3-min churns into corpses → whiteout loop (money → $0, no re-shop within a run).
- **Fix (b7c21d0) [VERIFIED, recon_revive_verify.py — forced ace-faint chain vs Agatha]:**
  `_menu_rows()` (menu-time content = walk truth), `_party_focus()` (B-first on the sub-menu —
  pixel discriminator (210,130)+(230,130) white vs teal; DOWN-probe requires real movement;
  never A unfocused), fswitch/switch/item-aim all pick rows by content at menu time (aim kinds
  'active'/'fainted', species-pinned switch confirm). Repro result: revive consumed at menu
  rows 3 AND 5 across two opens (the rows really move), 0 NOT-consumed, 0 focus failures, ace
  resurrected, AGATHA WON. `gBattlerPartyIndexes=0x02023BCE` confirmed + billed.
- **Ops:** vendored mgba `memory.__setitem__` is BROKEN (wrong arg count) — use
  `u16.raw_write(addr, val)`; recon_revive_verify = standing party-walk regression fixture.
- **ETHER truths (8a811ec — run15 postmortem; run15 cleared Lorelei in 59s then livelocked
  at Agatha):** the aimed item walk aims ONCE then confirms blind (the Ether opens a
  MOVE-SELECT sub-box after the mon confirm; per-iteration re-focus B-cancelled it every
  lap = itemfail_34 forever). Ether OFFER gated on move slot 0 damaging + CONNECTS + 0 PP —
  an IMMUNITY famine can't be cured by PP ('won't have any effect' x8 on full-PP fodder).
  [VERIFIED: recon_ether_verify PASS (used on first walk, immunity offers suppressed) +
  recon_revive_verify re-PASS after the aim-once change. Probe law: gBattleMons rebuilds
  during the battle intro — RAM writes stick only after GBATTLE_MENU_UP==1.]
- **THE POCKET-HOLE BUG (7f83916 — run16's all-offers collapse, forensics-confirmed live in
  run17):** consuming the LAST of an item mid-battle leaves a zero HOLE in the RAM Items
  pocket; `_items_pocket` broke at the first zero id, so the Ether (display row 0) hitting
  x0 at Agatha made the WHOLE pocket read empty for the rest of the process — no potion/
  cure/revive offers ever again (run16: Agatha's last mon at 17 HP, then 5 whiteouts with
  FR x10 "in the bag") while camp.bag_count (scan-all) kept the kit-check green. The new
  `revive-check` forensic line named it in one glance: `revive_item=None, worthy=0`.
  Fixed scan-all-skip-holes; hole assertion rides recon_ether_verify. [VERIFIED both fixtures]
- **Cure instinct reads the ACTIVE battler now (dd56205)** — `_lead_status` read
  gPlayerParty[0], so a sleep-locked fodder post-switch never got a cure while the dead
  ace's 'none' was consulted. Decodes state['ours'].status1 (gBattleMons). [WIRED, fixture-
  covered indirectly]
- **Eyes owed:** `_switch_to_slot` rewrite (famine/voluntary switch) fail-safed but only
  indirectly exercised — watch "[engine] switch:" lines; "_goto_pokemon failed" famine-switch
  flakiness on a dirty screen remains (pre-existing class, fail-safed).
**FRONTIER: run18 (full stack: order law + ether + cure-active + pocket-hole) through
Agatha → LANCE (the wall) → Gary → HALL OF FAME → CREDITS.**

### ── 2026-07-07 NIGHT SHIFT #13 — the LIVELOCK FAMILY dies: cb2 liveness + display-order walks ──
**CANONICAL unchanged = indigo_reach. e4_run9 IN FLIGHT (logs/longrun/e4_run9.log) — full fix
stack aboard.** Runs 7/8 postmortem (both died in Agatha livelocks; run7 attempt-1 CLEARED
Agatha again and whiteouted at LANCE — Lance is the wall). Commits 18ec09b, 8233b90, 5e94a63:
- **PHANTOM-BATTLE CLASS KILLED (run3's suspicion CONFIRMED):** GBATTLE_RES_PTR stays valid
  after a whiteout → pointer-only gates re-attach a DEAD battle (frozen struct, taps move
  nothing). LIVENESS = gMain.callback2 (0x030030F4): CB2_Overworld/CB2_WhiteOut = corpse.
  Wired into st.in_battle + recon_e4.fight_open. VERIFIED both polarities. [VERIFIED]
- **BURNED-FAMINE GUARD:** the famine switch fired with the BAG still open (the turn an item
  flow ended), failed, and CONSUMED its once-per-species try → status-spam → all-dry →
  Struggle/abort forever. Now closes the bag first, try not consumed. [VERIFIED, diag2]
- **PARTY-WALK FINAL FORM (two models overturned; run12 frame settled it):**
  gPlayerParty ITSELF is battle/display-ordered DURING a fight — a live-scanned slot
  (healthy/fainted/active) is ALREADY the display row; gBattlePartyCurrentOrder (0x0203B0DC)
  is only the RESTORE map (converting through it DOUBLE-converts). Walk identity + verify by
  OUTCOME (count-drop for items, healthy-active for switches), sweep rows on failure; fswitch
  adds a FOCUS PROBE (run11 frame: a "has no will to fight!" message box ate every tap while
  the lit border FAKED focus) + target rotation. [VERIFIED live: agatha_diag5 AGATHA WON 65s;
  runs 11-13 = aimed FRs, mid-battle revives consumed, Bruno felled through full faint chains]
- **Engine instincts added:** foe-aware PP famine (immune-only PP = famine; _LEVITATE_SPECIES
  chart-hole table), REVIVE instinct (fainted mon out-levels all standing), PP-RESTORE instinct
  (Ether/Elixir at famine; canonical bag holds Ether x1 + Moon Stone). Item AIM was half-wired
  since 1a5ed9f (ZERO callers) — now wired + verified. [VERIFIED in repro]
- **Ops truths:** recon_e4.py self-sets BATTLE_DEBUG_DIR (runs 7-8 aborted with ZERO forensic
  frames — the env never rode my launches); recon_agatha.py = standing repro, boots banked_E4;
  the night-loop handover KILLS in-flight runs (run6 died externally mid-Agatha, not defeated).
**FRONTIER: run9 through Agatha → LANCE (the wall) → Gary → HALL OF FAME → CREDITS.**

### ── 2026-07-07 NIGHT SHIFT #12 — the Agatha DOUBLE-KILL: bag TRUE-row + war-must-advance ──
**CANONICAL unchanged = indigo_reach. e4_run3 IN FLIGHT (logs/longrun/e4_run3.log).**
e4_run2 postmortem (rooms 1-2 = Lorelei+Bruno fell first-try in ~50s; wall = Agatha, two
engine bugs, both fixed + VERIFIED, commit 928dd53):
- **BAG TRUE-ROW LAW (the mart law, battle-bag edition):** in-battle bag selection =
  cursor(0x0203AD04) + scrollOffset(0x0203AD0A = gBagMenuState.itemsAbove[0], derived +
  press-verified by recon_bagscroll.py), and BOTH persist between opens. use_item_in_battle
  navigated on the raw cursor byte → run2's "selected but NOT consumed" was A landing on
  Revive ("no effect") or CANCEL (bag closes; follow-up A's spilled into the action menu —
  the "BAG open at the turn loop" chaos). The Full Restore kit was a coin-flip exactly when
  the E4 plan is FR attrition. Fixed nav is VERIFIED live from the poisoned state
  (recon_bagscroll_verify.py: cursor=5/scroll=1 → FR consumed, HP 23→193, PASS).
- **WAR-MUST-ADVANCE (turn-based livelock class, run-existential):** a can't-flee TRAINER
  battle where every move is streaked/0-PP returned no_usable_move/no_effective_move and
  submitted NO action — in a turn-based game the foe never gets a turn, so the battle NEVER
  resolves: famine → anti-wedge abort → vehicle re-enters → forever. She couldn't even LOSE
  her way to the whiteout ratchet (respawn heals PP; DEFEATED flags ratchet cleared rooms).
  Now: trainer battle + moves-exhausted → clear the streak and re-fire the best PP-having
  move (connect-capable preferred, immune-damaging last resort — a passed turn still
  resolves); zero PP anywhere → FIGHT+A and the game substitutes STRUGGLE (_struggle()).
  Wild battles keep the old surface-to-flee behavior. VERIFIED by inspection + will show in
  run3's log ("war-must-advance" lines).
- **Agatha math (why the ratchet wins):** ghosts resist RL x0.5, EQ dead vs Levitate,
  Secret Power dead vs Ghost; arriving from Bruno with ~half PP famines mid-fight. Fresh
  full-PP arrival (post-whiteout, rooms 1-2 pass through on DEFEATED flags) + working FRs +
  sleep-lock should take her; Lance = RL chip + Secret Power x1; then Gary → HALL OF FAME.
- **Single-run law bite:** the "two processes" at shift start were ONE run — the .venv
  python.exe is a SHIM that parents the real interpreter (kill the TREE, `taskkill /F /T`).
- **⚠️ THE GAUNTLET RESETS ON WHITEOUT (run5 ground truth, kills the vehicle's premise):**
  after a whiteout the E4 DEFEATED flags do NOT carry — run5 re-fought Lorelei (lead
  100%→57%) and Bruno on the re-entry chain. THE E4 MUST FALL IN ONE UNBROKEN RUN.
  Per-room ratchet banking is still useful for diagnosis but NOT for progress. The
  winning formula = full kit (canonical's $63k → FR×10/Rev×6/FH×4) + aimed items +
  war-must-advance + the Struggle endgame (PP-free 40bp offense whose recoil the FR
  stack funds) + bench PP pools. run6 runs this from canonical: ~3 full-kit gauntlet
  attempts fit the deadline (whiteout → re-shop ~$16k → retry from Lorelei).
- **PP economy is the structural wall** (Venusaur ~50 damaging PP vs ~55-60 needed over
  5 fights; only ONE Ether aboard). If run6's attempts near-miss at Lance/Gary, next
  levers: Elixir/Max-Ether loot sweep (Victory Road/Mansion item balls, billed as
  backlog) or an Ether-instinct (needs mon→move two-level item targeting).
- run3's "phantom battle" wedge stands as filed (stale GBATTLE_RES_PTR post-whiteout;
  forensics now dump at every anti-wedge abort) — run5 did NOT reproduce it (its
  re-entry chain fought REAL battles), so it may need the specific mid-battle-loss
  shape; the dumps will catch it if it returns.
- FILED (unchanged): revives bought but never OFFERED in-battle; double-battle actuation.

### ── 2026-07-07 NIGHT SHIFT #11 — VR 2F-switch2 GHOST-BOULDER solved; the 3F drop detour ──
**CANONICAL unchanged = giovanni_badge8.** run8 aborted LOUD (the new armor worked:
27s to a clean diagnosis instead of run7's silent all-night spin) on "no boulder on
row 19". That was GROUND TRUTH, not a scan miss:
- **THE 0x058 MISREAD (shift-8):** the 2F row-19 boulder (33,19) carries
  FLAG_HIDE_VICTORY_ROAD_2F_BOULDER (0x058) — SET means HIDDEN, and it starts set.
  It only appears after the 3F (33,18) boulder is pushed through hole (34,18):
  HandleBoulderFallThroughHole (field_control_avatar.c:1066) clears the reveal
  flag stored in the boulder object's TRAINER_TYPE field (the pret firered
  falling-boulder idiom, same class as Seafoam).
- **THE DETOUR (BFS-derived, vr3f_probe2/3.py in G:/temp/longrun/pret; wired into
  recon_victory.py f3485dc, dispatches on flag 0x058):** 2F (34,9) ladder (the only
  up-ladder reachable pre-barrier2) → 3F: push (32,5) U2, L21 **along row 3** (the
  naive row-5 line is blocked by trainer Alexa (21,5) — defeated trainers persist
  as walls), D1, L5, D3, R1 → lands (7,7) switch → 3F barrier (12,12-13) opens →
  push (33,18) RIGHT into the hole → jump in after it (beh 0x66) → lands 2F (34,19)
  beside the revealed boulder → row-19 LEFT×19 as already coded.
- **NO BYPASS EXISTS:** the 3F (37,10) warp drops into the 2F east-pocket exit
  region, but its pocket is a SEALED 52-tile island (only entry = (39,17) ladder,
  which is behind barrier2). Boulder (35,13) can only be pushed INTO the pocket
  corridor (36,13) — a permanent SEAL risk; the vehicle never touches it.
**🏅🏅 VICTORY ROAD CLEARED — indigo_reach PROMOTED (15:45, sanctity VALID,
backup pre_indigo_reach_backup_20260707_154528).** Runs 9-11 (all RESUME_STAGE
ratchet): run9 walked the whole detour LIVE (3F switch 40s → drop+hole-jump 203s →
row-19 push → 2F-switch2 OPEN 210s — and the new drain armor fired+recovered twice
en route, run7's killer class is dead) then aborted on the LAST 11 tiles: Ray+Tyra
(38,13)/(39,13) are a trainerbattle_DOUBLE and the battle agent has no
double-target actuation (move → target submenu → stuck ×2 → LOUD). **FILED GAP:
double-battle actuation** (E4 = all singles; fix before any double route). run10
dodged them via column 36 (sight = 1 tile DOWN: avoid (38,14)/(39,14), go_warp
avoid= plumbing) and cleared VR in 9s flat — but banked DIRTY: Indigo Exterior
wasn't in CITY_PC_DOORS so heal_nearest wandered south to R23 (23% lead).
Registered (3,9)→(11,6) + **NURSE_FRONT_OVERRIDES** (the League center is NOT the
shared PC layout; nurse (13,10), stand (13,11)) — run11 healed at the League
center first try and banked CLEAN at (3,9)@(12,19), lead 100%, Venusaur **L66**,
$63,678. Commits f3485dc, 17a5b49.
**e4_run1 postmortem (both walls fixed, beadb1e):** (1) **THE ITEM INSTINCT WAS
ORACLE-GATED OFF in every headless vehicle** — _maybe_use_item returns False when
BattleAgent.choose is None, and no recon vehicle ever passed a chooser; Lorelei
wiped the party with FR×10 UNTOUCHED. recon_e4 now passes a deterministic chooser
(always take the offered heal/cure; choose only routes "battle_item"). ⚠️ BUG
CLASS: any headless vehicle expecting in-battle items has the same gate —
audit before relying on it. FILED: revives are never OFFERED by the instinct
(active-mon heal + status cure only) — the Agatha fodder-revive plan needs that
wire if attrition fails. (2) **post-whiteout FREEZE** — 22 no-move replans at the
center (hidden-modal-box class, dd_box blind to it): walk() freeze armor added
(3 no-move replans → snap frame + blind B/A drain) + fast bail when the map
changes mid-walk.
**IN FLIGHT: e4_run2 (e4_run2.log) — THE CREDITS STRIKE:** heal → kit → Lorelei→
Bruno→Agatha→Lance→GARY → HoF → banked_CREDITS. Watch the Agatha wall (Venusaur
can't touch Levitate-ghosts — sleep-lock + FR attrition; a 2-stuck abort there =
battle-agent move choice vs ghosts, not the vehicle).

### ── 2026-07-07 NIGHT SHIFT #8 — VICTORY ROAD: the three-switch truth; strike iterating ──
**CANONICAL unchanged = giovanni_badge8** (Viridian, ALL EIGHT BADGES, healed). Shift
work = the VR vehicle (recon_victory.py) iterated run2→run6 on look-ahead postmortems:
- **"1F no puzzle" was FALSE** — vr_solve.py was elevation-blind. Elevation-aware
  re-derivation (vr1f_probe1-6.py in %TEMP%/longrun/pret, every pusher stand verified
  per step): EVERY floor barrier opens only by pushing a boulder onto its 0x20
  STRENGTH_BUTTON switch (boulder-lands-on-switch fires the coord event —
  field_control_avatar.c:1076). 1F chain (7,18)→(20,16) [the (11,20) stand is the
  entrance arrow tile 0x65: fires on DOWN only, we press UP]; 2F puzzle1 corrected to
  D1,L2,D1,L2 (old L,L,D,D,L,L was elevation-ILLEGAL); 2F switch2 = row-19 boulder
  (33,19) LEFT×19 → (14,19) (present from game start, FLAG 0x058 clear; the 3F
  hole (34,18) is the game's own RESET-insurance for a wedged push, not a reveal
  requirement). **1F switch + 2F switch1 VERIFIED LIVE (run3/4: barriers open=True).**
- **SILENT FIELD WHITEOUT class:** faint/whiteout boxes get B-drained by
  handle_interrupts → she respawns at the Viridian center with zero log evidence
  (run3's 15s gap). Vehicle rebuilt as ONE whiteout-tolerant dispatch loop —
  progress RATCHETS (Gary var + gauntlet + switch vars persist in the save);
  switches skip-if-open; retreat-heal at Viridian when lead <50% at R23 south
  (the road home is battle-free post-Gary).
- **PHANTOM WARP ANCHORS:** R23's (9,154) gate warp sits on a col-1 tile — only
  walkable warp tiles are real entries (filter added). **DOOR WALK-OUT class:**
  gate exits (6-8,10) are beh-0 warp tiles that never fire on step — standing on
  one, the exit is pressing OUT through the door frame (go_warp handles it now).
- **recon_e4.py BUILT (COMPILES, unrun):** the credits vehicle — League mart
  stock-up (Full Restore×10/Revive×6/Full Heal×4 via tm_errand's true-index buy
  engine, per-unit money+bag verify), generic E4 room handler (rooms discovered
  from RAM warps, trainer = live NPC scan, talk→fight→north door), HoF bank
  (banked_CREDITS) + credits drain. Battle agent's heal instinct already prefers
  Full Restore (id 19 first in _HEAL_ITEMS_PREF). Door guard verbatim-verified
  flavor-only pre-National-Dex.
- Bag truth: NO potions/repels aboard; Viridian mart sells none worth buying.
  Money $86k = the E4 fund (each whiteout halves it — the retreat-heal protects it).
**IN FLIGHT: victory_run6** (victory_run6.log) — full loop with all fixes; on
indigo_reach bank: promote, launch recon_e4.py, THE CREDITS ARE THE NEXT BANK.

### ── 2026-07-07 NIGHT SHIFT #5 — 🏅 BLAINE BADGE 7 PROMOTED; Giovanni strike in flight ──
**CANONICAL = blaine_badge7** (Cinnabar (3,8)@(20,5), BADGES 7, healed, sanctity
VALID, round-trip verified; backup pre_blaine_badge7_backup_20260707_130328;
Venusaur L60). **blaine_run4 won FIRST TRY, 59s door→badge→bank.**
- **THE QUIZ-GYM CLASS SOLVED (recon_blaine.py, aaed50f):** six quiz doors, correct
  answers derived from pret (Q1 Y/Q2-4 N/Q5 Y/Q6 N); ONE drain key per station (B
  advances msgboxes AND selects NO); fail-safe both ways (wrong answer = the room's
  trainer walks to us, winning fires the same door-open). All 6 opened by correct
  answers in 12.5s, zero junior battles.
- **SETMETATILE DOORWAYS READ ELEVATION 0 (run1 truth):** opened-door tiles =
  collision-0/elev-0 beside elev-3 floor; per-TILE elevation laws in strike movers
  are WRONG (elev 0 is the game's wildcard) — Grid.edge_open's per-EDGE law already
  owns it. Removed from the vehicle's walk_path_to.
- **SE-CHUNK LATCH → sleep-lock at neutral damage (battle_agent, 9e3a447,
  E4-CRITICAL):** the lock had demanded our damage be resisted; vs Blaine (Normal x1
  out, fire x2 in, 4-deep potioning roster) it never armed → attrition whiteout loop
  (runs 2-3, 0-for-3). Now: foe class OBSERVED chunking us ≥18% max between decisions
  (above a burn tick) arms the lock on se_threat alone. run4: Sleep Powder neutralised
  the roster, first-try win. VERIFIED live.
- **THE BILL AMBUSH (island scripts truth):** beating Blaine arms a FORCED island
  scene — Bill at the gym doorstep, YES/NO where A-drain = sail to the Sevii Islands
  mid-bank. Post-badge island drains are B-ONLY (verified declined, run4).
- ⚠️ OPERATOR LESSON: a taskkill during run2's bank window cost a won badge (the
  fix was worth it — run4's win is deterministic, not run2's coin-flip). Never kill
  a strike between "badge=True" and "BANKED" — check the log end first.
**IN FLIGHT: recon_giovanni.py run1 (giovanni_run1.log)** — badge 8: sea road home
(five north crossings Cinnabar→R21S→R21N→Pallet→R1→Viridian, seafoam machinery),
gym door self-unlocks at 7 badges, spin_nav gets its second customer (the gym's
spin-tile maze), Giovanni (2,2)→front (2,3). Then: R22 Gary → Victory Road → E4.
Full recipe in NEXT_SESSION.md.

### ── 2026-07-07 NIGHT SHIFT #4 CLOSE (12:40) — 🏅🏅 cinnabar_reach + secret_key PROMOTED ──
**CANONICAL = secret_key** (Cinnabar (3,8)@(8,4), badges 6, SECRET KEY in bag, party
HEALED, sanctity VALID both promotions). Shift arc: Seafoam derivation (offline
meta-BFS on pret bins + scripts.inc) → recon_seafoam runs 1-10 (each wall frame/
stack-diagnosed: dd_box ice livelock → dual-clause; distance-culled boulders;
live-npc masks; B1F-rim elevation law → ENGINE-WIDE in Grid.edge_open; arrow-warp
exit class) → **run10 = 153s Fuchsia→Cinnabar autonomous** (boulder cascade, 0x2D2
layout swap, becalmed surf) → PROMOTED → recon_mansion runs 1-10 (coord-event script
masks; FACING_NORTH statues; anchor-warp truth; stair warps 0x6C-0x6F; the toggle
exit dance + SE back door) → **run10 = 72s door→key→healed→out** → PROMOTED.
Frontier: **BLAINE badge 7** (gym door (20,4) unlockable now; recipe in
NEXT_SESSION.md). Commits 7bb5a31→7a9b52e. Logs: seafoam_run1-10, mansion_run1-10.

### ── 2026-07-07 NIGHT SHIFT #4 (11:20→) — ⚔️ THE SEAFOAM STRIKE (runs 1-5) ──
**CANONICAL unchanged = surf_taught.** recon_seafoam.py = the strike vehicle (commits
7bb5a31, a794e03). **THE CROSSING IS FULLY DERIVED ON PRET GROUND TRUTH** (see
NEXT_SESSION.md for the mission table): no route with currents active (meta-BFS proven);
the B3F current stops when both boulders cascade down the 0x66 hole chain 1F→B1F→B2F
(flags 0x046/0x047 → 0x2D2 → CURRENT_STOPPED layout swap); route = fall B2F hole →
becalmed B3F surf → east ladder chain → F1 exit (32,21) → R20 west sea → Cinnabar.
**WALLS KILLED THIS SHIFT (each frame/stack-diagnosed):**
1. **THE BRIGHT-TILESET dd_box LIVELOCK (run3, py-spy stack + frame grab):**
   dialogue_drive.box_open's 200-threshold read the Seafoam ICE FLOOR (min-channel 239)
   as a permanently-open box → every sea_walk wedged in handle_interrupts/drain forever,
   SILENTLY (no log line — the interrupt path logs nothing). FIX (core-adjacent, flagged):
   threshold 242 — the FRLG message fill is engine-constant 248/255; measured populations
   separate cleanly. mgba scales 5-bit as (c<<3)|(c>>2) → 239 vs 255.
2. **DISTANCE-CULLED BOULDERS (run2):** live gObjectEvents don't exist >~8 tiles out —
   nearest_boulder now template-guided-walks first, live-scans after. Masks: LIVE npc
   body tiles (wanderers park off-template — run4's beach swimmer blocked the same step
   20×) + non-boulder templates; boulder tiles from LIVE coords only (post-push template
   tiles are floor).
3. MB_SAND (0x21) beach strips read walkable ✓ (not water) — R19's shore band is the
   mount seam; step-blocked-by-NPC ≠ step-blocked-by-water (diagnose with the live frame).
**Diagnosis additions to the playbook:** py-spy dump on a silent wedge (pip-installed in
.venv) beats log archaeology; a monitor filter must include the op lines.
Logs: logs/longrun/seafoam_run1-5.log. Mansion prep BANKED: all pret maps cached
(Secret Key = B1F (5,7); statue-toggle setmetatile diffs = collision bit in arg 4;
statues 1F(5,5) 2F(2,16) 3F(12,5) B1F(24,29)/(27,5) bg-events; floors (1,59)-(1,62)).

### ── 2026-07-07 NIGHT SHIFT #2 CLOSE (10:45) — 🏅🏅 safari_hms + surf_taught PROMOTED ──
**CANONICAL = surf_taught** (Fuchsia (3,7)@(33,32), badges 6; Lapras has SURF, Venusaur has
STRENGTH; both promotions sanctity-VALID + round-trip verified). Full story in
NEXT_SESSION.md (frontier-first, current). Shift arc: safari strikes 7-20 (the pond truth,
the dims-equal stale grid, the per-edge elevation law, the West component split — strike 20
landed 50s end-to-end) → HM teach (the 24px slot-anchor fix; Razor Leaf saved from the
forget-fallback) → sea road runs 1-7 (mount actuation PROVEN, reef class killed,
offset-aware crossings) → **THE SEAFOAM SEVERANCE** (R20 surface has no east↔west
connection; the crossing is the multi-floor Seafoam interior = pad_plan class; all warp
pairs billed). Commits: 0c53e3c, 6ae1a22, cb597cc, f1b1a0c, 915c072, c52e096.
**Next objective: Seafoam crossing (port pad_plan) → Cinnabar → Mansion Secret Key →
BLAINE badge 7.**

### ── 2026-07-07 NIGHT SHIFT #2 (09:03→) — 🦁 THE SAFARI POND TRUTH (strikes 7-13) ──
**CANONICAL unchanged = fuchsia_south** (Fuchsia (3,7)@(47,21), badges 6, $71,886, Lapras
L25 slot 5). The safari strike is the live objective; commits 0c53e3c + 6ae1a22.
**FIVE WALLS KILLED (each probe-diagnosed on ground truth, recon_safari_probe.py built):**
1. **THE SHORE TREADMILL (strikes 7/8):** surfable water reads **RAW COLLISION 0** (Gen-3
   gates water by BEHAVIOR) → col==0 BFS planned across the pond; the blocked step's
   sideways nudge landed in grass → encounter → the battle branch skipped dead-marking →
   (35,17)↔(34,17) forever. GENERAL KILLS: **Grid.walkable excludes Grid.water**
   (travel.py — surf planners OR water back in) + walk_path_to dead-marks
   battle-interrupted failed steps.
2. **THE DIMS-EQUAL STALE GRID (strikes 9/10):** Fuchsia and SafariZone_Center have
   IDENTICAL layout dims → the dims-only post-warp Grid guard passed while
   gBackupMapLayout still held Fuchsia content (planning ran on the wrong city; probe
   dumps of staged saves show the same). GENERAL KILL: **the Grid guard now
   CONTENT-VERIFIES** the backup vs the header layout's ROM map (24-sample metatile
   compare, ≥20 match). Harness pitfall class: dims-equal warp pairs.
3. **THE POND TRUTH (strike 11 + pret map.bin):** the Center's WEST doors (8,17-19) are
   **unreachable on foot from the entrance pocket** — the pond splits the map into two
   components (tv.bfs's None was CORRECT). Route rebuilt: the classic tour chain
   **Center → EAST (43,15-17) → Area 1 (NW (8,9-11)) → Area 2 (S (20-22,34)) → Area 3
   West**; return REVERSES the chain (West (40,26-28) lands on the shelf). Warp pairs
   billed from pret map.json (G:\temp\longrun\pret\).
4. **grass steps read as FAILED in _step_to's verify window** (strike 12) → each tile
   cost a replan + a budget try → len-37 walks exhausted tries → the East→Center
   nudge-onto-door bounce loop. FIX: movement never consumes a try (+ 400-hop cap).
   FILED (campaign, shared): _step_to's move-verify window is too short for grass steps.
5. Ball economy (strike 5/6, commit 0c53e3c): ONE throw-pair per SPECIES per run; the
   pay script can auto-warp her in (+ 180 settle frames; paced wedge retries).
**ALSO: recon_cinnabar.py DRAFT** (the sea road Fuchsia→R19→R20→Cinnabar; water-as-road
stepper with the mount toll at land→water edges; surfing detected BEHAVIORALLY = her coord
∈ Grid.water — no gPlayerAvatar reader). UNRUN until safari_hms + surf_taught bank.
**Diagnosis pattern that worked (promote to the playbook):** instrument coords into the
battle/replan lines → kill/relaunch (runs are deterministic from canonical) → probe the
staged save offline (grid dump + flood-frontier classify) → screenshot → pret map.bin.

### ── 2026-07-07 NIGHT SHIFT #1 (new loop, 08:10→) — 🏅 BADGE 6 + LAPRAS + FUCHSIA + the Safari strike ──
**FOUR PROMOTIONS this shift (commits 7631b1c, d6a45f4, 57bd88e):**
1. **sabrina_badge6** — recon_sabrina.py: **pad_plan() = the runtime PAD-GRAPH ROUTER**
   (same-map warps are pads; dest_warp_id indexes the landing; region flood + meta-BFS with
   pad rides as edges — zero hardcoded rooms). Badge in 33s. Two general leak kills: (a) an
   UNBOUNDED region flood escapes the collision-0 border ring; (b) the ELEVATION nibble
   (map-grid bits 12-15) — void strips read collision-0/elev-0 beside elev-3 floor; the game
   blocks cross-elevation steps (the Silph "elevation-sealed" class, now solved generally).
   Probe: recon_sabrina_probe.py (collision+elevation ASCII dump).
2. **lapras_party** — recon_lapras.py: PC-box competency #15 BOTH WAYS. Box RAM reader
   (gPokemonStoragePtr 0x03005010, BoxPokemon 80B, party-identical decrypt), MB_PC (0x83)
   behavior-scan finds ANY Center's console ("shared PC layout" is FALSE for stand tiles),
   and the storage-list CURSOR-MEMORY trap (reopens on DEPOSIT; blind A nearly STORE'd
   Persian). Mankey L10 → box; **LAPRAS L25 in party slot 5** (the Surf carrier).
3. **fuchsia_south** — recon_road.py: the deterministic world-graph road strike (loop
   world.next_step + campaign's own hop executors). Saffron→Fuchsia 21s, zero battles.
   Root truth: with 6 badges head_to_gym targets CINNABAR (island, needs Surf) → the
   longrun stalls in Saffron; recon_road is the standing answer for any learned road.
4. **(in flight at write) the SAFARI STRIKE** — recon_safari.py: pay-gate ($500 trigger row
   y=3), GOLD TEETH (West (28,14)) + HM03 SURF (Secret House) → Warden → HM04 STRENGTH.
   Kills so far: st.in_battle is BLIND to safari battles (no player battle-mon —
   gBattleMons[0] sanity fails; gate on GBATTLE_RES_PTR alone = fight_open()); travel's
   grass-free planner reads no_route where GRASS IS THE ONLY ROAD (step_warp = ride_pad
   pattern on grass-inclusive walk_path_to); battles must not consume walk try-budget;
   safari catches ride the dex doctrine (nidoran + rhyhorn caught by ball-throws at new
   species). Logs: logs/longrun/safari_strike1-5.log.
**CHAIN AFTER SAFARI:** teach SURF→Lapras (hm_teach.teach('surf',5), forget Mist-class) →
Surf actuation R19/R20 (fm.surf_edge_adjacent exists; the water step is the next capability
seam) → Cinnabar → Mansion (Secret Key) → BLAINE badge 7 → Giovanni badge 8 → Victory Road
(Strength in bag) → E4 → credits.
**SOUL-DEBT (carried):** Lapras joined via a PC menu — unmet, unnamed; her first fielded
moment deserves the roster-bond beat. Venusaur still "AAAAAAAAAA" (Name Rater, Lavender).
Safari catches (nidoran/rhyhorn) are silent RAM catches — no judged-catch narration fired.

### ── 2026-07-07 ATTENDED INTERVENTION 07:45 (🔧 frontier repair + the Lapras keyboard) ──
**Shift 11 killed mid-work at Jonny's order (stale-frontier launch). Debris swept:** no orphan
processes; canonical saffron_reach CLEAN; kira/ working-tree changes = the PRE-EXISTING Gemini
vision migration WIP (untouched, not night-shift debris); pokemon-side work salvaged+committed
(1fd4e74) — shift 11's diagnosis was VALID despite the suspect launch.
**THE "7F WEDGE" SOLVED BY ONE FRAME (wedge7f_frame.png): the LAPRAS GIFT NICKNAME KEYBOARD.**
engage()'s A-drains hit "give a nickname? [YES]" (defaults YES) → keyboard opened → dd_box
reads False → drain returns → every overworld step eaten. Same bug class that named Venusaur
"AAAAAAAAAA". Fix: engage(key=) + B-drains end-to-end on the gift (battle_agent's proven
decline pattern) + name_entry(b,"") START→OK escape hatch. Party full → LAPRAS → Bill's PC
(frame-proven "transferred to BILL's" — box-withdraw #15 now has a real customer: the Surf
carrier for Blaine).
**FRONTIER-FILE STALENESS root-caused:** shift 8 died at the context wall MID-STRIKE — the
contract only demanded NEXT_SESSION.md "current at close", so it froze at shift-7 vintage and
shifts 9-11 launched off it. **Contract amended (night_shift.ps1 preface pt 3): rewrite the
frontier BEFORE launching any long strike + at every bank, never only at close.**
NEXT_SESSION.md rewritten to the true frontier. WATCH=1 env added to recon_silph.py (live
pygame window, attended watch). silph_strike16 launched attended with the window up.

### ── 2026-07-07 NIGHT SHIFT #10 (🏢 SILPH CO — the pad chain; route truth from disasm) ──
**CANONICAL unchanged = saffron_reach** (Saffron (3,10)@(47,13), badges 5, TEA, full HP).
**Shift 9's real ending (strikes 8-12, unclosed):** strike12 was the breakthrough — CARD KEY
BANKED IN-RUN at 32.5s via the 9F pad (22,18) → 5F pocket (10,20) → ball (22,21) west front.
Then it hit the 7F wall: from stair arrival (27,4) there is NO path to Gary (2,5) / Lapras
(0,8) / the 11F pad (5,8) — and gary_done/lapras_done LATCHED ON FAILURE, then a full
tower-descent heal ate the clock; the shift died mid-battle on 3F.
**ROUTE TRUTH (pret 3F/7F/9F/11F map.json + strike12 live-learned warps — all cross-checked):**
- **7F's west pocket is PAD-ONLY** (Gary triggers (2,4)/(2,5), rival obj (2,6), Lapras guy
  (0,7), 11F pad (5,8)): no card door reaches x<=5. Entrances: 3F pad landing (5,4), 11F pad.
- **THE PAD CHAIN:** 9F (9,4) ↔ 3F (2,14) · 3F (13,14) ↔ 7F (5,4) · 7F (5,8) ↔ 11F (2,5).
- **9F HAS A FREE FULL HEAL:** hostage woman (2,16), NO hide flag, stays post-strike
  (SilphCo_9F_EventScript_HealWoman). Kills the tower-descent heal class entirely.
- Doors that matter: 3F west (9-10,12-13) / east (20-21,12-13) flank the pad room; 9F west
  corridor (2-3,10-11) gates the heal woman; 11F (5-6,16-17) guards the SOUTH entrance only —
  **Giovanni (6,11) is directly open from the 11F pad landing (2,5)** (triggers (5,15)/(6,15)
  are south-side; grunts' sight columns miss the x2-6 approach).
**STRIKE13 FIXES (recon_silph.py):** phase B = stairs→9F(heal)→pad chain; gary/lapras latch
only on SUCCESS (battle-counter evidence or flag 0x246); 7F-stairs-side routes to the chain
instead of sieging the pocket; heal_mode latch routes over pads to the 9F woman (key in hand);
walk_path_to drains coord-trigger SCENE dialogue (Gary's walk-up) before the battle check;
walk-out opens the 3F east door.
**🔥 THE TAP-TURN GHOST (commit 7634d33 — campaign.py _step_to, probe-diagnosed):** an 8-frame
tap in a direction she isn't FACING only TURNS her; the old "didn't move ⇒ blocked ⇒ nudge
RIGHT" logic UNDID the turn and oscillated forever (or drifted her east across the map — the
strike12/13 "east column wander"). EVERY "elevation-sealed tile" in the Silph siege was this
actuation bug, not geometry. Fix: tap the SAME key again (turn → exactly one step; a 26-frame
hold walks TWO tiles, trace-proven). Harness pitfall #27. ride_pad now also sorts pad-neighbor
approaches by real BFS distance (the fixed order walked a 40-step one-way trap first).
**9F DOOR ALGEBRA (strike14 postmortem + component analysis on the probe grid):** 9F is a
card-door maze; from the pad landing (22,18): stairs already open, heal woman needs the WMID
door (12-13,16-17), the 3F pad needs WMID+WEST (2-3,10-11); NE/pad-room doors never needed.
Real map width is 39 (x31-38 east area is REAL floor, not border). Ed the scientist's
walk-up vacating his 1-wide corridor is how phase A ever entered without doors. heal_mode now
STANDS DOWN after 12 dry attempts (lesson 24 applied to detours). IN FLIGHT: silph_strike15.
**THEN:** promote silph_cleared → longrun goal 0x825 (Sabrina; gym door unblocks; interior =
teleport-pad maze — pads are warps, enter_to-style routing may carry).

### ── 2026-07-07 NIGHT SHIFT #9 (🏢 SILPH CO — the 5F Card Key siege; probe-driven) ──
**CANONICAL unchanged = saffron_reach** (Saffron (3,10)@(47,13), badges 5, TEA, full HP).
**Strike iterations 4→8, each killed by a probe-diagnosed mechanism (all fixed, committed):**
- **strike3/4 lobby↔street livelock:** the Silph 1F entrance mat (8,20) is **0x65 DOWN-arrow**
  (probe-proven, recon_silph_probe.py) — the step-off-arrival-warp guard stepped SOUTH first =
  pressing the mat's own fire direction = out the front door, forever. LAW: **never step off a
  directional warp tile in its fire direction** (fixed in recon_silph + the same latent class
  in campaign.py's questline deeper-hop step-off).
- **strike5 pad rides:** travel BFS pathed ACROSS the 5F teleport pad (10,20) → silently rode
  it to 9F mid-ball-approach. Fix: strike goto() passes travel's existing `avoid=` (the Mt Moon
  mechanism) = warp tiles minus the target. ALSO: Phase A now DESCENDS when above 5F (a pad
  ride left her climbing keyless to a silent 11F dead end); enter_to steps OFF a pad she's
  standing on before re-firing it (pads don't re-fire underfoot).
- **strike6 travel thrash:** static grid BFS is NPC-BLIND — the shortest 5F path threads
  Grunt2's tile (35,7); after she BEATS him the body still stands there, and travel TTL-ages
  the block → replans west → Grunt1 (9,21) seals that → thrash forever. Fix: `walk_path_to`
  (strike-pattern deterministic mover): static BFS with warps+template-NPC tiles masked,
  stepped tile-by-tile, battles recompute. goto = walk_path_to first, travel fallback.
- **COORD-CONVENTION BUG (bug-class sweep done):** `tv.Grid.walkable(sx,sy)` takes SAVE coords
  (adds MAP_OFFSET itself) — callers pre-adding MAP_OFFSET read a tile +7,+7 off. Fixed live
  instances: campaign.py:4273 (questline step-off), recon_silph.py ×2. (recon_captain_door.py /
  recon_ship_2f.py have it too — dead one-shot archives, left.)
- **5F ROUTE TRUTH (probe4 + pret SilphCo_5F/map.json):** live (1,51)=SilphCo_5F exactly (all
  7 warps match: 3F=(1,49) 4F=(1,50) 6F=(1,52) 7F=(1,53) 9F=(1,55) elev=(1,58)). Card Key ball
  (22,21) IS itself a solid object sealing row 21 — **the EAST front (23,21) is the ONLY
  NPC-free-reachable front** (len 41 from the stairs); west front needs Grunt1's tile or the
  pad. Card doors: (7-8,9-10) (7-8,18-19) (18-19,13-14). Strike now tries LEFT/(23,21) first.
**IN FLIGHT AT WRITE: silph_strike8** — expect: Card Key → 3F pad (13,14)→7F → GARY #6 →
Lapras → pad (5,8)→11F → GIOVANNI → 0x3E → bank silph_cleared → promote → longrun goal 0x825
(Sabrina; gym door unblocks; interior = teleport-pad maze, pads are warps, enter_to may carry).
**Probe fleet (kept, reusable):** recon_silph_probe.py (behavior bytes), probe2 (objects+BFS),
probe3 (true ASCII floor map — display fixed), probe4 (path/mask forensics).

### ── 2026-07-07 NIGHT SHIFT #8 (🍵 TEA → 🏙️ SAFFRON ENTERED; Silph strike in flight) ──
**CANONICAL = saffron_reach: Saffron City (3,10)@(47,13) — AT the gym door, badges 5, TEA in
bag, party FULL HP, sanctity VALID** (backups pre_tea_banked_… + pre_saffron_reach_…).
**THE NIGHT'S CHAIN (commits 8cc1aa2 → f5ce337):**
- **Route 12 north gatehouse CROSSED** (the shift-7 frontier): four passthrough bugs killed —
  (a) enter_warp approach max_steps 300→900 (Route 12's pier maze died on the cap and misread
  as entry geometry; new 'no_reach' return), (b) need_heal/timeout no longer burn the candidate
  in `tried`, (c) the CROSSED test is now DOOR-relative (>3 tiles from the entry door), killing
  the rest-house false positive, (d) a `_pt_known` connector that fails to cross is forgotten.
- **WRONG-ENTRANCE RECOVERY (general)**: occupants VISIBLE but BFS-unreachable on an interior =
  sealed sub-region = same building, different entrance. The interact layer remembers the
  interior (`_ql_sibling_dest`); city door pickers skip entered-and-FAILED doors and prefer
  siblings warping into that interior. Proven live: Celadon Mansion back door burned → front
  door (30,11) → the old lady → **FLAG_GOT_TEA set, questline cleared** (sabrina_run5).
- **Saffron = (3,10), NOT (3,11)** (pret map_groups: index 10; (3,11) is SaffronCity_Connection).
  The wrong id sent travel EAST hunting a ghost map. KB corrected; **Sabrina GYMS row billed**
  (door (46,12), Sabrina front (14,12), badge flag 0x825, Center (24,38) in CITY_PC_DOORS;
  interiors are group 14: gym (14,3), Center (14,6)).
- **Want-aware door-passthrough**: candidates sort dest-first (want-map, then UNVISITED
  interiors, then known ground); off-want crossings are usable but never enshrined in
  `_pt_known`. Killed the R8↔R7 Underground-Path ping-pong (sabrina_run6).
- **Bag hash in the longrun stall sig** (recon_longrun): a quest-item acquisition is progress;
  run5 was declared stalled 4 ticks AFTER the Tea landed.
**SAFFRON ENTERED (sabrina_run7): the TEA gate opened, (3,10) bound, she walked to the gym —
`!! GYM: couldn't enter the Sabrina gym` ×14 = RocketGrunt3 (46,13) blocks it until
FLAG_HIDE_SAFFRON_ROCKETS (0x3E), set ONLY by Giovanni's defeat on Silph 11F.**
**IN FLIGHT AT WRITE: recon_silph.py (silph_strike2)** — the hideout/tower-class strike, disasm
truth banked in its docstring: Silph floors (1,47)..(1,57) LIVE-verified; 5F Card Key ball
(22,21) (item 355, pickup flag 0x192); card doors = BG 'sign' events ON the barrier tiles,
A-press opens with the key (per-floor flags 0x27C..0x28D); 3F pad (13,14)→7F (5,4); 7F: GARY
auto-triggers at (2,4)/(2,5) (rival #6 — the foes-seen ledger's live test), LAPRAS free from
(0,7) (flag 0x246, party full → PC), pad (5,8)→11F; 11F: GIOVANNI (6,11) → 0x3E + president
(9,9) hands the MASTER BALL (0x250). Exit reverse pads → heal → bank as silph_cleared.
**THEN:** relaunch the longrun (goal 0x825) — gym door unblocks; the interior is a TELEPORT-PAD
maze (32 warp events, group (14,3)) — pads are warps, enter_to-style routing may just work;
else a gym strike rides the same pattern. Sabrina: Psychic team L37-43; Venusaur L55 + the
NUKE-SLEEP opener should carry; watch for Alakazam's Psychic vs her poison typing (Venusaur is
GRASS/POISON — psychic hits x2; sleep it first).
**CANONICAL = koga_badge5: Fuchsia (3,7)@(9,33), badges 5 ($55k), Venusaur L55, sanctity VALID**
(backup pre_koga_badge5_backup_20260707_050954; party was HURT in-bank — Venusaur/Persian fainted
taking Weezing — but the NEXT leg's first tick heals at the Fuchsia Center, verified in
sabrina_run1/2: SURVIVAL-CRITICAL surfaced 'heal', full restore, then the road).
**THE KOGA KILL (koga_run8, 54s start-to-badge):** prep stand-down latched (2 dry grind attempts
→ the 'train first' plan dropped, loud) → head_to_gym → the ONE junior run3 missed (stale-attach
disarm verified working) → KOGA: **NUKE-SLEEP opener slept BOTH Koffings before they could
Self-Destruct**, Venusaur ground Muk through a mid-fight heal, Fearow finished Weezing.
`GYM: *** KOGA BADGE obtained ***`, badge5 in the fp, sanctity VALID, PROMOTED.
**BADGE-6 LEG OPEN (d373ccd):** roads/"Saffron City" billed from live-learned edges (zero
fabrication): Fuchsia→R15(e)→R14(e)→R13(n)→R12(n)→Lavender(w)→Route 8(w,pass)→Saffron (3,11
expected). exit_gates 3,26/west = the thirsty-guard TEA gate (FLAG_GOT_TEA 0x2A6; the
capabilities entry — Celadon Mansion old lady — was already billed). **sabrina_run2 IN FLIGHT
at write:** billed road verified carrying her R15→R14→R13→R12 northbound.
**KNOWN NEXT WALLS (in order):** (a) the Route 8 gatehouse TEA errand — questline must route
Route 8→Celadon via the Underground Path (door-passthrough, proven both ways) and back;
(b) **Sabrina's gym door is Rocket-BLOCKED until Silph Co. is cleared** (Card Key 5F, Giovanni #2
11F — recon_hideout/recon_tower are the strike pattern; Gary #6 is in Silph too — the attach-time
rival fix will get its live test); (c) no GYMS row for Sabrina yet (needs Saffron door/leader
coords from disasm when she arrives); (d) Saffron gym interior is a TELEPORT-PAD maze — likely
strike material, pads are warp events (read_warps sees them).

### ── 2026-07-07 NIGHT SHIFT #7 (🌆 FUCHSIA REACHED — Koga engaged; the Self-Destruct lesson) ──
**CANONICAL = fuchsia_reach: Route 15 (3,33)@(0,12), Venusaur L54, badges 4, party HEALED,
sanctity VALID** (backup pre_fuchsia_reach_backup_20260707_045518). Routes 12→13→14→15 fought
through, FUCHSIA REACHED, gym entered, **ALL SIX Koga juniors BEATEN** — then KOGA WIPED HER:
his L37 Koffing **SELF-DESTRUCTED on Venusaur turn one** (the ace one-shot at ANY dominance)
and the dead-weight bench (Pers36/Fear34/Rati31/Ekans15/Mankey10) fed itself to Muk/Weezing
(Muk down to 37HP — close even benched). Gym flags: juniors stay beaten; only Koga remains.
**FIXES SHIPPED (c93f10b + the grind-spin follow-up), all general engine assets:**
- **NUKE-SLEEP OPENER** (battle_agent `_NUKE_SPECIES`): Self-Destruct family (Geodude/Voltorb/
  Koffing lines) gets slept BEFORE it can detonate-trade, at ANY damage matchup — the old
  sleep-lock only fired when resisted+threatened, exactly why it sat out vs an x1 Koffing.
- **FOES-SEEN LEDGER + attach-time rival re-check** (kills the filed tower4 bug): engine records
  every foe read LIVE at action menus (`battle_agent.LAST_FOES_SEEN`, per-battle reset); campaign
  re-checks it post-battle against the rival counter-line (recomputed from her starter — the
  erika_run1 Ivysaur false-positive stays dead). No more manual Gary backfills.
- **ASYNC-WHITEOUT GUARD** (`_exit_to_overworld`): the pending whiteout warp fired MID-candidate
  (Koga loss → respawn while the exit leg walked) → every later candidate was a stale gym coord
  on the Center map (wedge ×6). Map-changed is now checked unconditionally per candidate.
- **GRASS-TARGET FAIL MEMORY** (`_grass_target`): (from-map → target) no_route failures
  remembered in-RAM; next tick tries a DIFFERENT grass (killed the 14-tick Route-15 stall —
  she sat in the one-way ledge pocket re-proposing east forever).
- **GRIND NO-GRASS SPIN** (koga_run4): grind()'s no-grass branch `break`→"ok" made GRIND-WEAK
  retry in a tight loop (celadon_run3 class, reborn on water Route (3,37)) — now returns
  `no_safe_grass` + records `_grind_dead`; grind-dead maps are vetoed as grass CANDIDATES in
  all three _grass_target sources (the optimistic "group-3 route = grass" test sent her to the
  grassless water route in the first place).
**GEOGRAPHY TRUTH (Fuchsia pocket):** Route 15's west end (0,12) is a one-way ledge pocket —
NO route east, NO reachable grass; water Route 19 south (3,37) has NO grass. The trainable
grass is WEST of Fuchsia (Route 18) — the graph-BFS finds it once the dead maps are vetoed.
Fuchsia Center = map (11,5), door row y=8, only (7,8) fires (0x65). Gym = (11,3), Koga front
(7,14), juniors objs [1,2,3,5,6,7].
**IN FLIGHT AT WRITE: koga_run5** (goal 0x824, logs/longrun/koga_run5.log) — expected line:
grind the bench on Route 18's grass (UNDERLEVEL-PREP wants floor ~L34) → re-enter the gym →
KOGA with the nuke-sleep opener. Watch for: the wall-gate on map (11,3) possibly vetoing the
retry (loss recorded there — if head_to_gym refuses, arm the strike pattern: recon_tower is
the template; disasm FuchsiaCity_Gym).

### ── 2026-07-07 NIGHT SHIFT #6 (🔭 SILPH SCOPE BANKED — hideout COMPLETE; spin crosser generalized) ──
**CANONICAL = CELADON (3,6)@(34,22), silph_scope_out PROMOTED** (chain tonight: silph_scope
[B4F, scope in bag] → silph_scope_out [street, HEALED at the Celadon Center]; backups pre_* both;
sanctity VALID both). Key Items now: [...TM case, 356 LIFT KEY, **359 SILPH SCOPE**]. Party
Venusaur47/Persian35/Fearow33/Raticate31/Ekans15/Mankey10, badges 4. Commit 121be05.
**THE B2F ELEVATOR WALL — full postmortem (hideout10-12 + recon_b2f_sim.py):**
(a) B1F is SPLIT by a full wall at y19-20; its elevator doors (23-25,25) live in the sealed SOUTH
half (served only by B2F's (23,12) stairs) — the B1F approach is a dead end from the north.
(b) The B2F east room holding the elevator (28-29,16, 0x69 doors) is entered THROUGH the west
spin maze — the old node_ok x>=12 "east bias" forbade exactly those routes. Offline glide-BFS
over a dumped behavior grid (grid_b2f.txt) found 308 rest tiles + a 46-press route; verified live.
(c) The maze is SEALED by a collectible ITEM BALL (MOON STONE at (2,5) — now in bag): with it
standing there is NO route; the game expects you to pick it up. The crosser now collects
reachable balls and replans.
(d) Live object reads are DISTANCE-CULLED (gObjectEvents): far balls are invisible to planning.
NEW **travel.read_object_templates(b)** reads the map's STATIC templates + spawn-flag truth
(MapHeader.events +0/+4, stride 0x18; ball gfx=0x5C; verified vs disasm on B2F) — planners union
it with _npc_tiles.
**SPIN CROSSER EXTRACTED → `spin_nav.py` (SpinNav)** — the general asset: glide sim + rest-tile
BFS + ball-collect sweep + BOX-DRAIN guard (a beaten grunt's re-talk box eats every direction
press — the exit1 wedge; drain before each press) + twice-failed-step avoidance (wandering-NPC
squatting) with relax-once. recon_hideout.py + recon_hideout_exit.py ride it. **Viridian Gym's
spin maze rides this later; wire into travel/campaign when a longrun bites (portability debt).**
**GIOVANNI BEATEN** (Onix/Rhyhorn/Kangaskhan, Razor Leaf x4s) + both door grunts; scope ball
pressed; elevator ride CODED both directions (panel bg (0,2); DOWN presses to descend, UP to
ascend; self-correcting landing). recon_hideout_exit.py = the walk-out vehicle (elevator → maze →
GC → street → heal → bank) so canonical NEVER sits mid-dungeon behind a spin maze generic nav
can't cross.
**🎺 THE FLUTE CHAIN FELL TOO (same shift): TOWER STRIKE (c40bc72, recon_tower.py) → POKE
FLUTE PROMOTED → SNORLAX WOKEN (c6cb914, recon_snorlax.py) → the coastal road is OPEN.**
- flute_run13 truth: the questline door-hint entered the Tower + go-deepered to 2F, but GARY
  (object (16,5), battle needs the multi-box talk escalation) wedged the tour. recon_tower:
  approach legs (Celadon→R7→UGP#2→R8→Lavender, enter_to = warp-dest routing on read_warps —
  NO hardcoded stair coords), climb as a STATE MACHINE on current-map (heal bounces just
  re-dispatch), GARY beaten (rival ledger now 3W-2L — the win needed a BACKFILL: the observer
  ATTACHED MID-BATTLE (scene fired during approach travel) and rival detection missed —
  **FILED BUG: rival detection misses attach-time battles**), ghost MAROWAK beaten en route
  (Scope = normal wild), 7F grunts, FUJI → his house → FLUTE (item 350 + flag 0x23D).
  GENERAL KILL (campaign.py): _enter_directional_warp's on-tile mount presses ×3 (first press
  eaten as TURN — Tower 2F 0x6D stair, both side stand-tiles walled, fires from ON the tile).
- koga_run1 walked to Route 12's north gate and stalled — the "Snorlax blocker" is at (14,70)
  (disasm), 70 tiles south; head_to_gym couldn't route past the body. recon_snorlax: gate
  pass-through (north doors (14,15)/(15,15)→(23,0), prefer='south' entry) → road gauntlet →
  face (14,70) + A → flute plays → wild L30 Snorlax ATTACKS → beaten → flag 0x253 + body gone.
- PROMOTION CHAIN tonight (all sanctity VALID): silph_scope → silph_scope_out → poke_flute →
  route12_snorlax → **snorlax_woken = CANONICAL: Route 12 (3,30)@(14,69), Venusaur L51,
  badges 4, Scope+Flute+Lift Key banked, rival 3W-2L.**
**IN FLIGHT AT WRITE: koga_run2** (goal 0x824, logs/longrun/koga_run2.log) — Routes 12-15 →
Fuchsia → KOGA (GYMS row EXISTS — campaign.py:323, disasm coords, flag 0x824; the shift-5
handoff wrongly said missing). Watch: Fuchsia city id (3,7) is EXPECTED-binds-on-walk; Koga's
gym is invisible-walls (RAM collision truthful per the registry note).

### ── 2026-07-07 NIGHT SHIFT #5 (👻 GHOST-VILEPLUME killed; Secret Power taught; hideout 80% cracked) ──
**CANONICAL = CELADON (3,6)@(11,15), tm43_secret_power PROMOTED** (backup pre_tm43_secret_power_
backup_20260707_025255). Party HEALED, badges 4, $13,378, sanctity VALID. **Venusaur [RazorLeaf,
PoisonPowder, SleepPowder, SECRET POWER 70-neutral] · Fearow [SECRET POWER (STAB ~105eff), Growl,
Leer, FuryAtk]** — the 0.25x-only moveset famine is CURED at the definition.
**👻 THE BIG KILL (run-9 frame+RAM diagnosis): the "Route 8 Vileplume trainer" of runs 7/8/9 NEVER
EXISTED.** The anchor-first warp step legitimately ends a tick INSIDE the UGP hut → the tick-top
"indoors = blackout" heuristic fired on normal transit → note_blackout attributed a loss to the
STALE last_foe (ERIKA'S gym vileplume, still in the save's RAM + strat sidecar) → phantom wall
gated Route 8 → bench-grind plan spawned → ejection rebuilt the hut hop-loop. Money never halved.
FIX (15b63f6): blackout branch demands BATTLE EVIDENCE (_battle_ran_this_action, set by
_observed_battle_runner, consumed per tick). **LEDGER LESSON: a "wall" diagnosis needs a battle
trace + money check — the whiteout-backstop can launder stale RAM into a convincing story.**
**TM ERRAND (d3d7dee, recon_tm_errand.py — reusable TM buy+teach vehicle, VERIFIED end-to-end):**
Dept Store door (11,14)→1F stairs (4,2)=0x6d→2F TM clerk behind the 0x80 counter col x=2
(customer at (3,6) faces WEST — counters forward A) → buy on TRUE INDEX = selectedRow+scrollOffset
(sShopData 0x02039934 +0xC/+0xE; row alone bought TM05 via CANCEL) → TeachFlow item/move
overrides → bank → PROMOTED. 2F stock: TMs [TM05 1000, TM15 7500, TM28 2000, TM31 3000, **TM43
3000**, TM45]; items clerk (1,8) has REVIVES (9 rows). GENERAL KILLS: enter_warp directional-door
fallback (0x62-0x6F fire only entered THEIR way) even when the approach tile is unreachable
(hut-exit-mat class, c3a3623) + MAT-ROW fallback (warp TABLE lists side tiles with behavior 0x00 —
only the center 0x65 mat actually fires; run-11 wedge); teach() START OPEN-VERIFY (menu cursor
bytes are stale across sessions); campaign shop nav on true index; hm_teach.tm_compatible().
**🏰 ROCKET HIDEOUT STRIKE (fb51898, recon_hideout.py — deterministic 40s from canonical, re-run
to reproduce):** poster grunt beaten → POSTER PRESSED (stairs unlock) → B1F→B2F→B3F → **SPIN MAZE
CROSSED (the slide crosser — the billed next build, DONE)** → B4F → **LIFT KEY IN BAG (356)** →
climbed back to B2F. SLIDE MODEL truth: spinners 0x54-0x57 redirect, 0x58 dots/walls stop, PLAIN
FLOOR DOES NOT STOP A SLIDE (momentum), a press on a wall-stopped spinner resumes ITS dir; NPC
tiles block; replan after mid-route battles; border-bounds guard (Grid wraps at x=0).
**⛔ THE LIVE FRONTIER: B4F's boss corridor (grunts (16,14)/(19,14), GIOVANNI (19,4), SCOPE ball
(20,5)) is WALLED OFF on foot — elevator-only (its B4F door (20-21,23) is inside the corridor).
The elevator ride is CODED (panel bg (0,2) in the elevator map; floor multichoice; self-correcting
landing check; wrong-floor re-board via nearest door) but UNREACHED: the B2F glide graph finds NO
route from (15,8)/(21,2) to the elevator (28-29,16). NEXT MOVE: re-run recon_hideout to B2F, dump
the full behavior grid + frame there, hand-derive the elevator entry (likely a walk corridor the
glide BFS can't see, or the alcove needs approach from a specific row), fix, complete: ride → beat
2 grunts → Giovanni → scope ball → verify ('item',359) → bank banked_SCOPE → promote.** Then the
longrun (goal 0x23D) drives Tower → Fuji → flute; wake Snorlax; Koga (GYMS row still missing!).
**Longrun gaps surfaced (run-12, owed):** the questline's "north-most warp" heuristic tours the
WRONG building repeatedly (Celadon mansion ↔ city loop = the run-12 stall) — needs entered-door
memory; the destination interaction needs KB interior RITUALS (grunt/poster class) — the strike
is the scripted proof + spec. Moveset-gap auto-arbitration still not wired general (errand = the
vehicle). Venusaur still "AAAAAAAAAA".

### ── 2026-07-07 NIGHT SHIFT #4 (🏅 BADGE 4 CANONICAL — Erika down; the flute chain armed) ──
**CANONICAL = CELADON, BADGES 4 (3,6)@(11,31)** (promoted erika_badge4; backup
pre_erika_badge4_backup_20260707_014247). Party Venusaur46/Persian35/Fearow33/Raticate31/Ekans15/
Mankey10 — Venusaur+Persian rode out at 0 HP post-gym (she heals at the registered Celadon Center
(48,11) on the next heal pick; NOT wedged). Dex 11, $19,378.
**THE ERIKA KILL CHAIN (erika_run2 postmortem → 3 general fixes → badge in 117.8s):**
(a) **PP-FAMINE arbitration** (3b99978): read_party_pp + slot_has_damaging_pp (Attacks substruct
word 2, ROM gBattleMoves power) → in-battle PP-FAMINE SWITCH (alive-but-winless active → best
reserve WITH damaging PP, once/species/battle); heal-before-the-gym gate (_party_gym_ready:
faints/ace<50%/famine → Center first — VERIFIED firing tick 1 of erika_run3/4); gauntlet famine
break → needs_heal (juniors stay beaten); a stuck leader fight NEVER falls through to the award
drain while the battle is live (the 'Do what with this POKéMON?' 300-step A-mash class);
_drain_overworld refuses to drain during a live battle.
(b) **LEVEL-DOMINANCE VETO** (deefab3): the matchup switch fielded L15 Ekans ("resists most") into
L21+ gym mons and she BLACKED OUT to a Gloom she outleveled by 24 → active ≥ foe+10 never
switches; candidates >5 under the foe are fodder; level breaks type ties.
(c) Erika front (6,5) was already ground-truthed (shift 3); the interior CUT trees ((6,8) center
aisle) are handled by travel's in-leg field_clear — RATICATE used CUT, twice, live.
**BADGE-5 LEG (in flight at close — read logs/longrun/flute_runN.log):** next_gym = Koga; both
Fuchsia roads are Snorlax-locked → the KB chain silph_scope → Poké Flute → wake Snorlax is armed by
the new Lavender-south exit gate. KB billed (463747f): Fuchsia road (Celadon→R7→UGP#2→R8→Lavender→
R12-15, expected ids 3,30-3,33 + Fuchsia 3,7 bind-on-walk). QUESTLINE ENGINE HARDENED (2068e83 +
3da5894, flute_run1/2 postmortems): **item-confirmed steps** (('item',id) success via key-items
pocket — FLAG_GOT_SILPH_SCOPE 0x037 is HIDE-class and READ SET SPURIOUSLY, skipping the whole
hideout); **door hints** (Game Corner = Celadon door (34,21)→interior (10,14); Tower = Lavender
(18,6) — disasm map.json truth); **step anchors** (a chain's errands span towns — dock/re-anchor/
past-anchor now use the STEP's from-map, reset per step); **ANCHOR-FIRST routing** (compass dirs
apply only on/past the anchor — killed the Lavender↔Route-10 ping-pong); **no_connector guard** on
quest door-entry + GO-DEEPER (she toured INTO Rock Tunnel hunting Mr. Fuji).
**ROCKET HIDEOUT GROUND TRUTH (disasm, for the strike if the generic tour stalls):** Game Corner
interior (10,14): grunt (11,2) guards poster sign (11,1) — beat grunt, press poster → stairs
(15,2)→B1F. B1F: down (17,2)+(15,30), elevator (23-25,25). B2F: arrive (28,2), down-to-B3F (21,2),
elevator (28-29,16) — SPIN-TILE maze floor. B3F: arrive (18,2), down-to-B4F (15,18). B4F: arrive
(11,15); GIOVANNI (19,4); Silph Scope ball (20,5); Lift Key ball (3,2) by grunt (4,2); grunts
(16,14)/(19,14) guard the boss door; elevator (20-21,23). Item ids (disasm): SILPH_SCOPE=359,
LIFT_KEY=356, POKE_FLUTE=350, CARD_KEY=355, SECRET_KEY=351, GOLD_TEETH=353, TEA=369, BICYCLE=360.
**KNOWN RISK AHEAD:** B2F/B3F spin-arrow tiles — travel's BFS doesn't model forced slides; if the
GO-DEEPER tour can't reach the down-stairs, build a slide-aware edge model (deterministic: arrow →
slide to stop tile; add as graph edges). The Tower needs the Scope FIRST (ghost Marowak floor 6);
channelers are normal trainer fights.
**CANONICAL = CELADON CITY (3,6)@(59,15)** (promotion chain tonight, every step sanctity-gated:
dex11_mankey → hm05_flash → rocktunnel_lavender → celadon_reach; backups pre_* alongside).
Party Venusaur45(FLASH)/Persian34/Fearow32/Raticate31/Ekans15(fainted-benchwarmer)/Mankey10; dex 11.
THE NIGHT'S ARC: Voltorb-hunt bug (stale-foe judgment) killed → judged Mankey catch (dex 11) →
Diglett's Cave crossed (3 nav kills) → Route 2 aide (warp-truth gate hunt) → HM05 → FLASH taught
to Venusaur (case-sort fix) → ROCK TUNNEL crossed LIT (use_field_move debut, section-DFS maze) →
Lavender (Center registered) → Route 8 → UGP #2 under Saffron → Route 7 → CELADON (Center (48,11)
+ gym (11,30) + Erika registered — front tile (4,4) is a first guess, badge run verifies).
ERIKA RUN (GOAL_FLAG=0x823) in flight at last update — read logs/longrun/erika_run1.log.
LATE-SESSION KILLS (committed): plan HYSTERESIS (replan tie-flip churn), proactive-bench sitting
cap (+6 bites) + grind-dead map memory + no_safe_grass honesty, tunnel banned as passthrough
connector, tunnel gate segment-aware (only_when_y_below), party-move read-twice hardening +
teach-bridge settled re-check, CRITICAL=fighting-core (bench faint ≠ heal-spin; the east_run2/
celadon_run7 stall class killed at the definition), heal-excursion honest fail, Lavender+Celadon
Centers registered. SOUL DEBT noted: the ace Venusaur is nicknamed "AAAAAAAAAA" (Name Rater is in
Lavender; owning-it beat vs rename = oracle/Jonny call).

**Canonical = DEX 11 / Route 4 (3,22)@(75,17)** (promoted dex11_mankey; backup
pre_dex11_mankey_backup_20260706_235322; party 6/6 Venusaur42/Persian33/Fearow32/Raticate31/
Ekans15/**Mankey10** — the judged fighting-coverage catch; sanctity VALID).
**KILLS (committed df3c208):** (a) CATCH-JUDGMENT STALE-FOE — judgment read gBattleMons[1] at
encounter hand-off = the PREVIOUS battle's foe (voltorb_run1 judged "voltorb L14", balled a 2nd
ekans while voicing electric coverage); foe source now gEnemyParty[0] (+read_enemy_level).
VERIFIED in voltorb_run2 (all judgments matched engine species). (b) SLEEP-THEN-THROW on
10+level-gap catches (pure sleep = 0 KO risk, x2 rate) — mankey caught with it live. (c)
BALL-LESS TEETH end-to-end: _shopping_list ball floor (<2), mart-first travel_targets on empty
pocket, longrun chooser rides the Mart travel option (verified: last ball broke on voltorb →
she walked to Cerulean, healed, stocked, returned, caught). (d) **ROM TM/HM COMPAT** —
gTMHMLearnsets @0x08252BC8 (recon_tmhm_scan.py, 14-fact control); retired _CUT_OK hand-table;
**shift-1's "none of her six learns Flash" was WRONG: Venusaur AND Persian are Flash-compatible
in FRLG** → no Voltorb needed; teach goes to Persian. (e) GRID MID-TRANSITION GUARD (travel.py)
— gBackupMapLayout keeps the OLD map's dims ~120 frames post-warp; a Grid built then reads
collision at the wrong stride (the Diglett's-Cave phantom-wall wedge class). Grid now spins
until backup tracks gMapHeader->mapLayout. (f) travel no-path retries rebuild the grid.
(g) cross_cave is DESTINATION-AWARE (read_warps dest truth: prefer the overworld door that
isn't the way we came; visited-memory kills ladder ping-pong) + door-MAT step-through nudge
(mats fire on the crossing step; presses need owner="agent" — the phantom-drop lesson).
**IN FLIGHT:** recon_hm05 run 6 (Route 4→Vermilion→Route 11→Diglett's Cave→Route 2 aide→HM05
→teach Flash→Persian→return→bank). **BUILT, UNRUN:** recon_rocktunnel.py (heal at R10 Center →
north mouth → use_field_move FLASH verified vs 0x806 → destination-aware maze cross → south
mouth → Lavender bank) + TeachFlow.use_field_move (START→POKEMON→slot→submenu row-k, RAM-flag
verified — the Flash-use primitive, party-menu class). NEXT: promote hm05 bank → rocktunnel
strike → promote → badge-4 road (Lavender→Route 8→UGP #2→Celadon→Erika, goal flag 0x823).

### ── 2026-07-07 NIGHT SHIFT #1 (BILLED ROADS + the Flash chain) ──
**Rope now ends at: 🛣️ ROUTE 10 (3,28), dex 10→(11?), PC-box deposit CRACKED.** Canonical promotions
tonight (all sanctity-gated): `route10_reach` → `dex10` (Ekans caught, party was 6/6) → `pcbox_spearow`
(dupe Spearow deposited at the R10 Center, party 5/6 — the slot freed FOR VOLTORB, the Flash mon).
Backups: `pre_route10_reach/ pre_dex10/ pre_pcbox_spearow_backup_*`.

**EAST_RUN2 ROOT CAUSE (killed):** head_to_gym's forward model was "base camp = GYM_SPINE predecessor +
march SOUTH" — badge-1..3 geometry. Badge-4's road bends EAST from CERULEAN, so she was sent back to
Vermilion from Cerulean forever (no_gym_route → 14-decision STALL). ALSO: a bench-faint to a wild oddish
recorded a SPATIAL wall that gated Route 6 and marked Vermilion "BLOCKED" in her world brief.

**NEW ENGINE PIECES (committed 6e5b88f, 7e181aa, b877034 — all VERIFIED via look-ahead):**
1. **BILLED ROADS** (KB `roads` + `campaign._gym_road/_road_step/_road_pull_anchor`): per-gym road as
   ordered legs (edge dirs + UGP pass-throughs), map ids LIVE-BOUND from the learned graph one map ahead.
   Celadon road billed Vermilion→…→Celadon. VERIFIED: road_run1 GOAL Vermilion→Route 10 in 41s/7 decisions.
2. **Wild losses no longer gate ROUTING** (trainer walls only — is_gated/_wall_avoid/_wall_blocked_dirs).
3. **CAPABILITY EXIT-GATES** (questline.exit_gate): `flag` may name an HM cap ('flash'), satisfied when a
   party mon KNOWS the move. Rock Tunnel billed: exit_gates 3,28/south, PITCH DARK, gate='flash'. The road
   follower checks gates BEFORE crossing (`road_gated` — VERIFIED she refuses the dark tunnel).
4. **LONGRUN_GOAL_DEX=N** + dex-goal chooser (wander_catch priority). VERIFIED: dex10 GOAL in 9.9s.
5. **PC-BOX DEPOSIT** (recon_pcbox.py — Tier-2 #15 first slice): PC console at interior (11,2) stand
   (11,3); storage system = a LIST MENU over the room (not full-screen until DEPOSIT chosen); pixel-probe
   driven; verify party count AFTER exit. VERIFIED live (6→5).
6. **Route 10 ground truth:** PC door (13,20) (now in CITY_PC_DOORS); ROCK TUNNEL = map (1,81), mouths
   (8,19) N / (8,57) S on R10.

**THE FLASH CHAIN (in flight at close — the badge-4 critical path):** none of her six could learn Flash
(Gen-3 compat) → boxed the dupe Spearow → catch VOLTORB on R10 (dex-11 run) → teach IT Flash (hm_teach;
Voltorb L14-16 has a free slot) → then the ROCK TUNNEL strike (cave nav machinery from Mt Moon; Flash-use
flow still unbuilt — party menu → Voltorb → FLASH inside the tunnel). `recon_hm05.py` (WRITTEN, UNRUN —
COMPILES only) fetches HM05: road home → Diglett's Cave crossing (2-warps-per-room walker) → aide
gatehouse (needs dex≥10 ✓) → teach → return → bank. Run it AFTER the Voltorb catch banks.

### ── 2026-07-06 STRIKE (the SOUTH GATE + soul-truth session) ──
**Rope now ends at: 🚪 ROUTE 5 — THE SOUTH GATE IS OPEN (run-3 `logs/longrun/south_run3.log`, GOAL in
190s/4 decisions/0 stalls). Canonical PROMOTED to Route 5 (3,23)@(32,0)** (backup
`states/campaign/pre_route5_backup_20260706_072125/`; sanctity-gated promotion, monotonic story check
PASS). Run 4 (nursery + Vermilion push) in flight at session close — read `logs/longrun/nursery_run4.log`.

**THE SOUTH-GATE GROUND TRUTH (recon_south_geometry.py + pret map.json + Bulbapedia, all agree):**
post-ticket, the cut tree at (26,32) STAYS — it is the ONLY base-collision fence gap; there is NO ledge
(the "green strip" is a hedge; behaviors dumped, zero MB_JUMP on the row). The pre-Cut route south is
**THROUGH THE BURGLED HOUSE** — front door (30,11) (freed by the ticket flag) → interior (7,1) → back
door → the fenced garden (Rocket grunt, TM28 fight — she beat him) → the east corridor (cols 39-40) →
the fence crossing at (39-40,32) → the south strip → Route 5. The plaza can NEVER reach this on foot
(flood-proven NONE with the tree object modeled).

**NEW ENGINE PRIMITIVES (all general, all look-ahead-verified tonight):**
1. **DOOR PASS-THROUGH (`campaign._door_passthrough` + `_edge_travel`)** — when an edge crossing reports
   a hard no-route (fenced region), try reachable doors as CONNECTORS: enter, multi-hop the interior
   warps (farthest-from-spawn heuristic, bounded 6 hops — depth 1 = the burgled house, depth 3 = the
   Underground Path huts), pop out elsewhere, retry the edge. Connector-fingerprint priority (a building
   with SEVERAL city-side warps — the burgled house has 3 — is the likeliest pass-through, tried first);
   proven connectors remembered per map (`_pt_known`) so heal/battle interrupts reuse instead of
   re-searching; interiors that dead-end (Center upstairs) are backed out of via `_exit_to_overworld`.
   Wired into ALL edge hops (EDGE-ROUTE / FORWARD-SPINE / south-block / travel:*). This ALSO covers the
   Saffron gatehouse + UGP shapes ahead.
2. **FIELD-OBSTACLE HONESTY + MEMORY (travel.py)** — a cut tree / Strength boulder on the only gap is
   classified by graphicsId (95/97), voiced honestly ("that's one of those small trees — the kind you
   can CUT down…"), remembered as a HARD BLOCK (`blocked_npcs`) and REROUTED around in-leg. Root fix for
   the crossed→tree→crossed loop of run 2: the NPC-allowing BFS prefers the SHORTEST path, so the near
   tree beat the real long detour every replan until the tree became a remembered block.
3. **BATTLE-LOOP BREAKER (travel.py)** — 3 consecutive `stuck` battle outcomes with the battle still
   open = abort the leg LOUD (the ×27 immortal-re-entry spin of run 1, killed as a class). Cause also
   fixed: `catch_pokemon` no_balls paths now FLEE the live battle before returning (an abandoned battle
   re-detects as a fresh encounter forever).
4. **ROSTER-SELECTION JUDGMENT (`pokemon_strategy.roster_judgment` + the catch_one hook,
   `POKEMON_CATCH_JUDGMENT=1`)** — soul-debt #3's choice framework: dupe / type-coverage / level / room,
   first-person REASON both ways, oracle decides live (headless follows the lean), skip-voice throttled
   per species. Unit-verified 5/5 (dupe-skip, coverage-catch, scrawny-skip…). Plus **PROACTIVE BENCH**
   (`POKEMON_PROACTIVE_BENCH=1`): wall-less prep target lead-8 when the floor sags >10 under the lead.
   Plus the **bond-hook fix**: ANY pick ending in a catch now fires roster_react (a travel:-pick catch
   was silently skipping naming/bonds — why soul.json stayed empty).

**RIDE-ALONGS SHIPPED (constitution teeth):** (a) false "[evolve]" beat FIXED — PID discriminator
(evolution keeps the mon's personality value; a party reorder swaps it) in `_soul_after_objective` +
play_live's battle watch; verified 3/3 (`recon_evolve_disc.py`). (b) **JOURNEY BACKFILL** — the run-5
Gary win (1W-2L) was missing from canonical strat/journey (runs 6-8 restarted from a pre-win bank and
the promotion regressed her story); backfilled from gary_run5.log ground truth into strat_memory +
regenerated journey_core (campaign-side AND core-side `states/kira/journey_core.json`, + a weight-4.0
saga beat with run-5's real timestamp). Her proudest moment is back in her memory. NOTE: the reported
"mojibake" was MY read artifact (default-encoding open() on Windows) — the files were always clean
UTF-8; all writers/readers verified explicit utf-8. (c) **SANCTITY VALIDATOR (`sanctity.py`)** — bank-time
schema/encoding/truth/monotonic validation; the monotonic rival-count rule catches the exact lost-win
class; wired into recon_longrun banks (DO-NOT-PROMOTE loud fail) + campaign continuity saves + the
promotion script. Ran live on runs 2-3: VALID. (d) **SLEEP_LOCK default ON** (battle_agent) — the gating
reason (move-list wedge) is fixed and the whiff cap bounds the worst case; a live GO now gets the real
strategy. (e) `recon_groundtruth.py` pocket-aware fix — pitfall #7 claimed the frontier report: she had
**5 Poké Balls** (balls pocket) + **a Nugget** (sellable 5000₽, no sell flow yet) all along.

**KB DATA-BILLED (gamedata/frlg_gates.json):** every gate to credits — HM01-05 flags (0x237-0x23B),
Bike Voucher/Bicycle, Poké Flute chain (Scope→Fuji→Flute→Snorlax 0x253), Tea (0x2A6), Gold Teeth→
Strength, Safari Surf, badge flags — directions+flags format, hide-flag ids (Scope 0x037/Lift Key 0x036)
caveated, + route_notes (UGP paths, Saffron unlock, Cinnabar/VR/E4). Deriver loads it clean.

**LATE-SESSION ADDENDUM (runs 4-10, `logs/longrun/nursery_run*.log`) — the road to Vermilion, walked to
Route 6:** run 10 PROVED the full autonomous chain **Cerulean → burgled-house pass-through → Route 5 →
region re-entry (offset +0, cols aligned) → the Underground Path north entrance (building at Route 5
(31,31) → interior (1,30)) → the tunnel doorway (7,4), a 0x6F DOWN-LEFT STAIR warp entered moving WEST
(`_enter_directional_warp`, the `_WARP_ENTRY` table) → the tunnel → out on ROUTE 6 (3,24)@(8,16)**. Fixes
en route: heal-dead-map strand guard (one-way pocket ⇒ suppress non-critical heal), FORWARD-FRONTIER
(past base camp ⇒ march away, never bounce — the Cerulean↔Route-5 ping-pong), pass-through candidates
from read_warps not _door_tiles (the daycare/hut confusion), REGION RE-ENTRY (`_reenter_at_column` — a
warp in a fenced region is reached by re-crossing the border at its column). KB truth: Route 5 (23,25)→
(17,0) is the DAYCARE; (31,31)→(1,30) is the UGP entrance; Route 6 = (3,24).
**NEXT BLOCKER (fresh session starts HERE): Route 6 @(17,25) hard travel wedge** — 10 consecutive
`head_to_gym -> stuck` with ZERO movement (opts collapsed to [head_to_gym] alone; ~13s/tick), until the
ring recovery warped her back to Route 5. UNDIAGNOSED — grab a frame at (3,24)@(17,25) first (arsenal #4;
suspect: a trainer/NPC gauntlet or a gate-side geometry the south-edge BFS can't reach). THEN: Vermilion
arrival + Mart/Center/gym mapping (task 6, the Cerulean recipe).
**NURSERY HONESTY: the choice framework is BUILT+unit-verified but NO judged catch has run live yet** —
runs 8-10 never offered wander_catch on Routes 5/6 because the stale Cerulean wall record's READINESS→GO
keeps PRUNING catch/grind ("the move is to travel to the gym"). Next session: let the nursery breathe
(clear the stale wall record or relax the prune when party<4 with balls), then the bar is 2-3 judged,
named, leveled catches + soul.json bonds filling. Also unpaid: quasi-dupe refinement (Pidgey-vs-Spearow).
**Canonical remains the promoted ROUTE 5 bank (clean, sanctity VALID).** Runs 4-10 were staging-only.

### ── 2026-07-06 SESSION 7 LATE (⚡ BADGE 3 CANONICAL — THE BAR IS MET) ──
**CANONICAL = BADGE 3/Vermilion (3,5)@(28,21)** (promoted badge3_bank_20260706_212559; backup
pre_badge3_backup_20260706_212610; sanctity VALID; party FULL Venusaur42/Persian32/Fearow31/
Raticate31; **flags 0x822 + 0x237 + 0x264 all verified ON-DISK in a fresh core**). Surge run 5
GOAL'd the WHOLE cascade in 211s: teach (Cut→Raticate over Focus Energy, judged forget, RAM-
verified) → **auto use_cut at the tree ("TIMBER!")** → **trash-can solver LIVE DEBUT** (bin-diving
beat → wrong-2nd reset → "the switches MOVED. we go again." → both switches, 0x264) → juniors →
**Lt. Surge DOWN — Thunder Badge**, her "badge number 3" beat in the log. First badge with a team
she built, grudge already 2-2 — THE BAR from the ship arc is MET.
**Cascade wedges killed en route (runs 1-4, committed):** (a) `_best_potion_for_sale` SOLD-HERE
fallback + shop-fail fingerprint (bcd507e); (b) HM questline steps succeed on the CAPABILITY not
the item flag (d57f850 — 0x237-satisfied ended the errand with Cut unlearned); (c) teach flow ×3:
case-cursor HOME (heap cursor REMEMBERS its row), tm_case_row = RAW pocket index (sorted model
picked TM39 'NOT ABLE!×4' where it expected HM01 at row 0), party-cursor border-RUN detector
(old x=225 probe sat inside the box) (commit after d57f850); (d) AUTO-CUT at the gym tree when
the capability is in hand — with Cut known the recognizer fell through to the WATER and armed a
Safari-Zone surf errand that poisoned her ctx; the probe now clears the tree on the spot.
**KNOWN QUIRK (characterized, worked around in the heal tool):** cut trees REGROW on savestate
reload — the badge-3 bank woke fenced inside the gym pocket; the heal cycle re-cut (19,24) first.
A general travel-layer "cuttable blocker + capability known ⇒ cut through" is OWED (today only
the gym-door probe auto-cuts). **East leg launched (`logs/longrun/east_run1.log`, GOAL_MAP=3,28
Route 10): Vermilion → Cerulean → Route 9 (Cut tree at its mouth — she has Cut) → Route 10;
judged catch en route closes dex 9→10 (Flash aide gate). DO NOT enter Rock Tunnel (Flash-dark
is its own strike).**

### ── 2026-07-06 SESSION 7 (OVERNIGHT — 🎖 HM01 CANONICAL; the ship is DONE) ──
**CANONICAL = HM01/Vermilion (3,5)@(23,32)** (promoted hm01_bank_20260706_203730; backup
pre_hm01_backup_20260706_203744; sanctity VALID; party FULL Venusaur41/Persian32/Fearow31/
Raticate31; **FLAG 0x237 verified ON-DISK in a fresh core** — recon_verify_hm01_bank.py). The ship
DEPARTED during the heal exit (cutscene, normal — gone forever). Gary grudge 2-2 (meeting #4
aboard). **Badge-3 cascade running (`logs/longrun/surge_run2.log`, goal 0x822): teach Cut→Persian
→ use_cut (19,24) → trash-can solver debut → SURGE.**
**Ship runs 14-20 + surge run 1, one wedge per cycle, all committed:** (a) re-sweep BOUNDED 2×/room
(f102342 — the Machoke cul-de-sac (1,28)); (b) `_ql_inside_target` SET on DEEPER + KEPT when a hub
exit stays interior (f102342 — the false-'I blacked out' ejections + fake note_blackout, runs 15);
(c) talk budget PER-MAP for the questline's LIFETIME (ecd801e — the galley's 6 wandering cooks;
re-transits re-burned per-entry budgets and out-ran the 14-decision stall sig, runs 16-17);
(d) heal ladder INTERIOR-FIRST rung — _exit_to_overworld before routing (7752982 — fainted-Persian
['heal'] spin in cabin (1,13), run 18); (e) `_street_gradient()` reverse-BFS over world-model warp
tables + gradient-second exit sort (3dd2a82 — the exit wandered 1F↔2F, run 19); (f)
`_enter_directional_warp` ON-TILE FALLBACK (3dd2a82 — **the captain's stair (1,6)@(30,2) is 0x6C
with a WALLED stand tile; stand ON it + press RIGHT → (1,11)**; recon_captain_door2.py proof;
captain gfx63@(5,4)); (g) `_best_potion_for_sale` SOLD-HERE fallback + shop-fail fingerprint
suppressor (surge run 1: wallet 2044 failed every 3×-comfort tier → fell back to plain Potion
which Vermilion doesn't sell → 'shop_failed'×16 STALL at the Mart door).
**SHIP MAP TRUTH (corrects session 6):** (1,10)=GALLEY (dead end, 6 cooks), (1,6)=the real 2F
corridor (9 warps incl. (30,2)→(1,11) captain), (1,7)/(1,9)=stern deck, (1,5)@(20,10)→(1,29)
untoured. **EMPLOYMENT TERMS encoded at CLAUDE.md top** (CEO 2026-07-06): credits-roll deliverable,
two-wall shift ends, bank-and-continue milestones, overnight loop, NEXT_SESSION.md at close.
**Heal-cycle tool:** recon_heal_hm01.py (bank → real heal machinery → healed bundle; MUST seed
camp.world from the BANK's world_model.json — canonical's map has never seen interiors).
**East-leg note:** dex=9 caught; the Flash aide needs **10 OWNED** — one judged catch on Route
9/10 closes the gap (KB already bills HM05/aide/Route-2-gate; Rock Tunnel needs billing).

### ── 2026-07-06 SESSION 6 (MARATHON — aboard the S.S. Anne; handoff at context budget) ──
**WHERE THE ROPE ENDS: she BOARDS, TOURS THE SHIP, and BEATS GARY ABOARD (grudge 2-2, verified in
runs 11-13 independently) — the captain not yet reached.** Ship runs 11-14, one wedge per cycle:
- **GO-DEEPER (1041a2d):** interact layer treated the exterior deck as "arrived" → now enters
  untried non-overworld warps, so she descends deck→gangway→corridors→holds. WORKS (8 ship maps).
- **PRUNER INTERLEAVE (785f860):** 'questline_resweep' returned deliberate no-move ticks; the
  roam's SILENT-NO-MOVE pruner counted 2 as a dead route and pruned head_to_gym BEFORE go-deeper
  could run — the 2F stairs sat untried. Re-sweep is now INLINE (no no-move returns).
- **UNVISITED-FIRST + STEP-OFF (9fa0b0b, riding in run 14):** deepest-first toured the same holds
  (stairs = unvisited map now sorts FIRST); spawning ON a hold's exit warp can't re-fire it → step
  off before the entry ritual. **Run 14 in flight (`ship_run14.log`, goal 0x237).**
- **GROUND TRUTH (stage state):** the 1F corridor (1,5) warps: 2F stairs **(2,18)→(1,10)** and
  **(3,20)→(1,10)**, cabins (5..23,10)→(1,12..17,29), exits (19,1)/(20,0)→(1,4). The captain's
  office is off the 2F (1,10). Gary auto-engages on the tour (LoS) — beaten every run at L41/31s.
**SINGLE-RUN LAW (2f44c5f, Jonny's housekeeping):** recon_longrun REAPS predecessors at launch —
PID-file for the run process + a PowerShell sweep killing bash 'longrun' watchers >2h old (the
9-11h nursery-watcher zombies were bash loops from a DEAD session — invisible to python checks and
holding no live handles; the ghost-notification/false-'Cut is live' class, closed at the root).
**MONITORS ARE DEMOTED: convenience only, never evidence** — run-11's monitor sat silent for 37
min on a run that had finished in 45s. Ground truth = tasklist + the raw log, always.
**RUN 14 RESULT (read before acting): STALL at (1,28)@(3,6)** — unvisited-first WORKED (a brand-new
room via (1,8)'s warp (18,2)); the stall class repeats: stuck in a room whose remaining exits won't
fire. PATTERN across runs 13-14: the terminal room's exit warp fails the entry ritual (spawn-on-tile
/ directional-entry side / entered-set) — grab a frame at the stall pos + dump that room's warps
and behaviors (the arsenal), fix the ENTRY, don't add more touring logic. The tour itself is sound:
5 hops in 30s, deterministic, Gary falls when he's on the path.
**NEXT SESSION (exact):** diagnose (1,28)@(3,6) as above, then read `ship_run14.log` END. If GOAL (0x237): the bank holds HM01 →
promote (promote_bank.py; party may need the surgical heal first — never promote hurt) → next run
teaches Cut (teach bridge fires; Persian has 4 moves now, judged forget ready; check compat —
Persian learns Cut ✓) → use_cut at the tree (19,24) → trash-can solver (env_puzzle, beat_gym
step 0) → SURGE. If STALL on the ship: read where the tour stopped (grep DEEPER + the stall pos),
fix that hop, relaunch — each cycle has been one clean wedge. ⚠ DEPARTURE: her first step onto the
exterior (1,4) after HM01 auto-fires the cutscene (forced walk → Vermilion (23,34), ship gone) —
expect a weird-looking warp in the log; it is NORMAL. ⚠ the bank's journey/strat re-record the
Gary ship win each run (2W-2L) — sanctity's monotonic check passes it; fine.
**Prunables:** ship_run1-14 logs (keep 11-14), pre_reload_*.state litter, the run-5 DO-NOT-PROMOTE
banked_STALL is superseded (its game state was promoted via the soulfix bundle).

### ── 2026-07-06 SESSION 5 (MARATHON day 1 cont. — the road to the ship; IN PROGRESS) ──
**REALITY CHECK: "Cut is live" was ahead of the disk again** — ship run 3 died with the session
shell mid-grind (no boarding ever happened). This session peeled the chain in 5 look-ahead cycles:
**CANONICAL = the L31+ SQUAD (pre_squad31_backup_20260706_180600):** Venusaur L41 / Persian L31 /
Fearow L31 / Raticate L31, all full, **the Meowth bond EVOLVED-IN-PLACE to Persian** (soulfix
bundle; sanctity VALID). Surge's Raichu is L24 — she's over-prepped.
**RUN-5 TRIAGE (5dac913):** (a) probe radius d≤3→10 (insufficient — see v3); (b) **BENCH-PIN
RE-ARM GUARD** — retire-then-re-pin was a 567-battle treadmill (Venusaur 36→41, 3 evolutions!);
a retired prep re-arms only on ROSTER CHANGE; (c) **SOUL-CONTINUITY SEED** — recon_longrun's no-op
_continuity_load skipped soul.load, so EVERY run banked soul={} unless a mid-run catch re-created
it (sanctity's 1→0 refusal was RIGHT — it saved the Meowth bond). Runs now seed from the boot
bundle; + the bond FOLLOWS evolution (pokemon_soul.note_evolve updates species in place).
**PROBE v3 (66b6546):** scan-first — the Grid is OPTIMISTIC about fences (discovered only by
bonking), so 'closest-to-door' walks drifted to the BEACH and armed SURF/Safari (runs 5-6). Now:
scan the loaded tree/boulder OBJECTS near the door, walk adjacent to the OBJECT, recognize there.
VERIFIED: Gate(hm_obstacle,'cut') arms at the tree (19,23)/(19,24). + QUESTLINE NO-PROGRESS
ABANDON (same gate+tile ×5 → drop loud) + ascii-safe probe logging (an emoji log crashed cp1252
in the except path and ATE the gate).
**DESTINATION-IN-CITY / THE PIER CLASS (d3633f2 + b693dbe):** run 7 ping-ponged 2 tiles — the cut
step says "SOUTH" but Vermilion has NO south connection; the dock is a WARP TRIO on the city map.
Standing on the anchor map with a connectionless step dir → `enter_warp(prefer=d)` (southmost
warps = the pier; ticket triggers self-clear) + `_ql_past_anchor` set on the warp (else the
re-anchor drags her back OFF the ship). Ship run 9 in flight (`ship_run9.log`, goal 0x237).
**Jonny directives encoded:** THE STAKES paragraph (CLAUDE.md vision top) + THE MANUAL-INTERRUPT
CLAUSE (persistent-world protocol) + the PRE-SHOWTIME CHECKLIST section (manual-play verify,
kill-test cadence).

### ── 2026-07-06 SESSION 4 (MARATHON day 1 — the HM pipeline; IN PROGRESS) ──
**REALITY CHECK at session start: canonical was STILL the Route 5 bank** (walk 3 ran all night
wedged at the gauntlet — the prompt's "Vermilion is canonical" was ahead of the disk). Two more
battle-wedge classes killed + verified this session:
**LAYER 8 (commit b6593e7) — the ABANDONED-BAG wedge (walk 3, caterpie 7/40 ×18 re-entries):**
`use_item_in_battle` returned 'used' with the BAG still open (its drain exited on `_white_box()`,
but the bag's USE/CANCEL box lights those pixels). On the open bag EVERY RAM byte lies (MENU_MODE
stale 2, MENU_UP stale 1, cursors frozen) → the turn loop "picked Peck" into USE/CANCEL forever.
Fix: `_bag_screen()` pale-yellow list-panel pixel detector (pocket-agnostic) + `_close_bag_screen()`
B-cascade + a bag check at the TOP of the turn loop + use_item/exit_bag never return with the bag
up. VERIFIED: wedge state → bag closed → Peck → win in 2s. Fired live in walk 4 mid-gauntlet.
**FAINTED-LEAD BATTLE-START (commit fa0d6f0) — the heal-excursion timeout class (walk 4):** a wild
encounter with a fainted lead OPENS on the forced "Choose a POKéMON" screen; flee() misread it and
`_reach_first_menu`'s A/B mash pre-selected the corpse's sub-menu → 90s timeouts ate the excursion
budget. Fix: `_reach_first_menu` stops at the party screen; flee() sends a healthy reserve FIRST;
`_force_switch` opens every attempt with B+settle (convergent retries). VERIFIED 3s on the wedge.
**SURGE REGISTERED (fa0d6f0):** GYMS row "Lt. Surge" — door (14,25), Surge (5,2)/front (5,3),
Thunder 0x822, juniors (2,11)/(8,13)/(7,8), exits (4-6,19) — so head_to_gym routes to the gym door,
whose approach is the CUT TREE → arms the HM_OBSTACLE gate → the hm01 questline → the S.S. Anne.
⚠ the gym ALSO has the trash-can electric-lock puzzle — beat_gym is NOT trustworthy for Surge until
that solver lands (recon next session).
**SHIP MECHANICS BILLED (pret):** captain gives HM01 on talk (rub-back scene), sets FLAG_GOT_HM01
**0x237** + VAR_MAP_SCENE_VERMILION_CITY=1; **the FIRST step onto SSAnne_Exterior after that
auto-fires the DEPARTURE cutscene** (forced 9-step walk, warp to Vermilion (23,34), ship gone
FOREVER). Banking discipline: bank pre-boarding + post-HM01-on-ship; ship trainers + the GARY
REMATCH (2F corridor, grudge-arc soul moment) come before the captain. Ship maps: exterior (1,4),
1F corridor (1,5), 2F (1,10), cabins (1,12..16).
**VERMILION IS CANONICAL (promote_bank.py, commit c45590e):** walk 5 GOAL'd in **48.8s/2 decisions/
0 stalls** (the full 8-layer fix stack held); party healed to FULL at the Vermilion Center through
the real heal machinery, then promoted through the sanctity gate (backup
`pre_vermilion_backup_20260706_152434`). Canonical = (3,5)@(24,0), Ivysaur L31/Spearow L18/
Rattata L17, all full. `promote_bank.py` is now the STANDING promotion tool.
**VERMILION MART CONTROL-VERIFIED (5103f1e):** ship run 1 insta-stalled on stock_up×15 (0 potions
post-gauntlet, stock unbilled). Rows billed + live-verified (2× Super Potion, bag-delta, money
3004→1604): [PokeBall, SuperPotion 700, IceHeal, Awakening, ParlyzHeal, Repel].
**HM TEACH — BUILT + RAM-VERIFIED (afc4985), the session's headline:**
`recon_teach_derive.py` derived: START-menu cursor **0x020370F4** (BAG=row 2); overworld bag pocket
byte 0x0203AD02 (shared w/ battle) but LIST cursor **0x0203AD06** (battle's AD04 doesn't track);
**ITEM_TM_CASE=364** (366 = Teachy TV — we opened it once); the case list = TMs sorted then HMs
(display row computed from the pocket array), its cursor is HEAP (no readback); the teach party
screen REMEMBERS its cursor across opens → closed-loop nav off the selected-slot ORANGE border
((255,107,34)@x225, box tops y{14,35,56,77,98}). `hm_teach.py` = a pixel-classified STATE MACHINE
(case/case_sub/bag/party/dialogue — a blind sequence once GAVE the TM as a held item!) +
free-slot-first `default_plan` + read_party_moves ground truth. **VERIFIED: TM28 Dig → Rattata
'taught'.** ⚠ RESIDUE (honest): the make-room/forget screen is A-walked and forgets ROW 0, not the
judged idx — free-slot targets (Meowth, 3 moves, Cut-compatible ✓) never hit it; a forget-screen
readback pass is OWED before any 4-move teach on a precious moveset. **TEACH BRIDGE** wired into
`_run_questline_step`: a ('cap',hm) step + item-in-case + move-unknown teaches on the spot (voiced
judgment) instead of looping to the giver NPC.
**SHIP RUN 2 (in flight, `ship_run2.log`):** the run is LIVING WELL — stock_up bought potions,
**Meowth judged-caught into the canonical lineage (party 4!)**, then a 15-min proactive bench
mega-grind fired **TWO EVOLUTIONS: VENUSAUR L33 + RATICATE L20** (+ Meowth L19), healed, arrived
Vermilion, and went BACK to grind the bench to ~L25 (narrated purpose — Fearow next). The gym-tree
→ hm01-questline → dock chain comes after the grind; the run may TIMEOUT-bank a leveled squad
first (promote it — squad-building IS forward progress).
**STAGE-3 (USE) WIRING (pre-existing, verify live):** `use_cut` oracle pick → `_route_action` →
`fm.FieldMoveActuator.clear_obstacle` (face+A → prompt → YES; recon-flagged path A) — exercised
automatically once HM01+teach land and she stands at the tree. DEPARTURE CAUTION: first exterior
step post-HM01 force-walks + warps her (cutscene) — watch the watchdogs don't fight it.
**JUDGED FORGET PAID (4eff093):** the KNOWN-MOVES screen is a first-class state — signature
(blue plate (0,108,191)@(200,4) + whitish boxes @x122) + closed-loop cursor off the selected row's
red border (tops {18,45,67,90,112}, MEASURED — a uniform-spacing guess read None while DOWN worked).
VERIFIED: forget_idx=1 swapped Tail Whip→Dig, Tackle survived. 4-move teaches are judgment-safe.
**BENCH PIN (cda8faf):** ship run 2's grind chased its own tail (participation XP levels the ACE →
live lead-8 target rose 25→26→27 every cycle, 376 battles, whole budget). The proactive-bench
target now FREEZES at arm time and retires when the floor crosses it.
**SQUAD PROMOTED (pre_squad_backup_20260706_162755):** ship run 2's TIMEOUT bank, healed via the
full adjacent-city excursion (Route 6→Vermilion Center→back, end-to-end clean), sanctity VALID:
**canonical = Route 6 (3,24)@(12,39), VENUSAUR L36 / MEOWTH L24 / RATICATE L24 / FEAROW L23, all
full HP, Meowth bonded in soul.json.** Ship run 3 in flight (`ship_run3.log`, goal 0x237 = HM01)
on the complete pipeline: pinned bench converges → gym tree → questline → dock → captain → teach
bridge (Meowth free-slot) → use_cut.
**TRASH-CAN PUZZLE BILLED (recon only, solver next session):** `SetVermilionTrashCans` randomizes
both cans into **VAR_TEMP_0/VAR_TEMP_1** (READABLE RAM — constitution: she searches honestly,
RAM is verify-only); wrong 2nd pick re-randomizes + beams back on;
FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES opens Surge.
**WATCH STATUS: canonical bank is CLEAN; she is on Route 6 south of Vermilion with a real four-mon
squad (Venusaur/Meowth/Raticate/Fearow, badges 2) about to work the S.S. Anne for HM01 Cut;
pop-in = `python pokemon_agent/play_live.py --resume --free-roam` — press GO and you'll see her
walk into Vermilion with the ocean ahead and the ship waiting at the dock.**

### ── 2026-07-06 SESSION 3 (the FINISHING strike — Vermilion push; IN PROGRESS) ──
**LAYER 7 KILLED + VERIFIED (commit e2ed4fd):** `_party_screen()` pixel detector (teal-stripe
signature, left column below the active-mon box — background at ANY party size; 4/4 on the wedge
frames, 0/4 on battle/overworld/gym/cave fixtures) + a deliberate drain handler: our-mon-down ⇒
`_force_switch()`; voluntary shift-prompt screen ⇒ ONE clean B out + a bare-B "No" to the re-shown
prompt (never A — A re-picks Yes one level up). The exact 240s wedge state now resolves in **3s**
(B out → trainer sent caterpie → Peck → win). The seven-layer gauntlet onion is CLOSED.
**WALK 1 (`logs/longrun/vermilion_walk1.log`) — the rope held to VERMILION in ONE tick** (canonical →
burgled house → Route 5 → UGP → Route 6 gauntlet → crossed into (3,5) at 242s), then exposed the next
blocker class: hurt from the gauntlet (Spearow 0 HP + a faint), heal from Route 6 found **no route to
any known Center** (world-graph only routes VISITED nodes; Vermilion unvisited) → the blind
"warp-south" fallback walked her through the SOUTHERNMOST doors in town = **THE S.S. ANNE DOCK — she
boarded the boat** and wandered its corridors the rest of the run. ⚠️ RE-IDENTIFICATION: the run-12/13
"Vermilion stair-house trap (1,5)/(1,10)" was actually the SHIP — (1,4)=exterior, (1,5)=1F corridor,
(1,10)=2F, (1,12..16)=cabins. TWO FIXES SHIPPED: (a) **Vermilion registered** — (3,5), PC door (15,6),
Mart door (29,17) (disasm-sourced, MART_STOCK left unbilled → buy loud-skips until a live visit);
(b) **adjacent-city heal for UNMAPPED maps** — the live map header knows UNVISITED neighbours; a
registered-Center neighbour one edge away gets a heal excursion before the graph/Viridian fallbacks
(Route-3→Pewter pattern generalized). Walk 2 relaunched (`vermilion_walk2.log`).
**SOUL-BUG AUDIT (AAAAAAAAAA):** action (1) from last session is ALREADY SATISFIED — the evolve
naming beat never opens the in-game keyboard (bond-layer only, campaign.py ~1639); starter + catch
flows decline via the guarded `_handle_nickname`. The burned nickname predates the guard; the fix
remains the Lavender Name Rater as an in-character beat. No code change needed.
**VERMILION PRE-BILL (disasm):** Center (15,6) **LIVE ROUND-TRIP VERIFIED** (interior (9,1), nurse
(7,2) = the shared PC layout; the generic heal healed her 79/79 through it) · Mart (29,17)
(registered; MART_STOCK unbilled → loud skip until a live visit) · **Gym (14,25), CUT-LOCKED**
(fence tree; HM01 = S.S. Anne captain — tomorrow's headline, NOT started) · Fan Club (12,17) =
Bike Voucher · dock warps (22-24,34) → SSAnne_Exterior (arrive ~(32,5); gangway (32,14)/(33,15) →
1F corridor; ticket triggers (22-23,32-33)). `recon_vermilion_map.py` runs the remaining door sweep.
**WALK 2 (`vermilion_walk2.log`) + THE CENTER-EXIT CLASS (commit b887205):** the adjacent-city heal
FIRED ("ADJACENT city Vermilion City (S)"), she healed at the Vermilion Center — then STALLED inside
the **Cable Club 2F** (map (9,2); 1F=(9,1)). Root causes, all fixed + micro-verified (2F→street in
1s): (a) **escalators are 0x6A/0x6B, board from the EAST** — step LEFT onto the tile; the south
approach is COLLISION-BLOCKED (where travel kept standing); warp fires ~60-120f AFTER standing on
it → added to `_WARP_ENTRY` + an on-tile delayed wait; (b) **60 settle frames post-warp** — the map
header reads 0s mid-fade, so back-to-back `_tile_behavior` reads lied; (c) **6-press budget** (was
3) — turn + eaten press + step-on + arrow-fire; the 0x65 exit mat fired one press past the old
budget; (d) **`_exit_to_overworld` STREET-FIRST sort** — dest group 3 beats interior hops,
actuatable behaviors beat dead arrival-mats (the Center exit row (6,8)/(7,8)/(8,8) has ONE live
0x65 arrow at (7,8)); the directional primitive now runs before blind travel. The old
nearest-first ping-ponged the floors then burned budgets on the attendant-blocked link-room warps.
Walk 3 in flight (`vermilion_walk3.log`).
**PRUNABLE:** ~10 `pre_reload_*.state` escape-hatch banks accumulating in `states/campaign/`
(disposable staging; the 2026-07-05 ones are stale).

### ── 2026-07-06 SESSION 2 (the NURSERY-BREATHES strike; commit 3142cf4) ──
**THE NURSERY RAN LIVE (runs 12-13): judged catches with voiced reasons are IN THE LOGS** — "a bug
type — I don't have ANY bug coverage, and L16 is workable. that's a real gap filled" (weedle), "my
bench is thin (3) and a L13 pidgey can pull weight" — and run 13 LANDED the first judged teammate:
**MEOWTH joined the party (4 mons, dex 5)**, with the proactive bench-target immediately planning
"level the weak ones (meowth) to ~L11 by fielding THEM." Wall #1 of the mission (Route 6 @(17,25))
was NOT spatial: BFS to the south border exists — it's the **bug-catcher gauntlet**, root-caused as
THREE battle classes, all fixed: (1) **offensive-resist matchup switch** (lead hard-resisted ≤0.25×
now swaps to a neutral/SE reserve — Ivysaur→Spearow vs the bugs); (2) **the Butterfree sleep/paralysis
wedge** — an immobilized turn (asleep/frozen/fully-paralyzed) read as "move didn't fire," benched the
moveset, aborted the unfleeable battle, and travel re-entered forever; now an immobilized turn is a
RESOLVED turn (voiced: "I'm asleep! hang in there…"); (3) **ace one-shot-during-weaken** — judged
keepers died to the L31 ace "wearing them down"; foes 10+ levels under the lead are now thrown at
full HP (early species catch fine at full HP). Plus: **WALL RETIREMENT** (readiness crossed + region
advanced ⇒ the conquered record retires — it was pruning wander_catch on every new route) + the
**nursery exemption** (thin team + balls ⇒ catching never pruned as "grind"); **full Kanto-151
SPECIES_NAME table** (she called her own Meowth "species#52" — voice lie; bonds wrote placeholder
names); **two-story-house exit rewrite** (taken-set warp-walk — enter_warp(prefer='south') warped
1F↔2F forever in the Vermilion stair house (1,5)/(1,10); this trapped runs 12-13 post-arrival).
**SHE PHYSICALLY ENTERED VERMILION (3,5) in run 12** (walked into a house before the goal check
sampled — the two-story trap ate the run).

**LATE SESSION-2 (runs 15-17 + the frame-diagnosis; commit abff453) — the gauntlet spiral peeled to
its LAST layer:** three more battle classes fixed + verified firing live: (a) the movelist open check
is now the RESPONSE PROBE alone (the MENU_MODE byte can be stale LOW post-item-use just as it was
stale HIGH — the caller's wrong-submenu B was CLOSING a genuinely-open list, A/B ×12 → stuck);
(b) **RE-ENTRY CORPSE GUARD** — a re-entered battle whose foe already reads 0 HP arms `_enemy_fainted`
at run start (a fresh agent never sees the 1→0 transition, so it move-picked into the switch prompt
forever — weedle 0/38 ×51); (c) the offensive-resist switch trigger is now MOVE-based, not type-based
(Ivysaur's poison TYPING scored 0.5 vs Weedle and hid her all-grass 0.25× moveset — the switch never
fired; now Spearow leads the gauntlet with Peck 2×, verified run 17).
**THE TERMINAL WEDGE, FRAME-DIAGNOSED (`recon_gauntlet_fight.py` + `scratchpad/fight_10.png`): the
post-faint drain oscillates A/B on the "Choose a POKéMON" PARTY SCREEN** (A selects a mon, B cancels,
A reopens — 240s verified repro). NEXT SESSION'S FIRST FIX: in the post-faint drain (battle_agent
~line 1398 block), DETECT the party screen (the `white_box=False` party-screen check `_switch_to_slot`
already uses) and handle it deliberately — our-mon-fainted ⇒ `_force_switch()`; otherwise a single
deliberate B + settle (it has a CANCEL, it's voluntary). That one handler ends the gauntlet spiral;
everything upstream of it is now proven.
**🚨 SOUL BUG (frame discovery): HER ACE'S NICKNAME IS "AAAAAAAAAA"** — a naming-keyboard A-mash from
some past prompt (suspect the evolution naming beat opening the keyboard headless). Two actions:
(1) guard every naming prompt against opening the keyboard when the oracle returns None (decline with
B — the catch flow already does this; the evolve/other flows may not); (2) the nickname itself is
burned into the save — no rename primitive exists (a Name Rater lives in Lavender — a future
in-character fix: "I'm getting this poor thing's name fixed"). Flag for Jonny: her Ivysaur has been
named AAAAAAAAAA since some earlier session; the journey narrative should not voice it until fixed.
Run artifacts: `logs/longrun/nursery_run15-17.log`; the stage state sits mid-gauntlet at (3,24)@(8,16).

**Old context below (pre-strike): rope previously ended at 🎫 S.S. TICKET OBTAINED (2026-07-05 strike,
run-8 `logs/longrun/ticket_run8.log`).**
Canonical save `pokemon_agent/states/campaign/kira_campaign.state` = inside Bill's Sea Cottage
(map (30,0)@(7,7)), party **Ivysaur L30 / Spearow L17 / Rattata L17** (ALL FULL HP), 2 badges,
**FLAG_GOT_SS_TICKET set**, Gary grudge 1W-2L, dex 4. Promoted from `banked_GOAL` (continuous real-play
lineage from the old canonical; round-trip + resume-check PASS — party/flag/coords verified on reload;
pre-promotion canonical backed up in `states/campaign/pre_ticket_backup_20260705_233354/`). The ticket
also sets FLAG_HIDE_NUGGET_BRIDGE_ROCKET + opens the Cerulean south exit → **NEXT STRETCH: south gate →
Vermilion → S.S. Anne (Cut) → Surge (badge 3)**. Old context below (pre-strike archaeology).

**VERIFIED LIVE 2026-07-05 (a fresh look-ahead from the canonical save — first run since keystone 1):**
the forward loop WORKS end-to-end up TO Gary — she shops (stock_up: buys potions+balls, 5936→336¥),
grinds Ivysaur L24→**L29**, forward-drives north through the Nugget-Bridge trainers, and REACHES Gary.
Gary's Pidgeotto **Sand-Attack** debuff (persists because Ivysaur never switches out) makes her powders +
Razor Leaf MISS (Charmander sits at 44/44). She loses most attempts but **DID beat Gary on attempt #5
(grudge 1W-4L)** — so L29 ace + sleep-lock + potions wins ~1-in-5 by attrition/variance, NOT impossible,
just **unreliable** (a fresh neutral attacker via the switch makes it consistent + watchable).

**TWO real blockers now (the full run clarified it):**
1. **Gary is unreliable** (~20% win) → fresh attacker (switch/levelled bench) for a clean, watchable kill.
2. **She does NOT progress even after WINNING Gary.** The 24-min run = **896 battles, only 9 decisions,
   ticket_flag still False, visited only (3,3)/(3,22)/(3,43)** — she's stuck in a
   lose→blackout→grind→forward-drive→re-fight LOOP at the Nugget Bridge and never completes the post-Gary
   **Route 24 (3,43) → Route 25 → Bill's house → talk Bill → S.S. Ticket** chain. She reaches Route 24 but
   loops instead of crossing. This is BOTH a progression gap (the bridge→Bill traversal/interaction isn't
   completing headless) AND a watchability disaster (a whole stream of re-fighting the same trainers).
   The switch fixes #1 and probably breaks the loop (#2) by making the crossing survivable — but verify the
   Bill completion separately (it was flagged pre-vacation as never e2e-verified).

**KEYSTONE 2 — IN-BATTLE SWITCH: ✅ SOLVED + VERIFIED (2026-07-05, committed `b6bc46c`).** The old
derivation used `PARTY_CURSOR=0x02020777`, a SHADOW byte that never moved the real selection (the live
party-menu cursor is in a HEAP-allocated `sPartyMenuInternal` struct → no fixed-address readback works;
that's why RAM-diff kept failing and why the prior "verified to slot 1" was a false verify — it checked the
byte, not the effect). **The working mechanism is BLIND DOWN, exactly like the proven `_force_switch`:**
`DOWN*(slot+1)` reaches `slot` (derived + verified on a real 3-mon fixture, `recon_switch3.py`: DOWN*2→slot1,
DOWN*3→slot2; the first DOWN doesn't leave the active mon), A=select → "Do what with X?" (defaults to SHIFT),
A=SHIFT, then advance the swap text until `gBattleMons[0]` species flips (ground truth). `_switch_to_slot`
rewritten; direct test PASS (ivysaur→spearow). Fail-safe B-out on any miss = never wedges → **armed by
default** (`POKEMON_BATTLE_SWITCH` + `POKEMON_GRIND_SWITCH` now default-on). This unblocks participation-XP
bench-leveling (weak lead → turn-1 switch to ace → weak mon banks XP) AND a fresh attacker vs Gary. NEW
fixture `states/workshop/canon_battle.state` (real 3-mon battle) for future in-battle iteration.
**Resume-safety (rule 17): NOT yet kill-tested — do that on the next banked checkpoint.**

**Everything ELSE works (verified tonight or previously):** move-list actuation keystone (a4ca84f), shop /
Cerulean Mart, heal-to-reachable-Center, forward-drive, questline (recognize→derive→execute→bend-discover→
destination-interact), Nugget-Bridge nav, in-battle SWITCH (2026-07-05). The rope is SOLID to Gary.

### ── 2026-07-05 STRIKE (the strand-fix session; supersedes wall #1 below) ──
**WALL #1 (grind-stranding) — ROOT-CAUSED + FIXED (3 real bugs, not geography). Look-ahead-verified in
stages; final proof run = `logs/longrun/grindswitch_run4.log`:**
1. **POISONED ESCAPE CHECKPOINT (fixed, verified).** The escape-hatch banked "last known-good" on GREEN
   ticks — including GREEN grind ticks standing IN the un-healable Route-4 pocket (84,15) → the true-strand
   reload went straight back into the strand, forever (run-1: infinite reload loop → STALL). FIX:
   `_center_reachable_here()` guard — only bank from a heal-safe spot (own-map Center door BFS-reachable, or
   Route 3). The permissive "can reach a border ⇒ can cross" proxy is WRONG (pocket reaches the east border
   but can't cross to Cerulean) — own-map-door BFS only. Plus: true strand falls back through the deep-wedge
   RING when recent-good is absent/declines. VERIFIED run-2: zero reload-loops.
2. **SWITCH-CONFLICT (fixed).** The matchup-switch (BATTLE_SWITCH) fired DURING the participation grind and
   pulled the just-switched-in ace back OUT, re-fielding the fragile mon → faint → strand. FIX: matchup
   switch suppressed while `PROTECT_LEAD_GRIND` is active (the ace stays in and tanks). NOTE: §0's old
   prescription (a) "route the weak-grind to Route 3" is GEOGRAPHICALLY IMPOSSIBLE (one-way ledges;
   Cerulean→Route 3 unreachable on foot — recon_route4_reach.py: 0/84 Route-4 grass tiles reach the Center)
   and UNNECESSARY once the switch conflict is fixed: with the ace reliably tanking, Route-4 grinding is
   safe (the weak mon never takes a hit). Attempted+reverted same session — do not re-attempt.
3. **THE IMMORTAL-EKANS WEDGE (fixed — a keystone-class find).** Run-3: after an in-battle ITEM use,
   `MENU_MODE` reads a STALE 2 → `_movelist_open()` short-circuited True BEFORE the FIGHT-opening A was
   pressed → move-cursor nav dead → the anti-wedge flee ALSO failed against the phantom menu → travel
   re-detected the SAME battle as a new encounter ~50× (same `ekans 27/27`, never damaged — one immortal
   battle, zero XP). Even "0-PP didn't-fire" was a misdiagnosis (a wedged menu also produces no PP-drop).
   FIX (doctrine: trust CURSOR-RESPONSE, not state bytes): `_movelist_open_verified()` — the list counts as
   open only if MOVE_CURSOR responds to a probe press; failed probe → re-home → A opens FIGHT properly.
   Plus an already-ace guard on the grind switch (skip when the active mon IS the ace).
**Weak-field NIGHTMARE-CLAUSE proof — LANDED (run-4, `logs/longrun/grindswitch_run4.log`):**
`party=[('Ivys', 27), ('Spea', 16), ('Ratt', 16)] prep=None` — Rattata **L8→16**, Spearow **L10→16**,
Ivysaur only +3 (participation share). **154 wins / 0 stuck / 0 losses**; every sampled battle led by the
weak mon (`before_sp 19` ×61); narrated purpose fired every grind tick (*"grinding rattata and spearow up
to ~L16 — fielding them, not my ace — then on to Vermilion"*). CLOSED empirically: the in-battle SHIFT
does NOT reorder gPlayerParty (run-3's `Lv25` read was a wedge artifact).
**GARY DOWN (run-5, `gary_run5.log`):** from the leveled state she beat Gary FIRST TRY (sleep-lock +
poison-chip + potions; grudge 1W-2L recorded), crossed the Nugget Bridge (15 battles), crossed into
Route 25, and walked to Bill's cape — the door (51,4) is CORRECT (warp learned). THREE last-mile bugs
then cost the ticket, all fixed same night: (a) **level-blind gain-sig** — two reverts rewound full
bench-levels because levels weren't counted as gains → level-sum added to `_gain_sig` (each level-up now
banks a ring piton); (b) **Route-25 heal unmapped** → heal→stuck ×10 → now world-graph multi-hop heal to
the nearest visited Center (+ the bank-guard mirrors it, so deep-route progress banks); (c) **door-approach
budget** — the Route-25 trainer gauntlet blew travel's 300s leg at half-distance → `enter_warp(budget_s=900)`
for questline doors. ALSO fixed en route: questline RE-ANCHOR (coarse dirs are gate-relative; off-frame
maps route back to the anchor city), hard-recovery one-way position break (heal-excursion returning to the
exact wedge tile was a designed no-op), band-aware heal-safe banking (the Route-4 "pocket" at (107,12) is
NOT a strand — the east excursion works; run-4 healed+returned twice), longrun chooser prefers travel over
empty talk. **RUNS 6-8 (the last mile, all fixed + verified):** run-6 exposed the "arrived" misfire (frontier-empty on
Route 24 ≠ arrived) → **BEND-CONTINUE** (remember maps entered via bend-exploration; hop back onto the bend
instead of declaring arrived) + the **position-loop/NPC-wait watchdog truce** (travel's deliberate wait read
as a spinner-wedge and aborted the door approach — dueling watchdogs again). Run-7 got INSIDE the cottage
and talked Bill but stalled on the scripted machine → recon + the pret disassembly (Route25_SeaCottage)
gave the exact sequence → THREE new GENERAL primitives, recon-proven then wired + verified in run-8:
**face-verified turn** (the first turn press after travel is routinely EATEN — a blind face+A interacts
with the WRONG tile; this single bug made Bill's console read "un-interactable" AND made talk_npc silently
"chat with empty air"), **wanderer-tracked talk** (re-read the NPC's live coords on arrival; honest 'talked'
only when a box actually opened), and **work-the-room** (interact BG-event machines/consoles read live from
the map header + bounded room re-sweeps as scripts change who's present). Run-8: door → talk-YES →
"KIRA initiated the TELEPORTER's Cell Separator" → human Bill → **"KIRA received an S.S. TICKET"** → GOAL.
**Constitution/CLAUDE.md updates this session:** SUPREME LAW preamble above the competency checklist;
SOUL-DEBT LEDGER section below; harness-doc pitfalls 11 (poisoned checkpoint) + 12 (dueling switches) +
13 (immortal-battle stale menu byte / cursor-response doctrine).
**GRIND_SWITCH default FLIPPED to "1"** (the chain proved end-to-end: bench-level → Gary → bridge → Bill →
TICKET). **KNOWN COSMETIC BUG (flagged, not fixed):** the in-battle grind-switch emits a FALSE
"[evolve] ivysaur evolved into spearow" voice beat (the swap misidentified as an evolution) — a soul/voice
lie a viewer would notice; fix the swap-vs-evolve discrimination before the next watch.
**WATCH STATUS: canonical bank is CLEAN; she is inside Bill's Sea Cottage, S.S. Ticket in the Key Items
pocket, party L30/17/17 at full HP; pop-in = `python pokemon_agent/play_live.py --resume --free-roam` —
press GO and you'll see her walk out of Bill's cottage and start the road south to Vermilion.**

**TWO WALLS FLAGGED TONIGHT (2026-06-28 archaeology — wall #1 SUPERSEDED by the strike above):**
1. **GRIND-STRANDING heal-wedge (blocks team-building / `GRIND_SWITCH`).** When the weak-grind fields a
   fragile mon, it can route into the **far-east Route-4 below-ledge grass pocket (≈84,15)**; a faint/loss
   there strands her — `heal_nearest` finds the local Center (12,5) unreachable AND the east edge to Cerulean
   unreachable (`_heal_excursion` can't path out of the pocket) → `heal → stuck` spin → real stall (~63s; the
   escape-hatch does NOT recover it). This is why `GRIND_SWITCH` is re-gated OFF. **ATTEMPTED + REVERTED
   (90b807e):** a blanket "grind only Center-reachable grass" filter REGRESSED the ace-grind — the look-ahead
   showed **ALL Route-4 grass is Center-UNREACHABLE** (Center (12,5) is ledge-isolated from every grass tile),
   so it blocked the grind entirely, but the TANKY ace legitimately grinds there without stranding (run_02
   leveled Ivysaur L24→L29 on it). **2nd/3rd attempt (SHIPPED but INSUFFICIENT — `GRIND_SWITCH` re-gated OFF):**
   added `grind(fragile=True)` = weak-grind paces only grass she can WALK BACK FROM to the safe start anchor
   (conditional, ace unaffected — infrastructure KEPT, it's correct + harmless). But she STILL strands at
   (84,15): the strand arrives via a DIFFERENT path than grass-pacing — a **`battle_loss` mid-travel** to a
   grass tile, or the grind **anchor itself being set inside the east area**, drops her in the pocket before
   the filter applies. **REAL FIX (the top rebuild item — bigger than a filter):** (a) route the weak-grind to
   a SAFE MAP — **Route 3** (flat, L3-6 = survivable, Center-reachable via Pewter-west) — never Route 4; AND/OR
   (b) make a true strand (`heal → stuck`, no reachable Center anywhere) FORCE an escape-hatch reload that
   actually recovers (today it just spins `heal → stuck` → stall). Both are real work; do NOT re-attempt the
   grass-filter (3 tries, tripwire hit). The verified switch mechanism is READY the moment the weak-grind can
   reach safe grass. `grind(fragile=)` + the rule-17 journey bank are shipped this session.
2. **BILL-LOOP (blocks Vermilion).** After she WINS Gary (~1-in-5) she correctly advances to Route 24 (3,43)
   and the questline bend-discovery crosses E toward Route 25/Bill — but `head_to_gym` **times out per tick**,
   crawling only a few tiles up the Nugget Bridge (y 39→31→28) while fighting; and every Gary LOSS (80%)
   triggers blackout→re-grind, eating the run budget (896 battles / 9 decisions / 24 min, ticket never
   obtained). ROOT = Gary's unreliability is the budget-sink; a reliable Gary (needs wall #1's bench-leveling)
   mostly dissolves it. Secondary: the per-tick head_to_gym travel budget is too small to complete the Route-24
   crossing in one tick — give the questline crossing a bigger/again-until-map-flips budget.

**Net: the SWITCH is the banked win tonight; both paths PAST Gary are gated on wall #1 (bench-leveling), which
is gated on the grind-stranding fix. That fix is the single highest-leverage next piton.**

**Standing harness:** `recon_longrun.py <save> <minutes>` — RUN IN BACKGROUND (foreground caps at 10min).
`POKEMON_SLEEP_LOCK=1 LONGRUN_BATTLE_LOG=1`. Redirect to a file; grep only `^\[ *[0-9.]+s\] # ` decision
lines + `RIVAL beat|END:|STALL:` — the per-decision `ctx` dumps are HUGE, never tail them raw.

**Session note:** the Gemini vision migration (`kira/senses/gemini_vision.py` + 4 files) remains UNCOMMITTED
(Jonny's core-side WIP, firewalled — not committed without his explicit smoke-test pass). CLAUDE.md rules
15 (self-help arsenal) + 16 (delegated authority) added; `POST_CREDITS_VISION.md` parking lot created.

---

## PRE-SHOWTIME CHECKLIST (verifications owed before the Kira timeline / heavy watch use)

- ☐ **MANUAL-INTERRUPT VERIFY (Jonny's clause, 2026-07-06):** Jonny plays ~2 min manually on the
  Sherpa save, walks away → bank (full sanctity bundle) → Kira resumes autonomously, clean,
  sanctity VALID, narration takes the changed world in stride ("huh, we're over here now" — no
  crash/confusion/saga break). Needs Jonny's hands for the manual part; the resume machinery is
  the standard pop-in path.
- ☐ **KILL-TEST cadence (rule 17):** ≥1 hard-kill + resume per banked checkpoint — run one on the
  next promoted bank (mechanically + narratively seamless).

## SOUL-DEBT LEDGER (CEO obligation #2 — the human halves we deferred; PAY before the Kira timeline starts)

**Definition:** a debt = a bedrock block whose MECHANICAL half is built but whose ENDEARING/human half is
deferred. Completion without these = a tech demo. The rebuild phase clears this ledger. Each entry:
`block · mechanical (have) · human debt (owe) · pay-plan`. Status: 💳 open / 🔨 partial / ✅ paid.

- 🔨 **#3 TEAM-BUILDING — the CHOICE is the soul.** LARGELY PAID 2026-07-06: `roster_judgment` (dupe/
  coverage/level/room framework) + the catch_one hook — she sizes up every wild, the oracle gets the call
  (`catch_judgment` seam), and she VOICES it both ways ("a fighting type — I don't have ANY fighting
  coverage… I want this one" / "I've already got one of those — a twin rattata doesn't make the team
  stronger… not this one"). Unit-verified 5/5; headless follows the framework's lean. RESIDUE: (a) live
  naming/choice needs oracle eyes (headless can't invent names); (b) quasi-dupe refinement (a Pidgey when
  she runs a Spearow passes the thin-bench rule — full type-overlap should read as "variety, not strength");
  (c) run-2 pre-judgment dupes (2× rattata, 2× ekans) exist only in a dead staging bank, not canonical.
- 💳 **#12 DIALOGUE — she advances text; she doesn't READ it like a first-timer.** HAVE: dialogue_drive
  (box-detect, read-along pacing), talk_npc. OWE: visible first-time REACTION to content (surprise, a
  guess about what it means, an opinion) + extracting quests/hints/directions from box CONTENT. She reads
  boxes but doesn't parse or react. PAY: dialogue content → oracle reaction seam (react + extract-intent).
- 🔨 **GRINDING-WITH-NARRATED-PURPOSE — does she say WHY, in character?** MOSTLY PAID (2026-07-05 strike,
  look-ahead-verified): every grind tick emits the in-character rationale — *"Team's under-levelled —
  grinding rattata and spearow up to ~L16 (fielding them, not my ace) so I can push through, then on to
  Vermilion City for Lt. Surge"* — and the oracle ctx carries the full first-person plan. RESIDUE: no
  occasional in-battle texture during the switch dance (e.g. "Rattata soaks up the experience, Ivysaur does
  the heavy lifting") — small, note for the next watchability pass. ALSO: the false "[evolve]" beat during
  grind-switches (see §0) is a VOICE LIE — fix before the Kira timeline.
- 💳 **ROSTER-AS-RELATIONSHIP — attachment engine, thin because the team doesn't exist yet.** HAVE:
  roster-naming-on-catch hook, soul persistence, + (2026-07-06) the BOND-HOOK FIX — a catch via ANY pick
  now fires roster_react/note_caught (a travel:-pick catch was silently skipping it; that's why soul.json
  bonds stayed empty despite a real catch). OWE: names that STICK + opinions + grief on faint + pride
  on a clutch win + a team story that accretes over the 30-40h run. PAY: unblocking now as the nursery
  fills the team; build the attachment hooks as it grows.

**Ledger law:** when a piton pays a debt, mark it ✅ here with the commit. No debt is silently cleared.

---

## APPENDIX — dated archaeology (historical; CURRENT TRUTH above supersedes)

### UPDATE 6 (2026-06-28 late, same session) — **KEYSTONE CRACKED** (committed a4ca84f): in-battle move-list cursor readback
**THE master blocker is FIXED.** Derived the RAM addresses (recon_movecursor_derive — drive route3_caught
into a wild battle via the Traveler, open the move list, RAM-diff after DOWN/RIGHT):
- **`MOVE_CURSOR = 0x02023FFC`** — single 0..3 grid index (TL0 TR1 / BL2 BR3; DOWN +2, RIGHT +1), 4 bytes
  after the action cursor 0x02023FF8. (The prior candidate 0x02024005 was WRONG.)
- **`MENU_MODE = 0x02023E82`** — action menu = 1, FIGHT move list open = 2.
- Built `_movelist_open()` (RAM `MENU_MODE==2` OR pixel — RAM survives the long core where pixel detect
  fails) + `_goto_move(idx)` (readback nav: verify each press moved `MOVE_CURSOR`, retry eaten presses —
  mirror of `_goto_bag`/`_mart_goto_row`). Wired into `_select_and_verify` + `_fire_move`.
- **VERIFIED:** recon_movecursor_verify (MENU_MODE 1↔2 correct; `_goto_move` hits every slot 0/1/2/3; a
  clean battle WINS through the new path). Look-ahead (POKEMON_SLEEP_LOCK=1, 14min): **ZERO stuck-spins**
  over a long grind of winning battles (was an infinite freeze before). **The keystone holds.**

**This UNBLOCKS the whole climb** — every gym + the E4 is multi-turn; long fights now resolve instead of
freeze-spinning. It also unblocks team-building (bench grind / catch run short fights that resolve).

**GARY's Charmander — the real counter is ACCURACY DEBUFF, not just resistance.** With the wedge gone she
reaches Charmander, but it stays 44/44 forever: Gary's **Pidgeotto Sand-Attacks Ivysaur** (and the debuff
PERSISTS across Gary's whole team since Ivysaur never switches out), so her 75%-acc powders AND her 0.25×
Razor Leaf all MISS. Added a **sleep-lock SAFETY CAP** (max 4 whiffs/foe → stop re-casting; was the 106-
stuck loop). Sleep-lock still default-OFF. **Implication:** the clean Gary kill needs a FRESH neutral
attacker (e.g. a levelled Spearow — Peck is neutral + accurate) brought in AFTER/around the debuff, i.e.
real team-building. The keystone fix makes that buildable.

**TEAM-BUILDING progress (look-ahead, solo-weak-grind ON):** 0 stuck; solo-grind fields the weak mon and
**levels it** (Rattata L8→L9 in 6 battles) — team-building mechanically WORKS now. Two remaining grind
issues being fixed: (a) **grind-stranding** — `grind()` targeted the FARTHEST grass (`gs[-1]`), drifting
her to Route-4's far-east ledge-pocket (x=107) where grass became unreachable → stall. FIXED: pace the
NEAREST reachable grass (stay local). (b) it's SLOW + targets the whole FLOOR to L29 (overkill — only need
one attacker ~L18). Solo-grind still default-OFF until the chain sustains a full bench-level + Gary win.

**PARTY-SWITCH (the 2nd keystone — WIP, gated):** applied the same readback method to the in-battle party
menu. **`PARTY_CURSOR = 0x02020777`** (gPartyMenu slot byte) derived; `_goto_party_slot(idx)` readback nav
**VERIFIED reaching slot 1**; the "Do what with X?" sub-menu (▶SHIFT/SUMMARY/CANCEL) is reached and SHIFT
triggers the swap text ("Come back X!"). **BUT the SHIFT-confirm doesn't complete reliably** — the sub-menu
REUSES PARTY_CURSOR (reads 2 after select) and the select can re-land on the active mon ("BULBASAUR is
already in battle"), so the cursor→slot mapping needs more derivation (derive the SUB-MENU cursor; detect
"swap text up" to confirm SHIFT deterministically). `_switch_to_slot` updated to use the readback nav +
pure-A confirm + a fail-safe, but it's **gated OFF** (GRIND_SWITCH/BATTLE_SWITCH default 0) so it can't
affect default play; the involuntary faint-switch (`_force_switch`) is independent + untouched. Finishing
this enables the participation-XP grind (fast bench-leveling) + a FRESH attacker vs Charmander (resets the
Sand-Attack accuracy debuff) — the clean Gary kill. recon: `recon_partycursor_derive.py`.

**NEXT:** finish the party-switch SHIFT-confirm (small, isolated — the cursor readback foundation is done),
which gives fast team-building; then beat Gary → Bill → S.S. Ticket → bank the first checkpoint → climb to
gym 3 (Vermilion). OR proceed via the (slower) solo-weak-grind. The hard part (move-list keystone) is done.

### UPDATE 5 (2026-06-28 late, same session) — LOOK-AHEAD ran the Gary stretch 4×; the KEYSTONE is in-battle move-list actuation on the long core
**What the look-ahead PROVED (4 runs from the canonical save, reading the sped-up logs):**
- **Shop-first works** (chooser reordered): she buys Super Potions + Poké Balls, then heads north.
- **She REACHES Gary** (through the Nugget Bridge gauntlet) — navigation/forward-drive is solid.
- **Gary is a MOVE-COVERAGE wall, not a level wall.** Her L24 Ivysaur out-levels Gary's ~L18-20 team, but
  her only damaging move (Razor Leaf) is resisted 0.5× by Pidgeotto / 0.25× by Charmander, and Charmander's
  Ember is 2× + burns her. With 10 *regular* Potions + poison-only she LOST the attrition (run 2, clean loss).
- **THE KEYSTONE (confirmed, the master blocker for the whole climb): in-battle MOVE-LIST actuation wedges
  on the long-running core.** With Super Potions + SLEEP-LOCK (sleep the SE hitter) the *strategy* is right —
  Sleep Powder fires, chips Pidgeotto 49→13 — but the LONGER fight then WEDGES: the move list stops
  actuating (`_select_and_verify` returns `stuck`: the FIGHT submenu won't open / isn't detected), the
  trainer battle can't be fled, and travel **infinitely re-enters → freeze-SPIN** (runs 3+4 burned the whole
  budget spinning). Run 2 (no sleep-lock, short fight) lost cleanly with NO wedge → **fight LENGTH triggers
  the wedge.** ROOT: the move-list nav (`_nav_move`/the FIGHT-open) uses BLIND taps + PIXEL detection, the
  one in-battle menu WITHOUT a cursor-readback — the bag (`BAG_CURSOR`) and action menu
  (`GBATTLE_ACTION_CURSOR`) have readbacks and work on the long core. **THE FIX (next session's keystone
  build): add a MOVE-LIST cursor RAM readback** (derive the addr by RAM-diff while moving the move cursor;
  mirror `_goto_bag`/`_mart_goto_row`) so opening FIGHT + nav + fire are RAM-verified, not pixel/blind.
  Note: the wedge only manifests after the core has run a while (STATE's "fresh core actuates 6/6, long-
  running core can't") so it's hard to iterate on in a short test — reproduce via a LONG headless fight.
- **SECONDARY BUG surfaced: infinite stuck-spin watchdog gap.** A trainer battle that returns `stuck` is
  re-entered by travel forever (no DECISION tick happens during the spin, so the decision-level stall
  detector never fires). The in-battle anti-wedge floor returns `stuck` but nothing catches the re-entry
  loop. Needs an in-battle/trainer-battle stuck circuit-breaker (surface to the watchdog layer).

**FIXES SHIPPED this session (mode-side, firewall-clean; flags noted):**
- **Sleep-lock vs SE hard-hitter** (battle_agent: sleep the foe, re-apply when it wakes, chip safely) +
  **enemy status read** (pokemon_state `status1`/`asleep` @ 0x4C). CORRECT strategy but **gated OFF**
  (`POKEMON_SLEEP_LOCK=0`) because it lengthens fights → triggers the wedge-spin; **arm it once the move-list
  readback lands.** The short one-status/foe poison chip stays on.
- **Super-Potion economy** (`_best_potion_for_sale` — buy the strongest potion the Mart sells that she can
  afford; counts all heal tiers toward stock). GENERAL, reused at every Mart. ON.
- **Heal-to-REACHABLE-center** (heal_nearest: if the local PC door is BFS-unreachable — Route 4's PC is
  across the Mt-Moon ledge split from the east grass — cross to an adjacent city with a reachable Center via
  the live map header). Fixes the run-1 hard-stall (`!! HEAL: couldn't reach the PC door` ×8 → death).
  GENERAL split-route danger fix. ON.
- **Solo weak-grind** (`POKEMON_SOLO_WEAK_GRIND=1`, default ON): field the weak member as lead and grind it
  SOLO (no in-battle participation-switch, which wedges) — viable now that Super Potions heal a weak lead
  mid-fight + the ace backstops a faint + heals are reachable. The real team-building unblock vs ace-
  overpower (which can't fix a type-resisted wall). Grind fights are SHORT → don't hit the move-list wedge.
  **Verifying now:** does it actually level Rattata/Spearow (run 5).
- **Look-ahead chooser**: shop-first → GRIND when underlevelled (don't re-charge a known-lost wall) →
  advance. (recon_longrun.py — the test harness, not engine.)

**THE STANDING TRUTH (the climb's gating dependency):** every meaningful fight up the mountain (gym leaders,
E4, any resist-wall) is multi-turn, so **the in-battle move-list actuation MUST be made robust on the long
core (cursor-readback) before the climb can proceed past hard fights.** Team-building (solo-grind) gives her
a real attacker so individual fights are SHORTER (less wedge exposure), but gym-leader fights will still be
long enough to need the readback. **Next session: build the move-list cursor readback (the keystone), then
re-run the Gary stretch with sleep-lock armed.**

### UPDATE 4 (2026-06-28 late — fresh session) — #6 CERULEAN MART **SOLVED + VERIFIED**; Mart misID corrected; Gary diagnosis sharpened
- **GROUND TRUTH re-read (recon_groundtruth.py + movedump):** live save sits INSIDE the Cerulean Pokémon
  Center (map **(7,3)** @ (7,4), the blackout-respawn spot — NOT the overworld). Party: **Ivysaur L24
  [Razor Leaf, PoisonPowder, Sleep Powder, EMPTY 4th slot]**, Rattata L8 [Tackle/Tail Whip/Quick Attack],
  Spearow L10 [Peck/Growl/Leer]. **0 balls, 0 potions, 5936¥**, 2 badges, dex 4.
- **PRIOR-SESSION MART MISIDENTIFICATION CORRECTED (rule 4 proactive):** the Mart is NOT (7,1)/door(30,11).
  That door is the **POLICEMAN-blocked robbed-house** — STATE's own §0 note records `BlockExits` parks a
  POLICEMAN at (30,12) (the sole approach to door (30,11)) until `FLAG_GOT_SS_TICKET`. Confirmed STATIONARY
  (recon_npcwatch: 40s, never moved). The prior "by elimination → (7,1)" was wrong because the flaky
  buy-test gave INCONCLUSIVE "cursor didn't respond" (not a clean reject) on a long-running core.
- **REAL Cerulean Mart = interior (7,7), door (29,28)** — identified ROBUSTLY by interior layout (clerk
  object at (2,3), the verified Viridian/Pewter signature), NOT menu actuation (recon_findmart.py). Door
  (29,28) is freely reachable.
- **BUY FLOW VERIFIED at (7,7)** in a clean run (recon_buytest): `_mart_enter_buylist()->True` (prior
  failure was the long-running-core artifact of entering 6 buildings first). **Cerulean stock rows
  (cursor-readback + bag-delta control-verified):** row0 Poké Ball(200) · row1 Super Potion(700) · row2
  Potion(300) · row3 Antidote(100) · row4 Repel(350); 5-item list.
- **GENERAL BUG FIXED — pocket-aware buy-verify (campaign.py, reusable at EVERY Mart):** `bag_count` reads
  only the Items pocket (SaveBlock1+0x310); **Poké Balls live in a separate balls pocket (+0x430)**, so a
  ball purchase was invisible → `buy_at_mart`'s `bag_count != before+1` verify ALWAYS failed → **she
  literally couldn't buy balls**. Added `_balls_pocket_count(item_id)` + `_item_count(item_id)` (dispatch by
  ball-id range 1-12) and wired `buy_at_mart` to use it. Registered `CERULEAN_MART_DOOR=(29,28)` in
  `CITY_MART_DOORS` + `MART_STOCK[CERULEAN]=[4,22,13,14]`.
- **E2E AUTONOMOUS SHOP — PASS (recon_shop_e2e.py):** from the live save, exit Center → walk to Mart →
  `buy_at_mart([Potion×6, PokéBall×6])` → potions 0→6, balls 0→6, money 5936→2936, exit to overworld.
  **#6 = COMPILES+WIRED (real stock_up path)+VERIFIED.** NOT committed yet (firewall: commit only specific
  pokemon_agent files + campaign.py; Jonny's uncommitted vision-swap WIP in kira/ stays untouched).
- **GARY DIAGNOSIS SHARPENED:** Ivysaur is *over*-levelled vs Gary's ~L18-20 team — the wall is **move
  coverage**, not level. Her only damaging move (Razor Leaf) is resisted 0.25× by Charmander, BUT she has
  **Sleep Powder + PoisonPowder** (type-independent). Clean kill = **Sleep Powder → PoisonPowder → stall +
  heal-through with Potions** (poison ticks Charmander dead while sleep neutralizes Ember). The ONLY missing
  ingredient was survival margin = Potions, now buyable. So team-building (#3) is the broader-climb need, but
  **Gary may fall to existing-team + potions + stall** — being verified by the look-ahead now.
- NEXT: look-ahead from canonical save (she can now shop) → read where it actually stalls (Gary win? Bridge?
  Bill interaction?) → fix that blocker → climb. Harness chooser reordered to shop-before-charging-a-wall.



**Rope laid solidly to:** post-Misty Cerulean (her real `states/campaign/kira_campaign.state`). The next
pitch (Nugget Bridge → Bill → S.S. Ticket) is **NOT yet cleared** — the autonomous look-ahead found a real
wall there, partially fixed (details below).

### Standing infrastructure BUILT this session (the big deliverable) — COMPILES+WIRED+VERIFIED
- **`pokemon_agent/recon_longrun.py` — the LOOK-AHEAD ORACLE + RESUMABLE CHECKPOINT.** Runs her REAL
  `free_roam` loop headless at max emulator speed (**measured ceiling 14.3× real-time** = 856 fps; AudioPump
  is the only throttle and it's off; video can't be disabled — pixel detection needs the framebuffer), for a
  LONG stretch until the GOAL (S.S.-Ticket flag 0x234) or a genuine STALL. Rich per-decision + per-battle
  logging. PIECE 2: canonical save is PROTECTED (all in-run persistence redirected to a staging dir), and the
  staged savestate+sidecars are banked + **round-trip verified (save→load→party/badges/flag identical, PASS)**.
  Env knobs: `LONGRUN_BARGE=1` (inject N potions + clear stale wall + forward-push chooser, for premise tests),
  `LONGRUN_POTIONS=N`, `LONGRUN_BATTLE_LOG=1`. **This is the standing verification tool — use it, not
  micro-tests (now CLAUDE.md rule 8).**
- The faithful chooser handles `kind="battle_item"` (uses heals) + `kind="action"` (follows the machinery's
  steering). It stands in for the LLM oracle (which only fires per-tick, wireable later via the HTTP endpoint).

### UPDATE (later same night) — battle fixes landed; Gary is a TEAM-STRENGTH wall
Committed `97ca143` (+ `a725bb6`): **in-battle item-use WORKS** (the earlier "broken" was a LOGGING
artifact — the success path `emit()`s, which the harness silences; potions ARE consumed, confirmed
`count 40→39→…`). **Matchup-aware heal** (heal at 50% vs a super-effective hitter, not the 30% floor) +
the **status strategy** (poison once/foe) now get her **all the way to Gary's LAST mon** (Pidgeotto dead →
Charmander chipped to 9 → fainted → Abra → Rattata 16/34) — but she **loses the attrition war at the end**:
Ivysaur faints to Charmander's 2× Embers (crit variance), and the **L8/L10 bench is too weak to finish**.
**KEY FINDING: this is a TEAM-STRENGTH wall, not a battle-AI wall.** 40 injected potions don't fix it (you
can't Potion a fainted mon). Hard-won nuances for the next session (the Mt-Moon-lesson kind):
- **Don't over-stack status moves.** A 2nd status play/foe (sleep+poison) made the LONG healing fight
  WEDGE/time-out ("stuck" ×10) — likely the move-list nav to non-adjacent slots (PoisonPowder s1 → Sleep
  Powder s2 → Razor Leaf s0) stalls on the long core. Reverted to ONE status/foe (poison), which performed
  best. Keep battle fights SHORT.
- **Production battle cap = 180s** (`play_live.py:329`); the harness now matches it (was 40s — a 40s cap
  made multi-mon healing fights falsely "stuck"/reset Gary to full). If you see "stuck" on a long fight,
  suspect the cap, not the AI.
- **THE REAL FIX = level the bench.** Two routes: (a) the in-battle participation-XP **switch** — the
  settle-fix landed (party screen now OPENS via `_settle_action_menu()` before `_goto_pokemon`), but the
  slot-SELECT still fails (species stays the weak lead after DOWN×slot+A+A — the FRLG party-screen nav /
  a party-cursor readback is the missing piece; `_force_switch`/faint uses blind DOWN and works, so the
  voluntary path differs — recon the party-screen layout + find the party-menu cursor RAM addr, mirror the
  `_goto_bag` readback). `POKEMON_GRIND_SWITCH=0` until then. (b) **low-level-grass grind** (route the weak
  mon to Route-3-class L3-6 grass where it survives + wins solo — reliable, no actuation, but slow). Route
  (a) is higher-leverage (also E4 switching).
- **Cerulean Mart is UNMAPPED** (`CITY_MART_DOORS` = Pewter/Viridian only) → she can't buy potions/balls
  (has 5936 money, 0 potions, 0 balls). Cerulean overworld (3,3) building doors (live-read): Center=(22,19)→
  (7,3) [blackout respawn]; other building warps → (7,0)/(7,1)/(7,2)/(7,5)/(7,6)/(7,7)/(7,9) at doors
  (10,11)/(30,11)/(15,17)/(31,21)/(13,28)/(29,28)/(17,11) — the MART is one of these (NOT yet identified;
  enter each + detect the buy clerk). Mapping it enables autonomous stock-up, but **won't alone beat Gary**
  (team-strength-bound).

### UPDATE 3 (late night) — ROOT-CAUSE PIVOT: she's a blank slate; building the BEDROCK competency map
The real root (Jonny): she has **no team-building instinct** + **no model of what the game is FOR**, so her
bench is random and she walls at every gym. Committed this session:
- **f24d59d — FOUNDATIONAL GAME-MODEL** wired into her decision/voice ctx (`_spine_and_history`): win-cond
  (8 badges→E4→credits), what a TEAM is for (6, balanced, solo+dead-bench loses), catching/Pokédex central,
  roster-selection judgment, the full arc + a party-aware "your team is thin/lopsided" nudge. **VERIFIED
  wired** (all elements present in the ctx; live behaviour-shaping needs a full look-ahead). + **ACE-OVERPOWER**
  grind fallback (switch gated → level the ACE to overpower) — works mechanically but **SLOW for an
  over-levelled ace** (low wild XP); this *confirms* team-building is the efficient/real fix (a fresh L10
  catch levels fast on the same wilds).
- **e2e772d — BEDROCK MAP** in CLAUDE.md ("Kira's player-competency checklist", 15 blocks status'd, Tier 1/2/3).
  THE FRAME for all remaining work: build proactively just ahead of her feet. Tier-1 build order: #6 Mart/
  economy → #3 team-building → #5 in-battle switch → #12 dialogue extraction. (#1 game-model + #7 healing done.)
- **CERULEAN MART located = interior (7,1), door (30,11)** (by elimination — the buy-clerk test rejected all
  6 other reachable Cerulean buildings). **BUT the door approach is blocked by an NPC** ("won't budge") + the
  travel/enter_warp BFS won't path onto the warp tile (my plain BFS reaches it, so it's a travel/NPC-routing
  bug, not geometry). So the buy-test never confirmed + it's not yet auto-mappable. NEXT: resolve the
  NPC-block/door-approach (Layer-A route-around or talk-the-NPC) → confirm buy-clerk → add `CERULEAN_MART_DOOR`
  to `CITY_MART_DOORS` → balls+potions unblock catching (#3) AND the barge (#6).
- **GARY remains the immediate pitch**, now understood as a TEAM-STRENGTH wall whose CLEAN fix is team-building
  (catch a good mon near Cerulean/Route24-25 → level it fast → real squad → beat Gary properly), with
  ace-overpower as the slow fallback. Sequencing: Mart reach → catch/build → beat Gary → Bill → **bank the
  first checkpoint** → gym-3 push.

### THE NUGGET-BRIDGE / BILL BLOCKER — fully diagnosed via the look-ahead
The wall is **GARY (the rival) in Cerulean** — `trainer:Cerulean City:` lead `charmander` (Fire), recorded
3× loss. Root-cause chain (each found by reading the sped-up playthrough log):
1. **Ivysaur's only damaging move is Grass (Razor Leaf), which Charmander RESISTS 0.25×** — she can't
   out-damage it; her backups are Rattata L8 / Spearow L10 (Spearow's neutral Peck works but it's too weak).
2. **In-battle BAG wouldn't open** (`_open_bag` did a blind `_tap('RIGHT')` that gets eaten on the long core →
   "eaten RIGHT") → she couldn't use Potions. **FIXED:** new `_goto_bag` navigates by `GBATTLE_ACTION_CURSOR`
   readback (mirror of the proven `_goto_pokemon`). General fix; no flag.
3. **Move policy ignored STATUS moves** (power 0) → spammed resisted Razor Leaf. **FIXED:** `_select_and_verify`
   now, when every damaging move is resisted (best eff ≤0.5) and the foe is fresh, fires a status move —
   PoisonPowder/Leech/Toxic chip TYPE-INDEPENDENTLY (bypass the resistance), sleep neutralizes. General,
   E4-relevant. Reset **per-foe** (so Gary's Charmander gets poisoned, not just his lead).
4. **In-battle item use still SILENTLY NO-OPS** (the precise remaining blocker): the chooser correctly picks
   `use_potion` (item 13) 11× per fight, `_goto_bag` is fixed, but `use_item_in_battle` returns on the
   **unlogged `return "no_item"`** path — the in-battle items-pocket read (`_items_pocket`/`_items_count` /
   the `_HEAL_ITEMS_PREF` heal-item mapping at battle_agent ~line 692) isn't seeing the potions. **NEXT
   ACTION: fix the in-battle items-pocket read so a held Potion is found + consumed.** Once that lands, the
   Potion-barge (poison-chip + heal-through) should beat Charmander headless.
5. **In-battle SWITCH actuation wedges the battle on the long core** (`outcome=stuck`) — same menu-on-long-core
   class as the bag bug, but the PARTY-menu nav (post-`_goto_pokemon` list nav) still needs the readback
   treatment. So the **participation-XP grind-switch is gated OFF** (`POKEMON_GRIND_SWITCH=0`) — the live
   look-ahead proved it one-shots/wedges. The weak-mon grind therefore can't level the floor yet (it needs
   either the switch fixed OR low-level-grass routing).

### Capability fixes shipped this session (additive, mode/engine-side; **battle-regression NOT yet re-run**)
- `_goto_bag` readback (item-use bag open) — battle_agent. **General.**
- Status-move strategy when resisted (poison/sleep/leech, per-foe) — battle_agent. **General, E4-relevant.**
- Strategic underlevel-grind (recognise underlevel → field WEAK members via save-safe party-reorder → exit on
  team FLOOR) — campaign + pokemon_strategy, `POKEMON_STRATEGIC_GRIND=1`. Mechanics VERIFIED 5/5
  (`recon_strategic_grind.py`); but its participation-XP switch is gated off (see #5), so it can't yet level
  the weak team in real battles.
- Grind-switch (lead weak → turn-1 switch to ace for participation XP) — battle_agent, `POKEMON_GRIND_SWITCH=0`
  (gated off: wedges on long core until the party-menu nav gets the readback fix).

### FORWARD ROUTE MAP (the survey ahead)
- **Gym-3 approach (Nugget Bridge→Bill, NOW):** blocked at Gary/Charmander. Closest to clear — needs the
  in-battle item-use read fix (#4). Then poison-chip + Potions should win. Then Mart-buy autonomy: **Cerulean
  Mart is UNMAPPED** in `CITY_MART_DOORS` (only Pewter/Viridian) → `stock_up` never offered at Cerulean → she
  can't buy Potions herself (she has 5936 money, 0 potions). **Map the Cerulean Mart door** so the barge is
  fully autonomous (no injected potions).
- **Gym 3 (Surge/Vermilion):** needs S.S. Anne → **HM01 Cut** (Cut actuation gated `POKEMON_FIELD_MOVES=0`,
  unverified on long core). Destination-interaction layer handles the S.S. Anne handoff.
- **Gym 4+ / Rock Tunnel:** needs **Flash**. Later gates staged in `FORWARD_CLIMB_STAGING.md`.
- **CROSS-CUTTING keystone (blocks the whole game):** **in-battle MENU ACTUATION on the long-running core**
  (bag use, party switch). The `_goto_bag` readback fix is the template; the PARTY menu + the items-pocket
  read need the same. This recurs at every gym/E4 (item use + switching are E4-critical). Solve it generally.

---

## 1. HOW IT FLOWS TOGETHER (architecture at decision time)

The Pokémon harness is a **separate subprocess** (`pokemon_agent/play_live.py`) that drives the emulator.
It talks to core Kira over HTTP (`KiraVoice` → control_server). Core Kira owns ALL personality/voice; the
harness owns game mechanics. Four channels:

- **DECISIONS:** `campaign._soul_choose` → `voice.choose` (HTTP) → `/cmd/pokemon_choose` → `bot._pokemon_choose`
  → LLM. The LLM prompt = `_POKEMON_CHARACTER_RULES` + `_POKEMON_DECIDE_FRAMING` + **live run-state block
  (FIX 2)** + the ctx (`place` seam carrying goal-layers/recalibration/wall-awareness from campaign) +
  `_build_self_block` (her mood/want/bond). Returns her pick. → **run-state + goals REACH the brain here.**
- **REACTIONS:** `campaign.on_event` → `voice.emit` (HTTP, **deduped** — FIX 1) → `/cmd/pokemon_event` →
  `bot._pokemon_react` → `_execute_interjection` → LLM. Prompt = `_POKEMON_CHARACTER_RULES` + **run-state
  (FIX 2)** + **saga on tier≥2 (B-4)** + `_build_self_block`.
- **STATE/DISPLAY:** `campaign._publish_health` → `health.json` → `/cmd/pokemon_health` → operator dashboard
  + `/pokemon_hud.json` → stream HUD. The bot ALSO reads `health.json` for the brain (FIX 2) — **one source
  of truth shared by display AND decision.**
- **IDENTITY:** `bot.pokemon_mode` (auto-set True on launch) flips the `_build_self_block` header to
  player-mode; off = cohost (byte-identical). **CONTINUITY:** `voice.journey` → `journey_core.json` →
  `_pokemon_journey_block` (idle chat + now react tier≥2).

The battle ENGINE (`battle_agent`) is deterministic policy (type-chart), NOT the LLM — the oracle is only
consulted for items + (gated) switching. Move selection is the hands; her voice reacts.

---

## 2. POKÉMON HARNESS — feature reality

| Feature | COMPILES | WIRED (where) | VERIFIED | REACHES |
|---|---|---|---|---|
| Battle flee floor (anti-wedge) | ✓ | run loop `_unresolved_turns` | ✓ LIVE (08:20 watch) | BRAIN |
| Repetition floor (FIX 1: 0-PP/dialogue/overworld) | ✓ | move pick + emit dedup + dialogue cycle + roam nudge | ✓ regression 3/3 + unit | BRAIN |
| Ineffective-move aversion (B-1) | ✓ | `_select_and_verify` pick | ✓ offline (Normal→Ghost=0) | BRAIN |
| In-battle party switch (B-1) | ✓ | run loop, gated `POKEMON_BATTLE_SWITCH=0` | matchup math ✓ offline; **actuation needs live eyes** | BRAIN (gated) |
| Run-state → voice/decision (FIX 2) | ✓ | `_pokemon_react`/`process_and_respond`/`_pokemon_choose` | content ✓ vs health.json; live pending | BRAIN+DISPLAY |
| 3-tier goal-layers | ✓ | decision place-seam + voice (FIX 2) + dashboard | content ✓; live pending | BRAIN+DISPLAY |
| Recalibration (`_active_objective`) | ✓ | roam ctx + health + dashboard | pending live (detour→resume) | BRAIN+DISPLAY |
| Strategic-stuck floor + readiness→GO | ✓ | `_available_actions` prune + ctx | two-tier unit ✓; live pending | BRAIN |
| Strategic underlevel-grind (field WEAK members) | ✓ | `_prep_team_target`→`grind_weak_members` (exec) + `_available_actions` reframe + ctx fold + dashboard rationale | mechanics headless ✓ 5/5 (`recon_strategic_grind.py`); **real-battle leveling + weak-lead survival need live eyes** | BRAIN |
| World-model (`pokemon_world`) | ✓ | spatial brief + travel targets → oracle | persists-resume (claimed) | BRAIN |
| Catch procedure (weaken+PP) | ✓ | `catch_pokemon` | **pending live** (no catch in watch) | BRAIN |
| Resolved/looping-NPC guard (B-2) | ✓ | `_drain_overworld`→`_looped_spots`→talk gates | regression ✓; live trigger pending | BRAIN |
| Travel routes around plain blocking NPCs (LAYER A) | ✓ | travel gauntlet→unified `_blocked_npcs`→plan/talk both read it; `no_route_npc_blocked`→oracle | wiring ✓ (shared-by-ref); **live Slowbro state pending** | BRAIN |
| Universal wall-clock watchdog (LAYER B) | ✓ | `wf.StuckWatch`←play_live render feed→`_stuck_request`→roam disengage + travel cancel | unit ✓ 8/8 (frozen-box/Slowbro toggle/legit-read); **live timing pending** | BRAIN |
| Warp/spinner position-loop escape (B-3) | ✓ | `travel` sliding-window → `stuck` | bounded logic ✓; live trigger pending | BRAIN |
| Gary arc at ALL encounters (B-4) | ✓ | `_observed_battle_runner` → `note_rival_encounter` | regression no-false-fire ✓; live rival pending | BRAIN |
| Saga → in-game reactions (B-4) | ✓ | `_pokemon_react` tier≥2 | code path ✓; live pending | BRAIN |
| Identity flip (play-mode) | ✓ | `_build_self_block` header | ✓ LIVE (first-person watch) | BRAIN |

**Gated-OFF (with reason):** `POKEMON_BATTLE_SWITCH=0` (actuation unverified), `POKEMON_FIELD_MOVES=0`
(Cut/Surf/Strength actuation unverified — gym 7/8 gatekeepers), `POKEMON_ITEM_PICKUP=0` (unverified),
`POKEMON_GUIDE_SEARCH=0` (Google Custom Search API disabled), `CATCH_SUBCORE=0` (legacy jump-cut path).

**Pokémon GHOSTS / half-wired:** *(the two big ones are now FIXED today)* — Gary arc (was opening-only →
**FIXED B-4**), saga-in-reactions (was chat-only → **FIXED B-4**). Remaining: HUD goal refresh is
display-only (HUD being redone — low priority).

---

## 3. CORE KIRA — feature reality (from the core audit)

**Good news: the major core features REACH THE BRAIN** (verified by the audit tracing prompt injection):
repetition-awareness (`avoidance_block`), emotional state, current-want, Jonny-bond, sentiment/memory
ledger, entity theories + called-shots, chat director, salience gating, visual perception + staleness,
ambient audio + dialogue summary, running bits, voice guardrails. None of these are ghosts.

**GHOSTS / unwired / aspirational (core):**
- **Dread→struggle→catharsis arcs, vendetta, naivety:** the audit found them only as comments in
  `streamer_overlay.py`, NOT tracked/injected. **CAVEAT:** memory says batch-7 shipped these via
  `persona/private/personality.txt`, which is **GITIGNORED** — so they're live-local in Jonny's persona
  file (reaching the prompt as persona text) but invisible to a code audit and uncommitted. **Status:
  WIRED-via-persona (live-local), NOT in code.** Decision needed: promote to tracked state, or accept as
  persona-only.
- **Activity-Director taxonomy** (`DIRECTOR_TAXONOMY_ENABLED`): shapes the REPLY path only; proactive
  interjections bypass it (always base shape). Partial — reply-only, not a full ghost.
- **`web_search` import** (bot.py ~38): imported per a TODO, never wired to a command. DEAD import — prune.
- **`LOOPBACK_POST_TTS_COOLDOWN_S` / `LOOPBACK_SUMMARY_AGEOUT_S`:** declared in config, grep finds no
  consumption — likely orphaned env vars. Verify/prune.
- **VRAM telemetry:** diagnostic logging only (intentional, not a ghost).

**DUPLICATES / parallel (core):**
- **Self-block split:** interjections use `_build_self_block` (compact); replies assemble self piecemeal
  across `dynamic_context` (~50 lines). BOTH reach the LLM but via different scaffolding — a mood tweak can
  affect replies vs drives differently. Not broken; fragile. **Canonical to keep:** `_build_self_block`;
  recommend replies call the same factory. (Not done — flagged, no behavior bug today.)
- Jonny-bond renders twice in replies (in `get_state_block` ctx + `_build_self_block`) — minor redundancy.

**DEAD/DEPRECATED (confirmed):** repo-root `control_server.py` — the audit confirms it does NOT exist
(good; only `kira/dashboard/control_server.py`). `play_live --cable` arg deprecated/unused.

---

## 4. WIP / deferred (with reason)

- **In-battle switch actuation** — built+wired+gated; needs a live control (savestate + Jonny) before
  arming `POKEMON_BATTLE_SWITCH=1`. Deferred-armed because unverified menu-nav could wedge a battle.
- **Surf/Strength HM actuation** — gym 7/8 gatekeepers; dedicated live-verify session pending.
- **Forward gyms 3–8** — data-bill staged; coords need recon (`FORWARD_CLIMB_STAGING.md`).
- **Full warp/spinner puzzle-solver** — only the loop-ESCAPE is built (can't-get-stuck); the solver
  (route a spinner/warp deliberately) is scoped-next, not needed before gym 3.
- **Catch procedure live-verify** — no catch happened in the last watch; `_can_weaken`/`need_pp` unproven live.
- **Self-block unification** (core) — flagged; no behavior bug, low priority.
- **Type-immune defensive matchup** returns 1.0 not 0 in `_matchup_def` (cosmetic; logic driven by offense).

## 5. PRUNABLE (cleanup candidates, non-blocking)
- `web_search` dead import (bot.py). Orphaned loopback env vars. `--cable` arg. ~80 `recon_*.py` archive
  scripts (not dead — methodology; clutter). Stale "CANDIDATE/UNVERIFIED" comments on de-facto-verified
  `pokemon_state.py` offsets.

---

## 7. QUEUED — post-watch (do not lose)

1. **In-battle switch — dedicated actuation verify → then arm.** The verb is built/wired/gated
   (`POKEMON_BATTLE_SWITCH=0`). Run a controlled live check (savestate: active mon out-typed + a stronger
   reserve) with Jonny watching that the party-menu nav lands the switch and the battle continues. Only
   then set `POKEMON_BATTLE_SWITCH=1`. (E4-blocking until done.)
2. **Gym-3 GymSpec build (Lt. Surge / Vermilion).** Needs live coord-recon (gym door tile + Surge front
   tile + junior count, same method as Brock/Misty) AND Jonny's decision on the trash-can switch puzzle
   (he vetoed hardcoded presses → capability-not-script preferred). Until built, `head_to_gym` grace-wanders
   at Vermilion (no freeze). See `pokemon_agent/FORWARD_CLIMB_STAGING.md`.
3. **Persona-only emotional arcs (dread→struggle→catharsis, vendetta, naivety).** Currently live-local in
   the GITIGNORED `persona/private/personality.txt` (reach the prompt as persona text, but not tracked in
   code or committed). Jonny to decide: promote to tracked `kira_state` arc-tracking (like called-shots),
   or accept as persona-deep. Either way, record the decision here.
4. **Off-thread decision/event HTTP (the DEEPER lag fix).** `_soul_choose`→`voice.choose` and
   `voice.emit`/`on_dialogue` are SYNCHRONOUS blocking `urllib` calls on the MAIN render thread
   (`pokemon_voice.py:271-287`) — every LLM decision freezes game render + music for its duration. The
   post-watch throttle (silent-no-move guard) stops the *rapid* stutter by ending the stuck re-pick loop,
   but a single blocking decision still micro-stutters even during normal play. DEEPER FIX (queued, NOT
   mid-firefight): run these HTTP calls off the render thread (worker thread / async) so LLM latency never
   touches the frame loop. Risky surgery on the live path — schedule a dedicated pass with Jonny.
5. **Lapras/foreknowledge confabulation (HELD for Jonny).** She invents game knowledge she hasn't seen
   this run ("get Lapras"). Source = the play-mode oracle prompt `_POKEMON_DECIDE_FRAMING` (`bot.py:3233`)
   has no "only reference what you've actually encountered this run" grounding. Fix = one line there, but
   it's CORE-KIRA voice + overlaps the gitignored naivety arc → needs Jonny's sign-off before touching.
7. **Warp-routing: Cerulean→Vermilion forward chain — IN PROGRESS, NOT yet traversing (HANDOFF DETAIL).**
   STATUS by part (three-state):
   - **Warp ENGINE — DONE + VERIFIED (offline).** `travel.read_warps(b)` reads the live map-header warp
     table (verified vs disasm: Route 4 = (19,5)→MtMoon/(12,5)→PC/(32,5)→MtMoon, save-coords, null=0).
     World-model has warp edges; `route()`/`next_step()` traverse EDGES∪WARPS; `head_to_gym` executes an
     edge hop OR a warp hop (travel-to-tile + `enter_warp(pick=tile)`); warps learned live + persisted.
     Verified offline: route(Route4→MtMoon(1,1)), next_step→('warp',(19,5)) vs ('edge','north'), save/load.
   - **Live geography cross-checked (DONE):** Cerulean (3,3) connections live = N→Route24 (3,43, Nugget
     Bridge), **S→Route5 (3,23)**, W→Route4 (3,22), E→Route9 (3,27). So **Cerulean→Route5 is a plain south
     EDGE** (head_to_gym already walks it). The Underground Path warp is ON Route 5 (past hop 1). NOTE: the
     disasm route-number export is unreliable (Route24 is (3,43), not the contiguous pattern) → ALWAYS
     cross-check route IDs live. City block (0-10) is reliable (Vermilion=(3,5) etc.).
   - **HARNESS — BUILT:** `pokemon_agent/recon_warptrace.py` — stub oracle picks head_to_gym each tick,
     runs the REAL recovery machinery (Layer-A route-around + watchdog + no-move guard + off-spine), reads
     each map's warps LIVE, no-ops canonical saves, heal-patches HP each tick. `--fight` forces real
     battles; default flees wilds for speed. Confirmed it learns warps live (read Cerulean's 14 warps).
   - **BUILT + VERIFIED (offline) 2026-06-28 — GENERAL gate-unlock questline capability (Phases 1-4,
     commits 5b1100c→ac7f2e0).** New `pokemon_agent/questline.py` + `gamedata/frlg_gates.json` (curated
     disasm KB) + `campaign` wiring: **recognise** a typed Gate (HM_OBSTACLE / STORY_NPC / ITEM_GATE /
     BADGE_GATE) → **derive** an ordered questline from the KB capability chain (live-cross-checked, prereqs
     first) → **execute** it via `head_to_gym` (routes the unlock ERRAND instead of the gated wall, reusing
     travel). VERIFIED headless on the live Cerulean save: she recognises the Slowbro story-gate, OPENS the
     S.S.-Ticket questline, narrates it in character ("I need the S.S. Ticket — a guy named Bill, north…"),
     and drives NORTH to Cerulean's Nugget-Bridge edge (reverses off the south wall), persisting across
     ticks + a blackout, reaching her DECISION ctx via the place seam + health.json (dashboard). Self-clears
     when `FLAG_GOT_SS_TICKET` reads set. Generalises to Surf/Strength/Fly/Flash/item-gates by the SAME
     pipeline (proven via synthetic-KB test). GuideSearch is wired as the secondary deriver fallback
     (no-op until the Custom Search 403 clears). `POKEMON_QUESTLINE=0` disables. **NOT yet live-verified:**
     the Nugget-Bridge gauntlet → Bill's cottage → ticket COMPLETION needs a healthy live run (heal between
     the un-fleeable trainers); the shipped KB is Cerulean/Bill/Cut only (other gates added disasm-checked
     as she nears them).
   - **FIXED 2026-06-28 — PROACTIVE FORWARD DRIVE (the backward-grind root fix). REACHES: BRAIN.** The live
     bug: post-Misty she WANTED 'grind on the way' → chose `travel:3,22` (Route 4, a cleared dead-end BEHIND
     her); she walked backward to grind, never bonked the Slowbro south gate, so the questline never opened.
     ROOT CAUSE (recon, not symptom): the gate/questline was recognised **only REACTIVELY** inside the
     `head_to_gym` execution branch, so at DECISION time `_available_actions` offered the backward grind
     (`travel:*`/`battle`/`wander_catch`) on EQUAL footing with `head_to_gym`, and a grind-want picked
     backward. Worse, the canonical save sits ON Route 4 (a side-branch WEST of base camp) and `head_to_gym`'s
     own routing would walk the local 'south' edge to Route 3 — **further backward**. THREE-PART FIX (all
     mode-side `campaign.py`, firewall intact, flag `POKEMON_FORWARD_DRIVE=1`):
       (1) `_ensure_forward_questline(state)` — recognises the forward (south) gate and OPENS the questline
           PROACTIVELY each tick BEFORE the action set is built (no longer waits for a wall-bonk).
       (2) `_available_actions` forward-drive — when a forward-unlock questline is open OR she's drifted
           off-branch (graph can't route to the gym yet AND she's off the base-camp city), `head_to_gym` is
           reframed as the DOMINANT forward pull and the backward-grind options are PRUNED (travel targets
           no closer to base camp + standalone grind; grind now happens ON THE WAY via the forward march).
           Stands down for survival (critical-heal) + the strategic-stuck floor (which owns the lost-
           repeatedly case). Strictly conditional/reversible (feature OFF restores the full set).
       (3) `head_to_gym` FORWARD-SPINE recovery + `_base_camp(state)` — when the graph can't route to the gym
           city yet, route toward the base-camp city (GYM_SPINE predecessor, e.g. Cerulean for Vermilion)
           instead of blindly walking 'south' into a further-backward branch; the proactive questline takes
           over once she's there. VERIFIED headless from the ACTUAL live Route-4 save (`recon_forward_drive.py`
           end-to-end: Route 4 → EAST to Cerulean → questline OPENS → heads NORTH toward Bill, never backward
           to Route 3; `recon_forward_drive2.py` action-set: backward pruned, forward kept, reframed,
           reversible). Reaches her DECISION ctx (the reframed `head_to_gym` description + questline narration
           via the place seam) AND the dashboard (health.json `questline`/`rationale`). Fixed a latent `→`
           UnicodeEncodeError in the new log line (now ASCII `->`). NOTE: she ends short of Bill in 8 ticks
           because her L8/L10 teammates lose the un-fleeable Nugget-Bridge gauntlet (real game difficulty —
           team is underlevelled, not a fix bug); the heal floor correctly interrupts + RECALIBRATE resumes
           the questline objective after.
   - **FIX (BUG 2) 2026-06-28 — dashboard RATIONALE freshness. REACHES: DISPLAY (already WIRED; lag fixed).**
     RECON FINDING (contra the handoff's "not wired" premise): the live "why I'm doing this" rationale was
     ALREADY fully wired end-to-end and committed (081dfd7): `campaign._rationale_line` → `self._rationale` →
     `health.json` (`_publish_health`, line ~4385) → `pokemon_proc.health()` `game` → control-server
     `pokemon_health` → `web_dashboard/index.html` renders `g.rationale` (line 768); the `/` dashboard is
     served `no-store` (not browser-cached). The one real defect was a 1-tick LAG: `_publish_health` runs at
     the TOP of the tick (for the watchdog light) BEFORE the pick/rationale exist, so the dashboard showed the
     PREVIOUS tick's 'why' during the (visible) action execution. FIX: re-publish health right after the
     rationale is computed (before the action runs) so the dashboard reflects the CURRENT decision live.
     VERIFIED: `health.json` carries a fresh non-empty `rationale` after a run (+ the `questline` field so
     Jonny reads WHY she's off the direct path). Dashboard pixel-render is code-traced (no-cache + `g.rationale`
     bound) — only literal live-eyes pending. **COMMITTED 50b72b7.**
   - **LOSS-RESILIENCE — VERIFIED PASS 2026-06-28 (`recon_forward_loss.py`, 20-tick drive, no force-heal).**
     The real test Jonny asked for: does a LOSS break the forward drive? It does NOT. Across the run she took
     2 `battle_loss` + 4 `need_heal` (hurt retreats) on the un-fleeable Nugget-Bridge gauntlet, leveled
     L24->L26 (grinding en route), and NEVER once picked a backward-travel option — the gate questline stayed
     OPEN every tick and she stayed pointed NORTH at Bill. So a loss -> heal + grind-toward-strength while
     pointed at the objective, exactly as wanted; it never knocks her backward to a cleared dead-end.
   - **NEXT BLOCKER (foreseen headless, NOT a loss issue) — Route 24 north-traversal wedge.** The SAME loss
     trace showed her reach Route 24 (3,43, Nugget Bridge) forward, then wedge: `head_to_gym ->
     questline_no_edge` because `_run_questline_step` finds no NORTH edge from `_map_connections()` at her
     position and no-ops (returns the string instead of EXPLORING/discovering north into the unexplored
     Route 25). head_to_gym then gets no-move-pruned -> she's left with `talk_npc` + `travel:3,3` (back to
     Cerulean) -> talk_npc spam / Cerulean<->Route24 loop. The questline progression gap, exposed by the
     forward drive working well enough to GET her there. FIX (next increment): the questline executor must
     discover/explore in the step direction when there's no known edge (walk to the north map-edge to cross
     into Route 25, mirroring head_to_gym's south-discovery) AND handle the Nugget-Bridge gauntlet — needs
     its own recon (is the north connection genuinely absent at (3,43), a sub-map, or a walk-to-edge issue?).
     Until built, a full Bill-COMPLETION watch wedges at Nugget Bridge; the forward-drive BEHAVIOR
     (east->Cerulean->questline opens->drives north, survives losses) is watch-ready.
   - **RECON RESOLVED + BEND-FIX BUILT 2026-06-28 (`recon_route24.py`). The (3,43) no-edge was a BENDING
     ROUTE, not a missing connection.** Live header recon (drove her onto Route 24 with a verification boost):
     **(3,43) = Route 24 (Nugget Bridge), header `conns=[('S',(3,3)),('E',(3,44))]` — NO north exit; it
     connects EAST to (3,44) = Route 25.** So the path BENDS: Cerulean -N-> Route 24 -E-> Route 25 -> Bill.
     The KB step carries a single COARSE compass bearing ("Bill is north"), and the old executor only checked
     that one dir against the current map's edges -> at the bend (Route 24, no north edge) it no-op'd and
     stranded her. **FIX (general, `_run_questline_step`):** on no coarse-dir edge, EXPLORE the frontier —
     cross into an UNVISITED connected map (excluding the reverse of the coarse dir) to learn the bending
     route live. VERIFIED FIRING: she now logs `QUESTLINE EXPLORE: no north edge from (3,43) — crossing E
     into unexplored (3,44)` and climbs the bridge eastward, instead of the old `talk_npc` wedge (strictly
     better watchability). **STILL NOT a full chain — REMAINING (each its own increment, STOP-and-report):**
       (a) **Nugget Bridge gauntlet traversal** — a long single-file trainer line; in the headless loop she
           advances only a few tiles/tick, `need_heal` fires (her real L8/L10 teammates are underlevelled),
           she heals and bounces, never crossing in one run. A RAM HP-boost does NOT brute-force it (the game
           RECOMPUTES stats on battle entry, wiping the boosted HP) — so verifying the crossing needs a
           genuinely levelled team (grind first) or a fresh boosted SAVE, not a live RAM poke. Also saw a
           `no_path` to the east edge once — recon the bridge geometry/approach.
       (b) **Bill's-house destination-interaction — NOT BUILT.** The executor only does map-edge/frontier
           travel; the KB step is `via=talk_npc, npc=Bill, sets_flag=FLAG_GOT_SS_TICKET`. It still needs:
           enter Bill's cottage WARP (compose `enter_warp`) + TALK to Bill (`talk_npc`) to trigger the flag.
           Primitives exist (`enter_warp`/`talk_npc`); the executor's destination layer that composes them
           (and identifies Bill's specific door, no map-number per the cross-check rule) is the build.
       (c) Then verify FLAG_GOT_SS_TICKET sets + the Cerulean south gate self-clears (the questline already
           self-clears on that flag — proven). **Bottom line: bend-fix DONE+verified; full Cerulean->Bill
           traversal is NOT yet headless-verified — gated on (a) team strength + (b) the Bill interaction.**
   - **DESTINATION-INTERACTION LAYER BUILT + MECHANICS-VERIFIED 2026-06-28 (the (b) build; `recon_dest_interact.py`).**
     The general capability that makes questlines COMPLETE (not just APPROACH): a `via=talk_npc` step now,
     once traversal to the destination is exhausted, composes **enter the building (warp) -> talk the
     occupant(s) -> re-check the success flag -> exit-if-wrong-building -> try the next** — until the flag
     flips (the deriver's flag read is the done-signal, so NO map number is hardcoded; cross-check rule
     honoured). New `_questline_interact` + `_questline_unentered_door`; entered doors tracked
     (`_ql_entered_doors`, no re-entry loop); and a `_ql_inside_target` flag makes the blackout-recovery
     (which auto-exits any interior at tick-start) **cooperate** — it leaves her inside a building she
     entered ON PURPOSE for the quest, then she exits normally on a wrong building / on completion. Bill is
     the first instance; the SAME layer serves the S.S. Anne Cut handoff + every fetch-quest NPC. **VERIFIED
     (mechanics, isolated at a Cerulean building):** overworld->enter (group 3->7) -> talk occupants ×4 ->
     recognise wrong building -> exit to overworld, and it did **NOT** false-complete the questline on a
     non-Bill NPC (flag correctly stayed False). Other `via` kinds (board/use_hm) return a surfaced
     'no_interaction' (future layers). REACHES the executor/decision path; committed.
   - **STRATEGIC UNDERLEVEL-GRIND BUILT + MECHANICS-VERIFIED 2026-06-28 (Task B — the autonomous way she
     reaches gauntlet-readiness HERSELF; `recon_strategic_grind.py` 5/5). REACHES: BRAIN + DISPLAY.** The
     smart middle between the old "ace farms grass, nothing else levels" (aimless) and "charge the wall,
     lose, charge again" (stubborn). **ROOT DIAGNOSIS (recon, not symptom):** the recognition/surfacing
     ALREADY existed — `_goal_layers` said "train the team toward ~L{foe}", `loss_awareness` said "you were
     under-levelled" — but the EXECUTION fielded the WRONG mon: the `battle` action ran `grind(lead+2)`,
     training slot-0 = the ACE (Ivysaur). Classic "shown-on-display, not-wired-to-the-action" half-wire.
     **FIX (all mode-side `campaign.py` + `pokemon_strategy.py`, firewall intact, `POKEMON_STRATEGIC_GRIND=1`):**
       (1) RECOGNISE — `strat.underlevel_target()` derives the readiness target from the LIVE foe she lost
           to (`active_wall_rec()["lead_level"]` + `UNDERLEVEL_MARGIN`, default 1) — self-calibrating, no
           hardcoded map/disasm KB (cross-check rule honoured: it's the foe level she actually observed).
           `_prep_team_target(state)` fires only when there's a real active wall AND her team FLOOR (weakest
           member) is below that target — distinguishing genuine UNDERLEVEL from a type/strategy loss (floor
           already ≥ foe → None) and from a nav-bug-stuck (requires a recorded loss; the watchdog owns bugs).
       (2) ACTIVATE/SURFACE — `_available_actions` reframes `battle` to "STRENGTHEN FIRST — train the WEAK
           ones ({named}) to ~L{t} by leading with THEM, not your strongest"; the prep plan folds into her
           decision/voice ctx (place seam → BRAIN) via `prep_team_note`; `_goal_layers` SHORT + `_rationale_line`
           name the weak-grind on the dashboard (DISPLAY). Forward-drive STANDS DOWN (doesn't prune the grind)
           while prepping — so the weak-grind survives at the wall (the smart middle), not the stubborn-charge.
       (3) GRIND THE WEAK — `grind_weak_members(t)` fields each weakest under-target member as lead via
           `_swap_party_slots` (a save-safe intact 100-byte struct move — exactly the in-menu "switch order";
           XP goes to who's sent out), grinds it (existing heal-when-low = survival), repeats until the FLOOR
           crosses, then `_restore_ace` puts the highest-level mon back in slot 0.
       (4) EXIT — floor ≥ target → ace restored → existing readiness→GO / forward-drive resumes the march.
     **VERIFIED headless 5/5:** C1 party-reorder round-trips byte-for-byte (+ swap-back) = save-safe; C2
     recognition fires on real underlevel, returns None for a higher-level/strategy loss; C3 the loop fields
     rattata→spearow (NEVER the ace), exits when the floor crosses, restores ivysaur to slot 0, species
     follow their structs; C4 `battle` reframed to weak-grind + forward-drive stands down (grind survives at
     the wall); C5 `POKEMON_STRATEGIC_GRIND=0` fully reverts to `grind(lead+2)`. **NOT yet live-verified
     (same gate as Task A — needs a levelled save / live run):** real-battle XP gain (does fielding the weak
     lead actually level it) + weak-lead SURVIVAL (an L8 lead can be one-shot by a wild before the
     between-battle heal floor triggers; the proper fix = in-battle ace safety-switch, gated on the
     unverified `POKEMON_BATTLE_SWITCH`). GENERAL — recurs at every gym/gauntlet/E4. NOT committed (awaiting
     review). PROACTIVE FINDING: the live `kira_campaign.state` has MOVED to an interior map (7,3) with no
     reachable grass (handoff said Route-4) — so `recon_forward_drive2.py` P3 now reads INSPECT (its
     `"battle" in reverted` assertion can't hold with no grass); confirmed pre-existing (fails identically on
     stashed pre-change code), likely Jonny re-banking mid-grind.
   - **THE ONE REMAINING GAP for full end-to-end Bill verification = a genuinely LEVELLED team.** Confirmed
     hard: her L24/L8/L10 team can't clear the Nugget-Bridge trainer gauntlet (travel blows its wall-clock
     budget fighting the (22,5) entrance trainer), and a RAM poke CANNOT fake strength — the level write
     (0x54) doesn't even stick (the game recomputes level from EXP) and an HP/stat write is wiped by the
     battle stat-recompute. So all CODE pieces of the chain are now built+individually-verified (forward
     drive, bend-discovery traversal, destination-interaction, flag self-clear), but the full
     Cerulean->Bill->ticket->gate-opens run is NOT yet headless-verified end-to-end — it needs a properly
     LEVELLED save/checkpoint to cross the gauntlet (a Jonny grind, or a long headless grind), exactly as
     anticipated. The earlier east-edge `no_path` looks like underlevel bouncing, not a hard pathing bug
     (the bend-fix routes her east fine when she survives). **DO NOT watch until the full chain traverses+
     completes headless from a levelled save.**
   - **GROUND TRUTH RESOLVED 2026-06-28 (pret/pokefirered disasm + live RAM): the immediate gate is a
     STORY-GATE, not Cut. `kira_campaign.state` is NOT mis-positioned — it's a valid post-Misty/pre-Bill
     state.** `CeruleanCity_MapScripts → CeruleanCity_OnTransition` calls `CeruleanCity_EventScript_BlockExits`
     **on every map entry while `FLAG_GOT_SS_TICKET` (0x234) is UNSET**, which does
     `setobjectxyperm SLOWBRO 26,31` + `LASS 27,31` + `POLICEMAN 30,12` — deliberately parking the Slowbro
     (gfx 0x81) on the sole south gap to WALL the exit until you fetch the S.S. Ticket. So the correct
     canonical next step from here is **NORTH** (Route 24 Nugget Bridge → Route 25 → Bill's house →
     S.S. Ticket); the game FORCES north-first. Cut (HM01) comes LATER, from the S.S. Anne captain in
     Vermilion. The cut tree at (26,32) (gfx 95, flag `FLAG_TEMP_13`) is a SECONDARY/adjacent obstacle
     (the LittleBoy hint "if Slowbro wasn't there, could cut tree"); whether the post-ticket south road
     also needs Cut for it is a residual nuance (geometry leans yes; the canonical walkthrough walks south
     freely post-ticket) — does NOT change the immediate action (go north). So this hop is a STORY/QUESTLINE
     gate she must learn to follow, not a nav bug and not (yet) a Cut gate. See the gate-unlock design.
   - **RE-DIAGNOSED 2026-06-28 (the handoff's "two wanderer NPCs" call was WRONG — STOP-and-report):**
     the Cerulean south exit is **gated by a CUTTABLE TREE**, not blockable NPCs. Source-confirmed via
     live RAM + the rendered frame: the sole gap in the south fence is tile **(26,32)**, occupied by an
     object event with **graphicsId 95 = `GFX_CUT_TREE`** (`field_moves.GFX_CUT_TREE`, source-cited from
     pokefirered). It is flanked by two real NPCs at (26,31) gfx 129 (a pink Pokémon) + (27,31) gfx 22 (a
     person). Reachability proof (live `kira_campaign.state`, player wedged at (27,30)): her 528-tile
     reachable area reaches **ZERO** south-edge tiles without the tree; with the tree cut, all Route-5
     exits (cols 15-18, 29-32) open. So the tree is **THE SOLE south gate.**
   - **WHY THE HANDOFF MISDIAGNOSED IT:** the prior trace booted `workshop/misty_done` (player far at
     (31,22)). At that distance the cut-tree object isn't spawned (FRLG object-spawn radius) — only the two
     flanking NPC objects load — so the trace saw "two NPCs on the gap" and guessed "wanderers." The live
     campaign save (player adjacent at (27,30)) spawns the tree and reveals the truth. Confirmed the NPCs
     are STATIONARY (0 movement over 90s), same elevation (no z-mismatch), and talking/pushing does nothing.
   - **THE PRESCRIBED "wait/TTL for wanderers" FIX MUST NOT BE BUILT** — a tree never moves; waiting would
     spin forever. The Layer-A sticky-block isn't the bug either; it's just mislabelling a Cut obstacle as
     a "plain NPC."
   - **WHAT'S ACTUALLY NEEDED (two parts, both bigger than a travel.py tweak):**
     1. *Recognition (small, correct, do-able now):* in `travel._npc_tiles`/Layer-A, treat object-event
        gfx **95/97/92** (cut-tree / boulder / item-ball) as HM/field obstacles — NOT plain NPCs to
        permanent-block. `field_moves.scan_field_objects` already detects them (source-cited). Surface
        "there's a Cut tree here; I need Cut" to the oracle instead of sticky-blocking. This makes the
        diagnosis HONEST in-game but does **not** unblock the chain by itself.
     2. *Progression (the real gate — WIP, currently OFF):* she does **NOT** know HM01 Cut (party moves
        carry no move-id 15; she DOES have the Cascade badge, so the badge gate is satisfied — she just
        lacks the move). Passing the tree needs the **Cut questline** (Bill on Route 25 → S.S. Ticket →
        S.S. Anne in Vermilion → HM01 Cut) AND Cut **actuation** (`field_moves`, `POKEMON_FIELD_MOVES=0`,
        actuation unverified on a long-running core). NOTE the apparent chicken-and-egg (Cut comes from
        Vermilion, which is south past the tree) — needs a ground-truth pass on the intended FRLG route
        (is there an alternate early path, or is this save mis-positioned?). **This is a real progression
        wall, not a nav bug — Cerulean→Vermilion CANNOT traverse headless until Cut + actuation exist.**
   - **LATENT LIVE CRASH — FIXED 2026-06-28:** `play_live.py` did NOT force utf-8 stdout, and `campaign.py`
     logs `→` in goal/plan ctx lines (2994/3747/4262). On a cp1252 console that raises `UnicodeEncodeError`
     and kills the run on that tick (the same crash that hit the trace harness). Added the utf-8
     `sys.stdout/stderr.reconfigure` guard at the top of `play_live.py` (isolated, additive; syntax-checked).
   - **EXACT NEXT STEPS for a fresh context:** (a) DECISION for Jonny/PM: build the Cut questline +
     field-move actuation (the real unblock), or re-confirm the intended FRLG Cerulean→Vermilion route
     first (recon whether an alternate early path exists / whether `kira_campaign.state` is mis-positioned
     in a Mart pocket). (b) Land the small "recognise gfx 95/97/92 as field obstacles, don't plain-block"
     change in travel/Layer-A so the in-game state is honest. (c) Only after Cut works: continue the chain
     Route5→UGP→Route6→Vermilion, each hop cross-checked live. NO watch until the headless trace reaches
     Vermilion (3,5). Recon scripts for this live in `pokemon_agent/recon_cerulean_*.py` +
     `recon_choke_verdict.py` (read-only).
8. **Vision confirming-vote + Gemini swap (recon delivered, NOT built).** Layer B is wired for a pixel-vote
   to plug in later; core-Kira vision OpenAI→Gemini swap recon done (valid key = `GEMINI_IMAGE_API_KEY`;
   `google-genai` installed; recommend `gemini-3.1-flash-lite` heartbeat → `gemini-3-flash-preview` escalate).
   Separate step; firewall (all modes). Needs a real-frame test before commit.

## 6. BOTTOM LINE
Core Kira's decision-wiring is healthy (the major features reach the brain). The Pokémon harness had two
half-wires (Gary arc, saga-in-reactions) — both fixed today. The remaining honest gaps are all either
GATED-with-reason (switch/HM actuation pending live verify) or flagged WIP — none are silent "looks done
but isn't." The one core decision outstanding: whether the dread/vendetta/naivety persona arcs should be
promoted from gitignored-persona-only into tracked code.
