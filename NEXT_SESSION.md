# KIRA FIRERED - SINGLE MISSION (2026-07-13). SUPERSEDES EVERYTHING.
CEO decision: the 5x battery is CANCELLED. The asymptote = ONE QUALIFYING RUN, today. When it lands, STOP THE TRAIN. Read ONLY this file + the latest survey + live log tails. Prior directives are archived at NEXT_SESSION_archive_2026-07-13_*.md - consult ONLY for the proven per-stage launch recipes; do not re-derive them; no history spelunking.

## ▶ MONITOR fresh_go_3 (attended questline-guard fix SHIPPED 2026-07-13 14:00, commit 5add821). TRAIN RE-ARMED.
The 2nd-disqualify ace-runaway root was FIXED attended (Jonny at desk): BOTH questline bench guards
(`_road_bench_xp_arm` ~7317 AND `_bench_severely_lopsided` ~7480) were relaxed + re-gated on MAP-TYPE, so the
bench leads/grinds on OPEN GROUND while the true ace still leads inside caves/gauntlets (helpers
`_on_overworld_now` / `_questline_march_bench_ok`, flag `POKEMON_QUESTLINE_BENCH_RELAX` default ON in code).
VERIFIED live: on fresh_go_3 Route 4, WITH the S.S.-Ticket questline ACTIVE, `LOPSIDED-BENCH` fired and pulled
the weak bench mon up (rattata L8→L10, ace held, 0 wedge, 0 tracebacks) — behavior the old code suppressed.
Full write-up = NIGHT_REPORT.md; details = memory `pokemon-freshgo2-halt-levelgap-questlineguard`.
**FRONTIER = monitor fresh_go_3** (cold FRESH, detached, watchdog, log `G:/temp/longrun/fresh_go_3.log`).
Cleared the opening → Misty = badge 2 → free_roam, building the six at the Nugget-Bridge team-depth wall.
**SHIFT-22 STAMP (16:49) — HEALTHY, BEAT SABRINA (gym 6) → badge 6 + CAUGHT LAPRAS, now on the Seafoam route to Blaine (gym 7); ACUTE gap WIDENED ~29→33, glance-clean 0-fix.**
Single watchdog verified (bash 144483 started 15:48 → python child **147532 = ITER 3, relaunched 16:46:28 via CLEAN
carry-forward** — iter 2 hit a self-recovering LOUD **STALL** "no progress for 14 decisions" at (3,4)@(11,19), banked_STALL,
watchdog carried forward from `banked_LIVE` badges=6 → iter 3 booted map=(3,10)@(46,13) badges=6; NORMAL watchdog machinery,
single-run law holds, qualifying-safe self-recovering abort — NOT a hard wedge). Log LIVE (mtime=now @16:48, **102,337 lines,
+1,380 since shift 21**, grew during the glance), **0 tracebacks whole-log**. **208 motion / 3 spin in last 400 = moving-not-
spinning** (live Seafoam Strength puzzle: `push`/`ladder`/`fall`/`strength` ops at map (1,83-85); Bridge INPUT OWNER=agent sole
writer). **PROGRESS SINCE SHIFT 21:** **BEAT SABRINA (gym 6) → badge 6 (Marsh)**, **CAUGHT LAPRAS** (the Surf-mon — raticate
boxed to field it, so party is still SIX DISTINCT), and is now marching Fuchsia → Route 19/20 → **SEAFOAM ISLANDS** doing the
Strength boulder-push puzzle to still the current and surf to Cinnabar for **BLAINE (gym 7)**. **THE HISTORICAL SURF-MON
SEA-GATE WEDGE IS DEAD HERE** (memory `pokemon-freshgo-surf-mon-seagate-wedge`) — this organic run built a Surf-capable six,
so the Cinnabar crossing that wedged every prior organic fresh run is passable (mirrors fresh_go_2 shift-25). Party **[venusaur
L58, kadabra L30, dugtrio L29, growlithe L28, fearow L29, lapras L25] — SIX DISTINCT, 0 dups, dex 13, badges=6** (Boulder,
Cascade, Thunder, Rainbow, Soul, Marsh); raticate boxed for Lapras (still six distinct in party, qualifying OK). **NO live
banked_CREDITS** (real sentinels OUTCOME:CREDITS / Hall-of-Fame / rolled-credits = **0** whole-log; archived banks are prior-
run/fresh_go_2 leftovers, NOT fresh_go_3). **ACUTE gap WIDENED ~29→33 this window** (ace crept L57→L58 solo-leading the Sabrina
gym fight while the bench floor DROPPED to L25 with the fresh Lapras catch; growlithe still L28 → ~30 gap excl. Lapras) —
expected ace-runaway on the gym-fight + sea-questline leg (ace-led); **now widened 5 windows running — THIS IS THE RUN'S CENTRAL
QUALIFYING RISK**, the bench MUST start climbing at the post-badge-6 grind + Blaine/Giovanni/VR milestone ladder or the run
fails qualifying on team-shape (E4 entry needs all ≥L42, gap ≤15; Lapras L25 + growlithe L28 must reach ~L50 while the ace
holds). Nothing to fix, no flags flipped mid-run (monitor-only). NEXT SHIFT = same glance (clear Seafoam → surf to Cinnabar →
beat Blaine(7) → Giovanni(8) → VR → E4 → credits; six stay distinct; **track the gap TURNS/CLOSES — it must reverse now or the
run fails qualifying**; on banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-21 STAMP (16:46) — HEALTHY, cleared Silph pad-maze + Sabrina's juniors, now ENGAGING SABRINA (gym 6); ACUTE gap widened ~27→29, glance-clean 0-fix.**
Single watchdog verified (bash 43448 started 15:48 → bash 41612 → py 32920 shim → py 9732 real, all iter-2 @16:37:55 —
SAME long-lived iter as shift 20, single-run law holds). Log LIVE (mtime=now @16:46, **100,957 lines, +389 since shift
20**, grew during the glance), **0 tracebacks whole-log**. **143 motion / 3 spin in last 400 = moving-not-spinning** (all
3 spins = `TRAVEL WEDGE: identical fp x4 → returning to roam LOUD (no inner spin)` = self-recovering, NOT wedges; the
Sabrina-challenge `frozen world x15/90` = normal dialogue pacing). **PROGRESS SINCE SHIFT 20:** cleared the Silph Co
teleport-pad maze, **beat Sabrina's junior trainers** (`GYM: all junior trainers cleared (beaten obj [1, 3])`) and is
now **ENGAGING SABRINA (gym 6)** at the leader tile (14,12) in her challenge dialogue (`'I had a vision of your
arrival...'`, lead **venusaur L57, party=6**), in Saffron City. Party **[venusaur L57, raticate L29, kadabra L30, dugtrio
L29, growlithe L28, fearow L29] — SIX DISTINCT, 0 dups, dex 13, badges=5** (Boulder, Cascade, Thunder, Rainbow, Soul).
banked_LIVE fresh 16:46. **NO live banked_CREDITS** (real sentinels OUTCOME:CREDITS / Hall-of-Fame / rolled-credits = **0**
whole-log; the SABRINA/SILPH/CINNABAR/E4/VICTORY checkpoint banks on disk are OLD 2026-07-07..07-11 leftovers from prior
runs/fresh_go_2 — verified by mtime, NOT fresh_go_3; `banked_CREDITS_archived_*` is fresh_go_2's morning credits). **ACUTE
gap WIDENED ~27→29 this window** (ace crept L55→L57 across the Silph pad-maze / Gary 7F / Sabrina juniors while bench floor
HELD L28 growlithe; fearow crept L28→L29) — the pin rose to L40 last shift but the Silph/Sabrina **gym-fight leg** let the
ace run AGAIN (expected: gym fights are ace-led). The TURN must come from the **post-Sabrina bench grind + gyms 7-8 + VR**
milestone ladder — **now widened 4 windows running; this is the run's central qualifying risk — watch it REVERSE next
shift or the run fails qualifying on team-shape** (E4 entry needs all ≥L42, gap ≤15). Nothing to fix, no flags flipped
mid-run. NEXT SHIFT = same glance (beat Sabrina → Blaine(7) → Giovanni(8) → VR → E4 → credits; six stay distinct;
**track the gap TURNS/CLOSES — bench MUST start climbing at the Sabrina/Silph grind**; on banked_CREDITS → order 6).
Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-20 STAMP (16:42) — HEALTHY, now INSIDE SILPH CO. (Saffron) en route to Sabrina (gym 6); gap-closing pin RAISED to L40, glance-clean 0-fix.**
Single watchdog verified (bash 144483 started 15:48 + python child 146950 = iter 2 started 16:37:55 — SAME long-lived
iter as shift 19, NOT a relaunch; single-run law holds). Log LIVE (mtime=now @16:42, **100,568 lines, +335 since shift
19**, grew during the glance), **0 tracebacks whole-log**. **193 motion / 0 spin in last 400 = moving-not-spinning**
(tail = live pad-rides `[pad-3f]/[pad-7f]` + `[9f] hostage heal -> talked` + `7F pocket: walking the rival trigger row
(Gary auto-engages)` + a `[lapras]` step — all live Silph-Co questline actuation, Bridge INPUT OWNER=agent sole writer).
**PROGRESS SINCE SHIFT 19:** advanced from the Saffron *approach* → INTO **SILPH CO.**, now climbing the teleport-pad
maze (floors 3F/7F/9F), **fighting Gary on 7F** and working the Silph questline (hostage-heal / Lapras / Master-Ball floor)
en route to **Sabrina (gym 6)**. **GAP-CLOSING NOW ARMED:** the RATIONALE shows the bench milestone pin **RAISED to
~L40** (`"Level the weak ones — raticate, kadabra, dugtrio, growlithe and fearow — to ~L40 by fielding THEM in the
grass, THEN retry"`) — the mechanism to reverse the ace-runaway is engaged. Party **[venusaur L55, raticate L29, kadabra
L30, dugtrio L29, growlithe L28, fearow L28] — SIX DISTINCT, 0 dups, dex 12, badges=5** (Boulder, Cascade, Thunder,
Rainbow, Soul). banked_LIVE/banked_STALL both 16:37 = carry-forward from the iter-2 boot, NOT fresh stalls. **NO live
banked_CREDITS** (real sentinels OUTCOME:CREDITS / Hall-of-Fame / rolled-credits = **0** whole-log; the
`banked_CREDITS_archived_*` dir is fresh_go_2's morning credits, NOT fresh_go_3). **ACUTE gap HELD ~27** (ace L55 vs
bench floor L28 — unchanged this window; but the pin just rose L28→L40, so the close should begin at the Silph/Sabrina
grind — **watch it TURN next shift**; widened 3 windows through shift 19, must now reverse). Must still close to ≤15 (all
≥L42) by E4. Nothing to fix, no flags flipped. NEXT SHIFT = same glance (clear Silph Co → beat Sabrina → Blaine(7) →
Giovanni(8) → VR → E4 → credits; six stay distinct; **track the gap CLOSES now the pin is L40**; on banked_CREDITS →
order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-19 STAMP (16:40) — HEALTHY, BEAT KOGA (gym 5) → badge 5, now at SABRINA (gym 6) approach in Saffron, glance-clean 0-fix.**
Single watchdog verified (bash 144483 started 15:48; python child **146950 = iter 2, relaunched 16:37:55 via CLEAN
carry-forward** — iter 1 stalled/exited rc=0, iter 2 booted `banked_LIVE` at the Koga tile map(11,3)@(7,14) badges=4,
BEAT Koga → real **badge 4→5** progress; NORMAL watchdog machinery, single-run law holds). Log LIVE (mtime=now @16:40,
**100,233 lines, +3,433 since shift 18**, grew ~600 lines DURING this glance), **0 tracebacks whole-log**. **265 motion /
5 spin in last 400 = moving-not-spinning** (2 spins = Koga-challenge dialogue-pacing `frozen world x15/30/90`; 3 = deep-
wedge-ring / TRAVEL-WEDGE self-recovering LOUD "no inner spin" — all benign, NOT wedges). **PROGRESS SINCE SHIFT 18:**
**beat KOGA (gym 5) → badge 5 (Soul)** → marched Route 7/8 → Lavender → Route 12 → **now IN Saffron City (58,27)** for
**Sabrina (gym 6)**. Party **[venusaur L55, raticate L29, kadabra L30, dugtrio L29, growlithe L28, fearow L28] — SIX
DISTINCT, 0 dups, dex 12, badges=5** (Boulder, Cascade, Thunder, Rainbow, Soul). **NO live banked_CREDITS** (real sentinels
OUTCOME:CREDITS / Hall-of-Fame / rolled-credits = **0** whole-log; the `banked_CREDITS_archived_*` dir is fresh_go_2's
morning credits, NOT fresh_go_3). **ACUTE gap WIDENED 26→27 this window** (ace crept L54→L55 beating Koga while bench floor
HELD L28 — expected ace-runaway on the gym-fight leg; the Sabrina(6) pin + gyms 6-8 + VR milestone ladder must pull the
bench up and REVERSE this — **now widened 3 windows running, watch it TURN at the Sabrina/Silph pins**). Must still close to
≤15 (all ≥L42) by E4. Nothing to fix, no flags flipped. NEXT SHIFT = same glance (beat Sabrina → Blaine(7) → Giovanni(8) →
VR → E4 → credits; six stay distinct; track the gap REVERSES; on banked_CREDITS → order 6). Healthy → glance cheap, exit.
Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-18 STAMP (16:36) — HEALTHY, reached Fuchsia and now ENGAGING KOGA (gym 5), glance-clean 0-fix.**
Single watchdog chain verified (bash 43448 → 40260 → py 27064 shim → 42948 real, ALL started 15:48 — SAME
long-lived iter as shifts 14-17, single-run law holds, NOT a relaunch). Log LIVE (mtime=now @16:35, **96,800
lines, +2,165 since shift 17**), **0 tracebacks whole-log**. **312 motion / 2 spin in last 400 = moving-not-
spinning** (the 2 "spins" are the normal Koga-challenge dialogue-pacing counter `frozen world x15/90`, NOT a
wedge). **PROGRESS SINCE SHIFT 17:** advanced from the Fuchsia *approach* → into **Fuchsia City** → now
**ENGAGING KOGA (gym 5)** at the leader tile (7,14), in the challenge dialogue (`[dlg Koga-challenge] 'KOGA.
Fwahahaha! A mere child like you dares to challenge me?'`, lead **venusaur Lv54, party=6**). Party **[venusaur
L54, kadabra L30, raticate L29, dugtrio L29, growlithe L28, fearow L28] — SIX DISTINCT, 0 dups, dex 12,
badges=4** (Boulder, Cascade, Thunder, Rainbow). **NO live banked_CREDITS** (real credits sentinels
OUTCOME:CREDITS / Hall-of-Fame / rolled-credits = **0** whole-log; the 604 "champion" grep hits are ALL oracle
goal-text "beat the Elite Four … Champion down the line", NOT a credits event). One `[strat] trainer battle
returned ambiguous 'ended' → RECORDING as LOSS` in tail = the designed swallow-proof loss-recording (a Koga
junior), NOT a crash. **ACUTE gap WIDENED 24→26 this window** (ace crept L52→L54 on the Fuchsia march while
bench floor HELD L28 — expected ace-runaway-on-questline-leg; the Koga pin + gyms 5-8 milestone ladder must
pull the bench up and REVERSE this — now widened 2 windows running, watch it TURN at the Koga pin). Must still
close to ≤15 (all ≥L42) by E4. Nothing to fix, no flags flipped. NEXT SHIFT = same glance (beat Koga → Sabrina/
Silph(6) → Blaine(7) → Giovanni(8) → VR → E4 → credits; six stay distinct; track the gap REVERSES; on
banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-17 STAMP (16:33) — HEALTHY, advanced out of Pokémon Tower to the FUCHSIA approach (Koga/gym-5 march), glance-clean 0-fix.**
Single watchdog chain verified (bash 43448 → 40260 → py 27064 shim → 42948 real, ALL started 15:48; single-run
law holds — SAME long-lived iter as shifts 14-16, NOT a relaunch. NOTE: shifts 15/16 cited PIDs 144483/144491 —
those are the git-bash-namespace PIDs for the SAME processes; the Windows PIDs are 43448/40260/27064/42948, real
python 27064/42948 consistent across shifts 14-17). Log LIVE (mtime=now @16:32, **94,635 lines, +380 since shift
16**), **0 tracebacks whole-log**. **205 motion / 0 spin-wedge in last 400 = moving-not-spinning** (live travel
steps on map (3,33) + `blocker is a TRAINER - fighting through`, Bridge INPUT OWNER=agent sole writer; a
self-recovering `HEAL-RETURN` excursion routing to Viridian in flight = LOUD, not a wedge). **PROGRESS SINCE
SHIFT 16:** advanced out of **POKÉMON TOWER** → now at the **FUCHSIA approach (map (3,33))** fighting through
route trainers toward **Koga (gym 5)**. Party **[venusaur L52, raticate L28, kadabra L30, dugtrio L28, growlithe
L28, fearow L28] — SIX DISTINCT, 0 dups, dex 12, badges=4** (Boulder, Cascade, Thunder, Rainbow). banked_LIVE
fresh 16:30; **NO live banked_CREDITS** (a `banked_CREDITS*` glob matched ONLY the archived
`banked_CREDITS_archived_fresh_go_2_final_0713_1354` dir from this morning — VERIFIED not a fresh_go_3 credits;
log has 0 CREDITS/champion events). **ACUTE gap WIDENED 21→24 this window** (ace crept L49→L52 on the
Tower→Fuchsia questline leg while bench floor HELD L28 — the expected ace-runaway-on-questline-legs pattern; the
milestone ladder at the Koga pin + gyms 5-8 must pull the bench up to reverse it). Must still close to ≤15 (all
≥L42) by E4 — **watch the gap REVERSES at the Koga pin next shift, not keeps opening**. Nothing to fix, no flags
flipped. NEXT SHIFT = same glance (Fuchsia→Koga→Sabrina→Blaine→Giovanni→VR→E4→credits; six stay distinct; on
banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-16 STAMP (16:29) — HEALTHY, bench CROSSED the L28 Koga pin + advanced to POKÉMON TOWER, glance-clean 0-fix.**
Single watchdog chain verified (bash 144483 → py 144491, both started 15:48; single-run law holds — SAME
long-lived iter as shifts 14/15, NOT a relaunch). Log LIVE (mtime=now @16:29, growing +7 lines/6s, **94,255
lines, +1.5K since shift 15**), **0 tracebacks whole-log**. 184 motion / 2 "spin" in last 400 — but BOTH spin
hits were `TRAVEL WEDGE ... returning to roam LOUD (no inner spin)` = self-recovering, NOT real spin-wedges =
moving-not-spinning. **PROGRESS SINCE SHIFT 15: the Route-7 L28 bench grind COMPLETED** — decision #32→#54
(sim 1715→2398s, +683 sim-sec), **all five bench mons crossed the L28 Koga pin and are now UNIFORM L28** (was
mixed L24-26). She then advanced Route 7 → **POKÉMON TOWER** (map (1,88)/(1,93), (11,18)) working the Silph-
Scope/Mr-Fuji/**Poké-Flute→Snorlax** questline (ql="the Poké Flute … wakes a sleeping Snorlax") — a sensible
Celadon-area leg en route to Fuchsia/Koga. Live tail = climbing Tower floors (path 37→31→24 to exit = forward),
fighting trainers (`blocker battle outcome=win`), fresh ENCOUNTERs, Bridge INPUT OWNER=agent (sole writer).
Party **[venusaur L49, raticate L28, kadabra L28, dugtrio L28, growlithe L28, fearow L28] — SIX DISTINCT, 0
dups, dex 12, badges=4** (Boulder, Cascade, Thunder, Rainbow). banked_LIVE fresh 16:26, no banked_CREDITS.
`banked_TIMEOUT_healed` (16:27) = a Route-7 heal-excursion that SELF-HEALED (many `HEAL-EXCURSION: healed +
back on (3,25)` lines then she LEFT for the Tower) — self-recovering, not a wedge. **ACUTE gap NARROWED 23→21**
(ace L49 vs bench floor L28 — bench OUTPACED the ace this window: +4 floor vs +2 ace = right direction); must
still close to ≤15 (all ≥L42) by E4 via the milestone ladder; track the floor each glance. Nothing to fix, no
flags flipped. NEXT SHIFT = same glance (Poké Flute→Snorlax→Fuchsia→Koga→Sabrina→Blaine→Giovanni→VR→E4→credits;
six stay distinct; on banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in:
`play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-15 STAMP (16:26) — HEALTHY, bench grind to Koga L28 pin STILL ADVANCING, glance-clean 0-fix.**
Single watchdog chain verified (bash 144483 → py 144491, shim 27064 / real 42948, all started 15:48; single-run
law holds — SAME long-lived iter as shift 14, NOT a relaunch). Log LIVE (mtime=now @16:25, 92,713 lines, **+1.9K
since shift 14**), **0 tracebacks whole-log**. 297 motion / 0 spin-wedge in last 400 = moving-not-spinning (Route 7
(3,25) grass grind). Decision **#32 @1715.7s** still the current long grind action (spanning many battles at one
sim-stamp = EXPECTED not a freeze — verified via +1.9K log lines & 297 motion events since shift 14, and raticate
crept **L24→L26** = real forward XP). Party **[venusaur L47, kadabra L27, dugtrio L26, growlithe L26, raticate L26,
fearow L24] — SIX DISTINCT, 0 dups, dex 12, badges=4**; bench floor fearow L24, ace HELD L47 (milestone-pinned,
NOT runaway). banked_LIVE fresh 16:15, no banked_CREDITS. **ACUTE gap = ace L47 vs bench floor L24 = ~23** — must
close to ≤15 (all ≥L42) by E4 via the milestone ladder; track the floor each glance. NEXT SHIFT = same glance
(cross L28 pin → Fuchsia → Koga → Sabrina → Blaine → Giovanni → VR → E4 → credits; six stay distinct; on
banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-14 STAMP (16:23) — HEALTHY, bench grind to Koga L28 pin STILL ADVANCING, glance-clean 0-fix.**
Single watchdog chain verified (bash 43448 → 40260 → py 27064 shim → 42948 real, all started 15:48; single-run
law holds — SAME long-lived watchdog iter, NOT a relaunch). Log LIVE (mtime=now @16:22, 90,845 lines, **+1.8K
since shift 13**), **0 tracebacks whole-log**. 281 motion / 0 spin-wedge in last 400 = moving-not-spinning (raw
tail = live travel steps 13,5→13,4 on Route 7 (3,25) + one normal NPC-gap wait + fresh ENCOUNTER; Bridge INPUT
OWNER = agent sole writer). Decision **#32 @1715.7s** still the current long grind action (spanning many battles
at one sim-stamp = EXPECTED not a freeze — verified via +1.8K log lines & 281 motion events since shift 13);
GRIND-WEAK fielding slot 3 (dugtrio L26) as lead. Party **[venusaur L47, kadabra L27, dugtrio L26, growlithe L26,
raticate L26, fearow L24] — SIX DISTINCT, 0 dups, dex 12, badges=4** — unchanged from shift 13 (2-min glance);
bench floor fearow L24, ace HELD L47 (milestone-pinned, NOT runaway). banked_LIVE fresh 16:15, no banked_CREDITS.
**ACUTE gap = ace L47 vs bench floor L24 = ~23** — must close to ≤15 (all ≥L42) by E4 via the milestone ladder;
track the floor each glance. Move-drop decisions firing (move management healthy — kept coverage). Nothing to fix,
no flags flipped. NEXT SHIFT = same glance (cross L28 pin → Fuchsia → Koga → Sabrina → Blaine → Giovanni → VR →
E4 → credits; six stay distinct; on banked_CREDITS → order 6). Healthy → glance cheap, exit. Canonical UNTOUCHED.
Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-13 STAMP (16:20) — HEALTHY, bench grind to Koga ADVANCING, glance-clean 0-fix.** Same watchdog
iter as shift 12 — **ONE** chain verified (py 144491 → bash 144483 → 144478 → 27064, all started 15:48;
single-run law holds). Log live (mtime=now @16:20), **0 tracebacks (89,035 lines, +1.2K since shift 12)**,
0 spin-wedge / 80 win-events in last 400 = moving-not-spinning. Decisions ADVANCING #30→#31→#32
(922s→1610s→1715s); since shift 12 rattata→raticate crept L24→L26 (real forward XP). Party **[venusaur L47,
kadabra L27, dugtrio L26, growlithe L26, raticate L26, fearow L24] — SIX DISTINCT, 0 dups, dex 12, badges=4**;
bench floor now fearow L24, ace HELD L47 (milestone-pinned, NOT runaway). Still Route 7 (3,25), prep=28
(Koga L28 pin), grinding the bench for Fuchsia/Koga (gym 5). banked_LIVE fresh 16:15, no banked_CREDITS.
**ACUTE gap = ace L47 vs bench floor L24 = ~23** — must close to ≤15 (all ≥L42) by E4 via the milestone
ladder; track the floor each glance. Nothing to fix, no flags flipped. NEXT SHIFT = same glance
(cross L28 pin → Fuchsia → Koga → Sabrina → Blaine → Giovanni → VR → E4 → credits; six stay distinct; on
banked_CREDITS → order 6: run_stats → QUALIFYING eval → CREDITS/WATCHABILITY-GAPS/HALT line 1 of
NIGHT_REPORT.md). Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-12 STAMP (16:16) — HEALTHY, grind stint RESOLVED with real progress, glance-clean 0-fix.** The
shift-9..11 decision-#30 Route-7 grind stint has COMPLETED: since shift 11 the bench climbed floor L23→L24
and across the board (kadabra L24→L27, growlithe L23→L26, raticate L24→L26) and **DIGLETT EVOLVED TO
DUGTRIO** — unambiguous forward progress, the shift-11 stall-watch is closed. Party now **[venusaur L47,
kadabra L27, dugtrio L26, growlithe L26, raticate L26, fearow L24] — SIX DISTINCT, 0 dups, badges=4**.
**ONE** watchdog verified (bash 144483 + py 144491, ancestor 27064, all started 15:48; new python PID vs
shift 11's 42948 = normal watchdog carry-forward relaunch between grind stints — single-run law holds via
parent-PID chain). Log live (mtime=now @16:16), **0 tracebacks (87.8K lines, +2.2K since shift 11)**,
0 spin-wedge / 69 win-events in last 400 = moving-not-spinning (live travel steps 15,4→14,5 on Route 7
(3,25) + fresh ENCOUNTERs). GRIND-WEAK/LOPSIDED firing textbook (*"grinding kadabra, dugtrio, growlithe,
raticate and fearow up to ~L28 (fielding them, not my ace)"*, fielding slot 5 fearow L24 as lead); ace
HELD/crept L45→L47 (milestone-pinned, NOT runaway). banked_LIVE fresh, no banked_CREDITS. **ACUTE gap =
ace L47 vs bench floor L24 = ~23** (absolute wider bc ace crept +2, but bench IS climbing steadily) — must
close to ≤15 (all ≥L42) by E4 via the milestone ladder; track the floor each glance. PACING NOTE (Jonny
adjudication only): the L28 grind spanned shifts 9-12 but is bounded + now completing (bench +3 across the
board, an evolution) — WATCHABILITY item, NOT a fix/relaunch trigger; no flags flipped mid-run. NEXT SHIFT =
same glance (Koga→Sabrina→Blaine→Giovanni→VR→E4→credits; six stay distinct; on banked_CREDITS → order 6).
Healthy → glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-11 STAMP (16:12) — HEALTHY, unchanged trajectory, glance-clean 0-fix.** Same watchdog iter as
shifts 6-10 — **ONE** watchdog (bash 43448 nohup-script → child subshell 40260) + worker = 2-PID venv shim
(py 27064 shim + py 42948 real, CPU 1451s), all started 15:48 — single-run law verified via parent-PID chain.
Log live (mtime=now @16:12), **0 tracebacks (85.6K lines)**, 0 spin-wedge / 79 win-events in last 400 =
moving-not-spinning (grass-grind on Route 7 (3,25); raw tail = live travel steps 16,4→15,4→15,3→16,3 + fresh
ENCOUNTERs). banked_LIVE fresh 16:03, no banked_CREDITS. Party **[venusaur L45, diglett L23, growlithe L23,
kadabra L24, raticate L24, fearow L24] — SIX DISTINCT, 0 dups, dex 11, badges=4**. Still decision **#30 @922s**
— now the **3rd shift on this one long grind action** to the L28 Koga pin; verified GENUINE not frozen (+2.2K
log lines & 79 fresh wins since shift 10; bench floor slow-climbing L22→L23, raticate L22→L24; ace HELD L45).
**ACUTE gap = ace L45 vs bench floor L23 = ~22**, must close to ≤15 (all ≥L42) by E4 via the milestone ladder —
track the floor each glance. **PACING NOTE (Jonny adjudication only, per mission addendum):** 3 shifts on one
Route-7 low-XP grind stint to L28 is a WATCHABILITY item, **NOT a fix/relaunch trigger** — the NS#5
`_better_grind_spot` efficiency lever (flag-OFF) is the theoretical fix but do **NOT** flip flags mid-run;
level curve is monotonic, build doing its job. Nothing to fix, no flags flipped. NEXT SHIFT = same glance
(Koga→Sabrina→Blaine→Giovanni→VR→E4→credits; six stay distinct; on banked_CREDITS → order 6). Healthy →
glance cheap, exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
———————————————————————————————————————————————————————————————
**SHIFT-10 STAMP (16:09) — HEALTHY, unchanged trajectory, glance-clean 0-fix.** Same watchdog iter as
shifts 6-9 (py 144491 + bash 144483, started 15:48) alive; log live (mtime=now), **0 tracebacks (83.4K lines)**,
0 spin-wedge / 79 win-events in last 400 = moving-not-spinning (grass-grind on Route 7 (3,25), coords
oscillating 15-17,3-5 + one normal NPC-gap wait). banked_LIVE fresh 16:03, no banked_CREDITS. Party
**[venusaur L45, diglett L23, growlithe L23, kadabra L24, raticate L24, fearow L24] — SIX DISTINCT, 0 dups,
dex 11, badges=4**. Still decision **#30 @922s** (same long grind action as shift 9 spanning many battles at
one sim-stamp = EXPECTED not a freeze — verified via 79 fresh wins + log +1.1K lines since shift 9). GRIND-WEAK
fielding weak slots to the L28 Koga pin, ace HELD L45. **ACUTE gap = ace L45 vs bench floor L23 = ~22**, must
close to ≤15 (all ≥L42) by E4 via the milestone ladder — track the floor each glance. Nothing to fix, no flags
flipped. NEXT SHIFT = same glance (Koga→Sabrina→Blaine→Giovanni→VR→E4→credits; six stay distinct; on
banked_CREDITS → order 6). ————————————————————————————————————————————————
**SHIFT-8 STAMP (16:05) — HEALTHY, unchanged trajectory, glance-clean 0-fix.** Same watchdog iter as
shift 7 (py 144491 + bash 144483, started 15:48) alive; log live (mtime=now), **0 tracebacks (80.8K lines)**,
61 motion / 0 spin-wedge lines in last 300 = moving-not-spinning (travel ARRIVED crossing (3,25)↔(3,6)); banked_LIVE
fresh 16:03, no banked_CREDITS. Party **[venusaur L45, diglett L23, growlithe L23, kadabra L24, raticate L24,
fearow L24] — SIX DISTINCT, 0 dups, dex 11, badges=4**, GRIND-WEAK/LOPSIDED firing textbook (ace **HELD L45**,
bench climbing to the L28 pin, narrated *"fielding them, not my ace"*), heading Celadon→Fuchsia for Koga. Ace
crept +1 (L44→L45) vs shift 7 — milestone-pinned, NOT runaway. **ACUTE gap = ace L45 vs bench floor L23 = ~22**,
must close to ≤15 (all ≥L42) by E4 via the milestone ladder — track the floor each glance. Nothing to fix, no
flags flipped. NEXT SHIFT = same glance (Koga→Sabrina→Blaine→Giovanni→VR→E4→credits; six stay distinct; on
banked_CREDITS → order 6). ————————————————————————————————————————————————
**SHIFT-7 STAMP (16:02) — HEALTHY, unchanged trajectory, glance-clean 0-fix.** Same watchdog iter as
shift 6 (py 144491 + bash 144483, started 15:48) alive; log live (mtime=now), **0 tracebacks (79.5K lines)**,
40 motion events / 0 spin-wedge lines in tail = moving-not-spinning; banked_LIVE fresh 15:51, no banked_CREDITS.
Party **[venusaur L44, diglett L23, growlithe L23, kadabra L24, raticate L22, fearow L24] — SIX DISTINCT, 0
dups, dex 11, badges=4**, GRIND-WEAK/LOPSIDED firing textbook (ace HELD L44, bench climbing kadabra L22→L24,
narrated *"fielding them, not my ace"*), heading Celadon→Fuchsia for Koga. **ACUTE gap = ace L44 vs bench floor
L22 = ~22**, must close to ≤15 (all ≥L42) by E4 via the milestone ladder — track the floor each glance. Nothing
to fix, no flags flipped. NEXT SHIFT = same glance as below (Koga→Sabrina→Blaine→Giovanni→VR→E4→credits;
six stay distinct; on banked_CREDITS → order 6). ————————————————————————————————————————————————
**SHIFT-6 STAMP (15:59) — HEALTHY, still bench-grinding to Koga (gym 5), glance-clean.** Fresh watchdog iter
(py 144491, started 15:48) at sim ~635s: on **Route 7 (map 3,25), badges=4**, GRIND-WEAK/LOPSIDED-BENCH firing
textbook (`forcing ONE dedicated grind stint`, ace **HELD**, RATIONALE *"grinding … up to ~L28 (fielding them,
not my ace) … then on to Fuchsia for Koga"*). Party **[venusaur L44, diglett L23, growlithe L23, kadabra L22,
raticate L22, fearow L24] — SIX DISTINCT, 0 dups, dex 11**. **0 tracebacks** (77.9K lines), log live (mtime=now),
sim advancing (#26→#28), bench climbing (growlithe L20→L23, fearow L22→L24), banked_LIVE 15:51. **ACUTE gap =
ace L44 vs bench floor L22 = ~22**, GRIND-WEAK closing it (Koga pin L28). Same trajectory as shift 5; nothing to
fix, no flags flipped. **NEXT SHIFT — glance:** does she cross the L28 pin → reach **Fuchsia → beat Koga (gym 5)**
→ Sabrina/Silph(6) → Cinnabar/Blaine(7) → Giovanni(8) → VR → E4 → credits; track the floor+gap each glance (must
reach ≥L42 all, gap ≤15 by E4 entry via milestone ladder); six stay distinct. On banked_CREDITS → order 6
(run_stats → QUALIFYING eval → CREDITS/WATCHABILITY-GAPS/HALT line 1 of NIGHT_REPORT.md). Healthy → glance cheap,
exit. Canonical UNTOUCHED. Pop-in: `play_live --resume --free-roam`.

**SHIFT-5 STAMP (15:56) — HEALTHY, BADGE 4 (Erika) BEATEN, glance-clean 0-commit.** Since shift 4's relaunch
past the two Lavender→Celadon wedges, fresh_go_3 **cleared Celadon Gym → badge 4 (Rainbow)** and is now
bench-grinding toward **Koga (gym 5)**, marching Fuchsia-way (Route 7/Route 25/26 band, map (3,25)). Party
**[venusaur L43, diglett L23, fearow L22, kadabra L22, raticate L22, growlithe L21] — SIX DISTINCT species,
ZERO dups, dex 11**. GRIND-WEAK/LOPSIDED firing textbook: RATIONALE *"Team's under-levelled — grinding
diglett, fearow, kadabra, raticate and growlithe up to ~L28 (fielding them, not my ace) so I can push
through, then on to Fuchsia City for Koga."* Ace venusaur crept L41→L43 (milestone-pinned, incidental — NOT
runaway); bench pinned to L28. **0 tracebacks whole-log (76.5K lines / 10.3MB)**, python 144491 + watchdog
bash 144483 alive, log LIVE (+48 lines/6s, 35 win-events per 200 lines = moving-not-spinning through the
grass grind + Route legs). All travel wedges = self-recovering LOUD ("returning to roam, no inner spin");
deep-wedge ring banking safe checkpoints across Celadon/Route 7 (normal map-hopping). No STALL/carry-forward
loop. banked_LIVE fresh 15:51. **ACUTE QUALIFYING WATCH (the run's central risk):** ace **L43** vs bench floor
**L21** = **~22-level gap** (slightly WIDER than shift 4's ~19 because the ace crept +2 while the bench held
during the Celadon push). GRIND-WEAK is now actively closing it (bench target L28, ace HELD). For QUALIFYING
at E4 entry EVERY member must be ≥L42, gap ≤15 — so the milestone ladder MUST pull the bench from ~L21 up to
~L55 across gyms 5-8 + Victory Road while the ace stays out of runaway. Plenty of milestone runway remains;
track the floor each glance. **NEXT SHIFT — glance:** (a) reaches **Fuchsia → beats Koga (gym 5)** → Sabrina/
Silph (gym 6) → Cinnabar/Blaine (gym 7) → Giovanni (gym 8) → Victory Road → E4 → Champion (proven credits
path, now with a distinct six); (b) **does the bench close the gap?** track the floor + gap each glance — if
at E4 entry bench <L42 or gap >15, the run fails qualifying on team-shape (the deferred order-3b crux is the
questline-guard already-shipped fix + one more relaunch); (c) six stay distinct through credits. On
banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_3.log RUN_STATS_fresh_go_3.md` →
evaluate QUALIFYING → CREDITS (PASS) / WATCHABILITY-GAPS / HALT (same root twice) as line 1 of NIGHT_REPORT.md.
Healthy → glance cheap, exit. Hard wedge → repro + ROOT-fix + resume from bank. Canonical `states/campaign/`
UNTOUCHED. Pop-in: `play_live --resume --free-roam`.
**SHIFT-4 STAMP (15:45) — TWO HARD WEDGES ROOT-KILLED, run relaunched PAST both.** On resume fresh_go_3
was in a **terminal 15-iteration stall** (iter 10→24, identical sig `(3,1,124,5,3,9152)@Lavender`, party
FROZEN [venusaur L39, diglett L22, fearow/kadabra/raticate L21], badge 3): the watchdog kept re-booting the
same wedged bank and re-stalling in ~54s. Diagnosed + fixed TWO stacked nav wedges on the Lavender→Celadon
(Erika/badge-4) approach, each verified live from the wedged bank:
  1. **Rock Tunnel south-mouth edge-warp (commit `c8fe10a`):** `_cross_warp_maze` found the south exit
     `(18,37)` reachable but `travel` landed her ONE TILE SHORT at `(18,36)` (border cave-mouth warp isn't a
     standable target) → "didn't fire despite reachability" → thrashed floors (1,81)↔(1,82) forever. Fix:
     when travel lands ADJACENT to the target exit warp, step directionally ONTO it (edge/mat warps fire on
     the step-onto). General for all border cave mouths. → she now crosses to Route 8.
  2. **Route 8→Celadon UGP pin (commit `dc1dc4a`):** past the tunnel she oscillated Route-8-east-edge↔Lavender.
     ROOT: TEAM-BRAIN PRE-BUILD *dominance* popped head_to_gym on Route 8 grass to force a squad-build, but the
     growlithe keeper stayed DUE so `_plan_wants_prebuild` never cleared → head_to_gym (the only action that
     crosses the Route 7-8 Underground Path) never returned. Fix: gate the head_to_gym pop on pc≤2 (truly-thin
     solo/duo); at pc 3-5 a functional lopsided team KEEPS head_to_gym and marches forward, catching the keeper
     IN PASSING. Plus an anchor-first `via:pass` hand-off (call `_door_passthrough` directly for the UGP hut).
  **VERIFIED LIVE:** crosses Route 8 → Underground Path (1,34→1,33) → Route 7 → **CELADON**, catches
  **growlithe** in passing → **full DISTINCT SIX** [venusaur L41, diglett L23, fearow L22, kadabra L22,
  raticate L22, growlithe L20], 0 dups, dex 11 → enters Erika's gym for badge 4. This is a QUALIFYING-shape
  six. Watchdog RELAUNCHED from banked_LIVE with all fixes; canonical UNTOUCHED. **NEXT SHIFT:** glance the
  climb — (a) beats Erika (badge 4) → Koga/Rock-Tunnel-already-done → the proven credits path; (b) the
  qualifying gap (ace L41 vs bench ~L22 = ~19) must CLOSE up the milestone pins to E4 (every member ≥L42,
  gap ≤15) — the questline-guard fix (5add821) + LOPSIDED-BENCH should pull the bench up on open ground; (c)
  six stay distinct. Hard wedge → repro + ROOT-fix + resume. Healthy → glance cheap, exit.

**SHIFT-3 LIVE STAMP (14:48):** HEALTHY, gap CLOSING — since shift 2 the bench climbed L15→L21 (+6) while
the ace crept L35→L39, so the ace-floor gap NARROWED **20→18** (bench outpacing ace = right direction). Now
navigating **ROCK TUNNEL** (maps 1,81/1,82) toward Celadon for **Erika (gym 4)**; `path N to exit` decrementing
20→10→2 = forward progress, NOT a wedge. Party=[**venusaur L39**, diglett L22, **fearow L21** (spearow evolved),
kadabra L21, **raticate L21** (rattata evolved)] — **5 DISTINCT species, 0 dups, dex 10**. GRIND-WEAK/LOPSIDED
firing 41×/800 lines (prep pin 27, ace HELD). banked_STALL@14:47 = SELF-RECOVERING watchdog abort (keeper_unreach
growlithe-fetch + FLAG_GOT_TEA questline no-progress → abandoned errand, re-recognized fresh, carried fwd from
banked_LIVE, progressing through tunnel; qualifying-safe). 0 tracebacks whole-log (45.8K lines), procs alive
(bash 143966 + py 143284/143286), log live 0s. NEXT: exit Rock Tunnel → Celadon/Erika (gym 4) → Koga… → proven
credits path. 6th distinct (growlithe→arcanine, Route 7/8) still uncaught. ————— PRIOR **SHIFT-2 (14:30):** advanced
badge 2→**badge 3**; on gym 4 grinding Route 4 (3,22). Party=[venusaur L35 (ivysaur evolved), spearow L16, rattata L15,
kadabra L15 (abra evolved), diglett L19] — 5 DISTINCT, 0 dups, dex 7 (box-aware de-dup holding; 6th = growlithe→arcanine planned
Route 7/8). **GRIND-WEAK firing textbook** (re-pin L21, *"fielding the weak ones — not my ace"*, rotating slots
2/3), ace **HELD L35**. Gap ace-to-floor = **20** (widened from shift-1's 15 — the Surge questline leg let the
ace lead; GRIND-WEAK now closing it on open ground). **0 tracebacks** whole-log (34.5K lines), procs alive
(venv py 39320 + emulator 44504 + watchdog bash 42708/40000), banked_GOAL 14:20 / banked_LIVE 14:27 fresh, no
banked_CREDITS. Coord oscillation (86-87,13-16) = intentional grass-grind, NOT a wedge (battles resolving, log
growing). The questline-guard fix (5add821) is working as designed on open ground. [shift-1 prior: badge 2,
Route 4, ivysaur L27, gap 15.] Boot's cosmetic `soul seed failed (NoneType)` at 0.0s (FRESH boot) is NOT a
blocker. The
questline-guard fix is confirmed closing the gap on open ground; keep watching it holds up the milestone pins.
**ACUTE QUALIFYING WATCH** (the whole point of the fix): at E4 entry every member ≥L42, ace-bench gap ≤15,
ace nowhere near L100 — this is what the fix must deliver; watch the gap CLOSES up the milestone pins, not opens.
**ON banked_CREDITS → run order 6:** `tools/run_stats.py G:/temp/longrun/fresh_go_3.log RUN_STATS_fresh_go_3.md`
→ evaluate QUALIFYING → CREDITS (PASS) / WATCHABILITY-GAPS (team-shape OK, pacing off) / HALT (same root twice)
as line 1 of NIGHT_REPORT.md. Healthy → glance cheap, exit. Hard wedge → capture repro, ROOT-fix, resume from bank.
Canonical `states/campaign/` UNTOUCHED. Pop-in: `play_live --resume --free-roam`.

## FACTS (verified this morning)
- fresh_go_1 ROLLED CREDITS last night (~20:30-21:00) from a cold bedroom start, fully autonomous: 8 badges, Victory Road, E4, Champion. 0 tracebacks, 0 hard wedges. The machine (nav rope, quest chain, battles, watchdog banking) is PROVEN end-to-end.
- BUT it won in the BANNED shape: venusaur solo-ground to L100 in VR, bench flat L40-47, DUPLICATE dugtrios, ~4h of silent grass grinding. Solo-ace+fodder = disqualified + unwatchable. Roots are known; they are today's build.
- Victory is NOT RECOGNIZED by the harness: champion flag set but no CREDITS outcome/banked_CREDITS; the watchdog mis-called the post-game victory lap a STALL and killed the run. Must fix.
- The watchdog compound fix lives ONLY in scratch (G:/temp/longrun/fresh_go_watchdog.sh). Landmine until ported.

## MISSION (one sentence)
Land ONE qualifying run: fresh cold bedroom -> credits, zero human touches, with an organically built, properly leveled, duplicate-free six and bounded watchable pacing - then emit stats, write the sentinel, stop the train.

## BUILD ORDER
0. PORT + LOOP HYGIENE: (a) port the scratch watchdog compound logic into the repo, commit. (b) patch night_shift.ps1: adaptive cadence (relaunch 60s after a commit shift; 15 MINUTES after a glance-clean 0-commit shift while the run is healthy); FAST-FAIL lines -> logs/fastfail.log, never NIGHT_REPORT.md; sentinel semantics: line 1 of NIGHT_REPORT.md starting with CREDITS stops the loop (exists) - ADD line 1 starting with HALT also stops it. Commit.
1. VICTORY RECOGNITION: champion flag / Hall of Fame -> OUTCOME: CREDITS + banked_CREDITS; post-game = victory-lap mode, never STALL. Commit.
2. STATS GENERATOR -> RUN_STATS_<run>.md: wall-clock start/end/duration; sim seconds; per-badge wall-clock splits; E4 attempts; whiteouts; every catch (species/where/when/kept-vs-boxed); evolutions; per-slot level curves over time; time-share battle/travel/grind/menus; longest continuous grind window; final party+levels; dex; money. RETRO-RUN IT ON fresh_go_1.log FIRST -> RUN_STATS_fresh_go_1.md (Jonny wants yesterday's numbers today). Commit.
3. TEAM-DEPTH (mission-central - why yesterday disqualified): (a) POKEMON_BENCH_TO_MILESTONE default ON. (b) KILL ace-hogs-XP - trainer/road/gym XP reaches the bench (participation share / lead rotation below milestone pins), not only slot 0. (c) extend milestone ladder to an E4 pin (whole six arrives Indigo >= ~L46; tune the number on evidence, the SHAPE is law). (d) de-dup: pokemon_planner._recompute_status scans PARTY not PC BOX -> make box-aware; never re-catch an owned species; prefer NEW species for gaps. (e) verify with ONE decisive cheap look-ahead (evened-kit style: bench climbs pins across >=2 gyms, no parking) - one look-ahead, then GO. No exploratory batteries.
4. LAUNCH fresh_go_2: cold bedroom start, detached (survives shift end), turbo, watchdog on, log G:/temp/longrun/fresh_go_2.log. Canonical saves UNTOUCHED. Single-run law.
5. WHILE COOKING: glance cheap, exit fast. Hard wedge -> capture repro, ROOT-fix, resume from bank. Healthy -> exit.
6. COMPLETION: credits roll -> evaluate QUALIFYING CRITERIA. PASS -> RUN_STATS_fresh_go_2.md + final survey <=20 lines (team-and-why: each member - where caught, why kept, what it answered) + write CREDITS as line 1 of NIGHT_REPORT.md. Train stops. Mission over. Credits but NOT qualifying -> one-page why -> fix root -> relaunch ONE more fresh run. SAME root fails twice at any stage -> HALT line 1 + one-page diagnosis. No qualifying run in flight by 22:00 -> HALT + diagnosis. The train never runs to infinity.

## QUALIFYING CRITERIA (all must hold)
- Fresh cold start; zero human touches; 0 crashes; 0 hard livelocks (self-recovering LOUD aborts OK).
- Final party: SIX DISTINCT species, no duplicates, all acquired this run.
- Levels at E4 entry: every member >= L42; highest-to-lowest gap <= 15; nothing near L100.
- Participation: >= 4 of 6 record wins in gyms/E4.
- Pacing: no continuous wild-grind window > ~20 sim-min without a logged milestone reason; total grind share <= ~35% of run. Bounded, purposeful, narrated is the law.
- RUN_STATS generated.

## DISCIPLINE
Fresh USD 250 credits now; weekly bucket refills 9am. Surveys <= 15 lines. No side quests. Organic HARD RULE absolute: no transplants, no struct-grinding, no pre-E4 solo grind. Escalate to Jonny ONLY via HALT; otherwise decide and act.

## ADDENDUM (pre-launch clarifications)
- STATS GENERATOR: parse fresh_go_1.log with a SCRIPT (commit it as tools/run_stats.py) - do NOT read the 34MB log into your own context. If a stat is not derivable from the log, write 'not instrumented' and add the missing logging for fresh_go_2 instead of guessing.
- ORDER 3e look-ahead: reuse an existing banked fixture (fuchsia_evened_kit / evened-kit style) - do NOT build new fixtures; that path is a known token sink.
- 22:00 rule, precise: a run IN FLIGHT and healthy at 22:00 CONTINUES (detached) - the deadline halts idle/blocked TRAINS, never a live healthy run. HALT semantics: write the one-page diagnosis INTO NIGHT_REPORT.md under the HALT line.
- If fresh_go_2 rolls credits but fails QUALIFYING on a pacing/participation metric only (team shape correct: six distinct, levels in band), do NOT relaunch - write CREDITS line 1 + stats + a WATCHABILITY GAPS section; Jonny adjudicates pacing personally.

## SHIFT-1 PROGRESS (2026-07-13, updated live)
DONE + COMMITTED:
- Order 0a: watchdog PORTED to repo `pokemon_agent/fresh_go_watchdog.sh` (was scratch-only landmine). Same compound carry-forward-banked_GOAL logic; LOG param (default fresh_go_2.log); stops on banked_CREDITS.
- Order 0b: `night_shift.ps1` — HALT line-1 sentinel stops loop; fast-fails -> `logs/fastfail.log` (not NIGHT_REPORT.md); adaptive cadence (60s after commit/fast-fail shift, 900s after 0-commit glance-clean).
- Order 1: VICTORY RECOGNITION in `recon_longrun.py` — reuses campaign's firewalled post_game (8 badges AND (game-clear 0x82C OR Hall/Champion room)); fires on boot->post_game TRANSITION; new CREDITS outcome -> banked_CREDITS (stops watchdog). Safety net after free_roam returns.
- Order 2: `tools/run_stats.py` (streaming, never loads 34MB log) + `RUN_STATS_fresh_go_1.md`. CONFIRMS disqual shape: ace Venusaur 6->100, bench flat ~L40 (60-lvl gap at badge 8), DUPLICATE Dugtrios, 83.6% travel/9.2% battle.
- Order 3a: `BENCH_TO_MILESTONE` default OFF->ON (campaign.py:147). Treadmill root killed NS#36 (verified). Decision-logic verifier `recon_bench_milestone_check.py` ALL 6 CASES PASS with the flip.
- Order 3d: box-aware de-dup (roster_judgment `also_owned` param + `_box_species_ids()` + catch_one force-catch flee-if-owned + planner `_recompute_status` unions boxed-species cache).
- Order 3c: milestone ladder ALREADY pins whole six to L55 at E4 (`_prep_e4_target`) — no edit needed.

- Order 3e VERIFY: decision-logic verifier `recon_bench_milestone_check.py` PASSES ALL 6 (re-pin +6 bites, park-proof release, retire-when-close). TWO behavioral look-aheads (surge_done_kit, koga_done_kit) were CONFOUNDED by pre-existing fixture NAV wedges (Diglett's-Cave entry geometry / Route-9 (41,6) zone-gap) before either reached the bench-grind — NOT my regression (zero errors from new code; wedges surface LOUD/no-inner-spin). Mechanism verified at logic level + error-free live; per mission "one look-ahead then GO" -> launched.
- Order 4 DONE: **fresh_go_2 LAUNCHED 06:32** — cold FRESH bedroom start, detached (nohup), turbo (SDL dummy), watchdog `fresh_go_watchdog.sh`, log `/g/temp/longrun/fresh_go_2.log`. Stale resume banks archived to `banked_*_archived_fresh_go_1_0631`. Canonical UNTOUCHED (workshop mode). Booted FRESH map(0,0) badges=0 into the proven opening spine.

FRONTIER NOW (shift 25, 10:08): **fresh_go_2 is at CINNABAR ISLAND (badge 6 done) working the Pokémon Mansion statue-puzzle for BLAINE (gym 7) — HEALTHY, and it just CRACKED THE HISTORICAL CREDITS-BLOCKER.** Since shift 24 (Koga prep, badge 4→): she cleared **Koga (gym 5) + Sabrina (gym 6) → badge 6**, evolved **diglett→DUGTRIO**, **CAUGHT LAPRAS**, and **SURFED THE WEST SEA to Cinnabar** — the exact Surf-mon sea-gate that wedged every prior organic fresh run (memory: `pokemon-freshgo-surf-mon-seagate-wedge`) is DEAD here because this run built a Surf-capable six. Log: *"WEST SEA @ (3,38) — surfing for Cinnabar → CINNABAR ISLAND after 19 battles — healing"*, now toggling Mansion statues (behind door (25,27), statue op (24,29)). Party **[venusaur L56, kadabra L33, dugtrio L27, fearow L26, gloom L26, lapras L25] — SIX DISTINCT species, ZERO dups, box-aware de-dup holding**. **0 tracebacks whole-log (63,342 lines / 6.7MB)**, watchdog bash 141524 alive since 08:14, python 142329 (current iter, relaunched 10:01 via clean carry-forward) alive, log mtime=now, log +4 lines/4s = moving-not-spinning through the Mansion puzzle. banked_GOAL fresh 08:27. **KEY QUALIFYING WATCH (the run's central risk, now ACUTE):** ace **venusaur is L56** while bench floor is **L25** — a **~31-level gap, WIDER than shift 24's ~22**. The sea crossing was a questline leg (19 battles, order-3b `_road_bench_xp_arm` guard disables bench participation-XP during questlines) → the **ace soloed the sea and ran away +10 levels (L46→L56)** while the bench barely moved. Venusaur is already OVER the E4 pin (L55). For QUALIFYING at E4 entry EVERY member must be ≥L42 with gap ≤15 — so gyms 7-8 + Victory Road milestone-grind MUST pull the bench from ~L25 up to ~L55 while the ace holds. **NEXT SHIFT — glance:** (a) Mansion Secret Key → **beat Blaine (gym 7)** → Viridian **Giovanni (gym 8)** → Victory Road → E4 → Champion (the proven fresh_go_1 credits path, now with a distinct six + Lapras);  (b) **does the bench close the gap?** — track the floor each glance; if at E4 entry bench <L42 or gap >15, the run FAILS qualifying on team-shape and the deferred **order-3b questline-guard relax** (campaign.py:7179 — let overworld-route/sea questline legs give bench participation-XP, NOT dark-cave gauntlets; RISKY in-cave switch livelock, validate isolated, DO NOT ship blind mid-run) is the crux fix + one more fresh relaunch;  (c) confirm the six stay distinct through credits. On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume from bank. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 25; nothing to fix, no flags flipped mid-run. PACING (Jonny adjudication only): 6 badges + past the sea-gate is EXCELLENT vs fresh_go_1's ~4h grind; the ace-runaway on questline legs is the qualifying/watchability item to adjudicate at credits. ————— PRIOR (shift 24, 09:50): **fresh_go_2 is on GYM 5 (Koga) prep, bench-grinding the full DISTINCT SIX — HEALTHY, LOPSIDED-BENCH guard now actively closing the gap.** Since shift 23 (badge 4 / Erika done): sim 3798s→**4687.3s** (+889 sim-sec), decision #147→#148, bench floor climbing **L21→L24→L25** (party **[venusaur L46, fearow L25, kadabra L28, gloom L25, beedrill L24, diglett L25] — SIX DISTINCT species, ZERO dups, dex 13**). **0 tracebacks whole-log (5.46MB)**, python 141813 under watchdog bash 141524 alive, log mtime=now, moving-not-spinning (grinding grass, coords (16,4)→(15,5), 51 win/target events in last 300 lines). **THE QUALIFYING MECHANISM IS FIRING:** `[roam] !! LOPSIDED-BENCH: bench severely behind milestone L45 + the ace (solo-carry shape) — forcing ONE dedicated grind stint` PRUNED the march and is rotating the weak slots as lead (slots 1/3/4/5), `restoring the ace` when it surfaces then re-fielding weak — the exact anti-fresh_go_1-disqualify guard, narrated/watchable (RATIONALE *"grinding fearow, gloom, beedrill and diglett up to ~L26 (fielding them, not my ace) so I can push through, then on to Fuchsia City for Koga"*). Ace venusaur crept L44→L46 (milestone-pinned, incidental — NOT runaway). Gap now ~L46 ace vs L24 floor (~22); the LOPSIDED-BENCH forced stint should keep closing it up the pins. Coverage plan verbalized: catch a **growlithe→arcanine** on Route 7/8 (fire answer for Champion). **NEXT SHIFT:** ————— PRIOR (shift 23, 09:33): **fresh_go_2 BEAT ERIKA (gym 4) + is on GYM 5 (Koga) grinding the bench with the full DISTINCT SIX — HEALTHY, best shape this run, past halfway.** Since shift 22: **cleared Celadon Gym → badge 4 (Rainbow)**, oddish→**GLOOM evolved**, and is now on Route 7 heading Fuchsia-way for Koga (gym 5). Party **[venusaur L44, gloom L23, kadabra L28, diglett L23, fearow L21, beedrill L23] — SIX DISTINCT species, ZERO dups, dex 13** (box-aware de-dup holding). **0 tracebacks whole-log (49,113 lines)**, python 141813 under watchdog bash 141524 alive, log mtime=now (sim 3798s/~63 min), banked_LIVE fresh 09:24, moving-not-spinning (coords varying on Route 7). GRIND-WEAK firing textbook (77 events): RATIONALE *"Team's under-levelled — grinding gloom, diglett, fearow and beedrill up to ~L26 (fielding them, not my ace)"*, currently fielding slot-4 fearow (L21) as lead. Ace venusaur HELD L44 (milestone-pinned, NOT climbing while roaming — no runaway). Coverage plan verbalized: catch a **growlithe→arcanine** on Route 7/8 (fire answer for Champion). **NEXT SHIFT:** glance the free_roam climb — (a) does she reach **Fuchsia → beat Koga (gym 5)** → Sabrina/Silph → onward toward the proven fresh_go_1 credits path?; (b) **KEY QUALIFYING WATCH (the run's central risk):** ace **venusaur L44 at badge 4** while bench floor is **L21** — a **~23-level gap**. The milestone ladder MUST pull the bench up across gyms 5-8 so at E4 EVERY member ≥L42 with gap ≤15 and the ace stays out of runaway (fresh_go_1 disqualified on ace→L100). GRIND-WEAK is pulling the bench (pin now ~L26 for Koga) with the ace HELD — the gap should CLOSE as bench climbs the pins; track each glance whether it does. If the ace over-climbs / bench stays thin at E4 entry, the deferred **order-3b `_road_bench_xp_arm` questline-guard relax** (campaign.py:7179) is the crux fix — RISKY (in-cave switch livelock), needs isolated validation, do NOT ship blind (see DEFERRED). (c) Confirm the six stay distinct at every glance through to credits. On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume from bank. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 23; nothing to fix, no flags flipped mid-run. PACING (Jonny adjudication only): 4 badges in ~63 sim-min is EXCELLENT pace vs fresh_go_1's ~4h grind — bench catch-up is participation-XP + bounded narrated grind windows (watchable), watch total grind share stays <=~35%. ————— PRIOR (shift 22, 09:16): **fresh_go_2 BEAT SURGE (gym 3) + is marching to Erika (gym 4) with the full DISTINCT SIX — HEALTHY, best shape this run.** Since shift 21: caught up the bench, **cleared Vermilion Gym → badge 3 (Thunder)**, and is now on Route 8 heading Celadon-way for Erika. Party **[venusaur L41, diglett L21, kadabra L27, fearow L20, beedrill L20, oddish L20] — SIX DISTINCT species, ZERO dups, dex 12** (box-aware de-dup holding). **0 tracebacks whole-log (4.3MB)**, python 141813 under watchdog bash 141524 alive, log mtime=now, moving-not-spinning (coords 71,9→61,14 fighting through Route-8 trainers; the `TRAVEL WEDGE x4` was self-recovering LOUD → re-pathed via a door candidate, then fighting a blocker-trainer — normal travel, NOT a wedge). Bench gaining participation-XP on the road (kadabra L26→27 mid-glance). Coverage plan verbalized: catch a **growlithe→arcanine** on Route 7/8 as the fire answer for Erika + Champion. **NEXT SHIFT:** glance the free_roam climb — (a) does she reach **Celadon → beat Erika (gym 4)**?; (b) **KEY QUALIFYING WATCH (the run's central risk):** ace **venusaur is L41 at badge 3** (currently HELD, not climbing while roaming — good) while bench floor is L20 — a **~21-level gap**. The milestone ladder MUST pull the bench up across gyms 4-8 so at E4 EVERY member ≥L42 with gap ≤15 and the ace stays out of runaway (fresh_go_1 disqualified on ace→L100). Track each glance whether the gap CLOSES (bench climbing via trainer participation + milestone pins) or the ace keeps running away; if the ace over-climbs / bench stays thin, the deferred **order-3b `_road_bench_xp_arm` questline-guard relax** (campaign.py:7179) is the crux fix — RISKY (in-cave switch livelock), needs isolated validation, do NOT ship blind (see DEFERRED). (c) Confirm the six stay distinct at every glance through to credits. On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume from bank. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 22; nothing to fix, no flags flipped mid-run. PACING (Jonny adjudication only): sim ~49 min at badge 3 heading to gym 4 is GOOD pace vs fresh_go_1's ~4h grind — bench catch-up is participation-XP on the road (watchable), watch total grind share stays <=~35%. ————— PRIOR (shift 21, 08:59): **fresh_go_2 HAS THE FULL DISTINCT SIX + is grinding the weak bench for Surge — HEALTHY, best shape this run.** Since shift 20: caught a **spearow** and evolved **abra→KADABRA**, so party is now **[venusaur L37, kadabra L17, beedrill L16, oddish L17, diglett L19, spearow L15] — SIX DISTINCT species, ZERO dups, box-aware de-dup holding**. She pushed to Vermilion, **blacked out at Surge (gym 3) under-levelled — 1 whiteout, EXPECTED**, and is now running the textbook team-depth recovery: narrated *"their electric lead has the type edge… the fix isn't trying harder, it's coming back stronger — leveling the weak ones (field THEM, not my ace) to ~L24, then re-cross"* — grinding the bench on Route 25 (map 3,29), coords moving, **winning battles, 0 tracebacks whole-log**, python 18784 alive, log mtime=now. Bench climbing (oddish L14→17, beedrill L15→16, kadabra L17). This is the BENCH_TO_MILESTONE build working exactly as designed, with a watchable narrated reason (a real Surge loss). **NEXT SHIFT:** glance — (a) does the bench floor cross the ~L24 pin → re-cross to Vermilion → **Cut the gym tree → beat Surge (gym 3)**?; (b) **KEY QUALIFYING WATCH (the run's central risk):** the ace **venusaur is already L37 at badge 2** while the bench is L14-19 — a **~20-level gap**. The milestone ladder MUST pull the bench up across gyms 3-8 so at E4 EVERY member ≥L42 with gap ≤15 and the ace stays out of runaway (fresh_go_1 disqualified on ace→L100). Track each glance whether the gap CLOSES or the ace keeps running away; if the ace over-climbs / bench stays thin, the deferred **order-3b `_road_bench_xp_arm` questline-guard relax** (campaign.py:7179 — let overworld-route questline legs give the bench participation-XP, NOT dark caves/gauntlets) is the crux fix — RISKY (in-cave switch livelock), needs isolated validation, do NOT ship blind (see DEFERRED). (c) Confirm the six stay distinct at every glance through to credits (run_stats logs every catch). On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume from bank. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 21; nothing to fix, no flags flipped mid-run. PACING (Jonny adjudication only): the Route-25 bench-grind to ~L24 is a bounded, narrated-purpose window (triggered by a real Surge whiteout) — watchable, but watch total grind share stays <=~35% and the ace doesn't keep creeping. ————— PRIOR (shift 20, 08:41): **fresh_go_2 CLEARED THE S.S. ANNE + GYM-3 PREREQ — gym 3 (Surge) is now one Cut-tree away.** free_roam (iter 3, badges=2) climbed Cerulean→Vermilion and, at gym 3, hit the S.S. Anne / HM-Cut gate and cleared the WHOLE thing autonomously: (a) prefight-grind evolved the ace **ivysaur L29→L32 → VENUSAUR** on Route 6 (the proven Gary-killer floor, `POKEMON_RIVAL_PREP_LEVEL=32`); (b) **boarded the S.S. Anne, beat the rival, got HM01 Cut from the captain, TAUGHT Cut** (`Field moves ready: CUT`, `KIRA obtained HM01`); (c) the HM-Cut questline step went **SATISFIED (ql=None)**; (d) disembarked → Vermilion, stocked up at the Mart. Now at #46 (sim 686.7s), RATIONALE textbook team-depth: *"Team's under-levelled — grinding abra and beedrill up to ~L14 (fielding them, not my ace)"* — the BENCH_TO_MILESTONE build fielding the WEAK ones. Party **[venusaur L32, abra L10, oddish L14, beedrill L12, diglett L19] — 5 DISTINCT species, 0 dups, dex 8** (box-aware de-dup holding). **0 tracebacks whole-log (27,207 lines)**, only 2 self-recovering travel-wedges in last 500 lines (LOUD, no inner spin), watchdog bash 141524 + python 141813 (iter 3) alive, log mtime=now. IMPORTANT — MY EARLIER-IN-SHIFT MISREAD (recorded so a successor doesn't repeat it): a `grep|tail -3` of decision headers caught a STALE #28 (ivysaur L29, Route-6) captured DURING the long evolve-grind action, which looked like a Vermilion↔Route6 ANCHOR-FIRST/BOARDING-GATE ping-pong wedge. It was NOT a wedge — the grind was mid-execution and then rocketed through the entire S.S. Anne sequence. LESSON: on a long single action the header freezes; verify against the LATEST `party=` snapshot (any layer) + newest headers, not just `tail -3` of decision headers. **NEXT SHIFT:** glance the free_roam climb — (a) does she Cut the Vermilion Gym tree → beat **Surge (gym 3)**?; (b) **KEY QUALIFYING WATCH** — the prefight solo-grind bumped the ace to L32 while bench is L10-19 (a ~22-lvl gap NOW); the milestone ladder MUST pull the bench up across gyms 3-8 so at E4 every member ≥L42 with gap ≤15 — track whether the bench actually closes the gap or the ace keeps running away (if ace over-climbs / bench stays thin, deferred order-3b `_road_bench_xp_arm` questline-guard relax is the crux fix, see DEFERRED); (c) 6th DISTINCT species still to be caught (abra→alakazam already in, planning coverage), no dups. On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 20; nothing to fix, no flags flipped. PACING (Jonny adjudication only): the ace evolve-grind to L32 was a bounded ~320-sim-sec window on Route 6 (proven-necessary for the ship rival) — acceptable, but it IS ace-focused XP; the qualifying-band bench-catch-up is the thing to watch. ————— PRIOR (shift 19, 08:22, SUPERSEDED by the above): **fresh_go_2 CLEARED THE ENTIRE OPENING — the badge-1 Mt-Moon wall is DEAD.** iter 1 (FRESH boot 08:14:11, map(0,0) badges=0) ran the scripted spine CLEAN through all 6 objectives in 430.5 sim-sec: WALK_TO_MAP→Viridian ✓, DELIVER_PARCEL ✓, GRIND_PRE_BROCK (solo starter→L13) ✓, ADVANCE_NORTH→Pewter→**Brock** ✓, Route3→Route4 (north conn) ✓, **CLEAR_MT_MOON** (mtmoon_plan.json) ✓, Route4→Cerulean ✓, **Misty→CASCADE = badge 2** ✓ → `all_segments_complete` @430.5s → **FRESH HANDOFF → free_roam** @badge 2, party=[ivysaur L22]. banked_LIVE STAGED @badges=2 (shift-17 gate working — only banks POST-Mt-Moon). NO Route-3 blackout even occurred this iter (spine ran clean; the shift-17 `_KNOCKBACK_CROSS_PRED` re-approach stands as an untriggered safety net). **0 tracebacks whole-log (3831 lines)**, python 141532 alive under watchdog bash 141524, log mtime=now. free_roam now building the six: oracle in Cerulean, `stock_up` at the Mart, teamplan `catch_keeper (abra→alakazam answers Erika, Koga, Sabrina, Bruno, Agatha, Champion)`, forward-drive → Vermilion for Surge (gym 3). Team-depth build (BENCH_TO_MILESTONE default-ON) + box-aware de-dup live. THIS IS THE BEST STATE THIS RUN HAS REACHED — the opening wedge that consumed shifts 15-18 is GONE; free_roam now carries the proven fresh_go_1 path toward credits but with the evened, distinct, de-duped six. **NEXT SHIFT:** glance the free_roam climb — (a) party growing to 6 DISTINCT species (abra→Route24/25 planned first), no dups (box-aware de-dup should hold); (b) bench climbing milestone pins as levels rise (ace ivysaur held, not runaway); (c) Surge(3)→Erika/Rock-Tunnel/Flash→Koga→... toward the proven credits path. On banked_CREDITS → order 6: `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` → evaluate QUALIFYING → CREDITS (or WATCHABILITY-GAPS) line 1 of NIGHT_REPORT.md, train stops. Hard wedge (whole-log spin, 0 progress many iters) → ROOT-fix + resume from bank. Healthy → glance cheap, exit fast. Glance-clean 0-commit shift 19; nothing to fix, no flags flipped. PACING (Jonny adjudication only): the badge-1 Route-2 over-grind of shifts 15-18 is GONE this iter (spine solo-grinds only to L13 pre-Brock then advances) — watch total grind share stays <=~35%. ————— PRIOR (shift 18, 08:18, SUPERSEDED by the above — was mid-opening-spine, now cleared): **fresh_go_2 FRESH boot HEALTHY & advancing the opening spine — shift-17 cold relaunch is climbing clean.** iter 1 booted FRESH 08:14:11 (map(0,0) badges=0, workshop, resume=False) into the scripted spine, and in ~4 min has already: OBJ 1/6 WALK_TO_MAP→Viridian ✓, OBJ 2/6 DELIVER_PARCEL (Pokédex + 5 balls) ✓, OBJ 3/6 GRIND_PRE_BROCK trained solo starter Lv5→**Lv13 (Brock-ready)** ✓, now OBJ 4/6 **ADVANCE_NORTH** (Viridian→Route 2→Forest→Pewter, at map(3,20)). Log LIVE (mtime=now, 2200+ lines, growing), python 141532 alive under single watchdog bash 141524, moving-not-spinning (coords advancing through the north band), **0 tracebacks whole-log**. This is the exact shift-5-proven ADVANCE_NORTH path (the `_advance_north_legs` live-edge-crossing fix guards the PEWTER no_path warp-wedge here). Glance-clean 0-commit shift; nothing to fix, no flags flipped mid-run. **NEXT SHIFT (the shift-17 fix is not yet exercised):** the run hasn't reached the Route-3 crossing yet (that comes AFTER Brock). Glance for: (a) Brock beaten → **Route-3 blackout SURVIVED** via the `WALK: knocked back ... re-approaching` line (shift-17's `_KNOCKBACK_CROSS_PRED` fix — THE thing to verify) → Route 4 → Mt Moon → Cerulean → **Misty = badge 2** → hands to free_roam PAST the Mt-Moon wall; (b) banked_LIVE stays ABSENT until badges>=2 (shift-17 gate); (c) past Cerulean, free_roam carries the proven fresh_go_1 path to credits with the evened-six team-depth build. If the FRESH spine crashes on a NEW opening blocker → capture repro, ROOT-fix (same knockback/gate self-recovery class), re-FRESH. If it clears the opening → the badge-1 Mt-Moon wall is DEAD and the run should climb like fresh_go_1 but watchably evened. [shift 17, 08:05]: **ROOT-FIXED the fresh_go_2 TERMINAL WEDGE + COLD-RELAUNCHED FRESH (commit 79f0e6e).** Shift 16's "marching to Cerulean" was a MIRAGE — head_to_gym was walking her BACKWARD (Viridian→Route1→Pallet), dead-ending at the Pallet split-map, STALLing, and the watchdog carry-forward re-booted badge-1 into free_roam which re-wedged identically (iter3→iter4, terminal loop, 0 real forward progress).
  THE ROOT (fully diagnosed, Mt-Moon-lesson-grade — DO NOT re-derive): **free_roam CANNOT cross Mt Moon** (only the scripted spine's `clear_mt_moon()` following `mtmoon_plan.json` can; there is NO billed Cerulean road, and the learned-graph edges for Route2/Pewter/Route3 are EMPTY — verified in the banked world_model). So once a badge-1 state lands in free_roam it is stranded PRE-Mt-Moon: head_to_gym has no KB road (Cerulean absent from `roads`), no graph route, base-camp=Pewter is unreachable in-graph, OPENING NORTH-MARCH is scoped Pewter-only → it falls to the generic SOUTH-discovery = BACKWARD to Pallet. Two failures combined to force free_roam-at-badge-1: (1) the FRESH spine crashed in the opening on a Route-3 blackout knockback (`route3_to_cerulean:WALK_TO_MAP:stuck` — respawn at Pewter Center, 2 maps back, 'north to Route 4' can't fire); (2) the watchdog staged banked_LIVE ~180s in at that badge-1 pre-Mt-Moon state, then resumed it into free_roam FOREVER.
  THE FIX (79f0e6e, both AST-clean, additive/failure-path-only): (1) `walk_to_map` blackout-knockback RE-APPROACH (`_KNOCKBACK_CROSS_PRED`: Route4←north FROM Route3) — graph-routes back to the crossing map, so the spine survives the Route-3 KO. (2) `recon_longrun` live-bank GATED on badges>=2 — banked_LIVE now only ever holds POST-Mt-Moon states, so a mid-opening crash re-runs FRESH from the bedroom instead of trapping free_roam. Mid/late game (always badges>=2) byte-unchanged.
  RELAUNCHED 08:0x COLD: killed the wedged watchdog, archived stale banks, cleared banked_LIVE/GOAL so iter 1 boots FRESH (scripted bedroom→Misty spine → hands to free_roam at Cerulean, past the wall). Detached, turbo, watchdog `fresh_go_watchdog.sh`, log `/g/temp/longrun/fresh_go_2.log`. Canonical UNTOUCHED.
  NEXT SHIFT: glance the FRESH run — does the spine now CLEAR the opening (bedroom→Brock→**Route-3 blackout survived→Route4→Mt Moon→Cerulean→Misty=badge 2**) and hand to free_roam? Watch specifically: (a) the `WALK: knocked back ... re-approaching` line fires if a Route-3 KO happens; (b) banked_LIVE does NOT appear until badges>=2; (c) past Cerulean, free_roam carries the proven fresh_go_1 path toward credits with the team-depth build (bench evened to milestones). If the spine crashes on a NEW opening blocker → capture repro, ROOT-fix (same class: a knockback/gate the spine can't self-recover), re-FRESH. If it clears the opening → the badge-1 wall is DEAD and the run should climb like fresh_go_1 but with a watchable evened six. [shift 16, 07:46 — SUPERSEDED, was a mirage: "marching to Cerulean" was actually the backward-to-Pallet wedge described above; bench WAS genuinely evened [23,20,20,20] 4-distinct/dex 9, that team-depth progress is real and reproduces fast on FRESH.] [shift 15, 07:43:] fresh_go_2 **ITER 3, HEALTHY, badge 1 (Boulder), free_roam, Route 2 Misty-prep bench-grind (map 3,20), re-pin L20, STILL ADVANCING** — sim #18 @2021.9s → decision @**2378.3s** (+356 sim-sec since shift 14), log **31,099→33,505 lines**, python 47004 alive, mtime=now, **0 tracebacks whole-log**. Bench climbed **[22,18,18,17]→[23,19,18,19]** (floor L17→L18, climbing to the L20 pin); ace HELD L22→L23 (incidental crept, **milestone-pinned NOT ace-pinned** — no runaway). GRIND-WEAK firing textbook (`fielding the weak ones (not the ace)`, rotating slots, `restoring the ace` when close then re-fielding weak). Party **[ivysaur L23, rattata L19, pidgeotto L18, beedrill L19] — 4 DISTINCT/0 dups/dex 8**. banked_LIVE 07:36. Glance-clean 0-commit shift; nothing to fix, no flags flipped mid-run. PACING NOTE (Jonny's adjudication only, unchanged): still badge-1 on Route-2 L3-5 grass ~35+ sim-min — a WATCHABILITY concern per mission addendum, NOT a fix/relaunch trigger (level curve monotonic, build doing its job). NEXT: floor crosses L20 → re-pin retires → march to Cerulean for Misty (gym 2). [shift 14, 07:39:] fresh_go_2 ITER 3 HEALTHY badge 1 Route 2 MID #18 re-pin L20; log grew 27.5K→**31,099 lines** since shift 13, mtime=now, python 47004 alive, tail = live battle wins (`[travel] battle outcome=win; resuming pathfind`, `reached target coord`) + deep-wedge ring banking safe checkpoints (ring=4/4, moving not spinning), **0 tracebacks whole-log**. Progress since shift 13: **pidgey → pidgeotto EVOLVED (organic, L18)**; bench climbing the L20 pin. Party **[ivysaur L22, beedrill L18, pidgeotto L18, rattata L17] — 4 DISTINCT/0 dups/dex 8**. banked_LIVE fresh 07:36 levels=[22,18,18,17]. #18 fields the weak slots sequentially to the L20 pin (a single roam decision spans many battles → same sim-stamp 2021.9s across the grind is EXPECTED, not a freeze — verified via live wins + growing log). Once floor crosses L20 → re-pin retires + march to Cerulean for Misty. Glance-clean 0-commit shift; nothing to fix, no flags flipped mid-run. PACING NOTE (unchanged, for Jonny's adjudication only): still badge-1 on Route-2 L3-5 grass — a WATCHABILITY concern per mission addendum, NOT a fix/relaunch trigger (build doing its job, level curve monotonic + an evolution this window). [shift 13, 07:36:] #17 @1301s → #18 @2021.9s (+720 sim-sec), bench [22,14,14,14]→[22,18,18,17], ring 52→64. [shift 12, 07:32:] decision #17 re-pin L20; log 19K→25.5K, ring 51→64. [shift 11:] (sim 1103s→1301s since shift 10; bench floor L14 CROSSED — rattata L10→L14, all bench now L14; ace HELD L22 milestone-pinned NOT ace-pinned; re-pin ~L20 for Misty). Party **[ivysaur L22, beedrill L14, rattata L14, pidgey L14] — 4 DISTINCT/0 dups/dex 7**. 0 tracebacks whole-log (banked_CRASH 07:02 = known self-recovered FRESH-spine crash, iter 3 clean since). python 47004 alive, banked_LIVE fresh (07:21). Team-depth build (BENCH_TO_MILESTONE default-ON) firing textbook — bench pulled up slot-by-slot, ace not runaway. Nothing to fix; glance-clean 0-commit shift. [prior shift-10 detail below] — log LIVE (mtime=now), python 47004 alive, winning battles. **MILESTONE CROSSED since shift 9: bench floor crossed L14** (`floor crossed L14 (levels [14,22,14,14]) — done`) → ladder **re-pinned to ~L20** for Misty prep, grinding resumed. Party **[ivysaur L22, beedrill L14, rattata L14, pidgey L14] — 4 DISTINCT species, ZERO dups, dex 7**. SHIFT-10 verified NO latent unbounded-grind bug: the NS#10 productivity gate (`_bench_poor_maps` park-proof, campaign.py:12022-12030) RETIRES the pin + marches to better grass if a Route-2 bite gives < BENCH_BITE_MIN levels → the L14→L20 re-pin is BOUNDED + self-correcting, not a treadmill. Team-depth build (BENCH_TO_MILESTONE default-ON) firing textbook: ace HELD L22 (milestone-pinned NOT ace-pinned), bench pulled up slot-by-slot 10→14 crossed→climbing to 20. Coverage-catch abra @ Route24/25 planned ("answer for Erika AND Koga"). 0 real tracebacks whole-log (19179 lines). PACING: ~20+ sim-min still badge-1 on Route-2 L3-5 grass (low kill-XP) — a WATCHABILITY concern for Jonny's adjudication (addendum), NOT a fix/relaunch trigger; the re-pin to L20 EXTENDS the badge-1 grass window (arguably over-grinding for Misty whose ace is L21) → flag for pacing adjudication, do NOT flip flags mid-run.  [prior shift-9 detail below] — sim ADVANCED 778s→~1103s→bench-L14-crossed→L20-repin, log live. Party **[ivysaur L22, rattata L10, beedrill L13, pidgey L14] — 4 DISTINCT species, ZERO dups, dex 7**. SHIFT-9 verified the grind is ADVANCING, NOT thrashing: clean monotonic level curve (checkpoints #5→#16) — Weedle CAUGHT (#6) → Kakuna → **Beedrill (evolved, #13)**; bench climbed 10→14 distributed slot-by-slot; ace correctly HELD (crept 19→22 incidental, milestone-pinned L14 NOT ace-pinned = no runaway). 0 real tracebacks whole-log (the lone `CRITICAL` match = a healthy SURVIVAL heal-narration, not a crash). rattata L10 is the last bench mon under the L14 floor → once it catches up she marches to Cerulean for Misty. PACING: 18 sim-min still badge-1 (Route-2 L3-5 grass = low kill-XP) — a WATCHABILITY concern for Jonny's adjudication (addendum), NOT a fix/relaunch trigger; 0-commit-fix is honest (verified build doing its job). This is the OLD sim-wall pre-shift-8 (~778s was the shift-8 glance number, cited below). Party **[ivysaur L22, beedrill L10→13→L14-bound, pidgey L10, rattata L10] — 4 DISTINCT species, ZERO dups, dex 7**. Shift-8 glance RE-CONFIRMED BENCH_TO_MILESTONE firing textbook: `GRIND-WEAK: team floor under L14 — fielding the weak ones (not the ace)` rotating slot-by-slot (beedrill L10→13, now pidgey), **ace HELD L22, badge-pinned milestone L14 (Misty prep), NOT ace-pinned** = no runaway. Coverage reasoning live: she's planning to catch abra @ Route 24/25 "my answer for Erika AND Koga." 0 tracebacks whole-log. THIS WATCHDOG (started 06:54:50) hit both known FRESH-spine crashes — ADVANCE_NORTH:stuck (iter1) + WALK_TO_MAP:stuck (iter2), BOTH self-recovered per design (carry-forward → iter3 booted banked_LIVE badges=1 into free_roam @07:02, healthy since). Frontier slow-but-healthy (Route-2 L3-5 grass = low XP/kill, so L10→14 is a LONG window) — a PACING concern = Jonny's adjudication per mission addendum, NOT a relaunch/fix trigger. 0-commit is honest (nothing to fix — verified build doing its job). GLANCE cheap, exit fast. On banked_CREDITS -> order 6 (run_stats -> RUN_STATS_fresh_go_2.md -> evaluate QUALIFYING -> CREDITS/WATCHABILITY-GAPS line 1 of NIGHT_REPORT.md). Hard wedge (whole-log spin, 0 progress many iters) -> ROOT-fix + resume. Healthy -> exit. If ace over-climbs / bench < L42 at E4 (qualifying fail) -> deferred order-3b `_road_bench_xp_arm` questline-guard relax is next build (see DEFERRED). WATCH the FINAL six are DISTINCT at credits. WATCH ALSO: grind-spot adequacy — if the Route-2 grind window balloons past ~20 sim-min, the NS#5 `_better_grind_spot` picker (flag-OFF) is the efficiency lever, but do NOT flip mid-run; note for pacing adjudication only.

SHIFT-5 ACTION (07:05, committed `2332671`): ROOT-FIXED a FRESH-spine crash. iter 1 crashed at OBJ 4/6 ADVANCE_NORTH — after the pre-Brock grind on Route 2, `travel(target=PEWTER)` returned no_path (PEWTER not yet in the FRESH world graph), so the enter_warp(prefer='north') fallback grabbed a BUILDING door and warped INTO it (5,4) -> 'no north edge' -> stuck. FIX: `_advance_north_legs` now, on travel no_path/stuck, reads the LIVE map header (`_map_connections`) for a real N edge and WALK-crosses it via `_edge_travel(north,'north')` BEFORE the gate-house warp (mirrors free_roam's proven OPENING NORTH-MARCH). Additive — only the crash path changes, happy 'arrived' leg byte-unchanged. AST-checked. iter 2 (old code, coord-lucky) had already CROSSED north + BEAT BROCK, so the wedge is coord-dependent; fix hardens it for every FRESH boot.
  SECOND wedge characterized (NOT fixed — self-recovers): after Brock, `route3_to_cerulean:WALK_TO_MAP:stuck` — ivysaur blacked out fighting through Route 3, respawned, then WALK_TO_MAP repeated "can't find a way through" 8× -> watchdog banked badge-1 + relaunched from bank into FREE_ROAM, which recovers the Pewter->Cerulean leg via the robust oracle/head_to_gym nav (iter 3 now past it). This is a SELF-RECOVERING LOUD abort (qualifying-criteria OK: "0 hard livelocks, self-recovering aborts OK"), different root from ADVANCE_NORTH (blackout-renav, not far-target no_path). Did NOT blind-fix the scripted spine (risk of regressing the credits-producing path). NEXT-SHIFT candidate if a FRESH qualifying run needs a cleaner spine: WALK_TO_MAP should, after a blackout respawn inside the Pewter Center, exit the building before re-pathing Route3->Route4->Cerulean (same "cross the live edge" class as the ADVANCE_NORTH fix — reuse `_map_connections`/`_edge_travel`).

SHIFT-4 ACTION (06:52): the shift-3 DUP-race WATCH item REPRODUCED PERSISTENTLY — iter 2's fetch_keeper caught TWO digletts into open party slots (party was 3, both landed slots 4-5, neither in the PC box) → party=[ivysaur,abra,weedle,diglett L19,diglett L16]. This is the exact fresh_go_1 duplicate-Dugtrio disqualify. ROOT: catch_one's force-catch de-dup checked `_box_species_ids()` ONLY; diglett #1 was PARTY-resident (not boxed) so #2 wasn't seen as owned. FIX (committed): new `_owned_species_ids()` = party ∪ box; target-catch de-dup now flees any species owned anywhere (campaign.py ~5393). The dirty 2-diglett run was cheap (2 badges, ~17min) so I KILLED it, archived its banks/log (`*_archived_fresh_go_2iter_0651`), and RELAUNCHED fresh_go_2 COLD with the fix. Canonical UNTOUCHED. Log reset to `/g/temp/longrun/fresh_go_2.log`. Old dirty log = `fresh_go_2_iter1-2_dirty_0651.log`.

LIVE CHECKPOINT (shift 3, 06:47 — pre-relaunch, HISTORICAL): fresh_go_2 HEALTHY & PROGRESSING WELL. Team-depth build CONFIRMED WORKING — the shift-2 critical check RESOLVED POSITIVE. 0 tracebacks whole-log (7690 lines).
  ITER 2 (live now, sim ~290s): party=[ivysaur L28, abra L10, weedle L7] — THREE DISTINCT species, NO dups; Pokédex=4; picked `fetch_keeper` (NOT solo-grind); GOT the S.S. Ticket from Bill (was inside Bill's house map(30,0)), backtracked Cerulean→Route 5→Route 6, now EN ROUTE to Vermilion City for gym 3 (Surge). Strategic keeper plan verbalized: "catch diglett → dugtrio, my answer for Surge AND Koga." Order-3 team-depth + coverage-reasoning TAKING beautifully.
  ITER 1 recovered cleanly (don't chase): it caught abra then had a ONE-OFF race — a second abra catch (dup) + a fetch_keeper↔head_to_gym tug (`questline_abandoned` NO MOVEMENT at (3,22)@(107,12)) → 14-decision STALL. Watchdog banked banked_STALL, carried forward from last clean banked_LIVE (pre-catch solo), relaunched. Iter 2 did NOT reproduce (caught abra+weedle, cleared Bill). So it was non-deterministic, not structural — watchdog machinery worked as designed.
  *** SHIFT-4 WATCH: (1) does iter 2 clear Surge (gym 3) and keep bench climbing the milestone pins as levels rise? (2) The iter-1 DUP-ABRA race is a latent disqualify risk if it ever produces a persistent duplicate in the FINAL six — run_stats logs every catch, so verify at credits the six are DISTINCT. If a dup survives to the party late-game (no stall to wipe it), ROOT-FIX the catch/plan-satisfaction race (planner not marking a species' keeper-plan satisfied the instant it's caught → allows a 2nd catch of same species within the same decision window). Not chasing now (non-reproducing). ***
  NOTE (not a bug — don't chase): every bank logs `sanctity FAILED — badge_count REGRESS 8->2` — CORRECT: validator compares staged bundle vs canonical states/campaign/ = fresh_go_1's completed 8-badge save; firewall correctly refuses to overwrite a completed canonical with an early fresh run. Live-banks still STAGE fine; final credits (badges 8) passes monotonic → recognition unaffected. Cosmetic noise only.

NOT DONE / DEFERRED (next shift if fresh_go_2 disqualifies on ace over-climb):
- Order 3b DEEP: `_road_bench_xp_arm` questline guard (campaign.py:7179) disables bench participation-XP during ALL questlines incl. Victory Road -> ace solos VR to L100. Relaxing it to allow overworld-route questline legs (NOT dark caves/gauntlet interiors) is the remaining crux fix, but RISKY (in-cave switch livelock) — needs isolated validation, do NOT ship blind.

## FRESH_GO_2 LAUNCH PROCEDURE (order 4 — the exact recipe)
1. Ensure COLD start: remove/rename `/g/temp/longrun/banked_LIVE` and any stale `banked_GOAL/STALL/TIMEOUT/CREDITS` so iter 1 boots FRESH (scripted bedroom spine).
2. Detached turbo launch (survives shift end): `cd pokemon_agent; SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy FRESH_GO_LOG=/g/temp/longrun/fresh_go_2.log nohup bash fresh_go_watchdog.sh &`
3. Canonical `states/campaign/` UNTOUCHED (recon monkeypatches _save to STAGE). Single-run law: exactly ONE watchdog at a time.
4. Watch for banked_CREDITS (stops watchdog) -> run `tools/run_stats.py G:/temp/longrun/fresh_go_2.log RUN_STATS_fresh_go_2.md` -> evaluate QUALIFYING CRITERIA.
