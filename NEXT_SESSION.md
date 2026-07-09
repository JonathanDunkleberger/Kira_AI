# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 11 entry)

## SHIFT 11 HEAD (read FIRST — supersedes everything below)

**ROOT CAUSE of the shift 8-10 Route-5 wedge FOUND + FIXED (not the NPC-poison — a WRONG-BUILDING routing
bug).** Map `(17,1)` is NOT the Underground Path entrance — it's a DECOY DEAD-END house (Day Care is its
twin `(17,0)`). The PROVEN tunnel crossing is the Route 5 warp **`(31,31) -> (1,30)` → `(1,31)` [long
tunnel] → `(1,32)` → Route 6** — used by BOTH the canonical Cut-run (its `world_model.json`) AND
`crossing_fixA` (which only reached Route 6 AFTER escaping `(17,1)` north and taking `31,31`). The bug:
`_door_passthrough`'s `-len(dest_doors)` "likeliest-connector" heuristic ranks `(17,1)` FIRST because it
has TWO door tiles (24,32 + 25,32) vs the real UGP warp's one — so she enters the decoy, wedges on its
wandering old-man/counter, and the wedge aborts the passthrough BEFORE it ever cycles to `31,31`. The
whole 22-min `ughut_fix` run burned wedged in `(17,1)` via both its doors; she never once tried `31,31`.
The shift-10 NPC-stale-release fix (eb4a644) fires correctly but can't help — she should never be IN
`(17,1)` at all.

**FIX APPLIED + COMMITTED (shift 11, gamedata/frlg_gates.json):** added `"17,0"` + `"17,1"` to the
`no_connector` maps list. Both the passthrough (campaign.py ~L1477) and the GO-DEEPER building tour
(~L5396) skip any warp whose DEST is a no_connector map, so the two decoy houses are excluded and
`(31,31)->(1,30)` becomes the SOLE Route-5 crossing candidate → she takes the proven tunnel. Pure
game-knowledge-layer edit (a FireRed fact in `gamedata/`), portable-principle clean; the Route 6 reverse
hut `(19,13)->(1,32)` is a different map and unaffected.

**RE-RUN IN FLIGHT:** `python pokemon_agent/recon_longrun.py ss_ticket.state 22` →
`logs/debug/ugpfix2.log`. Convert UTF-16, then grep:
- `WARPED .* -> \(1, 30\)` or `31, 31` (took the PROVEN tunnel entrance — SHOULD appear; `(17, 1)` should NOT),
- `STATE IN: the Underground Path` / `STATE IN: Route 6` (crossed the tunnel — Route-5 wedge CLEARED),
- `STATE IN: .*Vermilion` / `S\.S\.` / `Anne` (reached the ship — back to the shift-9 S.S. Anne frontier),
- then the shift-9 ship-interior markers (`(1, 6)` 2F, `(1, 11)` captain, `HM01`, `rival`).

**IF the tunnel crosses → frontier returns to the S.S. Anne interior (shift-9 head below is live again):**
the exit-lobby guard (c2c4b80) should let her climb 1F→2F→captain for HM01 Cut. **IF she STILL enters
`(17,1)`:** the no_connector parse didn't take — check `_no_connector_maps()` / the inline `_nc` set both
include `(17,1)`. **IF she reaches `31,31` but can't traverse `(1,30)/(1,31)`:** that's a fresh in-tunnel
nav blocker (the tunnel is a long map, coord y~60) — instrument the multi-hop walk. **Past the ship →
Cut → Vermilion Gym tree → Lt. Surge (badge 3).** Thin bench (L14-16 vs L21-24 electric) may need a grind
or a Diglett (Diglett's Cave west of Vermilion) — separate under-level finding, NOT a nav bug.

---

## SHIFT 10 HEAD (superseded — the NPC-poison was a symptom; the real bug was wrong-building routing)

**SHIFT 9's exit-lobby guard (c2c4b80) is committed, but its re-run (`ssanne_fix.log`) NEVER REACHED
THE SHIP — it wedged much earlier, at the Underground Path Route-5 entrance HUT, map (17,1).** Root
cause (log-proven, NOT the exit-lobby guard): she warps into the hut at (4,2), needs the down-stairs
warp `(3,9)/(4,9)->(3,10)` (the tunnel), but a PATROLLING old man wanders row 5. The travel
micro-watchdog marked his transient tiles `(3,5)/(5,5)` into the PERMANENT `static_blocked`, sealing
both detour columns around the real row-6 counter `(3,6)/(4,6)`. She never leaves the map, so the seal
never clears → infinite `TRAVEL WEDGE` at (4,5). crossing_fixA only got through by LUCK (it exited+
re-entered the hut repeatedly, resetting the per-call set + reshuffling the NPC).

**FIX APPLIED + COMMITTED (shift 10, eb4a644, travel.py):** the micro-watchdog now feeds the SHARED,
stale-releasable `blocked_npcs` memory (like the LAYER-A chokepoint path) but WITHOUT `_fresh_marks`,
so the existing staleness-release un-marks the tile the instant it reads empty (wanderer stepped off) —
while a genuinely STATIONARY NPC (the Slowbro chokepoint) always reads occupied (`t in npc`) and stays
blocked. The npc-read guard discriminates wanderer vs squatter. General for every patrolled chokepoint.

**RE-RUN IN FLIGHT:** `python pokemon_agent/recon_longrun.py ss_ticket.state 22` →
`logs/debug/ughut_fix.log`. Convert UTF-16, then grep:
- `map=\(17, 1\)` + `releasing .* stale NPC block` (the fix un-marking the wanderer tile — SHOULD appear),
- `STATE IN: the Underground Path` (reached the tunnel — hut CLEARED),
- `STATE IN: .*Vermilion` / `S\.S\.` / `Anne` (reached the ship again — back to the shift-9 frontier),
- then the shift-9 ship-interior markers below (`(1, 6)` 2F, `(1, 11)` captain, `HM01`, `rival`).

**IF the hut clears → the frontier returns to the S.S. Anne interior (shift-9 head below is then live
again):** the exit-lobby guard should let her climb 1F→2F→captain for HM01 Cut. Read the ship markers.
**IF she still wedges in (17,1):** the wanderer read is unreliable — next fix = on a hut/building leg
whose target is a warp, after N wedges EXIT via a known entrance-warp and RE-ENTER (what crossing_fixA
did by luck), generalizing the poison-reset. **IF she gets past the ship → Cut → Vermilion Gym tree at
(3,5)@~(19,24) → Lt. Surge (badge 3).** Thin bench (L14-16 vs L30 ace) may need a grind or a Diglett
(Diglett's Cave west of Vermilion) — separate under-level finding, NOT a nav bug.

---

## SHIFT 9 HEAD (ship-interior detail — live again once the (17,1) hut clears)

**SHIFT 8's FIX A (a3900d6) IS VERIFIED FAR BEYOND ITS OWN CLAIM.** The `crossing_fixA.log` shows she
autonomously crosses **Cerulean → Route 5 → Underground Path → Route 6 → Vermilion**, boards the
**S.S. Anne**, tours the Vermilion Gym exterior, and catches **oddish + meowth** (party grew
ivysaur L30 / rattata L14 / spearow L15 / oddish L16 / meowth L10; dex 4→6). The whole badge-3 APPROACH
is autonomous. (Convert the UTF-16 log: `iconv -f UTF-16LE -t UTF-8 logs/debug/crossing_fixA.log`.)

**TRUE FRONTIER = INSIDE the S.S. Anne — she never reaches the captain (HM01 Cut).** She boards, wanders
1F cabins + the kitchen, retreats to the ship EXTERIOR, declares "wrong building", leaves, re-boards, and
repeats — **she never climbs the 1F→2F stairs**, so she never fights the rival / talks the captain / gets
Cut. Party stayed L29 3-mon the entire time on the ship. Then she drifts back to Vermilion and grinds.

**AUTHORITATIVE SHIP TOPOLOGY** (mined from `states/campaign/hm01_bank_20260706_203730/world_model.json`
— the canonical run that DID get Cut). Route to the captain:
- **(1,4)** S.S. Anne Exterior → `(32,14)/(33,15)` → **(1,5)** 1F ; and `(31,5)/(32,5)/(33,5)` → (3,5) Vermilion (the EXIT).
- **(1,5)** 1F → **stairs `(3,8)` → (1,6)** 2F ; `(28,17)`→(1,8) 1F-wing ; kitchen `(2,18)/(3,20)`→(1,10) ; cabins `(5,10)`→(1,12) … `(23,10)`→(1,17), `(20,10)`→(1,29).
- **(1,6)** 2F → **`(30,2)` → (1,11) = the CAPTAIN'S CABIN** (dead-end, warps only back to 2F) ; `(3,12)`→(1,7) deeper ; cabins (1,18)-(1,23).
- The RIVAL (Gary) fight fires on the 2F approach to the captain; her Ivysaur L30 should win.

**ROOT CAUSE (log-proven):** the GO-DEEPER tour (`campaign.py _questline_interact`, ~L5385) ranks
warps **farthest-first** among unvisited. The 2F stairs `(3,8)` sit at LOW-x, NEAR her 1F landing spots,
so they sort LAST; she tours the far cabins first, then picks a warp back to the **EXTERIOR** (a
visited-but-valid candidate that merely passed the old `dest[0]!=3` filter), lands on the exit lobby,
finds nothing deeper, and gives up — **before ever reaching the stairs.**

**FIX APPLIED (shift 9, campaign.py — UNCOMMITTED until the re-run verifies):**
1. **EXIT-LOBBY GUARD** in GO-DEEPER: exclude any warp whose destination map itself warps OUT to the
   overworld (group 3) — that map is the building's EXIT LOBBY, and touring INTO it is retreating, not
   descending. Now she can't leave the ship before exhausting the true interior (stairs included).
   Fail-open (unknown dest ≠ lobby → fresh-world discovery unchanged).
2. **DIAGNOSTICS:** logs the ranked GO-DEEPER candidate list each tick + a loud `LEAVING with UNVISITED
   deeper warps still skipped (reachability blocker?)` line if she exits with unvisited interior warps
   left — so if the NEXT blocker is feet-partition / an NPC on the stairs tile `(3,8)`, the log names it.

**RE-RUN IN FLIGHT:** `python pokemon_agent/recon_longrun.py ss_ticket.state 18`
→ `logs/debug/ssanne_fix.log`. Convert UTF-16, then grep:
- `GO-DEEPER cand` (the ranked tour — confirm `(3,8)->(1,6)` is a candidate),
- `DEEPER:.*-> (1, 6)` and `STATE IN: the S.S. Anne (1, 6)` (reached 2F?),
- `STATE IN: the S.S. Anne (1, 11)` (reached the captain's cabin?),
- `rival|Gary|FLAG_GOT_HM01|TEACH BRIDGE|questline_step_done` (got + taught Cut?),
- `LEAVING with UNVISITED` (still retreating? → reachability is the next blocker on `(3,8)`).

**IF she reaches (1,6) 2F but stalls entering `(3,8)`** (an NPC sits on/at the stairs — line 155 of
crossing_fixA.log listed an NPC object at (3,8)): next fix = in GO-DEEPER, before `enter_warp(wt)`, if the
warp tile is NPC-occupied, APPROACH+talk it (clears a guard / triggers the rival battle) then retry —
generalize the "an NPC guards the deeper warp" case. **IF she gets Cut:** next stretch = teach Cut (the
TEACH BRIDGE at ~L5079 auto-teaches) → Cut the Vermilion gym tree at (3,5)@~(19,24) → **Lt. Surge (badge
3, electric L21-24)**. Her bench is thin (L14-16 vs L30 ace) → Surge may need a real grind or a ground-type
(Diglett from Diglett's Cave, west of Vermilion) — that is a SEPARATE under-level finding, NOT a nav bug.

## KEY FACTS / TOOLS

- **venv python:** `G:/JonnyD/NeuroAI_Bot/.venv/Scripts/python.exe` (there is NO bare `python` on PATH).
- **Logs are UTF-16LE** (`*>` redirection): `iconv -f UTF-16LE -t UTF-8 <log> > /tmp/x.txt` before grep.
- **Look-ahead oracle** (default verification): `recon_longrun.py <state> <min>`; non-FRESH boot =
  free_roam from a named state, loads the CANONICAL world model (`C.WORLD_JSON`).
- **ss_ticket.state** (workshop) boots INSIDE Bill's house (30,0)@(7,7) with the ticket + badge 2;
  the run walks Bill → Cerulean → … → Vermilion → boards the ship (~5 min of the run). No ship state
  is banked yet — the recon has to re-walk the approach each iteration (unavoidable from ss_ticket).
- **hm01_bank_20260706_203730/** = the canonical bank RIGHT AFTER Cut — its `world_model.json` holds the
  full ship warp graph (used above). The `.state` there is post-Cut (not useful as a pre-Cut ship probe).
- **SINGLE-RUN LAW:** one emulator recon at a time; venv python = a 2-PID shim — never taskkill your own run.
- **states/campaign = SHERPA CANONICAL (untouchable). states/workshop = scratch.** Canonical is
  STAGE-redirected in the look-ahead. Commit per fix (`git add pokemon_agent/`); VERIFIED not asserted.

## GUARDRAILS (non-negotiable)

- **NAV / HARNESS FIXES ONLY.** Core Kira identity / voice / oracle / memory / vision are sacred +
  OFF-LIMITS. This shift's edit is pure mode-side questline-nav (`_questline_interact` GO-DEEPER) —
  additive, fresh-world path unchanged (fail-open lobby check).
- **AUDIO END STATE = ON** (`POKEMON_GAME_AUDIO=1`); audio-off is only the committed floor/fallback.
- Canonical Sherpa save (Champion) already rolled credits and is UNTOUCHED — never clobber it.

## STOP CONDITIONS

(a) clean bedroom→credits with audio ON demonstrated; OR (b) ~80-85% context → clean handoff (rule 11)
/ two-consecutive-no-progress brake; OR (c) balance exhausted.

---

WATCH STATUS: canonical Sherpa bank is CLEAN (Champion at home, untouched). The look-ahead now crosses
bedroom → Brock → Misty → Cerulean → Nugget Bridge → Bill → Vermilion → boards the S.S. Anne; the shift-9
fix (in flight) should let her climb to the captain for HM01 Cut. Pop-in (Sherpa) = `python pokemon_agent/watch.py`.

READY FOR THE TRAIN.
