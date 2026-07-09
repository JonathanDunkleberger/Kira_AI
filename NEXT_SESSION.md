# NEXT_SESSION — NIGHT-TRAIN FRONTIER (2026-07-09, shift 9 entry)

## SHIFT 9 HEAD (read FIRST — supersedes everything below)

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
