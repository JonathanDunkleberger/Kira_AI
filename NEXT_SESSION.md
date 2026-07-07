# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #7 close)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 NIGHT SHIFT #7 blocks first (both). Last shift's arc
(commits c93f10b → cf24958):

🏆 **BADGE 5 BANKED — KOGA BEATEN.** koga_run3 reached Fuchsia and cleared all six gym juniors,
then Koga's L37 Koffing SELF-DESTRUCTED on Venusaur L54 turn one → full wipe. Five general
fixes, then koga_run8 took the badge in 54 seconds:
- **NUKE-SLEEP opener** (battle_agent `_NUKE_SPECIES` — Geodude/Voltorb/Koffing families):
  sleep the bomber BEFORE it detonates, at ANY damage matchup. Verified live: both Koffings
  slept, zero detonations.
- **PREP STAND-DOWN**: 2 straight dry grind attempts → the "train first" plan drops LOUD, the
  rematch/forward road wins. Resets ONLY on real grinding (an A↔B shuttle "arrives" every tick
  — that reset bug cost run6).
- **Grass fail-memory + grind-dead maps** (`_grass_unreach`, `_grind_dead`): one-way ledge
  pockets (Route 15 west end) and grassless "routes" (water Route 19) are vetoed as grind
  candidates in every `_grass_target` source including the last-resort branch.
- **Stale-attach disarms BOTH ways** (battle_agent): the save's display struct can hold the
  LAST fight's corpse at attach — foe-side (harmless) AND our-side (silent B-drain livelock,
  now disarmed by live read). The filed tower4 rival-miss is properly fixed: the engine keeps
  `LAST_FOES_SEEN` (live action-menu reads, can't be stale); campaign re-checks it post-battle
  against the rival counter-line. Gary #6 in Silph Co will be its live test.
- **ASYNC-WHITEOUT guard** (`_exit_to_overworld`): the respawn warp fires mid-candidate; check
  map-changed unconditionally per candidate (the Fuchsia Center ×6 wedge).

**CANONICAL = koga_badge5: Fuchsia (3,7)@(9,33), badges 5, Venusaur L55, $55k, sanctity VALID.**
(Party was hurt in-bank; every new run's first tick heals at the Fuchsia Center — verified ×3.
Check `ls pokemon_agent/states/campaign/pre_*` for anything sabrina_run3+ promoted after this
was written.)

🛣️ **BADGE-6 LEG OPEN — the Saffron road is billed and VERIFIED to carry her** (roads/"Saffron
City" in gamedata/frlg_gates.json, all directions from live-learned world-model edges):
Fuchsia→R15(e)→R14(e)→R13(n)→R12(n,**via pass**)→Lavender(w)→R8(w, via pass)→Saffron (3,11
expected). New exit_gates "3,26"/west = the thirsty-guard TEA gate (FLAG_GOT_TEA 0x2A6;
capabilities/FLAG_GOT_TEA = Celadon Mansion old lady, already billed).

⛔ **FIRST MOVE: fix the Route 12 northbound gate crossing (FULLY DIAGNOSED), then run the leg.**
sabrina_run3 wedged on it and was killed at close. The frontier is the **Route 12 NORTH
GATEHOUSE** (south door (14,21)→interior (23,0), out (14,15)/(15,15) on the Lavender side —
all live-learned warps on (3,30)). Route 12's top row is SEALED (no north edge — grid-dump
proven; the leg is billed via='pass' now, cf24958). The passthrough failure mechanism, exact
(campaign.py `_door_passthrough` ~line 1229):
1. Candidates sort multi-warp-first, so the GATE (14,21) IS tried first — but its ENTRY fails
   (likely the entry-method dispatch: recon_snorlax entered these gate doors southbound with
   prefer='south'; northbound needs entry walking NORTH into (14,21) — check what the entry
   step actually did in `logs/longrun/sabrina_run2.log`/`run3.log`), and it lands in the
   per-session `tried` set — one shot, gone.
2. The (12,86) REST HOUSE then "crosses" — a FALSE POSITIVE (single-warp house: she pops out
   beside her own entry, position technically changed) — and gets remembered as the map's
   proven connector (`self._pt_known[m0]`), which is retried FIRST every subsequent attempt,
   forever ("popped out beside the entry — not a crossing" ×∞).
FIXES (both needed): (a) a connector that pops out beside its entry must NOT set `_pt_known`
(and should join `tried`); (b) the gate entry — give (14,21) the directional-entry treatment
(_enter_directional_warp / prefer-north) or strike the crossing scripted (recon_snorlax.py
already walks this exact gate SOUTHBOUND — reverse it, ~20 lines, then bank at Lavender and
let the longrun continue). The STATIC grid path from the pocket to the gate is proven:
(12,87) → west lane x=8-10 → east along row 72 → north through the x=14 choke (rows 69-71,
the woken-Snorlax tile (14,70) is CLEAR) → gate. Relaunch after:
`LONGRUN_GOAL_FLAG=0x825 LONGRUN_BATTLE_LOG=1 POKEMON_SLEEP_LOCK=1 POKEMON_CATCH_JUDGMENT=1
POKEMON_PROACTIVE_BENCH=0 .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py
kira_campaign.state 90 > logs/longrun/sabrina_runN.log 2>&1`

THEN the leg continues: Lavender → Route 8 west → the gatehouse guard refuses → the TEA
questline arms (exit_gates 3,26/west) → route to Celadon via the Underground Path (door-
passthrough, proven both ways) → Mansion old lady → TEA → back → Saffron. KNOWN WALLS after
entry: **Sabrina's gym door is Rocket-BLOCKED until SILPH CO is cleared** (Card Key on 5F opens
doors, Giovanni #2 on 11F, GARY #6 mid-tower — recon_hideout.py/recon_tower.py are the strike
pattern; plan a `recon_silph.py` strike, don't wait for the questline to learn 11 floors);
no GYMS row for Sabrina yet (needs gym door/leader coords — disasm SaffronCity map.json when
she arrives); the gym interior is a TELEPORT-PAD maze (pads are warp events — read_warps sees
them; warp-dest routing like recon_tower's enter_to may just work).

KNOWN GAPS (owed): passthrough nearest-door vs multi-warp ordering (above); spin_nav not wired
into travel/campaign (Viridian Gym); questline can't CLIMB interiors generically (two strikes
prove the spec — promote the state-machine pattern when a third dungeon bites); Venusaur still
named "AAAAAAAAAA" (Name Rater in Lavender — cheap now, she passes through Lavender THIS leg;
consider folding it into the run as a soul beat); bench still dead weight (Ekans L15/Mankey L10
— Koga cost the ace two faints; E4 needs a real squad — Route 18 west of Fuchsia has grass if a
grind plan ever needs a real target near Fuchsia).

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire, arsenal,
single-run law, ground-truth-only (grid-dump/battle-trace before believing any wall),
NEXT_SESSION.md at close. GO.

---

## Morning survey pointers (for Jonny's 60-second read)
- **Promoted tonight (2):** fuchsia_reach (Routes 12-15 + all six Koga juniors cleared) →
  **koga_badge5** (BADGE 5, Venusaur L55). Both sanctity VALID.
- **The night's real story:** Koga wiped her once — his Koffing Self-Destruct-traded her ace
  turn one, the classic. The fix is exactly what a human does: sleep the bomber first. Rematch
  took 54 seconds. Around it: five general engine kills (stand-down for unexecutable plans,
  one-way-pocket memories, both stale-attach disarms, the async-whiteout guard) and the
  attach-time rival bug from shift 6 is properly dead (foes-seen ledger).
- **Human-hours advanced:** Routes 12-15 gauntlet + Fuchsia + the full gym (juniors + Koga,
  with one wipe-and-rematch) ≈ 2-3 hours of human gameplay; badge-6 road billed and marching.
- **Owed/honest:** Route 12 northbound gate crossing was the live frontier at close
  (sabrina_run3 in flight); Silph Co strike is the big rock between her and badge 6.

WATCH STATUS: canonical bank is CLEAN (heals on first tick); she is at Fuchsia with badge 5 —
or northbound on Route 12 if sabrina_run3 promoted — heading for Lavender/Saffron;
pop-in = `python pokemon_agent/play_live.py --resume --free-roam`.
