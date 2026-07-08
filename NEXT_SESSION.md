# NEXT_SESSION — THE STANDING NIGHT-TRAIN MANDATE (rewritten 2026-07-08 ~00:00 after the quiet-window session; CEO: THE DESCENT restructure)

Paste this to the fresh session / this IS the night-train's standing order. Employment terms
in force: loop until done, bank per phase, honest three-state surveys, stop ONLY for a
needs-eyes item on the ledger below or ~85% context (then rewrite THIS file frontier-first
and hand off). Rule 16 covers everything else — decide and execute.

## BURN DISCIPLINE (CEO directive — density is mandatory)
1. **THE VALUE LINE.** Every shift survey ends: `this shift bought: [deliverable] —
   [shipped/verified/landed]`. Can't write a true one → STOP, surface needs-eyes.
2. **BOUNDED RECON.** Build-list in ≤1 shift, then BUILD.
3. **DEPTH OVER SPREAD.** 3 things fully > 10 things partially.
4. **NO IDLE-GRIND.** Blocked on Jonny → close early and cheap.

## ⛰️ THE DESCENT — HUMANITY-PASS DOCTRINE (CEO, after 3 watched soul-on runs; the showcase gate)
The Sherpa run proved the mountain climbs; the showcase requires it climbed like a HUMAN.
**HUMAN-PACE PLAY:** the rope (Sherpa knowledge) is her INSTINCT, not her GPS. In showcase/
soul mode she plays like a person — reads signs, talks to NPCs (mom!), notices rooms, fishes
if she wants, lingers, voices detours. **Speedrun behavior (beelines, skipped dialogue,
robotic efficiency) is a DEFECT in this mode even when mechanically optimal.** Endearing
beats efficient — the supreme law governs. The descent (F-11) is the quality gate in front
of the Kira-timeline GO; phases E/G/H/I/K remain behind it.

---

## FRONTIER (quiet-window session close, 2026-07-08 ~00:00 — Jonny watched 3 soul-on runs live)
**Commits this session: ffe16fd (rehearsal fix batch) + this file.** The GO-button throwaway
ran SIX takes (3 watched by Jonny → his notes = F-1..F-11 below; 3 headless verify).

**LANDED + VERIFIED live this session:**
- **Voiced starter choice** (the constitution's purest moment): `_choose_starter_soul()` →
  her soul picked **CHARMANDER 5-for-5 across independent asks** (a real preference), T3
  "your partner" beat fired to her voice (Jonny heard it, take 3). The old silent default-0
  is dead.
- **Judged nickname** (F-3 CEO principle): free-text soul ask before the pick — take 5:
  "keep the species name (her choice)". The AAAAAAAAA race (streaming-prompt answer landing
  late → A-drain typing on the keyboard) is fixed with settle-until-stable + verified-decline.
- **Scripted-intercept abort** (F-1's opening symptom): advance_north gains abort_when +
  leg_seconds; the 2-5 min lab wall-bumping (takes 1-2) → prompt lab arrival (take 3).
- **F-2 name-grounding** line in _spine_and_history (the "who told them my name?" tic).
- **F-8 harness half**: _muse_seed no longer types buildings as caves (INDOOR_MUSE; only
  map group 1 = dungeons keeps cave gloom) + Pallet interior names in _PLACE_NAMES ("at
  Oak's lab", not "an unfamiliar area").
- **C-1 READING register VERIFIED in production** (first-timer dialogue framing, 8+ fires);
  cold-open self-naming (KIRA), rival naming (GARY), Gary-loss canon + grudge arc
  ("0W-1L… this rivalry is just getting started"), post-loss want continuity ("Charmander
  starter") — ALL GOOD per Jonny (= F-4 protect list, add to Phase-H regression).

**✅ TAKE 7 — THE VOICED-CHOICE CHAIN IS FULLY GREEN (1d204b1):** soul chose CHARMANDER
(6-for-6 across takes — a real preference, fresh oracle POST every take, NO pin anywhere;
F-3 fossil-audited on CEO ask) → judged nickname (her deliberate pass, verified-gone) →
hands grabbed HER ball → `picked charmander (#4)`, no mismatch. **TEST DOCTRINE (F-3):
a test verifies choice==grab (the species-agnostic GRAB-VERIFY), NEVER a specific species —
if a future check greps for a particular starter name as its pass condition, that's the
fossil re-forming; kill it.** En route
takes 5-6 killed two latent bugs: greedy `_step_to` lab-wandering (BFS-first now) and a
RECON-ERA DATA BUG — the lab table is the RBY order B-SQUIRTLE-C, so ball 1's tile held
Squirtle (never caught before; no run had ever picked non-zero). GRAB-VERIFY now screams
on any species/choice mismatch. Take 6 also proved she can WIN the opening rival fight.

**✅ QW-4 SUMMIT WALK-OUT — PASS, with a NEW BUG at the tail** (`logs/quietwindow_summit.log`):
resume loaded the summit perfectly; her self-chosen want = **"catch Mewtwo"** (the love-
letter arc, unprompted); `leave_building` offered → SHE PICKED IT → EXIT (1,80)→(3,9)
Indigo Plateau — the summit strand is dead. She then traveled + battled (Sleep Powder
kit use). THEN: **THE VOID-CORE CLASS (new, watch-mode)** — after a battle+movelearn
sequence her position reads went inconsistent (impossible jumps: Cerulean→Lavender→Saffron
in one action) and the core ended at map (0,0), party 0, with the roam happily "playing"
the dead world (and SAVING it — to the SANDBOX only; canonical verified untouched,
last-write still 20:25). CONFOUND to exclude in the repro: a concurrent throwaway take was
killed mid-summit-run. DIAGNOSE with a clean single-run repro off banked_CREDITS.
Two more findings from the same log: (a) her want said "Lapras from Silph Co" WHILE LAPRAS
IS IN HER PARTY — wants aren't grounded against the roster (F-8 class, decision-side);
(b) the "!! DECISION LATENCY blocked the main thread" warning predates the frame-pump
(the world stays live now) — make the warning pump-aware or it cries wolf.

**NOT RUN this window (ready-to-run):** **prefetch A/B B-side** — bot RESTART with
`KIRA_TTS_PREFETCH=1` + one conversation with Jonny; compare [TurnTiming]/[LATENCY]
inter-sentence gaps vs the serial baseline he ear-calibrated tonight. Rides F-7(d).

## PHASE F — RESTRUCTURED INTO THE DESCENT (Jonny's notes, priority order)
- **F-1 LOCOMOTION (CRITICAL, still first — nothing grades honestly until her legs work).**
  Symptom fixed (intercept-abort); the ARCHITECTURE work remains: DOCTRINE — soul chooses
  destinations/intents and narrates; deterministic pathfinding executes ALL movement.
  Diagnose movement routing in soul mode end-to-end; wire intents→pathfinder; add the
  WANDER TRIPWIRE (no nav progress ~20s → harness takes the wheel to the current
  objective). Kills the token burn too (minutes of pointless oracle calls while wedged).
- **F-7 MOMENT ALIGNMENT (second — the seam viewers feel; reactions land 500ms-1s late).**
  A SYNC problem, not just latency: (a) per-stage timestamps on the full event→voice chain
  (extend the 3860ms measurement); (b) FRESHNESS WINDOWS — every reaction tagged with its
  triggering moment; would-miss → drop or reframe as explicit afterthought, never
  stale-as-live; (c) SPECULATIVE PREFETCH for predictable beats (dialogue boxes/battle
  events are known before they resolve — generate early, fire on trigger); (d) wire
  KIRA_TTS_PREFETCH default-ON if the A/B holds. Target: salient reaction perceived <2.5s
  AND aligned.
- **F-6 SOCIAL FABRIC:** NPC/object salience layer — key figures (family/rival/leaders/
  quest NPCs) + first-time key objects (the pokéball table!) exert pull on intents; "mom on
  route, day one" is near-mandatory; a deliberate skip must be VOICED. Salience feeds
  destination choice (F-1); pathfinder executes.
- **F-8 GROUNDED PERCEPTION (harness half landed):** remaining — every soul tick carries a
  location context block (map name, indoor/outdoor, region flavor from SEED_NODES); assert
  only what's grounded, ungrounded impressions as curiosity/questions; audit other
  commentary surfaces (music/weather/NPCs) for the confabulation path.
- **F-9 VERBAL TIC:** "doing a lot of heavy lifting" repeats + playful ban-evasion. Core
  fix → post-arc general recon (LOGGED). NOW: mode-side phrase-frequency governor so the
  showcase isn't wallpapered.
- **F-10 TEXTBOX DURABILITY:** 10-15s stall on Oak's FIRST textbox only — a first-encounter
  warm-up wedge class. Diagnose (cold TTS? first oracle round-trip? dialogue-reader init?)
  and fix — first impressions are where stalls can't live.
- **F-5 EXIT BAR (per-take):** bedroom→starter ~90s of travel, zero wall-grinding, voiced
  nickname call. **Jonny grades round 3 with the same notepad** once F-1 proper + the ball
  grab are green.
- **F-4 PROTECT (bank, don't touch):** self-naming, canon Gary loss, post-loss continuity,
  grudge arc, judged catches → Phase-H regression list.

## F-11 THE DESCENT ITSELF (the big preemptive pass — night-train grindable)
Using the watch rig's sandbox spawns, re-run EACH major arc of the completed rope SOUL-ON:
bedroom/Pallet → Viridian-Pewter → Mt Moon → Cerulean/Misty → Vermilion/ship → Rock Tunnel/
Lavender → Celadon/Rocket → Fuchsia/Safari → Seafoam/Cinnabar → Viridian-2 → Victory Road →
E4. GRADE each against the humanity doctrine + the wedge classes (F-1 locomotion, F-6
salience, F-7 alignment, F-8 grounding, F-9 repetition, F-10 warm-up stalls). Machine-
gradable checks run headless-with-soul; failures get fixed and re-run; each arc BANKS a
humanity-pass line in the report. Jonny spot-watches the machine-flagged riskiest arcs +
2-3 random ones — his notes are the final gate per arc. **EXIT: every arc passes machine
checks + Jonny has zero immersion-break notes on his spot-watches. THEN the GO button is
showcase-ready.** SEQUENCING: F-1 first, then F-7, then the descent proper.

## OTHER PHASES (unchanged, behind the descent gate)
- **E GO BUTTON:** built + rehearsed (go.py; 6 takes). Cold-open recap machinery untested on
  a RESUME (all takes were fresh runs) — verify when a show save exists.
- **G MAGIC AUDIT:** G-1 latency partially done (async voice, prefetch built; F-7 absorbs
  the rest), G-2 chat-advisors + G-3 cold-open SHIPPED flag-OFF (feel-tests pending), G-4
  cohost smoke = needs-eyes.
- **H REGRESSION GUARD:** standing law; F-4's protect list joins it.
- **I ONE-KIRA MODE MATRIX:** I-1 audit banked (MODE_TRANSITION_AUDIT.md, 6 seams); I-2
  attention director (extend SessionIntensity); I-3 media pacing (dead _PACING table
  bot.py:2752 gets a consumer).
- **K THE CLIPPER:** items 1/3/4 + selection-half landed; remaining: kira_moment logging
  (core touch), D3 paths, the 007 exit run → needs-eyes item 6.

## NEEDS-EYES LEDGER (the ONLY loop-stoppers)
1. **Round-3 throwaway grading** (F-5 bar) — after F-1 proper + ball grab green.
2. **Prefetch A/B B-side** — bot restart w/ flag + one conversation (2 min of Jonny).
3. **Descent spot-watches** (F-11 per-arc final gate).
4. **Cohost smoke session** (G-4 exit, 20 min).
5. **Tri-mode session** (Phase I exit, 15 min).
6. **Clipper output review** (K exit).
7. **Final showtime sign-off** — the Kira-timeline launch is HIS press, always.

## WORKING-TREE LAW
kira/* = Jonny's + approved core work under loud-log law. Prune targets (C-4 leftovers):
untracked recon_* fleet remainder, repo-root states/, verify_dialogue.py, NIGHT_REPORT.md.
Throwaway sandboxes: `go.py --clean-throwaways` + `watch.py --clean`.

---

WATCH STATUS: canonical bank is CLEAN (hall_of_fame — Champion, SACRED, untouched all
session; every rehearsal ran in disposable sandboxes). The soul-on watch rig + GO button are
live; the opening plays voiced end-to-end except the ball-grab (take-6 verdict in
logs/quietwindow_go_take6.log). Pop-in = `python pokemon_agent/watch.py`.
