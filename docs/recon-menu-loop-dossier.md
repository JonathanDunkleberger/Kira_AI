# RECON DOSSIER — The Menu-Loop Disease (Pokémon FireRed autonomous harness)

**Date:** 2026-08-03. **Prepared for:** external review by a second model, at the operator's request.
**Repo:** `Kira_AI` (local name "OG Kira"). **Runtime:** Windows PC, libmgba core driven by Python
(`pokemon_agent/`), streamed live via OBS. The agent ("Kira") has legitimately earned 4 badges and
previously beaten the entire game autonomously (`docs/RUN_STATS_fresh_go_6.md`).

**The complaint, in the operator's words:** the agent repeatedly gets stuck in *menu loops* —
in-battle (re-selecting a 0-PP move forever; applying the wrong medicine forever; opening the
party menu and re-picking the active Pokémon forever) and in the overworld (opening the bag and
trying to use a Super Potion on a full-HP team forever). ~15 fix iterations over 48h have narrowed
but not killed the class. This document is the full first-principles account: architecture, ground
truth we hold, ranked root causes with evidence, the fix ladder so far, and open questions.

---

## 1. Architecture (what presses the buttons)

- `pokemon_agent/play_live.py` — the streamed session process. Owns the emulator (`b`), pygame
  window, audio, voice. Runs `campaign.py` for free-roam decision-making. Battles encountered
  during travel are run by constructing `battle_agent.BattleAgent(b, ...).run()`.
- `pokemon_agent/campaign.py` (~18k lines) — free-roam loop ("ROAM TICK"): objectives, travel,
  healing decisions, watchdogs, checkpoint banking. Persists a campaign save
  (`campaign/kira_campaign.state`, a full emulator savestate + JSON bundle).
- `pokemon_agent/battle_agent.py` (~4k lines) — the in-battle engine. A turn loop:
  read battle state from RAM → pick move/item/switch → navigate menus → press → verify.
- `pokemon_agent/travel.py` — overworld pathing; engages blocking trainers ("fighting through").
- `pokemon_agent/dialogue_drive.py`, `dialogue_reader.py` — overworld dialogue advancement and
  in-RAM text decoding (`gStringVar3`-equivalent buffer at `DialogueReader.ACTIVE_MSG`).
- **Supervision:** a supervisor relaunches `play_live` on exit. Exit-code contract
  (`play_live.py` ~line 690): `0`=credits, `3`=window-closed → relaunch, `1`=crash/stuck →
  resume with crash-loop guard. A "dead-man's switch" posts to Discord and auto-resumes.
- **Deploys:** operator runs `pokemon_agent\resume_marathon.ps1` — kills python, `git pull`,
  relaunches, and commits a soak report (log tails + inventory) back to the repo.

### Input & screen sensing primitives (all in `battle_agent.py` unless noted)

| Primitive | Mechanism | Trust level |
|---|---|---|
| `_white_box()` | pixel check: white action-panel rows | good, has false positives (prize text) |
| `_at_action_menu()` | `MENU_MODE==1` (`0x02023E82`) + white box | good |
| `_at_move_list()` | `MENU_MODE==2`, else pixels | **misses while a refusal text box is up** |
| action cursor | `GBATTLE_ACTION_CURSOR` (`0x02023FF8`), RAM **write**+readback (`_poke_action_cursor`) | verified live |
| move cursor | `MOVE_CURSOR` (`0x02023FFC`), read + d-pad nav (`_goto_move`) | **SUSPECT — see RC1** |
| party cursor | `PARTY_CURSOR`; switch verified by active-species flip (`_switch_to_slot`) | verified live (species flip = ground truth) |
| battle state | `st.read_battle()` over `gBattleMons` | **tears: HP, status1, PP all observed stale/wrong live** |
| party truth | `gPlayerParty` (+0x50 status, +0x56/0x58 HP) | reliable (used as tear-guards) |
| battle text | `_battle_text()` decoding the active message buffer | works when it works; must never be load-bearing alone |

**Known-good axiom (hard-won, "pitfall 13"):** menus must be navigated by *cursor readback*,
never by state bytes alone, and a **blind A press is never safe** — the cursor may be parked
somewhere explosive (e.g. BAG → re-opens Super Potion).

---

## 2. Today's evidence chain (all in-repo, `docs/soak-reports/2026-08-03*`)

The same scenario replayed all morning: campaign save banked at Route 13 `(35,11)` at 08:48 with
**Blastoise L49, PARALYZED, 52/153 HP, and (at least) Tackle at 0 PP**, standing one screen away
from a mandatory trainer (fisherman, Spearow L25). Every session: resume → walk west → same
trainer → battle → loop → session ends (window-closed rc=3, or operator restart) → **resume
reloads the same pre-battle save** → repeat. Operator photos (09:04) show the move list with
cursor on TACKLE and the game text "There's no PP left for this move" — across four photos taken
over ~10 seconds **the cursor never moves off Tackle**.

Voice-line logs (the only battle telemetry that existed — see RC5) show, in every session:
`"no — Water Pulse didn't happen, I'm fully paralyzed!"` / `"no — Bite didn't happen…"` dozens of
times, interleaved, plus `"menus are glitched — bailing this fight."` — i.e. the engine
classified non-firing moves as paralysis immobilization, and its escape paths pressed A on the
move list.

Statistical impossibility that nails the misclassification: a *real* full-paralysis turn still
runs the turn — the foe attacks, HP changes, the verifier sees a resolution. Fifteen consecutive
"fully paralyzed" classifications with zero HP delta cannot be paralysis (0.25^15).

---

## 3. Ranked root causes (first principles)

### RC1 — The actuation lie: what we press is not what we think we press
The move-list navigation reads `MOVE_CURSOR` and taps d-pad until the byte matches the target
slot. The photo evidence shows the **drawn cursor never left Tackle** while the engine believed
it was trying Water Pulse and Bite. Two candidate mechanisms (unresolved which, likely both):
(a) the cursor byte reads a stale/shadow value so `_goto_move` returns "already there" without
moving; (b) d-pad taps get eaten by the long-running core (a documented failure class here —
"eaten press") and callers ignored `_goto_move`'s return value (`_struggle` did; `_select_and_verify`
fell back to `_nav_move` then pressed A regardless).
**Consequence:** every confirm fires the highlighted slot (Tackle), while bookkeeping records an
attempt on the intended slot.

### RC2 — Ledger poisoning: refusals attributed to the wrong slot
The countermeasure for stale PP bytes is a per-slot refusal ledger (`_move_refused`; ≥2 = slot
exiled, exiled slots count as dry for the PP-famine switch). Because of RC1, the refusals were
tallied on Bite/Water Pulse while **Tackle — the slot actually refused — stayed clean**, its
stale PP byte kept it "usable", the famine test never returned True, and the escape paths
("steer to the least-refused slot") steered **back to Tackle**. The rescue machinery was
correct and *armed*, and structurally unreachable.

### RC3 — Classifier interference: each guard resets another guard's evidence
- Refusal-vs-immobilization: a non-firing move on a paralyzed mon was classified as
  "fully paralyzed" (bounded ~6 laps) — but a refusal lap between two immob laps **reset the
  immobilization streak**, so the cap never engaged (log-confirmed: alternating WP-immob /
  Bite-refusal for minutes).
- The cosmetic `stall` counter resets on any screen change, so refusal-text flicker hides a
  wedge from it forever (known, previously fixed with `_unresolved_turns`).
- The menu-wedge escape was a one-shot latch; when its own escape press re-entered the loop,
  no escape could ever fire again (fixed to re-armable ×3).
**Pattern:** guards keyed on *classified* events can be starved by misclassification. Only a
guard keyed on "total zero-change laps, however classified" converges unconditionally.

### RC4 — The outer ring: a poisoned save replayed forever
The campaign save is banked at roam-start, *before* the fight, with a party state (paralysis +
0 PP) that makes the fight unwinnable-by-moves. Session death (rc=3 window-closed, dead-man
restart, or operator restart) reloads that save. There is **no cross-restart memory** of "this
exact battle has eaten N consecutive sessions" and no automatic divergence (heal first, ring
checkpoint, alternate route). The in-battle loop and the meta ring compound each other: any
in-battle fix that needs >~2 minutes to converge is killed and reset before it finishes.

### RC5 — Flying blind: the battle engine's logs were discarded
`play_live.py` constructed the battle agent with `log=lambda m: None` (and the campaign's
retreat paths likewise). **Every [engine] log line from every on-stream battle went to
/dev/null.** All diagnosis for weeks ran on voice lines, screenshots, and phone photos. The
engine has rich forensics ("GAME REFUSED slot", "STREAM COMMIT", famine decisions) that were
never captured. Additionally, `resume_marathon.ps1`'s `git pull` output was swallowed by a
PowerShell stderr quirk, so soak reports couldn't prove which commit a session ran — at least
three operator restarts today raced the fix pushes by 1–2 minutes and tested stale builds.

### RC6 — Data-source tears (the enabling substrate)
`gBattleMons` (the battle struct) has been caught lying three independent ways, live:
HP (full-HP ace read as hurt → potion loop), status1 (paralysis decoded as poison → Antidote
loop), PP (0-PP Tackle read as usable → this loop). Party-struct (`gPlayerParty`) reads have
never been caught lying and now back-stop HP and status ("tear guards"). PP has **no reliable
second source** (party-struct move/PP fields are in encrypted substructures), which is why the
refusal ledger + game-text detection exist.

---

## 4. The fix ladder shipped today (chronological, all on `main`)

| Commit | Fix | Root cause addressed |
|---|---|---|
| `f7395df`… (morning) | Cursor re-homing to FIGHT after any bag/party trip; screen-aware `_war_advance_press` (no blind A anywhere); re-armable menu wedge; dialogue-driver menu-text guard; watchdog stray-menu hole | the Super-Potion loop (bag-parked cursor + blind A) |
| `743e26a`/`f117ad2` | Status tear-guard (party struct is truth) + wrong-medicine abort | Antidote-on-paralyzed loop |
| `0f91b4b` | Refusal ledger (`_move_refused`, exile at 2); text-based instant exile; struggle-walk; back-at-move-list ⇒ refusal not immobilization; re-enabled battle switching (famine + must-leave) | stale-PP re-fire; paralysis mask |
| `c94a297` | Fast refusal detection (bail verify loop when parked back at move list) | 10s/refusal → ~2s |
| `673b486` | **Futility breaker**: battle-level count of fruitless move-list confirms (slot-agnostic, event-based); at ≥4, stop touching the move list, B out, bench-switch via species-flip-verified party nav; refusals attributed to the slot actually under the cursor at press time (+ loud CURSOR MISMATCH log); `_must_leave_active` honors the futility floor | RC1, RC2, RC3 |
| `cd9c2d5` | All immobilization classifications bounded at 6 per streak; resume script prints `running commit:` | RC3, RC5 |
| (this commit) | Immob laps also feed the futility counter; immob streak no longer reset by interleaved refusal laps; **battle engine logs un-muted** (`play_live`, campaign retreats); soak reports extract an `engine_tail.log` | RC3, RC5 |

### Why the futility breaker should be the one that holds (the convergence argument)
Every prior guard could be starved because it counted *classified* events (refusals on a slot,
immobs of a status) and the classifications were wrong. The futility counter counts **any
move-list confirm that produced zero change**, no matter which code path pressed it, which slot
got blamed, or how the lap was classified. It resets only on real progress (HP/PP change, item
consumed, switch confirmed). At 4 it abandons the move list entirely and performs the one action
whose success signal cannot be faked by move-RAM: a party switch verified by the **active
species changing**. Bounded ×2/battle; if the bench is dead too → Struggle / lose → whiteout →
Pokémon Center, which cures paralysis *and* refills all PP — the game's own built-in recovery
ratchet. Worst case is now a *lost battle*, never a frozen stream.

---

## 5. What is still open (honest list, for the reviewer)

1. **RC1's mechanism is unproven.** Is `MOVE_CURSOR` (`0x02023FFC`) the true
   `gMoveSelectionCursor`, or a shadow byte that goes stale in some battle UI states? The party
   cursor had exactly this bug (documented: "the old PARTY_CURSOR was a shadow byte"). A recon
   script that opens a battle, writes/steps the cursor, and diffs RAM vs rendered pixels would
   settle it. Until then the futility breaker contains the damage but move *selection* may
   still be unreliable (she may fire slot 0 when she wanted slot 2 in ordinary, non-dry fights
   — invisible today because any firing move counts as success).
2. **The outer ring (RC4) is unfixed.** Proposed: (a) on battle *resolution* (win/loss/flee),
   re-bank the campaign save so a restart never replays a finished fight; (b) a cross-restart
   counter keyed on (map, coords, foe species): ≥2 consecutive session deaths in the same
   battle ⇒ boot-time divergence — heal at the ring checkpoint first, or approach with a
   different lead. Both touch the sanctity-gated banking path, so they need care.
   **Interim manual override:** write a checkpoint directory name into
   `pokemon_agent/PROMOTE_TARGET.txt` — the harness teleports there on next boot.
3. **Why does the pygame window close mid-battle (rc=3)?** Twice today at ~140s uptime. A stray
   pygame QUIT during heavy menu churn is suspicious in itself (input-owner conflict? OBS
   capture hook?). Unexplained.
4. **PP has no reliable second source** — acceptable now (refusal ledger + text + futility),
   but a decrypted party-substructure reader would remove the whole "stale PP" class.
5. **`_battle_text` dependency reliability** on the PC (import path of `dialogue_reader` from
   within `battle_agent`) has not been verified in the live env — if it fails it degrades
   silently to timeouts (by design), which is slower but safe. The new engine logs will show it.

## 6. What the reviewer should sanity-check

- The convergence claim in §4: is there any code path that can press A on the move list without
  incrementing `_amove_futile` and without producing progress? (The verify-loop's inner "drain
  text" A-presses are the intended exception — they act inside one counted lap.)
- The futility threshold (4) vs. legitimate slow turns: a genuinely paralyzed mon whose foe
  spams status moves produces zero-HP-delta turns that classify as immob and now count toward
  futility → a bench switch after 4 such turns. We judge that acceptable play (switching a
  paralyzed mon is good), but it's a behavior change worth eyes.
- The species-flip verification in `_switch_to_slot` (battle_agent.py ~2945): is there any
  battle state (foe mid-faint, forced-switch prompt) where entering the party menu and
  confirming SHIFT could soft-lock rather than fail-safe B-out?
- RC4 proposals in §5.2 for save-banking hazards (the canonical save is sacred; all writes go
  through the checkpoint/banking path).

## 7. Key file/function index

- `pokemon_agent/battle_agent.py` — constants & RAM map (top ~130 lines); `_war_advance_press`
  + `_futility_bench_switch` (~1434); `_struggle` walk (~1298); `_goto_move` (~2100);
  `_select_and_verify` (turn commit + verify + classify, ~2130–2450); famine/must-leave/switch
  block in the turn loop (~3800–3900); anti-wedge floor + menu wedge (~4050–4120).
- `pokemon_agent/play_live.py` — battle construction (~482); exit-code contract (~684).
- `pokemon_agent/campaign.py` — watchdog scene gate; `_stray_menu_kind`/sweep; checkpoint
  banking; dead-man's switch (~18148).
- `pokemon_agent/dialogue_drive.py` — `_MENU_PROMPT_SNIPPETS` guard (B-close menus mistaken
  for dialogue).
- Evidence: `docs/soak-reports/20260803_*/` (esp. `085330`, `090134`, `091711`);
  future reports will include `engine_tail.log` and a `running commit:` line.
