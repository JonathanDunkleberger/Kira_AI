# STATE OF PROJECT — reality audit (2026-06-28)

The honest map of what's REAL vs a disconnected GHOST vs DEAD. Three-state per the operating rules
(`CLAUDE.md`): **COMPILES** (runs) · **WIRED** (data reaches the system meant to use it — esp. Kira's
DECISION/voice, with where) · **VERIFIED** (proven, or "needs live eyes"). The #1 column is **REACHES**:
BRAIN (decision/voice) vs DISPLAY-ONLY vs DEAD. "Exists but unwired" is the most important category —
that's the failure that's burned us (goals were built+displayed but never reached her brain).

Companion docs: `pokemon_agent/CODEBASE_AUDIT.md` (pokemon detail + stuck-vector list),
`pokemon_agent/FORWARD_CLIMB_STAGING.md` (gym 3→8 plan).

---

## 0. CURRENT FRONTIER — the rope as of 2026-06-28 night (read FIRST)

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

**NEXT:** confirm the grind sustains + levels a useful attacker (Spearow), then beat Gary → Bill → S.S.
Ticket → bank the first checkpoint → climb to gym 3 (Vermilion). The keystone (the hard part) is done.

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
