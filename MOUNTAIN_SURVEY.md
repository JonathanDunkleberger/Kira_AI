# MOUNTAIN SURVEY — Sherpa reconnaissance, 2026-07-05 night

The master-Sherpa down-climb report. What's laid, what's ahead (data-billed from FRLG disasm/Bulbapedia
even where unbuilt), the walls, the rebuild recommendations, roster readiness, and the Sanctity Audit
(rule 17). The live durable state is `STATE_OF_PROJECT.md §0`; this is the strategic overview on top of it.

---

## 1. ROPE — furthest point reached
- **Furthest verified: the Nugget-Bridge GARY wall, just north of Cerulean.** Post-Misty save, party
  **Ivysaur L24 / Rattata L8 / Spearow L10**, 2 badges (Boulder, Cascade).
- **Verified end-to-end TO Gary this session:** shop (Cerulean Mart, potions+balls) → grind → forward-drive
  through the Nugget-Bridge trainers → reach Gary. Nav / shop / heal / questline all hold.
- **Gary: beatable but UNRELIABLE (~1-in-5).** Pidgeotto's Sand-Attack tanks her accuracy so powders +
  Razor Leaf miss. She won attempt #5 in a 24-min run (1W-4L) but never banked the S.S. Ticket (see wall #2).
- **Badges banked: 2.** **Checkpoints banked: the canonical `kira_campaign.state` (unchanged — protected).**
- **THE session's banked WIN: in-battle SWITCH cracked + committed** (the project's single biggest blocker —
  every gym/E4 fight and all bench-leveling depend on it). See STATE §0.

## 2. THE MOUNTAIN AHEAD — data-bill (disasm/Bulbapedia; ⛏ = built, ▢ = unbuilt)
Route order (canonical FRLG):
- **Gym 3 — Lt. Surge, Vermilion (Electric).** ▢ Reach: Cerulean → Route 5 → Underground Path → Route 6 →
  Vermilion (all plain edges/warps; questline engine ⛏ can route it). Gym gate: a **trash-can twin-switch
  puzzle** (two hidden switches; the 2nd is always cardinally adjacent to the 1st — a GENERAL solver reads
  the pair, doesn't hardcode). Surge: Raichu/Voltorb/Pikachu → counter Ground/Grass (Ivysaur fits).
- **HM01 CUT.** ▢ From the **S.S. Anne captain** (ship docked at Vermilion harbor; board → work through →
  Cut). Needed to leave Vermilion east + many town trees. Actuation gated `POKEMON_FIELD_MOVES=0` (unverified
  on the long core — apply the same DOWN-blind / readback pattern the switch now uses).
- **Flash HM05.** ▢ Route 2 north house aide (after ~10 species). Lights **Rock Tunnel**.
- **Rock Tunnel (dark cave).** ▢ Route 9 → Route 10 → Rock Tunnel → Lavender. Mt-Moon-class nav — apply
  warp-graph seeding + bend-discovery PROACTIVELY (not after wedge). Flash improves visibility.
- **Gym 4 — Erika, Celadon (Grass).** ▢ Via Lavender → Route 8 → Route 7 → Celadon. Erika:
  Victreebel/Tangela/Vileplume → counter **Fire/Flying/Ice/Psychic/Bug** (this is why a Fire or Flying
  ROSTER catch pays off — Spearow→Fearow already helps; a caught Growlithe/Vulpix would sweep her).
- **Rocket Hideout (Celadon) → Silph Scope; Pokémon Tower (Lavender) → Poké Flute (wake Snorlax).** ▢ Both
  are STORY_NPC/ITEM_GATE questlines the engine's pipeline handles (recognize→derive→execute).
- **Gym 5 — Koga, Fuchsia (Poison).** ▢ Via Cycling Road (needs **Bike**: Bike Voucher from the Vermilion
  Pokémon Fan Club chairman → Cerulean Bike Shop) or Routes 12-15. Counter Psychic/Ground.
- **HM03 Surf + HM04 Strength.** ▢ Fuchsia / Safari Zone (Warden's Gold Teeth → Strength; Secret House →
  Surf). Both are hard gates for Victory Road + Cinnabar.
- **Gym 6 — Sabrina, Saffron (Psychic).** ▢ Saffron is Rocket-locked until **Silph Co.** is cleared
  (Giovanni #2). Counter Bug/Ghost/Dark.
- **Gym 7 — Blaine, Cinnabar (Fire).** ▢ Surf to Cinnabar; Mansion has the gym key. Counter Water/Ground/Rock.
- **Gym 8 — Giovanni, Viridian (Ground).** ▢ Opens after Blaine. Counter Water/Grass/Ice.
- **Victory Road → Elite Four.** ▢ Needs Surf + Strength. E4: Lorelei (Ice/Water) → Bruno (Fighting/Rock) →
  Agatha (Ghost/Poison) → Lance (Dragon/Flying) → Champion Gary. No heals between the 5 — hard requirement on
  the SWITCH (⛏ now), items (⛏), and a real levelled 6-mon team (▢ — see roster).

## 3. WALLS — flagged, routed around (exact state in STATE §0)
1. **GRIND-STRANDING heal-wedge** (blocks bench-leveling / `GRIND_SWITCH`). Weak-grind routed a fragile mon
   into the far-east Route-4 below-ledge pocket (84,15); a faint stranded her (no Center reachable) → heal
   'stuck' → stall. **A blanket "Center-reachable grass only" fix was tried + REVERTED** (it regressed the
   ace-grind: ALL Route-4 grass is Center-unreachable, but the tanky ace uses it fine). Exact next fix in
   STATE §0: make it conditional (fragile-mon only) or route the weak-grind to Route 3. Then re-arm `GRIND_SWITCH`.
2. **BILL-LOOP** (blocks Vermilion). Post-Gary-win she reaches Route 24 but `head_to_gym` times out per tick
   crawling the bridge, and Gary losses (80%) eat the budget re-grinding. ROOT = Gary unreliability; reliable
   Gary (via bench-leveling) mostly dissolves it. Secondary: give the questline crossing a bigger per-tick
   travel budget.
3. **Gary reliability** — the keystone dependency of both above. Needs a fresh accurate attacker (leveled
   Spearow→Fearow via the now-working switch, or a Fire/Flying catch) to beat the Sand-Attack accuracy debuff.

## 4. WHAT WE CHANGE — master-Sherpa rebuild recommendations
- **HARDEN FIRST: the grind/heal spatial layer.** The recurring failure class all session was spatial
  (below-ledge pockets, per-tick travel timeouts, Center-reachability). This is the fragile crevasse the whole
  climb crosses repeatedly. Invest in: true ledge-awareness in `travel` (one-way `MB_JUMP` edges — a standing
  TODO), reachability-aware grind/catch routing, and a per-tick travel budget that scales to the crossing.
- **PROMOTE the readback/blind-nav pattern to a documented primitive.** Three menus now use it (move-list,
  bag, switch). Cut/Surf/Strength actuation should reuse it directly — don't re-derive per HM.
- **The switch's auto-triggers need tuning, not the mechanism.** Grind-switch (participation) and matchup-
  switch both fire correctly; the value is in WHEN they fire (don't ping-strand; fire vs the accuracy-debuff
  case, not just type-disadvantage).
- **OVERBUILT / can thin:** the 220 recon_*.py archive; the 6-layer STATE archaeology (now appendix'd).
- **Rebuild order when we three climb:** (1) spatial layer hardening → (2) re-arm GRIND_SWITCH + verify
  bench-leveling → (3) reliable Gary → (4) Bill/ticket → (5) then the pipeline REPEATS per gym (it's proven).

## 5. ROSTER + READINESS
- **Current: Ivysaur L24 (real carry) + Rattata L8 + Spearow L10 (dead weight).** This is a solo-ace team.
- **Honest E4 read: NO — not remotely.** A solo L24 grass starter with two sub-L10 benchwarmers loses at
  Koga, let alone the E4. She needs the roster the CEO wants: 4-6 chosen, leveled ~into the 30s-40s by E4,
  with type coverage (a Fire/Flying/Water/Psychic spread). The blocker is not catching (⛏ verified prior) —
  it's LEVELING the catches, which is gated on the grind-stranding fix + the now-working switch. **Unblocking
  bench-leveling is the single thing that turns her from a solo ace into a real trainer.**

## 6. SANCTITY AUDIT (rule 17) — what breaks TODAY if the Kira timeline started tomorrow
Ranked by severity:
- **[HIGH] Resume of the full narrative/continuity stack is UNVERIFIED end-to-end.** The `.state` round-trips
  (harness checks it), but a hard-kill + resume of world-model + strat + soul + journey continuity TOGETHER
  has not been kill-tested this session. Do the kill-test before any live run.
- **[HIGH] Spatial wedges are not all recoverable in-character.** The grind-strand hit a hard STALL, not a
  graceful recovery — on stream that's a frozen avatar. The escape-hatch/deep-wedge ring exists but did NOT
  recover the pocket-strand. Every wedge must degrade to an in-character recovery, never a stall.
- **[MED] Gary-loop watchability.** Even when mechanically fine, 896 battles / 9 decisions in 24 min is an
  unwatchable grind-loop. Reliability isn't just win-rate; it's not making the audience watch the same fight
  50×. This is a SHOW requirement, not just a mechanics one.
- **[MED] Field-move / HM actuation unverified on the long core** — a Cut/Surf/Strength failure mid-stream
  strands her at a gate. Verify each with the readback pattern before it's on the critical path.
- **[LOW] The uncommitted Gemini vision migration** (Jonny's core-side WIP) still sits in the working tree —
  a week+ old, touches core-Kira's eyes in all modes. Resolve (smoke-test + commit, or stash) before a run.

---
**Bottom line:** the DOOR (the in-battle switch) is cracked and banked — the biggest single unblock of the
whole project. Past it, every wall is the SAME spatial-layer fragility, and hardening that layer once turns
the proven per-gym pipeline loose on the rest of the mountain.
