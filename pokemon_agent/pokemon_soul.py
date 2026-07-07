"""pokemon_soul.py - BATCH 2 SCAFFOLD: the hooks that let Kira's PERSONALITY flow into the Pokémon
engine. This is POKÉMON-MODE PLUMBING, not core-Kira: every reaction routes OUT through the existing
reaction seam (the `emit` callback == campaign.on_event == play_live's voice.emit), and the actual
DECISIONS are SEAMS that Batch 2 fills from her soul. We NEVER touch _pokemon_react / _build_self_block
/ voice / mood / bond / bridge.py - those are core-Kira, always-on, all-modes.

The four levers (decision points are BEATS):
  1. WANTS      - she holds wants (strategic + characterful), informed by game-knowledge, that surface
                  in reactions + can bias choices. The wants EMERGE (Batch 2); here is the structure.
  2. ROSTER-AS-FAMILY - teammates are family: she wants them, names them, fields them, holds opinions
                  that persist. Here is the bond store + the relational reaction hooks.
  3. MOVE-LEARN AS A BEAT - never auto-delete a good/super-effective move (SAFETY); make dropping a
                  move a deliberate, reasoned moment, not a mash-through.
  4. (felt stakes / strategic type-awareness ride on top of 1-3 + the existing battle engine.)

This hour lays HOOKS + SAFETY. The rich personality + the live watch come when the soul is wired in.
"""

# ── GAME KNOWLEDGE: what she KNOWS is possible (the substrate her wants can draw on). NOT a script of
# wants - a fact table her personality reads from. Extend freely. ──────────────────────────────────
GAME_KNOWLEDGE = {
    "eevee": "an Eevee lives in Celadon City - one teammate, eight possible futures",
    "fossil": "Mt Moon hides a fossil - Dome (Kabuto) or Helix (Omanyte), but only ONE",
    "starters": "Bulbasaur / Charmander / Squirtle - the choice that colors the whole run",
    "rare": {"scyther", "pinsir", "kangaskhan", "lapras", "snorlax", "dratini", "eevee", "hitmonlee",
             "hitmonchan", "porygon", "tauros", "chansey"},
    "legendaries": {"articuno", "zapdos", "moltres", "mewtwo"},
}

# Status / utility moves whose VALUE isn't captured by raw power - protect them from a naive
# "drop the lowest power" reserve (safety). Sleep/para/leech/sharp-stat moves earn their slot.
HIGH_VALUE_LOW_POWER = {
    79, 147, 95, 47, 142,        # Sleep Powder, Spore, Hypnosis, Sing, Lovely Kiss (sleep = catch+control)
    73, 77, 78, 86,              # Leech Seed, PoisonPowder, StunSpore, ThunderWave
    104, 97, 116, 14,            # Double Team, Agility, Focus Energy, Swords Dance
}


class PokemonSoul:
    """Holds Kira's evolving Pokémon-self (wants + roster bonds) and turns game beats into reactions
    through the `emit` seam. `emit(text, kind=, tier=)` is campaign.on_event (-> voice.emit). DECISION
    hooks return a default (safe/simple) but are the seam Batch 2 routes through her actual reasoning."""

    def __init__(self, emit=None, choose=None):
        self.emit = emit or (lambda *a, **k: None)
        self.choose = choose          # optional Batch-2 decision oracle: choose(kind, options, ctx)->pick
        self.wants = []               # active wants (strings); EMERGE from her, not hardcoded here
        self.bonds = {}               # nickname/species -> {species, nickname, caught, note} (family)

    # ── CONTINUITY: persist the Pokémon-SELF (roster bonds + wants) across SHOW sessions ──────────
    # The GAME state (roster/names/badges/location) is the savestate's job; THIS is the subjective
    # layer — how she FEELS about each teammate + what she wants — so a new stream resumes with her
    # relationships intact, not a blank slate. Scoped to the kira lineage by the caller (campaign).
    def save(self, path):
        """Write {bonds, wants} to `path` (JSON). Best-effort; never raises. Returns True on success."""
        import json
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"bonds": self.bonds, "wants": self.wants}, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"   [soul] continuity save failed: {e}", flush=True)
            return False

    def load(self, path):
        """Restore {bonds, wants} written by save(). Missing/corrupt -> blank (fresh run). Returns
        True iff continuity was loaded."""
        import json
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("bonds"), dict):
                self.bonds = data["bonds"]
            if isinstance(data.get("wants"), list):
                self.wants = data["wants"]
            print(f"   [soul] continuity loaded: {len(self.bonds)} roster bond(s), "
                  f"{len(self.wants)} want(s)", flush=True)
            return True
        except Exception as e:
            print(f"   [soul] continuity load failed: {e}", flush=True)
            return False

    # ── lever 1: WANTS ───────────────────────────────────────────────────────────────────────────
    def surface_want(self, context):
        """A beat where a want could surface (entering a town, seeing a rare foe, a roster gap). The
        WANT itself is hers (Batch 2 via self.choose); here we only carry the game-knowledge + emit.
        Capability-not-script: we never hardcode 'she wants Eevee' - we give her the knowledge + a
        moment, and let the soul decide if/what she wants."""
        if self.choose:
            want = self.choose("want", GAME_KNOWLEDGE, context)
            if want:
                self.wants.append(want)
                self.emit(want, kind="want", tier=2)

    # ── lever 2: ROSTER AS FAMILY ────────────────────────────────────────────────────────────────
    def note_caught(self, species, nickname, where=None):
        """A new teammate joins the family - record the bond + a relational reaction (not a stat line)."""
        key = (nickname or species or "").lower()
        self.bonds[key] = {"species": species, "nickname": nickname, "caught": where, "note": "new"}
        who = nickname if (nickname and nickname.lower() != (species or "").lower()) else species
        self.emit(f"{who} is part of the team now" + (f" - caught {where}" if where else ""),
                  kind="roster", tier=2)

    def note_faint(self, who):
        """A teammate goes down - a felt beat for family, not a neutral 'fainted'."""
        self.emit(f"{who} is down - I've got you, take a rest", kind="roster", tier=2)

    def note_evolve(self, before, after, who=None):
        # THE BOND FOLLOWS THE EVOLUTION (2026-07-06): meowth→persian must not orphan the family
        # entry — same friend, new form. Any bond whose species matches `before` updates in place
        # (nickname/key preserved; the relationship accretes, never resets).
        for _k, bnd in list((self.bonds or {}).items()):
            if isinstance(bnd, dict) and (bnd.get("species") or "").lower() == (before or "").lower():
                bnd["species"] = after
                bnd["note"] = f"evolved from {before}"
        self.emit(f"{who or before} evolved into {after}", kind="evolve", tier=3)

    def note_outcome(self, won, what=None):
        """Battle->mood SIGNAL (the inverse loop): a win lifts her, a blackout sours her. Mood itself
        lives in core-Kira and we NEVER set it — we only EMIT a tagged hint (kind=mood_up/mood_down)
        through the same seam, which the bot MAY let color her mood. Wiring + signal only; the mood
        math is core's, not ours (firewall)."""
        if won:
            self.emit(f"that felt good — {what}" if what else "that went well", kind="mood_up", tier=1)
        else:
            self.emit(f"that one stung — {what}" if what else "we went down", kind="mood_down", tier=1)

    def roster_opinion(self, who):
        return self.bonds.get((who or "").lower())

    # ── lever 3: MOVE-LEARN AS A BEAT (with SAFETY) ──────────────────────────────────────────────
    def move_drop_decision(self, moves_info, n):
        """Choose which `n` of the lead's current moves to DROP (so new ones auto-learn), as a
        deliberate beat. moves_info: list of (slot, move_id, name, power). SAFETY: never drop the
        single highest-power move, and never drop a HIGH_VALUE_LOW_POWER status move while plain
        filler exists. Returns (drop_slots, reasons). The choice is a SEAM (Batch 2 via self.choose);
        the default is the safe heuristic."""
        real = [m for m in moves_info if m[1]]
        if len(real) - n < 1:
            n = max(0, len(real) - 1)                              # always keep >= 1 move
        # rank DROP-ability: high-value status moves are least droppable; then by power ascending
        def droppability(m):
            slot, mid, name, power = m
            protected = 1 if mid in HIGH_VALUE_LOW_POWER else 0
            return (protected, power)                              # low protected + low power = drop first
        order = sorted(real, key=droppability)
        # never drop the single best attacker (highest power)
        best = max(real, key=lambda m: m[3]) if real else None
        droppable = [m for m in order if m is not best]
        if self.choose:
            picked = self.choose("move_drop", droppable, {"n": n, "all": real})
            if picked:
                droppable = picked
        drop = droppable[:n]
        reasons = []
        for slot, mid, name, power in drop:
            reasons.append(f"I never really use {name} - dropping it to make room")
            self.emit(f"I never really use {name} - I'll drop it for the new move", kind="move", tier=2)
        return [m[0] for m in drop], reasons
