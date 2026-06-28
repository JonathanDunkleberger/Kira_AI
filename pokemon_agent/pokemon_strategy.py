"""pokemon_strategy.py — BATCH 3 PHASE 2: the STRATEGIC AWARENESS layer (mode plumbing, not core).

The live gap this fixes: she had a survival floor but NO strategic brain. She lost the Route-22
rival (Gary: Pidgeotto lead, 4 mons vs her solo Ivysaur — a known hard wall), whited out, and ran
straight back to die again with zero reflection. A 12hr die→run-back→die loop is unwatchable.

This module is a PURE MEMORY + AWARENESS-NOTE generator. It NEVER decides for her and it NEVER edits
core. It observes battles (who she fought, who beat her, how many times), and turns that into
plain-language AWARENESS that the free-roam loop folds into the oracle ctx via the SAME `place`-string
seam survival/shop/stuck already use (the one general field her oracle prompt renders — firewall: no
core edit). She is then AWARE of the wall and reasons about it from HER OWN knowledge (type charts,
roster-building) — but she still CHOOSES. Stepping back to build a team is on the table; so is a
stubborn solo-run. Capability-not-script: awareness FEEDS her, it does not DICTATE.

The three Phase-2 levers all live here:
  2A LOSS-LEARNING / ANTI-DEATH-LOOP — track losses (foe + count); after a repeat loss, surface
     "this approach isn't working — leveling / a teammate / a counter / come back later are options".
  2B ROSTER AWARENESS — notice "I'm rolling with one Pokémon, that's not a team" and let the want
     to build a roster surface (the oracle decides solo vs team).
  2C OPPONENT-READING — remember what a foe showed her (lead species + type, party size, a move that
     locked her turn) and feed it back so she can reason about counters from her own knowledge.

Source-first discipline: 2A rests only on ALREADY-VERIFIED reads (read_battle + the trainer-flag bit +
map id). The enemy-ROSTER-SIZE read (2C enrichment) is a CANDIDATE (gEnemyParty); it's gated by a live
self-check and degrades to "unknown size" + a LOUD log if it doesn't confirm — never a silent guess.
"""
import pokemon_state as st
import firered_ram as ram

# gEnemyParty base. ✅ CONFIRMED 2026-06-27 by an EWRAM brute-scan: this base decodes the FULL enemy
# roster correctly across four battle states (brock=geodude+onix, forest_trainer=weedle+caterpie, two
# wilds=1 mon) — matches the long-standing firered_ram.py recon note. The naive GPLAYER_PARTY+6*100 is
# WRONG here (that decoded nothing). Still re-checked live by _verify_enemy_roster() before any count is
# trusted (defence-in-depth); on a mismatch, party-size reads suppress + log LOUD (never a silent guess).
GENEMY_PARTY = 0x0202402C
_P_LEVEL_OFF = 0x54

# How many times she must lose the SAME fight before the note escalates from "that stung, regroup" to
# "brute-forcing clearly isn't working — change approach". Tunable; 2 = the spec's "same fight ≥2x".
WALL_REPEAT = 2


def _species_at(b, base):
    """Decrypt the species at an arbitrary 100-byte party-mon base (same algorithm as
    st.read_party_species, but parameterised so it works for gEnemyParty too). 0 = empty slot."""
    try:
        pid = b.rd32(base + 0)
        otid = b.rd32(base + 4)
        key = pid ^ otid
        order = st._SUBSTRUCT_ORDER[pid % 24]
        growth = base + 32 + order.index("G") * 12
        return (b.rd32(growth) ^ key) & 0xFFFF
    except Exception:
        return 0


class StrategicMemory:
    """Her road-memory: what she's fought, who's walled her, how her team stacks up. Emits NOTHING on
    its own (no on_event coupling) — it only RETURNS awareness strings the free-roam loop folds into the
    oracle ctx. Default-constructed and always present on the Campaign; headless-safe (pure reads)."""

    def __init__(self, log=print):
        self.log = log
        self._roster_ok = None          # tri-state: None=unchecked, True/False=gEnemyParty self-check result
        self.cur = None                 # opponent snapshot for the battle in progress (set at battle start)
        self.losses = {}                # foe_key -> {count, name, lead, types, size, place, is_trainer}
        self.recent = []                # chronological [foe_key] of recent losses (tail = most recent)
        self.active_wall = None         # the foe_key of the wall she's currently up against (cleared on a win)
        self._last_roster_n = None      # de-dupe the roster note so it isn't re-logged every tick

    # ── enemy-roster self-check (CANDIDATE gate) ─────────────────────────────────────────────────
    def _verify_enemy_roster(self, b):
        """Confirm GENEMY_PARTY really is the enemy party: slot-0's decrypted species must MATCH the
        active foe read the verified path already gives us (st.read_battle enemy species). One-time per
        process; on mismatch, party-size reads are disabled + logged LOUD (never a silent wrong count)."""
        if self._roster_ok is not None:
            return self._roster_ok
        ok = False
        try:
            rb = st.read_battle(b)
            active_sp = rb["enemy"]["species"] if rb else 0
            slot0 = _species_at(b, GENEMY_PARTY)
            ok = bool(active_sp) and slot0 == active_sp
            if ok:
                self.log(f"   [strat] gEnemyParty CONFIRMED (slot0={slot0} == active foe) — roster reads live")
            else:
                self.log(f"   [strat] !! gEnemyParty UNCONFIRMED (slot0={slot0} != active foe "
                         f"{active_sp}) — party-SIZE reads DISABLED (loss-learning still works on the "
                         f"verified active-foe read)")
        except Exception as e:
            self.log(f"   [strat] !! gEnemyParty self-check errored: {e} — party-SIZE reads DISABLED")
            ok = False
        self._roster_ok = ok
        return ok

    def _enemy_roster_size(self, b):
        """Number of mons in the enemy party (1..6), or None if the read isn't trusted. Only used as
        2C ENRICHMENT — the loss-learning core never depends on it."""
        if not self._verify_enemy_roster(b):
            return None, None
        names = []
        for s in range(6):
            sp = _species_at(b, GENEMY_PARTY + s * st.PARTY_MON_SIZE)
            if 1 <= sp <= 411:
                names.append(st.SPECIES_NAME.get(sp, f"species#{sp}"))
            else:
                break                                    # party is contiguous; first empty = end
        return (len(names) or None), (names or None)

    # ── battle observation (called by the Campaign's battle-runner wrapper) ──────────────────────
    def observe_battle_start(self, b, place="an unfamiliar area"):
        """Snapshot the foe at battle start. PURE reads. `place` is the human place-name (from
        read_live_state) so a wall is keyed to WHERE it is. Defensive: a flaky mid-intro read just
        yields a thin snapshot (the loss still records on the active-foe species)."""
        self.cur = None
        try:
            rb = st.read_battle(b)
            if not rb or not rb.get("enemy"):
                return
            e = rb["enemy"]
            is_trainer = bool(b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08)
            lead = st.SPECIES_NAME.get(e["species"], f"species#{e['species']}")
            types = [t for t in e.get("types", []) if t and t != "???"]
            size, roster = (self._enemy_roster_size(b) if is_trainer else (1, [lead]))
            kind = "trainer" if is_trainer else "wild"
            self.cur = {
                "key": f"{kind}:{place}:{lead}",
                "is_trainer": is_trainer,
                "name": ("a trainer" if is_trainer else f"a wild {lead}"),
                "lead": lead, "lead_level": e.get("level"), "types": types,
                "size": size, "roster": roster, "place": place,
            }
        except Exception as e:
            self.log(f"   [strat] observe_battle_start err {e}")
            self.cur = None

    def observe_battle_end(self, b, outcome):
        """Record the result of the battle just observed. `outcome` is the battle-runner's raw return
        ('loss' = party wiped; 'win'/'ended'/'fled' otherwise). A loss increments the foe's wall count;
        a WIN over a foe she'd been walled by CLEARS that wall (she broke through — no longer awareness
        she needs)."""
        cur = self.cur
        self.cur = None
        if cur is None:
            return
        key = cur["key"]
        lost = str(outcome).lower() in ("loss", "battle_loss", "blackout", "whiteout")
        if lost:
            rec = self.losses.get(key)
            if rec:
                rec["count"] += 1
            else:
                rec = {**cur, "count": 1}
                self.losses[key] = rec
            self.recent.append(key)
            self.recent = self.recent[-8:]
            self.active_wall = key
            self.log(f"   [strat] LOSS recorded vs {key} (now {rec['count']}x) — wall active")
        else:
            # a win (or flee) against the current wall clears it — she's past it / it isn't beating her
            if self.active_wall == key:
                self.log(f"   [strat] WIN/clear vs {key} — wall lifted")
                self.active_wall = None
            if key in self.losses and str(outcome).lower() in ("win",):
                self.losses.pop(key, None)

    # ── awareness notes (folded into the oracle ctx by free_roam, same `place` seam as survival) ──
    def loss_awareness(self):
        """The 2A + 2C note: if she's up against an active wall, hand the oracle the FACTS (who beat
        her, how they're built, how many times) + the menu of strategic options — framed as HER call,
        never a command. Returns '' when there's nothing to surface."""
        key = self.active_wall
        if not key or key not in self.losses:
            return ""
        r = self.losses[key]
        n = r["count"]
        # opponent reading (2C): describe how the foe is built so she can reason about counters herself
        who = r["name"]
        desc = who
        if r["is_trainer"]:
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            lead_t = ("/".join(r["types"]) + "-type ") if r.get("types") else ""
            desc = f"that trainer (their lead was {r['lead']}, {lead_t}and they had {sz})"
        else:
            lead_t = ("/".join(r["types"]) + "-type ") if r.get("types") else ""
            desc = f"the wild {r['lead']} ({lead_t}around here)"
        where = f" at {r['place']}" if r.get("place") else ""
        if n >= WALL_REPEAT:
            return (f"Reality check: you've now lost to {desc}{where} {n} times. Brute-forcing the same "
                    f"fight with the same team clearly isn't working — this is a real wall. Stepping "
                    f"back to level up, catch a teammate or two for backup, or line up a type that "
                    f"counters them are all on the table. Or try again if you've got a new idea — your "
                    f"call, but going in unchanged just blacks you out again.")
        return (f"You just lost to {desc}{where}. One loss isn't the end — but think about WHY before "
                f"charging back in: are you under-levelled, out-numbered, or type-disadvantaged? "
                f"Leveling up, grabbing a teammate, or a better matchup are options. Your call.")

    def roster_awareness(self, party):
        """The 2B note: notice the team's shape. A solo run is a valid CHOICE, not a bug — so this only
        OBSERVES the gap (and only when she's thin), letting the want to build a team surface. `party`
        is read_live_state()['party'] (list of {species, level}). Returns '' for a healthy-sized team."""
        n = len(party or [])
        if n == 0:
            return ""
        if n != self._last_roster_n:
            self._last_roster_n = n
            self.log(f"   [strat] roster size = {n}")
        if n == 1:
            who = party[0]["species"]
            return (f"Roster check: it's just {who} carrying the whole run solo — no backup, no type "
                    f"coverage, nothing to switch to if you hit a bad matchup. Catching a teammate or "
                    f"two would give you options. (A solo run is a real choice too — your call.)")
        if n == 2:
            return ("Roster check: two Pokémon is a thin bench — a third or fourth would give you more "
                    "answers to bad matchups. Worth a thought, if you want.")
        return ""
