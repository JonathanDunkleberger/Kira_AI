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
import json
import os

import pokemon_state as st
import firered_ram as ram

# gEnemyParty base (centralised in firered_ram, ✅ CONFIRMED 2026-06-27 via EWRAM brute-scan). Still
# re-checked live by _verify_enemy_roster() before any count is trusted (defence-in-depth); on a
# mismatch, party-size reads suppress + log LOUD (never a silent guess).
GENEMY_PARTY = ram.GENEMY_PARTY
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
    def observe_battle_start(self, b, place="an unfamiliar area", map_id=None,
                             coords=None, party_count=None, lead_level=None):
        """Snapshot the foe at battle start. PURE reads. `place` is the human place-name; `map_id` is
        the raw (group,num) so a wall becomes SPATIAL (Batch 4 Phase 2 — route around it). `party_count`
        + `lead_level` snapshot HER strength so a wall can later be judged "still gated" vs "worth a
        retry now that I've grown". Defensive: a flaky mid-intro read yields a thin snapshot (the loss
        still records on the verified active-foe species)."""
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
                "map_id": tuple(map_id) if map_id else None, "coords": coords,
                "my_party": party_count, "my_level": lead_level,
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
                # refresh the SPATIAL + strength snapshot to the latest loss (she may have grown a bit
                # and still lost — the gate should compare against the most recent attempt)
                rec["map_id"] = cur.get("map_id") or rec.get("map_id")
                rec["coords"] = cur.get("coords") or rec.get("coords")
                rec["my_party"] = cur.get("my_party") if cur.get("my_party") is not None else rec.get("my_party")
                rec["my_level"] = cur.get("my_level") if cur.get("my_level") is not None else rec.get("my_level")
            else:
                rec = {**cur, "count": 1}
                self.losses[key] = rec
            self.recent.append(key)
            self.recent = self.recent[-8:]
            self.active_wall = key
            self.log(f"   [strat] LOSS recorded vs {key} (now {rec['count']}x) at map={rec.get('map_id')} "
                     f"— SPATIAL wall active (gated until stronger)")
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
        her, how they're built) + a CONCRETE read on WHY she lost + the menu of options — framed as HER
        call, never a command. Batch-4 Phase 3: the rethink is STRONG after the FIRST loss to a
        meaningful wall (don't make the audience sit through loss #2); loss #2+ is harder still."""
        key = self.active_wall
        if not key or key not in self.losses:
            return ""
        r = self.losses[key]
        n = r["count"]
        # opponent reading (2C): describe how the foe is built so she can reason about counters herself
        if r["is_trainer"]:
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            lead_t = ("/".join(r["types"]) + "-type ") if r.get("types") else ""
            desc = f"that trainer (their lead was {r['lead']}, {lead_t}and they had {sz})"
        else:
            lead_t = ("/".join(r["types"]) + "-type ") if r.get("types") else ""
            desc = f"the wild {r['lead']} ({lead_t}around here)"
        where = f" at {r['place']}" if r.get("place") else ""
        # CONCRETE "why you lost" from the data she actually has — so the rethink is specific, not vague.
        reasons = []
        if r.get("my_party") == 1:
            reasons.append("you're running solo, so one bad matchup and there's nothing to switch to")
        if r.get("my_level") and r.get("lead_level") and r["my_level"] < r["lead_level"]:
            reasons.append(f"you were under-levelled (you ~L{r['my_level']} vs their L{r['lead_level']})")
        if r.get("is_trainer") and (r.get("size") or 1) >= 3:
            reasons.append(f"they out-number you {r['size']}-to-{r.get('my_party') or 1}")
        if r.get("types"):
            reasons.append(f"their {'/'.join(r['types'])} lead may have the type edge on you")
        why = "; and ".join(reasons) if reasons else "you were simply out-matched"
        meaningful = bool(r["is_trainer"] or (r.get("size") or 1) >= 2)
        if n >= WALL_REPEAT:
            return (f"Reality check: you've now lost to {desc}{where} {n} times. Same team, same result "
                    f"— this is a real wall and brute-forcing it isn't working. WHY: {why}. The move is "
                    f"to come back STRONGER, not angrier: level up, grab a teammate for backup/coverage, "
                    f"or line up a type that counters them. (Still your call — but unchanged just blacks "
                    f"you out again.)")
        if meaningful:
            # FIRST loss to a REAL wall — already a strong, concrete rethink (Phase 3: don't defer it).
            return (f"That one stung — {desc}{where} just put you down. Be honest about WHY before you "
                    f"charge back in: {why}. The fix here isn't trying again harder, it's coming back "
                    f"stronger — leveling up or grabbing a teammate for backup and type coverage is the "
                    f"real play. You CAN retry once if you're feeling it, but going in unchanged ends the "
                    f"same way. (Your call.)")
        # minor one-off (a single wild mon got a lucky KO) — keep it light, no big rethink.
        return (f"You just went down to {desc}{where} — bit of bad luck. Shake it off; maybe heal up or "
                f"pick your fights a little better. Your call.")

    # ── SPATIAL WALL (Batch 4 Phase 2): persist the wall as a place + judge "still gated vs grown" ──
    def active_wall_rec(self):
        key = self.active_wall
        return self.losses.get(key) if key else None

    def stronger_since_wall(self, party_count, lead_level):
        """Has she GROWN since the wall last beat her? — a teammate added OR a level gained. If so the
        gate is worth re-testing (she changed something). Unknown snapshot -> conservatively 'no' so the
        gate holds rather than feeding her back into a loss."""
        r = self.active_wall_rec()
        if not r:
            return True
        wp, wl = r.get("my_party"), r.get("my_level")
        if wp is None or wl is None or party_count is None or lead_level is None:
            return False
        return party_count > wp or lead_level > wl

    def is_gated(self, map_id, party_count, lead_level):
        """SPATIAL gate: is `map_id` the map where her active wall keeps beating her, AND she's no
        stronger than her last attempt? True = routing onto that map just re-walks into the wall."""
        r = self.active_wall_rec()
        if not r or not r.get("map_id") or map_id is None:
            return False
        if tuple(map_id) != tuple(r["map_id"]):
            return False
        return not self.stronger_since_wall(party_count, lead_level)

    def wall_gate_note(self, goal_desc="get there"):
        """The strong SPATIAL note when a goal routes back through the gated wall map: she's TOLD the
        way is blocked by a wall she can't beat yet, + the options — she still CHOOSES (capability-not-
        script). '' if there's no active spatial wall."""
        r = self.active_wall_rec()
        if not r or not r.get("map_id"):
            return ""
        place = r.get("place") or "that route"
        who = r["name"]
        if r["is_trainer"]:
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            who = f"that trainer ({r['lead']} lead, {sz})"
        return (f"Heads up: the way there runs back through {place}, where {who} keeps blacking you out "
                f"— that route is GATED until you're stronger. Don't just walk back into them. Level up "
                f"or grab a teammate on THIS side first, then come back and take that bridge. (Still your "
                f"call — but you can't pass them yet.)")

    # ── ADDENDUM D: PERSIST the loss/wall history (the FACTUAL basis of her grudge) ────────────────
    # Recon found StrategicMemory was IN-MEMORY ONLY — so 'Gary killed me 4x at the bridge' died on every
    # restart, and a --resume climb forgot who'd walled her. This makes that survive (sidecar JSON next to
    # the campaign save), so she resumes KNOWING where she got stuck and who beat her. This is the
    # game-mechanic FACT layer (who walls her where -> routing + the grudge's basis). The EMOTIONAL grudge
    # itself is core-Kira continuity (One-Kira firewall: she should carry grudges in ANY game) — that
    # seam (a meaningful loss writing into core memory) is decided + flagged for the batch-two build.
    def to_dict(self):
        return {"losses": self.losses, "recent": self.recent, "active_wall": self.active_wall}

    def from_dict(self, d):
        self.losses = d.get("losses") or {}
        self.recent = d.get("recent") or []
        self.active_wall = d.get("active_wall")

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, path)                     # atomic — never a half-written history
            return True
        except Exception as e:
            self.log(f"   [strat] !! loss-history save failed: {e} (LOUD)")
            return False

    def load(self, path):
        try:
            if not os.path.exists(path):
                self.log("   [strat] no loss-history sidecar yet — fresh strategic memory")
                return False
            with open(path, encoding="utf-8") as f:
                self.from_dict(json.load(f))
            wall = self.losses.get(self.active_wall) if self.active_wall else None
            self.log(f"   [strat] continuity loaded: {len(self.losses)} remembered wall(s)"
                     + (f"; active grudge vs {wall.get('name')} at {wall.get('place')} "
                        f"({wall.get('count')}x)" if wall else ""))
            return True
        except Exception as e:
            self.log(f"   [strat] !! loss-history load failed: {e} — starting fresh (LOUD)")
            return False

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
