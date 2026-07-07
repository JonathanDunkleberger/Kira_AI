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


# ── BLOCK #3 — ROSTER-SELECTION JUDGMENT (2026-07-06 nursery build; soul-debt #3's mechanical half) ──
# The framework a real player runs on every wild encounter: is it NEW, does it COVER a type gap, is it
# a decent LEVEL, is there ROOM? Pure function — it LEANS, with a first-person-ready REASON either way;
# the ORACLE decides live (capability-not-script), headless follows the lean. Never edits anything.
def roster_judgment(team, foe, dex_new=None):
    """team = [{'species_id', 'level', 'types': [...]}]; foe = {'species_id', 'name', 'level',
    'types': [...]}; dex_new = True when she has NEVER OWNED this species (DEX DOCTRINE 2026-07-06:
    a first-of-a-kind carries real positive weight when the catch is cheap — the dex is her ambient
    pride stat, never a pre-credits grind target). Returns (recommend_catch, reason, facts)."""
    name = foe.get("name") or "it"
    f_types = [t for t in (foe.get("types") or []) if t]
    team_ids = {m.get("species_id") for m in team}
    team_types = {t for m in team for t in (m.get("types") or []) if t}
    levels = [m.get("level") or 0 for m in team] or [1]
    floor, lead = min(levels), max(levels)
    new_types = [t for t in dict.fromkeys(f_types) if t not in team_types]
    coverage = [t for t in new_types if t != "normal"]
    facts = {"dupe": foe.get("species_id") in team_ids, "new_types": new_types,
             "coverage": coverage, "foe_level": foe.get("level"), "floor": floor,
             "lead": lead, "room": len(team) < 6, "dex_new": bool(dex_new)}
    lv = foe.get("level") or 0
    if not facts["room"]:
        return (False, f"my team's full — {name} would need someone to make way, and I like my six",
                facts)
    if facts["dupe"]:
        return (False, f"I've already got one of those — a twin {name} doesn't make the team stronger",
                facts)
    if coverage:
        tt = "/".join(coverage)
        if lv >= max(2, floor - 6):
            return (True, f"a {tt} type — I don't have ANY {tt} coverage, and L{lv} is workable. "
                          f"that's a real gap filled", facts)
        return (True, f"a {tt} type — I have zero {tt} coverage. it's only L{lv} so it needs raising, "
                      f"but the gap's worth it", facts)
    if dex_new and lv >= max(2, floor - 8):
        return (True, f"wait — I've never caught a {name} before. new species, ball's already in my "
                      f"bag… the dex grows today", facts)
    if len(team) < 4 and lv >= max(2, floor - 4):
        return (True, f"nothing new type-wise, but my bench is thin ({len(team)}) and a L{lv} {name} "
                      f"can pull weight", facts)
    if lv < max(2, floor - 6):
        return (False, f"L{lv}? way under my crew — it'd ride the bench forever", facts)
    return (False, f"{name} doesn't add anything my team doesn't already do", facts)


# How many times she must lose the SAME fight before the note escalates from "that stung, regroup" to
# "brute-forcing clearly isn't working — change approach". Tunable; 2 = the spec's "same fight ≥2x".
WALL_REPEAT = 2
# STRATEGIC-STUCK FLOOR (the Gary death-loop killer). After this many losses to the SAME wall with NO
# roster/level change between attempts, the loop is run-existential and the floor escalates hard: her
# APPROACH is wrong, not her execution. 2 = catch it on the 2nd identical loss, to prevent the 3rd.
STRATEGIC_STUCK_LOSSES = int(os.getenv("POKEMON_STRATEGIC_STUCK_LOSSES", "2"))
# READINESS → GO bar (the EXIT from strengthen-mode). stronger_since_wall() releases the PRUNE on ANY
# +1 level / +1 teammate (low bar: "stop blocking"). ready_to_retry() is the HIGHER bar that turns the
# return into an ACTIVE pull ("you've prepared — GO BACK now"): she's added a teammate OR gained this
# many levels since the wall last beat her. Two tiers on purpose — between them she's free to keep
# strengthening with no block AND no nag; crossing the high bar makes "go take the wall" the attractive
# move. Without this she un-blocks but never returns (grinds forever by inertia — the live symptom).
READY_RETRY_LEVELS = int(os.getenv("POKEMON_READY_RETRY_LEVELS", "2"))
# STRATEGIC UNDERLEVEL-GRIND readiness margin. When a wall keeps beating her because her team is
# under-levelled, the level her team FLOOR (weakest member) should reach is the foe's level she lost
# to + this margin (self-calibrating off the LIVE foe she actually fought — no hardcoded map KB). 1 =
# "get the weak ones up to roughly the trainers' level". Tunable.
UNDERLEVEL_MARGIN = int(os.getenv("POKEMON_UNDERLEVEL_MARGIN", "1"))
# ACE-OVERPOWER margin: when the bench can't be safely levelled (the participation-XP switch is gated/
# unverified), the reliable autonomous play vs a wall is to level the ACE (strong lead, solo-grinds fine)
# until it OUTLEVELS the wall enough to bulldoze it — bigger bulk to tank a super-effective hit + faster
# KOs so a long healing fight can't time out. Target = the wall foe's level + this margin (~one evolution's
# worth of levels above where she LOST). Keyed off her ace's level when it lost (not the foe's lead —
# her ace often already out-levels the foe's lead; the loss is matchup/bulk, and a few more levels of bulk
# let the ace tank the super-effective hit and finish the fight).
OVERPOWER_MARGIN = int(os.getenv("POKEMON_OVERPOWER_MARGIN", "5"))


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


def _foe_key(kind, place, lead, is_trainer, roster):
    """STABLE foe identity for loss-tracking. ROOT-CAUSE FIX (2026-06-28): the old key was
    f'{kind}:{place}:{lead}' where `lead` = the foe's ACTIVE mon at snapshot time — so the
    SAME trainer (Gary: pidgeotto/abra/rattata/charmander) recorded as THREE separate walls
    (…:pidgeotto / …:charmander / …:rattata), each count=1, and the strategic-stuck floor
    (needs count≥STRATEGIC_STUCK_LOSSES on ONE key) NEVER fired — the live strat_memory proved
    it. A trainer's identity is their full ROSTER, not whichever mon happened to be out, so key
    trainers on the sorted roster signature (stable across attempts). Wilds stay keyed on the
    species (that IS their identity). Degrades to the lead only if the roster read didn't
    confirm (the caller logs the unconfirmed-roster case)."""
    if is_trainer and roster:
        sig = "+".join(sorted(str(s) for s in roster))
        return f"trainer:{place}:{sig}"
    return f"{kind}:{place}:{lead}"


class StrategicMemory:
    """Her road-memory: what she's fought, who's walled her, how her team stacks up. Emits NOTHING on
    its own (no on_event coupling) — it only RETURNS awareness strings the free-roam loop folds into the
    oracle ctx. Default-constructed and always present on the Campaign; headless-safe (pure reads)."""

    def __init__(self, log=print):
        self.log = log
        self._roster_ok = None          # tri-state: None=unchecked, True/False=gEnemyParty self-check result
        self.cur = None                 # opponent snapshot for the battle in progress (set at battle start)
        self.last_foe = None            # persists past observe_battle_end: who she last fought (whiteout-backstop)
        self.losses = {}                # foe_key -> {count, name, lead, types, size, place, is_trainer}
        self.recent = []                # chronological [foe_key] of recent losses (tail = most recent)
        self.active_wall = None         # the foe_key of the wall she's currently up against (cleared on a win)
        self._last_roster_n = None      # de-dupe the roster note so it isn't re-logged every tick
        # ── GARY NEMESIS ARC (Phase 4): the escalating rival grudge across the WHOLE game. Each
        # encounter references the last and builds to a payoff. This is the EMOTIONAL spine the
        # loss-history (above) only feeds the facts to. Persisted, so the grudge survives sessions and
        # (via the journey seam) reaches core Kira / idle chat. name is fixed unless the player renamed him.
        self.rival = {"name": "Gary", "encounters": []}   # encounters: [{won, place, lead, my_party, my_level}]

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
                "key": _foe_key(kind, place, lead, is_trainer, roster),
                "is_trainer": is_trainer,
                "name": ("a trainer" if is_trainer else f"a wild {lead}"),
                "lead": lead, "lead_level": e.get("level"), "types": types,
                "size": size, "roster": roster, "place": place,
                "map_id": tuple(map_id) if map_id else None, "coords": coords,
                "my_party": party_count, "my_level": lead_level,
            }
            self.last_foe = self.cur          # persists past observe_battle_end (the whiteout-backstop reads it)
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

    def note_blackout(self):
        """WHITEOUT-BACKSTOP (the swallow-proof loss record). The live strat_memory showed losses={}/
        active_wall=null DESPITE a 3× death-loop — the battle-end HP read swallowed the outcome, so the
        wall never recorded and the strategic-stuck floor + spatial gate were both STARVED. The whiteout
        itself (map group != 3, the keystone the codebase trusts) is irrefutable proof she lost, whatever
        the battle returned. The free_roam blackout handler calls this so the loss records regardless.
        Records `last_foe` as a loss via the normal path (reuses observe_battle_end's loss branch).
        Caller guards against double-recording a battle the in-battle path already caught."""
        if not self.last_foe:
            self.log("   [strat] blackout detected but no last-foe snapshot — can't attribute the loss (LOUD)")
            return False
        self.cur = dict(self.last_foe)
        self.observe_battle_end(None, "blackout")   # -> lost=True -> records the loss + sets active_wall
        self.log(f"   [strat] WHITEOUT-BACKSTOP: recorded the loss vs {self.last_foe.get('key')} from the "
                 f"whiteout (the battle-end read had swallowed it) — active_wall now set")
        return True

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

    # ── GARY NEMESIS ARC (Phase 4) ───────────────────────────────────────────────────────────────
    def note_rival_encounter(self, won, place="", lead="", my_party=None, my_level=None):
        """Record a Gary/rival encounter as a STORY BEAT. The harness calls this when it knows a battle
        is the rival (the opening Pallet rival is known; later rivals are recon-flagged — wire as their
        detection lands). Each beat escalates the arc."""
        self.rival["encounters"].append({
            "won": bool(won), "place": place or "", "lead": lead or "",
            "my_party": my_party, "my_level": my_level,
        })
        self.rival["encounters"] = self.rival["encounters"][-12:]
        w = sum(1 for e in self.rival["encounters"] if e["won"])
        l = len(self.rival["encounters"]) - w
        self.log(f"   [strat] RIVAL beat #{len(self.rival['encounters'])} vs {self.rival['name']} "
                 f"(won={won}, place={place!r}) — grudge now {w}W-{l}L")

    def rival_grudge_note(self):
        """The escalating-grudge note: references the LAST encounter and builds toward the payoff, framed
        for HER to voice in character. '' until they've actually met. The arc, not just a stat line."""
        enc = self.rival["encounters"]
        if not enc:
            return ""
        name = self.rival["name"]
        last = enc[-1]
        w = sum(1 for e in enc if e["won"])
        l = len(enc) - w
        n = len(enc)
        where = f" at {last['place']}" if last.get("place") else ""
        if n == 1:
            if last["won"]:
                return (f"You and {name} have history now — you beat him in that first showdown{where}. "
                        f"He won't take it well. This rivalry is just getting started.")
            return (f"{name} got you in that first showdown{where} — smug as ever. That one's going to "
                    f"stick. You'll get him next time. This rivalry is just getting started.")
        # ongoing arc — reference the running tally + the last meeting, build momentum toward the payoff
        tally = f"{w}-{l}" if (w or l) else "even"
        if last["won"]:
            lead_in = f"Last time{where} you finally got the better of {name}"
        else:
            lead_in = f"Last time{where} {name} beat you again"
        edge = ("you're ahead in this rivalry now" if w > l else
                "he's still got your number — for now" if l > w else
                "you're dead even, and it's personal")
        return (f"The {name} grudge runs deep — you've met {n} times ({tally}). {lead_in}, and "
                f"{edge}. Every time you cross paths it's the whole journey on the line. The next one matters.")

    # ── SPATIAL WALL (Batch 4 Phase 2): persist the wall as a place + judge "still gated vs grown" ──
    def active_wall_rec(self):
        key = self.active_wall
        return self.losses.get(key) if key else None

    def retire_active_wall(self, reason=""):
        """RETIRE a conquered/outgrown wall record (2026-07-06). A wall's job is the strategic-stuck
        floor + the readiness grind-exit; once she's crossed the bar AND advanced past its region,
        keeping it alive poisons every NEW route (the READINESS→GO pull pruned the nursery forever).
        Deletes the record so a fresh loss re-records cleanly. LOUD."""
        key = self.active_wall
        if not key:
            return
        rec = self.losses.pop(key, None)
        self.active_wall = None
        self.recent = [k for k in self.recent if k != key]
        self.log(f"   [strat] WALL RETIRED: {key} ({(rec or {}).get('count', '?')}x losses) — {reason}; "
                 f"a fresh loss re-records")

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

    # ── STRATEGIC-STUCK FLOOR (sibling to deep-wedge, but for STRATEGY not navigation) ──────────────
    # The navigational watchdog can't see this loop: she's walking + fighting + healing, so the world
    # fingerprint keeps CHANGING (stays GREEN) while she loses to the same wall over and over. This
    # detects the STRATEGIC deadlock — repeated identical losses with no roster/level gain between them —
    # the die→re-charge→die loop that is a run-existence threat. It only READS already-tracked data.
    def strategically_stuck(self, party_count, lead_level):
        """The wall rec if she's strategically stuck (≥STRATEGIC_STUCK_LOSSES to the same wall AND no
        stronger than her last attempt — nothing changed between tries), else None. Losing once and
        retrying is free will; losing identically N times unchanged is the loop the floor breaks."""
        r = self.active_wall_rec()
        if not r or r.get("count", 0) < STRATEGIC_STUCK_LOSSES:
            return None
        if self.stronger_since_wall(party_count, lead_level):
            return None                                   # she changed something -> a legit retry, not the loop
        return r

    def ready_to_retry(self, party_count, lead_level):
        """The EXIT from strengthen-mode (the 'am I ready? → GO' sense). HIGHER bar than stronger_since_
        wall: she's prepared ENOUGH that going back to the wall is now the move to ACTIVELY surface, not
        just un-block. True when, since the wall last beat her, she's added ≥1 teammate OR gained
        ≥READY_RETRY_LEVELS levels. Returns the wall rec (so the caller can name who/where) or None.
        Without this she un-blocks on the low bar but never RETURNS — grinds forever by inertia (the
        live symptom: caught a teammate + leveled, kept circling the grass)."""
        r = self.active_wall_rec()
        if not r:
            return None
        wp, wl = r.get("my_party"), r.get("my_level")
        if wp is None or wl is None or party_count is None or lead_level is None:
            return None
        grew_team = party_count >= wp + 1
        grew_levels = lead_level >= wl + READY_RETRY_LEVELS
        return r if (grew_team or grew_levels) else None

    def ready_to_retry_note(self, party_count=None, lead_level=None):
        """The DOMINANT 'go back NOW' directive — the positive pull that makes RETURNING attractive (not
        just the absence of a block). Names what she built + points her back at the wall. Capability-not-
        script: still her call, but the moment is surfaced. '' if she hasn't crossed the readiness bar."""
        r = self.ready_to_retry(party_count, lead_level)
        if not r:
            return ""
        if r.get("is_trainer"):
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            who = f"{r['name']} ({r['lead']} lead, {sz})" if r.get("name") else f"that trainer ({r['lead']} lead, {sz})"
        else:
            who = f"the wild {r['lead']}"
        place = r.get("place") or "that route"
        grew = []
        if party_count is not None and r.get("my_party") is not None and party_count > r["my_party"]:
            grew.append(f"you've got backup now ({party_count} on the team, up from {r['my_party']})")
        if lead_level is not None and r.get("my_level") is not None and lead_level > r["my_level"]:
            grew.append(f"you're up to L{lead_level} (from L{r['my_level']} when they beat you)")
        grew_str = "; ".join(grew) or "you've trained up since"
        return (f"THIS IS THE MOMENT — you're ready now: {grew_str}. You strengthened up exactly so you "
                f"could take {who} at {place} — so stop circling the grass and GO. Head back and take "
                f"that wall; this time you've got what you didn't before. (Still your call — but the prep "
                f"is done, the move now is to go finish it.)")

    def strategic_stuck_note(self):
        """The DOMINANT directive (stronger than loss_awareness): name the loop and the fix. Capability-
        not-script — it tells her strengthening comes FIRST and lists how, but she still picks which way."""
        r = self.active_wall_rec()
        if not r:
            return ""
        n = r.get("count", 0)
        if r.get("is_trainer"):
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            who = f"that trainer ({r['lead']} lead, {sz})"
        else:
            who = f"the wild {r['lead']}"
        return (f"STRATEGIC DEAD-END — stop and think. You've now lost to {who} {n} times with the SAME "
                f"team at the SAME level. This is NOT something you grind out by trying again harder — your "
                f"APPROACH is wrong, and re-charging in unchanged WILL black you out a {n + 1}th time. Do "
                f"NOT march back into them. The only move is to get STRONGER first: grind levels in safe "
                f"grass, catch a teammate so you're not running solo, or stock up on healing — THEN come "
                f"back and take them. You choose HOW you strengthen, but strengthening comes first, period.")

    # ── STRATEGIC UNDERLEVEL-GRIND (Task B): recognise "the wall beats me because my TEAM is too
    # weak" and derive the level the floor must reach. The wall she's losing to carries the FOE's level
    # she fought (observe_battle_start snapshots `lead_level`), so the readiness target self-calibrates
    # off the LIVE opponent — no hardcoded map/disasm KB to drift (cross-check rule honoured: it's the
    # foe level she actually observed). PURE read; the campaign compares it against her team FLOOR. ──
    def underlevel_target(self):
        """The level her team FLOOR should reach to have a fair shot at the active wall — the foe's
        level she lost to + UNDERLEVEL_MARGIN. None if there's no active wall or the foe level is
        unknown (then it's not a recognisable level problem — strategy/type, handled by loss_awareness)."""
        r = self.active_wall_rec()
        if not r:
            return None
        fl = r.get("lead_level")
        if not fl:
            return None
        return int(fl) + UNDERLEVEL_MARGIN

    def overpower_target(self):
        """The level her ACE should reach to bulldoze the active wall when the bench can't be levelled
        (switch gated): her ACE's level WHEN IT LOST + OVERPOWER_MARGIN (a few levels of bulk above where
        the wall beat her). Falls back to the foe's lead level + a wider margin if her-level is unknown.
        None if there's no active wall. Self-calibrating: if she grinds up and STILL loses, my_level
        refreshes to the higher loss level, so the next target ratchets up until she wins."""
        r = self.active_wall_rec()
        if not r:
            return None
        my = r.get("my_level")
        if my:
            return int(my) + OVERPOWER_MARGIN
        fl = r.get("lead_level")
        return (int(fl) + OVERPOWER_MARGIN + 6) if fl else None

    def prep_team_note(self, weak_names, target):
        """The strategic rationale she voices while prepping: name the under-levelled members and the
        plan (level THEM, not the ace, to readiness, THEN push through). Capability-not-script — the
        reason, surfaced; she still drives. '' if there's no active wall to prep for."""
        r = self.active_wall_rec()
        if not r or not target:
            return ""
        if r.get("is_trainer"):
            sz = f"{r['size']} Pokémon" if r.get("size") else "a full team"
            wall = f"those trainers ({r['lead']} lead, {sz})"
        else:
            wall = f"the wild {r['lead']}"
        where = f" at {r['place']}" if r.get("place") else ""
        if weak_names:
            who = (weak_names[0] if len(weak_names) == 1
                   else (", ".join(weak_names[:-1]) + " and " + weak_names[-1]))
            lead_in = f"{who} {'is' if len(weak_names) == 1 else 'are'} too weak for {wall}{where}"
        else:
            lead_in = f"my team's under-levelled for {wall}{where}"
        return (f"{lead_in} — so before I push through, I'm going to level the weak ones up to about "
                f"L{target} (field THEM in the grass, not my strongest — that's how the whole team gets "
                f"stronger). Then I go back and cross. (My call — but charging in under-levelled just "
                f"blacks me out again.)")

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
        return {"losses": self.losses, "recent": self.recent, "active_wall": self.active_wall,
                "rival": self.rival}

    def from_dict(self, d):
        self.losses = d.get("losses") or {}
        self.recent = d.get("recent") or []
        self.active_wall = d.get("active_wall")
        r = d.get("rival")
        if isinstance(r, dict):
            self.rival = {"name": r.get("name") or "Gary", "encounters": r.get("encounters") or []}
        self._consolidate_trainer_losses()   # heal old active-mon-keyed fragmentation on load

    def _consolidate_trainer_losses(self):
        """MIGRATION (2026-06-28): old saves recorded ONE trainer as several active-mon-keyed
        walls (see _foe_key — the live Gary save had 3 count-1 keys). Merge any trainer losses
        that share (place, roster) into a single stable-keyed wall with the SUMMED count, so a
        wall that beat her N times finally reads as N and the strategic-stuck floor can fire on
        resume. Idempotent (re-keys to the same canonical key → running twice is a no-op)."""
        if not self.losses:
            return
        merged, remap = {}, {}
        for old_key, rec in self.losses.items():
            if rec.get("is_trainer") and rec.get("roster"):
                new_key = _foe_key("trainer", rec.get("place"), rec.get("lead"),
                                   True, rec.get("roster"))
            else:
                new_key = old_key
            remap[old_key] = new_key
            if new_key in merged:
                m = merged[new_key]
                m["count"] = m.get("count", 0) + rec.get("count", 0)
                if (rec.get("my_level") or 0) >= (m.get("my_level") or 0):   # latest attempt's snapshot
                    for f in ("map_id", "coords", "my_party", "my_level", "lead",
                              "lead_level", "types", "size"):
                        if rec.get(f) is not None:
                            m[f] = rec[f]
            else:
                nr = dict(rec); nr["key"] = new_key
                merged[new_key] = nr
        if len(merged) < len(self.losses):
            n_before = len(self.losses)
            self.losses = merged
            self.recent = [remap.get(k, k) for k in self.recent]
            if self.active_wall:
                self.active_wall = remap.get(self.active_wall, self.active_wall)
            wall = self.losses.get(self.active_wall) if self.active_wall else None
            self.log(f"   [strat] CONSOLIDATED fragmented trainer losses {n_before}->{len(merged)} "
                     f"keys (active-mon-keyed walls merged on stable roster id)"
                     + (f"; active wall vs {wall.get('lead')} now {wall.get('count')}x "
                        f"— strategic-stuck floor can now fire" if wall else ""))
        else:
            self.losses = merged   # re-key in place (no count change) so future records unify

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
