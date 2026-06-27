"""pokemon_voice.py - HARNESS-LEVEL soul-flow wiring + SALIENCE TIERING (additive, no engine touch).

Connects the autonomous Pokemon arc's NEUTRAL game-events + the overworld DialogueReader to
Kira's EXISTING reaction seam:

    POST /cmd/pokemon_event {name, tier}  ->  bot._pokemon_react(summary, tier)  ->  her self/voice

This is the WIRE + the SALIENCE SPINE, not the soul. It touches NO personality/engine code:
battle_agent, campaign, bridge, dialogue_reader, _pokemon_react's reaction generation,
_build_self_block, voice/mood/bond are all UNMODIFIED. We only (a) attach reaction hooks to the
already-built objects through their existing seams and (b) classify each event into a SALIENCE
TIER that three levers read:

  TIER 0 Ambient  - generic swing / minor HP / stuck     -> SKIP (don't even POST)
  TIER 1 Brisk    - routine wild win/faint/encounter/NPC -> fire, rolling min-gap, snappy + no hold
  TIER 2 Savor    - level-up / trainer / rare / TYPE line -> always fire, ~3s hold, Sonnet
  TIER 3 Big      - gym leader / BADGE / evolution / blackout -> always fire, full savor, Opus

The tier rides the POST as a HINT; the bot reads it ONLY to pick the model + length (Opus/Sonnet,
max_tokens) and to scale nothing about WHAT she says. The post-fight HOLD (play_live.pace) and the
FIRE-RATE (here) also read the tier. One spine, three levers.

Constraint #3 (silent failure is the enemy): every emit/skip is LOGGED, and a POST failure (bot
down) is ANNOUNCED, not swallowed.
"""
import json
import os
import time
import urllib.request

# ── fire-rate knobs (env-tunable live, no code edit) ──────────────────────────
GRIND_GAP_S = float(os.getenv("POKEMON_GRIND_GAP_S", "1.8"))   # Tier-1 rolling min-gap (chatty grind)
# Tier-0 ambient TRICKLE: fire generic swings on a LONG gap (occasional "ooh nice hit" riffing)
# instead of full-drop. Set POKEMON_AMBIENT_GAP_S=0 to drop ambient entirely. Kept fairly long so
# T0 swing-by-swing stays LIGHT (no clog) while T1/travel-muse carry the cadence.
AMBIENT_GAP_S = float(os.getenv("POKEMON_AMBIENT_GAP_S", "5.0"))
# GLOBAL SILENCE FLOOR: guarantee at least this gap between ANY two reactions (T1/T2) so there's a
# natural window for Jonny to start talking. Big beats (T3) bypass it — never gate the badge.
FLOOR_S = float(os.getenv("POKEMON_FLOOR_S", "1.0"))
# IN A SAVOR/BIG FIGHT (trainer/gym/rare) the floor drops to near-dedupe-only so rapid battle beats
# land — she shit-talks the fight like before. No firehose risk: the bot's is_speaking gate already
# rate-limits to ~one reaction per spoken line, so this only un-floors the beats she'd otherwise
# silently drop. The GRIND keeps the full FLOOR_S for Jonny's breathing room. (Item 3.)
BATTLE_FLOOR_S = float(os.getenv("POKEMON_BATTLE_FLOOR_S", "0.4"))

# species (by lowercased name) that are a SAVOR-worthy rare encounter for this leg
RARE_SPECIES = {"pikachu"}

# trainer / gym-guide BANTER markers (pre-battle challenge + post-battle concession): bump the
# line to Tier 2 ("Tier 1.5") so the text lands with a short hold, without making every NPC big.
TRAINER_DIALOGUE_MARKERS = (
    "let's battle", "battle 'em", "i give", "you're good", "you're strong", "darn",
    "you have pok", "wait, you", "you're not bad",
)

# Tier names for logs
TIER_NAME = {0: "ambient", 1: "brisk", 2: "savor", 3: "BIG"}


def classify(summary, tags=None, ctx=None):
    """Map an event -> salience tier (0..3) from signals that ALREADY EXIST: the summary text,
    the DialogueReader salience tags (GYM_LEADER/TYPE/PLACE/ITEM), and a small battle CONTEXT
    dict (trainer-vs-wild + rare enemy) the harness reads from RAM. Pure function, no I/O."""
    s = (summary or "").lower()
    tags = tags or []
    ctx = ctx or {}

    # ── TIER 3 — big beats (savor hard) ──
    if any(k in s for k in ("boulderbadge", "beat brock", "boulder badge", "badge from")):
        return 3
    if "evolved into" in s or "is evolving" in s:
        return 3
    if "GYM_LEADER" in tags:
        return 3
    if any(k in s for k in ("you lost", "blacked out", "knocked out", "we got knocked")):
        return 3                                   # a loss is a big (bad) beat worth dwelling on

    # ── TIER 2 — savor ──
    if "leveled up" in s:
        return 2
    if "TYPE" in tags:
        return 2
    # MOVE-NARRATION (a swing) — routed BEFORE the trainer catch so a real fight stays CHATTY-BUT-
    # QUICK: inside a battle (trainer/rare) a routine swing is a SNAPPY T1 jab (she shit-talks the
    # fight without each swing becoming a ~10s T2 savor, so more lands per fight); outside a battle
    # it's ambient T0 (drop/trickle). The grind has no battle context -> unchanged. Big hits
    # (solid hit / super / critical) fall through to the T2 savor below. (Item 3.)
    if s.startswith("used ") and not any(k in s for k in ("solid hit", "super", "critical")):
        return 1 if (ctx.get("trainer") or ctx.get("rare")) else 0
    if ctx.get("trainer"):
        return 2                                   # any other event inside a trainer battle
    if ctx.get("rare"):
        return 2                                   # any event inside a rare-encounter battle
    if "the trainer sent out" in s:
        return 2
    if any(m in s for m in TRAINER_DIALOGUE_MARKERS):
        return 2                                   # trainer/gym-guide banter ("Tier 1.5") -> let it land

    # ── TIER 0 — ambient / low value (skip) ──
    # (generic "used X" swings are handled above — snappy T1 in a battle, T0 in the grind)
    if any(k in s for k in ("you took a big hit", "low hp", "the battle ended",
                            "blocking the way", "taking forever", "properly stuck")):
        return 0

    # ── TIER 1 — brisk routine (the grind default) ──
    return 1


def _json_safe(o):
    """json.dumps `default`: coerce sets/frozensets (e.g. GAME_KNOWLEDGE['rare']/['legendaries'])
    to a sorted list so the ORACLE body serializes. Sets are the RIGHT structure upstream (membership
    fact-table) — we coerce only here, at the JSON boundary, never at the source."""
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class KiraVoice:
    """The single chokepoint from the dumb game harness up to Kira's reaction seam, with the
    salience spine. Fire-rate lever lives here; post-hold + model-tier read the same classify()."""

    def __init__(self, url="http://127.0.0.1:8766", log=print, timeout=6):
        self.url = url.rstrip("/")
        self.log = log
        self.timeout = timeout
        self._last_summary = None          # de-dupe identical back-to-back fires
        self._last_fire_ts = 0.0           # GLOBAL silence-floor clock (any fired reaction)
        self._last_grind_ts = 0.0          # Tier-1 rolling-gap clock
        self._last_ambient_ts = 0.0        # Tier-0 ambient-trickle clock
        self.last_dialogue_tier = None     # tier of the last dialogue fire (play_live reads it to hold)
        self._ctx = {}                     # battle context (trainer / rare) set by the harness
        self.n_sent = 0
        self.n_skipped = 0
        self.n_failed = 0
        self.stream = []                   # ordered [(tier, kind, summary)] - the proof readback

    # ── battle context (RAM-derived salience the text alone can't carry) ─────────
    def set_context(self, **kw):
        self._ctx.update(kw)

    def clear_context(self):
        self._ctx = {}

    def tier_of(self, summary, tags=None):
        """Public classify against the live context - used by play_live.pace() to scale the hold."""
        return classify(summary, tags, self._ctx)

    # ── transport ──────────────────────────────────────────────────────────────
    def _post(self, action, **body):
        req = urllib.request.Request(f"{self.url}/cmd/{action}",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def is_speaking(self):
        try:
            with urllib.request.urlopen(f"{self.url}/state", timeout=3) as r:
                return bool(json.load(r).get("is_speaking"))
        except Exception:
            return False

    # ── the reaction sink (with the FIRE-RATE lever) ─────────────────────────────
    def emit(self, summary, *, kind="event", tier=None, tags=None, **_):
        """Route a NEUTRAL summary to her seam, gated by salience. `tier` overrides the classifier
        (level-up/evolution come pre-tiered from the harness). Returns the tier actually fired, or
        None if skipped. `**_` swallows the battle engine's extra kwargs."""
        summary = (summary or "").strip()
        if not summary or summary == self._last_summary:
            return None
        if tier is None:
            tier = classify(summary, tags, self._ctx)

        # FIRE-RATE lever -------------------------------------------------------
        now = time.time()
        # GLOBAL SILENCE FLOOR: leave breathing room after any reaction so Jonny can talk into it.
        # T3 big beats bypass (never suppress the badge / a gym-leader moment). In a savor/big fight
        # the floor drops to BATTLE_FLOOR_S so the fight stays chatty (item 3).
        floor = BATTLE_FLOOR_S if (self._ctx.get("trainer") or self._ctx.get("rare")) else FLOOR_S
        if tier < 3 and (now - self._last_fire_ts) < floor:
            self.n_skipped += 1
            self.log(f"   [kira-voice] ·floor· (<{floor:g}s since last) T{tier} {summary!r}")
            return None
        if tier == 0:
            # ambient TRICKLE: voice the occasional generic swing on a long gap (riffing density)
            # instead of full-drop. AMBIENT_GAP_S<=0 -> drop entirely.
            if AMBIENT_GAP_S <= 0 or (now - self._last_ambient_ts) < AMBIENT_GAP_S:
                self.n_skipped += 1
                self.log(f"   [kira-voice] ·skip· (ambient) {summary!r}")
                return None
            self._last_ambient_ts = now
            self._last_grind_ts = now
            tier = 1                                   # voiced as a brisk aside
        elif tier == 1:
            if (now - self._last_grind_ts) < GRIND_GAP_S:
                self.n_skipped += 1
                self.log(f"   [kira-voice] ·throttle· (brisk <{GRIND_GAP_S:g}s) {summary!r}")
                return None
            self._last_grind_ts = now
        else:                                          # Tier 2/3 always fire AND reset the grind clock
            self._last_grind_ts = now

        self._last_fire_ts = now                       # arm the global silence floor
        self._last_summary = summary
        self.stream.append((tier, kind, summary))
        try:
            res = self._post("pokemon_event", name=summary, tier=tier)
            self.n_sent += 1
            fired = res.get("fired") if isinstance(res, dict) else res
            self.log(f"   [kira-voice] -> T{tier}·{TIER_NAME[tier]}· ({kind}) {summary!r}  fired={fired}")
        except Exception as e:
            self.n_failed += 1
            self.log(f"   [kira-voice] !! POST FAILED (bot down?) T{tier} ({kind}) {summary!r}: {e}")
        return tier

    # ── the SOUL ORACLE client (Batch-2 keystone: a choice becomes HERS) ──────────
    def choose(self, kind, options, ctx=None, timeout=30):
        """Ask her SELF (over the seam) to make a STRUCTURED pick — the decision counterpart to
        emit's fire-and-forget reaction. BLOCKING request/response (it waits on her LLM, so a longer
        timeout than _post's 6s). `options` is a list OR a dict {key: description}; a dict's
        descriptions ride along in ctx['detail'] to inform her WITHOUT constraining a 'want'.
        Returns her pick (str) or None — no pick / bot down / unparsable, LOGGED loud, never raises.
        The CALLER (campaign._soul_choose) re-validates a constrained pick against the offered set."""
        if isinstance(options, dict):
            opt_keys, detail = list(options.keys()), dict(options)
        else:
            opt_keys, detail = [str(o) for o in (options or [])], {}
        body = {"kind": kind, "options": opt_keys, "ctx": {**(ctx or {}), "detail": detail}}
        try:
            req = urllib.request.Request(f"{self.url}/cmd/pokemon_choose",
                                         data=json.dumps(body, default=_json_safe).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.load(r)
            choice = (res.get("choice") if isinstance(res, dict) else None) or None
            self.log(f"   [kira-voice] ORACLE kind={kind} options={opt_keys} -> pick={choice!r}")
            return choice
        except Exception as e:
            self.log(f"   [kira-voice] !! ORACLE POST FAILED (bot down?) kind={kind}: {e} -> None")
            return None

    def beat(self, summary):
        """Traveler 'savor' beat (wild encounter / new area) -> classified like any event."""
        return self.emit(summary, kind="beat")

    def on_dialogue(self, line, tags):
        """DialogueReader hook: an overworld line she just read -> her seam. The salience tags
        (GYM_LEADER -> Tier 3, TYPE -> Tier 2) drive the tier, so Brock's speech savors and a
        random signpost stays brisk."""
        line = (line or "").strip()
        if not line:
            return None
        lead = "the gym leader says" if "GYM_LEADER" in (tags or []) else "you read"
        t = self.emit(f'{lead}: "{line}"', kind="dialogue", tags=tags)
        self.last_dialogue_tier = t            # play_live reads this to hold the A-advance on a T2+ line
        if t is not None and tags:
            self.log(f"   [kira-voice]    (dialogue tags: {', '.join(x.lower() for x in tags)})")
        return t

    # ── summary ──────────────────────────────────────────────────────────────────
    def report(self):
        from collections import Counter
        byt = Counter(t for t, _, _ in self.stream)
        self.log(f"   [kira-voice] === {self.n_sent} fired, {self.n_skipped} gated "
                 f"({self.n_failed} failed) | by tier: "
                 f"T3={byt[3]} T2={byt[2]} T1={byt[1]} (incl ambient trickle) ===")
        return self.stream
