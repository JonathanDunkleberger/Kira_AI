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
import re
import threading
import time
import urllib.request
from collections import deque
from queue import Queue, Empty

# FIX 1 (dialogue half) — REPETITION-AVERSE commentary: looping game/NPC text (e.g. a wild Slowbro that
# keeps "turning away" / "loafing around") was re-commentated as new for minutes. The exact back-to-back
# dedup missed it because the lines ALTERNATE. This keeps a small window of recently-voiced lines
# (normalized) and drops a LOW-tier (grind/dialogue) line that repeats one already in the window — so
# she stops narrating the same loop. Big beats (T2/T3: badges, level-ups, gym leaders) are NEVER deduped.
DEDUP_WINDOW = int(os.getenv("POKEMON_DEDUP_WINDOW", "8"))


def _norm(s: str) -> str:
    """Normalize for repeat-detection: lowercase, punctuation/whitespace collapsed to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

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

# ── OFF-RENDER-THREAD HTTP (the deeper lag fix — STATE §0 item 4) ─────────────────────────────
# The reaction/dialogue POSTs and the is_speaking GET used to run SYNCHRONOUSLY on the main render
# thread: every LLM decision froze game video + music for the HTTP round-trip (2000ms+ exhibits in
# the pop-in log). With async ON (default), fire-and-forget POSTs (emit/journey/alert) go to a FIFO
# worker thread, is_speaking reads a ~150ms-cached background poll (no per-frame HTTP), and a blocking
# choose() pumps frames while it waits so the world stays live while she thinks. Kill-switch: set
# POKEMON_VOICE_ASYNC=0 to revert to the old fully-synchronous path (one-variable safety revert).
VOICE_ASYNC = os.getenv("POKEMON_VOICE_ASYNC", "1") == "1"
# F-7(b) FRESHNESS WINDOW: a T0/T1 brisk aside that sat queued longer than this is a would-miss —
# dropped loud rather than fired stale-as-live (moment alignment beats completeness for asides).
# T2+ beats are exempt (a badge/gym moment always lands). 0 disables the drop entirely.
STALE_DROP_S = float(os.getenv("POKEMON_STALE_DROP_S", "6.0"))
SPEAKING_POLL_S = float(os.getenv("POKEMON_SPEAKING_POLL_S", "0.15"))   # is_speaking cache refresh

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
    if "shiny" in s or "a legendary" in s:           # Phase 2D big-moment recognition (the clippable one)
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
        self._recent_norm = deque(maxlen=DEDUP_WINDOW)   # FIX 1: recently-voiced lines (normalized) to
        #                                                  drop LOW-tier repeats (the looping-NPC loop)
        self._last_fire_ts = 0.0           # GLOBAL silence-floor clock (any fired reaction)
        self._last_grind_ts = 0.0          # Tier-1 rolling-gap clock
        self._last_ambient_ts = 0.0        # Tier-0 ambient-trickle clock
        self.last_dialogue_tier = None     # tier of the last dialogue fire (play_live reads it to hold)
        self._ctx = {}                     # battle context (trainer / rare) set by the harness
        self.n_sent = 0
        self.n_skipped = 0
        self.n_failed = 0
        self.stream = []                   # ordered [(tier, kind, summary)] - the proof readback

        # OFF-THREAD HTTP plumbing (the lag fix). frame_pump is set by the harness (play_live) to a
        # zero-input "run one frame + draw" callback so choose() keeps the emulator (video+music) live
        # while it blocks on her decision. Left None -> choose blocks as before (headless/proof runs).
        self.frame_pump = None
        self._async = VOICE_ASYNC
        self._speaking = False             # background-polled cache of the bot's is_speaking
        self._closed = False
        if self._async:
            self._q = Queue()              # FIFO of fire-and-forget POSTs (order preserved)
            self._sender = threading.Thread(target=self._sender_loop, name="kira-voice-sender", daemon=True)
            self._sender.start()
            self._poller = threading.Thread(target=self._speaking_loop, name="kira-voice-speaking", daemon=True)
            self._poller.start()

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
        # ASYNC: return the background-polled cache (no HTTP on the render thread — this is called
        # every frame inside pace()'s hold loops, so a live GET here was a per-frame stutter).
        if self._async:
            return self._speaking
        try:
            with urllib.request.urlopen(f"{self.url}/state", timeout=3) as r:
                return bool(json.load(r).get("is_speaking"))
        except Exception:
            return False

    # ── background workers (async path) ──────────────────────────────────────────
    def _speaking_loop(self):
        """Poll /state off the render thread and cache is_speaking, so pace()'s per-frame reads
        are instant. Best-effort; a bot-down poll just caches False and keeps trying."""
        while not self._closed:
            try:
                with urllib.request.urlopen(f"{self.url}/state", timeout=3) as r:
                    self._speaking = bool(json.load(r).get("is_speaking"))
            except Exception:
                self._speaking = False
            time.sleep(SPEAKING_POLL_S)

    def _sender_loop(self):
        """Drain the FIFO of fire-and-forget events and POST them off the render thread. Single
        worker so her reactions still fire IN ORDER; the render loop never blocks on the round-trip."""
        while not self._closed:
            try:
                job = self._q.get(timeout=0.25)
            except Empty:
                continue
            if job is None:
                break
            kind, body = job
            self._send_event(*body) if kind == "event" else self._send_raw(*body)
            self._q.task_done()

    def _send_event(self, tier, kind, summary, trig_ts=None):
        """The actual pokemon_event POST + loud logging (formerly inline in emit). Runs on the
        sender thread when async, or inline when sync.
        F-7 MOMENT ALIGNMENT: `trig_ts` is the moment the reaction was TRIGGERED. (a) The queue age
        + POST round-trip are logged per fire, so the event→voice chain measures itself on every
        watched run (the F-10 first-textbox stall will name its component next take). (b) FRESHNESS
        WINDOW: a LOW-tier (T0/T1 brisk aside) line that sat queued past STALE_DROP_S is a would-miss
        — DROPPED loud, never fired as-live (a viewer reads a 8s-late 'ooh a sign' as broken). Big
        beats (T2+) always land — late relief still beats swallowed relief."""
        age = (time.time() - trig_ts) if trig_ts else 0.0
        if trig_ts and tier < 2 and STALE_DROP_S > 0 and age > STALE_DROP_S:
            self.n_skipped += 1
            self.log(f"   [kira-voice] ·stale-drop· T{tier} ({kind}) aged {age:.1f}s in queue "
                     f"(> {STALE_DROP_S:g}s freshness window) {summary!r}")
            return
        try:
            t0 = time.time()
            res = self._post("pokemon_event", name=summary, tier=tier, kind=kind)
            post_ms = (time.time() - t0) * 1000
            self.n_sent += 1
            fired = res.get("fired") if isinstance(res, dict) else res
            self.log(f"   [kira-voice] -> T{tier}·{TIER_NAME[tier]}· ({kind}) {summary!r}  fired={fired}"
                     + (f"  [age {age:.1f}s + post {post_ms:.0f}ms]" if trig_ts else ""))
        except Exception as e:
            self.n_failed += 1
            self.log(f"   [kira-voice] !! POST FAILED (bot down?) T{tier} ({kind}) {summary!r}: {e}")

    def _send_raw(self, action, body):
        """Off-thread fire-and-forget for journey/alert (no response needed)."""
        try:
            self._post(action, **body)
        except Exception as e:
            self.log(f"   [kira-voice] ·{action} skip· (bot down?) {e}")

    def close(self, drain_s=2.0):
        """Flush pending reactions and stop the workers (called from the harness finally-block).
        Daemon threads would die with the process anyway; this just lets the last few POSTs land."""
        if not self._async or self._closed:
            self._closed = True
            return
        deadline = time.time() + drain_s
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._closed = True

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

        # FIX 1 — REPETITION-AVERSE: drop a LOW-tier (grind/dialogue) line that repeats one she's voiced
        # in the recent window. Kills the re-commentary of looping game/NPC text (the 3.5-min Slowbro
        # loop). Big beats (T2/T3 — badges/level-ups/gym leaders) are never deduped (always land).
        _n = _norm(summary)
        if tier < 2 and _n and _n in self._recent_norm:
            self.n_skipped += 1
            self.log(f"   [kira-voice] ·dedup· (repeat text, T{tier}, looping?) {summary!r}")
            return None

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
        if _n:
            self._recent_norm.append(_n)               # FIX 1: remember it so a repeat is dropped
        self.stream.append((tier, kind, summary))
        # ASYNC: the gating/dedup/fire-rate above is pure CPU (microseconds) and stays on the caller
        # thread; only the HTTP round-trip is offloaded, so the render loop never blocks on it. The
        # tier is returned immediately (callers read it for the dialogue hold) — the POST lands async.
        # F-7(a): `now` rides along as the TRIGGERING MOMENT so the sender can measure queue age
        # (moment-alignment telemetry) and drop a stale low-tier line instead of firing it as-live.
        if self._async:
            self._q.put(("event", (tier, kind, summary, now)))
        else:
            self._send_event(tier, kind, summary, now)
        return tier

    # ── CONTINUITY-INTO-CORE seam (Phase 4): push her journey narrative to core Kira ──
    def journey(self, state):
        """Fire-and-forget the journey-continuity snapshot to core Kira (POST /cmd/pokemon_journey),
        so she resumes KNOWING her story and can speak it in IDLE CHAT / any game — persisted core-side,
        independent of how the Pokémon session launched. `state` is campaign._journey_narrative(). Never
        raises (a continuity write must never disturb the run); a bot-down skip just logs."""
        if not state:
            return
        if self._async:
            self._q.put(("raw", ("pokemon_journey", {"ctx": state})))   # off-thread, order-preserved
            return
        try:
            self._post("pokemon_journey", ctx=state)   # nested under ctx to match the seam's body model
        except Exception as e:
            self.log(f"   [kira-voice] ·journey skip· (bot down?) {e}")

    # ── DEAD-MAN'S SWITCH seam (Phase 2): alert Jonny when recovery itself failed ──
    def alert(self, message):
        """Fire a critical out-of-band alert to the human (POST /cmd/pokemon_alert -> Discord webhook +
        loud log). Used by the dead-man's switch when deep-wedge recovery is exhausted. Never raises —
        a failed alert must not crash the (already-abandoned) run; it just logs."""
        if not message:
            return
        try:
            self._post("pokemon_alert", name=message)
        except Exception as e:
            self.log(f"   [kira-voice] !! ALERT POST FAILED (bot down?): {e}")

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

        def _do_choose():
            """The blocking oracle round-trip. Returns (choice, error)."""
            try:
                req = urllib.request.Request(f"{self.url}/cmd/pokemon_choose",
                                             data=json.dumps(body, default=_json_safe).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    res = json.load(r)
                return ((res.get("choice") if isinstance(res, dict) else None) or None), None
            except Exception as e:
                return None, e

        # LAG FIX: choose() BLOCKS on her LLM decision (up to `timeout`s). Without pumping, the
        # emulator freezes for that whole think — dead video + silent music, the worst watch stutter.
        # When the harness gave us a frame_pump, run the round-trip on a worker thread and keep the
        # world LIVE (zero-input frames: she idles in place while she thinks). No frame_pump (headless
        # proof runs) -> plain blocking call, unchanged.
        if self.frame_pump and not self._closed:
            box = {}
            th = threading.Thread(target=lambda: box.update(zip(("choice", "err"), _do_choose())),
                                  name="kira-voice-choose", daemon=True)
            th.start()
            while th.is_alive():
                try:
                    self.frame_pump()
                except Exception:
                    time.sleep(0.005)   # pump failed (window gone?) — don't hot-spin the wait
            choice, err = box.get("choice"), box.get("err")
        else:
            choice, err = _do_choose()

        if err is not None:
            self.log(f"   [kira-voice] !! ORACLE POST FAILED (bot down?) kind={kind}: {err} -> None")
            return None
        self.log(f"   [kira-voice] ORACLE kind={kind} options={opt_keys} -> pick={choice!r}")
        return choice

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
