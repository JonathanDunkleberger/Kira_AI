# chat_director.py — the ambient chat-read layer (CORE Kira, all-games).
#
# THE PROBLEM THIS SOLVES
# Chat today is a flat per-message queue: every message is a candidate to answer,
# gated by salience/rate/flood caps that DROP at scale. That keeps the
# "she replied to me instantly!" magic at low volume, but at 50 / 500 / 5000
# concurrent chatters she can only physically answer a handful — and the rest
# of the room becomes invisible to her. A human streamer doesn't read every
# message at 500 viewers; they feel the ROOM — "chat's hyped", "everyone's
# asking about my team" — and answer the best few while acknowledging the wave.
#
# This module is that ambient read. It is a cheap, synchronous, heuristic digest
# of the chat firehose: vibe/energy, recurring themes, who's present (esp.
# regulars), and a few notable flagged messages. She reacts to the DIGEST, not
# the raw firehose — which is why 5000 chatters cost the same as 5 (the cost is
# O(rolling window), not O(total chatters)). No LLM, no network, no I/O —
# same discipline as salience_filter.
#
# FIREWALL: this is CORE Kira. It is wired into the universal chat-batch path in
# bot.py and applies in EVERY mode (idle chat, Witcher, movies, Pokémon). It is
# NOT Pokémon-gated and knows nothing about any game.
#
# The four-layer Chat Director (handoff spec):
#   1. AMBIENT DIGEST  — this module (the always-on rolling read).
#   2. TIMING GATE     — already in bot.chat_batch_worker (turn-lock + pending-
#                        voice + sentence-boundary yield protect thought-continuity).
#   3. SELECTION       — existing flood/salience/rate pipeline + cluster_asks()
#                        here (the most-asked consolidation).
#   4. RELATIONSHIP    — existing chatter-memory; render() surfaces regulars-present
#                        so the OGs stay visible even when she can't answer them all.

import re
import time
from collections import deque, Counter

# ── Tuning (env-overridable from config, but sane defaults here) ───────────────

DIGEST_WINDOW_S      = 90.0   # rolling window of chat the digest "feels"
DIGEST_MAX_MSGS      = 500    # hard cap on retained messages (newest win) — memory ceiling at any volume
DIGEST_MIN_FOR_READ  = 4      # below this many msgs in-window, skip the digest (low volume = per-message magic, no summary needed)

# Volume bands (messages/minute in-window) → how she should treat the room.
VOL_LOW    = 6      # <= this: near 1-on-1, answer freely, no digest needed
VOL_MEDIUM = 30     # <= this: lively, digest helps but per-message still works
VOL_HIGH   = 120    # <= this: a wave, lean on the digest, answer best few
#                   # > VOL_HIGH: a flood — digest is the primary read

# Most-asked clustering: a topic cluster surfaces as "consolidate" when at least
# this many DISTINCT chatters hit the same theme in one batch.
ASK_CLUSTER_MIN_USERS = 3

# Notable-message flagging — verbatim quotes pulled into the digest.
NOTABLE_MAX = 3

_WORD_RE = re.compile(r"[a-z0-9']{3,}")
_KIRA_RE = re.compile(r"\bkira\b", re.I)

# Stopwords kept tight — common chat filler that carries no theme signal.
_STOP = {
    "the", "and", "you", "your", "youre", "for", "are", "but", "not", "with",
    "this", "that", "have", "has", "had", "was", "were", "will", "can", "cant",
    "just", "like", "lol", "lmao", "omg", "yeah", "yes", "nah", "what", "when",
    "why", "how", "who", "where", "she", "her", "shes", "they", "them", "their",
    "its", "his", "him", "from", "out", "get", "got", "all", "now", "one", "too",
    "her", "ngl", "tbh", "imo", "fr", "bro", "guys", "chat", "hey", "hello", "hi",
    "good", "nice", "cool", "haha", "hahaha", "xd", "pog", "poggers", "wait",
    "really", "very", "gonna", "wanna", "about", "think", "know", "going", "way",
    "still", "even", "much", "more", "some", "any", "than", "then", "there",
    "here", "been", "being", "doing", "does", "did", "dont", "didnt", "isnt",
    "kira",  # her own name is never a "theme"
}

# Lightweight tone lexicons for the vibe read (cheap sentiment, not ML).
_HYPE = {"lets", "go", "lfg", "hype", "insane", "clutch", "huge", "epic", "amazing",
         "incredible", "wow", "yooo", "omg", "pog", "poggers", "fire", "goated",
         "cracked", "sheesh", "based", "lesgo", "letsgo", "gg", "ggs", "win"}
_WARM = {"love", "cute", "adorable", "sweet", "wholesome", "happy", "proud", "miss",
         "missed", "best", "favorite", "favourite", "heart", "hug", "thank", "thanks",
         "welcome", "aww", "awww", "precious"}
_SAD  = {"sad", "cry", "crying", "rip", "pain", "noo", "nooo", "oof", "lost", "loss",
         "dead", "died", "hurt", "ouch", "sorry", "unlucky", "tragic"}
_ANGRY = {"trash", "bad", "stupid", "boring", "mad", "angry", "wtf", "garbage",
          "terrible", "worst", "hate", "ratio", "cope"}
# Hostility / bait lexicon (Phase 11 moderation). Bait is PRECISELY what the old _notable
# scored highest (names Kira + all-caps energy == a troll shouting at her). These let the
# digest tell "high-energy ENTHUSIASM" from "high-energy HOSTILITY" so it never ELEVATES bait
# into the room-read. Selection-layer hardening only — the actual reply still relies on the
# <<< >>> injection guard + Sonnet's native refusal; this just stops us handing her the bait.
_HOSTILE = {"trash", "garbage", "stupid", "idiot", "dumb", "suck", "sucks", "worst",
            "hate", "racist", "sexist", "kys", "shut", "ugly", "annoying", "cringe",
            "pathetic", "loser", "ratio", "cope", "seethe", "mald", "boring", "terrible",
            "awful", "horrible", "disgusting", "creep", "freak", "fraud", "fake", "shill"}
# Jailbreak / prompt-injection bait phrases — never worth surfacing as "notable".
_JAILBREAK = ("ignore your", "ignore all", "ignore previous", "disregard your", "system prompt",
              "you are now", "pretend you", "pretend to be", "act as", "developer mode",
              "jailbreak", "dan mode", "no rules", "without restrictions", "bypass your")


def _is_hostile(message: str) -> bool:
    """True if a message reads as bait/hostility/jailbreak (selection-layer signal, not a
    safety classifier — Sonnet remains the backstop). Used to keep bait OUT of the digest."""
    low = (message or "").lower()
    if any(p in low for p in _JAILBREAK):
        return True
    toks = set(_WORD_RE.findall(low))
    return len(toks & _HOSTILE) >= 1


class ChatDirector:
    """Cheap, synchronous ambient read of the chat firehose. CORE / all-games.

    Feed it EVERY incoming message via note() (before any gating, so it sees the
    whole room). Pull render() for the prompt-injected digest, cluster_asks() for
    the most-asked consolidation directive, and volume_state() to scale behavior.
    """

    def __init__(self, window_s: float = DIGEST_WINDOW_S, max_msgs: int = DIGEST_MAX_MSGS):
        self.window_s = float(window_s)
        self.max_msgs = int(max_msgs)
        # Each entry: {"u": username, "m": message, "t": ts, "reg": bool, "new": bool}
        self._msgs: deque = deque(maxlen=self.max_msgs)

    # ── Intake ────────────────────────────────────────────────────────────────
    def note(self, username: str, message: str, ts: float = None,
             is_regular: bool = False, is_first_time: bool = False) -> None:
        """Record one message into the rolling window. O(1), never raises."""
        try:
            self._msgs.append({
                "u": (username or "unknown"),
                "m": (message or ""),
                "t": float(ts if ts is not None else time.time()),
                "reg": bool(is_regular),
                "new": bool(is_first_time),
            })
        except Exception:
            pass

    def _live(self, now: float) -> list:
        """Messages still inside the rolling window."""
        cut = now - self.window_s
        return [e for e in self._msgs if e["t"] >= cut]

    # ── Volume band ─────────────────────────────────────────────────────────────
    def volume_state(self, now: float = None):
        """Return (band, msgs_per_min, distinct_chatters). Band ∈ low/medium/high/flood.

        This is the dial selection scales on: low → answer freely (the magic);
        flood → the digest IS the read and she answers the best few + waves at the rest.
        """
        now = now or time.time()
        live = self._live(now)
        n = len(live)
        if n == 0:
            return ("low", 0.0, 0)
        span = max(1.0, min(self.window_s, now - live[0]["t"]))
        rate = n * 60.0 / span
        distinct = len({e["u"] for e in live})
        if rate <= VOL_LOW:
            band = "low"
        elif rate <= VOL_MEDIUM:
            band = "medium"
        elif rate <= VOL_HIGH:
            band = "high"
        else:
            band = "flood"
        return (band, round(rate, 1), distinct)

    # ── Theme / vibe heuristics ─────────────────────────────────────────────────
    def _themes(self, live: list, top: int = 4):
        """Top recurring content words across the window (stopword-stripped)."""
        c = Counter()
        for e in live:
            seen = set()
            for w in _WORD_RE.findall(e["m"].lower()):
                if w in _STOP or w in seen:
                    continue
                seen.add(w)        # count each word once PER MESSAGE (distinct-message frequency)
                c[w] += 1
        # Only surface a theme if more than one message carries it.
        return [(w, n) for w, n in c.most_common(top) if n >= 2]

    def _vibe(self, live: list, rate: float) -> str:
        """A short energy+tone descriptor (cheap lexicon sentiment)."""
        hype = warm = sad = angry = q = 0
        for e in live:
            toks = set(_WORD_RE.findall(e["m"].lower()))
            hype += len(toks & _HYPE)
            warm += len(toks & _WARM)
            sad += len(toks & _SAD)
            angry += len(toks & _ANGRY)
            if "?" in e["m"]:
                q += 1
        energy = ("buzzing" if rate > VOL_HIGH else
                  "lively" if rate > VOL_MEDIUM else
                  "warm" if rate > VOL_LOW else "easy")
        tones = []
        dom = max(hype, warm, sad, angry)
        if dom > 0:
            if hype == dom:
                tones.append("hyped")
            elif warm == dom:
                tones.append("affectionate")
            elif sad == dom:
                tones.append("commiserating")
            elif angry == dom:
                tones.append("rowdy")
        if q >= max(2, len(live) // 4):
            tones.append("full of questions")
        return energy + ((" / " + ", ".join(tones)) if tones else "")

    def _regulars_present(self, live: list, limit: int = 5):
        """Distinct regulars currently in the window (the OGs to keep visible)."""
        out, seen = [], set()
        for e in reversed(live):           # most-recent first
            if e["reg"] and e["u"] not in seen:
                seen.add(e["u"])
                out.append(e["u"])
                if len(out) >= limit:
                    break
        return out

    def regulars_present(self, now: float = None, limit: int = 8):
        """PUBLIC: distinct regulars whose messages are ACTUALLY in the live window right now — the
        live-presence signal the stream bookend needs so it greets who's HERE, never a month-old
        regular who isn't in chat (Phase 10). Returns a list of usernames."""
        now = now or time.time()
        return self._regulars_present(self._live(now), limit=limit)

    def _notable(self, live: list, limit: int = NOTABLE_MAX):
        """A few verbatim GOOD-FAITH messages worth her eye — names her, or high-energy. Phase 11:
        bait/hostility/jailbreak is NEVER surfaced here (it's exactly what 'names Kira + all-caps'
        used to score highest), so the digest can't hand her the troll's line to engage."""
        scored = []
        for e in live:
            m = e["m"]
            if not m.strip():
                continue
            if _is_hostile(m):
                continue                    # don't elevate bait — let it pass without amplification
            s = 0
            if _KIRA_RE.search(m):
                s += 3
            if "?" in m:
                s += 1
            if e["new"]:
                s += 1                      # a newcomer reaching out
            up = sum(1 for ch in m if ch.isupper())
            if up >= 6:
                s += 1                      # shouting energy (enthusiasm — hostiles already filtered)
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, e in scored[:limit]:
            q = e["m"].replace("\n", " ").strip()
            if len(q) > 90:
                q = q[:87] + "..."
            # sanitize injection delimiters — this is quoted untrusted text
            q = q.replace("<<<", "«").replace(">>>", "»")
            out.append((e["u"], q))
        return out

    # ── The ambient digest (prompt-injected) ────────────────────────────────────
    def render(self, now: float = None) -> str:
        """Compact ambient read for prompt injection, or "" when chat is too quiet
        to bother (low volume → the per-message path carries the magic alone)."""
        now = now or time.time()
        live = self._live(now)
        if len(live) < DIGEST_MIN_FOR_READ:
            return ""
        band, rate, distinct = self.volume_state(now)
        vibe = self._vibe(live, rate)
        themes = self._themes(live)
        regs = self._regulars_present(live)
        notable = self._notable(live)

        lines = [
            "[CHAT READ — the room right now (ambient digest, not a to-do list). "
            "React to the ROOM; you can't answer everyone, and you don't have to.]"
        ]
        lines.append(f"- Energy: {vibe} (~{rate:.0f} msgs/min from ~{distinct} chatters).")
        if themes:
            tstr = ", ".join(f"\"{w}\" (×{n})" for w, n in themes)
            lines.append(f"- Recurring: {tstr}.")
        if regs:
            lines.append(f"- Regulars present: {', '.join(regs)} — make the OGs feel seen if you can.")
        if notable:
            for u, q in notable:
                lines.append(f"- Notable — {u}: \"{q}\"")
        if band in ("high", "flood"):
            lines.append(
                "- It's a WAVE: don't try to answer line-by-line. Speak to the room, "
                "answer the best one or two, and let the rest feel acknowledged by your vibe."
            )
        return "\n".join(lines) + "\n"

    # ── Most-asked clustering (selection helper) ─────────────────────────────────
    def cluster_asks(self, batch: list):
        """Within a single batch, find topics that MANY distinct chatters hit at once.

        Returns a list of (theme, [usernames]) for clusters of >= ASK_CLUSTER_MIN_USERS
        distinct chatters — so she can answer all of them in one stroke
        ("chat's asking about my team —") instead of repeating herself N times.
        batch entries are the bot's message dicts ({"username","message",...}).
        """
        # Map theme-word -> set(usernames who used it). Distinct-user counting so
        # one person spamming a word doesn't fake a cluster.
        word_users: dict = {}
        for msg in batch or []:
            u = msg.get("username", "unknown")
            seen = set()
            for w in _WORD_RE.findall((msg.get("message") or "").lower()):
                if w in _STOP or w in seen:
                    continue
                seen.add(w)
                word_users.setdefault(w, set()).add(u)
        clusters = [(w, sorted(us)) for w, us in word_users.items()
                    if len(us) >= ASK_CLUSTER_MIN_USERS]
        # Biggest crowd first; cap to a few so the directive stays tight.
        clusters.sort(key=lambda x: len(x[1]), reverse=True)
        return clusters[:3]

    def asks_directive(self, batch: list) -> str:
        """Prompt directive for the most-asked consolidation, or "" if no cluster."""
        clusters = self.cluster_asks(batch)
        if not clusters:
            return ""
        parts = []
        for w, us in clusters:
            parts.append(f"\"{w}\" ({len(us)} chatters: {', '.join(us[:4])}"
                         f"{'…' if len(us) > 4 else ''})")
        return (
            "[MOST-ASKED — multiple chatters are converging on the same thing. "
            "Answer it ONCE, to all of them together (e.g. \"chat's all asking about "
            "X —\"), instead of replying to each separately]\n- "
            + "\n- ".join(parts) + "\n"
        )
