# identity_manager.py — Kira's identity & continuity system.
#
# Three responsibilities:
#   1. Entity anchors (Tier 1 + Tier 2): who is real, what are their attributes,
#      what is Kira's relationship to them. Prevents identity leaks.
#   2. Temporal self-continuity: when did Kira last talk with Jonny, what were
#      they doing. Makes her feel continuous rather than booting cold each session.
#   3. Source attribution: produces labelled dialogue-line prefixes so Kira always
#      knows whether input is Jonny's voice, a chat viewer, game dialogue, or ambient
#      audio — before the LLM sees it.
#
# Storage: memory_db/identity.json (flat JSON — not ChromaDB).
# Entity identity is a LOOKUP problem, not a similarity problem. No embeddings.
# Human-editable. Survives ChromaDB corruption.
#
# Design rules:
#   • All public methods are synchronous — zero async, zero latency on hot paths.
#   • load() is called once at startup; the in-memory dict is the source of truth.
#   • get_continuity_block() returns at most ~8 lines for prompt injection.
#   • Attributes marked "unknown" stay "unknown" — never inferred or guessed.

import json
import os
import time
from typing import Optional

_IDENTITY_PATH = os.path.join("memory_db", "identity.json")

# Singleton loaded at startup.
_identity: dict = {}


# ─── Load / Save ──────────────────────────────────────────────────────────────

def load() -> None:
    """Load identity.json into memory. Call once at bot startup."""
    global _identity
    if not os.path.exists(_identity_path()):
        print("   [Identity] identity.json not found — starting empty.")
        _identity = {"schema": 1, "permanent": {}, "activity_entities": {}, "sessions": []}
        return
    try:
        with open(_identity_path(), "r", encoding="utf-8") as f:
            _identity = json.load(f)
        p = len(_identity.get("permanent", {}))
        s = len(_identity.get("sessions", []))
        print(f"   [Identity] Loaded — {p} permanent anchors, {s} session records.")
    except Exception as e:
        print(f"   [Identity] Load failed: {e}. Starting empty.")
        _identity = {"schema": 1, "permanent": {}, "activity_entities": {}, "sessions": []}


def _save() -> None:
    """Persist current in-memory identity to disk. Called after mutations only."""
    try:
        with open(_identity_path(), "w", encoding="utf-8") as f:
            json.dump(_identity, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   [Identity] Save failed: {e}")


def _identity_path() -> str:
    return _IDENTITY_PATH


# ─── Session records (Piece 2 — temporal self-continuity) ─────────────────────

def record_session(start_ts: float, end_ts: float, activity: str, slug: str = "") -> None:
    """Append a completed session record. Call at stream_logger.finish() time."""
    sessions = _identity.setdefault("sessions", [])
    sessions.append({
        "start": int(start_ts),
        "end":   int(end_ts),
        "activity": activity,
        "slug": slug,
    })
    # Keep only the last 30 sessions — more than enough for temporal context.
    if len(sessions) > 30:
        _identity["sessions"] = sessions[-30:]
    _save()


def get_time_context() -> str:
    """Returns a natural-language phrase describing when the last session was
    and what was being played. At most one sentence. Empty string if no prior
    session exists (first ever session)."""
    sessions = _identity.get("sessions", [])
    if not sessions:
        return ""
    last = sessions[-1]
    end_ts = last.get("end", 0)
    if not end_ts:
        return ""
    elapsed_s = time.time() - end_ts
    if elapsed_s < 3600:
        phrase = "earlier today"
    elif elapsed_s < 7200:
        phrase = "a couple of hours ago"
    elif elapsed_s < 86400:
        phrase = f"about {int(elapsed_s // 3600)} hours ago"
    elif elapsed_s < 172800:
        phrase = "yesterday"
    elif elapsed_s < 604800:
        phrase = f"{int(elapsed_s // 86400)} days ago"
    else:
        phrase = f"{int(elapsed_s // 86400)} days ago"
    activity = last.get("activity", "").replace("_", " ")
    if activity and activity != "general":
        return f"Last session: {phrase}, playing {activity}."
    return f"Last session: {phrase}."


# ─── Entity anchors (Piece 1 — tiered entity memory) ──────────────────────────

def get_entity_anchors(activity_slug: str = "") -> str:
    """Returns a compact identity anchor block for the current activity.
    Injected into dynamic_context at the start of every response.
    Maximum ~8 lines total across Tier 1 + Tier 2.

    Format:
      [IDENTITY ANCHORS]
      Jonny — your creator and the person you talk with. Voice = Jonny, never a game character.
      009 — game character, male, status: dead. DO NOT confuse with Jonny.
      ...
    """
    lines = ["[IDENTITY ANCHORS]"]

    # Tier 1: permanent anchors — always included
    for name, entry in _identity.get("permanent", {}).items():
        role = entry.get("role", "")
        anchors = entry.get("anchors", [])
        never = entry.get("never_confuse_with", "")
        akas = entry.get("also_known_as", [])

        parts = [f"{name} — {role}"]
        if akas:
            parts.append(f"(also: {', '.join(akas[:3])})")
        if anchors:
            # Only include the most identity-critical anchor (first one)
            parts.append(anchors[0])
        if never:
            parts.append(f"NOT a {never}.")
        lines.append("  " + " | ".join(parts))

    # Tier 2: current activity entities (if any)
    if activity_slug:
        entities = _identity.get("activity_entities", {}).get(activity_slug, {})
        if entities:
            lines.append(f"  [Characters in {activity_slug.replace('_', ' ')}]")
            for name, entry in entities.items():
                gender = entry.get("gender", "unknown")
                role = entry.get("role", "")
                status = entry.get("status", "unknown")
                confidence = entry.get("source_confidence", "")

                # Build a compact one-liner. Skip "unknown" attributes to avoid
                # clutter — her charming vagueness on unknowns is intentional.
                attr_parts = []
                if gender and gender != "unknown":
                    attr_parts.append(gender)
                if role:
                    attr_parts.append(role)
                if status and status not in ("unknown", "active"):
                    attr_parts.append(f"status: {status}")

                display_name = name.replace("_", " ")
                if attr_parts:
                    lines.append(f"    {display_name}: {', '.join(attr_parts)}")
                else:
                    # Entity is real but all attributes are unknown — still anchor the name
                    lines.append(f"    {display_name}: game character (attributes not confirmed)")

    if len(lines) == 1:
        return ""  # No anchors at all — return empty, don't inject a useless header
    return "\n".join(lines)


def _canonical_key(name: str) -> str:
    """Return the canonical permanent-anchor key for a name/alias, lower-cased.
    Returns the name itself lowercased if not found in permanent anchors."""
    for key, entry in _identity.get("permanent", {}).items():
        if key.lower() == name.lower():
            return key.lower()
        akas = entry.get("also_known_as", [])
        if any(a.lower() == name.lower() for a in akas):
            return key.lower()
    return name.lower()


def get_entity(name: str, activity_slug: str = "") -> Optional[dict]:
    """Look up a single entity by name. Checks permanent first, then activity_entities.
    Returns None if not found."""
    # Permanent anchor lookup (case-insensitive)
    for key, entry in _identity.get("permanent", {}).items():
        if key.lower() == name.lower():
            return entry
        akas = entry.get("also_known_as", [])
        if any(a.lower() == name.lower() for a in akas):
            return entry

    # Activity entity lookup
    if activity_slug:
        entities = _identity.get("activity_entities", {}).get(activity_slug, {})
        for key, entry in entities.items():
            if key.lower().replace("_", " ") == name.lower().replace("_", " "):
                return entry
    return None


def resolve_alias(name: str) -> str:
    """Return the canonical display name for a given name or alias.

    If 'name' matches any permanent anchor (by key or also_known_as), returns
    the properly-cased canonical key so all downstream paths see a single
    consistent name regardless of which handle was used.

    Examples:
        resolve_alias("Militele3")  → "Jonny"
        resolve_alias("@Militele3") → "Jonny"
        resolve_alias("classiccoldfish") → "classiccoldfish"  (no anchor match)

    This is intentionally a pure lookup — no fuzzy matching, no embeddings.
    Identity is a lookup problem, not a similarity problem.
    """
    for key, entry in _identity.get("permanent", {}).items():
        if key.lower() == name.lower():
            return key
        akas = entry.get("also_known_as", [])
        if any(a.lower() == name.lower() for a in akas):
            return key
    return name


# ─── Continuity block (combined Piece 1 + Piece 2 injection) ──────────────────

def get_continuity_block(activity_slug: str = "") -> str:
    """Returns the full identity + temporal continuity block for prompt injection.
    This is the single injection point — one call in process_and_respond().
    Returns empty string if nothing useful to inject (first ever session, no anchors).

    Format injected into dynamic_context:
      [CONTINUITY] Last session: 2 days ago, playing 007 First Light.
      [IDENTITY ANCHORS]
        Jonny — your creator and the person you talk with | ...
        [Characters in 007 first light]
          Isola: antagonist-adjacent, betrayed Bond on a fjord boat
          ...
    """
    parts = []
    time_ctx = get_time_context()
    if time_ctx:
        parts.append(f"[CONTINUITY] {time_ctx}")
    anchors = get_entity_anchors(activity_slug)
    if anchors:
        parts.append(anchors)
    return "\n".join(parts)


# ─── Source attribution (Piece 3) ─────────────────────────────────────────────

def label_for_source(source: str, username: str = "", activity_slug: str = "") -> str:
    """Produce a dialogue-line prefix tag that tells Kira who is speaking.

    source values:
      "voice"         — Jonny speaking via microphone
      "chat"          — Twitch/YouTube viewer (username required)
      "game_dialogue" — character dialogue captured by vision/loopback
      "ambient_npc"   — background NPCs, set-dressing audio
      "system"        — internal system messages (not a real speaker)

    Returns a bracketed prefix string, e.g.:
      "[JONNY — your creator, speaking to you]"
      "[CHAT — classiccoldfish, viewer]"
      "[GAME DIALOGUE — character speech, not addressed to you]"
    """
    if source == "voice":
        # Voice is always Jonny — look up tier 1 label
        entry = get_entity("Jonny")
        if entry:
            role = entry.get("role", "your creator and the person you talk with")
            return f"[JONNY — {role}, speaking to you]"
        return "[JONNY — speaking to you]"

    if source == "chat":
        if not username:
            return "[CHAT — viewer]"
        # Check if this is a known regular (tier 1)
        entry = get_entity(username)
        if entry and entry.get("tier") == 1:
            role = entry.get("role", "viewer")
            # If the entry resolves to Jonny (same person as the mic voice),
            # make the label unambiguous so Kira never treats it as a stranger.
            canonical_key = _canonical_key(username)
            if canonical_key == "jonny":
                return f"[JONNY (via chat as {username}) — your creator, same person as the voice]"
            return f"[CHAT — {username}, {role}]"
        return f"[CHAT — {username}, viewer in chat]"
    if source == "game_dialogue":
        return "[GAME DIALOGUE — character speech, not addressed to you]"

    if source == "ambient_npc":
        return "[AMBIENT — background set-dressing, not addressed to you]"

    if source == "system":
        return "[SYSTEM]"

    # Fallback — unknown source
    return f"[{source.upper()}]"


# ─── Tier 2 write-back (called by playthrough_memory after session end) ────────

def upsert_activity_entity(
    activity_slug: str,
    name: str,
    gender: str = "unknown",
    role: str = "",
    status: str = "unknown",
    notes: str = "",
    source_confidence: str = "inferred",
) -> None:
    """Write or update a Tier 2 entity for a given activity.

    NEVER called with guessed attributes — callers must pass "unknown" for any
    attribute that isn't confirmed in the source material.

    Called by playthrough_memory.sync_entities_to_identity() after session end,
    not during the hot path.
    """
    entities = _identity.setdefault("activity_entities", {}).setdefault(activity_slug, {})

    # Normalise name to a stable key
    key = name.strip().replace(" ", "_")

    existing = entities.get(key, {})

    # Merge: "unknown" in new data never overwrites a known value from prior sessions.
    def _merge_attr(old_val: str, new_val: str) -> str:
        if not old_val or old_val == "unknown":
            return new_val
        if not new_val or new_val == "unknown":
            return old_val
        return new_val  # More recent info wins if both are known

    entities[key] = {
        "tier": 2,
        "gender": _merge_attr(existing.get("gender", "unknown"), gender),
        "role": _merge_attr(existing.get("role", ""), role) or existing.get("role", ""),
        "status": _merge_attr(existing.get("status", "unknown"), status),
        "notes": notes or existing.get("notes", ""),
        "source_confidence": source_confidence,
    }
    _save()
