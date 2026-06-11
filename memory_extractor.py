# memory_extractor.py — Claude-powered fact extraction with strict JSON schema
import asyncio
import json
import re

from config import CLAUDE_HAIKU_MODEL

# Haiku matches Sonnet on real facts but is slightly looser on ambiguous turns
# (it occasionally fabricates a detail, e.g. inventing "at work"). Raising the
# confidence floor from 0.7 → 0.9 for the Haiku path trims those false positives
# while keeping every genuine memory. False memories are worse than missed ones.
HAIKU_CONFIDENCE_FLOOR = 0.9

EXTRACTOR_SYSTEM = """You are a memory archivist for an AI companion named Kira. Your job is to extract DURABLE facts about her friend Jonny from conversation — facts that will still matter in a month.

THE BAR IS HIGH. Most conversation turns produce NO memories. Reject:
- Temporary states ("I'm tired", "I'm in the lab")
- Questions Jonny asks (questions are not facts about him)
- Trivial reactions ("that's cool", "thanks")
- Filler ("okay", "yeah", "hmm")
- Things Kira said about Jonny (only extract from what Jonny said about himself)
- Vague generalizations ("Jonny is interested in things")

ONLY save:
- Concrete preferences he explicitly states ("my favorite anime is X")
- Life facts he states ("I work in finance", "I'm applying to grad school in Dublin")
- Strong opinions he expresses ("I think Kurisu is the best character because...")
- Goals or plans he commits to ("I'm going to play through all of Clannad")
- Significant emotional moments ("I cried at that scene")

Output STRICT JSON. No prose before or after. Schema:
{
  "memories": [
    {
      "subject": "Jonny",
      "category": "preference | life_fact | opinion | plan | emotional_moment",
      "fact": "natural sentence stating the fact in clean English, no underscores or snake_case",
      "confidence": 0.0-1.0
    }
  ]
}

If nothing meets the bar, output: {"memories": []}

Use natural English in the "fact" field. Never use snake_case or underscores. Never reference "the assistant" or "the AI" — only facts about Jonny himself.
"""


def _extract_json(text: str) -> dict | None:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None


async def extract_memories(ai_core, user_text: str, conversation_history: list) -> list[dict]:
    """Extracts durable Jonny-facts from a conversation turn using Claude Haiku
    (cheapest tier — this is the highest-frequency Claude call). Same prompt and
    confidence filter as before. Falls back to NO extraction if Claude is
    unavailable (better than garbage)."""

    if len(user_text) < 8:
        return []

    # Skip if Claude isn't available — local extraction produces too much garbage
    if not getattr(ai_core, "anthropic_client", None):
        return []

    context = "\n".join(
        f"{'Jonny' if m['role'] == 'user' else 'Kira'}: {m['content'][:300]}"
        for m in conversation_history[-3:]
    )

    user_msg = (
        f"Recent context:\n{context}\n\n"
        f"Jonny's latest line: \"{user_text}\"\n\n"
        f"Extract durable facts about Jonny only. Return JSON."
    )

    try:
        raw = await asyncio.wait_for(
            ai_core.claude_chat_inference(
                messages=[{"role": "user", "content": user_msg}],
                system_prompt=EXTRACTOR_SYSTEM,
                max_tokens=400,
                model_override=CLAUDE_HAIKU_MODEL,
            ),
            timeout=15,
        )
        if not raw:
            return []
        data = _extract_json(raw)
        if not data or "memories" not in data:
            return []

        valid = []
        for mem in data["memories"]:
            confidence = float(mem.get("confidence", 0))
            fact = mem.get("fact", "").strip()
            if confidence < HAIKU_CONFIDENCE_FLOOR:
                print(f"   [Memory] Rejected (conf {confidence:.2f} < {HAIKU_CONFIDENCE_FLOOR}): {fact or '(empty)'}")
                continue
            if not fact or len(fact) < 10 or "_" in fact:
                print(f"   [Memory] Rejected (malformed fact): {fact or '(empty)'}")
                continue

            valid.append({
                "subject": mem.get("subject", "Jonny"),
                "category": mem.get("category", "opinion"),
                "fact": fact,
                "confidence": confidence,
                "memory_type": "profile_fact" if mem.get("category") in ("life_fact", "preference") else "episodic",
                "predicate": mem.get("category", "fact"),
                "object": fact,
                "salience": confidence,
            })
            print(f"   [Memory] Extracted: {fact} (conf: {confidence})")

        return valid

    except asyncio.TimeoutError:
        print("   [WARN] memory_extractor LLM call timed out after 15s — skipping")
        return []
    except Exception as e:
        print(f"   [WARN] memory_extractor: extraction failed: {e}")
        return []
