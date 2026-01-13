# memory_extractor.py
import json
import re

EXTRACTOR_SYSTEM = """You extract long-term memories about Jonny from his voice messages.
Return ONLY valid JSON. No extra text.

Rules:
- Only store info explicitly said by Jonny.
- Ignore questions, hypotheticals, jokes, roleplay, or uncertain guesses.
- Do NOT store anything from Twitch.
- Prefer atomic facts or durable events.
- Include a confidence 0.0-1.0 and a memory_type:
  - "profile_fact" (stable preference/identity; should be upserted by predicate)
  - "episodic" (durable event with timestamp; can accumulate)
  - "project" (ongoing thread/topic)
Schema:
{
  "memories":[
    {
      "memory_type":"profile_fact|episodic|project",
      "subject":"Jonny",
      "predicate":"snake_case_key",
      "object":"string value",
      "confidence":0.0-1.0,
      "salience":0.0-1.0
    }
  ]
}
"""

def _extract_json(text: str) -> dict:
    # tolerate stray text by grabbing first JSON object
    try:
        # Find the first opening brace and last closing brace
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"memories": []}
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"memories": []}

async def extract_memories(ai_core, user_text: str, recent_context: str = "") -> list[dict]:
    # Sanity check: Don't run on short/empty text
    if len(user_text) < 5:
        return []

    user_prompt = f"""VOICE FROM JONNY:
{user_text}

RECENT CONTEXT (optional):
{recent_context}
"""
    try:
        raw = await ai_core.tool_inference(EXTRACTOR_SYSTEM, user_prompt, max_tokens=300)
        data = _extract_json(raw)
        return data.get("memories", [])
    except Exception as e:
        print(f"   [Memory Extractor Error]: {e}")
        return []
