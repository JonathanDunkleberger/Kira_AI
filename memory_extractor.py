# memory_extractor.py
import json
import re

# THE BIOGRAPHER PROTOCOL
# Instead of hardcoding topics, we define "Significance."
EXTRACTOR_SYSTEM = """
You are a Memory Archivist for a user named Jonny.
Your goal is to build a high-resolution mental model of who Jonny is.

THE "FOREVER TEST":
Before saving a memory, ask: "Will this information be relevant in 1 month?"
- If YES -> SAVE IT.
- If NO -> IGNORE IT.

WHAT TO SAVE (The "Long-Term Value" Criteria):
1. **Core Identity:** Who is he? (Values, fears, deeply held beliefs, personality quirks).
2. **Aspirations & Plans:** What does he want? (Future goals, "hopes," specific long-term plans).
3. **Preferences:** Specific tastes that don't change often (Media, food, hobbies).
4. **Context:** Facts that provide context to his life (Job, relationships, living situation).
5. **Canon Events:** Major life events that just happened.

WHAT TO IGNORE (Ephemera):
1. **Temporary States:** "I am tired," "I am driving," "I am coding right now."
2. **Trivial Plans:** "I'm going to get water," "I might play a game later."
3. **Conversational Filler:** Small talk, greetings, or reactions to the AI's jokes.
4. **Questions vs. Facts:** If Jonny asks "What is your favorite book?", do NOT save "Jonny doesn't know books." Only save facts he explicitly states about HIMSELF.

Response Format (JSON ONLY):
{
  "reasoning": "Explain WHY this passes the 'Forever Test'",
  "save_memory": true,
  "memories": [
    {
      "subject": "Jonny",
      "predicate": "verb_phrase", 
      "object": "noun_phrase",
      "confidence": 0.0-1.0
    }
  ]
}
"""

def _extract_json(text: str) -> dict:
    try:
        # Grep for the first JSON object (handles if model yaps before JSON)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return None
        return json.loads(match.group(0))
    except Exception:
        return None

async def extract_memories(ai_core, user_text: str, conversation_history: list) -> list[dict]:
    # 1. Heuristic Filter: Skip very short inputs (e.g., "Yeah", "Okay")
    if len(user_text) < 5: return []
    
    # 2. Contextual Prompting
    # We pass the last 2 turns so the AI understands pronouns (It, Him, That)
    context_str = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history[-2:]])
    
    prompt = f"""
    RECENT CONTEXT:
    {context_str}
    
    USER'S LATEST INPUT:
    "{user_text}"
    
    Apply the 'Forever Test'. Extract significant facts now.
    """

    try:
        # Lower temp = more consistent formatting
        raw = await ai_core.tool_inference(EXTRACTOR_SYSTEM, prompt, max_tokens=300)
        data = _extract_json(raw)
        
        if not data or not data.get("save_memory", False):
            return []
            
        valid_memories = []
        for mem in data.get("memories", []):
            # We trust the reasoning more now, so 0.75 is a good threshold
            if mem.get("confidence", 0) > 0.75:
                valid_memories.append(mem)
                print(f"   [Memory] Extracted: {mem['subject']} {mem['predicate']} {mem['object']}")
                print(f"            └─ Reason: {data.get('reasoning')}")
                
        return valid_memories

    except Exception as e:
        print(f"   [Memory Extractor Error]: {e}")
        return []
