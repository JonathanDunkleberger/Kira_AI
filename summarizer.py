# summarizer.py
from config import AI_NAME

import re

class SummarizationManager:
    def __init__(self, ai_core, memory_manager):
        self.ai_core = ai_core
        self.memory_manager = memory_manager

    def _get_summarization_prompt(self, transcript: str) -> str:
        return (f"Analyze this interaction between 'Jonny' (Creator) and '{AI_NAME}' (AI). "
                f"Extract **PERMANENT FACTS** about Jonny to store in long-term memory.\n\n"
                f"[RULES]\n"
                f"1. **ATOMIC FACTS ONLY:** Format as: 'Jonny's [attribute] is [value].' (e.g., 'Jonny's favorite anime is Steins;Gate.').\n"
                f"2. **IGNORE:** Kira's confusion, meta-commentary, or questions.\n"
                f"3. **IGNORE:** Temporary states (e.g., 'Jonny is talking now').\n"
                f"4. If no *new* permanent fact is found, return ONLY: 'NO_MEMORY'.\n"
                f"5. **VERIFY NAMES:** The user is ALWAYS 'Jonny'. Do not record memories about other names unless Jonny explicitly talks about a third party.\n"
                f"6. **CONFIDENCE CHECK:** If a fact is vague, guessing, or based on a misheard word, return 'NO_MEMORY'.\n\n"
                f"[TRANSCRIPT]\n{transcript}\n\n"
                f"Permanent Fact:")

    async def consolidate_and_store(self, conversation_history: list):
        if not conversation_history: return

        transcript = ""
        for turn in conversation_history:
            role = turn['role']
            name = "Jonny" if role == "user" else AI_NAME
            transcript += f"{name}: {turn['content']}\n"

        TOOL_SYSTEM = "You extract durable user facts. Output ONLY atomic facts or NO_MEMORY."
        
        # Use specialized tool inference
        summary = await self.ai_core.tool_inference(TOOL_SYSTEM, self._get_summarization_prompt(transcript))

        clean_summary = summary.strip().replace("Permanent Fact:", "").strip()
        
        # Filter out bad summaries with Regex checks
        is_valid = True
        if "NO_MEMORY" in clean_summary: is_valid = False
        if len(clean_summary) < 10 or len(clean_summary) > 200: is_valid = False
        if not clean_summary.startswith("Jonny's"): is_valid = False
        if " is " not in clean_summary and " are " not in clean_summary: is_valid = False
        if "?" in clean_summary: is_valid = False
        
        if is_valid:
            print(f"   [Memory] Consolidating: {clean_summary}")
            self.memory_manager.add_fact(clean_summary)
        else:
            # print(f"   [Memory] Rejected summary: {clean_summary}") 
            pass