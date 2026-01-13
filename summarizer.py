# summarizer.py
from config import AI_NAME

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

        from persona import EmotionalState
        summary = await self.ai_core.llm_inference(
            messages=[{"role": "user", "content": self._get_summarization_prompt(transcript)}],
            current_emotion=EmotionalState.NEUTRAL
        )

        clean_summary = summary.strip().replace("Permanent Fact:", "").strip()
        
        # Filter out bad summaries
        if "NO_MEMORY" not in clean_summary and len(clean_summary) > 5 and len(clean_summary) < 200:
             # double check it doesn't contain "not found"
             if "not found" not in clean_summary.lower():
                print(f"   [Memory] Consolidating: {clean_summary}")
                self.memory_manager.add_summarized_memory(clean_summary)
        else:
            print("   [Memory] No new facts to store.")