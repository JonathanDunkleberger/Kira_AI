# summarizer.py - Handles conversation summarization.

class SummarizationManager:
    def __init__(self, ai_core, memory_manager):
        self.ai_core = ai_core
        self.memory_manager = memory_manager

    def _get_summarization_prompt(self, transcript: str) -> str:
        return (f"You are a Memory Consolidation AI. Below is a conversation transcript between 'Jonny' and 'Kira'. "
                f"Your task is to extract the single most important, lasting piece of information or memory from this exchange. "
                f"The memory should be a concise, third-person statement about Jonny's preferences, decisions, or feelings. "
                f"**Be strictly factual based ONLY on the provided transcript. Do not add your own commentary or notes.** "
                f"If no significant new memory was formed, respond ONLY with the word 'NO_MEMORY'.\n\n"
                f"Examples of good memories:\n"
                f"- Jonny's favorite character in Baldur's Gate 3 is Karlach.\n"
                f"- Jonny feels that story is more important than gameplay.\n\n"
                f"Conversation Transcript:\n---\n{transcript}\n---\n\nSingle most important memory:")

    async def consolidate_and_store(self, conversation_history: list):
        if not conversation_history: return

        transcript = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history])
        prompt = self._get_summarization_prompt(transcript)
        
        from config import EmotionalState
        summary = await self.ai_core.llm_inference(
            messages=[{"role": "user", "content": prompt}],
            current_emotion=EmotionalState.HAPPY
        )

        if summary and "NO_MEMORY" not in summary:
            # Clean any potential notes from the summary
            summary = summary.split('(Note:')[0].strip()
            self.memory_manager.add_summarized_memory(summary)
        else:
            print("   No significant memory to consolidate from segment.")