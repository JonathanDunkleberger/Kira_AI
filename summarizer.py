# summarizer.py (Conceptual)

class SummarizationManager:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        # You would initialize your LLM connection here
        # self.llm = YourLLMInterface() 

    def get_summarization_prompt(self, transcript: str) -> str:
        """Creates the specialized prompt for the LLM to summarize a conversation."""
        
        # This prompt is key. It instructs the LLM on its specific, one-shot task.
        prompt = f"""
You are a Memory Consolidation AI. Below is a conversation transcript between 'User (Jonny)' and 'Assistant (Kira)'. 
Your task is to extract the single most important, lasting piece of information or memory from this exchange.
The memory should be a concise, third-person statement about Jonny's preferences, decisions, feelings, or key events.
If no significant new memory was formed, respond ONLY with the word 'NO_MEMORY'.

Examples of good memories:
- Jonny's favorite character in Baldur's Gate 3 is Karlach because of her infernal engine.
- Jonny feels that the story in a game is more important than the gameplay.
- Jonny decided to pursue the 'Whispering Mountain' quest line.

Conversation Transcript:
---
{transcript}
---

Single most important memory:"""
        return prompt

    def consolidate_and_store(self, conversation_history: list):
        """Takes a list of conversation turns, gets a summary, and stores it."""
        
        if not conversation_history:
            return

        # Format the history into a simple string transcript
        transcript = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        
        # Create the specialized prompt for this task
        summarization_prompt = self.get_summarization_prompt(transcript)
        
        # --- MAKE THE LLM CALL ---
        # summary = self.llm.generate(summarization_prompt) 
        summary = "Jonny's favorite character in Baldur's Gate 3 is Karlach because of her infernal engine." # Placeholder for actual LLM call

        if summary and summary.strip() != "NO_MEMORY":
            print(f"   Consolidated Memory: {summary}")
            # Use a new method in MemoryManager to store this special type of memory
            self.memory_manager.add_summarized_memory(summary)