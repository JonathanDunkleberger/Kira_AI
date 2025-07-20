# persona.py - Loads the system prompt that defines the AI's personality.

import os

def load_personality_prompt():
    """Loads the personality prompt from the text file."""
    try:
        with open("kira_personality.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("FATAL: kira_personality.txt not found. Please ensure the file exists.")
        return "You are a helpful assistant." # Fallback prompt

AI_PERSONALITY_PROMPT = load_personality_prompt()