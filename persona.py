# persona.py - Holds the system prompt defining the AI's personality.

import os

PERSONALITY_FILE = os.path.join(os.path.dirname(__file__), "kira_personality.txt")
with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    AI_PERSONALITY_PROMPT = f.read()