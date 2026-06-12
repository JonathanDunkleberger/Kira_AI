# persona.py
# Aligned with personality.txt — these are the ONLY valid emotional states for Kira.
from enum import Enum

class EmotionalState(Enum):
    HAPPY       = "HAPPY"        # Default. Cheerful, curious, playful.
    MOODY       = "MOODY"        # Withdrawn, angsty, sarcastic, brief.
    SASSY       = "SASSY"        # Wit razor-sharp, maximum teasing.
    EMOTIONAL   = "EMOTIONAL"    # Open, earnest, genuinely heartfelt.
    HYPERACTIVE = "HYPERACTIVE"  # Excited, talkative, rambling.