import logging
import re
import sys
import os

# Configure logging
LOG_LEVEL = logging.INFO
if os.getenv("DEBUG_PROMPT_PREVIEW", "false").lower() == "true":
    LOG_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("KiraAI")

def sanitize_text(text: str) -> str:
    """Sanitizes sensitive info and truncates long blocks."""
    if not text: return ""
    
    # Redact specific phrases
    redact_patterns = [
        r"AZURE_SPEECH_KEY",
        r"TWITCH_OAUTH_TOKEN",
        r"sk-[a-zA-Z0-9]+", # OpenAI keys
        r"You are Kira.*", # System prompt start attempt
        r"Key Personality Traits",
    ]
    
    sanitized = text
    for pattern in redact_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE | re.DOTALL)
        
    # Truncate if too long (keep first 120 chars)
    if len(sanitized) > 400:
        sanitized = sanitized[:120] + "...(truncated)..."
        
    return sanitized

def sanitize_messages(messages: list) -> list:
    """Redacts system role content completely and sanitizes others."""
    clean_messages = []
    for msg in messages:
        clean_msg = msg.copy()
        if clean_msg.get("role") == "system":
            clean_msg["content"] = "[SYSTEM_PROMPT_REDACTED]"
        elif isinstance(clean_msg.get("content"), str):
            clean_msg["content"] = sanitize_text(clean_msg["content"])
        clean_messages.append(clean_msg)
    return clean_messages
