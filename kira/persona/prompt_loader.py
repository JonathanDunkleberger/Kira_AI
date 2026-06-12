import os

_PRIVATE_PERSONA = os.path.join("persona", "private", "personality.txt")
_EXAMPLE_PERSONA = os.path.join("persona", "example", "personality.example.txt")


def load_persona() -> str:
    """Load personality prompt: private file if present, else example fallback."""
    for path in (_PRIVATE_PERSONA, _EXAMPLE_PERSONA):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                print(f"[persona] Loaded from: {path}")
                return text
    raise FileNotFoundError(
        f"No persona file found. Add your personality prompt to {_PRIVATE_PERSONA} "
        f"(private, gitignored) or edit {_EXAMPLE_PERSONA} as a starting point."
    )


def load_personality_txt(path: str = "personality.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. This file is the source of truth for Kira's personality.")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path} is empty. Fill it with Kira's personality prompt.")
    return text
