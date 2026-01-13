import os

def load_personality_txt(path: str = "personality.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. This file is the source of truth for Kira's personality.")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path} is empty. Fill it with Kira's personality prompt.")
    return text
