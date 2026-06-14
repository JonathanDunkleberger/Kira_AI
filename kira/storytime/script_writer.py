# script_writer.py — Claude script→beats generator for Storytime.
# ─────────────────────────────────────────────────────────────────────────────
# Turns a one-line theme into a short shadow-puppet story segmented into N beats,
# each with: narration (what Kira speaks) + image_prompt (the silhouette scene).
#
# The pipeline is GENERATE-THEN-PERFORM: this step runs in the pre-show, output
# is reviewed in the dashboard, and only then does the show perform. Output is
# strict JSON so the orchestrator can pre-gen one image per beat.
#
# CONTENT POLICY: the system prompt keeps stories tame (folk-tale / fable register,
# no gore/sexual/hateful content) so the downstream image API doesn't refuse beats.

from __future__ import annotations

import json
import re

from kira.config import ANTHROPIC_API_KEY, CLAUDE_CHAT_MODEL


# Locked visual identity for EVERY beat. Prepended to each beat's image_prompt by
# the orchestrator (not by the model) so the style can't drift scene-to-scene.
STYLE_PREAMBLE = (
    "Inky black paper-cut shadow-puppet silhouette theatre, in the style of the "
    "Deathly Hallows 'Tale of the Three Brothers' animation. Flat solid-black "
    "cut-paper figures with crisp clean edges, backlit against a warm parchment / "
    "amber-glow background, soft vignette, subtle paper grain. High contrast, no "
    "color inside the silhouettes, no text, no words, no letters. Single coherent "
    "stage with a theatrical hand-shadow aesthetic. Scene: "
)

_SYSTEM = (
    "You are Kira, an AI VTuber, writing a SHORT shadow-puppet story to perform on "
    "stream. It is a gentle folk-tale / fable in your dry, warm voice. Keep it "
    "wholesome and tame: no gore, no sexual content, no hate, no real people, "
    "nothing a family audience couldn't watch. The visuals are black paper-cut "
    "silhouettes, so describe SHAPES and SILHOUETTES, never colors or fine facial "
    "detail. Output ONLY valid JSON, no markdown fence, no commentary."
)


def build_request(theme: str, n_beats: int) -> str:
    theme = (theme or "a quiet little fable").strip()
    return (
        f"Write a shadow-puppet story on this theme: \"{theme}\".\n\n"
        f"Segment it into exactly {n_beats} scene beats that flow in order and tell "
        f"one complete little story (beginning, turn, ending). For EACH beat give:\n"
        f"  - \"narration\": 2-4 sentences you will SPEAK aloud over the scene. In "
        f"character, dry and warm, no stage directions, no quotation marks.\n"
        f"  - \"image_prompt\": a vivid description of the SILHOUETTE scene for that "
        f"beat — the shapes, figures, and staging only (the art style is added "
        f"automatically, so do NOT mention color, lighting, or art style). One or "
        f"two sentences. Keep figures simple enough to read as black cut-paper.\n\n"
        f"Return JSON shaped exactly like:\n"
        f"{{\"title\": \"...\", \"beats\": [{{\"narration\": \"...\", "
        f"\"image_prompt\": \"...\"}}]}}\n"
        f"Exactly {n_beats} beats. Output ONLY the JSON object."
    )


def _strip_fence(text: str) -> str:
    """Remove an accidental ```json fence if the model adds one."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_script(raw: str) -> dict:
    """Parse the model's JSON into {title, beats:[{narration, image_prompt}]}.
    Tolerant: strips fences and falls back to the first {...} block."""
    t = _strip_fence(raw)
    try:
        data = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError("Script writer did not return JSON.")
        data = json.loads(m.group(0))

    title = str(data.get("title", "")).strip() or "Untitled"
    beats_in = data.get("beats", []) or []
    beats: list[dict] = []
    for b in beats_in:
        narration = str(b.get("narration", "")).strip()
        image_prompt = str(b.get("image_prompt", "")).strip()
        if narration or image_prompt:
            beats.append({"narration": narration, "image_prompt": image_prompt})
    if not beats:
        raise ValueError("Script writer returned no usable beats.")
    return {"title": title, "beats": beats}


async def generate_script(theme: str, n_beats: int = 16) -> dict:
    """Generate and parse a Storytime script. Returns
    {title, beats:[{narration, image_prompt}]}. Raises on failure."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is empty — cannot write the story.")
    n_beats = max(6, min(24, int(n_beats)))
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=CLAUDE_CHAT_MODEL,
        max_tokens=2200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": build_request(theme, n_beats)}],
    )
    raw = resp.content[0].text if (resp.content and len(resp.content) > 0) else ""
    return parse_script(raw)
