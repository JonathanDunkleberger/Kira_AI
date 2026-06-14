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
#
# Locked to the "Tale of the Three Brothers" (Deathly Hallows) shadow-puppet look,
# REGISTER #1 — the clean side-on tableau: the workhorse register. Crisp, sparse,
# readable silhouettes on a clear ground line. The atmospheric-depth register (#2)
# is a sparing big-moment variant added later, not the default.
STYLE_PREAMBLE = (
    "Shadow-puppet paper-theatre tableau in the style of the 'Tale of the Three "
    "Brothers' from Harry Potter and the Deathly Hallows. "
    # Composition / framing
    "Cinematic widescreen 2.39:1 aspect with solid black letterbox bars top and "
    "bottom. Clean side-on stage tableau: figures arranged on a clear horizontal "
    "ground line, sparse and readable, plenty of negative space. "
    # Figures
    "Flat solid-black silhouette figures and objects read ENTIRELY by shape, pose "
    "and outline, with NO interior detail. Figures look like articulated hand-cut "
    "paper marionettes — jointed paper cutouts with slightly rough hand-cut edges "
    "and a papery quality, not smooth digital vectors. "
    # Backlight / scrim
    "Backlit against a warm aged-parchment scrim in sepia, amber, ochre and cream; "
    "brightest at a central glowing sun-or-moon light source and vignetting to "
    "dark shadowed corners. Visible paper grain and parchment texture showing "
    "through the lit areas. "
    # Wispy organic elements
    "Wispy organic cut-paper elements catching the backlight where fitting — bare "
    "branches, drifting smoke, flocks of crows, fabric, floating ash and particles. "
    # Hard constraints
    "High contrast. No color inside the silhouettes. No text, no words, no letters, "
    "no signatures. One coherent backlit stage. Scene: "
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
