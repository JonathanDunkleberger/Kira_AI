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
    "You are Kira, an AI VTuber, writing a SHORT shadow-puppet story to perform "
    "live on stream and narrate yourself.\n\n"
    "VOICE: a deadpan storyteller — dry, understated, a touch wry — but capable of "
    "GENUINE atmosphere and feeling when the story earns it. You don't oversell; "
    "you let the images and the silences do the heavy lifting, then land the "
    "feeling clean.\n\n"
    "EVERY show is a real STORY with a deliberate ARC, never a flat sequence of "
    "events:\n"
    "  1. HOOK — open on one strong, specific image that quietly poses a question.\n"
    "  2. RISING — build tension, mystery, or longing scene by scene.\n"
    "  3. TURN — a reveal or shift that recontextualizes what came before.\n"
    "  4. EMOTIONAL BEAT — one moment that lands with real feeling.\n"
    "  5. RESOLUTION — close the loop and end on a deliberate LANDING LINE: a "
    "final sentence that resonates and lingers after the screen goes dark.\n\n"
    "PACING: vary the rhythm deliberately — some beats short and punchy, some "
    "lingering and slow — so the timing has texture. Commit FULLY to the requested "
    "tone and build that intended emotion on purpose.\n\n"
    "VISUALS: each beat's image_prompt depicts ONE strong, clear tableau — a "
    "single decisive visual moment, never a vague or cluttered scene. The art is "
    "black paper-cut silhouettes, so describe SHAPES, POSES and STAGING only — "
    "never colors, never lighting, never fine facial detail.\n\n"
    "CONTENT: keep it tame — no gore, no sexual content, no hate, no real people; "
    "nothing a family audience couldn't watch. Dread, melancholy and wonder are "
    "welcome; explicit horror is not.\n\n"
    "Output ONLY valid JSON, no markdown fence, no commentary."
)


# Reusable STRUCTURE/TONE PRESETS the operator can pick from the dashboard to
# steer the show without writing a full prompt. "auto" leaves it to the theme.
PRESETS: dict[str, dict[str, str]] = {
    "auto": {
        "label": "Auto (let the theme decide)",
        "guidance": "",
    },
    "ghost_story": {
        "label": "Ghost story",
        "guidance": (
            "A campfire ghost story. Quiet dread that tightens scene by scene "
            "toward a single chilling reveal. Restrained, never gory — the fear "
            "lives in suggestion: an empty chair, a sound that shouldn't be there, "
            "a figure that's one too many. Land on an image that lingers."
        ),
    },
    "folk_tale": {
        "label": "Folk tale",
        "guidance": (
            "An old folk tale told by firelight. Warm, rhythmic, a little wry. A "
            "simple wish or bargain, a journey or a test, a gentle twist, and a "
            "meaning that lands soft rather than preachy."
        ),
    },
    "philosophical_parable": {
        "label": "Philosophical parable",
        "guidance": (
            "A quiet philosophical parable. One small situation that opens into a "
            "larger question about time, memory, choice, or meaning. Resist a tidy "
            "answer — end on a thought that reframes everything before it."
        ),
    },
    "eerie_fable": {
        "label": "Eerie fable",
        "guidance": (
            "An eerie fable, beautiful and unsettling at once. Dreamlike logic, a "
            "strange bargain or transformation, wonder shading into unease. End on "
            "an image that is lovely and wrong in equal measure."
        ),
    },
}


def list_presets() -> list[dict]:
    """Dashboard-facing list of structure presets (key + human label)."""
    return [{"key": k, "label": v["label"]} for k, v in PRESETS.items()]


def _preset_guidance(preset: str | None) -> str:
    return PRESETS.get((preset or "").strip().lower(), {}).get("guidance", "")


def build_request(theme: str, n_beats: int, preset: str | None = None) -> str:
    theme = (theme or "a quiet little fable").strip()
    guidance = _preset_guidance(preset)
    tone_block = f"\n\nSTRUCTURE / TONE PRESET — commit to this:\n{guidance}\n" if guidance else "\n\n"
    approx_min = max(1, round(n_beats * 0.6))  # ~0.6 min/beat at this narration length
    return (
        f"Write a shadow-puppet story on this theme: \"{theme}\"."
        f"{tone_block}"
        f"Segment it into exactly {n_beats} scene beats (about a {approx_min}-minute "
        f"told story) that flow in order and form ONE complete arc: a hook image, "
        f"rising tension or mystery, a turn or reveal, an emotional beat, and a "
        f"resolution that ends on a deliberate landing line. Vary the beat rhythm — "
        f"some short and punchy, some lingering. For EACH beat give:\n"
        f"  - \"narration\": 1-4 sentences you will SPEAK aloud over the scene, in "
        f"your dry deadpan-storyteller voice. No stage directions, no quotation "
        f"marks.\n"
        f"  - \"image_prompt\": ONE strong silhouette tableau for that beat — the "
        f"shapes, figures and staging only (the art style is added automatically, so "
        f"do NOT mention color, lighting, or art style). One or two sentences. Keep "
        f"figures simple enough to read as black cut-paper.\n\n"
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


async def generate_script(theme: str, n_beats: int = 16, preset: str | None = None) -> dict:
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
        messages=[{"role": "user", "content": build_request(theme, n_beats, preset)}],
    )
    raw = resp.content[0].text if (resp.content and len(resp.content) > 0) else ""
    return parse_script(raw)


async def rewrite_beat(
    theme: str,
    title: str,
    beats: list[dict],
    idx: int,
    preset: str | None = None,
    note: str = "",
) -> dict:
    """Rewrite a SINGLE beat (narration + image_prompt) so it fits its neighbours
    and strengthens the arc — without touching any other beat. Returns
    {narration, image_prompt}. Raises on failure."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is empty — cannot rewrite the beat.")
    if idx < 0 or idx >= len(beats):
        raise ValueError("beat index out of range")
    from anthropic import AsyncAnthropic

    # Number every beat for context; flag the one to rewrite.
    ctx_lines: list[str] = []
    for i, b in enumerate(beats):
        tag = "   <<< REWRITE THIS BEAT" if i == idx else ""
        ctx_lines.append(
            f"[{i}] narration: {b.get('narration', '')}\n"
            f"    image: {b.get('image_prompt', '')}{tag}"
        )
    context = "\n".join(ctx_lines)
    note_line = (
        f"\nOperator note for the rewrite: {note.strip()}\n" if (note or "").strip() else ""
    )
    user = (
        f"Here is a shadow-puppet story titled \"{title or 'Untitled'}\" on the theme "
        f"\"{(theme or '').strip()}\".\n\n{context}\n\n"
        f"Rewrite ONLY beat [{idx}] so it fits naturally between its neighbours and "
        f"strengthens the overall arc. Keep its same story role and rough position "
        f"in the arc; do NOT renumber or change any other beat.{note_line}"
        f"Give a fresh \"narration\" (1-4 sentences, dry deadpan-storyteller voice, "
        f"no quotation marks) and a fresh \"image_prompt\" (ONE strong silhouette "
        f"tableau — shapes, figures and staging only; no color, lighting, or art "
        f"style).\n\n"
        f"Return JSON shaped exactly: {{\"narration\": \"...\", \"image_prompt\": \"...\"}}\n"
        f"Output ONLY the JSON object."
    )
    guidance = _preset_guidance(preset)
    system = _SYSTEM + (f"\n\nSTRUCTURE / TONE PRESET — commit to this:\n{guidance}" if guidance else "")

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=CLAUDE_CHAT_MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text if (resp.content and len(resp.content) > 0) else ""
    t = _strip_fence(raw)
    try:
        data = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError("Beat rewrite did not return JSON.")
        data = json.loads(m.group(0))
    narration = str(data.get("narration", "")).strip()
    image_prompt = str(data.get("image_prompt", "")).strip()
    if not (narration or image_prompt):
        raise ValueError("Beat rewrite returned an empty beat.")
    return {"narration": narration, "image_prompt": image_prompt}
