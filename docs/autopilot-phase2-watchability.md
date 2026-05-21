# Autopilot Phase 2 — The Watchability Layer
**Status: Design** (build after Phase 1 mechanical loop is validated)

## Goal
Turn "Kira can mechanically navigate a VN" into "Kira can carry a 6-hour solo stream that is genuinely compelling — aware, invested, varied in energy, comfortable with dead chat, and growing in personality across the playthrough."

Phase 1 proves capability. Phase 2 adds watchability. They are different problems.

**The test:** Could a stranger watch 20 minutes of Kira solo-playing a kinetic VN with dead chat and not click away? Clear it and "Kira Plays" is a viable evergreen series. Miss it and she's a tech demo.

---

## The Core Problem

A mechanical autopilot reacting at a flat rate with flat energy for 6 hours becomes wallpaper by hour 2. A human streamer naturally varies energy, goes quiet during intense moments, rambles during slow ones, talks to absent chat, builds theories, gets attached. Phase 2 is the six systems that produce "alive, invested, varied, present" instead of "consistent reaction machine."

---

## System 1 — Dynamic Energy / Pacing
**Priority: FIRST. Biggest single watchability lever.**

**Problem:** Flat 1-in-5 reaction rate at flat energy = hour 4 feels identical to hour 1 = viewers leave.

**Solution:** Reaction rate AND energy scale with a running scene intensity estimate.

**Intensity states:** `calm` / `building` / `intense` / `climactic` / `aftermath`

Inferred from:
- Text content (dialogue tone, revelations, deaths, humor, connective tissue)
- Scene change pacing (rapid transitions vs. long-held single scene)
- System 5 narrative weight score (when available)
- System 6 audio mood (when available)

**Behavior per state:**

| State | Reaction Rate | Energy / Style |
|-------|--------------|----------------|
| `calm` / connective | High — ramble, theorize, fill space | Low stakes, tangents, talking to absent chat |
| `building` | Moderate | Anticipation, tense, leaning in |
| `intense` | Low — selective | React only if it adds; silence is valid |
| `climactic` | Near-zero — let it land | Reverential quiet OR one heavy reaction, never both |
| `aftermath` | Moderate — process | Digesting, feeling the weight, not immediately moving on |

**The Mayuri instinct systematized:** The silence during a death scene is more powerful than commentary. Kira already proved she knows this. Phase 2 makes it automatic.

**Prompt modification:** Reaction-decision prompt receives current intensity state:
> "Scene is climactic — react only if you have something that adds genuine weight. Otherwise stay silent and let the moment land."
> "Scene is slow/connective — good time to ramble, theorize, or check in with chat."

---

## System 2 — Solo / Dead-Chat Behavior
**Priority: HIGH (Jonny specifically flagged this)**

**Problem:** Chat will often be empty. Reaction-only Kira goes dead-silent between text boxes when there's no chat to bounce off.

**Solution:** Explicit solo-mode behavior when chat has been quiet for N minutes.

**Behaviors:**
- **Internal monologue** — thinking out loud about the story, not just reacting to boxes
- **Addressing absent chat** — "okay chat, I know you're lurking, watch this part" / "no one's here but I'm gonna say it anyway"
- **Self-directed tangents** — comfortable carrying a one-sided conversation; riffing, wondering aloud, narrating her own reactions
- **Void-talking** — the streamer skill of performing to an empty room without it feeling sad

**Detection:**
- Track `time_since_last_chat_message` and `active_chatter_count`
- When chat has been dead for N minutes → increase solo-monologue weighting
- When chat IS active → dial back monologue, engage chat instead

**Integration points:**
- `_decide_reaction()` in `vn_autopilot.py` receives a `chat_dead_for_min` param
- Solo-mode prompt variant for the reaction-decision call
- May also generate occasional unprompted "aside" reactions independent of dialogue boxes (System 1 routing: during `calm` stretches)

---

## System 3 — Theory-Building / Anticipation
**Priority: HIGH (the compelling lets-play factor)**

**Problem:** Reaction-only content is shallow. The best lets-play content is anticipation: "if that's true, then the earlier thing means..." Being wrong. Being right. Updating.

**Solution:** An active forward-reasoning layer on top of narrative memory.

**How it works:**
- Periodically (every scene or every N boxes — NOT every box), Kira forms explicit theories
- "Where is this going? What am I suspicious of? What do I predict?"
- Theories stored in session state: `active_theories: list[dict]` with status (`open` / `confirmed` / `busted`)
- When a theory resolves, she reacts to being right/wrong — some of the most engaging lets-play content

**Prompt layer:**
> "Given the story so far, voice a genuine theory or prediction if you have one. What are you suspicious of? Where do you think this is going?"
Fed in during `calm` stretches (System 1 routes to theory-building here).

**Cross-session persistence:**
- Active (unresolved) theories carry across sessions via Playthrough Memory
- "Last session I had a theory about X" → confirmation or busting lands with earned payoff

**Example (Clannad):**
- Hour 1: "Something feels off about Nagisa's health — they keep mentioning it"
- Hour 3: "Okay I called this but I hoped I was wrong"

---

## System 4 — Within-Session Emotional State (Earned Payoffs)
**Priority: MEDIUM-HIGH**

**Problem:** If hour-6 reaction to a character death lands the same as hour 1, there's no earned weight. "Personality growing" requires her relationship to the story to evolve.

**Solution:** A running emotional/investment state about the story itself, separate from the existing HAPPY/SASSY/MOODY mood engine.

**Tracks per-session:**
- `character_attachment: dict[str, float]` — attachment level per character (0.0–1.0), built from screen time, dialogue focus, emotional moments
- `story_investment: float` — overall investment in this particular story (increases as plot hooks land)
- `emotional_trajectory: str` — e.g., "started skeptical → got invested → now defensive of Nagisa"

**Behavior:**
- Reaction calibration weights by attachment: a minor character's death = flat; an hour-6 attached character's death = earned, heavy
- Investment level affects solo behavior (System 2): high investment → more engaged monologue, less tangent-wandering
- Trajectory surfaces naturally: "I know I keep defending her but I'm attached and I'm not going to pretend I'm not"

**Persistence:**
- At session end, attachment levels + trajectory stored in Playthrough Memory
- Carry across sessions: by hour 40 of Clannad, Nagisa-attachment is a known fact, recorded and loaded

**Note:** This is DISTINCT from `current_emotion` (general mood). This is specifically her relationship to the story and its characters.

---

## System 5 — Narrative Weight Detection
**Priority: MEDIUM (ties Systems 1 and 4 together)**

**Problem:** For reactions to land, she must recognize "this IS the moment" vs "this is connective tissue." Humans feel this through music swells, art changes, timing.

**Solution:** Combine signals into a `narrative_weight` score per moment.

**Input signals:**
| Signal | Source | Weight |
|--------|--------|--------|
| Full-screen CG / art change | Vision agent (Phase 1 already uses this) | High |
| Dramatic lighting / character close-up | Vision agent | Medium |
| Scene composition shift | Vision agent | Medium |
| Dialogue content (revelation, death, confession, climax) | Text classifier | High |
| Pacing: long-held single scene | Screen change rate | Medium |
| Music: swell, silence, shift | Audio agent (System 6) | High |

**Output:** `narrative_weight: float` (0.0–1.0) fed into:
- System 1 (intensity estimate — high weight pushes toward `climactic`)
- System 4 (emotional response calibration — high weight on an attached character = big reaction)

**Build order:** Stub with text-only detection first (text content is the most reliable signal anyway). Add visual and audio signals iteratively.

---

## System 6 — Audio Restoration + Soft-Pause

### 6a — Audio Understanding for Narrative Weight
**Priority: MEDIUM-LOW (highest-value audio signal but a sub-project)**

Music is the single best big-moment telegraph. A tense swell, a sudden silence, a hopeful theme — these signal narrative weight better than anything visual.

**Build path:**
- Local Whisper (already in models/) for any speech
- Light audio classification for music mood: `tense / calm / sad / hopeful / absent / dramatic`
- Even coarse mood detection massively improves System 5
- In solo autonomous mode this matters more than co-op (no human to verbally cue her)
- Referenced in earlier audio-fix discussion — separate sub-project

### 6b — Soft-Pause for Jonny Conversations
**Priority: MEDIUM-LOW (polish)**

**Problem:** Mid-autopilot, Jonny speaks to Kira. Currently she might try to read a text box and answer simultaneously.

**Solution:** Gentle "hold on, Jonny's talking" pause — distinct from the hard failsafe (choices).

**Behavior:**
- VAD detects Jonny speaking → autopilot enters `soft_pause` state
- Kira pauses advancing, has a real exchange
- N seconds of silence → auto-resume (no button required)
- Unlike failsafe: auto-resumes, no dashboard interaction needed
- Does not reset the current screen classification — resumes where it left off

**Integration:** New `soft_pause` / `soft_resume` methods on `VNAutopilot`, called from the VAD pipeline when voice is detected while autopilot is running.

---

## Build Order Within Phase 2

1. **System 1** (Dynamic Energy/Pacing) — biggest lever, build first
2. **System 2** (Solo/Dead-Chat) — high priority per Jonny's needs
3. **System 3** (Theory-Building) — the compelling-content factor
4. **System 4** (Within-Session Emotional State) — earned payoffs
5. **System 5** (Narrative Weight Detection) — ties 1 and 4 together
6. **System 6** (Audio + Soft-Pause) — polish + audio sub-project

Systems 1, 2, 3 get to "watchable." Systems 4, 5, 6 get to "genuinely compelling and growing."

---

## Integration Points (anticipated)

| System | Primary File | Key Changes |
|--------|-------------|-------------|
| 1 | `vn_autopilot.py` | `_estimate_intensity()`, intensity param on `_decide_reaction()` |
| 2 | `vn_autopilot.py` | `chat_dead_for_min` param, solo-mode prompt variant, unprompted aside generation |
| 3 | `vn_autopilot.py` | `active_theories` state, theory-prompt periodic call, resolution detection |
| 3 (persist) | `playthrough_memory.py` | Store/load open theories per game |
| 4 | `vn_autopilot.py` | `character_attachment` dict, `story_investment`, weighted reaction prompt |
| 4 (persist) | `playthrough_memory.py` | Attachment levels stored in session entry / loaded on game load |
| 5 | `vn_autopilot.py` | `_estimate_narrative_weight()`, feeds Systems 1 + 4 |
| 6a | Audio sub-project | New audio classifier, feeds System 5 |
| 6b | `vn_autopilot.py` | `soft_pause()`, `soft_resume()`, VAD integration in `bot.py` |

---

## Content Difficulty Note

Match game choice to how dialed-in Phase 2 is:

| Difficulty | Games | Why |
|-----------|-------|-----|
| **Easiest** | Planetarian, Narcissu | Kinetic, no choices, emotionally legible, short. Ideal for tuning. |
| **Mid** | Clannad | Sparse choices, long, big payoffs. Great series material once tuned. |
| **Hard** | Saya no Uta and similar | Disturbing/psychological. Reaction register is unforgiving. Save for when Phase 2 is dialed. Tone-deaf or cheesy reactions here are damaging. |

Do not debut hard-mode emotional content on an under-tuned reaction system.

---

## The Full Roadmap

| Phase | Doc | Goal | Status |
|-------|-----|------|--------|
| **Autopilot Phase 1** | `autopilot-phase1-spec.md` | Mechanical loop, prove capability | ✅ Implemented |
| **Playthrough Memory** | `playthrough-memory-design.md` | Persistent experience logs, global retrieval | ✅ Implemented (activates as playthroughs accumulate) |
| **Autopilot Phase 2** | This doc | Watchability layer, make it compelling | 🔲 Design complete, build pending |
