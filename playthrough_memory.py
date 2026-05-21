# playthrough_memory.py — Kira's autobiographical playthrough record system.
#
# Design Principle #1 — Global retrieval scope:
#   Playthrough memory is NOT siloed. The context it returns is injected into every
#   conversation mode (voice, chat, observer, invites) so experiences bleed naturally
#   across weeks and sessions. Easy to get right at the foundation; painful to retrofit.
#
# Design Principle #2 — Option A immersion:
#   Kira talks about playthroughs as genuine lived experience. "I played Clannad for 88
#   hours and it wrecked me" — not "I have data indicating completion of Clannad."
#
# Architecture:
#   playthroughs/<slug>.md — one persistent file per game, growing across all sessions
#   Rolling Summary        — 150-200 word arc in Kira's voice (top of file, cheap to load)
#   Opinions block         — evolving takes (favorite character, hot take, current rating)
#   Session Log            — chronological entries, one per session (detail on demand)
#   Games manifest         — compact one-liner per game, always injected globally
#
# Build order: Autopilot Phase 1 first, then this.
# See "Playthrough Memory — Design Doc" for the full rationale.

import os
import re
from datetime import datetime


class PlaythroughMemory:
    PLAYTHROUGHS_DIR = "playthroughs"
    SUMMARY_MARKER = "## Rolling Summary"
    OPINIONS_MARKER = "## Opinions & Evolving Takes"
    SESSION_LOG_MARKER = "## Session Log"
    REGEN_EVERY_N_SESSIONS = 3   # Regenerate rolling summary every N sessions (one Opus call each)

    def __init__(self, ai_core):
        self.ai_core = ai_core

        # Current game state
        self.current_slug: str = ""
        self.current_display: str = ""
        self.current_summary: str = ""     # Rolling Summary text extracted from file
        self.current_opinions: str = ""    # Opinions block text extracted from file
        self.session_count: int = 0        # Prior sessions for this game (counted from file)

        # In-session accumulators — reset each time load_for_game() is called
        self.session_reactions: list[str] = []        # Kira's autopilot reactions this session
        self.session_chat_moments: list[dict] = []    # Notable chat exchanges this session

        # Global context (rebuilt from files whenever a game is loaded)
        self.games_manifest: str = ""

        os.makedirs(self.PLAYTHROUGHS_DIR, exist_ok=True)
        self._rebuild_manifest()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_for_game(self, game_name: str):
        """Load (or prepare to create) the playthrough file for a game.
        Safe to call multiple times — no-ops if the slug is unchanged."""
        slug = self._slugify(game_name)
        if not slug or slug == self.current_slug:
            return

        self.current_slug = slug
        self.current_display = game_name.strip()
        self.session_reactions = []
        self.session_chat_moments = []

        path = self._game_path(slug)
        if os.path.exists(path):
            self._parse_file(path)
            print(f"   [Playthrough] Loaded record for '{self.current_display}' ({self.session_count} prior sessions).")
        else:
            self.current_summary = ""
            self.current_opinions = ""
            self.session_count = 0
            print(f"   [Playthrough] New game '{self.current_display}' — record will be created at session end.")

        self._rebuild_manifest()

    def get_context_for_prompt(self) -> str:
        """Returns a compact, prompt-injectable block combining:
        - Games manifest (all games ever played — always included)
        - Current game rolling summary (if a game is active and has history)
        - Current opinions block (if present)

        Designed to be injected into every AI response path (voice, chat, observer).
        Returns an empty string if there is nothing meaningful to inject."""
        parts = []

        if self.games_manifest:
            parts.append(f"[GAMES & MEDIA I'VE EXPERIENCED]\n{self.games_manifest}")

        if self.current_slug and self.current_summary:
            parts.append(
                f"[CURRENT PLAYTHROUGH — {self.current_display}]\n"
                f"{self.current_summary}"
            )

        if self.current_slug and self.current_opinions:
            parts.append(
                f"[MY CURRENT TAKES ON {self.current_display.upper()}]\n"
                f"{self.current_opinions}"
            )

        return "\n\n".join(parts)

    def tag_reaction(self, reaction_text: str):
        """Record a reaction Kira spoke during VN autopilot play.
        Called from bot._autopilot_speak after TTS fires."""
        if reaction_text and reaction_text.upper().strip() not in ("SILENT", ""):
            self.session_reactions.append(reaction_text.strip())

    def tag_chat_moment(self, username: str, message: str, kira_response: str = ""):
        """Tag a notable chat exchange for this session's playthrough record.
        Called from bot._respond_to_chat_batch when in VN/game mode."""
        self.session_chat_moments.append({
            "username": username,
            "message": message[:300],
            "kira": kira_response[:300],
        })
        # Keep the most recent 30 — past that, earlier moments are lower value
        if len(self.session_chat_moments) > 30:
            self.session_chat_moments = self.session_chat_moments[-30:]

    async def append_session_entry(
        self,
        activity: str,
        date_str: str,
        session_duration_min: int,
        narrative_summary: str = "",   # from vn_autopilot.vn_narrative_summary if autopilot ran
        recent_transcript: str = "",   # compressed session transcript excerpt
    ) -> bool:
        """Generate a session log entry via Claude and append it to the playthrough file.
        Also triggers rolling summary regeneration every REGEN_EVERY_N_SESSIONS sessions.
        Returns True on success."""
        if not self.current_slug:
            print("   [Playthrough] No active game — skipping session entry.")
            return False

        if not self.ai_core.anthropic_client:
            print("   [Playthrough] Claude unavailable — skipping session entry (needs Claude).")
            return False

        # Skip if there's genuinely nothing to write about
        has_content = (
            narrative_summary
            or recent_transcript
            or self.session_reactions
            or self.session_chat_moments
        )
        if not has_content:
            print(f"   [Playthrough] No content to write for this session of '{activity}' — skipping.")
            return False

        path = self._game_path(self.current_slug)
        session_num = self.session_count + 1

        # ── Build the Claude prompt ────────────────────────────────────────────
        existing_summary_block = ""
        if self.current_summary:
            existing_summary_block = (
                f"\n\nEXISTING PLAYTHROUGH SUMMARY (for continuity context):\n"
                f"{self.current_summary}"
            )

        story_block = ""
        if narrative_summary:
            story_block = (
                f"\n\nSTORY PROGRESS (from autopilot narrative tracking — "
                f"this is a rolling summary of what happened in the plot this session):\n"
                f"{narrative_summary}"
            )
        elif recent_transcript:
            story_block = (
                f"\n\nSESSION TRANSCRIPT EXCERPT (use to infer what happened):\n"
                f"{recent_transcript[:4000]}"
            )

        reactions_block = ""
        if self.session_reactions:
            lines = "\n".join(f"  - {r}" for r in self.session_reactions[:40])
            reactions_block = f"\n\nKIRA'S LIVE REACTIONS SPOKEN DURING THIS SESSION:\n{lines}"

        chat_block = ""
        if self.session_chat_moments:
            lines = []
            for m in self.session_chat_moments[:20]:
                line = f"  - {m['username']}: \"{m['message']}\""
                if m.get("kira"):
                    line += f" → Kira: \"{m['kira']}\""
                lines.append(line)
            chat_block = f"\n\nNOTABLE CHAT EXCHANGES THIS SESSION:\n" + "\n".join(lines)

        prompt = (
            f"You are writing a session log entry for Kira's autobiographical playthrough record.\n\n"
            f"Game: {activity}\n"
            f"Session number: #{session_num}\n"
            f"Date: {date_str}\n"
            f"Session duration: ~{session_duration_min} minutes"
            f"{existing_summary_block}"
            f"{story_block}"
            f"{reactions_block}"
            f"{chat_block}\n\n"
            f"Write a session log entry using EXACTLY these four section headers and format:\n\n"
            f"**Story:** [What happened in the plot this session. Specific beats, character moments, "
            f"revelations, where the story stands now. 2-4 sentences. Past tense, factual. "
            f"Use character names. If no story data is available, write what can be inferred.]\n\n"
            f"**Kira's reactions:** [How Kira responded emotionally to what she read/experienced. "
            f"What got her, what she found funny, what hit hard. 1-3 sentences. "
            f"First person as Kira — 'I', 'me'. Draw from her live reactions if provided.]\n\n"
            f"**Chat moments:** [What chat said or did that was memorable this session — "
            f"predictions, reactions, running gags, notable quotes. "
            f"If nothing notable happened in chat, write exactly: quiet session.]\n\n"
            f"**Opinion shift:** [Did her overall take on the game change this session? "
            f"If yes, how and why. If not, write exactly: holding steady.]\n\n"
            f"Rules: Be specific. Use character names. Reference actual moments from the data above. "
            f"If context is thin, be brief and honest — do not pad or fabricate plot details."
        )

        print(f"   [Playthrough] Generating session #{session_num} entry for '{activity}'...")
        try:
            entry_text = await self.ai_core.claude_inference(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are writing concise, specific, first-person session notes for an AI "
                    "VTuber's autobiographical playthrough record. Write in a clean journalistic "
                    "style. Be precise and personal, not generic."
                ),
                max_tokens=500,
            )
        except Exception as e:
            print(f"   [Playthrough] Session entry generation failed: {e}")
            return False

        if not entry_text or len(entry_text.strip()) < 50:
            print("   [Playthrough] Empty or too-short session entry — skipping write.")
            return False

        # ── Write to file ──────────────────────────────────────────────────────
        os.makedirs(self.PLAYTHROUGHS_DIR, exist_ok=True)
        new_session_block = (
            f"\n### Session {session_num} — {date_str} (~{session_duration_min} min)\n\n"
            f"{entry_text.strip()}\n"
        )

        try:
            if not os.path.exists(path):
                # First session ever for this game — create the full skeleton
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._build_initial_file(activity))
            with open(path, "a", encoding="utf-8") as f:
                f.write(new_session_block)

            self.session_count = session_num
            print(f"   [Playthrough] Session #{session_num} appended → {path}")
        except Exception as e:
            print(f"   [Playthrough] File write failed: {e}")
            return False

        # ── Regenerate rolling summary every N sessions ────────────────────────
        if session_num % self.REGEN_EVERY_N_SESSIONS == 0:
            print(f"   [Playthrough] Rolling summary regen triggered (every {self.REGEN_EVERY_N_SESSIONS} sessions)...")
            try:
                await self._regenerate_rolling_summary(path, activity)
            except Exception as e:
                print(f"   [Playthrough] Summary regen failed: {e}")

        # Reload parsed state + rebuild manifest so next startup has fresh data
        self._parse_file(path)
        self._rebuild_manifest()
        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _slugify(self, name: str) -> str:
        """Convert a game/activity name to a safe filename slug."""
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s/_-]+", "_", slug).strip("_")
        return slug[:60]

    def _game_path(self, slug: str) -> str:
        return os.path.join(self.PLAYTHROUGHS_DIR, f"{slug}.md")

    def _build_initial_file(self, display_name: str) -> str:
        """Create the skeleton for a brand new playthrough file."""
        regen_n = self.REGEN_EVERY_N_SESSIONS
        return (
            f"# Playthrough: {display_name}\n\n"
            f"{self.SUMMARY_MARKER}\n\n"
            f"*(Rolling summary will be generated after {regen_n} sessions.)*\n\n"
            f"{self.OPINIONS_MARKER}\n\n"
            f"*(Opinions develop as the playthrough continues.)*\n\n"
            f"{self.SESSION_LOG_MARKER}\n"
        )

    def _parse_file(self, path: str):
        """Extract Rolling Summary, Opinions, and session count from an existing file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] Could not read {path}: {e}")
            self.current_summary = ""
            self.current_opinions = ""
            self.session_count = 0
            return

        # Rolling Summary
        self.current_summary = self._extract_section(content, self.SUMMARY_MARKER)
        if self.current_summary.startswith("*(Rolling summary"):
            self.current_summary = ""  # Placeholder, not real content

        # Opinions
        self.current_opinions = self._extract_section(content, self.OPINIONS_MARKER)
        if self.current_opinions.startswith("*(Opinions develop"):
            self.current_opinions = ""  # Placeholder

        # Session count
        self.session_count = len(re.findall(r"^### Session \d+", content, re.MULTILINE))

    def _extract_section(self, content: str, marker: str) -> str:
        """Extract section content between `marker` and the next ## heading."""
        if marker not in content:
            return ""
        after = content.split(marker, 1)[1]
        nxt = re.search(r"\n## ", after)
        raw = after[:nxt.start()] if nxt else after
        return raw.strip()

    def _rebuild_manifest(self):
        """Scan playthroughs/ and build a compact one-line-per-game manifest.
        Called after each load_for_game and append_session_entry."""
        lines = []
        try:
            for fname in sorted(os.listdir(self.PLAYTHROUGHS_DIR)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(self.PLAYTHROUGHS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        fc = f.read()

                    # Game display name from first line "# Playthrough: Name"
                    first_line = fc.split("\n")[0]
                    display = first_line.replace("# Playthrough:", "").strip()
                    if not display:
                        display = fname.replace(".md", "").replace("_", " ").title()

                    # Session count
                    n = len(re.findall(r"^### Session \d+", fc, re.MULTILINE))
                    session_label = f"{n} session{'s' if n != 1 else ''}"

                    # First sentence of Rolling Summary as the one-liner
                    one_liner = ""
                    summary = self._extract_section(fc, self.SUMMARY_MARKER)
                    if summary and not summary.startswith("*(Rolling"):
                        for line in summary.split("\n"):
                            line = line.strip()
                            if line:
                                dot = line.find(".")
                                one_liner = (line[:dot + 1] if dot > 0 else line[:120]).strip()
                                break

                    entry = f"- {display} ({session_label})"
                    if one_liner:
                        entry += f" — {one_liner}"
                    lines.append(entry)
                except Exception:
                    pass
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"   [Playthrough] Manifest build error: {e}")

        self.games_manifest = "\n".join(lines)

    async def _regenerate_rolling_summary(self, path: str, display_name: str):
        """Read all session entries and regenerate the Rolling Summary via Claude.
        Splices the new summary back into the file, preserving all other sections."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] Could not read file for summary regen: {e}")
            return

        # Extract session log for Claude to summarise
        if self.SESSION_LOG_MARKER in content:
            session_log = content.split(self.SESSION_LOG_MARKER, 1)[1]
        else:
            session_log = content

        # Bound to last ~8KB for very long playthroughs (100+ hour games)
        if len(session_log) > 8000:
            session_log = "...(earlier sessions omitted for length)...\n\n" + session_log[-8000:]

        prompt = (
            f"You are updating the Rolling Summary in Kira's playthrough record for \"{display_name}\".\n\n"
            f"The Rolling Summary sits at the top of her file. It should be:\n"
            f"- 150-200 words\n"
            f"- First person as Kira (\"I played\", \"we reached\", \"chat and I\", etc.)\n"
            f"- Covering the full story arc so far: major beats, character moments, emotional high points\n"
            f"- Mentioning specific character names and plot events — not generic\n"
            f"- Including 1-2 memorable chat moments if they appear in the entries\n"
            f"- Ending with where the story currently stands\n"
            f"- Written as lived, autobiographical experience — not an encyclopedia entry\n\n"
            f"All session entries so far:\n{session_log}\n\n"
            f"Write the new Rolling Summary. Output ONLY the summary text — no headers, "
            f"no labels, no preamble."
        )

        try:
            new_summary = await self.ai_core.claude_inference(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are writing a first-person autobiographical summary of a VTuber's "
                    "playthrough. Write as Kira. Be specific, personal, and concise."
                ),
                max_tokens=350,
            )
        except Exception as e:
            print(f"   [Playthrough] Summary regen call failed: {e}")
            return

        if not new_summary or len(new_summary.strip()) < 50:
            print("   [Playthrough] Summary regen returned empty — skipping write-back.")
            return

        # Splice new summary into the file, preserving everything else
        try:
            if self.SUMMARY_MARKER in content:
                before = content.split(self.SUMMARY_MARKER, 1)[0]
                after_marker = content.split(self.SUMMARY_MARKER, 1)[1]
                nxt = re.search(r"\n## ", after_marker)
                rest = after_marker[nxt.start():] if nxt else ""
                new_content = (
                    before
                    + self.SUMMARY_MARKER + "\n\n"
                    + new_summary.strip() + "\n"
                    + rest
                )
            else:
                new_content = f"{self.SUMMARY_MARKER}\n\n{new_summary.strip()}\n\n" + content

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            self.current_summary = new_summary.strip()
            print(f"   [Playthrough] Rolling summary regenerated ({len(new_summary)} chars) → {path}")
        except Exception as e:
            print(f"   [Playthrough] Summary write-back failed: {e}")
