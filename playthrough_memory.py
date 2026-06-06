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

import json
import os
import re
import time
from datetime import datetime


class PlaythroughMemory:
    PLAYTHROUGHS_DIR = "playthroughs"
    CHECKPOINT_DIR = "playthroughs/.checkpoints"
    SUMMARY_MARKER = "## Rolling Summary"
    OPINIONS_MARKER = "## Opinions & Evolving Takes"
    SESSION_LOG_MARKER = "## Session Log"
    SIGNATURE_MARKER = "## Signature Moments"
    REGEN_EVERY_N_SESSIONS = 3   # Regenerate rolling summary every N sessions (one Opus call each)
    MAX_SIGNATURE_MOMENTS = 10   # Hard cap; oldest drop off the end when exceeded

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

        # Parsed from file at load_for_game time
        self.recent_session_reactions: list[tuple[int, str]] = []  # [(session_num, reaction_text), ...]
        self.signature_moments: list[str] = []        # All-time best moments, capped at MAX_SIGNATURE_MOMENTS

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

        # Recover any in-RAM state from a prior session that crashed before writing
        self._try_recover_checkpoint()
        self._rebuild_manifest()

    def get_context_for_prompt(self) -> str:
        """Returns a compact, prompt-injectable block combining:
        - Games manifest (all games ever played — always included)
        - Current game rolling summary (if a game is active and has history)
        - Current opinions block (if present)
        - Last-2-session Kira's-reactions (experiential: "she was there")
        - All-time signature moments (iconic callbacks across the full playthrough)

        Designed to be injected into every AI response path (voice, chat, observer).
        Returns an empty string if there is nothing meaningful to inject.

        B-READY: get_recent_session_reactions() and get_signature_moments() are
        separate methods. Swap them for a ChromaDB query result when upgrading
        to Option B (selective retrieval) beyond ~10-12 sessions."""
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

        # Experiential recall: recent first-person reactions (last 2 sessions)
        recent_rx = self.get_recent_session_reactions(n_sessions=2)
        if recent_rx:
            parts.append(
                f"[WHAT I REMEMBER FROM BEING THERE — {self.current_display}]\n"
                f"{recent_rx}"
            )

        # All-time iconic moments (persists beyond rolling-summary lag)
        sig = self.get_signature_moments()
        if sig:
            parts.append(
                f"[MOMENTS FROM THIS PLAYTHROUGH I KEEP COMING BACK TO]\n"
                f"{sig}"
            )

        return "\n\n".join(parts)

    def get_recent_session_reactions(self, n_sessions: int = 2) -> str:
        """Return the 'Kira's reactions' text from the last n_sessions entries,
        formatted for context injection.

        B-READY: this method is the direct-injection implementation (Option A).
        When upgrading to Option B, replace the body with a ChromaDB query
        against a 'game_moments' collection filtered by slug, returning the
        top-k semantically relevant moments for the current query."""
        if not self.recent_session_reactions:
            return ""
        # recent_session_reactions is a list of (session_num, reaction_text) tuples,
        # sorted oldest-first. Take the last n_sessions.
        window = self.recent_session_reactions[-n_sessions:]
        lines = []
        for session_num, reaction_text in window:
            lines.append(f"Session {session_num}: {reaction_text}")
        return "\n".join(lines)

    def get_signature_moments(self) -> str:
        """Return the all-time signature moments as a bullet list for context injection.

        B-READY: same swap point as get_recent_session_reactions() — replace
        with a ChromaDB query when selective retrieval matters."""
        if not self.signature_moments:
            return ""
        return "\n".join(f"- {m}" for m in self.signature_moments)

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
        open_theories: list[dict] | None = None,        # autopilot active_theories (open only)
        character_attachment: dict[str, float] | None = None,  # char name → 0.0–1.0
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

        theories_block = ""
        if open_theories:
            theory_lines = "\n".join(f"  - {t['theory']}" for t in open_theories[:5])
            theories_block = (
                f"\n\nOPEN THEORIES Kira formed this session (still unresolved — "
                f"include in Opinion Shift if relevant):\n{theory_lines}"
            )

        attachment_block = ""
        if character_attachment:
            top = sorted(character_attachment.items(), key=lambda x: -x[1])[:6]
            top = [(c, v) for c, v in top if v >= 0.2]
            if top:
                char_lines = "\n".join(f"  - {c}: {v:.2f}" for c, v in top)
                attachment_block = (
                    f"\n\nCHARACTER ATTACHMENT LEVELS this session "
                    f"(0.0=stranger, 1.0=deeply attached; use to calibrate 'Kira's reactions' "
                    f"section — high attachment = heavier emotional weight):\n{char_lines}"
                )

        prompt = (
            f"You are writing a session log entry for Kira's autobiographical playthrough record.\n\n"
            f"Game: {activity}\n"
            f"Session number: #{session_num}\n"
            f"Date: {date_str}\n"
            f"Session duration: ~{session_duration_min} minutes"
            f"{existing_summary_block}"
            f"{story_block}"
            f"{reactions_block}"
            f"{chat_block}"
            f"{theories_block}"
            f"{attachment_block}\n\n"
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

        print(f"   [Playthrough] Generating session #{session_num} entry for '{activity}'... [I: ⚡ Sonnet — evaluate in-voice quality after next session]")
        try:
            entry_text = await self.ai_core.claude_inference(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are writing concise, specific, first-person session notes for an AI "
                    "VTuber's autobiographical playthrough record. Write in a clean journalistic "
                    "style. Be precise and personal, not generic."
                ),
                max_tokens=500,
                use_sonnet=True,  # I: playthrough session entry — Sonnet
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
            # Checkpoint is no longer needed — clean session ended successfully
            self._delete_checkpoint()
        except Exception as e:
            print(f"   [Playthrough] File write failed: {e}")
            return False

        # ── Extract signature moments for this session (cheap Sonnet call) ───────────────
        try:
            await self._extract_and_store_signature_moments(path, entry_text, session_num)
        except Exception as e:
            print(f"   [Playthrough] Signature moment extraction failed: {e}")

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

    async def backfill_signature_moments(self, slug: str = "") -> bool:
        """One-time backfill: extract signature moments from ALL existing session entries
        in a playthrough file. Designed to run once on files created before per-session
        extraction was added. Safe to re-run — fully overwrites the Signature Moments section.

        Spread logic: per-session cap = min(2, max(1, MAX // n_sessions)) so early iconic
        sessions get equal representation as later ones. If total moments still exceed
        MAX_SIGNATURE_MOMENTS after collection, excess is trimmed from the END, meaning
        the oldest (most historically iconic) entries survive the cut."""
        target_slug = slug or self.current_slug
        if not target_slug:
            print("   [Playthrough] backfill_signature_moments: no slug.")
            return False

        path = self._game_path(target_slug)
        if not os.path.exists(path):
            print(f"   [Playthrough] backfill_signature_moments: file not found: {path}")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] backfill_signature_moments: read failed: {e}")
            return False

        session_headers = list(re.finditer(r"^### Session (\d+)", content, re.MULTILINE))
        if not session_headers:
            print(f"   [Playthrough] backfill_signature_moments: no sessions in {path}")
            return False

        n_sessions = len(session_headers)
        # min(2, ...) keeps the per-session ask consistent with the ongoing extraction cap.
        # For ≤5 sessions this is 2; for 6-10 it drops to 1 so we don't overflow the cap.
        per_session_cap = min(2, max(1, self.MAX_SIGNATURE_MOMENTS // n_sessions))
        print(
            f"   [Playthrough] backfill: '{target_slug}' — {n_sessions} session(s), "
            f"up to {per_session_cap} moment(s) each."
        )

        import json as _json
        all_moments: list[str] = []

        for i, m in enumerate(session_headers):
            session_num = int(m.group(1))
            start = m.start()
            end = session_headers[i + 1].start() if i + 1 < n_sessions else len(content)
            block = content[start:end]

            rx_match = re.search(
                r"\*\*Kira'?s reactions:\*\*\s*(.+?)(?=\n\n\*\*|\n### Session|\Z)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if not rx_match:
                print(f"   [Playthrough] backfill: session {session_num} — no reactions block, skipping.")
                continue
            reactions_text = rx_match.group(1).strip()
            if len(reactions_text) < 20:
                continue

            extraction_prompt = (
                f"From the session reactions below, pick exactly 0 to {per_session_cap} single "
                f"sentence(s) that are the sharpest, most specific, most memorable — the kind of "
                f"reaction worth reading on stream 10 sessions from now. Prefer concrete images "
                f"over general feelings. Return a JSON array of strings only, no preamble. "
                f"If nothing qualifies, return [].\n\n"
                f"Reactions:\n{reactions_text}"
            )

            print(f"   [Playthrough] backfill: extracting session {session_num}...")
            try:
                raw = await self.ai_core.claude_chat_inference(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    system_prompt=(
                        "You are a ruthless editor extracting only the most vivid, specific "
                        "first-person reaction sentences. Output a JSON array of strings only."
                    ),
                    max_tokens=150,
                )
            except Exception as e:
                print(f"   [Playthrough] backfill: session {session_num} extraction failed: {e}")
                continue

            if not raw:
                print(f"   [Playthrough] backfill: session {session_num} — empty response.")
                continue

            try:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[^\n]*\n?", "", raw).rstrip("`").strip()
                moments: list[str] = _json.loads(raw)
                if not isinstance(moments, list):
                    continue
                moments = [
                    f"Session {session_num}: {s.strip()}"
                    for s in moments[:per_session_cap]
                    if isinstance(s, str) and len(s.strip()) > 15
                ]
                all_moments.extend(moments)
            except Exception as e:
                print(
                    f"   [Playthrough] backfill: session {session_num} JSON parse failed: "
                    f"{e} | raw: {raw[:80]}"
                )
                continue

        if not all_moments:
            print(f"   [Playthrough] backfill: no moments extracted — nothing to write.")
            return False

        # Trim to cap from the end — oldest sessions' moments are protected
        all_moments = all_moments[: self.MAX_SIGNATURE_MOMENTS]

        new_sig_block = "\n".join(f"- {m}" for m in all_moments)

        # Re-read in case any async call above somehow modified the file (defensive)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] backfill: re-read failed: {e}")
            return False

        if self.SIGNATURE_MARKER in content:
            before = content.split(self.SIGNATURE_MARKER, 1)[0]
            after_marker = content.split(self.SIGNATURE_MARKER, 1)[1]
            nxt = re.search(r"\n## ", after_marker)
            rest = after_marker[nxt.start():] if nxt else ""
            new_content = (
                before
                + self.SIGNATURE_MARKER + "\n\n"
                + new_sig_block + "\n"
                + rest
            )
        else:
            if self.OPINIONS_MARKER in content:
                new_content = content.replace(
                    self.OPINIONS_MARKER,
                    self.SIGNATURE_MARKER + "\n\n" + new_sig_block + "\n\n" + self.OPINIONS_MARKER,
                    1,
                )
            else:
                new_content = content + f"\n\n{self.SIGNATURE_MARKER}\n\n{new_sig_block}\n"

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            if target_slug == self.current_slug:
                self.signature_moments = all_moments
            print(
                f"   [Playthrough] backfill complete: {len(all_moments)} signature moments "
                f"from {n_sessions} session(s) → {path}"
            )
            return True
        except Exception as e:
            print(f"   [Playthrough] backfill: write failed: {e}")
            return False

    # ── Crash-recovery checkpoint ──────────────────────────────────────────────

    def flush_checkpoint(self, activity: str = "", session_start_time: float = 0.0) -> bool:
        """Write current session accumulators to a crash-recovery checkpoint file.
        Sync and safe to call directly from an asyncio context for small payloads.
        Returns True if something was written, False if there was nothing to save."""
        if not self.current_slug:
            return False
        if not self.session_reactions and not self.session_chat_moments:
            return False

        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        cp_path = os.path.join(self.CHECKPOINT_DIR, f"{self.current_slug}.json")
        cp_data = {
            "slug": self.current_slug,
            "display": self.current_display,
            "activity": activity or self.current_display,
            # session_count is the number of COMPLETED sessions already on disk.
            # The in-progress session would be session_count + 1.
            "expected_session_number": self.session_count + 1,
            "last_checkpoint_time": time.time(),
            "session_start_time": session_start_time,
            "closed_cleanly": False,
            "session_reactions": list(self.session_reactions),
            "session_chat_moments": list(self.session_chat_moments),
        }
        tmp = cp_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cp_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, cp_path)  # atomic overwrite on NTFS
            print(
                f"   [Checkpoint] Flushed {len(self.session_reactions)} reactions, "
                f"{len(self.session_chat_moments)} chat moments → {cp_path}"
            )
            return True
        except Exception as e:
            print(f"   [Checkpoint] Write failed: {e}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False

    def _load_checkpoint(self) -> dict | None:
        """Read the checkpoint file for the current slug. Returns None if absent or unreadable."""
        if not self.current_slug:
            return None
        cp_path = os.path.join(self.CHECKPOINT_DIR, f"{self.current_slug}.json")
        if not os.path.exists(cp_path):
            return None
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"   [Checkpoint] Read failed: {e}")
            return None

    def _delete_checkpoint(self) -> None:
        """Remove the checkpoint file for the current slug (after a clean session end)."""
        if not self.current_slug:
            return
        cp_path = os.path.join(self.CHECKPOINT_DIR, f"{self.current_slug}.json")
        try:
            if os.path.exists(cp_path):
                os.unlink(cp_path)
        except Exception as e:
            print(f"   [Checkpoint] Delete failed: {e}")

    def _try_recover_checkpoint(self) -> None:
        """Check for a crash checkpoint after loading a game. If the checkpoint is
        valid (not cleanly closed, for the expected next session), restore its
        session_reactions and session_chat_moments so they feed into this session's
        append_session_entry as if the session continued."""
        cp = self._load_checkpoint()
        if cp is None:
            return

        if cp.get("closed_cleanly", True):
            self._delete_checkpoint()
            return

        expected = cp.get("expected_session_number", -1)
        if expected != self.session_count + 1:
            # Checkpoint is for a different session number — stale, discard
            print(
                f"   [Checkpoint] Stale checkpoint for '{self.current_display}' "
                f"(checkpoint expected session {expected}, file has {self.session_count} sessions) "
                f"— discarding."
            )
            self._delete_checkpoint()
            return

        recovered_reactions = cp.get("session_reactions", [])
        recovered_moments = cp.get("session_chat_moments", [])
        if not recovered_reactions and not recovered_moments:
            self._delete_checkpoint()
            return

        self.session_reactions = list(recovered_reactions)
        self.session_chat_moments = list(recovered_moments)
        age_min = max(0, int((time.time() - cp.get("last_checkpoint_time", time.time())) / 60))
        print(
            f"   [Playthrough] RECOVERED crash checkpoint for '{self.current_display}': "
            f"{len(self.session_reactions)} reactions, {len(self.session_chat_moments)} chat "
            f"moments (~{age_min}m old) — will be included in this session's entry."
        )

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
            f"{self.SIGNATURE_MARKER}\n\n"
            f"*(Signature moments are extracted after each session.)*\n\n"
            f"{self.OPINIONS_MARKER}\n\n"
            f"*(Opinions develop as the playthrough continues.)*\n\n"
            f"{self.SESSION_LOG_MARKER}\n"
        )

    def _parse_file(self, path: str):
        """Extract Rolling Summary, Opinions, Signature Moments, recent session
        reactions, and session count from an existing file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] Could not read {path}: {e}")
            self.current_summary = ""
            self.current_opinions = ""
            self.session_count = 0
            self.recent_session_reactions = []
            self.signature_moments = []
            return

        # Rolling Summary
        self.current_summary = self._extract_section(content, self.SUMMARY_MARKER)
        if self.current_summary.startswith("*(Rolling summary"):
            self.current_summary = ""  # Placeholder, not real content

        # Opinions
        self.current_opinions = self._extract_section(content, self.OPINIONS_MARKER)
        if self.current_opinions.startswith("*(Opinions develop"):
            self.current_opinions = ""  # Placeholder

        # Signature Moments
        sig_raw = self._extract_section(content, self.SIGNATURE_MARKER)
        if sig_raw.startswith("*(Signature"):
            self.signature_moments = []
        else:
            self.signature_moments = [
                line.lstrip("- ").strip()
                for line in sig_raw.splitlines()
                if line.strip() and not line.strip().startswith("*(")  
            ]

        # Session count
        self.session_count = len(re.findall(r"^### Session \d+", content, re.MULTILINE))

        # Recent session reactions: parse last 2 session entries' "Kira's reactions" subsection.
        # Stored as [(session_num, reaction_text)] oldest-first.
        self.recent_session_reactions = self._parse_recent_reactions(content, n=2)

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

    def _parse_recent_reactions(self, content: str, n: int = 2) -> list[tuple[int, str]]:
        """Extract the 'Kira's reactions' subsection from the last n session entries.
        Returns a list of (session_num, reaction_text) tuples, oldest-first."""
        # Find all session blocks: ### Session N — date (~dur min)
        session_headers = list(re.finditer(
            r"^### Session (\d+)", content, re.MULTILINE
        ))
        if not session_headers:
            return []
        # Take last n
        target_headers = session_headers[-n:]
        results: list[tuple[int, str]] = []
        for i, m in enumerate(target_headers):
            session_num = int(m.group(1))
            start = m.start()
            # Block ends at start of next session header (or end of file)
            end = target_headers[i + 1].start() if i + 1 < len(target_headers) else len(content)
            block = content[start:end]
            # Extract "Kira's reactions:" subsection within this block
            rx_match = re.search(
                r"\*\*Kira'?s reactions:\*\*\s*(.+?)(?=\n\n\*\*|\n### Session|\Z)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if rx_match:
                reaction_text = rx_match.group(1).strip()
                if reaction_text:
                    results.append((session_num, reaction_text))
        return results

    async def _extract_and_store_signature_moments(
        self, path: str, entry_text: str, session_num: int
    ) -> None:
        """Extract 0-2 standout sentences from this session's 'Kira's reactions' text
        via a cheap Claude Sonnet call, then prepend them to the Signature Moments
        section in the playthrough file (oldest drop off the end at MAX cap).

        B-READY: the extracted strings are also the natural unit to embed into a
        game_moments ChromaDB collection. When upgrading to Option B, add an
        .add() call here alongside (or instead of) the file write."""
        # Extract the reactions section from the freshly-written entry
        rx_match = re.search(
            r"\*\*Kira'?s reactions:\*\*\s*(.+?)(?=\n\n\*\*|\Z)",
            entry_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not rx_match:
            return
        reactions_text = rx_match.group(1).strip()
        if not reactions_text or len(reactions_text) < 20:
            return

        extraction_prompt = (
            f"From the session reactions below, pick 0-2 single sentences that are "
            f"the sharpest, most specific, most memorable — the kind of reaction worth "
            f"reading on stream 10 sessions from now. Prefer concrete images over "
            f"general feelings. Return a JSON array of strings only, no preamble. "
            f"If nothing qualifies, return [].\n\n"
            f"Reactions:\n{reactions_text}"
        )

        try:
            raw = await self.ai_core.claude_chat_inference(
                messages=[{"role": "user", "content": extraction_prompt}],
                system_prompt=(
                    "You are a ruthless editor extracting only the most vivid, specific "
                    "first-person reaction sentences. Output a JSON array of strings only."
                ),
                max_tokens=150,
            )
        except Exception as e:
            print(f"   [Playthrough] Signature extraction call failed: {e}")
            return

        if not raw:
            return

        # Parse JSON array
        import json as _json
        try:
            raw = raw.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[^\n]*\n?", "", raw).rstrip("`").strip()
            new_moments: list[str] = _json.loads(raw)
            if not isinstance(new_moments, list):
                return
            new_moments = [
                f"Session {session_num}: {s.strip()}"
                for s in new_moments
                if isinstance(s, str) and len(s.strip()) > 15
            ]
        except Exception as e:
            print(f"   [Playthrough] Signature extraction JSON parse failed: {e} | raw: {raw[:100]}")
            return

        if not new_moments:
            return

        # Prepend new moments to the file's Signature Moments section
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"   [Playthrough] Signature read-back failed: {e}")
            return

        # Rebuild the moments list: new ones first, then existing, capped at MAX
        existing_sig_raw = self._extract_section(content, self.SIGNATURE_MARKER)
        existing = [
            line.lstrip("- ").strip()
            for line in existing_sig_raw.splitlines()
            if line.strip() and not line.strip().startswith("*(") 
        ]
        combined = new_moments + existing
        combined = combined[: self.MAX_SIGNATURE_MOMENTS]

        new_sig_block = "\n".join(f"- {m}" for m in combined)
        if self.SIGNATURE_MARKER in content:
            before = content.split(self.SIGNATURE_MARKER, 1)[0]
            after_marker = content.split(self.SIGNATURE_MARKER, 1)[1]
            nxt = re.search(r"\n## ", after_marker)
            rest = after_marker[nxt.start():] if nxt else ""
            new_content = (
                before
                + self.SIGNATURE_MARKER + "\n\n"
                + new_sig_block + "\n"
                + rest
            )
        else:
            # Section missing (old file) — inject before ## Opinions
            if self.OPINIONS_MARKER in content:
                new_content = content.replace(
                    self.OPINIONS_MARKER,
                    self.SIGNATURE_MARKER + "\n\n" + new_sig_block + "\n\n" + self.OPINIONS_MARKER,
                    1,
                )
            else:
                new_content = content + f"\n\n{self.SIGNATURE_MARKER}\n\n{new_sig_block}\n"

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            self.signature_moments = combined
            print(
                f"   [Playthrough] Signature moments updated: "
                f"{len(new_moments)} new, {len(combined)} total → {path}"
            )
        except Exception as e:
            print(f"   [Playthrough] Signature write-back failed: {e}")

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
                use_sonnet=True,  # J: rolling summary regen — Sonnet
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
