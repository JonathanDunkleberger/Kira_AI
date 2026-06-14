# show.py — Storytime / Puppet Show orchestrator.
# ─────────────────────────────────────────────────────────────────────────────
# GENERATE-THEN-PERFORM, review-gated (like the diary):
#   1. prepare(theme)   → Claude writes script → N beats; pre-gen one silhouette
#                         image per beat in a LOCKED shadow-puppet style. Beat 0
#                         is the style anchor; every later beat is conditioned on
#                         it (Nano-Banana reference image) for cross-scene
#                         consistency. Images land on disk, servable by the
#                         existing static mount. Status → "ready".
#   2. (REVIEW GATE)    → the dashboard shows beats + thumbnails; the operator
#                         can regenerate_beat(i) to re-roll any ugly scene.
#                         NOTHING performs unseen.
#   3. perform()        → push each scene to the puppet overlay, await Kira's TTS
#                         narration per beat. Beat-level sync (scene swaps land on
#                         narration boundaries).
#
# Isolation: this subsystem is fully additive. It only READS the TTS + overlay
# bus through injected callables at perform time; it never touches the brain,
# vision, audio, or any existing mode state.

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path

from .image_client import ImageGenError, ImageProvider, get_image_provider
from .script_writer import STYLE_PREAMBLE, generate_script, rewrite_beat

# Repo root → kira/storytime/show.py is two levels under the package root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Generated scenes live under the statically-mounted web_dashboard dir so the
# overlay (and dashboard previews) can load them by URL. Git-ignored.
_SCENES_ROOT = _REPO_ROOT / "web_dashboard" / "storytime"
_URL_BASE = "/web_dashboard/storytime"

# Beat lifecycle.
PENDING = "pending"
GENERATING = "generating"
DONE = "done"
ERROR = "error"

# Show lifecycle.
IDLE = "idle"
SCRIPTING = "scripting"
GEN = "generating"
READY = "ready"
PERFORMING = "performing"
DONE_SHOW = "done"
ERROR_SHOW = "error"

# Bounded concurrency for batch image gen — keeps us well under provider rate
# limits while still pre-generating ~16 scenes in a couple of minutes.
_GEN_CONCURRENCY = 3


class Beat:
    """One scene: narration line + its silhouette image."""

    __slots__ = ("idx", "narration", "image_prompt", "status", "error", "_mtime")

    def __init__(self, idx: int, narration: str, image_prompt: str):
        self.idx = idx
        self.narration = narration
        self.image_prompt = image_prompt
        self.status = PENDING
        self.error = ""
        self._mtime = 0.0

    def to_dict(self, show_id: str) -> dict:
        url = ""
        if self.status == DONE:
            # Cache-buster so a regenerated scene refreshes in dashboard/overlay.
            url = f"{_URL_BASE}/{show_id}/scene_{self.idx:02d}.png?v={int(self._mtime)}"
        return {
            "idx": self.idx,
            "narration": self.narration,
            "image_prompt": self.image_prompt,
            "status": self.status,
            "error": self.error,
            "url": url,
        }


class StorytimeShow:
    """Single-show orchestrator. One show in flight at a time."""

    def __init__(self, image_provider: ImageProvider | None = None):
        # Provider built lazily (keeps bot startup clean if SDK/key absent).
        self._provider = image_provider
        self.status: str = IDLE
        self.error: str = ""
        self.title: str = ""
        self.theme: str = ""
        self.preset: str = ""
        self.show_id: str = ""
        self.beats: list[Beat] = []
        self._dir: Path | None = None
        self._anchor_bytes: bytes | None = None
        self._busy: bool = False        # guards prepare/regenerate overlap
        self._stop: bool = False        # abort flag for perform()
        # Optional music bed. OFF by default: the current TTS playback path calls
        # pygame.mixer.stop()/music.stop() at the START of every spoken line, so a
        # pygame bed would be cut on each narration beat. Left as an honest seam —
        # a non-conflicting audio path can fill this later.
        self.music_path: str = ""

    # ── Provider ─────────────────────────────────────────────────────────────

    def _get_provider(self) -> ImageProvider:
        if self._provider is None:
            self._provider = get_image_provider()
        return self._provider

    # ── Status helpers ───────────────────────────────────────────────────────

    def _set_status(self, status: str, error: str = "") -> None:
        self.status = status
        self.error = error

    # ── Prepare (script + batch pre-gen) ─────────────────────────────────────

    async def prepare(self, theme: str, n_beats: int = 16, preset: str = "") -> None:
        """Write the script, then pre-generate every scene image. On completion
        status is READY (even if some beats errored — those can be regenerated).
        `preset` steers tone/structure (see script_writer.PRESETS).
        """
        if self._busy:
            return
        self._busy = True
        try:
            self.reset(keep_busy=True)
            self.theme = (theme or "").strip()
            self.preset = (preset or "").strip()
            self._set_status(SCRIPTING)
            try:
                script = await generate_script(
                    self.theme or "a quiet little fable", n_beats, self.preset or None
                )
            except Exception as e:
                self._set_status(ERROR_SHOW, f"script failed: {e}")
                return

            self.title = script["title"]
            self.show_id = self._make_show_id(self.title)
            self._dir = _SCENES_ROOT / self.show_id
            self._dir.mkdir(parents=True, exist_ok=True)
            self.beats = [
                Beat(i, b["narration"], b["image_prompt"])
                for i, b in enumerate(script["beats"])
            ]

            self._set_status(GEN)
            # Anchor beat (0) first, with NO reference, to establish the style +
            # character silhouettes. It becomes the reference for all later beats.
            await self._gen_one(0, reference=None)
            if self.beats and self.beats[0].status == DONE:
                self._anchor_bytes = self._read_scene_bytes(0)

            # Remaining beats, concurrently, each conditioned on the anchor.
            sem = asyncio.Semaphore(_GEN_CONCURRENCY)

            async def _worker(i: int):
                async with sem:
                    await self._gen_one(i, reference=self._anchor_bytes)

            await asyncio.gather(*(_worker(i) for i in range(1, len(self.beats))),
                                 return_exceptions=True)

            self._set_status(READY)
        finally:
            self._busy = False

    async def regenerate_beat(self, idx: int) -> None:
        """Re-roll a single beat's image (the review-gate re-roll). Uses the
        anchor as reference (beat 0 re-rolls without one and refreshes the anchor)."""
        if self._busy:
            return
        if idx < 0 or idx >= len(self.beats):
            return
        self._busy = True
        try:
            if idx == 0:
                await self._gen_one(0, reference=None)
                if self.beats[0].status == DONE:
                    self._anchor_bytes = self._read_scene_bytes(0)
            else:
                await self._gen_one(idx, reference=self._anchor_bytes)
        finally:
            self._busy = False

    def edit_beat_narration(self, idx: int, narration: str) -> bool:
        """Hand-edit a single beat's spoken line (review-gate text fix). Synchronous,
        leaves the image untouched. Returns True if applied."""
        if idx < 0 or idx >= len(self.beats):
            return False
        self.beats[idx].narration = (narration or "").strip()
        return True

    async def rewrite_beat_script(self, idx: int, note: str = "") -> None:
        """AI-rewrite ONE beat's script (narration + image_prompt) in the context of
        the surrounding story, then regenerate just that beat's image to match. The
        rest of the show is untouched — the single-beat analogue of an image re-roll."""
        if self._busy:
            return
        if idx < 0 or idx >= len(self.beats):
            return
        self._busy = True
        try:
            beats_ctx = [
                {"narration": b.narration, "image_prompt": b.image_prompt}
                for b in self.beats
            ]
            try:
                new = await rewrite_beat(
                    self.theme, self.title, beats_ctx, idx, self.preset or None, note
                )
            except Exception as e:
                self.beats[idx].error = f"rewrite failed: {e}"
                return
            beat = self.beats[idx]
            beat.narration = new.get("narration", beat.narration) or beat.narration
            beat.image_prompt = new.get("image_prompt", beat.image_prompt) or beat.image_prompt
            beat.error = ""
            # The image is now stale — regenerate it to match the new prompt.
            if idx == 0:
                await self._gen_one(0, reference=None)
                if self.beats[0].status == DONE:
                    self._anchor_bytes = self._read_scene_bytes(0)
            else:
                await self._gen_one(idx, reference=self._anchor_bytes)
        finally:
            self._busy = False

    async def _gen_one(self, idx: int, reference: bytes | None) -> None:
        beat = self.beats[idx]
        beat.status = GENERATING
        beat.error = ""
        prompt = STYLE_PREAMBLE + (beat.image_prompt or "an empty stage")
        try:
            provider = self._get_provider()
            png = await provider.generate(prompt, reference_image=reference)
        except ImageGenError as e:
            beat.status = ERROR
            beat.error = str(e)
            return
        except Exception as e:
            beat.status = ERROR
            beat.error = f"unexpected: {e}"
            return
        try:
            path = self._scene_path(idx)
            with open(path, "wb") as f:
                f.write(png)
            beat._mtime = time.time()
            beat.status = DONE
        except Exception as e:
            beat.status = ERROR
            beat.error = f"write failed: {e}"

    # ── Perform ──────────────────────────────────────────────────────────────

    async def perform(self, speak_coro, push_coro) -> None:
        """Run the show live. `speak_coro(text)` is an awaitable that blocks until
        a narration line finishes (ai_core.speak_text). `push_coro(event)` pushes
        an overlay event onto /ws/overlays. Beat-level sync: scene swaps land on
        narration boundaries.
        """
        if self.status not in (READY, DONE_SHOW):
            return
        playable = [b for b in self.beats if b.status == DONE]
        if not playable:
            self._set_status(ERROR_SHOW, "no generated scenes to perform")
            return

        self._stop = False
        self._set_status(PERFORMING)
        self._start_music()
        try:
            for beat in playable:
                if self._stop:
                    break
                try:
                    await push_coro({
                        "type": "scene_show",
                        "src": f"{_URL_BASE}/{self.show_id}/scene_{beat.idx:02d}.png?v={int(beat._mtime)}",
                        "idx": beat.idx,
                        "crossfade_ms": 700,
                    })
                except Exception:
                    pass  # overlay push is best-effort; never block narration
                if self._stop:
                    break
                try:
                    await speak_coro(beat.narration)
                except Exception as e:
                    print(f"   [Storytime] narration error on beat {beat.idx}: {e}")
        finally:
            try:
                await push_coro({"type": "scene_hide"})
            except Exception:
                pass
            self._stop_music()
            self._set_status(DONE_SHOW)

    def stop(self) -> None:
        """Abort an in-progress performance after the current narration line."""
        self._stop = True

    # ── Music bed (optional, off by default) ─────────────────────────────────

    def _start_music(self) -> None:
        # See note on self.music_path: the TTS path stops pygame audio per line,
        # so a continuous bed isn't viable through pygame.mixer today. No-op
        # unless a music_path is set AND a non-conflicting path is wired later.
        if not self.music_path:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(self.music_path)
            pygame.mixer.music.set_volume(0.18)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"   [Storytime] music bed skipped: {e}")

    def _stop_music(self) -> None:
        if not self.music_path:
            return
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass

    # ── Paths / IO ───────────────────────────────────────────────────────────

    def _scene_path(self, idx: int) -> Path:
        assert self._dir is not None
        return self._dir / f"scene_{idx:02d}.png"

    def _read_scene_bytes(self, idx: int) -> bytes | None:
        try:
            with open(self._scene_path(idx), "rb") as f:
                return f.read()
        except Exception:
            return None

    @staticmethod
    def _make_show_id(title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title or "show").strip("_").lower()[:32] or "show"
        return f"{time.strftime('%Y-%m-%d_%H-%M')}_{slug}"

    # ── State surface (dashboard) ────────────────────────────────────────────

    def reset(self, keep_busy: bool = False) -> None:
        self.status = IDLE
        self.error = ""
        self.title = ""
        self.theme = ""
        self.preset = ""
        self.show_id = ""
        self.beats = []
        self._dir = None
        self._anchor_bytes = None
        self._stop = False
        if not keep_busy:
            self._busy = False

    def snapshot(self) -> dict:
        done = sum(1 for b in self.beats if b.status == DONE)
        return {
            "status": self.status,
            "error": self.error,
            "title": self.title,
            "theme": self.theme,
            "preset": self.preset,
            "show_id": self.show_id,
            "busy": self._busy,
            "progress": f"{done}/{len(self.beats)}" if self.beats else "0/0",
            "beats": [b.to_dict(self.show_id) for b in self.beats],
        }
