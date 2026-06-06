# Code Review / Cleanup Notes

Ongoing notes about technical debt, debug artifacts, and maintenance tasks.
Not urgent items — review periodically.

---

## Log Rotation & Debug Artifacts

### `logs/songid_debug_*.wav` — Song ID debug dumps

**What they are:** `identify_song()` in `music_tools.py` writes short WAV clips to `logs/` during song fingerprinting. These are debug artifacts — they were never intended to persist.

**What happened:** 29 WAV files (19.3 MB) accumulated from 2026-05-23 to 2026-05-28, manually archived to `logs/_archive/songid_debug/` on 2026-05-29.

**Recommended fix (not yet implemented):**
- Write to a temp location instead of `logs/` — e.g. `tempfile.mkstemp()` or a dedicated `logs/_debug/` folder
- OR add a session-end cleanup hook: `glob("logs/songid_debug_*.wav")` → delete on bot shutdown
- OR auto-delete after 24h via a startup sweep (cheap, one-liner)

If the song ID feature is not actively being debugged, the WAV writes could be gated behind a `SONGID_DEBUG=true` env flag and disabled by default.

---

## `logs/_archive/` — Review Policy

The `_archive` folder is for "I might want this later" files — not permanent storage.

**Policy:** Review monthly. Hard-delete anything older than 30 days unless there's a specific reason to keep it.

Current contents (as of 2026-05-29):
- `logs/_archive/songid_debug/` — 29 WAV files, May 2026. Review/delete by **2026-06-29**.

---
