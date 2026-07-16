# SECRETS SCAN + PUBLIC-REPO HYGIENE AUDIT

_Night-shift #1 — PUBLIC-PUSH GATE (post-asymptote rung 1) — 2026-07-15. Read-and-report only.
Destructive remediation (history rewrites, file purges, author-history rewrites, key rotation) =
**NEEDS-JONNY**, never done unilaterally by this train._

**HEADLINE: ✅ GO for public push — no secrets in git history, no ROM/savestate/Nintendo binaries
tracked or in history.** All blockers below are LOW/advisory (Jonny's call), none are hard NO-GO.
`.env` with live keys exists but is gitignored and was **never** committed.

---

## SCAN METHOD (tools + fallback)

`gitleaks`, `trufflehog`, `python`/`pip`, `go`, and `docker` are **not installed** on this machine —
so the automated scanners could not be run. Fell back to **thorough manual git sweeps** (the
documented fallback in the orders):

- **Full-history blob dump:** `git rev-list --all --objects` → **7,141 objects → 3,245 unique blobs**
  across ALL refs (main, feature/pokemon-agent, backup/vision-gemini-wip, all remotes) piped through
  `git cat-file --batch` and grepped in one pass. This covers **every commit, every branch, every blob**
  — not just HEAD.
- **All-paths-ever-added:** `git log --all --diff-filter=A --name-only` → 668 unique paths, extension-filtered.
- **Live-key confirmation:** extracted the 11 distinct live values from the working-tree `.env` and grep'd
  every history blob for each fragment.

---

## ORDER 1 — HISTORY-WIDE SECRETS SCAN

### Result: **CLEAN — zero secrets found in tracked files or git history.**

| Scan | Pattern(s) | Hits |
|---|---|---|
| Key-format regex over all 3,245 blobs | `sk-ant-`, `gsk_…`, `AIza…{35}`, `sk-proj-`, `AKIA…`, `xoxb-`, `hooks.slack.com`, `discord.com/api/webhooks`, `-----BEGIN … PRIVATE KEY-----` | **0** |
| Live `.env` value-fragment sweep over all blobs | 11 distinct live key fragments (Anthropic, Groq, ElevenLabs, Azure, Twitch OAuth, Google, AudD, Fish, Lichess, Discord webhook, Gemini-image) | **0** |
| Broad assignment scan over all blobs | `(API_KEY\|_TOKEN\|_SECRET\|OAUTH_TOKEN\|WEBHOOK_URL\|PASSWORD) = <16+ char value>` (excluding `your/example/os.environ/getenv`) | **0** |
| Email addresses in blob contents | `user@host.tld` regex | only regex garbage from binary/minified blobs (`z@u.pN` etc.); **no real address**, `jonny.dunk52@gmail.com` absent from all file contents |
| `.env` ever committed? | path scan of all-added-paths | **No** — only `.env.example` was ever tracked |

### The live `.env` (working tree — NOT tracked, NOT in history)

`.env` sits in the working tree and holds **many live-looking keys** but is correctly gitignored
(`.env` + `.env.*` with a `!.env.example` exception) and confirmed **absent from every history blob**.
It will **not** ship. **No git-history-based rotation is required.** Listed here only so Jonny knows
which providers have keys on this machine (rotation is his hands, and only if he suspects exposure
outside git):

- Anthropic, Groq, ElevenLabs, Azure Speech, Twitch OAuth, Google API/CSE, AudD, Fish TTS,
  Lichess bot, Discord webhook, Gemini-image. (`OPENAI` key was already purged per the in-file P-7 note.)

None of these appear in any tracked or historical blob, so **no key needs rotation on account of the
git history.** ✅

### `.env.example` (tracked) — clean

All 10 secret-bearing fields (`FISH_API_KEY`, `ELEVENLABS_API_KEY`, `AUDD_API_TOKEN`,
`TWITCH_OAUTH_TOKEN`, `GOOGLE_API_KEY`, `GEMINI_IMAGE_API_KEY`, `LICHESS_BOT_TOKEN`,
`DISCORD_WEBHOOK_URL`, …) are **empty placeholders** with `# REQUIRED-IF-…` comments. No real values. ✅

---

## ORDER 2 — PUBLIC-REPO HYGIENE AUDIT

### (a) ROM / savestate / Nintendo-copyrighted binaries — **✅ CRITICAL CHECK PASSES**

- **No `.gba` / `.gbc` / `.gb` / `.sav` / `.state` / savestate binary anywhere** — not in tracked files,
  not in any history blob. (The path matches for "rom"/"firered" are source filenames like
  `firered_ram.py`, `promote_bank.py`, `prompt_loader.py` — code, not ROMs.)
- `roms/` directory: **nothing tracked** (gitignored; empty in git).
- Largest history blobs are all legitimate: the demo GIF (11.5 MB, intentionally whitelisted) and ~30
  historical revisions of the 1 MB `pokemon_agent/campaign.py` (a genuinely huge source file — history
  bloat, not a secret).
- **One stray non-secret binary in HISTORY only** (advisory, LOW): `DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe.db`
  (16 KB Deno/winget cache junk) — **added** in `e7f393b` "Fresh start without large binaries",
  **removed** in `bc7873b` "Remove legacy files". Not copyrighted, tiny, already deleted from the tree.
  Only survives in history. Harmless; a future history-rewrite (if one is ever done for other reasons)
  would sweep it.

### (b) `.gitignore` adequacy for public — **✅ EXCELLENT**

Comprehensive and correctly anchored: `.env` + `.env.*` (with `!.env.example`), `*.token`, `.vts_token`,
`roms/` + `*.gba/*.gbc/*.gb`, `logs/`, `states/`, `memory_db/`, `models/`, `persona/private/`, `lore/`,
`cookie_data.json`, `audio_device.json`, `/clips/` + `recordings/` + `transcripts/`, all video exts,
`*.png/*.jpg` (with explicit whitelists for the demo gif / architecture svg / dashboard assets),
`data/`, `mpv.exe`, `.venv/`. Verified: `.vts_token`, `cookie_data.json`, `audio_device.json`,
`persona/private/`, `lore/`, `states/`, `logs/` all confirmed **untracked**. No gap found.

### (c) Tracked operational / night-train files — **recommend EXCLUDE (keep-not-delete; Jonny's call)**

No secrets in these, but they are internal night-train operational churn and read as noise in a public
repo. **Recommendation: relocate to a gitignored `docs/internal/` or add to `.gitignore` before the push
— do NOT delete (they are the durable project memory).**

- **Exclude (operational churn):** `NIGHT_REPORT.md`, `NEXT_SESSION.md`, `night_shift.ps1`,
  `MOUNTAIN_SURVEY.md`, `COUCH_NOTES.md`, `CLIPPER_RECON.md`, `MODE_TRANSITION_AUDIT.md`, and the large
  archives: `NEXT_SESSION_archive_2026-07-13_0605.md` (432 KB), `NIGHT_REPORT_archive_2026-07-13_0546.md`
  (379 KB), plus the `_HALT*` / `_pre_mission` archives and `NIGHT_REPORT_POKEMON_SUMMIT.md`.
- **Keep as public certification evidence (optional, curated):** `RUN_STATS_fresh_go_6.md` (the clean
  certifying run) — and possibly a single trimmed mountain-survey. `STATE_OF_PROJECT.md` (200 KB) is
  internal reality-map — recommend exclude or heavily trim.
- **CLAUDE.md** (42 KB internal operating brief): exposes internal doctrine + the absolute working-dir
  path. Recommend exclude from public or replace with a short contributor guide. Jonny's call.

### (d) LICENSE — **✅ AGPL v3 present and correct**

`LICENSE` = "GNU AFFERO GENERAL PUBLIC LICENSE, Version 3, 19 November 2007", full verbatim text.
(Confirm a project copyright line / per-file headers if desired — the file itself is the standard AGPLv3.)

### (e) Hardcoded absolute paths in code — **inventory only (do NOT refactor yet)**

Core `kira/` is **clean** — the broad regex hits in `kira/bot.py`, `kira_state.py`,
`playthrough_memory.py`, `vn_autopilot.py`, `audio_agent.py` are **false positives** (regex literals like
`re.search(r'\[POLL:…')`, not paths). Real machine-specific absolute paths live only in throwaway/ops
scripts:

| File(s) | Path | Impact |
|---|---|---|
| `pokemon_agent/recon_*.py` (**~12 files:** recon_bill_eyes, recon_bill_track, recon_console_eyes, recon_gauntlet_fight, recon_itemuse, recon_mart, recon_route6_wedge, recon_teach_derive, recon_vermilion_map, recon_seafoam, recon_giovanni, recon_diglett_probe …) | `r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\<uuid>\scratchpad"` | throwaway recon scripts; break on other machines + expose local dir + username. Candidates to **delete before push** (not shipped-value code). |
| `pokemon_agent/fresh_go_watchdog.sh:8` | `cd /g/JonnyD/NeuroAI_Bot/pokemon_agent` | night-train launcher; machine-specific. Exclude with the ops files. |
| `pokemon_agent/promote_to_workshop.py:9`, `tools/run_stats.py:12` | `G:/temp/longrun/…` | **docstring example text only** — harmless, cosmetic. |

Markdown docs `CLAUDE.md`, `NEXT_SESSION.md`, `NIGHT_REPORT_POKEMON_SUMMIT.md` also contain the literal
`G:\JonnyD\…` path — covered by the (c) exclude recommendation.

---

## PERSONAL IDENTIFIERS (in git METADATA — NEEDS-JONNY, destructive to change)

Not secrets, but flagged per orders. Changing these requires an author-history rewrite (destructive) →
**Jonny's call**, not this train's:

- **All 1,344 commits** authored/committed as `Jonny <jonny.dunk52@gmail.com>` — personal Gmail exposed
  in every commit. (Common; many public repos accept this or switch to a GitHub `noreply` email going forward.)
- **2 early commits** as `Jonny Dunkleberger <jonnydunkleberger@Jonnys-MacBook-Pro.local>` — exposes the
  **full real name** and a **machine hostname**. This is the one metadata item most worth a conscious
  decision before going public.
- Local username **`JonnyD`** appears in: `VTS_PLUGIN_DEVELOPER` default (`kira/config.py`, `.env.example`
  — benign), the recon scratchpad paths above, the watchdog `cd` path, and a GitHub namespace URL in
  `tools/codenames_sync.user.js` (`https://github.com/JonnyD/NeuroAI_Bot` — cosmetic).

---

## RANKED BLOCKER LIST (all LOW / advisory — none block the push)

1. **[LOW] Author metadata** exposes personal Gmail (all commits) + full real name + machine hostname
   (2 commits). Rewriting author history is destructive → **NEEDS-JONNY decision** (accept, or rewrite +
   set a `noreply` email going forward). *Not a NO-GO — this is normal for public repos.*
2. **[LOW] Internal ops docs tracked** (NIGHT_REPORT/NEXT_SESSION/STATE_OF_PROJECT/CLAUDE.md/archives) —
   noise + exposes local paths & internal doctrine. Recommend gitignore/relocate before push. Keep, don't delete.
3. **[LOW] ~12 recon_*.py scratchpad-path scripts** — throwaway, machine-specific. Recommend delete before push.
4. **[VERY LOW] Winget `.db` (16 KB) in history only** — already deleted from tree; harmless junk.
5. **[VERY LOW] Repo size** — 11.5 MB demo GIF (whitelisted) + ~30 historical 1 MB `campaign.py` copies
   bloat clone size. Cosmetic; not a blocker.

**No key rotation required on the basis of git history. No ROM/binary offender. No secret in any blob.
The repo is secrets-clean and ROM-clean — GO, pending Jonny's cosmetic/metadata calls above.**
