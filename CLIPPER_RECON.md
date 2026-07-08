# CLIPPER_RECON — Phase K step 1 (Rule-3 audit of the 007-tested pipeline, 2026-07-07 night shift 1)

Read-only audit per the Phase K mandate ("recon FIRST: keep what's good, name what's weak,
rebuild ONLY the weak"). Ground truth from a full code sweep; file:line pointers verified.

## THE MAP (what exists, where)
| File | Role |
|---|---|
| `kira/clips/clip_cutter.py` (1730 loc) | THE ENGINE: candidates → wall-clock anchor (quote→events.jsonl fuzzy match ≥0.72) → recording map → ffmpeg cut (copy, NVENC/x264 fallback) → Haiku titles → 4 output folders + clips_report.md |
| `scripts/cut_clips.py` | CLI entry → `cut_session()` |
| `kira/clips/obs_anchor.py` | OBS WS record-start anchor (opt-in) |
| `scripts/stitch_clips.py` | ORPHANED 007-era reel/recap stitcher — wants `cuts_*.json` which NOTHING produces |
| `scripts/transcribe_vod.py` | Whisper VOD → timestamped transcript |
| `scripts/backfill_clips.py` | transcripts → clips/*.md candidates (Sonnet, prompt DUPLICATED from bot.py:8822) |
| `kira/bot.py:10966,8670` | live highlight loop + session-end artifacts → `clips/<date>_<activity>.md` candidates |
| `kira/brain/cost_tracker.py` | cost telemetry — wired into ai_core, NOT into this pipeline |

Note: repo-root `clips/` = candidate MARKDOWN (input), not video. Rendered video lands under
`OBS_RECORDINGS_DIR/clips/<date>/` (off-repo). Outputs today: `01_clips_by_type/`,
`02_reel_best_of/` (cap 300s), `03_highlight_vod/` (teaser+chronological), `04_short_candidate/`
(landscape COPY of the top clip — no reframe).

## VERDICTS vs the Phase-K deliverable
- (a) highlights cut scaled to VOD length — **WEAK** (length is emergent from candidate count; no target-runtime pacing)
- (b) 3-4 min best-of — **SOLID-ish** (exists at 5-min cap `CLIP_REEL_MAX_SECONDS=300`; retune + done)
- (c) 9:16 shorts + BURNED-IN subs — **MISSING ENTIRELY** (zero vertical reframing, zero subtitle burn; code punts to "Premiere by hand" at clip_cutter.py:1246,1432). The non-negotiable gap.
- (d) titles/desc/tags — **SOLID** mechanism (Haiku JSON + sidecars); missing only the Neuro-comparison/personality-hook doctrine in the prompt
- (e) review queue + manifest — **WEAK** (staging folder + human `clips_report.md` exist; no machine `manifest.json`, no approved-flag state; stitch_clips' manifest gap is D2)
- (f) cost instrumentation — **MISSING** (clip_cutter.py:714,1100 + backfill_clips.py:155 call Anthropic RAW, bypassing cost_tracker — per-VOD cost unknowable)
- CLIP SELECTION — **WEAK for K's bar**: purely LLM-transcript scoring (bot.py:8828). Structured KIRA-MOMENT signals EXIST but are unmined: tier-3 salience beats (`_pokemon_react` tier / `_promote_saga_beat`), `highlight_captured` events. "She named the rat and declared war" cannot currently outrank "boss died".

## DEBT REGISTER
- **D1 tail bug, half-fixed:** asymmetric mode grows the FRONT on min-floor (right); fixed-window FALLBACK still grows the TAIL (clip_cutter.py:568-569) → dead-air tails on fallback clips. Finish by mirroring the front-grow.
- **D2 stitch_clips.py orphaned:** consumes `cuts_*.json` nothing emits; falls back to fake ordering (`clip_id*100`). Kill or feed it the new manifest.
- **D3 hardcoded paths** (.env:11-12,116 — whisper models dir, OBS dir).
- **D4 duplicated candidate prompt** (bot.py:8822 vs backfill_clips.py:126 — drift risk).
- **D5 VOD-alignment fragility:** single-anchor 0.45-confidence match only WARNS before cutting a possibly misaligned VOD — no hard stop.

## THE MINIMAL REBUILD (keep the 007-proven core: anchoring, ffmpeg strategy, Whisper start-derivation, titling)
1. **Vertical-short renderer** (the one from-scratch build): top-N clips → 1080×1920 crop/scale + per-short `.ass` captions from the existing Whisper segments → ffmpeg `subtitles=` burn. 3-5 shorts, not 1.
2. **Kira-moment selection layer:** log tier/soul/highlight beats cutter-readably (events.jsonl already carries `highlight_captured`; add tiered pokemon/reaction beats), then blend into candidate scoring alongside the transcript pass.
3. **manifest.json + review-queue state** (id/paths/score/type/anchor/duration/approved) next to clips_report.md; point stitch_clips at it or retire it (D2).
4. **Cost line per VOD:** route the pipeline's LLM calls through cost_tracker; print "this VOD cost $X to process" in the report (Phase-J forward ref, inline until J lands).
5. **Length targets:** highlight cut scaled to VOD duration; best-of cap → 3-4 min.
6. **D1 finish + D3/D4 hygiene.**

Exit stays per mandate: one 007/dev-stream VOD in → postable outputs, zero manual editing →
needs-eyes ledger item 6.
