# Kira AI - Config loads from .env for secrets and sensitive info
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Repo root = parent of the kira/ package dir. Used for default data/model paths
# so they resolve at the repo root regardless of where config.py lives.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model and runtime config (safe to share)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
# ↑ Primary VRAM lever. -1 = all layers on GPU (~6-7 GB for 8B Q4_K_M).
#   If streaming a VRAM-hungry game (e.g. Bond at 4K) causes OOM, drop to ~28
#   to offload the remainder to CPU. CPU layers add ~30-80ms latency per token
#   but keep the bot stable. Profile with torch.cuda.memory_reserved() first.
N_CTX = int(os.getenv("N_CTX", 16384))
N_BATCH = int(os.getenv("N_BATCH", 512))

LLM_MAX_RESPONSE_TOKENS = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", 512))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

# [VRAM] telemetry: how often (seconds) to log Kira's per-process GPU footprint.
# Cheap (one NVML query); set high to quiet it. Diagnostic instrument, not a fix.
VRAM_LOG_INTERVAL_S = int(os.getenv("VRAM_LOG_INTERVAL_S", "45"))

# Wheel ceremony timing (env-tunable to dial spin feel live). The backend SENDS
# these to the overlay AND waits on them, so the result reaction always fires
# after the wheel visibly lands. entrance + spin + buffer = time from spin-event
# to the slice reaction.
WHEEL_ENTRANCE_MS    = int(os.getenv("WHEEL_ENTRANCE_MS", "1800"))   # anticipation hold before the spin
WHEEL_SPIN_MS        = int(os.getenv("WHEEL_SPIN_MS", "6000"))       # dramatic spin duration
WHEEL_LAND_BUFFER_MS = int(os.getenv("WHEEL_LAND_BUFFER_MS", "600")) # let the landing read before she reacts
# Wheel chat-vote (Layer 3): how long chat has to vote on a parameter, and
# whether keyword votes count in addition to the option number. Window is short
# enough to keep the show moving, long enough for live chat to type a digit.
WHEEL_VOTE_WINDOW_S       = int(os.getenv("WHEEL_VOTE_WINDOW_S", "25"))            # seconds chat has to vote
WHEEL_VOTE_ALLOW_KEYWORDS = os.getenv("WHEEL_VOTE_ALLOW_KEYWORDS", "true").lower() == "true"  # accept keyword votes, not just numbers
WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", os.path.join(_REPO_ROOT, "models", "whisper"))
# Vision master kill-switch. Default ON: general perception means she can see at a
# calm cadence WITHOUT arming a mode. Set false to make her blind everywhere
# (the one switch that disables all vision, even on-demand visual questions).
ENABLE_VISION = os.getenv("ENABLE_VISION", "true").lower() == "true"
# Calm (general-conversation) vision heartbeat cadence, in seconds. The fast 10s
# cadence stays mode-tuned (game/media); this relaxed rate keeps her aware in plain
# conversation without hammering the vision model.
VISION_CALM_HEARTBEAT_SECONDS = float(os.getenv("VISION_CALM_HEARTBEAT_SECONDS", 40.0))
# Vision capture dedup: skip a heartbeat capture if an on-demand capture started within
# this window (likely still in flight) — avoids a 2x concurrent gpt-4o-mini call. Safe-
# mechanical; default ON. Window ~ a capture's max duration so it covers the in-flight time.
VISION_CAPTURE_DEDUP_ENABLED = os.getenv("VISION_CAPTURE_DEDUP_ENABLED", "true").lower() == "true"
VISION_CAPTURE_DEDUP_WINDOW_S = float(os.getenv("VISION_CAPTURE_DEDUP_WINDOW_S", "5.0"))

# ── Turbo Vision slideshow (multi-frame "what happened" context; default OFF) ───
# Ports the MediaWatch slideshow into the ALWAYS-ON Turbo Vision path: when Turbo
# Vision is engaged (the deep_senses lever), a dedicated FULL-SCREEN capture loop
# fills a rolling N-frame buffer, and every analysis interval the buffer is sent to
# gpt-4o-mini as ONE multi-image "what HAPPENED across these frames" call, appended
# to a bounded episode timeline Kira can answer from (richer than a single stale
# snapshot). Only runs while Turbo Vision is ON, so cost is bounded to dialed-in
# sessions (~the MediaWatch profile Jonny already accepted, ~$0.10-0.25/episode).
# All three knobs are tunable so fidelity-vs-cost can be dialed without a code edit:
#   CAPTURE_INTERVAL_S — capture cadence (lower = tighter action window, more cost)
#   BUFFER_SIZE        — frames per analysis (window ≈ interval * size)
#   ANALYSIS_INTERVAL_S— how often the buffer is analyzed into the timeline
# OFF -> no slideshow capture/analysis at all (today's single-frame behavior exactly).
TURBO_VISION_SLIDESHOW_ENABLED = os.getenv("TURBO_VISION_SLIDESHOW_ENABLED", "false").lower() == "true"
TURBO_VISION_CAPTURE_INTERVAL_S = float(os.getenv("TURBO_VISION_CAPTURE_INTERVAL_S", "1.75"))
TURBO_VISION_BUFFER_SIZE = int(os.getenv("TURBO_VISION_BUFFER_SIZE", "8"))
# Analysis interval is the REAL freshness bottleneck for what she KNOWS (the 1.75s
# capture only refreshes the EYES thumbnail; her reasoning uses the analysis output).
# Lowered 10.0 -> 5.0 (2026-06-22) for ~5s-fresh vision at ~2x gpt-4o-mini cost
# (~+$1.80/day continuous). Do NOT chase freshness via the capture interval — that's
# wasted money (it doesn't change what she knows). Env-tunable.
TURBO_VISION_ANALYSIS_INTERVAL_S = float(os.getenv("TURBO_VISION_ANALYSIS_INTERVAL_S", "5.0"))

# ── Emotion → voice (audible mood; default OFF) ────────────────────────────────
# Her computed emotional state currently drives her FACE + word choice but NEVER her
# VOICE — every line plays at the same flat rate/pitch. ON: nest a small per-emotion
# prosody delta INSIDE the base AZURE_PROSODY_RATE/PITCH (reusing the VN nested-<prosody>
# scaffold), so HAPPY/MOODY/SASSY/EMOTIONAL/HYPERACTIVE actually SOUND different. Also a
# <break> at "..." for the deadpan beat (Piece 2). Covers the main streaming reply AND
# interjections (both route through _speak_single). OFF -> byte-for-byte today's flat SSML.
# Zero added latency (string interpolation). Built on <prosody> + <break> only (guaranteed
# on AshleyNeural) — does NOT depend on Azure express-as styles.
VOICE_EMOTION_ENABLED = os.getenv("VOICE_EMOTION_ENABLED", "false").lower() == "true"
VOICE_EMOTION_BREAK_MS = int(os.getenv("VOICE_EMOTION_BREAK_MS", "180"))  # deadpan beat inserted at "..." (0 = off)
# Per-emotion prosody DELTAS (rate, pitch), nested inside the base prosody. TUNABLE TABLE —
# edit these to dial each mood's voice. Small/tasteful = audible mood, not cartoonish.
# RATE is pinned to +0% for EVERY emotion on purpose: one constant speaking speed is
# what makes deadpan land. Mood rides on PITCH ONLY. Speed shifts killed the comedic
# timing (and dragged the baseline during slow moods), so rate variation is gone —
# do not re-add it. Only the pitch column should ever be tuned here.
VOICE_EMOTION_PROSODY = {
    "HAPPY":       ("+0%",  "+0%"),   # neutral-bright (base is already +25% pitch)
    "MOODY":       ("+0%",  "-4%"),   # lower — withdrawn (same speed)
    "SASSY":       ("+0%",  "+5%"),   # brighter — snappy via pitch, not rate
    "EMOTIONAL":   ("+0%",  "-3%"),   # softer — intimate (same speed)
    "HYPERACTIVE": ("+0%",  "+0%"),   # base only — +8% chipmunked on the bright +25% base (was the chipmunk culprit)
}

# ── Emotion swing (mood persistence; default OFF) ──────────────────────────────
# The prosody above is only as alive as her emotion SWINGS. Today emotion follows the
# per-turn classifier, which reverts to HAPPY the moment a turn reads neutral — so a mood
# beat doesn't linger. ON: once she enters a non-HAPPY mood, HOLD it through up to
# EMOTION_SWING_HOLD_TURNS subsequent neutral/HAPPY readings before reverting (a genuinely
# NEW mood still switches immediately; the existing 8-turn decay cap still bounds it, so
# she never locks). Makes moods last long enough to be audible without being erratic.
# OFF -> today's per-turn behavior exactly.
EMOTION_SWING_ENABLED = os.getenv("EMOTION_SWING_ENABLED", "false").lower() == "true"
EMOTION_SWING_HOLD_TURNS = int(os.getenv("EMOTION_SWING_HOLD_TURNS", "4"))  # extra turns a mood lingers through neutral reads

# ── Persistent self ("the Eevee": constant core, evolving form) ────────────────
# Her self (mood + feelings + standing takes + current want + bond with Jonny) reached
# her REPLIES but NOT her DRIVES (Director/interjections fired from perception alone).
# These route the SAME compact self-block into the drive path so proactive lines come
# FROM her self, add a current-want through-line, and let her feelings about Jonny evolve.
# Flag-gated for A/B on stream. Default ON.
# Autonomous Pokémon agent (M1) — the reaction layer is OFF by default so it can
# never affect a normal stream until explicitly armed. The HANDS (emulator/policy)
# live fully isolated under pokemon_agent/; this only gates the Kira REACTION seam.
POKEMON_AGENT_ENABLED = os.getenv("POKEMON_AGENT_ENABLED", "false").lower() == "true"
# Pokémon mode gates the desktop audio-CLASSIFIER (so she stops hearing/reacting to the game music
# that shares her loopback endpoint). Each game event refreshes a self-reverting linger of this many
# seconds; once events stop for this long, normal desktop hearing resumes. Mic + game-event seam are
# never affected. Set to 0 to disable the auto-gate (the dashboard forced toggle still works).
POKEMON_HEARING_SUPPRESS_S = float(os.getenv("POKEMON_HEARING_SUPPRESS_S", "60.0"))
DRIVE_SELF_BLOCK_ENABLED = os.getenv("DRIVE_SELF_BLOCK_ENABLED", "true").lower() == "true"   # ① self into drives
CURRENT_WANT_ENABLED     = os.getenv("CURRENT_WANT_ENABLED", "true").lower() == "true"        # ② through-line
JONNY_BOND_ENABLED       = os.getenv("JONNY_BOND_ENABLED", "true").lower() == "true"          # ④ relational evolution

# ── Self-aware glitch beats (default OFF) ──────────────────────────────────────
# Surface HER OWN glitches (loopback went deaf, model fallback, mishear) as a RARE,
# rate-limited fourth-wall wink ("...I completely blanked, give me a sec") via a
# low-priority interjection — pairs with the FOURTH-WALL AI JOKES disposition. HARD
# requirement: aggressively rate-limited so it's a rare wink, not a running complaint —
# a long cooldown AND a probability gate (only sometimes, even off cooldown). OFF -> silent
# (glitches stay dev-log only, today's behavior).
GLITCH_AWARE_ENABLED = os.getenv("GLITCH_AWARE_ENABLED", "false").lower() == "true"
GLITCH_AWARE_COOLDOWN_S = float(os.getenv("GLITCH_AWARE_COOLDOWN_S", "600.0"))  # min seconds between glitch beats (10 min)
GLITCH_AWARE_CHANCE = float(os.getenv("GLITCH_AWARE_CHANCE", "0.3"))            # only react this fraction of the time, even off cooldown

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
# TTS backend selector — overrides TTS_ENGINE for choosing Azure vs Fish Audio.
# "azure"  -> Azure Cognitive Speech SDK (default; full word-boundary timing for captions)
# "fish"   -> Fish Audio SDK (pay-as-you-go; falls back to Azure on any error)
# Switchable at runtime via the dashboard TTS toggle without editing code.
TTS_BACKEND = os.getenv("TTS_BACKEND", "azure")
# Fish Audio credentials (from .env — never commit real values)
FISH_API_KEY    = os.getenv("FISH_API_KEY", "")
FISH_VOICE_ID   = os.getenv("FISH_VOICE_ID", "")  # reference_id of the chosen voice model
FISH_LATENCY    = os.getenv("FISH_LATENCY", "balanced")  # "normal" | "balanced"
FISH_FORMAT     = os.getenv("FISH_FORMAT", "mp3")         # "mp3" | "wav" | "pcm" | "opus"
AI_NAME = os.getenv("AI_NAME", "Kira")
# Tuning VAD for faster response (0.4s silence triggers end-of-speech)
PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", 0.4))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
# Loopback STT mic-gate (Task 2, self-healing). The loopback transcriber skips
# ticks while Jonny's mic is genuinely active so his voice isn't transcribed as
# desktop audio. "Active" is derived from the timestamp of the last mic speech
# frame and auto-expires this many seconds after he stops — short, because the
# transcriber adds its own ~10s post-mic cooldown on top. Kept well above the
# PAUSE_THRESHOLD (0.4s) utterance gap so the gate never flickers mid-sentence.
MIC_GATE_ACTIVE_WINDOW_S = float(os.getenv("MIC_GATE_ACTIVE_WINDOW_S", 2.0))
# Loopback STT mic gate: suppress desktop transcription while Jonny's mic is active
# (+a ~10s cooldown). Its ONLY purpose is to stop his mic voice being transcribed as
# ambient dialogue WHEN the loopback device carries a mic-mixed signal (e.g. a
# VB-Audio virtual cable). On a headphone OUTPUT-loopback rig his mic is NOT in that
# signal, so the gate is pure harm — on chat-heavy streams (near-constant talking)
# it blacks out loopback ~12s after every utterance, chaining into 30-45s gaps and
# turning her constant desktop hearing intermittent. Default FALSE for that setup;
# set true ONLY if your loopback device mixes in the mic (virtual cable / mic
# monitoring / sidetone). Does NOT affect the self-TTS gate (that stays on).
LOOPBACK_MIC_GATE_ENABLED = os.getenv("LOOPBACK_MIC_GATE_ENABLED", "false").lower() == "true"
# Headphone setups have ZERO mic bleed: Kira's TTS never reaches the mic, so the
# self-hearing guard that discards mic frames while she speaks is pure liability
# — it eats the FIRST WORD of the user's real speech whenever is_speaking is True
# (most of every turn). When true, the mic captures continuously THROUGH her
# speech (frames are buffered, never discarded) so his full utterance survives
# from word one. She stays non-interruptible regardless (the VAD never reaches
# the interruption check while she speaks). Default false keeps speaker setups
# safe — those still need the discard guard to avoid transcribing her own voice.
ASSUME_NO_MIC_BLEED = os.getenv("ASSUME_NO_MIC_BLEED", "false").lower() == "true"
MEMORY_PATH = os.getenv("MEMORY_PATH", os.path.join(_REPO_ROOT, "memory_db"))

# ── Smart memory retrieval (read-layer rework; HIGHEST blast radius -- default OFF) ─
# Reworks get_semantic_context's vector path: fetch distances+metadata (today it's
# documents-only), drop barely-related hits by a MEASURED similarity floor, re-rank by
# a similarity/recency blend, and frame results by confidence ([WHAT YOU KNOW] vs
# [VAGUE RECOLLECTIONS]) -- routing Kira's own highlights/summaries into a SEPARATE
# "you might remember from a stream" lane instead of laundering them as verified facts.
# Fixes within-session forgetting + the hallucinated-lore ("Stick") path. OFF -> the
# original get_semantic_context runs VERBATIM (the fallback). Flip on to feel-test;
# loud [SmartMem] logging shows candidates fetched / floor-drops / survivors / lanes.
MEMORY_SMART_RETRIEVAL_ENABLED = os.getenv("MEMORY_SMART_RETRIEVAL_ENABLED", "false").lower() == "true"
# L2 MAX distance -- keep candidates with distance <= floor. From offline calibration
# (scripts/calibrate_memory_floor.py): true-match p90=0.95, noise min=1.40 -> floor between,
# 100% true recall / 0 noise leak. Re-run the calibrator if the embedding model changes.
MEMORY_SIM_FLOOR = float(os.getenv("MEMORY_SIM_FLOOR", "1.2178"))
MEMORY_CANDIDATE_N = int(os.getenv("MEMORY_CANDIDATE_N", "10"))          # pool fetched before floor + re-rank
MEMORY_RECENCY_WEIGHT = float(os.getenv("MEMORY_RECENCY_WEIGHT", "0.25"))  # recency vs similarity (0..1; a tiebreaker, not a dominator)
MEMORY_RECENCY_HALFLIFE_S = float(os.getenv("MEMORY_RECENCY_HALFLIFE_S", "21600.0"))  # 6h: within-session + recent-day ramp; env-tunable
MEMORY_CONF_HIGH = float(os.getenv("MEMORY_CONF_HIGH", "0.85"))          # >= -> [WHAT YOU KNOW]; below -> [VAGUE RECOLLECTIONS]

# Secrets and API keys (must be in .env, never commit real values)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "")
AZURE_PROSODY_PITCH = os.getenv("AZURE_PROSODY_PITCH", "")
AZURE_PROSODY_RATE = os.getenv("AZURE_PROSODY_RATE", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TWITCH_OAUTH_TOKEN = os.getenv("TWITCH_OAUTH_TOKEN", "")
TWITCH_BOT_USERNAME = os.getenv("TWITCH_BOT_USERNAME", "")
TWITCH_CHANNEL_TO_JOIN = os.getenv("TWITCH_CHANNEL_TO_JOIN", "")
VIRTUAL_AUDIO_DEVICE = os.getenv("VIRTUAL_AUDIO_DEVICE", "")

# Audio understanding config
ENABLE_AUDIO_AGENT = os.getenv("ENABLE_AUDIO_AGENT", "true").lower() == "true"
# Audio-mood always-on: when true (default), the audio agent boots straight into an
# active mood-reading state (MEDIA) so _audio_mood() colors GENERAL conversation, not
# just armed modes. None=no-op already, so this is safe. Set false to boot OFF and only
# hear once a mode is armed.
AUDIO_MOOD_ALWAYS_ON = os.getenv("AUDIO_MOOD_ALWAYS_ON", "true").lower() == "true"
# Loopback Whisper transcriber — SEPARATE opt-in flag from the audio-MOOD agent.
# Distil-large-v3 is English-only and produces token-loop garbage on JP/VN content.
# Default ON for general perception (she hears game/show dialogue); the WhisperModel
# is only loaded when this is true. Set false to skip the second Whisper entirely.
ENABLE_LOOPBACK_TRANSCRIBER = os.getenv("ENABLE_LOOPBACK_TRANSCRIBER", "true").lower() == "true"
# Whether the loopback STT AUTO-STARTS (no manual toggle) once MEDIA audio is active.
# Default true: combined with the always-on audio above, dialogue transcription comes
# up on boot. Set false to require the dashboard LOOPBACK STT toggle. Honored both at
# boot and whenever MEDIA audio mode is switched on from the dashboard.
LOOPBACK_STT_DEFAULT = os.getenv("LOOPBACK_STT_DEFAULT", "true").lower() == "true"

# ── Loopback dialogue-summary escape hatches (the "music-freeze" fix; Batch 1) ──────
# The rolling "story so far" summary accumulates across windows (wanted for a long story
# game). BUG: it FREEZES on the last narrative scene when audio drops to music/quiet —
# the summarizer keeps returning NO_UPDATE, which preserves the dead scene forever. AGE-OUT
# is the escape hatch: if no GENUINE narrative segment has refreshed the summary for this
# long, stop holding the frozen scene and honestly stale-mark it ("audio is music/quiet").
# Only fires after at least one real update; every stale-mark logs loudly. ~a few minutes.
LOOPBACK_SUMMARY_AGEOUT_S = float(os.getenv("LOOPBACK_SUMMARY_AGEOUT_S", "180.0"))

# ── Content-switch reset (the "scene-bleed" fix; Batch 2) ───────────────────────────
# The summarizer is told to track continuity, so when content CHANGES outright (a film
# ends and a game begins — Bond → Pragmata) the old characters/plot bleed into the new
# scene. Content-switch reset drops `previous` and regenerates fresh when the recent
# transcript window's vocabulary barely overlaps the current summary AND there's enough
# new dialogue that this isn't just an in-scene lull. Conservative by design — a reset
# needs BOTH near-zero overlap and a substantial new window — so a quiet stretch never
# triggers it; every reset logs so sensitivity can be tuned live.
#   OVERLAP   = fraction of the summary's content words still present in the recent
#               window; BELOW this ⇒ candidate switch (lower = harder to trigger).
#   MIN_WORDS = content-word floor in the recent window; guards against false-firing on
#               a lull (few words ⇒ never resets).
LOOPBACK_SUMMARY_SWITCH_OVERLAP = float(os.getenv("LOOPBACK_SUMMARY_SWITCH_OVERLAP", "0.08"))
LOOPBACK_SUMMARY_SWITCH_MIN_WORDS = int(os.getenv("LOOPBACK_SUMMARY_SWITCH_MIN_WORDS", "12"))

# ── Dialogue-summary name-drift guard (default ON) ──────────────────────────────
# The dialogue summarizer is told to preserve proper names verbatim — so a MISHEARD
# name from Whisper ("Korra" -> "Cora") gets faithfully cemented, then propagated
# every 15s by the continuity instruction. That wrong name feeds Kira's context and
# she references a character who doesn't exist. When an activity TITLE is set (e.g.
# "Legend of Korra"), this guard anchors the summarizer on it: correct an obvious
# speech-to-text mis-transcription toward the canonical character name, and if a
# garbled name matches NO known character, refer to them by role rather than commit
# to a wrong guess (confident vagueness over confident wrongness). Strictly a no-op
# when no activity name is set — the prompt is then byte-identical to before.
# OFF -> today's summary prompt exactly. Does NOT touch watch-party/MediaWatch.
LOOPBACK_NAME_DRIFT_GUARD_ENABLED = os.getenv("LOOPBACK_NAME_DRIFT_GUARD_ENABLED", "true").lower() == "true"

# ── Loopback post-TTS cooldown (the "deaf during dense talking" knob) ───────────
# After Kira's TTS ends, the loopback STT pump skips for this long so the tail of
# her own voice (mixed into the same headphone endpoint it taps) isn't transcribed
# as "dialogue". The old hard-coded 8s (WINDOW_SECONDS) overlapped continuously
# when she talked densely (chat/streamer register) and starved the pump → 0
# segments ("alive but deaf").
#
# As of the self-echo fix (2026-06-22) the cooldown is NO LONGER the only guard
# against her speech tail leaking. The transcription window is now clamped to a
# buffer high-water-mark (it never reaches back past _speech_last_active_ts, the
# real end of her TTS), so a SHORT cooldown stays leak-free. Default dropped to
# 1.0s — just acoustic decay — because the high-water-mark, not the cooldown,
# now does the heavy lifting. A self-echo fingerprint backstop kills any residual.
# Lower = more responsive (she hears the show sooner in the gaps); raise only if
# the acoustic tail still bleeds. Surfaced in the boot [Config] line.
LOOPBACK_POST_TTS_COOLDOWN_S = float(os.getenv("LOOPBACK_POST_TTS_COOLDOWN_S", "1.0"))

# Smart Game Mode configuration (dashboard ACTIVATE button).
# When true (default), clicking ACTIVATE in the Game Mode panel automatically configures
# all subsystems to stream-ready state: Vision ON, Audio=MEDIA, Immersive OFF,
# Highlight Extraction ON, Loopback per LOOPBACK_STT_DEFAULT.
# Set false to restore the old manual "dumb" activation behavior.
GAME_MODE_AUTO_CONFIGURE = os.getenv("GAME_MODE_AUTO_CONFIGURE", "true").lower() == "true"

# Highlight Extraction global kill-switch.
# When true (default), the background Opus loop fires on the interval below whenever
# an activity is active (ACTIVITY_GAME, VN, or MEDIA). Set false to fully disable.
HIGHLIGHT_EXTRACTION_ENABLED = os.getenv("HIGHLIGHT_EXTRACTION_ENABLED", "true").lower() == "true"

# How often (seconds) the highlight extraction loop fires.
# Default 300s (5 min) = 12 Opus calls/hr. Old default was 90s (40 calls/hr).
# Tune via env: HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS=180
HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS = int(os.getenv("HIGHLIGHT_EXTRACTION_INTERVAL_SECONDS", "300"))

# How often (seconds) the playthrough crash-recovery checkpoint is flushed to disk.
# Default 300s (5 min): a crash loses at most 5 minutes of reactions/chat moments.
# Lower for finer granularity; raise if the file I/O is a concern on slow storage.
CHECKPOINT_INTERVAL_SECONDS = int(os.getenv("CHECKPOINT_INTERVAL", "300"))

# Persistent stream logging (transcript.md + events.jsonl + summary.md per session).
# Written to logs/streams/YYYY-MM-DD_HH-MM_<activity>/. All writes are async and
# non-blocking. Set false only for development / offline testing.
# Note: "Test / Companion Mode" preset also forces logging off via the dashboard.
STREAM_LOGGING_ENABLED = os.getenv("STREAM_LOGGING_ENABLED", "true").lower() == "true"

# Diary→recap bridge: feed the orphaned Opus session summary's "## Personality
# Highlights" section back into the startup-brief consolidation, so she opens every
# session with an unconditional "previously, with you and chat" (emotional/relational
# beats included) instead of keyword-fishing for it. Reads ONLY the relational section
# (not the dev-telemetry Stats/Issues/Suggestions), as a third source alongside
# lore/clips — output size unchanged. Degrades silently to lore+clips on any miss
# (no summary yet / PENDING / parse fail); every branch logs [DiaryBridge]. Default ON.
DIARY_RECAP_ENABLED = os.getenv("DIARY_RECAP_ENABLED", "true").lower() == "true"

AUDIO_HEARTBEAT_SECONDS = float(os.getenv("AUDIO_HEARTBEAT_SECONDS", "12.0"))
AUDIO_CLIP_SECONDS = float(os.getenv("AUDIO_CLIP_SECONDS", "8.0"))
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "gpt-4o-mini-audio-preview-2024-12-17")
# Audio content classifier (2026-06-22): the mood call ALSO returns a dominant
# content tag (SPEECH/MUSIC/AMBIENT/MIXED) — zero extra API cost. MUSIC/AMBIENT are
# BACKGROUNDED (audio_summary_is_event forced False: no proactive trigger, no
# intensity spike — she experiences the score as background, not a story event);
# SPEECH/MIXED stay foreground (dialogue priority). Default ON; flip to A/B on stream.
AUDIO_CONTENT_CLASSIFY_ENABLED = os.getenv("AUDIO_CONTENT_CLASSIFY_ENABLED", "true").lower() == "true"

# Feature Flags
ENABLE_TWITCH_CHAT = os.getenv("ENABLE_TWITCH_CHAT", "true").lower() == "true"
ENABLE_YOUTUBE_CHAT = os.getenv("ENABLE_YOUTUBE_CHAT", "true").lower() == "true"
# Allow messages from the broadcaster/channel-owner account to reach Kira's brain.
# In TwitchIO 2.x, message.echo=True whenever author==bot.nick. If the OAuth token
# belongs to the broadcaster account, ALL broadcaster messages are silently dropped.
# Set this true when bot-account == broadcaster-account (common in small setups).
# Safe to leave true in production — it only passes the broadcaster's OWN messages
# through; all other accounts are unaffected by this flag.
ALLOW_BROADCASTER_CHAT = os.getenv("ALLOW_BROADCASTER_CHAT", "false").lower() == "true"

# Chat POSTING (Kira sending messages, separate from reading). Default OFF —
# turn it on only when you've decided you want her to be able to send. Hard
# rate-limited via CHAT_POST_COOLDOWN_SEC to keep this flavor, not spam.
# Twitch posting works out of the box via the existing TWITCH_OAUTH_TOKEN.
# YouTube posting is NOT yet implemented (pytchat is read-only; sending
# requires a separate OAuth2 flow on the YouTube Data API).
ENABLE_CHAT_POSTING = os.getenv("ENABLE_CHAT_POSTING", "false").lower() == "true"
CHAT_POST_COOLDOWN_SEC = float(os.getenv("CHAT_POST_COOLDOWN_SEC", "60.0"))
CHAT_POST_MAX_LEN = int(os.getenv("CHAT_POST_MAX_LEN", "450"))  # Twitch hard cap is 500
ENABLE_YOUTUBE_POSTING = os.getenv("ENABLE_YOUTUBE_POSTING", "false").lower() == "true"

# Kira's [CHAT: ...] tool — her own typing-in-chat channel. These caps layer ON
# TOP of CHAT_POST_COOLDOWN_SEC above (the transport floor shared by all posters).
# The bar: a viewer should occasionally see "kira just typed in chat??" — rare
# enough to be an event, never spam. Tighter length than the transport cap too.
CHAT_POST_KIRA_INTERVAL_SEC = float(os.getenv("CHAT_POST_KIRA_INTERVAL_SEC", "300.0"))  # min 5 min between her posts
CHAT_POST_KIRA_MAX_PER_SESSION = int(os.getenv("CHAT_POST_KIRA_MAX_PER_SESSION", "8"))   # hard session ceiling
CHAT_POST_KIRA_MAX_LEN = int(os.getenv("CHAT_POST_KIRA_MAX_LEN", "200"))                  # chat messages stay SHORT

# Discord daily-diary webhook (Phase 1). Kira writes an in-character end-of-
# session diary entry that is SAVED for review, NOT auto-posted. Posting to the
# webhook is a deliberate manual action from the dashboard ("Post to Discord").
# DISCORD_AUTOPOST stays false until the tone is trusted over several sessions;
# flipping it true would let the diary fire to the webhook automatically.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_AUTOPOST = os.getenv("DISCORD_AUTOPOST", "false").lower() == "true"

# On-screen captions (Neuro-sama style word-by-word overlay).
# When ENABLE_CAPTIONS=true, a local WebSocket server starts on
# CAPTION_SERVER_PORT and broadcasts Kira's spoken lines (with Azure
# word-boundary timing) to caption_overlay/index.html — add that file as
# an OBS browser source. Fail-graceful: TTS never blocks on caption I/O.
# Only the Azure TTS engine supplies per-word timing; on Edge TTS the
# captions degrade to a single-frame reveal at the line start.
ENABLE_CAPTIONS = os.getenv("ENABLE_CAPTIONS", "false").lower() == "true"
CAPTION_SERVER_PORT = int(os.getenv("CAPTION_SERVER_PORT", "8765"))
CAPTION_CLEAR_DELAY_MS = int(os.getenv("CAPTION_CLEAR_DELAY_MS", "1500"))

# Web dashboard control server (FastAPI + uvicorn, 127.0.0.1 only)
# Separate from the caption WS on 8765. Provides /state, /ws (500ms push),
# /vision/thumbnail, and POST /cmd/{action} for all dashboard commands.
CONTROL_SERVER_PORT = int(os.getenv("CONTROL_SERVER_PORT", "8766"))

# Chat batching config
CHAT_BATCH_WINDOW = float(os.getenv("CHAT_BATCH_WINDOW", "5.0"))
CHAT_RESPONSE_COOLDOWN = float(os.getenv("CHAT_RESPONSE_COOLDOWN", "8.0"))

# ── Activity-aware chat governance (the "activity governor", chat half) ────────
# In a story game, raise the bar so only worth-it chat (direct @Kira / questions)
# interrupts gameplay; in watch-along / just-chatting, stay chat-heavy. Each chat
# message is scored by salience_filter and kept only if its tier clears an
# activity floor. Default OFF — flip the master switch to feel-test one variable
# at a time. First-timers and known regulars always bypass (set in bot.py).
CHAT_SALIENCE_GATE_ENABLED = os.getenv("CHAT_SALIENCE_GATE_ENABLED", "false").lower() == "true"
# Per-activity tier floor a chat message must clear to earn a response.
# Tiers (salience_filter): HIGH > MEDIUM > LOW > DROP. Plain chat scores MEDIUM;
# only @Kira / "?" reach HIGH — so a HIGH floor is what actually gates a game.
CHAT_FLOOR_BY_ACTIVITY = {
    "game":    os.getenv("CHAT_FLOOR_GAME",    "HIGH").upper(),
    "media":   os.getenv("CHAT_FLOOR_MEDIA",   "MEDIUM").upper(),
    "vn":      os.getenv("CHAT_FLOOR_VN",      "MEDIUM").upper(),
    "general": os.getenv("CHAT_FLOOR_GENERAL", "LOW").upper(),
}
# Dedicated manual override for the chat floor — kept SEPARATE from presence_level
# (which governs ONLY the boredom/dead-air rate; the two dials must stay
# decoupled). "none" = use the activity default; "raise" = one tier stricter;
# "lower" = one tier looser (never below LOW, so true-spam DROP stays gated).
CHAT_FLOOR_OVERRIDE = os.getenv("CHAT_FLOOR_OVERRIDE", "none").lower()

# ── Second-stage scaling governor: chat rate cap + fairness ────────────────────
# Sits AFTER the salience floor (above). The floor decides WHICH messages are
# worth-it; this caps HOW MANY distinct chatters actually get a reply per rolling
# minute, so 10 chatters and 1000 chatters cost the same — she physically can't
# exceed the cap. When contended, least-recently-answered wins (the loudest typers
# don't eat the budget). First-timers + known regulars bypass the cap. A ceiling,
# not a brain. Default OFF — feel-test as its own variable, separate from the floor.
CHAT_RATE_CAP_ENABLED = os.getenv("CHAT_RATE_CAP_ENABLED", "false").lower() == "true"
CHAT_RATE_CAP_PER_MIN = int(os.getenv("CHAT_RATE_CAP_PER_MIN", "12"))

# ── Chat-as-advisors + reject-with-reason (Phase G-2, conversation engine) ─────
# When ON, the chat-batch prompt frames chat as her ADVISOR GALLERY, not her
# director: suggestions ("do X", "say Y", backseating) are input she WEIGHS; she
# takes one only when she actually likes it (and owns it as her choice), and she
# may DECLINE one IN CHARACTER with a one-beat reason — a why, not a lecture —
# instead of silently complying or silently ignoring. Streamer register: chat
# informs, she decides. Default OFF — OFF preserves today's chat prompt
# byte-for-byte; feel-test as its own variable per the cadence plan.
CHAT_ADVISORS_ENABLED = os.getenv("CHAT_ADVISORS_ENABLED", "false").lower() == "true"

# Final stale-chat guard: drop any message older than this at RESPONSE time (not at
# drain time). The worker's 60s pre-eviction runs before the turn-lock + the whole
# response pipeline, so messages can age far past it (observed up to ~137s active,
# HOURS after an idle nap) and get answered stale. Measured right before the prompt
# build, this is immune to pipeline/lock/restore/idle timing — she never answers
# anything older than this. Default 180s (a few minutes); lower it to tighten.
CHAT_MAX_AGE_S = float(os.getenv("CHAT_MAX_AGE_S", "180.0"))

# ── "Catch up on chat" — banked-chat surfacing (the heads-down humanizer) ──────
# Even when chat is suppressed (Lock-In heads-down, or a focused game gating chat
# down via the salience floor), the suppressed/gated-out messages are BANKED, not
# dropped — then surfaced in deliberate catch-up beats, like a streamer who plays
# heads-down for a stretch then comes up for air. So nothing is missed; it's batched,
# not lost. A beat fires on this timer OR when Jonny invites it ("what's chat saying?").
# Default ON — this is baseline presence, not a feel-test toggle. Every fire logs loudly.
CHAT_CATCHUP_ENABLED = os.getenv("CHAT_CATCHUP_ENABLED", "true").lower() == "true"
CHAT_CATCHUP_S = float(os.getenv("CHAT_CATCHUP_S", "720.0"))         # ~12 min between timed catch-ups
CHAT_CATCHUP_MAX_MSGS = int(os.getenv("CHAT_CATCHUP_MAX_MSGS", "3")) # best N banked messages surfaced per beat
CHAT_BANK_CAP = int(os.getenv("CHAT_BANK_CAP", "60"))               # max banked messages kept (newest win)

# ── Lock-In heads-down clamp (the one honest situational toggle) ───────────────
# Lock-In is the "shut up and play" nuclear option for immersive games: she goes
# near-SILENT on chat — still RECEIVES, understands, and memory-records every message
# (nothing missed), but does NOT speak to chat. Everything is banked for a catch-up
# beat. The ONLY automatic break-through is a genuinely exceptional message: salience
# score >= this floor. Chat salience tops out at 75 — base 55 + a single bump, and the
# bumps DON'T stack: naming Kira = +20 (→75), a "?" = +15 (→70). So the default 75 means
# only a message that DIRECTLY NAMES Kira pierces heads-down; questions/filler bank. Lower
# it (e.g. 70) to also let bare questions through; raise it (e.g. 999) for absolute
# silence-until-catch-up. She still drives the GAME under Lock-In (Director stays active).
LOCK_IN_BREAKTHROUGH_SCORE = float(os.getenv("LOCK_IN_BREAKTHROUGH_SCORE", "75.0"))

# ── Progress watchdog (objective agency — the quiz-passivity fix) ──────────────
# When Jonny assigns an explicit task ("read the page and answer", "solve the
# quiz") and then goes quiet, Kira ACTS on it after this much silence instead of
# waiting forever. The objective expires after MAX_AGE if never acted on.
# Conservative + explicit-only; default-ON, every act logged loudly.
OBJECTIVE_ACT_SILENCE_S = float(os.getenv("OBJECTIVE_ACT_SILENCE_S", "12.0"))
OBJECTIVE_MAX_AGE_S = float(os.getenv("OBJECTIVE_MAX_AGE_S", "300.0"))

# ── Activity Director (Pass 2 — the first-mover loop; now DEFAULT-ON, ASSERTIVE) ─
# Generalizes the objective watchdog into a proactive driver: when an activity is
# focused, on a HARD min-gap, Kira reacts to fresh perception or fills dead air with
# her own initiative (drives/opinions/bits) instead of waiting to be addressed. Rides
# the same P1 interjection lane (turn-lock + sentence yield; never interrupts Jonny)
# AND the Pass-1 content guardrail (_kira_voice_guardrails + the _speak_single denylist).
# This is now WHO SHE IS, not a feature flag — driving is the baseline, default-ON and
# tuned ASSERTIVE (Neuro-level presence). The dashboard Director toggle is now an
# "ease OFF" lever (turn off → drop to reactive), NOT an on-switch. The min-gap is
# live-tunable from the dashboard (self.director_min_gap_s) so cadence can be pulled
# back mid-stream WITHOUT a restart — the real-time brake on assertive driving.
# Still activity-focused only (stays out of plain hangout). Under Focus/Lock-In she
# keeps driving the GAME but suppresses chat-directed yapping (see chat_lock_in).
ACTIVITY_DIRECTOR_ENABLED = os.getenv("ACTIVITY_DIRECTOR_ENABLED", "true").lower() == "true"
DIRECTOR_MIN_GAP_S = float(os.getenv("DIRECTOR_MIN_GAP_S", "10.0"))    # Neuro-tier default floor between Director utterances (= presence 'normal'; boot default, live-tunable)
DIRECTOR_DEAD_AIR_S = float(os.getenv("DIRECTOR_DEAD_AIR_S", "20.0"))  # silence that triggers "create" (assertive)
# Turn-taking guards (anti-talk-over, empirically tuned from [TurnTiming] data 2026-06-22).
# POST_SPEECH_HOLD: never fire a proactive line within this many seconds of Jonny's last
#   mic speech-frame (_vad_mic_last_ts) — bridges his 0.7-1.3s between-thought pauses so she
#   stops firing 0.1-2.5s into his speech. Keyed on the raw timestamp, NOT _mic_recently_active
#   (which returns False when LOOPBACK_MIC_GATE_ENABLED is off).
# FRESH_MIN_SILENCE: the fresh-vision proactive path must ALSO see this much real silence —
#   stops the Turbo metronome (fresh_ok permanently True) from firing at silence=1s. The
#   dead-air path keeps its own longer DIRECTOR_DEAD_AIR_S gate.
# Raised 3.0 -> 8.0 (2026-06-23, live-feel-tested): the 3s hold let her fire ~5s after
# Jonny spoke, i.e. INTO the pauses of a live conversation (logged since_mic=4.4-4.9s
# fires; "shut up I'm watching a show"). 8s = she holds during conversation but stays
# alive to re-engage. She still fills genuine dead air (DIRECTOR_DEAD_AIR_S=20s, where
# since_mic is naturally huge) but not mid-exchange pauses. Env-tunable.
# FOLLOW-UP (queued): a STATIC hold is the stopgap; the real fix is ADAPTIVE-to-context
# (shorter when he's clearly done, longer mid-thought) - see queued-backlog.
DIRECTOR_POST_SPEECH_HOLD_S = float(os.getenv("DIRECTOR_POST_SPEECH_HOLD_S", "8.0"))
DIRECTOR_FRESH_MIN_SILENCE_S = float(os.getenv("DIRECTOR_FRESH_MIN_SILENCE_S", "3.0"))

# Presence → Director drive-gap presets (C7: presence is the SINGLE cadence dial).
# Confirmed in code (bot.py: Director fires when now - last_fire >= director_min_gap_s):
# LOWER gap = more frequent/yappier, HIGHER = sparser. Picking a presence sets
# director_min_gap_s to its preset on the LIVE Director path; the dashboard gap
# slider then fine-tunes from there (clamp 3-120s). Neuro-tier tight by default.
DRIVE_GAP_CHATTY = float(os.getenv("DRIVE_GAP_CHATTY", "5.0"))    # yappy / max presence
DRIVE_GAP_NORMAL = float(os.getenv("DRIVE_GAP_NORMAL", "10.0"))   # Neuro-tier default
DRIVE_GAP_SLEEPY = float(os.getenv("DRIVE_GAP_SLEEPY", "28.0"))   # sparse / laid-back

# ── Reading the room (INVISIBLE drive-cadence modifier; default OFF) ────────────
# Silently dials HOW MUCH the Director drives based on the BEHAVIORAL texture of the
# interaction (silence, reply terseness, turn-gap, intensity, chat heat) -- NEVER named
# in speech. Cadence-only: the scalar lives at the Director gate + a [RoomRead] log and
# nowhere else (zero speech-leak). Errs toward BACKING OFF (asymmetric): widens readily,
# tightens gently, uncertain -> 1.0. OFF -> multiplier pinned 1.0 -> byte-for-byte today's
# cadence. All feel knobs (no offline calibration -- dial live watching [RoomRead]).
READING_THE_ROOM_ENABLED = os.getenv("READING_THE_ROOM_ENABLED", "false").lower() == "true"
ROOM_TRACKER_N = int(os.getenv("ROOM_TRACKER_N", "6"))                  # rolling window of Jonny's last N voice turns
ROOM_ENGAGED_CHARS = float(os.getenv("ROOM_ENGAGED_CHARS", "120.0"))    # reply length that reads as "fully engaged"
ROOM_QUIET_GAP_S = float(os.getenv("ROOM_QUIET_GAP_S", "90.0"))         # turn-gap that reads as fully heads-down
ROOM_SILENCE_SPAN_S = float(os.getenv("ROOM_SILENCE_SPAN_S", "60.0"))   # current-silence span mapped to silence energy
ROOM_CHAT_BUSY_RPM = float(os.getenv("ROOM_CHAT_BUSY_RPM", "15.0"))     # chat msgs/min that reads as a busy social room
# Combining weights (renormalized over inputs that have data this tick):
ROOM_W_TERSE = float(os.getenv("ROOM_W_TERSE", "0.30"))
ROOM_W_GAP = float(os.getenv("ROOM_W_GAP", "0.25"))
ROOM_W_SILENCE = float(os.getenv("ROOM_W_SILENCE", "0.20"))
ROOM_W_INTENSITY = float(os.getenv("ROOM_W_INTENSITY", "0.10"))
ROOM_W_CHAT = float(os.getenv("ROOM_W_CHAT", "0.15"))
# Asymmetric energy->multiplier map (room_energy: 0=heads-down .. 1=loose):
ROOM_E_NEUTRAL = float(os.getenv("ROOM_E_NEUTRAL", "0.5"))              # energy mapping to multiplier 1.0
ROOM_WIDEN_CEIL = float(os.getenv("ROOM_WIDEN_CEIL", "2.0"))            # max widen (back off) at energy 0
ROOM_TIGHTEN_FLOOR = float(os.getenv("ROOM_TIGHTEN_FLOOR", "0.85"))     # min multiplier at energy 1 (set 1.0 = back-off-ONLY mode)
# Smoothing (build-blocking: dial DRIFTS, never jitters) + absolute caps (dead-air floor):
ROOM_SMOOTH_TAU_S = float(os.getenv("ROOM_SMOOTH_TAU_S", "30.0"))       # EMA time-constant (uses tick_dt; holds at 1s & 5s ticks)
ROOM_MAX_SLEW = float(os.getenv("ROOM_MAX_SLEW", "0.03"))               # max multiplier change PER SECOND
ROOM_DEAD_AIR_MAX_S = float(os.getenv("ROOM_DEAD_AIR_MAX_S", "60.0"))   # true dead air ALWAYS filled within this
ROOM_MIN_GAP_MAX_S = float(os.getenv("ROOM_MIN_GAP_MAX_S", "120.0"))    # effective min-gap ceiling (matches dashboard brake max)

# ── Barge-in yield (turn-arbiter; default OFF) ─────────────────────────────────
# A proactive (Director) turn holds processing_lock through generation AND TTS, but STT
# needs that same lock — so Jonny's voice arriving mid-interjection can't be transcribed
# until the turn ends, and the existing sentence-boundary yield (_execute_interjection)
# is starved. ON: release processing_lock ONLY around the interjection's speak loop so
# STT runs concurrently -> _voice_response_pending sets -> she finishes the current
# sentence and yields, and his speech is transcribed (nothing lost). _active_turn_lock
# stays held throughout (two full turns still can't run at once). OFF -> byte-for-byte
# today's behavior (lock held through the whole interjection; his in-window speech dropped).
# Her REAL reply (P0 speak_streaming) is never affected — only the proactive interjection path.
BARGE_IN_YIELD_ENABLED = os.getenv("BARGE_IN_YIELD_ENABLED", "true").lower() == "true"

# ── Airiness / comedic disposition (behavioral; dialable by config) ────────────
# How airy/weird/unhinged she chooses to be, injected at the generation chokepoint
# (between persona and tool rules) so it shapes EVERY utterance — voice, interjection,
# Director, deep, fallback. It's a DISPOSITION (how she picks what to say), not a persona
# rewrite: her identity/traits/lore/maturity are untouched, and the chaos always loops
# back to her sweet/grounded core (the anchor). 0.0 = off (restrained, ~today's behavior);
# 0.7 = bold default (noticeably airier — dial DOWN from here if it's too much). Edit this
# in .env + restart to re-dial; no code change. Stays inside the content boundary always.
AIRINESS_LEVEL = max(0.0, min(1.0, float(os.getenv("AIRINESS_LEVEL", "0.7"))))

# ── Director self-driven-speech taxonomy (teaching the 5 variants) ─────────────
# Phase 1 (default ON): instead of ONE generic "say something proactive" prompt, the
# Director picks among CALLBACK / NOTICING / PIVOT each beat (cheap priority ladder, no
# extra LLM call), and a LIVE THREAD rail anchors EVERY variant to what you two were
# just on — so noticing is connected, not "the music swelled," and pivot is associative,
# not random. Rides the SAME fire plumbing + guardrails. OFF → byte-for-byte the legacy
# react/dead_air Director. Builds ON the assertive default-on Director + the live brake.
DIRECTOR_TAXONOMY_ENABLED = os.getenv("DIRECTOR_TAXONOMY_ENABLED", "true").lower() == "true"
DIRECTOR_BIT_RIPE_S = float(os.getenv("DIRECTOR_BIT_RIPE_S", "90.0"))  # age before an open bit is "ripe" for a callback payoff
# Cross-session bit fatigue (default ON): running bits persist across streams, but the
# reference-cooldown that stops over-calling was session-scoped — so a bit worn out
# Tuesday was "fresh" again Friday. A small durable sidecar (lore/bit_fatigue.json) tracks
# each bit's LIFETIME invocation count; the existing doubling cooldown is driven by that
# lifetime (de-prioritize ramp), and _ripe_open_bit retires a bit once its lifetime hits
# the threshold (it goes quiet instead of resurfacing). OFF → original session-scoped
# behavior, byte-for-byte. Reuses the bits store + cooldown + _ripe_open_bit (no rebuild).
DIRECTOR_BIT_FATIGUE_ENABLED = os.getenv("DIRECTOR_BIT_FATIGUE_ENABLED", "true").lower() == "true"
DIRECTOR_BIT_RETIRE_CALLBACKS = int(os.getenv("DIRECTOR_BIT_RETIRE_CALLBACKS", "5"))  # lifetime invocations before a bit is retired from proactive callbacks
# Phase 2 — trigger surgery, default OFF (cadence + intensity-gate changes; flip one at a
# time and watch). CONTINUATION: a short-gap self-continue — she extends her OWN line a
# beat after Jonny goes quiet, on its own fast gap (NOT the 15s Director min-gap), with a
# hard streak cap so she never monologues (resets when Jonny speaks). SINCERE DROP: lets
# exactly ONE sincere beat pierce the high-intensity suppression gate, on a long cooldown,
# NEVER during a cutscene — the tonal pivot that currently can't fire mid-chaos.
DIRECTOR_CONTINUATION_ENABLED = os.getenv("DIRECTOR_CONTINUATION_ENABLED", "false").lower() == "true"
DIRECTOR_CONTINUE_GAP_S = float(os.getenv("DIRECTOR_CONTINUE_GAP_S", "2.5"))        # short self-continue gap (floor of the window)
DIRECTOR_CONTINUE_MAX_STREAK = int(os.getenv("DIRECTOR_CONTINUE_MAX_STREAK", "1"))  # max consecutive self-continues before she must yield
DIRECTOR_SINCERE_DROP_ENABLED = os.getenv("DIRECTOR_SINCERE_DROP_ENABLED", "false").lower() == "true"
SINCERE_DROP_COOLDOWN_S = float(os.getenv("SINCERE_DROP_COOLDOWN_S", "240.0"))      # long cooldown — a rare tonal pivot, not a habit

# ── Game-engagement channel (the "activity governor", perception half) ────────
# Opens the perception→speech path during a story game: on a throttle, Kira fires
# a proactive interjection about what she SEES/HEARS on screen, so constant
# vision/audio perception actually becomes presence (it surfaced ~8x all night in
# the 06-17 review). Rides the priority=1 interjection plumbing (turn-lock +
# sentence-boundary yield) so it can NEVER interrupt Jonny's reply. Default OFF.
GAME_REACT_ENABLED = os.getenv("GAME_REACT_ENABLED", "false").lower() == "true"
GAME_REACT_MIN_GAP_S = float(os.getenv("GAME_REACT_MIN_GAP_S", "60.0"))

# Presence dial — probability that a bored-loop line becomes a question to chat.
# One value per presence level (Sleepy / Normal / Chatty). The dial maps onto
# EXISTING mode + carry behavior; these just make the chat-question rate tunable
# per level instead of hardcoded in the observer loop.
ASK_CHAT_P_SLEEPY = float(os.getenv("ASK_CHAT_P_SLEEPY", "0.05"))
ASK_CHAT_P_NORMAL = float(os.getenv("ASK_CHAT_P_NORMAL", "0.15"))
ASK_CHAT_P_CHATTY = float(os.getenv("ASK_CHAT_P_CHATTY", "0.25"))
# Threshold multiplier per presence level — scales the observer-loop silence
# thresholds. >1 = waits longer before speaking unprompted (sleepier);
# <1 = fills dead air sooner (chattier).
PRESENCE_THRESHOLD_MULT_SLEEPY = float(os.getenv("PRESENCE_THRESHOLD_MULT_SLEEPY", "1.8"))
PRESENCE_THRESHOLD_MULT_NORMAL = float(os.getenv("PRESENCE_THRESHOLD_MULT_NORMAL", "1.0"))
PRESENCE_THRESHOLD_MULT_CHATTY = float(os.getenv("PRESENCE_THRESHOLD_MULT_CHATTY", "0.8"))
ENABLE_CHATTER_MEMORY = os.getenv("ENABLE_CHATTER_MEMORY", "true").lower() == "true"
# Twitch native polls require affiliate status. Set true only if you're affiliate;
# otherwise [POLL:] tags get stripped silently (no failed API call).
ENABLE_TWITCH_POLLS = os.getenv("ENABLE_TWITCH_POLLS", "false").lower() == "true"

# Inference backend for local-Llama-style calls (triage, classification,
# emotion analysis, response generation fallback). Claude (Sonnet/Opus) is
# routed separately and is NOT affected by this flag.
#   "groq"  -> Groq cloud (llama-3.1-8b-instant). Saves ~6-7 GB VRAM.
#   "local" -> local llama_cpp GGUF (legacy path).
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "5.0"))
# Fallback behavior when INFERENCE_BACKEND=groq and Groq fails. Default "false"
# is AAA-SAFE: a transient Groq blip degrades that ONE turn loudly + gracefully
# (triage defaults to RESPOND; her voice reply is Claude, unaffected) and NEVER
# parks a 6-7 GB Llama on the GPU. The non-"false" modes load that Llama — a
# one-way ratchet that never frees, tipping a 16 GB card over mid-stream (the
# slideshow regression). Set non-false only if you knowingly want cloud-outage
# resilience and have the VRAM headroom.
#   "false"      -> raise; callers degrade gracefully. No VRAM cost. (DEFAULT, AAA-safe)
#   "true"       -> local Llama kept loaded at startup as a warm fallback (~6-7 GB, always).
#   "lazy_load"  -> local Llama loaded into VRAM on first Groq failure and kept the
#                   rest of the session (~6-7 GB, never freed — the VRAM trap).
GROQ_FALLBACK_TO_LOCAL = os.getenv("GROQ_FALLBACK_TO_LOCAL", "false").lower()

# Hybrid Brain
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Canonical Claude model-name constants — single source of truth. Reference these
# (CLAUDE_SONNET_MODEL / CLAUDE_OPUS_MODEL) instead of hard-coding model strings.
# Env var names are unchanged (CLAUDE_DEEP_MODEL / CLAUDE_CHAT_MODEL) so .env stays valid.
CLAUDE_OPUS_MODEL   = os.getenv("CLAUDE_DEEP_MODEL", "claude-opus-4-7")
CLAUDE_SONNET_MODEL = os.getenv("CLAUDE_CHAT_MODEL", "claude-sonnet-4-6")
# Haiku — cheapest tier, for high-frequency structured calls (per-turn memory
# extraction) where Sonnet-grade reasoning isn't visible in the output.
CLAUDE_HAIKU_MODEL  = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5")
# Legacy aliases — keep existing import sites working unchanged.
CLAUDE_DEEP_MODEL = CLAUDE_OPUS_MODEL
CLAUDE_CHAT_MODEL = CLAUDE_SONNET_MODEL
ENABLE_CLAUDE_BRAIN = os.getenv("ENABLE_CLAUDE_BRAIN", "true").lower() == "true"
ENABLE_CLAUDE_CHAT = os.getenv("ENABLE_CLAUDE_CHAT", "true").lower() == "true"
ENABLE_PROMPT_CACHING = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
ENABLE_CLAUDE_STREAMING = os.getenv("ENABLE_CLAUDE_STREAMING", "true").lower() == "true"

# AudD audio fingerprinting (paid API; only fires on explicit user song-ID intent).
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN", "")

# Storytime / Puppet Show (pre-generated shadow-puppet shows).
# Image generation uses Google Gemini 2.5 Flash Image ("Nano-Banana") for its
# style/character consistency. The key below is a FRESH Google AI Studio key,
# SEPARATE from GOOGLE_API_KEY (which is Custom Search). Do not conflate them.
GEMINI_IMAGE_API_KEY = os.getenv("GEMINI_IMAGE_API_KEY", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
# Which image provider the Storytime pipeline uses. Swappable: the client is a
# provider-agnostic module, so this can change without touching the orchestrator.
STORYTIME_IMAGE_PROVIDER = os.getenv("STORYTIME_IMAGE_PROVIDER", "gemini").lower()

# Cutscene-aware observer suppression — only active during ACTIVITY_GAME mode.
# When true, the observer loop and per-turn triage check for cutscene cues
# (vision scene summary + audio mood keywords) and suppress interjections while
# a cutscene is likely playing. Has zero effect outside ACTIVITY_GAME mode.
CUTSCENE_AWARE = os.getenv("CUTSCENE_AWARE", "true").lower() == "true"

# VTube Studio integration — drives Live2D facial expressions from Kira's emotional state.
# Default ON. Token is auto-created on first approved handshake and persisted to disk.
ENABLE_VTS_EXPRESSIONS = os.getenv("ENABLE_VTS_EXPRESSIONS", "true").lower() == "true"
VTS_WS_URL = os.getenv("VTS_WS_URL", "ws://localhost:8001")
VTS_PLUGIN_NAME = os.getenv("VTS_PLUGIN_NAME", "Kira AI")
VTS_PLUGIN_DEVELOPER = os.getenv("VTS_PLUGIN_DEVELOPER", "JonnyD")
VTS_TOKEN_PATH = os.getenv("VTS_TOKEN_PATH", os.path.join(_REPO_ROOT, ".vts_token"))

# Chess Mode (Phase 1) — Kira plays Lichess against Stockfish, on stream.
# Disabled unless armed from the dashboard. Requires a SEPARATE Lichess BOT
# account + a bot:play token (see README for the one-time upgrade step). The
# engine is a local Stockfish binary, Elo-capped so Kira plays (and blunders)
# like a human club player. CPU only — no GPU contention with vision/STT.
LICHESS_BOT_TOKEN = os.getenv("LICHESS_BOT_TOKEN", "")
CHESS_ENGINE_PATH = os.getenv("CHESS_ENGINE_PATH", "stockfish.exe")
CHESS_KIRA_ELO    = int(os.getenv("CHESS_KIRA_ELO", "1400"))
CHESS_MOVETIME_MS = int(os.getenv("CHESS_MOVETIME_MS", "150"))

# Auto-clip pipeline (offline post-session) — scripts/cut_clips.py turns Kira's
# clip-candidate artifacts into pre-cut, pre-titled video files. Runs AFTER a
# stream, never during. OBS_RECORDINGS_DIR is where OBS writes local recordings;
# cut clips land in OBS_RECORDINGS_DIR/clips/YYYY-MM-DD/.
OBS_RECORDINGS_DIR = os.getenv("OBS_RECORDINGS_DIR", "")

# OBS recording-start anchor (clip alignment, opt-in). When enabled, Kira queries OBS
# WebSocket at Go-Live and logs a `recording_start` event giving the clip cutter a
# guaranteed shared clock with the video (vs the creation_time / Whisper fallback).
# Default OFF + fully graceful: if disabled or OBS is unreachable, alignment uses
# today's fallback unchanged. Requires OBS → Tools → WebSocket Server enabled.
OBS_RECORD_ANCHOR_ENABLED = os.getenv("OBS_RECORD_ANCHOR_ENABLED", "false").lower() == "true"
OBS_WEBSOCKET_URL = os.getenv("OBS_WEBSOCKET_URL", "ws://127.0.0.1:4455")
OBS_WEBSOCKET_PASSWORD = os.getenv("OBS_WEBSOCKET_PASSWORD", "")
# Window around the detected moment: lead-in before (setup) + payoff after.
# Widened 2026-06-13 after reel review: pre 5→6.5s, post 3→3.5s (cuts ran a
# touch early on both ends). Override via env or --pre/--post CLI flags.
# Min clip length is 12s (enforced in clip_cutter.py regardless).
CLIP_PRE_SECONDS   = float(os.getenv("CLIP_PRE_SECONDS", "6.5"))
# Tail after the punch. Was 3.5 → bolted a 3.5s dead-air tail onto every clip (the
# anchor is logged at the END of Kira's line, so +post is pure dead air). Dropped to
# 0.5 to kill that tail today; the precise punch-landing out-cut is Phase-3 per-clip
# anchoring. Env-tunable (CLIP_POST_SECONDS).
CLIP_POST_SECONDS  = float(os.getenv("CLIP_POST_SECONDS", "0.5"))

# ── Asymmetric cut rule (Clip Phase 3 — the watchability fix) ──────────────────
# CLIP_PRE/CLIP_POST above are a FIXED window padded symmetrically around the matched
# anchor. Phase 3 instead anchors each clip to the actual comedic beats: OUT hard on the
# punch (Kira's response end — kira_response is logged at end-of-utterance) and IN just
# before the SETUP line (the poke that preceded her response in events.jsonl), trimmed
# tight. The 2-4s response-latency beat between setup and answer is KEPT — that "...what's
# she gonna say?" tension is content, not dead air — because everything between in and out
# is preserved; only the BUFFERS are asymmetric (tight front, hard-ish out). Applied only
# when BOTH a punch (a matched kira_response) AND a setup resolve cleanly and the span is
# plausible; otherwise the cutter degrades to the CLIP_PRE/CLIP_POST fixed window and logs
# the fallback. The 12s min-clip floor (clip_cutter.py) grows the FRONT, never the tail, so
# the hard out-cut on the punch survives. All three env-tunable.
CLIP_SETUP_BUFFER_S = float(os.getenv("CLIP_SETUP_BUFFER_S", "1.0"))   # tight lead-in before the setup line
CLIP_PUNCH_TAIL_S   = float(os.getenv("CLIP_PUNCH_TAIL_S", "0.5"))     # hard-ish tail after the punch lands
CLIP_MAX_EXCHANGE_S = float(os.getenv("CLIP_MAX_EXCHANGE_S", "90.0"))  # setup→punch span over this ⇒ bad match, use fixed window

# ── Clip output shaping (Phase 4 — the four labeled outputs) ───────────────────
# Minimum output clip length. Floor protects against unusable stubs; DROP IT for
# short-form one-liners. In asymmetric mode the floor grows the FRONT (earlier
# in-point), never the tail, so the hard out-cut on the punch is preserved.
CLIP_MIN_SECONDS = float(os.getenv("CLIP_MIN_SECONDS", "12.0"))
# (b) best-of reel: pick clips top-by-score until cumulative length reaches this cap.
# Phase-K target is a 3-4 minute best-of, so the default cap is 4 min.
CLIP_REEL_MAX_SECONDS = float(os.getenv("CLIP_REEL_MAX_SECONDS", "240.0"))
# (c) highlight VOD scaled to session length (Phase K item 5): the chronological
# body targets FRACTION x session span, clamped to MAX; lowest-score clips are
# dropped (loudly) until it fits. 0 fraction = no scaling beyond the max cap.
CLIP_HIGHLIGHT_FRACTION = float(os.getenv("CLIP_HIGHLIGHT_FRACTION", "0.10"))
CLIP_HIGHLIGHT_MAX_SECONDS = float(os.getenv("CLIP_HIGHLIGHT_MAX_SECONDS", "900.0"))
# (c) highlight-VOD cold-open teaser: the N punchiest clips, each trimmed to this
# many seconds (the tail-end landing on the punch), spliced BEFORE the chronological
# body as a rapid hook — snippets, never full-clip duplicates.
CLIP_TEASER_COUNT = int(os.getenv("CLIP_TEASER_COUNT", "3"))
CLIP_TEASER_SECONDS = float(os.getenv("CLIP_TEASER_SECONDS", "0.5"))

# (d) vertical shorts (Phase K item 1): the top-N scored clips rendered 9:16
# 1080x1920 (blur-pad reframe, full frame preserved) with BURNED-IN captions
# (.ass from faster-whisper on each short's own audio). Clips longer than the
# max are trimmed to a punch-landing tail window rather than skipped.
CLIP_SHORTS_COUNT = int(os.getenv("CLIP_SHORTS_COUNT", "5"))
CLIP_SHORT_MAX_SECONDS = float(os.getenv("CLIP_SHORT_MAX_SECONDS", "60.0"))

# Minimum session length (minutes) below which the reel is skipped.
# Also requires at least 3 aligned candidates. Override via env.
REEL_MIN_MINUTES   = int(os.getenv("REEL_MIN_MINUTES", "20"))
# Recording container extensions to scan, comma-separated.
CLIP_VIDEO_EXTS    = os.getenv("CLIP_VIDEO_EXTS", ".mkv,.mp4,.flv,.mov,.ts")

# ── YouTube auto-connect ──────────────────────────────────────────────────────
# Set YOUTUBE_CHANNEL_ID in .env to enable polling on boot.  Kira will poll
# the YouTube Data API v3 search.list endpoint every YT_AUTO_CONNECT_POLL_S
# seconds for up to YT_AUTO_CONNECT_TIMEOUT_S total (default: every 60s /
# 15 minutes), then give up and wait for a manual connect via the dashboard.
YOUTUBE_CHANNEL_ID        = os.getenv("YOUTUBE_CHANNEL_ID", "")
YT_AUTO_CONNECT_TIMEOUT_S = int(os.getenv("YT_AUTO_CONNECT_TIMEOUT_S", "900"))   # 15 min
YT_AUTO_CONNECT_POLL_S    = int(os.getenv("YT_AUTO_CONNECT_POLL_S", "60"))

# ── Chat queue / instrumentation ─────────────────────────────────────────────
# ACK_THRESHOLD_S: if the oldest pending chat message has been waiting longer
# than this (seconds), a brief acknowledgment directive is injected into the
# next response prompt so the chatter's name gets a quick mention while Kira
# finishes the current turn.
ACK_THRESHOLD_S = float(os.getenv("ACK_THRESHOLD_S", "20.0"))

# Talk-budget governor — off by default until instrumentation data is collected.
# When enabled, per-chatter response counts are used to bias batch ordering so
# chatters with fewer responses get answered first.
CHAT_BUDGET_ENABLED       = os.getenv("CHAT_BUDGET_ENABLED", "false").lower() == "true"
CHAT_BUDGET_RESPOND_ALL_N = int(os.getenv("CHAT_BUDGET_RESPOND_ALL_N", "5"))

# ── Phrase throttle ───────────────────────────────────────────────────────────
# Prevents distinctive constructions from becoming templates by tracking per-
# session n-grams and injecting a soft do-not-reuse constraint into every LLM
# prompt when a phrase hits PHRASE_THROTTLE_THRESHOLD uses.
# PHRASE_THROTTLE_WATCHLIST: comma-separated list of specific phrases to monitor
# even if they wouldn't surface from the n-gram statistics (e.g. short idioms).
PHRASE_THROTTLE_ENABLED   = os.getenv("PHRASE_THROTTLE_ENABLED", "true").lower() == "true"
PHRASE_THROTTLE_THRESHOLD = int(os.getenv("PHRASE_THROTTLE_THRESHOLD", "2"))
# How many recent responses the catchphrase throttle remembers. The old 40 ≈ ~7 min of reactions, so a
# bit reused every ~10 min fell out of the window between uses and never tripped — fine for short chats,
# too short for a multi-HOUR Pokémon playthrough. 120 ≈ ~20 min; BATCH 6 PHASE 6 bumps the default to
# 200 ≈ ~33 min for a 30-hr run, so a bit reused every half-hour still trips (n-gram rebuild stays
# trivial at this size). Core anti-repetition -> this helps her EVERYWHERE (idle chat, other games), not
# just Pokémon (One-Kira firewall: a general soul property, not a mode hack). Env-tunable per session.
PHRASE_THROTTLE_CAPACITY  = int(os.getenv("PHRASE_THROTTLE_CAPACITY", "200"))
PHRASE_THROTTLE_WATCHLIST = [
    p.strip() for p in os.getenv(
        "PHRASE_THROTTLE_WATCHLIST",
        "three words and a vibe,and a vibe,I respect it,I respect the commitment,I'll wait",
    ).split(",") if p.strip()
]

# Deterministic cooldown (seconds) for the word-count narration tic — the
# "three words and a vibe" / "one word and a vibe" family. Once Kira fires the
# construction, _kira_voice_guardrails hard-bans it (and any narration of the
# user's brevity) for this many seconds. Separate from the n-gram throttle so
# it catches every variant, not just exact repeats.
FRAGMENT_QUIP_COOLDOWN_S = int(os.getenv("FRAGMENT_QUIP_COOLDOWN_S", "240"))

# Reference (running-bit) cooldown. When Kira invokes a running bit, it goes on a
# cooldown that DOUBLES each reuse (base, 2x, 4x, …) capped at MAX, and is omitted
# from the performance prompt while cooling so she doesn't lean on it. Resets at
# stream end (the bits themselves are durable). BIT_REF_MATCH_MIN_RATIO is the
# detection STRICTNESS knob: the fraction of a bit's distinctive name-words that
# must appear in her line to count as "invoked". 1.0 = all of them (conservative,
# under-detects on purpose — the safe direction); lower it (e.g. 0.6) to catch
# looser references. Dial this like the react-gap from stream observation.
BIT_REF_COOLDOWN_BASE_S = int(os.getenv("BIT_REF_COOLDOWN_BASE_S", "180"))
BIT_REF_COOLDOWN_MAX_S = int(os.getenv("BIT_REF_COOLDOWN_MAX_S", "1800"))
BIT_REF_MATCH_MIN_RATIO = float(os.getenv("BIT_REF_MATCH_MIN_RATIO", "1.0"))
# De-dupe window for _stamp_bit_invocation: a bit stamped within this many seconds is
# NOT re-stamped. Stops double-penalising when the Director fires a callback (stamps the
# bit) AND her resulting spoken line then word-matches the same bit a beat later.
BIT_STAMP_DEDUP_S = float(os.getenv("BIT_STAMP_DEDUP_S", "10.0"))

# ── Chess owner / data dir ────────────────────────────────────────────────────
# Challenges from CHESS_OWNER_LICHESS_ID (case-insensitive) are Jonny's own
# practice games and do NOT trigger the spectate embed on stream.
CHESS_OWNER_LICHESS_ID = os.getenv("CHESS_OWNER_LICHESS_ID", "Militele3")

# Local state files (gitignored) — lifetime chess stats, etc.
DATA_DIR = os.path.join(_REPO_ROOT, "data")
