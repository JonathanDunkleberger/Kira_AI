# NEXT_SESSION — resume prompt (write date 2026-07-07, POST-CREDITS PHASE 1 — WATCH RIG landed)

Paste this to the fresh session:

---

RESUME — fresh session. The Sherpa timeline is SUMMITED (credits rolled 2026-07-07;
canonical = hall_of_fame, SACRED, never touched). This session built the **POST-CREDITS
PHASE 1** deliverables. Read STATE §0 top (the GO PROCEDURE + the new SOUL-ON WATCH RIG
block) first. Commits this session: 7a416ef, 93ee1f9, 3df63d1, 25d8426 (+ docs).

## WHAT LANDED (all committed, working tree clean on these)

**PHASE A — GEMINI VISION MIGRATION (7a416ef, VERIFIED).** kira/ eyes off OpenAI
gpt-4o-mini onto Google Gemini (gemini_vision.py = single chokepoint; flash-lite
heartbeat / flash-preview escalation). Smoke-tested live: synthetic frame graded exact,
verbatim OCR exact, 10-token classify clean, live screen described accurately. This was
Jonny's uncommitted WIP for days — now landed. **kira/* is otherwise HIS — never sweep.**

**PHASE B — THE SOUL-ON WATCH RIG (93ee1f9, verified headless).** `pokemon_agent/watch.py`
— ONE command to watch her play soul-on from ANY banked point. Picker / `--at <alias>` /
`--canonical` / `--list` / `--clean`. THREE guarantees: (1) preflights :8766, REFUSES
LOUDLY if the bot is down (tonight's 10061 soul-blind pop-in is now impossible by
accident); (2) copies the banked bundle to a DISPOSABLE sandbox + points play_live there
via `POKEMON_CAMPAIGN_DIR` (campaign.py honors it) — canonical never written, hard-kill-
safe; (3) `--roam-seconds` (24h default) so a watch runs till Ctrl-C.

**PHASE D — LAG FIX + NAMING + POST-GAME DETECT (3df63d1, verified).** (b) `POKEMON_VOICE_ASYNC`
(default on): reaction POSTs → worker thread, is_speaking → cache, oracle choose → pumps
frames — no render stutter. (c) Named the whole finale in SEED_NODES + _PLACE_NAMES (ids
VERIFIED vs pret disasm) — "(1,80) an unfamiliar area" → "the Hall of Fame". (a) post_game
detection (Champion flag OR 8-badges-in-the-Hall) in read_live_state.

**PHASE C — GRIEF-ON-FAINT beat (25d8426).** Roster-as-relationship soul-debt: a teammate
going down on a win now gets a bittersweet T2 pang. (Wedge-recovery lines / clutch-pride /
judged-catch voicing were already wired — verified.)

## THE WATCH — JONNY'S EYES-ON (the one needs-eyes item)

The rig is STAGED. To watch the E4 soul-on with a notepad:
1. `python run.py`  (own terminal — boots bot/TTS/VTS/mic; wait for :8766)
2. `python pokemon_agent/watch.py --at pre-e4`   (or `--canonical` for the summit victory lap)
His notes → the next batch.

## STILL OWED (flagged, needs eyes or Jonny sign-off — NOT shipped blind)

1. **SUMMIT-WATCH strand (a, residual):** post-game detection+naming shipped, but whether she
   autonomously WALKS OUT of the league (vs idles in the Hall) needs the summit watch to
   confirm. Scoped fix if she idles: a post-game "exit league → Cerulean Cave" objective +
   heal (team's at 1/6 from the final stand). Pre-credits spawns won't strand (normal gym objs).
2. **#12 dialogue first-timer REACTION generation** — the reaction quality lives in core
   `_pokemon_react` (bot.py) → identity firewall → needs Jonny's sign-off, not a mode-side edit.
3. **Lapras first-field set-piece** + **sticking nicknames** (Name-Rater/AAAAAAAAAA errand).
4. **Portable-engine harvest + prune** (rules 14/prune) — untouched this session; the recon_*
   one-shot fleet in pokemon_agent/ (untracked) + NIGHT_REPORT.md (modified) are night-loop
   leftovers, still to prune per the summit NEXT_SESSION list.

**WORKING-TREE LAW:** kira/* = Jonny's (Gemini now landed; future kira edits are his). The
untracked pokemon_agent/recon_*.py fleet + NIGHT_REPORT.md are prune candidates, not mine to commit.

---

WATCH STATUS: canonical bank is CLEAN (hall_of_fame — Champion, SACRED/untouched). The watch
rig is GO: `python run.py` then `python pokemon_agent/watch.py` — pick any point (summit for the
victory lap, pre-e4 for the gauntlet) and watch her play it soul-on, true speed, her voice live.
