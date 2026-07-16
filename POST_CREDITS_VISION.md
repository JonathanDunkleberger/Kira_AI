# POST_CREDITS_VISION.md — the blue-sky parking lot

**STATUS: PARKING LOT, NOT A WORK ORDER.** Nothing here is built until FireRed credits roll.
This exists so good ideas born during the climb are banked, not forgotten. The mountain is the
job; this is the view from the summit. (CLAUDE.md rule 16 — CEO directive.)

---

## 1. WATCH-PARTY 2.0 — restraint-first companion
The thesis: for passive media the **timing engine beats the content engine**. A friend who reacts
to the RIGHT three moments in an episode feels more present than one who talks over every scene.
- Moment-classifier: tension / comedy / beat / lull → each gates a different interjection budget.
- Interjection budget (spend fewer, better) instead of a fixed cadence.
- Scene-aware Gemini vision (the migrated core-eyes): what's ON SCREEN shapes when she speaks.
- **Carry over from Pokémon:** RoomRead (dead-air/place-reactive), salience gating, arc pacing,
  decision-points-as-beats (a battle choice ≈ a plot turn — react to the pivot, not the filler).

## 2. SHARED-HISTORY CONTINUITY — "she remembers watching it with you"
The differentiator no other AI-VTuber has: episodic **media** memory.
- What you watched together, her theories about it, callbacks when a theory resolves ("I called
  this in episode 2").
- **Carry over from Pokémon:** the sentiment ledger, called-shots machinery, journey-continuity
  patterns (`journey_core.json` saga promote/decay) — already proven to produce hr-28 callbacks.

## 3. LATENCY PROGRAM — post-credits surgery list (target: sub-1s perceived short-turn)
- Async the `pokemon_voice` render-thread HTTP (KNOWN, already queued — STATE §7.4): the blocking
  `urllib` decision/emit calls on the main render thread freeze frame+music per LLM call.
- Speculative first-clause TTS (start speaking the opening while the rest generates).
- VAD-close tightening (shave the trailing silence before her turn starts).

## 4. THE EVOLUTION EVENT — the upgrade as a public narrative moment
Post-credits, roll every Pokémon-mode lesson into core Kira as a STREAMED milestone: "Kira leveled
up." The play mode was the training arc; core Kira is who she becomes. Make the promotion visible.

---

## 6. PRODUCTION-VALUE DOCTRINE — the "no way this is real" tier
One-line test for any future feature: **does it make a first-time viewer say "there's no way this is real"?**
Five pillars:
- **(a) Zero-jank reliability as the fourth wall** — the stuck-spin breaker, latency, render-thread fix are
  SHOW quality, not just infra. A single visible freeze breaks the illusion harder than any weak line.
- **(b) Lore depth** — unprompted long-range callbacks to her own journey are the signature "holy shit"
  moment; powered by continuity memory (item 2).
- **(c) Broadcast polish** — HUD/overlay visual grammar that reads as a SHOW, not a debug window.
- **(d) Pacing** — decision-points-as-beats, restraint, quiet stretches so the payoffs land.
- **(e) Emergent real stakes** — the drama is unscripted because the autonomy is REAL. Harness quality IS
  show quality: the better she actually plays unaided, the realer the stakes read.

## 5. LESSONS LEDGER — Pokémon mechanisms that are candidate core-Kira promotions
*(rule 12 discovery-lab principle. Append one line each as the climb surfaces them. These are the
mechanisms that would make her better EVERYWHERE, not just in FireRed.)*

- **Verify the EFFECT, not the byte** (2026-07-05): keystone-2 "verified party cursor to slot 1" by reading
  the byte value, but the real selection never moved — a false verify. Core-Kira analogue: never mark a
  feature WIRED because a value is *set*; confirm the downstream *behavior* changed (the three-state
  discipline, generalized). A candidate lint for the whole codebase, not just Pokémon.

## 7. "PLAYED LIKE NINTENDO INTENDED" — the full-completionist showcase format (Jonny, couch session 2026-07-08)
A future episode FORMAT, not a capability: Kira plays a run the way the designers dreamed —
guides open, every secret, every side quest, full dex energy, reading the world like a
superfan. The inverse register of the first-timer showcase: instead of wonder-at-the-unknown,
mastery-as-love-letter. Banked as P-4 from couch fix-pass 1; no build until the showcase
format itself is proven.

## 8. NATIVE MEMORY / BOND / EMOTIONAL-CONTINUITY for the run (Jonny, 2026-07-11) — north-star, PHASE 3+
A STANDALONE, NATIVE memory system built HERE in Kira-local (KiraState + the soul hooks / pokemon_soul.py
substrate), FOR the playthrough. **Do NOT copy from the web app — the flow is Kira-local → web app later, never
the reverse.** She accumulates REAL LIVED MEMORIES across the ~30h run — the starter pick, the first catch, the
bosses she hated, the grind sessions, the exact level-up that unlocked a win, the mon that clutched a loss — and
REFERENCES them live as she plays ("remember when this little guy could barely scratch anything?"), culminating in
an **endgame recap / payoff** that lands because the memories are real and earned, not scripted. Distinct from the
existing per-run soul continuity: this is a deliberate, lived-memory ACCUMULATION + live-recall + payoff arc, native
to the project. Sequenced AFTER the autonomous-credits asymptote + the long-duration live-soak (it's a soul-depth
capability, not a blocker to the beat-the-game build). Candidate to later flow OUTWARD to the web app.

## 9. CHAT MODERATION / PRESENCE — she bans with teeth (Jonny, 2026-07-11) — north-star, PHASE 3+
Wire the Twitch mod-action integration so Kira has REAL authority + presence in HER OWN chat: she can
ban/timeout viewers herself, with sassy-endearing-but-with-teeth energy ("keep it up and I'll ban you 💅").
**Banning for vibe / bit / annoyance reasons is FINE and is GOOD CONTENT** — even an occasional playful WRONG-ban
is content, part of her presence, not a bug. Give her the ban power; it makes the chat feel like her room.
**ONE HARD GUARDRAIL (the real risk is a MISS on the dangerous category):** Kira is NOT the sole line against
genuinely serious harm — **CSAM, doxxing, credible threats, hate-raids stay backstopped by HUMAN mods.** So:
Kira bans FREELY for moderation/fun/bit; human mods hold the floor on the serious stuff. Build her the authority +
presence; keep humans as the safety net for the dangerous edge cases. Sequenced PHASE 3+ (post-asymptote,
post-live-soak); needs the live stream + a Twitch mod-scope integration, so it's inherently a live/supervised build.
