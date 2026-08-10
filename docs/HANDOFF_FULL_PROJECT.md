# HANDOFF — Local Kira × Web App × FireRed Marathon (Jonny D)

**Write date:** 2026-08-08  
**Author context:** Cloud agent session on `cursor/endgame-credits-path-bdbc` after overnight endgame work (Articuno caught; Zapdos/E4 path re-wired).  
**Who this is for:** The next coding agent (Cursor, Claude Code fork, or similar) picking up Jonny’s dual-product world without rediscovering pain.

Read this whole document before changing code or asking Jonny to type more than one resume command.

---

## TL;DR (read this first)

Jonny D builds **Kira** — an AI VTuber / companion that *actually talks to chat* and (in the local product) **plays Pokémon FireRed by herself**. The stream is growing because the product is fun and rare: Neuro-sama–class energy, but with real conversation + an autonomous FireRed run. The stream is also the **funnel** into a parallel **web SaaS** (premium “Shangri-La / concierge” companion product) meant to make money.

**Local OG Kira (this repo)** = the hybrid desktop product: voice, Twitch/YouTube chat, VTube Studio, memory, and the FireRed harness.  
**Web app (separate repo)** = the monetizable companion product people discover because the stream slapped.

**Current FireRed live state (as of soak `20260807_202018`):**
- **8 badges** — Elite Four is the next story wall
- Party: **Blastoise L69, Lapras L26, Moltres L50, Articuno L50, Kadabra L19, Diglett L18**
- **Moltres ✅ · Articuno ✅ · Zapdos ❌** (never actually caught — narration lied; chat called it)
- Location last night: **Fuchsia / southern Kanto**, crashed every `victory_lap` tick on missing `_zapdos_north_staging`
- **Fix shipped:** commit `9e4eeec` on branch `cursor/endgame-credits-path-bdbc` — Zapdos north staging + Diglett boxed / Kadabra kept + E4 grind path

**Target E4 six:** Blastoise · Zapdos · Articuno · Moltres · Lapras · **Kadabra** (box Diglett; no Eevee detour).

**Jonny’s one PC resume command:**
```powershell
taskkill /F /IM python.exe /T
cd G:\JonnyD\NeuroAI_Bot
git fetch origin
git checkout cursor/endgame-credits-path-bdbc
git pull
git log -1 --oneline
# expect: 9e4eeec fix(endgame): wire Zapdos north path + Kadabra E4 six for credits
powershell -ExecutionPolicy Bypass -File pokemon_agent\resume_marathon.ps1
```

**North star:** Frankfurt Test — Jonny starts stream, says hi, leaves (literally could fly first class), comes back, and she’s still live: narrating, talking to chat, closing on credits. Then clip it, funnel to the web app, sell the companion.

---

## 1. Who / what / why

### People
- **Jonny D** (Jonathan Dunkleberger) — owner, streamer, product brain. Non-jargon about internals; extremely sharp on product symptoms. Trust his stream observations; they’ve been right repeatedly.
- **Kira** — the character / product. Local “OG Kira” is the full hybrid (voice + game + chat). Web Kira is the SaaS companion cut of the same soul.
- **Chat** — Twitch / YouTube. They are part of the product. Kira talks to them. That’s a deliberate competitive edge.

### Positioning (say this out loud)
- **Neuro-sama is #1.** We are not trying to dethrone her with cope. The ambition is **clear #2** — same “AI VTuber who plays and entertains” energy, different wedge.
- **Wedge vs Neuro:** Neuro has so much scale that she often **can’t really talk to chat**. Kira **does** — she answers people, holds the room, and right now she’s also **solo-running FireRed** with legendary catches that make chat explode.
- **Web brand vibe:** premium **concierge / Shangri-La** — not cheap chatbot sludge. Warm, high-touch, “you’re somewhere special,” not purple-glow AI cliché. The stream proves the soul is real; the site sells living with her.
- **Business loop:** Fun stream → attention + clips → funnel to web app → paid companion SaaS. Art first, then money. Don’t ship a boring funnel product that kills the vibe.

### Two products, one soul

| | **Local OG Kira** (this repo) | **Web app** (parallel) |
|---|---|---|
| Path (PC / Mac examples) | `G:\JonnyD\NeuroAI_Bot` (Windows runtime) | Separate repo (historically `…/Kira/Kira_App` on Mac) |
| GitHub | `JonathanDunkleberger/Kira_AI` | Separate web repo (kira-web / xoxokira.com stack) |
| What it is | Desktop hybrid: mic/STT, TTS, Twitch/YT, VTS, memory, **FireRed agent** | Hosted companion SaaS people subscribe to |
| Job today | Marathon stream that people can’t stop watching | Convert stream attention into revenue |
| Funnel URL | Pins / plugs → **xoxokira.com** | Landing + product |
| Shared | Same OpenRouter key family, same Kira identity — **never fork the soul** |

If you only work in this repo, still remember: every stream fix is also a **go-to-market** fix for the web app.

---

## 2. Vision & definition of done

### The Frankfurt Test (local mission)
From `docs/marathon-stream-plan.md`:

Jonny clicks Start Streaming, says hi, and leaves. He can fly to Frankfurt, drink champagne, sleep, come back — and Kira is still live: playing FireRed at a human let’s-play pace, narrating in character, talking to chat, closing on the Elite Four. No dead air, no visible crash, no babysitting. **Start of stream → Hall of Fame credits, fully autonomous.**

**Watchability bar:** a stranger can watch any 20 minutes with dead chat and not click away — sustained for the whole marathon.

### The business test (web mission)
The marathon (and the daily FireRed streams) are the attention engine. Clips from the VOD are the ad campaign. Viewers who fall in love with “she’s actually talking to me while catching Zapdos” hit the site and pay for a private companion experience with Shangri-La polish.

### Engine proof (already done once)
`fresh_go_6` (2026-07-15, `docs/RUN_STATS_fresh_go_6.md`): bedroom → 8 badges → E4 → **Hall of Fame**, autonomously (headless ~14×). The engine can beat the game. The live problem is **watchable, chat-aware, true-speed, crash-resilient show** — and right now, finishing **this** live canonical run’s victory lap without getting lost.

---

## 3. Repo map & environments

### Git
- **Repo:** https://github.com/JonathanDunkleberger/Kira_AI  
- **Preferred base branch:** `main`  
- **Active endgame work branch:** `cursor/endgame-credits-path-bdbc` (PR #6)  
- **Cloud agent runs:** Cursor Cloud (e.g. https://cursor.com/agents/…) — agents push branches; Jonny pulls on the PC.

### Machines
1. **Windows PC (runtime / stream machine)**  
   - Path: `G:\JonnyD\NeuroAI_Bot`  
   - CUDA GPU (Whisper), VB-Audio Cable, mpv, OBS, VTube Studio  
   - This is where FireRed actually runs. Agents usually **do not** remote-control it; Jonny runs one resume script.
2. **Dev / cloud / Mac copies**  
   - Agents edit code, push GitHub, Jonny pulls.  
   - Historical Mac path mentioned in older handoffs: `/Users/jonnydunkleberger/Desktop/OG Kira`  
   - Cloud workspace may be `/workspace`.

### Sacred rules (non-negotiable)
From `AGENTS.md`:
- **Canonical save is sacred.** Never write it directly; use checkpoint / banking / `PROMOTE_TARGET.txt` / sanctity gate.
- **Kira’s identity is sacred.** Persona/soul intact; game modes behind toggles.
- **Verify by running, not vibes.** Headless oracle (`recon_longrun.py`) + soak logs.
- **Menus by cursor-readback**, never state bytes alone (pitfall 13 — immortal battle wedge).
- **No ROMs, saves, or Nintendo assets** in the repo or git history. Ever.
- **Claude via OpenRouter only** (`kira/brain/claude_gateway.py`). Anthropic account banned. Need `OPENROUTER_API_KEY` in `.env` (same key family as web app). Never construct `AsyncAnthropic` directly.

### Boot (local bot)
```powershell
cd G:\JonnyD\NeuroAI_Bot
.\.venv\Scripts\Activate.ps1
python run.py
# dashboard: http://127.0.0.1:8766/
```

### Marathon / stream launch doctrine
- `watch.py` = **couch test only** — no supervisor, disposable sandbox under `%TEMP%\kira_watch\`, does **not** bank canonical. Not the show.
- **Real show:** `python run.py` (bot first), then supervised free-roam:  
  `python pokemon_agent/supervisor.py --timeline sherpa --audio`  
  (or the wrapped `resume_marathon.ps1` which does forensics + relaunch).
- `go.py` showtime spine is **not** full-marathon-ready (historically ends early); sherpa free-roam is the marathon path.

### Resume loop (how agents + Jonny work)
1. Agent pushes code (+ optional `pokemon_agent/PROMOTE_TARGET.txt` directive).
2. Jonny runs `resume_marathon.ps1`.
3. Script: kills Python, pulls, collects soak forensics into `docs/soak-reports/<stamp>/`, honors promote directive, relaunches bot + supervisor.
4. Agent reads the new soak report → fixes → push → repeat.

`PROMOTE_TARGET.txt` values: `CANONICAL`, `SNAPSHOT <file.state>`, `SNAPSHOT kira/<file.state>`, `RESTORE_WORLD`, `MIGRATE_SHOWTIME`, `NEW_CAMPAIGN` (consumed after run).

---

## 4. Architecture (local) — what the pieces are

### Layers
- **`kira/`** — companion brain: voice loop, memory, Twitch/YouTube, control server, modes (VN autopilot, etc.), TTS/STT, soul.
- **`pokemon_agent/`** — autonomous FireRed harness (the proving ground for a *general* game harness).
- **`web_dashboard/`** — OBS/browser HUD bits (e.g. pokemon HUD).
- **`docs/`** — mission docs, soak reports, handoffs, run stats.

### FireRed harness (portable idea)
From `pokemon_agent/AUTONOMOUS_GAME_HARNESS.md`: the real product is **not** “a FireRed bot” — it’s a **generalizable autonomous-game harness**. FireRed is game #1. Engine vs game-knowledge split; pitfalls are expensive — don’t re-pay them.

**Big files / concepts:**
- `campaign.py` — free-roam brain, victory lap, questlines, E4 gates (huge).
- `play_live.py` — process entry, render, audio, voice hooks, watchdog feed.
- `supervisor.py` — crash/hang relaunch with `--resume`.
- `travel.py`, `battle_agent.py`, `dialogue_drive.py`, `hm_teach.py`, `fly_nav.py`
- `legendary_strikes.py` — Moltres / Articuno / Zapdos / Mewtwo strikes
- `e4_strike.py`, `victory_road.py` — League path
- `pokemon_planner.py` + `gamedata/frlg_team_plan.json` — team archetypes
- `pokemon_voice.py` — salience tiers T0–T3 → bot narration
- `recon_longrun.py` — headless max-speed look-ahead oracle
- `resume_marathon.ps1` — the human-facing one-button ops script

### Decision loop (mental model)
Each free-roam tick: sense world → build options → (often) oracle/LLM pick → deterministic actuators (travel/battle/talk/PC/HM). Watchdogs (`StuckWatch`, macro GREEN/YELLOW/RED, wedge memory) fight softlocks. **Mechanical reflexes must not wait on LLM** (pressing A to clear dialogue is not a thinking problem).

### Victory lap (endgame checklist)
`VICTORY_LAP_ORDER` (code, not mood):
`earthquake → box_bench → moltres → articuno → fly → zapdos → ice_beam → repack`

While any item is pending, League actions are held. Items can be **honestly skipped** after bounded fails (sticky in-process; **clears on process restart**). Mewtwo is **post-credits only**.

---

## 5. Where the live run stands (2026-08-08)

### Done (celebrate this — chat already did)
- All **8 badges**
- **Moltres** caught (Sevii round-trip)
- **Articuno** caught (Seafoam)
- Blastoise is a real L68–69 ace
- Stream energy is real: bird catches go wild; funnel to web app is working

### Not done
- **Zapdos** — still at Power Plant; last night she never entered for real
- **Ice Beam** teach / confirm on Blastoise (lap item; sometimes already “done” depending on coins/TM)
- **Level Lapras + Kadabra** for E4 readiness gate (floor ≥ ~L42, ace–floor gap ≤ 15)
- **Victory Road → Elite Four → Champion → credits**
- **Mewtwo** after credits

### Last night’s failure mode (so you don’t gaslight chat)
1. Articuno done → Seafoam egress / Fly / Route 10 confusion for hours  
2. Route 10 is a **split map** — south segment (from Lavender) cannot reach Power Plant water; need **north** via Cerulean → Route 9  
3. Diglett L18 + PC shuttle + Eevee noise stole focus  
4. Final soak: **AttributeError** every tick — `_zapdos_north_staging` called but missing → froze in Fuchsia  
5. She *said* she found Zapdos; **dex/party disagree**. Chat was correct.

### What `9e4eeec` wired for resume
1. Implement `_zapdos_north_staging` (Fuchsia → R8 → Saffron → Cerulean → R9 → R10 NORTH)  
2. E4 roster: **KEEP** Blastoise / Kadabra / Lapras / birds; **DROP** Diglett / Fearow  
3. Re-arm `box_bench` when Diglett blocks a Zapdos seat (or bird auto-boxed)  
4. E4 readiness floor **ignores** DROP species so Diglett L18 can’t NON-CONVERGE the grind  
5. Tests in `pokemon_agent/recon_fly_nav_test.py`

### Target composition for credits
| Seat | Mon | Role |
|------|-----|------|
| 1 | Blastoise | Ace / Surf / Ice Beam |
| 2 | Zapdos | Electric (Lorelei / Gyarados) — **catch next** |
| 3 | Articuno | Ice (Lance) |
| 4 | Moltres | Fire coverage |
| 5 | Lapras | Bulk / second ice — **needs levels** |
| 6 | Kadabra | Psychic (Bruno / Agatha) — **needs levels** |

Box Diglett. Skip Eevee/Jolteon (redundant with Zapdos; was poisoning the lap). Skip fossil/Aerodactyl chase for credits.

---

## 6. How Jonny likes to work (agent operating manual)

### Communication
- Lead with **outcomes**, plain language, short. He watches the stream — symptoms first.
- **Bold sparingly.** Don’t bury the answer under process theater.
- Cost-sensitive: no performative recon. Investigation ends in a **shipped fix** or a **falsified hypothesis with log evidence**.
- Prefer letting **her** run (soaks, oracle) over endless agent speculation. Runtime is cheap; agent tokens are not.

### Coding style for this repo
- Match existing harness style: LOUD logs (`[lap]`, `[box]`, `[travel]`), bounded attempts, honest skips, fail-closed.
- Don’t “simplify” the sanctity gate or identity firewall.
- Don’t commit ROMs/saves/Nintendo assets.
- Small focused commits; push the working branch; update the PR.
- Branch naming for cloud agents: `cursor/<descriptive-name>-bdbc`.

### Verification
- Unit tests where they exist (`recon_*_test.py`) — run them.
- Live proof = soak report under `docs/soak-reports/` after Jonny resumes.
- Headless oracle for systemic claims.

### What “done” looks like for the current stream arc
1. Zapdos caught (party or confirmed dex + withdraw)  
2. Diglett not on the E4 six  
3. Lapras + Kadabra ground enough for readiness gate GREEN  
4. She walks Victory Road and rolls **credits** on stream  
5. Chat loses its mind; clips cut; site plug lands  
6. Postgame: Mewtwo (optional victory-lap after credits)

---

## 7. Web app scope (parallel — don’t lose the plot)

Even when your hands are in FireRed code, keep the product story:

- **Site:** xoxokira.com (funnel target from stream pins / plugs)  
- **Repo:** separate from `Kira_AI` (web/SaaS codebase)  
- **Brand:** premium concierge, Shangri-La calm luxury — not generic AI SaaS purple  
- **Promise:** the girl on stream is the girl in the product — talks to *you*, remembers, companion-grade  
- **Why FireRed matters to SaaS:** proof of life + entertainment moat. People don’t convert from a landing page paragraph; they convert after watching her catch Articuno and roast chat.  
- **Shared infra:** OpenRouter keys / model routing philosophy; identity consistency.

If an agent is only hired for “finish FireRed,” still avoid changes that make her colder, quieter, or less chat-aware — that kills the funnel.

---

## 8. Read-first list (in order)

1. **This file** — `docs/HANDOFF_FULL_PROJECT.md`  
2. `docs/marathon-stream-plan.md` — Frankfurt Test + build order  
3. `AGENTS.md` — non-negotiables  
4. `CLAUDE.md` / `STATE_OF_PROJECT.md` / `NEXT_SESSION.md` — **if present on Jonny’s PC** (gitignored); they outrank process guesses  
5. `pokemon_agent/AUTONOMOUS_GAME_HARNESS.md` — pitfalls  
6. Latest `docs/soak-reports/<stamp>/` — especially `canonical_health.json` + supervisor tails  
7. `docs/RUN_STATS_fresh_go_6.md` — proof the engine can credits

Older incident handoffs (context, not current truth):
- `docs/handoff-nuclear-recon.md` — July 30 stuck-loop / watchdog era  
- `docs/handoff-cerulean-border-war.md` — mid-game travel war  

---

## 9. Ops cheat sheet

### Resume marathon (canonical)
```powershell
cd G:\JonnyD\NeuroAI_Bot
powershell -ExecutionPolicy Bypass -File pokemon_agent\resume_marathon.ps1
```

### Switch to endgame branch first (current)
```powershell
git fetch origin
git checkout cursor/endgame-credits-path-bdbc
git pull
git log -1 --oneline   # 9e4eeec …
```

### Forensics locations
- `docs/soak-reports/<YYYYMMDD_HHMMSS>/canonical_health.json`  
- `tail_supervisor_*.log`, `engine_tail.log`, `resume_marathon.log`  
- Live debug: `logs/debug/console_*.log`, `playlive_faulthandler.log`

### Env flags you will see
- `POKEMON_AGENT_ENABLED` — must be true or she plays mute  
- `POKEMON_VICTORY_LAP`, `POKEMON_BOX_FLOW`, `POKEMON_E4_STRIKE`, `POKEMON_E4_READINESS_GATE`  
- `POKEMON_EEVEE_FETCH` — endgame muted at badge 8 on purpose  
- `OPENROUTER_API_KEY` — required  

---

## 10. Competitive / cultural notes (for tone)

- Chat is co-pilot and audience. When they say “you didn’t go to the Power Plant,” believe the save file, not her bit.
- Bird catches and League moments are **content peaks** — don’t bury them in menu loops or silent grinding without narration.
- Copyright-safe stream posture called out in older handoffs: game audio + her voice; know the lane Jonny is playing.
- The emotional product is: “She’s alive with us.” Technical cleverness that makes her feel stuck or fake is a regression even if the pathfinder is correct.

---

## 11. Immediate next agent checklist

- [ ] Confirm PC is on `cursor/endgame-credits-path-bdbc` @ `9e4eeec` or newer  
- [ ] Jonny runs `resume_marathon.ps1`  
- [ ] Watch soak/log for: `[box] RE-ARM box_bench` (Diglett out) → `ZAPDOS NORTH STAGING` → Power Plant → catch  
- [ ] Confirm Zapdos in party (not just narration)  
- [ ] Confirm Diglett boxed; Kadabra + Lapras retained  
- [ ] Let E4 readiness grind run (VR escalation if open grass insufficient)  
- [ ] League → credits  
- [ ] Cut clips → plug xoxokira.com  
- [ ] Only then: Mewtwo / postgame flex  

If she strands again: **do not** invent a new side quest (Eevee, fossil, dex padding). Fix the path to Zapdos / levels / League. Credits first.

---

## 12. One-paragraph pitch (paste-ready)

> Local Kira is Jonny D’s hybrid AI VTuber: she talks to Twitch/YouTube chat for real and is autonomously playing Pokémon FireRed on stream — she already has all eight badges and two legendary birds (Moltres, Articuno), and the room goes crazy for it. The technical mission is the Frankfurt Test: start stream, walk away, she narrates and plays to the Hall of Fame without babysitting. The business mission is parallel: those streams funnel into a premium Shangri-La–vibe web companion SaaS (xoxokira.com) aimed at being the clear number-two AI VTuber energy to Neuro-sama — with the wedge that Kira actually talks to people. Runtime lives at `G:\JonnyD\NeuroAI_Bot`; code at `JonathanDunkleberger/Kira_AI`; current endgame branch wires Zapdos → Kadabra six → Elite Four → credits.

---

*End of handoff. If this conflicts with PC-local `CLAUDE.md` / `STATE_OF_PROJECT.md`, those win on process; this wins on dual-product vision + 2026-08-08 live endgame state.*
