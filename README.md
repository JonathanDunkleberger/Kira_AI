# Kira — the AI companion who just beat Pokémon by herself

![Kira playing Pokemon FireRed](VTuber%20Demo%20-%20Kirav3.gif)

On July 15, 2026, an autonomous AI VTuber named Kira played **Pokémon FireRed from a fresh save to the credits — 8 badges, the Elite Four, the Champion — with zero human input.** No scripted decisions, no human hands mid-run, no reloads. She built her own team, got flattened by the League, healed up, restocked with her own prize money, and walked back in until she won.

Her own words the first time she beat it:

> "...that's it. that's the whole thing. eight badges, the Elite Four, the Champion — I actually did it. we did it."

## The certified run

| | |
|---|---|
| Result | Fresh save → 8 badges → Elite Four → **Champion → credits** |
| Human input | **Zero** — no code changes, no resumes, no touches, the whole run |
| Wall clock | 6h 08m headless at ~14x speed (models ~86 human-hours of gameplay) |
| Battles / decisions | 3,850 real battles · 353 strategic decisions |
| Crashes | 0 tracebacks across a 200,000-line run log |
| Final team | Venusaur 87 · Lapras 65 · Raticate 62 · Arbok 62 · Kadabra 62 · Dugtrio 61 |

The team was assembled entirely within the run — caught, gifted, and evolved on the road — and she entered the League only after her own readiness gate said the *whole team* was ready, not just her starter.

The honest footnote, because honesty is the whole point: she entered at the minimum bar and the League flattened her eight times. Her recovery loop (heal → restock → walk back in) ground it out, and the eleventh career fight against her rival was the one that made her Champion. I would not trade that arc for a clean sweep.

## What Kira actually is

Not a Pokémon bot. Pokémon is just what she is pointed at this month.

Kira is a continuous AI companion — one self, rotating objects. She plays games, watches movies, listens to music, talks — and carries memory, moods, opinions, and grudges across all of it. The design target is the intersection of three characters:

- **Maya (Sesame)** — best-in-class conversational presence
- **Neuro-sama** — a face, a fandom, the VTuber format
- **Samantha (Her)** — continuity; actually being known over time

...with one hard rule: none of Ava's tricks (Ex Machina). No engagement-extraction loops, no manufactured attachment, no dark patterns. That is doctrine, not marketing.

## How the run worked

- **RAM-read game state** — she reads the game's memory directly (party, badges, map, battle state). No screen-scraping for state: vision is for vibes, RAM is for truth.
- **Warp-graph navigation** — a learned world model plus a warp/connection graph drives cross-map travel; a BFS travel engine handles the tiles.
- **The E4-readiness gate** — she refuses to enter the Elite Four until every team member is at least L42 and the ace-to-bench gap is 15 or less. No solo-carry with a dead-weight bench.
- **The ace-cap** — while the gate is red, her strongest mon is benched from XP so the rest of the team levels. She raises a family, not a carry.
- **Soul / oracle layer** — an LLM makes her decisions in character, with journey memory, wants, and a genuinely held grudge against her rival.
- **The night train** — this repo was substantially built by an unattended loop of AI coding shifts, directed by a person who cannot traditionally code. A methodology write-up is coming.

## Who built this

I am an ex-M&A banker with an econ degree. Twelve months ago I could not write a for-loop. This is ~700 hours of directing AI to build her, one shift at a time. The commit history is the receipt — all 1,300+ of them.

## Run her yourself

- **You must bring your own legally obtained FireRed ROM.** No ROM, no saves, no Nintendo assets exist in this repo or anywhere in its history.
- Python 3.10 · libmgba bridge · LLM keys (Anthropic / Groq / Gemini) · Azure TTS.
- Fair warning: she was built on one machine and is still being generalized — some paths are hardcoded and setup docs are thin. Open an issue or come ask in Discord; making her easy to run is the next chapter.

## Links

- 💬 **Talk to Kira** (the web companion): https://xoxokira.com
- 🟣 Discord — The Kira Agency: https://discord.gg/8tg5wZy4RW
- ☕ Support: https://buymeacoffee.com/jonnyd11
- 📧 Business: kirachan.vtuber@gmail.com

## License

AGPL-3.0. If this project made you feel something, a star genuinely helps more than you would think — it is the number I am building against.

