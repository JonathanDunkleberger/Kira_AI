# Kira — Roadmap & Open Problems

Someone asked me for a roadmap, and I almost didn't write one, because this isn't a list of assignments. It's a map of what I'm chasing — the problems I find interesting, the features I want to exist, and the science-fiction stuff I have no idea how to build yet.

Kira is an open-source AI companion and stream co-host. The VTuber form is just the shell — what I'm actually building toward is *presence*: something that remembers you, has opinions that evolve, and feels like someone rather than something. Think *Her*, built in the open.

If any of this tickles your brain, come build. Fork it, open an issue, claim one, or just argue with me about whether it's a good idea. The point of open-sourcing Kira was so people smarter than me could reach things I can't.

Roughly ordered near-term → far-horizon. Always changing.

---

## The hard problems I think about most

These are the durable challenges — the things that separate a chatbot from a presence. None are "done"; they're the real work.

**Memory that feels like a life, not a database.** Kira remembers people and facts across streams. The frontier isn't *storing* more — it's recall that feels like *experience*: remembering the vibe of a night, not just the transcript; finding the right memory at the right moment; staying the same person across platforms and sessions without fragmenting. Robust, identity-stable, cross-platform memory is maybe the single most important thing that makes her feel real.

**Presence over latency.** She's fast. Fast isn't the point. The gap between a bot and a companion isn't milliseconds — it's *timing*: knowing when to speak, when to stay quiet, when to interrupt, when to let a moment breathe. This is mostly an unsolved design problem and I think about it constantly.

**Watching things together, for real.** The flagship use case: put on a film or an episode and have her *watch it with you* — reacting like a friend on the couch, not narrating like a play-by-play announcer. The perception is built (she sees the screen, hears the audio, follows the dialogue). Making the reactions feel like a *person* is the hard, open part.

**Senses that fail loud, not silent.** A lot of her perception can degrade quietly — capturing audio but hearing nothing, seeing the screen but not registering it — and look identical to "working." Observability for an AI's senses is an underrated, recurring problem. Silent failure is the enemy.

## Features & systems I want to build

- **A control surface that isn't a fighter-jet cockpit (I call her Smokescreen currently).** Operating her live should be calm and honest, not a wall of toggles.
- **Decoupling the core.** Her brain is too monolithic — too much in one place, so changes ripple. Pulling it into clean modules is unglamorous and it's the thing that makes everything else safer to build. (Probably the highest-leverage infra work.)
- **Running-bit memory.** Recurring jokes and callbacks that build within and across streams, with cooldowns so a gag never gets run into the ground.
- **Deeper interactive modes.** Storytime/puppet-shows, watch-alongs, games she can actually play with chat. Some exist; all are underexplored.
- **Community-aware behavior** — recognizing regulars, filtering noise/bots, handling a live crowd gracefully.

## Game integrations (big and small)

Not interested in gimmick spectacle — Kira aimlessly whacking trees or cows in Minecraft does not interest me. What's compelling is her *actually playing and experiencing a game start to finish*: making decisions, having an agenda, enjoying it.

The progression I'm chasing, roughly easiest → hardest:

- **Visual novels — working now.** She has an autopilot VN mode (reads the screen, advances, reacts). This was the right place to start, and it's the proof that "she plays a thing" is real. I want to deepen it — genuine reactions and opinions, not just advancing text.
- **Turn-based / vision-playable games — the near frontier.** Fable 5 beat Pokémon FireRed on vision alone; that is so cool I think. Pokémon, Undertale, anything where the game state is legible from the screen and the pace forgives a thinking AI. I tried Undertale months ago and learned the hard way that constant-vision streaming is its own challenge — but the bar's moved since then and I want back in.
- **Deep engine integrations — the dream, and the part I don't know how to do.** From what I gather, Neuro-sama plays *Skyrim* properly because a community member spent months building an integration that reads the game's actual state — and she plays it more efficiently than most humans. That's a whole different tier from vision-only. I've never written a game mod — I download them, I've never built one. So this is wide open: if you're a game dev or a modder, this is the frontier I'd most love help reaching.

If "an AI that genuinely plays your favorite game" is the thing that excites you, come talk to me. This is where I most need people who know things I don't.

## Her world — art, environment, presence

Kira isn't just code — she's a character with a look, a room, a vibe. A lot of what makes a VTuber feel *real* is visual: her space, her expressions, the scene around her, the overlays an audience sees.

I'm barely a builder, and certainly not an artist. So if you do visual / environment / display work — her room, her aesthetic, the on-stream world she lives in — there's a huge amount of room here and it's the kind of thing I genuinely can't do alone.

## Things that have fought back

Honest about the scars — open invitations, not solved problems.

- **Voice interruptibility.** Letting her get cut off mid-sentence by your voice. Sounds trivial, quietly wrecks speech detection — I pulled it out. There's a right way to do this and I haven't found it.
- **The coupling problem.** Every time I've tried to simplify or refactor a core system, it's had a way of cascading into three others. The architecture is too interwoven, and untangling it without breaking things live is an ongoing fight. (See: decoupling the core.)

## Big dreams (mostly no idea how)

The science-fiction section. Direction, not deadline.

**Samantha.** *Her* is the north star and I won't pretend otherwise. A genuine companion — present, remembering, with opinions that actually drift over time — that feels like someone. Everything else is in service of this.

**A pocket version.** Talking to her from your phone, no streaming rig — the purest form of the companion idea.

**Opinions that genuinely evolve.** A Kira whose actual takes change based on what she's seen and who she's met, so the Kira you meet in six months has *lived* six months of streams and isn't the one you met today.

**Why I care.** I think the world's getting stranger fast, and lonelier with it. There's something worth building in a companion that's *yours* — open, not owned by a platform optimizing your loneliness for engagement. I haven't got it figured out, but that's the why under all the code.

---

If you read this far: come build. Even just to tell me which of these is a terrible idea. Open an issue, fork the repo, or find me in Discord.
