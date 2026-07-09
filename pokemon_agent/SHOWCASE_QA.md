# SHOWCASE QA HARNESS — goal-pinned watch spawns (2026-07-08)

How Jonny spot-checks the key showcase moments (each gym, the Elite Four, Champion Gary) as
~5-10min "enter → win" clips, to validate the 40-hour run before going live.

## The mechanism
Every banked save is post-credits, so a bare spawn makes the Champion pursue post-game goals
(Mewtwo/grind) instead of the era moment. **`watch.py --goal "<objective>"`** pins an era-correct
objective: it **suppresses the post-game victory-lap frame** and makes that objective her dominant
directive. Read-only — RAM stays ground truth for battles; this only reshapes her goal/context.

```
python pokemon_agent/watch.py --at <spawn> --goal "<era objective>"
```
(Boot the bot first — `python run.py` — the watch rig preflights it. Confirm your audio device
before boot; the SDL-audio mouth-flap bug is fixed but the go-live ritual still applies.)

## THE HIGH-VALUE ONE — first-time Elite Four + Champion Gary (what Jonny most wanted to watch)
```
python pokemon_agent/watch.py --at pre-e4 --goal "fight through the Elite Four for the first time — Lorelei, then Bruno, Agatha, Lance — then beat Gary and become Champion"
```
`banked_E4` = badges 8 at the Indigo Plateau doorstep. Goal-pin suppresses the victory-lap so she
runs the gauntlet forward. **Highest confidence — start here.**

## THE GYMS (each: enter → win → badge)
| Moment | Command |
|---|---|
| Gym 6 — Sabrina | `watch.py --at pre-sabrina --goal "beat Sabrina and win the Marsh Badge"` |
| Gym 7 — Blaine | `watch.py --at pre-blaine --goal "beat Blaine and win the Volcano Badge"` |
| Gym 8 — Giovanni | `watch.py --at pre-giovanni --goal "beat Giovanni and win the Earth Badge"` |
| Gym 5 — Koga | `watch.py --at <path>/pre_koga_badge5_backup_20260707_050954 --goal "beat Koga and win the Soul Badge"` |
| Gym 4 — Erika | `watch.py --at <path>/pre_erika_badge4_backup_20260707_014247 --goal "beat Erika and win the Rainbow Badge"` |
| Gym 3 — Surge | `watch.py --at <path>/pre_vermilion_backup_20260706_152434 --goal "beat Lt. Surge and win the Thunder Badge"` |

`<path>` = `pokemon_agent/states/campaign/`. Gyms 6-8 have curated aliases (clean, ready); gyms
3-5 spawn from the `pre_*` backup dirs by path.

## HONEST CAVEATS / FOLLOW-UPS (flagged, not forced)
- **Gym routing from post-credits saves — HANDLED:** when `--goal` names a gym leader (Misty,
  Surge, Erika, Koga, Sabrina, Blaine, Giovanni), watch.py auto-sets the era gym as her
  `next_gym`, so even a badge-8 save paths INTO the gym instead of stalling. (The E4 clip needs no
  leader name — the E4 is the forward path at badge 8.) If a gym clip STILL wanders, tell me the
  save's badge count and I'll widen the override.
- **Gym 2 — Misty: no clean pre-Misty bundle exists** (`badge3_bank` / `stall_cerulean` are near
  but not positioned pre-Misty). Small reposition follow-up if you want that clip.
- **`pre_*` backups**: verify each is a complete bundle (kira_campaign.state + world_model +
  strat + soul + journey) on first spawn; if one won't load, it needs a reposition bank.

## What to watch for (the QA bar)
Enter → win in ~5-10min, no aimless circling, era-correct behavior (not "let me go find Mewtwo"),
battle quips landing, reactions fast (<6s), and — post the audio fix — **her mouth flapping ONLY
when she actually speaks**, never to the game music.
