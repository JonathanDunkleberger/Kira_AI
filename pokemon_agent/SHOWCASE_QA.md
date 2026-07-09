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

## ⚠️ ALIAS FIX (2026-07-08) — READ THIS
A headless probe of every bank found the old `pre-e4` alias pointed at `banked_E4`, which is the
**post-victory Hall of Fame ceremony** (Lance already beaten, credits rolling — no fights left).
That's why the last spawn dropped Jonny at the credits. **Fixed:** `pre-e4` now points at
`banked_VICTORY` — the TRUE doorstep (Indigo Plateau, 8 badges, **game_clear=FALSE, E4 not yet
beaten**, L66 team). Real first-time E4 combat IS watchable there. Also: `pre-blaine` now → the
real pre-Blaine save (`banked_CINNABAR`, badge 6). The old post-win banks are kept under
`hall-of-fame` / `blaine-done` for the record.

## THE HIGH-VALUE ONE — first-time Elite Four + Champion Gary (CONFIRMED combat ahead)
```
python pokemon_agent/watch.py --at pre-e4 --goal "fight through the Elite Four for the first time — Lorelei, then Bruno, Agatha, Lance — then beat Gary and become Champion"
```
Now spawns at Indigo Plateau with the E4 genuinely un-beaten. The goal-pin also presents the
present-tense era self-model (she's ABOUT to fight it) and drops the Champion saga, so she's no
longer incoherent ("do the thing I remember already doing"). **Start here.**

## CONFIRMED combat-ahead spawns (from the probe — game_clear=FALSE, fight genuinely ahead)
| Moment | Command |
|---|---|
| **Elite Four + Gary** | `watch.py --at pre-e4 --goal "fight through the Elite Four and beat Gary"` |
| **Gym 7 — Blaine** | `watch.py --at pre-blaine --goal "beat Blaine and win the Volcano Badge"` |

## THE OTHER GYMS — MOSTLY POST-WIN (honest finding)
The probe showed the curated `banked_SABRINA` (6 badges), `banked_BLAINE` (7), `banked_GIOVANNI`
(8) are **post-win** checkpoints (banked at the gain-seam AFTER the badge) — so those gyms are
already beaten in RAM; a goal-pin narrates but there's no fight left. The `pre_*_reach` backups in
`pokemon_agent/states/campaign/` (pre_saffron_reach, pre_celadon_reach, pre_fuchsia_reach,
pre_vermilion_backup, etc.) are the LIKELY pre-gym states — spawn by full path and goal-pin them:
```
python pokemon_agent/watch.py --at pokemon_agent/states/campaign/pre_vermilion_backup_20260706_152434 --goal "beat Lt. Surge and win the Thunder Badge"
```
**Verify on first use** (does she enter the gym and fight?). If a `pre_*_reach` bank turns out
post-win too, that gym needs a small reposition bank — flag it and I'll cut one from a fresh run.
Gym 2 (Misty) has no clean pre-Misty bank at all — reposition follow-up.

## Mid-run trainer fights (always real combat, any time)
Any `pre_*` route/reach bank has wild + trainer battles on the way — a quick 5-min "watch her
fight and quip" grade without needing a gym. `--at pre-e4` (Victory Road / Indigo) has trainers +
the E4 itself; `--at rock-tunnel`, `--at silph` etc. all have live combat en route.

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
