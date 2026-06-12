# Kira Cam — Ambient Audio Loops

Drop CC0-licensed audio loops into **this folder**.  
The dashboard (`Stream Screens → Kira Cam Ambience`) will list them automatically.

## Accepted formats
`.mp3` · `.ogg` · `.wav` · `.flac`

## CC0 audio sources

| Source | URL | Notes |
|--------|-----|-------|
| **Freesound.org** | https://freesound.org/search/?f=license:0 | Filter by "Creative Commons 0" — use the Licence filter in the sidebar. Search: `snow wind`, `rain loop`, `cafe ambience`, `forest night`, `fireplace`. |
| **Pixabay Audio** | https://pixabay.com/sound-effects/ | All Pixabay audio is CC0/royalty-free. Search: `rain`, `wind`, `cafe`, `night`. |
| **Freemusicarchive.org** | https://freemusicarchive.org/genre/Ambient/ | Filter by CC0. Good for longer lo-fi loops. |
| **Soundsnap free tier** | https://www.soundsnap.com/ | Limited free downloads; check licence per file. |
| **NASA audio** | https://soundcloud.com/nasa | Public domain by default; mostly space ambiences. |
| **Kenney.nl** | https://kenney.nl/assets/tag:audio | CC0 SFX packs — contains useful short ambience clips. |

## Recommended files to look for

```
snow_wind.mp3       — soft howling wind, occasional gusts
city_hum.mp3        — distant traffic, urban at night
rain_light.mp3      — gentle rain on a window
rain_heavy.mp3      — heavier storm rain
fireplace.mp3       — crackling fire, cozy
cafe_murmur.mp3     — soft coffee-shop chatter
forest_night.mp3    — crickets, light breeze
lo_fi_hum.mp3       — generic lo-fi room tone
```

## OBS setup (REQUIRED for stream audio)

1. In OBS, right-click the **Kira Cam** browser source.
2. Select **Properties**.
3. Check **"Control audio via OBS"**.
4. The browser source will now appear as an audio source in the OBS mixer.
5. Set its level in the mixer to taste (the dashboard volume slider controls the
   in-page gain — OBS fader is an additional level on top).

## Technical notes

* The player uses the **Web Audio API** `AudioBufferSourceNode.loop = true` for
  true gapless looping — no gap/click between buffer end and restart (unlike
  `<audio loop>` which has a brief gap on MP3).
* Volume is applied via a `GainNode`; changes from the dashboard slider are
  smoothed with a 50ms time constant to avoid clicks.
* The bot WS connection (`/ws`) carries the `ambience` state (file + volume)
  in every snapshot — the Kira Cam page picks it up automatically.
