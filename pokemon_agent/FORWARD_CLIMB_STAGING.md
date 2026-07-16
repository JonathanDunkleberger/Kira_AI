# Forward-climb staging — gym 3 → 8 + Elite Four

Recon + staging artifact (NOT live wiring). Captures the capability bill and the exact data each
forward gym needs, so the next batch can wire them without re-reconning. **Nothing here runs a live
gym climb.** Created 2026-06-28.

## Current wired state
- `GYMS` registry (`campaign.py:216`) wires **only Brock + Misty** (verified door/leader coords).
- `_GYM_ORDER` (`campaign.py:1103`) lists all 8 gyms → `next_gym` + the 3-tier LONG goal already
  target the next gym by badge count. So forward gyms are already **known long-term goals**; what's
  missing is the **actuation data** to fight them.
- HM field-move detection + actuation exist (`field_moves.py`) but actuation is `POKEMON_FIELD_MOVES=0`
  and **unverified on a long-running core**.

## Capability bill (what the loop handles vs needs)
| Gym | City | Leader | Gate | Class | New verb |
|----|------|--------|------|-------|----------|
| 3 | Vermilion | Lt. Surge | trash-can switch puzzle | **B** | puzzle solve (approach TBD — NOT hardcoded presses) |
| 4 | Celadon | Erika | none | **A — loop handles** | — |
| 5 | Fuchsia | Koga | none (Snorlax bypassable via Route 15) | **A** | — |
| 6 | Saffron | Sabrina | Tea / NPC item-gate (no HM) | **A** | — |
| 7 | Cinnabar | Blaine | **Surf** water crossing | **B** | verify Surf actuation |
| 8 | Viridian | Giovanni | **Strength** (Victory Road) | **B** | verify Strength actuation |
| E4 | League | 5 trainers + Champion | pure battles | **A** | beat-E4 loop |

A = existing nav/travel/battle verbs suffice. B = needs one new/verified verb.

## Gym data bill — what each A-class gym needs before it can be added to GYMS
`GymSpec(name, city_map_id, gym_door_tile, leader_front_tile, badge_flag, junior_count, face_dir)`.
For Pewter/Cerulean these were sourced live. For 3–6 the spatial coords are **NOT in the repo** and
must be reconned (live walk or disasm) — do NOT guess them (bad coords = mis-routing):

| Field | Erika (Celadon) | Koga (Fuchsia) | Sabrina (Saffron) | Surge (Vermilion) |
|-------|-----------------|----------------|-------------------|-------------------|
| city map id | ❓ recon | ❓ | ❓ | ❓ |
| gym door tile | ❓ | ❓ | ❓ | ❓ |
| leader front tile | ❓ | ❓ | ❓ | ❓ |
| junior trainer count | ❓ | ❓ | ❓ | ❓ |
| badge flag | `0x823` Rainbow | `0x825` Marsh | `0x824` Soul | `0x822` Thunder |
| face_dir | likely UP | UP | UP | UP |

Badge flags follow the verified sequential pattern (`0x820` Boulder … `0x827` Earth) — those are
known. Everything marked ❓ is a ~10-min live recon per gym (same method that sourced Brock/Misty:
walk to the gym, dump the door tile + leader front tile + count the juniors).

Erika/Koga/Sabrina also need: any **gate item** before the city is reachable (e.g. Saffron needs the
guard-Tea; Celadon is post-Surge). Those are route-gates, not gym-internal — recon as the path opens.

## The two real gatekeepers — schedule a DEDICATED verification session
Surf (gym 7) and Strength (gym 8) actuation exist in code but are **unverified on a long-running
core** (the move-menu-wedge lesson says be cautious). Before relying on either:
1. Arm `POKEMON_FIELD_MOVES=1` on a controlled save in front of a cuttable tree / pushable boulder /
   surfable edge, with Jonny's live eyes.
2. Confirm the "Use HM?" prompt actuates and the obstacle clears, 3–5 runs, on a core that's been
   running a while (not just a fresh core).
3. Only after both pass do gyms 7–8 / Victory Road become reachable.

## Surge / gym 3 puzzle — solve approach DEFERRED
The trash-can switch puzzle is a gate. Jonny flagged he does NOT want "hardcoded switch presses".
Preferred = capability-not-script (detect the switches as objects / let her reason the solve). Decide
the approach when we reach it — staged here, not built.

## Hard guardrail
Do NOT run unsupervised live gym climbs past Misty. Everything above is recon + a data bill; the next
batch wires gyms 4–6 once their coords are reconned, then a verified Surf/Strength session gates 7–8.
