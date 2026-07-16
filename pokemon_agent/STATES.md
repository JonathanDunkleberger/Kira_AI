# Pokémon save lineages & run modes

Three physically-separate save buckets under `pokemon_agent/states/` (the whole `states/` tree is
git-ignored — these are local saves, not repo content):

| bucket | what lives here | who writes it |
|--------|-----------------|---------------|
| `states/workshop/` | the SMALL sherpa set: `og_postopening` (beginning), `brock_done`, `mtmoon_cleared`, `misty_done`, plus every state a live segment loads (`viridian_parcel_done`, `route3_caught`, `mtmoon_interior`, `brock_ready`, `after_pick_bulbasaur`). | WORKSHOP runs |
| `states/kira/` | Kira's **sacred** continuous playthrough save(s) — her let's-play. Old attempts are timestamp-archived to `states/kira/archive_<ts>/`, never clobbered. | **SHOW runs only** |
| `states/archive/` | every handicapped-era teaching capture + orphaned scratch/recon state. Out of the live path; not deleted. | the one-time `archive_sweep.py` |

`campaign.resolve_state(name)` finds a named state across `workshop → kira → flat states → archive`,
so boots and `GATE_NEEDS_STATE` loads keep resolving no matter which bucket holds the file.

## Two run modes (one toggle)

- **WORKSHOP** (default — our 90%): `play_live.py --go`. Resume from any `states/workshop/`
  checkpoint; jump/test freely. **Physically cannot write `states/kira/`** — `_save_checkpoint`
  asserts `show_mode` before any write under `kira/` and refuses LOUD otherwise.
- **SHOW**: `play_live.py --show`. Fresh bedroom boot, OR resume Kira's own `states/kira/` save if a
  playthrough is in progress. Walks AUTO with **zero skips**; banks progress to `states/kira/`. Any
  `GATE_NEEDS_STATE` fallback mid-run logs `SHOW-MODE SKIP VIOLATION: <segment>` and is counted — a
  clean SHOW run reports **zero violations**.

## Kira-run lifecycle (how we stream)

- **Start a fresh Kira run:** `play_live.py --show --fresh-kira` — timestamp-archives any existing
  Kira save to `states/kira/archive_<ts>/`, then begins a new sacred playthrough from the bedroom.
- **Resume the Kira run:** `play_live.py --show` — continues from the furthest `states/kira/` checkpoint.
- **If a Kira run breaks mid-playthrough:** drop to WORKSHOP (`--go`), fix on the sherpa timeline,
  then resume (`--show`) or restart (`--show --fresh-kira`) the Kira run. At most one active Kira run.

The dashboard (Kira Control Center) surfaces the WORKSHOP|SHOW toggle, the current mode, and a Go
button that respects it.
