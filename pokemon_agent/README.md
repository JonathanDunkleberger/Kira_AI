# pokemon_agent — autonomous FireRed (the "Kira plays Pokémon" project)

Fully isolated from the Kira bot (M0 imports nothing from `kira/`). The only Kira
touchpoint is the narrow reaction **seam** in `kira/bot.py` (`_pokemon_react`,
`_pokemon_choose_starter`), flag-gated by `POKEMON_AGENT_ENABLED` (default OFF).

## Setup (not in git)
1. **ROM** (copyright — supply your own): place a FireRed USA ROM at `roms/firered.gba`
   (game_code `AGB-BPRE`).
2. **Emulator bridge** (vendored libmgba-py, 42MB — gitignored): download the prebuilt
   Windows artifact and extract its `mgba/` package into `pokemon_agent/vendor/`:
   ```
   gh release download 0.2.0-2 -R hanzi/libmgba-py -p libmgba-py_0.2.0_win64.zip -D pokemon_agent/vendor
   python -c "import zipfile;zipfile.ZipFile('pokemon_agent/vendor/libmgba-py_0.2.0_win64.zip').extractall('pokemon_agent/vendor')"
   ```

## Milestones
- **M0** (`m0_sandbox.py`) — proves the loop: load ROM, render window, auto-clear the
  intro to the overworld, read RAM (coords/party), button-changes-state. CONFIRMED.
  - `bridge.py` (4 primitives), `firered_ram.py` (offsets; overworld coords/party CONFIRMED).
- **M1** (`m1_starter.py`) — **the starter pick**: boot → intro → Oak's lab → her *self*
  chooses a starter → selects → reacts in character.
- **M2 (banked)** — `pokemon_state.py` / `pokemon_policy.py` / `battle_agent.py` / `m1_battle.py`:
  battle reading + policy (policy TESTED; battle RAM offsets are CANDIDATES, unverified).

Run windowed: `.venv\Scripts\python.exe pokemon_agent\m0_sandbox.py`
(arrows = D-pad, Z = A, X = B, Enter = START).
