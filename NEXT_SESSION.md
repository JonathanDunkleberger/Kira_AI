# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## ⛏️ NS9 IN FLIGHT: grind Lapras → run the FULL VALIDATION TAIL (re-badge → VR → E4) → CREDITS or pinpoint the true E4 blocker

**THE NS9 STRATEGIC SHIFT (rule-8 look-ahead):** stop grinding BLINDLY toward an unvalidated bar. The
downstream chain is 100% code-ready (verified NS9): every leg is `_STATE`-parametrized with `_resolve_state`
+ kit sidecars — `recon_seafoam` (SEAFOAM_STATE) → `recon_mansion` (MANSION_STATE) → `recon_blaine`
(BLAINE_STATE→badge 0x826) → `recon_giovanni` (GIOVANNI_STATE→badge 0x827) → `recon_victory` (VICTORY_STATE
+RESUME_STAGE) → `recon_e4` (E4_STATE). So the ONLY unknown is the E4 LEVEL BAR. NS9 validates it: grind
Lapras to ~38, then run the WHOLE tail as one sweep — it either **ROLLS CREDITS** (the job) or **pinpoints
the exact E4 battle/mon that fails**, so the next grind is TARGETED (grind only what the sweep fingered),
never blind. This converts "grind Abra→Kadabra for 3 shifts hoping it's enough" into "know the bar."

**GRIND STATE (NS9):** running from `bench_grind_kit.state` (badges=6, surf_ready CONNECTED-graph lineage).
Route 18 (map 3,36, west open grass L23-29), participation-switch PROVEN. Banks every ~150s → `banked_GRIND`
(BARE sidecar names: world_model.json etc. — promote renames to `<name>.world_model.json`). Party = Lapras
(131) climbing / Spearow(21) L13 / Rattata(19) L8 / Abra(63) L10 / Drowzee(96) L14 / **Venusaur(3) L60** ace.
Rate ≈ 1 level / 2.5 min (slows as Lapras out-levels L29 wilds).

### ▶ THE VALIDATION-TAIL SWEEP (run when Lapras ≈ 37-38; each leg ~90s; promote BETWEEN legs).
`bench_grind_kit` is **badges=6** → VR/E4 need badges 7+8 RE-EARNED first (Venusaur L60 solos every re-badge
leg regardless of Lapras level — Lapras level only bites at VR/E4). From `pokemon_agent/`, each leg banks
`banked_<NAME>` (bare sidecars); promote to a workshop fixture (state + renamed `<name>.<sidecar>.json`)
before the next leg. Promote helper (NS9 wrote it): `python promote_to_workshop.py <banked_dir> <basename>`.
Sequence:
```
# 0. promote the leveled grind bank first:
python promote_to_workshop.py G:/temp/longrun/banked_GRIND bench_grind_kit
# 1. Seafoam -> Cinnabar
SEAFOAM_STATE=bench_grind_kit.state ../.venv/Scripts/python.exe -u recon_seafoam.py > G:/temp/longrun/ns9_seafoam.log 2>&1
python promote_to_workshop.py G:/temp/longrun/banked_SEAFOAM cinnabar_kit_g
# 2. Mansion -> Secret Key
MANSION_STATE=cinnabar_kit_g ../.venv/Scripts/python.exe -u recon_mansion.py > G:/temp/longrun/ns9_mansion.log 2>&1
python promote_to_workshop.py G:/temp/longrun/banked_MANSION secretkey_kit_g
# 3. Blaine -> badge 7 (0x826)
BLAINE_STATE=secretkey_kit_g ../.venv/Scripts/python.exe -u recon_blaine.py > G:/temp/longrun/ns9_blaine.log 2>&1
python promote_to_workshop.py G:/temp/longrun/banked_BLAINE blaine_kit_g
# 4. Giovanni -> badge 8 (0x827), crosses to Viridian
GIOVANNI_STATE=blaine_kit_g ../.venv/Scripts/python.exe -u recon_giovanni.py > G:/temp/longrun/ns9_giovanni.log 2>&1
python promote_to_workshop.py G:/temp/longrun/banked_GIOVANNI giovanni_kit_g
# 5. Victory Road (now badges=8)
VICTORY_STATE=giovanni_kit_g ../.venv/Scripts/python.exe -u recon_victory.py > G:/temp/longrun/ns9_victory.log 2>&1
python promote_to_workshop.py G:/temp/longrun/banked_VICTORY indigo_kit_g
# 6. Elite Four -> CREDITS or blocker
E4_STATE=indigo_kit_g ../.venv/Scripts/python.exe -u recon_e4.py > G:/temp/longrun/ns9_e4.log 2>&1
```
**IF CREDITS ROLL:** write `CREDITS` as LINE 1 of NIGHT_REPORT.md (stops the loop) + full mountain survey.
**IF A LEG STALLS:** read its log for the exact battle/mon, characterize in NIGHT_REPORT + here, then grind
EXACTLY that (E4 fails at Agatha → grind Abra→Kadabra to the level the log implies; fails on Lapras low at
the Champion → push Lapras higher). Restart the grind toward the validated target.

**RESTART/CONTINUE GRIND (from pokemon_agent/):**
```
GRIND_STATE=bench_grind_kit.state GRIND_TARGET=38 GRIND_SPECIES=131,63,64 GRIND_MAP=3,36 GRIND_DIR=west \
  GRIND_MIN=90 GRIND_PROBE_S=150 ../.venv/Scripts/python.exe -u recon_grind_bench.py > G:/temp/longrun/nsX_grind.log 2>&1 &
```
GRIND_SPECIES priority-ordered: 131=Lapras (Charizard/Lance answer) first, then 63=Abra (→Kadabra @L16,
Agatha answer), then 64. banked_GRIND ratchets every pass — promote → bench_grind_kit after meaningful banks.

**⚠️ NAV LESSONS (carried, still true):** giovanni_done_kit's Viridian is a world-graph ISLAND (grind from
it fails); Route 15 grass is guardhouse-DIVIDED (warp won't fire); caves/VR have no grass. Route 18 (west,
OPEN) is the working grind spot from surf_ready_kit lineage. `grind()` needs GRASS + a reachable Center.

---

## 🏁 ALL 8 BADGES already WON on the kit line (NS5) — banked chain giovanni_done_kit (badges=8).
The E4 team-wall (solo Venusaur + weak bench) is the ONLY thing between here and CREDITS. NS7 built+proved
`recon_grind_bench.py` (participation-XP switch, no PC-box); NS8/NS9 continue the grind. See memory
[[pokemon-nightshift7-bench-grind-nav-island]] + [[pokemon-e4-gauntlet-truths]] + [[pokemon-e4-livelock-family-killed]].

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW — venv = 2-PID shim, that's ONE logical run;
kill `taskkill //F //IM python.exe //T`). recon_longrun arg1 = state BASENAME with .state, arg2 = max_min,
goal via `LONGRUN_GOAL_FLAG=0x827`. Flags module = `field_moves` (fm.read_flag).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = bench-grind (Lapras ~L33→38)
at Route 18 → validation tail → CREDITS or pinpoint the E4 bar. Pop-in = `python pokemon_agent/watch.py`.
