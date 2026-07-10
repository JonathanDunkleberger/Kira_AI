# NEXT SESSION — resume prompt (frontier-first, kept CURRENT)

## 🔨 IN FLIGHT (night-shift 2): BADGE 6 (Sabrina) — SILPH CO. LIBERATION QUESTLINE BUILT, verifying
The badge-6 root blocker from NS-1 (no armed Silph questline → Sabrina's Rocket-blocked gym structurally
parks) is now WIRED. **What was built this shift (all COMPILES; WIRED; verification in progress):**
- **`pokemon_agent/silph_strike.py`** (NEW) — faithful port of the proven `recon_silph.py` state machine
  into the camp-driven strike-module shape (same as hideout_strike/tower_strike). `run_strike(camp, log,
  dbg_dir)` → `'freed_saffron'` on FLAG_HIDE_SAFFRON_ROCKETS (0x3E) set + walked out to Saffron. Enters
  Silph via Saffron street door (33,30)→(1,47), pre-dungeon heal, then the pad-maze climb: 9F pad→5F pocket
  Card Key (355) → pad chain 9F(9,4)→3F(2,14), 3F(13,14)→7F(5,4) [Gary + Lapras], 7F(5,8)→11F → Giovanni
  #2 → president. Returns `in_silph` (flag set, exit WIP) / `not_here` / `failed`.
- **`gamedata/frlg_gates.json`** — added flag `FLAG_HIDE_SAFFRON_ROCKETS` (id 62 / 0x3E) + a capability of
  the same key (`via:"strike"`, NO door → strike fires first per campaign.py:6125, success=("flag",...)).
- **`campaign.py`**: (1) `GYM_PREREQS` table (Sabrina → 0x3E) + `_gym_prereq_gate(gym)` — synthesizes a
  STORY_NPC gate when a gym's DOOR is story-locked (no tree for the HM probe). (2) Wired into the gym-stuck
  branch (~line 9037): story-prereq gate armed FIRST (before `_gym_gate_probe`) when `_active_questline is
  None`. (3) Registered `("flag","FLAG_HIDE_SAFFRON_ROCKETS")` → `_silph` in the `_questline_strike`
  registry; `in_silph` handled alongside `in_hideout` in the exit-WIP branch.

**THE FLOW:** koga_done_kit (badges=5, Fuchsia; has Tea + Fuji) → Tea gatehouses → Saffron → beat_gym stuck
on Sabrina → `_gym_prereq_gate` arms the Silph questline → `_run_questline_step` → door-less strike step →
`_questline_strike` runs `silph_strike` → 0x3E set → gym unblocks → beat_gym enters the pad maze (already
handled by beat_gym's recon_sabrina pad-router) → Sabrina (Marsh 0x825).

**VERIFY COMMAND (isolates the NEW code):**
`LONGRUN_GOAL_FLAG=0x3E .venv/Scripts/python.exe -u pokemon_agent/recon_longrun.py koga_done_kit.state 35`
→ GOAL when Saffron is freed. Then re-run with `LONGRUN_GOAL_FLAG=0x825 ... 45` for the full Sabrina badge.
Bank the GOAL checkpoint (G:/temp/longrun/banked_GOAL) → promote to workshop/silph_done_kit.state (then
sabrina_done_kit.state). If the Silph strike stalls mid-tower, read the log for the first floor/pad that
wedges and fix in silph_strike.py (the pad coords are the recon_silph ground truth — check read_warps live).

**WATCH FOR (predicted next blockers after Silph):** (a) beat_gym pad-router crossing Sabrina's interior
(strike-solved historically; port seam = recon_sabrina if the general walk wedges); (b) an ATTRITION/type
wall vs Sabrina's Alakazam (Venusaur solo-carry) — the generalizes-to-Sabrina fix is adding "Sabrina" to
`POTION_STALL_GYMS` (Saffron HAS a Mart + Dept store; the potion-stall leg is proven vs Koga).

## RESIDUAL (watchability, non-fatal, unchanged): gym maze junior-spin
Invisible-wall gyms (Koga; likely Sabrina) leave ~4 juniors un-engageable; she tries each 4× then burns
the clear-round cap before bailing to the leader (correct bail, not fatal). Fix in `_clear_junior_trainers`
(campaign.py ~3200): reachability pre-check — defer an unreachable junior in 1 try, remember across retries.

## ✅ BANKED PRIOR: BADGE 5 (Koga) WON FULLY UNAIDED (NS-1, b10bfdf)
koga_done_kit.state (badges=5, Fuchsia (9,33)). Fuchsia Mart mapped + POTION_STALL_GYMS={"Koga":30}.
See [[pokemon-nightshift1-fuchsia-mart-mapped]].

KEY FACTS: venv `.venv/Scripts/python.exe` (SINGLE-RUN LAW; kill `taskkill //F //IM python.exe //T`).
recon_longrun arg1 = state BASENAME **with .state**, arg2 = max_minutes, goal via `LONGRUN_GOAL_FLAG=0x825`.
Flags module = `field_moves` (fm.read_flag). Silph anchors = {(3,10)} | SILPH_MAPS ((1,47)..(1,58)).

WATCH STATUS: canonical Champion bank CLEAN + untouched. Sherpa frontier = post-Koga (badges=5) at Fuchsia,
Silph liberation questline WIRED, verification long-run in flight. Pop-in = `python pokemon_agent/watch.py`.
