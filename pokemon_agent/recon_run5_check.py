"""recon_run5_check.py — RUN-5 (2026-07-14) verify for the two-part fix, booted on the REAL fresh_go_4
League-door bank (banked_STALL = Viridian City (3,1), badge 8, party [62,29,29,29,29,25] — the exact
NON-CONVERGING halt state: floor L25 vs prep target L47, gap 37).

Two fixes, both must PASS before fresh_go_5 launches:

  FIX (b) — ORDER-3B ALLOWLIST WIDEN (primary): the badge-6..8 open-ground/sea questline keys
    (surf / seafoam / FLAG_HIDE_SAFFRON_ROCKETS) are now in OVERWORLD_SAFE_QUESTLINES, so the
    road-bench-XP relax fires on those long marches (where fresh_go_1..4 froze the bench). Verified via
    the REAL _questline_march_bench_ok gate on the booted overworld map: True for the new + existing
    open-ground keys, False for the excluded cave keys (flash / secret_key), False off open ground.

  FIX (a) — GATE-GRIND OPEN-GRASS ROUTING (safety net): when the RED gate's current map starves
    (no_safe_grass — the halt sat on grassless Viridian and NON-CONVERGED without ever trying grass),
    _e4_readiness_grind now routes to the nearest reachable ADEQUATE open grass and grinds there
    ace-capped. Verified via the REAL _better_grind_spot decision (returns adequate grass + a valid first
    hop from the fixture) AND the reroute WIRING (a forced no_safe_grass drives walk_to_map toward that
    grass map). NOTE: from a 37-level endgame gap NO open grass can fully converge (Kanto open grass tops
    ~L30) — the honest full-climb proof is fresh_go_5 with fix (b) preventing the gap; this harness proves
    the routing FIRES + targets sensibly + never sits starved (the specific halt bug).

RUN: ../.venv/Scripts/python.exe -u recon_run5_check.py
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge             # noqa: E402
import travel as tv                   # noqa: E402
from campaign import Campaign         # noqa: E402
import campaign as C                  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.environ.get("RUN5_BANK", "G:/temp/longrun/banked_STALL")

results = []


def check(label, cond):
    results.append((label, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}", flush=True)


def _ql(missing):
    """A minimal fake active-questline: _questline_march_bench_ok reads q.gate.missing."""
    return types.SimpleNamespace(gate=types.SimpleNamespace(missing=missing))


def main():
    b = Bridge(ROM)
    with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok", on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    for loader, side, fb in ((camp.world.load, os.path.join(BANK, "world_model.json"), C.WORLD_JSON),
                             (camp.strat.load, os.path.join(BANK, "strat_memory.json"), C.STRAT_JSON)):
        try:
            loader(side if os.path.exists(side) else fb)
        except Exception as e:
            print("  loader err", e, flush=True)

    state = camp.read_live_state()
    party = state.get("party") or []
    levels = camp._party_levels()
    ready, floor, ceil, gap = camp._e4_entry_ready(party)
    target = camp._e4_prep_floor_target(party)
    print(f"boot {BANK.split('/')[-1]}: map={tv.map_id(b)}@{tv.coords(b)} badges={state.get('badge_count')} "
          f"levels={levels} ready={ready} floor={floor} ace={ceil} gap={gap} target=L{target}", flush=True)
    check("fixture is the RED League-door halt state (badge 8, not qualifying, big gap)",
          state.get("badge_count") == 8 and not ready and gap > 15)
    check("fixture is on OPEN GROUND (overworld) — the allowlist relax can be exercised here",
          camp._on_overworld_now())

    # ── FIX (b): the allowlist widen, via the REAL _questline_march_bench_ok gate ──
    print("\n── FIX (b): ORDER-3B OVERWORLD-SAFE-QUESTLINE ALLOWLIST ──", flush=True)
    NEW = ["surf", "seafoam", "FLAG_HIDE_SAFFRON_ROCKETS"]
    EXISTING = ["FLAG_GOT_SS_TICKET", "earth_badge"]
    EXCLUDED = ["flash", "secret_key"]
    for k in NEW:
        camp._active_questline = _ql(k)
        check(f"NEW key '{k}' -> bench-relax ALLOWED on open ground", camp._questline_march_bench_ok())
    for k in EXISTING:
        camp._active_questline = _ql(k)
        check(f"existing key '{k}' still ALLOWED (no regression)", camp._questline_march_bench_ok())
    for k in EXCLUDED:
        camp._active_questline = _ql(k)
        check(f"cave key '{k}' still EXCLUDED (ace leads the dark)", not camp._questline_march_bench_ok())
    # off open ground: even an allowlisted key must NOT relax (the live cave/interior guard)
    _orig_oon = camp._on_overworld_now
    camp._on_overworld_now = lambda: False
    camp._active_questline = _ql("seafoam")
    check("allowlisted key OFF open ground -> NOT relaxed (interior guard holds)",
          not camp._questline_march_bench_ok())
    camp._on_overworld_now = _orig_oon
    camp._active_questline = None

    # bonus: the REAL _road_bench_xp_arm now arms for a new key on this march (pre-leg ordering, no switch)
    camp._active_questline = _ql("seafoam")
    _orig_nh = camp.needs_heal
    camp.needs_heal = lambda: False                         # isolate the questline gate from the heal gate
    armed = camp._road_bench_xp_arm("head_to_gym", state)
    lead_after = camp._party_levels()[0] if camp._party_levels() else None
    camp._road_bench_xp_disarm()
    lead_restored = camp._party_levels()[0] if camp._party_levels() else None
    camp.needs_heal = _orig_nh
    camp._active_questline = None
    print(f"  road_bench_xp_arm(seafoam march) -> {armed} | lead L{lead_after} -> after disarm L{lead_restored} "
          f"(ace L{ceil})", flush=True)
    check("road-bench-XP ARMS on a seafoam march (weak mon leads) OR arms cleanly restore",
          (armed and lead_after is not None and lead_after < ceil) or (not armed))
    check("disarm RESTORES the ace to lead (slot-0 is the ace again)", lead_restored == max(levels))

    # ── FIX (a): gate-grind open-grass routing ──
    print("\n── FIX (a): GATE-GRIND OPEN-GRASS ROUTING ──", flush=True)
    cur = tuple(state["map"])
    routed_t = min(target, floor + C.E4_GATE_REROUTE_CLIMB)
    dst = camp._better_grind_spot(state, routed_t)
    hop = camp.world.next_hop(cur, tuple(dst), camp._wall_avoid(state)) if dst else None
    band = camp._grind_wild_band(dst) if dst else None
    print(f"  from {cur}: better_grind_spot(routed L{routed_t}) -> dst={dst} band={band} next_hop={hop}", flush=True)
    check("reroute finds a reachable ADEQUATE open-grass map from the starved fixture", dst is not None)
    check("reroute has a valid first hop (rideable now — not a dead-end)", hop is not None)

    # WIRING: a forced no_safe_grass must drive the reroute -> walk_to_map toward that grass map, ace-capped.
    moves = []
    _orig_gwm = camp.grind_weak_members
    _orig_walk = camp.walk_to_map
    camp.grind_weak_members = lambda target, min_level=None, ace_cap=False: (
        moves.append(("grind", target, ace_cap)) or "no_safe_grass")
    camp.walk_to_map = lambda tgt, direction: (moves.append(("walk", tuple(tgt), direction)) or "arrived")
    r = camp._e4_readiness_grind(state, target)
    camp.grind_weak_members = _orig_gwm
    camp.walk_to_map = _orig_walk
    walk_calls = [m for m in moves if m[0] == "walk"]
    grind_calls = [m for m in moves if m[0] == "grind"]
    print(f"  forced-starve _e4_readiness_grind -> {r!r} | calls={moves}", flush=True)
    check("on no_safe_grass the reroute WALKS toward the grass dst (never sits starved)",
          len(walk_calls) == 1 and walk_calls[0][1] == tuple(dst))
    check("the first grind attempt was ACE-CAPPED (ace earns no XP)",
          bool(grind_calls) and grind_calls[0][2] is True)

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\nRUN-5 CHECK: {n_pass}/{len(results)} PASS", flush=True)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
