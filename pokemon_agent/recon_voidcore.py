"""recon_voidcore.py — clean single-run repro of the QW-4 VOID-CORE class (2026-07-08 night shift).

THE BUG (logs/quietwindow_summit.log): mid-watch, position/party/badge reads went inconsistent —
whole ticks read a DEAD world (map (0,0), coords None, party 0, badges 0, $0, facing 0) alternating
with valid Champion-world ticks, plus impossible position jumps INSIDE valid ticks. The run happily
"played" (and stage-saved) the dead world. CONFOUND to exclude: a concurrent throwaway take was
killed mid-summit-run.

THIS HARNESS: boots banked_CREDITS (the summit bundle), runs the REAL free_roam loop HEADLESS with
a SCRIPTED oracle that mirrors the summit run's picks (leave_building → battle → battle → travel →
travel → wander_catch …) AND a synthetic frame-pump (~2s of zero-input frames before every decision
— exactly what the watch-rig pump does while the LLM thinks). No bot needed. Single process — if
the void still appears, the concurrent-kill confound is EXCLUDED and the mechanism is internal.

TRIPWIRES (mechanism caught at the exact frame, not theorized):
  T1 — frame-counter REGRESSION guard: wraps core.run_frame; the host frame counter only grows on
       a live core, so any decrease = core reset OR load_state. PNG + dump on trip.
  T2 — every Bridge.load_state call logged LOUD with its caller (any legit rewind is attributed).
  T3 — per-tick VOID probe: wraps camp.read_live_state; a dead-world signature after a valid tick
       → PNG of the actual screen (title? credits? black?) + raw sb1/sb2 pointer dump. The PNG is
       the verdict: arsenal #4, grab a frame and LOOK.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_voidcore.py [max_minutes]
"""
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
# sandbox ALL campaign-dir writes (incl. escape-hatch pre_reload backups — they leaked into the
# real states/campaign on the first runs); must be set BEFORE campaign is imported.
os.environ["POKEMON_CAMPAIGN_DIR"] = os.path.join(
    os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe", "campaign_sandbox")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
import firered_ram as ram                                   # noqa: E402
import pokemon_state as st                                  # noqa: E402
import travel as tv                                         # noqa: E402
from battle_agent import BattleAgent                        # noqa: E402
import campaign as C                                        # noqa: E402
from campaign import Campaign, resolve_state                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
SNAPDIR = os.path.join(SCRATCH, "voidcore_probe")
STAGE = os.path.join(SNAPDIR, "stage")

T0 = time.time()


def L(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main():
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    os.makedirs(SNAPDIR, exist_ok=True)
    os.makedirs(STAGE, exist_ok=True)

    boot = os.path.join(SCRATCH, "banked_CREDITS", "kira_campaign.state")
    if not os.path.exists(boot):
        boot = resolve_state("banked_CREDITS")
    b = Bridge(ROM)
    with open(boot, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    # boot-drain (mimic play_live's resume): the HoF bank can hold a live textbox — a raw load
    # leaves it up and the building-exit then fails ONLY in the harness (QW-4 passed live).
    for _ in range(10):
        b.press("B", 6, 8)
    L(f"boot banked_CREDITS: map={tv.map_id(b)} coords={tv.coords(b)} "
      f"party={b.rd8(ram.GPLAYER_PARTY_CNT)} frame={b.frame}")

    trips = {"t1": 0, "t3": 0, "loads": 0}

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(SNAPDIR, name + ".png"))
            L(f"   frame -> {SNAPDIR}\\{name}.png")
        except Exception as e:
            L(f"   snap failed: {e}")

    def raw_dump(tag):
        try:
            sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
            sb2 = b.rd32(ram.GSAVEBLOCK2_PTR)
            L(f"   [{tag}] sb1={sb1:#010x} (valid={ram.valid_ewram_ptr(sb1)}) "
              f"sb2={sb2:#010x} (valid={ram.valid_ewram_ptr(sb2)}) "
              f"frame={b.frame} party_cnt={b.rd8(ram.GPLAYER_PARTY_CNT)} "
              f"map={tv.map_id(b)} coords={tv.coords(b)}")
        except Exception as e:
            L(f"   [{tag}] raw dump failed: {e}")

    # ── T1: frame-counter regression guard (host counter never decreases on a live core) ────────
    _orig_rf = b.core.run_frame
    _last_f = [b.core.frame_counter]

    def _guarded_rf():
        _orig_rf()
        f = b.core.frame_counter
        if f < _last_f[0]:
            trips["t1"] += 1
            L(f"!!!! T1 FRAME-COUNTER REGRESSION #{trips['t1']}: {_last_f[0]} -> {f} "
              f"(core reset or state load). Caller:")
            for ln in traceback.format_stack(limit=8)[:-1]:
                L("      " + ln.strip().replace("\n", " | "))
            raw_dump("t1")
            snap(f"t1_regress_{trips['t1']}")
        _last_f[0] = f
    b.core.run_frame = _guarded_rf

    # ── T2: attribute every load_state ─────────────────────────────────────────────────────────
    _orig_ls = b.load_state

    def _loud_ls(data):
        trips["loads"] += 1
        c = traceback.extract_stack(limit=3)[0]
        L(f"!! T2 load_state #{trips['loads']} <- {os.path.basename(c.filename)}:{c.lineno}")
        return _orig_ls(data)
    b.load_state = _loud_ls

    # ── real campaign, staged persistence (canonical untouched) ────────────────────────────────
    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: None, choose=None).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)

    def _stage_save(reason="tick"):
        try:
            with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
                f.write(b.save_state())
            return True
        except Exception as e:
            L(f"!! stage save failed: {e}")
            return False
    camp._save_campaign = _stage_save
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(C.WORLD_JSON)          # her real learned graph, read-only
    except Exception:
        pass

    # ── T3: void-world probe on the loop's own state reads ─────────────────────────────────────
    _orig_rls = camp.read_live_state
    _had_valid = [False]

    def _probed_rls():
        s = _orig_rls()
        mp = tuple(s.get("map") or (0, 0))
        void = (mp == (0, 0) and not s.get("party"))
        if void and _had_valid[0]:
            trips["t3"] += 1
            L(f"!!!! T3 VOID READ #{trips['t3']} after a valid tick — dead-world signature")
            raw_dump("t3")
            snap(f"t3_void_{trips['t3']}")
        elif not void:
            _had_valid[0] = True
        return s
    camp.read_live_state = _probed_rls

    # ── scripted oracle mirroring the summit picks, with the watch-pump simulated ───────────────
    seq = {"n": 0, "move_drops": 0}
    ORDER = ("battle", "battle", "travel", "travel", "wander_catch", "battle", "travel")

    def pump(frames=120):
        for _ in range(frames):
            b.run_frame()

    def chooser(kind, options, ctx):
        pump(120)                              # ~2s of zero-input frames = the LLM think, pumped
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        if not opts and kind != "want":
            return None
        if kind == "move_drop":
            seq["move_drops"] += 1
            keep = next((o for o in opts if "keep" in o.lower()), None)
            pick = keep if (seq["move_drops"] == 1 and keep) else opts[0]
            L(f"[chooser] move_drop #{seq['move_drops']} -> {pick!r}")
            return pick
        if kind == "want":
            return "catch Mewtwo"
        if kind != "action":
            return opts[0] if opts else None
        if "leave_building" in opts:
            return "leave_building"
        want = ORDER[seq["n"] % len(ORDER)]
        seq["n"] += 1
        if want == "travel":
            pick = next((o for o in opts if o.startswith("travel:")), None)
            if pick:
                return pick
        if want in opts:
            return want
        return next((o for o in opts if o.startswith("travel:")), opts[0])

    camp._oracle_choose = chooser

    L(f"== FREE ROAM (scripted, pumped, headless) for {max_minutes:.0f} min ==")
    try:
        out = camp.free_roam(max_ticks=100000, max_seconds=int(max_minutes * 60), want_every=3)
    except Exception as e:
        L(f"!!!! free_roam crashed: {e!r}")
        L(traceback.format_exc())
        out = "crashed"

    L(f"== DONE: roam={out} | T1 frame-regressions={trips['t1']} | T3 void-reads={trips['t3']} | "
      f"load_state calls={trips['loads']} ==")
    L("VERDICT: " + ("VOID-CORE REPRODUCED — see PNGs in " + SNAPDIR if (trips["t1"] or trips["t3"])
                     else "NO void in a clean single run — concurrent-kill confound implicated"))


if __name__ == "__main__":
    main()
