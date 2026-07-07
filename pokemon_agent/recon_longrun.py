"""recon_longrun.py — THE LOOK-AHEAD ORACLE + RESUMABLE CHECKPOINT (standing autonomous-play harness).

The reusable way we verify + advance autonomous play. Instead of Jonny watching a 2-hour real-pace
watch, this runs her REAL play loop headless at MAX emulator speed (no audio/TTS/render throttle →
~14x real-time, measured) and logs the sped-up playthrough so Claude Code reads what ACTUALLY happens
over a long horizon, then fixes the real next blocker.

PIECE 1 — LOOK-AHEAD: boots her current real save and runs the REAL free_roam loop (forward-drive,
questline, strategic underlevel-grind, blackout recovery, watchdog — all of it). The per-tick oracle
PICK is filled by a faithful local chooser that FOLLOWS the machinery's steering (the heavy ctx framing
already encodes the "right" move; the LLM adds voice, not capability). Pluggable: point `_oracle_choose`
at the bot's HTTP endpoint for an LLM-fidelity pass. Runs LONG — until the GOAL (S.S. Ticket flag) or a
genuine STALL (no progress across many decisions), not a fixed tick cutoff.

PIECE 2 — RESUMABLE CHECKPOINT: the canonical living save is PROTECTED — all in-run persistence is
redirected to a STAGING dir, so an experimental look-ahead NEVER clobbers her real timeline. On the run's
end the staged savestate + sidecars (world-model, strat/loss history, soul) are banked to a named
checkpoint and ROUND-TRIP verified (load → party/levels/flags/coords identical). The real Sherpa timeline
can then resume from that checkpoint — progress made headless becomes real forward progress. (Promotion to
the canonical save is left to Jonny — never silently overwrite the spine.)

RUN:  python pokemon_agent/recon_longrun.py [boot_state] [max_minutes]
  boot_state   default = kira_campaign.state (her real save). A banked checkpoint name resumes from it.
  max_minutes  default = 20  (wall-clock cap; ~14x → ~4.5 game-hours).
"""
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"          # no muse-gap dwell headless
os.environ.setdefault("POKEMON_GRIND_WEAK_BUDGET_S", "3000")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import pokemon_state as st                                # noqa: E402
import travel as tv                                       # noqa: E402
import field_moves as fm                                  # noqa: E402
from battle_agent import BattleAgent                      # noqa: E402
import campaign as C                                      # noqa: E402
from campaign import Campaign, resolve_state, STATES_CAMPAIGN  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
PMON = st.PARTY_MON_SIZE
SS_TICKET = 0x234                                          # FLAG_GOT_SS_TICKET — the pre-ticket goal signal
STALL_DECISIONS = 14                                      # no-progress decisions in a row = a genuine stall
# GOAL CONFIG (2026-07-06, post-ticket era): the goal is now configurable per stretch —
#   LONGRUN_GOAL_MAP="3,23"  -> GOAL when she stands on that map (e.g. Route 5 for the south gate)
#   LONGRUN_GOAL_FLAG="0x234"-> GOAL when that flag TRANSITIONS to set during the run
# Default (neither set): the classic ticket-flag goal, but only if it was UNSET at boot — a save that
# already holds the ticket must not insta-GOAL (that was the hardcode's failure mode).
_gm = os.getenv("LONGRUN_GOAL_MAP", "")
GOAL_MAP = tuple(int(x) for x in _gm.split(",")) if _gm else None
GOAL_FLAG = int(os.getenv("LONGRUN_GOAL_FLAG", "0"), 0) or None

SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage")                    # in-run persistence lands HERE (canonical untouched)


class _Done(Exception):
    def __init__(self, why):
        self.why = why


def _party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    out = []
    for s in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + s * PMON
        out.append((st.SPECIES_NAME.get(st.read_party_species(b, s), f"#{s}").title(),
                    b.rd8(base + 0x54), b.rd16(base + 0x56), b.rd16(base + 0x58)))
    return out


def _badges(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    return sum(1 for i in range(8) if (b.rd8(sb1 + 0x0EE0 + ((0x820 + i) >> 3)) >> ((0x820 + i) & 7)) & 1)


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "kira_campaign.state"
    max_minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    os.makedirs(STAGE, exist_ok=True)
    # fresh stage each run
    # SINGLE-RUN LAW (2026-07-06, Jonny's directive): ONE live look-ahead at a time — a new launch
    # REAPS its predecessor first. Zombie runs kept farming after their session died, wrote stale
    # banks, and their watchers ghost-notified later sessions (the false-"Cut is live" incident).
    pidf = os.path.join(SCRATCH, "longrun.pid")
    try:
        if os.path.exists(pidf):
            old = int(open(pidf).read().strip() or 0)
            if old and old != os.getpid():
                chk = subprocess.run(["tasklist", "/FI", f"PID eq {old}"],
                                     capture_output=True, text=True).stdout
                if str(old) in chk and "python" in chk.lower():
                    subprocess.run(["taskkill", "/F", "/PID", str(old)], capture_output=True)
                    print(f"[reap] SINGLE-RUN LAW: killed predecessor look-ahead PID {old}", flush=True)
        with open(pidf, "w") as pf:
            pf.write(str(os.getpid()))
    except Exception as _re:
        print(f"[reap] predecessor check failed: {_re}", flush=True)
    # …and stale WATCHER loops (2026-07-06 ground-truth lesson): watchers are BASH until/tail-grep
    # loops, not python — the PID file never covered them, and ones launched by a DEAD session have
    # no live handle anywhere. Kill bash processes whose commandline references our longrun logs and
    # that are older than 2h (no legit watcher outlives a 75-min run; the age gate protects this
    # launch's own shell chain and any interactive shells).
    try:
        _ps = ("Get-CimInstance Win32_Process -Filter \"name='bash.exe'\" | "
               "Where-Object { $_.CommandLine -match 'longrun' -and "
               "((Get-Date) - $_.CreationDate).TotalHours -gt 2 } | "
               "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "
               "Write-Output ('reaped watcher ' + $_.ProcessId) }")
        _out = subprocess.run(["powershell", "-NoProfile", "-Command", _ps],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        if _out:
            print(f"[reap] {_out}", flush=True)
    except Exception as _rw:
        print(f"[reap] watcher sweep failed: {_rw}", flush=True)

    for f_ in os.listdir(STAGE):
        try:
            os.remove(os.path.join(STAGE, f_))
        except Exception:
            pass

    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    T0 = time.time()
    log_lines = []

    def L(msg):
        line = f"[{time.time()-T0:7.1f}s] {msg}"
        log_lines.append(line)
        print(line, flush=True)

    battles = []          # per real battle: fielded mon state AFTER + outcome

    blog = (print if os.getenv("LONGRUN_BATTLE_LOG") == "1" else (lambda m: None))

    def runner():
        # IMPORTANT: pass choose=chooser so the engine can consult the oracle for IN-BATTLE ITEM use
        # (heal when crit-low). Without it _maybe_use_item bails (self.choose is None) and she never
        # heals through a fight — the false "she can't barge" result. chooser is in scope via closure.
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=blog, choose=chooser).run(max_seconds=180)
        base = ram.GPLAYER_PARTY
        sp = st.SPECIES_NAME.get(st.read_party_species(b, 0), "?").title()
        hp, lv = b.rd16(base + 0x56), b.rd8(base + 0x54)
        battles.append({"t": round(time.time() - T0, 1), "out": out, "lead": sp,
                        "lvl": lv, "hp": hp, "fainted": hp == 0})
        return out

    events = []
    camp = Campaign(b, battle_runner=runner,
                    on_event=lambda s, **k: events.append((round(time.time() - T0, 1), k.get("kind", ""), s)),
                    beat=lambda *a, **k: None, render=lambda: None)

    # ── PIECE 2: PROTECT the canonical save — redirect ALL in-run persistence to STAGE. The canonical
    # living save + sidecars are NEVER written during an experimental look-ahead. ────────────────────
    def _stage_save(reason="tick"):
        try:
            data = b.save_state()
            with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            L(f"!! STAGE SAVE FAILED [{reason}]: {e}")
            return False

    def _stage_continuity():
        try:
            camp.world.save(os.path.join(STAGE, "world_model.json"))
        except Exception:
            pass
        try:
            camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
        except Exception:
            pass
        try:
            if camp.soul is not None:
                camp.soul.save(os.path.join(STAGE, "soul.json"))
        except Exception:
            pass
        try:                                                  # RULE 17 — bank her saga WITH the checkpoint
            import json as _json
            with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
                _json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
    camp._continuity_load = lambda *a, **k: None          # boot already loaded the live save's RAM
    # load her real mental map so routing/forward-drive have her learned graph (read-only from canonical)
    try:
        camp.world.load(C.WORLD_JSON)
    except Exception:
        pass
    # ── CONTINUITY SEED (2026-07-06, the bond-regression class): the no-op load above also skipped
    # the SOUL, so every run started with EMPTY bonds and banked soul={} unless a catch happened
    # mid-run — sanctity then (rightly) refused the promotion (bonds 1->0, the Meowth bond). Seed
    # the subjective layer from the BOOT bundle's own sidecar (a named bank) else canonical, so her
    # story carries INTO the run and ACCRETES instead of resetting. ────────────────────────────────
    try:
        _boot_dir = os.path.dirname(resolve_state(boot))
        _soul_src = os.path.join(_boot_dir, "soul.json")
        if not os.path.exists(_soul_src):
            _soul_src = os.path.join(STATES_CAMPAIGN, "soul.json")
        if camp.soul is not None and os.path.exists(_soul_src):
            camp.soul.load(_soul_src)
            L(f"soul continuity seeded from {_soul_src} ({len(camp.soul.bonds)} bond(s))")
    except Exception as _se:
        L(f"!! soul seed failed: {_se}")
    try:
        camp.strat.load(C.STRAT_JSON)
    except Exception:
        pass
    # NOTE: trav.battle_runner LEFT as the real fighting runner (default) so grind wilds are FOUGHT for XP.

    # ── BARGE PREMISE TEST (LONGRUN_BARGE=1): does Ivysaur + Potions beat the Cerulean/Nugget-Bridge
    # Gary and reach Bill, WITHOUT grinding? Injects N potions (sim of a Cerulean-Mart stock-up the game
    # gap currently blocks), clears the stale Gary wall (so the machinery offers the forward push instead
    # of pruning to an ineffective grind), and the chooser prefers FORWARD. Diagnostic config of the
    # standing tool — proves/refutes the barge path before we build the Mart+strategy autonomy. ──────────
    BARGE = os.getenv("LONGRUN_BARGE", "0") == "1"
    npot = int(os.getenv("LONGRUN_POTIONS", "0"))
    if npot > 0:                                          # inject potions (sim of a Mart stock-up) — works
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)                 # with OR without clearing the wall
        key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for s in range(42):
            slot = sb1 + 0x0310 + s * 4
            iid = b.rd16(slot)
            if iid in (13, 0):                            # Potion or empty
                b.core.memory.u16.raw_write(slot, 13)
                b.core.memory.u16.raw_write(slot + 2, npot ^ key)
                break
    if BARGE:                                             # forward-push test: clear the stale wall so the
        camp.strat.active_wall = None                     # machinery offers the push instead of grind
        camp.strat.losses = {}

    visited = set()
    decisions = [0]
    last_sig = [None]
    stall_run = [0]
    # was the goal-flag ALREADY set at boot? (a held ticket must not insta-GOAL — flag goals fire on
    # the TRANSITION only)
    boot_goal_flag = [bool(fm.read_flag(b, GOAL_FLAG if GOAL_FLAG is not None else SS_TICKET))]
    start_party = _party(b)
    start_levels = [lv for _, lv, _, _ in start_party]

    def progress_sig():
        lv = [b.rd8(ram.GPLAYER_PARTY + s * PMON + 0x54)
              for s in range(min(b.rd8(ram.GPLAYER_PARTY_CNT), 6))]
        return (_badges(b), 1 if fm.read_flag(b, SS_TICKET) else 0, sum(lv),
                b.rd8(ram.GPLAYER_PARTY_CNT), len(visited))

    def chooser(kind, options, ctx):
        # IN-BATTLE ITEM use (kind="battle_item"): the engine offers a heal/cure when a mon is crit-low.
        # Her real LLM, told "you're about to faint and you HAVE one", uses it — so a FAITHFUL look-ahead
        # must too, or it under-tests her (a barge/gauntlet depends on healing through). Use the offered
        # item (prefer the heal). Returning None here = never heal = the false "she can't barge" result.
        if kind == "battle_item":
            if isinstance(options, dict) and options:
                return ("use_potion" if "use_potion" in options else next(iter(options.keys())))
            return None
        if kind != "action":
            return None                                   # naming/want — let it default headless
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        decisions[0] += 1
        mp, co = tv.map_id(b), tv.coords(b)
        visited.add(mp)
        party = _party(b)
        prep = None
        try:
            prep = camp._prep_team_target(camp.read_live_state())
        except Exception:
            prep = None
        ql = camp._active_questline
        ql_doing = (ql.actionable.human if (ql and ql.actionable) else None)
        b_desc = options.get("battle", "") if isinstance(options, dict) else ""
        h_desc = options.get("head_to_gym", "") if isinstance(options, dict) else ""
        # GOAL — configurable per stretch (map reached / flag transition / classic ticket default).
        if GOAL_MAP is not None:
            if mp == GOAL_MAP:
                raise _Done(f"GOAL: reached map {GOAL_MAP} at {co}")
        elif GOAL_FLAG is not None:
            if fm.read_flag(b, GOAL_FLAG) and not boot_goal_flag[0]:
                raise _Done(f"GOAL: flag 0x{GOAL_FLAG:X} set")
        elif fm.read_flag(b, SS_TICKET) and not boot_goal_flag[0]:
            raise _Done("GOAL: S.S. Ticket obtained (FLAG_GOT_SS_TICKET set)")
        # STALL — no progress on any dimension across STALL_DECISIONS consecutive decisions.
        sig = progress_sig()
        if sig == last_sig[0]:
            stall_run[0] += 1
        else:
            stall_run[0] = 0
            last_sig[0] = sig
        # faithful chooser — model a SENSIBLE player facing an under-levelled wall:
        #   EQUIP the cheap consumable FIRST (stock_up: potions+balls are right there, cheap, fast) ->
        #   once STOCKED, TRY the wall with your items (head_to_gym: a Potion+status-stall fight can win
        #   where an empty-handed charge didn't) -> only GRIND (prep battle) when you can't shop and have
        #   no potions -> else the survive/advance/grind/catch/talk fallback. This fixes the prior run's
        #   trap where the prep-grind override sent her to grind the ACE in east Route-4 grass she couldn't
        #   heal out of, instead of shopping + trying Gary with potions.
        try:
            pots = camp.bag_count(C.ITEM_POTION) + camp.bag_count(22)   # Potion + Super Potion
        except Exception:
            pots = 0
        pick = None
        if len(opts) == 1:
            pick = opts[0]
        elif "stock_up" in opts:
            pick = "stock_up"                              # EQUIP first (cheap consumable, self-limiting)
        elif (not BARGE) and "wander_catch" in opts and len(party) < 4 \
                and camp._balls_pocket_count(C.ITEM_POKE_BALL) > 0:
            pick = "wander_catch"                          # NURSERY (2026-07-06): thin team + balls +
            #                                                fresh grass -> a sensible player builds the
            #                                                squad FIRST (judgment-gated; dupes skipped)
        elif (not BARGE) and prep is not None and "battle" in opts:
            pick = "battle"                                # underlevelled + items-alone can't beat the
            #                                                Smokescreen Charmander -> GRIND/team-build first
        else:
            # travel:* BEFORE talk_npc (run-4 lesson: at the post-grind Route-4 spot the opts collapsed to
            # ['talk_npc','travel:3,3'] and the old order spammed no_npc talk 7x instead of walking to
            # Cerulean — a real player moves toward somewhere useful before re-talking an empty route).
            # HEAL only when someone's actually CRITICAL/fainted (2026-07-06 nursery run-4 lesson: a
            # 43%-HP Spearow made the chooser slavishly re-pick heal in a pocket that can't reach a
            # Center — a real player pushes to the NEXT town's Center when merely dinged).
            critical = any(hp == 0 or (mx and hp / mx < 0.25) for _, _, hp, mx in party)
            for pref in (("heal",) if critical else ()) + ("head_to_gym", "battle", "wander_catch", "heal"):
                if pref in opts:
                    pick = pref
                    break
            if pick is None:
                pick = next((o for o in opts if o.startswith("travel:")),
                            "talk_npc" if "talk_npc" in opts else (opts[0] if opts else None))
        L(f"#{decisions[0]:>3} {str(mp):>9}@{str(co):<9} party={[ (s[:4],lv) for s,lv,_,_ in party]} "
          f"prep={prep} ql={ql_doing!r}")
        L(f"      opts={sorted(opts)} -> PICK {pick}"
          + (f"  [battle: {b_desc[:70]}]" if pick == "battle" and b_desc else "")
          + (f"  [h2g: {h_desc[:70]}]" if pick == "head_to_gym" and h_desc else ""))
        if stall_run[0] >= STALL_DECISIONS:
            raise _Done(f"STALL: no progress for {STALL_DECISIONS} decisions (sig={sig}) at {mp}@{co}; "
                        f"last opts={sorted(opts)}")
        return pick
    camp._oracle_choose = chooser

    L("==== LOOK-AHEAD ORACLE — autonomous Nugget-Bridge stretch ====")
    L(f"boot={boot} map={tv.map_id(b)}@{tv.coords(b)} badges={_badges(b)} "
      f"ticket_flag={bool(fm.read_flag(b, SS_TICKET))}")
    L(f"START party: {start_party}")
    L(f"speed: ~14x real-time headless; wall-clock cap {max_minutes:.0f} min\n")

    why = "timeout (wall-clock cap)"
    try:
        camp.free_roam(max_ticks=100000, max_seconds=int(max_minutes * 60), want_every=999)
    except _Done as d:
        why = d.why
    except Exception as e:
        import traceback
        why = f"CRASH: {e}"
        L("!! CRASH TRACE:\n" + traceback.format_exc())

    # ── outcome ──
    elapsed = round(time.time() - T0, 1)
    end_party = _party(b)
    end_levels = [lv for _, lv, _, _ in end_party]
    L(f"\n==== END: {why} | {elapsed}s wall | {decisions[0]} decisions | {len(battles)} real battles ====")
    L(f"PARTY  start: {start_party}")
    L(f"PARTY  end  : {end_party}")
    L(f"LEVELS start {start_levels} -> end {end_levels} (floor {min(start_levels)}->{min(end_levels)}, "
      f"sum {sum(start_levels)}->{sum(end_levels)})")
    L(f"badges {_badges(b)} | dex {ram.pokedex_owned_count(b)} | ticket_flag {bool(fm.read_flag(b, SS_TICKET))} | maps visited {len(visited)}: "
      f"{sorted(visited)}")

    # which mon got fielded in battles (the strategic-grind question)
    lead_counts = {}
    faints = {}
    for e in battles:
        lead_counts[e["lead"]] = lead_counts.get(e["lead"], 0) + 1
        if e["fainted"]:
            faints[e["lead"]] = faints.get(e["lead"], 0) + 1
    L(f"battles by FIELDED lead: {lead_counts} | faints (one-shot/KO'd as lead): {faints}")
    L("\nlast 25 events (her voice/beats):")
    for t, kind, s in events[-25:]:
        L(f"   {t:7.1f}s [{kind}] {s[:110]}")

    # ── PIECE 2: bank the staged checkpoint + round-trip verify ──
    outcome_tag = ("GOAL" if why.startswith("GOAL") else "STALL" if why.startswith("STALL")
                   else "TIMEOUT" if why.startswith("timeout") else "CRASH")
    _stage_save("final")
    _stage_continuity()
    banked = os.path.join(SCRATCH, f"banked_{outcome_tag}")
    if os.path.isdir(banked):
        shutil.rmtree(banked, ignore_errors=True)
    shutil.copytree(STAGE, banked)
    # round-trip: load the banked savestate fresh, confirm party/flags/coords identical
    rt = "n/a"
    try:
        b2 = Bridge(ROM)
        with open(os.path.join(banked, "kira_campaign.state"), "rb") as f:
            b2.load_state(f.read())
        for _ in range(20):
            b2.run_frame()
        rt_ok = (_party(b2) == end_party and _badges(b2) == _badges(b)
                 and bool(fm.read_flag(b2, SS_TICKET)) == bool(fm.read_flag(b, SS_TICKET)))
        rt = f"{'OK' if rt_ok else 'MISMATCH'} (party/badges/flag {'match' if rt_ok else 'DIFFER'})"
    except Exception as e:
        rt = f"FAILED: {e}"
    # RIDE-ALONG 0c (2026-07-06): SANCTITY VALIDATION — story corruption fails a bank like a wedge
    # fails a run. prev = the canonical bundle this bank would replace (monotonic story guard).
    try:
        import sanctity
        sane, s_issues = sanctity.validate_bundle(
            banked, prev_dir=STATES_CAMPAIGN, live_badges=_badges(b), log=L)
    except Exception as _se:
        sane, s_issues = False, [f"validator crashed: {_se}"]
        L(f"   [sanctity] !! validator crashed: {_se}")
    L(f"\nBANKED checkpoint -> {banked}  (savestate + world_model/strat/soul/journey sidecars)")
    L(f"  round-trip (save->load->identical): {rt}")
    L(f"  sanctity: {'VALID' if sane else 'FAILED — DO NOT PROMOTE (' + '; '.join(s_issues)[:200] + ')'}")
    L(f"  canonical save UNTOUCHED (look-ahead persisted only to {STAGE})")
    L(f"  to advance the real Sherpa timeline: promote {banked}/* into {STATES_CAMPAIGN}/ (Jonny's call)")

    # write the full sped-up playthrough log to a file for reading
    logpath = os.path.join(SCRATCH, f"playthrough_{outcome_tag}.log")
    with open(logpath, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n==== full log: {logpath} ====", flush=True)
    print(f"==== OUTCOME: {outcome_tag} — {why} ====", flush=True)


if __name__ == "__main__":
    main()
