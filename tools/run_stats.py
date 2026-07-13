#!/usr/bin/env python3
"""run_stats.py — self-contained stats generator for a recon_longrun.py playthrough log.

recon_longrun.py (pokemon_agent/) runs Kira's real free_roam / fresh-spine play loop headless at
~14x and prints a per-decision oracle trace plus per-segment outcome blocks. A watchdog
(fresh_go_watchdog.sh) relaunches it in a loop, appending everything to ONE big log. This script
streams that log (line-by-line — it never loads the whole file) and emits a Markdown stats report.

USAGE:
    python tools/run_stats.py <log_path> [out_path]

    <log_path>  the recon_longrun watchdog log (e.g. G:/temp/longrun/fresh_go_1.log)
    out_path    optional; default RUN_STATS_<logbasename>.md in the repo root (cwd).

DESIGN NOTES / LINE FORMATS PARSED (verified against fresh_go_1.log):
  Watchdog markers (from fresh_go_watchdog.sh):
    "=== fresh_go watchdog start [<date>] ==="        -> a watchdog (re)start
    "=== launch iter N boot=<state> [<date>] ==="     -> one recon_longrun segment begins
    "=== recon_longrun exited rc=N iter N ==="         -> segment process ended
    "=== carry-forward: <src> -> banked_LIVE ... ==="  -> resume point advanced
    "[live-bank] STAGE -> banked_LIVE  badges=N levels=[...]"  -> throttled durable snapshot
  Per-decision (from recon_longrun.chooser via L()):
    "[   T.Ts] #  N   (map)@(coords)  party=[('Spec', lvl), ...] prep=<n> ql=<repr>"
    "[   T.Ts]       opts=[...] -> PICK <action>  [battle: ...]/[h2g: ...]"
  Per-segment outcome block (from recon_longrun end):
    "==== END: <why> | <secs>s wall | N decisions | M real battles ===="
    "[T] PARTY  start: [(...)]"   /  "[T] PARTY  end  : [(...)]"
    "[T] LEVELS start [...] -> end [...] (floor a->b, sum c->d)"
    "[T] badges N | dex N | ticket_flag X | maps visited N: [...]"
    "[T] battles by FIELDED lead: {...} | faints ...: {...}"
    "==== OUTCOME: <TAG> — <why> ===="
  Game events (from campaign):
    "note_caught FIRE -> species=X nickname=Y where=Z"          -> a catch
    "kind=name ctx={'place': 'your X just evolved into a Y!"    -> an evolution
    "WHITEOUT-BACKSTOP: recorded the loss ..."                  -> a battle whiteout
    "!! BLACKOUT/STRANDED: ..."                                 -> a nav blackout/strand
    "HALL OF FAME — CREDITS INBOUND"                            -> credits reached
    "CREDITS SEQUENCE DRAINED | battles N | money $M"           -> credits stat line

Robustness: every line is parsed inside a per-line try/except; an unexpected line can never crash the
run. Per-segment relative timestamps ("[  T.Ts]") reset each iteration, so a cumulative wall-clock is
maintained by summing each segment's reported "Ns wall".
"""
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# regexes (compiled once)
# ---------------------------------------------------------------------------
RE_TIME       = re.compile(r"^\[\s*([0-9.]+)s\]")
RE_WD_START   = re.compile(r"===\s*fresh_go watchdog start\s*(.*?)\s*===")
RE_LAUNCH     = re.compile(r"===\s*launch iter\s+(\d+)\s+boot=(\S+)\s*(.*?)\s*===")
RE_EXIT       = re.compile(r"===\s*recon_longrun exited rc=(-?\d+)\s+iter\s+(\d+)\s*===")
RE_CARRY      = re.compile(r"===\s*carry-forward:\s*(.*?)\s*===")
RE_LIVEBANK   = re.compile(r"\[live-bank\]\s*STAGE -> banked_LIVE\s+badges=(\d+)\s+levels=\[([^\]]*)\]")
RE_DECISION   = re.compile(r"#\s*(\d+)\s+(\([^)]*\))@(\([^)]*\))\s+party=\[(.*?)\]\s+prep=(\S+)\s+ql=(.*)$")
RE_PARTY_MON  = re.compile(r"\('([^']+)',\s*(\d+)\)")
RE_PICK       = re.compile(r"opts=\[.*?\]\s*->\s*PICK\s+(\S+)")
RE_END        = re.compile(r"====\s*END:\s*(.*?)\s*\|\s*([0-9.]+)s wall\s*\|\s*(\d+)\s*decisions\s*\|\s*(\d+)\s*real battles\s*====")
RE_PARTY_ST   = re.compile(r"PARTY\s+start:\s*\[(.*)\]\s*$")
RE_PARTY_EN   = re.compile(r"PARTY\s+end\s*:\s*\[(.*)\]\s*$")
RE_PARTY_FULL = re.compile(r"\('([^']+)',\s*(\d+),")   # ('Venusaur', 100, 294, 302)
RE_LEVELS     = re.compile(r"LEVELS start \[([^\]]*)\] -> end \[([^\]]*)\].*sum\s*(\d+)->(\d+)")
RE_BADGEDEX   = re.compile(r"badges\s+(\d+)\s*\|\s*dex\s+(\d+)\s*\|\s*ticket_flag\s+(\w+)\s*\|\s*maps visited\s+(\d+)")
RE_FIELDED    = re.compile(r"battles by FIELDED lead:\s*(\{[^}]*\})\s*\|\s*faints[^:]*:\s*(\{[^}]*\})")
RE_OUTCOME    = re.compile(r"====\s*OUTCOME:\s*(\S+)\s*[—-]+\s*(.*?)\s*====")
RE_CAUGHT     = re.compile(r"note_caught FIRE -> species=(\S+)\s+nickname=(\S+)\s+where=(.*)$")
RE_EVOLVE     = re.compile(r"your (\w+) just evolved into a (\w+)!")
RE_CREDITS_DR = re.compile(r"CREDITS SEQUENCE DRAINED\s*\|\s*battles\s+(\d+)\s*\|\s*money\s*\$(\d+)")
RE_DICT_KV    = re.compile(r"'([^']+)':\s*(\d+)")

# PICK action -> time-share bucket
BUCKET = {
    "battle": "battle/grind",
    "wander_catch": "battle/grind",
    "fetch_keeper": "team-build",
    "box_chaff": "team-build",
    "stock_up": "menus/economy",
    "heal": "menus/economy",
    "head_to_gym": "travel",
    "head_to_league": "travel",
    "enter_league": "travel",
    "leave_building": "travel",
    "talk_npc": "npc/dialogue",
}


def bucket_of(action):
    if action is None:
        return "other"
    if action.startswith("travel:"):
        return "travel"
    return BUCKET.get(action, "other")


def parse_party(blob):
    """Return list of (species, level) from a decision-line party blob like ('Bulb', 6), ('Mank', 14)."""
    return [(m.group(1), int(m.group(2))) for m in RE_PARTY_MON.finditer(blob)]


def parse_full_party(blob):
    """Return list of (species, level) from a PARTY start/end blob ('Venusaur', 100, 294, 302), ..."""
    return [(m.group(1), int(m.group(2))) for m in RE_PARTY_FULL.finditer(blob)]


def parse_dict(blob):
    return {m.group(1): int(m.group(2)) for m in RE_DICT_KV.finditer(blob)}


class Stats:
    def __init__(self):
        self.watchdog_starts = 0
        self.first_abs_time = None          # first date string on a launch/start marker
        self.last_abs_time = None
        self.segments = []                  # list of dicts, one per launched iter
        self.cur = None                     # current segment dict
        self.cum_wall_base = 0.0            # sum of completed segments' wall seconds
        self.cur_rel = 0.0                  # latest relative timestamp within current segment
        self.global_decision = 0            # running global decision counter

        self.picks = Counter()             # action -> count
        self.buckets = Counter()           # bucket -> count

        self.catches = []                   # (species, where, global_decision, cum_wall)
        self.evolutions = []                # (from, to, global_decision, cum_wall)
        self.whiteouts = 0
        self.blackouts = 0
        self.credits_reached = 0            # HALL OF FAME occurrences
        self.credits_stats = None           # (battles, money) from the drained line

        # per-badge first-appearance: badge_count -> (global_decision, cum_wall)
        self.badge_first = {}
        self._last_badges = 0

        # level-curve samples: list of (cum_wall, global_decision, {species: level}, ace_level, bench_min)
        self.curve = []

        # live-bank snapshots: (cum_wall, badges, [levels])
        self.livebanks = []

        # grind windows: track consecutive battle/grind picks (global stream)
        self.cur_grind = 0
        self.longest_grind = 0
        self.longest_grind_at = None        # (global_decision_start, cum_wall)

        # outcome tags
        self.outcomes = Counter()

        # final observed state
        self.final_party_end = None
        self.final_badges = None
        self.final_dex = None
        self.total_real_battles = 0
        self.total_decisions_reported = 0

        self.unparsed_flavors = Counter()   # crude accounting of lines we touched but couldn't classify

    # -- per-segment bookkeeping ---------------------------------------------
    def new_segment(self, iter_n, boot, date):
        # close nothing here (END closes wall accounting); just start a fresh segment record
        self.cur = {
            "iter": iter_n, "boot": boot, "date": date,
            "wall": None, "decisions": None, "battles": None,
            "why": None, "tag": None,
            "party_start": None, "party_end": None,
            "badges": None, "dex": None, "fielded": None, "faints": None,
            "start_cum_wall": self.cum_wall_base,
        }
        self.segments.append(self.cur)
        self.cur_rel = 0.0

    def cum_wall(self):
        return self.cum_wall_base + self.cur_rel

    # -- line dispatch --------------------------------------------------------
    def feed(self, line):
        # capture relative timestamp if present
        mt = RE_TIME.match(line)
        if mt:
            try:
                self.cur_rel = float(mt.group(1))
            except Exception:
                pass

        # watchdog start
        if "watchdog start" in line:
            m = RE_WD_START.search(line)
            if m:
                self.watchdog_starts += 1
                self._note_abs(m.group(1))
                return

        # launch iter
        if "launch iter" in line:
            m = RE_LAUNCH.search(line)
            if m:
                self._note_abs(m.group(3))
                self.new_segment(int(m.group(1)), m.group(2), m.group(3).strip() or None)
                return

        # exit rc
        if "recon_longrun exited" in line:
            m = RE_EXIT.search(line)
            if m and self.cur is not None:
                self.cur["rc"] = int(m.group(1))
            return

        # carry-forward
        if "carry-forward:" in line:
            return  # tracked implicitly; not needed for stats

        # live-bank
        if "[live-bank]" in line:
            m = RE_LIVEBANK.search(line)
            if m:
                try:
                    lv = [int(x) for x in m.group(2).split(",") if x.strip()]
                except Exception:
                    lv = []
                self.livebanks.append((self.cum_wall(), int(m.group(1)), lv))
            return

        # decision line
        if "# " in line and "party=" in line and "prep=" in line:
            m = RE_DECISION.search(line)
            if m:
                self.global_decision += 1
                party = parse_party(m.group(4))
                cw = self.cum_wall()
                if party:
                    curve = {sp: lv for sp, lv in party}
                    ace = max(lv for _, lv in party)
                    bench = min(lv for _, lv in party)
                    self.curve.append((cw, self.global_decision, curve, ace, bench))
                return

        # PICK line
        if "-> PICK" in line:
            m = RE_PICK.search(line)
            if m:
                action = m.group(1).rstrip(":")  # 'travel:' captured as 'travel' by \S+? keep base
                raw = m.group(1)
                self.picks[raw] += 1
                b = bucket_of(raw if not raw.startswith("travel") else "travel:x")
                self.buckets[b] += 1
                # grind-window tracking
                if b == "battle/grind":
                    if self.cur_grind == 0:
                        self._grind_start = (self.global_decision, self.cum_wall())
                    self.cur_grind += 1
                    if self.cur_grind > self.longest_grind:
                        self.longest_grind = self.cur_grind
                        self.longest_grind_at = self._grind_start
                else:
                    self.cur_grind = 0
                return

        # catch
        if "note_caught FIRE" in line:
            m = RE_CAUGHT.search(line)
            if m:
                self.catches.append((m.group(1), m.group(3).strip(),
                                     self.global_decision, self.cum_wall()))
            return

        # evolution
        if "evolved into" in line:
            m = RE_EVOLVE.search(line)
            if m:
                self.evolutions.append((m.group(1), m.group(2),
                                        self.global_decision, self.cum_wall()))
            return

        # whiteout / blackout
        if "WHITEOUT-BACKSTOP" in line:
            self.whiteouts += 1
            return
        if "BLACKOUT/STRANDED" in line:
            self.blackouts += 1
            return

        # credits
        if "HALL OF FAME" in line:
            self.credits_reached += 1
            return
        if "CREDITS SEQUENCE DRAINED" in line:
            m = RE_CREDITS_DR.search(line)
            if m:
                self.credits_stats = (int(m.group(1)), int(m.group(2)))
            return

        # END block
        if "==== END:" in line:
            m = RE_END.search(line)
            if m and self.cur is not None:
                self.cur["why"] = m.group(1)
                self.cur["wall"] = float(m.group(2))
                self.cur["decisions"] = int(m.group(3))
                self.cur["battles"] = int(m.group(4))
                self.total_real_battles += int(m.group(4))
                self.total_decisions_reported += int(m.group(3))
                # advance cumulative wall by this segment's wall seconds
                self.cum_wall_base += float(m.group(2))
            return

        if "PARTY  start:" in line:
            m = RE_PARTY_ST.search(line)
            if m and self.cur is not None:
                self.cur["party_start"] = parse_full_party(m.group(1))
            return
        if "PARTY  end" in line:
            m = RE_PARTY_EN.search(line)
            if m and self.cur is not None:
                self.cur["party_end"] = parse_full_party(m.group(1))
                self.final_party_end = self.cur["party_end"]
            return

        if "badges" in line and "dex" in line and "ticket_flag" in line:
            m = RE_BADGEDEX.search(line)
            if m and self.cur is not None:
                bc = int(m.group(1))
                self.cur["badges"] = bc
                self.cur["dex"] = int(m.group(2))
                self.final_badges = bc
                self.final_dex = int(m.group(2))
                # per-badge first appearance (via segment summaries)
                if bc > self._last_badges:
                    for k in range(self._last_badges + 1, bc + 1):
                        if k not in self.badge_first:
                            self.badge_first[k] = (self.global_decision, self.cum_wall())
                    self._last_badges = bc
            return

        if "battles by FIELDED lead:" in line:
            m = RE_FIELDED.search(line)
            if m and self.cur is not None:
                self.cur["fielded"] = parse_dict(m.group(1))
                self.cur["faints"] = parse_dict(m.group(2))
            return

        if "==== OUTCOME:" in line:
            m = RE_OUTCOME.search(line)
            if m:
                tag = m.group(1)
                self.outcomes[tag] += 1
                if self.cur is not None:
                    self.cur["tag"] = tag
            return

    def _note_abs(self, s):
        s = (s or "").strip()
        if "2026" in s or re.search(r"\d{4}", s):
            if self.first_abs_time is None:
                self.first_abs_time = s
            self.last_abs_time = s

    # -- also fold live-bank badge counts into per-badge first appearance -----
    def finalize(self):
        for cw, bc, lv in self.livebanks:
            if bc not in self.badge_first:
                # approximate: use the live-bank cum_wall; decision unknown here
                self.badge_first[bc] = (None, cw)


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------
def fmt_hms(seconds):
    try:
        s = int(seconds)
    except Exception:
        return "n/a"
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def downsample(rows, target=45):
    if len(rows) <= target:
        return rows
    step = len(rows) / target
    return [rows[int(i * step)] for i in range(target)]


def render(st, log_path):
    L = []
    w = L.append
    base = os.path.basename(log_path)
    w(f"# RUN STATS — `{base}`")
    w("")
    w(f"_Generated by tools/run_stats.py — streaming parse of the recon_longrun watchdog log._")
    w("")

    # ---- wall-clock / segments ----
    completed = [s for s in st.segments if s.get("wall") is not None]
    total_wall = sum(s["wall"] for s in completed)
    w("## Wall-clock & segments")
    w("")
    w(f"- Watchdog (re)starts in log: **{st.watchdog_starts}**")
    w(f"- Segments launched (recon_longrun iterations): **{len(st.segments)}**")
    w(f"- Segments that reported an END/OUTCOME block: **{len(completed)}**")
    w(f"- First absolute timestamp in log: **{st.first_abs_time or 'not instrumented'}** "
      f"(only the very first watchdog session stamped its iter markers with a date)")
    w(f"- Last absolute timestamp in log: **{st.last_abs_time or 'not instrumented'}**")
    w(f"- Absolute end-of-run timestamp: **not instrumented** "
      f"(later iter markers carry no date; use the file mtime for a true wall end)")
    w(f"- **Total summed segment wall-clock: {total_wall:,.0f}s ≈ {fmt_hms(total_wall)}** "
      f"(sum of each segment's `Ns wall`; at ~14x that models ~{total_wall*14/3600:.0f} human-hours)")
    w(f"- Sim seconds total: **not separately instrumented** — the harness logs only wall seconds "
      f"per segment (`Ns wall`); sim-time = wall × ~14 (headless emulator speed, not logged per-tick).")
    w(f"- Total decisions (summed across segments): **{st.total_decisions_reported:,}** "
      f"(global decision events seen: {st.global_decision:,})")
    w(f"- Total real battles (summed across segments): **{st.total_real_battles:,}**")
    w("")
    w("### Outcome tag distribution (per segment)")
    w("")
    for tag, n in st.outcomes.most_common():
        w(f"- `{tag}`: {n}")
    w("")

    # ---- credits ----
    w("## Credits / endgame")
    w("")
    if st.credits_reached:
        w(f"- **CREDITS REACHED** — `HALL OF FAME — CREDITS INBOUND` fired **{st.credits_reached}×**.")
        if st.credits_stats:
            b_, mo = st.credits_stats
            w(f"- Credits-drain stat line: **{b_} battles**, **money ${mo:,}**.")
    else:
        w("- Credits: **not reached** in this log (no Hall of Fame line).")
    w(f"- Final badges observed: **{st.final_badges if st.final_badges is not None else 'not instrumented'}**")
    w(f"- Final dex observed: **{st.final_dex if st.final_dex is not None else 'not instrumented'}**")
    w("")

    # ---- per-badge splits ----
    w("## Per-badge splits (first appearance)")
    w("")
    w("_When each badge count first appeared, by cumulative wall-clock (and global decision index "
      "when derivable from a segment summary; live-bank-derived rows have no decision index)._")
    w("")
    if st.badge_first:
        w("| badges | first seen @ cum wall | @ global decision |")
        w("|---|---|---|")
        for bc in sorted(st.badge_first):
            gd, cw = st.badge_first[bc]
            w(f"| {bc} | {cw:,.0f}s ({fmt_hms(cw)}) | {gd if gd is not None else 'n/a (live-bank)'} |")
    else:
        w("- not instrumented")
    w("")

    # ---- catches ----
    w("## Catches")
    w("")
    w("_Source: `note_caught FIRE` lines. Kept-vs-boxed is **not instrumented** as a catch-time field; "
      "infer retention from the final party below._")
    w("")
    if st.catches:
        w("| species | where | @ global decision | @ cum wall |")
        w("|---|---|---|---|")
        for sp, where, gd, cw in st.catches:
            w(f"| {sp} | {where} | {gd} | {cw:,.0f}s |")
    else:
        w("- No `note_caught FIRE` catches logged.")
    w("")

    # ---- evolutions ----
    w("## Evolutions")
    w("")
    if st.evolutions:
        w("| from | to | @ global decision | @ cum wall |")
        w("|---|---|---|---|")
        for a, b, gd, cw in st.evolutions:
            w(f"| {a} | {b} | {gd} | {cw:,.0f}s |")
    else:
        w("- No evolution lines logged.")
    w("")

    # ---- whiteouts / blackouts ----
    w("## Whiteouts / blackouts / E4")
    w("")
    w(f"- Battle whiteouts (`WHITEOUT-BACKSTOP` loss records): **{st.whiteouts}**")
    w(f"- Navigation blackout/strand events (`BLACKOUT/STRANDED`): **{st.blackouts}**")
    w(f"- E4 / Hall-of-Fame arrivals: **{st.credits_reached}**")
    w("- E4 *attempts* (distinct): **not instrumented** — no explicit 'E4 attempt N' marker; "
      "count of `enter_league`/`head_to_league` picks is a proxy "
      f"({st.picks.get('enter_league',0)} enter_league, {st.picks.get('head_to_league',0)} head_to_league).")
    w("")

    # ---- time-share ----
    w("## Time-share (by PICK action count)")
    w("")
    total_picks = sum(st.picks.values()) or 1
    w("_Approximate activity mix from oracle PICK counts (decision-share, not wall-share — "
      "per-action wall duration is not instrumented)._")
    w("")
    w("| bucket | picks | share |")
    w("|---|---|---|")
    for b, n in st.buckets.most_common():
        w(f"| {b} | {n:,} | {100*n/total_picks:.1f}% |")
    w("")
    w("### Raw PICK breakdown")
    w("")
    w("| action | count |")
    w("|---|---|")
    for a, n in st.picks.most_common():
        w(f"| `{a}` | {n:,} |")
    w("")

    # ---- longest grind window ----
    w("## Longest continuous grind window")
    w("")
    if st.longest_grind_at:
        gd0, cw0 = st.longest_grind_at
        w(f"- Longest run of consecutive `battle`/`wander_catch` picks with no intervening "
          f"travel/gym/heal pick: **{st.longest_grind} picks**, starting ~global decision {gd0} "
          f"(~{cw0:,.0f}s cum wall).")
        w("- Note: this counts consecutive in-grass/battle decisions; a badge advance also resets it "
          "implicitly since a badge push routes through travel/gym picks.")
    else:
        w("- No grind windows detected.")
    w("")

    # ---- per-slot level curve ----
    w("## Per-slot / per-species level curve (THE solo-ace-vs-flat-bench diagnostic)")
    w("")
    w("_Sampled from per-decision `party=[...]` lines across the whole run. Party slots reorder "
      "during grind, so this tracks levels **by species** (the max-level mon = the 'ace'). "
      "`ace` = highest level in party that decision; `bench_min` = lowest._")
    w("")
    if st.curve:
        # collect species universe
        species = []
        for _, _, curve, _, _ in st.curve:
            for sp in curve:
                if sp not in species:
                    species.append(sp)
        rows = downsample(st.curve, 45)
        header = "| cum wall | dec# | " + " | ".join(species) + " | ACE | bench_min |"
        w(header)
        w("|" + "---|" * (len(species) + 4))
        for cw, gd, curve, ace, bench in rows:
            cells = [f"{curve.get(sp,'')}" for sp in species]
            w(f"| {cw:,.0f}s | {gd} | " + " | ".join(cells) + f" | **{ace}** | {bench} |")
        w("")
        # ace vs bench spread summary
        aces = [ace for _, _, _, ace, _ in st.curve]
        benches = [b for _, _, _, _, b in st.curve]
        w(f"- Ace level range across run: **{min(aces)} → {max(aces)}**")
        w(f"- Bench-min level range across run: **{min(benches)} → {max(benches)}**")
        w(f"- Final observed ace/bench spread: **{aces[-1] - benches[-1]} levels** "
          f"(ace {aces[-1]} vs bench-min {benches[-1]}).")
    else:
        w("- not instrumented")
    w("")

    # ---- live-bank level snapshots ----
    w("## Live-bank snapshots (durable resume points)")
    w("")
    if st.livebanks:
        w("| cum wall | badges | party levels |")
        w("|---|---|---|")
        for cw, bc, lv in st.livebanks:
            w(f"| {cw:,.0f}s | {bc} | {lv} |")
    else:
        w("- none")
    w("")

    # ---- final state ----
    w("## Final party & state")
    w("")
    if st.final_party_end:
        w("Final observed `PARTY end` (last segment with a party-end block):")
        w("")
        w("| slot | species | level |")
        w("|---|---|---|")
        # count duplicate species
        spcount = Counter(sp for sp, _ in st.final_party_end)
        for i, (sp, lv) in enumerate(st.final_party_end, 1):
            w(f"| {i} | {sp} | {lv} |")
        w("")
        dupes = {sp: c for sp, c in spcount.items() if c > 1}
        if dupes:
            w(f"- **Duplicate species in final party:** {dupes}")
        else:
            w("- No duplicate species in final party.")
    else:
        w("- not instrumented")
    w(f"- Final badges: **{st.final_badges if st.final_badges is not None else 'not instrumented'}**")
    w(f"- Final dex: **{st.final_dex if st.final_dex is not None else 'not instrumented'}**")
    w("- Money: **not instrumented** in per-segment summaries — only appears once, in the "
      f"`CREDITS SEQUENCE DRAINED` line ({'$'+format(st.credits_stats[1],',') if st.credits_stats else 'n/a'}).")
    w("")

    # ---- missing instrumentation ----
    w("## MISSING INSTRUMENTATION (add to recon_longrun.py for future runs)")
    w("")
    for item in [
        "**Absolute wall-clock timestamps** on every `L()` line (or at least every END block). Today "
        "only the first watchdog session stamps a date on its iter markers; total real duration must be "
        "inferred by summing per-segment `Ns wall` or reading the file mtime. Add an ISO timestamp prefix.",
        "**Sim-seconds counter** — log the emulated game-time / frame count per segment, not just wall "
        "seconds, so the 14x factor isn't an assumption.",
        "**Money / cash** in the per-segment `badges | dex | ...` summary line. Currently money only "
        "appears in the one-off `CREDITS SEQUENCE DRAINED` line, so mid-run economy is invisible.",
        "**Explicit E4-attempt markers** (`E4 ATTEMPT N — <who> <result>`). Attempts are only "
        "inferable from `enter_league` picks + Hall-of-Fame lines; a per-attempt line with the leader "
        "beaten and party HP would make E4 diagnosis first-class.",
        "**Catch kept-vs-boxed flag** at catch time (`note_caught ... kept=True/boxed`). Retention is "
        "currently only inferable from the final party.",
        "**Per-action wall duration** — each PICK bucket's real time cost. Time-share here is "
        "decision-count share, not wall share; a head_to_gym march and a single battle count equally.",
        "**Stable per-slot identity** in decision `party=[...]` lines (slots reorder for grind, so "
        "level curves must be reconstructed by species). Log a stable party ID or PID per slot.",
        "**Badge-earned event line** (`BADGE N earned @ <gym> t=<wall>`). Per-badge splits here are "
        "derived from `badges=N` first-appearances in segment/live-bank summaries, which lag the actual "
        "badge moment and lack a decision index when only a live-bank captured them.",
        "**Level-up / XP events** per mon — would make the solo-ace-vs-flat-bench curve exact rather "
        "than sampled from decision lines.",
    ]:
        w(f"- {item}")
    w("")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: run_stats.py <log_path> [out_path]", file=sys.stderr)
        return 2
    log_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(log_path))[0]
        out_path = f"RUN_STATS_{base}.md"

    st = Stats()
    n = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            try:
                st.feed(line.rstrip("\n"))
            except Exception:
                # never crash on a surprising line
                pass
    st.finalize()

    report = render(st, log_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"parsed {n:,} lines -> {out_path}")
    print(f"segments={len(st.segments)} completed={sum(1 for s in st.segments if s.get('wall'))} "
          f"credits={st.credits_reached} catches={len(st.catches)} evos={len(st.evolutions)} "
          f"total_wall={sum(s['wall'] for s in st.segments if s.get('wall')):.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
