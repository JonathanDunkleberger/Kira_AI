"""boulder_puzzle.py — the REUSABLE Strength-boulder room solver (2026-08-05, the Mt. Ember loop).

THE LOOP THIS KILLS (live, Jonny watching): 'she does one or two strength moves, moves one or
two boulders, and then just leaves the thing... then she goes back... tens of minutes of loops.'
Two compounding bugs in the old per-strike push plans (MoltresHunt.board_mission and
VictoryRoad.run_puzzle shared the pattern):

  1. NON-IDEMPOTENT RE-RUNS: re-running a plan on a PARTIALLY/FULLY solved board re-matched the
     already-moved boulder through nearest_boulder's radius-8 fuzz and pushed it FURTHER —
     over-shooting the target, wedging the board, and turning every retry into a fresh wedge.
  2. RETREAT-FIRST RECOVERY: the first wedge answered with a door round-trip (map exit), and a
     failed strike attempt handed the roam layer a reason to walk her clear off the mountain —
     but FRLG boulders are FLAG_TEMP objects that RESET to template tiles on EVERY map re-entry,
     so any exit re-zeroes the puzzle. Exit is the one move guaranteed to make things worse.

THE DOCTRINE (each room is fixed content — a deterministic scripted solve beats generic pathing):
  - GROUND TRUTH, NEVER MEMORY: boulder positions are read LIVE from the map's object events
    each decision. THE CAMERA LAW: the GBA unloads object events off-camera, so 'not in the
    scan' only means 'not visible' — the solver WALKS NEAR the chain's path and re-scans before
    believing anything; an approach failure means UNVERIFIED and the chain fails LOUD (the old
    'absent -> assume pushed' lie is dead). No push state is ever stored across calls, so a map
    transition can never leave a stale assumption: re-entry re-derives from the fresh template.
  - IDEMPOTENT CHAINS: each room is a set of CHAINS (one physical boulder each) with the full
    tile-by-tile path from pret template start to intended target. The solver locates the
    boulder ON its path and pushes only the REMAINING steps, one push at a time, with live
    readback after every push. Already at target -> zero presses. Off its path -> diverged,
    never guessed at.
  - FAIL-IN-PLACE: a failed push/approach retries from live boulder truth ON THE MAP. The door
    round-trip reset (which re-zeroes the board by design) is the LAST resort, and after it the
    solve restarts from the fresh template — correct by construction, no memory to clear.
  - MILESTONE DURABILITY: an optional checkpoint callback fires after verified pushes
    (`ckpt_every`) and at chain completion — emulator savestates capture full RAM, so pushed
    boulders SURVIVE a checkpoint reload (unlike in-game re-entry); a recovery mid-puzzle
    resumes seconds back with the board intact.

The RIG (duck-type) supplies the proven per-strike actuators; the module owns ORCHESTRATION only:
    rig.live_boulders() -> [(x,y), ...]          # live object-event scan (GFX_BOULDER)
    rig.sea_walk(goal_pred, label, ...) -> bool  # on-map BFS walk (excludes warp tiles)
    rig.push(approx, key, n[, allow]) -> bool    # one verified shove (readback: boulder moved)
    rig.ensure_strength(approx) -> bool          # arm HM04 for this map load
    rig.enter_step(tile, dest, label) -> bool    # door round-trip legs (reset path only)
    rig.handle_interrupts() -> bool              # battles/boxes between presses
    rig.log(str)

Ground truth shipped here: Mt. Ember exterior ascent/descent + summit (pret map.json templates,
.tmp_sym 2026-08-04). Victory Road 1F/2F/3F wire in via room_from_ops() over victory_road.py's
hand-solved, live-proven VRnF_PUZZLE op tables (elevation-aware, incl. the never-push (35,13)
pocket law) — same engine, same idempotence, ready for the E4 run.
"""

DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def chain_path(start, segs):
    """Every tile the boulder occupies executing `segs` from `start`, in order (incl. both ends)."""
    path = [tuple(start)]
    for seg in segs:
        d = DELTA[seg[0]]
        for _ in range(int(seg[1])):
            path.append((path[-1][0] + d[0], path[-1][1] + d[1]))
    return path


def room(map_id, name, chains, strength_at=None, reset=None, ckpt_every=3):
    """Build a room spec. chains: [{'start': (x,y), 'segs': [(key, n[, allow_tiles])...],
    'vanish_ok': bool}]. reset: (((out_tile), out_map), ((back_tile), home_map)) door
    round-trip that re-zeroes the board (LAST resort), or None (never exit — VR floors)."""
    built = []
    for ch in chains:
        segs = [(s[0], int(s[1]), tuple(s[2]) if len(s) > 2 else ()) for s in ch["segs"]]
        path = chain_path(ch["start"], segs)
        # per-step key/allow arrays (step i moves the boulder path[i] -> path[i+1])
        keys, allows = [], []
        for key, n, allow in segs:
            keys.extend([key] * n)
            allows.extend([allow] * n)
        built.append({"start": tuple(ch["start"]), "path": path, "keys": keys,
                      "allows": allows, "vanish_ok": bool(ch.get("vanish_ok"))})
    return {"map": tuple(map_id), "name": name, "chains": built,
            "strength_at": tuple(strength_at or built[0]["start"]),
            "reset": reset, "ckpt_every": int(ckpt_every)}


def room_from_ops(map_id, name, ops, reset=None, ckpt_every=4):
    """Translate a recon_victory-style op table ([('strength', tile), ('push', approx, key, n
    [, allow])...]) into ONE chained room, VERIFYING the ops chain exactly (each push's approx
    must equal the previous push's landing tile) — a mis-chained table fails at build time,
    never mid-climb."""
    strength_at, segs, start, cur = None, [], None, None
    for op in ops:
        if op[0] == "strength":
            strength_at = tuple(op[1])
            continue
        approx, key, n = tuple(op[1]), op[2], int(op[3])
        allow = tuple(op[4]) if len(op) > 4 else ()
        if start is None:
            start = cur = approx
        if approx != cur:
            raise ValueError(f"{name}: op {op!r} does not chain (boulder is at {cur})")
        d = DELTA[key]
        cur = (cur[0] + n * d[0], cur[1] + n * d[1])
        segs.append((key, n, allow))
    return room(map_id, name, [{"start": start, "segs": segs}],
                strength_at=strength_at, reset=reset, ckpt_every=ckpt_every)


# ── Mt. Ember ground truth (pret pokefirered map.json templates, .tmp_sym 2026-08-04) ──────────
_EMBER_EXT, _EMBER_1F, _EMBER_SUMMIT, _KINDLE = (1, 97), (1, 98), (1, 101), (3, 45)

# exterior ASCENT (Kindle doors (28/29,48) -> 1F door (14,24)): two boulders LEFT x3 each
EMBER_ASCENT = room(_EMBER_EXT, "ext-ascent",
                    [{"start": (22, 45), "segs": [("LEFT", 3)]},
                     {"start": (17, 46), "segs": [("LEFT", 3)]}],
                    reset=(((28, 48), _KINDLE), ((11, 6), _EMBER_EXT)), ckpt_every=1)
# exterior DESCENT (1F door -> Kindle doors): fresh template board, one RIGHT shove each
EMBER_DESCENT = room(_EMBER_EXT, "ext-descent",
                     [{"start": (17, 46), "segs": [("RIGHT", 1)]},
                      {"start": (22, 45), "segs": [("RIGHT", 1)]}],
                     reset=(((14, 24), _EMBER_1F), ((2, 15), _EMBER_EXT)), ckpt_every=1)
# summit (entrance (9,15) -> beside Moltres (9,6)): 4 boulders, 6 pushes; the last chain
# walks ONE boulder UP then RIGHT x2, parking it at (10,9) clear of the corridor
EMBER_SUMMIT_BOARD = room(_EMBER_SUMMIT, "summit-board",
                          [{"start": (10, 12), "segs": [("UP", 1)]},
                           {"start": (9, 12), "segs": [("LEFT", 1)]},
                           {"start": (8, 11), "segs": [("LEFT", 1)]},
                           {"start": (8, 10), "segs": [("UP", 1), ("RIGHT", 2)]}],
                          reset=(((9, 15), _EMBER_EXT), ((29, 7), _EMBER_SUMMIT)),
                          ckpt_every=1)


# ── the solver ─────────────────────────────────────────────────────────────────────────────────
def _locate(rig, ch, name, L):
    """Index of the chain's boulder on its path, from LIVE truth only. -1 = UNVERIFIED (the
    look itself failed — never assume anything), -2 = verified GONE from the whole path."""
    path = ch["path"]

    def _hits():
        live = {tuple(t) for t in rig.live_boulders()}
        found = [i for i, t in enumerate(path) if t in live]
        if len(found) > 1:
            # a FOREIGN boulder parked on our chain's path (none in the shipped rooms'
            # geometry, but say so if the world disagrees) — resume from the furthest hit,
            # the push readback + final-target verification keep this honest either way
            L(f"!! [{name}] {len(found)} boulders on the chain path "
              f"{[path[i] for i in found]} — resuming from the furthest (LOUD)")
        return max(found) if found else None

    i = _hits()
    if i is not None:
        return i
    verified = False
    for anchor in (path[0], path[-1]):        # CAMERA LAW: walk near BOTH ends before believing
        if rig.sea_walk(lambda c, a=anchor: abs(c[0] - a[0]) + abs(c[1] - a[1]) <= 3,
                        f"{name}-look"):
            verified = True
            i = _hits()
            if i is not None:
                return i
    return -2 if verified else -1


def _solve_chain(rig, room_, ci, ch, ckpt, L):
    name = f"{room_['name']}#{ci}"
    path, keys, allows = ch["path"], ch["keys"], ch["allows"]
    last = len(path) - 1
    for _step in range(last + 4):             # bounded: remaining pushes + a few relocates
        while rig.handle_interrupts():
            pass
        idx = _locate(rig, ch, name, L)
        if idx == last:
            L(f"   [{name}] boulder VERIFIED at target {path[last]}")
            ckpt(f"{room_['name']}-chain{ci}", force=True)
            return True
        if idx == -1:
            L(f"!! [{name}] boulder UNVERIFIABLE (the look-approach failed) — never assume; "
              f"chain fails LOUD")
            return False
        if idx == -2:
            if ch["vanish_ok"]:
                L(f"   [{name}] boulder verified GONE (vanish_ok: hole/drop row) — done")
                ckpt(f"{room_['name']}-chain{ci}", force=True)
                return True
            L(f"!! [{name}] boulder verified GONE from its whole path — the board diverged")
            return False
        pos, key, allow = path[idx], keys[idx], allows[idx]
        if idx > 0:
            L(f"   [{name}] RESUME mid-chain: boulder live at {pos} ({idx}/{last} done) — "
              f"pushing only the remainder (idempotent, never over-push)")
        ok = rig.push(pos, key, 1, allow) if allow else rig.push(pos, key, 1)
        if not ok:
            L(f"!! [{name}] push {pos} {key} FAILED (live truth kept; retry replans in place)")
            return False
        ckpt(f"{room_['name']}-push")
    L(f"!! [{name}] step budget spent without reaching {path[last]} (LOUD)")
    return False


def solve_room(rig, room_, checkpoint=None, log=None, max_rounds=3):
    """Solve one boulder room from live ground truth. Returns True iff EVERY chain is verified
    at its target. Round 1 failures retry IN PLACE (map exit resets the board — it is the one
    move guaranteed to make things worse); the door round-trip reset runs once, before the
    final round, and only if the room defines one."""
    L = log or getattr(rig, "log", print)
    n_since = [0]

    def ckpt(reason, force=False):
        if checkpoint is None:
            return
        n_since[0] += 1
        if force or n_since[0] >= room_["ckpt_every"]:
            n_since[0] = 0
            try:
                checkpoint(reason)
            except Exception as e:
                L(f"   [{room_['name']}] checkpoint '{reason}' skipped: {e}")

    for round_ in range(max_rounds):
        if not rig.ensure_strength(room_["strength_at"]):
            L(f"!! [{room_['name']}] Strength never armed — no pushes this round")
        elif all(_solve_chain(rig, room_, ci, ch, ckpt, L)
                 for ci, ch in enumerate(room_["chains"])):
            L(f"   [{room_['name']}] SOLVED — every chain verified at target")
            return True
        if round_ >= max_rounds - 1:
            break
        if round_ == max_rounds - 2 and room_.get("reset"):
            L(f"   [{room_['name']}] board diverged beyond in-place repair — door round-trip "
              f"RESET (LAST resort by design: re-entry respawns the template boulders)")
            (out_tile, out_map), (back_tile, home_map) = room_["reset"]
            if not (rig.enter_step(tuple(out_tile), tuple(out_map), f"{room_['name']}-reset-out")
                    and rig.enter_step(tuple(back_tile), tuple(home_map),
                                       f"{room_['name']}-reset-back")):
                L(f"!! [{room_['name']}] reset round-trip failed (LOUD)")
                return False
        else:
            L(f"   [{room_['name']}] attempt {round_ + 1} failed — retrying IN PLACE from live "
              f"boulder truth (never exit mid-puzzle)")
    L(f"!! [{room_['name']}] unsolved after {max_rounds} rounds (LOUD)")
    return False
