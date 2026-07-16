"""NS#8 decision-check (emulator-free) for the Saffron-gatehouse door-passthrough skip.

The Route-6 Celadon leg is via:'pass'. _door_passthrough's candidate sort tie-breaks on
-len(dest_doors), so the 2-door Saffron SOUTH gatehouse ((12,5),(13,5)->(18,0)) out-ranks the
1-door Underground Path hut ((19,13)->(1,32)) and she wedges in the guard-blocked gatehouse
pre-Tea (ns6_bail025: 8x wedge at (18,0)). The fix skips the pre-Tea Saffron gatehouse maps as
connector candidates so the UGP hut wins. This replicates the exact cands-filter + _dest_rank sort
from campaign._door_passthrough against the REAL learned Route-6 warp geometry, with the fix in a
'got_tea' switch, and asserts: pre-Tea the gatehouse is dropped + the UGP hut is chosen; post-Tea
the gatehouse is a candidate again (zero behaviour change once the guards open)."""

SAFFRON = (3, 10)
SAFFRON_GATE_MAPS = {(18, 0), (19, 0)}

# REAL learned Route-6 (3,24) warps (ns6_bail025 world model): the south gatehouse (2 doors) + the
# Underground Path hut (1 door, the intended crossing under Saffron to Route 5).
ROUTE6_WARPS = [
    ((12, 5), (18, 0)),   # south gatehouse door A
    ((13, 5), (18, 0)),   # south gatehouse door B
    ((19, 13), (1, 32)),  # Underground Path hut -> UGP interior toward Route 5
]
WANT_MAP = (3, 23)        # Route 5 (the billed next leg)
VISITED = {(1, 32), (18, 0)}   # both interiors learned by the time she's re-attempting (worst case for the fix)


def saffron_gate_dead_ends(got_tea):
    """Mirror of campaign._saffron_gate_dead_ends: empty post-Tea, the gate set pre-Tea."""
    return set() if got_tea else set(SAFFRON_GATE_MAPS)


def choose_door(warps, want_map, visited, got_tea):
    """Replica of _door_passthrough's cands filter + sort. Returns the chosen door tile."""
    dest_doors = {}
    for (wxy, wdest) in warps:
        dest_doors.setdefault(tuple(wdest), set()).add(tuple(wxy))
    door_dest = {t: dest for dest, ts in dest_doors.items() for t in ts}
    saf_gate = saffron_gate_dead_ends(got_tea)
    cands = []
    for wt in door_dest:
        if tuple(door_dest.get(wt) or ()) in saf_gate:   # THE FIX
            continue
        cands.append(wt)

    def _dest_rank(t):
        dm = tuple(door_dest.get(tuple(t)) or ())
        if want_map and dm == tuple(want_map):
            return 0
        return 2 if dm in visited else 1

    pos0 = (5, 8)   # standing near the doors (distance only tie-breaks; irrelevant to the outcome)
    cands.sort(key=lambda t: (_dest_rank(t),
                              -len(dest_doors.get(door_dest.get(tuple(t)), ())),
                              abs(t[0] - pos0[0]) + abs(t[1] - pos0[1])))
    return cands[0] if cands else None, cands, door_dest


def main():
    fails = []

    # 1. Helper: pre-Tea returns the gate set; post-Tea empty.
    if saffron_gate_dead_ends(False) != SAFFRON_GATE_MAPS:
        fails.append("pre-Tea helper should return the Saffron gate maps")
    if saffron_gate_dead_ends(True) != set():
        fails.append("post-Tea helper should be empty")

    # 2. PRE-TEA (no fix would pick the 2-door gatehouse): with the fix, the gatehouse is dropped and
    #    the single-door UGP hut (19,13) is chosen.
    chosen, cands, dd = choose_door(ROUTE6_WARPS, WANT_MAP, VISITED, got_tea=False)
    if any(dd[c] == (18, 0) for c in cands):
        fails.append(f"pre-Tea: gatehouse door should be filtered out of cands, got {cands}")
    if chosen != (19, 13):
        fails.append(f"pre-Tea: should choose the UGP hut (19,13), chose {chosen}")

    # 2b. CONTROL — without the fix (treat as post-Tea filter), the 2-door gatehouse wins (proves the
    #     tie-break bug the fix targets is real).
    chosen_nofix, _, _ = choose_door(ROUTE6_WARPS, WANT_MAP, VISITED, got_tea=True)
    if chosen_nofix not in ((12, 5), (13, 5)):
        fails.append(f"control: without the skip the 2-door gatehouse should win, chose {chosen_nofix}")

    # 3. POST-TEA: gatehouse is a candidate again (no behaviour change once guards open).
    _, cands_pt, dd_pt = choose_door(ROUTE6_WARPS, WANT_MAP, VISITED, got_tea=True)
    if not any(dd_pt[c] == (18, 0) for c in cands_pt):
        fails.append("post-Tea: gatehouse door should be a candidate again")

    # 4. UGP-only map (no gatehouse) is unaffected pre-Tea.
    chosen_ugp, _, _ = choose_door([((19, 13), (1, 32))], WANT_MAP, VISITED, got_tea=False)
    if chosen_ugp != (19, 13):
        fails.append(f"pre-Tea UGP-only: should still choose (19,13), chose {chosen_ugp}")

    n = 6
    if fails:
        print(f"FAIL {n - len(fails)}/{n}")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"PASS {n}/{n} — Saffron-gatehouse door-skip: pre-Tea drops the guard-blocked gatehouse "
          "so the UGP hut wins; control proves the tie-break bug; post-Tea is unchanged.")


if __name__ == "__main__":
    main()
