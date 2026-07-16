"""ramdiff.py - a RAM-DIFF cursor finder (the working replacement for the broken mgba
Memory.search). Snapshot an EWRAM region (plain reads, which work), drive an input, snapshot
again, and isolate the byte(s) whose value tracks a known sequence. Used to locate menu-cursor
addresses by elimination. No dependency on the broken built-in search.

Battle structs live ~0x02022000-0x02024C00; default snapshot covers that.
"""

LO, HI = 0x02022000, 0x02024C00


def snapshot(b, lo=LO, hi=HI):
    """Read [lo,hi) into a bytes object (one read per byte; ~10KB default = fast enough)."""
    return bytes(b.rd8(a) for a in range(lo, hi))


def find_tracking(snaps, want, lo=LO):
    """snaps: list of byte-snapshots taken at known cursor positions. want: the list of expected
    cursor VALUES at each snapshot (e.g. [0,1,3,2,0]). Returns the addresses whose byte equals
    `want[i]` in EVERY snapshot i - i.e. the byte(s) that track the cursor exactly."""
    n = len(snaps[0])
    out = []
    for i in range(n):
        if all(snaps[k][i] == want[k] for k in range(len(snaps))):
            out.append(lo + i)
    return out


def changed(a, b_, lo=LO):
    """Addresses whose byte differs between two snapshots a and b_."""
    return [lo + i for i in range(len(a)) if a[i] != b_[i]]
