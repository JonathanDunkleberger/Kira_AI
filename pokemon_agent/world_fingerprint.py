"""world_fingerprint.py - THE KEYSTONE of the free-roam self-unstuck watchdog.

ONE question, asked at two cadences: *did the world actually change?*  A single compact,
EXPLICIT-SEMANTIC snapshot of the things an ACTION is supposed to move - never a raw RAM hash
(that would let an animation/RNG tick masquerade as progress). Two equal snapshots taken across
an action that SHOULD have changed something == no progress == the signal every stuck-escape keys on.

  - MICRO (real-time, INSIDE the brute-force primitives): the dialogue driver and the travel
    blocker-A guard snapshot before/after each repeated press to tell a productive interaction
    (battle started / NPC stepped / new text) from a doomed mash (same box, same tile, forever).
  - MACRO (per free-roam tick - increment 3): the ProgressLedger fingerprints each tick to catch
    higher-order oscillation no single handler can see.

FIELDS (deliberate - INCLUDE only what an action is meant to move; EXCLUDE the noise that makes
mashing look like progress):
  INCLUDE  map_id, x, y, facing, menu_or_dialogue (bool), battle_active (bool),
           party (species+level+HP per mon), badges (count), money, bag (item ids+qtys)
  EXCLUDE  frame counters, RNG state, animation/sprite timers, NPC/overworld-object positions
           and (deliberately) the dialogue TEXT itself - text is the dialogue driver's OWN
           progress signal (it already reads gStringVar3); folding it into the shared fingerprint
           would make every read-along page look like a state change.

ALL ADDRESSES ARE EMPIRICALLY CONFIRMED (recon_fingerprint.py controls, 2026-06-27):
  badge flags decode to 0/1/2 across after_pick/brock_done/misty_done; money (SaveBlock1+0x290 XOR
  SaveBlock2.encryptionKey+0xF20) decrypts to sane, frame-stable values; the bag pocket table is
  anchored on the already-control-verified PokeBalls pocket @ SaveBlock1+0x430.

DEFERRED, on purpose, NOT guessed (see the recon note in the increment-2 summary):
  - battle_turn_count: no control-verifiable address sourced AND the ONE-KIRA-FIREWALL says the
    battle menu is battle_agent's domain (detect/flag/abort only). v1 carries battle_active (bool);
    a turn counter, if increment 3 needs it, gets its own sourced+controlled recon first.
  - full overworld-MENU detection (bag/party/yes-no): menu_or_dialogue here is the pixel-verified
    overworld message band (box_open) - it reliably covers the micro layer's actual contexts
    (overworld dialogue). The bag/yes-no RAM signal is increment 4's job; its address is NOT
    guessed in here.

PURE READS - no input, no frames, no writes (safe to call inside a press loop)."""
from collections import namedtuple
import zlib

import firered_ram as ram
import pokemon_state as st

# ── confirmed addresses (recon_fingerprint.py) ──────────────────────────────────────────────
_OB0          = 0x02036E38      # object-event 0 = the player
_FACE_OFF     = 0x18            # facing-direction nibble (1=down 2=up 3=left 4=right)
_SB1_MAP_GRP  = 0x04
_SB1_MAP_NUM  = 0x05
_SB1_FLAGS    = 0x0EE0          # flag array within SaveBlock1
_BADGE_FLAG0  = 0x820           # FLAG_BADGE01_GET .. +7
_SB1_MONEY    = 0x0290          # SaveBlock1.money (u32, stored money ^ encryptionKey)
_SB2_ENC_KEY  = 0x0F20          # SaveBlock2.encryptionKey (u32)
_P_LEVEL, _P_HP = 0x54, 0x56    # within a 100-byte party-mon struct
# FRLG bag pockets within SaveBlock1; PokeBalls@0x430 is already control-verified (the catch fix).
_POCKETS = ((0x0310, 42), (0x03B8, 30), (0x0430, 13), (0x0464, 58), (0x054C, 43))

# ── MICRO no-progress thresholds (named, tunable) ───────────────────────────────────────────
# Consecutive IDENTICAL-fingerprint presses before the micro layer calls "no progress" and stops
# mashing. Used where the fingerprint is a RELIABLE per-press signal: the overworld / travel blocker
# / the macro tick (increment 3). Context-aware: a box up earns more rope (STALL_N_DIALOGUE).
STALL_N_DEFAULT  = 3            # generic repeated-press contexts (overworld, no box up)
STALL_N_DIALOGUE = 8            # a box is up + the fingerprint still moves (e.g. a menu cursor)
STALL_N_BLOCKER  = 2            # travel blocker-A: a SECOND identical box = plain NPC, not a trainer
# DIALOGUE is the EXCEPTION the recon forced (recon_microwatch.py, 2026-06-27): a long overworld
# message scrolls inside ONE constant gStringVar3 string (the Viridian parcel NPC = 36 presses, the
# buffer never changing), the box pixels animate every frame (the blinking continue-arrow), and the
# world fingerprint is static throughout - so there is NO reliable PER-PRESS "did the page advance?"
# signal. An immediate N=3 stop would BUTCHER healthy dialogue. The only safe dialogue guard is a
# FROZEN backstop: stop only once the SAME line has sat up, with NOTHING in the world moving, for
# this many presses - far above the longest real message (36 observed), so it never false-fires on a
# legit line yet still escapes a truly-stuck box in seconds instead of the old 300 blind presses.
DIALOGUE_FROZEN_LIMIT = 90     # same line + frozen world this many presses = a genuinely stuck box


WorldFingerprint = namedtuple(
    "WorldFingerprint",
    "map_id x y facing menu_or_dialogue battle_active party badges money bag")


def _facing(b):
    return b.rd8(_OB0 + _FACE_OFF) & 0x0F


def _badge_count(b, sb1):
    return sum(1 for i in range(8)
               if b.rd8(sb1 + _SB1_FLAGS + ((_BADGE_FLAG0 + i) >> 3)) & (1 << ((_BADGE_FLAG0 + i) & 7)))


def _money(b, sb1, key):
    return (b.rd32(sb1 + _SB1_MONEY) ^ key) & 0xFFFFFFFF


def _party(b):
    cnt = min(b.rd8(ram.GPLAYER_PARTY_CNT), 6)
    out = []
    for s in range(cnt):
        base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
        out.append((st.read_party_species(b, s), b.rd8(base + _P_LEVEL), b.rd16(base + _P_HP)))
    return tuple(out)


def _bag(b, sb1, key):
    """Sorted ((itemId, qty), ...) over every non-empty bag slot - catches pickups/purchases.
    qty is XOR'd with the low-16 encryption key; itemId is plain."""
    k16 = key & 0xFFFF
    items = []
    for off, cnt in _POCKETS:
        for s in range(cnt):
            slot = sb1 + off + s * 4
            iid = b.rd16(slot)
            if iid:
                items.append((iid, b.rd16(slot + 2) ^ k16))
    return tuple(sorted(items))


def fingerprint(b):
    """Snapshot the world into a hashable, comparable WorldFingerprint. Returns None if the save
    isn't allocated yet (pre-game) or any read faults - callers treat None as 'can't judge this
    press' and simply don't advance the no-progress count."""
    try:
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
        sb2 = b.rd32(ram.GSAVEBLOCK2_PTR)
        if not (ram.valid_ewram_ptr(sb1) and ram.valid_ewram_ptr(sb2)):
            return None
        key = b.rd32(sb2 + _SB2_ENC_KEY)
        from dialogue_drive import box_open       # local import: avoid a module-load cycle
        return WorldFingerprint(
            map_id=(b.rd8(sb1 + _SB1_MAP_GRP), b.rd8(sb1 + _SB1_MAP_NUM)),
            x=b.rds16(sb1 + ram.SB1_OFF_POS_X), y=b.rds16(sb1 + ram.SB1_OFF_POS_Y),
            facing=_facing(b),
            menu_or_dialogue=bool(box_open(b)),
            battle_active=bool(st.in_battle(b)),
            party=_party(b),
            badges=_badge_count(b, sb1),
            money=_money(b, sb1, key),
            bag=_bag(b, sb1, key))
    except Exception:
        return None


def brief(fp):
    """Compact one-line view for the WATCH=1 log: semantic fields + a crc of the noisy-but-bounded
    party/bag tuples so a viewer can SEE the fingerprint move (or not) tick to tick."""
    if fp is None:
        return "fp=None"
    pc = zlib.crc32(repr(fp.party).encode()) & 0xFFFF
    bc = zlib.crc32(repr(fp.bag).encode()) & 0xFFFF
    flags = ("D" if fp.menu_or_dialogue else "-") + ("B" if fp.battle_active else "-")
    return (f"fp[{fp.map_id} ({fp.x},{fp.y}) f{fp.facing} {flags} "
            f"badge{fp.badges} ${fp.money} party:{pc:04x} bag:{bc:04x}]")


class MicroWatch:
    """The MICRO watchdog: feed it a fresh fingerprint after each repeated press; it counts how
    many CONSECUTIVE presses left the world identical, and tells the caller when that crosses the
    (context-aware) threshold so the caller can STOP brute-forcing instead of mashing forever.

    The caller owns what to DO on a stall (dialogue driver: press B + return 'exhausted'; travel
    blocker: B + re-path). `progressed=True` lets a caller force-reset on a domain signal the shared
    fingerprint deliberately excludes - the dialogue driver passes new-text-appeared so a long but
    healthy multi-page conversation never trips."""

    def __init__(self, label=""):
        self.label = label
        self._last = None
        self.stall = 0          # consecutive identical-fingerprint presses

    def feed(self, fp, progressed=False):
        """Record one press's resulting fingerprint. Returns the current stall count. A None fp
        (unreadable) is treated as 'no judgement' - neither progress nor stall."""
        if fp is None:
            return self.stall
        if progressed or (self._last is not None and fp != self._last):
            self.stall = 0
        elif self._last is not None:
            self.stall += 1
        self._last = fp
        return self.stall

    def stalled(self, fp, limit=None):
        """True once enough consecutive presses have left the world unchanged. `limit` overrides
        the context-aware default (STALL_N_DIALOGUE while a box is up, else STALL_N_DEFAULT)."""
        if limit is None:
            limit = STALL_N_DIALOGUE if (fp is not None and fp.menu_or_dialogue) else STALL_N_DEFAULT
        return self.stall >= limit
