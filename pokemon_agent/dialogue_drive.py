"""dialogue_drive.py - ONE general OVERWORLD dialogue-advance primitive (additive, isolated).

Pacing is first-class: a viewer should be able to READ ALONG. This drives every overworld text
box (NPCs, signs, the gym badge/TM award, cutscenes, item pickups) reliably to close, at a
human-watchable cadence - never instant-mash, never a crawl, never wedged.

Box detection is SIDE-EFFECT-FREE via the bottom message-box pixel band (the same proven idea as
the battle engine's _white_box): when an overworld dialogue box is open, that band is solid white
(~27/28 sampled px); on the open map it is 0/28. gStringVar3 (via DialogueReader) gives the text
CONTENT for read-along holds + logging, but the PIXEL band - not the (stale-after-close) buffer -
is the authority on whether a box is still up.

BATTLE text stays the battle engine's job (battle_agent._advance_text) - this is overworld only,
so the proven, regression-green battle path is left completely untouched.

PACING IS IN WALL-CLOCK SECONDS, not frames: the emulator free-runs at 10-15x real-time headless
and is throttled to real-time only on the live stream (AudioPump's blocking write). A frame-count
hold would therefore mean a totally different real duration in each context. Seconds make the
read-along dwell identical on stream and in any test - the same convention play_live already uses.
"""
import os
import time

from dialogue_reader import DialogueReader
import world_fingerprint as wf      # MICRO watchdog: stop A-mashing an exhausted/looping NPC

# FIX 1 (dialogue half) — a CYCLING NPC (it rotates through several distinct lines forever, e.g. the
# wild Slowbro "turned away.." / "loafing around..") defeats the consecutive-same-line `frozen` counter
# because `new_line` keeps firing. So we ALSO count how many times each distinct line reappears within
# one drive() call: once any line has been shown DIALOGUE_LOOP_REPEAT times, it's looping, not
# progressing -> disengage (B, walk away). Tolerance is low (a real speech never repeats a line 3x).
DIALOGUE_LOOP_REPEAT = int(os.getenv("POKEMON_DIALOGUE_LOOP_REPEAT", "3"))

# Bottom overworld message-box band: solid white while a box is open, dark (map) when closed.
# Verified across misty_done (box up: 27-28/28 per row) vs cerulean/pewter/viridian (0/28).
_BOX_ROWS = (126, 132, 138, 144)
_BOX_XS = tuple(range(16, 226, 10))           # ~21 samples per row


_OB0 = 0x02036E38               # object-event 0 = the player
_FACE_OFF = 0x18                # facing direction nibble (1=down 2=up 3=left 4=right)


def box_open(b):
    """True iff an overworld dialogue box is currently displayed (bottom band solid white)."""
    p = b.frame_rgb().load()
    # DUAL CLAUSE (night shift 4, both populations measured): a real FRLG message box
    # is 255-white fill + dark text/border pixels -> >=40% pure-white AND >=78% bright.
    # The Seafoam ICE floor false-positived the old single 200-threshold FOREVER (the
    # silent drain-livelock class): 100% of its band sits in 200-242 with ZERO pixels
    # >242 -> fails the pure-white clause. A text-heavy trainer-intro box measured
    # 61% >242 / 87% >200 -> passes both. (mgba scales 5-bit via (c<<3)|(c>>2).)
    n = len(_BOX_ROWS) * len(_BOX_XS)
    vals = [min(p[x, y]) for y in _BOX_ROWS for x in _BOX_XS]
    return (sum(1 for v in vals if v > 242) >= int(0.40 * n)
            and sum(1 for v in vals if v > 200) >= int(0.78 * n))


def player_facing(b):
    return b.rd8(_OB0 + _FACE_OFF) & 0x0F


def _player_coords(b):
    return (b.rds16(_OB0 + 0x10) - 7, b.rds16(_OB0 + 0x12) - 7)


class DialogueDriver:
    # HINT-EXTRACTION TAP (soul-debt #12, the info half): campaign sets this ONCE to its
    # HintLedger feeder; every ad-hoc DialogueDriver instance then reports each accepted NEW
    # line through it (drive() below). Class-level on purpose — drivers are constructed
    # ad hoc all over campaign, and the tap must catch all of them (one campaign per process).
    # None = no-op (recon tools unaffected). Failures are swallowed per-call: intel must
    # never break the proven drive loop.
    line_sink = None

    def __init__(self, b, render=None, log=print, owner="agent"):
        self.b = b
        self.render = render or (lambda: None)
        self.log = log
        self.owner = owner
        self.dr = DialogueReader(b)

    def _wait(self, n):
        """Frame-based wait - for GAME-mechanic timing (input registration), where the unit is
        emulator frames, not viewer seconds."""
        for _ in range(n):
            self.b.run_frame()
            self.render()

    def _hold_s(self, seconds):
        """Wall-clock viewer hold: keep the (static) text on screen for `seconds` real seconds,
        running frames so the window stays live + audio keeps flowing. Faithful at any fps."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.b.run_frame()
            self.render()

    def _control_returned(self):
        """SIDE-EFFECT-FREE check that overworld control is back (the cutscene/script has fully
        released): press a direction the player is NOT already facing and see if the player TURNS
        (facing nibble changes) or steps. While a script holds the field locked, input is ignored
        -> no facing change. Once control returns, the press turns the player even against a wall
        (a harmless in-place turn), so a facing change is the authoritative 'done' signal. This is
        what distinguishes a TRUE close from the 1-2 frame box flicker BETWEEN cutscene pages."""
        f0 = player_facing(self.b)
        c0 = _player_coords(self.b)
        probe = "DOWN" if f0 != 1 else "RIGHT"            # a key that implies a different facing
        self.b.press(probe, 8, 8, self.render, owner=self.owner)
        self._wait(6)
        return player_facing(self.b) != f0 or _player_coords(self.b) != c0

    def _close_box(self, max_tries=12):
        """Bounded B-mash to dismiss an EXHAUSTED/looping dialogue box once the micro-watchdog has
        decided more A-pressing is futile: stop pushing A (which kept re-initiating the NPC), tap B
        a few times to clear the current box, confirm control is back. Best-effort + LOUD if it
        won't close - the caller still treats the NPC as done and re-paths AWAY (so it's not
        re-triggered), which is the actual un-stick."""
        for _ in range(max_tries):
            if not box_open(self.b) and self._control_returned():
                return True
            self.b.press("B", 4, 10, self.render, owner=self.owner)
            self._hold_s(0.03)
        self.log(f"   [dlg] !! B-close did not regain control after {max_tries} tries - LOUD "
                 f"(stopping the mash anyway; caller re-paths away)")
        return False

    def _read_seconds(self, text, min_s, max_s, base_s, per_char_s):
        """Seconds to hold a freshly-shown message so a viewer can read ALONG: proportional to its
        length (short lines quick, long sentences linger), clamped to [min_s, max_s]. ~per_char_s
        s/char approximates a comfortable reading rate."""
        return min(max_s, max(min_s, base_s + per_char_s * len(text)))

    # Default curve TUNED LIVE 2026-06-26 by Jonny at TRUE real-time (AudioPump throttle) WITH music
    # - the only valid baseline (headless free-runs ~10-15x, which made every earlier read too fast).
    # 50% FASTER pass (Jonny, after watching the opening debut at true speed): halved from the first
    # locked curve (min 0.09/max 0.45/base 0.05/per_char 0.0053/gap 0.043) -> snappier everywhere.
    # 2nd 50% FASTER pass (Jonny 2026-06-27, watchability — FireRed has a LOT of text): halved again
    # (min 0.022/max 0.11/base 0.012/per_char 0.0014/gap 0.011). This only shortens the read-along
    # HOLD + the A-tap cadence; the frozen-backstop is PRESS-COUNT based (DIALOGUE_FROZEN_LIMIT) and
    # box_open/_control_returned are pixel/press based, so detection + the stuck-box guard are untouched.
    def drive(self, stop_when=None, label="", min_s=0.022, max_s=0.11, base_s=0.012,
              per_char_s=0.0014, page_gap_s=0.011, max_steps=300):
        """Advance an open overworld dialogue at a watchable pace until control RETURNS (or
        stop_when() fires). Returns 'stopped' | 'closed' | 'exhausted' | 'timeout'(loud).

        Each NEW message is held for a WALL-CLOCK duration SCALED BY ITS LENGTH (min_s..max_s) so
        the viewer can read along - short lines breeze by, long sentences linger. page_gap_s is the
        steady real-time cadence between A-taps for continuation pages (no instant-mash). Seconds,
        not frames, so the feel is identical on the real-time stream and in any (fast) test.

        Box state is read from the bottom pixel band; CONTENT from gStringVar3 (for read-along
        holds). 'Done' is decided by control returning (player can turn), NOT by the box momentarily
        clearing between pages - which is exactly the flicker that made naive close-on-empty bail
        mid-cutscene and leave the field locked. Battle text stays the battle engine's job.

        MICRO-WATCHDOG (increment 2): the old failure mode was up to 300 BLIND A-presses into a box
        that never closes (a stuck/looping NPC = the free-roam wedge). Recon proved dialogue has no
        reliable per-press progress signal (a long message scrolls inside ONE constant buffer, the
        box pixels animate every frame, the world fingerprint is static) - so we use a SAFE FROZEN
        BACKSTOP, not a hair-trigger: only when the SAME line has sat up with NOTHING in the world
        changing for DIALOGUE_FROZEN_LIMIT straight presses (far above the longest real message) do
        we call it a genuinely stuck box -> stop mashing A, tap B, return 'exhausted' so the caller
        re-paths away. A healthy long dialogue closes (box clears -> control returns) long before."""
        if not box_open(self.b) and self._control_returned():
            return "closed"                               # nothing to drive (no box, control active)
        last = None
        frozen = 0                                        # consecutive presses: same line + frozen world
        fp_prev = None
        seen = {}                                         # FIX 1: normalized line -> times shown (cycle detect)
        for _ in range(max_steps):
            if stop_when and stop_when():
                return "stopped"
            if box_open(self.b):
                cur = self.dr._read_buffer()
                new_line = bool(cur and cur != last)
                if new_line:                              # a NEW message -> snap to FULL, then read-along
                    last = cur
                    # FIX 1 — CYCLE DETECT: a looping NPC rotates distinct lines forever (frozen-world
                    # counter never trips). If any one line has now reappeared DIALOGUE_LOOP_REPEAT
                    # times this call, it's a loop, not progress -> disengage (don't read it again).
                    _k = " ".join(cur.lower().split())
                    seen[_k] = seen.get(_k, 0) + 1
                    if seen[_k] >= DIALOGUE_LOOP_REPEAT:
                        self.log(f"   [dlg{(' ' + label) if label else ''}] !! LOOPING DIALOGUE — "
                                 f"line seen x{seen[_k]} ({cur.replace(chr(10), ' ')[:48]!r}) -> B, "
                                 f"walking away (cycle-watchdog)")
                        self._close_box()
                        return "exhausted"
                    self.log(f"   [dlg{(' ' + label) if label else ''}] "
                             f"{cur.replace(chr(10), ' ')[:72]!r}")
                    # CLEAN RENDER FIX (2026-06-27, watchability): the box used to be advanced by a single
                    # A WHILE the text was still typing — A speeds the in-progress print, so each page
                    # rendered as normal-typing -> faster-while-A-held -> snap = the chunky/zippy/uneven
                    # look (and short lines that finished first looked clean = the inconsistency tell). So
                    # FIRST complete the print at once (one A finishes an in-progress page WITHOUT moving
                    # on), so the whole line shows cleanly+instantly; THEN read-along HOLD on the full
                    # text; THEN the advance below moves to the next page. Only the RENDER is made clean —
                    # the fast advance cadence (the halved curve) is UNCHANGED.
                    self.b.press("A", 4, 6, self.render, owner=self.owner)
                    last = self.dr._read_buffer() or last  # adopt what's shown now (snapped-full text)
                    if DialogueDriver.line_sink is not None:
                        try:                              # hint tap: report the full accepted line
                            DialogueDriver.line_sink(last.replace("\n", " "))
                        except Exception:
                            pass
                    self._hold_s(self._read_seconds(last, min_s, max_s, base_s, per_char_s))
                self.b.press("A", 4, 10, self.render, owner=self.owner)  # advance one page, paced
                self._hold_s(page_gap_s)
                fp = wf.fingerprint(self.b)               # did this press change the line OR the world?
                advanced = new_line or (fp is None) or (fp != fp_prev)
                frozen = 0 if advanced else frozen + 1
                fp_prev = fp
                if frozen and frozen % 15 == 0:           # WATCH=1 visibility: show a box sitting still
                    self.log(f"   [dlg-uwatch{(' ' + label) if label else ''}] {wf.brief(fp)} "
                             f"same line, frozen world x{frozen}/{wf.DIALOGUE_FROZEN_LIMIT}")
                if frozen >= wf.DIALOGUE_FROZEN_LIMIT:
                    self.log(f"   [dlg{(' ' + label) if label else ''}] !! BOX STUCK - same line / "
                             f"frozen world x{frozen} (never closing) -> B, STOP mashing (micro-watchdog)")
                    self._close_box()
                    return "exhausted"
            else:
                # box not showing: either DONE (control back) or a brief between-pages gap (locked).
                if self._control_returned():
                    return "closed"
                self.b.press("A", 4, 10, self.render, owner=self.owner)  # still locked -> push gap
                self._hold_s(page_gap_s)
        self.log(f"   [dlg{(' ' + label) if label else ''}] !! max_steps={max_steps} reached, "
                 f"control not regained - LOUD (not silently giving up)")
        return "timeout"
