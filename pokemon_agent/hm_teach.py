"""hm_teach.py — teach a TM/HM from the overworld TM CASE to a party mon (stage 2 of the HM pipeline).

DERIVED 2026-07-06 (recon_teach_derive.py, live on the Vermilion canonical, throwaway cores):
  - START-menu cursor  = 0x020370F4 (rows post-dex: 0 POKEDEX, 1 POKEMON, 2 BAG, 3 <player>,
    4 SAVE, 5 OPTION, 6 EXIT). Readback-navigable.
  - Overworld BAG      : pocket byte 0x0203AD02 (0 Items / 1 Key Items / 2 Poke Balls) — SAME address
    as the in-battle bag; the LIST cursor is 0x0203AD06 (the in-battle 0x0203AD04 does NOT track here).
  - TM CASE            : ITEM_TM_CASE=364 in Key Items (366 is the Teachy TV — the first sin of this
    recon). Its list shows TMs sorted by number, then HMs sorted, then CLOSE; the applet's cursor is
    heap-state (NOT readable at a fixed address — the party-menu lesson), so row nav is BLIND DOWN
    with generous settles and the WHOLE flow is ground-truthed at the end via read_party_moves.
  - The teach party screen + the make-room/forget screens are blind-A/DOWN navigable; the forget
    list is rows 0-3 = current moves, row 4 = the new move (picking it = give up learning).

FAIL-SAFE: every path B-cascades back to the overworld; the ONLY success signal is the target mon's
decrypted move list containing the move id afterward. A wrong/eaten press can waste a pass, never
mis-report. The caller supplies mon/forget choices (the oracle's judgment); `default_plan` gives the
headless policy lean.
"""
import time

import firered_ram as ram
import pokemon_state as st

START_CURSOR = 0x020370F4        # pause-menu row (derived, recon_teach_derive)
BAG_POCKET = 0x0203AD02          # shared with the in-battle bag
BAG_LIST_CURSOR = 0x0203AD06     # overworld bag list row (in-battle uses 0x0203AD04)
ITEM_TM_CASE = 364
HM_ITEM = {"cut": 339, "fly": 340, "surf": 341, "strength": 342, "flash": 343,
           "rock_smash": 344, "waterfall": 345, "dive": 346}
TM_FIRST = 289                   # ITEM_TM01
HM_FIRST = 339                   # ITEM_HM01
KEY_ITEMS_OFF, TM_CASE_OFF = 0x3B8, 0x464


def _sb1(b):
    return b.rd32(ram.GSAVEBLOCK1_PTR)


def pocket_items(b, off, n):
    out = []
    for i in range(n):
        iid = b.rd16(_sb1(b) + off + i * 4)
        if iid == 0:
            break
        out.append(iid)
    return out


def tm_case_row(b, item_id):
    """DISPLAY row of `item_id` in the TM Case = its RAW POCKET-ARRAY index. Frame-proven
    (surge run 3): the old sorted-TMs-then-HMs model computed row 3 for HM01, but selecting
    row 3 lit 'NOT ABLE!' on all four mons — that row was TM39 Rock Tomb, exactly the raw
    array's index 3 ([HM01, TM03, TM28, TM39]). The case lists the pocket as-is; HM01 sat
    at row 0. None if the item isn't in the case."""
    have = pocket_items(b, TM_CASE_OFF, 58)
    return have.index(item_id) if item_id in have else None


class TeachFlow:
    """Drives START -> BAG -> Key Items -> TM CASE -> row -> USE -> party slot -> (make room ->
    forget idx) -> verify. Duck-types campaign (.b, .render)."""

    def __init__(self, campaign, log=print, on_event=None):
        self.c = campaign
        self.b = campaign.b
        self.log = log
        self.emit = on_event or (lambda *a, **k: None)

    def _press(self, key, settle=24, hold=8, rel=12):
        self.b.press(key, hold, rel, self.c.render, owner="agent")
        for _ in range(settle):
            self.b.run_frame()
            self.c.render()

    def _nav_byte(self, addr, target, down="DOWN", up="UP", tries=12):
        for _ in range(tries):
            v = self.b.rd8(addr)
            if v == target:
                return True
            self._press(down if v < target else up, settle=16)
        return self.b.rd8(addr) == target

    # screen classifiers (pixel truth — same doctrine as battle_agent's _party_screen/_bag_screen).
    # CASE vs BAG share the pale-yellow list palette; the CASE has the BLUE description panel across
    # the bottom (the bag's bottom stays tan). Points are native 240x160.
    def _classify(self):
        p = self.b.frame_rgb().load()

        def near(x, y, rgb, tol=90):
            c = p[x, y][:3]
            return sum(abs(c[i] - rgb[i]) for i in range(3)) < tol

        teal_hits = sum(1 for x, y in ((30, 110), (60, 115), (20, 90), (70, 108))
                        if p[x, y][0] < 100 and p[x, y][1] > 120 and p[x, y][2] > 120
                        and abs(p[x, y][1] - p[x, y][2]) < 40)
        if teal_hits >= 3:
            return "party"
        # the FORGET/"KNOWN MOVES" summary (tk_020): blue plate at (200,4)=(0,108,191) + the
        # whitish-cyan move boxes at x=122 — no other screen in the flow has this pair.
        if (near(200, 4, (0, 108, 191)) and p[122, 67][0] > 225 and p[122, 67][1] > 240
                and p[122, 90][0] > 225 and p[122, 90][1] > 240):
            return "forget"
        yellow_hits = sum(1 for x, y in ((160, 30), (200, 60), (120, 10))
                          if p[x, y][0] > 240 and p[x, y][1] > 240 and 180 < p[x, y][2] < 230)
        if yellow_hits >= 2:
            # measured ground truth (teach_2/3/5, th_013): the CASE has a GRAY header plate at
            # (20,20)=(187,187,187) + the yellow disc art at (30,60)=(247,214,57); the BAG's header
            # is ORANGE (232,139,65). The USE/GIVE/EXIT sub-box turns (200,115)/(215,125)/(205,135)
            # WHITE (they are the blue description panel otherwise).
            if near(20, 20, (187, 187, 187)) and near(30, 60, (247, 214, 57)):
                white_sub = sum(1 for x, y in ((200, 115), (215, 125), (205, 135))
                                if min(p[x, y][:3]) > 200) >= 2
                return "case_sub" if white_sub else "case"
            return "bag"
        return "dialogue"

    # party-screen cursor READBACK (visual — the heap cursor has no fixed address): the SELECTED
    # right-column slot draws an orange border AROUND ITS BOX OUTLINE. The old single-pixel probe
    # at x=225 sat INSIDE the box and read nothing on the teach chooser (surge run 3) — detect the
    # TOP BORDER as a horizontal RUN instead (≥3 of 5 sampled x's orange on one row). Two anchor
    # sets per slot: measured live on the teach chooser (10+21k) + the legacy tj_004 tops (14+21k).
    # THIRD anchor set (surf_teach 2026-07-07, measured live): the 6-mon teach chooser
    # spaces rows 24px (10+24k) — the 21k anchors miss slots 4/5 entirely (the cursor
    # walked past Lapras to CANCEL and the teach failed LOUD).
    _SLOT_TOPS = {1: (10, 14), 2: (34, 31, 35), 3: (58, 52, 56),
                  4: (82, 73, 77), 5: (106, 94, 98)}

    def _party_cursor(self):
        p = self.b.frame_rgb().load()
        for slot, tops in self._SLOT_TOPS.items():
            for y0 in tops:
                for dy in (-2, -1, 0, 1, 2, 3):
                    n = 0
                    for x in (100, 130, 160, 190, 220):
                        r, g, b = p[x, y0 + dy][:3]
                        if r > 240 and 80 < g < 140 and b < 70:
                            n += 1
                    if n >= 3:
                        return slot
        return None                                   # lead or CANCEL (no slot border lit)

    def _party_goto(self, target, tries=14):
        """Closed-loop cursor walk to `target` slot on the OVERWORLD party screen. The menu
        REMEMBERS its last position across opens (tj_004: it opened on slot 2), so counted blind
        presses walk off the end — read the border, step, re-read."""
        for _ in range(tries):
            cur = self._party_cursor()
            if cur == target:
                return True
            if cur is None:                           # on the lead (or CANCEL): RIGHT enters the column
                if target == 0:
                    return True                       # want the lead and no slot is lit -> likely there
                self._press("RIGHT", settle=18)
                if self._party_cursor() is None:      # still nothing lit -> we were on CANCEL: UP = slot 5
                    self._press("UP", settle=18)
                continue
            if target == 0:
                self._press("LEFT", settle=18)        # any slot -> LEFT returns to the lead panel
                continue
            self._press("DOWN" if cur < target else "UP", settle=18)
        return self._party_cursor() == target

    # forget/"KNOWN MOVES" screen cursor: the selected row's RED-ORANGE border draws TWO
    # horizontal runs (box top + bottom). MEASURED live (recon_forget_probe, victory_run1
    # postmortem): tops = 18/46/74/102/130, 28px spacing — the old 67/90/112 were unmeasured
    # probes and missed rows 2-4 entirely (the EQ-teach B-out). A row's BOTTOM run lands in
    # the NEXT row's top window, so rows are checked ASCENDING (the true top border wins).
    # The cursor WRAPS row4 -> row0 (measured), so the closed-loop goto always converges.
    _FORGET_TOPS = (18, 46, 74, 102, 130)

    def _forget_cursor(self):
        p = self.b.frame_rgb().load()
        for k, y0 in enumerate(self._FORGET_TOPS):
            for dy in (-2, -1, 0, 1, 2):
                n = 0
                for x in (116, 119, 122, 125, 128):
                    r, g, b = p[x, y0 + dy][:3]
                    if r > 200 and g < 120 and b < 80:
                        n += 1
                if n >= 2:
                    return k
        return None

    def _forget_goto(self, target, tries=12):
        for _ in range(tries):
            cur = self._forget_cursor()
            if cur == target:
                return True
            if cur is None:
                self._press("DOWN", settle=18)
                continue
            self._press("DOWN" if cur < target else "UP", settle=18)
        return self._forget_cursor() == target

    def _b_cascade(self, n=12):
        import travel as tv
        for _ in range(n):
            if tv.map_id(self.b)[0] in (1, 3, 9) and self.b.rd8(START_CURSOR) is not None:
                # cheap 'menus gone' proxy: two Bs beyond the last visible change are harmless
                pass
            self._press("B", settle=16)

    def use_field_move(self, mon_slot, verify, label="field-move", max_seconds=60):
        """Use a FIELD MOVE from the overworld party menu: START -> POKEMON -> `mon_slot` ->
        the mon's submenu (field moves list FIRST, above SUMMARY/SWITCH/...) -> A; `verify()`
        (RAM truth — e.g. FLAG_SYS_FLASH_ACTIVE for Flash) decides success. Attempt k selects
        submenu row k (no cursor address known for this submenu, so rows are blind — but each
        attempt reopens the menu fresh, making the count deterministic). Returns
        'used' | 'failed'; fail-safe B-cascade back to the overworld either way."""
        t0 = time.time()
        self.b.set_input_owner("agent")
        for attempt in range(3):
            if time.time() - t0 > max_seconds:
                break
            self._b_cascade(6)                                   # clean slate
            self._press("START", settle=60)
            if not self._nav_byte(START_CURSOR, 1):              # row 1 = POKEMON (post-dex menu)
                self.log(f"   [{label}] !! START-menu cursor no-response — retrying")
                continue
            self._press("A", settle=90)                          # open the party screen
            ok_party = False
            for _ in range(10):
                if self._classify() == "party":
                    ok_party = True
                    break
                self._press("A", settle=20)
            if not ok_party:
                self.log(f"   [{label}] !! party screen never came up (attempt {attempt})")
                continue
            if not self._party_goto(mon_slot):
                self.log(f"   [{label}] !! party cursor couldn't reach slot {mon_slot}")
                continue
            self._press("A", settle=40)                          # the mon's action submenu
            for _ in range(attempt):                             # attempt k -> submenu row k
                self._press("DOWN", settle=14)
            self._press("A", settle=40)                          # fire the field move
            for _ in range(90):                                  # drain the animation
                if verify():
                    break
                self.b.run_frame(); self.c.render()
            for _ in range(4):                                   # a held "used FLASH!" box
                if verify():
                    break
                self._press("B", settle=20)
            if verify():
                self._b_cascade()
                self.log(f"   [{label}] VERIFIED used (attempt {attempt}, submenu row {attempt})")
                return "used"
            self.log(f"   [{label}] attempt {attempt} (row {attempt}) did not verify — backing out")
        self._b_cascade()
        self.log(f"   [{label}] !! FAILED — field move never verified (LOUD)")
        return "failed"

    def teach(self, hm_key, mon_slot, forget_idx=None, max_seconds=120,
              item_override=None, move_override=None):
        """Teach HM `hm_key` to party `mon_slot`; forget_idx = which current move to overwrite when
        the mon already has 4 (None = the mon has room / caller believes so). item/move_override
        drive the SAME flow for an arbitrary TM (the control-test vehicle). Returns
        'taught' | 'not_in_case' | 'failed' (fail-safe: overworld restored, nothing mis-reported)."""
        t0 = time.time()
        item = item_override or HM_ITEM.get(hm_key)
        move_id = move_override or {"cut": 15, "fly": 19, "surf": 57, "strength": 70, "flash": 148,
                                    "rock_smash": 249, "waterfall": 127, "dive": 291}[hm_key]
        row = tm_case_row(self.b, item)
        if row is None:
            return "not_in_case"
        before = st.read_party_moves(self.b, mon_slot)
        if move_id in before:
            return "taught"
        self.b.set_input_owner("agent")
        self.log(f"   [teach] {hm_key} -> slot {mon_slot} (case row {row}, forget_idx {forget_idx})")
        # 1. START menu -> BAG (readback nav on the derived cursor).
        # OPEN-VERIFY first: every menu cursor byte is STALE across sessions (tm_errand run-16:
        # a second back-to-back teach trusted teach 1's parked values straight down the chain
        # and drove the OVERWORLD with A) — the cursor must RESPOND to a press before any nav
        # trusts it. A DOWN on a closed menu steps the player, so this only retries START.
        opened = False
        self._press("START", settle=60)
        for _ in range(4):
            c0 = self.b.rd8(START_CURSOR)
            self._press("DOWN", settle=24)
            if self.b.rd8(START_CURSOR) != c0:
                opened = True
                break
            self._press("START", settle=60)
        if not opened or not self._nav_byte(START_CURSOR, 2):
            self.log("   [teach] !! START menu never opened / cursor no-response — aborting (B out)")
            self._b_cascade(); return "failed"
        self._press("A", settle=80)                              # open the bag
        # 2. Key Items pocket
        for _ in range(4):
            if self.b.rd8(BAG_POCKET) == 1:
                break
            self._press("RIGHT" if self.b.rd8(BAG_POCKET) < 1 else "LEFT", settle=20)
        if self.b.rd8(BAG_POCKET) != 1:
            self.log("   [teach] !! couldn't reach Key Items — aborting"); self._b_cascade(); return "failed"
        # 3. TM CASE row (readback) -> open it
        ki = pocket_items(self.b, KEY_ITEMS_OFF, 30)
        if ITEM_TM_CASE not in ki:
            self.log("   [teach] !! no TM Case in Key Items"); self._b_cascade(); return "failed"
        if not self._nav_byte(BAG_LIST_CURSOR, ki.index(ITEM_TM_CASE)):
            self.log("   [teach] !! bag list cursor no-response — aborting"); self._b_cascade(); return "failed"
        self._press("A", settle=40)                              # select TM CASE
        self._press("A", settle=100)                             # USE -> the case UI (applet fade)
        # 4-6. STATE MACHINE (blind press SEQUENCES desynced — frame-diagnosed: one shifted beat
        # turned USE into GIVE and handed the TM to the lead as a HELD ITEM). Each iteration
        # classifies the SCREEN (pixel truth, the stale-byte doctrine) and takes ONE step:
        #   case list  -> DOWN toward the row (counted), then A (select) -> A on the sub-box = USE
        #   party teal -> RIGHT/DOWN to the slot (counted), then A picks the mon
        #   dialogue   -> A (advances make-room Y/N with YES-default + all learn text)
        #   forget-summary (only when forget_idx is set) -> DOWN x idx + A, once
        downs_left, picked, sub_seen, party_navved, forgot = row, False, False, False, False
        post_pick_a = 0
        case_homed = False
        for _ in range(70):
            if move_id in st.read_party_moves(self.b, mon_slot):
                break
            if time.time() - t0 > max_seconds:
                break
            scr = self._classify()
            if scr == "case_sub":
                sub_seen = True
                self._press("A", settle=90)                      # USE (top row default)
            elif scr == "case":
                if sub_seen and picked:
                    # NOT done — the make-room/learn dialogue renders OVER the case UI (frame-
                    # diagnosed, it05): walk it with A. The loop's top move-check is the real
                    # done signal; post_pick_a bounds the walk so a silent dead-end still exits.
                    post_pick_a += 1
                    if post_pick_a > 12:
                        break
                    self._press("A", settle=60)
                elif not case_homed:
                    # the case UI SORTS its backing array on open (HMs first): a row computed from
                    # the raw array right after an acquisition is STALE (hm05 run 7: the aide's
                    # give-script appended HM05 at raw 5; the open case displayed it at row 1 and
                    # the counted DOWNs picked a wrong row). Re-read now that the case is open.
                    row2 = tm_case_row(self.b, item)
                    if row2 is not None and row2 != row:
                        self.log(f"   [teach] case re-sorted on open: row {row} -> {row2}")
                        row = row2
                        downs_left = row2
                    # the case list cursor is HEAP-allocated with NO readback and it REMEMBERS
                    # its row across opens (surge run 3: DOWNs counted from a parked cursor
                    # selected the wrong row -> a case<->bag oscillation that never reached the
                    # party chooser). UP clamps at row 0 (recon-proven live) — HOME first.
                    for _ in range(row + 9):
                        self._press("UP", settle=18)
                    case_homed = True
                elif downs_left > 0:
                    self._press("DOWN", settle=24)
                    downs_left -= 1
                else:
                    self._press("A", settle=50)                  # select the row -> sub-box
            elif scr == "party":
                if not party_navved:
                    if not self._party_goto(mon_slot):           # closed-loop (the menu REMEMBERS its
                        self.log("   [teach] !! party cursor never reached the slot — B out")
                        break                                    #  position across opens — never count blind)
                    party_navved = True
                    self._press("A", settle=90)                  # pick the mon
                    picked = True
                else:
                    self._press("A", settle=60)                  # teach dialogue over the party UI
            elif scr == "bag":
                self._press("A", settle=80)                      # the case-USE press hasn't landed yet
                #                                                  (bag cursor is readback-parked on the case)
            elif scr == "forget" and not forgot:
                tgt = forget_idx if forget_idx is not None else 0
                if not self._forget_goto(tgt):
                    self.log(f"   [teach] !! forget cursor never reached row {tgt} — B out")
                    break
                self._press("A", settle=90)
                forgot = True
            else:                                                # dialogue / transition -> advance
                self._press("A", settle=50)
        # 7. B-cascade out of case/bag/menu regardless of outcome
        self._b_cascade()
        after = st.read_party_moves(self.b, mon_slot)
        if move_id in after:
            self.log(f"   [teach] VERIFIED: slot {mon_slot} moves {before} -> {after}")
            return "taught"
        self.log(f"   [teach] !! NOT taught (moves {after} unchanged) — failed LOUD")
        return "failed"


# ── headless policy: which mon takes the HM + which move makes room ───────────
# TM/HM compatibility = ROM TRUTH, not a hand table. gTMHMLearnsets is u64[species] (8 bytes per
# species): bit 0 = TM01 … bit 49 = TM50, bit 50 = HM01 Cut … bit 57 = HM08 Dive. Base address
# CONTROL-verified 2026-07-07 (recon_tmhm_scan.py: known compat facts all match, incl. the
# live-taught Cut→Raticate). Retires the hand-maintained _CUT_OK class — which mis-called Flash
# ("none of her six learns it"): Venusaur AND Persian are Flash-compatible in FRLG.
GTMHM_LEARNSETS = 0x08252BC8
_HM_BIT = {"cut": 50, "fly": 51, "surf": 52, "strength": 53, "flash": 54,
           "rocksmash": 55, "waterfall": 56, "dive": 57}


def hm_compatible(b, hm_key, species):
    """Can `species` learn this HM? Read from ROM gTMHMLearnsets (authoritative; game-agnostic
    pattern — only the base address is per-game). False on unknown key / species 0 / read error."""
    bit = _HM_BIT.get(hm_key)
    if bit is None or not species:
        return False
    try:
        lo = b.rd32(GTMHM_LEARNSETS + species * 8)
        hi = b.rd32(GTMHM_LEARNSETS + species * 8 + 4)
        return bool((((hi << 32) | lo) >> bit) & 1)
    except Exception:
        return False


def tm_compatible(b, tm_no, species):
    """Can `species` learn TM `tm_no` (1-50)? Same ROM gTMHMLearnsets row as hm_compatible —
    bits 0-49 are TM01-TM50 (bit = tm_no-1), 50-57 the HMs. False on bad input / read error."""
    if not species or not (1 <= tm_no <= 50):
        return False
    try:
        lo = b.rd32(GTMHM_LEARNSETS + species * 8)
        hi = b.rd32(GTMHM_LEARNSETS + species * 8 + 4)
        return bool((((hi << 32) | lo) >> (tm_no - 1)) & 1)
    except Exception:
        return False


# expendable move classes for the forget choice: pure-status/no-power utility first, never the
# mon's strongest damaging move.
_PRECIOUS = {73}                              # leech seed etc. — never auto-forget


def default_plan(b, hm_key, party_count):
    """(mon_slot, forget_idx, reason) headless lean: the lowest-level COMPATIBLE non-lead with a
    free slot, else the compatible mon whose weakest no-power move can go. None if no candidate."""
    cands = []
    for s in range(party_count):
        sp = st.read_party_species(b, s)
        if not hm_compatible(b, hm_key, sp):
            continue
        lv = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
        moves = st.read_party_moves(b, s)
        free = 0 in moves or len([m for m in moves if m]) < 4
        cands.append((s, sp, lv, moves, free))
    if not cands:
        return None
    # free slot first, then lowest level (don't burn the ace's moveset), lead last
    cands.sort(key=lambda c: (not c[4], c[0] == 0, c[2]))
    s, sp, lv, moves, free = cands[0]
    if free:
        return s, None, f"slot {s} has room — no move given up"
    scored = []
    for i, m in enumerate(moves):
        if not m or m in _PRECIOUS:
            continue
        _t, power = st.move_info(b, m)        # ROM gBattleMoves truth (same read the engine trusts)
        scored.append((power or 0, i, m))
    scored.sort()
    if not scored:
        return s, 0, "overwriting the first move (no scoring data)"
    return s, scored[0][1], f"forgetting move {scored[0][2]} (weakest, power {scored[0][0]})"
