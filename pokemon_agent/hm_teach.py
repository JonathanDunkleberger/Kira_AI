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
ITEMS_OFF = 0x310                # SaveBlock1 Items pocket (potions/cures), 42 slots
P_STATUS, P_HP, P_MAXHP, PARTY_MON_SIZE = 0x50, 0x56, 0x58, 100


def items_pocket_rows(b):
    """[(item_id, qty), ...] of the ITEMS pocket in DISPLAY order — hole-skipped (a zero-qty
    slot mid-pocket is skipped by the game's list too), qty decrypted with the SB2 low-16 key.
    Same doctrine as battle_agent._items_pocket (the run17 break-at-first-zero collapse)."""
    sb1 = _sb1(b)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    out = []
    for s in range(42):
        slot = sb1 + ITEMS_OFF + s * 4
        iid = b.rd16(slot)
        qty = b.rd16(slot + 2) ^ key
        if iid and qty:
            out.append((iid, qty))
    return out


def items_pocket_qty(b, item_id):
    return sum(q for i, q in items_pocket_rows(b) if i == item_id)


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
        # The "which move to forget?" screen ALWAYS opens with the cursor on row 0
        # (FRLG). Vision readback (_forget_cursor) proved FLAKY for non-zero rows
        # (NS12 EQ-teach idx=1: silent no-op, moves unchanged), so drive it BLIND
        # from the known row-0 home first — press DOWN `target` times — then close-
        # loop-CORRECT with vision if it can see the cursor, but NEVER hard-fail on
        # a None read (the blind position is already right). This makes forgetting a
        # non-row-0 slot (keep Razor Leaf, drop Cut) as reliable as the row-0 path.
        for _ in range(target):
            self._press("DOWN", settle=18)
        for _ in range(tries):
            cur = self._forget_cursor()
            if cur is None or cur == target:
                return True
            self._press("DOWN" if cur < target else "UP", settle=18)
        return True

    def _b_cascade(self, n=12):
        import travel as tv
        for _ in range(n):
            if tv.map_id(self.b)[0] in (1, 3, 9) and self.b.rd8(START_CURSOR) is not None:
                # cheap 'menus gone' proxy: two Bs beyond the last visible change are harmless
                pass
            self._press("B", settle=16)
        # START-MENU TAIL (2026-08-05 #2, the Mt. Ember EXIT-cursor wedge): every hm_teach
        # flow opens via START, and START runs UNDER CB2_Overworld — a blind cascade whose
        # Bs got eaten can leave it up with every cb2 gate reading 'world fine'. gTasks is
        # the readback: while Task_StartMenuHandleInput is alive, keep B'ing (bounded).
        for _ in range(4):
            if not ram.start_menu_open(self.b):
                break
            self._press("B", settle=24)

    def _confirm_world_back(self, label, max_presses=10, extra=6):
        """WORLD-BACK POSTCONDITION (2026-08-05, the Mt. Ember bag wedge): _b_cascade is
        BLIND — on a just-attached/long-running core its Bs can be EATEN wholesale, leaving
        the bag menu open UNDER a 'VERIFIED' return. A leaked menu PAUSES the world (movement
        eaten, NPC coords frozen), so every walker upstream reads phantom 'step blocked'
        forever — the live (15,33)->(15,34) boulder-approach spin. Ground truth is
        gMain.callback2 (ram.battle_cb2_dead — the exact read the roam watchdog's scene gate
        trusts): press B, settle, RE-READ, bounded; still open -> log LOUD and hammer a few
        more with long settles. NEVER returns with the menu knowingly open and unsaid.
        Returns True iff the world callback is confirmed back. A cb2 READ error fails OPEN
        (an unreadable byte must not convert a verified item use into a failure; the
        campaign's pixel sweep + watchdog own that backstop)."""
        def _world():
            try:
                # cb2 back in the world AND no START menu owning input — START runs UNDER
                # CB2_Overworld (the Mt. Ember EXIT-cursor wedge), so cb2 alone can lie.
                return ram.battle_cb2_dead(self.b) and not ram.start_menu_open(self.b)
            except Exception:
                return None
        w = _world()
        if w is None:
            return True                                   # unreadable cb2 -> old blind semantics
        for _ in range(max_presses):
            if _world():
                break
            self._press("B", settle=24)
        if not _world():
            self.log(f"   [{label}] !! menu stack STILL open after {max_presses} B's "
                     f"(callback2 non-overworld) — hammering {extra} more, long settles (LOUD)")
            for _ in range(extra):
                self._press("B", settle=60)
                if _world():
                    break
        if _world() is False:
            return False
        # two safety Bs: the START menu runs UNDER the overworld callback, so cb2 alone can
        # read 'world' with it still up — B on a clean overworld is a no-op, never a confirm
        self._press("B", settle=16)
        self._press("B", settle=16)
        return True

    def use_field_move(self, mon_slot, verify, label="field-move", max_seconds=60,
                       drain_frames=90, fixed_row=None):
        """Use a FIELD MOVE from the overworld party menu: START -> POKEMON -> `mon_slot` ->
        the mon's submenu (field moves list FIRST, above SUMMARY/SWITCH/...) -> A; `verify()`
        (RAM truth — e.g. FLAG_SYS_FLASH_ACTIVE for Flash) decides success. Attempt k selects
        submenu row k (no cursor address known for this submenu, so rows are blind — but each
        attempt reopens the menu fresh, making the count deterministic). `fixed_row` pins the
        SAME submenu row every attempt (a mon with exactly one field move lists it at row 0 —
        the row scan only exists for multi-field-move mons). `drain_frames` sizes the animation
        wait BEFORE any B is pressed: Flash verifies instantly (90 is plenty), but a WARP like
        Teleport fades + relocates over several real seconds — 2026-07-31 live, the 90-frame
        drain expired mid-fade and the flow read a fired Teleport as 'did not verify'. Returns
        'used' | 'failed'; fail-safe B-cascade back to the overworld either way."""
        t0 = time.time()
        self.b.set_input_owner("agent")
        for attempt in range(3):
            if time.time() - t0 > max_seconds:
                break
            self._b_cascade(6)                                   # clean slate
            if verify():
                # the PREVIOUS attempt's move landed while we were backing out (a slow warp
                # can complete during the cascade) — count it, don't re-fire
                self.log(f"   [{label}] VERIFIED late (landed during backout of attempt {attempt - 1})")
                return "used"
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
            row = attempt if fixed_row is None else fixed_row
            for _ in range(row):
                self._press("DOWN", settle=14)
            self._press("A", settle=40)                          # fire the field move
            for _ in range(drain_frames):                        # drain the animation (NO input)
                if verify():
                    break
                self.b.run_frame(); self.c.render()
            for _ in range(4):                                   # a held "used FLASH!" box
                if verify():
                    break
                self._press("B", settle=20)
            if verify():
                self._b_cascade()
                self.log(f"   [{label}] VERIFIED used (attempt {attempt}, submenu row {row})")
                return "used"
            self.log(f"   [{label}] attempt {attempt} (row {row}) did not verify — backing out")
        if verify():
            self.log(f"   [{label}] VERIFIED late (landed after the final backout)")
            return "used"
        self._b_cascade()
        self.log(f"   [{label}] !! FAILED — field move never verified (LOUD)")
        return "failed"

    def field_cure(self, item_id, mon_slot, max_seconds=75):
        """OVERWORLD status cure (2026-08-03: 'she's not removing poisons between battles!!').
        START -> BAG -> Items pocket -> `item_id` -> USE -> party `mon_slot` -> A, then verify
        by GROUND TRUTH ONLY: the item count DROPPED and the slot's status u32 CLEARED. The bag
        list row is walked BLIND (UP-clamp home, then counted DOWNs) — same zero-RAM-trust
        doctrine as the in-battle blind walks; the overworld cursor byte is never load-bearing.
        Returns 'cured' | 'no_item' | 'failed' (fail-safe: B-cascade restores the overworld;
        a wasted pass can never mis-report success)."""
        t0 = time.time()
        rows = [i for i, _q in items_pocket_rows(self.b)]
        if item_id not in rows:
            return "no_item"
        row = rows.index(item_id)
        if row > 9:
            self.log(f"   [cure] item {item_id} sits at bag row {row} — too deep for the blind "
                     f"walk, skipping (LOUD)")
            return "failed"
        qty0 = items_pocket_qty(self.b, item_id)
        stat_addr = ram.GPLAYER_PARTY + mon_slot * PARTY_MON_SIZE + P_STATUS
        if not (self.b.rd32(stat_addr) & 0xFF):
            return "cured"                                   # already clean — nothing to do
        self.b.set_input_owner("agent")
        self.log(f"   [cure] field cure: item {item_id} (bag row {row}) -> party slot {mon_slot}")
        # 1. START menu open-verify -> BAG (row 2). Same stale-cursor doctrine as teach().
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
            self.log("   [cure] !! START menu never opened — aborting (B out)")
            self._b_cascade(); return "failed"
        self._press("A", settle=80)                          # open the bag
        # 2. Items pocket (0). POCKET LIVENESS PROBE first (the mid-fight Teachy-TV/Helix-
        #    Fossil hover, 13:01): a frozen pocket byte reads 0 while the REAL pocket is Key
        #    Items. A healthy byte must RESPOND to a press; a mute one -> BLIND LEFT x4
        #    (the pocket strip clamps at Items).
        _p0 = self.b.rd8(BAG_POCKET)
        self._press("RIGHT", settle=20)
        _live = self.b.rd8(BAG_POCKET) != _p0
        if not _live:
            self._press("LEFT", settle=20)
            _live = self.b.rd8(BAG_POCKET) != _p0
        if _live:
            for _ in range(4):
                if self.b.rd8(BAG_POCKET) == 0:
                    break
                self._press("LEFT", settle=20)
            if self.b.rd8(BAG_POCKET) != 0:
                self.log("   [cure] !! couldn't reach the Items pocket — aborting")
                self._b_cascade(); return "failed"
        else:
            self.log("   [cure] pocket byte is MUTE (frozen RAM) — BLIND clamp LEFT x4")
            for _ in range(4):
                self._press("LEFT", settle=20)
        # 3. BLIND row walk: UP x (row+8) clamps home to row 0 from any parked position (the
        #    bag list remembers its row across opens), then DOWN x row lands the true row.
        for _ in range(row + 8):
            self._press("UP", settle=12)
        for _ in range(row):
            self._press("DOWN", settle=16)
        self._press("A", settle=50)                          # select the item -> USE/GIVE/TOSS box
        self._press("A", settle=90)                          # USE (top row default) -> party chooser
        # 4. STATE MACHINE (bounded): party teal -> nav + pick; anything else -> A-drain the
        #    cure dialogue. The loop's top status-check is the real done signal.
        party_navved = False
        for _ in range(40):
            if not (self.b.rd32(stat_addr) & 0xFF) and items_pocket_qty(self.b, item_id) < qty0:
                break
            if time.time() - t0 > max_seconds:
                break
            scr = self._classify()
            if scr == "party":
                if not party_navved:
                    if not self._party_goto(mon_slot):
                        self.log("   [cure] !! party cursor never reached the slot — B out")
                        break
                    party_navved = True
                    self._press("A", settle=90)              # pick the mon -> cure applies
                else:
                    self._press("A", settle=60)              # 'X was cured of its poisoning!' text
            elif scr == "bag" and not party_navved:
                self._press("A", settle=70)                  # USE press hadn't landed yet
            else:
                self._press("A", settle=50)                  # dialogue / transition
        self._b_cascade()
        world_back = self._confirm_world_back("cure")
        cured = not (self.b.rd32(stat_addr) & 0xFF)
        consumed = items_pocket_qty(self.b, item_id) < qty0
        if cured and consumed:
            if not world_back:
                self.log(f"   [cure] !! cure LANDED but the MENU STACK never closed — "
                         f"'menu_stuck' (LOUD; same postcondition as field_heal)")
                return "menu_stuck"
            self.log(f"   [cure] VERIFIED: slot {mon_slot} status cleared, item {item_id} consumed")
            return "cured"
        self.log(f"   [cure] !! NOT cured (status_clear={cured} consumed={consumed} "
                 f"world_back={world_back}) — failed LOUD")
        return "failed"

    def field_heal(self, item_id, mon_slot, max_seconds=75):
        """OVERWORLD HP heal (2026-08-05, the Mt. Ember climb: 'she is not healing outside of
        battle when she probably should' — grinding up Kindle Road/Summit Path with the bag
        full of potions and no Center on the mountain). START -> BAG -> Items pocket ->
        `item_id` -> USE -> party `mon_slot` -> A, then verify by GROUND TRUTH ONLY: the item
        count DROPPED and the slot's HP ROSE. Identical skeleton to field_cure (the proven
        blind-row-walk + pocket-liveness-probe rails); only the target read differs.
        Returns 'healed' | 'no_item' | 'already_full' | 'fainted' | 'failed' (fail-safe:
        B-cascade restores the overworld; a wasted pass can never mis-report success)."""
        t0 = time.time()
        rows = [i for i, _q in items_pocket_rows(self.b)]
        if item_id not in rows:
            return "no_item"
        row = rows.index(item_id)
        if row > 9:
            self.log(f"   [fieldheal] item {item_id} sits at bag row {row} — too deep for the "
                     f"blind walk, skipping (LOUD)")
            return "failed"
        qty0 = items_pocket_qty(self.b, item_id)
        base = ram.GPLAYER_PARTY + mon_slot * PARTY_MON_SIZE
        hp0, mx = self.b.rd16(base + P_HP), self.b.rd16(base + P_MAXHP)
        if mx <= 0 or hp0 <= 0:
            return "fainted"                             # the game REFUSES a potion on a corpse
        if hp0 >= mx:
            return "already_full"                        # nothing to heal — never waste the press
        self.b.set_input_owner("agent")
        self.log(f"   [fieldheal] field heal: item {item_id} (bag row {row}) -> party slot "
                 f"{mon_slot} ({hp0}/{mx} HP)")
        # 1. START menu open-verify -> BAG (row 2). Same stale-cursor doctrine as field_cure.
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
            self.log("   [fieldheal] !! START menu never opened — aborting (B out)")
            self._b_cascade(); return "failed"
        self._press("A", settle=80)                      # open the bag
        # 2. Items pocket (0) with the liveness probe; mute byte -> blind LEFT clamp.
        _p0 = self.b.rd8(BAG_POCKET)
        self._press("RIGHT", settle=20)
        _live = self.b.rd8(BAG_POCKET) != _p0
        if not _live:
            self._press("LEFT", settle=20)
            _live = self.b.rd8(BAG_POCKET) != _p0
        if _live:
            for _ in range(4):
                if self.b.rd8(BAG_POCKET) == 0:
                    break
                self._press("LEFT", settle=20)
            if self.b.rd8(BAG_POCKET) != 0:
                self.log("   [fieldheal] !! couldn't reach the Items pocket — aborting")
                self._b_cascade(); return "failed"
        else:
            self.log("   [fieldheal] pocket byte is MUTE (frozen RAM) — BLIND clamp LEFT x4")
            for _ in range(4):
                self._press("LEFT", settle=20)
        # 3. BLIND row walk (UP-clamp home, counted DOWNs) -> select -> USE -> party chooser.
        for _ in range(row + 8):
            self._press("UP", settle=12)
        for _ in range(row):
            self._press("DOWN", settle=16)
        self._press("A", settle=50)                      # select the item -> USE/GIVE/TOSS box
        self._press("A", settle=90)                      # USE (top row default) -> party chooser
        # 4. STATE MACHINE (bounded): party teal -> nav + pick; anything else -> A-drain the
        #    'restored HP' dialogue. The loop's top HP-rise check is the real done signal.
        party_navved = False
        for _ in range(40):
            if self.b.rd16(base + P_HP) > hp0 and items_pocket_qty(self.b, item_id) < qty0:
                break
            if time.time() - t0 > max_seconds:
                break
            scr = self._classify()
            if scr == "party":
                if not party_navved:
                    if not self._party_goto(mon_slot):
                        self.log("   [fieldheal] !! party cursor never reached the slot — B out")
                        break
                    party_navved = True
                    self._press("A", settle=90)          # pick the mon -> the heal applies
                else:
                    self._press("A", settle=60)          # 'X's HP was restored by N points!' text
            elif scr == "bag" and not party_navved:
                self._press("A", settle=70)              # USE press hadn't landed yet
            else:
                self._press("A", settle=50)              # dialogue / transition
        self._b_cascade()
        world_back = self._confirm_world_back("fieldheal")
        hp1 = self.b.rd16(base + P_HP)
        consumed = items_pocket_qty(self.b, item_id) < qty0
        if hp1 > hp0 and consumed:
            if not world_back:
                # HARD POSTCONDITION (2026-08-05 live wedge): the heal LANDED but the bag
                # never closed — returning 'healed' here is how the Mt. Ember session died
                # (a paused world under a green log line). The caller backs off; the
                # watchdog/strike sweeps own the leaked screen.
                self.log(f"   [fieldheal] !! heal LANDED (HP {hp0}->{hp1}, item consumed) but "
                         f"the MENU STACK never closed — 'menu_stuck' (LOUD, never a silent leak)")
                return "menu_stuck"
            self.log(f"   [fieldheal] VERIFIED: slot {mon_slot} HP {hp0} -> {hp1}/{mx}, "
                     f"item {item_id} consumed (world callback restored)")
            return "healed"
        self.log(f"   [fieldheal] !! NOT healed (hp {hp0}->{hp1} consumed={consumed} "
                 f"world_back={world_back}) — failed LOUD")
        return "failed"

    def give_item(self, item_id, mon_slot, max_seconds=75):
        """OVERWORLD hold-item give (2026-08-03, the Exp. Share equip: 'she needs exp share').
        START -> BAG -> Items pocket -> `item_id` -> GIVE (one DOWN below USE in the sub-box)
        -> party `mon_slot` -> A, then verify by GROUND TRUTH ONLY: the bag count DROPPED
        (the item now rides on the mon). Identical skeleton to field_cure — same blind row
        walk, same pocket liveness probe, same B-cascade fail-safe. If the target already
        holds something the game asks to switch; the A-drain answers YES (the old item
        returns to the bag, so the count-drop verify still reads true for `item_id`).
        Returns 'given' | 'no_item' | 'failed'."""
        t0 = time.time()
        rows = [i for i, _q in items_pocket_rows(self.b)]
        if item_id not in rows:
            return "no_item"
        row = rows.index(item_id)
        if row > 9:
            self.log(f"   [give] item {item_id} sits at bag row {row} — too deep for the blind "
                     f"walk, skipping (LOUD)")
            return "failed"
        qty0 = items_pocket_qty(self.b, item_id)
        self.b.set_input_owner("agent")
        self.log(f"   [give] hold-item give: item {item_id} (bag row {row}) -> party slot {mon_slot}")
        # 1. START menu open-verify -> BAG (row 2). Same stale-cursor doctrine as field_cure.
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
            self.log("   [give] !! START menu never opened — aborting (B out)")
            self._b_cascade(); return "failed"
        self._press("A", settle=80)                          # open the bag
        # 2. Items pocket (0) with the liveness probe; mute byte -> blind LEFT clamp.
        _p0 = self.b.rd8(BAG_POCKET)
        self._press("RIGHT", settle=20)
        _live = self.b.rd8(BAG_POCKET) != _p0
        if not _live:
            self._press("LEFT", settle=20)
            _live = self.b.rd8(BAG_POCKET) != _p0
        if _live:
            for _ in range(4):
                if self.b.rd8(BAG_POCKET) == 0:
                    break
                self._press("LEFT", settle=20)
            if self.b.rd8(BAG_POCKET) != 0:
                self.log("   [give] !! couldn't reach the Items pocket — aborting")
                self._b_cascade(); return "failed"
        else:
            self.log("   [give] pocket byte is MUTE (frozen RAM) — BLIND clamp LEFT x4")
            for _ in range(4):
                self._press("LEFT", settle=20)
        # 3. BLIND row walk (UP-clamp home, counted DOWNs), then the sub-box: USE is the top
        #    row, GIVE sits ONE DOWN — the only structural difference from field_cure.
        for _ in range(row + 8):
            self._press("UP", settle=12)
        for _ in range(row):
            self._press("DOWN", settle=16)
        self._press("A", settle=50)                          # select the item -> USE/GIVE/TOSS box
        self._press("DOWN", settle=30)                       # USE -> GIVE
        self._press("A", settle=90)                          # GIVE -> party chooser
        # 4. STATE MACHINE (bounded): party teal -> nav + pick; then A-drain the 'gave it to
        #    hold' (or switch-items) dialogue. Done signal = the bag count dropped.
        party_navved = False
        for _ in range(40):
            if items_pocket_qty(self.b, item_id) < qty0:
                break
            if time.time() - t0 > max_seconds:
                break
            scr = self._classify()
            if scr == "party":
                if not party_navved:
                    if not self._party_goto(mon_slot):
                        self.log("   [give] !! party cursor never reached the slot — B out")
                        break
                    party_navved = True
                    self._press("A", settle=90)              # pick the mon -> give applies
                else:
                    self._press("A", settle=60)              # 'X is now holding …!' / switch prompt
            elif scr == "bag" and not party_navved:
                self._press("A", settle=70)                  # GIVE press hadn't landed yet
            else:
                self._press("A", settle=50)                  # dialogue / transition
        self._b_cascade()
        given = items_pocket_qty(self.b, item_id) < qty0
        if given:
            self.log(f"   [give] VERIFIED: item {item_id} left the bag — party slot {mon_slot} holds it")
            return "given"
        self.log("   [give] !! NOT given (bag count unchanged) — failed LOUD")
        return "failed"

    def stone_evolve(self, item_id, mon_slot, want_species, max_seconds=150):
        """OVERWORLD stone evolution (2026-08-03 OP-team pass: the Eevee -> Jolteon rite).
        START -> BAG -> Items pocket -> the stone -> USE -> party `mon_slot` -> A, then WAIT OUT
        the evolution cutscene and verify by GROUND TRUTH ONLY: the slot's SPECIES flipped to
        `want_species` and the stone count dropped. Two doctrine points beyond field_cure:
          1. NEVER press B once the mon is picked — in Gen 3, B during the evolution animation
             CANCELS it (the stone would be consumed for nothing... actually FRLG refunds a
             cancelled stone-use by not consuming, but either way the rite fails). The scene is
             waited out passively (frame-running) polling the species byte; only A drains the
             'Congratulations!' text after the flip.
          2. A refusal ('It won't have any effect.') leaves qty AND species unchanged -> 'failed'
             LOUD, the caller backs off (never a retry loop on a wrong-stone/wrong-mon pairing).
        Returns 'evolved' | 'no_item' | 'failed' (fail-safe: B-cascade AFTER the verdict)."""
        t0 = time.time()
        rows = [i for i, _q in items_pocket_rows(self.b)]
        if item_id not in rows:
            return "no_item"
        row = rows.index(item_id)
        if row > 9:
            self.log(f"   [stone] item {item_id} sits at bag row {row} — too deep for the blind "
                     f"walk, skipping (LOUD)")
            return "failed"
        qty0 = items_pocket_qty(self.b, item_id)
        if st.read_party_species(self.b, mon_slot) == want_species:
            return "evolved"                                 # already done — nothing to do
        self.b.set_input_owner("agent")
        self.log(f"   [stone] stone evolve: item {item_id} (bag row {row}) -> party slot "
                 f"{mon_slot} (want species {want_species})")
        # 1-3. Identical rails to field_cure: START open-verify -> BAG (row 2) -> Items pocket
        #      (liveness-probed) -> blind row walk -> select -> USE -> party chooser.
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
            self.log("   [stone] !! START menu never opened — aborting (B out)")
            self._b_cascade(); return "failed"
        self._press("A", settle=80)
        _p0 = self.b.rd8(BAG_POCKET)
        self._press("RIGHT", settle=20)
        _live = self.b.rd8(BAG_POCKET) != _p0
        if not _live:
            self._press("LEFT", settle=20)
            _live = self.b.rd8(BAG_POCKET) != _p0
        if _live:
            for _ in range(4):
                if self.b.rd8(BAG_POCKET) == 0:
                    break
                self._press("LEFT", settle=20)
            if self.b.rd8(BAG_POCKET) != 0:
                self.log("   [stone] !! couldn't reach the Items pocket — aborting")
                self._b_cascade(); return "failed"
        else:
            self.log("   [stone] pocket byte is MUTE (frozen RAM) — BLIND clamp LEFT x4")
            for _ in range(4):
                self._press("LEFT", settle=20)
        for _ in range(row + 8):
            self._press("UP", settle=12)
        for _ in range(row):
            self._press("DOWN", settle=16)
        self._press("A", settle=50)                          # select the stone -> USE/GIVE/TOSS
        self._press("A", settle=90)                          # USE -> party chooser
        # 4. Pick the mon, then HANDS OFF: the cutscene owns the screen. Poll the species byte
        #    (the one signal that cannot lie) while frame-running; A-drain only after the flip.
        picked = False
        while time.time() - t0 < max_seconds:
            if st.read_party_species(self.b, mon_slot) == want_species:
                break
            scr = self._classify()
            if scr == "party" and not picked:
                if not self._party_goto(mon_slot):
                    self.log("   [stone] !! party cursor never reached the slot — B out")
                    self._b_cascade(); return "failed"
                picked = True
                self._press("A", settle=90)                  # pick the mon -> the rite begins
            elif not picked:
                self._press("A", settle=60)                  # USE press hadn't landed yet
            else:
                # cutscene / 'Congratulations!' text: A is safe, B is the cancel button — never B.
                # A refusal bounce ('It won't have any effect.') also lands here and drains out.
                self._press("A", settle=60)
                if items_pocket_qty(self.b, item_id) == qty0 and self._classify() == "bag":
                    self.log("   [stone] !! bounced back to the bag with the stone unspent — "
                             "the game refused this pairing (failed LOUD)")
                    self._b_cascade(); return "failed"
        self._b_cascade()
        evolved = st.read_party_species(self.b, mon_slot) == want_species
        consumed = items_pocket_qty(self.b, item_id) < qty0
        if evolved:
            self.log(f"   [stone] VERIFIED: slot {mon_slot} is now species {want_species} "
                     f"(stone consumed={consumed})")
            return "evolved"
        self.log(f"   [stone] !! NOT evolved (species={st.read_party_species(self.b, mon_slot)} "
                 f"consumed={consumed}) — failed LOUD")
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
                if not sub_seen:
                    # WRONG FLOW (2026-08-03): a party chooser BEFORE the case sub-box means the
                    # USE landed on a NON-TM bag item (heal/potion — stale cursor/pocket byte).
                    # A-ing forward from here is the 'Super Potion on a full team' loop — abort.
                    self.log("   [teach] !! party chooser opened WITHOUT the TM-case sub-box — "
                             "a non-TM item got selected (stale bag cursor). ABORT, B out (LOUD)")
                    break
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
# Field / battle-useless fillers — prefer these when a 4-move mon must forget for a bag TM.
_FORGET_FIRST = {100}                         # Teleport (Abra's only move until Kadabra)


def score_tm_recipient(dmg_moves, mon_types, tm_type, tm_power, *, plan_boost=False, is_ace=False):
    """How badly `mon` needs this damaging TM. dmg_moves = [(type, power), ...] already known.
    Returns score (>=0 worth teaching) or -1 (skip — already has comparable coverage).
    Pure logic for recon; campaign uses this to pick Abra-over-Blastoise etc."""
    tm_type = (tm_type or "").lower()
    tm_power = int(tm_power or 0)
    if tm_power <= 0 or not tm_type:
        return -1
    for mt, mp in dmg_moves:
        if (mt or "").lower() == tm_type and int(mp or 0) >= int(tm_power * 0.8):
            return -1
    score = 0
    if not dmg_moves:
        score += 1000                               # Teleport-only Abra — the dream case
    elif max(int(p or 0) for _, p in dmg_moves) < 40:
        score += 200                                # thin offense
    types = {(t or "").lower() for t in (mon_types or []) if t and t != "???"}
    if tm_type in types:
        score += 80                                 # STAB platform
    if plan_boost:
        score += 400                                # TeamPlanner teach_plan due
    if is_ace and dmg_moves:
        score -= 40                                 # prefer projects when the ace already fights
    score += tm_power
    return score


def forget_idx_for_tm(b, mon_slot):
    """Forget index for a bag-TM teach: None if room; else Teleport/0-power first, never precious."""
    moves = st.read_party_moves(b, mon_slot)
    real = [m for m in moves if m]
    if len(real) < 4 or 0 in moves:
        return None
    scored = []
    for i, m in enumerate(moves):
        if not m or m in _PRECIOUS:
            continue
        _t, power = st.move_info(b, m)
        # Teleport first, then pure status, then weakest damage
        tier = 0 if m in _FORGET_FIRST else (1 if (power or 0) <= 0 else 2)
        scored.append((tier, power or 0, i))
    if not scored:
        return 0
    scored.sort()
    return scored[0][2]


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
