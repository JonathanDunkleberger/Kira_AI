"""battle_agent.py - the reusable, TURN-GATED battle engine (the HANDS for any battle).

Reads battle state from RAM (verified offsets), decides via the type-chart policy
(no LLM - fast deterministic hands), and presses the menu ONLY on the rising edge of
the verified turn gate (GBATTLE_MY_TURN: 2 = waiting for my action, 0 = busy). NEVER
blind-mashes. Emits NEUTRAL game-events through on_event; the bot binds
on_event=_pokemon_react -> her self -> _ok_to_self_speak, so her VOICE narrates - the
engine never speaks in character. Performance beats yield the floor (pace callback)
so her line lands before the hands press on. Reusable for every trainer/gym -> E4.

Input is owner-attributed ('agent'): the single Bridge owner. Any non-agent press is
dropped + logged - no masher/timer can inject input mid-turn.
"""
import time

import firered_ram as ram
import pokemon_state as st
import pokemon_policy as pol

HOLD = 8


def _hp_frac(mon):
    return (mon["hp"] / mon["maxhp"]) if mon and mon["maxhp"] else 1.0


class BattleAgent:
    def __init__(self, bridge, on_event=None, render=None, hold_frames=HOLD,
                 pace=None, owner="agent", log=print):
        self.b = bridge
        self.on_event = on_event or (lambda s, **k: print(f"   [EVENT] {s}"))
        self.render = render or (lambda: None)
        self.hold = hold_frames
        self.pace = pace                 # optional: called at a beat to yield to her voice
        self.owner = owner
        self.log = log
        self.b.set_input_owner(owner)    # single deliberate owner; phantoms dropped+logged
        self._prev = None
        self._started = False
        self._enemy_fainted = False
        self._we_fainted = False
        self._no_progress = 0          # consecutive action-menu visits with no battle change
        self._last_progress = None
        # menu-agnostic recovery (the live party-submenu trap): global stall watchdog
        self._recovery_attempts = 0
        self._last_global = None
        self._stale = 0
        self._acted_once = False       # have we landed/attempted a move yet? (the battle
                                       # intro+settle legitimately makes NO hp change - the
                                       # global-stall watchdog must not fire during it)

    # ── input (owner-attributed) ───────────────────────────────────────────────
    def _tap(self, key):
        self.b.press(key, self.hold, self.hold, self.render, owner=self.owner)

    def _wait(self, frames):
        for _ in range(frames):
            self.b.run_frame(); self.render()

    def _is_trainer_battle(self):
        """BATTLE_TYPE_TRAINER (0x08). Valid in-battle. Wild = can flee, trainer = can't."""
        return bool(self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08)

    # ── events + performance beats ─────────────────────────────────────────────
    def emit(self, summary, beat=False):
        """NEUTRAL game-event -> her self. beat=True is a PERFORMANCE moment: yield the
        floor so her voice lands before the hands advance (brisk on non-beats)."""
        self.on_event(summary)
        if beat and self.pace:
            self.pace(summary)

    # ── PRESS-UNTIL-SETTLED core (rebuilt 2026-06-23) ──────────────────────────
    # The lesson of the cursor-desync saga: GBATTLE_PHASE is a free-running FRAME COUNTER,
    # not a phase register, and the action cursor only obeys the D-pad once the menu is
    # genuinely SETTLED - acting mid-transition gets every press eaten. So we never race
    # the emulator to read state at the right frame. We advance until the game is plainly
    # WAITING for input (RAM static), THEN navigate the cursor with eaten-press tolerance
    # and VERIFY the move actually registered (HP moved). Never a blind A/B that could open
    # the wrong submenu or select RUN (flee).
    def _bstate(self):
        s = st.read_battle(self.b)
        return (s["enemy"]["hp"], s["ours"]["hp"]) if s else None

    def _settle(self, need=10, timeout=900):
        """Advance frames (no input) until the battle is WAITING for input: enemy+our HP
        hold steady for `need` consecutive frames. Narrates HP diffs en route so her voice
        stays live. Returns when settled or the battle ends/timeout."""
        last, stable = None, 0
        for _ in range(timeout):
            if not st.in_battle(self.b):
                return
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur)
                self._prev = cur
            key = (cur["enemy"]["hp"], cur["ours"]["hp"]) if cur else None
            stable = stable + 1 if key == last else 0
            last = key
            if stable >= need:
                return
            self.b.run_frame(); self.render()

    def _nav_to(self, idx):
        """engage + position the move-list cursor on slot idx. A directional press is
        required before the confirm-A registers (the eaten-press quirk); for slot 0 we
        wiggle DOWN/UP to engage while netting slot 0."""
        if idx == 1:
            self._tap("RIGHT")
        elif idx == 2:
            self._tap("DOWN")
        elif idx == 3:
            self._tap("RIGHT"); self._tap("DOWN")
        else:
            self._tap("DOWN"); self._tap("UP")

    def _swap_battle_moves(self, i, j):
        """Swap our ACTIVE mon's move i<->j (ID + current PP) in the BATTLE COPY gBattleMons[0]
        ONLY - never the party/save. Proven 2026-06-24 over 6 passes: this emulator's FIGHT move
        list can ONLY confirm slot 0 (the cursor never navigates), so to use a non-slot-0 move we
        battle-copy it into slot 0, let the menu confirm slot 0, then swap back after it fires -
        the PP decrements on slot 0 (= the chosen move) and the swap-back returns it (with the
        decremented PP) to its real slot, leaving the move set + PP correct for later turns and
        the battle-end party writeback. The move executes legitimately: real PP, damage, type."""
        base = ram.GBATTLE_MONS
        mw, bw = self.b.core.memory.u16.raw_write, self.b.core.memory.u8.raw_write
        mi, mj = self.b.rd16(base + st.F_MOVES + i * 2), self.b.rd16(base + st.F_MOVES + j * 2)
        mw(base + st.F_MOVES + i * 2, mj); mw(base + st.F_MOVES + j * 2, mi)
        pi, pj = self.b.rd8(base + st.F_PP + i), self.b.rd8(base + st.F_PP + j)
        bw(base + st.F_PP + i, pj); bw(base + st.F_PP + j, pi)

    def _goto_fight(self, tries=10):
        """READ the action cursor and walk it to FIGHT (top-left). Eaten-press tolerant
        (a wasted press just costs a try; a settled menu accepts them). Returns True only
        when the cursor is confirmed on FIGHT (0); False if the cursor isn't a valid
        action cell (we're not at the action menu)."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_FIGHT:
                return True
            if c == ram.ACT_BAG:
                self._tap("LEFT")
            elif c in (ram.ACT_POKEMON, ram.ACT_RUN):
                self._tap("UP")
            else:
                return False                          # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_FIGHT

    def _goto_run(self, tries=10):
        """READ the action cursor and walk it to RUN (bottom-right). Eaten-press tolerant.
        Returns True only when the cursor is confirmed on RUN (3); False if not at the menu."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_RUN:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("RIGHT"); self._tap("DOWN")
            elif c == ram.ACT_BAG:
                self._tap("DOWN")
            elif c == ram.ACT_POKEMON:
                self._tap("RIGHT")
            else:
                return False                          # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_RUN

    def flee(self, max_seconds=90):
        """RETREAT: flee a WILD battle (the wounded heal-return path - fighting our way back
        through the grass is what blacks us out). Forced TRAINER battles can't be fled, so we
        WIN those via the normal engine. Selects RUN with eaten-press tolerance and verifies the
        battle actually ended (in_battle clears). Returns 'fled' / 'win' / 'loss' / 'stuck'."""
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "fled"
        self._prev = st.read_battle(self.b)
        self._reach_first_menu(t0, max_seconds)
        if self._is_trainer_battle():                 # can't flee a trainer -> WIN it
            return self.run(max_seconds=max_seconds)
        for _ in range(40):
            if not st.in_battle(self.b):
                return "fled"
            self._settle()
            if not st.in_battle(self.b):
                return "fled"
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] == 0:
                return "loss"
            if self._white_box() and self._goto_run():
                self._tap("RIGHT"); self._tap("DOWN") # engage (eaten-press; RUN stays at corner)
                self._tap("A"); self._wait(20)        # confirm RUN -> "got away safely" / retry
            else:
                self._advance_text()                  # advance the escape/"can't escape" message
        return "fled" if not st.in_battle(self.b) else "stuck"

    # ── SCREEN-based menu detection (the RAM has NO clean menu-state flag - every candidate
    # is a frame counter or a one-state false positive; diagnosed 2026-06-23). The UI is
    # battle-independent: the action menu + move list draw a WHITE panel bottom-right; a
    # text/dialogue box is BLUE there. Pixel (160,150) is white at the action menu but DARK
    # in the move list - so the three states are cleanly separable from the screen. ──
    _WHITE_PTS = ((135, 138), (200, 138), (135, 150), (190, 150), (150, 150), (175, 150))

    def _white_box(self):
        """True iff the bottom-right white menu panel is up (action menu OR move list) - i.e.
        NOT a blue text/dialogue box. The reliable 'a menu is waiting for me' signal."""
        p = self.b.frame_rgb().load()
        return sum(1 for x, y in self._WHITE_PTS if min(p[x, y]) > 200) >= 4

    def _in_move_list(self):
        """True iff the FIGHT move list is open (white panel up AND the action-menu marker
        pixel (160,150) is dark - it is white at the action menu, dark over the move names)."""
        p = self.b.frame_rgb().load()
        if sum(1 for x, y in self._WHITE_PTS if min(p[x, y]) > 200) < 4:
            return False
        return min(p[160, 150]) < 100

    def _select_and_verify(self, state):
        """Called ONLY when the white action-menu panel is up (screen-gated). Put the cursor
        on FIGHT, then ENGAGE the menu before the committing A.

        THE EATEN-FIRST-PRESS (diagnosed 2026-06-23, the last desync layer): the first input
        after the menu draws is dropped, so an A on FIGHT was silently eaten and no move ever
        opened (battle hung at the menu). We defeat it with a directional that CANNOT move the
        cursor off FIGHT - FIGHT is the top-left cell, so LEFT/UP are absorbed at the boundary
        (cursor stays 0) yet still consume the eaten press. Then A opens the move list, we pick
        the move, confirm, and VERIFY by an HP change (the only ground truth). We never press B
        at the action menu (that flees a wild battle) and never blind-press off FIGHT (that
        opens the POKEMON submenu)."""
        if not self._goto_fight():
            return "stuck"                            # cursor not on FIGHT -> don't risk a press
        ours, enemy = state["ours"], state["enemy"]
        idx, desc, low = pol.choose_move(ours["moves"], enemy["types"], _hp_frac(ours))
        eff = pol.effectiveness(ours["moves"][idx].get("type", "normal"),
                                enemy["types"]) if 0 <= idx < len(ours["moves"]) else 1.0
        self.log(f"   [engine] action menu: {desc} (eff x{eff:g}) vs "
                 f"{st.SPECIES_NAME.get(enemy['species'], '?')} {enemy['hp']}/{enemy['maxhp']}")
        # SWAP FIRST (diagnosed 2026-06-24, the "Weedle stuck at 2/26" wedge): the move-list open
        # is timing-flaky, so we must drive it with retries - but a confirm-A can land on slot 0
        # at ANY point in that retry. If slot 0 still held the dry Tackle, that A fired a 0-PP move
        # ("There's no PP left for this move!") and the turn stalled forever. So we battle-copy the
        # chosen move into slot 0 BEFORE any press: now every confirm path fires the INTENDED move
        # (which has PP). We swap it back only after it LANDS (the enemy-HP gate in the verify
        # loop), so the slot still holds the chosen move when the engine reads it at execution.
        swapped = idx if idx != 0 else None
        if swapped is not None:
            self._swap_battle_moves(0, swapped)
        before = self._bstate()
        enemy_hp0 = before[0] if before else None
        # OPEN THE MOVE LIST ROBUSTLY (the old engine pressed ONCE and ASSUMED it opened - on a
        # flaked open the move never fired and the enemy sat at low HP while the faster foe chipped
        # us -> 30 dead attempts -> loud abort -> blackout). Engage (LEFT/UP absorbed at the FIGHT
        # boundary, cursor stays on slot 0) then A, and RETRY until the list is pixel-confirmed
        # open. Because we already swapped, slot 0 holds the chosen move, so if an engage/A here
        # happens to confirm early it fires the INTENDED move (not a dry 0-PP Tackle).
        opened = False
        for _ in range(8):
            self._tap("LEFT"); self._tap("UP"); self._wait(6)
            if self._in_move_list():
                opened = True; break
            self._tap("A"); self._wait(8)             # some frames need an A to actually open it
            if self._in_move_list():
                opened = True; break
        if not opened:
            return "stuck"                            # never opened -> clean retry (re-settle)
        self._tap("A"); self._wait(12)                # confirm slot 0 -> fires the chosen move
        self._last_desc, self._last_eff = desc, eff   # narrated when the hit actually lands
        result = None
        for _ in range(200):                          # VERIFY: the turn starts -> HP moves; if
            if not st.in_battle(self.b):              # nothing moves, report stuck -> retry clean
                result = "done"; break
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                # SWAP-BACK timing (critical, diagnosed on forest_wedge.state): the game decrements
                # the chosen move's PP and THEN re-reads slot 0 to execute. Swapping back on the PP
                # drop slots the 0-PP Tackle in before that read -> Tackle "fires" (no PP) -> enemy
                # frozen. So restore ONLY once our move has clearly LANDED - the enemy HP dropped,
                # which is strictly AFTER the execution read. (Foe-first hits change OUR hp first;
                # we wait past that for the enemy-hp effect.)
                if swapped is not None and enemy_hp0 is not None and cur["enemy"]["hp"] < enemy_hp0:
                    self._swap_battle_moves(0, swapped); swapped = None
                if (cur["enemy"]["hp"], cur["ours"]["hp"]) != before and swapped is None:
                    result = "done"; break
            self.b.run_frame(); self.render()
        if swapped is not None:                       # safety: restore order (move never landed)
            self._swap_battle_moves(0, swapped); swapped = None
        return result or "stuck"

    def _advance_text(self, force_b=False):
        """Advance battle dialogue/animation SAFELY. Diagnosed 2026-06-23: (a) mashing A
        *into* an animation (the player walk-in, a faint, the EXP bar) WEDGES the input and
        the text then never advances - so we WAIT a beat for the animation to settle first;
        (b) the wild 'X appeared!' / 'X fainted!' gates advance on B, not A - so after a clean
        A tap we also tap B, but ONLY if the white action-menu panel is NOT up (so B can never
        be read as RUN/flee). Clean discrete taps (short hold, long release) - a held/too-fast
        press reads as one input.
        force_b (2026-06-24): in the POST-FAINT drain the foe already fainted (no flee risk),
        and the TRAINER defeat/prize screen lights the white-panel pixels as a FALSE POSITIVE
        while actually needing B to advance - so force_b taps B regardless of _white_box, which
        is what lets a trainer battle exit cleanly after its last mon faints."""
        self._wait(18)
        self.b.press("A", 2, 12, self.render, owner=self.owner)
        if force_b or not self._white_box():
            self.b.press("B", 2, 12, self.render, owner=self.owner)

    def _reach_first_menu(self, t0, max_seconds):
        """Advance the battle intro (walk-in + 'X appeared!' + 'Go MON!') to the first action
        menu (white panel up), so the foe species (gBattleMons[1] is stale until the intro
        advances) reads true."""
        for _ in range(40):
            if not st.in_battle(self.b) or time.time() - t0 > max_seconds:
                return
            if self._white_box():
                return                                # action menu reached
            self._advance_text()

    # ── one battle, start to finish ────────────────────────────────────────────
    def run(self, max_seconds=120):
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "timeout"
        self._started = True
        self._prev = st.read_battle(self.b)
        self._reach_first_menu(t0, max_seconds)
        state = st.read_battle(self.b) or self._prev
        if state:
            foe = st.SPECIES_NAME.get(state["enemy"]["species"], "a wild pokemon")
            self.emit(f"a battle started against {foe}", beat=True)
            self._prev = state

        last_glob, stall = None, 0
        while time.time() - t0 < max_seconds:
            if not st.in_battle(self.b):
                return self._finish()
            # END SEQUENCE (checked FIRST, before settling): once a side has actually FAINTED
            # (a real alive->0 transition, not a stale battle-start read), the outcome is
            # decided; the rest is the victory/loss chain - faint anim -> "X fainted!" -> EXP
            # bar -> level-up -> exit. _advance_text walks it (waits out animations, A+B taps)
            # until the battle exits to overworld (in_battle -> False -> _finish). Never selects.
            if self._enemy_fainted or self._we_fainted:
                # POST-FAINT: drain the chain (faint anim -> "X fainted!" -> EXP -> level-up),
                # then DECIDE. A faint does NOT always end the battle: a TRAINER whose mon
                # faints SENDS THE NEXT ONE. So after each advance we check the enemy slot - if
                # a FRESH LIVE mon is on the field (full HP), it's a switch-in: reset the faint
                # flag and fall back into the normal fight loop. Otherwise keep draining toward
                # the exit (wild win / our loss / the trainer's LAST mon) until in_battle clears
                # -> _finish. (Before this, the engine assumed first-faint=won and never fought
                # the second mon -> trainer battles hung until timeout.)
                for _i in range(60):
                    if not st.in_battle(self.b):
                        break
                    cur = st.read_battle(self.b)
                    if cur:
                        self._emit_diffs(self._prev, cur); self._prev = cur
                    enemy = cur["enemy"] if cur else None
                    if (self._enemy_fainted and not self._we_fainted and enemy
                            and enemy["hp"] > 0 and enemy["hp"] == enemy["maxhp"]
                            and 1 <= enemy["species"] <= 411):
                        self._enemy_fainted = False        # next mon is out -> fight it
                        self._prev = cur
                        self.emit(f"the trainer sent out "
                                  f"{st.SPECIES_NAME.get(enemy['species'], 'another Pokemon')}",
                                  beat=True)
                        break
                    self._advance_text(force_b=True)      # faint -> EXP -> level-up -> defeat -> exit
                continue
            self._settle()                            # advance to a wait-point (narrates diffs)
            if not st.in_battle(self.b):
                return self._finish()
            glob = self._bstate()
            if glob != last_glob:                     # real progress -> reset the wedge guard
                last_glob, stall = glob, 0
            if self._white_box():                     # the action menu is up (white panel) ->
                state = st.read_battle(self.b)         # pick + commit a move, verify it lands
                res = self._select_and_verify(state) if state else "stuck"
                if res == "done":
                    self._acted_once = True
                    stall = 0
                else:
                    stall += 1                        # menu up but flaky -> settle re-checks, retry
            else:
                self._advance_text()                  # BLUE dialogue/animation box -> advance it
                stall += 1
            if stall >= 30:                           # genuine wedge -> loud abort, never silent
                self.log("   [engine] !! battle wedged - no progress over 30 attempts, aborting loudly")
                self.emit("okay I'm properly stuck, the menu's glitched", beat=False)
                return "stuck"
        return "timeout"

    def _finish(self):
        prev = self._prev or {}
        ours = prev.get("ours", {})
        if self._enemy_fainted or (prev.get("enemy", {}).get("hp", 1) == 0):
            self.emit("you won the battle", beat=True)
            return "win"
        if ours.get("hp", 1) == 0:
            self.emit("you lost - your Pokemon fainted", beat=True)
            return "loss"
        self.emit("the battle ended", beat=False)
        return "ended"

    def _emit_diffs(self, prev, cur):
        if not prev:
            return
        pe, ce = prev["enemy"], cur["enemy"]
        po, co = prev["ours"], cur["ours"]
        # narrate the move from the OBSERVED hit (ground truth), not per button-press,
        # so it fires exactly once per landed move - never spammy.
        if ce["hp"] < pe["hp"] and ce["hp"] > 0:
            desc = getattr(self, "_last_desc", "an attack")
            self.emit(f"used {desc}", beat=(getattr(self, "_last_eff", 1.0) >= 2))
        if ce["hp"] == 0 and pe["hp"] > 0:
            self._enemy_fainted = True
            self.emit(f"{st.SPECIES_NAME.get(ce['species'], 'the enemy')} fainted", beat=True)
        if co["hp"] == 0 and po["hp"] > 0:
            self._we_fainted = True
            self.emit("your Pokemon fainted", beat=True)
        elif co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit("you took a big hit", beat=True)
        elif co["maxhp"] and co["hp"] / co["maxhp"] < 0.25 and po["hp"] / max(po["maxhp"], 1) >= 0.25:
            self.emit("low HP - this is getting tense", beat=True)
