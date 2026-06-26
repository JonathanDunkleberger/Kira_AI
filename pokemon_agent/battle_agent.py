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

    # NOTE: the slot-0 move-swap hack (_swap_battle_moves), the action-cursor walker that read
    # the stale 0x02023FF8 latch (_goto_fight), and the engage-wiggle _nav_to were RETIRED
    # 2026-06-25 once real menu nav worked (phantom-A fix). Move selection now navigates the
    # real move list (_home_to_fight + _nav_move in _select_and_verify). flee() still uses
    # _goto_run below (proven working); a clean-nav rewrite of it is a later follow-up.

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

    # ── autonomous CATCH (real bag nav; the phantom-A bug that made this impossible was
    # fixed 2026-06-25 — see [[pokemon-battle-menu-nav-cracked]]). Flow, screenshot- and
    # control-verified (party N->N+1) on forest_battle.state: action menu opens on FIGHT each
    # turn -> RIGHT = BAG -> A opens the bag (lands on the Poké Balls pocket, cursor on the
    # ball) -> A selects -> "POKé BALL is selected. USE/CANCEL" (cursor on USE) -> A throws.
    # Then advance the catch sequence (B dismisses the "give a nickname?" Yes/No). We SETTLE
    # after the bag-open fade (acting mid-transition = eaten, the same quirk as the move list).
    def _ball_count(self):
        """Number of Poké Balls (item id 4) in the bag's balls pocket (ids are plain; qty is
        key-encrypted so we only assert PRESENCE). 0 -> can't throw."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        for i in range(16):
            iid = self.b.rd16(sb1 + 0x430 + i * 4)
            if iid == 0:
                break
            if iid == 4:
                return 1
        return 0

    def throw_ball(self, max_seconds=45):
        """Throw a Poké Ball at a WILD foe via real menu nav. Returns 'caught' (party+1),
        'broke_free' (battle continued/ended w/o catch), 'trainer' (can't catch), 'no_balls',
        or 'stuck'. Assumes a fresh/settled action menu (turn start). Control-proven party+1."""
        t0 = time.time()
        if self._is_trainer_battle():
            return "trainer"
        if self._ball_count() == 0:
            self.log("   [engine] throw_ball: no Poké Balls in the bag")
            return "no_balls"
        if not self._white_box():
            self._reach_first_menu(t0, max_seconds)
        self._settle()
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        self._tap("RIGHT")                            # FIGHT -> BAG
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # open bag
        self._wait(45)                                # wait out the bag-open fade
        self._tap("UP")                               # ensure top item (POKé BALL), not CANCEL
        self._wait(12)
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # select -> USE/CANCEL
        self._wait(18)
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # confirm USE -> throw
        self.emit("alright — throwing a Poké Ball", beat=True)
        while time.time() - t0 < max_seconds:
            if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                self.emit("gotcha — it's caught!", beat=True)
                return "caught"
            if not st.in_battle(self.b):
                return "broke_free"
            # The ball BROKE and we're back at the action menu (white panel) - the turn is over.
            # Return here WITHOUT pressing anything: B at the action menu is RUN, so mashing B would
            # FLEE (the old one-throw-then-wander bug). The caller decides whether to throw again
            # (commit to the catch) or move on. PRE-2026-06-26 this fled after every failed throw.
            if self._white_box():
                return "broke_free"
            # B-ONLY advance for the catch-sequence BLUE boxes (the "broke free!" text and the
            # post-catch "give a nickname?" Yes/No, which defaults YES — an A would open the naming
            # keyboard and wedge it). B dismisses them safely. Wait so we never mash into animation.
            self._wait(18)
            self.b.press("B", 2, 12, self.render, owner=self.owner)
        return "stuck"

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

    def _home_to_fight(self):
        """Park the action cursor on FIGHT (top-left) WITHOUT reading the stale cursor latch:
        UP then LEFT are absorbed at the top/left boundary, so from ANY of the 4 cells they net
        the top-left corner = FIGHT. Reliable now that input is clean (no phantom-A confirm)."""
        self._tap("UP"); self._tap("LEFT"); self._wait(4)

    def _nav_move(self, idx):
        """Move the move-list cursor from slot 0 (where the list opens) to slot idx in the 2x2
        grid: TL=0 TR=1 / BL=2 BR=3 (RIGHT = column, DOWN = row). Settles after so the confirm-A
        isn't eaten mid cursor-move (the slot-2 lesson)."""
        if idx == 1:
            self._tap("RIGHT")
        elif idx == 2:
            self._tap("DOWN")
        elif idx == 3:
            self._tap("RIGHT"); self._tap("DOWN")
        self._wait(14)

    def _select_and_verify(self, state):
        """Called when the white action-menu panel is up (screen-gated). REAL move-list nav
        (slot-0 swap retired 2026-06-25 after the phantom-A fix): home the cursor to FIGHT, open
        the move list, NAVIGATE to the policy-chosen move, confirm it, and VERIFY by the chosen
        move's PP dropping (the move actually executed - robust for status moves too, which the
        old HP-change check missed). She now fires the move she CHOSE (e.g. a super-effective
        Vine Whip), not a pre-swapped slot 0. We never press B at the action menu (that flees a
        wild battle); B is only used to back out of a wrongly-opened submenu."""
        ours, enemy = state["ours"], state["enemy"]
        idx, desc, low = pol.choose_move(ours["moves"], enemy["types"], _hp_frac(ours))

        def _usable(i):                                # a real move with PP
            m = ours["moves"][i]
            return m.get("id", 0) != 0 and m.get("pp", 0) > 0
        if not (0 <= idx < 4) or not _usable(idx) or idx == self._skip_idx:
            # policy pick is empty / 0-PP / the move that JUST failed: rotate to the best OTHER
            # usable move by expected DAMAGE (power x effectiveness) - prefer a real attack, not status.
            cands = [i for i in range(4) if _usable(i) and i != self._skip_idx]
            if not cands:                              # only the skip move is left -> allow it
                cands = [i for i in range(4) if _usable(i)]
            if cands:
                idx = max(cands, key=lambda i: max(ours["moves"][i].get("power", 0), 1)
                          * pol.effectiveness(ours["moves"][i].get("type", "normal"), enemy["types"]))
                desc = ours["moves"][idx].get("name", desc)
        eff = pol.effectiveness(ours["moves"][idx].get("type", "normal"),
                                enemy["types"]) if 0 <= idx < len(ours["moves"]) else 1.0
        self.log(f"   [engine] action menu: {desc} -> slot {idx} (eff x{eff:g}) vs "
                 f"{st.SPECIES_NAME.get(enemy['species'], '?')} {enemy['hp']}/{enemy['maxhp']}")
        # OPEN THE MOVE LIST ROBUSTLY: home to FIGHT, A, pixel-confirm the list opened; retry the
        # A if it was eaten (still at the white action menu); if a wrong submenu opened (bag/
        # POKEMON - NOT the white action panel) back out with B and re-home. Bounded.
        opened = False
        self._home_to_fight()
        for _ in range(8):
            if self._in_move_list():
                opened = True; break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            if self._in_move_list():
                opened = True; break
            if not self._white_box():                 # a wrong submenu opened -> back out + re-home
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
                self._home_to_fight()
        if not opened:
            return "stuck"                            # never opened -> clean retry (re-settle)
        self._nav_move(idx)                           # walk the cursor to the chosen move
        pp0 = ours["moves"][idx].get("pp", 0)
        before = self._bstate()
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        self._last_desc, self._last_eff = desc, eff   # narrated when the hit actually lands
        # VERIFY the move EXECUTED. The old fixed 220-frame window was TOO SHORT for a full trainer
        # turn (when we're slower, our hit lands AFTER the foe's move + animations) -> it timed out
        # on WORKING moves and (with benching) exiled them, losing winnable fights. Now we wait for
        # the TURN TO RESOLVE: a PP drop or ANY HP change = it fired (damage dealt or taken means a
        # move went off); battle ending = it fired (KO). Only if the turn settles back at the menu
        # with NO PP drop and NO HP change is it a true non-fire (Disable / can't-act).
        result = None
        last_hp, stable = before, 0
        for _ in range(900):
            if not st.in_battle(self.b):
                result = "done"; break                # battle ended (KO) = our move resolved
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                if cur["ours"]["moves"][idx].get("pp", 0) < pp0:      # our chosen move's PP dropped
                    result = "done"; break
                hp = (cur["enemy"]["hp"], cur["ours"]["hp"])
                if before and hp != before:           # ANY HP moved this turn -> a move executed
                    result = "done"; break
                stable = stable + 1 if hp == last_hp else 0
                last_hp = hp
                if self._white_box() and stable >= 30:   # settled back at the menu, nothing happened
                    break                                 # = the move never fired (Disabled/blocked)
            self.b.run_frame(); self.render()
        if result == "done":
            self._skip_idx = None                      # it fired -> nothing to avoid next turn
        else:
            # didn't fire (no PP drop, no HP change) = Disabled / couldn't act: avoid this slot for
            # the NEXT selection only (then it's eligible again - never permanently exiled).
            self._skip_idx = idx
            self.log(f"   [engine] move slot {idx} didn't fire (disabled / blocked) -> "
                     f"trying a different move next turn")
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

    # ── forced faint-switch (party>=2; the lead goes down mid-battle) ───────────
    # Until the phantom-A fix (a463055), an incidental A confirmed the "Choose a POKéMON" menu
    # so the switch "just happened"; with input clean it must be navigated explicitly. None of
    # the party=1 regression fixtures exercised this, so the fix exposed it. Now buildable.
    def _healthy_reserve_slot(self):
        """First party slot with current-HP > 0, or None. Party current-HP is UNencrypted at
        +0x56 in the 100-byte party struct (level is at +0x54, used elsewhere)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) > 0:
                return s
        return None

    def _force_switch(self):
        """Lead fainted with a healthy reserve -> the 'Choose a POKéMON' party menu is up.
        Navigate to the first healthy slot (cursor opens on slot 0) and confirm SEND OUT
        (the select submenu defaults to SEND OUT). Returns True once a healthy mon is active.
        Reconned flow on rf_leg0: DOWN*slot -> A (select) -> A (SEND OUT)."""
        slot = self._healthy_reserve_slot()
        if slot is None:
            return False
        for _attempt in range(6):
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0:
                return True                               # a healthy mon is active -> switched
            self._wait(18)                                # let the party menu settle
            for _ in range(slot):                         # cursor opens on slot 0 -> down to healthy
                self._tap("DOWN")
            self._wait(6)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # select mon
            self._wait(12)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # -> SEND OUT
            self._wait(20)
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0:
                return True
            self._advance_text()                          # menu not up yet -> drain a beat, retry
        cur = st.read_battle(self.b)
        return bool(cur and cur["ours"]["hp"] > 0)

    # ── one battle, start to finish ────────────────────────────────────────────
    def run(self, max_seconds=120):
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "timeout"
        self._started = True
        self._skip_idx = None              # the ONE move slot to avoid for the NEXT selection only:
        # the move that just failed to fire (Disable / slow-turn verify). A 1-turn skip rotates us
        # off a disabled move WITHOUT permanently exiling our best attack - permanent benching caused
        # the PoisonPowder-spam (a working Tackle got benched, forcing 11 turns of useless status).
        # Cleared the instant any move fires.
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
                    # OUR mon fainted but we have a healthy reserve -> this is a FORCED SWITCH,
                    # not a loss: navigate the "Choose a POKéMON" menu and send the next mon, then
                    # fall back into the normal fight loop (roster-depth survival, now explicit).
                    if (self._we_fainted and st.in_battle(self.b) and cur
                            and cur["ours"]["hp"] == 0 and self._healthy_reserve_slot() is not None):
                        if self._force_switch():
                            self._we_fainted = False
                            self._prev = st.read_battle(self.b)
                            self.emit("that one's down - sending out my next Pokemon", beat=True)
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
