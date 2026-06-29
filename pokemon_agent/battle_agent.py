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
import os
import time

import firered_ram as ram
import pokemon_state as st
import pokemon_policy as pol

HOLD = 8

# Gen-1 legendaries (FireRed national-dex ids) — a big-beat recognition in run() (Phase 2D).
_LEGENDARY_SPECIES = frozenset({144, 145, 146, 150, 151})   # Articuno, Zapdos, Moltres, Mewtwo, Mew

# ── BATCH 2 PART B: in-battle "use your items" instinct ───────────────────────────────────────────
# An active mon at/under this HP fraction WITH a heal item in the bag -> the oracle is OFFERED "use a
# potion" (capability-not-script: she chooses, but she never faints with unused heals because the option
# was never surfaced). Named/tunable.
BATTLE_CRIT_FRAC = 0.30
BAG_CURSOR = 0x0203AD04      # u8 in-battle bag LIST row cursor (recon_itemuse triangulation 2026-06-27,
#                             adjacent to GBAG_POCKET 0x0203AD02; verified to step 0->1 down the list)
_ITEMS_POCKET_OFF = 0x0310   # SaveBlock1 Items pocket (potions + status cures live here), 42 slots
# Gen-3 item ids for the in-battle instinct (CANDIDATES; the use is self-verified by the item count
# dropping, so a wrong id simply doesn't fire -> 'failed' -> keep fighting, never a wrong action).
_HEAL_ITEMS_PREF = (19, 20, 21, 22, 13)   # Full Restore, Max, Hyper, Super, Potion (strongest usable first)
_STATUS_CURE_ITEM = {"poison": 14, "burn": 15, "freeze": 16, "sleep": 17, "paralysis": 18}
_FULL_HEAL = 23

# ── ANTI-WEDGE BATTLE FLOOR (run-existential) ─────────────────────────────────────────────────────
# A turn that never RESOLVES — no PP drop, no HP change, no faint — livelocks the fight. The trigger we
# hit live: every move depleted (Sleep Powder at 0 PP). The game refuses a 0-PP move ("There's no PP
# left for this move!"), that text flicker keeps changing the screen so the cosmetic `stall` guard
# resets forever, and the enemy never gets a turn either. This floor counts UNRESOLVED turns (ONLY a
# real resolution clears it, so the flicker can't hide the wedge) and, past the threshold, ESCAPES:
# a WILD battle is FLED (self-preservation — a watchable, in-character retreat; the sibling to the
# overworld deep-wedge floor, but inside combat); a TRAINER battle (un-fleeable) is aborted LOUD. On by
# default — a frozen session is strictly worse than a flee — but disable-able / tunable. Capability-
# not-script: she still picks her move every turn; this only catches the dead-end where NO move resolves.
BATTLE_FLEE_FLOOR = os.getenv("POKEMON_BATTLE_FLEE_FLOOR", "1") == "1"
UNRESOLVED_FLEE_AT = int(os.getenv("POKEMON_UNRESOLVED_FLEE_AT", "3"))
# B-1 — IN-BATTLE MATCHUP SWITCHING (E4-critical). The matchup MATH is offline-verified, but the
# party-menu ACTUATION (cursor nav on a long-running libmgba core) is UNVERIFIED — the standing
# menu-nav lesson. So it's GATED OFF by default until a live control passes (arm POKEMON_BATTLE_SWITCH=1
# with Jonny watching). FAIL-SAFE regardless: if a switch doesn't confirm, she backs out and fights —
# never wedges. When off, she still AVOIDS ineffective moves (that path is on + verified).
BATTLE_SWITCH_ENABLED = os.getenv("POKEMON_BATTLE_SWITCH", "0") == "1"
# PARTICIPATION-XP GRIND SWITCH (Task B fix — the autonomous underlevel cure). When grinding the weak
# team, the weak mon leads (so it's "sent out" and eligible for XP) but is ONE-SHOT before it can earn
# any — so it gains nothing while the ace mops up and takes the XP (the live look-ahead proved this).
# The real-player fix: lead the weak mon, turn-1 SWITCH to the ace — the weak mon participated (gets a
# share of XP) and never takes a hit (it's benched before the enemy's turn), while the tanky ace KOs.
# `PROTECT_LEAD_GRIND` is toggled by campaign.grind_weak_members AROUND its grind battles only (off in
# normal play). FAIL-SAFE: a switch that doesn't confirm falls through to fighting (never wedges).
# DEFAULT OFF: the live look-ahead proved the in-battle party-menu actuation WEDGES the wild battle on
# this long-running core (the standing menu-nav-on-long-core risk — same reason BATTLE_SWITCH is gated).
# A wedged grind battle returns 'stuck' and blacks her out. Kept (code-complete) for when the in-battle
# switch actuation is made reliable (it's the real weak-mon-leveling cure for the E4); until then OFF, and
# the underlevel grind leans on other paths. Arm with POKEMON_GRIND_SWITCH=1 once switch nav is verified.
GRIND_SWITCH_ENABLED = os.getenv("POKEMON_GRIND_SWITCH", "0") != "0"
PROTECT_LEAD_GRIND = False                 # set True by grind_weak_members only; read per battle in run()
# party-mon STATUS1 (u32 @ +0x50 in the 100-byte struct) — the reliable party-only block (== campaign).
_P_STATUS = 0x50
_ST_SLEEP, _ST_PSN, _ST_BRN, _ST_FRZ, _ST_PAR, _ST_TOX = 0x07, 0x08, 0x10, 0x20, 0x40, 0x80


def _decode_status(s):
    if s & _ST_SLEEP:
        return "sleep"
    if s & (_ST_PSN | _ST_TOX):
        return "poison"
    if s & _ST_BRN:
        return "burn"
    if s & _ST_FRZ:
        return "freeze"
    if s & _ST_PAR:
        return "paralysis"
    return None


def _hp_frac(mon):
    return (mon["hp"] / mon["maxhp"]) if mon and mon["maxhp"] else 1.0


class BattleAgent:
    def __init__(self, bridge, on_event=None, render=None, hold_frames=HOLD,
                 pace=None, owner="agent", log=print, choose=None):
        self.b = bridge
        self.on_event = on_event or (lambda s, **k: print(f"   [EVENT] {s}"))
        self.render = render or (lambda: None)
        self.hold = hold_frames
        self.pace = pace                 # optional: called at a beat to yield to her voice
        self.owner = owner
        self.log = log
        # BATCH 2 PART B: optional SOUL ORACLE (choose(kind, options, ctx)->pick). When a mon is crit-low
        # or afflicted AND a matching item is in the bag, the in-battle loop OFFERS "use a potion/cure" to
        # her; she decides (capability-not-script). None -> the instinct is silent (pure policy battle).
        self.choose = choose
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
        self._unresolved_turns = 0     # ANTI-WEDGE FLOOR: turns that never RESOLVED (no PP drop/
                                       # HP change/faint). Cleared only by a real resolution, so the
                                       # 0-PP "no PP left!" flicker can't reset it like `stall` does.
        self._skip_streak = set()      # FIX 1: every move slot that failed to fire THIS streak — so she
                                       # rotates through her WHOLE moveset (never re-spams a dead/0-PP
                                       # move) and only flees once all are exhausted. Clears on any fire.

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
    def emit(self, summary, beat=False, tier=None):
        """NEUTRAL game-event -> her self. beat=True is a PERFORMANCE moment: yield the
        floor so her voice lands before the hands advance (brisk on non-beats). `tier` (Phase 2D)
        forwards an explicit salience tier for big beats (shiny/legendary) — the live on_event is
        voice.emit which reads it; the default/headless sinks accept **k and ignore it."""
        if tier is not None:
            self.on_event(summary, tier=tier)
        else:
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
        for _ in range(3):                            # ensure the ACTION menu, not the move list:
            if not self._in_move_list():              # _white_box can't tell them apart, so RUN nav
                break                                 # from an open move-list fires a move + never
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)  # escapes (flee 'stuck'
            self._wait(10)                            # loop). Same class as the catch bag-nav bug.
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
        """REAL number of Poké Balls (item id 4) in the bag's balls pocket. Item ids are plain;
        the QUANTITY is XOR-encrypted with the SaveBlock2 security key (FireRed: SaveBlock2+0xF20,
        low 16 bits). Decrypting it lets callers gate on the true count (e.g. throw-until-caught,
        out-of-balls handling) instead of mere presence. 0 -> can't throw."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for i in range(16):
            iid = self.b.rd16(sb1 + 0x430 + i * 4)
            if iid == 0:
                break
            if iid == 4:
                return self.b.rd16(sb1 + 0x430 + i * 4 + 2) ^ key
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
        balls_at_start = self._ball_count()           # baseline BEFORE the bag opens (throw-verify gate)
        if os.environ.get("CATCH_RECON"):             # RECON: what menu are we actually on at throw-start?
            try:
                _s = st.read_battle(self.b)
                self.log(f"      [catch-recon] throw-start: white_box={self._white_box()} "
                         f"in_move_list={self._in_move_list()} balls={self._ball_count()} "
                         f"in_battle={st.in_battle(self.b)} enemy_hp="
                         f"{_s['enemy']['hp'] if _s else '?'}/{_s['enemy']['maxhp'] if _s else '?'}")
            except Exception as _e:
                self.log(f"      [catch-recon] read err {_e}")
        # ENSURE THE ACTION MENU (not the move list): a prior weaken move (_fire_move/_weaken_hp) can
        # leave the FIGHT move-list open, and _white_box() can't tell action-menu from move-list — so
        # navigating to BAG from the move list fires a MOVE instead of opening the bag (no ball thrown
        # -> the catch spins, never consuming a ball). Back out to the action menu first.
        for _ in range(3):
            if not self._in_move_list():
                break
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        # SETTLE ONTO THE ACTION MENU before opening the bag: after a broke-free the foe's turn / poison
        # text can still be up, and opening the bag mid-text EATS the throw (the flaky 2nd-throw 'stuck').
        # menu_up==1 (+ white panel) is the reliable 'action menu is waiting' signal. Advance blue text;
        # back out of a stray move list; bounded so a genuinely wedged box still falls through (and the
        # pocket-nav/throw-verify below aborts loudly rather than silently spinning).
        for _ in range(30):
            if self._white_box() and self.b.rd8(ram.GBATTLE_MENU_UP) == 1:
                break
            if self._white_box():                     # white panel but not the action menu -> move list
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            else:
                self._advance_text()                  # blue dialogue/animation box -> advance it
        # OPEN THE BAG and NAVIGATE TO THE POKé BALLS POCKET (root-caused 2026-06-27 — the long-standing
        # "141 dead throws / ball count never decrements" bug). The in-battle bag opens on the LAST-VIEWED
        # pocket, NOT always Poké Balls: on Route 3 it opens on the (empty) ITEMS pocket, so the old blind
        # UP+A+A selected CANCEL and threw NOTHING. FRLG pocket layout is FIXED (Items=0, Key Items=1,
        # Poké Balls=2, ...), so we STEER the LIVE pocket index (ram.GBAG_POCKET) to the balls pocket
        # WITHOUT pressing A on any other pocket (A on an empty pocket's CANCEL poisons pocket-switching).
        # Being ON pocket 2 with the cursor at the top IS a Poké Ball (we early-returned 'no_balls' if the
        # count were 0), so we don't trust the STALE gSpecialVar_ItemId — we press A and VERIFY a ball
        # actually LEFT (count dropped vs throw-start / caught / battle ended), retrying an eaten select/
        # confirm. Selecting may itself throw (no USE prompt) or need one more A; re-checking before each
        # press never double-throws. Control-proven: Route 3 fail-state pocket 0->1->2, balls 5->1, caught.
        def _thrown():
            return (self._ball_count() < balls_at_start or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0
                    or not st.in_battle(self.b))
        # OPEN THE BAG and VERIFY it actually opened before trusting the pocket var: if the open-A is
        # eaten (we're not fully settled) we stay at the action menu and ram.GBAG_POCKET reads STALE
        # (e.g. 2 from a prior throw) — a false "on balls pocket" that then mashes A into the move list.
        # The bag being open == the white action-menu panel is GONE (_white_box False); retry the open.
        opened = False
        for _ in range(4):
            self._home_to_fight()                     # FIGHT is cursor home -> RIGHT = BAG (re-home each retry)
            self._tap("RIGHT")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # open bag
            self._wait(50)                            # wait OUT the bag-open fade (acting on the fade = eaten)
            if not self._white_box():                 # left the action-menu panel -> the bag is open
                opened = True; break
        if not opened:
            self.log("   [engine] !! throw_ball: bag would not open (open-A eaten) — aborting LOUDLY")
            return "stuck"
        on_balls_pocket = False
        for _ in range(8):                            # steer the live pocket index toward Poké Balls
            if self.b.rd8(ram.GBAG_POCKET) == ram.POCKET_POKE_BALLS:
                on_balls_pocket = True
                break
            self._tap("RIGHT" if self.b.rd8(ram.GBAG_POCKET) < ram.POCKET_POKE_BALLS else "LEFT")
            self._wait(12)
        if os.environ.get("CATCH_RECON"):
            self.log(f"      [catch-recon] bag nav: pocket={self.b.rd8(ram.GBAG_POCKET)} "
                     f"on_balls_pocket={on_balls_pocket} item={self.b.rd16(ram.GSPECIALVAR_ITEMID)}")
        if not on_balls_pocket:
            self.log("   [engine] !! throw_ball: couldn't reach the Poké Balls pocket — aborting LOUDLY")
            for _ in range(4):                        # leave the menu clean for the caller
                if self._white_box():
                    break
                self.b.press("B", 2, 12, self.render, owner=self.owner); self._wait(8)
            return "stuck"
        self._tap("UP"); self._wait(8)                # top of the balls pocket = the ball
        # SELECT + THROW, then STOP the instant a ball leaves. Press A (select -> USE/throw); the throw
        # removes the ball from the bag IMMEDIATELY (count drops) — so after each A we POLL for the throw
        # to register and break the MOMENT it does. This is critical: if we kept mashing A, the extra
        # press lands on the post-catch "give a nickname? [YES]" prompt and opens the naming KEYBOARD,
        # wedging the next throw (the forest 2nd-throw bug). Selecting may itself throw (1 A) or need a
        # USE confirm (2 A); the per-A poll handles both and retries an eaten press. LOUD abort, no spin.
        for _ in range(4):
            if _thrown():
                break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            for _ in range(40):                       # watch for the ball to leave BEFORE pressing again
                if _thrown():
                    break
                self.b.run_frame(); self.render()
        if not _thrown():
            self.log("   [engine] !! throw_ball: ball selected but no throw consumed a ball — aborting LOUDLY")
            return "stuck"
        self.emit("alright — throwing a Poké Ball", beat=True)
        while time.time() - t0 < max_seconds:
            if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                self.emit("gotcha — it's caught!", beat=True)
                # The party grew (catch banked) BUT the "give a nickname? [YES/NO]" prompt is still up
                # (FRLG adds the mon, then asks). Decline with B (never leave it for the next op to A into
                # -> naming keyboard) and let the battle EXIT to the overworld before returning.
                for _ in range(20):
                    if not st.in_battle(self.b):
                        break
                    self._wait(10); self.b.press("B", 2, 12, self.render, owner=self.owner)
                return "caught"
            if not st.in_battle(self.b):
                # Battle ended. On a CATCH the "give a nickname? [YES/NO]" prompt HOLDS the party-increment
                # until dismissed — and its cursor defaults to YES, so an A opens the naming keyboard. The
                # old 40 BLANK frames here neither dismissed it nor saw the party tick, so a real catch
                # returned 'broke_free' and LEFT the prompt up -> the next throw's A typed into the keyboard
                # (the forest 2nd-throw wedge). Press B (decline) while watching for the party to grow (the
                # unfakeable catch signal): finalizes the catch AND never leaves a prompt dangling.
                for _ in range(20):
                    if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                        break
                    self._wait(10)
                    self.b.press("B", 2, 12, self.render, owner=self.owner)
                if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                    self.emit("gotcha — it's caught!", beat=True)
                    return "caught"
                return "broke_free"
            # The ball BROKE and we're back at the ACTION MENU - the turn is over. Gate this on the
            # TRUE action menu (white panel AND menu_up==1), NOT white_box alone: a white frame flashes
            # DURING the catch sequence, and returning then leaves the "give a nickname?" prompt up for
            # the caller to A into (-> naming keyboard wedge). menu_up==1 only holds at the real action
            # menu, which a CATCH never reaches (the battle ends), so this returns broke_free only on a
            # genuine break. Return WITHOUT pressing: B here = RUN (would flee). The caller re-throws.
            if self._white_box() and self.b.rd8(ram.GBATTLE_MENU_UP) == 1:
                return "broke_free"
            # B-ONLY advance for the catch-sequence BLUE boxes (the "broke free!" text and the
            # post-catch "give a nickname?" Yes/No, which defaults YES — an A would open the naming
            # keyboard and wedge it). B dismisses them safely. Wait so we never mash into animation.
            self._wait(18)
            self.b.press("B", 2, 12, self.render, owner=self.owner)
        return "stuck"

    # ── autonomous CATCH FLOW (mirrors the proven live play: weaken/status, then commit to throws)
    _SLEEP_MOVES = {79, 147, 95, 47, 142, 1}        # Sleep Powder, Spore, Hypnosis, Sing, Lovely Kiss
    _STATUS_MOVES = _SLEEP_MOVES | {77, 78, 86}     # + PoisonPowder, StunSpore, ThunderWave
    CATCH_WEAKEN_CEIL = 0.85   # if she CAN'T weaken (depleted PP) AND the foe is still above this HP
    #                            fraction, don't dump balls into a low-odds full-HP catch — flee + heal.

    def _can_weaken(self, state):
        """True iff she has a move that can actually SOFTEN the foe — a usable status move OR a usable
        damaging move (PP>0). False = fully depleted: can't sleep it, can't chip it. The catch flow uses
        this to refuse ball-dumping a near-full-HP foe she has no way to weaken (the live ball-burn)."""
        moves = state["ours"]["moves"]
        has_status = any(m.get("id", 0) in self._STATUS_MOVES and m.get("pp", 0) > 0 for m in moves)
        has_damage = any(m.get("id", 0) and m.get("pp", 0) > 0 and m.get("id", 0) not in self._STATUS_MOVES
                         for m in moves)
        return has_status or has_damage

    def _catch_weaken_move(self, state):
        """Slot index of a move to SOFTEN the wild foe before throwing - prefer a SLEEP move (asleep
        = x2 catch rate in Gen 3 AND it stops the foe attacking us), else another status. Returns
        None if the foe is already low (just throw) or we have no usable status move."""
        foe = state["enemy"]
        if foe["maxhp"] and foe["hp"] <= foe["maxhp"] * 0.35:
            return None                                  # weak enough already
        moves = state["ours"]["moves"]
        for pool in (self._SLEEP_MOVES, self._STATUS_MOVES):
            for i, m in enumerate(moves):
                if m.get("id", 0) in pool and m.get("pp", 0) > 0:
                    return i
        return None

    def _weaken_hp(self, target_frac=0.40, max_hits=4):
        """Chip the wild foe's HP into the catchable band so a HANDFUL of balls suffices (a status
        alone leaves it near full HP -> 5 balls broke free). Fires the LOWEST-base-power damaging move
        one hit at a time, re-reading HP, and STOPS once HP <= target_frac (faint-guard: never swing
        at an already-low foe; one-at-a-time re-check avoids overkill). Best-effort — a stray faint
        just means catch_pokemon returns 'fainted' and the wander finds another wild."""
        for _ in range(max_hits):
            state = st.read_battle(self.b)
            if not state or not st.in_battle(self.b):
                return
            foe = state["enemy"]
            if not foe.get("maxhp") or foe["hp"] <= 0:
                return
            if foe["hp"] / foe["maxhp"] <= target_frac:
                return                                  # already in the catchable band
            cand = [(i, m.get("id", 0)) for i, m in enumerate(state["ours"]["moves"])
                    if m.get("id", 0) and m.get("pp", 0) > 0 and m.get("id", 0) not in self._STATUS_MOVES]
            if not cand:
                return                                  # no damaging move with PP -> just throw
            cand.sort(key=lambda im: st.move_info(self.b, im[1])[1] or 0)   # gentlest (lowest power)
            self._fire_move(cand[0][0])

    def _fire_move(self, idx):
        """Open the move list, navigate to slot idx, fire it + verify it executed (PP drop / HP
        change / battle end). Separate from _select_and_verify (which policy-PICKS a move) so the
        proven fight path stays untouched; used by catch_pokemon to fire a chosen weaken move."""
        opened = False
        self._home_to_fight()
        for _ in range(8):
            if self._in_move_list():
                opened = True; break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            if self._in_move_list():
                opened = True; break
            if not self._white_box():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
                self._home_to_fight()
        if not opened:
            return "stuck"
        self._nav_move(idx)
        state = st.read_battle(self.b)
        pp0 = state["ours"]["moves"][idx].get("pp", 0) if state else 0
        before = self._bstate()
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        for _ in range(900):
            if not st.in_battle(self.b):
                return "done"
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                if cur["ours"]["moves"][idx].get("pp", 0) < pp0:
                    return "done"
                if before and (cur["enemy"]["hp"], cur["ours"]["hp"]) != before:
                    return "done"
                if self._white_box():
                    return "done"
            self.b.run_frame(); self.render()
        return "stuck"

    def catch_pokemon(self, max_seconds=150, weaken=True):
        """Catch the WILD foe (the proven live flow, automated): optionally WEAKEN/STATUS it once to
        boost the catch rate + stop it attacking, then THROW Poké Balls until caught. COMMITS - it
        re-throws after a break instead of abandoning after one ball (the live Ekans flow). Returns
        'caught' | 'no_balls' | 'trainer' | 'fled' | 'fainted' | 'stuck'. Gen-3 trainer mons can't
        be caught (returns 'trainer')."""
        t0 = time.time()
        if self._is_trainer_battle():
            return "trainer"
        self._started = True
        self._skip_streak = set()
        self._reach_first_menu(t0, max_seconds)
        self._prev = st.read_battle(self.b)
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        softened = False

        def _ended():
            """Battle ended: settle, then a party+1 means we CAUGHT it (the 'Gotcha!' can end the
            battle a beat before the party count ticks - don't mislabel a real catch as 'fled')."""
            for _ in range(40):
                self.b.run_frame(); self.render()
            return "caught" if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0 else "fled"

        while time.time() - t0 < max_seconds:
            if not st.in_battle(self.b):
                return _ended()
            if self._ball_count() <= 0:
                self.emit("I'm out of Poké Balls - I'll come back for this one", beat=True)
                return "no_balls"
            self._settle()
            if not st.in_battle(self.b):
                return _ended()
            if not self._white_box():
                # B-ONLY advance in the catch loop: a blue box here is catch-sequence text OR the
                # post-catch "give a nickname? [YES/NO]" prompt. _advance_text presses A first, which on
                # that prompt selects YES and opens the naming keyboard (spins forever). B safely advances
                # the text AND declines the nickname. (B is unsafe only at the action menu = RUN, which is
                # white_box and excluded here.)
                self._wait(18); self.b.press("B", 2, 12, self.render, owner=self.owner); continue
            state = st.read_battle(self.b)
            if state and state["enemy"]["hp"] <= 0:
                return "fainted"                         # we KO'd it - can't catch a fainted foe
            if weaken and not softened and state is not None:
                # PHASE 4 GUARD: if she can't weaken AT ALL (no status + no damaging move with PP —
                # depleted) and the foe is still near full HP, DON'T throw — a full-HP catch is low-odds
                # and that's exactly how she burned her whole ball supply tonight. Flee (preserve the
                # balls) and surface that she needs to restore PP (a Center tops up PP). Roam then heals.
                if not self._can_weaken(state) and state["enemy"].get("maxhp") \
                        and state["enemy"]["hp"] > state["enemy"]["maxhp"] * self.CATCH_WEAKEN_CEIL:
                    self.emit("I can't even dent it — I'm out of PP to weaken it, and I'm not burning my "
                              "Poké Balls on a full-health throw. Backing out to restore my moves first.",
                              beat=True, tier=2)
                    self.log("   [engine] catch: CAN'T WEAKEN (depleted PP) + foe near full HP -> fleeing "
                             "to preserve balls (not ball-dumping)")
                    self.flee(max_seconds=45)
                    return "cant_weaken"
                wi = self._catch_weaken_move(state)
                if wi is not None:
                    self.emit("let me wear it down first", beat=True)
                    self._fire_move(wi)
                self._weaken_hp()                    # chip HP into the catchable band (faint-guarded)
                softened = True
                continue
            res = self.throw_ball(max_seconds=max(20, int(max_seconds - (time.time() - t0))))
            if res in ("caught", "no_balls", "trainer"):
                return res
            # 'broke_free' / 'stuck' -> the foe took its turn; loop and throw again (commit)
        return "stuck"

    # ── BATCH 2 PART B: USE A HEAL / CURE IN BATTLE (live-reconned recon_itemuse.py 2026-06-27) ────────
    # Flow proven on a live wild battle: settle to the ACTION menu (pixel-gated) -> FIGHT home -> RIGHT
    # (=BAG) -> A opens the bag -> steer GBAG_POCKET to the Items pocket (0) -> DOWN to the item's row
    # (the pocket list shows the bag array IN ORDER; nav by the BAG_CURSOR readback) -> A walks
    # select->USE->target(default lead)->apply. GROUND-TRUTH success = the item COUNT drops (HP rise is
    # incidental). FAIL-SAFE (Jonny's mandate): every step is bounded + readback-gated; on ANY failure we
    # B-out to a clean menu and return 'failed' so the battle loop just KEEPS FIGHTING — never a wedge,
    # never a wrong action (the apply A-loop only runs once we've CONFIRMED pocket==0 AND cursor==row).
    def _items_pocket(self):
        """[(item_id, qty), ...] in the Items pocket in display order (qty XOR'd with the low-16 key)."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        out = []
        for s in range(42):
            slot = sb1 + _ITEMS_POCKET_OFF + s * 4
            iid = self.b.rd16(slot)
            if iid == 0:
                break
            out.append((iid, self.b.rd16(slot + 2) ^ key))
        return out

    def _items_count(self, item_id):
        return next((q for i, q in self._items_pocket() if i == item_id), 0)

    def _lead_status(self):
        """Status NAME of party slot 0 (STATUS1 @ +0x50, the reliable party-only block) or None. In
        singles the active mon is the lead unless a mid-battle switch happened (then the cure offer just
        won't fire — fail-safe, never a wrong cure)."""
        return _decode_status(self.b.rd32(ram.GPLAYER_PARTY + _P_STATUS))

    def _settle_action_menu(self, tries=30):
        """Reach the real ACTION menu using the PIXEL signals (menu_up RAM is stale on this core): action
        menu = white panel AND NOT the move-list pixel. Back out of the move list with B; advance text."""
        self._settle()
        for _ in range(tries):
            if self._white_box() and not self._in_move_list():
                return True
            if self._in_move_list():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            elif not self._white_box():
                self._advance_text()
            else:
                return True
        return self._white_box() and not self._in_move_list()

    def _goto_bag(self, tries=10):
        """Walk the action cursor to BAG (top-right, ACT_BAG=1) by READBACK, not a blind RIGHT — the
        live look-ahead proved a blind _tap('RIGHT') gets EATEN on this long core, so A landed on FIGHT
        and the bag never opened ('eaten RIGHT'), and she could never heal through a fight. Mirror of
        _goto_pokemon: grid FIGHT(0,TL) BAG(1,TR) / POKEMON(2,BL) RUN(3,BR). Confirms ACT_BAG before A."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_BAG:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("RIGHT")
            elif c == ram.ACT_RUN:
                self._tap("UP")
            elif c == ram.ACT_POKEMON:
                self._tap("UP")                           # -> FIGHT, then RIGHT next iter -> BAG
            else:
                return False                              # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_BAG

    def _open_bag(self, tries=4):
        """From the ACTION menu: cursor->BAG (verified by readback) -> A. The bag is open iff the white
        action panel is GONE (a blue description box). If A didn't open it (white panel stays), B out +
        retry. The readback nav (vs a blind RIGHT) is the fix for the long-core 'eaten RIGHT' that left
        her unable to use a Potion mid-fight — never fires a move (that needs a 2nd A in the move list)."""
        for _ in range(tries):
            if not self._goto_bag():
                self._settle_action_menu(); continue
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(50)
            if not self._white_box():
                return True
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            self._settle_action_menu()
        return False

    def _exit_bag(self):
        """Best-effort B back to a clean menu/battle so a FAILED item-use never leaves a menu dangling
        for the battle loop to A into (which could fire a stray move). Bounded."""
        for _ in range(6):
            if self._white_box() or not st.in_battle(self.b):
                return
            self.b.press("B", 2, 12, self.render, owner=self.owner); self._wait(10)

    def use_item_in_battle(self, item_id, max_seconds=30):
        """Use one `item_id` from the Items pocket on the active (lead) mon. Returns 'used' (count
        dropped) | 'no_item' | 'failed'. FAIL-SAFE: anything but 'used' leaves the battle fightable."""
        ids = [i for i, _ in self._items_pocket()]
        if item_id not in ids:
            self.log(f"   [engine] use_item: item {item_id} NOT in pocket {ids[:8]} — no_item (LOUD)")
            return "no_item"
        row = ids.index(item_id)
        cnt0 = self._items_count(item_id)
        if not self._settle_action_menu():
            self.log("   [engine] use_item: couldn't reach the action menu — keep fighting (LOUD)")
            return "failed"
        if not self._open_bag():
            self.log("   [engine] use_item: bag wouldn't open (eaten RIGHT?) — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        for _ in range(8):                                   # steer GBAG_POCKET to the Items pocket (0)
            if self.b.rd8(ram.GBAG_POCKET) == 0:
                break
            self._tap("LEFT"); self._wait(12)
        if self.b.rd8(ram.GBAG_POCKET) != 0:
            self.log("   [engine] use_item: couldn't reach the Items pocket — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        for _ in range(12):                                  # nav to the item's row via cursor readback
            if self.b.rd8(BAG_CURSOR) == row:
                break
            self._tap("DOWN" if self.b.rd8(BAG_CURSOR) < row else "UP"); self._wait(10)
        if self.b.rd8(BAG_CURSOR) != row:
            self.log(f"   [engine] use_item: couldn't reach row {row} (cursor stuck) — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        # CONFIRMED in the Items pocket on the right row -> A walks select->USE->target(lead)->apply.
        # Break THE INSTANT the count drops (the use registered): a further A could re-open the bag.
        for _ in range(8):
            if self._items_count(item_id) < cnt0:
                break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)
            if not st.in_battle(self.b):
                break
        if self._items_count(item_id) < cnt0:
            self.log(f"   [engine] use_item: USED item {item_id} (count {cnt0}->{self._items_count(item_id)})")
            self.emit("used an item — that's better", beat=True)
            for _ in range(6):                               # drain the "X recovered!" text back to battle
                if self._white_box() or not st.in_battle(self.b):
                    break
                self._advance_text()
            return "used"
        self.log(f"   [engine] use_item: pocket={self.b.rd8(ram.GBAG_POCKET)} cursor={self.b.rd8(BAG_CURSOR)} "
                 f"row={row} — selected but item {item_id} NOT consumed (count still {cnt0}) — keep fighting (LOUD)")
        self._exit_bag(); return "failed"

    def _maybe_use_item(self, state):
        """OFFER the in-battle item instinct to the oracle when it's a REAL option: the active mon is
        crit-low AND a heal item is in the bag, or it's afflicted AND a matching cure is in the bag. She
        DECIDES (capability-not-script). Returns True iff an item was actually used (the turn is spent),
        so run() skips move selection this turn. Any non-'used' outcome -> fall through to a normal move
        (fail-safe — she never wedges, and never faints with unused heals because the option was surfaced)."""
        if not self.choose:
            return False
        ours = state["ours"]
        frac = _hp_frac(ours)
        offers, plan = {}, {}
        if frac <= BATTLE_CRIT_FRAC:
            heal = next((i for i in _HEAL_ITEMS_PREF if self._items_count(i) > 0), None)
            if heal is not None:
                plan["use_potion"] = heal
                offers["use_potion"] = (f"use a healing item — you're at {ours['hp']}/{ours['maxhp']} HP, "
                                        f"about to faint, and you HAVE one in the bag")
        status = self._lead_status()
        if status:
            cure = self._STATUS_CURE_for(status)
            if cure is not None:
                plan["use_cure"] = cure
                offers["use_cure"] = f"use the cure for {status} — it's hurting you and you have the item"
        if not offers:
            return False
        offers["keep_fighting"] = "keep attacking — push through it"
        ctx = {"hp": f"{ours['hp']}/{ours['maxhp']}", "status": status or "none",
               "foe": st.SPECIES_NAME.get(state["enemy"]["species"], "the foe")}
        self.log(f"   [engine] ITEM-INSTINCT offer: {list(offers)} ctx={ctx}")
        pick = self.choose("battle_item", offers, ctx)
        if pick and pick in plan:
            self.log(f"   [engine] ITEM-INSTINCT pick -> {pick} (item {plan[pick]})")
            return self.use_item_in_battle(plan[pick]) == "used"
        self.log(f"   [engine] ITEM-INSTINCT pick -> {pick!r} (keep fighting)")
        return False

    def _STATUS_CURE_for(self, status):
        """The cure item id for a status that's actually in the bag (specific cure, else Full Heal)."""
        spec = _STATUS_CURE_ITEM.get(status)
        if spec is not None and self._items_count(spec) > 0:
            return spec
        if self._items_count(_FULL_HEAL) > 0:
            return _FULL_HEAL
        return None

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
        if not (0 <= idx < 4) or not _usable(idx) or idx in self._skip_streak:
            # FIX 1 — REPETITION-AVERSE move pick: exclude EVERY move that already failed to fire this
            # streak (not just the last one), so she pivots through her whole moveset and NEVER re-spams
            # a dead/0-PP/blocked move (the Mankey case: she had 3 unused moves). Pick the best one she
            # HASN'T tried yet by expected damage. The streak clears the instant any move fires (below),
            # so a working move is never permanently benched (the PoisonPowder-spam lesson).
            cands = [i for i in range(4) if _usable(i) and i not in self._skip_streak]
            if cands:
                idx = max(cands, key=lambda i: max(ours["moves"][i].get("power", 0), 1)
                          * pol.effectiveness(ours["moves"][i].get("type", "normal"), enemy["types"]))
                desc = ours["moves"][idx].get("name", desc)
            else:
                # Every usable move has already failed to fire this streak (or none are usable at all —
                # the 0-PP Mankey wedge). Do NOT re-fire a known-dead move; surface so the run-loop
                # anti-wedge floor flees (wild) / aborts (trainer) instead of livelocking.
                self.log("   [engine] !! MOVES EXHAUSTED — every usable move tried with no effect this "
                         "streak (or none usable); not re-spamming a dead move")
                return "no_usable_move"
        eff = pol.effectiveness(ours["moves"][idx].get("type", "normal"),
                                enemy["types"]) if 0 <= idx < len(ours["moves"]) else 1.0
        # B-1 — INEFFECTIVE-MOVE AVERSION: never swing a DAMAGING move that does NOTHING (type-immune,
        # eff==0 — e.g. a Normal move into a Ghost). That's the "keeps using a move that does nothing"
        # failure. Re-pick a move that can actually connect (resisted/0.5 moves are still useful and the
        # policy already deprioritizes them; status moves at power 0 are never excluded). If she has NO
        # move that can connect, signal 'no_effective_move' — the turn loop offers a SWITCH (better
        # matchup) / else flees. Capability-not-script: she still chooses among the moves that work.
        def _useful(i):
            m = ours["moves"][i]
            if m.get("id", 0) == 0 or m.get("pp", 0) <= 0 or i in self._skip_streak:
                return False
            return not (m.get("power", 0) > 0
                        and pol.effectiveness(m.get("type", "normal"), enemy["types"]) == 0)
        if 0 <= idx < 4 and ours["moves"][idx].get("power", 0) > 0 and eff == 0:
            _uc = [i for i in range(4) if _useful(i)]
            if _uc:
                idx = max(_uc, key=lambda i: max(ours["moves"][i].get("power", 0), 1)
                          * pol.effectiveness(ours["moves"][i].get("type", "normal"), enemy["types"]))
                desc = ours["moves"][idx].get("name", desc)
                eff = pol.effectiveness(ours["moves"][idx].get("type", "normal"), enemy["types"])
                self.log(f"   [engine] avoided a type-immune move -> {desc} (eff x{eff:g}) instead")
            else:
                self.log("   [engine] !! NO EFFECTIVE MOVE — every usable move is type-immune here "
                         "(need a better matchup: switch / flee)")
                return "no_effective_move"
        # ── STATUS-MOVE STRATEGY (general, E4-critical): when EVERY damaging move is RESISTED (best
        # eff <= 0.5 — e.g. Ivysaur's Grass moves into Gary's Fire Charmander, the live look-ahead wall),
        # raw chipping loses the damage race. A STATUS move is the real play: poison/Leech-Seed chips
        # TYPE-INDEPENDENTLY (bypasses the resistance), sleep neutralizes the foe. Fire it ONCE, early,
        # when the foe is still fresh (worth the turn), then go back to chipping while the status works.
        # Capability-not-script + general (cracks any resist-wall, not just Charmander). ───────────────
        if not getattr(self, "_status_played", False):
            _dmg_effs = [pol.effectiveness(ours["moves"][i].get("type", "normal"), enemy["types"])
                         for i in range(4) if _usable(i) and ours["moves"][i].get("power", 0) > 0]
            best_dmg_eff = max(_dmg_effs) if _dmg_effs else 1.0
            foe_frac = enemy["hp"] / max(enemy.get("maxhp", 1), 1)
            if best_dmg_eff <= 0.5 and foe_frac > 0.5:
                # prefer CHIP statuses (they win the fight) over pure neutralize, both type-independent
                _STATUS_PREF = ["leechseed", "toxic", "poisonpowder", "spore", "sleeppowder",
                                "hypnosis", "lovelykiss", "sing", "stunspore"]
                _norm = lambda s: "".join(s.lower().split())
                for want in _STATUS_PREF:
                    si = next((i for i in range(4) if _usable(i)
                               and _norm(ours["moves"][i].get("name", "")) == want), None)
                    if si is not None:
                        idx, desc, self._status_played = si, ours["moves"][si].get("name", want), True
                        self.log(f"   [engine] STATUS STRATEGY: damage is resisted (best x{best_dmg_eff:g}) "
                                 f"-> {desc} to chip/neutralize past the wall (type-independent)")
                        break
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
            self._skip_streak.clear()                  # a move FIRED -> whole moveset eligible again
            #                                            (resets the streak; never permanently benches)
        else:
            # didn't fire (no PP drop, no HP change) = 0-PP / Disabled / couldn't act: add to the streak
            # so the NEXT pick rotates to a move she hasn't tried — and once all are tried, she flees
            # rather than re-spamming. The streak is per-no-progress-run, cleared on any successful fire.
            self._skip_streak.add(idx)
            self.log(f"   [engine] move slot {idx} didn't fire (0-PP / disabled / blocked) -> rotating "
                     f"to an untried move (streak now {sorted(self._skip_streak)})")
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

    # ── B-1: TYPE-MATCHUP AWARENESS + VOLUNTARY SWITCH (the E4-critical verb) ────
    def _goto_pokemon(self, tries=10):
        """Walk the action cursor to POKEMON (bottom-left, ACT_POKEMON=2). Mirror of _goto_run; grid is
        FIGHT(0,TL) BAG(1,TR) / POKEMON(2,BL) RUN(3,BR). Returns True only when confirmed on POKEMON."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_POKEMON:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("DOWN")
            elif c == ram.ACT_BAG:
                self._tap("DOWN"); self._tap("LEFT")
            elif c == ram.ACT_RUN:
                self._tap("LEFT")
            else:
                return False                              # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_POKEMON

    @staticmethod
    def _matchup_def(my_types, enemy_types):
        """How hard the enemy's STAB hits `my_types` (max eff of any enemy type vs mine). >=2 = enemy
        super-effective on me (bad); <=0.5 = I resist (good). Enemy moves are unknown, so its own types
        are the STAB proxy."""
        worst = 0.0
        for et in enemy_types:
            if et:
                worst = max(worst, pol.effectiveness(et, my_types))
        return worst or 1.0

    @staticmethod
    def _matchup_off(my_types, enemy_types):
        """Best eff of MY types vs the enemy (STAB proxy) — can I hit it hard?"""
        best = 0.0
        for t in my_types:
            if t:
                best = max(best, pol.effectiveness(t, enemy_types))
        return best

    def _best_switch_slot(self, state):
        """A CLEARLY-better healthy reserve to switch into, or None. Conservative (never churn): only
        when the ACTIVE mon is at a real disadvantage — enemy hits it super-effectively OR it can't
        damage the enemy at all — AND a healthy reserve exists that the enemy does NOT hit
        super-effectively. Ranks candidates by (resists-most, hits-hardest). Pure type math (offline-
        testable); reads non-lead species from RAM."""
        enemy_types = [t for t in (state.get("enemy", {}).get("types") or []) if t]
        if not enemy_types:
            return None
        active_types = [t for t in (state.get("ours", {}).get("types") or []) if t]
        active_bad = self._matchup_def(active_types, enemy_types) >= 2 \
            or self._matchup_off(active_types, enemy_types) == 0
        if not active_bad:
            return None
        active_sp = state.get("ours", {}).get("species")
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        best, best_key = None, None
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            sp = st.read_party_species(self.b, s)
            if sp == active_sp:
                continue                                  # (probably) the one already out
            types = st.species_types(sp)
            if not types:
                continue
            cdef = self._matchup_def(types, enemy_types)
            if cdef >= 2:
                continue                                  # also weak — not an improvement
            key = (-cdef, self._matchup_off(types, enemy_types))   # resists most, then hits hardest
            if best_key is None or key > best_key:
                best, best_key = s, key
        return best

    def _switch_to_slot(self, slot, before_sp):
        """Switch the active mon to a SPECIFIC party slot via the proven nav (action-cursor->POKEMON ->
        open list -> DOWN*slot -> A select -> A SEND OUT), confirming the active SPECIES actually changed.
        FAIL-SAFE: if it doesn't confirm, B back to the action menu and return False (caller fights —
        never wedges). Returns 'switched' or False. Shared by the matchup switch + the grind switch."""
        if not self._goto_pokemon():
            return False
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)  # open party list
        for _attempt in range(4):
            for _ in range(slot):                         # party cursor opens on slot 0 -> down to N
                self._tap("DOWN")
            self._wait(6)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)  # select
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(20)  # SEND OUT
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0 and cur["ours"].get("species") != before_sp:
                return "switched"
            self._advance_text()
        for _ in range(3):                                # didn't confirm -> back to the action menu, FIGHT
            if self._white_box():
                break
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        return False

    def _voluntary_switch(self, state):
        """Mid-battle switch to a better-matchup reserve. GATED + FAIL-SAFE. Returns 'switched' or False."""
        slot = self._best_switch_slot(state)
        if slot is None:
            return False
        self.log(f"   [engine] MATCHUP SWITCH: active is out-typed -> trying party slot {slot}")
        r = self._switch_to_slot(slot, state.get("ours", {}).get("species"))
        if r == "switched":
            self.emit("switching it up — this is a better matchup", beat=True, tier=2)
            self._skip_streak.clear()
            return "switched"
        self.log("   [engine] matchup switch did not confirm -> fighting instead (fail-safe, no wedge)")
        return False

    def _ace_reserve_slot(self):
        """The highest-level ALIVE party member that is NOT slot 0 — the ace to switch the weak grind
        lead out to (it tanks + KOs while the benched weak mon banks participation XP). None if no alive
        reserve outranks the lead (then there's nothing to switch to — just fight)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        lead_lv = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        best, best_lv = None, lead_lv
        for s in range(1, min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            if lv > best_lv:
                best, best_lv = s, lv
        return best

    # ── one battle, start to finish ────────────────────────────────────────────
    def run(self, max_seconds=120):
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "timeout"
        self._started = True
        self._bigmoment_done = False       # Phase 2D: fire shiny/legendary recognition once per battle
        self._grind_switched = False       # GRIND SWITCH: protect-lead switch fires at most once per battle
        self._status_played = False        # STATUS STRATEGY: poison/sleep/leech fires at most once per battle
        self._skip_streak = set()          # FIX 1: move slots that failed to fire this no-progress streak.
        # She rotates through her WHOLE moveset (never re-spams a 0-PP/disabled move), and only flees once
        # all are exhausted. CLEARED on any successful fire -> a working move is never permanently exiled
        # (the PoisonPowder-spam lesson: don't permanently bench a move, just rotate off it this streak).
        # Cleared the instant any move fires.
        self._prev = st.read_battle(self.b)
        self._reach_first_menu(t0, max_seconds)
        state = st.read_battle(self.b) or self._prev
        if state:
            foe = st.SPECIES_NAME.get(state["enemy"]["species"], "a wild pokemon")
            self.emit(f"a battle started against {foe}", beat=True)
            self._prev = state

        # ── BIG-MOMENT RECOGNITION (Batch 3 Phase 2D): situational SIGNIFICANCE ───────────────────────
        # SHINY is the most clippable moment the game can produce — treating one as normal is a tragedy.
        # Detect it source-first off the CONFIRMED gEnemyParty PID/otId, FREAK OUT in character, and for
        # a WILD shiny DIVERT the whole battle to careful capture (weaken, never KO, throw balls) — the
        # existing catch_pokemon path. A trainer's shiny can't be caught -> freak-out only. Shininess is
        # ~1/8192, so this branch can NEVER fire in a normal battle / the regression fixtures (verified
        # all-False) — zero risk to the battle suites. LEGENDARY/rare gets a big beat too (id check).
        if state and not getattr(self, "_bigmoment_done", False):
            self._bigmoment_done = True
            esp = state["enemy"]["species"]
            foe = st.SPECIES_NAME.get(esp, "this Pokémon")
            if st.enemy_is_shiny(self.b):
                self.emit(f"WAIT — STOP. that {foe} is SHINY. chat, do you SEE this — you can play this "
                          f"game for five years and never see one. this is real, this is happening.",
                          beat=True, tier=3)
                if not self._is_trainer_battle():
                    self.log(f"   [engine] ✨✨ SHINY wild {foe} — diverting to CAREFUL CAPTURE "
                             f"(weaken+catch, never KO)")
                    res = self.catch_pokemon(max_seconds=max(150, max_seconds), weaken=True)
                    # SAFETY: if the catch failed (no balls / broke out / timed out) and the shiny is
                    # STILL on the field, FLEE rather than fight — never KO a shiny (the tragedy), and
                    # never leave the battle hanging (a wedge). A clean catch ends the battle -> return.
                    if res != "caught" and st.in_battle(self.b):
                        self.emit(f"I couldn't catch it ({res}) — I am NOT killing a shiny, I'm backing "
                                  f"out. that hurts.", beat=True, tier=3)
                        self.log(f"   [engine] shiny capture failed ({res}) — fleeing to avoid KOing it")
                        return self.flee(max_seconds=60)
                    return res
                self.log(f"   [engine] ✨ SHINY trainer {foe} — uncatchable, fighting (freak-out only)")
            elif esp in _LEGENDARY_SPECIES:
                self.emit(f"that's a {foe}. a LEGENDARY. okay — okay, do NOT mess this up.",
                          beat=True, tier=3)

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
                        self._status_played = False         # NEW foe -> reconsider a status move (so Gary's
                        #                                     Charmander gets poisoned, not just his lead)
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
                # PART B: SURVIVAL INSTINCT FIRST — if a mon is crit-low/afflicted with a matching item,
                # offer the bag to the oracle. If she uses one, the turn is spent (skip move selection).
                # Any non-use falls through to the proven move path (fail-safe; never wedges).
                if state and not (self._enemy_fainted or self._we_fainted) and self._maybe_use_item(state):
                    self._acted_once = True
                    stall = 0
                    continue
                # PARTICIPATION-XP GRIND SWITCH: while grinding the weak team (PROTECT_LEAD_GRIND), the weak
                # mon LEADS (eligible for XP) but would be one-shot — so turn 1, switch it to the ace. The
                # weak mon banks a share of XP and never takes a hit (benched before the enemy's turn); the
                # tanky ace KOs. Fires at most once/battle; fail-safe (a non-confirm just fights).
                if (GRIND_SWITCH_ENABLED and PROTECT_LEAD_GRIND and not self._grind_switched
                        and state and not (self._enemy_fainted or self._we_fainted)):
                    self._grind_switched = True            # one attempt/battle, whatever the result
                    ace = self._ace_reserve_slot()
                    if ace is not None:
                        self.log(f"   [engine] GRIND SWITCH: weak lead out -> switching to ace slot {ace} "
                                 f"(weak mon banks participation XP, ace does the fighting)")
                        if self._switch_to_slot(ace, state.get("ours", {}).get("species")) == "switched":
                            self._acted_once = True
                            stall = 0
                            self._unresolved_turns = 0
                            continue
                        self.log("   [engine] grind switch did not confirm -> fighting (fail-safe)")
                # B-1 — MATCHUP SWITCH (gated POKEMON_BATTLE_SWITCH, fail-safe): before swinging, if the
                # active mon is badly out-typed AND a better reserve exists, switch instead. Off by
                # default until the actuation is live-verified; a failed switch backs out and fights.
                if (BATTLE_SWITCH_ENABLED and state and not (self._enemy_fainted or self._we_fainted)
                        and self._voluntary_switch(state) == "switched"):
                    self._acted_once = True
                    stall = 0
                    self._unresolved_turns = 0
                    continue
                res = self._select_and_verify(state) if state else "stuck"
                if res == "done":
                    self._acted_once = True
                    stall = 0
                    self._unresolved_turns = 0        # a real resolution clears the anti-wedge floor
                else:
                    stall += 1                        # menu up but flaky -> settle re-checks, retry
                    # ANTI-WEDGE FLOOR — the run-existential one. `stall` resets on ANY screen change,
                    # so the 0-PP "no PP left!" flicker hides the wedge from it forever. This counter
                    # only clears on a real resolution above, so a depleted/blocked turn can't hide:
                    # past the threshold we ESCAPE rather than livelock (flee a wild fight = watchable
                    # self-preservation; a trainer can't be fled -> loud abort). 'no_usable_move' rides
                    # the same counter (so a one-frame PP misread can't trip a spurious flee).
                    self._unresolved_turns += 1
                    if BATTLE_FLEE_FLOOR and self._unresolved_turns >= UNRESOLVED_FLEE_AT:
                        if not self._is_trainer_battle():
                            self.log(f"   [engine] !! ANTI-WEDGE FLOOR: {self._unresolved_turns} "
                                     f"unresolved turns (last={res}) in a WILD battle -> FLEEING "
                                     f"(self-preservation, never a frozen session)")
                            self.emit("nothing's landing and I'm out of good moves — I'm backing out "
                                      "of this one.", beat=True, tier=2)
                            return self.flee(max_seconds=60)
                        self.log(f"   [engine] !! ANTI-WEDGE FLOOR: {self._unresolved_turns} unresolved "
                                 f"turns (last={res}) in a TRAINER battle -> can't flee; LOUD abort")
                        self.emit("I'm jammed up in here — can't get a move to land.", beat=False)
                        return "stuck"
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
            # BATCH 5 PHASE 3 — mark the SIDE so she never narrates her own WIN as a loss. The bare
            # "{species} fainted" read as HER mon dying (she mourned a Nidoran she'd just KO'd). gBattleMons[1]
            # is the ENEMY, so this faint is always a victory. (Avoid the substrings 'knocked out'/'you lost'
            # — pokemon_voice.classify tiers those as a T3 LOSS; "took it down" stays the correct T1.)
            self.emit(f"the enemy's {st.SPECIES_NAME.get(ce['species'], 'Pokemon')} fainted — you took it down",
                      beat=True)
        if co["hp"] == 0 and po["hp"] > 0:
            self._we_fainted = True
            self.emit("your Pokemon fainted", beat=True)
        elif co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit("you took a big hit", beat=True)
        elif co["maxhp"] and co["hp"] / co["maxhp"] < 0.25 and po["hp"] / max(po["maxhp"], 1) >= 0.25:
            self.emit("low HP - this is getting tense", beat=True)
