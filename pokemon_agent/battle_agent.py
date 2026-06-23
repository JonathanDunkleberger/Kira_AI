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
        self._no_progress = 0          # consecutive action-menu visits with no battle change
        self._last_progress = None
        # menu-agnostic recovery (the live party-submenu trap): global stall watchdog
        self._recovery_attempts = 0
        self._last_global = None
        self._stale = 0

    # ── input (owner-attributed) ───────────────────────────────────────────────
    def _tap(self, key):
        self.b.press(key, self.hold, self.hold, self.render, owner=self.owner)

    def _wait(self, frames):
        for _ in range(frames):
            self.b.run_frame(); self.render()

    # ── RELIABLE action-menu detection (the phase register is noise - it reads 0x580
    # during intro text too; verified by screenshot). The action cursor RESPONDS to the
    # D-pad only when the FIGHT/BAG/POKEMON/RUN menu is interactive, so we probe it. ──
    def _action_menu_ready(self):
        """True iff the action menu is up + interactive. Net-zero on the menu
        (FIGHT->POKEMON->FIGHT); a no-op during intro/animations; in a submenu it
        nudges that submenu's cursor harmlessly and reports False."""
        c0 = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
        self._tap("DOWN")
        c1 = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
        self._tap("UP")
        return c1 != c0 and c1 in (0, 1, 2, 3)

    def _settle_to_action_menu(self, timeout=80):
        """PREVENT: advance the battle intro/text until the action menu is interactive,
        so we never press a selection into an unsettled menu (the fresh-battle desync
        that opened the POKEMON party submenu). Bounded; stops if the battle ends."""
        for _ in range(timeout):
            if not st.in_battle(self.b):
                return False
            if self._action_menu_ready():
                return True
            self._tap("A")            # advance intro / text
        return False

    def _escape_submenu(self):
        """RECOVER: if we fumbled into a submenu (party/bag), press B to back out
        toward the action menu, then re-settle. B never CONFIRMS a wrong selection."""
        for _ in range(5):
            if self._action_menu_ready():
                return True
            self._tap("B")
        return self._settle_to_action_menu()

    # ── perception ─────────────────────────────────────────────────────────────
    def _phase(self):
        return self.b.rd32(ram.GBATTLE_PHASE)

    def _at_action_menu(self):
        return self._phase() == ram.PHASE_ACTION_MENU

    def _move_list_up(self):
        return self._phase() == ram.PHASE_MOVE_LIST

    # ── events + performance beats ─────────────────────────────────────────────
    def emit(self, summary, beat=False):
        """NEUTRAL game-event -> her self. beat=True is a PERFORMANCE moment: yield the
        floor so her voice lands before the hands advance (brisk on non-beats)."""
        self.on_event(summary)
        if beat and self.pace:
            self.pace(summary)

    # ── execute one chosen move through the menu (2x2 grid) ────────────────────
    # IMPORTANT (found empirically vs ground truth): after A opens the move list, a
    # bare confirm-A is IGNORED until a DIRECTIONAL press engages the cursor. So we
    # always issue a directional input before confirming (navigation provides it; for
    # the default slot 0 we wiggle DOWN/UP), then confirm + VERIFY the move was
    # accepted (my_turn 2->0), retrying if not. No blind mashing - every step checked.
    def _nav_to(self, idx):
        if idx == 0:
            self._tap("DOWN"); self._tap("UP")        # net slot 0, but engages the menu
        elif idx == 1:
            self._tap("RIGHT")
        elif idx == 2:
            self._tap("DOWN")
        elif idx == 3:
            self._tap("RIGHT"); self._tap("DOWN")

    def _goto_fight(self):
        """READ the action cursor (0x02023FF8) and move it to FIGHT deterministically -
        never assume the position. FIGHT is top-left: UP from POKEMON/RUN, LEFT from BAG.
        Returns True only when the cursor is confirmed on FIGHT (0)."""
        for _ in range(5):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_FIGHT:
                return True
            if c == ram.ACT_BAG:
                self._tap("LEFT")
            elif c in (ram.ACT_POKEMON, ram.ACT_RUN):
                self._tap("UP")
            else:
                return False                          # not a valid action menu (garbage)
            self._wait(4)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_FIGHT

    def _execute(self, idx, desc, eff):
        # Re-center on FIGHT by READING the cursor (fixes the blind-nav POKEMON-menu
        # oscillation), then open the move list and select. A directional press is
        # required before the confirm-A registers (the eaten-press quirk); verify the
        # move was accepted by the phase leaving the action menu (0x580).
        if not self._goto_fight():
            self.log("   [engine] could not reach FIGHT (cursor desync) - skipping press")
            return False
        self._tap("A"); self._wait(12)                # open FIGHT move list
        self._nav_to(idx); self._wait(6)              # engage + position on the slot
        self._tap("A"); self._wait(12)                # confirm
        if self._at_action_menu() and self._goto_fight():   # not accepted -> retry once
            self._tap("A"); self._wait(8)
            self._nav_to(idx); self._wait(4)
            self._tap("A"); self._wait(12)
        self._last_desc, self._last_eff = desc, eff   # narrated when the hit actually lands
        return True

    # ── one battle, start to finish ────────────────────────────────────────────
    def run(self, max_seconds=120):
        t0 = time.time()
        outcome = "timeout"
        last_key, stable, acted = None, 0, False
        while time.time() - t0 < max_seconds:
            self._wait(1)
            if not st.in_battle(self.b):
                if self._started:
                    outcome = self._finish()
                    break
                continue
            state = st.read_battle(self.b)
            if not self._started:
                self._started = True
                foe = st.SPECIES_NAME.get(state["enemy"]["species"], "a wild pokemon")
                self.emit(f"a battle started against {foe}", beat=True)
                self._settle_to_action_menu()   # PREVENT: don't act on an unsettled menu
                self._prev = state
                continue
            self._emit_diffs(self._prev, state)
            self._prev = state
            # RECOVER (menu-agnostic): if the battle makes NO progress for a long window
            # we're stuck somewhere - classically the party submenu the fresh-battle
            # desync opened, where _goto_fight can't help. Back out (B) + re-settle
            # before the loud abort. Won't trip in a normal battle (HP keeps changing).
            glob = (state["enemy"]["hp"], state["ours"]["hp"])
            if glob == self._last_global:
                self._stale += 1
            else:
                self._last_global, self._stale = glob, 0
            if self._stale >= 240:
                self._recovery_attempts += 1
                if self._recovery_attempts > 4:
                    self.log("   [engine] !! STUCK after B-escape recovery - aborting loudly")
                    self.emit("okay I'm properly stuck, the menu's glitched", beat=False)
                    return "stuck"
                self.log(f"   [engine] global stall (recovery {self._recovery_attempts}) - "
                         f"B-escape any submenu + re-settle")
                self._escape_submenu()
                self._stale = 0
                continue
            if self._at_action_menu():
                if not acted:                       # rising edge: choose + execute once
                    ours, enemy = state["ours"], state["enemy"]
                    # STUCK DETECTION: the action menu keeps returning with no change to
                    # the battle = we're spinning (e.g. cursor desync). Fail LOUD, never
                    # silently forever.
                    prog = (enemy["hp"], ours["hp"])
                    self._no_progress = self._no_progress + 1 if prog == self._last_progress else 0
                    self._last_progress = prog
                    if self._no_progress >= 8:
                        # don't just give up - TRY to escape (the desync may have left us
                        # in a submenu). Back out (B) + re-settle, retry; abort only if
                        # recovery itself keeps failing.
                        self._recovery_attempts += 1
                        if self._recovery_attempts > 4:
                            self.log("   [engine] !! STUCK after B-escape recovery - aborting loudly.")
                            self.emit("ugh, I'm stuck in this menu - something's glitched", beat=False)
                            return "stuck"
                        self.log(f"   [engine] action-menu stall (recovery {self._recovery_attempts}) "
                                 f"- B-escape any submenu + re-settle, then retry")
                        self._escape_submenu()
                        self._no_progress = 0
                        acted = False
                        continue
                    idx, desc, low = pol.choose_move(ours["moves"], enemy["types"], _hp_frac(ours))
                    eff = pol.effectiveness(ours["moves"][idx].get("type", "normal"),
                                            enemy["types"]) if 0 <= idx < len(ours["moves"]) else 1.0
                    self.log(f"   [engine] action menu: {desc} (eff x{eff:g}) vs "
                             f"{st.SPECIES_NAME.get(enemy['species'], '?')} {enemy['hp']}/{enemy['maxhp']}")
                    self._execute(idx, desc, eff)
                    acted = True
                    last_key, stable = None, 0
            else:
                acted = False                       # left the menu -> ready for next turn
                key = (state["enemy"]["hp"], state["ours"]["hp"], self._phase())
                stable = stable + 1 if key == last_key else 0
                last_key = key
                if stable >= 14:                    # a settled text box -> advance it
                    self._tap("A")
                    last_key, stable = None, 0
        return outcome

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
            self.emit("your Pokemon fainted", beat=True)
        elif co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit("you took a big hit", beat=True)
        elif co["maxhp"] and co["hp"] / co["maxhp"] < 0.25 and po["hp"] / max(po["maxhp"], 1) >= 0.25:
            self.emit("low HP - this is getting tense", beat=True)
