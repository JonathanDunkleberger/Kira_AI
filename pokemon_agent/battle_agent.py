"""battle_agent.py - the HANDS loop for ONE battle. Mirrors ChessAgent's pattern:
ENGINE decides (pokemon_policy, no LLM), executes via button presses, and emits
NEUTRAL event summaries through an on_event callback. ZERO Kira imports - the bot
binds on_event = self._pokemon_react (the seam). Pure game-events here, never dialogue.

UNVERIFIED until run against a real battle savestate (battle offsets are CANDIDATES).
"""
import time

import pokemon_state as st
import pokemon_policy as pol

# Battle menu cursor layout (FireRed): FIGHT(0,0) BAG(1,0) / POKEMON(0,1) RUN(1,1);
# move list is a 2x2 of slots 0..3. Navigation timing reuses M0's reliable hold=8.
HOLD = 8


def _hp_frac(mon):
    return (mon["hp"] / mon["maxhp"]) if mon and mon["maxhp"] else 1.0


class BattleAgent:
    def __init__(self, bridge, on_event=None, render=None, hold_frames=HOLD):
        self.b = bridge
        self.on_event = on_event or (lambda s, **k: print(f"   [EVENT] {s}"))
        self.render = render or (lambda: None)
        self.hold = hold_frames
        self._prev = None          # last battle snapshot
        self._announced = False

    # ── primitive helpers ────────────────────────────────────────────────────
    def _tap(self, key):
        self.b.press(key, self.hold, self.hold, self.render)

    def _wait(self, frames):
        for _ in range(frames):
            self.b.run_frame(); self.render()

    def emit(self, summary, **kw):
        # NEUTRAL game-event only. Kira's self turns it into dialogue downstream.
        self.on_event(summary, **kw)

    # ── decide + execute one move ─────────────────────────────────────────────
    def take_turn(self, state):
        ours, enemy = state["ours"], state["enemy"]
        idx, desc, low = pol.choose_move(ours["moves"],
                                         [t for t in enemy["types"]],
                                         _hp_frac(ours))
        # execute: A (open FIGHT) -> navigate to move slot idx -> A (confirm)
        self._tap("A")                     # FIGHT (top-left default)
        self._wait(self.hold)
        if idx in (1, 3):                  # right column
            self._tap("RIGHT")
        if idx in (2, 3):                  # bottom row
            self._tap("DOWN")
        self._tap("A")                     # confirm move
        return desc, low

    # ── main loop for one battle ──────────────────────────────────────────────
    def run(self, max_seconds=180):
        t0 = time.time()
        while time.time() - t0 < max_seconds:
            self._wait(2)
            state = st.read_battle(self.b)
            if state is None:
                if self._announced:
                    # battle just ended
                    self.emit("the battle ended", bypass=False)
                    return "ended"
                continue
            if not self._announced:
                self._announced = True
                self.emit("a battle just started", bypass=False)
                self._prev = state
                continue

            # diff vs previous snapshot -> neutral events
            self._emit_diffs(self._prev, state)
            self._prev = state

            # if it's our turn (heuristic: stable HP, menu likely up), act
            desc, low = self.take_turn(state)
            self.emit(f"used {desc}", bypass=False)
            if low:
                self.emit("your Pokemon is in the red", bypass=False)
            self._wait(self.hold * 6)      # let the turn animate before reading again
        return "timeout"

    def _emit_diffs(self, prev, cur):
        if not prev:
            return
        if cur["enemy"]["hp"] == 0 and prev["enemy"]["hp"] > 0:
            self.emit("the enemy Pokemon fainted", bypass=False)
        if cur["ours"]["hp"] == 0 and prev["ours"]["hp"] > 0:
            self.emit("your Pokemon fainted", bypass=False)
        # big chunk taken off us this exchange
        po, co = prev["ours"], cur["ours"]
        if co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit("you took a big hit", bypass=False)
