"""recon_revive_verify.py — verify the MENU-TIME party-walk fix end-to-end (run14 wall).

Reproduces the exact chain that killed every run14 attempt: the ace faints -> forced
switch (new _party_focus + menu-time row pick) -> ITEM-INSTINCT offers use_revive ->
the aimed walk must find the fainted ace's CURRENT menu row (content-resolved, not a
pre-menu slot index) and actually consume the Revive.

Boots banked_E4 (Agatha's room), writes gPlayerParty[0].hp=1 so Venusaur faints on the
first hit, and runs the REAL BattleAgent with a chooser that REFUSES heals (so the faint
happens) but ACCEPTS use_revive. PASS = the log shows 'USED item' for the Revive AND
Venusaur reads hp>0 at some point after (resurrection observed).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_verify.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import firered_ram as ram            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SNAP = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_E4")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_verify")
os.environ.setdefault("BATTLE_DEBUG_DIR", DBG)


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    os.makedirs(DBG, exist_ok=True)
    b = Bridge(ROM)
    with open(os.path.join(SNAP, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)}")

    def hp(i):
        return b.rd16(ram.GPLAYER_PARTY + i * 100 + 0x56)

    ace_sp = st.read_party_species(b, 0)
    b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + 0x56, 1)
    L(f"ace sp={ace_sp} hp forced to {hp(0)} (faints on the first hit)")

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR)) \
            and not ram.battle_cb2_dead(b)

    for _ in range(12):
        if fight_open():
            break
        b.press("UP", 26, 10, lambda: None, owner="agent")
        for _ in range(60):
            b.run_frame()
        b.press("A", 8, 12, lambda: None, owner="agent")
        for _ in range(90):
            b.run_frame()
            if fight_open():
                break
    if not fight_open():
        L("!! battle never opened")
        return 1
    L("AGATHA battle OPEN")

    picked = {"revive": 0}

    def _choose(ptype, offers, ctx):
        if "use_revive" in offers:
            picked["revive"] += 1
            return "use_revive"
        return "keep_fighting"      # refuse heals: the ace must FAINT to arm the test

    lines = []

    def log(m):
        lines.append(m)
        print(m, flush=True)

    resurrected = [False]
    _orig = BattleAgent._maybe_use_item

    def _spy(self, state):
        r = _orig(self, state)
        if hp(0) > 1 or any(
                st.read_party_species(self.b, i) == ace_sp and hp(i) > 1
                for i in range(6)):
            resurrected[0] = True
        return r

    BattleAgent._maybe_use_item = _spy
    for attempt in range(6):
        if not fight_open():
            break
        res = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=log, choose=_choose).run(max_seconds=420)
        L(f"fight() #{attempt} -> {res}")
        if res in ("win", "loss", "ended", "caught"):
            break
        for _ in range(120):
            b.run_frame()
    used = any("use_item: USED item" in ln and "aim" not in ln for ln in lines)
    used_revive = any("ITEM-INSTINCT pick -> use_revive" in ln for ln in lines) and used
    notcons = sum(1 for ln in lines if "NOT consumed" in ln)
    nofocus = sum(1 for ln in lines if "never took focus" in ln or "never regained focus" in ln)
    ace_final = max((hp(i) for i in range(6)
                     if st.read_party_species(b, i) == ace_sp), default=0)
    L(f"VERDICT: revive_offered={picked['revive']} used_any_item={used} "
      f"not_consumed_events={notcons} focus_failures={nofocus} "
      f"ace_hp_final={ace_final} resurrected_seen={resurrected[0]}")
    ok = picked["revive"] > 0 and used_revive and (ace_final > 1 or resurrected[0])
    L("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
