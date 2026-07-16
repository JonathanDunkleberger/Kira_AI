"""recon_ether_verify.py — verify the ETHER flow end-to-end (run15 Agatha livelock).

run15 died at Agatha: PP famine armed the Ether instinct, but the aim walk re-focused
the party list EVERY iteration — the Ether opens a move-select sub-box AFTER the mon
confirm, _party_focus read it as a stray sub-menu and B-cancelled it every lap, so the
item never consumed (itemfail_34 loop -> anti-wedge aborts forever). Fix = AIM ONCE,
then confirm blind (count drop = truth).

Boots banked_E4, enters the fight, zeroes gBattleMons[0].pp[0..3] (F_PP=0x24; famine
armed vs any foe), and runs the REAL BattleAgent with a chooser that accepts use_ether.
PASS = 'USED item' for the Ether (bag count dropped).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_ether_verify.py
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
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "ether_verify")
os.environ.setdefault("BATTLE_DEBUG_DIR", DBG)

ETHER = 34


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
    L("battle OPEN — waiting for the action menu (gBattleMons rebuilds during the intro; "
      "a pre-menu PP write gets overwritten — verify2 lesson)")
    for _ in range(1200):
        b.run_frame()
        if b.rd8(ram.GBATTLE_MENU_UP) == 1:
            break
    for i in range(4):
        b.core.memory.u8.raw_write(ram.GBATTLE_MONS + st.F_PP + i, 0)
    stx = st.read_battle(b)
    L(f"   active pp now {[m['pp'] for m in stx['ours']['moves']]} "
      f"(foe={st.SPECIES_NAME.get(stx['enemy']['species'])})")

    def _choose(ptype, offers, ctx):
        if "use_ether" in offers:
            return "use_ether"
        return "keep_fighting"

    lines = []

    def log(m):
        lines.append(m)
        print(m, flush=True)

    ba = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                     log=log, choose=_choose)
    res = ba.run(max_seconds=240)
    L(f"fight() -> {res}")
    used = any(f"use_item: USED item {ETHER}" in ln for ln in lines)
    offered = any("pick -> use_ether" in ln for ln in lines)
    notcons = sum(1 for ln in lines if "NOT consumed" in ln)
    # THE HOLE ASSERTION (run16/17 collapse): consuming the LAST Ether (display row 0)
    # leaves a zero hole in the RAM pocket — the pocket read must still see everything
    # BEHIND the hole (Revives!), or every later offer dies for the rest of the process.
    pocket = ba._items_pocket()
    revives_visible = any(i == 24 and q > 0 for i, q in pocket)
    L(f"   post-battle pocket (hole present): {pocket}")
    L(f"VERDICT: ether_offered={offered} ether_used={used} not_consumed_events={notcons} "
      f"revives_visible_after_hole={revives_visible}")
    ok = offered and used and notcons == 0 and revives_visible
    L("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
