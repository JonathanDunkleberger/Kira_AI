"""recon_agatha.py — reproduce + diagnose the e4_run3 Struggle-loop terminal livelock.

Boots the room-3 snapshot (Agatha refight start: healed, FR x5, Full Heal x3), fights her
with the REAL BattleAgent, and on every abort/oddity dumps GROUND TRUTH: a frame PNG +
GBATTLE_ACTION_CURSOR + GBATTLE_RES_PTR validity + in_battle + white-box/move-list pixels +
the active mon's move PP — so the "ZERO PP -> Struggle -> no_usable_move -> abort" cycle
shows its mechanism in one pass.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_agatha.py
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
# banked_E4 = the ratchet bank the last e4 run wrote at Agatha's door (room3 entry) — the
# LIVE fixture. The old agatha_room3_probe_state is run3-era (boots Lance's room, 3 fainted).
SNAP = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_E4")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "agatha_probe")
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

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
            L(f"   frame -> {name}.png")
        except Exception as e:
            L(f"   snap failed: {e}")

    def party_pp_hp():
        out = []
        for i in range(6):
            base = ram.GPLAYER_PARTY + i * 100
            hp = b.rd16(base + 0x56)
            out.append(f"s{i} hp={hp}")
        return " | ".join(out)

    def forensics(tag):
        ac = b.rd8(ram.GBATTLE_ACTION_CURSOR)
        res = b.rd32(ram.GBATTLE_RES_PTR)
        state = st.read_battle(b)
        moves = state["ours"]["moves"] if state else None
        pp = [m.get("pp") for m in moves] if moves else None
        L(f"   [{tag}] action_cursor={ac} res_ptr={res:#010x} valid={ram.valid_ewram_ptr(res)} "
          f"in_battle={st.in_battle(b)} ours_pp={pp} "
          f"ours_hp={(state['ours']['hp'] if state else '?')} "
          f"foe={(state['enemy']['species'] if state else '?')} "
          f"foe_hp={(state['enemy']['hp'] if state else '?')}")
        L(f"   [{tag}] party: {party_pp_hp()}")
        snap(tag)

    # walk up + talk until the battle opens (trainer is 2 tiles up)
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
        forensics("no_battle")
        return 1
    L("AGATHA battle OPEN")

    def _choose(ptype, offers, ctx):
        for k in ("use_potion", "use_cure", "use_ether", "use_revive"):
            if k in offers:
                return k
        return "keep_fighting"

    for attempt in range(12):
        if not fight_open():
            break
        res = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True),
                          choose=_choose).run(max_seconds=420)
        L(f"fight() #{attempt} -> {res}")
        forensics(f"after_fight{attempt}_{res}")
        if res in ("win", "loss", "ended", "caught"):
            break
        # drain a little like the vehicle does, then loop (capped, unlike the vehicle)
        for _ in range(120):
            b.run_frame()
    L(f"final: map={tv.map_id(b)} coords={tv.coords(b)} fight_open={fight_open()}")
    forensics("final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
