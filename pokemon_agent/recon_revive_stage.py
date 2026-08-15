"""recon_revive_stage.py — bake a REUSABLE mid-battle fixture for the Revive-cursor work.

The live failure (docs/soak-reports/20260814_173116) is Lorelei with 6 Revives in the bag and a
0-HP wincon on the floor: every attempt logged "REFUSE A - still on living lead after RIGHT/DOWN".
Reproducing it by re-running the whole gauntlet costs minutes per iteration; this bakes the exact
position to disk ONCE so the instrument/fix loop is seconds.

Boots the League-Center frontier state, warps through the League door into Lorelei's room, walks
north into her line of sight, and the instant the battle is open:
    * forces gPlayerParty[DEAD_SLOT].hp = 0   (the fainted wincon — the live case was row 2)
    * settles to the ACTION menu
    * writes states/workshop/e4_revive_stage.state

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_stage.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge              # noqa: E402
import firered_ram as ram              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402
from battle_agent import BattleAgent    # noqa: E402
from campaign import Campaign          # noqa: E402
import campaign as C                   # noqa: E402
import e4_strike as e4                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(os.environ.get("TEMP", _HERE), "longrun",
                   "sell_check_stage", "kira_campaign.state")
DST = os.path.join(_HERE, "states", "workshop", "e4_revive_stage.state")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_stage")
DEAD_SLOT = int(os.environ.get("DEAD_SLOT", "2"))     # Zapdos in this party = the live case


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    os.makedirs(DBG, exist_ok=True)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    b = Bridge(ROM)
    with open(SRC, "rb") as f:
        b.load_state(f.read())
    for _ in range(120):
        b.run_frame()
    b.set_input_owner("agent")

    def fight():
        return "win"          # never used: we stop the moment the battle opens

    camp = Campaign(b, battle_runner=fight, on_event=lambda s, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    ef = e4.EliteFour(camp, L, dbg_dir=DBG)
    L(f"boot map={tv.map_id(b)}@{tv.coords(b)} revives={camp.bag_count(e4.REVIVE)} "
      f"party_alive={ef.party_alive()}")

    here = tuple(tv.map_id(b))
    if here == e4.LEAGUE_CENTER:
        if not ef.go_warp(e4.LEAGUE_DOOR, "league-door"):
            L("!! league door never fired")
            return 1
        L(f"through the door -> {tv.map_id(b)}@{tv.coords(b)}")

    # Walk north into the seat's line of sight. The room is a straight hall; UP is enough.
    for i in range(40):
        if ef.fight_open():
            break
        b.press("UP", 26, 10, lambda: None, owner="agent")
        for _ in range(60):
            b.run_frame()
            if ef.fight_open():
                break
        if ef.fight_open():
            break
        b.press("A", 8, 12, lambda: None, owner="agent")
        for _ in range(90):
            b.run_frame()
            if ef.fight_open():
                break
    if not ef.fight_open():
        L(f"!! battle never opened (at {tv.map_id(b)}@{tv.coords(b)})")
        ef.snap("stage_fail")
        return 1
    L(f"BATTLE OPEN at {tv.map_id(b)}@{tv.coords(b)} seat={e4.ROOM_SEAT.get(tuple(tv.map_id(b)))}")

    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=L)
    ag.owner = "agent"
    ag._reach_first_menu(time.time(), 40)
    if not ag._settle_action_menu():
        L("!! never reached the action menu")
        ef.snap("stage_no_action_menu")
        return 1

    base = ram.GPLAYER_PARTY + DEAD_SLOT * 100
    b.core.memory.u16.raw_write(base + 0x56, 0)
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    for i in range(min(cnt, 6)):
        bs = ram.GPLAYER_PARTY + i * 100
        L(f"   slot{i}: sp={st.read_party_species(b, i)} "
          f"({st.SPECIES_NAME.get(st.read_party_species(b, i), '?')}) "
          f"lvl={b.rd8(bs + 0x54)} hp={b.rd16(bs + 0x56)}/{b.rd16(bs + 0x58)}")
    L(f"   battler_party_idx={b.rd16(ram.GBATTLER_PARTY_IDX)} "
      f"action_menu={ag._at_action_menu()} revives={camp.bag_count(e4.REVIVE)}")

    with open(DST, "wb") as f:
        f.write(b.save_state())
    b.frame_rgb().save(os.path.join(DBG, "stage_ok.png"))
    L(f"WROTE {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
