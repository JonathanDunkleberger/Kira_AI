"""recon_flash_errand.py — FAST cave/aide test harness for the Flash errand (night-train shift 1).

Boots a state (default the badge-3 surge_done fixture), loads the workshop world model so nav is warm,
then DRIVES campaign._flash_errand() in a loop — logging map/dex/phase each tick. Banks a Route-11
checkpoint (states/workshop/route11_flash.state + sidecars) the first time she stands on Route 11, so
subsequent runs can boot THERE and iterate on the cave/aide without re-walking the ~10-min back-legs.
Stops when Flash is taught (party knows move 148) or the wall-clock cap. Canonical-safe: all persistence
staged; nothing written to states/campaign or states/kira.

RUN:  .venv/Scripts/python.exe -u pokemon_agent/recon_flash_errand.py [boot_state] [max_minutes]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402
import firered_ram as ram              # noqa: E402
import field_moves as fm               # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign, resolve_state, STATES_WORKSHOP  # noqa: E402
import campaign as C                   # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
FLASH_MOVE, FLAG_HM05, ROUTE11 = 148, 0x23B, (3, 29)


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "surge_done.state"
    max_minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    # OPTIONAL ball injection (FLASH_INJECT_BALLS=N) — isolates the DEX-REACHABILITY question (can she
    # reach 10 owned on the Route 11 -> Route 2 grass?) from the ball ECONOMY. Sim of a Mart stock-up.
    _inj = int(os.getenv("FLASH_INJECT_BALLS", "0"))
    if _inj > 0:
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
        key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for s in range(16):                       # balls pocket = SB1+0x430, 16 slots (qty XOR key)
            slot = sb1 + 0x430 + s * 4
            iid = b.rd16(slot)
            if iid in (4, 0):                     # Poké Ball or empty
                b.core.memory.u16.raw_write(slot, 4)
                b.core.memory.u16.raw_write(slot + 2, _inj ^ key)
                break
        print(f"[inject] {_inj} Poké Balls -> balls pocket", flush=True)

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    # load the BOOT bundle's world model so the back-legs nav is warm (roads learned)
    boot_dir = os.path.dirname(resolve_state(boot))
    for loader, path in ((camp.world.load, os.path.join(boot_dir, "world_model.json")),
                         (camp.strat.load, os.path.join(boot_dir, "strat_memory.json"))):
        try:
            loader(path)
        except Exception as e:
            L(f"world/strat load skipped: {e}")

    def pc():
        return b.rd8(ram.GPLAYER_PARTY_CNT)

    banked_r11 = [False]

    def bank_r11():
        if banked_r11[0]:
            return
        banked_r11[0] = True
        dst = os.path.join(STATES_WORKSHOP, "route11_flash.state")
        try:
            with open(dst, "wb") as f:
                f.write(b.save_state())
            camp.world.save(os.path.join(STATES_WORKSHOP, "route11_flash.world_model.json"))
            L(f"BANKED Route-11 fixture -> {dst} (dex={ram.pokedex_owned_count(b)})")
        except Exception as e:
            L(f"!! bank Route-11 failed: {e}")

    L(f"boot={boot} map={tv.map_id(b)}@{tv.coords(b)} dex={ram.pokedex_owned_count(b)} "
      f"party={pc()} flash_known={st.party_knows_move(b, FLASH_MOVE, pc())}")

    last = None
    stall = 0
    stuck = 0
    while time.time() - t0 < max_minutes * 60:
        if st.party_knows_move(b, FLASH_MOVE, pc()) is not None:
            L("✅ FLASH TAUGHT — errand complete")
            break
        if tuple(tv.map_id(b)) == ROUTE11:
            bank_r11()
        r = camp._flash_errand()
        if r == "flash_stuck":
            stuck += 1
            if stuck >= 3:
                L(f"errand surfaced flash_stuck ×{stuck} (guard working — no freeze-spin); stopping")
                break
        else:
            stuck = 0
        sig = (tuple(tv.map_id(b)), tuple(tv.coords(b) or ()), ram.pokedex_owned_count(b))
        L(f"errand -> {r} | map={tv.map_id(b)}@{tv.coords(b)} dex={ram.pokedex_owned_count(b)} "
          f"hm05={bool(fm.read_flag(b, FLAG_HM05))}")
        if sig == last:
            stall += 1
            if stall >= 8:
                L(f"!! STALL: {stall} identical ticks at {sig} (r={r}) — stopping")
                break
        else:
            stall = 0
            last = sig

    L(f"FINAL: map={tv.map_id(b)}@{tv.coords(b)} dex={ram.pokedex_owned_count(b)} "
      f"hm05={bool(fm.read_flag(b, FLAG_HM05))} flash_slot={st.party_knows_move(b, FLASH_MOVE, pc())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
