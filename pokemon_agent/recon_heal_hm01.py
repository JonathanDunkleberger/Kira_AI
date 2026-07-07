"""recon_heal_hm01.py — heal the HM01 bank's party through the REAL machinery, then re-bank.

Loads banked_GOAL (captain's office (1,11)@(5,5), Persian 51/88), runs heal_nearest():
the interior-first rung exits the ship — the DEPARTURE CUTSCENE auto-fires on the first
exterior step (forced walk -> Vermilion (23,34), ship gone: NORMAL, HM01 is in hand) —
then heals at the Vermilion Center. Saves the healed .state over a COPY of the bank
bundle in states/campaign/hm01_bank_<ts>/ (sidecars unchanged — healing adds no story).
RUN: python pokemon_agent/recon_heal_hm01.py
"""
import os, shutil, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import field_moves as fm               # noqa: E402
import pokemon_state as st             # noqa: E402
import firered_ram as ram              # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = r"G:\temp\longrun\banked_GOAL"
PMON = st.PARTY_MON_SIZE


def party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    out = []
    for s in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + s * PMON
        out.append((st.SPECIES_NAME.get(st.read_party_species(b, s), f"#{s}").title(),
                    b.rd8(base + 0x54), b.rd16(base + 0x56), b.rd16(base + 0x58)))
    return out


def main():
    b = Bridge(ROM)
    with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"boot: map={tv.map_id(b)} pos={tv.coords(b)} party={party(b)}", flush=True)

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    # seed her mental map from the BANK's sidecar — the canonical world model has never seen
    # the ship, so the exit gradient reads empty and the complex exit wanders (first attempt).
    camp.world.load(os.path.join(BANK, "world_model.json"))
    r = camp.heal_nearest()
    print(f"heal_nearest -> {r}", flush=True)
    pt = party(b)
    full = all(hp == mx for _, _, hp, mx in pt)
    flag = bool(fm.read_flag(b, 0x237))
    print(f"end: map={tv.map_id(b)} pos={tv.coords(b)} party={pt}", flush=True)
    print(f"FLAG 0x237={flag} party_full={full}", flush=True)
    if not (full and flag):
        print("!! NOT bankable (hurt or flag lost) — stopping without a bundle", flush=True)
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(_HERE, "states", "campaign", f"hm01_bank_{ts}")
    shutil.copytree(BANK, dest)
    with open(os.path.join(dest, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    print(f"HEALED BUNDLE -> {dest}", flush=True)
    print("promote with: python pokemon_agent/promote_bank.py " + dest + " hm01", flush=True)


if __name__ == "__main__":
    main()
