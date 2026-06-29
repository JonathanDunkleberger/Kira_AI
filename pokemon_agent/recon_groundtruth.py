"""recon_groundtruth.py — read-only dump of the LIVE campaign save (where is she, what does she have).
Grounds a fresh session in reality before any build. No writes, no actuation."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import Campaign, resolve_state, ITEM_POTION, ITEM_POKE_BALL, ITEM_NAMES  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    b = Bridge(ROM)
    p = resolve_state("kira_campaign.state")
    print(f"save: {p}", flush=True)
    with open(p, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    s = camp.read_live_state()
    print(f"\nMAP={s['map']} place={s['place']!r} coords={s['coords']}")
    print(f"badges={s['badges']} count={s['badge_count']}  next_gym={s['next_gym']}")
    print(f"money={camp.money()}  on_grass_map={s['on_grass_map']}  dex_caught={s['dex_caught']}")
    print(f"party_count={s['party_count']}")
    for m in s["party"]:
        print(f"   {m['species']:12} L{m['level']:<3} HP {m['hp']}/{m['maxhp']}")
    print(f"\nballs(PokeBall id4)={camp.bag_count(ITEM_POKE_BALL)}  "
          f"GreatBall(3)={camp.bag_count(3)}  potions(id13)={camp.bag_count(ITEM_POTION)}")
    for iid, nm in ITEM_NAMES.items():
        c = camp.bag_count(iid)
        if c:
            print(f"   bag: {nm}({iid}) x{c}")
    print(f"\narc: {s['arc']}")


if __name__ == "__main__":
    main()
