"""recon_mkfixture.py — bank a REAL 3-mon in-battle fixture from the canonical save (Ivysaur/Rattata/
Spearow), so the switch can be tested against a proper party (all archived battle states are 1-mon).
Boots canonical, runs the real free_roam; the FIRST battle entry saves the live state to
states/workshop/canon_battle.state and stops. Read-only vs the canonical save (never writes it)."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
import firered_ram as ram                                          # noqa: E402
from campaign import Campaign, resolve_state                       # noqa: E402
import campaign as C                                               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.path.join(_HERE, "states", "workshop", "canon_battle.state")


class _Got(Exception):
    pass


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    def runner():
        # first real battle: bank the in-battle state + stop
        if st.in_battle(b):
            data = b.save_state()
            with open(OUT, "wb") as fo:
                fo.write(data)
            n = b.rd8(ram.GPLAYER_PARTY_CNT)
            print(f"BANKED canon_battle.state: in_battle=True count={n} "
                  f"party={[st.SPECIES_NAME.get(st.read_party_species(b,i),'?') for i in range(min(n,6))]}")
            raise _Got()
        return "ok"

    camp = Campaign(b, battle_runner=runner, on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)

    def chooser(kind, options, ctx):
        if kind == "battle_item":
            return None
        if kind != "action":
            return None
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        for pref in ("battle", "head_to_gym", "wander_catch"):   # anything that triggers a battle fast
            if pref in opts:
                return pref
        return next((o for o in opts if o.startswith("travel:")), opts[0] if opts else None)
    camp._oracle_choose = chooser
    # protect canonical: redirect any campaign save to a scratch path
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(C.WORLD_JSON)
    except Exception:
        pass
    print("driving canonical -> first battle (banking fixture)...", flush=True)
    try:
        camp.free_roam(max_ticks=100000, max_seconds=240, want_every=999)
    except _Got:
        print("done.")
    except Exception as e:
        print(f"stopped: {e}")
    if not os.path.exists(OUT):
        print("!! no battle reached in budget — fixture NOT created")


if __name__ == "__main__":
    main()
