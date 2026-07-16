"""recon_hm_boxswap_check.py — VERIFY the NS#28 HM box-swap credits-blocker fix.
Boots banked_LIVE (badge 5, party = venusaur + 3 diglett + mankey + kadabra, LAPRAS in box0 slot4),
drives _box_swap_for_hm('surf', state) — the exact new code the teach bridge calls when default_plan
returns None — and asserts: (a) it returns True, (b) LAPRAS (surf-capable) is now in the party, (c)
a redundant DIGLETT was boxed, (d) ht.default_plan(surf) now returns a real teach plan (the teach
bridge's next step). Live PC actuation (deposit+withdraw round-trip through the Fuchsia... — actually
banked_LIVE is at Saffron/Sabrina; the swap needs a mapped-Center city, so we position at Fuchsia via
the state's current map). Read-only w.r.t. canonical (RAM copy).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_TEAM_PLANNER", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from bridge import Bridge
import travel as tv
import firered_ram as ram
import pokemon_state as st
import hm_teach as ht
import campaign as C
from campaign import Campaign

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BOOT = sys.argv[1] if len(sys.argv) > 1 else "G:/temp/longrun/banked_LIVE/kira_campaign.state"

b = Bridge(ROM)
with open(BOOT, "rb") as f: b.load_state(f.read())
for _ in range(20): b.run_frame()
camp = Campaign(b, battle_runner=lambda: "win",
                on_event=lambda s, **k: print(f"   [voice] {s}"), beat=lambda *a, **k: None, render=lambda: None)
camp._continuity_load = lambda *a, **k: None
camp._save_campaign = lambda *a, **k: True
camp._continuity_save = lambda *a, **k: None
side = os.path.join(os.path.dirname(BOOT), "world_model.json")
try: camp.world.load(side if os.path.exists(side) else C.WORLD_JSON)
except Exception as e: print("world load:", e)

state = camp.read_live_state()
print(f"BOOT map={tv.map_id(b)} party_count={state.get('party_count')}")
party0 = [st.SPECIES_NAME.get(st.read_party_species(b, s), '?') for s in range(ram.read_party_count(b))]
print(f"party BEFORE: {party0}")
cb, occ = camp._box_scan()
print(f"box BEFORE (open={cb}): " + ", ".join(f"{bx}.{sl}={st.SPECIES_NAME.get(sp,sp)}" for (bx,sl),sp in sorted(occ.items())))

r = camp._box_swap_for_hm("surf", state)
print(f"\n_box_swap_for_hm('surf') -> {r}")

n1 = ram.read_party_count(b)
party1 = [st.SPECIES_NAME.get(st.read_party_species(b, s), '?') for s in range(n1)]
print(f"party AFTER: {party1}")
has_surfer = any(ht.hm_compatible(b, 'surf', st.read_party_species(b, s)) for s in range(n1))
plan = ht.default_plan(b, 'surf', n1)
print(f"party has surf-learner: {has_surfer}")
print(f"default_plan('surf') -> {plan}")

fails = []
def ck(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}");
    if not cond: fails.append(name)
ck("swap returned True", r is True)
ck("lapras now in party", 'lapras' in [p.lower() for p in party1])
ck("party still 6", n1 == 6)
ck("a diglett was boxed (redundant duplicate)", party1.count('diglett') < party0.count('diglett'))
ck("default_plan(surf) now non-None", plan is not None)
print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
