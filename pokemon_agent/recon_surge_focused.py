"""recon_surge_focused.py — FOCUSED verification of the Surge stretch (shift 16).
Boots a Vermilion state where a party mon already KNOWS Cut, arms field moves, and drives
the gym directly: does the gym-gate-probe AUTO-CUT the (19,24) tree, enter the gym, solve the
trash-can puzzle, and beat Surge? Isolates the fix from the slow ship/Gary chain.
Persistence is redirected away from canonical (in-RAM only). Prints a clear PASS/FAIL trail.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")   # the fix under test
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from bridge import Bridge
import travel as tv
import pokemon_state as st
import field_moves as fm
from battle_agent import BattleAgent
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = sys.argv[1] if len(sys.argv) > 1 else r"G:/temp/longrun/stage/kira_campaign.state"

b = Bridge(ROM)
with open(STATE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.run_frame()
b.set_input_owner("agent")

print("map", tv.map_id(b), "coords", tv.coords(b))
cnt = b.rd8(0x02024029)
cut_slot = st.party_knows_move(b, 15, cnt)   # move 15 = Cut
print("party_cnt", cnt, "| cut_slot (knows Cut):", cut_slot)
for i in range(cnt):
    sp = st.read_party_species(b, i); lv = b.rd8(0x02024284 + i * 100 + 0x54)
    print("  slot", i, st.SPECIES_NAME.get(sp, sp), "L", lv)
print("can_use(cut):", fm.can_use(b, "cut", cnt))
print("FIELD_MOVES_ENABLED:", __import__("campaign").FIELD_MOVES_ENABLED)

if cut_slot is None:
    print("\n!! This state's party does NOT know Cut — not a valid focused fixture. "
          "Need a post-teach Vermilion state.")
    sys.exit(2)

# build a Campaign and drive the gym directly (match recon_longrun's construction)
def runner():
    return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                       log=lambda m: None).run(max_seconds=180)

camp = Campaign(b, battle_runner=runner,
                on_event=lambda s, **k: print(f"   [event/{k.get('kind','')}] {s}"),
                beat=lambda *a, **k: None, render=lambda: None)

print("\n==== driving beat_gym('Lt. Surge') ====")
try:
    out = camp.beat_gym("Lt. Surge")
    print(f"\n==== beat_gym RESULT: {out} ====")
    print("final map", tv.map_id(b), "coords", tv.coords(b),
          "badge3(Thunder)=", camp.has_badge(fm.FLAG_BADGE[3]) if hasattr(camp, "has_badge") else "?")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("!! beat_gym raised", e)
