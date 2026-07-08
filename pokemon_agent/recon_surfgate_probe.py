"""recon_surfgate_probe.py — night shift 10: why did can_surf read False for the whole
banked_E4 window? (Every no_route wedge in descent_regrade_e4surf_shift9.log sits on a water
boundary — Route 23 grass, Route 21 line, Pallet south shore — i.e. the water-aware planner
never engaged.) Load the bundle, read the gate's inputs directly."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                # noqa: E402
import pokemon_state as st               # noqa: E402
import field_moves as fm                 # noqa: E402
import travel as tv                      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
arc = sys.argv[1] if len(sys.argv) > 1 else "banked_E4"
statef = os.path.join(LONGRUN, arc, "kira_campaign.state")
print(f"arc={arc} state={statef} exists={os.path.exists(statef)}")
b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()

print(f"map={tv.map_id(b)} coords={tv.coords(b)}")
print(f"party_count read: {st.party_count(b) if hasattr(st, 'party_count') else 'n/a'}")
print(f"usable_hms = {fm.usable_hms(b)}")
print(f"can_use surf = {fm.can_use(b, 'surf')}")
# the exact gate travel.py rides (import inside function + swallow):
try:
    import field_moves as _fms
    print(f"travel-gate replica: can_surf = {_fms.can_use(b, 'surf')}")
except Exception as e:
    print(f"travel-gate replica RAISED: {type(e).__name__}: {e}")
# party moves raw — which slots know SURF (move id 57)?
for slot in range(6):
    try:
        mv = st.party_moves(b, slot) if hasattr(st, "party_moves") else None
        print(f"  slot {slot}: moves={mv}")
    except Exception as e:
        print(f"  slot {slot}: read failed {e}")
