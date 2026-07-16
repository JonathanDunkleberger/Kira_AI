"""recon_voidlook.py — arsenal #4: LOAD the summit sandbox's saved states and LOOK at them.
The QW-4 summit run stage-saved the 'dead world' (map (0,0), party 0). Its sandbox still holds
kira_campaign.state (final save) + two pre_reload_*.state escape-hatch backups. Boot each, snap a
PNG, dump the RAM truth — the frame tells us WHAT the void actually is (title? credits? intro?)."""
import os
import sys
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge          # noqa: E402
import firered_ram as ram          # noqa: E402
import travel as tv                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SBOX = os.path.join(os.environ.get("TEMP", _HERE), "kira_watch", "sandbox_canonical_20260707_232814")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe")
os.makedirs(OUT, exist_ok=True)

for path in [os.path.join(SBOX, "kira_campaign.state")] + sorted(glob.glob(os.path.join(SBOX, "pre_reload_*.state"))):
    name = os.path.basename(path).replace(".state", "")
    b = Bridge(ROM)
    with open(path, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    sb2 = b.rd32(ram.GSAVEBLOCK2_PTR)
    print(f"{name}: map={tv.map_id(b)} coords={tv.coords(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)} "
          f"sb1={sb1:#010x}(v={ram.valid_ewram_ptr(sb1)}) sb2={sb2:#010x}(v={ram.valid_ewram_ptr(sb2)}) "
          f"frame={b.frame}", flush=True)
    png = os.path.join(OUT, f"look_{name}.png")
    b.frame_rgb().resize((480, 320)).save(png)
    print(f"   -> {png}", flush=True)
    # run 10 more seconds of frames to see if the world moves on its own (a pending script/credits)
    for _ in range(600):
        b.run_frame()
    print(f"   +600f: map={tv.map_id(b)} coords={tv.coords(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)}", flush=True)
    b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, f"look_{name}_after600.png"))
