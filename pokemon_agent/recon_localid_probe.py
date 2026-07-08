"""recon_localid_probe.py — validate live ObjectEvent.localId @+0x08 (shift 12).

travel.culled_template_tiles (adc280e) matches templates to live objects by localId:
template @+0, live @+0x08 — the offset is LAYOUT-derived (pret ObjectEvent), not yet
live-probed. This probe loads a bundle and cross-checks: for every ACTIVE live object,
(a) its localId must appear among the map's template localIds, and (b) its initialCoords
(@+0x0C, save frame) must equal that template's (x, y). Both holding proves the layout
(localId/mapNum/mapGroup/initialCoords block) in one shot. Also prints the culled mask
so a sealed-maze phantom is visible by eye.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_localid_probe.py [bundle...]
     (default: banked_BLAINE — the arc whose WARN motivated the fix)
"""
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

from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")


def main():
    bundles = sys.argv[1:] or ["banked_BLAINE"]
    fails = checked = 0
    for name in bundles:
        p = os.path.join(LONGRUN, name, "kira_campaign.state")
        if not os.path.exists(p):
            print(f"   {name}: no bundle state — skipped", flush=True)
            continue
        b = Bridge(ROM)
        with open(p, "rb") as f:
            b.load_state(f.read())
        for _ in range(240):
            b.run_frame()
        print(f"==== {name} @ map {tv.map_id(b)} ====", flush=True)

        # templates WITH localId (same read as travel.read_object_templates + t+0)
        ev = b.rd32(tv.GMAPHEADER + 0x04)
        n = b.rd8(ev)
        arr = b.rd32(ev + 0x04)
        tpl = {}
        for i in range(n):
            t = arr + i * 0x18
            tpl[b.rd8(t)] = (b.rds16(t + 4), b.rds16(t + 6))
        print(f"   templates ({n}): {sorted(tpl.items())}", flush=True)

        for i in range(1, 16):
            o = tv.OBJ_EVENTS + i * tv.OBJ_EVENT_SZ
            if not (b.rd8(o) & 1):
                continue
            lid = b.rd8(o + 0x08)
            cur = (b.rds16(o + 0x10) - tv.MAP_OFFSET, b.rds16(o + 0x12) - tv.MAP_OFFSET)
            init = (b.rds16(o + 0x0C) - tv.MAP_OFFSET, b.rds16(o + 0x0E) - tv.MAP_OFFSET)
            checked += 1
            if lid not in tpl:
                print(f"!! live[{i}] localId {lid} NOT in templates (cur={cur})", flush=True)
                fails += 1
            elif tpl[lid] != init:
                print(f"!! live[{i}] localId {lid}: initialCoords {init} != template "
                      f"{tpl[lid]} (cur={cur})", flush=True)
                fails += 1
            else:
                moved = " MOVED" if cur != init else ""
                print(f"   live[{i}] localId {lid}: spawn {init} == template, "
                      f"cur={cur}{moved}", flush=True)
        print(f"   culled mask: {sorted(tv.culled_template_tiles(b))}", flush=True)
        del b
    if checked == 0:
        print("RESULT: INCONCLUSIVE — no live objects checked", flush=True)
        sys.exit(2)
    print(f"==== RESULT: {'FAIL' if fails else 'PASS'} ==== "
          f"({checked} live objects checked, {fails} mismatches)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
