"""recon_questline_p1.py — Phase 1 verify: the Gate Recognizer classifies the real Cerulean blocker.

Expected: on the live campaign save (player wedged at the south of Cerulean, FLAG_GOT_SS_TICKET unset),
recognize(map=(3,3), blocked_dir='south') -> STORY_NPC gate, missing FLAG_GOT_SS_TICKET. After SETTING
the flag (cross-check), the gate vanishes (road open). Read-only w.r.t. saves.
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

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402
import firered_ram as ram          # noqa: E402
import field_moves as fm           # noqa: E402
from pokemon_world import WorldModel  # noqa: E402
from questline import GateRecognizer, load_kb, STORY_NPC  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def main():
    b = Bridge(ROM)
    b.load_state(open(os.path.join(_HERE, "states", "campaign", "kira_campaign.state"), "rb").read())
    for _ in range(60):
        b.run_frame()
    m, xy = tv.map_id(b), tv.coords(b)
    print(f"==== map={m} player={xy} ====", flush=True)

    kb = load_kb()
    print(f"KB loaded: flags={list((kb.get('flags') or {}))}, "
          f"exit_gates_maps={[k for k in (kb.get('exit_gates') or {}) if k != '_doc']}, "
          f"capabilities={[k for k in (kb.get('capabilities') or {}) if k != '_doc']}", flush=True)

    world = WorldModel(log=lambda *a, **k: None)
    rec = GateRecognizer(b, world, kb=kb, party_count_fn=lambda: ram.read_party_count(b))

    # live flag read (cross-check)
    ticket = fm.read_flag(b, 0x234)
    print(f"\nlive FLAG_GOT_SS_TICKET (0x234) = {ticket}  (expect False/unset pre-Bill)", flush=True)

    g = rec.recognize(m, player_xy=xy, blocked_dir="south")
    print(f"\nrecognize(south) -> {g!r}", flush=True)
    ok1 = g is not None and g.kind == STORY_NPC and g.missing == "FLAG_GOT_SS_TICKET"
    print(f"   ctx line: {g.to_ctx() if g else '(none)'}", flush=True)
    print(f"   [{'PASS' if ok1 else 'FAIL'}] classified as STORY_NPC / FLAG_GOT_SS_TICKET", flush=True)

    # cross-check: SET the flag -> the gate must vanish (road open)
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    addr = sb1 + 0x0EE0 + (0x234 >> 3)
    b.core.memory.u8.raw_write(addr, b.rd8(addr) | (1 << (0x234 & 7)))
    print(f"\nset FLAG_GOT_SS_TICKET; live read now = {fm.read_flag(b, 0x234)}", flush=True)
    g2 = rec.recognize(m, player_xy=xy, blocked_dir="south")
    ok2 = g2 is None
    print(f"recognize(south) after ticket -> {g2!r}", flush=True)
    print(f"   [{'PASS' if ok2 else 'FAIL'}] gate clears once the flag is set (live cross-check works)",
          flush=True)

    # a direction with no KB gate should be None
    g3 = rec.recognize(m, player_xy=xy, blocked_dir="north")
    print(f"\nrecognize(north, no KB gate) -> {g3!r}  [{'PASS' if g3 is None else 'FAIL'}]", flush=True)

    print(f"\n==== Phase-1 recognizer: {'ALL PASS' if (ok1 and ok2 and g3 is None) else 'SOME FAIL'} ====",
          flush=True)


if __name__ == "__main__":
    main()
