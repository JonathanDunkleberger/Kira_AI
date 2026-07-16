"""recon_questline_p4.py — Phase 4 verify: the pipeline is GATE-KIND-AGNOSTIC + the GuideSearch fallback.

Proves the SAME recognise→derive pipeline handles Surf / Strength / item-gates exactly like Cut — using
a SYNTHETIC test-KB so we don't ship unverified far-future data (the real KB stays disasm-verified-only;
rows for surf/strength/etc. get added, disasm-cross-checked, as she approaches them). Also checks the
unresolved→GuideSearch fallback: no-op (no crash) when search is down; attaches a hint when it's up.
Read-only.
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
import questline as ql             # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")

# A SYNTHETIC KB (test-only) exercising other gate kinds through the SAME code — NOT the shipped KB.
TEST_KB = {
    "flags": {"FLAG_GOT_GOLD_TEETH": {"id": 999}},
    "exit_gates": {},
    "capabilities": {
        "surf":     {"kind": "hm", "name": "HM03 Surf", "move_id": 57, "badge_no": 5,
                     "obtain": {"via": "talk_npc", "npc": "a man in the Safari Zone", "from": "3,7",
                                "dir": None, "place_name": "the Safari Zone Secret House (Fuchsia)",
                                "prereq": None, "sets_flag": None, "gives_cap": "surf"},
                     "human": "HM Surf, from the Safari Zone"},
        "strength": {"kind": "hm", "name": "HM04 Strength", "move_id": 70, "badge_no": 4,
                     "obtain": {"via": "talk_npc", "npc": "the Safari Zone Warden", "from": "3,7",
                                "dir": None, "place_name": "the Warden's house (Fuchsia)",
                                "prereq": "FLAG_GOT_GOLD_TEETH", "sets_flag": None, "gives_cap": "strength"},
                     "human": "HM Strength, from the Warden once you return his Gold Teeth"},
        "FLAG_GOT_GOLD_TEETH": {"kind": "item", "name": "the Gold Teeth",
                     "obtain": {"via": "pickup", "npc": None, "from": "3,7", "dir": None,
                                "place_name": "the Safari Zone (find them on the ground)",
                                "prereq": None, "sets_flag": "FLAG_GOT_GOLD_TEETH", "gives_cap": None},
                     "human": "the Warden's Gold Teeth, found in the Safari Zone"},
    },
}


class _StubGuide:
    def __init__(self, up):
        self._up = up
        self.calls = []
    def available(self):
        return self._up
    def search(self, q, reason="stuck"):
        self.calls.append(q)
        return "Snorlax blocks the road; you need the Poké Flute from Mr. Fuji in Lavender Town." if self._up else None


def main():
    b = Bridge(ROM)
    b.load_state(open(os.path.join(_HERE, "states", "campaign", "kira_campaign.state"), "rb").read())
    for _ in range(40):
        b.run_frame()
    m = tv.map_id(b)
    pcf = lambda: ram.read_party_count(b)
    passes = []

    # 1. SURF gate -> 1-step questline (no prereq), same pipeline.
    gSurf = ql.Gate(ql.HM_OBSTACLE, missing="surf", where=m, human="deep water in the way")
    qSurf = ql.derive_questline(gSurf, TEST_KB, b, party_count_fn=pcf)
    print(f"\nSURF gate -> steps={[s.missing for s in qSurf.steps]} actionable={qSurf.actionable.missing if qSurf.actionable else None}", flush=True)
    print(f"   narration: {qSurf.narration()}", flush=True)
    ok = qSurf.actionable is not None and qSurf.actionable.missing == "surf" and qSurf.derivable
    passes.append(("surf single-step", ok))

    # 2. STRENGTH gate -> 2-step CHAIN (Gold Teeth prereq first), same chaining as Cut<-Ticket.
    gStr = ql.Gate(ql.HM_OBSTACLE, missing="strength", where=m, human="a boulder blocking the path")
    qStr = ql.derive_questline(gStr, TEST_KB, b, party_count_fn=pcf)
    print(f"\nSTRENGTH gate -> steps={[s.missing for s in qStr.steps]} actionable={qStr.actionable.missing if qStr.actionable else None}", flush=True)
    print(f"   narration: {qStr.narration()}", flush=True)
    ok = ([s.missing for s in qStr.steps] == ["FLAG_GOT_GOLD_TEETH", "strength"]
          and qStr.actionable.missing == "FLAG_GOT_GOLD_TEETH")
    passes.append(("strength chains via Gold Teeth prereq", ok))

    # 3. UNRESOLVED gate (not in KB) + GuideSearch DOWN -> unresolved step, no crash, no hint.
    gFlute = ql.Gate(ql.ITEM_GATE, missing="poke_flute", where=m, human="a huge Snorlax asleep on the road")
    qDown = ql.derive_questline(gFlute, TEST_KB, b, party_count_fn=pcf, guide=_StubGuide(up=False))
    print(f"\nUNRESOLVED (guide DOWN) -> resolved={qDown.derivable} narration={qDown.narration()!r}", flush=True)
    passes.append(("unresolved degrades cleanly when search is down", not qDown.derivable and qDown.actionable is not None))

    # 4. UNRESOLVED gate + GuideSearch UP -> hint attached (the fallback lights up when the 403 clears).
    sg = _StubGuide(up=True)
    qUp = ql.derive_questline(gFlute, TEST_KB, b, party_count_fn=pcf, guide=sg)
    hint_attached = qUp.actionable is not None and "piece together" in (qUp.actionable.human or "")
    print(f"\nUNRESOLVED (guide UP) -> searched={sg.calls} hint_step={qUp.actionable.human!r}", flush=True)
    passes.append(("guide fallback attaches a hint when search is live", hint_attached and bool(sg.calls)))

    print("\n---- results ----", flush=True)
    for name, ok in passes:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    print(f"\n==== Phase-4 generality + fallback: {'ALL PASS' if all(o for _n, o in passes) else 'SOME FAIL'} ====",
          flush=True)


if __name__ == "__main__":
    main()
