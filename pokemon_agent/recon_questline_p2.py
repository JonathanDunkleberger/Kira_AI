"""recon_questline_p2.py — Phase 2 verify: the Deriver turns a Gate into an ordered, live-checked plan.

Cases (live campaign save, pre-Bill):
  A. STORY gate (missing FLAG_GOT_SS_TICKET) -> 1 step, unsatisfied, actionable = get ticket from Bill.
  B. CUT gate (missing 'cut') -> 2 steps [FLAG_GOT_SS_TICKET prereq, cut]; ticket unset -> actionable is
     the TICKET step (prereq first).
  C. After SETTING the ticket flag -> the cut gate's prereq is satisfied (live cross-check), actionable
     becomes the CUT step. Read-only w.r.t. saves.
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
import questline as ql             # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def show(name, q):
    print(f"\n[{name}] steps:", flush=True)
    for s in q.steps:
        print(f"     {s!r}", flush=True)
    print(f"   actionable: {q.actionable!r}", flush=True)
    print(f"   narration: {q.narration()}", flush=True)


def main():
    b = Bridge(ROM)
    b.load_state(open(os.path.join(_HERE, "states", "campaign", "kira_campaign.state"), "rb").read())
    for _ in range(60):
        b.run_frame()
    m, xy = tv.map_id(b), tv.coords(b)
    print(f"==== map={m} player={xy} ticket={fm.read_flag(b, 0x234)} ====", flush=True)
    kb = ql.load_kb()
    world = WorldModel(log=lambda *a, **k: None)
    rec = ql.GateRecognizer(b, world, kb=kb, party_count_fn=lambda: ram.read_party_count(b))
    pcf = lambda: ram.read_party_count(b)

    # A. story gate
    gA = rec.recognize(m, player_xy=xy, blocked_dir="south")
    qA = ql.derive_questline(gA, kb, b, party_count_fn=pcf)
    show("A story-gate", qA)
    okA = (qA.actionable is not None and qA.actionable.missing == "FLAG_GOT_SS_TICKET"
           and len(qA.steps) == 1 and qA.derivable)
    print(f"   [{'PASS' if okA else 'FAIL'}] 1 actionable step = get the S.S. Ticket", flush=True)

    # B. synthetic CUT gate
    gB = ql.Gate(ql.HM_OBSTACLE, missing="cut", where=m, human="a cuttable tree in the way")
    qB = ql.derive_questline(gB, kb, b, party_count_fn=pcf)
    show("B cut-gate (pre-ticket)", qB)
    okB = ([s.missing for s in qB.steps] == ["FLAG_GOT_SS_TICKET", "cut"]
           and qB.actionable is not None and qB.actionable.missing == "FLAG_GOT_SS_TICKET")
    print(f"   [{'PASS' if okB else 'FAIL'}] chain=[ticket, cut], actionable=ticket (prereq first)", flush=True)

    # C. set the ticket -> cut gate's prereq satisfied, actionable advances to cut
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    addr = sb1 + 0x0EE0 + (0x234 >> 3)
    b.core.memory.u8.raw_write(addr, b.rd8(addr) | (1 << (0x234 & 7)))
    qC = ql.derive_questline(gB, kb, b, party_count_fn=pcf)
    show("C cut-gate (post-ticket)", qC)
    okC = (qC.steps[0].satisfied and qC.actionable is not None and qC.actionable.missing == "cut")
    print(f"   [{'PASS' if okC else 'FAIL'}] ticket step satisfied (live), actionable advances to cut",
          flush=True)

    print(f"\n==== Phase-2 deriver: {'ALL PASS' if (okA and okB and okC) else 'SOME FAIL'} ====", flush=True)


if __name__ == "__main__":
    main()
