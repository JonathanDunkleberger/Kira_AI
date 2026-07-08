"""recon_partytruth.py — settle the in-battle PARTY DATA truth (run14 wall diagnosis).

THE CONTRADICTION (run14 frame bwedge_fswitch_retry1_1783471443.png vs the same-second log):
the screen showed Ekans FNT on the LEAD panel + Venusaur/Persian/Fearow/Raticate all FNT,
only Lapras standing — while gPlayerParty HP read [0, 106, 101, 77, 37, 105] (five alive).
So EITHER gPlayerParty doesn't track battle HP for slots 1-5, OR our model of which struct
the party menu displays is wrong. Both the fswitch reserve-picker and the Revive aim run on
those reads — they steered every run14 attempt into corpses until whiteout.

QUESTIONS this probe answers with ground truth (frames + full dumps each switch menu):
 1. Does gPlayerParty[i].hp (i>0) track during battle?
 2. Is gPlayerParty physically REORDERED on switch (b6bfda3 run12 law) or stable
    (order-map law, firered_ram.py:131)?
 3. Where does the party menu's live data live — EWRAM-scan every copy of each party
    PID and watch which copy's +0x56 tracks the on-screen faints.
 4. gBattlerPartyIndexes candidate 0x02023BCE (u16[4], derived from the ExecFlags/
    gBattleMons bracket): values 0..5? updates on switch?

Method: boot banked_E4 (Agatha's room), write gPlayerParty[0].hp=1 so the ace faints on
the first hit, enter the fight, fire move-slot-0 every action menu, and at every forced
party menu dump EVERYTHING + a frame, then blind-send the next mon (DOWN x deaths, A, A).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_partytruth.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import firered_ram as ram            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SNAP = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_E4")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "partytruth")

GBATTLERS_COUNT = 0x02023BCC          # candidate (gBattleControllerExecFlags + 4)
GBATTLER_PARTY_IDX = 0x02023BCE       # candidate u16[4]


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    os.makedirs(DBG, exist_ok=True)
    b = Bridge(ROM)
    with open(os.path.join(SNAP, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(240):
        b.run_frame()
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)}")

    ba = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                     log=lambda m: None, choose=None)   # helpers only, never run()

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
            L(f"   frame -> {name}.png")
        except Exception as e:
            L(f"   snap failed: {e}")

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR)) \
            and not ram.battle_cb2_dead(b)

    # ── party identity: PID + species + hp per gPlayerParty slot ──────────────
    def party_row(i):
        base = ram.GPLAYER_PARTY + i * 100
        pid = b.rd32(base)
        sp = st.read_party_species(b, i)
        name = st.SPECIES_NAME.get(sp, f"#{sp}")
        return pid, name, b.rd16(base + 0x56), b.rd16(base + 0x58)

    pids = {}
    for i in range(6):
        pid, name, hp, mx = party_row(i)
        pids[pid] = name
        L(f"   party[{i}] pid={pid:#010x} {name:<10} hp={hp}/{mx}")

    def scan_copies():
        """Every aligned EWRAM u32 equal to a party PID, excluding gPlayerParty itself."""
        hits = {}
        for a in range(ram.EWRAM_LO, ram.EWRAM_HI, 4):
            v = b.rd32(a)
            if v in pids:
                if ram.GPLAYER_PARTY <= a < ram.GPLAYER_PARTY + 600:
                    continue
                hits.setdefault(v, []).append(a)
        return hits

    def dump(tag, copies):
        rows = [party_row(i) for i in range(6)]
        L(f"   [{tag}] gPlayerParty: " + " | ".join(
            f"s{i}:{r[1]}={r[2]}/{r[3]}" for i, r in enumerate(rows)))
        ob = [b.rd8(ram.GBATTLE_PARTY_ORDER + i) for i in range(3)]
        pi = [b.rd16(GBATTLER_PARTY_IDX + 2 * i) for i in range(4)]
        L(f"   [{tag}] order_bytes={[hex(x) for x in ob]} "
          f"battlersCount={b.rd8(GBATTLERS_COUNT)} partyIdx={pi}")
        state = st.read_battle(b)
        if state:
            L(f"   [{tag}] active: {st.SPECIES_NAME.get(state['ours']['species'])} "
              f"hp={state['ours']['hp']}/{state['ours']['maxhp']} "
              f"vs {st.SPECIES_NAME.get(state['enemy']['species'])} hp={state['enemy']['hp']}")
        for pid, addrs in copies.items():
            for a in addrs:
                L(f"   [{tag}] copy {pids[pid]:<10} @{a:#010x} "
                  f"+0x56={b.rd16(a + 0x56)} +0x28={b.rd16(a + 0x28)}")

    # ── stack the deck: ace faints on the first hit ───────────────────────────
    b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + 0x56, 1)
    L(f"   wrote gPlayerParty[0].hp=1 (readback {b.rd16(ram.GPLAYER_PARTY + 0x56)})")

    # ── enter the fight (trainer is 2 tiles up) ───────────────────────────────
    for _ in range(12):
        if fight_open():
            break
        b.press("UP", 26, 10, lambda: None, owner="agent")
        for _ in range(60):
            b.run_frame()
        b.press("A", 8, 12, lambda: None, owner="agent")
        for _ in range(90):
            b.run_frame()
            if fight_open():
                break
    if not fight_open():
        L("!! battle never opened")
        snap("no_battle")
        return 1
    L("AGATHA battle OPEN — scanning EWRAM for party copies")
    copies = scan_copies()
    for pid, addrs in copies.items():
        L(f"   copies of {pids[pid]}: {[hex(a) for a in addrs]}")
    dump("battle_open", copies)
    snap("battle_open")

    # ── the fight loop: fire move 0; on every party menu, dump + blind-send ───
    switches = 0
    deadline = time.time() + 300
    idle = 0
    while time.time() < deadline and switches < 4:
        if not fight_open():
            L("battle over")
            break
        if ba._party_screen():
            switches += 1
            L(f"== PARTY MENU #{switches} ==")
            dump(f"menu{switches}_pre", copies)
            snap(f"menu{switches}_pre")
            # focus probe: does DOWN move the cursor?
            c0 = ba._party_cursor_slot()
            b.press("DOWN", 8, 10, lambda: None, owner="agent")
            for _ in range(20):
                b.run_frame()
            c1 = ba._party_cursor_slot()
            L(f"   probe: cursor {c0} -> {c1} (lead={ba._party_cursor_on_lead()})")
            # walk DOWN to the first row whose A yields SEND OUT; blind: try current
            b.press("A", 8, 10, lambda: None, owner="agent")
            for _ in range(30):
                b.run_frame()
            snap(f"menu{switches}_afterA")
            b.press("A", 8, 10, lambda: None, owner="agent")
            for _ in range(60):
                b.run_frame()
            dump(f"menu{switches}_post", copies)
            # if still on the party screen (picked a corpse / summary), B out + drift
            for _ in range(10):
                if not ba._party_screen():
                    break
                b.press("B", 8, 10, lambda: None, owner="agent")
                for _ in range(20):
                    b.run_frame()
                b.press("DOWN", 8, 10, lambda: None, owner="agent")
                for _ in range(20):
                    b.run_frame()
                b.press("A", 8, 10, lambda: None, owner="agent")
                for _ in range(30):
                    b.run_frame()
                b.press("A", 8, 10, lambda: None, owner="agent")
                for _ in range(40):
                    b.run_frame()
            continue
        if b.rd8(ram.GBATTLE_MENU_UP) == 1:
            idle = 0
            dump("turn", {})
            b.press("A", 8, 10, lambda: None, owner="agent")   # FIGHT (default cell)
            for _ in range(40):
                b.run_frame()
            b.press("A", 8, 10, lambda: None, owner="agent")   # move slot 0
            for _ in range(60):
                b.run_frame()
            continue
        idle += 1
        if idle % 40 == 0:
            snap(f"idle{idle}")
        b.press("A", 8, 10, lambda: None, owner="agent")       # advance text
        for _ in range(30):
            b.run_frame()

    dump("final", copies)
    snap("final")
    L(f"done: switches={switches} fight_open={fight_open()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
