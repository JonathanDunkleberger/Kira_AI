"""recon_teach_derive.py — derive the overworld TEACH chain (START menu → Bag → TM Case → mon).

Stage 2 of the HM pipeline. Uses TM39 (Rock Tomb, present in canonical's TM Case) as the
derivation vehicle on a THROWAWAY core — the TM/HM teach chain is identical, so every cursor
address + press cadence derived here replays for HM01 Cut verbatim. Never saves; canonical
untouched.

Phases (each prints ground truth + saves a frame to the scratchpad):
  1. START menu: open, ramdiff the menu cursor across DOWN presses.
  2. BAG: open from START, confirm the in-battle GBAG_POCKET/BAG_CURSOR addresses also track
     the OVERWORLD bag (they are gBagMenuState fields, expected same).
  3. Key Items pocket → TM CASE (item 366) → A: the TM Case list UI; ramdiff ITS cursor.
  4. Select a TM row → A → USE → the party screen → slot 0 → the teach/replace prompts.
RUN: python pokemon_agent/recon_teach_derive.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import firered_ram as ram              # noqa: E402
import ramdiff                         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.environ.get("TEACH_OUT",
                     r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\2024bc1a-9a72-476a-b3c1-327e88224705\scratchpad")
# overworld menu state lives well above the battle structs — MART_CURSOR (0x02039940) anchors it
LO, HI = 0x02035000, 0x0203B000


def shot(b, name):
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, name))
    except Exception:
        pass


def press(b, key, hold=8, rel=12, settle=30):
    b.press(key, hold, rel, None, owner="agent")
    for _ in range(settle):
        b.run_frame()


def main():
    b = Bridge(ROM)
    with open(os.path.join(_HERE, "states", "campaign", "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"boot {tv.map_id(b)}@{tv.coords(b)}", flush=True)

    # ── 1. START menu + its cursor ────────────────────────────────────────────
    press(b, "START", settle=60)
    shot(b, "teach_1_start.png")
    snaps, want = [ramdiff.snapshot(b, LO, HI)], [0]
    for i in range(1, 4):                      # DOWN x3, expect cursor 0->1->2->3
        press(b, "DOWN", settle=20)
        snaps.append(ramdiff.snapshot(b, LO, HI))
        want.append(i)
    addrs = ramdiff.find_tracking(snaps, want, LO)
    print(f"START-menu cursor candidates: {[hex(a) for a in addrs][:8]}", flush=True)
    # park back on top (UP x3) so the BAG row derivation below starts known
    for _ in range(3):
        press(b, "UP", settle=15)

    # ── 2. find + open BAG (walk rows pressing A only when we can verify) ─────
    # FRLG pause menu rows (post-dex): POKEDEX, POKEMON, BAG, <PLAYER>, SAVE, OPTION, EXIT → BAG = row 2.
    cur_addr = addrs[0] if addrs else None
    if cur_addr:
        for _ in range(8):
            if b.rd8(cur_addr) == 2:
                break
            press(b, "DOWN" if b.rd8(cur_addr) < 2 else "UP", settle=15)
        print(f"START cursor now {b.rd8(cur_addr)} (want 2=BAG)", flush=True)
    press(b, "A", settle=80)                   # open the bag
    shot(b, "teach_2_bag.png")
    print(f"bag pocket byte (battle addr {hex(ram.GBAG_POCKET)}): {b.rd8(ram.GBAG_POCKET)}", flush=True)

    # ── 3. Key Items pocket → TM CASE → open it ───────────────────────────────
    for _ in range(4):                         # pocket RIGHT until Key Items (expect byte==1)
        if b.rd8(ram.GBAG_POCKET) == 1:
            break
        press(b, "RIGHT", settle=20)
    print(f"pocket now {b.rd8(ram.GBAG_POCKET)} (want 1=KeyItems)", flush=True)
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key_items = [b.rd16(sb1 + 0x3B8 + i * 4) for i in range(30)]
    row_tmcase = key_items.index(364) if 364 in key_items else 0   # ITEM_TM_CASE=364 (366=Teachy TV!)
    print(f"key items {[k for k in key_items if k]}; TM Case row {row_tmcase}", flush=True)
    # derive the OVERWORLD bag list cursor (the in-battle BAG_CURSOR 0x0203AD04 does NOT track here)
    snaps, want = [ramdiff.snapshot(b, LO, HI)], [0]
    for i in range(1, 3):
        press(b, "DOWN", settle=20)
        snaps.append(ramdiff.snapshot(b, LO, HI))
        want.append(i)
    bag_addrs = ramdiff.find_tracking(snaps, want, LO)
    print(f"overworld-bag cursor candidates: {[hex(a) for a in bag_addrs][:10]}", flush=True)
    if not bag_addrs:
        shot(b, "teach_2b_bagnav.png")
        print("!! no bag cursor found — read the frame", flush=True)
        return
    bcur = bag_addrs[0]
    for _ in range(10):
        if b.rd8(bcur) == row_tmcase:
            break
        press(b, "DOWN" if b.rd8(bcur) < row_tmcase else "UP", settle=15)
    print(f"bag cursor ({hex(bcur)}) now {b.rd8(bcur)} (want {row_tmcase})", flush=True)
    press(b, "A", settle=40)                   # select TM CASE
    press(b, "A", settle=90)                   # USE -> the TM Case UI
    shot(b, "teach_3_tmcase.png")

    # ── 4. TM Case list cursor derivation (3 TMs in the case) ────────────────
    snaps, want = [ramdiff.snapshot(b, LO, HI)], [0]
    for i in range(1, 3):
        press(b, "DOWN", settle=20)
        snaps.append(ramdiff.snapshot(b, LO, HI))
        want.append(i)
    tm_addrs = ramdiff.find_tracking(snaps, want, LO)
    print(f"TM-Case cursor candidates: {[hex(a) for a in tm_addrs][:10]}", flush=True)
    for _ in range(2):
        press(b, "UP", settle=15)              # back to row 0 (TM39 Rock Tomb)
    shot(b, "teach_4_tmlist.png")

    # ── 5. select row 0 → USE → party → teach to slot 0 ──────────────────────
    press(b, "A", settle=40)                   # select the TM -> USE/GIVE/EXIT sub-menu
    shot(b, "teach_5_tmsel.png")
    press(b, "A", settle=90)                   # USE -> "Boot up the TM?" / party screen
    shot(b, "teach_6_use.png")
    press(b, "A", settle=90)                   # confirm / first party slot
    shot(b, "teach_7_party.png")
    press(b, "A", settle=90)                   # select slot 0 (the lead)
    shot(b, "teach_8_confirm.png")
    for i in range(6):                         # walk the teach dialogue, shooting each beat
        press(b, "A", settle=60)
        shot(b, f"teach_9_{i}.png")
    # ground truth: did the lead learn Rock Tomb (move id 317)?
    import pokemon_state as st
    moves = st.read_party_moves(b, 0)
    print(f"lead moves after: {moves}", flush=True)
    print("done — read the frames + candidates; NOTHING saved (throwaway core)", flush=True)


if __name__ == "__main__":
    main()
