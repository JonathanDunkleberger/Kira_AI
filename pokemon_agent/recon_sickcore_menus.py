"""recon_sickcore_menus.py — WHICH RAM READS ARE DEAD on the live 55-hour core?

THE LIVE WEDGE (2026-08-15, Bruno/Agatha, playlive_2026-08-15_11-44-09 + _12-06-39):
every "press A at a thing and drive the menu" path fails on the live core while movement and
the battle action menu work perfectly:

    [cure] !! START menu never opened - aborting (B out)
    use_item: pocket byte MUTE + NO bag/white pixels + callback2 not the bag - bag never opened
    use_cure item 23 not consumed            <- aims at row 0, ZERO d-pad taps
    use_item: revive pos=0 want=3 lap=0/12   <- 36 DOWN taps, cursor byte never moved

Every one of those verdicts is a RAM READ, not an observed fact about the game. A savestate is a
full RAM snapshot, so a state banked after 55 hours carries that heap with it and a fresh process
inherits the disease. The open question this file answers is WHICH reads rot:

  * if the cursor/pocket bytes are dead but the game is fine, the engine is aborting menus that
    actually opened (false negatives) and the fix is to stop gating on those bytes;
  * if `_items_count` is dead, then "item not consumed" is itself a false negative and she may
    have been reviving successfully all along;
  * if party HP / money / callback2 are alive, those are the honest primitives to rebuild on.

Cross-checks a SICK state against the HEALTHY fixture, so every line is a comparison, not a
guess. Read-only: loads states, runs frames, never writes a save.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_sickcore_menus.py [sick.state ...]
      (default sick state: %TEMP%\\opencode\\sick\\live_now.state if present)
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

from bridge import Bridge                  # noqa: E402
import firered_ram as ram                   # noqa: E402
import travel as tv                         # noqa: E402
import pokemon_state as st                  # noqa: E402
import battle_agent as ba                   # noqa: E402
from battle_agent import BattleAgent        # noqa: E402
import hm_teach as ht                       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
HEALTHY = os.path.join(_HERE, "states", "workshop", "e4_revive_stage.state")
DEFAULT_SICK = os.path.join("G:\\", "temp", "opencode", "sick", "live_now.state")
REVIVE, FULL_HEAL = 24, 23
_ITEMS_POCKET_OFF = 0x310


def log(m):
    print(m, flush=True)


def pocket(b):
    """The Items pocket, decrypted with the saveblock2 key (the honest bag read)."""
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    out = []
    for s in range(42):
        slot = sb1 + _ITEMS_POCKET_OFF + s * 4
        iid = b.rd16(slot)
        if not iid:
            continue
        qty = b.rd16(slot + 2) ^ key
        if qty > 0:
            out.append((iid, qty))
    return out


def boot(path):
    b = Bridge(ROM)
    with open(path, "rb") as f:
        b.load_state(f.read())
    for _ in range(120):
        b.run_frame()
    b.set_input_owner("agent")
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                     log=lambda m: None)
    ag.owner = "agent"
    return b, ag


def probe(path, label):
    if not os.path.exists(path):
        log(f"--- {label}: MISSING ({path})")
        return None
    log("")
    log("=" * 78)
    log(f"{label}: {path}")
    log("=" * 78)
    b, ag = boot(path)
    out = {}

    # 1) THE PRIMITIVES WE TRUST ELSEWHERE (party HP / money / map / callback2) ────────────
    try:
        out["map"] = tuple(tv.map_id(b))
        out["coords"] = tuple(tv.coords(b))
        out["in_battle"] = bool(st.in_battle(b))
    except Exception as e:
        out["map"] = f"read failed: {e}"
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    party = []
    for i in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + i * 100
        party.append((i, st.SPECIES_NAME.get(st.read_party_species(b, i), "?"),
                      b.rd8(base + 0x54), b.rd16(base + 0x56), b.rd16(base + 0x58)))
    out["party"] = party
    log(f"  map={out['map']} coords={out.get('coords')} in_battle={out.get('in_battle')}")
    log(f"  party_count={cnt}")
    for row in party:
        log(f"    slot{row[0]}: {row[1]:<10} L{row[2]:<3} hp={row[3]}/{row[4]}")

    # 2) THE BAG — is `_items_count` telling the truth? Cross-check the engine's own reader
    #    against the decrypted pocket walk. If these DISAGREE, then "item not consumed" is a
    #    false negative and the whole revive diagnosis inverts.
    poc = pocket(b)
    out["pocket"] = poc
    log(f"  pocket(decrypted) = {poc}")
    for iid, name in ((REVIVE, "REVIVE"), (FULL_HEAL, "FULL HEAL")):
        truth = next((q for i, q in poc if i == iid), 0)
        try:
            engine = ag._items_count(iid)
        except Exception as e:
            engine = f"raised {e}"
        agree = (engine == truth)
        out[f"count_{iid}"] = (truth, engine, agree)
        log(f"  {name:<10} pocket={truth}  _items_count()={engine}  "
            f"{'AGREE' if agree else '!! DISAGREE — the count check is LYING'}")

    # 3) THE CURSOR BYTES the menu code gates on. Static reads only (no menu open yet), so
    #    these are just "is the address sane", but a wild value is already a red flag.
    for nm, addr in (("ITEM_PARTY_CURSOR", getattr(ba, "ITEM_PARTY_CURSOR", None)),
                     ("PARTY_CURSOR", getattr(ba, "PARTY_CURSOR", None)),
                     ("BAG_CURSOR", getattr(ba, "BAG_CURSOR", None)),
                     ("START_CURSOR", getattr(ht, "START_CURSOR", None)),
                     ("BAG_POCKET", getattr(ht, "BAG_POCKET", None))):
        if addr is None:
            log(f"  {nm:<18} (symbol not found)")
            continue
        try:
            log(f"  {nm:<18} @0x{addr:08X} = {b.rd8(addr)}")
        except Exception as e:
            log(f"  {nm:<18} @0x{addr:08X} read failed: {e}")

    # 4) DOES THE START MENU ACTUALLY OPEN? The live code decides this by watching START_CURSOR
    #    move under a DOWN tap. Judge it TWO ways — the byte AND gMain.callback2 — because if the
    #    byte is dead while cb2 says the menu is up, the engine is aborting a menu that opened.
    if not out.get("in_battle"):
        cb2_before = b.rd32(ram.GMAIN_CB2)
        c0 = b.rd8(ht.START_CURSOR) if getattr(ht, "START_CURSOR", None) else None
        b.press("START", 8, 10, lambda: None, owner="agent")
        for _ in range(90):
            b.run_frame()
        cb2_open = b.rd32(ram.GMAIN_CB2)
        c1 = b.rd8(ht.START_CURSOR) if getattr(ht, "START_CURSOR", None) else None
        b.press("DOWN", 8, 10, lambda: None, owner="agent")
        for _ in range(40):
            b.run_frame()
        c2 = b.rd8(ht.START_CURSOR) if getattr(ht, "START_CURSOR", None) else None
        byte_says_open = (c2 != c1)
        cb2_says_changed = (cb2_open != cb2_before)
        out["start_menu"] = (c0, c1, c2, byte_says_open, cb2_says_changed)
        log(f"  START menu: cursor {c0} -> {c1} -> (after DOWN) {c2}   "
            f"byte_says_opened={byte_says_open}")
        log(f"              gMain.callback2 0x{cb2_before:08X} -> 0x{cb2_open:08X}  "
            f"changed={cb2_says_changed}")
        if cb2_says_changed and not byte_says_open:
            log("  !! THE FALSE NEGATIVE: callback2 moved (a menu DID open) but the cursor byte "
                "never budged — every START-gated path aborts a menu that is actually up.")
        elif not cb2_says_changed:
            log("  -> callback2 did not move: the START press really did nothing here.")
        for _ in range(6):
            b.press("B", 6, 10, lambda: None, owner="agent")
            for _ in range(14):
                b.run_frame()
    else:
        log("  START menu: skipped (state is mid-battle)")
    del b
    return out


def main():
    args = [a for a in sys.argv[1:] if a]
    sick = args or ([DEFAULT_SICK] if os.path.exists(DEFAULT_SICK) else [])
    if not sick:
        log(f"!! no sick state given and {DEFAULT_SICK} is missing.")
        log("   copy one first:  Copy-Item pokemon_agent\\states\\campaign\\kira_campaign.state "
            f"{DEFAULT_SICK}")
    healthy = probe(HEALTHY, "HEALTHY fixture (e4_revive_stage)")
    results = [("HEALTHY", healthy)]
    for i, p in enumerate(sick):
        results.append((f"SICK[{i}]", probe(p, f"SICK[{i}] live core")))
    log("")
    log("=" * 78)
    log("VERDICT — which reads can the engine still trust?")
    log("=" * 78)
    for name, r in results:
        if not r:
            continue
        bad = [f"item-count({iid})" for iid in (REVIVE, FULL_HEAL)
               if r.get(f"count_{iid}") and not r[f"count_{iid}"][2]]
        sm = r.get("start_menu")
        if sm and sm[4] and not sm[3]:
            bad.append("START_CURSOR (cb2 moved, byte did not)")
        log(f"  {name:<9} map={r.get('map')} "
            f"{'ALL PROBED READS AGREE' if not bad else 'DEAD: ' + ', '.join(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
