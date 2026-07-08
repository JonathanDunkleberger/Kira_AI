"""recon_name_rater.py - the NAME RATER ERRAND (Phase C-3 design + offline verify).

THE DEBT: sticking nicknames. The ace has carried a mash-typed junk nickname (the AAAAAAAAAA
class) since the early keyboard era; roster-as-relationship wants names SHE chose. The Name
Rater can re-nickname any own-OT mon, free, repeatably — the in-game fix for every junk name.

GROUND TRUTH (pret/pokefirered, fetched 2026-07-07 -> G:/temp/longrun/pret/LavenderTown_House2*):
  - Name Rater = LAVENDER TOWN House2. Town door warp at (10,16); house exit tiles (3..5,7).
  - Inside: Gentleman obj (4,4), MOVEMENT_TYPE_FACE_DOWN, static -> stand (4,5), face UP, A.
  - Script LavenderTown_House2_EventScript_NameRater:
      msgbox YESNO 'rate a nickname?'          -> A (YES is default cursor)
      special ChoosePartyMon                    -> the standard party menu (ORDER LAW applies:
                                                   row i IS gPlayerParty[i] only while open)
      egg/traded-OT checks (gift LAPRAS has the player's OT -> PASSES; a real trade refuses)
      msgbox YESNO 'give it a nicer name?'      -> A (YES)
      EventScript_ChangePokemonNickname         -> THE NAMING KEYBOARD (the hazard: never
                                                   blind-A here; naming.name_entry owns it)
      confirm text                              -> drain, verify nickname changed in RAM.
  - No one-shot flag: the errand is repeatable for every junk name aboard.

THE HAZARD (memory: gift-mon keyboard class): a blind A-drain inside the naming screen TYPES.
The state machine below never drains while the keyboard is up — it hands the whole screen to
naming.name_entry (the proven starter-naming driver) and only resumes draining after.

MODES:
  NR_DRY=1 (DEFAULT — the offline verify): boot a COPY of a bank, decode every party nickname,
    flag junk names (repeated-char mash / all-same), print the full errand plan + the exact
    per-mon steps. No walking, no writes. This is tonight's verifiable half.
  NR_DRY=0 (LIVE, needs oracle eyes): assumes the boot bank is IN LAVENDER TOWN (walk there
    via the campaign travel graph first — coastal road known since shift 6). Executes the
    house entry + per-mon rename loop; SHE picks each name via the soul oracle when the bot
    is up (fallback: NR_NAME env). Banks to banked_NAMES staging, never canonical.

RUN (offline verify):
  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_name_rater.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.environ.get("WATCH") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402
import firered_ram as ram              # noqa: E402
from dialogue_reader import decode     # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")

# ── pret-derived constants (LavenderTown_House2) ─────────────────────────────
TOWN_DOOR = (10, 16)          # Lavender Town -> House2
RATER_STAND = (4, 5)          # stand here, face UP at the Gentleman (4,4)
HOUSE_EXIT = (4, 7)           # any of (3..5,7) exits back to town
NICK_OFF, NICK_LEN = 8, 10    # Gen-3 party struct: nickname bytes 8..17


def read_nickname(b, slot):
    raw = b.read_bytes(ram.GPLAYER_PARTY + slot * st.PARTY_MON_SIZE + NICK_OFF, NICK_LEN)
    s, junk = decode(raw)
    return s.strip(), junk


def is_junk_name(name):
    """The AAAAAAAAAA class: >=4 of the same char in a row, or the whole name one letter,
    or empty/undecodable. Species-default names (e.g. 'VENUSAUR') are NOT junk — she may
    still WANT to rename those, but that's her call, not a defect."""
    if not name or len(name) < 2:
        return True
    if re.search(r"(.)\1{3,}", name):
        return True
    return len(set(name.replace(" ", ""))) == 1


def main():
    boot = os.environ.get("NR_BOOT", CANON)
    dry = os.environ.get("NR_DRY", "1") == "1"
    b = Bridge(ROM)
    with open(os.path.join(boot, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    print(f"BOOT {boot}: map={tv.map_id(b)} coords={tv.coords(b)} dry={dry}", flush=True)

    cnt = min(b.rd8(ram.GPLAYER_PARTY_CNT), 6)
    plan = []
    for s in range(cnt):
        sp = st.SPECIES_NAME.get(st.read_party_species(b, s), "?")
        nick, junkiness = read_nickname(b, s)
        junk = is_junk_name(nick)
        print(f"  slot {s}: {sp:<12} nickname={nick!r} decode_junk={junkiness:.2f} "
              f"{'<< JUNK NAME (rename target)' if junk else ''}", flush=True)
        if junk:
            plan.append((s, sp, nick))

    if not plan:
        print("no junk names aboard — errand unnecessary from this bank", flush=True)
        return 0
    print(f"\nERRAND PLAN ({len(plan)} rename(s)):", flush=True)
    print("  0. travel:LavenderTown via campaign graph (coastal road, known since shift 6)")
    print(f"  1. enter House2 via town door {TOWN_DOOR} (go_warp pattern, recon_tutor_de)")
    for s, sp, nick in plan:
        print(f"  -> {sp} ({nick!r}): stand {RATER_STAND} face UP, A;"
              f" drive Y/N (A=YES); ChoosePartyMon party menu -> MENU-TIME row for slot {s}"
              f" (ORDER LAW: re-derive the row while the menu is OPEN, never carry the index);"
              f" A; Y/N (A=YES); *** naming.name_entry(b, <her pick>) — NO blind drains while"
              f" the keyboard is up ***; drain confirm; re-read nickname to verify")
    print(f"  N. exit via {HOUSE_EXIT}, bank to banked_NAMES (staging), promote via promote_bank")
    if dry:
        print("\nDRY MODE — stopping here (the live walk + keyboard needs oracle eyes; "
              "flip NR_DRY=0 from a Lavender-adjacent bank).", flush=True)
        return 0
    print("!! LIVE MODE not yet armed this pass — the walk/keyboard half ships with the live "
          "verify window (needs-eyes). Nothing executed.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
