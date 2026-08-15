"""recon_league_sell_check.py — OFFLINE proof the League-clerk SELL flow cashes the loot.

Boots the newest roam-start checkpoint in a STAGING dir (canonical save never touched),
stands at the League clerk (checkpoint spawns (13,0)@(2,7) = CLERK_STAND itself), runs
EliteFour._league_sell_loot() for real, and asserts:
  - money JUMPED by the Nugget's $5000 (the bag's x2 stack is MOON STONE — id 94,
    mart refuses it, $0; the only sale-ready loot here is Nugget)
  - Nugget emptied from the Items pocket; Moon Stone x2 UNTOUCHED
  - Revive x3, Super Potion x6, Full Heal x2 NOT sold
  - flow terminates back at clerk floor (no wedge)

RUN:  ../.venv/Scripts/python.exe -u recon_league_sell_check.py [checkpoint_dir]
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from campaign import Campaign        # noqa: E402
import e4_strike as e4               # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON_CKPT = os.path.join(_HERE, "states", "campaign", "checkpoints")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "sell_check_stage")

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print(f"FAIL  {msg}")
    else:
        print(f"ok    {msg}")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    src = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
           else os.path.join(CANON_CKPT,
                             "20260814_164348_a-building-in-indigo-plateau_8b_52h51m_roam-start"))
    check(os.path.exists(os.path.join(src, "kira_campaign.state")),
          f"checkpoint exists: {src}")
    # STAGE: copy the bundle sideways; every write path points at the stage.
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    shutil.copytree(src, STAGE)
    L(f"staged checkpoint -> {STAGE}")

    b = Bridge(ROM)
    with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    # boot-menu sweep (play_live: 'BOOT INTO OPEN MENU' — the bank was mid-flow)
    for _ in range(6):
        b.press("B", 6, 10)
        for _ in range(12):
            b.run_frame()

    def _stage_save(reason="tick"):
        with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
            f.write(b.save_state())
        return True

    camp = Campaign(b, battle_runner=lambda: "done",
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = _stage_save
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    check(tuple(tv.map_id(b)) == e4.LEAGUE_CENTER,
          f"boots in the League Center (map {tuple(tv.map_id(b))})")
    L(f"boot @ {tv.coords(b)} money=${camp.money()} "
      f"Nugget x{camp.bag_count(e4.NUGGET)} Mushroom x{camp.bag_count(94)} "
      f"Revive x{camp.bag_count(e4.REVIVE)} Super x{camp.bag_count(22)}")

    m0 = camp.money()
    rev0, sup0 = camp.bag_count(e4.REVIVE), camp.bag_count(22)
    elite = e4.EliteFour(camp, L, dbg_dir=STAGE)
    gained = elite._league_sell_loot()

    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(STAGE, "end_state.png"))
        L(f"end-state frame -> {STAGE}\\end_state.png")
    except Exception as e:
        L(f"frame snap failed: {e}")

    m1 = camp.money()
    L(f"RESULT: money ${m0} -> ${m1} (gained ${gained})")
    check(m1 > m0, f"money JUMPED after the sell (${m0} -> ${m1})")
    check(m1 - m0 >= 5000, f"gained >= $5000 (the Nugget; got ${m1 - m0})")
    check(camp.bag_count(e4.NUGGET) == 0,
          f"Nugget sold out (x{camp.bag_count(e4.NUGGET)} left)")
    check(camp.bag_count(94) == 2,
          f"Moon Stone x2 UNTOUCHED (mart refuses stones; x{camp.bag_count(94)} left)")
    check(camp.bag_count(103) == 0 and camp.bag_count(102) == 0,
          "no mushrooms in this bag at all (Jonny's 'Big Mushroom x2' was the "
          "Moon Stone stack)")
    check(camp.bag_count(e4.REVIVE) == rev0,
          f"Revives UNTOUCHED (x{rev0} -> x{camp.bag_count(e4.REVIVE)})")
    check(camp.bag_count(22) == sup0,
          f"Super Potions UNTOUCHED (x{sup0} -> x{camp.bag_count(22)})")
    check(camp.bag_count(23) == 2,
          f"Full Heal x2 UNTOUCHED (x{camp.bag_count(23)} left)")

    # ── full stock_up: with $5364 she must buy the Revive stack + a Full Heal ──
    L("running stock_up (sell no-op -> BUY plan) ...")
    m2 = camp.money()
    elite.stock_up()
    rev1 = camp.bag_count(e4.REVIVE)
    fh1 = camp.bag_count(23)
    L(f"KIT: money ${m2} -> ${camp.money()} | Revive x{rev0} -> x{rev1} "
      f"| Full Heal x2 -> x{fh1}")
    check(rev1 >= e4.RESTOCK_REVIVE_MIN,
          f"Revive stack >= {e4.RESTOCK_REVIVE_MIN} after stock_up (x{rev1})")
    check(camp.money() < m2, f"money spent on the kit (${m2} -> ${camp.money()})")
    check(not e4.e4_must_earn(rev1, camp.money(), camp.bag_count(e4.FULL_RESTORE),
                              here=e4.LEAGUE_CENTER),
          "no money errand armed after the sale (no doomed Fly-to-Vermilion)")
    check(not e4.e4_must_restock(rev1, camp.money(), "Lorelei",
                                 camp.bag_count(e4.FULL_RESTORE),
                                 here=e4.LEAGUE_CENTER),
          "restock gate PASSES — she may walk into Lorelei")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
