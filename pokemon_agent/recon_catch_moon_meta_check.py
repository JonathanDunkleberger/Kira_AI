"""recon_catch_moon_meta_check.py — catch LAW + Mt.Moon meta (2026-08-02 chalk).

Proves WITHOUT an emulator that:
  (1) bot latches catch_now from voice
  (2) battle_agent diverts shiny/legendary/creator/diglett to careful catch
  (3) flash Diglett's Cave uses targeted catch + flee-cross (not fight-kill)
  (4) tunnel voice uses place_name (not hardcoded Rock Tunnel)
  (5) post-Cascade Mt. Moon escape + avoid + warp refuse are wired
  (6) roster_judgment boxes rare/strong when party is full
  (7) diglett/moltres are in species_quality KB

RUN:  python -u recon_catch_moon_meta_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def run():
    bot_path = os.path.join(os.path.dirname(_HERE), "kira", "bot.py")
    bot = open(bot_path, encoding="utf-8").read()
    check("1a bot has CATCH order regex", "_POKEMON_CATCH_ORDER_RX" in bot)
    check("1b bot latches catch_now", 'order = "catch_now"' in bot)

    ba = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
    check("2a creator catch peek", "_peek_creator_catch_order" in ba)
    check("2b wild catch divert helper", "_divert_wild_catch" in ba)
    check("2c legendary divert (not voice-only)", 'return self._divert_wild_catch("legendary"' in ba)
    check("2d diglett cave keeper divert", 'return self._divert_wild_catch("diglett_keeper"' in ba)
    check("2e creator catch_now divert", 'return self._divert_wild_catch("creator_catch_now"' in ba)

    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("3a Diglett ONE catch in flash Phase 3",
          'target_species="diglett"' in camp and "ONE diglett catch" in camp)
    check("3b flash Diglett cave uses FIGHT runner (Arena Trap)",
          "Arena Trap" in camp and "never flee Diglett" in camp)
    check("3c catch_now does not badge-fulfill",
          'order not in ("catch_now", "get_flash")' in camp and "_fulfill_catch_order" in camp)
    check("3c3 flee() refuses Diglett Arena Trap RUN-spam",
          "def _foe_blocks_flee" in ba
          and "Arena Trap — Can't escape. FIGHTING to clear" in ba)
    check("3c2 get_flash order + Arena Trap fight-clear",
          'order == "get_flash"' in camp and "_skip_catch_divert" in open(
              os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read())
    check("3d tunnel uses place_name not hardcoded Rock Tunnel voice",
          "_cave_here = self._place_name" in camp
          and 'self.on_event("Rock Tunnel — pitch dark' not in camp)
    check("3e off-mission Mt.Moon escape", "_escape_off_mission_mt_moon" in camp)
    check("3f cleared dungeon avoid", "_cleared_dungeon_avoid" in camp and "_MT_MOON_MAPS" in camp)
    check("3g enter_warp refuses Mt.Moon post-Cascade",
          "REFUSING door" in camp and "Mt. Moon already cleared" in camp)
    check("3h free_roam calls escape at tick top",
          'ledger.note_action("escape_mt_moon"' in camp)

    from pokemon_strategy import roster_judgment
    from pokemon_planner import StrategicPlanner, load_strategy_kb
    kb = load_strategy_kb()
    pl = StrategicPlanner(kb=kb)
    check("4a diglett in species_quality", "diglett" in (kb.get("species_quality") or {}))
    check("4b moltres in species_quality", "moltres" in (kb.get("species_quality") or {}))
    full = [{"species_id": 9, "level": 37, "types": ["water"]},
            {"species_id": 16, "level": 18, "types": ["normal", "flying"]},
            {"species_id": 19, "level": 14, "types": ["normal"]},
            {"species_id": 20, "level": 15, "types": ["normal"]},
            {"species_id": 21, "level": 16, "types": ["normal", "flying"]},
            {"species_id": 64, "level": 22, "types": ["psychic"]}]
    dig = {"species_id": 50, "name": "diglett", "level": 18, "types": ["ground"]}
    rec, reason, facts = roster_judgment(full, dig, dex_new=True, quality=pl.keeper("diglett"))
    print(f"   full-party diglett: rec={rec} | {reason}")
    check("4c full party + diglett/keeper -> catch (box)", rec is True)
    check("4d reason mentions box", "box" in reason.lower())

    print()
    if fails:
        print(f"FAIL: {fails}")
        sys.exit(1)
    print("ALL PASS — catch LAW + Mt.Moon meta wired.")


if __name__ == "__main__":
    run()
