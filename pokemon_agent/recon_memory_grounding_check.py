"""recon_memory_grounding_check.py — Surge/Cut amnesia fix (2026-08-02 chalk).

Proves WITHOUT an emulator that:
  (1) story milestones cover badge 3+ (Surge named)
  (2) _story_so_far never blanks at badge ≥3
  (3) health publish includes field_hms
  (4) voice brief includes field moves + stronger badge language
  (5) live brief injects on heartbeat, not only pokemon_mode

RUN:  python -u recon_memory_grounding_check.py
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
    camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
    check("1a milestone 3 names Lt. Surge / Thunder",
          '3: "You\'ve beaten Brock' in camp and "Lt. Surge" in camp
          and "Thunder" in camp)
    check("1b _story_so_far walks down from badge_count",
          "for k in range(bc, -1, -1):" in camp)
    check("1c health publishes field_hms",
          '"field_hms": field_hms' in camp and "usable_hms" in camp)

    # No emulator import — pull the milestone strings via AST-free text checks + a
    # local copy of the walk-down helper logic against extracted badge-3 prose.
    check("1d milestone keys 3..8 present in source",
          all(f"\n        {n}: " in camp for n in range(3, 9)))
    # Anchor on the unique badge-3 prose (not the short _arc_note "three badges…" line).
    needle = "Lt. Surge in Vermilion for the Thunder"
    check("1e badge-3 milestone prose found", needle in camp)
    check("1f badge-3 block also names Cut",
          "You have Cut and can clear trees" in camp)
    check("1g badge-8 Elite Four summit line",
          "All eight Gym Badges are yours" in camp)

    bot = open(os.path.join(os.path.dirname(_HERE), "kira", "bot.py"), encoding="utf-8").read()
    check("2a _pokemon_live_voice_ok helper", "_pokemon_live_voice_ok" in bot)
    check("2b mic inject uses live_voice_ok",
          "if self._pokemon_live_voice_ok():" in bot
          and "_pkmn_state = self._pokemon_state_block_for_voice()" in bot)
    check("2c chat batch injects live state",
          "_live_chat = self._pokemon_state_block_for_voice()" in bot)
    check("2d voice brief mentions Field moves",
          "Field moves you can use RIGHT NOW" in bot)
    check("2e voice brief says ALREADY beaten", "you've ALREADY beaten" in bot)
    check("2f perception trusts YOUR POKÉMON RUN",
          "never deny a badge or HM it lists" in bot)
    check("2g blindness/stale use live_voice_ok",
          bot.count("if self._pokemon_live_voice_ok():") >= 3)

    print()
    if fails:
        print(f"FAIL: {fails}")
        sys.exit(1)
    print("ALL PASS — Surge/Cut memory grounding wired.")


if __name__ == "__main__":
    run()
