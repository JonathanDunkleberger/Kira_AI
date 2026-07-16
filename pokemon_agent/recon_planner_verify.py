"""recon_planner_verify.py — headless verification of the strategic-intelligence mega-batch (2026-07-08).

Pure-logic checks (no emulator): the StrategicPlanner emits sane PROACTIVE intents, roster_judgment
recognises keepers, matchup foresight fires BEFORE the wall, the E4 note surfaces at badges==8, and the
firewall holds (planner is mode-side only). Run:  .venv\\Scripts\\python.exe pokemon_agent\\recon_planner_verify.py
Exit 0 = all pass.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pokemon_state as st
from pokemon_planner import StrategicPlanner, load_strategy_kb
from pokemon_strategy import roster_judgment

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _mon(species, level):
    sid = next((k for k, v in st.SPECIES_NAME.items() if v == species), 0)
    return {"species": species, "level": level, "species_id": sid,
            "types": st.SPECIES_TYPES.get(sid, [])}


def _state(party, badges=0, next_gym=None, post_game=False, watch_goal=None):
    return {"party": party, "party_count": len(party), "badge_count": badges,
            "next_gym": next_gym, "post_game": post_game, "watch_goal": watch_goal}


print("=== KB load ===")
kb = load_strategy_kb()
check("KB has all 8 gyms + 4 E4 + champion", len(kb.get("threats", {})) == 13, str(len(kb.get("threats", {}))))
check("KB has keeper species", len(kb.get("species_quality", {})) >= 12)
check("KB has key moves (ice beam)", "ice beam" in kb.get("key_moves", {}))

pl = StrategicPlanner(kb=kb)

print("\n=== (a) proactive intents at key points ===")
# pre-Brock, WITH an answer (bulbasaur = grass, Brock weak_to grass)
n_brock = pl.plan_note(_state([_mon("bulbasaur", 8)], badges=0, next_gym={"city": "Pewter City", "leader": "Brock"}))
print("   Brock/bulbasaur:", n_brock)
check("pre-Brock note fires", bool(n_brock))
check("pre-Brock names Brock + shows she has an answer", "Brock" in n_brock and ("edge" in n_brock or "answer" in n_brock))

# pre-Brock, WITHOUT an answer (charmander = fire, not in Brock weak_to) -> should WANT water/grass
n_brock2 = pl.plan_note(_state([_mon("charmander", 8)], badges=0, next_gym={"city": "Pewter City", "leader": "Brock"}))
print("   Brock/charmander:", n_brock2)
check("(b) matchup foresight: no-answer -> a WANT before the wall", "want" in n_brock2.lower())

print("\n=== (b) matchup foresight fires BEFORE a loss (Sabrina, all-Normal team) ===")
n_sab = pl.plan_note(_state([_mon("raticate", 30), _mon("pidgeot", 30)], badges=5,
                            next_gym={"city": "Saffron City", "leader": "Sabrina"}))
print("   Sabrina/normal-team:", n_sab)
check("Sabrina no-answer warns + wants bug/ghost", "want" in n_sab.lower() and ("bug" in n_sab or "ghost" in n_sab))

print("\n=== (c) whole-team / bench-development alarm ===")
lop = [_mon("venusaur", 67), _mon("persian", 38), _mon("ekans", 15)]
n_lop = pl.plan_note(_state(lop, badges=0, next_gym={"city": "Cerulean City", "leader": "Misty"}))
print("   lopsided:", n_lop)
check("bench alarm fires on a lopsided team", "bench" in n_lop.lower() or "whole team" in n_lop.lower())

print("\n=== (d) E4 note at badges==8 — FRESH RUN (unpinned): full prep actionable ===")
lorelei_team = [_mon("venusaur", 67), _mon("persian", 38), _mon("fearow", 36),
                _mon("raticate", 31), _mon("ekans", 15), _mon("lapras", 26)]
n_e4 = pl.plan_note(_state(lorelei_team, badges=8, next_gym=None))   # no watch_goal = fresh run
print("   E4 (fresh):", n_e4)
check("E4 note fires at badges==8", bool(n_e4))
check("E4 names the gauntlet + Lance", "elite four" in n_e4.lower() and "Lance" in n_e4)
check("E4 spotlights ICE / the dragon-slayer", "ice" in n_e4.lower() or "dragon-slayer" in n_e4.lower())
check("E4 grounds it in HER Lapras (the underleveled counter)", "lapras" in n_e4.lower())
check("E4 (fresh) gives the prep actionable: level Lapras BEFORE the gauntlet",
      "before the gauntlet" in n_e4.lower())

print("\n=== (d2) E4 note — GOAL-PINNED WATCH: anticipation kept, walk-out directive DROPPED ===")
n_e4p = pl.plan_note(_state(lorelei_team, badges=8, next_gym=None,
                            watch_goal="fight through the Elite Four"))
print("   E4 (pinned):", n_e4p)
check("pinned E4 still fires + still cute (Lance/ice/Lapras)",
      "Lance" in n_e4p and "ice" in n_e4p.lower() and "lapras" in n_e4p.lower())
check("pinned E4 does NOT tell her to grind before the fight (no walk-out)",
      "before the gauntlet" not in n_e4p.lower() and "level up" not in n_e4p.lower())

print("\n=== post-game = NO prep (victory lap) ===")
n_pg = pl.plan_note(_state([_mon("venusaur", 95)], badges=8, post_game=True))
check("post-game emits no prep beat", n_pg == "")

print("\n=== (keepers) roster_judgment species-quality ===")
team = [_mon("venusaur", 30), _mon("pidgeot", 28)]
# a wild Pikachu — rare_strong keeper, team has no electric
pika = {"species_id": 25, "name": "pikachu", "level": 8, "types": ["electric"]}
rec, reason, facts = roster_judgment(team, pika, dex_new=True, quality=pl.keeper("pikachu"))
print("   pikachu:", rec, "|", reason)
check("recognises a KEEPER pikachu -> catch", rec is True)
check("keeper voice is excited (not just type-gap)", "keeper" in reason.lower())
# a wild Rattata — no quality, already effectively junk, team has normal via pidgeot? pidgeot=normal/flying
rat = {"species_id": 19, "name": "rattata", "level": 6, "types": ["normal"]}
rec2, reason2, _ = roster_judgment(team, rat, dex_new=False, quality=pl.keeper("rattata"))
print("   rattata:", rec2, "|", reason2)
check("a junk Rattata (no keeper, no gap) -> skip", rec2 is False)
# magikarp = project keeper
karp = {"species_id": 129, "name": "magikarp", "level": 5, "types": ["water"]}
rec3, reason3, _ = roster_judgment([_mon("venusaur", 30)], karp, dex_new=True, quality=pl.keeper("magikarp"))
print("   magikarp:", rec3, "|", reason3)
check("project keeper magikarp -> catch (long game)", rec3 is True)

print("\n=== (e) FIREWALL: planner imports nothing from core kira/ ===")
import pokemon_planner
import inspect
src = inspect.getsource(pokemon_planner)
check("planner never imports core kira package", "import kira" not in src and "from kira" not in src)
check("planner only reads mode-side state + KB (no bot/core handles)", "self.bot" not in src)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
