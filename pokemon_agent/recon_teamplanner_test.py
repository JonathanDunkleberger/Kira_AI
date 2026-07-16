"""recon_teamplanner_test.py — GATE B for the TeamPlanner brain (Part B).

Deterministic assess() tests over (party, badges) fixtures + a save/resume identity check. Pure logic,
no emulator. Run: .venv/Scripts/python.exe -u pokemon_agent/recon_teamplanner_test.py
"""
import copy
import os
import sys
import tempfile

from pokemon_planner import TeamPlanner

FAILS = []
def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)

def mon(species, level):
    return {"species": species, "level": level, "hp": level, "maxhp": level, "species_id": 0}

def fresh():
    return TeamPlanner(log=lambda *a, **k: None)

print("=== T1: solo starter @ badge 1 -> catch the psychic sweeper (Abra) ===")
p = fresh()
act = p.assess([mon("ivysaur", 18)], badges=1)
check(act["kind"] == "catch_keeper", f"kind=catch_keeper (got {act['kind']})")
check(act.get("species") == "abra", f"species=abra (got {act.get('species')})")
check("Koga" in act.get("serves", []) and "Sabrina" in act.get("serves", []),
      f"Abra serves Koga+Sabrina (got {act.get('serves')})")

print("=== T2: has psychic, missing ground @ badge 2 -> catch Diglett ===")
p = fresh()
act = p.assess([mon("venusaur", 30), mon("kadabra", 26)], badges=2)
check(act["kind"] == "catch_keeper" and act.get("species") == "diglett",
      f"catch diglett (got {act['kind']}/{act.get('species')})")
check("Lt. Surge" in act.get("serves", []), f"Diglett serves Surge (got {act.get('serves')})")

print("=== T3: full six, underleveled for the next milestone -> bounded grind ===")
p = fresh()
party6 = [mon("venusaur", 40), mon("kadabra", 40), mon("dugtrio", 40),
          mon("arcanine", 40), mon("snorlax", 40), mon("lapras", 40)]
act = p.assess(party6, badges=6)   # next=Blaine, milestone 48
check(act["kind"] == "grind_to", f"kind=grind_to (got {act['kind']})")
check(act.get("level") == 48 and act.get("threat") == "Blaine",
      f"grind to L48 for Blaine (got L{act.get('level')}/{act.get('threat')})")

print("=== T4: stone-evo keeper in hand + stone in bag -> deliberate evolve (beats grind) ===")
p = fresh()
party = [mon("venusaur", 40), mon("kadabra", 40), mon("dugtrio", 40),
         mon("growlithe", 40), mon("snorlax", 40), mon("lapras", 40)]
act = p.assess(party, badges=5, bag={"fire-stone": 1})
check(act["kind"] == "evolve" and act.get("species") == "growlithe",
      f"evolve growlithe (got {act['kind']}/{act.get('species')})")
check(act.get("into") == "arcanine", f"-> arcanine (got {act.get('into')})")

print("=== T5: team fielded + on-level -> teach the due coverage TM ===")
p = fresh()
party6b = [mon("venusaur", 50), mon("kadabra", 50), mon("dugtrio", 50),
           mon("arcanine", 50), mon("snorlax", 50), mon("lapras", 50)]
act = p.assess(party6b, badges=6)   # Blaine milestone 48, top 50 -> not underleveled -> teach
check(act["kind"] == "teach_tm", f"kind=teach_tm (got {act['kind']})")
check(act.get("mon") in ("lapras", "kadabra", "arcanine"),
      f"teaches a real target (got {act.get('mon')}/{act.get('move')})")

print("=== T6: post-game -> on_track, no voice (firewall: nothing to prep) ===")
p = fresh()
act = p.assess([mon("venusaur", 60)], badges=8, post_game=True)
check(act["kind"] == "on_track" and act["voice"] == "", "post-game silent on_track")

print("=== T7: TEAM_PLANNER flag off -> None (kill-switch) ===")
import pokemon_planner as pp
_saved = pp.TEAM_PLANNER_ENABLED
pp.TEAM_PLANNER_ENABLED = False
check(fresh().assess([mon("ivysaur", 18)], badges=1) is None, "flag off -> assess None")
pp.TEAM_PLANNER_ENABLED = _saved

print("=== T8: save + resume identity (rule 17 — plan-state survives a hard kill) ===")
p = fresh()
p.init_plan([mon("ivysaur", 18)], badges=1)
p.on_acquire("abra", party=[mon("ivysaur", 18), mon("abra", 10)])
p.on_teach("TM29", "kadabra")
before = copy.deepcopy(p.state)
d = tempfile.mkdtemp(prefix="teamplan_")
ok = p.save(d)
q = fresh()
loaded = q.load(d)
check(ok and loaded, "save + load both succeeded")
check(q.state == before, "resumed plan-state is byte-identical")
# and a resumed planner still assesses correctly
act = q.assess([mon("venusaur", 30), mon("kadabra", 26)], badges=2)
check(act["kind"] == "catch_keeper" and act.get("species") == "diglett",
      f"resumed planner assesses next=diglett (got {act.get('species')})")

print("=== T9: voice is first-person + names the mon (soul check) ===")
p = fresh()
act = p.assess([mon("ivysaur", 18)], badges=1)
v = act["voice"].lower()
check("abra" in v and ("my" in v or "i " in v or "grab" in v), f"voiced in first person: {act['voice']!r}")

print()
if FAILS:
    print(f"!!! {len(FAILS)} GATE-B FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("*** GATE B: ALL TEAMPLANNER TESTS PASS ***")
