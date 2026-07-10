"""recon_kb_verify.py — GATE A1-A6 verifier for the mission-pivot deep KB (gamedata/).

Loads every Part-A file and asserts the structural gates from TEAM_PLANNER_DESIGN.md. Pure disk checks
(no emulator) — the RAM spot-check for rosters (A1) is a separate live step (Koga already VERIFIED shift 9).
Run: .venv/Scripts/python.exe -u pokemon_agent/recon_kb_verify.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GD = os.path.join(HERE, "gamedata")

FAILS = []
def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(msg)
    print(f"  [{tag}] {msg}")
    return cond

def load(name):
    with open(os.path.join(GD, name), encoding="utf-8") as f:
        return json.load(f)

print("=== GATE A1: frlg_rosters.json ===")
r = load("frlg_rosters.json")
gyms = r["gyms"]; e4 = r["e4"]
for leader in ["Brock", "Misty", "Lt. Surge", "Erika", "Koga", "Sabrina", "Blaine", "Giovanni"]:
    team = gyms.get(leader) or []
    check(len(team) >= 1 and all("species" in m and "level" in m for m in team),
          f"gym {leader}: {len(team)} mons, each has species+level")
for seat in ["Lorelei", "Bruno", "Agatha", "Lance"]:
    team = e4.get(seat) or []
    check(len(team) == 5 and all("species" in m and "level" in m for m in team),
          f"e4 {seat}: {len(team)} mons")
champ = r["champion"]["team"]
check(len(champ) == 6, f"champion: {len(champ)} mons (bulbasaur branch)")
check(any(m["species"] == "charizard" for m in champ), "champion has Charizard ace (bulbasaur branch)")

print("=== GATE A2: frlg_evolutions.json ===")
ev = load("frlg_evolutions.json")
ev_species = {k: v for k, v in ev.items() if not k.startswith("_")}
check(len(ev_species) >= 40, f"{len(ev_species)} evolution lines (>=40)")
for s in ["bulbasaur", "charmander", "squirtle", "abra", "kadabra", "eevee", "growlithe", "diglett"]:
    check(s in ev_species, f"has {s}")
check(ev["kadabra"]["method"] == "trade", "kadabra->alakazam is a trade")
check(ev["eevee"]["method"] == "stone" and "options" in ev["eevee"], "eevee stone-branch present")
check(ev["growlithe"]["method"] == "stone" and ev["growlithe"]["stone"] == "fire-stone", "growlithe fire-stone")
stone_evos = [k for k, v in ev_species.items() if v.get("method") == "stone"]
trade_evos = [k for k, v in ev_species.items() if v.get("method") == "trade"]
check(len(stone_evos) >= 5, f"{len(stone_evos)} stone evos")
check(len(trade_evos) >= 3, f"{len(trade_evos)} trade evos")

print("=== GATE A3: frlg_learnsets.json ===")
ls = load("frlg_learnsets.json")
ls_species = {k: v for k, v in ls.items() if not k.startswith("_")}
check(len(ls_species) >= 18, f"{len(ls_species)} learnset species (~20)")
for s in ["venusaur", "kadabra", "dugtrio", "arcanine", "lapras", "snorlax"]:
    rec = ls_species.get(s) or {}
    check("level_up" in rec and "tm" in rec, f"{s} has level_up + tm compat")
# coverage-move compat must be present where the plan needs it
check("TM13" in ls["lapras"]["tm"], "lapras can learn Ice Beam (TM13)")
check("TM24" in ls["lapras"]["tm"], "lapras can learn Thunderbolt (TM24)")
check("TM26" in ls["snorlax"]["tm"], "snorlax can learn Earthquake (TM26)")
check("TM26" in ls["dugtrio"]["tm"], "dugtrio can learn Earthquake (TM26)")
check("TM29" in ls["kadabra"]["tm"], "kadabra can learn Psychic (TM29)")
check("TM35" in ls["arcanine"]["tm"], "arcanine can learn Flamethrower (TM35)")

print("=== GATE A4: frlg_tms.json ===")
tm = load("frlg_tms.json")
tm_ids = [k for k in tm if k.startswith("TM")]
hm_ids = [k for k in tm if k.startswith("HM")]
check(len(tm_ids) == 50, f"{len(tm_ids)} TMs (==50)")
check(len(hm_ids) == 7, f"{len(hm_ids)} HMs (FRLG has 7, no Dive/HM08)")
for k in ["TM13", "TM24", "TM26", "TM29", "TM30", "TM35"]:
    check(tm[k].get("where"), f"{k} {tm[k]['name']} has a 'where'")

print("=== GATE A5: frlg_encounters.json ===")
enc = load("frlg_encounters.json")
keepers = enc["keepers"]; areas = enc["areas"]
# every acquire species in the primary team plan must resolve to a location
tp = load("frlg_team_plan.json")
arch = tp["archetypes"][0]
unresolved = []
for slot in arch["slots"]:
    acq = slot.get("acquire") or {}
    sp = acq.get("species")
    if not sp:
        continue  # gift/static handled by 'where' string
    if sp not in keepers and not any(sp in [e["species"] for e in meth]
                                     for area in areas.values() for meth in area.values()):
        unresolved.append(sp)
check(not unresolved, f"every catchable plan keeper resolves to a location (unresolved: {unresolved})")
check("abra" in keepers and "diglett" in keepers and "growlithe" in keepers, "key keepers indexed")

print("=== GATE A6: frlg_team_plan.json ===")
check(len(arch["slots"]) == 6, f"balanced-classic has {len(arch['slots'])} slots (==6)")
for slot in arch["slots"]:
    acq = slot.get("acquire") or {}
    ok = bool(acq.get("where")) and ("by_badge" in acq)
    check(ok, f"slot {slot['role']}: acquire has where + by_badge")
# coverage: every gym+E4 weakness answer type must be in coverage_target OR a threat_answer
strat = load("frlg_strategy.json")["threats"]
cov = set(arch["coverage_target"])
ta = arch["threat_answers"]
missing = []
for name, rec in strat.items():
    ans = set(rec.get("answer_types") or [])
    slot_ans = set(ta.get(name, {}).get("answer_types") or [])
    if not ans:
        continue  # champion 'mixed'
    if not (ans & cov) and not (ans & slot_ans):
        missing.append((name, sorted(ans)))
check(not missing, f"every threat has a covered answer type (gaps: {missing})")
check(len(arch["level_milestones"]) >= 12, f"{len(arch['level_milestones'])} level milestones")

print()
if FAILS:
    print(f"!!! {len(FAILS)} GATE FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("*** ALL GATES A1-A6 PASS ***")
