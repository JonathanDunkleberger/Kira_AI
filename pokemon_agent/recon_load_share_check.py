#!/usr/bin/env python
"""NS23 decision-check for the LOAD-SHARE lever (battle_agent._best_switch_slot).

Pure offline type-math: a mock bridge serves a fake party so we can assert the switch DECISION without
booting the emulator. Verifies (1) the fix rotates a CRITICAL SE active to a healthy SE partner, (2) it
NEVER reintroduces the SE<->non-SE churn the anti-churn rule kills, (3) flag-OFF is byte-inert (the
original `return None` for an SE active), and (4) the non-SE triggers are untouched.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import battle_agent
import pokemon_state as st
import firered_ram as ram

# move_id -> (type, power)
MOVES = {1: ("water", 95), 2: ("ice", 95), 3: ("grass", 55),
         4: ("normal", 35), 5: ("normal", 85), 6: ("psychic", 65)}


class MockBridge:
    def __init__(self, party):
        self.party = party           # list of dicts: species, level, hp, maxhp, move_ids

    def rd8(self, addr):
        if addr == ram.GPLAYER_PARTY_CNT:
            return len(self.party)
        off = addr - ram.GPLAYER_PARTY
        s, o = off // 100, off % 100
        if 0 <= s < len(self.party) and o == 0x54:
            return self.party[s]["level"]
        return 0

    def rd16(self, addr):
        off = addr - ram.GPLAYER_PARTY
        s, o = off // 100, off % 100
        if 0 <= s < len(self.party):
            if o == 0x56:
                return self.party[s]["hp"]
            if o == 0x58:
                return self.party[s]["maxhp"]
        return 0


def _patch_st():
    st.read_party_species = lambda b, s: b.party[s]["species"] if s < len(b.party) else 0
    st.read_party_moves = lambda b, s: b.party[s]["move_ids"] if s < len(b.party) else []
    st.move_info = lambda b, mid: MOVES.get(mid, ("normal", 0))


def _agent(party):
    _patch_st()
    a = battle_agent.BattleAgent.__new__(battle_agent.BattleAgent)
    a.b = MockBridge(party)
    a.log = lambda *args, **kw: None
    return a


def _ours(species, types, level, hp, maxhp, moves):
    return {"species": species, "types": types, "level": level, "hp": hp, "maxhp": maxhp,
            "moves": [{"id": i, "type": t, "power": p, "pp": 10} for (i, t, p) in moves]}


# Species/type shorthands (real ids for species_types, which stays live)
LAPRAS, VENU, KADA, RATT = 131, 3, 64, 19

RHYDON = {"species": 112, "types": ["ground", "rock"], "level": 60}   # Venu RazorLeaf 4x, Lapras Surf 4x
PIDGEOT = {"species": 18, "types": ["normal", "flying"], "level": 61}  # Lapras IceBeam 2x, Venu RazorLeaf 0.5x
GENGAR = {"species": 94, "types": ["ghost", "poison"], "level": 56}    # Kadabra Psychic 2x, Ghost hits Psychic 2x

cases = []


def check(name, party, state, share_on, healthy_frac, expect):
    battle_agent.BATTLE_LOAD_SHARE = share_on
    battle_agent.SWITCH_SHARE_HEALTHY_FRAC = healthy_frac
    got = _agent(party).best_switch_wrap(state) if False else _agent(party)._best_switch_slot(state)
    ok = (got == expect)
    cases.append((name, ok, expect, got))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expect={expect} got={got}")


# Lapras active with Surf(water) + IceBeam(ice) — SE (>=2x) on Rhydon & Pidgeot
lap_moves = [(1, "water", 95), (2, "ice", 95), (5, "normal", 85)]
# Kadabra active with Psybeam(psychic) — 2x on Gengar (poison)
kad_moves = [(6, "psychic", 65)]

# --- Scenario A: THE FIX — critical SE Lapras, healthy SE Venusaur (both 4x on Rhydon) -> rotate to Venu (slot1)
partyA = [{"species": LAPRAS, "level": 60, "hp": 40, "maxhp": 200, "move_ids": [1, 2, 5]},   # 20% crit
          {"species": VENU, "level": 71, "hp": 180, "maxhp": 220, "move_ids": [3, 4]},        # 82% healthy, RazorLeaf 4x
          {"species": RATT, "level": 9, "hp": 20, "maxhp": 25, "move_ids": [4]}]
stA = {"enemy": RHYDON, "ours": _ours(LAPRAS, ["water", "ice"], 60, 40, 200, lap_moves)}
check("A fix fires (crit SE -> healthy SE partner)", partyA, stA, True, 0.5, 1)
check("A flag OFF -> inert (None)", partyA, stA, False, 0.5, None)

# --- Scenario B: churn-guard — critical SE Lapras but the only reserve is a NON-SE Venu (Pidgeot: RazorLeaf 0.5x)
stB = {"enemy": PIDGEOT, "ours": _ours(LAPRAS, ["water", "ice"], 60, 40, 200, lap_moves)}
check("B churn-guard (crit SE, no SE partner -> stay None)", partyA, stB, True, 0.5, None)

# --- Scenario C: SE active HEALTHY -> stays regardless (not critical)
partyC = [{"species": LAPRAS, "level": 60, "hp": 160, "maxhp": 200, "move_ids": [1, 2, 5]},   # 80%
          {"species": VENU, "level": 71, "hp": 180, "maxhp": 220, "move_ids": [3, 4]}]
stC = {"enemy": RHYDON, "ours": _ours(LAPRAS, ["water", "ice"], 60, 160, 200, lap_moves)}
check("C healthy SE active -> stay None", partyC, stC, True, 0.5, None)

# --- Scenario D: THE ANTI-CHURN REGRESSION — Kadabra Psybeam 2x into Gengar (active_bad also true), only
#     reserve is a 0.5x Venusaur. Even at critical HP the fix must NOT rotate (no SE partner) -> None.
partyD = [{"species": KADA, "level": 50, "hp": 15, "maxhp": 120, "move_ids": [6]},            # 12% crit
          {"species": VENU, "level": 71, "hp": 200, "maxhp": 220, "move_ids": [3, 4]}]         # RazorLeaf 1x on Gengar
stD = {"enemy": GENGAR, "ours": _ours(KADA, ["psychic"], 50, 15, 120, kad_moves)}
check("D anti-churn preserved (crit Kadabra, non-SE Venu -> None)", partyD, stD, True, 0.5, None)
check("D flag OFF -> None", partyD, stD, False, 0.5, None)

# --- Scenario E: healthy-floor — critical SE Lapras, the SE Venu partner is at 40% (< 0.5 floor) -> None
partyE = [{"species": LAPRAS, "level": 60, "hp": 40, "maxhp": 200, "move_ids": [1, 2, 5]},
          {"species": VENU, "level": 71, "hp": 88, "maxhp": 220, "move_ids": [3, 4]}]           # 40% not healthy
stE = {"enemy": RHYDON, "ours": _ours(LAPRAS, ["water", "ice"], 60, 40, 200, lap_moves)}
check("E healthy-floor (SE partner at 40% < floor -> None)", partyE, stE, True, 0.5, None)
check("E lower floor 0.35 -> now rotates to slot1", partyE, stE, True, 0.35, 1)

# --- Scenario F: non-SE trigger UNTOUCHED — Venu active 0.25x into Charizard-like, healthy SE Lapras reserve
#     -> trigger 2 fields the specialist (slot1). Confirms the SE-active block doesn't shadow the non-SE path.
CHARIZ = {"species": 6, "types": ["fire", "flying"], "level": 63}
venu_vs_char = [(3, "grass", 55)]   # grass vs fire/flying = 0.25x
partyF = [{"species": VENU, "level": 71, "hp": 200, "maxhp": 220, "move_ids": [3]},
          {"species": LAPRAS, "level": 60, "hp": 160, "maxhp": 200, "move_ids": [1, 2]}]   # Surf 2x on Charizard
stF = {"enemy": CHARIZ, "ours": _ours(VENU, ["grass", "poison"], 71, 200, 220, venu_vs_char)}
check("F non-SE trigger2 still fields specialist (slot1)", partyF, stF, True, 0.5, 1)
check("F non-SE unaffected by flag OFF (slot1)", partyF, stF, False, 0.5, 1)

n_pass = sum(1 for _, ok, _, _ in cases if ok)
print(f"\n{'ALL PASS' if n_pass == len(cases) else 'FAIL'} — {n_pass}/{len(cases)}")
raise SystemExit(0 if n_pass == len(cases) else 1)
