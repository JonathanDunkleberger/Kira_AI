"""tower_strike.py — THE POKÉMON TOWER STRIKE, in-loop (night shift #7).

The general questline door-hint lands her INSIDE the Tower and even go-deepers toward 2F, but the
climb is a scripted RITUAL the blind tour can't do: GARY on 2F (his body wedges the floor; his
battle needs the multi-box talk escalation), the Marowak GHOST coord-trigger on 6F (a wild fight
once the Silph Scope is in bag), a 7F grunt gauntlet, and Mr. Fuji's two-stage give. She also LOSES
the Gary/channeler war of attrition with a worn lead + thin bench (run6). This is a faithful in-loop
port of the champion's proven recon_tower.py (climb state machine + attrition heal-valve), driven by
the live `camp`, hooked into _questline_strike for the poke_flute errand (step.success==('item',350)).
FireRed coords isolated here (rule 14 — portability debt).

Ground truth (disasm pret PokemonTower_1F..7F map.json + the champion's proven runs):
  1F (1,88): up (18,9). 2F (1,89): GARY (16,5), up (4,10). 3F (1,90): up (18,10). 4F (1,91): up
  (4,10). 5F (1,92): up (18,10). 6F (1,93): MAROWAK ghost coord-trigger flanks up (11,16) — Scope in
  bag = a normal wild fight. 7F (1,94): grunts (9,10)/(13,8)/(9,6) sight-4; MR. FUJI (11,4) talk ->
  auto-warp to his house -> POKE FLUTE (item 350; flag FLAG_GOT_POKE_FLUTE 0x23D). Success = item 350.
"""
import os

import field_moves as fm
import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Pokémon-Tower fact table (game-knowledge layer; rule 14 portability debt) ────────────
LAVENDER = (3, 4)
TOWER = [(1, 88), (1, 89), (1, 90), (1, 91), (1, 92), (1, 93), (1, 94)]  # 1F..7F
TOWER_MAPS = set(TOWER)
GARY_FRONT = (16, 6)               # rival at (16,5) on 2F, face UP
FUJI_FRONT = (11, 5)               # Fuji at (11,4) on 7F, face UP
GRUNTS_7F = ((9, 10), (13, 8), (9, 6))
POKE_FLUTE, FLAG_FLUTE = 350, 0x23D


class TowerStrike:
    def __init__(self, camp, log, dbg_dir=None):
        self.camp = camp
        self.b = camp.b
        self.log = log
        self.dbg = dbg_dir
        if dbg_dir:
            try:
                os.makedirs(dbg_dir, exist_ok=True)
            except Exception:
                self.dbg = None

    def key_items(self):
        return ht.pocket_items(self.b, ht.KEY_ITEMS_OFF, 30)

    def got_flute(self):
        return POKE_FLUTE in self.key_items() or fm.read_flag(self.b, FLAG_FLUTE)

    def lead_frac(self):
        cur = self.b.rd16(ram.GPLAYER_PARTY + 0x56)
        mx = self.b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def drain(self, max_a=40, key="A"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press(key, 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    def goto(self, tile, label):
        b, camp = self.b, self.camp
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120)
        if st.in_battle(b):
            self.log(f"   [{label}] battle en route -> {camp.battle_runner()}")
            self.drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120)
        if r != "arrived" and not camp._step_to(tile):
            return False
        return tuple(tv.coords(b) or ()) == tile

    def engage(self, front, face, label, drains=1):
        """Stand at `front`, face `face`, press A; run any battle via the OBSERVED runner
        (strat/rival/evolution recording). `drains` = box-advances for a multi-box talk escalation
        (Gary/Fuji) so the fight actually starts."""
        b, camp = self.b, self.camp
        if not self.goto(front, label):
            self.log(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if st.in_battle(b):
                self.log(f"   [{label}] battle -> {camp.battle_runner()}")
                self.drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    self.drain()
                    if st.in_battle(b):        # the talk escalated into the fight (rival class)
                        self.log(f"   [{label}] battle -> {camp.battle_runner()}")
                        self.drain()
                        return "battled"
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def enter_to(self, dest, label):
        """Warp to `dest` map via ANY warp on the current map whose table dest matches (live
        read_warps + enter_warp's full ritual stack)."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            self.log(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                self.log(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):
                self.log(f"   [{label}] battle on approach -> {camp.battle_runner()}")
                self.drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def climb_and_get_flute(self):
        """The Tower climb state machine (ported from recon_tower.py). Assumes she's IN the Tower
        or in Lavender. Returns 'got_flute' | 'failed'."""
        b, camp, L = self.b, self.camp, self.log

        # heal to full in Lavender before the climb (Gary + channelers wear a worn lead down)
        if tuple(tv.map_id(b)) == LAVENDER and self.lead_frac() < 0.95:
            hr = camp.heal_nearest()
            L(f"   tower: pre-climb heal (lead {self.lead_frac():.0%}) -> {hr}")

        gary_done = fm.read_flag(b, FLAG_FLUTE)  # if flute already, gary irrelevant
        wedges = {}
        for _step in range(400):
            if self.got_flute():
                return "got_flute"
            here = tuple(tv.map_id(b))
            if here == TOWER[6]:
                break
            # ATTRITION VALVE: wild ghosts + Gary chip the lead; descend + heal + re-climb.
            if here in TOWER and self.lead_frac() < 0.5:
                L(f"   tower: lead at {self.lead_frac():.0%} — descending to heal (from {here})")
                if here != TOWER[0]:
                    self.enter_to(TOWER[TOWER.index(here) - 1], "heal-descent")
                else:
                    self.enter_to(LAVENDER, "tower-exit")
                continue
            if here == LAVENDER:
                if self.lead_frac() < 0.95:
                    camp.heal_nearest()
                if not self.enter_to(TOWER[0], "tower-(re)enter"):
                    self.snap("25_enter_fail")
                    L("!! tower: couldn't enter the Tower from Lavender — abort")
                    return "failed"
                continue
            if here not in TOWER:
                L(f"   tower: off-route at {here} (heal interior?) — exiting to the overworld")
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
                if tuple(tv.map_id(b)) == here:
                    self.snap("26_offroute")
                    L(f"!! tower: stuck off-route at {here}@{tv.coords(b)} — abort")
                    return "failed"
                continue
            idx = TOWER.index(here)
            if here == TOWER[1] and not gary_done:
                rg = self.engage(GARY_FRONT, "UP", "gary", drains=3)
                L(f"   GARY on 2F -> {rg}")
                gary_done = True
                self.snap("20_post_gary")
                continue                             # re-dispatch (a loss bounced her; a win climbs)
            if not self.enter_to(TOWER[idx + 1], f"floor{idx + 1}-up"):
                if tuple(tv.map_id(b)) == here:      # still on the floor — wild-interrupt flake?
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        self.snap(f"30_floor{idx + 1}_fail")
                        L(f"!! tower: climb wedged x3 on floor {idx + 1} "
                          f"(at {tv.map_id(b)}@{tv.coords(b)})")
                        return "failed"
                    self.drain()
                    L(f"   tower: floor {idx + 1} retry {wedges[here]}/3")
                continue
            wedges.pop(here, None)

        if tuple(tv.map_id(b)) != TOWER[6]:
            L(f"!! tower: climb never reached 7F (at {tv.map_id(b)}@{tv.coords(b)}) — abort")
            return "failed"

        # 7F: the grunt gauntlet (LoS, travel fights through), then MR. FUJI
        self.snap("40_7f")
        for gx, gy in GRUNTS_7F:
            rg = self.engage((gx, gy + 1), "UP", f"grunt{gx}x{gy}")
            L(f"   7F grunt ({gx},{gy}) -> {rg}")
        r = self.engage(FUJI_FRONT, "UP", "fuji", drains=6)
        L(f"   FUJI -> {r}; now {tv.map_id(b)}@{tv.coords(b)}")
        for _ in range(600):                      # the scripted warp to his house + the give
            b.run_frame()
            if dd_box(b):
                b.press("A", 8, 12, camp.render, owner="agent")
        self.drain()
        if not self.got_flute() and tuple(tv.map_id(b)) != TOWER[6]:
            # in his house — he may need one more talk; he stands beside the arrival
            npcs = sorted(camp.trav._npc_tiles())
            L(f"   in Fuji's house ({tv.map_id(b)}@{tv.coords(b)}) — talking; npcs {npcs}")
            cur = tuple(tv.coords(b) or (0, 0))
            for nt in sorted(npcs, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))[:3]:
                for front, face in (((nt[0], nt[1] + 1), "UP"), ((nt[0] - 1, nt[1]), "RIGHT"),
                                    ((nt[0] + 1, nt[1]), "LEFT"), ((nt[0], nt[1] - 1), "DOWN")):
                    self.engage(front, face, "fuji-house", drains=4)
                    if self.got_flute():
                        break
                if self.got_flute():
                    break

        ok = self.got_flute()
        L(f"   POKE FLUTE: item={POKE_FLUTE in self.key_items()} "
          f"flag={fm.read_flag(b, FLAG_FLUTE)} (key_items={self.key_items()})")
        self.snap("50_final")
        return "got_flute" if ok else "failed"


def run_strike(camp, log, dbg_dir=None):
    """Run the Pokémon Tower strike from inside the Tower / Lavender, in ONE call. Returns:
      'got_flute' — Poké Flute in Key Items (climb complete; she ends in/near Lavender).
      'not_here'  — no strike applies from here (caller falls through to the general layer).
      'failed'    — strike aborted mid-way (caller surfaces / recovery reacts)."""
    b = camp.b
    ts = TowerStrike(camp, log, dbg_dir=dbg_dir)
    if ts.got_flute():
        return "not_here"                          # already have it — nothing to strike
    here = tuple(tv.map_id(b))
    if here not in (TOWER_MAPS | {LAVENDER}):
        return "not_here"
    return ts.climb_and_get_flute()
