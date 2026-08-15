"""ROM-free Lorelei lead / Zapdos-gun / revive-forget checks (2026-08-13).

Proves we never assume Zapdos has Thunderbolt, and we lead Blastoise when he doesn't.
RUN: python -u recon_zapdos_lorelei_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import e4_strike as e4  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print(f"FAIL  {msg}")
    else:
        print(f"ok    {msg}")


ZAP, MOLT, BLAST, ART = e4.ZAPDOS_SP, e4.MOLTRES_SP, e4.BLASTOISE_SP, e4.ARTICUNO_SP
DEWGONG, JYNX, GYARADOS = 87, 124, 130
# Wild Power-Plant Zapdos
WILD = [86, 97, 197, 65]  # TWave, Agility, Detect, Drill Peck
WITH_BOLT = [85, 97, 197, 65]

check(not e4.moveset_has_electric_damage(WILD), "wild Zapdos has NO Electric damage")
check(e4.moveset_has_electric_damage(WITH_BOLT), "Thunderbolt id 85 counts as Electric damage")
check(not e4.move_is_electric_damage(86, "electric", 0), "Thunder Wave is NOT a gun")
check(e4.move_is_electric_damage(85, "electric", 0), "Thunderbolt id counts even if power byte stale")
check(e4.zapdos_forget_idx(WILD) == 1, "forget Agility (idx 1), not Drill Peck")
check(e4.zapdos_forget_idx([65, 65, 65, 65]) is None, "refuse to overwrite Drill Peck-only set")

check(BLAST not in e4.lorelei_banned_species(False),
      "Blastoise ALLOWED when Zapdos has no gun")
check(BLAST in e4.lorelei_banned_species(True),
      "Blastoise banned when Zapdos has the gun")
check(ART in e4.lorelei_banned_species(False), "Articuno always banned")

# Waters, no gun: stay on Blastoise / switch TO Blastoise, do not hold Zapdos
r = e4.lorelei_inbattle_switch(ZAP, ["water", "ice"], DEWGONG, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2)
check(r == 2, f"no-gun Zapdos vs Dewgong -> switch to Blastoise (got {r})")
r = e4.lorelei_inbattle_switch(BLAST, ["water", "ice"], DEWGONG, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2)
check(r == "stay", f"Blastoise vs Dewgong stays (got {r})")

# Waters, WITH gun: Zapdos holds
r = e4.lorelei_inbattle_switch(ZAP, ["water"], DEWGONG, zap_slot=0, molt_slot=3,
                               zap_has_electric=True, blast_slot=2)
check(r == "stay", f"gun Zapdos vs Dewgong stays (got {r})")
r = e4.lorelei_inbattle_switch(BLAST, ["water"], DEWGONG, zap_slot=0, molt_slot=3,
                               zap_has_electric=True, blast_slot=2)
check(r == 0, f"gun: Blastoise vs Dewgong -> switch to Zapdos (got {r})")

# Jynx: L75 Blastoise Surf stays / is preferred. Do NOT send L50 Moltres into Kiss.
r = e4.lorelei_inbattle_switch(BLAST, ["ice", "psychic"], JYNX, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2)
check(r == "stay", f"Blastoise vs Jynx stays (got {r})")
r = e4.lorelei_inbattle_switch(ZAP, ["ice", "psychic"], JYNX, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2)
check(r == 2, f"Zapdos vs Jynx -> Blastoise not Moltres (got {r})")
r = e4.lorelei_inbattle_switch(MOLT, ["ice", "psychic"], JYNX, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2)
check(r == 2, f"Moltres vs Jynx -> switch to Blastoise (got {r})")
r = e4.lorelei_inbattle_switch(ZAP, ["ice", "psychic"], JYNX, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=None)
check(r == 3, f"Jynx, Blastoise down -> Moltres (got {r})")

src = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
check("fight instead (Thunderbolt)" not in src,
      "battle_agent does not hardcode Thunderbolt as Zapdos's move")
check("fight instead (Electric STAB)" in src,
      "hold message is Electric STAB, not Thunderbolt")

# ── 14:00 Lapras wipe: Surf-into-Water banned; EQ dry → not Surf ────────────────
LAPRAS = 131
SURF, EQ, SKULL, BITE = 57, 89, 130, 44
blast_moves = [
    {"id": EQ, "name": "Earthquake", "type": "ground", "power": 100, "pp": 10},
    {"id": SURF, "name": "Surf", "type": "water", "power": 95, "pp": 15},
    {"id": SKULL, "name": "Skull Bash", "type": "normal", "power": 100, "pp": 15},
    {"id": BITE, "name": "Bite", "type": "dark", "power": 60, "pp": 25},
]
check(e4.e4_banned_move(SURF, "water", BLAST, ["water", "ice"], LAPRAS),
      "Surf-into-Water banned for Blastoise in E4")
check(not e4.e4_banned_move(EQ, "ground", BLAST, ["water", "ice"], LAPRAS),
      "Earthquake vs Lapras is NOT banned")
check(e4.e4_preferred_move_index(blast_moves, BLAST, ["water", "ice"], LAPRAS) == 0,
      "EQ with PP is the Lapras wincon index")
dry = [dict(m) for m in blast_moves]
dry[0]["pp"] = 0
check(e4.e4_eq_pp_dry(dry, BLAST, ["water", "ice"], LAPRAS),
      "Earthquake PP=0 vs Lapras is EQ-dry (Surf still having PP does not hide it)")
filt = e4.e4_filter_moves(dry, BLAST, ["water", "ice"], LAPRAS)
check(filt[1]["pp"] == 0, "filtered: Surf PP zeroed when EQ is dry")
check(filt[2]["pp"] > 0 and filt[3]["pp"] > 0,
      "filtered: Skull Bash and Bite still have PP")
import pokemon_policy as pol
idx, desc, _low = pol.choose_move(filt, ["water", "ice"], 1.0, our_types=["water"])
check(filt[idx]["id"] != SURF, f"EQ dry -> picker does not Surf Lapras (got {desc})")
check(e4.e4_banned_move(58, "ice", ART, ["water", "ice"], LAPRAS),
      "Articuno Ice Beam into Lapras is banned (0.25x)")
check(not e4.e4_banned_move(58, "ice", ART, ["water", "flying"], GYARADOS),
      "Articuno Ice Beam vs Gyarados is NOT banned (1x Water/Flying; live 21:08 Agility whiteout)")
check(e4.articuno_ice_into_water_ice(["water", "ice"], LAPRAS),
      "Lapras is the Ice-into-Water/Ice ban")
check(not e4.articuno_ice_into_water_ice(["water", "flying"], GYARADOS),
      "Gyarados is NOT the Ice-into-Water/Ice ban")
art_gyara = [
    {"id": 54, "name": "Mist", "type": "ice", "power": 0, "pp": 30},
    {"id": 97, "name": "Agility", "type": "psychic", "power": 0, "pp": 30},
    {"id": 170, "name": "Mind Reader", "type": "normal", "power": 0, "pp": 5},
    {"id": 58, "name": "Ice Beam", "type": "ice", "power": 95, "pp": 10},
]
check(e4.e4_preferred_move_index(art_gyara, ART, ["water", "flying"], GYARADOS) == 3,
      "Articuno vs Gyarados: Ice Beam, never Agility")
filt_gyara = e4.e4_filter_moves(art_gyara, ART, ["water", "flying"], GYARADOS)
check(filt_gyara[3]["pp"] == 10,
      "Ice Beam PP stays live vs Gyarados (fail-open, not Lapras-style zero)")
check(e4.e4_between_room_revive_species("Lance")[0] == ZAP,
      "scarce Revive before Lance stands Zapdos up first (not Blastoise)")
bruno_fight = [
    {"id": 130, "name": "Skull Bash", "type": "normal", "power": 100, "pp": 15},
    {"id": 57, "name": "Surf", "type": "water", "power": 95, "pp": 14},
    {"id": 89, "name": "Earthquake", "type": "ground", "power": 100, "pp": 4},
]
check(e4.e4_preferred_move_index(bruno_fight, BLAST, ["fighting"], 107, seat="Bruno") == 1,
      "Bruno Hitmon: STAB Surf, save EQ PP for Arbok (live 21:08 EQ-dry)")

# ── Lance: Zapdos-with-gun vs Gyarados; Articuno vs dragons; never Moltres ──
DRAGONITE = 149
check(e4.PREFERRED_LEAD_IDS["Lance"][0] == ZAP,
      "Lance preferred lead #1 is Zapdos (4x Gyarados if he has the gun)")
check(e4.PREFERRED_LEAD_IDS["Lance"][1] == ART,
      "Lance preferred lead #2 is Articuno (Ice vs the dragon wave)")
check(MOLT not in e4.PREFERRED_LEAD_IDS["Lance"],
      "Lance lead list does not include Moltres (Fire 0.5x Gyarados)")
check(e4.lance_foe_is_gyarados(["water", "flying"], GYARADOS),
      "Gyarados counts as Lance's Water/Flying")
check(e4.lance_foe_is_dragon(["dragon", "flying"], DRAGONITE),
      "Dragonite counts as a Lance dragon")
r = e4.lance_inbattle_switch(ART, ["water", "flying"], GYARADOS, zap_slot=0, art_slot=1,
                             zap_has_electric=True, blast_slot=2)
check(r == 0, f"Articuno vs Gyarados + gun -> Zapdos (got {r})")
r = e4.lance_inbattle_switch(ZAP, ["water", "flying"], GYARADOS, zap_slot=0, art_slot=1,
                             zap_has_electric=True, blast_slot=2)
check(r == "stay", f"gun Zapdos vs Gyarados stays (got {r})")
r = e4.lance_inbattle_switch(MOLT, ["water", "flying"], GYARADOS, zap_slot=0, art_slot=1,
                             zap_has_electric=True, blast_slot=2)
check(r == 0, f"Moltres vs Gyarados + gun -> Zapdos (got {r})")
r = e4.lance_inbattle_switch(MOLT, ["water", "flying"], GYARADOS, zap_slot=0, art_slot=1,
                             zap_has_electric=False, blast_slot=2)
check(r == 1, f"no-gun Moltres vs Gyarados -> Articuno not Drill Peck (got {r})")
r = e4.lance_inbattle_switch(ART, ["dragon", "flying"], DRAGONITE, zap_slot=0, art_slot=1,
                             zap_has_electric=True, blast_slot=2)
check(r == "stay", f"Articuno vs Dragonite stays (got {r})")
pref = e4.e4_force_send_pref("Lance", ["water", "flying"], GYARADOS, zap_has_electric=True)
check(pref[0] == ZAP, f"faint-send vs Gyarados + gun is Zapdos (got {pref})")
pref = e4.e4_force_send_pref("Lorelei", ["water", "ice"], LAPRAS, zap_has_electric=False)
check(pref[0] == BLAST, f"faint-send vs Lapras no-gun is Blastoise (got {pref})")
check(ART not in pref, "faint-send vs Lapras never picks Articuno")
check(e4.e4_banned_move(SURF, "water", BLAST, ["dragon", "flying"], DRAGONITE),
      "Surf-into-Dragon banned for Blastoise (0.5x Dragonite)")
check(not e4.e4_banned_move(SKULL, "normal", BLAST, ["dragon", "flying"], DRAGONITE),
      "Skull Bash vs Dragonite is NOT banned")
src_e4 = open(os.path.join(_HERE, "e4_strike.py"), encoding="utf-8").read()
check("if dragon and sp == ARTICUNO_SP" in src_e4,
      "fainted Articuno is a Lance-dragon wincon revive")
src_ba = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
check("LAST-BODY revive kept" in src_ba,
      "E4 does not strip LAST-BODY insurance revive offers")
src_camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
check("trying the NEXT fainted" in src_camp,
      "e4-between-room revive fail continues to the next fainted (no 10-min abort)")

# ── Agatha wipe: Blastoise Surf vs Gengar; never Skull Bash / EQ; revive Blastoise ──
GENGAR = 94
check(e4.PREFERRED_LEAD_IDS["Agatha"][0] == BLAST,
      "Agatha preferred lead #1 is Blastoise (Surf vs Gengar)")
check(e4.PREFERRED_LEAD_IDS["Bruno"][0] == BLAST,
      "Bruno preferred lead #1 is Blastoise (Surf 4x Onix, not Articuno)")
ONIX, GOLBAT, ARBOK = 95, 42, 24
r = e4.bruno_inbattle_switch(ART, ["rock", "ground"], ONIX, blast_slot=2)
check(r == 2, f"Articuno vs Onix -> Blastoise (got {r})")
r = e4.bruno_inbattle_switch(BLAST, ["rock", "ground"], ONIX, blast_slot=2)
check(r == "stay", f"Blastoise vs Onix stays (got {r})")
r = e4.agatha_inbattle_switch(ART, ["poison", "flying"], GOLBAT, blast_slot=2,
                              zap_slot=1, zap_has_electric=True, frail={ART})
check(r == 1, f"frail Articuno vs Golbat -> healthy Zapdos (got {r})")
r = e4.agatha_inbattle_switch(BLAST, ["poison", "flying"], GOLBAT, blast_slot=2,
                              zap_slot=1, zap_has_electric=True, frail={ZAP})
check(r == "stay", f"Blastoise vs Golbat stays when Zapdos is frail (got {r})")
r = e4.agatha_inbattle_switch(ZAP, ["poison"], ARBOK, blast_slot=2,
                              zap_slot=1, zap_has_electric=True)
check(r == 2, f"Zapdos vs Arbok -> Blastoise EQ (got {r})")
r = e4.agatha_inbattle_switch(BLAST, ["poison"], ARBOK, blast_slot=2)
check(r == "stay", f"Blastoise vs Arbok stays (got {r})")
eq_arbok = [
    {"id": 130, "name": "Skull Bash", "type": "normal", "power": 100, "pp": 15},
    {"id": 57, "name": "Surf", "type": "water", "power": 95, "pp": 5},
    {"id": 89, "name": "Earthquake", "type": "ground", "power": 100, "pp": 10},
]
check(e4.e4_preferred_move_index(eq_arbok, BLAST, ["poison"], ARBOK, seat="Agatha") == 2,
      "Arbok: Earthquake is the 4x pick")
check(e4.e4_preferred_move_index(eq_arbok, BLAST, ["ghost", "poison"], GENGAR) == 1,
      "Gengar: Surf (EQ is 0x Levitate)")
check(e4.e4_banned_move(57, "water", BLAST, ["poison"], ARBOK),
      "Surf vs Arbok banned so Gengar still has a gun")
dry_eq = [
    {"id": 130, "name": "Skull Bash", "type": "normal", "power": 100, "pp": 15},
    {"id": 57, "name": "Surf", "type": "water", "power": 95, "pp": 2},
    {"id": 89, "name": "Earthquake", "type": "ground", "power": 100, "pp": 0},
]
check(e4.e4_preferred_move_index(dry_eq, BLAST, ["poison"], ARBOK, seat="Agatha") == 0,
      "Arbok + EQ dry: Skull Bash, save Surf for Gengar")
check(e4.e4_preferred_move_index(eq_arbok, BLAST, ["rock", "ground"], ONIX) == 1,
      "Onix: Surf 4x beats EQ 2x")
pref = e4.e4_force_send_pref("Bruno", ["rock", "ground"], ONIX)
check(pref[0] == BLAST, f"faint-send vs Onix is Blastoise (got {pref})")
check(ART not in pref, "faint-send vs Onix never picks Articuno")
check(e4.agatha_foe_is_ghost(["ghost", "poison"], GENGAR),
      "Gengar counts as an Agatha ghost")
check(not e4.e4_banned_move(SURF, "water", BLAST, ["ghost", "poison"], GENGAR),
      "Surf vs Gengar is NOT banned (the wincon)")
check(e4.e4_banned_move(SKULL, "normal", BLAST, ["ghost", "poison"], GENGAR),
      "Skull Bash vs Gengar IS banned (0x Ghost)")
check(e4.e4_banned_move(EQ, "ground", BLAST, ["ghost", "poison"], GENGAR),
      "Earthquake vs Gengar IS banned (0x Levitate)")
check(e4.e4_banned_move(EQ, "ground", BLAST, ["poison", "flying"], GOLBAT),
      "Earthquake vs Golbat IS banned (0x Flying — chat)")
check(e4.e4_banned_move(SKULL, "normal", e4.MOLTRES_SP, ["ghost", "poison"], GENGAR),
      "Skull Bash vs Gengar banned for ANY species (0x Ghost — chat)")
check(e4.e4_is_setup_move(97, 0), "Agility is a setup move")
check(e4.e4_is_setup_move(203, 0), "Endure is a setup move")
check(not e4.e4_is_setup_move(53, 95), "Flamethrower is not setup")
check(e4.e4_is_arbok_wincon(BLAST, ["poison"], ARBOK),
      "fainted Blastoise is the Arbok wincon revive")
molt_moves = [
    {"id": 97, "name": "Agility", "type": "psychic", "power": 0, "pp": 30},
    {"id": 203, "name": "Endure", "type": "normal", "power": 0, "pp": 10},
    {"id": 53, "name": "Flamethrower", "type": "fire", "power": 95, "pp": 14},
]
check(e4.e4_preferred_move_index(molt_moves, e4.MOLTRES_SP, ["poison"], ARBOK) == 2,
      "Moltres vs Arbok: Flamethrower, never Agility")
check(e4.e4_max_damage_index(molt_moves, e4.MOLTRES_SP, ["poison"], ARBOK) == 2,
      "max-damage never picks Agility when Flamethrower has PP")
check(e4.e4_is_ghost_wincon(BLAST, ["ghost", "poison"], GENGAR),
      "fainted Blastoise is Agatha-ghost wincon revive")
check(not e4.e4_is_ghost_wincon(ZAP, ["ghost", "poison"], GENGAR),
      "fainted Zapdos is NOT the Agatha-ghost wincon")
r = e4.agatha_inbattle_switch(ZAP, ["ghost", "poison"], GENGAR, blast_slot=2)
check(r == 2, f"Zapdos vs Gengar -> switch to Blastoise (got {r})")
r = e4.agatha_inbattle_switch(BLAST, ["ghost", "poison"], GENGAR, blast_slot=2)
check(r == "stay", f"Blastoise vs Gengar stays (got {r})")
check("keep Blastoise" in src_ba and "do not swap Zapdos back" in src_ba,
      "WHIFF RECOVERY does not swap Zapdos back onto Gengar")
check("species only" in src_ba and "no battle-wide potion block" in src_ba,
      "bag MUTE is species latch only, not battle-wide potion block")

# 18:44: frozen Blastoise is still the Lapras answer (Moltres died staying in)
r = e4.lorelei_inbattle_switch(MOLT, ["water", "ice"], LAPRAS, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2, locked={BLAST})
check(r == 2, f"Moltres vs Lapras -> frozen Blastoise (got {r})")
r = e4.lorelei_inbattle_switch(ART, ["water", "ice"], LAPRAS, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2, locked={BLAST})
check(r == 2, f"Articuno vs Lapras -> frozen Blastoise not Ice Beam (got {r})")
r = e4.lorelei_inbattle_switch(BLAST, ["water", "ice"], LAPRAS, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=2, locked={BLAST})
check(r == "stay", f"frozen Blastoise vs Lapras stays (got {r})")
r = e4.lorelei_inbattle_switch(ART, ["water", "ice"], LAPRAS, zap_slot=0, molt_slot=3,
                               zap_has_electric=False, blast_slot=None, locked={BLAST})
check(r == 3, f"Blastoise fainted: Articuno vs Lapras -> Moltres (got {r})")

# Articuno Ice Beam vs Lapras: filter NEVER fail-opens (18:44 whiteout)
art_only = [
    {"id": 58, "name": "Ice Beam", "type": "ice", "power": 95, "pp": 10},
    {"id": 97, "name": "Agility", "type": "psychic", "power": 0, "pp": 30},
    {"id": 54, "name": "Mist", "type": "ice", "power": 0, "pp": 30},
    {"id": 115, "name": "Reflect", "type": "psychic", "power": 0, "pp": 20},
]
filt_art = e4.e4_filter_moves(art_only, ART, ["water", "ice"], LAPRAS)
check(filt_art[0]["pp"] == 0,
      "Articuno Ice Beam PP zeroed vs Lapras even as only damaging move")

# Serebii squirtle-start Champion: Gyarados 59 then Arcanine 61
import json
_rost = json.load(open(os.path.join(_HERE, "gamedata", "frlg_rosters.json"), encoding="utf-8"))
_sq = _rost["champion"]["by_starter"]["squirtle"]
check(_sq[3]["species"] == "gyarados" and _sq[3]["level"] == 59,
      "Serebii squirtle-champ slot 4 is Gyarados 59")
check(_sq[4]["species"] == "arcanine" and _sq[4]["level"] == 61,
      "Serebii squirtle-champ slot 5 is Arcanine 61")

ba = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
check("tried rows" in ba or "BLIND-landed 0-HP" in ba,
      "revive scans every 0-HP row / blind-lands (not latch on first cursor miss)")
check("first miss" in ba and "no battle-wide latch" in ba,
      "first Revive cursor miss retries (no battle-wide latch)")
check("yanked him to Moltres" in ba or "STAY. Live 18:44" in ba,
      "MUST-LEAVE does not pull frozen Blastoise off Lorelei waters")
check("Surf is the ONLY hit" in ba or "EQ 0x Levitate" in ba,
      "MUST-LEAVE does not pull Blastoise off Gengar while Surf has PP")
check("def _bruno_switch_policy" in ba, "battle_agent has Bruno in-battle switch policy")

ba = open(os.path.join(_HERE, "battle_agent.py"), encoding="utf-8").read()
check("legendary bird" in ba and "NOT fodder" in ba,
      "ACE-FIRST does not skip birds (E4 legendary-not-fodder)")
check("_bag_blocked" in ba and "BAG-FAIL LATCH" in ba,
      "bag-open fail latches -- one fail, fight")
check("SLEEP-POTION BAN" in ba,
      "will not Super Potion a sleeping mon (live Lorelei Moltres loop)")
check("bag-count lagged" in ba or "count lagged" in ba,
      "Full Heal count-lag does not return no_effect after a real consume")
check("ping-pong OVERRIDE" in ba,
      "MUST-LEAVE may leave a sleeper even if ping-pong flagged")
check("party screen + active DOWN" in ba,
      "fainted active on party screen force-sends (no continue-A mash)")
check("E4 PREPLAN SWITCH" in ba,
      "E4 matchup switch fires BEFORE items (think once, then act)")
check("E4 RETRY" in ba and "menu miss" in ba,
      "E4 retries the wincon after a menu miss (never Surf->Skull Bash)")
check("ITEM-INSTINCT FORCED -> use_revive" in ba,
      "E4 wincon revive is forced (oracle cannot keep_fighting)")
check("NOT latching" in ba and "actuation miss" in ba,
      "heal actuation miss does not latch the species off items")
check("_switch_cap = 3 if self._league_seat()" in ba,
      "E4 matchup switch gets 3 tries (Zapdos->Blastoise vs Arbok)")
check("if self._party_screen() or self._bag_screen()" in ba,
      "_advance_text never A-mashes a party/bag menu")
check("E4 SEND:" in ba, "force-switch prefers E4 matchup species over highest level")
check("def _lance_switch_policy" in ba, "battle_agent has Lance in-battle switch policy")
check("TM34 Shock Wave" in open(os.path.join(_HERE, "e4_strike.py"), encoding="utf-8").read(),
      "teach_zapdos_electric also tries TM34 Shock Wave")
check("self._cure_blocked = set()" in ba,
      "CURE-BLOCK is per-species, not battle-wide")
check("e4-between-room" in open(os.path.join(_HERE, "e4_strike.py"), encoding="utf-8").read()
      or "BETWEEN-ROOM HEAL" in open(os.path.join(_HERE, "e4_strike.py"), encoding="utf-8").read(),
      "between-room heal exists")
e4s = open(os.path.join(_HERE, "e4_strike.py"), encoding="utf-8").read()
check("between_room_heal(camp, L" in e4s,
      "between-room heal is called from e4 room-clear")
check("scarce Revive ->" in e4s or "e4_between_room_revive_species" in e4s,
      "between-room scarce Revive prefers the next seat's answer")
check("party_wide=True" in e4s, "between-room heal is party-wide (birds included)")
_strat = json.load(open(os.path.join(_HERE, "gamedata", "frlg_strategy.json"), encoding="utf-8"))
check("Lovely Kiss" in _strat["threats"]["Lorelei"]["counter"],
      "Serebii: Jynx Lovely Kiss is in Lorelei counter")
check("Protect" in _strat["threats"]["Lorelei"]["counter"],
      "Serebii: Cloyster Protect PP-drain is in Lorelei counter")
check("Poke Flute" in _strat["threats"]["Agatha"]["counter"]
      or "Poké Flute" in _strat["threats"]["Agatha"]["counter"],
      "Serebii: Poké Flute wake is in Agatha counter")
check("Hyper Beam" in _strat["threats"]["Lance"]["counter"],
      "Serebii: Hyper Beam recharge is in Lance counter")

# ── 2026-08-14: thin-party Revive (1-2 alive + corpses) + Gary after Lance ──
thin_2v2 = [
    {"row": 0, "species": BLAST, "hp": 200, "level": 76},
    {"row": 1, "species": ZAP, "hp": 0, "level": 51},
    {"row": 2, "species": ART, "hp": 140, "level": 51},
    {"row": 3, "species": MOLT, "hp": 0, "level": 51},
]
check(e4.e4_thin_party_revive_index(thin_2v2, prefer_species=(ZAP, ART, BLAST, MOLT)) == 1,
      "2 alive / 2 dead: Revive prefers Zapdos (Lance Gyarados gun)")
thin_1v3 = [
    {"row": 0, "species": BLAST, "hp": 0, "level": 76},
    {"row": 1, "species": ZAP, "hp": 0, "level": 51},
    {"row": 2, "species": ART, "hp": 80, "level": 51},
    {"row": 3, "species": MOLT, "hp": 0, "level": 51},
]
check(e4.e4_thin_party_revive_index(thin_1v3, prefer_species=(ZAP, ART, BLAST, MOLT)) == 1,
      "1 alive / 3 dead: Revive still prefers Zapdos over L76 Blastoise")
thin_blast_only = [
    {"row": 0, "species": BLAST, "hp": 0, "level": 76},
    {"row": 1, "species": ZAP, "hp": 120, "level": 51},
    {"row": 2, "species": ART, "hp": 140, "level": 51},
]
check(e4.e4_thin_party_revive_index(thin_blast_only) == 0,
      "2 alive / Blastoise dead: Revive the corpse (not keep attacking)")
fat = [
    {"row": 0, "species": BLAST, "hp": 200, "level": 76},
    {"row": 1, "species": ZAP, "hp": 120, "level": 51},
    {"row": 2, "species": ART, "hp": 140, "level": 51},
    {"row": 3, "species": MOLT, "hp": 0, "level": 51},
]
check(e4.e4_thin_party_revive_index(fat) is None,
      "3 alive / 1 dead: thin-party does NOT force a Revive")
check(e4.e4_thin_party_revive_index([]) is None, "empty party is not a thin revive")
pref = e4.e4_force_send_pref("Gary", ["normal", "flying"], e4.PIDGEOT_SP)
check(pref[0] == ART, f"faint-send vs Pidgeot is Articuno (got {pref})")
pref = e4.e4_force_send_pref("Gary", ["water", "flying"], GYARADOS, zap_has_electric=True)
check(pref[0] == ZAP, f"faint-send vs champ Gyarados + gun is Zapdos (got {pref})")
pref = e4.e4_force_send_pref("Gary", ["grass", "poison"], e4.VENUSAUR_SP)
check(pref[0] == MOLT, f"faint-send vs Venusaur is Moltres (got {pref})")
pref = e4.e4_force_send_pref("Gary", ["ground", "rock"], e4.RHYDON_SP)
check(pref[0] == BLAST, f"faint-send vs Rhydon is Blastoise Surf (got {pref})")
check(e4.PREFERRED_LEAD_IDS["Gary"][0] == ART,
      "Gary preferred lead #1 is Articuno (Ice vs Pidgeot)")
check("THIN-PARTY" in ba and "spend the turn" in ba,
      "E4 does not strip thin-party Revive offers")
check("n_rev >= 1" in ba, "LAST-BODY insurance arms with Revive x1 (not x2)")
check("cap = 6 if e4" in ba, "E4 Revive cursor misses retry 6 times before latch")
check("def _revive_land_fainted_row" in ba,
      "Revive uses a dedicated party walk (not pixel confirm)")
check("never _party_focus" in ba,
      "Revive aim does not B-cancel the party screen")
check("never LEFT" in ba and "lead-LEFT is CANCEL" in ba,
      "Revive never LEFTs (lead-LEFT is CANCEL; live 12:00 Moltres no-effect)")
# ── THE BLINK-COUNTER LAW (2026-08-14, recon_revive_cursor2 — the six-unused-Revives wall) ──
# PARTY_CURSOR 0x02020777 is the SWITCH screen's cursor. On the ITEM-USE target screen it is the
# highlight's 3-phase BLINK counter (measured [2,1,0,2,1,0,...] while the highlight walked
# lead->1->2->3->CANCEL). The old walk asked for row N, the counter read N by coincidence on lap
# 1, the loop broke WITHOUT PRESSING ANYTHING, and the 0-HP guard refused A: live Lorelei burned
# 3 attempts with 6 Revives in the bag and a dead Zapdos on the floor. These assertions pin the
# measured design so nobody re-derives the walk off that byte.
check("ITEM_PARTY_CURSOR = 0x0203B0A9" in ba,
      "the measured item-use slotId (0x0203B0A9) is the one the item screen uses")
check("BLINK-COUNTER LAW" in ba,
      "the blink-counter law is recorded at the top of battle_agent")
_reader = ba.split("def _item_slot_id")[1][:1400]
check("_party_cursor_slot" in _reader and "ITEM_PARTY_CURSOR" in _reader,
      "item-use position reads ORANGE first, ITEM_PARTY_CURSOR second")
check("rd8(PARTY_CURSOR)" not in _reader,
      "item-use position NEVER reads the blink byte PARTY_CURSOR")
check("def _item_party_walk" in ba and "def _item_party_settle" in ba,
      "one measured primitive (settle + DOWN-only ring walk) serves both landers")
_walk = ba.split("def _item_party_walk")[1].split("def _confirm_party_row")[0] \
    if "def _confirm_party_row" in ba else ba.split("def _item_party_walk")[1][:2000]
for _banned in ('self._tap("LEFT")', 'self._tap("UP")', 'self._tap("RIGHT")',
                'self._tap("B")'):
    check(_banned not in _walk,
          f"the item-use ring walk is DOWN-only (no {_banned})")
check('self._tap("DOWN")' in _walk, "the item-use ring walk presses DOWN")
check("_item_party_settle" in _walk,
      "the ring walk waits out the opening fade before tapping (eaten-tap window)")
_lander = ba.split("def _item_land_party_row")[1].split("def _e4_force_send_pick")[0]
check("self._tap(\"LEFT\")" not in _lander,
      "item-use party land never LEFTs (never LEFT from 0)")
check("_item_party_walk" in _lander,
      "the heal/cure/ether lander walks via the measured primitive")
_revive = ba.split("def _revive_land_fainted_row")[1].split("def _item_land_party_row")[0] \
    if ba.find("def _item_land_party_row") > ba.find("def _revive_land_fainted_row") \
    else ba.split("def _revive_land_fainted_row")[1][:3000]
check("self._party_blind_goto" not in _revive,
      "Revive land does not call force-switch LEFT home")
check("self._tap(\"LEFT\")" not in _revive,
      "Revive pick never LEFTs (lead-LEFT is CANCEL)")
check("_item_party_walk" in _revive,
      "Revive lands via the measured DOWN-only ring walk")
check("hp != 0" in _revive and "REFUSE A" in _revive,
      "Revive still refuses A unless the highlight is verifiably 0-HP")
check("def _item_land_party_row" in ba,
      "heal still uses _item_land_party_row (never LEFT)")
check("_item_land_party_row(_row, kind=\"heal\")" in ba
      or "_item_land_party_row(_row, kind='heal')" in ba,
      "in-battle heal uses the shared measured walk")
check("def _confirm_party_row" not in ba,
      "_confirm_party_row (key-pressing probe over the blink byte) is gone")
check("def _item_leave_cancel" not in ba,
      "_item_leave_cancel (UP/RIGHT off CANCEL) is gone — DOWN wraps instead")
check("spawn" not in _lander.lower() and "lance" not in _lander.lower(),
      "item walk has no Lance spawn helpers")
check("spawn" not in _revive.lower() and "lance" not in _revive.lower(),
      "Revive lander has no Lance spawn helpers")
check("A once" in ba and "no A-spam" in ba,
      "Revive presses A once after the walk, does not bag-spam")
check("revive WALK landed" in ba,
      "Revive logs the walk and presses A")
check("orange is the selection" in ba and "live 16:03" in ba,
      "Revive A is refused when orange is on a living mon (16:03 Articuno)")
check("never insta-A the fighter" in _revive,
      "Revive refuses A on a living highlight after the pick")
check("def _dismiss_item_no_effect" in ba and "won't have any effect" in ba,
      "Failed Revive-on-living dismisses the no-effect box")
check("ITEM-INSTINCT FORCED -> use_revive" in ba
      and "_e4_thin_party" in ba,
      "E4 thin-party Revive is forced (oracle cannot keep_fighting)")
check("E4 waiting for RAM" in ba and "E4 REFUSE A" in ba,
      "E4 STREAM COMMIT waits for RAM cursor == wanted slot before A")

# ── 2026-08-14 09:27: heal-north out of Agatha into Lance ──
check((1, 77) in e4.LEAGUE_CHAIN_MAPS and (13, 0) in e4.LEAGUE_CHAIN_MAPS
      and (3, 9) in e4.LEAGUE_CHAIN_MAPS,
      "League chain includes Agatha's room, Center, and Indigo exterior")
_ag_warps = [((6, 2), (1, 78)), ((6, 12), (1, 76))]
check(e4.e4_south_heal_warp((1, 77), _ag_warps, "Lance") == (6, 12),
      "Agatha heal warp is SOUTH to Bruno, never north to Lance")
check(e4.e4_south_heal_warp((1, 77), _ag_warps, "Lance") != (6, 2),
      "Agatha north door (Lance) is not a heal warp")
_bruno_warps = [((6, 2), (1, 77)), ((6, 12), (1, 75))]
check(e4.e4_south_heal_warp((1, 76), _bruno_warps, "Agatha") == (6, 12),
      "Bruno restock warp is SOUTH to Lorelei, never north to Agatha")
check(e4.e4_room_entry_sealed((1, 76)),
      "Bruno's Room south door is CloseEntry-sealed")
check(e4.e4_room_entry_sealed((1, 77)),
      "Agatha's Room south door is CloseEntry-sealed")
check(not e4.e4_room_entry_sealed((13, 0)),
      "League Center is not sealed — this is where she shops")
check(e4.SHOP_REVIVE_FLOOR >= 6 and e4.SHOPPING[1][2] >= 8,
      "Center shop wants a Revive STACK (floor 6, want 8), not 1")
check(e4.e4_must_restock(0, 17064, "Lance"),
      "0 Revives + $17064 at Center before Lance = keep shopping")
check(e4.e4_must_restock(1, 17064, "Lance", 4),
      "Revive x1 at Center before Lance = keep shopping (need a stack)")
check(e4.e4_must_restock(2, 17064, "Lance", 4),
      "Revive x2 at Center before Lance = still shop (stack is 6)")
check(e4.e4_must_restock(4, 17064, "Lance", 2),
      "Revive x4 + FR x2 at Center still shops (stack is 6, live 12:11)")
check(not e4.e4_must_restock(6, 17064, "Lance", 2),
      "Revive x6 + FR x2 at Center does NOT block the League door")
check(e4.e4_must_restock(2, 17064, "Lance", 0),
      "FR x0 at Center before Lance = keep shopping")
check(e4.e4_must_restock(1, 11624, "Agatha", 0),
      "FR x0 Revive x1 at Center before Agatha = shop first")
check(not e4.e4_must_restock(1, 11624, "Agatha", 0, here=(1, 76)),
      "thin kit IN Bruno's Room does NOT restock-south (CloseEntry sealed)")
check(not e4.e4_must_restock(0, 17064, "Lance", 0, here=(1, 77)),
      "thin kit IN Agatha's Room does NOT restock-south (sealed)")
check(not e4.e4_must_restock(0, 17064, "Bruno"),
      "empty kit before Bruno does NOT retreat (between-room items)")
check(not e4.e4_must_restock(0, 400, "Lance"),
      "broke + 0 Revives off-Center does NOT retreat (can't shop)")
check(e4.e4_must_restock(3, 364, "Lorelei", 0, here=(13, 0)),
      "broke + thin at Center HOLDS the Lorelei door")
check(e4.e4_must_earn(3, 364, 0, here=(13, 0)),
      "broke + thin at Center is a money errand")
check(not e4.e4_must_earn(3, 5364, 0, here=(13, 0)),
      "after Nugget sale she shops Revives in place (no Vermilion for $136)")
check(not e4.e4_must_earn(6, 864, 0, here=(13, 0)),
      "Revive stack + $864 + FR x0 walks into Lorelei (no dead Fly errand)")
check(not e4.e4_must_earn(0, 17064, 0, here=(13, 0)),
      "rich + thin at Center shops, does not fly out")
check(not e4.e4_must_restock(6, 2000, "Lorelei", 1, here=(13, 0)),
      "Revive x6 + FR x1 + can't afford FR#2 = enter (don't hold forever)")
check(103 in e4.SELL_LOOT and 110 in e4.SELL_LOOT,
      "endgame SELL list is Nugget + Big Mushroom (Items pocket, not TMs)")
check(94 not in e4.SELL_LOOT and 93 not in e4.SELL_LOOT,
      "SELL list never includes 93/94 (Sun/Moon Stone — mart refuses, $0)")
check("max 2" in e4s and "DOWN to SELL" in e4s,
      "League sell dismisses greeting then DOWN to SELL (never A-spam BUY)")
# ── 2026-08-14 16:43: blind BUY/SELL nav ate the DOWN, landed in BUY ──
check("pocket byte flips" in e4s and "BUY-list cursor" in e4s,
      "League sell entry is OUTCOME-verified (pocket-flip / buy-list cursor probes)")
check("_menu_select_sell" in e4s and "tap-turn law" in e4s,
      "sell select LEFT-faces then DOWN to SELL (no accidental steps off the clerk)")
check("SELL bag never confirmed" in e4s,
      "sell entry aborts LOUD after bounded retries — never blind-sells")
check("sell_entry_money_drop" in e4s,
      "sell entry aborts LOUD on any money drop (accidental buy guard)")
check("mouse down" not in e4s.lower() and "recon-only" not in e4s.lower(),
      "sell entry drives the live bridge, not a recon shim")
check("center-mat" in e4s and "CENTER_MAT" in e4s,
      "Center exit is (11,15) then DOWN (not go_warp on the warp tile)")
check(not e4.e4_must_earn(3, 364, 0, here=(1, 76)),
      "sealed Bruno never leaves to earn")
check(e4.e4_kit_cash_needed(3, 0, 364) == 10136,
      "3 Revive + 0 FR + $364 needs $10136 more for the stack")
check(e4.ITEM_VS_SEEKER == 362,
      "VS Seeker is key item 362 (Vermilion Center girl)")
check("def earn_kit_cash" in e4s and "VS Seeker" in e4s,
      "League Center has a trainer-payout money errand")
check(e4.e4_must_restock(1, 8184, "Lorelei", 0, here=(13, 0)),
      "thin kit at League Center after whiteout shops before Lorelei")
check(e4.e4_must_restock(0, 17064, "Lorelei", 0, here=(13, 0)),
      "0 Revives at Center before Lorelei = shop")
check(not e4.e4_must_restock(6, 17064, "Lorelei", 2, here=(13, 0)),
      "full stack at Center before Lorelei does not re-shop")
check(not e4.e4_must_restock(0, 17064, "Lorelei"),
      "empty kit before Lorelei without here does NOT retreat")
check(e4.e4_warp_approach_cluster((6, 12)) == ((6, 12), (6, 13), (6, 11), (7, 12), (5, 12)),
      "south-door cluster includes (6,11) so dual-warp BFS can path")
check((6, 11) in e4.e4_warp_approach_cluster((6, 12)),
      "tile in front of south door is walkable/goal (live no-path from (6,7))")
zap_gyara = [
    {"id": 86, "name": "Thunder Wave", "pp": 20, "power": 0, "type": "electric"},
    {"id": 351, "name": "Shock Wave", "pp": 17, "power": 60, "type": "electric"},
    {"id": 197, "name": "Detect", "pp": 5, "power": 0, "type": "fighting"},
    {"id": 65, "name": "Drill Peck", "pp": 20, "power": 80, "type": "flying"},
]
check(e4.e4_preferred_move_index(zap_gyara, ZAP, ["water", "flying"], GYARADOS) == 1,
      "Zapdos vs Gyarados prefers Shock Wave (never Agility/Detect/Drill Peck)")
src_camp = open(os.path.join(_HERE, "campaign.py"), encoding="utf-8").read()
check("never heal-north into Lance" in src_camp
      or "NEVER north into the next seat" in src_camp,
      "roam/heal will not walk Agatha north into Lance")
check("already on League " in src_camp and "never heal-north into Lance" in src_camp,
      "League-map roam forces enter_league, not generic heal")
check("survival-critical ON League maps" in src_camp,
      "critical HP on League maps does not freeze on generic heal")
check("NEXT E4 seat" in src_camp, "exit-building skips the next E4 seat door")
check("entry door sealed" in e4s and "cannot shop mid-gauntlet" in e4s,
      "e4_strike does not restock-south from a CloseEntry-sealed room")
check("shop again before Lorelei" in e4s,
      "thin kit at the Center blocks the Lorelei door until she shops")
check("every fainted wincon" in e4s and "Revive ->" in e4s,
      "between-room revive consumes a stack on every fainted wincon")
check("LANCE LADDER" in e4s and "LADDER = (" in e4s,
      "stock_up buys a BALANCED kit (round-robin ladder), not a single-item floor")
check("LADDER" in e4s.split("def stock_up")[1][:3000],
      "the ladder is what stock_up's pass 1 actually walks")
_lad = e4.FULL_RESTORE, e4.REVIVE
check(e4s.split("LADDER = (")[1].split(")")[0].strip().startswith("FULL_RESTORE"),
      "the ladder leads with a HEAL (the L88 ace's 266 HP beats an L52 bird at half HP)")
check("REVIVE" in e4s.split("LADDER = (")[1].split(")")[0],
      "the ladder still funds Revives (never heals-only)")
check("empty heal pocket" in e4s and "$17240" in e4s,
      "the measured empty-heal-pocket-at-Lance failure is cited in stock_up")
check("CloseEntry" in e4s and "5-7,11-12" in e4s,
      "CloseEntry sealed-door geometry is cited (pret ground truth)")
check("refusing north-door" not in e4s,
      "north door is NOT refused mid-room (that was the 11:38 south-loop)")
check("bag items only" in src_camp and "CloseEntry sealed" in src_camp,
      "campaign heal uses bag items in E4 rooms, never south into a closed door")

# ── 2026-08-14 12:11: empty bag + 17 HP lead walked into Lance ──
check(e4.e4_should_switch_dying_lead(17, 240, True),
      "17/240 + healthy reserve = switch (12:11 Lance wipe-prevention)")
check(not e4.e4_should_switch_dying_lead(17, 240, False),
      "17/240 + no healthy reserve = cannot switch")
check(not e4.e4_should_switch_dying_lead(200, 240, True),
      "healthy lead does not panic-switch")
check(not e4.e4_should_switch_dying_lead(0, 240, True),
      "fainted lead is Revive's job, not a lead-switch")
check(e4.e4_heal_pocket_empty(0, 0, 0, 0, 0),
      "heal pocket empty when FR/Max/Hyper/Super/Potion are all 0")
check(not e4.e4_heal_pocket_empty(0, 0, 0, 1, 0),
      "Super Potion counts — field_heal must not ignore it")
check(not e4.e4_heal_pocket_empty(0, 0, 0, 0, 1),
      "Potion counts — field_heal must not ignore it")
check("13, 22, 21, 20, 19" in src_camp,
      "field_heal cheapest-adequate includes Potion/Super/Hyper/Max/FR")
check("BETWEEN-ROOM SWITCH" in e4s and "empty bag" in e4s,
      "between_room_heal switches the lead when heal cannot save them")
check("switch_dying_lead" in e4s and "switch_dying_lead" in src_camp,
      "dying-lead switch is wired in between-room AND campaign E4-room HEAL")
check(e4.RESTOCK_REVIVE_MIN >= 6 and e4.SHOP_REVIVE_FLOOR >= 6,
      "Center restock floor is Revive >=6 (only shop window before CloseEntry)")
check("def _item_land_party_row" in open(os.path.join(_HERE, "hm_teach.py"), encoding="utf-8").read(),
      "overworld field_heal/field_revive share the never-LEFT item walk")
ht = open(os.path.join(_HERE, "hm_teach.py"), encoding="utf-8").read()
check("self._item_land_party_row(mon_slot)" in ht
      and ht.split("def field_heal")[1].split("def field_revive")[0].count("_party_goto") == 0,
      "field_heal does not LEFT-home via _party_goto")
check(ht.split("def field_revive")[1].split("def field_pp_restore")[0].count("_party_goto") == 0,
      "field_revive does not LEFT-home via _party_goto")

# ── 2026-08-14 12:34: SEND 17HP Blastoise then switch; Revive Zapdos vs Dragonair ──
DRAGONAIR = 148
cands_gyara = [
    {"species": BLAST, "hp": 17, "maxhp": 240, "row": 1},
    {"species": ART, "hp": 157, "maxhp": 157, "row": 3},
]
pref_gyara = e4.e4_force_send_pref("Lance", ["water", "flying"], GYARADOS,
                                  zap_has_electric=True)
pick = e4.e4_pick_send_cand(pref_gyara, cands_gyara)
check(pick and pick["species"] == ART,
      f"SEND vs Gyarados skips 17HP Blastoise for healthy Articuno (got {pick})")
cands_only_dying = [{"species": BLAST, "hp": 17, "maxhp": 240, "row": 1}]
pick = e4.e4_pick_send_cand(pref_gyara, cands_only_dying)
check(pick and pick["species"] == BLAST,
      "SEND seats dying Blastoise only when no healthy pref exists")
rows_lance = [
    {"row": 0, "species": ZAP, "hp": 0, "level": 52},
    {"row": 1, "species": BLAST, "hp": 17, "level": 79},
    {"row": 2, "species": MOLT, "hp": 86, "level": 51},
    {"row": 3, "species": ART, "hp": 0, "level": 51},
]
check(e4.e4_revive_target_from_rows(rows_lance, "Lance", ["water", "flying"],
                                    GYARADOS, True) == 0,
      "vs Gyarados + gun: Revive Zapdos (4x), not the highest-level corpse")
check(e4.e4_revive_target_from_rows(rows_lance, "Lance", ["dragon"],
                                    DRAGONAIR, True) == 3,
      "vs Dragonair: Revive Articuno, not Zapdos (OHKO bait / highest-level)")
check(e4.e4_revive_pref("Lance", ["dragon"], DRAGONAIR, True)[0] == ART,
      "Lance-dragon revive pref starts with Articuno")
rows_aero = [
    {"row": 0, "species": ZAP, "hp": 0, "level": 52},
    {"row": 1, "species": BLAST, "hp": 0, "level": 79},
    {"row": 2, "species": MOLT, "hp": 86, "level": 51},
]
check(e4.e4_revive_target_from_rows(rows_aero, "Lance", ["rock", "flying"],
                                    e4.AERODACTYL_SP, True) == 1,
      "vs Aerodactyl: do not revive 4x-Rock Zapdos when Blastoise is also down")
check(e4.e4_should_hold_north(17, 240, 2, 0, 0),
      "dying ace + potions in bag = hold the north door")
check(not e4.e4_should_hold_north(17, 240, 0, 1, 0),
      "dying ace + Revive only (nobody fainted) = switch and go; Revive cannot heal living")
check(e4.e4_should_hold_north(200, 240, 0, 1, 2),
      "fainted wincon + Revive in bag = hold north until consumed")
check(not e4.e4_should_hold_north(17, 240, 0, 0, 0),
      "empty bag + dying ace = do not invent items; switch + log thin")
check(e4.e4_preferred_move_index(molt_moves, MOLT, ["dragon"], DRAGONAIR) == 2,
      "Moltres vs Dragonair: Flamethrower, never Agility")
check(e4.e4_preferred_move_index(zap_gyara, ZAP, ["water", "flying"], GYARADOS) == 1,
      "Zapdos vs Gyarados is Electric damage only (Shock Wave)")
check("HOLD THE DOOR" in e4s and "cannot buy 1 and walk" in e4s,
      "Center shop holds the Lorelei door until the stack is bought")
check("Lance will be attempted on a thin kit" in e4s,
      "between-room logs a thin-kit Lance attempt (no invented items)")
check("skipped dying stepping-stone" in ba and "e4_pick_send_cand" in ba,
      "E4 SEND uses healthy-pref pick (not 17HP Blastoise as a stepping stone)")
check("e4_revive_target_from_rows" in ba,
      "in-battle Revive aims the wincon row, not highest-level fainted")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
