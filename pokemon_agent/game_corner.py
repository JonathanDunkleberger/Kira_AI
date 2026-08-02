"""game_corner.py — Celadon Game Corner prize TMs (Ice Beam first).

Human let's-play beat: before Erika, buy TM13 Ice Beam with cash→coins and teach the
ace (Blastoise). Ice into Grass sails the gym; same TM carries into Giovanni / Lance later.

DISASM GROUND TRUTH (pret/pokefirered 2026-08-02):
  Celadon (3,6)
    Restaurant door (37,29) -> (10,17); Coin Case man (1,2) FACE_DOWN → stand (1,3) UP
    Game Corner door (34,21) -> (10,14); Coins clerk (6,2) FACE_DOWN → stand (6,3) UP
      multichoice: 0 = 50 coins / 1000¥, 1 = 500 coins / 10000¥
    Prize Room door (39,20) -> (10,15); TM clerk (6,2) FACE_DOWN → stand (6,3) UP
      multichoice: 0=TM13 Ice Beam 4000, 1=TM23, 2=TM24, 3=TM30, 4=TM35, 5=CANCEL
  Coins RAM: SaveBlock1+0x294 u16 XOR SaveBlock2.encryptionKey (same key as money)
  ITEM_COIN_CASE=260, ITEM_TM13=301, MOVE_ICE_BEAM=58

Never enters the hideout stairs (15,2). Fail-closed: any step LOUD-skips; beat_gym continues.
"""
import os

import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Celadon Game-Corner fact table (rule 14) ───────────────────────────
CELADON = (3, 6)
GC_DOOR = (34, 21)
PRIZE_DOOR = (39, 20)
RESTAURANT_DOOR = (37, 29)
GC = (10, 14)
PRIZE = (10, 15)
RESTAURANT = (10, 17)

COIN_CASE_FRONT = (1, 3)          # man at (1,2) FACE_DOWN
COINS_CLERK_FRONT = (6, 3)        # clerk at (6,2) FACE_DOWN
PRIZE_TM_CLERK_FRONT = (6, 3)     # TM clerk at (6,2) FACE_DOWN

ITEM_COIN_CASE = 260
ITEM_TM13 = 301
MOVE_ICE_BEAM = 58
TM13_NO = 13
TM13_COIN_COST = 4000
COINS_PACK_500 = 500
COINS_PACK_COST = 10000           # ¥ for 500 coins
# Keep a Mart floor after the spend (mirrors campaign.SHOP_MONEY_FLOOR default).
MONEY_FLOOR = int(os.getenv("POKEMON_SHOP_MONEY_FLOOR", "500"))

ICE_BEAM_ERRAND_ENABLED = os.getenv("POKEMON_ICE_BEAM_ERRAND", "1") != "0"


def coins(b):
    """Player coin balance (SaveBlock1+0x294 XOR encryptionKey)."""
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20)
    return (b.rd16(sb1 + 0x0294) ^ (key & 0xFFFF)) & 0xFFFF


def has_coin_case(b):
    return ITEM_COIN_CASE in ht.pocket_items(b, ht.KEY_ITEMS_OFF, 30)


def tm_case_qty(b, item_id):
    sb1 = ht._sb1(b)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    for i in range(58):
        slot = sb1 + ht.TM_CASE_OFF + i * 4
        iid = b.rd16(slot)
        if iid == 0:
            break
        if iid == item_id:
            return b.rd16(slot + 2) ^ key
    return 0


class IceBeamErrand:
    """Coin Case → buy ≥4000 coins → prize TM13 → teach ace. Best-effort; never raises."""

    def __init__(self, camp, log=print):
        self.camp = camp
        self.b = camp.b
        self.log = log

    def drain(self, max_a=40):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(24):
                    b.run_frame()

    def goto(self, tile, label, max_seconds=60):
        b, camp = self.b, self.camp
        r = camp.trav.travel(target_map=None, arrive_coord=tile,
                             max_steps=200, max_seconds=max_seconds)
        if st.in_battle(b):
            self.log(f"   [icebeam] battle en route ({label}) -> {camp.battle_runner()}")
            self.drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile,
                                 max_steps=200, max_seconds=max_seconds)
        return r == "arrived" or tv.coords(b) == tile

    def talk(self, front, face, label):
        """Stand, face, A into dialogue; leave the box OPEN for the caller to drive menus."""
        b, camp = self.b, self.camp
        if not self.goto(front, label):
            self.log(f"   [icebeam] !! couldn't reach {label} @ {front}")
            return False
        for _ in range(6):
            b.press(face, 8, 8, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
            if dd_box(b):
                return True
        return dd_box(b)

    def ensure_celadon(self):
        b, camp = self.b, self.camp
        if tuple(tv.map_id(b)) == CELADON:
            return True
        if tuple(tv.map_id(b)) in (GC, PRIZE, RESTAURANT):
            camp._exit_to_overworld()
            for _ in range(40):
                b.run_frame()
            return tuple(tv.map_id(b)) == CELADON
        # beat_gym calls us from the gym city — travel if somehow elsewhere
        r = camp.trav.travel(target_map=CELADON, max_steps=800, max_seconds=240)
        if st.in_battle(b):
            camp.battle_runner()
            self.drain()
            r = camp.trav.travel(target_map=CELADON, max_steps=800, max_seconds=240)
        return tuple(tv.map_id(b)) == CELADON or r == "arrived"

    def ensure_coin_case(self):
        b = self.b
        if has_coin_case(b):
            return True
        self.log("   [icebeam] fetching Coin Case (restaurant south of Game Corner)")
        if not self.ensure_celadon():
            return False
        if self.camp.enter_warp(pick=RESTAURANT_DOOR) != "warped":
            self.log("   [icebeam] !! restaurant door failed")
            return False
        for _ in range(50):
            b.run_frame()
        if tuple(tv.map_id(b)) != RESTAURANT:
            self.log(f"   [icebeam] !! expected restaurant, at {tv.map_id(b)}")
            self.camp._exit_to_overworld()
            return False
        if not self.talk(COIN_CASE_FRONT, "UP", "coin-case-man"):
            self.log("   [icebeam] !! Coin Case man didn't talk")
            self.camp._exit_to_overworld()
            return False
        self.drain(max_a=60)                      # giveitem fanfare
        ok = has_coin_case(b)
        self.log(f"   [icebeam] Coin Case -> {'HAVE' if ok else 'MISSING'}")
        self.camp._exit_to_overworld()
        for _ in range(40):
            b.run_frame()
        return ok

    def _buy_one_500_pack(self):
        """One Coins-clerk purchase of 500 coins. Verifies coin+money deltas."""
        b, camp = self.b, self.camp
        before_c, before_m = coins(b), camp.money()
        if before_m < COINS_PACK_COST + MONEY_FLOOR:
            return "broke"
        if not self.talk(COINS_CLERK_FRONT, "UP", "coins-clerk"):
            return "no_talk"
        # Welcome msgbox -> multichoice (50 / 500 / Cancel). One A advances the prompt.
        b.press("A", 8, 12, camp.render, owner="agent")
        for _ in range(30):
            b.run_frame()
        # Select row 1 = 500 coins (blind DOWN once from default row 0).
        b.press("DOWN", 8, 10, camp.render, owner="agent")
        for _ in range(16):
            b.run_frame()
        b.press("A", 8, 12, camp.render, owner="agent")
        self.drain(max_a=50)
        after_c, after_m = coins(b), camp.money()
        if after_c >= before_c + COINS_PACK_500 and after_m <= before_m - COINS_PACK_COST:
            self.log(f"   [icebeam] bought 500 coins ({before_c}->{after_c}; "
                     f"¥{before_m}->{after_m})")
            return "bought"
        self.log(f"   [icebeam] !! coin buy verify failed "
                 f"(coins {before_c}->{after_c}, ¥{before_m}->{after_m})")
        # B out of any leftover menu
        for _ in range(4):
            b.press("B", 8, 12, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
        self.drain()
        return "verify_fail"

    def buy_coins_to(self, target=TM13_COIN_COST):
        b, camp = self.b, self.camp
        have = coins(b)
        if have >= target:
            return True
        need = target - have
        packs = (need + COINS_PACK_500 - 1) // COINS_PACK_500
        cost = packs * COINS_PACK_COST
        if camp.money() < cost + MONEY_FLOOR:
            self.log(f"   [icebeam] !! broke for Ice Beam coins "
                     f"(need ¥{cost}+floor, have ¥{camp.money()}) — skip")
            return False
        self.log(f"   [icebeam] buying {packs}x500 coins (have {have}, need {target})")
        if not self.ensure_celadon():
            return False
        if camp.enter_warp(pick=GC_DOOR) != "warped":
            self.log("   [icebeam] !! Game Corner door failed")
            return False
        for _ in range(50):
            b.run_frame()
        if tuple(tv.map_id(b)) != GC:
            self.log(f"   [icebeam] !! expected Game Corner, at {tv.map_id(b)}")
            camp._exit_to_overworld()
            return False
        for i in range(packs + 2):                # +2 retries for flaky UI
            if coins(b) >= target:
                break
            r = self._buy_one_500_pack()
            if r == "broke":
                break
            if r != "bought":
                self.log(f"   [icebeam] coin pack attempt {i + 1} -> {r}")
        ok = coins(b) >= target
        camp._exit_to_overworld()
        for _ in range(40):
            b.run_frame()
        return ok

    def exchange_tm13(self):
        b, camp = self.b, self.camp
        if tm_case_qty(b, ITEM_TM13) > 0:
            return True
        if coins(b) < TM13_COIN_COST:
            self.log(f"   [icebeam] !! not enough coins for TM13 ({coins(b)}<{TM13_COIN_COST})")
            return False
        if not self.ensure_celadon():
            return False
        if camp.enter_warp(pick=PRIZE_DOOR) != "warped":
            self.log("   [icebeam] !! prize room door failed")
            return False
        for _ in range(50):
            b.run_frame()
        if tuple(tv.map_id(b)) != PRIZE:
            self.log(f"   [icebeam] !! expected prize room, at {tv.map_id(b)}")
            camp._exit_to_overworld()
            return False
        before_c, before_q = coins(b), tm_case_qty(b, ITEM_TM13)
        if not self.talk(PRIZE_TM_CLERK_FRONT, "UP", "prize-tm-clerk"):
            camp._exit_to_overworld()
            return False
        # "We exchange coins for prizes" -> WhichPrize multichoice. Row 0 = Ice Beam.
        b.press("A", 8, 12, camp.render, owner="agent")
        for _ in range(30):
            b.run_frame()
        b.press("A", 8, 12, camp.render, owner="agent")   # pick TM13
        for _ in range(30):
            b.run_frame()
        # YESNO ("You want the TM13 Ice Beam?") — YES is top; A.
        b.press("A", 8, 12, camp.render, owner="agent")
        self.drain(max_a=80)
        after_c, after_q = coins(b), tm_case_qty(b, ITEM_TM13)
        ok = after_q > before_q and after_c <= before_c - TM13_COIN_COST
        self.log(f"   [icebeam] prize TM13 -> {'OK' if ok else 'FAIL'} "
                 f"(coins {before_c}->{after_c}, case {before_q}->{after_q})")
        if not ok:
            for _ in range(4):
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(16):
                    b.run_frame()
            self.drain()
        camp._exit_to_overworld()
        for _ in range(40):
            b.run_frame()
        return ok

    def pick_recipient(self):
        """Ace if it can learn TM13; else highest-level compatible party mon."""
        b = self.b
        pc = b.rd8(ram.GPLAYER_PARTY_CNT)
        if not pc:
            return None, None
        cands = []
        for s in range(pc):
            sp = st.read_party_species(b, s)
            if not ht.tm_compatible(b, TM13_NO, sp):
                continue
            moves = st.read_party_moves(b, s)
            if MOVE_ICE_BEAM in moves:
                return s, "already"
            lv = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
            cands.append((lv, s, sp))
        if not cands:
            return None, None
        cands.sort(reverse=True)
        return cands[0][1], "teach"

    def teach_ice_beam(self):
        b, camp = self.b, self.camp
        slot, why = self.pick_recipient()
        if slot is None:
            self.log("   [icebeam] !! no party mon can learn Ice Beam")
            return "no_learner"
        if why == "already":
            return "have_ice_beam"
        if tm_case_qty(b, ITEM_TM13) <= 0:
            return "no_tm"
        forget = ht.forget_idx_for_tm(b, slot)
        mon = st.SPECIES_NAME.get(st.read_party_species(b, slot), f"slot{slot}")
        self.log(f"   [icebeam] teaching TM13 Ice Beam -> {mon} slot {slot} "
                 f"(forget_idx={forget})")
        camp.on_event(f"Game Corner run — teaching {mon} Ice Beam. flower lady's gonna hate this.",
                      kind="gym", tier=2)
        tf = ht.TeachFlow(camp, log=self.log, on_event=camp.on_event)
        res = tf.teach("_tm", slot, forget, item_override=ITEM_TM13, move_override=MOVE_ICE_BEAM)
        self.log(f"   [icebeam] teach -> {res}")
        return res

    def run(self):
        """Full errand. Returns status string; never raises."""
        if not ICE_BEAM_ERRAND_ENABLED:
            return "disabled"
        b, camp = self.b, self.camp
        slot, why = self.pick_recipient()
        if why == "already":
            self.log("   [icebeam] ace/party already knows Ice Beam — skip")
            return "have_ice_beam"
        if slot is None:
            self.log("   [icebeam] no compatible learner in party — skip")
            return "no_learner"
        if tm_case_qty(b, ITEM_TM13) > 0:
            self.log("   [icebeam] TM13 already in case — teaching")
            return self.teach_ice_beam()
        # Budget gate before walking anywhere
        have_c = coins(b)
        need_c = max(0, TM13_COIN_COST - have_c)
        packs = (need_c + COINS_PACK_500 - 1) // COINS_PACK_500 if need_c else 0
        if packs and camp.money() < packs * COINS_PACK_COST + MONEY_FLOOR:
            self.log(f"   [icebeam] !! skip — need ¥{packs * COINS_PACK_COST}+floor "
                     f"for {need_c} more coins (have ¥{camp.money()})")
            return "broke"
        camp.on_event("quick Game Corner stop — Ice Beam for the grass gym. classic trainer move.",
                      kind="gym", tier=2)
        if not self.ensure_coin_case():
            return "no_coin_case"
        if not self.buy_coins_to(TM13_COIN_COST):
            return "coin_buy_failed"
        if not self.exchange_tm13():
            return "prize_failed"
        return self.teach_ice_beam()
