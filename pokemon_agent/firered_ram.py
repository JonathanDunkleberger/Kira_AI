"""firered_ram.py - FireRed (USA, AGB-BPRE) RAM offset CANDIDATES.

These are from the community / pokefirered disassembly map. They are CANDIDATES
until empirically confirmed (M0 confirms party-count + coords by watching them
change). Nothing here is trusted silently - callers print CANDIDATE vs CONFIRMED.

DMA note: FireRed shuffles SaveBlock1's location, so player coords are NOT at a
fixed address. The pointer at gSaveBlock1Ptr is fixed and always points to the
current SaveBlock1; coords are read as *(SaveBlock1Ptr) + offset.
"""

# ── Fixed-address candidates (EWRAM/IWRAM) ───────────────────────────────────
GSAVEBLOCK1_PTR   = 0x03005008   # pointer -> current SaveBlock1 (DMA-shuffled target)
GSAVEBLOCK2_PTR   = 0x0300500C   # pointer -> SaveBlock2
GPLAYER_PARTY_CNT = 0x02024029   # u8 party count. ✅ LOCKED 2026-06-22: read 1 from after_pick.state
# ── BATCH 6 PHASE 4: Pokédex caught-count (clean RAM read, no menu) ──────────────────────────────
# SOURCED from pret/pokefirered include/global.h: struct SaveBlock2.pokedex @ 0x18, struct Pokedex.owned
# @ +0x10 -> owned[] at SaveBlock2 + 0x28. DEX_FLAGS_NO = ROUND_BITS_TO_BYTES(NUM_SPECIES=412) = 52, and
# the struct places `seen` at Pokedex+0x44 = owned+0x34 (=52), confirming the byte-length. Dex flags are
# NOT security-key encrypted (unlike money/items) -> a plain popcount of owned[] = species CAUGHT.
SB2_POKEDEX_OWNED_OFF = 0x28
DEX_OWNED_BYTES       = 52
GPLAYER_PARTY     = 0x02024284   # party data (100 B/mon). ✅ LOCKED: species decrypted to 4=charmander
GENEMY_PARTY      = 0x0202402C   # enemy party (100 B/mon). ✅ CONFIRMED 2026-06-27 via EWRAM brute-scan:
#                                  decodes the full enemy roster across 4 battle states (brock=geodude+
#                                  onix, forest_trainer=weedle+caterpie, wilds=1). PID@+0 / otId@+4 are
#                                  unencrypted (same layout as the player party) -> enables shiny detection.

# ── Offsets WITHIN SaveBlock1 (add to the dereferenced pointer) ───────────────
SB1_OFF_POS_X     = 0x0000       # s16 player map X
SB1_OFF_POS_Y     = 0x0002       # s16 player map Y

# ── Battle (✅ VERIFIED 2026-06-22 vs battle.state: Bulbasaur L6 21/21 vs Pidgey L3) ─
# gBattleTypeFlags is UNRELIABLE as a gate - it holds STALE values out of battle
# (read 0x1c in the overworld after the rival fight). The reliable in-battle signal
# is the battle-RESOURCES pointer: null out of battle, a valid EWRAM ptr in battle.
GBATTLE_TYPE_FLAGS = 0x02022B4C  # u32 battle type bitflags - STALE out of battle, do NOT gate on it
GMOVE_TO_LEARN     = 0x02024022  # u16 gMoveToLearn (pret symbols 2026-07-07): armed with the pending
#                                  move id when a level-up wants to teach with 4 moves known (the
#                                  "Delete an older move?" flow). STALE after the flow — treat only a
#                                  CHANGED nonzero value as a live prompt (snapshot at battle attach).
GBATTLE_RES_PTR    = 0x02023FE8  # ✅ battle-resources ptr: valid EWRAM addr ONLY during battle (the GATE)
GBATTLE_MONS       = 0x02023BE4  # ✅ gBattleMons[4] base (was 0x02024084 - WRONG). 88 bytes each.
GBATTLE_MON_SIZE   = 88
# ✅ BATTLE PHASE REGISTER (derived 2026-06-22, validated dynamically across two live
# action menus + a text-wait + animation): a u32 that takes a distinct value per battle
# phase. The action menu ("What will X do?", FIGHT/BAG/POKEMON/RUN) is the one we gate
# move-selection on; the move list confirms FIGHT opened; everything else = advance/wait.
GBATTLE_PHASE      = 0x03002258
PHASE_ACTION_MENU  = 0x580        # action menu up - my turn to choose an action
PHASE_MOVE_LIST    = 0x680        # FIGHT move list up - confirms FIGHT opened
# ✅ ACTION-SELECTION CURSOR (derived 2026-06-22 by moving the cursor from battle_menu):
# which of the 4 action cells is highlighted. FIGHT=0 (top-left), BAG=1 (top-right),
# POKEMON=2 (bottom-left), RUN=3 (bottom-right). The engine READS this to reach FIGHT
# deterministically instead of assuming the position (the blind-nav desync bug).
GBATTLE_ACTION_CURSOR = 0x02023FF8
ACT_FIGHT, ACT_BAG, ACT_POKEMON, ACT_RUN = 0, 1, 2, 3
# ⛔ CATCH-ARC FINDING (2026-06-25): the ACTION cursor is UN-NAVIGABLE in this libmgba core (same as
# the move list) — the first d-pad press at the action menu confirms FIGHT (menu_up->0); _goto_run
# fails. The engine only wins because attacks use the DEFAULT FIGHT cell (cursor 0) + move-swap. So
# BAG(1) is unreachable via menu → a Poké Ball can't be thrown by navigating. The catch path is to
# RAM-WRITE the battle action (extend the proven move-swap pattern): write USE_ITEM + the ball to the
# action/chosen-item RAM, bypassing the cursor. Ball inventory: gSaveBlock1 + BAG_BALLS_OFF below.
BAG_BALLS_OFF = 0x430            # PokéBalls pocket off gSaveBlock1 ({u16 itemId, u16 qty^key}); id4=PokéBall
# ── IN-BATTLE BAG pocket navigation (root-caused 2026-06-27 — the "wrong pocket = 141 dead throws" bug).
# The in-battle bag opens on the LAST-VIEWED pocket, NOT always Poké Balls: on Route 3 it opens on the
# (empty) ITEMS pocket, so the old blind UP+A+A selected CANCEL and threw NOTHING (ball count never
# decremented). These let throw_ball steer the cursor to the Poké Balls pocket. Addresses found by an
# EWRAM diff across pocket switches; VALIDATED BY A CATCH CONTROL on the Route 3 failing state (the live
# pocket index read 0->1->2 to the balls pocket, item id 4 selected, balls 5->1, party 1->2 = caught).
GBAG_POCKET        = 0x0203AD02   # u8: current bag pocket index (0=Items, 1=Key Items, 2=Poké Balls, ...)
GSPECIALVAR_ITEMID = 0x0203AD30   # u16: gSpecialVar_ItemId — the SELECTED bag item id (4 = Poké Ball)
POCKET_POKE_BALLS  = 2            # Poké Balls pocket index (FRLG fixed layout; control-read ==2 on balls)
BALL_ITEM_IDS = frozenset(range(1, 13))   # Master(1)..Premier(12) — any Poké Ball variant is throwable
# ── BATTLE-ACTION INJECTION (catch arc) — addresses sourced from CFRU BPRE.ld, cross-validated vs the
# verified anchors (gBattleMons/gBattleStruct MATCH) AND a behavioral CONTROL. The action-menu cursor
# is un-navigable, so we inject the chosen action by RAM-write: on the controller-completion edge
# (gBattleControllerExecFlags bit0 1->0) the engine reads the action from gBattleBufferB[battler][1];
# swap that byte to inject. CONTROL-PROVEN 2026-06-25: inject B_ACTION_RUN(3) -> wild battle FLEES;
# no-op USE_MOVE(0) -> fights (enemy faints). X->X/Y->Y, falsifiable. (Observational finds were
# lookalikes off by ~0x50 — every claim MUST pass a control, never a constant.)
GBATTLE_BUFFER_A   = 0x02022BC4   # gBattleBufferA[MAX_BATTLERS][0x200] (engine->controller)
GBATTLE_BUFFER_B   = 0x020233C4   # gBattleBufferB[MAX_BATTLERS][0x200] (controller->engine); [b][1]=action/item
GBATTLE_EXEC_FLAGS = 0x02023BC8   # gBattleControllerExecFlags (bit b = battler b's controller running)
GCHOSEN_ACTION     = 0x02023D7C   # gChosenActionByBattler[MAX_BATTLERS] (copied from gBattleBufferB[b][1])
# B_ACTION_*: USE_MOVE=0 USE_ITEM=1 SWITCH=2 RUN=3 ... WALLY_THROW=9 (tutorial-only, NOT a real capture).
# Ball-throw via USE_ITEM still PENDING: needs the item past the un-navigable bag (gSpecialVar_ItemId
# addr unsourced; bag-cancel buffer-swap did NOT throw — 51 attempts/0 catch). recon_inject.py = harness.
# ── MOVE LIST (FIGHT submenu) findings 2026-06-24, screenshot-verified ────────
# It is a 2x2 GRID: TACKLE(TL,0) GROWL(TR,1) / LEECH SEED(BL,2) VINE WHIP(BR,3). Cursor opens on
# slot 0. So the OLD _nav_to 2x2 layout (RIGHT=col, DOWN=row) is CORRECT; the bug is the eaten-
# first-press (the engage trick is needed in the move list too).
# OPEN SIGNAL: GBATTLE_MENU_UP (0x02023E86) = 1 at the ACTION menu, 0 at the MOVE LIST (white
# panel up; blue panel = a text box). So: action-menu = white & menu_up==1; move-list = white &
# menu_up==0; text = blue & menu_up==0. (This IS the reliable open signal.)
# FIXED 2026-06-24 (the Vine-Whip/Brock wedge): this build's move list can only CONFIRM slot 0
# (the cursor never navigates), and the move-list OPEN is timing-flaky. _select_and_verify now
# (1) SWAP-FIRST battle-copies the chosen move into slot 0 before any press (so an early confirm
# fires the intended move, never a dry 0-PP Tackle -> "There's no PP left for this move!"),
# (2) RETRIES the open (engage+A up to 8x) verified by the _in_move_list() pixel-check, then
# confirms slot 0, (3) swaps back only on ENEMY-HP-DROP (after the engine's execution slot-read).
# The old engine pressed the open ONCE and ASSUMED it -> flaked opens hung ~30 attempts (the
# "Weedle stuck at 2/26" wedge). mgba Memory.search wrapper returns 0 (broken in this build) -
# can't use it to isolate the cursor byte. Fixtures: forest_4move.state, live_wedge.state.
# NOTE: loaded mid-battle states are UNFAITHFUL (load_state desyncs the first menu interaction's
# pixel/input path); validate menu changes on the LIVE leg, not loaded-state replays.
# ⚠️ GBATTLE_PHASE above is NOT a phase register - it's a FREE-RUNNING per-frame counter
# (+0x100 every frame, low byte always 0x80). It passes through 0x580/'action menu' once
# every ~16 frames during ANIMATIONS too, so it cannot gate anything. (Diagnosed 2026-06-23
# by advancing frames with no input: the value never plateaus even while the menu waits.)
# The real 'action menu is up' signal, found by diffing battle-moment savestates (menu vs
# animation/damage-text/enemy-turn) and confirmed dynamically (0 through the intro, 1 the
# instant the cursor becomes movable):
GBATTLE_MENU_UP = 0x02023E86      # u8: 1 = action menu (FIGHT/BAG/POKEMON/RUN) up, 0 otherwise

EWRAM_LO, EWRAM_HI = 0x02000000, 0x02040000


def valid_ewram_ptr(p: int) -> bool:
    return EWRAM_LO <= p < EWRAM_HI


# ── BATTLE LIVENESS (the e4_run3 phantom-battle class) ──────────────────────────────────
# GBATTLE_RES_PTR can stay a valid EWRAM address AFTER a whiteout (stale pointer), so a
# pointer check alone re-attaches to a DEAD battle's frozen struct (phantom 0-PP reads, no
# action menu, an eternal abort->re-enter livelock). gMain.callback2 is the ground truth:
# during any real battle it is a battle-family callback (BattleMainCB2/transitions — verified
# live: only 0x08010509/0x08011101 over 1200 battle frames incl. menus); back in the world it
# is CB2_Overworld, and the whiteout transition is CB2_WhiteOut (pret pokefirered.sym).
GMAIN_CB2 = 0x030030F4                  # gMain (0x030030F0) + 4 = callback2
# In-battle party DISPLAY order (pret gBattlePartyCurrentOrder, 6 high-first nibbles).
# ⚠️ THE ORDER LAW (recon_partytruth, 2026-07-07 — supersedes every earlier model): while
# the in-battle party MENU is open, the game PHYSICALLY rearranges gPlayerParty into this
# display order and RESTORES it when the menu closes; gPlayerParty HP is live/accurate at
# all times. So no walk should convert through this map — resolve target rows by CONTENT
# from gPlayerParty at MENU TIME (battle_agent._menu_rows) and never carry a slot index
# across the menu-open boundary. Kept for forensics dumps only.
GBATTLE_PARTY_ORDER = 0x0203B0DC
# gBattlerPartyIndexes (u16[4]) — ✅ CONFIRMED 2026-07-07 (recon_partytruth: [0] tracked
# 0->3->5 across two forced switches): the ORIGINAL gPlayerParty index of each battler.
GBATTLER_PARTY_IDX = 0x02023BCE
_CB2_OVERWORLD = 0x080565B4 | 1         # thumb bit set as stored
_CB2_WHITEOUT = 0x080566A4 | 1


def battle_cb2_dead(bridge) -> bool:
    """True iff gMain.callback2 says we are IN THE WORLD (overworld/whiteout) — any battle
    struct still pointing somewhere is a corpse, not a fight."""
    return bridge.rd32(GMAIN_CB2) in (_CB2_OVERWORLD, _CB2_WHITEOUT)


def read_player_coords(bridge):
    """Returns (x, y) or None if SaveBlock1 isn't allocated yet (pre-game)."""
    sb1 = bridge.rd32(GSAVEBLOCK1_PTR)
    if not valid_ewram_ptr(sb1):
        return None
    x = bridge.rds16(sb1 + SB1_OFF_POS_X)
    y = bridge.rds16(sb1 + SB1_OFF_POS_Y)
    return (x, y)


def read_party_count(bridge):
    return bridge.rd8(GPLAYER_PARTY_CNT)


def pokedex_owned_count(bridge):
    """BATCH 6 PHASE 4 — how many species she's CAUGHT, read straight from RAM (no menu). Popcount of
    the SaveBlock2 Pokédex owned-flag array (offset sourced above). Returns an int, or None if SaveBlock2
    isn't allocated yet (pre-game). Bits past NUM_SPECIES are always 0, so popcounting the full 52 bytes
    is exact. Lets her answer 'how many have I caught?' on demand — including when chat asks."""
    sb2 = bridge.rd32(GSAVEBLOCK2_PTR)
    if not valid_ewram_ptr(sb2):
        return None
    base = sb2 + SB2_POKEDEX_OWNED_OFF
    return sum(bin(bridge.rd8(base + i)).count("1") for i in range(DEX_OWNED_BYTES))


def pokedex_owns(bridge, species_id):
    """DEX DOCTRINE (2026-07-06): does she OWN this species? Kanto internal ids 1-151 equal
    national dex numbers, so the owned bit is (species_id-1). None on bad ptr/out-of-range."""
    if not species_id or species_id > 151:
        return None
    sb2 = bridge.rd32(GSAVEBLOCK2_PTR)
    if not valid_ewram_ptr(sb2):
        return None
    idx = species_id - 1
    return bool(bridge.rd8(sb2 + SB2_POKEDEX_OWNED_OFF + (idx >> 3)) & (1 << (idx & 7)))
