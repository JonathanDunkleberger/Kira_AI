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
GPLAYER_PARTY     = 0x02024284   # party data (100 B/mon). ✅ LOCKED: species decrypted to 4=charmander

# ── Offsets WITHIN SaveBlock1 (add to the dereferenced pointer) ───────────────
SB1_OFF_POS_X     = 0x0000       # s16 player map X
SB1_OFF_POS_Y     = 0x0002       # s16 player map Y

# ── Battle (✅ VERIFIED 2026-06-22 vs battle.state: Bulbasaur L6 21/21 vs Pidgey L3) ─
# gBattleTypeFlags is UNRELIABLE as a gate - it holds STALE values out of battle
# (read 0x1c in the overworld after the rival fight). The reliable in-battle signal
# is the battle-RESOURCES pointer: null out of battle, a valid EWRAM ptr in battle.
GBATTLE_TYPE_FLAGS = 0x02022B4C  # u32 battle type bitflags - STALE out of battle, do NOT gate on it
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
