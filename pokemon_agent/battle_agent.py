"""battle_agent.py - the reusable, TURN-GATED battle engine (the HANDS for any battle).

Reads battle state from RAM (verified offsets), decides via the type-chart policy
(no LLM - fast deterministic hands), and presses the menu ONLY on the rising edge of
the verified turn gate (GBATTLE_MY_TURN: 2 = waiting for my action, 0 = busy). NEVER
blind-mashes. Emits NEUTRAL game-events through on_event; the bot binds
on_event=_pokemon_react -> her self -> _ok_to_self_speak, so her VOICE narrates - the
engine never speaks in character. Performance beats yield the floor (pace callback)
so her line lands before the hands press on. Reusable for every trainer/gym -> E4.

Input is owner-attributed ('agent'): the single Bridge owner. Any non-agent press is
dropped + logged - no masher/timer can inject input mid-turn.
"""
import os
import time

import firered_ram as ram
import pokemon_state as st
import pokemon_policy as pol

HOLD = 8

# Gen-1 legendaries (FireRed national-dex ids) — a big-beat recognition in run() (Phase 2D).
_LEGENDARY_SPECIES = frozenset({144, 145, 146, 150, 151})   # Articuno, Zapdos, Moltres, Mewtwo, Mew

# SELF-DESTRUCT FAMILY (FireRed national-dex ids) — foes that can NUKE-TRADE our active: Self-
# Destruct/Explosion one-shots even a dominant lead (koga_run3 2026-07-07: Koga's L37 Koffing
# detonated on Venusaur L54 turn one; the bench then fed itself to Muk/Weezing — full wipe). The
# human answer: SLEEP the bomber first — it can't detonate while asleep. Geodude/Graveler/Golem,
# Voltorb/Electrode, Koffing/Weezing. (Game-knowledge in engine code — portability debt, same
# class as _LEGENDARY_SPECIES.)
_NUKE_SPECIES = frozenset({74, 75, 76, 100, 101, 109, 110})

# ── BATCH 2 PART B: in-battle "use your items" instinct ───────────────────────────────────────────
# An active mon at/under this HP fraction WITH a heal item in the bag -> the oracle is OFFERED "use a
# potion" (capability-not-script: she chooses, but she never faints with unused heals because the option
# was never surfaced). Named/tunable.
BATTLE_CRIT_FRAC = 0.30
BAG_CURSOR = 0x0203AD04      # u8 in-battle bag LIST row cursor (recon_itemuse triangulation 2026-06-27,
#                             adjacent to GBAG_POCKET 0x0203AD02; verified to step 0->1 down the list)
BAG_SCROLL = 0x0203AD0A      # u16 itemsAbove[0] — rows hidden above the window. TRUE selection =
#                             BAG_CURSOR + BAG_SCROLL, and BOTH persist between bag opens (derived +
#                             press-verified by recon_bagscroll 2026-07-07: the e4_run2 'selected but
#                             NOT consumed' class was A landing on Revive/CANCEL off a stale scroll).
# FIGHT move-list cursor + menu-mode (recon_movecursor_derive 2026-06-28). MOVE_CURSOR is a single 0..3
# index in the 2x2 grid (TL0 TR1 / BL2 BR3): DOWN +2 (row), RIGHT +1 (col) — sits 4 B after the action
# cursor 0x02023FF8. MENU_MODE == 1 on the FIGHT/BAG/POKEMON/RUN action menu, == 2 when the move list is
# open. These let the move-list nav use RAM READBACK (open-detect + per-press verify) instead of blind
# taps + pixel detection, which WEDGE on the long-running core (the keystone freeze-spin).
MOVE_CURSOR = 0x02023FFC
MENU_MODE = 0x02023E82
# In-battle PARTY-LIST cursor = gPartyMenu.slotId (recon_partycursor_derive 2026-06-28): DOWN increments,
# UP decrements (0=lead, 1=2nd, ... + a CANCEL entry past the last mon). Lets the in-battle SWITCH nav by
# readback instead of blind DOWN*slot taps that wedge/mis-land on the long core (the gated switch's gap).
PARTY_CURSOR = 0x02020777
_ITEMS_POCKET_OFF = 0x0310   # SaveBlock1 Items pocket (potions + status cures live here), 42 slots
# Gen-3 item ids for the in-battle instinct (CANDIDATES; the use is self-verified by the item count
# dropping, so a wrong id simply doesn't fire -> 'failed' -> keep fighting, never a wrong action).
_HEAL_ITEMS_PREF = (19, 20, 21, 22, 13)   # Full Restore, Max, Hyper, Super, Potion (strongest usable first)
_REVIVE_ITEMS_PREF = (25, 24)             # Max Revive, Revive
_ETHER_ITEMS_PREF = (37, 36, 35, 34)      # Max Elixir, Elixir, Max Ether, Ether (all-move first)
# Kanto species whose ability is ALWAYS Levitate -> Ground does NOTHING despite the chart's x2
# (Agatha's gengars: EQ 'connects' on paper, so chart-only famine never fires while EQ has PP).
# Game-knowledge inline (portability debt: belongs in gamedata/ when the ability layer generalizes).
_LEVITATE_SPECIES = {92, 93, 94, 109, 110}  # Gastly, Haunter, Gengar, Koffing, Weezing


def _eff(move, enemy):
    """Move-vs-foe effectiveness = the type chart WITH the ability layer on top. Levitate blocks
    ALL Ground moves (status included, Gen 3), so Ground into a Levitate species is x0 despite the
    chart's x2 — run21 donated 4+ free turns per Gengar picking EQ as 'super-effective' for zero
    damage. Every move-pick judgment goes through THIS; pol.effectiveness stays the raw chart."""
    enemy = enemy or {}
    if move.get("type") == "ground" and enemy.get("species") in _LEVITATE_SPECIES:
        return 0.0
    return pol.effectiveness(move.get("type", "normal"), enemy.get("types") or [])
_STATUS_CURE_ITEM = {"poison": 14, "burn": 15, "freeze": 16, "sleep": 17, "paralysis": 18}
_FULL_HEAL = 23

# ── ANTI-WEDGE BATTLE FLOOR (run-existential) ─────────────────────────────────────────────────────
# A turn that never RESOLVES — no PP drop, no HP change, no faint — livelocks the fight. The trigger we
# hit live: every move depleted (Sleep Powder at 0 PP). The game refuses a 0-PP move ("There's no PP
# left for this move!"), that text flicker keeps changing the screen so the cosmetic `stall` guard
# resets forever, and the enemy never gets a turn either. This floor counts UNRESOLVED turns (ONLY a
# real resolution clears it, so the flicker can't hide the wedge) and, past the threshold, ESCAPES:
# a WILD battle is FLED (self-preservation — a watchable, in-character retreat; the sibling to the
# overworld deep-wedge floor, but inside combat); a TRAINER battle (un-fleeable) is aborted LOUD. On by
# default — a frozen session is strictly worse than a flee — but disable-able / tunable. Capability-
# not-script: she still picks her move every turn; this only catches the dead-end where NO move resolves.
BATTLE_FLEE_FLOOR = os.getenv("POKEMON_BATTLE_FLEE_FLOOR", "1") == "1"
UNRESOLVED_FLEE_AT = int(os.getenv("POKEMON_UNRESOLVED_FLEE_AT", "3"))
# B-1 — IN-BATTLE MATCHUP SWITCHING (E4-critical). The matchup MATH is offline-verified, but the
# party-menu ACTUATION (cursor nav on a long-running libmgba core) is UNVERIFIED — the standing
# menu-nav lesson. So it's GATED OFF by default until a live control passes (arm POKEMON_BATTLE_SWITCH=1
# with Jonny watching). FAIL-SAFE regardless: if a switch doesn't confirm, she backs out and fights —
# never wedges. When off, she still AVOIDS ineffective moves (that path is on + verified).
# ARMED 2026-07-05: the in-battle switch is now VERIFIED (recon_switch3.py + direct _switch_to_slot test on
# the canon 3-mon fixture — SWITCHED ivysaur->spearow). The wedge was a WRONG-ADDRESS derivation (the old
# PARTY_CURSOR=0x2020777 was a shadow byte); the real nav is BLIND DOWN*(slot+1) like the working
# _force_switch (live cursor is a heap struct). Fail-safe B-out on any miss = never wedges, so default-on is safe.
BATTLE_SWITCH_ENABLED = os.getenv("POKEMON_BATTLE_SWITCH", "1") == "1"
# ── NS23: LOAD-SHARE between two SE attackers (the E4-Champion team-depth lever). The anti-churn rule
# (an SE attacker >=2x STAYS and swings, _best_switch_slot) is load-bearing but makes a LONE specialist
# solo a whole gauntlet to death while a healthy party-mate that is ALSO SE idles — e4_tactical_v2:
# Lapras L60 solo'd 5 of Gary's 6 (fainting at Gyarados before Charizard) while a healthy L71 Venusaur
# (Razor Leaf 4x on Rhydon) sat unused -> whiteout. The refinement: when the SE active is CRITICALLY low
# AND a HEALTHY reserve is ALSO SE on this foe, rotate to that fresh SE body. Churn-safe: the target is
# itself >=2x, so once it's out the same anti-churn return keeps it in (no SE<->non-SE ping-pong), and HP
# only decreases in reserve so there's no oscillation. Flag-gated DEFAULT OFF — more switches = more
# white-box-menu actuation exposure on the LIVE path, so flip only after an attended frame-grab pass.
BATTLE_LOAD_SHARE = os.getenv("POKEMON_BATTLE_LOAD_SHARE", "0") == "1"
SWITCH_SHARE_HEALTHY_FRAC = float(os.getenv("POKEMON_LOAD_SHARE_HEALTHY_FRAC", "0.5"))
# The PRE-HEAL load-share rotates a WORN SE attacker (<=this) to a NEAR-FULL SE partner INSTEAD of
# spending a heal — the ns23 headless E4 finding: the critical-HP gate in _best_switch_slot is preempted
# by the survival-instinct heal (SURVIVAL FIRST in run()), so it never fired; the whiteout is a Full-
# Restore FAMINE (4 FRs, all spent by room 4, FR x0 at the Champion). Rotating to a fresh SE body spreads
# the gauntlet's damage across two attackers AND conserves the scarce heals. Churn-safe: the near-full
# gate is monotonic (a benched mon doesn't regen, so it can't bounce back above near-full to be re-picked).
SWITCH_SHARE_WORN_FRAC = float(os.getenv("POKEMON_LOAD_SHARE_WORN_FRAC", "0.5"))
SWITCH_SHARE_NEARFULL_FRAC = float(os.getenv("POKEMON_LOAD_SHARE_NEARFULL_FRAC", "0.85"))
# WHIFF-SPIRAL BREAKER (2026-07-10, night shift 9 — the S.S. Anne Gary ROOT CAUSE). Accuracy-lowering foe
# moves (Sand-Attack/Smokescreen/Kinesis) debuff the active mon until it MISSES every swing, freezing the
# foe's HP while our PP drains -> famine -> a LOSS even at a crushing level lead (a full-PP Venusaur L32
# lost Gary this way, on repeat). No existing trigger catches "my move FIRED but did no damage" (a miss),
# because a PP drop reads as a resolved turn. Gen-3 resets stat stages (incl. accuracy) on SWITCH-OUT, so
# the fix is a switch OUT+back to clear the debuff. Bounded per battle so it can never switch-loop.
WHIFF_BREAKER_ENABLED = os.getenv("POKEMON_WHIFF_BREAKER", "1") == "1"
WHIFF_SPIRAL_AT = int(os.getenv("POKEMON_WHIFF_SPIRAL_AT", "3"))          # consecutive misses -> reset
# FOE-EVASION CORRECTION (2026-07-10, night shift 17 — the BADGE-5 KOGA ROOT CAUSE). The switch-out reset
# only clears OUR accuracy debuff (Sand-Attack et al.); it CANNOT reset the FOE's evasion (Minimize/
# Double-Team) — Koga's Muk minimizes, every Cut whiffs, and the breaker mis-read it as a self-debuff and
# repeatedly benched L52 Venusaur (the sole carry) for L13 fodder that instantly fainted -> whole bench
# fed to death -> PP famine -> blackout, on repeat (4 straight Koga losses). TWO guards: (a) a reset is
# worth it only when there's a reserve WORTH benching the ace for — never sacrifice the carry for a mon far
# below its level (solo-carry -> fight on: a miss still lands ~1-eva% and the ace had Muk at 9/130 before
# the switch threw it). (b) Cap resets at 2: a true self-debuff clears on the FIRST switch; if whiffs resume
# it's foe-evasion and more switches only feed the bench. Revert: POKEMON_WHIFF_MAX_RECOVERIES=6.
WHIFF_MAX_RECOVERIES = int(os.getenv("POKEMON_WHIFF_MAX_RECOVERIES", "2"))  # accuracy-resets per battle
WHIFF_RESERVE_LEVEL_BAND = int(os.getenv("POKEMON_WHIFF_RESERVE_BAND", "15"))  # reserve must be within N lv
# PARTICIPATION-XP GRIND SWITCH (Task B fix — the autonomous underlevel cure). When grinding the weak
# team, the weak mon leads (so it's "sent out" and eligible for XP) but is ONE-SHOT before it can earn
# any — so it gains nothing while the ace mops up and takes the XP (the live look-ahead proved this).
# The real-player fix: lead the weak mon, turn-1 SWITCH to the ace — the weak mon participated (gets a
# share of XP) and never takes a hit (it's benched before the enemy's turn), while the tanky ace KOs.
# `PROTECT_LEAD_GRIND` is toggled by campaign.grind_weak_members AROUND its grind battles only (off in
# normal play). FAIL-SAFE: a switch that doesn't confirm falls through to fighting (never wedges).
# DEFAULT OFF: the live look-ahead proved the in-battle party-menu actuation WEDGES the wild battle on
# this long-running core (the standing menu-nav-on-long-core risk — same reason BATTLE_SWITCH is gated).
# A wedged grind battle returns 'stuck' and blacks her out. Kept (code-complete) for when the in-battle
# switch actuation is made reliable (it's the real weak-mon-leveling cure for the E4); until then OFF, and
# the underlevel grind leans on other paths. Arm with POKEMON_GRIND_SWITCH=1 once switch nav is verified.
# RE-GATED OFF 2026-07-05 (3rd attempt, tripwire): grind(fragile=True) reachable-grass filter is NECESSARY but
# INSUFFICIENT — she still strands at Route-4 (84,15) because the strand arrives via a DIFFERENT path than
# grass-pacing (a battle_loss mid-travel, or the grind anchor itself being set inside the east area). The real
# fix is bigger: route the weak-grind to a SAFE MAP (Route 3: flat, L3-6, Center-reachable via Pewter) rather
# than Route 4 at all, OR make a true strand (heal 'stuck', no reachable Center) force an escape-hatch reload
# that recovers. Flagged in STATE §0 as the top rebuild item. Switch MECHANISM + BATTLE_SWITCH stay armed/verified.
GRIND_SWITCH_ENABLED = os.getenv("POKEMON_GRIND_SWITCH", "1") != "0"   # DEFAULT-ON 2026-07-05: chain
#   proven end-to-end (bench L8/10->16 via participation XP -> Gary first-try -> bridge -> Bill -> TICKET)
PROTECT_LEAD_GRIND = False                 # set True by grind_weak_members only; read per battle in run()
# SELECTIVE SOLO (2026-07-11 NS#26 — the bench-leveling KILL-XP lever, the frontier #2). The participation
# GRIND SWITCH hands the KO to the ace, so the fielded weak lead banks only a SHARE of participation XP —
# that (not PP-famine, which the ACE-DOWN guard handles) is the real throttle behind the slow endgame bench
# climb (L28->L30 over a ~10-min stint). WHEN the weak lead SAFELY out-levels THIS foe (>= SOLO_OVERLEVEL_MARGIN
# above it) it one-shots the wild taking ~0 damage — no faint, no in-battle heal (so no white-box menu exposure)
# — so let it SOLO for the FULL kill XP (~2x the share). Implemented by SUPPRESSING the grind switch (ace=None ->
# fall through to fight) while KEEPING PROTECT_LEAD_GRIND True so the MATCHUP switch stays suppressed (line ~2735
# gates on `not PROTECT_LEAD_GRIND` -> no strand/churn). Unlike the old SOLO_WEAK_GRIND this never touches an
# in-battle switch (it removes one) and is per-FOE self-correcting (a higher wild still gets the ace-protect
# switch). Default OFF, verify-gated: needs a fresh multi-gym look-ahead confirming the bench climbs FASTER with
# no faint-thrash/park/matchup-churn before the flip. Tune the margin at that look-ahead.
SOLO_OVERLEVEL_GRIND = os.getenv("POKEMON_SOLO_OVERLEVEL_GRIND", "0") == "1"   # default OFF; verify-gated
SOLO_OVERLEVEL_MARGIN = int(os.getenv("POKEMON_SOLO_OVERLEVEL_MARGIN", "8"))   # weak lead >= foe + this -> solo
# SLEEP-LOCK (re-apply sleep vs a super-effective hard-hitter). DEFAULT-ON 2026-07-06 (war-room call):
# the reason it was gated — long fights exposing the move-list wedge — is FIXED (the a4ca84f cursor
# readback + the 2026-07-05 _movelist_open_verified immortal-battle fix), and the whiff SAFETY CAP
# (max 4 misses/foe, the Sand-Attack lesson) bounds the worst case. A live GO watch deserves the
# correct strategy, not the gated one. Disarm with POKEMON_SLEEP_LOCK=0 if a long-fight wedge recurs.
SLEEP_LOCK_ENABLED = os.getenv("POKEMON_SLEEP_LOCK", "1") != "0"
# party-mon STATUS1 (u32 @ +0x50 in the 100-byte struct) — the reliable party-only block (== campaign).
_P_STATUS = 0x50
_ST_SLEEP, _ST_PSN, _ST_BRN, _ST_FRZ, _ST_PAR, _ST_TOX = 0x07, 0x08, 0x10, 0x20, 0x40, 0x80


def _decode_status(s):
    if s & _ST_SLEEP:
        return "sleep"
    if s & (_ST_PSN | _ST_TOX):
        return "poison"
    if s & _ST_BRN:
        return "burn"
    if s & _ST_FRZ:
        return "freeze"
    if s & _ST_PAR:
        return "paralysis"
    return None


def _hp_frac(mon):
    return (mon["hp"] / mon["maxhp"]) if mon and mon["maxhp"] else 1.0


# FOES-SEEN LEDGER (the attach-time rival miss, tower4 2026-07-07): every species READ LIVE on the
# field this battle (display struct, HP>0 — can't be stale). When the observer ATTACHES to an
# already-running scene battle (Gary's Tower approach), gEnemyParty still holds the PREVIOUS fight
# at scan time, so the campaign's start-scan rival detection misses; it re-checks THIS ledger after
# the fight. Reset at each run() entry; module-level so the campaign wrapper can read it without
# holding the (per-battle) agent instance.
LAST_FOES_SEEN = set()

# F-7(c) slice 2 (2026-07-08): True once the IN-DRAIN level-up beat fired for the current
# battle — play_live's post-battle level check reads this to dedup its own (drain-late) emit.
# Module-level for the same reason as LAST_FOES_SEEN: the wrapper never holds the agent.
LEVELUP_EMITTED = False


class BattleAgent:
    def __init__(self, bridge, on_event=None, render=None, hold_frames=HOLD,
                 pace=None, owner="agent", log=print, choose=None):
        self.b = bridge
        self.on_event = on_event or (lambda s, **k: print(f"   [EVENT] {s}"))
        self.render = render or (lambda: None)
        self.hold = hold_frames
        self.pace = pace                 # optional: called at a beat to yield to her voice
        self.owner = owner
        self.log = log
        # BATCH 2 PART B: optional SOUL ORACLE (choose(kind, options, ctx)->pick). When a mon is crit-low
        # or afflicted AND a matching item is in the bag, the in-battle loop OFFERS "use a potion/cure" to
        # her; she decides (capability-not-script). None -> the instinct is silent (pure policy battle).
        self.choose = choose
        self.b.set_input_owner(owner)    # single deliberate owner; phantoms dropped+logged
        self._prev = None
        self._started = False
        self._enemy_fainted = False
        self._we_fainted = False
        self._no_progress = 0          # consecutive action-menu visits with no battle change
        self._last_progress = None
        # menu-agnostic recovery (the live party-submenu trap): global stall watchdog
        self._recovery_attempts = 0
        self._last_global = None
        self._stale = 0
        self._acted_once = False       # have we landed/attempted a move yet? (the battle
                                       # intro+settle legitimately makes NO hp change - the
                                       # global-stall watchdog must not fire during it)
        self._unresolved_turns = 0     # ANTI-WEDGE FLOOR: turns that never RESOLVED (no PP drop/
                                       # HP change/faint). Cleared only by a real resolution, so the
                                       # 0-PP "no PP left!" flicker can't reset it like `stall` does.
        self._skip_streak = set()      # FIX 1: every move slot that failed to fire THIS streak — so she
                                       # rotates through her WHOLE moveset (never re-spams a dead/0-PP
                                       # move) and only flees once all are exhausted. Clears on any fire.
        self._win_emitted = False      # F-7(c): the certain-win beat already fired at the faint —
                                       # _finish must not voice the same win again 5-15s later.
        self._catching = False         # F-7(c) guard: KOing a CATCH target is a failure, never a
                                       # "you won" beat — set for the catch_pokemon flow.

    # ── input (owner-attributed) ───────────────────────────────────────────────
    def _tap(self, key):
        self.b.press(key, self.hold, self.hold, self.render, owner=self.owner)

    def _wait(self, frames):
        for _ in range(frames):
            self.b.run_frame(); self.render()

    def _is_trainer_battle(self):
        """BATTLE_TYPE_TRAINER (0x08). Valid in-battle. Wild = can flee, trainer = can't."""
        return bool(self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08)

    def _enemy_live_remaining(self):
        """F-7(c): how many LIVE mons remain in gEnemyParty (valid species, HP > 0). The party
        struct's plaintext HP (+0x56) is synced on every damage write (pret: Cmd_datahpupdate
        SetMonData's back to the party), so at the moment the active foe faints this answers
        'is the battle DECIDED?' — 0 means no switch-in can come, the win is certain.
        Defensive: any read error returns a big count (never a false battle-over)."""
        try:
            n = 0
            for s in range(6):
                sp = st.read_enemy_species(self.b, s)
                if not (1 <= sp <= 411):
                    continue
                if self.b.rd16(ram.GENEMY_PARTY + s * st.PARTY_MON_SIZE + 0x56) > 0:
                    n += 1
            return n
        except Exception:
            return 99

    # ── events + performance beats ─────────────────────────────────────────────
    def emit(self, summary, beat=False, tier=None):
        """NEUTRAL game-event -> her self. beat=True is a PERFORMANCE moment: yield the
        floor so her voice lands before the hands advance (brisk on non-beats). `tier` (Phase 2D)
        forwards an explicit salience tier for big beats (shiny/legendary) — the live on_event is
        voice.emit which reads it; the default/headless sinks accept **k and ignore it."""
        if tier is not None:
            self.on_event(summary, tier=tier)
        else:
            self.on_event(summary)
        if beat and self.pace:
            self.pace(summary)

    # ── PRESS-UNTIL-SETTLED core (rebuilt 2026-06-23) ──────────────────────────
    # The lesson of the cursor-desync saga: GBATTLE_PHASE is a free-running FRAME COUNTER,
    # not a phase register, and the action cursor only obeys the D-pad once the menu is
    # genuinely SETTLED - acting mid-transition gets every press eaten. So we never race
    # the emulator to read state at the right frame. We advance until the game is plainly
    # WAITING for input (RAM static), THEN navigate the cursor with eaten-press tolerance
    # and VERIFY the move actually registered (HP moved). Never a blind A/B that could open
    # the wrong submenu or select RUN (flee).
    def _note_foe(self, state):
        """Record a LIVE-read foe species into LAST_FOES_SEEN (see the module-level note)."""
        try:
            e = (state or {}).get("enemy") or {}
            if 1 <= e.get("species", 0) <= 411 and e.get("hp", 0) > 0:
                LAST_FOES_SEEN.add(e["species"])
        except Exception:
            pass

    def _bstate(self):
        s = st.read_battle(self.b)
        return (s["enemy"]["hp"], s["ours"]["hp"]) if s else None

    def _settle(self, need=10, timeout=900):
        """Advance frames (no input) until the battle is WAITING for input: enemy+our HP
        hold steady for `need` consecutive frames. Narrates HP diffs en route so her voice
        stays live. Returns when settled or the battle ends/timeout."""
        last, stable = None, 0
        for _ in range(timeout):
            if not st.in_battle(self.b):
                return
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur)
                self._prev = cur
            key = (cur["enemy"]["hp"], cur["ours"]["hp"]) if cur else None
            stable = stable + 1 if key == last else 0
            last = key
            if stable >= need:
                return
            self.b.run_frame(); self.render()

    # NOTE: the slot-0 move-swap hack (_swap_battle_moves), the action-cursor walker that read
    # the stale 0x02023FF8 latch (_goto_fight), and the engage-wiggle _nav_to were RETIRED
    # 2026-06-25 once real menu nav worked (phantom-A fix). Move selection now navigates the
    # real move list (_home_to_fight + _nav_move in _select_and_verify). flee() still uses
    # _goto_run below (proven working); a clean-nav rewrite of it is a later follow-up.

    def _goto_run(self, tries=10):
        """READ the action cursor and walk it to RUN (bottom-right). Eaten-press tolerant.
        Returns True only when the cursor is confirmed on RUN (3); False if not at the menu."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_RUN:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("RIGHT"); self._tap("DOWN")
            elif c == ram.ACT_BAG:
                self._tap("DOWN")
            elif c == ram.ACT_POKEMON:
                self._tap("RIGHT")
            else:
                return False                          # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_RUN

    def flee(self, max_seconds=90):
        """RETREAT: flee a WILD battle (the wounded heal-return path - fighting our way back
        through the grass is what blacks us out). Forced TRAINER battles can't be fled, so we
        WIN those via the normal engine. Selects RUN with eaten-press tolerance and verifies the
        battle actually ended (in_battle clears). Returns 'fled' / 'win' / 'loss' / 'stuck'."""
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "fled"
        self._prev = st.read_battle(self.b)
        self._reach_first_menu(t0, max_seconds)
        # FAINTED-LEAD BATTLE START (the heal-excursion timeout class): the wild battle opens on the
        # FORCED "Choose a POKéMON" screen — no action menu exists until someone is sent out, so the
        # old path misread it (ours.hp==0 -> false 'loss'; or run()'s A/B drain oscillated 90s). Send
        # the healthy reserve out FIRST, then run away like a normal fight.
        if self._party_screen():
            cur0 = st.read_battle(self.b)
            if (cur0 is None or cur0["ours"]["hp"] == 0) and self._healthy_reserve_slot() is not None:
                self.log("   [engine] flee: battle opened on the forced send-out screen -> sending a "
                         "healthy reserve, then running")
                self._force_switch()
                self._prev = st.read_battle(self.b)
        if self._is_trainer_battle():                 # can't flee a trainer -> WIN it
            return self.run(max_seconds=max_seconds)
        for _ in range(3):                            # ensure the ACTION menu, not the move list:
            if not self._in_move_list():              # _white_box can't tell them apart, so RUN nav
                break                                 # from an open move-list fires a move + never
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)  # escapes (flee 'stuck'
            self._wait(10)                            # loop). Same class as the catch bag-nav bug.
        for _ in range(40):
            if not st.in_battle(self.b):
                return "fled"
            self._settle()
            if not st.in_battle(self.b):
                return "fled"
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] == 0:
                return "loss"
            if self._white_box() and self._goto_run():
                self._tap("RIGHT"); self._tap("DOWN") # engage (eaten-press; RUN stays at corner)
                self._tap("A"); self._wait(20)        # confirm RUN -> "got away safely" / retry
            else:
                self._advance_text()                  # advance the escape/"can't escape" message
        return "fled" if not st.in_battle(self.b) else "stuck"

    # ── autonomous CATCH (real bag nav; the phantom-A bug that made this impossible was
    # fixed 2026-06-25 — see [[pokemon-battle-menu-nav-cracked]]). Flow, screenshot- and
    # control-verified (party N->N+1) on forest_battle.state: action menu opens on FIGHT each
    # turn -> RIGHT = BAG -> A opens the bag (lands on the Poké Balls pocket, cursor on the
    # ball) -> A selects -> "POKé BALL is selected. USE/CANCEL" (cursor on USE) -> A throws.
    # Then advance the catch sequence (B dismisses the "give a nickname?" Yes/No). We SETTLE
    # after the bag-open fade (acting mid-transition = eaten, the same quirk as the move list).
    def _ball_count(self):
        """REAL number of Poké Balls (item id 4) in the bag's balls pocket. Item ids are plain;
        the QUANTITY is XOR-encrypted with the SaveBlock2 security key (FireRed: SaveBlock2+0xF20,
        low 16 bits). Decrypting it lets callers gate on the true count (e.g. throw-until-caught,
        out-of-balls handling) instead of mere presence. 0 -> can't throw."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for i in range(16):
            iid = self.b.rd16(sb1 + 0x430 + i * 4)
            if iid == 0:
                break
            if iid == 4:
                return self.b.rd16(sb1 + 0x430 + i * 4 + 2) ^ key
        return 0

    def throw_ball(self, max_seconds=45):
        """Throw a Poké Ball at a WILD foe via real menu nav. Returns 'caught' (party+1),
        'broke_free' (battle continued/ended w/o catch), 'trainer' (can't catch), 'no_balls',
        or 'stuck'. Assumes a fresh/settled action menu (turn start). Control-proven party+1."""
        t0 = time.time()
        if self._is_trainer_battle():
            return "trainer"
        if self._ball_count() == 0:
            self.log("   [engine] throw_ball: no Poké Balls in the bag")
            return "no_balls"
        if not self._white_box():
            self._reach_first_menu(t0, max_seconds)
        self._settle()
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        balls_at_start = self._ball_count()           # baseline BEFORE the bag opens (throw-verify gate)
        if os.environ.get("CATCH_RECON"):             # RECON: what menu are we actually on at throw-start?
            try:
                _s = st.read_battle(self.b)
                self.log(f"      [catch-recon] throw-start: white_box={self._white_box()} "
                         f"in_move_list={self._in_move_list()} balls={self._ball_count()} "
                         f"in_battle={st.in_battle(self.b)} enemy_hp="
                         f"{_s['enemy']['hp'] if _s else '?'}/{_s['enemy']['maxhp'] if _s else '?'}")
            except Exception as _e:
                self.log(f"      [catch-recon] read err {_e}")
        # ENSURE THE ACTION MENU (not the move list): a prior weaken move (_fire_move/_weaken_hp) can
        # leave the FIGHT move-list open, and _white_box() can't tell action-menu from move-list — so
        # navigating to BAG from the move list fires a MOVE instead of opening the bag (no ball thrown
        # -> the catch spins, never consuming a ball). Back out to the action menu first.
        for _ in range(3):
            if not self._in_move_list():
                break
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        # SETTLE ONTO THE ACTION MENU before opening the bag: after a broke-free the foe's turn / poison
        # text can still be up, and opening the bag mid-text EATS the throw (the flaky 2nd-throw 'stuck').
        # menu_up==1 (+ white panel) is the reliable 'action menu is waiting' signal. Advance blue text;
        # back out of a stray move list; bounded so a genuinely wedged box still falls through (and the
        # pocket-nav/throw-verify below aborts loudly rather than silently spinning).
        for _ in range(30):
            if self._white_box() and self.b.rd8(ram.GBATTLE_MENU_UP) == 1:
                break
            if self._white_box():                     # white panel but not the action menu -> move list
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            else:
                self._advance_text()                  # blue dialogue/animation box -> advance it
        # OPEN THE BAG and NAVIGATE TO THE POKé BALLS POCKET (root-caused 2026-06-27 — the long-standing
        # "141 dead throws / ball count never decrements" bug). The in-battle bag opens on the LAST-VIEWED
        # pocket, NOT always Poké Balls: on Route 3 it opens on the (empty) ITEMS pocket, so the old blind
        # UP+A+A selected CANCEL and threw NOTHING. FRLG pocket layout is FIXED (Items=0, Key Items=1,
        # Poké Balls=2, ...), so we STEER the LIVE pocket index (ram.GBAG_POCKET) to the balls pocket
        # WITHOUT pressing A on any other pocket (A on an empty pocket's CANCEL poisons pocket-switching).
        # Being ON pocket 2 with the cursor at the top IS a Poké Ball (we early-returned 'no_balls' if the
        # count were 0), so we don't trust the STALE gSpecialVar_ItemId — we press A and VERIFY a ball
        # actually LEFT (count dropped vs throw-start / caught / battle ended), retrying an eaten select/
        # confirm. Selecting may itself throw (no USE prompt) or need one more A; re-checking before each
        # press never double-throws. Control-proven: Route 3 fail-state pocket 0->1->2, balls 5->1, caught.
        def _thrown():
            return (self._ball_count() < balls_at_start or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0
                    or not st.in_battle(self.b))
        # OPEN THE BAG and VERIFY it actually opened before trusting the pocket var: if the open-A is
        # eaten (we're not fully settled) we stay at the action menu and ram.GBAG_POCKET reads STALE
        # (e.g. 2 from a prior throw) — a false "on balls pocket" that then mashes A into the move list.
        # The bag being open == the white action-menu panel is GONE (_white_box False); retry the open.
        opened = False
        for _ in range(4):
            self._home_to_fight()                     # FIGHT is cursor home -> RIGHT = BAG (re-home each retry)
            self._tap("RIGHT")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # open bag
            self._wait(50)                            # wait OUT the bag-open fade (acting on the fade = eaten)
            if not self._white_box():                 # left the action-menu panel -> the bag is open
                opened = True; break
        if not opened:
            self.log("   [engine] !! throw_ball: bag would not open (open-A eaten) — aborting LOUDLY")
            return "stuck"
        on_balls_pocket = False
        for _ in range(8):                            # steer the live pocket index toward Poké Balls
            if self.b.rd8(ram.GBAG_POCKET) == ram.POCKET_POKE_BALLS:
                on_balls_pocket = True
                break
            self._tap("RIGHT" if self.b.rd8(ram.GBAG_POCKET) < ram.POCKET_POKE_BALLS else "LEFT")
            self._wait(12)
        if os.environ.get("CATCH_RECON"):
            self.log(f"      [catch-recon] bag nav: pocket={self.b.rd8(ram.GBAG_POCKET)} "
                     f"on_balls_pocket={on_balls_pocket} item={self.b.rd16(ram.GSPECIALVAR_ITEMID)}")
        if not on_balls_pocket:
            self.log("   [engine] !! throw_ball: couldn't reach the Poké Balls pocket — aborting LOUDLY")
            for _ in range(4):                        # leave the menu clean for the caller
                if self._white_box():
                    break
                self.b.press("B", 2, 12, self.render, owner=self.owner); self._wait(8)
            return "stuck"
        self._tap("UP"); self._wait(8)                # top of the balls pocket = the ball
        # SELECT + THROW, then STOP the instant a ball leaves. Press A (select -> USE/throw); the throw
        # removes the ball from the bag IMMEDIATELY (count drops) — so after each A we POLL for the throw
        # to register and break the MOMENT it does. This is critical: if we kept mashing A, the extra
        # press lands on the post-catch "give a nickname? [YES]" prompt and opens the naming KEYBOARD,
        # wedging the next throw (the forest 2nd-throw bug). Selecting may itself throw (1 A) or need a
        # USE confirm (2 A); the per-A poll handles both and retries an eaten press. LOUD abort, no spin.
        for _ in range(4):
            if _thrown():
                break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            for _ in range(40):                       # watch for the ball to leave BEFORE pressing again
                if _thrown():
                    break
                self.b.run_frame(); self.render()
        if not _thrown():
            self.log("   [engine] !! throw_ball: ball selected but no throw consumed a ball — aborting LOUDLY")
            return "stuck"
        self.emit("alright — throwing a Poké Ball", beat=True)
        while time.time() - t0 < max_seconds:
            if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                self.emit("gotcha — it's caught!", beat=True)
                # The party grew (catch banked) BUT the "give a nickname? [YES/NO]" prompt is still up
                # (FRLG adds the mon, then asks). Decline with B (never leave it for the next op to A into
                # -> naming keyboard) and let the battle EXIT to the overworld before returning.
                for _ in range(20):
                    if not st.in_battle(self.b):
                        break
                    self._wait(10); self.b.press("B", 2, 12, self.render, owner=self.owner)
                return "caught"
            if not st.in_battle(self.b):
                # Battle ended. On a CATCH the "give a nickname? [YES/NO]" prompt HOLDS the party-increment
                # until dismissed — and its cursor defaults to YES, so an A opens the naming keyboard. The
                # old 40 BLANK frames here neither dismissed it nor saw the party tick, so a real catch
                # returned 'broke_free' and LEFT the prompt up -> the next throw's A typed into the keyboard
                # (the forest 2nd-throw wedge). Press B (decline) while watching for the party to grow (the
                # unfakeable catch signal): finalizes the catch AND never leaves a prompt dangling.
                for _ in range(20):
                    if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                        break
                    self._wait(10)
                    self.b.press("B", 2, 12, self.render, owner=self.owner)
                if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                    self.emit("gotcha — it's caught!", beat=True)
                    return "caught"
                return "broke_free"
            # The ball BROKE and we're back at the ACTION MENU - the turn is over. Gate this on the
            # TRUE action menu (white panel AND menu_up==1), NOT white_box alone: a white frame flashes
            # DURING the catch sequence, and returning then leaves the "give a nickname?" prompt up for
            # the caller to A into (-> naming keyboard wedge). menu_up==1 only holds at the real action
            # menu, which a CATCH never reaches (the battle ends), so this returns broke_free only on a
            # genuine break. Return WITHOUT pressing: B here = RUN (would flee). The caller re-throws.
            if self._white_box() and self.b.rd8(ram.GBATTLE_MENU_UP) == 1:
                return "broke_free"
            # B-ONLY advance for the catch-sequence BLUE boxes (the "broke free!" text and the
            # post-catch "give a nickname?" Yes/No, which defaults YES — an A would open the naming
            # keyboard and wedge it). B dismisses them safely. Wait so we never mash into animation.
            self._wait(18)
            self.b.press("B", 2, 12, self.render, owner=self.owner)
        return "stuck"

    # ── autonomous CATCH FLOW (mirrors the proven live play: weaken/status, then commit to throws)
    _SLEEP_MOVES = {79, 147, 95, 47, 142, 1}        # Sleep Powder, Spore, Hypnosis, Sing, Lovely Kiss
    _STATUS_MOVES = _SLEEP_MOVES | {77, 78, 86}     # + PoisonPowder, StunSpore, ThunderWave
    CATCH_WEAKEN_CEIL = 0.85   # if she CAN'T weaken (depleted PP) AND the foe is still above this HP
    #                            fraction, don't dump balls into a low-odds full-HP catch — flee + heal.

    def _can_weaken(self, state):
        """True iff she has a move that can actually SOFTEN the foe — a usable status move OR a usable
        damaging move (PP>0). False = fully depleted: can't sleep it, can't chip it. The catch flow uses
        this to refuse ball-dumping a near-full-HP foe she has no way to weaken (the live ball-burn)."""
        moves = state["ours"]["moves"]
        has_status = any(m.get("id", 0) in self._STATUS_MOVES and m.get("pp", 0) > 0 for m in moves)
        has_damage = any(m.get("id", 0) and m.get("pp", 0) > 0 and m.get("id", 0) not in self._STATUS_MOVES
                         for m in moves)
        return has_status or has_damage

    def _catch_weaken_move(self, state):
        """Slot index of a move to SOFTEN the wild foe before throwing - prefer a SLEEP move (asleep
        = x2 catch rate in Gen 3 AND it stops the foe attacking us), else another status. Returns
        None if the foe is already low (just throw) or we have no usable status move."""
        foe = state["enemy"]
        if foe["maxhp"] and foe["hp"] <= foe["maxhp"] * 0.35:
            return None                                  # weak enough already
        moves = state["ours"]["moves"]
        for pool in (self._SLEEP_MOVES, self._STATUS_MOVES):
            for i, m in enumerate(moves):
                if m.get("id", 0) in pool and m.get("pp", 0) > 0:
                    return i
        return None

    def _weaken_hp(self, target_frac=0.40, max_hits=4):
        """Chip the wild foe's HP into the catchable band so a HANDFUL of balls suffices (a status
        alone leaves it near full HP -> 5 balls broke free). Fires the LOWEST-base-power damaging move
        one hit at a time, re-reading HP, and STOPS once HP <= target_frac (faint-guard: never swing
        at an already-low foe; one-at-a-time re-check avoids overkill). Best-effort — a stray faint
        just means catch_pokemon returns 'fainted' and the wander finds another wild."""
        for _ in range(max_hits):
            state = st.read_battle(self.b)
            if not state or not st.in_battle(self.b):
                return
            foe = state["enemy"]
            if not foe.get("maxhp") or foe["hp"] <= 0:
                return
            if foe["hp"] / foe["maxhp"] <= target_frac:
                return                                  # already in the catchable band
            cand = [(i, m.get("id", 0)) for i, m in enumerate(state["ours"]["moves"])
                    if m.get("id", 0) and m.get("pp", 0) > 0 and m.get("id", 0) not in self._STATUS_MOVES]
            if not cand:
                return                                  # no damaging move with PP -> just throw
            cand.sort(key=lambda im: st.move_info(self.b, im[1])[1] or 0)   # gentlest (lowest power)
            self._fire_move(cand[0][0])

    def _fire_move(self, idx):
        """Open the move list, navigate to slot idx, fire it + verify it executed (PP drop / HP
        change / battle end). Separate from _select_and_verify (which policy-PICKS a move) so the
        proven fight path stays untouched; used by catch_pokemon to fire a chosen weaken move."""
        opened = False
        self._home_to_fight()
        for _ in range(12):
            if self._movelist_open():
                opened = True; break
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            if self._movelist_open():
                opened = True; break
            if not (self._white_box() or self._movelist_open()):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
                self._home_to_fight()
        if not opened:
            return "stuck"
        if not self._goto_move(idx):
            return "stuck"
        state = st.read_battle(self.b)
        pp0 = state["ours"]["moves"][idx].get("pp", 0) if state else 0
        before = self._bstate()
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        for _ in range(900):
            if not st.in_battle(self.b):
                return "done"
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                if cur["ours"]["moves"][idx].get("pp", 0) < pp0:
                    return "done"
                if before and (cur["enemy"]["hp"], cur["ours"]["hp"]) != before:
                    return "done"
                if self._white_box():
                    return "done"
            self.b.run_frame(); self.render()
        return "stuck"

    def catch_pokemon(self, max_seconds=150, weaken=True):
        """Catch the WILD foe (the proven live flow, automated): optionally WEAKEN/STATUS it once to
        boost the catch rate + stop it attacking, then THROW Poké Balls until caught. COMMITS - it
        re-throws after a break instead of abandoning after one ball (the live Ekans flow). Returns
        'caught' | 'no_balls' | 'trainer' | 'fled' | 'fainted' | 'stuck'. Gen-3 trainer mons can't
        be caught (returns 'trainer')."""
        t0 = time.time()
        if self._is_trainer_battle():
            return "trainer"
        self._started = True
        self._catching = True              # F-7(c): KOing a catch target is a FAILURE — the
        #                                    certain-win beat stays silent for this whole flow
        self._skip_streak = set()
        self._reach_first_menu(t0, max_seconds)
        self._prev = st.read_battle(self.b)
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        softened = False
        status_only = False
        sleep_tries = 0
        # 2026-07-06 NURSERY FIX: a strong ace "wearing down" a much-weaker wild ONE-SHOTS it (run-12:
        # 3 judged keepers KO'd mid-weaken, labeled 'fled'). Early-route species catch fine at full HP —
        # when the foe is 10+ levels under the lead, never CHIP it (one hit would KO). But a pure SLEEP
        # move is damage-free with ZERO KO risk and x2 catch rate in Gen 3 — with a thin ball supply
        # that's the difference between "caught" and "the last ball broke free". Sleep-then-throw.
        try:
            _rb0 = st.read_battle(self.b)
            if weaken and _rb0 and (_rb0["ours"].get("level", 0) - _rb0["enemy"].get("level", 0)) >= 10:
                status_only = True
                self.log("   [engine] catch: foe is 10+ levels under the lead — no chipping (would KO); "
                         "SLEEP-then-throw if a sleep move is up")
        except Exception:
            pass

        def _ended():
            """Battle ended: settle, then a party+1 means we CAUGHT it (the 'Gotcha!' can end the
            battle a beat before the party count ticks - don't mislabel a real catch as 'fled')."""
            for _ in range(40):
                self.b.run_frame(); self.render()
            return "caught" if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0 else "fled"

        while time.time() - t0 < max_seconds:
            if not st.in_battle(self.b):
                return _ended()
            if self._ball_count() <= 0:
                self.emit("I'm out of Poké Balls - I'll come back for this one", beat=True)
                # 2026-07-06 WEDGE FIX: never walk away from a LIVE battle — an abandoned battle gets
                # re-detected by travel as a fresh encounter forever (south_run1: 'stuck' ×27 spin).
                # Resolve it first (flee the wild), THEN report no_balls.
                self.flee(max_seconds=45)
                return "no_balls"
            self._settle()
            if not st.in_battle(self.b):
                return _ended()
            if not self._white_box():
                # B-ONLY advance in the catch loop: a blue box here is catch-sequence text OR the
                # post-catch "give a nickname? [YES/NO]" prompt. _advance_text presses A first, which on
                # that prompt selects YES and opens the naming keyboard (spins forever). B safely advances
                # the text AND declines the nickname. (B is unsafe only at the action menu = RUN, which is
                # white_box and excluded here.)
                self._wait(18); self.b.press("B", 2, 12, self.render, owner=self.owner); continue
            state = st.read_battle(self.b)
            if state and state["enemy"]["hp"] <= 0:
                return "fainted"                         # we KO'd it - can't catch a fainted foe
            if status_only and not softened and state is not None:
                # BIG-LEVEL-GAP path: sleep is the ONLY safe soften (any hit could KO). Fire a sleep
                # move (75%-acc powders miss — retry once), re-checking asleep each loop; then throw.
                # No sleep move usable / both tries spent -> throw at full HP (the old behavior).
                if not state["enemy"].get("asleep") and sleep_tries < 2:
                    si = next((i for i, m in enumerate(state["ours"]["moves"])
                               if m.get("id", 0) in self._SLEEP_MOVES and m.get("pp", 0) > 0), None)
                    if si is not None:
                        sleep_tries += 1
                        self.emit("let me put it to sleep first — easier to catch that way", beat=True)
                        self._fire_move(si)
                        continue
                softened = True
                continue
            if weaken and not status_only and not softened and state is not None:
                # PHASE 4 GUARD: if she can't weaken AT ALL (no status + no damaging move with PP —
                # depleted) and the foe is still near full HP, DON'T throw — a full-HP catch is low-odds
                # and that's exactly how she burned her whole ball supply tonight. Flee (preserve the
                # balls) and surface that she needs to restore PP (a Center tops up PP). Roam then heals.
                if not self._can_weaken(state) and state["enemy"].get("maxhp") \
                        and state["enemy"]["hp"] > state["enemy"]["maxhp"] * self.CATCH_WEAKEN_CEIL:
                    self.emit("I can't even dent it — I'm out of PP to weaken it, and I'm not burning my "
                              "Poké Balls on a full-health throw. Backing out to restore my moves first.",
                              beat=True, tier=2)
                    self.log("   [engine] catch: CAN'T WEAKEN (depleted PP) + foe near full HP -> fleeing "
                             "to preserve balls (not ball-dumping)")
                    self.flee(max_seconds=45)
                    return "cant_weaken"
                wi = self._catch_weaken_move(state)
                if wi is not None:
                    self.emit("let me wear it down first", beat=True)
                    self._fire_move(wi)
                self._weaken_hp()                    # chip HP into the catchable band (faint-guarded)
                softened = True
                continue
            res = self.throw_ball(max_seconds=max(20, int(max_seconds - (time.time() - t0))))
            if res == "no_balls" and st.in_battle(self.b):
                self.flee(max_seconds=45)        # same wedge fix: resolve the live battle before reporting
            if res in ("caught", "no_balls", "trainer"):
                return res
            # 'broke_free' / 'stuck' -> the foe took its turn; loop and throw again (commit)
        return "stuck"

    # ── BATCH 2 PART B: USE A HEAL / CURE IN BATTLE (live-reconned recon_itemuse.py 2026-06-27) ────────
    # Flow proven on a live wild battle: settle to the ACTION menu (pixel-gated) -> FIGHT home -> RIGHT
    # (=BAG) -> A opens the bag -> steer GBAG_POCKET to the Items pocket (0) -> DOWN to the item's row
    # (the pocket list shows the bag array IN ORDER; nav by the BAG_CURSOR readback) -> A walks
    # select->USE->target(default lead)->apply. GROUND-TRUTH success = the item COUNT drops (HP rise is
    # incidental). FAIL-SAFE (Jonny's mandate): every step is bounded + readback-gated; on ANY failure we
    # B-out to a clean menu and return 'failed' so the battle loop just KEEPS FIGHTING — never a wedge,
    # never a wrong action (the apply A-loop only runs once we've CONFIRMED pocket==0 AND cursor==row).
    def _items_pocket(self):
        """[(item_id, qty), ...] in the Items pocket in DISPLAY order (qty XOR'd with the
        low-16 key). Scans ALL 42 slots and SKIPS empty/zero-qty slots instead of breaking:
        consuming the LAST of an item mid-battle leaves a zero HOLE in the RAM pocket, and
        the old break-at-first-zero made the ENTIRE pocket read empty from then on (run17
        forensics, frame+RAM: the Ether at display row 0 hit x0 at Agatha -> revive_item=None
        with 6 Revives physically in the bag -> every offer (potion/cure/revive) silently died
        for the rest of the process while camp.bag_count (scan-all) reported FR x10 — the
        run16 all-attempts collapse). The displayed list skips holes too, so hole-skipped
        order stays the TRUE-row order the bag cursor navigates."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        out = []
        for s in range(42):
            slot = sb1 + _ITEMS_POCKET_OFF + s * 4
            iid = self.b.rd16(slot)
            if not iid:
                continue
            qty = self.b.rd16(slot + 2) ^ key
            if qty > 0:
                out.append((iid, qty))
        return out

    def _items_count(self, item_id):
        return next((q for i, q in self._items_pocket() if i == item_id), 0)

    def _action_cursor_alive(self, probes=2):
        """The run19 Lance livelock (2026-07-07): a dangling item message box on the in-battle
        PARTY screen ("PERSIAN's HP was restored...") lights the SAME white-panel pixels as the
        action menu, so every pixel-gated path believed a menu was up while the game waited for
        a box-dismissing press — the directional walks (_goto_fight/_goto_pokemon) tapped into a
        STALE GBATTLE_ACTION_CURSOR forever and the abort->re-enter cycle never pressed A/B.
        Pixel truth is not enough: the action menu is only REAL if the cursor RESPONDS. Tap
        toward a horizontal neighbour and demand the readback moves (retry — single taps get
        eaten on this core). Leaves the cursor one step over; all callers walk by readback."""
        c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
        if c not in (ram.ACT_FIGHT, ram.ACT_BAG, ram.ACT_POKEMON, ram.ACT_RUN):
            return False
        t = "RIGHT" if c in (ram.ACT_FIGHT, ram.ACT_POKEMON) else "LEFT"
        for _ in range(probes):
            self._tap(t); self._wait(5)
            if self.b.rd8(ram.GBATTLE_ACTION_CURSOR) != c:
                return True
        return False

    def _settle_action_menu(self, tries=30):
        """Reach the real ACTION menu using the PIXEL signals (menu_up RAM is stale on this core): action
        menu = white panel AND NOT the move-list pixel, VERIFIED by cursor responsiveness — a dangling
        message box / party screen / USE-CANCEL sub-box lights the same pixels (the run19 Lance livelock).
        Impostors are drained with B (dismisses boxes, backs out of party/bag, no-op at the real menu).
        Back out of the move list with B; advance text."""
        self._settle()
        for _ in range(tries):
            if not st.in_battle(self.b):
                return False
            if self._in_move_list():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            elif not self._white_box():
                self._advance_text()
            elif self._action_cursor_alive():
                return True
            else:
                self.log("   [engine] action-menu impostor (white box, DEAD cursor) -> B-drain "
                         "(dangling box / party screen)")
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
        return False

    def _goto_bag(self, tries=10):
        """Walk the action cursor to BAG (top-right, ACT_BAG=1) by READBACK, not a blind RIGHT — the
        live look-ahead proved a blind _tap('RIGHT') gets EATEN on this long core, so A landed on FIGHT
        and the bag never opened ('eaten RIGHT'), and she could never heal through a fight. Mirror of
        _goto_pokemon: grid FIGHT(0,TL) BAG(1,TR) / POKEMON(2,BL) RUN(3,BR). Confirms ACT_BAG before A."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_BAG:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("RIGHT")
            elif c == ram.ACT_RUN:
                self._tap("UP")
            elif c == ram.ACT_POKEMON:
                self._tap("UP")                           # -> FIGHT, then RIGHT next iter -> BAG
            else:
                return False                              # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_BAG

    def _goto_fight(self, tries=10):
        """Walk the action cursor to FIGHT (top-left, ACT_FIGHT=0) by readback. Mirror of _goto_bag;
        grid is FIGHT(0,TL) BAG(1,TR) / POKEMON(2,BL) RUN(3,BR)."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_FIGHT:
                return True
            if c == ram.ACT_BAG:
                self._tap("LEFT")
            elif c == ram.ACT_POKEMON:
                self._tap("UP")
            elif c == ram.ACT_RUN:
                self._tap("UP")                           # -> BAG, then LEFT next iter -> FIGHT
            else:
                return False                              # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_FIGHT

    def _struggle(self):
        """ZERO PP anywhere in a can't-flee battle: A on FIGHT — the game substitutes Struggle
        ("X has no moves left!"), which passes the turn and lets the battle actually resolve.
        Returns 'done' if the confirm fired, else 'no_usable_move' (rides the anti-wedge floor)."""
        if not self._settle_action_menu() or not self._goto_fight():
            return "no_usable_move"
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
        self._wait(30)
        for _ in range(6):                                # drain the "no moves left!" text
            if not st.in_battle(self.b) or self._white_box():
                break
            self._advance_text()
        return "done"

    def _open_bag(self, tries=4):
        """From the ACTION menu: cursor->BAG (verified by readback) -> A. The bag is open iff the white
        action panel is GONE (a blue description box). If A didn't open it (white panel stays), B out +
        retry. The readback nav (vs a blind RIGHT) is the fix for the long-core 'eaten RIGHT' that left
        her unable to use a Potion mid-fight — never fires a move (that needs a 2nd A in the move list)."""
        for _ in range(tries):
            if not self._goto_bag():
                self._settle_action_menu(); continue
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(50)
            if not self._white_box():
                return True
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            self._settle_action_menu()
        return False

    def _exit_bag(self):
        """Best-effort B back to a clean menu/battle so a FAILED item-use never leaves a menu dangling
        for the battle loop to A into (which could fire a stray move). Bounded. The white_box exit is
        gated on the bag being GONE — the bag's USE/CANCEL box lights the same pixels (layer 8)."""
        for _ in range(10):
            if not st.in_battle(self.b) or (self._white_box() and not self._bag_screen()):
                return
            self.b.press("B", 2, 12, self.render, owner=self.owner); self._wait(10)

    def use_item_in_battle(self, item_id, max_seconds=30, target=None):
        """Use one `item_id` from the Items pocket. Returns 'used' (count dropped) | 'no_item' |
        'failed'. FAIL-SAFE: anything but 'used' leaves the battle fightable. `target` aims the
        item's party screen: 'active' (the mon that is OUT — always menu row 0, the lead panel)
        or 'fainted' (the strongest downed mon — Revive). The row is resolved by CONTENT at MENU
        TIME (_menu_rows order law): run14 frame-proof — a Revive aimed at a PRE-menu slot index
        confirmed the healthy active mon's panel, ate 'It won't have any effect.' boxes all night,
        and never consumed. None keeps the legacy un-aimed walk; aim taps are party-screen-gated
        (pixel truth) so a lagging party open never taps into the bag's USE/CANCEL sub-box."""
        ids = [i for i, _ in self._items_pocket()]
        if item_id not in ids:
            self.log(f"   [engine] use_item: item {item_id} NOT in pocket {ids[:8]} — no_item (LOUD)")
            return "no_item"
        row = ids.index(item_id)
        cnt0 = self._items_count(item_id)
        if not self._settle_action_menu():
            self.log("   [engine] use_item: couldn't reach the action menu — keep fighting (LOUD)")
            return "failed"
        if not self._open_bag():
            self.log("   [engine] use_item: bag wouldn't open (eaten RIGHT?) — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        for _ in range(8):                                   # steer GBAG_POCKET to the Items pocket (0)
            if self.b.rd8(ram.GBAG_POCKET) == 0:
                break
            self._tap("LEFT"); self._wait(12)
        if self.b.rd8(ram.GBAG_POCKET) != 0:
            self.log("   [engine] use_item: couldn't reach the Items pocket — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        def _sel():
            # TRUE selection = cursor + scrollOffset (the mart-list law, recon_bagscroll-verified).
            # The raw cursor byte alone LIES after any scrolled visit — the list remembers both.
            return self.b.rd8(BAG_CURSOR) + self.b.rd16(BAG_SCROLL)
        for _ in range(14):                                  # nav to the item's TRUE row (cursor+scroll)
            if _sel() == row:
                break
            self._tap("DOWN" if _sel() < row else "UP"); self._wait(10)
        self._wait(8)                                        # settle scroll animation, then re-verify
        if _sel() != row:
            self.log(f"   [engine] use_item: couldn't reach true row {row} "
                     f"(cursor={self.b.rd8(BAG_CURSOR)} scroll={self.b.rd16(BAG_SCROLL)}) — "
                     f"keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        # CONFIRMED in the Items pocket on the right row -> A walks select->USE->target->apply.
        # A#0 selects the item, A#1 hits USE (party screen opens); then AIM ONCE — focus the
        # list, resolve the row by MENU-TIME CONTENT ('active' = row 0, the lead panel IS the
        # mon that's out by the order law; 'fainted' = the strongest hp==0 row) — and from
        # there on CONFIRM BLIND. Aiming on every iteration was the run15 Ether livelock:
        # the Ether opens a move-select sub-box AFTER the mon confirm, _party_focus read it
        # as a stray sub-menu and B-cancelled it every lap, so the item never consumed.
        # Count drop is the only truth; a mis-aim just exhausts the walk -> 'failed' ->
        # keep fighting (fail-safe, never a wedge).
        aimed = target is None                             # no aim requested = nothing to do
        for n in range(10):
            if self._items_count(item_id) < cnt0:
                break
            if not aimed and self._party_screen():
                self._wait(8)                              # let the screen finish drawing
                if not self._party_focus():
                    self.log("   [engine] use_item: party list never took focus (fail-safe)")
                rows = self._menu_rows()
                if isinstance(target, int):                # an EXACT party slot (revive routing)
                    _row = target
                elif target == "fainted":
                    _row = next((r["row"] for r in sorted(rows, key=lambda r: -r["level"])
                                 if r["hp"] == 0), 0)
                else:                                      # 'active' -> the lead panel
                    _row = 0
                if not self._party_goto_slot(_row):
                    self.log(f"   [engine] use_item: aim couldn't reach menu row {_row} "
                             f"— confirming where the cursor is (fail-safe)")
                aimed = True
            self.log(f"   [engine] use_item walk n={n}: party={self._party_screen()} "
                     f"bag={self._bag_screen()} white={self._white_box()} "
                     f"pcur={self._party_cursor_slot()} lead={self._party_cursor_on_lead()}")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)
            if not st.in_battle(self.b):
                break
        if self._items_count(item_id) < cnt0:
            self.log(f"   [engine] use_item: USED item {item_id} (count {cnt0}->{self._items_count(item_id)})")
            self.emit("used an item — that's better", beat=True)
            # LAYER 8 FIX: close the BAG first — the old drain exited on _white_box(), but the bag's
            # USE/CANCEL sub-box LIGHTS those pixels, so 'used' could return with the bag still open
            # and the next turn's presses landed in it forever (caterpie 7/40, walk 3).
            self._close_bag_screen()
            # Drain the "X recovered!" text back to a LIVE menu/battle. The old 6-lap exit broke
            # on _white_box alone — the PARTY screen's result box ("PERSIAN's HP was restored...")
            # lights those pixels, so 'used' returned with the box still up and the turn loop
            # wedged against it forever (the run19 Lance livelock). _settle_action_menu now
            # demands cursor responsiveness and B-drains impostors, so route through it.
            self._settle_action_menu(tries=12)
            return "used"
        self.log(f"   [engine] use_item: pocket={self.b.rd8(ram.GBAG_POCKET)} cursor={self.b.rd8(BAG_CURSOR)} "
                 f"scroll={self.b.rd16(BAG_SCROLL)} row={row} — selected but item {item_id} NOT consumed "
                 f"(count still {cnt0}) — keep fighting (LOUD)")
        self._debug_snap(f"itemfail_{item_id}")
        self._exit_bag(); return "failed"

    def _maybe_use_item(self, state):
        """OFFER the in-battle item instinct to the oracle when it's a REAL option: the active mon is
        crit-low AND a heal item is in the bag, or it's afflicted AND a matching cure is in the bag. She
        DECIDES (capability-not-script). Returns True iff an item was actually used (the turn is spent),
        so run() skips move selection this turn. Any non-'used' outcome -> fall through to a normal move
        (fail-safe — she never wedges, and never faints with unused heals because the option was surfaced)."""
        if not self.choose:
            return False
        ours = state["ours"]
        frac = _hp_frac(ours)
        offers, plan = {}, {}
        # MATCHUP-AWARE HEAL THRESHOLD (general, E4-critical): a foe that hits us SUPER-EFFECTIVELY can
        # 2HKO from high HP, so the 30% crit floor heals too LATE (one more hit faints us). Against such a
        # threat, heal EARLY — at HALF — so a Potion can OUT-HEAL its chip while we win the fight another
        # way (poison/chip). This is exactly what cracks Gary's Charmander (Ember 2x on Ivysaur): heal
        # through the Embers while poison kills it. An even/resisted foe still heals at the crit floor.
        _myt = [t for t in (ours.get("types") or []) if t and t != "???"]
        _foet = [t for t in (state.get("enemy", {}).get("types") or []) if t and t != "???"]
        threat = self._matchup_def(_myt, _foet) if (_myt and _foet) else 1.0
        # LEVEL-AWARE THREAT (2026-07-11 NS#4 — the Rock Tunnel white-box wedge): the early-heal-at-HALF
        # exists to survive a SE 2HKO from a REAL threat. A foe we vastly OUT-LEVEL can't 2HKO us even off
        # a super-effective hit, so healing it early is wasted — and every mid-battle item use risks the
        # action-menu white-box impostor wedge (she over-potioned a weak Rock-Tunnel foe that reads Fire-
        # type, 2x on her Venusaur, and livelocked the whole tunnel crossing). Only treat SE as a heal-
        # early threat when the foe is within striking level; a much-weaker foe heals at the normal crit
        # floor (so a truly low-HP save still fires, but she powers through weak SE chip like a real player).
        _ours_lv = ours.get("level") or 0
        _foe_lv = (state.get("enemy", {}) or {}).get("level") or 0
        _much_weaker = bool(_ours_lv and _foe_lv and (_ours_lv - _foe_lv) >= 10)
        heal_frac = 0.5 if (threat >= 2 and not _much_weaker) else BATTLE_CRIT_FRAC
        # AIM every heal/cure at the mon actually OUT — by the _menu_rows order law the mon
        # that's out is ALWAYS menu row 0 (the lead panel) while the party screen is open,
        # so the aim is a KIND resolved at menu time, never a slot index carried across the
        # menu-open boundary (the run14 Revive-on-the-wrong-row class).
        aim = "active"
        # FINISH-THE-FOE GUARD (night shift #3 — the Silph PP-famine/potion-loop wall): don't spend the
        # turn healing when the active foe is one hit from fainting AND we're not in genuine faint danger.
        # The Gary gauntlet exposed the failure: at 35% HP vs a 4/98 Exeggcute the 50% matchup-threshold
        # kept picking use_potion instead of the finishing hit, so she never KO'd the foe, drained her
        # damaging PP to famine, then switch-fed a fodder mon and LOST a fight she was out-chipping. A real
        # player finishes a near-dead foe. Suppress the heal offer when foe <=25% HP and we're above the
        # hard crit floor (a life-saving heal at truly-low HP is still offered). General battle-brain fix
        # (helps every long fight incl. the E4); shared-plumbing, additive, no identity/mode-state touch.
        foe = state.get("enemy") or {}
        foe_mx = foe.get("maxhp") or 0
        foe_frac = (foe.get("hp", 0) / foe_mx) if foe_mx else 1.0
        finishable = foe_frac <= 0.25 and frac > BATTLE_CRIT_FRAC
        if frac <= heal_frac and not finishable:
            heal = next((i for i in _HEAL_ITEMS_PREF if self._items_count(i) > 0), None)
            if heal is not None:
                plan["use_potion"] = (heal, aim)
                offers["use_potion"] = (f"use a healing item — you're at {ours['hp']}/{ours['maxhp']} HP, "
                                        f"about to faint, and you HAVE one in the bag")
        elif finishable and frac <= heal_frac:
            self.log(f"   [engine] FINISH-THE-FOE: foe at {int(foe_frac*100)}% (<=25%), us {int(frac*100)}% "
                     f"(> crit) -> no heal, land the KO instead")
        # Status of the mon actually OUT (gBattleMons ground truth via read_battle) — the old
        # _lead_status read gPlayerParty[0], so post-switch a sleep-locked FODDER never got a
        # cure offer while the dead ace's 'none' was consulted instead (run16 attempts 2+).
        status = _decode_status((state.get("ours") or {}).get("status1", 0) or 0)
        if status:
            cure = self._STATUS_CURE_for(status)
            if cure is not None:
                plan["use_cure"] = (cure, aim)
                offers["use_cure"] = f"use the cure for {status} — it's hurting you and you have the item"
        # REVIVE INSTINCT (night shift #13): the fallen-ace case that killed e4_run3 at Lance —
        # Revives rode the bag unused while bench-warmers tanked on. Offer resurrection only when
        # the fainted mon out-levels everything still standing (fodder fainting never triggers it).
        revive = next((i for i in _REVIVE_ITEMS_PREF if self._items_count(i) > 0), None)
        if revive is not None:
            down = self._revive_worthy_slot()
            if down is not None:
                # route via "fainted" (the STRONGEST downed mon — the proven no-wedge path). The
                # int-slot routing wedged the revive item-application at Gary (run10: "item 24 NOT
                # consumed"). At Gary the strongest fainted IS the type-answer (Lapras L56 > Kadabra
                # L50), so "fainted" revives the right mon. _revive_worthy_slot still gates the OFFER.
                plan["use_revive"] = (revive, "fainted")
                offers["use_revive"] = ("spend this turn reviving your fallen heavy-hitter — "
                                        "it's stronger than anyone still standing and you HAVE a Revive")
        if "use_revive" not in offers:
            # FORENSICS (run16 mystery): attempt 1's Arbok endgame cycled 5 fodder mons past a
            # dead L66 ace with 6 Revives in the bag and never a single offer. Whenever a
            # fainted mon exists and no revive is offered, log the exact inputs — this line is
            # the diagnosis when it happens again.
            try:
                rows = [(st.read_party_species(self.b, i),
                         self.b.rd16(ram.GPLAYER_PARTY + i * 100 + 0x56),
                         self.b.rd8(ram.GPLAYER_PARTY + i * 100 + 0x54)) for i in range(6)]
                if any(sp and hp == 0 for sp, hp, _lv in rows):
                    self.log(f"   [engine] revive-check: NO offer (revive_item={revive}, "
                             f"worthy={self._revive_worthy_slot()}, party sp/hp/lv={rows})")
            except Exception as e:
                self.log(f"   [engine] revive-check: read error {e} (LOUD)")
        # PP-RESTORE INSTINCT (night shift #13): foe-aware famine + an Ether/Elixir in the bag ->
        # offer restoring PP BEFORE the famine switch pulls the ace out. Rides the same aimed
        # walk: A selects the aimed mon, the move-restore box defaults to move 0 (the workhorse
        # slot), count-drop verifies — a mis-walk is 'failed' -> keep fighting, never a wedge.
        if self._active_pp_famine(state):
            # The flow restores MOVE SLOT 0 (the workhorse — the move box defaults there), so
            # only offer when that actually buys a usable attack: slot 0 damaging + CONNECTS +
            # genuinely EMPTY. An IMMUNITY famine can't be cured by PP (ether_verify: the game
            # ate 8 walks with 'won't have any effect' on full-PP fodder vs Gengar).
            mv = (state.get("ours") or {}).get("moves") or []
            m0 = mv[0] if mv else None
            if m0 and m0.get("pp", 0) == 0 and m0.get("power", 0) > 0 \
                    and self._move_connects(m0, state):
                ether = next((i for i in _ETHER_ITEMS_PREF if self._items_count(i) > 0), None)
                if ether is not None:
                    plan["use_ether"] = (ether, aim)
                    offers["use_ether"] = ("restore PP with your Ether — you're out of moves that can "
                                           "hit this foe and it puts your best move back in the fight")
        if not offers:
            return False
        offers["keep_fighting"] = "keep attacking — push through it"
        ctx = {"hp": f"{ours['hp']}/{ours['maxhp']}", "status": status or "none",
               "foe": st.SPECIES_NAME.get(state["enemy"]["species"], "the foe")}
        self.log(f"   [engine] ITEM-INSTINCT offer: {list(offers)} ctx={ctx}")
        pick = self.choose("battle_item", offers, ctx)
        if pick and pick in plan:
            item, kind = plan[pick]
            self.log(f"   [engine] ITEM-INSTINCT pick -> {pick} (item {item}, aim={kind})")
            return self.use_item_in_battle(item, target=kind) == "used"
        self.log(f"   [engine] ITEM-INSTINCT pick -> {pick!r} (keep fighting)")
        return False

    def _revive_worthy_slot(self):
        """Party slot of the strongest FAINTED mon, iff it out-levels every mon still standing.
        That's the revive-worth-a-turn test: the ace is down and the field is held by fodder.
        Returns None otherwise (never revives Ekans-class bench weight mid-fight) — EXCEPT
        the LAST-BODY INSURANCE (shift-15, run18 Gary postmortem): the worthy gate held the
        whole Lance fight (worthy=None past 3-5 corpses because the L70 ace stood), so she
        entered the Champion room with ONE body and an ace faint = instant whiteout. When
        the active mon is the LAST body standing, it's genuinely hurt (<=50%), and >=2
        revives remain, a revived fodder IS worth the turn regardless of level: it converts
        'ace faints = loss' into the proven comeback cycle (fodder tanks the KO turn, the
        ace gets revived behind it — the revive_verify Agatha win). >=2 keeps the last
        revive reserved for the ace itself. The old 50% HP floor is GONE (shift-17,
        run19/20 postmortem): a healthy last-body ace walked the whole Lance room with no
        spare body banked, so one crit/sleep = instant whiteout — at alive==1 a bench body
        is ALWAYS worth the turn, and the gate self-closes at alive==2 so it can't drain
        the kit.

        TYPE-ANSWER REVIVE (e4 run5 Gary/Charizard postmortem): the level gate below never
        revives a fainted specialist while the higher-level ace stands — but vs a foe the ace
        CAN'T hit (Venusaur 0.25x into Gary's Charizard), the dead L50 Lapras's Ice Beam (2x)
        is the ONLY answer. If NO alive mon is super-effective on the CURRENT foe but a FAINTED
        reserve's STAB is (>=2x), revive that type-answer so it can come in and swing."""
        try:
            foe_types = st.species_types(st.read_enemy_species(self.b, 0))
            if foe_types:
                alive_se, dead_answer = False, None
                for i in range(6):
                    sp = st.read_party_species(self.b, i)
                    if not sp:
                        continue
                    off = self._matchup_off(st.species_types(sp), foe_types)
                    if self.b.rd16(ram.GPLAYER_PARTY + i * 100 + 0x56) > 0:
                        if off >= 2:
                            alive_se = True
                    elif off >= 2 and dead_answer is None:
                        dead_answer = i
                if dead_answer is not None and not alive_se:
                    self.log(f"   [engine] revive-check: TYPE-ANSWER revive slot {dead_answer} "
                             f"(no standing mon is SE on this foe; the fainted one is)")
                    return dead_answer
        except Exception:
            pass
        try:
            alive, best = [], None
            for i in range(6):
                if not st.read_party_species(self.b, i):
                    continue
                hp = self.b.rd16(ram.GPLAYER_PARTY + i * 100 + 0x56)
                mx = self.b.rd16(ram.GPLAYER_PARTY + i * 100 + 0x58)
                lv = self.b.rd8(ram.GPLAYER_PARTY + i * 100 + 0x54)
                if hp > 0:
                    alive.append((hp, mx, lv))
                elif best is None or lv > best[1]:
                    best = (i, lv)
        except Exception:
            return None
        if best is None:
            return None
        if not alive or best[1] > max(lv for _hp, _mx, lv in alive):
            return best[0]
        if len(alive) == 1:
            hp, mx, _lv = alive[0]
            n_rev = sum(self._items_count(i) for i in _REVIVE_ITEMS_PREF)
            if n_rev >= 2:
                self.log("   [engine] revive-check: LAST-BODY INSURANCE armed "
                         f"(alive=1 at {hp}/{mx}, revives x{n_rev})")
                return best[0]
        return None

    def _STATUS_CURE_for(self, status):
        """The cure item id for a status that's actually in the bag (specific cure, else Full Heal)."""
        spec = _STATUS_CURE_ITEM.get(status)
        if spec is not None and self._items_count(spec) > 0:
            return spec
        if self._items_count(_FULL_HEAL) > 0:
            return _FULL_HEAL
        return None

    # ── SCREEN-based menu detection (the RAM has NO clean menu-state flag - every candidate
    # is a frame counter or a one-state false positive; diagnosed 2026-06-23). The UI is
    # battle-independent: the action menu + move list draw a WHITE panel bottom-right; a
    # text/dialogue box is BLUE there. Pixel (160,150) is white at the action menu but DARK
    # in the move list - so the three states are cleanly separable from the screen. ──
    _WHITE_PTS = ((135, 138), (200, 138), (135, 150), (190, 150), (150, 150), (175, 150))

    def _debug_snap(self, tag):
        """Save the current frame when a wedge trips (BATTLE_DEBUG_DIR env, set by recon
        vehicles). The victory_run7 lesson: a silent wedge with no frame costs a shift of
        log archaeology; a frame costs one glance. No-op when the env is unset (play_live)."""
        d = os.environ.get("BATTLE_DEBUG_DIR")
        if not d:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(
                os.path.join(d, f"bwedge_{tag}_{int(time.time())}.png"))
            self.log(f"   [engine] wedge frame -> bwedge_{tag}.png")
        except Exception as e:
            self.log(f"   [engine] wedge snap failed: {e}")

    def _white_box(self):
        """True iff the bottom-right white menu panel is up (action menu OR move list) - i.e.
        NOT a blue text/dialogue box. The reliable 'a menu is waiting for me' signal."""
        p = self.b.frame_rgb().load()
        return sum(1 for x, y in self._WHITE_PTS if min(p[x, y]) > 200) >= 4

    def _in_move_list(self):
        """True iff the FIGHT move list is open (white panel up AND the action-menu marker
        pixel (160,150) is dark - it is white at the action menu, dark over the move names)."""
        p = self.b.frame_rgb().load()
        if sum(1 for x, y in self._WHITE_PTS if min(p[x, y]) > 200) < 4:
            return False
        return min(p[160, 150]) < 100

    # The PARTY SCREEN ("Choose a POKéMON") replaces the whole battle UI with a teal striped
    # background. Sample points sit in the LEFT column BELOW the active-mon box — background
    # at ANY party size (the 5 reserve slots stack in the RIGHT column). Ground truth
    # (fight_10.png, the layer-7 gauntlet diagnosis): stripes are (71,168,161)/(60,145,144) —
    # G≈B both >120, R<100; battle/overworld/gym/cave frames score 0/4 (Route-3 grass
    # (115,206,165) fails R<100; Brock's floor (24,165,107) fails B>120). 3-of-4 = screen up.
    _PARTY_PTS = ((30, 110), (60, 115), (20, 90), (70, 108))

    def _party_screen(self):
        p = self.b.frame_rgb().load()
        hits = 0
        for x, y in self._PARTY_PTS:
            r, g, bl = p[x, y][:3]
            if r < 100 and g > 120 and bl > 120 and abs(g - bl) < 40:
                hits += 1
        return hits >= 3

    # The BAG SCREEN (layer 8, the caterpie-7/40 wedge, frame stage_l8.png): an in-battle item flow
    # can leave/return the battle to the open bag, and EVERY state byte lies there (MENU_MODE reads a
    # stale 2, GBATTLE_MENU_UP a stale 1, and the USE/CANCEL sub-box lights the white-panel pixels) —
    # so the turn loop "picked moves" into USE/CANCEL forever. Pixel truth: the item LIST PANEL is a
    # pale yellow (r,g>240, 180<b<230) no battle screen has — 3/3 on the wedge frame, 0/3 on
    # battle/party/overworld/gym/cave/Center fixtures. Panel points sit clear of the header plate
    # (whose hue varies per pocket) so this reads True for ANY pocket.
    _BAG_PTS = ((160, 30), (200, 60), (120, 10))

    def _bag_screen(self):
        p = self.b.frame_rgb().load()
        hits = 0
        for x, y in self._BAG_PTS:
            r, g, bl = p[x, y][:3]
            if r > 240 and g > 240 and 180 < bl < 230:
                hits += 1
        return hits >= 3

    def _close_bag_screen(self, tries=10):
        """Deliberate B-cascade out of an open bag back to the battle (USE/CANCEL box -> item list ->
        pocket -> closed). Bounded; returns True when the bag is gone."""
        for _ in range(tries):
            if not self._bag_screen() or not st.in_battle(self.b):
                return True
            self.b.press("B", 2, 12, self.render, owner=self.owner)
            self._wait(14)
        return not self._bag_screen()

    def _home_to_fight(self):
        """Park the action cursor on FIGHT (top-left) WITHOUT reading the stale cursor latch:
        UP then LEFT are absorbed at the top/left boundary, so from ANY of the 4 cells they net
        the top-left corner = FIGHT. Reliable now that input is clean (no phantom-A confirm)."""
        self._tap("UP"); self._tap("LEFT"); self._wait(4)

    def _nav_move(self, idx):
        """Move the move-list cursor from slot 0 (where the list opens) to slot idx in the 2x2
        grid: TL=0 TR=1 / BL=2 BR=3 (RIGHT = column, DOWN = row). Settles after so the confirm-A
        isn't eaten mid cursor-move (the slot-2 lesson)."""
        if idx == 1:
            self._tap("RIGHT")
        elif idx == 2:
            self._tap("DOWN")
        elif idx == 3:
            self._tap("RIGHT"); self._tap("DOWN")
        self._wait(14)

    def _movelist_open(self):
        """RAM truth for 'the FIGHT move list is open' (MENU_MODE == 2), OR the pixel check — either
        suffices. The RAM signal is what survives the long-running core (the pixel detect intermittently
        fails there, the keystone wedge)."""
        try:
            if self.b.rd8(MENU_MODE) == 2:
                return True
        except Exception:
            pass
        return self._in_move_list()

    def _movelist_open_verified(self):
        """_movelist_open + a CURSOR-RESPONSE cross-check. THE immortal-Ekans wedge (2026-07-05 look-ahead):
        after an in-battle ITEM use, MENU_MODE reads a STALE 2, so the open-check short-circuited True
        BEFORE the FIGHT-opening A was ever pressed — the move list was never open, _goto_move's presses
        landed on the action menu, MOVE_CURSOR never moved, the flee also failed against the phantom
        state, and travel re-entered the SAME battle ~50x (ekans 27/27 every time). Doctrine (the
        cursor-desync lesson): trust CURSOR-RESPONSE, not a state byte. The list counts as open only if
        MOVE_CURSOR actually responds to a probe press (probe toward a neighbor, readback, restore).
        Known edge: a 1-move mon's cursor can't move (probe reads as closed) → the caller presses A,
        which on a truly-open 1-move list just fires slot 0 — the only move, harmless.
        2026-07-06 (run-14, the post-item-use A/B livelock): the byte can be stale LOW just as it was
        stale HIGH — gating the probe on _movelist_open() made a GENUINELY-open list read as closed,
        so the caller's wrong-submenu B closed it, the next A reopened it, ×12 → 'stuck' forever at
        the Route-6 gauntlet. The RESPONSE PROBE alone is the ground truth — no byte gate either way.
        (A stray probe on the ACTION menu just nudges its cursor; the caller re-homes with UP+LEFT.)"""
        cur = self.b.rd8(MOVE_CURSOR)
        probe = "RIGHT" if cur % 2 == 0 else "LEFT"
        back = "LEFT" if probe == "RIGHT" else "RIGHT"
        self._tap(probe); self._wait(8)
        if self.b.rd8(MOVE_CURSOR) == cur:            # cursor didn't respond -> not open (or 1-move edge)
            return False
        self._tap(back); self._wait(8)                # restore the cursor (readback nav re-verifies anyway)
        return True

    def _goto_move(self, idx, tries=12):
        """Walk the move-list cursor to slot idx by RAM READBACK of MOVE_CURSOR (0..3 in the 2x2 grid:
        index = row*2 + col), VERIFYING each press actually moved the cursor — an eaten d-pad press on
        the long core is simply retried (it can never silently land on the wrong move). Mirrors
        _goto_bag/_mart_goto_row. Returns True on arrival."""
        for _ in range(tries):
            cur = self.b.rd8(MOVE_CURSOR)
            if cur == idx:
                return True
            cr, cc = cur // 2, cur % 2
            tr, tc = idx // 2, idx % 2
            if cr != tr:
                self._tap("DOWN" if tr > cr else "UP")
            else:
                self._tap("RIGHT" if tc > cc else "LEFT")
            self._wait(8)
        return self.b.rd8(MOVE_CURSOR) == idx

    def _goto_party_slot(self, slot, tries=10):
        """Walk the in-battle party-list cursor to `slot` by RAM READBACK of PARTY_CURSOR (gPartyMenu.slotId)
        — DOWN increments, UP decrements; verify each press moved it (an eaten press is retried), so the
        switch never blind-lands on the wrong mon on the long core. Returns True on arrival."""
        for _ in range(tries):
            cur = self.b.rd8(PARTY_CURSOR)
            if cur == slot:
                return True
            self._tap("DOWN" if (cur < slot or cur > 5) else "UP")
            self._wait(8)
        return self.b.rd8(PARTY_CURSOR) == slot

    def _select_and_verify(self, state):
        """Called when the white action-menu panel is up (screen-gated). REAL move-list nav
        (slot-0 swap retired 2026-06-25 after the phantom-A fix): home the cursor to FIGHT, open
        the move list, NAVIGATE to the policy-chosen move, confirm it, and VERIFY by the chosen
        move's PP dropping (the move actually executed - robust for status moves too, which the
        old HP-change check missed). She now fires the move she CHOSE (e.g. a super-effective
        Vine Whip), not a pre-swapped slot 0. We never press B at the action menu (that flees a
        wild battle); B is only used to back out of a wrongly-opened submenu."""
        ours, enemy = state["ours"], state["enemy"]
        idx, desc, low = pol.choose_move(ours["moves"], enemy["types"], _hp_frac(ours))

        def _usable(i):                                # a real move with PP
            m = ours["moves"][i]
            return m.get("id", 0) != 0 and m.get("pp", 0) > 0
        if not (0 <= idx < 4) or not _usable(idx) or idx in self._skip_streak:
            # FIX 1 — REPETITION-AVERSE move pick: exclude EVERY move that already failed to fire this
            # streak (not just the last one), so she pivots through her whole moveset and NEVER re-spams
            # a dead/0-PP/blocked move (the Mankey case: she had 3 unused moves). Pick the best one she
            # HASN'T tried yet by expected damage. The streak clears the instant any move fires (below),
            # so a working move is never permanently benched (the PoisonPowder-spam lesson).
            cands = [i for i in range(4) if _usable(i) and i not in self._skip_streak]
            if cands:
                idx = max(cands, key=lambda i: max(ours["moves"][i].get("power", 0), 1)
                          * _eff(ours["moves"][i], enemy))
                desc = ours["moves"][idx].get("name", desc)
            else:
                # Every usable move has already failed to fire this streak (or none are usable at all —
                # the 0-PP Mankey wedge). A WILD battle surfaces to the anti-wedge floor and FLEES. A
                # TRAINER battle cannot flee — and idling submits NO action, so the turn-based game waits
                # forever (e4_run2 Agatha: PP famine -> no_usable_move -> abort -> re-enter, an infinite
                # livelock in which the foe never even got a turn, so she couldn't even LOSE her way to
                # the whiteout ratchet that refills PP). WAR-MUST-ADVANCE: re-fire the best PP-having
                # move anyway — even a failing move passes the turn, the foe acts, and the battle reaches
                # a real resolution (win, faint->forced switch, or whiteout->center ratchet).
                usable_all = [i for i in range(4) if _usable(i)]
                if self._is_trainer_battle() and usable_all:
                    self._skip_streak.clear()
                    idx = max(usable_all, key=lambda i: (
                        # prefer moves that can CONNECT (status counts); immune-damaging is last resort
                        1 if (ours["moves"][i].get("power", 0) == 0
                              or _eff(ours["moves"][i], enemy) > 0) else 0,
                        max(ours["moves"][i].get("power", 0), 1)
                        * _eff(ours["moves"][i], enemy)))
                    desc = ours["moves"][idx].get("name", desc)
                    self.log(f"   [engine] MOVES EXHAUSTED in a TRAINER battle -> war-must-advance: "
                             f"re-firing {desc} (idling never resolves a can't-flee fight)")
                elif self._is_trainer_battle():
                    # ZERO PP anywhere: A on FIGHT makes the game substitute STRUGGLE — the built-in
                    # resolver for exactly this state. Never idle a can't-flee battle.
                    self.log("   [engine] ZERO PP anywhere in a TRAINER battle -> FIGHT+A "
                             "(the game substitutes Struggle)")
                    return self._struggle()
                else:
                    self.log("   [engine] !! MOVES EXHAUSTED — every usable move tried with no effect "
                             "this streak (or none usable); not re-spamming a dead move")
                    return "no_usable_move"
        eff = _eff(ours["moves"][idx], enemy) if 0 <= idx < len(ours["moves"]) else 1.0
        # B-1 — INEFFECTIVE-MOVE AVERSION: never swing a DAMAGING move that does NOTHING (type-immune,
        # eff==0 — e.g. a Normal move into a Ghost). That's the "keeps using a move that does nothing"
        # failure. Re-pick a move that can actually connect (resisted/0.5 moves are still useful and the
        # policy already deprioritizes them; status moves at power 0 are never excluded). If she has NO
        # move that can connect, signal 'no_effective_move' — the turn loop offers a SWITCH (better
        # matchup) / else flees. Capability-not-script: she still chooses among the moves that work.
        def _useful(i):
            m = ours["moves"][i]
            if m.get("id", 0) == 0 or m.get("pp", 0) <= 0 or i in self._skip_streak:
                return False
            return not (m.get("power", 0) > 0 and _eff(m, enemy) == 0)
        if 0 <= idx < 4 and ours["moves"][idx].get("power", 0) > 0 and eff == 0:
            _uc = [i for i in range(4) if _useful(i)]
            if _uc:
                idx = max(_uc, key=lambda i: max(ours["moves"][i].get("power", 0), 1)
                          * _eff(ours["moves"][i], enemy))
                desc = ours["moves"][idx].get("name", desc)
                eff = _eff(ours["moves"][idx], enemy)
                self.log(f"   [engine] avoided a type-immune move -> {desc} (eff x{eff:g}) instead")
            elif not self._is_trainer_battle():
                self.log("   [engine] !! NO EFFECTIVE MOVE — every usable move is type-immune here "
                         "(need a better matchup: switch / flee)")
                return "no_effective_move"
            else:
                # WAR-MUST-ADVANCE (trainer battles can't flee, and the switch path already had its
                # shot upstream): swing the immune move anyway — "it doesn't affect..." still passes
                # the turn, the foe acts, and the battle resolves instead of livelocking (e4_run2).
                self.log(f"   [engine] NO EFFECTIVE MOVE in a TRAINER battle -> war-must-advance: "
                         f"firing {desc} anyway (a passed turn beats an eternal menu)")
        # ── STATUS-MOVE STRATEGY (general, E4-critical): when EVERY damaging move is RESISTED (best
        # eff <= 0.5 — e.g. Ivysaur's Grass moves into Gary's Fire Charmander, the live look-ahead wall),
        # raw chipping loses the damage race. A STATUS move is the real play: poison/Leech-Seed chips
        # TYPE-INDEPENDENTLY (bypasses the resistance), sleep neutralizes the foe. Fire it ONCE, early,
        # when the foe is still fresh (worth the turn), then go back to chipping while the status works.
        # Capability-not-script + general (cracks any resist-wall, not just Charmander). ───────────────
        _dmg_effs = [_eff(ours["moves"][i], enemy)
                     for i in range(4) if _usable(i) and ours["moves"][i].get("power", 0) > 0]
        best_dmg_eff = max(_dmg_effs) if _dmg_effs else 1.0
        foe_frac = enemy["hp"] / max(enemy.get("maxhp", 1), 1)
        # SLEEP-LOCK vs a SUPER-EFFECTIVE hard-hitter (general, E4-critical): when the foe hits US
        # super-effectively (it can fast-KO us) AND we can't out-damage it (best dmg resisted <=0.5),
        # SLEEP beats poison — it stops the incoming damage ENTIRELY instead of just chipping. Re-apply
        # whenever the foe is AWAKE (keep it locked) so our weak resisted chip wins the race safely; skip
        # while it's already asleep (don't waste the turn — chip instead). This is what actually cracks
        # Gary's Charmander (Ember 2x + burn on Ivysaur, Razor Leaf only 0.25x back). Soul-safe: she
        # learns "put the scary one to sleep, then whittle it" — a real player's resist-wall answer.
        _myt = [t for t in (ours.get("types") or []) if t and t != "???"]
        _foet = [t for t in (enemy.get("types") or []) if t and t != "???"]
        se_threat = (self._matchup_def(_myt, _foet) >= 2) if (_myt and _foet) else False
        # OBSERVED-SE-CHUNK LATCH (blaine_run2, the Cinnabar whiteout loop): the generic
        # sleep-lock below demands our damage be RESISTED (<=0.5) — vs Blaine our Normal
        # moves are x1 so it never armed, while his fire chunked our grass ace x2 through
        # a 4-deep potioning roster (attrition loss at a 12-level advantage). A real player
        # sleeps the scary one REGARDLESS of their own damage. Latch when we OBSERVE the
        # foe class actually chunk us (>=18% of our max between decisions — above a burn
        # tick's 12.5%, so wild trash and chip never arm it), then let the sleep-lock fire
        # on se_threat alone. Per-attach state, same whiff cap as the lock.
        _okey = (ours.get("species"), ours.get("maxhp"))
        _ohp = ours.get("hp", 0)
        if (se_threat and getattr(self, "_hp_key", None) == _okey
                and _ohp < getattr(self, "_hp_last", _ohp)
                and (self._hp_last - _ohp) / max(ours.get("maxhp", 1), 1) >= 0.18
                and not getattr(self, "_se_chunk_latch", False)):
            self._se_chunk_latch = True
            self.log("   [engine] SE-CHUNK observed: this foe class hits us super-effectively "
                     "and HARD -> sleep-lock armed even at neutral damage")
        self._hp_key, self._hp_last = _okey, _ohp
        sleep_done = False
        # SAFETY CAP: stop re-casting sleep after a few whiffs on the SAME foe — a foe that lowers our
        # accuracy (Smokescreen/Sand-Attack, e.g. Gary's Charmander) makes the 75%-acc powder MISS every
        # turn, so an uncapped sleep-lock loops forever (the 106-stuck regression). Past the cap, drop the
        # status play and just chip (the real answer to such a foe is a stronger teammate, not more sleep).
        # NUKE-SLEEP OPENER (koga_run3, the Koga wipe): a Self-Destruct-family foe can one-shot-
        # trade our active at ANY matchup — sleep it BEFORE it detonates, whatever our damage eff
        # (the generic sleep-lock below only fires when we're resisted+threatened, which is exactly
        # why it sat out vs Koga's x1-neutral Koffing). Shares the whiff cap; skips once the foe is
        # low (kill it instead — a KO can't detonate either).
        if (SLEEP_LOCK_ENABLED and enemy.get("species") in _NUKE_SPECIES
                and not enemy.get("asleep") and foe_frac > 0.30
                and getattr(self, "_sleep_casts", 0) < 4):
            si = next((i for i in range(4) if _usable(i)
                       and ours["moves"][i].get("id", 0) in self._SLEEP_MOVES), None)
            if si is not None:
                idx, desc, sleep_done = si, ours["moves"][si].get("name", "sleep"), True
                self._sleep_casts = getattr(self, "_sleep_casts", 0) + 1
                self.log(f"   [engine] NUKE-SLEEP: {st.SPECIES_NAME.get(enemy['species'], '?')} is "
                         f"Self-Destruct family -> {desc} first (it can't detonate asleep; "
                         f"try {self._sleep_casts}/4)")
        # NEVER sleep-lock a foe we're SUPER-EFFECTIVE on (best_dmg_eff >= 2): the E4 diag
        # (ns1) caught the battle-long _se_chunk_latch mis-firing sleep on Lorelei's Cloyster
        # (Razor Leaf x2 = an OHKO-range hit) — 4 wasted Sleep Powder turns per such foe across
        # rooms 1-4 burned the Full Restores Lapras needs to solo Gary at the Champion. If we can
        # 2x it, just KO it; the sleep-lock is only for foes we CANNOT out-damage.
        if (SLEEP_LOCK_ENABLED and not sleep_done and se_threat and best_dmg_eff < 2
                and (best_dmg_eff <= 0.5 or getattr(self, "_se_chunk_latch", False))
                and not enemy.get("asleep") and foe_frac > 0.30
                and getattr(self, "_sleep_casts", 0) < 4):
            si = next((i for i in range(4) if _usable(i)
                       and ours["moves"][i].get("id", 0) in self._SLEEP_MOVES), None)
            if si is not None:
                idx, desc, sleep_done = si, ours["moves"][si].get("name", "sleep"), True
                self._sleep_casts = getattr(self, "_sleep_casts", 0) + 1
                why = ("damage resisted" if best_dmg_eff <= 0.5
                       else "it chunks us at neutral damage")
                self.log(f"   [engine] SLEEP-LOCK: super-effective threat + {why} "
                         f"(best x{best_dmg_eff:g}) -> {desc} (neutralise its hits, then chip safely; "
                         f"try {self._sleep_casts}/4)")
        # ONE non-sleep status/foe (poison/leech — type-independent CHIP that bypasses the resistance) when
        # sleep-lock isn't the play; a 2nd status play made long fights wedge/time-out in the look-ahead.
        if not sleep_done and not getattr(self, "_status_played", False):
            if best_dmg_eff <= 0.5 and foe_frac > 0.5:
                _STATUS_PREF = ["leechseed", "toxic", "poisonpowder", "stunspore"]
                _norm = lambda s: "".join(s.lower().split())
                for want in _STATUS_PREF:
                    si = next((i for i in range(4) if _usable(i)
                               and _norm(ours["moves"][i].get("name", "")) == want), None)
                    if si is not None:
                        idx, desc, self._status_played = si, ours["moves"][si].get("name", want), True
                        self.log(f"   [engine] STATUS STRATEGY: damage resisted (best x{best_dmg_eff:g}) "
                                 f"-> {desc} (type-independent chip/neutralise past the wall)")
                        break
        self.log(f"   [engine] action menu: {desc} -> slot {idx} (eff x{eff:g}) vs "
                 f"{st.SPECIES_NAME.get(enemy['species'], '?')} {enemy['hp']}/{enemy['maxhp']}")
        # OPEN THE MOVE LIST ROBUSTLY: home to FIGHT, A, pixel-confirm the list opened; retry the
        # A if it was eaten (still at the white action menu); if a wrong submenu opened (bag/
        # POKEMON - NOT the white action panel) back out with B and re-home. Bounded.
        opened = False
        self._home_to_fight()
        for _ in range(12):
            # VERIFIED open (cursor-response, not just the MENU_MODE byte): a stale-2 byte after an item
            # use short-circuited this check before A was ever pressed = the immortal-Ekans wedge.
            if self._movelist_open_verified():
                opened = True; break
            self._home_to_fight()                     # a failed probe may have nudged the ACTION cursor
            #                                           (RIGHT lands on BAG) — re-home so A opens FIGHT
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
            if self._movelist_open_verified():
                opened = True; break
            if not (self._white_box() or self._movelist_open()):   # a wrong submenu opened -> back out
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
                self._home_to_fight()
        if not opened:
            return "stuck"                            # never opened -> clean retry (re-settle)
        if not self._goto_move(idx):                  # RAM-readback nav (verify each press moved the cursor)
            self.log(f"   [engine] move-cursor didn't reach slot {idx} (now {self.b.rd8(MOVE_CURSOR)}) "
                     f"-> clean retry")
            return "stuck"
        pp0 = ours["moves"][idx].get("pp", 0)
        before = self._bstate()
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        self._last_desc, self._last_eff = desc, eff   # narrated when the hit actually lands
        # VERIFY the move EXECUTED. The old fixed 220-frame window was TOO SHORT for a full trainer
        # turn (when we're slower, our hit lands AFTER the foe's move + animations) -> it timed out
        # on WORKING moves and (with benching) exiled them, losing winnable fights. Now we wait for
        # the TURN TO RESOLVE: a PP drop or ANY HP change = it fired (damage dealt or taken means a
        # move went off); battle ending = it fired (KO). Only if the turn settles back at the menu
        # with NO PP drop and NO HP change is it a true non-fire (Disable / can't-act).
        result = None
        last_hp, stable = before, 0
        for _ in range(900):
            if not st.in_battle(self.b):
                result = "done"; break                # battle ended (KO) = our move resolved
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                if cur["ours"]["moves"][idx].get("pp", 0) < pp0:      # our chosen move's PP dropped
                    result = "done"; break
                hp = (cur["enemy"]["hp"], cur["ours"]["hp"])
                if before and hp != before:           # ANY HP moved this turn -> a move executed
                    result = "done"; break
                stable = stable + 1 if hp == last_hp else 0
                last_hp = hp
                if self._white_box() and stable >= 30:   # settled back at the menu, nothing happened
                    break                                 # = the move never fired (Disabled/blocked)
            self.b.run_frame(); self.render()
        if result == "done":
            self._skip_streak.clear()                  # a move FIRED -> whole moveset eligible again
            #                                            (resets the streak; never permanently benches)
            self._immob_streak = 0                     # a resolved turn breaks any paralysis-spin count
        else:
            # 2026-07-06 THE BUTTERFREE SLEEP WEDGE: if OUR mon is ASLEEP/FROZEN/paralysis-skipped, the
            # turn RESOLVED — we just didn't act (no PP drop, and a Sleep-Powder/Harden foe changes no
            # HP either). Counting that as "didn't fire" benched every move, aborted the unfleeable
            # trainer battle, and travel re-entered it forever (Route 6, Butterfree 54/54 ×∞). An
            # immobilized turn is a REAL turn: report it resolved, keep the moveset eligible, and let
            # the fight continue — she wakes in 1-4 turns and Peck ends it.
            try:
                cur = st.read_battle(self.b)
                st1 = (cur or {}).get("ours", {}).get("status1", 0)
            except Exception:
                st1 = 0
            # IMMOBILIZATION vs a MASKED FAILURE (2026-07-10, night shift 8 — the S.S. Anne Gary
            # paralysis LIVELOCK): a set status bit does NOT prove THIS turn failed BECAUSE of it.
            # Paralysis (0x40) immobilizes only ~25% of turns, so a move that never fires while merely
            # paralysis-FLAGGED is really failing for another reason — almost always 0-PP (a PP-famine
            # move masked by the bit). VERIFIED: 183 turns of "fully paralyzed" vs a full-HP Kadabra
            # with 0 damage was a 0-PP Tackle spinning forever, because the paralysis branch swallowed
            # the 0-PP rotation. TWO GUARDS: (a) if the chosen move had 0 PP it CANNOT have fired
            # regardless of status -> it's a 0-PP non-fire, rotate; (b) cap CONSECUTIVE paralysis
            # attributions — >6 in a row is statistically impossible for real 25% paralysis, so stop
            # trusting the bit and rotate/flee. Sleep(0x07)/freeze(0x20) legitimately immobilize many
            # turns in a row, so they keep the old trust-indefinitely behaviour (guard (a) still applies).
            _slp_frz = st1 & 0x27                      # sleep | freeze — legitimately multi-turn
            _par = st1 & 0x40                          # paralysis — at most ~25%/turn
            _ims = getattr(self, "_immob_streak", 0)
            _real_immob = pp0 > 0 and (_slp_frz or (_par and _ims < 6))
            if _real_immob:
                self._immob_streak = _ims + 1
                why = "asleep" if st1 & 0x07 else ("frozen" if st1 & 0x20 else "fully paralyzed")
                self.log(f"   [engine] turn resolved by IMMOBILIZATION ({why}) — not a dead move; "
                         f"fighting on (she'll come around)")
                self.emit(f"no — {desc} didn't happen, I'm {why}! hang in there…", beat=True, tier=1)
                return "done"
            self._immob_streak = 0                     # not (or no longer) a trusted immobilization
            # didn't fire (no PP drop, no HP change) = 0-PP / Disabled / couldn't act: add to the streak
            # so the NEXT pick rotates to a move she hasn't tried — and once all are tried, she flees
            # rather than re-spamming. The streak is per-no-progress-run, cleared on any successful fire.
            self._skip_streak.add(idx)
            self.log(f"   [engine] move slot {idx} didn't fire (0-PP / disabled / blocked) -> rotating "
                     f"to an untried move (streak now {sorted(self._skip_streak)})")
        return result or "stuck"

    def _advance_text(self, force_b=False):
        """Advance battle dialogue/animation SAFELY. Diagnosed 2026-06-23: (a) mashing A
        *into* an animation (the player walk-in, a faint, the EXP bar) WEDGES the input and
        the text then never advances - so we WAIT a beat for the animation to settle first;
        (b) the wild 'X appeared!' / 'X fainted!' gates advance on B, not A - so after a clean
        A tap we also tap B, but ONLY if the white action-menu panel is NOT up (so B can never
        be read as RUN/flee). Clean discrete taps (short hold, long release) - a held/too-fast
        press reads as one input.
        force_b (2026-06-24): in the POST-FAINT drain the foe already fainted (no flee risk),
        and the TRAINER defeat/prize screen lights the white-panel pixels as a FALSE POSITIVE
        while actually needing B to advance - so force_b taps B regardless of _white_box, which
        is what lets a trainer battle exit cleanly after its last mon faints."""
        self._wait(18)
        self.b.press("A", 2, 12, self.render, owner=self.owner)
        if force_b or not self._white_box():
            self.b.press("B", 2, 12, self.render, owner=self.owner)

    def _reach_first_menu(self, t0, max_seconds):
        """Advance the battle intro (walk-in + 'X appeared!' + 'Go MON!') to the first action
        menu (white panel up), so the foe species (gBattleMons[1] is stale until the intro
        advances) reads true."""
        for _ in range(40):
            if not st.in_battle(self.b) or time.time() - t0 > max_seconds:
                return
            if self._white_box():
                return                                # action menu reached
            # a battle can OPEN on the forced send-out party screen (fainted lead) — A/B mashing
            # there selects/cancels the fainted slot-0 forever; return and let the caller's
            # party-screen handling own it (flee's send-out-first / run()'s post-faint drain).
            if self._party_screen():
                return
            self._advance_text()

    # ── forced faint-switch (party>=2; the lead goes down mid-battle) ───────────
    # Until the phantom-A fix (a463055), an incidental A confirmed the "Choose a POKéMON" menu
    # so the switch "just happened"; with input clean it must be navigated explicitly. None of
    # the party=1 regression fixtures exercised this, so the fix exposed it. Now buildable.
    def _healthy_reserve_slot(self, skip=()):
        """First party slot with current-HP > 0 (not in `skip`), or None. Party current-HP is
        UNencrypted at +0x56 in the 100-byte party struct (level is at +0x54, used elsewhere)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            if s in skip:
                continue
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) > 0:
                return s
        return None

    # BATTLE party-screen cursor READBACK (erika_run1 wedge, 2026-07-07): the party menu REMEMBERS
    # its cursor across opens, so after mid-battle switches the forced-switch screen opens on the
    # FAINTED active mon's slot — blind DOWN*slot from an assumed slot-0 start selected the corpse
    # ("MANKEY has no energy left to battle!") 65 times to timeout. Read the selected slot's ORANGE
    # border (255,115,49 — measured on switch_right.png) instead. NOTE: battle right-column slot
    # tops are y=10+24*(s-1) (24px pitch, measured on s3_down1.png) — NOT the overworld menu's 21px
    # pitch (hm_teach._SLOT_TOPS); the two screens differ, don't share anchors.
    @staticmethod
    def _cursor_orange(c):
        """The selection-border orange, bright phase (255,115,49 — switch_right.png ground truth)
        OR its palette-fade dim phase (~(123,90,57), s2_list0.png was captured mid-fade). A miss on
        a fading frame just costs one retry loop, but accepting both phases reads through it."""
        r, g, bl = c[:3]
        return (r > 240 and 80 < g < 140 and bl < 70) or \
               (100 < r < 170 and 60 < g < 125 and bl < 80 and r > g > bl)

    def _party_cursor_slot(self):
        """Selected RIGHT-column slot (1-5) on the in-battle party screen, or None (lead/CANCEL/
        no border found). Border = horizontal orange run across the slot's top edge."""
        p = self.b.frame_rgb().load()
        for slot in (1, 2, 3, 4, 5):
            y0 = 10 + 24 * (slot - 1)
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                n = sum(1 for x in (110, 140, 170, 200, 225)
                        if self._cursor_orange(p[x, y0 + dy]))
                if n >= 4:
                    return slot
        return None

    def _party_cursor_on_lead(self):
        """True iff the LEAD (left panel) is the selected slot — its top border (y≈26, x 4..90)
        lights the same orange (switch_right.png ground truth: rows 26-27 lit across x 10-80)."""
        p = self.b.frame_rgb().load()
        for y in (25, 26, 27, 28):
            n = sum(1 for x in (10, 30, 50, 70) if self._cursor_orange(p[x, y]))
            if n >= 3:
                return True
        return False

    # ── THE PARTY-MENU ORDER LAW (recon_partytruth, 2026-07-07 — settles the flip-flop) ──
    # gPlayerParty HP is LIVE and accurate at all times (probe: Raticate ticked 37->24->11->
    # 7->0 at its own slot while active). While the in-battle party MENU is open, the game
    # PHYSICALLY rearranges gPlayerParty into display order (= gBattlePartyCurrentOrder
    # nibbles) and restores it when the menu closes (probe: raticate sat at s0 during menu2,
    # back at s3 the next turn). So: display row i IS gPlayerParty[i] — but ONLY while the
    # menu is open. Both prior models were half-right; the whole bug family (run12 double-
    # convert, run14 Revive-on-the-wrong-row, voluntary switches mis-landing post-switch)
    # was carrying a slot index ACROSS the menu-open boundary. NEVER do that: decide WHAT
    # to target before the menu (species/fainted/active), resolve WHICH ROW at menu time.
    def _menu_rows(self):
        """Per-display-row content of the OPEN in-battle party menu: [{row, species, hp,
        maxhp, level}]. Only valid while the party screen is up (the order law above)."""
        rows = []
        for i in range(6):
            sp = st.read_party_species(self.b, i)
            if not sp:
                break
            base = ram.GPLAYER_PARTY + i * 100
            rows.append({"row": i, "species": sp,
                         "hp": self.b.rd16(base + 0x56),
                         "maxhp": self.b.rd16(base + 0x58),
                         "level": self.b.rd8(base + 0x54)})
        return rows

    # The SEND OUT/SHIFT sub-menu box (bottom-right, 3 rows) — pixel ground truth measured
    # across menu1_afterA.png + both run14 fswitch frames vs the plain list (teal stripes
    # (69,164,158) at these points on every plain-list frame). _WHITE_PTS can NOT tell the
    # two apart (the plain list's bottom bar scores 4/6 there).
    _SUBMENU_PTS = ((210, 130), (230, 130))

    def _party_submenu(self):
        """True iff the party menu's SEND OUT/SHIFT/SUMMARY sub-menu (or an equally-placed
        sub-box) is open over the party screen."""
        p = self.b.frame_rgb().load()
        return all(min(p[x, y]) > 200 for x, y in self._SUBMENU_PTS)

    def _party_focus(self, tries=8):
        """Make the party LIST own the input focus before any cursor walk. Kills BOTH
        tap-eater classes caught on frames tonight: the SEND OUT sub-menu (run14: the old
        blind DOWN probe moved the SUB-MENU cursor to SUMMARY, so the confirm A opened the
        summary screen — 3 minutes of churn into corpses) and the 'has no will to fight!'
        message box (run11). Sub-menu up -> B it away FIRST; then probe with DOWN and
        require the list cursor to actually MOVE; eaten taps -> B-dismiss + retry. Never
        presses A (an unfocused A is how the churn re-armed itself)."""
        for _ in range(tries):
            if not self._party_screen():
                return False
            if self._party_submenu():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(16)
                continue
            c0 = self._party_cursor_slot()
            self._tap("DOWN")
            self._wait(14)
            c1 = self._party_cursor_slot()
            if c1 != c0 or (c1 is None and self._party_cursor_on_lead()):
                return True
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(20)
        return False

    def _party_goto_slot(self, target, tries=14):
        """Closed-loop cursor walk on the BATTLE party screen. target = party index 0-5 (0 = lead
        panel). Returns True only when the border readback confirms the cursor is on target."""
        for _ in range(tries):
            cur = self._party_cursor_slot()
            if cur is None and self._party_cursor_on_lead():
                cur = 0
            if cur == target:
                return True
            if cur is None:                               # nothing lit -> CANCEL (bottom-right)
                self._tap("UP"); self._wait(16)           # UP from CANCEL lands slot 5
                continue
            if target == 0:
                self._tap("LEFT"); self._wait(16)         # any right slot -> LEFT = lead panel
                continue
            if cur == 0:
                self._tap("RIGHT"); self._wait(16)        # lead -> enter the right column
                continue
            self._tap("DOWN" if cur < target else "UP"); self._wait(16)
        cur = self._party_cursor_slot()
        if cur is None and self._party_cursor_on_lead():
            cur = 0
        return cur == target

    def _force_switch(self):
        """Lead fainted with a healthy reserve -> the 'Choose a POKéMON' party menu is up.
        Walk the cursor CLOSED-LOOP (border readback) to the first healthy slot and confirm
        SEND OUT (the select submenu defaults to SEND OUT). Returns True once a healthy mon
        is active. Last-resort attempts fall back to the legacy blind DOWN*slot walk (covers
        a palette/geometry miss on the readback — it logs which path ran)."""
        if self._healthy_reserve_slot() is None:
            return False
        _tried = set()                                    # SPECIES that failed 2x -> rotate past them
        _fails = {}                                       # (species-keyed: display rows move between
        #                                                    menu opens, species identity doesn't)
        for _attempt in range(8):
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0:
                return True                               # a healthy mon is active -> switched
            self._wait(18)                                # let the party menu settle
            if not self._party_screen():
                self._advance_text()                      # faint text still playing -> drain a beat
                continue
            # LIST FOCUS first (sub-menu/message tap-eaters), then resolve the target row by
            # CONTENT at menu time — the order law: row i IS gPlayerParty[i] only while the
            # menu is open; any slot picked before it opened is in a different order.
            if not self._party_focus():
                self.log("   [engine] fswitch: party list never regained focus -> retry")
                self._debug_snap(f"fswitch_nofocus{_attempt}")
                continue
            rows = self._menu_rows()
            live = [r for r in rows if r["row"] > 0 and r["hp"] > 0
                    and r["species"] not in _tried]
            if not live:
                _tried.clear()
                live = [r for r in rows if r["row"] > 0 and r["hp"] > 0]
                if not live:
                    return False                          # nothing standing on the bench
            tgt = max(live, key=lambda r: r["level"])     # send the strongest thing standing
            if _attempt >= 1:                             # retry forensics
                self.log(f"   [engine] fswitch retry {_attempt}: target row {tgt['row']} "
                         f"sp={tgt['species']} menu_rows="
                         f"{[(r['species'], r['hp']) for r in rows]}")
                self._debug_snap(f"fswitch_retry{_attempt}")
            _fails[tgt["species"]] = _fails.get(tgt["species"], 0) + 1
            if _fails[tgt["species"]] > 2:
                _tried.add(tgt["species"])
            if not self._party_goto_slot(tgt["row"]):
                self.log(f"   [engine] fswitch: cursor readback couldn't reach row {tgt['row']} "
                         f"(cursor={self._party_cursor_slot()}) -> retry")
                continue
            self.log(f"   [engine] fswitch: cursor confirmed on row {tgt['row']} "
                     f"(sp={tgt['species']}, menu-time content)")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # select mon
            for _ in range(8):                            # WAIT for the sub-menu to draw — an early
                if self._party_submenu():                 # 2nd A used to land back on the LIST and
                    break                                 # leave the sub-menu dangling (run14 churn)
                self._wait(8)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # -> SEND OUT
            self._wait(20)
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0:
                return True
            self._advance_text()                          # send-out text -> drain a beat, re-check
        cur = st.read_battle(self.b)
        return bool(cur and cur["ours"]["hp"] > 0)

    # ── B-1: TYPE-MATCHUP AWARENESS + VOLUNTARY SWITCH (the E4-critical verb) ────
    def _goto_pokemon(self, tries=10):
        """Walk the action cursor to POKEMON (bottom-left, ACT_POKEMON=2). Mirror of _goto_run; grid is
        FIGHT(0,TL) BAG(1,TR) / POKEMON(2,BL) RUN(3,BR). Returns True only when confirmed on POKEMON."""
        for _ in range(tries):
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_POKEMON:
                return True
            if c == ram.ACT_FIGHT:
                self._tap("DOWN")
            elif c == ram.ACT_BAG:
                self._tap("DOWN"); self._tap("LEFT")
            elif c == ram.ACT_RUN:
                self._tap("LEFT")
            else:
                return False                              # not the action menu
            self._wait(3)
        return self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_POKEMON

    @staticmethod
    def _matchup_def(my_types, enemy_types):
        """How hard the enemy's STAB hits `my_types` (max eff of any enemy type vs mine). >=2 = enemy
        super-effective on me (bad); <=0.5 = I resist (good). Enemy moves are unknown, so its own types
        are the STAB proxy."""
        worst = 0.0
        for et in enemy_types:
            if et:
                worst = max(worst, pol.effectiveness(et, my_types))
        return worst or 1.0

    @staticmethod
    def _matchup_off(my_types, enemy_types):
        """Best eff of MY types vs the enemy (STAB proxy) — can I hit it hard?"""
        best = 0.0
        for t in my_types:
            if t:
                best = max(best, pol.effectiveness(t, enemy_types))
        return best

    def _best_switch_slot(self, state):
        """A CLEARLY-better healthy reserve to switch into, or None. Conservative (never churn): only
        when the ACTIVE mon is at a real disadvantage — enemy hits it super-effectively OR it can't
        damage the enemy at all — AND a healthy reserve exists that the enemy does NOT hit
        super-effectively. Ranks candidates by (resists-most, hits-hardest). Pure type math (offline-
        testable); reads non-lead species from RAM."""
        enemy_types = [t for t in (state.get("enemy", {}).get("types") or []) if t]
        if not enemy_types:
            return None
        active_types = [t for t in (state.get("ours", {}).get("types") or []) if t]
        # 2026-07-06 OFFENSIVE-RESIST trigger, MOVE-BASED (run-16 lesson): the TYPE proxy lied —
        # Ivysaur's poison TYPING scores 0.5 vs Weedle, but her only damaging MOVES are grass (0.25x
        # Razor Leaf), so the type math never tripped and the gauntlet fight chipped for 15 minutes.
        # Judge by the best USABLE damaging move she actually has; a hard-resisted moveset (<=0.25x)
        # swaps to a reserve with a neutral/SE hit (Spearow's Peck). 0.5x stays acceptable (no churn).
        _dmg = [_eff(m, state.get("enemy") or {})
                for m in (state.get("ours", {}).get("moves") or [])
                if m.get("id", 0) and m.get("pp", 0) > 0 and m.get("power", 0) > 0]
        best_move_eff = max(_dmg) if _dmg else 1.0
        # NEVER ABANDON A SUPER-EFFECTIVE ATTACKER (ns14 anti-churn): if the active mon's best
        # damaging move is >=2x, it's winning the exchange — pulling it out for a defensive matchup
        # just churns. The infinite loop this kills: Kadabra's Psybeam is 2x into Agatha's Poison
        # line (stay + sweep), but Agatha's Ghost hits Psychic 2x, so the disadvantage trigger kept
        # yanking Kadabra out for Venusaur, whose Razor Leaf is 0.5x, so trigger 2 pulled Kadabra
        # straight back — Venusaur<->Kadabra forever, bleeding both. A glass cannon that out-damages
        # STAYS and swings.
        # SE-ACTIVE: resolved AFTER the reserve scan (below) so the ns23 load-share exception can
        # reference a healthy SE partner. The plain anti-churn `return None` is preserved there.
        active_se = best_move_eff >= 2.0
        active_bad = self._matchup_def(active_types, enemy_types) >= 2 or best_move_eff <= 0.25
        foe_lv = state.get("enemy", {}).get("level") or 0
        act_lv = state.get("ours", {}).get("level") or 0
        active_sp = state.get("ours", {}).get("species")
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        # ONE reserve scan feeds two picks: `best` (defensive — resists-most, then hits-hardest)
        # for the disadvantage trigger; `best_atk` (the SUPER-EFFECTIVE specialist — hits-hardest)
        # for the offensive-upgrade trigger. Pure type math (offline-testable), species from RAM.
        best, best_key = None, None
        best_atk, best_atk_key = None, None
        best_share, best_share_key = None, None          # ns23: healthy SE reserve for the load-share
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            sp = st.read_party_species(self.b, s)
            if sp == active_sp:
                continue                                  # (probably) the one already out
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            types = st.species_types(sp)
            if not types:
                continue
            cdef = self._matchup_def(types, enemy_types)
            coff = self._matchup_off(types, enemy_types)
            # OFFENSIVE-SPECIALIST pick (ns14): a reserve whose STAB is SUPER-EFFECTIVE (>=2x).
            # LENIENT floor (lv+15 vs the def pick's lv+5): a 2x type edge is worth ~2 level-tiers
            # of frailty, so an under-levelled specialist (Kadabra L40 Psychic into Agatha's L54
            # Poison line) is still the right body — the whole point is to spare the ace's PP.
            # A >=4x answer (Lapras's Ice Beam into Lance's Dragon/Flying line) is a near-certain
            # OHKO from any healthy body — field it regardless of level. The lv+15 floor wrongly
            # vetoed the bulky L39 Lapras vs the ~L55 dragons, stranding Ice Beam in reserve while
            # the ace tanked to a whiteout (e4_tactical run1 Lance postmortem). The 2x case keeps the
            # lenient lv+15 frailty floor (a 2x edge is worth ~2 tiers, but not unlimited under-level).
            # MOVE-GATE (ns15, the Route-22 Gary FREEZE): coff is TYPE-based, so this fielded a mon
            # whose TYPE is SE but that has NO actual SE MOVE — giovanni_kit_g's Lapras is Ice-TYPE
            # (2x vs Grass exeggcute) but its moveset is [Surf, Body Slam] with no Ice move. Then each
            # out-typed mon had an SE TYPE but no SE move, so the switch pick ping-ponged A<->B every
            # turn WITHOUT ever attacking — a hard livelock (123 switch/no-progress churns observed).
            # Require the reserve to actually HAVE a damaging move that's SE (r_eff, via st.move_info +
            # _eff so the Levitate layer applies), not merely an SE typing. The proven E4 specialists
            # (Kadabra Psybeam, Lapras Ice Beam) still qualify — they carry the move. r_eff replaces
            # coff for the offensive gate/key; coff stays only in the defensive `best` tie-break below.
            r_eff = 0.0
            for _mid in st.read_party_moves(self.b, s):
                if not _mid:
                    continue
                _mt, _mp = st.move_info(self.b, _mid)
                if _mp and _mp > 0:
                    r_eff = max(r_eff, _eff({"type": _mt or "normal"}, state.get("enemy") or {}))
            _floor_ok = (r_eff >= 4.0) or not (foe_lv and lv + 15 < foe_lv)
            if r_eff >= 2.0 and _floor_ok:
                akey = (r_eff, -cdef, lv)                  # hits hardest (real move), resists, level
                if best_atk_key is None or akey > best_atk_key:
                    best_atk, best_atk_key = s, akey
                # ns23 LOAD-SHARE partner: this SE reserve is ALSO eligible to relieve a critical SE
                # active — but only if it's genuinely HEALTHY (a fresh tank, not another dying body).
                # Rank by (hits-hardest, healthiest, level). Same >=2x gate keeps the swap churn-safe.
                if BATTLE_LOAD_SHARE:
                    s_hp = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
                    s_mx = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
                    s_frac = (s_hp / s_mx) if s_mx else 1.0
                    if s_frac >= SWITCH_SHARE_HEALTHY_FRAC:
                        skey = (r_eff, s_frac, lv)
                        if best_share_key is None or skey > best_share_key:
                            best_share, best_share_key = s, skey
            if foe_lv and lv + 5 < foe_lv:
                continue                                  # FODDER FLOOR: switching INTO a faint is
                                                          # never an improvement (the Ekans churn)
            if cdef >= 2:
                continue                                  # also weak — not an improvement
            # resists most, then hits hardest, then the higher level (level breaks type ties)
            key = (-cdef, coff, lv)
            if best_key is None or key > best_key:
                best, best_key = s, key
        # SE-ACTIVE ANTI-CHURN (was the early return at the top): a >=2x attacker wins the exchange and
        # STAYS — pulling it out just churns (the Kadabra<->Venusaur loop: Psybeam 2x into Agatha but her
        # Ghost hits Psychic 2x, so trigger 1 would yank it for a 0.5x Venusaur, then trigger 2 pulls it
        # straight back). LOAD-SHARE EXCEPTION (ns23): if it's CRITICALLY low AND a HEALTHY reserve is
        # ALSO >=2x on this foe, rotate to that fresh SE body so one specialist doesn't solo a gauntlet to
        # death. The target is itself >=2x -> once out it hits this same return None and STAYS; no ping-pong.
        if active_se:
            if BATTLE_LOAD_SHARE and best_share is not None \
                    and _hp_frac(state.get("ours") or {}) <= BATTLE_CRIT_FRAC:
                self.log(f"   [engine] LOAD-SHARE: SE active critical -> field healthy SE reserve slot {best_share}")
                return best_share
            return None
        # TRIGGER 1 — DISADVANTAGE (existing): the active is out-typed OR can barely scratch the
        # foe (<=0.25x). LEVEL-DOMINANCE VETO (erika_run3): a crushing level lead wins through 0.5x
        # resistance — but offensive famine (<=0.25x) overrides it (flute_run7: Venusaur's 0.25x
        # into a Vileplume is a stall, not dominance). Send the defensive pick.
        if active_bad:
            # A SUPER-EFFECTIVE reserve beats grinding the ace through the type disadvantage, EVEN at a
            # crushing level lead: field the specialist so the ace stops eating 2x hits (ns1 Champion —
            # L88 Venusaur traded itself to Pidgeot on Cut x1 while Lapras's Ice Beam 2x sat in reserve;
            # the level-dominance veto below kept the ace in to die). Fielding Lapras for the ace's bad
            # matchups (Ice/Fire/Flying) also cuts the heal spend in rooms 1-4 so Full Restores survive
            # to the Champion. Anti-churn holds: once the specialist is out and hitting >=2x, line ~1931
            # returns None (it stays); the lv+15 floor already kept frail chaff out of best_atk.
            if best_atk is not None:
                return best_atk
            if foe_lv and act_lv >= foe_lv + 10 and best_move_eff > 0.25:
                return None
            return best
        # TRIGGER 2 — OFFENSIVE-UPGRADE (ns14 Lance postmortem, E4-critical): the active can only hit
        # RESISTED (<=0.5x — Venusaur's Razor Leaf into Agatha's Poison, its Normal moves IMMUNE to her
        # Ghosts, or its 0.25x into Gary's Charizard) while a healthy reserve is SUPER-EFFECTIVE. Field
        # the specialist so the ace's scarce STAB PP survives the gauntlet. Kept at <=0.5x (NOT widened
        # to <=1x): e4_tactical run4 proved a <=1x gate over-fields the FRAIL glass-cannon (Kadabra base
        # HP 40) into OHKOs at Bruno/Agatha, burning the bench before Lance — worse than the tank line.
        # The bulky Lapras still fields vs Dragonite (>=4x override) and vs Charizard (Venusaur 0.25x).
        if best_atk is not None and best_move_eff <= 0.5:
            return best_atk
        return None

    def _switch_to_slot(self, slot, before_sp):
        """Switch the active mon to a SPECIFIC party slot, confirming the active SPECIES actually changed.
        FAIL-SAFE: if it doesn't confirm, B back to the action menu and return False (caller fights —
        never wedges). Returns 'switched' or False. Shared by the matchup switch + the grind switch.

        `slot` is a PRE-MENU gPlayerParty index — its SPECIES is read before the menu opens, then the
        target ROW is re-resolved by content once the menu is up (the order law at _menu_rows: the menu
        physically rearranges gPlayerParty while open, so a pre-menu index walked on the open menu lands
        on the wrong mon after any earlier switch). A = select -> sub-menu (cursor defaults to SHIFT),
        A = SHIFT -> the swap; then PURE-A advance the "Come back X! / Go Y!" text until the active
        SPECIES flips to the TARGET (the ground-truth success signal)."""
        want_sp = st.read_party_species(self.b, slot)             # identity survives the reorder
        if not want_sp or want_sp == before_sp:
            return False
        if not self._settle_action_menu():
            self.log("   [engine] switch: couldn't reach a clean action menu")
            return False
        if not self._goto_pokemon():
            self.log("   [engine] switch: _goto_pokemon failed (cursor not on POKEMON)")
            return False
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
        self._wait(30)                                            # open party list + SETTLE
        for _ in range(8):
            if self._party_screen():
                break
            self._wait(8)
        if not self._party_screen() or not self._party_focus():
            self.log("   [engine] switch: party screen never took focus -> B out (fail-safe)")
            self._exit_bag()
            return False
        row = next((r["row"] for r in self._menu_rows()
                    if r["species"] == want_sp and r["hp"] > 0), None)
        self.log(f"   [engine] switch: target party slot {slot} sp={want_sp} -> menu row {row}")
        if row is None or row == 0 or not self._party_goto_slot(row):
            self.log("   [engine] switch: target row unreachable -> B out (fail-safe)")
            for _ in range(4):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                if self._white_box() and not self._party_screen():
                    break
            return False
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)   # select -> sub-menu
        for _ in range(8):
            if self._party_submenu():
                break
            self._wait(8)
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(18)  # confirm SHIFT
        for _adv in range(16):                                    # advance swap text until the SPECIES flips
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0 and cur["ours"].get("species") == want_sp:
                self.log(f"   [engine] switch: SWITCHED to species {want_sp} (slot {slot})")
                return "switched"
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
        for _ in range(3):                                # didn't confirm -> back to the action menu, FIGHT
            if self._white_box():
                break
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        return False

    def _load_share_slot(self, state):
        """NS23 PRE-HEAL load-share: if the active is a WORN SE (>=2x) attacker AND a NEAR-FULL reserve is
        ALSO >=2x on this foe, return that fresh SE slot to rotate into INSTEAD of spending a heal —
        spreading gauntlet attrition across two SE bodies and conserving the scarce heal items (the E4
        Champion whiteout is a Full-Restore famine, not a level wall). Pure type math + RAM HP reads.

        Churn-safe: the near-full gate (SWITCH_SHARE_NEARFULL_FRAC) is monotonic — a benched mon doesn't
        regenerate, so once it drops below near-full it can't bounce back to be re-picked; and the target
        is itself >=2x, so the anti-churn rule keeps it in once it's out. At most one rotation per fresh
        partner. Returns a party slot or None."""
        if not BATTLE_LOAD_SHARE:
            return None
        ours = state.get("ours") or {}
        if _hp_frac(ours) > SWITCH_SHARE_WORN_FRAC:
            return None                                   # active still fresh — nothing to share
        enemy = state.get("enemy") or {}
        enemy_types = [t for t in (enemy.get("types") or []) if t]
        if not enemy_types:
            return None
        # active must itself be an SE attacker — the ONLY reason it's staying in is the anti-churn rule;
        # this is the exact case that solos a gauntlet to death while a healthy SE partner idles.
        _dmg = [_eff(m, enemy) for m in (ours.get("moves") or [])
                if m.get("id", 0) and m.get("pp", 0) > 0 and m.get("power", 0) > 0]
        if not _dmg or max(_dmg) < 2.0:
            return None
        active_sp = ours.get("species")
        foe_lv = enemy.get("level") or 0
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        best, best_key = None, None
        for s in range(min(cnt, 6)):
            hp = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
            mx = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
            if hp <= 0 or not mx:
                continue
            frac = hp / mx
            if frac < SWITCH_SHARE_NEARFULL_FRAC:
                continue                                  # must be a genuinely FRESH body
            sp = st.read_party_species(self.b, s)
            if sp == active_sp:
                continue
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            r_eff = 0.0
            for _mid in st.read_party_moves(self.b, s):
                if not _mid:
                    continue
                _mt, _mp = st.move_info(self.b, _mid)
                if _mp and _mp > 0:
                    r_eff = max(r_eff, _eff({"type": _mt or "normal"}, enemy))
            if r_eff < 2.0:
                continue
            if not ((r_eff >= 4.0) or not (foe_lv and lv + 15 < foe_lv)):
                continue                                  # frailty floor (same as _best_switch_slot)
            key = (r_eff, frac, lv)                        # hits hardest, then freshest, then level
            if best_key is None or key > best_key:
                best, best_key = s, key
        return best

    def _voluntary_switch(self, state):
        """Mid-battle switch to a better-matchup reserve. GATED + FAIL-SAFE. Returns 'switched' or False."""
        slot = self._best_switch_slot(state)
        if slot is None:
            return False
        self.log(f"   [engine] MATCHUP SWITCH: active is out-typed -> trying party slot {slot}")
        r = self._switch_to_slot(slot, state.get("ours", {}).get("species"))
        if r == "switched":
            self.emit("switching it up — this is a better matchup", beat=True, tier=2)
            self._skip_streak.clear()
            return "switched"
        self.log("   [engine] matchup switch did not confirm -> fighting instead (fail-safe, no wedge)")
        return False

    def _active_pp_famine(self, state):
        """True iff the ACTIVE mon has no damaging move with PP left that can CONNECT vs the
        CURRENT foe (gBattleMons ground truth — power from ROM gBattleMoves via read_battle).
        Status-move PP doesn't count (can never KO), and neither does a type-IMMUNE damaging
        move (e4_run7 Agatha: Venusaur with only Normal-type PP vs Gengar burned ~10
        war-must-advance turns + 2 Full Restores while Persian's Bite sat on the bench —
        immune-only PP is famine vs THIS foe, and the famine switch is the only winning line).
        Unknown foe types count as connecting (never over-trigger)."""
        mv = (state.get("ours") or {}).get("moves") or []
        return not any(m.get("id") and m.get("pp", 0) > 0 and m.get("power", 0) > 0
                       and self._move_connects(m, state) for m in mv)

    def _move_connects(self, m, state):
        """Can this move DAMAGE the current foe at all? (_eff = type chart + the ability layer;
        unknown foe types count as connecting — never over-trigger a famine.)"""
        enemy = state.get("enemy") or {}
        foet = [t for t in (enemy.get("types") or []) if t and t != "???"]
        if not foet:
            return True
        return _eff(m, enemy) > 0

    def _pp_reserve_slot(self, state):
        """The best ALIVE party member that still has DAMAGING PP and is not the mon already out
        (species compare — after a switch the active is no longer gPlayerParty[0], the ace-guard
        lesson). Highest level wins. None if nobody qualifies (whole party dry -> heal, not switch)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        active_sp = (state.get("ours") or {}).get("species")
        best, best_lv = None, -1
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            try:
                if st.read_party_species(self.b, s) == active_sp:
                    continue                              # already the mon that's out
            except Exception:
                pass
            if not st.slot_has_damaging_pp(self.b, s):
                continue
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            if lv > best_lv:
                best, best_lv = s, lv
        return best

    def _ace_reserve_slot(self):
        """The highest-level ALIVE party member that is NOT slot 0 — the ace to switch the weak grind
        lead out to (it tanks + KOs while the benched weak mon banks participation XP). None if no alive
        reserve outranks the lead (then there's nothing to switch to — just fight)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        lead_lv = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        best, best_lv = None, lead_lv
        for s in range(1, min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            if lv > best_lv:
                best, best_lv = s, lv
        return best

    def _solo_overlevel_ok(self, state):
        """SELECTIVE SOLO gate (NS#26, POKEMON_SOLO_OVERLEVEL_GRIND — see the flag comment). True iff the
        fielded weak lead SAFELY out-levels THIS foe (>= SOLO_OVERLEVEL_MARGIN above it) so it one-shots the
        wild taking ~0 damage — no faint, no in-battle heal (no white-box menu exposure). When True the
        participation GRIND SWITCH is SKIPPED so the weak mon SOLOS for the FULL kill XP (~2x the share the
        switch banks) while PROTECT_LEAD_GRIND stays True (the matchup switch stays suppressed). Per-foe
        self-correcting: a wild within the margin still gets the ace-protect switch. Byte-inert (False) OFF."""
        if not SOLO_OVERLEVEL_GRIND or state is None:
            return False
        our_lv = (state.get("ours") or {}).get("level") or 0
        foe_lv = (state.get("enemy") or {}).get("level") or 0
        return bool(our_lv and foe_lv and our_lv >= foe_lv + SOLO_OVERLEVEL_MARGIN)

    def _ours_dmg_pp(self, state):
        """Total remaining PP across the ACTIVE mon's DAMAGING moves (power>0). The whiff-spiral
        baseline: a status move (Sleep Powder) firing changes no foe HP by design and must NOT be
        counted as a missed attack — only a damaging move that drops PP with the foe's HP frozen is."""
        return sum(m.get("pp", 0) for m in (state.get("ours", {}).get("moves") or [])
                   if m.get("id", 0) and m.get("power", 0) > 0)

    def _any_healthy_reserve(self, state):
        """A non-active party slot with HP>0 WORTH benching the ace for during the accuracy-reset
        maneuver. Prefer the HIGHEST-level reserve (best chance of surviving the one incoming hit before
        we switch the ace back). None if the ace is alone OR every reserve is far below the ace's level —
        a solo-carry must NEVER swap its L52 ace out for L13 fodder to 'reset accuracy' (the badge-5 Koga
        loss: Muk's Minimize is FOE evasion, which the switch can't reset, so it only sacrificed the bench
        one by one). Gated by WHIFF_RESERVE_LEVEL_BAND; below the band -> None -> the caller fights on."""
        active_sp = (state or {}).get("ours", {}).get("species")
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        # the ace's own level = the floor a reserve must come within to be worth the bench-and-return
        ace_lv = 0
        for s in range(min(cnt, 6)):
            if st.read_party_species(self.b, s) == active_sp and \
                    self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) > 0:
                ace_lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54); break
        floor = ace_lv - WHIFF_RESERVE_LEVEL_BAND
        best, best_lv = None, -1
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue                                  # fainted
            sp = st.read_party_species(self.b, s)
            if sp == active_sp:
                continue                                  # the one already out
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            if lv < floor:
                continue                                  # fodder — not worth sacrificing the carry for
            if lv > best_lv:
                best, best_lv = s, lv
        return best

    def _slot_of_healthy_species(self, species):
        """Current gPlayerParty slot (0..5) holding `species` with HP>0, or None. Scans ALL slots
        because an in-battle switch can shuffle party indices — the whiff switch-back must find the
        ace wherever it now sits, not assume it's still a slot 1..5 reserve."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue
            if st.read_party_species(self.b, s) == species:
                return s
        return None

    def _classify_prev_whiff(self, state):
        """Race-free whiff classification (2026-07-10, night shift 11 — the S.S. Anne Gary ROOT CAUSE,
        corrected). The old in-turn read (a short _settle right after the move COMMITTED) raced our
        move's pre-damage text/animation window: battle text needs ACTIVE advancement (the turn loop's
        _settle_action_menu presses A/B) for the damage to actually apply, so a passive read saw the
        foe's HP still frozen at its pre-damage value and flagged EVERY landing hit as a MISS. That
        false whiff-spiral fired the breaker's pointless ace<->frail-bench switches and LOST winnable
        fights — the 10-shift Gary wall was a measurement artifact (log proof: 'frozen at 50 / 47 / 14'
        i.e. HP visibly DROPPING while each turn was called a whiff). The truth is only readable at the
        NEXT clean menu-up: compare the foe HP at the START of the PREVIOUS turn to the START of this
        turn (both post-text, damage-applied). A real miss leaves the SAME foe's HP unchanged across two
        consecutive turn-starts while a damaging move fired. Called at the top of the turn (menu up),
        BEFORE the whiff-breaker reads _whiff_streak."""
        if not self._whiff_prev_fired:
            return
        self._whiff_prev_fired = False                     # consume — classify each fired move once
        enemy = (state or {}).get("enemy") or {}
        hp_now, sp_now = enemy.get("hp"), enemy.get("species")
        if (self._whiff_prev_hp is None or hp_now is None or sp_now != self._whiff_prev_sp
                or hp_now != self._whiff_prev_hp):
            self._whiff_streak = 0                          # damage landed / KO'd / next mon -> accuracy fine
            return
        self._whiff_streak += 1                            # same foe, HP truly frozen a FULL turn -> a MISS
        self.log(f"   [engine] WHIFF: foe HP unchanged at {hp_now} across the full turn "
                 f"-> accuracy-debuff spiral (streak {self._whiff_streak}/{WHIFF_SPIRAL_AT})")

    def _party_levels(self):
        """Per-slot level snapshot (bounded raw read, +0x54 in the 100-byte party struct) —
        the in-drain level-up detect's baseline. [] on any read error (detect stays quiet)."""
        try:
            cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            return [self.b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
                    for s in range(min(cnt, 6))]
        except Exception:
            return []

    # ── one battle, start to finish ────────────────────────────────────────────
    def run(self, max_seconds=120):
        t0 = time.time()
        while time.time() - t0 < max_seconds and not st.in_battle(self.b):
            self._wait(1)
        if not st.in_battle(self.b):
            return "timeout"
        self._started = True
        global LEVELUP_EMITTED
        LEVELUP_EMITTED = False            # F-7(c) slice 2: the in-drain level-up beat re-arms
        self._lv0 = self._party_levels()   # per-slot baseline (any participant can gain)
        LAST_FOES_SEEN.clear()             # foes-seen ledger: fresh per battle (attach-time rival fix)
        self._win_emitted = False          # F-7(c): fresh engagement -> the certain-win beat re-arms
        self._catching = False             # (a prior catch on this agent must not mute the win beat)
        self._bigmoment_done = False       # Phase 2D: fire shiny/legendary recognition once per battle
        self._grind_switched = False       # GRIND SWITCH: protect-lead switch fires at most once per battle
        self._status_played = False        # STATUS STRATEGY: one status move per foe (reset per foe below)
        self._sleep_casts = 0              # SLEEP-LOCK whiff cap, reset per foe
        self._famine_tried = set()         # PP-FAMINE SWITCH: active species already offered a famine
        # switch this battle (one shot per species — a forced re-entry retries once, never churns).
        self._whiff_streak = 0             # WHIFF-SPIRAL: consecutive fired-but-no-damage (missed) turns
        self._whiff_recovering = None      # ace species we switched OUT to reset accuracy (switch back next)
        self._whiff_recoveries = 0         # bounded accuracy-resets this battle (never a switch-loop)
        self._whiff_prev_hp = None         # DEFERRED whiff-classify: foe HP at the PREVIOUS turn's menu-up
        self._whiff_prev_sp = None         # foe species at the previous turn's start (KO/next-mon detect)
        self._whiff_prev_fired = False     # a damaging move fired last turn -> classify it once this turn
        self._skip_streak = set()          # FIX 1: move slots that failed to fire this no-progress streak.
        # She rotates through her WHOLE moveset (never re-spams a 0-PP/disabled move), and only flees once
        # all are exhausted. CLEARED on any successful fire -> a working move is never permanently exiled
        # (the PoisonPowder-spam lesson: don't permanently bench a move, just rotate off it this streak).
        # Cleared the instant any move fires.
        self._prev = st.read_battle(self.b)
        # 2026-07-06 RE-ENTRY CORPSE GUARD (the Route-6 gauntlet livelock): a previous engagement can
        # abort mid-faint (budget/stuck), and travel re-enters the SAME battle with the foe already at
        # 0 HP — a FRESH agent never sees the 1->0 transition, so the faint flag never sets and the
        # engine move-picks into the "will you switch Pokémon?" prompt forever (weedle 0/38 ×51).
        # Joining mid-faint = the faint already happened: arm the flag so the post-faint drain
        # (force-B answers the prompt, fresh-enemy detect resumes the fight) owns it from turn one.
        if self._prev:
            if self._prev["enemy"]["hp"] == 0:
                self._enemy_fainted = True
                self.log("   [engine] re-entry: foe is already DOWN — draining the faint/switch chain")
            if self._prev["ours"]["hp"] == 0:
                self._we_fainted = True
                self.log("   [engine] re-entry: OUR active is already down — forced-switch chain owns it")
        self._reach_first_menu(t0, max_seconds)
        state = st.read_battle(self.b) or self._prev
        if state:
            foe = st.SPECIES_NAME.get(state["enemy"]["species"], "a wild pokemon")
            self.emit(f"a battle started against {foe}", beat=True)
            self._prev = state
            self._note_foe(state)

        # ── BIG-MOMENT RECOGNITION (Batch 3 Phase 2D): situational SIGNIFICANCE ───────────────────────
        # SHINY is the most clippable moment the game can produce — treating one as normal is a tragedy.
        # Detect it source-first off the CONFIRMED gEnemyParty PID/otId, FREAK OUT in character, and for
        # a WILD shiny DIVERT the whole battle to careful capture (weaken, never KO, throw balls) — the
        # existing catch_pokemon path. A trainer's shiny can't be caught -> freak-out only. Shininess is
        # ~1/8192, so this branch can NEVER fire in a normal battle / the regression fixtures (verified
        # all-False) — zero risk to the battle suites. LEGENDARY/rare gets a big beat too (id check).
        if state and not getattr(self, "_bigmoment_done", False):
            self._bigmoment_done = True
            esp = state["enemy"]["species"]
            foe = st.SPECIES_NAME.get(esp, "this Pokémon")
            if st.enemy_is_shiny(self.b):
                self.emit(f"WAIT — STOP. that {foe} is SHINY. chat, do you SEE this — you can play this "
                          f"game for five years and never see one. this is real, this is happening.",
                          beat=True, tier=3)
                if not self._is_trainer_battle():
                    self.log(f"   [engine] ✨✨ SHINY wild {foe} — diverting to CAREFUL CAPTURE "
                             f"(weaken+catch, never KO)")
                    res = self.catch_pokemon(max_seconds=max(150, max_seconds), weaken=True)
                    # SAFETY: if the catch failed (no balls / broke out / timed out) and the shiny is
                    # STILL on the field, FLEE rather than fight — never KO a shiny (the tragedy), and
                    # never leave the battle hanging (a wedge). A clean catch ends the battle -> return.
                    if res != "caught" and st.in_battle(self.b):
                        self.emit(f"I couldn't catch it ({res}) — I am NOT killing a shiny, I'm backing "
                                  f"out. that hurts.", beat=True, tier=3)
                        self.log(f"   [engine] shiny capture failed ({res}) — fleeing to avoid KOing it")
                        return self.flee(max_seconds=60)
                    return res
                self.log(f"   [engine] ✨ SHINY trainer {foe} — uncatchable, fighting (freak-out only)")
            elif esp in _LEGENDARY_SPECIES:
                self.emit(f"that's a {foe}. a LEGENDARY. okay — okay, do NOT mess this up.",
                          beat=True, tier=3)

        last_glob, stall = None, 0
        # victory_run7 (2026-07-07): gMoveToLearn is STALE across battles — snapshot at attach so
        # only a CHANGED nonzero value reads as a live level-up move prompt. The drain-armor
        # fingerprint lives on self so it survives the outer loop's re-entries into the drain.
        self._learn_seen = self.b.rd16(ram.GMOVE_TO_LEARN)
        self._drain_fp, self._drain_noprog = None, 0
        while time.time() - t0 < max_seconds:
            if not st.in_battle(self.b):
                return self._finish()
            # END SEQUENCE (checked FIRST, before settling): once a side has actually FAINTED
            # (a real alive->0 transition, not a stale battle-start read), the outcome is
            # decided; the rest is the victory/loss chain - faint anim -> "X fainted!" -> EXP
            # bar -> level-up -> exit. _advance_text walks it (waits out animations, A+B taps)
            # until the battle exits to overworld (in_battle -> False -> _finish). Never selects.
            if self._enemy_fainted or self._we_fainted:
                # POST-FAINT: drain the chain (faint anim -> "X fainted!" -> EXP -> level-up),
                # then DECIDE. A faint does NOT always end the battle: a TRAINER whose mon
                # faints SENDS THE NEXT ONE. So after each advance we check the enemy slot - if
                # a FRESH LIVE mon is on the field (full HP), it's a switch-in: reset the faint
                # flag and fall back into the normal fight loop. Otherwise keep draining toward
                # the exit (wild win / our loss / the trainer's LAST mon) until in_battle clears
                # -> _finish. (Before this, the engine assumed first-faint=won and never fought
                # the second mon -> trainer battles hung until timeout.)
                for _i in range(60):
                    if not st.in_battle(self.b):
                        break
                    cur = st.read_battle(self.b)
                    if cur:
                        self._emit_diffs(self._prev, cur); self._prev = cur
                    # F-7(c) slice 2 — LEVEL-UP EARLY BEAT: the party level byte flips the moment
                    # the level-up APPLIES, while "grew to LV. N!" is still on screen — but the
                    # beat used to fire only in play_live after the whole drain + ~4s LLM chain
                    # (deep into the overworld). Emit ONCE here so the chain runs DURING the rest
                    # of the drain and her line lands ON the jingle. One beat per battle;
                    # play_live dedups via LEVELUP_EMITTED.
                    if not LEVELUP_EMITTED and self._lv0:
                        _lvs = self._party_levels()
                        for _s in range(min(len(_lvs), len(self._lv0))):
                            if _lvs[_s] > self._lv0[_s]:
                                LEVELUP_EMITTED = True
                                _nm = st.SPECIES_NAME.get(st.read_party_species(self.b, _s),
                                                          "my Pokemon")
                                self.emit(f"my {_nm} just leveled up to level {_lvs[_s]}",
                                          beat=True, tier=2)
                                break
                    enemy = cur["enemy"] if cur else None
                    if (self._enemy_fainted and not self._we_fainted and enemy
                            and enemy["hp"] > 0 and enemy["hp"] == enemy["maxhp"]
                            and 1 <= enemy["species"] <= 411):
                        self._enemy_fainted = False        # next mon is out -> fight it
                        self._win_emitted = False          # F-7(c) defensive: a switch-in proves the win
                        #                                    read was premature — re-arm the real one
                        self._status_played = False         # NEW foe -> poison/sleep the next mon too (e.g.
                        self._sleep_casts = 0
                        #                                     Gary's Charmander, not just his lead)
                        self._prev = cur
                        self._note_foe(cur)
                        self.emit(f"the trainer sent out "
                                  f"{st.SPECIES_NAME.get(enemy['species'], 'another Pokemon')}",
                                  beat=True)
                        break
                    # STALE-ATTACH DISARM (koga_run7, the obj7 silent drain): _we_fainted can be armed
                    # at re-entry from the SAVE's display struct still holding the LAST battle's corpse
                    # (run3's Koga loss left Mankey at 0 in the struct; the first battle of the next
                    # process attached with "OUR active is already down" while Venusaur stood at full).
                    # If the LIVE read shows our active healthy and no mandatory party screen, we are
                    # NOT in a faint chain — disarm and fight normally (symmetric to the fresh-enemy
                    # detect above; a real faint keeps ours at 0 until the forced switch, so this can't
                    # fire mid-chain).
                    if (self._we_fainted and cur and cur["ours"]["hp"] > 0
                            and not self._party_screen()):
                        self._we_fainted = False
                        self._prev = cur
                        self.log("   [engine] stale-attach: our active is actually STANDING "
                                 "(display struct held the last battle's corpse) -> fighting normally")
                        break
                    # OUR mon fainted but we have a healthy reserve -> this is a FORCED SWITCH,
                    # not a loss: navigate the "Choose a POKéMON" menu and send the next mon, then
                    # fall back into the normal fight loop (roster-depth survival, now explicit).
                    if (self._we_fainted and st.in_battle(self.b) and cur
                            and cur["ours"]["hp"] == 0 and self._healthy_reserve_slot() is not None):
                        if self._force_switch():
                            self._we_fainted = False
                            self._prev = st.read_battle(self.b)
                            self.emit("that one's down - sending out my next Pokemon", beat=True)
                            break
                    # LAYER 7 (the gauntlet's terminal wedge, frame-diagnosed 2026-07-06): the
                    # "Choose a POKéMON" PARTY SCREEN inside this drain. force_b's A/B pair
                    # OSCILLATES on it forever (A selects a mon -> "Do what with X?", B cancels
                    # back — 240s verified repro, fight_01-81 all the same frame). Handle it
                    # deliberately: our active mon down = the screen is MANDATORY (send-next ->
                    # the proven _force_switch); otherwise it's the VOLUNTARY shift-prompt screen
                    # (it has a CANCEL) -> ONE clean B backs out ("No"), the trainer sends its
                    # next mon and the fresh-enemy detect above resumes the fight. Each loop
                    # iteration re-checks, so a B that only closed the sub-menu just B's again.
                    if self._party_screen():
                        if cur and cur["ours"]["hp"] == 0 and self._healthy_reserve_slot() is not None:
                            self.log("   [engine] party screen in drain: our mon is DOWN -> forced switch")
                            if self._force_switch():
                                self._we_fainted = False
                                self._prev = st.read_battle(self.b)
                                self.emit("that one's down - sending out my next Pokemon", beat=True)
                                break
                        else:
                            self.log("   [engine] party screen in drain: voluntary (shift prompt) -> single B out")
                            self._wait(10)
                            self.b.press("B", 2, 12, self.render, owner=self.owner)
                            self._wait(20)
                            if not self._party_screen():
                                # screen cleared -> the game may RE-SHOW "Will you switch POKéMON?";
                                # answer with a bare B (= No). Never A here: A re-picks Yes and the
                                # whole cycle restarts one level up. A stray B is a harmless advance.
                                self.b.press("B", 2, 12, self.render, owner=self.owner)
                                self._wait(20)
                        continue
                    # LEVEL-UP MOVE PROMPT (the 4-moves-known "Delete an older move?" flow; armed
                    # by a NEW gMoveToLearn value — victory_run7's L64 Venusaur sits one level from
                    # SolarBeam, so this WILL fire mid-E4). Handle it DELIBERATELY: B declines the
                    # delete, A confirms the stop — the B,A pair resolves the flow from ANY phase
                    # (B: Delete?->Stop?; A: Stop?->done) and both keys are plain text-advances on
                    # the surrounding msgboxes, so a stale/early read costs nothing. The proven
                    # quartet is load-bearing; choosing to REPLACE is a future roster-policy hook.
                    mv = self.b.rd16(ram.GMOVE_TO_LEARN)
                    if mv and mv != self._learn_seen and mv <= 354:
                        self._learn_seen = mv
                        mname = st.MOVE_NAMES.get(mv, f"move#{mv}")
                        self.log(f"   [engine] LEVEL-UP MOVE PROMPT: wants to learn {mname} over a "
                                 f"full moveset -> DECLINING deliberately (B=keep the set, A=confirm)")
                        self.emit(f"ooh — {mname} on offer. tempting, but I know my four. "
                                  f"we keep the set.", beat=True, tier=1)
                        for _ in range(12):
                            if (not st.in_battle(self.b) or self._party_screen()
                                    or self._white_box()):
                                break
                            self.b.press("B", 2, 14, self.render, owner=self.owner)
                            self._wait(16)
                            self.b.press("A", 2, 14, self.render, owner=self.owner)
                            self._wait(16)
                        continue
                    # DRAIN ARMOR (victory_run7's silent 7-minute spin): this drain was exempt from
                    # every anti-wedge guard — no stall count, no unresolved floor — so an
                    # unrecognized box left ONLY the 420s battle timeout, and the vehicle then
                    # re-entered the same wedge forever. Fingerprint progress; escalate LOUDLY:
                    # 40 no-progress advances -> snap a frame + B-first pairs (decline-class boxes);
                    # 80 -> one START tap (keyboard-class escape) and keep pairing; 120 -> return
                    # "stuck" with a frame so the caller's wedge machinery owns it. Any HP/state
                    # change resets — a normal victory chain is ~10-25 advances, never 40.
                    fp = ((cur["enemy"]["hp"], cur["ours"]["hp"]) if cur else None,
                          st.in_battle(self.b))
                    if fp == self._drain_fp:
                        self._drain_noprog += 1
                    else:
                        self._drain_fp, self._drain_noprog = fp, 0
                    if self._drain_noprog == 40:
                        self.log("   [engine] !! post-faint drain: 40 advances, zero progress -> "
                                 "switching to B-first decline pairs (wedge frame saved)")
                        self._debug_snap("drain40")
                    elif self._drain_noprog == 80:
                        self.log("   [engine] !! drain still frozen at 80 -> START tap "
                                 "(keyboard-class escape), pairing on")
                        self._debug_snap("drain80")
                        self.b.press("START", 2, 14, self.render, owner=self.owner)
                        self._wait(16)
                    elif self._drain_noprog >= 120:
                        self.log("   [engine] !! post-faint drain WEDGED (120 no-progress advances) "
                                 "-> LOUD stuck + frame; never the silent 420s spin again")
                        self._debug_snap("drain120")
                        return "stuck"
                    if self._drain_noprog >= 40:
                        self.b.press("B", 2, 14, self.render, owner=self.owner)
                        self._wait(16)
                        self.b.press("A", 2, 14, self.render, owner=self.owner)
                        self._wait(16)
                    else:
                        self._advance_text(force_b=True)  # faint -> EXP -> level-up -> defeat -> exit
                continue
            self._settle()                            # advance to a wait-point (narrates diffs)
            if not st.in_battle(self.b):
                return self._finish()
            # LAYER 8 (the abandoned-bag wedge): if the BAG is on screen, every menu byte is a stale
            # lie and every "move pick" lands on USE/CANCEL — close it deliberately before anything
            # else reads the screen. Covers re-entered battles that inherited an open bag too.
            if self._bag_screen():
                self.log("   [engine] BAG is open at the turn loop (abandoned item flow) -> B-closing it")
                self._close_bag_screen()
                continue
            glob = self._bstate()
            if glob != last_glob:                     # real progress -> reset the wedge guard
                last_glob, stall = glob, 0
            if self._white_box():                     # the action menu is up (white panel) ->
                state = st.read_battle(self.b)         # pick + commit a move, verify it lands
                self._note_foe(state)                  # foes-seen ledger (live turn read)
                self._classify_prev_whiff(state)       # race-free: judge last turn's move at this clean
                #                                        menu-up read (before the whiff-breaker acts below)
                # NS23 LOAD-SHARE (pre-heal, flag-gated default OFF): BEFORE spending a heal, if the worn
                # active is an SE attacker and a NEAR-FULL SE partner is on the bench, rotate to the fresh
                # body instead — spreads the gauntlet's attrition across two SE attackers AND conserves the
                # scarce heal items (the E4 Champion whiteout is a Full-Restore famine). Not during a
                # participation grind (the ace-protect switch owns that). Fail-safe: an unconfirmed switch
                # just falls through to the heal path; churn-safe by the near-full gate (see _load_share_slot).
                if (BATTLE_LOAD_SHARE and state and not PROTECT_LEAD_GRIND
                        and not (self._enemy_fainted or self._we_fainted)):
                    _ls = self._load_share_slot(state)
                    if _ls is not None:
                        self.log(f"   [engine] LOAD-SHARE: worn SE attacker -> fresh SE partner slot {_ls} "
                                 f"(spread damage, conserve heals)")
                        if self._switch_to_slot(_ls, state.get("ours", {}).get("species")) == "switched":
                            self._acted_once = True
                            stall = 0
                            self._unresolved_turns = 0
                            continue
                        self.log("   [engine] load-share switch did not confirm -> heal/fight (fail-safe)")
                # PART B: SURVIVAL INSTINCT FIRST — if a mon is crit-low/afflicted with a matching item,
                # offer the bag to the oracle. If she uses one, the turn is spent (skip move selection).
                # Any non-use falls through to the proven move path (fail-safe; never wedges).
                if state and not (self._enemy_fainted or self._we_fainted) and self._maybe_use_item(state):
                    self._acted_once = True
                    stall = 0
                    continue
                # PP-FAMINE SWITCH (2026-07-07, erika_run2 postmortem — the gym-gauntlet PP wall): the
                # active mon can be ALIVE but WINLESS — every damaging move at 0 PP after a long gauntlet,
                # leaving only status moves that can never KO (Fearow Growl/Leer'd a 60/60 Gloom until the
                # anti-wedge abort, ×12 futile battles, while Venusaur sat full-HP/full-PP on the bench).
                # That's not a matchup question, it's a hard constraint: if a bench mon still has damaging
                # PP, switching is the ONLY line that can win. Fires BEFORE grind/matchup logic (it
                # overrides both — a PP-dry ace can't grind either), once per active species per battle
                # (a forced re-entry of the same dry mon gets one more try, never a churn loop). Fail-safe:
                # an unconfirmed switch just fights on; no reserve -> log LOUD and let the anti-wedge
                # floor + the campaign's needs_heal gate own it.
                if (state and not (self._enemy_fainted or self._we_fainted)
                        and self._active_pp_famine(state)
                        and state.get("ours", {}).get("species") not in self._famine_tried):
                    # DIRTY-SCREEN GUARD (e4_run8 Agatha): the famine often trips the very turn an
                    # item flow ends, with the BAG still on screen — the switch nav then can't reach
                    # POKEMON ("cursor not on POKEMON") and the once-per-species try was BURNED, dooming
                    # the battle to status-spam -> all-dry -> Struggle livelock. Close the bag and let
                    # the next iteration retry famine with a clean action menu; consume the try only
                    # when the attempt starts from a real menu.
                    if self._bag_screen():
                        self.log("   [engine] PP FAMINE deferred: bag still open -> B-closing it first "
                                 "(try not consumed)")
                        self._close_bag_screen()
                        continue
                    self._famine_tried.add(state.get("ours", {}).get("species"))
                    _fs = self._pp_reserve_slot(state)
                    if _fs is not None:
                        self.log(f"   [engine] PP FAMINE: active has no damaging PP left -> switching to "
                                 f"party slot {_fs} (the only line that can still win)")
                        if self._switch_to_slot(_fs, state.get("ours", {}).get("species")) == "switched":
                            self.emit("I'm out of real moves on this one — switching to someone who can "
                                      "still hit.", beat=True, tier=2)
                            self._skip_streak.clear()
                            self._acted_once = True
                            stall = 0
                            self._unresolved_turns = 0
                            continue
                        self.log("   [engine] famine switch did not confirm -> fighting on (fail-safe)")
                    else:
                        self.log("   [engine] !! PP FAMINE: no reserve with damaging PP either — the whole "
                                 "party is dry (needs a Center; the campaign's readiness gate owns that)")
                # PARTICIPATION-XP GRIND SWITCH: while grinding the weak team (PROTECT_LEAD_GRIND), the weak
                # mon LEADS (eligible for XP) but would be one-shot — so turn 1, switch it to the ace. The
                # weak mon banks a share of XP and never takes a hit (benched before the enemy's turn); the
                # tanky ace KOs. Fires at most once/battle; fail-safe (a non-confirm just fights).
                if (GRIND_SWITCH_ENABLED and PROTECT_LEAD_GRIND and not self._grind_switched
                        and state and not (self._enemy_fainted or self._we_fainted)):
                    self._grind_switched = True            # one attempt/battle, whatever the result
                    ace = self._ace_reserve_slot()
                    # SELECTIVE SOLO (NS#26, gated POKEMON_SOLO_OVERLEVEL_GRIND — see _solo_overlevel_ok): if the
                    # weak lead SAFELY out-levels THIS foe it one-shots the wild -> DROP the ace-protect switch
                    # (ace=None -> falls through to the normal fight) so the weak mon SOLOS for the FULL kill XP
                    # (~2x the share the participation switch banks — the real bench-climb throttle). Suppresses
                    # a switch, never adds one; PROTECT_LEAD_GRIND stays True so the matchup switch stays off.
                    if ace is not None and self._solo_overlevel_ok(state):
                        self.log(f"   [engine] SOLO-OVERLEVEL: weak lead L{(state.get('ours') or {}).get('level')}"
                                 f" out-levels foe L{(state.get('enemy') or {}).get('level')} by "
                                 f">={SOLO_OVERLEVEL_MARGIN} -> soloing for FULL kill XP (no ace switch; "
                                 f"matchup switch stays suppressed)")
                        ace = None                         # skip the participation switch -> fight solo
                    # ALREADY-ACE GUARD (2026-07-05): after a mid-battle switch the ACTIVE mon is no longer
                    # gPlayerParty[0], so "is the lead weak?" must compare against the mon actually OUT.
                    # If the active species IS the ace's species, there's nothing to protect — switching
                    # would pull the tank OUT (the run-3 misfire: 'weak lead out' fired at an Ivysaur that
                    # was already fighting). Species match beats a level compare here: read_battle's 'ours'
                    # is the ground truth for who's out.
                    if ace is not None:
                        try:
                            ace_sp = st.read_party_species(self.b, ace)
                        except Exception:
                            ace_sp = None
                        if ace_sp is not None and state.get("ours", {}).get("species") == ace_sp:
                            self.log("   [engine] GRIND SWITCH: ace is ALREADY the active mon — no switch needed")
                            ace = None
                    if ace is not None:
                        self.log(f"   [engine] GRIND SWITCH: weak lead out -> switching to ace slot {ace} "
                                 f"(weak mon banks participation XP, ace does the fighting)")
                        if self._switch_to_slot(ace, state.get("ours", {}).get("species")) == "switched":
                            self._acted_once = True
                            stall = 0
                            self._unresolved_turns = 0
                            continue
                        self.log("   [engine] grind switch did not confirm -> fighting (fail-safe)")
                # B-1 — MATCHUP SWITCH (gated POKEMON_BATTLE_SWITCH, fail-safe): before swinging, if the
                # active mon is badly out-typed AND a better reserve exists, switch instead. Off by
                # default until the actuation is live-verified; a failed switch backs out and fights.
                # STRAND-ROOT FIX (2026-07-05 strike): NOT during a participation grind. PROTECT_LEAD_GRIND
                # just brought the tanky ACE in so the weak mon banks XP without taking a hit — the matchup
                # switch would immediately pull the ace back out (it reads Ivysaur as "out-typed" vs the
                # wild) and re-field the fragile mon, which then faints and STRANDS her (the observed
                # Route-4 (84,15) strand: GRIND SWITCH in, MATCHUP SWITCH straight back out). During a grind
                # the ace STAYS and tanks — no matchup churn.
                if (BATTLE_SWITCH_ENABLED and not PROTECT_LEAD_GRIND
                        and state and not (self._enemy_fainted or self._we_fainted)
                        and self._voluntary_switch(state) == "switched"):
                    self._acted_once = True
                    stall = 0
                    self._unresolved_turns = 0
                    continue
                # WHIFF-SPIRAL BREAKER (2026-07-10, night shift 9 — the S.S. Anne Gary root cause): an
                # accuracy-lowering foe (Sand-Attack/Smokescreen/Kinesis) debuffs the active mon until it
                # MISSES every swing — foe HP frozen while our PP drains -> famine -> a loss even at a
                # crushing level lead. Gen-3 resets stat stages on switch-out, so we switch the ace OUT
                # (accuracy resets) then BACK the next turn to swing fresh. _classify_prev_whiff counts misses;
                # here we execute the reset. Bounded per battle (WHIFF_MAX_RECOVERIES) so it never loops.
                if (WHIFF_BREAKER_ENABLED and state
                        and not (self._enemy_fainted or self._we_fainted)):
                    if self._whiff_recovering is not None:
                        # (a) mid-recovery — the ace is benched with reset accuracy; bring it back to swing.
                        if state.get("ours", {}).get("species") == self._whiff_recovering:
                            self._whiff_recovering = None   # already back (forced-switch) — fight fresh
                        else:
                            ace = self._slot_of_healthy_species(self._whiff_recovering)
                            if ace is not None:
                                self.log(f"   [engine] WHIFF RECOVERY: accuracy reset -> switching the ace "
                                         f"(sp {self._whiff_recovering}) back in to swing clean")
                                if self._switch_to_slot(ace, state.get("ours", {}).get("species")) == "switched":
                                    self._whiff_recovering = None
                                    self._whiff_streak = 0
                                    self._skip_streak.clear()
                                    self._acted_once = True; stall = 0; self._unresolved_turns = 0
                                    continue
                                self.log("   [engine] whiff recovery switch-back did not confirm -> fighting")
                                self._whiff_recovering = None
                            else:
                                self._whiff_recovering = None   # ace gone/active elsewhere — clear cleanly
                    elif (self._whiff_streak >= WHIFF_SPIRAL_AT
                          and self._whiff_recoveries < WHIFF_MAX_RECOVERIES):
                        # (b) trigger — the spiral is confirmed; switch the ace OUT to reset its accuracy.
                        _rs = self._any_healthy_reserve(state)
                        if _rs is not None:
                            _ace_sp = state.get("ours", {}).get("species")
                            self.log(f"   [engine] WHIFF-SPIRAL ({self._whiff_streak} misses): accuracy "
                                     f"debuffed -> switching OUT to slot {_rs} to reset it (recovery "
                                     f"{self._whiff_recoveries + 1}/{WHIFF_MAX_RECOVERIES})")
                            if self._switch_to_slot(_rs, _ace_sp) == "switched":
                                self.emit("it keeps making me miss — swapping out to shake off the "
                                          "accuracy drop.", beat=True, tier=2)
                                self._whiff_recovering = _ace_sp
                                self._whiff_recoveries += 1
                                self._whiff_streak = 0
                                self._acted_once = True; stall = 0; self._unresolved_turns = 0
                                continue
                            self.log("   [engine] whiff-spiral switch did not confirm -> fighting (fail-safe)")
                        else:
                            # ace is alone (frail bench dead) — no in-battle reset possible; stop
                            # re-logging every turn and fight on (war-must-advance; a miss still lands ~33%).
                            self.log("   [engine] WHIFF-SPIRAL but no healthy reserve to reset with -> fighting on")
                            self._whiff_streak = 0
                _enemy_hp_pre = (state or {}).get("enemy", {}).get("hp")
                _dmg_pp_pre = self._ours_dmg_pp(state) if state else 0
                res = self._select_and_verify(state) if state else "stuck"
                if res == "done":
                    self._acted_once = True
                    stall = 0
                    self._unresolved_turns = 0        # a real resolution clears the anti-wedge floor
                    # DEFERRED WHIFF STORE (see _classify_prev_whiff): remember this turn's CLEAN
                    # menu-up foe HP + whether a damaging move fired; the NEXT turn's menu-up read
                    # judges it race-free (PP is decremented at commit, so 'fired' is reliable now;
                    # foe HP is NOT reliable until the next turn's post-text read).
                    _cur = st.read_battle(self.b)
                    _dmg_pp_now = self._ours_dmg_pp(_cur) if _cur else _dmg_pp_pre
                    self._whiff_prev_fired = _dmg_pp_now < _dmg_pp_pre
                    self._whiff_prev_hp = _enemy_hp_pre
                    self._whiff_prev_sp = (state.get("enemy") or {}).get("species") if state else None
                else:
                    stall += 1                        # menu up but flaky -> settle re-checks, retry
                    # ANTI-WEDGE FLOOR — the run-existential one. `stall` resets on ANY screen change,
                    # so the 0-PP "no PP left!" flicker hides the wedge from it forever. This counter
                    # only clears on a real resolution above, so a depleted/blocked turn can't hide:
                    # past the threshold we ESCAPE rather than livelock (flee a wild fight = watchable
                    # self-preservation; a trainer can't be fled -> loud abort). 'no_usable_move' rides
                    # the same counter (so a one-frame PP misread can't trip a spurious flee).
                    self._unresolved_turns += 1
                    if BATTLE_FLEE_FLOOR and self._unresolved_turns >= UNRESOLVED_FLEE_AT:
                        if not self._is_trainer_battle():
                            self.log(f"   [engine] !! ANTI-WEDGE FLOOR: {self._unresolved_turns} "
                                     f"unresolved turns (last={res}) in a WILD battle -> FLEEING "
                                     f"(self-preservation, never a frozen session)")
                            self.emit("nothing's landing and I'm out of good moves — I'm backing out "
                                      "of this one.", beat=True, tier=2)
                            return self.flee(max_seconds=60)
                        _pp = None
                        try:
                            _pp = [m.get("pp") for m in (state or {}).get("ours", {}).get("moves", [])]
                        except Exception:
                            pass
                        self.log(f"   [engine] !! ANTI-WEDGE FLOOR: {self._unresolved_turns} unresolved "
                                 f"turns (last={res}) in a TRAINER battle -> can't flee; LOUD abort "
                                 f"[forensics: action_cursor={self.b.rd8(ram.GBATTLE_ACTION_CURSOR)} "
                                 f"white_box={self._white_box()} move_list={self._in_move_list()} "
                                 f"ours_pp={_pp}]")
                        self._debug_snap("antiwedge_trainer")
                        self.emit("I'm jammed up in here — can't get a move to land.", beat=False)
                        return "stuck"
            else:
                self._advance_text()                  # BLUE dialogue/animation box -> advance it
                stall += 1
            if stall >= 30:                           # genuine wedge -> loud abort, never silent
                self.log("   [engine] !! battle wedged - no progress over 30 attempts, aborting loudly")
                self.emit("okay I'm properly stuck, the menu's glitched", beat=False)
                return "stuck"
        return "timeout"

    def _finish(self):
        prev = self._prev or {}
        ours = prev.get("ours", {})
        if self._enemy_fainted or (prev.get("enemy", {}).get("hp", 1) == 0):
            # F-7(c): the certain-win beat already voiced this win AT THE FAINT (the drain +
            # LLM chain aligned her reaction with the victory screen) — never voice it twice.
            if not self._win_emitted:
                self.emit("you won the battle", beat=True)
            return "win"
        if ours.get("hp", 1) == 0:
            self.emit("you lost - your Pokemon fainted", beat=True)
            return "loss"
        self.emit("the battle ended", beat=False)
        return "ended"

    def _emit_diffs(self, prev, cur):
        if not prev:
            return
        pe, ce = prev["enemy"], cur["enemy"]
        po, co = prev["ours"], cur["ours"]
        # narrate the move from the OBSERVED hit (ground truth), not per button-press,
        # so it fires exactly once per landed move - never spammy.
        if ce["hp"] < pe["hp"] and ce["hp"] > 0:
            desc = getattr(self, "_last_desc", "an attack")
            self.emit(f"used {desc}", beat=(getattr(self, "_last_eff", 1.0) >= 2))
        if ce["hp"] == 0 and pe["hp"] > 0:
            self._enemy_fainted = True
            # F-7(c) SPECULATIVE PREFETCH (the certain-win early beat): when THIS faint leaves no
            # live mon anywhere in gEnemyParty, the battle is DECIDED at this frame — but the win
            # line used to fire only after the whole victory drain (faint anim → EXP → level-up,
            # 5-15s at human pace) PLUS the ~4s LLM chain, so "we won!" landed ~10s into the
            # overworld. Emit ONE merged win beat NOW instead: the generation chain runs DURING
            # the drain and her voice lands on the victory screen. One line, not two — a win emit
            # microseconds after "took it down" would be floor-dropped by the voice gate. Guards:
            # our mon alive (a double-faint is a loss path), never in the catch flow (KOing a
            # catch target is a failure), certain only when zero live foes remain.
            if (not self._win_emitted and not self._catching and not self._we_fainted
                    and cur["ours"]["hp"] > 0 and self._enemy_live_remaining() == 0):
                self._win_emitted = True
                self.emit(f"the enemy's {st.SPECIES_NAME.get(ce['species'], 'Pokemon')} went down — "
                          f"that's the battle, you won", beat=True)
                return
            # BATCH 5 PHASE 3 — mark the SIDE so she never narrates her own WIN as a loss. The bare
            # "{species} fainted" read as HER mon dying (she mourned a Nidoran she'd just KO'd). gBattleMons[1]
            # is the ENEMY, so this faint is always a victory. (Avoid the substrings 'knocked out'/'you lost'
            # — pokemon_voice.classify tiers those as a T3 LOSS; "took it down" stays the correct T1.)
            self.emit(f"the enemy's {st.SPECIES_NAME.get(ce['species'], 'Pokemon')} fainted — you took it down",
                      beat=True)
        if co["hp"] == 0 and po["hp"] > 0:
            self._we_fainted = True
            self.emit("your Pokemon fainted", beat=True)
        elif co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit("you took a big hit", beat=True)
        elif co["maxhp"] and co["hp"] / co["maxhp"] < 0.25 and po["hp"] / max(po["maxhp"], 1) >= 0.25:
            self.emit("low HP - this is getting tense", beat=True)
