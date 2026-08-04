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
# Diglett's Cave floors (campaign._PLACE_NAMES / frlg_connections) — common keepers; never KO on sight
# when unowned (2026-08-02 Diglett chalk: Flash cave-cross fought every Diglett after Jonny said catch).
_DIGLETT_CAVE_MAPS = frozenset({(1, 36), (1, 37), (1, 38)})
_DIGLETT_LINE = frozenset({50, 51})  # Diglett, Dugtrio
# Creator "catch that!" latch path (kira/bot.py -> states/*/creator_order.json).
_CREATOR_ORDER_TTL_S = float(os.getenv("POKEMON_CREATOR_ORDER_TTL_S", "1800"))

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
# gMain.callback2 TRUTH for the in-battle BAG/PARTY sub-screens (2026-08-04, the Revive
# insta-click: the aim block only ran on laps where the PIXEL party classifier fired, so on
# frozen frames the bare walk-A confirmed the target screen's HOME cursor — the alive lead —
# "It won't have any effect." forever). The game physically SWITCHES gMain.callback2 to these
# menu loops while a sub-screen owns input (same ground truth as ram.battle_cb2_dead), so a
# RAM read beats any frozen frame. pret pokefirered.sym rev0 — CB2_Overworld 0x080565B4 in the
# same map matches the live-verified ram._CB2_OVERWORLD. Stored with the thumb bit set.
_CB2_PARTY_MENU = {0x0811EBA0 | 1,   # CB2_UpdatePartyMenu  (steady-state party/target screen)
                   0x0811EBD0 | 1,   # CB2_InitPartyMenu    (opening fade)
                   0x08124C8C | 1}   # CB2_ShowPartyMenuForItemUse (bag USE -> target transition)
_CB2_BAG_MENU = {0x08107EE0 | 1,     # CB2_BagMenuRun       (steady-state bag list)
                 0x08107F10 | 1,     # CB2_OpenBagMenu
                 0x08107ECC | 1,     # CB2_BagMenuFromBattle
                 0x08124D90 | 1}     # CB2_ReturnToBagMenu  (target screen -> bag)
_ITEMS_POCKET_OFF = 0x0310   # SaveBlock1 Items pocket (potions + status cures live here), 42 slots
# Gen-3 item ids for the in-battle instinct (CANDIDATES; the use is self-verified by the item count
# dropping, so a wrong id simply doesn't fire -> 'failed' -> keep fighting, never a wrong action).
_HEAL_ITEMS_PREF = (19, 20, 21, 22, 13)   # Full Restore, Max, Hyper, Super, Potion (strongest usable first)
_REVIVE_ITEMS_PREF = (25, 24)             # Max Revive, Revive
# CHEAPEST-FIRST ether order (2026-08-03 potion-economics pass): an Ether (10 PP, one move)
# almost always un-famines the workhorse slot — burning the Max Elixir first was hoard-in-reverse.
_ETHER_ITEMS_PREF = (34, 36, 35, 37)      # Ether, Elixir, Max Ether, Max Elixir
# How much each potion tier heals (Gen 3): the right-sized-potion picker's fact table.
_POTION_HEALS = {13: 20, 22: 50, 21: 200, 20: 9999, 19: 9999}
ITEM_QTY_NAMES = {13: "a Potion", 22: "a Super Potion", 21: "a Hyper Potion", 20: "a Max Potion",
                  19: "a Full Restore", 14: "an Antidote", 15: "a Burn Heal", 16: "an Ice Heal",
                  17: "an Awakening", 18: "a Parlyz Heal", 23: "a Full Heal"}
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
# Was 3 — on LIVE stream menu thrash, 3 unresolved turns ≈ minutes of scrolling. Bail at 2.
UNRESOLVED_FLEE_AT = int(os.getenv("POKEMON_UNRESOLVED_FLEE_AT", "2"))
# Hard wall-clock escape when menu understanding is wedged (seconds since LAST real progress —
# PP drop / HP change / faint / item consume). Was "45s from battle start" which false-fired
# mid-Gary and killed the stream with "menus are glitched" on a progressing fight (2026-08-02).
BATTLE_MENU_WEDGE_S = float(os.getenv("POKEMON_BATTLE_MENU_WEDGE_S", "60"))
# Force-switch wall-clock: party scroll theater must not eat 60–90s of stream (2026-08-02 docks).
FSWITCH_BUDGET_S = float(os.getenv("POKEMON_FSWITCH_BUDGET_S", "5"))
# PP-FAMINE SWITCH RETRIES (2026-07-31, Jonny stream debrief — the 10-minute Teleport-Abra fight):
# the famine switch used to be ONE-SHOT per species per battle, and the try was consumed even when
# the switch nav FAILED to confirm — so a Teleport-only Abra whose single try misfired was doomed
# to war-must-advance its failing move for the rest of the battle (and every 180s travel re-entry).
# Bounded retries keep the anti-churn intent (never an infinite switch loop) while making one flaky
# menu nav non-fatal.
FAMINE_SWITCH_TRIES = int(os.getenv("POKEMON_FAMINE_SWITCH_TRIES", "3"))
# FUTILITY BREAKER (2026-08-03 09:07, the parked-on-Tackle photo evidence): total fruitless
# move-list confirms per battle before the engine stops trusting the move list entirely and
# bench-switches. This is the slot-agnostic floor UNDER the per-slot refusal ledger: when the
# cursor readback and the drawn cursor disagree, refusals get tallied on the wrong slots and
# the ledger never converges — this counter converges anyway, because it counts EVENTS not slots.
FUTILE_AMOVE_MAX = int(os.getenv("POKEMON_FUTILE_AMOVE_MAX", "4"))
# NS#12 — HEAL-CONSUME-FAILED LATCH (the Route-10/Rock-Tunnel bag-USE/CANCEL livelock). An in-battle
# heal can open the bag, reach "ITEM is selected -> USE/CANCEL", and FAIL to consume (count never drops)
# — _bag_screen() doesn't fingerprint that sub-box, so the turn-top close is bypassed and every "pick a
# move" press lands in the still-open bag -> unresolved -> anti-wedge abort -> travel RE-ENTERS the same
# unfleeable trainer battle -> infinite livelock (a weak participation-fielded mon frozen vs an Onix the
# benched ace one-shots). This latch (mirrors _famine_tried) suppresses the RE-OFFER once a heal PROVES
# it won't consume this battle, so the mon fights/faints -> the ace comes in -> the battle resolves. Only
# fires after a proven-failed use (a 'used' heal never latches), so E4/gym heals are untouched. The deep
# bag-USE/CANCEL actuation fix is separate (needs attended frame-grabs); this breaks the livelock safely.
HEAL_FAIL_LATCH = os.getenv("POKEMON_HEAL_FAIL_LATCH", "1") == "1"
# B-1 — IN-BATTLE MATCHUP SWITCHING (E4-critical). The matchup MATH is offline-verified, but the
# party-menu ACTUATION (cursor nav on a long-running libmgba core) is UNVERIFIED — the standing
# menu-nav lesson. So it's GATED OFF by default until a live control passes (arm POKEMON_BATTLE_SWITCH=1
# with Jonny watching). FAIL-SAFE regardless: if a switch doesn't confirm, she backs out and fights —
# never wedges. When off, she still AVOIDS ineffective moves (that path is on + verified).
# ARMED 2026-07-05: the in-battle switch is now VERIFIED (recon_switch3.py + direct _switch_to_slot test on
# the canon 3-mon fixture — SWITCHED ivysaur->spearow). The wedge was a WRONG-ADDRESS derivation (the old
# PARTY_CURSOR=0x2020777 was a shadow byte); the real nav is BLIND DOWN*(slot+1) like the working
# _force_switch (live cursor is a heap struct). Fail-safe B-out on any miss = never wedges, so default-on is safe.
# 2026-08-03 NUCLEAR (morning): DEFAULT OFF after the Blastoise↔POKEMON thrash on stream.
# 2026-08-03 RE-ARMED (same day, 08:28 live): the thrash's REAL root is now fixed at the source —
# the blind-A/cursor-parking disease (every settled action menu re-homes to FIGHT; no escape path
# presses a blind A; the party-thrash guard hard-bans voluntary POKEMON after 3 sightings). With
# the ban still on, a PP-dry mon had NO winning line: Jonny watched her re-fire a refused Tackle
# forever because the famine/must-leave switches — the exact rescue for that state — were gated
# off. The switch nav itself is recon-verified (recon_switch3.py, 2026-07-05) and fail-safe
# (non-confirm -> B-out -> fight). Killing the rescue was the wrong nuke; default back ON.
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
# 2026-08-03 NUCLEAR: DEFAULT OFF with BATTLE_SWITCH — same party-menu thrash class.
GRIND_SWITCH_ENABLED = os.getenv("POKEMON_GRIND_SWITCH", "0") == "1"
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
SOLO_OVERLEVEL_GRIND = os.getenv("POKEMON_SOLO_OVERLEVEL_GRIND", "1") == "1"   # DEFAULT-ON 2026-07-12 NS#26:
#   look-ahead-verified — the whole bench evened 28->34 (~2x the participation-share baseline, floor 34 even vs
#   31 uneven at ~1600s), 0 faints, 0 grind-switch (fully replaced), 0 matchup-churn, marches on after the
#   milestone (no park). Env-revert POKEMON_SOLO_OVERLEVEL_GRIND=0. Structurally level-relative (generalises
#   across gym targets); multi-gym breadth (targets 45/55) is the next confirmation.
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
        self._switch_fail_n = 0        # voluntary matchup-switch fails this battle (latch at 1)

    # ── input (owner-attributed) ───────────────────────────────────────────────
    def _tap(self, key):
        self.b.press(key, self.hold, self.hold, self.render, owner=self.owner)

    def _wait(self, frames):
        for _ in range(frames):
            self.b.run_frame(); self.render()

    def _is_trainer_battle(self):
        """BATTLE_TYPE_TRAINER (0x08). Valid in-battle. Wild = can flee, trainer = can't."""
        return bool(self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08)

    def _creator_catch_order_path(self):
        """First live creator_order.json under states/campaign or states/kira (same dirs the bot writes)."""
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "states")
        for sub in ("campaign", "kira"):
            p = os.path.join(root, sub, "creator_order.json")
            if os.path.isfile(p):
                return p
        return None

    def _peek_creator_catch_order(self):
        """True when Jonny's voice latched order=catch_now and the TTL hasn't expired."""
        import json as _j
        path = self._creator_catch_order_path()
        if not path:
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = _j.load(f) or {}
            if data.get("order") != "catch_now":
                return False
            ts = float(data.get("ts") or 0)
            if not ts or time.time() - ts > _CREATOR_ORDER_TTL_S:
                return False
            return True
        except Exception:
            return False

    def _clear_creator_catch_order(self):
        """Release catch_now after a committed catch attempt settles (caught or abandoned)."""
        import json as _j
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "states")
        for sub in ("campaign", "kira"):
            p = os.path.join(root, sub, "creator_order.json")
            try:
                if not os.path.isfile(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    data = _j.load(f) or {}
                if data.get("order") == "catch_now":
                    os.remove(p)
                    self.log("   [engine] creator catch_now order CLEARED (fulfilled/released)")
            except Exception as e:
                self.log(f"   [engine] creator-catch clear skipped ({sub}): {e}")

    def _party_owns_species(self, species_id):
        """Cheap party-only owned check (box scan lives on Campaign; party covers the live Diglett case)."""
        try:
            n = min(self.b.rd8(ram.GPLAYER_PARTY_CNT), 6)
            for i in range(n):
                if st.read_party_species(self.b, i) == species_id:
                    return True
        except Exception:
            pass
        return False

    def _dex_owns_species(self, species_id):
        """Pokédex owned bit — true once she's EVER caught this species (party OR box)."""
        try:
            return ram.pokedex_owns(self.b, species_id) is True
        except Exception:
            return self._party_owns_species(species_id)

    def _resolve_open_battle(self, max_seconds=120):
        """NEVER return while in_battle is still True (Arena Trap / failed flee leaves a live fight —
        travel then re-enters the same battle forever → RUN spam loop). Fight it out."""
        if not st.in_battle(self.b):
            return "win"
        self.log("   [engine] battle still open after catch/flee fail — FIGHTING to clear "
                 "(Arena Trap / Can't escape class)")
        self._skip_catch_divert = True
        try:
            return self.run(max_seconds=max_seconds)
        finally:
            self._skip_catch_divert = False

    def _divert_wild_catch(self, reason, foe_name, max_seconds):
        """Shared careful-capture divert (shiny / legendary / creator catch_now / Diglett keeper).

        2026-08-02 LIVE: catch_now + ZERO balls used to fall through to `run()` (= FIGHT) and
        Blastoise one-shot the Route-12 Snorlax Jonny ordered caught. Creator / shiny / legendary
        NEVER fight-clear on empty balls — flee, keep the order live, Mart first."""
        _never_ko = reason in ("shiny", "legendary", "creator_catch_now")
        try:
            _is_mewtwo = (reason == "legendary"
                          and (st.read_battle(self.b) or {}).get("enemy", {}).get("species") == 150)
        except Exception:
            _is_mewtwo = False
        if self._ball_count() <= 0 and not (_is_mewtwo and self._ball_qty(self._BALL_MASTER) > 0):
            if _never_ko:
                self.log(f"   [engine] WILD CATCH DIVERT ({reason}) — ZERO balls; "
                         f"FLEEING (NOT fighting — that KOs the catch target)")
                self.emit(f"I've got ZERO Poké Balls — I am NOT killing this {foe_name}. "
                          f"backing out. Mart first, then we catch.", beat=True, tier=3)
                # KEEP catch_now live — clearing it made the next wake/fight a free KO.
                fled = self.flee(max_seconds=60)
                if not st.in_battle(self.b):
                    return "no_balls"
                if self._foe_blocks_flee():
                    # Diglett Arena Trap + 0 balls + catch order: can't flee, can't catch.
                    # Fighting is the only exit — LOUD. (Snorlax is fleeable; never hits here.)
                    self.log("   [engine] catch_now + 0 balls + Arena Trap — must fight clear (LOUD)")
                    self._skip_catch_divert = True
                    try:
                        return self.run(max_seconds=max(120, max_seconds))
                    finally:
                        self._skip_catch_divert = False
                return "no_balls"
            self.log(f"   [engine] WILD CATCH DIVERT skipped ({reason}) — no balls; fighting instead")
            self._skip_catch_divert = True
            try:
                return self.run(max_seconds=max(120, max_seconds))
            finally:
                self._skip_catch_divert = False
        self.log(f"   [engine] WILD CATCH DIVERT ({reason}) — {foe_name}: weaken+balls, never KO")
        # BALL TIER doctrine (2026-08-04): a legendary/shiny gets the STRONGEST ball first
        # (catch rate 3 with a Poké Ball is theater), and Mewtwo — alone — may spend the
        # Silph Co. Master Ball (the classic move; any other target would waste it).
        _pref = "best" if reason in ("legendary", "shiny", "creator_catch_now") else "cheap"
        _master = False
        if reason == "legendary":
            try:
                _master = (st.read_battle(self.b) or {}).get("enemy", {}).get("species") == 150
            except Exception:
                _master = False
        res = self.catch_pokemon(max_seconds=max(150, max_seconds), weaken=True,
                                 ball_pref=_pref, allow_master=_master)
        if reason == "creator_catch_now" and res == "caught":
            self._clear_creator_catch_order()
        elif reason == "creator_catch_now" and res in ("fainted",):
            # KO'd the catch target — LOUD failure, clear so she doesn't re-latch forever.
            self.log("   [engine] !! catch_now FAILED — target fainted (KO). order cleared.")
            self.emit(f"no — I knocked out the {foe_name}. that was the catch order. I'm an idiot.",
                      beat=True, tier=3)
            self._clear_creator_catch_order()
        elif reason == "creator_catch_now" and res in ("no_balls", "cant_weaken", "fled", "stuck"):
            # Keep order — she still owes the catch after Mart / retry.
            self.log(f"   [engine] catch_now unresolved ({res}) — LAW order KEPT for retry")
        if res == "caught" or not st.in_battle(self.b):
            return res
        if _never_ko:
            self.emit(f"I couldn't catch it ({res}) — I am NOT killing a {reason}, I'm backing out.",
                      beat=True, tier=3)
            self.log(f"   [engine] {reason} capture failed ({res}) — fleeing to avoid KOing it")
            fled = self.flee(max_seconds=60)
            if not st.in_battle(self.b):
                return fled
            if reason == "creator_catch_now" and not self._foe_blocks_flee():
                return "no_balls"
            self.emit("can't run — finishing the fight carefully.", beat=True, tier=2)
        return self._resolve_open_battle(max_seconds=max(120, max_seconds))

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

    def _decided_win(self):
        """True when the fight is already OVER for us — last foe down, we still standing.
        Used to forbid stuck/timeout aborts that make travel RE-ENTER the victory drain
        (2026-08-02 Rock Tunnel chalk: win beat → rewind into last seconds of the fight)."""
        try:
            if self._we_fainted:
                return False
            if not (self._enemy_fainted or self._win_emitted):
                return False
            return self._enemy_live_remaining() == 0
        except Exception:
            return False

    def _drain_decided_win(self, grace_s=120):
        """Keep mashing the victory chain until the battle exits. NEVER return stuck/timeout
        while the win is decided — those aborts are what travel re-enters as a 'fight reset'."""
        t0 = time.time()
        n = 0
        self.log(f"   [engine] DECIDED-WIN DRAIN: last foe down — finishing victory chain "
                 f"(grace {grace_s}s; will NOT abort mid-win for re-entry)")
        while time.time() - t0 < grace_s:
            if not st.in_battle(self.b):
                return self._finish()
            n += 1
            # Escalate clears the same way as drain armor, but never bail to stuck.
            if n == 40:
                self.log("   [engine] decided-win drain: 40 advances — B-first pairs")
                self._debug_snap("decided_win40")
            elif n == 80:
                self.log("   [engine] decided-win drain: 80 advances — START tap")
                self._debug_snap("decided_win80")
                self.b.press("START", 2, 14, self.render, owner=self.owner)
                self._wait(16)
            elif n > 0 and n % 120 == 0:
                self.log(f"   [engine] decided-win drain still live at {n} advances — "
                         f"keeping on (NOT stuck; win is decided)")
                self._debug_snap(f"decided_win{n}")
                self.b.press("START", 2, 14, self.render, owner=self.owner)
                self._wait(16)
            if n >= 40:
                self.b.press("B", 2, 14, self.render, owner=self.owner)
                self._wait(16)
                self.b.press("A", 2, 14, self.render, owner=self.owner)
                self._wait(16)
            else:
                self._advance_text(force_b=True)
        # Grace spent but still in battle — one last hard mash, then finish as win if cleared.
        self.log("   [engine] !! decided-win grace spent — hard mash then exit (LOUD)")
        for _ in range(60):
            if not st.in_battle(self.b):
                return self._finish()
            self.b.press("B", 2, 10, self.render, owner=self.owner)
            self._wait(8)
            self.b.press("A", 2, 10, self.render, owner=self.owner)
            self._wait(8)
        if not st.in_battle(self.b):
            return self._finish()
        # Still open: report win anyway so travel does NOT treat this as stuck and re-enter.
        # The next tick's _wait_overworld / a fresh attach will keep draining; aborting as
        # stuck is what caused the visible fight-reset loop on stream.
        self.log("   [engine] !! decided-win still in_battle after hard mash — returning win "
                 f"(refuse re-entry loop; in_battle={st.in_battle(self.b)})")
        if not self._win_emitted:
            self._win_emitted = True
        return "win"

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

    def _foe_blocks_flee(self):
        """True when the wild foe's ability makes RUN impossible (Diglett/Dugtrio = Arena Trap).
        Fleeing then livelocks on 'Can't escape' + RUN spam (2026-08-02 Diglett chalk)."""
        try:
            esp = st.read_enemy_species(self.b, 0)
            return esp in _DIGLETT_LINE
        except Exception:
            return False

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
        # ARENA TRAP (Diglett/Dugtrio): RUN never works — fight clear immediately. Do NOT spam RUN
        # for `max_seconds` (live stream chalk: she narrated "running" and stuck in Diglett forever).
        if self._foe_blocks_flee():
            foe = st.SPECIES_NAME.get(st.read_enemy_species(self.b, 0), "Diglett")
            self.log(f"   [engine] flee: {foe} has Arena Trap — Can't escape. FIGHTING to clear "
                     f"(not RUN-spamming)")
            self.emit(f"Arena Trap — you can't run from {foe}. finishing the fight.",
                      beat=True, tier=2)
            self._skip_catch_divert = True
            try:
                return self.run(max_seconds=max(120, max_seconds))
            finally:
                self._skip_catch_divert = False
        for _ in range(3):                            # ensure the ACTION menu, not the move list:
            if not self._in_move_list():              # _white_box can't tell them apart, so RUN nav
                break                                 # from an open move-list fires a move + never
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)  # escapes (flee 'stuck'
            self._wait(10)                            # loop). Same class as the catch bag-nav bug.
        cant_escape = 0
        for _ in range(40):
            if not st.in_battle(self.b):
                return "fled"
            self._settle()
            if not st.in_battle(self.b):
                return "fled"
            # Ability / Mean Look class: if RUN keeps failing, stop spamming and fight clear.
            if self._foe_blocks_flee() or cant_escape >= 2:
                self.log(f"   [engine] flee: Can't escape (streak={cant_escape}) — FIGHTING clear")
                self.emit("can't run from this one — fighting out.", beat=True, tier=2)
                self._skip_catch_divert = True
                try:
                    return self.run(max_seconds=max(120, max_seconds))
                finally:
                    self._skip_catch_divert = False
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] == 0:
                return "loss"
            if self._white_box() and self._goto_run():
                self._tap("RIGHT"); self._tap("DOWN") # engage (eaten-press; RUN stays at corner)
                self._tap("A"); self._wait(20)        # confirm RUN -> "got away safely" / retry
                if st.in_battle(self.b):
                    cant_escape += 1                  # still in fight = Can't escape / failed flee
            else:
                self._advance_text()                  # advance the escape/"can't escape" message
        if st.in_battle(self.b):
            self.log("   [engine] flee: still in battle after RUN budget — FIGHTING clear (LOUD)")
            return self._resolve_open_battle(max_seconds=max(120, max_seconds))
        return "fled"

    # ── autonomous CATCH (real bag nav; the phantom-A bug that made this impossible was
    # fixed 2026-06-25 — see [[pokemon-battle-menu-nav-cracked]]). Flow, screenshot- and
    # control-verified (party N->N+1) on forest_battle.state: action menu opens on FIGHT each
    # turn -> RIGHT = BAG -> A opens the bag (lands on the Poké Balls pocket, cursor on the
    # ball) -> A selects -> "POKé BALL is selected. USE/CANCEL" (cursor on USE) -> A throws.
    # Then advance the catch sequence (B dismisses the "give a nickname?" Yes/No). We SETTLE
    # after the bag-open fade (acting mid-transition = eaten, the same quirk as the move list).
    # BALL TIERS (2026-08-04, Jonny: 'catching mew or mewtwo as a final endgame project ...
    # all cool legendaries'): everything here was hard-coded to plain Poké Ball (id 4) — a
    # catch-rate-3 legendary with Poké Balls is <1% a throw, a guaranteed pocket-drain wedge.
    # Gen-3 ball item ids: 1 Master, 2 Ultra, 3 Great, 4 Poké, 5 Safari, 6-12 specials.
    _BALL_MASTER, _BALL_ULTRA, _BALL_GREAT, _BALL_POKE = 1, 2, 3, 4
    _SPENDABLE_BALLS = (2, 3, 4, 6, 7, 8, 9, 10, 11, 12)   # throwable freely (Master/Safari out)
    _BALL_ORDER_CHEAP = (4, 3, 2, 12, 11, 10, 9, 8, 7, 6)  # dex-push trash: weakest first
    _BALL_ORDER_BEST = (2, 3, 12, 11, 10, 9, 8, 7, 6, 4)   # legendary/shiny: strongest first

    def _balls_pocket(self):
        """The bag's Poké Balls pocket as [(item_id, qty)] rows in DISPLAY order. Ids are plain;
        the QUANTITY is XOR-encrypted with the SaveBlock2 security key (SaveBlock2+0xF20 low16)."""
        out = []
        try:
            sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
            key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
            for i in range(13):                          # FRLG balls pocket = 13 slots
                iid = self.b.rd16(sb1 + 0x430 + i * 4)
                if iid == 0:
                    break
                out.append((iid, self.b.rd16(sb1 + 0x430 + i * 4 + 2) ^ key))
        except Exception:
            pass
        return out

    def _ball_qty(self, iid):
        return next((q for i, q in self._balls_pocket() if i == iid), 0)

    def _ball_count(self):
        """Throwable-ball total across the SPENDABLE tiers (Ultra/Great/Poké + specials). The
        Master Ball is deliberately EXCLUDED — it is Mewtwo's, never a wild-catch statistic."""
        return sum(q for iid, q in self._balls_pocket() if iid in self._SPENDABLE_BALLS)

    def _pick_ball(self, pref="cheap", allow_master=False):
        """(item_id, display_row) of the ball to throw, or (None, None). 'cheap' spends weakest
        first (dex push); 'best' spends strongest first (legendary/shiny). Master only when
        explicitly allowed (the Mewtwo seat)."""
        rows = self._balls_pocket()
        if allow_master:
            r = next((n for n, (i, q) in enumerate(rows) if i == self._BALL_MASTER and q > 0), None)
            if r is not None:
                return self._BALL_MASTER, r
        order = self._BALL_ORDER_BEST if pref == "best" else self._BALL_ORDER_CHEAP
        for want in order:
            r = next((n for n, (i, q) in enumerate(rows) if i == want and q > 0), None)
            if r is not None:
                return want, r
        return None, None

    def throw_ball(self, max_seconds=45, pref="cheap", allow_master=False):
        """Throw a ball at a WILD foe via real menu nav. Returns 'caught' (party+1),
        'broke_free' (battle continued/ended w/o catch), 'trainer' (can't catch), 'no_balls',
        or 'stuck'. Assumes a fresh/settled action menu (turn start). Control-proven party+1.
        `pref`/`allow_master`: the BALL TIER doctrine (2026-08-04) — 'cheap' spends weakest
        first, 'best' strongest first, Master only when explicitly allowed (Mewtwo)."""
        t0 = time.time()
        if self._is_trainer_battle():
            return "trainer"
        ball_id, ball_row = self._pick_ball(pref=pref, allow_master=allow_master)
        if ball_id is None:
            self.log("   [engine] throw_ball: no throwable ball in the bag")
            return "no_balls"
        if not self._white_box():
            self._reach_first_menu(t0, max_seconds)
        self._settle()
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        # baseline BEFORE the bag opens (throw-verify gate) — ALL tiers incl. Master, so a
        # Master throw registers as _thrown() too
        balls_at_start = sum(q for _i, q in self._balls_pocket())
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
            return (sum(q for _i, q in self._balls_pocket()) < balls_at_start
                    or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0
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
        # BALL-TIER ROW WALK (2026-08-04): the old 'UP = top of pocket' threw whatever was
        # row 0 — with Ultras stocked that could be the wrong tier either way. Blind clamp
        # to the top (the list clamps, eaten taps are harmless), then DOWN to the chosen row.
        n_rows = max(1, len(self._balls_pocket()))
        for _ in range(n_rows + 2):
            self._tap("UP"); self._wait(6)
        for _ in range(int(ball_row)):
            self._tap("DOWN"); self._wait(8)
        if ball_row or n_rows > 1:
            self.log(f"   [engine] throw_ball: aiming row {ball_row} = item {ball_id} "
                     f"({'MASTER' if ball_id == 1 else 'ultra' if ball_id == 2 else 'great' if ball_id == 3 else 'poké/other'} ball, pref={pref})")
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
    # SPECIES THAT ESCAPE ON THEIR FIRST FREE TURN (2026-07-30, Jonny live report: full-HP ball at an
    # Abra, it Teleported). Wild Abra/Kadabra in FireRed know only Teleport — EVERY turn you spend
    # weakening/switching hands it the exit. The only play is ball-on-sight; the throw itself resolves
    # before the foe acts, so a break-free still costs the encounter but a weaken ALWAYS does.
    _FLEES_ON_FREE_TURN = {63: "Teleport", 64: "Teleport"}          # Abra, Kadabra
    # CHIPPER-SWITCH band (the 'weaken with NOT the ace' play): when the lead out-levels the wild by
    # 10+ (any hit could KO) and no sleep move is up, switch to a teammate whose level sits within
    # this margin ABOVE the foe — close enough that its gentlest move chips instead of one-shots.
    CATCH_CHIPPER_MAX_OVER = int(os.getenv("POKEMON_CATCH_CHIPPER_OVER", "9"))

    def _catch_chipper_slot(self, foe_level):
        """Best party slot to do the CHIPPING when the lead would one-shot the catch target: alive,
        >40% HP (it will eat one wild hit during the switch turn), level above the foe (it must win
        the trade) but within CATCH_CHIPPER_MAX_OVER of it (its hits stay survivable). Prefers the
        strongest in-band teammate. None = nobody fits (caller keeps the old full-HP-throw path)."""
        if not foe_level:
            return None
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        best, best_lv = None, 0
        for s in range(1, min(cnt, 6)):
            base = ram.GPLAYER_PARTY + s * 100
            hp, maxhp = self.b.rd16(base + 0x56), self.b.rd16(base + 0x58)
            if hp <= 0 or (maxhp and hp / maxhp <= 0.40):
                continue
            lv = self.b.rd8(base + 0x54)
            if foe_level < lv <= foe_level + self.CATCH_CHIPPER_MAX_OVER and lv > best_lv:
                best, best_lv = s, lv
        return best

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
        just means catch_pokemon returns 'fainted' and the wander finds another wild.

        DAMAGE-AWARE (2026-07-30, Jonny live report: 'she KOs the ones she wants to catch'): the old
        loop only checked hp > target_frac before swinging — a foe sitting just ABOVE the band (e.g.
        45% after one chip that hit for 55%) took one more 'gentle' chip and died. Now each chip's
        actual damage is MEASURED, and we refuse the next swing when the foe wouldn't clearly survive
        a repeat of it (1.6x safety margin — crits happen). A catch thrown at 50% HP costs an extra
        ball at worst; a KO'd keeper is gone forever."""
        last_dmg = None
        for _ in range(max_hits):
            state = st.read_battle(self.b)
            if not state or not st.in_battle(self.b):
                return
            foe = state["enemy"]
            if not foe.get("maxhp") or foe["hp"] <= 0:
                return
            if foe["hp"] / foe["maxhp"] <= target_frac:
                return                                  # already in the catchable band
            if last_dmg and foe["hp"] <= last_dmg * 1.6:
                self.log(f"   [engine] catch-weaken: STOPPING — foe at {foe['hp']}/{foe['maxhp']} HP, "
                         f"last chip hit for {last_dmg}; another swing risks a KO. Throwing now.")
                return                                  # next hit plausibly kills it — throw instead
            cand = [(i, m.get("id", 0)) for i, m in enumerate(state["ours"]["moves"])
                    if m.get("id", 0) and m.get("pp", 0) > 0 and m.get("id", 0) not in self._STATUS_MOVES]
            if not cand:
                return                                  # no damaging move with PP -> just throw
            cand.sort(key=lambda im: st.move_info(self.b, im[1])[1] or 0)   # gentlest (lowest power)
            hp_before = foe["hp"]
            self._fire_move(cand[0][0])
            after = st.read_battle(self.b)
            if after and after["enemy"].get("hp") is not None and after["enemy"]["hp"] < hp_before:
                last_dmg = hp_before - after["enemy"]["hp"]   # measure what a chip actually costs

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

    def catch_pokemon(self, max_seconds=150, weaken=True, ball_pref="cheap", allow_master=False):
        """Catch the WILD foe (the proven live flow, automated): optionally WEAKEN/STATUS it once to
        boost the catch rate + stop it attacking, then THROW balls until caught. COMMITS - it
        re-throws after a break instead of abandoning after one ball (the live Ekans flow). Returns
        'caught' | 'no_balls' | 'trainer' | 'fled' | 'fainted' | 'stuck'. Gen-3 trainer mons can't
        be caught (returns 'trainer'). ball_pref/allow_master: the BALL TIER doctrine (2026-08-04) —
        legendaries get 'best' (Ultra-first) and Mewtwo alone may spend the Master Ball."""
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
        chipper_tried = False
        try:
            _rb0 = st.read_battle(self.b)
            # FLEE-ON-SIGHT SPECIES (Abra/Kadabra): every weaken/status/switch turn hands it the
            # Teleport exit. Skip ALL softening and throw immediately — the deliberate, narrated
            # version of what previously looked like a mistake.
            _foe_sp0 = (_rb0 or {}).get("enemy", {}).get("species")
            if _foe_sp0 in self._FLEES_ON_FREE_TURN:
                weaken = False
                _fname = st.SPECIES_NAME.get(_foe_sp0, "this one")
                self.log(f"   [engine] catch: {_fname} escapes ({self._FLEES_ON_FREE_TURN[_foe_sp0]}) "
                         f"on its first free turn — skipping weaken, ball-on-sight")
                self.emit(f"{_fname} teleports away the second it gets a turn — no time to weaken it, "
                          f"I have to throw RIGHT NOW and pray.", beat=True, tier=1)
            elif weaken and _rb0 and (
                    (_rb0["ours"].get("level", 0) - _rb0["enemy"].get("level", 0)) >= 10
                    or _foe_sp0 == 143):  # Snorlax: Surf/Hydro from Blastoise OHKOs — never chip
                status_only = True
                self.log("   [engine] catch: foe is 10+ levels under the lead (or Snorlax) — "
                         "no chipping (would KO); SLEEP-then-throw / CHIPPER, never ace Surf")
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
            if self._ball_count() <= 0 and not (allow_master
                                                and self._ball_qty(self._BALL_MASTER) > 0):
                self.emit("I'm out of Poké Balls - I'll come back for this one", beat=True)
                # NEVER fight-clear a catch_now / Snorlax / shiny / legendary — that was the
                # 2026-08-02 LIVE KO of Jonny's ordered Snorlax (Blastoise Surf, zero balls).
                _foe_sp = (st.read_battle(self.b) or {}).get("enemy", {}).get("species")
                _protect = (self._peek_creator_catch_order()
                            or _foe_sp in (143, 144, 145, 146, 150, 151)  # snorlax + legendaries
                            or st.enemy_is_shiny(self.b))
                if self._foe_blocks_flee() and not _protect:
                    self.log("   [engine] catch no_balls + Arena Trap — FIGHTING clear (no RUN attempt)")
                    self._skip_catch_divert = True
                    try:
                        self.run(max_seconds=120)
                    finally:
                        self._skip_catch_divert = False
                else:
                    self.log("   [engine] catch no_balls — FLEEING (protect catch target from KO)")
                    self.flee(max_seconds=45)
                    if st.in_battle(self.b) and not _protect:
                        self.log("   [engine] catch no_balls: flee failed — fighting clear")
                        self._resolve_open_battle(max_seconds=120)
                    elif st.in_battle(self.b) and _protect:
                        self.log("   [engine] !! catch no_balls + protected target still in battle "
                                 "— refusing fight-clear KO; fleeing again")
                        self.flee(max_seconds=30)
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
                # CHIPPER SWITCH (2026-07-30, Jonny live report: 'she needs to weaken it with NOT the
                # ace'): no sleep move up and the ace would one-shot the target — the real-player play
                # is to field a CLOSE-LEVEL teammate whose hits chip instead of KO. Reuses the proven,
                # fail-safe grind-switch actuation (_switch_to_slot: species-confirmed, B-out on any
                # non-confirm -> we just fall through to the old full-HP throw). One attempt per catch;
                # the wild's free turn during the switch is priced in (flee-risk species never reach
                # here — they take the ball-on-sight path above).
                if not chipper_tried and not state["enemy"].get("asleep"):
                    chipper_tried = True
                    ch = self._catch_chipper_slot(state["enemy"].get("level"))
                    if ch is not None:
                        _ch_nm = st.SPECIES_NAME.get(st.read_party_species(self.b, ch), "a teammate")
                        _my_nm = st.SPECIES_NAME.get(state["ours"].get("species"), "my ace")
                        self.log(f"   [engine] catch: CHIPPER SWITCH — {_my_nm} would one-shot it; "
                                 f"fielding slot {ch} ({_ch_nm}) to chip it into the catchable band")
                        if self._switch_to_slot(ch, state["ours"].get("species")) == "switched":
                            self.emit(f"{_my_nm} hits way too hard for this — {_ch_nm}, you're up. "
                                      f"soften it, don't finish it.", beat=True, tier=1)
                            self._weaken_hp()            # damage-aware chip with the close-level mon
                            softened = True
                            continue
                        self.log("   [engine] catch: chipper switch didn't confirm — falling back to "
                                 "the full-HP throw (fail-safe)")
                if not softened and not state["enemy"].get("asleep"):
                    self.emit("no safe way to weaken this one — full-health throw it is. wish me luck.",
                              beat=True)
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
            res = self.throw_ball(max_seconds=max(20, int(max_seconds - (time.time() - t0))),
                                  pref=ball_pref, allow_master=allow_master)
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
        """DEPRECATED probe — NEVER tap d-pad here.

        On this libmgba core the first d-pad at the action menu CONFIRMS FIGHT (opens the move
        list). The old LEFT/RIGHT 'alive' probe WAS the Fight↔Bag / Fight↔moves stream thrash
        (2026-08-02 LIVE). Truth is GBATTLE_MENU_UP==1, not cursor motion."""
        return self._at_action_menu()

    def _settle_action_menu(self, tries=30):
        """Reach the real ACTION menu via GBATTLE_MENU_UP (not d-pad probes / white_box alone).

        Impostors (party/bag/text) drained with B. Move list left alone if we only need 'a menu'
        — callers that need ACTION specifically see menu_up==1."""
        self._settle(need=6, timeout=200)
        for _ in range(tries):
            if not st.in_battle(self.b):
                return False
            if self._bag_screen() or self._party_screen():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
                continue
            if self._at_action_menu():
                # A settled action menu is one a blind A may hit (anti-wedge floor, stall mash) —
                # park the cursor on FIGHT so that A can never re-open the BAG/POKEMON screen
                # (2026-08-03 Route-13 fisherman: cursor left on BAG = infinite Super-Potion loop).
                if self.b.rd8(ram.GBATTLE_ACTION_CURSOR) != ram.ACT_FIGHT:
                    self._poke_action_cursor(ram.ACT_FIGHT)
                return True
            if self._at_move_list():
                # Back to action — ONE B only (don't A/B oscillate).
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
                continue
            if not self._white_box():
                self._advance_text()
            else:
                # White impostor (trainer prize / item text) — B drain, no d-pad.
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
        return self._at_action_menu()

    def _poke_action_cursor(self, want):
        """Write GBATTLE_ACTION_CURSOR and verify. On this libmgba core, d-pad from FIGHT can
        CONFIRM Fight (open the move list) instead of moving the cursor — that was the
        'alive Blastoise can't leave / switch loop' theater (2026-08-03 Jonny: Voltorb docks).
        Returns True iff the byte reads back as `want` while still on the action menu."""
        if not self._at_action_menu():
            return False
        try:
            self.b.core.memory.u8.raw_write(ram.GBATTLE_ACTION_CURSOR, int(want) & 0xFF)
        except Exception as e:
            self.log(f"   [engine] action-cursor write failed: {e}")
            return False
        self._wait(2)
        return self._at_action_menu() and self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == want

    def _goto_bag(self, tries=10):
        """Park the action cursor on BAG (ACT_BAG=1). Prefer RAM write; d-pad is fallback only
        and must B-out immediately if it opens the move list (Fight confirm)."""
        for _ in range(tries):
            if self._at_move_list():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                continue
            if not self._at_action_menu():
                return False
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_BAG:
                return True
            if self._poke_action_cursor(ram.ACT_BAG):
                return True
            # Fallback d-pad once — if it confirmed FIGHT, B back next iter.
            if c == ram.ACT_FIGHT:
                self._tap("RIGHT")
            elif c == ram.ACT_RUN:
                self._tap("UP")
            elif c == ram.ACT_POKEMON:
                self._tap("UP")
            else:
                return False
            self._wait(3)
            if self._at_move_list():
                self.log("   [engine] _goto_bag: d-pad confirmed FIGHT — B out, retry write")
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
        return self._at_action_menu() and self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_BAG

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
        """ZERO usable moves in a can't-flee battle: A on FIGHT — if ALL moves are truly dry the
        game substitutes Struggle ("X has no moves left!") and the turn resolves. 2026-08-03
        HARDENING (the Tackle-spam endgame): when OUR ledger says every slot is dry/exiled but
        the game still OPENS the move list (our PP reads were the liars, not the game), the old
        code returned 'done' having done NOTHING — a silent per-turn no-op that upgraded into the
        anti-wedge A-mash. Now we do what a human does: WALK the list and try each slot (readback
        nav), skipping exiled ones first; the game fires whichever really has PP. A slot the game
        refuses gets exiled on the spot (text is truth). Only if every slot refuses do we return
        'no_usable_move' (the anti-wedge floor owns it)."""
        if not self._settle_action_menu() or not self._goto_fight():
            return "no_usable_move"
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
        self._wait(30)
        if not self._at_move_list():
            # FROZEN-DETECTOR GUARD (2026-08-03 11:49 logs: "ZERO PP anywhere -> FIGHT+A"
            # spammed for minutes): moves=False here CAN be a lie — the battle-UI RAM block
            # desyncs for whole battles. If pixels say a white menu is up, the list very
            # likely opened (the game only opens it when SOMETHING has PP) — fall through
            # to the slot walk below with BLIND nav instead of returning a silent no-op.
            if not self._white_box():
                for _ in range(6):                        # true zero-PP: drain "no moves left!" text
                    if not st.in_battle(self.b) or self._white_box():
                        break
                    self._advance_text()
                return "done"
            self.log("   [engine] struggle-walk: RAM says no move list but pixels say white "
                     "menu — BLIND-WALKING the slots (frozen-detector guard)")
        # Move list opened — the game thinks something is usable. Try slots, least-refused first.
        _ram_list = self._at_move_list()                  # False here = frozen detectors; go blind
        order = sorted(range(4), key=lambda i: self._move_refused.get(i, 0))
        for slot in order:
            if not st.in_battle(self.b):
                return "done"
            if _ram_list and not self._at_move_list():
                return "done"                             # something fired / battle moved on
            if not _ram_list and not self._white_box():
                return "done"                             # blind mode: white menu gone = resolved
            # FUTILITY GATE (09:07): if the list has already proven itself a tar pit, bench-switch
            # instead of feeding it more confirms.
            if (getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX
                    and self._is_trainer_battle()):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(14)
                if self._futility_bench_switch():
                    return "done"
            # BLIND-FIRST (12:07 relapse): deterministic walk owns positioning; the RAM write
            # runs after only as a correction when the byte is still readable and disagrees.
            self._blind_goto_move(slot)
            if _ram_list and self.b.rd8(MOVE_CURSOR) != slot:
                self._goto_move(slot)
            pressed = slot
            before = self._bstate()
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(20)
            for _fn in range(240):                        # ~4s: did the turn actually leave?
                if not st.in_battle(self.b):
                    return "done"
                _bt = self._battle_text() if _fn % 12 == 0 else ""
                if _bt and any(sn in _bt for sn in self._MOVE_REFUSAL_SNIPPETS):
                    self._move_refused[pressed] = max(self._move_refused.get(pressed, 0), 2)
                    self._amove_futile = getattr(self, "_amove_futile", 0) + 1
                    self.log(f"   [engine] struggle-walk: game refused slot {pressed} — exiled, "
                             f"trying the next (futility {self._amove_futile}/{FUTILE_AMOVE_MAX})")
                    self._wait(20)                        # let the refusal box clear
                    break
                cur = st.read_battle(self.b)
                if cur:
                    hp = (cur["enemy"]["hp"], cur["ours"]["hp"])
                    if before and hp != before:
                        self.log(f"   [engine] struggle-walk: slot {slot} FIRED (the PP ledger "
                                 f"was lying) — turn resolved")
                        self._note_battle_progress(f"struggle-walk slot {slot}")
                        return "done"
                if not self._white_box():
                    self.b.press("A", 2, 8, self.render, owner=self.owner)
                self.b.run_frame(); self.render()
        self.log("   [engine] struggle-walk: every slot refused — no_usable_move (LOUD)")
        return "no_usable_move"

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
                break
            self.b.press("B", 2, 12, self.render, owner=self.owner); self._wait(10)
        self._rehome_fight_cursor()

    def _rehome_fight_cursor(self):
        """2026-08-03 THE FISHERMAN LOOP (Route 13 dock): after ANY bag/party trip the action-menu
        cursor stays parked on BAG (FRLG remembers it), and the bag's list cursor stays on the last
        item (Super Potion). Every later BLIND A — the anti-wedge floor's 'mash FIGHT+A', the wedge
        bail, the stall mash — then RE-OPENS the bag on the potion: 'SUPER POTION is selected.' ->
        'Use on which POKMON?' -> B-drain -> blind A -> forever, on stream, for minutes. Pitfall 13
        applies to US too: never leave a cursor where a blind press detonates. Park it back on FIGHT
        (RAM write + readback; d-pad fallback) every time we leave a bag/party screen."""
        try:
            if st.in_battle(self.b) and self._at_action_menu():
                # RAM write ONLY — a d-pad fallback here can CONFIRM instead of move on this
                # core (the _home_to_fight no-op lesson). A failed write just logs; the blind-A
                # bans elsewhere still protect us.
                if not self._poke_action_cursor(ram.ACT_FIGHT):
                    self.log("   [engine] rehome-FIGHT: cursor write didn't verify (LOUD; "
                             "blind-A bans still guard the bag)")
        except Exception as e:
            self.log(f"   [engine] rehome-FIGHT failed: {e} (LOUD)")

    def _blind_menu_unwind(self, presses=8):
        """2026-08-04 LIVE (the Hyper-Potion-forever golbat fight — 19 minutes on stream): every
        screen classifier lied AT ONCE (bag=False party=False/True menu_up=1 white-box lit) while
        the REAL screen was the bag list -> 'Use on which POKéMON?', so every screen-AWARE escape
        (_war_advance_press, the party-thrash B-loop) consulted the same lying reads and pressed
        the loop right back open. B is the one blind-safe key in FRLG battle menus — it only
        cancels or advances text, NEVER confirms — so when the wedge signature shows, stop
        trusting screens entirely: B-storm the whole menu stack down unconditionally, then park
        the action cursor back on FIGHT so no later blind A can re-open the bag."""
        for _ in range(presses):
            if not st.in_battle(self.b):
                return
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(14)
        self._rehome_fight_cursor()

    def _true_active_party_hp(self):
        """Tear-safe HP for the mon currently OUT: gPlayerParty[gBattlerPartyIndexes[0]].
        gBattleMons HP can tear (looks hurt while the bar is full) and was the 2026-08-02
        LIVE Super-Potion-on-full-ace loop. Returns (hp, maxhp) or (None, None)."""
        try:
            pi = self.b.rd16(ram.GBATTLER_PARTY_IDX)
            if not (0 <= pi < 6):
                return None, None
            base = ram.GPLAYER_PARTY + pi * 100
            hp = self.b.rd16(base + 0x56)
            mx = self.b.rd16(base + 0x58)
            if not (1 <= mx <= 999 and 0 <= hp <= mx):
                return None, None
            return hp, mx
        except Exception:
            return None, None

    def _battle_text(self):
        """Best-effort read of the ACTIVE game message (gStringVar3, the dialogue-reader
        address — battle bag/refusal strings were observed flowing through it live).
        Lowercased/normalized; '' when unreadable. GROUND TRUTH accelerant only — every
        caller must also work when this returns '' (the struct-timeout path still counts
        refusals)."""
        try:
            from dialogue_reader import decode, DialogueReader
            s, junk = decode(self.b.read_bytes(DialogueReader.ACTIVE_MSG, 0xC0))
            if not s or junk > 0.3:
                return ""
            return " ".join(s.lower().split())
        except Exception:
            return ""

    # The game's own move-rejection strings — authoritative: the slot is DEAD this battle
    # no matter what the (tear-prone) PP byte claims.
    _MOVE_REFUSAL_SNIPPETS = ("no pp left", "is disabled")

    def _war_advance_press(self):
        """ONE screen-aware press for the war-must-advance paths (anti-wedge floor, menu-wedge
        bail, stall mash). The 2026-08-03 Route-13 fisherman loop: those paths pressed BLIND A,
        the action cursor was parked on BAG after an item trip, so every 'mash FIGHT' RE-OPENED
        the bag on Super Potion — an infinite selected/Use-on-which cycle their own counters
        kept resetting. Rule: NEVER a blind A. Bag/party -> B. Action menu -> cursor to FIGHT
        (RAM write, readback) THEN A. Move list -> A (fires the highlighted move — resolution).
        Anything else -> advance text."""
        if self._bag_screen() or self._party_screen():
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(12)
            return "B"
        if self._at_action_menu():
            # FROZEN-DETECTOR GUARD (11:49 logs: action=True held for a WHOLE battle while the
            # move list was open — every 'A@FIGHT' here actually confirmed the parked slot).
            # Once the futility floor is breached, stop trusting this byte: blind-walk to the
            # least-refused slot and A. Real action menu: worst case the walk confirms FIGHT
            # and then navigates the list it opened — still fires the intended slot.
            if getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX:
                _ref = getattr(self, "_move_refused", {})
                _best = min(range(4), key=lambda i: _ref.get(i, 0))
                self._blind_goto_move(_best)
                self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(16)
                self._amove_futile += 1
                return f"A@blind{_best}"
            if self.b.rd8(ram.GBATTLE_ACTION_CURSOR) != ram.ACT_FIGHT:
                self._poke_action_cursor(ram.ACT_FIGHT)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(14)
            return "A@FIGHT"
        if self._at_move_list():
            # FUTILITY BREAKER (2026-08-03 09:07, the parked-on-Tackle photos): if the move list
            # has already eaten FUTILE_AMOVE_MAX confirms this battle with zero real progress,
            # one more A here is theater — the cursor readback is lying or every slot the game
            # will actually select is dry. Stop touching the list: B out and bench-switch (the
            # party path verifies by the active SPECIES flipping — move RAM can't lie to it).
            if (getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX
                    and self._is_trainer_battle()):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(14)
                if self._futility_bench_switch():
                    return "SWITCHED(futility)"
                # no bench / didn't confirm -> fall through and A anyway (Struggle path)
            self._amove_futile = getattr(self, "_amove_futile", 0) + 1
            # Never confirm a slot the game already refused (the highlighted move IS the
            # refused one after a refusal bounce) — steer to the least-refused slot first.
            _ref = getattr(self, "_move_refused", {})
            if _ref:
                _best = min(range(4), key=lambda i: _ref.get(i, 0))
                if _ref.get(self.b.rd8(MOVE_CURSOR), 0) > _ref.get(_best, 0):
                    self._goto_move(_best)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(16)
            return "A@move"
        self._advance_text()
        return "text"

    def _futility_bench_switch(self):
        """LAST-RESORT resolver for a move list that eats every confirm (the 09:07 Tackle loop:
        cursor readback said 'nav ok', the drawn cursor never left TACKLE, refusals tallied on
        the WRONG slots, famine never tripped). Uses only reads proven live: party HP/species
        from gPlayerParty and the species-flip-verified _switch_to_slot. Bounded per battle."""
        if getattr(self, "_futility_switches", 0) >= 2:
            return False
        try:
            pi = self.b.rd16(ram.GBATTLER_PARTY_IDX)
            active_sp = st.read_party_species(self.b, pi) if 0 <= pi < 6 else None
            cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            best, best_lv = None, -1
            for s in range(min(cnt, 6)):
                if s == pi or self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                    continue
                sp = st.read_party_species(self.b, s)
                if not sp or sp == active_sp:
                    continue
                lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
                if lv > best_lv:
                    best, best_lv = s, lv
            if best is None:
                self.log("   [engine] FUTILITY BREAKER: no live bench mon — the move list is all "
                         "we have (Struggle/whiteout owns it)")
                return False
            self._futility_switches = getattr(self, "_futility_switches", 0) + 1
            self.log(f"   [engine] !! FUTILITY BREAKER: {self._amove_futile} fruitless move-list "
                     f"confirms — the list is a tar pit (lying cursor/PP RAM). Benching the active "
                     f"for party slot {best} (L{best_lv}) — switch verifies by species flip, "
                     f"not move RAM (use {self._futility_switches}/2)")
            if self._switch_to_slot(best, active_sp) == "switched":
                self.emit("my moves are jammed — tagging in someone who can actually swing.",
                          beat=True, tier=2)
                self._note_battle_progress("futility bench switch")
                self._skip_streak.clear()
                return True
            self.log("   [engine] FUTILITY BREAKER: switch did not confirm (fail-safe, fighting on)")
        except Exception as e:
            self.log(f"   [engine] FUTILITY BREAKER failed: {e} (LOUD)")
        return False

    def _true_active_party_status(self):
        """Tear-safe STATUS for the mon currently OUT — gPlayerParty[gBattlerPartyIndexes[0]]
        status u32 at +0x50, decoded to 'sleep'/'poison'/'burn'/'freeze'/'paralysis'/None.
        2026-08-03 LIVE (the Antidote-on-paralyzed-Blastoise loop): gBattleMons status1 tore
        and decoded as POISON while the game (and chat) knew he was PARALYZED — she confirmed
        Antidote into 'It won't have any effect.' over and over. Same lying struct as the HP
        tear; same cure: the party struct is the truth. Returns (decoded, ok) — ok=False means
        the read failed and callers must NOT trust either source."""
        try:
            pi = self.b.rd16(ram.GBATTLER_PARTY_IDX)
            if not (0 <= pi < 6):
                return None, False
            s = self.b.rd32(ram.GPLAYER_PARTY + pi * 100 + 0x50)
            return _decode_status(s & 0xFF), True
        except Exception:
            return None, False

    def _note_battle_progress(self, why=""):
        """Real fight progress (not menu flicker) — resets the MENU WEDGE stall clock."""
        self._last_battle_progress_t = time.time()
        self._amove_futile = 0             # the move list produced something real again
        self._immob_streak = 0             # real progress = the stillness streak is over too
        if why:
            self.log(f"   [engine] battle-progress: {why}")

    def use_item_in_battle(self, item_id, max_seconds=30, target=None):
        """Use one `item_id` from the Items pocket. Returns 'used' (count dropped) | 'no_item' |
        'failed' | 'no_effect'. FAIL-SAFE: anything but 'used' leaves the battle fightable. `target`
        aims the item's party screen: 'active' (the mon that is OUT — always menu row 0, the lead
        panel) or 'fainted' (the strongest downed mon — Revive). The row is resolved by CONTENT at
        MENU TIME (_menu_rows order law): run14 frame-proof — a Revive aimed at a PRE-menu slot
        index confirmed the healthy active mon's panel, ate 'It won't have any effect.' boxes all
        night, and never consumed. None keeps the legacy un-aimed walk; aim taps are
        party-screen-gated (pixel truth) so a lagging party open never taps into the bag's
        USE/CANCEL sub-box."""
        ids = [i for i, _ in self._items_pocket()]
        if item_id not in ids:
            self.log(f"   [engine] use_item: item {item_id} NOT in pocket {ids[:8]} — no_item (LOUD)")
            return "no_item"
        row = ids.index(item_id)
        cnt0 = self._items_count(item_id)
        _is_heal = item_id in _HEAL_ITEMS_PREF
        _is_cure = item_id in set(_STATUS_CURE_ITEM.values()) | {_FULL_HEAL}
        if not self._settle_action_menu():
            self.log("   [engine] use_item: couldn't reach the action menu — keep fighting (LOUD)")
            return "failed"
        if not self._open_bag():
            self.log("   [engine] use_item: bag wouldn't open (eaten RIGHT?) — keep fighting (LOUD)")
            self._exit_bag(); return "failed"
        # POCKET LIVENESS PROBE (2026-08-03 13:01 — 'hovering over Teachy TV and Helix Fossil
        # mid fight'): Teachy TV/Helix Fossil live in the KEY ITEMS pocket. The pocket byte
        # read 0 (Items) while the REAL pocket was Key Items — frozen-RAM disease — so zero
        # LEFTs were pressed and the row walk selected key items. A healthy byte must RESPOND
        # to a d-pad press before steering trusts it (the START-menu open-verify doctrine);
        # a mute byte = frozen -> BLIND clamp: LEFT x4 (the pocket strip clamps at Items).
        _p0 = self.b.rd8(ram.GBAG_POCKET)
        self._tap("RIGHT"); self._wait(12)
        _live = self.b.rd8(ram.GBAG_POCKET) != _p0
        if not _live:
            self._tap("LEFT"); self._wait(12)
            _live = self.b.rd8(ram.GBAG_POCKET) != _p0
        if _live:
            for _ in range(8):                               # steer GBAG_POCKET to the Items pocket (0)
                if self.b.rd8(ram.GBAG_POCKET) == 0:
                    break
                self._tap("LEFT"); self._wait(12)
            if self.b.rd8(ram.GBAG_POCKET) != 0:
                self.log("   [engine] use_item: couldn't reach the Items pocket — keep fighting (LOUD)")
                self._exit_bag(); return "failed"
        else:
            # 2026-08-04 LIVE (Gary, Blastoise at 12/170): pocket byte MUTE *and* zero bag/
            # white pixels on screen = the bag very likely never opened despite _open_bag's
            # white-gone pass (mid-transition frame). Blind LEFTs at a NON-bag screen are
            # poison — on this core a d-pad press at the action menu can CONFIRM. Re-check
            # once after a settle; still nothing -> bail fightable instead of spraying keys.
            if not self._bag_screen() and not self._white_box() and not self._bag_menu_cb2():
                self._wait(40)
                if not self._bag_screen() and not self._white_box() and not self._bag_menu_cb2():
                    self.log("   [engine] use_item: pocket byte MUTE + NO bag/white pixels + "
                             "callback2 not the bag — bag never opened; keep fighting "
                             "(LOUD, no blind taps fired)")
                    self._exit_bag(); return "failed"
            self.log("   [engine] use_item: pocket byte is MUTE (frozen RAM) — BLIND clamp "
                     "LEFT x4 to the Items pocket")
            for _ in range(4):
                self._tap("LEFT"); self._wait(12)
        def _sel():
            # TRUE selection = cursor + scrollOffset (the mart-list law, recon_bagscroll-verified).
            # The raw cursor byte alone LIES after any scrolled visit — the list remembers both.
            return self.b.rd8(BAG_CURSOR) + self.b.rd16(BAG_SCROLL)
        if _live:
            for _ in range(14):                              # nav to the item's TRUE row (cursor+scroll)
                if _sel() == row:
                    break
                self._tap("DOWN" if _sel() < row else "UP"); self._wait(10)
            self._wait(8)                                    # settle scroll animation, then re-verify
        # Mute pocket byte = the whole bag RAM block is suspect: a frozen _sel() that happens
        # to equal `row` would FALSE-PASS and select the wrong item — skip readback nav
        # entirely and let the blind walk below own positioning.
        if not _live or _sel() != row:
            # BLIND BAG WALK (2026-08-03 12:07: 'couldn't reach true row 4 (cursor=1 scroll=0)'
            # killed the one Ether attempt of the whole loop): the bag cursor/scroll bytes have
            # the same frozen-RAM disease as the battle cursors. The list CLAMPS at the top, so
            # UP x (row+6) homes to row 0 from any position even with eaten presses, then
            # DOWN x row lands the true row with zero RAM trust. Count-drop stays the only
            # consume-truth, and a mis-select just exhausts the A-walk -> 'failed' (fail-safe).
            if row <= 8:
                self.log(f"   [engine] use_item: RAM says row {_sel()} wanted {row} — bytes may "
                         f"be frozen; BLIND bag walk (clamp-home + DOWN x{row})")
                for _ in range(row + 6):
                    self._tap("UP"); self._wait(8)
                for _ in range(row):
                    self._tap("DOWN"); self._wait(8)
            else:
                self.log(f"   [engine] use_item: couldn't reach true row {row} "
                         f"(cursor={self.b.rd8(BAG_CURSOR)} scroll={self.b.rd16(BAG_SCROLL)}) — "
                         f"too deep for a blind walk; keep fighting (LOUD)")
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
        # Cap raised 4->6 (2026-08-04 Gary): frozen-frame battles burn 2-3 laps before the
        # party screen even registers, so a 4-lap budget died the instant aiming started.
        # The wrong-aim A-spam risk that set the old cap is now closed by the no_effect
        # aborts BEFORE the confirm.
        for n in range(6):
            if self._items_count(item_id) < cnt0:
                break
            if not aimed and (self._party_screen() or self._party_menu_cb2()):
                # ^ pixel OR callback2 truth (2026-08-04, the Revive insta-click: on frozen
                # frames the pixel classifier missed the freshly-opened target screen, this
                # block never ran, and the bare walk-A below confirmed the HOME cursor — the
                # alive lead. gMain.callback2 IS the party menu while it owns input.)
                self._wait(8)                              # let the screen finish drawing
                # 2026-08-04 LIVE (the Gary wipe — FOUR Revives 'selected', ZERO consumed,
                # full team down): _party_focus()'s eaten-tap retry presses B, and on the
                # item-target screen ("Use on which POKéMON?") B CANCELS back to the bag
                # list — the follow-up goto taps then scrolled the BAG cursor instead
                # ('REVIVE is selected.' -> 'NUGGET is selected.') and the confirm A
                # inspected a Nugget, every attempt, while the team bled out. The target
                # screen always opens with the cursor HOME (lead panel), so no focus probe
                # is needed: dismiss a stray sub-box, then aim by border readback — and if
                # the orange border can't be seen (fade/half-drawn frames), walk BLIND from
                # the clamped home. NEVER press B inside this block.
                if self._party_submenu():
                    self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(16)
                rows = self._menu_rows()
                if isinstance(target, int):                # an EXACT party slot (revive routing)
                    _row = target
                elif target == "fainted":
                    _row = next((r["row"] for r in sorted(rows, key=lambda r: -r["level"])
                                 if r["hp"] == 0), None)
                    if _row is None:
                        # No fainted row visible at menu time (torn/frozen party block) —
                        # confirming the healthy default row 0 is the "won't have any
                        # effect" A-loop. Abort BEFORE the confirm instead.
                        self.log("   [engine] use_item: no fainted row at menu time — "
                                 "aborting revive (no_effect; fail-safe)")
                        self._exit_bag()
                        return "no_effect"
                else:                                      # 'active' -> the lead panel
                    _row = 0
                    # 2026-08-02 LIVE: heal aimed at FULL ace (torn gBattleMons said hurt) —
                    # "It won't have any effect." A-spam forever. Abort BEFORE confirming.
                    if _is_heal and rows:
                        lead = next((r for r in rows if r["row"] == 0), rows[0])
                        if lead.get("maxhp") and lead.get("hp", 0) >= lead["maxhp"]:
                            self.log(f"   [engine] use_item: lead {st.SPECIES_NAME.get(lead.get('species'), '?')} "
                                     f"is FULL HP ({lead['hp']}/{lead['maxhp']}) — aborting potion "
                                     f"(no_effect; was the full-ace Super Potion loop)")
                            self._exit_bag()
                            return "no_effect"
                    # Cure verification at MENU TIME (party struct first — the tear-safe truth):
                    # (a) no status at all -> the Awakening re-open loop; (b) WRONG MEDICINE —
                    # 2026-08-03 LIVE: Antidote confirmed on a PARALYZED Blastoise, "It won't
                    # have any effect." forever. The item must treat the LIVE status (exact
                    # match or Full Heal) or we abort before the confirm.
                    if _is_cure:
                        _cur_status, _sok = self._true_active_party_status()
                        if not _sok:
                            _st = st.read_battle(self.b) or {}
                            _cur_status = _decode_status((_st.get("ours") or {}).get("status1", 0) or 0)
                        if not _cur_status:
                            self.log("   [engine] use_item: active has NO status — aborting cure "
                                     "(no_effect; was the Awakening re-open loop)")
                            self._exit_bag()
                            return "no_effect"
                        _needed = _STATUS_CURE_ITEM.get(_cur_status)
                        if item_id != _FULL_HEAL and item_id != _needed:
                            self.log(f"   [engine] use_item: WRONG MEDICINE — item {item_id} does not "
                                     f"treat {_cur_status!r} (needs {_needed} or Full Heal {_FULL_HEAL}) "
                                     f"— aborting BEFORE the confirm (no_effect; was the Antidote-on-"
                                     f"paralysis loop)")
                            self.emit(f"wrong medicine — that won't fix {_cur_status}. fighting on.",
                                      beat=True, tier=2)
                            self._exit_bag()
                            return "no_effect"
                _seen = (self._party_cursor_slot() is not None
                         or self._party_cursor_on_lead())
                if not _seen or not self._party_goto_slot(_row):
                    # Border readback blind (fade frames / frozen pixels) or the closed-loop
                    # walk couldn't confirm — walk BLIND from the clamped home (LEFT clamps
                    # the lead panel; RIGHT + DOWN×(row-1) lands any right-column row). Only
                    # d-pad taps: on this screen a stray B cancels, a stray A confirms.
                    self.log(f"   [engine] use_item: cursor unreadable/unreached for row {_row} "
                             f"— BLIND party walk (LEFT home, RIGHT + DOWN x{max(0, _row - 1)})")
                    self._party_blind_goto(_row)
                aimed = True
            self.log(f"   [engine] use_item walk n={n}: party={self._party_screen()} "
                     f"bag={self._bag_screen()} white={self._white_box()} "
                     f"cb2party={self._party_menu_cb2()} cb2bag={self._bag_menu_cb2()} "
                     f"pcur={self._party_cursor_slot()} lead={self._party_cursor_on_lead()}")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)
            if not st.in_battle(self.b):
                break
        if self._items_count(item_id) < cnt0:
            self.log(f"   [engine] use_item: USED item {item_id} (count {cnt0}->{self._items_count(item_id)})")
            self.emit("used an item — that's better", beat=True)
            self._note_battle_progress(f"item {item_id} consumed")
            # LAYER 8 FIX: close the BAG first — the old drain exited on _white_box(), but the bag's
            # USE/CANCEL sub-box LIGHTS those pixels, so 'used' could return with the bag still open
            # and the next turn's presses landed in it forever (caterpie 7/40, walk 3).
            self._close_bag_screen()
            # Drain the "X recovered!" text back to a LIVE menu/battle. The old 6-lap exit broke
            # on _white_box alone — the PARTY screen's result box ("PERSIAN's HP was restored...")
            # lights those pixels, so 'used' returned with the box still up and the turn loop
            # wedged against it forever (the run19 Lance livelock). _settle_action_menu now
            # demands cursor responsiveness and B-drains impostors, so route through it.
            if not self._settle_action_menu(tries=12):
                # 2026-08-04 LIVE (the Hyper-Potion golbat loop): the settle came back UNCONFIRMED
                # — the bag/target screen was still up but invisible to every classifier, and the
                # next turn's 'move commit' re-confirmed the parked potion forever. Don't return
                # 'used' on a lying screen: blind B-unwind the whole stack first.
                self.log("   [engine] use_item: post-use settle NOT confirmed — blind B-unwind "
                         "(the bag can sit open invisible to every classifier)")
                self._blind_menu_unwind(8)
            # Cursor is parked on BAG after any item trip — re-home to FIGHT so no later
            # blind A can re-open the bag (the Route-13 fisherman Super-Potion loop).
            self._rehome_fight_cursor()
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
        # HEAL-CONSUME-FAILED LATCH (NS#12): a prior in-battle item use for THIS mon already proved it
        # won't consume this battle (the bag USE/CANCEL non-consume wedge) — re-offering just re-opens the
        # bag and re-wedges the turn -> the unfleeable-trainer livelock. Skip the whole item instinct for
        # this mon; it fights/faints and the next mon (the ace) resolves the battle. Fail-safe & scoped:
        # cleared per battle, per species, only after a PROVEN failure (a working heal never latches).
        if HEAL_FAIL_LATCH and (ours.get("species") in self._heal_failed):
            return False
        if getattr(self, "_potion_blocked", False):
            # Still allow cure/revive/ether below — only potions are blocked after no_effect.
            pass
        frac = _hp_frac(ours)
        # 2026-08-02 LIVE: gBattleMons HP tore (looked crit) while party + HUD said FULL —
        # she opened Super Potion on the ace forever. Trust party struct over battle mons.
        _php, _pmx = self._true_active_party_hp()
        if _pmx and _php is not None:
            _party_frac = _php / _pmx
            if _party_frac >= 0.99 and frac < 0.99:
                self.log(f"   [engine] HP-TEAR GUARD: gBattleMons says {ours.get('hp')}/{ours.get('maxhp')} "
                         f"but party active is FULL {_php}/{_pmx} — trusting party (no potion)")
                frac = _party_frac
            elif abs(frac - _party_frac) >= 0.20:
                self.log(f"   [engine] HP-TEAR GUARD: battle frac={frac:.2f} vs party "
                         f"{_php}/{_pmx} ({_party_frac:.2f}) — trusting party")
                frac = _party_frac
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
        # Status of the mon actually OUT — decoded HERE (above the potion branch) so the
        # right-sized potion picker can prefer a Full Restore on a hurt+statused ace.
        # gBattleMons decode CROSS-CHECKED against the party struct (STATUS-TEAR GUARD,
        # 2026-08-03: torn status1 decoded POISON on a PARALYZED Blastoise -> Antidote
        # confirmed into 'It won't have any effect.' forever).
        status = _decode_status((state.get("ours") or {}).get("status1", 0) or 0)
        _pstatus, _pok = self._true_active_party_status()
        if _pok and _pstatus != status:
            self.log(f"   [engine] STATUS-TEAR GUARD: gBattleMons says {status!r} but the party "
                     f"struct says {_pstatus!r} — trusting party (anti wrong-medicine loop)")
            status = _pstatus
        if (frac <= heal_frac and not finishable
                and not getattr(self, "_potion_blocked", False)):
            # ACE-FIRST POTION ECONOMY (2026-07-31, the Misty chalk Jonny watched): the aim is
            # always the ACTIVE mon (correct — never a bench row), but after the ace faints the
            # game FORCE-SWITCHES fodder in, and this offer then spent the whole potion stock
            # keeping L8-13 bench mons alive while the L28 carry lay dead (no Revive in the bag
            # that early = no counter-offer). A real player never potions fodder: if the active
            # mon is out-leveled by a party member (alive OR fainted) by 8+, the heals are the
            # ACE's — suppress the offer LOUD and let the fodder fight/faint. The ace (or any
            # mon within 8 of the top) still heals exactly as before.
            _fodder = False
            try:
                _top_lv = 0
                for _pi in range(min(self.b.rd8(ram.GPLAYER_PARTY_CNT) or 0, 6)):
                    if st.read_party_species(self.b, _pi):
                        _top_lv = max(_top_lv, self.b.rd8(ram.GPLAYER_PARTY + _pi * 100 + 0x54))
                _fodder = bool(_ours_lv and _top_lv and (_top_lv - _ours_lv) >= 8)
            except Exception:
                _fodder = False
            if _fodder:
                self.log(f"   [engine] ACE-FIRST POTIONS: active L{_ours_lv} is fodder next to the "
                         f"team's L{_top_lv} carry — NOT offering a heal on it (the potions are "
                         f"for the ace, not L8-13 bench mons)")
            else:
                _missing = max(0, int(round((1.0 - frac) * (ours.get("maxhp") or 0))))
                heal = self._pick_heal_item(_missing, status)
                if heal is not None:
                    plan["use_potion"] = (heal, aim)
                    offers["use_potion"] = (f"use {ITEM_QTY_NAMES.get(heal, 'a healing item')} — "
                                            f"you're at {ours['hp']}/{ours['maxhp']} HP, about to "
                                            f"faint, and it's sized to the damage")
        elif finishable and frac <= heal_frac:
            self.log(f"   [engine] FINISH-THE-FOE: foe at {int(foe_frac*100)}% (<=25%), us {int(frac*100)}% "
                     f"(> crit) -> no heal, land the KO instead")
        # CURE TIMING (2026-08-03 13:04 'she doesn't know WHEN to use antidotes'): a status
        # cure costs the turn. Against a nearly-dead foe, landing the KO beats curing —
        # UNLESS the status stops her from acting at all (sleep/freeze; full-para is a
        # gamble but she can still move). Poison/burn chip is survivable for one finishing hit.
        _cure_now = status and not getattr(self, "_cure_blocked", False)
        if _cure_now and finishable and status not in ("sleep", "freeze"):
            self.log(f"   [engine] CURE TIMING: {status} can wait — foe at {int(foe_frac*100)}% "
                     f"is one hit from down; KO first, cure after (or the Center does it free)")
            _cure_now = False
        if _cure_now:
            cure = self._STATUS_CURE_for(status)
            # HURT + STATUSED = the Full Restore case (one turn fixes both). Only when
            # genuinely hurt — a Full Restore on a scratched-but-paralyzed mon is a waste.
            if frac <= 0.65 and self._items_count(19) > 0:
                cure = 19
            if cure is not None:
                plan["use_cure"] = (cure, aim)
                offers["use_cure"] = (f"use {ITEM_QTY_NAMES.get(cure, 'the cure')} for {status} — "
                                      f"it's hurting you and you have the item")
        elif status and getattr(self, "_cure_blocked", False):
            self.log(f"   [engine] CURE-BLOCK: status={status} but cures latched off this battle "
                     f"(anti Awakening re-open loop)")
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
        # REFUSAL-PROVEN FAMINE (2026-08-03 12:07): the PP bytes can lie (frozen-RAM battles
        # read pp>0 on slots the game refuses every turn), so the RAM famine gate above can
        # miss the exact fight that needs the Ether most. If the GAME ITSELF has refused
        # confirms past the futility floor, the famine is proven by behavior — offer it.
        _refusal_famine = (getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX
                           and sum(1 for v in getattr(self, "_move_refused", {}).values()
                                   if v >= 2) >= 1)
        if "use_ether" not in plan and _refusal_famine:
            ether = next((i for i in _ETHER_ITEMS_PREF if self._items_count(i) > 0), None)
            if ether is not None:
                plan["use_ether"] = (ether, aim)
                offers["use_ether"] = ("restore PP with your Ether — the game keeps refusing your "
                                       "attacks (out of PP); this is the only way to keep hitting")
        if not offers:
            return False
        offers["keep_fighting"] = "keep attacking — push through it"
        ctx = {"hp": f"{ours['hp']}/{ours['maxhp']}", "status": status or "none",
               "foe": st.SPECIES_NAME.get(state["enemy"]["species"], "the foe")}
        if "use_ether" in plan:
            ctx["pp"] = ("OUT OF PP — the game is refusing your attacks; you CANNOT keep "
                         "attacking without restoring PP or switching out")
        self.log(f"   [engine] ITEM-INSTINCT offer: {list(offers)} ctx={ctx}")
        # FORCED PICK (12:07 logs): her oracle chose 'keep_fighting' FIFTEEN times in a row
        # ("saving the Ether for Koga") while zero of her attacks could fire — the persona
        # can flavor the line, but it doesn't get to veto physics. Refusal-proven famine +
        # an Ether in the bag = the Ether gets used, no vote.
        if "use_ether" in plan and _refusal_famine:
            pick = "use_ether"
            self.log("   [engine] ITEM-INSTINCT FORCED -> use_ether (refusal-proven famine; "
                     "oracle bypassed — she cannot attack at all)")
            self.emit("okay, no more juice in my moves — Ether time, no debate.", beat=True, tier=1)
        # FORCED POTION (2026-08-03 12:42 — the crucial gym-road battle lost with potions in
        # the bag): a potion offer only exists when the active mon is genuinely hurt AND not
        # fodder next to the carry (ace-first economy already filtered that). If it's ALSO at
        # the hard crit floor, one more hit ends the fight — that heal is physics, not a
        # persona choice. The oracle keeps its vote in the 30-50% early-heal comfort zone;
        # below the floor the potion just happens.
        elif "use_potion" in plan and frac <= BATTLE_CRIT_FRAC:
            pick = "use_potion"
            self.log(f"   [engine] ITEM-INSTINCT FORCED -> use_potion (active at {int(frac*100)}% "
                     f"<= crit floor {int(BATTLE_CRIT_FRAC*100)}% — a faint loses the fight; "
                     f"oracle bypassed)")
            self.emit("nope, I'm about to drop — potion FIRST, pride later.", beat=True, tier=1)
        else:
            pick = self.choose("battle_item", offers, ctx)
        if pick and pick in plan:
            item, kind = plan[pick]
            self.log(f"   [engine] ITEM-INSTINCT pick -> {pick} (item {item}, aim={kind})")
            res = self.use_item_in_battle(item, target=kind)
            if res != "used" and HEAL_FAIL_LATCH:
                # PROVEN non-consume (the bag USE/CANCEL wedge, or a genuinely no-effect item): latch this
                # mon OFF for the rest of the battle so we don't re-open the bag and re-wedge next turn.
                sp = ours.get("species")
                if sp:
                    self._heal_failed.add(sp)
                    self.log(f"   [engine] HEAL-FAIL LATCH: item {item} did not consume ({res}) -> "
                             f"suppressing further in-battle item offers for species {sp} this battle "
                             f"(anti bag-USE/CANCEL livelock — fight/faint, let the next mon resolve it)")
                # Full-ace / no_effect: block ALL potions this battle (species latch alone didn't
                # stop the loop when gBattleMons kept lying about a different "hurt" reading).
                if pick == "use_potion" or (res == "no_effect" and pick == "use_potion"):
                    self._potion_blocked = True
                    self.log("   [engine] POTION-BLOCK: no more heal-item offers this battle "
                             "(full-HP / no_effect abort)")
                if pick == "use_cure" or (res == "no_effect" and pick == "use_cure"):
                    self._cure_blocked = True
                    self.log("   [engine] CURE-BLOCK: no more status-cure offers this battle "
                             "(already-clear / no_effect abort)")
            return res == "used"
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

    def _pick_heal_item(self, missing_hp, status=None):
        """RIGHT-SIZED POTION (2026-08-03 13:04 'she doesn't know WHEN to use super potions...
        I want her OP'): Bulbapedia-nerd item economics. Hurt AND statused -> Full Restore
        (one turn fixes both, the endgame play). Otherwise the SMALLEST potion that covers
        the missing HP — a Potion on a 15-HP dent, a Hyper on a 150-HP crater — so the big
        bottles survive for the fights that need them. Nothing covers it fully -> the biggest
        bottle in the bag (max value for the turn). None = bag has no heals at all."""
        if status and self._items_count(19) > 0:
            return 19
        for iid in (13, 22, 21, 20, 19):                    # smallest sufficient tier
            if self._items_count(iid) > 0 and _POTION_HEALS[iid] >= missing_hp:
                return iid
        for iid in (19, 20, 21, 22, 13):                    # nothing covers -> biggest available
            if self._items_count(iid) > 0:
                return iid
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

    def _party_menu_cb2(self):
        """gMain.callback2 says the PARTY/target-select screen owns input right now. RAM truth
        the frozen-frame disease can't touch (the Revive insta-click class). Fail-closed."""
        try:
            return self.b.rd32(ram.GMAIN_CB2) in _CB2_PARTY_MENU
        except Exception:
            return False

    def _bag_menu_cb2(self):
        """gMain.callback2 says the BAG list owns input right now. Fail-closed."""
        try:
            return self.b.rd32(ram.GMAIN_CB2) in _CB2_BAG_MENU
        except Exception:
            return False

    # The BAG SCREEN (layer 8, the caterpie-7/40 wedge, frame stage_l8.png): an in-battle item flow
    # can leave/return the battle to the open bag, and EVERY state byte lies there (MENU_MODE reads a
    # stale 2, GBATTLE_MENU_UP a stale 1, and the USE/CANCEL sub-box lights the white-panel pixels) —
    # so the turn loop "picked moves" into USE/CANCEL forever. Pixel truth: the item LIST PANEL is a
    # pale yellow (r,g>240, 180<b<230) no battle screen has — 3/3 on the wedge frame, 0/3 on
    # battle/party/overworld/gym/cave/Center fixtures. Panel points sit clear of the header plate
    # (whose hue varies per pocket) so this reads True for ANY pocket.
    # NS#12 (the Route-10/Rock-Tunnel bag-USE/CANCEL livelock): a SHORT item list (3 rows: e.g.
    # NUGGET / SUPER POTION / CANCEL, with the "ITEM selected -> USE/CANCEL" sub-box up) tucks a
    # dark row-gap / border under the calibrated (160,30) point, so only 2/3 hit and _bag_screen read
    # FALSE -> the turn-top close (run loop) + _close_bag_screen believed the bag was gone and never
    # B-dismissed the sub-box -> every "pick a move" press landed in the still-open bag -> anti-wedge
    # abort -> travel RE-ENTERS the same unfleeable trainer battle -> infinite livelock (frame-proof:
    # bwedge_antiwedge_trainer, forensics bag=False on an open bag). FIX: sample MORE panel-interior
    # points and require >=3 of them. The interior is reliably pale-yellow (r,g>240, 180<b<230) at
    # many points, so a short list still scores >=3 while a NON-bag screen stays at 0 — pure-white
    # menus/movelists fail the b<230 tint test, blue text boxes fail r,g>240, so no false B-drains.
    _BAG_PTS = ((160, 30), (200, 60), (120, 10), (150, 20), (200, 30), (180, 45))

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

    def _menu_up(self):
        """GBATTLE_MENU_UP (0x02023E86): 1 = action menu (FIGHT/BAG/POKEMON/RUN), 0 otherwise.
        Verified signal (firered_ram) — NOT the free-running GBATTLE_PHASE counter."""
        try:
            return self.b.rd8(ram.GBATTLE_MENU_UP) == 1
        except Exception:
            return False

    def _at_action_menu(self):
        """TRUE FIGHT/BAG/POKEMON/RUN menu. menu_up==1 is authoritative; _white_box alone is
        shared with the move list (the thrash root — treating both as 'action' then probing)."""
        if self._bag_screen() or self._party_screen():
            return False
        return self._menu_up() and self._white_box()

    def _at_move_list(self):
        """FIGHT move list is up. menu_up==0 + white panel (or MENU_MODE==2). Never d-pad probe."""
        if self._bag_screen() or self._party_screen() or self._menu_up():
            return False
        try:
            if self.b.rd8(MENU_MODE) == 2:
                return True
        except Exception:
            pass
        return self._white_box() and self._in_move_list()

    def _home_to_fight(self):
        """NO-OP on this libmgba core (2026-08-02 LIVE ROOT CAUSE).

        firered_ram CATCH-ARC finding: the FIRST d-pad press at the action menu CONFIRMS FIGHT
        (menu_up 1→0, opens the move list). UP/LEFT 'homing' was the stream look — endless
        Fight↔move / Fight↔Bag theater. Default cell IS FIGHT; just press A when menu_up==1."""
        return

    def _nav_move(self, idx):
        """Move the move-list cursor from slot 0 (where the list opens) to slot idx in the 2x2
        grid: TL=0 TR=1 / BL=2 BR=3 (RIGHT = column, DOWN = row). Settles after so the confirm-A
        isn't eaten mid cursor-move (the slot-2 lesson). ONLY call when _at_move_list()."""
        if idx == 1:
            self._tap("RIGHT")
        elif idx == 2:
            self._tap("DOWN")
        elif idx == 3:
            self._tap("RIGHT"); self._tap("DOWN")
        self._wait(14)

    def _movelist_open(self):
        """True iff the FIGHT move list is open — menu_up==0 path, never a d-pad probe."""
        return self._at_move_list()

    def _movelist_open_verified(self):
        """Alias of _at_move_list — kept for call sites. Zero taps. Ever."""
        return self._at_move_list()

    def _goto_move(self, idx, tries=12):
        """Park the move-list cursor on slot idx. 2026-08-03 11:43 (the third parked-on-Tackle
        photo set): d-pad walking this cursor was observed NOT MOVING IT for minutes at a time
        (eaten presses and/or a read that already claimed arrival) — every confirm fired the
        drawn slot (Tackle, 0 PP) forever. The action cursor had the IDENTICAL disease and the
        proven cure is a RAM WRITE + readback (_poke_action_cursor, verified live): the game's
        confirm reads gMoveSelectionCursor — the byte IS the selection, whatever the pixels
        show. Write first; d-pad walk only as fallback; return False LOUD (callers attribute
        refusals to the cursor byte, never assume arrival)."""
        try:
            if self.b.rd8(MOVE_CURSOR) != idx:
                self.b.core.memory.u8.raw_write(MOVE_CURSOR, int(idx) & 0xFF)
                self._wait(2)
            if self.b.rd8(MOVE_CURSOR) == idx:
                return True
        except Exception as e:
            self.log(f"   [engine] move-cursor write failed: {e} (falling back to d-pad)")
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

    def _blind_goto_move(self, idx):
        """Move-list navigation with ZERO RAM trust (2026-08-03: the frozen-detector battle —
        MENU_MODE/cursor bytes desynced for a whole fight; every readback-based layer failed
        while the real list sat open on screen). The FRLG move grid is 2x2 with NO wraparound,
        so clamping makes doubled presses idempotent: LEFT,LEFT,UP,UP homes to slot 0 from any
        start even if half the presses are eaten; DOWN,DOWN / RIGHT,RIGHT then reach any slot
        deterministically. Safe everywhere: on a text box d-pads are no-ops; on the action menu
        the worst case is confirming FIGHT (which opens the list this walk then navigates)."""
        for d in ("LEFT", "LEFT", "UP", "UP"):
            self._tap(d)
            self._wait(6)
        steps = {0: (), 1: ("RIGHT", "RIGHT"), 2: ("DOWN", "DOWN"),
                 3: ("DOWN", "DOWN", "RIGHT", "RIGHT")}.get(int(idx) & 3, ())
        for d in steps:
            self._tap(d)
            self._wait(6)

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
        _our_types = [t for t in (ours.get("types") or []) if t and t != "???"]
        idx, desc, low = pol.choose_move(
            ours["moves"], enemy["types"], _hp_frac(ours), our_types=_our_types)

        def _usable(i):                                # a real move with PP the game hasn't disproven
            m = ours["moves"][i]
            # A slot the GAME refused >=2 times is dry whatever its (tear-prone) PP byte says —
            # the Tackle re-fire loop: stale pp>0 + war-must-advance streak-clear = forever.
            if self._move_refused.get(i, 0) >= 2:
                return False
            return m.get("id", 0) != 0 and m.get("pp", 0) > 0

        def _dmg_score(i):
            # STAB × type × accuracy — same yardstick as choose_move (2026-08-02).
            return pol.move_score(ours["moves"][i], enemy["types"], _our_types)

        if not (0 <= idx < 4) or not _usable(idx) or idx in self._skip_streak:
            # FIX 1 — REPETITION-AVERSE move pick: exclude EVERY move that already failed to fire this
            # streak (not just the last one), so she pivots through her whole moveset and NEVER re-spams
            # a dead/0-PP/blocked move (the Mankey case: she had 3 unused moves). Pick the best one she
            # HASN'T tried yet by expected damage. The streak clears the instant any move fires (below),
            # so a working move is never permanently benched (the PoisonPowder-spam lesson).
            cands = [i for i in range(4) if _usable(i) and i not in self._skip_streak]
            if cands:
                idx = max(cands, key=_dmg_score)
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
                        _dmg_score(i)))
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
            if not _usable(i) or i in self._skip_streak:
                return False
            return not (m.get("power", 0) > 0 and _eff(m, enemy) == 0)
        if 0 <= idx < 4 and ours["moves"][idx].get("power", 0) > 0 and eff == 0:
            _uc = [i for i in range(4) if _useful(i)]
            if _uc:
                idx = max(_uc, key=_dmg_score)
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
        # LOUD scoreboard of all 4 FIGHT slots (TL=0 TR=1 BL=2 BR=3) so a missed STAB
        # / wrong-column pick is visible in the soak without a frame dump.
        try:
            _board = []
            for _i, _m in enumerate(ours["moves"]):
                if not _m.get("id"):
                    continue
                _sc = pol.move_score(_m, enemy["types"], _our_types)
                # pp= is the LIE DETECTOR (2026-08-03): print what the decision function BELIEVES
                # each slot's PP is, next to ref= (times the game refused it). A slot with pp>0
                # and ref>=2 in the same line is gBattleMons caught lying, in the soak report,
                # with no phone photos needed.
                _board.append(
                    f"[{_i}]{_m.get('name','?')}(p{_m.get('power',0)}"
                    f"{'*STAB' if pol.stab_mult(_m.get('type'), _our_types) > 1 else ''}"
                    f",pp={_m.get('pp', '?')},ref={self._move_refused.get(_i, 0)}"
                    f",sc={_sc:g})")
            self.log(f"   [engine] action menu: {desc} -> slot {idx} (eff x{eff:g}) vs "
                     f"{st.SPECIES_NAME.get(enemy['species'], '?')} "
                     f"{enemy['hp']}/{enemy['maxhp']} | {' '.join(_board)}")
        except Exception:
            self.log(f"   [engine] action menu: {desc} -> slot {idx} (eff x{eff:g}) vs "
                     f"{st.SPECIES_NAME.get(enemy['species'], '?')} {enemy['hp']}/{enemy['maxhp']}")
        # STREAM COMMIT (2026-08-02 LIVE ROOT): ZERO d-pad on the action menu.
        # menu_up==1 → A opens the move list (default cell is FIGHT). Already on move list →
        # nav + A. Never UP/LEFT/RIGHT at action (those CONFIRM FIGHT / scroll forever on this core).
        self.log(f"   [engine] STREAM COMMIT: {desc} slot {idx} "
                 f"(menu_up={int(self._menu_up())} action={self._at_action_menu()} "
                 f"moves={self._at_move_list()})")
        if (self._bag_screen() or self._party_screen()
                or self._bag_menu_cb2() or self._party_menu_cb2()):
            self.b.press("B", self.hold, self.hold, self.render, owner=self.owner); self._wait(12)
        if self._at_action_menu():
            # FORCE FIGHT FIRST (2026-08-04 LIVE, the mid-fight 'Teachy TV / Helix Fossil'
            # inspections Jonny watched): the action-cursor byte PERSISTS wherever the last
            # flow parked it (BAG after an item turn, POKéMON after a switch prompt) — the
            # 'default cell is FIGHT' assumption below only holds on a fresh battle. A@BAG
            # opens the bag on the REMEMBERED pocket (Key Items after any TeachFlow errand),
            # and the blind move-walk then d-pads INSIDE the bag and inspects key items for
            # whole turns. Same cure as the move list: the byte IS the selection — write it.
            try:
                if self.b.rd8(ram.GBATTLE_ACTION_CURSOR) != ram.ACT_FIGHT:
                    self._poke_action_cursor(ram.ACT_FIGHT)
            except Exception:
                pass
            # ONE A — opens move list. Do NOT d-pad first.
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)
            # STRAY-BAG RESCUE: the A opened the BAG/party anyway (the byte lied or the write
            # missed) — B out once, re-force FIGHT, re-open. One bounded lap, never a loop;
            # a still-open bag after this falls through to the verify loop's failure branch.
            if (self._bag_screen() or self._party_screen()
                    or self._bag_menu_cb2() or self._party_menu_cb2()):
                self.log("   [engine] STREAM COMMIT: A opened the BAG/PARTY instead of FIGHT "
                         "(parked action cursor) — B out + re-forcing FIGHT")
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(16)
                try:
                    self._poke_action_cursor(ram.ACT_FIGHT)
                except Exception:
                    pass
                if self._at_action_menu():
                    self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(16)
        elif not self._at_move_list() and self._white_box():
            # Ambiguous white — try A once (opens fight or confirms if already on list).
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(16)
        # BLIND-FIRST NAV (2026-08-03 12:07 relapse): the RAM-verified path re-poisons the
        # whole battle the moment the cursor byte lies while the detector says fine — the
        # write-then-readback "verifies" against dead memory, A fires the parked slot, and
        # the refusal is tallied on the slot we INTENDED (ledger poisoning, round 3). The
        # blind walk is deterministic with ZERO RAM trust (2x2 grid, no wrap, doubled taps
        # clamp), so it is now the PRIMARY nav on every commit; the RAM write runs after
        # only as a correction when it can still be read (healthy RAM: both agree, no-op).
        self._blind_goto_move(idx)
        if self._at_move_list() and self.b.rd8(MOVE_CURSOR) != idx:
            self.log(f"   [engine] STREAM COMMIT: RAM cursor disagrees after blind walk "
                     f"(reads {self.b.rd8(MOVE_CURSOR)}, want {idx}) — RAM-correcting")
            self._goto_move(idx)
        # ATTRIBUTION (2026-08-03 12:07, blind-first era): the blind walk is deterministic —
        # the real cursor IS on `idx` regardless of what the byte claims (frozen-RAM battles
        # read garbage here; trusting the byte was ledger-poisoning round 3). Attribute
        # refusals to idx. A disagreeing byte is now a frozen-RAM SIGNAL, not the truth.
        _pressed = idx
        _byte = self.b.rd8(MOVE_CURSOR)
        if self._at_move_list() and 0 <= _byte < 4 and _byte != idx:
            self.log(f"   [engine] !! CURSOR BYTE DISAGREES post-blind-walk (byte={_byte}, "
                     f"blind={idx}) — RAM block likely frozen; trusting the blind walk")
        # FORCE-WRITE THE SELECTION (2026-08-04 LIVE, the Silph 'Tackle theater': engine chose
        # Water Pulse every turn vs Pidgeot while TACKLE's PP drained 30->26 — eaten blind-walk
        # taps left the REAL cursor on slot 0, the byte read 0 truthfully, and the 12:07 rule
        # above trusted the walk over the byte, so A confirmed Tackle for a whole fight).
        # The game's confirm reads gMoveSelectionCursor — the byte IS the selection (the
        # _goto_move doctrine, action-cursor-proven). A raw WRITE lands on real memory even
        # when the READ path is frozen, so writing idx right before A is correct in BOTH
        # failure classes: eaten-taps (fixes the desync) and frozen reads (harmless no-op
        # on an already-right cursor). Best-effort; a write fault never blocks the confirm.
        try:
            self.b.core.memory.u8.raw_write(MOVE_CURSOR, int(idx) & 0xFF)
            self._wait(2)
        except Exception as _wf:
            self.log(f"   [engine] move-cursor force-write failed ({_wf}) — blind walk only")
        pp0 = ours["moves"][idx].get("pp", 0) if 0 <= idx < 4 else 0
        # Full PP snapshot for the post-turn WRONG-SLOT AUDIT below (menu-time state).
        _pp_all0 = [int((m or {}).get("pp", 0) or 0) for m in (ours.get("moves") or [])]
        before = self._bstate()
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner); self._wait(10)
        self._last_desc, self._last_eff = desc, eff
        result = None
        last_hp, stable = before, 0
        for _vn in range(600):                        # was 900 — shorter wait, faster retry if miss
            if not st.in_battle(self.b):
                result = "done"; break
            # GAME-TEXT REFUSAL (2026-08-03, the Tackle spam): "There's no PP left..." /
            # "X is disabled!" is the game itself vetoing this slot — exile INSTANTLY
            # (don't burn the 600-frame timeout twice before the famine switch can fire).
            if _vn % 12 == 0:
                _bt = self._battle_text()
                if _bt and any(sn in _bt for sn in self._MOVE_REFUSAL_SNIPPETS):
                    self._move_refused[_pressed] = max(self._move_refused.get(_pressed, 0), 2)
                    self._amove_futile = getattr(self, "_amove_futile", 0) + 1
                    self.log(f"   [engine] GAME REFUSED slot {_pressed} ({_bt[:48]!r}) -> instant "
                             f"exile this battle (text is truth; PP byte was lying; "
                             f"futility {self._amove_futile}/{FUTILE_AMOVE_MAX})")
                    break                             # result stays None -> failure branch
            cur = st.read_battle(self.b)
            if cur:
                self._emit_diffs(self._prev, cur); self._prev = cur
                try:
                    if 0 <= idx < 4 and cur["ours"]["moves"][idx].get("pp", 0) < pp0:
                        result = "done"; break
                except Exception:
                    pass
                hp = (cur["enemy"]["hp"], cur["ours"]["hp"])
                if before and hp != before:
                    # WRONG-SLOT AUDIT (2026-08-04, the Silph Tackle theater): an HP change
                    # proves A turn ran — NOT that the CHOSEN move ran. When the cursor
                    # desynced, Tackle fired, HP moved, this branch said 'done', and the
                    # wrong-fire stayed INVISIBLE for a whole fight. Compare the PP vector:
                    # a different slot consumed while the chosen one didn't = the confirm
                    # landed elsewhere. The force-write above should make this extinct;
                    # if it still fires, it's LOUD evidence (never a silent pass again).
                    try:
                        _ppv = [int((m or {}).get("pp", 0) or 0)
                                for m in (cur["ours"].get("moves") or [])]
                        _drop = [j for j, (a, c) in enumerate(zip(_pp_all0, _ppv)) if c < a]
                        if _drop and idx not in _drop:
                            self._wrong_fires = getattr(self, "_wrong_fires", 0) + 1
                            self.log(f"   [engine] !! WRONG SLOT FIRED: chose {idx} but slot "
                                     f"{_drop[0]} was consumed (cursor desync survived the "
                                     f"force-write; wrong-fires {self._wrong_fires}) — LOUD")
                    except Exception:
                        pass
                    result = "done"; break
                stable = stable + 1 if hp == last_hp else 0
                last_hp = hp
                # Back at ACTION menu (menu_up==1) with no change = turn didn't leave.
                if self._at_action_menu() and stable >= 20:
                    break
                # Parked back at the MOVE LIST with no change = the game REFUSED the pick
                # ("no PP left" bounce). Don't burn the whole 600-frame budget per refusal —
                # that made the (working) exile->famine->switch ladder take minutes on stream
                # (2026-08-03 08:56: "the exact same loop"). Bail fast; the failure branch
                # counts the refusal.
                if self._at_move_list() and stable >= 30:
                    break
            # Advance text without B while white (B flees / closes menus).
            if not self._white_box():
                self.b.press("A", 2, 8, self.render, owner=self.owner)
            self.b.run_frame(); self.render()
        if result == "done":
            self._skip_streak.clear()
            self._immob_streak = 0
        else:
            try:
                cur = st.read_battle(self.b)
                st1 = (cur or {}).get("ours", {}).get("status1", 0)
            except Exception:
                st1 = 0
            _slp_frz = st1 & 0x27
            _par = st1 & 0x40
            _ims = getattr(self, "_immob_streak", 0)
            # PARALYSIS MASK (2026-08-03 08:50 live — "paralyzed + no PP + won't swap"): a MENU
            # REFUSAL ("There's no PP left...") bounces straight back to the MOVE LIST — the turn
            # never ran, the foe never acted. A real immobilization RUNS the turn (animations,
            # foe attacks) and never parks us back on the move list. The old classifier called
            # every non-fire on a paralyzed mon "immobilization" for 6 laps, so NO refusal was
            # counted, NO exile, NO famine switch — 70+ seconds of dead-move spam while the
            # rescue sat armed. Back-at-the-move-list = refusal, whatever the status byte says.
            _at_list = self._at_move_list()
            # BOUNDED for ALL statuses (09:04 photos: the 08:54 session STILL emitted 'fully
            # paralyzed' repeats — the at-list probe can miss while the refusal box is up, and
            # sleep/freeze had NO cap, so a lying status byte could eat laps forever). Six
            # zero-change "immobilizations" in a row is not a thing a real battle produces;
            # past that every non-fire counts as a refusal and feeds the futility breaker.
            _real_immob = pp0 > 0 and not _at_list and (_slp_frz or _par) and _ims < 6
            if _real_immob:
                self._immob_streak = _ims + 1
                # An "immobilization" here means the 600-frame verify saw ZERO change (a real
                # full-para turn almost always moves HP — the foe attacks). It's evidence-free
                # stillness, so it ALSO feeds the futility breaker (soak 091711: WP/Bite immob
                # emits interleaved with refusal laps for minutes — the breaker must converge
                # on total dead laps, not on how each lap got classified).
                self._amove_futile = getattr(self, "_amove_futile", 0) + 1
                why = "asleep" if st1 & 0x07 else ("frozen" if st1 & 0x20 else "fully paralyzed")
                self.log(f"   [engine] turn resolved by IMMOBILIZATION ({why}) — not a dead move; "
                         f"fighting on (futility {self._amove_futile}/{FUTILE_AMOVE_MAX})")
                self.emit(f"no — {desc} didn't happen, I'm {why}! hang in there…", beat=True, tier=1)
                return "done"
            # DO NOT reset _immob_streak here (2026-08-03 11:24 forensics): refusal laps and
            # immob laps INTERLEAVE (WP-immob, Bite-refusal, WP-immob...), and the old reset
            # re-armed the immob classifier every time a refusal lap landed between — the 6-lap
            # cap never engaged and 'fully paralyzed' theater ran for minutes. The streak only
            # resets when a move REALLY fires (the result=='done' branch above).
            self._skip_streak.add(idx)
            self._move_refused[_pressed] = self._move_refused.get(_pressed, 0) + 1
            self._amove_futile = getattr(self, "_amove_futile", 0) + 1
            self.log(f"   [engine] move slot {_pressed} didn't fire (0-PP / disabled / blocked) -> "
                     f"rotating to an untried move (streak now {sorted(self._skip_streak)}, "
                     f"refusals {self._move_refused}, futility {self._amove_futile}/{FUTILE_AMOVE_MAX})")
            # Always 'done' to the outer loop — never 'stuck' (stuck re-settles and re-probes).
            return "done"
        return result or "done"

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

    def _party_blind_goto(self, target):
        """Best-effort party-list walk without trusting pixel/RAM cursors: LEFT homes the lead
        panel, RIGHT enters the right column, DOWN*(target-1) lands the row. Used when orange-
        border readback can't see the selection (half-dead benches, fade frames)."""
        self._tap("LEFT"); self._wait(12)
        if target == 0:
            return True
        self._tap("RIGHT"); self._wait(12)
        for _ in range(max(0, int(target) - 1)):
            self._tap("DOWN"); self._wait(12)
        return True

    def _force_switch(self):
        """Lead fainted with a healthy reserve -> the 'Choose a POKéMON' party menu is up.
        Walk to a HEALTHY row and confirm SEND OUT. CRITICAL (2026-08-02 gym chalk): never press
        the confirm-A unless the SEND OUT submenu is actually open — A on a corpse just loops
        "X has no energy left to battle!" for 60–180s while 1–2 bench mons are still alive.
        Returns True once a healthy mon is active.

        2026-08-02 docks chalk: pixel submenu false-negatives + 10 retries still looked like a
        60–90s party scroll. Hard wall-clock budget; on expiry do ONE blind send of the strongest
        live row (LEFT/RIGHT/DOWN*n, A, A) and return."""
        if self._healthy_reserve_slot() is None:
            return False
        # Only legal opener of the POKEMON menu this battle (voluntary paths are banned).
        self._allow_pokemon_menu = True
        _skip_rows = set()                                # rows that refused SEND OUT this menu
        _tried_sp = set()                                 # species that got submenu but didn't swap
        _t0 = time.time()
        try:
            return self._force_switch_inner(_skip_rows, _tried_sp, _t0)
        finally:
            self._allow_pokemon_menu = False

    def _force_switch_inner(self, _skip_rows, _tried_sp, _t0):
        for _attempt in range(10):
            if time.time() - _t0 >= FSWITCH_BUDGET_S:
                self.log(f"   [engine] fswitch: BUDGET {FSWITCH_BUDGET_S:.0f}s hit — "
                         f"blind-sending strongest live (anti 60–90s party theater)")
                break
            cur = st.read_battle(self.b)
            if cur and cur["ours"]["hp"] > 0:
                self._note_battle_progress("force-switch seated")
                return True                               # a healthy mon is active -> switched
            self._wait(10)                                # let the party menu settle
            if not self._party_screen():
                self._advance_text()                      # faint text still playing -> drain a beat
                continue
            # Sub-menu or "no energy" box still up from a prior miss — B it clear FIRST.
            if self._party_submenu():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(14)
            # LIST FOCUS first (sub-menu/message tap-eaters), then resolve the target row by
            # CONTENT at menu time — the order law: row i IS gPlayerParty[i] only while the
            # menu is open; any slot picked before it opened is in a different order.
            if not self._party_focus():
                self.log("   [engine] fswitch: party list never regained focus -> retry")
                # Don't burn the whole budget on focus — blind-B and continue.
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                continue
            rows = self._menu_rows()
            live = [r for r in rows if r["hp"] > 0 and r["row"] not in _skip_rows
                    and r["species"] not in _tried_sp]
            # Prefer the right-column bench (row>0); allow row 0 only if it's the sole survivor.
            bench = [r for r in live if r["row"] > 0]
            cands = bench if bench else live
            if not cands:
                _skip_rows.clear()
                _tried_sp.clear()
                cands = [r for r in rows if r["hp"] > 0 and r["row"] > 0] or \
                        [r for r in rows if r["hp"] > 0]
                if not cands:
                    return False                          # nothing standing on the bench
            tgt = max(cands, key=lambda r: r["level"])     # send the strongest thing standing
            # Prefer RAM party cursor match over pixel border when available.
            _ram_cur = self.b.rd8(PARTY_CURSOR)
            if _attempt >= 1:
                self.log(f"   [engine] fswitch retry {_attempt}: target row {tgt['row']} "
                         f"sp={tgt['species']} skip_rows={sorted(_skip_rows)} "
                         f"party_cur={_ram_cur} menu_rows="
                         f"{[(r['species'], r['hp']) for r in rows]}")
            reached = self._party_goto_slot(tgt["row"])
            if not reached and self.b.rd8(PARTY_CURSOR) != tgt["row"]:
                self.log(f"   [engine] fswitch: border goto missed row {tgt['row']} "
                         f"(cursor={self._party_cursor_slot()} ram={self.b.rd8(PARTY_CURSOR)}) "
                         f"-> blind walk")
                self._party_blind_goto(tgt["row"])
            # Re-check content at the row we're about to A — never knowingly pick a corpse.
            rows = self._menu_rows()
            if tgt["row"] >= len(rows) or rows[tgt["row"]]["hp"] <= 0:
                self.log(f"   [engine] fswitch: row {tgt['row']} is dead at confirm — skip")
                _skip_rows.add(tgt["row"])
                continue
            self.log(f"   [engine] fswitch: selecting row {tgt['row']} "
                     f"(sp={tgt['species']} L{tgt['level']} hp={rows[tgt['row']]['hp']})")
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)  # select mon
            submenu = False
            for _ in range(12):                           # WAIT for the SEND OUT sub-menu
                if self._party_submenu():
                    submenu = True
                    break
                # RAM-cursor path: if we selected a live row and party cursor stuck on it,
                # a second A often IS Send Out even when pixels miss the submenu.
                self._wait(6)
            if not submenu:
                # Live-row second chance: one confirm-A if RAM cursor still on the live target
                # (pixel submenu false-negative). Corpse rows stay skipped (hp check above).
                if self.b.rd8(PARTY_CURSOR) == tgt["row"] and rows[tgt["row"]]["hp"] > 0:
                    self.log(f"   [engine] fswitch: no pixel submenu but RAM cursor on live "
                             f"row {tgt['row']} — one SEND OUT confirm-A")
                    self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(16)
                    for _ in range(14):
                        cur = st.read_battle(self.b)
                        if cur and cur["ours"]["hp"] > 0:
                            self._note_battle_progress("force-switch seated (ram confirm)")
                            return True
                        self._advance_text()
                        self._wait(6)
                # CORPSE / cursor-miss: "has no energy left" — do NOT keep confirm-A (that was the
                # 60–180s loop). Skip this row, B-dismiss the message, try the next alive mon.
                self.log(f"   [engine] fswitch: no SEND OUT submenu after A on row {tgt['row']} "
                         f"sp={tgt['species']} — corpse/miss, skipping (NOT confirm-A)")
                _skip_rows.add(tgt["row"])
                for _ in range(5):
                    if self._party_submenu():
                        self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                        self._wait(12)
                        break
                    if not self._party_screen():
                        break
                    self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(12)
                continue
            # Submenu is up — confirm SEND OUT (default top row).
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(16)
            for _ in range(14):
                cur = st.read_battle(self.b)
                if cur and cur["ours"]["hp"] > 0:
                    self._note_battle_progress("force-switch seated")
                    return True
                self._advance_text()
                self._wait(6)
            # Submenu confirmed but swap didn't take — don't hammer the same species forever.
            _tried_sp.add(tgt["species"])
            self.log(f"   [engine] fswitch: SEND OUT on sp={tgt['species']} didn't seat a "
                     f"healthy active -> rotate")
        # BUDGET / attempt exhaust — ONE blind send of strongest live (stream must move).
        try:
            if self._party_screen():
                rows = self._menu_rows()
                live = [r for r in rows if r["hp"] > 0 and r["row"] > 0] or \
                       [r for r in rows if r["hp"] > 0]
                if live:
                    tgt = max(live, key=lambda r: r["level"])
                    self.log(f"   [engine] fswitch BLIND: row {tgt['row']} "
                             f"{st.SPECIES_NAME.get(tgt['species'], '?')} L{tgt['level']}")
                    self._party_blind_goto(tgt["row"])
                    self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(14)
                    self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
                    self._wait(20)
                    for _ in range(20):
                        cur = st.read_battle(self.b)
                        if cur and cur["ours"]["hp"] > 0:
                            self._note_battle_progress("force-switch seated (blind)")
                            return True
                        self._advance_text()
                        self._wait(6)
        except Exception as e:
            self.log(f"   [engine] fswitch blind path error: {e}")
        cur = st.read_battle(self.b)
        return bool(cur and cur["ours"]["hp"] > 0)

    # ── B-1: TYPE-MATCHUP AWARENESS + VOLUNTARY SWITCH (the E4-critical verb) ────
    def _goto_pokemon(self, tries=10):
        """Park the action cursor on POKEMON (ACT_POKEMON=2). Prefer RAM write over d-pad —
        DOWN from FIGHT on this core can confirm Fight and open the move list, which looks
        exactly like 'she's trying to switch but keeps checking attacks' (stream-end docks).

        2026-08-03 NUCLEAR: refused unless BATTLE_SWITCH is armed OR `_allow_pokemon_menu`
        (force-switch path). Default env is OFF — that was the Pokemon→Blastoise thrash."""
        if not BATTLE_SWITCH_ENABLED and not getattr(self, "_allow_pokemon_menu", False):
            self.log("   [engine] POKEMON menu BANNED (POKEMON_BATTLE_SWITCH=0 — "
                     "anti Blastoise thrash). Faint→force-switch still works.")
            return False
        for _ in range(tries):
            if self._at_move_list():
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                continue
            if not self._at_action_menu():
                return False
            c = self.b.rd8(ram.GBATTLE_ACTION_CURSOR)
            if c == ram.ACT_POKEMON:
                return True
            if self._poke_action_cursor(ram.ACT_POKEMON):
                return True
            if c == ram.ACT_FIGHT:
                self._tap("DOWN")
            elif c == ram.ACT_BAG:
                self._tap("DOWN"); self._tap("LEFT")
            elif c == ram.ACT_RUN:
                self._tap("LEFT")
            else:
                return False
            self._wait(3)
            if self._at_move_list():
                self.log("   [engine] _goto_pokemon: d-pad confirmed FIGHT — B out, retry write")
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
        return self._at_action_menu() and self.b.rd8(ram.GBATTLE_ACTION_CURSOR) == ram.ACT_POKEMON

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
        # EMPTY DAMAGING SET = CAN'T HIT AT ALL (2026-07-31, the Teleport-only Abra hole): the old
        # `else 1.0` scored a moveless mon as NEUTRAL, so the matchup trigger never saw a reason to
        # pull it — it sat active failing its status move forever. 0.0 reads as the hardest possible
        # resist, so trigger 1 fields a reserve that can actually fight (famine usually fires first;
        # this is the backstop when it's spent).
        best_move_eff = max(_dmg) if _dmg else 0.0
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
        # NUCLEAR: voluntary POKEMON off unless operator armed BATTLE_SWITCH (or force-switch allow).
        if not BATTLE_SWITCH_ENABLED and not getattr(self, "_allow_pokemon_menu", False):
            return False
        want_sp = st.read_party_species(self.b, slot)             # identity survives the reorder
        if not want_sp or want_sp == before_sp:
            self.log(f"   [engine] switch: refused — want_sp={want_sp} is already active "
                     f"(before={before_sp}); never 'switch Blastoise for Blastoise'")
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
            # BLIND POKEMON FALLBACK (2026-08-03 12:07: frozen-RAM battles — _goto_pokemon
            # "verified" against a dead cursor byte, A opened the MOVE LIST instead, and
            # every switch path died here). B back to the action menu, then DOWN,DOWN + A:
            # DOWN is NOT in the confirm-hazard set (only UP/LEFT/RIGHT confirm FIGHT on
            # this core) and the 2x2 grid clamps, so from the FIGHT home this deterministically
            # lands POKEMON. Worst case (cursor was in the right column) it lands RUN and the
            # game answers "Can't escape!" — drained below, harmless. _party_screen() is
            # pixel truth, so the success check works even when every RAM byte lies.
            self.log("   [engine] switch: party didn't open via RAM nav — BLIND fallback "
                     "(B out, DOWN DOWN A)")
            for _ in range(3):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
            for d in ("DOWN", "DOWN"):
                self._tap(d); self._wait(8)
            self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)
            self._wait(30)
            for _ in range(8):
                if self._party_screen():
                    break
                self._wait(8)
        if not self._party_screen() or not self._party_focus():
            self.log("   [engine] switch: party screen never took focus -> B out (fail-safe)")
            self._exit_bag()
            return False
        # NEVER row 0 — after rearrange that panel IS the mon already out. Picking it is the
        # 'switch him out for himself' loop Jonny described (alive but can't fight).
        row = next((r["row"] for r in self._menu_rows()
                    if r["species"] == want_sp and r["hp"] > 0 and r["row"] != 0), None)
        self.log(f"   [engine] switch: target party slot {slot} sp={want_sp} -> menu row {row}")
        if row is None or not self._party_goto_slot(row):
            self.log("   [engine] switch: target row unreachable (or only on active panel) "
                     "-> B out (fail-safe)")
            for _ in range(4):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                if self._white_box() and not self._party_screen():
                    break
            return False
        self.b.press("A", self.hold, self.hold, self.render, owner=self.owner)   # select -> sub-menu
        submenu = False
        for _ in range(12):
            if self._party_submenu():
                submenu = True
                break
            self._wait(6)
        if not submenu:
            # Same corpse-loop class as _force_switch: A without SEND OUT/SHIFT just eats time.
            self.log("   [engine] switch: no SHIFT submenu after select -> B out (fail-safe)")
            for _ in range(4):
                self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                self._wait(12)
                if self._white_box() and not self._party_screen():
                    break
            return False
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
        # FAIL LATCH (2026-08-02 LIVE): ONE failed party-menu attempt = stop. Party-list DOWN
        # probes look like "scrolling forever" on stream; two tries was still unwatchable.
        if getattr(self, "_switch_fail_n", 0) >= 1:
            return False
        # LEVEL-DOMINANCE (stream): Blastoise L48 vs Route-12 trash must NOT open POKEMON to
        # "optimize" — just Surf. Crushing lead = fight, no menu theater.
        try:
            _al = (state.get("ours") or {}).get("level") or 0
            _fl = (state.get("enemy") or {}).get("level") or 0
            if _al and _fl and _al >= _fl + 8 and _hp_frac(state.get("ours") or {}) > 0.35:
                return False
        except Exception:
            pass
        slot = self._best_switch_slot(state)
        if slot is None:
            return False
        self.log(f"   [engine] MATCHUP SWITCH: active is out-typed -> trying party slot {slot}")
        r = self._switch_to_slot(slot, state.get("ours", {}).get("species"))
        if r == "switched":
            self._switch_fail_n = 0
            # SAY THE TYPE MATH (2026-07-31, Jonny: "she needs to know what's good vs what — and
            # SAY it"): name who's coming in and WHY in type-chart terms, from the same reads the
            # scorer used — so the stream hears "Spearow eats bugs for breakfast", not a vague
            # "better matchup". Fail-safe: any read hiccup falls back to the plain line.
            try:
                _in_nm = st.SPECIES_NAME.get(st.read_party_species(self.b, slot), "my pick")
                _foe = state.get("enemy") or {}
                _foe_ty = "/".join(t for t in (_foe.get("types") or []) if t) or "mystery"
                _re = 0.0
                for _mid in st.read_party_moves(self.b, slot):
                    if _mid:
                        _mt, _mp = st.move_info(self.b, _mid)
                        if _mp and _mp > 0:
                            _re = max(_re, _eff({"type": _mt or "normal"}, _foe))
                if _re >= 4.0:
                    _ln = (f"{_in_nm}, you're up — that's a {_foe_ty} type and you hit it "
                           f"DOUBLE super-effective. this is just bullying.")
                elif _re >= 2.0:
                    _ln = (f"{_in_nm} eats {_foe_ty} types for breakfast — super-effective, "
                           f"switching in. the type chart never lies.")
                else:
                    _ln = (f"{_in_nm} takes this one — the {_foe_ty} typing is chewing through "
                           f"who I had out there.")
                self.emit(_ln, beat=True, tier=2)
            except Exception:
                self.emit("switching it up — this is a better matchup", beat=True, tier=2)
            self._skip_streak.clear()
            return "switched"
        self._switch_fail_n = getattr(self, "_switch_fail_n", 0) + 1
        self.log(f"   [engine] matchup switch did not confirm "
                 f"(fail {self._switch_fail_n}/1) -> fighting instead (fail-safe, no wedge)")
        return False

    def _active_pp_famine(self, state):
        """True iff the ACTIVE mon has no damaging move with PP left that can CONNECT vs the
        CURRENT foe (gBattleMons ground truth — power from ROM gBattleMoves via read_battle).
        Status-move PP doesn't count (can never KO), and neither does a type-IMMUNE damaging
        move (e4_run7 Agatha: Venusaur with only Normal-type PP vs Gengar burned ~10
        war-must-advance turns + 2 Full Restores while Persian's Bite sat on the bench —
        immune-only PP is famine vs THIS foe, and the famine switch is the only winning line).
        Unknown foe types count as connecting (never over-trigger).
        2026-08-03: a slot the GAME refused >=2 times counts as DRY whatever its (tear-prone)
        PP byte says — that's what lets the famine switch rescue the stale-PP Tackle loop."""
        mv = (state.get("ours") or {}).get("moves") or []
        return not any(m.get("id") and m.get("pp", 0) > 0 and m.get("power", 0) > 0
                       and getattr(self, "_move_refused", {}).get(i, 0) < 2
                       and self._move_connects(m, state) for i, m in enumerate(mv))

    def _crushing_lead(self, state):
        """True when the active massively out-levels the foe and still has a connecting attack.
        2026-08-03 Voltorb chalk: L48 Blastoise vs Electric fodder does NOT need matchup-switch,
        potion-bag theater, or must-leave — JUST FIGHT. Hierarchy gate for the whole turn loop."""
        ours = state.get("ours") or {}
        enemy = state.get("enemy") or {}
        # A crushing level lead means nothing when the move list eats every confirm — the 11:43
        # loop: L49 vs L25 kept _just_fight True (stale PP said Bite/WP usable), which skipped
        # the must-leave/famine hierarchy forever. Futility overrides chalk.
        if getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX:
            return False
        act_lv = ours.get("level") or 0
        foe_lv = enemy.get("level") or 0
        if not (act_lv and foe_lv and act_lv >= foe_lv + 8):
            return False
        mv = ours.get("moves") or []
        return any(m.get("id") and m.get("pp", 0) > 0 and m.get("power", 0) > 0
                   and getattr(self, "_move_refused", {}).get(i, 0) < 2
                   and self._move_connects(m, state) for i, m in enumerate(mv))

    def _must_leave_active(self, state):
        """Alive but shouldn't stay in — sleep/freeze lock, or true PP famine (no connecting damage).

        2026-08-03 FIX: do NOT treat skip_streak exhaustion as must-leave. Menu thrash marks every
        move 'failed to fire' without spending PP → skip_streak full → we opened the party and
        looked like 'switching Blastoise for no reason' vs Voltorb with no status. Skip-streak
        belongs to war-must-advance / Struggle, not a voluntary switch."""
        ours = state.get("ours") or {}
        if (ours.get("hp") or 0) <= 0:
            return False
        # Crushing lead: stay in and swing (unless hard-locked by sleep/freeze).
        status = _decode_status(ours.get("status1", 0) or 0)
        if status in ("sleep", "freeze"):
            return True
        # FUTILITY (2026-08-03 09:07): the move list has eaten FUTILE_AMOVE_MAX confirms with zero
        # progress — whatever the per-slot ledger/PP bytes claim, staying in cannot win. This is the
        # turn-loop's route to the same bench switch the war paths reach via the futility breaker.
        if getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX:
            return True
        if self._crushing_lead(state):
            return False
        if self._active_pp_famine(state):
            return True
        return False

    def _alive_bench_slot(self, state):
        """Strongest HP>0 party member that is NOT the active species. Never returns the mon
        already out — that was the 'switch him for himself' failure mode."""
        active_sp = (state.get("ours") or {}).get("species")
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        best, best_lv = None, -1
        for s in range(min(cnt, 6)):
            if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                continue
            try:
                sp = st.read_party_species(self.b, s)
            except Exception:
                continue
            if not sp or sp == active_sp:
                continue
            lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
            if lv > best_lv:
                best, best_lv = s, lv
        return best

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
        ours = state.get("ours") or {}
        # MOVELESS GUARD (2026-07-31, the Abra shotgun-seat): a lead with zero damaging PP can
        # never solo ANYTHING regardless of the level gap (Teleport one-shots nothing) — dropping
        # the ace switch for it would strand it spamming a failing move. The whole point of
        # fielding a moveless mon is the participation switch; never suppress it.
        if not any(m.get("id") and m.get("pp", 0) > 0 and m.get("power", 0) > 0
                   for m in (ours.get("moves") or [])):
            return False
        our_lv = ours.get("level") or 0
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
        self._famine_tried = {}            # PP-FAMINE SWITCH: species -> attempts this battle (bounded
        # by FAMINE_SWITCH_TRIES; a try only counts when the nav actually ran — never an infinite churn,
        # but one flaky menu nav no longer dooms a moveless mon to spam its failing move all battle).
        self._heal_failed = set()          # HEAL-CONSUME-FAILED LATCH: active species whose in-battle
        # item use PROVED it won't consume this battle (bag USE/CANCEL non-consume) -> stop re-offering.
        self._potion_blocked = False       # 2026-08-02 LIVE: Super Potion on FULL ace forever —
        # any no_effect/failed heal this battle kills ALL further potion offers (not just one species).
        self._cure_blocked = False         # same class: Awakening spam after already awake (stream end)
        self._last_battle_progress_t = time.time()  # MENU WEDGE: stall clock, NOT battle-start clock
        self._menu_wedge_n = 0                     # re-armable, bounded (was a one-shot latch)
        self._must_leave_tried = {}        # species -> tries: alive-but-stuck voluntary switch
        self._allow_pokemon_menu = False   # NUCLEAR: only force-switch (faint) may open POKEMON
        self._party_thrash_n = 0           # consecutive unexpected party-screen sightings
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
        self._move_refused = {}            # 2026-08-03 (the Tackle re-fire loop): slot -> times the GAME
        # refused it ("There's no PP left..." — no PP drop, no HP change). gBattleMons PP can read STALE
        # (the same lying struct as the HP/status tears), so a truly-dry move can look usable forever;
        # skip_streak alone can't hold it because war-must-advance CLEARS the streak when it's the only
        # candidate. >=2 refusals = EXILED for this battle whatever its PP byte says: dropped from move
        # picks AND counted as dry by the famine test (so the famine switch — the real rescue — fires).
        self._amove_futile = 0             # 2026-08-03 09:07 (the parked-on-Tackle photos): count EVERY
        # fruitless move-list confirm this battle, whoever pressed it (turn loop, struggle-walk, war
        # paths). The per-slot ledger can be poisoned when the cursor readback and the DRAWN cursor
        # disagree (nav "succeeds", A fires TACKLE, the refusal is tallied on the slot we INTENDED) —
        # then Tackle never exiles, famine never trips, and every escape path steers back to the
        # "least-refused" slot: Tackle, forever, on stream. This counter doesn't care WHICH slot lied:
        # >= FUTILE_AMOVE_MAX fruitless confirms = the move list itself is a tar pit -> STOP touching
        # it and bench-switch (party nav is verified by species-flip, not by move RAM). Reset only by
        # _note_battle_progress (real progress).
        self._futility_switches = 0        # bounded uses of the futility bench switch per battle
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
            _lead = st.SPECIES_NAME.get(state["ours"].get("species"), None)
            self.emit(f"a battle started against {foe}"
                      + (f" — your {_lead} is up front" if _lead else ""), beat=True)
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
            _wild = not self._is_trainer_battle()
            if st.enemy_is_shiny(self.b):
                self.emit(f"WAIT — STOP. that {foe} is SHINY. chat, do you SEE this — you can play this "
                          f"game for five years and never see one. this is real, this is happening.",
                          beat=True, tier=3)
                if _wild:
                    return self._divert_wild_catch("shiny", foe, max_seconds)
                self.log(f"   [engine] ✨ SHINY trainer {foe} — uncatchable, fighting (freak-out only)")
            elif esp in _LEGENDARY_SPECIES:
                self.emit(f"that's a {foe}. a LEGENDARY. okay — okay, do NOT mess this up.",
                          beat=True, tier=3)
                # Wild legendaries (Moltres/Articuno/Zapdos/Mewtwo) are CATCH moments — never KO on sight
                # (2026-08-02 Jonny: "what if she finds the fire bird and just kills it").
                if _wild:
                    self.emit(f"balls out. weaken it carefully — we are NOT knocking out a legendary.",
                              beat=True, tier=3)
                    return self._divert_wild_catch("legendary", foe, max_seconds)
            elif _wild and self._peek_creator_catch_order() and not getattr(self, "_skip_catch_divert", False):
                # Creator LAW (2026-08-02 Diglett chalk): "catch that!" must divert THIS battle — the
                # conversational OK without a harness latch was the kill-loop. One battle only
                # (_divert clears the latch); never re-ball every Diglett on the walk out.
                self.emit(f"catch order — switching to balls on this {foe}. weaken, don't KO.",
                          beat=True, tier=2)
                return self._divert_wild_catch("creator_catch_now", foe, max_seconds)
            elif _wild and esp in _DIGLETT_LINE and not getattr(self, "_skip_catch_divert", False):
                # Diglett's Cave: catch the FIRST unowned Diglett/Dugtrio ONLY (dex bit = party∪box).
                # Arena Trap makes flee impossible — after one is owned, FIGHT the rest.
                try:
                    import travel as _tv
                    mid = tuple(_tv.map_id(self.b))
                except Exception:
                    mid = (0, 0)
                _already = (self._dex_owns_species(50) or self._dex_owns_species(51)
                            or self._party_owns_species(50) or self._party_owns_species(51))
                if mid in _DIGLETT_CAVE_MAPS and not _already and self._ball_count() > 0:
                    self.emit(f"a {foe} — Diglett's Cave keeper. catching ONE for the Ground slot, "
                              f"then we're done balling Digletts.",
                              beat=True, tier=2)
                    return self._divert_wild_catch("diglett_keeper", foe, max_seconds)
            elif _wild and not getattr(self, "_skip_catch_divert", False) and 1 <= esp <= 151:
                # DEX PUSH DIVERT (2026-08-04, Jonny: 'she is not catching enough pokemon for
                # the exp share thing' — the dex sat at 13/50 while the catch-every-new-species
                # doctrine lived ONLY in the strategist brief, a text hint this fight engine
                # never read). The Route 15 aide pays the EXP. SHARE at 50 CAUGHT species, so
                # every KO'd new species is a wasted bar tick. Programmatic now: an UNOWNED
                # wild species is a CATCH, not a fight, while badges>=5, the aide's flag is
                # unclaimed, the bar is short, and there are balls to spare (>=2 — the last
                # ball stays reserved for a shiny/legendary moment). Fail-safe end to end:
                # catch_pokemon is time-bounded, a miss fight-clears as normal, and any read
                # fault just skips the divert (she fights like before).
                try:
                    import field_moves as _fm
                    _dexable = (not _fm.read_flag(self.b, 0x256)      # Exp. Share unclaimed
                                and sum(1 for i in range(8)
                                        if _fm.read_flag(self.b, 0x820 + i)) >= 5
                                and (ram.pokedex_owned_count(self.b) or 0) < 50
                                and self._dex_owns_species(esp) is False
                                and self._ball_count() >= 2)
                except Exception:
                    _dexable = False
                if _dexable:
                    _owned_n = ram.pokedex_owned_count(self.b) or 0
                    self.emit(f"a {foe} — that's a NEW one for the dex. balls out "
                              f"({_owned_n}/50 caught — the Route 15 aide hands over an "
                              f"Exp. Share at fifty).", beat=True, tier=2)
                    return self._divert_wild_catch("dex_push", foe, max_seconds)

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
                        # DECIDED WIN (2026-08-02 Rock Tunnel): never abort mid-victory — travel
                        # re-enters the same fight and the stream watches the last seconds rewind.
                        if self._decided_win():
                            self.log("   [engine] !! post-faint drain wedged BUT win is DECIDED — "
                                     "extending decided-win drain (refuse fight-reset re-entry)")
                            self._debug_snap("drain120_decided_win")
                            return self._drain_decided_win()
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
            # 2026-08-03 NUCLEAR: party open while OUR active is ALIVE = Pokemon↔Blastoise thrash.
            # B out only — A re-selects the ace and is the sticky loop Jonny filmed.
            if self._party_screen():
                _cur = st.read_battle(self.b)
                if _cur and (_cur.get("ours") or {}).get("hp", 0) > 0:
                    self._party_thrash_n = getattr(self, "_party_thrash_n", 0) + 1
                    self.log(f"   [engine] !! PARTY THRASH #{self._party_thrash_n}: open but "
                             f"active ALIVE — B-closing (never A)")
                    for _ in range(8):
                        if not self._party_screen():
                            break
                        self.b.press("B", self.hold, self.hold, self.render, owner=self.owner)
                        self._wait(12)
                    if self._party_thrash_n >= 3:
                        self._allow_pokemon_menu = False
                        self._switch_fail_n = 99
                        self.log("   [engine] PARTY THRASH latch — voluntary POKEMON banned this battle")
                        # 2026-08-04 LIVE (thrash #1..#23, 19 minutes of Hyper-Potion theater): the
                        # party screen kept RE-OPENING because a bag/target layer sat open UNDERNEATH
                        # it, invisible to _bag_screen(), and every 'move commit' re-confirmed the
                        # parked potion. B-closing only the party layer can never break that cycle —
                        # blind-unwind the WHOLE menu stack, then re-home the cursor to FIGHT.
                        self._blind_menu_unwind(8)
                    continue
            glob = self._bstate()
            if glob != last_glob:                     # real progress -> reset the wedge guard
                last_glob, stall = glob, 0
            # MENU TRUTH (2026-08-02 LIVE): gate on ACTION (menu_up==1) OR MOVE LIST — not
            # white_box alone (shared by both → every turn re-opened / probed forever).
            if self._at_action_menu() or self._at_move_list():
                state = st.read_battle(self.b)         # pick + commit a move, verify it lands
                self._note_foe(state)                  # foes-seen ledger (live turn read)
                self._classify_prev_whiff(state)       # race-free: judge last turn's move at this clean
                #                                        menu-up read (before the whiff-breaker acts below)
                # Already on the move list (stray open) — commit immediately, skip switch/item theater.
                if self._at_move_list() and not self._at_action_menu():
                    self.log("   [engine] move list already open at turn top — STREAM COMMIT, "
                             "skipping switch/item menu theater")
                    res = self._select_and_verify(state) if state else "done"
                    if res == "done":
                        self._acted_once = True
                        stall = 0
                        self._unresolved_turns = 0
                    else:
                        stall += 1
                        self._unresolved_turns += 1
                    continue
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
                # JUST FIGHT (2026-08-03): crushing level lead + connecting damage → skip bag /
                # switch theater entirely. Voltorb/Electrode logs: no status, Blastoise half-HP,
                # Water Pulse ending them — potion/POKEMON menus were pure watchability poison.
                # UNGATED FUTILITY CHECK — ABOVE every gate (2026-08-03 11:43, the third
                # parked-on-Tackle photo set): the 09:18 session proved the breaker can be
                # starved by its OWN gates — crushing-lead (_just_fight) skipped must-leave
                # because stale PP bytes said Bite/WP were usable, so the only routes to the
                # breaker were the 60s menu wedge's war presses, which kept landing on refusal
                # boxes ("text") instead of the move list. A battle whose move list has eaten
                # FUTILE_AMOVE_MAX confirms with zero change gets a bench switch attempt at the
                # TOP of every turn, ungated by level leads, item offers, or ledger state.
                if (getattr(self, "_amove_futile", 0) >= FUTILE_AMOVE_MAX
                        and self._is_trainer_battle()
                        and not (self._enemy_fainted or self._we_fainted)
                        and getattr(self, "_futility_switches", 0) < 2):
                    if self._futility_bench_switch():
                        self._acted_once = True
                        stall = 0
                        self._unresolved_turns = 0
                        continue
                _just_fight = bool(state and self._crushing_lead(state)
                                   and _hp_frac(state.get("ours") or {}) > BATTLE_CRIT_FRAC
                                   and _decode_status((state.get("ours") or {}).get("status1", 0) or 0)
                                   not in ("sleep", "freeze"))
                if _just_fight and getattr(self, "_just_fight_logged", None) != id(state):
                    self._just_fight_logged = id(state)
                    _o, _e = state.get("ours") or {}, state.get("enemy") or {}
                    self.log(f"   [engine] JUST FIGHT: L{_o.get('level')} vs L{_e.get('level')} "
                             f"+ connecting damage — skipping bag/switch hierarchy this turn")
                # PART B: SURVIVAL INSTINCT FIRST — if a mon is crit-low/afflicted with a matching item,
                # offer the bag to the oracle. If she uses one, the turn is spent (skip move selection).
                # Any non-use falls through to the proven move path (fail-safe; never wedges).
                if (state and not _just_fight and not (self._enemy_fainted or self._we_fainted)
                        and self._maybe_use_item(state)):
                    self._acted_once = True
                    stall = 0
                    continue
                # ALIVE-BUT-STUCK SWITCH (2026-08-03 Jonny: Blastoise vs Voltorb — half HP, can't
                # move / sleep-freeze / move theater, 3 live on bench, kept "switching" into himself).
                # Force a voluntary leave to a DIFFERENT living species; never row-0 / same species.
                # NUCLEAR: dead while POKEMON_BATTLE_SWITCH=0 (default) — was the thrash opener.
                if (BATTLE_SWITCH_ENABLED and state and not _just_fight
                        and not (self._enemy_fainted or self._we_fainted)
                        and self._must_leave_active(state)):
                    _asp = state.get("ours", {}).get("species")
                    if self._must_leave_tried.get(_asp, 0) < FAMINE_SWITCH_TRIES:
                        if self._bag_screen():
                            self._close_bag_screen()
                            continue
                        self._must_leave_tried[_asp] = self._must_leave_tried.get(_asp, 0) + 1
                        _bs = self._alive_bench_slot(state)
                        if _bs is not None:
                            self.log(f"   [engine] MUST-LEAVE: active alive but stuck "
                                     f"(status/famine/exhausted) -> bench slot {_bs} "
                                     f"(NOT re-picking species {_asp})")
                            if self._switch_to_slot(_bs, _asp) == "switched":
                                self.emit("this one's stuck — switching to someone who can still fight.",
                                          beat=True, tier=2)
                                self._skip_streak.clear()
                                self._acted_once = True
                                stall = 0
                                self._unresolved_turns = 0
                                self._note_battle_progress("must-leave switch")
                                continue
                            self.log("   [engine] must-leave switch did not confirm -> fight/Struggle")
                        else:
                            self.log("   [engine] MUST-LEAVE but no other live bench — Struggle/fight")
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
                if (BATTLE_SWITCH_ENABLED and state
                        and not (self._enemy_fainted or self._we_fainted)
                        and self._active_pp_famine(state)
                        and self._famine_tried.get(state.get("ours", {}).get("species"), 0)
                        < FAMINE_SWITCH_TRIES):
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
                    _fsp = state.get("ours", {}).get("species")
                    self._famine_tried[_fsp] = self._famine_tried.get(_fsp, 0) + 1
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
                            # NARRATE THE MANEUVER (2026-07-30, Jonny live report: this deliberate switch
                            # read as 'she benched the mon she said she was training' — and her own
                            # commentary then credited the wrong mon). Name BOTH parties from ground
                            # truth (the battle read + the verified ace slot) so voice matches screen.
                            try:
                                _wk_sp = (state.get("ours") or {}).get("species")
                                _wk_nm = st.SPECIES_NAME.get(_wk_sp, "the little one")
                                _ace_nm = st.SPECIES_NAME.get(st.read_party_species(self.b, ace), "my ace")
                                # Name Intimidate as the LEAD's ability so chat doesn't pin it on the ace
                                # (2026-08-02: "Blastoise got Ekans's ability" after the turn-1 switch).
                                if _wk_sp in (23, 24, 58, 59):
                                    self.emit(f"that Intimidate was {_wk_nm}'s — {_ace_nm} is in now "
                                              f"for the actual fight. bench XP, then the real hitters.",
                                              beat=True, tier=1)
                                else:
                                    self.emit(f"{_wk_nm} showed up for the XP share — now {_ace_nm} takes it "
                                              f"from here. that's how you train a rookie.", beat=True, tier=1)
                            except Exception:
                                pass
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
                # Skip matchup switch once a turn already failed to resolve — opening POKEMON
                # mid-thrash is more menu theater (2026-08-02 LIVE).
                if (BATTLE_SWITCH_ENABLED and not PROTECT_LEAD_GRIND and not _just_fight
                        and getattr(self, "_unresolved_turns", 0) < 1
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
                    # Only real PP/HP change counts as progress (STREAM COMMIT returns "done" on miss).
                    _ehp = (_cur or {}).get("enemy", {}).get("hp")
                    if self._whiff_prev_fired or (_ehp is not None and _enemy_hp_pre is not None
                                                  and _ehp != _enemy_hp_pre):
                        self._note_battle_progress(
                            "move resolved" if self._whiff_prev_fired else "foe HP changed")
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
                                 f"turns (last={res}) in a TRAINER battle -> mash FIGHT+A (war-must-advance) "
                                 f"[forensics: menu_up={int(self._menu_up())} "
                                 f"action={self._at_action_menu()} moves={self._at_move_list()} "
                                 f"bag={self._bag_screen()} party={self._party_screen()} ours_pp={_pp}]")
                        self._debug_snap("antiwedge_trainer")
                        # NEVER return stuck (travel re-enters the same fight). Screen-aware
                        # presses ONLY — the old blind 'A A' here re-opened a BAG-parked cursor
                        # forever (the Route-13 fisherman Super-Potion loop).
                        p1 = self._war_advance_press()
                        p2 = self._war_advance_press()
                        self.log(f"   [engine] anti-wedge presses: {p1}, {p2}")
                        self._unresolved_turns = 0
                        stall = 0
                        continue
            else:
                self._advance_text()                  # BLUE dialogue/animation box -> advance it
                stall += 1
            # STALL-CLOCK MENU WEDGE (2026-08-03): fire only when NOTHING real has happened for
            # BATTLE_MENU_WEDGE_S — NOT "battle older than 45s". The old t0 clock false-fired mid
            # multi-mon trainer fight ("menus are glitched") and killed the stream.
            _stall_for = time.time() - getattr(self, "_last_battle_progress_t", t0)
            # RE-ARMABLE (2026-08-03): this was a ONE-SHOT latch — when the single bail's own
            # blind A re-opened the bag (fisherman loop), NO escape could EVER fire again and
            # the fight span minutes of Super-Potion theater. Bounded re-fires (each spaced by
            # a full BATTLE_MENU_WEDGE_S of zero progress) keep it loud but never dead.
            if (BATTLE_MENU_WEDGE_S > 0 and _stall_for >= BATTLE_MENU_WEDGE_S
                    and not self._enemy_fainted and not self._we_fainted
                    and getattr(self, "_menu_wedge_n", 0) < 8):
                self._menu_wedge_n = getattr(self, "_menu_wedge_n", 0) + 1
                self.log(f"   [engine] !! MENU WEDGE {_stall_for:.0f}s with NO progress — escaping the "
                         f"scroll theater (fire {self._menu_wedge_n}/8, menu_up={int(self._menu_up())} "
                         f"action={self._at_action_menu()} moves={self._at_move_list()} "
                         f"bag={self._bag_screen()} party={self._party_screen()})")
                if self._menu_wedge_n == 1:
                    self.emit("menus are glitched — bailing this fight.", beat=True, tier=2)
                if not self._is_trainer_battle():
                    return self.flee(max_seconds=45)
                if self._menu_wedge_n >= 2:
                    # 2026-08-04 LIVE (the Hyper-Potion golbat loop): fires 1-3 ran the screen-AWARE
                    # presses below, which consult the SAME lying classifiers that caused the wedge
                    # (action=True while the real screen was the bag list) — so all three fires
                    # burned and the loop ran 19 minutes until the window died. From the second
                    # fire on, stop trusting screens: blind B-unwind the whole stack (B never
                    # confirms), then re-home FIGHT. Cap raised 3->8 so the escape can't go dead.
                    self.log("   [engine] MENU WEDGE escalation: blind B-unwind (classifiers untrusted)")
                    self._blind_menu_unwind(10)
                else:
                    # Trainer, first fire: screen-aware presses only (B out of bag/party; A only on
                    # a FIGHT-homed action menu / move list). The old 'B then always A' alternation
                    # was itself the infinite selected/Use-on-which cycle.
                    for _ in range(8):
                        if not st.in_battle(self.b) or self._enemy_fainted:
                            break
                        self._war_advance_press()
                self._unresolved_turns = 0
                stall = 0
                self._last_battle_progress_t = time.time()  # don't re-trigger every iteration
            if stall >= 30:                           # genuine wedge -> loud abort, never silent
                if self._decided_win():
                    self.log("   [engine] !! stall≥30 but win DECIDED — finishing victory chain "
                             "(refuse fight-reset re-entry)")
                    return self._drain_decided_win()
                if not self._is_trainer_battle():
                    self.log("   [engine] !! battle wedged stall≥30 WILD -> FLEE")
                    return self.flee(max_seconds=45)
                self.log("   [engine] !! battle wedged stall≥30 TRAINER -> screen-aware press, "
                         "not stuck-abort (never a blind A — the bag-parked-cursor lesson)")
                self._war_advance_press()
                stall = 0
                continue
        # Budget exhausted: if the fight is already won, NEVER hand travel a timeout that
        # re-attaches mid-victory (Rock Tunnel trainer loop, 2026-08-02).
        if self._decided_win():
            self.log(f"   [engine] !! battle budget exhausted ({max_seconds}s) but win DECIDED — "
                     f"extending decided-win drain (refuse timeout re-entry)")
            return self._drain_decided_win()
        return "timeout"

    def _finish(self):
        prev = self._prev or {}
        ours = prev.get("ours", {})
        _mine = st.SPECIES_NAME.get(ours.get("species"), "your Pokemon")
        if self._enemy_fainted or (prev.get("enemy", {}).get("hp", 1) == 0):
            # F-7(c): the certain-win beat already voiced this win AT THE FAINT (the drain +
            # LLM chain aligned her reaction with the victory screen) — never voice it twice.
            if not self._win_emitted:
                self.emit(f"you won the battle — your {_mine} finished it", beat=True)
            return "win"
        if ours.get("hp", 1) == 0:
            self.emit(f"you lost - your {_mine} fainted", beat=True)
            return "loss"
        self.emit("the battle ended", beat=False)
        return "ended"

    def _emit_diffs(self, prev, cur):
        if not prev:
            return
        pe, ce = prev["enemy"], cur["enemy"]
        po, co = prev["ours"], cur["ours"]
        # WHO'S ACTUALLY OUT (2026-07-30, Jonny live report: "spearow's eating well tonight" while
        # WARTORTLE made the kill): every event below now names HER ACTIVE battler from gBattleMons[0]
        # (`cur["ours"]` — live ground truth, correct across mid-battle switches). Without the name in
        # the event string, the voice LLM only saw the foe + her full roster and INVENTED the attacker.
        mine = st.SPECIES_NAME.get(co.get("species"), "your Pokemon")
        # narrate the move from the OBSERVED hit (ground truth), not per button-press,
        # so it fires exactly once per landed move - never spammy.
        if ce["hp"] < pe["hp"] and ce["hp"] > 0:
            desc = getattr(self, "_last_desc", "an attack")
            self.emit(f"your {mine} used {desc}", beat=(getattr(self, "_last_eff", 1.0) >= 2))
        if ce["hp"] == 0 and pe["hp"] > 0:
            self._enemy_fainted = True
            self._note_battle_progress("enemy fainted")
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
                self.emit(f"the enemy's {st.SPECIES_NAME.get(ce['species'], 'Pokemon')} went down to "
                          f"your {mine} — that's the battle, you won", beat=True)
                return
            # BATCH 5 PHASE 3 — mark the SIDE so she never narrates her own WIN as a loss. The bare
            # "{species} fainted" read as HER mon dying (she mourned a Nidoran she'd just KO'd). gBattleMons[1]
            # is the ENEMY, so this faint is always a victory. (Avoid the substrings 'knocked out'/'you lost'
            # — pokemon_voice.classify tiers those as a T3 LOSS; "took it down" stays the correct T1.)
            self.emit(f"the enemy's {st.SPECIES_NAME.get(ce['species'], 'Pokemon')} fainted — "
                      f"your {mine} took it down", beat=True)
        if co["hp"] == 0 and po["hp"] > 0:
            self._we_fainted = True
            self._note_battle_progress("our mon fainted")
            self.emit(f"your {mine} fainted", beat=True)
        elif co["maxhp"] and (po["hp"] - co["hp"]) > 0.4 * co["maxhp"]:
            self.emit(f"your {mine} took a big hit", beat=True)
        elif co["maxhp"] and co["hp"] / co["maxhp"] < 0.25 and po["hp"] / max(po["maxhp"], 1) >= 0.25:
            self.emit(f"your {mine} is at low HP - this is getting tense", beat=True)
