"""campaign.py - the continuous CAMPAIGN DRIVER (the skeleton). Executes a KNOWN
FireRed objective sequence Pallet -> Pewter -> Brock end-to-end via reusable handlers,
never exiting on arrival. The harness knows WHERE to go and WHAT the objectives are
(documented game route/gates); Kira's existing self provides ALL battle decisions +
commentary + personality (UNTOUCHED). Engine layer (battle_agent, _pokemon_react,
voice/mood/bond) is NOT modified here. Stuck-detection is LOUD.

OBJECTIVE TYPES (each a reusable handler, built once for the whole game):
  WALK_TO_MAP(map, dir)   - travel to a connected map by crossing its dir edge (BFS,
                            NPC-aware, grass-aware - all already in travel.Traveler)
  WALK_TO_COORD(x, y)     - BFS to a specific tile on the current map
  ENTER_WARP(target_map)  - step onto a door/warp tile, confirm the map flip
  INTERACT(dir)           - face dir, press A, advance the dialogue to its end
  BATTLE()                - hand the pad to the 5/5 battle engine, resume after
  GRIND(map, level)       - battle in grass until our lead mon reaches `level`

A gate that needs a savestate (a scripted quest the harness can't yet script) is
flagged GATE_NEEDS_STATE so the run stops LOUD with exactly what to hand-play.

Run:  .venv\\Scripts\\python.exe pokemon_agent\\campaign.py --headless
"""
import argparse
import os
import sys
import time
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402
import travel as tv              # noqa: E402
import world_fingerprint as wf   # noqa: E402  (MACRO ProgressLedger + fingerprint keystone)
from pokemon_strategy import StrategicMemory, roster_judgment  # noqa: E402  (Batch 3 Phase 2 + block-#3 choice)
from pokemon_search import GuideSearch  # noqa: E402  (Batch 6 Phase 5: silent strategy-guide when stuck)
from pokemon_world import WorldModel  # noqa: E402  (Batch-WORLD: visited-map memory + capability registry — sense of PLACE)
import questline as ql                # noqa: E402  (GATE-UNLOCK: recognise a gate -> derive a questline -> execute)
# GATE-UNLOCK questline driving (recognise a story/HM gate -> route the unlock errand instead of the wall).
# Tightly scoped (only KB-defined gates with the flag UNSET, live-read) + self-clearing; default ON since
# it's the approved forward-progress feature. POKEMON_QUESTLINE=0 disables.
QUESTLINE_ENABLED = os.getenv("POKEMON_QUESTLINE", "1") != "0"
FORWARD_DRIVE_ENABLED = os.getenv("POKEMON_FORWARD_DRIVE", "1") != "0"  # forward objective dominates the grind impulse
# STRATEGIC UNDERLEVEL-GRIND (Task B): when a forward wall keeps beating her because the TEAM is
# under-levelled, grinding fields the WEAK party members (not the ace) to readiness, then resumes the
# march. Extends the forward-drive family; firewall-clean (mode-side only). OFF restores grind(lead+2).
STRATEGIC_GRIND_ENABLED = os.getenv("POKEMON_STRATEGIC_GRIND", "1") != "0"
# BLOCK #3 (2026-07-06 nursery): she JUDGES a wild before throwing — dupe/coverage/level/room — and
# voices the choice both ways ("not this one because…" / "THIS one because…"). The oracle decides
# live; headless follows the framework's lean. OFF restores catch-whatever-appears.
CATCH_JUDGMENT_ENABLED = os.getenv("POKEMON_CATCH_JUDGMENT", "1") != "0"
# Wall-less bench-leveling: a real player raises a fresh catch instead of hauling dead weight — when
# the bench floor sits far under the lead (default gap 10), prep fires toward lead-8 even with no
# recorded wall (the wall-based target still dominates when present).
PROACTIVE_BENCH = os.getenv("POKEMON_PROACTIVE_BENCH", "1") != "0"
PROACTIVE_BENCH_GAP = int(os.getenv("POKEMON_PROACTIVE_BENCH_GAP", "10"))
# SOLO weak-grind: field the weak member as lead and let it grind SOLO in the grass (no in-battle
# participation-switch needed — that switch wedges the long core). Viable now that she can buy Super
# Potions (the in-battle heal instinct keeps a weak lead alive) + heals route to a reachable Center.
# The ace backstops in slot 1 (a weak-lead faint finishes the wild, no blackout). This is the real
# team-building unblock (vs ace-overpower, which can't fix a type-resisted wall like Gary's Charmander).
# Default OFF: the path is RIGHT (team-building) but unfinished — a weak L8 mon fielded in Route-4's
# L8-13 grass FAINTS before earning XP (so the floor doesn't rise), and at the route's far edge no grass
# is reachable to re-engage -> a churn stall. NEEDS (next session): route the weak mon to SURVIVABLE
# low-level grass (its level >= wild levels) + verify the ace-backstop sends in on a weak-lead faint (so
# the fight is WON, weak mon banks participation XP) + field the less-weak mon first. Arm with
# POKEMON_SOLO_WEAK_GRIND=1 to continue. Until then ace-overpower (heals fine via the reachable-Center fix).
SOLO_WEAK_GRIND = os.getenv("POKEMON_SOLO_WEAK_GRIND", "0") != "0"
# Wall-clock ceiling for ONE grind_weak_members() call (a single tick's worth of weak-grinding). grind()
# itself caps at 480s/member; this outer bound stops a multi-weak-member loop from running away on a tick.
GRIND_WEAK_BUDGET_S = int(os.getenv("POKEMON_GRIND_WEAK_BUDGET_S", "600"))
try:
    import field_moves as fm          # noqa: E402  (capability reads: knows-HM AND has-badge)
except Exception:
    fm = None
import battle_agent  # noqa: E402  (module ref for the PROTECT_LEAD_GRIND toggle)
from battle_agent import BattleAgent  # noqa: E402
from dialogue_drive import DialogueDriver, box_open as dd_box_open  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
# ── THREE SAVE LINEAGES (Part A) ──────────────────────────────────────────────
# states/workshop/  = the SMALL set of real sherpa checkpoints WE use to jump up/down the game to
#                     lay hooks (beginning, brock_done, mtmoon_done, misty_done) + any state a live
#                     segment loads. WORKSHOP runs resume-from-any here AND bank here.
# states/kira/      = her SACRED playthrough save(s). ONLY a SHOW run may write here; a WORKSHOP run
#                     is PHYSICALLY FORBIDDEN to write into it (see _save_checkpoint guard). Old runs
#                     are timestamp-archived under kira/archive_<ts>/ (kira_run.py), never clobbered.
# states/archive/   = every handicapped-era TEACHING capture + orphaned scratch/recon state. Out of
#                     the live path so nothing mistakes archaeology for current. Not deleted.
# resolve_state() finds a named state across the live buckets (workshop -> kira -> flat -> archive),
# so boots/gates keep loading after the archive sweep regardless of which bucket holds the file.
STATES_WORKSHOP = os.path.join(STATES, "workshop")
STATES_KIRA = os.path.join(STATES, "kira")
STATES_ARCHIVE = os.path.join(STATES, "archive")
# states/campaign/ = Batch-5 PERSISTENT CAMPAIGN (the playthrough mode): her ONE living save that the
# free-roam run writes to as she progresses and RESUMES from on the next GO — so a session picks up
# where she actually is, never resetting to a frozen fragment. Isolated from the workshop fragments
# (those stay as fallbacks) and PHYSICALLY separate from the canonical states/kira/ spine.
STATES_CAMPAIGN = os.path.join(STATES, "campaign")
CAMPAIGN_SAVE = "kira_campaign.state"      # the single living-campaign savestate filename
# ADDENDUM D — NARRATIVE continuity sidecars (persist her SAGA, not just game RAM): her team bonds/wants
# live in the canonical soul JSON; the loss/wall history (the factual basis of her Gary grudge) lives next
# to the campaign save. Both are loaded at free-roam (Sherpa) start + saved at every campaign anchor, so a
# --resume climb resumes KNOWING her story. Launch-independent (file-based, not tied to any endpoint).
SOUL_JSON = os.path.join(STATES_KIRA, "pokemon_soul.json")
STRAT_JSON = os.path.join(STATES_CAMPAIGN, "strat_memory.json")
# Batch-WORLD — her mental MAP (visited nodes + connectivity + traits + capabilities) persists
# next to the campaign save, so a --resume climb wakes up KNOWING where she's been (no more
# spatial amnesia that left the only place in her head to go = into the wall).
WORLD_JSON = os.path.join(STATES_CAMPAIGN, "world_model.json")
# HUD — the WHOLE-PLAYTHROUGH timer (persists across sessions, unlike per-session run_uptime). Stamped
# once at the first free-roam and read forever after, so the stream HUD shows "how long this journey
# has taken" not "how long since this launch".
PLAYTHROUGH_JSON = os.path.join(STATES_CAMPAIGN, "playthrough.json")
# BATCH 6 PHASE 7 — HEALTH READOUT (for JONNY's cockpit, distinct from the viewer HUD): the free-roam
# loop publishes a tiny JSON the dashboard reads cross-process, so a glance at hour 6 says "she's fine"
# or "she's been wedged an hour". Game-side fields only (progress/where/badges/last-checkpoint); the
# dashboard merges in API spend from the bot's own cost-tracker.
HEALTH_JSON = os.path.join(STATES_CAMPAIGN, "health.json")
# RULE 17 SANCTITY — bank her NARRATIVE saga (grudge/team-feelings/arc) WITH the checkpoint bundle, next to
# strat/world/soul, so a checkpoint restore resumes her STORY (not just game+strategy). Core Kira keeps its
# own states/kira/journey_core.json; this campaign-side copy makes the resume bundle COMPLETE.
JOURNEY_JSON = os.path.join(STATES_CAMPAIGN, "journey_core.json")
CAMPAIGN_SAVE_EVERY = int(os.getenv("POKEMON_CAMPAIGN_SAVE_EVERY", "5"))   # heartbeat-save every N roam ticks
# BATCH 6 PHASE 6 + ADDENDUM A — LAST-RESORT wedge escape-hatch (the 30-hr-run insurance). Fires only
# AFTER the forced-heal hard-recovery has already been tried and RED persists (a GENUINE fingerprint-
# frozen wedge, not idle) — between PROGRESS_HARD_TICKS and PROGRESS_ABANDON_TICKS. Bounded so a self-
# re-wedging position still reaches ABANDON instead of reload-looping forever.
PROGRESS_ESCAPE_TICKS = int(os.getenv("POKEMON_ESCAPE_TICKS", "5"))   # RED this many ticks -> try a reload
MAX_ESCAPE_RELOADS    = int(os.getenv("POKEMON_MAX_ESCAPE_RELOADS", "2"))  # per wedge-episode, then abandon
# BATCH 7 PHASE 1 — DEEP-WEDGE RECOVERY (the floor beneath the escape-hatch, so Jonny can sleep). The
# escape-hatch reloads the most-recent GREEN snapshot — but a STRUCTURAL wedge re-wedges from there
# (the recent-good is at the wedge lip). The deep-wedge floor keeps a ROLLING RING of checkpoints
# banked at definitely-safe GAIN SEAMS (a badge / a new teammate / a fresh catch) and, once the
# escape-hatch is exhausted AND the world is still frozen, reverts PROGRESSIVELY FURTHER BACK through
# that ring to a seam that is guaranteed clear — surfaced in-character ("ugh, something glitched, let
# me back up a sec"), never a silent blink. Only when the whole ring is spent does it ABANDON. N kept.
SAFE_RING_N        = int(os.getenv("POKEMON_SAFE_RING_N", "4"))        # last N gain-seam checkpoints kept
DEEPWEDGE_TICKS    = int(os.getenv("POKEMON_DEEPWEDGE_TICKS", "8"))    # RED ticks (post escape-exhaust) -> deep revert
# Move "junk floor" (Batch 6 no-gut guarantee): a move whose value (power + coverage/utility bonuses) is
# below this is safe to drop to free a slot; if EVERY droppable move is at/above it, KEEP them all rather
# than gut a good set (a gutted moveset fails invisibly at the E4). Tunable.
MOVE_JUNK_FLOOR = int(os.getenv("POKEMON_MOVE_JUNK_FLOOR", "50"))
# PHASE 2 (HM field moves) — surface "use Cut/Strength/Surf to clear the obstacle in front"
# as an oracle CHOICE when she has the HM + badge and an adjacent clearable obstacle is
# detected (pure RAM reads). DEFAULT-OFF: the obstacle DETECTION is source-solid, but the
# in-game ACTUATION (the use-HM prompt) is UNVERIFIED on a long-running libmgba core (the
# move-list-wedge lesson) — so it's behind a flag for a one-variable live control before it
# rides the canonical climb. Flip POKEMON_FIELD_MOVES=1 to arm it for the verification run.
FIELD_MOVES_ENABLED = os.getenv("POKEMON_FIELD_MOVES", "0") == "1"
# PHASE 3 (overworld item pickups) — offer "grab the item over there" as an oracle choice when a
# ground item ball (gfx 92) is BFS-reachable AND it's SAFE: only on the OVERWORLD surface (map group
# 3 = routes/towns), never inside a cave/interior, so a roadside potion-grab can NEVER disrupt a
# hard-won fragile traversal like Mt Moon (cave/building item balls are DEFERRED, recon-flagged).
# Bounded detour distance so she never wanders far off-route into danger for an item. DEFAULT-OFF:
# the pickup actuation (grab_item) reuses the proven _talk face→A→drain mechanic but the bag-delta
# confirmation is unverified here (the catch-path lesson) — arm POKEMON_ITEM_PICKUP=1 for the control.
ITEM_PICKUP_ENABLED = os.getenv("POKEMON_ITEM_PICKUP", "0") == "1"
# STRATEGIC-STUCK FLOOR (the Gary death-loop killer) — run-existential, so DEFAULT-ON (it's logic only:
# it re-shapes the option set + ctx, no risky actuation). Revert switch if ever needed. FIREWALL: this
# lives in the autonomous-play free_roam loop, which ONLY runs when SHE is playing — so the GRIND-ACTION
# it steers toward is play-gated by construction and can never fire while she's cohosting a human-played
# game. The FEELING about the loss is the separate emitted reaction (the core voice seam).
STRATEGIC_STUCK_ENABLED = os.getenv("POKEMON_STRATEGIC_STUCK", "1") == "1"
ITEM_GRAB_MAX_DIST  = int(os.getenv("POKEMON_ITEM_GRAB_MAX_DIST", "18"))   # max grass-free steps to detour


def resolve_state(name):
    """Find a named savestate across the live buckets, preferring the live path over archaeology:
    campaign/ -> workshop/ -> kira/ -> flat states/ -> archive/. Returns the full path or None."""
    for d in (STATES_CAMPAIGN, STATES_WORKSHOP, STATES_KIRA, STATES, STATES_ARCHIVE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def kira_checkpoints():
    """The seg_*.state checkpoints present in states/kira/ (a Kira run is in progress iff non-empty)."""
    if not os.path.isdir(STATES_KIRA):
        return []
    return sorted(f for f in os.listdir(STATES_KIRA) if f.endswith(".state"))


def archive_kira_save(stamp):
    """Timestamp-archive the current Kira playthrough so a FRESH show run never clobbers her save:
    move states/kira/*.state -> states/kira/archive_<stamp>/. `stamp` is passed in (caller stamps the
    clock). Returns the archive dir, or None if there was no save to move."""
    import shutil
    saves = kira_checkpoints()
    if not saves:
        return None
    dst = os.path.join(STATES_KIRA, f"archive_{stamp}")
    os.makedirs(dst, exist_ok=True)
    for f in saves:
        shutil.move(os.path.join(STATES_KIRA, f), os.path.join(dst, f))
    soul = os.path.join(STATES_KIRA, "pokemon_soul.json")    # archive her soul-continuity too, so a
    if os.path.exists(soul):                                 # FRESH run starts with a blank roster/wants
        shutil.move(soul, os.path.join(dst, "pokemon_soul.json"))
    return dst

# ── known FireRed map IDs (group, num) - documented, not reconned ────────────
PALLET, VIRIDIAN, PEWTER = (3, 0), (3, 1), (3, 2)
ROUTE1, ROUTE2, ROUTE22 = (3, 19), (3, 20), (3, 41)
ROUTE3, ROUTE4 = (3, 21), (3, 22)      # Pewter->Route 3 (east), Mt Moon->Route 4 (Cerulean side)
# Viridian Forest + gym interiors live in other groups; resolved at runtime on entry.
# Viridian Pokemon Center (RED-roof bldg, reconned 2026-06-24): town door (26,26) -> interior
# (5,4); nurse at (7,2), talked to from (7,4) ACROSS the counter; entrance mat at (7,8).
VIRIDIAN_PC_DOOR, VIRIDIAN_PC_MAP, NURSE_FRONT, PC_ENTRY = (26, 26), (5, 4), (7, 4), (7, 8)
# Pewter Gym (reconned 2026-06-24): town door (15,16) -> interior (6,2); climb the centre column
# (x=6) to Brock's front tile (6,6); the lone gym trainer auto-engages off the centre path.
PEWTER_GYM_DOOR, BROCK_FRONT = (15, 16), (6, 6)
# Pewter Pokemon Center door (disasm PewterCity warp_events: x17 y25 -> POKEMON_CENTER_1F). The
# nearest heal once on Route 3 (cross WEST to Pewter) — PC interiors share ONE layout so the generic
# heal_at_center(door) works for any city. (Pewter Mart door = (28,18), banked for later TM/shopping.)
PEWTER_PC_DOOR, PEWTER_MART_DOOR = (17, 25), (28, 18)
# Oak's Parcel quest (disasm: ViridianCity/PalletTown + their interiors). Viridian Mart town door
# (36,19) -> interior; clerk at (2,3) -> talk from the front tile (2,4) facing UP. Oak's Lab town
# door (16,13) in Pallet -> interior; Oak at (6,3) -> talk from (6,4) UP. The 5 Poke Balls are GIVEN
# by the lab ReceiveDexScene (giveitem ITEM_POKE_BALL,5) on delivery — NO shop-buy. FLAG_SYS_POKEDEX_
# GET=0x829 is set on delivery (the "parcel delivered" proof, alongside ball count 0->5).
VIRIDIAN_MART_DOOR, MART_CLERK_FRONT = (36, 19), (2, 4)
OAK_LAB_DOOR, OAK_FRONT = (16, 13), (6, 4)
FLAG_POKEDEX_GET = 0x829
# Mt Moon entrance (disasm Route4 warp_events): the 1F mouth is a warp at (19,5) ON ROUTE 4 — Route 3
# has NO Mt Moon warp; it connects UP to Route 4 (offset 60), and the cave is entered from Route 4.
MT_MOON_ENTRANCE = (19, 5)
ROUTE4_PC_DOOR = (12, 5)        # Route 4 has its own Pokemon Center (disasm Route4 warp_events)
# Cerulean Gym (reconned 2026-06-26, recon_gymscan): town door (31,21) -> interior; Misty (8,6,
# trainerType=0) is fought from the front tile (8,7). TWO junior trainers gate her: (10,12) and a
# WANDERER (4,7..7,7) - both trainerType!=0. badge flag 0x821 (Cascade) follows Boulder (0x820).
CERULEAN, CERULEAN_GYM_DOOR, MISTY_FRONT = (3, 3), (31, 21), (8, 7)
# Cerulean Pokemon Center door: (22,19) -> MAP_CERULEAN_CITY_POKEMON_CENTER_1F. Sourced from the
# pokefirered disasm (CeruleanCity warp_events) AND cross-checked against the live RAM warp table.
CERULEAN_PC_DOOR = (22, 19)
# Cerulean Poke Mart: town door (29,28) -> interior (7,7), clerk at (2,3) (the verified Mart-layout
# signature; recon_findmart 2026-06-28). NOTE: the prior session mis-tagged (30,11)/(7,1) as the Mart
# — that's the POLICEMAN-blocked robbed-house (BlockExits gate, NPC perma-parked at (30,12) until
# FLAG_GOT_SS_TICKET). Real Mart confirmed by entering + reading the clerk object, not menu actuation.
CERULEAN_MART_DOOR = (29, 28)
# Vermilion City (live block (3,5) — confirmed on foot, vermilion_walk1 2026-07-06). Doors from the
# pret disasm VermilionCity warp_events (the same source that matched Route 4/Cerulean live exactly;
# live-confirm on first use): Pokémon Center (15,6); Mart (29,17); Gym (14,25) — CUT-LOCKED behind
# the fence tree (HM01 from the S.S. Anne captain first); Fan Club (12,17) = the Bike Voucher.
# ⚠ THE DOCK: (22,34)/(23,34)/(24,34) are the S.S. ANNE BOARDING PIER (ticket-gated triggers at
# (22-23,32-33)) — the SOUTHERNMOST doors in town, which is how the blind "warp south" heal
# heuristic BOARDED THE BOAT (vermilion_walk1; also the true identity of the run-12/13 "stair-house
# trap": (1,4)=ship exterior, (1,5)=1F corridor, (1,10)=2F, (1,12..16)=cabins — the SHIP, no house).
VERMILION = (3, 5)
VERMILION_PC_DOOR, VERMILION_MART_DOOR = (15, 6), (29, 17)
# FLAG_BADGE0x_GET in the SaveBlock1 flag array (base + 0x0EE0): Boulder 0x820, Cascade 0x821.
FLAG_BADGE_BOULDER, FLAG_BADGE_CASCADE = 0x820, 0x821

# CITY -> its own Pokemon Center door (PC interiors share ONE layout, so heal_at_center(door) heals
# in ANY of them — only the overworld door differs). The map-keyed table replaces heal_nearest's old
# hardcoded Forest-era default (everything-not-Route3/4 -> Viridian), which routed Cerulean cross-
# region south. EXTEND as new cities are reached; an unmapped city heals LOUD-fallback, never silent.
CITY_PC_DOORS = {VIRIDIAN: VIRIDIAN_PC_DOOR, PEWTER: PEWTER_PC_DOOR,
                 CERULEAN: CERULEAN_PC_DOOR, ROUTE4: ROUTE4_PC_DOOR,
                 VERMILION: VERMILION_PC_DOOR}

# ── GYM REGISTRY: one row per leader, so beat_gym is data-driven + general (gyms gate the leader
# behind junior trainers - beat all juniors, THEN the leader). reserve = move-slots to free for an
# expected level-up double-learn (Brock's L15); leader_dir = key that faces the leader from the
# front tile (UP for Brock + Misty - the leader sits directly above their front tile). ──
GymSpec = namedtuple("GymSpec", ["name", "city", "door", "leader_front", "badge_flag",
                                 "reserve", "leader_dir"])
# B-4 — the three Kanto starter EVOLUTION families. No vanilla Kanto trainer EXCEPT the rival (Gary)
# fields a starter-line Pokémon, so a trainer battle containing one of these = a Gary encounter, at ANY
# fight (he leads with Pidgey-line later, so we scan his whole party, not just the active lead).
_STARTER_LINES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})

GYMS = {
    "Brock": GymSpec("Brock", PEWTER, PEWTER_GYM_DOOR, BROCK_FRONT, FLAG_BADGE_BOULDER, 2, "UP"),
    "Misty": GymSpec("Misty", CERULEAN, CERULEAN_GYM_DOOR, MISTY_FRONT, FLAG_BADGE_CASCADE, 0, "UP"),
}
# party-mon cached HP (unencrypted): current 0x56, max 0x58 (off GPLAYER_PARTY)
P_HP, P_MAXHP = 0x56, 0x58
# party-mon non-volatile STATUS condition (unencrypted u32 @ +0x50 in the 100-byte struct — the same
# party-only block as level/HP at 0x54/0x56/0x58, so reliable). Gen-3 STATUS1 bitfield (PART B/C):
P_STATUS = 0x50
STATUS_SLEEP_MASK, STATUS_POISON, STATUS_BURN = 0x07, 0x08, 0x10
STATUS_FREEZE, STATUS_PARALYSIS, STATUS_BAD_POISON = 0x20, 0x40, 0x80


def decode_status(s):
    """STATUS1 u32 -> a status NAME ('sleep'/'poison'/'burn'/'freeze'/'paralysis') or None. Toxic
    (bad poison) folds to 'poison' (Antidote cures both). Sleep is the low 3 bits = turns remaining."""
    if s & STATUS_SLEEP_MASK:
        return "sleep"
    if s & (STATUS_POISON | STATUS_BAD_POISON):
        return "poison"
    if s & STATUS_BURN:
        return "burn"
    if s & STATUS_FREEZE:
        return "freeze"
    if s & STATUS_PARALYSIS:
        return "paralysis"
    return None


# ── BATCH 2 PART C: ITEM SEMANTICS (Gen-3 item ids). CANDIDATES sourced from the disasm AND control-
# verified at runtime by the buy bag-delta (a wrong id -> the target count won't rise -> LOUD abort, so
# nothing is trusted blind). Pewter BUY list control-verified 2026-06-27 (recon_mart): Potion id 13. ──
ITEM_POTION = 13
ITEM_POKE_BALL = 4
# weakest -> strongest healing potions (what "stock up on potions" buys, cheapest first)
HEAL_ITEMS = {13: "Potion", 22: "Super Potion", 21: "Hyper Potion", 20: "Max Potion", 19: "Full Restore"}
# status NAME -> (cure item id, cure name). Awakening(17) cures sleep; Parlyz Heal(18) paralysis; etc.
STATUS_CURE = {"poison": (14, "Antidote"), "burn": (15, "Burn Heal"), "freeze": (16, "Ice Heal"),
               "sleep": (17, "Awakening"), "paralysis": (18, "Parlyz Heal")}
ITEM_NAMES = {**HEAL_ITEMS, 4: "Poké Ball", 3: "Great Ball", 23: "Full Heal",
              **{cid: cn for cid, cn in STATUS_CURE.values()}}

# CITY -> Mart town door (extend as towns are reached; an unmapped city = no stock-up offered, LOUD).
CITY_MART_DOORS = {PEWTER: PEWTER_MART_DOOR, VIRIDIAN: VIRIDIAN_MART_DOOR,
                   CERULEAN: CERULEAN_MART_DOOR,
                   # Vermilion Mart door registered; MART_STOCK rows NOT yet control-verified —
                   # buy_at_mart loud-skips until a live visit bills the row order.
                   VERMILION: VERMILION_MART_DOOR}
# CITY -> the BUY list's item ids IN ROW ORDER (cursor row = stock.index(id)). The mart item list is in
# ROM (no EWRAM array), so the row order is data here — control-verified per town by the buy bag-delta;
# an unverified/wrong row simply fails the per-purchase verify and aborts LOUD (never silent mis-buy).
# Pewter row order CONTROL-VERIFIED (recon_mart, brock_done): PokéBall,Potion,Antidote,ParlyzHeal,
# Awakening,BurnHeal. Viridian early stock is the same family (verify on first visit; bag-delta guards).
MART_STOCK = {
    PEWTER:   [4, 13, 14, 18, 17, 15],
    VIRIDIAN: [4, 13, 14, 18, 17, 15],
    # Cerulean BUY rows CONTROL-VERIFIED by the cursor-readback + bag-delta (recon_buytest 2026-06-28):
    # row0 Poke Ball(4,200) · row1 Super Potion(22,700) · row2 Potion(13,300) · row3 Antidote(14,100) ·
    # row4 Repel(350, not bought so id unregistered — buy-verify guards anything not listed).
    CERULEAN: [4, 22, 13, 14],
}
MART_CURSOR = 0x02039940   # u8 highlighted-row index in the BUY list (CONTROL-found 2026-06-27 recon_mart)
# Shopping policy (named, tunable): top potions up to this; buy this many of each needed cure; keep this
# much money in reserve (never drain the wallet). Quantities are sensible, not min-max hoarding.
SHOP_POTION_TARGET = 6
# BATCH 6 PHASE 2 — FORESIGHT target: when she's up against a wall she can't beat yet, a real player
# stocks up DEEPER before pushing on (she has $7-9k here). Bumps the buy-to target so "stock up before
# I take that bridge" is a live, characterful option — not only a restock when she's already empty.
SHOP_POTION_FORESIGHT = 10
SHOP_CURE_QTY = 2
# BATCH 6 PHASE 3 — Poké Ball FORESIGHT: a teammate is the answer to a wall (Phase 2), and you can't
# catch one with an empty bag. When she's running a thin team and low on balls, "grab some Poké Balls"
# surfaces as a real shopping need — so she goes into the grass equipped to actually come home with a
# new family member, not empty-handed. Thin = party at/below this; restock up to the ball target.
SHOP_THIN_PARTY = 3
SHOP_BALL_TARGET = 5
SHOP_MONEY_FLOOR = 500

# ── THE CAMPAIGN: Pallet -> Pewter -> Brock, as an objective list ─────────────
# Each objective: (TYPE, *args, label). The driver executes them in order, never
# exiting on arrival. Gates the harness can't script yet are GATE_NEEDS_STATE.
def build_objectives():
    return [
        ("WALK_TO_MAP", VIRIDIAN, "north", "Route 1 -> Viridian City"),
        # Oak's Parcel STORY BEAT (Skip 4, AUTO): Viridian Mart clerk gives OAK'S PARCEL -> carry it
        # to Oak in Pallet -> Pokedex + 5 Poke Balls (the lab ReceiveDexScene's giveitem, NOT a shop
        # buy). The balls are what unblock the Route 3 catch. Interior nav + dialogue only.
        ("DELIVER_PARCEL", "Viridian Mart -> OAK'S PARCEL -> Oak's lab -> Pokedex + 5 Poke Balls"),
        ("ADVANCE_NORTH", PEWTER, "Viridian -> Route 2 -> Forest -> Pewter (auto walk/warp)"),
        ("LEVEL_CHECK", 13, "Brock-readiness (need ~Lv13 for Vine Whip + to survive Onix)"),
        ("BEAT_GYM", "Brock", "Pewter Gym -> Brock -> Boulder Badge (beat_gym enters + fights)"),
    ]


# ── SEGMENT MANIFEST: the resumable spine that grows to the credits ──────────
# A SEGMENT is a named run of objectives ending at a stable OVERWORLD checkpoint. run_segments()
# plays them in order and AUTO-SAVES seg_<name>.state after each, so a continuous "Go" RESUMES from
# the furthest checkpoint instead of replaying the whole game. Hand-play-only gates stay
# GATE_NEEDS_STATE. Each future route/gym is ONE more line here. Additive — build_objectives()/run()
# are untouched, so --fast and the proven Brock arc keep working exactly as before.
Segment = namedtuple("Segment", ["name", "objectives", "checkpoint"])


def build_segments():
    """The whole-game spine as ordered, resumable segments. TODAY: the proven Pallet->Brock arc as
    segment 1, checkpointing the post-badge Pewter overworld. Later segments append below — each its
    OWN recon+build (east-edge travel, Mt Moon cave nav, Mart buy, catch, Misty), so we never ship
    a leg blind. Uncomment/extend as each is built and proven."""
    return [
        # THE AUTONOMOUS OPENING (replaces the old after_pick_bulbasaur hand-bank): from a FRESH ROM
        # boot, Kira names herself + rival + picks gender -> bedroom -> Pallet -> Oak's lab -> picks
        # her starter -> rival battle -> Pallet. Run segment 0 from a FRESH boot (no --boot state);
        # its checkpoint seg_opening.state (= a clean post-opening Pallet) feeds pallet_to_brock.
        Segment("the_opening", [
            ("OPENING", "KIRA", "GARY",
             "fresh new game -> name self+rival (her choice) -> pick starter -> rival battle -> Pallet"),
        ], "seg_opening.state"),
        Segment("pallet_to_brock", build_objectives(), "seg_boulder_badge.state"),
        # Pewter -> Route 3: AUTO (Skip 1 solved). The "collision paradox" was a travel bug, not a game
        # wall: E/W map connections load a DEEP overlap of the neighbour's tiles past the edge (Pewter
        # east reaches x=55, past sx_hi=48; disasm offset=10 -> Route 3). The old "x==sx_hi" exit goal
        # trapped the agent the instant it stepped into the overlap (BFS dragged it back west). travel
        # now crosses on "x>=sx_hi" and presses east THROUGH the overlap until the map flips. Control:
        # brock_done.state Pewter(3,2)@(15,16) -> walk east -> MAP TRANSITION -> Route 3(3,21)@(0,11).
        Segment("pewter_to_route3", [
            ("WALK_TO_MAP", ROUTE3, "east", "Pewter -> Route 3 (cross the east connection band)"),
        ], "seg_route3_entered.state"),
        # CATCH = AUTO (Skip 2 + Skip 4). The "agent can't drive the battle menu" wall was a stale
        # claim (the phantom-A bug was fixed 2026-06-25; catch_pokemon is control-proven party N->N+1).
        # catch_one wanders Route 3 grass (warp-avoiding), heals the lone starter at Pewter between
        # gauntlet trainers, and commits Poke Balls to the wild — using the 5 balls Oak's Parcel
        # (Skip 4) put in the bag. ROSTER_REACT then voices her new teammate. Control: from a Route 3
        # state with balls, catch_one -> party 1->2 (caught a Pidgey).
        Segment("route3_catch", [
            ("CATCH", "wander Route 3 grass + catch a teammate (whatever appears — her journey, not a script)"),
            ("ROSTER_REACT", "Kira reacts to her new caught teammate, in her own voice"),
        ], "seg_route3_caught.state"),
        # Route 3 -> Mt Moon -> Route 4: the cave clear is PROVEN (recon_mtmoon_clear: warp chain +
        # fossil/Miguel + east exit, mtmoon_cleared/cerulean_caught banked) but lives in a recon
        # Route 3 -> Mt Moon -> Route 4 -> Cerulean: FULLY AUTO now (Skip 3). The "entrance seam" was
        # a geography misread — Route 3 has NO Mt Moon warp; it connects UP to Route 4, and the cave
        # mouth is a warp at (19,5) ON ROUTE 4 (disasm). So: cross north to Route 4, step into the Mt
        # Moon entrance, then hand to the PROVEN clear_mt_moon (untouched nav; now flees wilds / fights
        # only trainers in the cave, per the solo-Ivysaur reality). Cave-clear emerges on Route 4 and
        # crosses east to Cerulean itself.
        Segment("route3_to_cerulean", [
            ("WALK_TO_MAP", ROUTE4, "north", "Route 3 -> Route 4 (north connection)"),
            ("HEAL_NEAREST", "top up at the Route 4 Pokemon Center before the cave (survive Miguel)"),
            ("ENTER_WARP", MT_MOON_ENTRANCE, "step into the Mt Moon entrance (Route 4 (19,5))"),
            ("CLEAR_MT_MOON", "follow the saved route (run_plan) Mt Moon -> Route 4 -> Cerulean; "
             "flee wilds, fight trainers; a Miguel death -> blackout-recovery retries the segment"),
        ], "seg_cerulean.state"),
        # Misty gym: FULLY AUTO + proven (general beat_gym - clears the 2 junior trainers, beats
        # Misty, dialogue-paced award -> Cascade flag). From cerulean_caught (Route 4) walk EAST into
        # Cerulean, then beat the gym.
        Segment("beat_misty", [
            ("WALK_TO_MAP", CERULEAN, "east", "Route 4 -> Cerulean City"),
            ("BEAT_GYM", "Misty", "Cerulean Gym -> Cascade Badge"),
        ], "seg_cascade_badge.state"),
    ]


def log(m):
    print(f"   [campaign] {m}", flush=True)


# ── reusable objective handlers ──────────────────────────────────────────────
class Campaign:
    def __init__(self, bridge, battle_runner, on_event=None, beat=None, render=None, choose=None,
                 journey=None, alert=None):
        self.b = bridge
        # PHASE 4 — continuity-into-core seam: push her journey narrative to core Kira at anchors, so
        # she resumes KNOWING her story across launch methods + in idle chat. Headless/no-bot -> no-op.
        self._journey_post = journey or (lambda _s: None)
        # PHASE 2 (Batch 7) — dead-man's-switch alert seam: ping the human if recovery is exhausted.
        self._alert_post = alert or (lambda _m: None)
        # STRATEGIC AWARENESS (Batch 3 Phase 2): road-memory of who she's fought + who's walled her.
        # Always present (pure reads, headless-safe). The injected battle-runner is WRAPPED so EVERY
        # battle path (travel-into-trainer, gym juniors, segments) is observed from one place — no
        # battle_agent / play_live surgery, and the wrapper is pure-additive (real runner unchanged).
        self.strat = StrategicMemory(log=log)
        # RECALIBRATION — the STANDING objective she returns to after a detour (a 25-min trainer
        # gauntlet, a heal run). Set when she commits to a going-somewhere action; preserved through
        # detour actions (heal/talk/grind) so she resumes "I was heading to X" instead of re-deciding
        # from scratch + wandering. {action, label, tick} or None. Surfaced to her ctx + the dashboard.
        self._active_objective = None
        # FIX 1 (overworld half) — repeat-pick awareness: the action she chose last tick + how many
        # ticks in a row she's chosen it. Used to nudge "you keep doing X and nothing's changed — try
        # something else" ONLY when the world fingerprint is ALSO stuck (non-GREEN), so a legit
        # multi-tick action (walking to a Center to heal) is never nagged.
        self._last_action_pick = None
        self._repeat_pick_n = 0
        # PROBLEM 3 — SILENT-NO-MOVE guard: a movement action (head_to_gym/travel/wander_catch) that
        # returns WITHOUT changing her RAM position is silently failing. Track a STREAK of no-move ticks
        # plus the SET of move-picks that came back dead during it — so even if she ALTERNATES dead routes
        # (head_to_gym <-> travel:X), ALL of them get pruned together once the streak hits the floor,
        # forcing her onto an action that actually does something. Both reset on any real movement.
        self._dead_moves = set()
        self._nomove_streak = 0
        # B-2 — RESOLVED/LOOPING-NPC guard: (map_id, coords) spots where a dialogue loop was disengaged
        # (the cycle-watchdog bailed). She won't re-initiate talk there, so she can't get re-sucked into
        # the same looping NPC / re-trigger a beaten trainer's restarting lines.
        self._looped_spots = set()
        # LAYER A — UNIFIED "NPCs that block me" memory: {(map_id, body_tile)} of plain (non-trainer)
        # NPCs sitting on a path tile. ONE set shared by BOTH the travel path (routes AROUND these
        # body tiles — the live Slowbro-near-Cerulean-Mart wedge) AND the talk path (`_npc_is_resolved`
        # skips them). Persists across roam ticks (the per-travel-call `static_blocked` reset every tick,
        # so a blocker was re-discovered + re-bumped forever). Fed from travel's chokepoint gauntlet,
        # the B-2 dialogue-loop disengage, and the Layer-B watchdog wedge-spot mark — one source of truth.
        self._blocked_npcs = set()
        # LAYER B — UNIVERSAL WALL-CLOCK WATCHDOG: created at free_roam start. `_stuck_request` is the
        # latched disengage the live render hook sets when StuckWatch trips (honored at the top of the
        # roam loop AND cooperatively by travel via stuck_check); `_watchdog_trips` counts trips this
        # wedge-episode so a re-trip escalates to the escape-hatch. Reset on real (GREEN) progress.
        self._stuckwatch = None
        self._stuck_request = None
        self._watchdog_trips = 0
        # Batch-WORLD — her WORLD-MODEL + CAPABILITY-MODEL (sense of place). Pure data/awareness,
        # headless-safe; feeds the oracle via the same `place` seam, never decides for her. Seeded
        # with the known overworld nodes, enriched live each tick, persisted across --resume.
        self.world = WorldModel(log=log)
        self.guide = GuideSearch(log=log)     # BATCH 6 PHASE 5: silent guide (no-op until dep is live)
        # GATE-UNLOCK questline (recognise → derive → execute). KB + recogniser built once; the active
        # questline is an `_active_objective` sibling that survives detours and self-clears when its
        # success flag/cap reads satisfied LIVE. Pure harness routing; the derived PLAN reaches her
        # DECISION ctx via the `place` seam (firewall — she narrates + still chooses).
        self._questline_kb = ql.load_kb(log=log)
        self._gate_recognizer = ql.GateRecognizer(
            self.b, self.world, kb=self._questline_kb,
            party_count_fn=lambda: self.b.rd8(ram.GPLAYER_PARTY_CNT), log=log)
        self._active_questline = None
        self._ql_entered_doors = set()        # doors entered for the active questline (no re-entry loop)
        self._ql_inside_target = False        # True while deliberately inside a quest-target building
        self._raw_battle_runner = battle_runner
        self.battle_runner = self._observed_battle_runner
        battle_runner = self._observed_battle_runner   # travel + every consumer gets the observed one
        # default sink must accept emit's kwargs (PokemonSoul.emit calls on_event(text, kind=, tier=));
        # a 1-arg default crashed headless soul fires (live passes voice.emit which already accepts them).
        self.on_event = on_event or (lambda s, **_k: log(f"[event] {s}"))
        self.beat = beat or (lambda s: None)
        self.render = render or (lambda: None)
        # BATCH-2 SOUL ORACLE: injected HTTP oracle (play_live -> voice.choose -> her self/LLM). None in
        # headless/no-bot runs -> _soul_choose returns None (no want surfaces; constrained kinds fall back).
        self._oracle_choose = choose
        # heal-when-low: the pause fires ONLY during forward traversal, never during a SERVICE
        # navigation (the heal-return south, the PC routing) - else it would re-trigger mid-heal.
        self._suppress_heal = False
        # CLIMAX guard: True only inside beat_gym's scripted award/TM A-drain. The soul-on render
        # (play_live) reads it and STOPS polling/holding dialogue there, so the scripted cutscene
        # drains exactly like the proven headless path (a reading-pace hold injected into the award
        # was freezing the badge moment ~30-100s). Default False -> normal reactive perception.
        self.draining_award = False
        # MODE (Part A): SHOW vs WORKSHOP. SHOW = fresh/canonical spine, banks to states/kira/, any
        # GATE_NEEDS_STATE load is a LOUD violation (target 0). WORKSHOP = resume-from-any scratch,
        # banks to states/workshop/, never writes states/kira/. ckpt_dir is the write/resume target.
        self.show_mode = False
        self._show_violations = 0
        self.ckpt_dir = STATES_WORKSHOP
        # BATCH 2 SCAFFOLD: her Pokémon-self (wants + roster bonds + move-learn beats). Routes every
        # reaction OUT through on_event (the core-Kira seam); decisions are seams Batch 2 fills. The
        # firewall (_pokemon_react/voice/mood/bond/bridge) is NEVER touched - this is mode plumbing.
        try:
            from pokemon_soul import PokemonSoul
            self.soul = PokemonSoul(emit=self.on_event, choose=getattr(self, "_soul_choose", None))
        except Exception:
            self.soul = None
        self._last_lead_species = None        # for the note_evolve soul hook (lead species-change watch)
        self._last_lead_pid = None            # slot-0 personality value — evolution keeps the PID, a
                                              # party REORDER swaps it (the false-"[evolve]" discriminator)
        # BATCH 2 PART C: statuses that have afflicted the party recently (sampled each free-roam tick),
        # so "shop with intent" buys the SPECIFIC cures for what actually hurt her, not a generic kit.
        self._afflict_seen = set()
        # BATCH 6 PHASE 2 (the Gary death-loop killer) — a LEARNED map-connection graph + grass memory,
        # accumulated from REAL header reads as she travels (never guessed). Lets _grass_target route to
        # the nearest NON-gated grass even when it's BEHIND her, several hops back on already-cleared
        # routes (the live gap: the only grass she'd consider was across the gated Nugget bridge, so she
        # died, healed, re-crossed, died — 4x). _conn_graph: {(grp,num): {edge: (grp,num)}}; _grass_maps:
        # maps where reachable huntable grass was confirmed underfoot at least once.
        self._conn_graph = {}
        self._grass_maps = set()
        # ADDENDUM A — escape-hatch memory: the last KNOWN-GOOD live savestate (captured on a progressing
        # tick, so it's from BEFORE any wedge and already includes prior gains) + the gain-fingerprint at
        # that moment (badges, party size, dex caught) so a reload can NEVER rewind past a real gain.
        self._last_good_state = None
        self._last_good_gain = None
        # PHASE 1 (deep-wedge floor) — rolling ring of checkpoints banked at GAIN SEAMS (badge/teammate/
        # catch): the progressively-older fallbacks the deep-wedge revert walks back through when the
        # escape-hatch (recent-good) keeps re-wedging. Each entry: {state, gain, label}. In-memory (the
        # disk anchor already covers crash-resume); newest at the right, oldest auto-dropped at SAFE_RING_N.
        from collections import deque as _deque
        self._safe_ring = _deque(maxlen=SAFE_RING_N)
        self._ring_last_gain = None        # last gain-sig banked into the ring (de-dupe; bank only on a real gain)
        self._deepwedge_reverts = 0        # ring entries consumed this wedge-episode (reset on real progress)
        # Deterrence (Phase 2c): consecutive ticks the routing hard-refused to cross the active wall.
        # Escalates the "this ends the same way" framing so the blind re-walk stops being her default.
        self._wall_gated_streak = 0
        # one Traveler reused for every WALK leg (BFS + NPC-aware + grass-aware + handoff)
        # stuck_check (LAYER B): travel polls this each step; when the universal watchdog has latched a
        # disengage, travel bails its leg LOUD ("stuck") so the wedge unwinds to the roam loop's
        # top-level recovery — instead of spinning a sub-tick loop the per-tick ledger can't see.
        self.trav = tv.Traveler(bridge, battle_runner=battle_runner, render=self.render,
                                on_event=self.on_event, beat=self.beat,
                                pause_check=lambda: self.needs_heal() and not self._suppress_heal,
                                stuck_check=lambda: self._stuck_request is not None,
                                blocked_npcs=self._blocked_npcs)   # LAYER A: shared route-around memory
        # PHASE 2 — HM field-move actuator (Cut/Strength/Surf via the in-game prompt path).
        # Pure-additive; detection is always safe, actuation is gated by FIELD_MOVES_ENABLED.
        try:
            from field_moves import FieldMoveActuator
            self.field = FieldMoveActuator(self)
        except Exception:
            self.field = None

    def _observed_battle_runner(self):
        """Phase-2 wrapper around the injected battle-runner: snapshot the foe at battle start, run the
        REAL battle untouched, then record win/loss into strategic memory. Pure-additive — the runner's
        behavior + return value are unchanged (battle suites stay green); any strat error is swallowed
        LOUD and never affects the fight. Called only when a battle is live (in_battle); a stray
        non-battle call no-ops safely (observe_battle_start finds no enemy → records nothing)."""
        pre_sp = pre_lvl = None
        try:
            mid = tv.map_id(self.b)
            place = self._PLACE_NAMES.get(mid, "an unfamiliar area")
            pre_sp = st.read_party_species(self.b, 0)          # Phase 3A: lead species + level before battle
            pre_lvl = self.b.rd8(ram.GPLAYER_PARTY + self._PARTY_LEVEL_OFF)
            # Batch-4 Phase 2: feed the SPATIAL wall + HER strength snapshot so a loss becomes a place she
            # can route around, and the gate can later judge "grown enough to retry".
            self.strat.observe_battle_start(self.b, place, map_id=mid, coords=tv.coords(self.b),
                                            party_count=self.b.rd8(ram.GPLAYER_PARTY_CNT), lead_level=pre_lvl)
        except Exception as _e:
            log(f"   [strat] start-observe skipped: {_e}")
        # B-4 — GARY DETECTION (at the start, while the enemy party is loaded): a trainer fight whose
        # party contains a starter-line ace IS the rival. Capture his active lead name for the beat.
        _is_rival, _riv_lead_nm = False, ""
        try:
            if self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08:        # trainer battle
                _is_rival = any(st.read_enemy_species(self.b, s) in _STARTER_LINES for s in range(6))
                if _is_rival:
                    _rb0 = st.read_battle(self.b)
                    _riv_lead_nm = st.SPECIES_NAME.get(_rb0["enemy"]["species"], "") if _rb0 else ""
        except Exception as _re:
            _is_rival = False
        out = self._raw_battle_runner()
        # ROBUST WALL-RECORD (strategic-stuck keystone — the swallow guard). _finish() returns "loss"
        # ONLY if it captures ours.hp==0 at the battle-transition boundary; a flaky read there falls
        # through to "ended" and SWALLOWS the loss, so active_wall never sets and the strategic-stuck
        # floor is silently starved (the death-loop never breaks). A TRAINER battle is un-fleeable and
        # only ends by WIN (enemy_fainted -> "win") or LOSS — so an ambiguous "ended" from a trainer
        # fight is almost certainly a swallowed loss. Coerce the RECORDED outcome to "loss" (loud), but
        # return the ORIGINAL `out` so no downstream control-flow changes. Wilds untouched ("ended" on a
        # wild is normal). strat.cur is still set here (observe_battle_end clears it).
        record_out = out
        try:
            if out == "ended" and self.strat.cur and self.strat.cur.get("is_trainer"):
                log("   [strat] !! trainer battle returned ambiguous 'ended' (likely a loss swallowed at "
                    "the whiteout boundary) -> RECORDING as LOSS so the strategic-stuck floor isn't starved")
                record_out = "loss"
        except Exception as _ce:
            log(f"   [strat] outcome-coercion skipped: {_ce}")
        try:
            self._drive_evolution(pre_sp, pre_lvl)             # Phase 3A: drive a post-battle evolution (gated on level-up)
        except Exception as _e:
            log(f"   [evolve] drive skipped: {_e}")
        try:
            self.strat.observe_battle_end(self.b, record_out)   # record_out = swallow-guarded outcome
        except Exception as _e:
            log(f"   [strat] end-observe skipped: {_e}")
        # B-4 — GARY NEMESIS ARC: record the rival encounter at EVERY fight (not just the opening) so the
        # persisted grudge actually ESCALATES across the run, and voice the escalating-grudge beat.
        if _is_rival:
            try:
                self.strat.note_rival_encounter(
                    won=(str(record_out).lower() == "win"), place=place, lead=_riv_lead_nm,
                    my_party=self.b.rd8(ram.GPLAYER_PARTY_CNT), my_level=pre_lvl)
                gn = self.strat.rival_grudge_note()
                if gn:
                    self.on_event(gn, kind="rival", tier=3)
                log(f"   [strat] GARY encounter #{len(self.strat.rival['encounters'])} recorded "
                    f"({record_out}) at {place} — grudge escalates")
            except Exception as _re:
                log(f"   [strat] rival-encounter record skipped: {_re}")
        # Did the in-battle path record this as a loss? The free_roam whiteout-backstop reads this so it
        # only force-records when the in-battle read SWALLOWED the loss (no double-count).
        self._loss_recorded_this_battle = str(record_out).lower() in ("loss", "battle_loss", "blackout", "whiteout")
        # PHASE 6 — STRUGGLE: the middle of the dread→struggle→catharsis arc. If the fight left her down
        # to her LAST conscious Pokémon (others fainted), name it — "down to my last one, this is bad" —
        # so a later win is EARNED relief, not a flat "we won". Read-only; throttled to fire ONCE per
        # down-to-last episode (re-arms when she recovers a healthy bench), so it lands instead of nagging.
        try:
            cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            conscious = 0
            for s in range(min(cnt, 6)):
                base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
                if self.b.rd16(base + P_HP) > 0:
                    conscious += 1
            if conscious == 1 and cnt >= 2 and out not in ("battle_loss", "loss", "blackout"):
                if not getattr(self, "_struggle_flagged", False):
                    self._struggle_flagged = True
                    self.on_event("okay this is bad — I'm down to my last one. everyone else is out. "
                                  "if this one drops, we're done. come on…", kind="struggle", tier=2)
            elif conscious >= 2:
                self._struggle_flagged = False             # bench recovered — re-arm for the next scare
        except Exception as _e:
            log(f"   [struggle] check skipped: {_e}")
        return out

    def _advance_dialogue(self, taps=12):
        """Press A to advance/close any open textbox until movement is free again."""
        for _ in range(taps):
            if tv.coords(self.b) is not None and not st.in_battle(self.b):
                # heuristic: if a direction press moves us, dialogue is closed
                pass
            self.b.press("A", 6, 6, self.render, owner="agent")

    def walk_to_map(self, target_map, direction):
        # travel.Traveler now crosses N/S rows AND E/W columns (east = the Pewter->Route 3 unlock).
        if direction not in ("north", "south", "east", "west"):
            log(f"   (unknown edge '{direction}', defaulting 'north')")
            direction = "north"
        # HEAL-AND-RETRY: a lone starter crossing a gauntlet route drops below the heal line mid-leg;
        # travel yields need_heal. Heal at the NEAREST center (location-aware) and retry, so the leg
        # actually completes instead of stopping short (the Route 3 -> Route 4 stall).
        for _ in range(8):
            if tv.map_id(self.b)[0] != 3:           # stuck INSIDE a building (a fragile heal-excursion
                self._exit_to_overworld()           # left her inside) -> get to the overworld first, or
                #                                     travel can never path to a route edge from an interior
            r = self.trav.travel(target_map=target_map, edge=direction)
            if r == "need_heal":
                if self.heal_nearest() == "stuck":
                    log("   !! WALK: heal failed mid-leg - LOUD"); return "stuck"
                continue
            if r in ("no_path", "stuck"):
                # transient NPC-wedge (often a PC door after a heal) — retry; the traveler waits for
                # NPCs each attempt and a fresh plan usually clears once they step off. Bounded by the
                # loop, so a genuine wall still stops loud.
                log(f"   WALK: {r} - retrying (transient NPC-wedge?)")
                continue
            return r
        return "stuck"

    def _center_reachable_here(self):
        """Pure predicate — from the CURRENT position, is a Pokémon Center reachable (is this spot
        HEAL-SAFE)? THE strand guard: a GREEN tick in an un-healable spot must NOT be banked as the
        escape target, or the reload loops straight back into the strand (the observed infinite STALL).
        TWO accurate signals, mirroring what heal_nearest can ACTUALLY do:
          1. OWN-map Center door BFS-reachable (or Route 3, which heals west at Pewter).
          2. An ADJACENT Center-city excursion is crossable: a connection to a CITY_PC_DOORS neighbour
             whose CONNECTION BAND (border rows where the neighbour's overlap tile is walkable — the
             same signal travel's edge-cross uses) has a BFS-reachable tile. Run-4 proof this matters:
             at Route-4 (107,12) the own Center is ledge-isolated but the east excursion to Cerulean
             works (healed + returned twice in the log) — refusing to bank there threw away a full
             L16/16 bench-level on the next revert. NOTE the earlier naive version of (2) ("can reach
             any border tile ⇒ crossable") was WRONG — the BAND is what makes it accurate.
        Over-conservative = skip a bank that tick (harmless); false-'safe' = re-poisons the loop (fatal)."""
        try:
            m = tv.map_id(self.b)
            here = tv.coords(self.b)
            if here is None:
                return False
            if m == ROUTE3:                          # Route 3 heals west at Pewter (its normal connection)
                return True
            grid = tv.Grid(self.b)
            door = CITY_PC_DOORS.get(m)              # 1) own-map Center door reachable from here?
            if door is not None:
                appr = (door[0], door[1] + 1)        # stand just south of the door
                if tv.bfs(grid, here, lambda t, a=appr: t == a, walkable=grid.walkable):
                    return True
            # 2) adjacent Center-city excursion crossable (band-accurate, mirrors heal_nearest)
            _EDGE = {"N": "north", "S": "south", "E": "east", "W": "west"}
            for d, nbr in self._map_connections():
                if nbr not in CITY_PC_DOORS or nbr == m or d not in _EDGE:
                    continue
                if self._edge_band_reachable(grid, here, _EDGE[d]):
                    return True
            # 3) multi-hop: the world graph reaches a known Center city from this map (heal_nearest's
            #    graph-heal path) AND the FIRST hop's edge band is tile-reachable from here. Mirrors what
            #    heal can ACTUALLY do — without this, every Route-25 tile read unsafe and the deep Bill
            #    progress could never bank (run-5: 'snapshot SKIPPED' at (27,6) → the revert lost it all).
            try:
                targets = self.world.reachable_with_trait(tuple(m), "has_pokecenter", None)
                dest = next((tuple(mid) for mid, _nm, _h in targets if tuple(mid) in CITY_PC_DOORS), None)
                if dest is not None:
                    hop = self.world.next_hop(tuple(m), dest, None)
                    if hop is not None:
                        _nxt, edge = hop
                        if self._edge_band_reachable(grid, here, edge):
                            return True
            except Exception:
                pass
            return False
        except Exception as _e:
            log(f"   [safe] center-reachable check errored: {_e!r} — treating as UNSAFE (conservative)")
            return False

    @staticmethod
    def _edge_band_reachable(grid, here, edge):
        """Can `here` BFS-reach a border tile in the map-edge's CONNECTION BAND (the rows/cols where the
        neighbour's overlap tile just past the border is walkable)? The same crossable-edge signal
        travel's edge-goal uses — a border tile OUTSIDE the band is a hard wall, so plain border
        reachability is a false 'crossable' (the (84,15) lesson)."""
        if edge == "east":
            line, past_d, axis = grid.sx_hi, 1, 0
        elif edge == "west":
            line, past_d, axis = grid.sx_lo, -1, 0
        elif edge == "north":
            line, past_d, axis = grid.sy_lo, -1, 1
        else:
            line, past_d, axis = grid.sy_hi, 1, 1
        if axis == 0:
            band = {p for p in range(grid.sy_lo, grid.sy_hi + 1) if grid.walkable(line + past_d, p)}
            goal = lambda t: t[0] == line and t[1] in band
        else:
            band = {p for p in range(grid.sx_lo, grid.sx_hi + 1) if grid.walkable(p, line + past_d)}
            goal = lambda t: t[1] == line and t[0] in band
        if not band:
            return False
        return bool(tv.bfs(grid, here, goal, walkable=grid.walkable))

    def heal_nearest(self):
        """Heal at the Pokemon Center NEAREST the CURRENT map. Returns 'ok' | 'stuck'.

        FIX (was a hardcoded Forest-era table that defaulted EVERY unlisted map — including Cerulean —
        to a cross-region Viridian return): now heals at the CURRENT map's OWN Center when it has one
        (CITY_PC_DOORS), so Cerulean heals in Cerulean. Route 3 has no Center -> cross WEST to Pewter.
        An UNMAPPED map falls back to the Viridian return but logs LOUD (no silent cross-region heal —
        the sleeping-stream bar: a missing city must scream, not quietly walk her across Kanto)."""
        m = tv.map_id(self.b)
        if m == ROUTE3:                                   # Route 3 has no Center: nearest is Pewter (west)
            return "ok" if self._heal_excursion(PEWTER, PEWTER_PC_DOOR, "west",
                                                ROUTE3, "east") == "ok" else "stuck"
        if m in CITY_PC_DOORS:                            # this map HAS its own Center
            door = CITY_PC_DOORS[m]
            grid = tv.Grid(self.b)
            appr = (door[0], door[1] + 1)                # approach tile (stand south of the door)
            if tv.bfs(grid, tv.coords(self.b), lambda t: t == appr, walkable=grid.walkable):
                return ("ok" if self.heal_at_center(door) in ("healed", "healed_stuck_inside")
                        else "stuck")
            # Own Center UNREACHABLE from here — e.g. Route 4's PC (12,5) is across the Mt-Moon ledge
            # split from the EAST grass, so a grind there could never heal and hard-STALLED (the look-
            # ahead wedge). Heal at an ADJACENT city whose Center IS reachable instead of looping on an
            # un-healable route. General split-route danger-awareness fix (uses the live map header, so
            # it works for any future split route, not just Route 4).
            _EDGE = {"N": "north", "S": "south", "E": "east", "W": "west"}
            _REV = {"north": "south", "south": "north", "east": "west", "west": "east"}
            log(f"   HEAL: own Center door {door} UNREACHABLE from {tv.coords(self.b)} on {m} "
                f"-> crossing to an adjacent city to heal")
            for d, nbr in self._map_connections():
                if nbr in CITY_PC_DOORS and nbr != m and d in _EDGE:
                    if self._heal_excursion(nbr, CITY_PC_DOORS[nbr], _EDGE[d], m, _REV[_EDGE[d]]) == "ok":
                        return "ok"
            log(f"   !! HEAL: no reachable adjacent Center from {m} (conns={self._map_connections()}) — LOUD")
            # RULE 17 / STUCK-SPIN BREAKER (2026-07-05): a TRUE strand (no Center reachable ANYWHERE — e.g. a
            # faint in a one-way-ledge grass pocket) must NOT return 'stuck' and spin the roam loop on an
            # un-healable 'heal' forever (unwatchable + unstreamable). Force an autonomous reload to a banked
            # SAFE checkpoint and continue from there. FIX (2026-07-05): the recent-good snapshot is now
            # poison-free (only banked from Center-reachable spots — see _center_reachable_here), so the
            # escape-hatch reloads OUT of the pocket instead of straight back into it (was the infinite
            # reload STALL). Belt-and-suspenders: if recent-good is absent/declines, fall back to the
            # deep-wedge RING (gain-seam checkpoints — a guaranteed-safe city like post-Misty Cerulean).
            try:
                if getattr(self, "_last_good_state", None) is not None and self._escape_hatch_reload():
                    log("   HEAL: TRUE strand -> escape-hatch reloaded to last good checkpoint (recovered)")
                    return "ok"
                if len(getattr(self, "_safe_ring", ()) or ()) and self._deep_wedge_revert():
                    log("   HEAL: TRUE strand -> deep-wedge ring reverted to a guaranteed-safe checkpoint (recovered)")
                    return "ok"
            except Exception as _he:
                log(f"   HEAL: escape-hatch on strand crashed: {_he!r} (LOUD) — falling through to stuck")
            return "stuck"
        # UNMAPPED map, ADJACENT-CITY FIRST (2026-07-06, the Route-6/S.S.-Anne lesson): the LIVE map
        # header knows neighbours she hasn't VISITED yet — the world graph below only routes VISITED
        # nodes, so a first approach to a new town (Route 6 hurt from the gauntlet, Vermilion one edge
        # south) found "no graph route" and the blind warp-south fallback BOARDED THE BOAT. One edge
        # cross to a registered-Center neighbour, heal, return — the Route-3->Pewter pattern, general.
        _EDGEU = {"N": "north", "S": "south", "E": "east", "W": "west"}
        _REVU = {"north": "south", "south": "north", "east": "west", "west": "east"}
        for d, nbr in self._map_connections():
            if tuple(nbr) in CITY_PC_DOORS and tuple(nbr) != tuple(m) and d in _EDGEU:
                log(f"   HEAL: unmapped {m} — ADJACENT city {self.world.name(tuple(nbr))} ({d}) has a "
                    f"registered Center; one-edge heal excursion")
                if self._heal_excursion(tuple(nbr), CITY_PC_DOORS[tuple(nbr)], _EDGEU[d],
                                        m, _REVU[_EDGEU[d]]) == "ok":
                    return "ok"
        # UNMAPPED map (2026-07-05, the Route-25 heal->stuck x10): route via HER WORLD GRAPH to the
        # nearest VISITED Center city (Route 25 -> Route 24 -> Cerulean), multi-hop, fleeing wilds.
        # General: works for every future unmapped route she's walked to. Does NOT walk back afterward —
        # the roam re-derives position next tick (beaten trainers make the re-walk cheap).
        try:
            targets = self.world.reachable_with_trait(tuple(m), "has_pokecenter", None)
        except Exception as _wt:
            log(f"   !! HEAL: world-graph Center lookup failed ({_wt})"); targets = []
        dest = next((tuple(mid) for mid, _nm, _h in targets if tuple(mid) in CITY_PC_DOORS), None)
        if dest is not None:
            log(f"   HEAL: no local Center mapped for {m} — world-graph routing to "
                f"{self.world.name(dest)}'s Center (multi-hop, fleeing wilds)")
            saved, saved_runner = self._suppress_heal, self.trav.battle_runner
            self._suppress_heal = True
            self.trav.battle_runner = self._flee_runner
            try:
                for _hop in range(8):
                    cur = tuple(tv.map_id(self.b))
                    if cur == dest:
                        break
                    hop = self.world.next_hop(cur, dest, None)
                    if hop is None:
                        log(f"   !! HEAL: graph route to {dest} broke at {cur} (LOUD)"); break
                    nxt, edge = hop
                    if self.trav.travel(target_map=nxt, edge=edge) == "battle_loss":
                        return "ok"        # blacked out en route -> respawn auto-heals at a Center
                if tuple(tv.map_id(self.b)) == dest:
                    return ("ok" if self.heal_at_center(CITY_PC_DOORS[dest])
                            in ("healed", "healed_stuck_inside") else "stuck")
            finally:
                self._suppress_heal, self.trav.battle_runner = saved, saved_runner
        # LAST resort: the old Viridian return, LOUD (constraint #3 — never silent-degrade)
        log(f"   !! HEAL: no graph route to any known Center from {m} — FALLBACK to a cross-region "
            f"Viridian heal (FIX: add {m}'s PC door to CITY_PC_DOORS)")
        return "ok" if self.return_to_center() not in ("stuck", "battle_loss") else "stuck"

    def enter_warp(self, prefer="nearest", pick=None, budget_s=None):
        """REAL warp entry: find door/warp tiles (behavior 0x6x), walk to the tile just
        beside a chosen one, step INTO it, and confirm the map flips. `pick` (x,y) targets a
        specific door; else `prefer` = 'nearest' / 'north' (forward-progress gates are the
        northmost door, approached from the south, stepped UP) / 'south' (the heal-return exit,
        the southmost door, approached from the north, stepped DOWN). `budget_s` overrides the
        approach leg's wall-clock budget — a TRAINER-GAUNTLET approach (Route 25 to Bill: 6+
        fights en route) blows the 300s default at half-distance (run-5: aborted at step 72,
        door 28 tiles away, 'no reachable door warped')."""
        before = tv.map_id(self.b)
        doors = self._door_tiles()
        if not doors:
            log("   no door/warp tiles (0x6x) found on this map"); return "no_warp"
        cur = tv.coords(self.b)
        if pick is not None:
            order = [pick]
        elif prefer == "north":
            order = sorted(doors, key=lambda t: (t[1], abs(t[0] - cur[0])))
        elif prefer == "south":
            order = sorted(doors, key=lambda t: (-t[1], abs(t[0] - cur[0])))
        else:
            order = sorted(doors, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        # approach from the side OPPOSITE the travel direction, then step INTO the door.
        # north/nearest: stand south of the door, step UP. south: stand north, step DOWN.
        approach_dy, step_key = (-1, "DOWN") if prefer == "south" else (1, "UP")
        log(f"   {len(doors)} door(s); candidates (chosen order): {order}")
        for door in order:
            approach = (door[0], door[1] + approach_dy)  # stand just beside the door
            # FAST reachability pre-check (BFS only) so unreachable doors are skipped
            # instantly instead of a slow walk-then-fail. Uses grass-ALLOWED collision
            # (grid.walkable, not _safe): grass-heavy maps like Viridian Forest are
            # wall-to-wall tall grass, so a grass-free pre-check wrongly skips every door.
            grid = tv.Grid(self.b)
            if not tv.bfs(grid, tv.coords(self.b), lambda t: t == approach,
                          walkable=grid.walkable):
                continue
            log(f"   reachable door {door}: walking to approach {approach}")
            r = self.trav.travel(target_map=None, arrive_coord=approach, max_steps=300,
                                 max_seconds=(budget_s or 300))
            if r == "need_heal":
                return "need_heal"          # heal interrupt during the approach -> let caller heal
            if r != "arrived":
                continue
            for _ in range(5):                           # step INTO the doorway -> warp
                self.b.press(step_key, 8, 8, self.render, owner="agent")
                self._advance_dialogue(taps=2)           # gate buildings may print a line
                if tv.map_id(self.b) != before:
                    log(f"   WARPED {before} -> {tv.map_id(self.b)} via door {door}")
                    return "warped"
        log(f"   no reachable door warped (entry geometry?) - LOUD")
        return "no_warp"

    def _door_tiles(self):
        ml = self.b.rd32(0x02036DFC)
        attr = (self.b.rd32(self.b.rd32(ml + 0x10) + 0x14),
                self.b.rd32(self.b.rd32(ml + 0x14) + 0x14))
        w = self.b.rd32(tv.BACKUP_LAYOUT); h = self.b.rd32(tv.BACKUP_LAYOUT + 4)
        mp = self.b.rd32(tv.BACKUP_LAYOUT + 8)
        out = []
        for by in range(h):
            for bx in range(w):
                e = self.b.rd16(mp + (bx + w * by) * 2); mid = e & 0x3FF
                base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
                if 0x60 <= (self.b.rd32(base + idx * 4) & 0xFF) <= 0x6F:
                    out.append((bx - 7, by - 7))
        return out

    # DIRECTIONAL WARP TILES (pret metatile_behaviors.h): stair/arrow warps fire ONLY when entered
    # moving in their direction — the UGP tunnel doorway is 0x6F MB_DOWN_LEFT_STAIR_WARP (enter
    # moving WEST); walking onto it any other way just walks THROUGH it (run-9 lesson).
    _WARP_ENTRY = {0x62: ("RIGHT", (1, 0)), 0x63: ("LEFT", (-1, 0)), 0x64: ("UP", (0, -1)),
                   0x65: ("DOWN", (0, 1)), 0x6C: ("RIGHT", (1, 0)), 0x6D: ("LEFT", (-1, 0)),
                   0x6E: ("RIGHT", (1, 0)), 0x6F: ("LEFT", (-1, 0)),
                   # ESCALATORS 0x6A/0x6B (the Center 1F<->2F Cable Club pair; live-derived on the
                   # Vermilion Center 2026-07-06): board from the EAST — step LEFT onto the tile
                   # (the south approach is collision-blocked; UP from below never enters), and the
                   # warp fires ~60-120 frames AFTER standing on it — the on-tile wait below.
                   0x6A: ("LEFT", (-1, 0)), 0x6B: ("LEFT", (-1, 0))}

    def _tile_behavior(self, sx, sy):
        """Metatile behavior byte at save coords (the same read Grid/doors use). 0 on failure."""
        try:
            ml = self.b.rd32(0x02036DFC)
            attr = (self.b.rd32(self.b.rd32(ml + 0x10) + 0x14),
                    self.b.rd32(self.b.rd32(ml + 0x14) + 0x14))
            w = self.b.rd32(tv.BACKUP_LAYOUT)
            mp = self.b.rd32(tv.BACKUP_LAYOUT + 8)
            e = self.b.rd16(mp + ((sx + 7) + w * (sy + 7)) * 2)
            mid = e & 0x3FF
            base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
            return self.b.rd32(base + idx * 4) & 0xFF
        except Exception:
            return 0

    def _enter_directional_warp(self, wt):
        """Enter a stair/arrow warp tile the way the game demands: stand on the tile OPPOSITE its
        entry direction and step INTO it. Returns True iff the map flipped."""
        bh = self._tile_behavior(*wt)
        ent = self._WARP_ENTRY.get(bh)
        if ent is None:
            return False
        key, (dx, dy) = ent
        stand = (wt[0] - dx, wt[1] - dy)
        m0 = tuple(tv.map_id(self.b))
        if self.trav.travel(target_map=None, arrive_coord=stand, max_steps=120,
                            max_seconds=60) != "arrived":
            return False
        # 6 presses, not 3: the first press after travel is a TURN (and is routinely EATEN — the
        # face-verified-turn class), the next steps ONTO the tile, and an arrow warp (0x65 mats)
        # fires only on a FURTHER press while standing on it — 3 ran out one press short.
        for _ in range(6):
            self.b.press(key, 8, 10, self.render, owner="agent")
            for _ in range(30):
                self.b.run_frame()
                self.render()
            # DELAYED warps (escalators): the map flips a beat AFTER standing on the tile — if
            # we're ON it, wait it out; a re-press would walk OFF the tile and lose the entry.
            if tuple(tv.map_id(self.b)) == m0 and tuple(tv.coords(self.b) or ()) == tuple(wt):
                for _ in range(240):
                    self.b.run_frame()
                    if tuple(tv.map_id(self.b)) != m0:
                        break
            if tuple(tv.map_id(self.b)) != m0:
                for _ in range(60):                   # SETTLE the fade-in: the map header/layout
                    self.b.run_frame()                # pointers are mid-transition the instant the
                #                                       id flips — a back-to-back tile read gets 0s
                log(f"   directional warp {wt} (behavior 0x{bh:02x}) entered via {key}")
                return True
        return False

    # ── THE DOOR PASS-THROUGH PRIMITIVE (2026-07-06 — the south-gate class, GENERAL) ──────────────
    # GROUND TRUTH (recon_south_geometry + pret map.json + Bulbapedia): post-ticket Cerulean's ONLY
    # pre-Cut route south is THROUGH the burgled house — front door (30,11) → interior → back door →
    # the fenced garden → the east corridor (cols 39-40) → the fence crossing at (39-40,32) → the
    # south strip → Route 5. The overworld BFS can NEVER see this (the region is fence-isolated), so
    # when an edge crossing reports no-route, doors are the remaining connectors. The SAME shape is
    # the Underground Path huts and the Saffron gatehouses — build once, reuse everywhere.
    def _door_passthrough(self, budget_s=240):
        """Enter a reachable, untried door on this map; inside, exit through a DIFFERENT warp; if we
        pop out materially elsewhere (across a fence / on another map), report 'crossed' so the
        caller retries its edge crossing from the new region. Bounded (each door once per map per
        session); on a dead end she exits back out the way she came (and roam's blackout-recovery
        auto-exits interiors anyway). Returns 'crossed' | 'no_passthrough' | 'need_heal'."""
        b = self.b
        m0 = tuple(tv.map_id(b))
        pos0 = tuple(tv.coords(b))
        if not hasattr(self, "_pt_tried"):
            self._pt_tried = {}
        tried = self._pt_tried.setdefault(m0, set())
        grid = tv.Grid(b)
        # connector fingerprint: a building with SEVERAL warp tiles on this map (front + back doors,
        # like the burgled house's three) is the likeliest pass-through — try those FIRST.
        dest_doors = {}
        for (wxy, wdest, _wid) in tv.read_warps(b):
            dest_doors.setdefault(tuple(wdest), set()).add(tuple(wxy))
        door_dest = {t: dest for dest, ts in dest_doors.items() for t in ts}
        # Candidates = ALL warp events (read_warps), not just door-behavior tiles — the Underground
        # Path hut is a plain warp mat _door_tiles can't see (run-6 lesson). Reachable = the tile
        # itself or ANY orthogonal neighbor (gates face north; the old south-approach convention
        # skipped them). Entry method dispatches per kind below.
        door_behav = {tuple(d) for d in self._door_tiles()}
        cands = []
        for wt in door_dest:
            if wt in tried:
                continue
            if tv.bfs(grid, pos0, lambda t, w=wt: t == w or
                      (abs(t[0] - w[0]) + abs(t[1] - w[1]) == 1), walkable=grid.walkable):
                cands.append(wt)
        cands.sort(key=lambda t: (-len(dest_doors.get(door_dest.get(tuple(t)), ())),
                                  abs(t[0] - pos0[0]) + abs(t[1] - pos0[1])))
        # a connector that WORKED here before is remembered — try it first, always (re-attempts after
        # a heal interrupt must not burn the search again or skip the proven door via the tried-set)
        known = getattr(self, "_pt_known", {}).get(m0)
        if known:
            cands = [known] + [c for c in cands if tuple(c) != tuple(known)]
        if not cands:
            # PHASE 2 — REGION RE-ENTRY (2026-07-06, the Route-5 middle-terrace class): a warp on THIS
            # map can sit in a region only enterable from the NEIGHBOR map at aligned columns (the UGP
            # hut: middle corridor, entered from Cerulean's strip; crossing offset +6). For each
            # locally-unreachable warp: cross back to the neighbor, walk to the border column that
            # aligns with the warp's x, re-cross, and see if the warp became reachable.
            far = [wt for wt in door_dest if wt not in tried]
            far.sort(key=lambda t: -len(dest_doors.get(door_dest.get(tuple(t)), ())))
            for wt in far:
                tried.add(wt)
                if self._reenter_at_column(wt) == "entered":
                    grid2 = tv.Grid(b)
                    if tv.bfs(grid2, tuple(tv.coords(b)), lambda t, w=wt: t == w or
                              (abs(t[0] - w[0]) + abs(t[1] - w[1]) == 1), walkable=grid2.walkable):
                        tried.discard(wt)
                        cands = [wt]
                        pos0 = tuple(tv.coords(b))
                        log(f"   PASSTHROUGH: region re-entry unlocked warp {wt} — proceeding")
                        break
            if not cands:
                return "no_passthrough"
        log(f"   PASSTHROUGH: overworld can't reach the target — trying {len(cands)} door(s) as "
            f"connectors (a building can open onto the far side)")
        self.on_event("hmm — no way through out here. maybe one of these buildings has a back way…",
                      kind="route", tier=1)
        for door in cands:
            if tuple(tv.map_id(b)) != m0:
                # a previous candidate left us INSIDE — recover to the overworld before continuing
                self._exit_to_overworld()
                if tuple(tv.map_id(b)) != m0:
                    log("   PASSTHROUGH: !! stranded off the start map — aborting the search LOUD "
                        "(roam's recovery owns it)")
                    return "no_passthrough"
            tried.add(tuple(door))
            def _settle(frames):
                for _ in range(frames):
                    b.run_frame()
                    self.render()
                return tuple(tv.map_id(b))
            if tuple(door) in door_behav:
                r = self.enter_warp(pick=door, budget_s=budget_s)
                if r == "need_heal":
                    return "need_heal"
                if r != "warped":
                    continue
            else:
                # a plain warp mat (hut stairs / hole): walk ONTO it; nudge through if standing on it
                r = self.trav.travel(target_map=None, arrive_coord=tuple(door),
                                     max_steps=300, max_seconds=budget_s)
                if r == "need_heal":
                    return "need_heal"
                if _settle(30) == m0:
                    if tuple(tv.coords(b)) == tuple(door):
                        for key in ("DOWN", "UP", "LEFT", "RIGHT"):
                            self.b.press(key, 8, 8, self.render, owner="agent")
                            if _settle(36) != m0:
                                break
                    if tuple(tv.map_id(b)) == m0:
                        continue                              # never fired — not a usable connector
            # MULTI-HOP WARP-WALK (bounded): keep taking the farthest-unentered warp of each interior
            # until we pop back onto the OVERWORLD. Depth 1 = the burgled house (house -> garden);
            # depth 3 = the Underground Path (hut -> tunnel -> hut -> the far route). A room whose only
            # exits lead back where we came from dead-ends the walk (the Center-upstairs case).
            dead_end = False
            for hop in range(6):
                m_in = tuple(tv.map_id(b))
                if m_in[0] == 3:
                    break                                     # back on the overworld — evaluate below
                spawn = tuple(tv.coords(b))
                exits = sorted((tuple(w[0]) for w in tv.read_warps(b)),
                               key=lambda t: -(abs(t[0] - spawn[0]) + abs(t[1] - spawn[1])))
                moved = False
                for wt in exits:
                    if abs(wt[0] - spawn[0]) + abs(wt[1] - spawn[1]) <= 1:
                        continue                              # the warp we arrived by — skip
                    self.trav.travel(target_map=None, arrive_coord=wt, max_steps=260, max_seconds=110)
                    if _settle(30) != m_in:
                        moved = True
                        break                                 # walking onto the mat warped us
                    if tuple(tv.coords(b)) == wt:             # standing ON it: nudge through
                        # one press at a time, SETTLED before judging — a press queued during the
                        # warp fade executes on arrival and can step us onto the twin mat (the
                        # (31,9) re-entry bug: she popped out and instantly walked back inside).
                        for key in ("UP", "DOWN", "LEFT", "RIGHT"):
                            self.b.press(key, 8, 8, self.render, owner="agent")
                            if _settle(36) != m_in:
                                break
                        if tuple(tv.map_id(b)) != m_in:
                            moved = True
                            break
                    # DIRECTIONAL stair/arrow warp (the UGP tunnel doorway 0x6F): enter moving in
                    # its direction — walk-onto and UP-entry both walk THROUGH it.
                    if self._enter_directional_warp(tuple(wt)) and tuple(tv.map_id(b)) != m_in:
                        moved = True
                        break
                    # DOOR-TYPE inner warp: fires only when ENTERED from its front tile — the
                    # stand-beside + step-in ritual, not walk-onto.
                    if self.enter_warp(pick=tuple(wt), budget_s=90) == "warped" \
                            and tuple(tv.map_id(b)) != m_in:
                        moved = True
                        break
                if not moved:
                    dead_end = True
                    break
            _settle(60)                                       # finish the door-exit walk animation
            m_out, pos1 = tuple(tv.map_id(b)), tuple(tv.coords(b))
            if dead_end or m_out[0] != 3:
                # no onward exit (single-door house / Center upstairs loop) — recover and try the next
                log(f"   PASSTHROUGH: {door} dead-ends inside ({m_out}) — backing out")
                self._exit_to_overworld()
                continue
            if m_out != m0 or abs(pos1[0] - pos0[0]) + abs(pos1[1] - pos0[1]) > 6:
                log(f"   PASSTHROUGH: CROSSED via door {door}: {m0}@{pos0} -> {m_out}@{pos1}")
                self.on_event("ha — knew it! through the building and out the other side. "
                              "the road's open.", kind="route", tier=2)
                # remember the proven connector (re-attempts after heal/battle interrupts reuse it)
                if not hasattr(self, "_pt_known"):
                    self._pt_known = {}
                self._pt_known[m0] = tuple(door)
                tried.discard(tuple(door))
                # (the roam tick's note_visit warp-learning records the connector in the graph)
                return "crossed"
            # popped out beside where we entered (same building, two front doors) — keep trying
            log(f"   PASSTHROUGH: door {door} popped out beside the entry ({pos1}) — not a crossing")
        return "no_passthrough"

    def _reenter_at_column(self, warp_tile):
        """REGION RE-ENTRY: leave this map via its NORTH/SOUTH connection, walk the neighbor's border
        to the column aligned with `warp_tile` (using the real connection OFFSET from the header), and
        step back across — landing in the fenced region that holds the warp. Returns 'entered'|'failed'.
        First instance: Route 5's middle terrace (the UGP hut) from Cerulean's strip, offset +6."""
        b = self.b
        m0 = tuple(tv.map_id(b))
        GH = 0x02036DFC
        try:
            cp = b.rd32(GH + 0x0C)
            cnt, arr = b.rd32(cp), b.rd32(cp + 4)
            conns = []
            for i in range(min(cnt, 8)):
                e = arr + i * 12
                d = b.rd32(e)
                off = b.rd32(e + 4)
                if off >= 0x80000000:
                    off -= 0x100000000
                conns.append((d, off, (b.rd8(e + 9), b.rd8(e + 8))))
        except Exception:
            return "failed"
        # prefer the NORTH neighbor (2) then SOUTH (1) — the region-split class is a N/S phenomenon
        for want_dir, back_edge, back_word in ((2, "south", "north"), (1, "north", "south")):
            hit = next(((off, m) for d, off, m in conns if d == want_dir), None)
            if hit is None:
                continue
            off, nmap = hit
            # cross to the neighbor first (any reachable band tile will do)
            if self.trav.travel(target_map=tuple(nmap), edge=back_word, max_seconds=180) != "arrived":
                continue
            # on the neighbor: this map's column x maps to neighbor column x + off... the offset is
            # applied entering m0 FROM the neighbor as (neighbor_x - off) -> m0_x, so target
            # neighbor_x = warp_x + off. Walk to the border row at that column, then step across.
            # offset SIGN is convention-ambiguous — hedge both; the post-re-entry BFS verifies anyway
            tx = warp_tile[0] + off
            tx2 = warp_tile[0] - off
            g2 = tv.Grid(b)
            border_y = g2.sy_hi if back_word == "north" else 0
            goal_cols = list(dict.fromkeys(list(range(tx - 2, tx + 3)) + list(range(tx2 - 2, tx2 + 3))))
            got = None
            for cx in sorted(goal_cols, key=lambda c: min(abs(c - tx), abs(c - tx2))):
                if self.trav.travel(target_map=None, arrive_coord=(cx, border_y),
                                    max_steps=400, max_seconds=150) == "arrived":
                    got = cx
                    break
            if got is None:
                continue
            step_key = "DOWN" if back_word == "north" else "UP"
            for _ in range(4):
                self.b.press(step_key, 8, 10, self.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                    self.render()
                if tuple(tv.map_id(b)) == m0:
                    log(f"   PASSTHROUGH: re-entered {m0} at column {got} (offset {off:+d}) "
                        f"targeting warp {warp_tile}")
                    return "entered"
            # couldn't step across at this column — give up this direction
        return "failed"

    def _edge_travel(self, target_map, edge, budget_s=None):
        """An edge hop with the pass-through fallback: when the crossing reports a hard no-route
        (fence/tree/NPC-walled region), try a building connector, then retry the crossing once.
        Both legs AVOID warp tiles (the Mt-Moon lesson: pathing across a door mat fires it — the
        garden retry walked straight back into the burgled house without this)."""
        kw = {} if budget_s is None else {"max_seconds": budget_s}
        avoid = {tuple(w[0]) for w in tv.read_warps(self.b)}
        r = self.trav.travel(target_map=target_map, edge=edge, avoid=avoid, **kw)
        hard_noroute = (r in ("no_route_hm_blocked", "no_route_npc_blocked")
                        or (r in ("no_path", "stuck")
                            and getattr(self.trav, "last_fail_reason", "") in ("no_route", "npc_blocked")))
        if hard_noroute:
            pt = self._door_passthrough()
            if pt == "need_heal":
                return "need_heal"
            if pt == "crossed":
                if tuple(tv.map_id(self.b)) == tuple(target_map):
                    return "arrived"                           # the connector itself crossed the map
                avoid = {tuple(w[0]) for w in tv.read_warps(self.b)}
                return self.trav.travel(target_map=target_map, edge=edge, avoid=avoid, **kw)
        return r

    def advance_north(self, target_map, max_legs=60):
        """Generic 'head north until target_map' - auto-picks WALK (cross a route edge)
        vs WARP (step through a gate-house door) at each map, so the route->gate-house->
        route->forest chain self-discovers instead of being hardcoded leg by leg. The
        Traveler handles BFS/maze/NPC/grass/battle-handoff; this only sequences legs."""
        for leg in range(max_legs):
            m = tv.map_id(self.b)
            if m == target_map:
                return "arrived"
            out = self.trav.travel(target_map=target_map, max_steps=800)
            if out == "arrived":
                continue                       # crossed an edge (maybe more legs north)
            if out == "battle_loss":
                return "battle_loss"
            if out == "need_heal":             # heal-when-low: go south, heal, resume north
                r = self.return_to_center()
                if r in ("stuck", "battle_loss"):
                    return r
                continue
            if out in ("no_path", "stuck"):
                # no walkable north edge here -> it's an interior/gate house: warp north
                log(f"   leg {leg}: no north edge on map {m} - warping north")
                w = self.enter_warp(prefer="north")
                if w == "need_heal":            # got low crossing toward the door -> heal, retry
                    r = self.return_to_center()
                    if r in ("stuck", "battle_loss"):
                        return r
                    continue
                if w != "warped":
                    log(f"   !! ADVANCE stuck on map {m} (no north edge, no north warp)")
                    return "stuck"
        return "timeout"

    HEAL_HP_FRAC = 0.75     # heal-return when lead HP drops below this fraction of max. The
                            # return FLEES wild fights (see _flee_runner) so the retreat costs
                            # ~0 HP - but forest TRAINERS can't be fled, so we keep the trigger
                            # high enough that she meets a return-trainer near full HP and wins
                            # (esp. the fragile first foray at L8).

    # ── BATCH 2 PART A: PARTY-WIDE "take care of yourself" thresholds (named, tunable). needs_heal()
    # above stays LEAD-ONLY (it gates travel's pause_check + the heal-excursions all over the spine —
    # proven, must not change). These ADD a party-wide survival signal that free_roam surfaces to the
    # oracle so a badly-hurt Kira reliably picks "heal" before wandering into grass — the documented
    # 7%-HP-boot blackout class (_boot_state_sanity screams about it; PART A makes her ACT on it).
    PARTY_HURT_FRAC = 0.50   # ANY party member below this fraction of max -> "hurt" (heal offered)
    PARTY_CRIT_FRAC = 0.30   # ANY alive member at/under this (or any fainted) -> CRITICAL (heal surfaced HARD)

    def needs_heal(self):
        """True when the lead mon is low enough to risk a blackout. HP-gated: HP depletes
        faster than Tackle's 35 PP, so an HP heal-return tops up PP as a side effect (keeping
        Tackle her Forest move and the Vine-Whip swap a Brock-only tool)."""
        hp, mx = self.lead_hp()
        return mx > 0 and hp < self.HEAL_HP_FRAC * mx

    def party_health(self):
        """Per-member (slot, hp, maxhp, frac) for every party slot with maxhp>0. Same read used by
        _boot_state_sanity / _healthy_reserve_slot (cur HP 0x56, max 0x58 in the 100-byte struct)."""
        out = []
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            hp, mx = self.b.rd16(base + P_HP), self.b.rd16(base + P_MAXHP)
            if mx > 0:
                out.append((s, hp, mx, hp / mx))
        return out

    def _hurt_severity(self):
        """PARTY-WIDE survival signal for the free-roam oracle. Returns (severity, note): severity is
        None | 'hurt' | 'critical'. 'critical' = a member is at/under PARTY_CRIT_FRAC or fainted (the
        blackout-risk band a real player NEVER walks into grass on); 'hurt' = a member under
        PARTY_HURT_FRAC. The note NAMES the worst mon so her pick is grounded in her real situation
        (capability-not-script: this only INFORMS the oracle; she still chooses)."""
        ph = self.party_health()
        if not ph:
            return (None, "")
        s, hp, mx, frac = min(ph, key=lambda t: t[3])     # the worst-off member
        nm = st.SPECIES_NAME.get(st.read_party_species(self.b, s), f"slot{s}").title()
        fainted = any(h == 0 for _, h, _, _ in ph)
        if frac <= self.PARTY_CRIT_FRAC or fainted:
            return ("critical",
                    f"Your team is badly hurt — {nm} is at {hp}/{mx} HP"
                    f"{' and a teammate has fainted' if fainted else ''}. A real trainer heals at a "
                    f"Pokémon Center before doing ANYTHING else — wandering into grass this hurt risks "
                    f"a blackout.")
        if frac < self.PARTY_HURT_FRAC:
            return ("hurt",
                    f"{nm} is hurt ({hp}/{mx} HP) — topping up at a Pokémon Center would be smart "
                    f"before pushing on.")
        return (None, "")

    def _heal_excursion(self, city, pc_door, out_edge, back_map, back_edge):
        """Low HP on a ROUTE with no PC of its own: cross to an ADJACENT city, heal at its PC, cross
        back to the route. Generic city-heal for routes (return_to_center is the Viridian-south
        special case for the pre-Pewter arc). Flees wilds both ways so the trip costs ~0 HP. Returns
        'ok' (back on back_map, healed) | 'stuck'."""
        saved, saved_runner = self._suppress_heal, self.trav.battle_runner
        self._suppress_heal = True                    # don't re-trigger heal WHILE going to heal
        self.trav.battle_runner = self._flee_runner
        try:
            log(f"   HEAL-EXCURSION: lead at {self.lead_hp()} - crossing {out_edge} to {city} to heal")
            for _ in range(6):
                if tv.map_id(self.b) == city:
                    break
                if self.trav.travel(target_map=city, edge=out_edge,
                                    max_steps=400, max_seconds=120) != "arrived":
                    break
            if tv.map_id(self.b) != city:
                log(f"   !! HEAL-EXCURSION: didn't reach {city} (at {tv.map_id(self.b)})")
                return "stuck"
            self.heal_at_center(pc_door)
            healed = self._party_fully_healed()                # whole party topped up = the heal succeeded
            for _ in range(6):
                if tv.map_id(self.b) == back_map:
                    break
                if self.trav.travel(target_map=back_map, edge=back_edge,
                                    max_steps=400, max_seconds=120) != "arrived":
                    break
            if tv.map_id(self.b) == back_map:
                log(f"   HEAL-EXCURSION: healed + back on {back_map} at {tv.coords(self.b)}")
                return "ok"
            # The HEAL succeeded; the cross-back wedged (typically an NPC parked at the PC door). That
            # is NOT a hard stall — return ok so the caller re-navs toward the real goal FROM HERE
            # (moving away from the door), retrying as the NPC steps off. Heal is best-effort transport.
            log(f"   HEAL-EXCURSION: healed (HP {self.lead_hp()}) but still on {tv.map_id(self.b)} - "
                f"non-fatal, caller re-navs")
            return "ok" if healed else "stuck"
        finally:
            self._suppress_heal, self.trav.battle_runner = saved, saved_runner

    def return_to_center(self):
        """Heal-return to the Viridian Center (the only PC before Pewter), then heal (restores HP +
        PP). DIRECTION-AWARE: Viridian is NORTH when we're on Route 1 / in Pallet (the parcel arc) and
        SOUTH when we're up in Route 2 / the Forest (the Brock arc). The old hard-coded 'south' bounced
        the agent toward Pallet whenever it got low on Route 1. Cross that edge (or warp it) until we're
        on the Viridian map, then heal_at_center (walks to the PC, heals, returns to the pre-heal spot)."""
        up = tv.map_id(self.b) in (PALLET, ROUTE1)     # Viridian is NORTH of us -> head north to heal
        edge = "north" if up else "south"
        log(f"   HEAL-RETURN: lead at {self.lead_hp()} - routing {edge} to Viridian to heal")
        saved = self._suppress_heal
        saved_runner = self.trav.battle_runner
        self._suppress_heal = True              # don't re-trigger heal WHILE returning to heal
        self.trav.battle_runner = self._flee_runner   # RETREAT: flee wild fights on the way out
        try:
            for leg in range(20):
                m = tv.map_id(self.b)
                if m == VIRIDIAN:
                    break
                out = self.trav.travel(target_map=VIRIDIAN, edge=edge, max_steps=800)
                if out == "arrived":
                    continue                       # crossed an edge toward Viridian
                if out == "battle_loss":
                    return "battle_loss"           # blacked out en route -> auto-heals at Viridian
                log(f"   HEAL-RETURN leg {leg}: no {edge} edge on map {m} - warping {edge}")
                if self.enter_warp(prefer=edge) != "warped":
                    log(f"   !! HEAL-RETURN stuck on map {m} (no {edge} edge, no {edge} warp)")
                    return "stuck"
            if tv.map_id(self.b) != VIRIDIAN:
                log(f"   !! HEAL-RETURN did not reach Viridian (at {tv.map_id(self.b)})")
                return "stuck"
            return self.heal_at_center()
        finally:
            self._suppress_heal = saved
            self.trav.battle_runner = saved_runner

    def _flee_runner(self):
        """Battle handler used DURING the heal-return: flee wild fights (retreat costs ~0 HP),
        win forced trainers. Built fresh per battle; routes events to the campaign's voice."""
        return BattleAgent(self.b, on_event=lambda s, **k: self.on_event(s),
                           render=self.render, log=lambda m: None).flee(max_seconds=90)

    def level_check(self, min_level, leader="Brock"):
        """LOUD Brock-readiness check at the Forest exit / before the gym. Reads the lead
        Pokemon's level; flags underleveled with a grind recommendation rather than
        silently walking into a gym loss. (Brock: Geodude L12 / Onix L14.)"""
        lvl = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        if lvl >= min_level:
            log(f"   LEVEL CHECK: lead Pokemon Lv{lvl} >= {min_level} - {leader}-ready")
            return "ok"
        # NON-FATAL: a fresh flee-to-survive run reaches Pewter ~Lv12 (one short of the conservative
        # Lv13 gate). Bulbasaur's Vine Whip is 2x on Brock's Rock/Ground (Boulder badge proven at this
        # level), and a gym LOSS is caught by the segment's blackout-recovery (respawn + retry) — never
        # a hard stop. So WARN + PROCEED rather than stalling the whole run on a soft readiness gate.
        # (grind() is a built capability for explicit/aggressive leveling, not auto-run here — a full
        # level headless is slow, and it must never block the 'walk away' spine.)
        log(f"   LEVEL CHECK: lead Lv{lvl} < {min_level} for {leader} — proceeding (Vine Whip 2x; a "
            f"loss is recoverable), not stalling on the soft gate")
        self.on_event(f"I'm only level {lvl}, but Vine Whip should still do the job on {leader}")
        return "ok"

    # ── move-learn handler (the empty-slot auto-learn fallback) ────────────────
    # WHY: a mid-battle level-up with 4 moves shows "Delete a move? / Stop learning?" Y/N boxes.
    # Those boxes are UN-ACTUATABLE in libmgba's continuous core (diagnosed 2026-06-24/25: the
    # key reaches KEYINPUT and a FRESH core confirms 6/6, but a long-running core can't land the
    # press at ANY timing/phase, and save+load doesn't reset the hidden core-instance state - only
    # a brand-new core does). So instead of fighting the box, we make it NEVER appear: reserve
    # open move slots so the level-up move AUTO-LEARNS. Bulbasaur learns TWO moves at L15
    # (PoisonPowder + Sleep Powder), so we reserve 2. Keeps her strongest moves (Tackle + Vine
    # Whip); the new moves fill the open slots with no prompt.
    def _attacks_block(self):
        base = ram.GPLAYER_PARTY
        key = self.b.rd32(base) ^ self.b.rd32(base + 4)        # PID ^ OTID
        a = st._SUBSTRUCT_ORDER[self.b.rd32(base) % 24].index("A")
        return base, key, base + 32, a                         # base, key, data_addr, A-substruct idx

    def _lead_moves(self):
        base, key, data, a = self._attacks_block()
        w0 = self.b.rd32(data + a * 12) ^ key
        w1 = self.b.rd32(data + a * 12 + 4) ^ key
        return [w0 & 0xFFFF, (w0 >> 16) & 0xFFFF, w1 & 0xFFFF, (w1 >> 16) & 0xFFFF]

    def _lead_pps(self):
        base, key, data, a = self._attacks_block()
        w = self.b.rd32(data + a * 12 + 8) ^ key               # the 4 PP bytes (one u32)
        return [w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF]

    def _lead_pp_low(self):
        """True when the lead is critically low on PP — only ≤1 real move can still fire. A Pokémon
        Center restores PP (not just HP), so this lets _available_actions offer heal even at FULL HP,
        breaking the depleted-PP spiral that froze her vs Mankey and made her ball-dump full-HP foes.
        Best-effort RAM read; any failure -> False (never a spurious heal)."""
        try:
            moves, pps = self._lead_moves(), self._lead_pps()
            usable = sum(1 for mid, p in zip(moves, pps) if mid and p > 0)
            return usable <= 1
        except Exception as e:
            log(f"   [roam] lead-PP read skipped: {e}")
            return False

    def _set_lead_moves(self, moves, pps):
        """Write the lead's 4 move IDs + 4 PP bytes wholesale, keeping the Gen-3 checksum valid
        (decrypt the 48-byte data block, overwrite the A-substruct's 3 words, recompute the u16-sum
        checksum, re-encrypt). Used to COMPACT moves (no empty gaps between real moves)."""
        base, key, data, a = self._attacks_block()
        words = [self.b.rd32(data + i * 4) ^ key for i in range(12)]
        words[a * 3] = (moves[0] & 0xFFFF) | ((moves[1] & 0xFFFF) << 16)
        words[a * 3 + 1] = (moves[2] & 0xFFFF) | ((moves[3] & 0xFFFF) << 16)
        words[a * 3 + 2] = ((pps[0] & 0xFF) | ((pps[1] & 0xFF) << 8)
                            | ((pps[2] & 0xFF) << 16) | ((pps[3] & 0xFF) << 24))
        chk = 0
        for w in words:
            chk = (chk + (w & 0xFFFF) + ((w >> 16) & 0xFFFF)) & 0xFFFF
        for i in range(12):
            self.b.core.memory.u32.raw_write(data + i * 4, words[i] ^ key)
        self.b.core.memory.u16.raw_write(base + 0x1C, chk)

    def _reserve_move_slots(self, n):
        """Free `n` move slots on the lead so upcoming level-up moves AUTO-LEARN with no (un-
        actuatable) Y/N box. We COMPACT, not delete-in-place: keep the (4-n) strongest moves in the
        TOP slots and push the empties to the END. WHY compact: the in-battle FIGHT menu displays
        real moves CONTIGUOUSLY, but BattleAgent navigates by RAM slot index - so a gap (e.g.
        [Tackle, _, _, VineWhip]) made it walk the cursor to grid-3 while Vine Whip showed at grid-1,
        and the super-effective move NEVER FIRED (it could only Tackle). Compacting makes RAM order
        == menu order, so the chosen move is actually navigable; trailing empties still feed the
        L15 auto-learn (new moves land in the lowest empty slot, i.e. the end). Keeps >= 2 moves."""
        moves, pps = self._lead_moves(), self._lead_pps()
        real = [(i, m) for i, m in enumerate(moves) if m]      # (slot, moveId) of real moves
        # MOVE-LEARN-AS-A-BEAT (Batch 2) is SCAFFOLDED in pokemon_soul.move_drop_decision but NOT wired
        # here yet: its first cut DROPPED the super-effective move (broke Brock), so per the safety
        # rail we keep the PROVEN keep-strongest-by-power reserve until the beat is debugged.
        keep_count = max(2, len(real) - n)                     # free up to n, never drop below 2
        ranked = sorted(real, key=lambda im: st.move_info(self.b, im[1])[1], reverse=True)
        keep = sorted(ranked[:keep_count], key=lambda im: im[0])   # strongest kept, in slot order
        new_moves = [m for _, m in keep] + [0] * (4 - len(keep))
        new_pps = [pps[i] for i, _ in keep] + [0] * (4 - len(keep))
        self._set_lead_moves(new_moves, new_pps)
        for _ in range(4):
            self.b.run_frame()

    def _ensure_move_room(self):
        """PHASE 3B — kill the free-roam move-learn FREEZE by PREVENTION (the 'Delete a move?' Y/N box
        is un-actuatable in the continuous core, so we make it never appear). Called before a free-roam
        leveling action: if the lead has 4 FULL moves, free ONE slot now so any move learned this battle
        AUTO-LEARNS with no box. WHICH move to drop is HER call — the soul oracle picks among the SAFE-
        to-drop set (never the single strongest, never a high-value status move while filler exists);
        headless / no engagement -> the proven keep-strongest default. Returns the dropped move name or
        None. A teammate that already has room is left untouched (no needless sacrifice)."""
        moves, pps = self._lead_moves(), self._lead_pps()
        real = [(i, m) for i, m in enumerate(moves) if m]
        if len(real) < 4:
            return None                                       # already has an open slot — nothing to do
        try:
            from pokemon_soul import HIGH_VALUE_LOW_POWER
        except Exception:
            HIGH_VALUE_LOW_POWER = set()
        # BATCH 6 ADDENDUM (MENU MASTERY): moves are PRECIOUS — she leveled for Vine Whip, one-shot Brock
        # with it, then MASHED it away. The box is un-actuatable in the continuous core, so the deliberate
        # choice happens HERE (proactively) instead of at the un-navigable menu. Build the full picture —
        # name, TYPE, power, and coverage flags — so she (and the safe default) protect the hard-hitter
        # AND type coverage. A super-effective / unique-type move is never tossed for raw power again.
        from collections import Counter
        info = [(i, m, st.MOVE_NAMES.get(m, f"move#{m}"), *st.move_info(self.b, m)) for i, m in real]
        # info rows: (slot, id, name, type, power)
        dmg_types = Counter(t for _i, _m, _n, t, p in info if p and p > 0)
        best_slot = max(info, key=lambda x: x[4] or 0)[0]     # the hard-hitter (highest power) — NEVER drop

        def _value(x):                                        # higher = more precious = keep
            _slot, mid, _n, t, power = x
            v = (power or 0)
            if mid in HIGH_VALUE_LOW_POWER:
                v += 100                                      # Sleep Powder / Leech Seed / etc.
            if (power or 0) > 0 and dmg_types.get(t, 0) == 1:
                v += 60                                       # UNIQUE-type coverage (Vine Whip!) — precious
            return v

        def _desc(x):
            _slot, mid, n, t, power = x
            tags = []
            if (power or 0) > 0 and dmg_types.get(t, 0) == 1:
                tags.append(f"your ONLY {t}-type move")
            if mid in HIGH_VALUE_LOW_POWER:
                tags.append("a key status move")
            if not power:
                tags.append("non-damaging")
            return f"{n} ({t}, {power or 0} pwr{'; ' + ', '.join(tags) if tags else ''})"

        # droppable = everything EXCEPT the hard-hitter; least-precious first (the coverage-safe default)
        droppable = sorted([x for x in info if x[0] != best_slot], key=_value)
        # TM/move-learn NO-GUT GUARANTEE (Batch 6 addendum): if NOTHING here is clearly junk (every move
        # is a hard-hitter / coverage / utility), the SAFE DEFAULT is KEEP THEM ALL — never gut a good set
        # to slot a new move. A quietly-gutted moveset doesn't fail until the E4 (Gen-3: no move relearner,
        # TMs are one-use), so integrity beats a possible box-freeze here. Junk = value below MOVE_JUNK_FLOOR
        # (a non-damaging non-utility filler, or a weak redundant attacker).
        least_value = _value(droppable[0]) if droppable else 999
        all_precious = least_value >= MOVE_JUNK_FLOOR
        options = {x[2]: f"let go of {_desc(x)}" for x in droppable}
        options["keep them all"] = ("keep your current four and skip the new move — if nothing's worth "
                                    "losing, you lose nothing")
        ctx = {"place": "your moveset is full and you're leveling, so to learn anything new you'd have to "
               "part with ONE move. Your current moves: " + "; ".join(_desc(x) for x in info) + ". Think "
               "like a real trainer: your hard-hitter and your type COVERAGE are PRECIOUS — a "
               "super-effective or only-of-its-type move must NEVER be tossed for raw power, and a good "
               "move can't be re-learned. If one move is clearly redundant or junk, drop THAT; if nothing "
               "is worth losing, KEEP THEM ALL. Which do you do?"}
        pick = self._soul_choose("move_drop", options, ctx)
        # KEEP path: she chose to, OR (headless/unparsable) everything is precious -> don't gut the set.
        if pick == "keep them all" or (not pick and all_precious):
            log(f"   [movelearn] KEEP-ALL (no-gut guarantee): nothing junk to drop (least_value={least_value}) "
                f"— holding the set; a new move is declined rather than gutting a good one")
            self.on_event("nah — all four of these are pulling weight. I'm keeping my set.",
                          kind="move", tier=1)
            return None
        drop_slot = None
        if pick and pick != "keep them all":
            for x in droppable:
                if x[2] == pick:
                    drop_slot = x[0]
                    break
        if drop_slot is None:                                 # headless with clear junk -> drop the junkiest
            drop_slot = droppable[0][0]
        dropped = st.MOVE_NAMES.get(moves[drop_slot], f"move#{moves[drop_slot]}")
        kept = [(i, m) for i, m in real if i != drop_slot]    # compact: kept in slot order, empty at end
        new_moves = [m for _, m in kept] + [0] * (4 - len(kept))
        new_pps = [pps[i] for i, _ in kept] + [0] * (4 - len(kept))
        self._set_lead_moves(new_moves, new_pps)
        for _ in range(4):
            self.b.run_frame()
        log(f"   [movelearn] PHASE 3B reserved a slot — dropped {dropped} (her pick; kept the rest) so a "
            f"new move auto-learns with no un-actuatable box")
        self.on_event(f"my moveset's full — I'll let {dropped} go to make room for something new",
                      kind="move", tier=2)
        return dropped

    def _drive_evolution(self, pre_species, pre_level):
        """PHASE 3A — drive the post-battle EVOLUTION cutscene so it NEVER freezes. After a battle the
        evolution scene plays on the overworld (a full-screen animation box_open can't see), and the old
        code's _wait_overworld returned 'idle' while input was still locked -> the next action wedged.

        GATE: evolution only happens on a LEVEL-UP, so if the lead didn't level this battle we return
        immediately with ZERO presses (no spurious steps on the common no-evolution case). On a level-up
        we settle (plain frames), check ONCE whether control is locked (a cutscene playing); if it's free
        there's nothing to drive (just a normal level-up) -> no-op. If LOCKED, we A-advance (B-free: A
        proceeds an evolution, B would CANCEL it — we never press B) until control returns / timeout.
        Default-proceed = evolving is almost always wanted (a 30s oracle call can't fit the few-second
        cancel window, so cancel isn't wired — proceed is the honest default). On a real species change
        we fire the NAMING soul beat (a new family member)."""
        cur_level = self.b.rd8(ram.GPLAYER_PARTY + self._PARTY_LEVEL_OFF)
        if not pre_level or cur_level <= pre_level:
            return                                            # no level-up -> no evolution possible
        from dialogue_drive import DialogueDriver
        dd = DialogueDriver(self.b, render=self.render, log=log)
        import time as _t
        t0 = _t.time()
        drove = False
        # settle (plain frames, no probing) so an evolution cutscene has a moment to START
        for _ in range(45):
            if st.in_battle(self.b):
                return
            self.b.run_frame(); self.render()
        # ONE control check: if control is already back, no cutscene is playing -> nothing to drive
        if dd._control_returned():
            return
        log("   [evolve] PHASE 3A: control LOCKED after a level-up — driving the cutscene (A-only)")
        # A-advance (B-free) until control returns or we time out (evolution anim can run ~20-30s)
        while _t.time() - t0 < 35:
            if st.in_battle(self.b):
                return
            if dd._control_returned():
                break
            drove = True
            self.b.press("A", 6, 10, self.render, owner="agent")
            self.render()
        post = st.read_party_species(self.b, 0)
        if pre_species and post and post != pre_species:
            before = st.SPECIES_NAME.get(pre_species, "my Pokemon")
            after = st.SPECIES_NAME.get(post, "something new")
            log(f"   [evolve] PHASE 3A drove the evolution cutscene: {before} -> {after} — naming beat")
            self.soul.note_evolve(before, after) if self.soul else None
            # NAME the new family member (continuity bond layer — NOT the un-navigable in-game keyboard).
            try:
                nm = self._soul_choose("name", {}, {"place": f"your {before} just evolved into a {after}! "
                                                    f"this is a new chapter for a teammate you love — give "
                                                    f"this {after} a name, just the name."})
                if nm and self.soul is not None:
                    self.soul.bonds[nm.lower()] = {"species": after, "nickname": nm,
                                                   "caught": "evolved", "note": "evolved"}
                    self.on_event(f"you're not just {after} to me — you're {nm} now. always.",
                                  kind="roster", tier=3)
            except Exception as _e:
                log(f"   [evolve] naming beat skipped: {_e}")
        elif drove:
            log("   [evolve] drove a post-battle cutscene (no species change — not an evolution)")

    # ── PHASE 4 traversal tools she can't have YET — honest stubs (never fake; log LOUD) ──────────────
    # RUNNING SHOES are LIVE (travel.py _press co-holds B -> ~1.85x outdoors, game-gated, verified). BIKE
    # and FLY are NOT owned this early (bike = the Cerulean voucher/Bike-Shop quest; Fly = HM02 from
    # Celadon + a compatible teammate + the badge to use it — both far ahead). There is also no overworld
    # key-item / HM / FLAG reader yet, so possession can't be confirmed. Rather than fake a capability she
    # doesn't have, these scream if ever called and return "not_owned" (they're NOT wired into the action
    # set, so they can't surface as a dead choice). Banked designs below for when she earns them.
    def use_bike(self):
        """STUB: the bicycle (Cerulean voucher -> Bike Shop). Not owned this early; no key-item reader
        exists to confirm. When built: register/use the Bike key item; ~2x bike speed, gated on terrain."""
        log("   [traversal] !! use_bike() called but the BIKE is not owned yet (no voucher/Bike-Shop "
            "done) and there's no key-item reader to confirm — DEFERRED, returning not_owned (LOUD).")
        return "not_owned"

    def fly_to(self, city):
        """STUB: Fly / fast-travel. Needs HM02 (Celadon), a teammate that can learn it, and the badge to
        use it — none yet. BANKED NAV DESIGN (so it's not lost): open the menu -> Pokémon that knows Fly
        -> Fly -> the Town Map opens; navigate the DESTINATION list with UP/DOWN + a SINGLE A to pick
        (NEVER an a_until_end_of_dialog mash — that overshoots the list). Only flyable to visited cities."""
        log(f"   [traversal] !! fly_to({city!r}) called but FLY is not owned yet (no HM02 / flyer / badge) "
            f"and there's no HM-flag reader to confirm — DEFERRED, returning not_owned (LOUD).")
        return "not_owned"

    def has_badge(self, flag):
        """Read any FLAG_BADGE0x_GET from the SaveBlock1 flag array (base + 0x0EE0)."""
        sb1 = self.b.rd32(0x03005008)                          # gSaveBlock1Ptr (DMA-shuffled target)
        fa = sb1 + 0x0EE0 + (flag >> 3)
        return bool(self.b.rd8(fa) & (1 << (flag & 7)))

    def has_boulder_badge(self):
        return self.has_badge(FLAG_BADGE_BOULDER)

    # ── BATCH-2 free-roam: LIVE STATE read (finding #1 — the oracle was state-BLIND, so she wanted
    # already-finished things, e.g. the Mt Moon fossil while standing in Cerulean). PURE RAM reads, no
    # bot. The free-roam loop injects this into the oracle ctx EVERY tick so her wants/picks are
    # grounded in WHERE SHE ACTUALLY IS. Names/gym-order are real game knowledge (verified ids only,
    # unknown map -> honest "an unfamiliar area"); progress flags are read from the save (badge array). ──
    _PLACE_NAMES = {
        (3, 0): "Pallet Town", (3, 1): "Viridian City", (3, 2): "Pewter City",
        (3, 3): "Cerulean City", (3, 19): "Route 1", (3, 20): "Route 2",
        (3, 21): "Route 3", (3, 22): "Route 4",
    }
    _BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
    _GYM_ORDER = [("Pewter City", "Brock"), ("Cerulean City", "Misty"),
                  ("Vermilion City", "Lt. Surge"), ("Celadon City", "Erika"),
                  ("Fuchsia City", "Koga"), ("Saffron City", "Sabrina"),
                  ("Cinnabar Island", "Blaine"), ("Viridian City", "Giovanni")]
    _PARTY_LEVEL_OFF = 0x54        # level byte within a 100-byte party-mon struct (== play_live LEAD_LEVEL)

    def read_live_state(self):
        """LIVE game-state snapshot for the soul oracle. PURE RAM reads (no bot). Returns a dict the
        free-roam loop feeds the oracle each tick so her reasoning is grounded in her real situation."""
        b = self.b
        mp = tv.map_id(b)
        co = tv.coords(b)
        badges = [nm for i, nm in enumerate(self._BADGE_NAMES) if self.has_badge(0x820 + i)]
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        party = []
        for s in range(min(cnt, 6)):
            sp = st.read_party_species(b, s)
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            lvl = b.rd8(base + self._PARTY_LEVEL_OFF)
            hp, mx = b.rd16(base + P_HP), b.rd16(base + P_MAXHP)   # PHASE 8 HUD: party-as-family HP bars
            party.append({"species": st.SPECIES_NAME.get(sp, f"species#{sp}"), "level": lvl,
                          "hp": hp, "maxhp": mx, "species_id": sp})
        try:                                      # current-map grass: does catch work HERE (vs needing to travel)?
            on_grass_map = bool(tv.Grid(b).grass)
        except Exception:
            on_grass_map = False
        ng = self._GYM_ORDER[len(badges)] if len(badges) < len(self._GYM_ORDER) else None
        place = self._PLACE_NAMES.get(mp, "an unfamiliar area")
        dex = ram.pokedex_owned_count(b)          # BATCH 6 PHASE 4: caught-count from RAM (no menu)
        arc = self._arc_note(len(badges))         # PHASE 4: where she is in the WHOLE journey (momentum)
        progress = (f"{len(badges)} badge(s) earned ({', '.join(badges) or 'none'}). "
                    + (f"Next gym: {ng[1]} of {ng[0]}." if ng else "All 8 badges earned.")
                    + (f" Pokédex: {dex} caught." if dex is not None else "")
                    + f" {arc}")
        return {"map": mp, "place": place, "coords": co,
                "badges": badges, "badge_count": len(badges),
                "party": party, "party_count": cnt,
                "on_grass_map": on_grass_map, "dex_caught": dex,
                "next_gym": ({"city": ng[0], "leader": ng[1]} if ng else None),
                "arc": arc, "progress": progress}

    # PHASE 4 — MILESTONE / ARC AWARENESS: she knows where she is in the OVERALL story (8 badges ->
    # Elite Four), so she can narrate progress with MOMENTUM ("third badge — a third of the way, credits
    # in sight") instead of treating each gym as an isolated errand. Pure framing off the badge count;
    # injected into the oracle ctx every tick. CORE-style narrative awareness, fed by the live HUD.
    def _arc_note(self, n):
        if n <= 0:
            return "The journey's just begun — no badges yet, the whole road to the Pokémon League ahead."
        if n >= 8:
            return "All 8 badges — Victory Road and the Elite Four are next. This is the endgame."
        frac = {1: "one badge in — the adventure's really starting",
                2: "two badges down, a quarter of the way",
                3: "three badges — over a third of the way there",
                4: "four badges — halfway to the League now",
                5: "five badges, well past halfway — the Elite Four's coming into view",
                6: "six badges down, only two to go",
                7: "seven badges — one gym from Victory Road, the League almost in reach"}[n]
        return f"She's {frac} (of 8 badges, then the Elite Four)."

    def pokedex_count(self):
        """BATCH 6 PHASE 4 — public accessor for 'how many have I caught?' (clean RAM popcount, no menu).
        Returns an int, or None pre-game. So she can answer on demand — incl. when chat asks (the mode
        exposes the read; the chat-Q&A wire to core is a thin pull, firewall-safe)."""
        return ram.pokedex_owned_count(self.b)

    # ── GENERAL gym-trainer gauntlet (a gym LEADER is gated behind the junior trainers) ──────────
    _OB, _SZ = 0x02036E38, 0x24
    _DELTA = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}     # facing nibble -> look direction
    _TOWARD = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}

    def _gym_trainers(self):
        """Every loaded TRAINER object event (trainerType 1/2): (object index, coord, facing). We
        track 'beaten' by INDEX, not coord - a gym trainer can WANDER (Cerulean's 2nd swimmer roams
        (4,7)->(7,7)), and a DEFEATED trainer still scans as trainerType, so coord-tracking would
        both re-engage a wanderer and lose track of who's done. The index is stable."""
        out = []
        for i in range(1, 16):
            o = self._OB + i * self._SZ
            if (self.b.rd8(o) & 1) and self.b.rd8(o + 0x07) in (1, 2):
                c = (self.b.rds16(o + 0x10) - 7, self.b.rds16(o + 0x12) - 7)
                out.append((i, c, self.b.rd8(o + 0x18) & 0x0F))
        return out

    def _engage_trainer(self, T, facing):
        """Walk to a tile adjacent to a junior trainer (its facing-front first, then any reachable
        side), face it, A -> battle. Its line of sight may auto-fire mid-approach (caught)."""
        order = [self._DELTA.get(facing, (0, 1)), (0, 1), (0, -1), (-1, 0), (1, 0)]
        seen = set()
        for adj in order:
            if adj in seen:
                continue
            seen.add(adj)
            front = (T[0] + adj[0], T[1] + adj[1])
            self.trav.travel(target_map=None, arrive_coord=front, max_steps=200, max_seconds=90)
            if st.in_battle(self.b):
                return True                                    # LoS fired during the walk-in
            if tv.coords(self.b) != front:
                continue                                       # couldn't reach this side - try next
            face = self._TOWARD[(-adj[0], -adj[1])]            # turn back toward the trainer body
            for _ in range(4):
                if st.in_battle(self.b):
                    return True
                self.b.press(face, 8, 8, self.render, owner="agent")
                self.b.press("A", 6, 12, self.render, owner="agent")
                for _ in range(20):
                    self.b.run_frame()
            if st.in_battle(self.b):
                return True
        return False

    def _talkable_npcs(self):
        """Active NON-trainer object events (trainerType==0): the plain townsfolk she can chat to —
        same object table as _gym_trainers but the talkable folk. (Signs read as trainerType 0 too; a
        press just reads them, harmless.) Returns [(index, coord, facing)]."""
        out = []
        for i in range(1, 16):
            o = self._OB + i * self._SZ
            if (self.b.rd8(o) & 1) and self.b.rd8(o + 0x07) == 0:
                c = (self.b.rds16(o + 0x10) - 7, self.b.rds16(o + 0x12) - 7)
                out.append((i, c, self.b.rd8(o + 0x18) & 0x0F))
        return out

    def _npc_is_resolved(self, body):
        """B-2 + LAYER A: has a dialogue loop already been disengaged in front of this NPC, OR has travel
        flagged its body tile as a plain blocker? Either way don't re-initiate — it's resolved/looping.
        Reads the UNIFIED block memory so the talk path and the travel path agree on one set of NPCs."""
        mp = tuple(tv.map_id(self.b))
        if (mp, tuple(body)) in self._blocked_npcs:        # travel marked this body tile a plain blocker
            return True
        return any((mp, (body[0] + ax, body[1] + ay)) in self._looped_spots
                   for ax, ay in ((0, 1), (0, -1), (-1, 0), (1, 0)))

    def _untalked_npc_here(self):
        """True iff there's at least one not-yet-talked-to townsperson on this map (cheap scan, no BFS) —
        the _available_actions gate so 'talk_npc' is only offered when it could DO something. Skips NPCs
        she's already resolved/looped on (B-2)."""
        talked = getattr(self, "_talked_npcs", {}).get(tv.map_id(self.b), set())
        return any(idx not in talked and not self._npc_is_resolved(c)
                   for idx, c, _f in self._talkable_npcs())

    _FACING_VAL = {"DOWN": 1, "UP": 2, "LEFT": 3, "RIGHT": 4}

    def _face_verified(self, key, tries=6):
        """READBACK-VERIFIED turn (2026-07-05, the Bill's-console fix): press the direction until the
        facing byte actually reads it. The FIRST turn press right after a travel leg is routinely EATEN
        (the walk animation still settling) — a blind face+A then interacts with the WRONG tile, which
        is exactly why the cottage PC read as 'un-interactable' and talk_npc silently 'chatted' with
        empty air. Same doctrine as the move-list / bag cursor readbacks: never trust a press, verify
        the effect."""
        from dialogue_drive import player_facing
        want = self._FACING_VAL[key]
        for _ in range(tries):
            if player_facing(self.b) == want:
                return True
            self.b.press(key, 8, 8, self.render, owner="agent")
            for _f in range(16):
                self.b.run_frame(); self.render()
        return player_facing(self.b) == want

    def talk_npc(self):
        """BATCH 5 PHASE 4 — her earliest want ('talk to the locals'). Pick a reachable townsperson she
        hasn't talked to here yet, walk to them, face + A, and let the READ+REACT seam fire (the live
        DialogueReader poll already voices what they say in HER words). Tracks talked NPCs per-map so she
        works the room instead of mashing one person. Returns 'talked' | 'no_npc'.

        2026-07-05 (the Bill last-mile): (a) WANDERER TRACKING — re-read the NPC's LIVE coords on arrival
        (a LOOK_AROUND/wander NPC moves mid-approach; the old code then A'd the tile they LEFT — the
        silent 'chatted with empty air' bug); (b) FACE-VERIFIED turn; (c) HONEST 'talked' — only counts
        if a dialogue box actually OPENED (box_open readback), so a missed interaction retries instead
        of poisoning the talked-set."""
        from dialogue_drive import box_open as _box
        mp = tv.map_id(self.b)
        if not hasattr(self, "_talked_npcs"):
            self._talked_npcs = {}
        talked = self._talked_npcs.setdefault(mp, set())
        for idx, body0, _facing in self._talkable_npcs():
            if idx in talked or self._npc_is_resolved(body0):   # B-2: skip talked + looped/resolved NPCs
                continue
            for _attempt in range(4):                           # wanderer tracking: re-approach live coords
                body = next((c for i, c, _f in self._talkable_npcs() if i == idx), None)
                if body is None:
                    break                                       # despawned (a scripted NPC can leave)
                grid = tv.Grid(self.b)
                cur = tv.coords(self.b)
                moved_on = False
                for adj in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                    front = (body[0] + adj[0], body[1] + adj[1])
                    if front != cur and not tv.bfs(grid, cur, lambda t, f=front: t == f,
                                                   walkable=grid.walkable):
                        continue
                    if front != cur:
                        log(f"   TALK: approaching NPC obj#{idx} at {body} via {front}")
                        self.trav.travel(target_map=None, arrive_coord=front, max_steps=200, max_seconds=60)
                    if st.in_battle(self.b):
                        return "talked"                        # an interaction started a battle — loop fights it
                    if tv.coords(self.b) != front:
                        break                                  # couldn't reach this side — re-read + retry
                    live = next((c for i, c, _f in self._talkable_npcs() if i == idx), None)
                    if live != body:
                        moved_on = True
                        break                                  # they wandered mid-walk — re-approach fresh
                    self._face_verified(self._TOWARD[(-adj[0], -adj[1])])
                    opened = False
                    for _a in range(3):                        # a post-walk A can be eaten too — bounded retry
                        self.b.press("A", 6, 12, self.render, owner="agent")
                        for _ in range(20):
                            self.b.run_frame(); self.render()
                        if _box(self.b):
                            opened = True
                            break
                    if not opened:
                        moved_on = True                        # no box = interaction missed -> retry fresh
                        break
                    self._drain_overworld(label="npc-talk")    # read + advance their line (her reaction fires)
                    talked.add(idx)
                    log(f"   TALK: chatted with NPC obj#{idx} (box verified)")
                    return "talked"
                if not moved_on:
                    break                                      # sides exhausted for a stationary NPC
        log("   TALK: no reachable un-talked NPC here")
        return "no_npc"

    def _drain_overworld(self, label="dlg"):
        """Drive any overworld dialogue box (a trainer's post-battle line, an NPC, the badge award)
        to a clean close at a watchable pace via the general primitive. No box -> returns at once.
        Returns the drive result. B-2: on 'exhausted' (the cycle-watchdog bailed on a LOOPING NPC),
        records this spot so she won't re-engage it (the full re-talking-beaten-NPC class, not just
        the literal Slowbro instance)."""
        self.b.set_input_owner("agent")
        res = DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(label=label)
        if res == "exhausted":
            try:
                mp = tuple(tv.map_id(self.b))
                cur = tuple(tv.coords(self.b))
                self._looped_spots.add((mp, cur))
                # LAYER A — feed the UNIFIED block memory too: the tile she's FACING is the looping NPC's
                # body, so travel (not just the talk path) routes AROUND it next time. One shared set.
                d = self._FACE_DELTA.get(wf._facing(self.b))
                if d is not None:
                    body = (cur[0] + d[0], cur[1] + d[1])
                    self._blocked_npcs.add((mp, body))
                log(f"   [roam] !! LOOPING/RESOLVED NPC disengaged at {(mp, cur)} — won't re-initiate "
                    f"talk there + travel routes around its body (B-2 + LAYER A unified guard)")
            except Exception:
                pass
        return res

    def _clear_gym_trainers(self, leader_front, max_rounds=10):
        """Beat EVERY junior trainer (re-scanning each round, since far ones are proximity-loaded
        and one may wander), THEN return so the caller engages the gated leader. General gym
        mechanic - reusable for Surge/Erika/etc."""
        beaten = set()                                         # object indices already cleared
        for rnd in range(max_rounds):
            self.b.set_input_owner("agent")
            trs = [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]
            if trs:
                px, py = tv.coords(self.b) or (0, 0)
                trs.sort(key=lambda t: abs(t[1][0] - px) + abs(t[1][1] - py))
                idx, T, facing = trs[0]
                log(f"   GYM: engaging junior trainer obj{idx} at {T} (facing {facing})")
                self._engage_trainer(T, facing)
                if st.in_battle(self.b):
                    log(f"   GYM: junior trainer -> {self.battle_runner()}")
                    self._drain_overworld(label="trainer")
                beaten.add(idx)                                # cleared (or unreachable) -> don't reloop
                continue
            # none loaded -> advance toward the leader to load/trigger far trainers (LoS may fire)
            self.trav.travel(target_map=None, arrive_coord=leader_front,
                             max_steps=200, max_seconds=90)
            if st.in_battle(self.b):
                log(f"   GYM: en-route LoS trainer -> {self.battle_runner()}")
                self._drain_overworld(label="trainer")
                nt = [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]
                if nt:
                    px, py = tv.coords(self.b) or (0, 0)
                    nt.sort(key=lambda t: abs(t[1][0] - px) + abs(t[1][1] - py))
                    beaten.add(nt[0][0])
                continue
            if [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]:
                continue                                       # a far trainer just loaded -> engage it
            log(f"   GYM: all junior trainers cleared (beaten obj {sorted(beaten)})")
            return
        log(f"   !! GYM: hit {max_rounds} clear-rounds with trainers still loaded - proceeding LOUD")

    def beat_gym(self, name):
        """GENERAL gym handler (gyms gate the leader behind their junior trainers): reserve move
        slots if this gym has a level-up double-learn, enter, BEAT EVERY JUNIOR TRAINER, then engage
        the gated leader, beat them with the current team, drive the badge/TM award to a clean close
        at a watchable pace, and confirm the badge flag. Data-driven via GYMS - one row per leader."""
        gym = GYMS.get(name)
        if gym is None:
            log(f"   !! GYM: no spec for '{name}'"); return "stuck"
        if gym.reserve:
            self._reserve_move_slots(gym.reserve)
            log(f"   GYM: reserved {gym.reserve} slot(s) for a level-up learn; moves now "
                f"{[st.MOVE_NAMES.get(m, '#' + str(m)) for m in self._lead_moves()]}")
        # BATCH 6 PHASE 2 — freeze-prevention for the CLIMB: a gym is a dense run of level-ups, and a
        # FULL moveset hitting a learn shows the un-actuatable "Delete a move?" box (a ~3-min battle hang
        # in the continuous core). gym.reserve only covers gyms with a KNOWN double-learn (Brock); this
        # makes it UNIVERSAL — if the lead is still at 4 moves, free ONE slot so any learn auto-fills
        # (her pick via the soul, never the best/super-effective move). No-op if she already has room.
        self._ensure_move_room()
        if tv.map_id(self.b) == gym.city:
            if self.enter_warp(pick=gym.door) != "warped":
                log(f"   !! GYM: couldn't enter the {name} gym"); return "stuck"
        for _ in range(45):
            self.b.run_frame()
        gym_map = tv.map_id(self.b)                             # the gym interior we just entered
        log(f"   GYM: inside {gym_map} at {tv.coords(self.b)} - clearing junior trainers")
        # 1) BEAT THE JUNIOR TRAINERS FIRST (the leader is gated until they're all down)
        self._clear_gym_trainers(gym.leader_front)
        # BLACKOUT during the juniors -> she whited out + respawned in the city PC (the map left the
        # gym interior). _clear_gym_trainers can't tell (no trainers load in the PC), so detect it here
        # and propagate -> the segment's blackout-recovery respawns + RE-RUNS the gym (beaten juniors
        # stay beaten in-game, so only the one that won re-fights; a fresh full-HP, on-level run wins).
        if tv.map_id(self.b) != gym_map:
            log(f"   GYM: no longer in the gym ({tv.map_id(self.b)} != {gym_map}) - blackout during "
                f"the juniors; caller recovers + retries the gym")
            return "battle_loss"
        # 2) engage the LEADER (now ungated): walk to the front tile, face them + A to initiate,
        # then DRIVE the pre-battle challenge speech INTO the battle (Brock's "So, you're here..."
        # is a multi-box speech that a few A-taps don't fully clear; the primitive advances it and
        # stops the instant the battle starts). Misty starts the battle near-immediately - same path.
        self.b.set_input_owner("agent")
        self.trav.travel(target_map=None, arrive_coord=gym.leader_front, max_steps=200, max_seconds=90)
        _lvl = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        _sp = st.SPECIES_NAME.get(st.read_party_species(self.b, 0), "?")
        log(f"   GYM: at {tv.coords(self.b)} (leader front {gym.leader_front}) - engaging {name} "
            f"[lead {_sp} Lv{_lvl}, party={self.b.rd8(ram.GPLAYER_PARTY_CNT)}]")
        for _ in range(6):
            if st.in_battle(self.b) or dd_box_open(self.b):
                break
            self.b.press(gym.leader_dir, 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        if not st.in_battle(self.b):                           # advance the challenge speech -> battle
            DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(
                stop_when=lambda: st.in_battle(self.b), label=f"{name}-challenge")
        if st.in_battle(self.b):
            res = self.battle_runner()
            log(f"   GYM: {name.upper()} -> {res}")
            # LOSING the leader is a SURVIVAL blackout (she whites out -> respawns at the city PC),
            # not a nav stuck. Propagate it so the segment's blackout-recovery respawns + retries the
            # whole gym (re-walk, junior trainers stay beaten, fresh full-HP leader attempt). Without
            # this, beat_gym fell through to the award drain on a loss -> 'stuck' inside the PC.
            if res == "loss":
                log(f"   GYM: lost to {name} (blackout) - caller recovers + retries the gym")
                return "battle_loss"
        # 3) AWARD (the climax): drive the badge + TM gift to a clean close at a WATCHABLE pace via
        # the general dialogue primitive. draining_award tells the soul-on render (play_live) to stop
        # polling/holding here so nothing competes with the drain. The proven brisk A-mash stays as a
        # BACKSTOP - the climax must NEVER freeze, so if the paced drive hasn't set the badge we fall
        # back to it immediately (preserves the regression-green reliability).
        self.draining_award = True
        try:
            self._drain_overworld(label=f"{name}-award")
            if not self.has_badge(gym.badge_flag):
                log("   !! GYM: paced award didn't set the badge - A-mash backstop")
                for k in range(160):
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
                    if self.has_badge(gym.badge_flag):
                        break
                for _ in range(30):                            # clear any TM gift -> overworld
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
        finally:
            self.draining_award = False
        if self.has_badge(gym.badge_flag):
            log(f"   GYM: *** {name.upper()} BADGE obtained ***")
            self.on_event(f"I beat {name} - that's the badge!")
            if tv.map_id(self.b)[0] != 3:                      # still in the gym interior -> leave to
                self._exit_to_overworld()                      # the city via the general hardened exit
                log(f"   GYM: exit -> now {tv.map_id(self.b)}@{tv.coords(self.b)}")
            return "badge"
        log(f"   !! GYM: {name} not beaten / no badge flag"); return "stuck"

    # ── new-game CHARACTER CREATION (the autonomous opening debut) ──────────────
    def create_character(self, player_name, rival_name, girl=True):
        """Drive FireRed's new-game intro AUTONOMOUSLY to the bedroom: Oak's welcome -> GENDER pick
        -> name HERSELF -> name the RIVAL. CAPABILITY-not-script: the names + gender are passed in
        (Kira's choices / a Batch-2 seam), NEVER hardcoded. Assumes the boot is already past the
        title into the intro (the opening driver does title->New Game). Returns 'bedroom'|'stuck'.
        Proven: KIRA/GARY/GIRL -> bedroom, committed playerName+gender+rival all correct."""
        from naming import name_entry
        names = [player_name, rival_name]
        ni = 0
        for _ in range(700):
            if tv.map_id(self.b) == (4, 1):                    # reached the bedroom
                return "bedroom"
            low = self._read_overworld_text().lower()
            if "boy" in low and "girl" in low:                 # GENDER select (default BOY at top)
                if girl:
                    self.b.press("DOWN", 8, 10, self.render, owner="agent")
                    for _ in range(10):
                        self.b.run_frame()
                self.b.press("A", 8, 10, self.render, owner="agent")
                for _ in range(24):
                    self.b.run_frame()
                continue
            if ("name" in low and "?" in low) and ni < 2:      # name prompt -> open kb + type
                for _ in range(6):
                    if not dd_box_open(self.b):
                        break
                    self.b.press("A", 4, 10, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
                for _ in range(40):
                    self.b.run_frame()
                if not dd_box_open(self.b):                     # keyboard is up
                    name_entry(self.b, names[ni], render=self.render)
                    log(f"   OPENING: named {'self' if ni == 0 else 'rival'} = {names[ni]!r}")
                    ni += 1
                    continue
            self.b.press("A", 4, 10, self.render, owner="agent")  # advance the intro dialogue
            for _ in range(12):
                self.b.run_frame()
        return "stuck"

    def _read_overworld_text(self):
        from dialogue_reader import DialogueReader
        try:
            return DialogueReader(self.b)._read_buffer()
        except Exception:
            return ""

    def _adv_dialogue(self, n=50):
        """Advance overworld dialogue (A) until the box closes or n presses."""
        for _ in range(n):
            if not dd_box_open(self.b):
                break
            self.b.press("A", 4, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()

    def _title_to_newgame(self):
        """From a FRESH ROM boot: mash A/START through the title + copyright screens into a New Game,
        stopping once Oak's intro dialogue is rolling (so drive_opening can take over)."""
        from dialogue_reader import DialogueReader
        dr = DialogueReader(self.b)
        for k in range(80):
            low = dr._read_buffer().lower()
            if "welcome" in low or "boy" in low:
                return
            self.b.press("A" if k % 2 else "START", 4, 8, self.render, owner="agent")
            for _ in range(10):
                self.b.run_frame()

    def drive_opening(self, player_name, rival_name, girl=True, starter_choice=0):
        """THE AUTONOMOUS OPENING (the personality debut), assuming the boot is already at Oak's
        intro (the caller does title->New Game). Character creation (her gender + names) -> bedroom
        -> downstairs (FireRed stairs fire on walking INTO them from the SIDE, not step-on) -> Mom
        -> Pallet -> walk north -> Oak intercepts to the lab -> PICK her starter -> the rival battle
        -> walk back out to Pallet. CAPABILITY-not-script: gender/names/starter are HER choices
        (params/seams), never hardcoded. Returns the final map id."""
        if self.create_character(player_name, rival_name, girl) != "bedroom":
            log("   !! OPENING: character creation didn't reach the bedroom"); return tv.map_id(self.b)
        self._adv_dialogue()                                   # bedroom "legend unfold" + NES cutscene
        log(f"   OPENING: in bedroom {tv.coords(self.b)} - heading downstairs")
        self.trav.travel(target_map=None, arrive_coord=(11, 2), max_steps=60, max_seconds=40)
        for _ in range(4):                                     # walk WEST into the stairs (side-trigger)
            self.b.press("LEFT", 8, 8, self.render, owner="agent")
            for _ in range(8):
                self.b.run_frame()
            if tv.map_id(self.b) != (4, 1):
                break
        self._adv_dialogue()                                   # Mom (1F)
        if not self._exit_to_overworld():                      # general hardened building-exit
            log("   !! OPENING: couldn't exit the house"); return tv.map_id(self.b)
        self._adv_dialogue()
        log(f"   OPENING: in Pallet {tv.coords(self.b)} - walking north (Oak will intercept)")
        for _ in range(160):                                   # north -> Oak intercept -> lab (4,3)
            if tv.map_id(self.b) == (4, 3):
                break
            if dd_box_open(self.b):
                self.b.press("A", 4, 10, self.render, owner="agent")
                for _ in range(12):
                    self.b.run_frame()
                continue
            out = self.advance_north(PALLET if tv.map_id(self.b) != PALLET else ROUTE1, max_legs=2)
            if out in ("stuck", "timeout"):
                self.b.press("UP", 8, 8, self.render, owner="agent")
                for _ in range(8):
                    self.b.run_frame()
        if tv.map_id(self.b) != (4, 3):
            log("   !! OPENING: Oak intercept didn't reach the lab"); return tv.map_id(self.b)
        log("   OPENING: in Oak's lab - picking her starter")
        self._starter_choice = starter_choice
        if self.pick_starter() != "picked":
            log("   !! OPENING: starter pick failed"); return tv.map_id(self.b)
        for _ in range(60):                                    # advance until the RIVAL battle starts
            if st.in_battle(self.b):
                break
            self.b.press("A", 6, 10, self.render, owner="agent")
            for _ in range(14):
                self.b.run_frame()
        if st.in_battle(self.b):
            _riv_lead = ""
            try:
                _rb = st.read_battle(self.b)
                _riv_lead = st.SPECIES_NAME.get(_rb["enemy"]["species"], "") if _rb else ""
            except Exception:
                pass
            _riv_out = self.battle_runner()
            log(f"   OPENING: RIVAL battle -> {_riv_out}")                 # win OR lose - game goes on
            # GARY NEMESIS ARC: the encounter is recorded by the CENTRAL hook (_observed_battle_runner
            # detects the rival by his starter-line ace at EVERY fight, opening included) — no separate
            # opening record here, so the grudge is wired uniformly across the whole run (B-4).
            self.b.set_input_owner("agent")
        # walk out to Pallet - ROBUST to win OR loss (the post-battle dialogue + door differ a little
        # by outcome): drain dialogue + retry the south exit until we're actually out of the lab.
        for _ in range(4):
            if tv.map_id(self.b) != (4, 3):
                break
            self._drain_overworld(label="post-rival")
            self.b.set_input_owner("agent")
            self._exit_to_overworld()
        log(f"   OPENING: *** DONE -> {tv.map_id(self.b)}@{tv.coords(self.b)} "
            f"party={self.b.rd8(ram.GPLAYER_PARTY_CNT)} ***")
        return tv.map_id(self.b)

    # ── starter pick (Oak's lab) - CAPABILITY not DECISION ──────────────────────
    # The three Poké Balls on Oak's table (recon_starter / starter.state): left->right.
    STARTER_BALLS = {0: (8, 4), 1: (9, 4), 2: (10, 4)}         # ball tiles (face UP from below)
    STARTER_SPECIES = {0: 1, 1: 4, 2: 7}                       # Bulbasaur / Charmander / Squirtle

    def choose_starter(self):
        """DECISION SEAM - which starter Kira takes. Capability-not-script: the HANDS (pick_starter)
        work for ANY of the three; WHICH she picks is HER choice, set on self._starter_choice (Batch
        2 routes this through her soul). NOT hardcoded into the pick mechanics. Default 0 (Bulbasaur)
        is only a fallback when nothing set the choice - the point is the seam exists + is honoured."""
        c = getattr(self, "_starter_choice", None)
        return c if c in (0, 1, 2) else 0

    def choose_nickname(self):
        """DECISION SEAM - does Kira nickname her new Pokemon, and what? Returns a name str to name
        it (via name_entry) or None to DECLINE. Capability-not-script: set self._nickname (Batch 2
        routes this through her soul). Default None = decline cleanly (NOT A-spam)."""
        nk = getattr(self, "_nickname", None)
        return nk if isinstance(nk, str) and nk.strip() else None

    def _handle_nickname(self):
        """The 'Give a nickname to <X>?' Yes/No prompt after acquiring a Pokemon. Her choice: name it
        (name_entry on the SAME keyboard as self/rival naming) or decline (No). Fixes the A-spam that
        nicknamed the starter 'AAAA'."""
        from naming import name_entry
        nick = self.choose_nickname()
        for _ in range(30):                                    # advance toward the nickname prompt
            low = self._read_overworld_text().lower()
            if "nickname" in low:
                if nick:
                    for _ in range(8):                                     # YES, then advance the
                        if not dd_box_open(self.b):                        # Yes/No -> the keyboard
                            break                                          # (opens after a few A's)
                        self.b.press("A", 8, 10, self.render, owner="agent")
                        for _ in range(16):
                            self.b.run_frame()
                    for _ in range(40):                                    # SETTLE: the fresh keyboard
                        self.b.run_frame()                                 # eats the first tap otherwise
                    name_entry(self.b, nick, render=self.render)           # type her chosen name
                    log(f"   STARTER: nicknamed it {nick!r} (her choice)")
                else:
                    self.b.press("B", 8, 10, self.render, owner="agent")   # NO -> decline (no A-spam)
                    for _ in range(20):
                        self.b.run_frame()
                    log("   STARTER: declined a nickname (her choice)")
                return
            self.b.press("A", 4, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()

    def pick_starter(self, idx=None):
        """HANDS: walk below the chosen ball, face it, take it + confirm YES, drive Oak's dialogue
        until the party grows, then handle the nickname prompt (her choice). Choice-agnostic (ball
        0/1/2). Returns 'picked' | 'stuck'. WORKS when the lab is reached CONTINUOUSLY (Oak's full
        intro grants pick-permission) - the dead starter.state mid-dialogue capture was the old wall."""
        if idx is None:
            idx = self.choose_starter()
        bx, by = self.STARTER_BALLS[idx]
        log(f"   STARTER: ball {idx} at {(bx, by)} (her choice; hands are choice-agnostic)")
        # 1) advance Oak's intro speech ("Those are POKé BALLS...") until control returns
        DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(label="oak-intro")
        # 2) walk below the chosen ball + open the "Do you want X?" prompt
        self._step_to((bx, by + 1))                            # stand directly below the ball
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for _ in range(6):
            self.b.press("UP", 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(18):
                self.b.run_frame()
            if dd_box_open(self.b) or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                break
        # 3) confirm YES (default cursor) + drive Oak's dialogue until the party grows
        for _ in range(40):
            if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:         # stop the instant we own it
                break
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
            sp = st.read_party_species(self.b, 0)
            log(f"   STARTER: *** picked {st.SPECIES_NAME.get(sp, '?')} (#{sp}) ***")
            self.on_event(f"I'll go with {st.SPECIES_NAME.get(sp, 'this one')}")
            self._handle_nickname()                            # the "give a nickname?" prompt - her choice
            return "picked"
        log("   !! STARTER: no Pokemon obtained - pick stuck (LOUD)")
        return "stuck"

    # ── Oak's Parcel quest (the story beat that yields the Pokedex + the 5 catch-enabling balls) ──
    def _ball_count(self):
        """Poke Balls (item id 4) in the bag's balls pocket; quantity XOR'd with the SaveBlock2 key
        (mirrors BattleAgent._ball_count). 0 -> nothing to throw (the catch gate before this quest)."""
        return self._balls_pocket_count(4)

    def _balls_pocket_count(self, item_id):
        """Count of a BALLS-pocket item (Gen-3 ids 1-12) in the bag's balls pocket (SaveBlock1+0x430,
        16 slots; qty XOR the low-16 key). The Items-pocket bag_count() can't see balls, so the buy
        verify + catch gate must read here for any ball id."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for i in range(16):
            iid = self.b.rd16(sb1 + 0x430 + i * 4)
            if iid == 0:
                break
            if iid == item_id:
                return self.b.rd16(sb1 + 0x430 + i * 4 + 2) ^ key
        return 0

    def _item_count(self, item_id):
        """Pocket-aware bag count: Poke Balls (ids 1-12) live in the balls pocket; everything else in
        the Items pocket. The buy-verify reads THIS so a Poke-Ball purchase (invisible to bag_count)
        verifies correctly instead of aborting LOUD on a false 'count didn't rise'."""
        if 1 <= item_id <= 12:
            return self._balls_pocket_count(item_id)
        return self.bag_count(item_id)

    def _ball_note(self, state=None):
        """Batch-WORLD Phase 5 — BALL PRE-CHECK awareness: if she has ZERO Poké Balls, catching is
        impossible no matter how much grass she reaches, so the oracle should KNOW a Mart run comes
        first. Empty unless she's out of balls. Live RAM read (verified on Jonny's watch)."""
        try:
            if self._ball_count() == 0:
                return ("Heads up: you have ZERO Poké Balls — you literally can't catch a teammate until "
                        "you buy some at a Mart, so a Mart run has to come before any catching.")
        except Exception as e:
            log(f"   [roam] ball-note read skipped: {e}")
        return ""

    def _talk(self, front, face_dir, label):
        """Interior NPC interaction: drain any cutscene already running, step to the front tile,
        face the NPC, A, then drive the dialogue/cutscene to a clean close (the giveitem_msg lines)."""
        self.b.set_input_owner("agent")
        if dd_box_open(self.b):
            self._drain_overworld(label=label + "-pre")
        self._step_to(front)
        for _ in range(6):
            if dd_box_open(self.b) or st.in_battle(self.b):
                break
            self.b.press(face_dir, 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 10, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        self._drain_overworld(label=label)

    def deliver_parcel(self):
        """Oak's Parcel STORY BEAT (disasm-seeded), converting the viridian_parcel_done GATE -> AUTO.
        From Viridian: enter the Mart -> the clerk gives OAK'S PARCEL; carry it south to Oak's lab in
        Pallet -> deliver -> the ReceiveDexScene grants the Pokedex (FLAG 0x829) + 5 Poke Balls
        (giveitem, NOT a shop buy). All dialogue + interior nav. Returns 'done'|'stuck'. The real
        proof is the ball count going 0 -> 5 (that's what unblocks the Route 3 catch)."""
        b0 = self._ball_count()
        if b0 >= 5 and self.has_badge(FLAG_POKEDEX_GET):       # idempotent: a retry after it's DONE
            log("   PARCEL: already delivered (Pokedex + balls present) - done"); return "done"
        log(f"   PARCEL: starting Oak's Parcel quest (balls={b0}, map={tv.map_id(self.b)})")
        # This is a SHORT, low-danger errand that passes through Viridian (which HAS a PC). Suppress
        # the heal-when-low BOUNCE for the whole errand and instead heal ONCE explicitly at the
        # Viridian PC up front — else every Route 1 wild dropping the lone starter below 75% would
        # yield need_heal and abort each travel leg (the south-to-Pallet stall).
        saved = self._suppress_heal
        saved_runner = self.trav.battle_runner
        self._suppress_heal = True
        # FLEE wild encounters for the whole fetch errand — Route 1/2 have no required trainers before
        # the parcel, and a just-picked Lv5 starter can't tank the round-trip's wilds (it blacked out
        # in Part C). Fleeing costs ~0 HP. A genuine flee-fail death PROPAGATES as battle_loss so the
        # segment's blackout-recovery retries it (instead of being swallowed into a non-recoverable 'stuck').
        self.trav.battle_runner = self._flee_runner
        try:
            for _ in range(5):                                 # 1) reach Viridian (the Mart is here)
                if tv.map_id(self.b) == VIRIDIAN:
                    break
                if self.advance_north(VIRIDIAN, max_legs=3) == "battle_loss":
                    return "battle_loss"
            if tv.map_id(self.b) != VIRIDIAN:
                log(f"   !! PARCEL: not at Viridian (at {tv.map_id(self.b)})"); return "stuck"
            if self.needs_heal():                              # 1b) top up so the round trip is safe
                self.heal_at_center(VIRIDIAN_PC_DOOR)
            if self.enter_warp(pick=VIRIDIAN_MART_DOOR) != "warped":   # 2) Mart -> clerk -> OAK'S PARCEL
                log("   !! PARCEL: couldn't enter the Viridian Mart"); return "stuck"
            for _ in range(60):
                self.b.run_frame()
            self._talk(MART_CLERK_FRONT, "UP", "parcel-pickup")
            log("   PARCEL: collected from the Mart counter; routing to Oak in Pallet")
            self._exit_to_overworld()                          # exit Mart -> Viridian (general exit)
            for _ in range(5):                                 # 3) south to Pallet
                if tv.map_id(self.b) == PALLET:
                    break
                r = self.trav.travel(target_map=PALLET, edge="south", max_steps=600, max_seconds=200)
                if r == "battle_loss":
                    return "battle_loss"
                if r != "arrived":
                    break
            if tv.map_id(self.b) != PALLET:
                log(f"   !! PARCEL: didn't reach Pallet (at {tv.map_id(self.b)})"); return "stuck"
            if self.enter_warp(pick=OAK_LAB_DOOR) != "warped":  # into Oak's lab
                log("   !! PARCEL: couldn't enter Oak's lab"); return "stuck"
            for _ in range(60):
                self.b.run_frame()
            self._talk(OAK_FRONT, "UP", "parcel-deliver")       # 4) deliver -> Pokedex + 5 balls
            for _ in range(2):                                  # the ReceiveDexScene is multi-box
                self._drain_overworld(label="dex-scene")
            b1 = self._ball_count()
            dex = self.has_badge(FLAG_POKEDEX_GET)
            log(f"   PARCEL: after delivery balls {b0}->{b1}  pokedex={dex}")
            self._exit_to_overworld()                           # 5) exit lab (general) -> north to Viridian
            for _ in range(5):
                if tv.map_id(self.b) == VIRIDIAN:
                    break
                if self.advance_north(VIRIDIAN, max_legs=3) == "arrived":
                    continue
                break
        finally:
            self._suppress_heal = saved
            self.trav.battle_runner = saved_runner
        if b1 > b0 and dex:
            self.on_event("delivered Oak's parcel — got the Pokedex and a handful of Poke Balls")
            return "done"
        log(f"   !! PARCEL: quest incomplete (balls {b0}->{b1}, pokedex={dex}) - LOUD"); return "stuck"

    def _catch_in_subcore(self, max_seconds=160):
        """Run the CATCH BATTLE in a FRESH sub-core, then load the result back into the main core.
        WHY: the in-bag ball THROW does not actuate in a long-running core — DIAGNOSED 2026-06-27: a
        FRESH core catches on the first ball, but after a long session the bag OPENS fine (verified via
        screenshot: cursor on POKé BALL x5) yet the select/throw never registers, so 100+ balls are
        'thrown' with the count never decrementing and the foe never weakening. This is the documented
        continuous-core menu-actuation wall (same class as the move-learn Y/N box) — only a NEW core
        resets it. Overworld nav + the FIGHT menu DO still actuate on the main core, so we hand off ONLY
        the catch battle: save the mid-battle state -> fresh Bridge (proven to actuate) -> catch_pokemon
        -> on success load the post-catch state back (the main core inherits the new teammate at the same
        tile). Two-core coexistence + state round-trip are control-proven (party 1->2 inherited). LOUD
        fallback to the main core on any sub-core error. Returns catch_pokemon's verdict.

        SHOW-VISIBILITY NOTE (Jonny's call): the throw happens on the HIDDEN sub-core, so a SHOW stream
        shows the main frame frozen then jump-cuts to 'caught' (she still NARRATES via on_event). To make
        the throw stream LIVE, point play_live's render at the sub-core during the handoff — a small
        follow-up I left for you because it changes the SHOW/watch experience."""
        try:
            mid = self.b.save_state()
            sub = Bridge(ROM)                                 # fresh core == fresh actuation
            sub.load_state(mid)
            for _ in range(8):
                sub.run_frame()
            sub.set_input_owner("agent")
            res = BattleAgent(sub, on_event=self.on_event, render=(lambda: None),
                              log=lambda m: log(m)).catch_pokemon(max_seconds=max_seconds)
            log(f"   CATCH[sub-core]: catch_pokemon -> {res}")
            if res == "caught":
                self.b.load_state(sub.save_state())           # main core inherits the new teammate
                for _ in range(8):
                    self.b.run_frame()
                self.b.set_input_owner("agent")
            return res
        except Exception as e:
            log(f"   !! CATCH[sub-core] FAILED ({e}) — falling back to main-core catch (LOUD)")
            res = BattleAgent(self.b, on_event=self.on_event, render=self.render,
                              log=lambda m: log(m)).catch_pokemon(max_seconds=max_seconds)
            self.b.set_input_owner("agent")
            return res

    def catch_one(self, max_seconds=300):
        """AUTO catch a teammate (converts the route3_catch hand-play GATE): WANDER in the grass
        until a WILD encounter, then catch it (BattleAgent.catch_pokemon - weaken/status then commit
        to throws). A trainer's line of sight en route is fought (trainer mons can't be caught), then
        we keep wandering. Returns 'caught' | 'no_balls' | 'timeout' | 'battle_loss'."""
        t0 = time.time()
        cur0 = tv.coords(self.b)
        # Grid.grass is in BUFFER coords (save + MAP_OFFSET); travel's arrive_coord + coords() are
        # SAVE coords. Convert here so the nearest-grass sort AND the walk targets share one space —
        # the old code fed buffer-coord grass straight to travel (a 7-tile miss that never stepped
        # onto grass -> no wild encounter -> the catch looked "impossible").
        off = tv.MAP_OFFSET
        g_now = tv.Grid(self.b)
        grass = sorted(((g[0] - off, g[1] - off) for g in g_now.grass),
                       key=lambda g: abs(g[0] - cur0[0]) + abs(g[1] - cur0[1]))
        if not grass:
            log("   !! CATCH: no grass on this map - can't find a wild Pokemon here")
            return "no_grass"
        # Warp/door tiles to keep OUT of: pathing across a door warps us into a building (the Route 3
        # PC / Mt Moon mouth) and the catch wedges on an NPC inside (the (5,4) trap). Treat doors as
        # permanent blocks for the whole wander — same avoid-param pattern as cave warp-to-warp nav.
        doors = frozenset(self._door_tiles())
        # REACHABILITY SANITY (increment 3.5): never hand travel a grass tile it can't path to. BFS
        # from here (permissive — allow NPC tiles, those clear on their own) and drop the unreachable.
        # If NONE is reachable (walled off, or a bad early-spawn target — the live (9,60)->(2,2) wedge),
        # return UP to roam so the oracle picks a different action, instead of travel spinning a dead
        # target for minutes (capability-not-script: she decides, we don't script the escape).
        def _reach(tile):
            return bool(tv.bfs(g_now, cur0, lambda t: t == tile,
                               walkable=lambda sx, sy: g_now.walkable(sx, sy) and (sx, sy) not in doors))
        reachable = [t for t in grass if _reach(t)]
        if not reachable:
            log(f"   !! CATCH: {len(grass)} grass tile(s) here but NONE reachable from {cur0} "
                f"(walled off / bad target) - returning to roam, the oracle decides (LOUD)")
            return "no_reachable_target"
        grass = reachable
        log(f"   CATCH: {len(grass)} reachable grass tile(s) (nearest {grass[0]}); avoiding "
            f"{len(doors)} warp tile(s) - traversing grass to catch one")
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        caught = [False]
        out_of_balls = [False]
        cant_weaken = [False]     # PHASE 4: depleted PP -> can't weaken a full-HP foe -> stop the wander
        #                           (don't burn balls); roam then heals (a Center restores PP) and retries
        healed_out = [False]      # BATCH 5 P2: healed mid-catch -> end cleanly, free_roam re-decides
        trainer_secs = [0.0]      # BATCH 5 P2: wall-clock spent in FORCED trainer fights en route — does
        #                           NOT count against the catch budget (those are mandatory, not failure)

        def catch_runner():
            """Travel hands every encounter here. Wild -> CATCH (commit); trainer -> normal fight.
            Always returns 'win' on a wild so travel keeps walking the grass for more encounters."""
            if self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08:           # trainer: can't catch -> fight
                # BATCH 5 P2: a Nugget-Bridge-style gauntlet en route to grass is MANDATORY, not wasted
                # time — punching through it IS progress. Time it and credit it back to the budget so the
                # catch isn't aborted before she ever reaches wild grass (the 164s/177s blow-out bug).
                log("   CATCH: a trainer engaged en route - fighting through (time NOT counted vs catch budget)")
                _tb = time.time()
                out = self.battle_runner()
                trainer_secs[0] += time.time() - _tb
                return out
            # BLOCK #3 — THE CHOICE (2026-07-06 nursery): a real player doesn't ball everything that
            # rustles the grass. Size the wild up (dupe/coverage/level/room), offer the call to the
            # oracle (hers, live), follow the framework's lean headless — and VOICE it both ways.
            if CATCH_JUDGMENT_ENABLED:
                try:
                    rb = st.read_battle(self.b)
                    foe_mon = rb["enemy"] if rb else None
                except Exception:
                    foe_mon = None
                if foe_mon and foe_mon.get("species"):
                    fid = foe_mon["species"]
                    fname = st.SPECIES_NAME.get(fid, f"pokémon #{fid}")
                    team = []
                    for i in range(min(self.b.rd8(ram.GPLAYER_PARTY_CNT), 6)):
                        sid = st.read_party_species(self.b, i)
                        team.append({"species_id": sid,
                                     "level": self.b.rd8(ram.GPLAYER_PARTY + i * st.PARTY_MON_SIZE + 0x54),
                                     "types": st.SPECIES_TYPES.get(sid, [])})
                    foe_desc = {"species_id": fid, "name": fname, "level": foe_mon.get("level"),
                                "types": st.SPECIES_TYPES.get(fid, foe_mon.get("types") or [])}
                    rec, reason, facts = roster_judgment(team, foe_desc)
                    pick = self._soul_choose(
                        "catch_judgment",
                        {"catch": f"throw a ball at this {fname} (L{foe_desc['level']})",
                         "skip": f"pass on the {fname} — fight or flee, keep hunting"},
                        {"place": f"a wild {fname} (L{foe_desc['level']}, "
                                  f"{'/'.join(t for t in foe_desc['types'] if t) or 'unknown type'}) "
                                  f"just appeared. sizing it up: {reason}. your call — is this one "
                                  f"joining the team?"})
                    decision = pick if pick in ("catch", "skip") else ("catch" if rec else "skip")
                    log(f"   CATCH-JUDGMENT: {fname} L{foe_desc['level']} -> {decision} "
                        f"(lean={'catch' if rec else 'skip'}, oracle={pick!r}) — {reason}")
                    if decision == "skip":
                        if not hasattr(self, "_catch_skip_voiced"):
                            self._catch_skip_voiced = set()
                        if fid not in self._catch_skip_voiced:
                            self._catch_skip_voiced.add(fid)
                            self.on_event(f"a {fname}… {reason}. not this one — moving on.",
                                          kind="roster", tier=1)
                        return self.battle_runner()          # fight/resolve it normally, keep hunting
                    self.on_event(f"oh — a {fname}! {reason}. I want this one.", kind="roster", tier=2)
            log("   CATCH: WILD encounter - committing to the catch")
            # CATCH ON THE MAIN CORE (default). The old "throw doesn't actuate in a long-running core" was
            # MISDIAGNOSED as decay; the real bug was the in-battle bag opening on the wrong (Items) pocket
            # so the throw selected CANCEL and never fired — root-caused + fixed 2026-06-27 (battle_agent
            # throw_ball pocket-steering + B-only catch resolution). Control: clean wild Route 3 catch,
            # party 1->2, no sub-core, no jump-cut. The fresh sub-core handoff is RETIRED to an opt-in
            # LOUD fallback (CATCH_SUBCORE=1) — off by default; it causes a frozen-frame jump-cut on stream.
            if os.environ.get("CATCH_SUBCORE", "0") == "1":
                log("   CATCH: CATCH_SUBCORE=1 -> using the legacy fresh sub-core fallback (jump-cut on stream)")
                res = self._catch_in_subcore(max_seconds=160)
            else:
                res = BattleAgent(self.b, on_event=self.on_event, render=self.render,
                                  log=lambda m: log(m)).catch_pokemon(max_seconds=160)
                self.b.set_input_owner("agent")
            log(f"   CATCH: catch_pokemon -> {res}")
            if res == "caught":
                caught[0] = True
                self.on_event("I caught a new teammate!")
            elif res == "no_balls":
                out_of_balls[0] = True              # STOP the wander — wandering ball-less just hits
            elif res == "cant_weaken":
                cant_weaken[0] = True               # depleted PP -> STOP; roam heals (restores PP) + retries
            return "win"                            # more wilds forever (the 8636-loop). a break_free is fine

        # Reuse the BFS Traveler (pathfinds around walls AND walks through grass, handing each
        # encounter to our catch_runner) - pace between near/far grass tiles to keep stepping in it.
        orig = self.trav.battle_runner
        self.trav.battle_runner = catch_runner
        try:
            waypoints = [grass[0], grass[-1], grass[len(grass) // 2]]
            wi = 0
            fails = 0
            FAIL_BUDGET = len(waypoints) + 2     # ~each waypoint once, then surface to roam
            # BUDGET (Batch 5 P2): measure only GENUINE catch-wandering time — credit back the forced
            # trainer-fight wall-clock so a trainer gauntlet between her and the grass can't time the
            # catch out before she gets a single throw.
            while (time.time() - t0 - trainer_secs[0]) < max_seconds and not caught[0] \
                    and not out_of_balls[0] and not cant_weaken[0]:
                before = tv.coords(self.b)
                r = self.trav.travel(target_map=None, arrive_coord=waypoints[wi % len(waypoints)],
                                     max_steps=120, max_seconds=80, avoid=doors)
                if r == "need_heal":
                    # BATCH 5 P2: heal at the NEAREST Center (location-aware) — NOT a hardcoded Pewter
                    # cross (the old Route-3-era line dead-ended post-Misty: "didn't reach (3,2)"). Then
                    # END this catch cleanly: heal_nearest may leave her at a Center on a DIFFERENT map,
                    # so chasing the old grass waypoints would be stale — instead let free_roam re-decide
                    # from her healed position (she re-picks wander_catch and re-acquires grass HERE, now
                    # healthy; no heal-loop since she's topped up). Can't route -> still break (LOUD).
                    hr = self.heal_nearest()
                    healed_out[0] = (hr != "stuck")
                    log("   CATCH: healed mid-catch — returning to roam to re-acquire grass healthy"
                        if healed_out[0] else "   !! CATCH: heal could not route from here - back to roam (LOUD)")
                    break
                wi += 1
                if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                    caught[0] = True; break
                # BOUND THE SPIN (increment 3.5): a travel that REACHED the grass ('arrived') or moved
                # us is progress (keep wandering for encounters). A travel that pathed nowhere is a dead
                # waypoint right now (NPC won't clear / unreachable). Tally those and, after FAIL_BUDGET,
                # surface to roam so the oracle changes action — NEVER re-cycle the dead set for minutes.
                progressed = (r == "arrived") or (tv.coords(self.b) != before)
                fails = 0 if progressed else fails + 1
                if not progressed:
                    log(f"   CATCH: travel -> {r} (reason={getattr(self.trav, 'last_fail_reason', '')!r}, "
                        f"no progress) fails={fails}/{FAIL_BUDGET}")
                if fails >= FAIL_BUDGET:
                    log(f"   !! CATCH: {fails} no-progress travels — grass not productively reachable, "
                        f"returning to roam so the oracle picks a different action (LOUD)")
                    return "no_reachable_target"
        finally:
            self.trav.battle_runner = orig
            self.b.set_input_owner("agent")
        if caught[0] or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
            return "caught"
        if out_of_balls[0]:
            log("   !! CATCH: ran out of Poké Balls before a catch - LOUD"); return "no_balls"
        if cant_weaken[0]:
            log("   !! CATCH: depleted PP — couldn't weaken a full-HP foe; stopped before burning balls. "
                "Roam will heal (restore PP) and retry (LOUD)")
            return "need_pp"           # not a failure: roam offers heal (Center restores PP), then retries
        if healed_out[0]:
            return "healed_retry"      # not a failure: she topped up mid-catch; free_roam re-decides
        log("   !! CATCH: no catch within budget (genuine wander time, trainer fights excluded) - LOUD")
        return "timeout"

    def _cave_runner(self):
        """Cave battle policy (CLARIFICATION 1): FLEE wild encounters — a solo Ivysaur can't grind
        every Zubat/Geodude through Mt Moon and survive — but FIGHT forced trainers (can't flee them).
        This is how the cave was cleared autonomously; the default fight-everything runner would
        attrition the lone starter to a blackout."""
        if self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08:         # trainer battle -> must fight
            return self.battle_runner()
        return self._flee_runner()                             # wild -> flee (costs ~0 HP)

    # Mt Moon fossil stand-tiles (face UP): touching a fossil triggers the Super Nerd (Miguel), who
    # battles you for it; beating him + the pick MOVES him, opening the gated B2F->B1F warp (5,10).
    MTMOON_FOSSILS = [((13, 8), "UP"), ((14, 8), "UP")]

    def clear_mt_moon(self):
        """Clear Mt Moon by FOLLOWING the saved warp route (states/mtmoon_plan.json) via the proven
        warp navigator (CaveNav._enter, the same primitive run_plan uses) — PLUS the one step the plan
        omits: the B2F FOSSIL/MIGUEL event, which moves Miguel to open the gated (5,10) exit (confirmed
        by nav-isolation: even win+heal can't enter (5,10) until the fossil event fires). Battle policy:
        FLEE wilds / FIGHT trainers; heal suppressed (no PC in a cave). A death -> 'blackout' (caller's
        blackout-recovery respawns + retries). Returns 'cleared' | 'blackout' | 'stuck'."""
        import json
        from cave_nav import CaveNav
        plan_path = resolve_state("mtmoon_plan.json") or os.path.join(STATES, "mtmoon_plan.json")
        if not os.path.exists(plan_path):
            log("   !! MTMOON: saved route mtmoon_plan.json not found - LOUD"); return "stuck"
        plan = json.load(open(plan_path))
        saved, saved_heal = self.trav.battle_runner, self._suppress_heal
        self.trav.battle_runner = self._cave_runner
        self._suppress_heal = True       # no PC in a cave -> survive-or-blackout, never excursion out
        try:
            nav = CaveNav(self.b, self, fog_path=None, on_event=self.on_event,
                          render=self.render, log=lambda m: log(m))
            # ENTRY NORMALIZATION: the real Route 4 warp lands on 1F at ~(18,37); the plan starts at
            # (15,30). Walk there so the first warp is reachable from EITHER entry. Navigation, not a skip.
            if tv.map_id(self.b) == (1, 1) and tv.coords(self.b) != (15, 30):
                self.trav.travel(target_map=None, arrive_coord=(15, 30), max_steps=400, max_seconds=150)
            for exp_map, wxy in plan:                          # follow the saved warp sequence
                cur = tv.map_id(self.b)
                if tuple(cur) != tuple(exp_map):
                    log(f"   !! MTMOON: PLAN MISMATCH on {cur}, expected {exp_map}"); return "stuck"
                if tuple(cur) == (1, 3):                        # B2F: clear Miguel BEFORE the gated (5,10)
                    if self._mtmoon_fossil_event() == "blackout":
                        log("   MTMOON: blacked out at the fossil/Miguel (SURVIVAL)"); return "blackout"
                    if tv.map_id(self.b)[0] != 1:               # left the cave (PC respawn) = a blackout
                        log("   MTMOON: left B2F during the Miguel fight (blackout) - caller retries")
                        return "blackout"
                r = nav._enter(tuple(wxy))
                # NUDGE-RETRY: the exit warp (45,4) is finicky to approach from (39,4) — the proven
                # port nudged one tile toward it (44,4) and retried up to 4x. run_plan does a single
                # _enter, which dropped that; re-add it so the final exit isn't a one-shot stall.
                tries = 0
                while r is None and tries < 4 and tuple(tv.map_id(self.b)) == tuple(exp_map):
                    tries += 1
                    self.trav.travel(target_map=None, arrive_coord=(wxy[0] - 1, wxy[1]),
                                     max_steps=200, max_seconds=90)
                    r = nav._enter(tuple(wxy))
                if r == "BLACKOUT":
                    log("   MTMOON: blacked out mid-cave (SURVIVAL) - caller retries"); return "blackout"
                if r is None:
                    log(f"   !! MTMOON: warp {tuple(wxy)} unenterable after {tries} nudge(s) "
                        f"(map={tv.map_id(self.b)})"); return "stuck"
            if not nav._fwd_reachable("east"):                 # last warp done -> cross out to Cerulean
                log(f"   !! MTMOON: warps done at {tv.map_id(self.b)} but east edge unreachable"); return "stuck"
            self.trav.travel(target_map=(99, 99), edge="east", max_steps=400, max_seconds=200)
            m = tv.map_id(self.b)
            if m[0] != 1 and m != (3, 22):
                log(f"   MTMOON: *** CLEARED via the saved route -> emerged on {m} ***")
                self.on_event("made it through Mt Moon")
                return "cleared"
            log(f"   !! MTMOON: not fully out (map={m})"); return "stuck"
        finally:
            self.trav.battle_runner, self._suppress_heal = saved, saved_heal

    def _mtmoon_fossil_event(self):
        """Touch a B2F fossil -> the Super Nerd (Miguel) battles you for it; win + the pick MOVES him,
        opening the gated (5,10) exit warp. _cave_runner FIGHTS him (trainer). Returns 'ok'|'blackout'."""
        for stand, facing in self.MTMOON_FOSSILS:
            self.trav.travel(target_map=None, arrive_coord=stand, max_steps=600, max_seconds=300)
            if tv.coords(self.b) == stand:
                log(f"   MTMOON: at fossil stand {stand} - triggering the Miguel event")
                self.b.press(facing, 8, 8, self.render, owner="agent")   # FACE the fossil ONCE
                for _ in range(16):
                    # A-ONLY from here — never re-press the d-pad, or grabbing the fossil lets the
                    # next UP walk off the stand-tile into the warp above it (the map=(16,0) bug).
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(8):
                        self.b.run_frame(); self.render()
                    if st.in_battle(self.b):                              # Miguel challenges for it
                        r = self._cave_runner()
                        if r == "loss" or tv.map_id(self.b) != (1, 3):    # lost -> whited out off B2F
                            return "blackout"
                self._drain_overworld(label="fossil-pick")
                return "ok"
        log("   !! MTMOON: couldn't reach a fossil stand-tile (Miguel uncleared) - (5,10) will gate")
        return "ok"

    # ── healing (the ordinary survival loop) ──────────────────────────────────
    def lead_hp(self):
        return (self.b.rd16(ram.GPLAYER_PARTY + P_HP),
                self.b.rd16(ram.GPLAYER_PARTY + P_MAXHP))

    def _step_to(self, tile, steps=30):
        """Obstacle-aware greedy stepper for tiny INTERIORS (the Traveler refuses an NPC's
        facing tile - e.g. a nurse/counter - so interiors need a manual stepper). Tries the
        dominant axis; if a press doesn't move us, tries the other; nudges sideways if boxed."""
        for _ in range(steps):
            cur = tv.coords(self.b)
            if cur == tile:
                return True
            if cur is None:
                for _ in range(8):
                    self.b.run_frame()
                continue
            dx, dy = tile[0] - cur[0], tile[1] - cur[1]
            if abs(dx) >= abs(dy):
                order = ([("RIGHT" if dx > 0 else "LEFT")] if dx else []) + \
                        ([("DOWN" if dy > 0 else "UP")] if dy else [])
            else:
                order = ([("DOWN" if dy > 0 else "UP")] if dy else []) + \
                        ([("RIGHT" if dx > 0 else "LEFT")] if dx else [])
            moved = False
            for key in order:
                self.b.press(key, 8, 8, self.render, owner="agent")
                if tv.coords(self.b) != cur:
                    moved = True
                    break
            if not moved:
                self.b.press("RIGHT", 8, 8, self.render, owner="agent")
        return tv.coords(self.b) == tile

    def heal_at_center(self, pc_door=VIRIDIAN_PC_DOOR):
        """AUTHENTIC Pokemon Center heal (no RAM poking): route to a Pokemon Center door (default
        Viridian; pass any city's PC door for NEAREST-center healing), enter, walk to the nurse
        counter, drive the YES heal dialogue, verify HP -> max, exit back to the city. PC interiors
        share ONE layout so only the overworld door differs. Returns 'healed' or 'stuck' (LOUD)."""
        h0 = self.lead_hp()
        city = tv.map_id(self.b)                  # the city we heal in (generic; was Viridian-only)
        # Heal if ANY party member needs it, not just the lead: a fainted BENCH mon (e.g. a weak mon KO'd
        # during a solo-grind) left the old lead-only guard skipping the heal -> the fainted mon never
        # revived -> the survival floor offered only 'heal' forever (infinite heal-loop, the look-ahead
        # stall). The nurse revives + tops the WHOLE party, so gate on whole-party health.
        if self._party_fully_healed():
            log(f"   HEAL: whole party already full - skipping"); return "healed"
        return_to = tv.coords(self.b)             # heal is TRANSPARENT: come back here after
        log(f"   HEAL: lead at {h0[0]}/{h0[1]} -> routing to the Pokemon Center on {city} "
            f"via door {pc_door} (will return to {return_to})")
        # 1) to the PC door + step in
        if self.trav.travel(target_map=None, arrive_coord=(pc_door[0],
                            pc_door[1] + 1), max_steps=400, max_seconds=120) != "arrived":
            # 2026-07-06 STRAND GUARD (nursery run-4): from a one-way pocket (Cerulean's post-ticket
            # south strip) the own-map Center door is UNREACHABLE and re-picking heal ping-pongs two
            # tiles forever (position changes each tick, so the no-move pruning never fires). Remember
            # the dead heal AT THIS SPOT so _available_actions stops offering it while she's non-
            # critical here — the road forward (the next town's Center) is the real heal.
            if not hasattr(self, "_heal_dead_maps"):
                self._heal_dead_maps = set()
            self._heal_dead_maps.add(tuple(tv.map_id(self.b)))
            log(f"   !! HEAL: couldn't reach the PC door (at {tv.coords(self.b)}) — remembered as "
                f"heal-dead on this map (offer suppressed while non-critical; push to the next Center)")
            self.on_event("can't get back to the Pokémon Center from this side — I'll patch up at the "
                          "next town ahead.", kind="recover", tier=1)
            return "no_reachable_center"
        before = tv.map_id(self.b)
        for _ in range(6):
            self.b.press("UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
            if tv.map_id(self.b) != before:
                break
        if tv.map_id(self.b) == city:
            log(f"   !! HEAL: did not enter the Center (still on {tv.map_id(self.b)})"); return "stuck"
        for _ in range(90):                       # entrance auto-walk settle (input lock)
            self.b.run_frame()
        self.b.set_input_owner("agent")
        # 2) to the nurse counter + drive the YES heal dialogue
        self._step_to(NURSE_FRONT)
        self.b.press("UP", 6, 8, self.render, owner="agent")     # face the nurse
        for _ in range(26):
            # YES/NO box eats the first press; UP engages (YES is top, can't move off), A=YES
            self.b.press("UP", 4, 4, self.render, owner="agent")
            self.b.press("A", 6, 10, self.render, owner="agent")
            for _ in range(30):
                self.b.run_frame()
            if self._party_fully_healed():            # WHOLE party revived+topped (not just the lead — a
                break                                 #   full lead + fainted bench used to break instantly)
        if not self._party_fully_healed():
            h1 = self.lead_hp()
            log(f"   !! HEAL: nurse dialogue did not complete (party not full; lead {h1[0]}/{h1[1]})")
            return "stuck"
        log(f"   HEAL: restored party to full (lead was {h0[0]}/{h0[1]})")
        # a successful heal on this map proves the Center reachable from HERE — clear any strand-guard
        # memo (the mark was for a one-way pocket; from the normal side the offer must return)
        getattr(self, "_heal_dead_maps", set()).discard(tuple(tv.map_id(self.b)))
        # 3) EXIT: the post-heal script holds control for a beat and the exit mat only fires when you
        # STEP onto it. PATIENTLY clear the nurse's closing text (B, harmless in the overworld) and
        # walk DOWN the counter column onto the mat, retrying until we're back in the city. The old
        # single DOWN-burst was too impatient and froze at the counter (the Cerulean heal-exit bug).
        for _ in range(6):                          # clear the nurse's closing text (B = harmless)
            self.b.press("B", 3, 8, self.render, owner="agent")
        self._exit_to_overworld()                   # the general, stress-tested building-exit (south
        #                                             door / DOWN-mat fallback). The old DOWN-only loop
        #                                             wedged inside some PCs ('stuck inside' at (5,4)).
        log(f"   HEAL: exited Center -> map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
        if tv.map_id(self.b)[0] != 3:               # still not on the overworld -> genuinely stuck
            return "healed_stuck_inside"
        # walk back to the pre-heal spot so HEAL is transparent (the next objective resumes
        # from where it expected to be, not the PC doorstep)
        if return_to and tv.coords(self.b) != return_to:
            self.b.set_input_owner("agent")
            r = self.trav.travel(target_map=None, arrive_coord=return_to,
                                 max_steps=400, max_seconds=120)
            log(f"   HEAL: returned toward {return_to} -> {r} (now {tv.coords(self.b)})")
        return "healed"

    def grind(self, target_level, fragile=False):
        """Train the lead to target_level in the grass, healing when low. Self-sufficient gym-readiness
        capability (the 'walk away' vision needs autonomous leveling). From a CITY with no grass
        (Pewter), cross EAST to the adjacent route (Route 3: grass + a trainer gauntlet = fast XP),
        grind, then cross back. Wilds are FOUGHT (XP), not fled. Returns 'ok' | 'battle_loss'.

        `fragile=True` (weak bench-mon grind): the lead can FAINT, so ONLY pace grass she can WALK BACK
        FROM to the safe start anchor — never cross a one-way ledge into a pocket where a faint would
        strand her un-healably (the Route-4-east heal-wedge). The tanky ace (fragile=False) is unfiltered
        (it never faints there, so Center-unreachable grass is fine — filtering it regressed the ace-grind)."""
        def lvl():
            return self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        if lvl() >= target_level:
            log(f"   GRIND: already Lv{lvl()} >= {target_level}"); return "ok"
        off = tv.MAP_OFFSET
        home = tv.map_id(self.b)

        def grass_save():
            return [(x - off, y - off) for (x, y) in tv.Grid(self.b).grass]
        log(f"   GRIND: Lv{lvl()} < {target_level} - heading to the grass to train"
            + (" [FRAGILE: reachable grass only]" if fragile else ""))
        if not grass_save() and home == PEWTER:               # city has no grass -> Route 3 (east)
            self.walk_to_map(ROUTE3, "east")
        anchor = tv.coords(self.b) or (0, 0)                   # safe start (Center-reachable); fragile grind
        #                                                        must be able to walk BACK here from any grass
        t0 = time.time()
        while lvl() < target_level and time.time() - t0 < 480:
            gs = grass_save()
            if not gs:
                log(f"   !! GRIND: no grass reachable on {tv.map_id(self.b)} - stopping LOUD"); break
            cur = tv.coords(self.b) or (0, 0)
            if fragile:                                        # keep only grass she can RETURN from (no one-way strand)
                grid_now = tv.Grid(self.b)
                safe = [g for g in gs if g != anchor
                        and tv.bfs(grid_now, g, lambda t, a=anchor: t == a, walkable=grid_now.walkable)]
                if safe:
                    gs = safe
                elif [g for g in gs if g != cur]:              # ALL grass is a one-way strand-risk here ->
                    log(f"   !! GRIND[fragile]: all grass on {tv.map_id(self.b)} is a one-way strand from "
                        f"{anchor} - stopping this map's fragile grind LOUD"); break
            # Pace the NEAREST grass tiles only (not the farthest) so a long grind stays LOCAL and never
            # drifts across a one-way ledge into a pocket where the grass behind becomes unreachable.
            gs.sort(key=lambda g: abs(g[0] - cur[0]) + abs(g[1] - cur[1]))
            nearby = [g for g in gs if g != cur][:4] or gs[:1]
            doors = frozenset(self._door_tiles())
            for wp in nearby:
                if lvl() >= target_level:
                    break
                r = self.trav.travel(target_map=None, arrive_coord=wp,
                                     max_steps=60, max_seconds=80, avoid=doors)
                if r == "battle_loss":
                    return "battle_loss"
                if r == "need_heal":
                    self.heal_nearest()
        log(f"   GRIND: trained to Lv{lvl()} (target {target_level})")
        if tv.map_id(self.b) != home and home == PEWTER:      # back to Pewter for the gym
            self.walk_to_map(PEWTER, "west")
        return "ok"

    # ── STRATEGIC UNDERLEVEL-GRIND (Task B): the smart middle between "ace farms grass, nothing else
    # levels" (aimless) and "charge the wall, lose, charge again" (stubborn). A real player who hits an
    # under-levelled wall benches the strong mon and FIELDS the weak ones against wild Pokémon to raise
    # the team FLOOR, then crosses. We field the weak member by REORDERING it to slot 0 (XP goes to who's
    # sent out) — exactly what the in-menu "switch order" does, so it's save-safe (each 100-byte mon
    # struct is self-contained + checksummed over its OWN data; an intact move between slots changes no
    # checksum). Survival = the existing heal-when-low floor in grind(). In-battle "ace as safety-switch"
    # is a gated enhancement (POKEMON_BATTLE_SWITCH, unverified) — noted, not v1. ──────────────────────
    def _party_levels(self):
        """Live level byte (+0x54) of every party member, slot order. Pure read."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        return [self.b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + self._PARTY_LEVEL_OFF)
                for s in range(min(cnt, 6))]

    def _party_fully_healed(self):
        """True iff EVERY party member is at full HP (none hurt, none fainted). The whole-party heal
        gate — a fainted bench mon must still trigger a Center visit even when the lead is full."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            if self.b.rd16(base + P_HP) < self.b.rd16(base + P_MAXHP):
                return False
        return True

    def _swap_party_slots(self, i, j):
        """Swap two party slots' raw 100-byte structs in gPlayerParty (the same intact move the in-menu
        'switch order' does — save-safe, see the block comment). Overworld-only (NEVER mid-battle, where
        the active-mon pointer would dangle). No-op if i==j or either slot is out of the live count."""
        if i == j:
            return
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if not (0 <= i < cnt and 0 <= j < cnt):
            log(f"   GRIND-WEAK: !! refusing slot swap {i}<->{j} (party count {cnt}) — no-op")
            return
        base = ram.GPLAYER_PARTY
        ai, aj = base + i * st.PARTY_MON_SIZE, base + j * st.PARTY_MON_SIZE
        for k in range(st.PARTY_MON_SIZE // 4):           # 100 B = 25 u32 words
            wi = self.b.rd32(ai + k * 4)
            wj = self.b.rd32(aj + k * 4)
            self.b.core.memory.u32.raw_write(ai + k * 4, wj)
            self.b.core.memory.u32.raw_write(aj + k * 4, wi)

    def _restore_ace(self):
        """Put the highest-level member back in slot 0 (the ace leads for the real wall + everyday play).
        Order-independent (re-reads levels), so it cleans up after any number of weak-grind swaps."""
        levels = self._party_levels()
        if not levels:
            return
        ace = max(range(len(levels)), key=lambda s: levels[s])
        if ace != 0:
            log(f"   GRIND-WEAK: restoring the ace (slot {ace}, L{levels[ace]}) to lead")
            self._swap_party_slots(0, ace)

    def _prep_team_target(self, state):
        """The readiness target IF she should be prepping for the active wall, else None. Single source of
        truth (action executor + framing + dashboard rationale read THIS). Two modes:
          • SWITCH ARMED (`POKEMON_GRIND_SWITCH=1`): bench-leveling works via participation XP, so the
            target is the FLOOR (weakest member) vs the foe — field the weak ones to readiness.
          • SWITCH GATED (default): the bench can't be safely levelled (the in-battle switch wedges), so
            the reliable play is ACE-OVERPOWER — level the ACE (strong lead) until it OUTLEVELS the wall;
            target = foe + OVERPOWER_MARGIN, fire while the ACE is below it.
        Fires only with STRATEGIC_GRIND on + a real active wall with a known foe level (a recognised level
        problem, not a nav-bug or pure type loss)."""
        if not STRATEGIC_GRIND_ENABLED:
            return None
        try:
            party = state.get("party") or []
            if not party:
                return None
            if not battle_agent.GRIND_SWITCH_ENABLED:
                t = self.strat.overpower_target()           # ACE-OVERPOWER (switch gated)
                if not t:
                    return None
                ace = max(m["level"] for m in party)
                return t if ace < t else None
            t = self.strat.underlevel_target()              # FIELD-WEAK (switch armed)
            floor = min(m["level"] for m in party)
            if not t and PROACTIVE_BENCH and len(party) >= 2:
                # WALL-LESS bench-raising (2026-07-06 nursery): a fresh catch shouldn't ride the bench
                # at L5 while the ace is L30 — when the floor sags PROACTIVE_BENCH_GAP under the lead,
                # prep toward lead-8 (modest, bounded; a recorded wall's target still dominates above).
                lead = max(m["level"] for m in party)
                if floor < lead - PROACTIVE_BENCH_GAP:
                    t = lead - 8
            if not t:
                return None
            return t if floor < t else None
        except Exception as e:
            log(f"   [roam] prep-team target skipped: {e}")
            return None

    def _prep_team_weak(self, state, target):
        """The under-target members' species names (for the rationale/framing). Pure read off state."""
        return [m["species"] for m in (state.get("party") or []) if m["level"] < target]

    def grind_weak_members(self, target):
        """Field the WEAK members (not the ace) and level the team FLOOR to `target`, then restore the
        ace. Each loop: pick the weakest under-target member, reorder it to lead, grind() it toward
        `target` (which heals it when low — survival), repeat until the floor crosses. Bounded by a
        wall-clock budget so a tick can't run away. Returns 'ready' (floor crossed) | 'battle_loss' |
        'ok' (budget/grass-out — partial progress, the next tick re-enters). Restores the ace on EVERY
        exit path (a faint/loss must not strand the weak mon as lead)."""
        # ── ACE-OVERPOWER (switch gated — the reliable path): the participation-XP switch wedges on the
        # long core, so fielding the weak bench just gets them one-shot (no XP). Instead level the ACE
        # (strong lead, solo-grinds reliably) to `target` (= foe + OVERPOWER_MARGIN) so it bulldozes the
        # wall — bigger bulk to tank the super-effective hit + faster KOs (no fight-length time-out). The
        # ace already leads in normal play; ensure it, then grind IT. grind() loops to target + heals. ──
        # Path choice: (1) in-battle participation-switch armed (GRIND_SWITCH) — weak lead switches to ace
        # turn 1; (2) SOLO weak-grind — weak lead fights solo (Super-Potion heal + ace backstop); (3)
        # ace-overpower fallback. (1)/(2) BOTH field the weak member (the real team-building); (3) only
        # levels the ace (can't fix a type-resisted wall). Prefer (1) if armed, else (2) if enabled.
        use_switch = battle_agent.GRIND_SWITCH_ENABLED
        if not use_switch and not SOLO_WEAK_GRIND:
            self._restore_ace()                              # ace in slot 0
            log(f"   GRIND-OVERPOWER: bench-leveling needs the (gated) in-battle switch — instead leveling "
                f"the ACE to L{target} to overpower the wall; levels now {self._party_levels()}")
            return self.grind(target)
        log(f"   GRIND-WEAK{'' if use_switch else '-SOLO'}: team floor under L{target} — fielding the weak "
            f"ones (not the ace) to raise the team's floor; levels now {self._party_levels()}")
        t0 = time.time()
        # PARTICIPATION-XP GRIND SWITCH (only when armed): switch the weak lead out to the ace on turn 1 so
        # it banks XP without being one-shot. In SOLO mode it's OFF — the weak lead fights itself (Super
        # Potions heal it, the ace backstops a faint). Scoped to THIS grind only; restored in the finally.
        battle_agent.PROTECT_LEAD_GRIND = use_switch
        try:
            while time.time() - t0 < GRIND_WEAK_BUDGET_S:
                levels = self._party_levels()
                weak = [s for s, l in enumerate(levels) if l < target]
                if not weak:
                    log(f"   GRIND-WEAK: floor crossed L{target} (levels {levels}) — done");
                    self._restore_ace()
                    return "ready"
                wk = min(weak, key=lambda s: levels[s])
                if wk != 0:
                    log(f"   GRIND-WEAK: fielding slot {wk} (L{levels[wk]}) as lead to train it")
                    self._swap_party_slots(0, wk)
                r = self.grind(target, fragile=True)         # weak mon can faint -> reachable grass only
                if r == "battle_loss":
                    self._restore_ace()
                    return "battle_loss"
                if r != "ok":                                # no grass reachable etc. — surface, restore, retry next tick
                    log(f"   GRIND-WEAK: grind returned {r!r} — restoring ace, retry next tick")
                    self._restore_ace()
                    return "ok"
            log(f"   GRIND-WEAK: wall-clock budget — partial progress (levels {self._party_levels()}), "
                f"restoring ace; the next tick re-enters")
            self._restore_ace()
            return "ok"
        except Exception as e:
            log(f"   GRIND-WEAK: !! errored ({e}) — restoring ace to leave the party sane")
            try:
                self._restore_ace()
            except Exception:
                pass
            return "ok"
        finally:
            battle_agent.PROTECT_LEAD_GRIND = False        # disarm — normal play never grind-switches

    # ── BATCH 2 PART C: SHOP WITH INTENT (bag/money reads + the verified Mart buy primitive) ───────
    def money(self):
        """Player money (SaveBlock1+0x290 XOR SaveBlock2.encryptionKey+0xF20). Same decode as the
        fingerprint; the ground-truth 'did a purchase go through?' signal the buy loop gates on."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20)
        return (self.b.rd32(sb1 + 0x0290) ^ key) & 0xFFFFFFFF

    def bag_count(self, item_id):
        """Count of item_id in the Items pocket (SaveBlock1+0x310, 42 slots). qty XOR'd with the
        low-16 key; itemId plain. The buy verify reads this before/after each unit (ground truth)."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for s in range(42):
            slot = sb1 + 0x0310 + s * 4
            if self.b.rd16(slot) == item_id:
                return self.b.rd16(slot + 2) ^ key
        return 0

    def party_statuses(self):
        """Set of status NAMES currently afflicting any party member (decode_status of each STATUS1 @
        +0x50). Drives PART C's 'buy the cure for what hurt me' (sampled each free-roam tick) and is
        the read PART B's in-battle cure offer will reuse."""
        out = set()
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            if self.b.rd16(base + P_HP) <= 0:        # a fainted mon's status byte is meaningless
                continue
            nm = decode_status(self.b.rd32(base + P_STATUS))
            if nm:
                out.add(nm)
        return out

    def _mart_goto_row(self, row, tries=12):
        """Move the BUY-list cursor to `row`, VERIFYING via the cursor-index readback (MART_CURSOR)
        each press so an eaten d-pad press can't leave us buying the wrong item. Returns True on arrival."""
        for _ in range(tries):
            cur = self.b.rd8(MART_CURSOR)
            if cur == row:
                return True
            self.b.press("DOWN" if cur < row else "UP", 8, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()
        return self.b.rd8(MART_CURSOR) == row

    def _mart_buy_one(self, max_a=26):
        """Buy ONE of the highlighted item: press A (advancing the slow clerk text + confirming the
        qty default 1 + the 'OK?' default YES) and STOP THE INSTANT money drops — the ground-truth
        'bought one' signal (control-proven recon_mart). Breaking on the first drop is critical: a
        further A would re-open the item and buy a second. Returns the price paid, or 0 (nothing
        bought — can't afford / wrong menu -> caller aborts LOUD)."""
        m0 = self.money()
        for _ in range(max_a):
            self.b.press("A", 6, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()
            d = m0 - self.money()
            if d > 0:
                # bought — DRAIN the "Here you are! Thank you!" message back to the list (A only WHILE a
                # box is up; the A that closes the message returns to the list WITHOUT selecting an item,
                # so the next nav reads a live cursor). Without this, the open box freezes _mart_goto_row.
                for _ in range(8):
                    if not dd_box_open(self.b):
                        break
                    self.b.press("A", 6, 10, self.render, owner="agent")
                    for _ in range(12):
                        self.b.run_frame()
                return d
        return 0

    def _mart_enter_buylist(self):
        """Talk to the clerk, DRAIN the greeting (A while the message box is up), pick BUY (one A on the
        BUY/SELL menu, default top), then POSITIVELY CONFIRM the item list via the cursor: a DOWN press
        moves MART_CURSOR (0->1) only in the list. Critical: we stop A-pressing the moment the greeting
        box closes — pressing extra A's once the list is up would SELECT + buy the top item (the runaway
        that drained the wallet). A money-drop during entry = an accidental purchase -> abort LOUD.
        Returns True once the list is confirmed; LOUD False otherwise (never blind-buys)."""
        self.b.set_input_owner("agent")
        guard = self.money()
        self._step_to(MART_CLERK_FRONT)
        for _ in range(6):                                # engage: face the clerk until the box opens
            if dd_box_open(self.b):
                break
            self.b.press("UP", 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 10, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        for _ in range(12):                               # DRAIN greeting: A only WHILE a box is up
            if not dd_box_open(self.b):
                break                                     # box closed -> BUY/SELL menu is up
            self.b.press("A", 8, 12, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        self.b.press("A", 8, 10, self.render, owner="agent")   # pick BUY (default top of BUY/SELL)
        for _ in range(120):                              # let the fade settle into the item list
            self.b.run_frame()
        if self.money() < guard:                          # an A accidentally bought something -> bail LOUD
            log(f"   !! MART: money dropped during entry ({guard}->{self.money()}) — accidental buy, abort")
            return False
        c0 = self.b.rd8(MART_CURSOR)                       # CONFIRM the list: DOWN must move the cursor
        self.b.press("DOWN", 8, 10, self.render, owner="agent")
        for _ in range(12):
            self.b.run_frame()
        if self.b.rd8(MART_CURSOR) != c0:
            self._mart_goto_row(0)                         # back to the top, ready to shop
            return True
        log("   !! MART: could not confirm the BUY list (cursor didn't respond) — aborting LOUD")
        return False

    def buy_at_mart(self, mart_door, shopping_list):
        """SHOP WITH INTENT (PART C): enter the city's Mart, BUY each (item_id, qty) one unit at a time,
        VERIFYING every purchase against the real bag + money (a wrong row/id or eaten press -> the
        target count won't rise -> abort that item LOUD, never silent mis-buy). Bounded by SHOP_MONEY_
        FLOOR (never drains the wallet). Returns a {item_id: bought} dict. Reuses enter_warp / _step_to /
        _exit_to_overworld; the cursor-readback + bag-delta make it robust on the long-running core."""
        city = tv.map_id(self.b)
        stock = MART_STOCK.get(city)
        if stock is None:
            log(f"   !! MART: no stock row-order mapped for {city} — can't shop here (add it to "
                f"MART_STOCK). Skipping."); return {}
        if self.enter_warp(pick=mart_door) != "warped":
            log("   !! MART: couldn't enter the Mart"); return {}
        for _ in range(60):
            self.b.run_frame()
        if not self._mart_enter_buylist():
            self._exit_to_overworld(); return {}
        bought = {}
        for item_id, qty in shopping_list:
            nm = ITEM_NAMES.get(item_id, f"item#{item_id}")
            if item_id not in stock:
                log(f"   MART: {nm} not sold here — skipping"); continue
            row = stock.index(item_id)
            for _ in range(qty):
                if self.money() < SHOP_MONEY_FLOOR:
                    log(f"   MART: at money floor ({self.money()}<{SHOP_MONEY_FLOOR}) — stopping shopping"); break
                before = self._item_count(item_id)
                if not self._mart_goto_row(row):
                    log(f"   !! MART: couldn't reach {nm}'s row {row} (cursor stuck) — abort {nm} LOUD"); break
                price = self._mart_buy_one()
                if price == 0 or self._item_count(item_id) != before + 1:
                    log(f"   !! MART: buy-verify FAILED for {nm} (price={price}, "
                        f"x{before}->x{self._item_count(item_id)}) — abort {nm} LOUD"); break
                bought[item_id] = bought.get(item_id, 0) + 1
            if bought.get(item_id):
                log(f"   MART: bought {bought[item_id]}x {nm}")
        for _ in range(8):                                # B out of the list + clerk menu
            self.b.press("B", 6, 12, self.render, owner="agent")
            for _ in range(14):
                self.b.run_frame()
        self._exit_to_overworld()
        log(f"   MART: shopping done — {bought} (money now {self.money()})")
        return bought

    def _walled(self, state=None):
        """BATCH 6 PHASE 2 — is she up against a SPATIAL wall she's not yet strong enough to pass? Drives
        the foresight stock-up + the deterrence escalation. Conservative: unknown strength -> treat as
        walled (so she prepares rather than charging)."""
        r = self.strat.active_wall_rec()
        if not r or not r.get("map_id"):
            return False
        st_ = state or {}
        pc = st_.get("party_count")
        pl = st_["party"][0]["level"] if st_.get("party") else None
        return not self.strat.stronger_since_wall(pc, pl)

    def _wall_avoid(self, state):
        """Batch-WORLD — maps the active spatial wall makes off-limits to ROUTING (a route-trainer
        she can't beat yet). The HUB case (the wall is on the map she's standing on — e.g. Gary
        triggering on the Cerulean City map) is handled by blocked-DIRS instead, so she's never
        trapped on the very hub she heals at. Empty once she's grown enough to retry."""
        r = self.strat.active_wall_rec()
        if not r or not r.get("map_id"):
            return set()
        pc = state.get("party_count")
        pl = state["party"][0]["level"] if state.get("party") else None
        if self.strat.stronger_since_wall(pc, pl):
            return set()
        wm, cur = tuple(r["map_id"]), tuple(state["map"])
        return set() if wm == cur else {wm}

    def _wall_blocked_dirs(self, state):
        """Batch-WORLD — when the active wall is ON her current map (a trainer guarding an exit, like
        Gary at Cerulean's north edge), the blocked thing is a DIRECTION, not a neighbouring map. Pick
        the exit nearest the wall's coords (from the live grid bounds), so the spatial brief can show
        'NORTH → BLOCKED' truthfully. Best-effort: empty if coords/bounds aren't readable (the brief
        just won't tag a direction; routing still steers her to known places, which are backward)."""
        r = self.strat.active_wall_rec()
        if not r or not r.get("map_id") or not r.get("coords"):
            return set()
        pc = state.get("party_count")
        pl = state["party"][0]["level"] if state.get("party") else None
        if self.strat.stronger_since_wall(pc, pl) or tuple(r["map_id"]) != tuple(state["map"]):
            return set()
        try:
            g = tv.Grid(self.b)
            x, y = r["coords"]
            dist = {"north": y - g.sy_lo, "south": g.sy_hi - y,
                    "west": x - g.sx_lo, "east": g.sx_hi - x}
            present = {self._EDGE[d] for d, _m in self._map_connections()}
            cand = {d: v for d, v in dist.items() if d in present and v >= 0}
            return {min(cand, key=cand.get)} if cand else set()
        except Exception:
            return set()

    def _guide_lookup(self, state):
        """BATCH 6 PHASE 5 — form a templated guide query from the LIVE situation (mode forming a tool
        query from game facts — NOT scripting her personality) and return a characterful ctx line folding
        the result in as HER OWN piecing-together (never 'I googled'). None if nothing usable comes back.
        Scarce + silent: GuideSearch enforces the budget/cooldown; this only shapes the query + framing."""
        place = state.get("place") or "around here"
        wall = self.strat.active_wall_rec()
        if wall and wall.get("lead"):
            who = wall["lead"]
            query = f"Pokemon FireRed how to beat {who} {place} best counter"
            reason = "wall"
        else:
            query = f"Pokemon FireRed {place} where to go next walkthrough"
            reason = "stuck"
        res = self.guide.search(query, reason=reason)
        if not res:
            return None
        snippet = res.splitlines()[0][:200]
        # framed as recollection/working-it-out, so the viewer sees the RESULT in her voice, not a search.
        return (f"(Something you can half-remember about this: {snippet}) Use it if it helps — your call.")

    # ── GATE-UNLOCK questline executor (recognise → derive → execute) ────────────────────────
    def _derive_questline(self, gate):
        """Derive (or re-derive against LIVE RAM) a Questline for a Gate. Central so every call uses the
        same party-count reader + KB."""
        return ql.derive_questline(gate, self._questline_kb, self.b,
                                   party_count_fn=lambda: self.b.rd8(ram.GPLAYER_PARTY_CNT),
                                   log=log, guide=self.guide)   # GuideSearch = the human-walkthrough fallback

    def _open_questline(self, gate, state):
        """Spawn a gate-unlock questline from a recognised Gate. Returns True if there's a real ACTIONABLE
        plan to pursue (so head_to_gym redirects the hands to the unlock errand). Surfaces her DERIVED plan
        once, in character (firewall: `place`/event seam — she narrates, she still chooses)."""
        if not QUESTLINE_ENABLED:
            return False
        q = self._derive_questline(gate)
        if q.actionable is None:
            return False                       # every step already satisfied → the gate is open, no errand
        self._active_questline = q
        log(f"   [roam] 🎯 QUESTLINE OPENED: gate={gate.kind}/{gate.missing} → "
            f"steps={[s.missing for s in q.steps]} actionable={q.actionable.missing} "
            f"(resolved={q.derivable})")
        narr = q.narration()
        if narr:
            self.on_event(narr, kind="route", tier=2)
        return True

    def _clear_questline(self, reason):
        if self._active_questline is not None:
            log(f"   [roam] questline cleared ({reason})")
        self._active_questline = None
        self._ql_entered_doors = set()        # fresh building-tracking for the next questline
        self._ql_inside_target = False        # not deliberately inside any quest building anymore
        self._ql_past_anchor = False          # next questline starts anchor-relative again
        self._ql_bend_maps = set()            # forget the explored bend with the questline
        self._ql_room_sweeps = 0              # fresh room-sweep budget for the next questline
        self._ql_bg_done = set()              # fresh machine/sign tracking too

    def _run_questline_step(self, state):
        """EXECUTOR: advance the active questline by routing toward its current actionable step (reusing
        travel), re-deriving against LIVE RAM each tick so a just-completed step advances/clears. Returns
        a travel-style outcome string. Never spins silently — an un-routable step surfaces and releases."""
        if self._active_questline is None:
            return "no_questline"
        q = self._derive_questline(self._active_questline.gate)
        self._active_questline = q
        if q.complete:
            self.on_event("there — that's sorted. now I can actually get where I was trying to go.",
                          kind="route", tier=2)
            self._clear_questline("complete")
            return "questline_done"
        step = q.actionable
        if step is None or not step.resolved:
            # KB can't resolve this gate → the GuideSearch fallback (Phase 4) fills it in; until then surface
            # + release to normal roam so she never spins on an underivable gate.
            log(f"   [roam] !! QUESTLINE unresolved ({self._active_questline.gate.missing}) — needs the "
                f"guide (Phase 4)/a hand; releasing to normal roam")
            return "questline_unresolved"
        cur_map = tuple(state["map"])
        _L2W = {"N": "north", "S": "south", "E": "east", "W": "west"}
        _OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
        d = step.dir or "north"
        letter = {"north": "N", "south": "S", "east": "E", "west": "W"}.get(d)
        pc, pl = state.get("party_count"), (state["party"][0]["level"] if state.get("party") else None)
        log(f"   [roam] 🧭 QUESTLINE STEP: '{step.human}' — heading {d.upper()} toward "
            f"{step.place_name or 'the destination'} (success={step.success})")
        conns = self._map_connections()
        nbr = next(((g, n) for dd, (g, n) in conns if dd == letter), None)
        if nbr is None:
            # RE-ANCHOR (2026-07-05, run-4 lesson): the step's coarse dir is relative to the GATE map
            # (e.g. "Bill is north" means north OF CERULEAN). Standing on a map BEHIND the anchor with no
            # step-dir edge (post-grind on Route 4, west of Cerulean), the old fallbacks misfire: the
            # frontier is empty (Cerulean already visited) → "arrived at the destination area" → interact
            # with nothing → head_to_gym no-ops forever (the (107,12) wedge). If we haven't yet crossed
            # PAST the anchor this questline, the move is simply: route back TO the anchor city first.
            gate_where = tuple(self._active_questline.gate.where or ()) or None
            if (gate_where and cur_map != gate_where
                    and not getattr(self, "_ql_past_anchor", False)):
                hop = self.world.next_hop(cur_map, gate_where, avoid=self._wall_avoid(state))
                if hop:
                    nxt, edge2 = hop
                    if not self.strat.is_gated(tuple(nxt), pc, pl):
                        log(f"   [roam] 🧭 QUESTLINE RE-ANCHOR: '{step.human}' is relative to "
                            f"{self.world.name(gate_where)} and we're off-frame at "
                            f"{self.world.name(cur_map)} — routing {edge2} -> {nxt} back to the anchor first")
                        return self.trav.travel(target_map=nxt, edge=edge2)
            # The route BENDS: the step's single coarse dir (KB stores a COARSE compass bearing, e.g. "Bill
            # is north") has no map-edge on THIS map. RECON-CONFIRMED case: Cerulean -N-> Route 24 (3,43),
            # whose only exits are S (back) + E -> Route 25 -> Bill. The old code no-op'd here ("no N edge")
            # and stranded her. Instead, EXPLORE the frontier — cross into an UNVISITED connected map
            # (discovery), never back the way the coarse dir came from (exclude the opposite edge), so she
            # learns the bending route LIVE (the world graph then carries it). This is the general
            # "questline handles an unlearned/bending forward segment" capability, not a Route-24 one-off.
            frontier = [(dd, nb) for dd, nb in conns
                        if dd != _OPP.get(letter) and not self.world.visited(tuple(nb))]
            if frontier:
                dd, nb = frontier[0]
                nb = tuple(nb)
                dword = _L2W[dd]
                if self.strat.is_gated(nb, pc, pl):
                    log(f"   [roam] questline: frontier hop {nb} ({dword}) is wall-gated — surfacing")
                    return "wall_gated"
                log(f"   [roam] 🧭 QUESTLINE EXPLORE: no {d} edge from {cur_map} — the route bends; "
                    f"crossing {dd} into unexplored {nb} toward {step.place_name or 'the destination'}")
                if not hasattr(self, "_ql_bend_maps"):
                    self._ql_bend_maps = set()
                self._ql_bend_maps.add(nb)         # remember the bend so a bounce-back can CONTINUE it
                return self.trav.travel(target_map=nb, edge=dword)
            # BEND-CONTINUE (2026-07-05, run-6's Route-24 'arrived' misfire): once the bend has been
            # explored (Route 25 visited), the frontier goes empty on EVERY map — but an empty frontier
            # on a NON-bend map (bounced back to Route 24 after a heal) does NOT mean 'arrived'; it means
            # 'get back on the bend'. Hop toward the recorded bend map; only a map we entered VIA the bend
            # (or a questline with no bend at all) may declare 'arrived' and interact.
            bends = getattr(self, "_ql_bend_maps", set())
            if bends and cur_map not in bends:
                nxt = next(((dd, tuple(nb)) for dd, nb in conns if tuple(nb) in bends), None)
                if nxt is not None:
                    dd, nb = nxt
                    log(f"   [roam] 🧭 QUESTLINE BEND-CONTINUE: back on the explored bend — "
                        f"crossing {dd} into {nb} toward {step.place_name or 'the destination'}")
                    return self.trav.travel(target_map=nb, edge=_L2W[dd])
                # no connection to the bend from here — route back to the anchor and re-walk it
                log(f"   [roam] questline: off the bend at {cur_map} with no direct hop — re-walking "
                    f"via the anchor")
            # No coarse-dir edge AND no unexplored frontier → she's ARRIVED at the destination AREA. The
            # step now COMPLETES by an INTERACTION (talk an NPC, board a ship, use an HM), not more travel —
            # hand off to the general destination-interaction layer (Bill is the first instance; the same
            # path serves the S.S. Anne Cut handoff + every fetch-quest NPC after).
            log(f"   [roam] questline: arrived at the destination area ({cur_map}, no further forward edge) "
                f"— switching to the '{step.via}' interaction")
            return self._questline_interact(state, step)
        if self.strat.is_gated(nbr, pc, pl):
            log(f"   [roam] questline: {d} hop {nbr} is wall-gated — surfacing")
            return "wall_gated"
        # crossing the step-dir edge FROM the anchor map = we're going PAST the anchor now; the re-anchor
        # fallback must stop firing (else a bending route past the anchor would get dragged back to it).
        if cur_map == (tuple(self._active_questline.gate.where or ()) or None):
            self._ql_past_anchor = True
        return self.trav.travel(target_map=nbr, edge=d)

    def _questline_interact(self, state, step):
        """DESTINATION-INTERACTION LAYER (general — the capability that makes questlines COMPLETE, not just
        APPROACH). A `via='talk_npc'` step finishes by TALKING an NPC (Bill → S.S. Ticket, the S.S. Anne
        captain → Cut, any fetch-quest giver), who is usually INSIDE a building. The success FLAG (re-checked
        each tick by the deriver) is the real done-signal, so this NEVER hardcodes a map number (cross-check
        rule) — it just does the obvious interaction until the flag flips:
          • INSIDE a building (interior = map group != 3): talk the occupant(s). Next tick the deriver re-reads
            the flag; when it sets, the questline self-clears (proven). Talked everyone + still no flag →
            wrong building → exit and keep looking.
          • OUTSIDE (overworld, group 3): enter an un-entered building door here (the NPC's inside); if there's
            no building but an NPC standing out here, talk them.
        Bounded: entered doors are tracked per-questline so she can't loop re-entering the same one. Reuses
        the existing enter_warp / talk_npc / _exit_to_overworld primitives — this only SEQUENCES them."""
        if step.via != "talk_npc":
            # other step kinds (board a ship, use an HM at the destination) are future interaction layers
            log(f"   [roam] questline: arrived but step.via='{step.via}' has no interaction layer yet — surfacing")
            return "questline_no_interaction"
        if not hasattr(self, "_ql_entered_doors"):
            self._ql_entered_doors = set()
        interior = tv.map_id(self.b)[0] != 3
        if interior:
            # She's inside a building she ENTERED for the quest (the blackout-recovery leaves her be while
            # `_ql_inside_target` is set — they cooperate). Talk the occupant(s); the flag re-checks next tick.
            r = self.talk_npc()
            if r == "talked":
                log("   [roam] 🗣️ QUESTLINE TALK: spoke to someone inside — re-checking the flag next tick")
                return "questline_talked"
            # WORK THE ROOM (2026-07-05, the Bill Cell-Separation class): quest buildings have scripted
            # MACHINES/consoles — BG events read live from the map header (no hardcoded coords). Interact
            # them facing-correct; a fired script (box opened) means world state advanced — re-check next tick.
            if self._questline_bg_sweep():
                log("   [roam] 🖥️ QUESTLINE ROOM: worked a machine/sign in here — re-checking the flag next tick")
                return "questline_worked_room"
            # RE-SWEEP (bounded): scripts change WHO is in the room (Bill re-appears human after the
            # separation) — clear the talked-set and go again before giving up on the building.
            sweeps = getattr(self, "_ql_room_sweeps", 0)
            if sweeps < 2:
                self._ql_room_sweeps = sweeps + 1
                try:
                    self._talked_npcs.get(tv.map_id(self.b), set()).clear()
                    getattr(self, "_ql_bg_done", set()).clear()   # scripts are state-dependent — re-arm them too
                except Exception:
                    pass
                log(f"   [roam] questline: room re-sweep {sweeps + 1}/2 (scripts change who's here)")
                return "questline_resweep"
            # nobody left to talk + the flag didn't set → wrong building; release the 'stay inside' marker
            # so the blackout-recovery can exit her, and keep looking at the next candidate building.
            log("   [roam] questline: no one left to talk to in here and the flag's not set — leaving to keep looking")
            self._ql_room_sweeps = 0
            self._ql_inside_target = False
            self._exit_to_overworld()
            return "questline_wrong_building"
        # overworld: the NPC is almost always inside a building → enter an un-entered door (NPC's in there)
        door = self._questline_unentered_door()
        if door is not None:
            self.on_event("this looks like the place — let me head inside.", kind="route", tier=1)
            before = tv.map_id(self.b)
            # 900s approach budget: the door can sit past a trainer GAUNTLET (Route 25 -> Bill = 6+
            # fights en route); the 300s default aborted at half-distance (run-5 'enter_failed').
            self.enter_warp(pick=door, budget_s=900)
            if tuple(tv.map_id(self.b)) != tuple(before):
                self._ql_entered_doors.add((tuple(before), tuple(door)))
                self._ql_inside_target = True       # tell the blackout-recovery to LEAVE HER inside (she's
                log(f"   [roam] 🚪 QUESTLINE ENTER: stepped into the building at door {door}")  # here on purpose
                return "questline_entered"
            log(f"   [roam] questline: tried to enter the building at {door} but didn't warp — surfacing")
            return "questline_enter_failed"
        # no building to enter here → maybe the quest NPC is standing out on the route
        r = self.talk_npc()
        if r == "talked":
            log("   [roam] 🗣️ QUESTLINE TALK: spoke to someone out here — re-checking the flag next tick")
            return "questline_talked"
        log("   [roam] questline: arrived but found no building/NPC to interact with here — surfacing")
        return "questline_arrived_no_target"

    def _questline_bg_sweep(self):
        """Interact the CURRENT map's BG events (scripted machines/consoles/signs — Bill's Cell-Separation
        PC is instance #1), read LIVE from the map header (general, zero hardcoded coords). Approaches the
        facing-correct side per the event's kind (1=face N, 2=S, 3=E, 4=W, 0=any side), FACE-VERIFIED turn,
        A with a box readback (+ a longer settle so a triggered cutscene can play). Tracks interacted tiles
        per building visit. Returns True iff a box actually opened (a script fired = world advanced)."""
        from dialogue_drive import box_open as _box
        if not hasattr(self, "_ql_bg_done"):
            self._ql_bg_done = set()
        mp = tuple(tv.map_id(self.b))
        _SIDES = {1: [((0, 1), "UP")], 2: [((0, -1), "DOWN")], 3: [((-1, 0), "RIGHT")], 4: [((1, 0), "LEFT")],
                  0: [((0, 1), "UP"), ((0, -1), "DOWN"), ((-1, 0), "RIGHT"), ((1, 0), "LEFT")]}
        for (bx, by), kind in tv.read_bg_events(self.b):
            if kind > 4 or (mp, (bx, by)) in self._ql_bg_done:
                continue
            grid = tv.Grid(self.b)
            cur = tv.coords(self.b)
            for (adx, ady), face in _SIDES.get(kind, []):
                front = (bx + adx, by + ady)
                if not grid.walkable(*front):
                    continue
                if front != cur and not tv.bfs(grid, cur, lambda t, f=front: t == f, walkable=grid.walkable):
                    continue
                if front != cur:
                    self.trav.travel(target_map=None, arrive_coord=front, max_steps=60, max_seconds=40)
                if tv.coords(self.b) != front:
                    continue
                self._face_verified(face)
                for _a in range(2):
                    self.b.press("A", 6, 12, self.render, owner="agent")
                    for _f in range(50):
                        self.b.run_frame(); self.render()
                    if _box(self.b):
                        log(f"   [roam] 🖥️ BG-EVENT fired at ({bx},{by}) (facing {face}) — driving")
                        self._drain_overworld(label="bg-event")
                        for _f in range(90):               # let a triggered cutscene finish
                            self.b.run_frame(); self.render()
                        self._ql_bg_done.add((mp, (bx, by)))
                        return True
                break                                      # correct side tried clean — this event is inert now
            self._ql_bg_done.add((mp, (bx, by)))           # unreachable/inert — don't re-try every sweep
        return False

    def _questline_unentered_door(self):
        """Nearest reachable building door she hasn't ENTERED for the current questline (so she works through
        the candidate buildings at the destination instead of re-entering one). (x,y) door tile or None.
        Overworld only — reuses _door_tiles + a BFS reachability check."""
        if not hasattr(self, "_ql_entered_doors"):
            self._ql_entered_doors = set()
        try:
            cur_map = tuple(tv.map_id(self.b))
            co = tv.coords(self.b)
            grid = tv.Grid(self.b)
        except Exception:
            return None
        if co is None:
            return None
        cands = []
        for dr in self._door_tiles():
            if (cur_map, tuple(dr)) in self._ql_entered_doors:
                continue
            approach = (dr[0], dr[1] + 1)               # stand just SOUTH of the door, step UP to enter
            if tv.bfs(grid, co, lambda t, a=approach: t == a, walkable=grid.walkable):
                cands.append(tuple(dr))
        if not cands:
            return None
        cands.sort(key=lambda t: abs(t[0] - co[0]) + abs(t[1] - co[1]))
        return cands[0]

    def _ensure_forward_questline(self, state):
        """PROACTIVE forward drive (ROOT FIX for the backward-grind). Recognise the gate on the FORWARD
        road to the next gym and OPEN/refresh the unlock questline at DECISION time — BEFORE the action set
        is built — so the errand pulls her FORWARD every tick, not only when she physically bonks the wall.
        This is what makes the forward objective DOMINATE the grind impulse (she heads NORTH to Bill for the
        S.S. Ticket instead of west to Route 4 to grind). Mirrors the reactive recognition the head_to_gym
        branch does, run a beat earlier. Self-clears the instant the gate's success flag reads satisfied
        LIVE. No-op when QUESTLINE is off, on an interior/cave map, or there's no forward gate here.

        FORWARD = south on the current spine (Cerulean→Vermilion), matching the reactive recognition in
        _route_action(head_to_gym) and the curated KB exit_gates. The recogniser no-ops on any map/dir the
        KB has no gate for, so this is safe to call every tick everywhere — it only opens where a real
        story/HM gate sits on the road ahead. Firewall: harness routing; her DERIVED plan reaches the
        decision ctx via the existing `place` seam (she narrates + still chooses)."""
        if not QUESTLINE_ENABLED:
            return
        try:
            cur_map = tuple(state["map"])
            # Already on an errand → re-derive against LIVE RAM so a just-completed step advances or the
            # whole questline self-clears (the gate opened). Keeps the ctx + action-set honest WITHOUT
            # walking a step here (the executor walks it when she picks head_to_gym).
            if self._active_questline is not None:
                q = self._derive_questline(self._active_questline.gate)
                if q.complete:
                    self._clear_questline("flag satisfied (proactive refresh)")
                else:
                    self._active_questline = q
                return
            # No active errand → is the FORWARD exit a story/HM gate she can't pass yet (the Cerulean
            # Slowbro / S.S.-Ticket story-block, read LIVE)? Recognise it and open the unlock questline so
            # head_to_gym drives THAT and the action-set reframes around it.
            gate = self._gate_recognizer.recognize(cur_map, blocked_dir="south")
            if gate:
                self._open_questline(gate, state)
        except Exception as _q:
            log(f"   [roam] proactive questline check skipped: {_q}")

    def _thin_team(self):
        """BATCH 6 PHASE 3 — is she running a thin bench (≤ SHOP_THIN_PARTY)? Drives Poké Ball foresight:
        a thin team is the one that most needs to come home from the grass with a new member."""
        try:
            return self.b.rd8(ram.GPLAYER_PARTY_CNT) <= SHOP_THIN_PARTY
        except Exception:
            return False

    def _shopping_list(self, foresight=False):
        """What a sensible player would BUY here, given the bag + what's hurt her: top Potions up to the
        target (SHOP_POTION_FORESIGHT when she's walled and stocking up before a push, else
        SHOP_POTION_TARGET), and SHOP_CURE_QTY of the cure for each status seen recently (_afflict_seen).
        Bounded quantities (survival, not hoarding); returns [(item_id, qty), ...] (empty = well-stocked)."""
        sl = []
        target = SHOP_POTION_FORESIGHT if foresight else SHOP_POTION_TARGET
        # count ALL healing-potion tiers she already carries (a Super Potion is still "a potion" for stock)
        have = sum(self.bag_count(i) for i in (ITEM_POTION, 22, 21, 20, 19))
        pot_need = target - have
        if pot_need > 0:
            sl.append((self._best_potion_for_sale(), pot_need))
        # BATCH 6 PHASE 3 — Poké Ball foresight: thin team + low on balls = she should leave the Mart
        # equipped to actually catch a teammate (the real answer to a wall). Bag-delta verified like any buy.
        if self._thin_team() and self._ball_count() < SHOP_BALL_TARGET:
            sl.append((ITEM_POKE_BALL, SHOP_BALL_TARGET - self._ball_count()))
        for status in sorted(self._afflict_seen):
            cure = STATUS_CURE.get(status)
            if not cure:
                continue
            need = SHOP_CURE_QTY - self.bag_count(cure[0])
            if need > 0:
                sl.append((cure[0], need))
        return sl

    def _best_potion_for_sale(self):
        """The STRONGEST healing potion the current city's Mart sells that she can comfortably afford — a
        real player buys Super Potions (50 HP) over Potions (20 HP) when flush, because a war of attrition
        vs a super-effective hitter (Gary's Charmander) is lost 20 HP at a time. Falls back to Potion.
        GAME-KNOWLEDGE: heal-tier ids + rough prices are FRLG facts (rule 14 — isolate to gamedata on port)."""
        stock = MART_STOCK.get(tv.map_id(self.b), [])
        money = self.money()
        for iid, price in ((20, 2500), (21, 1200), (22, 700), (ITEM_POTION, 300)):   # Max/Hyper/Super/Potion
            if iid in stock and (money - SHOP_MONEY_FLOOR) >= price * 3:
                return iid
        return ITEM_POTION

    def _shop_note(self, foresight=False):
        """Characterful ctx line for the stock-up offer: names what hurt her + the restock need, so her
        pick reads as learning from the road ('paralysis cost me that fight — grabbing Parlyz Heals').
        FORESIGHT (Phase 2): when she's walled, lead with 'stock up BEFORE you push that wall'."""
        target = SHOP_POTION_FORESIGHT if foresight else SHOP_POTION_TARGET
        cures = [STATUS_CURE[s][1] for s in sorted(self._afflict_seen) if s in STATUS_CURE
                 and self.bag_count(STATUS_CURE[s][0]) < SHOP_CURE_QTY]
        bits = []
        if sum(self.bag_count(i) for i in (ITEM_POTION, 22, 21, 20, 19)) < target:
            bits.append("you're low on Potions" if not foresight
                        else "you could carry a lot more Potions before that next push")
        if self._thin_team() and self._ball_count() < SHOP_BALL_TARGET:
            bits.append("you're light on Poké Balls — grab some so you can actually catch a teammate out there")
        if cures:
            afflicts = ", ".join(sorted(self._afflict_seen & set(STATUS_CURE)))
            bits.append(f"{afflicts} has been hurting you — {', '.join(cures)} would help")
        if not bits:
            return ""
        head = ("There's a Mart right here, and you've got the money. " if foresight
                else "There's a Mart right here. ")
        tail = (" — stocking up on healing BEFORE you walk back into that wall is the smart play, not "
                "charging in empty-handed." if foresight
                else " — a real trainer stocks up before pushing on.")
        return head + "; ".join(bits) + tail

    def roster_react(self):
        """SOUL beat after a hand-play-banked catch (the agent can't drive this core's battle menus,
        so acquisition is hand-played — but the RELATIONSHIP is live). Emit a NEUTRAL roster-change
        event so Kira reacts to her new teammate IN HER OWN VOICE via the existing seam (fulfilling
        her want for a partner). Touches NO core (on_event -> _pokemon_react -> her self)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if cnt <= 1:
            log("   ROSTER_REACT: party size 1 — no new teammate to react to"); return "reacted"
        name = st.SPECIES_NAME.get(st.read_party_species(self.b, cnt - 1), "a new Pokemon")
        log(f"   ROSTER_REACT: party now {cnt} — newest teammate is {name}")
        # BATCH 5 PHASE 4 — ROSTER AS FAMILY: a catch is a real moment, not a stat line. She NAMES the
        # new teammate IN CHARACTER (the soul oracle, same pattern as the evolution naming), and it's
        # recorded as a family member whose name + her opinion PERSIST (soul.bonds -> continuity, and the
        # whole campaign anchors via Phase-1 save). Naming is hers; headless/no-oracle -> species name.
        nick = name
        place = self._PLACE_NAMES.get(tv.map_id(self.b))
        where = f"on {place}" if place else None
        if self.soul is not None:
            try:
                nm = self._soul_choose("name", {}, {"place":
                    f"you just caught a {name}" + (f" {where}" if where else "") + "! a brand-new member "
                    f"of your team — a teammate, family. give this {name} a name, just the name."})
                if nm:
                    nick = nm.strip().split("\n")[0][:12] or name
            except Exception as _e:
                log(f"   [soul] catch-naming skipped: {_e}")
            log(f"   [soul] note_caught FIRE -> species={name} nickname={nick} where={where}")
            self.soul.note_caught(name, nick, where)        # records bond (name + opinion) + emits via seam
            if nick.lower() != name.lower():
                self.on_event(f"welcome to the family, {nick}. you're one of us now.", kind="roster", tier=3)
        else:
            self.on_event(f"you've got a new teammate now — a {name} that's going to fight alongside you")
        return "reacted"

    # ── SOUL hook router (MODE-layer; reads RAM + calls soul hooks; never touches core mood/bond) ──
    def _soul_choose(self, kind, options, ctx):
        """Batch-2 SOUL ORACLE seam — where a choice becomes HERS. Delegates to the injected oracle
        (HTTP -> her self/LLM via bot._pokemon_choose), so the PICK is hers, NEVER authored here.

        BEHAVIORAL CONTROL: prints the decision point, the candidate options offered, and the pick
        returned — proof she's CHOOSING, not being scripted. Headless / no-bot -> returns None (no want
        surfaces; constrained callers fall back to their safe default). CONSTRAINED kinds are re-validated
        against `options` so a hallucinated or DEAD action (shop/fish this round) can NEVER be picked."""
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        log(f"   [soul] ORACLE decision: kind={kind} ctx={ctx} options={opts}")
        if self._oracle_choose is None:
            log("   [soul] ORACLE unwired (headless/no bot) -> None")
            return None
        pick = self._oracle_choose(kind, options, ctx)
        if pick and kind != "want" and opts and pick not in opts:
            log(f"   [soul] ORACLE pick {pick!r} NOT in offered options {opts} -> REJECTED (fall back)")
            return None
        log(f"   [soul] ORACLE pick -> {pick!r}")
        return pick

    def _soul_after_objective(self, kind, out):
        """Route post-objective SOUL beats through the seam: (a) note_evolve when the lead's species
        changed, (b) note_faint + note_outcome(won=False) on a blackout/loss, (c) note_outcome(won=True)
        on a gym clear. Signal + wiring + LOG only — mood/bond math is core's (firewall)."""
        try:
            sp = st.read_party_species(self.b, 0)
            pid = self.b.rd32(ram.GPLAYER_PARTY + 0)      # slot-0 personality value
            if self._last_lead_species and sp and sp != self._last_lead_species:
                before = st.SPECIES_NAME.get(self._last_lead_species, "?")
                after = st.SPECIES_NAME.get(sp, "?")
                # EVOLUTION keeps the mon's PID; a party REORDER (grind fielding / _restore_ace) puts a
                # DIFFERENT mon in slot 0. Only a same-PID species change is a real evolution — anything
                # else was the false "ivysaur evolved into spearow" voice lie (2026-07-05 known bug).
                if pid and self._last_lead_pid and pid == self._last_lead_pid:
                    log(f"   [soul] note_evolve FIRE -> {before} -> {after} (same PID — real evolution)")
                    self.soul.note_evolve(before, after)
                else:
                    log(f"   [soul] lead swap {before} -> {after} (PID changed — party reorder, "
                        f"NOT an evolution; no beat)")
            if sp:
                self._last_lead_species = sp
                self._last_lead_pid = pid
        except Exception as _e:
            log(f"   [soul] evolve-check err {_e}")
        if out in ("battle_loss", "blackout"):
            lead = st.SPECIES_NAME.get(self._last_lead_species or 0, "the lead")
            log(f"   [soul] note_faint FIRE -> {lead}; note_outcome FIRE -> won=False")
            self.soul.note_faint(lead)
            self.soul.note_outcome(False)
        elif kind == "BEAT_GYM" and out not in ("stuck", "no_path", "no_warp", "underleveled"):
            log("   [soul] note_outcome FIRE -> won=True (gym cleared)")
            self.soul.note_outcome(True, "a badge")

    # ════ BATCH-2 FREE ROAM — she decides her own next move, unscripted ═══════════════════════════
    _EDGE = {"N": "north", "S": "south", "W": "west", "E": "east"}

    def _map_connections(self):
        """Live map edge connections [(dir, (grp,num))] read from the map header — REAL ids, not guessed.
        FRLG MapHeader@0x02036DFC +0x0C -> {s32 count, MapConnection[]}; conn +0x00 dir +0x08 grp +0x09 num."""
        b = self.b
        try:
            ch = b.rd32(0x02036DFC + 0x0C)
            if ch < 0x02000000:
                return []
            cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
            if not (0 < cnt < 16) or arr < 0x02000000:
                return []
            dirn = {1: "S", 2: "N", 3: "W", 4: "E"}
            return [(dirn[d], (b.rd8(c + 0x08), b.rd8(c + 0x09)))
                    for i in range(cnt) for c in (arr + i * 0xC,) for d in (b.rd8(c),) if d in dirn]
        except Exception as e:
            log(f"   [roam] connection read failed: {e}")
            return []

    def _reachable_grass(self):
        """Confirmed huntable grass on the CURRENT map: a grass tile INSIDE playable bounds that BFS can
        reach from the player. Returns the reached grass save-coord, else None. Excludes connection-bleed
        grass (e.g. Route 5 grass in Cerulean's layout buffer, ~24 tiles outside the playable y-range)."""
        co = tv.coords(self.b)
        if co is None:
            return None
        try:
            g = tv.Grid(self.b)
        except Exception:
            return None
        playable = {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for (bx, by) in g.grass}
        playable = {(sx, sy) for (sx, sy) in playable
                    if g.sx_lo <= sx <= g.sx_hi and g.sy_lo <= sy <= g.sy_hi}
        if not playable:
            return None
        path = tv.bfs(g, co, lambda t: t in playable)
        return path[-1] if path else None

    def _learn_map(self, state=None):
        """BATCH 6 PHASE 2 — fold the CURRENT map's real connections + grass into the learned graph
        (pure reads, cached). Called once per free-roam tick so the multi-hop grass finder can later
        route her BACK to grass on already-cleared routes. Never guesses — only records what the live
        map header + a real BFS confirm."""
        cur = tuple(tv.map_id(self.b))
        conns = {self._EDGE[d]: tuple(m) for d, m in self._map_connections()}
        if conns:
            self._conn_graph.setdefault(cur, {}).update(conns)
        has_grass = False
        try:
            if self._reachable_grass() is not None:
                self._grass_maps.add(cur)
                has_grass = True
        except Exception:
            pass
        # Batch-WORLD — fold this map into her persistent mental map: live name + connectivity +
        # live-confirmed traits (grass underfoot; Mart/Center from the door tables; group!=3 = interior).
        try:
            live_traits = {"has_grass": has_grass, "has_wild": has_grass,
                           "has_mart": cur in CITY_MART_DOORS,
                           "has_pokecenter": cur in CITY_PC_DOORS,
                           "is_town": cur in CITY_MART_DOORS or cur in CITY_PC_DOORS}
            warps = tv.read_warps(self.b)      # WARP-ROUTING: learn this map's warp tiles + destinations live
            self.world.note_visit(cur, name=self._PLACE_NAMES.get(cur), live_traits=live_traits,
                                  edges=conns, warps=warps)
            if warps:
                log(f"   [world] learned {len(warps)} warp(s) on {cur}: "
                    + ", ".join(f"{xy}->{dst}" for xy, dst, _w in warps[:6]))
        except Exception as _w:
            log(f"   [world] note_visit skipped: {_w}")
        return cur

    def _refresh_world_caps(self):
        """Batch-WORLD — update her capability registry from the live game (knows-the-HM AND
        has-the-badge, via field_moves). Walk is always hers; Fly/Surf/etc. only when truly owned,
        so the spatial brief reflects reality and she USES Fly/Surf the moment she earns them."""
        can_use = None
        if fm is not None:
            pc = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            can_use = lambda hm: fm.can_use(self.b, hm, pc)   # noqa: E731
        self.world.refresh_caps(can_use)

    @staticmethod
    def _is_route_map(m):
        """A group-3 ROUTE (cities are num 0..~12; routes >=19 carry wild grass) — a grass CANDIDATE."""
        return bool(m) and m[0] == 3 and m[1] >= 19

    def _grass_via_graph(self, state):
        """BFS the LEARNED connection graph (seeded with the current map's LIVE connections) for the
        nearest map with huntable grass — confirmed (_grass_maps) OR a group-3 route candidate — reachable
        WITHOUT stepping onto a gated wall map. Returns (next_hop_map, edge) for the FIRST step toward it
        (she travels one hop/tick, re-evaluating), else None. THIS is what lets her turn AROUND to grass
        behind her instead of dying on the gated bridge forever (the live Gary death-loop)."""
        from collections import deque
        cur = self._learn_map(state)
        pcount = state.get("party_count")
        plevel = state["party"][0]["level"] if state.get("party") else None
        def gated(m):
            return self.strat.is_gated(m, pcount, plevel)
        def has_grass(m):
            return (m in self._grass_maps or self._is_route_map(m)) and m != cur
        seen = {cur}
        q = deque()
        for edge, nbr in self._conn_graph.get(cur, {}).items():
            if nbr in seen or gated(nbr):
                continue                              # never step toward the wall
            seen.add(nbr); q.append((nbr, edge))      # carry the FIRST-hop edge so we can return it
        while q:
            node, first_edge = q.popleft()
            if has_grass(node):
                # one hop toward the grass; if node is itself the adjacent grass, that's the hop.
                first_map = self._conn_graph.get(cur, {}).get(first_edge)
                return (first_map or node, first_edge)
            for edge, nbr in self._conn_graph.get(node, {}).items():
                if nbr in seen or gated(nbr):
                    continue
                seen.add(nbr); q.append((nbr, first_edge))
        return None

    def _grass_target(self, state):
        """Where she can HONESTLY hunt: ('here', tile) if reachable on this map, else ('route', (g,n),
        edge) for grass she can reach WITHOUT crossing the gated wall — an ungated adjacent route first,
        else (Phase 2) the nearest grass BEHIND her via the learned graph, else the gated route as a last
        resort (so _route_action surfaces the wall rather than silently offering a phantom hunt).
        catch_one re-verifies grass on arrival (no_grass backstop) so a wrong guess never freezes."""
        tile = self._reachable_grass()
        if tile is not None:
            return ("here", tile)
        pcount = state.get("party_count")
        plevel = state["party"][0]["level"] if state.get("party") else None
        # Batch-WORLD (Phase 4b) — prefer grass she KNOWS: the nearest VISITED grass she can reach
        # WITHOUT crossing a wall (adjacent or several hops BEHIND her). This is the missing verb — she
        # walks back to Route 4 / Mt Moon that she's already cleared, instead of blindly routing onto
        # the unvisited route across Gary's bridge (which the old live-connection scan below would pick).
        try:
            cur = tuple(state["map"])
            avoid = self._wall_avoid(state)
            known = self.world.reachable_with_trait(cur, "has_grass", avoid)
            if known:
                dst = known[0][0]
                hop = self.world.next_hop(cur, dst, avoid)
                if hop:
                    nxt, edge = hop
                    log(f"   [roam] grass she KNOWS: routing {edge} -> {nxt} toward {self.world.name(dst)} "
                        f"(visited grass, avoiding any wall)")
                    return ("route", nxt, edge)
        except Exception as _gt:
            log(f"   [roam] world grass-target skipped: {_gt}")
        routes = [(d, (grp, num)) for d, (grp, num) in self._map_connections()
                  if grp == 3 and num >= 19]
        # BATCH-4 PHASE 2 (route AROUND the wall): prefer a grass route that ISN'T gated by an active
        # wall. Adjacent ungated route wins (this already includes grass directly behind her).
        non_gated = [(d, m) for d, m in routes if not self.strat.is_gated(m, pcount, plevel)]
        if non_gated:
            d, m = non_gated[0]
            return ("route", m, self._EDGE[d])
        # BATCH-6 PHASE 2: no UNGATED grass directly adjacent — before surfacing the gated bridge (the
        # death-loop), BFS the learned graph for grass BEHIND her on already-cleared routes. One hop/tick.
        hop = self._grass_via_graph(state)
        if hop is not None:
            nxt, edge = hop
            log(f"   [roam] grass-behind-the-wall: routing one hop {edge} -> {nxt} toward known/likely grass "
                f"(avoiding the gated route)")
            return ("route", nxt, edge)
        # Truly only the gated route exists -> surface it; _route_action's gate tells her it's blocked.
        if routes:
            d, m = routes[0]
            return ("route", m, self._EDGE[d])
        return None

    def _travel_to_known(self, pick, state):
        """Batch-WORLD (Phase 4b/c) — actuate 'travel to a place you've been', BACKWARD or forward, via
        the learned warp graph. One adjacent hop per tick (re-evaluated each tick like head_to_gym), and
        NEVER across an active gated wall map (Phase 4c — _wall_avoid). Fly is used when she owns it and
        the destination is a visited town (Jonny wants to SEE her fly, not walk the whole map); it
        degrades to walking if the fly actuation isn't live yet. On arrival at grass she starts hunting
        so 'go to Route 4' visibly becomes catching. Returns a status string the loop logs + feeds soul."""
        try:
            g, n = pick.split(":", 1)[1].split(",")
            dst = (int(g), int(n))
        except Exception:
            return "bad_travel_target"
        cur = tuple(state["map"])
        if cur == dst:
            return "arrived"
        # FLY (capability-gated, Phase 3): fast-travel to a visited town when she actually owns Fly.
        if self.world.has_cap("fly") and self.world.is_town(dst):
            try:
                fr = self.fly_to(self.world.name(dst))
                if fr not in ("not_owned", "stuck", None, False):
                    log(f"   [roam] ✈️  FLEW to {self.world.name(dst)} -> {fr}")
                    return "flew"
                log(f"   [roam] fly to {self.world.name(dst)} unavailable ({fr}) — walking instead")
            except Exception as _fe:
                log(f"   [roam] fly attempt errored ({_fe}) — walking instead")
        avoid = self._wall_avoid(state)
        hop = self.world.next_hop(cur, dst, avoid=avoid)
        if hop is None:
            log(f"   [roam] travel to {self.world.name(dst)}: no wall-free route from {self.world.name(cur)}")
            return "no_route"
        nxt, edge = hop
        log(f"   [roam] TRAVEL: {self.world.name(cur)} -> {self.world.name(dst)} "
            f"(next hop {edge} -> {self.world.name(nxt)})")
        r = self._edge_travel(nxt, edge)
        if r != "arrived":
            return f"travel:{r}"
        if tuple(tv.map_id(self.b)) == dst:
            node = self.world.node(dst)
            if node and node["traits"].get("has_grass"):
                log(f"   [roam] arrived at {self.world.name(dst)} (grass) — starting to hunt")
                return self.catch_one()
            return "arrived"
        return "hop_ok"

    def _available_actions(self, state):
        """HONEST per-tick action set — only actions that actually DO something here (no phantom/no-op).
        head_to_gym always (progress exists); heal only if hurt; wander_catch/battle only if huntable
        grass is reachable (here or an adjacent route). This is the roamable-area-honesty guard."""
        a = {}
        sev, _note = self._hurt_severity()
        # HEAL-WHEN-HURT (PART A) — TWO TIERS so heal-when-hurt doubles as the auto-fix for a broken
        # low-HP boot (no hand-made healthy save needed):
        #   CRITICAL (near-death, <=PARTY_CRIT_FRAC or a fainted member): heal must DOMINATE, not be one
        #     option among equals. A 4/60 Kira wandering grass / grinding / marching to a gym is a
        #     blackout PATH, not a real choice — so we PRUNE those and offer ONLY heal. She still PICKS
        #     it and voices it in-character ("okay, I'm nearly dead — Center first"); this is honest
        #     action-set pruning (the same principle as offering only actions that DO something here),
        #     NOT a forced action (the step-3 RED hard-recovery is the only forced move; it backstops
        #     a heal that can't route). This is what reliably wins against catch/wander/gym at near-death.
        #   HURT (<PARTY_HURT_FRAC) or the existing lead convenient gate (needs_heal, lead <0.75): heal
        #     is offered as a strong option she WEIGHS against the rest — the softer nudge (needs_heal
        #     semantics unchanged).
        if sev == "critical":
            a["heal"] = ("you're about to faint — get to a Pokémon Center and heal NOW; this comes "
                         "FIRST, before anything else")
            return a
        ng = state.get("next_gym")
        if ng:
            a["head_to_gym"] = f"head toward the next gym - {ng['leader']} of {ng['city']}"
        # 2026-07-06 STRAND GUARD: heal is NOT offered (while non-critical) on a map where the last
        # heal attempt proved the Center unreachable (a one-way pocket, e.g. Cerulean's south strip) —
        # re-offering it was the run-4 ping-pong stall. Critical severity above still overrides (the
        # RED hard-recovery backstops a heal that can't route).
        _heal_dead = tuple(state.get("map") or ()) in getattr(self, "_heal_dead_maps", set())
        if (self.needs_heal() or sev == "hurt") and not _heal_dead:
            a["heal"] = "go to a Pokemon Center and heal the team up"
        # PP DEPLETION (Phase 4) — a Center restores PP, not just HP. When the lead can barely act
        # (≤1 move with PP left) she can neither weaken-to-catch nor fight safely — the depleted spiral
        # that froze her vs Mankey + burned her balls. Offer heal even at full HP so she tops up first.
        elif self._lead_pp_low():
            a["heal"] = ("you're nearly out of PP — swing by a Pokémon Center to restore your moves "
                         "(on empty you can't weaken-to-catch or fight safely)")
        if self._grass_target(state) is not None:
            a["battle"] = "train in the grass - fight wild pokemon to level the team up"
            # Phase 5 BALL PRE-CHECK: only offer catching when she actually HAS a ball — wandering
            # grass with zero balls can't catch anything (honest action set). With no balls, the
            # ball-note + a 'travel to a Mart' option steer her to buy some first.
            try:
                has_balls = self._ball_count() > 0
            except Exception:
                has_balls = True
            if has_balls:
                a["wander_catch"] = "wander the grass and try to catch a new teammate"
                # nursery drive (2026-07-06): with a thin team + balls in the bag, catching is
                # PURPOSEFUL — she's building the squad, judging keepers, not collecting at random.
                try:
                    if state.get("party_count", 6) < 4 and self._balls_pocket_count(ITEM_POKE_BALL) > 0:
                        a["wander_catch"] = (f"BUILD THE TEAM — you're carrying "
                                             f"{state.get('party_count')} and a real trainer fields a "
                                             f"balanced ~6. hunt the grass for a KEEPER (something that "
                                             f"covers a type you lack, at a workable level) — judge each "
                                             f"one, don't ball everything")
                except Exception:
                    pass
            # STRATEGIC UNDERLEVEL-GRIND (Task B) — FRAME the grind as the WEAK-member prep when the team
            # FLOOR is under the wall's level. This makes her PICK "battle" knowing it means "field
            # Rattata/Spearow, level THEM" (the executor does exactly that), and surfaces the strategic
            # reasoning to her decision/voice ctx — fixing the half-wire where the display said "train the
            # team" but the action trained the ace. Fires on the FIRST loss already (foe level is known).
            prep_t = self._prep_team_target(state)
            if prep_t is not None:
                if not battle_agent.GRIND_SWITCH_ENABLED:
                    # ACE-OVERPOWER framing — level the strong lead to bulldoze the wall.
                    a["battle"] = (f"STRENGTHEN FIRST — you keep losing that wall under-levelled. Grind your "
                                   f"strongest up to about L{prep_t} in the grass so you can bulldoze "
                                   f"through it — a clear level lead wins where an even fight didn't. THEN go "
                                   f"back and push through.")
                else:
                    weak = self._prep_team_weak(state, prep_t)
                    who = (weak[0] if len(weak) == 1
                           else (", ".join(weak[:-1]) + " and " + weak[-1])) if weak else "your weakest"
                    a["battle"] = (f"STRENGTHEN FIRST — your team's under-levelled for that wall. Train the "
                                   f"WEAK ones ({who}) up to about L{prep_t} by leading with THEM in the grass "
                                   f"(not your strongest) — that's how the whole team gets strong enough to "
                                   f"cross. THEN go back and push through.")
        else:
            prep_t = None
        # SHOP WITH INTENT (PART C) + BATCH-6 PHASE-2 FORESIGHT: offer "stock up" when it DOES something —
        # at a town with a mapped Mart and money above the floor, with EITHER a real restock need (low on
        # potions / missing a cure) OR she's walled and could prepare deeper before a push (foresight,
        # even if not yet empty). Naturally surfaces AFTER a heal (she's then in the town, next to the wall).
        if state["map"] in CITY_MART_DOORS and self.money() > SHOP_MONEY_FLOOR:
            fs = self._walled(state)
            if self._shopping_list(foresight=fs):
                a["stock_up"] = ("stock up at the Mart — load up on potions before you push that wall"
                                 if fs else
                                 "stock up at the Mart — buy potions and the cures for what's been hurting you")
        # TALK TO THE LOCALS (Batch 5 P4): offer only when there's actually a not-yet-talked-to
        # townsperson reachable-ish here (honest action set). Her earliest want — let her work the room.
        try:
            if self._untalked_npc_here():
                a["talk_npc"] = "go say hi to someone nearby — talk to the locals, see what they know"
        except Exception:
            pass
        # ── EXPLORING IS A MOVE (Batch-WORLD Phase 4a) — going to a PLACE she knows is a first-class
        # option, not just "advance toward the gym". When she's blocked or needs something (a team,
        # levels, balls) the obvious move is to travel to a place that HAS it. These are REACHABLE
        # visited places only (routed AROUND walls — never across the gated bridge); she still chooses.
        # WALK today; once she earns Fly, towns become a faster fly-to (capability-gated, Phase 3). ──
        try:
            cur = tuple(state["map"])
            tavoid = self._wall_avoid(state)
            can_fly = self.world.has_cap("fly")
            for mid, nm, why in self.world.travel_targets(cur, avoid=tavoid)[:4]:
                verb = "fly to" if (can_fly and self.world.is_town(mid)) else "head back to"
                a[f"travel:{mid[0]},{mid[1]}"] = f"{verb} {nm} — {why}"
        except Exception as _ta:
            log(f"   [roam] travel-options skipped: {_ta}")
        # ── STRATEGIC-STUCK FLOOR (the Gary death-loop killer) — the TEETH ─────────────────────────
        # loss_awareness already TELLS her "come back stronger", but advice without teeth lost 3× live.
        # When she's strategically stuck (≥N identical losses, no roster/level change) AND a real
        # strengthen path exists here, PRUNE the wall-re-approach (head_to_gym) from the pickable set so
        # "throw the same mon at the same wall again" stops being the path of least resistance. She still
        # CHOOSES how to strengthen (grind / catch / shop) — capability-not-script. If NO strengthen
        # option is reachable, head_to_gym is KEPT (never dead-end her); the dominant ctx note + the
        # mode-side backtrack fix handle that case. Play-gated by construction (free_roam = autonomous play).
        if STRATEGIC_STUCK_ENABLED:
            try:
                pc = state.get("party_count")
                pl = state["party"][0]["level"] if state.get("party") else None
                if self.strat.strategically_stuck(pc, pl):
                    # travel:* (go BACK to known grass/a Mart to build up) IS a strengthen path — so
                    # when the only way to get stronger is to travel backward, the wall re-approach
                    # still gets pruned (Batch-WORLD: backtracking is now a real, counted option).
                    strengthen = [k for k in a if k in ("wander_catch", "battle", "stock_up")
                                  or k.startswith("travel:")]
                    if strengthen:
                        a.pop("head_to_gym", None)        # may NOT march back into the wall unchanged
                        if "battle" in a:
                            a["battle"] = ("STRENGTHEN FIRST — train in the grass to level up; you can't "
                                           "beat that wall as you are")
                        if "wander_catch" in a:
                            a["wander_catch"] = ("STRENGTHEN FIRST — catch a teammate so you're not solo "
                                                 "against that wall")
                        if "stock_up" in a:
                            a["stock_up"] = "STRENGTHEN FIRST — stock up on healing before you face them again"
                        log("   [roam] !! STRATEGIC-STUCK: pruned head_to_gym (wall re-approach) — "
                            f"strengthen options only: {strengthen}")
            except Exception as _ss:
                log(f"   [roam] strategic-stuck pruning skipped: {_ss}")
            # PHASE 2 — READINESS → GO: once she's crossed the readiness bar, make the RETURN the
            # ATTRACTIVE option (not merely un-pruned). Reframing head_to_gym's description IS the weight
            # in this oracle architecture — so the return reads as a strong positive pull, not a bland
            # equal. She still chooses; this just stops "keep grinding" from being the path of least
            # resistance after she's already prepared.
            # FORWARD-ANCHORED: once she's past the readiness bar, the move is to TRAVEL to the next gym
            # — a concrete physical destination head_to_gym routes toward via live map connections — NOT
            # to circle the grass or chase a vague 'rematch' with no coordinates (the live aimless-grind
            # bug). Reframe head_to_gym as the dominant FORWARD pull AND prune the grind actions so
            # 'murder Rattatas forever' stops being offered. Never dead-ends her: only fires when
            # head_to_gym is a real option here (a forward route exists).
            try:
                _pc2 = state.get("party_count")
                _pl2 = state["party"][0]["level"] if state.get("party") else None
                wr = self.strat.ready_to_retry(_pc2, _pl2)
                ng2 = state.get("next_gym")
                # ACE-OVERPOWER guard: ready_to_retry releases on a small (+2) level gain, but the
                # overpower plan wants a CLEAR (+9) lead before pushing — so don't release her early while
                # the ace is still grinding toward the overpower target (prep_t still set). Without this she
                # pushes at +2, loses again, and loops.
                if prep_t is not None and not battle_agent.GRIND_SWITCH_ENABLED:
                    wr = None
                # WALL RETIREMENT (2026-07-06 nursery fix): readiness crossed AND she has moved PAST the
                # wall's map — the record's whole job (grind exit + the go-take-the-wall pull) is spent.
                # Keeping it alive made READINESS→GO prune the nursery on every NEW route forever
                # (runs 8-10: wander_catch never offered on Routes 5/6 because of a conquered Cerulean
                # trainer). Retire it; a fresh loss re-records. Scoped: only when she's off the wall map.
                if wr:
                    try:
                        rec = self.strat.active_wall_rec()
                        if rec and tuple(rec.get("map_id") or ()) != tuple(state["map"]):
                            self.strat.retire_active_wall("readiness crossed + region advanced past it")
                            wr = None
                    except Exception as _wrx:
                        log(f"   [roam] wall retirement skipped: {_wrx}")
                if wr and "head_to_gym" in a and ng2:
                    try:
                        badge = self._BADGE_NAMES[state.get("badge_count", 0)]
                    except Exception:
                        badge = "next"
                    a["head_to_gym"] = (f"YOU'RE READY — STOP GRINDING AND GO. Travel to {ng2['city']} and take "
                                        f"on {ng2['leader']} for the {badge} Badge — that's the way FORWARD, the "
                                        f"route ahead leads there. You trained up for exactly this; the move now "
                                        f"is to GO to the next gym, not circle the grass.")
                    # NURSERY EXEMPTION: with a THIN team + balls in the bag, catching IS forward
                    # progress (her own game-model says so) — never prune it as "grind".
                    try:
                        keep_catch = (state.get("party_count", 6) < 4
                                      and self._balls_pocket_count(ITEM_POKE_BALL) > 0)
                    except Exception:
                        keep_catch = False
                    prune_set = ("battle",) if keep_catch else ("battle", "wander_catch")
                    pruned = [g for g in prune_set if a.pop(g, None) is not None]
                    log(f"   [roam] !! READINESS → GO (grind-exit): past the bar — head_to_gym reframed "
                        f"FORWARD to {ng2['city']}, pruned grind {pruned}"
                        + ("; wander_catch KEPT (thin team + balls — the nursery breathes)" if keep_catch
                           else "") + "; the move is to travel to the gym")
            except Exception as _rr:
                log(f"   [roam] readiness reframe skipped: {_rr}")
        # ── PROACTIVE FORWARD DRIVE (the backward-grind killer — the ROOT fix) ───────────────────────
        # The live bug: post-Misty she WANTED 'grind on the way' and it resolved to travel BACKWARD into a
        # cleared dead-end (Route 4) — because at decision time head_to_gym (the forward objective) competed
        # on EQUAL footing with the backward grind, so a grind-want picked backward and she never reached
        # the gate. Fix: when there's a forward objective she's drifting from, make head_to_gym the DOMINANT
        # pull and PRUNE the options that lead AWAY from it. Two forward signals, one response:
        #   (1) a forward-unlock QUESTLINE is open (she's AT the gate — north to Bill before Vermilion), OR
        #   (2) she's DRIFTED off the road: the learned graph can't yet route to the next-gym city (the gate
        #       isn't crossed, so those maps are unlearned) AND she's not at the base-camp city it launches
        #       from — so 'forward' means get back to base camp (then the questline takes over there).
        # Grinding stays characterful — it happens ON THE WAY (head_to_gym walks her forward through the
        # trainers/grass). Never dead-ends her (head_to_gym kept), leaves heal/stock_up/talk alone, and
        # stands down for survival (critical-heal returned early) + the strategic-stuck floor (it OWNS the
        # lost-repeatedly case and prunes head_to_gym itself). Play-gated (free_roam only).
        if FORWARD_DRIVE_ENABLED and "head_to_gym" in a and state.get("next_gym"):
            stuck = False
            if STRATEGIC_STUCK_ENABLED:
                try:
                    stuck = self.strat.strategically_stuck(
                        state.get("party_count"),
                        state["party"][0]["level"] if state.get("party") else None)
                except Exception:
                    stuck = False
            # STRATEGIC UNDERLEVEL-GRIND (Task B): stand down ALSO while she's prepping an under-levelled
            # team for the wall — the grind ISN'T "backward drift" then, it's the deliberate forward move
            # (raise the floor → cross). Pruning battle/wander_catch here would be the stubborn-charge bug.
            if not stuck and prep_t is None:
                try:
                    cur = tuple(state["map"])
                    avoid = self._wall_avoid(state)
                    gym_city = self._next_gym_city_map(state.get("next_gym"))
                    route_exists = bool(gym_city and self.world.route(cur, gym_city, avoid))
                    ql_open = (QUESTLINE_ENABLED and self._active_questline is not None
                               and self._active_questline.actionable is not None
                               and self._active_questline.derivable)
                    # base camp matters only PRE-gate (no graph route to the gym yet); on-spine progress
                    # (route exists) is left untouched — head_to_gym already routes correctly there.
                    anchor = None if route_exists else self._base_camp(state)
                    off_branch = anchor is not None and anchor != cur
                    if ql_open or off_branch:
                        if ql_open:
                            q = self._active_questline
                            a["head_to_gym"] = (
                                f"PUSH FORWARD — the way ahead is blocked ({q.gate.human}), and the move "
                                f"that opens it is to {q.actionable.human}. Go DO that now: it's the road "
                                f"FORWARD toward the next badge. Grind the trainers and grass ON THE WAY if "
                                f"you want, but keep heading to the objective — never backward into "
                                f"somewhere you've already cleared.")
                        else:
                            aname = self.world.name(anchor)
                            a["head_to_gym"] = (
                                f"PUSH FORWARD — you've drifted off the road to "
                                f"{state['next_gym']['city']}. Head back toward {aname} to pick the way "
                                f"forward up again — that's where the next-gym road continues. Grind ON THE "
                                f"WAY if you want, but stop doubling back into places you've already cleared.")
                        # prune backward: standalone grind (it happens en route) + travel targets that
                        # don't get her CLOSER to base camp (lateral/backward). On the questline branch she's
                        # already at base camp, so every travel:* is a detour off the forward errand → prune.
                        cur_d = None
                        if off_branch:
                            p = self.world.route(cur, anchor, avoid)
                            cur_d = (len(p) - 1) if p else None
                        pruned = []
                        for k in list(a):
                            if not k.startswith("travel:"):
                                continue
                            keep = False
                            if off_branch and cur_d is not None:
                                try:
                                    tx = tuple(int(v) for v in k.split(":", 1)[1].split(","))
                                    tp = self.world.route(tx, anchor, avoid)
                                    keep = bool(tp) and (len(tp) - 1) < cur_d   # strictly closer = forward
                                except Exception:
                                    keep = False
                            if not keep:
                                a.pop(k, None); pruned.append(k)
                        for g in ("battle", "wander_catch"):
                            if a.pop(g, None) is not None:
                                pruned.append(g)
                        if pruned:
                            why = "questline" if ql_open else f"off-branch->{self.world.name(anchor)}"
                            log(f"   [roam] !! FORWARD-DRIVE ({why}): head_to_gym reframed as the dominant "
                                f"forward pull, pruned backward-grind {sorted(pruned)}")
                except Exception as _fd:
                    log(f"   [roam] forward-drive skipped: {_fd}")
        # USE AN HM (PHASE 2): offer "clear the obstacle in front" ONLY when there's an adjacent
        # cuttable tree / pushable boulder AND she has the HM + badge to clear it (honest action set,
        # same principle as the rest). Capability-not-script: she still CHOOSES to use it in
        # character. Gated by FIELD_MOVES_ENABLED until the actuation passes a live control.
        if FIELD_MOVES_ENABLED:
            try:
                fo = self._field_block(state)
                if fo:
                    a[fo["action"]] = fo["prompt"]
            except Exception:
                pass
        # GRAB AN ITEM (PHASE 3): offer when a ground item ball is reachable + SAFE (overworld
        # surface, bounded detour). She chooses to grab it in character. Gated until verified.
        if ITEM_PICKUP_ENABLED:
            try:
                if self._item_target(state):
                    a["grab_item"] = "grab that item over there — a free pickup, looks safe to reach"
            except Exception:
                pass
        # PROBLEM 3 — SILENT-NO-MOVE PRUNE: once she's had a no-move streak >=2 (a movement pick that
        # didn't move her, even ALTERNATING between several), remove EVERY dead route from this tick's
        # set so she can't thrash among them — she MUST take an action that actually does something
        # (grind in reachable grass, a different known destination, heal/talk). Never dead-end her:
        # only prune while at least one non-dead option remains. Reset by any real movement.
        if self._nomove_streak >= 2:
            removable = [k for k in self._dead_moves if k in a]
            if removable and (len(a) - len(removable)) >= 1:
                for k in removable:
                    a.pop(k, None)
                log(f"   [roam] !! NO-MOVE PRUNE: dead routes {sorted(removable)} haven't moved her "
                    f"(streak {self._nomove_streak}) — removed this tick so she picks something that "
                    f"works; remaining: {sorted(a)}")
            else:
                log(f"   [roam] !! NO-MOVE: dead routes {sorted(removable)} but they're the only options "
                    f"left — NOT pruning (won't dead-end her); strategic-stuck/macro recovery will carry it")
        return a

    def _item_target(self, state):
        """Nearest SAFE, reachable ground item ball: {'coord','stand','face','dist'} or None. SAFE =
        overworld surface (map group 3 — never a cave/interior, so a grab can't disrupt a fragile
        traversal like Mt Moon) + grass-free-BFS reachable within ITEM_GRAB_MAX_DIST. PHASE 3."""
        import field_moves as fm
        if tv.map_id(self.b)[0] != 3:                 # overworld surface only (skips caves/buildings)
            return None
        co = state.get("coords") or tv.coords(self.b)
        if not co:
            return None
        balls = fm.item_balls(self.b)
        if not balls:
            return None
        grid = tv.Grid(self.b)
        best = None
        for ball in balls:
            ix, iy = ball["coord"]
            for d, key in fm._DELTA_KEY.items():
                sx, sy = ix - d[0], iy - d[1]         # tile to stand on so facing `key` looks at the item
                if not grid.walkable_safe(sx, sy):
                    continue
                p = tv.bfs(grid, co, lambda c, t=(sx, sy): c == t, walkable=grid.walkable_safe)
                if p and (len(p) - 1) <= ITEM_GRAB_MAX_DIST:
                    dist = len(p) - 1
                    if best is None or dist < best["dist"]:
                        best = {"coord": ball["coord"], "stand": (sx, sy), "face": key, "dist": dist}
        return best

    def _field_block(self, state):
        """Detect an adjacent obstacle the player could clear with an HM she actually has. Returns
        {'action','prompt','hm','face'} or None. Pure RAM reads (field_moves). PHASE 2."""
        import field_moves as fm
        co = state.get("coords") or tv.coords(self.b)
        pc = state.get("party_count") or self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for ob in fm.obstacles_adjacent(self.b, co):
            if fm.can_use(self.b, ob["hm"], pc):
                name = fm.HM[ob["hm"]][2]
                return {"action": f"use_{ob['hm']}", "hm": ob["hm"], "face": ob["face"],
                        "prompt": f"use {name} to clear the {ob['kind']} blocking the way forward"}
        # Surf: an adjacent surfable-water edge + Surf + badge → offer to cross.
        try:
            grid = tv.Grid(self.b)
            face = fm.surf_edge_adjacent(self.b, grid, co)
            if face and fm.can_use(self.b, "surf", pc):
                return {"action": "use_surf", "hm": "surf", "face": face,
                        "prompt": "use Surf to cross the water ahead"}
        except Exception:
            pass
        return None

    # facing nibble (FRLG object-event facingDirection) -> the (dx,dy) of the tile she's FACING =
    # the blocker she's bumping. 1=south/down 2=north/up 3=west/left 4=east/right.
    _FACE_DELTA = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}

    def feed_watchdog(self, text="", now=None):
        """LAYER B — fed every live frame (throttled) from play_live's render hook. Reads the world
        fingerprint + the on-screen dialogue text and feeds the wall-clock StuckWatch; on a TRIP it
        latches `_stuck_request` (honored at the top of the roam loop + cooperatively by travel's
        stuck_check). No-op headless (no watchdog created) or before free_roam. Pure backstop — it
        never drives a normal decision, only catches a wedge no sub-layer surfaced."""
        if self._stuckwatch is None or now is None:
            return
        try:
            fp = wf.fingerprint(self.b)
        except Exception:
            return
        if self._stuckwatch.feed(fp, now, text=text or "") and self._stuck_request is None:
            self._stuck_request = {
                "reason": self._stuckwatch.reason,
                "map": tuple(tv.map_id(self.b)),
                "coords": tv.coords(self.b),
                "facing": (fp.facing if fp else None),
                "secs": round(self._stuckwatch.seconds_stuck(now), 1),
            }
            log(f"   [roam] !!!! WATCHDOG TRIPPED ({self._stuck_request['reason']}): nothing on screen "
                f"has meaningfully changed for {self._stuck_request['secs']}s at "
                f"{self._stuck_request['map']}@{self._stuck_request['coords']} — latching a top-level "
                f"disengage (catches a sub-layer wedge the per-tick ledger can't see). LOUD")

    # opposite of facing -> the press that WALKS AWAY from the NPC she's facing (1=down..4=right).
    _FACE_AWAY = {1: "UP", 2: "DOWN", 3: "RIGHT", 4: "LEFT"}

    def _disengage_overworld_npc(self, req):
        """LAYER B recovery EXECUTION (the real escape, not narration). She's wedged mashing A in a
        LOOPING overworld NPC's dialogue — and A only RE-LOOPS it, it can never end. The human escape
        is: press B to DISMISS the box, then STEP AWAY. This performs exactly that, then marks the NPC
        sticky-blocked so travel/oracle don't route her straight back into talking to it.
        Returns True once the box is closed."""
        from dialogue_drive import box_open
        self.b.set_input_owner("agent")
        # 1) CLOSE the box with B (NEVER A — A re-triggers the loop). Tap + settle until it's gone.
        closed = not box_open(self.b)
        for _ in range(6):
            if not box_open(self.b):
                closed = True
                break
            self.b.press("B", 6, 10, self.render, owner="agent")
            for _ in range(10):
                self.b.run_frame(); self.render()
        closed = closed or not box_open(self.b)
        log(f"   [roam] disengage: pressed B to DISMISS the looping box (never A) -> closed={closed}")
        # 2) mark the NPC body sticky-blocked FIRST (so even if the step fails, we won't re-route in)
        self._mark_wedge_spot(req)
        # 3) STEP AWAY: walk OPPOSITE the way she's facing the NPC; fall back to any open direction.
        cur0 = tv.coords(self.b)
        away = self._FACE_AWAY.get(req.get("facing"))
        order = ([away] if away else []) + ["DOWN", "UP", "LEFT", "RIGHT"]
        for d in order:
            for _ in range(2):
                self.b.press(d, 8, 8, self.render, owner="agent")
                for _ in range(6):
                    self.b.run_frame(); self.render()
            if tv.coords(self.b) != cur0:
                log(f"   [roam] disengage: stepped {d} AWAY from the NPC -> {tv.coords(self.b)} "
                    f"(closed box + walked off — the real escape)")
                return closed
        log("   [roam] disengage: box closed but couldn't step away yet — oracle re-routes next tick "
            "(NPC is sticky-blocked so it won't path back in)")
        return closed

    def _mark_wedge_spot(self, req):
        """LAYER A+B bridge: when the watchdog trips, the tile she's FACING is whatever she's been
        bumping (a blocking NPC / a closed door she keeps walking into). Record it in the UNIFIED
        block memory so travel routes AROUND it and the talk path won't re-engage it next time."""
        try:
            mp, cur, f = req.get("map"), req.get("coords"), req.get("facing")
            if not mp or cur is None:
                return
            d = self._FACE_DELTA.get(f)
            if d is None:
                return
            body = (cur[0] + d[0], cur[1] + d[1])
            self._blocked_npcs.add((mp, body))
            self._looped_spots.add((mp, cur))   # talk guard: don't re-initiate from this stand tile
            log(f"   [roam] wedge spot: blocking {body} on {mp} (route around it) + marking {cur} a "
                f"resolved talk-spot — unified block memory now {len(self._blocked_npcs)} tile(s)")
        except Exception as _e:
            log(f"   [roam] mark wedge spot skipped: {_e}")

    def _wait_overworld(self, max_frames=900):
        """Settle to overworld-idle (not in battle, no open dialogue box) before reading state / acting."""
        from dialogue_drive import box_open
        for _ in range(max_frames):
            if not st.in_battle(self.b) and not box_open(self.b):
                return True
            self.b.run_frame(); self.render()
        log("   [roam] !! _wait_overworld TIMEOUT — proceeding (state may be mid-transition)")
        return False

    def _next_gym_city_map(self, ng):
        """Map id (g,n) of the next gym's CITY (from the fixed spine), or None. Lets head_to_gym route
        toward a CONCRETE destination through the warp-aware graph, not just a blind south edge."""
        if not ng:
            return None
        try:
            from pokemon_world import GYM_SPINE
            for city, mid in GYM_SPINE:
                if city == ng.get("city"):
                    return tuple(mid)
        except Exception:
            pass
        return None

    def _base_camp(self, state):
        """The spine CITY the next-gym push launches from — the GYM_SPINE predecessor of the next gym (e.g.
        Cerulean for the Vermilion push). This is what 'forward' means when the learned graph can't yet
        route to the next-gym city (she hasn't crossed the gate, so those maps are unlearned): get back to
        base camp, then the gate-unlock questline takes over there. Map id (g,n) or None."""
        ng = state.get("next_gym")
        try:
            from pokemon_world import GYM_SPINE
            idx = next((i for i, (c, _m) in enumerate(GYM_SPINE) if c == (ng or {}).get("city")), None)
            if idx is not None and idx > 0:
                return tuple(GYM_SPINE[idx - 1][1])
        except Exception:
            pass
        return None

    def _route_action(self, pick, state):
        """Route an oracle ACTION pick to its wired handler; return the handler's outcome string."""
        # HUD now-state label (BATTLE is detected live via in_battle() at publish time).
        self._now_state = {"heal": "HEALING", "stock_up": "SHOPPING"}.get(pick, "EXPLORING")
        if pick == "heal":
            return self.heal_nearest()
        # USE AN HM (PHASE 2): she chose to clear the obstacle in front. Re-detect (state may have
        # shifted), then actuate via the field-move actuator. Returns 'used'/'no_prompt'/'cant'/'stuck'
        # → all non-fatal: a flake surfaces to the oracle next tick, never a freeze.
        if pick.startswith("use_") and FIELD_MOVES_ENABLED:
            hm = pick[4:]
            if self.field is None:
                return "cant"
            fo = self._field_block(state)
            face = fo["face"] if fo else None
            if face is None:
                return "no_prompt"
            nm = __import__("field_moves").HM.get(hm, (None, None, hm))[2]
            self.on_event(f"there's something blocking the way — let me use {nm}.", kind="field", tier=2)
            res = self.field.clear_obstacle(hm, face)
            log(f"   [roam] field-move use_{hm} (face {face}) -> {res}")
            return res
        # GRAB AN ITEM (PHASE 3): walk to the stand-tile beside the item, face it, A, drain the
        # 'found ITEM!' line. Non-fatal on miss (surfaces to the oracle next tick).
        if pick == "grab_item" and ITEM_PICKUP_ENABLED:
            if self.field is None:
                return "cant"
            it = self._item_target(state)
            if not it:
                return "no_item"
            self.on_event("ooh, there's something over there — let me grab it.", kind="item", tier=1)
            self.trav.travel(target_map=None, arrive_coord=it["stand"], max_steps=200, max_seconds=90)
            if tv.coords(self.b) != it["stand"]:
                log(f"   [roam] grab_item: couldn't reach stand tile {it['stand']} -> stuck")
                return "stuck"
            res = self.field.grab_item(it["coord"], it["face"])
            log(f"   [roam] grab_item at {it['coord']} (face {it['face']}) -> {res}")
            return res
        pcount = state.get("party_count")
        plevel = state["party"][0]["level"] if state.get("party") else None
        if pick.startswith("travel:"):
            return self._travel_to_known(pick, state)
        if pick == "head_to_gym":
            # GATE-UNLOCK: if a story/HM gate is walling the way forward, advancing the quest MEANS doing
            # the unlock errand — so while a questline is active, head_to_gym drives THAT (e.g. north to
            # Bill for the S.S. Ticket) instead of the gated wall. Self-clears the instant the success flag
            # reads satisfied LIVE, then normal gym-routing resumes.
            if QUESTLINE_ENABLED and self._active_questline is not None:
                return self._run_questline_step(state)
            # BATCH 6 PHASE 1 — SHE ACTUALLY CLIMBS. The loop's whole point: when she's AT the next gym's
            # city, don't just mill around — ENTER the gym, clear its junior trainers, beat the leader,
            # earn the badge, advance to the next base camp. beat_gym is the general, data-driven handler
            # (reserve -> enter -> juniors -> leader -> award -> badge); a loss propagates as battle_loss
            # (free_roam's blackout-recovery retries), a nav fail as 'stuck' (surfaces to the oracle —
            # never a freeze). She still CHOSE head_to_gym this tick (capability-not-script).
            ng = state.get("next_gym")
            gym = GYMS.get(ng["leader"]) if ng else None
            if gym is not None and state["map"] == gym.city:
                log(f"   [roam] ⛰️  AT {ng['leader']}'s city {gym.city} — ENTERING the gym to take the badge")
                # PHASE 6 — DREAD: a gym is a hard challenge; name the nerves BEFORE the fight so the
                # payoff has something to land against (earned relief requires prior worry). Tier-2
                # anticipation; her core voice colors the actual nerves. Sharper if she's walled here.
                walled = self._walled(state)
                if walled:
                    self.on_event(f"okay. {ng['leader']}'s gym — and this is the wall that's been beating me. "
                                  f"deep breath. this is the one. here we go.", kind="gym", tier=2)
                else:
                    self.on_event(f"alright, {ng['leader']}'s gym. this is it — I'm actually a little nervous. "
                                  f"okay. let's climb.", kind="gym", tier=2)
                out = self.beat_gym(ng["leader"])
                if out == "badge":
                    # PHASE 6 — CATHARSIS: reference the worry so the relief is EARNED, not "oh great,
                    # moving on". Tier-3 big beat → her core deep-reaction path RISES; the saga promotes it.
                    nb = state.get("badge_count", 0) + 1
                    self.on_event(f"YES — we DID it! {ng['leader']} is DOWN. I was genuinely nervous about that "
                                  f"one and we pulled it off — badge number {nb}. okay. okay! onward.",
                                  kind="gym", tier=3)
                elif out in ("battle_loss", "loss", "blackout"):
                    # PHASE 6 — the down-beat of the arc: a gym loss STINGS (and feeds the wall the next
                    # dread references). Tier-2 so it's felt, not swallowed.
                    self.on_event(f"no — no, we lost it. {ng['leader']} got me. okay… that one hurts. "
                                  f"I need to come back stronger.", kind="gym", tier=2)
                return out
            # WARP-AWARE forward routing toward the next gym CITY: route THROUGH warps/dungeons (the
            # Underground-Path class), not just map edges. Uses the world-model graph (live-learned warps
            # + the seeded spine). Returns None when the graph doesn't yet reach the gym -> fall through to
            # live south-edge DISCOVERY below (she explores forward, learning each map's warps, which fills
            # the graph). One hop per tick — she can still divert (true free roam, discovery preserved).
            cur_map = tuple(state["map"])
            target_city = self._next_gym_city_map(ng)
            if target_city and target_city != cur_map:
                try:
                    step = self.world.next_step(cur_map, target_city, avoid=self._wall_avoid(state))
                except Exception as _ws:
                    step = None
                    log(f"   [roam] warp-route step skipped: {_ws}")
                if step:
                    nxt_map, kind, detail = step
                    if self.strat.is_gated(nxt_map, pcount, plevel):
                        self.on_event(self.strat.wall_gate_note("reach the next gym"), kind="wall", tier=2)
                        log(f"   [roam] !! WALL-GATED: warp-route next hop {nxt_map} is the wall map")
                        return "wall_gated"
                    if kind == "warp":
                        log(f"   [roam] WARP-ROUTE toward {ng['city']}: {cur_map} -> {nxt_map} via warp tile {detail}")
                        self.on_event("there's a passage here heading the right way — taking it.",
                                      kind="route", tier=1)
                        before = tv.map_id(self.b)
                        self.trav.travel(target_map=None, arrive_coord=detail, max_steps=300)
                        if tv.map_id(self.b) != before:
                            return "warped"                       # step-on warp fired on arrival
                        self.enter_warp(pick=detail)              # door warp: stand beside + step in
                        return "warped" if tv.map_id(self.b) != before else "warp_failed"
                    log(f"   [roam] EDGE-ROUTE toward {ng['city']}: {cur_map} -> {nxt_map} (edge {detail})")
                    return self._edge_travel(nxt_map, detail)
            # FORWARD-SPINE RECOVERY (off-branch): we're here because the learned graph can't yet route to
            # the gym CITY (she hasn't crossed the gate, so those maps are unlearned). Do NOT blindly walk
            # the local 'south' edge below — on a side-branch (e.g. Route 4, whose 'south' is Route 3, even
            # FURTHER from the spine) that walks her BACKWARD. Instead route toward the base-camp city the
            # next-gym push launches from (Cerulean for Vermilion); once she's there the proactive questline
            # takes over (north to Bill). Only when she's actually off it — at base camp this no-ops and the
            # south block runs (south from Cerulean = the Slowbro gate → questline). Edges only (no coords).
            base = self._base_camp(state)
            if base is not None and base != cur_map:
                # FRONTIER-MARCH (2026-07-06, the Route-5 ping-pong fix): if base camp is the NEIGHBOR
                # we just came from, we are ON the forward road PAST it — bouncing "back to base camp"
                # while base camp says "go forward" ping-pongs two maps forever (run 5: Cerulean <->
                # Route 5 ×14). March AWAY from base instead: the opposite connection first, else any
                # non-base connection (a bend), else a DOOR pass-through (gatehouse/tunnel country).
                _dirword = {"N": "north", "S": "south", "E": "east", "W": "west"}
                _opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
                try:
                    conns = list(self._map_connections())
                except Exception:
                    conns = []
                back_dir = next((d for d, m in conns if tuple(m) == tuple(base)), None)
                if back_dir is not None:
                    fwd = next(((d, m) for d, m in conns if d == _opp[back_dir]), None) \
                          or next(((d, m) for d, m in conns if tuple(m) != tuple(base)), None)
                    if fwd is not None:
                        fdir, fmap = fwd
                        log(f"   [roam] FORWARD-FRONTIER: past base camp {self.world.name(base)} — "
                            f"marching {_dirword[fdir]} to {tuple(fmap)} (never bounce back to base)")
                        self.on_event("this is new ground — the road to the next gym runs on ahead. "
                                      "keeping on.", kind="route", tier=1)
                        return self._edge_travel(tuple(fmap), _dirword[fdir])
                    pt = self._door_passthrough()
                    if pt == "crossed":
                        return "passthrough_forward"
                try:
                    hop = self.world.next_hop(cur_map, base, avoid=self._wall_avoid(state))
                except Exception:
                    hop = None
                if hop:
                    back_map, edge_dir = hop
                    bname = self.world.name(base)
                    log(f"   [roam] FORWARD-SPINE: no graph route to {ng['city']} yet — heading toward base "
                        f"camp {bname} via {edge_dir} to {back_map} (then the gate questline takes over)")
                    self.on_event(f"I've drifted off the road — heading back toward {bname} to pick the way "
                                  f"forward back up.", kind="route", tier=1)
                    return self._edge_travel(back_map, edge_dir)
            # not at the gym city yet -> step toward it via the SOUTH connection, one map per tick (she can
            # still change her mind next tick — true free roam).
            south = next(((g, n) for d, (g, n) in self._map_connections() if d == "S"), None)
            if south is None:
                # OFF-SPINE RECOVERY (the forward-spine FLOOR): no forward (south) connection from here —
                # she's wandered OFF the main path (e.g. west to Route 4 to grind), so head_to_gym used to
                # dead-end on no_gym_route and she'd just keep grinding. Instead, route her BACK toward the
                # nearest KNOWN town on the spine (learned edges only — no fabricated coords) so the next
                # tick head_to_gym can resume heading south. This keeps her ALWAYS pointed at the next gym.
                cur_map = tuple(state["map"])
                try:
                    avoid = self._wall_avoid(state)
                    for tgt, nm, _h in self.world.reachable_with_trait(cur_map, "is_town", avoid=avoid):
                        hop = self.world.next_hop(cur_map, tgt, avoid=avoid)
                        if hop:
                            back_map, edge_dir = hop
                            log(f"   [roam] OFF-SPINE: no south route from {cur_map} — heading back toward "
                                f"{nm} (the spine) via {edge_dir} to {back_map}, then south to the gym")
                            self.on_event(f"I've drifted off the main path — heading back toward {nm} to "
                                          f"pick the road to the next gym back up.", kind="route", tier=1)
                            return self.trav.travel(target_map=back_map, edge=edge_dir)
                except Exception as _os:
                    log(f"   [roam] off-spine recovery skipped: {_os}")
                return "no_gym_route"
            # BATCH-4 PHASE 2 spatial wall: don't route her straight back into the trainer who keeps
            # blacking her out. If the next map is the gated wall map and she's no stronger, ABORT LOUD
            # + tell the oracle the way is gated — she picks a different action (train/heal here first).
            if self.strat.is_gated(south, pcount, plevel):
                self.on_event(self.strat.wall_gate_note("reach the next gym"), kind="wall", tier=2)
                log(f"   [roam] !! WALL-GATED: head_to_gym would route through {south} (the wall map) — "
                    f"NOT feeding her back into the wall; surfacing to the oracle")
                return "wall_gated"
            # GATE-UNLOCK (proactive): is the forward (south) exit a STORY/HM gate she can't pass yet —
            # the Cerulean Slowbro / S.S.-Ticket story-block (read LIVE; a satisfied flag = open road)? If
            # so, recognise it, derive the unlock questline, and pursue THAT (the errand) instead of
            # walking into the wall. She learns where to go (capability-not-script), narrates it, executes.
            if QUESTLINE_ENABLED:
                gate = self._gate_recognizer.recognize(cur_map, blocked_dir="south")
                if gate and self._open_questline(gate, state):
                    return self._run_questline_step(state)
            return self._edge_travel(south, "south")
        if pick in ("wander_catch", "battle"):
            # PHASE 3B: before any free-roam leveling, make sure the lead has a free move slot so a
            # level-up move auto-learns (the un-actuatable 'Delete a move?' box never appears). Her call
            # which move goes (soul oracle, safe-set only); no-op if she already has room.
            self._ensure_move_room()
            gt = self._grass_target(state)
            if gt is not None and gt[0] == "route":
                _, tgt, edge = gt
                # spatial wall: the grass she wants is across the gated bridge -> don't re-cross into the
                # wall. (_grass_target already PREFERS a non-gated route if one exists — route AROUND;
                # this catches the case where the gated route is the ONLY grass -> surface it.)
                if self.strat.is_gated(tgt, pcount, plevel):
                    self.on_event(self.strat.wall_gate_note("get to the grass"), kind="wall", tier=2)
                    log(f"   [roam] !! WALL-GATED: grass route {tgt} is the wall map — NOT crossing the "
                        f"bridge into the wall; surfacing to the oracle")
                    return "wall_gated"
                log(f"   [roam] no grass underfoot -> routing to grass route {tgt} ({edge}) first")
                r = self.trav.travel(target_map=tgt, edge=edge)
                if r != "arrived":
                    return f"to_grass:{r}"
            if pick == "wander_catch":
                return self.catch_one()
            # STRATEGIC UNDERLEVEL-GRIND (Task B): if the TEAM FLOOR is under the wall's level, field the
            # WEAK members (not the ace) and level THEM to readiness — the real fix for "the display said
            # 'train the team' but the action trained the ace". Else the ordinary lead bump.
            t = self._prep_team_target(state)
            if t is not None:
                return self.grind_weak_members(t)
            lead = state["party"][0]["level"] if state["party"] else 5
            return self.grind(lead + 2)
        if pick == "stock_up":
            door = CITY_MART_DOORS.get(state["map"])
            sl = self._shopping_list(foresight=self._walled(state))
            if door is None or not sl:
                return "nothing_to_buy"
            bought = self.buy_at_mart(door, sl)
            return "stocked" if bought else "shop_failed"
        if pick == "talk_npc":
            return self.talk_npc()
        return "unknown_action"

    def _boot_state_sanity(self):
        """DIAGNOSTIC (increment 4 PART C): at boot, SCREAM if the loaded SAVE is suspect, so a bad test
        state can't silently doom a run (the live viridian_parcel_done / seg_cascade_badge 4/60-HP case —
        she boots low, loses the first wild, blacks out, dead-ends). Two checks: (1) party critically low
        / fully fainted; (2) spawn has no overworld route to a real objective. Logs LOUD and changes
        NOTHING — it tells Jonny the SAVE is the problem, not the agent. Returns the warnings (empty=sane)."""
        susp = []
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        lows, alive = [], 0
        for s in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            hp, mx = self.b.rd16(base + P_HP), self.b.rd16(base + P_MAXHP)
            if mx <= 0:
                continue
            if hp > 0:
                alive += 1
            if hp == 0:
                lows.append(f"slot{s} FAINTED (0/{mx})")
            elif hp < 0.25 * mx:
                lows.append(f"slot{s} {hp}/{mx} ({100 * hp // mx}%)")
        if cnt > 0 and alive == 0:
            susp.append("WHOLE PARTY FAINTED — she'd black out on the first encounter")
        elif lows:
            susp.append("party critically low HP: " + ", ".join(lows) + " — likely to blackout soon")
        m = tv.map_id(self.b)
        if m[0] != 3:                                  # overworld is group 3; a building has no route out
            susp.append(f"spawn is INSIDE a building {m}@{tv.coords(self.b)} (map group != 3) — no "
                        f"overworld route; head_to_gym would dead-end")
        else:
            south = next(((g, n) for d, (g, n) in self._map_connections() if d == "S"), None)
            try:
                grass = self._reachable_grass()
            except Exception:
                grass = None
            if south is None and grass is None:
                susp.append(f"from {m}@{tv.coords(self.b)} there is NO south gym-route connection AND no "
                            f"reachable grass — head_to_gym would return no_gym_route (a dead-end spawn)")
        if susp:
            log("   [roam] !! BOOT STATE SUSPECT — the loaded SAVE may be the problem, not the agent:")
            for s in susp:
                log(f"   [roam] !!     - {s}")
            log("   [roam] !! (diagnostic only — proceeding; PART A/B recovery engages if it bites)")
        else:
            log("   [roam] boot state sane: party has usable HP and a real objective route exists")
        return susp

    def free_roam(self, max_ticks=12, max_seconds=900, want_every=3):
        """She's loose. Each tick: settle to idle -> read LIVE state -> compute HONEST available actions
        -> the soul ORACLE picks (HER choice via _soul_choose; validated in-set, dead/no-op unpickable)
        -> route to the wired handler -> soul outcome. surface_want fires on a cadence (state-aware now).
        BEHAVIORAL CONTROL printed every tick: STATE IN -> OPTIONS -> PICK -> RESULT -> want. Loops to a
        tick/time budget. Oracle unwired (headless/no bot) -> she idles (logged), never scripted."""
        import time as _t
        t0 = _t.time()
        ledger = wf.ProgressLedger()               # MACRO watchdog: cross-tick "am I getting anywhere?"
        # LAYER B — UNIVERSAL WALL-CLOCK WATCHDOG: the backstop ABOVE every sub-layer. Fed live from
        # play_live's per-frame render hook (so it catches SUB-TICK spins), it does NOT skip dialogue
        # boxes — it trips when the screen (world fp + on-screen text) sits frozen for WATCHDOG_STUCK_S
        # wall-clock seconds, the exact hole that let the Slowbro frozen-box wedge defeat the per-tick
        # ledger. Env POKEMON_WATCHDOG=0 disables (RAM ledger still runs).
        if os.getenv("POKEMON_WATCHDOG", "1") != "0":
            self._stuckwatch = wf.StuckWatch()
            log(f"   [roam] universal watchdog ARMED (trips after {self._stuckwatch.stuck_s:.0f}s "
                f"frozen-screen, sub-layer-agnostic)")
        self._stuck_request = None
        self._watchdog_trips = 0
        red_ticks = 0                              # consecutive RED ticks (step-3 hard-recovery counter)
        hard_recovered = False                     # forced one position-break this RED streak already?
        escape_reloads = 0                         # ADDENDUM A: escape-hatch reloads this wedge-episode
        run_start_ts = t0                          # PHASE 7: cockpit uptime
        last_badge_ts = None                       # PHASE 7: time-since-last-badge (None until one lands this run)
        _prev_badges = sum(1 for i in range(8) if self.has_badge(0x820 + i))
        log("==== FREE ROAM: she's loose — every move from here is HER call ====")
        self._continuity_load()                    # ADDENDUM D: resume KNOWING her saga (bonds/wants/grudge)
        # Batch-WORLD — seed her mental map by badge-proof (places she MUST have crossed) and read her
        # current travel capabilities, so from tick 1 she has a sense of place + knows she can walk back.
        try:
            self.world.seed_known(sum(1 for i in range(8) if self.has_badge(0x820 + i)))
            self._refresh_world_caps()
        except Exception as _ws:
            log(f"   [world] seed/caps skipped: {_ws}")
        self._boot_state_sanity()                  # PART C: scream NOW if the loaded save is suspect
        # BATCH 5 PHASE 1 — CAMPAIGN ANCHOR: bank her living save periodically + the moment she makes
        # real progress (a badge, a new area, a catch), so the next GO resumes the CLIMB from where she
        # actually is. _camp_sig is the cheap progress fingerprint we diff each tick.
        def _camp_sig():
            return (sum(1 for i in range(8) if self.has_badge(0x820 + i)),
                    tv.map_id(self.b), self.b.rd8(ram.GPLAYER_PARTY_CNT))
        last_camp_sig = _camp_sig()
        self._save_campaign("roam_start")          # anchor the moment she's loose (resume point exists immediately)
        self._continuity_save()                    # ADDENDUM D: bank her saga next to the position anchor
        for tick in range(1, max_ticks + 1):
            if _t.time() - t0 > max_seconds:
                log(f"   [roam] time budget {max_seconds}s reached — ending"); break
            # CAMPAIGN ANCHOR (Batch 5 P1) — checked at the TOP of every tick so it's NEVER skipped by a
            # mid-tick `continue` (idle / hard-recovery / blackout): re-anchor the instant the prior tick
            # made REAL progress (badge / new area / catch), plus a periodic heartbeat floor. Reads RAM
            # directly so it reflects whatever the last action accomplished.
            sig = _camp_sig()
            if sig != last_camp_sig:
                reason = ("badge" if sig[0] != last_camp_sig[0] else
                          "new_area" if sig[1] != last_camp_sig[1] else "catch")
                last_camp_sig = sig
                self._save_campaign(reason)
                self._continuity_save()            # ADDENDUM D: a real gain -> bank her saga too
            elif tick > 1 and (tick - 1) % CAMPAIGN_SAVE_EVERY == 0:
                self._save_campaign(f"tick{tick - 1}")
                self._continuity_save()
            self._wait_overworld()
            # BLACKOUT / STRANDED-IN-BUILDING RECOVERY (increment 4 PART A): a wild loss whites her out
            # and warps her INSIDE a Pokémon Center (map group != 3 — a building interior), healed. Her
            # overworld actions (head_to_gym routes via map CONNECTIONS) can't navigate out of a building,
            # so she'd sit on no_gym_route forever (the live (7,4)@(5,4) dead-end). Detect the building
            # (group != 3) and EXIT to the overworld so a real objective can re-establish from the Center
            # (a known-good anchor) — never leave her parked where nothing can succeed. The faint itself
            # is felt via _soul_after_objective(battle_loss); this is the explicit "I came to" beat.
            if tv.map_id(self.b)[0] != 3 and not getattr(self, "_ql_inside_target", False):
                # (…unless she's INSIDE a questline-target building on purpose — `_ql_inside_target`. The
                # destination-interaction layer entered Bill's cottage to talk him; don't eject her before
                # she does. It clears the marker itself on a wrong building / when the questline completes.)
                log(f"   [roam] !! BLACKOUT/STRANDED: in a building interior {tv.map_id(self.b)}"
                    f"@{tv.coords(self.b)} — exiting to the overworld to re-orient")
                # WHITEOUT-BACKSTOP (swallow-proof wall record): the whiteout is irrefutable proof she
                # LOST — but the live strat_memory showed losses={}/active_wall=null after a 3× loop, i.e.
                # the battle-end read SWALLOWED the outcome and the wall never recorded (starving the
                # spatial gate AND the strategic-stuck floor). If the in-battle path didn't already record
                # this loss, force-record it now from the whiteout so active_wall is set and the floor can
                # finally key on it. Guarded by _loss_recorded_this_battle so a caught loss isn't doubled.
                if not getattr(self, "_loss_recorded_this_battle", False):
                    try:
                        self.strat.note_blackout()
                    except Exception as _bk:
                        log(f"   [strat] whiteout-backstop record skipped: {_bk}")
                self._loss_recorded_this_battle = True   # consumed — don't re-record across interior ticks
                self.on_event("ugh… I blacked out and came to back at the Pokémon Center. okay — regroup.",
                              kind="blackout", tier=2)
                self._exit_to_overworld()
                self._wait_overworld()
            state = self.read_live_state()
            self._learn_map(state)         # BATCH-6 P2: fold this map's connections+grass into the graph
            self._refresh_world_caps()     # Batch-WORLD: keep Fly/Surf/etc. live (she uses them when earned)
            # SHOP-WITH-INTENT memory (PART C): sample what's afflicting the party THIS tick so a later
            # stock-up buys the cure for what actually hurt her (persists even after it's healed off).
            newly = self.party_statuses()
            if newly - self._afflict_seen:
                log(f"   [roam] afflicted by {sorted(newly - self._afflict_seen)} — remembering for a cure run")
            self._afflict_seen |= newly
            # GATE-UNLOCK (PROACTIVE forward drive — ROOT FIX for the backward-grind): recognise the gate
            # on the FORWARD road to the next gym and OPEN/refresh the unlock questline BEFORE the action
            # set is built, so the errand is a dominant forward pull THIS tick instead of only firing
            # reactively when she physically bonks the wall. Without it she could grind backward forever
            # (west to Route 4) and never trigger the gate. Self-clears the instant the flag reads satisfied.
            self._ensure_forward_questline(state)
            avail = self._available_actions(state)
            party_str = ", ".join(f"{m['species']} L{m['level']}" for m in state["party"]) or "(none)"
            # MACRO PROGRESS LEDGER (increment 3): fingerprint THIS tick + escalate if the world has
            # sat unchanged across her last actions. GREEN=progressing / YELLOW=not getting anywhere /
            # RED=stuck despite retries. Context-aware (box up -> static is expected, not escalated).
            fp = wf.fingerprint(self.b)
            macro = ledger.observe(fp)
            self._roam_progress = macro            # surfaced for the dashboard light to read later
            # ADDENDUM A — capture a KNOWN-GOOD snapshot on every PROGRESSING tick: this is the state the
            # escape-hatch rewinds to, so by construction it's from BEFORE any wedge and already carries
            # the latest gains. Cheap (one savestate held in memory); overwritten as she progresses.
            # STRAND-POISON GUARD (2026-07-05 strike): ONLY bank from a Center-reachable (heal-safe) spot.
            # A GREEN grind tick standing in an un-healable one-way pocket (Route-4 (84,15)) must NEVER
            # become the escape target — that was THE bug: the escape-hatch reloaded straight back into
            # the strand, looping forever (observed infinite STALL). A poisoned checkpoint is worse than
            # none. Both recent-good AND the gain-seam ring are guarded.
            if macro == ledger.GREEN and not self._center_reachable_here():
                log("   [roam] known-good snapshot SKIPPED — this spot can't reach a Center; refusing to "
                    "bank a poisoned escape target (would reload back into the strand)")
            elif macro == ledger.GREEN:
                try:
                    self._last_good_state = self.b.save_state()
                    self._last_good_gain = self._gain_sig()
                except Exception as _e:
                    log(f"   [roam] last-good snapshot skipped: {_e}")
                # PHASE 1 — bank a deep-wedge fallback into the ring at a GAIN SEAM (a badge / a new
                # teammate / a fresh catch since the last ring entry). Gain seams are guaranteed-clear
                # states well clear of any wedge lip, so reverting to one ALWAYS escapes a structural
                # wedge (unlike the recent-good, which can sit right at it). De-duped on the gain-sig.
                try:
                    g = self._last_good_gain
                    if g is not None and g != self._ring_last_gain and (
                            self._ring_last_gain is None or any(c > p for c, p in zip(g, self._ring_last_gain))):
                        self._safe_ring.append({"state": self._last_good_state, "gain": g,
                                                "label": f"gain{g}@{state['place']}"})
                        self._ring_last_gain = g
                        log(f"   [roam] deep-wedge ring: banked safe checkpoint {g} @ {state['place']} "
                            f"(ring={len(self._safe_ring)}/{SAFE_RING_N})")
                except Exception as _re:
                    log(f"   [roam] deep-wedge ring bank skipped: {_re}")
            log(f"-- ROAM TICK {tick}/{max_ticks} --")
            log(f"   [roam] STATE IN: {state['place']} {state['coords']} | badges={state['badge_count']} "
                f"({', '.join(state['badges']) or 'none'}) | party=[{party_str}] | {state['progress']}")
            log(f"   [roam] PROGRESS: {macro} (unchanged {ledger.stuck} ticks) | {wf.brief(fp)}")
            # PHASE 7 cockpit: stamp the time a badge lands, then publish the health snapshot this tick.
            if state.get("badge_count", 0) > _prev_badges:
                last_badge_ts = time.time()
                _prev_badges = state["badge_count"]
            self._publish_health(macro, state, last_badge_ts, run_start_ts)
            # LAYER B — UNIVERSAL WATCHDOG DISENGAGE (highest altitude, runs BEFORE the RED ladder): the
            # wall-clock backstop latched a wedge the per-tick ledger couldn't see (a frozen dialogue box,
            # a sub-tick spin — whatever sub-layer was stuck). Honor it: clear+reset the watch FIRST (so
            # the recovery's own travel/heal isn't instantly re-cancelled by the stale request), mark the
            # wedge spot blocked (travel routes around it next time), then force a real POSITION break and
            # tell her plainly to do something different. A 2nd trip this episode escalates to the
            # escape-hatch reload. Reset the trip counter on GREEN (real progress) below.
            if self._stuck_request is not None:
                req, self._stuck_request = self._stuck_request, None
                if self._stuckwatch is not None:
                    self._stuckwatch.reset()
                self._watchdog_trips += 1
                log(f"   [roam] !!!! WATCHDOG DISENGAGE #{self._watchdog_trips} at "
                    f"{req['map']}@{req['coords']} (reason={req['reason']}, frozen {req.get('secs')}s) "
                    f"— top-level recovery, sub-layer-agnostic")
                # THE REAL ESCAPE (not narration): if a dialogue box is up she's been mashing A into a
                # looping NPC — A can NEVER end that loop. Press B to DISMISS it, then STEP AWAY, and
                # mark the NPC sticky-blocked. Do this FIRST — heal/travel can't move while a box is open.
                # PROBLEM 2 crash-hardening: the whole recovery is guarded LOUD so a disengage fault can't
                # kill an unattended run (never swallow — full traceback logged).
                try:
                    from dialogue_drive import box_open as _bx
                    if req["reason"] == "frozen_box" or _bx(self.b):
                        self.on_event("this NPC's just looping the same lines and going nowhere — I'm "
                                      "closing it and walking away.", kind="recover", tier=2)
                        self._disengage_overworld_npc(req)      # B-to-close + step away + sticky mark
                    else:
                        self._mark_wedge_spot(req)
                        self.on_event("I've been stuck on the same spot going nowhere — backing off and "
                                      "trying something different.", kind="recover", tier=2)
                    # ESCALATE only if she's STILL re-wedging this episode (the B+step-away didn't take):
                    if self._watchdog_trips >= 2:
                        log("   [roam] !! WATCHDOG re-trip this episode — escalating beyond B+step-away")
                        if self._last_good_state:
                            self._escape_hatch_reload()
                        elif tv.map_id(self.b)[0] != 3:   # stranded in a building -> get to the overworld
                            self._exit_to_overworld()
                        else:
                            self.heal_nearest()           # last-resort known-reachable position break
                except Exception as _de:
                    import traceback as _tb
                    log(f"   [roam] !!!! WATCHDOG DISENGAGE CRASHED: {_de!r} — caught LOUD, continuing "
                        f"the run. Traceback:\n{_tb.format_exc()}")
                ledger.note_action("watchdog_disengage", req["reason"])
                self._wait_overworld()
                continue
            if macro == ledger.GREEN:                   # real progress -> new episode, reset trip budget
                self._watchdog_trips = 0
            # STEP-3 HARD RECOVERY (increment 4 PART B): awareness (inc3 oracle feedback) is step 1, but
            # SUSTAINED RED means re-asking isn't working — the system must guarantee a POSITION change or
            # stop loud, never re-ask an impossible question for 20+ ticks (the live wedge). Capability-
            # not-script: she still picks among REAL options; this only fires when the WORLD won't move.
            red_ticks = red_ticks + 1 if macro == ledger.RED else 0
            if macro != ledger.RED:
                hard_recovered = False
                escape_reloads = 0                 # new episode — reset the escape budget
                self._deepwedge_reverts = 0        # real progress — reset the deep-wedge ring budget too
            # ADDENDUM A — LAST-RESORT ESCAPE-HATCH: after the forced-heal hard-recovery has been tried and
            # RED *still* persists (a genuine fingerprint-frozen wedge, not idle), bank-current then reload
            # the last known-good state and continue — BEFORE giving up. Bounded (MAX_ESCAPE_RELOADS) so a
            # self-re-wedging spot still reaches ABANDON. Anti-misfire safety lives in _escape_hatch_reload.
            if (red_ticks >= PROGRESS_ESCAPE_TICKS and hard_recovered
                    and escape_reloads < MAX_ESCAPE_RELOADS):
                log(f"   [roam] !! ESCAPE-HATCH considering reload: RED {red_ticks} ticks, hard-recovery "
                    f"already tried, reload {escape_reloads + 1}/{MAX_ESCAPE_RELOADS}")
                if self._escape_hatch_reload():
                    escape_reloads += 1
                    red_ticks = 0                  # the reload broke the position — give it a fresh streak
                    continue
            if red_ticks >= max(wf.PROGRESS_ABANDON_TICKS, DEEPWEDGE_TICKS):
                # PHASE 1 — DEEP-WEDGE FLOOR: the escape-hatch (recent-good) is exhausted and the world is
                # STILL frozen — a structural wedge. Before abandoning, revert PROGRESSIVELY FURTHER BACK
                # through the gain-seam ring to a guaranteed-clear checkpoint. Each revert is surfaced
                # in-character (watchable, not a silent blink). Only when the ring is spent do we abandon.
                if self._deep_wedge_revert():
                    red_ticks = 0
                    continue
                log(f"   [roam] !!!! ROAM ABANDONED: RED for {red_ticks} ticks; escape-hatch AND the "
                    f"deep-wedge ring ({self._deepwedge_reverts} reverts) both exhausted — genuinely "
                    f"unrecoverable, this NEEDS A HUMAN (red light's real meaning)")
                self._roam_progress = "ABANDONED"
                self.on_event("I'm completely stuck — I've tried everything and I can't find a way forward "
                              "on my own. I need a hand here.", kind="abandoned", tier=3)
                self._fire_deadman_alert(state)    # PHASE 2 — dead-man's switch: ping Jonny, recovery failed
                return "abandoned"
            if red_ticks >= wf.PROGRESS_HARD_TICKS and not hard_recovered:
                log(f"   [roam] !! HARD RECOVERY: RED {red_ticks} ticks — FORCING a route to the nearest "
                    f"Pokémon Center to break the position (not re-asking the oracle this tick)")
                self.on_event("okay, this isn't working — I'm heading to the Pokémon Center to reset and "
                              "figure out my next move.", kind="recover", tier=2)
                hard_recovered = True
                _hr_pos = (tuple(tv.map_id(self.b)), tv.coords(self.b))
                self.heal_nearest()                # known-reachable anchor; restores HP -> fp moves -> escape
                # POSITION-BREAK GUARANTEE (2026-07-05, run-4 lesson): with a FULL party the heal-excursion
                # walks to the adjacent city and FAITHFULLY RETURNS to the exact wedge tile — a perfect
                # no-op that leaves the fingerprint frozen (RED kept climbing until the escape-hatch
                # reverted a whole bench-level away). Hard recovery's JOB is a position change: if heal
                # ended where it started, take the excursion ONE-WAY — cross to an adjacent Center city
                # and STAY there (the roam re-routes forward from the city next tick).
                if (tuple(tv.map_id(self.b)), tv.coords(self.b)) == _hr_pos:
                    log("   [roam] !! HARD RECOVERY was a no-op (healed + returned to the same tile) — "
                        "breaking the position ONE-WAY to an adjacent Center city")
                    _EDGE = {"N": "north", "S": "south", "E": "east", "W": "west"}
                    for _d, _nbr in self._map_connections():
                        if _nbr in CITY_PC_DOORS and tuple(_nbr) != tuple(tv.map_id(self.b)) and _d in _EDGE:
                            r_ow = self.trav.travel(target_map=_nbr, edge=_EDGE[_d])
                            log(f"   [roam] one-way position break {_EDGE[_d]} -> {_nbr}: {r_ow}")
                            if tuple(tv.map_id(self.b)) == tuple(_nbr):
                                break
                ledger.note_action("hard_recovery", "forced_center")
                continue                           # re-observe next tick from the broken position
            log(f"   [roam] OPTIONS OFFERED: {list(avail.keys())}")
            if not avail:
                log("   [roam] no honest action available here — ending free roam"); break
            if self.soul is not None and (tick == 1 or tick % want_every == 1):
                log(f"   [soul] surface_want FIRE -> {state['place']}")
                self.soul.surface_want({"place": state["place"], "map": state["map"],
                                        "badges": state["badges"], "progress": state["progress"],
                                        "party": self._party_brief(state),     # PHASE 1: team by NAME
                                        "goal": self._goal_layers(state)})      # PHASE 1: 3-tier goal
            # On YELLOW+, fold STUCK-AWARENESS into the oracle ctx via the existing `place` seam (the
            # only general field her oracle prompt renders — firewall: no core edit). She becomes AWARE
            # she's stuck; she still decides the next move HERSELF (capability-not-script).
            where = state["place"]
            # Batch-WORLD (Phase 2) — SENSE OF PLACE: lead the oracle ctx with a short spatial picture
            # from her visited-world memory (where she is, what's around, what's BLOCKED, what she can
            # walk back to, how she can travel). This is the MAP she never had — so when a path is
            # blocked she has somewhere in her head to go besides into the wall. Firewall: same `place`
            # seam; she still decides. Avoid = route-walls she can't beat; blocked_dirs = a wall on THIS map.
            try:
                avoid = self._wall_avoid(state)
                bdirs = self._wall_blocked_dirs(state)
                cur_map = tuple(state["map"])
                brief = self.world.spatial_brief(cur_map, avoid=avoid, blocked_dirs=bdirs)
                if brief:
                    where = f"{where}. {brief}"
                # GATE-UNLOCK: when she's mid-errand to unlock a gate, fold her DERIVED plan into the ctx
                # so she KNOWS why she's heading where she is (reaches the DECISION context — load-bearing;
                # she narrates + still chooses). Self-clears when the success flag reads satisfied LIVE.
                if (QUESTLINE_ENABLED and self._active_questline is not None
                        and not self._active_questline.complete):
                    qn = self._active_questline.narration()
                    if qn:
                        where = f"{where}. {qn}"
                # Phase 5 — BALL PRE-CHECK: zero Poké Balls makes wander_catch impossible no matter how
                # much grass she finds, so tell her plainly (info she lacks). Live read; confirmed on the
                # watch. The capability to act on it already exists (travel to a Mart / stock_up).
                bn = self._ball_note(state)
                if bn:
                    where = f"{where} {bn}"
                # Phase 5 — the ONE allowed explicit nudge (info she lacks, NOT a forced action): when she's
                # strategically stuck at a wall she can't pass, name plainly the KNOWN places back the way
                # she came that have what she needs. Right now she doesn't know that option exists — so we
                # tell her it does. She still DECIDES.
                if STRATEGIC_STUCK_ENABLED and self.strat.strategically_stuck(
                        state.get("party_count"),
                        state["party"][0]["level"] if state.get("party") else None):
                    backs = self.world.travel_targets(cur_map, avoid=avoid)
                    if backs:
                        names = ", ".join(nm for _m, nm, _w in backs[:3])
                        where = (f"{where} You can't pass this wall as you are — and the places you've "
                                 f"already been that have what you need (wild Pokémon to train on, a Mart "
                                 f"for Poké Balls) are back the way you came: {names}. Going back to build "
                                 f"up and then returning is the real move here.")
            except Exception as _wb:
                log(f"   [world] spatial brief skipped: {_wb}")
            # SURVIVAL AWARENESS (PART A): fold the party-hurt note into the oracle ctx via the same
            # `place` seam the stuck-note uses (the only general field her prompt renders — firewall: no
            # core edit). A badly-hurt Kira is TOLD she's hurt + that healing comes first; she still
            # decides. Folded BEFORE the stuck-note so survival framing leads when both are present.
            sev, hurt_note = self._hurt_severity()
            if hurt_note:
                where = f"{where}. {hurt_note}"
                log(f"   [roam] !! SURVIVAL {sev.upper()}: {hurt_note!r} — surfacing 'heal' to the oracle")
            # SHOP-WITH-INTENT awareness (PART C): when stock_up is on offer, fold the characterful
            # restock/affliction note into the same ctx seam so her pick reads as learning from the road.
            if "stock_up" in avail:
                snote = self._shop_note(foresight=self._walled(state))
                if snote:
                    where = f"{where}. {snote}"
                    log(f"   [roam] SHOP: {snote!r} — surfacing 'stock_up' to the oracle")
            if macro in (ledger.YELLOW, ledger.RED):
                note = ledger.stuck_note()
                where = f"{where}. {note}"
                log(f"   [roam] !! MACRO {macro}: no progress {ledger.stuck} ticks — feeding awareness "
                    f"back to the oracle: {note!r}")
            # BATCH 6 PHASE 5 — SILENT GUIDE: a genuine gap (truly stuck, or a wall she can't crack) is
            # exactly when a real player reaches for the strategy guide. She quietly checks and the RESULT
            # folds into her reasoning IN CHARACTER (firewall: same `place` seam; she still decides). No-op
            # unless POKEMON_GUIDE_SEARCH=1 AND the Custom Search API is enabled — enrichment, never a dep.
            if self.guide.available() and (macro == ledger.RED or self.strat.active_wall):
                hint = self._guide_lookup(state)
                if hint:
                    where = f"{where}. {hint}"
            # STRATEGIC AWARENESS (Batch 3 Phase 2): fold LOSS-LEARNING (she's hit a wall — same fight
            # lost ≥2x = brute-forcing isn't working) + ROSTER-SHAPE (one Pokémon isn't a team) into the
            # SAME `place` seam. FACTS + the menu of options (level / teammate / counter / come back),
            # never a command — she still picks (capability-not-script; a stubborn solo-run stays valid).
            # Loss-awareness leads: it's the run-existential one (the die→run-back→die loop killer).
            # STRATEGIC-STUCK FLOOR — the DOMINANT note (leads, ahead of the softer loss_awareness). When
            # she's truly stuck (≥N identical losses, no change), this names the loop and that strengthening
            # comes FIRST. Paired with the option-pruning in _available_actions, this is the floor with teeth.
            _pc = state.get("party_count")
            _pl = state["party"][0]["level"] if state.get("party") else None
            # PHASE 2 — READINESS → GO (the grind's EXIT). Once she's crossed the readiness bar since the
            # wall, the move is to RETURN — so fold the DOMINANT 'go back now' pull AND suppress the now-
            # stale 'come back stronger' loss-awareness (which otherwise keeps telling her NOT to go back
            # even after she's grown — the inertia that made her grind forever tonight).
            _ready = self.strat.ready_to_retry(_pc, _pl) if STRATEGIC_STUCK_ENABLED else None
            if _ready:
                rgn = self.strat.ready_to_retry_note(_pc, _pl)
                if rgn:
                    where = f"{where}. {rgn}"
                    log(f"   [roam] !!!! READINESS → GO vs {self.strat.active_wall}: prep bar crossed — "
                        f"surfacing the 'go back and take the wall' pull (the grind now has an exit)")
            if STRATEGIC_STUCK_ENABLED and not _ready and self.strat.strategically_stuck(_pc, _pl):
                ssn = self.strat.strategic_stuck_note()
                if ssn:
                    where = f"{where}. {ssn}"
                    log(f"   [roam] !!!! STRATEGIC-STUCK FLOOR vs {self.strat.active_wall}: dominant "
                        f"'strengthen-first' directive folded — breaking the die→re-charge→die loop")
            if not _ready:                                    # don't nag 'come back stronger' once she IS
                la = self.strat.loss_awareness()
                if la:
                    where = f"{where}. {la}"
                    log(f"   [roam] !! STRATEGY wall vs {self.strat.active_wall}: feeding loss-awareness to the oracle")
            # STRATEGIC UNDERLEVEL-GRIND (Task B): when the wall keeps her down because the TEAM FLOOR is
            # under-levelled, fold the CONCRETE prep plan (level the WEAK ones — by name — to readiness,
            # field THEM not the ace, THEN cross) into her decision/voice ctx. This is the reasoning Jonny
            # SEES her act on: she steps aside, grinds the weak members with a stated purpose, and returns.
            if not _ready:
                _pt = self._prep_team_target(state)
                if _pt is not None:
                    ptn = self.strat.prep_team_note(self._prep_team_weak(state, _pt), _pt)
                    if ptn:
                        where = f"{where}. {ptn}"
                        log(f"   [roam] !! UNDERLEVEL-PREP vs {self.strat.active_wall}: team floor < L{_pt} "
                            f"— folding the 'level the weak ones, field THEM' plan into her decision ctx")
            # BATCH-4 PHASE 2 (persistent SPATIAL wall): while a wall gates a route and she hasn't grown,
            # keep telling her the way forward is blocked — so she stops blindly choosing to re-cross it
            # (the routing also hard-blocks it; this is the awareness half so the CHOICE is informed).
            wr = self.strat.active_wall_rec()
            if wr and wr.get("map_id") and not self.strat.stronger_since_wall(
                    state.get("party_count"), state["party"][0]["level"] if state.get("party") else None):
                wg = self.strat.wall_gate_note("get past here")
                if wg:
                    where = f"{where}. {wg}"
                    log("   [roam] !! SPATIAL WALL: folding gate-awareness (route blocked until stronger)")
                # BATCH-6 PHASE 2c — when she's KEPT trying the blocked path, escalate hard: name the loop
                # out loud and point at the concrete THIS-SIDE moves (build a team in the grass behind you,
                # stock up at the Mart) so the audience sees her break the die→re-walk→die pattern.
                if self._wall_gated_streak >= 2:
                    where = (f"{where}. You keep turning back to that same blocked path — and it's blocked "
                             f"every time, this is the exact loop that just blacks you out. Stop pushing it. "
                             f"The real move is on THIS side: grind the grass behind you to level up, catch a "
                             f"teammate for backup, or stock up at the Mart — THEN come back and take it.")
                    log(f"   [roam] !! DETER: wall_gated streak {self._wall_gated_streak} — escalating "
                        f"this-side options to break the re-walk loop")
            ra = self.strat.roster_awareness(state["party"])
            if ra:
                where = f"{where}. {ra}"
                log(f"   [roam] STRATEGY roster: feeding team-shape awareness to the oracle")
            # PHASE 1 — fold her 3-tier PLAN + team-by-name into the rendered `place` seam so she works
            # toward a stated objective AND narrates it to chat (and never asks who's on her team / what
            # she's doing — the immersion bug). She's the one PLAYING; first-person framing is explicit.
            _goals = self._goal_layers(state)
            _brief = self._party_brief(state)
            # FIX 1+2 — fold the FIXED forward spine + her real run history in FIRST, so every decision is
            # anchored to "always heading to the next gym" and grounded in what she's actually done.
            try:
                where = f"{where}. {self._spine_and_history(state)}"
            except Exception as _sh:
                log(f"   [roam] spine/history fold skipped: {_sh}")
            where = (f"{where}. YOUR PLAN — right now: {_goals['short']}. Next: {_goals['medium']}. "
                     f"Bigger picture: {_goals['long']}. Your team right now: {_brief}. Remember YOU are "
                     f"the one playing this — narrate your plan to chat in your own voice (first person: "
                     f"\"I'm grinding\", \"my team\"), and never ask anyone what you're doing or who's on "
                     f"your team — you know.")
            # RECALIBRATION — if she had a standing objective before the last detour, remind her of it so
            # she picks the thread back up (battles/healing are interruptions, not a change of plan).
            if self._active_objective and self._active_objective.get("label") \
                    and self._active_objective.get("action") in avail:
                where = (f"{where}. RECALIBRATE — before the last detour you'd set out to: "
                         f"\"{self._active_objective['label']}\". That's still available. Unless something "
                         f"fundamental has actually changed, pick it back up and continue — don't lose the "
                         f"thread to aimless wandering.")
            # FIX 1 (overworld half) — REPETITION-AVERSE: if she repeated the same action last tick AND
            # the world fingerprint is STUCK (non-GREEN = nothing changed), that action isn't working —
            # tell her to STOP repeating it and do something different. Gated on non-GREEN so a legit
            # multi-tick action (walking to a Center) — where the fingerprint keeps changing — is never
            # nagged. One repeat is the ceiling a viewer accepts; this fires at the first stuck repeat.
            if self._repeat_pick_n >= 1 and macro != ledger.GREEN and self._last_action_pick in avail:
                where = (f"{where}. HEADS UP — you just chose '{self._last_action_pick}' and NOTHING "
                         f"changed (same spot, no progress). Repeating it again will do the same nothing. "
                         f"Pick a DIFFERENT action this time and break the loop.")
                log(f"   [roam] !! REPEAT-NO-PROGRESS: '{self._last_action_pick}' repeated with a stuck "
                    f"fingerprint — nudging a different action")
            # PROBLEM 1 (lag) TRACE: _soul_choose -> voice.choose is a SYNCHRONOUS blocking HTTP/LLM
            # call on this (main) thread — while it runs, game render + music are frozen. Time it so a
            # slow decision (the periodic stutter, worst when ticks fire fast on a stuck loop) is VISIBLE.
            _t_choose = _t.time()
            pick = self._soul_choose("action", avail,
                                     {"place": where, "progress": state["progress"], "party": _brief,
                                      "goal": _goals})
            _choose_ms = (_t.time() - _t_choose) * 1000
            if _choose_ms > 500:
                log(f"   [roam] !! DECISION LATENCY: _soul_choose blocked the main thread {_choose_ms:.0f}ms "
                    f"(game+music stutter while it waits on the LLM over HTTP) — the periodic lag source "
                    f"when ticks fire rapidly")
            if not pick:
                log("   [roam] PICK: (oracle returned nothing — no bot / unparsable) -> idle this tick")
                ledger.note_action(None, "idle")
                continue
            log(f"   [roam] PICK OUT: {pick}" + (f"  <- her RECOVERY move ({macro})" if macro != ledger.GREEN else ""))
            # FIX 1 — track consecutive identical picks (for the repeat-no-progress nudge above).
            self._repeat_pick_n = self._repeat_pick_n + 1 if pick == self._last_action_pick else 0
            self._last_action_pick = pick
            # FIX 3 — live RATIONALE for the dashboard ("why am I doing THIS right now"), from her real
            # current plan + pick. Published in the next health snapshot so Jonny reads purposeful-vs-lost.
            self._rationale = self._rationale_line(state, pick, _goals)
            log(f"   [roam] RATIONALE: {self._rationale}")
            # FIX 3 (freshness) — re-publish so the dashboard shows THIS decision's rationale DURING the
            # (visible) action execution below, not a tick late. The top-of-tick publish (which the
            # watchdog light needs) ran before the pick existed, so it carried the PREVIOUS tick's 'why';
            # this second best-effort write lands her live reasoning while she's actually doing the thing.
            # Cheap atomic JSON write; never blocks the run.
            self._publish_health(macro, state, last_badge_ts, run_start_ts)
            # RECALIBRATION — record a PURPOSEFUL, going-somewhere pick as the standing objective she
            # returns to. Detour actions (heal / talk_npc / stock_up / grind / idle) deliberately do NOT
            # overwrite it, so the thread survives them. Cleared when its action completes (below).
            if pick == "head_to_gym" or pick == "wander_catch" or pick.startswith("travel:"):
                self._active_objective = {"action": pick, "label": avail.get(pick, pick), "tick": tick}
            # PROBLEM 3 TRACE — capture her RAM position around the action so a SILENT travel failure
            # (head_to_gym fires but she never moves) is logged, not theorized.
            _pos0 = (tv.map_id(self.b), tv.coords(self.b))
            try:
                out = self._route_action(pick, state)
            except Exception as _re:
                # PROBLEM 2 crash-hardening: one bad handler must NOT kill a 30-hr unattended run. Log
                # LOUD with the full traceback (never swallow — CLAUDE.md rule 3) and treat as a stuck
                # outcome so the recovery ladder engages next tick instead of the process dying.
                import traceback as _tb
                log(f"   [roam] !!!! ACTION CRASHED: '{pick}' raised {_re!r} — caught LOUD, continuing "
                    f"the run (not dying). Traceback:\n{_tb.format_exc()}")
                out = "action_error"
            _pos1 = (tv.map_id(self.b), tv.coords(self.b))
            _moved = _pos0 != _pos1
            log(f"   [roam] RESULT: {pick} -> {out} | pos {_pos0[0]}{_pos0[1]} -> {_pos1[0]}{_pos1[1]} "
                f"({'MOVED' if _moved else 'NO MOVEMENT'})")
            ledger.note_action(pick, out)              # remember for next tick's progress check + feedback
            # PROBLEM 3 — SILENT-NO-MOVE GUARD: a movement pick that returned WITHOUT moving her and
            # WITHOUT arriving is silently failing (the live "head_to_gym isn't moving me" loop). Drop the
            # standing objective so recalibration stops re-pushing a dead route, and count it; the prune in
            # _available_actions removes the pick after the 2nd consecutive no-move so she MUST do something
            # else (capability-not-script — the action genuinely isn't doing anything here). Reset on a move.
            _is_move_pick = (pick == "head_to_gym" or pick == "wander_catch" or pick.startswith("travel:"))
            if _is_move_pick and not _moved and out not in ("arrived", "badge", "caught"):
                self._nomove_streak += 1
                self._dead_moves.add(pick)
                log(f"   [roam] !! SILENT NO-MOVE: '{pick}' returned '{out}' but her position never changed "
                    f"(streak {self._nomove_streak}, dead routes {sorted(self._dead_moves)}) — not moving "
                    f"her; ALL dead routes get pruned next tick so she must do something that works")
                if self._active_objective and pick == self._active_objective.get("action"):
                    self._active_objective = None
            elif _moved:
                self._nomove_streak = 0
                self._dead_moves.clear()
            # RECALIBRATION — clear the standing objective once it's actually achieved (badge won, teammate
            # caught, destination reached) so the recalibrate nudge stops pushing a finished goal; the next
            # purposeful pick sets the new one.
            if self._active_objective and pick == self._active_objective.get("action") \
                    and (out in ("badge", "caught") or "arrived" in str(out)):
                log(f"   [roam] standing objective complete ({pick} -> {out}) — clearing it")
                self._active_objective = None
            # LAYER A — DISTINCT 'NPC blocks the only route' outcome: travel already added the blocker to
            # the shared memory and confirmed there's no way around it from here. DROP the standing
            # objective so RECALIBRATE stops pushing this same now-blocked route every tick, and name it
            # in-character so the next pick is a genuinely different objective (capability-not-script —
            # she still chooses; we just stop re-issuing into the same door).
            if out == "no_route_npc_blocked":
                log(f"   [roam] !! {pick} -> no_route_npc_blocked: route blocked by a stuck NPC with no "
                    f"way around — clearing the standing objective so the oracle picks differently")
                if self._active_objective and pick == self._active_objective.get("action"):
                    self._active_objective = None
                self.on_event("huh — the only way that direction is blocked by someone who won't move. "
                              "no point pushing it; I'll do something else and come back to it.",
                              kind="recover", tier=2)
            # BATCH-6 PHASE 2c — DETER the blind re-walk: when the routing HARD-REFUSED to cross the wall
            # (wall_gated), count it. A rising streak means she keeps trying the one path that's blocked —
            # the next-tick oracle ctx escalates "this ends the same way; go build/stock on THIS side"
            # (folded below via _wall_gated_streak). She can still choose it (agency) but it stops being
            # the default loop. Any productive move (heal/stock/grass-behind/talk) clears the streak.
            if out == "wall_gated":
                self._wall_gated_streak += 1
                log(f"   [roam] !! WALL-GATED streak={self._wall_gated_streak} — the way forward stays "
                    f"blocked; escalating 'build a team / stock up on THIS side' to the oracle")
            else:
                self._wall_gated_streak = 0
            if pick == "heal" and out == "ok":
                # SOUL BEAT (PART A): she took care of herself — voice it through the seam (firewall).
                self.on_event("okay — patched up and ready. let's go.", kind="heal", tier=2)
            if pick == "stock_up" and out == "stocked":
                # SOUL BEAT (PART C): characterful restock — name what she learned to carry (firewall).
                cured = ", ".join(STATUS_CURE[s][1] for s in sorted(self._afflict_seen) if s in STATUS_CURE)
                self.on_event(("stocked up — and grabbed " + cured + " so that doesn't cost me again")
                              if cured else "stocked up on potions — better to have them and not need them",
                              kind="shop", tier=2)
            # ANY pick that ends in a catch gets the family beat — the 2026-07-06 nursery fix: a catch
            # via travel:<grass-map> (which auto-hunts on arrival) was skipping the naming/bond hook,
            # which is why soul.json bonds stayed EMPTY despite a real catch (the soul-debt #3 gap).
            if out == "caught" and self.soul is not None:
                self.roster_react()                        # note_caught: bond + react to the new teammate
            if self.soul is not None:
                self._soul_after_objective("FREE_ROAM", out)   # note_evolve / note_faint+outcome(loss)
        # CLEAN EXIT: bank her final position so the next GO resumes exactly here (the run ending — time
        # budget / ticks — is NOT a reset; she stays anchored where she climbed to). One last sig-check
        # so a catch/area-change on the FINAL tick is reflected in the reason.
        self._save_campaign("roam_end")
        self._continuity_save()                    # ADDENDUM D: persist her saga at the clean exit too
        log("==== FREE ROAM complete ====")
        return "roam_done"

    # ── the continuous driver ────────────────────────────────────────────────
    def run(self, objectives):
        for i, obj in enumerate(objectives):
            kind, label = obj[0], obj[-1]
            _t0 = time.time()
            log(f"OBJECTIVE {i+1}/{len(objectives)}: {kind} - {label}  "
                f"[at map={tv.map_id(self.b)} coords={tv.coords(self.b)}]")
            if kind == "OPENING":
                self._title_to_newgame()             # fresh ROM -> title -> New Game -> Oak's intro
                m = self.drive_opening(obj[1], obj[2])    # her char-creation + starter + rival
                out = "complete" if m == PALLET else "stuck"
            elif kind == "WALK_TO_MAP":
                out = self.walk_to_map(obj[1], obj[2])
            elif kind == "ADVANCE_NORTH":
                out = self.advance_north(obj[1])
            elif kind == "ENTER_WARP":
                spec = obj[1]                        # (x,y) door pick | "north"/"nearest" | None
                out = (self.enter_warp(pick=spec) if isinstance(spec, tuple)
                       else self.enter_warp(prefer=spec or "nearest"))
            elif kind == "LEVEL_CHECK":
                out = self.level_check(obj[1])
            elif kind == "HEAL":
                out = self.heal_at_center()
            elif kind == "GRIND":
                out = self.grind(obj[1])
            elif kind == "BEAT_GYM":
                out = self.beat_gym(obj[1])
            elif kind == "CATCH":
                out = self.catch_one()
            elif kind == "DELIVER_PARCEL":
                out = self.deliver_parcel()
            elif kind == "CLEAR_MT_MOON":
                out = self.clear_mt_moon()
            elif kind == "HEAL_NEAREST":
                out = "ok" if self.heal_nearest() == "ok" else "ok"   # heal is best-effort, never a stop
            elif kind == "ROSTER_REACT":
                out = self.roster_react()
            elif kind == "GATE_NEEDS_STATE":
                state, what = obj[1], obj[2]
                path = self._resolve_gate(state)
                if self.show_mode:
                    # A banked-state fallback in SHOW mode is a HARD-FAILURE we want SURFACED, not
                    # silently skipped: log LOUD + count. The run still loads (if the file exists) so
                    # ONE show run reveals EVERY remaining gate; violations>0 = the spine isn't AUTO.
                    self._show_violations += 1
                    log(f"   !!!! SHOW-MODE SKIP VIOLATION #{self._show_violations}: {label} "
                        f"(would load banked sherpa state {state}) — this skip is NOT yet AUTO")
                if path:
                    log(f"   gate savestate present ({state}) - loading past the gate")
                    with open(path, "rb") as f:
                        self.b.load_state(f.read())
                    for _ in range(40):
                        self.b.run_frame()
                    out = "loaded"
                else:
                    log(f"   !! GATE NEEDS SAVESTATE: {state}")
                    log(f"      hand-play once: {what}")
                    log(f"      then save it to states/workshop/{state} and re-run.")
                    return f"blocked_gate:{state}"
            else:
                out = "unknown"
            log(f"   -> objective result: {out} in {time.time()-_t0:.0f}s "
                f"(now map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            if self.soul is not None:
                self._soul_after_objective(kind, out)     # evolve + faint + battle->mood, via the seam
            if out in ("stuck", "no_path", "battle_loss", "no_warp", "underleveled", "blackout"):
                log(f"!! CAMPAIGN STOP (loud) at objective {i+1} ({kind}): {out}")
                return f"stopped:{kind}:{out}"
        log("CAMPAIGN COMPLETE (all objectives done)")
        return "complete"

    # ── SEGMENT MANIFEST driver (the resumable whole-game spine) ──────────────────
    def _resolve_gate(self, name):
        """Find a banked sherpa/gate state across the live buckets (see resolve_state). Returns the
        full path or None. (Each gate converted to AUTO drops its load entirely.)"""
        return resolve_state(name)

    def _save_checkpoint(self, name) -> bool:
        """Auto-save a resumable overworld checkpoint between segments into self.ckpt_dir. LOUD on
        failure (Constraint #3). HARD GUARD: a WORKSHOP run is physically forbidden to write into
        states/kira/ (her canonical saves) — only a SHOW run banks there."""
        if not name:
            return False
        target_dir = self.ckpt_dir
        kira_abs = os.path.abspath(STATES_KIRA)
        if not self.show_mode and os.path.abspath(target_dir).startswith(kira_abs):
            log("   !! CHECKPOINT REFUSED: WORKSHOP mode may not write into states/kira/ (canonical)")
            return False
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, name)
            data = self.b.save_state()
            with open(path, "wb") as f:
                f.write(data)
            where = "kira" if self.show_mode else "workshop"
            log(f"   CHECKPOINT saved -> {where}/{name}  "
                f"(map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            return True
        except Exception as e:
            log(f"   !! CHECKPOINT SAVE FAILED ({name}): {e}")
            return False

    def _save_campaign(self, reason="tick") -> bool:
        """BATCH 5 PHASE 1 — bank her LIVING campaign save (the Sherpa-timeline anchor): the single
        savestate that the next GO resumes from, so she keeps climbing from where she actually is and is
        never rappelled back to a frozen fragment. Writes ATOMICALLY (temp + os.replace) so a kill/crash
        mid-write can never corrupt the anchor (a half-written save would lose the whole climb). Lands in
        states/campaign/ ONLY — physically separate from the workshop fragments (kept as fallbacks) and
        the canonical states/kira/ spine. LOUD on success + failure (Constraint #3)."""
        try:
            os.makedirs(STATES_CAMPAIGN, exist_ok=True)
            path = os.path.join(STATES_CAMPAIGN, CAMPAIGN_SAVE)
            tmp = path + ".tmp"
            data = self.b.save_state()
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)                      # atomic swap — the anchor is never half-written
            badges = sum(1 for i in range(8) if self.has_badge(0x820 + i))
            party = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            log(f"   ⛰️  CAMPAIGN SAVE [{reason}] -> campaign/{CAMPAIGN_SAVE}  "
                f"(map={tv.map_id(self.b)} coords={tv.coords(self.b)} badges={badges} party={party})")
            return True
        except Exception as e:
            log(f"   !! CAMPAIGN SAVE FAILED [{reason}]: {e} — her position is NOT anchored this tick (LOUD)")
            return False

    def _playthrough_elapsed(self):
        """HUD — seconds since this PLAYTHROUGH began (persists across sessions). Stamps the start once
        (first ever free-roam) into playthrough.json, then reports elapsed forever after. Best-effort:
        returns None if it can't read/write (the HUD just hides the timer rather than lying)."""
        import json as _json
        try:
            start = None
            if os.path.exists(PLAYTHROUGH_JSON):
                with open(PLAYTHROUGH_JSON, encoding="utf-8") as f:
                    start = _json.load(f).get("start_ts")
            if not start:
                start = time.time()
                os.makedirs(STATES_CAMPAIGN, exist_ok=True)
                tmp = PLAYTHROUGH_JSON + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    _json.dump({"start_ts": start}, f)
                os.replace(tmp, PLAYTHROUGH_JSON)
                log(f"   [hud] playthrough timer started (stamped {PLAYTHROUGH_JSON})")
            return max(0.0, time.time() - start)
        except Exception as e:
            log(f"   [hud] playthrough timer read skipped: {e}")
            return None

    def _party_brief(self, state):
        """Her team, by NAME — '<nickname> the <species> (Lxx)' when she's named one (soul.bonds), else
        '<species> (Lxx)'. So her context always carries WHO is on her team — she states them instead of
        asking Jonny 'who's in my party?' (the immersion bug). Plain string; '(no team yet)' if empty."""
        out = []
        bonds = {}
        try:
            bonds = self.soul.bonds or {} if self.soul is not None else {}
        except Exception:
            bonds = {}
        for m in state.get("party", []):
            sp = m.get("species")
            nick = None
            try:
                for _k, v in bonds.items():
                    if isinstance(v, dict) and v.get("species") == sp and v.get("nickname"):
                        nick = v["nickname"]; break
            except Exception:
                nick = None
            label = f"{nick} the {sp}" if nick and nick != sp else (sp or "a Pokémon")
            out.append(f"{label} (L{m.get('level')})")
        return ", ".join(out) or "(no team yet)"

    def _goal_layers(self, state):
        """THREE-HORIZON goal read (short / medium / long), DERIVED from her real live state — never a
        hardcoded script. Extends the existing self-want + next_gym (not a parallel system):
          SHORT  = what she's doing right now (the active strengthen task w/ END CONDITION when walled,
                   else her current soul want),
          MEDIUM = the obstacle she's preparing for (the active wall — 'beat <who> at <place>'), or the
                   road to the next gym,
          LONG   = the next badge milestone (next gym), or the Elite Four.
        Best-effort; returns {'short','medium','long'} of plain strings (may be '')."""
        try:
            pc = state.get("party_count")
            party = state.get("party") or []
            pl = party[0]["level"] if party else None
            team_n = len(party)
            ng = state.get("next_gym")
            # LONG — the next badge milestone
            if ng:
                try:
                    badge = self._BADGE_NAMES[state.get("badge_count", 0)]
                    long = f"Reach {ng['city']}, beat {ng['leader']} for the {badge} Badge"
                except Exception:
                    long = f"Reach {ng['city']} and beat {ng['leader']}"
            else:
                long = "Take on the Elite Four"
            wr = self.strat.active_wall_rec()
            rdy = self.strat.ready_to_retry(pc, pl) if STRATEGIC_STUCK_ENABLED else None
            # MEDIUM — the road ahead. Once READY, it's FORWARD to the next gym city (a concrete place she
            # can route to), NOT a vague 'go back and rematch' with no coords (the aimless-grind trap). The
            # wall only frames MEDIUM while she's still UNDER-prepared (the genuine 'build up first' phase).
            if wr and not rdy:
                if wr.get("is_trainer"):
                    who = wr.get("name") or (f"{wr.get('lead')}" if wr.get("lead") else "that rival")
                else:
                    who = f"the wild {wr.get('lead')}" if wr.get("lead") else "that wall"
                place = wr.get("place") or "that route"
                onward = f", then push on toward {ng['city']}" if ng else ""
                medium = f"Build a team strong enough to beat {who} at {place}{onward}"
            elif ng:
                medium = (f"You're ready — travel to {ng['city']} and take on {ng['leader']}" if rdy
                          else f"Make your way to {ng['city']}")
            else:
                medium = "Press on through the league"
            # SHORT — current want, or the concrete strengthen task with an END CONDITION when walled
            want = ""
            try:
                if self.soul is not None and self.soul.wants:
                    want = str(self.soul.wants[-1])
            except Exception:
                want = ""
            if wr and not rdy:
                target = wr.get("lead_level") or 0
                # STRATEGIC UNDERLEVEL-GRIND (Task B): when the TEAM FLOOR is the problem, the concrete
                # task is to level the WEAK members (named) by fielding them — not a vague "train the team".
                pt = self._prep_team_target(state) if STRATEGIC_GRIND_ENABLED else None
                if pt is not None:
                    weak = self._prep_team_weak(state, pt)
                    who = (weak[0] if len(weak) == 1
                           else (", ".join(weak[:-1]) + " and " + weak[-1])) if weak else "the weak ones"
                    short = f"Level the weak ones ({who}) to ~L{pt} by fielding THEM in the grass, THEN retry"
                elif target and pl is not None:
                    short = f"Train the team toward ~L{target} (or catch one more teammate), THEN retry"
                elif team_n < 2:
                    short = "Catch a teammate so you're not going in solo, then retry"
                else:
                    short = "Grind the team up a couple levels, then go back and retry"
            elif rdy:
                _bc = state.get("badge_count", 0)
                _bn = self._BADGE_NAMES[_bc] if _bc < len(self._BADGE_NAMES) else None
                short = (f"You're ready — stop grinding and head to {ng['city']}"
                         + (f" for the {_bn} Badge" if _bn else "") if ng
                         else "You're ready — stop grinding and push forward to the next objective")
            elif want:
                short = want
            else:
                short = "Explore and keep the team battle-ready"
            return {"short": short, "medium": medium, "long": long}
        except Exception as e:
            log(f"   [goals] layer build skipped: {e}")
            return {"short": "", "medium": "", "long": ""}

    # FIX 2 — RUN HISTORY (her story-so-far), keyed by what her badges PROVE she's done this run. Honest
    # past, NOT omniscient future (the Lapras-confabulation guard: she knows where she's BEEN, discovers
    # what's ahead). Extend as she earns more badges.
    _STORY_MILESTONES = {
        0: "You just set out from Pallet Town with your first Pokémon — the whole journey's ahead of you.",
        1: "You started in Pallet Town, crossed Route 1-2 and Viridian Forest, and beat Brock in Pewter "
           "City for the Boulder Badge. Now you're pushing on toward the next gym.",
        2: "You started in Pallet Town, beat Brock in Pewter (Boulder Badge), fought up Route 3 and "
           "through the Mt. Moon cave — a real maze you nearly got lost in — reached Cerulean City, and "
           "beat Misty for the Cascade Badge. Now you're after your third gym.",
    }

    def _story_so_far(self, badge_count):
        """Her real run history (what she's actually done), for grounding her decision + voice context."""
        return self._STORY_MILESTONES.get(badge_count, "")

    def _spine_and_history(self, state):
        """FIX 1+2 — the FIXED main-quest SPINE + her real RUN HISTORY, folded into her oracle ctx so she's
        ALWAYS pointed at the next gym (never aimless) and grounded in what SHE has done this run (not
        foreknowledge). Pure strings from live badge state + _GYM_ORDER; NO fabricated map coords — the
        actual forward routing is live-RAM via head_to_gym (and learned as she walks it)."""
        bc = state.get("badge_count", 0)
        ng = state.get("next_gym")
        # ── FOUNDATIONAL GAME-MODEL (she's a blank slate — make the game's POINT explicit so her choices
        # aren't random; see CLAUDE.md "Kira's foundational FireRed game-model"). Her grounded player-
        # understanding, NOT omniscient future content. ──────────────────────────────────────────────
        model = ("HOW POKÉMON WORKS (what you're playing for): the GOAL is to earn 8 Gym Badges, beat the "
                 "Elite Four, and roll the credits — that's beating the game. You build a TEAM of up to 6 "
                 "Pokémon you like AND that cover each other (types/roles) — a solo carry with a dead-weight "
                 "bench is a losing setup; a real trainer fields a balanced, levelled squad. CATCHING wild "
                 "Pokémon is central ('gotta catch 'em all') — you catch to build that team and to fill the "
                 "Pokédex. When you meet a wild Pokémon, weigh it: is it cool/strong/useful, does it cover a "
                 "gap, is it better than a bench-warmer? — keep your best ~6, and build a squad you actually "
                 "like. THE ARC: build a team -> beat the 8 gyms prepared -> Elite Four -> credits. ")
        # team-health nudge keyed on her ACTUAL party (capability-not-script: she decides what to do).
        party = state.get("party") or []
        if party:
            lvls = sorted(m["level"] for m in party)
            if len(party) < 3 or (len(party) >= 2 and lvls[0] <= max(6, lvls[-1] - 12)):
                model += ("RIGHT NOW your team is thin/lopsided (a strong lead but weak or too few others) — "
                          "that's why tough trainers wall you. Building up a real squad (catch a good new "
                          "teammate, level the weak ones) is how a prepared trainer gets past these. ")
        spine = model + ("THE MAIN QUEST IS A FIXED PATH you're always on: beat the 8 Gym Leaders in order, "
                         "then the Elite Four, then you win the game. ")
        if ng:
            spine += (f"You're on GYM {bc + 1} of 8 — next is {ng['leader']} in {ng['city']}. The road "
                      f"forward leads there: until you've won you are ALWAYS making progress toward the "
                      f"next gym. Grinding/catching/detours are fine — but ONLY with a clear purpose, and "
                      f"then you GET BACK ON THE ROAD to {ng['city']}. Never just circle the same grass.")
        else:
            spine += "All 8 badges are yours — the Elite Four is the final challenge before the credits."
        hist = self._story_so_far(bc)
        return spine + (f" YOUR STORY SO FAR (what you've actually done — your past, not the future): "
                        f"{hist}" if hist else "")

    def _rationale_line(self, state, pick, goals):
        """FIX 3 — a one-sentence 'why am I doing THIS right now' for the dashboard, synthesized from her
        REAL current plan (the concrete pick + her short/long goal), so Jonny can read purposeful-vs-lost
        at a glance. Grounded in actual state (not a canned string); the oracle's verbatim reasoning would
        need an oracle-return change (flagged — firewall)."""
        ng = state.get("next_gym")
        nxt = f" then on to {ng['city']} for {ng['leader']}" if ng else " then on to the Elite Four"
        short = (goals or {}).get("short") or "exploring"
        if pick == "head_to_gym":
            return f"Heading for the next gym{(' — ' + ng['city']) if ng else ''}: {short}."
        if pick in ("battle", "wander_catch"):
            # STRATEGIC UNDERLEVEL-GRIND (Task B): make the WHY explicit on the dashboard — she's leveling
            # the weak members on purpose to cross the wall, not aimlessly farming.
            try:
                pt = self._prep_team_target(state)
                if pt is not None and pick == "battle":
                    weak = self._prep_team_weak(state, pt)
                    who = (weak[0] if len(weak) == 1
                           else (", ".join(weak[:-1]) + " and " + weak[-1])) if weak else "the weak ones"
                    return (f"Team's under-levelled — grinding {who} up to ~L{pt} (fielding them, not my "
                            f"ace) so I can push through,{nxt}.")
            except Exception:
                pass
            return f"{short} — building up here with a purpose,{nxt}."
        if pick == "heal":
            return f"Topping up at a Pokémon Center, then back on the road{nxt}."
        if pick == "stock_up":
            return f"Stocking up on supplies, then continuing{nxt}."
        if pick and pick.startswith("travel:"):
            return f"Routing to a place I need{nxt}."
        if pick == "talk_npc":
            return f"Checking in with a local, then{nxt}."
        return f"{short}.{nxt}"

    def _publish_health(self, macro, state, last_badge_ts, run_start_ts):
        """BATCH 6 PHASE 7 — write the cockpit health snapshot (atomic JSON) the dashboard polls. Game-side
        only; the dashboard merges API spend from the bot's cost-tracker. Best-effort, never blocks the run."""
        import json as _json
        try:
            cp = os.path.join(STATES_CAMPAIGN, CAMPAIGN_SAVE)
            cp_mtime = os.path.getmtime(cp) if os.path.exists(cp) else None
            # HUD enrichment: now-state, one-line objective, her current want, whole-playthrough timer.
            try:
                now_state = "BATTLE" if st.in_battle(self.b) else getattr(self, "_now_state", "EXPLORING")
            except Exception:
                now_state = getattr(self, "_now_state", "EXPLORING")
            ng = state.get("next_gym")
            objective = (f"Beat {ng['leader']} of {ng['city']}" if ng else "Challenge the Elite Four")
            want = ""
            try:
                if self.soul is not None and self.soul.wants:
                    want = str(self.soul.wants[-1])
            except Exception:
                pass
            # PHASE 1 — 3-tier goal (short/medium/long), derived from her real state so Jonny can read
            # "stuck vs strategizing" at a glance + a flat one-liner for compact HUD surfaces.
            goals = self._goal_layers(state)
            plan = " → ".join(p for p in (
                f"Now: {goals['short']}" if goals.get("short") else "",
                f"Next: {goals['medium']}" if goals.get("medium") else "",
                f"Goal: {goals['long']}" if goals.get("long") else "") if p)
            health = {
                "ts": time.time(),
                "progress": macro,                       # GREEN / YELLOW / RED / ABANDONED — the watchdog light
                "place": state.get("place"), "coords": state.get("coords"),
                "badge_count": state.get("badge_count"), "badges": state.get("badges"),
                "party": [f"{m['species']} L{m['level']}" for m in state.get("party", [])],
                # HUD — per-mon card data: sprite (by species_id), name, level, HP, and Gen-3 type badges.
                "party_hud": [{"species": m["species"], "level": m["level"],
                               "hp": m.get("hp"), "maxhp": m.get("maxhp"),
                               "species_id": m.get("species_id"),
                               "types": st.species_types(m.get("species_id"))} for m in state.get("party", [])],
                "party_count": state.get("party_count"), "dex_caught": state.get("dex_caught"),
                "next_gym": state.get("next_gym"),
                "now_state": now_state, "objective": objective, "want": want,
                "goals": goals, "plan": plan,            # PHASE 1 — 3-tier goal (short/medium/long) + flat line
                "rationale": getattr(self, "_rationale", ""),   # FIX 3 — live "why I'm doing this right now"
                "active_objective": (self._active_objective or {}).get("label"),   # the thread she returns to after detours
                # GATE-UNLOCK: the questline she's mid-errand on (e.g. "get the S.S. Ticket") — Jonny reads
                # WHY she's off the direct path. None when not gated.
                "questline": ({"missing": self._active_questline.gate.missing,
                               "doing": (self._active_questline.actionable.human
                                         if self._active_questline.actionable else None),
                               "steps": [s.missing for s in self._active_questline.steps]}
                              if self._active_questline is not None else None),

                "playthrough_s": self._playthrough_elapsed(),
                "last_badge_ts": last_badge_ts,
                "since_last_badge_s": (time.time() - last_badge_ts) if last_badge_ts else None,
                "last_checkpoint_ts": cp_mtime,
                "since_checkpoint_s": (time.time() - cp_mtime) if cp_mtime else None,
                "run_uptime_s": time.time() - run_start_ts,
                "guide_searches": len(getattr(self.guide, "history", []) or []),
            }
            os.makedirs(STATES_CAMPAIGN, exist_ok=True)
            tmp = HEALTH_JSON + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(health, f)
            os.replace(tmp, HEALTH_JSON)
        except Exception as e:
            log(f"   [health] publish skipped: {e}")

    def _continuity_load(self):
        """ADDENDUM D — load her NARRATIVE continuity at the start of a --resume climb: team bonds + wants
        (soul JSON) and the loss/wall history (who's walled her, her grudge's basis). So free-roam resumes
        KNOWING her saga, not just the game RAM. Scoped to the Sherpa free-roam path (the canonical living
        climb) — workshop/scratch runs keep their own soul handling untouched. Launch-independent."""
        if self.soul is not None:
            try:
                self.soul.load(SOUL_JSON)
            except Exception as e:
                log(f"   [soul] !! continuity load failed: {e} (LOUD)")
        try:
            self.strat.load(STRAT_JSON)
        except Exception as e:
            log(f"   [strat] !! continuity load failed: {e} (LOUD)")
        try:
            self.world.load(WORLD_JSON)     # Batch-WORLD: resume KNOWING the map she's built up
        except Exception as e:
            log(f"   [world] !! continuity load failed: {e} (LOUD)")

    def _continuity_save(self):
        """ADDENDUM D — persist her saga alongside a campaign anchor (bonds/wants + loss/wall history),
        so the next GO resumes her relationships AND her grudges, not just her position. Best-effort +
        LOUD on failure (Constraint #3). Scoped to the free-roam (Sherpa) save points."""
        if self.soul is not None:
            try:
                self.soul.save(SOUL_JSON)
            except Exception as e:
                log(f"   [soul] !! continuity save failed: {e} (LOUD)")
        try:
            self.strat.save(STRAT_JSON)
        except Exception as e:
            log(f"   [strat] !! continuity save failed: {e} (LOUD)")
        try:
            self.world.save(WORLD_JSON)     # Batch-WORLD: bank her mental map next to the anchor
        except Exception as e:
            log(f"   [world] !! continuity save failed: {e} (LOUD)")
        # PHASE 4 — push the journey narrative to core Kira (grudge + team feelings + arc), so it
        # persists core-side and surfaces in idle chat / any game, not just this Pokémon process.
        narrative = None
        try:
            narrative = self._journey_narrative()
            self._journey_post(narrative)
        except Exception as e:
            log(f"   [journey] !! continuity-into-core push failed: {e} (LOUD)")
        # RULE 17 SANCTITY — ALSO bank her saga campaign-side so a checkpoint restore resumes her STORY.
        try:
            import json as _json
            if narrative is None:
                narrative = self._journey_narrative()
            with open(JOURNEY_JSON, "w", encoding="utf-8") as f:
                _json.dump(narrative, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"   [journey] !! campaign-side saga bank failed: {e} (LOUD)")
        # RIDE-ALONG 0c (2026-07-06): validate the just-written bundle — schema/encoding/truth. Story
        # corruption must be LOUD at write time, not discovered by a viewer. Never crashes the save path.
        try:
            import sanctity as _sanctity
            _sanctity.validate_bundle(STATES_CAMPAIGN, log=log)
        except Exception as _se:
            log(f"   [sanctity] !! validation skipped: {_se}")

    def _journey_narrative(self):
        """PHASE 4 — assemble her Pokémon-journey continuity for the core seam: WHERE she is, her TEAM
        and how she feels about them, the GARY grudge, and the arc position. Pure reads + the persisted
        soul/strat. This is what lets core Kira speak her journey in idle chat ('how's the Pokémon run
        going?') and resume KNOWING her story. Best-effort; any field that errors is simply omitted."""
        try:
            state = self.read_live_state()
        except Exception:
            state = {}
        party = state.get("party") or []
        lead = party[0]["species"] if party else None
        solo = (state.get("party_count") == 1)
        # team feelings come from the soul bonds/wants (her own words), if present
        bonds = []
        wants = []
        try:
            if self.soul is not None:
                bonds = [b.get("nickname") or b.get("species") for b in (self.soul.bonds or {}).values()]
                wants = list(self.soul.wants or [])
        except Exception:
            pass
        grudge = ""
        try:
            grudge = self.strat.rival_grudge_note()
        except Exception:
            pass
        # a compact, declarative "her story right now" the core can inject as lived experience
        bits = []
        if lead:
            if solo:
                lead_line = ("a SOLO run with just her " + lead
                             + " (a wild one she bonded with), no other team yet")
            else:
                lead_line = "her team led by " + lead
            bits.append("She's playing Pokémon FireRed as Kira — " + lead_line + ".")
        if bonds and not solo:
            bits.append(f"Her team: {', '.join(str(x) for x in bonds if x)}.")
        if state.get("arc"):
            bits.append(state["arc"])
        if grudge:
            bits.append(grudge)
        if wants:
            bits.append(f"What she wants right now: {wants[0]}.")
        return {
            "summary": " ".join(bits),
            "place": state.get("place"),
            "badge_count": state.get("badge_count"),
            "party_count": state.get("party_count"),
            "lead": lead,
            "solo": solo,
            "grudge": grudge,
            "arc": state.get("arc"),
        }

    def _gain_sig(self):
        """ADDENDUM A — the IRREVERSIBLE-progress fingerprint: (badges, party size, dex caught, team
        LEVEL-SUM). Used to guarantee the escape-hatch never reloads PAST a real gain. LEVEL-SUM added
        2026-07-05 after the level-blind sig cost a full bench-level TWICE in one night: run-4 reverted a
        13-min L16/16 weak-grind, then run-5 reverted Ivysaur L30→28 + the Route-25 position — 154+ wins
        of XP registered as ZERO gain because levels weren't in the sig. Level-sum is monotonic (levels
        never go down), so every level-up now protects itself AND banks a fresh gain-seam ring checkpoint
        (each level = a piton). Pure reads."""
        badges = sum(1 for i in range(8) if self.has_badge(0x820 + i))
        party = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        dex = ram.pokedex_owned_count(self.b) or 0
        levels = sum(self._party_levels() or [0])
        return (badges, party, dex, levels)

    def _deep_wedge_revert(self):
        """PHASE 1 — the floor beneath the escape-hatch. The recent-good reload is exhausted and the
        world is STILL frozen (a structural wedge that re-wedges from the recent-good). Revert
        PROGRESSIVELY FURTHER BACK through the gain-seam ring: newest gain-seam first (most progress
        kept), then older on each repeat, until a checkpoint actually clears the wedge. Banks the
        current (even-wedged) state first so nothing is truly lost, then surfaces the revert
        in-character. Returns True if it reverted, False if the ring is exhausted (-> ABANDON)."""
        idx = len(self._safe_ring) - 1 - self._deepwedge_reverts
        if idx < 0:
            log(f"   [roam] deep-wedge ring EXHAUSTED ({self._deepwedge_reverts} reverts, "
                f"{len(self._safe_ring)} banked) — no clean checkpoint left to fall back to")
            return False
        entry = self._safe_ring[idx]
        try:
            # bank the current (wedged) state first — never blind-overwrite (an unbanked shiny survives a misfire)
            try:
                os.makedirs(STATES_CAMPAIGN, exist_ok=True)
                bpath = os.path.join(STATES_CAMPAIGN, f"pre_deepwedge_{int(time.time())}.state")
                with open(bpath, "wb") as f:
                    f.write(self.b.save_state())
            except Exception as _be:
                log(f"   [roam] deep-wedge pre-revert backup skipped: {_be}")
            self.b.load_state(entry["state"])
            self._deepwedge_reverts += 1
            log(f"   [roam] !!!! DEEP-WEDGE REVERT #{self._deepwedge_reverts}: escape-hatch spent + still "
                f"frozen -> reverted to safe checkpoint {entry['label']} (gain {entry['gain']})")
            # IN-CHARACTER COVER — watchable, never a silent blink (Constraint: announce, don't illusion-break)
            self.on_event("ugh — something properly glitched out on me there. let me just back up to where "
                          "things were working and pick it up again.", kind="recover", tier=2)
            self._wait_overworld()
            self._save_campaign("post_deepwedge_revert")
            return True
        except Exception as e:
            log(f"   [roam] !! DEEP-WEDGE REVERT FAILED ({e}) — falling through to ABANDON (LOUD)")
            return False

    def _fire_deadman_alert(self, state):
        """PHASE 2 — DEAD-MAN'S SWITCH. Deep-wedge recovery itself failed (escape-hatch AND the whole
        ring exhausted, still frozen): the run is genuinely abandoned and needs a human. Ping Jonny
        through the bot's alert seam (POST /cmd/pokemon_alert -> Discord webhook + loud log) so
        'autonomous overnight' never silently becomes 'abandoned at 3am'. Cheap insurance; never raises."""
        place = (state or {}).get("place", "an unknown area")
        coords = (state or {}).get("coords")
        badges = (state or {}).get("badge_count", "?")
        msg = (f"Kira's Pokémon run is ABANDONED and needs a hand — wedged at {place} {coords}, "
               f"{badges} badges. Escape-hatch + the deep-wedge ring are both exhausted; she's stopped "
               f"and is waiting. Last clean backups are in states/campaign/ (pre_deepwedge_*.state).")
        log(f"   [roam] !!!! DEAD-MAN'S SWITCH FIRING: {msg}")
        try:
            self._alert_post(msg)
        except Exception as e:
            log(f"   [roam] !! dead-man's-switch alert POST failed (non-fatal, logged loud above): {e}")

    def _escape_hatch_reload(self):
        """ADDENDUM A — LAST-RESORT wedge escape for a long run. Caller has already confirmed a GENUINE
        fingerprint-frozen wedge (sustained RED despite the forced-heal recovery — NOT idle). Steps, in
        the exact order Jonny's anti-misfire worry demands:
          1. BANK current live state to a timestamped pre-reload backup FIRST (never blind-overwrite — an
             unbanked shiny must survive even a misfire).
          2. GAIN GUARD: if the live gain-fingerprint shows MORE than the last-good snapshot (a badge /
             teammate / catch since), do NOT reload (it would rewind past a real gain) — re-anchor current
             as the new good state and decline (let ABANDON handle it loudly).
          3. Otherwise reload the last KNOWN-GOOD state, LOUD, and continue.
        Returns True if it reloaded (caller breaks the RED streak), False if it declined."""
        if not self._last_good_state:
            log("   [roam] !! ESCAPE-HATCH: no known-good snapshot captured yet — cannot reload safely (declining)")
            return False
        # 1) bank current FIRST — never lose what's live (the unbanked-shiny nightmare)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(STATES_CAMPAIGN, f"pre_reload_{ts}.state")
        try:
            os.makedirs(STATES_CAMPAIGN, exist_ok=True)
            tmp = backup + ".tmp"
            with open(tmp, "wb") as f:
                f.write(self.b.save_state()); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, backup)
            log(f"   [roam] !! ESCAPE-HATCH: banked CURRENT state -> {backup} (nothing live is lost, even on a misfire)")
        except Exception as e:
            log(f"   [roam] !! ESCAPE-HATCH: could not bank current state ({e}) — ABORTING reload (safety: "
                f"never reload if we couldn't back up first)")
            return False
        # 2) GAIN GUARD — never rewind past a real gain (badge / teammate / fresh catch)
        cur_gain, good_gain = self._gain_sig(), (self._last_good_gain or (0, 0, 0, 0))
        if any(c > g for c, g in zip(cur_gain, good_gain)):
            log(f"   [roam] !! ESCAPE-HATCH: current gain {cur_gain} EXCEEDS last-good {good_gain} — a real "
                f"gain happened since (badge/teammate/catch). REFUSING to reload past it; re-anchoring current "
                f"as the new known-good and declining (ABANDON will surface for a human instead).")
            self._last_good_state = self.b.save_state()
            self._last_good_gain = cur_gain
            self._save_campaign("escape_reanchor")
            return False
        # 3) reload the last known-good state — the actual escape
        try:
            self.b.load_state(self._last_good_state)
            log(f"   [roam] !!!! ESCAPE-HATCH FIRED: genuine wedge — RELOADED last known-good state "
                f"(gain {good_gain}); current backed up to pre_reload_{ts}.state. Continuing the climb. (LOUD)")
            self.on_event("something got me properly stuck back there — I'm backing up to where I knew what "
                          "I was doing and picking it up from there.", kind="recover", tier=2)
            self._wait_overworld()
            self._save_campaign("post_escape_reload")
            return True
        except Exception as e:
            log(f"   [roam] !! ESCAPE-HATCH: reload FAILED ({e}) — staying on current state (already backed up)")
            return False

    MAX_BLACKOUT_RETRIES = 12     # per segment — a thin solo roster needs several Miguel attempts;
                                  # each retry re-walks (more trainer XP) + re-heals, so it converges.
                                  # "die as many times as needed and still finish" — survival variance.

    @staticmethod
    def _is_blackout(result):
        """A segment result that is a SURVIVAL death (recoverable by respawn+retry), not a nav stuck.
        run() encodes these as 'stopped:<kind>:battle_loss' or 'stopped:<kind>:blackout'."""
        return isinstance(result, str) and (result.endswith("battle_loss") or result.endswith("blackout"))

    def _settle_after_blackout(self):
        """After a blackout the game whites out + respawns at the last-healed Center (auto-heals the
        party). Advance past the respawn animation + any closing text so we're back on the overworld,
        ready to re-run the segment from the Center."""
        self.b.set_input_owner("agent")
        for _ in range(180):                  # let the whiteout + respawn walk-in play out
            self.b.run_frame(); self.render()
        self._drain_overworld(label="blackout-respawn")
        for _ in range(40):
            self.b.run_frame()
        # The respawn drops her INSIDE a building (the Pallet house / any Pokémon Center). Get her
        # back to the walkable overworld before the segment re-nav (which can't path out of an
        # interior). General capability — works for EVERY PC/town (game-wide for the 24/7 vision).
        self._exit_to_overworld()
        log(f"   BLACKOUT-RECOVERY: respawned at {tv.map_id(self.b)}@{tv.coords(self.b)} "
            f"HP={self.lead_hp()}")

    def _exit_to_overworld(self, max_tries=5):
        """From INSIDE any building (PC / house / Mart / lab — map group != 3, the Kanto overworld),
        return to the walkable overworld. The general recovery primitive every blackout respawn needs.
        Tries the south exit door; falls back to walking DOWN onto the entrance mat (PC/house mats fire
        on a step-on from the north). Returns True once on the overworld (group 3)."""
        # 2026-07-06 REWRITE (the TWO-STORY HOUSE class, runs 12-13): the old 'south door' preference
        # warps 1F<->2F forever in a stair house (the stairs door IS the southmost warp on both floors).
        # Now: walk this floor's warp events NEAREST-FIRST, but each (floor, warp) is TAKEN once per
        # call — after the stairs are consumed, the next candidate is the actual street door. Entry per
        # kind: walk-onto (mats), directional stairs (the 0x6C-0x6F table), door ritual.
        self.b.set_input_owner("agent")
        taken = set()
        for _ in range(max_tries * 3):
            if tv.map_id(self.b)[0] == 3:
                return True
            before = tuple(tv.map_id(self.b))
            cur = tuple(tv.coords(self.b))
            ws = tv.read_warps(self.b)
            cands = [tuple(w[0]) for w in ws]
            # STREET-FIRST (2026-07-06, the Vermilion Center/Cable-Club stall): each warp's DEST is
            # already known — a warp straight to the overworld (group 3) beats every interior hop.
            # Nearest-first alone ping-ponged the two Center floors via the escalator and then burned
            # travel budgets on the attendant-blocked Cable-Club room warps.
            street = {tuple(w[0]) for w in ws if w[1][0] == 3}
            fresh = [w for w in cands if (before, w) not in taken]
            # within a tier, an ACTUATABLE warp (a behavior the directional table can fire) beats a
            # dead arrival-mat: the Center's exit row is (6,8)/(7,8)/(8,8) but only (7,8) is the 0x65
            # arrow that actually fires — trying the dead mats first burned a travel leg each.
            cands = sorted(fresh or cands,
                           key=lambda t: (t not in street,
                                          self._tile_behavior(*t) not in self._WARP_ENTRY,
                                          abs(t[0] - cur[0]) + abs(t[1] - cur[1])))
            moved = False
            for wt in cands:
                taken.add((before, wt))
                # a directional/escalator tile has a REQUIRED entry side + a delayed fire — the
                # dedicated primitive knows both; blind travel to the tile wedges on the blocked side
                if self._tile_behavior(*wt) in self._WARP_ENTRY:
                    if self._enter_directional_warp(wt) and tuple(tv.map_id(self.b)) != before:
                        moved = True
                        break
                self.trav.travel(target_map=None, arrive_coord=wt, max_steps=160, max_seconds=60)
                for _ in range(30):
                    self.b.run_frame()
                if tuple(tv.map_id(self.b)) != before:
                    moved = True
                    break
                if self._enter_directional_warp(wt) and tuple(tv.map_id(self.b)) != before:
                    moved = True
                    break
                if self.enter_warp(pick=wt, budget_s=60) == "warped" \
                        and tuple(tv.map_id(self.b)) != before:
                    moved = True
                    break
            if not moved:
                for _ in range(10):                   # last resort: the classic DOWN-mat walk
                    self.b.press("DOWN", 8, 8, self.render, owner="agent")
                    for _ in range(8):
                        self.b.run_frame()
                    if tuple(tv.map_id(self.b)) != before:
                        break
                if tuple(tv.map_id(self.b)) == before:
                    break                             # nothing on this floor moves us — give up LOUD
            if tuple(tv.map_id(self.b)) != before:
                log(f"   EXIT-BUILDING: {before} -> {tv.map_id(self.b)}@{tv.coords(self.b)}")
        if tv.map_id(self.b)[0] != 3:
            log(f"   !! EXIT-BUILDING: still inside {tv.map_id(self.b)}@{tv.coords(self.b)} - LOUD")
        return tv.map_id(self.b)[0] == 3

    def run_segments(self, segments, resume=True, mode="workshop"):
        """Play SEGMENTS in order, auto-checkpointing after each.
        mode='workshop' (default): resume-from-furthest in states/workshop/, bank there, NEVER touch
            states/kira/. Our skip-elimination scaffolding — jump anywhere, scratch freely.
        mode='show': the canonical/recordable spine. Banks into states/kira/ and resumes from a kira
            save if present (else fresh boot). Any GATE_NEEDS_STATE fallback logs a LOUD SHOW-MODE
            SKIP VIOLATION and is counted — a clean SHOW run has ZERO violations + zero skips."""
        self.show_mode = (mode == "show")
        self.ckpt_dir = STATES_KIRA if self.show_mode else STATES_WORKSHOP
        self._show_violations = 0
        log(f"RUN_SEGMENTS: ** {'SHOW' if self.show_mode else 'WORKSHOP'} MODE ** "
            f"-> banks to {os.path.basename(self.ckpt_dir)}/"
            + ("; any GATE load = LOUD violation (target 0)" if self.show_mode
               else "; states/kira/ is READ-ONLY here"))
        start, resume_cp = 0, None
        if resume:
            for i, seg in enumerate(segments):
                cp = os.path.join(self.ckpt_dir, seg.checkpoint) if seg.checkpoint else None
                if cp and os.path.exists(cp):
                    start, resume_cp = i + 1, cp
        if start >= len(segments):
            log(f"RUN_SEGMENTS: all {len(segments)} segment checkpoint(s) already present — done")
            return "all_segments_complete"
        if resume_cp:
            log(f"RUN_SEGMENTS: RESUME — checkpoint {os.path.basename(resume_cp)} present, skipping "
                f"{start} done segment(s); loading past them")
            with open(resume_cp, "rb") as f:
                self.b.load_state(f.read())
            for _ in range(40):
                self.b.run_frame()
            self.b.set_input_owner("agent")
        # CONTINUITY: a SHOW run restores her Pokémon-self (roster bonds + wants) so the stream
        # resumes with her relationships intact. SHOW-ONLY -> workshop/scratch runs never read or
        # write the kira-lineage continuity (same firewall as the save lineages).
        if self.show_mode and self.soul is not None:
            self.soul.load(os.path.join(STATES_KIRA, "pokemon_soul.json"))
        for i in range(start, len(segments)):
            seg = segments[i]
            log(f"==== SEGMENT {i + 1}/{len(segments)}: {seg.name}  "
                f"({len(seg.objectives)} objective(s)) ====")
            if self.soul is not None:                       # SOUL HOOK (surface_want): a new beat is a
                log(f"   [soul] surface_want FIRE -> context={seg.name}")  # moment a want could surface;
                self.soul.surface_want({"segment": seg.name, "map": tv.map_id(self.b)})  # oracle decides
            # BLACKOUT-RECOVERY (core resilience): a death (Miguel/cave/Route 3 with a thin solo
            # roster) is NOT a dead end — the game respawns + heals at the last Center, and we RE-RUN
            # the segment (the objectives re-navigate from the respawn point). "Die as many times as
            # needed and still finish." A PERMANENT stuck (can't proceed, not dying) still stops loud.
            attempts = 0
            while True:
                result = self.run(seg.objectives)
                if result == "complete":
                    break
                if self._is_blackout(result) and attempts < self.MAX_BLACKOUT_RETRIES:
                    attempts += 1
                    log(f"   ~~~~ BLACKOUT-RECOVERY: '{seg.name}' died ({result}); retry "
                        f"{attempts}/{self.MAX_BLACKOUT_RETRIES} — respawned + healed, re-running segment")
                    self.on_event("knocked out — back to the Center, dust off, and right back at it")
                    self._settle_after_blackout()
                    continue
                log(f"!! RUN_SEGMENTS STOP at segment '{seg.name}': {result}"
                    + (f" (blackout retries exhausted: {attempts})" if self._is_blackout(result) else ""))
                return f"segment_stopped:{seg.name}:{result}"
            self._save_checkpoint(seg.checkpoint)
            if self.show_mode and self.soul is not None:    # bank her Pokémon-self continuity (kira lineage)
                self.soul.save(os.path.join(STATES_KIRA, "pokemon_soul.json"))
            log(f"==== SEGMENT '{seg.name}' COMPLETE"
                + (f" (survived {attempts} blackout(s))" if attempts else "") + " ====")
        if self.show_mode:
            verdict = "ZERO skips — clean AUTO spine" if self._show_violations == 0 \
                else f"{self._show_violations} SKIP VIOLATION(S) — NOT yet zero-skip"
            log(f"RUN_SEGMENTS: SHOW MODE COMPLETE — {verdict}")
        log("RUN_SEGMENTS: ALL SEGMENTS COMPLETE")
        return "all_segments_complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--boot", default="after_pick_bulbasaur.state")
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    b = Bridge(ROM)
    boot_path = resolve_state(args.boot) or os.path.join(STATES, args.boot)
    with open(boot_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    log(f"booted {args.boot}: map={tv.map_id(b)} coords={tv.coords(b)}")

    def battle_runner():
        return BattleAgent(b, on_event=lambda s, **k: log(f"[battle] {s}"),
                           render=lambda: None).run(max_seconds=120)

    camp = Campaign(b, battle_runner=battle_runner)
    outcome = camp.run(build_objectives())
    print("\n" + "=" * 60)
    print(f"   CAMPAIGN RESULT: {outcome}")
    print(f"   final: map={tv.map_id(b)} coords={tv.coords(b)}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
