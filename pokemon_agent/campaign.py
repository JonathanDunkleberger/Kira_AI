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
from pokemon_planner import StrategicPlanner, TeamPlanner  # noqa: E402  (2026-07-08 mega-batch; TeamPlanner 2026-07-09 mission-pivot)
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
# ROAD BENCH XP (2026-07-11, PASS 3 team-depth NEW#1 — the mid-game "bench never levels" root fix).
# The participation-XP switch (battle_agent.PROTECT_LEAD_GRIND) was armed ONLY inside a dedicated
# grind_weak_members() session — which push-when-carrying almost never triggers mid-game — so the bench
# banked ZERO XP from the ~8 road trainer wins between towns while the ace ran away (NS#2 diagnosis: won
# Route24→Vermilion, lead L28→32, bench FROZEN L14/L14). This arms the SAME proven switch tightly around
# each forward-march leg (head_to_gym / travel:) when a bench member is under its milestone prep target:
# the weakest under-milestone mon LEADS (so it's "sent out" = XP-eligible), the participation switch
# fields the ace turn 1 so the weak mon never takes a hit, and the ace does the KO. The bench levels
# ORGANICALLY as she marches — no grind-wall (watchable: the whole team fights its way to the gym) — so
# she arrives at each gym near-milestone instead of L14, which also dissolves the grind-spot-adequacy
# wall. The weak lead is restored to the ace the instant the leg ends (never persists into readiness/heal
# reads). Fail-safe: an unconfirmed switch just fights on; disarm always restores the ace. Touches the
# wedge-prone in-battle switch (Tier-1 #5) so it's flag-gated for instant revert.
ROAD_BENCH_XP_ENABLED = os.getenv("POKEMON_ROAD_BENCH_XP", "1") != "0"
# The participation switch costs the ACE one free enemy hit per battle (the enemy attacks the mon we
# switch IN). Over a no-heal multi-trainer LEG that extra chip can tip a marginal gauntlet into a
# whiteout — observed on a TOP-HEAVY hand-bank (L48 ace soloing Route 13 while an L9-15 bench can't
# share the load). So only start a bench-XP leg when the ace has HP headroom to absorb it; a dinged ace
# defers to the heal path (needs_heal only fires <0.50 on ANY member — this is a higher, ACE-specific
# floor). On a fresh organically-built run the bench shares the gauntlet, so this rarely bites.
ROAD_XP_ACE_HP_FLOOR = float(os.getenv("POKEMON_ROAD_XP_ACE_HP_FLOOR", "0.6"))
# PREP-FOR-E4 band (2026-07-11, PASS 3 team-depth): at all 8 badges the whole party is floored to the
# team-plan's E4 milestone (~L55) so the bench survives the Center-less five-fight gauntlet (NS13/NS14:
# a top-heavy team where the ace solos then dies at Lance/Champion). A member counts as LEVELABLE only
# within this many levels of the target — far-below chaff (box fodder, unbuilt Tier-1 #15) is excluded so
# it never pins an unwinnable grind, keeping the E4-prep livelock-proof (it retires → forward to VR once
# every real member crosses or stalls out). 25 cleanly separates the real L40+ team from L9-14 fodder.
E4_PREP_BAND = int(os.getenv("POKEMON_E4_PREP_BAND", "25"))
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
# SEVERELY-LOPSIDED BENCH grind (2026-07-11, PASS 3 NS#6 — team-depth lever (a), the binding wall).
# NS#5 root: the DEDICATED bench grind (the 'battle' pick -> grind_weak_members) fired 0x the whole
# badge-3->4 climb — the proactive prep pin was ALWAYS out-voted by the forward-drive (head_to_gym),
# because a solo L48 carry can march the road forever so the oracle never STOPS to fix an L10-14 bench.
# Result: abra frozen at L13 (never evolves to Kadabra, the Koga/Sabrina answer), four badges in. This
# forces ONE bounded dedicated grind stint when the bench is SEVERELY lopsided vs BOTH the gym milestone
# AND the ace (the solo-carry + dead-weight shape) by suppressing head_to_gym so 'battle' wins — exactly
# how a real player stops to train a lopsided team (constitution-aligned, narrated). PARK-PROOF by three
# independent bounds: (1) it only fires when the +6 pin is armed (_prep_team_target non-None), which
# retires after ONE +6 bite and won't re-arm until the NEXT badge (NS#2 milestone machinery); (2) the
# milestone is marked done after a completed stint (_lopsided_grind_done) so a stalled/un-levelable bench
# releases immediately instead of looping; (3) no-reachable-grass hits the PREP STAND-DOWN. So it's <= one
# +6 stint per badge — never the celadon_run1 27-level marathon that parked the road. Flag-gated (instant
# revert); touches only action-menu dominance (no new battle code). The gaps below select SEVERE (solo
# carry) from MODEST (bench trails ~10 — road-bench-XP finishes that organically without a full stop).
LOPSIDED_GRIND_ENABLED = os.getenv("POKEMON_LOPSIDED_GRIND", "1") != "0"
LOPSIDED_MS_GAP = int(os.getenv("POKEMON_LOPSIDED_MS_GAP", "12"))    # floor >= this far under milestone
LOPSIDED_ACE_GAP = int(os.getenv("POKEMON_LOPSIDED_ACE_GAP", "15"))  # ace towers >= this over the floor
# BENCH-TO-MILESTONE climb (2026-07-11, PASS 3 NS#10 — team-depth lever, the Koga wall). NS#9 proved the
# ONE-bite-per-badge lopsided grind (36b4998) lands the bench ~25 levels UNDER the gym milestone (L20 vs
# Koga's L45) — the type-answers (Mr. Mime Psychic x4, Diglett Ground) FAINT at L23 and she LOSES Koga.
# Root: BOTH the +6 prep pin (retires after one bite, won't re-arm same milestone) AND _lopsided_grind_done
# cap the climb at a single +6 bite. This lets the bench CLIMB toward the milestone in repeated +6 bites,
# gated by OBSERVED grind productivity so it's park-proof WITHOUT a level KB: a bite that gains < BITE_MIN
# levels marks THIS map a poor spot for this milestone -> releases head_to_gym so she MARCHES to better
# grass (the route grass near a gym is level-appropriate) instead of a stationary weak-grass marathon (the
# celadon_run1 27-level park). The milestone is marked truly done only when the floor reaches within CLOSE
# of it (goal met) — never after a single bite. Flag-gated; default OFF preserves 36b4998 byte-identically.
BENCH_TO_MILESTONE = os.getenv("POKEMON_BENCH_TO_MILESTONE", "0") != "0"
BENCH_MS_CLOSE = int(os.getenv("POKEMON_BENCH_MS_CLOSE", "6"))     # floor within this of milestone => done
BENCH_BITE_MIN = int(os.getenv("POKEMON_BENCH_BITE_MIN", "2"))    # a bite gaining < this many levels = poor spot
# PREP STAND-DOWN map-scoping (2026-07-11, PASS 3 NS#11 — the DEADLOCK fix). The PREP STAND-DOWN
# (_prep_dry >= 2) drops the 'train first' plan when repeated grind attempts from a position find NO
# reachable grass. Its comment claims it "resets the moment any grass is actually reached" — but the ONLY
# reset (grind_weak_members returning ready/ok) sits INSIDE the grind path, which _prep_dry>=2 itself gates
# off (both the +6 prep pin AND the lopsided/bench-to-ms dominance). So a dry spell (e.g. grassless Celadon)
# latches _prep_dry at 2 PERMANENTLY: she then marches the whole Koga chain (Rocket Hideout -> Poké Flute ->
# Route 12/13/14/15 -> Fuchsia) past the LEVEL-APPROPRIATE Route 13-15 grass without ever stopping to grind,
# arriving at Koga (and stalling post-Koga on the gated Route-15 trainers) with a frozen L15-20 bench behind
# an L50+ ace. FIX: reset _prep_dry the moment she stands on GENUINELY-USABLE new grass — reachable grass
# here (_reachable_grass) AND this map not already marked grind-dead (the fragile one-way-strand / no-grass
# memory). That excludes grassless towns (no reset, no wasted attempt), gated grass behind a ledge
# (_reachable_grass None -> no reset -> no Fuchsia<->Route-15 shuttle), and fragile-strand maps (_grind_dead
# -> no same-map spin), while re-arming the bench-leveling machinery on the good Route 13-15 grass. Default
# ON; disable with POKEMON_PREP_DRY_RESET=0.
PREP_DRY_RESET = os.getenv("POKEMON_PREP_DRY_RESET", "1") != "0"
# CROSS-MAP KEEPER ROUTER (2026-07-11, PASS 3 team-depth NEW#2): the on-map catch un-gate
# (_plan_wants_prebuild / _plan_keeper_target) only grabs a planned keeper she happens to be STANDING on;
# a keeper on a nearby route she isn't on is marched past (the erika_done look-ahead: the planner wants
# 'abra -> alakazam' the whole way to Fuchsia but never fetches it -> she arrives with a leveled-chaff
# bench, no coverage). This offers a BOUNDED detour to a nearby reachable map that HOSTS the DUE keeper,
# then the existing on-map machinery catches it. Bounded (<= MAX_HOPS world-graph hops so it grabs
# corridor-adjacent keepers, never a whole-map backtrack), room-gated (party < 6; the full-party box swap
# is a separate unbuilt piece, Tier-1 #15), plan-gated, fail-closed. DEFAULT ON (NS#38, 2026-07-11): a
# behavioural look-ahead on bill_house_noabra (party-3, 1 warp hop to Route 25, NO gauntlet) proved the
# clean end-to-end fetch+catch — FETCH-KEEPER routed Bill's house -> Route 25, force-caught the target
# Abra (party 3->4), the plan advanced to the next keeper (diglett), and when THAT keeper's map was
# unreachable the router returned None (fetch_keeper un-offered) so she fell through to head_to_gym with
# NO livelock. Full-party (>=6) and post-game both short-circuit to None, so the canonical Champion
# timeline is unaffected. Disable with POKEMON_KEEPER_ROUTER=0. (Detour watchability / over-backtrack is a
# LIVE-EYES tuning item — turn MAX_HOPS down to 4 / STALL_CAP 3->2 if live detours read too far.)
KEEPER_ROUTER_ENABLED = os.getenv("POKEMON_KEEPER_ROUTER", "1") != "0"
KEEPER_ROUTER_MAX_HOPS = int(os.getenv("POKEMON_KEEPER_ROUTER_MAX_HOPS", "6"))
# After this many consecutive no-progress fetch legs on one keeper target, retire it (session-scoped
# _keeper_unreach) so a keeper the learned-graph traveler can't actually reach never livelocks the roam
# (the route3_caught look-ahead: world.route said 'reachable' but the executor couldn't ride it -> MACRO
# RED spin). She falls through to head_to_gym; the keeper is re-tried fresh only after a roster change.
KEEPER_ROUTER_STALL_CAP = int(os.getenv("POKEMON_KEEPER_ROUTER_STALL_CAP", "3"))
# STATIC-CONNECTION KEEPER ROUTE (NS#40): the learned world graph only knows VISITED maps, so a keeper HOST
# entered by a door from an as-yet-unwalked corridor (Diglett's Cave off Route 11/Route 2) is invisible to
# world.route -> _reachable_keeper_host returns None forever -> she never detours there -> she never visits
# it (the NS#39 keeper-reachability chicken-and-egg, ROOT-confirmed: NOT timing, an unvisited-map gap). This
# arms the STATIC pass: gamedata/frlg_connections.json says 'area X is entered from overworld GATEWAY(s) Y',
# so the router can plot a BOUNDED path to a reachable gateway, ride there via the learned graph, then step
# through the live-read door into the host and catch. DEFAULT OFF pending the surge_done_kit behavioural
# proof (re-probe _reachable_keeper_host returns (1,36) from Vermilion, then a look-ahead where she actually
# detours + catches Diglett). Isolated to the keeper router (does NOT touch the shared world.route -> no
# route3_caught livelock reintroduction; the _next_step_rideable guard still gates every offered hop).
KEEPER_STATIC_ROUTE_ENABLED = os.getenv("POKEMON_KEEPER_STATIC_ROUTE", "1") != "0"
# Per-floor wander budget (wall-clock seconds) for the cave step-encounter catch (NS#40 _cave_fetch_catch):
# how long to hunt ONE cave sub-map before declaring it barren and DESCENDING an internal warp to the next
# floor (Diglett's Cave (1,38) entrance vestibule has no wild table; the Diglett floor is (1,37)/(1,36)).
KEEPER_CAVE_FLOOR_WANDER_S = int(os.getenv("POKEMON_KEEPER_CAVE_FLOOR_WANDER_S", "45"))
# PC/BOX (Tier-1 #15, NS#38): the pairing gap for the FULL-party case — the keeper router only fires when
# party<6, so a full team of early-catch chaff (erika_done: Venusaur + Rattata/Spearow/Ekans/Meowth/Pidgey)
# can NEVER swap in a planned coverage keeper (abra/diglett). box_chaff deposits the lowest-value off-plan
# chaff at the current city's Center PC -> party 6->5 -> the router's room-gate opens -> the keeper is added.
# DEFAULT OFF: the PC deposit menu is menu-nav-on-the-long-core (wedge-prone); arm with POKEMON_PCBOX=1 after
# a live grab-and-look confirms the actuation on the show build. Decision/selection logic + a headless deposit
# are verified in recon_deposit_check.py. Never boxes an on-plan line (planner._is_target_line) or the lead.
PCBOX_ENABLED = os.getenv("POKEMON_PCBOX", "0") != "0"
# Wall-clock ceiling for ONE grind_weak_members() call (a single tick's worth of weak-grinding). grind()
# itself caps at 480s/member; this outer bound stops a multi-weak-member loop from running away on a tick.
GRIND_WEAK_BUDGET_S = int(os.getenv("POKEMON_GRIND_WEAK_BUDGET_S", "600"))
# PER-MON PROBE for the FRAGILE bench-grind (2026-07-09, the Rattata over-grind wedge): a genuinely
# too-weak bench mon (Rattata/Spearow L14 on Route 4, ace L29) earns no XP, but grind() ran the FULL
# 480s before returning — so the zero-gain STALL mark (grind_weak_members) took ~8 min PER hopeless mon,
# and with a 600s outer budget she burned entire look-ahead ticks (never advancing to Vermilion). A
# short probe caps each attempt: a levelable mon (a fresh catch like Ekans) still climbs in bites and
# gets re-fielded; a hopeless one stalls out in ~one probe and she pushes on with the strong core. This
# is also the more HUMAN read — a real trainer gives a weak mon a few fights, sees it's not taking, and
# moves on rather than grinding it for 8 minutes.
GRIND_WEAK_PROBE_S = int(os.getenv("POKEMON_GRIND_WEAK_PROBE_S", "180"))
# SPLIT-ROUTE HEAL-THRASH CAP (2026-07-09 shift 13, the S.S. Anne / Route-4 recovery livelock): grind()
# heals on need_heal and loops. On a route whose OWN Center is UNREACHABLE (Route 4's east grass, split
# from its PC (12,5) by the Mt-Moon ledge), each heal is an EXPENSIVE cross-city EXCURSION. A useless
# grind (ace L29 vs L3-6 wilds -> ~0 XP, target L31) then heal-thrashes for the whole 480s budget — the
# captain_fix look-ahead logged 22 excursions + 582 travel legs + ~0 net XP before the deep-wedge ring
# finally reverted (an unwatchable ~1000-game-second spin). Cap the EXCURSIONS-WITHOUT-LEVEL-GAIN per
# grind() call: N expensive excursions and ZERO levels earned => this grass is un-grindable from here ->
# stop LOUD, mark the map grind-dead, surface so the caller picks Center-reachable grass or stands down.
# Distinguishes the ace-thrash (0 progress) from a legit deep grind (grind_pre_brock earns levels per
# excursion, so lvl() rises and the cap never trips). Fail-open: healable routes rarely excursion at all.
GRIND_HEAL_EXCURSION_CAP = int(os.getenv("POKEMON_GRIND_HEAL_EXCURSION_CAP", "4"))
# GRIND GRASS-UNREACHABLE CAP (NS#5 Route-10 (11,79) livelock): grind() paces travel(arrive_coord=grass);
# when the grass she "knows" on this map sits behind an INTERNAL split (Route 10's north grass is cliff-
# sealed from the Lavender-side south pocket — the only crossing is Rock Tunnel), EVERY waypoint travel
# returns "no_path" fast, but the inner loop only caught battle_loss/need_heal — so it re-fired the same
# unreachable waypoints for the whole 480s budget (ns4_koga.log: 4,087 identical TRAVEL WEDGEs at (11,79)).
# Cap the no_path-WITHOUT-LEVEL-GAIN count per grind() call: N fast grass-unreachable travels and ZERO
# levels earned => this map's grass can't be reached from here -> mark grind-dead, surface no_safe_grass so
# the caller stands down and picks a Center-reachable spot / gets back on the road. Fail-open: any level
# gain (an encounter DID fire, so grass IS reachable) means lvl()>lvl_start and the cap can never trip.
GRIND_NOPATH_CAP = int(os.getenv("POKEMON_GRIND_NOPATH_CAP", "6"))
# ACE-DOWN GUARD (2026-07-11 ns11 Route-11 livelock): a FRAGILE bench-grind fields a weak mon and switches to
# the ACE to do the killing — the ace tanks a free enemy hit each wild with no inter-battle heal. If it wears
# down to a FAINT, the grind's premise is void: the next wild auto-sends a moveless bench mon (a Teleport-only
# Abra) that can't win AND can't switch (the white-box menu impostor), so every step re-encounters a battle
# that never resolves ('stuck') → an unwinnable-grass livelock that ALSO traps the heal excursion (it can't
# cross the grass to a Center). Heal the moment the ace dips into the faint-risk band — while it's STILL ALIVE
# to fight the escape battles out of the grass. Fail-open: a healthy ace one-shots wilds and never dips here.
GRIND_ACE_BAIL_FRAC = float(os.getenv("POKEMON_GRIND_ACE_BAIL_FRAC", "0.34"))
ACE_BAIL_ON = os.getenv("POKEMON_ACE_BAIL", "1") == "1"   # default ON; one-char revert to prior behavior
# GRIND-SPOT LEVEL AWARENESS (NS#5, PASS-3 grind-efficiency lever a) — a map whose wild_max is more than
# GRIND_POOR_GAP below the team's grind target gives ~0 XP (the NS#1/#14 E4-prep stall). The KB reader
# (_grind_wild_band / _grind_inadequate) + this gap land now (decision-verified); the picker WIRING that
# marks such a map grind-inadequate + prefers a reachable higher-level spot is flag-gated + verify-gated.
GRIND_POOR_GAP = int(os.getenv("POKEMON_GRIND_POOR_GAP", "18"))
GRIND_SPOT_LEVELAWARE = os.getenv("POKEMON_GRIND_SPOT_LEVELAWARE", "0") == "1"  # default OFF until live-verified
try:
    import field_moves as fm          # noqa: E402  (capability reads: knows-HM AND has-badge)
except Exception:
    fm = None
import battle_agent  # noqa: E402  (module ref for the PROTECT_LEAD_GRIND toggle)
from battle_agent import BattleAgent  # noqa: E402
from dialogue_drive import DialogueDriver, box_open as dd_box_open  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
# PHASE E (the GO button): POKEMON_KIRA_DIR redirects the ENTIRE states/kira lineage (the sacred
# show-timeline saves + soul + checkpoints) to a sandbox, exactly as POKEMON_CAMPAIGN_DIR does for
# the Sherpa campaign. Used ONLY by go.py --throwaway so the GO button can be TESTED end-to-end
# while remaining physically incapable of touching the real states/kira. Unset = unchanged.
_KIRA_DIR_OVERRIDE = os.getenv("POKEMON_KIRA_DIR", "").strip()
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
STATES_KIRA = _KIRA_DIR_OVERRIDE or os.path.join(STATES, "kira")   # PHASE E: go.py --throwaway sandbox
STATES_ARCHIVE = os.path.join(STATES, "archive")
# states/campaign/ = Batch-5 PERSISTENT CAMPAIGN (the playthrough mode): her ONE living save that the
# free-roam run writes to as she progresses and RESUMES from on the next GO — so a session picks up
# where she actually is, never resetting to a frozen fragment. Isolated from the workshop fragments
# (those stay as fallbacks) and PHYSICALLY separate from the canonical states/kira/ spine.
# TIME-MACHINE / WATCH-SANDBOX override (watch.py): redirect the ENTIRE persistent-campaign
# bundle (state + strat/world/journey/health/playthrough sidecars, all derived below) to a
# disposable sandbox dir so a watch session spawned from a BANKED point never writes canonical.
# Crash-safe by construction: canonical states/campaign is never opened for write when this is
# set, so even a hard-kill mid-watch leaves canonical untouched. Unset = canonical, unchanged.
# NOTE: promote_bank.py keeps its OWN canonical path, so promotions always target true canonical
# regardless of this override.
_CAMPAIGN_DIR_OVERRIDE = os.getenv("POKEMON_CAMPAIGN_DIR", "").strip()
STATES_CAMPAIGN = _CAMPAIGN_DIR_OVERRIDE or os.path.join(STATES, "campaign")
CAMPAIGN_SAVE = "kira_campaign.state"      # the single living-campaign savestate filename
# ADDENDUM D — NARRATIVE continuity sidecars (persist her SAGA, not just game RAM): her team bonds/wants
# live in the canonical soul JSON; the loss/wall history (the factual basis of her Gary grudge) lives next
# to the campaign save. Both are loaded at free-roam (Sherpa) start + saved at every campaign anchor, so a
# --resume climb resumes KNOWING her story. Launch-independent (file-based, not tied to any endpoint).
# In a watch-sandbox the soul rides IN the sandbox too (bundle soul.json is copied there as
# pokemon_soul.json by watch.py), so canonical states/kira soul stays untouched. Otherwise the
# canonical kira-lineage soul. Only the free-roam continuity path reads SOUL_JSON; the --show
# segment path keeps its own explicit states/kira soul handling below.
SOUL_JSON = (os.path.join(_CAMPAIGN_DIR_OVERRIDE, "pokemon_soul.json")
             if _CAMPAIGN_DIR_OVERRIDE else os.path.join(STATES_KIRA, "pokemon_soul.json"))
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
# DENSE AUTO-CHECKPOINT (2026-07-09) — the rolling campaign anchor above is a SINGLE overwriting file
# ("now" only). For a dev fix→reload→verify loop across a 25-30h run you must hop to JUST BEFORE a bug,
# not restart from the bedroom. So ALSO bank a GROWING, LABELED history of full sanctity bundles into
# states/campaign/checkpoints/<ts>_<place>_<badges>_<playtime>/ — every gain seam (badge/town/teammate)
# AND on a wall-clock cadence — pruned to the last N. FIREWALLED to the dev/campaign line: it copies the
# campaign-side bundle only and is structurally incapable of writing the sacred states/kira/ spine.
# watch.py --list surfaces them; watch.py --at <path|alias> reloads one into a canonical-safe sandbox.
AUTO_CKPT_ENABLED = os.getenv("POKEMON_AUTO_CKPT", "1") == "1"
CKPT_EVERY_S      = float(os.getenv("POKEMON_AUTO_CKPT_EVERY_S", "720"))   # wall-clock floor (~12 min)
CKPT_KEEP         = int(os.getenv("POKEMON_AUTO_CKPT_KEEP", "40"))         # retain last N (~16 MB cap)
# INTRA-SEGMENT PROGRESS (2026-07-09, Fix C) — the SHOW/segment spine only banked a checkpoint AFTER a
# whole segment completed, so a deep failure (e.g. losing to Brock at the END of pallet_to_brock) made
# the supervisor resume from seg_opening and REPLAY THE ENTIRE SEGMENT from the bedroom (~40 min). Now
# run() banks a rolling <segckpt>.progress.state after EACH completed objective; on resume the in-
# progress segment reloads that and skips the objectives already done — a gym loss retries LOCALLY (back
# at Brock in seconds), never from bedroom. Same firewall as _save_checkpoint (SHOW writes kira; a
# WORKSHOP run is physically forbidden to). Cleared when the segment completes. Revert: =0.
SEG_PROGRESS_ENABLED = os.getenv("POKEMON_SEG_PROGRESS", "1") == "1"
# F-1 WANDER TRIPWIRE (THE DESCENT): a standing nav objective that's gone this long without the
# map-graph distance improving means she's wandering/dithering — the harness takes the wheel.
NAV_TRIPWIRE_S = float(os.getenv("POKEMON_NAV_TRIPWIRE_S", "20"))
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
# GYM-PREP / TEAM-BUILDING (2026-07-09, Fix B) — ENFORCED pre-gym readiness. She kept soloing gyms with
# an underleveled starter and losing the attrition (Brock, and the same thin-bench root as the Gary wall).
# beat_gym now runs prep_for_gym FIRST: read the strategy KB, and if the team is thin / has no type answer
# / is underleveled vs the leader's ace, CATCH on nearby grass + GRIND to a KB-derived level before
# knocking. General (per-gym via the KB), wired through StrategicPlanner. Revert: POKEMON_GYM_PREP=0.
GYM_PREP_ENABLED     = os.getenv("POKEMON_GYM_PREP", "1") == "1"
GYM_PARTY_TARGET     = int(os.getenv("POKEMON_GYM_PARTY_TARGET", "3"))     # aim for ~3 by the early gyms
GYM_PREP_CATCH_TRIES = int(os.getenv("POKEMON_GYM_PREP_CATCH_TRIES", "3")) # bounded catch attempts per prep
GYM_COVERAGE_TEACH   = os.getenv("POKEMON_GYM_COVERAGE_TEACH", "1") == "1" # teach the ace a coverage move

# COVERAGE-TEACH KB (2026-07-10, the Erika grass wall) — when the ACE's whole offense is RESISTED by a
# gym's typing, teach it a neutral-or-better damaging move from a bag TM/HM. The Erika root: L43 Venusaur's
# moves are ALL Grass (Razor Leaf / Vine Whip / Absorb, x0.25 vs grass/poison) because auto-learn stripped
# her only neutral move (Tackle) as she levelled — so even a huge level lead can't out-damage, she PP-famines
# to status, and blacks out. A real player teaches a coverage TM. bag item_id -> (hm_key|None, tm_no|None,
# move_id, type, power). HMs (Cut/Strength) are first-class + proven-actuatable; TM move_ids are VALIDATED
# against ROM gBattleMoves (st.move_info) before teaching, so a wrong id can never fire (it's skipped, LOUD).
# 2-turn moves (Dig/Fly/SolarBeam/Dive) are DELIBERATELY excluded — the battle engine's single-turn actuation
# doesn't drive the charge/hit reliably. General bedrock #8 (TM/HM strategy). Revert: POKEMON_GYM_COVERAGE_TEACH=0.
_COVERAGE_MOVES = {
    339: ("cut", None, 15, "normal", 50),        # HM01 Cut — universal neutral (x1 vs most walls)
    342: ("strength", None, 70, "normal", 80),   # HM04 Strength — stronger neutral
    301: (None, 13, 58, "ice", 95),              # TM13 Ice Beam
    302: (None, 14, 59, "ice", 120),             # TM14 Blizzard
    312: (None, 24, 85, "electric", 95),         # TM24 Thunderbolt
    313: (None, 25, 87, "electric", 120),        # TM25 Thunder
    314: (None, 26, 89, "ground", 100),          # TM26 Earthquake
    317: (None, 29, 94, "psychic", 90),          # TM29 Psychic
    318: (None, 30, 247, "ghost", 80),           # TM30 Shadow Ball
    319: (None, 31, 280, "fighting", 75),        # TM31 Brick Break
    323: (None, 35, 53, "fire", 95),             # TM35 Flamethrower
    326: (None, 38, 126, "fire", 120),           # TM38 Fire Blast
    291: (None, 3, 145, "water", 60),            # TM03 Water Pulse
    327: (None, 39, 317, "rock", 50),            # TM39 Rock Tomb
    322: (None, 34, 351, "electric", 60),        # TM34 Shock Wave
}


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
# Vermilion Gym (disasm VermilionCity_Gym; interiors live-verify on first entry): town door (14,25)
# — CUT-LOCKED behind the fence tree until HM01. Surge at (5,2), fought from (5,3) UP; juniors at
# (2,11)/(8,13)/(7,8); exits (4-6,19). ⚠ the leader is ALSO gated by the TRASH-CAN electric-lock
# puzzle (two switches; disasm recon pending) — beat_gym's junior-then-leader pattern is NOT enough
# here; do not trust a beat_gym('Lt. Surge') run until the can-puzzle solver lands.
VERMILION_GYM_DOOR, SURGE_FRONT = (14, 25), (5, 3)
# FLAG_BADGE0x_GET in the SaveBlock1 flag array (base + 0x0EE0): Boulder 0x820, Cascade 0x821.
FLAG_BADGE_BOULDER, FLAG_BADGE_CASCADE = 0x820, 0x821
FLAG_BADGE_THUNDER = 0x822
FLAG_BADGE_RAINBOW = 0x823
# Celadon ground truth (probed live 2026-07-07 from the celadon_run9 bank): Center door (48,11)
# -> interior (10,12) arrival (7,8) standard layout; GYM door (11,30) -> interior (10,16), the
# statue hall at (6,18); a CUT tree sits on the approach path (in-leg field_clear handles it).
CELADON = (3, 6)
CELADON_PC_DOOR = (48, 11)
CELADON_GYM_DOOR = (11, 30)
LAVENDER = (3, 4)                 # the graph gateway to Route 7/8/Celadon, reached only across Rock Tunnel (Flash-gated)
# Fuchsia (badge 5, Koga) — door/NPC coords from the disasm (FuchsiaCity/FuchsiaCity_Gym
# map.json, 2026-07-07); the CITY MAP ID is the city-block extrapolation (Pallet 3,0 ..
# Celadon 3,6 confirmed live) — EXPECTED until she walks it (a wrong id = beat_gym just
# doesn't trigger; the billed road carries her there and the id binds on arrival).
FUCHSIA = (3, 7)
FUCHSIA_PC_DOOR = (25, 31)
FUCHSIA_GYM_DOOR = (9, 32)
# Fuchsia Mart: overworld door (11,15) = animated slide-door (behavior 0x69) -> interior (11,1),
# clerk gfx68 @ (2,3) (canonical mart layout). MAPPED + BUY row order control-verified via a live
# buy, night-shift 1 (2026-07-10): row2 = Super Potion (id 22, 700) — the badge-5 potion-stall stock.
FUCHSIA_MART_DOOR = (11, 15)
KOGA_FRONT = (7, 14)     # Koga NPC at (7,13) FACE_DOWN -> stand below at (7,14), face UP.
FLAG_BADGE_SOUL = 0x824
# Saffron (badge 6, Sabrina) — disasm 2026-07-07 (pret map_groups.json: SaffronCity is INDEX 10
# in the towns-and-routes group — the (3,11) extrapolation was off by one, (3,11) is
# SaffronCity_Connection). SaffronCity/map.json: gym door (46,12), Silph Co door (33,30),
# Center (24,38), Mart (40,21). SaffronCity_Gym/map.json: Sabrina object at (14,11) ->
# front tile (14,12), face UP; entrance warps (13-15,23); the interior is a TELEPORT-PAD
# maze (32 warp events) — travel BFS won't cross it; strike/warp-dest routing owns that.
# NOTE the gym door is Rocket-BLOCKED until Silph Co. clears (Card Key 5F, Giovanni #2 11F).
SAFFRON = (3, 10)
SAFFRON_PC_DOOR = (24, 38)
SAFFRON_GYM_DOOR = (46, 12)
# Saffron Mart door (pret SaffronCity map.json warp -> SAFFRON_CITY_MART, 2026-07-10) — the pre-Silph
# Hyper-Potion stock-up: Gary's Charizard (Fire/Flying) 2x-burns solo Venusaur while quad-resisting her
# Grass, so she must OUT-HEAL its Fire; Super Potions (50) only tread water, Hyper Potions (200) win it.
SAFFRON_MART_DOOR = (40, 21)
SABRINA_FRONT = (14, 12)
FLAG_BADGE_MARSH = 0x825
ERIKA_FRONT = (6, 5)     # Erika NPC at (6,4) FACE_DOWN (pret CeladonCity_Gym map.json) -> stand
                         # below her at (6,5), face UP. NOTE the gym has CUT TREES inside at
                         # (6,8) (center aisle -> her), (3,5), (9,6) — the run-1/2 travel wedges.
# Cinnabar (badge 7, Blaine) — descent pre-grade fix 2026-07-08: with NO GymSpec, head_to_gym at
# the gym's own city returned no_gym_route instantly forever (announce-the-gym + stand-still — the
# banked_CINNABAR WARN livelock). Door (20,4) -> interior (12,0), LIVE-learned warp matching
# recon_blaine's GYM const; Blaine (5,4) -> front (5,5), face UP. The door is LOCKED until the
# Secret Key (Mansion B1F) — beat_gym's 'stuck' at the door arms the gym-gate probe -> the unlock
# questline (the designed general flow). The interior QUIZ-DOOR chain is strike-solved only
# (recon_blaine QUIZ_CHAIN) — if a live general walk wedges there, that seam is the port target.
CINNABAR = (3, 8)
CINNABAR_GYM_DOOR = (20, 4)
BLAINE_FRONT = (5, 5)
FLAG_BADGE_VOLCANO = 0x826
# Viridian (badge 8, Giovanni) — same fix. Door (36,10) -> interior (5,1), LIVE-read off
# banked_GIOVANNI; Giovanni (2,2) -> front (2,3), face UP (recon_giovanni). The door unlocks via
# ViridianCity OnTransition once badges 2-7 are held. Interior is a SPIN-TILE floor (the
# Rocket-Hideout class — spin_nav's slide crosser is the seam if the general walk wedges).
VIRIDIAN_CITY = (3, 1)
VIRIDIAN_GYM_DOOR = (36, 10)
GIOVANNI_FRONT = (2, 3)
FLAG_BADGE_EARTH = 0x827

# CITY -> its own Pokemon Center door (PC interiors share ONE layout, so heal_at_center(door) heals
# in ANY of them — only the overworld door differs). The map-keyed table replaces heal_nearest's old
# hardcoded Forest-era default (everything-not-Route3/4 -> Viridian), which routed Cerulean cross-
# region south. EXTEND as new cities are reached; an unmapped city heals LOUD-fallback, never silent.
CITY_PC_DOORS = {VIRIDIAN: VIRIDIAN_PC_DOOR, PEWTER: PEWTER_PC_DOOR,
                 CERULEAN: CERULEAN_PC_DOOR, ROUTE4: ROUTE4_PC_DOOR,
                 VERMILION: VERMILION_PC_DOOR,
                 (3, 28): (13, 20),    # Route 10: the Center by the Rock Tunnel door (live 2026-07-07)
                 (3, 4): (6, 5),       # Lavender: NW building -> interior (8,0), arrival (7,8) (probed live)
                 CELADON: CELADON_PC_DOOR,   # Celadon: (48,11) -> (10,12), arrival (7,8) (probed live)
                 FUCHSIA: FUCHSIA_PC_DOOR,   # Fuchsia: (25,31) (disasm; city id EXPECTED, binds on walk)
                 SAFFRON: SAFFRON_PC_DOOR,   # Saffron: (24,38) (disasm SaffronCity map.json 2026-07-07)
                 (3, 8): (14, 11),           # Cinnabar: (disasm CinnabarIsland map.json 2026-07-07;
                                             # the seafoam_run10 loud-fallback gap)
                 (3, 9): (11, 6)}            # Indigo Plateau Exterior -> the League center door
                                             # (disasm IndigoPlateau_Exterior.json; victory_run10's
                                             # heal wandered south to R23 without this)

# PC interiors share ONE layout — EXCEPT the League center (2-story, nurse behind the long
# counter at (13,10), stand (13,11)). heal_at_center consults this by DOOR before defaulting
# to the shared NURSE_FRONT.
NURSE_FRONT_OVERRIDES = {(11, 6): (13, 11)}

# Bill's PC console stand in the SHARED Center interior — top wall, right of the nurse (screenshot-
# calibrated in recon_pcbox on the Route 10 Center). Same shared layout as NURSE_FRONT=(7,4), so this
# generalises to any standard town Center; the 2-story League center (door (11,6)) is the known exception
# (deposits happen mid-game at standard Centers, so it never binds there in practice).
PC_STAND = (13, 4)

# ── GYM REGISTRY: one row per leader, so beat_gym is data-driven + general (gyms gate the leader
# behind junior trainers - beat all juniors, THEN the leader). reserve = move-slots to free for an
# expected level-up double-learn (Brock's L15); leader_dir = key that faces the leader from the
# front tile (UP for Brock + Misty - the leader sits directly above their front tile). ──
GymSpec = namedtuple("GymSpec", ["name", "city", "door", "leader_front", "badge_flag",
                                 "reserve", "leader_dir"])
# B-4 — the three Kanto starter EVOLUTION families. The old claim "no vanilla trainer except the
# rival fields a starter-line mon" is FALSE (erika_run1, 2026-07-07: a Celadon Gym Cooltrainer's
# IVYSAUR fingerprinted as Gary and polluted the grudge ledger). The rival's ace is ALWAYS the
# counter-starter line to HERS (player Bulbasaur -> his Charmander line), so detection matches
# ONLY that line; her own line (gym Ivysaurs) and the third line can never be him.
_STARTER_LINES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
_RIVAL_LINE_BY_PLAYER_BASE = {1: frozenset({4, 5, 6}),    # she picked Bulbasaur -> Gary has Charmander line
                              4: frozenset({7, 8, 9}),    # Charmander -> Squirtle line
                              7: frozenset({1, 2, 3})}    # Squirtle -> Bulbasaur line

GYMS = {
    "Brock": GymSpec("Brock", PEWTER, PEWTER_GYM_DOOR, BROCK_FRONT, FLAG_BADGE_BOULDER, 2, "UP"),
    "Misty": GymSpec("Misty", CERULEAN, CERULEAN_GYM_DOOR, MISTY_FRONT, FLAG_BADGE_CASCADE, 0, "UP"),
    # Registered so head_to_gym routes to the door (whose approach is the CUT TREE — standing at it
    # arms the HM_OBSTACLE gate -> the hm01 questline -> the S.S. Anne). The gym itself stays
    # UNBEATABLE until the trash-can puzzle solver lands (see VERMILION_GYM_DOOR note).
    "Lt. Surge": GymSpec("Lt. Surge", VERMILION, VERMILION_GYM_DOOR, SURGE_FRONT,
                         FLAG_BADGE_THUNDER, 0, "UP"),
    "Erika": GymSpec("Erika", CELADON, CELADON_GYM_DOOR, ERIKA_FRONT,
                     FLAG_BADGE_RAINBOW, 0, "UP"),
    # Koga's invisible-wall maze is a VISION gimmick only — the RAM collision map is truthful,
    # so travel's BFS walks it like any gym. City id EXPECTED (see FUCHSIA note).
    "Koga": GymSpec("Koga", FUCHSIA, FUCHSIA_GYM_DOOR, KOGA_FRONT,
                    FLAG_BADGE_SOUL, 0, "UP"),
    # Registered so head_to_gym targets Saffron and beat_gym binds on arrival. The DOOR is
    # Rocket-blocked until Silph Co. clears, and the interior is a teleport-pad maze — the
    # leader approach needs pad routing (warp events; read_warps sees them) or a strike.
    "Sabrina": GymSpec("Sabrina", SAFFRON, SAFFRON_GYM_DOOR, SABRINA_FRONT,
                       FLAG_BADGE_MARSH, 0, "UP"),
    # Badge 7/8 (descent pre-grade fix): registered so head_to_gym AT the city enters/probes
    # instead of the no_gym_route stand-still livelock. Locked doors arm the gate questline;
    # interior gimmicks (Blaine's quiz doors, Giovanni's spin floor) are flagged at the consts.
    "Blaine": GymSpec("Blaine", CINNABAR, CINNABAR_GYM_DOOR, BLAINE_FRONT,
                      FLAG_BADGE_VOLCANO, 0, "UP"),
    "Giovanni": GymSpec("Giovanni", VIRIDIAN_CITY, VIRIDIAN_GYM_DOOR, GIOVANNI_FRONT,
                        FLAG_BADGE_EARTH, 0, "UP"),
}
# GYM STORY-PREREQ GATES: a gym whose DOOR is walled by a STORY event (not an HM obstacle, not the
# gym-door tree the HM probe handles) — beat_gym 'stuck' there needs a LIBERATION ERRAND armed, not a
# tree cut. leader -> (unmet-flag-id, KB-capability-key, in-character wall line). The capability key
# resolves through gamedata/frlg_gates.json (a strike step, no door) so _derive_questline -> the
# registered dungeon strike fires. FireRed facts isolated to the KB + this table (rule 14). Sabrina's
# gym is Rocket-blocked until Silph Co. clears (FLAG_HIDE_SAFFRON_ROCKETS 0x3E).
GYM_PREREQS = {
    "Sabrina": (0x3E, "FLAG_HIDE_SAFFRON_ROCKETS",
                "Sabrina's gym is blocked — Team Rocket has Silph Co. locked down in the middle of the city"),
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
                   VERMILION: VERMILION_MART_DOOR,
                   # Fuchsia Mart (night-shift 1) — the badge-5 Koga potion-stall stock-up.
                   FUCHSIA: FUCHSIA_MART_DOOR,
                   # Saffron Mart (night-shift 4) — the pre-Silph Hyper-Potion stock-up (Gary/Charizard).
                   SAFFRON: SAFFRON_MART_DOOR}

# DOOR APPROACH WAYPOINTS (game-knowledge, rule 14): some Mart/building doors sit in a pocket the
# direct BFS-travel OSCILLATES into and never reaches (Fuchsia's central ponds wall the Mart door
# (11,15) — isolated travel caps at the town centre from either spawn). Walk these waypoints to the
# door pocket first, then the normal one-tile approach step is a clean short hop. Keyed
# (city_map, door_tile) -> [waypoint coords]. VERIFIED night-shift 1 (probe_mart5).
DOOR_APPROACH_WAYPOINTS = {
    (FUCHSIA, FUCHSIA_MART_DOOR): [(20, 24), (19, 18), (15, 16)],
}
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
    # Vermilion rows per Bulbapedia (PokeBall, SuperPotion, IceHeal, Awakening, ParlyzHeal, Repel) —
    # control-verified by a live bag-delta buy 2026-07-06; a wrong row aborts LOUD, never mis-buys.
    VERMILION: [4, 22, 16, 17, 18, 86],
    # Fuchsia rows (UltraBall, GreatBall, SuperPotion, Revive, FullHeal, MaxRepel) — CONTROL-VERIFIED
    # night-shift 1 by a live bag-delta buy: row2 Super Potion(22,700), row4 Full Heal(23,600). Balls
    # (rows 0/1) land in the Balls pocket so the Items-pocket bag-delta can't confirm their id — the
    # per-purchase verify guards any row anyway. No Hyper Potion sold here; Super Potion is the top heal.
    FUCHSIA: [2, 3, 22, 28, 23, 88],
    # Saffron rows (GreatBall, HyperPotion, Revive, FullHeal, EscapeRope, MaxRepel) — order from the
    # pret SaffronCity_Mart scripts.inc pokemart list (2026-07-10). Hyper Potion(21) is row 1 — the
    # NS4 pre-Silph stock-up target; buy_at_mart's per-purchase bag-delta guards any mis-row LOUD.
    SAFFRON: [3, 21, 24, 23, 85, 88],
}
MART_CURSOR = 0x02039940   # u16 sShopData.selectedRow (pret sym 0x02039934+0xC) — the WINDOW row only
MART_SCROLL = 0x02039942   # u16 sShopData.scrollOffset — items hidden above the window
# TRUE BUY-LIST SELECTION = selectedRow + scrollOffset. On lists deeper than the window the row
# byte alone LIES (tm_errand run-11: (row 4, scroll 2) = CANCEL on the Celadon TM list; its A
# exited the shop and the buy-mash re-entered and bought row 0). Every shop nav reads the SUM.
# Shopping policy (named, tunable): top potions up to this; buy this many of each needed cure; keep this
# much money in reserve (never drain the wallet). Quantities are sensible, not min-max hoarding.
SHOP_POTION_TARGET = 6
# POTION-STALL gyms (night-shift 1): a leader that is an ATTRITION wall for a solo-carry — Koga's
# 4-mon poison gauntlet (~390 HP) out-damages a lone L53 Venusaur that has NO healing items (shift-17:
# 5+ straight losses without potions; WINS with them). A real player stocks Super Potions at the
# gym-city Mart and heals through it. gym.name -> how many potions to carry into the fight. Game-fact
# isolated here (rule 14); the pre-gym stock-up leg buys up to this from the gym-city Mart.
POTION_STALL_GYMS = {"Koga": 30}
# PRE-SILPH HYPER-POTION target (night-shift 4): Gary's Silph gauntlet ends on a Charizard (Fire/Flying)
# that 2x-burns solo Venusaur while QUAD-resisting her Grass — she can only chip with weak Cut, so the
# ONLY winning line is to OUT-HEAL the Fire. Super Potions (50 HP) ~= Charizard's chip (tread water ->
# faint at Charizard ~29/119); Hyper Potions (200 HP) win the attrition outright (NS4 proof: 12 Hyper
# Potions -> GARY WON -> Saffron freed). Count HYPER specifically (30 Supers must NOT read as stocked).
# Bought at the Saffron Mart, on-route, standing in Saffron right before the strike. Game-fact (rule 14).
# REVIVES are load-bearing too (NS4 autobuy postmortem): a single Charizard high-roll/crit (or a lost
# paralysis turn) faints Venusaur once, and with NO Revive the fodder bench can't finish Charizard ->
# instant loss (koga_done_kit carries 0 Revives). The PROVEN-WINNING kit was Hyper + Revive: she revives
# Venusaur behind a fodder body (the _revive_worthy comeback cycle) and keeps out-healing. Saffron Mart
# sells both (Hyper row1 @1200, Revive row2 @1500). Budget-fit for ~$30k: 15 Hyper + 5 Revive ≈ $25.5k.
SILPH_HYPER_TARGET = 15
SILPH_REVIVE_TARGET = 5
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
# NS#42 ball economy: when the team-plan has a coverage keeper DUE, stock a fuller ball pocket even at
# party>3 (the plain SHOP_BALL_TARGET only tops balls for a THIN team, so a party-4 hunter walked out with
# too few — ns41_finalproof/ns42_celadon: catch_pokemon -> no_balls in Diglett's Cave). A keeper hunt (even
# an easy diglett) can burn several balls to a break-free, and abra/growlithe are harder — so pre-stock ~8.
SHOP_BALL_KEEPER_TARGET = 8
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
        # GRIND before the Forest (2026-07-09 night train): the Forest leg FLEES wilds (the only reliable
        # crossing), so a solo starter reaches Pewter at Lv8 and loses Brock. Train to ~Lv13 on Route 2
        # FIRST (grass + a clean short heal-bounce to Viridian) — Vine Whip + the HP to survive Onix —
        # then flee-cross. A real player grinds the early route before the first gym; this is that.
        ("GRIND_PRE_BROCK", 13, "train the solo starter to ~Lv13 on Route 2 (Vine Whip + Onix survival) "
         "before the Forest crossing"),
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
        # STRATEGIC PLANNER (2026-07-08 mega-batch): the 'read every guide' forward-prep layer. Reads
        # gamedata/frlg_strategy.json and turns the NEXT threat into a proactive in-character prep beat
        # (matchup foresight, level-your-counter, catch-a-keeper, develop-the-bench) folded into the
        # oracle ctx via the SAME `place` seam the loss-awareness/folklore notes use. Firewall: mode-side
        # game-knowledge only, never core. Kill-switch POKEMON_STRATEGIC_PLANNER=0.
        self.planner = StrategicPlanner(log=log)
        # TEAM PLANNER (2026-07-09 mission-pivot Part B/C): the STANDING forward-planning team brain —
        # whole-game lookahead over the deep KB (gamedata/frlg_*.json) that plans a balanced-6 toward the
        # E4 from the start and emits the single highest-leverage next action (catch a keeper / evolve /
        # teach a coverage TM / bounded grind), voiced first-person through plan_note. This is the PRIMARY
        # forward-prep voice; the reactive StrategicPlanner above is its fallback (matchup foresight when
        # the brain is on-track/silent). Same `place` seam, same mode-side firewall, she still chooses.
        # Kill-switch POKEMON_TEAM_PLANNER=0 -> the brain never folds and StrategicPlanner takes over.
        self.team_planner = TeamPlanner(log=log)
        # RECALIBRATION — the STANDING objective she returns to after a detour (a 25-min trainer
        # gauntlet, a heal run). Set when she commits to a going-somewhere action; preserved through
        # detour actions (heal/talk/grind) so she resumes "I was heading to X" instead of re-deciding
        # from scratch + wandering. {action, label, tick} or None. Surfaced to her ctx + the dashboard.
        self._active_objective = None
        # F-1 WANDER TRIPWIRE (THE DESCENT) — per-objective nav-progress watch:
        # {action, best (route-length), ts (last improvement), fired} or None. See free_roam.
        self._nav_watch = None
        self._nav_tripwire_total = 0          # run-lifetime fire count (descent-grading telemetry)
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
        # F-11 STRUCTURAL DEAD-ROUTE MEMORY (descent pre-grade, the Cinnabar head_to_gym×25):
        # map-tuple -> set of picks that failed STRUCTURALLY there (no_gym_route — no route
        # EXISTS from that map). Unlike _dead_moves (cleared by ANY movement — right for
        # transient blocks like an NPC on the gap), these stay pruned while she's ON the map;
        # leaving clears the map's entry, so returning retries exactly once.
        self._dead_moves_structural = {}
        # SPLIT-MAP ROAD MEMORY (2026-07-08, the SURF_TAUGHT Route-19↔20 metronome): retry-once
        # (leave clears the map's entry) is right for transients but LAUNDERS a split-map dead end —
        # she bounces A→B ("arrived", movement clears everything), fails on B, leaves to A, forever.
        # Strikes count structural fails per (map, pick) and NEVER clear on movement; a pick with
        # ≥2 strikes survives the leave-clears rule (proven dead across visits). And when
        # head_to_gym proves dead on the map it JUST rode into, the SOURCE map's head_to_gym is
        # parked too (the leg that keeps feeding the dead end), with one narrated beat.
        self._dead_route_strikes = {}
        self._last_hg_leg = None              # (from_map, to_map) of the last map-crossing head_to_gym
        self._split_map_beat_done = set()     # one beat per dead road, not a lecture
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
        # SOUL-DEBT #12 (info half) — overheard-intel ledger: every accepted overworld line (via the
        # DialogueDriver class tap) runs the hint extractor; actionable ones persist in the campaign
        # dir and fold into her decision ctx ("things you've heard"). She READS boxes already — this
        # is what makes her USE them. Mode-side only; her voice reactions ride the existing seam.
        try:
            import dialogue_hints as dhints
            from dialogue_drive import DialogueDriver as _DD
            from dialogue_reader import salience_tags as _sal_tags
            self.hints = dhints.HintLedger(os.path.join(STATES_CAMPAIGN, "dialogue_hints.json"), log=log)
            _DD.line_sink = lambda line: self.hints.add(
                line, tags=_sal_tags(line), where=tuple(tv.map_id(self.b)))
        except Exception as _he:
            self.hints = None
            log(f"   [hints] !! ledger init failed (intel OFF this run, LOUD): {_he}")
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
        self._ql_inside_map = None            # the interior map she's deliberately in (whiteout-relocation guard)
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
        # PHASE C-2 — UNMET-TEAMMATE WATCHER (the Lapras first-field moment, generalized): PIDs seen
        # in the party. None = baseline pending (first tick marks everyone present as already-met,
        # silently — she's been traveling with them). A PID that APPEARS later without a witnessed
        # catch (gift / PC withdrawal / Jonny's hands) fires the introduction arc in _meet_new_teammates.
        self._met_pids = None
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
        # blackout-evidence: set by _observed_battle_runner, consumed per roam tick — the tick-top
        # "indoors = blackout" heuristic only fires when a battle actually RAN (run-9 ghost class)
        self._battle_ran_this_action = False
        # one Traveler reused for every WALK leg (BFS + NPC-aware + grass-aware + handoff)
        # stuck_check (LAYER B): travel polls this each step; when the universal watchdog has latched a
        # disengage, travel bails its leg LOUD ("stuck") so the wedge unwinds to the roam loop's
        # top-level recovery — instead of spinning a sub-tick loop the per-tick ledger can't see.
        self.trav = tv.Traveler(bridge, battle_runner=battle_runner, render=self.render,
                                on_event=self.on_event, beat=self.beat,
                                pause_check=lambda: self.needs_heal() and not self._suppress_heal,
                                stuck_check=lambda: self._stuck_request is not None,
                                blocked_npcs=self._blocked_npcs,   # LAYER A: shared route-around memory
                                field_clear=lambda hm, face: (
                                    self.field.clear_obstacle(hm, face)
                                    if getattr(self, "field", None) else "cant"),
                                # transit-time map learning: step-on mats fire MID-leg (UGP
                                # tunnel->hut), so every transition folds into the mental map
                                on_transition=lambda: self._learn_transit(),
                                # spinner-floor hand-off (shift 5): travel's wedge guard calls
                                # this ONCE per leg on maps with spin tiles (hideout B2F/B3F)
                                spin_assist=self._spin_assist)
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
        pre_sp = pre_lvl = pre_lvls = pre_specs = None
        self._battle_ran_this_action = True    # blackout-evidence: a whiteout implies a battle RAN
        # P-1(a): the FIRSTS — a first-timer's first wild rustle / first real trainer fight
        # are events a viewer should feel. One beat each, early-game gated in _first_beat.
        try:
            if bool(self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08):
                self._first_beat("trainer_battle",
                                 "my first REAL trainer battle — an actual person wants to "
                                 "fight me. okay. deep breath. this is what training's for.")
            else:
                self._first_beat("wild_battle",
                                 "the grass MOVED — that's my first wild Pokémon ever. "
                                 "okay okay okay. here we go, this is really happening.")
        except Exception:
            pass
        try:
            mid = tv.map_id(self.b)
            place = self._place_name(mid)
            pre_sp = st.read_party_species(self.b, 0)          # Phase 3A: lead species + level before battle
            pre_lvl = self.b.rd8(ram.GPLAYER_PARTY + self._PARTY_LEVEL_OFF)
            # ANY-SLOT snapshots (night shift 11): the evolution gate was LEAD-ONLY, so a fielded
            # bench mon (strategic grind) or a mid-battle switch-in that leveled+evolved left the
            # cutscene UNDRIVEN (watchdog territory). Snapshot every slot; the drive gates on any.
            pre_lvls = self._party_levels()
            pre_specs = [st.read_party_species(self.b, s) for s in range(len(pre_lvls))]
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
                # HER starter family (from her party) -> the rival's counter line. If her starter is
                # somehow not in the party (boxed — can't happen pre-box-mgmt), fall back to the old
                # any-starter-line detection rather than go blind.
                _riv_lines = _STARTER_LINES
                for _ps in range(self.b.rd8(ram.GPLAYER_PARTY_CNT)):
                    _psp = st.read_party_species(self.b, _ps)
                    if _psp in _STARTER_LINES:
                        _riv_lines = _RIVAL_LINE_BY_PLAYER_BASE[((_psp - 1) // 3) * 3 + 1]
                        break
                # STALE-SLOT GATE (the ship-class false Garys, encounters 4-7): gEnemyParty slots
                # beyond the CURRENT trainer's roster keep the PREVIOUS battle's mons, so after a
                # real Gary fight every small-roster trainer "contained" his ace. A REAL battle
                # starts with every enemy mon at FULL HP; the stale mons she beat sit at 0 — so a
                # rival-line hit only counts from a slot that is ALIVE (HP>0, plaintext +0x56).
                for _es in range(6):
                    if st.read_enemy_species(self.b, _es) in _riv_lines:
                        if self.b.rd16(ram.GENEMY_PARTY + _es * 100 + 0x56) > 0:
                            _is_rival = True
                            break
                        log(f"   [strat] rival-line species in enemy slot {_es} but at 0 HP "
                            f"-> STALE previous-battle data, not Gary (skipping)")
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
            self._drive_evolution(pre_sp, pre_lvl,             # Phase 3A: drive a post-battle evolution (gated on level-up)
                                  pre_levels=pre_lvls, pre_specs=pre_specs)
        except Exception as _e:
            log(f"   [evolve] drive skipped: {_e}")
        try:
            self.strat.observe_battle_end(self.b, record_out)   # record_out = swallow-guarded outcome
        except Exception as _e:
            log(f"   [strat] end-observe skipped: {_e}")
        # ATTACH-TIME RIVAL RE-CHECK (the filed tower4 bug): when the observer attaches to an
        # already-RUNNING scene battle (Gary walked up mid-travel), gEnemyParty still holds the
        # PREVIOUS fight at scan time — the start-scan above reads stale data and misses him (the
        # Tower win needed a manual ledger backfill). But the engine read every foe LIVE at its
        # action menus: re-check that foes-seen ledger against his counter line (recomputed from her
        # starter so the erika_run1 Ivysaur-Cooltrainer false-positive stays dead; if her starter
        # can't be found the check is SKIPPED, never widened).
        if not _is_rival:
            try:
                import battle_agent as _ba
                _lines = None
                for _ps in range(self.b.rd8(ram.GPLAYER_PARTY_CNT)):
                    _psp = st.read_party_species(self.b, _ps)
                    if _psp in _STARTER_LINES:
                        _lines = _RIVAL_LINE_BY_PLAYER_BASE[((_psp - 1) // 3) * 3 + 1]
                        break
                _late = (set(_ba.LAST_FOES_SEEN) & set(_lines)) if _lines else set()
                if _late:
                    _is_rival = True
                    _riv_lead_nm = st.SPECIES_NAME.get(sorted(_late)[0], "")
                    log(f"   [strat] RIVAL detected POST-BATTLE from the foes-seen ledger "
                        f"({sorted(_late)}) — the start-scan attached mid-scene and read stale "
                        f"gEnemyParty; recording him now (no backfill needed)")
            except Exception as _le:
                log(f"   [strat] late rival re-check skipped: {_le}")
        # B-4 — GARY NEMESIS ARC: record the rival encounter at EVERY fight (not just the opening) so the
        # persisted grudge actually ESCALATES across the run, and voice the escalating-grudge beat.
        # A TIMEOUT is an engine artifact, not a story beat — recording it as won=False writes a
        # false LOSS into the grudge ledger (erika_run1's phantom "Gary loss #8"). Skip it.
        if _is_rival and str(record_out).lower() == "timeout":
            log("   [strat] rival battle ended in engine TIMEOUT -> NOT recorded in the grudge ledger")
            _is_rival = False
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
    def _edge_band_reachable(grid, here, edge, walkable=None):
        """Can `here` BFS-reach a border tile in the map-edge's CONNECTION BAND (the rows/cols where the
        neighbour's overlap tile just past the border is walkable)? The same crossable-edge signal
        travel's edge-goal uses — a border tile OUTSIDE the band is a hard wall, so plain border
        reachability is a false 'crossable' (the (84,15) lesson). `walkable` overrides the collision
        layer — pass grid.walkable_or_surf when Surf is usable so water crossings (Pallet→R21 south
        band) and water-locked bands count as crossable."""
        wk = walkable or grid.walkable
        if edge == "east":
            line, past_d, axis = grid.sx_hi, 1, 0
        elif edge == "west":
            line, past_d, axis = grid.sx_lo, -1, 0
        elif edge == "north":
            line, past_d, axis = grid.sy_lo, -1, 1
        else:
            line, past_d, axis = grid.sy_hi, 1, 1
        if axis == 0:
            band = {p for p in range(grid.sy_lo, grid.sy_hi + 1) if wk(line + past_d, p)}
            goal = lambda t: t[0] == line and t[1] in band
        else:
            band = {p for p in range(grid.sx_lo, grid.sx_hi + 1) if wk(p, line + past_d)}
            goal = lambda t: t[1] == line and t[0] in band
        if not band:
            return False
        return bool(tv.bfs(grid, here, goal, walkable=wk))

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
        # INTERIOR-FIRST (2026-07-06, ship run 18): hurt DEEP inside a multi-room complex (a fainted
        # Persian in S.S. Anne cabin (1,13)) NO step below can route — interiors aren't graph nodes
        # and have no edge connections, so heal spun 'stuck' x14 into a STALL. Get OUT to the
        # overworld first (the street-first exit unwinds the ship, proven run 15), then heal from
        # the street — the ship exits into Vermilion, whose own Center is registered. Pre-HM01
        # re-boarding is proven; post-HM01 the run has already GOAL'd.
        if tuple(m)[0] != 3:
            log(f"   HEAL: inside a building complex {m} — exiting to the overworld before routing")
            try:
                self._exit_to_overworld()
            except Exception as _xe:
                log(f"   HEAL: exit-to-overworld crashed: {_xe!r} (LOUD)")
            m2 = tuple(tv.map_id(self.b))
            if m2[0] == 3:
                if m2 in CITY_PC_DOORS:
                    return ("ok" if self.heal_at_center(CITY_PC_DOORS[m2])
                            in ("healed", "healed_stuck_inside") else "stuck")
                m = m2                      # continue the ladder from the street
        # UNMAPPED map, ADJACENT-CITY FIRST (2026-07-06, the Route-6/S.S.-Anne lesson): the LIVE map
        # header knows neighbours she hasn't VISITED yet — the world graph below only routes VISITED
        # nodes, so a first approach to a new town (Route 6 hurt from the gauntlet, Vermilion one edge
        # south) found "no graph route" and the blind warp-south fallback BOARDED THE BOAT. One edge
        # cross to a registered-Center neighbour, heal, return — the Route-3->Pewter pattern, general.
        _EDGEU = {"N": "north", "S": "south", "E": "east", "W": "west"}
        _REVU = {"north": "south", "south": "north", "east": "west", "west": "east"}
        # REACHABILITY PRE-CHECK (2026-07-08, the Route-23 153-twedge storm): a map can be SPLIT into
        # components the connection graph can't see — Route 23's south half is walled from its north
        # half by design (Victory Road IS the road between them). "Indigo is adjacent north" is true
        # at map granularity and a lie at feet granularity; every un-checked excursion north burned a
        # travel-wedge cycle. Band-check from the FEET (surf-aware) before committing any leg.
        _hgrid = _hhere = _hwk = None
        try:
            _hgrid, _hhere = tv.Grid(self.b), tv.coords(self.b)
            _hwk = _hgrid.walkable_or_surf if self._surf_usable() else None
        except Exception as _ge:
            log(f"   HEAL: pre-check grid unavailable ({_ge!r}) — proceeding un-checked (LOUD)")
        for d, nbr in self._map_connections():
            if tuple(nbr) in CITY_PC_DOORS and tuple(nbr) != tuple(m) and d in _EDGEU:
                if _hgrid is not None and _hhere is not None and not self._edge_band_reachable(
                        _hgrid, _hhere, _EDGEU[d], walkable=_hwk):
                    log(f"   HEAL: ADJACENT city {self.world.name(tuple(nbr))} ({d}) band UNREACHABLE "
                        f"from feet (split map, Route-23 class) — skipping the excursion")
                    continue
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
                    # Route-23-class pre-check: a cardinal hop the FEET can't reach (split map)
                    # would wedge travel x4 and retry x8 — fail FAST to the next ladder rung.
                    if edge in ("north", "south", "east", "west"):
                        try:
                            _g = tv.Grid(self.b)
                            _w = _g.walkable_or_surf if self._surf_usable() else None
                            if not self._edge_band_reachable(_g, tv.coords(self.b), edge, walkable=_w):
                                log(f"   !! HEAL: graph hop {cur}->{nxt} ({edge}) band UNREACHABLE "
                                    f"from feet (split map) — abandoning the graph route (LOUD)")
                                break
                        except Exception:
                            pass               # pre-check is advisory; travel stays the arbiter
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
        # WAYPOINT-GUIDED APPROACH: a door in a maze-walled pocket (Fuchsia Mart (11,15)) makes the
        # direct BFS-travel oscillate and cap far away. If we have KB waypoints for this exact door,
        # walk them first so the normal approach-step below starts one clean hop from the door.
        if pick is not None:
            _wps = DOOR_APPROACH_WAYPOINTS.get((before, tuple(pick)))
            if _wps:
                log(f"   door {tuple(pick)}: waypoint-guided approach via {_wps}")
                for _wp in _wps:
                    self.trav.travel(target_map=None, arrive_coord=_wp, max_steps=2000, max_seconds=90)
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
            # SURF-AWARE (2026-07-08, Route-23 "no south warp" lie): the gate door at (8,153)
            # sits across the lake — a land-only pre-check skipped the ONLY exit from the
            # route's south half. Same water-start law as _reachable_grass/_door_passthrough.
            grid = tv.Grid(self.b)
            _wk = grid.walkable_or_surf if self._surf_usable() else grid.walkable
            if tv.bfs(grid, tv.coords(self.b), lambda t: t == approach,
                      walkable=_wk):
                log(f"   reachable door {door}: walking to approach {approach}")
                # max_steps 900 (was 300): a long-map approach (Route 12's ~100-row pier maze,
                # sabrina_run2) burns 300 steps on obstacle re-routing and dies 26 rows short —
                # a BUDGET failure misread as entry geometry. Wall-clock stays the real bound.
                r = self.trav.travel(target_map=None, arrive_coord=approach, max_steps=900,
                                     max_seconds=(budget_s or 300))
                if r == "need_heal":
                    return "need_heal"      # heal interrupt during the approach -> let caller heal
                if r == "timeout" and pick is not None:
                    log(f"   approach to {door} ran out of steps/time — retryable, NOT entry geometry")
                    return "no_reach"
                if r == "arrived":
                    for _ in range(5):                   # step INTO the doorway -> warp
                        self.b.press(step_key, 8, 8, self.render, owner="agent")
                        self._advance_dialogue(taps=2)   # gate buildings may print a line
                        if tv.map_id(self.b) != before:
                            log(f"   WARPED {before} -> {tv.map_id(self.b)} via door {door}")
                            self._learn_transit()
                            return "warped"
            # DIRECTIONAL STAIR/ARROW/ESCALATOR/MAT DOOR (dept-store-stairs + hut-exit-mat
            # classes): a blind UP-step walks straight past 0x62-0x6F tiles — they fire only
            # when entered moving their direction — AND their canonical approach tile can be
            # unwalkable (run-10: the hut exit mats' y+1 is past the wall, so the pre-check
            # `continue` skipped the ritual and she wedged inside on the LAST door). Run the
            # ritual regardless of approach reachability — it routes its own stand tile and
            # has the on-tile fallback.
            if (self._tile_behavior(*door) in self._WARP_ENTRY
                    and self._enter_directional_warp(door)):
                log(f"   WARPED {before} -> {tv.map_id(self.b)} via directional door {door}")
                return "warped"
        # MULTI-TILE MAT ROW (run-11 hut wedge): the warp TABLE lists side tiles ((5,8)/(7,8) →
        # overworld) whose behavior is 0x00 — only the CENTER mat carries the 0x65 arrow that
        # actually fires. A pick aimed at a side tile fails every ritual. Fall back to the
        # actuatable door tiles ADJACENT to the pick (never far ones — those lead elsewhere).
        if pick is not None:
            near = sorted((d for d in doors
                           if d != pick and abs(d[0] - pick[0]) + abs(d[1] - pick[1]) <= 2),
                          key=lambda d: abs(d[0] - pick[0]) + abs(d[1] - pick[1]))
            for door in near:
                log(f"   pick {pick} failed — trying adjacent actuatable door {door} (mat-row)")
                if (self._tile_behavior(*door) in self._WARP_ENTRY
                        and self._enter_directional_warp(door)):
                    log(f"   WARPED {before} -> {tv.map_id(self.b)} via mat-row door {door}")
                    return "warped"
        log(f"   no reachable door warped (entry geometry?) - LOUD")
        return "no_warp"

    def _learn_transit(self):
        """TRANSIT-TIME warp learning (2026-07-07, flute_run4): _learn_map fires once per roam
        TICK, but door-passthrough/enter_warp cross interiors WITHIN one action — the UGP huts'
        warps were never recorded (world model: visited=False, warps={}), so the mental-map BFS
        dead-ended at the hut door and read 'no route Lavender->Celadon' though she'd WALKED that
        road. Fold the map we just landed on into the graph immediately. Pure reads; never raises."""
        try:
            for _ in range(30):                 # settle past the mid-transition layout window
                self.b.run_frame()
            self._learn_map()
        except Exception as _lt:
            log(f"   [world] transit learn skipped: {_lt}")

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
        # SHIFT-6 REACHABILITY LAW at the ritual level (shift 12, banked_VICTORY ×27 wedge
        # storm): run-10 made this ritual run even when the CANONICAL approach tile is
        # unwalkable — right for mat rows, wrong in a walk-SEALED pocket (VR 2F landing
        # (3,3): the boulder-switch puzzle walls off the whole east half, and every ritual
        # call burned up to 5 blind travel legs = 5 wedges). ONE static BFS over the whole
        # candidate family first; nothing reachable -> nothing to actuate, bail leg-free.
        # Same-optimism grid as travel's own BFS, so this can never skip a door travel
        # could have reached.
        cand = {stand, tuple(wt), (wt[0], wt[1] + 1), (wt[0], wt[1] - 1),
                (wt[0] - 1, wt[1]), (wt[0] + 1, wt[1])}
        g0 = tv.Grid(self.b)
        c0 = tuple(tv.coords(self.b) or ())
        if c0 and c0 not in cand and not tv.bfs(g0, c0, lambda t: t in cand,
                                                walkable=g0.walkable):
            log(f"   directional warp {wt}: no stand/neighbor reachable from {c0} — "
                f"sealed pocket, skipping (0 legs)")
            return False
        if self.trav.travel(target_map=None, arrive_coord=stand, max_steps=120,
                            max_seconds=60) != "arrived":
            # ON-TILE FALLBACK (2026-07-06, the captain's-office 0x6C stair): the opposite
            # stand tile can be WALLED (SSAnne 2F reaches its stair only from the SOUTH; (29,2)
            # doesn't exist as floor). The stair still fires when you stand ON the tile and
            # press its entry direction — recon-proven: one RIGHT on (30,2) → the office.
            # Get on it via any reachable neighbor, then fall into the press loop below.
            if tuple(tv.coords(self.b) or ()) != tuple(wt):
                for nb, k2 in (((wt[0], wt[1] + 1), "UP"), ((wt[0], wt[1] - 1), "DOWN"),
                               ((wt[0] - 1, wt[1]), "RIGHT"), ((wt[0] + 1, wt[1]), "LEFT")):
                    if self.trav.travel(target_map=None, arrive_coord=nb, max_steps=120,
                                        max_seconds=45) != "arrived":
                        continue
                    # 3 presses, not 1: the first is routinely eaten as a TURN (tower1's 2F
                    # 0x6D stair — the probe needed UP×3 from (4,11) to actually mount (4,10))
                    for _try in range(3):
                        self.b.press(k2, 8, 10, self.render, owner="agent")
                        for _ in range(30):
                            self.b.run_frame()
                        if tuple(tv.map_id(self.b)) != m0:
                            for _ in range(60):
                                self.b.run_frame()
                            log(f"   directional warp {wt} (behavior 0x{bh:02x}) fired on approach via {k2}")
                            self._learn_transit()
                            return True
                        if tuple(tv.coords(self.b) or ()) == tuple(wt):
                            break
                    if tuple(tv.coords(self.b) or ()) == tuple(wt):
                        break
            if tuple(tv.coords(self.b) or ()) != tuple(wt):
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
                self._learn_transit()
                return True
        return False

    # ── THE DOOR PASS-THROUGH PRIMITIVE (2026-07-06 — the south-gate class, GENERAL) ──────────────
    # GROUND TRUTH (recon_south_geometry + pret map.json + Bulbapedia): post-ticket Cerulean's ONLY
    # pre-Cut route south is THROUGH the burgled house — front door (30,11) → interior → back door →
    # the fenced garden → the east corridor (cols 39-40) → the fence crossing at (39-40,32) → the
    # south strip → Route 5. The overworld BFS can NEVER see this (the region is fence-isolated), so
    # when an edge crossing reports no-route, doors are the remaining connectors. The SAME shape is
    # the Underground Path huts and the Saffron gatehouses — build once, reuse everywhere.
    def _door_passthrough(self, budget_s=240, want_map=None):
        """Enter a reachable, untried door on this map; inside, exit through a DIFFERENT warp; if we
        pop out materially elsewhere (across a fence / on another map), report 'crossed' so the
        caller retries its edge crossing from the new region. Bounded (each door once per map per
        session); on a dead end she exits back out the way she came (and roam's blackout-recovery
        auto-exits interiors anyway). Returns 'crossed' | 'no_passthrough' | 'need_heal'.
        `want_map` (sabrina_run6): the caller's desired far-side map. Candidates whose warp DEST is
        that map (or unvisited) sort first — the UGP hut ('crosses' AROUND Saffron to visited Route
        7 ground) otherwise beats the actual Saffron gate on raw distance, and the road-follow then
        steers back to Route 8 for an infinite tunnel ping-pong. A crossing that lands off-want is
        still usable (pre-Tea the tunnel IS the billed detour) but is never REMEMBERED as this
        map's proven connector."""
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
        # NO-CONNECTOR maps (KB, 2026-07-07): never enter a dark warp MAZE as a "connector" — the
        # bounded hop-walk crossed ROCK TUNNEL and carried her back through the dark (celadon_run5).
        try:
            _nc = {tuple(int(x) for x in k.split(","))
                   for k in (self._gate_recognizer.kb.get("no_connector") or {}).get("maps", [])}
        except Exception:
            _nc = set()
        cands = []
        # WATER-START/WATER-ROAD (night shift 10): a sea-route connector (the Seafoam entrances
        # on Route 20) sits across open water — the land-only reach test silently skipped it and
        # the passthrough reported no_passthrough from a surf stand. Same layer law as the grass
        # reachability: plan over land+water when Surf is usable.
        _wlk = grid.walkable_or_surf if self._surf_usable() else grid.walkable
        for wt in door_dest:
            if wt in tried:
                continue
            if tuple(door_dest.get(wt) or ()) in _nc:
                continue
            if tv.bfs(grid, pos0, lambda t, w=wt: t == w or
                      (abs(t[0] - w[0]) + abs(t[1] - w[1]) == 1), walkable=_wlk):
                cands.append(wt)
        def _dest_rank(t):
            dm = tuple(door_dest.get(tuple(t)) or ())
            if want_map and dm == tuple(want_map):
                return 0                              # a door straight into the wanted map
            try:
                return 2 if self.world.visited(dm) else 1   # unvisited interiors beat known ground
            except Exception:
                return 1
        cands.sort(key=lambda t: (_dest_rank(t),
                                  -len(dest_doors.get(door_dest.get(tuple(t)), ())),
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
                    tried.discard(tuple(door))   # entry never attempted — leave it retryable
                    return "need_heal"
                if r == "no_reach":
                    tried.discard(tuple(door))   # approach budget ran dry — retryable next attempt
                    continue
                if r != "warped":
                    continue
            else:
                # a plain warp mat (hut stairs / hole): walk ONTO it; nudge through if standing on it
                r = self.trav.travel(target_map=None, arrive_coord=tuple(door),
                                     max_steps=900, max_seconds=budget_s)
                if r == "need_heal":
                    tried.discard(tuple(door))   # entry never attempted — leave it retryable
                    return "need_heal"
                if r == "timeout":
                    tried.discard(tuple(door))
                    continue
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
                # TRANSIT LEARNING (2026-07-07, flute_run6): fold THIS interior into the mental map
                # NOW — mat warps fire on travel's arrival boundary, so neither travel's transition
                # hook nor enter_warp ever saw the UGP huts; the graph stayed one hop short and
                # anchor routing read 'no route Lavender->Celadon' over walked ground, three runs
                # straight. Each hop iteration stands on exactly the map that needs recording.
                self._learn_transit()
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
            # a crossing must pop out AWAY FROM ITS OWN ENTRY DOOR — the pos0-relative test alone
            # called the (12,86) rest house "crossed" because the search STARTED far away
            # (sabrina_run2: (15,119) -> in the door -> out beside it at (12,87), delta 35 > 6),
            # then _pt_known enshrined the false connector forever.
            d_door = abs(pos1[0] - door[0]) + abs(pos1[1] - door[1])
            if m_out != m0 or (d_door > 3 and abs(pos1[0] - pos0[0]) + abs(pos1[1] - pos0[1]) > 6):
                log(f"   PASSTHROUGH: CROSSED via door {door}: {m0}@{pos0} -> {m_out}@{pos1}")
                self.on_event("ha — knew it! through the building and out the other side. "
                              "the road's open.", kind="route", tier=2)
                # remember the proven connector (re-attempts after heal/battle interrupts reuse it)
                # — but NEVER enshrine a crossing that landed off the caller's want (the UGP hut
                # to Route 7 is a detour, not this map's road connector). SAME-MAP FENCE CROSSING
                # (2026-07-09, night-train shift 8, the Cerulean->Route5 loop): a burgled-house/
                # gatehouse door that pops back onto THIS map past an obstacle (the Cut tree at
                # Cerulean (26,32)) never satisfied m_out==want_map (want=Route5, m_out=Cerulean),
                # so the proven (30,11) door was forgotten every retry -> the passthrough re-searched,
                # picked wrong doors, dead-routed head_to_gym, and she wild-grind-looped instead of
                # re-crossing. A same-map crossing here already passed the strong d_door>3 & moved>6
                # guard above (so it's not the old false-positive), and a truly bad entry self-corrects
                # via the "forgot poisoned connector" branch below — remember it too.
                if want_map is None or m_out == tuple(want_map) or m_out == m0:
                    if not hasattr(self, "_pt_known"):
                        self._pt_known = {}
                    self._pt_known[m0] = tuple(door)
                tried.discard(tuple(door))
                # (the roam tick's note_visit warp-learning records the connector in the graph)
                return "crossed"
            # popped out beside where we entered (same building, two front doors) — keep trying
            log(f"   PASSTHROUGH: door {door} popped out beside the entry ({pos1}) — not a crossing")
            if getattr(self, "_pt_known", {}).get(m0) == tuple(door):
                # the remembered "proven" connector just failed to cross — it was a false positive;
                # forget it or it gets retried FIRST on every attempt, forever (the run-3 loop)
                del self._pt_known[m0]
                log(f"   PASSTHROUGH: forgot poisoned 'proven' connector {door} on {m0}")
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
                # struct MapConnection: direction@0, offset@4, mapGroup@8, mapNum@9 — our
                # convention is (group, num). Night shift 10: this read was TRANSPOSED
                # ((num, group)), sending travel to nonexistent maps like (19,3)/(39,3)
                # (Route 1/Route 21 flipped) — every re-entry leg no_routed forever.
                conns.append((d, off, (b.rd8(e + 8), b.rd8(e + 9))))
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
            pt = self._door_passthrough(want_map=tuple(target_map) if target_map else None)
            if pt == "need_heal":
                return "need_heal"
            if pt == "crossed":
                if tuple(tv.map_id(self.b)) == tuple(target_map):
                    return "arrived"                           # the connector itself crossed the map
                avoid = {tuple(w[0]) for w in tv.read_warps(self.b)}
                return self.trav.travel(target_map=target_map, edge=edge, avoid=avoid, **kw)
        return r

    def advance_north(self, target_map, max_legs=60, abort_when=None, leg_seconds=None,
                      flee_wilds=False, suppress_heal=False):
        """Generic 'head north until target_map' - auto-picks WALK (cross a route edge)
        vs WARP (step through a gate-house door) at each map, so the route->gate-house->
        route->forest chain self-discovers instead of being hardcoded leg by leg. The
        Traveler handles BFS/maze/NPC/grass/battle-handoff; this only sequences legs.

        abort_when/leg_seconds (quiet-window finding #2 — the SCRIPTED-INTERCEPT class): a
        cutscene can hijack the player mid-leg to a map that is neither here nor the target
        (Oak's grass-edge intercept -> the lab), after which the in-flight nav wedges against
        the script for minutes. abort_when is checked at every leg boundary (returns 'aborted');
        leg_seconds caps each leg's wall-clock so a hijacked leg yields fast. Defaults keep
        every existing caller byte-identical.

        flee_wilds/suppress_heal (2026-07-08 night train — the VIRIDIAN FOREST heal-bounce loop):
        a LONG grass crossing to a FAR Center (Viridian -> Route 2 -> Viridian Forest -> Pewter is
        ~140 grass tiles from the nearest Center) makes a thin solo starter drop below the heal
        floor MID-FOREST; the normal heal-when-low bounce then return_to_center's her ALL THE WAY
        back to Viridian and she re-enters to bounce again — an infinite no-progress loop (proven:
        the Forest NORTH gate (1,0)->(15,3)->Route2-north IS reachable, so it was never geometry).
        This is the SAME class deliver_parcel already tames for the Route-1 errand: FLEE wilds
        (chip-free crossing) + SUPPRESS the mid-leg heal-bounce so she crosses in one go. A genuine
        flee-fail death PROPAGATES as battle_loss -> the segment's blackout-recovery respawns her
        healed and re-runs the leg. Opt-in; defaults keep every existing caller byte-identical."""
        _saved_heal = self._suppress_heal
        _saved_runner = self.trav.battle_runner
        if suppress_heal:
            self._suppress_heal = True
        if flee_wilds:
            self.trav.battle_runner = self._flee_runner
        try:
            return self._advance_north_legs(target_map, max_legs, abort_when, leg_seconds)
        finally:
            self._suppress_heal = _saved_heal
            self.trav.battle_runner = _saved_runner

    def _advance_north_legs(self, target_map, max_legs=60, abort_when=None, leg_seconds=None):
        """The leg-sequencing loop (extracted so advance_north can wrap it with the flee/heal-
        suppress guard for long grass crossings without duplicating the loop)."""
        for leg in range(max_legs):
            if abort_when and abort_when():
                return "aborted"               # scripted intercept landed us elsewhere — yield
            m = tv.map_id(self.b)
            if m == target_map:
                return "arrived"
            _tkw = {"target_map": target_map, "max_steps": 800}
            if leg_seconds:
                _tkw["max_seconds"] = leg_seconds
            out = self.trav.travel(**_tkw)
            if abort_when and abort_when():
                return "aborted"               # hijack happened DURING the leg — don't warp/retry
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
        fainted_n = sum(1 for _, h, _, _ in ph if h == 0)
        lead_frac = next((f for sl, h, m, f in ph if sl == 0), 1.0)
        # CRITICAL = the FIGHTING CORE is in danger — not "any bench mon is down". celadon_run7:
        # a fainted L15 bench ekans behind a FULL-HP L45 lead collapsed the options to an
        # un-routable 'heal' one edge from Celadon and stalled the road. A real player walks a
        # fainted benchwarmer to the NEXT Center; they stop everything only when the team that
        # actually fights is about to black out.
        core_down = (lead_frac <= self.PARTY_CRIT_FRAC
                     or fainted_n >= max(1, len(ph) // 2)
                     or all(h == 0 or f <= self.PARTY_CRIT_FRAC for _, h, _, f in ph))
        if core_down:
            return ("critical",
                    f"Your team is badly hurt — {nm} is at {hp}/{mx} HP"
                    f"{f' and {fainted_n} teammate(s) are down' if fainted_n else ''}. A real trainer "
                    f"heals at a Pokémon Center before doing ANYTHING else — wandering into grass this "
                    f"hurt risks a blackout.")
        if fainted_n:
            return ("hurt",
                    f"{nm} is down ({fainted_n} fainted) — patch up at the next Pokémon Center on "
                    f"the way, but the road comes first while the rest of the team is standing.")
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
            # counts EXPENSIVE cross-city heals so grind() can bail a split-route heal-thrash (shift 13)
            self._heal_excursion_n = getattr(self, "_heal_excursion_n", 0) + 1
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
                if healed:
                    log(f"   HEAL-EXCURSION: healed + back on {back_map} at {tv.coords(self.b)}")
                    return "ok"
                # 2026-07-07 tunnel_run4: the remote PC was unreachable (heal-dead side of the
                # map), the round-trip still completed, and the unconditional 'ok' here banked a
                # fainted-ekans party as 'healed'. Honesty: no heal = no ok.
                log(f"   !! HEAL-EXCURSION: back on {back_map} but the party is NOT healed "
                    f"(remote Center unreachable) — honest fail")
                return "stuck"
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
            _EDGED = {"N": "north", "S": "south", "E": "east", "W": "west"}
            for leg in range(20):
                m = tv.map_id(self.b)
                if m == VIRIDIAN:
                    break
                # PER-LEG direction (2026-07-08, the Route-23 gate road): after a warp hop the fixed
                # north/south heuristic is stale — from Route 22 Viridian is EAST. When Viridian is a
                # live connection of THIS map, head straight at it.
                edge_leg = edge
                try:
                    for _d, _nbr in self._map_connections():
                        if tuple(_nbr) == tuple(VIRIDIAN) and _d in _EDGED:
                            edge_leg = _EDGED[_d]
                            break
                except Exception:
                    pass
                # Band pre-check (surf-aware): an edge the feet can't reach (split map / interior)
                # goes STRAIGHT to the warp attempt instead of burning a travel-wedge cycle.
                band_ok = True
                try:
                    _g = tv.Grid(self.b)
                    _w = _g.walkable_or_surf if self._surf_usable() else None
                    band_ok = self._edge_band_reachable(_g, tv.coords(self.b), edge_leg, walkable=_w)
                except Exception:
                    pass
                out = (self.trav.travel(target_map=VIRIDIAN, edge=edge_leg, max_steps=800)
                       if band_ok else "no_edge")
                if out == "arrived":
                    continue                       # crossed an edge toward Viridian
                if out == "battle_loss":
                    return "battle_loss"           # blacked out en route -> auto-heals at Viridian
                log(f"   HEAL-RETURN leg {leg}: no {edge_leg} edge on map {m} - warping {edge_leg}")
                if self.enter_warp(prefer=edge_leg) != "warped":
                    log(f"   !! HEAL-RETURN stuck on map {m} (no {edge_leg} edge, no {edge_leg} warp)")
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

    def _lead_attack_pp_low(self, floor=10):
        """True when the lead's DAMAGING moves are collectively low on PP — the PP-famine FORESIGHT
        signal that a status-move-heavy attacker masks. _lead_pp_low counts ANY move with PP, so
        Ivysaur (Razor Leaf + 2 powders) reads 'fine' even with Razor Leaf bone-dry — then it PP-famines
        mid-rival and loses a WINNABLE fight (the S.S. Anne Gary wall, night shift 4). Sums PP only over
        moves with base power > 0; below `floor` = not enough attacking PP to sustain a 4-mon rival fight,
        top off at a Center first. Best-effort; any read error -> False (never a spurious heal)."""
        try:
            moves, pps = self._lead_moves(), self._lead_pps()
            dmg_pp = sum(p for mid, p in zip(moves, pps)
                         if mid and p > 0 and (st.move_info(self.b, mid)[1] or 0) > 0)
            return dmg_pp < floor
        except Exception as e:
            log(f"   [roam] lead-attack-PP read skipped: {e}")
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
        # STAB AWARENESS (2026-07-09, night shift 15 — the Gary move-value bugs): the crude "raw power +
        # unique-coverage" score had TWO failures that gimped ivysaur for the rival Gary. (1) It PROTECTED
        # a weak NON-STAB filler (Tackle, 35 normal) with the full +60 unique-coverage bonus, so a strong
        # STAB level-up move (Razor Leaf, 55 grass) was declined and Tackle kept — a 1-effective-attacker
        # set. (2) It DROPPED a 2nd same-type STAB attacker (Vine Whip) as "low value" (35 power, no unique
        # bonus once Razor Leaf shares the type — VERIFIED live: _ensure_move_room turned a 2-grass-attacker
        # ivysaur back into one, mid-run). Fix: score a move's STAB (type matches the mon) as PRECIOUS, and
        # don't hand a weak non-STAB filler the unique-coverage prize. Read the lead's own types once.
        try:
            lead_types = set(st.species_types(st.read_party_species(self.b, 0)))
        except Exception:
            lead_types = set()
        # REDUNDANT-STATUS DEMOTION (2026-07-10, night shift 6 — the Route-6 flying/bug PP-famine wall):
        # HIGH_VALUE_LOW_POWER hands EVERY status move the full +100, so a mon hoarding TWO powders
        # (Ivysaur = Sleep Powder + Poison Powder) protected them over its only NEUTRAL attacker (Tackle),
        # leaving a grass-ONLY kit that gets resisted x0.25-0.5 by every Pidgey/Butterfree -> PP famine ->
        # loss. A guide-literate trainer keeps ONE status opener, not two. Elect a single primary status
        # (Sleep > Leech Seed > any) at full value; a 2nd redundant status is a luxury a coverage-poor mon
        # can't afford -> it drops before the neutral attacker / STAB backup.
        _status_pref = {79: 3, 73: 2}                         # sleep-powder, leech-seed = best openers
        _status_slots = [x for x in info if x[1] in HIGH_VALUE_LOW_POWER]
        _primary_status = (max(_status_slots, key=lambda x: (_status_pref.get(x[1], 1), -x[1]))[0]
                           if _status_slots else None)

        def _value(x):                                        # higher = more precious = keep
            _slot, mid, _n, t, power = x
            v = (power or 0)
            if mid in HIGH_VALUE_LOW_POWER:
                v += 100 if _slot == _primary_status else 25  # ONE opener precious; a 2nd status = luxury
            is_stab = (power or 0) > 0 and t in lead_types
            if is_stab:
                v += 40                                       # STAB attacker — keep it (2 STABs is NOT junk)
            # UNIQUE-type coverage is precious — but a WEAK NON-STAB filler (Tackle 35) doesn't earn it.
            if (power or 0) > 0 and dmg_types.get(t, 0) == 1 and ((power or 0) >= 40 or is_stab):
                v += 60                                       # UNIQUE-type coverage (Vine Whip!) — precious
            # OFF-TYPE COVERAGE LIFELINE: a low-power NEUTRAL/off-type attacker (Tackle, normal) is the ONLY
            # move that dents a foe RESISTING the mon's STAB. When every STAB attacker shares one commonly-
            # resisted type (grass is walled by flying/bug/poison/steel/fire/grass), that neutral move is
            # the coverage lifeline — protect it over a redundant status move even at <40 power.
            if ((power or 0) > 0 and t not in lead_types and dmg_types.get(t, 0) == 1 and (power or 0) < 40
                    and any((p2 or 0) > 0 and t2 in lead_types for _i, _m, _n2, t2, p2 in info)):
                v += 55                                       # off-type coverage on a STAB-locked mon
            return v

        # the most PRECIOUS move (by VALUE, not raw power) — NEVER drop it. Was max-by-power, which on a
        # power tie (Tackle 35 == Vine Whip 35) wrongly protected the non-STAB filler and left the STAB in
        # the drop pool. Value-ranking keeps the STAB/coverage move and lets the filler go.
        best_slot = max(info, key=_value)[0]

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
        # DAMAGE-POVERTY OVERRIDE (2026-07-09, night shift 15 — the S.S. Anne rival-Gary PP-famine
        # wall): the no-gut guarantee treats a FULL set of 3 status + 1 attacker as "all precious",
        # so a strong damaging level-up move (ivysaur's Razor Leaf at L20) is DECLINED in-battle and
        # she is left with ONE damaging move -> PP famine loses WINNABLE fights (she outlevels Gary
        # by 10+ but runs Vine Whip dry and can't finish). A moveset with <=1 damaging move is NOT
        # "all precious" — the redundancy IS the junk. When the mon is damage-poor, free ONE slot (the
        # least-valuable REDUNDANT move) so the upcoming attacker auto-learns. Protect the hard-hitter
        # (never in `droppable`) AND one sleep/control move; a 2nd attacker is nearly always coming and
        # is worth a redundant powder/leech. Fail-safe: only fires with a full set + a safe shed cand.
        _SLEEP_IDS = {79, 147, 95, 47, 142}                   # Sleep Powder/Spore/Hypnosis/Sing/L.Kiss
        damage_poor = sum(1 for x in info if (x[4] or 0) > 0) <= 1
        poverty_shed = None
        if damage_poor and len(real) >= 4:
            _sleeps = sorted([x for x in droppable if x[1] in _SLEEP_IDS], key=_value, reverse=True)
            _keep_sleep = _sleeps[0][0] if _sleeps else None  # keep her single best control move
            _shed_cands = [x for x in droppable if x[0] != _keep_sleep]
            if _shed_cands:
                poverty_shed = sorted(_shed_cands, key=_value)[0]
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
        # EXCEPT the damage-poverty override: a <=1-attacker set must shed a redundant move so the next
        # attacker can land (else PP famine — the Gary wall). That override wins over keep-all here.
        if (pick == "keep them all" or (not pick and all_precious)) and not poverty_shed:
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
        if drop_slot is None and poverty_shed and (pick in (None, "keep them all") or all_precious):
            # damage-poor + (she'd keep-all / no clear junk): shed the redundant move for an attacker
            drop_slot = poverty_shed[0]
            log(f"   [movelearn] DAMAGE-POVERTY: only {sum(1 for x in info if (x[4] or 0) > 0)} damaging "
                f"move — freeing {st.MOVE_NAMES.get(moves[drop_slot], moves[drop_slot])} (redundant; kept the "
                f"hard-hitter + a control move) so an upcoming attacker auto-learns (no PP-famine set)")
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

    def _evo_box_name(self):
        """EARLY DETECT for the evolution scene: scan the gStringVar block (the buffer the
        scene's text printer expands into) for the START box — 'What? X is evolving!'.
        Returns the on-screen name (nickname-honest) or None. The species byte only flips
        at the END of the ~20-30s animation, so this box is the only early signal."""
        try:
            import dialogue_reader as dr
            raw = self.b.read_bytes(dr.GSTRINGVAR_LO, dr.GSTRINGVAR_HI - dr.GSTRINGVAR_LO)
            for run in bytes(raw).split(b"\xff"):
                if not run:
                    continue
                txt, junk = dr.decode(run)
                if junk <= 0.3 and "evolving" in txt.lower():
                    import re as _re
                    m = _re.search(r"(\S+)\s+is\s+evolving", txt.replace("\n", " "),
                                   _re.IGNORECASE)
                    return (m.group(1).strip(".!? ").title() if m else "my Pokemon")
        except Exception:
            pass
        return None

    def _drive_evolution(self, pre_species, pre_level, pre_levels=None, pre_specs=None):
        """PHASE 3A — drive the post-battle EVOLUTION cutscene so it NEVER freezes. After a battle the
        evolution scene plays on the overworld (a full-screen animation box_open can't see), and the old
        code's _wait_overworld returned 'idle' while input was still locked -> the next action wedged.

        GATE: evolution only happens on a LEVEL-UP, so if nobody leveled this battle we return
        immediately with ZERO presses (no spurious steps on the common no-evolution case). Night shift
        11: the gate is ANY-SLOT when the caller snapshots the party (pre_levels) — the old lead-only
        read missed a fielded/switched-in bench mon's evolution and left the cutscene undriven. On a
        level-up we settle (plain frames), check ONCE whether control is locked (a cutscene playing); if
        it's free there's nothing to drive (just a normal level-up) -> no-op. If LOCKED, we A-advance
        (B-free: A proceeds an evolution, B would CANCEL it — we never press B) until control returns /
        timeout. Default-proceed = evolving is almost always wanted (a 30s oracle call can't fit the
        few-second cancel window, so cancel isn't wired — proceed is the honest default).

        EARLY BEAT (F-7c sibling, night shift 11): the 'What? X is evolving!' box sits on screen for
        the whole animation — emit ONCE the moment it's read so her reaction's LLM chain overlaps the
        animation and the line lands ON the white-flash, not 30s later on the overworld. On a real
        species change we fire the NAMING soul beat (a new family member)."""
        if pre_levels is not None:
            if not any(c > p for c, p in zip(self._party_levels(), pre_levels)):
                return                                        # nobody leveled -> no evolution possible
        else:                                                 # legacy caller: lead-only gate
            cur_level = self.b.rd8(ram.GPLAYER_PARTY + self._PARTY_LEVEL_OFF)
            if not pre_level or cur_level <= pre_level:
                return
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
        evo_beat_done = False
        # A-advance (B-free) until control returns or we time out (evolution anim can run ~20-30s)
        while _t.time() - t0 < 35:
            if st.in_battle(self.b):
                return
            if dd._control_returned():
                break
            if not evo_beat_done:
                _nm = self._evo_box_name()
                if _nm:
                    evo_beat_done = True
                    log(f"   [evolve] EARLY BEAT: '{_nm} is evolving!' box read — emitting now "
                        f"so the line lands on the animation")
                    self.on_event(f"wait — {_nm} is evolving!", kind="roster", tier=3)
            drove = True
            self.b.press("A", 6, 10, self.render, owner="agent")
            self.render()
        # SLOT-WISE species diff (any-slot honesty; safe here — no reorder can happen inside
        # the cutscene, so a changed slot IS an evolution, not a swap. Falls back to the
        # legacy lead-only compare when the caller didn't snapshot).
        changed = None
        if pre_specs:
            post_specs = [st.read_party_species(self.b, s) for s in range(len(pre_specs))]
            for _s in range(len(pre_specs)):
                if pre_specs[_s] and post_specs[_s] and post_specs[_s] != pre_specs[_s]:
                    changed = (pre_specs[_s], post_specs[_s])
                    break
        else:
            post = st.read_party_species(self.b, 0)
            if pre_species and post and post != pre_species:
                changed = (pre_species, post)
        if changed:
            before = st.SPECIES_NAME.get(changed[0], "my Pokemon")
            after = st.SPECIES_NAME.get(changed[1], "something new")
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
    # Map-id -> name for her ctx `place` line. Mirrors pokemon_world.SEED_NODES (the world-model's
    # own table) so her decision context and her mental map agree. Endgame/post-game ids VERIFIED vs
    # the pret disasm — this is what stops read_live_state emitting "(1,80) an unfamiliar area".
    _PLACE_NAMES = {
        (3, 0): "Pallet Town", (3, 1): "Viridian City", (3, 2): "Pewter City",
        (3, 3): "Cerulean City", (3, 4): "Lavender Town", (3, 5): "Vermilion City",
        (3, 6): "Celadon City", (3, 7): "Fuchsia City", (3, 8): "Cinnabar Island",
        (3, 9): "Indigo Plateau", (3, 10): "Saffron City",
        # F-8 round 2 (2026-07-08 night): FULL-ROPE name coverage, indices from the pret disasm
        # map_groups.json cross-checked against live RAM (Route 12=(3,30) matched banked_SNORLAX;
        # Victory Road / E4 / Cerulean Cave numbers matched the live-verified seeds). Route N =
        # (3, 18+N) holds through Route 20; Route 21 splits north/south, shifting 22-25 by one.
        # Kills the "an unfamiliar area" reads across the whole descent (unknown-place framing is
        # for genuinely new ground, not for roads she's already crossed).
        **{(3, 18 + n): f"Route {n}" for n in range(1, 21)},
        (3, 39): "Route 21", (3, 40): "Route 21",
        (3, 41): "Route 22", (3, 42): "Route 23", (3, 43): "Route 24", (3, 44): "Route 25",
        (1, 0): "Viridian Forest",
        (1, 1): "Mt. Moon", (1, 2): "Mt. Moon B1F", (1, 3): "Mt. Moon B2F",
        **{(1, n): "the S.S. Anne" for n in range(4, 30)},
        **{(1, n): "the Underground Path" for n in range(30, 36)},
        **{(1, n): "Diglett's Cave" for n in range(36, 39)},
        (1, 39): "Victory Road", (1, 40): "Victory Road 2F", (1, 41): "Victory Road 3F",
        **{(1, n): "the Rocket Hideout" for n in range(42, 47)},
        **{(1, n): "Silph Co." for n in range(47, 59)},
        **{(1, n): "the Pokémon Mansion" for n in range(59, 63)},
        **{(1, n): "the Safari Zone" for n in range(63, 72)},
        (1, 72): "Cerulean Cave", (1, 73): "Cerulean Cave 2F", (1, 74): "Cerulean Cave B1F",
        (1, 75): "Lorelei's Room", (1, 76): "Bruno's Room",
        (1, 77): "Agatha's Room", (1, 78): "Lance's Room", (1, 79): "the Champion's Room",
        (1, 80): "the Hall of Fame",
        (1, 81): "Rock Tunnel", (1, 82): "Rock Tunnel B1F",
        **{(1, n): "Seafoam Islands" for n in range(83, 88)},
        **{(1, n): "Pokémon Tower" for n in range(88, 95)},
        (1, 95): "the Power Plant",
        # Pallet interiors (group 4 — quiet-window rehearsal finding #4: the FIRST grudge beat read
        # "at an unfamiliar area" because Oak's lab had no name; these are the opening's stage).
        (4, 0): "home", (4, 1): "her bedroom", (4, 2): "Gary's house", (4, 3): "Oak's lab",
        # F-8 round 3 (2026-07-08 shift 3, descent pre-grade payload): CITY-INTERIOR groups.
        # Ground truth: interiors group = city index + 4, locked by SIX live anchors — Pallet=4
        # (this table), Cerulean=7 (Mart interior (7,7), campaign MART_SPECS live), Lavender=8
        # (banked_FLUTE boots (8,2) = Mr. Fuji's house, frame-verified tonight), Vermilion=9
        # (env_puzzle VERMILION_GYM=(9,7) live), Celadon=10 (gym interior (10,16) live),
        # Saffron=14 (SILPH arc's head_to_gym walked her inside (14,3) = Sabrina's gym).
        # Generic "a building in X" is TRUE and kills the "unfamiliar area" lie on ground the
        # rope crossed; specific rooms override only where live-verified.
        **{(4 + ci, n): f"a building in {city}" for ci, city in enumerate((
            "Pallet Town", "Viridian City", "Pewter City", "Cerulean City", "Lavender Town",
            "Vermilion City", "Celadon City", "Fuchsia City", "Cinnabar Island",
            "Indigo Plateau", "Saffron City")) for n in range(24) if (4 + ci, n) not in (
            (4, 0), (4, 1), (4, 2), (4, 3))},
        (7, 5): "the Cerulean Gym", (7, 7): "the Cerulean Poké Mart",
        (8, 0): "the Lavender Pokémon Center", (8, 2): "Mr. Fuji's house",
        (9, 7): "the Vermilion Gym",
        (10, 16): "the Celadon Gym",
        (14, 3): "the Saffron Gym — Sabrina's teleport maze",
        # Shift 12 (2026-07-08, descent-sweep log ground truth): gate/route-building groups.
        # (19,0) warps live-learned BOTH ways — (1,5-6)->Route 7 (3,25) and (11,5-6)->Saffron
        # (3,10) — so it IS the Route 7 gate; (21,0) proven by "HEAL: exited Center ->
        # (3,28)" (Route 10). Named ONLY where a sweep log proves the identity (standing
        # rule: disasm map numbers lie; live RAM is the truth).
        (19, 0): "the Route 7–Saffron gate",
        (21, 0): "the Route 10 Pokémon Center",
    }
    _HALL_MAPS = {(1, 80), (1, 79)}       # standing here with all 8 badges == she beat the game
    _BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
    _GYM_ORDER = [("Pewter City", "Brock"), ("Cerulean City", "Misty"),
                  ("Vermilion City", "Lt. Surge"), ("Celadon City", "Erika"),
                  ("Fuchsia City", "Koga"), ("Saffron City", "Sabrina"),
                  ("Cinnabar Island", "Blaine"), ("Viridian City", "Giovanni")]
    _PARTY_LEVEL_OFF = 0x54        # level byte within a 100-byte party-mon struct (== play_live LEAD_LEVEL)

    def _place_name(self, mid, default="an unfamiliar area"):
        """Best HONEST name for a map (shift 13, bedrock #11). Static table first; else DERIVE one
        from the live-learned world graph: an unnamed interior whose learned warps lead to named
        maps reads as 'the passage between X and Y' (the gatehouse/connection-building class —
        group 19 etc., where hardcoding names violates the disasm-numbers-lie rule) or 'a building
        off X'. Derivation only speaks from where her own feet have warped — no warps learned yet
        means the honest default stands. Narration-only; never feeds routing."""
        nm = self._PLACE_NAMES.get(mid)
        if nm:
            return nm
        try:
            node = self.world.node(mid)
            dests = []
            for d in (node or {}).get("warps", {}).values():
                g, n = (int(p) for p in str(d).split(","))
                dn = self._PLACE_NAMES.get((g, n))
                if dn and dn not in dests:
                    dests.append(dn)
            if len(dests) >= 2:
                return f"the passage between {dests[0]} and {dests[1]}"
            if len(dests) == 1:
                return f"a building off {dests[0]}"
        except Exception:
            pass
        return default

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
        place = self._place_name(mp)
        dex = ram.pokedex_owned_count(b)          # BATCH 6 PHASE 4: caught-count from RAM (no menu)
        arc = self._arc_note(len(badges))         # PHASE 4: where she is in the WHOLE journey (momentum)
        # POST-GAME (the summit-watch strand fix): the Champion flag is only set AFTER the post-Hall
        # warp-home cutscene, so a save banked IN the Hall of Fame won't have it yet — standing in the
        # Hall/Champion room WITH all 8 badges is unambiguous proof she won. Either signal => post-game.
        # Firewalled: false for every pre-credits state, so nothing here can touch the normal climb.
        game_clear = self.has_badge(0x82C)        # FLAG_SYS_GAME_CLEAR (SYS_FLAGS + 0x2C)
        post_game = (len(badges) >= 8) and (game_clear or mp in self._HALL_MAPS)
        # GOAL-PINNED WATCH SPAWNS (2026-07-08): every banked save is post-credits, so a spawn at
        # e.g. victory-road makes the Champion pursue post-game goals (Mewtwo) instead of the moment
        # Jonny wants to SEE ("fight through the Elite Four", "beat Sabrina"). POKEMON_WATCH_GOAL pins
        # an era-correct objective: it SUPPRESSES the post-game victory-lap frame and becomes her
        # dominant directive (injected in _spine_and_history / _goal_layers). Read-only override — RAM
        # is still ground truth for battles; this only reshapes her CONTEXT/goal. Empty = normal.
        watch_goal = os.getenv("POKEMON_WATCH_GOAL", "").strip()
        if watch_goal:
            post_game = False                     # re-living an era moment, not on the victory lap
            # NEXT-GYM OVERRIDE: a post-credits save has 8 badges, so next_gym derives to None and
            # head_to_gym has nowhere to route — the gym clip wouldn't ENTER the gym. POKEMON_WATCH_
            # NEXT_GYM (leader name substring) forces the era gym target so she paths into it. Only
            # for the gym clips; the E4 clip leaves it unset (badges=8 -> the E4 is the forward path).
            _wg = os.getenv("POKEMON_WATCH_NEXT_GYM", "").strip().lower()
            if _wg:
                for _c, _l in self._GYM_ORDER:
                    if _wg in _l.lower():
                        ng = (_c, _l); break
        if post_game:                             # never tell the Champion the E4 is "next"
            arc = ("Champion. The credits rolled — everything from here is the victory lap "
                   "(Cerulean Cave, the Pokédex, her world now).")
        progress = (f"{len(badges)} badge(s) earned ({', '.join(badges) or 'none'}). "
                    + ("You're the CHAMPION — the game is beaten. This is the post-game victory lap."
                       if post_game else
                       (f"Next gym: {ng[1]} of {ng[0]}." if ng else "All 8 badges earned."))
                    + (f" Pokédex: {dex} caught." if dex is not None else "")
                    + f" {arc}")
        if watch_goal:
            progress = f"🎯 RIGHT NOW YOUR ONE OBJECTIVE IS: {watch_goal}. " + progress
        return {"map": mp, "place": place, "coords": co,
                "badges": badges, "badge_count": len(badges),
                "party": party, "party_count": cnt,
                "on_grass_map": on_grass_map, "dex_caught": dex,
                "next_gym": ({"city": ng[0], "leader": ng[1]} if ng else None),
                "post_game": post_game, "watch_goal": watch_goal or None,
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

    # ── F-8 GROUNDED PERCEPTION (the descent) — the LOCATION CONTEXT BLOCK ──────────────────────
    # Every soul tick (and every want ask) now LEADS with a grounded where-am-I line built ONLY
    # from live reads: map GROUP is the indoor/outdoor truth (group 3 = TownsAndRoutes under open
    # sky, group 1 = gMapGroup_Dungeons = the only real caves, anything else = a building
    # interior), grass from the live Grid, town/Mart/Center from the world-model's curated seeds +
    # live-confirmed visits. This is the decision-side half of the F-8 fix (travel's _muse_seed was
    # the harness half): her oracle can no longer confabulate a cave inside Oak's lab, and an
    # UNKNOWN place explicitly tells her she doesn't know it — impressions stay curiosity or
    # questions, never asserted facts (the confabulation killer).
    _LOC_GROUP_DUNGEONS = 1        # pret gMapGroup_Dungeons — the only group that is truly a cave
    _LOC_GROUP_OVERWORLD = 3       # TownsAndRoutes — the only group under open sky

    def _location_block(self, state):
        mp = tuple(state["map"])
        place = state["place"]
        known = mp in self._PLACE_NAMES
        # group 1 is MIXED (caves + ships + towers + the open-air Safari Zone) — classify by
        # map number via travel's disasm-grounded tables, never by group alone (the Hall-of-Fame-
        # as-cave bug this replaced).
        if (mp[0] == self._LOC_GROUP_OVERWORLD
                or (mp[0] == self._LOC_GROUP_DUNGEONS and mp[1] in tv.G1_OUTDOOR)):
            setting = "outdoors under open sky"
        elif mp[0] == self._LOC_GROUP_DUNGEONS and mp[1] in tv.G1_CAVES:
            setting = "underground in a real cave — rock, gloom, echoes"
        else:
            setting = "indoors, inside a building (rooms and doors — NOT a cave)"
        bits = []
        try:
            node = self.world.nodes.get(f"{mp[0]},{mp[1]}") or {}
            tr = node.get("traits", {}) or {}
            if tr.get("is_town"):
                has = [n for f, n in (("has_pokecenter", "a Pokémon Center"),
                                      ("has_mart", "a Mart")) if tr.get(f)]
                bits.append("a town" + (f" with {' and '.join(has)}" if has else ""))
            elif tr.get("is_route"):
                bits.append("an open route")
        except Exception:
            pass
        if state.get("on_grass_map"):
            bits.append("tall grass around — wild Pokémon live in it")
        detail = "; ".join(bits)
        if known:
            prep = ("on" if place.startswith("Route")
                    else "at" if place in ("home",) else "in")
            return (f"You're {prep} {place} — {setting}"
                    + (f" ({detail})" if detail else "") + ".")
        return (f"You're {setting}, somewhere you DON'T recognize — you have no name for this "
                "place yet. Only claim what you can actually see; everything else stays a guess "
                "or a wondering-out-loud, never a stated fact.")

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
        """Walk to a LAND-standable tile adjacent to a junior trainer, face it, and let its line of
        sight fire the battle (a trainer can't be A-talked into a fight - it must SEE the player).
        POOL-GYM FIX (the Misty stuck, 2026-07-09): a swimmer stands ON water with only ONE
        land-standable side (Cerulean's (10,12) is engageable only from (9,12); the other 3 fronts
        are water tiles travel WEDGES on -> the old code wedged x4 then FALSELY marked it beaten ->
        walked into a still-gated Misty). So we (a) attempt ONLY standable fronts, and (b) after
        arriving on the sight axis, NUDGE off-and-back so the trainer's 'player entered sight' edge
        re-fires even when we arrived already standing inside its line. True iff a battle started."""
        try:
            grid = tv.Grid(self.b)
        except Exception:
            grid = None
        order = [self._DELTA.get(facing, (0, 1)), (0, 1), (0, -1), (-1, 0), (1, 0)]
        seen = set()
        for adj in order:
            if adj in seen:
                continue
            seen.add(adj)
            front = (T[0] + adj[0], T[1] + adj[1])
            if grid is not None and not grid.walkable(*front):
                continue                                       # water/wall front - travel would wedge
            self._gym_move(front, label="engage")
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
            # LoS didn't fire while we stood still (we arrived already inside its sight, so there was
            # no 'entered' edge). RE-TRIGGER: step to an adjacent standable tile and back onto the
            # front, forcing a fresh sight check. Each neighbour tried once.
            for nb in ((front[0], front[1] + 1), (front[0], front[1] - 1),
                       (front[0] - 1, front[1]), (front[0] + 1, front[1])):
                if nb == T:
                    continue
                if grid is not None and not grid.walkable(*nb):
                    continue
                self._gym_move(nb, label="engage-nudge")
                if st.in_battle(self.b):
                    return True
                self._gym_move(front, label="engage-reenter")
                for _ in range(10):
                    if st.in_battle(self.b):
                        return True
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

    # ── F-6 SOCIAL FABRIC (the descent) — KEY-FIGURE SALIENCE ───────────────────────────────────
    # A real player doesn't treat their mom like scenery. Key FIGURES (family, quest-adjacent
    # people) exert PULL on her intents: when one is here and un-greeted, a first-class
    # `greet:<id>` option appears with a warm framing, the pull rides the ctx `place` seam, and
    # walking out anyway gets VOICED (a deliberate skip is a choice a viewer should hear, not a
    # silent beeline). SHE still decides — salience feeds the choice, the pathfinder executes.
    # Greeted-state persists in world_model.json (sanctity bundle) so it survives resume.
    # GAME-KNOWLEDGE LAYER (portability): map-keyed, curated small; grows as the descent grades
    # arcs. Gym leaders / scripted quest NPCs are NOT here (beat_gym + questlines own those).
    # P-1(b) GYM FOLKLORE (game-knowledge layer): what trainers-on-the-road say about each
    # leader — type + one folksy implication. Surfaced in the spine as "word on the road",
    # so her anticipation habit has real material without ever reading like a walkthrough.
    _GYM_FOLKLORE = {
        "Brock": "he runs ROCK-types — folks say fire and flying moves bounce right off; "
                 "grass or water is how challengers get through",
        "Misty": "she runs WATER-types — a whole gym over a pool; grass and electric do the work",
        "Lt. Surge": "the lightning American — ELECTRIC-types; ground moves shut his whole act down",
        "Erika": "the flower lady — GRASS and poison; fire melts her garden, flyers do fine too",
        "Koga": "a literal ninja — POISON and confusion tricks; bring antidotes and patience",
        "Sabrina": "people whisper about her — PSYCHIC-types, sees your moves coming; "
                   "ghosts and bugs are the folk remedy",
        "Blaine": "the old fire scientist — FIRE-types on a volcano island; water washes him out",
        "Giovanni": "nobody talks straight about the Viridian leader — GROUND-types, they say, "
                    "and something colder underneath",
        # E4 + CHAMPION (2026-07-08 mega-batch): the folklore was gym-only, so at 8 badges (next_gym
        # None) the run's CLIMAX had zero 'word on the road'. These fire from the badges==8 branch of
        # _spine_and_history. Rumor register (flavor); the StrategicPlanner carries the actionable prep.
        "Lorelei": "the Elite Four open with an ice-and-water lady — folks say lightning and grass are "
                   "how you slip past her",
        "Bruno": "then comes the muscle, all Fighting and Rock — they reckon a Psychic type turns him to jelly",
        "Agatha": "then a ghost-and-poison woman — Psychic answers her, and Normal moves just phase "
                  "clean through her ghosts",
        "Lance": "then the dragon master himself — and everyone says the same thing: ice is the only "
                 "thing a dragon truly fears",
        "Gary": "and at the very top… him. your rival. whatever his team's become by now, this is the "
                "one the whole road's been building to",
    }

    _SALIENT_FIGURES = {
        (4, 0): [{"id": "mom", "who": "Mom",
                  "offer": "talk to Mom — she's right here, and you don't walk out on her "
                           "without a word",
                  "pull": "Your mom is here in the house with you — you haven't talked to "
                          "her yet."}],
        (4, 2): [{"id": "daisy", "who": "Daisy, Gary's sister",
                  "offer": "say hi to Daisy, Gary's sister — she's friendly, and she knows "
                           "things (and people say she gives travelers a Town Map)",
                  "pull": "Daisy — Gary's sister — is here, and you've never actually met."}],
    }

    # ── P-1(a) FIRSTS-AS-EVENTS (couch fix-pass 1) ─────────────────────────────────────────
    # A first-timer's firsts are EVENTS, not footnotes: first wild battle, first trainer
    # fight, first catch, first Center, first Mart each get ONE T2 beat. Persisted via the
    # world-model met() store (survives resume; absent on a fresh run). EARLY-GAME GATED:
    # with any badge earned the first is backfilled SILENTLY — the Champion revisiting a
    # Mart must never gush "my first shop ever".
    def _first_beat(self, key, text):
        try:
            k = f"first_{key}"
            if self.world.met(k):
                return
            self.world.mark_met(k)
            if sum(1 for i in range(8) if self.has_badge(0x820 + i)) > 0:
                return                              # veteran world — backfill, no beat
            self.on_event(text, kind="first", tier=2)
        except Exception as _fb:
            log(f"   [first] {key} beat skipped ({_fb!r}) (LOUD)")

    def _salient_unmet(self, state):
        """Un-greeted key figures on THIS map with a talkable body actually present (honest
        offer — a figure who isn't loaded right now is not offered)."""
        figs = self._SALIENT_FIGURES.get(tuple(state["map"]))
        if not figs:
            return []
        try:
            present = bool(self._talkable_npcs())
        except Exception:
            present = False
        if not present:
            return []
        return [f for f in figs if not self.world.met(f["id"])]

    def _greet_figure(self, fid, state):
        """Execute a greet:<id> pick — the existing tracked-talk primitive does the walking/
        facing/box-verify; success marks the figure met (persisted). A figure we provably can't
        reach resolves met-anyway (logged LOUD) so the offer can never become a forever-loop."""
        figs = self._SALIENT_FIGURES.get(tuple(state["map"]), [])
        fig = next((f for f in figs if f["id"] == fid), None)
        if fig is None:
            return "unknown_action"
        out = self.talk_npc()
        if out == "talked":
            self.world.mark_met(fid)
            log(f"   [social] F-6: greeted {fig['who']} — marked met (persists via world_model)")
            return "greeted"
        self.world.mark_met(fid)
        log(f"   [social] !! F-6: couldn't reach {fig['who']} ({out}) — resolving met-anyway "
            f"so the offer can't loop (LOUD; if this repeats on a graded arc, fix the reach)")
        return out

    def _social_tick(self, state):
        """F-6 skip-voicing: she was OFFERED a greet last tick and left the map without taking
        it -> voice the deliberate skip ONCE per figure per run (her LLM frames it; the seed is
        a neutral event). Keeps salience honest: pull, not compulsion."""
        cur = tuple(state["map"])
        prev_map, prev_ids = getattr(self, "_social_last", (None, ()))
        if prev_map is not None and cur != prev_map:
            for fid, who in prev_ids:
                if not self.world.met(fid) and fid not in getattr(self, "_social_skip_voiced", set()):
                    self._social_skip_voiced = getattr(self, "_social_skip_voiced", set()) | {fid}
                    log(f"   [social] F-6: she left {prev_map} without greeting {who} — voicing the skip")
                    self.on_event(f"you headed out without saying anything to {who}",
                                  kind="social", tier=1)
        self._social_last = (cur, tuple((f["id"], f["who"]) for f in self._salient_unmet(state)))

    def _drain_overworld(self, label="dlg"):
        """Drive any overworld dialogue box (a trainer's post-battle line, an NPC, the badge award)
        to a clean close at a watchable pace via the general primitive. No box -> returns at once.
        Returns the drive result. B-2: on 'exhausted' (the cycle-watchdog bailed on a LOOPING NPC),
        records this spot so she won't re-engage it (the full re-talking-beaten-NPC class, not just
        the literal Slowbro instance)."""
        # LIVE-BATTLE GUARD (2026-07-07, erika_run2): a stuck-abort can leave the BATTLE still live
        # (e.g. its forced-switch party menu up). Draining then A-mashes battle menus blind — 300
        # presses selecting fainted mons ("PERSIAN has no energy left!" forever). Battle UI belongs
        # to the battle engine, never this overworld primitive; the caller's loop re-enters the fight.
        if st.in_battle(self.b):
            log(f"   [roam] !! drain '{label}' requested during a LIVE battle -> skipping "
                f"(battle engine owns that UI)")
            return "in_battle"
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

    def _party_damaging_pp(self):
        """True iff ANY alive party member still has PP on a damaging move — the PP-famine ground
        truth (erika_run2 2026-07-07: the whole gauntlet ran dry and every battle went 'stuck').
        Defensive: any read error returns True (the check is a detour trigger, never a wall)."""
        try:
            cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            for s in range(min(cnt, 6)):
                if self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                    continue
                if st.slot_has_damaging_pp(self.b, s):
                    return True
            return False
        except Exception:
            return True

    def _party_gym_ready(self):
        """Battle-readiness for a GYM push (a dense, no-Center gauntlet): no fainted mons, the ace
        above half HP, and damaging PP somewhere in the party. A human ALWAYS taps the Center
        before a gym — and gym cities always have one, so the heal is cheap and returns to the
        pre-heal spot. Conservative on read errors (ready) — the gate detours, never walls."""
        try:
            cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
            best_lv, best_frac = -1, 1.0
            for s in range(min(cnt, 6)):
                hp = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
                mx = self.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
                if hp <= 0:
                    return False                       # a faint anywhere -> Center first (it's in town)
                lv = self.b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54)
                if lv > best_lv and mx > 0:
                    best_lv, best_frac = lv, hp / mx
            if best_frac < 0.5:
                return False                           # the ace is half-dead -> Center first
            return self._party_damaging_pp()
        except Exception:
            return True

    def _spin_assist(self, tile):
        """Injected into travel (the field_clear pattern): when a leg wedges on a SPINNER
        floor (forced-slide tiles partition the walkable pockets — hideout B2F/B3F class),
        glide-cross toward `tile` with spin_nav's deterministic slide simulator. True =
        pockets crossed (travel re-plans from there); False/raise = travel surfaces its
        wedge exactly as before. Battles mid-glide run through the real battle_runner."""
        import spin_nav
        t = tuple(tile)
        sn = spin_nav.SpinNav(self.b, self, self.battle_runner,
                              lambda: self._drain_overworld(label="spin-assist"), log=log)
        return bool(sn.cross(lambda c, tt=t: abs(c[0] - tt[0]) + abs(c[1] - tt[1]) <= 1,
                             "travel-assist", rounds=2))

    def _gym_move(self, tile, label="gym-move"):
        """Interior mover for gym handling: PAD-AWARE when this gym's router is armed (a
        warp-partitioned interior — travel's BFS is warp-blind and reads teleport pads as
        plain floor, so its paths ride pads mid-route), plain travel otherwise. True =
        standing on tile. The router is armed per-gym in beat_gym (pad_nav.same_map_pads)."""
        pn = getattr(self, "_gym_pads", None)
        if pn is not None and tuple(tv.map_id(self.b)) == pn.map:
            if pn.goto_region({tile}, label=label):
                pn.walk(tile, label=label)
            return tuple(tv.coords(self.b) or ()) == tile
        self.trav.travel(target_map=None, arrive_coord=tile, max_steps=200, max_seconds=90)
        return tv.coords(self.b) == tile

    def _los_retrigger(self, front, avoid=None):
        """Force a trainer's line-of-sight to re-fire by stepping OFF `front` and back ON (a trainer
        only initiates on the player's STEP INTO its sight tile; arriving via a zero-step travel —
        already standing there — leaves it un-triggered, so an A-press opens a GREETING box instead
        of the battle: the Misty stuck, where she sits at (8,6) looking DOWN onto the (8,7) front).
        Steps to each walkable neighbour of `front` and back once. True iff a battle started.

        `avoid` = the LEADER's OWN tile (front stepped in leader_dir). It must be EXCLUDED (Fix A,
        2026-07-09): the static collision grid isn't NPC-aware, so grid.walkable reports the leader's
        tile as free — the nudge then travels ONTO it, and travel classifies the standing leader as a
        'PLAIN NPC blocker' and burns its whole budget trying to route AROUND him (the Brock (6,5)
        thrash: 'NO ROUTE around plain-NPC blocker'). The leader is never a stand tile; skip it."""
        try:
            grid = tv.Grid(self.b)
        except Exception:
            grid = None
        for nb in ((front[0], front[1] + 1), (front[0], front[1] - 1),
                   (front[0] - 1, front[1]), (front[0] + 1, front[1])):
            if avoid is not None and tuple(nb) == tuple(avoid):
                continue                                       # never nudge onto the leader's own tile
            if grid is not None and not grid.walkable(*nb):
                continue
            self._gym_move(nb, label="los-nudge")
            if st.in_battle(self.b):
                return True
            self._gym_move(front, label="los-reenter")
            for _ in range(12):
                if st.in_battle(self.b):
                    return True
                self.b.run_frame()
            if st.in_battle(self.b):
                return True
        return False

    def _clear_gym_trainers(self, leader_front, max_rounds=14):
        """Beat EVERY junior trainer (re-scanning each round, since far ones are proximity-loaded
        and one may wander), THEN return so the caller engages the gated leader. General gym
        mechanic - reusable for Surge/Erika/etc. Returns 'pp_famine' when a battle went stuck with
        the party PP-dry (the erika_run2 wall) — the caller heals and re-takes the gym (beaten
        juniors stay beaten in-game); None otherwise."""
        beaten = set()                                         # object indices CONFIRMED fought
        fails = {}                                             # idx -> failed-engage rounds (pool class)
        PER_TRAINER_TRIES = 4
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
                    _jr = self.battle_runner()
                    log(f"   GYM: junior trainer -> {_jr}")
                    self._drain_overworld(label="trainer")
                    if _jr == "stuck" and not self._party_damaging_pp():
                        log("   GYM: gauntlet PP FAMINE (stuck battle + no damaging PP anywhere) "
                            "-> needs_heal (juniors stay beaten; we come back full)")
                        return "pp_famine"
                    beaten.add(idx)                            # a REAL battle happened -> truly done
                else:
                    # No battle this round. A pool-gym swimmer WANDERS and is engageable only from its
                    # one land tile when it turns the right way (the Misty stuck: the old code FALSELY
                    # marked this 'beaten' and walked into a still-GATED leader -> "face other TRAINERS"
                    # -> stuck). RETRY across rounds; only defer (LOUD) after real effort, never spin.
                    fails[idx] = fails.get(idx, 0) + 1
                    if fails[idx] >= PER_TRAINER_TRIES:
                        log(f"   !! GYM: junior obj{idx} at {T} un-engageable after {fails[idx]} "
                            f"tries (wandering/water-locked) - deferring LOUD so the pass proceeds")
                        beaten.add(idx)
                    else:
                        log(f"   GYM: junior obj{idx} didn't engage (try {fails[idx]}/"
                            f"{PER_TRAINER_TRIES}) - re-scanning, will retry")
                continue
            # none loaded -> advance toward the leader to load/trigger far trainers (LoS may fire)
            self._gym_move(leader_front, label="leader-advance")
            if st.in_battle(self.b):
                _lr = self.battle_runner()
                log(f"   GYM: en-route LoS trainer -> {_lr}")
                self._drain_overworld(label="trainer")
                if _lr == "stuck" and not self._party_damaging_pp():
                    log("   GYM: gauntlet PP FAMINE (stuck battle + no damaging PP anywhere) "
                        "-> needs_heal (juniors stay beaten; we come back full)")
                    return "pp_famine"
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

    def _bump_gym_prep(self, gym_name, step=2):
        """Fix B: on a gym LOSS, escalate the prep demand so the retry doesn't throw the SAME losing team
        back in — prep_for_gym reads this as extra levels to grind + a bigger team to field. Reset when the
        badge lands (the gym is done)."""
        self._gym_prep_bump = getattr(self, "_gym_prep_bump", {})
        self._gym_prep_bump[gym_name] = self._gym_prep_bump.get(gym_name, 0) + step
        log(f"   GYM-PREP: '{gym_name}' loss -> prep demand escalated (+{step} → bump "
            f"{self._gym_prep_bump[gym_name]}); the retry grinds/teams up higher")

    def prep_for_gym(self, gym):
        """Fix B (2026-07-09) — ENFORCED pre-gym readiness, wired through the StrategicPlanner KB. Before a
        gym she must field a real TEAM (not a solo carry), have a TYPE ANSWER to the leader's ace, and be at
        a sane LEVEL — else she solos underleveled and loses the attrition (the Brock loss; the Gary-wall
        root). Reads planner.gym_readiness(gym, party); if not ready, CATCHES on nearby grass (bounded) and
        GRINDS to a KB-derived level, narrated in her voice. Best-effort + bounded — beat_gym proceeds
        regardless (blackout-recovery + the on-loss bump escalate the demand each retry). Suppressed on a
        goal-pinned watch (go straight at the gym). Gameplay logic only; no core-Kira touch."""
        self._gym_prep_bump = getattr(self, "_gym_prep_bump", {})
        if not GYM_PREP_ENABLED or getattr(self, "planner", None) is None:
            return "off"
        if os.getenv("POKEMON_WATCH_GOAL"):                 # focused watch → don't wander off to prep
            return "pinned"
        try:
            party = (self.read_live_state() or {}).get("party") or []
        except Exception as e:
            log(f"   GYM-PREP: state read failed ({e}) — skipping (LOUD)"); return "no_state"
        bump = self._gym_prep_bump.get(gym.name, 0)
        r = self.planner.gym_readiness(gym.name, party, party_target=GYM_PARTY_TARGET, loss_bump=bump)
        if not r:
            return "no_kb"
        if r["ready"]:
            log(f"   GYM-PREP [{gym.name}]: READY — party {r['party_size']}, top L{r['top_level']} "
                f">= L{r['level_target']}, type answer ✓")
            return "ready"
        log(f"   GYM-PREP [{gym.name}]: NOT ready (loss-bump {bump}) — party {r['party_size']}/"
            f"{r['target_size']}, topL {r['top_level']}/{r['level_target']}, type_answer={r['has_type_answer']}, "
            f"want={r['want_types']} — prepping BEFORE the gym")
        # her voice: ONE plan beat (soul-safe raw fact; the oracle colours it) — the endearing "did my homework"
        if not r["has_type_answer"] and r["want_types"]:
            self.on_event(f"{gym.name}'s a {r['ace']} wall — I really want a {' or '.join(r['want_types'][:2])} "
                          f"type before I knock. Let me go catch one.", kind="gym", tier=2)
        elif r["thin"]:
            self.on_event(f"I'm not walking into {gym.name} with just my starter — let me round the team out "
                          f"first.", kind="gym", tier=2)
        elif r["underleveled"]:
            self.on_event(f"{gym.name}'s ace is about level {r['ace_level']} and I'm a touch thin — quick grind, "
                          f"then we knock.", kind="gym", tier=2)
        # 1) CATCH — a thin bench and/or no type answer, while she has balls + grass is reachable
        tries = 0
        while (r["thin"] or not r["has_type_answer"]) and tries < GYM_PREP_CATCH_TRIES and self._ball_count() > 0:
            cr = self.catch_one()
            tries += 1
            log(f"   GYM-PREP [{gym.name}]: catch attempt {tries}/{GYM_PREP_CATCH_TRIES} -> {cr}")
            if cr in ("no_grass", "no_balls"):
                break                                       # nothing catchable here / out of balls -> grind
            try:
                party = (self.read_live_state() or {}).get("party") or []
            except Exception:
                break
            r = self.planner.gym_readiness(gym.name, party, party_target=GYM_PARTY_TARGET, loss_bump=bump)
            if r["ready"] or (not r["thin"] and r["has_type_answer"]):
                break
        # 1.5) COVERAGE-TEACH — a type wall the CATCH couldn't answer (no reachable counter / out of balls):
        # if the ace's whole offense is resisted by this gym's typing, teach it a coverage move from a bag
        # TM/HM so it can actually land hits. Fixes the Erika grass wall (all-Grass Venusaur, x0.25). Bounded,
        # best-effort; a crash here never blocks the gym (LOUD). Skipped when the team already has an answer.
        if GYM_COVERAGE_TEACH and not r["has_type_answer"]:
            try:
                _cvr = self._teach_gym_coverage(gym, self.planner.threats.get(gym.name) or {})
                log(f"   GYM-PREP [{gym.name}]: coverage-teach dispatch -> {_cvr}")
            except Exception as e:
                log(f"   GYM-PREP [{gym.name}]: coverage-teach dispatch crashed ({e}) (LOUD)")
        # 2) GRIND — bring the team up to the KB-derived level target (ace level + margin, escalated on loss)
        if r["underleveled"]:
            log(f"   GYM-PREP [{gym.name}]: grinding to L{r['level_target']} (ace ~L{r['ace_level']})")
            try:
                gr = self.grind(r["level_target"])
                log(f"   GYM-PREP [{gym.name}]: grind -> {gr}")
            except Exception as e:
                log(f"   GYM-PREP [{gym.name}]: grind skipped ({e}) (LOUD)")
        return "prepped"

    def _stock_potions_for_gym(self, gym):
        """PRE-GYM POTION-STALL STOCK-UP (night-shift 1, badge-5 Koga). Some leaders beat a solo-carry by
        ATTRITION, not type: Koga's 4-mon poison gauntlet out-damages a lone L53 Venusaur with no heals
        (verified 5+ losses w/o potions, WINS with them). A real player buys Super Potions at the gym-city
        Mart and heals through it. For a POTION_STALL_GYMS gym, if she's short of the target, buy the
        shortfall (strongest potion on the shelf) from the current city's Mart. Best-effort + bounded; the
        buy_at_mart bag-delta guards every purchase, and a no-mart / too-poor case just enters as-is (LOUD).
        Returns a short status string. General resource/economy bedrock #6; game-facts in POTION_STALL_GYMS."""
        target = POTION_STALL_GYMS.get(gym.name)
        if not target:
            return "not_flagged"
        city = tv.map_id(self.b)
        door = CITY_MART_DOORS.get(city)
        if door is None:
            log(f"   POTION-STALL [{gym.name}]: not standing in a Mart city ({city}) — can't stock here (LOUD)")
            return "no_mart"
        have = sum(self.bag_count(i) for i in (ITEM_POTION, 22, 21, 20, 19))
        if have >= target:
            log(f"   POTION-STALL [{gym.name}]: already carrying {have} potion(s) (>= {target}) — good to go")
            return "already_stocked"
        pot = self._best_potion_for_sale() or 22
        if self.money() < SHOP_MONEY_FLOOR + 700:
            log(f"   POTION-STALL [{gym.name}]: too poor ({self.money()}) to stock potions — entering as-is (LOUD)")
            return "too_poor"
        need = target - have
        self.on_event(f"{gym.name}'s a war of attrition — I'm not tanking that gauntlet on one Pokémon with "
                      f"an empty bag. Quick Mart run for potions first.", kind="gym", tier=2)
        log(f"   POTION-STALL [{gym.name}]: have {have}/{target} — buying {need}x "
            f"{ITEM_NAMES.get(pot, pot)} at {city} Mart before the fight")
        bought = self.buy_at_mart(door, [(pot, need)])
        got = sum(self.bag_count(i) for i in (ITEM_POTION, 22, 21, 20, 19))
        log(f"   POTION-STALL [{gym.name}]: bought {bought} — now carrying {got} potion(s)")
        return "bought" if bought else "buy_failed"

    def stock_hyper_potions(self, hyper_target=SILPH_HYPER_TARGET, revive_target=SILPH_REVIVE_TARGET):
        """Pre-Silph SILPH KIT stock-up (night-shift 4). Gary's Silph gauntlet ends on a Charizard
        (Fire/Flying) that 2x-burns solo Venusaur while QUAD-resisting her Grass (Razor Leaf x0.25) —
        her only neutral hit is weak Cut, so the sole winning line is to OUT-HEAL the Fire with HYPER
        Potions (200 HP; Super Potions at 50 merely tread water -> faint at Charizard ~29/119) AND to
        carry REVIVES so a single crit/paralysis-turn faint isn't game-over (revive Venusaur behind a
        fodder body, keep healing — the proven-winning kit: NS4 injected 20 Hyper + 5 Revive -> 12 Hyper
        + 2 Revive used -> GARY WON -> Saffron freed; the autobuy w/ Hyper-only-no-Revive LOST). Buys the
        shortfall of BOTH at the Saffron Mart while standing in Saffron before the strike. Counts each id
        SPECIFICALLY (the 30 Super Potions she already carries must NOT read as 'stocked'). Best-effort +
        bounded; buy_at_mart's per-purchase bag-delta guards every unit and its money floor never drains
        the wallet. Resource/economy bedrock #6; game-facts isolated to SAFFRON_MART_DOOR / MART_STOCK /
        SILPH_HYPER_TARGET / SILPH_REVIVE_TARGET (rule 14). Returns a short status string."""
        city = tv.map_id(self.b)
        door = CITY_MART_DOORS.get(city)
        if door is None:
            log(f"   HYPER-STALL: not standing in a Mart city ({city}) — can't stock here (LOUD)")
            return "no_mart"
        have_h, have_r = self.bag_count(21), self.bag_count(24)
        if have_h >= hyper_target and have_r >= revive_target:
            log(f"   HYPER-STALL: already carrying {have_h} Hyper + {have_r} Revive (>= "
                f"{hyper_target}/{revive_target}) — good to go")
            return "already_stocked"
        if self.money() < SHOP_MONEY_FLOOR + 1200:
            log(f"   HYPER-STALL: too poor ({self.money()}) for the Silph kit — entering as-is (LOUD)")
            return "too_poor"
        buy = []
        if have_h < hyper_target:
            buy.append((21, hyper_target - have_h))          # Hyper Potion FIRST (the primary win-lever)
        if have_r < revive_target:
            buy.append((24, revive_target - have_r))          # Revive second (the faint-insurance)
        self.on_event("Gary's in there and his Charizard torches my Venusaur — Super Potions won't keep "
                      "up. I need Hyper Potions and a few Revives before I go in.", kind="gym", tier=2)
        log(f"   HYPER-STALL: have {have_h}/{hyper_target} Hyper, {have_r}/{revive_target} Revive — "
            f"buying {buy} at {city} Mart before Silph")
        bought = self.buy_at_mart(door, buy)
        got_h, got_r = self.bag_count(21), self.bag_count(24)
        log(f"   HYPER-STALL: bought {bought} — now carrying {got_h} Hyper + {got_r} Revive")
        return "bought" if (bought.get(21) or bought.get(24)) else "buy_failed"

    def _teach_gym_coverage(self, gym, rec):
        """When the ACE's whole offense is RESISTED by the gym's typing, teach it a neutral-or-better
        damaging move from a bag TM/HM (ROM learn-compat). THE ERIKA WALL (2026-07-10): L43 Venusaur's
        moves are all Grass (x0.25 vs grass/poison) because auto-learn stripped her only neutral move —
        so a huge level lead still can't out-damage, she PP-famines to status, and blacks out. A real
        player equips a coverage move. Scores the bag against the gym's DEFENDING types (rec['types'])
        via the type chart, prefers super-effective then power, teaches the ace, forgetting its weakest
        damaging move (keeps status + best STAB). Best-effort + bounded; the caller never blocks on it.
        General bedrock #8 (TM/HM strategy); no core-Kira touch. Returns a short status string."""
        import hm_teach as ht
        import pokemon_policy as pol
        deftypes = [t for t in (rec.get("types") or []) if t]
        if not deftypes:
            return "no_types"
        pc = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if not pc:
            return "no_party"
        # the ACE = highest-level mon (its CURRENT slot; _restore_ace moves it to slot 0 for the fight)
        ace_lv, ace_slot = max((self.b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54), s)
                               for s in range(pc))
        ace_sp = st.read_party_species(self.b, ace_slot)
        moves = st.read_party_moves(self.b, ace_slot)
        # COVERAGE DEPTH (2026-07-10, the Erika PP-FAMINE wall — the true badge-4 root): count the ace's
        # NEUTRAL-OR-BETTER DAMAGING moves vs this gym, not just "does one exist". The original guard
        # skipped the instant ONE existed — but a mono-type gym's JUNIOR GAUNTLET famines a lone neutral
        # move: Venusaur's only neutral move here is Tackle (35 PP), which is the BEST pick vs every
        # grass/poison junior, so she spams it through the gauntlet, it drains to 0 PP before the leader,
        # and choose_move then falls to RESISTED Razor Leaf (x0.25) and she blacks out (verified: attempt-1
        # loss on Razor Leaf x0.25 vs victreebel; the retry only wins because whiteout REFILLS PP and the
        # beaten juniors stay beaten). A real player carries TWO ways to hit the wall. So skip only when she
        # already has >=2 neutral damaging moves (genuine PP depth); with 0 or 1, teach a bag coverage move
        # (Cut → Tackle+Cut = 65 PP of neutral coverage) so she reaches the leader with PP to spare.
        # Self-limiting: after the teach she has 2 neutral moves, so a re-check returns have_coverage.
        best_have = 0.0
        neutral_dmg = 0
        for m in moves:
            if not m:
                continue
            mt, mp = st.move_info(self.b, m)
            if (mp or 0) <= 0:
                continue
            eff_m = pol.effectiveness(mt or "normal", deftypes)
            best_have = max(best_have, eff_m)
            if eff_m >= 1.0:
                neutral_dmg += 1
        if neutral_dmg >= 2:
            return "have_coverage"
        # find the best bag coverage move the ace CAN learn and that improves on its resisted offense.
        cands = []
        for item, (hm_key, tm_no, move_id, mtype, power) in _COVERAGE_MOVES.items():
            if ht.tm_case_row(self.b, item) is None:         # not in the bag
                continue
            if move_id in moves:                             # already knows it
                continue
            learnable = (ht.hm_compatible(self.b, hm_key, ace_sp) if hm_key
                         else ht.tm_compatible(self.b, tm_no, ace_sp))
            if not learnable:
                continue
            if tm_no is not None:                            # ROM-validate the TM move_id (wrong id -> skip)
                rt, rp = st.move_info(self.b, move_id)
                if (rt or "").lower() != mtype:
                    log(f"   GYM-PREP [{gym.name}]: TM{tm_no:02d} move#{move_id} ROM type "
                        f"{rt!r}!={mtype!r} — skipping (LOUD)")
                    continue
            eff = pol.effectiveness(mtype, deftypes)
            if eff < 1.0:                                    # still resisted — no improvement
                continue
            cands.append((eff, power, bool(hm_key), item, hm_key, tm_no, move_id, mtype))
        depth_only = best_have >= 1.0            # she HAS a neutral move — this teach is for PP depth
        if not cands:
            log(f"   GYM-PREP [{gym.name}]: ace {'thin on neutral coverage' if depth_only else 'all-resisted'} "
                f"(best x{best_have:g}, {neutral_dmg} neutral dmg move(s)) but NO learnable coverage TM/HM "
                f"in bag — walking in as-is (LOUD)")
            return "no_candidate"
        # best: highest eff, then power, then prefer an HM (proven single-turn actuation)
        cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        eff, power, _is_hm, item, hm_key, tm_no, move_id, mtype = cands[0]
        # forget the ace's WEAKEST damaging move vs these types (never a status move — keep Sleep Powder etc.)
        scored = []
        for i, m in enumerate(moves):
            if not m:
                continue
            mt, mp = st.move_info(self.b, m)
            if (mp or 0) <= 0:
                continue
            scored.append((mp * pol.effectiveness(mt or "normal", deftypes), i))
        forget_idx = min(scored)[1] if scored else 0
        mon = st.SPECIES_NAME.get(ace_sp, f"slot {ace_slot}")
        label = hm_key.title() if hm_key else f"TM{tm_no:02d}"
        eff_word = "super-effective" if eff >= 2 else "neutral"
        if depth_only:
            self.on_event(f"one move won't last {gym.name}'s whole gym — teaching {mon} {label} so I "
                          f"don't run dry on the leader.", kind="gym", tier=2)
        else:
            self.on_event(f"{gym.name}'s type wall shrugs off everything I've got — teaching {mon} a "
                          f"{mtype} move so I actually land hits.", kind="gym", tier=2)
        log(f"   GYM-PREP [{gym.name}]: COVERAGE-TEACH {label} ({mtype}, x{eff:g} {eff_word}) -> {mon} "
            f"slot {ace_slot}, forgetting move-idx {forget_idx}")
        tf = ht.TeachFlow(self, log=log, on_event=self.on_event)
        if hm_key:
            res = tf.teach(hm_key, ace_slot, forget_idx)
        else:
            res = tf.teach("_tm", ace_slot, forget_idx, item_override=item, move_override=move_id)
        log(f"   GYM-PREP [{gym.name}]: coverage-teach {label} -> {res}")
        if res == "taught":
            self.on_event(f"{mon} knows {label} now — let's see the flower lady laugh that off.",
                          kind="gym", tier=2)
        return res

    def beat_gym(self, name):
        """GENERAL gym handler (gyms gate the leader behind their junior trainers): reserve move
        slots if this gym has a level-up double-learn, enter, BEAT EVERY JUNIOR TRAINER, then engage
        the gated leader, beat them with the current team, drive the badge/TM award to a clean close
        at a watchable pace, and confirm the badge flag. Data-driven via GYMS - one row per leader."""
        gym = GYMS.get(name)
        if gym is None:
            log(f"   !! GYM: no spec for '{name}'"); return "stuck"
        # Fix B: ENFORCED pre-gym readiness (catch a team / type answer / level) BEFORE entering — she
        # must never solo a gym underleveled. Best-effort; a crash here never blocks the gym (LOUD).
        try:
            self.prep_for_gym(gym)
        except Exception as e:
            log(f"   !! GYM-PREP crashed ({e}) — walking into {name} unprepped (LOUD)")
        # POTION-STALL STOCK-UP (night-shift 1, the badge-5 Koga attrition wall): for gyms flagged as a
        # solo-carry attrition wall, buy a stall stock of Super Potions at the gym-city Mart BEFORE
        # entering, so the in-battle item instinct can heal her ace through the gauntlet (the proven
        # potion-stall win). Runs here (not inside prep_for_gym) so it fires even on a goal-pinned watch
        # — prep_for_gym returns "pinned" and skips its body. Best-effort + bounded; a crash never blocks
        # the gym (LOUD). Resource/economy bedrock #6; game-facts isolated to POTION_STALL_GYMS.
        try:
            self._stock_potions_for_gym(gym)
        except Exception as e:
            log(f"   !! GYM potion-stall stock-up crashed ({e}) — entering {name} as-is (LOUD)")
        # HEAL-BEFORE-THE-GYM GATE (2026-07-07, erika_run2): she walked into Erika's gauntlet with
        # two fainted mons and a PP-dry lead and status-moved her way to a 12-battle futility wall.
        # A human ALWAYS taps the Center first — gym cities always have one and heal_at_center
        # returns to the pre-heal spot. ONE attempt, never a loop: if the heal fails we still take
        # the gym (the in-battle famine switch is the backstop).
        if not self._party_gym_ready():
            log("   GYM: party not battle-ready (faints / low ace HP / PP famine) -> Center first, "
                "then the gym")
            self.on_event("hold on — nobody walks into a gym half-dead. Center first, then we do "
                          "this properly.", kind="gym", tier=2)
            hr = self.heal_nearest()
            log(f"   GYM: pre-gym heal -> {hr}")
        # ACE-LEAD THE GYM (2026-07-10, the badge-4 Erika wall): the strategic underlevel-grind reorders a
        # weak bench mon to slot 0 for participation XP, and when she diverts straight from the grass into
        # a gym that order PERSISTS — she then fights the LEADER with an L8 lead, PP-famines on status
        # moves, and blacks out while her L43 ace sits at slot 3 (the Erika STALL: led pidgey/rattata vs
        # Victreebel, lost, gym marked a spatial wall "gated until stronger" -> post-loss ping-pong stall).
        # A real player leads a gym with their strongest. Restore the ace to slot 0 before the fight
        # (save-safe slot swap; the reserve/move-room below then operate on the ace that actually fights).
        # No-op when the ace already leads (normal play), so it never disturbs a deliberately-ordered team.
        self._restore_ace()
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
        # PAD ROUTER (the Saffron-gym class, recon_sabrina-proven; ported 2026-07-08): a
        # warp-partitioned interior (same-map teleport pads) reads no_route to EVERYTHING on
        # a walk BFS — the old handler travel-wedged x4 and A-mashed the leader from 11 tiles
        # away (the SILPH pre-grade WARN). Arm the per-gym router; _gym_move rides the pads.
        self._gym_pads = None
        try:
            import pad_nav
            _pads = pad_nav.same_map_pads(self.b)
            if _pads:
                self._gym_pads = pad_nav.PadNav(self.b, self, log=log)
                log(f"   GYM: teleport-pad interior ({len(_pads)} pads) - pad router ARMED")
        except Exception as e:
            log(f"   !! GYM: pad-router arm failed ({e}) - walking blind")
        # 0) ENVIRONMENT PUZZLE (general seam; instance #1 = Vermilion's trash cans, 2026-07-06):
        # some gyms gate the leader behind a puzzle, not just juniors. Solve it HONESTLY (narrated
        # hunt; RAM verify-only) before the trainer sweep — beams block the leader approach anyway.
        if name == "Lt. Surge":
            import env_puzzle
            from field_moves import read_flag as _rf
            if not _rf(self.b, env_puzzle.FLAG_BOTH_SWITCHES):
                pz = env_puzzle.TrashCanPuzzle(self, log=log)
                pr = pz.run()
                log(f"   GYM: trash-can puzzle -> {pr}")
                if pr == "stuck":
                    return "stuck"
        # 1) BEAT THE JUNIOR TRAINERS FIRST (the leader is gated until they're all down)
        if self._clear_gym_trainers(gym.leader_front) == "pp_famine":
            return "needs_heal"                                # heal + re-take (juniors stay beaten)
        # BLACKOUT during the juniors -> she whited out + respawned in the city PC (the map left the
        # gym interior). _clear_gym_trainers can't tell (no trainers load in the PC), so detect it here
        # and propagate -> the segment's blackout-recovery respawns + RE-RUNS the gym (beaten juniors
        # stay beaten in-game, so only the one that won re-fights; a fresh full-HP, on-level run wins).
        if tv.map_id(self.b) != gym_map:
            log(f"   GYM: no longer in the gym ({tv.map_id(self.b)} != {gym_map}) - blackout during "
                f"the juniors; caller recovers + retries the gym")
            self._bump_gym_prep(name)                       # Fix B: the retry preps harder (grind/team up)
            return "battle_loss"
        # 2) engage the LEADER (now ungated): walk to the front tile, face them + A to initiate,
        # then DRIVE the pre-battle challenge speech INTO the battle (Brock's "So, you're here..."
        # is a multi-box speech that a few A-taps don't fully clear; the primitive advances it and
        # stops the instant the battle starts). Misty starts the battle near-immediately - same path.
        self.b.set_input_owner("agent")
        self._gym_move(gym.leader_front, label="leader")
        _lvl = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        _sp = st.SPECIES_NAME.get(st.read_party_species(self.b, 0), "?")
        log(f"   GYM: at {tv.coords(self.b)} (leader front {gym.leader_front}) - engaging {name} "
            f"[lead {_sp} Lv{_lvl}, party={self.b.rd8(ram.GPLAYER_PARTY_CNT)}]")
        # LoS RE-TRIGGER FIRST (the Misty stuck, 2026-07-09): a leader whose sight faces the front
        # tile only fires on a FRESH STEP into it. We usually arrive at leader_front via a zero-step
        # travel (already standing there), so her sight never re-checks and an A-press opens a
        # GREETING box, not the battle -> award drain finds no badge -> stuck. Nudge off-and-back so
        # the fight initiates. No-op / already-won cases fall straight through. Skip if she's already
        # been beaten (badge set en route via a travel fight-through) — then it's a pure award drain.
        if not st.in_battle(self.b) and not self.has_badge(gym.badge_flag):
            # Fix A (2026-07-09): pass the LEADER's own tile (front stepped in leader_dir) so the nudge
            # NEVER steps onto it — else travel treats the standing leader as a plain-NPC blocker and
            # burns its budget routing around him (the Brock (6,5) 'NO ROUTE around plain-NPC' thrash).
            _ld = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.get(gym.leader_dir, (0, -1))
            _leader_tile = (gym.leader_front[0] + _ld[0], gym.leader_front[1] + _ld[1])
            self._los_retrigger(gym.leader_front, avoid=_leader_tile)
        for _ in range(6):
            if st.in_battle(self.b) or dd_box_open(self.b):
                break
            self.b.press(gym.leader_dir, 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        if not st.in_battle(self.b) and not self.has_badge(gym.badge_flag):    # speech -> battle
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
                self._bump_gym_prep(name)                   # Fix B: the retry preps harder (grind/team up)
                return "battle_loss"
            # A STUCK leader fight with the battle STILL LIVE must never reach the award drain —
            # erika_run2's award drain A-mashed the live battle's forced-switch party menu for 300
            # steps ("Do what with this POKéMON?"). PP famine -> heal + re-take; anything else
            # surfaces honestly (the roam loop's encounter hand-off re-enters the live fight).
            if res == "stuck" and st.in_battle(self.b):
                if not self._party_damaging_pp():
                    log(f"   GYM: {name} fight STUCK on PP famine -> needs_heal (heal, come back, "
                        f"re-take — juniors stay beaten)")
                    return "needs_heal"
                log(f"   !! GYM: {name} fight stuck with the battle still live (not PP) - "
                    f"surfacing stuck, never draining a live battle")
                return "stuck"
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
            getattr(self, "_gym_prep_bump", {}).pop(name, None)   # Fix B: gym done -> clear the loss-bump
            self.on_event(f"I beat {name} - that's the badge!")
            if tv.map_id(self.b)[0] != 3:                      # still in the gym interior -> leave to
                pn = getattr(self, "_gym_pads", None)          # pad maze: ride back beside a door first
                if pn is not None and tuple(tv.map_id(self.b)) == pn.map:
                    _here = tuple(tv.map_id(self.b))
                    _adj = {(w[0][0] + dx, w[0][1] + dy)
                            for w in tv.read_warps(self.b) if tuple(w[1]) != _here
                            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))}
                    if _adj:
                        pn.goto_region(_adj, label="pads->exit")
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
        # P-2 (couch fix-pass 1): MOM IS A MUST-STOP on journey day one — a real goodbye
        # beat, not scenery. The F-6 salience only rides free-roam ticks; the OPENING walks
        # this floor exactly once, so the stop is anchored HERE. talk_npc owns the walking/
        # facing; best-effort — an unreachable mom logs LOUD and the journey still starts
        # (never wedge the debut on a hug).
        try:
            if self.talk_npc() == "talked":
                self.world.mark_met("mom")
                self.on_event("that's Mom — one real goodbye before the whole world happens. "
                              "okay. deep breath. love you, I'll be back with a full Pokédex.",
                              kind="social", tier=2)
            else:
                log("   !! OPENING: couldn't reach Mom for the day-one goodbye (LOUD) — "
                    "journey starts anyway")
        except Exception as _me:
            log(f"   !! OPENING: mom goodbye crashed ({_me!r}) — continuing")
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
            # Oak's intercept is a SCRIPTED HIJACK (finding #2): it can fire mid-leg and land her
            # in the lab (4,3) while this nav call is still walking "north" — abort_when + short
            # legs make the nav yield in seconds instead of wedging against the cutscene for minutes.
            out = self.advance_north(PALLET if tv.map_id(self.b) != PALLET else ROUTE1, max_legs=2,
                                     abort_when=lambda: tv.map_id(self.b) == (4, 3), leg_seconds=25)
            if out == "aborted":
                continue                       # loop top sees map==(4,3) and breaks to the pick
            if out in ("stuck", "timeout"):
                self.b.press("UP", 8, 8, self.render, owner="agent")
                for _ in range(8):
                    self.b.run_frame()
        if tv.map_id(self.b) != (4, 3):
            log("   !! OPENING: Oak intercept didn't reach the lab"); return tv.map_id(self.b)
        log("   OPENING: in Oak's lab - picking her starter")
        self._starter_choice = starter_choice
        # VOICE THE CHOICE (rehearsal finding #1, the endearing half): a neutral moment-fact —
        # WHICH partner she's committed to — fired through the reaction seam so she voices the WHY
        # in her own live words while the hands walk to the ball. Tier 3: this is the run's first
        # big beat (the partner decision), never a silent acquisition.
        _snm = {0: "Bulbasaur", 1: "Charmander", 2: "Squirtle"}.get(starter_choice, "Bulbasaur")
        self.on_event(f"three Poké Balls on Oak's table — your mind's made up: {_snm} is your partner",
                      kind="event", tier=3)
        # HER NICKNAME CALL (F-3 CEO principle: names/choices are hers via the judged system or they
        # don't happen — never a hardcoded default). The proven free-text naming ask (the teammate-
        # intro pattern); "NONE" or a dead oracle -> keep the species name, logged as her call.
        self._nickname = None
        try:
            _nm = self._soul_choose("name", {}, {"place":
                f"{_snm} is about to become your partner — the one beside you all the way. If you "
                f"want to give it a personal name, answer with JUST the name (12 letters max). If "
                f"you'd rather keep calling it {_snm}, answer exactly: NONE"})
            if _nm:
                _nm = _nm.strip().split("\n")[0].strip()[:12]
                self._nickname = None if (not _nm or _nm.upper() == "NONE") else _nm
        except Exception as _nne:
            log(f"   STARTER: nickname ask skipped ({_nne})")
        log(f"   STARTER: nickname -> " + (f"{self._nickname!r} (her choice)" if self._nickname
                                           else "keep the species name (her choice)"))
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
    # Ball tiles (face UP from below). TABLE ORDER IS THE RBY LAYOUT — left→right:
    # BULBASAUR, SQUIRTLE, CHARMANDER (empirical, take-6 2026-07-08: standing (9,5) grabbed
    # SQUIRTLE #7 when the old map claimed (9,4)=charmander; never caught before because no
    # run had ever picked a non-zero ball). Index stays the species intent (0=bulba 1=char
    # 2=squirtle); the TILE maps to where that species actually sits.
    STARTER_BALLS = {0: (8, 4), 1: (10, 4), 2: (9, 4)}
    STARTER_SPECIES = {0: 1, 1: 4, 2: 7}                       # Bulbasaur / Charmander / Squirtle

    def choose_starter(self):
        """DECISION SEAM - which starter Kira takes. Capability-not-script: the HANDS (pick_starter)
        work for ANY of the three; WHICH she picks is HER choice, set on self._starter_choice (Batch
        2 routes this through her soul). NOT hardcoded into the pick mechanics. Default 0 (Bulbasaur)
        is only a fallback when nothing set the choice - the point is the seam exists + is honoured."""
        c = getattr(self, "_starter_choice", None)
        return c if c in (0, 1, 2) else 0

    def _choose_starter_soul(self):
        """SOUL-DEBT #3 AT MINUTE ONE (quiet-window rehearsal finding #1): her PARTNER for the whole
        run is HER pick through the generic soul oracle — never authored, never silently defaulted.
        Facts offered as detail (types/tradeoffs any player knows from the box art blurbs); her
        taste decides. Headless / no bot / no pick -> the old default 0 (Bulbasaur), LOUD."""
        opts = {
            "bulbasaur": "the Grass/Poison one — calm and steady; the early gyms (rock, water) come easy",
            "charmander": "the Fire one — the hard-mode start, but it becomes a dragon (Charizard)",
            "squirtle": "the Water one — a tough all-rounder with a great look",
        }
        # P-5 SELF-CANON: her cross-session identity rides the ctx as HER OWN memory —
        # she still chooses; canon is the taste she chooses with (never a forced pick).
        try:
            from pokemon_soul import SELF_CANON as _CANON
            _canon_line = _CANON.get("starter", "")
        except Exception:
            _canon_line = ""
        pick = self._soul_choose("starter", opts, {
            "moment": ("Professor Oak's lab, three Poké Balls on the table. This is your PARTNER "
                       "for the entire journey — the one beside you from this bedroom morning all "
                       "the way to the credits. Which one calls to you?"
                       + ((" " + _canon_line) if _canon_line else "")),
        })
        idx = {"bulbasaur": 0, "charmander": 1, "squirtle": 2}.get((pick or "").strip().lower())
        if idx is None:
            log(f"   !! STARTER: no soul pick ({pick!r}) -> default Bulbasaur (LOUD fallback)")
            return 0
        log(f"   STARTER: her soul chose {pick.upper()} -> ball {idx}")
        return idx

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
                # RACE FIX (F-3, the AAAAAAAAA class): 'nickname' shows while the prompt is still
                # STREAMING — an answer pressed now just speeds text, the REAL Yes/No lands after
                # we return, and the next stage's A-drain answers YES + types A's on the keyboard.
                # (Take 1 logged "declined (her choice)" while the screen showed AAAAAAAAA — the
                # log told the truth about its press and missed the outcome.) Settle until the
                # box text is STABLE (the cursor is really up), THEN answer, THEN VERIFY.
                _prev = low
                for _ in range(40):
                    for _ in range(10):
                        self.b.run_frame()
                    _cur = self._read_overworld_text().lower()
                    if _cur == _prev:
                        break
                    _prev = _cur
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
                    # decline + VERIFY: the prompt text must actually be GONE before we trust it.
                    for _try in range(4):
                        self.b.press("B", 8, 10, self.render, owner="agent")
                        for _ in range(24):
                            self.b.run_frame()
                        if "nickname" not in self._read_overworld_text().lower():
                            break
                        log(f"   STARTER: decline didn't land (try {_try + 1}) — pressing B again")
                    log("   STARTER: declined a nickname (her choice) — verified prompt gone")
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
        # 1+2) drain Oak's intro -> stand below the ball. DRAIN-AND-RETRY (take-3 finding): Oak's
        # intro streams in WAVES with lulls (Gary's interjections) — one DialogueDriver pass can end
        # at a lull while the SCRIPT still owns input, so the walk never lands (the fast-arrival
        # regression the intercept-abort exposed; take 1 only worked because 2 min of lab wandering
        # accidentally drained the waves). Loop: drain -> try to stand -> if the tile didn't take,
        # the cutscene still has the wheel -> drain the next wave and retry.
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        _stood = False
        for _wave in range(6):
            DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(label="oak-intro")
            # BFS FIRST (take-5 lesson): the greedy stepper WANDERS in the crowded lab (Oak/Gary/
            # table) — it drifted her to (9,2) behind the table reading a bookshelf. The Traveler's
            # NPC-aware BFS paths to the stand tile or LOGS the blocking NPC; greedy is the fallback
            # for the last nudge only.
            try:
                self.trav.travel(target_map=None, arrive_coord=(bx, by + 1),
                                 max_steps=60, max_seconds=15)
            except Exception as _te:
                log(f"   STARTER: BFS leg error ({_te}) — greedy fallback")
            if tv.coords(self.b) == (bx, by + 1) or self._step_to((bx, by + 1), steps=16):
                _stood = True
                break
            log(f"   STARTER: stand tile {(bx, by + 1)} not reached (wave {_wave + 1}; "
                f"at {tv.coords(self.b)}) — draining the next wave and retrying")
        if not _stood:
            log("   !! STARTER: never reached the ball's stand tile (LOUD)")
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
            # GRAB-VERIFY (take-6 lesson): the obtained species must MATCH her chosen index — a
            # wrong-ball grab (the B-S-C table-order class) must scream, never ship silently.
            _want_sp = self.STARTER_SPECIES.get(idx)
            if _want_sp and sp != _want_sp:
                log(f"   !! STARTER MISMATCH (LOUD): she chose {st.SPECIES_NAME.get(_want_sp)} "
                    f"(ball {idx}) but obtained {st.SPECIES_NAME.get(sp, '?')} — tile map wrong?")
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

    def _species_on_map(self, species, mid):
        """True if `species` appears in the encounters KB for the CURRENT map's area (any catch method).
        Resolves the map -> its human place name (Route 24, Diglett's Cave...) then reads the frlg_
        encounters areas table the TeamPlanner already loaded. Mode-side, pure lookup, fail-closed."""
        try:
            areas = (self.team_planner.encounters or {}).get("areas") or {}
            entry = areas.get(self._place_name(tuple(mid), default=""))
            if not entry:
                return False
            sp = (species or "").lower()
            for method, mons in entry.items():
                if str(method).startswith("_"):
                    continue
                if any((m.get("species") or "").lower() == sp for m in (mons or [])):
                    return True
        except Exception as e:
            log(f"   [roam] species-on-map lookup skipped: {e}")
        return False

    def _plan_keeper_target(self, state):
        """Part-C EXECUTOR: the forward-plan's DUE keeper species to SEEK, IF it can appear on the
        CURRENT map (else None — no point fleeing everything on a route the keeper doesn't live on).
        Reuses the standing TeamPlanner assessment; mode-side + plan-gated. Returns a species name/None."""
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return None
            act = self.team_planner.assess(state.get("party") or [], state.get("badge_count", 0),
                                           bag=state.get("bag"), dex=state.get("dex_caught"),
                                           post_game=bool(state.get("post_game")))
            if not act or act.get("kind") != "catch_keeper":
                return None
            sp = (act.get("species") or "").lower()
            if sp and self._species_on_map(sp, tv.map_id(self.b)):
                return sp
        except Exception as e:
            log(f"   [roam] plan keeper target skipped: {e}")
        return None

    def _keeper_due(self, state):
        """True if the forward team-plan wants a coverage keeper caught (assess -> catch_keeper), regardless
        of WHERE that keeper lives (unlike _plan_keeper_target, which is on-map-gated). Used to PRE-STOCK
        balls at the Mart before the hunt: _plan_keeper_target is None at the Mart (diglett doesn't live in
        Vermilion) so it can't drive the buy — this location-free check can. Mode-side + plan-gated."""
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return False
            act = self.team_planner.assess(state.get("party") or [], state.get("badge_count", 0),
                                           bag=state.get("bag"), dex=state.get("dex_caught"),
                                           post_game=bool(state.get("post_game")))
            return bool(act and act.get("kind") == "catch_keeper")
        except Exception as e:
            log(f"   [roam] keeper-due check skipped: {e}")
            return False

    def _plan_wants_prebuild(self, state):
        """Part-C EXECUTOR (rule 2 — make the brain LOAD-BEARING): True when the standing team brain says
        she should BUILD before pushing forward — a DUE keeper catch (or bench-develop) while the team is
        dangerously THIN (<=2 mons). The load-bearing case: forward-drive was marching a SOLO lead straight
        into the Nugget-Bridge gauntlet (misty_done look-ahead: solo Ivysaur -> Starmie -> blackout ->
        the road self-gates) even though the brain already knew 'catch a teammate first'. When this fires
        we keep catch/grind available against the forward-drive prune so she assembles a squad on the
        reachable grass FIRST, then crosses. Bounded: only <=2 mons, so it self-clears once the bench
        exists (party 3 -> forward-drive resumes). Mode-side, plan-gated, fail-closed."""
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return False
            pc = state.get("party_count")
            if pc is None:
                pc = len(state.get("party") or [])
            act = self.team_planner.assess(state.get("party") or [], state.get("badge_count", 0),
                                           bag=state.get("bag"), dex=state.get("dex_caught"),
                                           post_game=bool(state.get("post_game")))
            if not act:
                return False
            kind = act.get("kind")
            # THIN-TEAM pre-build (original): <=2 mons + a catch/bench-build DUE → build the squad before
            # the forward gauntlet (the misty_done solo-Ivysaur blackout this was born to stop).
            if pc <= 2 and kind in ("catch_keeper", "develop_bench"):
                return True
            # PAST-3 UN-GATE (2026-07-11, PASS 3 team-depth): the old hard `pc>2 → False` is WHY slots 4-6
            # were never pursued — once she had 3 mons forward-drive marched her past every planned keeper
            # forever (the root of "arrives at the E4 with ~2 usable mons"). Keep BUILDING toward a full six
            # while a planned keeper is DUE **and catchable on the CURRENT map** — i.e. she's standing on its
            # route (Abra on Route 24/25, Diglett's Cave, Growlithe on Route 7/8 all sit on the forward path),
            # so this grabs it IN PASSING rather than skipping it. Junk-catch-safe: _plan_keeper_target names
            # the SPECIFIC species and catch_one(target_species) flees every non-target. Bounded: on-map only
            # (no detour router yet — that's the TEAM_DEPTH_ROOT_FIX follow-up), party not full, one DUE
            # keeper at a time; self-clears the moment it's caught (assess stops returning it → march resumes).
            if pc < 6 and kind == "catch_keeper" and self._plan_keeper_target(state):
                return True
            return False
        except Exception as e:
            log(f"   [roam] plan-prebuild check skipped: {e}")
            return False

    def _place_to_map_index(self):
        """Reverse of _PLACE_NAMES (name -> ENTRANCE map_id). Multi-map areas (Diglett's Cave, caves)
        share one name across several ids — keep the LOWEST num (the entrance/first floor) so a route
        target lands at the mouth, not a deep interior. Cached (the table is static). Narration names
        already flow the other way; this is the routing-side reverse the keeper router needs."""
        idx = getattr(self, "_place2map_cache", None)
        if idx is None:
            idx = {}
            for mid, nm in self._PLACE_NAMES.items():
                if nm not in idx or mid[1] < idx[nm][1]:
                    idx[nm] = mid
            self._place2map_cache = idx
        return idx

    def _host_gateways(self):
        """STATIC-CONNECTION PRIORS (NS#40) — {host area name -> [overworld gateway map, ...]} from
        gamedata/frlg_connections.json. The learned graph only knows VISITED maps, so a cave/interior
        keeper host reached by a DOOR (Diglett's Cave) is never routable until she stands on it; this
        gives the router the static 'area X is entered from map Y' knowledge so it can ride to a
        reachable gateway and step through. Loaded once, cached; the swappable game-knowledge layer
        (rule 14) — keyed by the same place-name the encounters KB + _PLACE_NAMES use. Fail-open ({})."""
        idx = getattr(self, "_host_gw_cache", None)
        if idx is None:
            idx = {}
            try:
                import json
                path = os.path.join(_HERE, "gamedata", "frlg_connections.json")
                with open(path, encoding="utf-8") as f:
                    raw = (json.load(f) or {}).get("host_gateways") or {}
                for area, gws in raw.items():
                    idx[area] = [tuple(g) for g in (gws or []) if g and len(g) == 2]
            except Exception as e:
                log(f"   [roam] static-connection KB load skipped: {e} (LOUD)")
            self._host_gw_cache = idx
        return idx

    def _grind_wild_band(self, map_id):
        """GRIND-SPOT LEVEL KB (NS#5, rule 14) — the (wild_min, wild_max) encounter-level band for a
        grind area, from gamedata/frlg_grind_spots.json. Used by the level-aware grind-spot picker to
        recognise a GRIND-INADEQUATE map (wild_max far below the team's grind target => ~0 XP, the
        documented NS#1/#14 E4-prep stall) and prefer a reachable area that actually levels the team.
        Loaded once, cached; the swappable game-knowledge layer, keyed by "<group>,<num>" like the
        table. None = no data for this map (picker then falls back to the base behaviour). Fail-open."""
        idx = getattr(self, "_grind_band_cache", None)
        if idx is None:
            idx = {}
            try:
                import json
                path = os.path.join(_HERE, "gamedata", "frlg_grind_spots.json")
                with open(path, encoding="utf-8") as f:
                    raw = (json.load(f) or {}).get("spots") or {}
                for key, spec in raw.items():
                    try:
                        g, n = (int(x) for x in key.split(","))
                        lo, hi = int(spec["wild_min"]), int(spec["wild_max"])
                        idx[(g, n)] = (lo, hi, spec.get("terrain", "grass"))
                    except Exception:
                        continue
            except Exception as e:
                log(f"   [grind] grind-spot KB load skipped: {e} (LOUD)")
            self._grind_band_cache = idx
        band = idx.get(tuple(map_id))
        return (band[0], band[1]) if band else None

    def _grind_inadequate(self, map_id, target_level, poor_gap=None):
        """True iff this map's grass would give the team NEAR-ZERO XP for the current grind target —
        the documented NS#1/#14 stall (an L45+ mon on L8-19 grass gains ~0 per kill, so grind() spins
        its whole budget for ~0 levels). Predicate only (no side effects): wild_max is far below the
        target by more than `poor_gap`. Unknown map / no KB entry => False (never blocks a grind on
        data we don't have — fail-open). `poor_gap` default from POKEMON_GRIND_POOR_GAP (18): FRLG XP
        stays worthwhile until the foe is roughly this far below you, then falls off a cliff."""
        band = self._grind_wild_band(map_id)
        if band is None or not target_level:
            return False
        gap = poor_gap if poor_gap is not None else GRIND_POOR_GAP
        return band[1] < target_level - gap

    def _better_grind_spot(self, state, target_level):
        """The picker's PARK-SAFETY gate: return the nearest ADEQUATE grass map she can reach WITHOUT
        crossing a wall (ungated, not grind-dead, not already-inadequate) whose band actually levels a
        mon at `target_level` — or None if no such spot exists. The grind()/_grass_target level-aware
        branch only ever stands down from an inadequate spot when this returns a REAL better one, so a
        map that is the ONLY reachable grass is NEVER abandoned (she grinds the poor grass rather than
        freeze — the anti-park/anti-freeze invariant). Reuses the SAME world-graph reachability the
        base _grass_target trusts, so it can never propose a spot the base logic couldn't reach."""
        try:
            cur = tuple(state["map"])
        except Exception:
            return None
        avoid = self._wall_avoid(state)
        pcount = state.get("party_count")
        plevel = state["party"][0]["level"] if state.get("party") else None
        dead = getattr(self, "_grind_dead", set())
        inadeq = getattr(self, "_grind_inadequate_set", set())
        try:
            known = self.world.reachable_with_trait(cur, "has_grass", avoid) or []
        except Exception:
            return None
        best = None                                # (dst, wild_max) — prefer the highest usable band
        for _entry in known:
            dst = tuple(_entry[0])
            if dst == cur or dst in dead or dst in inadeq:
                continue
            if self.strat.is_gated(dst, pcount, plevel):
                continue
            if self._grind_inadequate(dst, target_level):
                continue                           # another poor spot is not "better"
            if self._grind_wild_band(dst) is None:
                continue                           # only propose spots we have level data for
            if not self.world.next_hop(cur, dst, avoid):
                continue                           # must have a real first hop (rideable now)
            hi = self._grind_wild_band(dst)[1]
            if best is None or hi > best[1]:
                best = (dst, hi)
        return best[0] if best else None

    def _keeper_gateway(self, host_area, cur, state, avoid=None):
        """STATIC-CONNECTION reachability (NS#40): the NEAREST overworld GATEWAY of `host_area` the
        learned-graph traveler can actually RIDE to now — or `cur` itself if she's already standing on
        one. Same bound + guards as the learned host scan (within KEEPER_ROUTER_MAX_HOPS, a real
        rideable next hop, not retired to _keeper_unreach) so a static offer is only ever made for a
        gateway she can genuinely reach this tick. None = no rideable gateway. Fail-closed."""
        gws = self._host_gateways().get(host_area) or []
        if not gws:
            return None
        if avoid is None:
            avoid = self._wall_avoid(state)
        unreach = getattr(self, "_keeper_unreach", set())
        best = None                                # (gateway_map, hops)
        for gw in gws:
            gw = tuple(gw)
            if gw == cur:
                return gw                          # already on a gateway -> step through the door
            if (cur, gw) in unreach:
                continue
            route = self.world.route(cur, gw, avoid)
            if not route:
                continue
            hops = len(route) - 1
            if hops > KEEPER_ROUTER_MAX_HOPS:
                continue
            if self._next_step_rideable(cur, gw, avoid) is None:
                continue
            if best is None or hops < best[1]:
                best = (gw, hops)
        return best[0] if best else None

    def _keeper_route_target(self, state):
        """CROSS-MAP KEEPER ROUTER (Part-C executor, PASS 3 NEW#2): the forward-plan's DUE keeper species
        + the NEARBY reachable map that HOSTS it, when it is NOT on the current map and the party has ROOM
        — returns (species, target_map) to route+catch there, else None. This is the off-map sibling of
        _plan_keeper_target (which only fires when she's already standing on the keeper's route): it lets
        her take a BOUNDED detour to a keeper on a corridor-adjacent map instead of marching past it.
        Deferrals (all return None): router flagged off; planner off; party full (>=6 — box swap is a
        separate unbuilt piece) or empty; no catch_keeper due; keeper already on THIS map (the on-map
        un-gate owns that); no world-graph route to any hosting map; nearest hosting map is farther than
        KEEPER_ROUTER_MAX_HOPS (avoid a whole-map backtrack — watchable). Mode-side, plan-gated, fail-closed."""
        if not KEEPER_ROUTER_ENABLED:
            return None
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return None
            pc = state.get("party_count")
            if pc is None:
                pc = len(state.get("party") or [])
            if pc >= 6 or pc < 1:
                return None                        # no room (box swap unbuilt) / empty party
            act = self.team_planner.assess(state.get("party") or [], state.get("badge_count", 0),
                                           bag=state.get("bag"), dex=state.get("dex_caught"),
                                           post_game=bool(state.get("post_game")))
            if not act or act.get("kind") != "catch_keeper":
                return None
            sp = (act.get("species") or "").lower()
            if not sp:
                return None
            cur = tuple(tv.map_id(self.b))
            if self._species_on_map(sp, cur):
                # ON the keeper's map. A GRASS map -> the on-map un-gate (wander_catch) owns the catch;
                # defer. But a CAVE host (interior, no grass) offers NO wander_catch (it needs reachable
                # grass) -> keep the fetch errand DRIVING so _fetch_keeper_errand runs the cave descend +
                # step-encounter wander here (NS#40), else she'd fall to head_to_gym and leave uncaught.
                if cur[0] != 3:
                    if (cur, cur) in getattr(self, "_keeper_unreach", set()):
                        return None                # cave hunted out (all floors barren) -> stop re-offering
                    return (sp, cur)
                return None                        # grass map -> the on-map un-gate catches it
            tmap = self._reachable_keeper_host(sp, cur, state)
            if tmap is None:
                return None
            return (sp, tmap)
        except Exception as e:
            log(f"   [roam] keeper-route target skipped: {e}")
            return None

    def _reachable_keeper_host(self, sp, cur, state):
        """Given a DUE keeper species `sp` and the current map `cur`, return the NEAREST hosting map the
        learned-graph traveler can actually RIDE to now (within KEEPER_ROUTER_MAX_HOPS), else None. Shared
        by the router (_keeper_route_target) AND the box-swap gate (_chaff_swap_target) so a full-party chaff
        is boxed ONLY when the keeper it makes room for is genuinely fetchable — never a box-for-nothing that
        thins the team and breaks the on-screen 'making room for the mon my plan wants' promise (NS#39: on
        erika_done_kit box_chaff boxed a chaff at Celadon for a Diglett that was un-routable → net team loss)."""
        areas = (self.team_planner.encounters or {}).get("areas") or {}
        p2m = self._place_to_map_index()
        # STORY-GATE avoid (NS#42): the keeper router must NOT route through Flute-gated Route 12/16 pre-Flute
        # — ns42_probe fetched 'growlithe' (Route 8, genuinely past Rock Tunnel) and world.route found a path
        # THROUGH Route 12, so the offer fired + the errand hopped onto Route 12 and wedged 9 ticks. Gating the
        # reachability check here means a keeper reachable ONLY via a gated dead-end is correctly deemed
        # unreachable -> not offered -> she falls to head_to_gym/billed road. Same fix as the off-road steer.
        # HARD-GATE avoid (NS#43): AND not across a hard gate she hasn't opened — Lavender/Rock-Tunnel pre-Flash
        # (east) OR Saffron pre-Tea (west). The growlithe (Route 7/8) offer otherwise wedged BOTH ways (east:
        # ping-pong vs the Flash errand; west: stuck in the Saffron gatehouse on 'leave_building').
        avoid = (self._wall_avoid(state) | self._story_gate_avoid(state)
                 | self._keeper_hard_gate_avoid(state))
        unreach = getattr(self, "_keeper_unreach", set())
        best = None                                # (target_map, hops)
        for place, entry in areas.items():
            present = any((m.get("species") or "").lower() == sp
                          for meth, mons in entry.items() if not str(meth).startswith("_")
                          for m in (mons or []))
            if not present:
                continue
            tmap = p2m.get(place)
            if not tmap or tuple(tmap) == cur or (cur, tuple(tmap)) in unreach:
                continue                           # retired: the executor couldn't ride there from here
            route = self.world.route(cur, tuple(tmap), avoid)
            if not route:
                continue
            hops = len(route) - 1
            if hops > KEEPER_ROUTER_MAX_HOPS:
                continue
            # OFFER⟺EXECUTABLE (route3_caught livelock fix): world.route can find a path over STATIC
            # connections the learned-graph traveler can't yet RIDE (unvisited maps) -> the errand spins
            # no_path forever. Require a real rideable next hop toward tmap NOW, so we only offer what
            # _travel_to_known can actually execute this tick (keeps the router to near/known keepers).
            if self._next_step_rideable(cur, tuple(tmap), avoid) is None:
                continue
            if best is None or hops < best[1]:
                best = (tuple(tmap), hops)
        if best is not None:
            return best[0]
        # STATIC-CONNECTION PASS (NS#40): no LEARNED route to any hosting map — but the host may be a
        # cave/interior entered by a DOOR from a corridor she hasn't walked yet (Diglett's Cave off
        # Route 11), invisible to the learned graph. Consult the static gateway KB: if a hosting area's
        # overworld gateway is ride-reachable NOW, the host IS fetchable (ride to the gateway, step
        # through the live-read door). Returns the host ENTRANCE map (multi-map areas keep the lowest
        # num). Flag-gated (KEEPER_STATIC_ROUTE) — isolated to the router, never touches world.route.
        if KEEPER_STATIC_ROUTE_ENABLED:
            static_best = None                     # (entrance_map, gateway_hops)
            for place, entry in areas.items():
                present = any((m.get("species") or "").lower() == sp
                              for meth, mons in entry.items() if not str(meth).startswith("_")
                              for m in (mons or []))
                if not present:
                    continue
                tmap = p2m.get(place)
                if not tmap or tuple(tmap) == cur or (cur, tuple(tmap)) in unreach:
                    continue
                if not self._host_gateways().get(place):
                    continue                       # only door-entered hosts have a static gateway
                gw = self._keeper_gateway(place, cur, state, avoid)
                if gw is None:
                    continue
                ghops = 0 if gw == cur else (len(self.world.route(cur, gw, avoid) or [gw]) - 1)
                if static_best is None or ghops < static_best[1]:
                    static_best = (tuple(tmap), ghops)
            if static_best is not None:
                return static_best[0]
        return None

    def _fetch_keeper_errand(self, state):
        """Handler for the `fetch_keeper` action: advance the keeper detour one leg via the PROVEN
        learned-graph traveler (_travel_to_known — the same one-hop-per-tick warp/edge machinery
        head_to_gym rides, NOT the naive trav.travel that spun no_path). On the keeper's map already ->
        targeted catch (roster-judgment bypassed — it's on the plan). Else route one hop; if that hop
        arrives, catch immediately. STALL GUARD (route3_caught livelock fix): a routing leg that makes NO
        map/coord progress increments a per-target counter; after KEEPER_ROUTER_STALL_CAP the (cur,tmap)
        pair is retired to _keeper_unreach so it stops being offered and she falls through to head_to_gym.
        Re-entrant + fail-closed: an evaporated target returns 'keeper_none' (benign — roam re-decides)."""
        tgt = self._keeper_route_target(state)
        if not tgt:
            return "keeper_none"
        sp, tmap = tgt
        cur = tuple(tv.map_id(self.b))
        host_area = self._place_name(tmap)
        # ALREADY at the host (any sub-map of a multi-map cave shares the area name/encounters) -> catch.
        if cur == tmap or self._species_on_map(sp, cur):
            # CAVE host (interior, no grass): the entrance sub-map may be a barren vestibule (Diglett's
            # Cave (1,38) has no wild table); wander THIS floor then DESCEND an internal warp toward the
            # encounter floor (NS#40 _cave_fetch_catch). A grass host -> the plain wander catch.
            if cur[0] != 3:
                return self._cave_fetch_catch(sp, host_area)
            log(f"   [roam] FETCH-KEEPER: on {self._place_name(cur)} — catching planned keeper '{sp}'")
            return self.catch_one(target_species=sp)
        avoid = (self._wall_avoid(state) | self._story_gate_avoid(state)      # NS#42: never route the errand through a Flute-gated dead-end
                 | self._keeper_hard_gate_avoid(state))                       # NS#43: nor across an un-opened hard gate (Rock Tunnel/Lavender, Saffron)
        pos0 = (cur, tuple(tv.coords(self.b) or ()))
        # STATIC-CONNECTION route (NS#40): if the host has no LEARNED route but IS a door-entered
        # cave/interior with a reachable overworld GATEWAY, drive toward the gateway; once standing ON
        # it, step through the live-read door into the host (lands on whatever sub-map the door goes to,
        # where _species_on_map fires). Isolated to the router; the learned overworld path is unchanged.
        gw = None
        if (KEEPER_STATIC_ROUTE_ENABLED and self._host_gateways().get(host_area)
                and self.world.route(cur, tmap, avoid) is None):
            gw = self._keeper_gateway(host_area, cur, state, avoid)
        if gw is not None and cur == gw:
            r = self._enter_host_via_gateway(sp, tmap, host_area)
        elif gw is not None:
            log(f"   [roam] FETCH-KEEPER: {self._place_name(cur)} -> {host_area} via gateway "
                f"{self._place_name(gw)} (static connection) for planned keeper '{sp}'")
            r = self._travel_to_known(f"travel:{gw[0]},{gw[1]}", state, hunt_on_arrival=False)
        else:
            log(f"   [roam] FETCH-KEEPER: routing to {host_area} for planned keeper '{sp}' "
                f"(party has room, within {KEEPER_ROUTER_MAX_HOPS} hops)")
            r = self._travel_to_known(f"travel:{tmap[0]},{tmap[1]}", state, hunt_on_arrival=False)
        now = tuple(tv.map_id(self.b))
        if now == tmap or self._species_on_map(sp, now):   # arrived at the host this leg -> targeted catch
            self._keeper_stall = {}                # reset the stall bookkeeping on real arrival
            # CAVE host (NS#41): the door may land on a barren vestibule (Diglett's Cave (1,38) has no wild
            # table); route the SAME-TICK arrival through the descend-aware bounded catch so she doesn't burn
            # the full plain-catch_one budget wandering the encounter-less entrance before ever descending.
            if now[0] != 3:
                return self._cave_fetch_catch(sp, host_area)
            return self.catch_one(target_species=sp)
        pos1 = (now, tuple(tv.coords(self.b) or ()))
        # A battle_loss BLACKOUT relocates her (respawn at a Center) — that position change is NOT progress
        # toward the keeper; counting it as progress reset the stall guard and let a thin team throw itself
        # at an unwinnable gauntlet between her and the keeper forever (misty_done solo-Ivysaur → Nugget
        # Bridge loss-loop). Treat a loss leg as non-progress so K of them retires the target → she resumes
        # leveling/the questline instead of the loss-loop.
        lost = isinstance(r, str) and "battle_loss" in r
        if pos1 != pos0 and not lost:              # real forward progress -> keep going next tick
            self._keeper_stall = {}
            return f"keeper_route:{r}"
        # NO progress (or a blackout bounce) this leg -> count toward retiring the target (never livelock)
        st_map = getattr(self, "_keeper_stall", None)
        if st_map is None:
            st_map = self._keeper_stall = {}
        key = (cur, tmap)
        st_map[key] = st_map.get(key, 0) + 1
        if st_map[key] >= KEEPER_ROUTER_STALL_CAP:
            if not hasattr(self, "_keeper_unreach"):
                self._keeper_unreach = set()
            self._keeper_unreach.add(key)
            log(f"   [roam] FETCH-KEEPER: '{sp}' at {self._place_name(tmap)} UNREACHABLE from "
                f"{self._place_name(cur)} ({st_map[key]} no-progress legs) — retiring it; back to the road")
            return "keeper_unreach"
        return f"keeper_route:{r}"

    def _enter_host_via_gateway(self, sp, tmap, host_area):
        """STATIC-CONNECTION door step (NS#40): standing on a gateway, find every live warp whose
        destination is the keeper HOST area, route to a reachable door tile and step through it (lands
        on whatever cave sub-map the door goes to, where _species_on_map(sp) fires for the caller's
        catch). Source-first: the door TILE is read live (tv.read_warps), not hardcoded — only the
        gateway map id lives in the KB. Reuses the proven read_warps + travel + enter_warp actuators.
        Returns a short status; the caller re-checks the live map after. Fail-closed + LOUD."""
        b = self.b
        before = tuple(tv.map_id(b))
        doors = []
        try:
            for wxy, wdest, _wid in tv.read_warps(b):
                if self._place_name(tuple(wdest)) == host_area:
                    doors.append(tuple(wxy))
        except Exception as e:
            log(f"   [roam] FETCH-KEEPER: gateway warp read failed: {e} (LOUD)")
            return "gateway_read_failed"
        if not doors:
            log(f"   [roam] FETCH-KEEPER: on {self._place_name(before)} but NO live door to {host_area} "
                f"in the map header from here — non-progress (LOUD)")
            return "gateway_no_door"
        for door in doors:
            log(f"   [roam] FETCH-KEEPER: {self._place_name(before)} -> {host_area} — stepping through "
                f"the door at {door} for planned keeper '{sp}'")
            try:
                self.trav.travel(target_map=None, arrive_coord=door, max_steps=400)
                self.enter_warp(pick=door)
            except Exception as e:
                log(f"   [roam] FETCH-KEEPER: door {door} step errored: {e} (LOUD)")
                continue
            if tuple(tv.map_id(b)) != before:
                return "gateway_entered"
        return "gateway_door_failed"

    def _cave_fetch_catch(self, sp, host_area):
        """CAVE step-encounter catch + DESCEND (NS#40): she's standing inside a cave keeper host. Wander
        THIS floor (bounded) for the target via the catch_one cave branch; if barren (no encounter/catch),
        step through an unvisited INTERNAL warp (dest = another sub-map of the SAME cave area) to the next
        floor and let the next tick hunt there. Diglett's Cave (1,38) is an encounter-less vestibule whose
        (6,4) warp descends to (1,37) — the on-map grass catch can't reach that floor, so the fetch errand
        drives the descent. When every reachable sub-map of this cave has been hunted with no target, the
        cave is retired (session _keeper_unreach) so she stops re-entering and returns to the road. Reuses
        read_warps + travel + enter_warp; per-floor bound KEEPER_CAVE_FLOOR_WANDER_S. Fail-closed + LOUD."""
        b = self.b
        cur = tuple(tv.map_id(b))
        p0 = b.rd8(ram.GPLAYER_PARTY_CNT)
        r = self.catch_one(max_seconds=KEEPER_CAVE_FLOOR_WANDER_S, target_species=sp)
        if r == "caught" or b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
            self._cave_floors_seen = set()             # reset the descent memo on a real catch
            return "caught"
        if r in ("no_balls", "need_pp", "healed_retry"):
            return r                                    # not a barren-floor signal — let roam handle it
        seen = self.__dict__.setdefault("_cave_floors_seen", set())
        seen.add(cur)
        try:
            warps = tv.read_warps(b)
        except Exception as e:
            log(f"   [roam] CAVE-FETCH: warp read failed on {self._place_name(cur)}: {e} (LOUD)")
            warps = []
        internal = [(tuple(xy), tuple(dest)) for xy, dest, _w in warps
                    if self._place_name(tuple(dest)) == host_area and tuple(dest) not in seen]
        if internal:
            xy, dest = internal[0]
            log(f"   [roam] CAVE-FETCH: {self._place_name(cur)} barren for {sp} — descending internal "
                f"warp {xy} -> the next {host_area} floor for the encounter table")
            before = cur
            try:
                self.trav.travel(target_map=None, arrive_coord=xy, max_steps=400)
                self.enter_warp(pick=xy)
            except Exception as e:
                log(f"   [roam] CAVE-FETCH: internal warp {xy} step errored: {e} (LOUD)")
            return "keeper_route:cave_descend" if tuple(tv.map_id(b)) != before else "keeper_route:cave_descend_stuck"
        # every reachable sub-map of this cave hunted, no target -> retire the cave, back to the road
        if not hasattr(self, "_keeper_unreach"):
            self._keeper_unreach = set()
        self._keeper_unreach.add((cur, cur))
        self._cave_floors_seen = set()
        log(f"   [roam] CAVE-FETCH: hunted every reachable {host_area} floor, no {sp} — retiring this cave "
            f"(back to the road; the on-map catch will re-try only after a roster change) (LOUD)")
        return "keeper_unreach"

    def catch_one(self, max_seconds=300, target_species=None):
        """AUTO catch a teammate (converts the route3_catch hand-play GATE): WANDER in the grass
        until a WILD encounter, then catch it (BattleAgent.catch_pokemon - weaken/status then commit
        to throws). A trainer's line of sight en route is fought (trainer mons can't be caught), then
        we keep wandering. Returns 'caught' | 'no_balls' | 'timeout' | 'battle_loss'.

        target_species (2026-07-09 Part-C executor): when set (a keeper species NAME, e.g. 'abra'),
        the wander SEEKS that ONE species — every non-target wild is FLED (no ball/PP waste, keep
        hunting) and the target is FORCE-caught (roster-judgment bypassed; it's on the plan, she's
        certain). None = the original judged forward-catch (roster_judgment picks keepers)."""
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
            # CAVE STEP-ENCOUNTER FALLBACK (NS#40): caves have NO grass but trigger wilds on every STEP.
            # For a TARGETED keeper fetch into a cave host (Diglett's Cave — the ground-slot Dugtrio
            # source the static-connection router now reaches), wander reachable WALKABLE non-warp tiles
            # so step-encounters fire and catch_runner takes the target on foot. Gated to a targeted fetch
            # in an interior whose area actually hosts the species — normal grass-catching is untouched,
            # and she never wanders a non-hosting interior (a Center/gym) looking for a wild.
            _here = tuple(tv.map_id(self.b))
            _cave = bool(target_species) and _here[0] != 3 and self._species_on_map(target_species, _here)
            if not _cave:
                log("   !! CATCH: no grass on this map - can't find a wild Pokemon here")
                return "no_grass"
            _doors0 = frozenset(self._door_tiles())
            _wlk0 = g_now.walkable
            def _reach0(tile):
                return bool(tv.bfs(g_now, cur0, lambda t: t == tile,
                                   walkable=lambda sx, sy: _wlk0(sx, sy) and (sx, sy) not in _doors0))
            # farthest-reachable-first so she STEPS the most (encounters fire per step); bounded BFS calls
            cand = [(sx, sy) for sy in range(g_now.sy_lo, g_now.sy_hi + 1)
                    for sx in range(g_now.sx_lo, g_now.sx_hi + 1)
                    if (sx, sy) != tuple(cur0) and (sx, sy) not in _doors0 and _wlk0(sx, sy)]
            cand.sort(key=lambda t: abs(t[0] - cur0[0]) + abs(t[1] - cur0[1]), reverse=True)
            wps, _tests = [], 0
            for t in cand:
                if _tests >= 60:
                    break
                _tests += 1
                if _reach0(t):
                    wps.append(t)
                if len(wps) >= 4:
                    break
            if not wps:
                log(f"   !! CATCH: cave {_here} hosts {target_species} but NO reachable walkable tile to "
                    f"wander from {cur0} for step-encounters — back to roam (LOUD)")
                return "no_reachable_target"
            grass = wps                       # reuse the wander loop below verbatim (waypoints + avoid=doors)
            log(f"   CATCH: cave step-encounter wander in {self._place_name(_here)} — {len(wps)} reachable "
                f"waypoint(s) {wps}; walking to draw out {target_species} (never a warp tile → won't exit)")
        # Warp/door tiles to keep OUT of: pathing across a door warps us into a building (the Route 3
        # PC / Mt Moon mouth) and the catch wedges on an NPC inside (the (5,4) trap). Treat doors as
        # permanent blocks for the whole wander — same avoid-param pattern as cave warp-to-warp nav.
        doors = frozenset(self._door_tiles())
        # REACHABILITY SANITY (increment 3.5): never hand travel a grass tile it can't path to. BFS
        # from here (permissive — allow NPC tiles, those clear on their own) and drop the unreachable.
        # If NONE is reachable (walled off, or a bad early-spawn target — the live (9,60)->(2,2) wedge),
        # return UP to roam so the oracle picks a different action, instead of travel spinning a dead
        # target for minutes (capability-not-script: she decides, we don't script the escape).
        # WATER-START (night shift 10): from a surf-mounted (water) stand the land-only layer dies
        # at its own start tile — ride land+water when Surf is usable (executor owns the ceremony).
        _wlk = g_now.walkable_or_surf if self._surf_usable() else g_now.walkable
        def _reach(tile):
            return bool(tv.bfs(g_now, cur0, lambda t: t == tile,
                               walkable=lambda sx, sy: _wlk(sx, sy) and (sx, sy) not in doors))
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
        stuck_battles = [0]       # shift 3: unresolved 'stuck' wild battles this wander — a within-tick
        STUCK_CAP = 6             #   freeze-spin guard (rule 18). The no-progress budget can't catch this
        #                           class: she keeps stepping in grass BETWEEN stuck battles, so every leg
        #                           reads 'progressed' and fails never accrues (the Route-11 run4 livelock).

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
            # TARGETED CATCH (2026-07-09 Part-C executor): the wander is SEEKING one planned keeper.
            # FLEE every non-target wild (no ball/PP waste, ends fast, keep hunting — the hunting-player
            # pattern, same as the dex-errand dupe flee); the target falls through to the commit below
            # with roster-judgment BYPASSED (it's on the plan — no "is this one good?" beat, she's here
            # for it). If the target never appears the STUCK_CAP / timeout backstops surface it to roam.
            _tgt = (target_species or "").lower()
            if _tgt:
                _tfid = st.read_enemy_species(self.b)
                _tfname = st.SPECIES_NAME.get(_tfid, "") if _tfid else ""
                if _tfname.lower() != _tgt:
                    _out = self._flee_runner()
                    if _out == "stuck":
                        stuck_battles[0] += 1
                    return _out
                self.on_event(f"there it is — a wild {_tfname}! this is the one I came for.",
                              kind="roster", tier=2)
                # fall through to the commit (skip the judgment block)
            # BLOCK #3 — THE CHOICE (2026-07-06 nursery): a real player doesn't ball everything that
            # rustles the grass. Size the wild up (dupe/coverage/level/room), offer the call to the
            # oracle (hers, live), follow the framework's lean headless — and VOICE it both ways.
            elif CATCH_JUDGMENT_ENABLED:
                # FOE SOURCE = gEnemyParty[0] (written at wild-mon CREATION, before the battle
                # intro even fades in), NOT the gBattleMons snapshot: at hand-off time
                # gBattleMons[1] can still hold the PREVIOUS battle's foe — the voltorb_run1
                # stale read that judged "voltorb L14" (the LAST battle's foe) and balled a
                # second ekans while she voiced wanting electric coverage. read_battle stays
                # as the fallback only (gEnemyParty zeroed / decrypt error).
                fid = st.read_enemy_species(self.b)
                flvl = st.read_enemy_level(self.b)
                if not fid:
                    try:
                        rb = st.read_battle(self.b)
                        foe_mon = rb["enemy"] if rb else None
                    except Exception:
                        foe_mon = None
                    fid = (foe_mon or {}).get("species") or 0
                    flvl = (foe_mon or {}).get("level") or 0
                if fid:
                    fname = st.SPECIES_NAME.get(fid, f"pokémon #{fid}")
                    team = []
                    for i in range(min(self.b.rd8(ram.GPLAYER_PARTY_CNT), 6)):
                        sid = st.read_party_species(self.b, i)
                        team.append({"species_id": sid,
                                     "level": self.b.rd8(ram.GPLAYER_PARTY + i * st.PARTY_MON_SIZE + 0x54),
                                     "types": st.SPECIES_TYPES.get(sid, [])})
                    foe_desc = {"species_id": fid, "name": fname, "level": flvl,
                                "types": st.SPECIES_TYPES.get(fid, [])}
                    try:
                        _dex_new = ram.pokedex_owns(self.b, fid) is False   # DEX DOCTRINE: first-of-a-kind leans catch
                    except Exception:
                        _dex_new = False
                    # SPECIES-QUALITY (2026-07-08 mega-batch): hand roster_judgment the guide's keeper
                    # record so she recognises a rare/strong catch on sight (a Pikachu!), not just a type-hole.
                    _quality = self.planner.keeper(fname)
                    rec, reason, facts = roster_judgment(team, foe_desc, dex_new=_dex_new, quality=_quality)
                    pick = self._soul_choose(
                        "catch_judgment",
                        {"catch": f"throw a ball at this {fname} (L{foe_desc['level']})",
                         "skip": f"pass on the {fname} — fight or flee, keep hunting"},
                        {"place": f"a wild {fname} (L{foe_desc['level']}, "
                                  f"{'/'.join(t for t in foe_desc['types'] if t) or 'unknown type'}) "
                                  f"just appeared. sizing it up: {reason}. your call — is this one "
                                  f"joining the team?"})
                    decision = pick if pick in ("catch", "skip") else ("catch" if rec else "skip")
                    # DEX-GATE OVERRIDE (2026-07-09 shift 2): an active OWNED>=10 gate (Flash errand) makes
                    # a NEW species a MUST-catch even at a full party — it boxes, the dex still climbs. Only
                    # bumps genuine first-of-a-kind (never a dupe); the oracle can still veto with 'skip'.
                    if (getattr(self, "_dex_catch_all", False) and _dex_new
                            and not facts.get("dupe") and pick != "skip" and decision == "skip"):
                        decision = "catch"
                        reason = (f"{reason} — but Flash needs ten kinds and I've never caught a {fname}; "
                                  f"box it for the dex, it counts")
                    log(f"   CATCH-JUDGMENT: {fname} L{foe_desc['level']} -> {decision} "
                        f"(lean={'catch' if rec else 'skip'}, oracle={pick!r}) — {reason}")
                    if decision == "skip":
                        if not hasattr(self, "_catch_skip_voiced"):
                            self._catch_skip_voiced = set()
                        if fid not in self._catch_skip_voiced:
                            self._catch_skip_voiced.add(fid)
                            self.on_event(f"a {fname}… {reason}. not this one — moving on.",
                                          kind="roster", tier=1)
                        # DEX ERRAND: FLEE the dupe (2026-07-09 shift 3, the Route-11 stuck livelock).
                        # The old path FOUGHT every skip; on a dupe-dense route (Route 11 ekans/spearow)
                        # that drains attacking PP over dozens of fights -> PP-famine -> the engine returns
                        # 'stuck', and the wander's no-progress budget never trips -> within-tick spin.
                        # A hunting player RUNS from dupes: no PP cost, the battle ends fast, and she
                        # reaches the route's fresh-species ceiling sooner. Scoped to the errand
                        # (_dex_catch_all); normal roam still fights skips for bench XP.
                        _out = (self._flee_runner() if getattr(self, "_dex_catch_all", False)
                                else self.battle_runner())
                        if _out == "stuck":
                            stuck_battles[0] += 1
                        return _out                          # resolve it, keep hunting
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
            # MAP-BOUND WANDER (night shift 10, the world-tour class): the grass list is THIS
            # map's; a leg that drifts across a connection (detour/encounter/edge) makes every
            # remaining waypoint a coordinate on the WRONG map — the old loop kept feeding them
            # (drift even counted as 'progress') and marched her Indigo→Viridian→Pewter→Cerulean
            # →Pallet inside one catch action. The wander is bound to the scanned map.
            m_grass = tuple(tv.map_id(self.b))
            # BUDGET (Batch 5 P2): measure only GENUINE catch-wandering time — credit back the forced
            # trainer-fight wall-clock so a trainer gauntlet between her and the grass can't time the
            # catch out before she gets a single throw.
            while (time.time() - t0 - trainer_secs[0]) < max_seconds and not caught[0] \
                    and not out_of_balls[0] and not cant_weaken[0] \
                    and stuck_battles[0] < STUCK_CAP:
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
                if tuple(tv.map_id(self.b)) != m_grass:
                    log(f"   CATCH: wander drifted off the grass map {m_grass} -> "
                        f"{tuple(tv.map_id(self.b))} — ending the walk cleanly; roam re-decides "
                        f"from here (stale waypoints never chased cross-map)")
                    break
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
        if stuck_battles[0] >= STUCK_CAP:
            log(f"   !! CATCH: {stuck_battles[0]} unresolved 'stuck' wild battles this wander — bailing so "
                f"the errand moves on (within-tick freeze-spin guard, rule 18) (LOUD)")
            return "no_reachable_target"
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
                # an 8-frame tap that didn't move us is usually a TURN (she wasn't facing
                # that way) — NOT a wall. Tap the SAME key once more (now facing → exactly
                # one step; a 26-frame hold walks TWO tiles and overshoots, trace-proven);
                # only then judge the tile. The blind RIGHT-nudge below otherwise undoes
                # the turn and OSCILLATES forever (the Silph "elevation seal" ghost).
                self.b.press(key, 8, 8, self.render, owner="agent")
                if tv.coords(self.b) != cur:
                    moved = True
                    break
            if not moved:
                self.b.press("RIGHT", 8, 8, self.render, owner="agent")
        return tv.coords(self.b) == tile

    def _regroup_walk(self):
        """The EMPTY-OPTIONS floor's honest mover (night shift 9): heal is a no-op when healthy
        and transparent (returns her to the same tile) otherwise, so the floor could offer it
        forever without her moving an inch (banked_E4, Saffron (39,24), 25 straight decisions).
        Regroup RELOCATES: walk to this city's Center door (the known anchor real options
        re-emerge from), else ride a connector building out of the sealed pocket. Returns
        'regrouped' | 'need_heal' | 'stuck' (loud)."""
        m = tv.map_id(self.b)
        door = CITY_PC_DOORS.get(m)
        if door:
            appr = (door[0], door[1] + 1)
            cur = tv.coords(self.b)
            if cur and tuple(cur) != appr:
                grid = tv.Grid(self.b)
                if tv.bfs(grid, tuple(cur), lambda t: t == appr, walkable=grid.walkable):
                    if self.trav.travel(target_map=None, arrive_coord=appr,
                                        max_steps=300, max_seconds=90) == "arrived":
                        log(f"   [roam] REGROUP: walked to the Center anchor {appr} on {m}")
                        return "regrouped"
        # no (reachable) Center on this map — a connector building is the honest way out of a
        # sealed pocket (the same primitive the questline passthrough rides).
        m0 = tuple(m) if m else None
        pt = self._door_passthrough(want_map=None)
        if pt == "crossed" or (m0 and tuple(tv.map_id(self.b)) != m0):
            # the attempt chain can relocate her across maps without reporting 'crossed'
            # (observed live: Route 23 -> a city) — where she STANDS is the truth, not pt.
            log("   [roam] REGROUP: rode a connector building out of the pocket")
            return "regrouped"
        if pt == "need_heal":
            return "need_heal"
        # CENTER-LESS TOWN (night shift 10, the e4surf Pallet terminal dead-air): this map has
        # no Center anchor and no connector fired — but the world graph knows cities that DO
        # have one (Pallet's is one land route north). Ride ONE hop toward the nearest per
        # regroup call; successive calls converge on a real anchor instead of looping 'stuck'
        # to the decision budget (25 straight regroup->stuck decisions, watched).
        try:
            best = None
            for city in CITY_PC_DOORS:
                if m0 is None or city == m0 or not self.world.node(city):
                    continue
                r = self.world.route(m0, city, [])
                if r and (best is None or len(r) < len(best[1])):
                    best = (city, r)
            if best:
                step = self._next_step_rideable(m0, best[0], [])
                if step is not None:
                    nxt, kind, detail = step
                    log(f"   [roam] REGROUP: no Center here — riding toward "
                        f"{self.world.name(best[0])} (next hop "
                        f"{('warp ' + str(detail)) if kind == 'warp' else detail} -> "
                        f"{self.world.name(nxt)})")
                    if kind == "warp":
                        before = tuple(tv.map_id(self.b))
                        self.trav.travel(target_map=None, arrive_coord=detail, max_steps=300)
                        if tuple(tv.map_id(self.b)) == before:
                            self.enter_warp(pick=detail)
                        if tuple(tv.map_id(self.b)) != before:
                            return "regrouped"
                    else:
                        self._edge_travel(nxt, detail)
                        if m0 and tuple(tv.map_id(self.b)) != m0:
                            return "regrouped"
        except Exception as _rge:
            log(f"   [roam] !! REGROUP center-hop errored ({_rge}) — falling through LOUD")
        # IMPOSSIBLE-STAND / PARTIAL-VOID (night shift 10, the e4surf Pallet limbo): the reads
        # say she 'stands' on a non-walkable tile with ZERO walkable/surfable neighbours. No
        # legit stand reads like this (a door mat has an open front) — it's the QW-4 reboot
        # family wearing a plausible map id (map (3,0), party 6, badges 8 — _world_lost is
        # blind to it). Escalate to void recovery instead of looping 'stuck' to the budget.
        try:
            if self._impossible_stand():
                log(f"   [roam] !!!! IMPOSSIBLE-STAND: she 'stands' on fully-enclosed "
                    f"non-walkable {tv.coords(self.b)} on {tv.map_id(self.b)} — desynced/"
                    f"void-class world (partial QW-4). Reloading the last real state (LOUD).")
                if self._void_recover():
                    return "regrouped"
        except Exception as _ise:
            log(f"   [roam] !! impossible-stand probe errored ({_ise}) — continuing")
        log(f"   [roam] !! REGROUP: no Center anchor and no connector fired from {m} "
            f"{tv.coords(self.b)} — LOUD (a true seal; watchdogs carry it)")
        return "stuck"

    def heal_at_center(self, pc_door=VIRIDIAN_PC_DOOR):
        """AUTHENTIC Pokemon Center heal (no RAM poking): route to a Pokemon Center door (default
        Viridian; pass any city's PC door for NEAREST-center healing), enter, walk to the nurse
        counter, drive the YES heal dialogue, verify HP -> max, exit back to the city. PC interiors
        share ONE layout so only the overworld door differs. Returns 'healed' or 'stuck' (LOUD)."""
        h0 = self.lead_hp()
        city = tv.map_id(self.b)                  # the city we heal in (generic; was Viridian-only)
        # P-1(a): first Pokémon Center — the "they heal you for FREE?" delight beat.
        self._first_beat("center",
                         "so THIS is a Pokémon Center — they just… fix your whole team? "
                         "for free? I love this world already.")
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
        self._step_to(NURSE_FRONT_OVERRIDES.get(tuple(pc_door), NURSE_FRONT))
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

    def _worst_chaff_slot(self, party=None):
        """PC/BOX chaff-picker (Tier-1 #15): the party slot best to DEPOSIT to free room for a planned
        keeper — the lowest-LEVEL party member whose species is OFF-PLAN (planner._is_target_line False)
        and is NOT the lead (slot 0, the ace). Returns an int slot or None (no boxable chaff — every
        non-lead mon is on the plan, or party too small). Authoritative 'on plan' via the planner so a
        freshly-caught low-level KEEPER (an Abra L10) is never boxed as if it were fodder."""
        if party is None:
            party = (self.read_live_state() or {}).get("party") or []
        if len(party) <= 1:
            return None
        tp = getattr(self, "team_planner", None)
        cand = []
        for i, m in enumerate(party):
            if i == 0:                                   # never box the lead/ace
                continue
            sp = (m.get("species") or "").lower() if isinstance(m, dict) else str(m).lower()
            if not sp:
                continue
            on_plan = False
            try:
                on_plan = bool(tp and tp._is_target_line(sp))
            except Exception:
                on_plan = False
            if on_plan:
                continue                                  # a planned keeper — keep it
            lvl = m.get("level", 0) if isinstance(m, dict) else 0
            cand.append((lvl, i))
        if not cand:
            return None
        cand.sort()                                       # lowest level first
        return cand[0][1]

    def _chaff_swap_target(self, state):
        """box_chaff GATE (PC/BOX, NS#38): returns (slot, pc_door) when it's worth boxing a chaff to make
        room for a due keeper, else None. Fires only when: flag on; planner on; party is FULL (6 — box swap
        is exactly the full-party case; a party with room already lets the router catch directly); a keeper
        catch is DUE (assess -> catch_keeper); a boxable OFF-PLAN chaff exists; and she's standing in a city
        whose Center PC door is known (CITY_PC_DOORS) — so the deposit errand can actually reach a PC. Mode-
        side, plan-gated, fail-closed (the wedge-prone menu never even opens unless all of this holds)."""
        if not PCBOX_ENABLED:
            return None
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return None
            party = state.get("party") or []
            pc = state.get("party_count") or len(party)
            if pc < 6:
                return None                               # room already -> router catches directly
            act = self.team_planner.assess(party, state.get("badge_count", 0),
                                           bag=state.get("bag"), dex=state.get("dex_caught"),
                                           post_game=bool(state.get("post_game")))
            if not act or act.get("kind") != "catch_keeper":
                return None
            slot = self._worst_chaff_slot(party)
            if slot is None:
                return None                               # no boxable chaff (all on-plan)
            cur = tuple(tv.map_id(self.b))
            pc_door = CITY_PC_DOORS.get(cur)
            if not pc_door:
                return None                               # not in a mapped-Center city -> can't deposit here
            # ROUTABILITY GATE (NS#39): box a member ONLY if the keeper it frees room for is actually
            # fetchable now — either already on this map (the on-map un-gate will catch it once there's room)
            # or the cross-map router can ride to a hosting map. Boxing for an un-routable keeper (Diglett's
            # Cave from Celadon) thins the team AND makes an on-screen promise she can't keep -> refuse.
            sp = (act.get("species") or "").lower()
            if sp and not self._species_on_map(sp, cur) \
                    and self._reachable_keeper_host(sp, cur, state) is None:
                return None
            return (slot, pc_door)
        except Exception as e:
            log(f"   [roam] chaff-swap target skipped: {e}")
            return None

    def _box_keeper_swap_target(self, state):
        """swap_keeper GATE (PC/BOX, NS#39): the mirror of box_chaff for the OTHER half of the loop —
        when a PLANNED coverage keeper is sitting in the box (FRLG auto-boxes a mon caught at party-6, so
        the keeper the router/on-map catch grabbed while full is stuck in storage), FIELD it. Returns
        (box, slot, chaff_slot, pc_door) — chaff_slot is the party mon to deposit first when the party is
        FULL (None when there's already room), else None (no swap warranted). Fires only when: flag on;
        planner on; a boxed occupant is ON-PLAN (planner._is_target_line) AND in the CURRENT open box
        (withdraw_mon can't box-switch); if full, a boxable off-plan chaff exists to make room; and she's
        in a mapped-Center city. Mode-side, plan-gated, fail-closed."""
        if not PCBOX_ENABLED:
            return None
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED
            if not TEAM_PLANNER_ENABLED:
                return None
            cur = tuple(tv.map_id(self.b))
            pc_door = CITY_PC_DOORS.get(cur)
            if not pc_door:
                return None                               # not in a mapped-Center city
            cb, occ = self._box_scan()
            tp = self.team_planner
            # a boxed keeper = an on-plan species sitting in the CURRENTLY-OPEN box (withdraw is box-local)
            keeper = None
            for (bx, sl), sp in sorted(occ.items()):
                if bx != cb:
                    continue
                name = st.SPECIES_NAME.get(sp, "").lower()
                try:
                    if name and tp._is_target_line(name):
                        keeper = (bx, sl)
                        break
                except Exception:
                    continue
            if keeper is None:
                return None                               # nothing on-plan is boxed here
            party = state.get("party") or []
            pc = state.get("party_count") or len(party)
            if pc < 6:
                return (keeper[0], keeper[1], None, pc_door)   # room already — just withdraw
            chaff = self._worst_chaff_slot(party)
            if chaff is None:
                return None                               # full of on-plan mons — can't make room
            return (keeper[0], keeper[1], chaff, pc_door)
        except Exception as e:
            log(f"   [roam] keeper-swap target skipped: {e}")
            return None

    def _find_pc_stand(self):
        """GENERAL PC-console locator (NS#38): scan the current interior's metatile behaviors for the PC
        tile (MB_PC = 0x83) and return the stand tile directly BELOW it (you face UP to use a PC). Centers
        do NOT all share the PC position (Route 10's PC is a different tile than Vermilion's (11,1)), so a
        hardcoded stand wedges — this reads the truth from RAM. Returns a save-coord (sx,sy) or None (no PC
        tile found → caller falls back to the PC_STAND default). Same behavior-read recipe as travel.Grid."""
        try:
            from travel import BACKUP_LAYOUT, GMAPHEADER, MAP_OFFSET, NUM_PRIMARY
            b = self.b
            w = b.rd32(BACKUP_LAYOUT); h = b.rd32(BACKUP_LAYOUT + 4); mp = b.rd32(BACKUP_LAYOUT + 8)
            ml = b.rd32(GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
            if not (0 < w < 1000 and 0 < h < 1000):
                return None
            for by in range(h):
                for bx in range(w):
                    e = b.rd16(mp + by * w * 2 + bx * 2); mid = e & 0x3FF
                    base, idx = (attr[0], mid) if mid < NUM_PRIMARY else (attr[1], mid - NUM_PRIMARY)
                    if (b.rd32(base + idx * 4) & 0xFF) == 0x83:          # MB_PC
                        return (bx - MAP_OFFSET, (by + 1) - MAP_OFFSET)  # stand one tile below, face UP
        except Exception as e:
            log(f"   PCBOX: PC-tile scan failed ({e}) — using default stand")
        return None

    def deposit_mon(self, slot, pc_door):
        """AUTHENTIC PC deposit (Tier-1 #15, ported from the screenshot-calibrated recon_pcbox drive):
        route into the current city's Center (reusing heal_at_center's proven enter/exit), boot Bill's PC
        at the shared-layout console (PC_STAND), drive DEPOSIT for party slot N, verify the party count
        dropped by 1, exit to the overworld. Returns 'deposited' | 'no_pc' | 'stuck' | 'unchanged'. Drives
        by SCREEN STATE (FRLG text boxes eat frame-timed presses), same as heal_at_center. Wedge-prone menu
        on the long core -> flag-gated (POKEMON_PCBOX); never poke RAM to move the mon (authentic only)."""
        import numpy as _np
        n0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if n0 <= 1 or slot <= 0 or slot >= n0:
            log(f"   !! PCBOX: refusing to deposit slot {slot} of a {n0}-mon party (guard)"); return "unchanged"
        city = tv.map_id(self.b)
        log(f"   PCBOX: depositing party slot {slot} at the Center on {city} (door {pc_door})")
        # 1) route to the PC door + step in (identical to heal_at_center's entry)
        if self.trav.travel(target_map=None, arrive_coord=(pc_door[0], pc_door[1] + 1),
                            max_steps=400, max_seconds=120) != "arrived":
            log(f"   !! PCBOX: couldn't reach the PC door (at {tv.coords(self.b)})"); return "no_pc"
        before = tv.map_id(self.b)
        for _ in range(6):
            self.b.press("UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
            if tv.map_id(self.b) != before:
                break
        if tv.map_id(self.b) == city:
            log("   !! PCBOX: did not enter the Center"); return "stuck"
        for _ in range(90):
            self.b.run_frame()
        self.b.set_input_owner("agent")
        # 2) walk to the PC console, boot it (RAM-located per-Center; PC_STAND is the shared-layout fallback)
        stand = self._find_pc_stand() or PC_STAND
        log(f"   PCBOX: PC console stand = {stand}"
            + ("" if stand != PC_STAND else " (default — no PC tile scanned)"))
        self._step_to(stand)

        def _press(key, settle=30):
            self.b.press(key, 8, 10, self.render, owner="agent")
            for _ in range(settle):
                self.b.run_frame()

        def _px():
            return _np.asarray(self.b.frame_rgb(), dtype=_np.int32)

        def _menu_open(img):                              # the "Which PC?" / storage list draws top-left
            reg = img[8:56, 8:110]
            return (reg > 235).all(axis=2).mean() > 0.5

        _press("UP", 16)                                  # face the console
        _press("A", 60)                                   # boot the PC
        # 3) drain boot text -> "Which PC?" list -> BILL'S PC -> storage menu
        ok = False
        for _ in range(6):
            _press("A", 45)
            if _menu_open(_px()):
                ok = True
                break
        if not ok:
            log("   !! PCBOX: PC-choice menu never appeared"); self._exit_to_overworld(); return "stuck"
        base_tr = _px()[4:28, 188:236]                    # top-right corner: only the full storage applet repaints it

        def _applet():
            return _np.abs(_px()[4:28, 188:236] - base_tr).mean() > 30

        _press("A", 45)                                   # BILL'S PC (cursor boots on top)
        storage = False
        for _ in range(8):
            if _menu_open(_px()):
                storage = True
                break
            _press("A", 70)
        if not storage:
            log("   !! PCBOX: storage menu never appeared"); self._exit_to_overworld(); return "stuck"
        _press("DOWN", 30)                                # WITHDRAW -> DEPOSIT POKeMON
        base_tr = _px()[4:28, 188:236]                    # room corner visible now; the deposit GUI kills it
        _press("A", 45)
        opened = False
        for _ in range(120):
            self.b.run_frame()
            if _applet():
                opened = True
                break
        if not opened:
            log("   !! PCBOX: deposit GUI never opened"); self._exit_to_overworld(); return "stuck"
        for _ in range(160):                              # let the GUI finish drawing
            self.b.run_frame()
        for _ in range(slot):                             # cursor down to the target party slot
            _press("DOWN", 24)
        _press("A", 90)                                   # pick the mon -> action submenu
        _press("A", 180)                                  # DEPOSIT -> box select / deposit animation
        _press("A", 260)                                  # confirm
        # 4) leave the box system + the Center
        for i in range(12):
            if tuple(tv.map_id(self.b))[0] == 3:
                break
            _press("B", 40)
        self._exit_to_overworld()
        n1 = self.b.rd8(ram.GPLAYER_PARTY_CNT)            # verify AFTER settle/exit (mid-animation reads lie)
        log(f"   PCBOX: party count {n0} -> {n1} (map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
        if n1 != n0 - 1:
            log("   !! PCBOX: DEPOSIT FAILED (party count unchanged)"); return "unchanged"
        self.on_event(f"benched one of the team for now — making room for the mon my plan actually wants.",
                      kind="roster", tier=1)
        return "deposited"

    def _box_scan(self):
        """RAM-TRUTH of the PC storage (NS#39): (current_open_box, {(box,slot): species_id}) for every
        occupant. Decrypts each BoxPokemon with the same substruct scheme as the party read (recon_lapras;
        gPokemonStoragePtr = 0x03005010, 80-byte BoxPokemon, 14 boxes × 30). Read-only. Lets a caller
        locate a boxed keeper (e.g. one FRLG auto-boxed on a party-6 catch) to withdraw + field."""
        GSTORAGE_PTR, BOX_MON_SIZE, BOXES, PER_BOX = 0x03005010, 80, 14, 30
        b = self.b
        base0 = b.rd32(GSTORAGE_PTR)
        if not base0:
            return (0, {})
        cur_box = b.rd8(base0)
        found = {}
        for bx in range(BOXES):
            for sl in range(PER_BOX):
                mbase = base0 + 4 + (bx * PER_BOX + sl) * BOX_MON_SIZE
                pid = b.rd32(mbase)
                if pid == 0 and b.rd32(mbase + 4) == 0:
                    continue
                key = pid ^ b.rd32(mbase + 4)
                order = st._SUBSTRUCT_ORDER[pid % 24]
                sp = (b.rd32(mbase + 32 + order.index("G") * 12) ^ key) & 0xFFFF
                if 1 <= sp <= 411:
                    found[(bx, sl)] = sp
        return cur_box, found

    def withdraw_mon(self, box, slot, pc_door):
        """AUTHENTIC PC withdraw (Tier-1 #15, NS#39 — the reverse of deposit_mon; closes the PC/BOX loop so
        a keeper FRLG auto-boxed at party-6 can be FIELDED). Route into the current city's Center, boot
        BILL'S PC, drive WITHDRAW for the boxed mon at grid (box, slot), verify the party count ROSE by 1,
        exit. Returns 'withdrawn' | 'full' | 'no_pc' | 'stuck' | 'wrong_box' | 'empty' | 'unchanged'. Box
        SWITCHING is unbuilt → if the target isn't in the OPEN box it aborts LOUD ('wrong_box', never
        withdraws the wrong mon). Screen-state drive (FRLG text boxes eat frame-timed presses), authentic
        (never pokes RAM). NB: the Center-entry + PC-boot + storage-menu prefix parallels deposit_mon; a
        future _open_bill_pc extraction (dedupe the two) is a clean-up candidate, deferred to keep the
        VERIFIED deposit path byte-identical."""
        import numpy as _np
        n0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if n0 >= 6:
            log(f"   !! PCBOX: party already full ({n0}) — nothing to withdraw into"); return "full"
        cb, occ = self._box_scan()
        if (box, slot) not in occ:
            log(f"   !! PCBOX: no mon at box {box} slot {slot} (occupants={len(occ)})"); return "empty"
        if box != cb:
            log(f"   !! PCBOX: target in box {box} but the applet opens on box {cb} — box switching "
                f"unbuilt, aborting LOUD"); return "wrong_box"
        want_sp = occ[(box, slot)]
        city = tv.map_id(self.b)
        log(f"   PCBOX: withdrawing {st.SPECIES_NAME.get(want_sp, want_sp)} from box {box} slot {slot} "
            f"at the Center on {city} (door {pc_door})")
        # 1) route to the PC door + step in (identical entry to deposit_mon)
        if self.trav.travel(target_map=None, arrive_coord=(pc_door[0], pc_door[1] + 1),
                            max_steps=400, max_seconds=120) != "arrived":
            log(f"   !! PCBOX: couldn't reach the PC door (at {tv.coords(self.b)})"); return "no_pc"
        before = tv.map_id(self.b)
        for _ in range(6):
            self.b.press("UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
            if tv.map_id(self.b) != before:
                break
        if tv.map_id(self.b) == city:
            log("   !! PCBOX: did not enter the Center"); return "stuck"
        for _ in range(90):
            self.b.run_frame()
        self.b.set_input_owner("agent")
        stand = self._find_pc_stand() or PC_STAND
        log(f"   PCBOX: PC console stand = {stand}"
            + ("" if stand != PC_STAND else " (default — no PC tile scanned)"))
        self._step_to(stand)

        def _press(key, settle=30):
            self.b.press(key, 8, 10, self.render, owner="agent")
            for _ in range(settle):
                self.b.run_frame()

        def _px():
            return _np.asarray(self.b.frame_rgb(), dtype=_np.int32)

        def _menu_open(img):
            reg = img[8:56, 8:110]
            return (reg > 235).all(axis=2).mean() > 0.5

        _press("UP", 16)                                  # face the console
        _press("A", 60)                                   # boot the PC
        ok = False
        for _ in range(6):
            _press("A", 45)
            if _menu_open(_px()):
                ok = True
                break
        if not ok:
            log("   !! PCBOX: PC-choice menu never appeared"); self._exit_to_overworld(); return "stuck"
        _press("A", 45)                                   # BILL'S PC (cursor boots on top)
        storage = False
        for _ in range(8):
            if _menu_open(_px()):
                storage = True
                break
            _press("A", 70)
        if not storage:
            log("   !! PCBOX: storage menu never appeared"); self._exit_to_overworld(); return "stuck"
        base_tr = _px()[4:28, 188:236]

        def _applet():
            return _np.abs(_px()[4:28, 188:236] - base_tr).mean() > 30

        # storage list boots with the cursor on WITHDRAW (top item) — no DOWN press (that's deposit_mon's)
        _press("A", 45)                                   # WITHDRAW POKeMON -> box grid applet
        opened = False
        for _ in range(120):
            self.b.run_frame()
            if _applet():
                opened = True
                break
        if not opened:
            log("   !! PCBOX: withdraw GUI never opened"); self._exit_to_overworld(); return "stuck"
        for _ in range(160):                              # let the grid finish drawing
            self.b.run_frame()
        row, col = slot // 6, slot % 6                    # box grid = 6 cols × 5 rows
        for _ in range(row):
            _press("DOWN", 24)
        for _ in range(col):
            _press("RIGHT", 24)
        _press("A", 90)                                   # pick the mon -> action submenu (WITHDRAW on top)
        _press("A", 260)                                  # WITHDRAW -> fly-to-party animation
        # leave the box system + the Center
        for i in range(12):
            if tuple(tv.map_id(self.b))[0] == 3:
                break
            _press("B", 40)
        self._exit_to_overworld()
        n1 = self.b.rd8(ram.GPLAYER_PARTY_CNT)            # verify AFTER settle/exit
        got = [st.read_party_species(self.b, s) for s in range(n1)]
        log(f"   PCBOX: party count {n0} -> {n1} | species {got} (map={tv.map_id(self.b)})")
        if n1 != n0 + 1 or want_sp not in got:
            log("   !! PCBOX: WITHDRAW FAILED (party count unchanged / species absent)"); return "unchanged"
        self.on_event("pulled one of mine back out of the box — bringing them onto the active team.",
                      kind="roster", tier=1)
        return "withdrawn"

    def _box_chaff_errand(self, state):
        """Handler for the `box_chaff` action: deposit the gated chaff so the keeper router gets room.
        Returns 'boxed' (party dropped -> router fires next tick) | 'box_none' (evaporated) | the deposit
        failure string (benign — roam re-decides; the errand never livelocks: a failed deposit leaves the
        party unchanged so the gate simply re-offers or falls through to head_to_gym)."""
        tgt = self._chaff_swap_target(state)
        if not tgt:
            return "box_none"
        slot, pc_door = tgt
        r = self.deposit_mon(slot, pc_door)
        return "boxed" if r == "deposited" else f"box_{r}"

    def _swap_keeper_errand(self, state):
        """Handler for the `swap_keeper` action (NS#39): FIELD a boxed plan keeper. When the party is full,
        deposit the gated chaff FIRST (6→5) so withdraw has room, then withdraw the keeper (5→6) — one Center
        visit, chaff-for-keeper. When there's already room, withdraw directly. Returns 'swapped' | 'swap_none'
        | a failure string (benign — a failed leg leaves the party unchanged so roam just re-decides / falls
        through to head_to_gym; never livelocks). One Center trip: deposit_mon and withdraw_mon each route in
        and out on their own, so a mid-fail simply stops the swap with the party intact-or-improved."""
        tgt = self._box_keeper_swap_target(state)
        if not tgt:
            return "swap_none"
        box, slot, chaff, pc_door = tgt
        if chaff is not None:                             # full party -> make room before the withdraw
            rd = self.deposit_mon(chaff, pc_door)
            if rd != "deposited":
                return f"swap_{rd}"                       # room not freed -> abort with the party intact
            # the box grid is unchanged by a deposit-to-a-free-slot, but re-locate defensively (truth is cheap)
            cb, occ = self._box_scan()
            if (box, slot) not in occ or box != cb:
                return "swap_moved"                       # keeper shifted/box changed -> bail (party is 5, benign)
        rw = self.withdraw_mon(box, slot, pc_door)
        return "swapped" if rw == "withdrawn" else f"swap_{rw}"

    def grind(self, target_level, fragile=False, budget_s=480):
        """Train the lead to target_level in the grass, healing when low. Self-sufficient gym-readiness
        capability (the 'walk away' vision needs autonomous leveling). From a CITY with no grass
        (Pewter), cross EAST to the adjacent route (Route 3: grass + a trainer gauntlet = fast XP),
        grind, then cross back. Wilds are FOUGHT (XP), not fled. Returns 'ok' | 'battle_loss'.

        `fragile=True` (weak bench-mon grind): the lead can FAINT, so ONLY pace grass she can WALK BACK
        FROM to the safe start anchor — never cross a one-way ledge into a pocket where a faint would
        strand her un-healably (the Route-4-east heal-wedge). The tanky ace (fragile=False) is unfiltered
        (it never faints there, so Center-unreachable grass is fine — filtering it regressed the ace-grind).

        `budget_s` caps the grass wall-clock (default 480 keeps every existing caller byte-identical). A
        deep gap like the pre-Brock solo-Bulbasaur climb (Lv8->13 on Route-2 wilds only) needs a longer
        window than a one-level top-up, so grind_pre_brock passes a wider budget."""
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
        lvl_start = lvl()                                       # SPLIT-ROUTE HEAL-THRASH CAP (shift 13):
        exc0 = getattr(self, "_heal_excursion_n", 0)           # snapshot expensive heals + start level; a
        #                                                        grind that racks up cross-city excursions
        #                                                        with ZERO level gain is a split-route thrash
        nopath_n = 0                                           # NS#5: grass-unreachable travel count (split-map pocket)
        while lvl() < target_level and time.time() - t0 < budget_s:
            # ACE-DOWN GUARD (ns11 Route-11 livelock — see GRIND_ACE_BAIL_FRAC): on a FRAGILE bench-grind the
            # ace is a switched-in protector taking chip each wild. The instant it dips into the faint-risk
            # band, HEAL (whole team) before it goes down — while it's still alive to fight the escape battles
            # out of the grass — then surface partial progress so roam re-grinds with a topped-up ace. This
            # kills the death-spiral where a downed ace force-fields a moveless bench mon into a grass livelock
            # that even the heal excursion can't cross. Only fragile (bench) grinds: the non-fragile ace-grind
            # has the ace AS lead, whose faint travel() already handles via need_heal/battle_loss.
            if fragile and ACE_BAIL_ON and self._ace_hp_frac() <= GRIND_ACE_BAIL_FRAC:
                _af = int(self._ace_hp_frac() * 100)
                log(f"   GRIND: ACE dinged to ~{_af}% mid bench-grind — healing before it faints so it can "
                    f"never force-field a moveless bench mon into an unwinnable-grass livelock")
                # CRITICAL: restore the ALIVE ace to slot 0 AND drop the participation switch BEFORE the heal
                # walk — else the weak grind-mon (a Teleport-only Abra) is still the auto-sent lead and its
                # switch-to-ace can wedge on the white-box impostor, re-wedging the very escape we're doing.
                # With the alive ace leading + no switch, it wins the grass battles on the way to the Center.
                try:
                    battle_agent.PROTECT_LEAD_GRIND = False
                except Exception:
                    pass
                self._restore_ace()
                self.heal_nearest()
                return "ace_healed"                            # DISTINCT: grind_weak_members restores + passes
                #                                                it up so roam re-grinds with a full ace (one
                #                                                heal per tick, not a same-call heal-thrash)
            gs = grass_save()
            if not gs:
                # NO GRASS AT ALL on this map (koga_run4, water Route (3,37)): the old bare `break`
                # fell through to `return "ok"`, which GRIND-WEAK's while-loop treats as retryable —
                # the celadon_run3 192× spin reborn through this branch (thousands of no-grass loops
                # burning the whole grind budget). Same medicine as the strand branch below: remember
                # the dead map + return the DISTINCT sentinel so the caller stands down this tick.
                log(f"   !! GRIND: no grass reachable on {tv.map_id(self.b)} - stopping LOUD")
                if not hasattr(self, "_grind_dead"):
                    self._grind_dead = set()
                self._grind_dead.add(tuple(tv.map_id(self.b)))
                return "no_safe_grass"
            # GRIND-SPOT LEVEL AWARENESS (NS#5 lever a, flag-gated OFF until a giovanni_kit_g look-ahead
            # proves no park). If THIS map's grass is far below target (~0 XP — the NS#1/#14 E4-prep stall)
            # AND a reachable higher-level spot exists, stand down (mark inadequate like _grind_dead) so the
            # caller re-routes there instead of spinning the budget for ~0 levels. NEVER fires when this is
            # the only reachable grass (_better_grind_spot returns None) -> she grinds poor grass, never freezes.
            if GRIND_SPOT_LEVELAWARE:
                _hm = tuple(tv.map_id(self.b))
                if self._grind_inadequate(_hm, target_level):
                    _st = {"map": _hm, "party_count": self.b.rd8(ram.GPLAYER_PARTY_CNT),
                           "party": [{"level": lvl()}]}
                    if self._better_grind_spot(_st, target_level):
                        log(f"   GRIND: {_hm} wilds ~L{self._grind_wild_band(_hm)[1]} give near-0 XP toward "
                            f"L{target_level} — standing down to route to a reachable higher-level spot")
                        if not hasattr(self, "_grind_inadequate_set"):
                            self._grind_inadequate_set = set()
                        self._grind_inadequate_set.add(_hm)
                        return "no_safe_grass"
            cur = tv.coords(self.b) or (0, 0)
            if fragile:                                        # keep only grass she can RETURN from (no one-way strand)
                grid_now = tv.Grid(self.b)
                safe = [g for g in gs if g != anchor
                        and tv.bfs(grid_now, g, lambda t, a=anchor: t == a, walkable=grid_now.walkable)]
                if safe:
                    gs = safe
                elif [g for g in gs if g != cur]:              # ALL grass is a one-way strand-risk here ->
                    log(f"   !! GRIND[fragile]: all grass on {tv.map_id(self.b)} is a one-way strand from "
                        f"{anchor} - stopping this map's fragile grind LOUD")
                    # GRIND-DEAD MAP MEMORY (2026-07-07 celadon_run2 spin): the chooser re-picked
                    # prep-battle forever while this map kept refusing — remember the refusal so
                    # the PROACTIVE pin stands down here (mirror of the heal-dead-map guard); a
                    # recorded WALL target still dominates (walls are worth a detour, bench is not).
                    if not hasattr(self, "_grind_dead"):
                        self._grind_dead = set()
                    self._grind_dead.add(tuple(tv.map_id(self.b)))
                    return "no_safe_grass"                     # DISTINCT from 'ok' — GRIND-WEAK's
                    #                                            while-loop treated 'ok' as retryable
                    #                                            and re-entered 192x (celadon_run3)
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
                if r == "no_path":
                    # Couldn't REACH this grass tile (travel wedged). If we rack up no_path with ZERO
                    # level gain, the grass she "knows" here is sealed off from her current pocket (the
                    # Route-10 (11,79) south-pocket → north-grass split) — spinning the budget re-trying
                    # the same unreachable waypoints. Mark grind-dead + stand down (mirrors the strand /
                    # heal-thrash guards). Any level gained resets nothing because lvl()>lvl_start makes
                    # the cap unreachable — a productive grind (encounters firing) never trips this.
                    nopath_n += 1
                    if nopath_n >= GRIND_NOPATH_CAP and lvl() <= lvl_start:
                        log(f"   !! GRIND: {nopath_n} grass-unreachable travels on {tv.map_id(self.b)} "
                            f"with 0 level gain — the grass here is sealed off from this pocket (split-map "
                            f"strand); stopping LOUD")
                        if not hasattr(self, "_grind_dead"):
                            self._grind_dead = set()
                        self._grind_dead.add(tuple(tv.map_id(self.b)))
                        return "no_safe_grass"
                    continue
                if r == "need_heal":
                    self.heal_nearest()
                    # BAIL a split-route heal-thrash: N expensive cross-city excursions AND no level
                    # gained since the grind started => this grass is un-grindable from here (own Center
                    # unreachable). Stop LOUD, mark the map grind-dead, surface the DISTINCT sentinel so
                    # the caller (prep stand-down / GRIND-WEAK) picks Center-reachable grass or moves on.
                    if (getattr(self, "_heal_excursion_n", 0) - exc0 >= GRIND_HEAL_EXCURSION_CAP
                            and lvl() <= lvl_start):
                        log(f"   !! GRIND: {getattr(self, '_heal_excursion_n', 0) - exc0} cross-city heal-"
                            f"excursions on {tv.map_id(self.b)} with 0 level gain — split-route heal-thrash, "
                            f"this spot is un-grindable from here; stopping LOUD")
                        if not hasattr(self, "_grind_dead"):
                            self._grind_dead = set()
                        self._grind_dead.add(tuple(tv.map_id(self.b)))
                        return "no_safe_grass"
        log(f"   GRIND: trained to Lv{lvl()} (target {target_level})")
        if tv.map_id(self.b) != home and home == PEWTER:      # back to Pewter for the gym
            self.walk_to_map(PEWTER, "west")
        return "ok"

    def grind_pre_brock(self, target_level=13):
        """Get the solo starter Brock-ready BEFORE the Viridian Forest crossing (2026-07-09 night train).

        THE WALL this solves: the Forest leg FLEES wilds (the only reliable crossing — a thin solo
        starter that FIGHTS through the Forest drops below the heal floor mid-maze and heal-bounces to
        Viridian forever). But fleeing = ~0 XP, so she reaches Pewter at Lv8 with Tackle/Growl (no Vine
        Whip, learned ~Lv13) and LOSES to Brock's Onix (Lv14). A real player GRINDS the early route
        before the first gym; this is that, narrated.

        WHERE: Route 2 (map group 3), NOT the Forest interior (group 1). Route 2 has grass AND heals
        cleanly — its own-Center-less overworld tile routes heal_nearest's adjacent-city excursion SOUTH
        to Viridian (short, proven). The Forest (group 1) trips heal_nearest's 'inside a building complex'
        branch (it's not group 3), so grinding there can't heal reliably — exactly the bounce we're
        escaping. Trainers live in the Forest, so Route-2 XP is wilds-only (slower) but ROCK-solid; the
        subsequent ADVANCE_NORTH still FLEE-crosses the Forest. Non-fatal throughout: a positioning miss
        or a soft grind wall WARNs + proceeds (LEVEL_CHECK + the segment's blackout-recovery still cover a
        loss) — the spine never hard-stalls on a readiness gate."""
        def lead_lvl():
            return self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        lvl = lead_lvl()
        if lvl >= target_level:
            log(f"   GRIND-BROCK: lead already Lv{lvl} >= {target_level} — Brock-ready, skipping the grind")
            return "ok"
        # CONSTITUTION obligation #3 (narrated purpose, bedrock block #4): say WHY she's training, in
        # character — a real first-timer sizes up the gym and trains, she doesn't grind silently.
        self.on_event(
            f"My {self._lead_species_name()}'s only level {lvl} — Brock's Onix is level 14 and made of "
            f"rock. I'm not walking in there thin. Quick training run on Route 2 first, then we knock.")
        # 1) POSITION onto Route 2 (grass + short reliable heal-bounce to Viridian).
        m = tv.map_id(self.b)
        if m != ROUTE2:
            if m in (PALLET, ROUTE1):                          # just delivered the parcel down south
                self.walk_to_map(VIRIDIAN, "north")
            if tv.map_id(self.b) == VIRIDIAN:
                self.walk_to_map(ROUTE2, "north")
        if tv.map_id(self.b) != ROUTE2:
            log(f"   !! GRIND-BROCK: couldn't reach Route 2 (at {tv.map_id(self.b)}) — proceeding "
                f"underleveled (LOUD; LEVEL_CHECK + blackout-recovery still cover a Brock loss)")
            return "ok"
        # 2) GRIND the solo lead on Route-2 wilds. Wider wall-clock than a top-up (Lv8->13 is a real
        #    climb); grind() fights wilds, heals via the adjacent-Viridian excursion, and leaves her on
        #    Route 2 (home != PEWTER -> no walk-back), where ADVANCE_NORTH takes the wheel for the cross.
        r = self.grind(target_level, budget_s=int(os.getenv("POKEMON_BROCK_GRIND_S", "1200")))
        now = lead_lvl()
        if r == "battle_loss":
            log(f"   GRIND-BROCK: blacked out grinding (now Lv{now}) — recovery heals + the segment "
                f"re-runs; proceeding")
            return "ok"
        if now >= target_level:
            log(f"   GRIND-BROCK: trained to Lv{now} (>= {target_level}) — Brock-ready")
            self.on_event(f"Level {now}. Vine Whip's online and we've got the HP to eat an Onix hit. "
                          f"Okay Brock — we're ready for you.")
        else:
            log(f"   GRIND-BROCK: budget spent at Lv{now} (< {target_level}) — proceeding; Vine Whip's "
                f"4x on Onix so it's still winnable, and a loss re-runs")
        return "ok"

    def _lead_species_name(self):
        """Best-effort display name of the lead mon (for narrated grind/prep beats). Any read failure
        falls back to a neutral word so a voice beat never crashes the spine."""
        try:
            return st.SPECIES_NAME.get(st.read_party_species(self.b, 0), "starter").title()
        except Exception:
            return "starter"

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

    def _ace_hp_frac(self):
        """HP fraction of the ACE (the highest-level member, wherever it's currently slotted), or 1.0 if
        unreadable. Used by grind()'s ACE-DOWN GUARD to bail a fragile bench-grind before the ace faints
        (the ns11 Route-11 livelock). Reuses party_health() (cur 0x56 / max 0x58) so the offsets stay in
        one place."""
        ph = self.party_health()
        levels = self._party_levels()
        if not ph or not levels:
            return 1.0
        ace = max(range(len(levels)), key=lambda s: levels[s])
        for (s, hp, mx, frac) in ph:
            if s == ace:
                return frac
        return 1.0

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

    def _prep_e4_target(self, state, party):
        """PASS-3 team-depth: the whole-party FLOOR level for E4 readiness, or None. Fires only with all 8
        badges earned (the Victory Road / Indigo gauntlet ahead) and pre-credits. Reads the TeamPlanner
        archetype's level_milestones for the E4 ENTRY (Lorelei's band, ~L55 — the guide-literate 'level the
        whole six before a Center-less five-fight gauntlet' number, NOT Champion's 60) and returns it while
        a LEVELABLE member is still under it. Livelock-proof: 'levelable' = within E4_PREP_BAND of the target
        AND not already stall-marked, so far-underleveled chaff (box fodder, Tier-1 #15) never pins the grind
        and the target retires cleanly (→ None → she pushes on to VR) once the real team crosses OR the last
        levelable member stalls out on the reachable grass. Mode-side, plan-gated, fail-closed.

        WHY this exists (rule 2 — the load-bearing wiring IS the feature): before this, NOTHING read the
        team-plan's E4 milestone — the only late-game bench target was the ace-relative `lead-8`, which is
        exactly the top-heavy shape that whited out at Lorelei/Agatha/Lance across NS9-14. The DECISION to
        floor the WHOLE party (via the participation-XP grind_weak_members) is the fix; grind-spot adequacy
        for a L45→55 push is a separate downstream blocker the look-ahead surfaces, not a reason to leave
        the target wrong."""
        try:
            from pokemon_planner import TEAM_PLANNER_ENABLED, _E4_SEQ
            if not (TEAM_PLANNER_ENABLED and STRATEGIC_GRIND_ENABLED):
                return None
            if state.get("post_game") or int(state.get("badge_count", 0)) < 8:
                return None
            if len(party) < 2:
                return None
            pstate = getattr(self.team_planner, "state", None) or {}
            ms = pstate.get("level_milestones") or {}
            e4_bands = [ms[t] for t in _E4_SEQ if t in ms]
            if not e4_bands:
                return None
            e4 = int(min(e4_bands))                          # the E4 ENTRY bar (Lorelei ~55), not Champion 60
            stalled = getattr(self, "_grind_stalled", None) or set()
            for i, m in enumerate(party):
                lv = m.get("level", 0)
                if lv >= e4 or lv < e4 - E4_PREP_BAND:       # crossed, or hopeless chaff → not a grind target
                    continue
                try:
                    pid = self.b.rd32(ram.GPLAYER_PARTY + i * st.PARTY_MON_SIZE)
                except Exception:
                    pid = None
                if pid is not None and pid in stalled:       # can't gain XP on the reachable grass — skip
                    continue
                return e4                                    # a real, levelable member is under the E4 bar
            return None                                      # every real member crossed/stalled → proceed
        except Exception as e:
            log(f"   [roam] prep-for-e4 target skipped: {e}")
            return None

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
        # PREP STAND-DOWN (koga_run5, the Fuchsia pocket): when repeated grind attempts from this
        # position found NO reachable grass (one-way ledges east, grassless water south, unexplored
        # west), 'train first' is an UNEXECUTABLE plan — folding it anyway just has the oracle pick
        # 'battle' into an instant failure forever (the 5-second STALL). Two straight dry attempts
        # drop the plan so the gym rematch / forward road becomes the natural pick. Resets the
        # moment any grass is actually reached.
        if getattr(self, "_prep_dry", 0) >= 2:
            if not getattr(self, "_prep_dry_logged", False):
                self._prep_dry_logged = True
                log("   [roam] !! PREP STAND-DOWN: repeated grind attempts found NO reachable grass "
                    "from here — dropping the 'train first' plan (rematch / move forward instead)")
            return None
        try:
            party = state.get("party") or []
            if not party:
                return None
            # PREP-FOR-E4 (2026-07-11, PASS 3 team-depth) takes precedence late-game: at all 8 badges the
            # E4 gauntlet is the wall, and the whole party (not the ace) must be floored to survive it. This
            # overrides the ace-relative bench target below (the E4 milestone is higher + team-wide). Bounded
            # + livelock-proof inside _prep_e4_target; the prep-dry stand-down above still protects the
            # no-reachable-grass case (it runs before this and returns None). Only fires at badge 8.
            _e4t = self._prep_e4_target(state, party)
            if _e4t is not None:
                return _e4t
            # RIVAL-GAUNTLET DEFER (2026-07-10, NIGHT SHIFT 16 — the real S.S. Anne Gary livelock). A 4+-mon
            # RIVAL wall (Gary on the S.S. Anne; later Tower/Silph) is a TEAM-STRENGTH wall the STRATEGIC grind
            # path handles WRONG two ways: (1) switch-armed, it fields fodder to the foe's LEAD level
            # (underlevel_target ≈ L19) — the fodder gets swept while the ACE sits idle, "readiness" falsely
            # crosses on the team floor, she boards at ace L30 and loses again (the 12-shift livelock, s15); and
            # (2) forcing ace-overpower here still trap-grinds, because grind()'s auto grass-finder wedges in the
            # gym-approach POCKET at the Cut tree (Vermilion) or dives DOWN the one-way Route-4 ledge off-anchor
            # (Cerulean whiteout) she then can't path back UP (s16: stranded on Route 4, head_to_gym no_path, ace
            # treadmilled L35→39, PP-starved). So for a 4+-mon rival, return None and DEFER entirely to the
            # questline ANCHOR-PATH prep (_run_questline_step → PREP-BEFORE-RIVAL → hardcoded Route-6 walk_to_map),
            # which levels the ACE to the proven Venusaur L32 on SAFE grass AT the anchor. With the strategic
            # grind suppressed, off-anchor she doesn't trap-grind — head_to_gym/the billed road carries her from
            # the respawn town (Cerulean, a spine town — routing works) back to Vermilion, where the anchor grind
            # fires before she boards. General per rule 14 (every 4+-mon rival). 3-mon gym leaders are unchanged
            # (no ship-anchor questline to defer to — the normal path below still handles them).
            _wr = self.strat.active_wall_rec()
            if _wr and _wr.get("is_trainer") and (_wr.get("size") or 0) >= 4:
                return None
            if not battle_agent.GRIND_SWITCH_ENABLED:
                t = self.strat.overpower_target()           # ACE-OVERPOWER (switch gated)
                if not t:
                    return None
                ace = max(m["level"] for m in party)
                return t if ace < t else None
            t = self.strat.underlevel_target()              # FIELD-WEAK (switch armed)
            floor = min(m["level"] for m in party)
            if (not t and PROACTIVE_BENCH and len(party) >= 2
                    and tuple(tv.map_id(self.b)) not in getattr(self, "_grind_dead", ())):
                # WALL-LESS bench-raising (2026-07-06 nursery): a fresh catch shouldn't ride the bench
                # at L5 while the ace is L30 — when the floor sags PROACTIVE_BENCH_GAP under the lead,
                # prep toward lead-8 (modest, bounded; a recorded wall's target still dominates above).
                # PINNED AT ARM TIME (ship-run-2 lesson): participation XP levels the ACE too, so a
                # LIVE lead-8 target rises every grind cycle (25→26→…) and the bench chases its own
                # tail for the whole run budget. The goal freezes when prep arms; it retires when the
                # floor crosses it (a later re-sag re-pins fresh).
                lead = max(m["level"] for m in party)
                pin = getattr(self, "_bench_pin", None)
                sig = tuple(sorted(m["species"] for m in party))
                # MID-GAME MILESTONE CAP (2026-07-11, PASS 3 team-depth — the mid-game sibling of
                # _prep_e4_target): cap the bench pin at the team-plan's NEXT gym milestone (Brock 14
                # … Giovanni 52) instead of the ace-relative lead-8, so the bench climbs TOWARD each
                # gym's level over the whole game rather than trailing the ace by a fixed 8 (the
                # "arrives thin" shape that whited out NS9-14). Falls back to lead-8 when no milestone
                # is known (post-game / planner off). Milestones move only on a badge-earn (infrequent,
                # discrete), so re-arming on a milestone RISE can't treadmill the way the live lead-8
                # target did (ship-run-2/5). The E4 case (badge 8) is handled above by _prep_e4_target.
                try:
                    _ms = self.team_planner._next_milestone(
                        int(state.get("badge_count", 0)), bool(state.get("post_game")))[1]
                except Exception:
                    _ms = 0
                milestone = _ms if _ms else (lead - 8)
                # NS#10 BENCH-TO-MILESTONE: keep re-pinning toward the milestone in +6 bites (instead of
                # retiring after one) so the bench actually reaches the gym level — UNLESS this map's grass
                # proved a poor grind spot for this milestone (an unproductive bite, marked by the executor),
                # in which case the pin retires here so head_to_gym is restored and she MARCHES to better
                # grass. The productivity gate (poor-map release) is what keeps this park-proof.
                _keep_climbing = (
                    BENCH_TO_MILESTONE and _ms and floor < (milestone - BENCH_MS_CLOSE)
                    and (tuple(tv.map_id(self.b)), milestone) not in getattr(self, "_bench_poor_maps", ()))
                if pin is not None:
                    if floor >= pin:
                        if _keep_climbing:
                            self._bench_pin = min(milestone, floor + 6)   # another bite toward the milestone
                            t = self._bench_pin
                        else:
                            self._bench_pin = None          # goal reached / poor spot — the pin retires
                            self._bench_done_sig = sig       # remember WHO was prepped
                            self._bench_done_milestone = milestone   # …and at WHICH gym milestone
                    else:
                        t = pin
                elif floor < lead - PROACTIVE_BENCH_GAP:
                    # RE-ARM GUARD (ship-run-5: 567 battles): the ace outruns the bench forever, so a
                    # retired prep must NOT re-pin at floor+2 every tick (a TREADMILL). It re-arms only
                    # when the ROSTER CHANGED (a new member needs raising) OR the MILESTONE ROSE (a new
                    # gym earned → the whole bench should climb toward the new bar). Both are infrequent,
                    # discrete events — never the ace's continuous drift, so no treadmill.
                    # NOTE the `_ms and` guard: the milestone-RISE re-arm is allowed ONLY for a REAL
                    # static milestone (moves on badge-earn). In the lead-8 FALLBACK (_ms falsy) the
                    # "milestone" is the live ace-relative bar, which drifts up every grind — re-arming
                    # on its rise would reinstate the ship-run-5 treadmill, so the fallback keeps the
                    # roster-only guard.
                    if (getattr(self, "_bench_done_sig", None) != sig
                            or (_ms and milestone > getattr(self, "_bench_done_milestone", 0))):
                        # SITTING CAP (2026-07-07 celadon_run1: Mankey L10 vs lead L45 pinned a
                        # 27-level marathon that parked the badge-4 road): a real player raises the
                        # bench a FEW levels per stop, not to the milestone in one sitting. Raise the
                        # floor in +6 bites toward the milestone; the pin retires/re-arms per the guard.
                        self._bench_pin = min(milestone, floor + 6)
                        t = self._bench_pin
            if not t:
                return None
            return t if floor < t else None
        except Exception as e:
            log(f"   [roam] prep-team target skipped: {e}")
            return None

    def _prep_team_weak(self, state, target):
        """The under-target members' species names (for the rationale/framing). Pure read off state."""
        return [m["species"] for m in (state.get("party") or []) if m["level"] < target]

    def grind_weak_members(self, target, min_level=None):
        """Field the WEAK members (not the ace) and level the team FLOOR to `target`, then restore the
        ace. Each loop: pick the weakest under-target member, reorder it to lead, grind() it toward
        `target` (which heals it when low — survival), repeat until the floor crosses. Bounded by a
        wall-clock budget so a tick can't run away. Returns 'ready' (floor crossed) | 'battle_loss' |
        'ok' (budget/grass-out — partial progress, the next tick re-enters). Restores the ace on EVERY
        exit path (a faint/loss must not strand the weak mon as lead).

        `min_level` (2026-07-11, PASS 3): a FLOOR below which a member is ignored as box-fodder chaff —
        the E4-prep path passes `target - E4_PREP_BAND` so a full-team floor of L55 does NOT drag an L8
        Rattata into the grass (unwatchable + faint-thrash; that fodder belongs in the box, Tier-1 #15).
        None (every other caller) = original behaviour: field every under-target member."""
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
        # STALL SET (2026-07-09, the Rattata loop): a bench mon too weak to survive/participate on this
        # route earns ZERO XP — grind() returns 'ok' with no level gain, GRIND-WEAK re-picks it as the
        # weakest, and it spins for the whole budget (then roam re-enters and it spins again). Track mons
        # that made no progress BY PID (survives slot-shuffles + roam re-entries) and stop re-fielding
        # them: a real trainer doesn't grind a hopeless mon forever — they push on with the strong core.
        stalled = getattr(self, "_grind_stalled", None)
        if stalled is None:
            stalled = self._grind_stalled = set()

        def _pid0():
            return self.b.rd32(ram.GPLAYER_PARTY)               # slot-0 personality value (per-mon id)
        try:
            while time.time() - t0 < GRIND_WEAK_BUDGET_S:
                levels = self._party_levels()

                def _slot_pid(s):
                    return self.b.rd32(ram.GPLAYER_PARTY + s * 100)
                weak = [s for s, l in enumerate(levels)
                        if l < target and _slot_pid(s) not in stalled
                        and (min_level is None or l >= min_level)]   # skip box-fodder chaff (E4-prep floor)
                if not weak:
                    remaining = [s for s, l in enumerate(levels) if l < target]
                    if remaining:                             # floor un-raisable — the under-target mon(s)
                        #                                       can't gain XP here; proceed, don't spin
                        log(f"   GRIND-WEAK: floor NOT crossed, but the remaining under-L{target} "
                            f"member(s) {remaining} can't earn XP here (stalled) — pushing on with the "
                            f"strong core (levels {levels}) instead of spinning")
                    else:
                        log(f"   GRIND-WEAK: floor crossed L{target} (levels {levels}) — done")
                    self._restore_ace()
                    return "ready"
                wk = min(weak, key=lambda s: levels[s])
                if wk != 0:
                    log(f"   GRIND-WEAK: fielding slot {wk} (L{levels[wk]}) as lead to train it")
                    self._swap_party_slots(0, wk)
                lv_before, pid_before = self.b.rd8(ram.GPLAYER_PARTY + 0x54), _pid0()
                r = self.grind(target, fragile=True, budget_s=GRIND_WEAK_PROBE_S)  # weak mon can faint ->
                #   reachable grass only; SHORT probe so a hopeless mon stalls out fast (not an 8-min sink)
                if r == "battle_loss":
                    self._restore_ace()
                    return "battle_loss"
                if r != "ok":                                # no grass reachable etc. — surface, restore, retry next tick
                    log(f"   GRIND-WEAK: grind returned {r!r} — restoring ace, retry next tick")
                    self._restore_ace()
                    return r                                 # pass the DISTINCT sentinel through (the
                    #                                          caller's prep stand-down counts dry runs;
                    #                                          masking it as 'ok' hid the failure)
                if self.b.rd8(ram.GPLAYER_PARTY + 0x54) <= lv_before:   # STALL: grind gained NO levels
                    stalled.add(pid_before)
                    log(f"   GRIND-WEAK: fielded mon made NO progress (still L{lv_before}) — marking it "
                        f"un-grindable on this route (stalled) so the pass never loops on it")
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

    def _road_bench_xp_arm(self, pick, state):
        """PASS-3 team-depth NEW#1 — organic bench XP on the road. Called right before a FORWARD-MARCH
        action (head_to_gym / travel:) executes. When a bench member is under its milestone prep target,
        lead with that weak mon so it PARTICIPATES in whatever road trainer/wild battles this leg triggers
        and banks a share of XP — the proven participation-XP switch (battle_agent.PROTECT_LEAD_GRIND,
        the same one grind_weak_members uses) fields the ACE on turn 1 so the weak mon never takes a hit.
        Returns True iff it armed (the caller disarms via _road_bench_xp_disarm the instant the leg ends,
        so the weak lead NEVER persists into the ctx/readiness/heal reads that assume slot-0 is the ace).
        Flag POKEMON_ROAD_BENCH_XP (default on). Fail-closed: any read error disarms and returns False.

        Scoped to forward-march picks only (never wander_catch/battle/beat_gym — a weak mon must never
        lead a gym leader or a catch attempt). Skips when hurt (a hurt party heals with the true ace up),
        thin (<2 mons), or the bench is already at milestone (_prep_team_target None). Picks the weakest
        LEVELABLE under-target member (mirrors grind_weak_members: not the ace, not stall-marked, not
        far-below box chaff), matching the milestone-cap prep so the two agree on 'the weak one'."""
        if not ROAD_BENCH_XP_ENABLED:
            return False
        if pick != "head_to_gym" and not str(pick).startswith("travel:"):
            return False
        # NAV-CRITICAL QUESTLINE GUARD (2026-07-11 NS#4): when a questline errand is active, head_to_gym
        # runs the ERRAND, not a plain march — and errands do nav-critical traversal (the Flash errand
        # crosses pitch-dark Diglett's Cave to the Route-2 aide; dungeon strikes cross gauntlet interiors).
        # A demoted ace there is a LIVELOCK: the weak lead can't clear an L29 cave Dugtrio with Cut, and
        # the in-battle switch-to-ace is unreliable (Tier-1 #5) → flee-loop that never traverses the cave
        # (observed: weak lead flee-looping Diglett's Cave (1,37), blocking the Route-2 crossing). Keep the
        # TRUE ace leading for errand ticks; the bench banks its XP on the many normal marching legs instead.
        if getattr(self, "_active_questline", None) is not None:
            return False
        try:
            if not (STRATEGIC_GRIND_ENABLED and battle_agent.GRIND_SWITCH_ENABLED):
                return False
            if self.needs_heal():
                return False                       # a hurt party heals with the true ace leading
            party = state.get("party") or []
            if len(party) < 2:
                return False
            # ACE-HP FLOOR: the switch costs the ace a free enemy hit per battle; don't start a bench-XP
            # leg with an already-dinged ace (a no-heal gauntlet could tip it over — the Route-13 hand-bank
            # observation). Defer to the heal path until the ace is comfortably topped up.
            _ace_m = max(party, key=lambda m: m.get("level", 0))
            _amx = _ace_m.get("maxhp") or 0
            if _amx and (_ace_m.get("hp", _amx) / _amx) < ROAD_XP_ACE_HP_FLOOR:
                return False
            prep_t = self._prep_team_target(state)
            if prep_t is None:
                return False                       # bench already at its milestone -> nothing to raise
            levels = self._party_levels()
            if not levels or len(levels) < 2:
                return False
            stalled = getattr(self, "_grind_stalled", None) or set()
            floor_min = prep_t - E4_PREP_BAND      # ignore far-below box-fodder chaff (Tier-1 #15)
            cand = []
            for s, l in enumerate(levels):
                if l >= prep_t or l < floor_min:   # crossed, or hopeless chaff -> not a lead candidate
                    continue
                try:
                    pid = self.b.rd32(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE)
                except Exception:
                    pid = None
                if pid is not None and pid in stalled:   # can't earn XP on this route -> skip
                    continue
                cand.append((l, s))
            if not cand:
                return False
            wk = min(cand)[1]                      # the weakest levelable under-target member
            ace = max(range(len(levels)), key=lambda s: levels[s])
            if wk == ace:
                return False                       # the weakest IS the ace -> nothing to protect
            if wk != 0:
                self._swap_party_slots(0, wk)      # weak mon leads -> "sent out" -> XP-eligible
            battle_agent.PROTECT_LEAD_GRIND = True
            log(f"   [roam] ROAD-BENCH-XP: leading with the weak L{levels[wk]} bench mon (target L{prep_t}) "
                f"so it banks participation XP on this leg — the ace fields turn 1 (fail-safe switch)")
            return True
        except Exception as e:
            log(f"   [roam] ROAD-BENCH-XP arm skipped ({e})")
            try:
                battle_agent.PROTECT_LEAD_GRIND = False
            except Exception:
                pass
            return False

    def _road_bench_xp_disarm(self):
        """Restore the ace to slot 0 and clear the participation toggle after a road-bench-XP leg, so the
        weak lead never persists into the next tick's readiness/heal/decision reads. Order-independent
        (_restore_ace re-reads levels), so it cleans up regardless of what the battle left the order as."""
        try:
            self._restore_ace()
        finally:
            battle_agent.PROTECT_LEAD_GRIND = False

    def _bench_severely_lopsided(self, state, prep_t):
        """PASS-3 NS#6 team-depth lever (a): return the gym milestone level IF the bench is SEVERELY
        lopsided vs BOTH the milestone AND the ace (the solo-carry + dead-weight-bench shape NS#5
        pinpointed — a lone L48 ace behind an L10-14 bench that the ace can always carry, so the oracle
        never STOPS to train the bench and it stays frozen), else None. Used to make the dedicated grind
        DOMINATE head_to_gym for ONE bounded +6 stint (symmetric to the TEAM-BRAIN PRE-BUILD dominance).

        Tightly gated so it can NEVER treadmill/park the road:
          • prep_t is not None — the +6 pin is armed, so the 'battle' executor levels the BENCH (not the
            ace via the grind(lead+2) fallback); the pin retires after +6 and won't re-arm until the next
            badge, so this is at most one +6 bite per milestone.
          • milestone not already in _lopsided_grind_done — a completed stint (incl. an un-levelable
            stall) releases it, so a bench that can't level here doesn't loop.
          • only with the participation switch available (the proven bench-leveling path).
          • a real bench (party >= 3), not hurt/critical, not mid nav-critical errand, grass not dry.
        The gap thresholds separate SEVERE (force a full stop) from MODEST (road-bench-XP handles it)."""
        if not LOPSIDED_GRIND_ENABLED or prep_t is None:
            return None
        if not (STRATEGIC_GRIND_ENABLED and battle_agent.GRIND_SWITCH_ENABLED):
            return None                              # need the proven participation switch to level a bench
        if getattr(self, "_active_questline", None) is not None:
            return None                              # errand ticks keep the TRUE ace leading (nav-critical)
        if getattr(self, "_prep_dry", 0) >= 2:
            return None                              # no reachable grass here — PREP STAND-DOWN owns it
        try:
            if self.needs_heal() or self._hurt_severity()[0] == "critical":
                return None
            party = state.get("party") or []
            if len(party) < 3:
                return None                          # a real bench, not just ace + 1
            levels = [m.get("level", 0) for m in party]
            floor, lead = min(levels), max(levels)
            _bc = int(state.get("badge_count", 0))
            try:
                # ensure_plan first: _next_milestone reads team_planner.state, which is None until the
                # plan is ensured — a raw call returns 0 (no static milestone) so the trigger silently
                # never fires (it's idempotent + already run all over assess()).
                self.team_planner.ensure_plan(party, _bc)
                milestone = self.team_planner._next_milestone(_bc, bool(state.get("post_game")))[1]
            except Exception:
                milestone = 0
            log(f"   [roam] LOPSIDED-DBG: floor={floor} lead={lead} ms={milestone} prep_t={prep_t} "
                f"ms_gap={milestone-floor} ace_gap={lead-floor} done={sorted(getattr(self,'_lopsided_grind_done',()))}")
            if not milestone:
                return None                          # no static milestone (post-game / planner off)
            if milestone in getattr(self, "_lopsided_grind_done", ()):
                return None                          # already forced this badge's stint (bounded)
            if (milestone - floor) < LOPSIDED_MS_GAP:
                return None                          # bench close enough — road-bench-XP finishes it
            if (lead - floor) < LOPSIDED_ACE_GAP:
                return None                          # not the solo-carry shape — no forced full stop
            return milestone
        except Exception as e:
            log(f"   [roam] lopsided-bench check skipped: {e}")
            return None

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

    def _mart_index(self):
        """TRUE BUY-list selection = selectedRow + scrollOffset (see MART_CURSOR/MART_SCROLL)."""
        return self.b.rd16(MART_CURSOR) + self.b.rd16(MART_SCROLL)

    def _mart_goto_row(self, row, tries=16):
        """Move the BUY-list selection to true index `row`, VERIFYING via the row+scroll readback
        each press so an eaten d-pad press can't leave us buying the wrong item. The row byte lags
        the scroll animation — arrival is re-verified after settling. Returns True on arrival."""
        for _ in range(tries):
            cur = self._mart_index()
            if cur == row:
                for _ in range(20):                        # settle: scroll animation lag
                    self.b.run_frame()
                if self._mart_index() == row:
                    return True
                continue
            self.b.press("DOWN" if cur < row else "UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
        return self._mart_index() == row

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
        c0 = self._mart_index()                            # CONFIRM the list: DOWN must move the selection
        self.b.press("DOWN", 8, 10, self.render, owner="agent")
        for _ in range(12):
            self.b.run_frame()
        if self._mart_index() != c0:
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
        """Batch-WORLD — maps off-limits to ROUTING. Two classes, unioned here so EVERY routing/
        reachability path inherits both from one source:
          (1) the active spatial wall (a route-trainer she can't beat yet). The HUB case (the wall is
              on the map she's standing on — e.g. Gary triggering on the Cerulean City map) is handled
              by blocked-DIRS instead, so she's never trapped on the very hub she heals at. Empty once
              she's grown enough to retry.
          (2) NS#43 — the story-gate maps (_story_gate_avoid: Route 12/16 pre-Flute). This bug class
              (a router computing avoid as wall-ONLY, then steering her onto Snorlax-gated Route 12)
              bit THREE times across three sites (off-road steer, keeper router, then the _grass_target
              'grass she KNOWS' east-steer). Folding the story-gate in at the shared source kills the
              whole class instead of patching sites one at a time. Post-Flute _story_gate_avoid is empty
              -> zero behavior change; pre-Flute nothing legitimate routes onto those dead-ends anyway
              (the road to Lavender is Rock Tunnel, not Route 12)."""
        gate = self._story_gate_avoid(state)
        r = self.strat.active_wall_rec()
        if not r or not r.get("map_id") or not r.get("is_trainer"):
            return set() | gate         # wild walls never gate ROUTING (see strat.is_gated)
        pc = state.get("party_count")
        pl = state["party"][0]["level"] if state.get("party") else None
        if self.strat.stronger_since_wall(pc, pl):
            return set() | gate
        wm, cur = tuple(r["map_id"]), tuple(state["map"])
        return ({wm} if wm != cur else set()) | gate

    # Snorlax-gated coastal roads: routable only once the Poké Flute (item 350) is in the bag.
    _FLUTE_GATED_MAPS = {(3, 30), (3, 34)}    # Route 12 (south of Lavender), Route 16 (west of Celadon)

    def _story_gate_avoid(self, state):
        """Story-gated maps to keep OUT of graph ROUTING until the key item is owned. The world graph is
        CAPABILITY-BLIND: while she detoured east for Flash she learned Route 11<->Route 12, so head_to_gym's
        warp-route now sends her Vermilion->Route 11->Route 12 toward Celadon — but Route 12 is Snorlax-
        blocked pre-Flute (the ONLY pre-Flute road to Lavender is ROCK TUNNEL — gamedata/frlg_gates.json
        roads['Celadon City']). She wedged at the Snorlax (13,70). Avoid the gated maps so the graph route
        falls through to the billed road (Rock Tunnel). Post-Flute (item 350) they clear and become roads."""
        try:
            if self._item_count(350) > 0:            # ITEM_POKE_FLUTE — Snorlax wakes, roads open
                return set()
        except Exception:
            pass
        return set(self._FLUTE_GATED_MAPS)

    def _keeper_hard_gate_avoid(self, state):
        """NS#43 — the KEEPER ROUTER (only) must not offer a keeper reachable only ACROSS a hard gate she
        hasn't opened. growlithe (Route 7 = 3,25 / Route 8 = 3,26) has TWO learned-graph approaches, each
        behind a DIFFERENT gate (same lesson as the Route-12 wedge: one entry path fixed reveals the next):
          • EAST — via Lavender Town (3,4): the header edge Route 10 (3,28) -> Lavender is a MAP CONNECTION
            but the real walk is pitch-black ROCK TUNNEL, un-navigable until HM05 FLASH is TAUGHT. Pre-Flash
            the offer fired, world.route hopped her east toward Lavender, and she PING-PONGED against the
            westward Flash errand -> STALL (ns43_shift1: 14 no-progress decisions at (3,28)@(0,10)).
          • WEST — via SAFFRON City (3,10): from Route 6 (3,24) world.route hops the gatehouse (18,0) toward
            Saffron, whose four guards are parched until the Tea (FLAG_GOT_TEA 678) from Celadon Mansion;
            the fetch errand walked INTO the gatehouse and wedged there on 'leave_building' (ns43_shift1b:
            7 late offers -> STALL at (18,0)@(4,9)).
        So avoid Lavender pre-Flash AND Saffron pre-Tea; the keeper is deemed unreachable until a gate opens
        (either one -> world.route uses that side -> offered again). Post-Flash the east opens; post-Tea the
        west opens. Fail-closed (block on read error): a keeper is never IN Lavender/Saffron, so the worst
        case is a benign missed post-gate fetch — never a wedge. Kept OUT of the shared _wall_avoid because
        BOTH are legitimate routing DESTINATIONS (head_to_gym pushes toward them to OPEN the gate); only the
        keeper DETOUR must decline them while shut."""
        gate = set()
        try:
            import field_moves as fm
            if st.party_knows_move(self.b, 148, self.b.rd8(ram.GPLAYER_PARTY_CNT)) is None:
                gate.add(LAVENDER)                   # HM05 Flash NOT taught -> Rock Tunnel dark (east shut)
            if not fm.read_flag(self.b, 678):        # FLAG_GOT_TEA NOT set -> Saffron guards block (west shut)
                gate.add(SAFFRON)
        except Exception:
            gate = {LAVENDER, SAFFRON}               # fail-closed: block both when the read fails
        return gate

    def _wall_blocked_dirs(self, state):
        """Batch-WORLD — when the active wall is ON her current map (a trainer guarding an exit, like
        Gary at Cerulean's north edge), the blocked thing is a DIRECTION, not a neighbouring map. Pick
        the exit nearest the wall's coords (from the live grid bounds), so the spatial brief can show
        'NORTH → BLOCKED' truthfully. Best-effort: empty if coords/bounds aren't readable (the brief
        just won't tag a direction; routing still steers her to known places, which are backward)."""
        r = self.strat.active_wall_rec()
        if not r or not r.get("map_id") or not r.get("coords") or not r.get("is_trainer"):
            return set()                # wild walls never gate ROUTING (see strat.is_gated)
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
        self._ql_prev_map = None              # anti-ping-pong for visited-fallback bend hops
        self._ql_room_sweeps = 0              # fresh room-sweep budget for the next questline
        self._ql_talks_map = {}               # fresh per-map talk budgets too
        self._ql_bg_done = set()              # fresh machine/sign tracking too
        self._ql_catch_voiced = False         # fresh dex-catch narration for the next questline

    # ── THE FLASH ERRAND (2026-07-09 night-train shift 1) ──────────────────────────────────────────
    # Ported from the PROVEN recon_hm05 strike (which cleared this exact stretch on the Champion climb)
    # + catch-en-route. The badge-4 wall: the Rock-Tunnel road is Flash-gated and the Route 2 aide won't
    # hand HM05 below 10 OWNED species. The catch-in-place driver species-EXHAUSTS on Route 10 (its grass
    # is all dupes: ekans/spearow/voltorb) — so DRIVE the real errand EAST: walk the billed back-legs to
    # Vermilion, east to Route 11, cross DIGLETT'S CAVE (Diglett/Dugtrio + Route 11 Drowzee = the fresh
    # species Route 10 lacks) to the Route 2 aide's gatehouse, talk until HM05, then TEACH Flash. Catching
    # en route clears the dex gate as a SIDE EFFECT of reaching the aide — one coherent errand, not two.
    _FLASH_BACK_LEGS = {(3, 28): ("west", (3, 27)), (3, 27): ("west", (3, 3)),
                        (3, 3): ("south", (3, 23)), (3, 23): ("pass", (3, 24)),
                        (3, 24): ("south", (3, 5)), (3, 5): ("east", (3, 29)),
                        (3, 22): ("east", (3, 3))}   # Route 4 joins at Cerulean
    # GRASSY legs the errand walks THROUGH that hold species she lacks — a real player fills the dex
    # on the routes they're already crossing (2026-07-09 shift 2). Route 6 (Oddish/Bellsprout/Pidgey/
    # Meowth = +3-4) supplies the headroom the Route-11-only sweep never could (ekans/spearow/drowzee
    # exhaust it at ~7) — and its south-to-Vermilion leg walks clean. Route 9/10 are DELIBERATELY
    # EXCLUDED: Route 10's grass is sparse (Voltorb only, +1) AND its west exit is NPC-pinched at (7,6),
    # so catching there strands her mid-grass in a repath/encounter oscillation (shift-2 legcatch2). She
    # still catches on Route 11 / Route 2 via the dedicated PHASE 2/4 sweeps. Route 6 does the heavy lift.
    _FLASH_CATCH_LEGS = {(3, 24): "Route 6"}

    def _flash_errand(self):
        """Advance the Flash errand one phase (free_roam re-enters between calls; internal budgets like
        head_to_gym's road-follow). Returns 'flash_done' | 'flash_progress' | 'flash_stuck'."""
        import field_moves as fm
        import hm_teach as ht
        FLAG_GOT_HM05, FLASH_MOVE = 0x23B, 148
        ROUTE11, ROUTE2 = (3, 29), (3, 20)
        b = self.b
        pc = lambda: b.rd8(ram.GPLAYER_PARTY_CNT)
        # DEX-GATE CATCH-ALL: while the OWNED>=10 gate is unmet, a NEW species must be caught even when
        # the party is full — it boxes, the dex still ticks (a real player boxes the new mon for the dex).
        # roster_judgment normally skips at party-6 (room=False); this flag lets the catch-judgment override
        # that for first-of-a-kind wilds ONLY. Cleared the instant Flash is known / dex hits 10.
        self._dex_catch_all = (st.party_knows_move(b, FLASH_MOVE, pc()) is None
                               and ram.pokedex_owned_count(b) < 10)
        if st.party_knows_move(b, FLASH_MOVE, pc()) is not None:
            self._dex_catch_all = False
            return "flash_done"

        def _catch_to_10(where, tries=6):
            # EARLY-BAIL on a dupe-heavy map (2026-07-09): catch_one wanders until it catches SOMETHING
            # or times out — on a near-exhausted map every encounter is a dupe (Route 11 = ekans/spearow)
            # so it burns its whole budget catching nothing new. Track dex per pass, use a SHORT per-catch
            # budget, and move on after 2 no-gain passes — the cave/next map supplies the fresh species.
            # Returns WHY it stopped so the caller only blacklists a leg on genuine dupe-exhaustion
            # (2026-07-11 NS#43): 'no_balls'/'no_grass' are TRANSIENT (restock / move on) and must NOT
            # mark the leg done — a 0-ball arrival on Route 6 was wrongly blacklisting the +4 dex route.
            misses = 0
            for _ in range(tries):
                have0 = ram.pokedex_owned_count(b)
                if have0 >= 10:
                    return "dex10"
                cr = self.catch_one(max_seconds=90)
                have1 = ram.pokedex_owned_count(b)
                log(f"   [flash-errand] catch on {where}: dex={have1}/10 -> {cr}")
                if cr == "no_balls":
                    return "no_balls"
                if cr in ("no_grass", "no_reachable_target"):
                    return "no_grass"
                if have1 > have0:
                    misses = 0
                else:
                    misses += 1
                    if misses >= 2:
                        log(f"   [flash-errand] {where}: fresh species exhausted this pass — moving on")
                        return "exhausted"
            return "progress"

        cur = tuple(tv.map_id(b))
        # DEX-PROGRESS GUARD (rule 18 — no freeze-spin): reachable grass (Route 11 + Route 2) caps the
        # OWNED count at ~8 — the missing species live in Diglett's Cave (no grass, catch_one can't reach
        # them) or on off-errand routes. Without this the Route-2 catch phase spins forever (verified: 182
        # no-gain ticks). Correct-by-construction top-of-call counter: on a CATCH-PHASE map (Route 11 /
        # Route 2 / inside the cave) with no dex gain since the high-water mark, increment; reset on ANY
        # gain. After DEX_STALL_CAP no-gain catch passes, SURFACE the blocker (release to roam) — never
        # loops. Leg-walk phases (PHASE 1) don't count (dex legitimately holds while she walks). Once she
        # OWNS 10 the errand is pure routing/teaching and is never "blocked".
        DEX_STALL_CAP = 4
        _dex = ram.pokedex_owned_count(b)
        # ROUTE 11 is NOT a stall map (2026-07-09 shift 2): from Route 11 the answer to "no fresh species
        # here" is ALWAYS cross Diglett's Cave to Route 2 (an un-swept catch source), NEVER give up. Counting
        # Route 11 tripped DEX-BLOCKED at dex 9 the moment she had balls to over-hunt it, short-circuiting the
        # cave cross below. Only Route 2 (the terminal catch strip) + inside the cave count toward the stall.
        _catch_phase = (cur == ROUTE2 or cur[0] != 3)
        if _dex > getattr(self, "_flash_dex_hi", -1):
            self._flash_dex_hi, self._flash_nogain = _dex, 0
        elif _catch_phase:
            self._flash_nogain = getattr(self, "_flash_nogain", 0) + 1
        if _dex < 10 and _catch_phase and getattr(self, "_flash_nogain", 0) >= DEX_STALL_CAP:
            log(f"   [flash-errand] !! DEX-BLOCKED at {_dex}/10 — reachable grass exhausted after "
                f"{self._flash_nogain} no-gain catch passes (missing species need Diglett's Cave floor-"
                f"catching or an off-errand route sweep, e.g. Route 24/25). Surfacing, not spinning.")
            return "flash_stuck"

        # PHASE 3 — inside Diglett's Cave (non-overworld): keep crossing toward Route 2.
        if cur[0] != 3:
            _catch_to_10("Diglett's Cave", tries=3)
            ok = self._cross_cave(None, ROUTE2)
            return "flash_progress" if ok else "flash_stuck"
        # PHASE 2 — on Route 11: catch fresh species (Drowzee), then cross Diglett's Cave (north mouth ->
        # Route 2). Exhausted-memo: once Route 11 yields no new species, skip the re-catch (else with a full
        # ball pocket catch_one wanders the whole route 90s/pass hunting a non-dupe that isn't there) and go
        # straight for the cave — Route 2's Caterpie/Weedle supply the last dex slot.
        if cur == ROUTE11:
            _legdone = self.__dict__.setdefault("_flash_leg_done", set())
            if ram.pokedex_owned_count(b) < 10 and ROUTE11 not in _legdone:
                if _catch_to_10("Route 11", tries=3) == "exhausted":
                    _legdone.add(ROUTE11)
            ok = self._cross_cave("north", ROUTE2)
            return "flash_progress" if ok else "flash_stuck"
        # PHASE 4 — on Route 2 (aide's strip): top up dex if short, reach the gatehouse, talk, teach.
        if cur == ROUTE2:
            if ram.pokedex_owned_count(b) < 10:
                _catch_to_10("Route 2")
                if ram.pokedex_owned_count(b) < 10:
                    log(f"   [flash-errand] still dex {ram.pokedex_owned_count(b)}/10 on Route 2")
                    return "flash_progress"
            if not self._flash_gatehouse(FLAG_GOT_HM05):
                return "flash_stuck"
            if st.party_knows_move(b, FLASH_MOVE, pc()) is None:
                plan = ht.default_plan(b, "flash", pc())
                if plan is not None:
                    slot, forget_idx, reason = plan
                    log(f"   [flash-errand] TEACH flash -> slot {slot} ({reason})")
                    ht.TeachFlow(self, log=lambda m: log(m), on_event=self.on_event).teach(
                        "flash", slot, forget_idx)
            if st.party_knows_move(b, FLASH_MOVE, pc()) is not None:
                # RETURN TO THE CELADON ROAD-HEAD (2026-07-09 shift 2): the aide strip (Route 2) is a
                # south-west DEAD POCKET — from here forward-drive ping-pongs Viridian<->Route 2 to a STALL
                # (the billed Celadon road starts at VERMILION, and the only link back is EAST through
                # Diglett's Cave, a warp-maze the world-graph won't route across). So the instant Flash is
                # learnt, steer her back ONCE: re-cross the cave east to Route 11 -> Vermilion, where
                # head_to_gym's billed Celadon road (Route 6 -> Underground Path) picks up cleanly.
                if not getattr(self, "_flash_returned", False):
                    self._flash_returned = True
                    log("   [flash-errand] Flash learnt — returning EAST across Diglett's Cave to the "
                        "Vermilion road-head (Route 2 is a SW dead-pocket)")
                    try:
                        # BUGFIX (2026-07-09 shift 3): the shift-2 return called _cross_cave(None,..)
                        # which means "already inside — skip the entry warp". But the teach leaves her
                        # INSIDE the aide GATEHOUSE (a group!=3 interior); _cross_cave then treated the
                        # gatehouse as the cave, walked her out its door to a Route-2 pocket, and returned
                        # False (never entering Diglett's Cave) -> _edge_travel skipped -> Route2<->Viridian
                        # ping-pong STALL. FIX: (1) exit to the Route 2 overworld first, (2) explicitly
                        # ENTER Diglett's Cave via its Route-2 mouth (17,11)->(1,36), (3) cross east to
                        # Route 11, (4) edge-travel to Vermilion. The forward cross EXITS the cave at
                        # (17,12) — right beside (17,11) — and reaches the aide from there, so the aide
                        # pocket reaches the mouth.
                        # The Diglett's Cave Route-2 mouth (17,11)->(1,36); its south-approach /
                        # cave-EXIT tile is (17,12). A CUT tree sits between the aide-door pocket
                        # and the mouth pocket, and it REGROWS on the gatehouse map-reload — so
                        # enter_warp's plain-BFS reachability pre-check (tree=wall) would skip the
                        # mouth. travel()'s obstacle handler CUTS the regrown tree (she owns Cut),
                        # so route to the cave-exit tile FIRST (crosses into the mouth pocket), THEN
                        # enter_warp fires trivially from right beside it.
                        DIGLETT_R2_MOUTH, CAVE_EXIT_TILE = (17, 11), (17, 12)
                        AIDE_SOUTH_DOOR = (18, 46)
                        # REACH THE CAVE-MOUTH (NORTH) POCKET (2026-07-09 shift 3, geometry-verified via
                        # recon_gate_exit): the aide gate (15,2) is a PASS-THROUGH. She enters from the
                        # SOUTH door (18,46) via a ONE-WAY LEDGE and the teach leaves her inside; its south
                        # interior door (7,10) drops back onto (18,47) [SOUTH pocket — walled off from the
                        # cave mouth by that ledge, un-cuttable], its NORTH door (7,1) exits to (18,41)
                        # [NORTH pocket, which reaches the mouth]. So exit NORTH; if she's already been
                        # dumped into the south pocket, re-enter the gate and exit north. Only THEN is the
                        # cave mouth walk-reachable.
                        def _north_pocket():
                            try:
                                g = tv.Grid(b)
                                return (tuple(tv.map_id(b)) == ROUTE2 and bool(tv.bfs(
                                    g, tuple(tv.coords(b)), lambda t: t == CAVE_EXIT_TILE,
                                    walkable=g.walkable)))
                            except Exception:
                                return False
                        for _ in range(4):
                            if _north_pocket():
                                break
                            if tv.map_id(b)[0] != 3:                # inside the gate -> exit north door
                                self.enter_warp(prefer="north")
                            else:                                  # on Route 2 (south pocket) -> re-enter
                                self.enter_warp(pick=AIDE_SOUTH_DOOR)
                        cur_r = tuple(tv.map_id(b))
                        log(f"   [flash-errand] return: on {cur_r}@{tuple(tv.coords(b))} "
                            f"(north_pocket={_north_pocket()}); routing to the Diglett's Cave mouth "
                            f"{DIGLETT_R2_MOUTH} (cutting the regrown tree)")
                        if cur_r == ROUTE2 and _north_pocket():
                            # travel to the cave-exit tile first — travel's obstacle handler CUTS the
                            # regrown tree (18,26) between the north-door landing and the mouth; enter_warp's
                            # plain-BFS pre-check treats a regrown tree as a wall and would skip the mouth.
                            self.trav.travel(target_map=None, arrive_coord=CAVE_EXIT_TILE,
                                             max_steps=400)
                            r = self.enter_warp(pick=DIGLETT_R2_MOUTH)
                            log(f"   [flash-errand] return: cave-mouth enter -> {r} "
                                f"(now on {tuple(tv.map_id(b))})")
                            if r == "warped" and self._cross_cave(None, ROUTE11):
                                self._edge_travel(VERMILION, "west")
                            else:
                                log("   [flash-errand] return: cave re-entry failed — releasing to "
                                    "roam (head_to_gym owns the road retry) (LOUD)")
                        else:
                            log(f"   [flash-errand] return: couldn't reach the cave-mouth pocket "
                                f"(on {cur_r}) — releasing to roam (LOUD)")
                    except Exception as _re:
                        log(f"   [flash-errand] road-head return skipped ({_re}) (LOUD)")
                return "flash_done"
            return "flash_progress"
        # PHASE 1 — walk the billed back-legs home + east to Route 11.
        if cur in self._FLASH_BACK_LEGS:
            # OPPORTUNISTIC DEX-FILL (2026-07-09 shift 2): before walking onward, sweep THIS leg's grass
            # for fresh species if she's still short. Reaching OWNED>=10 here (Route 6 has 3-4 she lacks)
            # means the errand never gets marooned on Route-11's exhausted ekans/spearow/drowzee.
            # EXHAUSTED-MEMO: a leg is re-entered every roam tick; without a cross-tick memo she re-catches
            # an already-drained route forever (the Route-10 healed_retry trap) and never walks DOWN to
            # Route 6. So: catch here once; if a pass yields NO new species, mark the leg done and never
            # catch it again — just walk through. Route 6 (the +4 route) is where she clears 10.
            _legdone = self.__dict__.setdefault("_flash_leg_done", set())
            if (ram.pokedex_owned_count(b) < 10 and cur in self._FLASH_CATCH_LEGS
                    and cur not in _legdone):
                if _catch_to_10(self._FLASH_CATCH_LEGS[cur], tries=3) == "exhausted":
                    _legdone.add(cur)
                    log(f"   [flash-errand] {self._FLASH_CATCH_LEGS[cur]} yielded no new species "
                        f"— marking leg catch-exhausted (walk through from here)")
            # BALL RESTOCK (2026-07-09 shift 2): the dex sweep burns balls fast (3-4 catches on Route 6) —
            # she ran DRY at dex 9 on Route 11 and couldn't take the last species. Vermilion (a leg she
            # crosses) sells Poké Balls (MART_STOCK row 0); top up here whenever the dex gate is still open
            # and the pocket's thin, so she reaches Route 11 / Route 2 with ammo for the final catch.
            # (2026-07-11 NS#43) restock at ANY mart town on the back-legs, not just Vermilion — Cerulean
            # (3,3) comes BEFORE the Route 6 catch leg, so topping balls there means she reaches the +4
            # dex route with ammo (she was arriving at Route 6 with 0 balls, wasting the best catch route).
            _FLASH_MART_LEGS = {CERULEAN: CERULEAN_MART_DOOR, VERMILION: VERMILION_MART_DOOR}
            if (cur in _FLASH_MART_LEGS and ram.pokedex_owned_count(b) < 10
                    and self._ball_count() < 6 and self.money() > SHOP_MONEY_FLOOR):
                want = 12 - self._ball_count()
                log(f"   [flash-errand] dex {ram.pokedex_owned_count(b)}/10 + {self._ball_count()} balls "
                    f"— restocking {want} Poké Ball(s) at {self.world.name(cur)} Mart for the catch")
                try:
                    self.buy_at_mart(_FLASH_MART_LEGS[cur], [(ITEM_POKE_BALL, want)])
                except Exception as _be:
                    log(f"   [flash-errand] ball restock skipped ({_be}) (LOUD)")
            go, nxt = self._FLASH_BACK_LEGS[cur]
            log(f"   [flash-errand] leg {cur} -{go}-> {nxt}")
            if go == "pass":
                r = self._door_passthrough(want_map=nxt)
                if r == "need_heal":
                    self.heal_nearest()
            else:
                r = self._edge_travel(nxt, go)
                if r == "need_heal":
                    self.heal_nearest()
            return "flash_progress"
        # off the billed road — release to roam so forward-drive steers her back toward the road.
        log(f"   [flash-errand] off the billed errand road at {cur} — releasing to roam")
        return "flash_stuck"

    def _cross_cave(self, into_prefer, out_map, budget_s=420):
        """Cross a warp-chain cave (Diglett's) by DESTINATION truth — ported from recon_hm05. into_prefer
        None = already inside (skip the entry warp). Each room: prefer an overworld-dest warp that isn't
        where we came in from (the far door), else the farthest unused warp (visited-memory kills the
        east<->west ping-pong). True once on overworld == out_map (or any group-3 if out_map is None)."""
        b = self.b
        m0 = tuple(tv.map_id(b))
        if into_prefer is not None:
            self.enter_warp(prefer=into_prefer)
            if tuple(tv.map_id(b)) == m0:
                log(f"   [flash-errand] cave: entry warp ({into_prefer}) didn't fire at {m0}")
                return False
        t = time.time()
        visited = set()
        while tv.map_id(b)[0] != 3:
            if time.time() - t > budget_s:
                log(f"   [flash-errand] cave crossing TIMEOUT at {tv.map_id(b)}")
                return False
            pos = tuple(tv.coords(b))
            mid = tuple(tv.map_id(b))
            warps = [(tuple(wxy), tuple(d)) for (wxy, d, _i) in tv.read_warps(b)]
            if not warps:
                log(f"   [flash-errand] cave room {mid} shows no warps — stuck")
                return False
            visited.add((mid, pos))
            dist = lambda w_: abs(w_[0] - pos[0]) + abs(w_[1] - pos[1])
            if out_map is not None:
                # TARGET-DIRECTED (2026-07-09 shift 3): exit ONLY via a warp whose dest IS out_map;
                # every other overworld warp is a WRONG exit. The old rule ('nearest overworld dest
                # != m0') broke the RETURN cross — she enters at cave room (1,36) whose (4,6)->Route2
                # warp it greedily took, walking her straight back out to the aide pocket. When no
                # target-exit is in this room, progress DEEPER via the farthest unused CAVE warp
                # (never an overworld one), visited-memory killing any two-room bounce.
                exits = [w_ for (w_, d) in warps if d == out_map]
                if exits:
                    far = min(exits, key=dist)
                else:
                    caves = [w_ for (w_, d) in warps if d[0] != 3]
                    fresh = [w_ for w_ in caves if (mid, w_) not in visited and w_ != pos]
                    cands = (fresh or [w_ for w_ in caves if w_ != pos]
                             or [w_ for (w_, d) in warps if w_ != pos])
                    far = max(cands, key=dist)
            else:
                outs = [w_ for (w_, d) in warps if d[0] == 3 and d != m0]
                if outs:
                    far = min(outs, key=dist)
                else:
                    fresh = [w_ for (w_, d) in warps if (mid, w_) not in visited and w_ != pos]
                    cands = fresh or [w_ for (w_, d) in warps if w_ != pos]
                    far = max(cands, key=dist)
            visited.add((mid, far))
            before = mid
            self.trav.travel(target_map=None, arrive_coord=far, max_steps=400)
            if tuple(tv.map_id(b)) == before:
                self.enter_warp(pick=far)
            if tuple(tv.map_id(b)) == before and tuple(tv.coords(b)) == far:
                for d_ in ("DOWN", "UP", "LEFT", "RIGHT"):     # walk-through mats fire on the crossing step
                    b.press(d_, 10, 6, lambda: None, owner="agent")
                    for _f in range(40):
                        b.run_frame()
                    if tuple(tv.map_id(b)) != before:
                        break
                    if tuple(tv.coords(b)) != far:
                        self.trav.travel(target_map=None, arrive_coord=far, max_steps=20)
            if tuple(tv.map_id(b)) == before:
                log(f"   [flash-errand] cave: warp {far} didn't fire")
                return False
        return out_map is None or tuple(tv.map_id(b)) == out_map

    def _cross_warp_maze(self, m0, budget_s=900):
        """Cross a PARTITIONED dark warp-maze (Rock Tunnel class: both mouths dest the SAME overworld
        map, floors split into sections only linked by interior ladders). Ported verbatim from the
        proven recon_rocktunnel strike (2026-07-07). Stronger than _cross_cave (the Diglett crosser):
        reachability is a SECTION-relative grid BFS (a global 'unreachable' mark killed run-3 — the
        south door, dead from the entry section, was never retried from the section that reaches it);
        `rode` guards cycles; a dead-end section backtracks by re-riding the arrival warp; walking to a
        warp AVOIDS the other warp tiles (the walk-through-fires trap). True once out on the overworld
        (dest != m0 preferred — pass m0=(3,255) so BOTH real Route-10 mouths count as exits)."""
        b = self.b
        t0 = time.time()
        rode = set()
        while tuple(tv.map_id(b))[0] != 3:
            if time.time() - t0 > budget_s:
                log(f"   [tunnel] maze crossing TIMEOUT at {tv.map_id(b)}"); return False
            pos = tuple(tv.coords(b))
            mid = tuple(tv.map_id(b))
            warps = [(tuple(wxy), tuple(d)) for (wxy, d, _i) in tv.read_warps(b)]
            if not warps:
                log(f"   [tunnel] room {mid} shows no warps — stuck"); return False
            g = tv.Grid(b)
            others = [w for (w, d) in warps if w != pos]

            def _reachable(w_):
                return bool(tv.bfs(g, pos, lambda t, ww=w_: t == ww,
                                   walkable=lambda sx, sy: g.walkable(sx, sy)
                                   and ((sx, sy) == w_ or (sx, sy) not in others or (sx, sy) == pos)))

            reach = [w for w in others if _reachable(w)]
            dist = lambda w_: abs(w_[0] - pos[0]) + abs(w_[1] - pos[1])
            exits = [w for (w, d) in warps if d[0] == 3 and d != m0
                     and w in reach and (mid, w) not in rode]
            if not exits:
                exits = [w for (w, d) in warps if d[0] == 3 and w in reach
                         and (mid, w) not in rode and w != pos]
            backtrack = False
            if exits:
                far = min(exits, key=dist)
                log(f"   [tunnel] room {mid}: at {pos}, EXIT door {far} reachable (of {warps})")
            else:
                fresh = [w for w in reach if (mid, w) not in rode]
                if fresh:
                    far = max(fresh, key=dist)
                    log(f"   [tunnel] room {mid}: at {pos}, riding fresh warp {far} (reach={reach})")
                elif any(w == pos for (w, d) in warps):
                    far = pos; backtrack = True
                    log(f"   [tunnel] room {mid}: section exhausted — BACKTRACK via arrival warp {pos}")
                else:
                    log(f"   [tunnel] room {mid}: exhausted, no arrival warp to backtrack on"); return False
            if not backtrack:
                rode.add((mid, far))
            before = mid
            if backtrack:
                for d_ in ("UP", "DOWN", "LEFT", "RIGHT"):     # step off then back on to re-fire the ladder
                    b.press(d_, 10, 6, lambda: None, owner="agent")
                    for _f in range(30):
                        b.run_frame()
                    if tuple(tv.coords(b)) != pos:
                        break
                self.trav.travel(target_map=None, arrive_coord=pos, max_steps=30)
                if tuple(tv.map_id(b)) == before:
                    self.enter_warp(pick=pos)
            else:
                self.trav.travel(target_map=None, arrive_coord=far, max_steps=600, max_seconds=240,
                                 avoid=[w for w in others if w != far])
                if tuple(tv.map_id(b)) == before:
                    self.enter_warp(pick=far)
            if tuple(tv.map_id(b)) == before and tuple(tv.coords(b)) == far:
                for d_ in ("DOWN", "UP", "LEFT", "RIGHT"):     # door mats fire on the crossing step
                    b.press(d_, 10, 6, lambda: None, owner="agent")
                    for _f in range(40):
                        b.run_frame()
                    if tuple(tv.map_id(b)) != before:
                        break
                    if tuple(tv.coords(b)) != far:
                        self.trav.travel(target_map=None, arrive_coord=far, max_steps=20)
            if tuple(tv.map_id(b)) == before and not backtrack:
                log(f"   [tunnel] room {before}: warp {far} didn't fire despite reachability — next")
        return True

    def _cross_tunnel_leg(self, cave_map, out_map, out_dir):
        """Cross a DARK warp-maze cave that a billed 'pass' leg runs THROUGH (Rock Tunnel: both mouths
        dest the SAME overworld map, floors partitioned). This is the leg head_to_gym previously handed
        to _door_passthrough (the HUT class) — which never crosses a cave, so she edge-oscillated Route
        9<->Route 10 and drifted back to the dex grounds (shift-4 frontier). Recipe (proven,
        recon_rocktunnel): walk to the cave-mouth warp reachable from her feet -> enter -> USE FLASH
        (verify; NEVER walk dark) -> _cross_warp_maze to a far mouth -> edge-travel `out_dir` into
        out_map. FLEE wilds / FIGHT trainers, heal suppressed (no PC in a cave). Returns
        'arrived' | 'road_passthrough' | 'need_flash' | None (couldn't start -> caller falls through)."""
        import field_moves as fm
        import hm_teach as ht
        b = self.b
        FLASH_MOVE = 148
        cur0 = tuple(tv.map_id(b))
        # FLASH must be taught before the dark cave (the upstream gate check normally guarantees this;
        # double-guard so we NEVER walk the tunnel dark).
        if st.party_knows_move(b, FLASH_MOVE, b.rd8(ram.GPLAYER_PARTY_CNT)) is None:
            log("   [tunnel] no party mon knows FLASH — cannot cross the dark cave; deferring")
            return "need_flash"
        try:
            warps = [(tuple(wxy), tuple(d)) for (wxy, d, _i) in tv.read_warps(b)]
        except Exception:
            warps = []
        mouths = [w for (w, d) in warps if d == tuple(cave_map)]
        if not mouths:
            log(f"   [tunnel] no warp on {cur0} leads into cave {cave_map} — deferring")
            return None
        # pick the mouth her feet can reach, nearest first (she arrives on the entrance section; the
        # far mouth is partitioned off on the overworld, so BFS reachability selects the right one).
        pos = tuple(tv.coords(b))
        try:
            grid = tv.Grid(b)
            reach = [w for w in mouths if w == pos
                     or tv.bfs(grid, pos, lambda t, ww=w: t == ww, walkable=grid.walkable)]
        except Exception:
            reach = mouths
        mouth = min(reach or mouths, key=lambda w: abs(w[0] - pos[0]) + abs(w[1] - pos[1]))
        self.trav.travel(target_map=None, arrive_coord=(mouth[0], mouth[1] + 1),
                         max_steps=400, max_seconds=180)
        self.enter_warp(pick=mouth)
        if tuple(tv.map_id(b))[0] == 3:
            log(f"   [tunnel] mouth {mouth} didn't fire (still on {tv.map_id(b)}) — deferring")
            return None

        def _lit():
            return bool(fm.read_flag(b, fm.FLAG_SYS_FLASH_ACTIVE))

        if not _lit():
            self.on_event("Rock Tunnel — pitch dark. good thing I carry Flash now.", kind="route", tier=2)
            slot = st.party_knows_move(b, FLASH_MOVE, b.rd8(ram.GPLAYER_PARTY_CNT))
            r = ht.TeachFlow(self, log=lambda m: log(m), on_event=self.on_event).use_field_move(
                slot, verify=_lit, label="flash")
            log(f"   [tunnel] use flash -> {r} (lit={_lit()})")
        if not _lit():
            log("   [tunnel] FLASH did not light the cave — refusing to walk dark; deferring")
            return None
        self.on_event("and there's light. okay, Rock Tunnel — let's do this properly.", kind="route", tier=2)
        saved_runner, saved_heal = self.trav.battle_runner, self._suppress_heal
        self.trav.battle_runner = self._cave_runner       # FLEE wilds / FIGHT trainers
        self._suppress_heal = True                         # no PC in a cave -> survive-or-blackout
        try:
            ok = self._cross_warp_maze((3, 255))           # sentinel: BOTH real mouths count as exits
        finally:
            self.trav.battle_runner, self._suppress_heal = saved_runner, saved_heal
        if not ok:
            log(f"   [tunnel] maze crossing failed at {tv.map_id(b)}")
            return None
        log(f"   [tunnel] OUT of the tunnel at {tv.map_id(b)} coords={tv.coords(b)}")
        # emerged on the far side of the re-emergence overworld map (Rock Tunnel -> Route 10 south);
        # the billed leg's edge (south) carries her into out_map (Lavender).
        if out_dir and out_map and tuple(tv.map_id(b)) != tuple(out_map):
            self._edge_travel(tuple(out_map), out_dir)
        if out_map and tuple(tv.map_id(b)) == tuple(out_map):
            self.on_event("Lavender Town — the little town with the tower. we made it through the dark.",
                          kind="route", tier=2)
            return "arrived"
        return "road_passthrough"

    def _flash_gatehouse(self, flag_hm05):
        """Reach the Route 2 aide's gatehouse (a walk-through GATE band on the building's north face, NOT
        a door) and talk until HM05 — ported from recon_hm05. True if the flag sets."""
        import field_moves as fm
        b = self.b
        tried_g = set()
        for _attempt in range(4):
            if fm.read_flag(b, flag_hm05):
                return True
            if tv.map_id(b)[0] == 3:
                pos = tuple(tv.coords(b))
                gws = [(tuple(w), tuple(d)) for (w, d, _i) in tv.read_warps(b)
                       if d[0] != 3 and tuple(w) != pos and tuple(w) not in tried_g]
                if not gws:
                    log("   [flash-errand] gatehouse: no untried interior warp — LOUD")
                    break
                south = [w for (w, d) in gws if w[1] >= pos[1]] or [w for (w, d) in gws]
                tgt = min(south, key=lambda w: abs(w[0] - pos[0]) + abs(w[1] - pos[1]))
                tried_g.add(tgt)
                log(f"   [flash-errand] gatehouse: heading to warp {tgt}")
                self.trav.travel(target_map=None, arrive_coord=tgt, max_steps=300, max_seconds=180)
                if tv.map_id(b)[0] == 3 and tuple(tv.coords(b)) == tgt:
                    for d_ in ("DOWN", "UP", "LEFT", "RIGHT"):
                        b.press(d_, 10, 6, lambda: None, owner="agent")
                        for _f in range(40):
                            b.run_frame()
                        if tv.map_id(b)[0] != 3:
                            break
                if tv.map_id(b)[0] == 3:
                    self.enter_warp(pick=tgt)
                if tv.map_id(b)[0] == 3:
                    continue
            talks = 0
            while not fm.read_flag(b, flag_hm05) and talks < 8:
                self.talk_npc()
                talks += 1
            if fm.read_flag(b, flag_hm05):
                return True
            self._exit_to_overworld()
        return bool(fm.read_flag(b, flag_hm05))

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
        # NO-PROGRESS ABANDON (2026-07-06, the surf-misdirection spin): an errand that leaves her
        # standing on the SAME tile call after call is the WRONG errand (or an unroutable one) —
        # drop it LOUD and let the roam re-recognize fresh (the probe may arm a better gate).
        _fp = (self._active_questline.gate.missing, tuple(tv.map_id(self.b)),
               tuple(tv.coords(self.b) or ()))
        if _fp == getattr(self, "_ql_fp", None):
            self._ql_fp_n = getattr(self, "_ql_fp_n", 0) + 1
            if self._ql_fp_n >= 5:
                log(f"   [roam] !! QUESTLINE NO-PROGRESS ×{self._ql_fp_n} ({_fp[0]}) — abandoning "
                    f"the errand; re-recognizing fresh")
                self.on_event("hm — this plan's going nowhere from here. let me rethink.",
                              kind="route", tier=1)
                self._clear_questline("no-progress abandon")
                self._ql_fp, self._ql_fp_n = None, 0
                return "questline_abandoned"
        else:
            self._ql_fp, self._ql_fp_n = _fp, 0
        step = q.actionable
        if step is None or not step.resolved:
            # KB can't resolve this gate → the GuideSearch fallback (Phase 4) fills it in; until then surface
            # + release to normal roam so she never spins on an underivable gate.
            log(f"   [roam] !! QUESTLINE unresolved ({self._active_questline.gate.missing}) — needs the "
                f"guide (Phase 4)/a hand; releasing to normal roam")
            return "questline_unresolved"
        cur_map = tuple(state["map"])
        # BLOCKER STRIKE — FIRE FIRST (night shift 8): a face-a-blocker errand (the Route 12 Snorlax)
        # has NO door hint and anchors ON the blocker's own map — so the ANCHOR-FIRST routing below
        # (step.from_map -> Lavender) would route her BACK toward the gate town before the tail-end
        # strike hook (past the teach bridge) ever runs — the run7 stall at (3,30)@(12,0), and the
        # fixture reproduces it. Give the strike first crack for DOOR-LESS steps when she's standing
        # on its anchor map; it's a cheap no-op off-anchor / for steps it doesn't own (registry-keyed
        # on step.success). Dungeon errands (hideout/tower) carry door hints -> excluded here, still
        # flow via the door-hint + _questline_interact path they were verified on.
        if not getattr(step, "door", None):
            _bstrike = self._questline_strike(step)
            if _bstrike is not None:
                return _bstrike
        # FLASH ERRAND (2026-07-09 night-train shift 1): the whole flash gate — catch to 10 owned AND
        # reach the Route 2 aide via Route 11/Diglett's Cave AND teach — runs as one proven errand
        # (the catch-in-place driver species-exhausts on Route 10; the errand walks east to fresh grass
        # + the aide together). Owns BOTH the dex-prereq step and the flash step.
        if self._active_questline.gate.missing == "flash":
            if not getattr(self, "_ql_catch_voiced", False):
                self.on_event(step.human, kind="route", tier=2)
                self._ql_catch_voiced = True
            r = self._flash_errand()
            log(f"   [roam] 🔦 FLASH ERRAND ({step.missing}) -> {r}")
            if r == "flash_done":
                self.on_event("Flash — lit. now Rock Tunnel won't be pitch black. onward.",
                              kind="route", tier=2)
                self._clear_questline("flash taught")
                return "questline_done"
            if r == "flash_stuck":
                return "questline_unresolved"     # release to roam; forward-drive re-steers, re-recognize
            return "questline_flash"              # advanced a phase (leg / catch / cave / gatehouse)
        # DEX-CATCH STEP (2026-07-09 night-train shift 1): the actionable step is a catch-to-N-owned
        # prerequisite (the Flash aide's 10-species gate). DRIVE catching on reachable grass until the
        # dex count clears the step (re-derived next tick → advances to the HM handoff). catch_one
        # wanders + judges (a NEW species leans catch, dupes skipped), so it naturally hunts fresh
        # kinds. If there's no grass here / none reachable, fall through to the anchor routing so she
        # walks toward billed catch ground instead of spinning. Voiced once per questline.
        if getattr(step, "via", None) == "catch":
            try:
                _have = ram.pokedex_owned_count(self.b)
            except Exception:
                _have = 0
            _tgt = step.success[1] if step.success else 10
            if not getattr(self, "_ql_catch_voiced", False):
                self.on_event(step.human, kind="roster", tier=2)
                self._ql_catch_voiced = True
            cr = self.catch_one()
            log(f"   [roam] 🎣 QUESTLINE DEX-CATCH: dex={_have}/{_tgt} on {self.world.name(cur_map)} "
                f"-> catch_one={cr}")
            if cr not in ("no_grass", "no_reachable_target"):
                return "questline_catch"
            log(f"   [roam] DEX-CATCH: no wild ground reachable on {self.world.name(cur_map)} — "
                f"routing toward the catch anchor")
            # fall through to the anchor-first routing (walks toward step.from_map's grass country)
        _L2W = {"N": "north", "S": "south", "E": "east", "W": "west"}
        _OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
        # STEP-ANCHOR (2026-07-07, flute_run1): a chain's steps carry their OWN from-maps — the
        # silph_scope errand anchors on CELADON while its gate was recognized in LAVENDER. All
        # anchor-relative logic below (dock warp, re-anchor, past-anchor) must use the STEP's
        # anchor, else the executor walks the gate town's compass dir (she marched north out of
        # Lavender into Rock Tunnel hunting a Celadon errand). Past-anchor/bends reset per STEP.
        step_anchor = None
        try:
            if step.from_map:
                step_anchor = tuple(int(x) for x in str(step.from_map).split(","))
        except Exception:
            step_anchor = None
        if step_anchor is None:
            step_anchor = tuple(self._active_questline.gate.where or ()) or None
        if getattr(self, "_ql_cur_step", None) != step.missing:
            self._ql_cur_step = step.missing
            self._ql_past_anchor = False
            self._ql_bend_maps = set()
            self._ql_prev_map = None
        # HEAL-BEFORE-A-RIVAL-FIGHT GATE (2026-07-10, night shift 4 — the S.S. Anne PP-famine wall):
        # a questline step that crosses a hard TRAINER GAUNTLET / RIVAL fight (the 'cut' errand:
        # "...the captain's cabin is at the bow PAST THE RIVAL FIGHT") is a BATTLE wall, not just a nav
        # goal — but only GYM pushes had a pre-fight heal gate (prep_for_gym), so she boarded the ship
        # with the ace's attacking PP already drained from the Route-6 gauntlet -> PP FAMINE -> lost a
        # WINNABLE fight (a full 25-PP Razor Leaf solos Gary's L16-20 team, as the old solo-Ivysaur
        # track proved). A human ALWAYS taps the Center before a rival battle. When the step names a
        # rival/trainer fight, she's standing on its anchor town (a Center is here), and the ace isn't
        # attack-fresh (low HP, a faint, or damaging-PP low — the status-move-masked famine), heal
        # FIRST. Data-driven (keyed on the KB place_name wording, not a FireRed hardcode) + bounded to
        # ONE heal per (step, town) approach so it never loops; heal_nearest returns to the town.
        _pn = f"{getattr(step, 'place_name', '') or ''} {getattr(step, 'human', '') or ''}".lower()
        _rival_gauntlet = ("rival" in _pn) or ("trainers on board" in _pn)
        # Re-arm per approach: leaving the anchor town (boarding, or a post-loss walk-back) resets the
        # attempt counter, so each fresh approach heals again. The PP state itself is the loop guard —
        # once healed to full, the floor check goes False and the gate stops firing (no re-heal spin).
        if step_anchor and cur_map != step_anchor:
            self._ql_prefight_tries = 0
            self._ql_prefight_grind = 0
        # PREP-BEFORE-RIVAL (2026-07-10, night shift 6/7 — the S.S. Anne is a TEAM-STRENGTH wall, not just a
        # PP one): even at full PP + Tackle coverage, an under-levelled solo ace can't clear the rival's
        # 4-mon team — Gary's Pidgeotto Sand-Attacks her into missing (7 wasted Tackles), Charmeleon
        # resists grass, and a frail L8-12 bench (Abra knows only Teleport) can't help. The PROVEN line is
        # a stronger ace: the reactive track's solo VENUSAUR (L32) beats Gary clean. So GRIND to the
        # milestone BEFORE boarding (like gym-prep does before a gym), narrated in her voice. The grind's
        # weak-mon participation-switch levels the bench in the same pass. Bounded once-per-approach;
        # re-arms on leaving town. Reuses self.grind (self-capping budget).
        #
        # NIGHT-SHIFT 7 FIX — TARGET = L32, NOT L30: shift-6 lowered the target to L30 "for budget
        # feasibility", but a VERIFIED look-ahead (s7_verify1.log) proved a full-PP L30/L31 *Ivysaur*
        # STILL loses Gary (grudge 4W-5L, a 9-loss re-attempt loop). The tipping point is the EVOLUTION:
        # Ivysaur -> Venusaur at L32 is a major bulk/power spike, and Venusaur L32 is the reactive track's
        # proven Gary-killer. L30 leaves her one level short of the evolution — so she keeps losing a
        # WINNABLE-if-evolved fight. Grinding L28->L32 on Route 6 wilds needs ~200s (L28->L30 took ~90s in
        # that run), so a WIDE budget (900s) reaches L32+evolve in ONE pass, boards as Venusaur, and skips
        # the loss loop entirely (no re-grind needed once the ace clears the fight solo).
        # Env-tunable (default 32) so a look-ahead can observe the raw rival fight (PREP=0 disables the
        # grind: `0 < _ace_lv < 0` is always False) or re-tune the milestone without a recompile.
        _RIVAL_PREP_LEVEL = int(os.environ.get("POKEMON_RIVAL_PREP_LEVEL", "32") or "32")
        # COMPOSE WITH THE CABIN SWEEP — CABINS FIRST, wild-grind only tops off (2026-07-10, night shift 8):
        # wild-grinding all 6 levels L26->L32 up front is slow + unwatchable (shift 6/7 dead end), AND she
        # levels to ~L28 just walking to Vermilion, so a naive level-margin trips the slow up-front grind
        # anyway. The SHIP CABIN SWEEP levels the ace fast + authentically (verified L26->L30 fighting the
        # L16-18 cabin trainers). So make cabins ALWAYS get first crack: the wild prep-grind fires only
        # (a) if she's ALREADY basically at the target (>= target-margin, margin default 1) — nothing left
        # for cabins to do — OR (b) if she's ALREADY LOST to this rival gauntlet once (the cabins ran and
        # still left her short) -> now top off the residual gap. So: board -> cabins carry the bulk ->
        # WIN pass 1 if they reach L32; else lose -> whiteout to Vermilion -> _lost_here -> prep tops off
        # the last 1-2 levels to Venusaur -> re-board (cabins beaten, walk through) -> full-PP win. The
        # slow wild grind is thus BOUNDED to the residual cabins couldn't cover, never the full 6 levels.
        _PREP_CABIN_MARGIN = int(os.environ.get("POKEMON_RIVAL_PREP_MARGIN", "1") or "1")
        _prep_floor = max(1, _RIVAL_PREP_LEVEL - _PREP_CABIN_MARGIN) if _PREP_CABIN_MARGIN else 1
        # Have the cabins already run and still left her short? (a prior LOSS at this rival gauntlet)
        # AND: has this gauntlet already been CLEARED (a WIN here)? Once the rival is beaten, the whole
        # prefight machinery (grind AND heal) must go dark (2026-07-10, NIGHT SHIFT 12 — the real Vermilion
        # gym-entry wall). A STALE questline step (the 'HM Cut' errand that never clears after HM01) kept
        # `_rival_gauntlet` True, so head_to_gym LIVELOCKED on an endless pre-rival Center tap for a rival
        # who's already DOWN (Gary beaten, Venusaur L32 + Cut in hand): head_to_gym -> questline_prefight_heal
        # -> Center -> back to town -> _ql_prefight_tries reset on leaving town -> re-arm -> forever, so she
        # NEVER reaches the gym door + never goes 'stuck' + the shift-11 gym-gate probe never runs. Per-gauntlet
        # by PLACE (a WIN dominates a prior loss at the same spot), so LATER rival fights (Tower, Silph) still
        # prefight normally, and the FIRST pre-Gary approach (no encounter recorded yet) is unaffected.
        _rival_lost_here = False
        _rival_won_here = False
        try:
            _pl_lc = (f"{getattr(step, 'place_name', '') or ''} {getattr(step, 'human', '') or ''}").lower()
            for _e in ((self.strat.rival or {}).get("encounters") or []):
                _ep = (_e.get("place") or "").lower()
                if _ep and (_ep in _pl_lc or "s.s. anne" in _ep):
                    if _e.get("won"):
                        _rival_won_here = True
                    else:
                        _rival_lost_here = True
            if _rival_won_here:
                self._ship_cabins_swept = 0   # gauntlet cleared -> reset the sweep backstop counter
        except Exception:
            _rival_lost_here = False
            _rival_won_here = False
        # ANCHOR-ONLY PREP-BEFORE-RIVAL (2026-07-10 — NIGHT SHIFT 16 supersedes the shift-15 off-anchor grind).
        # Grind the ACE to the proven Venusaur L32 on SAFE anchor grass (Route 6) BEFORE boarding, but ONLY
        # while she's standing on the anchor (Vermilion). Shift 15 tried to ALSO grind off-anchor (after a Gary
        # loss whites her out to Cerulean) via grind()'s local grass-finder — but that dives DOWN the one-way
        # Route-4 ledge she can't path back UP, stranding her (s16 run: head_to_gym no_path, ace treadmilled
        # L35→39, PP-starved — a soft livelock). The RIGHT move off-anchor is NOT to grind locally but to ROUTE
        # BACK to the anchor: from a spine town like Cerulean head_to_gym/the billed road works, so falling
        # through to the normal questline routing carries her to Vermilion, where THIS anchor grind then fires.
        # The strategic path no longer trap-grinds her at Cerulean either (see _prep_team_target RIVAL-GAUNTLET
        # DEFER), so nothing pins her off-anchor — she simply walks back and grinds here. The hardcoded Route-6
        # walk_to_map avoids grind()'s auto grass-finder wedging in the Cut-tree gym-approach pocket.
        _prep_at_anchor = bool(step_anchor) and cur_map == step_anchor
        if _rival_gauntlet and not _rival_won_here and _prep_at_anchor \
                and not getattr(self, "_ql_prefight_grind", 0):
            try:
                _party = (self.read_live_state() or {}).get("party") or []
                _ace_lv = max((int(m.get("level", 0) or 0) for m in _party), default=0)
            except Exception:
                _ace_lv = 99                                   # read fail -> no grind (fail-open, safe)
            # PREP ON THE **FIRST** APPROACH (2026-07-10, NIGHT SHIFT 16 — bypass the whole post-loss mess):
            # grind the ace to L32 at the anchor whenever she's underlevelled — do NOT wait for a first Gary
            # LOSS to trigger it (the old `_rival_lost_here or _ace_lv >= _prep_floor` gate boarded her at L26,
            # cabin-swept to L30, and LOST — and that loss whites her out to Cerulean where routing back to the
            # anchor deadlocks on the S.S. Anne SPATIAL WALL / Cut-tree-gated road, s16b STALL). Prepping to L32
            # BEFORE the first boarding means she boards evolved, cabin-skips, and WINS first try — she never
            # loses, never whites out, never hits the return-routing wall. `0 < _ace_lv < L32` still self-
            # terminates the grind (ace hits L32 Venusaur -> guard False -> board). General per rule 14.
            if 0 < _ace_lv < _RIVAL_PREP_LEVEL:
                self._ql_prefight_grind = 1
                log(f"   [roam] 🏋️ PREP-BEFORE-RIVAL: a rival gauntlet is a team-strength wall — ace "
                    f"L{_ace_lv} < L{_RIVAL_PREP_LEVEL}; grinding up BEFORE boarding at the ship anchor (an "
                    f"underlevelled solo loses to Gary's 4-mon team, whatever the PP).")
                self.on_event("that's my rival waiting on that ship — I want my team stronger before we "
                              "board. quick training, then we go.", kind="route", tier=2)
                try:
                    # ROUTE TO GRASS FIRST: the gate fires while she's pinned in the gym-approach pocket
                    # (at the Cut tree) where grind()'s local grass-finder targets a bad off-map coord
                    # and travel-wedges. Route 6 (map (3,24), NORTH) is the grass she just crossed;
                    # walk_to_map routes there via the proper connection edge, bounded (8 tries, no spin).
                    # PORTABILITY DEBT: (3,24)=Route 6 is a FireRed S.S.-Anne coupling (KB layer, rule 14).
                    _rr = self.walk_to_map((3, 24), "north")
                    log(f"   [roam] 🏋️ pre-rival grind: routed to Route 6 grass -> {_rr}")
                    # WIDE budget (900s vs the 480s default): reaching the L32 evolution from L28 is ~4
                    # levels of low-XP wilds — the default budget could undershoot and board her as an
                    # un-evolved Ivysaur -> the loss loop. One deep pass to Venusaur is the whole point.
                    gr = self.grind(_RIVAL_PREP_LEVEL, budget_s=900)
                except Exception as e:
                    gr = f"error:{e}"
                log(f"   [roam] 🏋️ pre-rival grind (target L{_RIVAL_PREP_LEVEL}) -> {gr}")
                return "questline_prefight_grind"
        # FLOOR = TOP-OFF, not "some PP left" (night shift 6): floor=20 read her ~50 post-Route-6 PP as
        # "fine" and she boarded the S.S. Anne — a LONG Center-less gauntlet (3-4 ship trainers + Gary's
        # 4 mons, ~10 foes, many resisting grass so each needs 2-4 hits) — under-fuelled -> famined on
        # Charmeleon (grass x0.25, Tackle already 0-PP) -> lost a WINNABLE fight. A gauntlet needs a FULL
        # bar at the dock, so top off unless essentially full (floor 70 ~= her 75-PP max: Razor Leaf 25 +
        # Vine Whip 15 + Tackle 35). Bounded by the per-approach try cap; re-arms on leaving town.
        if _rival_gauntlet and not _rival_won_here and step_anchor and cur_map == step_anchor and (
                self._lead_attack_pp_low(floor=70) or not self._party_gym_ready()):
            self._ql_prefight_tries = getattr(self, "_ql_prefight_tries", 0) + 1
            if self._ql_prefight_tries <= 2:      # belt-and-suspenders: a reachable town Center never
                log("   [roam] 🩹 HEAL-BEFORE-RIVAL: a rival gauntlet is ahead (ship trainers + the "
                    "rival drain attacking PP with NO Center aboard) and the ace isn't attack-fresh "
                    "-> Center FIRST, full HP+PP, THEN board (kills the S.S. Anne PP famine).")
                self.on_event("hold on — nobody boards for a rival fight on fumes. Center first, "
                              "full HP and PP, then we go.", kind="route", tier=2)
                try:
                    hr = self.heal_nearest()
                except Exception as e:
                    hr = f"error:{e}"
                log(f"   [roam] 🩹 pre-rival heal (try {self._ql_prefight_tries}/2) -> {hr}")
                return "questline_prefight_heal"
        # DOOR HINT (flute_run1): the target building sits INSIDE the anchor town (Pokemon Tower,
        # the Game Corner) — no compass dir reaches it. The KB bills the exact door tile; standing
        # in the anchor town, enter THAT door. One shot per questline (the entered-doors set).
        if getattr(step, "door", None) and cur_map == step_anchor:
            if not hasattr(self, "_ql_entered_doors"):
                self._ql_entered_doors = set()
            _dkey = (cur_map, tuple(step.door))
            if _dkey not in self._ql_entered_doors:
                log(f"   [roam] 🚪 QUESTLINE DOOR-HINT: {step.place_name or step.missing} is IN "
                    f"town — entering the billed door {tuple(step.door)} on {cur_map}")
                _before = tuple(tv.map_id(self.b))
                self.enter_warp(pick=tuple(step.door), budget_s=900)
                if tuple(tv.map_id(self.b)) != _before:
                    self._ql_entered_doors.add(_dkey)
                    self._ql_room_sweeps = 0
                    self._ql_past_anchor = True
                    self._ql_inside_target = True
                    log(f"   [roam] 🚪 QUESTLINE ENTER (door-hint): inside {tv.map_id(self.b)}")
                    return "questline_entered"
                log(f"   [roam] !! QUESTLINE DOOR-HINT: couldn't enter {tuple(step.door)} — "
                    f"falling through to the general search")
        d = step.dir or "north"
        # ANCHOR-FIRST ROUTING (2026-07-07, flute_run2): the step's compass dir is relative to ITS
        # anchor map — from anywhere else it's noise. The old re-anchor only lived in the no-edge
        # fallback, so at Lavender the CELADON errand's 'north' matched Lavender's real north edge
        # and she ping-ponged Lavender<->Route 10 to a STALL, never routing home. Not yet past the
        # anchor and not standing on it -> route toward it via the world graph FIRST; dir/bend/door
        # logic applies only on/past the anchor. Graph can't reach it -> fall through to discovery.
        if (step_anchor and cur_map != step_anchor
                and not getattr(self, "_ql_past_anchor", False)):
            _pc2 = state.get("party_count")
            _pl2 = state["party"][0]["level"] if state.get("party") else None
            # WARP-AWARE routing (flute_run3: next_hop is EDGE-only, so Lavender->Celadon — a road
            # that crosses the Underground Path — read 'no route' and she fell into dir noise).
            # next_step routes THROUGH learned warps, executed exactly like head_to_gym's warp-route.
            try:
                wstep = self._next_step_rideable(cur_map, step_anchor, avoid=self._wall_avoid(state))
            except Exception as _ws:
                wstep = None
                log(f"   [roam] questline anchor-first next_step skipped: {_ws}")
            if wstep:
                _nxt, _kind, _detail = wstep
                if self.strat.is_gated(tuple(_nxt), _pc2, _pl2):
                    log(f"   [roam] questline: anchor-first hop {_nxt} is wall-gated — surfacing")
                    return "wall_gated"
                log(f"   [roam] 🧭 QUESTLINE ANCHOR-FIRST: '{step.human}' anchors on "
                    f"{self.world.name(step_anchor)} and we're at {self.world.name(cur_map)} — "
                    f"{_kind} -> {_nxt} toward the anchor first")
                if _kind == "warp":
                    _before = tuple(tv.map_id(self.b))
                    self.trav.travel(target_map=None, arrive_coord=_detail, max_steps=300)
                    if tuple(tv.map_id(self.b)) != _before:
                        return "warped"
                    # DIRECTIONAL stair/arrow tiles (UGP tunnel doorway 0x6F) fire only when
                    # entered moving their way — try that ritual before the door step-in.
                    if (self._enter_directional_warp(tuple(_detail))
                            and tuple(tv.map_id(self.b)) != _before):
                        return "warped"
                    if tuple(tv.map_id(self.b)) == _before:
                        self.enter_warp(pick=_detail)
                    return "warped" if tuple(tv.map_id(self.b)) != _before else "warp_failed"
                return self._edge_travel(_nxt, _detail)
            log(f"   [roam] questline: no graph route {cur_map} -> anchor {step_anchor} yet — "
                f"falling through to dir/bend discovery")
        letter = {"north": "N", "south": "S", "east": "E", "west": "W"}.get(d)
        pc, pl = state.get("party_count"), (state["party"][0]["level"] if state.get("party") else None)
        # ── THE TEACH BRIDGE (HM pipeline stage 2, 2026-07-06) ─────────────────────────────────
        # A ('cap', hm) step reads satisfied only when a PARTY MON KNOWS the move — but the errand
        # obtains the ITEM. With HM01 in the TM Case and nobody knowing Cut, routing back to the
        # captain loops forever. The bridge: item in the case + move unknown -> TEACH it here (her
        # judgment picks the mon; free-slot targets avoid the forget screen; RAM-verified LOUD).
        if step.success and step.success[0] == "cap":
            import hm_teach as ht
            hm = step.success[1]
            _hm_move = {"cut": 15, "fly": 19, "surf": 57, "strength": 70, "flash": 148,
                        "rock_smash": 249, "waterfall": 127}.get(hm)
            if (_hm_move is not None
                    and ht.tm_case_row(self.b, ht.HM_ITEM.get(hm, -1)) is not None
                    and st.party_knows_move(self.b, _hm_move, pc or 6) is None):
                # SETTLED RE-CHECK (celadon_run4/5): a mid-transition window misread the ace's
                # 4th move slot as empty and armed a bogus re-teach of a KNOWN move. Settle frames,
                # re-read with the LIVE party count; known -> the step is already satisfied.
                for _f in range(30):
                    self.b.run_frame()
                if st.party_knows_move(self.b, _hm_move, self.b.rd8(ram.GPLAYER_PARTY_CNT)) is not None:
                    log(f"   [roam] TEACH BRIDGE: settled re-read says {hm} IS known — step satisfied")
                    return "questline_step_done"
                plan = ht.default_plan(self.b, hm, pc or 6)
                if plan is None:
                    log(f"   [roam] !! TEACH: no compatible party mon for {hm} — releasing (LOUD)")
                    return "questline_unresolved"
                slot, forget_idx, reason = plan
                mon = st.SPECIES_NAME.get(st.read_party_species(self.b, slot), f"slot {slot}")
                self.on_event(f"I've got the HM right here — teaching {hm.title()} to {mon}. "
                              f"{reason}.", kind="route", tier=2)
                log(f"   [roam] 🧭 TEACH BRIDGE: {hm} -> {mon} (slot {slot}, {reason})")
                r = ht.TeachFlow(self, log=log, on_event=self.on_event).teach(hm, slot, forget_idx)
                log(f"   [roam] TEACH BRIDGE result: {r}")
                if r == "taught":
                    self.on_event(f"{mon} knows {hm.title()} now — that tree isn't stopping us anymore.",
                                  kind="route", tier=2)
                    return "questline_step_done"
                return "questline_teach_failed"
        # BESPOKE STRIKE at the STEP handler (night shift 7): some errands never reach the
        # 'arrived' _questline_interact hook because head_to_gym no_paths at the blocker first — the
        # Route-12 Snorlax (she stands at its face but nothing wires "play the Flute at it", so
        # head_to_gym re-routes into the body -> SPLIT-MAP DEAD ROAD). Cheap no-op while approaching
        # (the strike returns None unless she's ON the errand's anchor map); fires the wake ritual
        # when she's on Route 12. Dungeon errands still fire via _questline_interact (door-hint first).
        strike = self._questline_strike(step)
        if strike is not None:
            return strike
        log(f"   [roam] 🧭 QUESTLINE STEP: '{step.human}' — heading {d.upper()} toward "
            f"{step.place_name or 'the destination'} (success={step.success})")
        conns = self._map_connections()
        nbr = next(((g, n) for dd, (g, n) in conns if dd == letter), None)
        if nbr is None:
            # DESTINATION-IN-CITY (2026-07-06, the PIER class — ship run 7's 2-tile ping-pong): the
            # step points a direction with NO map connection while we stand ON the anchor map — the
            # way on is a WARP on THIS map (Vermilion's dock = its SOUTHmost warp trio; the ticket
            # triggers on the pier clear themselves). Try the direction-most warp BEFORE re-anchor/
            # bend logic (which bounced east to Route 11 and back forever).
            gate_where0 = step_anchor            # anchor-relative: the STEP's map (flute_run1 fix)
            if gate_where0 and cur_map == gate_where0 and d in ("south", "north"):
                # RIVAL-SHIP BOARDING GATE (2026-07-10, NIGHT SHIFT 16 — belt-and-braces at the ACTUAL
                # boarding warp). The prep block above can miss on a freshly gym-gate-opened questline
                # (s16c: she reached Vermilion, the Cut-tree gate opened the HM-Cut questline, and she
                # boarded the S.S. Anne at ace L28 WITHOUT prepping -> underlevelled cabin-sweep + Gary).
                # Here, at the confirmed pier-boarding warp, refuse to board a RIVAL gauntlet ship while
                # the ace is under the proven Venusaur L32: grind on Route 6 FIRST, then board evolved and
                # win first try (no loss -> no whiteout -> no return-routing deadlock). Diagnostic log
                # carries the inputs so a miss is legible. General per rule 14.
                try:
                    _pty = (self.read_live_state() or {}).get("party") or []
                    _acelv = max((int(m.get("level", 0) or 0) for m in _pty), default=99)
                except Exception:
                    _acelv = 99
                # NOTE: do NOT gate on `_rival_won_here` here — it stale-matches ANY historical won rival
                # encounter whose place contains "s.s. anne" (the bill_done_kit fixture carries one), so it's
                # True even on the FIRST approach and wrongly skipped the prep (s16d diagnostic: won=True at
                # ace L28). Standing at THIS boarding warp for the HM-Cut step already proves she hasn't
                # completed it (no Cut yet -> the gym gate is still shut), so a stale "won" is irrelevant.
                _need_ship_prep = (_rival_gauntlet and 0 < _acelv < _RIVAL_PREP_LEVEL
                                   and not getattr(self, "_ql_prefight_grind", 0))
                log(f"   [roam] 🚢 BOARDING GATE: rival={_rival_gauntlet} won(stale)={_rival_won_here} ace=L{_acelv} "
                    f"prep_lv=L{_RIVAL_PREP_LEVEL} armed={getattr(self, '_ql_prefight_grind', 0)} "
                    f"-> {'GRIND FIRST' if _need_ship_prep else 'board'}")
                if _need_ship_prep:
                    self._ql_prefight_grind = 1
                    self.on_event("that's my rival waiting on that ship — I want my team stronger before we "
                                  "board. quick training on Route 6, then we go.", kind="route", tier=2)
                    try:
                        _rr = self.walk_to_map((3, 24), "north")   # Route 6 grass (safe; no Cut-tree pocket)
                        log(f"   [roam] 🏋️ rival-ship prep: routed to Route 6 grass -> {_rr}")
                        _gr = self.grind(_RIVAL_PREP_LEVEL, budget_s=900)
                    except Exception as _e:
                        _gr = f"error:{_e}"
                    log(f"   [roam] 🏋️ rival-ship prep grind (target L{_RIVAL_PREP_LEVEL}) -> {_gr}")
                    return "questline_prefight_grind"
                before_m = tuple(tv.map_id(self.b))
                # WRONG-ENTRANCE RECOVERY (sabrina_run4, the Celadon-Mansion back-door loop): pick
                # the d-most door OURSELVES instead of enter_warp(prefer=d), so we can (a) skip
                # doors this questline already entered-and-failed (the blind picker re-entered the
                # mansion's back stairwell forever) and (b) PREFER sibling doors into the interior
                # map that held a visible-but-unreachable occupant (same building, other entrance —
                # the tea room is sealed from the back door's spawn). Fall back to d-most order.
                if not hasattr(self, "_ql_entered_doors"):
                    self._ql_entered_doors = set()
                try:
                    _wd = {tuple(xy): tuple(dest) for (xy, dest, _wid) in tv.read_warps(self.b)}
                except Exception:
                    _wd = {}
                _sib = getattr(self, "_ql_sibling_dest", None)
                _cands = [tuple(t) for t in self._door_tiles()
                          if (cur_map, tuple(t)) not in self._ql_entered_doors]
                _cands.sort(key=lambda t: (0 if (_sib and _wd.get(t) == _sib) else 1,
                                           t[1] if d == "north" else -t[1]))
                _warped_in = False
                for _dr in _cands[:6]:
                    if self.enter_warp(pick=_dr, budget_s=300) == "warped" \
                            and tuple(tv.map_id(self.b)) != before_m:
                        # remember HOW we got in; only burned into entered_doors if the visit
                        # FAILS (wrong building) — a mid-quest heal-return may re-enter freely
                        self._ql_city_door = (cur_map, _dr)
                        _warped_in = True
                        break
                if _warped_in:
                    # entering the destination-ward warp = we are PAST the anchor now; without this
                    # the re-anchor would route her straight OFF the ship back to the city. And she
                    # is now deliberately INSIDE quest territory — the blackout interior auto-exit
                    # must LEAVE HER THERE (run 9: it ejected her off the ship one tick after
                    # boarding and the no-progress guard then killed the errand).
                    self._ql_past_anchor = True
                    self._ql_inside_target = True
                    log(f"   [roam] 🧭 QUESTLINE: no {d} edge on the anchor map — entered its "
                        f"{d}-most warp toward {step.place_name or 'the destination'} "
                        f"({before_m} -> {tv.map_id(self.b)})")
                    return "warped"
            # RE-ANCHOR (2026-07-05, run-4 lesson): the step's coarse dir is relative to the GATE map
            # (e.g. "Bill is north" means north OF CERULEAN). Standing on a map BEHIND the anchor with no
            # step-dir edge (post-grind on Route 4, west of Cerulean), the old fallbacks misfire: the
            # frontier is empty (Cerulean already visited) → "arrived at the destination area" → interact
            # with nothing → head_to_gym no-ops forever (the (107,12) wedge). If we haven't yet crossed
            # PAST the anchor this questline, the move is simply: route back TO the anchor city first.
            gate_where = step_anchor             # anchor-relative: the STEP's map (flute_run1 fix)
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
            _back = _OPP.get(letter)
            _came = getattr(self, "_ql_prev_map", None)
            frontier = [(dd, nb) for dd, nb in conns
                        if dd != _back and not self.world.visited(tuple(nb))]
            fresh_bend = bool(frontier)
            if not frontier:
                # PRE-MAPPED FORWARD (2026-07-09, the Route-24 -> Route-25 look-ahead wedge): the bend
                # keyed 'forward' off `not visited`, which holds on a FRESH world but NOT when the world
                # model is pre-loaded — the look-ahead loads the CANONICAL world (Champion run visited
                # every map, incl. Route 25/Bill), so no UNVISITED forward edge exists and the old code
                # mis-declared 'arrived' on Route 24, parking head_to_gym while Bill sat one map east.
                # (Same class bites late-game backtrack fetch-quests over known ground.) A VISITED
                # forward map is NOT arrival — cross a non-back connection we did NOT just come from
                # (anti-ping-pong), preferring the coarse-dir edge; BEND-CONTINUE + the interact layer
                # carry her the rest (on the destination map the frontier empties and she interacts).
                frontier = [(dd, nb) for dd, nb in conns
                            if dd != _back and tuple(nb) != _came]
                frontier.sort(key=lambda t: 0 if t[0] == letter else 1)
            if frontier:
                dd, nb = frontier[0]
                nb = tuple(nb)
                dword = _L2W[dd]
                if self.strat.is_gated(nb, pc, pl):
                    log(f"   [roam] questline: frontier hop {nb} ({dword}) is wall-gated — surfacing")
                    return "wall_gated"
                log(f"   [roam] 🧭 QUESTLINE EXPLORE: no {d} edge from {cur_map} — the route bends; "
                    f"crossing {dd} into {'unexplored' if fresh_bend else 'known-but-forward'} {nb} "
                    f"toward {step.place_name or 'the destination'}")
                if not hasattr(self, "_ql_bend_maps"):
                    self._ql_bend_maps = set()
                self._ql_bend_maps.add(nb)         # remember the bend so a bounce-back can CONTINUE it
                self._ql_prev_map = cur_map        # anti-ping-pong for the next visited-fallback hop
                r = self.trav.travel(target_map=nb, edge=dword)
                # FENCED BEND EDGE (descent re-grade 2026-07-08): the header bills the connection
                # but the walkable crossing is a CONNECTOR BUILDING (Route 8's west "edge" is the
                # sealed Saffron gatehouse; the real way on is the Underground Path hut). The
                # door-passthrough primitive owns exactly this class — try it before surfacing the
                # failed edge, and count the far side as bend ground so arrival gating holds.
                if r in ("no_path", "stuck"):
                    pt = self._door_passthrough(want_map=nb)
                    if pt == "crossed":
                        self._ql_bend_maps.add(tuple(tv.map_id(self.b)))
                        log(f"   [roam] 🧭 QUESTLINE PASSTHROUGH: the {dd} edge was fenced — a "
                            f"connector building carried us to {tv.map_id(self.b)}")
                        return "questline_passthrough"
                    if pt == "need_heal":
                        return "need_heal"
                return r
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
            # FALSE-ARRIVAL GUARD (descent re-grade 2026-07-08, the Route-10 south strand): "no
            # forward edge + no unvisited neighbor" is NOT arrival while the step has an anchor we
            # never reached — on a dead-end pocket it declared 'arrived', interacted with nothing,
            # and head_to_gym no-op'd forever. Off-anchor, off-bend, never past the anchor, and the
            # graph can't route there (RE-ANCHOR above already tried) → surface honestly; the
            # structural dead-route memory prunes head_to_gym on THIS map and the roam layer's other
            # options carry her until the graph learns a road toward the anchor.
            if (step_anchor and cur_map != step_anchor and cur_map not in bends
                    and not getattr(self, "_ql_past_anchor", False)):
                log(f"   [roam] questline: NOT arrived — {cur_map} is off-anchor "
                    f"({step_anchor} unreachable in the learned graph) with no local frontier; "
                    f"surfacing no-route")
                return "questline_no_route"
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
        # CONFIRMED-CROSSING LATCH (2026-07-09, the Nugget-Bridge grind wedge): only latch 'past the
        # anchor' when the crossing ACTUALLY changed maps. A bounced north crossing (the rival fight at
        # Cerulean's north exit -> heal-when-low, or any mid-cross need_heal) leaves her ON the anchor
        # while the old code latched anyway — permanently disabling re-anchor so every future off-anchor
        # tick misfired 'arrived at the destination' and she macro-looped grinding on Route 4 (the
        # (107,12) wedge). She defeated the rival on the failed pass, so the retry crosses clean.
        if cur_map == step_anchor:
            _before_cross = tuple(tv.map_id(self.b))
            r = self.trav.travel(target_map=nbr, edge=d)
            if tuple(tv.map_id(self.b)) != _before_cross:
                self._ql_past_anchor = True
            return r
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
        # BESPOKE INTERIOR STRIKE (2026-07-09 night shift 7): a few questline targets sit behind a
        # fixed multi-floor RITUAL the blind room-tour + GO-DEEPER cannot solve — the Silph Scope in
        # the Rocket Hideout (poster-gate -> arrow-tile spin mazes -> Lift Key -> B2F elevator ->
        # Giovanni -> Scope -> ride/climb back OUT), proven e2e by the champion's recon_hideout.py +
        # recon_hideout_exit.py, ported in-loop as hideout_strike.run_strike. Fires only when the
        # errand + her position match; returns None to fall through to the general layer otherwise.
        strike = self._questline_strike(step)
        if strike is not None:
            return strike
        if step.via != "talk_npc":
            # other step kinds (board a ship, use an HM at the destination) are future interaction layers
            log(f"   [roam] questline: arrived but step.via='{step.via}' has no interaction layer yet — surfacing")
            return "questline_no_interaction"
        if not hasattr(self, "_ql_entered_doors"):
            self._ql_entered_doors = set()
        interior = tv.map_id(self.b)[0] != 3
        if interior:
            # TALK BUDGET per room ENTRY (ship run 16, the KITCHEN class): each re-sweep CLEARS the
            # talked-set, and a room of WANDERING occupants (6 cooks) then re-feeds the first-pass
            # talk one 'questline_talked' tick apiece — ~18 identical-fingerprint ticks, tripping
            # the 14-decision STALL detector before the tour could move on. Budget = every occupant
            # once + a few re-talks (script flips like Bill need exactly one), hard-capped safely
            # under the stall threshold; a chatted-out room skips straight to the GO-DEEPER tour.
            try:
                n_npc = sum(1 for _ in self._talkable_npcs())
            except Exception:
                n_npc = 4
            talk_budget = max(4, min(10, n_npc + 3))
            # PER MAP for the questline's LIFETIME, not per entry (run 17): the tour re-transits
            # chatted-out rooms, and a per-entry reset re-burned the whole budget in each — the
            # STALL detector (whose sig can't see room-touring) ran out ~5 hops short of the
            # captain. A room once chatted out STAYS chatted out; re-entries go straight to the
            # tour. The dict resets with the questline (any step completion re-derives).
            if not hasattr(self, "_ql_talks_map"):
                self._ql_talks_map = {}
            _mp_key = tuple(tv.map_id(self.b))
            talks = self._ql_talks_map.get(_mp_key, 0)
            chatted_out = talks >= talk_budget
            if chatted_out:
                log(f"   [roam] questline: room chatted out ({talks}/{talk_budget} talks) — moving the tour on")
            # She's inside a building she ENTERED for the quest (the blackout-recovery leaves her be while
            # `_ql_inside_target` is set — they cooperate). Talk the occupant(s); the flag re-checks next tick.
            if not chatted_out:
                r = self.talk_npc()
                if r == "talked":
                    self._ql_talks_map[_mp_key] = talks + 1
                    log("   [roam] 🗣️ QUESTLINE TALK: spoke to someone inside — re-checking the flag next tick")
                    return "questline_talked"
                # WORK THE ROOM (2026-07-05, the Bill Cell-Separation class): quest buildings have scripted
                # MACHINES/consoles — BG events read live from the map header (no hardcoded coords). Interact
                # them facing-correct; a fired script (box opened) means world state advanced — re-check next tick.
                if self._questline_bg_sweep():
                    self._ql_talks_map[_mp_key] = talks + 1
                    log("   [roam] 🖥️ QUESTLINE ROOM: worked a machine/sign in here — re-checking the flag next tick")
                    return "questline_worked_room"
            # RE-SWEEP, INLINE (2026-07-06 rework): the old version RETURNED 'questline_resweep'
            # (a deliberate no-move tick) twice — and the roam's SILENT-NO-MOVE pruner counts two
            # no-move returns as a dead route, so head_to_gym got PRUNED before the third call
            # could ever GO DEEPER (ship run 12: the 2F stairs sat untried while she stalled).
            # Now the re-sweep happens INSIDE this call: clear the sets, talk+work once more, and
            # fall through to GO-DEEPER in the same tick — no intentional no-move returns exist.
            # BOUNDED per room ENTRY (ship run 14): a dead-end room with ONE chatty occupant (the
            # Machoke sailor, cabin (1,28)) made the unbounded re-sweep an infinite loop — clear
            # set → re-talk the same NPC → 'questline_talked' every tick, so GO-DEEPER below was
            # UNREACHABLE and the tour died in the first NPC'd cul-de-sac. Scripted rooms need the
            # re-sweep (Bill's cottage: console → Bill-as-human is re-sweep #1); two per entry
            # covers those, then the tour proceeds. _ql_room_sweeps resets on every room change.
            sweeps = getattr(self, "_ql_room_sweeps", 0)
            if not chatted_out and sweeps < 2:
                self._ql_room_sweeps = sweeps + 1
                try:
                    self._talked_npcs.get(tv.map_id(self.b), set()).clear()
                    getattr(self, "_ql_bg_done", set()).clear()   # scripts are state-dependent — re-arm too
                except Exception:
                    pass
                log(f"   [roam] questline: inline room re-sweep {sweeps + 1}/2 (scripts change who's here)")
                r = self.talk_npc()
                if r == "talked":
                    self._ql_talks_map[_mp_key] = self._ql_talks_map.get(_mp_key, 0) + 1
                    log("   [roam] 🗣️ QUESTLINE TALK (re-sweep): spoke to someone — re-checking the flag next tick")
                    return "questline_talked"
                if self._questline_bg_sweep():
                    self._ql_talks_map[_mp_key] = self._ql_talks_map.get(_mp_key, 0) + 1
                    log("   [roam] 🖥️ QUESTLINE ROOM (re-sweep): worked a machine — re-checking the flag next tick")
                    return "questline_worked_room"
            elif not chatted_out:
                log("   [roam] questline: room re-swept 2x with no new script — moving the tour on")
            # GO DEEPER (2026-07-06, the SHIP class): a multi-room quest building (S.S. Anne:
            # exterior deck → gangway → 1F corridor → 2F → the captain's office) has the target
            # several WARPS in — "nobody here" means the NEXT room, not the wrong building. Enter an
            # untried warp on THIS map that does NOT lead back to the overworld, deepest-first
            # (farthest from where we stand). The entered-doors set makes the tour FINITE; cabin
            # trainers met on the way are the intended XP.
            mp0 = tuple(tv.map_id(self.b))
            here0 = tuple(tv.coords(self.b) or (0, 0))
            ws0 = tv.read_warps(self.b)
            _dest_of = {tuple(xy): tuple(dest) for (xy, dest, _wid) in ws0}
            # DIRECTED INTERIOR NAV (night shift 12 — ends the 4-shift S.S. Anne captain loop): the
            # blind cabin tour below can't reliably reach a target several warps deep (the captain on
            # 2F) — the 2F stairs rank as a 'vis' node behind every NEW dead-end cabin, and she bails
            # to the exit lobby before climbing. When the game-knowledge KB has a DIRECTED warp-chain
            # for THIS errand, follow it ONE hop toward the target; the chain is world-model-independent
            # so it works on a fresh timeline too. Falls through to the blind tour if the hop can't move
            # her (unknown map / warp failed / walk-unreachable) so nothing regresses.
            _dir_tile = self._questline_interior_route(step)
            if _dir_tile is not None:
                # FIGHT-THE-CABINS-FIRST (2026-07-10, night shift 8 — the S.S. Anne Gary TEAM-STRENGTH
                # wall). The directed hop B-LINES up the stairs to the captain, SKIPPING every cabin
                # trainer on this deck (Gentlemen/Sailors, L16-18) — the intended level-up a human
                # fights through. Arriving underlevelled, her solo ace PP-famines vs Gary's 4-mon team
                # (Pidgeotto's Sand-Attack whiff-spiral wastes her PP) and the frail L8-12 bench can't
                # finish (VERIFIED loss at L29, grudge 4W-5L, s8_gary_observe.log). Wild-grinding to the
                # Venusaur milestone on Route 6 trash is too slow + unwatchable (shift 6/7 dead end). So
                # BEFORE climbing, sweep this deck's un-entered cabins — fight their trainers, level the
                # WHOLE team (bench included via participation) — and climb only once they're exhausted.
                # Bounded by _ql_entered_doors (finite); exclusions mirror the blind tour's (no overworld
                # exit, no maze, no re-entry, NOT the stairs tile itself). Fail-open + kill-switch. This
                # is the authentic, watchable line: a real trainer clears the ship, THEN faces the rival.
                # PORTABILITY (rule 14): general "clear a directed interior's trainers before the deep
                # target" — the S.S. Anne coupling lives in the KB (interior_routes), not here.
                # SWEEP ONLY WHEN THE XP IS ACTUALLY NEEDED (2026-07-10, NIGHT SHIFT 13 — the S.S. Anne
                # cabin-sweep LIVELOCK that ate the whole badge-3 run). Shift 8 built this sweep to
                # out-level Gary's whiff-spiral; shift 11 then KILLED the whiff-spiral (it was a
                # false-positive DETECTOR, not a real team wall), so an over-levelled ace now beats
                # S.S. Anne Gary (ace ~L20) outright and the sweep is vestigial — worse, it re-clears
                # ~9 cabins PER DECK across ~3 decks (the un-entered count resets 9->1 each new deck)
                # and burned the ENTIRE 126-tick / 40-min budget without ever reaching the captain or
                # the rival (s12_surge.log: 4000+ ticks bouncing the S.S. Anne, badges never left 2).
                # FIX: skip the sweep and B-line the directed climb once the ace clearly overpowers this
                # rival (nothing left for L16-18 cabin trainers to teach); ALSO cap total cabins swept
                # per approach as a backstop so an underlevelled ace fights a handful for XP then climbs
                # (never a re-sweep livelock) — the lose->prep-grind->reboard recovery covers any residual
                # gap. Env-tunable; kill-switch preserved.
                _sweep_on = os.environ.get("POKEMON_SHIP_CABIN_SWEEP", "1") == "1"
                try:
                    _sw_party = (self.read_live_state() or {}).get("party") or []
                    _sw_ace = max((int(m.get("level", 0) or 0) for m in _sw_party), default=0)
                except Exception:
                    _sw_ace = 99                                    # read fail -> no sweep (fail-open)
                # NIGHT-SHIFT 14 FIX — the skip level MUST match the Gary-winnable level, not sit BELOW it.
                # Shift 13 set the skip to L24 while the rival-prep target (the empirically-proven Gary-killer,
                # Venusaur L32) is L32 and the first-approach prep-grind floor is L31 — a DEAD-ZONE (L24-L30):
                # an L28 ace SKIPPED the cabins (28 >= 24) AND the first-approach prep-grind didn't fire yet
                # (28 < 31, no loss recorded), so she B-lined to Gary underprepared and took a GUARANTEED LOSS
                # (s13_surge.log: Razor Leaf resisted x0.25 by Charmeleon, Tackle PP-famined, frail L8-12 bench,
                # SE-CHUNK Fire hits) -> whited out to CERULEAN (not Vermilion) -> a fresh Gary-loss livelock
                # replacing the old cabin-sweep one. FIX: align the skip to _RIVAL_PREP_LEVEL (default 32) so
                # BELOW L32 the cabins run (bounded by the cap) and LEVEL the ace toward Venusaur, and only an
                # already-Venusaur ace (>= L32) skips + B-lines to a WINNABLE fight. The cap (persistent
                # _ship_cabins_swept counter, immune to the per-deck door-identity reset) is the true
                # livelock backstop; raise it so cabins can carry L28->~L32 in ONE approach (self-terminates
                # the instant _sw_ace hits L32 -> skip fires), the lose->prep-grind->reboard path tops off any
                # residual. Env-tunable; kill-switch (POKEMON_SHIP_CABIN_SWEEP=0) preserved.
                # Couple the skip default to the SAME rival-prep knob (POKEMON_RIVAL_PREP_LEVEL, default 32)
                # the pre-board grind targets, so the two thresholds can never drift into a dead-zone again.
                _prep_lv = int(os.environ.get("POKEMON_RIVAL_PREP_LEVEL", "32") or "32")
                _sweep_skip_lv = int(os.environ.get("POKEMON_SHIP_SWEEP_SKIP_LEVEL", str(_prep_lv))
                                     or str(_prep_lv))
                _sweep_cap = int(os.environ.get("POKEMON_SHIP_SWEEP_CAP", "10") or "10")
                _swept = getattr(self, "_ship_cabins_swept", 0)
                _do_sweep = _sweep_on and (0 < _sw_ace < _sweep_skip_lv) and (_swept < _sweep_cap)
                if _sweep_on and not _do_sweep:
                    if _sw_ace >= _sweep_skip_lv:
                        log(f"   [roam] 🥊 SHIP CABIN SWEEP SKIPPED — ace L{_sw_ace} already overpowers "
                            f"this rival (>= L{_sweep_skip_lv}); B-lining the climb to the captain/rival.")
                    elif _swept >= _sweep_cap:
                        log(f"   [roam] 🥊 SHIP CABIN SWEEP CAP reached ({_swept}/{_sweep_cap}) — enough "
                            f"XP; climbing to the rival now.")
                if _do_sweep:
                    def _exit_lobby(dmap):
                        nd = self.world.node(dmap)
                        return bool(nd) and any(str(v).split(",")[0] == "3"
                                                for v in nd.get("warps", {}).values())
                    _cabins = [tuple(xy) for (xy, dest, _wid) in ws0
                               if tuple(xy) != _dir_tile and dest[0] != 3
                               and (mp0, tuple(xy)) not in self._ql_entered_doors
                               and tuple(dest) not in self._no_connector_maps()
                               and not _exit_lobby(tuple(dest))
                               and self._warp_hop_reachable(tuple(xy))]
                    if _cabins:
                        _cabins.sort(key=lambda t: abs(t[0] - here0[0]) + abs(t[1] - here0[1]))
                        _cab = _cabins[0]
                        log(f"   [roam] 🥊 SHIP CABIN SWEEP: clearing this deck's trainers before the "
                            f"climb — cabin warp {_cab} ({len(_cabins)} un-entered here)")
                        self.on_event("this ship's crawling with trainers — I'm working the cabins for "
                                      "XP before I take on my rival up top. level the whole squad first.",
                                      kind="route", tier=2)
                        if tuple(tv.coords(self.b) or ()) == _cab:
                            g3 = tv.Grid(self.b)
                            _fire = (self._WARP_ENTRY.get(self._tile_behavior(*_cab)) or (None, None))[1]
                            for nb in ((_cab[0], _cab[1] + 1), (_cab[0], _cab[1] - 1),
                                       (_cab[0] + 1, _cab[1]), (_cab[0] - 1, _cab[1])):
                                if _fire is not None and (nb[0] - _cab[0], nb[1] - _cab[1]) == _fire:
                                    continue
                                if g3.walkable(nb[0], nb[1]):
                                    self.trav.travel(target_map=None, arrive_coord=nb,
                                                     max_steps=10, max_seconds=15)
                                    break
                        if self._tile_behavior(*_cab) in self._WARP_ENTRY:
                            self._enter_directional_warp(_cab)
                        if tuple(tv.map_id(self.b)) == mp0:
                            self.enter_warp(pick=_cab, budget_s=180)
                        if tuple(tv.map_id(self.b)) != mp0:
                            self._ql_entered_doors.add((mp0, _cab))
                            self._ship_cabins_swept = getattr(self, "_ship_cabins_swept", 0) + 1
                            self._ql_room_sweeps = 0
                            self._ql_inside_target = True   # a cabin is deliberate interiority
                            try:
                                getattr(self, "_ql_bg_done", set()).clear()
                                self._talked_npcs.get(tuple(tv.map_id(self.b)), set()).clear()
                            except Exception:
                                pass
                            log(f"   [roam] 🥊 CABIN: {mp0} -> {tv.map_id(self.b)} via {_cab} "
                                f"(fighting for XP before the rival climb)")
                            return "questline_deeper"
                        log("   [roam] 🥊 CABIN SWEEP: entry didn't move us — climbing instead")
                if not self._warp_hop_reachable(_dir_tile):
                    log(f"   [roam] questline: DIRECTED warp {_dir_tile} walk-unreachable from here — "
                        f"falling back to the blind tour")
                else:
                    log(f"   [roam] questline DIRECTED interior hop -> warp {_dir_tile} "
                        f"(KB chain to the target, skipping the blind cabin tour)")
                    # step OFF the warp tile if we're standing on it (same ritual as the tour below —
                    # never step off in a directional tile's own fire direction).
                    if tuple(tv.coords(self.b) or ()) == _dir_tile:
                        g3 = tv.Grid(self.b)
                        _fire = (self._WARP_ENTRY.get(self._tile_behavior(*_dir_tile)) or (None, None))[1]
                        for nb in ((_dir_tile[0], _dir_tile[1] + 1), (_dir_tile[0], _dir_tile[1] - 1),
                                   (_dir_tile[0] + 1, _dir_tile[1]), (_dir_tile[0] - 1, _dir_tile[1])):
                            if _fire is not None and (nb[0] - _dir_tile[0], nb[1] - _dir_tile[1]) == _fire:
                                continue
                            if g3.walkable(nb[0], nb[1]):
                                self.trav.travel(target_map=None, arrive_coord=nb,
                                                 max_steps=10, max_seconds=15)
                                break
                    if self._tile_behavior(*_dir_tile) in self._WARP_ENTRY:
                        self._enter_directional_warp(_dir_tile)
                    if tuple(tv.map_id(self.b)) == mp0:
                        self.enter_warp(pick=_dir_tile, budget_s=180)
                    if tuple(tv.map_id(self.b)) != mp0:
                        self._ql_entered_doors.add((mp0, _dir_tile))
                        self._ql_room_sweeps = 0
                        self._ql_inside_target = True   # a DEEPER hop is deliberate interiority
                        try:
                            getattr(self, "_ql_bg_done", set()).clear()
                            self._talked_npcs.get(tuple(tv.map_id(self.b)), set()).clear()
                        except Exception:
                            pass
                        log(f"   [roam] 🚪 QUESTLINE DIRECTED: {mp0} -> {tv.map_id(self.b)} via warp "
                            f"{_dir_tile} (climbing the KB chain to the captain)")
                        return "questline_deeper"
                    log("   [roam] questline: DIRECTED hop didn't move us — falling through to the blind tour")
            # EXIT-LOBBY GUARD (night shift 9, the S.S. Anne captain wedge): the GO-DEEPER tour
            # ranks warps farthest-first among unvisited — but the ship's 2F stairs (3,8)->(1,6)
            # sit at LOW-x, NEAR her frequent 1F landing spots, so they sorted LAST while she toured
            # the far cabins; before she reached them she picked a warp back to the EXTERIOR (a
            # visited-but-valid candidate that merely passed the dest[0]!=3 filter), landed on the
            # exit lobby, found nothing deeper, and declared 'wrong building' — leaving the ship
            # WITHOUT EVER CLIMBING to the captain on 2F. A map that itself warps OUT to the
            # overworld (group 3) is the building's EXIT LOBBY; touring INTO it is retreating, never
            # descending. Exclude it so the tour exhausts the true interior (the stairs included)
            # before it can give up. Fail-OPEN: an unknown/unvisited dest isn't treated as a lobby
            # (a FRESH world discovers forward exactly as before — the lobby is only knowable once
            # its overworld warp is learned, which on a fresh world happens as she walks in).
            def _is_exit_lobby(dmap):
                nd = self.world.node(dmap)
                if not nd:
                    return False
                return any(str(v).split(",")[0] == "3" for v in nd.get("warps", {}).values())
            cand = [tuple(xy) for (xy, dest, _wid) in ws0
                    if dest[0] != 3 and (mp0, tuple(xy)) not in self._ql_entered_doors
                    and tuple(dest) not in self._no_connector_maps()   # never tour INTO a maze
                    and not _is_exit_lobby(tuple(dest))]               # never RETREAT to the exit lobby
            # UNVISITED MAPS FIRST (run-13 lesson): deepest-first kept touring the same holds while
            # the 2F stairs (an unvisited map) sat untried — a warp into somewhere NEW is the whole
            # point of going deeper. Distance breaks ties.
            cand.sort(key=lambda t: (self.world.visited(_dest_of.get(t, (0, 0))),
                                     -(abs(t[0] - here0[0]) + abs(t[1] - here0[1]))))
            # DIAGNOSTIC (night shift 9): print the ranked tour so a re-run log shows exactly which
            # deeper warps were tried/skipped and whether the stairs are ever reached.
            log("   [roam] questline GO-DEEPER cand on {}: {}".format(
                mp0, [(t, _dest_of.get(t), "vis" if self.world.visited(_dest_of.get(t, (0, 0)))
                       else "NEW") for t in cand]))
            for wt in cand:
                # FEET-REACHABILITY LAW (night shift 8, the Celadon Mansion climb): from a
                # warp-partitioned landing (back stairwell) the FRONT stairs are visible but
                # walk-sealed — enter_warp then grinds its whole 180s budget against the gap
                # (~15 travel wedges + static-obstacle churn PER FLOOR). Same law as shift 6's
                # exit-chain sealed-door skip: pre-check before any leg, and DON'T consume the
                # candidate (floors mutate — it retries the moment the pocket unseals).
                if not self._warp_hop_reachable(wt):   # shift-11: was _tile_feet_reachable, a
                    #                                    never-defined name — AttributeError'd the
                    #                                    whole go-deeper tour since shift 8
                    log(f"   [roam] questline: deeper warp {wt} walk-unreachable from here — "
                        f"skipped (sealed pocket, not consumed)")
                    continue
                # spawning ON a warp tile doesn't re-fire it — step OFF first so the entry ritual
                # (approach + step-in) has somewhere to approach from (run-13: stalled standing on
                # the hold's own exit warp). NEVER step off in a directional tile's own fire
                # direction — pressing it fires the warp (silph strike4, probe-proven: the 1F
                # entrance mat is 0x65 DOWN-arrow; south-first stepped her out the front door).
                if tuple(tv.coords(self.b) or ()) == wt:
                    g3 = tv.Grid(self.b)
                    _fire = (self._WARP_ENTRY.get(self._tile_behavior(*wt)) or (None, None))[1]
                    for nb in ((wt[0], wt[1] + 1), (wt[0], wt[1] - 1),
                               (wt[0] + 1, wt[1]), (wt[0] - 1, wt[1])):
                        if _fire is not None and (nb[0] - wt[0], nb[1] - wt[1]) == _fire:
                            continue
                        # Grid.walkable takes SAVE coords (it adds MAP_OFFSET itself) — the old
                        # +OFFSET call read a tile shifted +7,+7 (silph probe3, 2026-07-07).
                        if g3.walkable(nb[0], nb[1]):
                            self.trav.travel(target_map=None, arrive_coord=nb,
                                             max_steps=10, max_seconds=15)
                            break
                if self._tile_behavior(*wt) in self._WARP_ENTRY:
                    self._enter_directional_warp(wt)
                if tuple(tv.map_id(self.b)) == mp0:
                    self.enter_warp(pick=wt, budget_s=180)
                if tuple(tv.map_id(self.b)) != mp0:
                    self._ql_entered_doors.add((mp0, wt))
                    self._ql_room_sweeps = 0
                    # a DEEPER hop is DELIBERATE interiority (run 15: the marker had been dropped
                    # mid-tour, so the blackout/stranded recovery ejected her from every new room
                    # — 2F included — with a FALSE 'I blacked out' beat + a fake note_blackout).
                    self._ql_inside_target = True
                    try:
                        getattr(self, "_ql_bg_done", set()).clear()
                        self._talked_npcs.get(tuple(tv.map_id(self.b)), set()).clear()
                    except Exception:
                        pass
                    log(f"   [roam] 🚪 QUESTLINE DEEPER: {mp0} -> {tv.map_id(self.b)} via warp {wt} "
                        f"(the target's further in)")
                    return "questline_deeper"
            # nobody left to talk + no deeper rooms → wrong building; release the 'stay inside' marker
            # so the blackout-recovery can exit her, and keep looking at the next candidate building.
            # DIAGNOSTIC (night shift 9): if UNVISITED interior warps remain but were all skipped for
            # walk-unreachability, this is a feet-partition/NPC-blocked-stairs blocker, not a truly
            # exhausted building — surface it loudly so the next fix targets reachability, not the tour.
            _left_new = [(t, _dest_of.get(t)) for t in cand
                         if not self.world.visited(_dest_of.get(t, (0, 0)))]
            if _left_new:
                log(f"   [roam] questline: LEAVING with UNVISITED deeper warps still skipped "
                    f"(reachability blocker?): {_left_new}")
            log("   [roam] questline: no one left to talk to in here and the flag's not set — leaving to keep looking")
            # WRONG-ENTRANCE detection (sabrina_run4): occupants VISIBLE on this map but with NO
            # walkable path to them = a sealed sub-region — same building, DIFFERENT entrance
            # (Celadon Mansion: the back door spawns in the stairwell, the tea room only opens to
            # the front doors). Remember the interior map; the city door pickers then prefer OTHER
            # overworld doors warping into it.
            try:
                _gi = tv.Grid(self.b)
                _hi = tuple(tv.coords(self.b))
                _unreach = [c for _i, c, _f in self._talkable_npcs()
                            if not tv.bfs(_gi, _hi, lambda t, cc=c:
                                          abs(t[0] - cc[0]) + abs(t[1] - cc[1]) == 1,
                                          walkable=_gi.walkable)]
                if _unreach:
                    self._ql_sibling_dest = mp0
                    log(f"   [roam] 🚪 QUESTLINE WRONG-ENTRANCE: {len(_unreach)} occupant(s) on "
                        f"{mp0} are VISIBLE but UNREACHABLE (sealed room, e.g. {_unreach[0]}) — "
                        f"will try a different door into this building")
            except Exception as _we:
                log(f"   [roam] wrong-entrance check skipped: {_we}")
            self._ql_room_sweeps = 0
            self._exit_to_overworld()
            # still interior after the exit hop ⇒ we're in a multi-room COMPLEX (the ship): this
            # room was a toured-out hub, not a wrong BUILDING — the tour continues from the outer
            # room next tick. Keep the deliberate-inside marker in that case, or the blackout
            # recovery ejects her off the ship with a false whiteout (run 15's terminal spiral).
            self._ql_inside_target = tuple(tv.map_id(self.b))[0] != 3
            if not self._ql_inside_target:
                # the whole BUILDING failed — burn the city door we entered by so neither picker
                # walks back into it this questline (the back-door re-entry loop)
                _cd = getattr(self, "_ql_city_door", None)
                if _cd:
                    self._ql_entered_doors.add(_cd)
                    self._ql_city_door = None
                    log(f"   [roam] questline: burned failed entrance {_cd[1]} on {_cd[0]}")
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
                self._ql_room_sweeps = 0
                self._ql_inside_target = True       # tell the blackout-recovery to LEAVE HER inside (she's
                log(f"   [roam] 🚪 QUESTLINE ENTER: stepped into the building at door {door}")  # here on purpose
                return "questline_entered"
            log(f"   [roam] questline: tried to enter the building at {door} but didn't warp — surfacing")
            return "questline_enter_failed"
        # DISTANT DOOR (2026-07-09, the Bill's-Sea-Cottage class): no door is reachable FROM HER FEET,
        # but the target building may sit ACROSS this map — Route 25's cottage is at the far EAST while
        # the Nugget Bridge deposits her at the WEST edge (0,9). _questline_unentered_door's live BFS only
        # sees doors reachable from her current tile (windowed), so it returned None; but _door_tiles scans
        # the WHOLE loaded layout, so the distant cottage door IS known. CLOSE THE DISTANCE: travel toward
        # the nearest un-entered door on this map, then the door-enter above fires next tick from the near
        # side (she was giving up + bouncing Cerulean<->Route 25 to a STALL). General fix for the "arrived on
        # the right map but the building is across it" class (route-with-a-cottage / far-gatehouse fetches).
        # Self-bounding: commits ONLY when the walk actually MOVES her — a no-move falls straight through to
        # the talk/surface path so a genuinely unreachable door never spins (the _ql_fp guard also caps it).
        if door is None:
            try:
                cur = tuple(tv.map_id(self.b))
                co = tv.coords(self.b)
                _nc2 = self._no_connector_maps()
                _wdest2 = {tuple(xy): tuple(dest) for (xy, dest, _wid) in tv.read_warps(self.b)}
                _doors_here = [tuple(dr) for dr in self._door_tiles()
                               if _wdest2.get(tuple(dr)) not in _nc2]
                far = [dr for dr in _doors_here if (cur, dr) not in self._ql_entered_doors]
                # RE-ENTRY AFTER AN INTERRUPTED VISIT (2026-07-09, the Bill-cottage door-burn): every
                # door on the arrival map is already ENTERED but the quest flag is STILL unset — she
                # entered the target building, then a Route-25 gauntlet blackout/heal yanked her out
                # BEFORE she talked Bill, and the entered-doors burn then locked _questline_unentered_door
                # + the approach out of it forever (arrived_no_target ping-pong). The flag gates
                # completion, so re-entering the RIGHT building is harmless — CLEAR this map's burn for a
                # clean retry (the room-tour + wrong-building logic re-burn a genuinely wrong one; the
                # _ql_fp no-progress guard caps a truly stuck retry). This is what lets the leveling she
                # banks each cycle eventually carry her through the crossing to finish Bill.
                if not far and _doors_here:
                    _burned = {k for k in self._ql_entered_doors if k[0] == cur}
                    if _burned:
                        self._ql_entered_doors -= _burned
                        log(f"   [roam] questline: all {len(_doors_here)} door(s) on {self.world.name(cur)} "
                            f"already entered but the flag's unset — an interrupted visit; clearing the "
                            f"burn {sorted(_burned)} for a clean re-approach")
                        far = list(_doors_here)
            except Exception as _dd:
                far, co = [], None
                log(f"   [roam] questline distant-door scan skipped: {_dd}")
            if far and co is not None:
                far.sort(key=lambda t: abs(t[0] - co[0]) + abs(t[1] - co[1]))
                tgt = far[0]
                before = tuple(tv.coords(self.b))
                self.on_event("the place I'm after is further along here — let me get to it.",
                              kind="route", tier=1)
                self.trav.travel(target_map=None, arrive_coord=(tgt[0], tgt[1] + 1),
                                 max_steps=400, max_seconds=300)
                if tuple(tv.coords(self.b)) != before:
                    log(f"   [roam] 🧭 QUESTLINE APPROACH: crossing {self.world.name(cur)} toward the "
                        f"un-entered door {tgt} (the target building is across this map)")
                    return "questline_approaching"
                log(f"   [roam] questline: un-entered door {tgt} on {cur} unreachable from {before} — "
                    f"falling through to talk/surface")
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

    def _no_connector_maps(self):
        """KB no_connector maps as a set of (g,n) tuples — warp MAZES (Rock Tunnel) that generic
        touring/pass-through must never enter; the dedicated strikes own those crossings."""
        try:
            return {tuple(int(x) for x in k.split(","))
                    for k in (self._gate_recognizer.kb.get("no_connector") or {}).get("maps", [])}
        except Exception:
            return set()

    def _questline_interior_route(self, step):
        """GAME-KNOWLEDGE (gamedata/frlg_gates.json 'interior_routes'): the next warp TILE toward a
        DEEP questline target the blind GO-DEEPER cabin tour can't reliably reach — the S.S. Anne
        captain (HM01 Cut) sits on 2F, several warps past 1F's dead-end cabins, and four shifts of
        tour-ranking tweaks never climbed there (the 2F stairs are a 'vis' node in the canonical
        world model, so they sorted behind every NEW cabin, and she bailed to the exit lobby first).
        Returns (x,y) of the warp to step onto FROM the current map, or None (not this errand /
        already at the target / current map not on the chain). Deterministic + world-model-independent
        → works on a fresh timeline. Coords isolated in gamedata (rule 14 — engine just follows them)."""
        try:
            if not (step and step.success and step.success[0] == "cap"):
                return None
            rt = (self._gate_recognizer.kb.get("interior_routes") or {}).get(step.success[1])
            if not rt:
                return None
            here = ",".join(str(x) for x in tuple(tv.map_id(self.b)))
            if here == rt.get("target"):
                return None   # already in the target room — let the normal talk flow take the captain
            tile = (rt.get("hops") or {}).get(here)
            if not tile:
                return None
            g = tile.split(",")
            return (int(g[0]), int(g[1]))
        except Exception:
            return None

    def _questline_strike(self, step):
        """BESPOKE INTERIOR STRIKE dispatcher (night shift 7). Some questline targets sit behind a
        fixed dungeon RITUAL the general room-tour/GO-DEEPER can't crack — registered by the step's
        success signature. Runs the proven strike as ONE decision (like beat_gym); returns a roam-
        result string when it ran, or None to fall through to the general interaction layer. Bounded
        by a per-errand try counter (a failing strike can't loop the run forever)."""
        try:
            if not (step and step.success):
                return None
            succ = tuple(step.success)
            here = tuple(tv.map_id(self.b))
            # STRIKE REGISTRY — success-sig -> (label, done-msg, importer). The importer returns
            # (run_fn, anchor_maps, good_results) so the strike modules stay lazily imported.
            def _hideout():
                from hideout_strike import HIDEOUT_MAPS, GC, CELADON, run_strike
                return run_strike, (HIDEOUT_MAPS | {GC, CELADON}), ("got_scope",), "hideout_probe"

            def _tower():
                from tower_strike import TOWER_MAPS, LAVENDER, run_strike
                return run_strike, (TOWER_MAPS | {LAVENDER}), ("got_flute",), "tower_probe"

            def _snorlax():
                from snorlax_strike import ANCHORS, run_strike
                return run_strike, ANCHORS, ("woke_snorlax",), "snorlax_probe"

            def _silph():
                from silph_strike import SAFFRON, SILPH_MAPS, run_strike
                return run_strike, ({SAFFRON} | SILPH_MAPS), ("freed_saffron",), "silph_probe"

            registry = {
                ("item", 359): ("Rocket Hideout (Silph Scope)",
                                "got it — the Silph Scope. now for that ghost in the tower.", _hideout),
                ("item", 350): ("Pokémon Tower (Poké Flute)",
                                "we saved Mr. Fuji — and he handed us the Poké Flute. now to wake "
                                "that Snorlax.", _tower),
                ("flag", "FLAG_WOKE_UP_ROUTE_12_SNORLAX"): (
                    "Route 12 Snorlax (Poké Flute wake)",
                    "the Snorlax is awake — and the road south to Fuchsia is finally open.", _snorlax),
                ("flag", "FLAG_HIDE_SAFFRON_ROCKETS"): (
                    "Silph Co. liberation (free Saffron / unblock Sabrina)",
                    "we did it — Silph Co. is free, Team Rocket's boss is beaten, and Saffron's ours. "
                    "now Sabrina's gym is finally open.", _silph),
            }
            if succ not in registry:
                return None
            label, done_msg, importer = registry[succ]
            run_fn, anchors, good, dbg_sub = importer()
            if here not in anchors:
                return None
            tries_map = getattr(self, "_ql_strike_tries_map", None)
            if tries_map is None:
                tries_map = self._ql_strike_tries_map = {}
            tries = tries_map.get(succ, 0)
            if tries >= 3:
                log(f"   [roam] questline STRIKE: 3 attempts exhausted for {label} — surfacing so "
                    f"recovery/other options carry her (not looping the strike)")
                return None
            tries_map[succ] = tries + 1
            log(f"   [roam] 🎯 QUESTLINE STRIKE: {label} — attempt {tries + 1}/3 from {here}")
            dbg = os.path.join(os.environ.get("TEMP", _HERE), "longrun", dbg_sub)
            res = run_fn(self, log, dbg_dir=dbg)
            log(f"   [roam] 🎯 QUESTLINE STRIKE -> {res} (now {tv.map_id(self.b)}@{tv.coords(self.b)})")
            if res in good:
                # objective in bag. The deriver re-reads the flag/item next tick and the step self-
                # clears -> questline advances. Clear the inside marker + reset this errand's tries.
                self._ql_inside_target = False
                tries_map[succ] = 0
                self.on_event(done_msg, kind="milestone", tier=2)
                return "questline_strike_done"
            if res in ("in_hideout", "in_silph"):
                # objective obtained (flag/item set) but still INSIDE the dungeon — keep the inside
                # marker so recovery doesn't eject her mid-dungeon; retry the exit next tick. (The
                # deriver already reads the flag as satisfied, so the questline self-clears next tick;
                # this just walks her out cleanly rather than dumping her at the boss floor.)
                self._ql_inside_target = True
                return "questline_strike_exit_wip"
            if res == "not_here":
                return None
            return "questline_strike_failed"
        except Exception as e:
            log(f"   [roam] questline STRIKE errored: {e} — falling through to the general layer")
            return None

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
        # NO-CONNECTOR guard (2026-07-07, flute_run1): the Rock Tunnel mouth is a "door" by tile
        # behavior — the quest tour entered it hunting Mr. Fuji and wandered the maze to a STALL.
        # A door whose warp destination is a billed no-connector maze is never a quest building.
        try:
            _nc = self._no_connector_maps()
            _wdest = {tuple(xy): tuple(dest) for (xy, dest, _wid) in tv.read_warps(self.b)}
        except Exception:
            _nc, _wdest = set(), {}
        cands = []
        for dr in self._door_tiles():
            if (cur_map, tuple(dr)) in self._ql_entered_doors:
                continue
            if _wdest.get(tuple(dr)) in _nc:
                continue
            approach = (dr[0], dr[1] + 1)               # stand just SOUTH of the door, step UP to enter
            if tv.bfs(grid, co, lambda t, a=approach: t == a, walkable=grid.walkable):
                cands.append(tuple(dr))
        if not cands:
            return None
        # WRONG-ENTRANCE RECOVERY: doors into the interior that held a visible-but-unreachable
        # occupant jump the queue (same building, other entrance — see _questline_interact)
        _sib = getattr(self, "_ql_sibling_dest", None)
        cands.sort(key=lambda t: (0 if (_sib and _wdest.get(t) == _sib) else 1,
                                  abs(t[0] - co[0]) + abs(t[1] - co[1])))
        return cands[0]

    def _gym_prereq_gate(self, gym):
        """A gym whose DOOR is STORY-LOCKED (not an HM obstacle) — return a synthesized STORY_NPC Gate
        whose questline is the liberation dungeon, when its prereq flag is UNMET. This is the wire the
        HM-only `_gym_gate_probe` was missing for Sabrina (Rocket-blocked until Silph Co. clears): there's
        no tree to recognize at her feet, so the probe returned None and head_to_gym structurally parked.
        Data-driven off GYM_PREREQS; the Gate.missing keys the KB capability (a no-door strike step) so the
        registered dungeon strike fires. Returns a Gate or None (already cleared / no prereq for this gym)."""
        spec = GYM_PREREQS.get(gym.name)
        if not spec:
            return None
        flag_id, missing_key, human = spec
        try:
            if fm.read_flag(self.b, flag_id):
                return None                     # LIVE cross-check: already liberated -> no gate
        except Exception:
            return None
        return ql.Gate(ql.STORY_NPC, missing=missing_key, where=tuple(gym.city), human=human,
                       detail={"flag": missing_key, "gym": gym.name})

    def _gym_gate_probe(self, gym):
        """beat_gym couldn't enter: walk as CLOSE to the gym door as the map allows (the BFS stops
        at the blocking obstacle — Vermilion's cut tree IS the chokepoint), then run the HM-obstacle
        recognizer AT HER FEET. This was the missing wire: recognize() was only ever called with
        blocked_dir (exit gates), so the adjacency-based HM_OBSTACLE gate was dead code from the
        roam loop. Returns a Gate or None."""
        try:
            grid = tv.Grid(self.b)
            cur = tuple(tv.coords(self.b))
            door = tuple(gym.door)
            if not hasattr(self, "_gym_cut_cache"):
                self._gym_cut_cache = {}     # map_id -> gym-gate cut-tree coord (shift 17)
            # phase A — SCAN FIRST: beat_gym's failed door approach already parked her near the
            # yard, where the fence-gap TREE OBJECT is loaded. Walking "closer to the door" first
            # is actively harmful — the Grid is OPTIMISTIC about fences (static obstacles are only
            # discovered by bonking them), so a distance-first walk drifted to the BEACH and armed
            # a SURF/Safari questline (runs 5-6). A tree/boulder near the door IS the gate: walk
            # adjacent to the OBJECT and recognize there (obstacles outrank water in the recognizer).
            import field_moves as _fm

            def _obstacle_probe():
                g2 = tv.Grid(self.b)
                here2 = tuple(tv.coords(self.b))
                # DIAGNOSTIC (2026-07-10 NIGHT SHIFT 16): the Surge gym-door Cut never actuates even on the
                # first stuck — instrument scan/walk/can_use so the next run reveals WHICH stage fails.
                _objs = list(_fm.scan_field_objects(self.b, {_fm.GFX_CUT_TREE, _fm.GFX_BOULDER}))
                log(f"   [roam] 🔎 OBSTACLE-PROBE: at {here2}, door {door}, scanned {len(_objs)} field object(s): "
                    f"{[(o.get('coord'), o.get('gfx')) for o in _objs][:8]}")
                for ob in _objs:
                    ox, oy = ob["coord"]
                    if abs(ox - door[0]) + abs(oy - door[1]) > 12:
                        continue                                 # unrelated obstacle across the map
                    if ob["gfx"] == _fm.GFX_CUT_TREE:
                        # CACHE the gym-gate tree the moment it's scanned near the door (shift 17).
                        # The pre-Cut approach reaches it adjacent; a LATER post-Cut re-entry lands
                        # far away (Vermilion (29,18), ~16 tiles off) where scan_field_objects loads
                        # NOTHING — the cache lets phase B walk straight back to it.
                        self._gym_cut_cache[tuple(tv.map_id(self.b))] = (ox, oy)
                    for stand in ((ox, oy - 1), (ox, oy + 1), (ox - 1, oy), (ox + 1, oy)):
                        _reach = bool(tv.bfs(g2, here2, lambda t, s=stand: t == s, walkable=g2.walkable))
                        log(f"   [roam] 🔎 tree@{ob['coord']} gfx={ob.get('gfx')} stand={stand} bfs_reachable={_reach}")
                        if _reach:
                            self.trav.travel(target_map=None, arrive_coord=stand,
                                             max_steps=200, max_seconds=90)
                            _arr = tuple(tv.coords(self.b) or ())
                            _cu = _fm.can_use(self.b, "cut", self.b.rd8(ram.GPLAYER_PARTY_CNT))
                            log(f"   [roam] 🔎 walked to stand -> at {_arr} (target {stand}, arrived={_arr == stand}); "
                                f"can_use(cut)={_cu} field={self.field is not None} gfx_is_tree={ob['gfx'] == _fm.GFX_CUT_TREE}")
                            if tuple(tv.coords(self.b) or ()) == stand:
                                # CAPABILITY IN HAND (surge run 4): a cut tree + a party mon
                                # that KNOWS Cut = clear it RIGHT HERE — the stage-3 automatic
                                # chain. Arming a gate instead makes the recognizer fall
                                # through to the WATER beside the gym and open a Safari-Zone
                                # surf errand that poisons her context ('blocked by water' —
                                # the oracle then never picks use_cut).
                                if (ob["gfx"] == _fm.GFX_CUT_TREE and self.field is not None
                                        and _fm.can_use(self.b, "cut",
                                                        self.b.rd8(ram.GPLAYER_PARTY_CNT))):
                                    face = {(0, -1): "UP", (0, 1): "DOWN",
                                            (-1, 0): "LEFT", (1, 0): "RIGHT"}[
                                        (ox - stand[0], oy - stand[1])]
                                    self.on_event("that little tree's right in the way — Cut "
                                                  "time. TIMBER!", kind="field", tier=2)
                                    res = self.field.clear_obstacle("cut", face)
                                    log(f"   [roam] auto use_cut at {ob['coord']} "
                                        f"(face {face}) -> {res}")
                                    if res == "used":
                                        return "cleared"
                                return True
                return False

            found = _obstacle_probe()
            if found == "cleared":
                return None      # tree is DOWN — no gate; the next tick's door leg walks through
            if not found:
                # phase B — reposition so the gate tree LOADS into scan range, then rescan.
                grid2 = tv.Grid(self.b)
                cur2 = tuple(tv.coords(self.b) or cur)
                moved = False
                # (i) CACHED-TREE WALK (shift 17, the post-Cut Vermilion fix): a prior (pre-Cut)
                #     approach scanned the gym-gate tree adjacent and cached its coord. Post-Cut she
                #     re-enters FAR away (e.g. (29,18)) with an EMPTY scan — "reposition toward the
                #     DOOR" is useless because the door-approach tile is INSIDE the fenced yard,
                #     behind the tree, and unreachable until the tree is cut. Walk to a reachable
                #     stand tile beside the CACHED tree instead: the object then loads and the
                #     rescan's auto-cut fires. Robust to disconnected regions (dock↔town) — logs
                #     the BFS reachability so a still-stuck case is self-diagnosing.
                cached = self._gym_cut_cache.get(tuple(tv.map_id(self.b)))
                if cached:
                    cx, cy = cached
                    try:
                        self.b.frame_rgb().save(os.path.join(
                            os.environ.get("LONGRUN_DUMP_DIR", "/g/temp"),
                            "gym_gate_phaseB.png"))
                    except Exception:
                        pass
                    for stand in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)):
                        _r = bool(tv.bfs(grid2, cur2, lambda t, s=stand: t == s,
                                         walkable=grid2.walkable))
                        log(f"   [roam] 🔎 phase B cached-tree {cached} stand {stand} "
                            f"bfs_reachable={_r} (at {cur2})")
                        if _r:
                            log(f"   [roam] 🔎 phase B: walking to cached gym-tree stand {stand}")
                            self.trav.travel(target_map=None, arrive_coord=stand,
                                             max_steps=300, max_seconds=90)
                            moved = True
                            break
                # (ii) FALLBACK — reposition toward the door (bounded) so nearer objects LOAD
                if not moved:
                    for d in range(2, 11, 2):
                        path = tv.bfs(grid2, cur2,
                                      lambda t, dd=d: abs(t[0] - door[0]) + abs(t[1] - door[1]) <= dd,
                                      walkable=grid2.walkable)
                        if path:
                            self.trav.travel(target_map=None, arrive_coord=tuple(path[-1]),
                                             max_steps=200, max_seconds=60)
                            break
                if _obstacle_probe() == "cleared":
                    return None
            here = tv.coords(self.b)
            gate = self._gate_recognizer.recognize(tuple(tv.map_id(self.b)),
                                                   player_xy=tuple(here) if here else None,
                                                   grid=tv.Grid(self.b))
            if gate:
                try:
                    log(f"   [roam] [!] GYM-GATE PROBE: {gate}")
                except Exception:
                    pass
            return gate
        except Exception as _e:
            log(f"   [roam] gym-gate probe skipped: {_e}")
            return None

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
            # FORWARD DIRECTION = the billed road leg under her feet when one exists (descent re-grade
            # 2026-07-08: at Lavender the road to Celadon runs WEST, but the hardcoded "south" armed the
            # Route-12 SNORLAX questline every tick, hijacking head_to_gym off the billed road — the
            # banked_ROCKTUNNEL stand-still). "south" remains the fallback for road-less spine geometry.
            fwd = "south"
            try:
                _road = self._gym_road(state.get("next_gym"))
                if _road:
                    _leg = next((l for l in _road if l["map"] == cur_map and l.get("go")), None)
                    if _leg:
                        fwd = _leg["go"]
            except Exception:
                pass
            gate = self._gate_recognizer.recognize(cur_map, blocked_dir=fwd)
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

    def _shopping_list(self, foresight=False, state=None):
        """What a sensible player would BUY here, given the bag + what's hurt her: top Potions up to the
        target (SHOP_POTION_FORESIGHT when she's walled and stocking up before a push, else
        SHOP_POTION_TARGET), and SHOP_CURE_QTY of the cure for each status seen recently (_afflict_seen).
        Bounded quantities (survival, not hoarding); returns [(item_id, qty), ...] (empty = well-stocked).
        NS#42: when a coverage keeper is DUE (state given + _keeper_due), stock a fuller ball pocket even at
        party>3 so the hunt isn't ball-starved (the plain thin-team-only ball floor left a party-4 hunter short)."""
        sl = []
        target = SHOP_POTION_FORESIGHT if foresight else SHOP_POTION_TARGET
        # count ALL healing-potion tiers she already carries (a Super Potion is still "a potion" for stock)
        have = sum(self.bag_count(i) for i in (ITEM_POTION, 22, 21, 20, 19))
        pot_need = target - have
        if pot_need > 0:
            sl.append((self._best_potion_for_sale(), pot_need))
        # BATCH 6 PHASE 3 — Poké Ball foresight: thin team + low on balls = she should leave the Mart
        # equipped to actually catch a teammate (the real answer to a wall). Bag-delta verified like any buy.
        # 2026-07-07 BALL FLOOR: catching is CENTRAL (dex doctrine) — a full-ish team doesn't excuse
        # walking out of a Mart with an empty ball pocket. Never leave with fewer than 2.
        keeper_due = bool(state) and self._keeper_due(state)
        ball_target = SHOP_BALL_KEEPER_TARGET if keeper_due else SHOP_BALL_TARGET
        if (self._thin_team() or self._ball_count() < 2 or keeper_due) and self._ball_count() < ball_target:
            sl.append((ITEM_POKE_BALL, ball_target - self._ball_count()))
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
        tiers = ((20, 2500), (21, 1200), (22, 700), (ITEM_POTION, 300))   # Max/Hyper/Super/Potion
        for iid, price in tiers:
            if iid in stock and (money - SHOP_MONEY_FLOOR) >= price * 3:
                return iid
        # SOLD-HERE FALLBACK (surge run 1): with 2044 in the wallet the 3x-comfort check failed
        # every tier and the old fallback returned a plain Potion — which Vermilion DOESN'T SELL
        # ('shop_failed' -> the surfacer re-offered stock_up forever -> a 14-tick STALL at the
        # Mart door). Buy the strongest tier that's ON THE SHELF and affordable for at least ONE.
        for iid, price in tiers:
            if iid in stock and (money - SHOP_MONEY_FLOOR) >= price:
                return iid
        return ITEM_POTION

    def _shop_note(self, foresight=False, state=None):
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
        # NS#42: narrate the keeper pre-stock ('grabbing balls for the diglett I'm about to go hunt') so
        # the buy reads as purposeful, not incidental — the constitution's grind/shop-with-narrated-reason.
        keeper_due = bool(state) and self._keeper_due(state)
        if (self._thin_team() or keeper_due) and self._ball_count() < (
                SHOP_BALL_KEEPER_TARGET if keeper_due else SHOP_BALL_TARGET):
            bits.append("you're light on Poké Balls — grab a good stock so you can actually catch the "
                        "teammate your plan wants" if keeper_due else
                        "you're light on Poké Balls — grab some so you can actually catch a teammate out there")
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

    def _party_pids(self):
        """(pid, species) per party slot — set-comparable (immune to the menu-time order law:
        a reorder shuffles slots, never the PID set). Bounded RAM read; safe at tick top."""
        out = []
        cnt = min(self.b.rd8(ram.GPLAYER_PARTY_CNT), 6)
        for s in range(cnt):
            base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
            out.append((self.b.rd32(base), st.read_party_species(self.b, s)))
        return out

    def _meet_new_teammates(self, state):
        """PHASE C-2 — the UNMET-TEAMMATE watcher (Lapras first-field, generalized). Each tick, diff
        the party PID set against everyone she's met. First tick baselines silently (present = met).
        A NEW pid whose arrival wasn't a witnessed catch (roster_react marks those met itself) gets a
        real INTRODUCTION: a neutral meet event -> her voice, her own name for them (soul oracle, the
        proven catch-naming pattern), a family bond via soul.note_met. A returning species that
        already has a bond = a smaller REUNION beat, not a re-introduction."""
        try:
            cur = self._party_pids()
        except Exception as _pe:
            log(f"   [meet] party read skipped: {_pe}")
            return
        pids = {p for p, _ in cur}
        if self._met_pids is None:
            # FIRST TICK BASELINE. Members already aboard are travel companions, not arrivals —
            # BUT (summit-canonical ground truth, 2026-07-07): bonds are sparse/stale (venusaur/
            # raticate pre-date the bond system; spearow's bond never followed the fearow
            # evolution; the gifted LAPRAS rode in the party unbonded, not in Bill's PC). So:
            # a boot-time member with GIFT_BACKSTORY and no bond IS the unmet-gift set-piece
            # (fires ONCE — the intro writes the bond); any other unbonded member gets a SILENT
            # bond backfill (data hygiene, not a beat — she's been traveling with them for ages).
            self._met_pids = set(pids)
            if self.soul is None:
                return
            from pokemon_soul import GIFT_BACKSTORY
            for pid, sp in cur:
                name = st.SPECIES_NAME.get(sp, "")
                if not name or self._bond_for_species(name):
                    continue
                if name.lower() in GIFT_BACKSTORY:
                    self._introduce_teammate(name, GIFT_BACKSTORY[name.lower()])
                else:
                    self.soul.bonds[name.lower()] = {"species": name, "nickname": None,
                                                     "caught": None,
                                                     "note": "longtime teammate (bond backfilled)"}
                    log(f"   [meet] bond backfilled silently for longtime teammate {name}")
            return
        fresh = [(p, sp) for p, sp in cur if p not in self._met_pids]
        self._met_pids |= pids
        if not fresh or self.soul is None:
            return
        from pokemon_soul import GIFT_BACKSTORY
        for pid, sp in fresh:
            name = st.SPECIES_NAME.get(sp, "a new Pokémon")
            bonded = self._bond_for_species(name)
            if bonded:                                    # an old friend re-fielded -> reunion, not intro
                who = bonded.get("nickname") or name
                log(f"   [meet] REUNION: {who} ({name}) back in the party")
                self.on_event(f"{who} is back in the party — good to have you back out here.",
                              kind="roster", tier=2)
                continue
            log(f"   [meet] INTRODUCTION: unmet teammate {name} (pid={pid:08x}) joined the party")
            self._introduce_teammate(name, GIFT_BACKSTORY.get(name.lower()))

    def _bond_for_species(self, name):
        """The soul bond dict for a species name, or None. Species-matched (bonds may be keyed
        by nickname); case-insensitive."""
        try:
            for _k, v in (self.soul.bonds or {}).items():
                if isinstance(v, dict) and (v.get("species") or "").lower() == (name or "").lower():
                    return v
        except Exception:
            pass
        return None

    def _introduce_teammate(self, name, how):
        """The INTRODUCTION arc (shared: boot-time unmet gift + post-boot non-catch arrival):
        a neutral meet beat -> her voice, her own name for them (soul oracle, the proven
        catch-naming pattern; soul-side mental name, NOT the in-game nickname keyboard), a
        family bond via note_met, banked immediately."""
        self.on_event(f"a {name} just joined your team — "
                      + (how if how else "you two haven't properly met until now")
                      + ". this is your first real moment together.", kind="roster", tier=3)
        nick = name
        try:
            nm = self._soul_choose("name", {}, {"place":
                f"a {name} just joined your team — "
                + (how if how else "your first time actually meeting")
                + f". give this {name} a name, just the name."})
            if nm:
                nick = nm.strip().split("\n")[0][:12] or name
        except Exception as _ne:
            log(f"   [meet] naming skipped: {_ne}")
        self.soul.note_met(name, nick, how)
        self._continuity_save()                           # the new bond survives a kill right now

    def roster_react(self):
        """SOUL beat after a hand-play-banked catch (the agent can't drive this core's battle menus,
        so acquisition is hand-played — but the RELATIONSHIP is live). Emit a NEUTRAL roster-change
        event so Kira reacts to her new teammate IN HER OWN VOICE via the existing seam (fulfilling
        her want for a partner). Touches NO core (on_event -> _pokemon_react -> her self)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        # P-1(a): the FIRST CATCH is a life event, not a roster update.
        self._first_beat("catch",
                         "I CAUGHT it. I actually caught it — my first ever. "
                         "I have a TEAM now. we're a team.")
        # PHASE C-2 guard: a witnessed catch IS the meeting — mark the whole current party met so the
        # unmet-teammate watcher never re-introduces someone she just caught and celebrated.
        try:
            if self._met_pids is not None:
                self._met_pids |= {p for p, _ in self._party_pids()}
        except Exception:
            pass
        if cnt <= 1:
            log("   ROSTER_REACT: party size 1 — no new teammate to react to"); return "reacted"
        name = st.SPECIES_NAME.get(st.read_party_species(self.b, cnt - 1), "a new Pokemon")
        log(f"   ROSTER_REACT: party now {cnt} — newest teammate is {name}")
        # BATCH 5 PHASE 4 — ROSTER AS FAMILY: a catch is a real moment, not a stat line. She NAMES the
        # new teammate IN CHARACTER (the soul oracle, same pattern as the evolution naming), and it's
        # recorded as a family member whose name + her opinion PERSIST (soul.bonds -> continuity, and the
        # whole campaign anchors via Phase-1 save). Naming is hers; headless/no-oracle -> species name.
        nick = name
        place = self._place_name(tv.map_id(self.b), default=None)
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
        # DEX DOCTRINE (2026-07-06): the Pokédex is her AMBIENT PRIDE STAT — one beat at a new
        # registration, a bigger one at round numbers. Never a lecture, never a grind target.
        try:
            dex = ram.pokedex_owned_count(self.b)
            if dex and dex != getattr(self, "_last_dex_count", None):
                self._last_dex_count = dex
                if dex % 10 == 0:
                    self.on_event(f"and that makes {dex} in the Pokédex — {dex}! we're really "
                                  f"building something here.", kind="roster", tier=2)
                else:
                    self.on_event(f"Pokédex ticked up — {dex} kinds caught now.", kind="roster", tier=1)
        except Exception:
            pass
        else:
            self.on_event(f"you've got a new teammate now — a {name} that's going to fight alongside you")
        return "reacted"

    # ── SOUL hook router (MODE-layer; reads RAM + calls soul hooks; never touches core mood/bond) ──
    def _objective_distance(self, state):
        """F-1 — how far (in learned-graph map hops) her STANDING objective's destination is from
        here. travel:g,n and head_to_gym are the nav objectives; anything else (or an unroutable /
        unknown target) returns None, which DISARMS the wander tripwire (never trip on what we
        can't measure). 0 = she's standing on the destination map."""
        act = (self._active_objective or {}).get("action") or ""
        cur = tuple(state.get("map") or ())
        if not cur:
            return None
        if act.startswith("travel:"):
            try:
                g, n = act.split(":", 1)[1].split(",")
                tgt = (int(g), int(n))
            except Exception:
                return None
        elif act == "head_to_gym":
            tgt = self._next_gym_city_map(state.get("next_gym"))
        else:
            return None
        if not tgt:
            return None
        if cur == tuple(tgt):
            return 0
        try:
            r = self.world.route(cur, tuple(tgt), avoid=self._wall_avoid(state))
        except Exception:
            return None
        return (len(r) - 1) if r else None

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

    def _surf_usable(self):
        """True when Surf would actually work (known + badge) — the gate for water-aware
        reachability layers. Fail-CLOSED on read flakes: a false negative degrades to the old
        land-only behavior, a false positive would bill unreachable water roads as roads."""
        try:
            import field_moves as _fm
            return bool(_fm.can_use(self.b, "surf"))
        except Exception:
            return False

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
        # WATER-START (night shift 10, the Route 21 wander ping-pong): a surf-mounted stand is a
        # WATER tile — the land-only BFS dies at its own start and every grass on a sea route reads
        # "unreachable", so wander_catch bounced between the two Route 21 halves all window. With
        # Surf usable, plan over land+water (the executor owns mount/dismount).
        walk = g.walkable_or_surf if self._surf_usable() else g.walkable
        path = tv.bfs(g, co, lambda t: t in playable, walkable=walk)
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
            self.world.note_visit(cur, name=self._place_name(cur, default=None), live_traits=live_traits,
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
        dead = (getattr(self, "_grind_dead", set())        # maps grind() proved grassless/strand-only
                | getattr(self, "_grind_inadequate_set", set()))   # + level-aware inadequate (empty if flag off)
        def has_grass(m):
            return ((m in self._grass_maps or self._is_route_map(m))
                    and m != cur and tuple(m) not in dead)
        unreach = getattr(self, "_grass_unreach", set())   # koga_run3: hops that failed from here
        seen = {cur}
        q = deque()
        for edge, nbr in self._conn_graph.get(cur, {}).items():
            if nbr in seen or gated(nbr) or (cur, tuple(nbr)) in unreach:
                continue                              # never step toward the wall / a proven no-route
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
        # GRIND-SPOT LEVEL AWARENESS (NS#5 lever a): if grind() marked THIS map grind-inadequate (only
        # done when a reachable higher-level spot exists), don't short-circuit on grass-underfoot — fall
        # through to the route branches below (which exclude the inadequate set) so she actually routes to
        # the better spot instead of re-picking the same near-0-XP grass. Empty set when flag off -> no change.
        if tile is not None and tuple(state.get("map") or ()) not in getattr(self, "_grind_inadequate_set", set()):
            return ("here", tile)
        pcount = state.get("party_count")
        plevel = state["party"][0]["level"] if state.get("party") else None
        # Batch-WORLD (Phase 4b) — prefer grass she KNOWS: the nearest VISITED grass she can reach
        # WITHOUT crossing a wall (adjacent or several hops BEHIND her). This is the missing verb — she
        # walks back to Route 4 / Mt Moon that she's already cleared, instead of blindly routing onto
        # the unvisited route across Gary's bridge (which the old live-connection scan below would pick).
        unreach = getattr(self, "_grass_unreach", set())   # koga_run3: (from-map, target) travel fails
        try:
            cur = tuple(state["map"])
            avoid = self._wall_avoid(state)
            known = self.world.reachable_with_trait(cur, "has_grass", avoid)
            _dead = (getattr(self, "_grind_dead", set())   # maps grind() proved grassless/strand-only
                     | getattr(self, "_grind_inadequate_set", set()))  # + inadequate (empty if flag off)
            for _entry in (known or []):                   # try EVERY known grass, not just the nearest
                dst = _entry[0]                            # entries are (dst, ...) — width varies
                if tuple(dst) in _dead:
                    continue                               # visited trait says grass; grind says dead
                hop = self.world.next_hop(cur, dst, avoid)
                if not hop:
                    continue
                nxt, edge = hop
                if (cur, tuple(nxt)) in unreach:
                    continue                               # this hop already failed from here — next
                log(f"   [roam] grass she KNOWS: routing {edge} -> {nxt} toward {self.world.name(dst)} "
                    f"(visited grass, avoiding any wall)")
                return ("route", nxt, edge)
        except Exception as _gt:
            log(f"   [roam] world grass-target skipped: {_gt}")
        routes = [(d, (grp, num)) for d, (grp, num) in self._map_connections()
                  if grp == 3 and num >= 19]
        # BATCH-4 PHASE 2 (route AROUND the wall): prefer a grass route that ISN'T gated by an active
        # wall. Adjacent ungated route wins (this already includes grass directly behind her).
        cur = tuple(state["map"])
        _dead = (getattr(self, "_grind_dead", set())       # proven-grassless maps aren't candidates
                 | getattr(self, "_grind_inadequate_set", set()))  # + inadequate (empty if flag off)
        non_gated = [(d, m) for d, m in routes if not self.strat.is_gated(m, pcount, plevel)
                     and (cur, tuple(m)) not in unreach and tuple(m) not in _dead]
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
        # (koga_run5: the last resort must ALSO honor the fail/dead memories — returning a proven
        # no-route hop here re-created the exact spin the memories exist to kill.)
        _last = [(d, m) for d, m in routes
                 if (cur, tuple(m)) not in unreach and tuple(m) not in _dead]
        if _last:
            d, m = _last[0]
            return ("route", m, self._EDGE[d])
        return None

    def _travel_to_known(self, pick, state, hunt_on_arrival=True):
        """Batch-WORLD (Phase 4b/c) — actuate 'travel to a place you've been', BACKWARD or forward, via
        the learned warp graph. One adjacent hop per tick (re-evaluated each tick like head_to_gym), and
        NEVER across an active gated wall map (Phase 4c — _wall_avoid). Fly is used when she owns it and
        the destination is a visited town (Jonny wants to SEE her fly, not walk the whole map); it
        degrades to walking if the fly actuation isn't live yet. On arrival at grass she starts hunting
        so 'go to Route 4' visibly becomes catching. Returns a status string the loop logs + feeds soul.

        hunt_on_arrival (2026-07-11, keeper router): the keeper-fetch errand reuses this pure ROUTING but
        wants its OWN targeted catch on arrival (the specific plan keeper, not a judged forward-catch), so
        it passes False to suppress the non-targeted auto-hunt and returns 'arrived' instead."""
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
        # NS#42: story-gate avoid here too — the general travel actuator (keeper errand + oracle travel:X
        # picks) must not route through Flute-gated Route 12/16 pre-Flute (else a travel: pick or the keeper
        # errand hops onto Route 12 and wedges on the Snorlax; _story_gate_avoid is empty once she has the Flute).
        avoid = self._wall_avoid(state) | self._story_gate_avoid(state)
        # WARP-AWARE (night shift 9, the banked_E4 Saffron seal): next_hop is EDGE-ONLY, but a
        # gate-locked city's every exit is a WARP (Saffron's four gatehouses) — the graph knew the
        # way and next_hop couldn't express hop #1, so every plain travel pick read no_route
        # forever. Use the same rideable next_step head_to_gym rides (feet-reachability law).
        step = self._next_step_rideable(cur, dst, avoid)
        if step is None:
            log(f"   [roam] travel to {self.world.name(dst)}: no wall-free route from {self.world.name(cur)}")
            return "no_route"
        nxt, kind, detail = step
        log(f"   [roam] TRAVEL: {self.world.name(cur)} -> {self.world.name(dst)} "
            f"(next hop {('warp ' + str(detail)) if kind == 'warp' else detail} -> {self.world.name(nxt)})")
        if kind == "warp":
            before = tuple(tv.map_id(self.b))
            self.trav.travel(target_map=None, arrive_coord=detail, max_steps=300)
            if tuple(tv.map_id(self.b)) == before:
                self.enter_warp(pick=detail)
            if tuple(tv.map_id(self.b)) == before:
                return "travel:warp_failed"
        else:
            r = self._edge_travel(nxt, detail)
            if r != "arrived":
                return f"travel:{r}"
        if tuple(tv.map_id(self.b)) == dst:
            node = self.world.node(dst)
            if node and node["traits"].get("has_grass") and hunt_on_arrival:
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
            # CRITICAL-HEAL FREEZE BREAKER (2026-07-09 night-train shift 8, the koga heal-return
            # stall): normally near-death PRUNES everything but heal. But if the heal router has
            # PROVEN it can't reach any Center from this map (heal-dead — e.g. she's stranded on
            # Route 13 south of the Route-12 gatehouse, whose split-map north band the naive
            # graph-hop can't cross), collapsing to only-'heal' hard-FREEZES her: heal no-ops, RED
            # climbs, no adjacent Center to break to. A real player in that spot walks to the NEXT
            # town's Center — so fall through to offer the forward push (head_to_gym below) instead
            # of freezing; a blackout en route just respawns her healed. The strand guard further
            # down keeps 'heal' itself suppressed while heal-dead, so she pushes forward, not circles.
            if tuple(state.get("map") or ()) not in getattr(self, "_heal_dead_maps", set()):
                a["heal"] = ("you're about to faint — get to a Pokémon Center and heal NOW; this comes "
                             "FIRST, before anything else")
                return a
            log("   [roam] survival-critical BUT heal proven-dead on this map — NOT freezing on "
                "only-'heal'; offering the forward push to the next town's Center instead")
        ng = state.get("next_gym")
        if ng:
            a["head_to_gym"] = f"head toward the next gym - {ng['leader']} of {ng['city']}"
        # POST-GAME (the summit-watch strand fix, scoped): a Champion parked inside the league (or any
        # interior) has no gym objective and no overworld route — offer the walk OUT so the victory lap
        # can actually start. Post-game-gated: pre-credits behavior untouched.
        elif state.get("post_game") and tuple(state.get("map") or (0,))[0:1] != (3,):
            a["leave_building"] = ("champion's walk — head out of this building and back into the "
                                   "world; the victory lap starts outside")
        # F-11 INTERIOR-WEDGE RECOVERY (descent pre-grade, the SCOPE-arc FAIL): pre-credits,
        # parked INSIDE an interior with her move options PROVEN dead (consecutive silent
        # no-moves — warp_failed/no_route), the honest escape is the walk out. Without this the
        # option set collapses to dead routes only ("NOT pruning — won't dead-end her") and she
        # paces a basement forever (Rocket Hideout B4F, watched it). Streak-gated so a normal
        # interior visit (a Mart, a questline building) never sees it; once offered, the dead-
        # route prune makes it dominant, and repeated hops ratchet her out floor by floor.
        if (tuple(state.get("map") or (0,))[0:1] != (3,) and "leave_building" not in a
                and getattr(self, "_nomove_streak", 0) >= 2):
            a["leave_building"] = ("nothing routes from in here — walk back out into the open "
                                   "and re-orient")
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
            if getattr(self, "_shop_fail_fp", None) == (tuple(state["map"]), self.money()):
                pass    # this exact shop already failed with this wallet — don't re-offer a spin
            elif self._shopping_list(foresight=fs, state=state):
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
        # F-6 SOCIAL FABRIC: un-greeted KEY figures here are first-class, warmly-framed options
        # (Mom is not "a local"). She still chooses; leaving anyway gets voiced by _social_tick.
        try:
            for f in self._salient_unmet(state):
                a[f"greet:{f['id']}"] = f["offer"]
        except Exception as _sf:
            log(f"   [social] salience offer skipped: {_sf}")
        # ── EXPLORING IS A MOVE (Batch-WORLD Phase 4a) — going to a PLACE she knows is a first-class
        # option, not just "advance toward the gym". When she's blocked or needs something (a team,
        # levels, balls) the obvious move is to travel to a place that HAS it. These are REACHABLE
        # visited places only (routed AROUND walls — never across the gated bridge); she still chooses.
        # WALK today; once she earns Fly, towns become a faster fly-to (capability-gated, Phase 3). ──
        try:
            cur = tuple(state["map"])
            tavoid = self._wall_avoid(state)
            can_fly = self.world.has_cap("fly")
            # 2026-07-07 BALL-LESS TEETH: with ZERO balls the nearest Mart is the destination that
            # matters — grass targets otherwise fill all 4 slots and the Mart never surfaces (how the
            # voltorb hunt would have wandered ball-less forever). Mart-first when the pocket is empty.
            _traits = (("has_mart", "has_grass") if self._ball_count() == 0
                       else ("has_grass", "has_mart"))
            for mid, nm, why in self.world.travel_targets(cur, avoid=tavoid, want_traits=_traits)[:4]:
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
        # ── CROSS-MAP KEEPER ROUTER (PASS 3 NEW#2) ───────────────────────────────────────────────────
        # Offer a BOUNDED detour to fetch a plan keeper that lives on a nearby reachable map (party has
        # room, keeper NOT on this map, within KEEPER_ROUTER_MAX_HOPS). Offered ALONGSIDE head_to_gym (the
        # oracle chooses) and placed BEFORE the forward-drive prune, which pops battle/wander_catch/travel:*
        # but not fetch_keeper — so a keeper detour survives the forward pull. Flag-gated (default OFF);
        # never offered when hurt (heal first) so a weak detour never starts on a dinged team. Self-clears
        # once caught / party fills (assess stops returning the keeper → _keeper_route_target None).
        # BALL GATE (NS#41): never offer a keeper detour with an empty bag — she'd route+enter the host
        # (esp. a cave, via the static connection) and spin re-entering it catching nothing (the 3-ball
        # cave3 soft-livelock). Zero balls -> the offer drops, _ball_note tells the oracle a Mart run comes
        # first, and she falls to head_to_gym/stock_up instead. Re-offered once she has balls again.
        if KEEPER_ROUTER_ENABLED and self._ball_count() > 0:
            _kr = self._keeper_route_target(state)
            if _kr:
                # HEAL-GATE (NS#39: don't start a detour on a dinged team) with an NS#41 SAFE-HOP relaxation:
                # the ns41_real look-ahead showed a mildly-dinged team (57% lead) marching PAST a keeper the
                # verified cave-catch could grab, because the strict `not needs_heal()` gate suppressed the
                # offer and the oracle never healed → keeper acquisition NEVER activated on a realistic run.
                # The gauntlet risk that motivates the gate (abra via Nugget Bridge) is a LEARNED-route host;
                # a STATIC-GATEWAY host is a door-cave adjacent to the route (Diglett's Cave off Route 11) with
                # NO gauntlet between her and the door — so offer it even when mildly hurt, JUST not when
                # CRITICALLY hurt (blackout risk still heals first). Learned-route hosts keep the strict gate.
                _safe_hop = (bool(self._host_gateways().get(self._place_name(_kr[1])))
                             and self._hurt_severity()[0] != "critical")
                if (not self.needs_heal()) or _safe_hop:
                    a["fetch_keeper"] = (
                        f"GO GET {_kr[0].upper()} — a teammate your plan wants for type coverage lives close "
                        f"by at {self._place_name(_kr[1])}, and you've got room on the team. Grab it while it's "
                        f"a short hop away, then straight back on the road — a real squad beats a lone carry.")
        # ── PC/BOX chaff-swap (Tier-1 #15) — the FULL-party sibling of the keeper router ──────────────
        # When the party is FULL of off-plan chaff and the plan wants a coverage keeper, box the weakest
        # chaff at the current city's Center so the router's room-gate opens next tick. Offered alongside
        # head_to_gym; survives the forward-drive prune (like fetch_keeper). Flag-gated (default OFF —
        # wedge-prone PC menu); never when hurt. Only fires in a mapped-Center city, so it can't strand.
        if PCBOX_ENABLED and not self.needs_heal():
            _cs = self._chaff_swap_target(state)
            if _cs:
                a["box_chaff"] = (
                    "MAKE ROOM — your team is full but it's carrying dead weight, and your plan wants a real "
                    "coverage teammate. Bench the weakest one at the PC so you can add the mon you actually "
                    "need. A prepared squad isn't six random catches — it's six that cover each other.")
        # ── PC/BOX keeper swap-IN (Tier-1 #15, NS#39) — the reverse of box_chaff, closing the loop ─────
        # A coverage keeper caught while the party was full gets AUTO-BOXED by FRLG — so it's on the plan
        # but sitting in storage. When she's at a Center, FIELD it: deposit the weakest chaff (if full) and
        # withdraw the keeper. Same gating shape as box_chaff (flag/planner/Center-city/never-when-hurt).
        if PCBOX_ENABLED and not self.needs_heal():
            _sw = self._box_keeper_swap_target(state)
            if _sw:
                a["swap_keeper"] = (
                    "BRING THEM UP — a teammate your plan actually wants is sitting in the PC box. Swap out "
                    "the dead weight and put the real coverage mon on the active team. Six that cover each "
                    "other beats a lone carry and five passengers.")
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
            _plan_build = self._plan_wants_prebuild(state)
            if _plan_build:
                log("   [roam] TEAM-BRAIN PRE-BUILD: team dangerously thin + a keeper/bench-build is DUE — "
                    "NOT pruning catch/grind to forward-drive (assemble the squad before the gauntlet)")
            if not stuck and prep_t is None and not _plan_build:
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
                    # ROAD-AWARE (2026-07-07): a billed road supersedes the base-camp pull — standing ON
                    # the road means she's NOT drifting (anchor=cur -> no reframe, options stay rich);
                    # off it, pull toward the deepest reachable road anchor, not the spine predecessor.
                    anchor = None
                    if not route_exists:
                        anchor = self._road_pull_anchor(state) or self._base_camp(state)
                    off_branch = anchor is not None and anchor != cur
                    # STRUCTURAL-DEAD GUARD (descent re-grade 2026-07-08): head_to_gym proven dead
                    # on THIS map (no_gym_route/questline_no_route) → do NOT reframe it as the
                    # dominant pull and prune everything else — that left ['talk_npc'] as the whole
                    # action set on Route 10 south while the questline couldn't route (the
                    # banked_ROCKTUNNEL 7-tick no_npc loop). The rich option set stays until she
                    # leaves the map (structural memory retries the route once on re-entry).
                    if "head_to_gym" in self._dead_moves_structural.get(cur, set()):
                        ql_open = off_branch = False
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
            # TEAM-BRAIN PRE-BUILD DOMINANCE (rule 2 — the load-bearing half): keeping catch/grind on the
            # menu wasn't enough — the oracle still picked head_to_gym and solo-charged the Nugget-Bridge
            # gauntlet (misty_done look-ahead: solo Ivysaur -> blackout, twice). So when the brain says
            # BUILD FIRST (thin team + a DUE keeper/bench-build) AND a real build option exists here (grass
            # reachable), SUPPRESS the forward push — symmetric to the critical-heal dominance. She
            # assembles a squad on the reachable grass, THEN the road returns. Bounded (<=2 mons -> clears
            # at party 3); never freezes (only when battle/wander_catch is actually available).
            if _plan_build and "head_to_gym" in a and ("wander_catch" in a or "battle" in a):
                a.pop("head_to_gym", None)
                log("   [roam] TEAM-BRAIN PRE-BUILD dominance: suppressing head_to_gym (would solo-charge "
                    "the trainer gauntlet the brain flagged) — build the squad on the reachable grass first")
        # ── SEVERELY-LOPSIDED BENCH dominance (PASS 3 NS#6 — team-depth lever a, the binding wall) ─────
        # When the bench is drastically under the gym milestone AND the ace towers over it, the ace can
        # carry the road forever so the dedicated bench grind ('battle') never gets chosen — the bench
        # stays frozen the whole climb (NS#5: grind fired 0x, abra L13/meowth L10 behind an L48 ace, 4
        # badges in; abra never reaches L16 to evolve into the Koga/Sabrina Psychic answer). Force ONE
        # bounded dedicated grind stint: suppress the march options (head_to_gym, wander_catch, travel:*)
        # so 'battle' -> grind_weak_members runs a +6 participation-XP bite that evens the bench up (and
        # crosses Abra's L16 evolution en route). Park-proof: _bench_severely_lopsided only fires with the
        # +6 pin armed (retires after one bite, re-arms only next badge) AND the milestone unmarked
        # (_lopsided_grind_done, set after the stint) — <= one stint per badge, never a road-parking
        # marathon. Only when 'battle' is on the menu (grass reachable) so it can't dead-end her.
        if "battle" in a and "head_to_gym" in a:
            _lop_ms = self._bench_severely_lopsided(state, prep_t)
            # NS#10 BENCH-TO-MILESTONE: if THIS map's grass already proved a poor grind spot for this
            # milestone (an unproductive bite), DON'T force a stop here — leave head_to_gym on the menu so
            # she MARCHES to better (near-gym, level-appropriate) grass. The pin also retires on a poor map
            # (_prep_team_target), so 'battle' stops out-competing the march. This is the KB-free
            # grind-spot-adequacy gate that keeps the to-milestone climb park-proof.
            if (_lop_ms is not None and BENCH_TO_MILESTONE
                    and (tuple(tv.map_id(self.b)), _lop_ms) in getattr(self, "_bench_poor_maps", ())):
                _lop_ms = None
            if _lop_ms is not None:
                self._lopsided_pending_ms = _lop_ms   # threaded to the executor to mark done after the stint
                pruned_lop = [k for k in list(a)
                              if k == "head_to_gym" or k == "wander_catch" or k.startswith("travel:")]
                for k in pruned_lop:
                    a.pop(k, None)
                log(f"   [roam] !! LOPSIDED-BENCH: bench severely behind milestone L{_lop_ms} + the ace "
                    f"(solo-carry shape) — forcing ONE dedicated grind stint; pruned march {sorted(pruned_lop)}")
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
            elif removable:
                # ALL options are proven-dead routes (night shift 9, the banked_E4 Saffron dead-air:
                # 4 travel picks all no_route, re-offered for ~25 straight decisions because "won't
                # dead-end her" predates the EMPTY-OPTIONS FLOOR below). Prune them ANYWAY — the
                # floor refills with the regroup-at-Center anchor, which is the one honest move a
                # real player has from a pocket where nothing routes.
                for k in removable:
                    a.pop(k, None)
                log(f"   [roam] !! NO-MOVE PRUNE (all-dead): every offered option "
                    f"{sorted(removable)} is a proven dead route (streak {self._nomove_streak}) — "
                    f"pruned to the regroup floor instead of thrashing them")
        # F-11 STRUCTURAL DEAD-ROUTE MEMORY (the banked_CINNABAR/BLAINE WARN class): a pick proven
        # structurally dead ON THIS MAP stays pruned regardless of movement — the movement-clears-
        # dead rule above resurrected head_to_gym every time an unrelated action moved her a few
        # tiles (announce-the-gym → stand still ×25, dead air a viewer feels). One retry on map
        # re-entry (leaving clears the entry); an ACTIVE questline re-enables head_to_gym since it
        # then drives the errand, not the dead route. Same never-empty guard as the prune above.
        try:
            sdead = {k for k in self._dead_moves_structural.get(tuple(state.get("map") or ()), set())
                     if k in a and not (k == "head_to_gym" and self._active_questline is not None)}
            if sdead and (len(a) - len(sdead)) >= 1:
                for k in sdead:
                    a.pop(k, None)
                log(f"   [roam] !! STRUCTURAL DEAD-ROUTE: {sorted(sdead)} proven dead on this map — "
                    f"pruned until she leaves it (or a questline arms); remaining: {sorted(a)}")
        except Exception as _sde:
            log(f"   [roam] structural dead-route prune skipped: {_sde}")
        # INTERIOR-WEDGE ESCAPE DOMINANCE (2026-07-09, Gate-C finish): the F-11 recovery OFFERS
        # leave_building when she's wedged in a dead-end interior (nomove_streak>=2, forward route
        # proven no-route), but the oracle kept picking talk_npc instead — and talk_npc MOVES her
        # (walks over to the NPC), so it never prunes as a dead route and re-arms head_to_gym every
        # tick = an endless talk<->no-route spin (misty_done Gate-C: a critical-heal mis-entered a
        # Cerulean house -> 14-decision STALL in house (7,3), leave_building offered but never picked).
        # When the walk-out is armed in a dead-end interior, SUPPRESS talk_npc so leave_building
        # dominates — she's in the WRONG building she can't route forward from; getting OUT is the
        # honest move (a legit questline NPC arrives via destination-interaction, not this wedge path).
        # Symmetric to the PRE-BUILD / critical-heal dominance. Never empties (leave_building remains).
        if ("leave_building" in a and "talk_npc" in a
                and tuple(state.get("map") or (0,))[0:1] != (3,)
                and getattr(self, "_nomove_streak", 0) >= 2):
            a.pop("talk_npc", None)
            log("   [roam] !! INTERIOR-WEDGE ESCAPE: wedged in a dead-end interior with the walk-out "
                "armed — suppressing talk_npc so leave_building dominates (get out of the wrong building)")
        # NEVER-EMPTY FLOOR (2026-07-08 soak finding): an EMPTY set ends free roam — on a live watch
        # that stops her cold mid-show (hit on the post-game victory lap at an unfamiliar strip: no
        # grass target, no known travel routes, no gym objective). There is always ONE honest move:
        # regroup at a Center (the heal excursion routes cross-region from anywhere, and from that
        # known anchor real options re-emerge next tick). Pre-credits this floor is naturally dormant
        # (head_to_gym is always offered while a next gym exists).
        if not a:
            # night shift 9: this was a["heal"], but heal is a NO-OP for a healthy party
            # (heal_at_center's fully-healed skip) and TRANSPARENT otherwise (returns her to the
            # same tile) — the banked_E4 window stood at Saffron (39,24) picking it for 25 straight
            # decisions. "regroup" actually RELOCATES: walk to the Center anchor, or ride a
            # connector building out of a sealed pocket.
            a["regroup"] = ("nothing obvious to do from this spot — regroup: head over to the "
                            "Pokémon Center and take stock from there")
            log("   [roam] !! EMPTY-OPTIONS FLOOR: no honest action was available here — offering "
                "the regroup-at-Center anchor so the roam never dead-ends")
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

    # ── BILLED ROADS (2026-07-07, the east_run2 fix) ──────────────────────────────────────────────
    # The old forward model when the graph can't route to the gym city was "go to the GYM_SPINE
    # predecessor, then march the SOUTH edge" — badge-1..3 geometry baked into a general rule. The
    # badge-4 road bends EAST from CERULEAN (two cities behind the spine predecessor), so at Cerulean
    # she was told "you've drifted — go back to Vermilion", and at Vermilion there was no south edge:
    # no_gym_route, loop, stall (east_run2). The fix is the KB's promise kept: curated DIRECTION
    # billing per gym road (gamedata/frlg_gates.json "roads"), executed leg-by-leg through the same
    # primitives (edge travel + door-passthrough), map ids bound live from the learned graph.
    def _gym_road(self, ng):
        """The billed road to the next gym city as parsed, LIVE-BOUND legs:
        [{'map': (g,n), 'name', 'go', 'via', 'note'}, ...]. None when no road is billed (the
        base-camp heuristic remains the fallback). Binding: a leg whose 'go'-edge neighbor is
        already in the learned graph overrides the next leg's expected id (connections are learned
        from the map header on visit, so binding lands one map AHEAD of her feet — a wrong expected
        id self-corrects before she ever stands on it). 'pass' legs cross via warps, not edges, so
        they never bind (Route 6's north edge is the Saffron gatehouse, not Route 5)."""
        if not ng:
            return None
        try:
            road = (self._questline_kb or {}).get("roads", {}).get(ng.get("city"))
        except Exception:
            road = None
        if not isinstance(road, list):
            return None
        legs = []
        for leg in road:
            try:
                g, n = str(leg["map"]).split(",")
                legs.append({**leg, "map": (int(g), int(n))})
            except Exception:
                continue
        if not legs:
            return None
        for i in range(len(legs) - 1):
            go = legs[i].get("go")
            if not go or legs[i].get("via") == "pass":
                continue
            nbr = self.world.edge_neighbor(legs[i]["map"], go)
            if nbr:
                legs[i + 1]["map"] = nbr
        return legs

    # ── ROAD-ANCHOR PARKING (2026-07-08 night shift 6, the banked_SCOPE twedge=271 class) ────────
    # From a walk-sealed pocket (hideout B4F's boss corridor) the world GRAPH still routes forward
    # through this map's warps, but her FEET can't reach any of them — travel read no_route and
    # enter_warp fanned five more doomed approaches, six travel wedges per tick, every tick,
    # forever ('warp_failed' is not a structural outcome, so forward-drive kept re-framing
    # head_to_gym as the dominant pull). The fix is a reachability PRE-CHECK with travel's own
    # pathfinder before any warp ride: an unreachable hop is parked for THIS query and the route
    # re-asked; when every forward hop is parked the caller falls through honestly to
    # no_gym_route — the structural outcome that stops the re-framing and leaves the oracle the
    # exit machinery that works (leave_building -> the elevator). Computed fresh from her feet
    # each tick, so it self-heals the moment she stands somewhere the warp IS reachable.
    def _warp_hop_reachable(self, tile):
        """True if her feet can walk to `tile` or a cardinal neighbor of it (door mats are entered
        from an adjacent stand-tile — the same fan enter_warp tries). travel-grade BFS, NPC-blind
        (bodies move; a genuine wall doesn't). Fail-OPEN on any read flake — a bad read must
        never park a real road."""
        try:
            grid = tv.Grid(self.b)
            cur = tuple(tv.coords(self.b))
            tx, ty = tile
            goal = {(tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)}
            if cur in goal:
                return True
            return bool(tv.bfs(grid, cur, lambda t: t in goal, walkable=grid.walkable))
        except Exception:
            return True

    def _next_step_rideable(self, cur, dst, avoid):
        """world.next_step, skipping warp hops whose every door tile is walk-unreachable from her
        feet. Returns (nxt_map, kind, detail) with `detail` swapped to a REACHABLE door when the
        first-listed one is sealed, or None when no rideable route remains."""
        dead = set(tuple(m) for m in (avoid or []))
        for _ in range(6):                     # bounded requery — never an inner spin
            try:
                step = self.world.next_step(cur, dst, avoid=list(dead))
            except Exception:
                return None
            if not step or step[1] != "warp":
                return step
            nxt_map, kind, detail = step
            doors = self.world.warp_tiles(cur, nxt_map) or [detail]
            ok = next((t for t in doors if self._warp_hop_reachable(t)), None)
            if ok is not None:
                return (nxt_map, kind, ok)
            dead.add(tuple(nxt_map))
            log(f"   [roam] !! ROAD-ANCHOR PARKED: every door {doors} from {cur} toward {nxt_map} "
                f"is walk-unreachable from her feet (sealed pocket) — re-asking the route without "
                f"that hop")
        return None

    def _road_step(self, state, road):
        """One forward move along the billed road. Standing ON a leg -> execute its billed crossing
        ('via':'pass' tries the proven door-passthrough first — the Underground Path class — then
        the plain edge). Off-road -> one graph hop toward the DEEPEST road anchor the learned world
        reaches. None = the road can't help from here (caller falls through to base-camp logic)."""
        cur = tuple(state["map"])
        # OFF-ROAD ANCHOR-STEER avoid (NS#42 Celadon-wedge fix): mirror the head_to_gym warp-route's
        # avoid set (10343) so the billed-road off-road steer (second loop below) is ALSO capability-
        # blind-proof. The warp-route already skips story-gated maps (Flute-gated Route 12/16) + bypasses
        # Saffron's guard-blocked gatehouses; without the SAME avoid here, the off-road steer routed her
        # EAST off Route 11 straight onto Snorlax-gated Route 12 (ns41_finalproof: 1,391× travel wedge at
        # (13,70) — the graph learned Route 11<->Route 12 but Route 12 is Snorlax-blocked pre-Flute). This
        # is a pure WIRING gap (the guard existed; this steer bypassed it). world.route always allows the
        # SRC map, so keeping cur in avoid still lets her escape a gated map she resumed ON.
        avoid = self._wall_avoid(state) | self._story_gate_avoid(state)
        _target_city = road[-1]["map"] if road else None
        if _target_city != SAFFRON:
            avoid = avoid | {SAFFRON}
        city = (state.get("next_gym") or {}).get("city", "the next gym")
        for i in range(len(road) - 1, -1, -1):
            leg = road[i]
            if leg["map"] != cur or not leg.get("go"):
                continue
            nxt = road[i + 1]["map"] if i + 1 < len(road) else None
            nm = road[i + 1]["name"] if i + 1 < len(road) else "the road ahead"
            # GATE CHECK before crossing (the Rock-Tunnel dark class): a billed leg can be story/
            # capability-gated (exit_gates) — recognize it and pursue the unlock questline instead
            # of walking into it (the pass-through would otherwise carry her into the pitch-dark
            # tunnel pre-Flash). Same seam as head_to_gym's south-gate block.
            try:
                gate = self._gate_recognizer.recognize(cur, blocked_dir=leg["go"])
            except Exception:
                gate = None
            if gate:
                if QUESTLINE_ENABLED and self._open_questline(gate, state):
                    return self._run_questline_step(state)
                # recognized gate + no pursuable errand -> NEVER cross it (the pass-through would
                # carry her into the pitch-dark tunnel); surface the why and let roam pick elsewhere
                self.on_event(gate.human, kind="route", tier=2)
                log(f"   [roam] ROAD to {city}: leg {leg['go']} off {leg['name']} is GATED "
                    f"({gate.missing}) — not crossing; surfacing")
                return "road_gated"
            log(f"   [roam] ROAD to {city}: on {leg['name']} — billed leg {leg['go']} toward {nm}"
                + (" (pass-through country)" if leg.get("via") == "pass" else ""))
            self.on_event(f"the road to {city} runs {leg['go']} from here — onward.",
                          kind="route", tier=1)
            if leg.get("via") == "pass":
                # CAVE-PASS (Rock Tunnel class): a 'pass' leg that runs THROUGH a dark warp-maze cave,
                # NOT a hut door-passthrough. The KB tags it with a 'cave' map id; hand it to the
                # warp-maze crosser (grid-BFS + section backtracking) instead of _door_passthrough
                # (which never crosses a cave -> she edge-oscillated Route 9<->Route 10, the shift-4
                # frontier). Gate check above already ensured Flash is taught before we get here.
                if leg.get("cave"):
                    try:
                        cg, cn = str(leg["cave"]).split(",")
                        r = self._cross_tunnel_leg((int(cg), int(cn)), nxt, leg["go"])
                    except Exception as e:
                        log(f"   [roam] tunnel-leg crossing errored ({e}) — falling through")
                        r = None
                    if r == "arrived":
                        return "arrived"
                    if r == "road_passthrough":
                        return "road_passthrough"
                    # need_flash / None -> fall through (door_passthrough/edge; defensive only)
                pt = self._door_passthrough(want_map=tuple(nxt) if nxt else None)
                if pt == "need_heal":
                    return "need_heal"
                if pt == "crossed":
                    if nxt and tuple(tv.map_id(self.b)) == tuple(nxt):
                        return "arrived"
                    return "road_passthrough"      # progressed; next tick continues from the new map
                # no connector fired — fall through to the plain edge (it may simply be open)
            return self._edge_travel(nxt, leg["go"]) if nxt else "no_gym_route"
        for leg in reversed(road):
            if leg["map"] == cur:
                return None                        # standing on the final leg (the city) — not ours
            try:
                step = self._next_step_rideable(cur, leg["map"], avoid)
            except Exception:
                step = None
            if step:
                nxt_map, kind, detail = step
                log(f"   [roam] ROAD to {city}: off-road at {cur} — steering toward road anchor "
                    f"{leg['name']} ({'warp ' + str(detail) if kind == 'warp' else detail})")
                if kind == "warp":
                    before = tv.map_id(self.b)
                    self.trav.travel(target_map=None, arrive_coord=detail, max_steps=300)
                    if tv.map_id(self.b) != before:
                        return "warped"
                    self.enter_warp(pick=detail)
                    return "warped" if tv.map_id(self.b) != before else "warp_failed"
                return self._edge_travel(nxt_map, detail)
        return None

    def _road_pull_anchor(self, state):
        """The forward-drive pull target when a road is billed: her own map when she's ON the road
        (no 'you've drifted' reframe — head_to_gym advances the leg under her feet), else the
        deepest road anchor the graph can route to. None -> caller uses the base-camp heuristic."""
        road = self._gym_road(state.get("next_gym"))
        if not road:
            return None
        cur = tuple(state["map"])
        if any(l["map"] == cur for l in road):
            return cur
        avoid = self._wall_avoid(state)
        for leg in reversed(road):
            try:
                if self.world.route(cur, leg["map"], avoid):
                    return leg["map"]
            except Exception:
                continue
        return None

    def _route_action(self, pick, state):
        """Route an oracle ACTION pick to its wired handler; return the handler's outcome string."""
        # HUD now-state label (BATTLE is detected live via in_battle() at publish time).
        self._now_state = {"heal": "HEALING", "stock_up": "SHOPPING"}.get(pick, "EXPLORING")
        if pick == "heal":
            _hp0 = (tuple(tv.map_id(self.b)), tuple(tv.coords(self.b) or ()))
            _hr = self.heal_nearest()
            _hp1 = (tuple(tv.map_id(self.b)), tuple(tv.coords(self.b) or ()))
            # HEAL PROVEN DEAD HERE (2026-07-09 shift 8, HARDENED shift 9): the full heal router
            # (own-map + adjacent-city + world-graph multi-hop + Viridian fallback + escape-hatch
            # reload) came back 'stuck' — it exhausted EVERY route and still couldn't heal, so no
            # Center is reachable from here. Remember it so _available_actions stops FREEZING on
            # only-'heal' at critical severity and lets the forward push to the next town's Center
            # carry her. Cleared on any successful heal (a later reachable Center discards its map).
            #   SHIFT-9 FIX (the Route-15 gate freeze, one crossing from Fuchsia): the old
            #   `_hp1 == _hp0` no-MOVE guard MISSED this whole class. From the Route-15 split, the
            #   Viridian fallback PING-PONGS her through the internal gate's two floors
            #   (24,0)<->(24,1) as it fails, so she DID move -> the mark never armed -> critical
            #   re-picked 'heal' forever, a hard freeze. A 'stuck' return is itself the honest
            #   "no Center reachable" proof (movement or not); mark on ANY 'stuck', and mark BOTH
            #   the map she started on AND the one the failing router left her on, so whichever
            #   she's standing on next tick trips the breaker into head_to_gym.
            if _hr == "stuck":
                if not hasattr(self, "_heal_dead_maps"):
                    self._heal_dead_maps = set()
                self._heal_dead_maps.add(_hp0[0])
                self._heal_dead_maps.add(_hp1[0])
                log(f"   [roam] !! HEAL stuck ({_hp0[0]}@{_hp0[1]} -> {_hp1[0]}@{_hp1[1]}) — marked "
                    f"heal-dead (critical-heal freeze breaker armed; she'll push to the next Center)")
            return _hr
        if pick == "regroup":
            return self._regroup_walk()
        # CROSS-MAP KEEPER ROUTER (PASS 3 NEW#2): fetch a plan keeper on a nearby reachable map, then the
        # on-map un-gate catches it. Re-entrant (one travel leg / catch per call). Never armed for
        # road-bench-xp (not a head_to_gym/travel: pick), so the ace — not a weak mon — leads the catch.
        if pick == "fetch_keeper":
            return self._fetch_keeper_errand(state)
        if pick == "box_chaff":
            return self._box_chaff_errand(state)
        if pick == "swap_keeper":
            return self._swap_keeper_errand(state)
        # POST-GAME champion's walk (the summit-watch strand fix): reuse the proven blackout
        # building-exit to step back onto the overworld, then normal actions take over next tick.
        if pick == "leave_building":
            self.on_event("okay — the Champion walks out the front door. let's go see my world."
                          if state.get("post_game") else
                          "this place is a maze and nothing's working — I'm heading back outside "
                          "to get my bearings.", kind="travel", tier=2)
            self._exit_to_overworld()
            # HONESTY FIX (void-core batch, repro'd 2026-07-08): the old unconditional
            # "left_building" LIED when the exit failed (recon_voidcore: she stayed wedged in the
            # Hall of Fame for 20+ ticks while every result said she'd left). Report what happened:
            # overworld = group 3; anything else means the exit didn't take -> 'stuck' surfaces to
            # the oracle/RED ladder instead of a green-looking no-op.
            return "left_building" if tv.map_id(self.b)[0] == 3 else "stuck"
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
                # GYM-GATE PROBE (2026-07-06): a 'stuck' entry can be an HM OBSTACLE on the door
                # approach (Vermilion's cut tree). Walk to the chokepoint, recognize at her feet,
                # and pursue the unlock errand instead of re-bonking the door forever.
                # CUT-IN-HAND FAST PATH (2026-07-10, night shift 11 — the Surge-entry wall past Gary):
                # the probe was gated behind `_active_questline is None`, but the 'HM Cut' errand does
                # NOT clear after HM01 is obtained from the captain (checkpoint ql stayed 'HM Cut...'),
                # so `_active_questline` stayed set, the probe was SKIPPED, and she could never cut the
                # Vermilion GYM-DOOR tree -> 'couldn't enter' -> a structural park + unproductive bench
                # grind. Run the probe on ANY stuck gym: it only clears a reachable obstacle she can
                # actually use (can_use('cut') inside), so cutting the tree is idempotent + safe even
                # mid-questline. Only OPEN a NEW gate errand when nothing's already in flight (preserve
                # the original guard against competing questlines).
                if out == "stuck" and QUESTLINE_ENABLED:
                    # STORY-PREREQ FIRST (2026-07-10 night shift 2, the Sabrina/Silph wall): a gym whose
                    # DOOR is Rocket/story-blocked has no tree for the HM probe to find — check the gym's
                    # story-prereq gate and arm the liberation errand (Silph Co.) BEFORE the HM probe
                    # walks her around hunting a non-existent obstacle. Only when nothing's in flight.
                    if self._active_questline is None:
                        pgate = self._gym_prereq_gate(gym)
                        if pgate and self._open_questline(pgate, state):
                            return self._run_questline_step(state)
                    gate = self._gym_gate_probe(gym)
                    if gate and self._active_questline is None and self._open_questline(gate, state):
                        return self._run_questline_step(state)
                # GYM-INTERIOR PING-PONG BREAKER (descent re-grade 2026-07-08, the banked_SILPH
                # Saffron loop): beat_gym 'stuck' can end with her INSIDE an interior she can't
                # cross (Sabrina's teleport pads — strike-solved, not general yet); the next tick's
                # warp-route walks her back OUT to the city, then re-enters, forever (each cycle
                # MOVES so the movement-cleared dead-route memory never bites). Two consecutive
                # unhandled stucks on the same gym → structurally dead on the CITY map (leaving the
                # map re-arms one retry) + one honest beat. A badge or questline clears the streak.
                if not hasattr(self, "_gym_stuck_streak"):
                    self._gym_stuck_streak = {}
                if out == "stuck":
                    n = self._gym_stuck_streak.get(ng["leader"], 0) + 1
                    self._gym_stuck_streak[ng["leader"]] = n
                    if n >= 2:
                        self._dead_moves_structural.setdefault(tuple(gym.city), set()).add("head_to_gym")
                        self.on_event(f"okay, {ng['leader']}'s gym has me beat for the moment — that layout "
                                      f"is a puzzle and I keep bouncing off it. I'll regroup and come back "
                                      f"at it fresh.", kind="gym", tier=2)
                        log(f"   [roam] !! GYM-INTERIOR WALL: beat_gym stuck x{n} on {ng['leader']} — "
                            f"head_to_gym structurally parked on {gym.city} until she leaves the map")
                else:
                    self._gym_stuck_streak.pop(ng["leader"], None)
                if out == "badge":
                    # PHASE 6 — CATHARSIS: reference the worry so the relief is EARNED, not "oh great,
                    # moving on". Tier-3 big beat → her core deep-reaction path RISES; the saga promotes it.
                    nb = state.get("badge_count", 0) + 1
                    self.on_event(f"YES — we DID it! {ng['leader']} is DOWN. I was genuinely nervous about that "
                                  f"one and we pulled it off — badge number {nb}. okay. okay! onward.",
                                  kind="gym", tier=3)
                elif out == "needs_heal":
                    # PP FAMINE / battle-unready mid-gym (erika_run2): heal NOW at the city's own
                    # Center, then the next tick re-picks head_to_gym (the RECALIBRATE nudge holds
                    # the thread) and re-takes the gym — beaten juniors stay beaten in-game.
                    self.on_event("we're running on empty — Center first, then we take this gym "
                                  "properly.", kind="gym", tier=2)
                    hr = self.heal_nearest()
                    log(f"   [roam] gym needs_heal -> heal_nearest {hr} (next tick re-takes the gym; "
                        f"juniors stay beaten)")
                    return "ok" if hr == "ok" else "stuck"
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
            # CAVE-PASS PRIORITY (2026-07-09 shift 6): when she's standing ON a billed 'pass' leg that
            # runs THROUGH a dark warp-maze cave (Rock Tunnel: map (3,28) -> Lavender), the billed road
            # MUST own the crossing (_cross_tunnel_leg), NOT the world-graph warp-route. Route 10's SOUTH
            # map-connection to Lavender (3,4) is seeded as a plain EDGE, but that band is cliff-sealed —
            # the only crossing is the tunnel — so _next_step_rideable returned an EDGE-ROUTE south that
            # _edge_travel dead-ended on 'no_path', and head_to_gym got structurally pruned on Route 10
            # (the shift-4 _cross_tunnel_leg fix never even ran). Same class as the shift-3 gated-shortcut
            # preemption (Snorlax/Saffron), but via a phantom walkable edge. Run the billed road first so
            # the tunnel crossing takes the leg; fall through to the warp-route only if it can't help.
            _cave_road = self._gym_road(ng)
            if _cave_road and any(leg["map"] == cur_map and leg.get("via") == "pass"
                                  and leg.get("cave") for leg in _cave_road):
                rr = self._road_step(state, _cave_road)
                if rr is not None:
                    return rr
            target_city = self._next_gym_city_map(ng)
            if target_city and target_city != cur_map:
                # SAFFRON BYPASS (2026-07-09 shift 3): Saffron's gatehouses are GUARD-BLOCKED (a static
                # obstacle refuses passage until a Celadon vending drink), and the world graph learned the
                # Route 6 <-> Saffron gatehouse warp — so the warp-route drove her into the passage (18,0)
                # and oscillated on the blocked (4,1) warp toward Saffron (3,10). Every billed gym road
                # BYPASSES Saffron (Underground Paths + Rock Tunnel), so avoid routing THROUGH Saffron
                # unless Saffron IS the destination (badge 6, when she has the drink) — this drops the
                # warp-route to the billed road's Underground-Path 'pass' leg.
                _avoid = self._wall_avoid(state) | self._story_gate_avoid(state)
                if target_city != SAFFRON:
                    _avoid = _avoid | {SAFFRON}
                try:
                    step = self._next_step_rideable(cur_map, target_city, avoid=_avoid)
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
            # BILLED-ROAD FALLBACK (2026-07-07): the learned graph can't route to the gym city yet —
            # follow the curated road from the KB (badge-4+ roads bend EAST/NORTH; the base-camp
            # 'march south' heuristic below is badge-1..3 geometry and looped Vermilion<->Cerulean
            # forever in east_run2). One leg per tick — free roam preserved.
            road = self._gym_road(ng)
            if road:
                rr = self._road_step(state, road)
                if rr is not None:
                    return rr
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
                    # GRASS-TARGET FAIL MEMORY (koga_run3, the Route-15 stall): from a one-way ledge
                    # pocket the SAME target fails with no_route forever — _grass_target re-proposed
                    # east->(3,32) fourteen straight ticks until the stall detector killed the run.
                    # Remember (from-map -> target) failures IN-RAM so the next tick falls through to
                    # the NEXT candidate (the graph-BFS then routes her out the other way, e.g. west
                    # through Fuchsia to Route 18's reachable grass). Session-scoped, never persisted.
                    if not hasattr(self, "_grass_unreach"):
                        self._grass_unreach = set()
                    self._grass_unreach.add((tuple(state["map"]), tuple(tgt)))
                    self._prep_dry = getattr(self, "_prep_dry", 0) + 1   # counts toward stand-down
                    log(f"   [roam] grass route {tgt} UNREACHABLE from {tuple(state['map'])} ({r}) — "
                        f"remembered; next tick tries a different grass")
                    return f"to_grass:{r}"
            if pick == "wander_catch":
                # PLAN-AWARE (Part-C executor): if the forward brain has a DUE keeper that lives on
                # THIS map (she walked onto its route), SEEK it specifically — flee the filler, catch
                # the keeper. Else the ordinary judged forward-catch. Voice already folded via plan_note.
                _kt = self._plan_keeper_target(state)
                if _kt:
                    log(f"   [roam] PLAN-CATCH: seeking planned keeper '{_kt}' on this route "
                        f"(fleeing non-targets)")
                return self.catch_one(target_species=_kt)
            # STRATEGIC UNDERLEVEL-GRIND (Task B): if the TEAM FLOOR is under the wall's level, field the
            # WEAK members (not the ace) and level THEM to readiness — the real fix for "the display said
            # 'train the team' but the action trained the ace". Else the ordinary lead bump.
            t = self._prep_team_target(state)
            if t is not None:
                # E4-PREP FLOOR (PASS 3): when the target came from prep_for_e4 (badge 8, whole-team floor
                # to ~L55), field only the LEVELABLE team (within E4_PREP_BAND) — never drag L8-14 box-fodder
                # into the grass. Recompute _prep_e4_target (cheap) to confirm the source; other grinds pass None.
                _e4t = self._prep_e4_target(state, state.get("party") or [])
                _ml = (t - E4_PREP_BAND) if (_e4t is not None and _e4t == t) else None
                _floor0 = min(self._party_levels()) if self._party_levels() else 0   # NS#10: pre-bite floor
                r = self.grind_weak_members(t, min_level=_ml)
                if r == "no_safe_grass":                 # a dry attempt counts toward stand-down
                    self._prep_dry = getattr(self, "_prep_dry", 0) + 1
                elif r in ("ready", "ok"):               # ACTUAL grinding happened (not mere arrival —
                    #                                      run6: the Fuchsia↔Route-15 shuttle 'arrived'
                    #                                      every other tick and reset the counter forever)
                    self._prep_dry, self._prep_dry_logged = 0, False
                _lop_ms = getattr(self, "_lopsided_pending_ms", None)
                if BENCH_TO_MILESTONE and _lop_ms is not None and r in ("ready", "ok"):
                    # NS#10 productivity gate (the grind-spot-adequacy signal, KB-free): measure the bite.
                    _floor1 = min(self._party_levels()) if self._party_levels() else 0
                    _cur_map = tuple(tv.map_id(self.b))
                    if _floor1 >= (_lop_ms - BENCH_MS_CLOSE):
                        # bench reached the gym level (within CLOSE) — milestone truly satisfied, stand down.
                        self._lopsided_grind_done = getattr(self, "_lopsided_grind_done", set())
                        self._lopsided_grind_done.add(_lop_ms)
                        log(f"   [roam] BENCH-TO-MS: floor L{_floor1} within {BENCH_MS_CLOSE} of milestone "
                            f"L{_lop_ms} — bench is gym-ready, standing down the forced climb")
                    elif (_floor1 - _floor0) < BENCH_BITE_MIN:
                        # this map's grass gave a POOR bite (< BITE_MIN levels) — a weak grind spot for this
                        # milestone. Mark it so the pin retires + the dominance releases head_to_gym here →
                        # she MARCHES toward better (near-gym, level-appropriate) grass instead of a
                        # stationary weak-grass marathon (the celadon_run1 park). Park-proof by construction.
                        self._bench_poor_maps = getattr(self, "_bench_poor_maps", set())
                        self._bench_poor_maps.add((_cur_map, _lop_ms))
                        log(f"   [roam] BENCH-TO-MS: bite on {_cur_map} gained {_floor1 - _floor0} level(s) "
                            f"(< {BENCH_BITE_MIN}) toward milestone L{_lop_ms} — poor spot, marching to better grass")
                    else:
                        log(f"   [roam] BENCH-TO-MS: productive bite {_floor0}->{_floor1} toward milestone "
                            f"L{_lop_ms} — climbing on (floor now L{_floor1})")
                    self._lopsided_pending_ms = None
                elif r == "ready" and _lop_ms is not None:
                    # NS#6 LOPSIDED-BENCH (flag off): a COMPLETED stint marks the forced milestone done, so
                    # the lopsided dominance fires at most once per badge even if the pin can't retire.
                    self._lopsided_grind_done = getattr(self, "_lopsided_grind_done", set())
                    self._lopsided_grind_done.add(_lop_ms)
                    self._lopsided_pending_ms = None
                return r
            # RIVAL-RETURN GUARD (NIGHT SHIFT 16): don't let a stray 'battle' pick trap-grind her into the
            # nearest grass when she OWES a return to a 4+-mon rival gauntlet she lost and is OFF the gym city.
            # The lead+2 fallback from a respawn town (Cerulean) dives down the one-way Route-4 ledge she can't
            # path back up (the s16 stranding). The prep belongs at the ANCHOR (Route 6, via the questline);
            # here just no-op so next tick's head_to_gym carries her back to the anchor to grind + board.
            _rw = self.strat.active_wall_rec()
            if _rw and _rw.get("is_trainer") and (_rw.get("size") or 0) >= 4:
                _ngc = (state.get("next_gym") or {}).get("city") or ""
                _place = state.get("place") or ""
                if not (_ngc and _ngc.lower() in _place.lower()):
                    log("   [roam] RIVAL-RETURN: owe a rematch to a 4+-mon rival and off the gym city — "
                        "skipping the local grind (would trap off-spine); routing back to the anchor instead")
                    return "rival_return_pending"
            lead = state["party"][0]["level"] if state["party"] else 5
            return self.grind(lead + 2)
        if pick == "stock_up":
            door = CITY_MART_DOORS.get(state["map"])
            sl = self._shopping_list(foresight=self._walled(state), state=state)
            if door is None or not sl:
                return "nothing_to_buy"
            bought = self.buy_at_mart(door, sl)
            if bought:
                self._shop_fail_fp = None
                return "stocked"
            # remember the failure fingerprint — same city + same wallet means retrying is a spin
            # (surge run 1: 'Potion not sold here' x16 stalled the roam at the Mart door)
            self._shop_fail_fp = (tuple(state["map"]), self.money())
            return "shop_failed"
        if pick == "talk_npc":
            return self.talk_npc()
        if pick.startswith("greet:"):
            return self._greet_figure(pick.split(":", 1)[1], state)
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
        # DENSE AUTO-CHECKPOINT — start the wall-clock cadence clock and drop an immediate labeled bundle,
        # so a resume point exists in the growing history from tick 0 (not only after the first gain/12min).
        self._last_ckpt_t = t0
        if AUTO_CKPT_ENABLED:
            self._auto_checkpoint("roam_start")
        void_recoveries = 0                        # VOID-CORE recovery budget this run (bounded)
        for tick in range(1, max_ticks + 1):
            if _t.time() - t0 > max_seconds:
                log(f"   [roam] time budget {max_seconds}s reached — ending"); break
            # ── VOID-CORE TRIPWIRE (2026-07-08, the QW-4 class — FIRST, before the anchor block,
            # which would otherwise read the zeroed badges as a "badge change" and BANK the title
            # screen). The game rebooted itself to title mid-run: do NOT play it, do NOT save it,
            # do NOT burn an oracle call on it — reload the newest real world and continue. Bounded:
            # repeated void within one run means the reboot re-triggers -> stop loud for a human.
            if self._world_lost():
                void_recoveries += 1
                log(f"   [roam] !!!! VOID-CORE TRIPWIRE #{void_recoveries}: the world reads DEAD "
                    f"(title-screen signature) — the game REBOOTED under us. Recovering, not playing it.")
                self.on_event("whoa — everything just went dark on me. hang on… rewinding to where "
                              "I last knew the world was real.", kind="recover", tier=2)
                if void_recoveries <= 3 and self._void_recover():
                    ledger.note_action("void_recover", "reloaded")
                    self._wait_overworld()
                    continue
                log("   [roam] !!!! VOID-CORE unrecoverable (no real state to reload, or the reboot "
                    "keeps re-triggering) — ABANDONING loud, this needs a human")
                self._roam_progress = "ABANDONED"
                self.on_event("the world's gone and I can't get it back on my own — I need a hand.",
                              kind="abandoned", tier=3)
                self._fire_deadman_alert({"place": "void (title screen)", "coords": None,
                                          "badge_count": 0})
                return "abandoned"
            # ── IMPOSSIBLE-STAND AT THE TICK TOP (night shift 10): the PARTIAL-void variant
            # burns a wedge STORM through wander/travel picks long before any floor collapses
            # to the regroup path where the original check lives (round 6: 20 wedges at
            # Pallet "(8,6)" and the floor never reached). Cheap gate: only probed on a tick
            # whose predecessor actually burned travel wedges; two consecutive impossible
            # reads -> the same recovery as void-core (shared bounded budget).
            _wt = getattr(self.trav, "wedge_total", 0)
            if _wt > getattr(self, "_last_wedge_total", 0) and self._impossible_stand():
                self._imp_stand_streak = getattr(self, "_imp_stand_streak", 0) + 1
            else:
                self._imp_stand_streak = 0
            self._last_wedge_total = _wt
            if self._imp_stand_streak >= 1:
                # streak 1 (round 7): the wedge-burn gate + the enclosed test already make a
                # false positive a coincidence of three rare reads; each extra tick of storm
                # is ~4 wedges and the desync never self-heals — fire on the first sighting.
                void_recoveries += 1
                self._imp_stand_streak = 0
                log(f"   [roam] !!!! IMPOSSIBLE-STAND TRIPWIRE (tick-top) #{void_recoveries}: "
                    f"enclosed non-walkable stand {tv.coords(self.b)} on {tv.map_id(self.b)} "
                    f"after a wedge-burning tick — partial-void world; recovering now "
                    f"instead of letting the storm run.")
                self.on_event("okay, something's properly wrong with where the world thinks I'm "
                              "standing — rewinding to solid ground.", kind="recover", tier=2)
                if void_recoveries <= 3 and self._void_recover():
                    ledger.note_action("void_recover", "impossible_stand")
                    self._wait_overworld()
                    continue
                log("   [roam] !!!! IMPOSSIBLE-STAND unrecoverable — ABANDONING loud")
                self._roam_progress = "ABANDONED"
                self._fire_deadman_alert({"place": "impossible stand (partial void)",
                                          "coords": tv.coords(self.b), "badge_count": 0})
                return "abandoned"
            # CAMPAIGN ANCHOR (Batch 5 P1) — checked at the TOP of every tick so it's NEVER skipped by a
            # mid-tick `continue` (idle / hard-recovery / blackout): re-anchor the instant the prior tick
            # made REAL progress (badge / new area / catch), plus a periodic heartbeat floor. Reads RAM
            # directly so it reflects whatever the last action accomplished.
            sig = _camp_sig()
            _saved_this_tick = False
            _gain_reason = None
            if sig != last_camp_sig:
                reason = ("badge" if sig[0] != last_camp_sig[0] else
                          "new_area" if sig[1] != last_camp_sig[1] else "catch")
                last_camp_sig = sig
                self._save_campaign(reason)
                self._continuity_save()            # ADDENDUM D: a real gain -> bank her saga too
                _saved_this_tick = True
                _gain_reason = reason
            elif tick > 1 and (tick - 1) % CAMPAIGN_SAVE_EVERY == 0:
                self._save_campaign(f"tick{tick - 1}")
                self._continuity_save()
                _saved_this_tick = True
            # DENSE AUTO-CHECKPOINT (2026-07-09) — a GROWING labeled history for the dev fix→reload→verify
            # loop (the rolling anchor above is "now" only). Bank a full sanctity bundle on every GAIN SEAM
            # (badge/town/teammate) AND on a wall-clock floor (~12 min). On the periodic path the canonical
            # bundle may not have been refreshed this tick, so ensure it's current before copying. Dev-line
            # only; cannot touch the sacred states/kira/ spine (see _auto_checkpoint firewall).
            if AUTO_CKPT_ENABLED:
                _now = _t.time()
                _due = (_now - self._last_ckpt_t) >= CKPT_EVERY_S
                if _gain_reason is not None or _due:
                    if not _saved_this_tick:
                        self._save_campaign("ckpt"); self._continuity_save()
                    self._auto_checkpoint(f"gain-{_gain_reason}" if _gain_reason else "periodic")
                    self._last_ckpt_t = _now
            self._wait_overworld()
            # WHITEOUT vs DELIBERATE INTERIORITY (night shift 14 — the S.S. Anne rival ping-pong): the
            # DIRECTED interior nav sets `_ql_inside_target` so the recovery below LEAVES her inside a
            # quest building (the ship). But a battle LOSS whites her out and warps her to a Pokémon Center
            # — she is no longer in that building. The flag stayed stale-True, SUPPRESSING the recovery
            # (both branches require `not _ql_inside_target`), so she sat inside Cerulean's Center after
            # losing to Gary — head_to_gym can't route out of a building -> an unwatchable ~11-tick no_route
            # ping-pong. FIX: track the interior map she's deliberately in (`_ql_inside_map`, refreshed each
            # tick she's inside with no battle); if a battle RAN this action AND she's now on a DIFFERENT
            # interior map, that's a whiteout relocation -> void the interiority so the recovery exits her.
            # A WON battle on the SAME building map keeps the flag (she stays aboard exploring). General:
            # any deep-building loss, not just Gary.
            _here_map = tuple(tv.map_id(self.b))
            if getattr(self, "_ql_inside_target", False):
                # HEAL-GATE (2026-07-09, the Bill's-cottage false-eject): a battle + a map change was
                # read as a whiteout relocation and cleared the interiority — but the DISTANT-DOOR
                # approach walks her ACROSS Route 25 through its trainer gauntlet and INTO Bill's cottage
                # in ONE action, so a real (won/fled) gauntlet fight + the deliberate door-warp tripped
                # this and EJECTED her from the cottage before she could talk Bill (misty_done Gate-C).
                # A real whiteout HEALS the party and dumps her in a Center; a deliberate entry leaves her
                # HURT in the target building. Only treat battle+relocation as a whiteout when she's
                # actually been healed — otherwise she walked in on purpose: keep + refresh interiority.
                if (getattr(self, "_battle_ran_this_action", False)
                        and _here_map != getattr(self, "_ql_inside_map", _here_map)
                        and self._party_fully_healed()):
                    self._ql_inside_target = False
                    log(f"   [roam] whiteout relocated her {getattr(self, '_ql_inside_map', None)} -> "
                        f"{_here_map} after a loss (party healed) -> clearing _ql_inside_target (let the "
                        f"stranded-in-building recovery exit her to the overworld)")
                else:
                    self._ql_inside_map = _here_map     # deliberate interior nav -> keep her location current
            # BLACKOUT / STRANDED-IN-BUILDING RECOVERY (increment 4 PART A): a wild loss whites her out
            # and warps her INSIDE a Pokémon Center (map group != 3 — a building interior), healed. Her
            # overworld actions (head_to_gym routes via map CONNECTIONS) can't navigate out of a building,
            # so she'd sit on no_gym_route forever (the live (7,4)@(5,4) dead-end). Detect the building
            # (group != 3) and EXIT to the overworld so a real objective can re-establish from the Center
            # (a known-good anchor) — never leave her parked where nothing can succeed. The faint itself
            # is felt via _soul_after_objective(battle_loss); this is the explicit "I came to" beat.
            if (tv.map_id(self.b)[0] != 3 and not getattr(self, "_ql_inside_target", False)
                    and not getattr(self, "_battle_ran_this_action", False)):
                # EVIDENCE GATE (run-9 ghost-vileplume class): being indoors at tick top is NOT proof of
                # a blackout — the questline's anchor-first warp step legitimately ENDS a tick inside a
                # transit hut. Without a battle having RUN since the last tick there is nothing to have
                # whited out FROM: the old unconditional branch here recorded a loss vs the STALE
                # last_foe snapshot (Erika's vileplume, still in the save's RAM), gated Route 8 on a
                # trainer that doesn't exist, narrated a false "I blacked out" memory, and its ejection
                # rebuilt the hut hop-loop. No battle -> normal transit -> continue from inside.
                log(f"   [roam] tick opened inside interior {tv.map_id(self.b)}@{tv.coords(self.b)} "
                    f"with no battle since last tick — NOT a blackout; continuing (mid-route interior)")
            elif tv.map_id(self.b)[0] != 3 and not getattr(self, "_ql_inside_target", False):
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
            self._battle_ran_this_action = False   # blackout-evidence is per-tick; consumed above
            state = self.read_live_state()
            self._learn_map(state)         # BATCH-6 P2: fold this map's connections+grass into the graph
            # PREP STAND-DOWN RE-ARM (NS#11 deadlock fix): _prep_dry latches at 2 from a grassless pocket and
            # never clears (its only reset is inside the grind path _prep_dry gates off), freezing the bench
            # for the rest of the run. Clear it the moment she stands on GENUINELY-USABLE new grass so the
            # +6 pin + lopsided dominance re-arm on the level-appropriate Route 13-15 grass she marches past.
            # Guarded so it can't reintroduce a spin/shuttle: reachable grass HERE (excludes towns + ledge-
            # gated grass) AND this map not marked grind-dead (excludes fragile one-way-strand / dead maps).
            if PREP_DRY_RESET and getattr(self, "_prep_dry", 0) >= 1:
                try:
                    _cur = tuple(tv.map_id(self.b))
                    if _cur not in getattr(self, "_grind_dead", set()) and self._reachable_grass() is not None:
                        self._prep_dry, self._prep_dry_logged = 0, False
                        log(f"   [roam] PREP RE-ARM: usable grass reachable on {_cur} — clearing the stale "
                            f"stand-down so the bench-leveling machinery re-arms here")
                except Exception as _pe:
                    log(f"   [roam] prep-dry re-arm skipped: {_pe}")
            self._refresh_world_caps()     # Batch-WORLD: keep Fly/Surf/etc. live (she uses them when earned)
            # PHASE C-2 — meet anyone who joined the party WITHOUT a witnessed catch (gift/PC/Jonny's
            # hands): the introduction arc (met/named/bonded), never a silently-deployed stranger.
            self._meet_new_teammates(state)
            # F-6 SOCIAL FABRIC — voice a deliberate walked-past-Mom skip (once per figure per run).
            try:
                self._social_tick(state)
            except Exception as _sx:
                log(f"   [social] tick skipped: {_sx}")
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
                self.soul.surface_want({"place": self._location_block(state), "map": state["map"],
                                        "badges": state["badges"], "progress": state["progress"],
                                        "party": self._party_brief(state),     # PHASE 1: team by NAME
                                        "goal": self._goal_layers(state)})      # PHASE 1: 3-tier goal
            # On YELLOW+, fold STUCK-AWARENESS into the oracle ctx via the existing `place` seam (the
            # only general field her oracle prompt renders — firewall: no core edit). She becomes AWARE
            # she's stuck; she still decides the next move HERSELF (capability-not-script).
            # F-8 (the descent): the tick ctx LEADS with the grounded location block, not the bare
            # place name — indoor/outdoor truth + only live-confirmed traits, so nothing downstream
            # (her voiced pick, the want, the asides) starts from an ungrounded sense of place.
            where = self._location_block(state)
            # F-6: un-greeted key figures here exert PULL on the decision (she still chooses).
            try:
                for _f in self._salient_unmet(state):
                    where = f"{where} {_f['pull']}"
            except Exception:
                pass
            # Batch-WORLD (Phase 2) — SENSE OF PLACE: lead the oracle ctx with a short spatial picture
            # from her visited-world memory (where she is, what's around, what's BLOCKED, what she can
            # walk back to, how she can travel). This is the MAP she never had — so when a path is
            # blocked she has somewhere in her head to go besides into the wall. Firewall: same `place`
            # seam; she still decides. Avoid = route-walls she can't beat; blocked_dirs = a wall on THIS map.
            try:
                avoid = self._wall_avoid(state)
                bdirs = self._wall_blocked_dirs(state)
                cur_map = tuple(state["map"])
                brief = self.world.spatial_brief(cur_map, avoid=avoid, blocked_dirs=bdirs,
                                                 named_already=True)   # F-8 block already names it
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
                snote = self._shop_note(foresight=self._walled(state), state=state)
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
            # trainer walls only — a wild loss never reads as "the route is GATED" (it isn't: the
            # routing side no longer gates wild walls, and telling her it does made her abandon the
            # forward road after one bench-faint to an oddish; loss_awareness still covers wild losses)
            wr = self.strat.active_wall_rec()
            if wr and wr.get("map_id") and wr.get("is_trainer") and not self.strat.stronger_since_wall(
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
            # STRATEGIC PLANNER (2026-07-08 mega-batch): fold the PROACTIVE forward-prep beat — what's
            # ahead, do I already have the answer, and the single smartest prep move (level my existing
            # counter / catch a keeper type / stop neglecting the bench). This is the antidote to the old
            # reactive-only pattern (she learned a wall only AFTER losing); now she anticipates it. Same
            # `place` seam, she still chooses. Skipped cleanly on post-game / no-threat / disabled.
            try:
                # THE FORWARD BRAIN takes precedence (mission-pivot): fold its proactive next-action
                # beat if it has one; only fall back to the reactive matchup-foresight note when the brain
                # is silent (on-track / post-game / disabled). Never double up two prep beats in one ctx.
                _plan = ""
                try:
                    _plan = self.team_planner.plan_note(state)
                except Exception as _tp:
                    log(f"   [roam] team-planner fold skipped: {_tp}")
                if not _plan:
                    _plan = self.planner.plan_note(state)
                if _plan:
                    where = f"{where}. {_plan}"
            except Exception as _pl:
                log(f"   [roam] planner fold skipped: {_pl}")
            # SOUL-DEBT #12 (info half) — fold her OVERHEARD INTEL (NPC/sign lines the extractor kept)
            # into the decision ctx, so "the captain is upstairs" can actually steer the pick. Her own
            # notes, not omniscience: only lines she really read this run.
            try:
                _hb = self.hints.ctx_brief() if self.hints is not None else ""
                if _hb:
                    where = f"{where}. {_hb}"
            except Exception as _hbe:
                log(f"   [roam] hints fold skipped: {_hbe}")
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
            # ── F-1 WANDER TRIPWIRE (THE DESCENT doctrine: the soul chooses DESTINATIONS; the
            # deterministic pathfinder executes ALL movement). Wandering/dithering reads GREEN to
            # the progress ledger (the fingerprint moves while she circles), so it needs its own
            # detector: a STANDING objective SHE chose whose learned-graph distance hasn't improved
            # in NAV_TRIPWIRE_S+ wall-clock. Trip -> the harness takes the wheel THIS tick: her own
            # objective's handler runs directly, NO oracle call (kills the wedged-token-burn too).
            # Distance unmeasurable -> disarmed; objective cleared -> watch cleared. Narrated once
            # per episode so a viewer sees decision, not glitch.
            _forced_pick = None
            try:
                _obj = self._active_objective
                if _obj and _obj.get("action") in avail:
                    _d = self._objective_distance(state)
                    _w = self._nav_watch
                    if _d is None:
                        self._nav_watch = None
                    elif not _w or _w.get("action") != _obj.get("action"):
                        # keyed on the ACTION STRING, not dict identity — re-committing to the same
                        # destination must NOT reset the clock (each pick builds a fresh obj dict)
                        self._nav_watch = {"action": _obj.get("action"), "best": _d,
                                           "ts": _t.time(), "fired": 0}
                    elif _d < _w["best"]:
                        _w["best"], _w["ts"] = _d, _t.time()
                    elif _t.time() - _w["ts"] > NAV_TRIPWIRE_S:
                        _w["ts"] = _t.time()
                        _w["fired"] += 1
                        self._nav_tripwire_total += 1
                        _forced_pick = _obj["action"]
                        log(f"   [roam] 🧭 F-1 WANDER TRIPWIRE #{_w['fired']}: objective "
                            f"\"{_obj.get('label')}\" has gone {NAV_TRIPWIRE_S:.0f}s+ without route "
                            f"progress (dist {_d} hops) — harness takes the wheel: executing her "
                            f"objective deterministically, no oracle burn this tick")
                        if _w["fired"] == 1:
                            self.on_event("okay, enough drifting — I said I was heading there, "
                                          "so let's actually go.", kind="travel", tier=1)
                else:
                    self._nav_watch = None
            except Exception as _nwx:
                log(f"   [roam] wander-tripwire skipped: {_nwx}")
            # ZERO-INPUT THINK GUARANTEE (void-core batch): the watch-rig pump runs frames while the
            # LLM decides — if any prior handler left a key held (an abort path that skipped its
            # release), she WALKS during the think (the intra-tick position-jump class the QW-4 log
            # showed: STATE IN on one map, the action reading another). Release NOW so pumped frames
            # are truly idle. Best-effort: owner-mismatch is dropped+logged by the bridge by design.
            try:
                self.b.release(owner="agent")
            except Exception:
                pass
            if _forced_pick is not None:
                pick = _forced_pick
            else:
                _t_choose = _t.time()
                pick = self._soul_choose("action", avail,
                                         {"place": where, "progress": state["progress"], "party": _brief,
                                          "goal": _goals})
                _choose_ms = (_t.time() - _t_choose) * 1000
                if _choose_ms > 500:
                    # F-7 pump-awareness: with play_live's frame_pump wired the world stays LIVE
                    # during the wait (she idles in place, music runs) — a long think is then a
                    # pacing note, not a freeze. Only the pump-less path is a real main-thread block.
                    if getattr(self, "_decision_pumped", False):
                        log(f"   [roam] DECISION THINK: {_choose_ms:.0f}ms (world stayed live — "
                            f"pumped idle, not a freeze)")
                    else:
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
            # ROAD-BENCH-XP (PASS 3 NEW#1): on a forward-march leg with an under-milestone bench, lead
            # with the weak mon so it banks participation XP from this leg's road battles; the disarm in
            # the finally restores the ace so the weak lead never outlives the leg. Fail-safe throughout.
            _road_xp = self._road_bench_xp_arm(pick, state)
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
            finally:
                if _road_xp:
                    self._road_bench_xp_disarm()
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
            # BENIGN-STILL OUTCOMES (night shift 9, the SABRINA/SCOPE WARN class): real work or a
            # deliberate yield that happens not to move her is NOT a silent failure — questline
            # talks/steps ARE the errand (pruning head_to_gym mid-questline stalls the forward
            # drive), and need_heal/healed_retry are the action handing control back on purpose.
            # Only genuine "returned without doing anything" outcomes feed the streak/dead set.
            _benign_still = ("arrived", "badge", "caught", "questline_talked",
                             "questline_worked_room", "questline_step_done", "questline_done",
                             "questline_passthrough", "questline_deeper", "questline_entered",
                             "questline_flash", "questline_catch",   # the Flash errand IS the work
                             "questline_strike_done", "questline_strike_exit_wip",  # the strike IS the work
                             "questline_prefight_heal",   # pre-rival Center tap IS the work (night shift 4)
                             "need_heal", "healed_retry")
            if _is_move_pick and not _moved and out not in _benign_still:
                self._nomove_streak += 1
                self._dead_moves.add(pick)
                # F-11 STRUCTURAL failure: no_gym_route / questline_no_route mean no route EXISTS
                # from this map — a few tiles of unrelated movement can't change that, so remember
                # it PER MAP (the prune consults this set independent of the movement-cleared one).
                # "no_route" (a plain travel pick with no walk route from this map) is just as
                # structural as no_gym_route — without it here, any small movement (a heal
                # excursion) cleared _dead_moves and resurrected the same doomed travels (the
                # banked_E4 Saffron heal↔dead-travel oscillation, night shift 9).
                if out in ("no_gym_route", "questline_no_route", "questline_arrived_no_target",
                           "no_route", "no_path"):
                    _sm = tuple(_pos0[0])
                    self._dead_moves_structural.setdefault(_sm, set()).add(pick)
                    self._dead_route_strikes[(_sm, pick)] = \
                        self._dead_route_strikes.get((_sm, pick), 0) + 1
                    # SPLIT-MAP: head_to_gym died on the map it just rode into → the SOURCE
                    # map's head_to_gym leg IS the dead end's feeder (Route 19↔20 at Seafoam,
                    # Route 23's halves at Victory Road). Park it too, for the session.
                    if (pick == "head_to_gym" and self._last_hg_leg
                            and tuple(self._last_hg_leg[1]) == _sm
                            and tuple(self._last_hg_leg[0]) != _sm):
                        _src = tuple(self._last_hg_leg[0])
                        self._dead_moves_structural.setdefault(_src, set()).add(pick)
                        self._dead_route_strikes[(_src, pick)] = max(
                            2, self._dead_route_strikes.get((_src, pick), 0) + 1)
                        self._dead_route_strikes[(_sm, pick)] = max(
                            2, self._dead_route_strikes[(_sm, pick)])
                        log(f"   [roam] !! SPLIT-MAP DEAD ROAD: head_to_gym dead on {_sm} right "
                            f"after riding in from {_src} — BOTH legs parked for the session "
                            f"(strikes persist across visits)")
                        if (_src, _sm) not in self._split_map_beat_done:
                            self._split_map_beat_done.add((_src, _sm))
                            self.on_event("okay, this road just dead-ends into water and rock — "
                                          "no getting through here the straight way. I'll find "
                                          "another route and come back when I can cross.",
                                          kind="travel", tier=2)
                log(f"   [roam] !! SILENT NO-MOVE: '{pick}' returned '{out}' but her position never changed "
                    f"(streak {self._nomove_streak}, dead routes {sorted(self._dead_moves)}) — not moving "
                    f"her; ALL dead routes get pruned next tick so she must do something that works")
                if self._active_objective and pick == self._active_objective.get("action"):
                    self._active_objective = None
            elif _moved:
                self._nomove_streak = 0
                self._dead_moves.clear()
                if tuple(_pos0[0]) != tuple(_pos1[0]):
                    if pick == "head_to_gym":
                        # remember the leg so a structural fail on arrival can park its feeder
                        self._last_hg_leg = (tuple(_pos0[0]), tuple(_pos1[0]))
                    # left the map -> its structural dead routes get ONE retry when she returns —
                    # EXCEPT picks with ≥2 strikes (proven dead across visits: the split-map class)
                    _lm = tuple(_pos0[0])
                    _kept = {k for k in self._dead_moves_structural.get(_lm, set())
                             if self._dead_route_strikes.get((_lm, k), 0) >= 2}
                    if _kept:
                        self._dead_moves_structural[_lm] = _kept
                        log(f"   [roam] structural dead routes KEPT on {_lm} across exit "
                            f"(≥2 strikes): {sorted(_kept)}")
                    else:
                        self._dead_moves_structural.pop(_lm, None)
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
    def run(self, objectives, _recovered_retry=False, start_obj=0):
        for i, obj in enumerate(objectives):
            if i < start_obj:                    # INTRA-SEGMENT RESUME: skip objectives a progress
                continue                         # checkpoint proves are already done (state reloaded)
            kind, label = obj[0], obj[-1]
            _t0 = time.time()
            # SEGMENT RUNG ANCHOR (couch fix-pass 1, 2026-07-08): roam ticks bank
            # _last_good_state; segments never did — so the recovery rung below had no
            # candidate on a fresh run. Each objective banks its own start (a savestate
            # rewind = a clean, idempotent objective retry).
            try:
                self._last_good_state = self.b.save_state()
            except Exception as _sae:
                log(f"   !! objective-start anchor failed ({_sae!r}) — rung degraded (LOUD)")
            log(f"OBJECTIVE {i+1}/{len(objectives)}: {kind} - {label}  "
                f"[at map={tv.map_id(self.b)} coords={tv.coords(self.b)}]")
            if kind == "OPENING":
                self._title_to_newgame()             # fresh ROM -> title -> New Game -> Oak's intro
                # QUIET-WINDOW FIX (2026-07-07 rehearsal finding #1): the starter was a silent
                # default-0 Bulbasaur — the oracle seam existed on both ends but OPENING never
                # called it. Her PARTNER for the run is the constitution's purest choice; route it
                # through her soul BEFORE the hands move. Headless/no-bot -> old default 0.
                m = self.drive_opening(obj[1], obj[2],
                                       starter_choice=self._choose_starter_soul())
                out = "complete" if m == PALLET else "stuck"
            elif kind == "WALK_TO_MAP":
                out = self.walk_to_map(obj[1], obj[2])
            elif kind == "ADVANCE_NORTH":
                # The one ADVANCE_NORTH leg (Viridian -> Route 2 -> Viridian Forest -> Pewter) is a
                # long grass crossing far from any Center — FLEE wilds + suppress the mid-leg heal
                # bounce so a thin solo starter crosses the Forest in one go instead of heal-looping
                # back to Viridian forever (2026-07-08 night train; the Forest north gate is reachable,
                # verified). A genuine death still propagates -> blackout-recovery re-runs the leg.
                out = self.advance_north(obj[1], flee_wilds=True, suppress_heal=True)
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
            elif kind == "GRIND_PRE_BROCK":
                out = self.grind_pre_brock(obj[1])
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
                # COUCH FIX-PASS 1 (2026-07-08): the fresh spine died at Route 1 where free-roam
                # recovers — the segment driver gets ONE rung of the roam tripwire ladder before
                # the STOP: an IMPOSSIBLE STAND (partial-void class — enclosed own-tile right
                # after a map flip) void-recovers to the objective-start anchor and the segment
                # retries once from this objective. A legit stuck still stops LOUD; the retry
                # flag caps recursion at depth 1 (one recovery per segment run).
                if not _recovered_retry and out in ("stuck", "no_path"):
                    _rec = False
                    try:
                        if self._impossible_stand():
                            log(f"   !! objective {kind} stuck on an IMPOSSIBLE STAND — "
                                f"void-recovering (segment-driver rung)")
                            _rec = bool(self._void_recover())
                    except Exception as _rex:
                        log(f"   !! segment recovery rung crashed: {_rex!r} (LOUD) — stopping honestly")
                    if _rec:
                        log(f"   segment driver RECOVERED — retrying from objective {i+1} ({kind}) "
                            f"at map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
                        return self.run(list(objectives[i:]), _recovered_retry=True)
                log(f"!! CAMPAIGN STOP (loud) at objective {i+1} ({kind}): {out}")
                return f"stopped:{kind}:{out}"
            # INTRA-SEGMENT PROGRESS (Fix C): this objective COMPLETED — bank a rolling resume point so a
            # LATER failure in this segment resumes from HERE, not from the segment start / bedroom. Only
            # inside a segment run (self._seg_progress set by run_segments) and not mid-recovery-recursion
            # (the slice there loses the absolute index). Best-effort; never blocks the objective flow.
            if SEG_PROGRESS_ENABLED and not _recovered_retry and getattr(self, "_seg_progress", None):
                try:
                    self._bank_seg_progress(i)
                except Exception as _spe:
                    log(f"   !! seg-progress bank skipped (obj {i+1}): {_spe} (LOUD)")
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

    # ── INTRA-SEGMENT PROGRESS (Fix C) ──────────────────────────────────────────────────────────
    def _seg_progress_paths(self, seg_ckpt):
        """The (state, meta) paths for a segment's rolling intra-segment progress checkpoint. Lives in
        self.ckpt_dir next to the segment's own checkpoint, named <segckpt-stem>.progress.{state,json}."""
        stem = seg_ckpt[:-6] if seg_ckpt.endswith(".state") else seg_ckpt
        return (os.path.join(self.ckpt_dir, stem + ".progress.state"),
                os.path.join(self.ckpt_dir, stem + ".progress.json"))

    def _bank_seg_progress(self, obj_index) -> bool:
        """Save a rolling resume point AFTER a completed objective so a later failure in this segment
        resumes locally. HARD GUARD (same as _save_checkpoint): a WORKSHOP run is physically forbidden to
        write into states/kira/ — only a SHOW run banks there. Atomic (.tmp + os.replace)."""
        ctx = getattr(self, "_seg_progress", None)
        if not ctx:
            return False
        if not self.show_mode and os.path.abspath(self.ckpt_dir).startswith(os.path.abspath(STATES_KIRA)):
            return False                          # firewall: never let a workshop run touch the kira spine
        state_p, meta_p = self._seg_progress_paths(ctx["seg_ckpt"])
        try:
            import json as _json
            os.makedirs(self.ckpt_dir, exist_ok=True)
            data = self.b.save_state()
            tmp = state_p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, state_p)              # atomic — a kill mid-write can't corrupt the resume point
            meta = {"seg_index": ctx["seg_index"], "seg_name": ctx["seg_name"], "obj_done": obj_index,
                    "map": list(tv.map_id(self.b)), "coords": list(tv.coords(self.b)),
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(meta_p, "w", encoding="utf-8") as f:
                _json.dump(meta, f, ensure_ascii=False, indent=2)
            log(f"   ⛳ SEG-PROGRESS [{ctx['seg_name']} obj {obj_index+1}] -> "
                f"{os.path.basename(self.ckpt_dir)}/{os.path.basename(state_p)}  "
                f"(map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            return True
        except Exception as e:
            log(f"   !! SEG-PROGRESS SAVE FAILED ({ctx['seg_name']} obj {obj_index+1}): {e} (LOUD)")
            return False

    def _clear_seg_progress(self, seg_ckpt):
        """Delete a segment's intra-segment progress files (called when the whole segment completes — its
        real checkpoint now supersedes the rolling one). Best-effort."""
        for p in self._seg_progress_paths(seg_ckpt):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                log(f"   [seg-progress] clear skipped ({os.path.basename(p)}): {e}")

    def _read_seg_progress(self, seg_ckpt, seg_index):
        """If a valid intra-segment progress checkpoint exists for THIS segment, return (state_path,
        obj_done); else None. Guards on seg_index so a stale progress from another segment is ignored."""
        state_p, meta_p = self._seg_progress_paths(seg_ckpt)
        if not (os.path.exists(state_p) and os.path.exists(meta_p)):
            return None
        try:
            import json as _json
            with open(meta_p, encoding="utf-8") as f:
                meta = _json.load(f)
            if meta.get("seg_index") != seg_index:
                return None
            return (state_p, int(meta.get("obj_done", -1)))
        except Exception as e:
            log(f"   [seg-progress] unreadable ({os.path.basename(meta_p)}): {e} — ignoring")
            return None

    # ── VOID-CORE CLASS KILLERS (2026-07-08 night shift — QW-4 diagnosis, repro'd clean in
    # recon_voidcore.py + frame-verified in recon_voidlook.py: the "dead world" IS the FireRed
    # TITLE SCREEN. The game can REBOOT ITSELF to title mid-run (the post-credits SoftReset class —
    # the dead states carry a DIFFERENT saveblock randomization than the live session, which only a
    # game boot produces). The QW-4 summit watch then PLAYED the title screen for whole ticks
    # (head_to_gym/heal offered, oracle calls burned) and STAGE-SAVED it over its campaign anchor —
    # and sanctity BLESSED the poisoned bundle. These three methods kill the whole class regardless
    # of what triggers the reboot. ─────────────────────────────────────────────────────────────────
    def _world_lost(self):
        """TRUE when live reads show the DEAD-WORLD signature: map (0,0) + coords None + party 0 +
        zero badges — the title screen after an in-game reboot. NO legit mid-roam moment reads like
        this (even a fresh bedroom start has a real map id, and free_roam only ever runs with a
        party). Cheap (4 reads); any read fault -> False (never block the loop on a probe)."""
        try:
            return (tv.map_id(self.b) == (0, 0)
                    and tv.coords(self.b) is None
                    and self.b.rd8(ram.GPLAYER_PARTY_CNT) == 0
                    and not any(self.has_badge(0x820 + i) for i in range(8)))
        except Exception:
            return False

    def _impossible_stand(self):
        """The PARTIAL-void signature (night shift 10): she 'stands' on a non-walkable tile
        with ZERO walkable/surfable neighbours — a door mat has an open front; no legit stand
        reads fully enclosed. Wears a plausible map id (Pallet, party 6, badges 8), so
        _world_lost is blind to it. Overworld-gated; False on any read flake."""
        try:
            cur = tuple(tv.coords(self.b) or ())
            if not cur or tuple(tv.map_id(self.b))[0:1] != (3,):
                return False
            g = tv.Grid(self.b)
            nbrs = [(cur[0] + dx, cur[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            return (not g.walkable(*cur) and not g.is_water(*cur)
                    and all(not g.walkable(*t) and not g.is_water(*t) for t in nbrs))
        except Exception:
            return False

    def _void_recover(self):
        """The world is GONE (title screen mid-run). Reload the newest REAL state we hold:
        the in-memory last-good snapshot first, else the on-disk campaign anchor. Each candidate is
        re-probed after load — a poisoned candidate (the summit banked the title screen!) is skipped
        LOUD, never trusted. Returns True on a verified-real world."""
        candidates = []
        if getattr(self, "_last_good_state", None):
            candidates.append(("last-good snapshot", lambda: self._last_good_state))
        _anchor = os.path.join(STATES_CAMPAIGN, CAMPAIGN_SAVE)
        if os.path.exists(_anchor):
            candidates.append(("campaign anchor", lambda: open(_anchor, "rb").read()))
        for label, get in candidates:
            try:
                self.b.load_state(get())
                self._wait_overworld()
                if not self._world_lost():
                    log(f"   [roam] !!!! VOID-CORE RECOVERED via {label}: world restored at "
                        f"{tv.map_id(self.b)}@{tv.coords(self.b)} "
                        f"party={self.b.rd8(ram.GPLAYER_PARTY_CNT)} (LOUD)")
                    return True
                log(f"   [roam] !! VOID-CORE: {label} is ITSELF a dead world (poisoned bank) — "
                    f"skipping it (LOUD)")
            except Exception as e:
                log(f"   [roam] !! VOID-CORE: reload via {label} failed ({e}) — trying next")
        return False

    def _save_campaign(self, reason="tick") -> bool:
        """BATCH 5 PHASE 1 — bank her LIVING campaign save (the Sherpa-timeline anchor): the single
        savestate that the next GO resumes from, so she keeps climbing from where she actually is and is
        never rappelled back to a frozen fragment. Writes ATOMICALLY (temp + os.replace) so a kill/crash
        mid-write can never corrupt the anchor (a half-written save would lose the whole climb). Lands in
        states/campaign/ ONLY — physically separate from the workshop fragments (kept as fallbacks) and
        the canonical states/kira/ spine. LOUD on success + failure (Constraint #3)."""
        if self._world_lost():
            # POISON GUARD (void-core class): NEVER anchor a dead world — the QW-4 summit run
            # overwrote its sandbox anchor with the TITLE SCREEN and the next resume would have
            # spawned into nothing. Refusing is loud; the previous anchor stays authoritative.
            log(f"   !! CAMPAIGN SAVE REFUSED [{reason}]: the live world reads DEAD (title-screen "
                f"signature — map (0,0), party 0). NOT poisoning the anchor (LOUD)")
            return False
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

    def _ckpt_label(self, reason):
        """Build a filesystem-safe checkpoint label: <ts>_<place>_<badges>b_<playtime>[_<reason>].
        Pure reads; every field degrades gracefully so a label can never fail the bank."""
        import re as _re
        ts = time.strftime("%Y%m%d_%H%M%S")
        try:
            place = self._place_name(tv.map_id(self.b)) or "unknown"
        except Exception:
            place = "unknown"
        place = _re.sub(r"[^A-Za-z0-9]+", "-", str(place)).strip("-").lower()[:28] or "unknown"
        try:
            badges = sum(1 for i in range(8) if self.has_badge(0x820 + i))
        except Exception:
            badges = 0
        try:
            secs = self._playthrough_elapsed()
            pt = f"{int(secs // 3600)}h{int((secs % 3600) // 60):02d}m" if secs is not None else "0h00m"
        except Exception:
            pt = "0h00m"
        rs = _re.sub(r"[^A-Za-z0-9]+", "-", str(reason)).strip("-").lower()[:20]
        return f"{ts}_{place}_{badges}b_{pt}" + (f"_{rs}" if rs else "")

    def _auto_checkpoint(self, reason="periodic") -> bool:
        """DENSE AUTO-CHECKPOINT — copy the CURRENT campaign sanctity bundle into a new growing, labeled
        dir under states/campaign/checkpoints/, so a dev can hop to JUST BEFORE any bug (not restart from
        the bedroom). Assumes the canonical bundle (kira_campaign.state + the four sidecars) was just
        written by _save_campaign()+_continuity_save(); copies those files, remapping pokemon_soul.json ->
        the bundle's soul.json. Written to a .partial dir then atomically renamed, so a kill mid-copy never
        leaves a half-bundle a reload could trust. Prunes to CKPT_KEEP. Best-effort + LOUD (Constraint #3).

        FIREWALL: writes ONLY under STATES_CAMPAIGN (the dev/campaign line, or a sandbox override). It is
        structurally incapable of writing the sacred states/kira/ spine — and refuses loudly if the
        campaign root ever resolved onto the kira lineage (belt-and-braces)."""
        import shutil as _shutil
        if os.path.abspath(STATES_CAMPAIGN) == os.path.abspath(STATES_KIRA) or \
                os.path.abspath(STATES_CAMPAIGN).startswith(os.path.abspath(STATES_KIRA) + os.sep):
            log("   !! AUTO-CKPT REFUSED: campaign root resolves onto the sacred states/kira/ spine — "
                "the dev checkpoint history must NEVER touch the livestream timeline (LOUD).")
            return False
        # the exact bundle a --resume free-roam / watch.py sandbox needs. soul lives as pokemon_soul.json
        # canonically but the bundle filename is soul.json (watch.py remaps it back on spawn).
        bundle = [(CAMPAIGN_SAVE, CAMPAIGN_SAVE), ("world_model.json", "world_model.json"),
                  ("strat_memory.json", "strat_memory.json"), ("journey_core.json", "journey_core.json"),
                  ("pokemon_soul.json", "soul.json")]
        optional = ("dialogue_hints.json", "team_plan_state.json")   # Part-C rule 17: plan HISTORY rides along
        try:
            root = os.path.join(STATES_CAMPAIGN, "checkpoints")
            os.makedirs(root, exist_ok=True)
            label = self._ckpt_label(reason)
            final = os.path.join(root, label)
            partial = final + ".partial"
            if os.path.exists(partial):
                _shutil.rmtree(partial, ignore_errors=True)
            os.makedirs(partial, exist_ok=True)
            # REQUIRED bundle files: the state must exist (no state = not a resumable checkpoint).
            src_state = os.path.join(STATES_CAMPAIGN, CAMPAIGN_SAVE)
            if not os.path.exists(src_state):
                log(f"   !! AUTO-CKPT SKIPPED [{reason}]: no {CAMPAIGN_SAVE} on disk yet (LOUD)")
                _shutil.rmtree(partial, ignore_errors=True)
                return False
            copied = []
            for src_name, dst_name in bundle:
                src = os.path.join(STATES_CAMPAIGN, src_name)
                if os.path.exists(src):
                    _shutil.copy2(src, os.path.join(partial, dst_name))
                    copied.append(dst_name)
            for f in optional:                              # ride-along intel ledger etc. (never required)
                src = os.path.join(STATES_CAMPAIGN, f)
                if os.path.exists(src):
                    _shutil.copy2(src, os.path.join(partial, f))
            # tiny metadata so the lister can label without re-parsing the .state
            try:
                import json as _json
                meta = {"label": label, "reason": reason, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "map": list(tv.map_id(self.b)), "coords": list(tv.coords(self.b)),
                        "badges": sum(1 for i in range(8) if self.has_badge(0x820 + i)),
                        "party": self.b.rd8(ram.GPLAYER_PARTY_CNT),
                        "playtime_s": self._playthrough_elapsed()}
                with open(os.path.join(partial, "checkpoint.json"), "w", encoding="utf-8") as f:
                    _json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as _me:
                log(f"   [auto-ckpt] metadata write skipped: {_me}")
            if os.path.exists(final):
                _shutil.rmtree(final, ignore_errors=True)
            os.replace(partial, final)                      # atomic: the dir appears complete or not at all
            # the checkpoint is COMMITTED at this point — a console-encoding hiccup in the success log must
            # NOT flip a written bundle to "FAILED" (guard it), and pruning proceeds regardless.
            try:
                log(f"   📌 AUTO-CHECKPOINT [{reason}] -> checkpoints/{label}  ({len(copied)} bundle file(s))")
            except Exception:
                pass
            self._prune_checkpoints(root)
            return True
        except Exception as e:
            log(f"   !! AUTO-CHECKPOINT FAILED [{reason}]: {e} (LOUD) — the rolling anchor is unaffected")
            return False

    def _prune_checkpoints(self, root):
        """Keep only the last CKPT_KEEP checkpoint dirs (ts-prefixed names sort chronologically). Also
        sweeps orphaned *.partial dirs from an interrupted copy. Best-effort."""
        import shutil as _shutil
        try:
            entries = []
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if not os.path.isdir(p):
                    continue
                if name.endswith(".partial"):
                    _shutil.rmtree(p, ignore_errors=True)   # orphaned interrupted copy
                    continue
                entries.append(name)
            entries.sort()                                  # chronological (ts prefix)
            for name in entries[:-CKPT_KEEP] if CKPT_KEEP > 0 else []:
                _shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        except Exception as e:
            log(f"   [auto-ckpt] prune skipped: {e}")

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
            elif state.get("post_game"):
                long = "Enjoy the post-game as Champion (Cerulean Cave, the Pokédex)"
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
            elif state.get("post_game"):
                medium = "Head out of the league, heal up, and pick the next adventure"
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
                 "like. THE ARC: build a team -> beat the 8 gyms prepared -> Elite Four -> credits. "
                 # F-2 (watch-notes): kill the "wait, who told them my name?" tic — a cute beat ONCE,
                 # a broken-record meta-observation when NPCs greet her by name every town.
                 "ALSO NORMAL: you entered your own name (and your rival's) when the game began — people "
                 "in this world knowing your names is ordinary, not a mystery to react to. ")
        # team-health nudge keyed on her ACTUAL party (capability-not-script: she decides what to do).
        party = state.get("party") or []
        if party and not state.get("post_game"):     # a Champion's victory lap is never "walled"
            lvls = sorted(m["level"] for m in party)
            if len(party) < 3 or (len(party) >= 2 and lvls[0] <= max(6, lvls[-1] - 12)):
                model += ("RIGHT NOW your team is thin/lopsided (a strong lead but weak or too few others) — "
                          "that's why tough trainers wall you. Building up a real squad (catch a good new "
                          "teammate, level the weak ones) is how a prepared trainer gets past these. ")
        spine = model + ("THE MAIN QUEST IS A FIXED PATH you're always on: beat the 8 Gym Leaders in order, "
                         "then the Elite Four, then you win the game. ")
        # GOAL-PINNED WATCH (2026-07-08): a spawn-time era objective dominates — she pursues THIS
        # moment (the thing Jonny wants to watch), not the post-game victory lap. Composes with the
        # gym/E4 context below (which still grounds WHERE that objective lives).
        if state.get("watch_goal"):
            spine += (f"⚡ BUT YOUR ONE FOCUS RIGHT NOW, above everything else: {state['watch_goal']}. "
                      f"This is a focused moment — go straight at that objective, don't drift onto "
                      f"side-quests, grinding for its own sake, or any victory-lap wandering. Pursue it "
                      f"directly until it's done. ")
        if ng:
            spine += (f"You're on GYM {bc + 1} of 8 — next is {ng['leader']} in {ng['city']}. The road "
                      f"forward leads there: until you've won you are ALWAYS making progress toward the "
                      f"next gym. Grinding/catching/detours are fine — but ONLY with a clear purpose, and "
                      f"then you GET BACK ON THE ROAD to {ng['city']}. Never just circle the same grass.")
            # P-1(b) ANTICIPATION FOLKLORE (couch fix-pass 1): what she's HEARD about the gym
            # ahead — trainer-talk a first-timer picks up, never walkthrough knowledge. Feeds
            # the forward-musing habit ("I hear the first gym is rock-type… that's bad for you,
            # buddy"). KB layer, one line, her voice decides if/when to say it.
            _folk = self._GYM_FOLKLORE.get(ng["leader"])
            if _folk:
                spine += f" Word on the road about {ng['leader']}: {_folk}"
        elif state.get("post_game"):
            # POST-CREDITS (the summit-watch residual): never tell the CHAMPION the E4 is still
            # ahead — the game is beaten. Point her at the victory lap instead (walk OUT of the
            # league, heal, then the post-game arcs: Cerulean Cave / the Pokédex).
            spine += ("YOU WON — you're the CHAMPION and the credits have rolled. This is the "
                      "post-game victory lap: head OUT of the league building, heal the team up, "
                      "and pick your own adventures now (Cerulean Cave just opened to champions, "
                      "and the Pokédex is far from full). Nothing is required anymore — this is "
                      "your world to enjoy.")
        else:
            spine += "All 8 badges are yours — the Elite Four is the final challenge before the credits."
            # E4 FOLKLORE SURFACING FIX (2026-07-08 mega-batch): the folklore path was keyed on `next_gym`,
            # which is None at 8 badges — so the run's CLIMAX (the E4) got NO 'word on the road'. Surface a
            # compact rumor beat here for the two most memorable seats (the opener + the dragon master); the
            # StrategicPlanner adds the personal, actionable prep (level MY Lapras for Lance).
            _e4_open = self._GYM_FOLKLORE.get("Lorelei")
            _e4_lance = self._GYM_FOLKLORE.get("Lance")
            if _e4_open or _e4_lance:
                spine += (" Word on the road about the Elite Four: "
                          + "; and ".join(x for x in (_e4_open, _e4_lance) if x) + ".")
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
        if pick and pick.startswith("greet:"):
            return f"Someone here matters to me — saying hi before anything else{nxt}."
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
            # TIMELINE-AWARE HEALTH (2026-07-08 dashboard fix): write to the ACTIVE timeline's own
            # file — SHOWTIME (show_mode) → states/kira/health.json, SHERPA → states/campaign/
            # health.json — so the dashboard panel never shows the other timeline's stale snapshot.
            health["timeline"] = "kira" if getattr(self, "show_mode", False) else "sherpa"
            _hdir = STATES_KIRA if getattr(self, "show_mode", False) else STATES_CAMPAIGN
            _hpath = os.path.join(_hdir, "health.json")
            os.makedirs(_hdir, exist_ok=True)
            tmp = _hpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(health, f)
            os.replace(tmp, _hpath)
        except Exception as e:
            log(f"   [health] publish skipped: {e}")

    def publish_health_tick(self, now=None):
        """Throttled cockpit-health publish, called EVERY FRAME from play_live's render loop so BOTH
        timelines (free_roam=sherpa AND run/run_segments=showtime) keep the dashboard panel live —
        previously only free_roam published, so a SHOWTIME run left the panel showing stale SHERPA
        data. Writes at most every ~2s; timeline-aware via _publish_health. Best-effort, never raises."""
        now = now or time.time()
        if now - getattr(self, "_last_health_tick_ts", 0.0) < 2.0:
            return
        self._last_health_tick_ts = now
        try:
            state = self.read_live_state()
            if state.get("badge_count", 0) > getattr(self, "_ht_prev_badges", -1):
                self._ht_last_badge_ts = now
                self._ht_prev_badges = state.get("badge_count", 0)
            if not getattr(self, "_ht_run_start_ts", None):
                self._ht_run_start_ts = now
            self._publish_health(state.get("progress") or "GREEN", state,
                                 getattr(self, "_ht_last_badge_ts", None), self._ht_run_start_ts)
        except Exception:
            pass

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
        try:
            # Part-C rule 17: resume the team-plan HISTORY (what she's caught/evolved/taught DELIBERATELY),
            # so a hard-kill mid-build doesn't lose the plan's memory. assess() re-derives live slot-status
            # from the party each tick, so this is additive continuity, never load-bearing for correctness.
            if self.team_planner.load(STATES_CAMPAIGN):
                log("   [teamplan] resumed plan-state from the campaign bundle")
        except Exception as e:
            log(f"   [teamplan] !! continuity load failed: {e} (LOUD)")

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
        try:
            if getattr(self.team_planner, "state", None) is not None:   # Part-C rule 17: bank the plan HISTORY
                self.team_planner.save(STATES_CAMPAIGN)
        except Exception as e:
            log(f"   [teamplan] !! continuity save failed: {e} (LOUD)")
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
        """PHASE 4 + MEMORY-MAGIC (2026-07-08) — assemble her Pokémon-journey continuity for the core
        seam, written FIRST PERSON as HER lived experience (never third-person 'she's playing' — that
        distancing is exactly what made chat-Kira narrate her own game as if someone ELSE were at the
        controls). Carries WHERE she is, her LIVE TEAM as relationships (name/species/level + how each
        joined + how she feels), the GARY grudge, arc, and want. Emits a structured `roster` the core
        renders richly. Pure reads + the persisted soul/strat; any field that errors is omitted."""
        try:
            state = self.read_live_state()
        except Exception:
            state = {}
        party = state.get("party") or []
        lead = party[0]["species"] if party else None
        solo = (state.get("party_count") == 1)
        bonds = {}
        wants = []
        try:
            if self.soul is not None:
                bonds = self.soul.bonds or {}
                wants = list(self.soul.wants or [])
        except Exception:
            pass
        # RICH ROSTER — each LIVE teammate as a RELATIONSHIP, not a species string: cross the live
        # party with soul.bonds (her own record of who they are to her). Nickname when she named one;
        # the joining story (caught where / evolved from / gifted) + her note become the FEELING the
        # core can speak. This is the roster-as-relationship layer, carried to chat.
        roster = []
        _roster_phrases = []
        for m in party:
            sp = m.get("species")
            lvl = m.get("level")
            nick = note = caught = None
            hist = []
            try:
                for _k, v in bonds.items():
                    if isinstance(v, dict) and v.get("species") == sp:
                        nick = v.get("nickname"); note = v.get("note"); caught = v.get("caught")
                        hist = v.get("history") or []
                        break
            except Exception:
                pass
            _real_nick = nick if (nick and nick != sp) else None
            # STORY = the accreted felt HISTORY (roster-as-relationship) when present — the moments
            # that made this teammate hers — else the origin note. This is what lets chat-Kira speak
            # each Pokémon as a companion with a past, not a species string.
            _story = []
            if hist:
                _story = list(dict.fromkeys(hist))[-3:]          # last few distinct moments, in order
            else:
                if caught:
                    _story.append(f"caught {caught}")
                if note and note not in ("new", None):
                    _story.append(note)
            roster.append({"nickname": _real_nick, "species": sp, "level": lvl,
                           "story": "; ".join(_story) or None})
            _label = f"{_real_nick} the {sp}" if _real_nick else sp
            _label = f"{_label} (L{lvl})" if lvl else _label
            _roster_phrases.append(_label + (f" — {'; '.join(_story)}" if _story else ""))
        grudge = ""
        try:
            grudge = self.strat.rival_grudge_note()
        except Exception:
            pass
        # a compact, declarative FIRST-PERSON "my story right now" the core injects as lived experience
        bits = []
        if lead:
            if solo:
                bits.append(f"I'm playing Pokémon FireRed — I'm the one on this journey, at the controls. "
                            f"It's a solo run right now, just my {lead}, a wild one I bonded with.")
            else:
                bits.append(f"I'm playing Pokémon FireRed — this is MY run, I'm the one playing it, "
                            f"and my team is led by my {lead}.")
        # NOTE: the roster is NOT baked into the summary string — that string is only re-POSTed at
        # anchors and would go stale (the 'still says rattata' bug). The core renders the `roster`
        # field LIVE instead (single source of team truth), so the summary carries only identity/arc/
        # grudge/want, which age gracefully.
        # WATCH-SCOPED ERA SELF-MODEL (2026-07-08): when a goal-pin re-lives an EARLIER moment, her
        # accumulated arc/grudge ("I'm the Champion, I beat Gary") CONTRADICTS the era objective ("go
        # fight the E4") — the incoherence Jonny heard. When pinned, present the PRESENT-TENSE era
        # self-model and DROP the post-hoc arc/grudge; the bot also drops the Champion saga (watch_scoped).
        _wg = state.get("watch_goal")
        if _wg:
            bits.append(f"Right now, in this moment: {_wg}. That's what's in front of me — "
                        f"I haven't done it yet; I'm about to.")
        else:
            if state.get("arc"):
                bits.append(state["arc"])      # native second-person self-address (house style)
            if grudge:
                bits.append(grudge)            # ditto — 'you've met Gary 6 times' reads as her own voice
            if wants:
                bits.append(f"What I want right now: {wants[0]}.")
        return {
            "summary": " ".join(bits),
            "place": state.get("place"),
            "badge_count": state.get("badge_count"),
            "party_count": state.get("party_count"),
            "lead": lead,
            "solo": solo,
            "grudge": None if _wg else grudge,
            "arc": None if _wg else state.get("arc"),
            "roster": roster,
            "watch_scoped": bool(_wg),
            # TIMELINE LABEL (2026-07-08 firewall): show_mode → the KIRA showtime timeline
            # (states/kira), else the SHERPA working/completed run (states/campaign). The bot
            # scopes her journey memory per-timeline off this, so the Champion/Venusaur saga
            # never bleeds into the fresh Charmander run. Constant self, separate journeys.
            "timeline": "kira" if getattr(self, "show_mode", False) else "sherpa",
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

    def _street_gradient(self):
        """Hop distance of every KNOWN map to the overworld (group 3), reverse-BFS over the
        world model's WARP graph. Lets a nested-complex exit DESCEND toward the street
        instead of wandering local-greedy (ship run 19: from the 1F corridor the nearest
        warp led back INTO the 2F corridor; the gangway to the deck was 2 hops of foresight
        away). Unknown maps read as 999 — deprioritized, never fatal."""
        try:
            nodes = self.world.nodes
        except Exception:
            return {}
        import collections
        rev = collections.defaultdict(set)
        keys = set()
        for key, nd in nodes.items():
            keys.add(key)
            for _xy, dk in (nd.get("warps") or {}).items():
                rev[dk].add(key)
                keys.add(dk)
        dist, dq = {}, collections.deque()
        for key in keys:
            if key.startswith("3,"):
                dist[key] = 0
                dq.append(key)
        while dq:
            cur = dq.popleft()
            for prv in rev.get(cur, ()):
                if prv not in dist:
                    dist[prv] = dist[cur] + 1
                    dq.append(prv)
        return dist

    def _engage_exit_guard(self, mapid):
        """SEALED-BY-A-GUARD (2026-07-08 night shift 6, the hideout-B1F barrier truth from pret):
        a floor's walk region can be sealed by a TRAINER-GATED BARRIER — RocketHideout_B1F keeps
        a coll-3 wall at (20-21,19-21) until TRAINER_TEAM_ROCKET_GRUNT_12 falls; beating that
        guard runs setmetatile + a map redraw and the floor OPENS. So when every exit door reads
        walk-unreachable, talk to / fight the nearest reachable object event (once each per map,
        LIVE coords — wanderers move), then let the exit loop re-plan on the fresh grid.
        Returns True if someone was engaged (the caller re-checks the doors)."""
        if not hasattr(self, "_exit_engaged"):
            self._exit_engaged = set()
        try:
            g = tv.Grid(self.b)
            cur = tuple(tv.coords(self.b))
            cands = []
            for i in range(1, 16):
                o = self._OB + i * self._SZ
                if not (self.b.rd8(o) & 1) or (mapid, i) in self._exit_engaged:
                    continue
                c = (self.b.rds16(o + 0x10) - 7, self.b.rds16(o + 0x12) - 7)
                cands.append((abs(c[0] - cur[0]) + abs(c[1] - cur[1]), i, c))
            for _d, i, c in sorted(cands):
                for dx, dy, face in ((0, 1, "UP"), (0, -1, "DOWN"),
                                     (1, 0, "LEFT"), (-1, 0, "RIGHT")):
                    front = (c[0] + dx, c[1] + dy)
                    if front != cur:
                        if not g.walkable(*front):
                            continue
                        if not tv.bfs(g, cur, lambda t, f=front: t == f, walkable=g.walkable):
                            continue
                    self._exit_engaged.add((mapid, i))
                    log(f"   EXIT: floor sealed — engaging the guard: NPC #{i} at {c} "
                        f"(from {front})")
                    self._talk(front, face, "exit-guard")
                    if st.in_battle(self.b):
                        out = self.battle_runner()
                        log(f"   EXIT: guard battle -> {out}")
                    return True
        except Exception as _ge:
            log(f"   !! EXIT: guard engagement failed ({_ge}) — walking on")
        return False

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
        _elev_seen = set()
        # PERSISTENT ROW CURSOR (shift 6, the scope-regrade ride waste): the row schedule used
        # to reset to 0 every call, so every leave_building tick re-burned the default/current-
        # floor rows before reaching one that rides (the bag TRUE-row cursor law, applied here).
        if not hasattr(self, "_elev_rows"):
            self._elev_rows = {}
        grad = self._street_gradient()
        for _ in range(max_tries * 3):
            if tv.map_id(self.b)[0] == 3:
                return True
            before = tuple(tv.map_id(self.b))
            cur = tuple(tv.coords(self.b))
            # ELEVATOR CAR (2026-07-08 shift 5, the banked_SCOPE 412-wedge class): a room whose
            # EVERY warp is dynamic ((127,127)) is an elevator — its door just steps back onto
            # the boarding floor, so this loop ping-ponged car<->floor forever (hideout B4F).
            # Ride the panel — ALL remaining rows while aboard (a failed pick stays in the car;
            # falling through to the door candidates after one flake walked us out for nothing).
            try:
                import elevator_nav
                if elevator_nav.is_car(self.b):
                    rode = False
                    while True:
                        _row = self._elev_rows.get(before, 0)
                        if _row > 4:
                            self._elev_rows.pop(before, None)   # exhausted — fresh cycle next boarding
                            break
                        self._elev_rows[before] = _row + 1
                        log(f"   EXIT: elevator car {before} — riding the panel (row {_row})")
                        if elevator_nav.ride(self.b, self, _row, avoid=_elev_seen, log=log):
                            rode = True
                            break
                    if rode:
                        continue
                else:
                    _elev_seen.add(before)     # this floor has had its exit chance this call
            except Exception as _ee:
                log(f"   !! EXIT: elevator ride failed ({_ee}) — walking on")
            ws = tv.read_warps(self.b)
            cands = [tuple(w[0]) for w in ws]
            # STREET-FIRST (2026-07-06, the Vermilion Center/Cable-Club stall): each warp's DEST is
            # already known — a warp straight to the overworld (group 3) beats every interior hop.
            # Nearest-first alone ping-ponged the two Center floors via the escalator and then burned
            # travel budgets on the attendant-blocked Cable-Club room warps.
            street = {tuple(w[0]) for w in ws if w[1][0] == 3}
            # GRADIENT-SECOND (ship run 19): in a deep complex nothing on this floor reaches the
            # street directly — descend the warp-graph gradient toward group 3 instead of
            # nearest-first wandering (the exit re-entered the 2F corridor from the 1F corridor).
            _dest = {tuple(w[0]): tuple(w[1]) for w in ws}
            def _gd(t):
                d = _dest.get(t)
                return grad.get(f"{d[0]},{d[1]}", 999) if d else 999
            fresh = [w for w in cands if (before, w) not in taken]
            # within a tier, an ACTUATABLE warp (a behavior the directional table can fire) beats a
            # dead arrival-mat: the Center's exit row is (6,8)/(7,8)/(8,8) but only (7,8) is the 0x65
            # arrow that actually fires — trying the dead mats first burned a travel leg each.
            cands = sorted(fresh or cands,
                           key=lambda t: (t not in street,
                                          _gd(t),
                                          self._tile_behavior(*t) not in self._WARP_ENTRY,
                                          abs(t[0] - cur[0]) + abs(t[1] - cur[1])))
            moved = False
            for wt in cands:
                # ASYNC-WHITEOUT GUARD (koga_run3, the Fuchsia Center wedge ×6): a pending whiteout
                # warp can fire MID-attempt (she engaged Koga, "lost", and the respawn executed while
                # the first exit leg was walking) — the map silently changes under us and every
                # remaining candidate is a STALE coord on the wrong map. The per-attempt checks below
                # are short-circuited when the attempt itself fails ('_enter_directional_warp(wt) and
                # map != before'), so check the map UNCONDITIONALLY before burning a leg on the next
                # stale candidate; a changed map restarts the outer loop with fresh warps.
                if tuple(tv.map_id(self.b)) != before:
                    moved = True
                    break
                # SEALED-DOOR SKIP (shift 6, the B1F barrier lobby): don't burn a travel budget
                # + the enter_warp approach fan (≈6 travel wedges) on a door her feet provably
                # can't reach right now. NOT added to `taken` — the flood check is per-tick truth,
                # so the same door rides the moment the floor unseals (guard beaten, NPC moved).
                if not self._warp_hop_reachable(wt):
                    log(f"   EXIT: door {wt} walk-unreachable from {tv.coords(self.b)} — "
                        f"skipped (sealed pocket)")
                    continue
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
                # SEALED-BY-A-GUARD (shift 6): before giving up, the room's PEOPLE are the
                # door — a beaten guard can run setmetatile and unseal the floor. Engage one,
                # then re-plan on the fresh grid.
                if self._engage_exit_guard(before):
                    continue
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
        # INTRA-SEGMENT RESUME (Fix C): does the NEXT (in-progress) segment have a rolling progress
        # checkpoint from a deep failure last run? If so, resume from THERE (skip the objectives already
        # done) instead of replaying the whole segment from its start — a gym loss retries at the gym,
        # not from the bedroom. The progress state is strictly further than resume_cp (the PREVIOUS
        # segment's end), so it takes precedence.
        resume_start_obj = 0
        prog = self._read_seg_progress(segments[start].checkpoint, start) if (
            resume and SEG_PROGRESS_ENABLED and segments[start].checkpoint) else None
        if prog:
            state_p, obj_done = prog
            resume_start_obj = obj_done + 1
            log(f"RUN_SEGMENTS: RESUME (INTRA-SEGMENT) — '{segments[start].name}' progress checkpoint "
                f"present ({obj_done + 1} objective(s) done); loading it + running from objective "
                f"{resume_start_obj + 1} (NOT replaying the segment from the start)")
            with open(state_p, "rb") as f:
                self.b.load_state(f.read())
            for _ in range(40):
                self.b.run_frame()
            self.b.set_input_owner("agent")
        elif resume_cp:
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
            # INTRA-SEGMENT PROGRESS context: run() banks a rolling resume point after each completed
            # objective of THIS segment (Fix C). None (banking off) for a checkpoint-less segment.
            self._seg_progress = ({"seg_ckpt": seg.checkpoint, "seg_index": i, "seg_name": seg.name}
                                  if (SEG_PROGRESS_ENABLED and seg.checkpoint) else None)
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
                # Only the FIRST segment run resumes mid-segment (from a progress checkpoint on disk);
                # a blackout RE-RUN starts from objective 0 (re-navigating from the post-whiteout respawn,
                # the proven path). resume_start_obj is consumed once.
                _start_obj = resume_start_obj if (i == start and attempts == 0) else 0
                result = self.run(seg.objectives, start_obj=_start_obj)
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
            self._clear_seg_progress(seg.checkpoint)        # Fix C: the full checkpoint now supersedes
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
