#!/usr/bin/env bash
# fresh_go_watchdog.sh — PORTED from scratch (single-mission order 0a, 2026-07-13).
# Fresh bedroom->credits watchdog. Detached; survives shift ends. Resumes from banked_LIVE on crash;
# STOPS when banked_CREDITS appears (recon_longrun now emits the CREDITS outcome — order 1).
#
# COLD-START a fresh run: remove/rename /g/temp/longrun/banked_LIVE first so iteration 1 boots FRESH.
# LOG defaults to fresh_go_2.log; override with FRESH_GO_LOG=/path/to.log.
cd /g/JonnyD/NeuroAI_Bot/pokemon_agent || exit 2
LOG="${FRESH_GO_LOG:-/g/temp/longrun/fresh_go_2.log}"
LIVE=/g/temp/longrun/banked_LIVE/kira_campaign.state
export LONGRUN_BANK_EVERY_S=180
# NS#37: keep the over-level-before-sealeg lever OFF (it re-deferred the Cinnabar crossing every iter).
export POKEMON_OVERLEVEL_SEALEG=0
# Guarantee NO leaked POKEMON_ verify flags (a leaked flag caused prior treadmills). Clear every
# POKEMON_ var except the one set above. TEAM-DEPTH levers (BENCH_TO_MILESTONE etc.) now DEFAULT ON
# in campaign.py code, so scrubbing the env does NOT disable them — they ride on the code default.
for _v in $(env | grep -oE '^POKEMON_[A-Z_]+' | grep -v '^POKEMON_OVERLEVEL_SEALEG$'); do unset "$_v"; done
PY=../.venv/Scripts/python.exe
export PATH="/usr/bin:/bin:$PATH"   # ensure coreutils on PATH under a hidden/non-login launch
echo "=== fresh_go watchdog start ($(date '+%Y-%m-%d %H:%M:%S')) ===" >> "$LOG"
i=0
while [ "$i" -lt 200 ]; do
  i=$((i+1))
  if [ -d /g/temp/longrun/banked_CREDITS ]; then
    echo "=== CREDITS bank detected — stopping watchdog (iter $i) ===" >> "$LOG"
    break
  fi
  # BOOT: resume mid-game from banked_LIVE if present; else run the SCRIPTED bedroom->credits SPINE
  # (WAR NS#7): free_roam can NOT cross the Oak's-Parcel opening gate, so a cold run must use the
  # scripted spine (char-creation -> deliver_parcel -> advance_north -> ... -> Misty) then hand off
  # to free_roam for badge 3 -> credits.
  if [ -f "$LIVE" ]; then
    BOOT="$LIVE"
  else
    BOOT="FRESH"
  fi
  echo "=== launch iter $i boot=$BOOT ($(date '+%H:%M:%S')) ===" >> "$LOG"
  "$PY" -u recon_longrun.py "$BOOT" 120 >> "$LOG" 2>&1
  echo "=== recon_longrun exited rc=$? iter $i ===" >> "$LOG"
  # CARRY PROGRESS FORWARD: promote the newest banked_GOAL into banked_LIVE so each segment's win
  # compounds toward credits. ONLY banked_GOAL (a genuine next-objective advance) — never a
  # STALL/TIMEOUT bank (resuming into those replays the wedge). A CREDITS bank ends the loop above.
  newest=""
  for d in banked_GOAL; do
    s="/g/temp/longrun/$d/kira_campaign.state"
    if [ -f "$s" ] && [ "$s" -nt "$LIVE" ]; then
      if [ -z "$newest" ] || [ "$s" -nt "$newest" ]; then newest="$s"; fi
    fi
  done
  if [ -n "$newest" ]; then
    src=$(dirname "$newest")
    echo "=== carry-forward: $src -> banked_LIVE (advancing resume point) ===" >> "$LOG"
    cp -f "$src"/world_model.json "$src"/strat_memory.json "$src"/soul.json "$src"/journey_core.json /g/temp/longrun/banked_LIVE/ 2>>"$LOG"
    cp -f "$src"/kira_campaign.state /g/temp/longrun/banked_LIVE/ 2>>"$LOG"
  fi
  sleep 4
done
echo "=== watchdog loop ended ($(date '+%H:%M:%S')) ===" >> "$LOG"
