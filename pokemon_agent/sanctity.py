"""sanctity.py — bank-time validation of the continuity bundle (2026-07-06 ride-along 0c).

THE ASYMMETRY THIS CLOSES: the mechanical side has three-state discipline + round-trip verifies; the
STORY side was write-and-hope — and it rotted silently (the canonical journey_core carried a mojibake'd
summary and had LOST the run-5 Gary win: "he's still got your number" about her proudest moment).
Story corruption now fails a bank the way a wedge fails a run.

Four check families, all read-only, all fast (pure JSON reads):
  1. SCHEMA    — every sidecar present, parses, and has its load-bearing keys/types.
  2. ENCODING  — no mojibake markers (double-encoded UTF-8: 'Ã', 'â€', U+FFFD) in any string value.
  3. TRUTH     — journey_core's grudge tally must MATCH the strat rival record (W-L cross-check);
                 badge_count cross-checked against the live read when the caller supplies one.
  4. MONOTONIC — vs the bundle being REPLACED (promotion guard): rival encounters, badge count, soul
                 bonds, world nodes must never DECREASE. (This is the rule that would have caught the
                 lost Gary win — runs 6-8 restarted from a pre-win bank and the promotion regressed.)

USAGE:
    ok, issues = sanctity.validate_bundle(bundle_dir, prev_dir=None, live_badges=None, log=print)
Callers: recon_longrun (bank step — a FAILED validation marks the bank DO-NOT-PROMOTE, loud) and
campaign._continuity_save (canonical writes — loud log, never crashes the save path).
"""
import json
import os
import re

SIDECARS = ("world_model.json", "strat_memory.json", "soul.json", "journey_core.json")
# OPTIONAL sidecars: ride the bundle when present (copy-if-exists at bank/promote/sandbox time)
# but are NEVER validation-required — old banks without them stay VALID and just start empty.
OPTIONAL_SIDECARS = ("dialogue_hints.json",)
_MOJIBAKE = ("Ã", "â€", "�")   # 'Ã', 'â€', replacement char
_TALLY_RE = re.compile(r"\((\d+)-(\d+)\)")


def _load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None, f"{name}: MISSING"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"{name}: UNPARSEABLE ({e})"


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _rival_tally(strat):
    enc = ((strat or {}).get("rival") or {}).get("encounters") or []
    w = sum(1 for e in enc if e.get("won"))
    return w, len(enc) - w, len(enc)


def validate_bundle(bundle_dir, prev_dir=None, live_badges=None, log=print):
    """Validate the continuity bundle in `bundle_dir`. Returns (ok, issues:list[str]).
    prev_dir: the bundle this one would REPLACE (usually states/campaign) -> monotonic checks.
    live_badges: the badge count read off the live/banked savestate, if the caller has it."""
    issues = []

    # 1 — SCHEMA ---------------------------------------------------------------------------------
    st_path = os.path.join(bundle_dir, "kira_campaign.state")
    if not os.path.exists(st_path):
        issues.append("kira_campaign.state: MISSING")
    elif os.path.getsize(st_path) < 100_000:
        issues.append(f"kira_campaign.state: suspiciously small ({os.path.getsize(st_path)} B)")
    data = {}
    for name in SIDECARS:
        obj, err = _load(bundle_dir, name)
        if err:
            issues.append(err)
            continue
        data[name] = obj
    world = data.get("world_model.json")
    strat = data.get("strat_memory.json")
    soul = data.get("soul.json")
    journey = data.get("journey_core.json")
    if world is not None and not isinstance(world.get("nodes"), dict):
        issues.append("world_model.json: no 'nodes' dict")
    if strat is not None and not isinstance(strat.get("losses"), dict):
        issues.append("strat_memory.json: no 'losses' dict")
    if soul is not None and (not isinstance(soul.get("bonds"), dict)
                             or not isinstance(soul.get("wants"), list)):
        issues.append("soul.json: bonds/wants malformed")
    if journey is not None and not (journey.get("summary") or "").strip():
        issues.append("journey_core.json: empty summary (her story is blank)")

    # 2 — ENCODING -------------------------------------------------------------------------------
    for name, obj in data.items():
        for s in _strings(obj):
            bad = [m for m in _MOJIBAKE if m in s]
            if bad:
                issues.append(f"{name}: MOJIBAKE {bad} in {s[:60]!r}")
                break

    # 3 — TRUTH ----------------------------------------------------------------------------------
    if strat is not None and journey is not None:
        w, l, n = _rival_tally(strat)
        grudge = journey.get("grudge") or ""
        if n >= 2:                      # the tally string only renders from the 2nd encounter on
            m = _TALLY_RE.search(grudge)
            if not grudge:
                issues.append(f"TRUTH: strat has {n} rival encounters but journey grudge is EMPTY")
            elif m and (int(m.group(1)), int(m.group(2))) != (w, l):
                issues.append(f"TRUTH: journey grudge says ({m.group(1)}-{m.group(2)}) but strat "
                              f"records {w}W-{l}L — her story disagrees with her memory")
        if live_badges is not None and journey.get("badge_count") is not None \
                and int(journey["badge_count"]) != int(live_badges):
            issues.append(f"TRUTH: journey badge_count={journey['badge_count']} but the savestate "
                          f"reads {live_badges}")

    # 4 — MONOTONIC vs the bundle being replaced --------------------------------------------------
    if prev_dir and os.path.isdir(prev_dir) and os.path.abspath(prev_dir) != os.path.abspath(bundle_dir):
        prev_strat, _ = _load(prev_dir, "strat_memory.json")
        prev_journey, _ = _load(prev_dir, "journey_core.json")
        prev_soul, _ = _load(prev_dir, "soul.json")
        prev_world, _ = _load(prev_dir, "world_model.json")
        if prev_strat is not None and strat is not None:
            _, _, n_prev = _rival_tally(prev_strat)
            _, _, n_new = _rival_tally(strat)
            if n_new < n_prev:
                issues.append(f"MONOTONIC: rival encounters would REGRESS {n_prev}->{n_new} "
                              f"(the lost-Gary-win class — this bank forgets story)")
        if prev_journey is not None and journey is not None \
                and (journey.get("badge_count") or 0) < (prev_journey.get("badge_count") or 0):
            issues.append(f"MONOTONIC: badge_count would REGRESS "
                          f"{prev_journey.get('badge_count')}->{journey.get('badge_count')}")
        if prev_soul is not None and soul is not None \
                and len(soul.get("bonds") or {}) < len(prev_soul.get("bonds") or {}):
            issues.append(f"MONOTONIC: soul bonds would REGRESS "
                          f"{len(prev_soul.get('bonds') or {})}->{len(soul.get('bonds') or {})}")
        if prev_world is not None and world is not None \
                and len(world.get("nodes") or {}) < len(prev_world.get("nodes") or {}):
            issues.append(f"MONOTONIC: world nodes would REGRESS "
                          f"{len(prev_world.get('nodes') or {})}->{len(world.get('nodes') or {})}")

    ok = not issues
    if ok:
        log("   [sanctity] bundle VALID (schema/encoding/truth/monotonic all clean)")
    else:
        log(f"   [sanctity] !! BUNDLE FAILED VALIDATION ({len(issues)} issue(s)) — DO NOT PROMOTE:")
        for i in issues:
            log(f"   [sanctity] !!   - {i}")
    return ok, issues
