"""pokemon_search.py — BATCH 6 PHASE 5: SILENT "reach for the guide when stuck" (like Neuro).

The vision: when she hits a genuine gap — a wall she can't crack, a maze, "where do I even find
one of these" — she quietly checks, the way a kid reaches for the strategy guide. The VIEWER sees the
RESULT folded into her reasoning, IN CHARACTER — never "let me Google it." It is ENRICHMENT, never a
dependency: if search is unavailable or errors, she just carries on from her own knowledge.

FIREWALL: this module only PROVIDES a capability + returns data. It NEVER decides for her and NEVER
edits core. The free-roam loop folds a result into the oracle ctx via the same `place` seam survival/
shop/stuck already use — she reasons about it herself.

DEPENDENCY STATUS (reconned 2026-06-27): the Google key + CSE id are present in .env, but the Google
Cloud project does NOT have the Custom Search JSON API enabled (live check returned HTTP 403 "This
project does not have access to Custom Search JSON API"). So this is built but defaults OFF —
  1. enable "Custom Search API" for the project in the Google Cloud console, then
  2. set POKEMON_GUIDE_SEARCH=1
and it lights up. Until then every call is a logged no-op (enrichment-only, never blocks her).

COST-AWARE: a hard per-run search budget + a tick cooldown (a scarce, deliberate tool — protects the
API bill AND keeps her human; she doesn't spam the guide). Every search is LOGGED to .history so it's
surfaceable later (the comedy/intimacy "you can see her search history" beat).
"""
import os
import time

# Load .env in THIS process so the flag + Google key/cx are seen on EVERY launch path (the pokemon
# process doesn't otherwise load it; only the dashboard-spawned path inherits the bot's env). Best-effort.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

# default OFF until the dependency is confirmed live (see DEPENDENCY STATUS above)
GUIDE_ENABLED = os.getenv("POKEMON_GUIDE_SEARCH", "0") == "1"
# circuit-breaker: after this many consecutive unavailable/error results (e.g. a key whose project lacks
# the Custom Search API — a confirmed 403), disable the guide for the REST of this run + log LOUD once, so
# a not-yet-working key never keeps firing doomed calls. A fresh run re-reads the flag and retries, so the
# instant a working key is in place it lights up — no code change, no re-arm.
GUIDE_BREAKER = int(os.getenv("POKEMON_GUIDE_BREAKER", "2"))
# scarce by design: at most this many searches per run, and at least this many seconds between them.
GUIDE_BUDGET = int(os.getenv("POKEMON_GUIDE_BUDGET", "8"))
GUIDE_COOLDOWN_S = float(os.getenv("POKEMON_GUIDE_COOLDOWN_S", "90"))

# sentinels the existing GoogleSearch returns when it can't actually search — treated as "unavailable"
# so a broken dependency degrades to a clean no-op instead of polluting her reasoning.
_UNAVAILABLE = ("not configured", "error searching", "No search results")


class GuideSearch:
    """Her silent strategy-guide. Best-effort, budgeted, logged. All public methods are safe to call
    headless / with the dependency down — they just return None and log."""

    def __init__(self, log=print, enabled=None, budget=None, cooldown_s=None):
        self.log = log
        self.enabled = GUIDE_ENABLED if enabled is None else enabled
        self.budget = GUIDE_BUDGET if budget is None else budget
        self.cooldown_s = GUIDE_COOLDOWN_S if cooldown_s is None else cooldown_s
        self.spent = 0
        self._last_t = 0.0
        self.history = []          # [(query, reason, result_or_None)] — surfaceable later
        self._search_fn = None     # lazily resolved; None if deps/config missing
        self._consec_unavail = 0   # circuit-breaker: consecutive unavailable/error results this run
        self._tripped = False      # breaker tripped -> guide disabled for the rest of this run (logged once)

    def _resolve(self):
        """Import the existing Google search lazily so a missing dependency never breaks the run."""
        if self._search_fn is not None:
            return self._search_fn
        try:
            from kira.tools.web_search import GoogleSearch
            self._search_fn = GoogleSearch
        except Exception as e:
            self.log(f"   [guide] search unavailable (import/config): {e} — running without it")
            self._search_fn = False
        return self._search_fn

    def available(self):
        return self.enabled and bool(self._resolve())

    def search(self, query, reason="stuck"):
        """Quietly look something up. Returns a short result string, or None when unavailable / budget
        spent / on cooldown / on error. NEVER raises, NEVER blocks meaningfully (the call itself is the
        only latency, gated behind the cooldown + budget so it's rare). LOGS every attempt + records it
        in .history. The caller folds a non-None result into her reasoning in character."""
        if not self.enabled or self._tripped:
            return None
        fn = self._resolve()
        if not fn:
            return None
        now = time.time()
        if self.spent >= self.budget:
            self.log(f"   [guide] budget spent ({self.spent}/{self.budget}) — not searching (cost guard)")
            return None
        if self._last_t and (now - self._last_t) < self.cooldown_s:
            return None                                   # on cooldown — stay scarce, no log spam
        self._last_t = now
        self.spent += 1
        try:
            result = fn(query, 2)                          # the existing sync GoogleSearch (top 2 snippets)
        except Exception as e:
            self.log(f"   [guide] search errored ({e}) — carrying on from her own knowledge")
            self.history.append((query, reason, None))
            self._note_unavailable()
            return None
        if not result or any(s in result for s in _UNAVAILABLE):
            self.log(f"   [guide] SEARCH [{reason}] {query!r} -> (no usable result; dependency may be off)")
            self.history.append((query, reason, None))
            self._note_unavailable()
            return None
        self._consec_unavail = 0                           # a real hit — the dependency is healthy
        one_line = " | ".join(result.splitlines())[:300]
        self.log(f"   [guide] SEARCH [{reason}] {query!r} -> {one_line}")
        self.history.append((query, reason, result))
        return result

    def _note_unavailable(self):
        """Circuit-breaker: trip after GUIDE_BREAKER consecutive unavailable/error results (e.g. a key
        whose project lacks the API) so a not-yet-working dependency stops firing doomed calls this run."""
        self._consec_unavail += 1
        if self._consec_unavail >= GUIDE_BREAKER and not self._tripped:
            self._tripped = True
            self.log(f"   [guide] !! CIRCUIT-BREAKER TRIPPED after {self._consec_unavail} unavailable "
                     f"results — disabling the guide for THIS run (likely the Google key's project lacks "
                     f"the Custom Search API). A fresh run with a working key re-enables it automatically. (LOUD)")

    def history_brief(self, n=5):
        """The last few searches as a one-liner (for the 'see her search history' surfacing beat)."""
        rows = [f"{q} ({'hit' if r else 'miss'})" for q, _reason, r in self.history[-n:]]
        return "; ".join(rows)
