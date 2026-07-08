# cost_tracker.py — Per-session LLM API cost telemetry.
#
# Singleton (module-level _tracker instance).  Thread-safe accumulation;
# events are forwarded to the StreamLogger as type="llm_usage" entries.
#
# PRICING DICT — edit when model prices change.  Numbers are USD per million
# tokens ($/MTok).  Prices were current as of 2026-06 but WILL drift; this
# dict is the single source of truth.  Mark cache_read separately where the
# API offers discounted cache-read pricing.
#
# PURPOSE TAGS (free-form strings, appear in the shutdown summary):
#   voice         — main response path (claude_chat_inference / claude_chat_inference_stream)
#   interjection  — autonomous interjection voice lines
#   triage        — classification / routing (Groq / local)
#   vision        — Gemini screen-cap calls (flash-lite heartbeat / flash-preview escalated)
#   artifact      — session-end artifacts (lore, clips, summary via Opus)
#   background    — startup-brief, general-opinions, other background calls
#   local         — local Llama calls (token estimate; zero cost)
#   fallback      — substitution from a cloud model to local Llama

import json
import threading
import time
from typing import Optional

# ── Pricing ($/MTok) ─────────────────────────────────────────────────────────
# NOTE: prices drift.  Update this dict; it is the source of truth.
PRICE_TABLE: dict[str, dict] = {
    # Anthropic Claude
    "claude-sonnet-4-5":  {"in": 3.00,  "out": 15.00, "cache_read": 0.30},
    "claude-sonnet-4-5-20251219": {"in": 3.00, "out": 15.00, "cache_read": 0.30},
    "claude-opus-4-5":    {"in": 15.00, "out": 75.00, "cache_read": 1.50},
    "claude-opus-4-5-20251101": {"in": 15.00, "out": 75.00, "cache_read": 1.50},
    "claude-haiku-3-5":   {"in": 0.80,  "out": 4.00,  "cache_read": 0.08},
    "claude-haiku-3-5-20251022": {"in": 0.80, "out": 4.00, "cache_read": 0.08},
    # Catch-all pattern keys (matched by prefix)
    "claude-sonnet": {"in": 3.00,  "out": 15.00, "cache_read": 0.30},
    "claude-opus":   {"in": 15.00, "out": 75.00, "cache_read": 1.50},
    "claude-haiku":  {"in": 0.80,  "out": 4.00,  "cache_read": 0.08},
    # Google Gemini (vision — core-Kira "eyes", all modes)
    #   flash-lite = always-on heartbeat tier; flash-preview = salient-moment escalation.
    #   Approx public Gemini 3 flash-class rates ($/MTok); update when prices drift.
    "gemini-3.1-flash-lite":  {"in": 0.10, "out": 0.40, "cache_read": 0.025},
    "gemini-3-flash-preview": {"in": 0.30, "out": 2.50, "cache_read": 0.075},
    "gemini-3.5-flash":       {"in": 0.30, "out": 2.50, "cache_read": 0.075},
    # Catch-all prefix key for any other gemini vision model
    "gemini-3":  {"in": 0.30, "out": 2.50, "cache_read": 0.075},
    # Groq (llama-3.1-8b-instant)
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08, "cache_read": 0.0},
    # Local Llama — zero cost; tracked for token counts only
    "local":  {"in": 0.0, "out": 0.0, "cache_read": 0.0},
}


def _lookup_price(model: str) -> dict:
    """Find pricing for a model name. Tries exact match, then prefix match."""
    lm = model.lower()
    if lm in PRICE_TABLE:
        return PRICE_TABLE[lm]
    # Prefix match: longest matching prefix wins
    best = None
    best_len = 0
    for key, val in PRICE_TABLE.items():
        if lm.startswith(key) and len(key) > best_len:
            best = val
            best_len = len(key)
    return best or {"in": 0.0, "out": 0.0, "cache_read": 0.0}


def _calc_cost(model: str, input_tok: int, output_tok: int, cache_read_tok: int = 0) -> float:
    p = _lookup_price(model)
    M = 1_000_000
    return (
        input_tok     * p["in"]         / M +
        output_tok    * p["out"]        / M +
        cache_read_tok * p["cache_read"] / M
    )


# ── Session accumulator ───────────────────────────────────────────────────────

class _CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        # per-model counters: {model: {in, out, cache_read, calls, cost_usd}}
        self._by_model: dict[str, dict] = {}
        # per-purpose counters: {purpose: cost_usd}
        self._by_purpose: dict[str, float] = {}
        self._total_cost: float = 0.0
        self._fallback_count: int = 0
        self._stream_logger = None   # injected by bot._main_loop after logger is started

    def set_stream_logger(self, logger) -> None:
        self._stream_logger = logger

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
        cache_read_tokens: int = 0,
        estimated: bool = False,
    ) -> None:
        """Thread-safe call recording.  Fires-and-forgets the JSONL entry."""
        cost = _calc_cost(model, input_tokens, output_tokens, cache_read_tokens)
        with self._lock:
            m = self._by_model.setdefault(model, {
                "in": 0, "out": 0, "cache_read": 0, "calls": 0, "cost_usd": 0.0
            })
            m["in"]         += input_tokens
            m["out"]        += output_tokens
            m["cache_read"] += cache_read_tokens
            m["calls"]      += 1
            m["cost_usd"]   += cost
            p = self._by_purpose.setdefault(purpose, 0.0)
            self._by_purpose[purpose] = p + cost
            self._total_cost          += cost

        # JSONL event (best-effort; non-blocking)
        if self._stream_logger is not None:
            try:
                self._stream_logger.log(
                    "llm_usage",
                    model=model,
                    purpose=purpose,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cost_usd=round(cost, 6),
                    estimated=estimated,
                )
            except Exception:
                pass

    def record_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        """Log a model substitution as WARNING and increment the fallback counter."""
        with self._lock:
            self._fallback_count += 1
        msg = f"   [COST/FALLBACK] WARNING: {from_model} → {to_model} (reason: {reason})"
        print(msg)
        if self._stream_logger is not None:
            try:
                self._stream_logger.log(
                    "llm_fallback",
                    from_model=from_model,
                    to_model=to_model,
                    reason=reason,
                )
            except Exception:
                pass

    def session_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost

    def snapshot(self) -> dict:
        """Return a copy of all counters (thread-safe snapshot)."""
        with self._lock:
            return {
                "total_cost_usd":  round(self._total_cost, 4),
                "fallback_count":  self._fallback_count,
                "by_model":   {k: dict(v) for k, v in self._by_model.items()},
                "by_purpose": dict(self._by_purpose),
            }

    def write_receipt(self, receipts_dir: str = "logs/receipts",
                      session_meta: Optional[dict] = None) -> Optional[str]:
        """Phase J RECEIPT: persist the session's cost breakdown as a durable artifact —
        one JSON receipt per session + an append-only LEDGER.jsonl (one line per session,
        the cross-session spend history). Called at shutdown next to print_summary().
        Best-effort: never blocks teardown; returns the receipt path or None. LOUD on
        failure (constraint #3 — silent failure is the enemy)."""
        import os
        try:
            s = self.snapshot()
            s["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if session_meta:
                s["session"] = session_meta
            os.makedirs(receipts_dir, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
            path = os.path.join(receipts_dir, f"receipt_{stamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
            ledger_line = {
                "ts_utc": s["ts_utc"],
                "total_cost_usd": s["total_cost_usd"],
                "fallback_count": s["fallback_count"],
                "by_purpose": s["by_purpose"],
                "receipt": os.path.basename(path),
            }
            if session_meta:
                ledger_line["session"] = session_meta
            with open(os.path.join(receipts_dir, "LEDGER.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(ledger_line, ensure_ascii=False) + "\n")
            print(f"   [COST] receipt written: {path} (total ${s['total_cost_usd']:.4f})")
            return path
        except Exception as e:
            print(f"   [COST] !! receipt write FAILED: {e!r} (LOUD — cost data lives only in logs)")
            return None

    def print_summary(self) -> None:
        """Print a formatted cost breakdown.  Called at shutdown."""
        s = self.snapshot()
        total = s["total_cost_usd"]
        print()
        print("   ─── SESSION COST SUMMARY ──────────────────────────────────")
        print(f"   Total:  ${total:.4f}")
        if s["fallback_count"]:
            print(f"   Model fallbacks: {s['fallback_count']} (check logs for details)")
        print()
        print("   By purpose:")
        for purpose, cost in sorted(s["by_purpose"].items(), key=lambda x: -x[1]):
            print(f"     {purpose:<18}  ${cost:.4f}")
        print()
        print("   By model:")
        for model, d in sorted(s["by_model"].items(), key=lambda x: -x[1]["cost_usd"]):
            tok_in  = d["in"]
            tok_out = d["out"]
            tok_cr  = d["cache_read"]
            print(
                f"     {model:<40}  ${d['cost_usd']:.4f}"
                f"  ({tok_in}in / {tok_out}out"
                + (f" / {tok_cr}cr" if tok_cr else "")
                + f"  ×{d['calls']})"
            )
        print("   ──────────────────────────────────────────────────────────")
        print()


# Module-level singleton — import and call cost_tracker.record(...)
cost_tracker = _CostTracker()
