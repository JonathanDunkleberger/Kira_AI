# chess_agent.py — Chess Mode (Phase 1): "Kira plays Lichess, on stream"
# ─────────────────────────────────────────────────────────────────────────────
# A SEPARATE, isolated mode. Kira plays real chess games on Lichess against
# Stockfish (or incoming human/bot challenges), and reacts to the moments in
# character via a throttled callback — the exact mirror of MediaWatch.on_react.
#
# Stack (all open, no AGPL code vendored from lichess-bot — design inspiration
# only):
#   - berserk        : Lichess Bot API client (synchronous / requests-based).
#   - python-chess   : Board model + UCI engine driver.
#   - local Stockfish: the move engine, Elo-capped so Kira plays like a human
#                      club player (and blunders like one).
#
# What it does:
#   - Opens the Lichess event stream; accepts standard challenges (casual AND
#     rated), declines variants / ultra-fast time controls politely.
#   - Can challenge Lichess's own Stockfish levels (challenge_ai).
#   - On our turn, the Elo-capped engine picks a move within a movetime cap;
#     we submit it through a single SERIALIZED sender (one request at a time,
#     429 backoff) — Lichess's hard rule.
#   - Tracks eval after each ply (shallow analyse) for blunder detection
#     (eval swing > ~1.5 pawns = blunder event).
#   - Fires on_react for the MOMENTS: game start, our move (with a one-line
#     "why"), opponent move, blunder (ours or theirs), draw offer, game end.
#     Same react_min_gap_s throttle as MediaWatch EXCEPT game start, game end
#     and blunders bypass the throttle (those are the moments).
#
# What it does NOT do (Phase 1):
#   - No open seeks / matchmaking (that's Phase 2).
#   - No dynamic in-game chat — one fixed greeting at start, one "gg" at end.
#     Dynamic chat needs the same guardrail treatment as [CHAT] (Phase 2).
#   - CPU only. No GPU contention with the vision/STT stack.

import asyncio
import json
import os
import threading
import time

try:
    import chess
    import chess.engine
    CHESS_AVAILABLE = True
except Exception:
    CHESS_AVAILABLE = False

try:
    import berserk
    BERSERK_AVAILABLE = True
except Exception:
    BERSERK_AVAILABLE = False


# ─── Tunables / constants ────────────────────────────────────────────────────

# Eval swing (in centipawns, White POV magnitude) that counts as a blunder.
_BLUNDER_CP = 150          # ~1.5 pawns

# Movetime used for the shallow eval-tracking analyse pass (kept small — this is
# for blunder detection, not move selection).
_ANALYSE_MS = 120

# Standard material values for the plain-language balance line.
_PIECE_VALUES = {
    "p": 1, "n": 3, "b": 3, "r": 5, "q": 9,
}

# Speeds we accept. ultraBullet is too fast for a real movetime budget.
_ACCEPTED_SPEEDS = {"bullet", "blitz", "rapid", "classical", "correspondence"}

# make_move retry: backoff seconds for attempts 1-4 after a transient failure.
_MOVE_RETRY_BACKOFFS = (0.5, 1.0, 2.0, 4.0)

# Error substrings that identify a transient network / server-side failure.
_MOVE_TRANSIENT_KEYWORDS = (
    "RemoteDisconnected", "ConnectionError", "ConnectionReset",
    "ConnectionAbortedError", "TimeoutError", "timeout",
    "Read timed out", "503", "502", "504", "500",
)

# Fixed Phase-1 in-game chat lines (no dynamic chat yet).
_GREETING = "gl hf! heads up — I'm an AI VTuber and this game is being streamed live."
_FAREWELL = "gg, thanks for the game!"


def _fmt_clock(ms) -> str:
    """Format a Lichess clock value (milliseconds, or a datetime for
    correspondence) into M:SS. Returns '—' when not a usable number."""
    try:
        total = int(ms) // 1000
    except (TypeError, ValueError):
        return "\u2014"
    if total < 0:
        return "\u2014"
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


class ChessAgent:
    """Lichess bot driver for Kira. Mirrors the MediaWatch lifecycle/react shape.

    Lifecycle:
        start()  — validate, open the engine, spawn the event-stream consumer.
        stop()   — bounded: stop loops, send no more moves, quit the engine
                   process, let daemon stream threads die with the connection.

    Threading model:
        berserk's streams are blocking generators. Each runs on a daemon thread
        that pushes items into an asyncio.Queue via call_soon_threadsafe; the
        async side consumes. ALL outbound HTTP goes through _serialized_call
        (single lock + 429 backoff) to honour Lichess's one-request-at-a-time
        rule. The Stockfish SimpleEngine is synchronous and driven via
        asyncio.to_thread under its own lock.
    """

    def __init__(self, *, token: str, engine_path: str,
                 kira_elo: int = 1800, movetime_ms: int = 150,
                 react_min_gap_s: float = 45.0):
        self.token = token or ""
        self.engine_path = engine_path or ""
        self.kira_elo = int(kira_elo)
        self.movetime_ms = int(movetime_ms)

        # State.
        self.enabled: bool = False
        self.is_running: bool = False
        # Set when the engine process fails to launch — chip shows ENGINE FAILED,
        # challenges are auto-declined, event stream is NOT opened.
        self._engine_failed: bool = False
        # Master switch: Jonny must explicitly open challenges each session.
        # DEFAULT OFF — never persisted. A fresh boot always starts closed so
        # Kira never silently plays games Jonny hasn't sanctioned.
        self.accepting_challenges: bool = False

        # Lichess client + identity.
        self._client = None
        self._my_id: str = ""

        # Engine.
        self._engine = None
        self._engine_lock = asyncio.Lock()

        # Serialized HTTP sender — one outbound request at a time.
        self._http_lock = asyncio.Lock()

        # Tasks.
        self._event_task: asyncio.Task | None = None
        self._game_task: asyncio.Task | None = None

        # Current game snapshot (read by get_board_block from the same loop).
        self._game_id: str = ""
        self._board = None                  # chess.Board | None
        self._kira_color = None             # chess.WHITE / chess.BLACK / None
        self._last_move_san: str = ""
        self._eval_plain: str = "even"
        self._last_eval_cp: int | None = None   # White POV centipawns
        self._wtime = None
        self._btime = None
        self._opp_name: str = ""
        self._opp_rating = None
        self._move_count_seen: int = 0
        self._greeted: bool = False
        # Game IDs we are resuming after a restart — suppresses re-greeting
        # and swaps the voice-line from "game started" to "resumed".
        self._resuming_games: set = set()

        # Session counters (dashboard / logging).
        self._session_start_ts: float = 0.0
        self._games_played: int = 0
        self._our_blunders: int = 0
        self._their_blunders: int = 0
        # Per-session W/L/D (resets on restart; opponent id tracking for viewer games).
        self._session_wins:   int = 0
        self._session_losses: int = 0
        self._session_draws:  int = 0
        self._opp_id: str = ""           # lowercase Lichess ID of current opponent
        self._is_viewer_game: bool = False  # True when opp is not CHESS_OWNER_LICHESS_ID
        # Lifetime W/L/D loaded from (and saved back to) data/chess_lifetime.json.
        self._lifetime_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "chess_lifetime.json"
        )
        self._lifetime_wins:   int = 0
        self._lifetime_losses: int = 0
        self._lifetime_draws:  int = 0
        self._load_lifetime()

        # Last-game result lingers on the chip until next game.
        self._last_result_str: str = ""   # e.g. "won vs MagnusFan1942"

        # Move-send state — guards against dropped HTTP calls bleeding the clock.
        # _move_pending_uci is set when we submit a move and cleared when the
        # game-state stream echoes it back, confirming it actually landed.
        self._move_pending_uci: str = ""  # UCI of the sent-but-unconfirmed move
        self._conn_lost: bool = False     # True when all send retries exhausted

        # Reaction callback (set by bot wiring). Signature: on_react(summary, *, bypass).
        self.on_react = None
        # Banner callback (set by bot wiring). Signature: on_banner(text, duration_s).
        # Called at game start/end with a human-readable announcement string.
        self.on_banner = None
        # Score update callback. Fires after every game end.
        # Signature: on_score_update(session_wins, session_losses, session_draws,
        #                            lifetime_wins, lifetime_losses, lifetime_draws)
        self.on_score_update = None
        # Spectate callbacks — fired for viewer games only (opp != CHESS_OWNER_LICHESS_ID).
        # on_spectate_show(url, opponent), on_spectate_hide()
        self.on_spectate_show = None
        self.on_spectate_hide = None
        self.react_min_gap_s: float = float(react_min_gap_s)
        self._last_react_ts: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _load_lifetime(self):
        """Read lifetime W/L/D from data/chess_lifetime.json. Silent on first run."""
        try:
            if os.path.exists(self._lifetime_path):
                with open(self._lifetime_path, "r", encoding="utf-8") as _f:
                    _d = json.load(_f)
                self._lifetime_wins   = int(_d.get("wins",   0))
                self._lifetime_losses = int(_d.get("losses", 0))
                self._lifetime_draws  = int(_d.get("draws",  0))
                print(f"   [Chess] Lifetime stats loaded: "
                      f"W{self._lifetime_wins}-L{self._lifetime_losses}-D{self._lifetime_draws}")
        except Exception as _e:
            print(f"   [Chess] Could not load lifetime stats: {_e}")

    def _save_lifetime(self):
        """Atomic write of lifetime W/L/D to data/chess_lifetime.json."""
        try:
            os.makedirs(os.path.dirname(self._lifetime_path), exist_ok=True)
            _tmp = self._lifetime_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                json.dump({
                    "wins":   self._lifetime_wins,
                    "losses": self._lifetime_losses,
                    "draws":  self._lifetime_draws,
                }, _f)
            os.replace(_tmp, self._lifetime_path)
        except Exception as _e:
            print(f"   [Chess] Could not save lifetime stats: {_e}")

    def get_score_data(self) -> dict:
        """Snapshot of session + lifetime W/L/D for score overlay and state_snapshot."""
        return {
            "session_wins":    self._session_wins,
            "session_losses":  self._session_losses,
            "session_draws":   self._session_draws,
            "lifetime_wins":   self._lifetime_wins,
            "lifetime_losses": self._lifetime_losses,
            "lifetime_draws":  self._lifetime_draws,
        }

    def start(self):
        """Validate prerequisites, open the engine, and spawn the event loop.
        Safe to call multiple times. No-ops (with a logged reason) when it can't
        arm — never raises."""
        if not self.enabled:
            return
        if self.is_running:
            return
        if not CHESS_AVAILABLE:
            print("   [Chess] python-chess not installed — cannot start. (pip install chess)")
            return
        if not BERSERK_AVAILABLE:
            print("   [Chess] berserk not installed — cannot start. (pip install berserk)")
            return
        if not self.token.strip():
            print("   [Chess] No LICHESS_BOT_TOKEN set — refusing to start.")
            return
        if not self.engine_path.strip():
            print("   [Chess] No CHESS_ENGINE_PATH set — refusing to start.")
            return

        # Open the Lichess session + verify identity.
        try:
            session = berserk.TokenSession(self.token)
            self._client = berserk.Client(session=session)
            account = self._client.account.get()
            self._my_id = (account.get("id") or account.get("username") or "").lower()
            if not self._my_id:
                print("   [Chess] Could not resolve Lichess account id — refusing to start.")
                return
            print(f"   [Chess] Connected to Lichess as '{self._my_id}'.")
        except Exception as e:
            print(f"   [Chess] Lichess connection failed: {e}")
            return

        # Open the engine (Elo-capped, single thread, CPU).
        # Clamp Elo to Stockfish's legal UCI_Elo range before configuring.
        _SF_ELO_MIN, _SF_ELO_MAX = 1320, 3190
        if not (_SF_ELO_MIN <= self.kira_elo <= _SF_ELO_MAX):
            _clamped = max(_SF_ELO_MIN, min(_SF_ELO_MAX, self.kira_elo))
            print(
                f"   [Chess] WARNING: CHESS_KIRA_ELO={self.kira_elo} is outside "
                f"Stockfish's legal UCI_Elo range ({_SF_ELO_MIN}–{_SF_ELO_MAX}). "
                f"Clamping to {_clamped}."
            )
            self.kira_elo = _clamped
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            self._engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": self.kira_elo,
                "Threads": 1,
            })
            print(
                f"   [Chess] Engine ready (Elo cap {self.kira_elo}, "
                f"movetime {self.movetime_ms}ms, 1 thread, CPU)."
            )
        except Exception as e:
            print(f"   [Chess] ENGINE LAUNCH FAILED ('{self.engine_path}'): {e}")
            try:
                if self._engine:
                    self._engine.quit()
            except Exception:
                pass
            self._engine = None
            self._engine_failed = True
            self.is_running = True   # so chip + /state reflect the failure
            self._session_start_ts = time.time()
            # Fire a voice-line so Kira acknowledges the failure on stream.
            asyncio.ensure_future(self._fire_react(
                "My chess brain didn’t boot, one second.",
                bypass=True,
            ))
            return

        self.is_running = True
        self._session_start_ts = time.time()
        self._event_task = asyncio.ensure_future(self._event_loop())
        # Rejoin any game that was in-progress when we last restarted.
        asyncio.ensure_future(self._resume_ongoing_games())
        print("   [Chess] Started — listening for challenges and games.")

    def stop(self):
        """Bounded shutdown: stop loops, quit the engine process, let the daemon
        stream threads die with the closed connection. Never raises."""
        was_running = self.is_running
        self.is_running = False
        self.enabled = False
        self._engine_failed = False   # reset so a re-arm after fix can retry
        self._move_pending_uci = ""
        self._conn_lost = False
        for t in (self._event_task, self._game_task):
            if t and not t.done():
                t.cancel()
        self._event_task = None
        self._game_task = None
        # Quit the engine process off the loop so a hung quit can't trap us.
        eng = self._engine
        self._engine = None
        if eng is not None:
            def _quit():
                try:
                    eng.quit()
                except Exception:
                    try:
                        eng.close()
                    except Exception:
                        pass
            threading.Thread(target=_quit, daemon=True).start()
        if was_running:
            runtime = int(time.time() - self._session_start_ts)
            m, s = divmod(runtime, 60)
            print(
                f"   [Chess] Stopped. {self._games_played} game(s), "
                f"blunders {self._our_blunders} ours / {self._their_blunders} theirs, "
                f"ran {m}m {s}s."
            )

    # ── Status (dashboard reads this) ─────────────────────────────────────────

    def get_status_str(self) -> str:
        """Chip text for the dashboard status strip.

        ♟ ENGINE FAILED      — engine process failed to start
        ♟ CLOSED            — running but not accepting challenges
        ♟ OPEN              — accepting, no active game
        ♟ vs Name (R) · M  — in game (spectate URL in get_spectate_url)
        ♟ last: result      — post-game, until next game starts
        """
        if not self.is_running:
            return "\u265f CLOSED"
        if self._engine_failed:
            return "\u265f ENGINE FAILED"
        if self._conn_lost:
            opp = self._opp_name or "?"
            return f"\u265f CONNECTION LOST \u2014 game vs {opp}"
        if self._game_id and self._board is not None:
            opp = self._opp_name or "?"
            if self._opp_rating:
                opp = f"{opp} ({self._opp_rating})"
            move_n = (self._board.fullmove_number
                      if self._board else 0)
            return (
                f"\u265f vs {opp} \u00b7 move {move_n} \u00b7 {self._eval_plain}"
            )
        if self._last_result_str and not self.accepting_challenges:
            return f"\u265f last: {self._last_result_str}"
        if self.accepting_challenges:
            if self._last_result_str:
                return f"\u265f OPEN \u2014 last: {self._last_result_str}"
            return "\u265f OPEN \u2014 awaiting challengers"
        return "\u265f CLOSED"

    def get_spectate_url(self) -> str:
        """Lichess spectate URL for the current game, or empty string."""
        if self._game_id:
            return f"https://lichess.org/{self._game_id}"
        return ""

    # ── Board block (the prompt-facing context; NO engine-speak) ──────────────

    def get_board_block(self) -> str:
        """Compact, natural-language board state for prompt injection. NEVER
        emits raw centipawn numbers — eval is translated to plain terms so
        Kira's commentary can't turn into engine-speak."""
        if self._board is None or self._kira_color is None:
            return ""
        b = self._board
        side_to_move = "Kira" if (b.turn == self._kira_color) else "the opponent"
        kira_side = "White" if self._kira_color == chess.WHITE else "Black"
        last = self._last_move_san or "(none yet)"
        material = self._material_balance_phrase(b)
        # Clocks from Kira's POV.
        if self._kira_color == chess.WHITE:
            kira_clock, opp_clock = _fmt_clock(self._wtime), _fmt_clock(self._btime)
        else:
            kira_clock, opp_clock = _fmt_clock(self._btime), _fmt_clock(self._wtime)
        opp = self._opp_name or "Opponent"
        if self._opp_rating:
            opp = f"{opp} ({self._opp_rating})"
        lines = [
            "[CHESS BOARD STATE \u2014 this is YOUR live game; talk about it like a "
            "confident club player, never in engine terms]",
            f"  You're playing {kira_side} vs {opp}.",
            f"  To move: {side_to_move}.",
            f"  Last move: {last}.",
            f"  Material: {material}.",
            f"  Position: {self._eval_plain}.",
            f"  Clocks: Kira {kira_clock} / {opp} {opp_clock}.",
        ]
        return "\n".join(lines)

    def has_context(self) -> bool:
        """True only while a game is active — board block must NOT be injected
        between games. Callers gate on this; the block is a few hundred chars
        and counts against the scene budget like any other context block."""
        return self.is_running and self._board is not None

    # ── Challenge Lichess's Stockfish ─────────────────────────────────────────

    async def challenge_ai(self, level: int = 3, clock_limit_s: int = 300,
                           clock_increment_s: int = 3) -> bool:
        """Challenge Lichess's own Stockfish at `level` (1-8). The resulting
        game arrives as a gameStart event on the main stream. Returns True if
        the challenge request was accepted by the API."""
        if not self.is_running or self._client is None:
            print("   [Chess] challenge_ai: not running.")
            return False
        level = max(1, min(8, int(level)))
        try:
            await self._serialized_call(
                self._client.challenges.create_ai,
                level=level,
                clock_limit=clock_limit_s,
                clock_increment=clock_increment_s,
                color="random",
                variant="standard",
                label=f"challenge_ai(level={level})",
            )
            print(f"   [Chess] Challenged Lichess Stockfish level {level}.")
            return True
        except Exception as e:
            print(f"   [Chess] challenge_ai failed: {e}")
            return False

    # ── Serialized HTTP sender (one request at a time, 429 backoff) ───────────

    async def _serialized_call(self, fn, *args, label: str = "", **kwargs):
        """Run a blocking berserk call under the single HTTP lock, off the event
        loop, with exponential backoff on 429 (rate limit). Honours Lichess's
        one-request-at-a-time rule. Re-raises non-rate-limit errors."""
        async with self._http_lock:
            for attempt in range(4):
                try:
                    return await asyncio.to_thread(lambda: fn(*args, **kwargs))
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    is_429 = (status == 429) or ("429" in str(e)) or ("rate" in str(e).lower())
                    if is_429 and attempt < 3:
                        delay = 10.0 * (2 ** attempt)   # Lichess asks for ~60s; ramp into it
                        print(
                            f"   [Chess] 429 rate-limited on {label or fn.__name__} "
                            f"(attempt {attempt+1}/4) — backing off {delay:.0f}s."
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

    # ── Main event stream ─────────────────────────────────────────────────────

    async def _event_loop(self):
        """Consume the Lichess incoming-event stream and dispatch challenges /
        game starts. The blocking generator runs on a daemon thread feeding an
        asyncio.Queue."""
        try:
            async for event in self._bridge_stream(
                lambda: self._client.bots.stream_incoming_events(),
                "events",
            ):
                etype = event.get("type")
                if etype == "challenge":
                    await self._handle_challenge(event.get("challenge", {}))
                elif etype == "gameStart":
                    game = event.get("game", {})
                    gid = game.get("gameId") or game.get("id") or game.get("fullId", "")[:8]
                    if gid:
                        await self._handle_game_start(gid)
                elif etype == "gameFinish":
                    pass  # game loop handles end-of-game itself
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"   [Chess] Event loop error: {e}")

    async def _bridge_stream(self, iterator_factory, label: str):
        """Bridge a blocking berserk generator to an async iterator via a daemon
        thread + asyncio.Queue. Stops when is_running flips False or the stream
        ends."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _worker():
            try:
                for item in iterator_factory():
                    if not self.is_running:
                        break
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("__stream_error__", str(e)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, sentinel)

        threading.Thread(target=_worker, name=f"chess-{label}", daemon=True).start()

        while self.is_running:
            item = await q.get()
            if item is sentinel:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__stream_error__":
                print(f"   [Chess] {label} stream error: {item[1]}")
                break
            yield item

    # ── Challenge handling ────────────────────────────────────────────────────

    async def _handle_challenge(self, ch: dict):
        """Phase 1 policy: accept standard chess (casual AND rated) at a sane
        speed; decline variants and ultra-fast controls politely."""
        cid = ch.get("id", "")
        if not cid:
            return
        challenger = (ch.get("challenger") or {}).get("name", "?")
        variant = (ch.get("variant") or {}).get("key", "standard")
        speed = ch.get("speed", "")
        rated = ch.get("rated", False)

        # Master switch: default OFF each boot. While closed, decline politely
        # and log the demand so Jonny can see it even when parked.
        if self._engine_failed:
            print(f"   [Chess] Challenge from {challenger} DECLINED (engine failed) "
                  f"— {speed} {'rated' if rated else 'casual'} {variant}.")
            await self._decline(cid, "generic")
            await self._send_chat_room(
                cid,
                "Chess engine failed to start — I can't play right now, sorry!"
            )
            return

        if not self.accepting_challenges:
            print(f"   [Chess] Challenge from {challenger} DECLINED (closed) — "
                  f"{speed} {'rated' if rated else 'casual'} {variant}.")
            await self._decline(cid, "generic")  # berserk sends generic message
            await self._send_chat_room(
                cid,
                "The Duchess is not taking challengers right now — catch the stream!"
            )
            return

        if variant != "standard":
            print(f"   [Chess] Declining {challenger}: variant '{variant}' (Phase 1 = standard only).")
            await self._decline(cid, "variant")
            return
        if speed and speed not in _ACCEPTED_SPEEDS:
            print(f"   [Chess] Declining {challenger}: speed '{speed}' (too fast for movetime).")
            await self._decline(cid, "timeControl")
            return

        # One concurrent game cap.
        if self._game_task and not self._game_task.done():
            print(f"   [Chess] Declining {challenger}: already in a game (one at a time).")
            await self._decline(cid, "later")
            await self._send_chat_room(
                cid,
                "Already in a game right now — one at a time! Challenge again when I'm free."
            )
            return

        print(f"   [Chess] Accepting {'rated' if rated else 'casual'} {speed} "
              f"challenge from {challenger} — game starting!")
        try:
            await self._serialized_call(
                self._client.bots.accept_challenge, cid,
                label="accept_challenge",
            )
        except Exception as e:
            print(f"   [Chess] Accept failed for {cid}: {e}")

    async def _decline(self, cid: str, reason: str):
        try:
            await self._serialized_call(
                self._client.bots.decline_challenge, cid, reason=reason,
                label="decline_challenge",
            )
        except Exception as e:
            print(f"   [Chess] Decline failed for {cid}: {e}")

    # ── Game handling ──────────────────────────────────────────────────────────

    async def _resume_ongoing_games(self):
        """On boot, query Lichess for any games this account is currently playing
        and rejoin them. Rebuilds the board from the full move list and continues
        playing. A bot restart mid-game becomes a hiccup, not a forfeit.

        Phase 1 cap: only resumes the first standard game found; additional
        in-progress games are logged and skipped (one active game at a time)."""
        # Short delay so the event stream is up before we open a game stream.
        await asyncio.sleep(2.0)
        if not self.is_running or self._client is None:
            return
        try:
            ongoing = await self._serialized_call(
                self._client.games.get_ongoing,
                count=5,
                label="get_ongoing",
            )
        except Exception as e:
            print(f"   [Chess] Resume check failed — could not fetch ongoing games: {e}")
            return

        if not ongoing:
            return  # Nothing to resume.

        resumed = 0
        for game in (ongoing or []):
            gid = game.get("gameId") or (game.get("fullId") or "")[:8]
            if not gid:
                continue
            variant = (game.get("variant") or {}).get("key", "standard")
            if variant != "standard":
                print(f"   [Chess] Skipping non-standard ongoing game {gid} ({variant}).")
                continue
            if resumed > 0:
                # Phase 1: one active game at a time.
                print(f"   [Chess] Additional in-progress game {gid} skipped (one-game Phase 1 cap).")
                continue
            opp_info = game.get("opponent") or {}
            opp_name = (
                f"Stockfish L{opp_info['aiLevel']}" if opp_info.get("aiLevel") is not None
                else opp_info.get("username") or "?"
            )
            print(f"   [Chess] Found in-progress game {gid} vs {opp_name} — scheduling resume.")
            self._resuming_games.add(gid)
            await self._handle_game_start(gid)
            resumed += 1

    async def _handle_game_start(self, game_id: str):
        """Start a game-state consumer for game_id. Phase 1 plays one game at a
        time — if one is already live, log and ignore the new start."""
        if self._game_task and not self._game_task.done():
            if game_id == self._game_id:
                # Stream duplicate for a game we already resumed — expected.
                print(f"   [Chess] Stream duplicate for active game {game_id} "
                      f"(normal after resume) — already running.")
            else:
                # A genuinely new game while one is in progress (Phase 1 cap).
                print(f"   [Chess] Already in game {self._game_id}; ignoring "
                      f"new gameStart {game_id} (Phase 1 one-game cap).")
            return
        self._game_id = game_id
        # For resumed games skip the greeting (already sent it before restart).
        self._greeted = game_id in self._resuming_games
        self._move_count_seen = 0
        self._last_eval_cp = None
        self._move_pending_uci = ""
        self._conn_lost = False
        self._game_task = asyncio.ensure_future(self._play_game(game_id))

    async def _play_game(self, game_id: str):
        """Consume the game NDJSON stream, make our moves, track eval, and fire
        reactions on the moments. One serialized sender for all HTTP."""
        print(f"   [Chess] Game {game_id} started — opening state stream.")
        try:
            async for upd in self._bridge_stream(
                lambda: self._client.bots.stream_game_state(game_id),
                f"game-{game_id}",
            ):
                utype = upd.get("type")
                if utype == "gameFull":
                    self._ingest_game_full(upd)
                    _opp = self._opp_name or "someone"
                    _rating_str = f" ({self._opp_rating})" if self._opp_rating else ""
                    _state_tmp = upd.get("state", {})
                    _move_n = len((_state_tmp.get("moves") or "").split())
                    if game_id in self._resuming_games:
                        self._resuming_games.discard(game_id)
                        print(
                            f"   [Chess] Resumed in-progress game vs "
                            f"{_opp}{_rating_str} (move {_move_n})."
                        )
                        await self._fire_react(
                            f"Bot restarted mid-game — resuming against "
                            f"{_opp}{_rating_str} at move {_move_n}. "
                            f"Picking up right where we left off.",
                            bypass=True,
                        )
                    else:
                        _url = self.get_spectate_url()
                        print(
                            f"   [Chess] GAME STARTED vs {_opp}{_rating_str} — "
                            f"spectate: {_url}"
                        )
                        await self._fire_react(
                            f"New game just started against {_opp}{_rating_str}. "
                            f"Challenger accepted. Let\u2019s go.",
                            bypass=True,
                        )
                        # Banner announcement for overlays
                        if self.on_banner:
                            try:
                                opp_label = f"{_opp}{_rating_str}" if _rating_str else _opp
                                await self.on_banner(f"\u265f DUCHESS vs {opp_label}", 10)
                            except Exception:
                                pass
                        # Spectate embed — only for viewer games (not Jonny's account)
                        try:
                            from kira.config import CHESS_OWNER_LICHESS_ID as _owner_id
                            self._is_viewer_game = (
                                bool(self._opp_id)
                                and self._opp_id != _owner_id.lower()
                            )
                        except Exception:
                            self._is_viewer_game = False
                        if self._is_viewer_game and self.on_spectate_show:
                            try:
                                await self.on_spectate_show(
                                    self.get_spectate_url(), self._opp_name
                                )
                            except Exception:
                                pass
                    state = upd.get("state", {})
                elif utype == "gameState":
                    state = upd
                elif utype in ("chatLine", "opponentGone"):
                    continue
                else:
                    continue

                if not self._greeted:
                    self._greeted = True
                    await self._send_chat(game_id, _GREETING)

                ended = await self._process_state(game_id, state)
                if ended:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"   [Chess] Game {game_id} loop error: {e}")
        finally:
            self._game_id = ""
            self._board = None
            self._move_pending_uci = ""
            self._conn_lost = False

    def _ingest_game_full(self, full: dict):
        """Pull static game info from the gameFull frame: our color, opponent,
        initial position."""
        white = full.get("white", {}) or {}
        black = full.get("black", {}) or {}
        white_id = (white.get("id") or "").lower()

        if white_id and white_id == self._my_id:
            self._kira_color = chess.WHITE
            opp = black
        elif (black.get("id") or "").lower() == self._my_id:
            self._kira_color = chess.BLACK
            opp = white
        else:
            # AI opponent (no id) — infer our color from which side lacks our id.
            if white.get("aiLevel") is not None:
                self._kira_color = chess.BLACK
                opp = white
            else:
                self._kira_color = chess.WHITE
                opp = black

        if opp.get("aiLevel") is not None:
            self._opp_name = f"Stockfish L{opp.get('aiLevel')}"
            self._opp_rating = None
            self._opp_id = ""   # AI — never a viewer game
        else:
            self._opp_name = opp.get("name") or opp.get("id") or "Opponent"
            self._opp_rating = opp.get("rating")
            self._opp_id = (opp.get("id") or "").lower()

        self._initial_fen = full.get("initialFen", "startpos")
        self._games_played += 1
        kira_side = "White" if self._kira_color == chess.WHITE else "Black"
        print(f"   [Chess] Playing {kira_side} vs {self._opp_name}.")

    def _build_board(self, moves_str: str):
        """Rebuild the board from the initial position + the move list. Cheap and
        desync-proof for single-game Phase 1."""
        if getattr(self, "_initial_fen", "startpos") in ("startpos", "", None):
            board = chess.Board()
        else:
            try:
                board = chess.Board(self._initial_fen)
            except Exception:
                board = chess.Board()
        last_san = ""
        for uci in (moves_str or "").split():
            try:
                mv = chess.Move.from_uci(uci)
                last_san = board.san(mv)
                board.push(mv)
            except Exception:
                break
        return board, last_san

    async def _process_state(self, game_id: str, state: dict) -> bool:
        """Handle one gameState. Returns True when the game has ended."""
        status = state.get("status", "started")
        self._wtime = state.get("wtime")
        self._btime = state.get("btime")
        moves_str = state.get("moves", "")

        # Stream-confirmation guard: if we have a pending move, check whether
        # the stream has now echoed it. Check the last two entries so we catch
        # the case where the opponent replied before this state was processed.
        if self._move_pending_uci:
            uci_list = moves_str.split() if moves_str else []
            if self._move_pending_uci in uci_list[-2:]:
                self._move_pending_uci = ""
                if self._conn_lost:
                    self._conn_lost = False
                    print("   [Chess] Connection restored — move confirmed by stream.")

        board, last_san = self._build_board(moves_str)
        self._board = board
        self._last_move_san = last_san
        move_count = len(moves_str.split()) if moves_str else 0

        # Draw offer from the opponent (their flag is set, not ours).
        opp_offer_key = "boffer" if self._kira_color == chess.WHITE else "woffer"
        if state.get(opp_offer_key):
            await self._fire_react(
                f"Your opponent {self._opp_name} just offered a draw. "
                f"Position: {self._eval_plain}.",
                bypass=True,
            )

        # Eval tracking + blunder detection on each NEW ply.
        if move_count > self._move_count_seen and move_count > 0:
            await self._track_eval_and_blunder(board, last_san)
            self._move_count_seen = move_count

        # Game end?
        if status != "started":
            await self._on_game_end(game_id, status, state.get("winner"))
            return True

        # Our move?
        if board.turn == self._kira_color:
            await self._make_our_move(game_id, board)

        return False

    async def _send_move_with_retry(self, game_id: str, uci: str) -> bool:
        """Send a move over HTTP with transient-error retry.
        Returns True if any attempt succeeds; False when all are exhausted."""
        _client_refreshed = False
        for attempt, backoff in enumerate(_MOVE_RETRY_BACKOFFS):
            is_last = (attempt == len(_MOVE_RETRY_BACKOFFS) - 1)
            try:
                await self._serialized_call(
                    self._client.bots.make_move, game_id, uci,
                    label=f"make_move({uci}) #{attempt + 1}",
                )
                return True
            except Exception as e:
                err_str = f"{type(e).__name__}: {e}"
                is_transient = (
                    any(kw in err_str for kw in _MOVE_TRANSIENT_KEYWORDS)
                    or isinstance(e, (ConnectionError, TimeoutError, OSError))
                )
                if is_transient and not is_last:
                    print(
                        f"   [Chess] make_move transient error "
                        f"(attempt {attempt + 1}/{len(_MOVE_RETRY_BACKOFFS)},"
                        f" retry in {backoff:.1f}s): {err_str}"
                    )
                    # Recreate the berserk HTTP session on first disconnect.
                    if "RemoteDisconnected" in err_str and not _client_refreshed:
                        try:
                            session = berserk.TokenSession(self.token)
                            self._client = berserk.Client(session=session)
                            _client_refreshed = True
                            print("   [Chess] Berserk client recreated after RemoteDisconnected.")
                        except Exception as ce:
                            print(f"   [Chess] Client recreate failed: {ce}")
                    await asyncio.sleep(backoff)
                else:
                    print(
                        f"   [Chess] make_move failed "
                        f"(attempt {attempt + 1}/{len(_MOVE_RETRY_BACKOFFS)}, "
                        f"{'all exhausted' if is_last else 'non-transient'}): {err_str}"
                    )
                    return False
        return False  # unreachable, satisfies type checker

    async def _retry_conn_lost_move(self, game_id: str, uci: str,
                                    san: str, why: str):
        """Background: keep trying to resend a dropped move every 5 s until
        the game-state stream confirms it landed, or the game ends.
        Recreates the berserk client before each attempt."""
        print(f"   [Chess] Conn-lost background retry started for {uci} ({san}).")
        while self.is_running and self._game_id == game_id:
            await asyncio.sleep(5.0)
            if not self.is_running or self._game_id != game_id:
                print(f"   [Chess] Conn-lost retry for {uci} aborted "
                      f"(game ended or agent stopped).")
                break
            if not self._move_pending_uci:
                # Stream confirmed the move while we slept.
                print(f"   [Chess] Conn-lost retry: {uci} already confirmed by stream.")
                self._conn_lost = False
                break
            try:
                session = berserk.TokenSession(self.token)
                self._client = berserk.Client(session=session)
            except Exception as e:
                print(f"   [Chess] Conn-lost retry: client recreate failed: {e}")
                continue
            try:
                await self._serialized_call(
                    self._client.bots.make_move, game_id, uci,
                    label=f"conn_lost_retry({uci})",
                )
                print(
                    f"   [Chess] CONNECTION RESTORED \u2014 move {uci} ({san}) resent. "
                    f"Waiting for stream confirmation."
                )
                # Leave _move_pending_uci / _conn_lost for the stream to clear.
                await self._fire_react(
                    f"Back in it \u2014 {san} is on the board. {self._eval_plain}.",
                    bypass=True,
                )
                break  # stream will echo and _process_state will clear state
            except Exception as e:
                print(
                    f"   [Chess] Conn-lost retry attempt failed for {uci}: "
                    f"{type(e).__name__}: {e}"
                )

    async def _make_our_move(self, game_id: str, board):
        """Pick a move with the Elo-capped engine and submit it via the
        serialized sender. Fires a throttled react with a one-line 'why'."""
        if self._engine is None:
            return
        try:
            async with self._engine_lock:
                result = await asyncio.to_thread(
                    self._engine.play, board,
                    chess.engine.Limit(time=self.movetime_ms / 1000.0),
                )
            move = result.move
            if move is None:
                return
            san = board.san(move)
            why = self._move_why(board, move)
        except Exception as e:
            print(f"   [Chess] Engine move failed: {e}")
            return

        uci = move.uci()
        self._move_pending_uci = uci   # held until stream echoes it back
        ok = await self._send_move_with_retry(game_id, uci)
        if not ok:
            opp = self._opp_name or "opponent"
            self._conn_lost = True
            print(
                f"   [Chess] !! CONNECTION LOST !! All {len(_MOVE_RETRY_BACKOFFS)} "
                f"move-send attempts failed for {uci} ({san}) vs {opp}. "
                f"Clock is running. Starting background retry."
            )
            await self._fire_react(
                "my connection to the board just dropped",
                bypass=True,
            )
            asyncio.ensure_future(
                self._retry_conn_lost_move(game_id, uci, san, why)
            )
            return

        print(f"   [Chess] Kira played {san} ({why}). {self._eval_plain}.")
        await self._fire_react(
            f"You (Kira) just played {san} \u2014 {why}. Position: {self._eval_plain}.",
            bypass=False,
        )

    @staticmethod
    def _move_why(board, move) -> str:
        """One-word-ish reason derived from the move type (no engine-speak)."""
        try:
            if board.is_castling(move):
                return "castling to safety"
            if board.is_capture(move):
                return "a capture"
            if board.gives_check(move):
                return "a check"
            if move.promotion:
                return "a promotion"
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                return "a pawn push"
            return "developing the position"
        except Exception:
            return "a move"

    async def _track_eval_and_blunder(self, board, last_san: str):
        """Shallow analyse → update plain eval, detect blunders by eval swing.
        The side that just moved is the one opposite to the side now to move."""
        if self._engine is None:
            return
        try:
            async with self._engine_lock:
                info = await asyncio.to_thread(
                    self._engine.analyse, board,
                    chess.engine.Limit(time=_ANALYSE_MS / 1000.0),
                )
        except Exception as e:
            print(f"   [Chess] analyse failed: {e}")
            return

        score = info.get("score")
        if score is None:
            return
        # White-POV centipawns (mate mapped to a large value).
        cp_white = score.white().score(mate_score=10000)
        if cp_white is None:
            return

        self._eval_plain = self._eval_to_plain(cp_white)

        if self._last_eval_cp is not None:
            swing = cp_white - self._last_eval_cp           # White POV
            mover_white = (board.turn == chess.BLACK)        # side that just moved
            mover_blundered = (
                (mover_white and swing <= -_BLUNDER_CP) or
                (not mover_white and swing >= _BLUNDER_CP)
            )
            if mover_blundered:
                mover_color = chess.WHITE if mover_white else chess.BLACK
                is_kira = (mover_color == self._kira_color)
                if is_kira:
                    self._our_blunders += 1
                    who = "You (Kira)"
                else:
                    self._their_blunders += 1
                    who = self._opp_name or "Your opponent"
                print(
                    f"   [Chess] BLUNDER by {'Kira' if is_kira else 'opponent'} "
                    f"after {last_san}. Now: {self._eval_plain}."
                )
                await self._fire_react(
                    f"{who} just blundered with {last_san}. Position swung \u2014 "
                    f"now {self._eval_plain}.",
                    bypass=True,
                )
        self._last_eval_cp = cp_white

    def _eval_to_plain(self, cp_white: int) -> str:
        """Translate White-POV centipawns into a Kira-POV plain-language verdict.
        NEVER returns numbers."""
        if self._kira_color == chess.BLACK:
            cp = -cp_white
        else:
            cp = cp_white
        if cp >= 9000:
            return "Kira has a forced mate"
        if cp <= -9000:
            return "Kira is getting mated"
        if cp >= 350:
            return "Kira is winning"
        if cp >= 150:
            return "Kira is clearly better"
        if cp >= 60:
            return "Kira is slightly better"
        if cp > -60:
            return "roughly equal"
        if cp > -150:
            return "Kira is slightly worse"
        if cp > -350:
            return "Kira is clearly worse"
        return "Kira is losing badly"

    def _material_balance_phrase(self, board) -> str:
        """Plain material balance from Kira's POV (normal club-player talk; this
        is piece counting, not engine eval)."""
        if self._kira_color is None:
            return "even"
        white_pts = 0
        black_pts = 0
        for piece in board.piece_map().values():
            val = _PIECE_VALUES.get(piece.symbol().lower(), 0)
            if piece.color == chess.WHITE:
                white_pts += val
            else:
                black_pts += val
        net = (white_pts - black_pts)
        if self._kira_color == chess.BLACK:
            net = -net
        if net == 0:
            return "even"
        word = "up" if net > 0 else "down"
        n = abs(net)
        if n == 1:
            return f"Kira is {word} a pawn"
        if n == 2:
            return f"Kira is {word} two pawns"
        if n == 3:
            return f"Kira is {word} a piece"
        if n == 5:
            return f"Kira is {word} a rook"
        if n == 9:
            return f"Kira is {word} a queen"
        return f"Kira is {word} roughly {n} points of material"

    async def _on_game_end(self, game_id: str, status: str, winner: str | None):
        """Send gg and fire the (throttle-bypassing) game-end react."""
        if self._kira_color == chess.WHITE:
            kira_won = (winner == "white")
        elif self._kira_color == chess.BLACK:
            kira_won = (winner == "black")
        else:
            kira_won = False

        if winner:
            outcome = "Kira WON" if kira_won else "Kira LOST"
            self._last_result_str = (
                f"won vs {self._opp_name}" if kira_won
                else f"lost to {self._opp_name}"
            )
        else:
            outcome = f"draw ({status})"
            self._last_result_str = f"draw vs {self._opp_name}"

        print(
            f"   [Chess] GAME OVER — {outcome} vs {self._opp_name} "
            f"(status={status}). Final: {self._eval_plain}."
        )

        await self._send_chat(game_id, _FAREWELL)
        await self._fire_react(
            f"The game just ended \u2014 {outcome} ({status}). Final position: "
            f"{self._eval_plain}.",
            bypass=True,
        )

        # \u2500\u2500 Update W/L/D counters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if winner:
            if kira_won:
                self._session_wins  += 1
                self._lifetime_wins += 1
            else:
                self._session_losses  += 1
                self._lifetime_losses += 1
        else:
            self._session_draws  += 1
            self._lifetime_draws += 1
        self._save_lifetime()

        # \u2500\u2500 Score overlay callback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if self.on_score_update:
            try:
                await self.on_score_update(
                    self._session_wins, self._session_losses, self._session_draws,
                    self._lifetime_wins, self._lifetime_losses, self._lifetime_draws,
                )
            except Exception:
                pass

        # Banner announcement for overlays
        if self.on_banner:
            try:
                if winner:
                    if kira_won:
                        banner_text = f"\u265f DUCHESS WINS \u2014 vs {self._opp_name}"
                    else:
                        banner_text = f"\u265f game over \u2014 lost to {self._opp_name}"
                else:
                    banner_text = f"\u265f draw \u2014 vs {self._opp_name} ({status})"
                await self.on_banner(banner_text, 10)
            except Exception:
                pass

        # \u2500\u2500 Spectate hide \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if self._is_viewer_game and self.on_spectate_hide:
            try:
                await self.on_spectate_hide()
            except Exception:
                pass
        self._is_viewer_game = False
        """Phase 1 fixed in-game chat (greeting / gg). Serialized like every
        other HTTP call."""
        try:
            await self._serialized_call(
                self._client.bots.post_message, game_id, text,
                label="post_message",
            )
            print(f"   [Chess] Chat \u2192 game/{game_id}: {text}")
        except Exception as e:
            print(f"   [Chess] post_message failed: {e}")

    async def _send_chat_room(self, challenge_id: str, text: str):
        """Best-effort polite message on a challenge room (before game starts).
        Failures are non-fatal — the decline already went through."""
        # The challenge room uses the same post_message endpoint but for Phase 1
        # we skip it if berserk doesn’t support it, rather than raising.
        try:
            await self._serialized_call(
                self._client.bots.post_message, challenge_id, text,
                label="challenge_chat",
            )
        except Exception:
            pass  # Non-fatal; the decline itself already replied.

    # ── React throttle (mirror of MediaWatch) ─────────────────────────────────

    async def _fire_react(self, summary: str, *, bypass: bool):
        """Fire the on_react callback for a substantive moment. Throttled to one
        per react_min_gap_s EXCEPT bypass events (game start/end, blunders)."""
        if not self.on_react or not summary:
            return
        now = time.time()
        if not bypass and (now - self._last_react_ts) < self.react_min_gap_s:
            print(f"   [Chess] React suppressed (throttle): {summary[:60]}")
            return
        self._last_react_ts = now
        try:
            maybe = self.on_react(summary, bypass=bypass)
            if asyncio.iscoroutine(maybe):
                asyncio.ensure_future(maybe)
        except TypeError:
            # Callback without the bypass kwarg — call positionally.
            try:
                maybe = self.on_react(summary)
                if asyncio.iscoroutine(maybe):
                    asyncio.ensure_future(maybe)
            except Exception as e:
                print(f"   [Chess] on_react error: {e}")
        except Exception as e:
            print(f"   [Chess] on_react error: {e}")
