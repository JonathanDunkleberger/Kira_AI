# youtube_bot.py — YouTube live chat listener
import asyncio
import re
from typing import Callable, Optional

try:
    import pytchat
    PYTCHAT_AVAILABLE = True
except ImportError:
    pytchat = None
    PYTCHAT_AVAILABLE = False


# A YouTube video ID is exactly 11 chars from [A-Za-z0-9_-].
_VIDEO_ID_RE = re.compile(r'[A-Za-z0-9_-]{11}')

# Known URL patterns where a video ID can appear. Tried in order; first hit wins.
# Studio URLs (studio.youtube.com/video/{ID}/livestreaming|edit|analytics|...) are
# the natural copy from a streamer's dashboard — must be supported.
_URL_PATTERNS = (
    re.compile(r'(?:^|[?&])v=([A-Za-z0-9_-]{11})'),                    # watch?v=ID
    re.compile(r'youtu\.be/([A-Za-z0-9_-]{11})'),                       # short
    re.compile(r'youtube\.com/live/([A-Za-z0-9_-]{11})'),               # live
    re.compile(r'youtube\.com/embed/([A-Za-z0-9_-]{11})'),              # embed
    re.compile(r'youtube\.com/shorts/([A-Za-z0-9_-]{11})'),             # shorts
    re.compile(r'studio\.youtube\.com/video/([A-Za-z0-9_-]{11})'),      # Studio (any subpage)
)


class InvalidYouTubeURLError(ValueError):
    """Raised when extract_video_id cannot find a usable ID. Carries a user-facing
    explanation in .reason so callers can surface a helpful message."""
    def __init__(self, raw: str, reason: str):
        super().__init__(reason)
        self.raw = raw
        self.reason = reason


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Accepts a full YouTube URL or a bare 11-char video ID. Returns the ID, or None.

    Handles standard watch URLs, youtu.be, youtube.com/live, /embed, /shorts, AND
    studio.youtube.com/video/{ID}/{livestreaming,edit,analytics,...} — the URL a
    streamer naturally copies from their own Studio dashboard.

    Explicitly rejects rtmp:// ingest URLs (OBS stream key, no chat there).
    Falls back to a generic 11-char [A-Za-z0-9_-] scan if no known pattern matched,
    so unusual URL shapes still work as long as a video ID is present somewhere.
    """
    if not url_or_id:
        return None
    s = url_or_id.strip()

    # RTMP ingest URLs are the OBS stream key, NOT a watch URL. Reject explicitly
    # so the caller can show a helpful "that's your stream key" error instead of
    # the generic "invalid URL".
    if s.lower().startswith(("rtmp://", "rtmps://")):
        raise InvalidYouTubeURLError(
            s,
            "That's your stream ingest URL (OBS stream key), not a watch URL. "
            "Paste the video's watch URL or Studio URL instead.",
        )

    # Bare 11-char ID
    if len(s) == 11 and _VIDEO_ID_RE.fullmatch(s):
        return s

    # Try known URL patterns first (more specific → safer than the generic scan).
    for pat in _URL_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)

    # Fallback: scan for any 11-char ID-shaped token in the string. Catches odd URL
    # variants (mobile, m.youtube.com paths, share links with extra prefixes, etc.)
    # without us having to enumerate every YouTube URL shape that exists.
    m = _VIDEO_ID_RE.search(s)
    if m:
        return m.group(0)

    return None


class YouTubeBot:
    """Polls YouTube live chat for a given video ID and pushes messages to the input_queue."""

    def __init__(self, input_queue: asyncio.Queue, timer_callback: Callable, chat_log: list):
        self.input_queue = input_queue
        self.timer_callback = timer_callback
        self.chat_log = chat_log
        self.video_id: Optional[str] = None
        self.chat = None
        self.task: Optional[asyncio.Task] = None
        self.running = False

    def start(self, video_id_or_url: str) -> bool:
        """Begins polling YouTube chat. Returns True on success, False if invalid ID
        or library missing."""
        print(f"   [YouTube] Input received: {video_id_or_url!r}")
        if not PYTCHAT_AVAILABLE:
            print("   [YouTube] pytchat not installed. Run: pip install pytchat")
            return False

        try:
            vid = extract_video_id(video_id_or_url)
        except InvalidYouTubeURLError as e:
            print(f"   [YouTube] {e.reason}")
            print(f"   [YouTube] (received: {e.raw!r})")
            return False
        if not vid:
            print(f"   [YouTube] Could not extract an 11-char video ID from: {video_id_or_url!r}")
            print("   [YouTube] Expected a watch URL (youtube.com/watch?v=ID), short URL "
                  "(youtu.be/ID), Studio URL (studio.youtube.com/video/ID/...), live URL "
                  "(youtube.com/live/ID), or a bare 11-char video ID.")
            return False
        print(f"   [YouTube] Extracted video ID: {vid!r}")

        if self.running:
            print("   [YouTube] Already running. Stop first.")
            return False

        try:
            print(f"   [YouTube] Connecting to live chat for {vid}...")
            # interruptable=False — pytchat otherwise installs a SIGINT handler via
            # signal.signal(), which raises "signal only works in main thread of the
            # main interpreter" when start() is invoked from a Tk/dashboard background
            # thread. Disabling it lets pytchat run from any thread.
            self.chat = pytchat.create(video_id=vid, interruptable=False)
            self.video_id = vid
            self.running = True
            self.task = asyncio.create_task(self._poll_loop())
            print(f"   [YouTube] Connected — polling chat for video ID: {vid}")
            return True
        except Exception as e:
            print(f"   [YouTube] Failed to connect: {e}")
            self.chat = None
            self.video_id = None
            return False

    async def post_message(self, text: str) -> bool:
        """Stub: posting to YouTube live chat is NOT currently supported.

        pytchat is a read-only library. Sending requires the YouTube Data API
        v3 ``liveChatMessages.insert`` endpoint, which needs an OAuth2 flow
        with the ``youtube.force-ssl`` scope and a fresh access token per
        broadcast. Until that's wired up, this method logs once and returns
        False so ChatPoster can fall through to Twitch cleanly.
        """
        if not getattr(self, "_post_unsupported_logged", False):
            print("   [YouTube] post_message: not implemented (pytchat is read-only). "
                  "Enable Twitch posting only, or wire YouTube Data API OAuth.")
            self._post_unsupported_logged = True
        return False

    def stop(self):
        """Stops the chat polling loop."""
        self.running = False
        if self.chat:
            try:
                self.chat.terminate()
            except Exception:
                pass
        self.chat = None
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None
        self.video_id = None
        print("   [YouTube] Disconnected.")

    async def _poll_loop(self):
        """Background loop: polls chat every ~2 seconds, pushes new messages.

        Resilience: pytchat occasionally reports is_alive() == False (or raises
        in get()/sync_items()) on a stream that is in fact still live — usually
        a transient network blip or a YouTube continuation-token hiccup.
        Silently giving up there means first-time YouTube chatters get dropped,
        which is exactly the audience we can't afford to lose. So instead of
        breaking on the first dead signal, we attempt a bounded reconnect
        sequence; only after that exhausts do we actually stop.
        """
        while self.running and self.chat:
            try:
                if not self.chat.is_alive():
                    # pytchat says the chat is gone. Could be a real end-of-stream,
                    # could be a false positive. Try to reconnect a few times before
                    # accepting it.
                    print("   [YouTube] Chat connection lost \u2014 reconnecting...")
                    if await self._attempt_reconnect():
                        print("   [YouTube] Reconnected.")
                        continue
                    # All reconnect attempts failed \u2014 stream is genuinely offline
                    # or YouTube is rejecting us. Now we can stop.
                    print("   [YouTube] Live chat ended (stream offline?) \u2014 giving up after retries.")
                    self.running = False
                    break

                # pytchat is sync; offload to a thread
                items = await asyncio.to_thread(self._get_items_sync)
                for author, message in items:
                    formatted = f"{author}: {message}"
                    print(f"[YouTube Chat] {formatted}")
                    self.chat_log.append(f"[YT] {formatted}")
                    if len(self.chat_log) > 200:
                        del self.chat_log[:50]
                    await self.input_queue.put(("youtube", formatted))
                    self.timer_callback(human_speech=True)

                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't immediately give up on the next iteration \u2014 if pytchat is
                # in a bad state (e.g. continuation token expired), reconnecting
                # is the only way to get messages flowing again.
                print(f"   [YouTube] Poll error: {e} \u2014 attempting reconnect...")
                if await self._attempt_reconnect():
                    print("   [YouTube] Reconnected.")
                    continue
                print("   [YouTube] Reconnect failed after retries \u2014 stopping listener.")
                self.running = False
                break

    async def _attempt_reconnect(self, max_attempts: int = 4) -> bool:
        """Tear down the current pytchat handle and try to re-create it for the
        same video_id with exponential-ish backoff. Returns True if a new live
        handle is in place, False if we should accept that the stream is gone.

        Backoff: 3s, 6s, 12s, 20s (capped). Total wait \u2248 41s before giving up,
        which is long enough to ride out typical YouTube blips without leaving
        chatters waiting forever if the stream really did end.
        """
        if not self.running or not self.video_id or not PYTCHAT_AVAILABLE:
            return False

        # Drop the old handle first so a fresh continuation token is issued.
        old = self.chat
        self.chat = None
        if old is not None:
            try:
                old.terminate()
            except Exception:
                pass

        delays = [3.0, 6.0, 12.0, 20.0]
        for attempt in range(1, max_attempts + 1):
            if not self.running:
                return False
            delay = delays[min(attempt - 1, len(delays) - 1)]
            try:
                await asyncio.sleep(delay)
                if not self.running:
                    return False
                print(f"   [YouTube] Reconnect attempt {attempt}/{max_attempts} for {self.video_id}...")
                new_chat = await asyncio.to_thread(
                    lambda: pytchat.create(video_id=self.video_id, interruptable=False)
                )
                # pytchat.create succeeds lazily; verify it's actually live before
                # declaring victory.
                if new_chat and new_chat.is_alive():
                    self.chat = new_chat
                    return True
                # Not alive \u2014 dispose and loop to next attempt.
                try:
                    new_chat.terminate()
                except Exception:
                    pass
            except Exception as e:
                print(f"   [YouTube] Reconnect attempt {attempt} error: {e}")
        return False

    def _get_items_sync(self):
        """Pulls the latest batch of chat items synchronously (called via to_thread)."""
        out = []
        try:
            for c in self.chat.get().sync_items():
                out.append((c.author.name, c.message))
        except Exception as e:
            print(f"   [YouTube] sync_items error: {e}")
        return out
