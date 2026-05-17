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


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Accepts a full YouTube URL or a bare 11-char video ID. Returns the ID, or None."""
    if not url_or_id:
        return None
    s = url_or_id.strip()
    # Already an ID?
    if len(s) == 11 and re.match(r'^[A-Za-z0-9_-]+$', s):
        return s
    # Common URL formats
    m = re.search(r'(?:v=|youtu\.be/|youtube\.com/live/|youtube\.com/embed/)([A-Za-z0-9_-]{11})', s)
    if m:
        return m.group(1)
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
        if not PYTCHAT_AVAILABLE:
            print("   [YouTube] pytchat not installed. Run: pip install pytchat")
            return False

        vid = extract_video_id(video_id_or_url)
        if not vid:
            print(f"   [YouTube] Invalid URL or video ID: {video_id_or_url}")
            return False

        if self.running:
            print("   [YouTube] Already running. Stop first.")
            return False

        try:
            self.chat = pytchat.create(video_id=vid)
            self.video_id = vid
            self.running = True
            self.task = asyncio.create_task(self._poll_loop())
            print(f"   [YouTube] Connected to live chat for video ID: {vid}")
            return True
        except Exception as e:
            print(f"   [YouTube] Failed to connect: {e}")
            self.chat = None
            self.video_id = None
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
        """Background loop: polls chat every ~2 seconds, pushes new messages."""
        while self.running and self.chat:
            try:
                if not self.chat.is_alive():
                    print("   [YouTube] Live chat ended (stream offline?).")
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
                print(f"   [YouTube] Poll error: {e}")
                await asyncio.sleep(5.0)

    def _get_items_sync(self):
        """Pulls the latest batch of chat items synchronously (called via to_thread)."""
        out = []
        try:
            for c in self.chat.get().sync_items():
                out.append((c.author.name, c.message))
        except Exception as e:
            print(f"   [YouTube] sync_items error: {e}")
        return out
