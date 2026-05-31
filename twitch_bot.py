# twitch_bot.py

import asyncio
from twitchio.ext import commands
from config import TWITCH_OAUTH_TOKEN, TWITCH_BOT_USERNAME, TWITCH_CHANNEL_TO_JOIN, ALLOW_BROADCASTER_CHAT
from typing import List, Callable
from music_tools import play_kira_song # Added music support

class TwitchBot(commands.Bot):
    # --- UPDATED: __init__ now accepts input_queue ---
    def __init__(self, chat_message_list: List[str], timer_callback: Callable[[], None], input_queue: asyncio.Queue = None, cookie_jar=None, stream_event_callback: Callable = None):
        super().__init__(
            token=TWITCH_OAUTH_TOKEN,
            nick=TWITCH_BOT_USERNAME,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL_TO_JOIN]
        )
        self.chat_message_list = chat_message_list
        self.timer_callback = timer_callback
        self.input_queue = input_queue
        self.cookie_jar = cookie_jar  # CookieJar instance (data layer for !cookies)
        # Optional async callback(kind: str, name: str, extra: dict) called when
        # a raid/sub/resub/gift event arrives via IRC USERNOTICE. See
        # _dispatch_stream_event() below.
        self.stream_event_callback = stream_event_callback

    async def event_ready(self):
        print(f'--- Twitch bot has logged in as | {self.nick} ---')
        print(f'--- Watching channel | {TWITCH_CHANNEL_TO_JOIN} ---')
        # Diagnostic: confirm we actually have a channel object after ready.
        try:
            chans = list(self.connected_channels) if self.connected_channels else []
            print(f"   [TwitchChat] connected_channels at ready: {[c.name for c in chans] or 'NONE'}")
            if not chans:
                print("   [TwitchChat] WARNING: event_ready fired with no connected_channels — "
                      "JOIN likely failed. Common cause: OAuth token missing 'chat:read' scope. "
                      "Regenerate at https://twitchtokengenerator.com (Bot Chat Token).")
        except Exception as _e:
            print(f"   [TwitchChat] connected_channels check failed: {_e}")

    async def event_channel_joined(self, channel):
        # Fires only when JOIN is confirmed by Twitch IRC. If you never see
        # this line after event_ready, the bot's OAuth scope or nickname is
        # being rejected silently — most often a missing chat:read scope.
        print(f"   [TwitchChat] JOIN confirmed for #{channel.name}")

    async def event_raw_data(self, data: str):
        # Lightweight diagnostic: surface inbound PRIVMSG / NOTICE / JOIN /
        # PART lines so we can see whether IRC traffic is actually arriving
        # at all. Filtered to interesting verbs so we don't spam PING/PONG.
        try:
            for line in data.splitlines():
                if not line:
                    continue
                # NOTICE often carries the real reason for auth failures.
                if " NOTICE " in line:
                    print(f"   [TwitchChat][IRC NOTICE] {line.strip()}")
                elif " PRIVMSG " in line:
                    print(f"   [TwitchChat][IRC PRIVMSG] {line.strip()[:200]}")
                elif " USERNOTICE " in line:
                    # Raids, subs, resubs, gifts — all land here.
                    print(f"   [TwitchChat][IRC USERNOTICE] {line.strip()[:300]}")
                    await self._parse_usernotice(line)
                elif " JOIN " in line or " PART " in line:
                    print(f"   [TwitchChat][IRC] {line.strip()}")
        except Exception as _e:
            print(f"   [TwitchChat] event_raw_data parse error: {_e}")

    async def _parse_usernotice(self, line: str) -> None:
        """Parse a USERNOTICE IRC line and dispatch raid/sub/gift events.

        IRC line format (with IRCv3 tags prefix):
          @tag1=val1;tag2=val2 :tmi.twitch.tv USERNOTICE #channel [:message]

        Relevant tags:
          msg-id            — raid | sub | resub | subgift | submysterygift
                              | anonsubgift | anonsubmysterygift | giftpaidupgrade
          display-name      — the actor (raider, subscriber, gifter)
          login             — lowercase username fallback
          msg-param-viewerCount        — raid: how many viewers came over
          msg-param-cumulative-months  — resub: streak length
          msg-param-recipient-display-name — subgift: who got the sub
          msg-param-mass-gift-count    — submysterygift: bomb size
          msg-param-sub-plan           — Prime / 1000 / 2000 / 3000 (tier)
        """
        if self.stream_event_callback is None:
            return
        if not line.startswith("@"):
            return
        try:
            tag_section, _, _rest = line.partition(" ")
            tags = {}
            for kv in tag_section[1:].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    tags[k] = v
            msg_id = tags.get("msg-id", "")
            if not msg_id:
                return
            display = tags.get("display-name") or tags.get("login") or "someone"
            extra = {
                "login": tags.get("login", ""),
                "system_msg": tags.get("system-msg", "").replace("\\s", " "),
            }
            if msg_id == "raid":
                try:
                    extra["viewer_count"] = int(tags.get("msg-param-viewerCount", "0"))
                except ValueError:
                    extra["viewer_count"] = 0
            elif msg_id in ("sub", "resub"):
                try:
                    extra["months"] = int(tags.get("msg-param-cumulative-months", "1"))
                except ValueError:
                    extra["months"] = 1
                extra["tier"] = tags.get("msg-param-sub-plan", "")
            elif msg_id in ("subgift", "anonsubgift"):
                extra["recipient"] = tags.get("msg-param-recipient-display-name", "") \
                    or tags.get("msg-param-recipient-user-name", "someone")
                extra["tier"] = tags.get("msg-param-sub-plan", "")
                if msg_id == "anonsubgift":
                    display = "An anonymous gifter"
            elif msg_id in ("submysterygift", "anonsubmysterygift"):
                try:
                    extra["mass_count"] = int(tags.get("msg-param-mass-gift-count", "1"))
                except ValueError:
                    extra["mass_count"] = 1
                if msg_id == "anonsubmysterygift":
                    display = "An anonymous gifter"
            else:
                # Not a kind we react to (bitsbadgetier, ritual, etc.)
                return
            print(f"   [TwitchChat] STREAM EVENT → {msg_id} from {display} extra={extra}")
            try:
                await self.stream_event_callback(msg_id, display, extra)
            except Exception as _cb_err:
                print(f"   [TwitchChat] stream_event_callback error: {_cb_err}")
        except Exception as _e:
            print(f"   [TwitchChat] _parse_usernotice error: {_e}")

    async def post_message(self, text: str) -> bool:
        """Send ``text`` to the joined Twitch channel. Returns True on success,
        False if no channel is connected yet or the underlying send raises.

        Called by ChatPoster, which handles rate limiting and sanitization —
        this method intentionally does NOT impose its own cooldown so that
        rate logic stays in one place.
        """
        try:
            channels = list(self.connected_channels) if self.connected_channels else []
            if not channels:
                # event_ready hasn't fired yet, or we got disconnected.
                print("   [Twitch] post_message: no connected channels yet.")
                return False
            await channels[0].send(text)
            return True
        except Exception as e:
            print(f"   [Twitch] post_message failed: {e}")
            return False

    async def event_message(self, message):
        # Stage 1 of pipeline. If you don't see [TwitchChat] Received lines
        # but DO see [IRC PRIVMSG] lines above, the echo/author guard is
        # dropping everything. If you see neither, IRC isn't delivering.
        if not message.author:
            print(f"   [TwitchChat] FILTERED reason=no_author content={message.content[:80]!r}")
            return

        # In TwitchIO 2.x, message.echo is True whenever author.name == bot.nick.
        # If the OAuth token is for the broadcaster account, ALL broadcaster messages
        # get echo=True and are silently dropped — this is the most common cause of
        # "chat isn't working" when testing with your own channel account.
        # ALLOW_BROADCASTER_CHAT=true bypasses the echo filter for the channel owner
        # so their messages (typed in browser, not sent by the bot itself) reach Kira.
        is_broadcaster = message.author.name.lower() == TWITCH_CHANNEL_TO_JOIN.lower()
        if message.echo and not (ALLOW_BROADCASTER_CHAT and is_broadcaster):
            print(f"   [TwitchChat] FILTERED reason=echo author={message.author.name!r} "
                  f"(set ALLOW_BROADCASTER_CHAT=true in .env if this is the channel owner testing)")
            return

        author_name = message.author.name
        print(f"   [TwitchChat] Received from {author_name}: {message.content[:160]}")

        # --- !cookies — personal + shared jar query ---
        if message.content.strip().lower() == '!cookies':
            if self.cookie_jar is not None:
                personal = self.cookie_jar.get_chatter(author_name)
                shared = self.cookie_jar.get_shared()
                reply = (
                    f"@{author_name} you have {personal} cookie"
                    f"{'s' if personal != 1 else ''} \U0001f36a — "
                    f"shared jar: {shared}/35"
                )
                try:
                    await self.post_message(reply)
                    print(f"   [Cookies] Replied to {author_name}: personal={personal} shared={shared}")
                except Exception as e:
                    print(f"   [Cookies] Reply failed: {e}")
            else:
                print("   [Cookies] !cookies used but no cookie_jar attached.")
            self.timer_callback(human_speech=True)
            return

        # --- NEW: Check for Song Requests ---
        if message.content.lower().startswith('!sr ') or message.content.lower().startswith('!play '):
            song_name = message.content[4:].strip() if message.content.lower().startswith('!sr ') else message.content[6:].strip()
            print(f"   [Twitch] Song Request detected: {song_name}")
            
            # 1. Play Song Immediately
            # Run in executor to not block the asyncio loop
            asyncio.get_event_loop().run_in_executor(None, play_kira_song, song_name)
            
            # 2. Inform Kira via System Message
            system_msg = f"[System: User {author_name} requested song: {song_name}]"
             # --- PUSH SYSTEM MSG TO QUEUE ---
            if self.input_queue:
                 await self.input_queue.put(("twitch", system_msg))
                 print(f"   [TwitchChat] Forwarded song-request system msg to brain queue.")
            
            # Don't add command itself to chat history or brain processing as user text
            self.timer_callback(human_speech=True)
            return

        formatted_message = f"{author_name}: {message.content}"
        
        # --- NEW: Push directly to input queue if available ---
        if self.input_queue:
             await self.input_queue.put(("twitch", formatted_message))
             print(f"   [TwitchChat] Forwarded to brain queue (qsize={self.input_queue.qsize()}).")
        else:
             self.chat_message_list.append(formatted_message)
             print(f"   [TwitchChat] FILTERED reason=no_input_queue — fell back to legacy list (brain won't see this).")
        
        # --- ADDED: Reset the main bot's idle timer every time a message arrives ---
        self.timer_callback(human_speech=True)