# twitch_bot.py

import asyncio
from twitchio.ext import commands
from config import TWITCH_OAUTH_TOKEN, TWITCH_BOT_USERNAME, TWITCH_CHANNEL_TO_JOIN
from typing import List, Callable
from music_tools import play_kira_song # Added music support

class TwitchBot(commands.Bot):
    # --- UPDATED: __init__ now accepts input_queue ---
    def __init__(self, chat_message_list: List[str], timer_callback: Callable[[], None], input_queue: asyncio.Queue = None):
        super().__init__(
            token=TWITCH_OAUTH_TOKEN,
            nick=TWITCH_BOT_USERNAME,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL_TO_JOIN]
        )
        self.chat_message_list = chat_message_list
        self.timer_callback = timer_callback
        self.input_queue = input_queue

    async def event_ready(self):
        print(f'--- Twitch bot has logged in as | {self.nick} ---')
        print(f'--- Watching channel | {TWITCH_CHANNEL_TO_JOIN} ---')

    async def event_message(self, message):
        if message.echo or not message.author:
            return

        # --- NEW: Check for Song Requests ---
        if message.content.lower().startswith('!sr ') or message.content.lower().startswith('!play '):
            song_name = message.content[4:].strip() if message.content.lower().startswith('!sr ') else message.content[6:].strip()
            print(f"   [Twitch] Song Request detected: {song_name}")
            
            # 1. Play Song Immediately
            # Run in executor to not block the asyncio loop
            asyncio.get_event_loop().run_in_executor(None, play_kira_song, song_name)
            
            # 2. Inform Kira via System Message
            system_msg = f"[System: User {message.author.name} requested song: {song_name}]"
             # --- PUSH SYSTEM MSG TO QUEUE ---
            if self.input_queue:
                 await self.input_queue.put(("twitch", system_msg))
            
            # Don't add command itself to chat history or brain processing as user text
            self.timer_callback()
            return

        formatted_message = f"{message.author.name}: {message.content}"
        print(f"[Twitch Chat] {formatted_message}")
        
        # --- NEW: Push directly to input queue if available ---
        if self.input_queue:
             await self.input_queue.put(("twitch", formatted_message))
        else:
             self.chat_message_list.append(formatted_message)
        
        # --- ADDED: Reset the main bot's idle timer every time a message arrives ---
        self.timer_callback()