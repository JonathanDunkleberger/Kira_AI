# twitch_bot.py

import asyncio
from twitchio.ext import commands
from config import TWITCH_OAUTH_TOKEN, TWITCH_BOT_USERNAME, TWITCH_CHANNEL_TO_JOIN
from typing import List, Callable

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

        formatted_message = f"{message.author.name}: {message.content}"
        print(f"[Twitch Chat] {formatted_message}")
        
        # --- NEW: Push directly to input queue if available ---
        if self.input_queue:
             await self.input_queue.put(("twitch", formatted_message))
        else:
             self.chat_message_list.append(formatted_message)
        
        # --- ADDED: Reset the main bot's idle timer every time a message arrives ---
        self.timer_callback()