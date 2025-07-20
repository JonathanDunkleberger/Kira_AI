# twitch_bot.py

import asyncio
from twitchio.ext import commands
from config import TWITCH_OAUTH_TOKEN, TWITCH_BOT_USERNAME, TWITCH_CHANNEL_TO_JOIN

class TwitchBot(commands.Bot):
    # --- UPDATED ---
    # We now pass the shared list directly instead of a queue
    def __init__(self, chat_message_list: list):
        super().__init__(
            token=TWITCH_OAUTH_TOKEN,
            nick=TWITCH_BOT_USERNAME,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL_TO_JOIN]
        )
        # This is now a direct reference to the list in VTubeBot
        self.chat_message_list = chat_message_list

    async def event_ready(self):
        print(f'--- Twitch bot has logged in as | {self.nick} ---')
        print(f'--- Watching channel | {TWITCH_CHANNEL_TO_JOIN} ---')

    async def event_message(self, message):
        if message.echo or not message.author:
            return

        formatted_message = f"{message.author.name}: {message.content}"
        print(f"[Twitch Chat] {formatted_message}")
        
        # --- UPDATED ---
        # Append the message to the shared list instead of putting it in a queue
        self.chat_message_list.append(formatted_message)