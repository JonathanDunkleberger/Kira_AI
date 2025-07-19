# twitch_bot.py
import asyncio
from twitchio.ext import commands
# Correctly import the updated variable names from config
from config import TWITCH_OAUTH_TOKEN, TWITCH_BOT_USERNAME, TWITCH_CHANNEL_TO_JOIN

class TwitchBot(commands.Bot):
    def __init__(self, chat_queue: asyncio.Queue):
        # Use the correct, imported variables here
        super().__init__(
            token=TWITCH_OAUTH_TOKEN,
            nick=TWITCH_BOT_USERNAME,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL_TO_JOIN]
        )
        self.chat_queue = chat_queue

    async def event_ready(self):
        print(f'--- Twitch bot has logged in as | {self.nick} ---')
        print(f'--- Watching channel | {TWITCH_CHANNEL_TO_JOIN} ---')

    async def event_message(self, message):
        if message.echo or not message.author:
            return

        formatted_message = f"{message.author.name}: {message.content}"
        print(f"[Twitch Chat] {formatted_message}")
        # Put the formatted message into the queue for the main bot to process
        await self.chat_queue.put(formatted_message)