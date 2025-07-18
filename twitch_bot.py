# twitch_bot.py - Handles Twitch chat connection and interaction.

import asyncio
import collections
from twitchio.ext import commands

from config import TWITCH_TOKEN, TWITCH_NICK, TWITCH_CHANNEL

class TwitchBot(commands.Bot):
    def __init__(self, ai_core, conversation_history, is_bot_processing_lock):
        super().__init__(token=TWITCH_TOKEN, prefix='!', initial_channels=[TWITCH_CHANNEL])
        self.ai_core = ai_core
        self.conversation_history = conversation_history
        self.is_bot_processing = is_bot_processing_lock
        self.chat_queue = collections.deque(maxlen=10)

    async def event_ready(self):
        print(f'--- Twitch bot has logged in as {self.nick} ---')
        print(f'--- Watching channel: {TWITCH_CHANNEL} ---')

    async def event_message(self, message):
        if message.author and message.author.name.lower() == self.nick.lower():
            return

        # Always add message to queue
        print(f"[Twitch Chat] {message.author.name}: {message.content}")
        self.chat_queue.append(f"{message.author.name} says in chat: {message.content}")

    async def process_chat_queue(self):
        """A background task that checks the queue and responds when the bot is free."""
        while True:
            await asyncio.sleep(5) # Check every 5 seconds

            if not self.is_bot_processing.locked() and self.chat_queue:
                # Batch process up to 2 messages from the queue
                messages_to_process = []
                while self.chat_queue and len(messages_to_process) < 2:
                    messages_to_process.append(self.chat_queue.popleft())
                
                if not messages_to_process:
                    continue

                print(f"--- Processing {len(messages_to_process)} queued chat messages... ---")
                
                # Combine messages into a single prompt for the AI
                combined_input = "You were busy, but here are some chat messages you missed. Briefly react to them. \n- "
                combined_input += "\n- ".join(messages_to_process)

                async with self.is_bot_processing:
                    try:
                        response_text = await self.ai_core.llm_inference(self.conversation_history, combined_input)
                        if response_text:
                            self.conversation_history.append({"role": "user", "content": combined_input})
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            await self.ai_core.speak_text(response_text)
                    except Exception as e:
                        print(f"--- ERROR processing queued Twitch messages: {e} ---")