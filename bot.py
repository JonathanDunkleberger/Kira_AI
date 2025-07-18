# bot.py - Main application file.

import asyncio
import webrtcvad
import collections
import pyaudio
import traceback

from ai_core import AI_Core
from twitch_bot import TwitchBot
from config import AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS
# from memory import MemoryManager  # Uncomment to enable memory
# from web_search import async_GoogleSearch # Uncomment to enable web search

class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.ai_core = AI_Core(self.interruption_event)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.is_bot_processing = asyncio.Lock()
        self.pa = None
        self.stream = None
        self.frames_per_buffer_val = int(16000 * 30 / 1000)
        self.conversation_history = []
        self.MAX_HISTORY_TURNS = 10
        self.was_interrupted = False
        
        # self.memory = MemoryManager() # Uncomment to enable memory
        
        self.twitch_bot = TwitchBot(self.ai_core, self.conversation_history, self.is_bot_processing)

    async def run(self):
        try:
            await self.ai_core.initialize()
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=self.frames_per_buffer_val
            )
            print("--- PyAudio microphone stream opened. ---")
            print(f"\n{'='*50}\n {AI_NAME} is now listening...\n{'='*50}\n")

            # Create tasks for VAD, Twitch listener, AND the new queue processor
            vad_task = asyncio.create_task(self.vad_loop())
            twitch_listen_task = asyncio.create_task(self.twitch_bot.start())
            twitch_queue_task = asyncio.create_task(self.twitch_bot.process_chat_queue())

            await asyncio.gather(vad_task, twitch_listen_task, twitch_queue_task)

        except Exception as e:
            print(f"A critical error occurred during VTubeBot run(): {e}")
            traceback.print_exc()

    async def vad_loop(self):
        while True:
            print("Listening for voice activity...")
            frames = collections.deque()
            triggered = False
            silent_chunks = 0
            max_silent_chunks = int(PAUSE_THRESHOLD * 1000 / 30)

            while True:
                try:
                    await asyncio.sleep(0.01) # CPU freeze fix

                    frame_data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer_val, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(frame_data, 16000)

                    if self.is_bot_processing.locked() and is_speech:
                        self.interruption_event.set()
                        self.was_interrupted = True
                        print("   User interruption detected.")
                        break

                    if is_speech:
                        if not triggered:
                            print("   Voice activity detected, recording...")
                            triggered = True
                        frames.append(frame_data)
                        silent_chunks = 0
                    elif triggered: # Only count silence after speech has started
                        frames.append(frame_data)
                        silent_chunks += 1
                        if silent_chunks > max_silent_chunks:
                            print("   End of speech detected.")
                            break
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed: continue
                    else: raise
            
            if self.was_interrupted:
                self.was_interrupted = False
                frames.clear()
                continue

            if len(frames) > 10:
                audio_data = b"".join(frames)
                asyncio.create_task(self.process_audio(audio_data, 16000))

    async def process_audio(self, audio_data: bytes, sample_rate: int):
        async with self.is_bot_processing:
            try:
                user_input = await self.ai_core.transcribe_audio(audio_data, sample_rate)
                if not user_input or len(user_input.split()) < 1:
                    print("   Ignoring empty or very short transcription.")
                    return

                print(f"\n--- User: {user_input} ---")
                
                # To integrate memory, you would add:
                # memory_context = self.memory.search_memories(user_input)
                # And pass memory_context to llm_inference
                response_text = await self.ai_core.llm_inference(self.conversation_history, user_input)
                
                if response_text:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    if len(self.conversation_history) > self.MAX_HISTORY_TURNS * 2:
                        self.conversation_history = self.conversation_history[-self.MAX_HISTORY_TURNS*2:]
                    
                    # self.memory.add_memory(user_input, response_text) # Uncomment to save memory
                    await self.ai_core.speak_text(response_text)
                    
            except Exception as e:
                print(f"--- ERROR in process_audio: {e} ---")
                traceback.print_exc()

    def shutdown(self):
        print("\nShutting down...")
        try:
            if self.stream:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
            if self.pa: self.pa.terminate()
        except Exception as e:
            print(f"Error during PyAudio shutdown: {e}")
        print("--- VTubeBot resources cleaned up. ---")

async def main():
    bot = VTubeBot()
    try:
        await bot.run()
    except asyncio.CancelledError:
        print("Main task cancelled.")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication shutdown requested by user.")