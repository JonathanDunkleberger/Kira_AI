# bot.py - Main application file.

import asyncio
import webrtcvad
import collections
import pyaudio
import time
import traceback

from ai_core import AI_Core
from config import AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS

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
        self.was_interrupted = False # --- ADDED: State flag for interruption ---

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
            await self.vad_loop()
        except Exception as e:
            print(f"A critical error occurred during VTubeBot run(): {e}")
            traceback.print_exc()
        # Shutdown is now handled only in __main__

    # --- REVISED: More robust VAD and interruption handling ---
    async def vad_loop(self):
        while True:
            print("Listening for voice activity...")
            frames = collections.deque()
            triggered = False
            silent_chunks = 0
            max_silent_chunks = int(PAUSE_THRESHOLD * 1000 / 30)

            # --- Recording Loop ---
            while True:
                try:
                    frame_data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer_val, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(frame_data, 16000)

                    # --- Interruption Logic ---
                    if self.is_bot_processing.locked() and is_speech:
                        self.interruption_event.set()
                        self.was_interrupted = True
                        print("   User interruption detected.")
                        break # Break out of the recording loop immediately

                    if is_speech:
                        if not triggered:
                            print("   Voice activity detected, recording...")
                            triggered = True
                        frames.append(frame_data)
                        silent_chunks = 0
                    elif triggered:
                        frames.append(frame_data)
                        silent_chunks += 1
                        if silent_chunks > max_silent_chunks:
                            print("   End of speech detected.")
                            break
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed: continue
                    else: raise
            
            # --- Processing Logic ---
            if self.was_interrupted:
                # If interrupted, clear the flag and discard the audio frames
                # This prevents the bot from responding to the interruption itself
                self.was_interrupted = False
                frames.clear()
                continue # Go back to the main listening state

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
                
                response_text = await self.ai_core.llm_inference(self.conversation_history, user_input)
                
                if response_text:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    if len(self.conversation_history) > self.MAX_HISTORY_TURNS * 2:
                        self.conversation_history = self.conversation_history[-(self.MAX_HISTORY_TURNS * 2):]
                    
                    await self.ai_core.speak_text(response_text)
                    
            except Exception as e:
                print(f"--- ERROR in process_audio: {e} ---")
                traceback.print_exc()

    def shutdown(self):
        print("\nShutting down...")
        try:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
            if self.pa:
                try:
                    self.pa.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        print("--- VTubeBot resources cleaned up. ---")

if __name__ == "__main__":
    bot = VTubeBot()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        print("\nApplication shutdown requested by user.")
    finally:
        bot.shutdown()
        try:
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except RuntimeError:
            # Suppress errors if the event loop is already closed or not running
            pass
        loop.close()