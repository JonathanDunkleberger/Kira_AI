# dashboard.py - Professional GUI Controller for VTubeBot
import customtkinter as ctk
import threading
import time
from PIL import Image, ImageTk
import asyncio
import sys
import queue

# Import Bot and Tools
from bot import VTubeBot
from music_tools import skip_song
from persona import EmotionalState
from config import AI_NAME

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DashboardApp(ctk.CTk):
    def __init__(self, bot_instance):
        super().__init__()

        self.bot = bot_instance
        self.title(f"{AI_NAME} Control Dashboard")
        self.geometry("1400x800")
        
        # Configure Grid Layout (3 Columns)
        self.grid_columnconfigure(0, weight=1) # Left Panel
        self.grid_columnconfigure(1, weight=3) # Center Panel
        self.grid_columnconfigure(2, weight=1) # Right Panel
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (Controls) ---
        self.left_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.create_left_panel()

        # --- CENTER PANEL (Feeds) ---
        self.center_frame = ctk.CTkFrame(self, corner_radius=0)
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.create_center_panel()

        # Start Update Loop
        self.update_gui()

    def create_left_panel(self):
        # Header
        lbl = ctk.CTkLabel(self.left_frame, text="CONTROLS", font=ctk.CTkFont(size=20, weight="bold"))
        lbl.pack(pady=20)

        # -- Music Section --
        music_frame = ctk.CTkFrame(self.left_frame)
        music_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(music_frame, text="Music Player", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.lbl_now_playing = ctk.CTkLabel(music_frame, text="Now Playing: None", wraplength=200)
        self.lbl_now_playing.pack(pady=5)
        
        ctk.CTkButton(music_frame, text="Skip Song", command=self.action_skip_song, fg_color="red", hover_color="#8B0000").pack(pady=5, padx=10, fill="x")
        # Clear queue not implemented in music_tools yet, but placeholder
        # ctk.CTkButton(music_frame, text="Clear Queue", command=self.action_clear_queue).pack(pady=5, padx=10, fill="x")

        # -- Bot Control --
        bot_frame = ctk.CTkFrame(self.left_frame)
        bot_frame.pack(pady=20, padx=10, fill="x")
        ctk.CTkLabel(bot_frame, text="Bot Actions", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)

        self.btn_pause = ctk.CTkButton(bot_frame, text="Stop Bot", command=self.action_toggle_pause, fg_color="red", hover_color="#8B0000")
        self.btn_pause.pack(pady=5, padx=10, fill="x")

        ctk.CTkButton(bot_frame, text="Force Save Memory", command=self.action_save_memory).pack(pady=5, padx=10, fill="x")

        # -- Emotion Override --
        emotion_frame = ctk.CTkFrame(self.left_frame)
        emotion_frame.pack(pady=20, padx=10, fill="x")
        ctk.CTkLabel(emotion_frame, text="Emotion Override", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.emotion_var = ctk.StringVar(value=self.bot.current_emotion.name)
        emotions = [e.name for e in EmotionalState]
        self.emotion_menu = ctk.CTkOptionMenu(emotion_frame, values=emotions, command=self.action_set_emotion, variable=self.emotion_var)
        self.emotion_menu.pack(pady=10, padx=10)

    def create_center_panel(self):
        # Tab View
        self.tabview = ctk.CTkTabview(self.center_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_transcript = self.tabview.add("Transcript")
        self.tab_twitch = self.tabview.add("Twitch")
        self.tab_thoughts = self.tabview.add("Thoughts")

        # Transcript Box
        self.txt_transcript = ctk.CTkTextbox(self.tab_transcript, font=("Consolas", 14))
        self.txt_transcript.pack(fill="both", expand=True)
        self.txt_transcript.configure(state="disabled")

        # Twitch Box
        self.txt_twitch = ctk.CTkTextbox(self.tab_twitch, font=("Consolas", 12))
        self.txt_twitch.pack(fill="both", expand=True)
        self.txt_twitch.configure(state="disabled")

        # Thoughts Box
        self.txt_thoughts = ctk.CTkTextbox(self.tab_thoughts, font=("Consolas", 12), text_color="#AAAAAA")
        self.txt_thoughts.pack(fill="both", expand=True)
        self.txt_thoughts.configure(state="disabled")

    # --- ACTIONS ---
    def action_skip_song(self):
        threading.Thread(target=skip_song).start()
    
    def action_save_memory(self):
        # self.bot.memory.save() # If explicit save exists
        print("Force Save triggered (Not implemented in MemoryManager yet).")

    def action_toggle_pause(self):
        # Implement pause logic in bot (e.g., set an event)
        # For now, toggling a dummy flag or event
        if not hasattr(self.bot, 'is_paused'): self.bot.is_paused = False
        
        self.bot.is_paused = not self.bot.is_paused
        
        if self.bot.is_paused:
            self.bot.interruption_event.set() # Stop current speech
            self.btn_pause.configure(text="Start Bot", fg_color="green", hover_color="darkgreen")
        else:
            self.bot.interruption_event.clear()
            self.btn_pause.configure(text="Stop Bot", fg_color="red", hover_color="#8B0000")

    def action_set_emotion(self, choice):
        print(f"Setting emotion to {choice}")
        self.bot.current_emotion = EmotionalState[choice]

    # --- UPDATE LOOP ---
    def update_gui(self):
        # 1. Update Transcript
        # Check for new history items
        try:
             # Basic log update - can be improved to avoid full rewrite
             # For now, just grab last 10 lines formatted
             history_text = ""
             for turn in self.bot.conversation_history[-8:]:
                 history_text += f"[{turn['role'].upper()}]: {turn['content']}\n\n"
             
             self.txt_transcript.configure(state="normal")
             self.txt_transcript.delete("0.0", "end")
             self.txt_transcript.insert("0.0", history_text)
             self.txt_transcript.see("end")
             self.txt_transcript.configure(state="disabled")
        except: pass

        # 2. Update Twitch
        try:
             # We need a way to access twitch logs. The Bot prints them.
             # Maybe the bot can store a separate log list for the dashboard.
             # For now, just a placeholder or read from a shared list if we added one.
             pass 
        except: pass

        # 3. Update Status
        self.lbl_now_playing.configure(text=f"Emotion: {self.bot.current_emotion.name}")

        self.after(500, self.update_gui)

def run_dashboard():
    # Initialize Bot
    bot = VTubeBot()
    
    # Run Bot in Background Thread
    bot_thread = threading.Thread(target=run_async_bot, args=(bot,), daemon=True)
    bot_thread.start()

    # Run Dashboard (Main Thread)
    app = DashboardApp(bot)
    app.mainloop()

def run_async_bot(bot):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    run_dashboard()
