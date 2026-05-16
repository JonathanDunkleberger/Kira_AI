# dashboard.py - Kira AI Control Dashboard v2
import customtkinter as ctk
import threading
from PIL import ImageGrab
import asyncio

from bot import VTubeBot
from music_tools import skip_song, clear_queue, get_now_playing
from persona import EmotionalState
from game_mode_controller import ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL
from config import AI_NAME

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")

C_BG      = "#F5F0E8"  # warm parchment
C_PANEL   = "#EDE6D6"  # soft beige
C_SURFACE = "#DDD0BA"  # tan
C_ACCENT  = "#8B5E3C"  # warm brown
C_GREEN   = "#5C7A5C"  # muted sage green
C_YELLOW  = "#A0783A"  # amber/honey
C_RED     = "#9B4040"  # muted terracotta
C_TEXT    = "#2E1F0F"  # dark espresso
C_MUTED   = "#9E8C78"  # warm grey-brown

EMOTION_COLORS = {
    "HAPPY":       C_GREEN,
    "SASSY":       C_ACCENT,
    "MOODY":       C_MUTED,
    "EMOTIONAL":   "#7B5EA7",  # dusty mauve
    "HYPERACTIVE": C_YELLOW,
}


class KiraDashboard(ctk.CTk):
    def __init__(self, bot: VTubeBot):
        super().__init__()
        self.bot = bot
        self.title(f"{AI_NAME} - Control Center")
        self.geometry("1600x920")
        self.minsize(1280, 720)
        self.configure(fg_color=C_BG)

        self._vision_lock = False
        self._current_image_ref = None
        self._last_hist_len = 0
        self._last_twitch_len = 0

        self.grid_columnconfigure(0, weight=0, minsize=270)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1, minsize=300)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._build_left()
        self._build_center()
        self._build_right()
        self._build_statusbar()

        self._update_loop()
        self._vision_loop()

    # LEFT PANEL

    def _build_left(self):
        frame = ctk.CTkScrollableFrame(
            self, width=260, corner_radius=0,
            fg_color=C_PANEL, scrollbar_button_color=C_SURFACE
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text=f"[ {AI_NAME.upper()} ]",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=C_ACCENT
        ).pack(pady=(16, 4))
        ctk.CTkLabel(
            frame, text="Control Center",
            font=ctk.CTkFont(size=11), text_color=C_MUTED
        ).pack(pady=(0, 12))

        self.mode_label = ctk.CTkLabel(
            frame, text="COMPANION MODE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_GREEN, fg_color=C_SURFACE,
            corner_radius=6, padx=10, pady=4
        )
        self.mode_label.pack(fill="x", padx=12, pady=(0, 6))
        self.mode_btn = ctk.CTkButton(
            frame, text="Switch to Streamer Mode",  # starts in companion
            fg_color=C_SURFACE, hover_color=C_ACCENT, text_color=C_TEXT,
            height=28, command=self._toggle_mode, font=ctk.CTkFont(size=11)
        )
        self.mode_btn.pack(fill="x", padx=12, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="CURRENT ACTIVITY",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 2))
        self.activity_entry = ctk.CTkEntry(
            frame, placeholder_text="e.g. playing Clannad",
            fg_color=C_SURFACE, border_color=C_ACCENT, text_color=C_TEXT,
            placeholder_text_color=C_MUTED, height=32
        )
        self.activity_entry.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkButton(
            frame, text="Set Activity", height=28,
            fg_color=C_ACCENT, hover_color="#6B4528",
            command=self._set_activity, font=ctk.CTkFont(size=11)
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.activity_display = ctk.CTkLabel(
            frame, text="None", font=ctk.CTkFont(size=11),
            text_color=C_MUTED, wraplength=220
        )
        self.activity_display.pack(padx=12, pady=(0, 8))

        self.immersive_switch = ctk.CTkSwitch(
            frame, text="Immersive Mode",
            command=self._toggle_immersive,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            font=ctk.CTkFont(size=12),
        )
        self.immersive_switch.pack(anchor="w", padx=14, pady=(0, 4))

        ctk.CTkLabel(
            frame,
            text="Auto-enables for VNs / movies / anime. Kira stays quiet unless invited. Off otherwise \u2014 she's normally chatty.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        _divider(frame)

        ctk.CTkLabel(frame, text="EMOTION",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 2))
        self.emotion_badge = ctk.CTkLabel(
            frame, text="HAPPY",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_GREEN, fg_color=C_SURFACE,
            corner_radius=6, padx=10, pady=4
        )
        self.emotion_badge.pack(fill="x", padx=12, pady=(0, 6))
        self.emotion_menu = ctk.CTkOptionMenu(
            frame, values=[e.name for e in EmotionalState],
            command=self._set_emotion, fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_PANEL, font=ctk.CTkFont(size=11), height=28
        )
        self.emotion_menu.pack(fill="x", padx=12, pady=(0, 10))

        _divider(frame)

        ctk.CTkLabel(frame, text="OBSERVER MODE",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 2))
        self.obs_switch = ctk.CTkSwitch(
            frame, text="Vision Active", command=self._toggle_observer,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            font=ctk.CTkFont(size=12)
        )
        if self.bot.game_mode_controller.is_active:
            self.obs_switch.select()
        self.obs_switch.pack(anchor="w", padx=14, pady=(4, 6))

        ctk.CTkLabel(
            frame,
            text="VN AUTO-PLAY (experimental) \u2014 Kira reads and advances dialogue herself. Off by default. Companion mode is the main experience.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 2))
        self.vn_switch = ctk.CTkSwitch(
            frame, text="VN Auto-Play",
            command=self._toggle_vn,
            button_color=C_YELLOW, progress_color=C_YELLOW,
            font=ctk.CTkFont(size=12)
        )
        self.vn_switch.pack(anchor="w", padx=14, pady=(0, 4))
        self.vn_status_label = ctk.CTkLabel(
            frame, text="VN: Standby",
            font=ctk.CTkFont(size=10), text_color=C_MUTED, wraplength=230
        )
        self.vn_status_label.pack(anchor="w", padx=14, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="BOT CONTROLS",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))
        self.btn_toggle = ctk.CTkButton(
            frame, text="Pause Bot",
            fg_color=C_RED, hover_color="#7A2E2E", height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._toggle_bot
        )
        self.btn_toggle.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkButton(
            frame, text="Reload Personality",
            fg_color=C_SURFACE, hover_color=C_ACCENT, height=28,
            font=ctk.CTkFont(size=11),
            command=lambda: self.bot.ai_core.reload_personality()
        ).pack(fill="x", padx=12, pady=(0, 16))

        _divider(frame)

        ctk.CTkLabel(frame, text="INVITE",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        self.btn_invite = ctk.CTkButton(
            frame, text="\U0001f4ac  Ask Kira's Thoughts",
            fg_color=C_GREEN, hover_color="#3D5C3D", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._invite_kira
        )
        self.btn_invite.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            frame,
            text="In Companion mode Kira stays quiet by default. Tap to invite her thoughts on what's happening.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="MUSIC PLAYER",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))
        self.music_label = ctk.CTkLabel(
            frame, text="Now Playing: Nothing",
            font=ctk.CTkFont(size=11), wraplength=220, text_color=C_TEXT
        )
        self.music_label.pack(padx=12, pady=(0, 6))
        music_row = ctk.CTkFrame(frame, fg_color="transparent")
        music_row.pack(fill="x", padx=12, pady=(0, 16))
        music_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            music_row, text="Skip Song", height=30,
            fg_color=C_RED, hover_color="#7A2E2E",
            command=lambda: threading.Thread(target=skip_song, daemon=True).start(),
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        ctk.CTkButton(
            music_row, text="Clear Queue", height=30,
            fg_color=C_SURFACE, hover_color=C_MUTED,
            command=lambda: threading.Thread(target=clear_queue, daemon=True).start(),
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=1, padx=(2, 0), sticky="ew")

    # CENTER PANEL

    def _build_center(self):
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color=C_PANEL)
        frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        frame.grid_rowconfigure(0, weight=3)
        frame.grid_rowconfigure(1, weight=0)
        frame.grid_rowconfigure(2, weight=2)
        frame.grid_columnconfigure(0, weight=1)

        vision_wrap = ctk.CTkFrame(frame, fg_color=C_SURFACE, corner_radius=8)
        vision_wrap.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 2))
        self.vision_label = ctk.CTkLabel(
            vision_wrap,
            text="Vision Offline  -  Enable Observer Mode to see screen",
            font=ctk.CTkFont(size=13), text_color=C_MUTED
        )
        self.vision_label.pack(fill="both", expand=True, padx=4, pady=4)

        vis_desc_wrap = ctk.CTkFrame(frame, fg_color=C_BG, corner_radius=0, height=38)
        vis_desc_wrap.grid(row=1, column=0, sticky="ew", padx=8)
        vis_desc_wrap.grid_propagate(False)
        self.vision_desc_label = ctk.CTkLabel(
            vis_desc_wrap, text="", wraplength=900,
            font=ctk.CTkFont(size=11), text_color=C_MUTED, anchor="w"
        )
        self.vision_desc_label.pack(side="left", padx=10, fill="y")

        transcript_wrap = ctk.CTkFrame(frame, fg_color=C_SURFACE, corner_radius=8)
        transcript_wrap.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        transcript_wrap.grid_rowconfigure(1, weight=1)
        transcript_wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            transcript_wrap, text="CONVERSATION",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))
        self.txt_transcript = ctk.CTkTextbox(
            transcript_wrap, font=("Consolas", 12),
            fg_color=C_SURFACE, text_color=C_TEXT, wrap="word",
            scrollbar_button_color=C_ACCENT
        )
        self.txt_transcript.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        self.txt_transcript.configure(state="disabled")

    # RIGHT PANEL

    def _build_right(self):
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color=C_PANEL)
        frame.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="TWITCH CHAT",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 2))
        self.txt_twitch = ctk.CTkTextbox(
            frame, font=("Consolas", 11),
            fg_color=C_SURFACE, text_color=C_TEXT, wrap="word",
            scrollbar_button_color=C_ACCENT
        )
        self.txt_twitch.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 8))
        self.txt_twitch.configure(state="disabled")

    # STATUS BAR

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color=C_BG)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.status_llm    = _status_pill(bar, "LLM: Loading",    column=0)
        self.status_tts    = _status_pill(bar, "TTS: -",           column=1)
        self.status_twitch = _status_pill(bar, "Twitch: Off",      column=2)
        self.status_vision = _status_pill(bar, "Vision: Off",      column=3)
        self.status_vn     = _status_pill(bar, "VN Agent: Off",    column=4)

    # ACTIONS

    def _toggle_mode(self):
        if self.bot.mode == "companion":
            self.bot.mode = "streamer"
            self.mode_label.configure(text="LIVE STREAMER MODE", text_color=C_ACCENT)
            self.mode_btn.configure(text="Switch to Companion Mode")
        else:
            self.bot.mode = "companion"
            self.mode_label.configure(text="COMPANION MODE", text_color=C_GREEN)
            self.mode_btn.configure(text="Switch to Streamer Mode")
        print(f"   [Dashboard] Mode: {self.bot.mode}")

    def _set_activity(self):
        text = self.activity_entry.get().strip()
        if not text:
            return
        self.bot.current_activity = text
        new_type = self.bot._classify_activity_type(text)
        self.bot.game_mode_controller.activity_type = new_type
        self.bot.vision_agent.activity_type = new_type
        self.bot.immersive = (new_type in (ACTIVITY_VN, ACTIVITY_MEDIA))
        print(f"   [Dashboard] Activity set: '{text}' (type: {new_type})")
        print(f"   [Dashboard] Immersive: {self.bot.immersive}")
        self.activity_display.configure(text=text, text_color=C_TEXT)
        if new_type == ACTIVITY_VN:
            self._activate_observer_and_vn()

    def _set_emotion(self, choice):
        try:
            self.bot.current_emotion = EmotionalState[choice]
        except KeyError:
            pass

    def _activate_observer_and_vn(self):
        """Turns on Observer mode and sets VN activity context. Auto-play is a separate toggle."""
        self.bot.game_mode_controller.activate(ACTIVITY_VN)
        self.bot.vision_agent.activity_type = ACTIVITY_VN
        self.obs_switch.select()
        print("   [Dashboard] VN activity context ON. Auto-play remains a separate toggle.")

    def _toggle_observer(self):
        active = bool(self.obs_switch.get())
        if active:
            act_type = self.bot.game_mode_controller.activity_type
            self.bot.game_mode_controller.activate(act_type)
        else:
            self.bot.game_mode_controller.deactivate()
            if self.vn_switch.get():
                self.vn_switch.deselect()
                self.vn_status_label.configure(text="VN: Standby", text_color=C_MUTED)
        print(f"   [Dashboard] Observer mode: {active}")

    def _toggle_vn(self):
        vn_on = bool(self.vn_switch.get())
        self.bot.vn_autoplay_enabled = vn_on
        if vn_on:
            self.vn_status_label.configure(
                text="VN Auto-Play ACTIVE \u2014 Kira will read and advance the VN. The VN window must be focused.",
                text_color=C_YELLOW,
            )
        else:
            self.vn_status_label.configure(text="VN Auto-Play: Off", text_color=C_MUTED)
        print(f"   [Dashboard] VN Auto-Play: {vn_on}")

    def _invite_kira(self):
        """Cross-thread call: schedule Kira's request_thoughts on the bot's event loop."""
        import asyncio
        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.bot.request_thoughts(),
                self.bot.event_loop,
            )
        else:
            print("   [Dashboard] Bot event loop not ready yet.")

    def _toggle_immersive(self):
        is_on = bool(self.immersive_switch.get())
        self.bot.immersive = is_on
        print(f"   [Dashboard] Immersive Mode toggled: {is_on}")

    def _toggle_bot(self):
        self.bot.is_paused = not self.bot.is_paused
        if self.bot.is_paused:
            self.bot.interruption_event.set()
            self.btn_toggle.configure(text="Resume Bot", fg_color=C_GREEN, hover_color="#3D5C3D")
        else:
            self.bot.interruption_event.clear()
            self.btn_toggle.configure(text="Pause Bot", fg_color=C_RED, hover_color="#7A2E2E")

    # UPDATE LOOPS

    def _update_loop(self):
        try:
            self._refresh_transcript()
            self._refresh_twitch()
            self._refresh_status()
            self._refresh_music()
            self._refresh_emotion()
            self._refresh_activity()
            self._refresh_vn_status()
            self._refresh_immersive()
        except Exception:
            pass
        self.after(500, self._update_loop)

    def _refresh_transcript(self):
        history = self.bot.conversation_history
        if len(history) == self._last_hist_len:
            return
        self._last_hist_len = len(history)
        lines = []
        for turn in history[-20:]:
            name = "Jonny" if turn["role"] == "user" else AI_NAME
            lines.append(f"[{name}]: {turn['content']}\n")
        text = "\n".join(lines)
        self.txt_transcript.configure(state="normal")
        self.txt_transcript.delete("0.0", "end")
        self.txt_transcript.insert("0.0", text)
        self.txt_transcript.see("end")
        self.txt_transcript.configure(state="disabled")

    def _refresh_twitch(self):
        log = self.bot.twitch_log
        if len(log) == self._last_twitch_len:
            return
        self._last_twitch_len = len(log)
        text = "\n".join(log[-60:])
        self.txt_twitch.configure(state="normal")
        self.txt_twitch.delete("0.0", "end")
        self.txt_twitch.insert("0.0", text)
        self.txt_twitch.see("end")
        self.txt_twitch.configure(state="disabled")

    def _refresh_status(self):
        if self.bot.ai_core.is_initialized:
            self.status_llm.configure(text="LLM: Ready", text_color=C_GREEN)
        else:
            self.status_llm.configure(text="LLM: Loading...", text_color=C_YELLOW)
        from config import TTS_ENGINE
        self.status_tts.configure(text=f"TTS: {TTS_ENGINE.upper()}", text_color=C_TEXT)
        if self.bot.game_mode_controller.is_active:
            self.status_vision.configure(text="Vision: ON", text_color=C_GREEN)
            self.vision_desc_label.configure(text=self.bot.vision_agent.last_description)
        else:
            self.status_vision.configure(text="Vision: OFF", text_color=C_MUTED)
        from config import ENABLE_TWITCH_CHAT
        if ENABLE_TWITCH_CHAT:
            self.status_twitch.configure(text="Twitch: ON", text_color=C_GREEN)
        else:
            self.status_twitch.configure(text="Twitch: OFF", text_color=C_MUTED)
        is_vn = (
            self.bot.game_mode_controller.is_active and
            self.bot.game_mode_controller.activity_type == ACTIVITY_VN
        )
        if is_vn:
            self.status_vn.configure(text="VN Agent: ON", text_color=C_YELLOW)
        else:
            self.status_vn.configure(text="VN Agent: OFF", text_color=C_MUTED)

    def _refresh_music(self):
        title = get_now_playing()
        self.music_label.configure(text=f"Now Playing: {title}")

    def _refresh_emotion(self):
        name = self.bot.current_emotion.name
        color = EMOTION_COLORS.get(name, C_TEXT)
        self.emotion_badge.configure(text=name, text_color=color)
        self.emotion_menu.set(name)

    def _refresh_activity(self):
        act = self.bot.current_activity or "None"
        self.activity_display.configure(
            text=act,
            text_color=C_TEXT if self.bot.current_activity else C_MUTED
        )

    def _refresh_vn_status(self):
        if self.bot.vn_autoplay_enabled and not self.vn_switch.get():
            self.vn_switch.select()
        elif not self.bot.vn_autoplay_enabled and self.vn_switch.get():
            self.vn_switch.deselect()

    def _refresh_immersive(self):
        if self.bot.immersive and not self.immersive_switch.get():
            self.immersive_switch.select()
        elif not self.bot.immersive and self.immersive_switch.get():
            self.immersive_switch.deselect()

    def _vision_loop(self):
        if self.bot.game_mode_controller.is_active and not self._vision_lock:
            self._vision_lock = True
            threading.Thread(target=self._do_vision_capture, daemon=True).start()
        self.after(2500, self._vision_loop)

    def _do_vision_capture(self):
        try:
            full = ImageGrab.grab()
            if hasattr(self.bot, "vision_agent"):
                self.bot.vision_agent.update_shared_frame(full)
            preview = full.resize((640, 360))
            tk_img = ctk.CTkImage(light_image=preview, size=(640, 360))
            self.after(0, lambda: self._apply_vision(tk_img))
        except Exception:
            pass
        finally:
            self._vision_lock = False

    def _apply_vision(self, tk_img):
        self._current_image_ref = tk_img
        self.vision_label.configure(image=tk_img, text="")


# Helpers

def _divider(parent):
    ctk.CTkFrame(parent, height=1, fg_color=C_SURFACE).pack(fill="x", padx=12, pady=4)


def _status_pill(parent, text: str, column: int) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=10), text_color=C_MUTED
    )
    lbl.grid(row=0, column=column, padx=8, pady=4)
    return lbl


# Entry Points

def run_async_bot(bot: VTubeBot):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


def run_dashboard():
    bot = VTubeBot()
    threading.Thread(target=run_async_bot, args=(bot,), daemon=True).start()
    app = KiraDashboard(bot)
    app.mainloop()


if __name__ == "__main__":
    run_dashboard()
