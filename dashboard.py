# dashboard.py - Kira AI Control Center
import customtkinter as ctk
import threading
import time
from PIL import ImageGrab
import asyncio

try:
    import keyboard as kb_lib
    KEYBOARD_AVAILABLE = True
except ImportError:
    kb_lib = None
    KEYBOARD_AVAILABLE = False

from bot import VTubeBot
from music_tools import skip_song, clear_queue, get_now_playing
from persona import EmotionalState
from game_mode_controller import ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL
from config import AI_NAME
from audio_agent import AUDIO_MODE_OFF, AUDIO_MODE_MEDIA, AUDIO_MODE_MUSIC

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

        self._register_global_hotkeys()
        self.after(500, self._refresh_audio_devices)

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
            frame, text="AUTONOMOUS VN MODE (Experimental)",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED,
        ).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            frame,
            text="Kira reads, paces, and reacts autonomously. Pauses on choices and menus.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230, justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 4))
        self.autopilot_switch = ctk.CTkSwitch(
            frame, text="Autonomous VN Mode",
            command=self._toggle_autopilot,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            font=ctk.CTkFont(size=12),
        )
        self.autopilot_switch.pack(anchor="w", padx=14, pady=(0, 6))

        ctk.CTkLabel(
            frame, text="VN window title (e.g. \"Narcissu\", \"planetarian\"):",
            font=ctk.CTkFont(size=10), text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(0, 2))
        self.vn_window_entry = ctk.CTkEntry(
            frame, placeholder_text="Window title substring",
            font=ctk.CTkFont(size=11), height=28,
        )
        self.vn_window_entry.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkLabel(
            frame, text="Advance key:",
            font=ctk.CTkFont(size=10), text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(0, 2))
        self.autopilot_key_menu = ctk.CTkOptionMenu(
            frame,
            values=["Space", "Enter", "Left Click"],
            command=self._set_autopilot_key,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_PANEL, font=ctk.CTkFont(size=11), height=28,
        )
        self.autopilot_key_menu.set("Space")
        self.autopilot_key_menu.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkLabel(
            frame, text="Read delay — base (s)",
            font=ctk.CTkFont(size=10), text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(0, 0))
        self.autopilot_base_slider = ctk.CTkSlider(
            frame, from_=0.5, to=5.0, number_of_steps=45,
            command=self._update_autopilot_pacing,
            button_color=C_ACCENT, progress_color=C_ACCENT,
        )
        self.autopilot_base_slider.set(2.5)
        self.autopilot_base_slider.pack(fill="x", padx=14, pady=(0, 4))

        ctk.CTkLabel(
            frame, text="Read delay — max (s)",
            font=ctk.CTkFont(size=10), text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(0, 0))
        self.autopilot_max_slider = ctk.CTkSlider(
            frame, from_=3.0, to=12.0, number_of_steps=45,
            command=self._update_autopilot_pacing,
            button_color=C_ACCENT, progress_color=C_ACCENT,
        )
        self.autopilot_max_slider.set(8.0)
        self.autopilot_max_slider.pack(fill="x", padx=14, pady=(0, 6))

        self.autopilot_status_label = ctk.CTkLabel(
            frame, text="Autopilot: OFF",
            font=ctk.CTkFont(size=10), text_color=C_MUTED, wraplength=230,
        )
        self.autopilot_status_label.pack(anchor="w", padx=14, pady=(0, 4))
        self.autopilot_resume_btn = ctk.CTkButton(
            frame, text="Resume after failsafe",
            command=self._resume_autopilot,
            fg_color=C_YELLOW, hover_color=C_ACCENT, text_color=C_BG,
            height=28, font=ctk.CTkFont(size=11), state="disabled",
        )
        self.autopilot_resume_btn.pack(fill="x", padx=12, pady=(0, 10))

        _divider(frame)

        ctk.CTkLabel(frame, text="AUDIO HEARING",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 2))

        self.audio_mode_menu = ctk.CTkOptionMenu(
            frame,
            values=["Off", "Media (game/anime)", "Music (singing/guitar)"],
            command=self._set_audio_mode,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_PANEL, font=ctk.CTkFont(size=11), height=30,
        )
        self.audio_mode_menu.set("Off")
        self.audio_mode_menu.pack(fill="x", padx=12, pady=(0, 4))

        self.audio_status_label = ctk.CTkLabel(
            frame,
            text="Audio: off",
            font=ctk.CTkFont(size=10),
            text_color=C_MUTED,
            wraplength=240,
            justify="left",
            anchor="w",
        )
        self.audio_status_label.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(frame, text="Audio Source Device",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(6, 0))

        self.audio_device_menu = ctk.CTkOptionMenu(
            frame,
            values=["Auto-detect"],
            command=self._set_audio_device,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_PANEL, font=ctk.CTkFont(size=10), height=26,
        )
        self.audio_device_menu.set("Auto-detect")
        self.audio_device_menu.pack(fill="x", padx=12, pady=(0, 4))

        self.btn_refresh_devices = ctk.CTkButton(
            frame, text="🔄 Refresh device list", height=24,
            fg_color=C_SURFACE, hover_color=C_ACCENT, text_color=C_TEXT,
            command=self._refresh_audio_devices, font=ctk.CTkFont(size=10),
        )
        self.btn_refresh_devices.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            frame,
            text="Media = Kira hears game BGM/voice acting/SFX. Music = Kira hears you playing/singing live. Off = no audio capture.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

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

        ctk.CTkLabel(frame, text="LIVE CONTROLS",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        self.btn_interrupt = ctk.CTkButton(
            frame, text="\U0001f6d1  Interrupt  (F8)",
            fg_color=C_RED, hover_color="#7A2E2E", height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._btn_interrupt
        )
        self.btn_interrupt.pack(fill="x", padx=12, pady=(0, 4))

        self.btn_mute = ctk.CTkButton(
            frame, text="\U0001f507  Mute 30s  (F9)",
            fg_color=C_YELLOW, hover_color="#7C5A2C", height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._btn_mute_toggle
        )
        self.btn_mute.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            frame,
            text="Interrupt cuts off her current sentence. Mute also keeps her quiet for 30s \u2014 useful when you\u2019re talking to chat. Hotkeys work even when this window isn\u2019t focused.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="YOUTUBE CHAT",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 2))

        self.yt_entry = ctk.CTkEntry(
            frame, placeholder_text="Paste YouTube live URL or video ID",
            fg_color=C_SURFACE, border_color=C_ACCENT, text_color=C_TEXT,
            placeholder_text_color=C_MUTED, height=32,
        )
        self.yt_entry.pack(fill="x", padx=12, pady=(0, 4))

        yt_row = ctk.CTkFrame(frame, fg_color="transparent")
        yt_row.pack(fill="x", padx=12, pady=(0, 4))
        yt_row.grid_columnconfigure((0, 1), weight=1)

        self.btn_yt_start = ctk.CTkButton(
            yt_row, text="Connect", height=30,
            fg_color=C_GREEN, hover_color="#3D5C3D",
            command=self._yt_start, font=ctk.CTkFont(size=11),
        )
        self.btn_yt_start.grid(row=0, column=0, padx=(0, 2), sticky="ew")

        self.btn_yt_stop = ctk.CTkButton(
            yt_row, text="Disconnect", height=30,
            fg_color=C_RED, hover_color="#7A2E2E",
            command=self._yt_stop, font=ctk.CTkFont(size=11),
        )
        self.btn_yt_stop.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        self.yt_status_label = ctk.CTkLabel(
            frame, text="YouTube: idle",
            font=ctk.CTkFont(size=10), text_color=C_MUTED, wraplength=230,
        )
        self.yt_status_label.pack(anchor="w", padx=14, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="STREAM CONTROL",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        stream_row = ctk.CTkFrame(frame, fg_color="transparent")
        stream_row.pack(fill="x", padx=12, pady=(0, 6))
        stream_row.grid_columnconfigure((0, 1), weight=1)

        self.btn_opener = ctk.CTkButton(
            stream_row, text="🎬  Start", height=36,
            fg_color=C_GREEN, hover_color="#3D5C3D",
            command=self._btn_stream_opener, font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.btn_opener.grid(row=0, column=0, padx=(0, 2), sticky="ew")

        self.btn_closer = ctk.CTkButton(
            stream_row, text="🎬  End", height=36,
            fg_color=C_RED, hover_color="#7A2E2E",
            command=self._btn_stream_closer, font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.btn_closer.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        ctk.CTkLabel(
            frame,
            text="Start fires an episodic opener (recognizes regulars). End fires a closer AND writes session lore + clip candidates to disk.",
            font=ctk.CTkFont(size=9), text_color=C_MUTED, wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        _divider(frame)

        ctk.CTkLabel(frame, text="VIBE METER",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        vibe_row = ctk.CTkFrame(frame, fg_color=C_SURFACE, corner_radius=6)
        vibe_row.pack(fill="x", padx=12, pady=(0, 4))

        self.vibe_chat_rate = ctk.CTkLabel(
            vibe_row, text="0 msg/min",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_chat_rate.pack(pady=(6, 0))
        ctk.CTkLabel(vibe_row, text="chat rate",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 4))

        self.vibe_since_kira = ctk.CTkLabel(
            vibe_row, text="—",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_since_kira.pack(pady=(2, 0))
        ctk.CTkLabel(vibe_row, text="since Kira spoke",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 4))

        self.vibe_chatters = ctk.CTkLabel(
            vibe_row, text="0 chatters",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_chatters.pack(pady=(2, 0))
        ctk.CTkLabel(vibe_row, text="session unique",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 6))

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
        # Load playthrough memory for game/VN activities (or switch game mid-session)
        if new_type in (ACTIVITY_VN, ACTIVITY_GAME) and self.bot.playthrough_memory:
            self.bot.playthrough_memory.load_for_game(text)
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
        """Legacy no-op — VN Auto-Play consolidated into Autonomous VN Mode toggle."""
        pass

    def _toggle_autopilot(self):
        """Enable or disable the Autonomous VN Mode (single master switch)."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            return
        enabled = bool(self.autopilot_switch.get())
        # Push window title from entry field
        title = self.vn_window_entry.get().strip() if hasattr(self, 'vn_window_entry') else ""
        ap.vn_window_title = title
        ap.enabled = enabled
        if enabled:
            self.bot.autopilot_paused_for_input = False
            import asyncio
            if self.bot.event_loop and self.bot.event_loop.is_running():
                self.bot.event_loop.call_soon_threadsafe(ap.start)
        else:
            if self.bot.event_loop and self.bot.event_loop.is_running():
                self.bot.event_loop.call_soon_threadsafe(ap.stop)
        print(f"   [Dashboard] Autopilot: {'ON' if enabled else 'OFF'}")

    def _set_autopilot_key(self, choice: str):
        """Map dropdown choice to the VNInputController advance key."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            return
        key_map = {"Space": "space", "Enter": "enter", "Left Click": "click"}
        ap.input_controller.set_advance_key(key_map.get(choice, "space"))

    def _update_autopilot_pacing(self, _=None):
        """Push slider values to the autopilot pacing config."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            return
        ap.pacing_base = float(self.autopilot_base_slider.get())
        ap.pacing_max = float(self.autopilot_max_slider.get())

    def _resume_autopilot(self):
        """Resume autopilot after Jonny has handled the failsafe screen."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None or not ap.is_paused:
            return
        self.bot.autopilot_paused_for_input = False
        import asyncio
        if self.bot.event_loop and self.bot.event_loop.is_running():
            self.bot.event_loop.call_soon_threadsafe(ap.resume_after_failsafe)
        print("   [Dashboard] Autopilot: resumed after failsafe.")
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

    def _register_global_hotkeys(self):
        """Sets up F8 (interrupt) and F9 (mute toggle) as global hotkeys.
        Works even when the dashboard is not in focus."""
        if not KEYBOARD_AVAILABLE:
            print("   [Dashboard] 'keyboard' package not installed \u2014 global hotkeys disabled.")
            return
        try:
            kb_lib.add_hotkey('f8', self._hotkey_interrupt)
            kb_lib.add_hotkey('f9', self._hotkey_mute_toggle)
            print("   [Dashboard] Global hotkeys registered: F8 = interrupt, F9 = mute toggle (30s)")
        except Exception as e:
            print(f"   [Dashboard] Failed to register hotkeys: {e}")

    def _hotkey_interrupt(self):
        self.bot.interrupt()

    def _hotkey_mute_toggle(self):
        if self.bot.is_muted():
            self.bot.unmute()
        else:
            self.bot.mute_for(30)

    def _btn_interrupt(self):
        self.bot.interrupt()

    def _btn_mute_toggle(self):
        if self.bot.is_muted():
            self.bot.unmute()
        else:
            self.bot.mute_for(30)

    def _btn_stream_opener(self):
        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.bot.run_stream_opener(), self.bot.event_loop)
            print("   [Dashboard] Opener triggered.")
        else:
            print("   [Dashboard] Bot event loop not ready.")

    def _btn_stream_closer(self):
        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.bot.run_stream_closer(), self.bot.event_loop)
            print("   [Dashboard] Closer triggered.")
        else:
            print("   [Dashboard] Bot event loop not ready.")

    def _yt_start(self):
        if not self.bot.youtube_bot:
            self.yt_status_label.configure(text="YouTube: not initialized", text_color=C_RED)
            return
        url = self.yt_entry.get().strip()
        if not url:
            self.yt_status_label.configure(text="YouTube: enter URL or video ID", text_color=C_YELLOW)
            return
        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self._yt_do_start(url), self.bot.event_loop)
        else:
            self.yt_status_label.configure(text="YouTube: bot not ready", text_color=C_RED)

    async def _yt_do_start(self, url: str):
        ok = self.bot.youtube_bot.start(url)
        vid = self.bot.youtube_bot.video_id if ok else None
        text = f"YouTube: live ({vid})" if ok else "YouTube: connect failed"
        color = C_GREEN if ok else C_RED
        self.after(0, lambda: self.yt_status_label.configure(text=text, text_color=color))

    def _yt_stop(self):
        if not self.bot.youtube_bot:
            return
        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self._yt_do_stop(), self.bot.event_loop)

    async def _yt_do_stop(self):
        self.bot.youtube_bot.stop()
        self.after(0, lambda: self.yt_status_label.configure(text="YouTube: idle", text_color=C_MUTED))

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
            self._refresh_mute_status()
            self._refresh_vibe_meter()
            self._refresh_audio_status()
            self._refresh_autopilot_status()
        except Exception:
            pass
        self.after(500, self._update_loop)

    def _refresh_autopilot_status(self):
        """Poll autopilot state every tick and update status label + resume button."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            return
        if ap.is_running:
            self.autopilot_status_label.configure(
                text="Autopilot: RUNNING", text_color=C_GREEN
            )
            self.autopilot_resume_btn.configure(state="disabled")
        elif ap.is_paused:
            reason = ap.pause_reason or "unknown"
            self.autopilot_status_label.configure(
                text=f"Autopilot: PAUSED — {reason}", text_color=C_YELLOW
            )
            self.autopilot_resume_btn.configure(state="normal")
        else:
            self.autopilot_status_label.configure(
                text="Autopilot: OFF", text_color=C_MUTED
            )
            self.autopilot_resume_btn.configure(state="disabled")

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
        """No-op — VN Auto-Play merged into Autonomous VN Mode."""
        pass


    def _refresh_immersive(self):
        if self.bot.immersive and not self.immersive_switch.get():
            self.immersive_switch.select()
        elif not self.bot.immersive and self.immersive_switch.get():
            self.immersive_switch.deselect()

    def _refresh_mute_status(self):
        if self.bot.is_muted():
            remaining = int(self.bot.mute_until - time.time())
            if remaining < 0:
                remaining = 0
            self.btn_mute.configure(text=f"\U0001f507  Muted ({remaining}s) — Unmute", fg_color="#6B5028")
        else:
            self.btn_mute.configure(text="\U0001f507  Mute 30s  (F9)", fg_color=C_YELLOW)

    def _refresh_vibe_meter(self):
        try:
            rate = self.bot.get_chat_rate_per_min()
            if rate >= 10:
                color = C_GREEN
            elif rate >= 3:
                color = C_YELLOW
            else:
                color = C_MUTED
            self.vibe_chat_rate.configure(text=f"{int(rate)} msg/min", text_color=color)

            last_spoke = self.bot.ai_core.last_speech_finish_time
            if last_spoke > 0:
                elapsed = int(time.time() - last_spoke)
                if elapsed < 60:
                    kira_text = f"{elapsed}s ago"
                    kira_color = C_GREEN if elapsed < 20 else C_TEXT
                else:
                    kira_text = f"{elapsed // 60}m {elapsed % 60}s ago"
                    kira_color = C_YELLOW if elapsed < 180 else C_RED
                self.vibe_since_kira.configure(text=kira_text, text_color=kira_color)
            else:
                self.vibe_since_kira.configure(text="—", text_color=C_MUTED)

            count = len(self.bot.session_chatters_seen)
            self.vibe_chatters.configure(text=f"{count} chatter{'s' if count != 1 else ''}", text_color=C_TEXT)
        except Exception:
            pass

    def _refresh_audio_devices(self):
        """Populates the audio device dropdown with available WASAPI loopback devices."""
        if not self.bot.audio_agent:
            return
        devices = self.bot.audio_agent.list_available_loopback_devices()
        if not devices:
            self.audio_device_menu.configure(values=["Auto-detect", "(no devices found)"])
            return
        labels = ["Auto-detect"]
        for name, is_virtual in devices:
            short = name[:40] + ("..." if len(name) > 40 else "")
            if is_virtual:
                short = f"⚠ {short} (virtual)"
            labels.append(short)
        self.audio_device_menu.configure(values=labels)
        print(f"   [Dashboard] Found {len(devices)} loopback device(s)")

    def _set_audio_device(self, label: str):
        """User picked a device from the dropdown. Stores the preference.
        User must toggle the mode dropdown OFF→ON to apply the change."""
        if not self.bot.audio_agent:
            return
        if label == "Auto-detect":
            self.bot.audio_agent.preferred_loopback_name = None
            print("   [Dashboard] Audio device: auto-detect (toggle mode OFF\u2192ON to apply)")
        else:
            cleaned = label.replace("\u26a0 ", "").replace(" (virtual)", "").rstrip(".")
            self.bot.audio_agent.preferred_loopback_name = cleaned
            print(f"   [Dashboard] Audio device set: {cleaned} (toggle mode OFF\u2192ON to apply)")

        if hasattr(self, "audio_status_label"):
            self.audio_status_label.configure(
                text="Device preference saved. Toggle Audio mode OFF then back ON to apply.",
                text_color=C_YELLOW,
            )

    def _set_audio_mode(self, choice: str):
        if not self.bot.audio_agent:
            self.audio_status_label.configure(text="Audio agent disabled in config", text_color=C_RED)
            return
        label_to_mode = {
            "Off": AUDIO_MODE_OFF,
            "Media (game/anime)": AUDIO_MODE_MEDIA,
            "Music (singing/guitar)": AUDIO_MODE_MUSIC,
        }
        mode = label_to_mode.get(choice, AUDIO_MODE_OFF)
        self.bot.audio_agent.set_mode(mode)
        if mode == AUDIO_MODE_OFF:
            self.audio_status_label.configure(text="Audio: off", text_color=C_MUTED)
        elif mode == AUDIO_MODE_MEDIA:
            self.audio_status_label.configure(text="Audio: listening to system audio", text_color=C_GREEN)
        else:
            self.audio_status_label.configure(text="Audio: listening to mic (music mode)", text_color=C_GREEN)

    def _refresh_audio_status(self):
        if not self.bot.audio_agent:
            return
        agent = self.bot.audio_agent
        if not agent.is_active():
            return
        if not agent.audio_summary or agent.audio_summary == "(quiet)":
            self.audio_status_label.configure(
                text=f"🎧 {agent.mode.upper()} \u2014 listening (quiet)",
                text_color=C_MUTED,
            )
            return
        rel = int(time.time() - agent.last_capture_time) if agent.last_capture_time else 0
        summary = agent.audio_summary
        if len(summary) > 300:
            summary = summary[:297] + "..."
        self.audio_status_label.configure(
            text=f"🎧 {rel}s ago:\n{summary}",
            text_color=C_GREEN,
        )

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
