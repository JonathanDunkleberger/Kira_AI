# dashboard.py - Kira AI Control Center
import customtkinter as ctk
import threading
import time
import sys
import traceback
from PIL import ImageGrab
import asyncio
import concurrent.futures

# ── Global exception hooks — make ALL crashes print a full traceback ──────────
def _excepthook(exc_type, exc_value, exc_tb):
    print("[CRASH] Unhandled exception in main thread:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    sys.__excepthook__(exc_type, exc_value, exc_tb)

def _thread_excepthook(args):
    print(f"[CRASH] Unhandled exception in thread '{args.thread.name}':", flush=True)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_tb, file=sys.stderr)
    sys.stderr.flush()

sys.excepthook = _excepthook
threading.excepthook = _thread_excepthook
# ─────────────────────────────────────────────────────────────────────────────

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
try:
    from config import LOOPBACK_STT_DEFAULT, GAME_MODE_AUTO_CONFIGURE
except ImportError:
    LOOPBACK_STT_DEFAULT = False
    GAME_MODE_AUTO_CONFIGURE = True

# DPI: must be set before any Tk/CTk call so dropdowns render crisp on HiDPI.
import ctypes as _ctypes
try:
    _ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI v2
except Exception:
    pass

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
ctk.deactivate_automatic_dpi_awareness()

# ── Dark sakura / lavender palette — sourced from theme.py ───────────────────
import theme as T

C_BG      = T.APP_BG        # darkest  — window background
C_PANEL   = T.CARD_BG       # mid      — panel / column container
C_SURFACE = T.CONTROL_BG    # lightest — widget fill, inputs, dropdowns
C_BORDER  = T.BORDER        # hairline border
C_TEXT    = T.TEXT_PRIMARY  # main text
C_MUTED   = T.TEXT_SECONDARY  # secondary / dim labels
C_ACCENT  = T.ACCENT        # lavender — ON states, GO, primary action
C_AMBER   = T.WARNING       # amber — Streamer badge, caution
C_RED     = T.DANGER        # dusty rose — interrupt / stop / exit
C_GOLD    = T.WARNING       # same amber for mute / caution
C_GREEN   = T.SUCCESS       # teal-green — active / healthy status dots
C_YELLOW  = T.WARNING       # alias for caution

EMOTION_COLORS = {
    "HAPPY":       T.ACCENT,
    "SASSY":       T.SAKURA,
    "MOODY":       T.TEXT_SECONDARY,
    "EMOTIONAL":   T.SAKURA_SOFT,
    "HYPERACTIVE": T.WARNING,
}


class KiraDashboard(ctk.CTk):
    def __init__(self, bot: VTubeBot):
        super().__init__()
        self.bot = bot
        self.title(f"{AI_NAME} - Control Center")
        self.geometry("1600x900")
        self.minsize(1280, 780)
        self.configure(fg_color=C_BG)

        self._register_global_hotkeys()
        self.after(500, self._refresh_audio_devices)

        self._vision_lock = False
        self._current_image_ref = None
        self._last_hist_len = 0
        self._last_twitch_len = 0
        self._closing = False
        self._selected_known_slug: str = ""  # set when user PICKS from autocomplete; cleared on freetype
        self._preset_modified: bool = False
        self._active_preset: str = ""

        # 4-column, 4-row grid (header / main / transcript / statusbar)
        # Perception live feed is a compact 22px bar inside the header frame.
        self.grid_columnconfigure(0, weight=1, minsize=260)   # A: State
        self.grid_columnconfigure(1, weight=1, minsize=295)   # B: Perception
        self.grid_columnconfigure(2, weight=1, minsize=340)   # C: Activity
        self.grid_columnconfigure(3, weight=2)                # D: Live Controls (gets extra)
        self.grid_rowconfigure(0, weight=0, minsize=72)       # Header (title + compact sense bar)
        self.grid_rowconfigure(1, weight=1)                   # Main columns
        self.grid_rowconfigure(2, weight=0, minsize=52)       # Transcript strip
        self.grid_rowconfigure(3, weight=0)                   # Status bar

        self._build_header()
        self._build_col_a()
        self._build_col_b()
        self._build_col_c()
        self._build_col_d()
        self._build_transcript()
        self._build_statusbar()

        # CRITICAL: the asyncio loop runs on a daemon thread. Without this
        # handler, closing the window kills that thread mid-await and any
        # pending Opus call (post-stream summary, lore, clips) is silently
        # lost. This handler runs shutdown_async to completion before destroy.
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._update_loop()
        self._vision_loop()

    def _on_window_close(self):
        """Block window close until shutdown_async() finishes (up to 120s).

        Without this, Tk mainloop returns → main thread exits → Python
        starts shutdown → all daemon threads (including the asyncio loop
        running our Opus calls) are killed instantly. Any awaited LLM call
        is murdered and the resulting artifacts never reach disk."""
        if self._closing:
            return
        self._closing = True

        # Best-effort UI feedback before we block the main thread.
        try:
            if hasattr(self, "game_mode_status"):
                self.game_mode_status.configure(
                    text="Saving artifacts… up to 2 min, please wait",
                    text_color=C_MUTED,
                )
            self.update_idletasks()
        except Exception:
            pass

        loop = getattr(self.bot, "event_loop", None)
        if loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.bot.shutdown_async(), loop
                )
                # Hard cap so a hung Opus call can't trap the user forever.
                fut.result(timeout=120)
            except concurrent.futures.TimeoutError:
                print("[Shutdown] shutdown_async exceeded 120s — forcing close.", flush=True)
            except Exception as e:
                print(f"[Shutdown] shutdown_async raised: {e}", flush=True)
                traceback.print_exc()

        try:
            self.destroy()
        except Exception:
            pass

    # LEFT PANEL

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD METHODS — 4-column layout (header · perception · activity · live)
    # No scrolling; everything visible at once.
    # ══════════════════════════════════════════════════════════════════════════

    def _build_header(self):
        """Persistent header.
        Row 0 (inside hdr): title · mode pill · sense on/off badges · activity
        Row 1 (inside hdr): compact 22px live perception bar — what each sense is actually seeing/hearing
        """
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color=C_PANEL)
        hdr.grid(row=0, column=0, columnspan=4, sticky="ew")
        hdr.grid_columnconfigure(2, weight=1)
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr, text=f"[ {AI_NAME.upper()} ]",
            font=ctk.CTkFont(size=17, weight="bold"), text_color=C_ACCENT
        ).grid(row=0, column=0, padx=(16, 8), pady=(10, 2), sticky="w")

        self.mode_pill = ctk.CTkButton(
            hdr, text="\u25cf  COMPANION MODE",
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34, width=192,
            command=self._toggle_companion_streamer,
        )
        self.mode_pill.grid(row=0, column=1, padx=8, pady=(10, 2), sticky="w")

        self.header_perception = ctk.CTkLabel(
            hdr, text="\U0001f441 off  \u00b7  \U0001f442 off  \u00b7  \U0001f3a4 off",
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
        )
        self.header_perception.grid(row=0, column=2, padx=8, pady=(10, 2), sticky="w")

        self.header_activity = ctk.CTkLabel(
            hdr, text="Activity: none",
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
        )
        self.header_activity.grid(row=0, column=3, padx=(8, 16), pady=(10, 2), sticky="e")

        # ── Compact sense bar — row 1 inside header, ~22px tall ───────────────
        # Shows live content from each sense without adding any outer grid row.
        perc = ctk.CTkFrame(hdr, corner_radius=0, fg_color=T.SUBPANEL_BG, height=22)
        perc.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        perc.grid_columnconfigure((0, 1, 2), weight=1)
        perc.grid_propagate(False)

        v_row = ctk.CTkFrame(perc, fg_color="transparent")
        v_row.grid(row=0, column=0, sticky="ew", padx=(12, 4))
        ctk.CTkLabel(v_row, text="\U0001f441", font=ctk.CTkFont(size=10),
                     text_color=T.HEADER).pack(side="left", padx=(0, 3))
        self.strip_vision = ctk.CTkLabel(
            v_row, text="Vision off", font=ctk.CTkFont(size=10),
            text_color=C_MUTED, anchor="w",
        )
        self.strip_vision.pack(side="left")
        ctk.CTkFrame(perc, width=1, height=14, fg_color=T.BORDER).grid(
            row=0, column=0, sticky="nse", pady=4)

        a_row = ctk.CTkFrame(perc, fg_color="transparent")
        a_row.grid(row=0, column=1, sticky="ew", padx=4)
        ctk.CTkLabel(a_row, text="\U0001f442", font=ctk.CTkFont(size=10),
                     text_color=T.HEADER).pack(side="left", padx=(0, 3))
        self.strip_audio = ctk.CTkLabel(
            a_row, text="Audio off", font=ctk.CTkFont(size=10),
            text_color=C_MUTED, anchor="w",
        )
        self.strip_audio.pack(side="left")
        ctk.CTkFrame(perc, width=1, height=14, fg_color=T.BORDER).grid(
            row=0, column=1, sticky="nse", pady=4)

        lb_row = ctk.CTkFrame(perc, fg_color="transparent")
        lb_row.grid(row=0, column=2, sticky="ew", padx=4)
        ctk.CTkLabel(lb_row, text="\U0001f3a4", font=ctk.CTkFont(size=10),
                     text_color=T.HEADER).pack(side="left", padx=(0, 3))
        self.strip_loopback = ctk.CTkLabel(
            lb_row, text="STT off", font=ctk.CTkFont(size=10),
            text_color=C_MUTED, anchor="w",
        )
        self.strip_loopback.pack(side="left")

    # ─── Column A: State ──────────────────────────────────────────────────────

    def _build_col_a(self):
        """Column A: Emotion badge/menu, vibe meter."""
        col = ctk.CTkFrame(self, corner_radius=8, fg_color=C_PANEL,
                           border_width=1, border_color=C_BORDER)
        col.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=6)
        col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            col, text="STATE",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=C_TEXT
        ).pack(anchor="w", padx=16, pady=(14, 6))

        _divider(col)

        ctk.CTkLabel(col, text="EMOTION",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(10, 2))
        self.emotion_badge = ctk.CTkLabel(
            col, text="HAPPY",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=C_ACCENT, fg_color=C_SURFACE,
            corner_radius=6, padx=10, pady=4
        )
        self.emotion_badge.pack(fill="x", padx=14, pady=(0, 4))
        self.emotion_menu = ctk.CTkOptionMenu(
            col, values=[e.name for e in EmotionalState],
            command=self._set_emotion,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_SURFACE, dropdown_text_color=C_TEXT,
            text_color=C_TEXT,
            font=ctk.CTkFont(size=11), height=30
        )
        self.emotion_menu.pack(fill="x", padx=14, pady=(0, 10))

        _divider(col)

        ctk.CTkLabel(col, text="VIBE METER",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(10, 4))
        vibe_card = ctk.CTkFrame(col, fg_color=C_SURFACE, corner_radius=8)
        vibe_card.pack(fill="x", padx=14, pady=(0, 10))

        self.vibe_chat_rate = ctk.CTkLabel(
            vibe_card, text="0 msg/min",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_chat_rate.pack(pady=(8, 0))
        ctk.CTkLabel(vibe_card, text="chat rate",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 4))

        self.vibe_since_kira = ctk.CTkLabel(
            vibe_card, text="\u2014",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_since_kira.pack(pady=(2, 0))
        ctk.CTkLabel(vibe_card, text="since Kira spoke",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 4))

        self.vibe_chatters = ctk.CTkLabel(
            vibe_card, text="0 chatters",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT,
        )
        self.vibe_chatters.pack(pady=(2, 0))
        ctk.CTkLabel(vibe_card, text="session unique",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(0, 8))

    # ─── Column B: Perception ──────────────────────────────────────────────────

    def _build_col_b(self):
        """Column B: Vision, hearing + device selector + live indicator, loopback, passive-watch."""
        col = ctk.CTkScrollableFrame(
            self, corner_radius=0, fg_color=C_PANEL,
            scrollbar_button_color=T.BORDER, scrollbar_button_hover_color=C_ACCENT,
        )
        col.grid(row=1, column=1, sticky="nsew", padx=3, pady=6)
        col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col, text="PERCEPTION",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(12, 4))

        # Vision Active
        self.obs_switch = ctk.CTkSwitch(
            col, text="Vision Active",
            command=self._toggle_observer,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=12),
        )
        if self.bot.game_mode_controller.is_active:
            self.obs_switch.select()
        self.obs_switch.pack(anchor="w", padx=16, pady=(0, 6))

        # Vision thumbnail
        vision_frame = ctk.CTkFrame(col, fg_color=C_SURFACE, corner_radius=8, height=113)
        vision_frame.pack(fill="x", padx=14, pady=(0, 0))
        vision_frame.pack_propagate(False)
        self.vision_label = ctk.CTkLabel(
            vision_frame, text="Vision Offline",
            font=ctk.CTkFont(size=10), text_color=C_MUTED
        )
        self.vision_label.pack(fill="both", expand=True)

        self.vision_desc_label = ctk.CTkLabel(
            col, text="", wraplength=260,
            font=ctk.CTkFont(size=9), text_color=C_MUTED, justify="left", anchor="w"
        )
        self.vision_desc_label.pack(anchor="w", padx=16, pady=(2, 6))

        _divider(col)

        # ── Hearing block: mode + device selector + live indicator ────────────
        ctk.CTkLabel(col, text="HEARING",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(6, 2))

        self.audio_mode_menu = ctk.CTkOptionMenu(
            col,
            values=["Off", "Media (game/anime)", "Music (singing/guitar)"],
            command=self._set_audio_mode,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_SURFACE, dropdown_text_color=C_TEXT,
            text_color=C_TEXT,
            font=ctk.CTkFont(size=11), height=30,
        )
        self.audio_mode_menu.set("Off")
        self.audio_mode_menu.pack(fill="x", padx=14, pady=(0, 2))

        # Audio Source Device — immediately under mode, never scrolls off
        ctk.CTkLabel(col, text="Audio source device",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(4, 0))
        dev_row = ctk.CTkFrame(col, fg_color="transparent")
        dev_row.pack(fill="x", padx=14, pady=(0, 2))
        dev_row.grid_columnconfigure(0, weight=1)
        self.audio_device_menu = ctk.CTkOptionMenu(
            dev_row, values=["Auto-detect"],
            command=self._set_audio_device,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_SURFACE, dropdown_text_color=C_TEXT,
            text_color=C_TEXT,
            font=ctk.CTkFont(size=10), height=28,
        )
        self.audio_device_menu.set("Auto-detect")
        self.audio_device_menu.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_refresh_devices = ctk.CTkButton(
            dev_row, text="\U0001f504", width=28, height=28,
            fg_color=C_SURFACE, hover_color=C_ACCENT, text_color=C_TEXT,
            command=self._refresh_audio_devices, font=ctk.CTkFont(size=11),
        )
        self.btn_refresh_devices.grid(row=0, column=1)

        # Live hearing indicator — summary text + "last heard Xs ago"
        self.audio_status_label = ctk.CTkLabel(
            col, text="Audio: off",
            font=ctk.CTkFont(size=9), text_color=C_MUTED,
            wraplength=260, justify="left", anchor="w"
        )
        self.audio_status_label.pack(fill="x", padx=16, pady=(2, 2))

        # Segment counter: "N segments · last Xs ago" — quick health check
        self.audio_activity_label = ctk.CTkLabel(
            col, text="",
            font=ctk.CTkFont(size=9), text_color=C_MUTED,
            anchor="w", justify="left",
        )
        self.audio_activity_label.pack(anchor="w", padx=16, pady=(0, 6))

        _divider(col)

        # Loopback STT
        lb_row = ctk.CTkFrame(col, fg_color="transparent")
        lb_row.pack(fill="x", padx=16, pady=(6, 2))
        lb_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            lb_row, text="Loopback STT",
            font=ctk.CTkFont(size=12), text_color=C_TEXT, anchor="w"
        ).grid(row=0, column=0, sticky="w")
        self.loopback_switch = ctk.CTkSwitch(
            lb_row, text="",
            command=self._toggle_loopback_stt,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            width=46, height=22,
        )
        self.loopback_switch.grid(row=0, column=1, sticky="e")
        self.loopback_status_label = ctk.CTkLabel(
            col, text="Loopback STT: off",
            font=ctk.CTkFont(size=9), text_color=C_MUTED,
            anchor="w", justify="left", wraplength=260
        )
        self.loopback_status_label.pack(anchor="w", padx=16, pady=(0, 6))

        _divider(col)

        # Passive Watching
        self.immersive_switch = ctk.CTkSwitch(
            col, text="Passive Watching (VN/media)",
            command=self._toggle_immersive,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
        )
        self.immersive_switch.pack(anchor="w", padx=16, pady=(8, 10))

    # ─── Column C: Activity ───────────────────────────────────────────────────

    def _build_col_c(self):
        """Column C: Game name, GO, carry mode, VN Autopilot + Media Watch tabs."""
        col = ctk.CTkFrame(self, corner_radius=8, fg_color=C_PANEL,
                           border_width=1, border_color=C_BORDER)
        col.grid(row=1, column=2, sticky="nsew", padx=3, pady=6)
        col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col, text="ACTIVITY",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=C_TEXT
                     ).pack(anchor="w", padx=16, pady=(14, 6))

        self.game_title_entry = ctk.CTkComboBox(
            col, values=[],
            command=self._on_game_selected,
            fg_color=C_SURFACE, border_color=C_ACCENT, text_color=C_TEXT,
            button_color=C_ACCENT, button_hover_color=T.ACCENT_HOVER,
            dropdown_fg_color=C_SURFACE, dropdown_text_color=C_TEXT,
            font=ctk.CTkFont(size=12), height=34,
        )
        self.game_title_entry.set("")
        self.game_title_entry.pack(fill="x", padx=14, pady=(0, 4))
        self.game_title_entry.bind("<KeyRelease>", lambda _: self._on_game_typed())
        self.after(200, self._refresh_game_suggestions)

        self.go_btn = ctk.CTkButton(
            col, text="\u25b6  GO",
            fg_color=C_ACCENT, hover_color=T.ACCENT_HOVER, height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=T.ON_ACCENT,
            command=self._go,
        )
        self.go_btn.pack(fill="x", padx=14, pady=(0, 4))

        self.activity_display = ctk.CTkLabel(
            col, text="Active: None",
            font=ctk.CTkFont(size=10), text_color=C_MUTED,
            wraplength=320, justify="left"
        )
        self.activity_display.pack(anchor="w", padx=16, pady=(0, 2))

        self.game_mode_status = ctk.CTkLabel(
            col, text="GENERAL mode",
            font=ctk.CTkFont(size=10), text_color=C_MUTED,
            wraplength=320, justify="left"
        )
        self.game_mode_status.pack(anchor="w", padx=16, pady=(0, 2))

        ctk.CTkButton(
            col, text="\u2297  Exit Game Mode", height=26,
            fg_color=C_SURFACE, hover_color=C_RED, text_color=C_MUTED,
            command=self._exit_game_mode, font=ctk.CTkFont(size=10)
        ).pack(fill="x", padx=14, pady=(0, 8))

        _divider(col)

        self.carry_mode_switch = ctk.CTkSwitch(
            col, text="Carry Mode  (gameplay momentum)",
            command=self._toggle_carry_mode,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
        )
        self.carry_mode_switch.pack(anchor="w", padx=16, pady=(8, 8))

        _divider(col)

        # Tabview: VN Autopilot | Media Watch
        tabs = ctk.CTkTabview(
            col,
            fg_color=C_SURFACE,
            segmented_button_fg_color=C_PANEL,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_selected_hover_color=T.ACCENT_HOVER,
            segmented_button_unselected_color=C_PANEL,
            segmented_button_unselected_hover_color=T.BORDER,
            text_color=C_TEXT,
            text_color_disabled=C_MUTED,
        )
        tabs.pack(fill="x", padx=14, pady=(4, 10))
        tabs.add("VN Autopilot")
        tabs.add("Media Watch")

        # ── VN Autopilot tab ──────────────────────────────────────────────────
        vn = tabs.tab("VN Autopilot")

        self.autopilot_switch = ctk.CTkSwitch(
            vn, text="Autonomous VN Mode",
            command=self._toggle_autopilot,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=12),
        )
        self.autopilot_switch.pack(anchor="w", padx=4, pady=(4, 4))

        ctk.CTkLabel(vn, text="VN window title:",
                     font=ctk.CTkFont(size=10), text_color=C_TEXT,
                     ).pack(anchor="w", padx=4, pady=(0, 2))
        vn_entry_row = ctk.CTkFrame(vn, fg_color="transparent")
        vn_entry_row.pack(fill="x", padx=4, pady=(0, 4))
        vn_entry_row.grid_columnconfigure(0, weight=1)
        self.vn_window_entry = ctk.CTkEntry(
            vn_entry_row,
            placeholder_text='e.g. "planetarian", "Narcissu"',
            fg_color=C_SURFACE, border_color=C_BORDER, text_color=C_TEXT,
            placeholder_text_color=C_MUTED,
            font=ctk.CTkFont(size=11), height=26,
        )
        self.vn_window_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(
            vn_entry_row, text="Re-detect",
            command=self._redetect_vn_window,
            fg_color=C_SURFACE, text_color=C_MUTED, hover_color=C_BORDER,
            font=ctk.CTkFont(size=10), height=26, width=72,
        ).grid(row=0, column=1)

        ctk.CTkLabel(vn, text="Advance key:",
                     font=ctk.CTkFont(size=10), text_color=C_TEXT,
                     ).pack(anchor="w", padx=4, pady=(0, 2))
        self.autopilot_key_menu = ctk.CTkOptionMenu(
            vn, values=["Space", "Enter", "Left Click"],
            command=self._set_autopilot_key,
            fg_color=C_SURFACE, button_color=C_ACCENT,
            dropdown_fg_color=C_SURFACE, dropdown_text_color=C_TEXT,
            text_color=C_TEXT,
            font=ctk.CTkFont(size=11), height=26,
        )
        self.autopilot_key_menu.set("Enter")
        self.autopilot_key_menu.pack(fill="x", padx=4, pady=(0, 4))

        # Compact side-by-side read-delay sliders
        slider_pair = ctk.CTkFrame(vn, fg_color="transparent")
        slider_pair.pack(fill="x", padx=4, pady=(0, 4))
        slider_pair.grid_columnconfigure((0, 1), weight=1)

        base_col = ctk.CTkFrame(slider_pair, fg_color="transparent")
        base_col.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkLabel(base_col, text="Base 0.5\u20135s",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(anchor="w")
        self.autopilot_base_slider = ctk.CTkSlider(
            base_col, from_=0.5, to=5.0, number_of_steps=45,
            command=self._update_autopilot_pacing,
            button_color=C_ACCENT, progress_color=C_ACCENT, height=16,
        )
        self.autopilot_base_slider.set(2.5)
        self.autopilot_base_slider.pack(fill="x")

        max_col = ctk.CTkFrame(slider_pair, fg_color="transparent")
        max_col.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(max_col, text="Max 3\u201312s",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(anchor="w")
        self.autopilot_max_slider = ctk.CTkSlider(
            max_col, from_=3.0, to=12.0, number_of_steps=45,
            command=self._update_autopilot_pacing,
            button_color=C_ACCENT, progress_color=C_ACCENT, height=16,
        )
        self.autopilot_max_slider.set(8.0)
        self.autopilot_max_slider.pack(fill="x")

        self.autopilot_status_label = ctk.CTkLabel(
            vn, text="Autopilot: OFF",
            font=ctk.CTkFont(size=10), text_color=C_MUTED, wraplength=270,
        )
        self.autopilot_status_label.pack(anchor="w", padx=4, pady=(2, 2))
        self.autopilot_resume_btn = ctk.CTkButton(
            vn, text="Resume after failsafe",
            command=self._resume_autopilot,
            fg_color=T.FAILSAFE_FG, hover_color=T.WARNING_HOVER, text_color=T.ON_WARNING,
            height=26, font=ctk.CTkFont(size=10), state="disabled",
        )
        self.autopilot_resume_btn.pack(fill="x", padx=4, pady=(0, 4))

        # ── Media Watch tab ───────────────────────────────────────────────────
        mw = tabs.tab("Media Watch")

        self.media_watch_switch = ctk.CTkSwitch(
            mw, text="Media Watch Mode",
            command=self._toggle_media_watch,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=12),
        )
        self.media_watch_switch.pack(anchor="w", padx=4, pady=(4, 4))

        ctk.CTkLabel(mw, text="Video window title:",
                     font=ctk.CTkFont(size=10), text_color=C_TEXT,
                     ).pack(anchor="w", padx=4, pady=(0, 2))
        self.media_watch_window_entry = ctk.CTkEntry(
            mw, placeholder_text='e.g. "VLC", "mpv", "YouTube"',
            fg_color=C_SURFACE, border_color=C_BORDER, text_color=C_TEXT,
            placeholder_text_color=C_MUTED,
            font=ctk.CTkFont(size=11), height=26,
        )
        self.media_watch_window_entry.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkLabel(mw, text="Analysis interval (s)",
                     font=ctk.CTkFont(size=9), text_color=C_MUTED
                     ).pack(anchor="w", padx=4)
        self.media_watch_interval_slider = ctk.CTkSlider(
            mw, from_=10.0, to=45.0, number_of_steps=35,
            command=self._update_media_watch_interval,
            button_color=C_ACCENT, progress_color=C_ACCENT, height=16,
        )
        self.media_watch_interval_slider.set(18.0)
        self.media_watch_interval_slider.pack(fill="x", padx=4, pady=(0, 4))

        self.media_watch_react_switch = ctk.CTkSwitch(
            mw, text="Spontaneous reactions",
            command=self._toggle_media_watch_react,
            button_color=C_ACCENT, progress_color=C_ACCENT,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
        )
        self.media_watch_react_switch.pack(anchor="w", padx=4, pady=(0, 4))
        self.media_watch_react_switch.select()

        self.media_watch_status_label = ctk.CTkLabel(
            mw, text="MediaWatch: OFF",
            font=ctk.CTkFont(size=10), text_color=C_MUTED, wraplength=270,
        )
        self.media_watch_status_label.pack(anchor="w", padx=4, pady=(0, 4))

    # ─── Column D: Live Controls ──────────────────────────────────────────────

    def _build_col_d(self):
        """Column D: Interrupt/mute/pause, stream control, invite, TTS, music."""
        col = ctk.CTkFrame(self, corner_radius=8, fg_color=C_PANEL,
                           border_width=1, border_color=C_BORDER)
        col.grid(row=1, column=3, sticky="nsew", padx=(3, 6), pady=6)
        col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col, text="LIVE CONTROLS",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=C_TEXT
                     ).pack(anchor="w", padx=16, pady=(14, 6))

        # Interrupt + Mute
        live_row = ctk.CTkFrame(col, fg_color="transparent")
        live_row.pack(fill="x", padx=14, pady=(0, 6))
        live_row.grid_columnconfigure((0, 1), weight=1)
        self.btn_interrupt = ctk.CTkButton(
            live_row, text="\U0001f6d1  Interrupt  (F8)",
            fg_color=C_RED, hover_color=T.DANGER_HOVER, height=38,
            font=ctk.CTkFont(size=11, weight="bold"), text_color=T.ON_DANGER,
            command=self._btn_interrupt
        )
        self.btn_interrupt.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.btn_mute = ctk.CTkButton(
            live_row, text="\U0001f507  Mute 60s  (F9)",
            fg_color=C_GOLD, hover_color=T.WARNING_HOVER, height=38,
            font=ctk.CTkFont(size=11, weight="bold"), text_color=T.ON_WARNING,
            command=self._btn_mute_toggle
        )
        self.btn_mute.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        # Pause Model
        self.btn_toggle = ctk.CTkButton(
            col, text="\u23f8  Pause Model",
            fg_color=T.BTN_SECONDARY_FG, hover_color=T.BTN_SECONDARY_HOVER, height=36,
            text_color=T.BTN_SECONDARY_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._toggle_bot
        )
        self.btn_toggle.pack(fill="x", padx=14, pady=(0, 2))
        self.lbl_pause_state = ctk.CTkLabel(
            col, text="Model: ACTIVE",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=C_ACCENT,
        )
        self.lbl_pause_state.pack(anchor="w", padx=16, pady=(0, 6))

        _divider(col)

        # Stream Control
        ctk.CTkLabel(col, text="STREAM CONTROL",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        stream_row = ctk.CTkFrame(col, fg_color="transparent")
        stream_row.pack(fill="x", padx=14, pady=(0, 4))
        stream_row.grid_columnconfigure((0, 1), weight=1)
        self.btn_opener = ctk.CTkButton(
            stream_row, text="\U0001f3ac  Start", height=36,
            fg_color=T.BTN_START_FG, hover_color=T.BTN_START_HOVER, text_color=T.BTN_START_TEXT,
            command=self._btn_stream_opener, font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.btn_opener.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.btn_closer = ctk.CTkButton(
            stream_row, text="\U0001f3ac  End", height=36,
            fg_color=T.BTN_END_FG, hover_color=T.BTN_END_HOVER, text_color=T.BTN_END_TEXT,
            command=self._btn_stream_closer, font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.btn_closer.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        _divider(col)

        # Invite
        ctk.CTkLabel(col, text="INVITE",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        self.btn_invite = ctk.CTkButton(
            col, text="\U0001f4ac  Ask Kira's Thoughts",
            fg_color=T.BTN_PRIMARY_FG, hover_color=T.BTN_PRIMARY_HOVER, height=40,
            font=ctk.CTkFont(size=12, weight="bold"), text_color=T.BTN_PRIMARY_TEXT,
            command=self._invite_kira
        )
        self.btn_invite.pack(fill="x", padx=14, pady=(0, 8))

        _divider(col)

        # TTS
        ctk.CTkLabel(col, text="TTS",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        self.btn_tts_toggle = ctk.CTkButton(
            col, text="TTS: Azure  \u2192  Switch to Fish",
            fg_color=C_SURFACE, hover_color=C_ACCENT, height=30,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
            command=self._toggle_tts_backend,
        )
        self.btn_tts_toggle.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkLabel(col, text="FISH VOICE ID",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(2, 2))
        self.lbl_fish_voice_active = ctk.CTkLabel(
            col, text="Active: (loading...)",
            font=ctk.CTkFont(size=9), text_color=C_TEXT, wraplength=290, justify="left",
        )
        self.lbl_fish_voice_active.pack(anchor="w", padx=16, pady=(0, 2))
        self.fish_voice_entry = ctk.CTkEntry(
            col, placeholder_text="Paste voice model ID here",
            fg_color=C_SURFACE, border_color=C_ACCENT, text_color=C_TEXT,
            placeholder_text_color=C_MUTED, height=28,
        )
        self.fish_voice_entry.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkButton(
            col, text="Apply Voice",
            fg_color=C_SURFACE, hover_color=C_ACCENT, height=28,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
            command=self._apply_fish_voice,
        ).pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkButton(
            col, text="Reload Personality",
            fg_color=C_SURFACE, hover_color=C_ACCENT, height=28,
            text_color=C_TEXT, font=ctk.CTkFont(size=11),
            command=lambda: self.bot.ai_core.reload_personality()
        ).pack(fill="x", padx=14, pady=(0, 8))

        _divider(col)

        # Music Player
        ctk.CTkLabel(col, text="MUSIC PLAYER",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        self.music_label = ctk.CTkLabel(
            col, text="Now Playing: Nothing",
            font=ctk.CTkFont(size=11), wraplength=290, text_color=C_TEXT
        )
        self.music_label.pack(padx=14, pady=(0, 4))
        music_row = ctk.CTkFrame(col, fg_color="transparent")
        music_row.pack(fill="x", padx=14, pady=(0, 10))
        music_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            music_row, text="Skip Song", height=30,
            fg_color=T.BTN_END_FG, hover_color=T.BTN_END_HOVER, text_color=T.BTN_END_TEXT,
            command=lambda: threading.Thread(target=skip_song, daemon=True).start(),
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        ctk.CTkButton(
            music_row, text="Clear Queue", height=30,
            fg_color=C_SURFACE, hover_color=C_BORDER, text_color=C_TEXT,
            command=lambda: threading.Thread(target=clear_queue, daemon=True).start(),
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=1, padx=(2, 0), sticky="ew")

    # ─── Transcript strip ─────────────────────────────────────────────────────

    def _build_transcript(self):
        """2-line dim transcript strip — last 2 turns, read-only."""
        strip = ctk.CTkFrame(self, height=52, corner_radius=0, fg_color=C_BG)
        strip.grid(row=2, column=0, columnspan=4, sticky="ew")
        strip.grid_propagate(False)
        strip.grid_columnconfigure(0, weight=1)
        strip.grid_rowconfigure(0, weight=1)
        self.txt_transcript = ctk.CTkTextbox(
            strip, font=("Consolas", 11),
            fg_color=C_BG, text_color=C_MUTED, wrap="word",
            scrollbar_button_color=C_SURFACE,
            activate_scrollbars=False,
        )
        self.txt_transcript.grid(row=0, column=0, sticky="nsew", padx=6, pady=2)
        self.txt_transcript.configure(state="disabled")

    # ─── Status bar ───────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color=C_BG)
        bar.grid(row=3, column=0, columnspan=4, sticky="ew")
        bar.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        self.status_llm    = _status_pill(bar, "LLM: Loading",   column=0)
        self.status_tts    = _status_pill(bar, "TTS: -",          column=1)
        self.status_twitch = _status_pill(bar, "Twitch: Off",     column=2)
        self.status_vision = _status_pill(bar, "Vision: Off",     column=3)
        self.status_vn     = _status_pill(bar, "VN Agent: Off",   column=4)
        self.status_vram   = _status_pill(bar, "VRAM: \u2014",    column=5)
        self.after(5000, self._refresh_vram)

    # ── GO button + mode defaults ─────────────────────────────────────────────

    # Toggle defaults applied by GO. Written once; toggles stay live after.
    # Activity-type defaults applied by GO (one-shot, not a lock).
    # bot.mode (companion/streamer) is NOT here — that's the pill button's domain.
    _MODE_DEFAULTS: dict = {
        "Just Chatting": dict(
            vision=True,  heartbeat=30.0, hearing="Off",
            loopback=False, immersive=False, carry=False,
            highlights=False,
        ),
        "Action Game": dict(
            vision=True,  heartbeat=10.0, hearing="Media (game/anime)",
            loopback=True, immersive=False, carry=False,
            highlights=True,
        ),
        "VN": dict(
            vision=True,  heartbeat=10.0, hearing="Media (game/anime)",
            loopback=True, immersive=True, carry=False,
            highlights=True,
        ),
        "Companion": dict(
            vision=False, heartbeat=30.0, hearing="Off",
            loopback=False, immersive=False, carry=False,
            highlights=False,
        ),
    }

    def _go(self):
        """GO button: activate activity (optional), apply activity-type defaults.

        Name is not required — GO with empty name switches to Just Chatting defaults.
        Companion/Streamer pill is NOT touched here.
        Stream logger is NOT restarted; an activity_switch marker is dropped instead."""
        name = self.game_title_entry.get().strip()

        # Step 1: activate/switch activity (or clear if no name)
        if name:
            _known = self._selected_known_slug
            self._selected_known_slug = ""   # consume before any early return
            new_type = self.bot.activate_game_mode(name, known_slug=_known)
            print(f"   [GO] Activity: '{name}' -> type: {new_type} | slug: {_known or '(normalized)'}")
        else:
            new_type = ACTIVITY_GENERAL
            self.bot.current_activity = ""
        self._refresh_game_suggestions()

        # Step 2: apply one-shot defaults based on detected activity type.
        # bot.mode (companion/streamer) is NOT in the defaults table.
        mode_key = {
            ACTIVITY_GAME:  "Action Game",
            ACTIVITY_VN:    "VN",
            ACTIVITY_MEDIA: "Just Chatting",
        }.get(new_type, "Just Chatting")
        self._apply_mode_defaults(mode_key)

        # Step 3: VN-specific extra (sets controller to ACTIVITY_VN)
        if new_type == ACTIVITY_VN:
            self._activate_observer_and_vn()

        # Step 4: update status labels
        if name:
            self.activity_display.configure(text=f"Active: {name}", text_color=C_TEXT)
            act_label = {
                ACTIVITY_GAME:  "Action Game",
                ACTIVITY_VN:    "VN Mode",
                ACTIVITY_MEDIA: "Media Watch",
            }.get(new_type, "Active")
            self.game_mode_status.configure(
                text=f"{act_label} — {name}", text_color=C_ACCENT
            )
        else:
            self.activity_display.configure(text="Active: None", text_color=C_MUTED)
            self.game_mode_status.configure(text="GENERAL mode", text_color=C_MUTED)

    def _on_game_selected(self, choice: str):
        """User picked an existing game from the autocomplete dropdown.
        Stores the exact slug — GO uses it directly, bypassing normalization."""
        pm = getattr(self.bot, "playthrough_memory", None)
        if pm is None:
            self._selected_known_slug = ""
            return
        self._selected_known_slug = next(
            (slug for display, slug in pm.get_existing_games() if display == choice),
            "",
        )
        print(f"   [Dashboard] Autocomplete: '{choice}' \u2192 slug '{self._selected_known_slug}'")

    def _on_game_typed(self):
        """User is typing a new name — clear the known-slug override so normalization runs."""
        self._selected_known_slug = ""

    def _refresh_game_suggestions(self):
        """Populate the game name combobox from existing playthrough files.
        Called on startup and after each GO (newly created games appear immediately)."""
        pm = getattr(self.bot, "playthrough_memory", None)
        if pm is None or not hasattr(self, "game_title_entry"):
            return
        values = [display for display, _ in pm.get_existing_games()]
        self.game_title_entry.configure(values=values)

    def _apply_mode_defaults(self, mode_name: str):
        """One-shot defaults write on GO. mode_name is 'Action Game', 'VN', etc.
        bot.mode (companion/streamer) is NOT touched — that's the pill button's domain.
        Runs for every GO so switching VN->Action is always a full clean slate."""
        d = self._MODE_DEFAULTS.get(mode_name, self._MODE_DEFAULTS["Just Chatting"])

        # Vision
        if d["vision"]:
            self.obs_switch.select()
            if not self.bot.game_mode_controller.is_active:
                self.bot.game_mode_controller.activate(
                    self.bot.game_mode_controller.activity_type
                )
            self.bot.vision_agent.heartbeat_interval = d["heartbeat"]
        else:
            self.obs_switch.deselect()
            self.bot.game_mode_controller.deactivate()
            self.bot.vision_agent.heartbeat_interval = 30.0

        # Hearing (before loopback — loopback checks audio_agent.is_active())
        self.audio_mode_menu.set(d["hearing"])
        self._set_audio_mode(d["hearing"])

        # Loopback
        if d["loopback"]:
            if hasattr(self, "loopback_switch"):
                self.loopback_switch.select()
            self._start_loopback_if_needed()
        else:
            if hasattr(self, "loopback_switch"):
                self.loopback_switch.deselect()
            self._stop_loopback_if_running()

        # Passive Watching
        self.bot.immersive = d["immersive"]
        if d["immersive"]:
            self.immersive_switch.select()
        else:
            self.immersive_switch.deselect()

        # Carry Mode
        self.bot.carry_mode = d["carry"]
        if d["carry"]:
            self.carry_mode_switch.select()
        else:
            self.carry_mode_switch.deselect()

        # Highlights
        self.bot.highlight_extraction_enabled = d["highlights"]

        # VN Autopilot: deselect when switching away from VN (clean slate).
        # The VN loop already guards on activity_type == ACTIVITY_VN, but clearing
        # the widget and flag prevents stale visual state after a VN->Action switch.
        if mode_name != "VN" and hasattr(self, "autopilot_switch"):
            self.autopilot_switch.deselect()
            self.bot.vn_autoplay_enabled = False

    def _start_loopback_if_needed(self):
        """Start loopback STT if audio is active and it's not already running."""
        lt = self.bot.loopback_transcriber
        if lt is None or lt.is_running():
            return
        if not self.bot.audio_agent or not self.bot.audio_agent.is_active():
            return
        ai_core_ref = self.bot.ai_core
        speaking_fn = (
            (lambda: bool(getattr(ai_core_ref, "is_speaking", False)))
            if ai_core_ref is not None else None
        )
        def _start():
            ok = lt.start(self.bot.audio_agent, speaking_fn)
            def _ui():
                if not ok and hasattr(self, "loopback_switch"):
                    self.loopback_switch.deselect()
                elif ok and hasattr(self, "loopback_status_label"):
                    self.loopback_status_label.configure(
                        text="Loopback STT: running", text_color=C_GREEN
                    )
            self.after(0, _ui)
        threading.Thread(target=_start, daemon=True, name="LoopbackSTT-go").start()

    def _stop_loopback_if_running(self):
        """Stop loopback STT if running. Non-blocking."""
        lt = self.bot.loopback_transcriber
        if lt is None or not lt.is_running():
            return
        threading.Thread(target=lt.stop, daemon=True, name="LoopbackSTT-go-stop").start()

    def _mark_preset_modified(self):
        """No-op \u2014 preset system replaced by mode defaults. Toggles are always live."""
        pass

    def _apply_preset(self, choice: str = ""):
        """No-op stub \u2014 replaced by mode dropdown + GO button."""
        pass

    def _toggle_loopback_stt(self):
        """Explicit ON/OFF toggle for the Loopback Whisper STT.
        ON: loads distil-large-v3 onto CUDA (~1.5 GB), starts transcription.
        OFF: stops transcription, unloads model, calls torch.cuda.empty_cache().
        Independent of audio mode — you can have audio on but loopback off (AAA-safe)."""
        lt = self.bot.loopback_transcriber
        if lt is None:
            self.loopback_status_label.configure(
                text="Loopback STT: disabled in config (ENABLE_LOOPBACK_TRANSCRIBER=false)",
                text_color=C_RED,
            )
            self.loopback_switch.deselect()
            return

        want_on = bool(self.loopback_switch.get())
        if want_on:
            if not self.bot.audio_agent or not self.bot.audio_agent.is_active():
                self.loopback_status_label.configure(
                    text="Loopback STT: enable Audio Hearing (Media mode) first",
                    text_color=C_YELLOW,
                )
                self.loopback_switch.deselect()
                return
            if lt.is_running():
                self.loopback_status_label.configure(
                    text="Loopback STT: already running", text_color=C_GREEN,
                )
                return
            ai_core_ref = self.bot.ai_core
            speaking_fn = (lambda: bool(getattr(ai_core_ref, "is_speaking", False))) \
                if ai_core_ref is not None else None
            audio_agent = self.bot.audio_agent
            self.loopback_status_label.configure(
                text="Loopback STT: loading model...", text_color=C_YELLOW,
            )
            def _start():
                ok = lt.start(audio_agent, speaking_fn)
                def _ui():
                    if ok:
                        self.loopback_status_label.configure(
                            text="Loopback STT: running", text_color=C_GREEN,
                        )
                    else:
                        self.loopback_status_label.configure(
                            text="Loopback STT: failed to start (check logs)", text_color=C_RED,
                        )
                        self.loopback_switch.deselect()
                self.after(0, _ui)
            threading.Thread(target=_start, daemon=True, name="LoopbackSTT-toggle-on").start()
        else:
            self.loopback_status_label.configure(
                text="Loopback STT: stopping + unloading...", text_color=C_YELLOW,
            )
            def _stop():
                lt.stop()
                def _ui():
                    self.loopback_status_label.configure(
                        text="Loopback STT: off — ~1.5 GB VRAM freed", text_color=C_MUTED,
                    )
                self.after(0, _ui)
            threading.Thread(target=_stop, daemon=True, name="LoopbackSTT-toggle-off").start()
        self._mark_preset_modified()

    def _refresh_vram(self):
        """Poll VRAM every 5s and update the status-bar pill.
        Colors: green < 11 GB, yellow 11-13 GB, red > 13 GB.
        Auto-degrade at >= 14.0 GB reserved: stop loopback STT + slow vision.
        Hard skip vision at >= 14.5 GB reserved."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb  = torch.cuda.memory_reserved()  / (1024 ** 3)
                total_gb     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                text = f"VRAM: {allocated_gb:.1f} / {total_gb:.0f} GB"
                if allocated_gb < 11:
                    color = C_GREEN
                elif allocated_gb < 13:
                    color = C_YELLOW
                else:
                    color = C_RED
                self.status_vram.configure(text=text, text_color=color)

                # ── Auto-degrade tier 1: >= 14.0 GB reserved ─────────────────
                if reserved_gb >= 14.0 and not getattr(self.bot, "_vram_degraded", False):
                    self.bot._vram_degraded = True
                    print(f"[VRAM CRITICAL] {reserved_gb:.1f} GB reserved — auto-degrading. "
                          "Loopback STT off, vision heartbeat → 20s. Manual reset required.")
                    lt = self.bot.loopback_transcriber
                    if lt is not None and lt.is_running():
                        def _stop_lt():
                            lt.stop()
                            def _ui():
                                if hasattr(self, "loopback_switch"):
                                    self.loopback_switch.deselect()
                                if hasattr(self, "loopback_status_label"):
                                    self.loopback_status_label.configure(
                                        text="Loopback STT: OFF — auto-degraded (VRAM)",
                                        text_color=C_RED,
                                    )
                            self.after(0, _ui)
                        threading.Thread(target=_stop_lt, daemon=True,
                                         name="LoopbackSTT-vram-degrade").start()
                    try:
                        old_hb = self.bot.vision_agent.heartbeat_interval
                        self.bot.vision_agent.heartbeat_interval = 20.0
                        self.bot.stream_logger.log(
                            "auto_degrade",
                            trigger="vram_threshold",
                            action="vision_heartbeat_increase",
                            **{"from": round(old_hb, 1), "to": 20.0},
                            reserved_gb=round(reserved_gb, 2),
                        )
                        self.bot.stream_logger.log(
                            "warning",
                            message=f"VRAM {reserved_gb:.1f} GB reserved — auto-degraded (loopback off, heartbeat 20s)",
                        )
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

                # ── Auto-degrade tier 2: >= 14.5 GB reserved ─────────────────
                if reserved_gb >= 14.5:
                    try:
                        self.bot.vision_agent.skip_next_frame = True
                    except Exception:
                        pass
                    print(f"[VRAM CRITICAL] {reserved_gb:.1f} GB reserved — skipping next vision frame.")
                    torch.cuda.empty_cache()
            else:
                self.status_vram.configure(text="VRAM: no CUDA", text_color=C_MUTED)
        except Exception:
            self.status_vram.configure(text="VRAM: —", text_color=C_MUTED)
        self.after(5000, self._refresh_vram)

    # ACTIONS

    # ── Stream Preset helpers ─────────────────────────────────────────────────

    def _mark_preset_modified(self):
        """Called by any individual toggle to flag that the user has deviated from the preset."""
        if not self._preset_modified and hasattr(self, "preset_menu"):
            self._preset_modified = True
            current = self.preset_menu.get()
            base = current.replace("Custom \u2014 ", "")
            self.preset_menu.set(f"Custom \u2014 {base}")
            self.preset_status_label.configure(
                text="Manual override active. Preset is a starting point, not a lock.",
                text_color=C_YELLOW,
            )


    def _toggle_tts_backend(self):
        current = getattr(self.bot.ai_core, "tts_backend", "azure")
        new_backend = "fish" if current == "azure" else "azure"
        self.bot.ai_core.tts_backend = new_backend
        if new_backend == "fish":
            self.btn_tts_toggle.configure(text="TTS: Fish  \u2192  Switch to Azure")
        else:
            self.btn_tts_toggle.configure(text="TTS: Azure  \u2192  Switch to Fish")
        print(f"   [Dashboard] TTS backend: {new_backend}")

    def _apply_fish_voice(self):
        new_id = self.fish_voice_entry.get().strip()
        if not new_id:
            return
        self.bot.ai_core.fish_voice_id = new_id
        self.fish_voice_entry.delete(0, "end")
        short = new_id[:24] + "..." if len(new_id) > 24 else new_id
        self.lbl_fish_voice_active.configure(text=f"Active: {short}")
        print(f"   [Dashboard] Fish voice ID updated: {new_id}")

    def _toggle_mode(self):
        """Shim — delegates to the new pill-button handler."""
        self._toggle_companion_streamer()

    def _toggle_companion_streamer(self):
        """Flip the audience-assumption flag.
        Companion: Kira does NOT assume an audience (default at boot).
        Streamer:  Kira knows she's live, engages chat as co-host.
        This is the ONLY difference. Gates nothing. All controls available in both."""
        if self.bot.mode == "companion":
            self.bot.mode = "streamer"
            if hasattr(self, "mode_pill"):
                self.mode_pill.configure(
                    text="\u25cf  LIVE STREAMER", text_color=C_AMBER
                )
            print("   [Dashboard] Mode: STREAMER (audience ON)")
        else:
            self.bot.mode = "companion"
            if hasattr(self, "mode_pill"):
                self.mode_pill.configure(
                    text="\u25cf  COMPANION MODE", text_color=C_ACCENT
                )
            print("   [Dashboard] Mode: COMPANION (audience OFF)")

    def _set_activity(self):
        """Removed — Field 1 (Set Activity) is gone. Slug comes only from game_title_entry."""
        pass

    def _set_emotion(self, choice):
        try:
            new_state = EmotionalState[choice]
        except KeyError:
            return
        if new_state == self.bot.current_emotion:
            return
        self.bot.current_emotion = new_state
        # Mirror the change into VTube Studio (safe no-op if disabled / VTS offline).
        try:
            self.bot.vts_expressions.fire_and_forget(new_state, loop=self.bot.event_loop)
        except Exception as e:
            print(f"   [Dashboard] VTS expression dispatch suppressed: {e}")

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
        self._mark_preset_modified()
        print(f"   [Dashboard] Observer mode: {active}")

    def _toggle_vn(self):
        """Legacy no-op — VN Auto-Play consolidated into Autonomous VN Mode toggle."""
        pass

    def _redetect_vn_window(self):
        """Run VN window auto-detection and populate the title entry with the result."""
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            print("   [Dashboard] Autopilot not available.")
            return
        import asyncio

        async def _detect_and_update():
            detected = await ap._autodetect_vn_window()
            if detected:
                def _update():
                    self.vn_window_entry.delete(0, "end")
                    self.vn_window_entry.insert(0, detected)
                self.after(0, _update)
                print(f"   [Dashboard] Window detected: '{detected}'")
            else:
                print("   [Dashboard] No VN window auto-detected — type the title manually.")

        if self.bot.event_loop and self.bot.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(_detect_and_update(), self.bot.event_loop)

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
        """Map dropdown choice to the VNInputController advance key.
        Also clears any auto-discovered working method so the new choice is
        actually used next advance (otherwise a previously-locked method
        would keep overriding the configured key).
        """
        ap = getattr(self.bot, 'vn_autopilot', None)
        if ap is None:
            return
        key_map = {"Space": "space", "Enter": "enter", "Left Click": "click"}
        new_key = key_map.get(choice, "enter")
        ap.input_controller.set_advance_key(new_key)
        if getattr(ap, '_working_advance_method', None) is not None:
            print(
                f"   [Dashboard] Clearing locked-in working method "
                f"('{ap._working_advance_method}') so new key '{new_key}' is used."
            )
            ap._working_advance_method = None
        # Also clear the oscillation history so the new key gets a clean slate.
        try:
            ap._recent_advance_hashes.clear()
        except Exception:
            pass
        print(f"   [Dashboard] Autopilot advance key: {new_key}")

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

    # ── Media Watch Mode handlers ─────────────────────────────────────────
    def _toggle_media_watch(self):
        """Enable or disable Media Watch Mode (separate from VN autopilot)."""
        mw = getattr(self.bot, 'media_watch', None)
        if mw is None:
            print("   [Dashboard] MediaWatch not available.")
            return
        enabled = bool(self.media_watch_switch.get())
        title = (
            self.media_watch_window_entry.get().strip()
            if hasattr(self, 'media_watch_window_entry') else ""
        )
        mw.window_title = title
        mw.enabled = enabled
        if hasattr(self, 'media_watch_interval_slider'):
            mw.analysis_interval_s = float(self.media_watch_interval_slider.get())
        if enabled:
            if not title:
                print("   [Dashboard] MediaWatch: enter a window title first.")
                self.media_watch_switch.deselect()
                mw.enabled = False
                return
            if self.bot.event_loop and self.bot.event_loop.is_running():
                self.bot.event_loop.call_soon_threadsafe(mw.start)
        else:
            if self.bot.event_loop and self.bot.event_loop.is_running():
                self.bot.event_loop.call_soon_threadsafe(mw.stop)
        print(f"   [Dashboard] MediaWatch: {'ON' if enabled else 'OFF'}")

    def _update_media_watch_interval(self, _=None):
        """Push slider value to MediaWatch's analysis interval."""
        mw = getattr(self.bot, 'media_watch', None)
        if mw is None:
            return
        mw.analysis_interval_s = float(self.media_watch_interval_slider.get())

    def _toggle_media_watch_react(self):
        """Enable/disable Kira's spontaneous in-character reactions to scenes."""
        mw = getattr(self.bot, 'media_watch', None)
        if mw is None:
            return
        on = bool(self.media_watch_react_switch.get())
        # Bot wires this to its _media_watch_react method on construction.
        # Toggling here flips it between the bot handler and None.
        mw.on_react = self.bot._media_watch_react if on else None
        print(f"   [Dashboard] MediaWatch reactions: {'ON' if on else 'OFF'}")

    def _refresh_media_watch_status(self):
        """Poll MediaWatch state every tick and update status label."""
        mw = getattr(self.bot, 'media_watch', None)
        if mw is None or not hasattr(self, 'media_watch_status_label'):
            return
        if mw.is_running:
            self.media_watch_status_label.configure(
                text=mw.get_status_str(), text_color=C_GREEN
            )
        else:
            self.media_watch_status_label.configure(
                text=mw.get_status_str(), text_color=C_MUTED
            )

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
        self._mark_preset_modified()
        print(f"   [Dashboard] Passive Watching Mode toggled: {is_on}")

    def _toggle_carry_mode(self):
        is_on = bool(self.carry_mode_switch.get())
        self.bot.carry_mode = is_on
        self._mark_preset_modified()
        print(f"   [Dashboard] Carry Mode toggled: {is_on} "
              f"(thresholds {'30s/60s' if is_on else '45s/90s'}, "
              f"ask_chat_p {'0.25' if is_on else '0.15'})")

    def _activate_game_mode(self):
        """Legacy stub — replaced by _go()."""
        self._go()

    def _activate_game_mode_inner(self):
        """Legacy stub — replaced by _go()."""
        self._go()

    def _exit_game_mode(self):
        """Dashboard 'Exit Game Mode' button handler.
        Writes session log (if applicable) then resets to GENERAL via async helper."""
        if not (self.bot.event_loop and self.bot.event_loop.is_running()):
            print("   [MANUAL MODE] Bot event loop not ready — cannot exit cleanly.")
            return
        # Fire-and-forget the async exit (writes session log, resets state)
        asyncio.run_coroutine_threadsafe(
            self.bot.deactivate_game_mode_async(),
            self.bot.event_loop,
        )
        # Sync UI immediately — async call runs in background (Opus may take ~30s)
        self.activity_display.configure(text="Active: None", text_color=C_MUTED)
        self.immersive_switch.deselect()
        self.obs_switch.deselect()
        self.audio_mode_menu.set("Off")
        if hasattr(self, 'loopback_switch'):
            self.loopback_switch.deselect()
        self.game_mode_status.configure(
            text="GENERAL mode — writing clips & session log… (may take ~30s)",
            text_color=C_MUTED,
        )
        # Clear the notice after Opus has had time to finish
        self.after(35000, lambda: self.game_mode_status.configure(
            text="GENERAL mode", text_color=C_MUTED
        ))

    def _toggle_bot(self):
        if self.bot.is_paused:
            self.bot.resume_model()
            self.btn_toggle.configure(text="\u23f8  Pause Model", fg_color=C_RED, hover_color="#7A2E2E")
            self.lbl_pause_state.configure(text="Model: ACTIVE", text_color=C_GREEN)
        else:
            self.bot.pause_model()
            self.btn_toggle.configure(text="\u25b6  Resume Model", fg_color=C_GREEN, hover_color="#3D5C3D")
            self.lbl_pause_state.configure(text="Model: PAUSED \u2014 all responses suppressed", text_color=C_RED)

    def _register_global_hotkeys(self):
        """Sets up F8 (interrupt) and F9 (mute toggle) as global hotkeys.
        Works even when the dashboard is not in focus."""
        if not KEYBOARD_AVAILABLE:
            print("   [Dashboard] 'keyboard' package not installed \u2014 global hotkeys disabled.")
            return
        try:
            kb_lib.add_hotkey('f8', self._hotkey_interrupt)
            kb_lib.add_hotkey('f9', self._hotkey_mute_toggle)
            print("   [Dashboard] Global hotkeys registered: F8 = interrupt, F9 = mute toggle (60s)")
        except Exception as e:
            print(f"   [Dashboard] Failed to register hotkeys: {e}")

    def _hotkey_interrupt(self):
        self.bot.interrupt()

    def _hotkey_mute_toggle(self):
        # Don't let the timed mute interfere with an indefinite Pause Model state.
        if self.bot.is_paused:
            print("   [Mute] Ignored \u2014 Pause Model is active (use Resume Model to release).")
            return
        if self.bot.is_muted():
            self.bot.unmute()
        else:
            self.bot.mute_for(60)

    def _btn_interrupt(self):
        self.bot.interrupt()

    def _btn_mute_toggle(self):
        if self.bot.is_paused:
            print("   [Mute] Ignored \u2014 Pause Model is active (use Resume Model to release).")
            return
        if self.bot.is_muted():
            self.bot.unmute()
        else:
            self.bot.mute_for(60)

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
        """No-op \u2014 YouTube connect button removed."""
        pass

    async def _yt_do_start(self, url: str):
        pass

    def _yt_stop(self):
        """No-op \u2014 YouTube disconnect button removed."""
        pass

    async def _yt_do_stop(self):
        pass

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
            self._refresh_loopback_transcript()
            self._refresh_autopilot_status()
            self._refresh_media_watch_status()
            self._refresh_header()
            self._refresh_perception_strip()
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
        """Write last 2 turns to the dim 2-line transcript strip."""
        history = self.bot.conversation_history
        if len(history) == self._last_hist_len:
            return
        self._last_hist_len = len(history)
        parts = []
        for turn in history[-2:]:
            name = "Jonny" if turn["role"] == "user" else AI_NAME
            content = (turn["content"] or "")[:130].replace("\n", " ")
            parts.append(f"[{name}]: {content}")
        text = "   \u00b7   ".join(parts)
        try:
            self.txt_transcript.configure(state="normal")
            self.txt_transcript.delete("0.0", "end")
            self.txt_transcript.insert("0.0", text)
            self.txt_transcript.configure(state="disabled")
        except Exception:
            pass

    def _refresh_twitch(self):
        """No-op \u2014 Twitch chat panel removed; handled by Streamlabs."""
        pass

    def _refresh_status(self):
        if self.bot.ai_core.is_initialized:
            self.status_llm.configure(text="LLM: Ready", text_color=C_GREEN)
        else:
            self.status_llm.configure(text="LLM: Loading...", text_color=C_YELLOW)
        from config import TTS_ENGINE
        backend = getattr(self.bot.ai_core, "tts_backend", TTS_ENGINE)
        self.status_tts.configure(text=f"TTS: {backend.upper()}", text_color=C_TEXT)
        # Keep the toggle button label in sync
        if hasattr(self, "btn_tts_toggle"):
            if backend == "fish":
                self.btn_tts_toggle.configure(text="TTS: Fish  \u2192  Switch to Azure")
            else:
                self.btn_tts_toggle.configure(text="TTS: Azure  \u2192  Switch to Fish")
        # Keep the active voice label in sync
        if hasattr(self, "lbl_fish_voice_active"):
            vid = getattr(self.bot.ai_core, "fish_voice_id", "") or ""
            short = vid[:24] + "..." if len(vid) > 24 else (vid or "(none set)")
            self.lbl_fish_voice_active.configure(text=f"Active: {short}")
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
            text=f"Active: {act}",
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
            self.btn_mute.configure(text=f"\U0001f507  Muted ({remaining}s) — Unmute", fg_color="#7A6A30")
        else:
            self.btn_mute.configure(text="\U0001f507  Mute 60s  (F9)", fg_color=C_GOLD)

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
        """User picked a device from the dropdown. Stores the preference AND
        immediately re-applies the current audio mode so the new device takes
        effect without requiring an OFF→ON toggle cycle."""
        if not self.bot.audio_agent:
            return
        if label == "Auto-detect":
            self.bot.audio_agent.preferred_loopback_name = None
            print("   [Dashboard] Audio device: auto-detect")
        else:
            cleaned = label.replace("\u26a0 ", "").replace(" (virtual)", "").rstrip(".")
            self.bot.audio_agent.preferred_loopback_name = cleaned
            print(f"   [Dashboard] Audio device set: {cleaned}")

        # Auto-reapply the current mode so device change is live immediately
        current_choice = self.audio_mode_menu.get()
        if current_choice != "Off":
            self._set_audio_mode(current_choice)
        elif hasattr(self, "audio_status_label"):
            self.audio_status_label.configure(
                text="Device saved. Select an Audio mode above to activate.",
                text_color=C_YELLOW,
            )
        self._mark_preset_modified()

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

        # Loopback Whisper transcriber — only auto-start if LOOPBACK_STT_DEFAULT is true
        # (default: false, so AAA gaming sessions start without the ~1.5 GB model in VRAM).
        # The explicit dashboard toggle (_toggle_loopback_stt) is the primary control.
        lt = self.bot.loopback_transcriber
        if lt is not None:
            if mode == AUDIO_MODE_MEDIA and LOOPBACK_STT_DEFAULT:
                if not lt.is_running():
                    ai_core_ref = self.bot.ai_core
                    speaking_fn = (lambda: bool(getattr(ai_core_ref, "is_speaking", False))) \
                        if ai_core_ref is not None else None
                    threading.Thread(
                        target=lambda: lt.start(self.bot.audio_agent, speaking_fn),
                        daemon=True,
                        name="LoopbackSTT-bootstrap",
                    ).start()
            elif mode != AUDIO_MODE_MEDIA:
                # Audio hearing turned off — stop and unload loopback STT too
                if lt.is_running():
                    threading.Thread(target=lt.stop, daemon=True).start()
                    if hasattr(self, "loopback_switch"):
                        self.loopback_switch.deselect()
                    if hasattr(self, "loopback_status_label"):
                        self.loopback_status_label.configure(
                            text="Loopback STT: off (audio off)", text_color=C_MUTED
                        )

        if mode == AUDIO_MODE_OFF:
            self.audio_status_label.configure(text="Audio: off", text_color=C_MUTED)
        elif mode == AUDIO_MODE_MEDIA:
            self.audio_status_label.configure(text="Audio: listening to system audio", text_color=C_GREEN)
        else:
            self.audio_status_label.configure(text="Audio: listening to mic (music mode)", text_color=C_GREEN)
        self._mark_preset_modified()

    def _refresh_audio_status(self):
        if not self.bot.audio_agent:
            return
        agent = self.bot.audio_agent
        if not agent.is_active():
            # Clear activity indicator when off
            if hasattr(self, "audio_activity_label"):
                self.audio_activity_label.configure(text="")
            return

        # ── Summary text (audio_status_label) ────────────────────────────────
        if not agent.audio_summary or agent.audio_summary == "(quiet)":
            self.audio_status_label.configure(
                text=f"\U0001f3a7 {agent.mode.upper()} \u2014 listening (quiet)",
                text_color=C_MUTED,
            )
        else:
            summary = agent.audio_summary
            if len(summary) > 160:
                summary = summary[:157] + "..."
            self.audio_status_label.configure(
                text=f"\U0001f3a7 {summary}",
                text_color=C_GREEN,
            )

        # ── Live activity indicator (audio_activity_label) ───────────────────
        # Shows: "last heard Xs ago · N captures this session"
        # This is the health check — if last_capture_time is very stale or
        # segment count is 0 after a minute, something is wrong.
        if hasattr(self, "audio_activity_label"):
            last_ts = getattr(agent, "last_capture_time", 0) or 0
            captures = getattr(agent, "capture_count", None)
            if last_ts:
                age = int(time.time() - last_ts)
                if age < 5:
                    age_str = "just now"
                elif age < 60:
                    age_str = f"{age}s ago"
                elif age < 3600:
                    age_str = f"{age // 60}m ago"
                else:
                    age_str = f"{age // 3600}h ago"
                if age > 45:
                    age_color = T.WARNING
                elif age > 20:
                    age_color = T.TEXT_SECONDARY
                else:
                    age_color = C_GREEN
                count_str = f" \u00b7 {captures} captures" if captures is not None else ""
                self.audio_activity_label.configure(
                    text=f"Last heard: {age_str}{count_str}",
                    text_color=age_color,
                )
            else:
                self.audio_activity_label.configure(
                    text="Waiting for first audio capture…",
                    text_color=C_MUTED,
                )

    def _refresh_loopback_transcript(self):
        """Update loopback status label. Transcript textbox removed in new layout."""
        lt = getattr(self.bot, "loopback_transcriber", None)
        if lt is None or not hasattr(self, "loopback_status_label"):
            return
        status = lt.get_status_summary()
        color = C_ACCENT if lt.is_running() else C_MUTED
        self.loopback_status_label.configure(text=status, text_color=color)

    def _refresh_perception_strip(self):
        """Update the live perception strip once per 500ms tick."""
        if not hasattr(self, "strip_vision"):
            return

        # ── Vision ────────────────────────────────────────────────────────────
        if self.bot.game_mode_controller.is_active:
            va = self.bot.vision_agent
            desc = (getattr(va, "scene_summary", "") or getattr(va, "last_description", "")).strip()
            if desc:
                # One line, max ~80 chars
                one_line = desc.replace("\n", " ")
                if len(one_line) > 82:
                    one_line = one_line[:79] + "…"
                ts = getattr(va, "last_capture_time", 0) or 0
                age = int(time.time() - ts) if ts else 0
                age_str = f"  ({age}s)" if age > 0 else ""
                self.strip_vision.configure(text=one_line + age_str, text_color=T.TEXT_PRIMARY)
            else:
                self.strip_vision.configure(text="Waiting for first frame…", text_color=C_MUTED)
        else:
            self.strip_vision.configure(text="Vision off", text_color=C_MUTED)

        # ── Audio ─────────────────────────────────────────────────────────────
        agent = self.bot.audio_agent
        if agent and agent.is_active():
            summary = (getattr(agent, "audio_summary", "") or "").strip()
            last_ts = getattr(agent, "last_capture_time", 0) or 0
            count = getattr(agent, "capture_count", 0)
            if last_ts:
                age = int(time.time() - last_ts)
                age_str = "just now" if age < 5 else f"{age}s ago"
                color = C_GREEN if age < 25 else (T.WARNING if age < 60 else T.DANGER)
            else:
                age_str = "waiting…"
                color = C_MUTED
            if summary and summary != "(quiet)":
                one = summary.replace("\n", " ")
                if len(one) > 72:
                    one = one[:69] + "…"
                self.strip_audio.configure(
                    text=f"{one}  ·  {age_str} · {count}×", text_color=color
                )
            else:
                self.strip_audio.configure(
                    text=f"Quiet  ·  last {age_str} · {count}×", text_color=C_MUTED
                )
        else:
            self.strip_audio.configure(text="Audio off", text_color=C_MUTED)

        # ── Loopback STT ──────────────────────────────────────────────────────
        lt = getattr(self.bot, "loopback_transcriber", None)
        if lt and lt.is_running():
            status = lt.get_status_summary()
            # Trim the full summary to one line
            one = status.replace("\n", "  ").strip()
            if len(one) > 80:
                one = one[:77] + "…"
            self.strip_loopback.configure(text=one, text_color=C_ACCENT)
        else:
            self.strip_loopback.configure(text="STT off", text_color=C_MUTED)

    def _refresh_header(self):
        """Sync header perception badges and mode pill with current bot state."""
        if not hasattr(self, "header_perception"):
            return
        # Mode pill
        if hasattr(self, "mode_pill"):
            if self.bot.mode == "streamer":
                self.mode_pill.configure(text="\u25cf  LIVE STREAMER", text_color=C_AMBER)
            else:
                self.mode_pill.configure(text="\u25cf  COMPANION MODE", text_color=C_ACCENT)
        # Perception badges
        vis = "\U0001f441 ON" if self.bot.game_mode_controller.is_active else "\U0001f441 off"
        aud_mode = self.audio_mode_menu.get() if hasattr(self, "audio_mode_menu") else "Off"
        if aud_mode == "Off":
            aud = "\U0001f442 off"
        elif "Music" in aud_mode:
            aud = "\U0001f442 Music"
        else:
            aud = "\U0001f442 Media"
        lt = getattr(self.bot, "loopback_transcriber", None)
        loop = "\U0001f3a4 ON" if (lt and lt.is_running()) else "\U0001f3a4 off"
        self.header_perception.configure(
            text=f"{vis}  \u00b7  {aud}  \u00b7  {loop}",
            text_color=C_TEXT if self.bot.game_mode_controller.is_active else C_MUTED,
        )
        # Activity in header
        act = self.bot.current_activity or "none"
        self.header_activity.configure(
            text=f"Activity: {act}",
            text_color=C_TEXT if self.bot.current_activity else C_MUTED,
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
            # Thumbnail for the left-panel Zone A widget (240×135)
            preview = full.resize((240, 135))
            tk_img = ctk.CTkImage(light_image=preview, size=(240, 135))
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
    ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=4)


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

    # Surface any exception that bubbles up to the loop (e.g. unawaited task
    # crashes, callback errors). Without this, asyncio swallows them at GC time.
    def _loop_exc_handler(_loop, ctx):
        msg = ctx.get("message") or "asyncio loop exception"
        exc = ctx.get("exception")
        print(f"[ASYNCIO LOOP EXCEPTION] {msg}", file=sys.stderr, flush=True)
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    loop.set_exception_handler(_loop_exc_handler)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        print("[CRASH] run_async_bot raised:", flush=True)
        traceback.print_exc()
    finally:
        loop.close()


def run_dashboard():
    bot = VTubeBot()
    threading.Thread(target=run_async_bot, args=(bot,), daemon=True).start()
    app = KiraDashboard(bot)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        # Ctrl+C on the terminal bypasses _on_window_close entirely (the main thread
        # is in Tkinter C code, so the WM_DELETE_WINDOW handler never fires).
        # Mirror that handler here: call shutdown_async on the asyncio loop so
        # lore / clips / playthrough are written before the process dies.
        print("\n[Shutdown] Ctrl+C detected — running graceful shutdown (up to 120s)...", flush=True)
        loop = getattr(bot, "event_loop", None)
        if loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(bot.shutdown_async(), loop)
                fut.result(timeout=120)
            except concurrent.futures.TimeoutError:
                print("[Shutdown] Graceful shutdown exceeded 120s — forcing exit.", flush=True)
            except Exception as e:
                print(f"[Shutdown] Shutdown error: {e}", flush=True)
                traceback.print_exc()
        try:
            app.destroy()
        except Exception:
            pass
    except Exception:
        print("[CRASH] app.mainloop() raised:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    run_dashboard()
