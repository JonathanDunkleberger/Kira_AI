"""run_panel.py — a SELF-CONTAINED CustomTkinter mini-dashboard for the Pokémon run.

Minimal by design: a WORKSHOP|SHOW switch, the current mode at a glance, and a Go button that
respects it. It does NOT touch core Kira (no import of bot/_pokemon_react/voice/etc.) — it just
shells out to `play_live.py` with the right flags, so it's firewall-safe and disposable.

  WORKSHOP (default, our 90%): play_live.py --go     (resume-from-any in states/workshop/)
  SHOW (the sacred run):       play_live.py --show    (resume Kira's states/kira/ save, or fresh)
       + "Fresh Kira run" box:  play_live.py --show --fresh-kira  (archive old save, start new)

Run:  .venv\\Scripts\\python.exe pokemon_agent\\run_panel.py
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = sys.executable
_PLAY = os.path.join(_HERE, "play_live.py")


def _launch(mode, fresh, audio):
    args = [_PY, "-u", _PLAY, "--show" if mode == "SHOW" else "--go"]
    if mode == "SHOW" and fresh:
        args.append("--fresh-kira")
    if audio:
        args.append("--audio")
    subprocess.Popen(args, cwd=_HERE)


def main():
    try:
        import customtkinter as ctk
    except Exception as e:
        print(f"customtkinter not available ({e}). Use the CLI instead:")
        print("  WORKSHOP:  play_live.py --go")
        print("  SHOW:      play_live.py --show   (add --fresh-kira to start a new Kira run)")
        return

    ctk.set_appearance_mode("dark")
    app = ctk.CTk()
    app.title("Kira × Pokémon — Run Control")
    app.geometry("380x300")

    mode = {"v": "WORKSHOP"}
    ctk.CTkLabel(app, text="Run mode", font=("", 14)).pack(pady=(18, 4))
    mode_lbl = ctk.CTkLabel(app, text="WORKSHOP", font=("", 26, "bold"), text_color="#5bc0de")
    mode_lbl.pack(pady=2)
    sub = ctk.CTkLabel(app, text="resume-from-any · never writes states/kira/", text_color="gray")
    sub.pack(pady=(0, 8))

    fresh_var = ctk.BooleanVar(value=False)
    fresh_box = ctk.CTkCheckBox(app, text="Fresh Kira run (archive old save)", variable=fresh_var)

    def refresh():
        is_show = mode["v"] == "SHOW"
        mode_lbl.configure(text=mode["v"], text_color="#e8a13a" if is_show else "#5bc0de")
        sub.configure(text="canonical · banks to states/kira/ · zero skips" if is_show
                      else "resume-from-any · never writes states/kira/")
        (fresh_box.pack(pady=4) if is_show else fresh_box.pack_forget())

    def toggle():
        mode["v"] = "SHOW" if mode["v"] == "WORKSHOP" else "WORKSHOP"
        refresh()

    ctk.CTkSwitch(app, text="SHOW mode", command=toggle).pack(pady=6)
    audio_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(app, text="Game audio", variable=audio_var).pack(pady=4)
    ctk.CTkButton(app, text="▶  Go", height=42, font=("", 18, "bold"),
                  command=lambda: _launch(mode["v"], fresh_var.get(), audio_var.get())).pack(pady=14)
    refresh()
    app.mainloop()


if __name__ == "__main__":
    main()
