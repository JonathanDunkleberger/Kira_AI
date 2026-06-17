#!/usr/bin/env python
"""Kira launcher — the documented way to start the bot.

Resolves the repo root from this file's location and ``os.chdir()``s there
before launching, so every CWD-relative runtime path (``logs/``, ``clips/``,
``lore/``, ``playthroughs/``, ``persona/private/``) keeps working no matter
where you invoke it from.

Usage:
    python run.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Install full verbatim console capture BEFORE importing kira.bot, so even
# import-time output (the heavy pyaudio / torch / faster-whisper imports — a real
# crash site) is mirrored to logs/debug/latest.log. Fail-graceful: never blocks
# startup. See kira/debug_tee.py.
from kira.debug_tee import install_console_tee
install_console_tee()

from kira.bot import launch

if __name__ == "__main__":
    launch()
