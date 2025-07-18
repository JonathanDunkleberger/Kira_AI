# vision.py - Handles screen capture functionality.

import asyncio
from PIL import ImageGrab # Requires Pillow library
import win32gui # Requires pywin32
import win32com.client # Requires pywin32
import base64
import io

def find_window_by_title(title_substring: str):
    """Finds a window handle by a substring of its title."""
    hwnds = []
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and title_substring.lower() in win32gui.GetWindowText(hwnd).lower():
            hwnds.append(hwnd)
        return True
    
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

async def capture_game_window_base64(window_title: str) -> str | None:
    """
    Captures a specific window, resizes it for the model, and returns it as a base64 encoded string.
    """
    loop = asyncio.get_running_loop()
    try:
        # Run blocking GUI operations in a separate thread
        hwnd = await loop.run_in_executor(None, find_window_by_title, window_title)
        
        if not hwnd:
            # print(f"   Vision: Game window '{window_title}' not found.") # Uncomment for debug
            return None

        # Use shell to bring window to front to ensure it's not obscured
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%') # Sends an Alt key press to prevent issues with SendKeys
        # SetForegroundWindow can fail if another window is already active and blocking.
        # It's a best effort.
        win32gui.SetForegroundWindow(hwnd)
        await asyncio.sleep(0.2) # Wait a moment for window to come to front

        bbox = win32gui.GetWindowRect(hwnd)
        
        # Capture the image from the screen region
        # all_screens=True is important if the window is partially off-screen or on another monitor
        img = ImageGrab.grab(bbox, all_screens=True)

        # --- Pre-process image for the vision model ---
        # Resize to a manageable size to save tokens and processing time for LLaVA
        # LLaVA typically prefers images around 224x224 or 336x336,
        # but 1024x1024 as a max dimension is a safe general resize for base64.
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS) # Use LANCZOS for quality resize
        
        # Convert to a byte stream
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG") # Use JPEG for compression
        
        # Encode to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        print(f"   Error capturing game window: {e}")
        return None

# Ensure Pillow's Resampling module is imported if used.
from PIL import Image