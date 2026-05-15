import subprocess
import shutil
import os
import yt_dlp
import threading

# Global state
current_player_process = None
current_song_title = "None"
is_playing = False
player_lock = threading.Lock()

def get_now_playing() -> str:
    """Returns the title of the currently playing song."""
    return current_song_title if is_playing else "Nothing"

def pause_song():
    """No-op placeholder: mpv subprocess doesn't support pause via Popen. Use skip instead."""
    pass

def resume_song():
    """No-op placeholder."""
    pass

def skip_song():
    """Immediately kills the current mpv process."""
    global current_player_process, current_song_title, is_playing
    with player_lock:
        if current_player_process:
            print("   [Music] Skipping song...")
            try:
                current_player_process.terminate()
            except Exception as e:
                print(f"   [Music] Error skipping: {e}")
            current_player_process = None
            current_song_title = "None"
            is_playing = False

def clear_queue():
    """Empties the pending song list (Stub)."""
    # Currently we don't maintain a queue list, we just play one song at a time.
    # So skipping the current song is effectively clearing the queue.
    skip_song()

def play_kira_song(query):
    """
    Finds the 1st result on YouTube and streams it directly to mpv player in the background.
    """
    global current_player_process
    if not query: return

    # Check for mpv in path or current directory
    if not shutil.which("mpv") and not os.path.exists("mpv.exe"):
         print("   [Music] Error: mpv not found. Please download mpv.exe to the bot folder.")
         return

    print(f"   [Music] Searching and queueing: {query}...")
    
    try:
        # Configure yt-dlp with the specified options
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            # Let yt-dlp auto-detect deno.exe since it's in the same folder
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'noplaylist': True,
            'default_search': 'ytsearch1',
        }

        url = None
        title = query

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            
            # Extract URL from search results
            if 'entries' in info:
                if len(info['entries']) > 0:
                    video_info = info['entries'][0]
                    url = video_info.get('url')
                    title = video_info.get('title', query)
            else:
                url = info.get('url')
                title = info.get('title', query)
        
        if url:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Stop previous song before starting new one
            skip_song()

            print(f"   [Music] Kira is now playing: {title}")
            with player_lock:
                global current_song_title, is_playing
                current_player_process = subprocess.Popen(
                    ['mpv', url, '--no-video', '--volume=50'], startupinfo=startupinfo
                )
                current_song_title = title
                is_playing = True
        else:
             print(f"   [Music] No URL found for: {query}")

    except Exception as e:
        print(f"   [Music] Error playing song: {e}") 
