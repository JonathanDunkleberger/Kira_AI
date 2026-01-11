import subprocess
import shutil
import os
import yt_dlp

def play_kira_song(query):
    """
    Finds the 1st result on YouTube and streams it directly to mpv player in the background.
    """
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
             # Play it in a hidden background process
             # --no-video makes it a background music player
             # --volume=50 sets initial volume
             # startupinfo to hide window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            print(f"   [Music] Kira is now playing: {title}")
            subprocess.Popen(['mpv', url, '--no-video', '--volume=50'], startupinfo=startupinfo)
        else:
             print(f"   [Music] No URL found for: {query}")

    except Exception as e:
        print(f"   [Music] Error playing song: {e}") 
