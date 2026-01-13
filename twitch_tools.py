import requests
import os
from dotenv import load_dotenv

load_dotenv()

TWITCH_ACCESS_TOKEN = os.getenv("TWITCH_OAUTH_TOKEN", "").replace("oauth:", "")
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "")
# Note: You need to manually fetch your broadcaster ID and put it in .env as BROADCASTER_ID
# or fetch it dynamically using the username. For now, let's assume it's in .env or we fetch it.
BROADCASTER_ID = os.getenv("BROADCASTER_ID", "") 

def start_twitch_poll(question, options):
    if not CLIENT_ID or not TWITCH_ACCESS_TOKEN:
        print("   [Twitch] Error: Missing CLIENT_ID or OAUTH_TOKEN.")
        return

    # If we don't have a broadcaster ID, we might need to fetch it first.
    # For this snippet, we'll try to fetch it if missing, using the username.
    global BROADCASTER_ID
    if not BROADCASTER_ID:
        username = os.getenv("TWITCH_CHANNEL_TO_JOIN", "")
        if username:
             try:
                user_url = f"https://api.twitch.tv/helix/users?login={username}"
                headers = {
                    "Authorization": f"Bearer {TWITCH_ACCESS_TOKEN}",
                    "Client-Id": CLIENT_ID
                }
                user_resp = requests.get(user_url, headers=headers).json()
                if "data" in user_resp and user_resp["data"]:
                    BROADCASTER_ID = user_resp["data"][0]["id"]
             except Exception as e:
                 print(f"   [Twitch] Failed to fetch Broadcaster ID: {e}")

    if not BROADCASTER_ID:
        print("   [Twitch] Error: Could not determine Broadcaster ID.")
        return

    url = "https://api.twitch.tv/helix/polls"
    headers = {
        "Authorization": f"Bearer {TWITCH_ACCESS_TOKEN}",
        "Client-Id": CLIENT_ID,
        "Content-Type": "application/json"
    }
    data = {
        "broadcaster_id": BROADCASTER_ID,
        "title": question[:60], # Twitch Max
        "choices": [{"title": opt[:25]} for opt in options],
        "duration": 60
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"   [Twitch] Successfully started poll: {question}")
        elif response.status_code == 401:
             print(f"   [Twitch] Auth Error (401). Check Token Scopes (channel:manage:polls needed).")
        elif response.status_code == 403:
             print(f"   [Twitch] Polls not available for this account (Permission Denied).")
        else:
            print(f"   [Twitch] Failed to start poll: {response.text}")
    except Exception as e:
        print(f"   [Twitch] Exception starting poll: {e}")
