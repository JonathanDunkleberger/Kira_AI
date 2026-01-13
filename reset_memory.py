import shutil
import os
import time

if os.path.exists("memory_db"):
    try:
        shutil.rmtree("memory_db")
        print("✅ Memory Wiped for Upgrade.")
    except Exception as e:
        print(f"❌ Error wiping memory: {e}")
        # Sometimes Windows locks files. Retry after a moment.
        time.sleep(1)
        try:
            shutil.rmtree("memory_db")
            print("✅ Memory Wiped for Upgrade (Retry SUCCESS).")
        except:
             print("❌ FATAL: Could not delete memory_db. Ensure the bot is NOT running.")
else:
    print("✨ Memory already clean.")