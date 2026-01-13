import os
import shutil

# List of files/folders that are NOW OBSOLETE
garbage_list = [
    "vision_agent.py",
    "undertale_bridge.py",
    "persona_backup.py",       # You mentioned this, safe to delete
    "persona_backup.txt",      # Just in case
    "setup_vision.py",
    "repair_vision.py",
    "force_fix_vision.py",
    "nuclear_fix.py",
    "check_gpu.py",            # Job done, delete to clean up
    "fix_gpu.ps1",             # Job done
    "force_gpu.ps1",           # Job done
    "debug_imports.py",
    "models/moondream",        # Folder
    "models/vision_llava"      # Folder
]

print("--- STARTING PRODUCTION CLEANUP ---")

for item in garbage_list:
    if os.path.exists(item):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"✅ Deleted Folder: {item}")
            else:
                os.remove(item)
                print(f"✅ Deleted File:   {item}")
        except Exception as e:
            print(f"❌ Error deleting {item}: {e}")
    else:
        print(f"✨ Already Clean:   {item}")

print("\nCleanup Complete. Your codebase is now Production Grade.")