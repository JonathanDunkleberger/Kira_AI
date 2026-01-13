import os
import shutil
from huggingface_hub import snapshot_download

# This must match the path in your config.py
local_model_path = "models/moondream"

print(f"--- NUKING CORRUPTED MODEL AT {local_model_path} ---")
if os.path.exists(local_model_path):
    try:
        shutil.rmtree(local_model_path)
        print("Deleted old folder.")
    except Exception as e:
        print(f"Error deleting folder: {e}. Please delete 'models/moondream' manually.")

# Create fresh directory
os.makedirs(local_model_path, exist_ok=True)

print("--- DOWNLOADING FRESH MOONDREAM BUNDLE ---")
# We download everything, including the python code files
snapshot_download(
    repo_id="vikhyatk/moondream2", 
    local_dir=local_model_path,
    local_dir_use_symlinks=False, 
    revision="main"
)

print("\n--- DOWNLOAD COMPLETE ---")
print("You can now run 'python bot.py' or 'python dashboard.py'")
