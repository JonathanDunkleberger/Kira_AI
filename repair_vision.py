# repair_vision.py
import os
import shutil
from huggingface_hub import snapshot_download

# Define the local path
model_path = "./models/moondream"

# 1. Nuke the old folder if it exists (Clean Install)
if os.path.exists(model_path):
    print(f"Removing corrupted model at {model_path}...")
    shutil.rmtree(model_path)

# 2. Download fresh
print("Downloading Moondream (vikhyatk/moondream2)... this may take a minute.")
snapshot_download(
    repo_id="vikhyatk/moondream2", 
    local_dir=model_path,
    local_dir_use_symlinks=False,  # Force real files, not symlinks
    revision="main"
)

print("Download complete! You can now restart dashboard.py.")
