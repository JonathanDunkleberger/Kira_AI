import os
import shutil
from huggingface_hub import snapshot_download

model_path = "models/moondream"

# 1. Delete the corrupted folder
if os.path.exists(model_path):
    print(f"Removing corrupted model at {model_path}...")
    shutil.rmtree(model_path)

# 2. Download ALL files (No symlinks, specific revision)
print("Downloading fresh Moondream bundle...")
snapshot_download(
    repo_id="vikhyatk/moondream2", 
    local_dir=model_path,
    local_dir_use_symlinks=False, 
    revision="main" # Forces the main branch
)

print("Download complete. Please restart dashboard.py")
