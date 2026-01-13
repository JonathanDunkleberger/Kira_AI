# force_fix_vision.py
import os
import shutil
from huggingface_hub import snapshot_download

# This must match what is in your config.py
local_model_path = "models/moondream" 

print(f"--- NUKING OLD MODEL AT {local_model_path} ---")
if os.path.exists(local_model_path):
    shutil.rmtree(local_model_path)
    print("Deleted old corrupted folder.")

os.makedirs(local_model_path, exist_ok=True)

print("--- DOWNLOADING FRESH MOONDREAM ---")
print("This includes the code files (lora.py, configuration_moondream.py) needed to run it.")

snapshot_download(
    repo_id="vikhyatk/moondream2", 
    local_dir=local_model_path,
    local_dir_use_symlinks=False, # IMPORTANT: Downloads actual files, not links
    revision="main"
)

print("\n--- DOWNLOAD COMPLETE ---")
print("Please verify that 'config.py' has this line:")
print(f'VISION_MODEL_PATH = "{local_model_path}"')
