# setup_vision.py - Downloads Moondream2 model

from huggingface_hub import snapshot_download
import os

def download_moondream():
    print("-> Downloading Moondream2 model to ./models/moondream...")
    
    model_path = "./models/moondream"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    try:
        snapshot_download(
            repo_id="vikhyatk/moondream2",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print("-> Success! Moondream model downloaded.")
    except Exception as e:
        print(f"-> Error downloading model: {e}")

if __name__ == "__main__":
    download_moondream()
