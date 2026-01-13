from huggingface_hub import hf_hub_download
import os

# We use the GGUF format from Bartowski (highly reliable)
repo_id = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
filename = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
model_dir = "models"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print(f"--- DOWNLOADING LLAMA 3.1 8B (Q4_K_M) ---")
print(f"Target Size: ~4.9 GB (Perfect for Single PC Streaming)")
try:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print("\nSUCCESS: Model downloaded.")
    print(f"Path: {model_dir}/{filename}")
except Exception as e:
    print(f"\nERROR: {e}")
