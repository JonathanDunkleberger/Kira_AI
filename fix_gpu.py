import os
import zipfile
import urllib.request
import shutil

url = "https://github.com/gerganov/llama.cpp/releases/download/b3438/llama-b3438-bin-win-cuda-cu12.2.0-x64.zip"
zip_path = "cuda_fix.zip"
extract_path = "cuda_fix_temp"
target_dir = r".\.venv\Lib\site-packages\llama_cpp\lib"

print("Downloading CUDA 12 DLLs...")
try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

print("Extracting...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
except Exception as e:
    print(f"Extraction failed: {e}")
    exit(1)

dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
source_bin = os.path.join(extract_path, "build", "bin")

# Check if build/bin exists, sometimes checking structure is needed
if not os.path.exists(source_bin):
    # Try looking in root or other folders if structure is different
    print(f"Warning: {source_bin} not found. Listing content of extract path:")
    for root, dirs, files in os.walk(extract_path):
        print(root, files)
    # Attempt to find dlls recursively
    found = False
    for root, dirs, files in os.walk(extract_path):
        for dll in dlls:
            if dll in files:
                source_bin = root
                found = True
                break
        if found: break

print(f"Copying DLLs from {source_bin} to {target_dir}...")
if not os.path.exists(target_dir):
    print(f"Target directory {target_dir} does not exist!")
    # Try to find site-packages/llama_cpp/lib dynamically
    # But for now let's hope the relative path works or fail.
    # We can try absolute path based on CWD
    target_dir = os.path.abspath(target_dir)
    print(f"Trying absolute path: {target_dir}")
    if not os.path.exists(target_dir):
         print("Still not found.")

for dll in dlls:
    src = os.path.join(source_bin, dll)
    dst = os.path.join(target_dir, dll)
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {dll}")
        else:
            print(f"Source {dll} not found in {source_bin}")
    except Exception as e:
        print(f"Error copying {dll}: {e}")

print("Cleaning up...")
if os.path.exists(zip_path):
    os.remove(zip_path)
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)

print("DLL Injection Complete.", flush=True)
