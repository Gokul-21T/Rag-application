import os
from huggingface_hub import hf_hub_download

# Set your Hugging Face token here
HF_TOKEN = "YOUR_ACESS TOKEN"

# Model repo and target directory
MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "models", "Phi-3-mini-4k-instruct-gguf")

# List of files to download (from the repo file list)
MODEL_FILES = [
    "Phi-3-mini-4k-instruct-q4.gguf"
]

os.makedirs(TARGET_DIR, exist_ok=True)

for filename in MODEL_FILES:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        token=HF_TOKEN,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False
    )
print("Download complete! Model files are in:", TARGET_DIR) 