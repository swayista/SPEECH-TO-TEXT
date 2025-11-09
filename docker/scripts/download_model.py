import os, glob, sys
from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN", "").strip()
if token:
    login(token=token, add_to_git_credential=False)

local_dir = "/models"
repo_id = "unsloth/Qwen3-1.7B-GGUF"
print(f"Downloading model {repo_id} to {local_dir} ...", flush=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=["**/Q4_K_M/**", "*Q4_K_M*.gguf"]
)

candidates = glob.glob(os.path.join(local_dir, "**", "Q4_K_M", "*.gguf"), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(local_dir, "**", "*Q4_K_M*.gguf"), recursive=True)
if not candidates:
    print("ERROR: No Q4_K_M .gguf files found.", file=sys.stderr)
    sys.exit(1)

target = sorted(candidates)[0]
link = os.path.join(local_dir, "model.gguf")
if os.path.islink(link) or os.path.exists(link):
    os.unlink(link)
os.symlink(target, link)
print(f"Linked {link} -> {target}", flush=True)
