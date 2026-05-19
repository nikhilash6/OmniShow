
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/models"
MODELS_DIR="$ROOT_DIR/models"

python - <<'PY'
import os
import sys
import glob

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except Exception as e:
    print("[ERROR] missing huggingface_hub. Install it via: python -m pip install -U huggingface_hub", file=sys.stderr)
    raise

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def _has_any(local_dir: str, pattern: str) -> bool:
    return len(glob.glob(os.path.join(local_dir, pattern))) > 0


def _is_satisfied(local_dir: str, required_patterns) -> bool:
    if not os.path.isdir(local_dir):
        return False
    for pat in (required_patterns or []):
        # `pat` can be a string, or a list/tuple meaning "any of these patterns".
        if isinstance(pat, (list, tuple)):
            if not any(_has_any(local_dir, p) for p in pat):
                return False
        else:
            if not _has_any(local_dir, pat):
                return False
    return True


def _snapshot_download(repo_id: str, allow_patterns, required_patterns=None):
    local_dir = os.path.join(MODELS_DIR, repo_id)
    os.makedirs(local_dir, exist_ok=True)

    # Skip if already downloaded.
    if required_patterns is None:
        required_patterns = allow_patterns
    if _is_satisfied(local_dir, required_patterns):
        print(f"[SKIP] {repo_id} already exists in: {local_dir}")
        return

    kwargs = dict(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        local_files_only=False,
    )

    for k, v in [
        ("resume_download", True),
        ("local_dir_use_symlinks", False),
    ]:
        try:
            snapshot_download(**kwargs, **{k: v})
            return
        except TypeError:
            continue
    snapshot_download(**kwargs)

print("[INFO] Downloading Wan-AI/Wan2.1-I2V-14B-480P ...")
_snapshot_download(
    "Wan-AI/Wan2.1-I2V-14B-480P",
    [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
)

print("[INFO] Downloading Wan-AI/Wan2.1-I2V-14B-720P ...")
_snapshot_download(
    "Wan-AI/Wan2.1-I2V-14B-720P",
    [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
)

print("[INFO] Downloading Wan-AI/Wan2.1-T2V-1.3B tokenizer assets ...")
_snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    [
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/special_tokens_map.json",
    ],
)

print("[INFO] Downloading facebook/wav2vec2-base-960h ...")
_snapshot_download(
    "facebook/wav2vec2-base-960h",
    ["*"],
    required_patterns=[
        "config.json",
        "preprocessor_config.json",
        # weights could be either safetensors or pytorch bin
        ("model.safetensors", "pytorch_model.bin"),
    ],
)

print("[DONE] Weights downloaded to:", MODELS_DIR)
PY
