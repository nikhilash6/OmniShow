#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/data"
DATA_DIR="$ROOT_DIR/data"

python - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception:
    print("[ERROR] missing huggingface_hub. Install it via: python -m pip install -U huggingface_hub", file=sys.stderr)
    raise

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def _snapshot_download(repo_id: str):
    local_dir = os.path.join(DATA_DIR, repo_id)
    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
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

print("[INFO] Downloading dataset donghao-zhou/OmniShow_example_dataset ...")
_snapshot_download("donghao-zhou/OmniShow_example_dataset")
print("[DONE] Dataset downloaded to:", DATA_DIR)
PY
