#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
CONDA_RETRIEVAL_ENV=${CONDA_RETRIEVAL_ENV:-civic-rag-retrieval}
PYTORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
cd "$PROJECT_ROOT"
if conda run -n "$CONDA_RETRIEVAL_ENV" python --version >/dev/null 2>&1; then
  conda env update -n "$CONDA_RETRIEVAL_ENV" -f deploy/environment-retrieval.yml
else
  conda env create -n "$CONDA_RETRIEVAL_ENV" -f deploy/environment-retrieval.yml
fi
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m pip install \
  --index-url "$PYTORCH_CUDA_INDEX_URL" --only-binary torch 'torch==2.6.0+cu124'
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m pip install \
  --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[retrieval]' 'modelscope==1.25.0'
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m pip check
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python - <<'PY'
import importlib.metadata
import sqlite3
import sys

import faiss
import sentence_transformers
import torch
from modelscope.hub.snapshot_download import snapshot_download

expected = {
    "torch": "2.6.0", "transformers": "4.51.3", "sentence-transformers": "3.4.1",
    "numpy": "1.26.4", "faiss-cpu": "1.10.0", "filelock": "3.18.0",
    "sentencepiece": "0.2.0", "modelscope": "1.25.0",
}
if sys.version_info[:2] != (3, 11):
    raise SystemExit("ERROR: expected Python 3.11")
for name, wanted in expected.items():
    actual = importlib.metadata.version(name)
    if actual.split("+", 1)[0] != wanted:
        raise SystemExit(f"ERROR: {name}={actual}, expected {wanted}")
    print(f"{name}={actual}")
if torch.version.cuda != "12.4":
    raise SystemExit(f"ERROR: expected CUDA 12.4 build, got {torch.version.cuda}")
with sqlite3.connect(":memory:") as db:
    db.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
print("environment_check=passed")
print("GPU availability is checked later on a GPU node; no model was loaded or downloaded.")
PY
