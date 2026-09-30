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
  --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[retrieval]'
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m pip check
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -c \
  'import torch, faiss, sentence_transformers; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "faiss", faiss.__version__, "sentence_transformers", sentence_transformers.__version__)'
