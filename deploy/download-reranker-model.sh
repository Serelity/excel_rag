#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
CONDA_RETRIEVAL_ENV=${CONDA_RETRIEVAL_ENV:-civic-rag-retrieval}
RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH:-$PROJECT_ROOT/models/bge-reranker-v2-m3}
RERANKER_MODEL_ID=${RERANKER_MODEL_ID:-BAAI/bge-reranker-v2-m3}
RERANKER_MODEL_REVISION=${RERANKER_MODEL_REVISION:-master}
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
exec conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
  python -m retrieval_baseline.prepare_reranker --output "$RERANKER_MODEL_PATH" \
  --model-id "$RERANKER_MODEL_ID" --revision "$RERANKER_MODEL_REVISION" "$@"
