#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
# These preparation scripts deliberately use explicit environment overrides,
# without sourcing the experiment's .env.retrieval settings.
CONDA_RETRIEVAL_ENV=${CONDA_RETRIEVAL_ENV:-civic-rag-retrieval}
RETRIEVAL_MODEL_PATH=${RETRIEVAL_MODEL_PATH:-$PROJECT_ROOT/models/bge-m3}
RETRIEVAL_MODEL_REVISION=${RETRIEVAL_MODEL_REVISION:-master}
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
exec conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
  python -m retrieval_baseline.prepare_model \
  --output "$RETRIEVAL_MODEL_PATH" --revision "$RETRIEVAL_MODEL_REVISION" "$@"
