#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
if [[ -f deploy/.env.retrieval ]]; then
  # shellcheck disable=SC1091
  source deploy/.env.retrieval
fi
: "${CONDA_RETRIEVAL_ENV:=civic-rag-retrieval}"
: "${RETRIEVAL_DATASET:=data/retrieval-baseline-v1/dataset}"
: "${RETRIEVAL_LEXICAL_INDEX:=data/retrieval-baseline-v1/index}"
: "${RETRIEVAL_BM25_REPORT:=data/retrieval-baseline-v1/dev-bm25}"
: "${RETRIEVAL_DENSE_REPORT:=data/retrieval-bge-m3-v1/dev-dense}"
: "${RETRIEVAL_HYBRID_REPORT:=data/retrieval-hybrid-v1/dev-hybrid}"
if (( $# != 0 )); then
  printf 'Usage: bash deploy/run-hybrid-baseline.sh\n' >&2
  exit 2
fi
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
exec conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
  python -m retrieval_baseline.hybrid \
  --dataset "$RETRIEVAL_DATASET" --index "$RETRIEVAL_LEXICAL_INDEX" \
  --bm25 "$RETRIEVAL_BM25_REPORT" --dense "$RETRIEVAL_DENSE_REPORT" \
  --output "$RETRIEVAL_HYBRID_REPORT" --split dev
