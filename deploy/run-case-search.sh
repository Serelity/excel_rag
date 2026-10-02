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
: "${RETRIEVAL_MODEL_PATH:=models/bge-m3}"
: "${RETRIEVAL_DATASET:=data/retrieval-baseline-v1/dataset}"
: "${RETRIEVAL_LEXICAL_INDEX:=data/retrieval-baseline-v1/index}"
: "${RETRIEVAL_DENSE_INDEX:=data/retrieval-bge-m3-v1/index}"
: "${RETRIEVAL_ADDRESS_INDEX:=data/case-search-v1/address-index}"
: "${RETRIEVAL_DEVICE:=cuda}"
: "${RETRIEVAL_DTYPE:=float16}"
: "${RETRIEVAL_MAX_LENGTH:=8192}"
: "${RETRIEVAL_BATCH_SIZE:=8}"
: "${RETRIEVAL_FAISS_THREADS:=4}"
: "${RERANKER_MODEL_PATH:=models/bge-reranker-v2-m3}"
: "${RERANKER_MAX_LENGTH:=1024}"
: "${RERANKER_BATCH_SIZE:=4}"

stage=${1:-}
case "$stage" in
  prepare-address|search) shift ;;
  *) printf 'Usage: bash deploy/run-case-search.sh prepare-address | search [query options]\n' >&2; exit 2 ;;
esac
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
runner=(conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python)
if [[ $stage == prepare-address ]]; then
  if (( $# != 0 )); then
    printf 'Usage: bash deploy/run-case-search.sh prepare-address\n' >&2
    exit 2
  fi
  exec "${runner[@]}" -m retrieval_baseline.address build \
    --dataset "$RETRIEVAL_DATASET" --output "$RETRIEVAL_ADDRESS_INDEX"
fi
exec "${runner[@]}" -m retrieval_baseline.search \
  --dataset "$RETRIEVAL_DATASET" --index "$RETRIEVAL_LEXICAL_INDEX" \
  --dense-index "$RETRIEVAL_DENSE_INDEX" --address-index "$RETRIEVAL_ADDRESS_INDEX" \
  --model "$RETRIEVAL_MODEL_PATH" --device "$RETRIEVAL_DEVICE" \
  --dtype "$RETRIEVAL_DTYPE" --max-length "$RETRIEVAL_MAX_LENGTH" \
  --batch-size "$RETRIEVAL_BATCH_SIZE" --threads "$RETRIEVAL_FAISS_THREADS" \
  --reranker-model "$RERANKER_MODEL_PATH" --reranker-max-length "$RERANKER_MAX_LENGTH" \
  --reranker-batch-size "$RERANKER_BATCH_SIZE" \
  "$@"
