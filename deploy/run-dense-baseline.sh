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
: "${RETRIEVAL_BM25_REPORT:=data/retrieval-baseline-v1/dev-bm25}"
: "${RETRIEVAL_DENSE_INDEX:=data/retrieval-bge-m3-v1/index}"
: "${RETRIEVAL_DENSE_REPORT:=data/retrieval-bge-m3-v1/dev-dense}"
: "${RETRIEVAL_COMPARISON:=data/retrieval-bge-m3-v1/comparison}"
: "${RETRIEVAL_DEVICE:=cuda}"
: "${RETRIEVAL_DTYPE:=float16}"
: "${RETRIEVAL_MAX_LENGTH:=8192}"
: "${RETRIEVAL_BATCH_SIZE:=8}"
: "${RETRIEVAL_SHARD_SIZE:=512}"
: "${RETRIEVAL_FAISS_THREADS:=4}"
stage=${1:-all}
case "$stage" in check|build|evaluate|compare|all) ;;
  *) printf 'Usage: bash deploy/run-dense-baseline.sh [check|build|evaluate|compare|all]\n' >&2; exit 2 ;;
esac
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
runner=(conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python)
model_args=(--model "$RETRIEVAL_MODEL_PATH" --device "$RETRIEVAL_DEVICE"
  --dtype "$RETRIEVAL_DTYPE" --max-length "$RETRIEVAL_MAX_LENGTH"
  --batch-size "$RETRIEVAL_BATCH_SIZE")
if [[ $stage == check ]]; then
  exec "${runner[@]}" -m retrieval_baseline.dense check "${model_args[@]}"
fi
if [[ $stage == all || $stage == build ]]; then
  "${runner[@]}" -m retrieval_baseline.dense build "${model_args[@]}" \
    --dataset "$RETRIEVAL_DATASET" --output "$RETRIEVAL_DENSE_INDEX" \
    --shard-size "$RETRIEVAL_SHARD_SIZE" --resume
fi
if [[ $stage == all || $stage == evaluate ]]; then
  "${runner[@]}" -m retrieval_baseline.dense evaluate "${model_args[@]}" \
    --dataset "$RETRIEVAL_DATASET" --dense-index "$RETRIEVAL_DENSE_INDEX" \
    --lexical-index "$RETRIEVAL_LEXICAL_INDEX" --output "$RETRIEVAL_DENSE_REPORT" \
    --split dev --case-k 50 --threads "$RETRIEVAL_FAISS_THREADS"
fi
if [[ $stage == all || $stage == compare ]]; then
  "${runner[@]}" -m retrieval_baseline.compare --dataset "$RETRIEVAL_DATASET" \
    --bm25 "$RETRIEVAL_BM25_REPORT" --dense "$RETRIEVAL_DENSE_REPORT" \
    --output "$RETRIEVAL_COMPARISON"
fi
