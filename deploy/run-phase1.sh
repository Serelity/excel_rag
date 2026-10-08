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
stage=${1:-}
case "$stage" in
  preflight|prepare-queries|finalize-queries|freeze|report) shift ;;
  *) printf 'Usage: bash deploy/run-phase1.sh {preflight|prepare-queries|finalize-queries|freeze|report} [options]\n' >&2; exit 2 ;;
esac
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
runner=(conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python)
case "$stage" in
  preflight) runner+=(-m retrieval_baseline.preflight) ;;
  report) runner+=(-m retrieval_baseline.case_report) ;;
  *) runner+=(-m retrieval_baseline.phase1 "$stage") ;;
esac
if [[ $stage == preflight || $stage == freeze ]]; then
  # Keep these defaults identical to run-case-evaluation.sh collect.
  runner+=(--dataset "${RETRIEVAL_DATASET:-data/retrieval-baseline-v1/dataset}"
    --index "${RETRIEVAL_LEXICAL_INDEX:-data/retrieval-baseline-v1/index}"
    --dense-index "${RETRIEVAL_DENSE_INDEX:-data/retrieval-bge-m3-v1/index}"
    --address-index "${RETRIEVAL_ADDRESS_INDEX:-data/case-search-v1/address-index}"
    --model "${RETRIEVAL_MODEL_PATH:-models/bge-m3}"
    --device "${RETRIEVAL_DEVICE:-cuda}" --dtype "${RETRIEVAL_DTYPE:-float16}"
    --max-length "${RETRIEVAL_MAX_LENGTH:-8192}" --batch-size "${RETRIEVAL_BATCH_SIZE:-8}"
    --threads "${RETRIEVAL_FAISS_THREADS:-4}"
    --reranker-model "${RERANKER_MODEL_PATH:-models/bge-reranker-v2-m3}"
    --reranker-max-length "${RERANKER_MAX_LENGTH:-1024}"
    --reranker-batch-size "${RERANKER_BATCH_SIZE:-4}")
elif [[ $stage == prepare-queries || $stage == finalize-queries ]]; then
  runner+=(--dataset "${RETRIEVAL_DATASET:-data/retrieval-baseline-v1/dataset}")
fi
exec "${runner[@]}" "$@"
