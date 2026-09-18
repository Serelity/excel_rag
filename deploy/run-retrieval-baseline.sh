#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}

if [[ -r $ENV_FILE ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${CONDA_RETRIEVAL_ENV:=civic-rag-retrieval}"
: "${RAG_INPUT_PATH:=$PROJECT_ROOT/data/raw/t_order_master.sanitized.v1_9.tsv}"
: "${RAG_RETRIEVAL_DATA_PATH:=$PROJECT_ROOT/data/retrieval/title-v1}"
: "${RAG_RUN_RECORDS_PATH:=$PROJECT_ROOT/run-records}"
: "${RAG_RETRIEVAL_RUN_PATH:=$RAG_RUN_RECORDS_PATH/retrieval-baselines}"
: "${GPU_ID:=0}"

usage() {
  cat <<'EOF'
Usage:
  bash deploy/run-retrieval-baseline.sh build [--overwrite]
  bash deploy/run-retrieval-baseline.sh popularity <dev|test> [top_k]
  bash deploy/run-retrieval-baseline.sh bm25 <dev|test> <content|goal|joint> [top_k]
  bash deploy/run-retrieval-baseline.sh bge <dev|test> <content|goal|joint> [top_k]
EOF
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

stage=$1
shift
cd "$PROJECT_ROOT"

case "$stage" in
  build)
    exec conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python build_retrieval_dataset.py \
      --input "$RAG_INPUT_PATH" \
      --output "$RAG_RETRIEVAL_DATA_PATH" \
      "$@"
    ;;
  popularity)
    split=${1:-}
    top_k=${2:-50}
    if [[ $split != dev && $split != test ]]; then
      usage >&2
      exit 2
    fi
    conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python run_popularity_baseline.py \
      --dataset "$RAG_RETRIEVAL_DATA_PATH" \
      --split "$split" \
      --top-k "$top_k" \
      --output-dir "$RAG_RETRIEVAL_RUN_PATH"
    for name in global-popularity category-popularity; do
      conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
        python evaluate_retrieval.py \
        --qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/$split.tsv" \
        --train-qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/train.tsv" \
        --run "$RAG_RETRIEVAL_RUN_PATH/$name.$split.trec" \
        --output "$RAG_RETRIEVAL_RUN_PATH/$name.$split.metrics.json"
    done
    ;;
  bm25)
    split=${1:-}
    query_view=${2:-}
    top_k=${3:-50}
    if [[ $split != dev && $split != test ]]; then
      usage >&2
      exit 2
    fi
    if [[ $query_view != content && $query_view != goal && $query_view != joint ]]; then
      usage >&2
      exit 2
    fi
    output="$RAG_RETRIEVAL_RUN_PATH/bm25-$query_view.$split.trec"
    conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python run_bm25_retrieval.py \
      --dataset "$RAG_RETRIEVAL_DATA_PATH" \
      --split "$split" \
      --query-view "$query_view" \
      --top-k "$top_k" \
      --output "$output"
    conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python evaluate_retrieval.py \
      --qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/$split.tsv" \
      --train-qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/train.tsv" \
      --run "$output" \
      --output "$RAG_RETRIEVAL_RUN_PATH/bm25-$query_view.$split.metrics.json"
    ;;
  bge)
    split=${1:-}
    query_view=${2:-}
    top_k=${3:-50}
    if [[ $split != dev && $split != test ]]; then
      usage >&2
      exit 2
    fi
    if [[ $query_view != content && $query_view != goal && $query_view != joint ]]; then
      usage >&2
      exit 2
    fi
    if [[ -z ${BGE_M3_MODEL_PATH:-} || $BGE_M3_MODEL_PATH != /* ]]; then
      echo "BGE_M3_MODEL_PATH must be an absolute local model directory" >&2
      exit 1
    fi
    if [[ ! -r $BGE_M3_MODEL_PATH/config.json ]]; then
      echo "BGE_M3_MODEL_PATH does not contain a readable config.json" >&2
      exit 1
    fi
    output="$RAG_RETRIEVAL_RUN_PATH/bge-m3-$query_view.$split.trec"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID} \
      HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
      conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python run_bge_retrieval.py \
      --dataset "$RAG_RETRIEVAL_DATA_PATH" \
      --split "$split" \
      --query-view "$query_view" \
      --model "$BGE_M3_MODEL_PATH" \
      --model-revision "${BGE_M3_MODEL_REVISION:-unknown}" \
      --device cuda:0 \
      --search-device cuda:0 \
      --top-k "$top_k" \
      --output "$output"
    conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" \
      python evaluate_retrieval.py \
      --qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/$split.tsv" \
      --train-qrels "$RAG_RETRIEVAL_DATA_PATH/qrels/train.tsv" \
      --run "$output" \
      --output "$RAG_RETRIEVAL_RUN_PATH/bge-m3-$query_view.$split.metrics.json"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
