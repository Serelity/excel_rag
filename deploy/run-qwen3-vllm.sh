#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env.semantic}
[[ -r $ENV_FILE ]] || { printf 'ERROR: missing environment file: %s\n' "$ENV_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${CONDA_EXTRACT_ENV:=civic-rag-extract}"
: "${GPU_ID:=0}"
: "${QWEN_SERVED_MODEL_NAME:=Qwen3-30B-A3B}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_MAX_MODEL_LEN:=16384}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_MAX_NUM_SEQS:=8}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=8192}"
: "${VLLM_CACHE_PATH:=$PROJECT_ROOT/.cache/vllm}"

[[ $VLLM_HOST == 127.0.0.1 ]] || { printf 'ERROR: VLLM_HOST must be 127.0.0.1\n' >&2; exit 2; }
[[ $VLLM_PORT =~ ^[0-9]+$ ]] && ((VLLM_PORT >= 1024 && VLLM_PORT <= 65535)) || \
  { printf 'ERROR: VLLM_PORT must be between 1024 and 65535\n' >&2; exit 2; }
[[ $VLLM_MAX_MODEL_LEN =~ ^[0-9]+$ ]] && ((VLLM_MAX_MODEL_LEN >= 4096)) || \
  { printf 'ERROR: VLLM_MAX_MODEL_LEN must be at least 4096\n' >&2; exit 2; }
[[ $VLLM_MAX_NUM_SEQS =~ ^[0-9]+$ ]] && ((VLLM_MAX_NUM_SEQS >= 1)) || \
  { printf 'ERROR: VLLM_MAX_NUM_SEQS must be positive\n' >&2; exit 2; }
[[ $VLLM_MAX_NUM_BATCHED_TOKENS =~ ^[0-9]+$ ]] && ((VLLM_MAX_NUM_BATCHED_TOKENS >= 1)) || \
  { printf 'ERROR: VLLM_MAX_NUM_BATCHED_TOKENS must be positive\n' >&2; exit 2; }
[[ -n ${QWEN_MODEL_PATH:-} && $QWEN_MODEL_PATH == /* ]] || \
  { printf 'ERROR: QWEN_MODEL_PATH must be absolute\n' >&2; exit 2; }
[[ ${QWEN_MODEL_FINGERPRINT_SHA256:-} =~ ^sha256:[0-9a-fA-F]{64}$ ]] || \
  { printf 'ERROR: invalid QWEN_MODEL_FINGERPRINT_SHA256\n' >&2; exit 2; }
[[ $QWEN_SERVED_MODEL_NAME == Qwen3-30B-A3B ]] || \
  { printf 'ERROR: served model alias must be Qwen3-30B-A3B\n' >&2; exit 2; }
if [[ -n ${VLLM_API_KEY:-} && ! ${VLLM_API_KEY:-} =~ ^[A-Za-z0-9_-]{32,128}$ ]]; then
  printf 'ERROR: VLLM_API_KEY must be empty or a 32-128 character URL-safe token\n' >&2
  exit 2
fi
gpu_visibility=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
[[ -n $gpu_visibility && $gpu_visibility != *,* ]] || \
  { printf 'ERROR: exactly one GPU must be visible\n' >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }

mkdir -p "$VLLM_CACHE_PATH"/{huggingface,vllm,torchinductor,triton}
cd "$PROJECT_ROOT"
env CUDA_VISIBLE_DEVICES="$gpu_visibility" conda run --no-capture-output \
  -n "$CONDA_EXTRACT_ENV" python -m semantic_extraction.validate_runtime
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m semantic_extraction.validate_model --model-dir "$QWEN_MODEL_PATH"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY

args=(
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --model "$QWEN_MODEL_PATH"
  --served-model-name "$QWEN_SERVED_MODEL_NAME"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
  --enable-prefix-caching
  --enable-chunked-prefill
  --generation-config vllm
  --seed 42
  --disable-log-requests
  --disable-uvicorn-access-log
)

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_visibility" \
  HF_HOME="$VLLM_CACHE_PATH/huggingface" \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  TRANSFORMERS_OFFLINE=1 \
  VLLM_CACHE_ROOT="$VLLM_CACHE_PATH/vllm" \
  VLLM_NO_USAGE_STATS=1 \
  TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_PATH/torchinductor" \
  TRITON_CACHE_DIR="$VLLM_CACHE_PATH/triton" \
  DO_NOT_TRACK=1 \
  TOKENIZERS_PARALLELISM=false \
  conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m vllm.entrypoints.openai.api_server "${args[@]}"
