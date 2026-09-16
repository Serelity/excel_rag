#!/usr/bin/env bash
set -euo pipefail

# Never expose sourced settings or API keys through shell tracing.
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}

if [[ ! -r $ENV_FILE ]]; then
  echo "deployment environment is not readable: $ENV_FILE" >&2
  echo "create it with: cp deploy/.env.example deploy/.env" >&2
  exit 1
fi

set +u
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
set -u

# This stage is entirely local. Prevent inherited platform proxies from
# intercepting loopback model traffic or influencing child HTTP clients.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY

: "${CONDA_EXTRACT_ENV:=civic-rag-extract}"
: "${GPU_ID:=0}"
: "${QWEN_MODEL_PATH:=$PROJECT_ROOT/models/Qwen3-30B-A3B}"
: "${QWEN_SERVED_MODEL_NAME:=Qwen3-30B-A3B}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_CACHE_PATH:=$PROJECT_ROOT/.cache/vllm}"
: "${VLLM_MAX_MODEL_LEN:=16384}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_MAX_NUM_SEQS:=8}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=8192}"
: "${VLLM_ENFORCE_EAGER:=0}"

GPU_VISIBILITY=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

[[ $VLLM_HOST == 127.0.0.1 ]] || die "VLLM_HOST must remain 127.0.0.1"
[[ $VLLM_PORT == 8000 ]] || die "VLLM_PORT must remain 8000 to match configs/config.yaml"
[[ $QWEN_SERVED_MODEL_NAME == Qwen3-30B-A3B ]] || \
  die "QWEN_SERVED_MODEL_NAME must remain Qwen3-30B-A3B"
if [[ -n ${VLLM_API_KEY:-} && \
  ! ${VLLM_API_KEY:-} =~ ^[A-Za-z0-9_-]{32,128}$ ]]; then
  die "VLLM_API_KEY must be empty or a 32-128 character URL-safe token"
fi
[[ -n ${QWEN_MODEL_PATH:-} && $QWEN_MODEL_PATH == /* ]] || \
  die "QWEN_MODEL_PATH must be an absolute path"
[[ $VLLM_CACHE_PATH == /* ]] || die "VLLM_CACHE_PATH must be an absolute path"
[[ -r $QWEN_MODEL_PATH/config.json ]] || \
  die "QWEN_MODEL_PATH must contain a readable config.json"
[[ -n ${QWEN_MODELSCOPE_REPO_ID:-} ]] || die "QWEN_MODELSCOPE_REPO_ID must be non-empty"
[[ -n ${QWEN_MODEL_REVISION:-} ]] || die "QWEN_MODEL_REVISION must be non-empty or 'unknown'"
[[ ${QWEN_MODEL_FINGERPRINT_SHA256:-} =~ ^sha256:[0-9a-fA-F]{64}$ ]] || \
  die "QWEN_MODEL_FINGERPRINT_SHA256 must have the form sha256:<64hex>"
[[ $VLLM_ENFORCE_EAGER == 0 || $VLLM_ENFORCE_EAGER == 1 ]] || \
  die "VLLM_ENFORCE_EAGER must be 0 or 1"
[[ -n $GPU_VISIBILITY && $GPU_VISIBILITY != *,* ]] || \
  die "exactly one GPU must be selected by CUDA_VISIBLE_DEVICES or GPU_ID"
command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"

mkdir -p -- "$VLLM_CACHE_PATH"
[[ -d $VLLM_CACHE_PATH && -w $VLLM_CACHE_PATH ]] || \
  die "VLLM_CACHE_PATH is not writable: $VLLM_CACHE_PATH"
mkdir -p -- \
  "$VLLM_CACHE_PATH/huggingface" \
  "$VLLM_CACHE_PATH/vllm" \
  "$VLLM_CACHE_PATH/torchinductor" \
  "$VLLM_CACHE_PATH/triton"

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" conda run --no-capture-output \
  -n "$CONDA_EXTRACT_ENV" python deploy/validate-runtime.py extract

# The single-job wrapper verifies every selected model file before setting this
# marker. A direct invocation performs the same expensive check itself.
if [[ ${RAG_MODEL_FINGERPRINT_VERIFIED:-0} != 1 ]]; then
  conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
    python deploy/model-fingerprint.py \
    --model-dir "$QWEN_MODEL_PATH" \
    --expect "$QWEN_MODEL_FINGERPRINT_SHA256"
fi

server_args=(
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
if [[ $VLLM_ENFORCE_EAGER == 1 ]]; then
  server_args+=(--enforce-eager)
fi

exec env \
  CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" \
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
  python deploy/vllm-entrypoint.py "${server_args[@]}"
