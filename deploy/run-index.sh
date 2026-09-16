#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}

if [[ ! -r $ENV_FILE ]]; then
  echo "deployment environment is not readable: $ENV_FILE" >&2
  echo "create it with: cp deploy/.env.example deploy/.env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${CONDA_INDEX_ENV:=civic-rag-index}"
: "${GPU_ID:=0}"
: "${QDRANT_HOST:=127.0.0.1}"
: "${QDRANT_HTTP_PORT:=6333}"
GPU_VISIBILITY=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

if [[ $QDRANT_HOST != 127.0.0.1 ]]; then
  echo "QDRANT_HOST must remain 127.0.0.1" >&2
  exit 1
fi
if [[ -z $GPU_VISIBILITY || $GPU_VISIBILITY == *,* ]]; then
  echo "exactly one GPU must be selected by CUDA_VISIBLE_DEVICES or GPU_ID" >&2
  exit 1
fi
if [[ -z ${BGE_M3_MODEL_PATH:-} || ! -r $BGE_M3_MODEL_PATH/config.json ]]; then
  echo "BGE_M3_MODEL_PATH must contain a readable config.json" >&2
  exit 1
fi
if [[ $BGE_M3_MODEL_PATH != /* ]]; then
  echo "BGE_M3_MODEL_PATH must be absolute" >&2
  exit 1
fi
if [[ ! ${BGE_M3_MODEL_REVISION:-} =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "BGE_M3_MODEL_REVISION must be the audited 40-character commit SHA" >&2
  exit 1
fi
if ! find "$BGE_M3_MODEL_PATH" -maxdepth 2 -name '*.safetensors' -print -quit | grep -q .; then
  echo "BGE_M3_MODEL_PATH does not contain safetensors weights" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if [[ -z ${QDRANT_API_KEY:-} ]]; then
  unset QDRANT_API_KEY
fi

for argument in "$@"; do
  case "$argument" in
    --model | --model=* | --model-revision | --model-revision=* | --device | --device=* | \
      --host | --host=* | --port | --port=* | --url | --url=* | --https | --no-https | \
      --api-key-env | --api-key-env=*)
      echo "run-index.sh owns model, device, and Qdrant connection options" >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" conda run --no-capture-output \
  -n "$CONDA_INDEX_ENV" python deploy/validate-runtime.py index

exec env \
  CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  TRANSFORMERS_OFFLINE=1 \
  DO_NOT_TRACK=1 \
  conda run --no-capture-output -n "$CONDA_INDEX_ENV" \
  python index_chunks.py \
  --model "$BGE_M3_MODEL_PATH" \
  --model-revision "$BGE_M3_MODEL_REVISION" \
  --device cuda:0 \
  --host "$QDRANT_HOST" \
  --port "$QDRANT_HTTP_PORT" \
  --no-https \
  "$@"
