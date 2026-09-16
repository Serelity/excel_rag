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

: "${CONDA_EXTRACT_ENV:=civic-rag-extract}"

if [[ ${RAG_JOB_WRAPPER_ACTIVE:-0} != 1 ]]; then
  echo "deploy/run-extraction.sh is internal; use deploy/run-extraction-job.sh" >&2
  exit 2
fi

if [[ ! ${QWEN_MODEL_FINGERPRINT_SHA256:-} =~ ^sha256:[0-9a-fA-F]{64}$ ]]; then
  echo "QWEN_MODEL_FINGERPRINT_SHA256 must have the form sha256:<64hex>" >&2
  exit 1
fi
if [[ -z ${QWEN_MODELSCOPE_REPO_ID:-} ]]; then
  echo "QWEN_MODELSCOPE_REPO_ID must record the ModelScope repository" >&2
  exit 1
fi
if [[ -z ${QWEN_MODEL_REVISION:-} ]]; then
  echo "QWEN_MODEL_REVISION must be non-empty or 'unknown'" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is not available on PATH" >&2
  exit 1
fi

if [[ -z ${RAG_CODE_COMMIT:-} ]]; then
  RAG_CODE_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)
fi
if [[ ! $RAG_CODE_COMMIT =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "RAG_CODE_COMMIT must identify the checked-out Git commit" >&2
  exit 1
fi
mapfile -t git_changes < <(
  git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=normal 2>/dev/null
)
if ((${#git_changes[@]} != 0)); then
  echo "repository worktree must be clean before extraction" >&2
  exit 1
fi
export RAG_CODE_COMMIT

cd "$PROJECT_ROOT"
exec env \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  TRANSFORMERS_OFFLINE=1 \
  VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 \
  TOKENIZERS_PARALLELISM=false \
  conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python main.py "$@"
