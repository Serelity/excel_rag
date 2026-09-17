#!/usr/bin/env bash
set -euo pipefail

# Never expose sourced settings or API keys through shell tracing.
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}
readonly LAUNCH_CODE_COMMIT=${RAG_CODE_COMMIT-}
readonly LAUNCH_CODE_BRANCH=${RAG_CODE_BRANCH-}

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

# The outer job wrapper validated these launch-scoped values. Restore them
# after sourcing .env so stale persistent settings cannot replace provenance.
RAG_CODE_COMMIT=$LAUNCH_CODE_COMMIT
RAG_CODE_BRANCH=$LAUNCH_CODE_BRANCH

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
if [[ ! $RAG_CODE_COMMIT =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "RAG_CODE_COMMIT must be a full lowercase 40- or 64-character hexadecimal commit" >&2
  exit 1
fi
if [[ ${#RAG_CODE_BRANCH} -gt 255 || \
  ! $RAG_CODE_BRANCH =~ ^[A-Za-z0-9_][A-Za-z0-9._/-]*$ || \
  $RAG_CODE_BRANCH == HEAD || \
  $RAG_CODE_BRANCH == *".."* || \
  $RAG_CODE_BRANCH == *"//"* || \
  $RAG_CODE_BRANCH == *"@{"* || \
  $RAG_CODE_BRANCH == */.* || \
  $RAG_CODE_BRANCH == *.lock/* || \
  $RAG_CODE_BRANCH == */ || \
  $RAG_CODE_BRANCH == *. || \
  $RAG_CODE_BRANCH == *.lock ]]; then
  echo "RAG_CODE_BRANCH must be a safe non-empty branch name" >&2
  exit 1
fi
export RAG_CODE_COMMIT
export RAG_CODE_BRANCH

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
