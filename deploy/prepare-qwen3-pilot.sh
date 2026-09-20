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
: "${RAG_PILOT_SIZE:=2000}"
: "${RAG_PILOT_SEED:=20260920}"
: "${RAG_PILOT_PATH:=$PROJECT_ROOT/data/derived/qwen3-pilot-${RAG_PILOT_SIZE}.jsonl}"

cd "$PROJECT_ROOT"
exec conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m semantic_extraction.selector \
  --input "$RAG_INPUT_PATH" \
  --output "$RAG_PILOT_PATH" \
  --size "$RAG_PILOT_SIZE" \
  --seed "$RAG_PILOT_SEED" \
  "$@"
