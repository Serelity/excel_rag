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
: "${RAG_EXTRACTION_CONCURRENCY:=4}"
: "${RAG_PILOT_PATH:=$PROJECT_ROOT/data/derived/qwen3-pilot-v2-${RAG_PILOT_SIZE}.jsonl}"
: "${RAG_SEMANTIC_OUTPUT_DIR:=$PROJECT_ROOT/data/processed/qwen3-semantic-v3-clean}"
: "${RAG_RUN_RECORDS_PATH:=$PROJECT_ROOT/run-records/qwen3-semantic-v3-clean}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_STARTUP_TIMEOUT_SECONDS:=1800}"

limit=20
mode=()
while (($#)); do
  case "$1" in
    --limit)
      (($# >= 2)) || { printf 'ERROR: --limit needs a value\n' >&2; exit 2; }
      limit=$2
      shift 2
      ;;
    --resume|--overwrite)
      ((${#mode[@]} == 0)) || { printf 'ERROR: choose only one run mode\n' >&2; exit 2; }
      mode=("$1")
      shift
      ;;
    -h|--help)
      printf 'Usage: bash deploy/run-qwen3-pilot.sh [--limit N] [--resume|--overwrite]\n'
      printf 'Default: process 20 new records from the fixed 2000-record pilot.\n'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done
[[ $RAG_PILOT_SIZE =~ ^[0-9]+$ ]] && ((RAG_PILOT_SIZE >= 1)) || \
  { printf 'ERROR: RAG_PILOT_SIZE must be positive\n' >&2; exit 2; }
[[ $RAG_EXTRACTION_CONCURRENCY =~ ^[0-9]+$ ]] && ((RAG_EXTRACTION_CONCURRENCY >= 1)) || \
  { printf 'ERROR: RAG_EXTRACTION_CONCURRENCY must be positive\n' >&2; exit 2; }
[[ $VLLM_STARTUP_TIMEOUT_SECONDS =~ ^[0-9]+$ ]] && ((VLLM_STARTUP_TIMEOUT_SECONDS >= 1)) || \
  { printf 'ERROR: VLLM_STARTUP_TIMEOUT_SECONDS must be positive\n' >&2; exit 2; }
[[ ${QWEN_MODEL_FINGERPRINT_SHA256:-} =~ ^sha256:[0-9a-fA-F]{64}$ ]] || \
  { printf 'ERROR: invalid QWEN_MODEL_FINGERPRINT_SHA256\n' >&2; exit 2; }
[[ $limit =~ ^[0-9]+$ ]] && ((limit >= 1 && limit <= RAG_PILOT_SIZE)) || \
  { printf 'ERROR: limit must be between 1 and %s\n' "$RAG_PILOT_SIZE" >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { printf 'ERROR: curl is not on PATH\n' >&2; exit 2; }
command -v setsid >/dev/null 2>&1 || { printf 'ERROR: setsid is not on PATH\n' >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { printf 'ERROR: flock is not on PATH\n' >&2; exit 2; }
command -v sha256sum >/dev/null 2>&1 || { printf 'ERROR: sha256sum is not on PATH\n' >&2; exit 2; }

mkdir -p "$RAG_SEMANTIC_OUTPUT_DIR" "$RAG_RUN_RECORDS_PATH"
exec 9>"$RAG_RUN_RECORDS_PATH/job.lock"
flock -n 9 || { printf 'ERROR: another Qwen3 extraction job holds the job lock\n' >&2; exit 2; }
cd "$PROJECT_ROOT"
if [[ ! -f $RAG_PILOT_PATH ]]; then
  printf 'Preparing deterministic %s-record pilot...\n' "$RAG_PILOT_SIZE"
  conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
    python -m semantic_extraction.selector \
    --input "$RAG_INPUT_PATH" \
    --output "$RAG_PILOT_PATH" \
    --size "$RAG_PILOT_SIZE" \
    --seed "$RAG_PILOT_SEED"
fi

output=$RAG_SEMANTIC_OUTPUT_DIR/pilot-${RAG_PILOT_SIZE}.jsonl
errors=$RAG_SEMANTIC_OUTPUT_DIR/pilot-${RAG_PILOT_SIZE}.errors.jsonl
cache=$RAG_SEMANTIC_OUTPUT_DIR/extraction-cache.sqlite3
job_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
vllm_log=$RAG_RUN_RECORDS_PATH/vllm-$job_id.log
extraction_log=$RAG_RUN_RECORDS_PATH/extraction-$job_id.log
status_log=$RAG_RUN_RECORDS_PATH/job-$job_id.status

printf 'job_id=%s\n' "$job_id" | tee "$status_log"
printf 'pilot_input=%s\noutput=%s\nerrors=%s\n' \
  "$RAG_PILOT_PATH" "$output" "$errors" | tee -a "$status_log"
printf 'vllm_log=%s\nextraction_log=%s\n' "$vllm_log" "$extraction_log" | tee -a "$status_log"
printf 'model_fingerprint=%s\nprompt_version=%s\n' \
  "$QWEN_MODEL_FINGERPRINT_SHA256" "case-content-semantic-v3" >> "$status_log"
printf 'pilot_sha256=%s\n' "$(sha256sum "$RAG_PILOT_PATH" | awk '{print $1}')" >> "$status_log"
printf 'Request logging is disabled; prompts and source text are not written to job logs.\n'

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n ${vllm_pid:-} ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    printf '%s stopping_vllm pid=%s\n' "$(date -u +%FT%TZ)" "$vllm_pid" >> "$status_log"
    kill -TERM -- "-$vllm_pid" 2>/dev/null || kill -TERM "$vllm_pid" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "$vllm_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  printf '%s job_finished exit_code=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$status_log"
  exit "$status"
}
trap cleanup EXIT INT TERM

if (exec 3<>"/dev/tcp/$VLLM_HOST/$VLLM_PORT") 2>/dev/null; then
  exec 3>&-
  printf 'ERROR: %s:%s is already occupied\n' "$VLLM_HOST" "$VLLM_PORT" >&2
  exit 2
fi

printf '%s starting_vllm\n' "$(date -u +%FT%TZ)" >> "$status_log"
setsid bash deploy/run-qwen3-vllm.sh >"$vllm_log" 2>&1 &
vllm_pid=$!

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
health_url=http://$VLLM_HOST:$VLLM_PORT/health
while ! curl --silent --show-error --fail --max-time 5 "$health_url" >/dev/null 2>&1; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    printf 'ERROR: vLLM exited during startup; inspect %s\n' "$vllm_log" >&2
    exit 2
  fi
  if ((SECONDS >= deadline)); then
    printf 'ERROR: vLLM did not become ready within %s seconds\n' \
      "$VLLM_STARTUP_TIMEOUT_SECONDS" >&2
    exit 2
  fi
  sleep 5
done
printf '%s vllm_health_ready\n' "$(date -u +%FT%TZ)" >> "$status_log"

models_url=http://$VLLM_HOST:$VLLM_PORT/v1/models
auth_args=()
if [[ -n ${VLLM_API_KEY:-} ]]; then
  auth_args=(-H "Authorization: Bearer $VLLM_API_KEY")
fi
models_json=$(curl --silent --show-error --fail --max-time 10 "${auth_args[@]}" "$models_url")
MODELS_JSON=$models_json conda run -n "$CONDA_EXTRACT_ENV" python - <<'PY'
import json
import os

payload = json.loads(os.environ["MODELS_JSON"])
names = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
expected = os.environ["QWEN_SERVED_MODEL_NAME"]
if expected not in names:
    raise SystemExit(f"served model alias missing: {expected}")
PY
unset models_json MODELS_JSON
printf '%s model_alias_ready name=%s\n' \
  "$(date -u +%FT%TZ)" "$QWEN_SERVED_MODEL_NAME" >> "$status_log"

printf '%s extraction_started limit=%s\n' "$(date -u +%FT%TZ)" "$limit" >> "$status_log"
set +e
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m semantic_extraction.run \
  --input "$RAG_PILOT_PATH" \
  --output "$output" \
  --errors "$errors" \
  --cache "$cache" \
  --limit "$limit" \
  --concurrency "$RAG_EXTRACTION_CONCURRENCY" \
  "${mode[@]}" >"$extraction_log" 2>&1
extraction_status=$?
set -e
tail -n 1 "$extraction_log" || true
output_count=$(wc -l < "$output" 2>/dev/null || printf '0')
error_count=$(wc -l < "$errors" 2>/dev/null || printf '0')
printf 'output_records=%s quarantine_records=%s\n' "$output_count" "$error_count" | tee -a "$status_log"
printf '%s extraction_finished exit_code=%s\n' \
  "$(date -u +%FT%TZ)" "$extraction_status" >> "$status_log"
exit "$extraction_status"
