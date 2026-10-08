#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null
umask 077

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
stage=${1:-}
case "$stage" in inspect|smoke|all) shift ;; *)
  printf 'Usage: bash deploy/run-case-contract.sh {inspect|smoke|all} --output NEW_DIR [--conda-env NAME] [--env-file FILE] [--input-dir DIR]\n' >&2
  exit 2 ;;
esac
output= env_file=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env.semantic}
contract_env=civic-rag-extract-v1
input_dir=$PROJECT_ROOT/data/case-relevance-phase1-v1/extraction-contract-v1-001
while (($#)); do
  (($# >= 2)) || { printf 'ERROR: option requires a value\n' >&2; exit 2; }
  case "$1" in
    --output) output=$2 ;;
    --env-file) env_file=$2 ;;
    --input-dir) input_dir=$2 ;;
    --conda-env) contract_env=$2 ;;
    *) printf 'ERROR: unknown option\n' >&2; exit 2 ;;
  esac
  shift 2
done
[[ -n $output && ! -e $output ]] || { printf 'ERROR: --output must be a new directory\n' >&2; exit 2; }
[[ $contract_env =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { printf 'ERROR: invalid environment name\n' >&2; exit 2; }
if [[ -r $env_file ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
elif [[ $stage != inspect ]]; then
  printf 'ERROR: missing semantic environment file\n' >&2; exit 2
fi
CONDA_BIN=${CONDA_EXE:-}
[[ -x $CONDA_BIN ]] || CONDA_BIN=$(type -P conda || true)
[[ -x $CONDA_BIN ]] || { printf 'ERROR: conda executable is unavailable\n' >&2; exit 2; }
CONTRACT_CONDA_BASE=$("$CONDA_BIN" info --base)
base_python=$CONTRACT_CONDA_BASE/bin/python
[[ -x $base_python ]] || { printf 'ERROR: base Python is unavailable\n' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
output=$(cd -P -- "$output" && pwd -P)
probe_args=(--conda "$CONDA_BIN" --conda-env "$contract_env" --output "$output/environment.json")
"$base_python" deploy/inspect-case-contract-env.py "${probe_args[@]}"
[[ $stage != inspect ]] || exit 0
client_prefix=$("$base_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["client"]["prefix"])' "$output/environment.json")
serve_prefix=$("$base_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["server"]["prefix"])' "$output/environment.json")
client=("$CONDA_BIN" run --no-capture-output -p "$client_prefix" python)
export PATH="$(dirname -- "$CONDA_BIN"):$PATH"

: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_STARTUP_TIMEOUT_SECONDS:=1800}"
: "${CASE_CONTRACT_MAX_TOKENS:=8192}"
: "${VLLM_MAX_MODEL_LEN:=16384}"
: "${QWEN_SERVED_MODEL_NAME:=Qwen3-30B-A3B}"
[[ $VLLM_HOST == 127.0.0.1 && $VLLM_PORT =~ ^[0-9]+$ ]] && \
  ((VLLM_PORT >= 1024 && VLLM_PORT <= 65535)) || { printf 'ERROR: invalid loopback endpoint\n' >&2; exit 2; }
[[ $VLLM_STARTUP_TIMEOUT_SECONDS =~ ^[0-9]+$ && $CASE_CONTRACT_MAX_TOKENS =~ ^[0-9]+$ && $VLLM_MAX_MODEL_LEN =~ ^[0-9]+$ ]] && \
  ((VLLM_STARTUP_TIMEOUT_SECONDS > 0 && CASE_CONTRACT_MAX_TOKENS > 0 && CASE_CONTRACT_MAX_TOKENS < VLLM_MAX_MODEL_LEN)) || \
  { printf 'ERROR: invalid startup/token settings\n' >&2; exit 2; }
for executable in curl setsid flock; do
  command -v "$executable" >/dev/null || { printf 'ERROR: missing %s\n' "$executable" >&2; exit 2; }
done
selection=()
[[ $stage != all ]] || selection=(--all)
"${client[@]}" -m semantic_extraction.case_contract.runner prepare \
  --prepared-dir "$input_dir" --output "$output/extraction" \
  --model "$QWEN_SERVED_MODEL_NAME" --model-fingerprint "${QWEN_MODEL_FINGERPRINT_SHA256:-}" \
  --max-tokens "$CASE_CONTRACT_MAX_TOKENS" --base-url "http://127.0.0.1:$VLLM_PORT/v1" \
  "${selection[@]}"

# Snapshot effective non-secret settings without overwriting the existing private .env.
effective_env=$output/runtime.env
CONDA_EXTRACT_PREFIX=$serve_prefix
for variable in CONDA_EXTRACT_PREFIX QWEN_MODEL_PATH QWEN_MODEL_FINGERPRINT_SHA256 \
    QWEN_SERVED_MODEL_NAME GPU_ID VLLM_HOST VLLM_PORT VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS VLLM_CACHE_PATH; do
  [[ ! -v $variable ]] || printf '%s=%q\n' "$variable" "${!variable}" >> "$effective_env"
done
mkdir -p "$PROJECT_ROOT/run-records/case-contract-v1"
exec 9>"$PROJECT_ROOT/run-records/case-contract-v1/port-$VLLM_PORT.lock"
flock -n 9 || { printf 'ERROR: another contract task owns this port\n' >&2; exit 2; }
if (exec 3<>"/dev/tcp/127.0.0.1/$VLLM_PORT") 2>/dev/null; then
  printf 'ERROR: port is occupied; existing service was not touched\n' >&2; exit 2
fi
vllm_pid=
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n $vllm_pid ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill -TERM -- "-$vllm_pid" 2>/dev/null || kill -TERM "$vllm_pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$vllm_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$output/job.status"
  # Bind the environment/configuration and final service log as well as extraction artifacts.
  "$base_python" - "$output" "$status" <<'PY' || status=1
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifacts = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name != "job-manifest.json"
}
with (root / "job-manifest.json").open("x", encoding="utf-8") as handle:
    json.dump({"exit_code": int(sys.argv[2]), "artifacts": artifacts}, handle, indent=2)
PY
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'started_utc=%s\nstage=%s\n' "$(date -u +%FT%TZ)" "$stage" > "$output/job.status"
RAG_ENV_FILE=$effective_env setsid bash deploy/run-qwen3-vllm.sh > "$output/vllm.log" 2>&1 &
vllm_pid=$!
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
while ! curl --silent --fail --max-time 5 --noproxy '*' "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; do
  if ! kill -0 "$vllm_pid" 2>/dev/null || ((SECONDS >= deadline)); then
    printf 'ERROR: model service startup failed; inspect the private vllm.log\n' >&2
    exit 2
  fi
  sleep 5
done
"${client[@]}" -m semantic_extraction.case_contract.runner run --run-dir "$output/extraction"
