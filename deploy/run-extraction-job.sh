#!/usr/bin/env bash
set -euo pipefail

# Never expose sourced secrets or command arguments through shell tracing.
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}
RUN_VLLM=$PROJECT_ROOT/deploy/run-vllm.sh
RUN_EXTRACTION=$PROJECT_ROOT/deploy/run-extraction.sh
VERIFY_LISTENER_OWNER=$PROJECT_ROOT/deploy/verify-listener-owner.py
DEFAULT_QWEN_MODEL_PATH=$PROJECT_ROOT/models/Qwen3-30B-A3B

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

usage() {
  printf '%s\n' \
    'Usage: bash deploy/run-extraction-job.sh [--full] [main.py options]' \
    'Without --limit, the wrapper adds --limit 1 --concurrency 1.' \
    'A bounded pilot accepts at most --limit 100.' \
    'A full pass requires: --full --resume [--concurrency N]'
}

for argument in "$@"; do
  if [[ $argument == -h || $argument == --help ]]; then
    usage
    exit 0
  fi
done

[[ -r $ENV_FILE ]] || die "deployment environment is not readable: $ENV_FILE"
set +u
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
set -u

: "${CONDA_EXTRACT_ENV:=civic-rag-extract}"
: "${QWEN_MODEL_PATH:=$DEFAULT_QWEN_MODEL_PATH}"
: "${QWEN_SERVED_MODEL_NAME:=Qwen3-30B-A3B}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_STARTUP_TIMEOUT_SECONDS:=1800}"
: "${VLLM_HEALTH_POLL_SECONDS:=5}"
: "${VLLM_HEALTH_FAILURE_LIMIT:=6}"
: "${VLLM_SHUTDOWN_TIMEOUT_SECONDS:=30}"
: "${RAG_RUN_RECORDS_PATH:=$PROJECT_ROOT/run-records}"
: "${RAG_JOB_LOG_DIR:=$RAG_RUN_RECORDS_PATH/extraction-jobs}"
: "${RAG_JOB_LOCK_PATH:=$RAG_RUN_RECORDS_PATH/extraction-job.lock}"
: "${RAG_INPUT_PATH:=$PROJECT_ROOT/data/raw/t_order_master.sanitized.v1_9.tsv}"
: "${RAG_INPUT_SHA256:=1b778548b618de5f051e749232e47648323979898397588ac79b53d2123aac0c}"
: "${RAG_INPUT_SIZE_BYTES:=655769345}"

export CONDA_EXTRACT_ENV
export QWEN_MODEL_PATH
export QWEN_SERVED_MODEL_NAME
export QWEN_MODEL_FINGERPRINT_SHA256
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
unset RAG_MODEL_FINGERPRINT_VERIFIED

[[ $VLLM_HOST == 127.0.0.1 ]] || die "VLLM_HOST must remain 127.0.0.1"
[[ $VLLM_PORT == 8000 ]] || \
  die "VLLM_PORT must remain 8000 to match configs/config.yaml"
[[ $QWEN_SERVED_MODEL_NAME == Qwen3-30B-A3B ]] || \
  die "QWEN_SERVED_MODEL_NAME must remain Qwen3-30B-A3B"
if [[ -n ${VLLM_API_KEY:-} && \
  ! ${VLLM_API_KEY:-} =~ ^[A-Za-z0-9_-]{32,128}$ ]]; then
  die "VLLM_API_KEY must be empty or a 32-128 character URL-safe token"
fi
[[ $VLLM_STARTUP_TIMEOUT_SECONDS =~ ^[0-9]+$ ]] && ((VLLM_STARTUP_TIMEOUT_SECONDS > 0)) || \
  die "VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer"
[[ $VLLM_HEALTH_POLL_SECONDS =~ ^[0-9]+$ ]] && ((VLLM_HEALTH_POLL_SECONDS > 0)) || \
  die "VLLM_HEALTH_POLL_SECONDS must be a positive integer"
[[ $VLLM_HEALTH_FAILURE_LIMIT =~ ^[0-9]+$ ]] && ((VLLM_HEALTH_FAILURE_LIMIT > 0)) || \
  die "VLLM_HEALTH_FAILURE_LIMIT must be a positive integer"
[[ $VLLM_SHUTDOWN_TIMEOUT_SECONDS =~ ^[0-9]+$ ]] && ((VLLM_SHUTDOWN_TIMEOUT_SECONDS > 0)) || \
  die "VLLM_SHUTDOWN_TIMEOUT_SECONDS must be a positive integer"
[[ $RAG_JOB_LOG_DIR == /* ]] || die "RAG_JOB_LOG_DIR must be an absolute path"
[[ $RAG_JOB_LOCK_PATH == /* ]] || die "RAG_JOB_LOCK_PATH must be an absolute path"
[[ -r $RUN_VLLM ]] || die "vLLM launcher is missing: $RUN_VLLM"
[[ -r $RUN_EXTRACTION ]] || die "extraction launcher is missing: $RUN_EXTRACTION"
[[ -r $VERIFY_LISTENER_OWNER ]] || \
  die "listener ownership verifier is missing: $VERIFY_LISTENER_OWNER"
command -v curl >/dev/null 2>&1 || die "curl is required for vLLM health checks"
command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"
command -v git >/dev/null 2>&1 || die "git is not available on PATH"
command -v realpath >/dev/null 2>&1 || die "realpath is required for path validation"
command -v setsid >/dev/null 2>&1 || \
  die "setsid is required so all vLLM and extraction subprocesses can be cleaned up"
setsid --help 2>&1 | grep -q -- '--fork' || \
  die "setsid must support --fork"
setsid --help 2>&1 | grep -q -- '--wait' || \
  die "setsid must support --wait"
command -v flock >/dev/null 2>&1 || die "flock is required for exclusive H100 job ownership"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required for input validation"
grep -q -- '--disable-log-requests' "$RUN_VLLM" || \
  die "refusing to start vLLM because request logging is not disabled"
grep -q -- '--disable-uvicorn-access-log' "$RUN_VLLM" || \
  die "refusing to start vLLM because access logging is not disabled"
[[ ${QWEN_MODEL_FINGERPRINT_SHA256:-} =~ ^sha256:[0-9a-fA-F]{64}$ ]] || \
  die "QWEN_MODEL_FINGERPRINT_SHA256 must have the form sha256:<64hex>"
[[ -n ${QWEN_MODELSCOPE_REPO_ID:-} ]] || \
  die "QWEN_MODELSCOPE_REPO_ID must record the ModelScope repository"
[[ -n ${QWEN_MODEL_REVISION:-} ]] || \
  die "QWEN_MODEL_REVISION must be non-empty or 'unknown'"
if ! listener_python=$(conda run -n "$CONDA_EXTRACT_ENV" \
  python -c 'import sys; print(sys.executable)'); then
  die "cannot resolve Python from Conda environment: $CONDA_EXTRACT_ENV"
fi
[[ -n $listener_python && $listener_python != *$'\n'* && $listener_python == /* ]] || \
  die "Conda returned an invalid Python executable path"
[[ -x $listener_python ]] || die "Conda Python is not executable: $listener_python"

extraction_args=()
full_run=no
has_limit=no
limit_value=
has_resume=no
arguments=("$@")
for ((argument_index = 0; argument_index < ${#arguments[@]}; argument_index++)); do
  argument=${arguments[$argument_index]}
  case $argument in
    --full)
      full_run=yes
      ;;
    --full=*)
      die "--full does not accept a value"
      ;;
    --limit)
      [[ $has_limit == no ]] || die "--limit may only be provided once"
      has_limit=yes
      extraction_args+=("$argument")
      argument_index=$((argument_index + 1))
      ((argument_index < ${#arguments[@]})) || die "--limit requires a value"
      limit_value=${arguments[$argument_index]}
      extraction_args+=("$limit_value")
      ;;
    --limit=*)
      [[ $has_limit == no ]] || die "--limit may only be provided once"
      has_limit=yes
      limit_value=${argument#--limit=}
      extraction_args+=("$argument")
      ;;
    --resume)
      has_resume=yes
      extraction_args+=("$argument")
      ;;
    *)
      extraction_args+=("$argument")
      ;;
  esac
done

if [[ $has_limit == yes ]]; then
  [[ $limit_value =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"
fi

default_smoke=no
if [[ $full_run == yes ]]; then
  [[ $has_limit == no ]] || die "--full and --limit are mutually exclusive"
  [[ $has_resume == yes ]] || die "--full requires --resume after the bounded pilots"
  export RAG_ALLOW_FULL_EXTRACTION=1
elif [[ $has_limit == no ]]; then
  extraction_args+=(--limit 1 --concurrency 1)
  default_smoke=yes
elif ((limit_value > 100)); then
  die "bounded pilots accept at most --limit 100; use --full --resume for the full pass"
fi
export RAG_JOB_WRAPPER_ACTIVE=1

if ! conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python - "$PROJECT_ROOT" "${extraction_args[@]}" <<'PY'
import sys

project_root, *arguments = sys.argv[1:]
sys.path.insert(0, project_root)

import main

parser = main.build_parser()
args = parser.parse_args(arguments)
if args.limit is not None and args.limit < 0:
    parser.error("--limit must be non-negative")
if args.concurrency is not None and args.concurrency < 1:
    parser.error("--concurrency must be positive")
if args.retry_failures and not args.resume:
    parser.error("--retry-failures requires --resume")
PY
then
  die "invalid extraction arguments"
fi

config_path=$PROJECT_ROOT/configs/config.yaml
for ((argument_index = 0; argument_index < ${#extraction_args[@]}; argument_index++)); do
  argument=${extraction_args[$argument_index]}
  case $argument in
    --config)
      argument_index=$((argument_index + 1))
      ((argument_index < ${#extraction_args[@]})) || die "--config requires a path"
      config_path=${extraction_args[$argument_index]}
      ;;
    --config=*)
      config_path=${argument#--config=}
      ;;
  esac
done
[[ -n $config_path ]] || die "--config path cannot be empty"
if [[ $config_path != /* ]]; then
  config_path=$PROJECT_ROOT/$config_path
fi

git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
git_branch=$(git -C "$PROJECT_ROOT" symbolic-ref --quiet --short HEAD 2>/dev/null || printf detached)
mapfile -t git_changes < <(
  git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=normal 2>/dev/null
)
[[ $git_commit =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  die "repository HEAD is not a valid Git commit"
((${#git_changes[@]} == 0)) || die "repository worktree must be clean before extraction"
export RAG_CODE_COMMIT=$git_commit

umask 077
mkdir -p -- "$RAG_JOB_LOG_DIR"
[[ -d $RAG_JOB_LOG_DIR && -w $RAG_JOB_LOG_DIR ]] || \
  die "job log directory is not writable: $RAG_JOB_LOG_DIR"
mkdir -p -- "$(dirname -- "$RAG_JOB_LOCK_PATH")"
if ! exec {job_lock_fd}> "$RAG_JOB_LOCK_PATH"; then
  die "cannot open the extraction job lock: $RAG_JOB_LOCK_PATH"
fi
if ! flock -n "$job_lock_fd"; then
  die "another extraction job holds the H100 job lock: $RAG_JOB_LOCK_PATH"
fi

job_id=$(date -u '+%Y%m%dT%H%M%SZ')-$$
export RAG_EXTRACTION_RUN_ID=run_$job_id
vllm_log=$RAG_JOB_LOG_DIR/vllm-$job_id.log
extraction_log=$RAG_JOB_LOG_DIR/extraction-$job_id.log
status_log=$RAG_JOB_LOG_DIR/job-$job_id.status
: > "$vllm_log"
: > "$extraction_log"
: > "$status_log"

vllm_pid=
vllm_wait_pid=
vllm_process_group=no
vllm_group_file=$RAG_JOB_LOG_DIR/vllm-$job_id.pgid
extraction_pid=
extraction_wait_pid=
extraction_process_group=no
extraction_group_file=$RAG_JOB_LOG_DIR/extraction-$job_id.pgid

process_is_alive() {
  local pid=$1
  local use_group=$2
  if [[ $use_group == yes ]]; then
    kill -0 -- "-$pid" 2>/dev/null
  else
    kill -0 "$pid" 2>/dev/null
  fi
}

stop_process() {
  local pid=$1
  local use_group=$2
  local label=$3
  local wait_pid=${4:-$pid}
  local deadline

  [[ -n $pid ]] || return 0
  if ! process_is_alive "$pid" "$use_group"; then
    wait "$wait_pid" 2>/dev/null || true
    return 0
  fi

  printf '%s stopping %s pid=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "$pid" \
    >> "$status_log" 2>/dev/null || true
  if [[ $use_group == yes ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi

  deadline=$((SECONDS + VLLM_SHUTDOWN_TIMEOUT_SECONDS))
  while process_is_alive "$pid" "$use_group" && ((SECONDS < deadline)); do
    sleep 1
  done
  if process_is_alive "$pid" "$use_group"; then
    printf '%s force-stopping %s pid=%s after timeout\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "$pid" \
      >> "$status_log" 2>/dev/null || true
    if [[ $use_group == yes ]]; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    else
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$wait_pid" 2>/dev/null || true
}

start_process_group() {
  local wait_variable=$1
  local group_variable=$2
  local marker_path=$3
  local log_path=$4
  local label=$5
  local require_alive=$6
  local wait_pid
  local process_group
  local pending_start_signal=
  shift 6

  trap 'pending_start_signal=130' INT
  trap 'pending_start_signal=143' TERM
  trap 'pending_start_signal=129' HUP
  : > "$marker_path"
  # The setsid waiter remains in the platform's old process group. Ignore
  # group-wide termination only in that child; the wrapper must keep recording
  # cancellation while the new session resets signals before exec.
  (
    trap '' INT TERM HUP
    exec setsid --fork --wait "$listener_python" -c '
import os
import signal
import sys

for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, signal.SIG_DFL)
marker_path, *command = sys.argv[1:]
with open(marker_path, "w", encoding="ascii") as marker:
    marker.write(f"{os.getpid()}\n")
    marker.flush()
    os.fsync(marker.fileno())
os.execvp(command[0], command)
    ' "$marker_path" "$@"
  ) > "$log_path" 2>&1 &
  wait_pid=$!
  printf -v "$wait_variable" '%s' "$wait_pid"

  while [[ ! -s $marker_path ]]; do
    if ! kill -0 "$wait_pid" 2>/dev/null; then
      wait "$wait_pid" 2>/dev/null || true
      die "$label launcher exited before recording its process group"
    fi
    sleep 0.05 || true
  done
  IFS= read -r process_group < "$marker_path" || \
    die "$label process-group marker is unreadable"
  [[ $process_group =~ ^[1-9][0-9]*$ ]] || \
    die "$label process-group marker is invalid"
  printf -v "$group_variable" '%s' "$process_group"
  if [[ $require_alive == yes ]] && ! process_is_alive "$process_group" yes; then
    die "$label process group exited during startup"
  fi
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  if [[ -n $pending_start_signal ]]; then
    exit "$pending_start_signal"
  fi
}

recover_process_group() {
  local group_variable=$1
  local wait_pid=$2
  local marker_path=$3
  local current_group=${!group_variable}
  local recovered_group

  [[ -z $current_group && -n $wait_pid && -s $marker_path ]] || return 0
  kill -0 "$wait_pid" 2>/dev/null || return 0
  IFS= read -r recovered_group < "$marker_path" || return 0
  [[ $recovered_group =~ ^[1-9][0-9]*$ ]] || return 0
  printf -v "$group_variable" '%s' "$recovered_group"
}

stop_managed_group() {
  local process_group=$1
  local wait_pid=$2
  local label=$3

  if [[ -n $process_group ]]; then
    stop_process "$process_group" yes "$label" "$wait_pid"
  elif [[ -n $wait_pid ]]; then
    stop_process "$wait_pid" no "$label launcher" "$wait_pid"
  fi
}

on_exit() {
  local job_rc=$?
  trap - EXIT
  trap '' INT TERM HUP
  recover_process_group extraction_pid "$extraction_wait_pid" "$extraction_group_file"
  recover_process_group vllm_pid "$vllm_wait_pid" "$vllm_group_file"
  stop_managed_group "$extraction_pid" "$extraction_wait_pid" extraction
  stop_managed_group "$vllm_pid" "$vllm_wait_pid" vllm
  : > "$extraction_group_file" 2>/dev/null || true
  : > "$vllm_group_file" 2>/dev/null || true
  printf '%s job_finished exit_code=%d\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$job_rc" \
    >> "$status_log" 2>/dev/null || true
  exit "$job_rc"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s job_started\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
printf 'git_commit=%s\n' "$git_commit" >> "$status_log"
printf 'git_branch=%s\n' "$git_branch" >> "$status_log"
printf 'git_worktree_changes=%d\n' "${#git_changes[@]}" >> "$status_log"
printf 'extraction_run_id=%s\n' "$RAG_EXTRACTION_RUN_ID" >> "$status_log"
printf 'job_lock_path=%s\n' "$RAG_JOB_LOCK_PATH" >> "$status_log"
printf 'default_smoke=%s\n' "$default_smoke" >> "$status_log"
printf 'job_id=%s\n' "$job_id"
printf 'extraction_run_id=%s\n' "$RAG_EXTRACTION_RUN_ID"
printf 'vllm_log=%s\n' "$vllm_log"
printf 'extraction_log=%s\n' "$extraction_log"
printf 'status_log=%s\n' "$status_log"
printf '%s\n' 'Request logging is disabled; CLI arguments and prompt/data content are not written by this wrapper.'

mapfile -t configured_paths < <(
  conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
    python - "$config_path" "$PROJECT_ROOT" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
with config_path.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
data = config["data"]
for key, fallback in (
    ("input", None),
    ("output", None),
    ("quarantine", "data/processed/problem_chunks.errors.jsonl"),
):
    raw_path = data.get(key, fallback)
    if not isinstance(raw_path, str) or not raw_path.strip() or "\n" in raw_path:
        raise SystemExit(f"invalid data.{key} path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    print(path.resolve())
llm = config.get("llm")
model = llm.get("model") if isinstance(llm, dict) else None
if not isinstance(model, str) or not model.strip() or "\n" in model:
    raise SystemExit("invalid llm.model")
if llm.get("base_url") != "http://127.0.0.1:8000/v1":
    raise SystemExit("llm.base_url must remain http://127.0.0.1:8000/v1")
if llm.get("enable_thinking") is not False:
    raise SystemExit("llm.enable_thinking must be false")
if llm.get("api_key_env", "OPENAI_API_KEY") != "VLLM_API_KEY":
    raise SystemExit("llm.api_key_env must remain VLLM_API_KEY")
if llm.get("api_key") is not None:
    raise SystemExit("inline llm.api_key is not allowed")
if llm.get("model_revision_env", "QWEN_MODEL_REVISION") != "QWEN_MODEL_REVISION":
    raise SystemExit("llm.model_revision_env must remain QWEN_MODEL_REVISION")
if llm.get("model_source_repo_env", "QWEN_MODELSCOPE_REPO_ID") != "QWEN_MODELSCOPE_REPO_ID":
    raise SystemExit("llm.model_source_repo_env must remain QWEN_MODELSCOPE_REPO_ID")
for key, environment_name in (
    ("model_revision", "QWEN_MODEL_REVISION"),
    ("model_source_repo", "QWEN_MODELSCOPE_REPO_ID"),
    ("model_artifact_fingerprint", "QWEN_MODEL_FINGERPRINT_SHA256"),
):
    explicit = llm.get(key)
    expected = os.environ.get(environment_name)
    if explicit is not None and (
        not isinstance(explicit, str) or explicit.strip().lower() != (expected or "").lower()
    ):
        raise SystemExit(f"llm.{key} must match {environment_name} when explicitly set")
print(model.strip())
PY
)
((${#configured_paths[@]} == 4)) || die "could not resolve paths/model from the extraction config"
configured_input_path=${configured_paths[0]}
configured_output_path=${configured_paths[1]}
configured_quarantine_path=${configured_paths[2]}
configured_model_name=${configured_paths[3]}
[[ $configured_model_name == "$QWEN_SERVED_MODEL_NAME" ]] || \
  die "llm.model in the extraction config must match QWEN_SERVED_MODEL_NAME"
[[ $RAG_INPUT_PATH == /* ]] || die "RAG_INPUT_PATH must be absolute"
declared_input_path=$(realpath -m -- "$RAG_INPUT_PATH")
[[ $configured_input_path == "$declared_input_path" ]] || \
  die "RAG_INPUT_PATH must exactly match data.input in the selected extraction config"
[[ -f $configured_input_path && -r $configured_input_path ]] || \
  die "sanitized input is not a readable file: $configured_input_path"
[[ $RAG_INPUT_SIZE_BYTES =~ ^[0-9]+$ ]] || \
  die "RAG_INPUT_SIZE_BYTES must be a non-negative integer"
[[ $RAG_INPUT_SHA256 =~ ^[0-9a-fA-F]{64}$ ]] || \
  die "RAG_INPUT_SHA256 must be a 64-character SHA256 value"
input_size=$(stat -c '%s' -- "$configured_input_path")
[[ $input_size == "$RAG_INPUT_SIZE_BYTES" ]] || \
  die "sanitized input size does not match RAG_INPUT_SIZE_BYTES"
input_sha256=$(sha256sum -- "$configured_input_path" | awk '{print $1}')
[[ ${input_sha256,,} == "${RAG_INPUT_SHA256,,}" ]] || \
  die "sanitized input SHA256 does not match RAG_INPUT_SHA256"
printf 'input_path=%s\n' "$configured_input_path" >> "$status_log"
printf 'input_size_bytes=%s\n' "$input_size" >> "$status_log"
printf 'input_sha256=%s\n' "$input_sha256" >> "$status_log"
printf 'input_identity_verified=true\n' >> "$status_log"
printf 'Sanitized input size and SHA256 passed. No record was decoded or printed.\n'

health_url=http://$VLLM_HOST:$VLLM_PORT/health

require_listener_free() {
  local phase=$1
  if ! "$listener_python" "$VERIFY_LISTENER_OWNER" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --expect-free >/dev/null; then
    die "listener address $VLLM_HOST:$VLLM_PORT is occupied $phase"
  fi
}

require_listener_free "before model validation"

listener_report=
probe_owned_listener() {
  local report
  local probe_rc

  if report=$("$listener_python" "$VERIFY_LISTENER_OWNER" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --expected-pgid "$vllm_pid" 2>&1); then
    listener_report=$report
    return 0
  else
    probe_rc=$?
  fi
  if ((probe_rc == 3)); then
    return 1
  fi
  printf '%s listener_owner_check_failed exit_code=%d\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$probe_rc" >> "$status_log"
  die "listener at $VLLM_HOST:$VLLM_PORT is not owned by this job's vLLM process group"
}

printf 'Validating the declared Qwen3 content fingerprint before GPU startup...\n'
if ! fingerprint_report=$(conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python "$PROJECT_ROOT/deploy/model-fingerprint.py" \
  --model-dir "$QWEN_MODEL_PATH" \
  --expect "$QWEN_MODEL_FINGERPRINT_SHA256"); then
  die "Qwen3 snapshot validation or content fingerprint check failed"
fi
printf '%s\n' "$fingerprint_report" >> "$status_log"
export RAG_MODEL_FINGERPRINT_VERIFIED=1
printf '%s model_fingerprint_verified\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"

require_listener_free "after model validation"

printf '%s starting vllm\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
start_process_group \
  vllm_wait_pid \
  vllm_pid \
  "$vllm_group_file" \
  "$vllm_log" \
  vllm \
  yes \
  bash "$RUN_VLLM"
vllm_process_group=yes

printf 'Waiting up to %s seconds for Qwen3/vLLM startup...\n' "$VLLM_STARTUP_TIMEOUT_SECONDS"
startup_deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
vllm_ready=no
while ((SECONDS < startup_deadline)); do
  if ! process_is_alive "$vllm_pid" "$vllm_process_group"; then
    die "vLLM exited before becoming healthy; inspect $vllm_log"
  fi
  if probe_owned_listener && \
    curl --silent --fail --max-time 5 "$health_url" >/dev/null 2>&1; then
    probe_owned_listener || \
      die "vLLM listener ownership changed during its health check"
    vllm_ready=yes
    break
  fi
  sleep "$VLLM_HEALTH_POLL_SECONDS"
done
if [[ $vllm_ready != yes ]] && \
  process_is_alive "$vllm_pid" "$vllm_process_group" && \
  probe_owned_listener && \
  curl --silent --fail --max-time 5 "$health_url" >/dev/null 2>&1; then
  probe_owned_listener || die "vLLM listener ownership changed during its health check"
  vllm_ready=yes
fi
[[ $vllm_ready == yes ]] || die "vLLM startup timed out; inspect $vllm_log"

printf '%s vllm_health_ready\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
printf '%s\n' "$listener_report" >> "$status_log"
printf '%s vllm_listener_owner_verified\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
printf 'vLLM health check passed. Verifying served model alias...\n'

probe_owned_listener || die "vLLM listener disappeared before model alias verification"

VLLM_MODEL_LIST_URL=http://$VLLM_HOST:$VLLM_PORT/v1/models \
EXPECTED_MODEL_NAME=$QWEN_SERVED_MODEL_NAME \
VLLM_API_KEY=${VLLM_API_KEY:-} \
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" python - <<'PY' || \
  die "vLLM does not serve the expected alias; inspect the launcher and $vllm_log"
import json
import os
import urllib.error
import urllib.request

request = urllib.request.Request(os.environ["VLLM_MODEL_LIST_URL"])
api_key = os.environ.get("VLLM_API_KEY")
if api_key:
    request.add_header("Authorization", f"Bearer {api_key}")

try:
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
except (OSError, ValueError, urllib.error.HTTPError):
    raise SystemExit(2)

models = payload.get("data") if isinstance(payload, dict) else None
model_ids = {
    item.get("id")
    for item in models or []
    if isinstance(item, dict) and isinstance(item.get("id"), str)
}
raise SystemExit(0 if os.environ["EXPECTED_MODEL_NAME"] in model_ids else 3)
PY

probe_owned_listener || die "vLLM listener disappeared after model alias verification"

printf '%s model_alias_ready name=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$QWEN_SERVED_MODEL_NAME" >> "$status_log"
printf 'Starting extraction. Follow progress with: tail -f %q\n' "$extraction_log"
printf '%s extraction_started\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"

start_process_group \
  extraction_wait_pid \
  extraction_pid \
  "$extraction_group_file" \
  "$extraction_log" \
  extraction \
  no \
  bash "$RUN_EXTRACTION" "${extraction_args[@]}"
extraction_process_group=yes

vllm_failed_during_extraction=no
health_failure_count=0
while kill -0 "$extraction_wait_pid" 2>/dev/null; do
  if ! process_is_alive "$vllm_pid" "$vllm_process_group"; then
    vllm_failed_during_extraction=yes
    printf '%s vllm_exited_during_extraction\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
    stop_process "$extraction_pid" yes extraction "$extraction_wait_pid"
    : > "$extraction_group_file"
    extraction_pid=
    extraction_wait_pid=
    break
  fi
  if probe_owned_listener && \
    curl --silent --fail --max-time 3 "$health_url" >/dev/null 2>&1 && \
    probe_owned_listener; then
    health_failure_count=0
  else
    health_failure_count=$((health_failure_count + 1))
    printf '%s vllm_health_failure consecutive=%d limit=%d\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "$health_failure_count" "$VLLM_HEALTH_FAILURE_LIMIT" >> "$status_log"
    if ((health_failure_count >= VLLM_HEALTH_FAILURE_LIMIT)); then
      # The extraction may have completed during the final health timeout.
      if ! kill -0 "$extraction_wait_pid" 2>/dev/null; then
        break
      fi
      vllm_failed_during_extraction=yes
      printf '%s vllm_unhealthy_during_extraction\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
      stop_process "$extraction_pid" yes extraction "$extraction_wait_pid"
      : > "$extraction_group_file"
      extraction_pid=
      extraction_wait_pid=
      break
    fi
  fi
  sleep "$VLLM_HEALTH_POLL_SECONDS"
done

if [[ $vllm_failed_during_extraction == yes ]]; then
  extraction_rc=2
else
  set +e
  wait "$extraction_wait_pid"
  extraction_rc=$?
  set -e
  if process_is_alive "$extraction_pid" yes; then
    stop_process \
      "$extraction_pid" yes "extraction descendants" "$extraction_wait_pid"
  fi
  : > "$extraction_group_file"
  extraction_pid=
  extraction_wait_pid=
  if ! process_is_alive "$vllm_pid" "$vllm_process_group"; then
    printf '%s vllm_exited_at_extraction_completion\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$status_log"
    extraction_rc=2
  fi
fi
output_count=0
quarantine_count=0
if [[ -f $configured_output_path ]]; then
  output_count=$(wc -l < "$configured_output_path")
fi
if [[ -f $configured_quarantine_path ]]; then
  quarantine_count=$(wc -l < "$configured_quarantine_path")
fi
printf 'output_record_count=%s\n' "$output_count" >> "$status_log"
printf 'quarantine_record_count=%s\n' "$quarantine_count" >> "$status_log"
printf '%s extraction_finished exit_code=%d\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$extraction_rc" >> "$status_log"
printf 'git_commit=%s output_records=%s quarantine_records=%s\n' \
  "$git_commit" "$output_count" "$quarantine_count"
printf 'Extraction finished with exit code %d. Stopping vLLM...\n' "$extraction_rc"

# on_exit stops only the process group created by this job and preserves this code.
exit "$extraction_rc"
