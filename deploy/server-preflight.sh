#!/usr/bin/env bash
set -uo pipefail

# Keep secrets from deploy/.env out of traces even if the caller used bash -x.
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}
DEFAULT_QWEN_MODEL_PATH=$PROJECT_ROOT/models/Qwen3-30B-A3B
readonly LAUNCH_CODE_COMMIT=${RAG_CODE_COMMIT-}
readonly LAUNCH_CODE_BRANCH=${RAG_CODE_BRANCH-}

failures=0
warnings=0

section() {
  printf '\n== %s ==\n' "$1"
}

pass() {
  printf 'PASS: %s\n' "$1"
}

warn() {
  warnings=$((warnings + 1))
  printf 'WARN: %s\n' "$1"
}

fail() {
  failures=$((failures + 1))
  printf 'FAIL: %s\n' "$1"
}

human_bytes() {
  local bytes=$1
  local units=(B KiB MiB GiB TiB)
  local unit=0
  while ((bytes >= 1024 && unit < ${#units[@]} - 1)); do
    bytes=$((bytes / 1024))
    unit=$((unit + 1))
  done
  printf '%s %s' "$bytes" "${units[$unit]}"
}

nearest_existing_path() {
  local candidate=$1
  while [[ ! -e $candidate && $candidate != / ]]; do
    candidate=$(dirname -- "$candidate")
  done
  printf '%s' "$candidate"
}

report_path() {
  local label=$1
  local path=$2
  local required=${3:-no}
  local probe
  local disk_summary
  local mount_summary

  printf '%s=%s\n' "$label" "$path"
  if [[ -e $path ]]; then
    pass "$label exists"
    [[ -r $path ]] || fail "$label is not readable"
    if [[ -w $path ]]; then
      printf '%s_access=readable,writable\n' "$label"
    else
      printf '%s_access=readable,not-writable\n' "$label"
    fi
    probe=$path
  else
    if [[ $required == yes ]]; then
      fail "$label does not exist"
    else
      warn "$label does not exist yet"
    fi
    probe=$(nearest_existing_path "$path")
    printf '%s_nearest_existing_parent=%s\n' "$label" "$probe"
    [[ -w $probe ]] || warn "$label cannot be created by the current user under $probe"
  fi

  if disk_summary=$(df -hP -- "$probe" 2>/dev/null | awk 'NR == 2 {print "size=" $2 ", used=" $3 ", available=" $4 ", use=" $5}'); then
    printf '%s_disk=%s\n' "$label" "$disk_summary"
  else
    warn "could not inspect disk capacity for $label"
  fi
  if command -v findmnt >/dev/null 2>&1; then
    mount_summary=$(findmnt -T "$probe" -n -o TARGET,FSTYPE 2>/dev/null || true)
    [[ -n $mount_summary ]] && printf '%s_mount=%s\n' "$label" "$mount_summary"
  fi
}

section "Environment file"
if [[ -r $ENV_FILE ]]; then
  set +u
  set -a
  # shellcheck disable=SC1090
  if source "$ENV_FILE"; then
    pass "loaded deployment settings from $ENV_FILE"
  else
    fail "could not load deployment settings from $ENV_FILE"
  fi
  set +a
  set -u
elif [[ -n ${RAG_ENV_FILE:-} ]]; then
  fail "RAG_ENV_FILE is not readable: $ENV_FILE"
else
  warn "deployment settings not found at $ENV_FILE; using diagnostic defaults"
fi

# Provenance is launch-scoped. Do not accept values that may have been left in
# the persistent .env by an older checkout.
RAG_CODE_COMMIT=$LAUNCH_CODE_COMMIT
RAG_CODE_BRANCH=$LAUNCH_CODE_BRANCH

QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-$DEFAULT_QWEN_MODEL_PATH}
QWEN_SERVED_MODEL_NAME=${QWEN_SERVED_MODEL_NAME:-Qwen3-30B-A3B}
CONDA_EXTRACT_ENV=${CONDA_EXTRACT_ENV:-civic-rag-extract}
GPU_ID=${GPU_ID:-0}
GPU_VISIBILITY=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
VLLM_CACHE_PATH=${VLLM_CACHE_PATH:-$PROJECT_ROOT/.cache/vllm}
RAG_RUN_RECORDS_PATH=${RAG_RUN_RECORDS_PATH:-$PROJECT_ROOT/run-records}
RAG_JOB_LOG_DIR=${RAG_JOB_LOG_DIR:-$RAG_RUN_RECORDS_PATH/extraction-jobs}
RAG_INPUT_PATH=${RAG_INPUT_PATH:-$PROJECT_ROOT/data/raw/t_order_master.sanitized.v1_9.tsv}
RAG_INPUT_SHA256=${RAG_INPUT_SHA256:-1b778548b618de5f051e749232e47648323979898397588ac79b53d2123aac0c}
RAG_INPUT_SIZE_BYTES=${RAG_INPUT_SIZE_BYTES:-655769345}
RAG_OUTPUT_DIR=${RAG_OUTPUT_DIR:-$PROJECT_ROOT/data/processed}

# No command in this diagnostic may fall back to a model hub.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1

section "Platform"
printf 'timestamp_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname=%s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'kernel=%s\n' "$(uname -srmo)"
printf 'architecture=%s\n' "$(uname -m)"
printf 'online_cpus=%s\n' "$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf unknown)"
if [[ -r /etc/os-release ]]; then
  os_name=$(
    set +u
    # shellcheck disable=SC1091
    source /etc/os-release
    printf '%s' "${PRETTY_NAME:-unknown}"
  )
  printf 'os=%s\n' "$os_name"
else
  warn "/etc/os-release is not readable"
fi
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-not-set}"

section "Code provenance"
code_provenance_valid=yes
if [[ $RAG_CODE_COMMIT =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  printf 'declared_git_commit=%s\n' "$RAG_CODE_COMMIT"
  pass "launch command supplied a full lowercase commit"
else
  fail "RAG_CODE_COMMIT must be supplied by the launch command as a full lowercase 40- or 64-character hexadecimal commit"
  code_provenance_valid=no
fi
if [[ ${#RAG_CODE_BRANCH} -le 255 && \
  $RAG_CODE_BRANCH =~ ^[A-Za-z0-9_][A-Za-z0-9._/-]*$ && \
  $RAG_CODE_BRANCH != HEAD && \
  $RAG_CODE_BRANCH != *".."* && \
  $RAG_CODE_BRANCH != *"//"* && \
  $RAG_CODE_BRANCH != *"@{"* && \
  $RAG_CODE_BRANCH != */.* && \
  $RAG_CODE_BRANCH != *.lock/* && \
  $RAG_CODE_BRANCH != */ && \
  $RAG_CODE_BRANCH != *. && \
  $RAG_CODE_BRANCH != *.lock ]]; then
  printf 'declared_git_branch=%s\n' "$RAG_CODE_BRANCH"
  pass "launch command supplied a safe branch name"
else
  fail "RAG_CODE_BRANCH must be supplied by the launch command as a safe non-empty branch name"
  code_provenance_valid=no
fi

git_runtime_available=false
git_commit_verified=false
git_branch_verified=false
git_worktree_verified=false
git_worktree_changes=unknown
if command -v git >/dev/null 2>&1 && git --version >/dev/null 2>&1; then
  git_runtime_available=true
  printf 'git_version=%s\n' "$(git --version)"
  if git_root=$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null); then
    if git_root=$(cd -P -- "$git_root" && pwd -P); then
      printf 'repository_root=%s\n' "$git_root"
    else
      fail "Git repository root could not be resolved"
      git_root=
    fi
    if [[ -n $git_root && $git_root != "$PROJECT_ROOT" ]]; then
      if [[ -e $PROJECT_ROOT/.git ]]; then
        fail "local Git metadata does not resolve to the project root"
      else
        warn "git found only ancestor repository metadata; project worktree checks are unverified"
      fi
    elif [[ -n $git_root ]]; then
      if git_head=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null); then
        printf 'git_head=%s\n' "$git_head"
        if [[ $code_provenance_valid == yes && $git_head == "$RAG_CODE_COMMIT" ]]; then
          git_commit_verified=true
          pass "declared commit matches repository HEAD"
        elif [[ $code_provenance_valid == yes ]]; then
          fail "RAG_CODE_COMMIT does not match repository HEAD"
        fi
      else
        fail "Git repository HEAD could not be read"
      fi

      if git_ref=$(git -C "$PROJECT_ROOT" symbolic-ref --quiet HEAD 2>/dev/null); then
        if [[ $git_ref == refs/heads/* ]]; then
          git_branch=${git_ref#refs/heads/}
          printf 'git_branch=%s\n' "$git_branch"
          if [[ $code_provenance_valid == yes && $git_branch == "$RAG_CODE_BRANCH" ]]; then
            git_branch_verified=true
            pass "declared branch matches the checked-out repository branch"
          elif [[ $code_provenance_valid == yes ]]; then
            fail "RAG_CODE_BRANCH does not match the checked-out repository branch"
          fi
        else
          fail "Git repository HEAD does not reference a local branch under refs/heads"
        fi
      else
        fail "Git repository is detached or its branch could not be read"
      fi

      if git_status=$(
        git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=normal 2>/dev/null
      ); then
        git_changes=()
        if [[ -n $git_status ]]; then
          mapfile -t git_changes <<< "$git_status"
        fi
        git_worktree_changes=${#git_changes[@]}
        if ((git_worktree_changes == 0)); then
          git_worktree_verified=true
          pass "Git worktree is clean"
        else
          fail "Git worktree has tracked or untracked changes"
        fi
      else
        fail "Git worktree status could not be read"
      fi
      if git -C "$PROJECT_ROOT" remote get-url origin >/dev/null 2>&1; then
        pass "Git origin is configured (URL suppressed)"
      else
        warn "Git origin is not configured"
      fi
    fi
  elif [[ -e $PROJECT_ROOT/.git ]]; then
    fail "Git metadata exists, but Git could not inspect the project worktree"
  elif [[ $code_provenance_valid == yes ]]; then
    warn "project has no local Git metadata; accepting the launch declaration without worktree verification"
  fi
elif [[ $code_provenance_valid == yes ]]; then
  warn "git is unavailable on this runtime; accepting the launch declaration without worktree verification"
fi
printf 'code_provenance_source=launch_environment\n'
printf 'git_runtime_available=%s\n' "$git_runtime_available"
printf 'git_commit_verified=%s\n' "$git_commit_verified"
printf 'git_branch_verified=%s\n' "$git_branch_verified"
printf 'git_worktree_verified=%s\n' "$git_worktree_verified"
printf 'git_worktree_changes=%s\n' "$git_worktree_changes"

section "Conda"
if command -v conda >/dev/null 2>&1; then
  printf 'conda_command=%s\n' "$(command -v conda)"
  printf 'conda_version=%s\n' "$(conda --version 2>&1)"
  printf 'conda_base=%s\n' "$(conda info --base 2>/dev/null || printf unknown)"
  if extract_runtime=$(conda run -n "$CONDA_EXTRACT_ENV" python -c \
    'import importlib.metadata as m, platform; print("python=" + platform.python_version() + ", vllm=" + m.version("vllm"))' \
    2>/dev/null); then
    printf 'extract_environment=%s\n' "$CONDA_EXTRACT_ENV"
    printf 'extract_runtime=%s\n' "$extract_runtime"
    pass "Conda extraction environment is runnable"
    if runtime_gate=$(CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" \
      conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
      python "$PROJECT_ROOT/deploy/validate-runtime.py" extract 2>&1); then
      printf '%s\n' "$runtime_gate"
      pass "PyTorch, CUDA, full H100 visibility, and vLLM native extensions passed"
    else
      printf '%s\n' "$runtime_gate"
      fail "Stage A H100 runtime gate failed"
    fi
  else
    fail "Conda extraction environment is missing or vLLM is not installed: $CONDA_EXTRACT_ENV"
  fi
else
  fail "conda is not available on PATH"
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_inventory=$(nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu,mig.mode.current \
    --format=csv,noheader 2>/dev/null); then
    printf 'gpu_columns=index, name, uuid, driver, memory_total, memory_free, utilization, mig_mode\n'
    printf '%s\n' "$gpu_inventory"
    pass "NVIDIA driver can query the allocated GPU"
  elif gpu_inventory=$(nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader 2>/dev/null); then
    printf 'gpu_columns=index, name, uuid, driver, memory_total, memory_free, utilization\n'
    printf '%s\n' "$gpu_inventory"
    warn "GPU query succeeded, but MIG mode could not be queried"
  else
    fail "nvidia-smi could not query a GPU"
  fi
  if compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); then
    compute_count=0
    while IFS= read -r compute_pid; do
      [[ -n $compute_pid ]] && compute_count=$((compute_count + 1))
    done <<< "$compute_pids"
    printf 'active_compute_process_count=%d\n' "$compute_count"
    ((compute_count == 0)) || warn "the allocated GPU already has active compute processes"
  fi
else
  fail "nvidia-smi is not available on PATH"
fi

section "Host memory"
if command -v free >/dev/null 2>&1; then
  free -h
else
  warn "free is not available; showing MemTotal and MemAvailable only"
  awk '/^(MemTotal|MemAvailable):/ {print}' /proc/meminfo 2>/dev/null || true
fi
printf 'open_file_limit=%s\n' "$(ulimit -n)"
report_path shared_memory /dev/shm yes

section "Persistent-path candidates"
printf '%s\n' 'NOTE: mount metadata cannot prove retention after a paid task ends; confirm retention in the platform settings.'
report_path project_root "$PROJECT_ROOT" yes
report_path qwen_model_path "$QWEN_MODEL_PATH" yes
report_path input_tsv "$RAG_INPUT_PATH" yes
report_path output_directory "$RAG_OUTPUT_DIR" no
report_path vllm_cache_path "$VLLM_CACHE_PATH" no
report_path run_records_path "$RAG_RUN_RECORDS_PATH" no
report_path job_log_directory "$RAG_JOB_LOG_DIR" no

section "Sanitized input identity"
if [[ -f $RAG_INPUT_PATH && -r $RAG_INPUT_PATH ]]; then
  input_size=$(stat -c '%s' -- "$RAG_INPUT_PATH" 2>/dev/null || printf unknown)
  printf 'input_tsv_size_bytes=%s\n' "$input_size"
  printf 'input_tsv_expected_size_bytes=%s\n' "$RAG_INPUT_SIZE_BYTES"
  if [[ $input_size == "$RAG_INPUT_SIZE_BYTES" ]]; then
    pass "sanitized TSV size matches"
  else
    fail "sanitized TSV size does not match the expected file"
  fi

  if [[ ! $RAG_INPUT_SHA256 =~ ^[0-9a-fA-F]{64}$ ]]; then
    fail "RAG_INPUT_SHA256 must be a 64-character SHA256 value"
  elif ! command -v sha256sum >/dev/null 2>&1; then
    fail "sha256sum is required to identify the sanitized input"
  else
    input_sha256=$(sha256sum -- "$RAG_INPUT_PATH" | awk '{print $1}')
    printf 'input_tsv_sha256=%s\n' "$input_sha256"
    printf 'input_tsv_expected_sha256=%s\n' "${RAG_INPUT_SHA256,,}"
    if [[ ${input_sha256,,} == "${RAG_INPUT_SHA256,,}" ]]; then
      pass "sanitized TSV SHA256 matches"
    else
      fail "sanitized TSV SHA256 does not match the expected file"
    fi
  fi
  pass "input was hashed as raw bytes; no record was decoded or printed"
else
  fail "sanitized input identity cannot be checked because the file is unreadable"
fi

section "Qwen3 model files"
printf 'qwen_served_model_name=%s\n' "$QWEN_SERVED_MODEL_NAME"
printf 'qwen_model_path=%s\n' "$QWEN_MODEL_PATH"
if [[ -n ${QWEN_MODELSCOPE_REPO_ID:-} ]]; then
  printf 'qwen_modelscope_repo_id=%s\n' "$QWEN_MODELSCOPE_REPO_ID"
  pass "QWEN_MODELSCOPE_REPO_ID records the model source repository"
else
  fail "QWEN_MODELSCOPE_REPO_ID must record the ModelScope source repository"
fi
if [[ $QWEN_MODEL_PATH != /* ]]; then
  fail "QWEN_MODEL_PATH must be absolute"
fi
if [[ -d $QWEN_MODEL_PATH && -r $QWEN_MODEL_PATH ]]; then
  pass "model directory is readable"
else
  fail "model directory is not readable"
fi

required_model_files=(config.json)
optional_model_files=(
  generation_config.json
  tokenizer_config.json
  tokenizer.json
  model.safetensors.index.json
)
for model_file in "${required_model_files[@]}"; do
  if [[ -r $QWEN_MODEL_PATH/$model_file ]]; then
    pass "model file exists: $model_file"
  else
    fail "required model file is missing or unreadable: $model_file"
  fi
done
for model_file in "${optional_model_files[@]}"; do
  if [[ -r $QWEN_MODEL_PATH/$model_file ]]; then
    pass "model file exists: $model_file"
  else
    warn "optional/common model file is absent: $model_file"
  fi
done

shopt -s nullglob
weight_files=("$QWEN_MODEL_PATH"/*.safetensors)
shopt -u nullglob
weight_bytes=0
for weight_file in "${weight_files[@]}"; do
  file_bytes=$(stat -c '%s' -- "$weight_file" 2>/dev/null || printf 0)
  [[ $file_bytes =~ ^[0-9]+$ ]] || file_bytes=0
  weight_bytes=$((weight_bytes + file_bytes))
done
printf 'safetensors_file_count=%d\n' "${#weight_files[@]}"
printf 'safetensors_total_bytes=%d\n' "$weight_bytes"
printf 'safetensors_total_approx=%s\n' "$(human_bytes "$weight_bytes")"
if ((${#weight_files[@]} == 0)); then
  fail "no top-level safetensors weights were found"
else
  pass "safetensors weights are present"
fi

broken_link_count=$(find "$QWEN_MODEL_PATH" -maxdepth 1 -xtype l -print 2>/dev/null | awk 'END {print NR + 0}')
printf 'model_broken_symlink_count=%s\n' "$broken_link_count"
if [[ $broken_link_count == 0 ]]; then
  pass "model directory has no broken top-level symlinks"
else
  fail "model directory contains broken top-level symlinks"
fi

if [[ -z ${QWEN_MODEL_REVISION:-} ]]; then
  fail "QWEN_MODEL_REVISION must record a non-empty ModelScope source revision"
elif [[ $QWEN_MODEL_REVISION == unknown ]]; then
  printf 'qwen_model_revision=%s\n' "$QWEN_MODEL_REVISION"
  warn "QWEN_MODEL_REVISION is unknown; preserve the strong content fingerprint instead"
else
  printf 'qwen_model_revision=%s\n' "$QWEN_MODEL_REVISION"
  pass "QWEN_MODEL_REVISION records a ModelScope source revision"
fi

section "Qwen3 config structure"
json_python=()
if command -v conda >/dev/null 2>&1 && \
  conda run -n "$CONDA_EXTRACT_ENV" python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
    >/dev/null 2>&1; then
  json_python=(conda run -n "$CONDA_EXTRACT_ENV" python)
elif command -v python3 >/dev/null 2>&1 && \
  python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
  json_python=(python3)
elif command -v python >/dev/null 2>&1 && \
  python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
  json_python=(python)
fi

if ((${#json_python[@]} == 0)); then
  fail "no Python interpreter is available for safe JSON metadata inspection"
elif [[ ! -r $QWEN_MODEL_PATH/config.json ]]; then
  fail "config.json cannot be summarized because it is unreadable"
else
  if ! "${json_python[@]}" - "$QWEN_MODEL_PATH/config.json" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
except (OSError, UnicodeError, json.JSONDecodeError):
    print("config_json_status=invalid")
    raise SystemExit(2)

if not isinstance(config, dict):
    print("config_json_status=not-an-object")
    raise SystemExit(2)

print("config_json_status=valid")
print(f"top_level_key_count={len(config)}")

scalar_keys = (
    "model_type",
    "torch_dtype",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "decoder_sparse_step",
    "tie_word_embeddings",
)
for key in scalar_keys:
    value = config.get(key)
    if isinstance(value, (str, int, float, bool)) and not isinstance(value, str) or isinstance(value, str) and len(value) <= 80:
        print(f"{key}={value}")

architectures = config.get("architectures")
if isinstance(architectures, list) and all(isinstance(item, str) for item in architectures):
    print("architectures=" + ",".join(architectures[:4]))

rope = config.get("rope_scaling")
if isinstance(rope, dict):
    rope_type = rope.get("rope_type", rope.get("type"))
    factor = rope.get("factor")
    if isinstance(rope_type, str) and len(rope_type) <= 80:
        print(f"rope_scaling.type={rope_type}")
    if isinstance(factor, (int, float)) and not isinstance(factor, bool):
        print(f"rope_scaling.factor={factor}")

quantization = config.get("quantization_config")
if isinstance(quantization, dict):
    method = quantization.get("quant_method")
    bits = quantization.get("bits")
    if isinstance(method, str) and len(method) <= 80:
        print(f"quantization.method={method}")
    if isinstance(bits, int) and not isinstance(bits, bool):
        print(f"quantization.bits={bits}")

model_type = config.get("model_type")
if model_type != "qwen3_moe":
    print("qwen3_model_type_check=unexpected")
    raise SystemExit(3)
print("qwen3_model_type_check=pass")

if not isinstance(architectures, list) or "Qwen3MoeForCausalLM" not in architectures:
    print("qwen3_architecture_check=unexpected")
    raise SystemExit(3)
print("qwen3_architecture_check=pass")
PY
  then
    fail "config.json is invalid or does not declare a Qwen3 model type"
  else
    pass "config.json is valid Qwen3 metadata; no weights or business data were read"
  fi
fi

if [[ -r $QWEN_MODEL_PATH/tokenizer_config.json && ${#json_python[@]} -gt 0 ]]; then
  if ! "${json_python[@]}" - "$QWEN_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
try:
    with (model_dir / "tokenizer_config.json").open(encoding="utf-8") as handle:
        tokenizer = json.load(handle)
except (OSError, UnicodeError, json.JSONDecodeError):
    print("tokenizer_config_status=invalid")
    raise SystemExit(2)

template = tokenizer.get("chat_template") if isinstance(tokenizer, dict) else None
embedded = ""
if isinstance(template, str):
    embedded = template
elif isinstance(template, (dict, list)):
    embedded = json.dumps(template, ensure_ascii=False)
independent_paths = sorted(
    {
        path
        for pattern in (
            "chat_template*.jinja",
            "chat_template*.json",
            "chat_templates/*.jinja",
            "chat_templates/*.json",
        )
        for path in model_dir.glob(pattern)
        if path.is_file()
    }
)
try:
    independent = "\n".join(path.read_text(encoding="utf-8") for path in independent_paths)
except (OSError, UnicodeError):
    print("independent_chat_template_status=unreadable")
    raise SystemExit(2)
supports_switch = "enable_thinking" in embedded + independent
print("tokenizer_config_status=valid")
print(f"chat_template_present={isinstance(template, str) and bool(template)}")
print(f"independent_chat_template_count={len(independent_paths)}")
print(f"chat_template_enable_thinking_switch={supports_switch}")
raise SystemExit(0 if supports_switch else 3)
PY
  then
    fail "tokenizer_config.json does not provide an enable_thinking chat-template switch"
  else
    pass "tokenizer chat template supports enable_thinking=false"
  fi
fi

if [[ -r $QWEN_MODEL_PATH/model.safetensors.index.json && ${#json_python[@]} -gt 0 ]]; then
  if ! "${json_python[@]}" - "$QWEN_MODEL_PATH" <<'PY'
import json
import os
import sys

model_dir = sys.argv[1]
index_path = os.path.join(model_dir, "model.safetensors.index.json")
try:
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
except (OSError, UnicodeError, json.JSONDecodeError):
    print("weight_index_status=invalid")
    raise SystemExit(2)

weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not all(isinstance(value, str) for value in weight_map.values()):
    print("weight_index_status=invalid-weight-map")
    raise SystemExit(2)

shards = set(weight_map.values())
missing = sum(not os.path.isfile(os.path.join(model_dir, shard)) for shard in shards)
print("weight_index_status=valid")
print(f"weight_tensor_count={len(weight_map)}")
print(f"weight_index_shard_count={len(shards)}")
print(f"weight_index_missing_shards={missing}")
raise SystemExit(0 if missing == 0 else 3)
PY
  then
    fail "the safetensors index is invalid or references missing shards"
  else
    pass "the safetensors index references existing shard files"
  fi
fi

section "Qwen3 content fingerprint"
fingerprint_helper=$PROJECT_ROOT/deploy/model-fingerprint.py
declared_fingerprint=${QWEN_MODEL_FINGERPRINT_SHA256:-}
if [[ ! -r $fingerprint_helper ]]; then
  fail "model fingerprint helper is missing: $fingerprint_helper"
elif ((${#json_python[@]} == 0)); then
  fail "model fingerprint cannot be calculated without Python"
elif ! fingerprint_report=$("${json_python[@]}" "$fingerprint_helper" \
  --model-dir "$QWEN_MODEL_PATH"); then
  [[ -n ${fingerprint_report:-} ]] && printf '%s\n' "$fingerprint_report"
  fail "Qwen3 snapshot validation or content hashing failed"
else
  printf '%s\n' "$fingerprint_report"
  actual_fingerprint=$(awk -F= '$1 == "model_fingerprint_sha256" {print $2}' <<< "$fingerprint_report")
  if [[ ! $declared_fingerprint =~ ^sha256:[0-9a-fA-F]{64}$ ]]; then
    fail "set QWEN_MODEL_FINGERPRINT_SHA256 to the computed sha256:<64hex> value"
  elif [[ ${declared_fingerprint,,} != "${actual_fingerprint,,}" ]]; then
    fail "QWEN_MODEL_FINGERPRINT_SHA256 does not match the local model snapshot"
  else
    pass "declared Qwen3 content fingerprint matches the local snapshot"
  fi
fi

section "Launcher contract"
if [[ -r $PROJECT_ROOT/deploy/run-vllm.sh ]]; then
  if grep -q -- '--disable-log-requests' "$PROJECT_ROOT/deploy/run-vllm.sh"; then
    pass "vLLM request logging is disabled by the launcher"
  else
    fail "run-vllm.sh does not disable request logging"
  fi
else
  fail "deploy/run-vllm.sh is missing"
fi
if [[ -r $PROJECT_ROOT/deploy/run-extraction.sh ]]; then
  pass "extraction launcher exists"
else
  fail "deploy/run-extraction.sh is missing"
fi
for required_command in curl setsid flock realpath sha256sum; do
  if command -v "$required_command" >/dev/null 2>&1; then
    pass "$required_command is available"
  else
    fail "$required_command is required by the Stage A workflow"
  fi
done

section "Summary"
printf 'failures=%d\n' "$failures"
printf 'warnings=%d\n' "$warnings"
printf '%s\n' 'No Hugging Face request, model inference, record decoding, or content printing was performed.'

if ((failures > 0)); then
  exit 1
fi
exit 0
