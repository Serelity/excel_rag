#!/usr/bin/env bash
# Shell regression with fake Conda/Python: no packages, GPU, model, or network calls.
set -euo pipefail
PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$PROJECT_ROOT"
mkdir -p .cache
fixture=$(mktemp -d "$PROJECT_ROOT/.cache/launch-interpreter-XXXXXXXX")
export LAUNCH_TEST_BASE=$fixture/base
export LAUNCH_TEST_TARGET="$fixture/selected env"
export LAUNCH_TEST_TRACE=$fixture/python.trace
mkdir -p "$LAUNCH_TEST_BASE/bin" "$LAUNCH_TEST_TARGET/bin" "$fixture/commands"

cat > "$LAUNCH_TEST_BASE/bin/conda" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ $1 == info && $2 == --base ]]; then
  printf '%s\n' "$LAUNCH_TEST_BASE"
  exit 0
fi
[[ $1 == run && $2 == --no-capture-output && $3 == -p ]] || exit 90
[[ $4 == "$LAUNCH_TEST_TARGET" ]] || exit 91
shift 4
# Leave PATH polluted, as can happen when Conda replaces a later active-env entry.
exec "$@"
SH
cat > "$LAUNCH_TEST_BASE/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == deploy/inspect-case-contract-env.py ]]; then
  shift
  while (($#)); do
    if [[ $1 == --output ]]; then report=$2; fi
    shift 2
  done
  printf '{"client":{"prefix":"%s"},"server":{"prefix":"%s"}}\n' \
    "$LAUNCH_TEST_TARGET" "$LAUNCH_TEST_TARGET" > "$report"
elif [[ ${1:-} == -c ]]; then
  printf '%s\n' "$LAUNCH_TEST_TARGET"
else
  printf 'Python 3.12.13 is unsupported; expected 3.11.x\n' >&2
  exit 93
fi
SH
cat > "$LAUNCH_TEST_TARGET/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == -m ]] || exit 94
printf '%s\t%s\t%s\n' "${BASH_SOURCE[0]}" "$2" "${3:-}" >> "$LAUNCH_TEST_TRACE"
# The outer wrapper must not alter PATH between its inspection and client calls.
[[ $PATH == "$LAUNCH_TEST_PATH" ]] || exit 95
case "$2" in
  semantic_extraction.case_contract.runner)
    [[ $3 == prepare ]] || exit 96 ;;
  semantic_extraction.validate_runtime|semantic_extraction.validate_model|vllm.entrypoints.openai.api_server)
    : ;;
  *) exit 97 ;;
esac
SH
# Stop the outer wrapper at its port lock, before any service/network operation.
printf '#!/usr/bin/env bash\nexit 1\n' > "$fixture/commands/flock"
for name in curl setsid; do
  printf '#!/usr/bin/env bash\nexit 98\n' > "$fixture/commands/$name"
done
chmod +x "$LAUNCH_TEST_BASE/bin/conda" "$LAUNCH_TEST_BASE/bin/python" \
  "$LAUNCH_TEST_TARGET/bin/python" "$fixture/commands/"*
export CONDA_EXE=$LAUNCH_TEST_BASE/bin/conda
export CUDA_VISIBLE_DEVICES=0
export PATH="$LAUNCH_TEST_BASE/bin:$LAUNCH_TEST_TARGET/bin:$fixture/commands:$PATH"
export LAUNCH_TEST_PATH=$PATH
printf -v fingerprint 'sha256:%064d' 0
{
  printf 'CONDA_EXTRACT_PREFIX=%q\n' "$LAUNCH_TEST_TARGET"
  printf 'QWEN_MODEL_PATH=%q\n' "$fixture/model"
  printf 'QWEN_MODEL_FINGERPRINT_SHA256=%q\n' "$fingerprint"
  printf 'VLLM_CACHE_PATH=%q\n' "$fixture/vllm-cache"
  printf 'VLLM_PORT=62017\n'
} > "$fixture/private.env"
export RAG_ENV_FILE=$fixture/private.env

# Negative control: a bare python really does select the wrong interpreter here.
status=0
"$CONDA_EXE" run --no-capture-output -p "$LAUNCH_TEST_TARGET" \
  python -m semantic_extraction.validate_runtime > "$fixture/control.log" 2>&1 || status=$?
[[ $status == 93 ]] || { printf 'FAIL: negative control did not detect wrong Python\n'; exit 1; }

# Runtime check, model validation, and service must all use the selected Python.
bash deploy/run-qwen3-vllm.sh > "$fixture/launcher.log" 2>&1
mapfile -t trace < "$LAUNCH_TEST_TRACE"
[[ ${#trace[@]} == 3 ]] || exit 1
[[ ${trace[0]} == "$LAUNCH_TEST_TARGET/bin/python"$'\tsemantic_extraction.validate_runtime\t' ]] || exit 1
[[ ${trace[1]} == "$LAUNCH_TEST_TARGET/bin/python"$'\tsemantic_extraction.validate_model\t--model-dir' ]] || exit 1
[[ ${trace[2]} == "$LAUNCH_TEST_TARGET/bin/python"$'\tvllm.entrypoints.openai.api_server\t--host' ]] || exit 1

# Exercise the actual outer wrapper through preparation and configuration handoff.
status=0
bash deploy/run-case-contract.sh smoke --env-file "$fixture/private.env" \
  --input-dir "$fixture/input" --output "$fixture/output" \
  > "$fixture/wrapper.log" 2>&1 || status=$?
[[ $status == 2 ]] || { cat "$fixture/wrapper.log"; exit 1; }
mapfile -t trace < "$LAUNCH_TEST_TRACE"
[[ ${#trace[@]} == 4 ]] || { cat "$fixture/wrapper.log"; exit 1; }
[[ ${trace[3]} == "$LAUNCH_TEST_TARGET/bin/python"$'\tsemantic_extraction.case_contract.runner\tprepare' ]] || exit 1
(
  unset CONDA_EXTRACT_PREFIX CONDA_EXE
  source "$fixture/output/runtime.env"
  [[ $CONDA_EXTRACT_PREFIX == "$LAUNCH_TEST_TARGET" ]]
  [[ $CONDA_EXE == "$LAUNCH_TEST_BASE/bin/conda" ]]
)
printf 'bare_python_negative_control=passed\n'
printf 'launcher_absolute_python=passed\n'
printf 'wrapper_absolute_python_and_handoff=passed\n'
printf 'regression_records=%s\n' "$fixture"
