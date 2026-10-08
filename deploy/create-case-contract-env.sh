#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null
umask 077

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
# Independent environment requested after the existing retrieval env lacked vLLM.
# Never read .env.semantic here: its legacy name must not redirect installation.
environment=civic-rag-extract-v1
[[ $# == 0 ]] || { printf 'Usage: bash deploy/create-case-contract-env.sh\n' >&2; exit 2; }
CONDA_BIN=${CONDA_EXE:-}
[[ -x $CONDA_BIN ]] || CONDA_BIN=$(type -P conda || true)
[[ -x $CONDA_BIN ]] || { printf 'ERROR: conda executable is unavailable\n' >&2; exit 2; }
CONTRACT_CONDA_BASE=$("$CONDA_BIN" info --base)
base_python=$CONTRACT_CONDA_BASE/bin/python
[[ -x $base_python ]] || { printf 'ERROR: base Python is unavailable\n' >&2; exit 2; }
records=$PROJECT_ROOT/data/case-relevance-phase1-v1
mkdir -p -- "$records"
record=$(mktemp -d "$records/contract-install-XXXXXXXX")
printf 'installation_record=%s\n' "$record"
"$CONDA_BIN" env list --json > "$record/conda-before.json"
"$base_python" - "$record/conda-before.json" "$environment" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as handle:
    prefixes = json.load(handle)["envs"]
if any(Path(prefix).name == sys.argv[2] for prefix in prefixes):
    raise SystemExit("ERROR: civic-rag-extract-v1 already exists; no environment was modified")
PY

finish() {
  local status=$?
  trap - EXIT
  printf 'exit_code=%s\n' "$status" > "$record/install.status"
  exit "$status"
}
trap finish EXIT
"$CONDA_BIN" env create --yes --name "$environment" \
  --file deploy/environment-semantic-extraction.yml
client=("$CONDA_BIN" run --no-capture-output -n "$environment" python)
"${client[@]}" -m pip install \
  --extra-index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}" \
  --only-binary=vllm,torch,torchvision,torchaudio \
  --report "$record/pip-install.json" \
  -r deploy/requirements-case-contract.txt -e .
"${client[@]}" -m pip check
"${client[@]}" -m pip freeze > "$record/pip-freeze.txt"
"${client[@]}" -c 'import torch, vllm, transformers; assert torch.version.cuda == "12.4", torch.version.cuda; print("package_imports=passed"); print("torch_cuda_runtime=" + torch.version.cuda)'
"$base_python" deploy/inspect-case-contract-env.py \
  --conda "$CONDA_BIN" --conda-env "$environment" --output "$record/environment.json"
printf 'environment_created=%s\n' "$environment"
printf 'GPU/model loading is checked by the H100 smoke task.\n'
