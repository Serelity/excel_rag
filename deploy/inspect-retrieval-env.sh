#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null
umask 077

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$PROJECT_ROOT"
if [[ -f deploy/.env.retrieval ]]; then
  # shellcheck disable=SC1091
  source deploy/.env.retrieval
fi
: "${CONDA_RETRIEVAL_ENV:=civic-rag-retrieval}"
: "${PHASE1_ROOT:=data/case-relevance-phase1-v1}"

CONDA_BIN=${CONDA_EXE:-}
if [[ ! -x "$CONDA_BIN" ]]; then
  CONDA_BIN=$(type -P conda || true)
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  printf 'ERROR: cannot find the conda executable. In the active shell run:\n' >&2
  printf '  CONDA_BASE="$(conda info --base)"; export PATH="$CONDA_BASE/bin:$PATH"\n' >&2
  exit 2
fi

output_dir=${PHASE1_ROOT%/}
mkdir -p "$output_dir"
report="$output_dir/environment-diagnostic-$(date +%Y%m%d-%H%M%S)-$$.txt"
exec > >(tee "$report") 2>&1

printf 'environment=%s\n' "$CONDA_RETRIEVAL_ENV"
printf 'diagnostic_file=%s\n' "$report"
printf 'conda_executable=%s\n' "$CONDA_BIN"
for variable in RETRIEVAL_DATASET RETRIEVAL_LEXICAL_INDEX RETRIEVAL_DENSE_INDEX \
    RETRIEVAL_ADDRESS_INDEX RETRIEVAL_MODEL_PATH RERANKER_MODEL_PATH; do
  if [[ -n ${!variable:-} ]]; then
    path=${!variable}
    if [[ -e "$path" ]]; then state=present; else state=missing; fi
    printf 'resource %-26s %s\n' "$variable" "$state"
  else
    printf 'resource %-26s unset\n' "$variable"
  fi
done

printf '\n--- GPU ---\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  printf 'nvidia-smi=unavailable\n'
fi

printf '\n--- Conda package records ---\n'
"$CONDA_BIN" list -n "$CONDA_RETRIEVAL_ENV" \
  | awk '$1 ~ /^(python|numpy|numpy-base|scipy|scikit-learn|torch|transformers|sentence-transformers|sentencepiece|faiss-cpu|filelock|modelscope|tokenizers|pydantic|openai)$/ '

printf '\n--- Python package versions and import probes ---\n'
"$CONDA_BIN" run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python - <<'PY'
import importlib.metadata as metadata
import subprocess
import sys

print(f"python_executable={sys.executable}")
print(f"python_prefix={sys.prefix}")
for package in (
    "numpy", "scipy", "scikit-learn", "torch", "transformers",
    "sentence-transformers", "sentencepiece", "faiss-cpu", "filelock",
    "modelscope", "tokenizers", "pydantic", "openai",
):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "missing"
    print(f"package {package}={version}")

probes = {
    "numpy": "import numpy",
    "scipy": "import scipy",
    "scikit-learn": "import sklearn",
    "transformers_model_classes": (
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification"
    ),
    "sentence_transformers": "from sentence_transformers import SentenceTransformer",
    "faiss": "import faiss",
    "torch_cuda": (
        "import torch; print('torch_cuda_available=' + str(torch.cuda.is_available())); "
        "print('torch_cuda_version=' + str(torch.version.cuda)); "
        "print('torch_device_count=' + str(torch.cuda.device_count()))"
    ),
    "modelscope": "from modelscope.hub.snapshot_download import snapshot_download",
}
for name, code in probes.items():
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        detail = ""
        if result.stdout.strip():
            detail = "; " + "; ".join(result.stdout.strip().splitlines()[-3:])
        print(f"import {name}=passed{detail}")
    else:
        lines = (result.stderr or result.stdout).strip().splitlines()
        detail = " | ".join(lines[-2:])[:500] if lines else "no error text"
        print(f"import {name}=failed; {detail}")
PY

printf '\nDiagnostic complete. Package state was not changed.\n'
