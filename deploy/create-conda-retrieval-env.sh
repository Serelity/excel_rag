#!/usr/bin/env bash
set -euo pipefail

# Keep deploy/.env values out of traces even when invoked with bash -x.
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}

if [[ -n ${RAG_ENV_FILE:-} && ! -r $ENV_FILE ]]; then
  echo "RAG_ENV_FILE is not readable: $ENV_FILE" >&2
  exit 1
fi
if [[ -r $ENV_FILE ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

ENV_NAME=${CONDA_RETRIEVAL_ENV:-civic-rag-retrieval}
PYTORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}
RAG_RUN_RECORDS_PATH=${RAG_RUN_RECORDS_PATH:-$PROJECT_ROOT/run-records}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  echo "refusing to reuse existing environment: $ENV_NAME" >&2
  echo "choose a new CONDA_RETRIEVAL_ENV name or explicitly remove the old environment" >&2
  exit 1
fi

pip_install_args=(--no-build-isolation --only-binary=:all:)
if [[ -n ${RAG_WHEELHOUSE:-} ]]; then
  if [[ $RAG_WHEELHOUSE != /* ]]; then
    echo "RAG_WHEELHOUSE must be an absolute path" >&2
    exit 1
  fi
  if [[ ! -d $RAG_WHEELHOUSE ]]; then
    echo "RAG_WHEELHOUSE is not a directory: $RAG_WHEELHOUSE" >&2
    exit 1
  fi
  pip_install_args+=(--no-index --find-links "$RAG_WHEELHOUSE")
fi
if [[ $RAG_RUN_RECORDS_PATH != /* ]]; then
  echo "RAG_RUN_RECORDS_PATH must be absolute" >&2
  exit 1
fi
mkdir -p "$RAG_RUN_RECORDS_PATH"

cd "$PROJECT_ROOT"
echo "Creating retrieval environment: $ENV_NAME"
conda env create --name "$ENV_NAME" --file deploy/environment-retrieval.yml
if [[ -z ${RAG_WHEELHOUSE:-} ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" \
    --only-binary torch 'torch==2.6.0+cu124'
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" \
    --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[retrieval,dev]'
else
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" 'torch==2.6.0+cu124'
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" -e '.[retrieval,dev]'
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip check
conda run --no-capture-output -n "$ENV_NAME" python -c '
import importlib.metadata as metadata
import subprocess

import torch

expected = {
    "torch": "2.6.0",
    "transformers": "4.51.3",
    "FlagEmbedding": "1.3.4",
    "pyserini": "1.2.0",
    "modelscope": "1.40.1",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual.split("+", 1)[0] != version:
        raise SystemExit(f"{package}={actual}; expected {version}")
cuda = torch.version.cuda or ""
if not cuda.startswith("12.4"):
    found_cuda = cuda or "none"
    raise SystemExit(f"torch CUDA runtime is {found_cuda}; expected 12.4")
java_result = subprocess.run(
    ["java", "-version"], capture_output=True, check=True, text=True
)
java_lines = (java_result.stderr or java_result.stdout).splitlines()
if not java_lines:
    raise SystemExit("java -version returned no version text")
java_version = java_lines[0]
if not (java_version.startswith("openjdk version \"21") or java_version.startswith("java version \"21")):
    raise SystemExit(f"unexpected Java runtime: {java_version}; expected Java 21")
print("Retrieval versions:", *(f"{name}={metadata.version(name)}" for name in expected))
print(f"PyTorch CUDA runtime: {cuda}")
print(f"Java runtime: {java_version}")
'

conda run -n "$ENV_NAME" python -m pip freeze --all \
  > "$RAG_RUN_RECORDS_PATH/retrieval-pip-freeze.txt"
conda env export --name "$ENV_NAME" \
  > "$RAG_RUN_RECORDS_PATH/retrieval-conda-environment.yml"
echo "Wrote environment records to: $RAG_RUN_RECORDS_PATH"

echo "Created: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"
