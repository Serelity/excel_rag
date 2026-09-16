#!/usr/bin/env bash
set -euo pipefail

# Keep secrets from deploy/.env out of traces even if invoked with bash -x.
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

EXTRACT_ENV_NAME=${CONDA_EXTRACT_ENV:-civic-rag-extract}
PYTORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}
RAG_RUN_RECORDS_PATH=${RAG_RUN_RECORDS_PATH:-$PROJECT_ROOT/run-records}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if conda run -n "$EXTRACT_ENV_NAME" python --version >/dev/null 2>&1; then
  echo "refusing to reuse existing environment: $EXTRACT_ENV_NAME" >&2
  echo "choose a new CONDA_EXTRACT_ENV name or explicitly remove the old environment" >&2
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
echo "Creating Stage A extraction environment: $EXTRACT_ENV_NAME"
conda env create --name "$EXTRACT_ENV_NAME" --file deploy/environment-extract.yml

if [[ -z ${RAG_WHEELHOUSE:-} ]]; then
  # Install the official CUDA 12.4 build before vLLM so a CPU or other CUDA
  # variant cannot satisfy vLLM's torch dependency by accident.
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" \
    --only-binary torch "torch==2.6.0+cu124"
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" \
    --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[serve]'
else
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" "torch==2.6.0+cu124"
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" -e '.[serve]'
fi

conda run --no-capture-output -n "$EXTRACT_ENV_NAME" python -m pip check
conda run --no-capture-output -n "$EXTRACT_ENV_NAME" python -c '
import importlib.metadata as metadata
import torch

expected = {"torch": "2.6.0", "transformers": "4.51.3", "vllm": "0.8.5"}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual.split("+", 1)[0] != version:
        raise SystemExit(f"{package}={actual}; expected {version}")
cuda = torch.version.cuda or ""
if not cuda.startswith("12.4"):
    found_cuda = cuda or "none"
    raise SystemExit(f"torch CUDA runtime is {found_cuda}; expected 12.4")
print("Stage A versions:", *(f"{name}={metadata.version(name)}" for name in expected))
print(f"PyTorch CUDA runtime: {cuda}")
'

conda run -n "$EXTRACT_ENV_NAME" python -m pip freeze --all \
  > "$RAG_RUN_RECORDS_PATH/extract-pip-freeze.txt"
conda env export --name "$EXTRACT_ENV_NAME" \
  > "$RAG_RUN_RECORDS_PATH/extract-conda-environment.yml"
echo "Wrote environment records to: $RAG_RUN_RECORDS_PATH"

echo "Created Stage A environment: $EXTRACT_ENV_NAME"
echo "Activate it with: conda activate $EXTRACT_ENV_NAME"
