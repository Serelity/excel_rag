#!/usr/bin/env bash
set -euo pipefail

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

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  echo "refusing to reuse existing environment: $ENV_NAME" >&2
  echo "choose a new CONDA_RETRIEVAL_ENV name or explicitly remove the old environment" >&2
  exit 1
fi

pip_install_args=(--no-build-isolation)
if [[ -n ${RAG_WHEELHOUSE:-} ]]; then
  if [[ $RAG_WHEELHOUSE != /* || ! -d $RAG_WHEELHOUSE ]]; then
    echo "RAG_WHEELHOUSE must be an existing absolute directory" >&2
    exit 1
  fi
  pip_install_args+=(--no-index --find-links "$RAG_WHEELHOUSE")
fi

cd "$PROJECT_ROOT"
conda env create --name "$ENV_NAME" --file deploy/environment-retrieval.yml
if [[ -z ${RAG_WHEELHOUSE:-} ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" 'torch==2.6.0'
else
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" 'torch==2.6.0'
fi
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install "${pip_install_args[@]}" -e '.[retrieval,dev]'
conda run -n "$ENV_NAME" python -m pip check
conda run -n "$ENV_NAME" java -version

if [[ -n ${RAG_RUN_RECORDS_PATH:-} ]]; then
  if [[ $RAG_RUN_RECORDS_PATH != /* ]]; then
    echo "RAG_RUN_RECORDS_PATH must be absolute" >&2
    exit 1
  fi
  mkdir -p "$RAG_RUN_RECORDS_PATH"
  conda run -n "$ENV_NAME" python -m pip freeze --all \
    > "$RAG_RUN_RECORDS_PATH/retrieval-pip-freeze.txt"
fi

echo "Created: $ENV_NAME"
