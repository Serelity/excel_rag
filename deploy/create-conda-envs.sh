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

EXTRACT_ENV_NAME=${CONDA_EXTRACT_ENV:-civic-rag-extract}
INDEX_ENV_NAME=${CONDA_INDEX_ENV:-civic-rag-index}
PYTORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if [[ $EXTRACT_ENV_NAME == "$INDEX_ENV_NAME" ]]; then
  echo "CONDA_EXTRACT_ENV and CONDA_INDEX_ENV must be different" >&2
  exit 1
fi

pip_install_args=(--no-build-isolation)
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

for env_name in "$EXTRACT_ENV_NAME" "$INDEX_ENV_NAME"; do
  if conda run -n "$env_name" python --version >/dev/null 2>&1; then
    echo "refusing to reuse existing environment: $env_name" >&2
    echo "choose a new CONDA_*_ENV name, or explicitly remove the old environment first" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
conda env create --name "$EXTRACT_ENV_NAME" --file deploy/environment-extract.yml
conda env create --name "$INDEX_ENV_NAME" --file deploy/environment-index.yml

if [[ -z ${RAG_WHEELHOUSE:-} ]]; then
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" \
    --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[serve,dev]'
else
  conda run --no-capture-output -n "$EXTRACT_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" -e '.[serve,dev]'
fi
if [[ -z ${RAG_WHEELHOUSE:-} ]]; then
  conda run --no-capture-output -n "$INDEX_ENV_NAME" \
    python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" 'torch==2.6.0'
else
  conda run --no-capture-output -n "$INDEX_ENV_NAME" \
    python -m pip install "${pip_install_args[@]}" 'torch==2.6.0'
fi
conda run --no-capture-output -n "$INDEX_ENV_NAME" \
  python -m pip install "${pip_install_args[@]}" -e '.[index]'

conda run -n "$EXTRACT_ENV_NAME" python -m pip check
conda run -n "$INDEX_ENV_NAME" python -m pip check

if [[ -n ${RAG_RUN_RECORDS_PATH:-} ]]; then
  if [[ $RAG_RUN_RECORDS_PATH != /* ]]; then
    echo "RAG_RUN_RECORDS_PATH must be absolute" >&2
    exit 1
  fi
  mkdir -p "$RAG_RUN_RECORDS_PATH"
  conda run -n "$EXTRACT_ENV_NAME" python -m pip freeze --all \
    > "$RAG_RUN_RECORDS_PATH/extract-pip-freeze.txt"
  conda run -n "$INDEX_ENV_NAME" python -m pip freeze --all \
    > "$RAG_RUN_RECORDS_PATH/index-pip-freeze.txt"
  echo "Wrote pip freeze records to: $RAG_RUN_RECORDS_PATH"
fi

echo "Created: $EXTRACT_ENV_NAME and $INDEX_ENV_NAME"
echo "Record exact packages with:"
echo "  conda run -n $EXTRACT_ENV_NAME python -m pip freeze --all"
echo "  conda run -n $INDEX_ENV_NAME python -m pip freeze --all"
