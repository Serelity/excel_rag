#!/usr/bin/env bash
set -euo pipefail
{ set +x; } 2>/dev/null

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env.semantic}
if [[ -r $ENV_FILE ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${CONDA_EXTRACT_ENV:=civic-rag-extract}"
: "${PYTORCH_CUDA_INDEX_URL:=https://download.pytorch.org/whl/cu124}"

command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
cd "$PROJECT_ROOT"

if conda run -n "$CONDA_EXTRACT_ENV" python --version >/dev/null 2>&1; then
  printf 'Updating existing Conda environment: %s\n' "$CONDA_EXTRACT_ENV"
  conda env update --name "$CONDA_EXTRACT_ENV" \
    --file deploy/environment-semantic-extraction.yml
else
  printf 'Creating Conda environment: %s\n' "$CONDA_EXTRACT_ENV"
  conda env create --name "$CONDA_EXTRACT_ENV" \
    --file deploy/environment-semantic-extraction.yml
fi

conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" \
  --only-binary torch "torch==2.6.0+cu124"
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m pip install --extra-index-url "$PYTORCH_CUDA_INDEX_URL" -e '.[serve]'
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" python -m pip check
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" python - <<'PY'
import importlib.metadata as metadata
import torch

expected = {"openai": "1.75.0", "pydantic": "2.11.4", "torch": "2.6.0", "vllm": "0.8.5"}
for package, wanted in expected.items():
    actual = metadata.version(package).split("+", 1)[0]
    if actual != wanted:
        raise SystemExit(f"{package}={actual}; expected {wanted}")
if not (torch.version.cuda or "").startswith("12.4"):
    raise SystemExit(f"expected PyTorch CUDA 12.4, got {torch.version.cuda}")
print("environment_check=passed")
PY
