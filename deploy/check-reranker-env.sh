#!/usr/bin/env bash
set -euo pipefail

# Reuse the pinned retrieval environment without upgrading its embedding packages.
CONDA_RETRIEVAL_ENV=${CONDA_RETRIEVAL_ENV:-civic-rag-retrieval}
command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
exec conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -c '
import importlib.metadata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modelscope.hub.snapshot_download import snapshot_download
expected = {"torch": "2.6.0", "transformers": "4.51.3", "sentencepiece": "0.2.0", "modelscope": "1.25.0"}
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    if actual.split("+", 1)[0] != version:
        raise SystemExit(f"ERROR: {name}={actual}; expected {version}. Check deploy/RETRIEVAL_SETUP.md.")
    print(f"{name}={actual}")
print("reranker_environment_check=passed; no packages changed and no model loaded")
'
