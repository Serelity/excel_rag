#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
CONDA_BASELINE_ENV=${CONDA_BASELINE_ENV:-civic-rag-extract}
BASELINE_INPUT=${BASELINE_INPUT:-$PROJECT_ROOT/data/raw/t_order_master.sanitized.v1_9.tsv}
BASELINE_OUTPUT=${BASELINE_OUTPUT:-$PROJECT_ROOT/data/retrieval-baseline-v1}

command -v conda >/dev/null || { printf 'ERROR: conda is not on PATH\n' >&2; exit 2; }
[[ -f $BASELINE_INPUT ]] || { printf 'ERROR: missing input: %s\n' "$BASELINE_INPUT" >&2; exit 2; }
[[ ! -e $BASELINE_OUTPUT ]] || {
  printf 'ERROR: output exists; use a new BASELINE_OUTPUT path: %s\n' "$BASELINE_OUTPUT" >&2
  exit 2
}
cd "$PROJECT_ROOT"

# No GPU, model server, network, Git, or deployment .env file is needed at runtime.
conda run --no-capture-output -n "$CONDA_BASELINE_ENV" python -c \
  'import sqlite3, sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"; sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE probe USING fts5(text)"); print("Python and SQLite FTS5: OK")'

conda run --no-capture-output -n "$CONDA_BASELINE_ENV" python -m retrieval_baseline build \
  --input "$BASELINE_INPUT" --output "$BASELINE_OUTPUT/dataset" \
  --dev-start 2026-01-01 --test-start 2026-02-01 --queries-per-split 200 --seed 42

conda run --no-capture-output -n "$CONDA_BASELINE_ENV" python -m retrieval_baseline index \
  --dataset "$BASELINE_OUTPUT/dataset" --output "$BASELINE_OUTPUT/index"

conda run --no-capture-output -n "$CONDA_BASELINE_ENV" python -m retrieval_baseline evaluate \
  --dataset "$BASELINE_OUTPUT/dataset" --index "$BASELINE_OUTPUT/index" \
  --output "$BASELINE_OUTPUT/dev-bm25" --split dev --case-k 50 --max-terms 32

printf 'Completed. Development report: %s/dev-bm25/report.json\n' "$BASELINE_OUTPUT"
