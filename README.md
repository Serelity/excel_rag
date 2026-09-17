# Civic RAG pipeline

This repository currently provides two recoverable offline stages:

1. Extract candidate problem chunks from a sanitized ticket TSV through a
   local OpenAI-compatible vLLM endpoint.
2. Generate dense BGE-M3 embeddings and upsert them into Qdrant.

Query retrieval, reranking, grounded answer generation, and evaluation are not
yet implemented.

## Stage A: Qwen3 extraction on one H100

The current server task is only semantic extraction with the local
`Qwen3-30B-A3B` ModelScope snapshot. Do not install or start BGE-M3 or Qdrant
yet. In a persistent test session, use the extraction-only Conda setup:

```bash
cd /path/to/this/repository
cp deploy/.env.example deploy/.env
# Edit deploy/.env for the persistent paths and package mirror.
bash deploy/create-conda-extract-env.sh
```

Run Git operations on the login/test node. Record the clean checkout's literal
`git rev-parse HEAD` and `git symbolic-ref --quiet --short HEAD` output for the
H100 launch command; do not persist these release-specific values in `.env`.

Do not run the H100 runtime gate in a CPU or V100 setup session. Use a short
first H100 task to compute the model fingerprint with
`deploy/model-fingerprint.py`. Put it into the persistent `.env` from the test
environment, then use a second H100 task for the read-only
`bash deploy/server-preflight.sh` before inference.

The model is served only from its local absolute path; Hugging Face access is
disabled. Record `QWEN_MODELSCOPE_REPO_ID` and the ModelScope source revision,
then bind execution to the local content with
`QWEN_MODEL_FINGERPRINT_SHA256=sha256:<64hex>`.
`deploy/model-fingerprint.py` validates the Qwen3-30B-A3B structure, BF16
non-quantized contract, tokenizer thinking switch, weight index, shards, and
broken links before hashing the sorted model files.

Run the read-only server diagnostic after configuring `.env`:

```bash
RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' \
  bash deploy/server-preflight.sh
```

It reports platform, declared code provenance, optional local Git verification,
Conda, GPU, memory, disk/mounts, persistent-path candidates, sanitized-input
size/SHA256, and safe model metadata. It does not decode ticket records, print
prompts, run inference, or access a model hub. Git is not required in the H100
runtime.

## H100 deployment

The primary deployment is Conda-native. For Stage A, one supervised job starts
vLLM on loopback, waits for health and the `Qwen3-30B-A3B` alias, runs
extraction, records private persistent logs, and stops only the process group it
created. Optional vLLM API authentication is supported.

Follow [`deploy/H100_RUNBOOK.md`](deploy/H100_RUNBOOK.md) for Stage A model
identity, offline installation, preflight checks, pilot/full/resume commands,
and H100 tuning. The current deployment path is Conda-only.

`data/`, `models/`, and `deploy/.env` are intentionally Git-ignored. A clone or
pull will not transfer them. Keep and verify the existing local Qwen3 snapshot;
do not prepare BGE-M3 yet. Keep secrets separate, and transfer only
`data/raw/t_order_master.sanitized.v1_9.tsv` through the site's controlled data
channel.

## Extraction job

With no arguments, the job deliberately processes only one record at
concurrency one. Submit each command below as a separate H100 platform task,
inspect its result, and only then continue to the next gate:

If the standard output paths already contain results from an older code or
prompt contract, first archive `problem_chunks.jsonl`, its `.errors.jsonl`, and
its `.manifest.json` together in a new, non-reused directory. The first command
below intentionally starts a fresh one-record run. After it has produced one
compatible terminal result, do not run it again; continue with the 19-record
resume command. If the task fails before that point, inspect the manifest and
counts using the runbook before choosing overwrite or resume. Never resume
across a changed commit or prompt version.

```bash
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh --overwrite
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh --resume --limit 19 --concurrency 2
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh --resume --limit 80 --concurrency 4
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh --full --resume --concurrency 4
```

`--limit` counts newly submitted rows, so the first three commands produce
1, then 20, then 100 attempted records in total. Any invocation without
`--limit` remains a one-row smoke unless `--full --resume` is explicit. A
non-full invocation rejects a limit above 100 to prevent an accidental full
run disguised as a pilot.

Each invocation takes the shared Stage A lock and verifies the declared local
model fingerprint before loading the GPU. After early launch validation passes,
logs and a status report containing the declared commit/branch, local Git
verification state, extraction run ID, final output counts, and exit code are
written under `RAG_JOB_LOG_DIR`. When Git and repository metadata are available,
the wrapper also requires the declaration to match a clean checkout. Otherwise
it records the worktree as unverified. The result manifest binds resume to the
declared commit. Runtime health failures stop the extraction instead of
quarantining the remaining file. vLLM request and access logging are disabled,
and the wrapper does not record CLI arguments, prompts, or ticket content. The
health monitor also verifies that the loopback listener remains attributable to
the vLLM process group and session created by that job. Stage A clears inherited
proxy variables so local model requests cannot be routed through a platform
proxy.

Successful chunks go to `data/processed/problem_chunks.jsonl`. Failed records
go to `data/processed/problem_chunks.errors.jsonl` without source text. Exit
code `1` means one or more records were quarantined; successful rows remain
durable and resumable. `--resume --retry-failures` retries quarantined IDs.

## Deferred indexing

Do not run this stage during the current Qwen3 pilot. When Stage B is approved,
stop vLLM before embedding on a single H100, start Qdrant in another terminal
with `bash deploy/run-qdrant.sh`, then use the guarded indexing wrapper:

```bash
bash deploy/run-index.sh --embedding-batch-size 16
```

Upserts use stable point IDs, so rerunning without `--recreate` is idempotent.
`--recreate` deletes an existing collection and is intended only for disposable
pilots or an explicitly planned rebuild. Batch size `16` and extraction
concurrency `4` are conservative pilot values, not measured recommendations.

## Verification

Run tests and Ruff in a developer environment with `.[dev]` installed. The
server's intentionally smaller Stage A environment uses these runtime gates:

```bash
conda run -n civic-rag-extract python -m pip check
GPU_VISIBILITY=${CUDA_VISIBLE_DEVICES:-0}
CUDA_VISIBLE_DEVICES="$GPU_VISIBILITY" conda run -n civic-rag-extract \
  python deploy/validate-runtime.py extract
```

The runtime chunk contract is `schemas/problem-chunk-v1.schema.json`. Chunks
are marked `candidate`; they must not be presented as human-verified evidence.
