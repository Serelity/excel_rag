# Single-H100 Conda runbook

This is the primary deployment path for the current repository. The immediate
task is Stage A only: local Qwen3 semantic extraction on one full 80 GB H100.
Stage B remains documented for later and must not be installed or started for
the first pilot.

The complete design uses two separate Conda environments and one H100 in two
sequential GPU stages:

1. `civic-rag-extract` runs Qwen3-30B-A3B under vLLM.
2. vLLM is stopped, then `civic-rag-index` runs BGE-M3 on the same GPU.

Qdrant runs from its native Linux binary and is CPU-side. The supplied launch
scripts force vLLM and Qdrant to listen on `127.0.0.1`. Do not expose either
port without a separately reviewed authentication, TLS, and firewall design.

## 1. Version and validation boundary

| Component | Pinned version or contract |
| --- | --- |
| Python | 3.11 (`>=3.11,<3.13` is supported) |
| vLLM | `0.8.5` from the `serve` project extra |
| Qdrant server | Native Linux binary `1.14.1` |
| PyTorch / Transformers | `2.6.0` / `4.51.3` |
| FlagEmbedding / Qdrant client | `1.3.4` / `1.14.2` |
| Extraction model alias | `Qwen3-30B-A3B` |
| Embedding contract | BGE-M3 dense FP16, 1,024 dimensions, cosine, normalized |
| Chunk contract | `schemas/problem-chunk-v1.schema.json` |

Do not install Conda `cudatoolkit`, `pytorch`, or `pytorch-cuda` into these
environments. vLLM and PyTorch are installed as pip wheels with their matching
CUDA 12 runtime libraries. The host still needs a sufficiently new NVIDIA
driver for those wheels. Mixing a second Conda CUDA/PyTorch stack is a common
source of loader errors and is outside this tested dependency layout.

Package versions do not prove model or binary identity. Record model source
revisions, strong model content fingerprints, the Qdrant archive checksum,
`nvidia-smi`, `conda list`, `pip freeze --all`, and the final config. The
extraction manifest records the full commit declared by the launch command,
served alias, source revision, and declared `QWEN_MODEL_FINGERPRINT_SHA256`.
Resume refuses to mix records across commits or model/prompt contracts. The
launch wrappers recompute the model fingerprint before GPU startup; a revision
label alone is not an identity check.

The repository has passed local unit and static checks. Real H100 throughput,
vLLM structured-output behavior, CUDA compatibility, and full Qdrant capacity
must be gated by the pilot below.

## 2. Capacity and input boundary

Only use the sanitized source at
`data/raw/t_order_master.sanitized.v1_9.tsv`. Do not copy, inspect, log, or send
an unsanitized source to the model service.

`data/`, `models/`, and `deploy/.env` are intentionally excluded from Git, so
`git clone` and `git pull` do not deploy them. For Stage A, verify the existing
local Qwen3 snapshot and transfer only the sanitized TSV named above through
the site's controlled channel. Never stage an unsanitized dataset on this host.

Known aggregate facts for that sanitized file:

- 982,435 records, 655,769,345 bytes (about 625 MiB), and no duplicate IDs.
- Combined content/goal length is p50 70, p95 212, p99 362, p99.9 792, and
  maximum 13,068 characters.
- 3,067 rows have both content and goal empty. They are expected to enter
  quarantine, so a full extraction may correctly return exit code `1`.

`llm.max_input_chars=15000` counts the system prompt plus compact user JSON,
including JSON escaping. It is a character guard, not a tokenizer proof. Verify
representative long rows against vLLM's `--max-model-len=16384` during the
pilot.

Initial planning values, to be replaced with target-host measurements:

- Qwen3-30B-A3B has about 56.8 GiB of BF16 weights. The `A3B` active-parameter
  count does not reduce stored weight memory. Use one exclusive, non-MIG 80 GB
  H100; do not start with quantization.
- Start with `gpu-memory-utilization=0.90`, `max-model-len=16384`,
  `max-num-seqs=8`, `max-num-batched-tokens=8192`, and extraction concurrency
  `4`. These are conservative pilot values, not measured production settings.
- Raw float32 vectors for one chunk per input row are about 3.75 GiB. Reserve
  8-15 GiB for the active Qdrant collection initially, plus separate space for
  snapshots and optimizer/WAL overhead.
- Use 64 GiB host RAM as a planning floor and 128 GiB as the preferred start.
- A practical starting target is 100 GiB free disk after models and packages
  are staged; 200 GiB is safer when retaining snapshots and benchmark runs.

## 3. Prepare code on the Git-capable login node

Run every command in this section on the login/test node against the persistent
project tree that the H100 task will mount. The H100 runtime itself needs
neither a Git executable nor local `.git` metadata. The server only pulls code
from GitHub; it does not push code, logs, data, or results. For a first checkout,
replace `<BRANCH>` with the branch supplied for the run:

```bash
cd <PERSISTENT_PARENT>
git clone --branch <BRANCH> https://github.com/Serelity/excel_rag.git excel_rag
cd excel_rag
git status --porcelain=v1 --untracked-files=normal
git rev-parse HEAD
git symbolic-ref --quiet --short HEAD
```

For an existing checkout, update without creating a server-side merge commit:

```bash
cd <SERVER_REPO>
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
git status --porcelain=v1 --untracked-files=normal
git rev-parse HEAD
git symbolic-ref --quiet --short HEAD
```

The status command must print nothing. Copy the literal full commit and branch
outputs into `RAG_CODE_COMMIT` and `RAG_CODE_BRANCH` in every H100 launch command
below. Do not use command substitution in the platform command and do not put
these release-specific values in `.env`. If the login and H100 nodes do not
mount the same persistent tree, stage that exact prepared tree through the
site's approved channel.

Do not pull or edit tracked files, or change the two declarations, between the
1/20/100/full tasks. The manifest rejects resume under a different declared
commit. When Git is available in a runtime checkout, the wrapper additionally
requires the declaration to match the current branch, commit, and clean
worktree. Without Git, it records those local checks as unverified. Finish or
deliberately abandon the current output before changing code. Root-level
scheduler-generated `*.out`, `*.err`, and `script_*.sh` files are ignored
because the platform creates them before the launch command starts. Move
personal untracked launchers such as `run_extract.sh` outside the checkout so
the login-node status check remains meaningful; use the tracked wrapper command
below for the H100 task.

The default model location is `<SERVER_REPO>/models/Qwen3-30B-A3B`. Create the
private configuration in the persistent checkout:

```bash
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
# Edit the ModelScope revision, model path if needed, package mirror, and later
# the fingerprint. Repository-local data/cache/log paths need no override.
```

`QWEN_MODEL_REVISION` may be a ModelScope revision string. Use `unknown` only
when the source revision is genuinely unavailable. Set `VLLM_API_KEY` to a
32-128 character URL-safe high-entropy value when local authentication is
required (for example, `python -c 'import secrets; print(secrets.token_urlsafe(32))'`).
An empty value is acceptable only because the service is forced to loopback.

## 4. Create the persistent Conda environment in a test session

This section does not require an H100. Use the platform's separate test
environment to create the persistent Stage A environment once:

```bash
cd <SERVER_REPO>
bash deploy/create-conda-extract-env.sh
```

The script creates only `civic-rag-extract`, installs pinned PyTorch 2.6.0
CUDA 12.4, Transformers 4.51.3, and vLLM 0.8.5, then runs `pip check`. It
intentionally refuses to reuse the same environment name. Do not run
`deploy/validate-runtime.py extract` in a CPU or V100 test session; that gate
requires one complete H100 and belongs in the next section.

For offline pip package installation, first build and transfer a complete
x86-64 Linux wheelhouse containing PyTorch, vLLM, the project, and all
transitive wheels. Then set its absolute path in `.env` and rerun the creator:

```bash
# In deploy/.env, set: RAG_WHEELHOUSE=/srv/rag/wheelhouse
bash deploy/create-conda-extract-env.sh
```

The wheelhouse does not make `conda env create` offline by itself. In a fully
disconnected setup, also configure Conda to use a local channel or a populated
package cache containing the pinned Python, pip, setuptools, and wheel packages.

For online or mirrored setup, configure `PYTORCH_CUDA_INDEX_URL` plus the
normal pip/Conda mirrors. ModelScope and Hugging Face are not contacted by the
setup script; inference uses the existing local snapshot.

## 5. Enter a one-H100 task and run preflight

In the paid-task UI, select the persisted `civic-rag-extract` environment, one
full non-MIG H100 80 GB, and the persistent project directory. The scripts use
`conda run`, so no interactive `conda activate` is required. Under a scheduler,
keep its `CUDA_VISIBLE_DEVICES`; otherwise `GPU_ID=0` selects the allocated GPU.

Use a short first H100 task to compute the large local snapshot fingerprint
without loading the model or reading ticket rows:

```bash
cd <SERVER_REPO>
set -a
. deploy/.env
set +a
conda run -n "$CONDA_EXTRACT_ENV" python deploy/model-fingerprint.py \
  --model-dir "$QWEN_MODEL_PATH"
```

After that task exits, put the reported
`model_fingerprint_sha256=sha256:...` value into the persistent `.env` from the
test environment as `QWEN_MODEL_FINGERPRINT_SHA256`. It hashes the Qwen3
configuration, tokenizer, chat template, weight index, and every indexed
safetensors shard. It reads roughly 57 GiB but performs no inference and uses
no model hub.

Start a second H100 task, run the read-only diagnostic once, and retain its
output:

```bash
cd <SERVER_REPO>
RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' \
  bash deploy/server-preflight.sh
```

Preflight checks the pinned Python/CUDA/vLLM ABI, exactly one full H100, the
declared code provenance, persistent paths, sanitized TSV size/SHA256, and the
local model fingerprint. If Git and repository metadata are available, it also
checks the commit, branch, and clean worktree. Otherwise those checks are
reported as unverified warnings, not failures. It does not load Qwen3, decode a
ticket, print prompts, or contact a model hub. Resolve every `FAIL` before
starting inference.

## 6. Stage A: single-job extraction with vLLM

The paid platform needs one foreground command. The job wrapper takes an
exclusive Stage A lock, validates the explicit code declaration, verifies the
local model fingerprint, starts `run-vllm.sh` in a private process group, waits
for `/health`, verifies the served alias, runs extraction, then stops only the
processes it created. When local Git inspection is possible it also verifies
the declared commit/branch and clean worktree. Its health monitor verifies
through Linux `/proc` that the loopback listener remains attributable to that
process group and session.
It aborts after consecutive runtime health failures, traps signals, and
preserves the extraction exit code. vLLM request and access logging are
disabled; the wrapper never records its CLI arguments, API key, prompt, or
ticket body. It clears inherited HTTP proxy variables and forces loopback into
`NO_PROXY`, because this stage has no remote HTTP dependency.

Submit each command below as a separate H100 task after inspecting the previous
task's output. Every invocation starts its own vLLM, waits for readiness, runs
the bounded extraction, and stops vLLM. With no arguments it intentionally runs
a one-record, concurrency-one smoke test.

Before overwriting an existing result, or after an interrupted one-record task,
run this metadata-only check on the login node. It does not decode or print a
ticket. `compatible=true` means the same final commit has a ready v4 manifest,
exactly one successful output record, and zero quarantine records:

```bash
cd <SERVER_REPO>
set -a
. deploy/.env
set +a
RAG_CODE_COMMIT='<FULL_COMMIT>' conda run --no-capture-output \
  -n "$CONDA_EXTRACT_ENV" python - <<'PY'
import json
import os
from pathlib import Path

output = Path("data/processed/problem_chunks.jsonl")
quarantine = Path("data/processed/problem_chunks.errors.jsonl")
manifest_path = Path(f"{output}.manifest.json")
required = (output, quarantine, manifest_path)
if not all(path.is_file() for path in required):
    print("compatible=false reason=missing-result-file")
    raise SystemExit(3)

with manifest_path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)

def line_count(path):
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)

output_count = line_count(output)
quarantine_count = line_count(quarantine)
commit = manifest.get("code", {}).get("git_commit")
prompt = manifest.get("extraction", {}).get("prompt_version")
state = manifest.get("state")
compatible = (
    state == "ready"
    and commit == os.environ["RAG_CODE_COMMIT"]
    and prompt == "problem-extraction-v4"
    and output_count == 1
    and quarantine_count == 0
)
print(f"state={state}")
print(f"git_commit={commit}")
print(f"prompt_version={prompt}")
print(f"output_records={output_count}")
print(f"quarantine_records={quarantine_count}")
print(f"compatible={str(compatible).lower()}")
raise SystemExit(0 if compatible else 3)
PY
```

If this reports `compatible=true`, skip both the archive and overwrite blocks
and continue with the 19-record resume command. An e9be781 v4 result is not
compatible with the later Gitless-runtime fix commit, even though its prompt is
also v4. A result with one quarantine record intentionally reports
`compatible=false`: the one-record quality gate requires one success and zero
quarantine. Investigate its error category and logs, do not continue to 20, and
only start another overwrite run after resolving the cause.

If the standard paths contain the known 20-record result from commit `19bd9170`
and prompt v3, preserve its output, quarantine, and manifest together before the
first v4 run. This block deliberately fails rather than reuse an existing
archive directory:

```bash
cd <SERVER_REPO>
set -euo pipefail
umask 077
archive_dir=data/processed/archive/v3-20-19bd9170
archive_files=(
  data/processed/problem_chunks.jsonl
  data/processed/problem_chunks.errors.jsonl
  data/processed/problem_chunks.jsonl.manifest.json
)
for archive_file in "${archive_files[@]}"; do
  [[ -f $archive_file ]] || { printf 'missing archive input: %s\n' "$archive_file" >&2; exit 2; }
done
mkdir -p data/processed/archive
mkdir "$archive_dir"
cp -p -- "${archive_files[@]}" "$archive_dir/"
wc -l -- "$archive_dir"/*
sha256sum -- "$archive_dir"/*
chmod -R a-w -- "$archive_dir"
```

Then submit this platform launch command. `--overwrite` starts a clean v4
one-record run at the standard output paths; it does not modify the archive.
After it produces one compatible terminal result, do not overwrite that result;
continue with the 19-record resume command instead. If the task exits before
then, rerun the metadata-only check and follow the reported state plus the job
error rather than blindly repeating `--overwrite`:

```bash
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh --overwrite
```

The command prints the extraction run ID plus the private vLLM, extraction, and
status-log paths after early launch validation passes. They are created under
`RAG_JOB_LOG_DIR`. Before GPU startup, the wrapper resolves
`data.input` from the selected config, requires it to equal `RAG_INPUT_PATH`,
and checks `RAG_INPUT_SIZE_BYTES` plus `RAG_INPUT_SHA256`. The status report
records that result, the declared commit/branch, Git availability and
verification state, model fingerprint, run ID, final output/quarantine counts,
and exit code. On a Gitless H100, `git_worktree_changes=unknown` is expected and
must not be read as a clean-worktree verification. Follow progress from another
shell only when needed:

```bash
tail -f /absolute/persistent/log/path/extraction-JOB_ID.log
```

Inspect the one-record output contract before continuing. Then add new records
in bounded resumable gates; do not skip directly to the full file:

```bash
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh \
  --resume --limit 19 --concurrency 2
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh \
  --resume --limit 80 --concurrency 4
```

`--limit` applies to newly submitted rows. Starting from the one-row smoke,
the two resume commands bring the cumulative attempted totals to 20 and 100.
A non-full task rejects `--limit` values above 100; the unbounded path requires
the explicit `--full --resume` pair.

Exit codes are `0` for no new failures, `1` when submitted rows were
quarantined, `2` for setup/runtime errors, and `129`/`130`/`143` for
HUP/INT/TERM. A `1` does not discard successful records. Existing outputs are
never silently replaced; use `--resume` when the manifest and inputs match. Use
`--overwrite` only for an intentional fresh run.

Only after the 1/20/100 attempted-record results and logs are correct, run the
full resumable pass explicitly:

```bash
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh \
  --full --resume --concurrency 4
```

If vLLM fails during startup with a GPU-memory error, first verify that the
H100 is exclusive. Then retry with `VLLM_MAX_NUM_SEQS=4`. If a long prefill
still OOMs, reduce `VLLM_MAX_NUM_BATCHED_TOKENS` from `8192` to `4096`; if CUDA
graph capture remains the blocker, set `VLLM_ENFORCE_EAGER=1`. Do not add a
reasoning parser or enable reasoning output. Increase GPU memory utilization to
at most `0.92` only after confirming no other process uses the full H100.

The extraction code provides deterministic backstops for common email, mobile,
landline, 15/18-digit ID-card, and redaction-placeholder formats. It is not a
general PII detector and does not reliably identify names or free-form street
addresses. Upstream sanitization, local inference, access control, and
candidate-output review remain mandatory.

Review quarantine categories without printing source text:

```bash
cd <SERVER_REPO>
set -a
. deploy/.env
set +a
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" python - <<'PY'
import collections
import json
from pathlib import Path

counts = collections.Counter()
path = Path("data/processed/problem_chunks.errors.jsonl")
if path.exists():
    with path.open(encoding="utf-8") as records:
        for line in records:
            row = json.loads(line)
            counts[(row.get("error_code"), row.get("status_code"))] += 1
print(dict(counts))
PY
```

Treat `UPSTREAM_RATE_LIMITED`, `UPSTREAM_TIMEOUT`, and
`UPSTREAM_CONNECTION_ERROR` as retry candidates. Inspect the HTTP status for
`UPSTREAM_HTTP_ERROR`. Retry only when the input, model, prompt, and decoding
contract have not changed:

```bash
cd <SERVER_REPO> && RAG_CODE_COMMIT='<FULL_COMMIT>' RAG_CODE_BRANCH='<BRANCH>' exec bash deploy/run-extraction-job.sh \
  --full --resume --retry-failures --concurrency 4
```

This retries every quarantined ID, including permanent empty-input failures,
and appends new audit rows. It cannot select failure classes.

### Tune extraction concurrency

The checked-in extraction concurrency `4`, vLLM `max-num-seqs=8`, and
`max-num-batched-tokens=8192` are conservative pilot defaults, not measured
production recommendations. On a fixed representative sanitized subset, test
application concurrency 4, 6, then 8. Change one parameter at a time. Record
output validity, success/failure counts, wall time, records/second, GPU
memory/utilization/power, CPU RSS, disk throughput, and vLLM
preemption/OOM/service errors. Include short and long inputs and separate
cold-start from steady-state timing. Do not log prompts or ticket bodies.

Useful observers are `/usr/bin/time -v` and
`nvidia-smi dmon -s pucm -d 1`. Keep `4` until a faster stable value passes the
same semantic and long-input checks.

## 7. GPU handoff and Stage B indexing

Do not perform this section during the current Stage A server setup. When Stage
B is approved, obtain the official native Qdrant `v1.14.1` x86-64 archive on a
connected compatible machine, verify its published checksum, and transfer it
through the controlled channel before following the indexing steps below.

Stop the foreground vLLM process with Ctrl-C or the supervisor's normal TERM
operation. Confirm that no vLLM worker remains before loading BGE-M3:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Do not use broad `pkill` commands on a shared server. Start native Qdrant in a
dedicated terminal or supervised process:

```bash
bash deploy/run-qdrant.sh
```

In the indexing terminal:

```bash
set -a
. deploy/.env
set +a
qdrant_headers=()
if [[ -n ${QDRANT_API_KEY:-} ]]; then
  qdrant_headers=(-H "api-key: $QDRANT_API_KEY")
fi
curl -fsS "${qdrant_headers[@]}" \
  "http://127.0.0.1:${QDRANT_HTTP_PORT}/healthz"
```

Create a small JSONL pilot without modifying the complete output:

```bash
head -n 1000 data/processed/problem_chunks.jsonl \
  > data/processed/problem_chunks.index-pilot.jsonl
```

Run a disposable pilot collection:

```bash
bash deploy/run-index.sh \
  --input data/processed/problem_chunks.index-pilot.jsonl \
  --embedding-batch-size 16 \
  --collection problem_chunks_bge_m3_pilot_v1
```

The indexer validates the entire JSONL before database mutation. It rejects a
collection with different vector settings or embedding provenance. Verify the
exact point count:

```bash
EXPECTED_POINTS=$(wc -l < data/processed/problem_chunks.index-pilot.jsonl)
curl -fsS "${qdrant_headers[@]}" -X POST \
  "http://127.0.0.1:${QDRANT_HTTP_PORT}/collections/problem_chunks_bge_m3_pilot_v1/points/count" \
  -H 'Content-Type: application/json' -d '{"exact":true}'
printf 'expected=%s\n' "$EXPECTED_POINTS"
```

Test embedding batches 16, 32, and 64 on the same representative input, with a
different disposable collection for each run. Keep the fastest repeatable
end-to-end setting that preserves output/count checks and GPU memory headroom.
An OOM or a fast short-only pilot is not evidence for the full length mix.

Run the complete idempotent upsert with the selected batch size:

```bash
EMBED_BATCH_SIZE=16
bash deploy/run-index.sh --embedding-batch-size "$EMBED_BATCH_SIZE"
```

After interruption, rerun the same command without `--recreate`. Stable UUIDs
make upserts idempotent, although embeddings are recomputed from the beginning.
Keep `BGE_M3_MODEL_PATH` at the same normalized absolute path for every rerun.
The path is part of the embedding provenance fingerprint, so mounting identical
weights at another path correctly causes a collection mismatch.
`--recreate` deletes the named collection and can leave it empty or partial if
a later step fails. For production rebuilds, use a new versioned collection,
verify count and retrieval quality, then switch the consuming application.

After verification, create a Qdrant snapshot and copy it to a different failure
domain. Snapshot and restore automation are outside this repository. Stop the
foreground Qdrant process with Ctrl-C after all writes are complete.

## 8. Release checks

Run code-quality checks in a developer environment before pushing. The
server's Stage A environment intentionally installs only `.[serve]`; its
on-host gate is `pip check` plus `deploy/validate-runtime.py extract`.

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
set -a
. deploy/.env
set +a
conda run -n "$CONDA_EXTRACT_ENV" python -m pip check
git diff --check
```

Candidate chunks are machine-extracted, not human-verified evidence. Retrieval,
reranking, answer generation, answer-level evaluation, authorization, restore
testing, remote TLS, and online monitoring are outside the current codebase.
Keep the sanitized source, outputs, quarantine, models, `.env`, and Qdrant
storage access-controlled.
