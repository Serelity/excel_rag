# Qwen3 semantic extraction

This stage tests whether an evidence-grounded Qwen3 representation improves
knowledge retrieval over raw `case_content`. It is an extraction experiment,
not a replacement for the source record and not a classifier benchmark.

## Experimental boundary

The model receives exactly one source field:

```json
{"case_content":"..."}
```

It does not receive `case_goal`, existing category fields, `knowledge_quote`,
department, status, satisfaction, address columns, or any other source field.
This prevents target leakage and makes a later raw-vs-extracted retrieval
comparison interpretable.

`semantic-extraction-v3` supports zero to six events. Every trigger, actor,
object, behavior, impact, request, location, and time expression is tied to an
exact source quote. Qwen only returns the quote text; deterministic code finds
its half-open source span (`text`, `start`, `end`). This avoids treating a
generative model as a character counter. If Qwen paraphrases an optional quote,
that field is rejected without discarding the record. A paraphrased trigger
falls back to another exact quote from the same event. An event with no exact
quote is dropped, and the record is quarantined only if all proposed events are
ungrounded. A normalized event type and generic search terms may be generated,
but factual evidence may not be. Missing evidence is represented by an empty
array; there is no `未知问题` placeholder.

The output includes five derived retrieval views:

- `evidence_core`: only selected source evidence, with no generated terms;
- `semantic_terms`: only normalized event types and generic expansion terms;
- `semantic_core`: event, evidence facts, impact, and normalized terms;
- `semantic_with_request`: core plus the caller's request;
- `semantic_with_location`: core plus request, location, and time.

These views separate evidence selection from normalization/expansion, keep
detailed locations out of the default knowledge query, and make generated-term,
location, and request ablations possible. Raw and raw-plus-semantic views are
constructed later by joining `source_id` back to the controlled source file;
the extraction output does not duplicate the raw text.

## Pilot design

`selector.py` makes a deterministic 2,000-record pilot from unique, non-empty
`case_content` values. Pilot v2 excludes source-boundary contamination: in this
export, 177 otherwise parseable records contain tab-separated fields or later
records inside `case_content`. Those records are quarantined from extraction
rather than misclassified as long narratives. The selector balances top-level
source category and five text length buckets, including the remaining clean
long tail. That makes it a stress/quality pilot, not a population-weighted
evaluation set. The category is used only for sample selection and is never
passed to Qwen3.

Long inputs are split at sentence boundaries at 8,000 characters. Segment
offsets are converted back to document offsets. A quote absent from the source
is rejected at field level. Repeated exact quotes are located nearest to the
event trigger and counted as ambiguous matches for quality review. The audit
reports proposed and rejected evidence quotes, trigger fallbacks, dropped
events, and `exact_evidence_quote_rate`. Fuzzy matching is never used to turn a
paraphrase into evidence.

Successful extraction results are cached by the SHA256 of the exact
`case_content` plus the model/Prompt contract. Duplicate text therefore needs
one model call while still producing one output record per source row.

## Server run

All commands are run from the repository root. Git is not used by any runtime
script.

Create the private configuration once and verify the paths:

```bash
cp deploy/.env.semantic.example deploy/.env.semantic
chmod 600 deploy/.env.semantic
```

If the existing `civic-rag-extract` environment from the earlier Qwen3 run is
still intact, refresh only this repository's editable package:

```bash
conda run --no-capture-output -n civic-rag-extract \
  python -m pip install --no-deps -e .
```

For a new or incomplete environment, run this in the platform's Conda/test
environment (it does not download a model):

```bash
bash deploy/create-semantic-extraction-env.sh
```

Prepare the fixed pilot in a CPU/test task before reserving the H100:

```bash
bash deploy/prepare-qwen3-pilot.sh
```

Then submit this single foreground command as one H100 task. The wrapper checks
the local model structure, starts loopback-only vLLM, verifies the served model
alias, extracts 20 new records, and always stops the vLLM process group:

```bash
bash deploy/run-qwen3-pilot.sh --limit 20 --overwrite
```

The model path and previously verified content fingerprint in the supplied
example are:

```text
/seu_share/home/huangkai/220243809/12345/excel_rag/models/Qwen3-30B-A3B
sha256:216fb80445d8028ae6a46fcdf762a5355e915fa74e0feebd1639a96fa41fb918
```

The launcher performs a fast config/index/shard validation. It records the
declared full content fingerprint but deliberately does not rehash 57 GiB on
every job; that fingerprint was already established on this server.

After the 20-record task, aggregate the result without printing source text:

```bash
set -a
. deploy/.env.semantic
set +a
conda run --no-capture-output -n "$CONDA_EXTRACT_ENV" \
  python -m semantic_extraction.audit \
  --output data/processed/qwen3-semantic-v3-clean/pilot-2000.jsonl \
  --errors data/processed/qwen3-semantic-v3-clean/pilot-2000.errors.jsonl
```

Proceed only if quarantine is zero and a manual evidence review is acceptable.
The next task processes the remaining 1,980 records; `--limit` means new
records, and `--resume` validates the prior manifest before appending:

```bash
bash deploy/run-qwen3-pilot.sh --limit 1980 --resume
```

Runtime logs live under `run-records/qwen3-semantic-v3-clean/`. Request and access
logging are disabled, and exception messages or source text are not written to
the quarantine file. Results, pilot data, cache, logs, and private environment
settings are ignored by Git.

## What this stage can establish

The extraction pilot measures schema validity, evidence grounding, empty-event
rate, multi-event rate, quarantine rate, and manual field precision/recall. It
does not by itself establish retrieval improvement. That claim requires the
same query set, corpus, embedder/BM25 settings, and relevance judgments across
raw, semantic, raw-plus-semantic, and dual-route retrieval, with Observed
Recall@10 as the primary metric.
