# Semantic extraction evaluation

This stage freezes `semantic-extraction-v4` as the first LLM extraction
baseline. Do not tune the prompt against impressions from the same 20 records.
First turn those records into a reviewed gold set, score v4, and use the observed
error types to design v5.

The evaluator deliberately does not use lexical similarity, embeddings, or an
LLM judge to decide whether two issue labels mean the same thing. Civic issue
names are open-ended, and an automatic matcher would make the reported score
depend on another unmeasured model. A reviewer therefore performs a small,
one-to-one alignment between each candidate event and each gold issue. The code
validates the alignment and computes all metrics deterministically.

All worksheet, gold, and adjudication files below live under `data/evaluation/`.
They can contain sensitive source excerpts and are excluded from Git.

## 1. Prepare the 20-record gold worksheet

Run this on a CPU/login task after the v4 20-record result exists. Use the
literal Conda environment name so the command does not depend on a shell
variable being loaded:

```bash
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.evaluation prepare-gold --input data/derived/qwen3-pilot-v2-2000.jsonl --predictions data/processed/qwen3-semantic-v4/pilot-2000.jsonl --output data/evaluation/semantic-gold-v1.worksheet.jsonl --limit 20
```

Each worksheet row includes `case_content`, a compact v4 prediction for
comparison, and an editable `gold` object. The prediction is context only: do
not copy its boundaries into gold without checking the source.

For each row:

- set `review_status` to `complete` only after review;
- set `annotator_id` to a stable reviewer name;
- choose one `case_status`;
- list every active issue that could require a distinct knowledge answer;
- put resolved, withdrawn, superseded, or narrative-only issues in
  `background_issues`;
- give each issue a record-local ASCII `issue_id` such as
  `property-fee-dispute`;
- mark `is_current_request=true` only when the issue directly expresses the
  caller's current or final request;
- copy one to eight indispensable, continuous source quotes into
  `evidence_quotes`;
- describe the knowledge needed to answer the issue in
  `required_knowledge_need`.

`case_status` has the following meaning:

| Value | Meaning |
| --- | --- |
| `active` | One or more issues still require handling. |
| `resolved` | The text explicitly says the issue is complete and has no new request. |
| `withdrawal` | The current request is to withdraw or cancel a case. |
| `consultation` | The current need is policy, eligibility, contact, or process information. |
| `follow_up` | The current need is progress, non-response, or follow-up on earlier handling. |
| `unclear` | The source does not establish a safer status. Explain why in notes. |

Issue boundaries follow the retrieval task, not sentence or timeline
boundaries. Merge descriptions that require the same answer. Split issues only
when a useful answer would need different policy or handling knowledge. Do not
silently cap gold at three issues; the gold set must expose any recall ceiling
created by a model schema.

## 2. Validate and freeze gold

The finalizer requires all rows to be complete, verifies source IDs and hashes,
checks every evidence quote against the exact `case_content`, and removes the
full source text and v4 prediction from the frozen gold file:

```bash
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.evaluation finalize-gold --input data/derived/qwen3-pilot-v2-2000.jsonl --worksheet data/evaluation/semantic-gold-v1.worksheet.jsonl --output data/evaluation/semantic-gold-v1.jsonl
```

Do not change the frozen gold to make a later model look better. Corrections
must be documented and applied before comparing all candidates again.

## 3. Adjudicate v4 against gold

Create a candidate-specific worksheet:

```bash
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.evaluation prepare-adjudication --gold data/evaluation/semantic-gold-v1.jsonl --predictions data/processed/qwen3-semantic-v4/pilot-2000.jsonl --output data/evaluation/semantic-v4.adjudication.jsonl
```

For every predicted `event_index`, add exactly one entry to either `matches` or
`spurious_event_indices`. A match identifies one `gold_scope` (`active` or
`background`) and one `gold_issue_id`. Each prediction and each gold issue may
be matched at most once. If one model event incorrectly combines two distinct
gold knowledge needs, match the best-supported one; the other remains a false
negative. This makes over-merging visible in recall.

Set `review_status=complete` after all predicted events in that record have been
accounted for. The worksheet stores a hash of each prediction; scoring refuses
stale adjudication after a candidate output changes.

## 4. Score

```bash
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.evaluation score --gold data/evaluation/semantic-gold-v1.jsonl --predictions data/processed/qwen3-semantic-v4/pilot-2000.jsonl --adjudication data/evaluation/semantic-v4.adjudication.jsonl
```

The primary extraction metrics are micro-averaged:

- `issue_precision`: active gold matches / all predicted issues;
- `issue_recall`: active gold matches / all active gold issues;
- `issue_f1`: harmonic mean of issue precision and recall;
- `current_request_recall`: matched active issues marked as current requests /
  all such gold issues;
- `background_leakage_rate`: predictions aligned only to background issues /
  all predicted issues;
- `spurious_issue_rate`: predictions aligned to no gold issue / all predicted
  issues;
- `active_polarity_accuracy`: correct polarity among matched active issues;
- `active_evidence_quote_recall`: required active gold quotes covered by matched
  event evidence / all required active gold quotes;
- `matched_issue_evidence_quote_recall`: the same evidence measure conditional
  on issue detection;
- `exact_record_rate`: records with every active issue matched and no background
  or spurious prediction.

Future schemas may emit `case_status`; the scorer reports status coverage and
accuracy when present. For v4, coverage is expected to be zero. These extraction
metrics diagnose representation quality, but they do not prove RAG improvement.
The later retrieval experiment must compare raw `case_content`,
`semantic_core`, `semantic_with_request`, and combined routes under the same
corpus, query set, relevance judgments, and retrieval settings. Observed
Recall@K remains the primary end-to-end metric.
