# Civic RAG rebuild

The repository contains three stages:

- [`data_analysis/`](data_analysis/): privacy-aware source profiling and field semantics;
- [`retrieval_baseline/`](retrieval_baseline/): time-isolated raw-text BM25, BGE-M3,
  and case-level hybrid retrieval for historical cases and observed knowledge-entry
  recommendation;
- [`semantic_extraction/`](semantic_extraction/): evidence-grounded Qwen3 extraction from
  `case_content` for controlled retrieval experiments.

GPU inference runs on the existing campus H100 server: LLM extraction, BGE-M3 encoding,
and BGE reranking. The local workstation handles development, input preparation,
evidence validation, human review, and offline reports. For the new extraction contract,
see the [v1 H100 integration note](deploy/CASE_EXTRACTION_V1_H100.md).

Start with [`retrieval_baseline/README.md`](retrieval_baseline/README.md) for the current
CPU-only dataset and development baseline. It does not require LLM annotations.
Prepare the separate Conda environment and ModelScope model using
[`deploy/RETRIEVAL_SETUP.md`](deploy/RETRIEVAL_SETUP.md).
Then continue with [`retrieval_baseline/DENSE.md`](retrieval_baseline/DENSE.md) for server-side
BGE-M3 encoding, resumable vector indexing and paired BM25/dense evaluation.
After both runs, use the [case-level hybrid procedure](retrieval_baseline/README.md#案例级混合检索)
to fuse their saved candidates and compare all three methods without new model calls.
For arbitrary new complaints, addresses, or problem descriptions, use
[`retrieval_baseline/CASE_SEARCH.md`](retrieval_baseline/CASE_SEARCH.md). This entry
returns historical complaint cases directly, without knowledge-reference voting.
It reuses the existing raw-text indexes and adds a separate address-name index.
Continue with [`retrieval_baseline/RERANKING.md`](retrieval_baseline/RERANKING.md) for
server-side BGE reranking, blind human case judgments, held-out comparisons and calibrated
relevance filtering. Reranker download and environment checks are separate steps.
For the formal phase-one workflow, follow the [campus H100 runbook](deploy/PHASE1_RUNBOOK.md):
verify existing assets, freeze reviewed query sources and labels, calibrate before inspecting
held-out metrics, and copy complete run/label/calibration bundles for offline reporting.
Use [`semantic_extraction/README.md`](semantic_extraction/README.md) for the separate
Conda and H100 extraction pilot procedure. Source data, derived samples, model files,
runtime logs, caches, and private deployment settings are excluded from Git.

After the 20-record Qwen3 v4 pilot, use
[`semantic_extraction/EVALUATION.md`](semantic_extraction/EVALUATION.md) to build
the private human gold set and score extraction errors before changing the
schema or running the remaining pilot records.

Use the [offline annotation page](semantic_extraction/ANNOTATION.md) to read,
annotate, and export the 20-record worksheet in a browser.
